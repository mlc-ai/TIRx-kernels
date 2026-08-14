# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

import ctypes
from dataclasses import asdict, dataclass
from functools import cache
from importlib.util import find_spec
from typing import Any
from unittest import SkipTest

import torch

import tvm
from tvm.ir.type import PointerType, PrimType
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import tirx as T
from tvm.script.ir_builder.base import IRBuilderFrame
from tvm.tirx import IterVar, Layout, is_buffer_var
from tvm.tirx.script.builder.ir import name_meta_class_value

_DEEP_GEMM_MODULE_NAME = "deep_gemm"
_SM100_SMEM_CAPACITY = 232448
_TEST_DIFF_THRESHOLD = 5e-6
_CONTEXT_PATTERNS = ("random_2d", "sglang_fixed", "sglang_ragged")
_COMPILE_CACHE_NAMESPACE = "deepgemm.paged_mqa_logits_fp8.compile"


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


@dataclass(frozen=True)
class PagedMQALogitsFP8Config:
    batch_size: int = 1
    next_n: int = 1
    max_num_pages: int = 4
    num_pages: int = 128
    num_heads: int = 64
    head_dim: int = 128
    page_size: int = 64
    logits_dtype: str = "float32"
    seed: int = 0
    num_sms: int = 148
    context_lens_2d: bool = True
    varlen: bool = False
    indices_pair_stride: int = 1
    context_pattern: str = "random_2d"

    @property
    def max_context_len(self) -> int:
        return self.max_num_pages * self.page_size

    @property
    def split_kv(self) -> int:
        return 256

    @property
    def block_kv(self) -> int:
        return self.page_size

    @property
    def logits_stride(self) -> int:
        return _align_up(self.max_context_len, self.split_kv)

    def validate(self) -> None:
        if self.batch_size <= 0 or self.next_n <= 0:
            raise ValueError("batch_size and next_n must be positive")
        if self.num_heads not in (32, 64):
            raise ValueError("num_heads must be 32 or 64")
        if self.head_dim not in (32, 64, 128):
            raise ValueError("head_dim must be 32, 64, or 128")
        if self.page_size not in (32, 64):
            raise ValueError("page_size must match DeepGEMM block_kv 32 or 64")
        if self.split_kv % self.page_size != 0:
            raise ValueError("split_kv must be divisible by page_size")
        if self.max_num_pages <= 0 or self.num_pages < self.max_num_pages:
            raise ValueError("num_pages must cover max_num_pages")
        if self.logits_dtype not in ("float32", "bfloat16"):
            raise ValueError("logits_dtype must be 'float32' or 'bfloat16'")
        if not self.context_lens_2d:
            raise ValueError("DeepGEMM paged FP8 API currently requires 2D context_lens")
        if self.varlen and self.next_n != 1:
            raise ValueError("DeepGEMM varlen paged mode requires next_n == 1")
        if self.indices_pair_stride <= 0:
            raise ValueError("indices_pair_stride must be positive")
        if self.context_pattern not in _CONTEXT_PATTERNS:
            raise ValueError(
                f"context_pattern must be one of {_CONTEXT_PATTERNS}, got {self.context_pattern!r}"
            )


def _make_config(**kwargs: Any) -> PagedMQALogitsFP8Config:
    kwargs = {key: value for key, value in kwargs.items() if key != "label"}
    config = PagedMQALogitsFP8Config(**kwargs)
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
    mode = "varlen" if config.get("varlen", False) else "fixed"
    context_suffix = {"random_2d": "", "sglang_fixed": "_sgfixed", "sglang_ragged": "_sgragged"}[
        config.get("context_pattern", "random_2d")
    ]
    return (
        f"b{config['batch_size']}_n{config['next_n']}_mp{config['max_num_pages']}_"
        f"ps{config['page_size']}_h{config['num_heads']}_d{config['head_dim']}_{dtype}_{mode}"
        f"{context_suffix}"
    )


def _make_case(
    *,
    batch_size: int,
    next_n: int,
    max_num_pages: int,
    num_pages: int,
    page_size: int,
    logits_dtype: str,
    seed: int,
    num_heads: int = 64,
    head_dim: int = 128,
    varlen: bool = False,
    context_pattern: str = "random_2d",
) -> dict[str, Any]:
    config = {
        "batch_size": batch_size,
        "next_n": next_n,
        "max_num_pages": max_num_pages,
        "num_pages": num_pages,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "page_size": page_size,
        "logits_dtype": logits_dtype,
        "seed": seed,
        "varlen": varlen,
        "context_pattern": context_pattern,
    }
    config["label"] = _config_label(config)
    return config


KERNEL_META = {
    "name": "deepgemm_sm100_fp8_paged_mqa_logits",
    "category": "deepgemm",
    "compute_capability": 10,
}

DSA_INDEXER_LIKE_COVERAGE = [
    _make_case(
        batch_size=batch_size,
        next_n=1,
        max_num_pages=max_num_pages,
        num_pages=max(11923, max_num_pages),
        page_size=64,
        logits_dtype=logits_dtype,
        seed=2000 + seed,
    )
    for seed, (batch_size, max_num_pages, logits_dtype) in enumerate(
        (batch_size, max_num_pages, logits_dtype)
        for logits_dtype in ("float32", "bfloat16")
        for batch_size in (1, 2, 4, 8, 16)
        for max_num_pages in (1, 8, 32, 128)
    )
]

# Complete upstream defaults from
# benchmark/kernels/deepseek/benchmark_cute_dsl_fp8_paged_mqa_logits.py:
#   batch_size = (1, 2, 4, 6, 8, 10, 12, 14, 16)
#   next_n = (1, 2, 4, 6)
#   context_len = (4096, 10240, 32768, 81920, 131072)
#   num_heads = 32, head_dim = 128, block_kv = 64, output_dtype = float32
#   varlen = False, use_cuda_graph = True
# The Cartesian product contains 9 * 4 * 5 = 180 configs. Extending the same
# grid to num_heads = (32, 64) contains 360 configs.
#
# SGLANG_BENCH_CONFIGS is currently a curated 80-config kernel-only subset:
#   decode: H=(32,64) x B=(1,2,4,8,16) x every context_len = 50
#   target verify: H=(32,64) x next_n=(2,4,6) x the five paired (B, pages)
#                  points below = 30
# max_num_pages is context_len / page_size, with page_size fixed at 64.
_SGLANG_CONTEXT_PAGES = (64, 160, 512, 1280, 2048)
_SGLANG_TARGET_VERIFY_POINTS = ((1, 64), (2, 2048), (6, 160), (10, 512), (16, 1280))

_SGLANG_DECODE_BENCH_CONFIGS = [
    _make_case(
        batch_size=batch_size,
        next_n=1,
        max_num_pages=max_num_pages,
        num_pages=max(11923, batch_size * max_num_pages),
        num_heads=num_heads,
        page_size=64,
        logits_dtype="float32",
        seed=3000 + seed,
        context_pattern="sglang_fixed",
    )
    for seed, (num_heads, batch_size, max_num_pages) in enumerate(
        (num_heads, batch_size, max_num_pages)
        for num_heads in (32, 64)
        for batch_size in (1, 2, 4, 8, 16)
        for max_num_pages in _SGLANG_CONTEXT_PAGES
    )
]

_SGLANG_TARGET_VERIFY_BENCH_CONFIGS = [
    _make_case(
        batch_size=batch_size,
        next_n=next_n,
        max_num_pages=max_num_pages,
        num_pages=max(11923, batch_size * max_num_pages),
        num_heads=num_heads,
        page_size=64,
        logits_dtype="float32",
        seed=4000 + seed,
        context_pattern="sglang_fixed",
    )
    for seed, (num_heads, next_n, batch_size, max_num_pages) in enumerate(
        (num_heads, next_n, batch_size, max_num_pages)
        for num_heads in (32, 64)
        for next_n in (2, 4, 6)
        for batch_size, max_num_pages in _SGLANG_TARGET_VERIFY_POINTS
    )
]

SGLANG_BENCH_CONFIGS = _SGLANG_DECODE_BENCH_CONFIGS + _SGLANG_TARGET_VERIFY_BENCH_CONFIGS

_SMOKE_CONFIGS = [
    _make_case(
        batch_size=1,
        next_n=1,
        max_num_pages=4,
        num_pages=128,
        page_size=64,
        logits_dtype="float32",
        seed=0,
    ),
    _make_case(
        batch_size=2,
        next_n=1,
        max_num_pages=4,
        num_pages=128,
        page_size=64,
        logits_dtype="bfloat16",
        seed=1,
    ),
    _make_case(
        batch_size=2,
        next_n=3,
        max_num_pages=4,
        num_pages=128,
        page_size=64,
        logits_dtype="float32",
        seed=2,
    ),
    _make_case(
        batch_size=1,
        next_n=1,
        max_num_pages=2,
        num_pages=128,
        num_heads=32,
        page_size=64,
        logits_dtype="float32",
        seed=10,
        context_pattern="sglang_fixed",
    ),
    _make_case(
        batch_size=2,
        next_n=2,
        max_num_pages=16,
        num_pages=128,
        num_heads=32,
        page_size=64,
        logits_dtype="float32",
        seed=11,
        context_pattern="sglang_fixed",
    ),
    _make_case(
        batch_size=4,
        next_n=3,
        max_num_pages=64,
        num_pages=256,
        num_heads=64,
        page_size=64,
        logits_dtype="float32",
        seed=12,
        context_pattern="sglang_ragged",
    ),
    _make_case(
        batch_size=2,
        next_n=4,
        max_num_pages=64,
        num_pages=128,
        num_heads=32,
        page_size=64,
        logits_dtype="float32",
        seed=13,
        context_pattern="sglang_fixed",
    ),
    _make_case(
        batch_size=1,
        next_n=5,
        max_num_pages=16,
        num_pages=128,
        num_heads=64,
        page_size=64,
        logits_dtype="float32",
        seed=14,
        context_pattern="sglang_fixed",
    ),
    _make_case(
        batch_size=2,
        next_n=6,
        max_num_pages=64,
        num_pages=128,
        num_heads=32,
        page_size=64,
        logits_dtype="float32",
        seed=15,
        context_pattern="sglang_ragged",
    ),
]

CONFIGS = _SMOKE_CONFIGS + DSA_INDEXER_LIKE_COVERAGE + SGLANG_BENCH_CONFIGS


def load_deep_gemm_paged_mqa() -> tuple[Any, str]:
    try:
        import deep_gemm as module
    except Exception as exc:
        raise SkipTest(
            f"DeepGEMM paged MQA logits runtime unavailable: {_DEEP_GEMM_MODULE_NAME}: {exc}"
        ) from exc

    if not hasattr(module, "fp8_fp4_paged_mqa_logits"):
        raise SkipTest("DeepGEMM runtime unavailable: missing fp8_fp4_paged_mqa_logits")
    if not hasattr(module, "get_paged_mqa_logits_metadata"):
        raise SkipTest("DeepGEMM runtime unavailable: missing get_paged_mqa_logits_metadata")
    return module, "installed"


def _make_context_lens(config: PagedMQALogitsFP8Config) -> torch.Tensor:
    max_context_len = config.max_context_len
    if config.context_pattern == "sglang_fixed":
        lens = torch.full((config.batch_size, 1), max_context_len, dtype=torch.int32, device="cuda")
        if config.next_n > 1:
            lens = (
                lens
                - config.next_n
                + torch.arange(1, config.next_n + 1, dtype=torch.int32, device="cuda")[None, :]
            )
    elif config.context_pattern == "sglang_ragged":
        low = max(config.next_n, config.page_size, int(0.7 * max_context_len))
        last_token_lens = torch.randint(
            low=low,
            high=max_context_len + 1,
            size=(config.batch_size, 1),
            dtype=torch.int32,
            device="cuda",
        )
        if config.next_n == 1:
            lens = last_token_lens
        else:
            lens = (
                last_token_lens
                - config.next_n
                + torch.arange(1, config.next_n + 1, dtype=torch.int32, device="cuda")[None, :]
            )
    elif max_context_len == config.page_size:
        lens = torch.full(
            (config.batch_size, config.next_n), max_context_len, dtype=torch.int32, device="cuda"
        )
    else:
        last_token_lens = torch.randint(
            low=max(1, config.page_size // 2),
            high=max_context_len + 1,
            size=(config.batch_size, 1),
            dtype=torch.int32,
            device="cuda",
        )
        if config.next_n == 1:
            lens = last_token_lens
        else:
            lens = (
                (last_token_lens + 1) * torch.rand(config.batch_size, config.next_n, device="cuda")
            ).to(torch.int32)
            lens[:, -1] = last_token_lens[:, 0]
    lens = torch.maximum(lens, torch.ones_like(lens))
    return lens.contiguous()


def _make_block_table(config: PagedMQALogitsFP8Config) -> torch.Tensor:
    page_ids = torch.arange(config.num_pages, dtype=torch.int32, device="cuda")
    rows = []
    for batch_idx in range(config.batch_size):
        start = (batch_idx * config.max_num_pages) % config.num_pages
        rows.append(page_ids.roll(-start)[: config.max_num_pages])
    return torch.stack(rows, dim=0).contiguous()


def _make_indices(config: PagedMQALogitsFP8Config) -> torch.Tensor | None:
    if not config.varlen:
        return None
    indices = torch.arange(config.batch_size, dtype=torch.int32, device="cuda")
    if config.indices_pair_stride > 1:
        indices = indices // config.indices_pair_stride
    return indices.contiguous()


def _make_fused_kv_cache(
    config: PagedMQALogitsFP8Config, *, keep_dequant: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    kv_bf16 = torch.randn(
        config.num_pages, config.page_size, config.head_dim, device="cuda", dtype=torch.bfloat16
    ).clamp_(-2.0, 2.0)
    scales = kv_bf16.abs().float().amax(dim=2, keepdim=True).clamp(1e-4) / 448.0
    kv_fp8 = (kv_bf16 * (1.0 / scales)).to(torch.float8_e4m3fn).contiguous()
    kv_dequant = (kv_fp8.float() * scales).to(torch.bfloat16) if keep_dequant else None
    scales = scales.squeeze(-1).contiguous()

    fused = torch.empty(
        (config.num_pages, config.page_size, 1, config.head_dim + 4),
        dtype=torch.uint8,
        device="cuda",
    )
    fused_flat = fused.view(config.num_pages, config.page_size * (config.head_dim + 4))
    fused_flat[:, : config.page_size * config.head_dim].copy_(
        kv_fp8.view(torch.uint8).reshape(config.num_pages, config.page_size * config.head_dim)
    )
    fused_flat[:, config.page_size * config.head_dim :].copy_(
        scales.view(torch.uint8).reshape(config.num_pages, config.page_size * 4)
    )
    return fused.contiguous(), kv_dequant


def _ref_paged_mqa_logits(
    q: torch.Tensor,
    kv_dequant: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    config: PagedMQALogitsFP8Config,
) -> torch.Tensor:
    q_f32 = q.float()
    kv_f32 = kv_dequant.float()
    weights_f32 = weights.view(config.batch_size, config.next_n, config.num_heads).float()
    output = torch.full(
        (config.batch_size * config.next_n, config.max_context_len),
        float("-inf"),
        device="cuda",
        dtype=torch.float32,
    )
    for batch_idx in range(config.batch_size):
        for next_idx in range(config.next_n):
            row = batch_idx * config.next_n + next_idx
            context_len = int(context_lens[batch_idx, next_idx].item())
            for page_col in range((context_len + config.page_size - 1) // config.page_size):
                page_id = int(block_table[batch_idx, page_col].item())
                token_start = page_col * config.page_size
                token_end = min(token_start + config.page_size, context_len)
                kv_tile = kv_f32[page_id, : token_end - token_start]
                score = torch.einsum("hd,td->ht", q_f32[batch_idx, next_idx], kv_tile)
                logits = (score.relu() * weights_f32[batch_idx, next_idx, :, None]).sum(dim=0)
                output[row, token_start:token_end] = logits
    return output


def _prepare_data(config: PagedMQALogitsFP8Config, *, compute_reference: bool) -> dict[str, Any]:
    deep_gemm, source = load_deep_gemm_paged_mqa()
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for SM100 FP8 paged MQA logits")
    if torch.cuda.get_device_capability()[0] < 10:
        raise SkipTest("SM100 FP8 paged MQA logits requires compute capability 10.x")

    torch.manual_seed(config.seed)
    runtime_config = PagedMQALogitsFP8Config(
        **{
            **asdict(config),
            "num_sms": int(getattr(deep_gemm, "get_num_sms", lambda: config.num_sms)()),
        }
    )
    q_bf16 = torch.randn(
        config.batch_size,
        config.next_n,
        config.num_heads,
        config.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ).clamp_(-2.0, 2.0)
    q_fp8 = q_bf16.to(torch.float8_e4m3fn).contiguous()
    fused_kv_cache, kv_dequant = _make_fused_kv_cache(config, keep_dequant=compute_reference)
    weights = torch.randn(
        config.batch_size * config.next_n, config.num_heads, device="cuda", dtype=torch.float32
    ).contiguous()
    context_lens = _make_context_lens(config)
    block_table = _make_block_table(config)
    indices = _make_indices(config)
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, config.page_size, runtime_config.num_sms, indices
    )
    tirx_schedule_meta = schedule_meta
    if not config.varlen and config.next_n >= 2:
        num_q_atoms = _align_up(config.next_n, 2) // 2
        atom_context_lens = context_lens[:, -1:].expand(config.batch_size, num_q_atoms).contiguous()
        tirx_schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
            atom_context_lens, config.page_size, runtime_config.num_sms
        )
        expected_end = config.batch_size * num_q_atoms
        if int(tirx_schedule_meta[-1, 0]) != expected_end:
            raise RuntimeError(
                f"TIRx schedule metadata ends at {int(tirx_schedule_meta[-1, 0])}, "
                f"expected {expected_end} q atoms"
            )
    data = {
        "config": runtime_config,
        "reference_source": source,
        "q": q_fp8,
        "fused_kv_cache": fused_kv_cache,
        "weights": weights,
        "context_lens": context_lens,
        "block_table": block_table,
        "indices": indices,
        "schedule_meta": schedule_meta,
        "tirx_schedule_meta": tirx_schedule_meta,
        "deep_gemm": deep_gemm,
    }
    if compute_reference:
        assert kv_dequant is not None
        data["reference"] = _ref_paged_mqa_logits(
            q_fp8.to(torch.bfloat16), kv_dequant, weights, context_lens, block_table, config
        )
    return data


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    return _prepare_data(_make_config(**kwargs), compute_reference=True)


def get_kernel(**kwargs: Any):
    from tvm.tirx.layout import S, TCol, TileLayout, TLane

    config = _make_config(**kwargs)
    num_heads = config.num_heads
    head_dim = config.head_dim
    page_size = config.page_size
    k_pad_odd_n = (not config.varlen) and (config.next_n % 2 == 1) and (config.next_n >= 3)
    next_n_atom = 2 if (config.varlen or config.next_n >= 2) else 1
    num_next_n_atoms = _align_up(config.next_n, next_n_atom) // next_n_atom
    num_q_stages = 3
    # UMMA (TMEM) pipeline depth per group: 2 stages let the next task's MMA
    # overlap the current task's epilogue TMEM read (beyond-CuTeDSL default 1).
    num_umma_stages = 1
    split_kv = config.split_kv
    umma_m = 128
    umma_k = 32
    umma_n = next_n_atom * num_heads
    num_math_warpgroups = split_kv // umma_m
    # One scheduler task covers split_kv tokens = num_tiles_per_split compute
    # tiles (one per math warpgroup); each compute tile spans num_pages_per_tile
    # physical pages (CuTeDSL NUM_BLOCKS_PER_MMA).
    num_tiles_per_split = split_kv // umma_m
    num_pages_per_tile = umma_m // page_size
    num_specialized_threads = 128
    num_specialized_registers = 24
    num_math_registers = 240
    num_math_threads = num_math_warpgroups * 128
    num_threads = num_specialized_threads + num_math_threads
    num_warps = num_threads // 32
    spec_warp_start = num_math_warpgroups * 4
    tma_warp_0 = spec_warp_start
    tma_warp_1 = spec_warp_start + 1
    umma_warp_0 = spec_warp_start + 2
    smem_alignment = head_dim * 8
    desc_sdo = 8 * head_dim // 16
    desc_swizzle = {32: 1, 64: 2, 128: 3}[head_dim]
    smem_q_size_per_stage = next_n_atom * num_heads * head_dim
    smem_kv_size_per_stage = umma_m * head_dim
    smem_kv_scale_size_per_stage = umma_m * 4
    smem_weight_size_per_stage = next_n_atom * num_heads * 4
    # Per-group KV pipeline depth (matches the CuTeDSL baseline: 3 stages/group).
    num_kv_stages = 3
    smem_q_offset = 0
    smem_kv_offset = smem_q_offset + smem_q_size_per_stage * num_q_stages
    smem_kv_scales_offset = (
        smem_kv_offset + smem_kv_size_per_stage * num_kv_stages * num_math_warpgroups
    )
    smem_weights_offset = (
        smem_kv_scales_offset + smem_kv_scale_size_per_stage * num_kv_stages * num_math_warpgroups
    )
    smem_barrier_offset = smem_weights_offset + smem_weight_size_per_stage * num_q_stages
    num_umma_barriers = num_math_warpgroups * num_umma_stages
    num_total_barriers = (
        num_q_stages * 2 + num_kv_stages * 2 * num_math_warpgroups + (num_umma_barriers * 2)
    )
    full_q_barrier_base = 0
    empty_q_barrier_base = full_q_barrier_base + num_q_stages
    full_kv_barrier_base = empty_q_barrier_base + num_q_stages
    empty_kv_barrier_base = full_kv_barrier_base + num_kv_stages * num_math_warpgroups
    full_umma_barrier_base = empty_kv_barrier_base + num_kv_stages * num_math_warpgroups
    empty_umma_barrier_base = full_umma_barrier_base + num_umma_barriers
    smem_tmem_ptr_offset = smem_barrier_offset + num_total_barriers * 8
    smem_total_bytes = smem_tmem_ptr_offset + 4
    if smem_total_bytes > _SM100_SMEM_CAPACITY:
        raise ValueError(f"dynamic shared memory {smem_total_bytes} exceeds SM100 capacity")
    num_tmem_cols = next_n_atom * num_heads * num_math_warpgroups * num_umma_stages
    if num_tmem_cols > 512:
        raise ValueError("tensor memory columns exceed SM100 single-CTA limit")
    tmem_layout = TileLayout(S[(128, num_tmem_cols) : (1 @ TLane, 1 @ TCol)])
    logits_tir_dtype = "float32" if config.logits_dtype == "float32" else "bfloat16"
    cache_hint_sm90_evict_normal = "evict_normal"
    cache_hint_sm100_evict_normal = "evict_normal"
    cache_policy_evict_normal = T.uint64(1152921504606846976)
    # One ptx spelling per rank: unicast (no .multicast::cluster), no
    # .cta_group modifier (the legacy raw form passed -1 to suppress it), with
    # the evict-normal L2 cache policy as a real operand.
    tma_g2s_2d = (
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
    )
    tma_g2s_3d = (
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
    )
    q_tma_block_inner = head_dim
    q_tma_swizzle_mode = head_dim
    q_tma_dtype_size = 1
    q_tma_block_inner_atom = (
        q_tma_block_inner if q_tma_swizzle_mode == 0 else q_tma_swizzle_mode // q_tma_dtype_size
    )
    q_tma_num_inner_atoms = q_tma_block_inner // q_tma_block_inner_atom
    weights_tma_block_inner = next_n_atom * num_heads
    weights_tma_swizzle_mode = 0
    weights_tma_dtype_size = 4
    weights_tma_block_inner_atom = (
        weights_tma_block_inner
        if weights_tma_swizzle_mode == 0
        else weights_tma_swizzle_mode // weights_tma_dtype_size
    )
    weights_tma_num_inner_atoms = weights_tma_block_inner // weights_tma_block_inner_atom
    kv_tma_block_inner = head_dim
    kv_tma_swizzle_mode = 0
    kv_tma_dtype_size = 1
    kv_tma_block_inner_atom = (
        kv_tma_block_inner if kv_tma_swizzle_mode == 0 else kv_tma_swizzle_mode // kv_tma_dtype_size
    )
    kv_tma_num_inner_atoms = kv_tma_block_inner // kv_tma_block_inner_atom
    kv_scales_tma_block_inner = page_size
    kv_scales_tma_swizzle_mode = 0
    kv_scales_tma_dtype_size = 4
    kv_scales_tma_block_inner_atom = (
        kv_scales_tma_block_inner
        if kv_scales_tma_swizzle_mode == 0
        else kv_scales_tma_swizzle_mode // kv_scales_tma_dtype_size
    )
    kv_scales_tma_num_inner_atoms = kv_scales_tma_block_inner // kv_scales_tma_block_inner_atom

    def atom_to_token_idx_expr(q_atom_idx):
        if config.varlen:
            return q_atom_idx
        if k_pad_odd_n:
            return q_atom_idx // T.uint32(num_next_n_atoms) * T.uint32(
                config.next_n
            ) + q_atom_idx % T.uint32(num_next_n_atoms) * T.uint32(next_n_atom)
        return q_atom_idx * T.uint32(next_n_atom)

    def atom_to_block_table_row_expr(q_atom_idx):
        if config.varlen:
            return q_atom_idx
        return q_atom_idx // T.uint32(num_next_n_atoms)

    def should_refresh_num_kv_expr(q_atom_idx):
        if config.varlen:
            return T.bool(True)
        return q_atom_idx % T.uint32(num_next_n_atoms) == T.uint32(0)

    def exist_q_atom_idx_expr(q_atom_idx, end_q_atom_idx, end_kv_idx):
        q_atom_idx = q_atom_idx.scalar if isinstance(q_atom_idx, T.scalar_wrapper) else q_atom_idx
        end_q_atom_idx = (
            end_q_atom_idx.scalar
            if isinstance(end_q_atom_idx, T.scalar_wrapper)
            else end_q_atom_idx
        )
        end_kv_idx = end_kv_idx.scalar if isinstance(end_kv_idx, T.scalar_wrapper) else end_kv_idx
        return T.Or(
            q_atom_idx < end_q_atom_idx,
            T.And(q_atom_idx == end_q_atom_idx, T.uint32(0) < end_kv_idx),
        )

    def lane_id_u32():
        return T.cast(T.cuda.mov_sreg(32, "laneid"), "uint32")

    def relu2_fma_f32x2(a, w, c):
        a_lo = T.alloc_local((1,), "float32")
        a_hi = T.alloc_local((1,), "float32")
        abs_lo = T.alloc_local((1,), "float32")
        abs_hi = T.alloc_local((1,), "float32")
        abs_pair = T.alloc_local((1,), "uint64")
        relu_pair = T.alloc_local((1,), "uint64")
        out = T.alloc_local((1,), "uint64")
        T.evaluate(T.ptx.mov.b64(a_lo[0], a_hi[0], a))
        T.evaluate(T.ptx.abs.f32(abs_lo[0], a_lo[0]))
        T.evaluate(T.ptx.abs.f32(abs_hi[0], a_hi[0]))
        T.evaluate(T.ptx.mov.b64(abs_pair[0], abs_lo[0], abs_hi[0]))
        T.evaluate(T.ptx.add.rn.f32x2(relu_pair[0], a, abs_pair[0]))
        T.evaluate(T.ptx.fma.rn.f32x2(out[0], relu_pair[0], w, c))
        return out[0]

    def fadd2_rn_noftz(a, b):
        out = T.alloc_local((1,), "uint64")
        T.evaluate(T.ptx.add.rn.f32x2(out[0], a, b))
        return out[0]

    def fadd_rn_noftz(a, b):
        out = T.alloc_local((1,), "float32")
        T.evaluate(T.ptx.add.rn.f32(out[0], a, b))
        return out[0]

    def fmul_rn_noftz(a, b):
        out = T.alloc_local((1,), "float32")
        T.evaluate(T.ptx.mul.rn.f32(out[0], a, b))
        return out[0]

    def cuda_grid_dependency_synchronize():
        T.evaluate(T.ptx.griddepcontrol.wait())

    # Block-table L2 warm-up coverage (plain Python ints, evaluated at trace
    # time): prefetch the whole table, capped at 512 lines (64 KB).
    num_block_table_bytes = config.batch_size * config.max_num_pages * 4
    num_prefetch_lines = (num_block_table_bytes + 127) // 128
    if num_prefetch_lines > 512:
        num_prefetch_lines = 512

    def mbarrier_init_cta(barrier_ptr, arrive_count):
        T.evaluate(T.ptx.mbarrier.init.shared.b64(barrier_ptr, T.uint32(arrive_count)))

    def mbarrier_wait_cta(barrier_ptr, phase):
        T.evaluate(T.cuda.mbarrier_wait(barrier_ptr, phase))

    def mbarrier_arrive_cta(barrier_ptr):
        T.evaluate(T.ptx.mbarrier.arrive.shared.b64(barrier_ptr, T.uint32(1)))

    def mbarrier_arrive_expect_tx_cta(barrier_ptr, transaction_bytes):
        T.evaluate(
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier_ptr, T.uint32(transaction_bytes))
        )

    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("sm100_fp8_paged_mqa_logits")
            batch_size = T.arg("batch_size", T.uint32())
            logits_stride = T.arg("logits_stride", T.uint32())
            block_table_stride = T.arg("block_table_stride", T.uint32())
            context_lens = T.arg(
                "context_lens", T.Buffer((config.batch_size, config.next_n), "int32")
            )
            logits = T.arg(
                "logits",
                T.Buffer(
                    (config.batch_size * config.next_n, config.logits_stride), logits_tir_dtype
                ),
            )
            block_table = T.arg(
                "block_table", T.Buffer((config.batch_size, config.max_num_pages), "int32")
            )
            indices = T.arg("indices", T.Buffer((config.batch_size,), "int32"))
            schedule_meta = T.arg("schedule_meta", T.Buffer((config.num_sms + 1, 2), "int32"))
            tensor_map_q = T.arg("tensor_map_q", T.TensorMap())
            tensor_map_kv = T.arg("tensor_map_kv", T.TensorMap())
            tensor_map_kv_scales = T.arg("tensor_map_kv_scales", T.TensorMap())
            tensor_map_weights = T.arg("tensor_map_weights", T.TensorMap())
            _builder_emit(T.device_entry())
            # TIRX_TRANSCRIBE_START sm100_fp8_paged_mqa_logits
            _builder_emit(T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1}))
            logits_flat = _builder_assign(
                "logits_flat",
                T.decl_buffer(
                    (config.batch_size * config.next_n * config.logits_stride,),
                    logits_tir_dtype,
                    data=logits.data,
                    scope="global",
                ),
                locals().get("logits_flat", _BUILDER_MISSING),
            )
            context_lens_flat = _builder_assign(
                "context_lens_flat",
                T.decl_buffer(
                    (config.batch_size * config.next_n,),
                    "int32",
                    data=context_lens.data,
                    scope="global",
                ),
                locals().get("context_lens_flat", _BUILDER_MISSING),
            )
            block_table_flat = _builder_assign(
                "block_table_flat",
                T.decl_buffer(
                    (config.batch_size * config.max_num_pages,),
                    "int32",
                    data=block_table.data,
                    scope="global",
                ),
                locals().get("block_table_flat", _BUILDER_MISSING),
            )
            schedule_meta_flat = _builder_assign(
                "schedule_meta_flat",
                T.decl_buffer(
                    ((config.num_sms + 1) * 2,), "int32", data=schedule_meta.data, scope="global"
                ),
                locals().get("schedule_meta_flat", _BUILDER_MISSING),
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
            lane_idx_u32 = _builder_bind("lane_idx_u32", lane_id_u32(), None)

            with T.If(warp_idx == spec_warp_start):
                with T.Then():
                    _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(tensor_map_q))))
                    _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(tensor_map_kv))))
                    _builder_emit(
                        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(tensor_map_kv_scales)))
                    )
                    _builder_emit(
                        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(tensor_map_weights)))
                    )

            _builder_emit(
                T.static_assert(
                    smem_q_size_per_stage % smem_alignment == 0, "Unaligned TMA swizzling"
                )
            )
            _builder_emit(
                T.static_assert(
                    smem_kv_size_per_stage % smem_alignment == 0, "Unaligned TMA swizzling"
                )
            )

            smem = _builder_assign(
                "smem",
                T.alloc_buffer(
                    [smem_total_bytes], "uint8", scope="shared.dyn", align=smem_alignment
                ),
                locals().get("smem", _BUILDER_MISSING),
            )
            _builder_emit(T.attr({"tirx.dyn_smem_bytes": smem_total_bytes}))
            smem_q_data = _builder_bind(
                "smem_q_data",
                T.reinterpret(PointerType(PrimType("float8_e4m3fn")), smem.ptr_to([smem_q_offset])),
                None,
            )
            smem_kv_data = _builder_bind(
                "smem_kv_data",
                T.reinterpret(
                    PointerType(PrimType("float8_e4m3fn")), smem.ptr_to([smem_kv_offset])
                ),
                None,
            )
            smem_kv_scales_data = _builder_bind(
                "smem_kv_scales_data",
                T.reinterpret(
                    PointerType(PrimType("float32")), smem.ptr_to([smem_kv_scales_offset])
                ),
                None,
            )
            smem_weights_data = _builder_bind(
                "smem_weights_data",
                T.reinterpret(PointerType(PrimType("float32")), smem.ptr_to([smem_weights_offset])),
                None,
            )
            smem_barrier_data = _builder_bind(
                "smem_barrier_data",
                T.reinterpret(PointerType(PrimType("uint64")), smem.ptr_to([smem_barrier_offset])),
                None,
            )
            smem_tmem_ptr_data = _builder_bind(
                "smem_tmem_ptr_data",
                T.reinterpret(PointerType(PrimType("uint32")), smem.ptr_to([smem_tmem_ptr_offset])),
                None,
            )
            smem_q = _builder_assign(
                "smem_q",
                T.decl_buffer(
                    (num_q_stages, next_n_atom * num_heads, head_dim),
                    "float8_e4m3fn",
                    data=smem_q_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=smem_alignment,
                ),
                locals().get("smem_q", _BUILDER_MISSING),
            )
            smem_kv = _builder_assign(
                "smem_kv",
                T.decl_buffer(
                    (num_math_warpgroups, num_kv_stages, umma_m, head_dim),
                    "float8_e4m3fn",
                    data=smem_kv_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=smem_alignment,
                ),
                locals().get("smem_kv", _BUILDER_MISSING),
            )
            smem_kv_scales = _builder_assign(
                "smem_kv_scales",
                T.decl_buffer(
                    (num_math_warpgroups, num_kv_stages, umma_m),
                    "float32",
                    data=smem_kv_scales_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=16,
                ),
                locals().get("smem_kv_scales", _BUILDER_MISSING),
            )
            smem_weights = _builder_assign(
                "smem_weights",
                T.decl_buffer(
                    (num_q_stages, next_n_atom, num_heads),
                    "float32",
                    data=smem_weights_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=16,
                ),
                locals().get("smem_weights", _BUILDER_MISSING),
            )
            smem_barriers = _builder_assign(
                "smem_barriers",
                T.decl_buffer(
                    (num_total_barriers,),
                    "uint64",
                    data=smem_barrier_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=8,
                ),
                locals().get("smem_barriers", _BUILDER_MISSING),
            )
            tmem_ptr_in_smem = _builder_assign(
                "tmem_ptr_in_smem",
                T.decl_buffer(
                    (1,),
                    "uint32",
                    data=smem_tmem_ptr_data,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=4,
                ),
                locals().get("tmem_ptr_in_smem", _BUILDER_MISSING),
            )
            tmem = _builder_assign(
                "tmem",
                T.decl_buffer(
                    (128, num_tmem_cols),
                    "float32",
                    scope="tmem",
                    allocated_addr=tmem_ptr_in_smem[0],
                    layout=tmem_layout,
                ),
                locals().get("tmem", _BUILDER_MISSING),
            )
            fetch_result = _builder_assign(
                "fetch_result",
                T.alloc_local((4,), "uint32"),
                locals().get("fetch_result", _BUILDER_MISSING),
            )
            scheduler_result = _builder_assign(
                "scheduler_result",
                T.alloc_local((7,), "uint32"),
                locals().get("scheduler_result", _BUILDER_MISSING),
            )
            num_kv_result = _builder_assign(
                "num_kv_result",
                T.alloc_local((1,), "uint32"),
                locals().get("num_kv_result", _BUILDER_MISSING),
            )
            atom_advance_result = _builder_assign(
                "atom_advance_result",
                T.alloc_local((1,), "uint32"),
                locals().get("atom_advance_result", _BUILDER_MISSING),
            )

            def mbarrier_wait_phase(barrier_ptr, phase):
                _builder_emit(mbarrier_wait_cta(barrier_ptr, phase))

            def mbarrier_arrive(barrier_ptr):
                _builder_emit(mbarrier_arrive_cta(barrier_ptr))

            def mbarrier_arrive_and_expect_tx(barrier_ptr, num_bytes):
                _builder_emit(mbarrier_arrive_expect_tx_cta(barrier_ptr, num_bytes))

            def get_q_pipeline(q_iter_idx):
                T.buffer_store(fetch_result, q_iter_idx % T.uint32(num_q_stages), [0])
                T.buffer_store(
                    fetch_result, (q_iter_idx // T.uint32(num_q_stages)) & T.uint32(1), [1]
                )

            def get_kv_pipeline(kv_iter_idx):
                T.buffer_store(fetch_result, kv_iter_idx % T.uint32(num_kv_stages), [2])
                T.buffer_store(
                    fetch_result, (kv_iter_idx // T.uint32(num_kv_stages)) & T.uint32(1), [3]
                )

            def load_num_kv(q_atom_idx_arg, runtime_batch_size_arg):
                if config.varlen:
                    context_idx = _builder_scalar("context_idx", q_atom_idx_arg, "uint32")
                    with T.If(q_atom_idx_arg + T.uint32(1) < runtime_batch_size_arg):
                        with T.Then():
                            index_0 = _builder_assign(
                                "index_0",
                                T.local_scalar("int32"),
                                locals().get("index_0", _BUILDER_MISSING),
                            )
                            index_1 = _builder_assign(
                                "index_1",
                                T.local_scalar("int32"),
                                locals().get("index_1", _BUILDER_MISSING),
                            )
                            _builder_emit(
                                T.ptx.ld.global_.s32(
                                    index_0, indices.ptr_to([T.cast(q_atom_idx_arg, "int32")])
                                )
                            )
                            _builder_emit(
                                T.ptx.ld.global_.s32(
                                    index_1,
                                    indices.ptr_to([T.cast(q_atom_idx_arg + T.uint32(1), "int32")]),
                                )
                            )
                            with T.If(index_0 == index_1):
                                with T.Then():
                                    context_idx = _builder_assign(
                                        "context_idx",
                                        q_atom_idx_arg + T.uint32(1),
                                        locals().get("context_idx", _BUILDER_MISSING),
                                    )
                    context_len = _builder_assign(
                        "context_len",
                        T.local_scalar("uint32"),
                        locals().get("context_len", _BUILDER_MISSING),
                    )
                    _builder_emit(
                        T.ptx.ld.global_.u32(
                            context_len, context_lens_flat.ptr_to([T.cast(context_idx, "int32")])
                        )
                    )
                else:
                    q_idx = _builder_scalar(
                        "q_idx", q_atom_idx_arg // T.uint32(num_next_n_atoms), "uint32"
                    )
                    lens_idx = _builder_scalar(
                        "lens_idx",
                        q_idx * T.uint32(config.next_n) + T.uint32(config.next_n - 1),
                        "uint32",
                    )
                    context_len = _builder_assign(
                        "context_len",
                        T.local_scalar("uint32"),
                        locals().get("context_len", _BUILDER_MISSING),
                    )
                    _builder_emit(
                        T.ptx.ld.global_.u32(
                            context_len, context_lens_flat.ptr_to([T.cast(lens_idx, "int32")])
                        )
                    )
                T.buffer_store(
                    num_kv_result, (context_len + T.uint32(umma_m - 1)) // T.uint32(umma_m), [0]
                )

            def load_atom_advance(q_atom_idx_arg, bound_arg):
                T.buffer_store(atom_advance_result, T.uint32(1), [0])
                if config.varlen:
                    with T.If(q_atom_idx_arg + T.uint32(1) < bound_arg):
                        with T.Then():
                            index_0 = _builder_assign(
                                "index_0",
                                T.local_scalar("int32"),
                                locals().get("index_0", _BUILDER_MISSING),
                            )
                            index_1 = _builder_assign(
                                "index_1",
                                T.local_scalar("int32"),
                                locals().get("index_1", _BUILDER_MISSING),
                            )
                            _builder_emit(
                                T.ptx.ld.global_.s32(
                                    index_0, indices.ptr_to([T.cast(q_atom_idx_arg, "int32")])
                                )
                            )
                            _builder_emit(
                                T.ptx.ld.global_.s32(
                                    index_1,
                                    indices.ptr_to([T.cast(q_atom_idx_arg + T.uint32(1), "int32")]),
                                )
                            )
                            with T.If(index_0 == index_1):
                                with T.Then():
                                    T.buffer_store(atom_advance_result, T.uint32(2), [0])

            def tma_load_2d_q(dst, barrier_ptr, tensor_map, coord0, coord1):
                _builder_emit(
                    T.static_assert(
                        cache_hint_sm90_evict_normal == cache_hint_sm100_evict_normal,
                        "Invalid cache hint",
                    )
                )
                _builder_emit(
                    T.static_assert(q_tma_num_inner_atoms == 1, "Unsupported split TMA atom")
                )
                _builder_emit(
                    T.evaluate(
                        T.ptx[tma_g2s_2d](
                            dst,
                            T.address_of(tensor_map),
                            T.cast(coord0, "int32"),
                            T.cast(coord1, "int32"),
                            barrier_ptr,
                            cache_policy_evict_normal,
                        )
                    )
                )

            def tma_load_2d_weights(dst, barrier_ptr, tensor_map, coord0, coord1):
                _builder_emit(
                    T.static_assert(
                        cache_hint_sm90_evict_normal == cache_hint_sm100_evict_normal,
                        "Invalid cache hint",
                    )
                )
                _builder_emit(
                    T.static_assert(weights_tma_num_inner_atoms == 1, "Unsupported split TMA atom")
                )
                _builder_emit(
                    T.evaluate(
                        T.ptx[tma_g2s_2d](
                            dst,
                            T.address_of(tensor_map),
                            T.cast(coord0, "int32"),
                            T.cast(coord1, "int32"),
                            barrier_ptr,
                            cache_policy_evict_normal,
                        )
                    )
                )

            def tma_load_3d_kv(dst, barrier_ptr, tensor_map, coord0, coord1, coord2):
                _builder_emit(
                    T.static_assert(
                        cache_hint_sm90_evict_normal == cache_hint_sm100_evict_normal,
                        "Invalid cache hint",
                    )
                )
                _builder_emit(
                    T.static_assert(kv_tma_num_inner_atoms == 1, "Unsupported split TMA atom")
                )
                _builder_emit(
                    T.evaluate(
                        T.ptx[tma_g2s_3d](
                            dst,
                            T.address_of(tensor_map),
                            T.cast(coord0, "int32"),
                            T.cast(coord1, "int32"),
                            T.cast(coord2, "int32"),
                            barrier_ptr,
                            cache_policy_evict_normal,
                        )
                    )
                )

            def tma_load_2d_kv_scales(dst, barrier_ptr, tensor_map, coord0, coord1):
                _builder_emit(
                    T.static_assert(
                        cache_hint_sm90_evict_normal == cache_hint_sm100_evict_normal,
                        "Invalid cache hint",
                    )
                )
                _builder_emit(
                    T.static_assert(
                        kv_scales_tma_num_inner_atoms == 1, "Unsupported split TMA atom"
                    )
                )
                _builder_emit(
                    T.evaluate(
                        T.ptx[tma_g2s_2d](
                            dst,
                            T.address_of(tensor_map),
                            T.cast(coord0, "int32"),
                            T.cast(coord1, "int32"),
                            barrier_ptr,
                            cache_policy_evict_normal,
                        )
                    )
                )

            def make_smem_desc(desc, smem_ptr):
                _builder_emit(
                    T.cuda.tcgen05.encode_matrix_descriptor(
                        T.address_of(desc), smem_ptr, ldo=0, sdo=desc_sdo, swizzle=desc_swizzle
                    )
                )

            def issue_tma_q(stage_idx, tma_q_atom_idx):
                with T.If(T.cuda.elect_sync()):
                    with T.Then():
                        q_token_idx = _builder_scalar(
                            "q_token_idx", atom_to_token_idx_expr(tma_q_atom_idx), "uint32"
                        )
                        _builder_emit(
                            tma_load_2d_q(
                                smem_q.ptr_to([stage_idx, 0, 0]),
                                smem_barriers.ptr_to([full_q_barrier_base + stage_idx]),
                                tensor_map_q,
                                T.uint32(0),
                                q_token_idx * T.uint32(num_heads),
                            )
                        )
                        _builder_emit(
                            tma_load_2d_weights(
                                smem_weights.ptr_to([stage_idx, 0, 0]),
                                smem_barriers.ptr_to([full_q_barrier_base + stage_idx]),
                                tensor_map_weights,
                                T.uint32(0),
                                q_token_idx,
                            )
                        )
                        _builder_emit(
                            mbarrier_arrive_and_expect_tx(
                                smem_barriers.ptr_to([full_q_barrier_base + stage_idx]),
                                smem_q_size_per_stage + smem_weight_size_per_stage,
                            )
                        )

            def fetch_next_task(
                current_q_atom_idx_arg,
                current_kv_idx_arg,
                current_num_kv_arg,
                end_q_atom_idx_arg,
                end_kv_idx_arg,
            ):
                def scalar_value(value):
                    return value.scalar if isinstance(value, T.scalar_wrapper) else value

                current_q_atom_idx_arg = scalar_value(current_q_atom_idx_arg)
                current_kv_idx_arg = scalar_value(current_kv_idx_arg)
                current_num_kv_arg = scalar_value(current_num_kv_arg)
                end_q_atom_idx_arg = scalar_value(end_q_atom_idx_arg)
                end_kv_idx_arg = scalar_value(end_kv_idx_arg)
                T.buffer_store(scheduler_result, current_q_atom_idx_arg, [0])
                T.buffer_store(scheduler_result, current_kv_idx_arg, [1])
                T.buffer_store(scheduler_result, current_num_kv_arg, [2])
                T.buffer_store(scheduler_result, current_q_atom_idx_arg, [4])
                T.buffer_store(scheduler_result, current_kv_idx_arg, [5])
                T.buffer_store(scheduler_result, current_num_kv_arg, [6])
                with T.If(
                    T.And(
                        current_q_atom_idx_arg == end_q_atom_idx_arg,
                        current_kv_idx_arg == end_kv_idx_arg,
                    )
                ):
                    with T.Then():
                        T.buffer_store(scheduler_result, T.uint32(0), [3])
                    with T.Else():
                        T.buffer_store(
                            scheduler_result,
                            current_kv_idx_arg + T.uint32(num_tiles_per_split),
                            [5],
                        )
                        with T.If(scheduler_result[5] >= current_num_kv_arg):
                            with T.Then():
                                T.buffer_store(scheduler_result, T.uint32(0), [5])
                                _builder_emit(
                                    load_atom_advance(current_q_atom_idx_arg, end_q_atom_idx_arg)
                                )
                                T.buffer_store(
                                    scheduler_result,
                                    current_q_atom_idx_arg + atom_advance_result[0],
                                    [4],
                                )
                                with T.If(
                                    T.And(
                                        should_refresh_num_kv_expr(scheduler_result[4]),
                                        exist_q_atom_idx_expr(
                                            scheduler_result[4], end_q_atom_idx_arg, end_kv_idx_arg
                                        ),
                                    )
                                ):
                                    with T.Then():
                                        _builder_emit(load_num_kv(scheduler_result[4], batch_size))
                                        T.buffer_store(scheduler_result, num_kv_result[0], [6])
                        T.buffer_store(scheduler_result, T.uint32(1), [3])

            # Early schedule-metadata load: issue the global loads before the
            # pipeline/barrier prologue so the ~200-cycle L2 latency overlaps with
            # the setup below (matches the CuTeDSL baseline).
            start_q_atom_idx = _builder_assign(
                "start_q_atom_idx",
                T.local_scalar("uint32"),
                locals().get("start_q_atom_idx", _BUILDER_MISSING),
            )
            start_kv_tile_idx = _builder_assign(
                "start_kv_tile_idx",
                T.local_scalar("uint32"),
                locals().get("start_kv_tile_idx", _BUILDER_MISSING),
            )
            end_q_atom_idx = _builder_assign(
                "end_q_atom_idx",
                T.local_scalar("uint32"),
                locals().get("end_q_atom_idx", _BUILDER_MISSING),
            )
            end_kv_tile_idx = _builder_assign(
                "end_kv_tile_idx",
                T.local_scalar("uint32"),
                locals().get("end_kv_tile_idx", _BUILDER_MISSING),
            )
            _builder_emit(
                T.ptx.ld.global_.u32(
                    start_q_atom_idx,
                    schedule_meta_flat.ptr_to([T.cast(sm_idx_u32 * T.uint32(2), "int32")]),
                )
            )
            _builder_emit(
                T.ptx.ld.global_.u32(
                    start_kv_tile_idx,
                    schedule_meta_flat.ptr_to(
                        [T.cast(sm_idx_u32 * T.uint32(2) + T.uint32(1), "int32")]
                    ),
                )
            )
            _builder_emit(
                T.ptx.ld.global_.u32(
                    end_q_atom_idx,
                    schedule_meta_flat.ptr_to(
                        [T.cast((sm_idx_u32 + T.uint32(1)) * T.uint32(2), "int32")]
                    ),
                )
            )
            _builder_emit(
                T.ptx.ld.global_.u32(
                    end_kv_tile_idx,
                    schedule_meta_flat.ptr_to(
                        [T.cast((sm_idx_u32 + T.uint32(1)) * T.uint32(2) + T.uint32(1), "int32")]
                    ),
                )
            )
            start_kv_idx = _builder_bind(
                "start_kv_idx", start_kv_tile_idx * T.uint32(num_tiles_per_split), None
            )
            end_kv_idx = _builder_bind(
                "end_kv_idx", end_kv_tile_idx * T.uint32(num_tiles_per_split), None
            )
            # Clamp the context-length read for zero-work CTAs (start == total q
            # atoms); the value is stale but never used because has_work is false
            # (same clamp as the CuTeDSL baseline).
            _builder_emit(
                load_num_kv(
                    T.min(
                        start_q_atom_idx.scalar,
                        batch_size * T.uint32(num_next_n_atoms) - T.uint32(1),
                    ),
                    batch_size,
                )
            )
            start_num_kv = _builder_bind("start_num_kv", num_kv_result[0], None)

            # Warm the block table into L2 as early as possible. Race-safe: a stale
            # prefetched line is invalidated by any later producer write, so the
            # PDL contract is unaffected. The first task's block-table read then
            # hits L2 instead of paying a full DRAM round trip (the benchmark
            # flushes L2 before every timed call, so the first read is always cold).
            with T.If(T.Or(warp_idx == tma_warp_0, warp_idx == tma_warp_1)):
                with T.Then():
                    with T.unroll(0, (num_prefetch_lines + 63) // 64) as pf_i:
                        IRBuilder.name("pf_i", pf_i)
                        line_idx = _builder_scalar(
                            "line_idx",
                            (
                                (warp_idx_u32 - T.uint32(tma_warp_0)) * T.uint32(32)
                                + lane_idx_u32
                                + T.uint32(pf_i * 64)
                            ),
                            "uint32",
                        )
                        with T.If(line_idx < T.uint32(num_prefetch_lines)):
                            with T.Then():
                                _builder_emit(
                                    T.ptx.prefetch.global_.L2(
                                        block_table_flat.ptr_to(
                                            [T.cast(line_idx * T.uint32(32), "int64")]
                                        )
                                    )
                                )

            with T.If(warp_idx == tma_warp_0):
                with T.Then():
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            with T.unroll(0, num_q_stages) as init_i:
                                IRBuilder.name("init_i", init_i)
                                _builder_emit(
                                    mbarrier_init_cta(
                                        smem_barriers.ptr_to([full_q_barrier_base + init_i]),
                                        T.uint32(1),
                                    )
                                )
                                _builder_emit(
                                    mbarrier_init_cta(
                                        smem_barriers.ptr_to([empty_q_barrier_base + init_i]),
                                        T.uint32(8),
                                    )
                                )
                            _builder_emit(T.ptx.fence.mbarrier_init.release.cluster())
            with T.If(warp_idx == tma_warp_1):
                with T.Then():
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            with T.unroll(0, num_kv_stages) as init_i:
                                IRBuilder.name("init_i", init_i)
                                _builder_emit(
                                    mbarrier_init_cta(
                                        smem_barriers.ptr_to([full_kv_barrier_base + init_i]),
                                        T.uint32(1),
                                    )
                                )
                                _builder_emit(
                                    mbarrier_init_cta(
                                        smem_barriers.ptr_to([empty_kv_barrier_base + init_i]),
                                        T.uint32(4),
                                    )
                                )
                            _builder_emit(T.ptx.fence.mbarrier_init.release.cluster())
            with T.If(warp_idx == umma_warp_0):
                with T.Then():
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            with T.unroll(0, num_kv_stages) as init_i:
                                IRBuilder.name("init_i", init_i)
                                _builder_emit(
                                    mbarrier_init_cta(
                                        smem_barriers.ptr_to(
                                            [
                                                full_kv_barrier_base
                                                + T.uint32(num_kv_stages)
                                                + init_i
                                            ]
                                        ),
                                        T.uint32(1),
                                    )
                                )
                                _builder_emit(
                                    mbarrier_init_cta(
                                        smem_barriers.ptr_to(
                                            [
                                                empty_kv_barrier_base
                                                + T.uint32(num_kv_stages)
                                                + init_i
                                            ]
                                        ),
                                        T.uint32(4),
                                    )
                                )
                            _builder_emit(T.ptx.fence.mbarrier_init.release.cluster())
            with T.If(warp_idx == umma_warp_0 + 1):
                with T.Then():
                    with T.If(T.cuda.elect_sync()):
                        with T.Then():
                            with T.unroll(0, num_umma_barriers) as init_i:
                                IRBuilder.name("init_i", init_i)
                                _builder_emit(
                                    mbarrier_init_cta(
                                        smem_barriers.ptr_to([full_umma_barrier_base + init_i]),
                                        T.uint32(1),
                                    )
                                )
                                _builder_emit(
                                    mbarrier_init_cta(
                                        smem_barriers.ptr_to([empty_umma_barrier_base + init_i]),
                                        T.uint32(4),
                                    )
                                )
                            _builder_emit(T.ptx.fence.mbarrier_init.release.cluster())
            _builder_emit(T.cuda.cta_sync())

            _builder_emit(cuda_grid_dependency_synchronize())

            with T.If(warp_idx == tma_warp_0):
                with T.Then():
                    # TMA warp 0: loads Q + weights (shared) and KV/scales for group 0.
                    _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(num_specialized_registers))
                    current_q_atom_idx = _builder_scalar(
                        "current_q_atom_idx", start_q_atom_idx.scalar, "uint32"
                    )
                    current_kv_idx = _builder_scalar("current_kv_idx", start_kv_idx, "uint32")
                    current_num_kv = _builder_scalar("current_num_kv", start_num_kv, "uint32")
                    q_iter_idx = _builder_scalar("q_iter_idx", T.uint32(0), "uint32")
                    kv_iter_idx = _builder_scalar("kv_iter_idx", T.uint32(0), "uint32")
                    q_stage_idx = _builder_scalar("q_stage_idx", T.uint32(0), "uint32")
                    q_phase = _builder_scalar("q_phase", T.uint32(0), "uint32")
                    q_atom_idx = _builder_scalar(
                        "q_atom_idx", batch_size * T.uint32(num_next_n_atoms), "uint32"
                    )
                    kv_idx = _builder_scalar("kv_idx", T.uint32(0), "uint32")
                    num_kv = _builder_scalar("num_kv", T.uint32(0), "uint32")
                    next_q_atom_idx = _builder_scalar(
                        "next_q_atom_idx", current_q_atom_idx, "uint32"
                    )
                    next_kv_idx = _builder_scalar("next_kv_idx", current_kv_idx, "uint32")
                    next_num_kv = _builder_scalar("next_num_kv", current_num_kv, "uint32")
                    _builder_emit(
                        fetch_next_task(
                            current_q_atom_idx,
                            current_kv_idx,
                            current_num_kv,
                            end_q_atom_idx,
                            end_kv_idx,
                        )
                    )
                    next_q_atom_idx = _builder_assign(
                        "next_q_atom_idx",
                        scheduler_result[0],
                        locals().get("next_q_atom_idx", _BUILDER_MISSING),
                    )
                    next_kv_idx = _builder_assign(
                        "next_kv_idx",
                        scheduler_result[1],
                        locals().get("next_kv_idx", _BUILDER_MISSING),
                    )
                    next_num_kv = _builder_assign(
                        "next_num_kv",
                        scheduler_result[2],
                        locals().get("next_num_kv", _BUILDER_MISSING),
                    )
                    fetched_next_task = _builder_scalar(
                        "fetched_next_task", scheduler_result[3] != T.uint32(0), "bool"
                    )
                    current_q_atom_idx = _builder_assign(
                        "current_q_atom_idx",
                        scheduler_result[4],
                        locals().get("current_q_atom_idx", _BUILDER_MISSING),
                    )
                    current_kv_idx = _builder_assign(
                        "current_kv_idx",
                        scheduler_result[5],
                        locals().get("current_kv_idx", _BUILDER_MISSING),
                    )
                    current_num_kv = _builder_assign(
                        "current_num_kv",
                        scheduler_result[6],
                        locals().get("current_num_kv", _BUILDER_MISSING),
                    )
                    with T.If(fetched_next_task):
                        with T.Then():
                            _builder_emit(issue_tma_q(T.uint32(0), next_q_atom_idx))
                            q_iter_idx = _builder_assign(
                                "q_iter_idx",
                                T.uint32(1),
                                locals().get("q_iter_idx", _BUILDER_MISSING),
                            )

                    kv_block_idx_ptr = _builder_scalar("kv_block_idx_ptr", T.uint32(32), "uint32")
                    cached_kv_blocks = _builder_assign(
                        "cached_kv_blocks",
                        T.alloc_local((num_pages_per_tile,), "uint32"),
                        locals().get("cached_kv_blocks", _BUILDER_MISSING),
                    )

                    with T.While(fetched_next_task):
                        _builder_emit(load_atom_advance(next_q_atom_idx, batch_size))
                        next_advance = _builder_scalar(
                            "next_advance", atom_advance_result[0], "uint32"
                        )
                        prefetch_q = _builder_scalar(
                            "prefetch_q",
                            T.And(
                                q_atom_idx != next_q_atom_idx,
                                exist_q_atom_idx_expr(
                                    next_q_atom_idx + next_advance, end_q_atom_idx, end_kv_idx
                                ),
                            ),
                            "bool",
                        )
                        with T.If(q_atom_idx != next_q_atom_idx):
                            with T.Then():
                                kv_block_idx_ptr = _builder_assign(
                                    "kv_block_idx_ptr",
                                    T.uint32(32),
                                    locals().get("kv_block_idx_ptr", _BUILDER_MISSING),
                                )
                        q_atom_idx = _builder_assign(
                            "q_atom_idx",
                            next_q_atom_idx,
                            locals().get("q_atom_idx", _BUILDER_MISSING),
                        )
                        kv_idx = _builder_assign(
                            "kv_idx", next_kv_idx, locals().get("kv_idx", _BUILDER_MISSING)
                        )
                        num_kv = _builder_assign(
                            "num_kv", next_num_kv, locals().get("num_kv", _BUILDER_MISSING)
                        )

                        # Prefetch the next Q atom as soon as the batch changes so the
                        # Q TMA overlaps the block-table load below (CuTeDSL order).
                        with T.If(prefetch_q):
                            with T.Then():
                                _builder_emit(get_q_pipeline(q_iter_idx))
                                q_stage_idx = _builder_assign(
                                    "q_stage_idx",
                                    fetch_result[0],
                                    locals().get("q_stage_idx", _BUILDER_MISSING),
                                )
                                q_phase = _builder_assign(
                                    "q_phase",
                                    fetch_result[1],
                                    locals().get("q_phase", _BUILDER_MISSING),
                                )
                                q_iter_idx = _builder_assign(
                                    "q_iter_idx",
                                    q_iter_idx + T.uint32(1),
                                    locals().get("q_iter_idx", _BUILDER_MISSING),
                                )
                                _builder_emit(
                                    mbarrier_wait_phase(
                                        smem_barriers.ptr_to([empty_q_barrier_base + q_stage_idx]),
                                        q_phase ^ T.uint32(1),
                                    )
                                )
                                _builder_emit(issue_tma_q(q_stage_idx, q_atom_idx + next_advance))

                        with T.If(kv_block_idx_ptr == T.uint32(32)):
                            with T.Then():
                                kv_block_idx_ptr = _builder_assign(
                                    "kv_block_idx_ptr",
                                    T.uint32(0),
                                    locals().get("kv_block_idx_ptr", _BUILDER_MISSING),
                                )
                                block_table_offset = _builder_scalar(
                                    "block_table_offset",
                                    T.cast(atom_to_block_table_row_expr(q_atom_idx), "uint64")
                                    * T.cast(block_table_stride, "uint64"),
                                    "uint64",
                                )
                                prefetch_tile_idx = _builder_scalar(
                                    "prefetch_tile_idx",
                                    kv_idx + lane_idx_u32 * T.uint32(num_tiles_per_split),
                                    "uint32",
                                )
                                block_table_index = _builder_scalar(
                                    "block_table_index",
                                    block_table_offset
                                    + T.cast(
                                        prefetch_tile_idx * T.uint32(num_pages_per_tile), "uint64"
                                    ),
                                    "uint64",
                                )
                                with T.unroll(0, num_pages_per_tile) as block_i:
                                    IRBuilder.name("block_i", block_i)
                                    # Guard the trailing partial tile: a valid compute tile
                                    # may still exceed the block table's row length, and an
                                    # out-of-range garbage page id would send TMA out of
                                    # bounds (page 0 is used as the masked-dumpster tile).
                                    with T.If(
                                        T.And(
                                            prefetch_tile_idx < num_kv,
                                            prefetch_tile_idx * T.uint32(num_pages_per_tile)
                                            + T.uint32(block_i)
                                            < T.uint32(config.max_num_pages),
                                        )
                                    ):
                                        with T.Then():
                                            _builder_emit(
                                                T.ptx.ld.global_.u32(
                                                    cached_kv_blocks[block_i],
                                                    block_table_flat.ptr_to(
                                                        [
                                                            T.cast(
                                                                block_table_index
                                                                + T.cast(block_i, "uint64"),
                                                                "int64",
                                                            )
                                                        ]
                                                    ),
                                                )
                                            )
                                        with T.Else():
                                            T.buffer_store(cached_kv_blocks, T.uint32(0), [block_i])
                        _builder_emit(T.cuda.warp_sync())

                        kv_block_idx = _builder_assign(
                            "kv_block_idx",
                            T.alloc_local((num_pages_per_tile,), "uint32"),
                            locals().get("kv_block_idx", _BUILDER_MISSING),
                        )
                        with T.unroll(0, num_pages_per_tile) as block_i:
                            IRBuilder.name("block_i", block_i)
                            T.buffer_store(
                                kv_block_idx,
                                T.cuda.__shfl_sync(
                                    T.uint32(0xFFFFFFFF),
                                    cached_kv_blocks[block_i],
                                    kv_block_idx_ptr,
                                    32,
                                ),
                                [block_i],
                            )
                        kv_block_idx_ptr = _builder_assign(
                            "kv_block_idx_ptr",
                            kv_block_idx_ptr + T.uint32(1),
                            locals().get("kv_block_idx_ptr", _BUILDER_MISSING),
                        )

                        _builder_emit(get_kv_pipeline(kv_iter_idx))
                        kv_stage_idx = _builder_scalar("kv_stage_idx", fetch_result[2], "uint32")
                        kv_phase = _builder_scalar("kv_phase", fetch_result[3], "uint32")
                        kv_iter_idx = _builder_assign(
                            "kv_iter_idx",
                            kv_iter_idx + T.uint32(1),
                            locals().get("kv_iter_idx", _BUILDER_MISSING),
                        )
                        _builder_emit(
                            mbarrier_wait_phase(
                                smem_barriers.ptr_to([empty_kv_barrier_base + kv_stage_idx]),
                                kv_phase ^ T.uint32(1),
                            )
                        )

                        with T.If(T.cuda.elect_sync()):
                            with T.Then():
                                with T.unroll(0, num_pages_per_tile) as block_i:
                                    IRBuilder.name("block_i", block_i)
                                    _builder_emit(
                                        tma_load_3d_kv(
                                            smem_kv.ptr_to(
                                                [0, kv_stage_idx, block_i * page_size, 0]
                                            ),
                                            smem_barriers.ptr_to(
                                                [full_kv_barrier_base + kv_stage_idx]
                                            ),
                                            tensor_map_kv,
                                            T.uint32(0),
                                            T.uint32(0),
                                            kv_block_idx[block_i],
                                        )
                                    )
                                    _builder_emit(
                                        tma_load_2d_kv_scales(
                                            smem_kv_scales.ptr_to(
                                                [0, kv_stage_idx, block_i * page_size]
                                            ),
                                            smem_barriers.ptr_to(
                                                [full_kv_barrier_base + kv_stage_idx]
                                            ),
                                            tensor_map_kv_scales,
                                            T.uint32(0),
                                            kv_block_idx[block_i],
                                        )
                                    )
                                _builder_emit(
                                    mbarrier_arrive_and_expect_tx(
                                        smem_barriers.ptr_to([full_kv_barrier_base + kv_stage_idx]),
                                        smem_kv_size_per_stage + smem_kv_scale_size_per_stage,
                                    )
                                )

                        next_q_atom_idx = _builder_assign(
                            "next_q_atom_idx",
                            current_q_atom_idx,
                            locals().get("next_q_atom_idx", _BUILDER_MISSING),
                        )
                        next_kv_idx = _builder_assign(
                            "next_kv_idx",
                            current_kv_idx,
                            locals().get("next_kv_idx", _BUILDER_MISSING),
                        )
                        next_num_kv = _builder_assign(
                            "next_num_kv",
                            current_num_kv,
                            locals().get("next_num_kv", _BUILDER_MISSING),
                        )
                        _builder_emit(
                            fetch_next_task(
                                current_q_atom_idx,
                                current_kv_idx,
                                current_num_kv,
                                end_q_atom_idx,
                                end_kv_idx,
                            )
                        )
                        next_q_atom_idx = _builder_assign(
                            "next_q_atom_idx",
                            scheduler_result[0],
                            locals().get("next_q_atom_idx", _BUILDER_MISSING),
                        )
                        next_kv_idx = _builder_assign(
                            "next_kv_idx",
                            scheduler_result[1],
                            locals().get("next_kv_idx", _BUILDER_MISSING),
                        )
                        next_num_kv = _builder_assign(
                            "next_num_kv",
                            scheduler_result[2],
                            locals().get("next_num_kv", _BUILDER_MISSING),
                        )
                        fetched_next_task = _builder_assign(
                            "fetched_next_task",
                            scheduler_result[3] != T.uint32(0),
                            locals().get("fetched_next_task", _BUILDER_MISSING),
                        )
                        current_q_atom_idx = _builder_assign(
                            "current_q_atom_idx",
                            scheduler_result[4],
                            locals().get("current_q_atom_idx", _BUILDER_MISSING),
                        )
                        current_kv_idx = _builder_assign(
                            "current_kv_idx",
                            scheduler_result[5],
                            locals().get("current_kv_idx", _BUILDER_MISSING),
                        )
                        current_num_kv = _builder_assign(
                            "current_num_kv",
                            scheduler_result[6],
                            locals().get("current_num_kv", _BUILDER_MISSING),
                        )
                with T.Else():
                    with T.If(warp_idx == tma_warp_1):
                        with T.Then():
                            # TMA warp 1: loads KV/scales for group 1 only.
                            _builder_emit(
                                T.ptx.setmaxnreg.dec.sync.aligned.u32(num_specialized_registers)
                            )
                            current_q_atom_idx = _builder_scalar(
                                "current_q_atom_idx", start_q_atom_idx.scalar, "uint32"
                            )
                            current_kv_idx = _builder_scalar(
                                "current_kv_idx", start_kv_idx, "uint32"
                            )
                            current_num_kv = _builder_scalar(
                                "current_num_kv", start_num_kv, "uint32"
                            )
                            kv_iter_idx = _builder_scalar("kv_iter_idx", T.uint32(0), "uint32")
                            q_atom_idx = _builder_scalar(
                                "q_atom_idx", batch_size * T.uint32(num_next_n_atoms), "uint32"
                            )
                            kv_idx = _builder_scalar("kv_idx", T.uint32(0), "uint32")
                            num_kv = _builder_scalar("num_kv", T.uint32(0), "uint32")
                            next_q_atom_idx = _builder_scalar(
                                "next_q_atom_idx", current_q_atom_idx, "uint32"
                            )
                            next_kv_idx = _builder_scalar("next_kv_idx", current_kv_idx, "uint32")
                            next_num_kv = _builder_scalar("next_num_kv", current_num_kv, "uint32")
                            _builder_emit(
                                fetch_next_task(
                                    current_q_atom_idx,
                                    current_kv_idx,
                                    current_num_kv,
                                    end_q_atom_idx,
                                    end_kv_idx,
                                )
                            )
                            next_q_atom_idx = _builder_assign(
                                "next_q_atom_idx",
                                scheduler_result[0],
                                locals().get("next_q_atom_idx", _BUILDER_MISSING),
                            )
                            next_kv_idx = _builder_assign(
                                "next_kv_idx",
                                scheduler_result[1],
                                locals().get("next_kv_idx", _BUILDER_MISSING),
                            )
                            next_num_kv = _builder_assign(
                                "next_num_kv",
                                scheduler_result[2],
                                locals().get("next_num_kv", _BUILDER_MISSING),
                            )
                            fetched_next_task = _builder_scalar(
                                "fetched_next_task", scheduler_result[3] != T.uint32(0), "bool"
                            )
                            current_q_atom_idx = _builder_assign(
                                "current_q_atom_idx",
                                scheduler_result[4],
                                locals().get("current_q_atom_idx", _BUILDER_MISSING),
                            )
                            current_kv_idx = _builder_assign(
                                "current_kv_idx",
                                scheduler_result[5],
                                locals().get("current_kv_idx", _BUILDER_MISSING),
                            )
                            current_num_kv = _builder_assign(
                                "current_num_kv",
                                scheduler_result[6],
                                locals().get("current_num_kv", _BUILDER_MISSING),
                            )

                            kv_block_idx_ptr = _builder_scalar(
                                "kv_block_idx_ptr", T.uint32(32), "uint32"
                            )
                            cached_kv_blocks = _builder_assign(
                                "cached_kv_blocks",
                                T.alloc_local((num_pages_per_tile,), "uint32"),
                                locals().get("cached_kv_blocks", _BUILDER_MISSING),
                            )

                            with T.While(fetched_next_task):
                                with T.If(q_atom_idx != next_q_atom_idx):
                                    with T.Then():
                                        kv_block_idx_ptr = _builder_assign(
                                            "kv_block_idx_ptr",
                                            T.uint32(32),
                                            locals().get("kv_block_idx_ptr", _BUILDER_MISSING),
                                        )
                                q_atom_idx = _builder_assign(
                                    "q_atom_idx",
                                    next_q_atom_idx,
                                    locals().get("q_atom_idx", _BUILDER_MISSING),
                                )
                                kv_idx = _builder_assign(
                                    "kv_idx", next_kv_idx, locals().get("kv_idx", _BUILDER_MISSING)
                                )
                                num_kv = _builder_assign(
                                    "num_kv", next_num_kv, locals().get("num_kv", _BUILDER_MISSING)
                                )

                                with T.If(kv_block_idx_ptr == T.uint32(32)):
                                    with T.Then():
                                        kv_block_idx_ptr = _builder_assign(
                                            "kv_block_idx_ptr",
                                            T.uint32(0),
                                            locals().get("kv_block_idx_ptr", _BUILDER_MISSING),
                                        )
                                        block_table_offset = _builder_scalar(
                                            "block_table_offset",
                                            T.cast(
                                                atom_to_block_table_row_expr(q_atom_idx), "uint64"
                                            )
                                            * T.cast(block_table_stride, "uint64"),
                                            "uint64",
                                        )
                                        prefetch_tile_idx = _builder_scalar(
                                            "prefetch_tile_idx",
                                            (
                                                kv_idx
                                                + T.uint32(1)
                                                + lane_idx_u32 * T.uint32(num_tiles_per_split)
                                            ),
                                            "uint32",
                                        )
                                        block_table_index = _builder_scalar(
                                            "block_table_index",
                                            block_table_offset
                                            + T.cast(
                                                prefetch_tile_idx * T.uint32(num_pages_per_tile),
                                                "uint64",
                                            ),
                                            "uint64",
                                        )
                                        with T.unroll(0, num_pages_per_tile) as block_i:
                                            IRBuilder.name("block_i", block_i)
                                            # Guard the trailing partial tile: a valid compute tile
                                            # may still exceed the block table's row length, and an
                                            # out-of-range garbage page id would send TMA out of
                                            # bounds (page 0 is used as the masked-dumpster tile).
                                            with T.If(
                                                T.And(
                                                    prefetch_tile_idx < num_kv,
                                                    prefetch_tile_idx * T.uint32(num_pages_per_tile)
                                                    + T.uint32(block_i)
                                                    < T.uint32(config.max_num_pages),
                                                )
                                            ):
                                                with T.Then():
                                                    _builder_emit(
                                                        T.ptx.ld.global_.u32(
                                                            cached_kv_blocks[block_i],
                                                            block_table_flat.ptr_to(
                                                                [
                                                                    T.cast(
                                                                        block_table_index
                                                                        + T.cast(block_i, "uint64"),
                                                                        "int64",
                                                                    )
                                                                ]
                                                            ),
                                                        )
                                                    )
                                                with T.Else():
                                                    T.buffer_store(
                                                        cached_kv_blocks, T.uint32(0), [block_i]
                                                    )
                                _builder_emit(T.cuda.warp_sync())

                                kv_block_idx = _builder_assign(
                                    "kv_block_idx",
                                    T.alloc_local((num_pages_per_tile,), "uint32"),
                                    locals().get("kv_block_idx", _BUILDER_MISSING),
                                )
                                with T.unroll(0, num_pages_per_tile) as block_i:
                                    IRBuilder.name("block_i", block_i)
                                    T.buffer_store(
                                        kv_block_idx,
                                        T.cuda.__shfl_sync(
                                            T.uint32(0xFFFFFFFF),
                                            cached_kv_blocks[block_i],
                                            kv_block_idx_ptr,
                                            32,
                                        ),
                                        [block_i],
                                    )
                                kv_block_idx_ptr = _builder_assign(
                                    "kv_block_idx_ptr",
                                    kv_block_idx_ptr + T.uint32(1),
                                    locals().get("kv_block_idx_ptr", _BUILDER_MISSING),
                                )

                                _builder_emit(get_kv_pipeline(kv_iter_idx))
                                kv_stage_idx = _builder_scalar(
                                    "kv_stage_idx", fetch_result[2], "uint32"
                                )
                                kv_phase = _builder_scalar("kv_phase", fetch_result[3], "uint32")
                                kv_iter_idx = _builder_assign(
                                    "kv_iter_idx",
                                    kv_iter_idx + T.uint32(1),
                                    locals().get("kv_iter_idx", _BUILDER_MISSING),
                                )
                                _builder_emit(
                                    mbarrier_wait_phase(
                                        smem_barriers.ptr_to(
                                            [
                                                empty_kv_barrier_base
                                                + T.uint32(num_kv_stages)
                                                + kv_stage_idx
                                            ]
                                        ),
                                        kv_phase ^ T.uint32(1),
                                    )
                                )

                                with T.If(T.cuda.elect_sync()):
                                    with T.Then():
                                        with T.unroll(0, num_pages_per_tile) as block_i:
                                            IRBuilder.name("block_i", block_i)
                                            _builder_emit(
                                                tma_load_3d_kv(
                                                    smem_kv.ptr_to(
                                                        [1, kv_stage_idx, block_i * page_size, 0]
                                                    ),
                                                    smem_barriers.ptr_to(
                                                        [
                                                            full_kv_barrier_base
                                                            + T.uint32(num_kv_stages)
                                                            + kv_stage_idx
                                                        ]
                                                    ),
                                                    tensor_map_kv,
                                                    T.uint32(0),
                                                    T.uint32(0),
                                                    kv_block_idx[block_i],
                                                )
                                            )
                                            _builder_emit(
                                                tma_load_2d_kv_scales(
                                                    smem_kv_scales.ptr_to(
                                                        [1, kv_stage_idx, block_i * page_size]
                                                    ),
                                                    smem_barriers.ptr_to(
                                                        [
                                                            full_kv_barrier_base
                                                            + T.uint32(num_kv_stages)
                                                            + kv_stage_idx
                                                        ]
                                                    ),
                                                    tensor_map_kv_scales,
                                                    T.uint32(0),
                                                    kv_block_idx[block_i],
                                                )
                                            )
                                        _builder_emit(
                                            mbarrier_arrive_and_expect_tx(
                                                smem_barriers.ptr_to(
                                                    [
                                                        full_kv_barrier_base
                                                        + T.uint32(num_kv_stages)
                                                        + kv_stage_idx
                                                    ]
                                                ),
                                                smem_kv_size_per_stage
                                                + smem_kv_scale_size_per_stage,
                                            )
                                        )

                                next_q_atom_idx = _builder_assign(
                                    "next_q_atom_idx",
                                    current_q_atom_idx,
                                    locals().get("next_q_atom_idx", _BUILDER_MISSING),
                                )
                                next_kv_idx = _builder_assign(
                                    "next_kv_idx",
                                    current_kv_idx,
                                    locals().get("next_kv_idx", _BUILDER_MISSING),
                                )
                                next_num_kv = _builder_assign(
                                    "next_num_kv",
                                    current_num_kv,
                                    locals().get("next_num_kv", _BUILDER_MISSING),
                                )
                                _builder_emit(
                                    fetch_next_task(
                                        current_q_atom_idx,
                                        current_kv_idx,
                                        current_num_kv,
                                        end_q_atom_idx,
                                        end_kv_idx,
                                    )
                                )
                                next_q_atom_idx = _builder_assign(
                                    "next_q_atom_idx",
                                    scheduler_result[0],
                                    locals().get("next_q_atom_idx", _BUILDER_MISSING),
                                )
                                next_kv_idx = _builder_assign(
                                    "next_kv_idx",
                                    scheduler_result[1],
                                    locals().get("next_kv_idx", _BUILDER_MISSING),
                                )
                                next_num_kv = _builder_assign(
                                    "next_num_kv",
                                    scheduler_result[2],
                                    locals().get("next_num_kv", _BUILDER_MISSING),
                                )
                                fetched_next_task = _builder_assign(
                                    "fetched_next_task",
                                    scheduler_result[3] != T.uint32(0),
                                    locals().get("fetched_next_task", _BUILDER_MISSING),
                                )
                                current_q_atom_idx = _builder_assign(
                                    "current_q_atom_idx",
                                    scheduler_result[4],
                                    locals().get("current_q_atom_idx", _BUILDER_MISSING),
                                )
                                current_kv_idx = _builder_assign(
                                    "current_kv_idx",
                                    scheduler_result[5],
                                    locals().get("current_kv_idx", _BUILDER_MISSING),
                                )
                                current_num_kv = _builder_assign(
                                    "current_num_kv",
                                    scheduler_result[6],
                                    locals().get("current_num_kv", _BUILDER_MISSING),
                                )
                        with T.Else():
                            with T.If(T.Or(warp_idx == umma_warp_0, warp_idx == umma_warp_0 + 1)):
                                with T.Then():
                                    # One UMMA warp per math warpgroup: waits for its group's KV stage,
                                    # then issues the 4 tcgen05 MMAs (K=32 each) for its group.
                                    _builder_emit(
                                        T.ptx.setmaxnreg.dec.sync.aligned.u32(
                                            num_specialized_registers
                                        )
                                    )
                                    umma_group_idx = _builder_bind(
                                        "umma_group_idx", warp_idx_u32 - T.uint32(umma_warp_0), None
                                    )
                                    # TMEM allocation happens off the full-CTA sync path (the ~300-cycle
                                    # tcgen05.alloc would otherwise hold back the TMA warps' first issue).
                                    # Only the UMMA and Math warps wait for it on named barrier 9 (the
                                    # TMA warps do not touch TMEM). Matches the CuTeDSL TmemAllocator
                                    # split: math warp allocs, consumers sync on a sub-CTA barrier.
                                    with T.If(warp_idx == umma_warp_0 + 1):
                                        with T.Then():
                                            _builder_emit(
                                                T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                                                    T.address_of(tmem_ptr_in_smem[0]),
                                                    T.uint32(num_tmem_cols),
                                                )
                                            )
                                    _builder_emit(
                                        T.ptx.barrier.sync(9, T.uint32(num_math_threads + 2 * 32))
                                    )
                                    current_q_atom_idx = _builder_scalar(
                                        "current_q_atom_idx", start_q_atom_idx.scalar, "uint32"
                                    )
                                    current_kv_idx = _builder_scalar(
                                        "current_kv_idx", start_kv_idx, "uint32"
                                    )
                                    current_num_kv = _builder_scalar(
                                        "current_num_kv", start_num_kv, "uint32"
                                    )
                                    q_iter_idx = _builder_scalar(
                                        "q_iter_idx", T.uint32(0), "uint32"
                                    )
                                    kv_iter_idx = _builder_scalar(
                                        "kv_iter_idx", T.uint32(0), "uint32"
                                    )
                                    q_stage_idx = _builder_scalar(
                                        "q_stage_idx", T.uint32(0), "uint32"
                                    )
                                    q_phase = _builder_scalar("q_phase", T.uint32(0), "uint32")
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
                                    desc_i = _builder_alloc_scalar("desc_i", "uint32")
                                    desc_a = _builder_alloc_scalar("desc_a", "uint64")
                                    desc_b = _builder_alloc_scalar("desc_b", "uint64")
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
                                    q_atom_idx = _builder_scalar(
                                        "q_atom_idx",
                                        batch_size * T.uint32(num_next_n_atoms),
                                        "uint32",
                                    )
                                    kv_idx = _builder_scalar("kv_idx", T.uint32(0), "uint32")
                                    next_q_atom_idx = _builder_scalar(
                                        "next_q_atom_idx", current_q_atom_idx, "uint32"
                                    )
                                    next_kv_idx = _builder_scalar(
                                        "next_kv_idx", current_kv_idx, "uint32"
                                    )
                                    next_num_kv = _builder_scalar(
                                        "next_num_kv", current_num_kv, "uint32"
                                    )
                                    _builder_emit(
                                        fetch_next_task(
                                            current_q_atom_idx,
                                            current_kv_idx,
                                            current_num_kv,
                                            end_q_atom_idx,
                                            end_kv_idx,
                                        )
                                    )
                                    next_q_atom_idx = _builder_assign(
                                        "next_q_atom_idx",
                                        scheduler_result[0],
                                        locals().get("next_q_atom_idx", _BUILDER_MISSING),
                                    )
                                    next_kv_idx = _builder_assign(
                                        "next_kv_idx",
                                        scheduler_result[1],
                                        locals().get("next_kv_idx", _BUILDER_MISSING),
                                    )
                                    next_num_kv = _builder_assign(
                                        "next_num_kv",
                                        scheduler_result[2],
                                        locals().get("next_num_kv", _BUILDER_MISSING),
                                    )
                                    fetched_next_task = _builder_scalar(
                                        "fetched_next_task",
                                        scheduler_result[3] != T.uint32(0),
                                        "bool",
                                    )
                                    current_q_atom_idx = _builder_assign(
                                        "current_q_atom_idx",
                                        scheduler_result[4],
                                        locals().get("current_q_atom_idx", _BUILDER_MISSING),
                                    )
                                    current_kv_idx = _builder_assign(
                                        "current_kv_idx",
                                        scheduler_result[5],
                                        locals().get("current_kv_idx", _BUILDER_MISSING),
                                    )
                                    current_num_kv = _builder_assign(
                                        "current_num_kv",
                                        scheduler_result[6],
                                        locals().get("current_num_kv", _BUILDER_MISSING),
                                    )
                                    umma_iter_idx = _builder_scalar(
                                        "umma_iter_idx", T.uint32(0), "uint32"
                                    )
                                    with T.While(fetched_next_task):
                                        with T.If(q_atom_idx != next_q_atom_idx):
                                            with T.Then():
                                                # Wait for the new Q stage (wait only; Math releases it).
                                                _builder_emit(get_q_pipeline(q_iter_idx))
                                                q_stage_idx = _builder_assign(
                                                    "q_stage_idx",
                                                    fetch_result[0],
                                                    locals().get("q_stage_idx", _BUILDER_MISSING),
                                                )
                                                q_phase = _builder_assign(
                                                    "q_phase",
                                                    fetch_result[1],
                                                    locals().get("q_phase", _BUILDER_MISSING),
                                                )
                                                q_iter_idx = _builder_assign(
                                                    "q_iter_idx",
                                                    q_iter_idx + T.uint32(1),
                                                    locals().get("q_iter_idx", _BUILDER_MISSING),
                                                )
                                                _builder_emit(
                                                    mbarrier_wait_phase(
                                                        smem_barriers.ptr_to(
                                                            [full_q_barrier_base + q_stage_idx]
                                                        ),
                                                        q_phase,
                                                    )
                                                )
                                        q_atom_idx = _builder_assign(
                                            "q_atom_idx",
                                            next_q_atom_idx,
                                            locals().get("q_atom_idx", _BUILDER_MISSING),
                                        )
                                        kv_idx = _builder_assign(
                                            "kv_idx",
                                            next_kv_idx,
                                            locals().get("kv_idx", _BUILDER_MISSING),
                                        )

                                        _builder_emit(get_kv_pipeline(kv_iter_idx))
                                        kv_stage_idx = _builder_scalar(
                                            "kv_stage_idx", fetch_result[2], "uint32"
                                        )
                                        kv_phase = _builder_scalar(
                                            "kv_phase", fetch_result[3], "uint32"
                                        )
                                        kv_iter_idx = _builder_assign(
                                            "kv_iter_idx",
                                            kv_iter_idx + T.uint32(1),
                                            locals().get("kv_iter_idx", _BUILDER_MISSING),
                                        )
                                        _builder_emit(
                                            mbarrier_wait_phase(
                                                smem_barriers.ptr_to(
                                                    [
                                                        full_kv_barrier_base
                                                        + umma_group_idx * T.uint32(num_kv_stages)
                                                        + kv_stage_idx
                                                    ]
                                                ),
                                                kv_phase,
                                            )
                                        )
                                        umma_stage_idx = _builder_scalar(
                                            "umma_stage_idx",
                                            umma_iter_idx % T.uint32(num_umma_stages),
                                            "uint32",
                                        )
                                        umma_phase = _builder_scalar(
                                            "umma_phase",
                                            (umma_iter_idx // T.uint32(num_umma_stages))
                                            & T.uint32(1),
                                            "uint32",
                                        )
                                        umma_iter_idx = _builder_assign(
                                            "umma_iter_idx",
                                            umma_iter_idx + T.uint32(1),
                                            locals().get("umma_iter_idx", _BUILDER_MISSING),
                                        )
                                        _builder_emit(
                                            mbarrier_wait_phase(
                                                smem_barriers.ptr_to(
                                                    [
                                                        empty_umma_barrier_base
                                                        + umma_group_idx * T.uint32(num_umma_stages)
                                                        + umma_stage_idx
                                                    ]
                                                ),
                                                umma_phase ^ T.uint32(1),
                                            )
                                        )
                                        _builder_emit(T.ptx.tcgen05.fence__after_thread_sync())
                                        _builder_emit(
                                            T.static_assert(
                                                head_dim % umma_k == 0, "Invalid head dim"
                                            )
                                        )
                                        with T.unroll(0, head_dim // umma_k) as k:
                                            IRBuilder.name("k", k)
                                            _builder_emit(
                                                make_smem_desc(
                                                    desc_a,
                                                    smem_kv.ptr_to(
                                                        [
                                                            umma_group_idx,
                                                            kv_stage_idx,
                                                            0,
                                                            k * umma_k,
                                                        ]
                                                    ),
                                                )
                                            )
                                            _builder_emit(
                                                make_smem_desc(
                                                    desc_b,
                                                    smem_q.ptr_to([q_stage_idx, 0, k * umma_k]),
                                                )
                                            )
                                            with T.If(T.cuda.elect_sync()):
                                                with T.Then():
                                                    _builder_emit(
                                                        T.ptx[
                                                            "tcgen05.mma.cta_group::1.kind::f8f6f4"
                                                        ](
                                                            umma_group_idx
                                                            * T.uint32(umma_n * num_umma_stages)
                                                            + umma_stage_idx * T.uint32(umma_n),
                                                            desc_a,
                                                            desc_b,
                                                            runtime_instr_desc_hi,
                                                            T.uint32(0),
                                                            T.uint32(0),
                                                            T.uint32(0),
                                                            T.uint32(0),
                                                            T.ptx.pred(T.uint32(k)),
                                                        )
                                                    )
                                        with T.If(T.cuda.elect_sync()):
                                            with T.Then():
                                                _builder_emit(
                                                    T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                                                        smem_barriers.ptr_to(
                                                            [
                                                                full_umma_barrier_base
                                                                + umma_group_idx
                                                                * T.uint32(num_umma_stages)
                                                                + umma_stage_idx
                                                            ]
                                                        )
                                                    )
                                                )
                                        next_q_atom_idx = _builder_assign(
                                            "next_q_atom_idx",
                                            current_q_atom_idx,
                                            locals().get("next_q_atom_idx", _BUILDER_MISSING),
                                        )
                                        next_kv_idx = _builder_assign(
                                            "next_kv_idx",
                                            current_kv_idx,
                                            locals().get("next_kv_idx", _BUILDER_MISSING),
                                        )
                                        next_num_kv = _builder_assign(
                                            "next_num_kv",
                                            current_num_kv,
                                            locals().get("next_num_kv", _BUILDER_MISSING),
                                        )
                                        _builder_emit(
                                            fetch_next_task(
                                                current_q_atom_idx,
                                                current_kv_idx,
                                                current_num_kv,
                                                end_q_atom_idx,
                                                end_kv_idx,
                                            )
                                        )
                                        next_q_atom_idx = _builder_assign(
                                            "next_q_atom_idx",
                                            scheduler_result[0],
                                            locals().get("next_q_atom_idx", _BUILDER_MISSING),
                                        )
                                        next_kv_idx = _builder_assign(
                                            "next_kv_idx",
                                            scheduler_result[1],
                                            locals().get("next_kv_idx", _BUILDER_MISSING),
                                        )
                                        next_num_kv = _builder_assign(
                                            "next_num_kv",
                                            scheduler_result[2],
                                            locals().get("next_num_kv", _BUILDER_MISSING),
                                        )
                                        fetched_next_task = _builder_assign(
                                            "fetched_next_task",
                                            scheduler_result[3] != T.uint32(0),
                                            locals().get("fetched_next_task", _BUILDER_MISSING),
                                        )
                                        current_q_atom_idx = _builder_assign(
                                            "current_q_atom_idx",
                                            scheduler_result[4],
                                            locals().get("current_q_atom_idx", _BUILDER_MISSING),
                                        )
                                        current_kv_idx = _builder_assign(
                                            "current_kv_idx",
                                            scheduler_result[5],
                                            locals().get("current_kv_idx", _BUILDER_MISSING),
                                        )
                                        current_num_kv = _builder_assign(
                                            "current_num_kv",
                                            scheduler_result[6],
                                            locals().get("current_num_kv", _BUILDER_MISSING),
                                        )
                                with T.Else():
                                    with T.If(warp_idx < spec_warp_start):
                                        with T.Then():
                                            _builder_emit(
                                                T.ptx.setmaxnreg.inc.sync.aligned.u32(
                                                    num_math_registers
                                                )
                                            )
                                            # Math warps consume TMEM: wait on named barrier 9 for the UMMA
                                            # warp's tcgen05.alloc (see the UMMA branch).
                                            _builder_emit(
                                                T.ptx.barrier.sync(
                                                    9, T.uint32(num_math_threads + 2 * 32)
                                                )
                                            )
                                            current_q_atom_idx = _builder_scalar(
                                                "current_q_atom_idx",
                                                start_q_atom_idx.scalar,
                                                "uint32",
                                            )
                                            current_kv_idx = _builder_scalar(
                                                "current_kv_idx", start_kv_idx, "uint32"
                                            )
                                            current_num_kv = _builder_scalar(
                                                "current_num_kv", start_num_kv, "uint32"
                                            )
                                            q_iter_idx = _builder_scalar(
                                                "q_iter_idx", T.uint32(0), "uint32"
                                            )
                                            kv_iter_idx = _builder_scalar(
                                                "kv_iter_idx", T.uint32(0), "uint32"
                                            )
                                            q_stage_idx = _builder_scalar(
                                                "q_stage_idx", T.uint32(0), "uint32"
                                            )
                                            q_phase = _builder_scalar(
                                                "q_phase", T.uint32(0), "uint32"
                                            )
                                            math_warpgroup_idx = _builder_scalar(
                                                "math_warpgroup_idx", warpgroup_idx, "int32"
                                            )
                                            math_wg_u32 = _builder_bind(
                                                "math_wg_u32",
                                                T.cast(math_warpgroup_idx, "uint32"),
                                                None,
                                            )
                                            tmem_start_base = _builder_bind(
                                                "tmem_start_base",
                                                math_wg_u32 * T.uint32(umma_n * num_umma_stages),
                                                None,
                                            )
                                            math_thread_idx = _builder_scalar(
                                                "math_thread_idx",
                                                (warp_idx_u32 % T.uint32(4)) * T.uint32(32)
                                                + lane_idx_u32,
                                                "uint32",
                                            )
                                            cached_weights = _builder_assign(
                                                "cached_weights",
                                                T.alloc_local((next_n_atom, num_heads), "float32"),
                                                locals().get("cached_weights", _BUILDER_MISSING),
                                            )
                                            q_atom_idx = _builder_scalar(
                                                "q_atom_idx",
                                                batch_size * T.uint32(num_next_n_atoms),
                                                "uint32",
                                            )
                                            next_q_atom_idx = _builder_scalar(
                                                "next_q_atom_idx", current_q_atom_idx, "uint32"
                                            )
                                            next_kv_idx = _builder_scalar(
                                                "next_kv_idx", current_kv_idx, "uint32"
                                            )
                                            next_num_kv = _builder_scalar(
                                                "next_num_kv", current_num_kv, "uint32"
                                            )
                                            _builder_emit(
                                                fetch_next_task(
                                                    current_q_atom_idx,
                                                    current_kv_idx,
                                                    current_num_kv,
                                                    end_q_atom_idx,
                                                    end_kv_idx,
                                                )
                                            )
                                            next_q_atom_idx = _builder_assign(
                                                "next_q_atom_idx",
                                                scheduler_result[0],
                                                locals().get("next_q_atom_idx", _BUILDER_MISSING),
                                            )
                                            next_kv_idx = _builder_assign(
                                                "next_kv_idx",
                                                scheduler_result[1],
                                                locals().get("next_kv_idx", _BUILDER_MISSING),
                                            )
                                            next_num_kv = _builder_assign(
                                                "next_num_kv",
                                                scheduler_result[2],
                                                locals().get("next_num_kv", _BUILDER_MISSING),
                                            )
                                            fetched_next_task = _builder_scalar(
                                                "fetched_next_task",
                                                scheduler_result[3] != T.uint32(0),
                                                "bool",
                                            )
                                            current_q_atom_idx = _builder_assign(
                                                "current_q_atom_idx",
                                                scheduler_result[4],
                                                locals().get(
                                                    "current_q_atom_idx", _BUILDER_MISSING
                                                ),
                                            )
                                            current_kv_idx = _builder_assign(
                                                "current_kv_idx",
                                                scheduler_result[5],
                                                locals().get("current_kv_idx", _BUILDER_MISSING),
                                            )
                                            current_num_kv = _builder_assign(
                                                "current_num_kv",
                                                scheduler_result[6],
                                                locals().get("current_num_kv", _BUILDER_MISSING),
                                            )
                                            umma_iter_idx = _builder_scalar(
                                                "umma_iter_idx", T.uint32(0), "uint32"
                                            )
                                            is_paired_atom = _builder_scalar(
                                                "is_paired_atom", T.bool(False), "bool"
                                            )
                                            _builder_emit(
                                                T.static_assert(num_heads % 8 == 0, "Invalid head")
                                            )

                                            def reduce_and_store(
                                                num_iters_c,
                                                kv_offset_arg,
                                                scale_kv_arg,
                                                umma_stage_idx_arg,
                                            ):
                                                accum = _builder_assign(
                                                    "accum",
                                                    T.alloc_local((num_heads,), "float32"),
                                                    locals().get("accum", _BUILDER_MISSING),
                                                )
                                                _builder_emit(
                                                    T.static_assert(
                                                        num_heads == 32 or num_heads == 64,
                                                        "Unsupported TMEM load size",
                                                    )
                                                )
                                                # relu(x) = (x + |x|) * 0.5; the epilogue accumulates 2*relu and
                                                # folds the 0.5 into the output scale so the ReLU runs on the
                                                # FMA pipe through the packed f32x2 add with abs source
                                                # modifiers instead of scalar FMNMX on the ALU pipe (the SM100
                                                # bottleneck for this kernel, per the CuTeDSL baseline).
                                                scale_kv_half = _builder_bind(
                                                    "scale_kv_half",
                                                    fmul_rn_noftz(scale_kv_arg, T.float32(0.5)),
                                                    None,
                                                )
                                                with T.unroll(0, num_iters_c) as q_inner_i:
                                                    IRBuilder.name("q_inner_i", q_inner_i)
                                                    tmem_addr = _builder_scalar(
                                                        "tmem_addr",
                                                        (
                                                            tmem_start_base
                                                            + umma_stage_idx_arg * T.uint32(umma_n)
                                                            + T.uint32(q_inner_i * num_heads)
                                                        ),
                                                        "uint32",
                                                    )
                                                    if num_heads == 32:
                                                        _builder_emit(
                                                            T.ptx[
                                                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                                            ](
                                                                accum[0],
                                                                accum[1],
                                                                accum[2],
                                                                accum[3],
                                                                accum[4],
                                                                accum[5],
                                                                accum[6],
                                                                accum[7],
                                                                accum[8],
                                                                accum[9],
                                                                accum[10],
                                                                accum[11],
                                                                accum[12],
                                                                accum[13],
                                                                accum[14],
                                                                accum[15],
                                                                accum[16],
                                                                accum[17],
                                                                accum[18],
                                                                accum[19],
                                                                accum[20],
                                                                accum[21],
                                                                accum[22],
                                                                accum[23],
                                                                accum[24],
                                                                accum[25],
                                                                accum[26],
                                                                accum[27],
                                                                accum[28],
                                                                accum[29],
                                                                accum[30],
                                                                accum[31],
                                                                T.uint32(tmem_addr),
                                                            )
                                                        )
                                                    if num_heads == 64:
                                                        _builder_emit(
                                                            T.ptx[
                                                                "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                                                            ](
                                                                accum[0],
                                                                accum[1],
                                                                accum[2],
                                                                accum[3],
                                                                accum[4],
                                                                accum[5],
                                                                accum[6],
                                                                accum[7],
                                                                accum[8],
                                                                accum[9],
                                                                accum[10],
                                                                accum[11],
                                                                accum[12],
                                                                accum[13],
                                                                accum[14],
                                                                accum[15],
                                                                accum[16],
                                                                accum[17],
                                                                accum[18],
                                                                accum[19],
                                                                accum[20],
                                                                accum[21],
                                                                accum[22],
                                                                accum[23],
                                                                accum[24],
                                                                accum[25],
                                                                accum[26],
                                                                accum[27],
                                                                accum[28],
                                                                accum[29],
                                                                accum[30],
                                                                accum[31],
                                                                accum[32],
                                                                accum[33],
                                                                accum[34],
                                                                accum[35],
                                                                accum[36],
                                                                accum[37],
                                                                accum[38],
                                                                accum[39],
                                                                accum[40],
                                                                accum[41],
                                                                accum[42],
                                                                accum[43],
                                                                accum[44],
                                                                accum[45],
                                                                accum[46],
                                                                accum[47],
                                                                accum[48],
                                                                accum[49],
                                                                accum[50],
                                                                accum[51],
                                                                accum[52],
                                                                accum[53],
                                                                accum[54],
                                                                accum[55],
                                                                accum[56],
                                                                accum[57],
                                                                accum[58],
                                                                accum[59],
                                                                accum[60],
                                                                accum[61],
                                                                accum[62],
                                                                accum[63],
                                                                T.uint32(tmem_addr),
                                                            )
                                                        )
                                                    _builder_emit(
                                                        T.ptx.tcgen05.wait__ld.sync.aligned()
                                                    )
                                                    with T.If(q_inner_i == num_iters_c - 1):
                                                        with T.Then():
                                                            # Release the UMMA stage right after the last TMEM load
                                                            # so the next MMA can start while the FMA chain and the
                                                            # store are still running (CuTeDSL order).
                                                            _builder_emit(
                                                                T.ptx.tcgen05.fence__before_thread_sync()
                                                            )
                                                            # One arrive per math warp in the group (lane 0): the
                                                            # tcgen05.ld above is warp-collective, so all lanes' TMEM
                                                            # reads are complete once lane 0's wait.ld returns.
                                                            with T.If(lane_idx == 0):
                                                                with T.Then():
                                                                    _builder_emit(
                                                                        mbarrier_arrive(
                                                                            smem_barriers.ptr_to(
                                                                                [
                                                                                    empty_umma_barrier_base
                                                                                    + math_wg_u32
                                                                                    * T.uint32(
                                                                                        num_umma_stages
                                                                                    )
                                                                                    + umma_stage_idx_arg
                                                                                ]
                                                                            )
                                                                        )
                                                                    )
                                                    sum_0 = _builder_scalar(
                                                        "sum_0",
                                                        T.cuda.make_float2(
                                                            T.float32(0), T.float32(0)
                                                        ),
                                                        "uint64",
                                                    )
                                                    sum_1 = _builder_scalar(
                                                        "sum_1",
                                                        T.cuda.make_float2(
                                                            T.float32(0), T.float32(0)
                                                        ),
                                                        "uint64",
                                                    )
                                                    with T.unroll(
                                                        0, num_heads // 4
                                                    ) as head_j_group:
                                                        IRBuilder.name("head_j_group", head_j_group)
                                                        head_j = _builder_assign(
                                                            "head_j",
                                                            head_j_group * 4,
                                                            locals().get(
                                                                "head_j", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        sum_0 = _builder_assign(
                                                            "sum_0",
                                                            relu2_fma_f32x2(
                                                                T.cuda.make_float2(
                                                                    accum[head_j], accum[head_j + 1]
                                                                ),
                                                                T.cuda.make_float2(
                                                                    cached_weights[
                                                                        q_inner_i, head_j
                                                                    ],
                                                                    cached_weights[
                                                                        q_inner_i, head_j + 1
                                                                    ],
                                                                ),
                                                                sum_0,
                                                            ),
                                                            locals().get("sum_0", _BUILDER_MISSING),
                                                        )
                                                        sum_1 = _builder_assign(
                                                            "sum_1",
                                                            relu2_fma_f32x2(
                                                                T.cuda.make_float2(
                                                                    accum[head_j + 2],
                                                                    accum[head_j + 3],
                                                                ),
                                                                T.cuda.make_float2(
                                                                    cached_weights[
                                                                        q_inner_i, head_j + 2
                                                                    ],
                                                                    cached_weights[
                                                                        q_inner_i, head_j + 3
                                                                    ],
                                                                ),
                                                                sum_1,
                                                            ),
                                                            locals().get("sum_1", _BUILDER_MISSING),
                                                        )
                                                    sum_v = _builder_bind(
                                                        "sum_v", fadd2_rn_noftz(sum_0, sum_1), None
                                                    )
                                                    result_f32 = _builder_bind(
                                                        "result_f32",
                                                        fmul_rn_noftz(
                                                            scale_kv_half,
                                                            fadd_rn_noftz(
                                                                T.cuda.float2_x(sum_v),
                                                                T.cuda.float2_y(sum_v),
                                                            ),
                                                        ),
                                                        None,
                                                    )
                                                    result = _builder_scalar(
                                                        "result",
                                                        T.cast(result_f32, logits_tir_dtype),
                                                        logits_tir_dtype,
                                                    )
                                                    logits_offset = _builder_scalar(
                                                        "logits_offset",
                                                        (
                                                            T.cast(kv_offset_arg, "uint64")
                                                            + T.cast(q_inner_i, "uint64")
                                                            * T.cast(logits_stride, "uint64")
                                                            + T.cast(math_thread_idx, "uint64")
                                                        ),
                                                        "uint64",
                                                    )
                                                    if config.logits_dtype == "float32":
                                                        _builder_emit(
                                                            T.ptx.st.global_.f32(
                                                                logits_flat.ptr_to([logits_offset]),
                                                                result,
                                                            )
                                                        )
                                                    else:
                                                        _builder_emit(
                                                            T.ptx.st.global_.b16(
                                                                logits_flat.ptr_to([logits_offset]),
                                                                result,
                                                            )
                                                        )

                                            with T.While(fetched_next_task):
                                                with T.If(q_atom_idx != next_q_atom_idx):
                                                    with T.Then():
                                                        with T.If(q_iter_idx > T.uint32(0)):
                                                            with T.Then():
                                                                # One arrive per math warp (lane 0); by the next Q change
                                                                # every lane's weight reads are long since consumed.
                                                                with T.If(lane_idx == 0):
                                                                    with T.Then():
                                                                        _builder_emit(
                                                                            mbarrier_arrive(
                                                                                smem_barriers.ptr_to(
                                                                                    [
                                                                                        empty_q_barrier_base
                                                                                        + (
                                                                                            q_iter_idx
                                                                                            - T.uint32(
                                                                                                1
                                                                                            )
                                                                                        )
                                                                                        % T.uint32(
                                                                                            num_q_stages
                                                                                        )
                                                                                    ]
                                                                                )
                                                                            )
                                                                        )
                                                        _builder_emit(get_q_pipeline(q_iter_idx))
                                                        q_stage_idx = _builder_assign(
                                                            "q_stage_idx",
                                                            fetch_result[0],
                                                            locals().get(
                                                                "q_stage_idx", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        q_phase = _builder_assign(
                                                            "q_phase",
                                                            fetch_result[1],
                                                            locals().get(
                                                                "q_phase", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        q_iter_idx = _builder_assign(
                                                            "q_iter_idx",
                                                            q_iter_idx + T.uint32(1),
                                                            locals().get(
                                                                "q_iter_idx", _BUILDER_MISSING
                                                            ),
                                                        )
                                                        _builder_emit(
                                                            mbarrier_wait_phase(
                                                                smem_barriers.ptr_to(
                                                                    [
                                                                        full_q_barrier_base
                                                                        + q_stage_idx
                                                                    ]
                                                                ),
                                                                q_phase,
                                                            )
                                                        )
                                                        with T.unroll(0, next_n_atom) as weight_i:
                                                            IRBuilder.name("weight_i", weight_i)
                                                            with T.unroll(
                                                                0, num_heads // 4
                                                            ) as weight_j:
                                                                IRBuilder.name("weight_j", weight_j)
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
                                                                            weight_i, weight_col
                                                                        ],
                                                                        cached_weights[
                                                                            weight_i, weight_col + 1
                                                                        ],
                                                                        cached_weights[
                                                                            weight_i, weight_col + 2
                                                                        ],
                                                                        cached_weights[
                                                                            weight_i, weight_col + 3
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
                                                        if config.varlen:
                                                            _builder_emit(
                                                                load_atom_advance(
                                                                    next_q_atom_idx, batch_size
                                                                )
                                                            )
                                                            is_paired_atom = _builder_assign(
                                                                "is_paired_atom",
                                                                atom_advance_result[0]
                                                                == T.uint32(2),
                                                                locals().get(
                                                                    "is_paired_atom",
                                                                    _BUILDER_MISSING,
                                                                ),
                                                            )
                                                q_atom_idx = _builder_assign(
                                                    "q_atom_idx",
                                                    next_q_atom_idx,
                                                    locals().get("q_atom_idx", _BUILDER_MISSING),
                                                )
                                                kv_idx = _builder_scalar(
                                                    "kv_idx", next_kv_idx, "uint32"
                                                )
                                                kv_offset = _builder_scalar(
                                                    "kv_offset",
                                                    T.cast(
                                                        atom_to_token_idx_expr(q_atom_idx), "uint64"
                                                    )
                                                    * T.cast(logits_stride, "uint64")
                                                    + T.cast(
                                                        (kv_idx + math_wg_u32) * T.uint32(umma_m),
                                                        "uint64",
                                                    ),
                                                    "uint64",
                                                )
                                                _builder_emit(get_kv_pipeline(kv_iter_idx))
                                                kv_stage_idx = _builder_scalar(
                                                    "kv_stage_idx", fetch_result[2], "uint32"
                                                )
                                                kv_phase = _builder_scalar(
                                                    "kv_phase", fetch_result[3], "uint32"
                                                )
                                                kv_iter_idx = _builder_assign(
                                                    "kv_iter_idx",
                                                    kv_iter_idx + T.uint32(1),
                                                    locals().get("kv_iter_idx", _BUILDER_MISSING),
                                                )
                                                _builder_emit(
                                                    mbarrier_wait_phase(
                                                        smem_barriers.ptr_to(
                                                            [
                                                                full_kv_barrier_base
                                                                + math_wg_u32
                                                                * T.uint32(num_kv_stages)
                                                                + kv_stage_idx
                                                            ]
                                                        ),
                                                        kv_phase,
                                                    )
                                                )
                                                scale_kv = _builder_alloc_scalar(
                                                    "scale_kv", "float32"
                                                )
                                                _builder_emit(
                                                    T.ptx.ld.shared.f32(
                                                        scale_kv,
                                                        smem_kv_scales.ptr_to(
                                                            [
                                                                math_warpgroup_idx,
                                                                kv_stage_idx,
                                                                math_thread_idx,
                                                            ]
                                                        ),
                                                    )
                                                )
                                                umma_stage_idx = _builder_scalar(
                                                    "umma_stage_idx",
                                                    umma_iter_idx % T.uint32(num_umma_stages),
                                                    "uint32",
                                                )
                                                umma_phase = _builder_scalar(
                                                    "umma_phase",
                                                    (umma_iter_idx // T.uint32(num_umma_stages))
                                                    & T.uint32(1),
                                                    "uint32",
                                                )
                                                umma_iter_idx = _builder_assign(
                                                    "umma_iter_idx",
                                                    umma_iter_idx + T.uint32(1),
                                                    locals().get("umma_iter_idx", _BUILDER_MISSING),
                                                )
                                                _builder_emit(
                                                    mbarrier_wait_phase(
                                                        smem_barriers.ptr_to(
                                                            [
                                                                full_umma_barrier_base
                                                                + math_wg_u32
                                                                * T.uint32(num_umma_stages)
                                                                + umma_stage_idx
                                                            ]
                                                        ),
                                                        umma_phase,
                                                    )
                                                )
                                                _builder_emit(
                                                    T.ptx.tcgen05.fence__after_thread_sync()
                                                )
                                                # One arrive per math warp in the group (lane 0); all lanes read
                                                # their scales before the UMMA wait above, so the SMEM reads are
                                                # complete by the time lane 0 arrives.
                                                with T.If(lane_idx == 0):
                                                    with T.Then():
                                                        _builder_emit(
                                                            mbarrier_arrive(
                                                                smem_barriers.ptr_to(
                                                                    [
                                                                        empty_kv_barrier_base
                                                                        + math_wg_u32
                                                                        * T.uint32(num_kv_stages)
                                                                        + kv_stage_idx
                                                                    ]
                                                                )
                                                            )
                                                        )
                                                if config.varlen:
                                                    with T.If(is_paired_atom):
                                                        with T.Then():
                                                            _builder_emit(
                                                                reduce_and_store(
                                                                    next_n_atom,
                                                                    kv_offset,
                                                                    scale_kv,
                                                                    umma_stage_idx,
                                                                )
                                                            )
                                                        with T.Else():
                                                            _builder_emit(
                                                                reduce_and_store(
                                                                    1,
                                                                    kv_offset,
                                                                    scale_kv,
                                                                    umma_stage_idx,
                                                                )
                                                            )
                                                elif k_pad_odd_n:
                                                    with T.If(
                                                        q_atom_idx % T.uint32(num_next_n_atoms)
                                                        == T.uint32(num_next_n_atoms - 1)
                                                    ):
                                                        with T.Then():
                                                            _builder_emit(
                                                                reduce_and_store(
                                                                    1,
                                                                    kv_offset,
                                                                    scale_kv,
                                                                    umma_stage_idx,
                                                                )
                                                            )
                                                        with T.Else():
                                                            _builder_emit(
                                                                reduce_and_store(
                                                                    next_n_atom,
                                                                    kv_offset,
                                                                    scale_kv,
                                                                    umma_stage_idx,
                                                                )
                                                            )
                                                else:
                                                    _builder_emit(
                                                        reduce_and_store(
                                                            next_n_atom,
                                                            kv_offset,
                                                            scale_kv,
                                                            umma_stage_idx,
                                                        )
                                                    )
                                                next_q_atom_idx = _builder_assign(
                                                    "next_q_atom_idx",
                                                    current_q_atom_idx,
                                                    locals().get(
                                                        "next_q_atom_idx", _BUILDER_MISSING
                                                    ),
                                                )
                                                next_kv_idx = _builder_assign(
                                                    "next_kv_idx",
                                                    current_kv_idx,
                                                    locals().get("next_kv_idx", _BUILDER_MISSING),
                                                )
                                                next_num_kv = _builder_assign(
                                                    "next_num_kv",
                                                    current_num_kv,
                                                    locals().get("next_num_kv", _BUILDER_MISSING),
                                                )
                                                _builder_emit(
                                                    fetch_next_task(
                                                        current_q_atom_idx,
                                                        current_kv_idx,
                                                        current_num_kv,
                                                        end_q_atom_idx,
                                                        end_kv_idx,
                                                    )
                                                )
                                                next_q_atom_idx = _builder_assign(
                                                    "next_q_atom_idx",
                                                    scheduler_result[0],
                                                    locals().get(
                                                        "next_q_atom_idx", _BUILDER_MISSING
                                                    ),
                                                )
                                                next_kv_idx = _builder_assign(
                                                    "next_kv_idx",
                                                    scheduler_result[1],
                                                    locals().get("next_kv_idx", _BUILDER_MISSING),
                                                )
                                                next_num_kv = _builder_assign(
                                                    "next_num_kv",
                                                    scheduler_result[2],
                                                    locals().get("next_num_kv", _BUILDER_MISSING),
                                                )
                                                fetched_next_task = _builder_assign(
                                                    "fetched_next_task",
                                                    scheduler_result[3] != T.uint32(0),
                                                    locals().get(
                                                        "fetched_next_task", _BUILDER_MISSING
                                                    ),
                                                )
                                                current_q_atom_idx = _builder_assign(
                                                    "current_q_atom_idx",
                                                    scheduler_result[4],
                                                    locals().get(
                                                        "current_q_atom_idx", _BUILDER_MISSING
                                                    ),
                                                )
                                                current_kv_idx = _builder_assign(
                                                    "current_kv_idx",
                                                    scheduler_result[5],
                                                    locals().get(
                                                        "current_kv_idx", _BUILDER_MISSING
                                                    ),
                                                )
                                                current_num_kv = _builder_assign(
                                                    "current_num_kv",
                                                    scheduler_result[6],
                                                    locals().get(
                                                        "current_num_kv", _BUILDER_MISSING
                                                    ),
                                                )
                                            _builder_emit(T.ptx.griddepcontrol.launch_dependents())
                                            _builder_emit(
                                                T.ptx.bar.sync(8, T.uint32(num_math_threads))
                                            )
                                            with T.If(warp_idx == 0):
                                                with T.Then():
                                                    _builder_emit(
                                                        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                                                            T.uint32(0), T.uint32(num_tmem_cols)
                                                        )
                                                    )

    return builder.get().with_attr(
        "tirx.kernel_launch_params",
        [
            "blockIdx.x",
            "threadIdx.x",
            "tirx.use_programtic_dependent_launch",
            "tirx.use_dyn_shared_memory",
        ],
    )


def _compile_tirx_paged_mqa_for_config(
    *,
    batch_size: int,
    next_n: int,
    max_num_pages: int,
    num_pages: int,
    num_heads: int,
    head_dim: int,
    page_size: int,
    logits_dtype: str,
    num_sms: int,
    context_lens_2d: bool,
    varlen: bool,
    indices_pair_stride: int,
) -> Any:
    import tvm

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100f"})
    kernel = get_kernel(
        batch_size=batch_size,
        next_n=next_n,
        max_num_pages=max_num_pages,
        num_pages=num_pages,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
        logits_dtype=logits_dtype,
        num_sms=num_sms,
        context_lens_2d=context_lens_2d,
        varlen=varlen,
        indices_pair_stride=indices_pair_stride,
    )
    with target:
        mod = tvm.IRModule({"main": kernel})
        return tvm.compile(mod, target=target, tir_pipeline="tirx")


_compile_tirx_paged_mqa_for_config = cache(_compile_tirx_paged_mqa_for_config)


def _compile_tirx_paged_mqa_kwargs(config: PagedMQALogitsFP8Config) -> dict[str, Any]:
    return {
        "batch_size": config.batch_size,
        "next_n": config.next_n,
        "max_num_pages": config.max_num_pages,
        "num_pages": config.num_pages,
        "num_heads": config.num_heads,
        "head_dim": config.head_dim,
        "page_size": config.page_size,
        "logits_dtype": config.logits_dtype,
        "num_sms": config.num_sms,
        "context_lens_2d": config.context_lens_2d,
        "varlen": config.varlen,
        "indices_pair_stride": config.indices_pair_stride,
    }


def _compile_tirx_paged_mqa_key(config: PagedMQALogitsFP8Config) -> tuple[tuple[str, Any], ...]:
    return tuple(_compile_tirx_paged_mqa_kwargs(config).items())


def _compile_tirx_paged_mqa(config: PagedMQALogitsFP8Config) -> Any:

    compile_kwargs = _compile_tirx_paged_mqa_kwargs(config)
    return _compile_tirx_paged_mqa_for_config(**compile_kwargs)


def _run_deepgemm_paged_mqa(data: dict[str, Any], *, clean_logits: bool = False) -> torch.Tensor:
    config: PagedMQALogitsFP8Config = data["config"]
    return data["deep_gemm"].fp8_fp4_paged_mqa_logits(
        q=(data["q"], None),
        kv_cache=data["fused_kv_cache"],
        weights=data["weights"],
        context_lens=data["context_lens"],
        block_table=data["block_table"],
        schedule_meta=data["schedule_meta"],
        max_context_len=config.max_context_len,
        clean_logits=clean_logits,
        logits_dtype=_torch_logits_dtype(config.logits_dtype),
        indices=data["indices"],
    )


def _sglang_cutedsl_available() -> bool:
    return find_spec("sglang") is not None and find_spec("cutlass") is not None


def _make_sglang_cutedsl_runner(data: dict[str, Any]) -> Any:
    config: PagedMQALogitsFP8Config = data["config"]
    if config.context_pattern == "random_2d" and config.next_n > 1:
        raise ValueError(
            "SGLang CuTeDSL requires causal context lengths when next_n > 1; "
            "use context_pattern='sglang_fixed' or 'sglang_ragged'"
        )

    from sglang.jit_kernel.dsa.cutedsl_paged_mqa_logits import (
        CuteDSLPagedMQALogitsRunner,
        pick_dsl_expand,
    )

    expand_factor, atom = pick_dsl_expand(
        config.next_n,
        batch_size=config.batch_size,
        max_ctx=config.max_context_len,
        num_sms=config.num_sms,
        num_heads=config.num_heads,
    )
    expanded_batch = config.batch_size * expand_factor
    q = data["q"].reshape(expanded_batch, atom, config.num_heads, config.head_dim)
    context_lens = data["context_lens"][:, -1].contiguous()
    block_table = data["block_table"]
    if expand_factor > 1:
        context_lens = context_lens.repeat_interleave(expand_factor)
        block_table = block_table.repeat_interleave(expand_factor, dim=0)
    block_table = block_table.contiguous()
    schedule_meta = data["deep_gemm"].get_paged_mqa_logits_metadata(
        context_lens.unsqueeze(-1), config.page_size, config.num_sms
    )
    output_dtype = _torch_logits_dtype(config.logits_dtype)

    def _run():
        return CuteDSLPagedMQALogitsRunner.forward(
            q,
            data["fused_kv_cache"],
            data["weights"],
            context_lens,
            block_table,
            schedule_meta,
            config.max_context_len,
            epi_dtype=torch.float32,
            acc_dtype=torch.float32,
            output_dtype=output_dtype,
        )

    return _run


def _allocate_logits(config: PagedMQALogitsFP8Config) -> torch.Tensor:
    return torch.full(
        (config.batch_size * config.next_n, config.logits_stride),
        float("-inf"),
        device="cuda",
        dtype=_torch_logits_dtype(config.logits_dtype),
    )


def _encode_tma_3d_desc(
    *,
    encode_tensormap: Any,
    tensor: torch.Tensor,
    gmem_inner_dim: int,
    gmem_mid_dim: int,
    gmem_outer_dim: int,
    smem_inner_dim: int,
    smem_mid_dim: int,
    smem_outer_dim: int,
    gmem_mid_stride: int,
    gmem_outer_stride: int,
    swizzle_mode: int,
    tensor_dtype: Any | None = None,
) -> Any:
    from tirx_kernels.deepgemm._sm100_fp8_fp4_mega_moe import spec as mega_moe

    elem_size = int(tensor.element_size())
    if swizzle_mode != 0:
        smem_inner_dim = swizzle_mode // elem_size
    desc = mega_moe._AlignedTensorMap()
    encode_tensormap(
        desc.ptr,
        mega_moe._torch_dtype_to_tvm_dtype(tensor) if tensor_dtype is None else tensor_dtype,
        3,
        ctypes.c_void_p(int(tensor.data_ptr())),
        int(gmem_inner_dim),
        int(gmem_mid_dim),
        int(gmem_outer_dim),
        int(gmem_mid_stride * elem_size),
        int(gmem_outer_stride * elem_size),
        int(smem_inner_dim),
        int(smem_mid_dim),
        int(smem_outer_dim),
        1,
        1,
        1,
        0,
        mega_moe._tensor_map_swizzle_from_mode(swizzle_mode),
        3,
        0,
    )
    return desc


def _build_tirx_tensor_maps(data: dict[str, Any]) -> dict[str, Any]:
    import tvm
    from tirx_kernels.deepgemm._sm100_fp8_fp4_mega_moe.spec import _encode_tma_2d_desc

    config: PagedMQALogitsFP8Config = data["config"]
    q_fp8 = data["q"]
    fused = data["fused_kv_cache"]
    weights = data["weights"]
    encode_tensormap = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    kv_flat = fused.view(torch.uint8).view(
        config.num_pages, config.page_size * (config.head_dim + 4)
    )
    kv_fp8 = (
        kv_flat[:, : config.page_size * config.head_dim]
        .view(torch.float8_e4m3fn)
        .reshape(config.num_pages, config.page_size, config.head_dim)
    )
    kv_scales = kv_flat[:, config.page_size * config.head_dim :].view(torch.float32)

    return {
        "tensor_map_q": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=q_fp8,
            gmem_inner_dim=config.head_dim,
            gmem_outer_dim=config.batch_size * config.next_n * config.num_heads,
            smem_inner_dim=config.head_dim,
            smem_outer_dim=(2 if (config.varlen or config.next_n >= 2) else 1) * config.num_heads,
            gmem_outer_stride=int(q_fp8.stride(2)),
            swizzle_mode=config.head_dim,
            tensor_dtype="float8_e4m3fn",
        ),
        "tensor_map_kv": _encode_tma_3d_desc(
            encode_tensormap=encode_tensormap,
            tensor=kv_fp8,
            gmem_inner_dim=config.head_dim,
            gmem_mid_dim=config.page_size,
            gmem_outer_dim=config.num_pages,
            smem_inner_dim=config.head_dim,
            smem_mid_dim=config.page_size,
            smem_outer_dim=1,
            gmem_mid_stride=int(kv_fp8.stride(1)),
            gmem_outer_stride=int(kv_fp8.stride(0)),
            swizzle_mode=config.head_dim,
            tensor_dtype="float8_e4m3fn",
        ),
        "tensor_map_kv_scales": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=kv_scales,
            gmem_inner_dim=config.page_size,
            gmem_outer_dim=config.num_pages,
            smem_inner_dim=config.page_size,
            smem_outer_dim=1,
            gmem_outer_stride=int(kv_scales.stride(0)),
            swizzle_mode=0,
        ),
        "tensor_map_weights": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=weights,
            gmem_inner_dim=config.num_heads,
            gmem_outer_dim=config.batch_size * config.next_n,
            smem_inner_dim=config.num_heads,
            smem_outer_dim=2 if (config.varlen or config.next_n >= 2) else 1,
            gmem_outer_stride=int(weights.stride(0)),
            swizzle_mode=0,
        ),
    }


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
    config: PagedMQALogitsFP8Config = data["config"]
    if logits is None:
        logits = _allocate_logits(config)
    if executable is None:
        executable = _compile_tirx_paged_mqa(config)
    return {
        "executable": executable,
        "logits": logits,
        "tensor_maps": _build_tirx_tensor_maps(data),
    }


def _run_tirx_invocation(data: dict[str, Any], invocation: dict[str, Any]) -> torch.Tensor:
    config: PagedMQALogitsFP8Config = data["config"]
    executable = invocation["executable"]
    tensor_maps = invocation["tensor_maps"]
    logits = invocation["logits"]
    indices = data["indices"]
    if indices is None:
        indices = torch.empty(
            (config.batch_size,), dtype=torch.int32, device=data["context_lens"].device
        )
    _prepare_global_barrier(executable)
    executable.mod(
        config.batch_size,
        config.logits_stride,
        data["block_table"].stride(0),
        data["context_lens"],
        logits,
        data["block_table"],
        indices,
        data["tirx_schedule_meta"],
        tensor_maps["tensor_map_q"].ptr,
        tensor_maps["tensor_map_kv"].ptr,
        tensor_maps["tensor_map_kv_scales"].ptr,
        tensor_maps["tensor_map_weights"].ptr,
    )
    return logits


def _launch_tirx_paged_mqa(
    data: dict[str, Any], logits: torch.Tensor | None = None
) -> torch.Tensor:
    return _run_tirx_invocation(data, _prepare_tirx_invocation(data, logits))


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x[:, : y.shape[1]].double()
    y = y.double()
    mask = y == float("-inf")
    x = x.masked_fill(mask, 0)
    y = y.masked_fill(mask, 0)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        return float("inf")
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return float((1 - sim).item())


def _calc_valid_diff(x: torch.Tensor, y: torch.Tensor, context_lens: torch.Tensor) -> float:
    expected_rows = context_lens.numel()
    if x.ndim != 2 or y.ndim != 2:
        raise AssertionError(f"expected rank-2 logits, got {x.shape=} and {y.shape=}")
    if x.shape[0] != expected_rows or y.shape[0] != expected_rows:
        raise AssertionError(
            f"logits row mismatch: expected {expected_rows}, got {x.shape[0]} and {y.shape[0]}"
        )
    required_width = int(context_lens.max().item())
    if x.shape[1] < required_width or y.shape[1] < required_width:
        raise AssertionError(
            f"logits width must cover {required_width}, got {x.shape[1]} and {y.shape[1]}"
        )

    width = min(x.shape[1], y.shape[1])
    valid = torch.arange(width, device=context_lens.device)[None, :] < context_lens.reshape(-1, 1)
    x = x[:, :width].double().masked_fill(~valid, 0)
    y = y[:, :width].double().masked_fill(~valid, 0)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        return float("inf")
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return float((1 - sim).item())


def _assert_correct(data: dict[str, Any], logits: torch.Tensor, *, name: str) -> float:
    reference = data["reference"]
    diff = _calc_diff(logits, reference)
    if diff >= _TEST_DIFF_THRESHOLD:
        raise AssertionError(f"{name} simulated diff {diff:.6g} >= {_TEST_DIFF_THRESHOLD}")
    return diff


def _assert_valid_correct(
    data: dict[str, Any], logits: torch.Tensor, reference: torch.Tensor, *, name: str
) -> float:
    diff = _calc_valid_diff(logits, reference, data["context_lens"])
    if diff >= _TEST_DIFF_THRESHOLD:
        raise AssertionError(f"{name} valid-logits diff {diff:.6g} >= {_TEST_DIFF_THRESHOLD}")
    return diff


def run_test(**kwargs: Any) -> None:
    data = prepare_data(**kwargs)
    config: PagedMQALogitsFP8Config = data["config"]
    deepgemm_logits = _run_deepgemm_paged_mqa(data, clean_logits=False)
    deepgemm_diff = _assert_correct(data, deepgemm_logits, name="DeepGEMM")
    tirx_logits = _launch_tirx_paged_mqa(data)
    torch.cuda.synchronize()
    tirx_diff = _assert_correct(data, tirx_logits, name="TIRx")
    if tirx_diff > max(deepgemm_diff, _TEST_DIFF_THRESHOLD):
        raise AssertionError(
            f"TIRx diff {tirx_diff:.6g} is worse than DeepGEMM diff {deepgemm_diff:.6g}"
        )
    if config.context_pattern.startswith("sglang_") and _sglang_cutedsl_available():
        cutedsl_runner = _make_sglang_cutedsl_runner(data)
        cutedsl_logits = cutedsl_runner()
        torch.cuda.synchronize()
        _assert_correct(data, cutedsl_logits, name="SGLang CuTeDSL")


def prepare_bench(**kwargs: Any):
    """Compile the paged MQA executable without allocating CUDA data."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    config = _make_config(**kwargs)
    executable = _compile_tirx_paged_mqa(config)
    return prepared_gpu_benchmark(run_gpu, {"config": dict(kwargs), "executable": executable})


def run_gpu(prepared, **kwargs: Any) -> dict[str, Any]:
    kwargs = {**prepared["config"], **kwargs}
    from tirx_kernels.runner import bench

    # Tiny (~8-11µs) paged kernel: event timing is launch-jitter-noisy (sporadic
    # 10-13% ratio spread) and ~2x inflated by launch overhead. timer=None inherits the
    # global default (proton) -> pure per-kernel GPU time (~4.5µs, verified stable).
    timer = kwargs.pop("timer", None)
    # warmup/repeat: no hardcoded default here; pass through (None = defer to the
    # timer's own default; the graph timers ignore them anyway). Overridable via the
    # suite/CLI when a specific case needs a longer rep.
    warmup = kwargs.pop("warmup", None)
    repeat = kwargs.pop("repeat", None)
    _rounds = kwargs.pop("rounds", 1)
    _cooldown_s = kwargs.pop("cooldown_s", 1.0)
    config_kwargs = dict(kwargs)
    config = _make_config(**config_kwargs)
    tirx_executable = prepared["executable"]

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    # The independent Python reference is intentionally omitted here: it iterates
    # page-by-page and is prohibitively slow for SGLang's 131K-context sweep.
    data = _prepare_data(config, compute_reference=False)
    invocation = _prepare_tirx_invocation(data, executable=tirx_executable)
    deepgemm_logits = _run_deepgemm_paged_mqa(data, clean_logits=False)
    tirx_logits = _run_tirx_invocation(data, invocation)
    torch.cuda.synchronize()
    max_diff = _assert_valid_correct(data, tirx_logits, deepgemm_logits, name="TIRx vs DeepGEMM")
    torch.cuda.empty_cache()

    def _deepgemm():
        return lambda: _run_deepgemm_paged_mqa(data, clean_logits=False)

    def _sglang_cutedsl():
        cutedsl_runner = _make_sglang_cutedsl_runner(data)
        cutedsl_logits = cutedsl_runner()
        torch.cuda.synchronize()
        _assert_valid_correct(
            data, cutedsl_logits, deepgemm_logits, name="SGLang CuTeDSL vs DeepGEMM"
        )
        return cutedsl_runner

    result = bench(
        {"tirx": lambda: _run_tirx_invocation(data, invocation)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=_rounds,
        cooldown_s=_cooldown_s,
        references={"deepgemm": _deepgemm, "sglang_cutedsl": _sglang_cutedsl},
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
    "DSA_INDEXER_LIKE_COVERAGE",
    "KERNEL_META",
    "SGLANG_BENCH_CONFIGS",
    "PagedMQALogitsFP8Config",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
