# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

import ctypes
import importlib
import sys
import types
from dataclasses import asdict, dataclass
from functools import cache
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K

_DEEP_GEMM_MODULE_NAME = "deep_gemm"
_SM100_SMEM_CAPACITY = 232448
_TEST_DIFF_THRESHOLD = 5e-6
_CONTEXT_PATTERNS = ("random_2d", "sglang_fixed", "sglang_ragged")
_COMPILE_CACHE_NAMESPACE = "deepgemm.paged_mqa_logits_fp8.compile"


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
    config = _make_config(**kwargs)

    num_heads = config.num_heads
    head_dim = config.head_dim
    page_size = config.page_size
    k_pad_odd_n = (not config.varlen) and (config.next_n % 2 == 1) and (config.next_n >= 3)
    next_n_atom = 2 if (config.varlen or config.next_n >= 2) else 1
    num_next_n_atoms = _align_up(config.next_n, next_n_atom) // next_n_atom
    num_q_stages = 3
    num_umma_stages = 1
    split_kv = config.split_kv
    umma_m = 128
    umma_k = 32
    umma_n = next_n_atom * num_heads
    num_math_warpgroups = split_kv // umma_m
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
    num_kv_stages = 3
    num_umma_barriers = num_math_warpgroups * num_umma_stages
    num_tmem_cols = next_n_atom * num_heads * num_math_warpgroups * num_umma_stages
    if num_tmem_cols > 512:
        raise ValueError("tensor memory columns exceed SM100 single-CTA limit")
    logits_tir_dtype = "float32" if config.logits_dtype == "float32" else "bfloat16"

    # Block-table L2 warm-up coverage (trace-time ints): the whole table, capped
    # at 512 lines (64 KB).
    num_block_table_bytes = config.batch_size * config.max_num_pages * 4
    num_prefetch_lines = min((num_block_table_bytes + 127) // 128, 512)

    TMA_G2S_2D = (
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
    )
    TMA_G2S_3D = (
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
    )
    MMA = "tcgen05.mma.cta_group::1.kind::f8f6f4"
    TC_LD = f"tcgen05.ld.sync.aligned.32x32b.x{num_heads}.b32"

    @K.kernel(warps=num_warps, arch="sm_100f", min_blocks_per_sm=1, grid=config.num_sms)
    def sm100_fp8_paged_mqa_logits(
        batch_size: K.u32,
        logits_stride: K.u32,
        block_table_stride: K.u32,
        context_lens_flat: K.gptr[K.i32],
        logits_flat: K.gptr[logits_tir_dtype],
        block_table_flat: K.gptr[K.i32],
        indices: K.gptr[K.i32],
        schedule_meta_flat: K.gptr[K.i32],
        tensor_map_q: K.TensorMap,
        tensor_map_kv: K.TensorMap,
        tensor_map_kv_scales: K.TensorMap,
        tensor_map_weights: K.TensorMap,
    ):
        cache_policy_evict_normal = K.uint64(1152921504606846976)
        sm_idx_u32 = K.Cast("uint32", K.cta_id())
        warp_idx = K.warp_id()
        warp_idx_u32 = K.Cast("uint32", warp_idx)
        warpgroup_idx = K.warpgroup_id([num_warps // 4])
        lane_idx = K.lane_id()
        # The original reads laneid through mov_sreg for its u32 uses and keeps
        # K.lane_id() for the `lane_idx == 0` guards. Two spellings, both kept.
        # A plain binding re-emits the sreg read at every use site (16 laneid
        # reads in the CUDA where the original has 5), but ptxas CSEs them back:
        # measured SASS byte-identical to the retired K.Bind on 5 specializations,
        # while materializing into a local scalar cost +16 instructions. So the
        # duplication is a source-level artefact only -- do not "fix" it.
        lane_idx_u32 = K.Cast("uint32", K.cuda.mov_sreg(32, "laneid"))

        with K.If(warp_idx == spec_warp_start), K.Then():
            K.ptx.prefetch.tensormap(K.address_of(tensor_map_q))
            K.ptx.prefetch.tensormap(K.address_of(tensor_map_kv))
            K.ptx.prefetch.tensormap(K.address_of(tensor_map_kv_scales))
            K.ptx.prefetch.tensormap(K.address_of(tensor_map_weights))

        # ---------------- SMEM ------------------------------------------
        # Every buffer is a RAW alloc (swizzle=None). The original declares all
        # of these without a layout: the 128 B swizzle lives in the TMA
        # descriptor (host side) and in the MMA descriptor's swizzle field, not
        # in the SMEM buffer's index map. Asking K for a swizzled tile here
        # would impose an index map the original does not have -- and would also
        # hit the 2-D/3-D-only restriction on smem_kv's 4-D shape.
        smem = K.smem_pool()
        smem_q = smem.alloc(
            (num_q_stages, next_n_atom * num_heads, head_dim), "float8_e4m3fn", align=smem_alignment
        )
        smem_kv = smem.alloc(
            (num_math_warpgroups, num_kv_stages, umma_m, head_dim),
            "float8_e4m3fn",
            align=smem_alignment,
        )
        smem_kv_scales = smem.alloc((num_math_warpgroups, num_kv_stages, umma_m), K.f32, align=16)
        smem_weights = smem.alloc((num_q_stages, next_n_atom, num_heads), K.f32, align=16)
        # Six typed families preserve the original allocation order.  Init is
        # still issued explicitly below because its per-warp interleaving is a
        # kernel-specific ordering boundary.
        full_q_barriers = K.TMABar(smem, num_q_stages)
        empty_q_barriers = K.MBarrier(smem, num_q_stages)
        full_kv_barriers = K.TMABar(smem, num_kv_stages * num_math_warpgroups)
        empty_kv_barriers = K.MBarrier(smem, num_kv_stages * num_math_warpgroups)
        full_umma_barriers = K.TCGen05Bar(smem, num_umma_barriers)
        empty_umma_barriers = K.MBarrier(smem, num_umma_barriers)
        tmem_ptr_in_smem = smem.alloc((1,), K.u32, align=4)
        # TMEM D is a fixed column map rooted at allocation base zero; keep it
        # as raw columns and addresses rather than a TIR tmem buffer.
        tmem_col = 0

        scheduler_result = K.alloc_local([7], "uint32")
        num_kv_result = K.local_scalar("uint32")
        atom_advance_result = K.local_scalar("uint32")

        # ---------------- trace-time helpers ----------------------------

        def atom_to_token_idx_expr(q_atom_idx):
            if config.varlen:
                return q_atom_idx
            if k_pad_odd_n:
                return q_atom_idx // K.uint32(num_next_n_atoms) * K.uint32(
                    config.next_n
                ) + q_atom_idx % K.uint32(num_next_n_atoms) * K.uint32(next_n_atom)
            return q_atom_idx * K.uint32(next_n_atom)

        def atom_to_block_table_row_expr(q_atom_idx):
            if config.varlen:
                return q_atom_idx
            return q_atom_idx // K.uint32(num_next_n_atoms)

        def should_refresh_num_kv_expr(q_atom_idx):
            if config.varlen:
                return K.bool(True)
            return q_atom_idx % K.uint32(num_next_n_atoms) == K.uint32(0)

        def exist_q_atom_idx_expr(q_atom_idx, end_q_atom_idx, end_kv_idx):
            return K.Or(
                q_atom_idx < end_q_atom_idx,
                K.And(q_atom_idx == end_q_atom_idx, K.uint32(0) < end_kv_idx),
            )

        def load_num_kv(q_atom_idx_arg, runtime_batch_size_arg):
            context_len = K.local_scalar("uint32")
            if config.varlen:
                context_idx = K.local_scalar("uint32", init=q_atom_idx_arg)
                with K.If(q_atom_idx_arg + K.uint32(1) < runtime_batch_size_arg), K.Then():
                    index_pair = K.alloc_local([2], "int32")
                    K.ptx.ld.global_.s32(
                        index_pair[0], indices.ptr_to([K.Cast("int32", q_atom_idx_arg)])
                    )
                    K.ptx.ld.global_.s32(
                        index_pair[1],
                        indices.ptr_to([K.Cast("int32", q_atom_idx_arg + K.uint32(1))]),
                    )
                    with K.If(index_pair[0] == index_pair[1]), K.Then():
                        K.assign(context_idx, q_atom_idx_arg + K.uint32(1))
                K.ptx.ld.global_.u32(
                    context_len, context_lens_flat.ptr_to([K.Cast("int32", context_idx)])
                )
            else:
                q_idx = q_atom_idx_arg // K.uint32(num_next_n_atoms)
                lens_idx = q_idx * K.uint32(config.next_n) + K.uint32(config.next_n - 1)
                K.ptx.ld.global_.u32(
                    context_len, context_lens_flat.ptr_to([K.Cast("int32", lens_idx)])
                )
            K.assign(num_kv_result, (context_len + K.uint32(umma_m - 1)) // K.uint32(umma_m))

        def load_atom_advance(q_atom_idx_arg, bound_arg):
            K.assign(atom_advance_result, K.uint32(1))
            if config.varlen:
                with K.If(q_atom_idx_arg + K.uint32(1) < bound_arg), K.Then():
                    index_pair = K.alloc_local([2], "int32")
                    K.ptx.ld.global_.s32(
                        index_pair[0], indices.ptr_to([K.Cast("int32", q_atom_idx_arg)])
                    )
                    K.ptx.ld.global_.s32(
                        index_pair[1],
                        indices.ptr_to([K.Cast("int32", q_atom_idx_arg + K.uint32(1))]),
                    )
                    with K.If(index_pair[0] == index_pair[1]), K.Then():
                        K.assign(atom_advance_result, K.uint32(2))

        # Epilogue arithmetic, transcribed from the PAGED original's own
        # helpers (relu2_fma_f32x2 / fadd2 / fadd / fmul). The non-paged
        # sibling spells the same maths as one flat `wrelu_reduce` over
        # ptx.mov.b64 pack/unpack pairs; copying that here produced a
        # structurally different instruction stream (98 packs where the
        # original has 32, and zero make_float2 where it has 98), which is
        # exactly the normalise-one-kernel-to-the-other error the bit-identity
        # standard exists to catch.
        def relu2_fma_f32x2(a, w, c):
            a_lo = K.local_scalar("float32")
            a_hi = K.local_scalar("float32")
            abs_lo = K.local_scalar("float32")
            abs_hi = K.local_scalar("float32")
            abs_pair = K.local_scalar("uint64")
            relu_pair = K.local_scalar("uint64")
            out = K.local_scalar("uint64")
            K.ptx.mov.b64(a_lo, a_hi, a)
            K.ptx.abs.f32(abs_lo, a_lo)
            K.ptx.abs.f32(abs_hi, a_hi)
            K.ptx.mov.b64(abs_pair, abs_lo, abs_hi)
            K.ptx.add.rn.f32x2(relu_pair, a, abs_pair)
            K.ptx.fma.rn.f32x2(out, relu_pair, w, c)
            return out

        def make_smem_desc(desc, smem_ptr):
            K.cuda.tcgen05.encode_matrix_descriptor(
                K.address_of(desc), smem_ptr, ldo=0, sdo=desc_sdo, swizzle=desc_swizzle
            )

        def issue_tma_q(stage_idx, tma_q_atom_idx):
            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                q_token_idx = atom_to_token_idx_expr(tma_q_atom_idx)
                K.ptx[TMA_G2S_2D](
                    smem_q.ptr_to([stage_idx, 0, 0]),
                    K.address_of(tensor_map_q),
                    K.int32(0),
                    K.Cast("int32", q_token_idx * K.uint32(num_heads)),
                    full_q_barriers.ptr_to([stage_idx]),
                    cache_policy_evict_normal,
                )
                K.ptx[TMA_G2S_2D](
                    smem_weights.ptr_to([stage_idx, 0, 0]),
                    K.address_of(tensor_map_weights),
                    K.int32(0),
                    K.Cast("int32", q_token_idx),
                    full_q_barriers.ptr_to([stage_idx]),
                    cache_policy_evict_normal,
                )
                full_q_barriers.arrive(
                    stage_idx, tx_count=smem_q_size_per_stage + smem_weight_size_per_stage
                )

        def fetch_next_task(cur_q_atom, cur_kv_idx, cur_num_kv, end_q_atom, end_kv):
            K.ptx.mov.b32(scheduler_result[0], cur_q_atom)
            K.ptx.mov.b32(scheduler_result[1], cur_kv_idx)
            K.ptx.mov.b32(scheduler_result[2], cur_num_kv)
            K.ptx.mov.b32(scheduler_result[4], cur_q_atom)
            K.ptx.mov.b32(scheduler_result[5], cur_kv_idx)
            K.ptx.mov.b32(scheduler_result[6], cur_num_kv)
            with K.If(K.And(cur_q_atom == end_q_atom, cur_kv_idx == end_kv)):
                with K.Then():
                    K.ptx.mov.b32(scheduler_result[3], K.uint32(0))
                with K.Else():
                    K.ptx.mov.b32(scheduler_result[5], cur_kv_idx + K.uint32(num_tiles_per_split))
                    with K.If(scheduler_result[5] >= cur_num_kv), K.Then():
                        K.ptx.mov.b32(scheduler_result[5], K.uint32(0))
                        load_atom_advance(cur_q_atom, end_q_atom)
                        K.ptx.mov.b32(scheduler_result[4], cur_q_atom + atom_advance_result)
                        with (
                            K.If(
                                K.And(
                                    should_refresh_num_kv_expr(scheduler_result[4]),
                                    exist_q_atom_idx_expr(scheduler_result[4], end_q_atom, end_kv),
                                )
                            ),
                            K.Then(),
                        ):
                            load_num_kv(scheduler_result[4], batch_size)
                            K.ptx.mov.b32(scheduler_result[6], num_kv_result)
                    K.ptx.mov.b32(scheduler_result[3], K.uint32(1))

        # ---------------- CTA-scope prologue ----------------------------
        # Early schedule-metadata load: issue the global loads before the
        # pipeline/barrier prologue so the ~200-cycle L2 latency overlaps setup.
        start_q_atom_idx = K.local_scalar("uint32")
        start_kv_tile_idx = K.local_scalar("uint32")
        end_q_atom_idx = K.local_scalar("uint32")
        end_kv_tile_idx = K.local_scalar("uint32")
        K.ptx.ld.global_.u32(
            start_q_atom_idx, schedule_meta_flat.ptr_to([K.Cast("int32", sm_idx_u32 * K.uint32(2))])
        )
        K.ptx.ld.global_.u32(
            start_kv_tile_idx,
            schedule_meta_flat.ptr_to([K.Cast("int32", sm_idx_u32 * K.uint32(2) + K.uint32(1))]),
        )
        K.ptx.ld.global_.u32(
            end_q_atom_idx,
            schedule_meta_flat.ptr_to([K.Cast("int32", (sm_idx_u32 + K.uint32(1)) * K.uint32(2))]),
        )
        K.ptx.ld.global_.u32(
            end_kv_tile_idx,
            schedule_meta_flat.ptr_to(
                [K.Cast("int32", (sm_idx_u32 + K.uint32(1)) * K.uint32(2) + K.uint32(1))]
            ),
        )
        start_kv_idx = K.local_scalar("uint32")
        end_kv_idx = K.local_scalar("uint32")
        K.assign(start_kv_idx, start_kv_tile_idx * K.uint32(num_tiles_per_split))
        K.assign(end_kv_idx, end_kv_tile_idx * K.uint32(num_tiles_per_split))
        # Clamp for zero-work CTAs (start == total q atoms); the value is stale
        # but never used, because has_work is false.
        load_num_kv(
            K.min(start_q_atom_idx, batch_size * K.uint32(num_next_n_atoms) - K.uint32(1)),
            batch_size,
        )
        start_num_kv = K.local_scalar("uint32", init=num_kv_result)

        # Warm the block table into L2 as early as possible. Race-safe: a stale
        # prefetched line is invalidated by any later producer write.
        with K.If(K.Or(warp_idx == tma_warp_0, warp_idx == tma_warp_1)), K.Then():
            for pf_i in range((num_prefetch_lines + 63) // 64):
                # A local, not a Python expression: the original declares this
                # `K.uint32` and the generated CUDA shows `line_idx_ptr[0]`.
                # Left as an expression it is re-substituted at every use, which
                # re-reads laneid each time (17 sreg reads vs the original's 5).
                line_idx = K.local_scalar(
                    "uint32",
                    init=(warp_idx_u32 - K.uint32(tma_warp_0)) * K.uint32(32)
                    + lane_idx_u32
                    + K.uint32(pf_i * 64),
                )
                with K.If(line_idx < K.uint32(num_prefetch_lines)), K.Then():
                    K.ptx.prefetch.global_.L2(
                        block_table_flat.ptr_to([K.Cast("int64", line_idx * K.uint32(32))])
                    )

        with K.If(warp_idx == tma_warp_0), K.Then():
            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                for init_i in range(num_q_stages):
                    K.ptx.mbarrier.init.shared.b64(full_q_barriers.ptr_to([init_i]), K.uint32(1))
                    K.ptx.mbarrier.init.shared.b64(empty_q_barriers.ptr_to([init_i]), K.uint32(8))
                K.ptx.fence.mbarrier_init.release.cluster()
        with K.If(warp_idx == tma_warp_1), K.Then():
            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                for init_i in range(num_kv_stages):
                    K.ptx.mbarrier.init.shared.b64(full_kv_barriers.ptr_to([init_i]), K.uint32(1))
                    K.ptx.mbarrier.init.shared.b64(empty_kv_barriers.ptr_to([init_i]), K.uint32(4))
                K.ptx.fence.mbarrier_init.release.cluster()
        with K.If(warp_idx == umma_warp_0), K.Then():
            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                for init_i in range(num_kv_stages):
                    K.ptx.mbarrier.init.shared.b64(
                        full_kv_barriers.ptr_to([num_kv_stages + init_i]), K.uint32(1)
                    )
                    K.ptx.mbarrier.init.shared.b64(
                        empty_kv_barriers.ptr_to([num_kv_stages + init_i]), K.uint32(4)
                    )
                K.ptx.fence.mbarrier_init.release.cluster()
        with K.If(warp_idx == umma_warp_0 + 1), K.Then():
            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                for init_i in range(num_umma_barriers):
                    K.ptx.mbarrier.init.shared.b64(full_umma_barriers.ptr_to([init_i]), K.uint32(1))
                    K.ptx.mbarrier.init.shared.b64(
                        empty_umma_barriers.ptr_to([init_i]), K.uint32(4)
                    )
                K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()
        K.ptx.griddepcontrol.wait()

        # ---------------- roles -----------------------------------------
        # The original writes this partition as a genuine if/elif chain, so
        sp = K.specialize()
        tma0 = sp.role("tma0", warps=[tma_warp_0], regs=num_specialized_registers)
        tma1 = sp.role("tma1", warps=[tma_warp_1], regs=num_specialized_registers)
        umma = sp.role("umma", warps=[umma_warp_0, umma_warp_0 + 1], regs=num_specialized_registers)
        math = sp.role("math", warps=list(range(spec_warp_start)), regs=num_math_registers)

        def scheduler_state():
            """Task scheduling state; pipeline cursors stay role-local."""
            state = {
                name: K.alloc_local([1], "uint32")
                for name in (
                    "cur_q",
                    "cur_kv",
                    "cur_num_kv",
                    "next_q",
                    "next_kv",
                    "next_num_kv",
                    "q_atom",
                    "kv_idx",
                    "num_kv",
                )
            }
            state["fetched"] = K.alloc_local([1], "uint32")
            K.assign(state["cur_q"][0], start_q_atom_idx)
            K.assign(state["cur_kv"][0], start_kv_idx)
            K.assign(state["cur_num_kv"][0], start_num_kv)
            for name in ("kv_idx", "num_kv"):
                K.assign(state[name][0], K.uint32(0))
            K.assign(state["q_atom"][0], batch_size * K.uint32(num_next_n_atoms))
            K.assign(state["next_q"][0], state["cur_q"][0])
            K.assign(state["next_kv"][0], state["cur_kv"][0])
            K.assign(state["next_num_kv"][0], state["cur_num_kv"][0])
            return state

        def pump(state):
            """One `fetch_next_task` + write-back, exactly as the original does."""
            K.assign(state["next_q"][0], state["cur_q"][0])
            K.assign(state["next_kv"][0], state["cur_kv"][0])
            K.assign(state["next_num_kv"][0], state["cur_num_kv"][0])
            fetch_next_task(
                state["cur_q"][0],
                state["cur_kv"][0],
                state["cur_num_kv"][0],
                end_q_atom_idx,
                end_kv_idx,
            )
            K.assign(state["next_q"][0], scheduler_result[0])
            K.assign(state["next_kv"][0], scheduler_result[1])
            K.assign(state["next_num_kv"][0], scheduler_result[2])
            K.assign(state["fetched"][0], scheduler_result[3])
            K.assign(state["cur_q"][0], scheduler_result[4])
            K.assign(state["cur_kv"][0], scheduler_result[5])
            K.assign(state["cur_num_kv"][0], scheduler_result[6])

        def load_block_table(state, kv_ptr, cached, lane_offset_tiles):
            """Block-table gather + the warp broadcast, at the original's
            position inside the task loop. The __shfl_sync is a value-returning
            warp collective in a loop: G3 forbids moving it."""
            with K.If(kv_ptr[0] == K.uint32(32)), K.Then():
                K.assign(kv_ptr[0], K.uint32(0))
                block_table_offset = K.local_scalar(
                    "uint64",
                    init=K.Cast("uint64", atom_to_block_table_row_expr(state["q_atom"][0]))
                    * K.Cast("uint64", block_table_stride),
                )
                prefetch_tile_idx = K.local_scalar(
                    "uint32",
                    init=state["kv_idx"][0]
                    + K.uint32(lane_offset_tiles)
                    + lane_idx_u32 * K.uint32(num_tiles_per_split),
                )
                block_table_index = K.local_scalar(
                    "uint64",
                    init=block_table_offset
                    + K.Cast("uint64", prefetch_tile_idx * K.uint32(num_pages_per_tile)),
                )
                for block_i in range(num_pages_per_tile):
                    # Guard the trailing partial tile: a valid compute tile may
                    # still exceed the block table's row length, and a garbage
                    # page id would send TMA out of bounds (page 0 is the
                    # masked-dumpster tile).
                    with K.If(
                        K.And(
                            prefetch_tile_idx < state["num_kv"][0],
                            prefetch_tile_idx * K.uint32(num_pages_per_tile) + K.uint32(block_i)
                            < K.uint32(config.max_num_pages),
                        )
                    ):
                        with K.Then():
                            K.ptx.ld.global_.u32(
                                cached[block_i],
                                block_table_flat.ptr_to(
                                    [K.Cast("int64", block_table_index + K.uint64(block_i))]
                                ),
                            )
                        with K.Else():
                            K.ptx.mov.b32(cached[block_i], K.uint32(0))
            K.cuda.warp_sync()
            kv_block_idx = K.alloc_local([num_pages_per_tile], "uint32")
            for block_i in range(num_pages_per_tile):
                K.ptx.shfl_sync.idx.b32(
                    kv_block_idx[block_i],
                    cached[block_i],
                    kv_ptr[0],
                    K.uint32(0x1F),
                    K.uint32(0xFFFFFFFF),
                )
            K.assign(kv_ptr[0], kv_ptr[0] + K.uint32(1))
            return kv_block_idx

        def issue_kv_tma(group, kv_state, kv_block_idx):
            base = group * num_kv_stages
            empty_kv_barriers.wait(K.uint32(base) + kv_state.stage, kv_state.phase ^ K.uint32(1))
            with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                for block_i in range(num_pages_per_tile):
                    K.ptx[TMA_G2S_3D](
                        smem_kv.ptr_to([group, kv_state.stage, block_i * page_size, 0]),
                        K.address_of(tensor_map_kv),
                        K.int32(0),
                        K.int32(0),
                        K.Cast("int32", kv_block_idx[block_i]),
                        full_kv_barriers.ptr_to([K.uint32(base) + kv_state.stage]),
                        cache_policy_evict_normal,
                    )
                    K.ptx[TMA_G2S_2D](
                        smem_kv_scales.ptr_to([group, kv_state.stage, block_i * page_size]),
                        K.address_of(tensor_map_kv_scales),
                        K.int32(0),
                        K.Cast("int32", kv_block_idx[block_i]),
                        full_kv_barriers.ptr_to([K.uint32(base) + kv_state.stage]),
                        cache_policy_evict_normal,
                    )
                full_kv_barriers.arrive(
                    K.uint32(base) + kv_state.stage,
                    tx_count=smem_kv_size_per_stage + smem_kv_scale_size_per_stage,
                )
            kv_state.advance()

        # ---------------- warp 8: Q + weights, and KV for group 0 --------
        with tma0:
            state = scheduler_state()
            # Stage 0 is issued unconditionally before the steady-state ring;
            # start the owned cursor at its exact successor instead of emitting
            # a runtime advance solely for that prologue issue.
            q_state = K.RingState(num_q_stages, stage=1)
            kv_state = K.RingState(num_kv_stages)
            pump(state)
            with K.If(state["fetched"][0] != K.uint32(0)), K.Then():
                issue_tma_q(K.uint32(0), state["next_q"][0])
            kv_ptr = K.alloc_local([1], "uint32")
            cached = K.alloc_local([num_pages_per_tile], "uint32")
            K.assign(kv_ptr[0], K.uint32(32))
            with K.While(state["fetched"][0] != K.uint32(0)):
                load_atom_advance(state["next_q"][0], batch_size)
                next_advance = K.local_scalar("uint32", init=atom_advance_result)
                prefetch_q = K.local_scalar("uint32", init=K.uint32(0))
                with (
                    K.If(
                        K.And(
                            state["q_atom"][0] != state["next_q"][0],
                            exist_q_atom_idx_expr(
                                state["next_q"][0] + next_advance, end_q_atom_idx, end_kv_idx
                            ),
                        )
                    ),
                    K.Then(),
                ):
                    K.assign(prefetch_q, K.uint32(1))
                with K.If(state["q_atom"][0] != state["next_q"][0]), K.Then():
                    K.assign(kv_ptr[0], K.uint32(32))
                K.assign(state["q_atom"][0], state["next_q"][0])
                K.assign(state["kv_idx"][0], state["next_kv"][0])
                K.assign(state["num_kv"][0], state["next_num_kv"][0])

                # Prefetch the next Q atom as soon as the batch changes so the Q
                # TMA overlaps the block-table load below (the original's order).
                with K.If(prefetch_q != K.uint32(0)), K.Then():
                    empty_q_barriers.wait(q_state.stage, q_state.phase ^ K.uint32(1))
                    issue_tma_q(q_state.stage, state["q_atom"][0] + next_advance)
                    q_state.advance()

                kv_block_idx = load_block_table(state, kv_ptr, cached, 0)
                issue_kv_tma(0, kv_state, kv_block_idx)
                pump(state)

        # ---------------- warp 9: KV for group 1 -------------------------
        with tma1:
            state = scheduler_state()
            kv_state = K.RingState(num_kv_stages)
            pump(state)
            kv_ptr = K.alloc_local([1], "uint32")
            cached = K.alloc_local([num_pages_per_tile], "uint32")
            K.assign(kv_ptr[0], K.uint32(32))
            with K.While(state["fetched"][0] != K.uint32(0)):
                with K.If(state["q_atom"][0] != state["next_q"][0]), K.Then():
                    K.assign(kv_ptr[0], K.uint32(32))
                K.assign(state["q_atom"][0], state["next_q"][0])
                K.assign(state["kv_idx"][0], state["next_kv"][0])
                K.assign(state["num_kv"][0], state["next_num_kv"][0])
                kv_block_idx = load_block_table(state, kv_ptr, cached, 1)
                issue_kv_tma(1, kv_state, kv_block_idx)
                pump(state)

        # ---------------- warps 10-11: UMMA issuers ----------------------
        with umma:
            umma_group_idx = K.Cast("uint32", K.warp_id_in_role())
            # TMEM allocation happens off the full-CTA sync path (the ~300-cycle
            # tcgen05.alloc would otherwise hold back the TMA warps' first
            # issue). Only UMMA and Math wait for it, on named barrier 9.
            # Keep the raw collective here so allocation stays on this
            # kernel-specific issue boundary.
            with K.If(umma_group_idx == K.uint32(1)), K.Then():
                K.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    K.address_of(tmem_ptr_in_smem[0]), K.uint32(num_tmem_cols)
                )
            K.ptx.bar.sync(9, K.uint32(num_math_threads + 2 * 32))
            tmem_allocated = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(tmem_allocated, tmem_ptr_in_smem.ptr_to([0]))
            K.cuda.trap_when_assert_failed(tmem_allocated == K.uint32(0))
            desc_i = K.local_scalar("uint32")
            desc_a = K.local_scalar("uint64")
            desc_b = K.local_scalar("uint64")
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
            # The dead shift round trip is the original's; ptxas folds the whole
            # thing to one UMOV immediate, so it costs nothing and removing it
            # would be a deviation with nothing to win.
            runtime_instr_desc = K.local_scalar("uint64")
            runtime_instr_desc_hi = K.local_scalar("uint32")
            K.assign(runtime_instr_desc, K.shift_left(K.Cast("uint64", desc_i), K.uint64(32)))
            K.assign(
                runtime_instr_desc_hi,
                K.Cast("uint32", K.shift_right(runtime_instr_desc, K.uint64(32))),
            )
            state = scheduler_state()
            q_state = K.RingState(num_q_stages)
            kv_state = K.RingState(num_kv_stages)
            umma_state = K.RingState(num_umma_stages)
            q_stage = K.local_scalar("uint32", init=K.uint32(0))
            pump(state)
            with K.While(state["fetched"][0] != K.uint32(0)):
                with K.If(state["q_atom"][0] != state["next_q"][0]), K.Then():
                    K.assign(q_stage, q_state.stage)
                    full_q_barriers.wait(q_stage, q_state.phase)
                    q_state.advance()
                K.assign(state["q_atom"][0], state["next_q"][0])
                K.assign(state["kv_idx"][0], state["next_kv"][0])

                kv_stage = kv_state.stage
                kv_phase = kv_state.phase
                full_kv_barriers.wait(umma_group_idx * K.uint32(num_kv_stages) + kv_stage, kv_phase)
                umma_stage = umma_state.stage
                umma_phase = umma_state.phase
                empty_umma_barriers.wait(
                    umma_group_idx * K.uint32(num_umma_stages) + umma_stage,
                    umma_phase ^ K.uint32(1),
                )
                K.ptx.tcgen05.fence__after_thread_sync()
                # G3, LAW: elect_sync wraps EACH MMA individually, inside the
                # unrolled k loop, with the descriptor recompute OUTSIDE the
                # elect. This is the original's placement and it is preserved
                # verbatim -- never normalised to the non-paged kernel's
                # whole-loop elect form. Hoisting a collective out of a loop
                # produced a deterministic launch failure elsewhere.
                for k_phase in range(head_dim // umma_k):
                    make_smem_desc(
                        desc_a, smem_kv.ptr_to([umma_group_idx, kv_stage, 0, k_phase * umma_k])
                    )
                    make_smem_desc(desc_b, smem_q.ptr_to([q_stage, 0, k_phase * umma_k]))
                    with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                        K.ptx[MMA](
                            K.uint32(tmem_col)
                            + umma_group_idx * K.uint32(umma_n * num_umma_stages)
                            + umma_stage * K.uint32(umma_n),
                            desc_a,
                            desc_b,
                            runtime_instr_desc_hi,
                            K.uint32(0),
                            K.uint32(0),
                            K.uint32(0),
                            K.uint32(0),
                            K.ptx.pred(K.uint32(k_phase)),
                        )
                with K.If(K.cuda.elect_sync() != K.uint32(0)), K.Then():
                    full_umma_barriers.arrive(
                        umma_group_idx * K.uint32(num_umma_stages) + umma_stage, cta_group=1
                    )
                kv_state.advance()
                umma_state.advance()
                pump(state)

        # ---------------- warps 0-7: math + epilogue ---------------------
        with math:
            # Math warps consume TMEM: wait on named barrier 9 for the UMMA
            # warp's tcgen05.alloc (see the UMMA role).
            K.ptx.bar.sync(9, K.uint32(num_math_threads + 2 * 32))
            math_wg_u32 = K.Cast("uint32", warpgroup_idx)
            tmem_start_base = K.uint32(tmem_col) + math_wg_u32 * K.uint32(
                umma_n * num_umma_stages
            )
            math_thread_idx = K.local_scalar(
                "uint32",
                init=(K.Cast("uint32", K.warp_id_in_role()) % K.uint32(4)) * K.uint32(32)
                + lane_idx_u32,
            )
            cached_weights = K.alloc_local([next_n_atom, num_heads], "float32")
            is_paired_atom = K.local_scalar("uint32", init=K.uint32(0))
            state = scheduler_state()
            q_state = K.RingState(num_q_stages)
            kv_state = K.RingState(num_kv_stages)
            umma_state = K.RingState(num_umma_stages)
            q_stage = K.local_scalar("uint32", init=K.uint32(0))
            has_q_stage = K.local_scalar("uint32", init=K.uint32(0))
            pump(state)

            def reduce_and_store(num_iters_c, kv_offset, scale_kv, umma_stage_idx_arg):
                accum = K.alloc_local([num_heads], "float32")
                # relu(x) = (x + |x|) * 0.5; the epilogue accumulates 2*relu and
                # folds the 0.5 into the output scale so the ReLU runs on the
                # FMA pipe through the packed f32x2 add with abs source
                # modifiers, instead of scalar FMNMX on the ALU pipe.
                scale_kv_half = K.local_scalar("float32")
                K.ptx.mul.rn.f32(scale_kv_half, scale_kv, K.float32(0.5))
                for q_inner_i in range(num_iters_c):
                    tmem_addr = (
                        tmem_start_base
                        + umma_stage_idx_arg * K.uint32(umma_n)
                        + K.uint32(q_inner_i * num_heads)
                    )
                    K.ptx[TC_LD](*[accum[h] for h in range(num_heads)], tmem_addr)
                    K.ptx.tcgen05.wait__ld.sync.aligned()
                    if q_inner_i == num_iters_c - 1:
                        # Release the UMMA stage right after the last TMEM load
                        # so the next MMA can start while the FMA chain and the
                        # store are still running (the original's order).
                        K.ptx.tcgen05.fence__before_thread_sync()
                        with K.If(lane_idx == 0), K.Then():
                            empty_umma_barriers.arrive(
                                math_wg_u32 * K.uint32(num_umma_stages) + umma_stage_idx_arg
                            )
                    sum_0 = K.cuda.make_float2(K.float32(0), K.float32(0))
                    sum_1 = K.cuda.make_float2(K.float32(0), K.float32(0))
                    for head_j_group in range(num_heads // 4):
                        head_j = head_j_group * 4
                        sum_0 = relu2_fma_f32x2(
                            K.cuda.make_float2(accum[head_j], accum[head_j + 1]),
                            K.cuda.make_float2(
                                cached_weights[q_inner_i, head_j],
                                cached_weights[q_inner_i, head_j + 1],
                            ),
                            sum_0,
                        )
                        sum_1 = relu2_fma_f32x2(
                            K.cuda.make_float2(accum[head_j + 2], accum[head_j + 3]),
                            K.cuda.make_float2(
                                cached_weights[q_inner_i, head_j + 2],
                                cached_weights[q_inner_i, head_j + 3],
                            ),
                            sum_1,
                        )
                    sum_v = K.local_scalar("uint64")
                    K.ptx.add.rn.f32x2(sum_v, sum_0, sum_1)
                    _add = K.local_scalar("float32")
                    K.ptx.add.rn.f32(_add, K.cuda.float2_x(sum_v), K.cuda.float2_y(sum_v))
                    result_f32 = K.local_scalar("float32")
                    K.ptx.mul.rn.f32(result_f32, scale_kv_half, _add)
                    result = K.Cast(logits_tir_dtype, result_f32)
                    logits_offset = (
                        K.Cast("uint64", kv_offset)
                        + K.Cast("uint64", K.uint32(q_inner_i)) * K.Cast("uint64", logits_stride)
                        + K.Cast("uint64", math_thread_idx)
                    )
                    if config.logits_dtype == "float32":
                        K.ptx.st.global_.f32(logits_flat.ptr_to([logits_offset]), result)
                    else:
                        K.ptx.st.global_.b16(logits_flat.ptr_to([logits_offset]), result)

            with K.While(state["fetched"][0] != K.uint32(0)):
                with K.If(state["q_atom"][0] != state["next_q"][0]), K.Then():
                    with K.If(has_q_stage != K.uint32(0)), K.Then():
                        with K.If(lane_idx == 0), K.Then():
                            empty_q_barriers.arrive(q_stage)
                    K.assign(q_stage, q_state.stage)
                    full_q_barriers.wait(q_stage, q_state.phase)
                    q_state.advance()
                    K.assign(has_q_stage, K.uint32(1))
                    for weight_i in range(next_n_atom):
                        for weight_j in range(num_heads // 4):
                            wc = weight_j * 4
                            K.ptx.ld.shared.v4.f32(
                                cached_weights[weight_i, wc],
                                cached_weights[weight_i, wc + 1],
                                cached_weights[weight_i, wc + 2],
                                cached_weights[weight_i, wc + 3],
                                smem_weights.ptr_to([q_stage, weight_i, wc]),
                            )
                    if config.varlen:
                        load_atom_advance(state["next_q"][0], batch_size)
                        K.assign(
                            is_paired_atom, K.Cast("uint32", atom_advance_result == K.uint32(2))
                        )
                K.assign(state["q_atom"][0], state["next_q"][0])
                K.assign(state["kv_idx"][0], state["next_kv"][0])
                kv_offset = K.local_scalar(
                    "uint64",
                    init=K.Cast("uint64", atom_to_token_idx_expr(state["q_atom"][0]))
                    * K.Cast("uint64", logits_stride)
                    + K.Cast("uint64", (state["kv_idx"][0] + math_wg_u32) * K.uint32(umma_m)),
                )
                kv_stage = kv_state.stage
                kv_phase = kv_state.phase
                full_kv_barriers.wait(math_wg_u32 * K.uint32(num_kv_stages) + kv_stage, kv_phase)
                scale_kv = K.local_scalar("float32")
                K.ptx.ld.shared.f32(
                    scale_kv, smem_kv_scales.ptr_to([math_wg_u32, kv_stage, math_thread_idx])
                )
                umma_stage = umma_state.stage
                umma_phase = umma_state.phase
                full_umma_barriers.wait(
                    math_wg_u32 * K.uint32(num_umma_stages) + umma_stage, umma_phase
                )
                K.ptx.tcgen05.fence__after_thread_sync()
                with K.If(lane_idx == 0), K.Then():
                    empty_kv_barriers.arrive(math_wg_u32 * K.uint32(num_kv_stages) + kv_stage)
                if config.varlen:
                    with K.If(is_paired_atom != K.uint32(0)):
                        with K.Then():
                            reduce_and_store(next_n_atom, kv_offset, scale_kv, umma_stage)
                        with K.Else():
                            reduce_and_store(1, kv_offset, scale_kv, umma_stage)
                elif k_pad_odd_n:
                    with K.If(
                        state["q_atom"][0] % K.uint32(num_next_n_atoms)
                        == K.uint32(num_next_n_atoms - 1)
                    ):
                        with K.Then():
                            reduce_and_store(1, kv_offset, scale_kv, umma_stage)
                        with K.Else():
                            reduce_and_store(next_n_atom, kv_offset, scale_kv, umma_stage)
                else:
                    reduce_and_store(next_n_atom, kv_offset, scale_kv, umma_stage)
                kv_state.advance()
                umma_state.advance()
                pump(state)
            K.ptx.griddepcontrol.launch_dependents()
            K.ptx.bar.sync(8, K.uint32(num_math_threads))
            with K.If(K.warp_id_in_role() == 0), K.Then():
                K.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                    K.uint32(tmem_col), K.uint32(num_tmem_cols)
                )

    # `@K.kernel` has no `attrs=`. NOTE the paged original sets ONLY
    # kernel_launch_params -- no tirx.persistent_kernel, unlike the non-paged
    # sibling.
    sm100_fp8_paged_mqa_logits.func = sm100_fp8_paged_mqa_logits.func.with_attr(
        "tirx.kernel_launch_params",
        [
            "blockIdx.x",
            "threadIdx.x",
            "tirx.use_programtic_dependent_launch",
            "tirx.use_dyn_shared_memory",
        ],
    )
    return sm100_fp8_paged_mqa_logits.func


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


@cache
def _load_sglang_cutedsl_reference() -> tuple[Any, Any]:
    """Load SGLang's kernel modules without initializing its unrelated frontend."""
    spec = find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("cannot find the pinned SGLang source checkout")
    root = Path(next(iter(spec.submodule_search_locations)))

    package_paths = {
        "sglang": root,
        "sglang.kernels": root / "kernels",
        "sglang.kernels.ops": root / "kernels" / "ops",
        "sglang.kernels.ops.attention": root / "kernels" / "ops" / "attention",
        "sglang.kernels.ops.attention.dsa": root / "kernels" / "ops" / "attention" / "dsa",
        "sglang.srt": root / "srt",
    }
    for name, path in package_paths.items():
        package = types.ModuleType(name)
        package.__package__ = name
        package.__path__ = [str(path)]
        # A spec-less module in sys.modules makes every later find_spec("sglang")
        # raise ValueError, failing the availability probe of subsequent configs.
        package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = [str(path)]
        sys.modules[name] = package

    utils = types.ModuleType("sglang.srt.utils")
    utils.is_sm100_supported = lambda: torch.cuda.get_device_capability()[0] == 10
    utils.__spec__ = importlib.machinery.ModuleSpec(utils.__name__, loader=None)
    sys.modules[utils.__name__] = utils

    module = importlib.import_module("sglang.kernels.ops.attention.dsa.cutedsl_paged_mqa_logits")
    return module.CuteDSLPagedMQALogitsRunner, module.pick_dsl_expand


def _make_sglang_cutedsl_runner(data: dict[str, Any]) -> Any:
    config: PagedMQALogitsFP8Config = data["config"]
    if config.context_pattern == "random_2d" and config.next_n > 1:
        raise ValueError(
            "SGLang CuTeDSL requires causal context lengths when next_n > 1; "
            "use context_pattern='sglang_fixed' or 'sglang_ragged'"
        )

    CuteDSLPagedMQALogitsRunner, pick_dsl_expand = _load_sglang_cutedsl_reference()

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
        data["context_lens"].view(-1),
        logits.view(-1),
        data["block_table"].view(-1),
        indices.view(-1),
        data["tirx_schedule_meta"].view(-1),
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
    from tirx_kernels.runner import bench, external_references_enabled

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
    deepgemm_logits = None
    max_diff = None
    if external_references_enabled():
        deepgemm_logits = _run_deepgemm_paged_mqa(data, clean_logits=False)
        tirx_logits = _run_tirx_invocation(data, invocation)
        torch.cuda.synchronize()
        max_diff = _assert_valid_correct(
            data, tirx_logits, deepgemm_logits, name="TIRx vs DeepGEMM"
        )
        torch.cuda.empty_cache()

    def _deepgemm():
        return lambda: _run_deepgemm_paged_mqa(data, clean_logits=False)

    def _sglang_cutedsl():
        assert deepgemm_logits is not None
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
    if max_diff is not None:
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
