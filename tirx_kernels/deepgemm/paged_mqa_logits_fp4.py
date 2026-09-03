# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

import ctypes
import os
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K

_DEEP_GEMM_MODULE_NAME = "deep_gemm"
_SM100_SMEM_CAPACITY = 232448
_TEST_DIFF_THRESHOLD = 5e-6
_COMPILE_CACHE_NAMESPACE = "deepgemm.paged_mqa_logits_fp4.compile"


def _paged_mqa_logits_fp4_cuda_postproc(code: str) -> str:
    if "sm100_fp4_paged_mqa_logits_kernel" not in code:
        return code

    original = code
    replacements = {
        "int* __restrict__ block_table_flat_ptr": "const int* __restrict__ block_table_flat_ptr",
        "int* __restrict__ context_lens_flat_ptr": "const int* __restrict__ context_lens_flat_ptr",
        "int* __restrict__ indices_ptr": "const int* __restrict__ indices_ptr",
        "int* __restrict__ indices_flat_ptr": "const int* __restrict__ indices_flat_ptr",
        "int* __restrict__ schedule_meta_u32_flat_ptr": (
            "const int* __restrict__ schedule_meta_u32_flat_ptr"
        ),
    }
    for old, new in replacements.items():
        code = code.replace(old, new)

    code = code.replace(
        "((uint*)schedule_meta_u32_flat_ptr)", "((const uint*)schedule_meta_u32_flat_ptr)"
    )
    code = code.replace(
        "((unsigned int*)schedule_meta_u32_flat_ptr)",
        "((const unsigned int*)schedule_meta_u32_flat_ptr)",
    )

    dump_dir = os.environ.get("PAGED_MQA_FP4_POSTPROC_DUMP_DIR")
    if dump_dir:
        path = Path(dump_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "original.cu").write_text(original)
        (path / "postproc.cu").write_text(code)

    return code


@dataclass(frozen=True)
class PagedMQALogitsFP4Config:
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
        if self.head_dim != 128:
            raise ValueError("head_dim must be 128 for the SM100 FP4 paged MQA logits kernel")
        if self.page_size not in (32, 64):
            raise ValueError("page_size must match DeepGEMM block_kv 32 or 64")
        if self.split_kv % self.page_size != 0:
            raise ValueError("split_kv must be divisible by page_size")
        if self.max_num_pages <= 0 or self.num_pages < self.max_num_pages:
            raise ValueError("num_pages must cover max_num_pages")
        if self.logits_dtype not in ("float32", "bfloat16"):
            raise ValueError("logits_dtype must be 'float32' or 'bfloat16'")
        if not self.context_lens_2d:
            raise ValueError("DeepGEMM paged FP4 API currently requires 2D context_lens")
        if self.varlen and self.next_n != 1:
            raise ValueError("DeepGEMM varlen paged mode requires next_n == 1")
        if self.indices_pair_stride <= 0:
            raise ValueError("indices_pair_stride must be positive")


def _make_config(**kwargs: Any) -> PagedMQALogitsFP4Config:
    kwargs = {key: value for key, value in kwargs.items() if key != "label"}
    config = PagedMQALogitsFP4Config(**kwargs)
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
    return (
        f"b{config['batch_size']}_n{config['next_n']}_mp{config['max_num_pages']}_"
        f"ps{config['page_size']}_h{config['num_heads']}_d{config['head_dim']}_{dtype}_{mode}"
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
    varlen: bool = False,
    indices_pair_stride: int = 1,
) -> dict[str, Any]:
    config = {
        "batch_size": batch_size,
        "next_n": next_n,
        "max_num_pages": max_num_pages,
        "num_pages": num_pages,
        "num_heads": 64,
        "head_dim": 128,
        "page_size": page_size,
        "logits_dtype": logits_dtype,
        "seed": seed,
        "varlen": varlen,
        "indices_pair_stride": indices_pair_stride,
    }
    config["label"] = _config_label(config)
    return config


KERNEL_META = {
    "name": "deepgemm_sm100_fp4_paged_mqa_logits",
    "category": "deepgemm",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a", "sm_110a"],
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

DSA_INDEXER_LIKE_COVERAGE = [
    _make_case(
        batch_size=batch_size,
        next_n=1,
        max_num_pages=max_num_pages,
        num_pages=max(11923, max_num_pages),
        page_size=page_size,
        logits_dtype=logits_dtype,
        seed=2000 + seed,
    )
    for seed, (batch_size, max_num_pages, page_size, logits_dtype) in enumerate(
        (batch_size, max_num_pages, page_size, logits_dtype)
        for logits_dtype in ("float32", "bfloat16")
        for page_size in (32, 64)
        for batch_size in (1, 2, 4, 8, 16)
        for max_num_pages in (1, 8, 32, 128)
    )
]

CONFIGS = DSA_INDEXER_LIKE_COVERAGE


def load_deep_gemm_paged_mqa() -> tuple[Any, str]:
    try:
        import deep_gemm as module
    except Exception as exc:
        raise SkipTest(
            f"DeepGEMM FP4 paged MQA logits runtime unavailable: {_DEEP_GEMM_MODULE_NAME}: {exc}"
        ) from exc

    if not hasattr(module, "fp8_fp4_paged_mqa_logits"):
        raise SkipTest("DeepGEMM runtime unavailable: missing fp8_fp4_paged_mqa_logits")
    if not hasattr(module, "get_paged_mqa_logits_metadata"):
        raise SkipTest("DeepGEMM runtime unavailable: missing get_paged_mqa_logits_metadata")
    return module, "installed"


def _make_context_lens(config: PagedMQALogitsFP4Config) -> torch.Tensor:
    max_context_len = config.max_context_len
    if max_context_len == config.page_size:
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


def _make_block_table(config: PagedMQALogitsFP4Config) -> torch.Tensor:
    page_ids = torch.arange(config.num_pages, dtype=torch.int32, device="cuda")
    rows = []
    for batch_idx in range(config.batch_size):
        start = (batch_idx * config.max_num_pages) % config.num_pages
        rows.append(page_ids.roll(-start)[: config.max_num_pages])
    return torch.stack(rows, dim=0).contiguous()


def _make_indices(config: PagedMQALogitsFP4Config) -> torch.Tensor | None:
    if not config.varlen:
        return None
    indices = torch.arange(config.batch_size, dtype=torch.int32, device="cuda")
    if config.indices_pair_stride > 1:
        indices = indices // config.indices_pair_stride
    return indices.contiguous()


def _make_schedule_meta(deep_gemm, context_lens, page_size: int, num_sms: int, indices=None):
    from tirx_kernels.target import prepare_cuda_arch

    if prepare_cuda_arch() == "sm_110a":
        from ._paged_mqa_schedule import make_schedule_metadata

        return make_schedule_metadata(context_lens, num_sms, indices)
    return deep_gemm.get_paged_mqa_logits_metadata(context_lens, page_size, num_sms, indices)


def _make_fused_kv_cache(
    config: PagedMQALogitsFP4Config, deep_gemm: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_bf16 = torch.randn(
        config.num_pages, config.page_size, 1, config.head_dim, device="cuda", dtype=torch.bfloat16
    ).clamp_(-2.0, 2.0)
    kv_fp4 = deep_gemm.utils.per_token_cast_to_fp4(
        kv_bf16.view(-1, config.head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    kv_packed = kv_fp4[0].view(config.num_pages, config.page_size, config.head_dim // 2)
    kv_scales = kv_fp4[1].view(config.num_pages, config.page_size)
    kv_dequant = deep_gemm.utils.cast_back_from_fp4(
        kv_fp4[0], kv_fp4[1], gran_k=32, use_packed_ue8m0=True
    ).view(config.num_pages, config.page_size, 1, config.head_dim)
    fused = torch.empty(
        (config.num_pages, config.page_size, 1, config.head_dim // 2 + 4),
        dtype=torch.uint8,
        device="cuda",
    )
    fused_flat = fused.view(config.num_pages, config.page_size * (config.head_dim // 2 + 4))
    fused_flat[:, : config.page_size * config.head_dim // 2].copy_(
        kv_packed.view(torch.uint8).reshape(
            config.num_pages, config.page_size * config.head_dim // 2
        )
    )
    fused_flat[:, config.page_size * config.head_dim // 2 :].copy_(
        kv_scales.view(torch.uint8).reshape(config.num_pages, config.page_size * 4)
    )
    return fused.contiguous(), kv_dequant.view(
        config.num_pages, config.page_size, config.head_dim
    ).to(torch.bfloat16)


def _ref_paged_mqa_logits(
    q: torch.Tensor,
    kv_dequant: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    config: PagedMQALogitsFP4Config,
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


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    deep_gemm, source = load_deep_gemm_paged_mqa()
    config = _make_config(**kwargs)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for SM100 FP4 paged MQA logits")
    from tirx_kernels.target import supports_sm100_kernel

    if not supports_sm100_kernel(torch.cuda.get_device_capability()):
        raise SkipTest("SM100 FP4 paged MQA logits requires SM100 or prepared Thor")

    torch.manual_seed(config.seed)
    runtime_config = PagedMQALogitsFP4Config(
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
    q_fp4 = deep_gemm.utils.per_token_cast_to_fp4(
        q_bf16.view(-1, config.head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    q_in = (
        q_fp4[0].view(config.batch_size, config.next_n, config.num_heads, config.head_dim // 2),
        q_fp4[1].view(config.batch_size, config.next_n, config.num_heads),
    )
    q_in = (q_in[0].contiguous(), q_in[1].contiguous())
    q_simulated = deep_gemm.utils.cast_back_from_fp4(
        q_fp4[0], q_fp4[1], gran_k=32, use_packed_ue8m0=True
    ).view(config.batch_size, config.next_n, config.num_heads, config.head_dim)
    fused_kv_cache, kv_dequant = _make_fused_kv_cache(config, deep_gemm)
    weights = torch.randn(
        config.batch_size * config.next_n, config.num_heads, device="cuda", dtype=torch.float32
    ).contiguous()
    context_lens = _make_context_lens(config)
    block_table = _make_block_table(config)
    indices = _make_indices(config)
    schedule_meta = _make_schedule_meta(
        deep_gemm, context_lens, config.page_size, runtime_config.num_sms, indices
    )
    tirx_schedule_meta = schedule_meta
    if not config.varlen and config.next_n >= 2:
        num_q_atoms = _align_up(config.next_n, 2) // 2
        atom_context_lens = context_lens[:, -1:].expand(config.batch_size, num_q_atoms).contiguous()
        tirx_schedule_meta = _make_schedule_meta(
            deep_gemm, atom_context_lens, config.page_size, runtime_config.num_sms
        )
        expected_end = config.batch_size * num_q_atoms
        if int(tirx_schedule_meta[-1, 0]) != expected_end:
            raise RuntimeError(
                f"TIRx schedule metadata ends at {int(tirx_schedule_meta[-1, 0])}, "
                f"expected {expected_end} q atoms"
            )
    reference = _ref_paged_mqa_logits(
        q_simulated.to(torch.bfloat16), kv_dequant, weights, context_lens, block_table, config
    )
    return {
        "config": runtime_config,
        "reference_source": source,
        "q": q_bf16,
        "q_in": q_in,
        "fused_kv_cache": fused_kv_cache,
        "weights": weights,
        "context_lens": context_lens,
        "block_table": block_table,
        "indices": indices,
        "schedule_meta": schedule_meta,
        "tirx_schedule_meta": tirx_schedule_meta,
        "reference": reference,
        "deep_gemm": deep_gemm,
    }


def get_kernel(**kwargs: Any):
    config = _make_config(**kwargs)

    num_heads = config.num_heads
    head_dim = config.head_dim
    page_size = config.page_size
    k_pad_odd_n = (not config.varlen) and (config.next_n % 2 == 1) and (config.next_n >= 3)
    next_n_atom = 2 if (config.varlen or config.next_n >= 2) else 1
    num_next_n_atoms = _align_up(config.next_n, next_n_atom) // next_n_atom
    num_q_stages = 3
    split_kv = config.split_kv
    umma_m = 128
    umma_k = 64
    umma_n = next_n_atom * num_heads
    num_math_warpgroups = split_kv // umma_m
    num_tiles_per_split = split_kv // umma_m
    num_pages_per_tile = umma_m // page_size
    num_utccp_aligned_elems = 128
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
    num_sfq_atom = _align_up(next_n_atom * num_heads, num_utccp_aligned_elems)
    num_sfkv = _align_up(umma_m, num_utccp_aligned_elems)
    real_num_sfq_atom = next_n_atom * num_heads
    smem_alignment = 8 * (head_dim // 2)
    desc_sdo = 8 * (head_dim // 2) // 16
    sf_desc_sdo = 8 * 4 * 4 // 16
    smem_q_size_per_stage = next_n_atom * num_heads * (head_dim // 2)
    smem_sf_q_size_per_stage = num_sfq_atom * 4
    smem_kv_size_per_stage = umma_m * (head_dim // 2)
    smem_sf_kv_size_per_stage = num_sfkv * 4
    smem_weight_size_per_stage = next_n_atom * num_heads * 4
    _kv_stage_bytes = (smem_kv_size_per_stage + smem_sf_kv_size_per_stage) * num_math_warpgroups
    _fixed_smem_bytes = (
        smem_q_size_per_stage * num_q_stages
        + smem_sf_q_size_per_stage * num_q_stages * num_math_warpgroups
        + smem_weight_size_per_stage * num_q_stages
        + 1024
    )
    num_kv_stages = min(8, max(3, (_SM100_SMEM_CAPACITY - _fixed_smem_bytes) // _kv_stage_bytes))
    num_tmem_stages = 2
    num_accum_tmem_cols = next_n_atom * num_heads * num_math_warpgroups * num_tmem_stages
    num_sfa_tmem_cols = (num_sfq_atom // 32) * num_math_warpgroups
    num_sfb_tmem_cols = (num_sfkv // 32) * num_math_warpgroups
    num_requested_tmem_cols = num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols
    if num_requested_tmem_cols > 512:
        num_tmem_stages = 1
        num_accum_tmem_cols = next_n_atom * num_heads * num_math_warpgroups * num_tmem_stages
        num_requested_tmem_cols = num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols
    num_tmem_cols = 32
    if num_requested_tmem_cols > 32:
        num_tmem_cols = 64
    if num_requested_tmem_cols > 64:
        num_tmem_cols = 128
    if num_requested_tmem_cols > 128:
        num_tmem_cols = 256
    if num_requested_tmem_cols > 256:
        num_tmem_cols = 512
    num_tmem_barriers = num_math_warpgroups * num_tmem_stages
    logits_tir_dtype = "float32" if config.logits_dtype == "float32" else "bfloat16"

    # The original's own static assertions, kept as trace-time asserts. They
    # check Python constants on both sides and emit nothing either way.
    assert num_specialized_threads == 128 and num_math_threads % 128 == 0, "Invalid threads"
    assert split_kv == num_math_warpgroups * umma_m and split_kv % num_utccp_aligned_elems == 0
    assert split_kv == umma_m * num_tiles_per_split, "Invalid `SPLIT_KV`"
    assert umma_m == page_size * num_pages_per_tile, "Invalid `UMMA_M`"
    assert smem_q_size_per_stage % smem_alignment == 0, "Unaligned TMA swizzling"
    assert smem_kv_size_per_stage % smem_alignment == 0, "Unaligned TMA swizzling"
    assert num_requested_tmem_cols <= 512 and num_tmem_cols <= 512, "Too many tensor memory"
    assert head_dim % umma_k == 0, "Invalid head dim"
    assert num_heads % 8 == 0, "Invalid head"
    assert num_heads in (32, 64), "Unsupported TMEM load size"

    # Block-table L2 warm-up coverage (trace-time ints): the whole table, capped
    # at 512 lines (64 KB).
    num_block_table_bytes = config.batch_size * config.max_num_pages * 4
    num_prefetch_lines = min((num_block_table_bytes + 127) // 128, 512)

    # One ptx spelling per rank: unicast, no .cta_group modifier, evict-normal
    # L2 cache policy as a real operand (the original's own strings).
    TMA_G2S_2D = (
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
    )
    TMA_G2S_3D = (
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
    )
    TCGEN05_CP = "tcgen05.cp.cta_group::1.32x128b.warpx4"
    TC_LD = f"tcgen05.ld.sync.aligned.32x32b.x{num_heads}.b32"

    @K.kernel(warps=num_warps, arch="sm_100a", min_blocks_per_sm=1, grid=config.num_sms)
    def sm100_fp4_paged_mqa_logits(
        batch_size: K.u32,
        logits_stride: K.u32,
        block_table_stride: K.u32,
        context_lens_flat: K.gptr[K.i32],
        logits_flat: K.gptr[logits_tir_dtype],
        block_table_flat: K.gptr[K.i32],
        indices: K.gptr[K.i32],
        schedule_meta_flat: K.gptr[K.i32],
        tensor_map_q: K.TensorMap,
        tensor_map_sf_q: K.TensorMap,
        tensor_map_kv: K.TensorMap,
        tensor_map_sf_kv: K.TensorMap,
        tensor_map_weights: K.TensorMap,
    ):
        cache_policy_evict_normal = K.uint64(1152921504606846976)

        sm_idx_u32 = K.local_scalar("uint32")
        K.ptx.mov.u32(sm_idx_u32, K.Cast("uint32", K.cta_id()))
        warp_idx = K.warp_id()
        warp_idx_u32 = K.Cast("uint32", warp_idx)
        warp_idx_presync_u32 = K.local_scalar("uint32")
        K.ptx.mov.u32(warp_idx_presync_u32, warp_idx_u32)
        warp_idx_presync = K.Cast("int32", warp_idx_presync_u32)
        warpgroup_idx = K.warpgroup_id([num_warps // 4])
        lane_idx = K.lane_id()
        lane_idx_u32 = K.local_scalar("uint32", init=K.Cast("uint32", lane_idx))

        with K.If(warp_idx_presync == spec_warp_start), K.Then():
            K.ptx.prefetch.tensormap(K.address_of(tensor_map_q))
            K.ptx.prefetch.tensormap(K.address_of(tensor_map_sf_q))
            K.ptx.prefetch.tensormap(K.address_of(tensor_map_weights))
            K.ptx.prefetch.tensormap(K.address_of(tensor_map_kv))
            K.ptx.prefetch.tensormap(K.address_of(tensor_map_sf_kv))

        # ---------------- SMEM / TMEM ownership -------------------------
        # Allocation order is the physical SMEM layout. These buffers remain
        # raw because the TMA maps and the hand-matched MMA descriptors own the
        # 128 B swizzle; the pool owns alignment and offsets.
        smem = K.smem_pool()
        smem_q = smem.alloc(
            (num_q_stages, next_n_atom * num_heads, head_dim // 2), K.u8, align=smem_alignment
        )
        smem_kv = smem.alloc(
            (num_math_warpgroups, num_kv_stages, umma_m, head_dim // 2), K.u8, align=smem_alignment
        )
        smem_sf_q = smem.alloc((num_math_warpgroups, num_q_stages, num_sfq_atom), K.u32, align=16)
        smem_sf_kv = smem.alloc((num_math_warpgroups, num_kv_stages, num_sfkv), K.u32, align=16)
        smem_weights = smem.alloc((num_q_stages, next_n_atom, num_heads), K.f32, align=16)

        # Typed barrier families preserve the original allocation and explicit
        # four-warp initialization order below.
        full_q_barriers = K.TMABar(smem, num_q_stages)
        empty_q_barriers = K.MBarrier(smem, num_q_stages)
        full_kv_barriers = K.TMABar(smem, num_kv_stages * num_math_warpgroups)
        empty_kv_barriers = K.TCGen05Bar(smem, num_kv_stages * num_math_warpgroups)
        full_tmem_barriers = K.TCGen05Bar(smem, num_tmem_barriers)
        empty_tmem_barriers = K.MBarrier(smem, num_tmem_barriers)
        tmem_ptr_in_smem = smem.alloc((1,), K.u32, align=4)
        if smem.bytes > _SM100_SMEM_CAPACITY:
            raise ValueError(f"dynamic shared memory {smem.bytes} exceeds SM100 capacity")

        # TMEM is a fixed column map: accumulator first, then SFQ and SFKV.
        # Keep these as raw columns; Kern deliberately has no tmem buffers.
        accum_tmem_col = 0
        sfq_tmem_col = num_accum_tmem_cols
        sfkv_tmem_col = num_accum_tmem_cols + num_sfa_tmem_cols

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
            if next_n_atom == 1:
                return q_atom_idx
            return q_atom_idx * K.uint32(next_n_atom)

        def atom_to_block_table_row_expr(q_atom_idx):
            if config.varlen:
                return q_atom_idx
            if num_next_n_atoms == 1:
                return q_atom_idx
            return q_atom_idx // K.uint32(num_next_n_atoms)

        def should_refresh_num_kv_expr(q_atom_idx):
            if config.varlen:
                return K.bool(True)
            if num_next_n_atoms == 1:
                return K.bool(True)
            return q_atom_idx % K.uint32(num_next_n_atoms) == K.uint32(0)

        def exist_q_atom_idx_expr(q_atom_idx, end_q_atom_idx_arg, end_kv_idx_arg):
            return K.Or(
                q_atom_idx < end_q_atom_idx_arg,
                K.And(q_atom_idx == end_q_atom_idx_arg, K.uint32(0) < end_kv_idx_arg),
            )

        def local(dtype, value=None):
            """One scalar local, the port's spelling of the original's `x: K.t = v`.

            The original declares these everywhere -- annotated (`q_idx:
            K.uint32`), unannotated (`weight_col = weight_j * 4`, which
            TVMScript still binds as a let) and via `K.local_scalar`. All three
            emit `alignas(64) T x_ptr[1]`, so all three are an `alloc_local`
            here. A Python name holding an unbound expression is not a binding
            and re-emits the whole expression at every use site.
            """
            slot = K.alloc_local([1], dtype)
            if value is not None:
                K.assign(slot[0], value)
            return slot

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

        def replace_smem_desc_addr(desc, smem_ptr):
            start_addr = K.Cast(
                "uint64",
                K.bitwise_and(
                    K.shift_right(K.cuda.cvta_generic_to_shared(smem_ptr), K.uint32(4)),
                    K.uint32(0x3FFF),
                ),
            )
            return K.bitwise_or(K.bitwise_and(desc, K.bitwise_not(K.uint64(0x3FFF))), start_addr)

        def make_runtime_instr_desc_with_sf_id(desc, sfa_id, sfb_id):
            runtime_desc = K.bitwise_and(desc, K.uint32(0x9FFFFFCF))
            runtime_desc = K.bitwise_or(
                runtime_desc, K.shift_left(K.Cast("uint32", sfa_id), K.uint32(29))
            )
            runtime_desc = K.bitwise_or(
                runtime_desc, K.shift_left(K.Cast("uint32", sfb_id), K.uint32(4))
            )
            return K.shift_left(K.Cast("uint64", runtime_desc), K.uint64(32))

        def make_sf_desc(desc, smem_ptr):
            K.cuda.tcgen05.encode_matrix_descriptor(
                K.address_of(desc), smem_ptr, ldo=0, sdo=sf_desc_sdo, swizzle=0
            )

        def make_smem_desc(desc, smem_ptr):
            K.cuda.tcgen05.encode_matrix_descriptor(
                K.address_of(desc), smem_ptr, ldo=0, sdo=desc_sdo, swizzle=2
            )

        def mma_mxf4_block32_ss(desc_a, desc_b, tmem_c, scale_c, desc, tmem_sfa, tmem_sfb):
            i_desc_hi = K.Cast("uint32", K.shift_right(desc, K.uint64(32)))
            K.ptx["tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X"](
                tmem_c, desc_a, desc_b, i_desc_hi, tmem_sfa, tmem_sfb, K.ptx.pred(scale_c)
            )

        def utccp_required_smem_warp_transpose(buf, prefix, base_offset):
            values = K.alloc_local([4], "uint32")
            for i in range(4):
                i_u32 = local("uint32", K.uint32(i))
                col = local(
                    "uint32",
                    K.bitwise_xor(i_u32[0], K.shift_right(lane_idx_u32, K.uint32(3))) * K.uint32(32)
                    + lane_idx_u32,
                )
                K.ptx.ld.shared.u32(
                    values[i], buf.ptr_to([*prefix, K.Cast("int32", base_offset + col[0])])
                )
            K.cuda.warp_sync()
            for i in range(4):
                i_u32 = local("uint32", K.uint32(i))
                col = local(
                    "uint32",
                    lane_idx_u32 * K.uint32(4)
                    + K.bitwise_xor(i_u32[0], K.shift_right(lane_idx_u32, K.uint32(3))),
                )
                K.ptx.st.shared.u32(
                    buf.ptr_to([*prefix, K.Cast("int32", base_offset + col[0])]), values[i]
                )

        def load_num_kv(q_atom_idx_arg, runtime_batch_size_arg):
            # The one-page specialization has exactly one positive BLOCK_KV
            # page, so it intentionally performs no metadata read.
            if not config.varlen and config.max_num_pages == 1:
                K.assign(num_kv_result, K.uint32(1))
            elif config.varlen:
                context_idx = local("uint32", q_atom_idx_arg)
                with K.If(q_atom_idx_arg + K.uint32(1) < runtime_batch_size_arg), K.Then():
                    index_0 = K.local_scalar("int32")
                    index_1 = K.local_scalar("int32")
                    K.ptx.ld.global_.s32(index_0, indices.ptr_to([K.Cast("int32", q_atom_idx_arg)]))
                    K.ptx.ld.global_.s32(
                        index_1, indices.ptr_to([K.Cast("int32", q_atom_idx_arg + K.uint32(1))])
                    )
                    with K.If(index_0 == index_1), K.Then():
                        K.assign(context_idx[0], q_atom_idx_arg + K.uint32(1))
                context_len = K.local_scalar("uint32")
                K.ptx.ld.global_.u32(
                    context_len, context_lens_flat.ptr_to([K.Cast("int32", context_idx[0])])
                )
                K.assign(num_kv_result, (context_len + K.uint32(umma_m - 1)) // K.uint32(umma_m))
            else:
                if num_next_n_atoms == 1:
                    q_idx = local("uint32", q_atom_idx_arg)
                else:
                    q_idx = local("uint32", q_atom_idx_arg // K.uint32(num_next_n_atoms))
                lens_idx = local(
                    "uint32", q_idx[0] * K.uint32(config.next_n) + K.uint32(config.next_n - 1)
                )
                context_len = K.local_scalar("uint32")
                K.ptx.ld.global_.u32(
                    context_len, context_lens_flat.ptr_to([K.Cast("int32", lens_idx[0])])
                )
                K.assign(num_kv_result, (context_len + K.uint32(umma_m - 1)) // K.uint32(umma_m))

        def load_atom_advance(q_atom_idx_arg, bound_arg):
            K.assign(atom_advance_result, K.uint32(1))
            if config.varlen:
                with K.If(q_atom_idx_arg + K.uint32(1) < bound_arg), K.Then():
                    index_0 = K.local_scalar("int32")
                    index_1 = K.local_scalar("int32")
                    K.ptx.ld.global_.s32(index_0, indices.ptr_to([K.Cast("int32", q_atom_idx_arg)]))
                    K.ptx.ld.global_.s32(
                        index_1, indices.ptr_to([K.Cast("int32", q_atom_idx_arg + K.uint32(1))])
                    )
                    with K.If(index_0 == index_1), K.Then():
                        K.assign(atom_advance_result, K.uint32(2))

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

        def issue_tma_q(stage_idx, tma_q_atom_idx):
            with K.If(K.cuda.elect_sync()), K.Then():
                q_token_idx = local("uint32", atom_to_token_idx_expr(tma_q_atom_idx))
                K.ptx[TMA_G2S_2D](
                    smem_q.ptr_to([stage_idx, 0, 0]),
                    K.address_of(tensor_map_q),
                    K.Cast("int32", K.uint32(0)),
                    K.Cast("int32", q_token_idx[0] * K.uint32(num_heads)),
                    full_q_barriers.ptr_to([stage_idx]),
                    cache_policy_evict_normal,
                )
                K.ptx[TMA_G2S_2D](
                    smem_sf_q.ptr_to([0, stage_idx, 0]),
                    K.address_of(tensor_map_sf_q),
                    K.Cast("int32", K.uint32(0)),
                    K.Cast("int32", q_token_idx[0]),
                    full_q_barriers.ptr_to([stage_idx]),
                    cache_policy_evict_normal,
                )
                K.ptx[TMA_G2S_2D](
                    smem_sf_q.ptr_to([1, stage_idx, 0]),
                    K.address_of(tensor_map_sf_q),
                    K.Cast("int32", K.uint32(0)),
                    K.Cast("int32", q_token_idx[0]),
                    full_q_barriers.ptr_to([stage_idx]),
                    cache_policy_evict_normal,
                )
                K.ptx[TMA_G2S_2D](
                    smem_weights.ptr_to([stage_idx, 0, 0]),
                    K.address_of(tensor_map_weights),
                    K.Cast("int32", K.uint32(0)),
                    K.Cast("int32", q_token_idx[0]),
                    full_q_barriers.ptr_to([stage_idx]),
                    cache_policy_evict_normal,
                )
                full_q_barriers.arrive(
                    stage_idx,
                    tx_count=(
                        smem_q_size_per_stage
                        + real_num_sfq_atom * 4 * 2
                        + smem_weight_size_per_stage
                    ),
                )

        # ---------------- CTA-scope prologue ----------------------------
        # Early schedule-metadata load: issue the global loads before the
        # pipeline/barrier prologue so the ~200-cycle L2 latency overlaps setup.
        # The original match_buffers schedule_meta as int32 and reads it through
        # a uint32 decl_buffer view; its generated CUDA carries the
        # `((uint*)schedule_meta_ptr)` cast. Same view here, over the flat gptr.
        schedule_meta_u32_flat = K.decl_buffer(
            ((config.num_sms + 1) * 2,),
            "uint32",
            data=schedule_meta_flat.data,
            scope="global",
            elem_offset=0,
        )
        start_q_atom_idx = K.local_scalar("uint32")
        start_kv_tile_idx = K.local_scalar("uint32")
        end_q_atom_idx = K.local_scalar("uint32")
        end_kv_tile_idx = K.local_scalar("uint32")
        K.ptx.ld.global_.u32(
            start_q_atom_idx,
            schedule_meta_u32_flat.ptr_to([K.Cast("int32", sm_idx_u32 * K.uint32(2))]),
        )
        K.ptx.ld.global_.u32(
            start_kv_tile_idx,
            schedule_meta_u32_flat.ptr_to(
                [K.Cast("int32", sm_idx_u32 * K.uint32(2) + K.uint32(1))]
            ),
        )
        K.ptx.ld.global_.u32(
            end_q_atom_idx,
            schedule_meta_u32_flat.ptr_to(
                [K.Cast("int32", (sm_idx_u32 + K.uint32(1)) * K.uint32(2))]
            ),
        )
        K.ptx.ld.global_.u32(
            end_kv_tile_idx,
            schedule_meta_u32_flat.ptr_to(
                [K.Cast("int32", (sm_idx_u32 + K.uint32(1)) * K.uint32(2) + K.uint32(1))]
            ),
        )
        start_kv_idx = start_kv_tile_idx * K.uint32(num_tiles_per_split)
        # Clamp the context-length read for zero-work CTAs (start == total q
        # atoms); the value is stale but never used, because has_work is false.
        load_num_kv(
            K.min(start_q_atom_idx, batch_size * K.uint32(num_next_n_atoms) - K.uint32(1)),
            batch_size,
        )
        start_num_kv = num_kv_result

        # Warm the block table into L2 as early as possible. Race-safe: a stale
        # prefetched line is invalidated by any later producer write, so the PDL
        # contract is unaffected. Guarded by `warp_idx`, not the opaque copy --
        # that is the original's own choice and it is preserved.
        with K.If(K.Or(warp_idx == tma_warp_0, warp_idx == tma_warp_1)), K.Then():
            for pf_i in range((num_prefetch_lines + 63) // 64):
                line_idx = local(
                    "uint32",
                    (warp_idx_u32 - K.uint32(tma_warp_0)) * K.uint32(32)
                    + lane_idx_u32
                    + K.uint32(pf_i * 64),
                )
                with K.If(line_idx[0] < K.uint32(num_prefetch_lines)), K.Then():
                    K.ptx.prefetch.global_.L2(
                        block_table_flat.ptr_to([K.Cast("int64", line_idx[0] * K.uint32(32))])
                    )

        # Four barrier-init blocks on the OPAQUE warp id. Note the first tests
        # the same warp as the tensormap-prefetch block above and is still a
        # separate `if`: that is the original's shape.
        with K.If(warp_idx_presync == tma_warp_0), K.Then():
            with K.If(K.cuda.elect_sync()), K.Then():
                for init_i in range(num_q_stages):
                    K.ptx.mbarrier.init.shared.b64(full_q_barriers.ptr_to([init_i]), K.uint32(1))
                    K.ptx.mbarrier.init.shared.b64(
                        empty_q_barriers.ptr_to([init_i]), K.uint32(num_math_threads)
                    )
                K.ptx.fence.mbarrier_init.release.cluster()
        with K.If(warp_idx_presync == tma_warp_1), K.Then():
            with K.If(K.cuda.elect_sync()), K.Then():
                for init_i in range(num_kv_stages):
                    K.ptx.mbarrier.init.shared.b64(full_kv_barriers.ptr_to([init_i]), K.uint32(1))
                    K.ptx.mbarrier.init.shared.b64(empty_kv_barriers.ptr_to([init_i]), K.uint32(1))
                K.ptx.fence.mbarrier_init.release.cluster()
        with K.If(warp_idx_presync == umma_warp_0), K.Then():
            with K.If(K.cuda.elect_sync()), K.Then():
                for init_i in range(num_kv_stages):
                    K.ptx.mbarrier.init.shared.b64(
                        full_kv_barriers.ptr_to([num_kv_stages + init_i]), K.uint32(1)
                    )
                    K.ptx.mbarrier.init.shared.b64(
                        empty_kv_barriers.ptr_to([num_kv_stages + init_i]), K.uint32(1)
                    )
                K.ptx.fence.mbarrier_init.release.cluster()
        with K.If(warp_idx_presync == umma_warp_0 + 1), K.Then():
            with K.If(K.cuda.elect_sync()), K.Then():
                for init_i in range(num_tmem_barriers):
                    K.ptx.mbarrier.init.shared.b64(full_tmem_barriers.ptr_to([init_i]), K.uint32(1))
                    K.ptx.mbarrier.init.shared.b64(
                        empty_tmem_barriers.ptr_to([init_i]), K.uint32(128)
                    )
                K.ptx.fence.mbarrier_init.release.cluster()
            # Warp-wide, OUTSIDE the elect: the original's placement.
            K.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                K.address_of(tmem_ptr_in_smem[0]), K.uint32(num_tmem_cols)
            )
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
            """Task/block-table state only; protocol cursors are K.RingState."""
            state = {
                name: K.alloc_local([1], "uint32")
                for name in (
                    "cur_q",
                    "cur_kv",
                    "cur_num_kv",
                    "q_atom",
                    "kv_idx",
                    "num_kv",
                    "next_q",
                    "next_kv",
                    "next_num_kv",
                )
            }
            K.assign(state["cur_q"][0], start_q_atom_idx)
            K.assign(state["cur_kv"][0], start_kv_idx)
            K.assign(state["cur_num_kv"][0], start_num_kv)
            K.assign(state["q_atom"][0], batch_size * K.uint32(num_next_n_atoms))
            K.assign(state["kv_idx"][0], K.uint32(0))
            K.assign(state["num_kv"][0], K.uint32(0))
            K.assign(state["next_q"][0], state["cur_q"][0])
            K.assign(state["next_kv"][0], state["cur_kv"][0])
            K.assign(state["next_num_kv"][0], state["cur_num_kv"][0])
            return state

        def pump(state, end_kv, fetched=None):
            """One `fetch_next_task` + write-back, exactly as the original does.

            The `next_* = cur_*` triple is emitted only on the in-loop calls.
            Each role's *first* `fetch_next_task` is not preceded by it in the
            original -- the initial values were just written by the declarations
            -- and adding it there would be twelve dead self-assignments the
            statement census sees (4 roles x 3).
            """
            if fetched is not None:
                K.assign(state["next_q"][0], state["cur_q"][0])
                K.assign(state["next_kv"][0], state["cur_kv"][0])
                K.assign(state["next_num_kv"][0], state["cur_num_kv"][0])
            fetch_next_task(
                state["cur_q"][0],
                state["cur_kv"][0],
                state["cur_num_kv"][0],
                end_q_atom_idx,
                end_kv,
            )
            K.assign(state["next_q"][0], scheduler_result[0])
            K.assign(state["next_kv"][0], scheduler_result[1])
            K.assign(state["next_num_kv"][0], scheduler_result[2])
            if fetched is None:
                fetched = local("bool", scheduler_result[3] != K.uint32(0))
            else:
                K.assign(fetched[0], scheduler_result[3] != K.uint32(0))
            K.assign(state["cur_q"][0], scheduler_result[4])
            K.assign(state["cur_kv"][0], scheduler_result[5])
            K.assign(state["cur_num_kv"][0], scheduler_result[6])
            return fetched

        def load_block_table(state, kv_ptr, cached, lane_offset_tiles):
            """Block-table gather + the warp broadcast, at the original's own
            position inside the task loop. The __shfl_sync is a value-returning
            warp collective in a loop: G3 forbids moving it."""
            with K.If(kv_ptr[0] == K.uint32(32)), K.Then():
                K.assign(kv_ptr[0], K.uint32(0))
                block_table_offset = local(
                    "uint64",
                    K.Cast("uint64", atom_to_block_table_row_expr(state["q_atom"][0]))
                    * K.Cast("uint64", block_table_stride),
                )
                prefetch_tile_idx = local(
                    "uint32",
                    state["kv_idx"][0]
                    + K.uint32(lane_offset_tiles)
                    + lane_idx_u32 * K.uint32(num_tiles_per_split),
                )
                block_table_index = local(
                    "uint64",
                    block_table_offset[0]
                    + K.Cast("uint64", prefetch_tile_idx[0] * K.uint32(num_pages_per_tile)),
                )
                for block_i in range(num_pages_per_tile):
                    # Guard the trailing partial tile: a valid compute tile may
                    # still exceed the block table's row length, and an
                    # out-of-range garbage page id would send TMA out of bounds
                    # (page 0 is used as the masked-dumpster tile).
                    with K.If(
                        K.And(
                            prefetch_tile_idx[0] < state["num_kv"][0],
                            prefetch_tile_idx[0] * K.uint32(num_pages_per_tile) + K.uint32(block_i)
                            < K.uint32(config.max_num_pages),
                        )
                    ):
                        with K.Then():
                            K.ptx.ld.global_.u32(
                                cached[block_i],
                                block_table_flat.ptr_to(
                                    [
                                        K.Cast(
                                            "int64",
                                            block_table_index[0] + K.Cast("uint64", block_i),
                                        )
                                    ]
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
            kv_stage_idx = K.local_scalar("uint32", init=kv_state.stage)
            kv_phase = K.local_scalar("uint32", init=kv_state.phase)
            kv_state.advance()
            base = K.uint32(group * num_kv_stages)
            empty_kv_barriers.wait(base + kv_stage_idx, kv_phase ^ K.uint32(1))
            with K.If(K.cuda.elect_sync()), K.Then():
                for block_i in range(num_pages_per_tile):
                    K.ptx[TMA_G2S_3D](
                        smem_kv.ptr_to([group, kv_stage_idx, block_i * page_size, 0]),
                        K.address_of(tensor_map_kv),
                        K.Cast("int32", K.uint32(0)),
                        K.Cast("int32", K.uint32(0)),
                        K.Cast("int32", kv_block_idx[block_i]),
                        full_kv_barriers.ptr_to([base + kv_stage_idx]),
                        cache_policy_evict_normal,
                    )
                    K.ptx[TMA_G2S_2D](
                        smem_sf_kv.ptr_to([group, kv_stage_idx, block_i * page_size]),
                        K.address_of(tensor_map_sf_kv),
                        K.Cast("int32", K.uint32(0)),
                        K.Cast("int32", kv_block_idx[block_i]),
                        full_kv_barriers.ptr_to([base + kv_stage_idx]),
                        cache_policy_evict_normal,
                    )
                full_kv_barriers.arrive(
                    base + kv_stage_idx, tx_count=smem_kv_size_per_stage + smem_sf_kv_size_per_stage
                )

        # ---------------- warp 8: Q + SFQ + weights, and KV for group 0 --
        with tma0:
            tma0_end_kv_idx = end_kv_tile_idx * K.uint32(num_tiles_per_split)
            state = scheduler_state()
            q_state = K.RingState(num_q_stages)
            kv_state = K.RingState(num_kv_stages)
            fetched = pump(state, tma0_end_kv_idx)
            with K.If(fetched[0]), K.Then():
                issue_tma_q(q_state.stage, state["next_q"][0])
                q_state.advance()
            kv_ptr = local("uint32", K.uint32(32))
            cached = K.alloc_local([num_pages_per_tile], "uint32")
            with K.While(fetched[0]):
                load_atom_advance(state["next_q"][0], batch_size)
                next_advance = local("uint32", atom_advance_result)
                prefetch_q = local(
                    "bool",
                    K.And(
                        state["q_atom"][0] != state["next_q"][0],
                        exist_q_atom_idx_expr(
                            state["next_q"][0] + next_advance[0], end_q_atom_idx, tma0_end_kv_idx
                        ),
                    ),
                )
                with K.If(state["q_atom"][0] != state["next_q"][0]), K.Then():
                    K.assign(kv_ptr[0], K.uint32(32))
                K.assign(state["q_atom"][0], state["next_q"][0])
                K.assign(state["kv_idx"][0], state["next_kv"][0])
                K.assign(state["num_kv"][0], state["next_num_kv"][0])

                with K.If(prefetch_q[0]), K.Then():
                    empty_q_barriers.wait(q_state.stage, q_state.phase ^ K.uint32(1))
                    issue_tma_q(q_state.stage, state["q_atom"][0] + next_advance[0])
                    q_state.advance()

                kv_block_idx = load_block_table(state, kv_ptr, cached, 0)
                issue_kv_tma(0, kv_state, kv_block_idx)
                pump(state, tma0_end_kv_idx, fetched)

        # ---------------- warp 9: KV for group 1 -------------------------
        with tma1:
            tma1_end_kv_idx = end_kv_tile_idx * K.uint32(num_tiles_per_split)
            state = scheduler_state()
            kv_state = K.RingState(num_kv_stages)
            fetched = pump(state, tma1_end_kv_idx)
            kv_ptr = local("uint32", K.uint32(32))
            cached = K.alloc_local([num_pages_per_tile], "uint32")
            with K.While(fetched[0]):
                with K.If(state["q_atom"][0] != state["next_q"][0]), K.Then():
                    K.assign(kv_ptr[0], K.uint32(32))
                K.assign(state["q_atom"][0], state["next_q"][0])
                K.assign(state["kv_idx"][0], state["next_kv"][0])
                K.assign(state["num_kv"][0], state["next_num_kv"][0])
                kv_block_idx = load_block_table(state, kv_ptr, cached, 1)
                issue_kv_tma(1, kv_state, kv_block_idx)
                pump(state, tma1_end_kv_idx, fetched)

        # ---------------- warps 10-11: UMMA + UTCCP issuers --------------
        with umma:
            umma_end_kv_idx = end_kv_tile_idx * K.uint32(num_tiles_per_split)
            umma_group_idx = warp_idx_u32 - K.uint32(umma_warp_0)
            tmem_allocated = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(tmem_allocated, tmem_ptr_in_smem.ptr_to([0]))
            K.cuda.trap_when_assert_failed(tmem_allocated == K.uint32(accum_tmem_col))
            desc_i = K.local_scalar("uint32")
            desc_sf = K.local_scalar("uint64")
            desc_a = K.local_scalar("uint64")
            desc_b = K.local_scalar("uint64")
            K.cuda.tcgen05.encode_instr_descriptor_block_scaled(
                K.address_of(desc_i),
                d_dtype="float32",
                a_dtype="float4_e2m1fn",
                b_dtype="float4_e2m1fn",
                sfa_dtype="float8_e8m0fnu",
                sfb_dtype="float8_e8m0fnu",
                sfa_tmem_addr=0,
                sfb_tmem_addr=0,
                M=umma_m,
                N=umma_n,
                K=umma_k,
                trans_a=False,
                trans_b=False,
                n_cta_groups=1,
            )
            make_sf_desc(desc_sf, K.reinterpret("handle", K.uint64(0)))

            state = scheduler_state()
            q_state = K.RingState(num_q_stages)
            kv_state = K.RingState(num_kv_stages)
            tmem_state = K.RingState(num_tmem_stages)
            q_stage = local("uint32", K.uint32(0))
            fetched = pump(state, umma_end_kv_idx)
            with K.While(fetched[0]):
                with K.If(state["q_atom"][0] != state["next_q"][0]), K.Then():
                    # Wait for the new Q stage (wait only; Math releases it),
                    # then copy this group's Q scale factors into its own TMEM
                    # sfb region (duplicated per group to stay cross-warp-free).
                    K.assign(q_stage[0], q_state.stage)
                    q_phase = K.local_scalar("uint32", init=q_state.phase)
                    q_state.advance()
                    full_q_barriers.wait(q_stage[0], q_phase)
                    for sfq_i in range(num_sfq_atom // num_utccp_aligned_elems):
                        sfq_base = local("uint32", K.uint32(sfq_i * num_utccp_aligned_elems))
                        utccp_required_smem_warp_transpose(
                            smem_sf_q, [umma_group_idx, q_stage[0]], sfq_base[0]
                        )
                        K.ptx.fence.proxy.async_.shared__cta()
                        K.assign(
                            desc_sf,
                            replace_smem_desc_addr(
                                desc_sf, smem_sf_q.ptr_to([umma_group_idx, q_stage[0], sfq_base[0]])
                            ),
                        )
                        with K.If(K.cuda.elect_sync()), K.Then():
                            K.ptx[TCGEN05_CP](
                                K.Cast(
                                    "uint32",
                                    K.uint32(sfq_tmem_col)
                                    + umma_group_idx * K.uint32(num_sfq_atom // 32)
                                    + sfq_i * 4,
                                ),
                                desc_sf,
                            )
                        K.cuda.warp_sync()
                K.assign(state["q_atom"][0], state["next_q"][0])
                K.assign(state["kv_idx"][0], state["next_kv"][0])

                kv_stage_idx = K.local_scalar("uint32", init=kv_state.stage)
                kv_phase = K.local_scalar("uint32", init=kv_state.phase)
                kv_state.advance()
                full_kv_barriers.wait(
                    umma_group_idx * K.uint32(num_kv_stages) + kv_stage_idx, kv_phase
                )
                for sfkv_i in range(num_sfkv // num_utccp_aligned_elems):
                    sfkv_base = local("uint32", K.uint32(sfkv_i * num_utccp_aligned_elems))
                    utccp_required_smem_warp_transpose(
                        smem_sf_kv, [umma_group_idx, kv_stage_idx], sfkv_base[0]
                    )
                    K.ptx.fence.proxy.async_.shared__cta()
                with K.If(K.cuda.elect_sync()), K.Then():
                    for sfkv_i in range(num_sfkv // num_utccp_aligned_elems):
                        sfkv_base = local("uint32", K.uint32(sfkv_i * num_utccp_aligned_elems))
                        K.assign(
                            desc_sf,
                            replace_smem_desc_addr(
                                desc_sf,
                                smem_sf_kv.ptr_to([umma_group_idx, kv_stage_idx, sfkv_base[0]]),
                            ),
                        )
                        K.ptx[TCGEN05_CP](
                            K.Cast(
                                "uint32",
                                K.uint32(sfkv_tmem_col)
                                + umma_group_idx * K.uint32(num_sfkv // 32)
                                + sfkv_i * 4,
                            ),
                            desc_sf,
                        )

                tmem_stage_idx = K.local_scalar("uint32", init=tmem_state.stage)
                tmem_phase = K.local_scalar("uint32", init=tmem_state.phase)
                tmem_state.advance()
                empty_tmem_barriers.wait(
                    umma_group_idx * K.uint32(num_tmem_stages) + tmem_stage_idx,
                    tmem_phase ^ K.uint32(1),
                )
                K.ptx.tcgen05.fence__after_thread_sync()
                tmem_addr = local(
                    "uint32",
                    K.uint32(accum_tmem_col)
                    + umma_group_idx * K.uint32(umma_n * num_tmem_stages)
                    + tmem_stage_idx * K.uint32(umma_n),
                )
                # G3, LAW: elect_sync wraps the whole k loop here and the
                # descriptor recompute is INSIDE it -- the original's placement,
                # preserved verbatim and never normalised to the fp8 sibling's
                # per-MMA elect form.
                with K.If(K.cuda.elect_sync()), K.Then():
                    for k in range(head_dim // umma_k):
                        runtime_desc_i = local(
                            "uint64", make_runtime_instr_desc_with_sf_id(desc_i, k * 2, k * 2)
                        )
                        make_smem_desc(
                            desc_a,
                            smem_kv.ptr_to([umma_group_idx, kv_stage_idx, 0, k * umma_k // 2]),
                        )
                        make_smem_desc(desc_b, smem_q.ptr_to([q_stage[0], 0, k * umma_k // 2]))
                        mma_mxf4_block32_ss(
                            desc_a,
                            desc_b,
                            tmem_addr[0],
                            K.uint32(k),
                            runtime_desc_i[0],
                            K.uint32(sfkv_tmem_col) + umma_group_idx * K.uint32(num_sfkv // 32),
                            K.uint32(sfq_tmem_col) + umma_group_idx * K.uint32(num_sfq_atom // 32),
                        )
                with K.If(K.cuda.elect_sync()), K.Then():
                    full_tmem_barriers.arrive(
                        umma_group_idx * K.uint32(num_tmem_stages) + tmem_stage_idx, cta_group=1
                    )
                    # Release the KV stage once the MMAs consuming it complete
                    # (Math never reads KV SMEM; the commit tracks it).
                    empty_kv_barriers.arrive(
                        umma_group_idx * K.uint32(num_kv_stages) + kv_stage_idx, cta_group=1
                    )
                pump(state, umma_end_kv_idx, fetched)

        # ---------------- warps 0-7: math + epilogue ---------------------
        with math:
            math_end_kv_idx = end_kv_tile_idx * K.uint32(num_tiles_per_split)
            state = scheduler_state()
            q_state = K.RingState(num_q_stages)
            tmem_state = K.RingState(num_tmem_stages)
            q_stage = local("uint32", K.uint32(0))
            has_q_stage = local("bool", K.bool(False))
            math_wg_idx = local("int32", warpgroup_idx)
            math_wg_u32 = K.Cast("uint32", math_wg_idx[0])
            math_thread_idx = local(
                "uint32", (warp_idx_u32 % K.uint32(4)) * K.uint32(32) + lane_idx_u32
            )
            accum = K.alloc_local([num_heads], "float32")
            cached_weights = K.alloc_local([next_n_atom, num_heads], "float32")
            fetched = pump(state, math_end_kv_idx)
            is_paired_atom = local("bool", K.bool(False))

            def reduce_and_store(num_iters_c, kv_offset_arg, tmem_stage_idx_arg):
                for q_inner_i in range(num_iters_c):
                    tmem_addr = local(
                        "uint32",
                        K.uint32(accum_tmem_col)
                        + math_wg_u32 * K.uint32(umma_n * num_tmem_stages)
                        + tmem_stage_idx_arg * K.uint32(umma_n)
                        + K.uint32(q_inner_i * num_heads),
                    )
                    K.ptx[TC_LD](*[accum[h] for h in range(num_heads)], K.uint32(tmem_addr[0]))
                    K.ptx.tcgen05.wait__ld.sync.aligned()
                    if q_inner_i == num_iters_c - 1:
                        # Release the TMEM stage right after the last TMEM load
                        # so the next MMA can start while the FMA chain and the
                        # store are still running.
                        K.ptx.tcgen05.fence__before_thread_sync()
                        empty_tmem_barriers.arrive(
                            math_wg_u32 * K.uint32(num_tmem_stages) + tmem_stage_idx_arg
                        )
                    sum_0 = local("uint64", K.cuda.make_float2(K.float32(0), K.float32(0)))
                    sum_1 = local("uint64", K.cuda.make_float2(K.float32(0), K.float32(0)))
                    for head_j_group in range(num_heads // 4):
                        # A local, not a Python int: `reduce_and_store` is
                        # `@K.inline` in the original, so TVMScript binds this
                        # unannotated assignment as a let and it appears as
                        # `head_j_ptr[0]` in the generated CUDA (16 sites).
                        head_j = local("int32", K.int32(head_j_group * 4))
                        K.assign(
                            sum_0[0],
                            relu2_fma_f32x2(
                                K.cuda.make_float2(accum[head_j[0]], accum[head_j[0] + 1]),
                                K.cuda.make_float2(
                                    cached_weights[q_inner_i, head_j[0]],
                                    cached_weights[q_inner_i, head_j[0] + 1],
                                ),
                                sum_0[0],
                            ),
                        )
                        K.assign(
                            sum_1[0],
                            relu2_fma_f32x2(
                                K.cuda.make_float2(accum[head_j[0] + 2], accum[head_j[0] + 3]),
                                K.cuda.make_float2(
                                    cached_weights[q_inner_i, head_j[0] + 2],
                                    cached_weights[q_inner_i, head_j[0] + 3],
                                ),
                                sum_1[0],
                            ),
                        )
                    sum_v = K.local_scalar("uint64")
                    K.ptx.add.rn.f32x2(sum_v, sum_0[0], sum_1[0])
                    # The 0.5 completes relu(x) = (x + |x|) * 0.5, folded across
                    # the packed-f32x2 ReLU accumulation in relu2_fma_f32x2.
                    _add = K.local_scalar("float32")
                    K.ptx.add.rn.f32(_add, K.cuda.float2_x(sum_v), K.cuda.float2_y(sum_v))
                    result_f32 = K.local_scalar("float32")
                    K.ptx.mul.rn.f32(result_f32, K.float32(0.5), _add)
                    result = local(logits_tir_dtype, K.Cast(logits_tir_dtype, result_f32))
                    logits_offset = local(
                        "uint64",
                        K.Cast("uint64", kv_offset_arg)
                        + K.Cast("uint64", q_inner_i) * K.Cast("uint64", logits_stride),
                    )
                    if config.logits_dtype == "float32":
                        K.ptx.st.global_.f32(logits_flat.ptr_to([logits_offset[0]]), result[0])
                    else:
                        K.ptx.st.global_.b16(logits_flat.ptr_to([logits_offset[0]]), result[0])

            with K.While(fetched[0]):
                with K.If(state["q_atom"][0] != state["next_q"][0]), K.Then():
                    with K.If(has_q_stage[0]), K.Then():
                        empty_q_barriers.arrive(q_state.stage)
                        q_state.advance()
                    K.assign(q_stage[0], q_state.stage)
                    q_phase = q_state.phase
                    full_q_barriers.wait(q_stage[0], q_phase)
                    K.assign(has_q_stage[0], K.bool(True))
                    for weight_i in range(next_n_atom):
                        for weight_j in range(num_heads // 4):
                            weight_col = local("int32", K.int32(weight_j * 4))
                            K.ptx.ld.shared.v4.f32(
                                cached_weights[weight_i, weight_col[0]],
                                cached_weights[weight_i, weight_col[0] + 1],
                                cached_weights[weight_i, weight_col[0] + 2],
                                cached_weights[weight_i, weight_col[0] + 3],
                                smem_weights.ptr_to([q_stage[0], weight_i, weight_col[0]]),
                            )
                    if config.varlen:
                        load_atom_advance(state["next_q"][0], batch_size)
                        K.assign(is_paired_atom[0], atom_advance_result == K.uint32(2))
                K.assign(state["q_atom"][0], state["next_q"][0])
                kv_idx = local("uint32", state["next_kv"][0])
                kv_offset = local(
                    "uint64",
                    K.Cast("uint64", atom_to_token_idx_expr(state["q_atom"][0]))
                    * K.Cast("uint64", logits_stride)
                    + K.Cast("uint64", (kv_idx[0] + math_wg_u32) * K.uint32(umma_m))
                    + K.Cast("uint64", math_thread_idx[0]),
                )
                tmem_stage_idx = K.local_scalar("uint32", init=tmem_state.stage)
                tmem_phase = K.local_scalar("uint32", init=tmem_state.phase)
                tmem_state.advance()
                full_tmem_barriers.wait(
                    math_wg_u32 * K.uint32(num_tmem_stages) + tmem_stage_idx, tmem_phase
                )
                K.ptx.tcgen05.fence__after_thread_sync()
                if config.varlen:
                    with K.If(is_paired_atom[0]):
                        with K.Then():
                            reduce_and_store(next_n_atom, kv_offset[0], tmem_stage_idx)
                        with K.Else():
                            reduce_and_store(1, kv_offset[0], tmem_stage_idx)
                elif k_pad_odd_n:
                    with K.If(
                        state["q_atom"][0] % K.uint32(num_next_n_atoms)
                        == K.uint32(num_next_n_atoms - 1)
                    ):
                        with K.Then():
                            reduce_and_store(1, kv_offset[0], tmem_stage_idx)
                        with K.Else():
                            reduce_and_store(next_n_atom, kv_offset[0], tmem_stage_idx)
                else:
                    reduce_and_store(next_n_atom, kv_offset[0], tmem_stage_idx)
                pump(state, math_end_kv_idx, fetched)
            K.ptx.griddepcontrol.launch_dependents()
            K.ptx.bar.sync(8, K.uint32(num_math_threads))
            with K.If(warp_idx == 0), K.Then():
                K.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                    K.uint32(accum_tmem_col), K.uint32(num_tmem_cols)
                )

    # `@K.kernel` has no `attrs=`. The paged original sets ONLY
    # kernel_launch_params -- no tirx.persistent_kernel.
    sm100_fp4_paged_mqa_logits.func = sm100_fp4_paged_mqa_logits.func.with_attr(
        "tirx.kernel_launch_params",
        [
            "blockIdx.x",
            "threadIdx.x",
            "tirx.use_programtic_dependent_launch",
            "tirx.use_dyn_shared_memory",
        ],
    )
    return sm100_fp4_paged_mqa_logits.func


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
    from tirx_kernels.runner import cuda_target

    target = cuda_target()
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
    previous_postproc = tvm.get_global_func("tvm_callback_cuda_postproc", allow_missing=True)

    @tvm.register_global_func("tvm_callback_cuda_postproc", override=True)
    def _postproc(code: str, target: Any) -> str:
        if previous_postproc is not None:
            code = previous_postproc(code, target)
        return _paged_mqa_logits_fp4_cuda_postproc(code)

    try:
        with target:
            mod = tvm.IRModule({"main": kernel})
            return tvm.compile(mod, target=target, tir_pipeline="tirx")
    finally:
        if previous_postproc is not None:
            tvm.register_global_func("tvm_callback_cuda_postproc", previous_postproc, override=True)
        else:
            tvm.register_global_func(
                "tvm_callback_cuda_postproc", lambda code, target: code, override=True
            )


_compile_tirx_paged_mqa_for_config = cache(_compile_tirx_paged_mqa_for_config)


def _compile_tirx_paged_mqa_kwargs(config: PagedMQALogitsFP4Config) -> dict[str, Any]:
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


def _compile_tirx_paged_mqa_key(config: PagedMQALogitsFP4Config) -> tuple[tuple[str, Any], ...]:
    return tuple(_compile_tirx_paged_mqa_kwargs(config).items())


def _compile_tirx_paged_mqa(config: PagedMQALogitsFP4Config) -> Any:
    compile_kwargs = _compile_tirx_paged_mqa_kwargs(config)
    return _compile_tirx_paged_mqa_for_config(**compile_kwargs)


def _run_deepgemm_paged_mqa(data: dict[str, Any], *, clean_logits: bool = False) -> torch.Tensor:
    config: PagedMQALogitsFP4Config = data["config"]
    return data["deep_gemm"].fp8_fp4_paged_mqa_logits(
        q=data["q_in"],
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


def _allocate_logits(config: PagedMQALogitsFP4Config) -> torch.Tensor:
    return torch.full(
        (config.batch_size * config.next_n, config.logits_stride),
        float("-inf"),
        device="cuda",
        dtype=_torch_logits_dtype(config.logits_dtype),
    )


def _encode_fp4_packed_smem_tma_2d_desc(
    *,
    tensor: torch.Tensor,
    gmem_inner_dim: int,
    gmem_outer_dim: int,
    smem_inner_dim: int,
    smem_outer_dim: int,
    gmem_outer_stride: int,
    swizzle_mode: int,
) -> Any:
    from tirx_kernels.deepgemm._sm100_fp8_fp4_mega_moe import spec as mega_moe

    desc = mega_moe._AlignedTensorMap()
    global_shape = (ctypes.c_uint64 * 2)(int(gmem_inner_dim), int(gmem_outer_dim))
    global_strides = (ctypes.c_uint64 * 1)(int(gmem_outer_stride * tensor.element_size()))
    box_dim = (ctypes.c_uint32 * 2)(int(smem_inner_dim), int(smem_outer_dim))
    element_strides = (ctypes.c_uint32 * 2)(1, 1)
    result = mega_moe._get_cuda_driver().cuTensorMapEncodeTiled(
        desc.ptr,
        13,
        ctypes.c_uint32(2),
        ctypes.c_void_p(int(tensor.data_ptr())),
        global_shape,
        global_strides,
        box_dim,
        element_strides,
        mega_moe._CUDA_TENSOR_MAP_INTERLEAVE_NONE,
        mega_moe._tensor_map_swizzle_from_mode(swizzle_mode),
        mega_moe._CUDA_TENSOR_MAP_L2_PROMOTION_L2_256B,
        mega_moe._CUDA_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed for FP4 align8 with CUresult={result}")
    return desc


def _encode_fp4_packed_smem_tma_3d_desc(
    *,
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
) -> Any:
    from tirx_kernels.deepgemm._sm100_fp8_fp4_mega_moe import spec as mega_moe

    desc = mega_moe._AlignedTensorMap()
    elem_size = int(tensor.element_size())
    global_shape = (ctypes.c_uint64 * 3)(
        int(gmem_inner_dim), int(gmem_mid_dim), int(gmem_outer_dim)
    )
    global_strides = (ctypes.c_uint64 * 2)(
        int(gmem_mid_stride * elem_size), int(gmem_outer_stride * elem_size)
    )
    box_dim = (ctypes.c_uint32 * 3)(int(smem_inner_dim), int(smem_mid_dim), int(smem_outer_dim))
    element_strides = (ctypes.c_uint32 * 3)(1, 1, 1)
    result = mega_moe._get_cuda_driver().cuTensorMapEncodeTiled(
        desc.ptr,
        13,
        ctypes.c_uint32(3),
        ctypes.c_void_p(int(tensor.data_ptr())),
        global_shape,
        global_strides,
        box_dim,
        element_strides,
        mega_moe._CUDA_TENSOR_MAP_INTERLEAVE_NONE,
        mega_moe._tensor_map_swizzle_from_mode(swizzle_mode),
        mega_moe._CUDA_TENSOR_MAP_L2_PROMOTION_L2_256B,
        mega_moe._CUDA_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != 0:
        raise RuntimeError(
            f"cuTensorMapEncodeTiled failed for FP4 3D align8 with CUresult={result}"
        )
    return desc


def _build_tirx_tensor_maps(data: dict[str, Any]) -> dict[str, Any]:
    import tvm
    from tirx_kernels.deepgemm._sm100_fp8_fp4_mega_moe.spec import _encode_tma_2d_desc

    config: PagedMQALogitsFP4Config = data["config"]
    q_fp4, sf_q = data["q_in"]
    fused = data["fused_kv_cache"]
    weights = data["weights"]
    encode_tensormap = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    kv_flat = fused.view(torch.uint8).view(
        config.num_pages, config.page_size * (config.head_dim // 2 + 4)
    )
    kv_fp4 = kv_flat[:, : config.page_size * config.head_dim // 2].reshape(
        config.num_pages, config.page_size, config.head_dim // 2
    )
    sf_kv = kv_flat[:, config.page_size * config.head_dim // 2 :].view(torch.int32)
    next_n_atom = 2 if (config.varlen or config.next_n >= 2) else 1

    return {
        "tensor_map_q": _encode_fp4_packed_smem_tma_2d_desc(
            tensor=q_fp4,
            gmem_inner_dim=config.head_dim,
            gmem_outer_dim=config.batch_size * config.next_n * config.num_heads,
            smem_inner_dim=config.head_dim,
            smem_outer_dim=next_n_atom * config.num_heads,
            gmem_outer_stride=int(q_fp4.stride(2)),
            swizzle_mode=config.head_dim // 2,
        ),
        "tensor_map_sf_q": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=sf_q,
            gmem_inner_dim=config.num_heads,
            gmem_outer_dim=config.batch_size * config.next_n,
            smem_inner_dim=config.num_heads,
            smem_outer_dim=next_n_atom,
            gmem_outer_stride=int(sf_q.stride(1)),
            swizzle_mode=0,
        ),
        "tensor_map_kv": _encode_fp4_packed_smem_tma_3d_desc(
            tensor=kv_fp4,
            gmem_inner_dim=config.head_dim,
            gmem_mid_dim=config.page_size,
            gmem_outer_dim=config.num_pages,
            smem_inner_dim=config.head_dim,
            smem_mid_dim=config.page_size,
            smem_outer_dim=1,
            gmem_mid_stride=int(kv_fp4.stride(1)),
            gmem_outer_stride=int(kv_fp4.stride(0)),
            swizzle_mode=config.head_dim // 2,
        ),
        "tensor_map_sf_kv": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=sf_kv,
            gmem_inner_dim=config.page_size,
            gmem_outer_dim=config.num_pages,
            smem_inner_dim=config.page_size,
            smem_outer_dim=1,
            gmem_outer_stride=int(sf_kv.stride(0)),
            swizzle_mode=0,
        ),
        "tensor_map_weights": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=weights,
            gmem_inner_dim=config.num_heads,
            gmem_outer_dim=config.batch_size * config.next_n,
            smem_inner_dim=config.num_heads,
            smem_outer_dim=next_n_atom,
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
    config: PagedMQALogitsFP4Config = data["config"]
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
    config: PagedMQALogitsFP4Config = data["config"]
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
        tensor_maps["tensor_map_sf_q"].ptr,
        tensor_maps["tensor_map_kv"].ptr,
        tensor_maps["tensor_map_sf_kv"].ptr,
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


def run_test(**kwargs: Any) -> None:
    data = prepare_data(**kwargs)
    from tirx_kernels.target import prepare_cuda_arch

    deepgemm_diff = None
    if prepare_cuda_arch() != "sm_110a":
        deepgemm_logits = _run_deepgemm_paged_mqa(data, clean_logits=False)
        deepgemm_diff = _assert_correct(data, deepgemm_logits, name="DeepGEMM")
    tirx_logits = _launch_tirx_paged_mqa(data)
    torch.cuda.synchronize()
    tirx_diff = _assert_correct(data, tirx_logits, name="TIRx")
    if deepgemm_diff is not None and tirx_diff > max(deepgemm_diff, _TEST_DIFF_THRESHOLD):
        raise AssertionError(
            f"TIRx diff {tirx_diff:.6g} is worse than DeepGEMM diff {deepgemm_diff:.6g}"
        )


def prepare_bench(**kwargs: Any):
    """Compile the paged MQA executable without allocating CUDA data."""
    from tirx_kernels.runner import hardware_num_sms, prepared_gpu_benchmark

    config = _make_config(**kwargs)
    compile_config = PagedMQALogitsFP4Config(
        **{**asdict(config), "num_sms": hardware_num_sms(config.num_sms)}
    )
    executable = _compile_tirx_paged_mqa(compile_config)
    return prepared_gpu_benchmark(run_gpu, {"config": dict(kwargs), "executable": executable})


def run_gpu(prepared, **kwargs: Any) -> dict[str, Any]:
    kwargs = {**prepared["config"], **kwargs}
    from tirx_kernels.runner import bench
    from tirx_kernels.target import prepare_cuda_arch

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

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    data = prepare_data(**config_kwargs)
    invocation = _prepare_tirx_invocation(data, executable=prepared["executable"])
    tirx_logits = _run_tirx_invocation(data, invocation)
    torch.cuda.synchronize()
    tirx_diff = _assert_correct(data, tirx_logits, name="TIRx")
    torch.cuda.empty_cache()

    def _deepgemm():
        return lambda: _run_deepgemm_paged_mqa(data, clean_logits=False)

    funcs_tirx_first = {"tirx": lambda: _run_tirx_invocation(data, invocation)}

    result = bench(
        funcs_tirx_first,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=_rounds,
        cooldown_s=_cooldown_s,
        references={"deepgemm": _deepgemm} if prepare_cuda_arch() != "sm_110a" else {},
    )
    result["max_diff"] = tirx_diff
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
    "PagedMQALogitsFP4Config",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
