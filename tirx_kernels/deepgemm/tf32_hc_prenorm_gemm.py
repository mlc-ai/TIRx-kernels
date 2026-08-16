# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any
from unittest import SkipTest

import torch

_SM100_SMEM_CAPACITY = 232448
_TEST_DIFF_THRESHOLD = 1e-8
_COMPILE_CACHE_NAMESPACE = "deepgemm.tf32_hc_prenorm_gemm.compile"


def _tf32_hc_cuda_postproc(code: str) -> str:
    original = code
    code, unroll_count = re.subn(
        r"(\n    )#pragma unroll\n(    for \(uint s_2 =)", r"\1#pragma unroll 12\n\2", code, count=1
    )
    dump_dir = os.environ.get("TF32_HC_POSTPROC_DUMP_DIR")
    if dump_dir:
        dump_path = Path(dump_dir)
        dump_path.mkdir(parents=True, exist_ok=True)
        (dump_path / "original.cu").write_text(original)
        (dump_path / "postproc.cu").write_text(code)
        (dump_path / "notes.txt").write_text(f"unroll12={unroll_count}\n")
    return code


@dataclass(frozen=True)
class TF32HCPrenormGemmConfig:
    m: int = 13
    n: int = 24
    k: int = 512
    num_splits: int = 1
    seed: int = 0
    num_sms: int = 148

    @property
    def block_m(self) -> int:
        return 64

    @property
    def block_k(self) -> int:
        return 64

    @property
    def block_n(self) -> int:
        return _align_up(self.n, 16)

    @property
    def num_threads(self) -> int:
        return 256

    @property
    def num_mma_threads(self) -> int:
        return 128

    @property
    def num_cast_and_reduce_threads(self) -> int:
        return 128

    @property
    def swizzle_cd_mode(self) -> int:
        return _get_swizzle_mode(self.block_n, torch.empty((), dtype=torch.float32).element_size())

    @property
    def smem_a_size_per_stage(self) -> int:
        return self.block_m * self.block_k * torch.empty((), dtype=torch.bfloat16).element_size()

    @property
    def smem_b_size_per_stage(self) -> int:
        return self.block_n * self.block_k * torch.empty((), dtype=torch.float32).element_size()

    @property
    def smem_cd_size(self) -> int:
        return self.block_m * self.swizzle_cd_mode

    @property
    def num_stages(self) -> int:
        num_stages = 12
        while num_stages > 0:
            smem_barriers = (num_stages * 4 + 1) * 8
            smem_tmem_ptr = 4
            smem_size = (
                (self.smem_a_size_per_stage + self.smem_b_size_per_stage) * num_stages
                + self.smem_cd_size
                + smem_barriers
                + smem_tmem_ptr
            )
            if smem_size <= _SM100_SMEM_CAPACITY:
                return num_stages
            num_stages -= 1
        raise ValueError("no valid stage count fits SM100 shared memory")

    @property
    def smem_size(self) -> int:
        num_stages = self.num_stages
        return (
            (self.smem_a_size_per_stage + self.smem_b_size_per_stage) * num_stages
            + self.smem_cd_size
            + (num_stages * 4 + 1) * 8
            + 4
        )

    @property
    def grid_blocks(self) -> int:
        return self.num_splits * _ceil_div(self.m, self.block_m)

    @property
    def num_k_blocks(self) -> int:
        return self.k // self.block_k

    @property
    def d_shape(self) -> tuple[int, ...]:
        if self.num_splits == 1:
            return (self.m, self.n)
        return (self.num_splits, self.m, self.n)

    @property
    def sqr_sum_shape(self) -> tuple[int, ...]:
        if self.num_splits == 1:
            return (self.m,)
        return (self.num_splits, self.m)

    def validate(self) -> None:
        if self.m <= 0 or self.n <= 0 or self.k <= 0:
            raise ValueError("m, n, and k must be positive")
        if self.n > 128 or self.n % 8 != 0:
            raise ValueError("DeepGEMM requires n <= 128 and n % 8 == 0")
        if self.k % self.block_k != 0:
            raise ValueError("DeepGEMM requires k % 64 == 0")
        if (
            self.swizzle_cd_mode // torch.empty((), dtype=torch.float32).element_size()
            != self.block_n
        ):
            raise ValueError("DeepGEMM requires swizzle_cd_mode / sizeof(float) == BLOCK_N")
        if self.num_splits <= 0:
            raise ValueError("num_splits must be positive")
        if self.num_sms <= 0:
            raise ValueError("num_sms must be positive")


def _make_config(**kwargs: Any) -> TF32HCPrenormGemmConfig:
    kwargs = {key: value for key, value in kwargs.items() if key != "label"}
    config = TF32HCPrenormGemmConfig(**kwargs)
    config.validate()
    return config


def _align_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _get_swizzle_mode(block_size: int, elem_size: int) -> int:
    for mode in (128, 64, 32, 16):
        if block_size * elem_size % mode == 0:
            return mode
    return 0


def _config_label(config: dict[str, Any]) -> str:
    split = config["num_splits"]
    return f"m{config['m']}_n{config['n']}_k{config['k']}_s{split}"


def _make_case(*, m: int, n: int, k: int, num_splits: int, seed: int) -> dict[str, Any]:
    config = {"m": m, "n": n, "k": k, "num_splits": num_splits, "seed": seed}
    config["label"] = _config_label(config)
    return config


KERNEL_META = {
    "name": "deepgemm_sm100_tf32_hc_prenorm_gemm",
    "category": "deepgemm",
    "compute_capability": 10,
}

# ── Bench shape set ─────────────────────────────────────────────────────────
# num_splits follows SGLang's _compute_num_split_for_mhc_pre with n_sms pinned
# to 148 (SM100 / B200):
#   grid = ceil(M/64); num_block_k = ceil(K/64)
#   num_splits = max(1, min(n_sms // max(grid, 1), num_block_k // 4))
_MHC_NUM_SMS = 148


def _compute_num_split_for_mhc_pre(num_tokens: int, hc_hidden_size: int) -> int:
    grid_size = (num_tokens + 63) // 64
    num_block_k = (hc_hidden_size + 63) // 64
    return max(1, min(_MHC_NUM_SMS // max(grid_size, 1), num_block_k // 4))


def _mhc_pre_token_count_representatives(
    max_num_tokens: int, hc_hidden_size: int
) -> tuple[int, ...]:
    """One representative M per distinct num_splits bucket over [1, max_tokens]
    (SGLang's get_mhc_pre_token_count_representatives)."""
    reps = {}
    for grid in range(1, (max(1, max_num_tokens) + 63) // 64 + 1):
        num_tokens = min(grid * 64, max_num_tokens)
        reps[_compute_num_split_for_mhc_pre(num_tokens, hc_hidden_size)] = num_tokens
    return tuple(sorted(reps.values()))


# Main set: the two production hc_hidden sizes x the M buckets from
# max_tokens 2048/4096/8192, deduped by (m, k, num_splits).
_PROD_HC_HIDDENS = (16384, 28672)
_MHC_PRE_MAX_TOKENS = (2048, 4096, 8192)

BENCH_CONFIGS = [
    _make_case(m=m, n=24, k=k, num_splits=s, seed=3000 + i)
    for i, (m, k, s) in enumerate(
        sorted(
            {
                (m, k, _compute_num_split_for_mhc_pre(m, k))
                for k in _PROD_HC_HIDDENS
                for max_tokens in _MHC_PRE_MAX_TOKENS
                for m in _mhc_pre_token_count_representatives(max_tokens, k)
            }
        )
    )
]

CONFIGS = [
    _make_case(m=13, n=24, k=512, num_splits=1, seed=0),
    _make_case(m=137, n=24, k=1024, num_splits=4, seed=1),
    _make_case(m=65, n=32, k=768, num_splits=3, seed=2),
]


def _prepare_data(config: TF32HCPrenormGemmConfig, *, compute_reference: bool) -> dict[str, Any]:
    from tirx_kernels.runner import hardware_num_sms

    if torch.cuda.is_available():
        torch.cuda.set_device(torch.cuda.current_device())
    else:
        raise SkipTest("CUDA is required for SM100 TF32 HC prenorm GEMM")
    if torch.cuda.get_device_capability()[0] < 10:
        raise SkipTest("SM100 TF32 HC prenorm GEMM requires compute capability 10.x")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(config.seed)

    runtime_config = TF32HCPrenormGemmConfig(
        **{**asdict(config), "num_sms": hardware_num_sms(config.num_sms)}
    )
    a = torch.randn((config.m, config.k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((config.n, config.k), dtype=torch.float32, device="cuda")
    d_tirx = torch.empty(config.d_shape, dtype=torch.float32, device="cuda")
    sqr_tirx = torch.empty(config.sqr_sum_shape, dtype=torch.float32, device="cuda")
    data = {"config": runtime_config, "a": a, "b": b, "d_tirx": d_tirx, "sqr_tirx": sqr_tirx}
    if compute_reference:
        data["reference_d"] = a.float() @ b.T
        data["reference_sqr"] = a.float().square().sum(dim=-1)
    return data


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    return _prepare_data(_make_config(**kwargs), compute_reference=True)


@dataclass
class TF32HCBenchCase:
    config: TF32HCPrenormGemmConfig
    a: torch.Tensor
    b: torch.Tensor
    d_tirx: torch.Tensor
    sqr_tirx: torch.Tensor
    tensor_maps: dict[str, Any]


def get_kernel(**kwargs: Any):
    from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
    from tvm.script import tirx as T
    from tvm.tirx.lang.pipeline import Pipeline, TCGen05Bar
    from tvm.tirx.layout import S, TCol, TileLayout, TLane, tcgen05_atom_layout

    config = _make_config(**kwargs)
    block_m = config.block_m
    block_n = config.block_n
    block_k = config.block_k
    num_splits = config.num_splits
    num_threads = config.num_threads
    num_warps = num_threads // 32
    num_mma_threads = config.num_mma_threads
    num_cast_and_reduce_threads = config.num_cast_and_reduce_threads
    num_mma_warps = num_mma_threads // 32
    num_stages = config.num_stages
    num_cast_stages = 2
    swizzle_b_mode = min(block_k * 4, 128)
    swizzle_cd_mode = config.swizzle_cd_mode
    smem_cd_size = config.smem_cd_size
    smem_a_size_per_stage = config.smem_a_size_per_stage
    smem_b_size_per_stage = config.smem_b_size_per_stage
    # SMEMPool bump-allocates cd | a | b | pipe barriers | tmem_ptr; all data
    # sizes are 1024-multiples so the pooled offsets reproduce the hand layout.
    num_tmem_cols = 256
    block_swizzled_bk = swizzle_b_mode // 4
    num_b_tma_atoms = block_k // block_swizzled_bk
    umma_k = 32 // 4
    d_tmem_start_col = block_k * num_cast_stages
    # Cast-warp per-thread register counts (compile-time Python ints): each of
    # the 128 cast/reduce threads owns ``cast_per_thread`` fp32 A registers in
    # the .16x256b atom; they span 2 Layout-F rows, with ``cast_pairs`` packed
    # f32x2 groups in each row.
    cast_per_thread = block_m * block_k // num_cast_and_reduce_threads
    cast_pairs = cast_per_thread // 4  # 2 rows x 2 fp32 values per packed pair
    tmem_layout = TileLayout(S[(128, num_tmem_cols) : (1 @ TLane, 1 @ TCol)])
    num_k_blocks = config.num_k_blocks
    num_k_blocks_per_split = num_k_blocks // num_splits
    remain_k_blocks = num_k_blocks % num_splits
    tma_g2s_2d = (
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
    )
    tma_s2g_2d = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
    tma_s2g_3d = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint"
    tcgen05_mma_tf32 = "tcgen05.mma.cta_group::1.kind::tf32"
    cache_policy_evict_first = T.uint64(0x12F0000000000000)
    cache_policy_evict_last = T.uint64(0x14F0000000000000)
    tf32_instr_desc = T.uint32(67635472)

    def add_smem_desc_offset(desc, offset):
        # Descriptor offsets wrap in the low 32 bits without carrying into the
        # encoded layout fields in the high half.
        desc_lo = T.alloc_local((1,), "uint32")
        desc_hi = T.alloc_local((1,), "uint32")
        result = T.alloc_local((1,), "uint64")
        T.evaluate(T.ptx.mov.b64(desc_lo[0], desc_hi[0], desc))
        T.evaluate(T.ptx.add.u32(desc_lo[0], desc_lo[0], T.cast(offset, "uint32")))
        T.evaluate(T.ptx.mov.b64(result[0], desc_lo[0], desc_hi[0]))
        return result[0]

    def cuda_grid_dependency_synchronize():
        T.evaluate(T.ptx.griddepcontrol.wait())

    @T.prim_func
    def sm100_tf32_hc_prenorm_gemm(
        shape_m: T.uint32,
        a: T.Buffer((config.m, config.k), "bfloat16"),
        b: T.Buffer((config.n, config.k), "float32"),
        d: T.Buffer(config.d_shape, "float32"),
        sqr_sum: T.Buffer((config.num_splits * config.m,), "float32"),
    ):
        # Build the same host-side tensor maps that copy_async previously
        # synthesized, but make them part of the public pre-lowering IR.
        d_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        if num_splits == 1:
            T.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                d_tensormap,
                "float32",
                2,
                T.handle_add_byte_offset(d.data, 0),
                config.n,
                config.m,
                T.uint32(config.n * 4),
                T.uint32(block_n),
                T.uint32(block_m),
                1,
                1,
                0,
                3,
                2,
                0,
            )
        else:
            T.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                d_tensormap,
                "float32",
                3,
                T.handle_add_byte_offset(d.data, 0),
                config.n,
                config.m,
                num_splits,
                T.uint32(config.n * 4),
                config.m * config.n * 4,
                T.uint32(block_n),
                T.uint32(block_m),
                1,
                1,
                1,
                1,
                0,
                3,
                2,
                0,
            )
        b_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            b_tensormap,
            "float32",
            2,
            T.handle_add_byte_offset(b.data, 0),
            config.k,
            config.n,
            config.k * 4,
            T.uint32(block_swizzled_bk),
            block_n,
            1,
            1,
            0,
            3,
            2,
            0,
            11,
        )
        a_tensormap: T.let[T.TensorMap()] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            a_tensormap,
            "bfloat16",
            2,
            T.handle_add_byte_offset(a.data, 0),
            config.k,
            config.m,
            config.k * 2,
            T.uint32(block_k),
            T.uint32(block_m),
            1,
            1,
            0,
            3,
            2,
            0,
        )
        T.device_entry()
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
        # TIRX_TRANSCRIBE_START sm100_tf32_hc_prenorm_gemm_impl
        warp_idx = T.warp_id([num_warps])
        T.warpgroup_id([num_warps // 4])
        lane_idx = T.lane_id([32])

        if warp_idx == 0:
            if T.cuda.elect_sync():
                T.evaluate(T.ptx.prefetch.tensormap(T.address_of(a_tensormap)))
        if warp_idx == 0:
            if T.cuda.elect_sync():
                T.evaluate(T.ptx.prefetch.tensormap(T.address_of(b_tensormap)))
        if warp_idx == 0:
            if T.cuda.elect_sync():
                T.evaluate(T.ptx.prefetch.tensormap(T.address_of(d_tensormap)))
        lane_u32: T.uint32 = T.cast(lane_idx, "uint32")

        # SMEMPool bump-allocates a 1024-aligned uint8 arena, reproducing the hand
        # layout (cd@0 | a | b | barriers | tmem_ptr) filling config.smem_size.
        smem = T.alloc_buffer([config.smem_size], "uint8", scope="shared.dyn", align=1024)
        T.attr({"tirx.dyn_smem_bytes": config.smem_size})
        pool = T.SMEMPool(ptr=smem.data)
        # D-epilogue staging buffer: reg tile staged in, then stored via TMA as a
        # 128B-swizzled mma_shared_layout atom.
        smem_cd_mma = pool.alloc(
            (block_m, block_n),
            "float32",
            align=1024,
            layout=mma_shared_layout("float32", SwizzleMode.SWIZZLE_128B_ATOM, (block_m, block_n)),
        )
        # A stages: TMA writes; cast warps read via ldmatrix.x4 into the .16x256b atom.
        smem_a_mma = pool.alloc(
            (num_stages, block_m, block_k),
            "bfloat16",
            align=1024,
            layout=mma_shared_layout(
                "bfloat16", SwizzleMode.SWIZZLE_128B_ATOM, (num_stages, block_m, block_k)
            ),
        )
        # B stages: TMA writes (spanning 2 x 128B atoms); gemm_async reads tf32.
        smem_b_mma = pool.alloc(
            (num_stages, block_n, block_k),
            "float32",
            align=1024,
            layout=mma_shared_layout(
                "float32", SwizzleMode.SWIZZLE_128B_ATOM, (num_stages, block_n, block_k)
            ),
        )
        # Pipes: smem (TMA full / MMA-commit empty), cast (128-thread deposit
        # full / MMA-commit empty), tmem (MMA signals D ready). Inits on warp 1.
        smem_pipe = Pipeline(
            pool,
            num_stages,
            full="tma",
            empty="tcgen05",
            init_full=1,
            init_empty=1,
            leader=(T.cuda.thread_rank() == 32),
        )
        cast_pipe = Pipeline(
            pool,
            num_cast_stages,
            full="mbar",
            empty="tcgen05",
            init_full=num_cast_and_reduce_threads,
            init_empty=1,
            leader=(T.cuda.thread_rank() == 32),
        )
        # One-way "tmem freed" signal, so a bare TCGen05Bar.
        tmem_pipe = TCGen05Bar(pool, 1, leader=(T.cuda.thread_rank() == 32))
        tmem_pipe.init(1)
        tmem_ptr_in_smem = pool.alloc((1,), "uint32", align=4)
        # Single full-256-col tcgen05.alloc (warp-2) + relinquish/dealloc (warp-1);
        # the TMEM base stays compile-time 0 so gemm_async never reloads it from SMEM.
        tmem_pool = T.TMEMPool(
            None,
            total_cols=num_tmem_cols,
            cta_group=1,
            alloc_warp=2,
            dealloc_warp=1,
            tmem_addr=tmem_ptr_in_smem,
            sync_after_alloc=False,
        )
        _tmem = tmem_pool.alloc((128, num_tmem_cols), "float32", layout=tmem_layout)

        # Make the inited barriers visible before the cta_sync.
        T.ptx.fence.mbarrier_init.release.cluster()
        tmem_pool.commit()  # warp-2-guarded tcgen05.alloc (self-guards via thread_rank)
        T.cuda.cta_sync()

        block_idx: T.uint32 = T.cast(T.cta_id([config.grid_blocks]), "uint32")
        m_block_idx: T.uint32 = block_idx // T.uint32(num_splits)
        k_split_idx: T.uint32 = block_idx % T.uint32(num_splits)
        k_offset: T.uint32 = (
            k_split_idx * T.uint32(num_k_blocks_per_split)
            + T.min(k_split_idx, T.uint32(remain_k_blocks))
        ) * T.uint32(block_k)
        m_offset: T.uint32 = shape_m * k_split_idx
        num_total_stages: T.uint32 = T.uint32(num_k_blocks_per_split) + T.cast(
            k_split_idx < T.uint32(remain_k_blocks), "uint32"
        )

        cuda_grid_dependency_synchronize()

        if warp_idx < num_mma_warps:
            if warp_idx == 0:
                if T.cuda.elect_sync():
                    # Loop-carried stage/phase counters (no per-iter div/mod on the uniform path).
                    tma_st: T.uint32 = T.uint32(0)
                    tma_ph: T.uint32 = T.uint32(1)
                    for s in T.serial(T.uint32(0), num_total_stages):
                        stage_idx: T.uint32 = tma_st
                        smem_pipe.empty.wait(stage_idx, tma_ph)
                        m_idx0: T.uint32 = m_block_idx * T.uint32(block_m)
                        k_idx0: T.uint32 = k_offset + s * T.uint32(block_k)
                        # A remains bf16 (exact in tf32); B's tensor map uses
                        # TFLOAT32 OOB-fill mode 11 so the load RN-truncates as
                        # before.  Coordinates are tensor-map order: K, then M/N.
                        T.ptx[tma_g2s_2d](
                            T.ptr_byte_offset(
                                smem_a_mma.ptr_to([0, 0, 0]),
                                stage_idx * T.uint32(block_m * block_k * 2),
                                "bfloat16",
                            ),
                            T.address_of(a_tensormap),
                            T.cast(k_idx0, "int32"),
                            T.cast(m_idx0, "int32"),
                            smem_pipe.full.ptr_to([stage_idx]),
                            cache_policy_evict_first,
                        )
                        for b_atom in T.unroll(num_b_tma_atoms):
                            T.ptx[tma_g2s_2d](
                                T.ptr_byte_offset(
                                    smem_b_mma.ptr_to([0, 0, 0]),
                                    stage_idx * T.uint32(block_n * block_k * 4)
                                    + T.uint32(b_atom * block_n * block_swizzled_bk * 4),
                                    "float32",
                                ),
                                T.address_of(b_tensormap),
                                T.cast(k_idx0 + T.uint32(b_atom * block_swizzled_bk), "int32"),
                                T.int32(0),
                                smem_pipe.full.ptr_to([stage_idx]),
                                cache_policy_evict_last,
                            )
                        smem_pipe.full.arrive(
                            stage_idx,
                            tx_count=T.uint32(smem_a_size_per_stage + smem_b_size_per_stage),
                        )
                        tma_st = stage_idx + T.uint32(1)
                        if tma_st == T.uint32(num_stages):
                            tma_st = T.uint32(0)
                            tma_ph = tma_ph ^ T.uint32(1)

            if warp_idx == 1:
                mma_st: T.uint32 = T.uint32(0)
                mma_cs: T.uint32 = T.uint32(0)
                mma_ph: T.uint32 = T.uint32(0)
                for s in T.serial(T.uint32(0), num_total_stages):
                    stage_idx: T.uint32 = mma_st
                    cast_stage_idx: T.uint32 = mma_cs
                    cast_pipe.full.wait(cast_stage_idx, mma_ph)
                    # TMEM A columns and the swizzled B matrix descriptor match
                    # the former tcgen05 tile dispatch exactly.
                    a_col: T.int32 = T.cast(cast_stage_idx * T.uint32(block_k), "int32")
                    desc_b: T.uint64
                    T.cuda.tcgen05.encode_matrix_descriptor(
                        T.address_of(desc_b),
                        smem_b_mma.ptr_to([0, 0, 0]),
                        ldo=256,
                        sdo=64,
                        swizzle=3,
                    )
                    for ki in T.unroll(block_k // umma_k):
                        if T.cuda.elect_sync():
                            T.ptx[tcgen05_mma_tf32](
                                T.uint32(d_tmem_start_col),
                                T.cast(a_col + T.int32(ki * umma_k), "uint32"),
                                add_smem_desc_offset(
                                    desc_b,
                                    (
                                        T.uint32((ki // 4) * 1024 + (ki % 4) * 8)
                                        + stage_idx * T.uint32(block_n * block_k)
                                    )
                                    // T.uint32(4),
                                ),
                                tf32_instr_desc,
                                T.uint32(0),
                                T.uint32(0),
                                T.uint32(0),
                                T.uint32(0),
                                T.ptx.pred((s != T.uint32(0)) if ki == 0 else T.bool(True)),
                            )
                    if T.cuda.elect_sync():
                        cast_pipe.empty.arrive(cast_stage_idx)
                        smem_pipe.empty.arrive(stage_idx)
                    mma_st = stage_idx + T.uint32(1)
                    if mma_st == T.uint32(num_stages):
                        mma_st = T.uint32(0)
                    mma_cs = cast_stage_idx ^ T.uint32(1)
                    if mma_cs == T.uint32(0):
                        mma_ph = mma_ph ^ T.uint32(1)
                if T.cuda.elect_sync():
                    tmem_pipe.arrive(0)

            tmem_pipe.wait(0, 0)
            # D epilogue, hand-aligned: 8 x [tcgen05.ld.32x32b.x4 + wait.ld +
            # st.shared.v4 (lane<16) + syncwarp] into the 128B-swizzled smem_cd.
            d_frag = T.alloc_local((4,), "float32")
            d_words = d_frag.view("uint32")
            for i in T.unroll(block_n // 4):
                taddr_d: T.uint32 = T.uint32(d_tmem_start_col + i * 4)
                T.ptx["tcgen05.ld.sync.aligned.32x32b.x4.b32"](
                    d_frag[0], d_frag[1], d_frag[2], d_frag[3], T.uint32(taddr_d)
                )
                T.ptx.tcgen05.wait__ld.sync.aligned()
                if lane_u32 < T.uint32(16):
                    # Per-thread 4-col slice store; ptr_to applies the same
                    # 128B-swizzled layout that the former tile copy selected.
                    m_row: T.uint32 = T.cast(warp_idx, "uint32") * T.uint32(16) + lane_u32
                    compose_m: T.uint32 = m_row * T.uint32(block_n) + T.uint32(i * 4)
                    compose_q: T.uint32 = compose_m // T.uint32(4)
                    smem_cd_offset: T.uint32 = (
                        (compose_q ^ ((compose_q & T.uint32(56)) >> T.uint32(3))) << T.uint32(2)
                    ) + compose_m % T.uint32(4)
                    T.ptx.st.shared.v4.u32(
                        T.ptr_byte_offset(
                            smem_cd_mma.ptr_to([0, 0]), smem_cd_offset * T.uint32(4), "float32"
                        ),
                        d_words[0],
                        d_words[1],
                        d_words[2],
                        d_words[3],
                    )
                T.cuda.warp_sync()

            T.ptx.fence.proxy.async_.shared__cta()
            T.ptx.bar.sync(0, T.uint32(num_mma_threads))
            if warp_idx == 0:
                if T.cuda.elect_sync():
                    # D store via TMA (writes only the valid region of boundary tiles).
                    m0: T.uint32 = m_block_idx * T.uint32(block_m)
                    if num_splits == 1:
                        T.ptx[tma_s2g_2d](
                            T.address_of(d_tensormap),
                            T.int32(0),
                            T.cast(m0, "int32"),
                            smem_cd_mma.ptr_to([0, 0]),
                            cache_policy_evict_first,
                        )
                    else:
                        ks: T.uint32 = k_split_idx
                        T.ptx[tma_s2g_3d](
                            T.address_of(d_tensormap),
                            T.int32(0),
                            T.cast(m0, "int32"),
                            T.cast(ks, "int32"),
                            smem_cd_mma.ptr_to([0, 0]),
                            cache_policy_evict_first,
                        )
                    T.ptx.cp.async_.bulk.commit_group()
            # Keep the TMEM teardown on warp 1, but spell the allocator-slot read
            # explicitly so the public pre-lowering IR contains a real PTX shared
            # load rather than a shared BufferLoad.
            if warp_idx == 1:
                T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                tmem_dealloc_addr: T.uint32
                T.ptx.ld.shared.u32(tmem_dealloc_addr, tmem_ptr_in_smem.ptr_to([0]))
                T.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](
                    tmem_dealloc_addr, T.uint32(num_tmem_cols)
                )
        else:
            sub_warp_idx: T.uint32 = T.cast(
                T.cast(warp_idx, "int32") - T.int32(num_mma_warps), "uint32"
            )
            # A cast/deposit register tiles. The bf16 A tile is ldmatrix-loaded
            # (below) into the per-warpgroup ``.16x256b`` tcgen05 register atom —
            # the same Layout-F M=64 distribution the gemm_async A-in-TMEM operand
            # reads — then cast to tf32 (T.cast) and DEPOSITED into TMEM cols
            # [cast_stage*block_k, +block_k) via T.copy_async (reg->tmem
            # tcgen05.st, the A operand gemm_async consumes). ``a_bf16`` is
            # declared with the *fp32* atom layout (one element per 32-bit slot,
            # NOT the dense 2-bf16-per-slot bf16 atom) so a_bf16 and a_fp32 share
            # an identical per-(lane, register) (row, col) mapping — the cast is
            # then a slot-for-slot widen and a_fp32's register order matches both
            # the ldmatrix output AND what the tcgen05.st deposit consumes (rel
            # D == 0). NB the LOAD stays hand ldmatrix: T.copy on this warpgroup
            # atom can't emit ldmatrix (its m8n8 per-warp lane distribution is
            # structurally incompatible with the wid_in_wg+split-laneid atom) and
            # falls to a 16x-scalar-LDS reg path that costs +25% on the latency-
            # bound tail. The square/accumulate runs on the flat ``.local()`` view
            # (per-thread private regs via explicit PTX). Its physical
            # register order interleaves the two Layout-F rows in packed pairs:
            # regs [4*p:4*p+2] feed row m_idx0 and regs [4*p+2:4*p+4] feed row
            # m_idx1 (+8).
            a_bf16 = T.alloc_buffer(
                (block_m, block_k),
                "bfloat16",
                layout=tcgen05_atom_layout("16x256b", (block_m, block_k), "float32"),
                scope="local",
            )
            a_fp32 = T.alloc_tcgen05_ldst_frag("16x256b", (block_m, block_k), "float32")
            # Dual packed fma.f32x2 sum-of-squares accumulators (hand's sum0/sum1);
            # the fused form is the only no-regression reduce shape.
            sqr0 = T.alloc_local((2,), "float32")
            sqr1 = T.alloc_local((2,), "float32")
            a_flat = a_fp32.local()  # 1D physical-register view, 32 fp32 values per thread
            a_words = a_flat.view("uint32")
            a_bf16_flat = a_bf16.local()
            a_bf16_u16 = a_bf16_flat.view("uint16")
            a_bf16_words = a_bf16_flat.view("uint32")
            sqr0[0] = T.float32(0)
            sqr0[1] = T.float32(0)
            sqr1[0] = T.float32(0)
            sqr1[1] = T.float32(0)
            cast_st: T.uint32 = T.uint32(0)
            cast_ph: T.uint32 = T.uint32(0)
            cast_cs: T.uint32 = T.uint32(0)
            cast_cph: T.uint32 = T.uint32(1)
            for s in T.serial(T.uint32(0), num_total_stages, unroll=True):
                stage_idx: T.uint32 = cast_st
                cast_stage_idx: T.uint32 = cast_cs
                a_col: T.int32 = T.cast(cast_stage_idx * T.uint32(block_k), "int32")
                smem_pipe.full.wait(stage_idx, cast_ph)
                # Four x4 ldmatrix instructions reproduce the warpgroup copy's
                # physical register order.  Keep the dispatcher's explicit
                # swizzled element offset so ptxas sees the same address DAG.
                for mm in T.unroll(4):
                    smem_off: T.uint32 = (
                        T.cast(
                            T.cast(sub_warp_idx, "int32") * T.int32(1024)
                            + T.int32((mm // 2) * 512),
                            "uint32",
                        )
                        + stage_idx * T.uint32(block_m * block_k)
                        + T.cast(lane_idx % T.int32(8) * T.int32(block_k), "uint32")
                        + (
                            T.cast(
                                T.int32((mm % 2) * 32) + lane_idx // T.int32(8) * T.int32(8),
                                "uint32",
                            )
                            ^ (
                                (
                                    T.cast(
                                        T.cast(sub_warp_idx, "int32") * T.int32(16)
                                        + T.int32((mm // 2) * 8),
                                        "uint32",
                                    )
                                    + stage_idx * T.uint32(block_k)
                                    + T.cast(lane_idx % T.int32(8) * T.int32(block_k), "uint32")
                                    // T.uint32(block_k)
                                )
                                & T.uint32(7)
                            )
                            << T.uint32(3)
                        )
                    )
                    reg_base = (mm % 2) * 8 + mm // 2
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_bf16_words[reg_base],
                        a_bf16_words[reg_base + 2],
                        a_bf16_words[reg_base + 4],
                        a_bf16_words[reg_base + 6],
                        T.ptr_byte_offset(smem_a_mma.ptr_to([0, 0, 0]), smem_off * 2, "bfloat16"),
                    )
                cast_pipe.empty.wait(cast_stage_idx, cast_cph)
                # bf16->tf32 + sqr-fma + TMEM deposit: interleaved per 8-col atom on
                # short mainloops (hand structure); single wide STTM.x8 on deep pipelines.
                if num_k_blocks_per_split <= 16:
                    for p in range(block_k // 8):
                        for f in range(2):
                            T.ptx.cvt.f32.bf16(a_flat[p * 4 + f * 2], a_bf16_u16[p * 4 + f * 2])
                            T.ptx.cvt.f32.bf16(
                                a_flat[p * 4 + f * 2 + 1], a_bf16_u16[p * 4 + f * 2 + 1]
                            )
                        # sqr{0,1} += a*a for this atom's packed pair per row.
                        pair0_lhs: T.uint64
                        pair0_rhs: T.uint64
                        pair0_acc: T.uint64
                        T.ptx.mov.b64(pair0_lhs, a_flat[p * 4], a_flat[p * 4 + 1])
                        T.ptx.mov.b64(pair0_rhs, a_flat[p * 4], a_flat[p * 4 + 1])
                        T.ptx.mov.b64(pair0_acc, sqr0[0], sqr0[1])
                        T.ptx.fma.rz.ftz.f32x2(pair0_lhs, pair0_lhs, pair0_rhs, pair0_acc)
                        T.ptx.mov.b64(sqr0[0], sqr0[1], pair0_lhs)
                        pair1_lhs: T.uint64
                        pair1_rhs: T.uint64
                        pair1_acc: T.uint64
                        T.ptx.mov.b64(pair1_lhs, a_flat[p * 4 + 2], a_flat[p * 4 + 3])
                        T.ptx.mov.b64(pair1_rhs, a_flat[p * 4 + 2], a_flat[p * 4 + 3])
                        T.ptx.mov.b64(pair1_acc, sqr1[0], sqr1[1])
                        T.ptx.fma.rz.ftz.f32x2(pair1_lhs, pair1_lhs, pair1_rhs, pair1_acc)
                        T.ptx.mov.b64(sqr1[0], sqr1[1], pair1_lhs)
                        T.ptx["tcgen05.st.sync.aligned.16x256b.x1.b32"](
                            T.cuda.get_tmem_addr(T.uint32(0), 0, a_col + p * 8),
                            a_words[p * 4],
                            a_words[p * 4 + 1],
                            a_words[p * 4 + 2],
                            a_words[p * 4 + 3],
                        )
                else:
                    for f in range(cast_per_thread // 2):
                        T.ptx.cvt.f32.bf16(a_flat[f * 2], a_bf16_u16[f * 2])
                        T.ptx.cvt.f32.bf16(a_flat[f * 2 + 1], a_bf16_u16[f * 2 + 1])
                    for p in T.unroll(cast_pairs):
                        pair0_lhs: T.uint64
                        pair0_rhs: T.uint64
                        pair0_acc: T.uint64
                        T.ptx.mov.b64(pair0_lhs, a_flat[p * 4], a_flat[p * 4 + 1])
                        T.ptx.mov.b64(pair0_rhs, a_flat[p * 4], a_flat[p * 4 + 1])
                        T.ptx.mov.b64(pair0_acc, sqr0[0], sqr0[1])
                        T.ptx.fma.rz.ftz.f32x2(pair0_lhs, pair0_lhs, pair0_rhs, pair0_acc)
                        T.ptx.mov.b64(sqr0[0], sqr0[1], pair0_lhs)
                        pair1_lhs: T.uint64
                        pair1_rhs: T.uint64
                        pair1_acc: T.uint64
                        T.ptx.mov.b64(pair1_lhs, a_flat[p * 4 + 2], a_flat[p * 4 + 3])
                        T.ptx.mov.b64(pair1_rhs, a_flat[p * 4 + 2], a_flat[p * 4 + 3])
                        T.ptx.mov.b64(pair1_acc, sqr1[0], sqr1[1])
                        T.ptx.fma.rz.ftz.f32x2(pair1_lhs, pair1_lhs, pair1_rhs, pair1_acc)
                        T.ptx.mov.b64(sqr1[0], sqr1[1], pair1_lhs)
                    T.ptx["tcgen05.st.sync.aligned.16x256b.x8.b32"](
                        T.cuda.get_tmem_addr(T.uint32(0), 0, a_col),
                        *[a_words[i] for i in range(cast_per_thread)],
                    )
                T.ptx.tcgen05.wait__st.sync.aligned()
                cast_pipe.full.arrive(cast_stage_idx)
                cast_st = stage_idx + T.uint32(1)
                if cast_st == T.uint32(num_stages):
                    cast_st = T.uint32(0)
                    cast_ph = cast_ph ^ T.uint32(1)
                cast_cs = cast_stage_idx ^ T.uint32(1)
                if cast_cs == T.uint32(0):
                    cast_cph = cast_cph ^ T.uint32(1)

            # Cross-lane sum-of-squares reduce over the 4 K-lanes (the hand's
            # shfl_xor 2,1), then store the two per-row results.
            sqr_part = T.alloc_local((2,), "float32")
            sqr_part[0] = sqr0[0] + sqr0[1]
            sqr_part[1] = sqr1[0] + sqr1[1]
            for spa in range(2):
                reduce_mask: T.uint32 = T.tvm_warp_activemask()
                sqr_part[spa] = sqr_part[spa] + T.tvm_warp_shuffle_xor(
                    reduce_mask, sqr_part[spa], 1, 32, 32
                )
                sqr_part[spa] = sqr_part[spa] + T.tvm_warp_shuffle_xor(
                    reduce_mask, sqr_part[spa], 2, 32, 32
                )
            reduced0: T.float32 = sqr_part[0]
            reduced1: T.float32 = sqr_part[1]
            m_idx0: T.uint32 = (
                m_block_idx * T.uint32(block_m)
                + sub_warp_idx * T.uint32(block_m // 4)
                + lane_u32 // T.uint32(4)
            )
            m_idx1: T.uint32 = m_idx0 + T.uint32(8)
            if (lane_u32 % T.uint32(4)) == T.uint32(0):
                if m_idx0 < shape_m:
                    T.ptx.st.global_.f32(
                        sqr_sum.ptr_to([T.cast(m_offset + m_idx0, "int32")]), reduced0
                    )
                if m_idx1 < shape_m:
                    T.ptx.st.global_.f32(
                        sqr_sum.ptr_to([T.cast(m_offset + m_idx1, "int32")]), reduced1
                    )

    return sm100_tf32_hc_prenorm_gemm.with_attr(
        "tirx.kernel_launch_params",
        [
            "blockIdx.x",
            "threadIdx.x",
            "tirx.use_programtic_dependent_launch",
            "tirx.use_dyn_shared_memory",
        ],
    )


def _compile_tirx_tf32_hc_for_config(
    *, m: int, n: int, k: int, num_splits: int, seed: int, num_sms: int
) -> Any:
    import tvm

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    kernel = get_kernel(m=m, n=n, k=k, num_splits=num_splits, seed=seed, num_sms=num_sms)
    previous_postproc = tvm.get_global_func("tvm_callback_cuda_postproc", allow_missing=True)

    @tvm.register_global_func("tvm_callback_cuda_postproc", override=True)
    def _postproc(code: str, target: Any) -> str:
        if previous_postproc is not None:
            code = previous_postproc(code, target)
        if "sm100_tf32_hc_prenorm_gemm_kernel" in code:
            code = _tf32_hc_cuda_postproc(code)
        return code

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


_compile_tirx_tf32_hc_for_config = cache(_compile_tirx_tf32_hc_for_config)


def _compile_tirx_tf32_hc_key(config: TF32HCPrenormGemmConfig) -> tuple[tuple[str, Any], ...]:
    return tuple(asdict(config).items())


def _compile_tirx_tf32_hc(config: TF32HCPrenormGemmConfig) -> Any:
    compile_kwargs = asdict(config)
    return _compile_tirx_tf32_hc_for_config(**compile_kwargs)


def _build_tirx_tensor_maps(data: dict[str, Any]) -> dict[str, Any]:
    # A, B and D are all raw gmem buffer params now (copy_async(tma) host-builds
    # every descriptor: A bf16, B TFLOAT32, D fp32 store). No hand tensor maps.
    return {}


def _run_tirx_with_tensor_maps(
    data: dict[str, Any], executable: Any, tensor_maps: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    config: TF32HCPrenormGemmConfig = data["config"]
    executable(config.m, data["a"], data["b"], data["d_tirx"], data["sqr_tirx"].reshape(-1))
    return data["d_tirx"], data["sqr_tirx"]


def _launch_tirx_hc(data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_tirx_with_tensor_maps(
        data, _compile_tirx_tf32_hc(data["config"]), _build_tirx_tensor_maps(data)
    )


def _final_outputs(
    d: torch.Tensor, sqr_sum: torch.Tensor, config: TF32HCPrenormGemmConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.num_splits == 1:
        return d, sqr_sum
    return d.sum(dim=0), sqr_sum.sum(dim=0)


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double()
    y = y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return float((1 - sim).item())


def _assert_correct(
    data: dict[str, Any], d: torch.Tensor, sqr_sum: torch.Tensor, *, name: str
) -> float:
    config: TF32HCPrenormGemmConfig = data["config"]
    final_d, final_sqr = _final_outputs(d, sqr_sum, config)
    diff = max(
        _calc_diff(final_d, data["reference_d"]), _calc_diff(final_sqr, data["reference_sqr"])
    )
    if diff >= _TEST_DIFF_THRESHOLD:
        raise AssertionError(f"{name} diff {diff:.10g} >= {_TEST_DIFF_THRESHOLD}")
    return diff


def run_test(**kwargs: Any) -> None:
    data = prepare_data(**kwargs)
    tirx_d, tirx_sqr = _launch_tirx_hc(data)
    torch.cuda.synchronize()
    _assert_correct(data, tirx_d, tirx_sqr, name="TIRx")


def _make_bench_case(config_kwargs: dict[str, Any]) -> TF32HCBenchCase:
    data = _prepare_data(_make_config(**config_kwargs), compute_reference=False)
    return TF32HCBenchCase(
        config=data["config"],
        a=data["a"],
        b=data["b"],
        d_tirx=data["d_tirx"],
        sqr_tirx=data["sqr_tirx"],
        tensor_maps=_build_tirx_tensor_maps(data),
    )


def _bench_tirx_case(case: TF32HCBenchCase, executable: Any) -> tuple[torch.Tensor, torch.Tensor]:
    executable(case.config.m, case.a, case.b, case.d_tirx, case.sqr_tirx.reshape(-1))
    return case.d_tirx, case.sqr_tirx


def prepare_bench(**kwargs: Any):
    """Compile the hardware-profile specialization before GPU assignment."""
    from tirx_kernels.runner import hardware_num_sms, prepared_gpu_benchmark

    config = _make_config(**kwargs)
    runtime_config = TF32HCPrenormGemmConfig(
        **{**asdict(config), "num_sms": hardware_num_sms(config.num_sms)}
    )
    executable = _compile_tirx_tf32_hc(runtime_config)
    return prepared_gpu_benchmark(run_gpu, {"config": dict(kwargs), "executable": executable})


def run_gpu(prepared, **kwargs: Any) -> dict[str, Any]:
    from tirx_kernels.runner import bench

    kwargs = {**prepared["config"], **kwargs}
    timer = kwargs.pop("timer", None)  # None inherits the canonical local timer default.
    warmup = kwargs.pop("warmup", None)
    repeat = kwargs.pop("repeat", None)
    _rounds = kwargs.pop("rounds", 1)
    _cooldown_s = kwargs.pop("cooldown_s", 1.0)
    config_kwargs = dict(kwargs)
    executable = prepared["executable"]

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    case = _make_bench_case(config_kwargs)

    def build_deepgemm():
        import deep_gemm

        output = torch.empty(case.config.d_shape, dtype=torch.float32, device="cuda")
        sqr_sum = torch.empty(case.config.sqr_sum_shape, dtype=torch.float32, device="cuda")
        return lambda: deep_gemm.tf32_hc_prenorm_gemm(
            case.a,
            case.b,
            output,
            sqr_sum,
            num_splits=None if case.config.num_splits == 1 else case.config.num_splits,
        )

    return bench(
        {"tirx": lambda: _bench_tirx_case(case, executable)},
        references={"deepgemm": build_deepgemm},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=_rounds,
        cooldown_s=_cooldown_s,
    )


def run_bench(**kwargs: Any) -> dict[str, Any]:
    protocol = {
        name: kwargs.pop(name)
        for name in ("warmup", "repeat", "timer", "rounds", "cooldown_s")
        if name in kwargs
    }
    return prepare_bench(**kwargs).run_gpu(**protocol)


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "TF32HCPrenormGemmConfig",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
