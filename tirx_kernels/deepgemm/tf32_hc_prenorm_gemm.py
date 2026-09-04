# This file is a TIRx port of code from DeepGEMM
# (https://github.com/deepseek-ai/DeepGEMM @ 559d79fb), Copyright (c) 2025 DeepSeek
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

import ctypes
import math
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K
import tvm

_DEEP_GEMM_MODULE_NAME = "deep_gemm"
_SM100_SMEM_CAPACITY = 232448
_TEST_DIFF_THRESHOLD = 1e-8
_COMPILE_CACHE_NAMESPACE = "deepgemm.tf32_hc_prenorm_gemm.compile"


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

DEEPGEMM_TEST_COVERAGE = [
    _make_case(m=m, n=n, k=k, num_splits=num_splits, seed=1000 + seed)
    for seed, (m, n, k, num_splits) in enumerate(
        (m, n, k, num_splits)
        for m in (13, 137, 4096, 8192)
        for n, k in ((24, 28672), (24, 7680), (24, 7168))
        for num_splits in (1, 16)
    )
]

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

CONFIGS = [
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

# Legacy shapes kept for regression continuity with the pinned baseline. The
# k=7168/7680 ones are edge (hidden=1792/1920, non-production) and stay out of
# the main set.
LEGACY_CONFIGS = [
    _make_case(m=13, n=24, k=7168, num_splits=1, seed=2000),  # edge: hidden=1792
    _make_case(m=137, n=24, k=7680, num_splits=16, seed=2001),  # edge: hidden=1920
    _make_case(m=4096, n=24, k=7168, num_splits=1, seed=2002),  # edge: hidden=1792
    _make_case(m=4096, n=24, k=28672, num_splits=16, seed=2003),
]

BENCH_CONFIGS = CONFIGS + LEGACY_CONFIGS


def load_deep_gemm_hc() -> tuple[Any, str]:
    from tirx_kernels.reference_variants import load_reference
    from tirx_kernels.target import prepare_cuda_arch

    if prepare_cuda_arch() == "sm_110a":
        return load_reference("deep-gemm"), "verified_thor_variant"

    try:
        import deep_gemm as module

        source = "installed"
    except Exception as exc:
        raise SkipTest(
            f"DeepGEMM HC prenorm GEMM runtime unavailable: {_DEEP_GEMM_MODULE_NAME}: {exc}"
        ) from exc

    if not hasattr(module, "tf32_hc_prenorm_gemm"):
        raise SkipTest("DeepGEMM runtime unavailable: missing tf32_hc_prenorm_gemm")
    return module, source


def _get_num_sms(default: int) -> int:
    from tirx_kernels.runner import hardware_num_sms

    return hardware_num_sms(default)


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    deep_gemm, source = load_deep_gemm_hc()
    config = _make_config(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.cuda.current_device())
    else:
        raise SkipTest("CUDA is required for SM100 TF32 HC prenorm GEMM")
    from tirx_kernels.target import supports_sm100_kernel

    if not supports_sm100_kernel(torch.cuda.get_device_capability()):
        raise SkipTest("SM100 TF32 HC prenorm GEMM requires SM100 or prepared Thor")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(config.seed)

    runtime_config = TF32HCPrenormGemmConfig(
        **{
            **asdict(config),
            "num_sms": int(
                getattr(deep_gemm, "get_num_sms", lambda: _get_num_sms(config.num_sms))()
            ),
        }
    )
    a = torch.randn((config.m, config.k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((config.n, config.k), dtype=torch.float32, device="cuda")
    d_deepgemm = torch.empty(config.d_shape, dtype=torch.float32, device="cuda")
    sqr_deepgemm = torch.empty(config.sqr_sum_shape, dtype=torch.float32, device="cuda")
    d_tirx = torch.empty(config.d_shape, dtype=torch.float32, device="cuda")
    sqr_tirx = torch.empty(config.sqr_sum_shape, dtype=torch.float32, device="cuda")
    reference_d = a.float() @ b.T
    reference_sqr = a.float().square().sum(dim=-1)
    return {
        "config": runtime_config,
        "reference_source": source,
        "a": a,
        "b": b,
        "d_deepgemm": d_deepgemm,
        "sqr_deepgemm": sqr_deepgemm,
        "d_tirx": d_tirx,
        "sqr_tirx": sqr_tirx,
        "reference_d": reference_d,
        "reference_sqr": reference_sqr,
        "deep_gemm": deep_gemm,
    }


@dataclass
class TF32HCBenchCase:
    config: TF32HCPrenormGemmConfig
    deep_gemm: Any
    a: torch.Tensor
    b: torch.Tensor
    d_deepgemm: torch.Tensor
    sqr_deepgemm: torch.Tensor
    d_tirx: torch.Tensor
    sqr_tirx: torch.Tensor
    reference_d: torch.Tensor
    reference_sqr: torch.Tensor
    tensor_maps: dict[str, Any]


def _make_kernel(*, m: int, n: int, k: int, num_splits: int, seed: int, num_sms: int):
    """Trace the canonical K-owned device body for one specialization."""
    config = _make_config(m=m, n=n, k=k, num_splits=num_splits, seed=seed, num_sms=num_sms)

    # orig:L361-403 -- all derived from the config, transcribed value for value.
    block_m = config.block_m
    block_n = config.block_n
    block_k = config.block_k
    num_threads = config.num_threads
    num_warps = num_threads // 32
    num_mma_threads = config.num_mma_threads
    num_cast_and_reduce_threads = config.num_cast_and_reduce_threads
    num_mma_warps = num_mma_threads // 32
    num_stages = config.num_stages
    num_cast_stages = 2
    swizzle_b_mode = min(block_k * 4, 128)
    smem_a_size_per_stage = config.smem_a_size_per_stage
    smem_b_size_per_stage = config.smem_b_size_per_stage
    num_tmem_cols = 256
    block_swizzled_bk = swizzle_b_mode // 4
    num_b_tma_atoms = block_k // block_swizzled_bk
    umma_k = 32 // 4
    d_tmem_start_col = block_k * num_cast_stages
    cast_per_thread = block_m * block_k // num_cast_and_reduce_threads
    cast_pairs = cast_per_thread // 4
    num_k_blocks = config.num_k_blocks
    num_k_blocks_per_split = num_k_blocks // num_splits
    remain_k_blocks = num_k_blocks % num_splits

    # Instruction spellings. These live inside the original's ``get_kernel`` as
    # function locals, so there is nothing importable to bind to; they are
    # transcribed. The per-builtin call-site census is what catches a drift.
    tma_g2s_2d = (
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
    )
    tma_s2g_2d = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
    tma_s2g_3d = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint"
    tcgen05_mma_tf32 = "tcgen05.mma.cta_group::1.kind::tf32"
    cache_policy_evict_first = K.uint64(0x12F0000000000000)
    cache_policy_evict_last = K.uint64(0x14F0000000000000)
    tf32_instr_desc = K.uint32(67635472)

    def local(dtype, value=None):
        """One declared local, the way the original declares one.

        The original's body is full of annotated (and unannotated) assignments,
        which TVMScript binds as declared locals -- ``alignas(64) uint x_ptr[1]``
        in the generated code. A traced body has no such binding: a Python name
        holding an Expr re-emits the WHOLE expression at every use site. Where
        the original declares a local, so does this port.
        """
        buf = K.alloc_local((1,), dtype)
        if value is not None:
            K.assign(buf[0], value)
        return buf

    def add_smem_desc_offset(desc, offset):
        # orig:L404-413. Descriptor offsets wrap in the low 32 bits without
        # carrying into the encoded layout fields in the high half.
        desc_lo = K.local_scalar("uint32")
        desc_hi = K.local_scalar("uint32")
        result = K.local_scalar("uint64")
        K.ptx.mov.b64(desc_lo, desc_hi, desc)
        K.ptx.add.u32(desc_lo, desc_lo, K.cast(offset, "uint32"))
        K.ptx.mov.b64(result, desc_lo, desc_hi)
        return result

    def cuda_grid_dependency_synchronize():
        K.ptx.griddepcontrol.wait()

    @K.kernel(
        warps=num_warps,
        arch="sm_100a",
        min_blocks_per_sm=1,  # orig:L511 -- pinned by the original, not a default
        grid=config.grid_blocks,  # orig:L601
    )
    def sm100_tf32_hc_prenorm_gemm(
        shape_m: K.u32,
        # A/B/D are never dereferenced by the device code -- every access goes
        # through a tensor map. They stay in the signature because the launch is
        # what keeps the tensors the maps point at alive.
        a: K.gptr[K.bf16],
        b: K.gptr[K.f32],
        d: K.gptr[K.f32],
        sqr_sum: K.gptr[K.f32],
        a_map: K.TensorMap,
        b_map: K.TensorMap,
        d_map: K.TensorMap,
    ):
        warp_idx = K.warp_id()
        lane_idx = K.lane_id()

        # orig:L517-525 -- three separate guarded prefetches, kept separate.
        with K.If(warp_idx == 0), K.Then():
            with K.If(K.cuda.elect_sync()), K.Then():
                K.ptx.prefetch.tensormap(K.address_of(a_map))
        with K.If(warp_idx == 0), K.Then():
            with K.If(K.cuda.elect_sync()), K.Then():
                K.ptx.prefetch.tensormap(K.address_of(b_map))
        with K.If(warp_idx == 0), K.Then():
            with K.If(K.cuda.elect_sync()), K.Then():
                K.ptx.prefetch.tensormap(K.address_of(d_map))
        lane_u32 = local("uint32", K.cast(lane_idx, "uint32"))

        # ---------------- smem plan -- orig:L528-594 ------------------------
        # Declaration order reproduces the original's byte layout
        # (cd | a | b | barriers | tmem_ptr).
        smem = K.smem_pool()
        # The original pins the arena at ``config.smem_size``, which is 160
        # bytes MORE than the pool's true high-water mark: its barrier term
        # ``(num_stages*4+1)*8`` budgets 49 barriers where the body allocates
        # 29. Pinning the original's number keeps the launch's dynamic-smem
        # request identical rather than merely sufficient.
        smem.commit(config.smem_size)
        # D-epilogue staging buffer: reg tile staged in, then stored via TMA as a
        # 128B-swizzled mma_shared_layout atom.
        smem_cd_mma = smem.alloc((block_m, block_n), "float32", swizzle=K.SW128B, align=1024)
        # A stages: TMA writes; cast warps read via ldmatrix.x4 into the .16x256b atom.
        smem_a_mma = smem.alloc(
            (num_stages, block_m, block_k), "bfloat16", swizzle=K.SW128B, align=1024
        )
        # B stages: TMA writes (spanning 2 x 128B atoms); the MMA reads tf32.
        smem_b_mma = smem.alloc(
            (num_stages, block_n, block_k), "float32", swizzle=K.SW128B, align=1024
        )
        # Pipes: smem (TMA full / MMA-commit empty), cast (128-thread deposit
        # full / MMA-commit empty), tmem (MMA signals D ready). Inits on warp 1.
        smem_pipe = K.Pipeline(
            smem.pool,
            num_stages,
            full="tma",
            empty="tcgen05",
            init_full=1,
            init_empty=1,
            leader=(K.cuda.thread_rank() == 32),
        )
        cast_pipe = K.Pipeline(
            smem.pool,
            num_cast_stages,
            full="mbar",
            empty="tcgen05",
            init_full=num_cast_and_reduce_threads,
            init_empty=1,
            leader=(K.cuda.thread_rank() == 32),
        )
        # One-way "tmem freed" signal, so a bare TCGen05Bar.
        tmem_pipe = K.TCGen05Bar(smem.pool, 1, leader=(K.cuda.thread_rank() == 32))
        tmem_pipe.init(1)
        tmem_ptr_in_smem = smem.alloc((1,), "uint32", align=4)
        # Single full-256-col tcgen05.alloc (warp-2) + relinquish/dealloc (warp-1);
        # the TMEM base stays compile-time 0 so the MMA never reloads it from SMEM.
        # Make the inited barriers visible before the cta_sync.
        K.ptx.fence.mbarrier_init.release.cluster()
        with K.If(warp_idx == 2), K.Then():
            K.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                K.address_of(tmem_ptr_in_smem[0]), K.uint32(num_tmem_cols)
            )
        K.cuda.cta_sync()

        block_idx = local("uint32", K.cast(K.cta_id(), "uint32"))
        m_block_idx = local("uint32", block_idx[0] // K.uint32(num_splits))
        k_split_idx = local("uint32", block_idx[0] % K.uint32(num_splits))
        k_offset = local(
            "uint32",
            (
                k_split_idx[0] * K.uint32(num_k_blocks_per_split)
                + K.min(k_split_idx[0], K.uint32(remain_k_blocks))
            )
            * K.uint32(block_k),
        )
        m_offset = local("uint32", shape_m * k_split_idx[0])
        num_total_stages = local(
            "uint32",
            K.uint32(num_k_blocks_per_split)
            + K.cast(k_split_idx[0] < K.uint32(remain_k_blocks), "uint32"),
        )

        cuda_grid_dependency_synchronize()

        # ---------------- roles -- orig:L615/L783 ---------------------------
        # The original dispatches on ``warp_idx < num_mma_warps``; K partitions
        # by warp. The two role blocks are deliberately ADJACENT and unwrapped
        # so K.specialize's chain_dispatch can fold them.
        sp = K.specialize(chain_dispatch=True)
        mma_side = sp.role("mma", warps=list(range(num_mma_warps)))
        cast_side = sp.role("cast", warps=list(range(num_mma_warps, num_warps)))

        with mma_side:
            # ==================== TMA / MMA / D epilogue ====================
            mw = K.warp_id_in_role()  # == warp_idx; this role starts at warp 0

            with K.If(mw == 0), K.Then():
                with K.If(K.cuda.elect_sync()), K.Then():
                    # -------- LOADER (TMA) -------- orig:L616-662
                    tma_state = K.PipelineState(num_stages, phase=1)
                    with K.serial(K.uint32(0), num_total_stages[0]) as s:
                        stage_idx = local("uint32", K.cast(tma_state.stage, "uint32"))
                        smem_pipe.empty.wait(stage_idx[0], K.cast(tma_state.phase, "uint32"))
                        m_idx0 = local("uint32", m_block_idx[0] * K.uint32(block_m))
                        k_idx0 = local("uint32", k_offset[0] + s * K.uint32(block_k))
                        # A remains bf16 (exact in tf32); B's tensor map uses
                        # TFLOAT32 OOB-fill mode 11 so the load RN-truncates as
                        # before. Coordinates are tensor-map order: K, then M/N.
                        K.ptx[tma_g2s_2d](
                            K.ptr_byte_offset(
                                smem_a_mma[0].ptr_to(0, 0),
                                stage_idx[0] * K.uint32(block_m * block_k * 2),
                                "bfloat16",
                            ),
                            K.address_of(a_map),
                            K.cast(k_idx0[0], "int32"),
                            K.cast(m_idx0[0], "int32"),
                            smem_pipe.full.ptr_to([stage_idx[0]]),
                            cache_policy_evict_first,
                        )
                        with K.unroll(num_b_tma_atoms) as b_atom:
                            K.ptx[tma_g2s_2d](
                                K.ptr_byte_offset(
                                    smem_b_mma[0].ptr_to(0, 0),
                                    stage_idx[0] * K.uint32(block_n * block_k * 4)
                                    + K.cast(b_atom * (block_n * block_swizzled_bk * 4), "uint32"),
                                    "float32",
                                ),
                                K.address_of(b_map),
                                K.cast(
                                    k_idx0[0] + K.cast(b_atom * block_swizzled_bk, "uint32"),
                                    "int32",
                                ),
                                K.int32(0),
                                smem_pipe.full.ptr_to([stage_idx[0]]),
                                cache_policy_evict_last,
                            )
                        smem_pipe.full.arrive(
                            stage_idx[0],
                            tx_count=K.uint32(smem_a_size_per_stage + smem_b_size_per_stage),
                        )
                        tma_state.advance()

            with K.If(mw == 1), K.Then():
                # -------- MMA (tcgen05) -------- orig:L664-713
                mma_smem_state = K.PipelineState(num_stages, phase=0)
                mma_cast_state = K.PipelineState(num_cast_stages, phase=0)
                with K.serial(K.uint32(0), num_total_stages[0]) as s:
                    stage_idx = local("uint32", K.cast(mma_smem_state.stage, "uint32"))
                    cast_stage_idx = local("uint32", K.cast(mma_cast_state.stage, "uint32"))
                    cast_pipe.full.wait(cast_stage_idx[0], K.cast(mma_cast_state.phase, "uint32"))
                    # TMEM A columns and the swizzled B matrix descriptor match
                    # the former tcgen05 tile dispatch exactly.
                    a_col = local("int32", K.cast(cast_stage_idx[0] * K.uint32(block_k), "int32"))
                    desc_b = K.local_scalar("uint64")
                    K.cuda.tcgen05.encode_matrix_descriptor(
                        K.address_of(desc_b), smem_b_mma[0].ptr_to(0, 0), ldo=256, sdo=64, swizzle=3
                    )
                    with K.unroll(block_k // umma_k) as ki:
                        with K.If(K.cuda.elect_sync()), K.Then():
                            K.ptx[tcgen05_mma_tf32](
                                K.uint32(d_tmem_start_col),
                                K.cast(a_col[0] + ki * umma_k, "uint32"),
                                add_smem_desc_offset(
                                    desc_b,
                                    (
                                        K.cast((ki // 4) * 1024 + (ki % 4) * 8, "uint32")
                                        + stage_idx[0] * K.uint32(block_n * block_k)
                                    )
                                    // K.uint32(4),
                                ),
                                tf32_instr_desc,
                                K.uint32(0),
                                K.uint32(0),
                                K.uint32(0),
                                K.uint32(0),
                                # NOT a Python conditional: ``ki`` is a symbolic
                                # unroll var, so ``bool(ki == 0)`` is False for
                                # every ki and all eight MMAs would accumulate
                                # unconditionally. The original's Python
                                # conditional expression is what the TVMScript
                                # parser rewrites into exactly this select.
                                K.ptx.pred(K.if_then_else(ki == 0, s != K.uint32(0), K.bool(True))),
                            )
                    with K.If(K.cuda.elect_sync()), K.Then():
                        cast_pipe.empty.arrive(cast_stage_idx[0])
                        smem_pipe.empty.arrive(stage_idx[0])
                    mma_smem_state.advance()
                    mma_cast_state.advance()
                with K.If(K.cuda.elect_sync()), K.Then():
                    tmem_pipe.arrive(0)

            tmem_pipe.wait(0, 0)
            # D epilogue, hand-aligned: 8 x [tcgen05.ld.32x32b.x4 + wait.ld +
            # st.shared.v4 (lane<16) + syncwarp] into the 128B-swizzled smem_cd.
            d_frag = K.alloc_local((4,), "float32")
            d_words = d_frag.view("uint32")
            with K.unroll(block_n // 4) as i:
                taddr_d = local("uint32", K.uint32(d_tmem_start_col + i * 4))
                K.ptx["tcgen05.ld.sync.aligned.32x32b.x4.b32"](
                    d_frag[0], d_frag[1], d_frag[2], d_frag[3], K.uint32(taddr_d[0])
                )
                K.ptx.tcgen05.wait__ld.sync.aligned()
                with K.If(lane_u32[0] < K.uint32(16)), K.Then():
                    # Per-thread 4-col slice store; the offset reproduces the
                    # 128B-swizzled layout the former tile copy selected.
                    m_row = local("uint32", K.cast(mw, "uint32") * K.uint32(16) + lane_u32[0])
                    compose_m = local("uint32", m_row[0] * K.uint32(block_n) + K.uint32(i * 4))
                    compose_q = local("uint32", compose_m[0] // K.uint32(4))
                    smem_cd_offset = local(
                        "uint32",
                        (
                            (compose_q[0] ^ ((compose_q[0] & K.uint32(56)) >> K.uint32(3)))
                            << K.uint32(2)
                        )
                        + compose_m[0] % K.uint32(4),
                    )
                    K.ptx.st.shared.v4.u32(
                        K.ptr_byte_offset(
                            smem_cd_mma.ptr_to(0, 0), smem_cd_offset[0] * K.uint32(4), "float32"
                        ),
                        d_words[0],
                        d_words[1],
                        d_words[2],
                        d_words[3],
                    )
                K.cuda.warp_sync()

            K.ptx.fence.proxy.async_.shared__cta()
            K.ptx.bar.sync(0, K.uint32(num_mma_threads))
            with K.If(mw == 0), K.Then():
                with K.If(K.cuda.elect_sync()), K.Then():
                    # D store via TMA (writes only the valid region of boundary tiles).
                    m0 = local("uint32", m_block_idx[0] * K.uint32(block_m))
                    if num_splits == 1:
                        K.ptx[tma_s2g_2d](
                            K.address_of(d_map),
                            K.int32(0),
                            K.cast(m0[0], "int32"),
                            smem_cd_mma.ptr_to(0, 0),
                            cache_policy_evict_first,
                        )
                    else:
                        ks = local("uint32", k_split_idx[0])
                        K.ptx[tma_s2g_3d](
                            K.address_of(d_map),
                            K.int32(0),
                            K.cast(m0[0], "int32"),
                            K.cast(ks[0], "int32"),
                            smem_cd_mma.ptr_to(0, 0),
                            cache_policy_evict_first,
                        )
                    K.ptx.cp.async_.bulk.commit_group()
            # Keep the TMEM teardown on warp 1, and spell the allocator-slot read
            # explicitly so the low-level IR contains a real PTX shared load.
            with K.If(mw == 1), K.Then():
                K.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
                tmem_dealloc_addr = K.local_scalar("uint32")
                K.ptx.ld.shared.u32(tmem_dealloc_addr, tmem_ptr_in_smem.ptr_to([0]))
                K.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](
                    tmem_dealloc_addr, K.uint32(num_tmem_cols)
                )

        with cast_side:
            # ============== CAST / SUM-OF-SQUARES warps ==============
            # ``warp_id_in_role`` is the original's ``sub_warp_idx``:
            # ``warp_idx - num_mma_warps``.
            sub_warp_idx = local("uint32", K.cast(K.warp_id_in_role(), "uint32"))
            # A cast/deposit register tiles -- orig:L784-822. Each participating
            # thread owns 32 bf16 inputs and 32 fp32 outputs in physical register
            # order; raw ldmatrix and tcgen05.st consume those arrays directly.
            a_bf16_flat = K.alloc_local((cast_per_thread,), "bfloat16")
            a_flat = K.alloc_local((cast_per_thread,), "float32")
            # Dual packed fma.f32x2 sum-of-squares accumulators (the hand kernel's
            # sum0/sum1); the fused form is the only no-regression reduce shape.
            sqr0 = K.alloc_local((2,), "float32")
            sqr1 = K.alloc_local((2,), "float32")
            a_words = a_flat.view("uint32")
            a_bf16_u16 = a_bf16_flat.view("uint16")
            a_bf16_words = a_bf16_flat.view("uint32")
            K.ptx.mov.b32(sqr0[0], K.float32(0))
            K.ptx.mov.b32(sqr0[1], K.float32(0))
            K.ptx.mov.b32(sqr1[0], K.float32(0))
            K.ptx.mov.b32(sqr1[1], K.float32(0))
            cast_smem_state = K.PipelineState(num_stages, phase=0)
            cast_tmem_state = K.PipelineState(num_cast_stages, phase=1)
            # ``unroll=True`` emits the bare ``#pragma unroll`` the original
            # emits, which the postproc then binds. Under the "native"
            # disposition the bound is spelled here instead (NOTES §4); the
            # generated text is the same either way.
            cast_unroll = 12
            with K.serial(K.uint32(0), num_total_stages[0], unroll=cast_unroll) as s:
                stage_idx = local("uint32", K.cast(cast_smem_state.stage, "uint32"))
                cast_stage_idx = local("uint32", K.cast(cast_tmem_state.stage, "uint32"))
                a_col = local("int32", K.cast(cast_stage_idx[0] * K.uint32(block_k), "int32"))
                smem_pipe.full.wait(stage_idx[0], K.cast(cast_smem_state.phase, "uint32"))
                # Four x4 ldmatrix instructions reproduce the warpgroup copy's
                # physical register order. Keep the dispatcher's explicit
                # swizzled element offset so ptxas sees the same address DAG.
                with K.unroll(4) as mm:
                    smem_off = local(
                        "uint32",
                        K.cast(
                            K.cast(sub_warp_idx[0], "int32") * K.int32(1024)
                            + (mm // 2) * K.int32(512),
                            "uint32",
                        )
                        + stage_idx[0] * K.uint32(block_m * block_k)
                        + K.cast(lane_idx % K.int32(8) * K.int32(block_k), "uint32")
                        + (
                            K.cast(
                                (mm % 2) * K.int32(32) + lane_idx // K.int32(8) * K.int32(8),
                                "uint32",
                            )
                            ^ (
                                (
                                    K.cast(
                                        K.cast(sub_warp_idx[0], "int32") * K.int32(16)
                                        + (mm // 2) * K.int32(8),
                                        "uint32",
                                    )
                                    + stage_idx[0] * K.uint32(block_k)
                                    + K.cast(lane_idx % K.int32(8) * K.int32(block_k), "uint32")
                                    // K.uint32(block_k)
                                )
                                & K.uint32(7)
                            )
                            << K.uint32(3)
                        ),
                    )
                    # Unannotated in the original -- and TVMScript binds an
                    # unannotated Expr assignment as a declared local all the
                    # same, so this is a local here too.
                    reg_base = local("int32", (mm % 2) * K.int32(8) + mm // 2)
                    K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_bf16_words[reg_base[0]],
                        a_bf16_words[reg_base[0] + 2],
                        a_bf16_words[reg_base[0] + 4],
                        a_bf16_words[reg_base[0] + 6],
                        K.ptr_byte_offset(smem_a_mma[0].ptr_to(0, 0), smem_off[0] * 2, "bfloat16"),
                    )
                cast_pipe.empty.wait(cast_stage_idx[0], K.cast(cast_tmem_state.phase, "uint32"))

                def sqr_fma(lo, hi, acc):
                    """One packed fma.f32x2 sum-of-squares accumulation."""
                    lhs = K.local_scalar("uint64")
                    rhs = K.local_scalar("uint64")
                    accu = K.local_scalar("uint64")
                    K.ptx.mov.b64(lhs, lo, hi)
                    K.ptx.mov.b64(rhs, lo, hi)
                    K.ptx.mov.b64(accu, acc[0], acc[1])
                    K.ptx.fma.rz.ftz.f32x2(lhs, lhs, rhs, accu)
                    K.ptx.mov.b64(acc[0], acc[1], lhs)

                # bf16->tf32 + sqr-fma + TMEM deposit: interleaved per 8-col atom on
                # short mainloops (hand structure); single wide STTM.x8 on deep
                # pipelines. orig:L882-939.
                if num_k_blocks_per_split <= 16:
                    with K.serial(block_k // 8) as p:
                        with K.serial(2) as f:
                            K.ptx.cvt.f32.bf16(a_flat[p * 4 + f * 2], a_bf16_u16[p * 4 + f * 2])
                            K.ptx.cvt.f32.bf16(
                                a_flat[p * 4 + f * 2 + 1], a_bf16_u16[p * 4 + f * 2 + 1]
                            )
                        # sqr{0,1} += a*a for this atom's packed pair per row.
                        sqr_fma(a_flat[p * 4], a_flat[p * 4 + 1], sqr0)
                        sqr_fma(a_flat[p * 4 + 2], a_flat[p * 4 + 3], sqr1)
                        K.ptx["tcgen05.st.sync.aligned.16x256b.x1.b32"](
                            K.cuda.get_tmem_addr(K.uint32(0), 0, a_col[0] + p * 8),
                            a_words[p * 4],
                            a_words[p * 4 + 1],
                            a_words[p * 4 + 2],
                            a_words[p * 4 + 3],
                        )
                else:
                    with K.serial(cast_per_thread // 2) as f:
                        K.ptx.cvt.f32.bf16(a_flat[f * 2], a_bf16_u16[f * 2])
                        K.ptx.cvt.f32.bf16(a_flat[f * 2 + 1], a_bf16_u16[f * 2 + 1])
                    with K.unroll(cast_pairs) as p:
                        sqr_fma(a_flat[p * 4], a_flat[p * 4 + 1], sqr0)
                        sqr_fma(a_flat[p * 4 + 2], a_flat[p * 4 + 3], sqr1)
                    K.ptx["tcgen05.st.sync.aligned.16x256b.x8.b32"](
                        K.cuda.get_tmem_addr(K.uint32(0), 0, a_col[0]),
                        *[a_words[i] for i in range(cast_per_thread)],
                    )
                K.ptx.tcgen05.wait__st.sync.aligned()
                cast_pipe.full.arrive(cast_stage_idx[0])
                cast_smem_state.advance()
                cast_tmem_state.advance()

            # Cross-lane sum-of-squares reduce over the 4 K-lanes (the hand
            # kernel's shfl_xor 2,1), then store the two per-row results.
            # G3: the activemask/shuffle collectives keep the original's loop
            # placement exactly -- inside the ``spa`` loop, not hoisted.
            sqr_part = K.alloc_local((2,), "float32")
            K.ptx.mov.b32(sqr_part[0], sqr0[0] + sqr0[1])
            K.ptx.mov.b32(sqr_part[1], sqr1[0] + sqr1[1])
            with K.serial(2) as spa:
                reduce_mask = local("uint32", K.tvm_warp_activemask())
                K.ptx.mov.b32(
                    sqr_part[spa],
                    sqr_part[spa]
                    + K.tvm_warp_shuffle_xor(reduce_mask[0], sqr_part[spa], 1, 32, 32),
                )
                K.ptx.mov.b32(
                    sqr_part[spa],
                    sqr_part[spa]
                    + K.tvm_warp_shuffle_xor(reduce_mask[0], sqr_part[spa], 2, 32, 32),
                )
            reduced0 = local("float32", sqr_part[0])
            reduced1 = local("float32", sqr_part[1])
            m_idx0 = local(
                "uint32",
                m_block_idx[0] * K.uint32(block_m)
                + sub_warp_idx[0] * K.uint32(block_m // 4)
                + lane_u32[0] // K.uint32(4),
            )
            m_idx1 = local("uint32", m_idx0[0] + K.uint32(8))
            with K.If((lane_u32[0] % K.uint32(4)) == K.uint32(0)), K.Then():
                with K.If(m_idx0[0] < shape_m), K.Then():
                    K.ptx.st.global_.f32(
                        sqr_sum.ptr_to([K.cast(m_offset[0] + m_idx0[0], "int32")]), reduced0[0]
                    )
                with K.If(m_idx1[0] < shape_m), K.Then():
                    K.ptx.st.global_.f32(
                        sqr_sum.ptr_to([K.cast(m_offset[0] + m_idx1[0], "int32")]), reduced1[0]
                    )

    # orig:L981-989 -- @K.kernel has no attrs= parameter, so the launch-param
    # attribute is attached to the PrimFunc afterwards (entry.py documents
    # ``func`` as a plain attribute and ``Kernel.mod`` reads it).
    sm100_tf32_hc_prenorm_gemm.func = sm100_tf32_hc_prenorm_gemm.func.with_attr(
        "tirx.kernel_launch_params",
        [
            "blockIdx.x",
            "threadIdx.x",
            "tirx.use_programtic_dependent_launch",
            "tirx.use_dyn_shared_memory",
        ],
    )
    return sm100_tf32_hc_prenorm_gemm


class _AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(192)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_tensor_map(
    dtype: str,
    rank: int,
    tensor: torch.Tensor,
    dims: tuple[int, ...],
    strides: tuple[int, ...],
    box: tuple[int, ...],
    swizzle: int,
    *,
    force_cu_dtype: int | None = None,
) -> _AlignedTensorMap:
    descriptor = _AlignedTensorMap()
    args = [
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
    ]
    if force_cu_dtype is not None:
        args.append(force_cu_dtype)
    tvm.get_global_func("runtime.cuTensorMapEncodeTiled")(*args)
    return descriptor


def _build_tirx_tensor_maps(data: dict[str, Any]) -> tuple[Any, Any, Any]:
    config: TF32HCPrenormGemmConfig = data["config"]
    block_swizzled_bk = min(config.block_k * 4, 128) // 4
    if config.num_splits == 1:
        d_map = _encode_tensor_map(
            "float32",
            2,
            data["d_tirx"],
            (config.n, config.m),
            (config.n * 4,),
            (config.block_n, config.block_m),
            3,
        )
    else:
        d_map = _encode_tensor_map(
            "float32",
            3,
            data["d_tirx"],
            (config.n, config.m, config.num_splits),
            (config.n * 4, config.m * config.n * 4),
            (config.block_n, config.block_m, 1),
            3,
        )
    b_map = _encode_tensor_map(
        "float32",
        2,
        data["b"],
        (config.k, config.n),
        (config.k * 4,),
        (block_swizzled_bk, config.block_n),
        3,
        force_cu_dtype=11,
    )
    a_map = _encode_tensor_map(
        "bfloat16",
        2,
        data["a"],
        (config.k, config.m),
        (config.k * 2,),
        (config.block_k, config.block_m),
        3,
    )
    return a_map, b_map, d_map


def get_kernel(**kwargs: Any):
    config = _make_config(**kwargs)
    return _make_kernel(**asdict(config)).func


def _compile_tirx_tf32_hc_for_config(
    *, m: int, n: int, k: int, num_splits: int, seed: int, num_sms: int
) -> Any:
    from tirx_kernels.runner import cuda_target

    target = cuda_target()
    kernel = get_kernel(m=m, n=n, k=k, num_splits=num_splits, seed=seed, num_sms=num_sms)
    with target:
        return tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")


_compile_tirx_tf32_hc_for_config = cache(_compile_tirx_tf32_hc_for_config)


def _compile_tirx_tf32_hc_key(config: TF32HCPrenormGemmConfig) -> tuple[tuple[str, Any], ...]:
    return tuple(asdict(config).items())


def _compile_tirx_tf32_hc(config: TF32HCPrenormGemmConfig) -> Any:
    compile_kwargs = asdict(config)
    return _compile_tirx_tf32_hc_for_config(**compile_kwargs)


def _run_tirx_with_tensor_maps(
    data: dict[str, Any], executable: Any, tensor_maps: tuple[Any, Any, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    config: TF32HCPrenormGemmConfig = data["config"]
    executable.mod(
        config.m,
        data["a"].view(-1),
        data["b"].view(-1),
        data["d_tirx"].view(-1),
        data["sqr_tirx"].view(-1),
        *(tensor_map.ptr for tensor_map in tensor_maps),
    )
    return data["d_tirx"], data["sqr_tirx"]


def _launch_tirx_hc(data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_tirx_with_tensor_maps(
        data, _compile_tirx_tf32_hc(data["config"]), _build_tirx_tensor_maps(data)
    )


def _run_deepgemm_hc(data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    config: TF32HCPrenormGemmConfig = data["config"]
    data["deep_gemm"].tf32_hc_prenorm_gemm(
        data["a"],
        data["b"],
        data["d_deepgemm"],
        data["sqr_deepgemm"],
        num_splits=None if config.num_splits == 1 else config.num_splits,
    )
    return data["d_deepgemm"], data["sqr_deepgemm"]


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
    diff = float((1 - sim).item())
    return diff if math.isfinite(diff) else float("inf")


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


def _assert_correct_case(
    case: TF32HCBenchCase, d: torch.Tensor, sqr_sum: torch.Tensor, *, name: str
) -> float:
    final_d, final_sqr = _final_outputs(d, sqr_sum, case.config)
    diff = max(_calc_diff(final_d, case.reference_d), _calc_diff(final_sqr, case.reference_sqr))
    if diff >= _TEST_DIFF_THRESHOLD:
        raise AssertionError(f"{name} diff {diff:.10g} >= {_TEST_DIFF_THRESHOLD}")
    return diff


def run_test(**kwargs: Any) -> None:
    data = prepare_data(**kwargs)
    deepgemm_d, deepgemm_sqr = _run_deepgemm_hc(data)
    torch.cuda.synchronize()
    deepgemm_diff = _assert_correct(data, deepgemm_d, deepgemm_sqr, name="DeepGEMM")
    tirx_d, tirx_sqr = _launch_tirx_hc(data)
    torch.cuda.synchronize()
    tirx_diff = _assert_correct(data, tirx_d, tirx_sqr, name="TIRx")
    if tirx_diff > max(deepgemm_diff, _TEST_DIFF_THRESHOLD):
        raise AssertionError(
            f"TIRx diff {tirx_diff:.10g} is worse than DeepGEMM diff {deepgemm_diff:.10g}"
        )


def _make_bench_case(config_kwargs: dict[str, Any]) -> TF32HCBenchCase:
    data = prepare_data(**config_kwargs)
    return TF32HCBenchCase(
        config=data["config"],
        deep_gemm=data["deep_gemm"],
        a=data["a"],
        b=data["b"],
        d_deepgemm=data["d_deepgemm"],
        sqr_deepgemm=data["sqr_deepgemm"],
        d_tirx=data["d_tirx"],
        sqr_tirx=data["sqr_tirx"],
        reference_d=data["reference_d"],
        reference_sqr=data["reference_sqr"],
        tensor_maps=_build_tirx_tensor_maps(data),
    )


def _bench_tirx_case(case: TF32HCBenchCase, executable: Any) -> tuple[torch.Tensor, torch.Tensor]:
    executable.mod(
        case.config.m,
        case.a.view(-1),
        case.b.view(-1),
        case.d_tirx.view(-1),
        case.sqr_tirx.view(-1),
        *(tensor_map.ptr for tensor_map in case.tensor_maps),
    )
    return case.d_tirx, case.sqr_tirx


def _bench_deepgemm_case(case: TF32HCBenchCase) -> tuple[torch.Tensor, torch.Tensor]:
    case.deep_gemm.tf32_hc_prenorm_gemm(
        case.a,
        case.b,
        case.d_deepgemm,
        case.sqr_deepgemm,
        num_splits=None if case.config.num_splits == 1 else case.config.num_splits,
    )
    return case.d_deepgemm, case.sqr_deepgemm


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
    timer = kwargs.pop("timer", None)  # None inherits the global default (proton)
    warmup = kwargs.pop("warmup", None)
    repeat = kwargs.pop("repeat", None)
    _rounds = kwargs.pop("rounds", 1)
    _cooldown_s = kwargs.pop("cooldown_s", 1.0)
    config_kwargs = dict(kwargs)
    executable = prepared["executable"]

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    case = _make_bench_case(config_kwargs)

    # Correctness gate for our kernel before timing (preserves the tirx half of
    # the old validate_case; the deepgemm reference is trusted).
    tirx_d, tirx_sqr = _bench_tirx_case(case, executable)
    torch.cuda.synchronize()
    tirx_diff = _assert_correct_case(case, tirx_d, tirx_sqr, name="TIRx")

    funcs = {"tirx": lambda: _bench_tirx_case(case, executable)}

    def _deepgemm():
        source_d, source_sqr = _bench_deepgemm_case(case)
        torch.cuda.synchronize()
        _assert_correct_case(case, source_d, source_sqr, name="DeepGEMM")
        return lambda: _bench_deepgemm_case(case)

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
    result["tirx_diff"] = tirx_diff
    result["max_diff"] = tirx_diff
    from tirx_kernels.reference_variants import reference_provenance

    result["reference_variant"] = reference_provenance("deep-gemm")
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
    "TF32HCPrenormGemmConfig",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
