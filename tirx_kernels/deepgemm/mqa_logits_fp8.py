# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any
from unittest import SkipTest

import torch

from tvm.backend.cuda.op import cuda_func_call

_DEEP_GEMM_MODULE_NAME = "deep_gemm"
_SM100_SMEM_CAPACITY = 232448
_TEST_DIFF_THRESHOLD = 5e-6


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


def _mqa_fp8_wrelu_reduce_src(num_heads: int) -> str:
    """DeepGEMM f32 wrelu epilogue (sm100_mqa_logits.cuh): relu as x+|x| packed into
    FADD2+FFMA2, tail /2; emitted as a kernel-local helper via T.cuda_func_call."""
    return (
        f"__forceinline__ __device__ float tvm_builtin_mqa_fp8_wrelu_reduce_{num_heads}("
        "const float* __restrict__ accum, const float* __restrict__ weights) {\n"
        "    float2 sum_0 = make_float2(0.0f, 0.0f);\n"
        "    float2 sum_1 = make_float2(0.0f, 0.0f);\n"
        "    #pragma unroll\n"
        f"    for (int j = 0; j < {num_heads}; j += 4) {{\n"
        "        float2 a_0 = make_float2(accum[j], accum[j + 1]);\n"
        "        float2 a_1 = make_float2(fabsf(accum[j]), fabsf(accum[j + 1]));\n"
        "        sum_0 = __ffma2_rn(__fadd2_rn(a_0, a_1),"
        " make_float2(weights[j], weights[j + 1]), sum_0);\n"
        "        float2 a_2 = make_float2(accum[j + 2], accum[j + 3]);\n"
        "        float2 a_3 = make_float2(fabsf(accum[j + 2]), fabsf(accum[j + 3]));\n"
        "        sum_1 = __ffma2_rn(__fadd2_rn(a_2, a_3),"
        " make_float2(weights[j + 2], weights[j + 3]), sum_1);\n"
        "    }\n"
        "    float2 sum = __fadd2_rn(sum_0, sum_1);\n"
        "    return (sum.x + sum.y) / 2.0f;\n"
        "}"
    )


def get_kernel(**kwargs: Any):
    from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
    from tvm.script import tirx as T
    from tvm.script.tirx import tile as Tx
    from tvm.tirx.lang.pipeline import Pipeline
    from tvm.tirx.layout import S, TCol, TileLayout, TLane, wg_local_layout

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

    def lane_id_u32():
        return T.cast(T.ptx.fetch_register(32, "laneid"), "uint32")

    def cuda_grid_dependency_synchronize():
        T.evaluate(T.ptx.griddepcontrol.wait())

    def named_barrier_sync_8(count):
        T.evaluate(T.ptx.bar.sync(8, count))

    @T.prim_func
    def sm100_fp8_mqa_logits(
        seq_len: T.uint32,
        seq_len_kv: T.uint32,
        max_seqlen_k: T.uint32,
        logits_stride: T.uint32,
        cu_seq_len_k_start_h: T.handle,
        cu_seq_len_k_end_h: T.handle,
        logits_h: T.handle,
        q_gmem_h: T.handle,
        kv_gmem_h: T.handle,
        kv_scales_gmem_h: T.handle,
        weights_gmem_h: T.handle,
    ):
        # seq_len / seq_len_kv are RUNTIME like DeepGEMM: one compiled kernel serves any
        # length; structure stays compile-time. match_buffer must precede device_entry.
        cu_seq_len_k_start = T.match_buffer(cu_seq_len_k_start_h, (seq_len,), "int32")
        cu_seq_len_k_end = T.match_buffer(cu_seq_len_k_end_h, (seq_len,), "int32")
        logits = T.match_buffer(
            logits_h,
            (
                (T.cast(seq_len, "int32") + T.int32(block_q - 1))
                // T.int32(block_q)
                * T.int32(block_q),
                T.cast(logits_stride, "int32"),
            ),
            config.logits_dtype,
        )
        q_gmem = T.match_buffer(q_gmem_h, (seq_len * num_heads, head_dim), "uint8")
        kv_gmem = T.match_buffer(kv_gmem_h, (seq_len_kv, head_dim), "uint8")
        kv_scales_gmem = T.match_buffer(kv_scales_gmem_h, (seq_len_kv,), "float32")
        weights_gmem = T.match_buffer(weights_gmem_h, (seq_len, num_heads), "float32")
        T.device_entry()
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
        # TIRX_TRANSCRIBE_START sm100_fp8_mqa_logits
        aligned_sl: T.int32 = (
            (T.cast(seq_len, "int32") + T.int32(block_q - 1)) // T.int32(block_q) * T.int32(block_q)
        )
        logits_flat = T.decl_buffer(
            (aligned_sl * T.cast(logits_stride, "int32"),),
            logits_tir_dtype,
            data=logits.data,
            scope="global",
        )
        num_q_blocks: T.uint32 = (seq_len + T.uint32(block_q - 1)) // T.uint32(block_q)
        sm_idx = T.cta_id([config.num_sms])
        sm_idx_u32: T.let = T.cast(sm_idx, "uint32")
        warp_idx = T.warp_id([num_warps])
        warp_idx_u32: T.let = T.cast(warp_idx, "uint32")
        warpgroup_idx = T.warpgroup_id([num_warps // 4])
        lane_idx = T.lane_id([32])
        lane_idx_u32: T.let = lane_id_u32()

        # SMEMPool owns the smem offsets; q/kv carry the 128B MMA swizzle layout. Under
        # the pool .data is the arena base — use .ptr_to([0,...]) for a buffer's start.
        pool = T.SMEMPool()
        smem_q = pool.alloc(
            (num_q_stages, block_q * num_heads, head_dim),
            "uint8",
            scope="shared.dyn",
            align=swizzle_alignment,
            layout=mma_shared_layout("uint8", _SWZ, (num_q_stages, block_q * num_heads, head_dim)),
        )
        smem_weights = pool.alloc((num_q_stages, block_q, num_heads), "float32", align=16)
        smem_kv = pool.alloc(
            (num_kv_stages, block_kv, head_dim),
            "uint8",
            scope="shared.dyn",
            align=swizzle_alignment,
            layout=mma_shared_layout("uint8", _SWZ, (num_kv_stages, block_kv, head_dim)),
        )
        smem_kv_scales = pool.alloc((num_kv_stages, block_kv), "float32", align=16)
        # Producer/consumer barrier pairs as Pipeline objects (full = data ready, empty
        # = slot free); each Pipeline runs mbarrier.init itself, no separate init loop.
        q_pipe = Pipeline(
            pool,
            num_q_stages,
            full="tma",
            empty="mbar",
            init_full=1,
            init_empty=num_math_threads + 1,
        )
        kv_pipe = Pipeline(
            pool, num_kv_stages, full="tma", empty="mbar", init_full=1, init_empty=num_math_threads
        )
        tmem_pipe = Pipeline(
            pool, num_tmem_stages, full="tcgen05", empty="mbar", init_full=1, init_empty=128
        )
        tmem_ptr_in_smem = pool.alloc((1,), "uint32", align=4)
        pool.commit()
        # TMEMPool gives a CONSTANT 0-based col_start so tmem addressing folds the base
        # into the col offset; manual tcgen05.alloc/dealloc below keep the lifecycle.
        tmem_pool = T.TMEMPool(
            pool, total_cols=num_tmem_cols, cta_group=1, tmem_addr=tmem_ptr_in_smem
        )
        tmem = tmem_pool.alloc(
            (128, num_tmem_cols), "float32", layout=tmem_layout, cols=num_tmem_cols
        )
        seq_k_start = T.alloc_local((block_q,), "uint32")
        seq_k_end = T.alloc_local((block_q,), "uint32")
        schedule_result = T.alloc_local((2,), "uint32")

        @T.inline
        def store_logits(flat_offset, value):
            # Scalar predicated store: per-thread non-contiguous output, so TMA/bulk
            # does not apply; bf16 stays a plain buffer store (predicated @P STG.E.U16).
            if config.logits_dtype == "float32":
                T.ptx.st(logits_flat.ptr_to([flat_offset]), value, space="global", ptx_type="f32")
            else:
                logits_flat[flat_offset] = value

        @T.inline
        def load_schedule(q_idx):
            schedule_start: T.uint32 = T.uint32(0xFFFFFFFF)
            schedule_end: T.uint32 = T.uint32(0)
            for schedule_i in T.unroll(0, block_q):
                row_idx: T.uint32 = T.min(
                    q_idx * T.uint32(block_q) + T.uint32(schedule_i), seq_len - T.uint32(1)
                )
                seq_k_start[schedule_i] = T.min(
                    T.cast(cu_seq_len_k_start[row_idx], "uint32"), seq_len_kv
                )
                seq_k_end[schedule_i] = T.min(
                    T.cast(cu_seq_len_k_end[row_idx], "uint32"), seq_len_kv
                )
                schedule_start = T.min(schedule_start, seq_k_start[schedule_i])
                schedule_end = T.max(schedule_end, seq_k_end[schedule_i])
            schedule_start = schedule_start // T.uint32(4) * T.uint32(4)
            num_kv_blocks = (schedule_end - schedule_start + T.uint32(block_kv - 1)) // T.uint32(
                block_kv
            )
            schedule_result[0] = schedule_start
            schedule_result[1] = num_kv_blocks

        # Pipeline constructors already ran mbarrier.init; fence + cta_sync publish them.
        T.ptx.fence.mbarrier_init()
        if warp_idx == spec_warp_start + 2:
            T.ptx.tcgen05.alloc(
                T.address_of(tmem_ptr_in_smem[0]), n_cols=num_tmem_cols, cta_group=1
            )
        T.cuda.cta_sync()

        cuda_grid_dependency_synchronize()

        if warp_idx == spec_warp_start:
            T.ptx.setmaxnreg(False, 56)
            if T.ptx.elect_sync():
                # Ring cursors with subtract-wrap (DeepGEMM RingPipeline): avoids ptxas
                # magic-number division for `% kNumStages` on these hot paths.
                q_stage_idx: T.uint32 = T.uint32(0)
                q_phase: T.uint32 = T.uint32(0)
                q_idx: T.uint32 = sm_idx_u32
                while q_idx < num_q_blocks:
                    q_pipe.empty.wait(q_stage_idx, q_phase ^ T.uint32(1))
                    # u32 row bases — the copy_async(tma) gmem-layout grouping now
                    # handles unsigned shape extents (no int32 cast needed).
                    q_row0: T.uint32 = q_idx * T.uint32(block_q * num_heads)
                    Tx.copy_async(
                        smem_q[q_stage_idx],
                        q_gmem[q_row0 : q_row0 + block_q * num_heads, :],
                        dispatch="tma_auto",
                        mbar=q_pipe.full.ptr_to([q_stage_idx]),
                        cta_group=1,
                        cache_hint="evict_normal",
                        prefetch_tensormap=True,
                    )
                    q_blk0: T.uint32 = q_idx * T.uint32(block_q)
                    Tx.copy_async(
                        smem_weights[q_stage_idx],
                        weights_gmem[q_blk0 : q_blk0 + block_q, :],
                        dispatch="tma_auto",
                        mbar=q_pipe.full.ptr_to([q_stage_idx]),
                        cta_group=1,
                        cache_hint="evict_normal",
                        prefetch_tensormap=True,
                    )
                    q_pipe.full.arrive(
                        q_stage_idx, tx_count=smem_q_size_per_stage + smem_weight_size_per_stage
                    )
                    q_idx = q_idx + T.uint32(config.num_sms)
                    q_stage_idx = q_stage_idx + T.uint32(1)
                    if q_stage_idx >= T.uint32(num_q_stages):
                        q_stage_idx = q_stage_idx - T.uint32(num_q_stages)
                        q_phase = q_phase ^ T.uint32(1)
            T.cuda.warp_sync()
        elif warp_idx == spec_warp_start + 1:
            T.ptx.setmaxnreg(False, 56)
            if T.ptx.elect_sync():
                kv_stage_idx: T.uint32 = T.uint32(0)
                kv_phase: T.uint32 = T.uint32(0)
                q_idx: T.uint32 = sm_idx_u32
                while q_idx < num_q_blocks:
                    load_schedule(q_idx)
                    kv_start: T.uint32 = schedule_result[0]
                    num_kv_blocks: T.uint32 = schedule_result[1]
                    kv_idx: T.uint32 = T.uint32(0)
                    while kv_idx < num_kv_blocks:
                        kv_pipe.empty.wait(kv_stage_idx, kv_phase ^ T.uint32(1))
                        kv_row0: T.uint32 = kv_start + kv_idx * T.uint32(block_kv)
                        Tx.copy_async(
                            smem_kv[kv_stage_idx],
                            kv_gmem[kv_row0 : kv_row0 + block_kv, :],
                            dispatch="tma_auto",
                            mbar=kv_pipe.full.ptr_to([kv_stage_idx]),
                            cta_group=1,
                            cache_hint="evict_normal",
                            prefetch_tensormap=True,
                        )
                        Tx.copy_async(
                            smem_kv_scales[kv_stage_idx, 0:block_kv],
                            kv_scales_gmem[kv_row0 : kv_row0 + block_kv],
                            dispatch="tma_auto",
                            mbar=kv_pipe.full.ptr_to([kv_stage_idx]),
                            cta_group=1,
                            cache_hint="evict_normal",
                            prefetch_tensormap=True,
                        )
                        kv_pipe.full.arrive(
                            kv_stage_idx,
                            tx_count=smem_kv_size_per_stage + smem_kv_scale_size_per_stage,
                        )
                        kv_idx = kv_idx + T.uint32(1)
                        kv_stage_idx = kv_stage_idx + T.uint32(1)
                        if kv_stage_idx >= T.uint32(num_kv_stages):
                            kv_stage_idx = kv_stage_idx - T.uint32(num_kv_stages)
                            kv_phase = kv_phase ^ T.uint32(1)
                    q_idx = q_idx + T.uint32(config.num_sms)
        elif warp_idx == spec_warp_start + 2:
            T.ptx.setmaxnreg(False, 56)
            T.cuda.trap_when_assert_failed(tmem_ptr_in_smem[0] == T.uint32(0))
            # fp8 (e4m3) views over the 128B-swizzled uint8 SMEM; with descI omitted the
            # tcgen05 dispatch constant-folds the dense instruction descriptor.
            smem_q_fp8 = smem_q.view("float8_e4m3fn")
            smem_kv_fp8 = smem_kv.view("float8_e4m3fn")
            # Whole MMA-warp loop in one elect scope: ring cursors stay elect-lane
            # locals on the uniform datapath (no R2UR per use).
            if T.ptx.elect_sync():
                q_stage_idx: T.uint32 = T.uint32(0)
                q_phase: T.uint32 = T.uint32(0)
                kv_stage_idx: T.uint32 = T.uint32(0)
                kv_phase: T.uint32 = T.uint32(0)
                tmem_stage_idx: T.uint32 = T.uint32(0)
                tmem_phase: T.uint32 = T.uint32(0)
                q_idx: T.uint32 = sm_idx_u32
                while q_idx < num_q_blocks:
                    load_schedule(q_idx)
                    num_kv_blocks: T.uint32 = schedule_result[1]
                    q_pipe.full.wait(q_stage_idx, q_phase)
                    kv_idx: T.uint32 = T.uint32(0)
                    while kv_idx < num_kv_blocks:
                        kv_pipe.full.wait(kv_stage_idx, kv_phase)
                        for math_wg_i in T.unroll(0, num_math_warpgroups):
                            tmem_addr: T.uint32 = tmem_stage_idx * T.uint32(umma_n)
                            # int32 column base: a uint32 base would promote the slice
                            # extent and fail the gemm dispatch's StructuralEqual check.
                            tmem_col_start: T.int32 = T.cast(tmem_addr, "int32")
                            tmem_pipe.empty.wait(tmem_stage_idx, tmem_phase ^ T.uint32(1))
                            # D = KV @ Q^T over full head_dim in one issue; the tcgen05
                            # dispatch unrolls the umma_k tiling and accumulates internally.
                            Tx.gemm_async(
                                tmem[:, tmem_col_start : tmem_col_start + umma_n],
                                smem_kv_fp8[
                                    kv_stage_idx,
                                    math_wg_i * umma_m : math_wg_i * umma_m + umma_m,
                                    :,
                                ],
                                smem_q_fp8[q_stage_idx, :, :],
                                accum=False,
                                dispatch="tcgen05",
                                cta_group=1,
                                # SMEM descriptor handling: see the gemm_async tcgen05
                                # dispatch for the hoist-vs-recompute rationale.
                                smem_desc="hoist",
                            )
                            tmem_pipe.full.arrive(tmem_stage_idx)
                            tmem_stage_idx = tmem_stage_idx + T.uint32(1)
                            if tmem_stage_idx >= T.uint32(num_tmem_stages):
                                tmem_stage_idx = tmem_stage_idx - T.uint32(num_tmem_stages)
                                tmem_phase = tmem_phase ^ T.uint32(1)
                        kv_idx = kv_idx + T.uint32(1)
                        kv_stage_idx = kv_stage_idx + T.uint32(1)
                        if kv_stage_idx >= T.uint32(num_kv_stages):
                            kv_stage_idx = kv_stage_idx - T.uint32(num_kv_stages)
                            kv_phase = kv_phase ^ T.uint32(1)
                    q_pipe.empty.arrive(q_stage_idx)
                    q_idx = q_idx + T.uint32(config.num_sms)
                    q_stage_idx = q_stage_idx + T.uint32(1)
                    if q_stage_idx >= T.uint32(num_q_stages):
                        q_stage_idx = q_stage_idx - T.uint32(num_q_stages)
                        q_phase = q_phase ^ T.uint32(1)
            T.cuda.warp_sync()
        elif warp_idx == spec_warp_start + 3:
            T.ptx.setmaxnreg(False, 56)
        elif warp_idx < spec_warp_start:
            T.ptx.setmaxnreg(True, 224)
            math_thread_idx: T.uint32 = warp_idx_u32 * T.uint32(32) + lane_idx_u32
            accum = T.alloc_local((num_heads,), "float32")
            cached_weights = T.alloc_local((block_q, num_heads), "float32")
            # Per-q-row logits base offset (= q_row * logits_stride): invariant across
            # the kv loop, so compute once per q block.
            # Single u64 row base; ptxas spilled the block_q-wide u64 array,
            # costing an LDL reload per clamped store.
            q_row_off_base = T.alloc_local((1,), "uint64")
            q_stage_idx: T.uint32 = T.uint32(0)
            q_phase: T.uint32 = T.uint32(0)
            kv_stage_idx: T.uint32 = T.uint32(0)
            kv_phase: T.uint32 = T.uint32(0)
            tmem_stage_idx: T.uint32 = T.cast(warpgroup_idx, "uint32")
            tmem_phase: T.uint32 = T.uint32(0)
            q_idx: T.uint32 = sm_idx_u32
            while q_idx < num_q_blocks:
                load_schedule(q_idx)
                kv_start: T.uint32 = schedule_result[0]
                num_kv_blocks: T.uint32 = schedule_result[1]
                q_pipe.full.wait(q_stage_idx, q_phase)
                if num_kv_blocks > T.uint32(0):
                    Tx.warpgroup.copy(cached_weights, smem_weights[q_stage_idx])
                    q_row_off_base[0] = T.cast(q_idx * T.uint32(block_q), "uint64") * T.cast(
                        logits_stride, "uint64"
                    )
                    # Publish the generic-proxy weight reads before this consumer
                    # eventually releases the Q stage for a subsequent TMA overwrite.
                    T.ptx.fence.proxy_async("shared::cta")
                    kv_offset: T.uint32 = kv_start + math_thread_idx
                    kv_idx: T.uint32 = T.uint32(0)
                    while kv_idx < num_kv_blocks:
                        kv_pipe.full.wait(kv_stage_idx, kv_phase)
                        scale_kv: T.float32 = T.ptx.ld(
                            smem_kv_scales.ptr_to([kv_stage_idx, math_thread_idx]),
                            "float32",
                            "f32",
                            space="shared",
                        )
                        tmem_pipe.full.wait(tmem_stage_idx, tmem_phase)
                        # Release the kv stage only after the tmem accumulator is
                        # ready, and fence the generic-proxy scale_kv read first.
                        T.ptx.fence.proxy_async("shared::cta")
                        kv_pipe.empty.arrive(kv_stage_idx)
                        tmem_stage_base: T.uint32 = tmem_stage_idx * T.uint32(umma_n)
                        for q_inner_i in T.unroll(0, block_q):
                            tmem_addr: T.uint32 = tmem_stage_base + T.uint32(q_inner_i * num_heads)
                            # TMEM->register read as 2x tcgen05.ld.32x32b.x32 with a
                            # fence between (DeepGEMM tmem_load_no_fence shape for 64 heads).
                            accum_2d = accum.view(128, num_heads, layout=wg_local_layout(num_heads))
                            tmem_addr_hi: T.uint32 = tmem_addr + T.uint32(num_heads // 2)
                            Tx.warpgroup.copy_async(
                                accum_2d[:, 0 : num_heads // 2],
                                tmem[:, tmem_addr : tmem_addr + num_heads // 2],
                            )
                            T.ptx.tcgen05.wait.ld()
                            Tx.warpgroup.copy_async(
                                accum_2d[:, num_heads // 2 : num_heads],
                                tmem[:, tmem_addr_hi : tmem_addr_hi + num_heads // 2],
                            )
                            T.ptx.tcgen05.wait.ld()
                            # Weighted-ReLU reduce via inline CUDA (see _mqa_fp8_wrelu_reduce_src).
                            reduced: T.float32 = cuda_func_call(
                                f"tvm_builtin_mqa_fp8_wrelu_reduce_{num_heads}",
                                T.address_of(accum[0]),
                                T.address_of(cached_weights[q_inner_i, 0]),
                                source_code=_mqa_fp8_wrelu_reduce_src(num_heads),
                                return_type="float32",
                            )
                            result = T.cast(scale_kv * reduced, logits_tir_dtype)
                            q_offset: T.uint64 = q_row_off_base[0] + T.cast(
                                q_inner_i, "uint64"
                            ) * T.cast(logits_stride, "uint64")
                            if config.compressed_logits:
                                # Unconditional store with the column clamped into the
                                # row's stride padding (a range guard becomes a BSSY/BRA region).
                                rel_kv: T.uint32 = kv_offset - seq_k_start[q_inner_i]
                                col: T.uint32 = T.min(rel_kv, logits_stride - T.uint32(1))
                                store_logits(q_offset + T.cast(col, "uint64"), result)
                            else:
                                store_logits(q_offset + T.cast(kv_offset, "uint64"), result)
                        # Release this tmem stage once per kv block AFTER the token loop;
                        # inside the last token ptxas fuses it with the compressed guard branch.
                        tmem_pipe.empty.arrive(tmem_stage_idx)
                        kv_idx = kv_idx + T.uint32(1)
                        kv_offset = kv_offset + T.uint32(block_kv)
                        kv_stage_idx = kv_stage_idx + T.uint32(1)
                        if kv_stage_idx >= T.uint32(num_kv_stages):
                            kv_stage_idx = kv_stage_idx - T.uint32(num_kv_stages)
                            kv_phase = kv_phase ^ T.uint32(1)
                        tmem_stage_idx = tmem_stage_idx + T.uint32(num_math_warpgroups)
                        if tmem_stage_idx >= T.uint32(num_tmem_stages):
                            tmem_stage_idx = tmem_stage_idx - T.uint32(num_tmem_stages)
                            tmem_phase = tmem_phase ^ T.uint32(1)
                q_pipe.empty.arrive(q_stage_idx)
                q_idx = q_idx + T.uint32(config.num_sms)
                q_stage_idx = q_stage_idx + T.uint32(1)
                if q_stage_idx >= T.uint32(num_q_stages):
                    q_stage_idx = q_stage_idx - T.uint32(num_q_stages)
                    q_phase = q_phase ^ T.uint32(1)
            named_barrier_sync_8(T.uint32(num_math_threads))
            if warp_idx == 0:
                T.ptx.tcgen05.dealloc(T.uint32(0), n_cols=num_tmem_cols, cta_group=1)

    sm100_fp8_mqa_logits = sm100_fp8_mqa_logits.with_attr("tirx.persistent_kernel", True)

    return sm100_fp8_mqa_logits.with_attr(
        "tirx.kernel_launch_params",
        [
            "blockIdx.x",
            "threadIdx.x",
            "tirx.use_programtic_dependent_launch",
            "tirx.use_dyn_shared_memory",
        ],
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


def _compile_tirx_mqa(config: MQALogitsFP8Config, max_seqlen_k: int) -> Any:
    # The kernel is independent of seq_len/seq_len_kv/disable_cp/logits_stride (all
    # runtime): canonical values let the cache dedup to one kernel per structural config.
    return _compile_tirx_mqa_for_config(
        seq_len=config.block_q,
        seq_len_kv=config.block_kv,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        logits_dtype=config.logits_dtype,
        compressed_logits=config.compressed_logits,
        disable_cp=True,
        num_sms=config.num_sms,
        logits_stride_override=None,
    )


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
    # Raw GMEM buffers: the in-kernel copy_async(tma) builds the descriptors; Q/KV move
    # as uint8 (fp8 e4m3 bytes), KV-scales and weights stay float32.
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


def run_bench(**kwargs: Any) -> dict[str, Any]:
    from tvm.tirx.bench import bench

    warmup = kwargs.pop("warmup", None)
    repeat = kwargs.pop("repeat", None)
    timer = kwargs.pop("timer", None)  # None inherits the global default (proton)
    _rounds = kwargs.pop("rounds", 1)
    _cooldown_s = kwargs.pop("cooldown_s", 1.0)
    config_kwargs = dict(kwargs)
    tirx_executable = _compile_tirx_mqa(_make_config(**config_kwargs), 0)

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
