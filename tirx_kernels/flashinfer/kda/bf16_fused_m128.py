# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400),
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""TIRx port of FlashInfer's FlashKDA SM100a BF16 recurrent-KDA prefill
M128 kernel (flashinfer-ai/flashinfer#4262, head e835e0f5).

Source CUDA: csrc/kda/flashkda_bf16_fused_m128.cu (kernel body at line 504),
launched by RunM128 in csrc/kda/flashkda_bf16_fused_m128_binding.cu with
shared launch helpers in csrc/kda/flashkda_binding_common.cuh.

Target instance: BF16, head_dim 128, sm_100a, grid = num_seqs * num_heads,
1024 threads, 227328 B dynamic smem, six host-encoded TMA descriptors.
"""

import math
from dataclasses import dataclass, fields
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K

D_HEAD = 128
THREADS = 1024
SMEM_TOTAL = 227328
DEFAULT_LOWER_BOUND = -5.0

# Canonical five-stage shared allocation.  Every typed element stride and raw
# byte stride below derives from this one shape; the overlay views keep their
# source offsets, but no role reconstructs the stage geometry independently.
SMEM_STAGE_ALLOCATION = (5, 20992)
SMEM_STAGE_BF16_STRIDE = SMEM_STAGE_ALLOCATION[1]
SMEM_STAGE_F32_STRIDE = SMEM_STAGE_BF16_STRIDE // 2
SMEM_STAGE_BYTE_STRIDE = SMEM_STAGE_BF16_STRIDE * 2

# .cu:29-106 (#define block, source order).
TMEM_TMEM_STATE_OFFSET = 64
TMEM_TMEM_STATE_INP_OFFSET = 0
TMEM_TMEM_U_ACC_OFFSET = 224
TMEM_TMEM_U2_INP_OFFSET = 224
TMEM_TMEM_U2_ACC_OFFSET = 0
TMEM_TMEM_OUT_OFFSET = 192
TMEM_TMEM_STATE_OUT_OFFSET = 64
SMEM_SMEM_G_RAW_ALL_OFF = 1024
SMEM_SMEM_G_RAW_ALL_STAGE_BYTES = 176128
SMEM_SMEM_RESTORE_FACTOR_ALL_OFF = 41984
SMEM_SMEM_RESTORE_FACTOR_ALL_STAGE_BYTES = 168452
SMEM_SMEM_GT_PREFIX_ALL_OFF = 41472
SMEM_SMEM_GT_PREFIX_ALL_STAGE_BYTES = 168448
SMEM_SMEM_GT_ALL_OFF = 31744
SMEM_SMEM_GT_ALL_STAGE_BYTES = 168448
SMEM_SMEM_PREP_BETA_ALL_OFF = 42500
SMEM_SMEM_PREP_BETA_ALL_STAGE_BYTES = 168064
SMEM_SMEM_GATE_RATE_ALL_OFF = 42628
SMEM_SMEM_GATE_RATE_ALL_STAGE_BYTES = 167940
SMEM_SMEM_V_ALL_OFF = 32384
SMEM_SMEM_V_ALL_STAGE_BYTES = 176128
SMEM_SMEM_GATE_ALL_OFF = 25600
SMEM_SMEM_GATE_ALL_STAGE_BYTES = 184320

# TMA descriptor byte slots inside descriptor_storage (binding_common.cuh:435-440).
TMA_SLOT_Q = 0
TMA_SLOT_K = 128
TMA_SLOT_V = 256
TMA_SLOT_G = 384
TMA_SLOT_BETA = 512
TMA_SLOT_OUT = 640

LAUNCH_TAGS = ("blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory")

# ---------------------------------------------------------------------------
# CUDA device helpers, transcribed in source order from
# csrc/kda/flashkda_bf16_fused_m128.cu:110-496.
#
# Helpers use the exact K.ptx/K.cuda wrappers available in the target TVM.
# ---------------------------------------------------------------------------

# tcgen05.mma spelling: kind::f16 from the (float32, bfloat16, bfloat16) dtypes; the A
# operand is a uint32 TMEM address (use_a_tmem), which selects the ts form.
_TCGEN05_MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
# cta_group::1 disable-output-lane mask group: the explicit default {0, 0, 0, 0}.
_MMA_ZERO_MASKS = (0, 0, 0, 0)
# Warp-level mma.sync zero-C operand group (the four "f"(0.f) inline-asm literals).
_MMA_ZERO_C = (K.float32(0.0), K.float32(0.0), K.float32(0.0), K.float32(0.0))
# TMA spellings: unicast g2s at CTA scope (the sm_100 wrapper emits the explicit default
# .cta_group::1 suffix for the unqualified inline instruction) and plain tile s2g.
_TMA_G2S_CTA = (
    "cp.async.bulk.tensor.{dim}d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1"
)
_TMA_S2G = "cp.async.bulk.tensor.{dim}d.global.shared::cta.tile.bulk_group"


def _mma_chain(taddr_d, b_lo, taddr_a, enable_d, *, b_offsets, dhi, idesc):
    # One @leader-predicated tcgen05.mma chain: `b_offsets` walks the B descriptor's
    # low half, taddr_a advances 8 per step, step 0 carries enable_d and the rest tie
    # it to 1.  Native emits the explicit default disable-output-lane {0, 0, 0, 0}.
    leader = K.cuda.elect_sync()
    b_lo32 = K.cast(b_lo, "uint32")
    for _i, _b_off in enumerate(b_offsets):
        _b_desc = (K.uint64(dhi) << K.uint64(32)) | K.cast(b_lo32 + K.uint32(_b_off), "uint64")
        K.ptx[_TCGEN05_MMA_F16](
            K.cast(taddr_d, "uint32"),
            K.cast(taddr_a + 8 * _i, "uint32"),
            _b_desc,
            K.uint32(idesc),
            *_MMA_ZERO_MASKS,
            K.ptx.pred(enable_d if _i == 0 else 1),
            pred=leader,
        )


# The three chains this kernel issues, by .cu block.
_MMA_QK_8STEP = dict(b_offsets=(0, 2, 4, 6, 256, 258, 260, 262), dhi=0x40004040, idesc=134743184)
_MMA_INV_2STEP = dict(b_offsets=(0, 64), dhi=0xC0004010, idesc=134743184)
_MMA_FINAL_2STEP = dict(b_offsets=(0, 128), dhi=0x40004040, idesc=136905872)


def _tensormap_acquire(tmap):
    return K.ptx.fence.proxy.tensormap__generic.acquire.gpu(tmap)


def _mma_m16n8k16_bf16_zero(acc, a, b):
    # mma.sync m16n8k16 bf16, zero C (acc/a/b: local arrays) -> native register-fragment
    # form; the explicit zero C slots feed "f"(0.f) == inline's 0f00000000 literals
    return K.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
        *[acc[i] for i in range(4)],
        *[a[i] for i in range(4)],
        *[b[i] for i in range(2)],
        *_MMA_ZERO_C,
    )


def _mma_m16n8k16_bf16_acc(acc, a, b):
    # mma.sync m16n8k16 bf16, accumulating C (C aliases acc == inline "+f" tied regs)
    return K.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
        *[acc[i] for i in range(4)],
        *[a[i] for i in range(4)],
        *[b[i] for i in range(2)],
        *[acc[i] for i in range(4)],
    )


def _mma_m16n8k8_bf16_zero(acc, a, b):
    # mma.sync m16n8k8 bf16, zero C -> native register-fragment form
    return K.ptx.mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32(
        *[acc[i] for i in range(4)],
        *[a[i] for i in range(2)],
        *[b[i] for i in range(1)],
        *_MMA_ZERO_C,
    )


def _mma_m16n8k16_bf16_zero_off4(acc, a, b):
    # second zero-C mma writing acc[4..7] with b_frag[2..3] (e.g. .cu:1725-1727)
    return K.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
        *[acc[4 + i] for i in range(4)],
        *[a[i] for i in range(4)],
        *[b[2 + i] for i in range(2)],
        *_MMA_ZERO_C,
    )


def _mma_m16n8k16_bf16_acc_off4(acc, a, b):
    # second accumulating mma writing acc[4..7] with b_frag[2..3]
    return K.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
        *[acc[4 + i] for i in range(4)],
        *[a[i] for i in range(4)],
        *[b[2 + i] for i in range(2)],
        *[acc[4 + i] for i in range(4)],
    )


def _shfl_bfly_f32(value, lane_xor):
    """``shfl.sync.bfly.b32`` at width 32: clamp/segmask 31, full member mask.

    DPS: the destination pins the warp collective to the call site, so the
    shuffle is emitted once here rather than re-emitted at every textual use
    of the returned value.
    """
    shfl_bfly = K.local_scalar("uint32")
    K.ptx.shfl_sync.bfly.b32(
        shfl_bfly,
        K.reinterpret("uint32", value),
        K.cast(lane_xor, "uint32"),
        K.uint32(31),
        K.uint32(0xFFFFFFFF),
    )
    return K.reinterpret("float32", shfl_bfly)


def _shfl_idx_f32(value, source_lane):
    """``shfl.sync.idx.b32`` at width 32: clamp/segmask 31, full member mask."""
    shfl_idx = K.local_scalar("uint32")
    K.ptx.shfl_sync.idx.b32(
        shfl_idx,
        K.reinterpret("uint32", value),
        K.cast(source_lane, "uint32"),
        K.uint32(31),
        K.uint32(0xFFFFFFFF),
    )
    return K.reinterpret("float32", shfl_idx)


def _mbarrier_wait(barrier, stage, phase):
    # .cu:158-171 -> native K.cuda.mbarrier_wait: same LAB_WAIT spin loop with
    # ticks=0x989680; mbarrier.try_wait.parity.shared::cta defaults to .acquire.cta
    # (the token/cluster wait variants at .cu:130-156,173-198 had no call sites)
    return K.cuda.mbarrier_wait(barrier.ptr_to([stage]), phase ^ barrier.phase_offset)


def _elect_commit(mbar_addr):
    # .cu:239-248
    leader = K.cuda.elect_sync()
    return K.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
        mbar_addr, pred=leader
    )


def _elect_commit2(barrier_a, barrier_b, stage):
    # .cu:239-248 two-barrier form: one elect_sync predicating both commits.
    leader = K.cuda.elect_sync()
    for _bar in (barrier_a, barrier_b):
        K.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
            _bar.ptr_to([stage]), pred=leader
        )


def _mbarrier_arrive(barrier, stage):
    # .cu:251-255 -> native; emits the same mbarrier.arrive.release.cta.shared::cta.b64 _, [addr]
    return K.ptx.mbarrier.arrive.release.cta.shared__cta.b64(barrier.ptr_to([stage]), K.uint32(1))


def _mbarrier_arrive_expect_tx(barrier, stage, bytes_):
    # .cu:258-262
    return K.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(
        barrier.ptr_to([stage]), K.uint32(bytes_)
    )


def _fence_async_shared():
    # .cu:416-418
    return K.ptx.fence.proxy.async_.shared__cta()


def _tma_3d_gmem2smem(smem_raw, smem, dst, tmap_ptr, x, y, z, mbar_ptr):
    # .cu:430-438 (raw tensor-map pointer form) -> native; on sm_100 the wrapper emits
    # the explicit default .cta_group::1 suffix for the unqualified inline instruction
    return K.ptx[_TMA_G2S_CTA.format(dim=3)](
        smem_raw.ptr_to([dst - smem]), K.reinterpret("uint64", tmap_ptr), x, y, z, mbar_ptr
    )


def _tma_2d_gmem2smem(smem_raw, smem, dst, tmap_ptr, x, y, mbar_ptr):
    # .cu:441-449 -> native (see _tma_3d_gmem2smem note)
    return K.ptx[_TMA_G2S_CTA.format(dim=2)](
        smem_raw.ptr_to([dst - smem]), K.reinterpret("uint64", tmap_ptr), x, y, mbar_ptr
    )


def _tma_4d_gmem2smem(smem_raw, smem, dst, tmap_ptr, x, y, z, w, mbar_ptr):
    # .cu:452-460 -> native (see _tma_3d_gmem2smem note)
    return K.ptx[_TMA_G2S_CTA.format(dim=4)](
        smem_raw.ptr_to([dst - smem]), K.reinterpret("uint64", tmap_ptr), x, y, z, w, mbar_ptr
    )


def _tma_store_4d(smem_raw, smem, tmap, x, y, z, w, smem_addr):
    # .cu:463-469 -> native s2g (cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group)
    return K.ptx[_TMA_S2G.format(dim=4)](
        K.reinterpret("uint64", tmap), x, y, z, w, smem_raw.ptr_to([smem_addr - smem])
    )


@dataclass
class FlashKDABf16FusedM128Config:
    label: str
    num_heads: int
    seq_lens: tuple[int, ...]
    packed: bool
    use_initial_state: bool = False
    store_final_state: bool = False
    lower_bound: float = DEFAULT_LOWER_BOUND
    seed: int = 0

    def validate(self) -> None:
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if not self.seq_lens or any(t <= 0 for t in self.seq_lens):
            raise ValueError("seq_lens must be non-empty and positive")
        if not math.isfinite(self.lower_bound) or self.lower_bound >= 0.0:
            raise ValueError("lower_bound must be finite and negative")
        if self.packed:
            if sum(self.seq_lens) <= len(self.seq_lens):
                raise ValueError("packed prefill requires total_tokens > num_seqs")
        else:
            if len(set(self.seq_lens)) != 1:
                raise ValueError("fixed layout requires uniform seq_lens")
            if self.seq_lens[0] <= 1:
                raise ValueError("fixed prefill requires T > 1")
            if len(self.seq_lens) == 1 and self.num_heads == 64:
                raise ValueError("fixed B=1 H=64 dispatches to the m64 kernel (out of scope)")

    @property
    def num_seqs(self) -> int:
        return len(self.seq_lens)

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)

    @property
    def beta_tma_tokens(self) -> int:
        return max(self.total_tokens, 32)

    @property
    def beta_tma_heads(self) -> int:
        return max(self.num_heads, 8)


def _cu_tensor_map_encode_tiled(
    data_ptr: int,
    rank: int,
    global_dim: list[int],
    global_strides: list[int],
    box_dim: list[int],
    swizzle: int,
) -> bytes:
    """ctypes replication of the host cuTensorMapEncodeTiled calls in
    flashkda_binding_common.cuh (BF16, interleave/L2/OOB all NONE)."""
    import ctypes

    libcuda = ctypes.CDLL("libcuda.so.1")
    tensor_map = (ctypes.c_uint64 * 16)()  # 128 B, opaque
    dims = (ctypes.c_uint64 * rank)(*global_dim)
    strides = (ctypes.c_uint64 * (rank - 1))(*global_strides)
    box = (ctypes.c_uint32 * rank)(*box_dim)
    elem_strides = (ctypes.c_uint32 * rank)(*([1] * rank))
    result = libcuda.cuTensorMapEncodeTiled(
        ctypes.byref(tensor_map),
        9,  # CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
        rank,
        ctypes.c_void_p(data_ptr),
        dims,
        strides,
        box,
        elem_strides,
        0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
        swizzle,
        0,  # CU_TENSOR_MAP_L2_PROMOTION_NONE
        0,  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    )
    if result != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed with CUresult={result}")
    return bytes(tensor_map)


_CU_TENSOR_MAP_SWIZZLE_128B = 3
_CU_TENSOR_MAP_SWIZZLE_NONE = 0


def _encode_tma_descriptors(case: dict[str, Any]) -> None:
    """Publish the six CUtensorMaps into descriptor_storage (host side of
    EncodeTmaPointers<128> in flashkda_binding_common.cuh)."""
    cfg: FlashKDABf16FusedM128Config = case["config"]
    tokens = cfg.total_tokens
    h = cfg.num_heads
    maps = bytearray()
    # EncodeQkTma(q/k): 4D {64, tokens, H, 2}, strides {H*256, 256, 128},
    # box {64, 32, 1, 2}, SWIZZLE_128B.
    for tensor in (case["q"], case["k"]):
        maps += _cu_tensor_map_encode_tiled(
            tensor.data_ptr(),
            4,
            [64, tokens, h, D_HEAD // 64],
            [h * D_HEAD * 2, D_HEAD * 2, 64 * 2],
            [64, 32, 1, 2],
            _CU_TENSOR_MAP_SWIZZLE_128B,
        )
    # EncodeValueTma<128>(v) / EncodeGateTma(g): 3D {128, H, tokens},
    # strides {256, 256*H}, box {128, 1, 32}, SWIZZLE_NONE.
    for tensor in (case["v"], case["g"]):
        maps += _cu_tensor_map_encode_tiled(
            tensor.data_ptr(),
            3,
            [D_HEAD, h, tokens],
            [D_HEAD * 2, D_HEAD * h * 2],
            [128, 1, 32],
            _CU_TENSOR_MAP_SWIZZLE_NONE,
        )
    # EncodeBetaTma(beta_tma): 2D {max(H,8), max(tokens,32)}, box {8, 32}.
    maps += _cu_tensor_map_encode_tiled(
        case["beta_tma"].data_ptr(),
        2,
        [cfg.beta_tma_heads, cfg.beta_tma_tokens],
        [cfg.beta_tma_heads * 2],
        [8, 32],
        _CU_TENSOR_MAP_SWIZZLE_NONE,
    )
    # EncodeOutputTma<128>(out): same layout as q/k, SWIZZLE_128B.
    maps += _cu_tensor_map_encode_tiled(
        case["out"].data_ptr(),
        4,
        [64, tokens, h, D_HEAD // 64],
        [h * D_HEAD * 2, D_HEAD * 2, 64 * 2],
        [64, 32, 1, 2],
        _CU_TENSOR_MAP_SWIZZLE_128B,
    )
    case["descriptor_storage"].copy_(
        torch.frombuffer(maps, dtype=torch.uint8).to(case["descriptor_storage"].device)
    )


def _m128_dispatch_reason(cfg: FlashKDABf16FusedM128Config) -> str:
    if not cfg.packed and cfg.num_seqs == 1 and cfg.num_heads == 64:
        return "out_of_scope: fixed B=1 H=64 dispatches to kernel_flashkda_bf16_fused_m64"
    layout = "packed" if cfg.packed else "fixed"
    return (
        f"m128: sm100a BF16 {layout} prefill dispatches to "
        "kernel_flashkda_bf16_fused_m128 (_select_flash_kda_prefill_variant)"
    )


_MIXED_SEQ_LENS = (1300, 547, 2048, 963, 271, 3063)

CONFIGS = [
    # Mirrors the recurrent-KDA prefill reference coverage.
    {"label": "fixed_h2_t4", "num_heads": 2, "seq_lens": (4, 4), "packed": False},
    {"label": "packed_h2_3_5", "num_heads": 2, "seq_lens": (3, 5), "packed": True},
    {
        "label": "fixed_h2_t4_state",
        "num_heads": 2,
        "seq_lens": (4, 4),
        "packed": False,
        "use_initial_state": True,
        "store_final_state": True,
    },
    {
        "label": "packed_h2_3_5_state",
        "num_heads": 2,
        "seq_lens": (3, 5),
        "packed": True,
        "use_initial_state": True,
        "store_final_state": True,
    },
    # Multi-chunk coverage: sequences longer than one 128-token chunk exercise the
    # 5-stage chunk pipeline; mixed lengths exercise packed boundary handling.
    {"label": "fixed_h4_t256", "num_heads": 4, "seq_lens": (256, 256), "packed": False},
    {"label": "packed_h4_chunks", "num_heads": 4, "seq_lens": (200, 264, 128), "packed": True},
]

BENCH_CONFIGS = [
    # Mirrors benchmarks/bench_recurrent_kda_prefill.py CASES, minus h64_fixed8192
    # which dispatches to the m64 kernel (out of scope for this module).  The PR's
    # raw FlashKDA peer writes a separate final state, so all benchmark configs do
    # the same for a like-for-like kernel scope.
    {
        "label": "h96_fixed8192",
        "num_heads": 96,
        "seq_lens": (8192,),
        "packed": False,
        "use_initial_state": True,
        "store_final_state": True,
        "seed": 10000,
    },
    {
        "label": "h96_mixed",
        "num_heads": 96,
        "seq_lens": _MIXED_SEQ_LENS,
        "packed": True,
        "use_initial_state": True,
        "store_final_state": True,
        "seed": 10001,
    },
    {
        "label": "h96_uniform",
        "num_heads": 96,
        "seq_lens": (1024,) * 8,
        "packed": True,
        "use_initial_state": True,
        "store_final_state": True,
        "seed": 10002,
    },
    {
        "label": "h64_mixed",
        "num_heads": 64,
        "seq_lens": _MIXED_SEQ_LENS,
        "packed": True,
        "use_initial_state": True,
        "store_final_state": True,
        "seed": 10004,
    },
    {
        "label": "h64_uniform",
        "num_heads": 64,
        "seq_lens": (1024,) * 8,
        "packed": True,
        "use_initial_state": True,
        "store_final_state": True,
        "seed": 10005,
    },
]

KERNEL_META = {
    "name": "flashkda_bf16_fused_m128",
    "category": "flashinfer",
    "compute_capability": 10,
}


def _cfg(**kwargs: Any) -> FlashKDABf16FusedM128Config:
    cfg_fields = {field.name for field in fields(FlashKDABf16FusedM128Config)}
    cfg_kwargs = {key: value for key, value in kwargs.items() if key in cfg_fields}
    if "seq_lens" in cfg_kwargs:
        cfg_kwargs["seq_lens"] = tuple(int(t) for t in cfg_kwargs["seq_lens"])
    if "label" not in cfg_kwargs:
        cfg_kwargs["label"] = "custom"
    cfg = FlashKDABf16FusedM128Config(**cfg_kwargs)
    cfg.validate()
    return cfg


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    cfg = _cfg(**kwargs)
    device = kwargs.get("device", "cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)

    total_tokens = cfg.total_tokens
    shape = (total_tokens, cfg.num_heads, D_HEAD)
    q = torch.randn(shape, device=device, dtype=torch.float32, generator=gen).to(torch.bfloat16)
    k = torch.randn(shape, device=device, dtype=torch.float32, generator=gen).to(torch.bfloat16)
    v = torch.randn(shape, device=device, dtype=torch.float32, generator=gen).to(torch.bfloat16)
    g = torch.randn(shape, device=device, dtype=torch.float32, generator=gen).to(torch.bfloat16)
    beta = torch.randn(
        (total_tokens, cfg.num_heads), device=device, dtype=torch.float32, generator=gen
    ).to(torch.bfloat16)
    A_log = 0.1 * torch.randn((cfg.num_heads,), device=device, dtype=torch.float32, generator=gen)
    dt_bias = 0.1 * torch.randn(
        (cfg.num_heads, D_HEAD), device=device, dtype=torch.float32, generator=gen
    )

    # Host-side beta TMA padding (flashinfer/kda_prefill.py::_beta_tma_source):
    # [max(tokens, 32), max(H, 8)], zero-filled padding.
    beta_tma = torch.zeros(
        (cfg.beta_tma_tokens, cfg.beta_tma_heads), device=device, dtype=torch.bfloat16
    )
    beta_tma[:total_tokens, : cfg.num_heads] = beta

    offsets = [0]
    for seq_len in cfg.seq_lens:
        offsets.append(offsets[-1] + seq_len)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device=device)
    # Packed bench mirrors the PR benchmark: order sequences by descending length
    # for better tail utilization; correctness configs keep the identity order.
    if cfg.packed and cfg.total_tokens > 512:
        order = sorted(range(cfg.num_seqs), key=lambda i: -cfg.seq_lens[i])
    else:
        order = list(range(cfg.num_seqs))
    seq_order = torch.tensor(order, dtype=torch.int32, device=device)

    state_shape = (cfg.num_seqs, cfg.num_heads, D_HEAD, D_HEAD)
    if cfg.use_initial_state:
        initial_state = (
            0.1 * torch.randn(state_shape, device=device, dtype=torch.float32, generator=gen)
        ).to(torch.bfloat16)
    else:
        initial_state = torch.empty(state_shape, device=device, dtype=torch.bfloat16)
    final_state = torch.empty(state_shape, device=device, dtype=torch.bfloat16)
    out = torch.empty(shape, device=device, dtype=torch.bfloat16)
    # Six CUtensorMap descriptors, 128 B each, 64 B aligned (torch storages are
    # 512 B aligned); the reference kernel publishes descriptors here.
    descriptor_storage = torch.empty((768,), device=device, dtype=torch.uint8)

    case = {
        "config": cfg,
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "beta_tma": beta_tma,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "cu_seqlens": cu_seqlens,
        "seq_order": seq_order,
        "initial_state": initial_state,
        "out": out,
        "final_state": final_state,
        "descriptor_storage": descriptor_storage,
        "scale": 1.0 / math.sqrt(D_HEAD),
        "dispatch_reason": _m128_dispatch_reason(cfg),
    }
    if device != "cpu" and torch.device(device).type != "cpu":
        _encode_tma_descriptors(case)
    return case


def _load_flashinfer_recurrent_kda():
    """Import the reference kernel from the installed flashinfer."""
    try:
        from flashinfer.kda import recurrent_kda
    except ImportError as e:
        raise RuntimeError(
            "flashinfer.kda is unavailable; the m128 reference needs a flashinfer "
            "release that carries it (flashinfer-ai/flashinfer#4262 or later)"
        ) from e

    return recurrent_kda


def _flashinfer_cuda_reference(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the m128 CUDA reference via flashinfer.recurrent_kda on the same inputs."""
    cfg: FlashKDABf16FusedM128Config = case["config"]
    recurrent_kda = _load_flashinfer_recurrent_kda()
    batch = 1 if cfg.packed else cfg.num_seqs
    seq_len = cfg.total_tokens if cfg.packed else cfg.seq_lens[0]

    def reshaped(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(batch, seq_len, cfg.num_heads, -1)

    ref_out, ref_state = recurrent_kda(
        q=reshaped(case["q"]),
        k=reshaped(case["k"]),
        v=reshaped(case["v"]),
        g=reshaped(case["g"]),
        beta=case["beta"].reshape(batch, seq_len, cfg.num_heads),
        A_log=case["A_log"],
        dt_bias=case["dt_bias"],
        scale=case["scale"],
        initial_state=case["initial_state"] if cfg.use_initial_state else None,
        output_final_state=cfg.store_final_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=cfg.lower_bound,
        cu_seqlens=case["cu_seqlens"] if cfg.packed else None,
        beta_is_logit=True,
        seq_order=case["seq_order"] if cfg.packed else None,
    )
    return ref_out.reshape(cfg.total_tokens, cfg.num_heads, D_HEAD), ref_state


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        case["q"].reshape(-1),
        case["k"].reshape(-1),
        case["v"].reshape(-1),
        case["g"].reshape(-1),
        case["beta"].reshape(-1),
        case["beta_tma"],
        case["A_log"],
        case["dt_bias"].reshape(-1),
        case["cu_seqlens"],
        case["seq_order"],
        case["initial_state"].reshape(-1),
        case["out"].reshape(-1),
        case["final_state"].reshape(-1),
        case["descriptor_storage"],
    )


def bf16_fused_m128(**kwargs: Any):
    cfg = _cfg(**kwargs)
    total_tokens = cfg.total_tokens
    h = cfg.num_heads
    num_seqs = cfg.num_seqs
    beta_tma_tokens = cfg.beta_tma_tokens
    beta_tma_heads = cfg.beta_tma_heads
    scale = 1.0 / math.sqrt(D_HEAD)
    lower_bound = cfg.lower_bound
    use_initial_state = cfg.use_initial_state
    store_final_state = cfg.store_final_state
    compute_regs, epilogue_regs, producer_regs_target, prep_regs = (
        (160, 40, 32, 56) if cfg.packed else (152, 40, 40, 56)
    )

    @K.kernel(warps=THREADS // 32, arch="sm_100a", min_blocks_per_sm=1, grid=num_seqs * h)
    def _kernel(
        q: K.gptr[K.bf16],
        k: K.gptr[K.bf16],
        v: K.gptr[K.bf16],
        g: K.gptr[K.bf16],
        beta: K.gptr[K.bf16],
        beta_tma: K.gptr[K.bf16, 2],
        A_log: K.gptr[K.f32],
        dt_bias: K.gptr[K.f32],
        cu_seqlens: K.gptr[K.i64],
        seq_order: K.gptr[K.i32],
        initial_state: K.gptr[K.bf16],
        out: K.gptr[K.bf16],
        final_state: K.gptr[K.bf16],
        descriptor_storage: K.gptr[K.u8],
    ):
        block_idx = K.cta_id()
        thread_idx = K.thread_id()
        warp = K.warp_id()
        lane = K.lane_id()

        roles = K.specialize()
        compute = roles.role("compute", warps=range(4), regs=compute_regs)
        epilogue = roles.role("epilogue", warps=range(4, 8), regs=epilogue_regs)
        producer_regs = roles.register_scope(
            "producer_regs", warps=range(8, 12), regs=producer_regs_target
        )
        roles.role("idle8", warps=[8], register_scope=producer_regs)
        mma = roles.role("mma", warps=[9], register_scope=producer_regs)
        load = roles.role("load", warps=[10], register_scope=producer_regs)
        roles.role("idle11", warps=[11], register_scope=producer_regs)
        prep_role_owner = roles.role("prep", warps=range(12, 32), regs=prep_regs)

        smem_pool = K.smem_pool()
        # Declaration order preserves the frozen 0..616 barrier byte map.
        qk_full = K.MBarrier(smem_pool.pool, 5, leader=True)
        gate_raw_full = K.TMABar(smem_pool.pool, 5, leader=True)
        qk_raw_full = K.TMABar(smem_pool.pool, 5, leader=True)
        v_full = K.TMABar(smem_pool.pool, 5, leader=True)
        v_free = K.MBarrier(smem_pool.pool, 5, phase_offset=1, leader=True)
        smem_free = K.TCGen05Bar(smem_pool.pool, 5, phase_offset=1, leader=True)
        raw_inputs_free = K.TCGen05Bar(smem_pool.pool, 5, phase_offset=1, leader=True)
        state_inp_ready = K.MBarrier(smem_pool.pool, 5, leader=True)
        old_out_ready = K.TCGen05Bar(smem_pool.pool, 5, leader=True)
        u_inp_ready = K.MBarrier(smem_pool.pool, 5, leader=True)
        u2_acc_ready = K.TCGen05Bar(smem_pool.pool, 5, leader=True)
        u2_inp_ready = K.MBarrier(smem_pool.pool, 5, leader=True)
        final_ready = K.TCGen05Bar(smem_pool.pool, 5, leader=True)
        out_empty = K.MBarrier(smem_pool.pool, 1, phase_offset=1, leader=True)
        tmem_dealloc_ready = K.MBarrier(smem_pool.pool, 1, leader=True)
        prep_diag_ready = K.MBarrier(smem_pool.pool, 5, leader=True)
        prep_inv16_ready = K.MBarrier(smem_pool.pool, 5, leader=True)
        tmem_addr_storage = smem_pool.alloc((1,), K.i32, align=4)
        if smem_pool.bytes != 620:
            raise AssertionError(f"unexpected mbarrier/TMEM header size: {smem_pool.bytes}")
        smem_pool.alloc((1024 - smem_pool.bytes,), K.u8)
        smem_stage_storage = smem_pool.alloc(SMEM_STAGE_ALLOCATION, K.bf16, align=1024)
        smem_out_storage = smem_pool.alloc((2, 32, 128), K.bf16)
        smem_pool.commit(SMEM_TOTAL)
        smem_raw = K.decl_buffer(
            (SMEM_TOTAL,), K.u8, data=qk_full.buf.data, scope="shared.dyn", align=1024
        )

        def shared_view(shape, dtype, byte_offset):
            return K.decl_buffer(
                shape,
                dtype,
                data=smem_raw.data,
                scope="shared.dyn",
                byte_offset=byte_offset,
                align=1024,
            )

        smem_g_raw_all = shared_view(
            (SMEM_SMEM_G_RAW_ALL_STAGE_BYTES // 2,), K.bf16, SMEM_SMEM_G_RAW_ALL_OFF
        )
        smem_v_all = shared_view((SMEM_SMEM_V_ALL_STAGE_BYTES // 2,), K.bf16, SMEM_SMEM_V_ALL_OFF)
        smem_gate_all = shared_view(
            (SMEM_SMEM_GATE_ALL_STAGE_BYTES // 4,), K.f32, SMEM_SMEM_GATE_ALL_OFF
        )
        smem_gt_all = shared_view((SMEM_SMEM_GT_ALL_STAGE_BYTES // 4,), K.f32, SMEM_SMEM_GT_ALL_OFF)
        smem_gt_prefix_all = shared_view(
            (SMEM_SMEM_GT_PREFIX_ALL_STAGE_BYTES // 4,), K.f32, SMEM_SMEM_GT_PREFIX_ALL_OFF
        )
        smem_restore_factor_all = shared_view(
            (SMEM_SMEM_RESTORE_FACTOR_ALL_STAGE_BYTES // 4,),
            K.f32,
            SMEM_SMEM_RESTORE_FACTOR_ALL_OFF,
        )
        smem_prep_beta_all = shared_view(
            (SMEM_SMEM_PREP_BETA_ALL_STAGE_BYTES // 4,), K.f32, SMEM_SMEM_PREP_BETA_ALL_OFF
        )
        smem_gate_rate_all = shared_view(
            (SMEM_SMEM_GATE_RATE_ALL_STAGE_BYTES // 4,), K.f32, SMEM_SMEM_GATE_RATE_ALL_OFF
        )

        q_tma = descriptor_storage.ptr_to([TMA_SLOT_Q])
        k_tma = descriptor_storage.ptr_to([TMA_SLOT_K])
        v_tma = descriptor_storage.ptr_to([TMA_SLOT_V])
        g_tma = descriptor_storage.ptr_to([TMA_SLOT_G])
        beta_tma_tmap = descriptor_storage.ptr_to([TMA_SLOT_BETA])
        out_tma = descriptor_storage.ptr_to([TMA_SLOT_OUT])

        with K.If(thread_idx == 0), K.Then():
            for _tmap in (q_tma, k_tma, v_tma, g_tma, beta_tma_tmap, out_tma):
                _tensormap_acquire(_tmap)
        K.cuda.cta_sync()

        smem = K.local_scalar(
            "uint32", init=K.cuda.cvta_generic_to_shared(K.address_of(smem_raw[0]))
        )
        bid = block_idx
        smem_stage_addr = K.local_scalar(
            "int32",
            init=K.cast(
                K.cuda.cvta_generic_to_shared(K.address_of(smem_stage_storage[0, 0])), "int32"
            ),
        )
        smem_qd_addr = smem_stage_addr
        smem_g_raw_addr = smem_stage_addr
        smem_kd_addr = smem_stage_addr + 8192
        smem_q_raw_prefetch_addr = smem_stage_addr + 16384
        smem_final_trans_addr = smem_q_raw_prefetch_addr
        smem_kr_trans_addr = smem_q_raw_prefetch_addr
        smem_ki_addr = smem_q_raw_prefetch_addr
        smem_gate_all_addr = K.local_scalar(
            "int32",
            init=K.cast(K.cuda.cvta_generic_to_shared(K.address_of(smem_gate_all[0])), "int32"),
        )
        smem_inv_addr = smem_gate_all_addr + 4096
        smem_v_all_addr = K.local_scalar(
            "int32",
            init=K.cast(K.cuda.cvta_generic_to_shared(K.address_of(smem_v_all[0])), "int32"),
        )
        smem_v_addr = smem_v_all_addr
        smem_inv_work_addr = smem_v_all_addr
        smem_restore_factor_all_addr = K.local_scalar(
            "int32",
            init=K.cast(
                K.cuda.cvta_generic_to_shared(K.address_of(smem_restore_factor_all[0])), "int32"
            ),
        )
        smem_beta_raw_addr = smem_restore_factor_all_addr
        smem_out_addr = K.local_scalar(
            "int32",
            init=K.cast(
                K.cuda.cvta_generic_to_shared(K.address_of(smem_out_storage[0, 0, 0])), "int32"
            ),
        )

        with K.If(thread_idx == 0), K.Then():
            qk_full.init(1)
            gate_raw_full.init(1)
            qk_raw_full.init(1)
            v_full.init(1)
            v_free.init(4)
            smem_free.init(1)
            raw_inputs_free.init(1)
            state_inp_ready.init(4)
            old_out_ready.init(1)
            u_inp_ready.init(4)
            u2_acc_ready.init(1)
            u2_inp_ready.init(4)
            final_ready.init(1)
            out_empty.init(1)
            tmem_dealloc_ready.init(2)
            prep_diag_ready.init(2)
            prep_inv16_ready.init(2)
            K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()

        with K.If(K.warp_id() == 0), K.Then():
            K.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                K.address_of(tmem_addr_storage[0]), K.uint32(256)
            )
        K.cuda.cta_sync()
        K.ptx.tcgen05.fence__after_thread_sync()
        taddr_storage = K.local_scalar(K.i32)
        K.ptx.ld.volatile.shared.s32(taddr_storage, K.address_of(tmem_addr_storage[0]))
        taddr = K.local_scalar("int32", init=taddr_storage)
        tmem_tmem_state_inp = taddr + TMEM_TMEM_STATE_INP_OFFSET
        tmem_tmem_u_acc = taddr + TMEM_TMEM_U_ACC_OFFSET
        tmem_tmem_u2_inp = taddr + TMEM_TMEM_U2_INP_OFFSET
        tmem_tmem_u2_acc = taddr + TMEM_TMEM_U2_ACC_OFFSET
        tmem_tmem_out = taddr + TMEM_TMEM_OUT_OFFSET
        tmem_tmem_state_out = taddr + TMEM_TMEM_STATE_OUT_OFFSET

        def _work_item():
            # .cu:733-739 -- the identical five-value preamble every role opens with.
            task_idx: K.int32 = bid
            seq_idx = K.local_scalar("int32")
            K.ptx.ld.global_.s32(seq_idx, seq_order.ptr_to([task_idx // h]))
            head_idx: K.int32 = task_idx % h
            bos = K.local_scalar("int64")
            K.ptx.ld.global_.s64(bos, cu_seqlens.ptr_to([seq_idx]))
            eos = K.local_scalar("int64")
            K.ptx.ld.global_.s64(eos, cu_seqlens.ptr_to([seq_idx + 1]))
            seq_len: K.int32 = K.cast(eos - bos, "int32")
            return seq_idx, head_idx, bos, eos, seq_len, (seq_len + 32 - 1) // 32

        def compute_role(compute_state):
            # ---- Role: compute (.cu:730-963) ----
            # .cu:732 compute_main
            seq_idx, head_idx, bos, eos, seq_len, num_chunks = _work_item()  # .cu:733-739
            warp_in_wg: K.int32 = warp % 4  # .cu:740
            tmem_row_base: K.int32 = (warp_in_wg * 32) << 16  # .cu:741
            state_row: K.int32 = warp_in_wg * 32 + lane  # .cu:742
            warp_id_in_role: K.int32 = K.warp_id_in_role()  # .cu:743
            compute_local_warp: K.int32 = warp_id_in_role  # .cu:744
            state_base: K.int64 = (
                (K.cast(seq_idx, "int64") * K.cast(h, "int64") + K.cast(head_idx, "int64")) * 128
                + K.cast(state_row, "int64")
            ) * 128  # .cu:745
            initial_state_u32 = K.decl_buffer(
                (num_seqs * h * D_HEAD * D_HEAD // 2,), "uint32", data=initial_state.data
            )
            final_state_u32 = K.decl_buffer(
                (num_seqs * h * D_HEAD * D_HEAD // 2,), "uint32", data=final_state.data
            )
            with K.unroll(4) as state_col_block:  # .cu:746-822 (#pragma unroll)
                state_frag = K.alloc_local((32,), "float32")
                with K.unroll(32) as _zi:  # .cu:749-780
                    K.ptx.mov.b32(state_frag[_zi], K.float32(0.0))
                if use_initial_state:  # .cu:781
                    # .cu:783-819: two halves, each 2x uint4 load + bf16->f32 shl/and unpack
                    for _half in (0, 16):
                        with K.unroll(2) as _blk:
                            _vld = K.alloc_local((4,), "uint32", align=16)
                            _base = state_base + state_col_block * 32
                            K.ptx.ld.global_.v4.b32(
                                _vld[0],
                                _vld[1],
                                _vld[2],
                                _vld[3],
                                initial_state_u32.ptr_to(
                                    [
                                        (_base + _half) // 2 + _blk * 4
                                        if _half
                                        else _base // 2 + _blk * 4
                                    ]
                                ),
                            )
                            with K.unroll(4) as _pair:
                                K.ptx.mov.b32(
                                    state_frag[_half + _blk * 8 + _pair * 2],
                                    K.cuda.uint_as_float(_vld[_pair] << K.uint32(16)),
                                )
                                K.ptx.mov.b32(
                                    state_frag[_half + _blk * 8 + _pair * 2 + 1],
                                    K.cuda.uint_as_float(_vld[_pair] & K.uint32(0xFFFF0000)),
                                )
                K.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
                    K.cast(taddr + 64 + tmem_row_base + state_col_block * 32, "uint32"),
                    *[state_frag[_j] for _j in range(32)],
                )  # .cu:821
            K.ptx.tcgen05.wait__st.sync.aligned()  # .cu:823 tcgen05.wait::st.sync.aligned
            with K.serial(
                0, num_chunks, unroll=False
            ) as chunk_idx:  # .cu:830-831 (#pragma unroll 1)
                _mbarrier_wait(qk_full, compute_state.stage, compute_state.phase)  # .cu:832
                compute_stage_byte_base = (
                    K.cast(compute_state.stage, "int32") * SMEM_STAGE_BYTE_STRIDE
                )
                compute_stage_f32_base = compute_stage_byte_base // 4
                compute_stage_bf16_base = compute_stage_byte_base // 2
                with K.serial(
                    0, 4, unroll=False
                ) as state_col_block_1:  # .cu:833-834 (#pragma unroll 1)
                    state_addr: K.int32 = (
                        taddr + 64 + tmem_row_base + state_col_block_1 * 32
                    )  # .cu:835
                    _tmem_load_0 = K.alloc_local((32,), "float32")
                    K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                        *[_tmem_load_0[_j] for _j in range(32)], K.cast(state_addr, "uint32")
                    )  # .cu:836-837
                    _tmem_load_0_bf16 = K.alloc_local((16,), "uint32")
                    with K.unroll(16) as _lp:  # .cu:839-843
                        K.ptx.mov.b32(
                            _tmem_load_0_bf16[_lp],
                            K.cuda.float22bfloat162_rn(
                                _tmem_load_0[_lp * 2 + 0], _tmem_load_0[_lp * 2 + 1 + 0]
                            ),
                        )
                    # .cu:844-848 (inline tcgen05.st x16 of packed bf16 pairs)
                    K.ptx["tcgen05.st.sync.aligned.32x32b.x16.b32"](
                        K.cast(taddr + tmem_row_base + state_col_block_1 * 16, "uint32"),
                        *[_tmem_load_0_bf16[_j] for _j in range(16)],
                    )
                    state_scale = K.alloc_local((16,), "float32")
                    with K.unroll(2) as state_half:  # .cu:850-859
                        with K.unroll(4) as state_vec:
                            K.ptx.ld.shared.v4.f32(
                                state_scale[state_vec * 4],
                                state_scale[state_vec * 4 + 1],
                                state_scale[state_vec * 4 + 2],
                                state_scale[state_vec * 4 + 3],
                                smem_gt_all.ptr_to(
                                    [
                                        compute_stage_f32_base
                                        + state_col_block_1 * 32
                                        + state_half * 16
                                        + state_vec * 4
                                    ]
                                ),
                            )  # .cu:854
                        with K.unroll(8) as _ls:  # .cu:857-858
                            _pk = K.local_scalar("uint64")
                            K.ptx.mul.rn.ftz.f32x2(
                                _pk,
                                K.cuda.make_float2(
                                    _tmem_load_0[state_half * 16 + _ls * 2],
                                    _tmem_load_0[state_half * 16 + _ls * 2 + 1],
                                ),
                                K.cuda.make_float2(state_scale[_ls * 2], state_scale[_ls * 2 + 1]),
                            )
                            K.ptx.mov.b32(
                                _tmem_load_0[state_half * 16 + _ls * 2], K.cuda.float2_x(_pk)
                            )
                            K.ptx.mov.b32(
                                _tmem_load_0[state_half * 16 + _ls * 2 + 1], K.cuda.float2_y(_pk)
                            )
                    K.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
                        K.cast(state_addr, "uint32"), *[_tmem_load_0[_j] for _j in range(32)]
                    )  # .cu:860
                K.ptx.tcgen05.wait__st.sync.aligned()  # .cu:862
                with K.If(K.cuda.elect_sync()), K.Then():  # .cu:863-865
                    _mbarrier_arrive(state_inp_ready, compute_state.stage)
                _mbarrier_wait(v_full, compute_state.stage, compute_state.phase)  # .cu:866
                _mbarrier_wait(old_out_ready, compute_state.stage, compute_state.phase)  # .cu:867
                _tmem_load_1 = K.alloc_local((32,), "float32")
                K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                    *[_tmem_load_1[_j] for _j in range(32)],
                    K.cast(taddr + 224 + tmem_row_base, "uint32"),
                )  # .cu:868-869
                with K.unroll(2) as residual_half:  # .cu:870-895
                    residual_v = K.alloc_local((16,), "float32")
                    residual_beta = K.alloc_local((16,), "float32")
                    with K.unroll(16) as residual_col:  # .cu:874-881
                        token_col: K.int32 = residual_half * 16 + residual_col  # .cu:876
                        _bits = K.local_scalar("uint16")
                        K.ptx.ld.shared.b16(
                            _bits,
                            smem_v_all.ptr_to(
                                [compute_stage_bf16_base + token_col * 128 + state_row]
                            ),
                        )
                        K.ptx.mov.b32(
                            residual_v[residual_col],
                            K.cuda.bfloat162float(K.reinterpret("bfloat16", _bits)),
                        )  # .cu:877-879
                        K.ptx.ld.shared.f32(
                            residual_beta[residual_col],
                            smem_prep_beta_all.ptr_to([compute_stage_f32_base + token_col]),
                        )  # .cu:880
                    with K.unroll(8) as _ls:  # .cu:882-884
                        _pk = K.local_scalar("uint64")
                        K.ptx.sub.rn.ftz.f32x2(
                            _pk,
                            K.cuda.make_float2(residual_v[_ls * 2], residual_v[_ls * 2 + 1]),
                            K.cuda.make_float2(
                                _tmem_load_1[residual_half * 16 + _ls * 2],
                                _tmem_load_1[residual_half * 16 + _ls * 2 + 1],
                            ),
                        )
                        K.ptx.mov.b32(residual_v[_ls * 2], K.cuda.float2_x(_pk))
                        K.ptx.mov.b32(residual_v[_ls * 2 + 1], K.cuda.float2_y(_pk))
                    with K.unroll(8) as _ls:  # .cu:885-887
                        _pk = K.local_scalar("uint64")
                        K.ptx.mul.rn.ftz.f32x2(
                            _pk,
                            K.cuda.make_float2(residual_v[_ls * 2], residual_v[_ls * 2 + 1]),
                            K.cuda.make_float2(residual_beta[_ls * 2], residual_beta[_ls * 2 + 1]),
                        )
                        K.ptx.mov.b32(residual_v[_ls * 2], K.cuda.float2_x(_pk))
                        K.ptx.mov.b32(residual_v[_ls * 2 + 1], K.cuda.float2_y(_pk))
                    residual_v_bf16 = K.alloc_local((8,), "uint32")
                    with K.unroll(8) as _lp:  # .cu:888-893
                        K.ptx.mov.b32(
                            residual_v_bf16[_lp],
                            K.cuda.float22bfloat162_rn(
                                residual_v[_lp * 2 + 0], residual_v[_lp * 2 + 1 + 0]
                            ),
                        )
                    K.ptx["tcgen05.st.sync.aligned.32x32b.x8.b32"](
                        K.cast(taddr + 224 + tmem_row_base + residual_half * 8, "uint32"),
                        *[residual_v_bf16[_j] for _j in range(8)],
                    )  # .cu:894
                K.ptx.tcgen05.wait__st.sync.aligned()  # .cu:896
                with K.If(K.cuda.elect_sync()), K.Then():  # .cu:897-900
                    _mbarrier_arrive(v_free, compute_state.stage)
                    _mbarrier_arrive(u_inp_ready, compute_state.stage)
                _mbarrier_wait(u2_acc_ready, compute_state.stage, compute_state.phase)  # .cu:901
                _tmem_load_2 = K.alloc_local((32,), "float32")
                K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                    *[_tmem_load_2[_j] for _j in range(32)], K.cast(taddr + tmem_row_base, "uint32")
                )  # .cu:902-903
                _tmem_load_2_bf16 = K.alloc_local((16,), "uint32")
                with K.unroll(16) as _lp:  # .cu:904-909
                    K.ptx.mov.b32(
                        _tmem_load_2_bf16[_lp],
                        K.cuda.float22bfloat162_rn(
                            _tmem_load_2[_lp * 2 + 0], _tmem_load_2[_lp * 2 + 1 + 0]
                        ),
                    )
                # .cu:910-914 (inline tcgen05.st x16)
                K.ptx["tcgen05.st.sync.aligned.32x32b.x16.b32"](
                    K.cast(taddr + 224 + tmem_row_base, "uint32"),
                    *[_tmem_load_2_bf16[_j] for _j in range(16)],
                )
                K.ptx.tcgen05.wait__st.sync.aligned()  # .cu:915
                with K.If(K.cuda.elect_sync()), K.Then():  # .cu:916-918
                    _mbarrier_arrive(u2_inp_ready, compute_state.stage)
                _mbarrier_wait(final_ready, compute_state.stage, compute_state.phase)  # .cu:919
                compute_state.advance()
            with K.If(store_final_state), K.Then():  # .cu:923-955
                with K.unroll(4) as state_col_block_2:
                    _tmem_load_3 = K.alloc_local((32,), "float32")
                    K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                        *[_tmem_load_3[_j] for _j in range(32)],
                        K.cast(taddr + 64 + tmem_row_base + state_col_block_2 * 32, "uint32"),
                    )  # .cu:926-927
                    with K.unroll(2) as _half2:  # .cu:928-953 (two 16-float groups)
                        _pk = K.alloc_local((8,), "uint32")
                        with K.unroll(8) as _pj:
                            K.ptx.mov.b32(
                                _pk[_pj],
                                K.cuda.float22bfloat162_rn(
                                    _tmem_load_3[_half2 * 16 + _pj * 2],
                                    _tmem_load_3[_half2 * 16 + _pj * 2 + 1],
                                ),
                            )
                        K.ptx.st.global_.v4.b32(
                            final_state_u32.ptr_to(
                                [(state_base + state_col_block_2 * 32 + _half2 * 16) // 2]
                            ),
                            _pk[0],
                            _pk[1],
                            _pk[2],
                            _pk[3],
                        )  # .cu:938/951 first uint4
                        K.ptx.st.global_.v4.b32(
                            final_state_u32.ptr_to(
                                [(state_base + state_col_block_2 * 32 + _half2 * 16) // 2 + 4]
                            ),
                            _pk[4],
                            _pk[5],
                            _pk[6],
                            _pk[7],
                        )  # .cu:939/952 second uint4
            K.ptx.bar.sync(K.uint32(10), K.uint32(128))  # .cu:956 barrier.sync 10, 128
            with K.If(compute_local_warp == 0), K.Then():  # .cu:957-961
                with K.If(K.cuda.elect_sync()), K.Then():
                    _mbarrier_arrive(tmem_dealloc_ready, 0)

        def epilogue_role(epilogue_state, output_state):
            # ---- Role: epilogue (.cu:964-1090) ----
            # .cu:966 epilogue_main
            seq_idx_1, head_idx_1, bos_1, eos_1, seq_len_1, num_chunks_1 = (
                _work_item()
            )  # .cu:967-973
            warp_id_in_role_1: K.int32 = K.warp_id_in_role()  # .cu:974
            epilogue_local_warp: K.int32 = warp_id_in_role_1  # .cu:975
            warp_in_wg_1: K.int32 = warp % 4  # .cu:976
            tmem_row_base_1: K.int32 = (warp_in_wg_1 * 32) << 16  # .cu:977
            state_row_1: K.int32 = warp_in_wg_1 * 32 + lane  # .cu:978
            with K.serial(0, num_chunks_1, unroll=False) as chunk_idx_1:  # .cu:982-983
                _mbarrier_wait(final_ready, epilogue_state.stage, epilogue_state.phase)  # .cu:984
                chunk_is_full: K.int32 = K.if_then_else(
                    seq_len_1 >= (chunk_idx_1 + 1) * 32, 1, 0
                )  # .cu:985
                with K.If(chunk_is_full != 0):
                    with K.Then():  # .cu:986
                        # .cu:987-993 (inline tcgen05.ld.16x256b.x4, 16 uint32 regs)
                        _tmem_load_4 = K.alloc_local((16,), "uint32")
                        K.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                            *[_tmem_load_4[_j] for _j in range(16)],
                            K.cast(taddr + 192 + tmem_row_base_1, "uint32"),
                        )
                        # .cu:994-1000 (same at TMEM row +16)
                        _tmem_load_5 = K.alloc_local((16,), "uint32")
                        K.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                            *[_tmem_load_5[_j] for _j in range(16)],
                            K.cast(taddr + 192 + tmem_row_base_1 + 1048576, "uint32"),
                        )
                        K.ptx.tcgen05.wait__ld.sync.aligned()  # .cu:1001
                        K.ptx.bar.sync(K.uint32(9), K.uint32(128))  # .cu:1002 barrier.sync 9, 128
                        with K.If(epilogue_local_warp == 0), K.Then():  # .cu:1003-1007
                            with K.If(K.cuda.elect_sync()), K.Then():
                                _mbarrier_arrive(out_empty, 0)
                        with K.If(epilogue_local_warp == 0), K.Then():  # .cu:1008-1012
                            with K.If(chunk_idx_1 >= 2), K.Then():
                                K.ptx.cp.async_.bulk.wait_group.read(1)
                        K.ptx.bar.sync(K.uint32(9), K.uint32(128))  # .cu:1013
                        out_stage_addr: K.int32 = (
                            smem_out_addr + K.cast(output_state.phase, "int32") * 8192
                        )  # .cu:1014
                        with K.unroll(2) as dim_half:  # .cu:1015-1049
                            out_packed = K.alloc_local((8,), "uint32", align=4)  # .cu:1017
                            with K.If(dim_half == 0):
                                with K.Then():  # .cu:1018-1023
                                    with K.unroll(8) as _lp:
                                        K.ptx.mov.b32(
                                            out_packed[_lp],
                                            K.cuda.float22bfloat162_rn(
                                                K.cuda.uint_as_float(_tmem_load_4[_lp * 2 + 0]),
                                                K.cuda.uint_as_float(_tmem_load_4[_lp * 2 + 1 + 0]),
                                            ),
                                        )
                                with K.Else():  # .cu:1024-1030
                                    with K.unroll(8) as _lp:
                                        K.ptx.mov.b32(
                                            out_packed[_lp],
                                            K.cuda.float22bfloat162_rn(
                                                K.cuda.uint_as_float(_tmem_load_5[_lp * 2 + 0]),
                                                K.cuda.uint_as_float(_tmem_load_5[_lp * 2 + 1 + 0]),
                                            ),
                                        )
                            with K.unroll(2) as token_group:  # .cu:1031-1048
                                mtx_idx: K.int32 = lane // 8  # .cu:1033
                                row_addr: K.int32 = lane & 7  # .cu:1034
                                dim_base: K.int32 = (
                                    epilogue_local_warp * 32 + dim_half * 16 + (mtx_idx & 1) * 8
                                )  # .cu:1035
                                token_base: K.int32 = (
                                    token_group * 16 + mtx_idx // 2 * 8
                                )  # .cu:1036
                                token_addr: K.int32 = token_base + row_addr  # .cu:1037
                                token_pair: K.int32 = token_addr // 2  # .cu:1038
                                token_parity: K.int32 = token_addr & 1  # .cu:1039
                                raw_row: K.int32 = token_pair + dim_base // 64 * 16  # .cu:1040
                                raw_col: K.int32 = (
                                    dim_base & 63 ^ (token_pair & 3) << 4 ^ token_parity << 3
                                ) + token_parity * 64  # .cu:1041
                                stsm_offset: K.int32 = (raw_row * 128 + raw_col) * 2  # .cu:1042
                                pack_base: K.int32 = token_group * 4  # .cu:1043
                                # .cu:1044-1047 stmatrix.x4.trans
                                K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                                    smem_raw.ptr_to([K.cast(out_stage_addr, "uint32") - smem + K.cast(stsm_offset, "uint32")]),
                                    out_packed[pack_base],
                                    out_packed[pack_base + 1],
                                    out_packed[pack_base + 2],
                                    out_packed[pack_base + 3],
                                )  # fmt: skip
                        _fence_async_shared()  # .cu:1050
                        K.ptx.bar.sync(K.uint32(9), K.uint32(128))  # .cu:1051
                        with K.If(epilogue_local_warp == 0), K.Then():  # .cu:1052-1057
                            with K.If(K.cuda.elect_sync()), K.Then():
                                _tma_store_4d(
                                    smem_raw,
                                    K.cast(smem, "int32"),
                                    out_tma,
                                    0,
                                    K.cast(bos_1 + K.cast(chunk_idx_1 * 32, "int64"), "int32"),
                                    head_idx_1,
                                    0,
                                    K.cast(
                                        smem_out_addr + K.cast(output_state.phase, "int32") * 8192,
                                        "uint32",
                                    ),
                                )  # .cu:1054
                            K.ptx.cp.async_.bulk.commit_group()  # .cu:1056
                        output_state.advance()  # .cu:1058
                    with K.Else():  # .cu:1059-1077 (partial chunk: scalar out stores)
                        _tmem_load_6 = K.alloc_local((32,), "float32")
                        K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                            *[_tmem_load_6[_j] for _j in range(32)],
                            K.cast(taddr + 192 + tmem_row_base_1, "uint32"),
                        )  # .cu:1060-1061
                        K.ptx.tcgen05.wait__ld.sync.aligned()  # .cu:1062
                        K.ptx.bar.sync(K.uint32(9), K.uint32(128))  # .cu:1063
                        with K.If(epilogue_local_warp == 0), K.Then():  # .cu:1064-1068
                            with K.If(K.cuda.elect_sync()), K.Then():
                                _mbarrier_arrive(out_empty, 0)
                        with K.unroll(32) as token_col_1:  # .cu:1069-1076
                            out_token: K.int64 = bos_1 + K.cast(
                                chunk_idx_1 * 32 + token_col_1, "int64"
                            )  # .cu:1071
                            with K.If(out_token < eos_1), K.Then():  # .cu:1072
                                out_idx: K.int64 = (
                                    out_token * K.cast(h, "int64") + K.cast(head_idx_1, "int64")
                                ) * 128 + K.cast(state_row_1, "int64")  # .cu:1073
                                K.ptx.st.global_.b16(
                                    out.ptr_to([out_idx]),
                                    K.reinterpret(
                                        "uint16", K.cast(_tmem_load_6[token_col_1], "bfloat16")
                                    ),
                                )  # .cu:1074
                epilogue_state.advance()
            with K.If(epilogue_local_warp == 0), K.Then():  # .cu:1081-1083
                K.ptx.cp.async_.bulk.wait_group(0)
            K.ptx.bar.sync(K.uint32(9), K.uint32(128))  # .cu:1084
            with K.If(epilogue_local_warp == 0), K.Then():  # .cu:1085-1089
                with K.If(K.cuda.elect_sync()), K.Then():
                    _mbarrier_arrive(tmem_dealloc_ready, 0)

        def mma_role(mma_state, out_state):
            # ---- Role: mma (.cu:1092-1247) ----
            # .cu:1093 mma_main
            seq_idx_2, _, bos_2, eos_2, seq_len_2, num_chunks_2 = _work_item()  # .cu:1094-1099
            with K.serial(0, num_chunks_2, unroll=False) as _chunk_idx:  # .cu:1106-1107
                _mbarrier_wait(qk_full, mma_state.stage, mma_state.phase)  # .cu:1108
                _mbarrier_wait(state_inp_ready, mma_state.stage, mma_state.phase)  # .cu:1109
                _mbarrier_wait(out_empty, 0, out_state.phase)  # .cu:1110
                out_state.advance()  # .cu:1111
                mma_stage_byte_base = K.cast(mma_state.stage, "int32") * SMEM_STAGE_BYTE_STRIDE
                _mma_b_addr_0: K.int32 = smem_qd_addr + mma_stage_byte_base  # .cu:1112
                _mma_b_lo_0 = K.uniform(K.cast((_mma_b_addr_0 >> 4) & 0x3FFF, "uint32"))  # .cu:1113
                _mma_chain(
                    tmem_tmem_out,
                    K.cast(_mma_b_lo_0, "int32"),
                    tmem_tmem_state_inp,
                    0,
                    **_MMA_QK_8STEP,
                )  # .cu:1114-1150
                _mma_b_addr_1: K.int32 = smem_kd_addr + mma_stage_byte_base  # .cu:1151
                _mma_b_lo_1 = K.uniform(K.cast((_mma_b_addr_1 >> 4) & 0x3FFF, "uint32"))  # .cu:1152
                _mma_chain(
                    tmem_tmem_u_acc,
                    K.cast(_mma_b_lo_1, "int32"),
                    tmem_tmem_state_inp,
                    0,
                    **_MMA_QK_8STEP,
                )  # .cu:1153-1189
                _elect_commit2(old_out_ready, raw_inputs_free, mma_state.stage)  # .cu:1190
                _mbarrier_wait(u_inp_ready, mma_state.stage, mma_state.phase)  # .cu:1191
                _mma_b_addr_2: K.int32 = smem_inv_addr + mma_stage_byte_base  # .cu:1192
                _mma_b_lo_2 = K.uniform(K.cast((_mma_b_addr_2 >> 4) & 0x3FFF, "uint32"))  # .cu:1193
                _mma_chain(
                    tmem_tmem_u2_acc,
                    K.cast(_mma_b_lo_2, "int32"),
                    tmem_tmem_u2_inp,
                    0,
                    **_MMA_INV_2STEP,
                )  # .cu:1194-1212
                _elect_commit(u2_acc_ready.ptr_to([mma_state.stage]))  # .cu:1213
                _mbarrier_wait(u2_inp_ready, mma_state.stage, mma_state.phase)  # .cu:1214
                _mma_b_addr_3: K.int32 = smem_final_trans_addr + mma_stage_byte_base  # .cu:1215
                _mma_b_lo_3 = K.uniform(
                    K.cast(((_mma_b_addr_3 >> 4) & 0x3FFF) | 0x1000000, "uint32")
                )  # .cu:1216
                _mma_chain(
                    tmem_tmem_state_out,
                    K.cast(_mma_b_lo_3, "int32"),
                    tmem_tmem_u2_inp,
                    1,
                    **_MMA_FINAL_2STEP,
                )  # .cu:1217-1235
                _elect_commit2(final_ready, smem_free, mma_state.stage)  # .cu:1236
                mma_state.advance()
            _mbarrier_wait(tmem_dealloc_ready, 0, K.uint32(0))  # .cu:1241
            _tmem_dealloc_addr = K.local_scalar("int32")  # .cu:1243
            K.ptx.ld.volatile.shared.s32(_tmem_dealloc_addr, K.address_of(tmem_addr_storage[0]))
            K.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                K.cast(_tmem_dealloc_addr, "uint32"), K.uint32(256)
            )  # .cu:1244
            # Preserve the source's dealloc-before-relinquish order.
            K.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()  # .cu:1245

        def load_role(load_state):
            # ---- Role: load (.cu:1248-1296) ----
            # .cu:1249 load_main
            seq_idx_3, head_idx_2, bos_3, eos_3, seq_len_3, num_chunks_3 = (
                _work_item()
            )  # .cu:1250-1256
            with K.serial(0, num_chunks_3, unroll=False) as chunk_idx_2:  # .cu:1260-1261
                _mbarrier_wait(v_free, load_state.stage, load_state.phase)  # .cu:1262
                _mbarrier_wait(qk_full, load_state.stage, load_state.phase)  # .cu:1263
                load_stage_byte_base = K.cast(load_state.stage, "int32") * SMEM_STAGE_BYTE_STRIDE
                load_stage_bf16_base = load_stage_byte_base // 2
                chunk_is_full_1: K.int32 = K.if_then_else(
                    seq_len_3 >= (chunk_idx_2 + 1) * 32, 1, 0
                )  # .cu:1264
                with K.If(K.cuda.elect_sync()), K.Then():  # .cu:1265-1270
                    with K.If(chunk_is_full_1 != 0), K.Then():
                        _mbarrier_arrive_expect_tx(v_full, load_state.stage, 8192)  # .cu:1267
                        _tma_3d_gmem2smem(
                            smem_raw,
                            K.cast(smem, "int32"),
                            smem_v_addr + load_stage_byte_base,
                            v_tma,
                            0,
                            head_idx_2,
                            K.cast(bos_3 + K.cast(chunk_idx_2 * 32, "int64"), "int32"),
                            v_full.ptr_to([load_state.stage]),
                        )  # .cu:1268
                with K.If(chunk_is_full_1 == 0), K.Then():  # .cu:1271-1285
                    with K.unroll(16) as v_load_iter:  # .cu:1272-1282
                        v_item: K.int32 = v_load_iter * 32 + lane  # .cu:1274
                        row: K.int32 = v_item // 16  # .cu:1275
                        segment: K.int32 = v_item % 16  # .cu:1276
                        token: K.int64 = bos_3 + K.cast(chunk_idx_2 * 32 + row, "int64")  # .cu:1277
                        token_valid: K.int32 = K.if_then_else(token < eos_3, 1, 0)  # .cu:1278
                        v_src: K.int64 = (
                            token * K.cast(h, "int64") + K.cast(head_idx_2, "int64")
                        ) * 128 + K.cast(segment * 8, "int64")  # .cu:1279
                        K.ptx["cp.async.cg.shared.global"](
                            smem_v_all.ptr_to(
                                [load_stage_bf16_base + row * 128 + segment * 8]
                            ),
                            v.ptr_to([v_src]),
                            16,
                            K.cast(K.if_then_else(token_valid != 0, 16, 0), "uint32"),
                        )  # .cu:1280-1281  # fmt: skip
                    K.ptx.cp.async_.commit_group()  # .cu:1283
                    K.ptx.cp.async_.wait_group(0)  # .cu:1284
                K.ptx.bar.sync(K.uint32(8), K.uint32(32))  # .cu:1286 barrier.sync 8, 32
                with K.If(K.cuda.elect_sync()), K.Then():  # .cu:1287-1292
                    with K.If(chunk_is_full_1 == 0), K.Then():
                        _fence_async_shared()  # .cu:1289
                        _mbarrier_arrive(v_full, load_state.stage)  # .cu:1290
                load_state.advance()

        def prep_role(prep_phase):
            # ---- Role: prep (.cu:1298-2550) ----
            # .cu:1300 prep_main
            seq_idx_4, head_idx_3, bos_4, eos_4, seq_len_4, num_chunks_4 = (
                _work_item()
            )  # .cu:1301-1307
            warp_id_in_role_2: K.int32 = K.warp_id_in_role()  # .cu:1310
            prep_instance: K.int32 = warp_id_in_role_2 // 4  # .cu:1308-1309
            prep_local_warp: K.int32 = warp_id_in_role_2 % 4  # .cu:1311
            prep_tid: K.int32 = prep_local_warp * 32 + lane  # .cu:1312
            num_prep_iters: K.int32 = (num_chunks_4 + 4 - prep_instance) // 5  # .cu:1313
            prep_stage: K.uint32 = K.cast(prep_instance, "uint32")  # .cu:1314
            gate_rate_stage_f32: K.int32 = prep_instance * SMEM_STAGE_F32_STRIDE  # .cu:1315

            def _prep_bar_sync():
                # switch(prep_instance) over the five per-instance named barriers 11..15.
                def _arm(i):
                    if i == 4:
                        K.ptx.bar.sync(K.uint32(15), K.uint32(128))
                        return
                    with K.If(prep_instance == i):
                        with K.Then():
                            K.ptx.bar.sync(K.uint32(11 + i), K.uint32(128))
                        with K.Else():
                            _arm(i + 1)

                _arm(0)

            with K.If(prep_tid == 0), K.Then():  # .cu:1316-1319
                a_log_value = K.local_scalar("float32")
                K.ptx.ld.global_.f32(a_log_value, A_log.ptr_to([head_idx_3]))
                _i32 = K.local_scalar("float32")
                K.ptx.ex2.approx.ftz.f32(_i32, a_log_value * K.float32(1.4426950408889634))
                K.ptx.st.shared.f32(smem_gate_rate_all.ptr_to([gate_rate_stage_f32]), _i32)
            _prep_bar_sync()  # .cu:1320-1332
            with K.serial(0, num_prep_iters, unroll=False) as prep_iter:  # .cu:1339-1340
                chunk_idx_3: K.int32 = prep_iter * 5 + prep_instance  # .cu:1341
                prep_stage_byte_base = (
                    K.cast(prep_stage, "int32") * SMEM_STAGE_BYTE_STRIDE
                )  # .cu:1342
                stage_f32 = prep_stage_byte_base // 4
                stage_bf16 = prep_stage_byte_base // 2  # .cu:1343
                chunk_is_full_2: K.int32 = K.if_then_else(
                    seq_len_4 >= (chunk_idx_3 + 1) * 32, 1, 0
                )  # .cu:1344
                early_beta_value = K.local_scalar("float32")  # .cu:1345
                early_gate0 = K.local_scalar("float32")  # .cu:1346
                K.assign(early_beta_value, K.float32(0.0))
                K.assign(early_gate0, K.float32(0.0))
                with K.If(chunk_is_full_2 != 0), K.Then():  # .cu:1347-1392
                    _mbarrier_wait(raw_inputs_free, prep_stage, prep_phase.phase)  # .cu:1348
                    with K.If(prep_local_warp == 0), K.Then():  # .cu:1349-1357
                        with K.If(K.cuda.elect_sync()), K.Then():
                            _mbarrier_arrive_expect_tx(gate_raw_full, prep_stage, 8704)  # .cu:1351  # fmt: skip
                            _tma_3d_gmem2smem(
                                smem_raw,
                                K.cast(smem, "int32"),
                                smem_g_raw_addr + prep_stage_byte_base,
                                g_tma,
                                0,
                                head_idx_3,
                                K.cast(bos_4 + K.cast(chunk_idx_3 * 32, "int64"), "int32"),
                                gate_raw_full.ptr_to([prep_stage]),
                            )  # .cu:1352
                            _tma_2d_gmem2smem(
                                smem_raw,
                                K.cast(smem, "int32"),
                                smem_beta_raw_addr + prep_stage_byte_base,
                                beta_tma_tmap,
                                head_idx_3 // 8 * 8,
                                K.cast(bos_4 + K.cast(chunk_idx_3 * 32, "int64"), "int32"),
                                gate_raw_full.ptr_to([prep_stage]),
                            )  # .cu:1353
                            _mbarrier_arrive_expect_tx(qk_raw_full, prep_stage, 16384)  # .cu:1354  # fmt: skip
                            _tma_4d_gmem2smem(
                                smem_raw,
                                K.cast(smem, "int32"),
                                smem_kd_addr + prep_stage_byte_base,
                                k_tma,
                                0,
                                K.cast(bos_4 + K.cast(chunk_idx_3 * 32, "int64"), "int32"),
                                head_idx_3,
                                0,
                                qk_raw_full.ptr_to([prep_stage]),
                            )  # .cu:1355
                    _mbarrier_wait(gate_raw_full, prep_stage, prep_phase.phase)  # .cu:1358
                    with K.If(K.And(prep_local_warp == 2, lane < 32)), K.Then():  # .cu:1359-1380
                        beta_raw_pair = K.alloc_local((1,), "uint32", align=4)  # .cu:1360
                        beta_raw_addr = K.local_scalar("int32")
                        K.ptx.add.s32(
                            beta_raw_addr,
                            smem_beta_raw_addr + prep_stage_byte_base,
                            lane * 16 + head_idx_3 % 8 // 2 * 4,
                        )
                        K.ptx.ld.shared.b32(beta_raw_pair[0], smem_raw.ptr_to([beta_raw_addr - K.cast(smem, "int32")]))  # .cu:1361  # fmt: skip
                        beta_raw_pair_fp32 = K.alloc_local((2,), "float32")  # .cu:1362
                        with K.unroll(1) as _pair:  # .cu:1363-1372
                            K.ptx.mov.b32(
                                beta_raw_pair_fp32[_pair * 2],
                                K.cuda.uint_as_float(beta_raw_pair[_pair + 0] << K.uint32(16)),
                            )
                            K.ptx.mov.b32(
                                beta_raw_pair_fp32[_pair * 2 + 1],
                                K.cuda.uint_as_float(
                                    beta_raw_pair[_pair + 0] & K.uint32(0xFFFF0000)
                                ),
                            )
                        beta_logit = K.local_scalar(
                            "float32", init=beta_raw_pair_fp32[0]
                        )  # .cu:1373
                        with K.If(head_idx_3 % 2 != 0), K.Then():  # .cu:1374-1376
                            K.assign(beta_logit, beta_raw_pair_fp32[1])
                        _tanh = K.local_scalar("float32")
                        K.ptx.tanh.approx.f32(_tanh, beta_logit * K.float32(0.5))
                        K.assign(
                            early_beta_value, _tanh * K.float32(0.5) + K.float32(0.5)
                        )  # .cu:1377-1379
                    with K.If(prep_tid < 128), K.Then():  # .cu:1381-1391
                        early_gate_rate = K.local_scalar("float32")  # .cu:1382
                        K.ptx.ld.shared.f32(early_gate_rate, smem_gate_rate_all.ptr_to([stage_f32]))
                        early_gate_bias = K.local_scalar("float32")  # .cu:1383
                        K.ptx.ld.global_.f32(
                            early_gate_bias, dt_bias.ptr_to([head_idx_3 * 128 + prep_tid])
                        )
                        _bits = K.local_scalar("uint16")
                        K.ptx.ld.shared.b16(_bits, smem_g_raw_all.ptr_to([stage_bf16 + prep_tid]))
                        early_gate_raw: K.f32 = K.cuda.bfloat162float(
                            K.reinterpret("bfloat16", _bits)
                        )  # .cu:1384-1385
                        early_gate_arg: K.f32 = early_gate_rate * (
                            early_gate_raw + early_gate_bias
                        )  # .cu:1386
                        _tanh = K.local_scalar("float32")
                        K.ptx.tanh.approx.f32(_tanh, early_gate_arg * K.float32(0.5))
                        early_gate_sigmoid: K.f32 = _tanh * K.float32(0.5) + K.float32(
                            0.5
                        )  # .cu:1387-1389
                        K.assign(
                            early_gate0,
                            K.float32(lower_bound)
                            * K.float32(1.4426950408889634)
                            * early_gate_sigmoid,
                        )  # .cu:1390
                _mbarrier_wait(smem_free, prep_stage, prep_phase.phase)  # .cu:1393
                with K.If(chunk_is_full_2 != 0), K.Then():  # .cu:1394-1400
                    with K.If(prep_local_warp == 0), K.Then():
                        with K.If(K.cuda.elect_sync()), K.Then():
                            _tma_4d_gmem2smem(
                                smem_raw,
                                K.cast(smem, "int32"),
                                smem_q_raw_prefetch_addr + prep_stage_byte_base,
                                q_tma,
                                0,
                                K.cast(bos_4 + K.cast(chunk_idx_3 * 32, "int64"), "int32"),
                                head_idx_3,
                                0,
                                qk_raw_full.ptr_to([prep_stage]),
                            )  # .cu:1397
                with K.If(chunk_is_full_2 == 0), K.Then():  # .cu:1401-1412
                    with K.unroll(4) as gate_load_pass:
                        gate_load_item: K.int32 = gate_load_pass * 128 + prep_tid  # .cu:1404
                        gate_load_row: K.int32 = gate_load_item // 16  # .cu:1405
                        gate_load_segment: K.int32 = gate_load_item % 16  # .cu:1406
                        gate_load_token: K.int64 = bos_4 + K.cast(
                            chunk_idx_3 * 32 + gate_load_row, "int64"
                        )  # .cu:1407
                        gate_load_base = K.local_scalar("int64")
                        K.ptx.mad.lo.s64(
                            gate_load_base, gate_load_token, K.int64(h), K.cast(head_idx_3, "int64")
                        )
                        K.ptx.mad.lo.s64(
                            gate_load_base,
                            gate_load_base,
                            K.int64(128),
                            K.cast(gate_load_segment * 8, "int64"),
                        )  # .cu:1408
                        K.ptx["cp.async.cg.shared.global"](
                            smem_g_raw_all.ptr_to(
                                [stage_bf16 + gate_load_item * 8]
                            ),
                            g.ptr_to([gate_load_base]),
                            16,
                            K.cast(K.if_then_else(chunk_idx_3 * 32 + gate_load_row < seq_len_4, 16, 0), "uint32"),
                        )  # .cu:1409-1410  # fmt: skip
                with K.If(chunk_is_full_2 == 0), K.Then():  # .cu:1413-1429
                    K.ptx.cp.async_.commit_group()  # .cu:1414
                    K.ptx.cp.async_.wait_group(0)  # .cu:1415
                    _prep_bar_sync()
                with K.If(K.And(prep_local_warp == 2, lane < 32)), K.Then():  # .cu:1430-1442
                    beta_value = K.local_scalar("float32", init=early_beta_value)  # .cu:1431
                    with K.If(chunk_is_full_2 == 0), K.Then():  # .cu:1432-1440
                        beta_token: K.int64 = bos_4 + K.cast(chunk_idx_3 * 32 + lane, "int64")
                        with K.If(chunk_idx_3 * 32 + lane < seq_len_4), K.Then():
                            _bits = K.local_scalar("uint16")
                            K.ptx.ld.global_.b16(
                                _bits,
                                beta.ptr_to(
                                    [beta_token * K.cast(h, "int64") + K.cast(head_idx_3, "int64")]
                                ),
                            )
                            beta_logit_1: K.f32 = K.cuda.bfloat162float(
                                K.reinterpret("bfloat16", _bits)
                            )  # .cu:1435
                            _tanh = K.local_scalar("float32")
                            K.ptx.tanh.approx.f32(_tanh, beta_logit_1 * K.float32(0.5))
                            K.assign(
                                beta_value, _tanh * K.float32(0.5) + K.float32(0.5)
                            )  # .cu:1436-1438
                    K.ptx.st.shared.f32(
                        smem_prep_beta_all.ptr_to([stage_f32 + lane]), beta_value
                    )  # .cu:1441
                with K.If(prep_tid < 128), K.Then():  # .cu:1443-1472
                    gate_col: K.int32 = prep_tid  # .cu:1444
                    gate_rate = K.local_scalar("float32")  # .cu:1445
                    K.ptx.ld.shared.f32(gate_rate, smem_gate_rate_all.ptr_to([stage_f32]))
                    gate_bias = K.local_scalar("float32")  # .cu:1446
                    K.ptx.ld.global_.f32(gate_bias, dt_bias.ptr_to([head_idx_3 * 128 + gate_col]))
                    prefix_log2 = K.local_scalar("float32", init=K.float32(0.0))  # .cu:1447
                    with K.serial(0, 32) as gate_row:  # .cu:1448-1471
                        gate_log2 = K.local_scalar("float32", init=K.float32(0.0))  # .cu:1450
                        gate_needs_compute = K.local_scalar("int32")
                        K.assign(gate_needs_compute, 1)  # .cu:1451
                        with K.If(gate_row == 0), K.Then():  # .cu:1452-1457
                            with K.If(chunk_is_full_2 != 0), K.Then():
                                K.assign(gate_log2, early_gate0)
                                K.assign(gate_needs_compute, 0)
                        with K.If(gate_needs_compute != 0), K.Then():  # .cu:1458-1468
                            with K.If(chunk_idx_3 * 32 + gate_row < seq_len_4), K.Then():
                                _bits = K.local_scalar("uint16")
                                K.ptx.ld.shared.b16(
                                    _bits,
                                    smem_g_raw_all.ptr_to([stage_bf16 + gate_row * 128 + gate_col]),
                                )
                                gate_raw: K.f32 = K.cuda.bfloat162float(
                                    K.reinterpret("bfloat16", _bits)
                                )  # .cu:1460-1461
                                gate_arg: K.f32 = gate_rate * (gate_raw + gate_bias)  # .cu:1462
                                _tanh = K.local_scalar("float32")
                                K.ptx.tanh.approx.f32(_tanh, gate_arg * K.float32(0.5))
                                gate_sigmoid: K.f32 = _tanh * K.float32(0.5) + K.float32(
                                    0.5
                                )  # .cu:1463-1465
                                K.assign(
                                    gate_log2,
                                    K.float32(lower_bound)
                                    * K.float32(1.4426950408889634)
                                    * gate_sigmoid,
                                )  # .cu:1466
                        K.assign(prefix_log2, prefix_log2 + gate_log2)  # .cu:1469
                        K.ptx.st.shared.f32(
                            smem_gate_all.ptr_to([stage_f32 + gate_row * 128 + gate_col]),
                            prefix_log2,
                        )  # .cu:1470
                _prep_bar_sync()  # .cu:1473-1485
                with K.If(chunk_is_full_2 != 0), K.Then():  # .cu:1486-1488
                    _mbarrier_wait(qk_raw_full, prep_stage, prep_phase.phase)
                with K.If(prep_tid < 128), K.Then():  # .cu:1489-1493
                    total_log2 = K.local_scalar("float32")  # .cu:1490
                    K.ptx.ld.shared.f32(
                        total_log2, smem_gt_prefix_all.ptr_to([stage_f32 + prep_tid])
                    )
                    _i64 = K.local_scalar("float32")
                    K.ptx.ex2.approx.ftz.f32(
                        _i64,
                        total_log2
                        - K.float32(lower_bound) * K.float32(1.4426950408889634) * K.float32(16.0),
                    )
                    K.ptx.st.shared.f32(
                        smem_restore_factor_all.ptr_to([stage_f32 + prep_tid]), _i64
                    )  # .cu:1491-1492
                with K.If(prep_tid == 0), K.Then():  # .cu:1494-1497
                    _i642 = K.local_scalar("float32")
                    K.ptx.ex2.approx.ftz.f32(
                        _i642,
                        K.float32(lower_bound) * K.float32(1.4426950408889634) * K.float32(16.0),
                    )
                    K.ptx.st.shared.f32(smem_restore_factor_all.ptr_to([stage_f32 + 128]), _i642)
                q_u32 = K.decl_buffer((total_tokens * h * D_HEAD // 2,), "uint32", data=q.data)
                k_u32 = K.decl_buffer((total_tokens * h * D_HEAD // 2,), "uint32", data=k.data)
                with K.serial(0, 4, unroll=False) as work_pass:  # .cu:1498-1694 (#pragma unroll 1)
                    work_item = K.local_scalar("int32")
                    K.assign(work_item, work_pass * 128 + prep_tid)  # .cu:1500
                    row_1: K.int32 = work_item // 16  # .cu:1501
                    segment_1: K.int32 = work_item % 16  # .cu:1502
                    token_1: K.int64 = bos_4 + K.cast(chunk_idx_3 * 32 + row_1, "int64")  # .cu:1503
                    token_valid_1: K.int32 = K.if_then_else(
                        chunk_idx_3 * 32 + row_1 < seq_len_4, 1, 0
                    )  # .cu:1504
                    gmem_base: K.int64 = (
                        token_1 * K.cast(h, "int64") + K.cast(head_idx_3, "int64")
                    ) * 128 + K.cast(segment_1 * 8, "int64")  # .cu:1505
                    qk_byte_off = segment_1 * 8 // 64 * 4096 + row_1 * 128 + segment_1 * 8 % 64 * 2
                    qk_swizzled_off = qk_byte_off ^ ((qk_byte_off >> 7 & 7) << 4)
                    q_raw_vec = K.alloc_local((8,), "float32")  # .cu:1506
                    k_raw_vec = K.alloc_local((8,), "float32")  # .cu:1507
                    with K.unroll(8) as _zi:  # .cu:1508-1523
                        K.assign(q_raw_vec[_zi], K.float32(0.0))
                        K.assign(k_raw_vec[_zi], K.float32(0.0))
                    with K.If(chunk_is_full_2 != 0):
                        with K.Then():  # .cu:1524-1562
                            packed = K.alloc_local((4,), "uint32", align=16)  # .cu:1525
                            K.ptx.ld.shared.v4.b32(packed[0], packed[1], packed[2], packed[3], smem_raw.ptr_to([smem_q_raw_prefetch_addr + prep_stage_byte_base + qk_swizzled_off - K.cast(smem, "int32")]))  # .cu:1526-1528  # fmt: skip
                            packed_fp32 = K.alloc_local((8,), "float32")  # .cu:1529
                            with K.unroll(4) as _pair:  # .cu:1530-1539
                                K.assign(
                                    packed_fp32[_pair * 2],
                                    K.cuda.uint_as_float(packed[_pair + 0] << K.uint32(16)),
                                )
                                K.assign(
                                    packed_fp32[_pair * 2 + 1],
                                    K.cuda.uint_as_float(packed[_pair + 0] & K.uint32(0xFFFF0000)),
                                )
                            with K.unroll(8) as value_idx:  # .cu:1540-1543
                                K.assign(q_raw_vec[value_idx], packed_fp32[value_idx])
                            packed_0 = K.alloc_local((4,), "uint32", align=16)  # .cu:1544
                            K.ptx.ld.shared.v4.b32(packed_0[0], packed_0[1], packed_0[2], packed_0[3], smem_raw.ptr_to([smem_kd_addr + prep_stage_byte_base + qk_swizzled_off - K.cast(smem, "int32")]))  # .cu:1545-1547  # fmt: skip
                            packed_0_fp32 = K.alloc_local((8,), "float32")  # .cu:1548
                            with K.unroll(4) as _pair:  # .cu:1549-1558
                                K.assign(
                                    packed_0_fp32[_pair * 2],
                                    K.cuda.uint_as_float(packed_0[_pair + 0] << K.uint32(16)),
                                )
                                K.assign(
                                    packed_0_fp32[_pair * 2 + 1],
                                    K.cuda.uint_as_float(
                                        packed_0[_pair + 0] & K.uint32(0xFFFF0000)
                                    ),
                                )
                            with K.unroll(8) as value_idx_1:  # .cu:1559-1562
                                K.assign(k_raw_vec[value_idx_1], packed_0_fp32[value_idx_1])
                        with K.Else():
                            with K.If(token_valid_1 != 0), K.Then():  # .cu:1563-1602
                                with K.unroll(1) as _blk:  # .cu:1564-1582
                                    _vldq = K.alloc_local((4,), "uint32", align=16)
                                    K.ptx.ld.global_.v4.b32(
                                        _vldq[0],
                                        _vldq[1],
                                        _vldq[2],
                                        _vldq[3],
                                        q_u32.ptr_to([gmem_base // 2 + _blk * 4]),
                                    )
                                    with K.unroll(4) as _pair:
                                        K.assign(
                                            q_raw_vec[0 + _blk * 8 + _pair * 2],
                                            (K.cuda.uint_as_float(_vldq[_pair] << K.uint32(16))),
                                        )
                                        K.assign(
                                            q_raw_vec[0 + _blk * 8 + _pair * 2 + 1],
                                            (
                                                K.cuda.uint_as_float(
                                                    _vldq[_pair] & K.uint32(0xFFFF0000)
                                                )
                                            ),
                                        )
                                with K.unroll(1) as _blk:  # .cu:1583-1601
                                    _vldk = K.alloc_local((4,), "uint32", align=16)
                                    K.ptx.ld.global_.v4.b32(
                                        _vldk[0],
                                        _vldk[1],
                                        _vldk[2],
                                        _vldk[3],
                                        k_u32.ptr_to([gmem_base // 2 + _blk * 4]),
                                    )
                                    with K.unroll(4) as _pair:
                                        K.assign(
                                            k_raw_vec[0 + _blk * 8 + _pair * 2],
                                            (K.cuda.uint_as_float(_vldk[_pair] << K.uint32(16))),
                                        )
                                        K.assign(
                                            k_raw_vec[0 + _blk * 8 + _pair * 2 + 1],
                                            (
                                                K.cuda.uint_as_float(
                                                    _vldk[_pair] & K.uint32(0xFFFF0000)
                                                )
                                            ),
                                        )
                    q_sum = K.local_scalar("float32")  # .cu:1603
                    k_sum = K.local_scalar("float32")  # .cu:1604
                    K.assign(q_sum, K.float32(0.0))
                    K.assign(k_sum, K.float32(0.0))
                    with K.serial(0, 8) as elem_in_segment:  # .cu:1605-1612
                        K.ptx.fma.rn.f32(
                            q_sum, q_raw_vec[elem_in_segment], q_raw_vec[elem_in_segment], q_sum
                        )  # .cu:1608
                        K.ptx.fma.rn.f32(
                            k_sum, k_raw_vec[elem_in_segment], k_raw_vec[elem_in_segment], k_sum
                        )  # .cu:1610
                    # .cu:1613-1628 butterfly reduction over the 16-lane segment (q then k per step).
                    for _off in (8, 4, 2, 1):
                        for _acc in (q_sum, k_sum):
                            K.assign(_acc, _acc + _shfl_bfly_f32(_acc, K.int32(_off)))
                    q_inv = K.local_scalar("float32")  # .cu:1629-1630
                    K.ptx.rsqrt.approx.ftz.f32(q_inv, q_sum + K.float32(1e-06))
                    k_inv = K.local_scalar("float32")  # .cu:1631-1632
                    K.ptx.rsqrt.approx.ftz.f32(k_inv, k_sum + K.float32(1e-06))
                    with K.unroll(4) as _ls:  # .cu:1633-1636
                        _pk = K.local_scalar("uint64")
                        K.ptx.mul.rn.ftz.f32x2(
                            _pk,
                            K.cuda.make_float2(q_raw_vec[_ls * 2], q_raw_vec[_ls * 2 + 1]),
                            K.cuda.make_float2(q_inv, q_inv),
                        )
                        K.assign(q_raw_vec[_ls * 2], K.cuda.float2_x(_pk))
                        K.assign(q_raw_vec[_ls * 2 + 1], K.cuda.float2_y(_pk))
                    with K.unroll(4) as _ls:  # .cu:1637-1640
                        _pk = K.local_scalar("uint64")
                        K.ptx.mul.rn.ftz.f32x2(
                            _pk,
                            K.cuda.make_float2(k_raw_vec[_ls * 2], k_raw_vec[_ls * 2 + 1]),
                            K.cuda.make_float2(k_inv, k_inv),
                        )
                        K.assign(k_raw_vec[_ls * 2], K.cuda.float2_x(_pk))
                        K.assign(k_raw_vec[_ls * 2 + 1], K.cuda.float2_y(_pk))
                    qd_vec = K.alloc_local((8,), "float32")  # .cu:1641
                    kd_vec = K.alloc_local((8,), "float32")  # .cu:1642
                    ki_vec = K.alloc_local((8,), "float32")  # .cu:1643
                    prefix_vec = K.alloc_local((8,), "float32")
                    with K.unroll(2) as prefix_vec_idx:
                        K.ptx.ld.shared.v4.f32(
                            prefix_vec[prefix_vec_idx * 4],
                            prefix_vec[prefix_vec_idx * 4 + 1],
                            prefix_vec[prefix_vec_idx * 4 + 2],
                            prefix_vec[prefix_vec_idx * 4 + 3],
                            smem_gate_all.ptr_to(
                                [stage_f32 + row_1 * 128 + segment_1 * 8 + prefix_vec_idx * 4]
                            ),
                        )
                    with K.serial(0, 8) as elem_in_segment_1:  # .cu:1644-1653
                        col: K.int32 = segment_1 * 8 + elem_in_segment_1  # .cu:1645
                        prefix: K.f32 = prefix_vec[elem_in_segment_1]  # .cu:1646
                        common_log2: K.f32 = (
                            K.float32(lower_bound) * K.float32(1.4426950408889634) * K.float32(16.0)
                        )  # .cu:1647
                        decay = K.local_scalar("float32")  # .cu:1648-1649
                        K.ptx.ex2.approx.ftz.f32(decay, prefix - common_log2)
                        K.assign(qd_vec[elem_in_segment_1], decay)  # .cu:1650
                        K.assign(kd_vec[elem_in_segment_1], decay)  # .cu:1651
                        K.assign(
                            ki_vec[elem_in_segment_1],
                            K.cuda.fdividef(k_raw_vec[elem_in_segment_1], decay),
                        )  # .cu:1652 (-use_fast_math: / -> div.approx.f32)
                    with K.unroll(4) as _ls:  # .cu:1654-1656
                        _pk = K.local_scalar("uint64")
                        K.ptx.mul.rn.ftz.f32x2(
                            _pk,
                            K.cuda.make_float2(qd_vec[_ls * 2], qd_vec[_ls * 2 + 1]),
                            K.cuda.make_float2(q_raw_vec[_ls * 2], q_raw_vec[_ls * 2 + 1]),
                        )
                        K.assign(qd_vec[_ls * 2], K.cuda.float2_x(_pk))
                        K.assign(qd_vec[_ls * 2 + 1], K.cuda.float2_y(_pk))
                    with K.unroll(4) as _ls:  # .cu:1657-1660
                        _pk = K.local_scalar("uint64")
                        K.ptx.mul.rn.ftz.f32x2(
                            _pk,
                            K.cuda.make_float2(qd_vec[_ls * 2], qd_vec[_ls * 2 + 1]),
                            K.cuda.make_float2(K.float32(scale), K.float32(scale)),
                        )
                        K.assign(qd_vec[_ls * 2], K.cuda.float2_x(_pk))
                        K.assign(qd_vec[_ls * 2 + 1], K.cuda.float2_y(_pk))
                    with K.unroll(4) as _ls:  # .cu:1661-1663
                        _pk = K.local_scalar("uint64")
                        K.ptx.mul.rn.ftz.f32x2(
                            _pk,
                            K.cuda.make_float2(kd_vec[_ls * 2], kd_vec[_ls * 2 + 1]),
                            K.cuda.make_float2(k_raw_vec[_ls * 2], k_raw_vec[_ls * 2 + 1]),
                        )
                        K.assign(kd_vec[_ls * 2], K.cuda.float2_x(_pk))
                        K.assign(kd_vec[_ls * 2 + 1], K.cuda.float2_y(_pk))
                    packed_1 = K.alloc_local((4,), "uint32", align=4)  # .cu:1664
                    with K.unroll(4) as _lp:  # .cu:1665-1669
                        K.assign(
                            packed_1[_lp],
                            K.cuda.float22bfloat162_rn(
                                qd_vec[_lp * 2 + 0], qd_vec[_lp * 2 + 1 + 0]
                            ),
                        )
                    with K.unroll(4) as word:  # .cu:1670-1673
                        K.ptx.st.shared.b32(smem_raw.ptr_to([smem_qd_addr + prep_stage_byte_base + qk_swizzled_off + word * 4 - K.cast(smem, "int32")]), packed_1[word])  # fmt: skip
                    packed_0_1 = K.alloc_local((4,), "uint32", align=4)  # .cu:1674
                    with K.unroll(4) as _lp:  # .cu:1675-1679
                        K.assign(
                            packed_0_1[_lp],
                            K.cuda.float22bfloat162_rn(
                                kd_vec[_lp * 2 + 0], kd_vec[_lp * 2 + 1 + 0]
                            ),
                        )
                    with K.unroll(4) as word_1:  # .cu:1680-1683
                        K.ptx.st.shared.b32(smem_raw.ptr_to([smem_kd_addr + prep_stage_byte_base + qk_swizzled_off + word_1 * 4 - K.cast(smem, "int32")]), packed_0_1[word_1])  # fmt: skip
                    packed_1_1 = K.alloc_local((4,), "uint32", align=4)  # .cu:1684
                    with K.unroll(4) as _lp:  # .cu:1685-1689
                        K.assign(
                            packed_1_1[_lp],
                            K.cuda.float22bfloat162_rn(
                                ki_vec[_lp * 2 + 0], ki_vec[_lp * 2 + 1 + 0]
                            ),
                        )
                    with K.unroll(4) as word_2:  # .cu:1690-1693
                        K.ptx.st.shared.b32(smem_raw.ptr_to([smem_ki_addr + prep_stage_byte_base + qk_swizzled_off + word_2 * 4 - K.cast(smem, "int32")]), packed_1_1[word_2])  # fmt: skip
                _prep_bar_sync()  # .cu:1695-1707
                pair_row_base: K.int32 = prep_local_warp // 2 * 16  # .cu:1708
                pair_col_base: K.int32 = prep_local_warp % 2 * 16  # .cu:1709
                a_frag = K.alloc_local((4,), "uint32", align=4)  # .cu:1710
                b_frag = K.alloc_local((4,), "uint32", align=4)  # .cu:1711
                acc = K.alloc_local((8,), "float32", align=4)  # .cu:1712
                with K.If(pair_row_base >= pair_col_base):
                    with K.Then():  # .cu:1713-1990
                        # .cu:1714-1990 swizzled ldmatrix cursors: one base per operand, then a
                        # fixed XOR walk (with a +256 tile step at index 4).  Cheap pure integer
                        # arithmetic, so plain Python values -- the expression is re-emitted at each
                        # of the two chains and ptxas folds the duplicates.
                        _a_cursors = [
                            lane // 16 // 8 * 256
                            + (pair_row_base + lane % 16) * 8
                            + (lane // 16 % 8 * 16 ^ ((pair_row_base + lane % 16 & 7) << 4)) // 16
                        ]
                        _b_cursors = [
                            lane % 16 // 8 // 8 * 256
                            + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8
                            + (
                                lane % 16 // 8 % 8 * 16
                                ^ ((pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4)
                            )
                            // 16
                        ]
                        for _x, _step in ((2, 0), (6, 0), (2, 0), (6, 256), (2, 0), (6, 0), (2, 0)):
                            _a = _a_cursors[-1] ^ _x
                            _b = (
                                ((_b_cursors[-1] + 256) ^ _x) + _step - 256
                                if _step
                                else ((_b_cursors[-1] + 256) ^ _x) - 256
                            )
                            _a_cursors.append(_a + _step if _step else _a)
                            _b_cursors.append(_b)

                        def _chain8(a_base):
                            # 8 ldmatrix.x4 pairs walking the swizzled cursors, feeding one accumulate
                            # chain: step 0 zeroes C, steps 1-7 tie C to acc (both 16x8x16 halves).
                            for _s in range(8):
                                K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                                    a_frag[0],
                                    a_frag[1],
                                    a_frag[2],
                                    a_frag[3],
                                    smem_raw.ptr_to([(a_base + prep_stage_byte_base + (_a_cursors[_s]) * 16) - K.cast(smem, "int32")]),
                                )  # fmt: skip
                                K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                                    b_frag[0],
                                    b_frag[1],
                                    b_frag[2],
                                    b_frag[3],
                                    smem_raw.ptr_to([(smem_ki_addr + prep_stage_byte_base + (_b_cursors[_s]) * 16) - K.cast(smem, "int32")]),
                                )  # fmt: skip
                                if _s == 0:
                                    _mma_m16n8k16_bf16_zero(acc, a_frag, b_frag)
                                    _mma_m16n8k16_bf16_zero_off4(acc, a_frag, b_frag)
                                else:
                                    _mma_m16n8k16_bf16_acc(acc, a_frag, b_frag)
                                    _mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag)

                        _chain8(smem_kd_addr)  # .cu:1714-1825 first chain (kd a-side, ki b-side)
                        row0: K.int32 = pair_row_base + lane // 4  # .cu:1826
                        row1: K.int32 = row0 + 8  # .cu:1827
                        col0: K.int32 = pair_col_base + lane % 4 * 2  # .cu:1828
                        beta0 = K.local_scalar("float32")  # .cu:1829
                        K.ptx.ld.shared.f32(beta0, smem_prep_beta_all.ptr_to([stage_f32 + row0]))
                        beta1 = K.local_scalar("float32")  # .cu:1830
                        K.ptx.ld.shared.f32(beta1, smem_prep_beta_all.ptr_to([stage_f32 + row1]))
                        seed = K.alloc_local((8,), "float32")  # .cu:1831
                        with K.unroll(8) as _zi:  # .cu:1832-1839
                            K.ptx.mov.b32(seed[_zi], K.float32(0.0))
                        # .cu:1840-1863 strict lower-triangular mask x beta, statically expanded.
                        for _i, (_row, _dc, _beta) in enumerate(
                            [
                                (row0, 0, beta0),
                                (row0, 1, beta0),
                                (row1, 0, beta1),
                                (row1, 1, beta1),
                                (row0, 8, beta0),
                                (row0, 9, beta0),
                                (row1, 8, beta1),
                                (row1, 9, beta1),
                            ]
                        ):
                            with K.If(_row > (col0 if _dc == 0 else col0 + _dc)), K.Then():
                                K.assign(seed[_i], acc[_i] * _beta)
                        seed_packed = K.alloc_local((4,), "uint32", align=4)  # .cu:1864
                        with K.unroll(4) as _lp:  # .cu:1865-1869
                            K.ptx.mov.b32(
                                seed_packed[_lp],
                                K.cuda.float22bfloat162_rn(
                                    seed[_lp * 2 + 0], seed[_lp * 2 + 1 + 0]
                                ),
                            )
                        seed_lane_row: K.int32 = lane % 16  # .cu:1870
                        seed_lane_col: K.int32 = lane // 16 * 8  # .cu:1871
                        byte_off: K.int32 = (pair_row_base + seed_lane_row) * 128 + (
                            pair_col_base + seed_lane_col
                        ) * 2  # .cu:1872
                        swizzled_off: K.int32 = byte_off ^ ((byte_off >> 7 & 7) << 4)  # .cu:1873
                        seed_addr: K.int32 = (
                            smem_inv_work_addr + prep_stage_byte_base + swizzled_off
                        )  # .cu:1874
                        # .cu:1875-1878 stmatrix.x4 (seed into inv_work)
                        K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                            smem_raw.ptr_to([(seed_addr) - K.cast(smem, "int32")]),
                            seed_packed[0],
                            seed_packed[1],
                            seed_packed[2],
                            seed_packed[3],
                        )
                        _chain8(smem_qd_addr)  # .cu:1879-1990 second chain (qd a-side, ki b-side)
                    with K.Else():  # .cu:1991-2000
                        with K.unroll(8) as _zi:
                            K.ptx.mov.b32(acc[_zi], K.float32(0.0))
                row0_1: K.int32 = pair_row_base + lane // 4  # .cu:2001
                row1_1: K.int32 = row0_1 + 8  # .cu:2002
                col0_1: K.int32 = pair_col_base + lane % 4 * 2  # .cu:2003
                mqk = K.alloc_local((8,), "float32")  # .cu:2004
                with K.unroll(8) as _zi:  # .cu:2005-2012
                    K.ptx.mov.b32(mqk[_zi], K.float32(0.0))
                # .cu:2013-2036 lower-triangular mask, statically expanded over the 8 acc slots.
                for _i, (_row, _dc) in enumerate(
                    [
                        (row0_1, 0),
                        (row0_1, 1),
                        (row1_1, 0),
                        (row1_1, 1),
                        (row0_1, 8),
                        (row0_1, 9),
                        (row1_1, 8),
                        (row1_1, 9),
                    ]
                ):
                    with K.If(_row >= (col0_1 if _dc == 0 else col0_1 + _dc)), K.Then():
                        K.assign(mqk[_i], acc[_i])
                mqk_packed = K.alloc_local((4,), "uint32", align=4)  # .cu:2037
                with K.unroll(4) as _lp:  # .cu:2038-2042
                    K.ptx.mov.b32(
                        mqk_packed[_lp],
                        K.cuda.float22bfloat162_rn(mqk[_lp * 2 + 0], mqk[_lp * 2 + 1 + 0]),
                    )
                with K.unroll(2) as publish_pair:  # .cu:2043-2051
                    publish_row: K.int32 = pair_col_base + publish_pair * 8 + (lane & 7)  # .cu:2045
                    publish_col: K.int32 = 128 + pair_row_base + lane // 8 * 8  # .cu:2046
                    _pub_base: K.int32 = (
                        publish_col // 64 * 4096 + publish_row * 128 + publish_col % 64 * 2
                    )
                    _pub_addr: K.int32 = (
                        smem_final_trans_addr
                        + prep_stage_byte_base
                        + (_pub_base ^ ((_pub_base >> 7 & 7) << 4))
                    )  # .cu:2047
                    # .cu:2048-2050 stmatrix.x2.trans
                    K.ptx.stmatrix.sync.aligned.m8n8.x2.trans.shared.b16(
                        smem_raw.ptr_to([(_pub_addr) - K.cast(smem, "int32")]),
                        mqk_packed[publish_pair * 2],
                        mqk_packed[publish_pair * 2 + 1],
                    )
                _prep_bar_sync()  # .cu:2052-2064
                with K.If(prep_tid < 128), K.Then():  # .cu:2065-2069
                    total_log2_1 = K.local_scalar("float32")  # .cu:2066
                    K.ptx.ld.shared.f32(
                        total_log2_1, smem_gt_prefix_all.ptr_to([stage_f32 + prep_tid])
                    )
                    _pk = K.local_scalar("float32")
                    K.ptx.ex2.approx.ftz.f32(_pk, total_log2_1)
                    K.ptx.st.shared.f32(
                        smem_gt_all.ptr_to([stage_f32 + prep_tid]), _pk
                    )  # .cu:2067-2068

                def _restore_block(n_passes, row_of):
                    # .cu:2070-2187 / 2405-2522 (same body, two row maps): reload the staged
                    # qd/kd/ki tiles, rescale by the chunk restore factor, republish.
                    restore_scale = K.local_scalar("float32")
                    K.ptx.ld.shared.f32(
                        restore_scale, smem_restore_factor_all.ptr_to([stage_f32 + 128])
                    )
                    restore_factor = K.alloc_local((8,), "float32")
                    restore_segment: K.int32 = lane & 15
                    with K.unroll(2) as restore_vec:
                        K.ptx.ld.shared.v4.f32(
                            restore_factor[restore_vec * 4],
                            restore_factor[restore_vec * 4 + 1],
                            restore_factor[restore_vec * 4 + 2],
                            restore_factor[restore_vec * 4 + 3],
                            smem_restore_factor_all.ptr_to(
                                [stage_f32 + restore_segment * 8 + restore_vec * 4]
                            ),
                        )
                    with K.serial(0, n_passes, unroll=False) as restore_pass:  # (#pragma unroll 1)
                        restore_row: K.int32 = row_of(restore_pass)
                        restore_byte_off = (
                            restore_segment * 8 // 64 * 4096
                            + restore_row * 128
                            + restore_segment * 8 % 64 * 2
                        )
                        restore_swizzled_off = restore_byte_off ^ ((restore_byte_off >> 7 & 7) << 4)
                        restore_qd_values = K.alloc_local((8,), "float32")
                        restore_kd_values = K.alloc_local((8,), "float32")
                        restore_ki_values = K.alloc_local((8,), "float32")
                        for _dst, _src_addr in (
                            (restore_qd_values, smem_qd_addr),
                            (restore_kd_values, smem_kd_addr),
                            (restore_ki_values, smem_ki_addr),
                        ):
                            _packed = K.alloc_local((4,), "uint32", align=16)
                            K.ptx.ld.shared.v4.b32(_packed[0], _packed[1], _packed[2], _packed[3], smem_raw.ptr_to([_src_addr + prep_stage_byte_base + restore_swizzled_off - K.cast(smem, "int32")]))  # fmt: skip
                            _packed_fp32 = K.alloc_local((8,), "float32")
                            with K.unroll(4) as _pair:
                                K.ptx.mov.b32(
                                    _packed_fp32[_pair * 2],
                                    K.cuda.uint_as_float(_packed[_pair + 0] << K.uint32(16)),
                                )
                                K.ptx.mov.b32(
                                    _packed_fp32[_pair * 2 + 1],
                                    K.cuda.uint_as_float(_packed[_pair + 0] & K.uint32(0xFFFF0000)),
                                )
                            with K.unroll(8) as _value_idx:
                                K.ptx.mov.b32(_dst[_value_idx], _packed_fp32[_value_idx])
                        restore_kr_values = K.alloc_local((8,), "float32")
                        with K.unroll(8) as restore_elem:
                            K.ptx.mov.b32(
                                restore_kr_values[restore_elem],
                                (restore_ki_values[restore_elem] * restore_factor[restore_elem]),
                            )
                        for _vals in (restore_qd_values, restore_kd_values):
                            with K.unroll(4) as _ls:
                                _pk = K.local_scalar("uint64")
                                K.ptx.mul.rn.ftz.f32x2(
                                    _pk,
                                    K.cuda.make_float2(_vals[_ls * 2], _vals[_ls * 2 + 1]),
                                    K.cuda.make_float2(restore_scale, restore_scale),
                                )
                                K.ptx.mov.b32(_vals[_ls * 2], K.cuda.float2_x(_pk))
                                K.ptx.mov.b32(_vals[_ls * 2 + 1], K.cuda.float2_y(_pk))
                        for _vals, _dst_addr in (
                            (restore_qd_values, smem_qd_addr),
                            (restore_kd_values, smem_kd_addr),
                            (restore_kr_values, smem_kr_trans_addr),
                        ):
                            _packed_out = K.alloc_local((4,), "uint32", align=4)
                            with K.unroll(4) as _lp:
                                K.ptx.mov.b32(
                                    _packed_out[_lp],
                                    K.cuda.float22bfloat162_rn(
                                        _vals[_lp * 2 + 0], _vals[_lp * 2 + 1 + 0]
                                    ),
                                )
                            with K.unroll(4) as _word:
                                K.ptx.st.shared.b32(smem_raw.ptr_to([_dst_addr + prep_stage_byte_base + restore_swizzled_off + _word * 4 - K.cast(smem, "int32")]), _packed_out[_word])  # fmt: skip

                with K.If(prep_local_warp >= 2), K.Then():  # .cu:2070-2187
                    _restore_block(
                        6, lambda _pass: 8 + (prep_local_warp - 2) * 12 + _pass * 2 + (lane >> 4)
                    )  # .cu:2082
                with K.If(prep_local_warp == 0), K.Then():  # .cu:2188-2250
                    inverse_row: K.int32 = lane  # .cu:2189
                    diag_block: K.int32 = inverse_row // 8  # .cu:2190
                    lane_in_diag: K.int32 = lane & 7  # .cu:2191
                    inv_row = K.alloc_local((8,), "float32")  # .cu:2192
                    packed_5 = K.alloc_local((4,), "uint32", align=16)  # .cu:2193
                    byte_off_1: K.int32 = inverse_row * 128 + diag_block * 8 * 2  # .cu:2194
                    swizzled_off_1: K.int32 = byte_off_1 ^ ((byte_off_1 >> 7 & 7) << 4)  # .cu:2195
                    K.ptx.ld.shared.v4.b32(
                        packed_5[0],
                        packed_5[1],
                        packed_5[2],
                        packed_5[3],
                        smem_raw.ptr_to(
                            [
                                smem_inv_work_addr
                                + prep_stage_byte_base
                                + swizzled_off_1
                                - K.cast(smem, "int32")
                            ]
                        ),
                    )  # .cu:2196-2198
                    packed_fp32_2 = K.alloc_local((8,), "float32")  # .cu:2199
                    with K.unroll(4) as _pair:  # .cu:2200-2209
                        K.ptx.mov.b32(
                            packed_fp32_2[_pair * 2],
                            K.cuda.uint_as_float(packed_5[_pair + 0] << K.uint32(16)),
                        )
                        K.ptx.mov.b32(
                            packed_fp32_2[_pair * 2 + 1],
                            K.cuda.uint_as_float(packed_5[_pair + 0] & K.uint32(0xFFFF0000)),
                        )
                    with K.unroll(8) as value_idx_5:  # .cu:2210-2213
                        K.ptx.mov.b32(inv_row[value_idx_5], packed_fp32_2[value_idx_5])
                    with K.unroll(8) as diag_elem:  # .cu:2214-2219
                        with K.If(lane_in_diag == diag_elem), K.Then():
                            K.assign(inv_row[diag_elem], K.float32(1.0))
                    diag_group_base: K.int32 = lane - lane_in_diag  # .cu:2220
                    # .cu:2221-2237 lower-triangular Gauss elimination, statically expanded over the
                    # 7 pivot rows (the source's #pragma unroll of a compile-time trip count).
                    for _p in range(7):
                        row_scale: K.f32 = -inv_row[_p]  # .cu:2223
                        for _j in range(_p):
                            pivot_lane: K.int32 = diag_group_base + _p  # .cu:2226
                            pivot = K.local_scalar(
                                "float32", init=_shfl_idx_f32(inv_row[_j], pivot_lane)
                            )  # .cu:2227-2228
                            with K.If(lane_in_diag > _p), K.Then():  # .cu:2229-2232
                                K.ptx.fma.rn.f32(
                                    inv_row[_j], row_scale, pivot, inv_row[_j]
                                )  # .cu:2230-2231
                        with K.If(lane_in_diag > _p), K.Then():  # .cu:2234-2236
                            K.ptx.mov.b32(inv_row[_p], row_scale)
                    packed_0_3 = K.alloc_local((4,), "uint32", align=4)  # .cu:2238
                    with K.unroll(4) as _lp:  # .cu:2239-2243
                        K.ptx.mov.b32(
                            packed_0_3[_lp],
                            K.cuda.float22bfloat162_rn(
                                inv_row[_lp * 2 + 0], inv_row[_lp * 2 + 1 + 0]
                            ),
                        )
                    byte_off_1_1: K.int32 = inverse_row * 128 + diag_block * 8 * 2  # .cu:2244
                    swizzled_off_2: K.int32 = byte_off_1_1 ^ (
                        (byte_off_1_1 >> 7 & 7) << 4
                    )  # .cu:2245
                    with K.unroll(4) as word_6:  # .cu:2246-2249
                        K.ptx.st.shared.b32(smem_raw.ptr_to([smem_inv_work_addr + prep_stage_byte_base + swizzled_off_2 + word_6 * 4 - K.cast(smem, "int32")]), packed_0_3[word_6])  # fmt: skip
                with K.If(prep_local_warp < 2), K.Then():  # .cu:2251-2256
                    with K.If(K.cuda.elect_sync()), K.Then():
                        _mbarrier_arrive(prep_diag_ready, prep_stage)
                    _mbarrier_wait(prep_diag_ready, prep_stage, prep_phase.phase)
                with K.If(prep_local_warp < 2), K.Then():  # .cu:2257-2322
                    lane_row: K.int32 = lane & 7  # .cu:2258
                    byte_off_2: K.int32 = (prep_local_warp * 16 + 8 + lane_row) * 128 + (
                        prep_local_warp * 16 + 8
                    ) * 2  # .cu:2259
                    swizzled_off_3: K.int32 = byte_off_2 ^ ((byte_off_2 >> 7 & 7) << 4)  # .cu:2260
                    matrix_addr = K.local_scalar("int32")
                    K.ptx.add.s32(
                        matrix_addr, smem_inv_work_addr + prep_stage_byte_base, swizzled_off_3
                    )  # .cu:2261
                    d_frag = K.alloc_local((2,), "uint32", align=4)  # .cu:2268
                    c_frag = K.alloc_local((1,), "uint32", align=4)  # .cu:2269
                    dc_acc = K.alloc_local((4,), "float32", align=4)  # .cu:2270
                    dc_bf16 = K.alloc_local((2,), "uint32", align=4)  # .cu:2271
                    inv_a_frag = K.alloc_local((1,), "uint32", align=4)  # .cu:2272
                    o_acc = K.alloc_local((4,), "float32", align=4)  # .cu:2273
                    o_bf16 = K.alloc_local((2,), "uint32", align=4)  # .cu:2274
                    # .cu:2275-2278 ldmatrix.x1 d_frag[0]
                    K.ptx.ldmatrix.sync.aligned.m8n8.x1.shared.b16(
                        d_frag[0], smem_raw.ptr_to([(matrix_addr) - K.cast(smem, "int32")])
                    )
                    # .cu:2279-2282 ldmatrix.x1 d_frag[1] (same address)
                    K.ptx.ldmatrix.sync.aligned.m8n8.x1.shared.b16(
                        d_frag[1], smem_raw.ptr_to([(matrix_addr) - K.cast(smem, "int32")])
                    )
                    # .cu:2283-2286 ldmatrix.x1.trans c_frag
                    K.ptx.xor.b32(matrix_addr, matrix_addr, K.int32(16))
                    K.ptx.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16(
                        c_frag[0], smem_raw.ptr_to([(matrix_addr) - K.cast(smem, "int32")])
                    )
                    _mma_m16n8k8_bf16_zero(dc_acc, d_frag, c_frag)  # .cu:2287-2289
                    with K.unroll(2) as _ls:  # .cu:2290-2293
                        _pk = K.local_scalar("uint64")
                        K.ptx.mul.rn.ftz.f32x2(
                            _pk,
                            K.cuda.make_float2(dc_acc[_ls * 2], dc_acc[_ls * 2 + 1]),
                            K.cuda.make_float2(K.float32(-1.0), K.float32(-1.0)),
                        )
                        K.ptx.mov.b32(dc_acc[_ls * 2], K.cuda.float2_x(_pk))
                        K.ptx.mov.b32(dc_acc[_ls * 2 + 1], K.cuda.float2_y(_pk))
                    with K.unroll(2) as _lp:  # .cu:2294-2298
                        K.ptx.mov.b32(
                            dc_bf16[_lp],
                            K.cuda.float22bfloat162_rn(
                                dc_acc[_lp * 2 + 0], dc_acc[_lp * 2 + 1 + 0]
                            ),
                        )
                    # .cu:2299-2302 ldmatrix.x1.trans inv_a_frag
                    K.ptx.add.s32(matrix_addr, matrix_addr, K.int32(-1024))
                    K.ptx.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16(
                        inv_a_frag[0], smem_raw.ptr_to([(matrix_addr) - K.cast(smem, "int32")])
                    )
                    _mma_m16n8k8_bf16_zero(o_acc, dc_bf16, inv_a_frag)  # .cu:2303-2305
                    with K.unroll(2) as _lp:  # .cu:2306-2310
                        K.ptx.mov.b32(
                            o_bf16[_lp],
                            K.cuda.float22bfloat162_rn(o_acc[_lp * 2 + 0], o_acc[_lp * 2 + 1 + 0]),
                        )
                    K.ptx.add.s32(matrix_addr, matrix_addr, K.int32(1024))
                    # .cu:2314-2317 stmatrix.x1
                    K.ptx.stmatrix.sync.aligned.m8n8.x1.shared.b16(
                        smem_raw.ptr_to([(matrix_addr) - K.cast(smem, "int32")]), o_bf16[0]
                    )
                    with K.If(K.cuda.elect_sync()), K.Then():  # .cu:2318-2320
                        _mbarrier_arrive(prep_inv16_ready, prep_stage)
                    _mbarrier_wait(prep_inv16_ready, prep_stage, prep_phase.phase)  # .cu:2321
                with K.If(prep_local_warp == 0):
                    with K.Then():  # .cu:2323-2404
                        lane_row_1: K.int32 = lane % 16  # .cu:2324
                        lane_col: K.int32 = lane // 16 * 8  # .cu:2325
                        byte_off_3: K.int32 = (16 + lane_row_1) * 128 + (
                            16 + lane_col
                        ) * 2  # .cu:2326
                        swizzled_off_4: K.int32 = byte_off_3 ^ (
                            (byte_off_3 >> 7 & 7) << 4
                        )  # .cu:2327
                        d_addr_1: K.int32 = (
                            smem_inv_work_addr + prep_stage_byte_base + swizzled_off_4
                        )  # .cu:2328
                        byte_off_0_1: K.int32 = (16 + lane_row_1) * 128 + lane_col * 2  # .cu:2329
                        swizzled_off_1_2: K.int32 = byte_off_0_1 ^ (
                            (byte_off_0_1 >> 7 & 7) << 4
                        )  # .cu:2330
                        c_addr_1: K.int32 = (
                            smem_inv_work_addr + prep_stage_byte_base + swizzled_off_1_2
                        )  # .cu:2331
                        byte_off_2_2: K.int32 = lane_row_1 * 128 + lane_col * 2  # .cu:2332
                        swizzled_off_3_2: K.int32 = byte_off_2_2 ^ (
                            (byte_off_2_2 >> 7 & 7) << 4
                        )  # .cu:2333
                        a_addr_1: K.int32 = (
                            smem_inv_work_addr + prep_stage_byte_base + swizzled_off_3_2
                        )  # .cu:2334
                        d32_frag = K.alloc_local((4,), "uint32", align=4)  # .cu:2335
                        c32_frag = K.alloc_local((4,), "uint32", align=4)  # .cu:2336
                        dc32_acc = K.alloc_local((8,), "float32", align=4)  # .cu:2337
                        dc32_bf16 = K.alloc_local((4,), "uint32", align=4)  # .cu:2338
                        a32_frag = K.alloc_local((4,), "uint32", align=4)  # .cu:2339
                        o32_acc = K.alloc_local((8,), "float32", align=4)  # .cu:2340
                        o32_bf16 = K.alloc_local((4,), "uint32", align=4)  # .cu:2341
                        zero32_bf16 = K.alloc_local((4,), "uint32", align=4)  # .cu:2342
                        # .cu:2343-2346 ldmatrix.x4 d32
                        K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                            d32_frag[0],
                            d32_frag[1],
                            d32_frag[2],
                            d32_frag[3],
                            smem_raw.ptr_to([(d_addr_1) - K.cast(smem, "int32")]),
                        )
                        _dpb: K.int32 = (
                            (16 + lane_col) // 16 * 1024
                            + (16 + lane_row_1) * 32
                            + (16 + lane_col) % 16 * 2
                        )
                        d_publish_addr: K.int32 = (
                            smem_inv_addr + prep_stage_byte_base + (_dpb ^ ((_dpb >> 7 & 1) << 4))
                        )  # .cu:2347
                        # .cu:2348-2351 stmatrix.x4 d32 publish
                        K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                            smem_raw.ptr_to([(d_publish_addr) - K.cast(smem, "int32")]),
                            d32_frag[0],
                            d32_frag[1],
                            d32_frag[2],
                            d32_frag[3],
                        )
                        # .cu:2352-2355 ldmatrix.x4.trans c32
                        K.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                            c32_frag[0],
                            c32_frag[1],
                            c32_frag[2],
                            c32_frag[3],
                            smem_raw.ptr_to([(c_addr_1) - K.cast(smem, "int32")]),
                        )
                        _mma_m16n8k16_bf16_zero(dc32_acc, d32_frag, c32_frag)  # .cu:2356-2358
                        _mma_m16n8k16_bf16_zero_off4(dc32_acc, d32_frag, c32_frag)  # .cu:2359-2361
                        with K.unroll(4) as _ls:  # .cu:2362-2365
                            _pk = K.local_scalar("uint64")
                            K.ptx.mul.rn.ftz.f32x2(
                                _pk,
                                K.cuda.make_float2(dc32_acc[_ls * 2], dc32_acc[_ls * 2 + 1]),
                                K.cuda.make_float2(K.float32(-1.0), K.float32(-1.0)),
                            )
                            K.ptx.mov.b32(dc32_acc[_ls * 2], K.cuda.float2_x(_pk))
                            K.ptx.mov.b32(dc32_acc[_ls * 2 + 1], K.cuda.float2_y(_pk))
                        with K.unroll(4) as _lp:  # .cu:2366-2370
                            K.ptx.mov.b32(
                                dc32_bf16[_lp],
                                K.cuda.float22bfloat162_rn(
                                    dc32_acc[_lp * 2 + 0], dc32_acc[_lp * 2 + 1 + 0]
                                ),
                            )
                        # .cu:2371-2374 ldmatrix.x4.trans a32
                        K.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                            a32_frag[0],
                            a32_frag[1],
                            a32_frag[2],
                            a32_frag[3],
                            smem_raw.ptr_to([(a_addr_1) - K.cast(smem, "int32")]),
                        )
                        _apb: K.int32 = lane_col // 16 * 1024 + lane_row_1 * 32 + lane_col % 16 * 2
                        a_publish_addr: K.int32 = (
                            smem_inv_addr + prep_stage_byte_base + (_apb ^ ((_apb >> 7 & 1) << 4))
                        )  # .cu:2375
                        # .cu:2376-2379 stmatrix.x4.trans a32 publish
                        K.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                            smem_raw.ptr_to([(a_publish_addr) - K.cast(smem, "int32")]),
                            a32_frag[0],
                            a32_frag[1],
                            a32_frag[2],
                            a32_frag[3],
                        )
                        _mma_m16n8k16_bf16_zero(o32_acc, dc32_bf16, a32_frag)  # .cu:2380-2382
                        _mma_m16n8k16_bf16_zero_off4(o32_acc, dc32_bf16, a32_frag)  # .cu:2383-2385
                        with K.unroll(4) as _lp:  # .cu:2386-2390
                            K.ptx.mov.b32(
                                o32_bf16[_lp],
                                K.cuda.float22bfloat162_rn(
                                    o32_acc[_lp * 2 + 0], o32_acc[_lp * 2 + 1 + 0]
                                ),
                            )
                        _opb: K.int32 = (
                            lane_col // 16 * 1024 + (16 + lane_row_1) * 32 + lane_col % 16 * 2
                        )
                        o_publish_addr: K.int32 = (
                            smem_inv_addr + prep_stage_byte_base + (_opb ^ ((_opb >> 7 & 1) << 4))
                        )  # .cu:2391
                        # .cu:2392-2395 stmatrix.x4 o32 publish
                        K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                            smem_raw.ptr_to([(o_publish_addr) - K.cast(smem, "int32")]),
                            o32_bf16[0],
                            o32_bf16[1],
                            o32_bf16[2],
                            o32_bf16[3],
                        )
                        with K.unroll(4) as zero_word:  # .cu:2396-2399
                            K.ptx.mov.b32(zero32_bf16[zero_word], K.uint32(0))
                        _zpb: K.int32 = (
                            (16 + lane_col) // 16 * 1024
                            + lane_row_1 * 32
                            + (16 + lane_col) % 16 * 2
                        )
                        zero_publish_addr: K.int32 = (
                            smem_inv_addr + prep_stage_byte_base + (_zpb ^ ((_zpb >> 7 & 1) << 4))
                        )  # .cu:2400
                        # .cu:2401-2404 stmatrix.x4 zero publish
                        K.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                            smem_raw.ptr_to([(zero_publish_addr) - K.cast(smem, "int32")]),
                            zero32_bf16[0],
                            zero32_bf16[1],
                            zero32_bf16[2],
                            zero32_bf16[3],
                        )
                    with K.Else():
                        with K.If(prep_local_warp == 1), K.Then():  # .cu:2405-2522
                            _restore_block(4, lambda _pass: _pass * 2 + (lane >> 4))  # .cu:2417
                _fence_async_shared()  # .cu:2523
                _prep_bar_sync()  # .cu:2524-2536
                with K.If(prep_local_warp == 0), K.Then():  # .cu:2537-2541
                    with K.If(K.cuda.elect_sync()), K.Then():
                        _mbarrier_arrive(qk_full, prep_stage)
                prep_phase.advance()

        with K.If((warp >= 8) & (warp <= 11)), K.Then():
            producer_regs.emit()
        with compute:
            compute_state = K.PipelineState(5, phase=0)
            compute_role(compute_state)
        with epilogue:
            epilogue_state = K.PipelineState(5, phase=0)
            output_state = K.PipelineState(1, phase=0)
            epilogue_role(epilogue_state, output_state)
        with mma:
            mma_state = K.PipelineState(5, phase=0)
            out_state = K.PipelineState(1, phase=0)
            mma_role(mma_state, out_state)
        with load:
            load_state = K.PipelineState(5, phase=0)
            load_role(load_state)
        with prep_role_owner:
            prep_phase = K.PipelineState(1, phase=0)
            prep_role(prep_phase)

    kernel = _kernel.func
    return kernel.with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS)).with_attr(
        "global_symbol", "bf16_fused_m128"
    )


def get_kernel(**kwargs: Any):
    return bf16_fused_m128(**kwargs)


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(**kwargs: Any) -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for FlashKDA bf16 fused m128")

    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    cfg: FlashKDABf16FusedM128Config = case["config"]
    if not case["dispatch_reason"].startswith("m128:"):
        raise SkipTest(case["dispatch_reason"])
    compile_kernel(bf16_fused_m128(**kwargs))(*_tirx_args(case))
    torch.cuda.synchronize()

    flashinfer_out, flashinfer_state = _flashinfer_cuda_reference(case)
    torch.testing.assert_close(case["out"], flashinfer_out, rtol=4.01 / 128, atol=5e-3)
    if cfg.store_final_state:
        # The state comparison must not silently vanish when the reference
        # declines to return one.
        if flashinfer_state is None:
            raise AssertionError("store_final_state set but the reference returned no state")
        torch.testing.assert_close(
            case["final_state"],
            flashinfer_state.reshape(case["final_state"].shape),
            rtol=4.01 / 128,
            atol=5e-3,
        )
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
        raise SkipTest("CUDA is required for FlashKDA bf16 fused m128 benchmark")

    from tirx_kernels.runner import bench

    case = prepare_data(**kwargs)
    cfg: FlashKDABf16FusedM128Config = case["config"]
    if not case["dispatch_reason"].startswith("m128:"):
        raise SkipTest(case["dispatch_reason"])
    args = _tirx_args(case)
    ex = executable
    funcs = {"tirx": lambda: ex(*args)}

    # Produce the expected buffers once.  The FlashKDA peer builder validates
    # against these outside the timed region before returning its pure launch.
    for func in funcs.values():
        func()
    torch.cuda.synchronize()

    def _flashinfer_builder():
        _load_flashinfer_recurrent_kda()
        flashinfer_case = dict(case)
        if cfg.use_initial_state:
            flashinfer_case["initial_state"] = case["initial_state"].clone()
        return lambda: _flashinfer_cuda_reference(flashinfer_case)

    flashkda_peer: dict[str, Any] = {}

    def _flashkda_raw_builder():
        from tirx_kernels.flashinfer.utils._flashkda_bench import prepare_flashkda_raw_reference

        peer = prepare_flashkda_raw_reference(case)
        flashkda_peer["reference"] = peer
        return peer.launch

    references = {"flashinfer_m128": _flashinfer_builder, "flashkda_raw": _flashkda_raw_builder}

    result = bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references=references,
        rounds=_rounds,
        cooldown_s=_cooldown_s,
    )
    peer = flashkda_peer.get("reference")
    if peer is not None:
        result["flashkda_raw_provenance"] = peer.provenance
        result["flashkda_raw_correctness"] = peer.correctness
    return result


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    config = dict(kwargs)
    protocol = {name: config.pop(name) for name in ("rounds", "cooldown_s") if name in config}
    prepared = prepare_bench(**config)
    return prepared.run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "bf16_fused_m128",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
