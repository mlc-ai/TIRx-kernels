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

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any
from unittest import SkipTest

import torch

import tvm
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as T

D_HEAD = 128
THREADS = 1024
SMEM_TOTAL = 227328
DEFAULT_LOWER_BOUND = -5.0

# .cu:29-106 (#define block, source order).
TMEM_TMEM_STATE_OFFSET = 64
TMEM_TMEM_STATE_INP_OFFSET = 0
TMEM_TMEM_U_ACC_OFFSET = 224
TMEM_TMEM_U2_INP_OFFSET = 224
TMEM_TMEM_U2_ACC_OFFSET = 0
TMEM_TMEM_OUT_OFFSET = 192
TMEM_TMEM_STATE_OUT_OFFSET = 64
SMEM_SMEM_QD_OFF = 1024
SMEM_SMEM_G_RAW_OFF = 1024
SMEM_SMEM_KD_OFF = 9216
SMEM_SMEM_Q_RAW_PREFETCH_OFF = 17408
SMEM_SMEM_FINAL_TRANS_OFF = 17408
SMEM_SMEM_KR_TRANS_OFF = 17408
SMEM_SMEM_MQK_TRANS_OFF = 25600
SMEM_SMEM_INV_OFF = 29696
SMEM_SMEM_V_OFF = 32384
SMEM_SMEM_KI_OFF = 17408
SMEM_SMEM_GATE_OFF = 25600
SMEM_SMEM_BETA_RAW_OFF = 41984
SMEM_SMEM_INV_WORK_OFF = 32384
SMEM_SMEM_OUT_OFF = 210944
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

# Mbarrier group byte offsets within the smem barrier area (.cu:695-711).
MBAR_QK_FULL_OFF = 0
MBAR_GATE_RAW_FULL_OFF = 40
MBAR_QK_RAW_FULL_OFF = 80
MBAR_V_FULL_OFF = 120
MBAR_V_FREE_OFF = 160
MBAR_SMEM_FREE_OFF = 200
MBAR_RAW_INPUTS_FREE_OFF = 240
MBAR_STATE_INP_READY_OFF = 280
MBAR_OLD_OUT_READY_OFF = 320
MBAR_U_INP_READY_OFF = 360
MBAR_U2_ACC_READY_OFF = 400
MBAR_U2_INP_READY_OFF = 440
MBAR_FINAL_READY_OFF = 480
MBAR_OUT_EMPTY_OFF = 520
MBAR_TMEM_DEALLOC_READY_OFF = 528
MBAR_PREP_DIAG_READY_OFF = 536
MBAR_PREP_INV16_READY_OFF = 576
SMEM_TMEM_ADDR_STORAGE_OFF = 616

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
# Helpers use the exact T.ptx/T.cuda wrappers available in the target TVM.
# ---------------------------------------------------------------------------

# tcgen05.mma spelling: kind::f16 from the (float32, bfloat16, bfloat16) dtypes; the A
# operand is a uint32 TMEM address (use_a_tmem), which selects the ts form.
_TCGEN05_MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
# cta_group::1 disable-output-lane mask group: the explicit default {0, 0, 0, 0}.
_MMA_ZERO_MASKS = (0, 0, 0, 0)
# Warp-level mma.sync zero-C operand group (the four "f"(0.f) inline-asm literals).
_MMA_ZERO_C = (T.float32(0.0), T.float32(0.0), T.float32(0.0), T.float32(0.0))
# TMA spellings: unicast g2s at CTA scope (the sm_100 wrapper emits the explicit default
# .cta_group::1 suffix for the unqualified inline instruction) and plain tile s2g.
_TMA_G2S_CTA = (
    "cp.async.bulk.tensor.{dim}d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1"
)
_TMA_S2G = "cp.async.bulk.tensor.{dim}d.global.shared::cta.tile.bulk_group"


def _mma_qk_8step(taddr_d, b_lo, taddr_a, enable_d):
    # .cu:1114-1150 and .cu:1153-1189 (same verbatim block, two call sites) ->
    # 8 native tcgen05.mma issues: b_lo offsets 0,2,4,6,256,258,260,262 (adds 2,2,2,250,2,2,2),
    # taddr_a +8 per step, dhi=0x40004040, idesc=134743184, step0 enable_d / steps1-7 tied 1,
    # all @leader-predicated. Native emits the explicit default disable-output-lane {0,0,0,0}.
    leader = T.cuda.elect_sync()
    b_lo32 = T.cast(b_lo, "uint32")
    for _i in range(8):
        _b_desc = (T.uint64(0x40004040) << T.uint64(32)) | T.cast(
            b_lo32 + T.uint32((0, 2, 4, 6, 256, 258, 260, 262)[_i]), "uint64"
        )
        T.evaluate(
            T.ptx[_TCGEN05_MMA_F16](
                T.cast(taddr_d, "uint32"),
                T.cast(taddr_a + 8 * _i, "uint32"),
                _b_desc,
                T.uint32(134743184),
                *_MMA_ZERO_MASKS,
                T.ptx.pred(enable_d if _i == 0 else 1),
                pred=leader,
            )
        )


def _mma_inv_2step(taddr_d, b_lo, taddr_a, enable_d):
    # .cu:1194-1212 -> 2 native tcgen05.mma issues (b_lo add 64; dhi=0xC0004010)
    leader = T.cuda.elect_sync()
    b_lo32 = T.cast(b_lo, "uint32")
    for _i in range(2):
        _b_desc = (T.uint64(0xC0004010) << T.uint64(32)) | T.cast(
            b_lo32 + T.uint32(64 * _i), "uint64"
        )
        T.evaluate(
            T.ptx[_TCGEN05_MMA_F16](
                T.cast(taddr_d, "uint32"),
                T.cast(taddr_a + 8 * _i, "uint32"),
                _b_desc,
                T.uint32(134743184),
                *_MMA_ZERO_MASKS,
                T.ptx.pred(enable_d if _i == 0 else 1),
                pred=leader,
            )
        )


def _mma_final_2step(taddr_d, b_lo, taddr_a, enable_d):
    # .cu:1217-1235 -> 2 native tcgen05.mma issues (b_lo add 128; dhi=0x40004040)
    leader = T.cuda.elect_sync()
    b_lo32 = T.cast(b_lo, "uint32")
    for _i in range(2):
        _b_desc = (T.uint64(0x40004040) << T.uint64(32)) | T.cast(
            b_lo32 + T.uint32(128 * _i), "uint64"
        )
        T.evaluate(
            T.ptx[_TCGEN05_MMA_F16](
                T.cast(taddr_d, "uint32"),
                T.cast(taddr_a + 8 * _i, "uint32"),
                _b_desc,
                T.uint32(136905872),
                *_MMA_ZERO_MASKS,
                T.ptx.pred(enable_d if _i == 0 else 1),
                pred=leader,
            )
        )


def _ld_global_v4_u32(dst, ptr):
    # verbatim plain uint4 load (.cu:783-806/1564-1601 pattern; plain ld.global.v4, NOT .nc)
    return T.ptx.ld.global_.v4.b32(dst[0], dst[1], dst[2], dst[3], ptr)


def _ld_global_s32(buffer, index):
    out = T.alloc_local((1,), "int32")
    T.evaluate(T.ptx.ld.global_.s32(out[0], buffer.ptr_to([index])))
    return out[0]


def _ld_global_s64(buffer, index):
    out = T.alloc_local((1,), "int64")
    T.evaluate(T.ptx.ld.global_.s64(out[0], buffer.ptr_to([index])))
    return out[0]


def _ld_global_f32(buffer, index):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.ld.global_.f32(out[0], buffer.ptr_to([index])))
    return out[0]


def _ld_global_bf16(buffer, index):
    bits = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.ld.global_.b16(bits[0], buffer.ptr_to([index])))
    return T.reinterpret("bfloat16", bits[0])


def _st_global_bf16(buffer, index, value):
    return T.ptx.st.global_.b16(buffer.ptr_to([index]), T.reinterpret("uint16", value))


def _ld_shared_f32(buffer, index):
    out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.ld.shared.f32(out[0], buffer.ptr_to([index])))
    return out[0]


def _ld_shared_v4_f32(dst, dst_offset, buffer, index):
    return T.ptx.ld.shared.v4.f32(
        dst[dst_offset],
        dst[dst_offset + 1],
        dst[dst_offset + 2],
        dst[dst_offset + 3],
        buffer.ptr_to([index]),
    )


def _ld_shared_bf16(buffer, index):
    bits = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.ld.shared.b16(bits[0], buffer.ptr_to([index])))
    return T.reinterpret("bfloat16", bits[0])


def _st_shared_f32(buffer, index, value):
    return T.ptx.st.shared.f32(buffer.ptr_to([index]), value)


def _tanh_approx(x):
    # tanh.approx.f32 (prep-role sigmoid)
    result = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.tanh.approx.f32(result[0], x))
    return result[0]


def _expf(x):
    # .cu:1317 __expf -> ex2.approx.ftz.f32(x * log2(e)) under -use_fast_math
    _out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.ex2.approx.ftz.f32(_out[0], x * T.float32(1.4426950408889634)))
    return _out[0]


def _rsqrtf(x):
    # CUDA's fast-math rsqrtf lowers to this exact PTX form.
    result = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.rsqrt.approx.ftz.f32(result[0], x))
    return result[0]


def _fmaf_rn(a, b, c):
    result = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.fma.rn.f32(result[0], a, b, c))
    return result[0]


def _tensormap_acquire(tmap):
    return T.ptx.fence.proxy.tensormap__generic.acquire.gpu(tmap)


def _ld_shared_b32(smem_raw, smem, addr):
    # ld.shared.b32
    _out = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.ld.shared.b32(_out[0], smem_raw.ptr_to([addr - smem])))
    return _out[0]


def _ld_shared_v4(smem_raw, smem, dst, addr):
    # ld.shared.v4.b32 (dst: 4-elem uint32 local array)
    return T.ptx.ld.shared.v4.b32(dst[0], dst[1], dst[2], dst[3], smem_raw.ptr_to([addr - smem]))


def _st_shared_b32(smem_raw, smem, addr, val):
    # st.shared.b32
    return T.ptx.st.shared.b32(smem_raw.ptr_to([addr - smem]), val)


def _mma_m16n8k16_bf16_zero(acc, a, b):
    # mma.sync m16n8k16 bf16, zero C (acc/a/b: local arrays) -> native register-fragment
    # form; the explicit zero C slots feed "f"(0.f) == inline's 0f00000000 literals
    return T.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
        *[acc[i] for i in range(4)],
        *[a[i] for i in range(4)],
        *[b[i] for i in range(2)],
        *_MMA_ZERO_C,
    )


def _mma_m16n8k16_bf16_acc(acc, a, b):
    # mma.sync m16n8k16 bf16, accumulating C (C aliases acc == inline "+f" tied regs)
    return T.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
        *[acc[i] for i in range(4)],
        *[a[i] for i in range(4)],
        *[b[i] for i in range(2)],
        *[acc[i] for i in range(4)],
    )


def _mma_m16n8k8_bf16_zero(acc, a, b):
    # mma.sync m16n8k8 bf16, zero C -> native register-fragment form
    return T.ptx.mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32(
        *[acc[i] for i in range(4)],
        *[a[i] for i in range(2)],
        *[b[i] for i in range(1)],
        *_MMA_ZERO_C,
    )


def _mma_m16n8k16_bf16_zero_off4(acc, a, b):
    # second zero-C mma writing acc[4..7] with b_frag[2..3] (e.g. .cu:1725-1727)
    return T.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
        *[acc[4 + i] for i in range(4)],
        *[a[i] for i in range(4)],
        *[b[2 + i] for i in range(2)],
        *_MMA_ZERO_C,
    )


def _mma_m16n8k16_bf16_acc_off4(acc, a, b):
    # second accumulating mma writing acc[4..7] with b_frag[2..3]
    return T.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
        *[acc[4 + i] for i in range(4)],
        *[a[i] for i in range(4)],
        *[b[2 + i] for i in range(2)],
        *[acc[4 + i] for i in range(4)],
    )


def _st_global_v4_u32(ptr, a, b, c, d):
    # verbatim uint4 store (STG.128 shape) used by the final-state store path
    return T.ptx.st.global_.v4.b32(ptr, a, b, c, d)


def _mbarrier_wait(smem_raw, smem, mbar_addr, phase):
    # .cu:158-171 -> native T.cuda.mbarrier_wait: same LAB_WAIT spin loop with
    # ticks=0x989680; mbarrier.try_wait.parity.shared::cta defaults to .acquire.cta
    # (the token/cluster wait variants at .cu:130-156,173-198 had no call sites)
    return T.cuda.mbarrier_wait(smem_raw.ptr_to([mbar_addr - smem]), phase)


def _elect_commit(mbar_addr):
    # .cu:239-248
    leader = T.cuda.elect_sync()
    return T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
        mbar_addr, pred=leader
    )


def _mbarrier_arrive(smem_raw, smem, mbar_addr):
    # .cu:251-255 -> native; emits the same mbarrier.arrive.release.cta.shared::cta.b64 _, [addr]
    return T.ptx.mbarrier.arrive.release.cta.shared__cta.b64(
        smem_raw.ptr_to([mbar_addr - smem]), T.uint32(1)
    )


def _mbarrier_arrive_expect_tx(mbar_addr, bytes_):
    # .cu:258-262
    return T.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(mbar_addr, T.uint32(bytes_))


def _tmem_ld_x32(dst, tmem_addr):
    # .cu:265-281
    return T.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
        *[dst[j] for j in range(32)], T.cast(tmem_addr, "uint32")
    )


def _tmem_st_x32_f32(tmem_addr, src):
    # .cu:297-313
    return T.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
        T.cast(tmem_addr, "uint32"), *[src[j] for j in range(32)]
    )


def _approx_exp2(x):
    # .cu:326-330
    _out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.ex2.approx.ftz.f32(_out[0], x))
    return _out[0]


def _mul_f32x2_inplace(a, b):
    # .cu:342-345
    _out = T.alloc_local((1,), "uint64")
    T.evaluate(T.ptx.mul.rn.ftz.f32x2(_out[0], a, b))
    return _out[0]


def _sub_f32x2_inplace(a, b):
    # .cu:352-355
    _out = T.alloc_local((1,), "uint64")
    T.evaluate(T.ptx.sub.rn.ftz.f32x2(_out[0], a, b))
    return _out[0]


def _fence_async_shared():
    # .cu:416-418
    return T.ptx.fence.proxy.async_.shared__cta()


def _tma_3d_gmem2smem(smem_raw, smem, dst, tmap_ptr, x, y, z, mbar_addr):
    # .cu:430-438 (raw tensor-map pointer form) -> native; on sm_100 the wrapper emits
    # the explicit default .cta_group::1 suffix for the unqualified inline instruction
    return T.ptx[_TMA_G2S_CTA.format(dim=3)](
        smem_raw.ptr_to([dst - smem]),
        T.reinterpret("uint64", tmap_ptr),
        x,
        y,
        z,
        smem_raw.ptr_to([mbar_addr - smem]),
    )


def _tma_2d_gmem2smem(smem_raw, smem, dst, tmap_ptr, x, y, mbar_addr):
    # .cu:441-449 -> native (see _tma_3d_gmem2smem note)
    return T.ptx[_TMA_G2S_CTA.format(dim=2)](
        smem_raw.ptr_to([dst - smem]),
        T.reinterpret("uint64", tmap_ptr),
        x,
        y,
        smem_raw.ptr_to([mbar_addr - smem]),
    )


def _tma_4d_gmem2smem(smem_raw, smem, dst, tmap_ptr, x, y, z, w, mbar_addr):
    # .cu:452-460 -> native (see _tma_3d_gmem2smem note)
    return T.ptx[_TMA_G2S_CTA.format(dim=4)](
        smem_raw.ptr_to([dst - smem]),
        T.reinterpret("uint64", tmap_ptr),
        x,
        y,
        z,
        w,
        smem_raw.ptr_to([mbar_addr - smem]),
    )


def _tma_store_4d(smem_raw, smem, tmap, x, y, z, w, smem_addr):
    # .cu:463-469 -> native s2g (cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group)
    return T.ptx[_TMA_S2G.format(dim=4)](
        T.reinterpret("uint64", tmap), x, y, z, w, smem_raw.ptr_to([smem_addr - smem])
    )


def _tmem_st_x8_u32(addr, src):
    # .cu:480-487
    return T.ptx["tcgen05.st.sync.aligned.32x32b.x8.b32"](
        T.cast(addr, "uint32"), *[src[j] for j in range(8)]
    )


def _make_warp_uniform(val):
    # .cu:490-495
    return T.cuda._shfl_sync(T.uint32(0xFFFFFFFF), val, 0, 32)


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


def _builder_name(name: str, value):
    """Name a directly constructed builder value and return it."""
    try:
        return IRBuilder.name(name, value)
    except (TypeError, ValueError):
        return value


def _builder_meta(name: str, value):
    """Name resources owned by an IR-builder meta-class instance."""
    from tvm.tirx.script.builder.ir import name_meta_class_value

    name_meta_class_value(name, value)
    return value


def _builder_scalar(name: str, value, dtype: str | None = None):
    """Materialize the mutable scalar semantics used by the former parser."""
    value_type = getattr(value, "ty", None)
    if value_type is not None and not isinstance(value_type, tvm.ir.PrimType):
        return _builder_bind(name, value, value.ty)
    if dtype is None:
        dtype = str(value.ty.dtype)
    scalar = T.alloc_scalar(dtype=dtype, scope="local")
    IRBuilder.name(name, scalar.scalar.buffer)
    T.buffer_store(scalar.scalar.buffer, value, [0])
    return scalar.scalar


def _builder_alloc_scalar(name: str, dtype: str):
    """Allocate a mutable scalar without inventing an initializer."""
    scalar = T.alloc_scalar(dtype=dtype, scope="local")
    IRBuilder.name(name, scalar.scalar.buffer)
    return scalar.scalar


def _builder_bind(name: str, value, type_annotation=None):
    """Emit and name an immutable builder Bind."""
    result = T.Bind(value, type_annotation)
    IRBuilder.name(name, result)
    return result


def _builder_enter(frame):
    """Enter a flat builder frame until its enclosing PrimFunc completes."""
    frame.add_callback(lambda: frame.__exit__(None, None, None))
    frame.__enter__()


def _builder_scope_enter(frame):
    """Enter a builder frame without adding Python source nesting."""
    frame.__enter__()
    return frame


def _builder_scope_exit(frame):
    """Exit a frame entered by :func:`_builder_scope_enter`."""
    frame.__exit__(None, None, None)


def _builder_emit(value):
    """Match TVMScript expression-statement emission in direct builder code."""
    if value is None or isinstance(value, tvm.ir.Var):
        return
    if tvm.ir.is_prim_expr(value) or isinstance(value, tvm.ir.Call):
        T.evaluate(value)


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


def _reference_torch(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-serial reference, mirrors tests/kda/test_recurrent_kda_prefill.py::_reference.

    O(total_tokens) python loop; intended for the small correctness configs.
    """
    import torch.nn.functional as F

    cfg: FlashKDABf16FusedM128Config = case["config"]
    scale = case["scale"]
    q_flat = F.normalize(case["q"].float(), dim=-1)
    k_flat = F.normalize(case["k"].float(), dim=-1)
    v_flat = case["v"].float()
    g_flat = case["g"].float()
    beta_flat = torch.sigmoid(case["beta"].float())
    gate = cfg.lower_bound * torch.sigmoid(
        torch.exp(case["A_log"]).reshape(1, cfg.num_heads, 1)
        * (g_flat + case["dt_bias"].reshape(1, cfg.num_heads, D_HEAD))
    )
    decay = torch.exp(gate)
    if cfg.use_initial_state:
        state = case["initial_state"].clone()
    else:
        state = torch.zeros(
            (cfg.num_seqs, cfg.num_heads, D_HEAD, D_HEAD),
            dtype=torch.bfloat16,
            device=case["q"].device,
        )
    out = torch.empty_like(q_flat)
    offsets = [0, *torch.cumsum(torch.tensor(cfg.seq_lens), dim=0).tolist()]
    for sequence in range(cfg.num_seqs):
        for token in range(offsets[sequence], offsets[sequence + 1]):
            state_f32 = state[sequence].float()
            decayed = state_f32 * decay[token].unsqueeze(1)
            predicted = torch.einsum("hk,hvk->hv", k_flat[token], decayed)
            residual = beta_flat[token].unsqueeze(-1) * (v_flat[token] - predicted)
            updated = decayed + residual.unsqueeze(-1) * k_flat[token].unsqueeze(1)
            state[sequence] = updated.to(torch.bfloat16)
            projected = torch.einsum("hk,hvk->hv", q_flat[token], state[sequence].float())
            out[token] = (scale * projected).to(torch.bfloat16)
    return out.to(torch.bfloat16), state


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


def _build_kernel(
    *,
    total_tokens: T.constexpr,
    h: T.constexpr,
    num_seqs: T.constexpr,
    beta_tma_tokens: T.constexpr,
    beta_tma_heads: T.constexpr,
    scale: T.constexpr,
    lower_bound: T.constexpr,
    use_initial_state: T.constexpr,
    store_final_state: T.constexpr,
):
    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("_kernel")
            q = T.arg("q", T.Buffer((total_tokens * h * D_HEAD,), "bfloat16"))
            k = T.arg("k", T.Buffer((total_tokens * h * D_HEAD,), "bfloat16"))
            v = T.arg("v", T.Buffer((total_tokens * h * D_HEAD,), "bfloat16"))
            g = T.arg("g", T.Buffer((total_tokens * h * D_HEAD,), "bfloat16"))
            beta = T.arg("beta", T.Buffer((total_tokens * h,), "bfloat16"))
            beta_tma = T.arg("beta_tma", T.Buffer((beta_tma_tokens, beta_tma_heads), "bfloat16"))
            A_log = T.arg("A_log", T.Buffer((h,), "float32"))
            dt_bias = T.arg("dt_bias", T.Buffer((h * D_HEAD,), "float32"))
            cu_seqlens = T.arg("cu_seqlens", T.Buffer((num_seqs + 1,), "int64"))
            seq_order = T.arg("seq_order", T.Buffer((num_seqs,), "int32"))
            initial_state = T.arg(
                "initial_state", T.Buffer((num_seqs * h * D_HEAD * D_HEAD,), "bfloat16")
            )
            out = T.arg("out", T.Buffer((total_tokens * h * D_HEAD,), "bfloat16"))
            final_state = T.arg(
                "final_state", T.Buffer((num_seqs * h * D_HEAD * D_HEAD,), "bfloat16")
            )
            descriptor_storage = T.arg("descriptor_storage", T.Buffer((768,), "uint8"))
            _builder_emit(T.device_entry())
            _builder_enter(T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1}))
            block_idx = _builder_name("block_idx", T.cta_id([num_seqs * h]))
            thread_idx = _builder_name("thread_idx", T.thread_id([THREADS]))
            _builder_emit(T.warpgroup_id([THREADS // 128]))
            _builder_emit(T.warp_id_in_wg([4]))
            _builder_emit(T.lane_id([32]))
            _builder_emit(T.thread_id_in_wg([128]))
            pool = _builder_meta("pool", T.SMEMPool())
            smem_raw = _builder_name("smem_raw", pool.alloc((SMEM_TOTAL,), "uint8", align=1024))
            tmem_addr_storage = _builder_name(
                "tmem_addr_storage",
                T.decl_buffer(
                    (1,),
                    "int32",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_TMEM_ADDR_STORAGE_OFF,
                    align=4,
                ),
            )
            smem_g_raw_all = _builder_name(
                "smem_g_raw_all",
                T.decl_buffer(
                    (SMEM_SMEM_G_RAW_ALL_STAGE_BYTES // 2,),
                    "bfloat16",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_SMEM_G_RAW_ALL_OFF,
                    align=1024,
                ),
            )
            smem_v_all = _builder_name(
                "smem_v_all",
                T.decl_buffer(
                    (SMEM_SMEM_V_ALL_STAGE_BYTES // 2,),
                    "bfloat16",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_SMEM_V_ALL_OFF,
                    align=1024,
                ),
            )
            smem_gate_all = _builder_name(
                "smem_gate_all",
                T.decl_buffer(
                    (SMEM_SMEM_GATE_ALL_STAGE_BYTES // 4,),
                    "float32",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_SMEM_GATE_ALL_OFF,
                    align=1024,
                ),
            )
            smem_gt_all = _builder_name(
                "smem_gt_all",
                T.decl_buffer(
                    (SMEM_SMEM_GT_ALL_STAGE_BYTES // 4,),
                    "float32",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_SMEM_GT_ALL_OFF,
                    align=1024,
                ),
            )
            smem_gt_prefix_all = _builder_name(
                "smem_gt_prefix_all",
                T.decl_buffer(
                    (SMEM_SMEM_GT_PREFIX_ALL_STAGE_BYTES // 4,),
                    "float32",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_SMEM_GT_PREFIX_ALL_OFF,
                    align=1024,
                ),
            )
            smem_restore_factor_all = _builder_name(
                "smem_restore_factor_all",
                T.decl_buffer(
                    (SMEM_SMEM_RESTORE_FACTOR_ALL_STAGE_BYTES // 4,),
                    "float32",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_SMEM_RESTORE_FACTOR_ALL_OFF,
                    align=1024,
                ),
            )
            smem_prep_beta_all = _builder_name(
                "smem_prep_beta_all",
                T.decl_buffer(
                    (SMEM_SMEM_PREP_BETA_ALL_STAGE_BYTES // 4,),
                    "float32",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_SMEM_PREP_BETA_ALL_OFF,
                    align=1024,
                ),
            )
            smem_gate_rate_all = _builder_name(
                "smem_gate_rate_all",
                T.decl_buffer(
                    (SMEM_SMEM_GATE_RATE_ALL_STAGE_BYTES // 4,),
                    "float32",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_SMEM_GATE_RATE_ALL_OFF,
                    align=1024,
                ),
            )
            _builder_emit(pool.commit())
            q_tma = _builder_scalar("q_tma", descriptor_storage.ptr_to([TMA_SLOT_Q]))
            k_tma = _builder_scalar("k_tma", descriptor_storage.ptr_to([TMA_SLOT_K]))
            v_tma = _builder_scalar("v_tma", descriptor_storage.ptr_to([TMA_SLOT_V]))
            g_tma = _builder_scalar("g_tma", descriptor_storage.ptr_to([TMA_SLOT_G]))
            beta_tma_tmap = _builder_scalar(
                "beta_tma_tmap", descriptor_storage.ptr_to([TMA_SLOT_BETA])
            )
            out_tma = _builder_scalar("out_tma", descriptor_storage.ptr_to([TMA_SLOT_OUT]))
            _builder_if_124_4 = _builder_scope_enter(T.If(thread_idx == 0))
            _builder_then_124_4 = _builder_scope_enter(T.Then())
            _builder_emit(_tensormap_acquire(q_tma))
            _builder_emit(_tensormap_acquire(k_tma))
            _builder_emit(_tensormap_acquire(v_tma))
            _builder_emit(_tensormap_acquire(g_tma))
            _builder_emit(_tensormap_acquire(beta_tma_tmap))
            _builder_emit(_tensormap_acquire(out_tma))
            _builder_scope_exit(_builder_then_124_4)
            _builder_scope_exit(_builder_if_124_4)
            _builder_emit(T.cuda.cta_sync())
            tid = _builder_bind("tid", thread_idx)
            warp = _builder_scalar(
                "warp", _make_warp_uniform(T.cast(tid, "uint32") // T.uint32(32))
            )
            lane = _builder_bind("lane", tid % 32)
            smem = _builder_scalar(
                "smem", T.cuda.cvta_generic_to_shared(T.address_of(smem_raw[0])), dtype="uint32"
            )
            bid = _builder_bind("bid", block_idx)
            num_bids = _builder_bind("num_bids", num_seqs * h)
            smem_qd_addr = _builder_scalar(
                "smem_qd_addr", T.cast(smem, "int32") + SMEM_SMEM_QD_OFF, dtype="int32"
            )
            smem_g_raw_addr = _builder_scalar(
                "smem_g_raw_addr", T.cast(smem, "int32") + SMEM_SMEM_G_RAW_OFF, dtype="int32"
            )
            smem_g_raw_all_addr = _builder_scalar(
                "smem_g_raw_all_addr",
                T.cast(smem, "int32") + SMEM_SMEM_G_RAW_ALL_OFF,
                dtype="int32",
            )
            smem_kd_addr = _builder_scalar(
                "smem_kd_addr", T.cast(smem, "int32") + SMEM_SMEM_KD_OFF, dtype="int32"
            )
            smem_q_raw_prefetch_addr = _builder_scalar(
                "smem_q_raw_prefetch_addr",
                T.cast(smem, "int32") + SMEM_SMEM_Q_RAW_PREFETCH_OFF,
                dtype="int32",
            )
            smem_final_trans_addr = _builder_scalar(
                "smem_final_trans_addr",
                T.cast(smem, "int32") + SMEM_SMEM_FINAL_TRANS_OFF,
                dtype="int32",
            )
            smem_kr_trans_addr = _builder_scalar(
                "smem_kr_trans_addr", T.cast(smem, "int32") + SMEM_SMEM_KR_TRANS_OFF, dtype="int32"
            )
            smem_mqk_trans_addr = _builder_scalar(
                "smem_mqk_trans_addr",
                T.cast(smem, "int32") + SMEM_SMEM_MQK_TRANS_OFF,
                dtype="int32",
            )
            smem_inv_addr = _builder_scalar(
                "smem_inv_addr", T.cast(smem, "int32") + SMEM_SMEM_INV_OFF, dtype="int32"
            )
            smem_v_addr = _builder_scalar(
                "smem_v_addr", T.cast(smem, "int32") + SMEM_SMEM_V_OFF, dtype="int32"
            )
            smem_ki_addr = _builder_scalar(
                "smem_ki_addr", T.cast(smem, "int32") + SMEM_SMEM_KI_OFF, dtype="int32"
            )
            smem_gate_addr = _builder_scalar(
                "smem_gate_addr", T.cast(smem, "int32") + SMEM_SMEM_GATE_OFF, dtype="int32"
            )
            smem_beta_raw_addr = _builder_scalar(
                "smem_beta_raw_addr", T.cast(smem, "int32") + SMEM_SMEM_BETA_RAW_OFF, dtype="int32"
            )
            smem_inv_work_addr = _builder_scalar(
                "smem_inv_work_addr", T.cast(smem, "int32") + SMEM_SMEM_INV_WORK_OFF, dtype="int32"
            )
            smem_out_addr = _builder_scalar(
                "smem_out_addr", T.cast(smem, "int32") + SMEM_SMEM_OUT_OFF, dtype="int32"
            )
            smem_restore_factor_all_addr = _builder_scalar(
                "smem_restore_factor_all_addr",
                T.cast(smem, "int32") + SMEM_SMEM_RESTORE_FACTOR_ALL_OFF,
                dtype="int32",
            )
            smem_gt_prefix_all_addr = _builder_scalar(
                "smem_gt_prefix_all_addr",
                T.cast(smem, "int32") + SMEM_SMEM_GT_PREFIX_ALL_OFF,
                dtype="int32",
            )
            smem_gt_all_addr = _builder_scalar(
                "smem_gt_all_addr", T.cast(smem, "int32") + SMEM_SMEM_GT_ALL_OFF, dtype="int32"
            )
            smem_prep_beta_all_addr = _builder_scalar(
                "smem_prep_beta_all_addr",
                T.cast(smem, "int32") + SMEM_SMEM_PREP_BETA_ALL_OFF,
                dtype="int32",
            )
            smem_gate_rate_all_addr = _builder_scalar(
                "smem_gate_rate_all_addr",
                T.cast(smem, "int32") + SMEM_SMEM_GATE_RATE_ALL_OFF,
                dtype="int32",
            )
            smem_v_all_addr = _builder_scalar(
                "smem_v_all_addr", T.cast(smem, "int32") + SMEM_SMEM_V_ALL_OFF, dtype="int32"
            )
            smem_gate_all_addr = _builder_scalar(
                "smem_gate_all_addr", T.cast(smem, "int32") + SMEM_SMEM_GATE_ALL_OFF, dtype="int32"
            )
            _builder_if_168_4 = _builder_scope_enter(T.If(warp == 0))
            _builder_then_168_4 = _builder_scope_enter(T.Then())
            leader = _builder_scalar("leader", T.cuda.elect_sync())
            _builder_if_170_8 = _builder_scope_enter(T.If(leader))
            _builder_then_170_8 = _builder_scope_enter(T.Then())
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([0]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([8]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([16]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([24]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([32]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([40]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([48]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([56]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([64]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([72]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([80]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([88]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([96]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([104]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([112]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([120]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([128]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([136]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([144]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([152]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([160]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([168]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([176]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([184]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([192]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([200]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([208]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([216]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([224]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([232]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([240]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([248]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([256]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([264]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([272]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([280]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([288]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([296]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([304]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([312]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([320]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([328]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([336]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([344]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([352]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([360]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([368]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([376]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([384]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([392]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([400]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([408]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([416]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([424]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([432]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([440]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([448]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([456]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([464]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([472]), T.uint32(4)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([480]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([488]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([496]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([504]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([512]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([520]), T.uint32(1)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([528]), T.uint32(2)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([536]), T.uint32(2)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([544]), T.uint32(2)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([552]), T.uint32(2)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([560]), T.uint32(2)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([568]), T.uint32(2)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([576]), T.uint32(2)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([584]), T.uint32(2)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([592]), T.uint32(2)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([600]), T.uint32(2)))
            _builder_emit(T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([608]), T.uint32(2)))
            _builder_scope_exit(_builder_then_170_8)
            _builder_scope_exit(_builder_if_170_8)
            _builder_emit(T.ptx.fence.mbarrier_init.release.cluster())
            _builder_scope_exit(_builder_then_168_4)
            _builder_scope_exit(_builder_if_168_4)
            _builder_emit(T.cuda.cta_sync())
            _builder_if_269_4 = _builder_scope_enter(T.If(warp == 0))
            _builder_then_269_4 = _builder_scope_enter(T.Then())
            _builder_emit(
                T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    T.address_of(tmem_addr_storage[0]), T.uint32(256)
                )
            )
            _builder_scope_exit(_builder_then_269_4)
            _builder_scope_exit(_builder_if_269_4)
            _builder_emit(T.cuda.cta_sync())
            _builder_emit(T.ptx.tcgen05.fence__after_thread_sync())
            mbar_base = _builder_scalar("mbar_base", T.cast(smem, "int32"), dtype="int32")
            taddr = _builder_alloc_scalar("taddr", "int32")
            _builder_emit(T.ptx.ld.volatile.shared.s32(taddr, T.address_of(tmem_addr_storage[0])))
            tmem_tmem_state = _builder_scalar(
                "tmem_tmem_state", taddr + TMEM_TMEM_STATE_OFFSET, dtype="int32"
            )
            tmem_tmem_state_inp = _builder_scalar(
                "tmem_tmem_state_inp", taddr + TMEM_TMEM_STATE_INP_OFFSET, dtype="int32"
            )
            tmem_tmem_u_acc = _builder_scalar(
                "tmem_tmem_u_acc", taddr + TMEM_TMEM_U_ACC_OFFSET, dtype="int32"
            )
            tmem_tmem_u2_inp = _builder_scalar(
                "tmem_tmem_u2_inp", taddr + TMEM_TMEM_U2_INP_OFFSET, dtype="int32"
            )
            tmem_tmem_u2_acc = _builder_scalar(
                "tmem_tmem_u2_acc", taddr + TMEM_TMEM_U2_ACC_OFFSET, dtype="int32"
            )
            tmem_tmem_out = _builder_scalar(
                "tmem_tmem_out", taddr + TMEM_TMEM_OUT_OFFSET, dtype="int32"
            )
            tmem_tmem_state_out = _builder_scalar(
                "tmem_tmem_state_out", taddr + TMEM_TMEM_STATE_OUT_OFFSET, dtype="int32"
            )
            _builder_if_291_4 = _builder_scope_enter(T.If(T.And(warp >= 8, warp <= 11)))
            _builder_then_291_4 = _builder_scope_enter(T.Then())
            _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(48))
            _builder_scope_exit(_builder_then_291_4)
            _builder_scope_exit(_builder_if_291_4)
            _builder_if_295_4 = _builder_scope_enter(T.If(warp <= 3))
            _builder_then_295_4 = _builder_scope_enter(T.Then())
            _builder_emit(T.ptx.setmaxnreg.inc.sync.aligned.u32(168))
            task_idx = _builder_scalar("task_idx", bid, dtype="int32")
            seq_idx = _builder_scalar(
                "seq_idx", _ld_global_s32(seq_order, task_idx // h), dtype="int32"
            )
            head_idx = _builder_scalar("head_idx", task_idx % h, dtype="int32")
            bos = _builder_scalar("bos", _ld_global_s64(cu_seqlens, seq_idx), dtype="int64")
            eos = _builder_scalar("eos", _ld_global_s64(cu_seqlens, seq_idx + 1), dtype="int64")
            seq_len = _builder_scalar("seq_len", T.cast(eos - bos, "int32"), dtype="int32")
            num_chunks = _builder_scalar("num_chunks", (seq_len + 32 - 1) // 32, dtype="int32")
            warp_in_wg = _builder_scalar("warp_in_wg", warp % 4, dtype="int32")
            tmem_row_base = _builder_scalar("tmem_row_base", warp_in_wg * 32 << 16, dtype="int32")
            state_row = _builder_scalar("state_row", warp_in_wg * 32 + lane, dtype="int32")
            warp_id_in_role = _builder_scalar("warp_id_in_role", warp - 0, dtype="int32")
            compute_local_warp = _builder_scalar(
                "compute_local_warp", warp_id_in_role, dtype="int32"
            )
            state_base = _builder_scalar(
                "state_base",
                (
                    (T.cast(seq_idx, "int64") * T.cast(h, "int64") + T.cast(head_idx, "int64"))
                    * 128
                    + T.cast(state_row, "int64")
                )
                * 128,
                dtype="int64",
            )
            initial_state_u32 = _builder_name(
                "initial_state_u32",
                T.decl_buffer(
                    (num_seqs * h * D_HEAD * D_HEAD // 2,), "uint32", data=initial_state.data
                ),
            )
            final_state_u32 = _builder_name(
                "final_state_u32",
                T.decl_buffer(
                    (num_seqs * h * D_HEAD * D_HEAD // 2,), "uint32", data=final_state.data
                ),
            )
            with T.unroll(4) as state_col_block:
                state_frag = _builder_name("state_frag", T.alloc_local((32,), "float32"))
                with T.unroll(32) as _zi:
                    T.buffer_store(state_frag, T.float32(0.0), [_zi])
                if use_initial_state:
                    with T.unroll(2) as _blk:
                        _vld = _builder_name("_vld", T.alloc_local((4,), "uint32", align=16))
                        _builder_emit(
                            _ld_global_v4_u32(
                                _vld,
                                initial_state_u32.ptr_to(
                                    [(state_base + state_col_block * 32) // 2 + _blk * 4]
                                ),
                            )
                        )
                        with T.unroll(4) as _pair:
                            T.buffer_store(
                                state_frag,
                                T.cuda.uint_as_float(_vld[_pair] << T.uint32(16)),
                                [0 + _blk * 8 + _pair * 2],
                            )
                            T.buffer_store(
                                state_frag,
                                T.cuda.uint_as_float(_vld[_pair] & T.uint32(4294901760)),
                                [0 + _blk * 8 + _pair * 2 + 1],
                            )
                    with T.unroll(2) as _blk:
                        _vld1 = _builder_name("_vld1", T.alloc_local((4,), "uint32", align=16))
                        _builder_emit(
                            _ld_global_v4_u32(
                                _vld1,
                                initial_state_u32.ptr_to(
                                    [(state_base + state_col_block * 32 + 16) // 2 + _blk * 4]
                                ),
                            )
                        )
                        with T.unroll(4) as _pair:
                            T.buffer_store(
                                state_frag,
                                T.cuda.uint_as_float(_vld1[_pair] << T.uint32(16)),
                                [16 + _blk * 8 + _pair * 2],
                            )
                            T.buffer_store(
                                state_frag,
                                T.cuda.uint_as_float(_vld1[_pair] & T.uint32(4294901760)),
                                [16 + _blk * 8 + _pair * 2 + 1],
                            )
                _builder_emit(
                    _tmem_st_x32_f32(taddr + 64 + tmem_row_base + state_col_block * 32, state_frag)
                )
            _builder_emit(T.ptx.tcgen05.wait__st.sync.aligned())
            compute_stage = _builder_scalar("compute_stage", 0, dtype="uint32")
            _phase_qk_full = _builder_scalar("_phase_qk_full", 0, dtype="uint32")
            _phase_v_full = _builder_scalar("_phase_v_full", 0, dtype="uint32")
            _phase_old_out_ready = _builder_scalar("_phase_old_out_ready", 0, dtype="uint32")
            _phase_u2_acc_ready = _builder_scalar("_phase_u2_acc_ready", 0, dtype="uint32")
            _phase_final_ready = _builder_scalar("_phase_final_ready", 0, dtype="uint32")
            with T.serial(0, num_chunks, unroll=False) as chunk_idx:
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_QK_FULL_OFF + compute_stage * 8,
                        _phase_qk_full,
                    )
                )
                with T.serial(0, 4, unroll=False) as state_col_block_1:
                    state_addr = _builder_scalar(
                        "state_addr",
                        taddr + 64 + tmem_row_base + state_col_block_1 * 32,
                        dtype="int32",
                    )
                    _tmem_load_0 = _builder_name("_tmem_load_0", T.alloc_local((32,), "float32"))
                    _builder_emit(_tmem_ld_x32(_tmem_load_0, state_addr))
                    _tmem_load_0_bf16 = _builder_name(
                        "_tmem_load_0_bf16", T.alloc_local((16,), "uint32")
                    )
                    with T.unroll(16) as _lp:
                        T.buffer_store(
                            _tmem_load_0_bf16,
                            T.cuda.float22bfloat162_rn(
                                _tmem_load_0[_lp * 2 + 0], _tmem_load_0[_lp * 2 + 1 + 0]
                            ),
                            [_lp],
                        )
                    _builder_emit(
                        T.ptx["tcgen05.st.sync.aligned.32x32b.x16.b32"](
                            T.cast(taddr + tmem_row_base + state_col_block_1 * 16, "uint32"),
                            *[_tmem_load_0_bf16[_j] for _j in range(16)],
                        )
                    )
                    state_scale = _builder_name("state_scale", T.alloc_local((16,), "float32"))
                    with T.unroll(2) as state_half:
                        with T.unroll(4) as state_vec:
                            _builder_emit(
                                _ld_shared_v4_f32(
                                    state_scale,
                                    state_vec * 4,
                                    smem_gt_all,
                                    compute_stage * 10496
                                    + state_col_block_1 * 32
                                    + state_half * 16
                                    + state_vec * 4,
                                )
                            )
                        with T.unroll(8) as _ls:
                            _pk = _builder_scalar(
                                "_pk",
                                _mul_f32x2_inplace(
                                    T.cuda.make_float2(
                                        _tmem_load_0[state_half * 16 + _ls * 2],
                                        _tmem_load_0[state_half * 16 + _ls * 2 + 1],
                                    ),
                                    T.cuda.make_float2(
                                        state_scale[_ls * 2], state_scale[_ls * 2 + 1]
                                    ),
                                ),
                            )
                            T.buffer_store(
                                _tmem_load_0, T.cuda.float2_x(_pk), [state_half * 16 + _ls * 2]
                            )
                            T.buffer_store(
                                _tmem_load_0, T.cuda.float2_y(_pk), [state_half * 16 + _ls * 2 + 1]
                            )
                    _builder_emit(_tmem_st_x32_f32(state_addr, _tmem_load_0))
                _builder_emit(T.ptx.tcgen05.wait__st.sync.aligned())
                _builder_if_413_12 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                _builder_then_413_12 = _builder_scope_enter(T.Then())
                _builder_emit(
                    _mbarrier_arrive(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_STATE_INP_READY_OFF + compute_stage * 8,
                    )
                )
                _builder_scope_exit(_builder_then_413_12)
                _builder_scope_exit(_builder_if_413_12)
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_V_FULL_OFF + compute_stage * 8,
                        _phase_v_full,
                    )
                )
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_OLD_OUT_READY_OFF + compute_stage * 8,
                        _phase_old_out_ready,
                    )
                )
                _tmem_load_1 = _builder_name("_tmem_load_1", T.alloc_local((32,), "float32"))
                _builder_emit(_tmem_ld_x32(_tmem_load_1, taddr + 224 + tmem_row_base))
                with T.unroll(2) as residual_half:
                    residual_v = _builder_name("residual_v", T.alloc_local((16,), "float32"))
                    residual_beta = _builder_name("residual_beta", T.alloc_local((16,), "float32"))
                    with T.unroll(16) as residual_col:
                        token_col = _builder_scalar(
                            "token_col", residual_half * 16 + residual_col, dtype="int32"
                        )
                        T.buffer_store(
                            residual_v,
                            T.cuda.bfloat162float(
                                _ld_shared_bf16(
                                    smem_v_all, compute_stage * 20992 + token_col * 128 + state_row
                                )
                            ),
                            [residual_col],
                        )
                        T.buffer_store(
                            residual_beta,
                            _ld_shared_f32(smem_prep_beta_all, compute_stage * 10496 + token_col),
                            [residual_col],
                        )
                    with T.unroll(8) as _ls:
                        _pk = _builder_scalar(
                            "_pk",
                            _sub_f32x2_inplace(
                                T.cuda.make_float2(residual_v[_ls * 2], residual_v[_ls * 2 + 1]),
                                T.cuda.make_float2(
                                    _tmem_load_1[residual_half * 16 + _ls * 2],
                                    _tmem_load_1[residual_half * 16 + _ls * 2 + 1],
                                ),
                            ),
                        )
                        T.buffer_store(residual_v, T.cuda.float2_x(_pk), [_ls * 2])
                        T.buffer_store(residual_v, T.cuda.float2_y(_pk), [_ls * 2 + 1])
                    with T.unroll(8) as _ls:
                        _pk = _builder_scalar(
                            "_pk",
                            _mul_f32x2_inplace(
                                T.cuda.make_float2(residual_v[_ls * 2], residual_v[_ls * 2 + 1]),
                                T.cuda.make_float2(
                                    residual_beta[_ls * 2], residual_beta[_ls * 2 + 1]
                                ),
                            ),
                        )
                        T.buffer_store(residual_v, T.cuda.float2_x(_pk), [_ls * 2])
                        T.buffer_store(residual_v, T.cuda.float2_y(_pk), [_ls * 2 + 1])
                    residual_v_bf16 = _builder_name(
                        "residual_v_bf16", T.alloc_local((8,), "uint32")
                    )
                    with T.unroll(8) as _lp:
                        T.buffer_store(
                            residual_v_bf16,
                            T.cuda.float22bfloat162_rn(
                                residual_v[_lp * 2 + 0], residual_v[_lp * 2 + 1 + 0]
                            ),
                            [_lp],
                        )
                    _builder_emit(
                        _tmem_st_x8_u32(
                            taddr + 224 + tmem_row_base + residual_half * 8, residual_v_bf16
                        )
                    )
                _builder_emit(T.ptx.tcgen05.wait__st.sync.aligned())
                _builder_if_472_12 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                _builder_then_472_12 = _builder_scope_enter(T.Then())
                _builder_emit(
                    _mbarrier_arrive(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_V_FREE_OFF + compute_stage * 8,
                    )
                )
                _builder_emit(
                    _mbarrier_arrive(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_U_INP_READY_OFF + compute_stage * 8,
                    )
                )
                _builder_scope_exit(_builder_then_472_12)
                _builder_scope_exit(_builder_if_472_12)
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_U2_ACC_READY_OFF + compute_stage * 8,
                        _phase_u2_acc_ready,
                    )
                )
                _tmem_load_2 = _builder_name("_tmem_load_2", T.alloc_local((32,), "float32"))
                _builder_emit(_tmem_ld_x32(_tmem_load_2, taddr + tmem_row_base))
                _tmem_load_2_bf16 = _builder_name(
                    "_tmem_load_2_bf16", T.alloc_local((16,), "uint32")
                )
                with T.unroll(16) as _lp:
                    T.buffer_store(
                        _tmem_load_2_bf16,
                        T.cuda.float22bfloat162_rn(
                            _tmem_load_2[_lp * 2 + 0], _tmem_load_2[_lp * 2 + 1 + 0]
                        ),
                        [_lp],
                    )
                _builder_emit(
                    T.ptx["tcgen05.st.sync.aligned.32x32b.x16.b32"](
                        T.cast(taddr + 224 + tmem_row_base, "uint32"),
                        *[_tmem_load_2_bf16[_j] for _j in range(16)],
                    )
                )
                _builder_emit(T.ptx.tcgen05.wait__st.sync.aligned())
                _builder_if_500_12 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                _builder_then_500_12 = _builder_scope_enter(T.Then())
                _builder_emit(
                    _mbarrier_arrive(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_U2_INP_READY_OFF + compute_stage * 8,
                    )
                )
                _builder_scope_exit(_builder_then_500_12)
                _builder_scope_exit(_builder_if_500_12)
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_FINAL_READY_OFF + compute_stage * 8,
                        _phase_final_ready,
                    )
                )
                T.buffer_store(compute_stage.buffer, compute_stage + 1, [0])
                _builder_if_513_12 = _builder_scope_enter(T.If(compute_stage == 5))
                _builder_then_513_12 = _builder_scope_enter(T.Then())
                T.buffer_store(compute_stage.buffer, T.uint32(0), [0])
                T.buffer_store(_phase_qk_full.buffer, _phase_qk_full ^ T.uint32(1), [0])
                T.buffer_store(_phase_v_full.buffer, _phase_v_full ^ T.uint32(1), [0])
                T.buffer_store(_phase_old_out_ready.buffer, _phase_old_out_ready ^ T.uint32(1), [0])
                T.buffer_store(_phase_u2_acc_ready.buffer, _phase_u2_acc_ready ^ T.uint32(1), [0])
                T.buffer_store(_phase_final_ready.buffer, _phase_final_ready ^ T.uint32(1), [0])
                _builder_scope_exit(_builder_then_513_12)
                _builder_scope_exit(_builder_if_513_12)
            if store_final_state:
                with T.unroll(4) as state_col_block_2:
                    _tmem_load_3 = _builder_name("_tmem_load_3", T.alloc_local((32,), "float32"))
                    _builder_emit(
                        _tmem_ld_x32(
                            _tmem_load_3, taddr + 64 + tmem_row_base + state_col_block_2 * 32
                        )
                    )
                    with T.unroll(2) as _half2:
                        _pk = _builder_name("_pk", T.alloc_local((8,), "uint32"))
                        with T.unroll(8) as _pj:
                            T.buffer_store(
                                _pk,
                                T.cuda.float22bfloat162_rn(
                                    _tmem_load_3[_half2 * 16 + _pj * 2],
                                    _tmem_load_3[_half2 * 16 + _pj * 2 + 1],
                                ),
                                [_pj],
                            )
                        _builder_emit(
                            _st_global_v4_u32(
                                final_state_u32.ptr_to(
                                    [(state_base + state_col_block_2 * 32 + _half2 * 16) // 2]
                                ),
                                _pk[0],
                                _pk[1],
                                _pk[2],
                                _pk[3],
                            )
                        )
                        _builder_emit(
                            _st_global_v4_u32(
                                final_state_u32.ptr_to(
                                    [(state_base + state_col_block_2 * 32 + _half2 * 16) // 2 + 4]
                                ),
                                _pk[4],
                                _pk[5],
                                _pk[6],
                                _pk[7],
                            )
                        )
            _builder_emit(T.ptx.bar.sync(T.uint32(10), T.uint32(128)))
            _builder_if_552_8 = _builder_scope_enter(T.If(compute_local_warp == 0))
            _builder_then_552_8 = _builder_scope_enter(T.Then())
            _builder_if_553_12 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
            _builder_then_553_12 = _builder_scope_enter(T.Then())
            _builder_emit(
                _mbarrier_arrive(
                    smem_raw, T.cast(smem, "int32"), mbar_base + MBAR_TMEM_DEALLOC_READY_OFF
                )
            )
            _builder_scope_exit(_builder_then_553_12)
            _builder_scope_exit(_builder_if_553_12)
            _builder_scope_exit(_builder_then_552_8)
            _builder_scope_exit(_builder_if_552_8)
            _builder_scope_exit(_builder_then_295_4)
            _builder_else_295_4 = _builder_scope_enter(T.Else())
            _builder_if_557_4 = _builder_scope_enter(T.If(T.And(warp >= 4, warp <= 7)))
            _builder_then_557_4 = _builder_scope_enter(T.Then())
            _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(48))
            task_idx_1 = _builder_scalar("task_idx_1", bid, dtype="int32")
            seq_idx_1 = _builder_scalar(
                "seq_idx_1", _ld_global_s32(seq_order, task_idx_1 // h), dtype="int32"
            )
            head_idx_1 = _builder_scalar("head_idx_1", task_idx_1 % h, dtype="int32")
            bos_1 = _builder_scalar("bos_1", _ld_global_s64(cu_seqlens, seq_idx_1), dtype="int64")
            eos_1 = _builder_scalar(
                "eos_1", _ld_global_s64(cu_seqlens, seq_idx_1 + 1), dtype="int64"
            )
            seq_len_1 = _builder_scalar("seq_len_1", T.cast(eos_1 - bos_1, "int32"), dtype="int32")
            num_chunks_1 = _builder_scalar(
                "num_chunks_1", (seq_len_1 + 32 - 1) // 32, dtype="int32"
            )
            warp_id_in_role_1 = _builder_scalar("warp_id_in_role_1", warp - 4, dtype="int32")
            epilogue_local_warp = _builder_scalar(
                "epilogue_local_warp", warp_id_in_role_1, dtype="int32"
            )
            warp_in_wg_1 = _builder_scalar("warp_in_wg_1", warp % 4, dtype="int32")
            tmem_row_base_1 = _builder_scalar(
                "tmem_row_base_1", warp_in_wg_1 * 32 << 16, dtype="int32"
            )
            state_row_1 = _builder_scalar("state_row_1", warp_in_wg_1 * 32 + lane, dtype="int32")
            epilogue_stage = _builder_scalar("epilogue_stage", 0, dtype="uint32")
            output_stage = _builder_scalar("output_stage", 0, dtype="uint32")
            _phase_final_ready_1 = _builder_scalar("_phase_final_ready_1", 0, dtype="uint32")
            with T.serial(0, num_chunks_1, unroll=False) as chunk_idx_1:
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_FINAL_READY_OFF + epilogue_stage * 8,
                        _phase_final_ready_1,
                    )
                )
                chunk_is_full = _builder_scalar(
                    "chunk_is_full",
                    T.if_then_else(seq_len_1 >= (chunk_idx_1 + 1) * 32, 1, 0),
                    dtype="int32",
                )
                _builder_if_586_12 = _builder_scope_enter(T.If(chunk_is_full != 0))
                _builder_then_586_12 = _builder_scope_enter(T.Then())
                _tmem_load_4 = _builder_name("_tmem_load_4", T.alloc_local((16,), "uint32"))
                _builder_emit(
                    T.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                        *[_tmem_load_4[_j] for _j in range(16)],
                        T.cast(taddr + 192 + tmem_row_base_1, "uint32"),
                    )
                )
                _tmem_load_5 = _builder_name("_tmem_load_5", T.alloc_local((16,), "uint32"))
                _builder_emit(
                    T.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                        *[_tmem_load_5[_j] for _j in range(16)],
                        T.cast(taddr + 192 + tmem_row_base_1 + 1048576, "uint32"),
                    )
                )
                _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                _builder_emit(T.ptx.bar.sync(T.uint32(9), T.uint32(128)))
                _builder_if_601_16 = _builder_scope_enter(T.If(epilogue_local_warp == 0))
                _builder_then_601_16 = _builder_scope_enter(T.Then())
                _builder_if_602_20 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                _builder_then_602_20 = _builder_scope_enter(T.Then())
                _builder_emit(
                    _mbarrier_arrive(
                        smem_raw, T.cast(smem, "int32"), mbar_base + MBAR_OUT_EMPTY_OFF
                    )
                )
                _builder_scope_exit(_builder_then_602_20)
                _builder_scope_exit(_builder_if_602_20)
                _builder_scope_exit(_builder_then_601_16)
                _builder_scope_exit(_builder_if_601_16)
                _builder_if_606_16 = _builder_scope_enter(T.If(epilogue_local_warp == 0))
                _builder_then_606_16 = _builder_scope_enter(T.Then())
                _builder_if_607_20 = _builder_scope_enter(T.If(chunk_idx_1 >= 2))
                _builder_then_607_20 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.cp.async_.bulk.wait_group.read(1))
                _builder_scope_exit(_builder_then_607_20)
                _builder_scope_exit(_builder_if_607_20)
                _builder_scope_exit(_builder_then_606_16)
                _builder_scope_exit(_builder_if_606_16)
                _builder_emit(T.ptx.bar.sync(T.uint32(9), T.uint32(128)))
                out_stage_addr = _builder_scalar(
                    "out_stage_addr",
                    smem_out_addr + T.cast(output_stage, "int32") * 8192,
                    dtype="int32",
                )
                with T.unroll(2) as dim_half:
                    out_packed = _builder_name("out_packed", T.alloc_local((8,), "uint32", align=4))
                    _builder_if_615_20 = _builder_scope_enter(T.If(dim_half == 0))
                    _builder_then_615_20 = _builder_scope_enter(T.Then())
                    with T.unroll(8) as _lp:
                        T.buffer_store(
                            out_packed,
                            T.cuda.float22bfloat162_rn(
                                T.cuda.uint_as_float(_tmem_load_4[_lp * 2 + 0]),
                                T.cuda.uint_as_float(_tmem_load_4[_lp * 2 + 1 + 0]),
                            ),
                            [_lp],
                        )
                    _builder_scope_exit(_builder_then_615_20)
                    _builder_else_615_20 = _builder_scope_enter(T.Else())
                    with T.unroll(8) as _lp:
                        T.buffer_store(
                            out_packed,
                            T.cuda.float22bfloat162_rn(
                                T.cuda.uint_as_float(_tmem_load_5[_lp * 2 + 0]),
                                T.cuda.uint_as_float(_tmem_load_5[_lp * 2 + 1 + 0]),
                            ),
                            [_lp],
                        )
                    _builder_scope_exit(_builder_else_615_20)
                    _builder_scope_exit(_builder_if_615_20)
                    with T.unroll(2) as token_group:
                        mtx_idx = _builder_scalar("mtx_idx", lane // 8, dtype="int32")
                        row_addr = _builder_scalar("row_addr", lane & 7, dtype="int32")
                        dim_base = _builder_scalar(
                            "dim_base",
                            epilogue_local_warp * 32 + dim_half * 16 + (mtx_idx & 1) * 8,
                            dtype="int32",
                        )
                        token_base = _builder_scalar(
                            "token_base", token_group * 16 + mtx_idx // 2 * 8, dtype="int32"
                        )
                        token_addr = _builder_scalar(
                            "token_addr", token_base + row_addr, dtype="int32"
                        )
                        token_pair = _builder_scalar("token_pair", token_addr // 2, dtype="int32")
                        token_parity = _builder_scalar(
                            "token_parity", token_addr & 1, dtype="int32"
                        )
                        raw_row = _builder_scalar(
                            "raw_row", token_pair + dim_base // 64 * 16, dtype="int32"
                        )
                        raw_col = _builder_scalar(
                            "raw_col",
                            (dim_base & 63 ^ (token_pair & 3) << 4 ^ token_parity << 3)
                            + token_parity * 64,
                            dtype="int32",
                        )
                        stsm_offset = _builder_scalar(
                            "stsm_offset", (raw_row * 128 + raw_col) * 2, dtype="int32"
                        )
                        pack_base = _builder_scalar("pack_base", token_group * 4, dtype="int32")
                        _builder_emit(
                            T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                                smem_raw.ptr_to(
                                    [
                                        T.cast(out_stage_addr, "uint32")
                                        - smem
                                        + T.cast(stsm_offset, "uint32")
                                    ]
                                ),
                                out_packed[pack_base],
                                out_packed[pack_base + 1],
                                out_packed[pack_base + 2],
                                out_packed[pack_base + 3],
                            )
                        )
                _builder_emit(_fence_async_shared())
                _builder_emit(T.ptx.bar.sync(T.uint32(9), T.uint32(128)))
                _builder_if_653_16 = _builder_scope_enter(T.If(epilogue_local_warp == 0))
                _builder_then_653_16 = _builder_scope_enter(T.Then())
                _builder_if_654_20 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                _builder_then_654_20 = _builder_scope_enter(T.Then())
                _builder_emit(
                    _tma_store_4d(
                        smem_raw,
                        T.cast(smem, "int32"),
                        out_tma,
                        0,
                        T.cast(bos_1 + T.cast(chunk_idx_1 * 32, "int64"), "int32"),
                        head_idx_1,
                        0,
                        T.cast(smem_out_addr + T.cast(output_stage, "int32") * 8192, "uint32"),
                    )
                )
                _builder_scope_exit(_builder_then_654_20)
                _builder_scope_exit(_builder_if_654_20)
                _builder_emit(T.ptx.cp.async_.bulk.commit_group())
                _builder_scope_exit(_builder_then_653_16)
                _builder_scope_exit(_builder_if_653_16)
                T.buffer_store(output_stage.buffer, output_stage ^ T.uint32(1), [0])
                _builder_scope_exit(_builder_then_586_12)
                _builder_else_586_12 = _builder_scope_enter(T.Else())
                _tmem_load_6 = _builder_name("_tmem_load_6", T.alloc_local((32,), "float32"))
                _builder_emit(_tmem_ld_x32(_tmem_load_6, taddr + 192 + tmem_row_base_1))
                _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                _builder_emit(T.ptx.bar.sync(T.uint32(9), T.uint32(128)))
                _builder_if_672_16 = _builder_scope_enter(T.If(epilogue_local_warp == 0))
                _builder_then_672_16 = _builder_scope_enter(T.Then())
                _builder_if_673_20 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                _builder_then_673_20 = _builder_scope_enter(T.Then())
                _builder_emit(
                    _mbarrier_arrive(
                        smem_raw, T.cast(smem, "int32"), mbar_base + MBAR_OUT_EMPTY_OFF
                    )
                )
                _builder_scope_exit(_builder_then_673_20)
                _builder_scope_exit(_builder_if_673_20)
                _builder_scope_exit(_builder_then_672_16)
                _builder_scope_exit(_builder_if_672_16)
                with T.unroll(32) as token_col_1:
                    out_token = _builder_scalar(
                        "out_token",
                        bos_1 + T.cast(chunk_idx_1 * 32 + token_col_1, "int64"),
                        dtype="int64",
                    )
                    _builder_if_681_20 = _builder_scope_enter(T.If(out_token < eos_1))
                    _builder_then_681_20 = _builder_scope_enter(T.Then())
                    out_idx = _builder_scalar(
                        "out_idx",
                        (out_token * T.cast(h, "int64") + T.cast(head_idx_1, "int64")) * 128
                        + T.cast(state_row_1, "int64"),
                        dtype="int64",
                    )
                    _builder_emit(
                        _st_global_bf16(out, out_idx, T.cast(_tmem_load_6[token_col_1], "bfloat16"))
                    )
                    _builder_scope_exit(_builder_then_681_20)
                    _builder_scope_exit(_builder_if_681_20)
                _builder_scope_exit(_builder_else_586_12)
                _builder_scope_exit(_builder_if_586_12)
                T.buffer_store(epilogue_stage.buffer, epilogue_stage + 1, [0])
                _builder_if_689_12 = _builder_scope_enter(T.If(epilogue_stage == 5))
                _builder_then_689_12 = _builder_scope_enter(T.Then())
                T.buffer_store(epilogue_stage.buffer, T.uint32(0), [0])
                T.buffer_store(_phase_final_ready_1.buffer, _phase_final_ready_1 ^ T.uint32(1), [0])
                _builder_scope_exit(_builder_then_689_12)
                _builder_scope_exit(_builder_if_689_12)
            _builder_if_692_8 = _builder_scope_enter(T.If(epilogue_local_warp == 0))
            _builder_then_692_8 = _builder_scope_enter(T.Then())
            _builder_emit(T.ptx.cp.async_.bulk.wait_group(0))
            _builder_scope_exit(_builder_then_692_8)
            _builder_scope_exit(_builder_if_692_8)
            _builder_emit(T.ptx.bar.sync(T.uint32(9), T.uint32(128)))
            _builder_if_695_8 = _builder_scope_enter(T.If(epilogue_local_warp == 0))
            _builder_then_695_8 = _builder_scope_enter(T.Then())
            _builder_if_696_12 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
            _builder_then_696_12 = _builder_scope_enter(T.Then())
            _builder_emit(
                _mbarrier_arrive(
                    smem_raw, T.cast(smem, "int32"), mbar_base + MBAR_TMEM_DEALLOC_READY_OFF
                )
            )
            _builder_scope_exit(_builder_then_696_12)
            _builder_scope_exit(_builder_if_696_12)
            _builder_scope_exit(_builder_then_695_8)
            _builder_scope_exit(_builder_if_695_8)
            _builder_scope_exit(_builder_then_557_4)
            _builder_else_557_4 = _builder_scope_enter(T.Else())
            _builder_if_700_4 = _builder_scope_enter(T.If(warp == 9))
            _builder_then_700_4 = _builder_scope_enter(T.Then())
            task_idx_2 = _builder_scalar("task_idx_2", bid, dtype="int32")
            seq_idx_2 = _builder_scalar(
                "seq_idx_2", _ld_global_s32(seq_order, task_idx_2 // h), dtype="int32"
            )
            bos_2 = _builder_scalar("bos_2", _ld_global_s64(cu_seqlens, seq_idx_2), dtype="int64")
            eos_2 = _builder_scalar(
                "eos_2", _ld_global_s64(cu_seqlens, seq_idx_2 + 1), dtype="int64"
            )
            seq_len_2 = _builder_scalar("seq_len_2", T.cast(eos_2 - bos_2, "int32"), dtype="int32")
            num_chunks_2 = _builder_scalar(
                "num_chunks_2", (seq_len_2 + 32 - 1) // 32, dtype="int32"
            )
            mma_stage = _builder_scalar("mma_stage", 0, dtype="uint32")
            _phase_qk_full_1 = _builder_scalar("_phase_qk_full_1", 0, dtype="uint32")
            _phase_state_inp_ready = _builder_scalar("_phase_state_inp_ready", 0, dtype="uint32")
            _phase_out_empty_0 = _builder_scalar("_phase_out_empty_0", 1, dtype="uint32")
            _phase_u_inp_ready = _builder_scalar("_phase_u_inp_ready", 0, dtype="uint32")
            _phase_u2_inp_ready = _builder_scalar("_phase_u2_inp_ready", 0, dtype="uint32")
            with T.serial(0, num_chunks_2, unroll=False) as _chunk_idx:
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_QK_FULL_OFF + mma_stage * 8,
                        _phase_qk_full_1,
                    )
                )
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_STATE_INP_READY_OFF + mma_stage * 8,
                        _phase_state_inp_ready,
                    )
                )
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_OUT_EMPTY_OFF,
                        _phase_out_empty_0,
                    )
                )
                T.buffer_store(_phase_out_empty_0.buffer, _phase_out_empty_0 ^ T.uint32(1), [0])
                _mma_b_addr_0 = _builder_scalar(
                    "_mma_b_addr_0",
                    smem_qd_addr + T.cast(mma_stage, "int32") * 41984,
                    dtype="int32",
                )
                _mma_b_lo_0 = _builder_scalar(
                    "_mma_b_lo_0", _make_warp_uniform(T.cast(_mma_b_addr_0 >> 4 & 16383, "uint32"))
                )
                _builder_emit(
                    _mma_qk_8step(
                        tmem_tmem_out, T.cast(_mma_b_lo_0, "int32"), tmem_tmem_state_inp, 0
                    )
                )
                _mma_b_addr_1 = _builder_scalar(
                    "_mma_b_addr_1",
                    smem_kd_addr + T.cast(mma_stage, "int32") * 41984,
                    dtype="int32",
                )
                _mma_b_lo_1 = _builder_scalar(
                    "_mma_b_lo_1", _make_warp_uniform(T.cast(_mma_b_addr_1 >> 4 & 16383, "uint32"))
                )
                _builder_emit(
                    _mma_qk_8step(
                        tmem_tmem_u_acc, T.cast(_mma_b_lo_1, "int32"), tmem_tmem_state_inp, 0
                    )
                )
                _leader_1190 = _builder_scalar("_leader_1190", T.cuda.elect_sync())
                _builder_emit(
                    T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                        smem_raw.ptr_to(
                            [
                                mbar_base
                                + MBAR_OLD_OUT_READY_OFF
                                + mma_stage * 8
                                - T.cast(smem, "int32")
                            ]
                        ),
                        pred=_leader_1190,
                    )
                )
                _builder_emit(
                    T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                        smem_raw.ptr_to(
                            [
                                mbar_base
                                + MBAR_RAW_INPUTS_FREE_OFF
                                + mma_stage * 8
                                - T.cast(smem, "int32")
                            ]
                        ),
                        pred=_leader_1190,
                    )
                )
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_U_INP_READY_OFF + mma_stage * 8,
                        _phase_u_inp_ready,
                    )
                )
                _mma_b_addr_2 = _builder_scalar(
                    "_mma_b_addr_2",
                    smem_inv_addr + T.cast(mma_stage, "int32") * 41984,
                    dtype="int32",
                )
                _mma_b_lo_2 = _builder_scalar(
                    "_mma_b_lo_2", _make_warp_uniform(T.cast(_mma_b_addr_2 >> 4 & 16383, "uint32"))
                )
                _builder_emit(
                    _mma_inv_2step(
                        tmem_tmem_u2_acc, T.cast(_mma_b_lo_2, "int32"), tmem_tmem_u2_inp, 0
                    )
                )
                _builder_emit(
                    _elect_commit(
                        smem_raw.ptr_to(
                            [
                                mbar_base
                                + MBAR_U2_ACC_READY_OFF
                                + mma_stage * 8
                                - T.cast(smem, "int32")
                            ]
                        )
                    )
                )
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_U2_INP_READY_OFF + mma_stage * 8,
                        _phase_u2_inp_ready,
                    )
                )
                _mma_b_addr_3 = _builder_scalar(
                    "_mma_b_addr_3",
                    smem_final_trans_addr + T.cast(mma_stage, "int32") * 41984,
                    dtype="int32",
                )
                _mma_b_lo_3 = _builder_scalar(
                    "_mma_b_lo_3",
                    _make_warp_uniform(T.cast(_mma_b_addr_3 >> 4 & 16383 | 16777216, "uint32")),
                )
                _builder_emit(
                    _mma_final_2step(
                        tmem_tmem_state_out, T.cast(_mma_b_lo_3, "int32"), tmem_tmem_u2_inp, 1
                    )
                )
                _leader_1236 = _builder_scalar("_leader_1236", T.cuda.elect_sync())
                _builder_emit(
                    T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                        smem_raw.ptr_to(
                            [
                                mbar_base
                                + MBAR_FINAL_READY_OFF
                                + mma_stage * 8
                                - T.cast(smem, "int32")
                            ]
                        ),
                        pred=_leader_1236,
                    )
                )
                _builder_emit(
                    T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                        smem_raw.ptr_to(
                            [mbar_base + MBAR_SMEM_FREE_OFF + mma_stage * 8 - T.cast(smem, "int32")]
                        ),
                        pred=_leader_1236,
                    )
                )
                T.buffer_store(mma_stage.buffer, mma_stage + 1, [0])
                _builder_if_808_12 = _builder_scope_enter(T.If(mma_stage == 5))
                _builder_then_808_12 = _builder_scope_enter(T.Then())
                T.buffer_store(mma_stage.buffer, T.uint32(0), [0])
                T.buffer_store(_phase_qk_full_1.buffer, _phase_qk_full_1 ^ T.uint32(1), [0])
                T.buffer_store(
                    _phase_state_inp_ready.buffer, _phase_state_inp_ready ^ T.uint32(1), [0]
                )
                T.buffer_store(_phase_u_inp_ready.buffer, _phase_u_inp_ready ^ T.uint32(1), [0])
                T.buffer_store(_phase_u2_inp_ready.buffer, _phase_u2_inp_ready ^ T.uint32(1), [0])
                _builder_scope_exit(_builder_then_808_12)
                _builder_scope_exit(_builder_if_808_12)
            _phase_tmem_dealloc_ready_0 = _builder_scalar(
                "_phase_tmem_dealloc_ready_0", 0, dtype="uint32"
            )
            _builder_emit(
                _mbarrier_wait(
                    smem_raw,
                    T.cast(smem, "int32"),
                    mbar_base + MBAR_TMEM_DEALLOC_READY_OFF,
                    _phase_tmem_dealloc_ready_0,
                )
            )
            T.buffer_store(
                _phase_tmem_dealloc_ready_0.buffer, _phase_tmem_dealloc_ready_0 ^ T.uint32(1), [0]
            )
            _tmem_dealloc_addr = _builder_alloc_scalar("_tmem_dealloc_addr", "int32")
            _builder_emit(
                T.ptx.ld.volatile.shared.s32(_tmem_dealloc_addr, T.address_of(tmem_addr_storage[0]))
            )
            _builder_emit(
                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                    T.cast(_tmem_dealloc_addr, "uint32"), T.uint32(256)
                )
            )
            _builder_emit(T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned())
            _builder_scope_exit(_builder_then_700_4)
            _builder_else_700_4 = _builder_scope_enter(T.Else())
            _builder_if_828_4 = _builder_scope_enter(T.If(warp == 10))
            _builder_then_828_4 = _builder_scope_enter(T.Then())
            task_idx_3 = _builder_scalar("task_idx_3", bid, dtype="int32")
            seq_idx_3 = _builder_scalar(
                "seq_idx_3", _ld_global_s32(seq_order, task_idx_3 // h), dtype="int32"
            )
            head_idx_2 = _builder_scalar("head_idx_2", task_idx_3 % h, dtype="int32")
            bos_3 = _builder_scalar("bos_3", _ld_global_s64(cu_seqlens, seq_idx_3), dtype="int64")
            eos_3 = _builder_scalar(
                "eos_3", _ld_global_s64(cu_seqlens, seq_idx_3 + 1), dtype="int64"
            )
            seq_len_3 = _builder_scalar("seq_len_3", T.cast(eos_3 - bos_3, "int32"), dtype="int32")
            num_chunks_3 = _builder_scalar(
                "num_chunks_3", (seq_len_3 + 32 - 1) // 32, dtype="int32"
            )
            load_stage = _builder_scalar("load_stage", 0, dtype="uint32")
            _phase_v_free = _builder_scalar("_phase_v_free", 1, dtype="uint32")
            _phase_qk_full_2 = _builder_scalar("_phase_qk_full_2", 0, dtype="uint32")
            with T.serial(0, num_chunks_3, unroll=False) as chunk_idx_2:
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_V_FREE_OFF + load_stage * 8,
                        _phase_v_free,
                    )
                )
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_QK_FULL_OFF + load_stage * 8,
                        _phase_qk_full_2,
                    )
                )
                chunk_is_full_1 = _builder_scalar(
                    "chunk_is_full_1",
                    T.if_then_else(seq_len_3 >= (chunk_idx_2 + 1) * 32, 1, 0),
                    dtype="int32",
                )
                _builder_if_857_12 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                _builder_then_857_12 = _builder_scope_enter(T.Then())
                _builder_if_858_16 = _builder_scope_enter(T.If(chunk_is_full_1 != 0))
                _builder_then_858_16 = _builder_scope_enter(T.Then())
                _builder_emit(
                    _mbarrier_arrive_expect_tx(
                        smem_raw.ptr_to(
                            [mbar_base + MBAR_V_FULL_OFF + load_stage * 8 - T.cast(smem, "int32")]
                        ),
                        8192,
                    )
                )
                _builder_emit(
                    _tma_3d_gmem2smem(
                        smem_raw,
                        T.cast(smem, "int32"),
                        smem_v_addr + T.cast(load_stage, "int32") * 41984,
                        v_tma,
                        0,
                        head_idx_2,
                        T.cast(bos_3 + T.cast(chunk_idx_2 * 32, "int64"), "int32"),
                        mbar_base + MBAR_V_FULL_OFF + load_stage * 8,
                    )
                )
                _builder_scope_exit(_builder_then_858_16)
                _builder_scope_exit(_builder_if_858_16)
                _builder_scope_exit(_builder_then_857_12)
                _builder_scope_exit(_builder_if_857_12)
                _builder_if_875_12 = _builder_scope_enter(T.If(chunk_is_full_1 == 0))
                _builder_then_875_12 = _builder_scope_enter(T.Then())
                with T.unroll(16) as v_load_iter:
                    v_item = _builder_scalar("v_item", v_load_iter * 32 + lane, dtype="int32")
                    row = _builder_scalar("row", v_item // 16, dtype="int32")
                    segment = _builder_scalar("segment", v_item % 16, dtype="int32")
                    token = _builder_scalar(
                        "token", bos_3 + T.cast(chunk_idx_2 * 32 + row, "int64"), dtype="int64"
                    )
                    token_valid = _builder_scalar(
                        "token_valid", T.if_then_else(token < eos_3, 1, 0), dtype="int32"
                    )
                    v_src = _builder_scalar(
                        "v_src",
                        (token * T.cast(h, "int64") + T.cast(head_idx_2, "int64")) * 128
                        + T.cast(segment * 8, "int64"),
                        dtype="int64",
                    )
                    _builder_emit(
                        T.ptx["cp.async.cg.shared.global"](
                            smem_raw.ptr_to(
                                [
                                    SMEM_SMEM_V_OFF
                                    + T.cast(load_stage, "int32") * 41984
                                    + (row * 128 + segment * 8) * 2
                                ]
                            ),
                            v.ptr_to([v_src]),
                            16,
                            T.cast(T.if_then_else(token_valid != 0, 16, 0), "uint32"),
                        )
                    )
                _builder_emit(T.ptx.cp.async_.commit_group())
                _builder_emit(T.ptx.cp.async_.wait_group(0))
                _builder_scope_exit(_builder_then_875_12)
                _builder_scope_exit(_builder_if_875_12)
                _builder_emit(T.ptx.bar.sync(T.uint32(8), T.uint32(32)))
                _builder_if_894_12 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                _builder_then_894_12 = _builder_scope_enter(T.Then())
                _builder_if_895_16 = _builder_scope_enter(T.If(chunk_is_full_1 == 0))
                _builder_then_895_16 = _builder_scope_enter(T.Then())
                _builder_emit(_fence_async_shared())
                _builder_emit(
                    _mbarrier_arrive(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_V_FULL_OFF + load_stage * 8,
                    )
                )
                _builder_scope_exit(_builder_then_895_16)
                _builder_scope_exit(_builder_if_895_16)
                _builder_scope_exit(_builder_then_894_12)
                _builder_scope_exit(_builder_if_894_12)
                T.buffer_store(load_stage.buffer, load_stage + 1, [0])
                _builder_if_903_12 = _builder_scope_enter(T.If(load_stage == 5))
                _builder_then_903_12 = _builder_scope_enter(T.Then())
                T.buffer_store(load_stage.buffer, T.uint32(0), [0])
                T.buffer_store(_phase_v_free.buffer, _phase_v_free ^ T.uint32(1), [0])
                T.buffer_store(_phase_qk_full_2.buffer, _phase_qk_full_2 ^ T.uint32(1), [0])
                _builder_scope_exit(_builder_then_903_12)
                _builder_scope_exit(_builder_if_903_12)
            _builder_scope_exit(_builder_then_828_4)
            _builder_else_828_4 = _builder_scope_enter(T.Else())
            _builder_if_907_4 = _builder_scope_enter(T.If(T.And(warp >= 12, warp <= 31)))
            _builder_then_907_4 = _builder_scope_enter(T.Then())
            _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(48))
            task_idx_4 = _builder_scalar("task_idx_4", bid, dtype="int32")
            seq_idx_4 = _builder_scalar(
                "seq_idx_4", _ld_global_s32(seq_order, task_idx_4 // h), dtype="int32"
            )
            head_idx_3 = _builder_scalar("head_idx_3", task_idx_4 % h, dtype="int32")
            bos_4 = _builder_scalar("bos_4", _ld_global_s64(cu_seqlens, seq_idx_4), dtype="int64")
            eos_4 = _builder_scalar(
                "eos_4", _ld_global_s64(cu_seqlens, seq_idx_4 + 1), dtype="int64"
            )
            seq_len_4 = _builder_scalar("seq_len_4", T.cast(eos_4 - bos_4, "int32"), dtype="int32")
            num_chunks_4 = _builder_scalar(
                "num_chunks_4", (seq_len_4 + 32 - 1) // 32, dtype="int32"
            )
            instance_id = _builder_scalar("instance_id", (warp - 12) // 4, dtype="int32")
            prep_instance = _builder_scalar("prep_instance", instance_id, dtype="int32")
            warp_id_in_role_2 = _builder_scalar("warp_id_in_role_2", warp - 12, dtype="int32")
            prep_local_warp = _builder_scalar(
                "prep_local_warp", warp_id_in_role_2 - prep_instance * 4, dtype="int32"
            )
            prep_tid = _builder_scalar("prep_tid", prep_local_warp * 32 + lane, dtype="int32")
            num_prep_iters = _builder_scalar(
                "num_prep_iters", (num_chunks_4 + 4 - prep_instance) // 5, dtype="int32"
            )
            prep_stage = _builder_scalar(
                "prep_stage", T.cast(prep_instance, "uint32"), dtype="uint32"
            )
            gate_rate_stage_f32 = _builder_scalar(
                "gate_rate_stage_f32", prep_instance * 10496, dtype="int32"
            )
            _builder_if_926_8 = _builder_scope_enter(T.If(prep_tid == 0))
            _builder_then_926_8 = _builder_scope_enter(T.Then())
            a_log_value = _builder_scalar(
                "a_log_value", _ld_global_f32(A_log, head_idx_3), dtype="f32"
            )
            _builder_emit(
                _st_shared_f32(smem_gate_rate_all, gate_rate_stage_f32, _expf(a_log_value))
            )
            _builder_scope_exit(_builder_then_926_8)
            _builder_scope_exit(_builder_if_926_8)
            _builder_if_929_8 = _builder_scope_enter(T.If(prep_instance == 0))
            _builder_then_929_8 = _builder_scope_enter(T.Then())
            _builder_emit(T.ptx.bar.sync(T.uint32(11), T.uint32(128)))
            _builder_scope_exit(_builder_then_929_8)
            _builder_else_929_8 = _builder_scope_enter(T.Else())
            _builder_if_931_8 = _builder_scope_enter(T.If(prep_instance == 1))
            _builder_then_931_8 = _builder_scope_enter(T.Then())
            _builder_emit(T.ptx.bar.sync(T.uint32(12), T.uint32(128)))
            _builder_scope_exit(_builder_then_931_8)
            _builder_else_931_8 = _builder_scope_enter(T.Else())
            _builder_if_934_12 = _builder_scope_enter(T.If(prep_instance == 2))
            _builder_then_934_12 = _builder_scope_enter(T.Then())
            _builder_emit(T.ptx.bar.sync(T.uint32(13), T.uint32(128)))
            _builder_scope_exit(_builder_then_934_12)
            _builder_else_934_12 = _builder_scope_enter(T.Else())
            _builder_if_936_12 = _builder_scope_enter(T.If(prep_instance == 3))
            _builder_then_936_12 = _builder_scope_enter(T.Then())
            _builder_emit(T.ptx.bar.sync(T.uint32(14), T.uint32(128)))
            _builder_scope_exit(_builder_then_936_12)
            _builder_else_936_12 = _builder_scope_enter(T.Else())
            _builder_emit(T.ptx.bar.sync(T.uint32(15), T.uint32(128)))
            _builder_scope_exit(_builder_else_936_12)
            _builder_scope_exit(_builder_if_936_12)
            _builder_scope_exit(_builder_else_934_12)
            _builder_scope_exit(_builder_if_934_12)
            _builder_scope_exit(_builder_else_931_8)
            _builder_scope_exit(_builder_if_931_8)
            _builder_scope_exit(_builder_else_929_8)
            _builder_scope_exit(_builder_if_929_8)
            _phase_raw_inputs_free = _builder_scalar("_phase_raw_inputs_free", 1, dtype="uint32")
            _phase_gate_raw_full = _builder_scalar("_phase_gate_raw_full", 0, dtype="uint32")
            _phase_smem_free = _builder_scalar("_phase_smem_free", 1, dtype="uint32")
            _phase_qk_raw_full = _builder_scalar("_phase_qk_raw_full", 0, dtype="uint32")
            _phase_prep_diag_ready = _builder_scalar("_phase_prep_diag_ready", 0, dtype="uint32")
            _phase_prep_inv16_ready = _builder_scalar("_phase_prep_inv16_ready", 0, dtype="uint32")
            with T.serial(0, num_prep_iters, unroll=False) as prep_iter:
                chunk_idx_3 = _builder_scalar(
                    "chunk_idx_3", prep_iter * 5 + prep_instance, dtype="int32"
                )
                stage_f32 = _builder_scalar(
                    "stage_f32", T.cast(prep_stage, "int32") * 10496, dtype="int32"
                )
                stage_bf16 = _builder_scalar(
                    "stage_bf16", T.cast(prep_stage, "int32") * 20992, dtype="int32"
                )
                chunk_is_full_2 = _builder_scalar(
                    "chunk_is_full_2",
                    T.if_then_else(seq_len_4 >= (chunk_idx_3 + 1) * 32, 1, 0),
                    dtype="int32",
                )
                early_beta_value = _builder_scalar("early_beta_value", T.float32(0.0), dtype="f32")
                early_gate0 = _builder_scalar("early_gate0", T.float32(0.0), dtype="f32")
                _builder_if_955_12 = _builder_scope_enter(T.If(chunk_is_full_2 != 0))
                _builder_then_955_12 = _builder_scope_enter(T.Then())
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_RAW_INPUTS_FREE_OFF + prep_stage * 8,
                        _phase_raw_inputs_free,
                    )
                )
                _builder_if_962_16 = _builder_scope_enter(T.If(prep_local_warp == 0))
                _builder_then_962_16 = _builder_scope_enter(T.Then())
                _builder_if_963_20 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                _builder_then_963_20 = _builder_scope_enter(T.Then())
                _builder_emit(
                    _mbarrier_arrive_expect_tx(
                        smem_raw.ptr_to(
                            [
                                mbar_base
                                + MBAR_GATE_RAW_FULL_OFF
                                + prep_stage * 8
                                - T.cast(smem, "int32")
                            ]
                        ),
                        8704,
                    )
                )
                _builder_emit(
                    _tma_3d_gmem2smem(
                        smem_raw,
                        T.cast(smem, "int32"),
                        smem_g_raw_addr + T.cast(prep_stage, "int32") * 41984,
                        g_tma,
                        0,
                        head_idx_3,
                        T.cast(bos_4 + T.cast(chunk_idx_3 * 32, "int64"), "int32"),
                        mbar_base + MBAR_GATE_RAW_FULL_OFF + prep_stage * 8,
                    )
                )
                _builder_emit(
                    _tma_2d_gmem2smem(
                        smem_raw,
                        T.cast(smem, "int32"),
                        smem_beta_raw_addr + T.cast(prep_stage, "int32") * 41984,
                        beta_tma_tmap,
                        head_idx_3 // 8 * 8,
                        T.cast(bos_4 + T.cast(chunk_idx_3 * 32, "int64"), "int32"),
                        mbar_base + MBAR_GATE_RAW_FULL_OFF + prep_stage * 8,
                    )
                )
                _builder_emit(
                    _mbarrier_arrive_expect_tx(
                        smem_raw.ptr_to(
                            [
                                mbar_base
                                + MBAR_QK_RAW_FULL_OFF
                                + prep_stage * 8
                                - T.cast(smem, "int32")
                            ]
                        ),
                        16384,
                    )
                )
                _builder_emit(
                    _tma_4d_gmem2smem(
                        smem_raw,
                        T.cast(smem, "int32"),
                        smem_kd_addr + T.cast(prep_stage, "int32") * 41984,
                        k_tma,
                        0,
                        T.cast(bos_4 + T.cast(chunk_idx_3 * 32, "int64"), "int32"),
                        head_idx_3,
                        0,
                        mbar_base + MBAR_QK_RAW_FULL_OFF + prep_stage * 8,
                    )
                )
                _builder_scope_exit(_builder_then_963_20)
                _builder_scope_exit(_builder_if_963_20)
                _builder_scope_exit(_builder_then_962_16)
                _builder_scope_exit(_builder_if_962_16)
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_GATE_RAW_FULL_OFF + prep_stage * 8,
                        _phase_gate_raw_full,
                    )
                )
                _builder_if_1008_16 = _builder_scope_enter(
                    T.If(T.And(prep_local_warp == 2, lane < 32))
                )
                _builder_then_1008_16 = _builder_scope_enter(T.Then())
                beta_raw_pair = _builder_name(
                    "beta_raw_pair", T.alloc_local((1,), "uint32", align=4)
                )
                T.buffer_store(
                    beta_raw_pair,
                    _ld_shared_b32(
                        smem_raw,
                        T.cast(smem, "int32"),
                        smem_beta_raw_addr
                        + T.cast(prep_stage, "int32") * 41984
                        + lane * 16
                        + head_idx_3 % 8 // 2 * 4,
                    ),
                    [0],
                )
                beta_raw_pair_fp32 = _builder_name(
                    "beta_raw_pair_fp32", T.alloc_local((2,), "float32")
                )
                with T.unroll(1) as _pair:
                    T.buffer_store(
                        beta_raw_pair_fp32,
                        T.cuda.uint_as_float(beta_raw_pair[_pair + 0] << T.uint32(16)),
                        [_pair * 2],
                    )
                    T.buffer_store(
                        beta_raw_pair_fp32,
                        T.cuda.uint_as_float(beta_raw_pair[_pair + 0] & T.uint32(4294901760)),
                        [_pair * 2 + 1],
                    )
                beta_logit = _builder_scalar("beta_logit", beta_raw_pair_fp32[0], dtype="f32")
                _builder_if_1024_20 = _builder_scope_enter(T.If(head_idx_3 % 2 != 0))
                _builder_then_1024_20 = _builder_scope_enter(T.Then())
                T.buffer_store(beta_logit.buffer, beta_raw_pair_fp32[1], [0])
                _builder_scope_exit(_builder_then_1024_20)
                _builder_scope_exit(_builder_if_1024_20)
                T.buffer_store(
                    early_beta_value.buffer,
                    _tanh_approx(beta_logit * T.float32(0.5)) * T.float32(0.5) + T.float32(0.5),
                    [0],
                )
                _builder_scope_exit(_builder_then_1008_16)
                _builder_scope_exit(_builder_if_1008_16)
                _builder_if_1029_16 = _builder_scope_enter(T.If(prep_tid < 128))
                _builder_then_1029_16 = _builder_scope_enter(T.Then())
                early_gate_rate = _builder_scalar(
                    "early_gate_rate", _ld_shared_f32(smem_gate_rate_all, stage_f32), dtype="f32"
                )
                early_gate_bias = _builder_scalar(
                    "early_gate_bias",
                    _ld_global_f32(dt_bias, head_idx_3 * 128 + prep_tid),
                    dtype="f32",
                )
                early_gate_raw = _builder_scalar(
                    "early_gate_raw",
                    T.cuda.bfloat162float(_ld_shared_bf16(smem_g_raw_all, stage_bf16 + prep_tid)),
                    dtype="f32",
                )
                early_gate_arg = _builder_scalar(
                    "early_gate_arg",
                    early_gate_rate * (early_gate_raw + early_gate_bias),
                    dtype="f32",
                )
                early_gate_sigmoid = _builder_scalar(
                    "early_gate_sigmoid",
                    _tanh_approx(early_gate_arg * T.float32(0.5)) * T.float32(0.5) + T.float32(0.5),
                    dtype="f32",
                )
                T.buffer_store(
                    early_gate0.buffer,
                    T.float32(lower_bound) * T.float32(1.4426950408889634) * early_gate_sigmoid,
                    [0],
                )
                _builder_scope_exit(_builder_then_1029_16)
                _builder_scope_exit(_builder_if_1029_16)
                _builder_scope_exit(_builder_then_955_12)
                _builder_scope_exit(_builder_if_955_12)
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_SMEM_FREE_OFF + prep_stage * 8,
                        _phase_smem_free,
                    )
                )
                _builder_if_1054_12 = _builder_scope_enter(T.If(chunk_is_full_2 != 0))
                _builder_then_1054_12 = _builder_scope_enter(T.Then())
                _builder_if_1055_16 = _builder_scope_enter(T.If(prep_local_warp == 0))
                _builder_then_1055_16 = _builder_scope_enter(T.Then())
                _builder_if_1056_20 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                _builder_then_1056_20 = _builder_scope_enter(T.Then())
                _builder_emit(
                    _tma_4d_gmem2smem(
                        smem_raw,
                        T.cast(smem, "int32"),
                        smem_q_raw_prefetch_addr + T.cast(prep_stage, "int32") * 41984,
                        q_tma,
                        0,
                        T.cast(bos_4 + T.cast(chunk_idx_3 * 32, "int64"), "int32"),
                        head_idx_3,
                        0,
                        mbar_base + MBAR_QK_RAW_FULL_OFF + prep_stage * 8,
                    )
                )
                _builder_scope_exit(_builder_then_1056_20)
                _builder_scope_exit(_builder_if_1056_20)
                _builder_scope_exit(_builder_then_1055_16)
                _builder_scope_exit(_builder_if_1055_16)
                _builder_scope_exit(_builder_then_1054_12)
                _builder_scope_exit(_builder_if_1054_12)
                _builder_if_1068_12 = _builder_scope_enter(T.If(chunk_is_full_2 == 0))
                _builder_then_1068_12 = _builder_scope_enter(T.Then())
                with T.unroll(4) as gate_load_pass:
                    gate_load_item = _builder_scalar(
                        "gate_load_item", gate_load_pass * 128 + prep_tid, dtype="int32"
                    )
                    gate_load_row = _builder_scalar(
                        "gate_load_row", gate_load_item // 16, dtype="int32"
                    )
                    gate_load_segment = _builder_scalar(
                        "gate_load_segment", gate_load_item % 16, dtype="int32"
                    )
                    gate_load_token = _builder_scalar(
                        "gate_load_token",
                        bos_4 + T.cast(chunk_idx_3 * 32 + gate_load_row, "int64"),
                        dtype="int64",
                    )
                    gate_load_base = _builder_scalar(
                        "gate_load_base",
                        (gate_load_token * T.cast(h, "int64") + T.cast(head_idx_3, "int64")) * 128
                        + T.cast(gate_load_segment * 8, "int64"),
                        dtype="int64",
                    )
                    _builder_emit(
                        T.ptx["cp.async.cg.shared.global"](
                            smem_raw.ptr_to(
                                [
                                    SMEM_SMEM_G_RAW_OFF
                                    + T.cast(prep_stage, "int32") * 41984
                                    + gate_load_item * 16
                                ]
                            ),
                            g.ptr_to([gate_load_base]),
                            16,
                            T.cast(
                                T.if_then_else(
                                    T.if_then_else(gate_load_token < eos_4, 1, 0) != 0, 16, 0
                                ),
                                "uint32",
                            ),
                        )
                    )
                _builder_scope_exit(_builder_then_1068_12)
                _builder_scope_exit(_builder_if_1068_12)
                _builder_if_1085_12 = _builder_scope_enter(T.If(chunk_is_full_2 == 0))
                _builder_then_1085_12 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.cp.async_.commit_group())
                _builder_emit(T.ptx.cp.async_.wait_group(0))
                _builder_if_1088_16 = _builder_scope_enter(T.If(prep_instance == 0))
                _builder_then_1088_16 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(11), T.uint32(128)))
                _builder_scope_exit(_builder_then_1088_16)
                _builder_else_1088_16 = _builder_scope_enter(T.Else())
                _builder_if_1090_16 = _builder_scope_enter(T.If(prep_instance == 1))
                _builder_then_1090_16 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(12), T.uint32(128)))
                _builder_scope_exit(_builder_then_1090_16)
                _builder_else_1090_16 = _builder_scope_enter(T.Else())
                _builder_if_1093_20 = _builder_scope_enter(T.If(prep_instance == 2))
                _builder_then_1093_20 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(13), T.uint32(128)))
                _builder_scope_exit(_builder_then_1093_20)
                _builder_else_1093_20 = _builder_scope_enter(T.Else())
                _builder_if_1095_20 = _builder_scope_enter(T.If(prep_instance == 3))
                _builder_then_1095_20 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(14), T.uint32(128)))
                _builder_scope_exit(_builder_then_1095_20)
                _builder_else_1095_20 = _builder_scope_enter(T.Else())
                _builder_emit(T.ptx.bar.sync(T.uint32(15), T.uint32(128)))
                _builder_scope_exit(_builder_else_1095_20)
                _builder_scope_exit(_builder_if_1095_20)
                _builder_scope_exit(_builder_else_1093_20)
                _builder_scope_exit(_builder_if_1093_20)
                _builder_scope_exit(_builder_else_1090_16)
                _builder_scope_exit(_builder_if_1090_16)
                _builder_scope_exit(_builder_else_1088_16)
                _builder_scope_exit(_builder_if_1088_16)
                _builder_scope_exit(_builder_then_1085_12)
                _builder_scope_exit(_builder_if_1085_12)
                _builder_if_1099_12 = _builder_scope_enter(
                    T.If(T.And(prep_local_warp == 2, lane < 32))
                )
                _builder_then_1099_12 = _builder_scope_enter(T.Then())
                beta_value = _builder_scalar("beta_value", early_beta_value, dtype="f32")
                _builder_if_1101_16 = _builder_scope_enter(T.If(chunk_is_full_2 == 0))
                _builder_then_1101_16 = _builder_scope_enter(T.Then())
                beta_token = _builder_scalar(
                    "beta_token", bos_4 + T.cast(chunk_idx_3 * 32 + lane, "int64"), dtype="int64"
                )
                _builder_if_1103_20 = _builder_scope_enter(T.If(beta_token < eos_4))
                _builder_then_1103_20 = _builder_scope_enter(T.Then())
                beta_logit_1 = _builder_scalar(
                    "beta_logit_1",
                    T.cuda.bfloat162float(
                        _ld_global_bf16(
                            beta, beta_token * T.cast(h, "int64") + T.cast(head_idx_3, "int64")
                        )
                    ),
                    dtype="f32",
                )
                T.buffer_store(
                    beta_value.buffer,
                    _tanh_approx(beta_logit_1 * T.float32(0.5)) * T.float32(0.5) + T.float32(0.5),
                    [0],
                )
                _builder_scope_exit(_builder_then_1103_20)
                _builder_scope_exit(_builder_if_1103_20)
                _builder_scope_exit(_builder_then_1101_16)
                _builder_scope_exit(_builder_if_1101_16)
                _builder_emit(_st_shared_f32(smem_prep_beta_all, stage_f32 + lane, beta_value))
                _builder_scope_exit(_builder_then_1099_12)
                _builder_scope_exit(_builder_if_1099_12)
                _builder_if_1113_12 = _builder_scope_enter(T.If(prep_tid < 128))
                _builder_then_1113_12 = _builder_scope_enter(T.Then())
                gate_col = _builder_scalar("gate_col", prep_tid, dtype="int32")
                gate_rate = _builder_scalar(
                    "gate_rate", _ld_shared_f32(smem_gate_rate_all, stage_f32), dtype="f32"
                )
                gate_bias = _builder_scalar(
                    "gate_bias", _ld_global_f32(dt_bias, head_idx_3 * 128 + gate_col), dtype="f32"
                )
                prefix_log2 = _builder_scalar("prefix_log2", T.float32(0.0), dtype="f32")
                with T.serial(0, 32) as gate_row:
                    gate_token = _builder_scalar(
                        "gate_token",
                        bos_4 + T.cast(chunk_idx_3 * 32 + gate_row, "int64"),
                        dtype="int64",
                    )
                    gate_log2 = _builder_scalar("gate_log2", T.float32(0.0), dtype="f32")
                    gate_needs_compute = _builder_scalar("gate_needs_compute", 1, dtype="int32")
                    _builder_if_1124_20 = _builder_scope_enter(T.If(gate_row == 0))
                    _builder_then_1124_20 = _builder_scope_enter(T.Then())
                    _builder_if_1125_24 = _builder_scope_enter(T.If(chunk_is_full_2 != 0))
                    _builder_then_1125_24 = _builder_scope_enter(T.Then())
                    T.buffer_store(gate_log2.buffer, early_gate0, [0])
                    T.buffer_store(gate_needs_compute.buffer, 0, [0])
                    _builder_scope_exit(_builder_then_1125_24)
                    _builder_scope_exit(_builder_if_1125_24)
                    _builder_scope_exit(_builder_then_1124_20)
                    _builder_scope_exit(_builder_if_1124_20)
                    _builder_if_1128_20 = _builder_scope_enter(T.If(gate_needs_compute != 0))
                    _builder_then_1128_20 = _builder_scope_enter(T.Then())
                    _builder_if_1129_24 = _builder_scope_enter(T.If(gate_token < eos_4))
                    _builder_then_1129_24 = _builder_scope_enter(T.Then())
                    gate_raw = _builder_scalar(
                        "gate_raw",
                        T.cuda.bfloat162float(
                            _ld_shared_bf16(smem_g_raw_all, stage_bf16 + gate_row * 128 + gate_col)
                        ),
                        dtype="f32",
                    )
                    gate_arg = _builder_scalar(
                        "gate_arg", gate_rate * (gate_raw + gate_bias), dtype="f32"
                    )
                    gate_sigmoid = _builder_scalar(
                        "gate_sigmoid",
                        _tanh_approx(gate_arg * T.float32(0.5)) * T.float32(0.5) + T.float32(0.5),
                        dtype="f32",
                    )
                    T.buffer_store(
                        gate_log2.buffer,
                        T.float32(lower_bound) * T.float32(1.4426950408889634) * gate_sigmoid,
                        [0],
                    )
                    _builder_scope_exit(_builder_then_1129_24)
                    _builder_scope_exit(_builder_if_1129_24)
                    _builder_scope_exit(_builder_then_1128_20)
                    _builder_scope_exit(_builder_if_1128_20)
                    T.buffer_store(prefix_log2.buffer, prefix_log2 + gate_log2, [0])
                    _builder_emit(
                        _st_shared_f32(
                            smem_gate_all, stage_f32 + gate_row * 128 + gate_col, prefix_log2
                        )
                    )
                _builder_scope_exit(_builder_then_1113_12)
                _builder_scope_exit(_builder_if_1113_12)
                _builder_if_1148_12 = _builder_scope_enter(T.If(prep_instance == 0))
                _builder_then_1148_12 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(11), T.uint32(128)))
                _builder_scope_exit(_builder_then_1148_12)
                _builder_else_1148_12 = _builder_scope_enter(T.Else())
                _builder_if_1150_12 = _builder_scope_enter(T.If(prep_instance == 1))
                _builder_then_1150_12 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(12), T.uint32(128)))
                _builder_scope_exit(_builder_then_1150_12)
                _builder_else_1150_12 = _builder_scope_enter(T.Else())
                _builder_if_1153_16 = _builder_scope_enter(T.If(prep_instance == 2))
                _builder_then_1153_16 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(13), T.uint32(128)))
                _builder_scope_exit(_builder_then_1153_16)
                _builder_else_1153_16 = _builder_scope_enter(T.Else())
                _builder_if_1155_16 = _builder_scope_enter(T.If(prep_instance == 3))
                _builder_then_1155_16 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(14), T.uint32(128)))
                _builder_scope_exit(_builder_then_1155_16)
                _builder_else_1155_16 = _builder_scope_enter(T.Else())
                _builder_emit(T.ptx.bar.sync(T.uint32(15), T.uint32(128)))
                _builder_scope_exit(_builder_else_1155_16)
                _builder_scope_exit(_builder_if_1155_16)
                _builder_scope_exit(_builder_else_1153_16)
                _builder_scope_exit(_builder_if_1153_16)
                _builder_scope_exit(_builder_else_1150_12)
                _builder_scope_exit(_builder_if_1150_12)
                _builder_scope_exit(_builder_else_1148_12)
                _builder_scope_exit(_builder_if_1148_12)
                _builder_if_1159_12 = _builder_scope_enter(T.If(chunk_is_full_2 != 0))
                _builder_then_1159_12 = _builder_scope_enter(T.Then())
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_QK_RAW_FULL_OFF + prep_stage * 8,
                        _phase_qk_raw_full,
                    )
                )
                _builder_scope_exit(_builder_then_1159_12)
                _builder_scope_exit(_builder_if_1159_12)
                _builder_if_1166_12 = _builder_scope_enter(T.If(prep_tid < 128))
                _builder_then_1166_12 = _builder_scope_enter(T.Then())
                total_log2 = _builder_scalar(
                    "total_log2",
                    _ld_shared_f32(smem_gt_prefix_all, stage_f32 + prep_tid),
                    dtype="f32",
                )
                _builder_emit(
                    _st_shared_f32(
                        smem_restore_factor_all,
                        stage_f32 + prep_tid,
                        _approx_exp2(
                            total_log2
                            - T.float32(lower_bound)
                            * T.float32(1.4426950408889634)
                            * T.float32(16.0)
                        ),
                    )
                )
                _builder_scope_exit(_builder_then_1166_12)
                _builder_scope_exit(_builder_if_1166_12)
                _builder_if_1178_12 = _builder_scope_enter(T.If(prep_tid == 0))
                _builder_then_1178_12 = _builder_scope_enter(T.Then())
                _builder_emit(
                    _st_shared_f32(
                        smem_restore_factor_all,
                        stage_f32 + 128,
                        _approx_exp2(
                            T.float32(lower_bound) * T.float32(1.4426950408889634) * T.float32(16.0)
                        ),
                    )
                )
                _builder_scope_exit(_builder_then_1178_12)
                _builder_scope_exit(_builder_if_1178_12)
                q_u32 = _builder_name(
                    "q_u32", T.decl_buffer((total_tokens * h * D_HEAD // 2,), "uint32", data=q.data)
                )
                k_u32 = _builder_name(
                    "k_u32", T.decl_buffer((total_tokens * h * D_HEAD // 2,), "uint32", data=k.data)
                )
                with T.serial(0, 4, unroll=False) as work_pass:
                    work_item = _builder_scalar(
                        "work_item", work_pass * 128 + prep_tid, dtype="int32"
                    )
                    row_1 = _builder_scalar("row_1", work_item // 16, dtype="int32")
                    segment_1 = _builder_scalar("segment_1", work_item % 16, dtype="int32")
                    token_1 = _builder_scalar(
                        "token_1", bos_4 + T.cast(chunk_idx_3 * 32 + row_1, "int64"), dtype="int64"
                    )
                    token_valid_1 = _builder_scalar(
                        "token_valid_1", T.if_then_else(token_1 < eos_4, 1, 0), dtype="int32"
                    )
                    gmem_base = _builder_scalar(
                        "gmem_base",
                        (token_1 * T.cast(h, "int64") + T.cast(head_idx_3, "int64")) * 128
                        + T.cast(segment_1 * 8, "int64"),
                        dtype="int64",
                    )
                    q_raw_vec = _builder_name("q_raw_vec", T.alloc_local((8,), "float32"))
                    k_raw_vec = _builder_name("k_raw_vec", T.alloc_local((8,), "float32"))
                    with T.unroll(8) as _zi:
                        T.buffer_store(q_raw_vec, T.float32(0.0), [_zi])
                        T.buffer_store(k_raw_vec, T.float32(0.0), [_zi])
                    _builder_if_1204_16 = _builder_scope_enter(T.If(chunk_is_full_2 != 0))
                    _builder_then_1204_16 = _builder_scope_enter(T.Then())
                    packed = _builder_name("packed", T.alloc_local((4,), "uint32", align=16))
                    _builder_emit(
                        _ld_shared_v4(
                            smem_raw,
                            T.cast(smem, "int32"),
                            packed,
                            smem_q_raw_prefetch_addr
                            + T.cast(prep_stage, "int32") * 41984
                            + (
                                segment_1 * 8 // 64 * 4096 + row_1 * 128 + segment_1 * 8 % 64 * 2
                                ^ (
                                    segment_1 * 8 // 64 * 4096
                                    + row_1 * 128
                                    + segment_1 * 8 % 64 * 2
                                    >> 7
                                    & 7
                                )
                                << 4
                            ),
                        )
                    )
                    packed_fp32 = _builder_name("packed_fp32", T.alloc_local((8,), "float32"))
                    with T.unroll(4) as _pair:
                        T.buffer_store(
                            packed_fp32,
                            T.cuda.uint_as_float(packed[_pair + 0] << T.uint32(16)),
                            [_pair * 2],
                        )
                        T.buffer_store(
                            packed_fp32,
                            T.cuda.uint_as_float(packed[_pair + 0] & T.uint32(4294901760)),
                            [_pair * 2 + 1],
                        )
                    with T.unroll(8) as value_idx:
                        T.buffer_store(q_raw_vec, packed_fp32[value_idx], [value_idx])
                    packed_0 = _builder_name("packed_0", T.alloc_local((4,), "uint32", align=16))
                    _builder_emit(
                        _ld_shared_v4(
                            smem_raw,
                            T.cast(smem, "int32"),
                            packed_0,
                            smem_kd_addr
                            + T.cast(prep_stage, "int32") * 41984
                            + (
                                segment_1 * 8 // 64 * 4096 + row_1 * 128 + segment_1 * 8 % 64 * 2
                                ^ (
                                    segment_1 * 8 // 64 * 4096
                                    + row_1 * 128
                                    + segment_1 * 8 % 64 * 2
                                    >> 7
                                    & 7
                                )
                                << 4
                            ),
                        )
                    )
                    packed_0_fp32 = _builder_name("packed_0_fp32", T.alloc_local((8,), "float32"))
                    with T.unroll(4) as _pair:
                        T.buffer_store(
                            packed_0_fp32,
                            T.cuda.uint_as_float(packed_0[_pair + 0] << T.uint32(16)),
                            [_pair * 2],
                        )
                        T.buffer_store(
                            packed_0_fp32,
                            T.cuda.uint_as_float(packed_0[_pair + 0] & T.uint32(4294901760)),
                            [_pair * 2 + 1],
                        )
                    with T.unroll(8) as value_idx_1:
                        T.buffer_store(k_raw_vec, packed_0_fp32[value_idx_1], [value_idx_1])
                    _builder_scope_exit(_builder_then_1204_16)
                    _builder_else_1204_16 = _builder_scope_enter(T.Else())
                    _builder_if_1239_16 = _builder_scope_enter(T.If(token_valid_1 != 0))
                    _builder_then_1239_16 = _builder_scope_enter(T.Then())
                    with T.unroll(1) as _blk:
                        _vldq = _builder_name("_vldq", T.alloc_local((4,), "uint32", align=16))
                        _builder_emit(
                            _ld_global_v4_u32(_vldq, q_u32.ptr_to([gmem_base // 2 + _blk * 4]))
                        )
                        with T.unroll(4) as _pair:
                            T.buffer_store(
                                q_raw_vec,
                                T.cuda.uint_as_float(_vldq[_pair] << T.uint32(16)),
                                [0 + _blk * 8 + _pair * 2],
                            )
                            T.buffer_store(
                                q_raw_vec,
                                T.cuda.uint_as_float(_vldq[_pair] & T.uint32(4294901760)),
                                [0 + _blk * 8 + _pair * 2 + 1],
                            )
                    with T.unroll(1) as _blk:
                        _vldk = _builder_name("_vldk", T.alloc_local((4,), "uint32", align=16))
                        _builder_emit(
                            _ld_global_v4_u32(_vldk, k_u32.ptr_to([gmem_base // 2 + _blk * 4]))
                        )
                        with T.unroll(4) as _pair:
                            T.buffer_store(
                                k_raw_vec,
                                T.cuda.uint_as_float(_vldk[_pair] << T.uint32(16)),
                                [0 + _blk * 8 + _pair * 2],
                            )
                            T.buffer_store(
                                k_raw_vec,
                                T.cuda.uint_as_float(_vldk[_pair] & T.uint32(4294901760)),
                                [0 + _blk * 8 + _pair * 2 + 1],
                            )
                    _builder_scope_exit(_builder_then_1239_16)
                    _builder_scope_exit(_builder_if_1239_16)
                    _builder_scope_exit(_builder_else_1204_16)
                    _builder_scope_exit(_builder_if_1204_16)
                    q_sum = _builder_scalar("q_sum", T.float32(0.0), dtype="f32")
                    k_sum = _builder_scalar("k_sum", T.float32(0.0), dtype="f32")
                    with T.serial(0, 8) as elem_in_segment:
                        T.buffer_store(
                            q_sum.buffer,
                            _fmaf_rn(q_raw_vec[elem_in_segment], q_raw_vec[elem_in_segment], q_sum),
                            [0],
                        )
                        T.buffer_store(
                            k_sum.buffer,
                            _fmaf_rn(k_raw_vec[elem_in_segment], k_raw_vec[elem_in_segment], k_sum),
                            [0],
                        )
                    T.buffer_store(
                        q_sum.buffer,
                        q_sum + T.cuda._shfl_xor_sync(T.uint32(4294967295), q_sum, 8, 32),
                        [0],
                    )
                    T.buffer_store(
                        k_sum.buffer,
                        k_sum + T.cuda._shfl_xor_sync(T.uint32(4294967295), k_sum, 8, 32),
                        [0],
                    )
                    T.buffer_store(
                        q_sum.buffer,
                        q_sum + T.cuda._shfl_xor_sync(T.uint32(4294967295), q_sum, 4, 32),
                        [0],
                    )
                    T.buffer_store(
                        k_sum.buffer,
                        k_sum + T.cuda._shfl_xor_sync(T.uint32(4294967295), k_sum, 4, 32),
                        [0],
                    )
                    T.buffer_store(
                        q_sum.buffer,
                        q_sum + T.cuda._shfl_xor_sync(T.uint32(4294967295), q_sum, 2, 32),
                        [0],
                    )
                    T.buffer_store(
                        k_sum.buffer,
                        k_sum + T.cuda._shfl_xor_sync(T.uint32(4294967295), k_sum, 2, 32),
                        [0],
                    )
                    T.buffer_store(
                        q_sum.buffer,
                        q_sum + T.cuda._shfl_xor_sync(T.uint32(4294967295), q_sum, 1, 32),
                        [0],
                    )
                    T.buffer_store(
                        k_sum.buffer,
                        k_sum + T.cuda._shfl_xor_sync(T.uint32(4294967295), k_sum, 1, 32),
                        [0],
                    )
                    q_inv = _builder_scalar("q_inv", _rsqrtf(q_sum + T.float32(1e-06)), dtype="f32")
                    k_inv = _builder_scalar("k_inv", _rsqrtf(k_sum + T.float32(1e-06)), dtype="f32")
                    with T.unroll(4) as _ls:
                        _pk = _builder_scalar(
                            "_pk",
                            _mul_f32x2_inplace(
                                T.cuda.make_float2(q_raw_vec[_ls * 2], q_raw_vec[_ls * 2 + 1]),
                                T.cuda.make_float2(q_inv, q_inv),
                            ),
                        )
                        T.buffer_store(q_raw_vec, T.cuda.float2_x(_pk), [_ls * 2])
                        T.buffer_store(q_raw_vec, T.cuda.float2_y(_pk), [_ls * 2 + 1])
                    with T.unroll(4) as _ls:
                        _pk = _builder_scalar(
                            "_pk",
                            _mul_f32x2_inplace(
                                T.cuda.make_float2(k_raw_vec[_ls * 2], k_raw_vec[_ls * 2 + 1]),
                                T.cuda.make_float2(k_inv, k_inv),
                            ),
                        )
                        T.buffer_store(k_raw_vec, T.cuda.float2_x(_pk), [_ls * 2])
                        T.buffer_store(k_raw_vec, T.cuda.float2_y(_pk), [_ls * 2 + 1])
                    qd_vec = _builder_name("qd_vec", T.alloc_local((8,), "float32"))
                    kd_vec = _builder_name("kd_vec", T.alloc_local((8,), "float32"))
                    ki_vec = _builder_name("ki_vec", T.alloc_local((8,), "float32"))
                    prefix_vec = _builder_name("prefix_vec", T.alloc_local((8,), "float32"))
                    with T.unroll(2) as prefix_vec_idx:
                        _builder_emit(
                            _ld_shared_v4_f32(
                                prefix_vec,
                                prefix_vec_idx * 4,
                                smem_gate_all,
                                stage_f32 + row_1 * 128 + segment_1 * 8 + prefix_vec_idx * 4,
                            )
                        )
                    with T.serial(0, 8) as elem_in_segment_1:
                        col = _builder_scalar(
                            "col", segment_1 * 8 + elem_in_segment_1, dtype="int32"
                        )
                        prefix = _builder_scalar(
                            "prefix", prefix_vec[elem_in_segment_1], dtype="f32"
                        )
                        common_log2 = _builder_scalar(
                            "common_log2",
                            T.float32(lower_bound)
                            * T.float32(1.4426950408889634)
                            * T.float32(16.0),
                            dtype="f32",
                        )
                        decay = _builder_scalar(
                            "decay", _approx_exp2(prefix - common_log2), dtype="f32"
                        )
                        T.buffer_store(qd_vec, decay, [elem_in_segment_1])
                        T.buffer_store(kd_vec, decay, [elem_in_segment_1])
                        T.buffer_store(
                            ki_vec,
                            T.cuda.fdividef(k_raw_vec[elem_in_segment_1], decay),
                            [elem_in_segment_1],
                        )
                    with T.unroll(4) as _ls:
                        _pk = _builder_scalar(
                            "_pk",
                            _mul_f32x2_inplace(
                                T.cuda.make_float2(qd_vec[_ls * 2], qd_vec[_ls * 2 + 1]),
                                T.cuda.make_float2(q_raw_vec[_ls * 2], q_raw_vec[_ls * 2 + 1]),
                            ),
                        )
                        T.buffer_store(qd_vec, T.cuda.float2_x(_pk), [_ls * 2])
                        T.buffer_store(qd_vec, T.cuda.float2_y(_pk), [_ls * 2 + 1])
                    with T.unroll(4) as _ls:
                        _pk = _builder_scalar(
                            "_pk",
                            _mul_f32x2_inplace(
                                T.cuda.make_float2(qd_vec[_ls * 2], qd_vec[_ls * 2 + 1]),
                                T.cuda.make_float2(T.float32(scale), T.float32(scale)),
                            ),
                        )
                        T.buffer_store(qd_vec, T.cuda.float2_x(_pk), [_ls * 2])
                        T.buffer_store(qd_vec, T.cuda.float2_y(_pk), [_ls * 2 + 1])
                    with T.unroll(4) as _ls:
                        _pk = _builder_scalar(
                            "_pk",
                            _mul_f32x2_inplace(
                                T.cuda.make_float2(kd_vec[_ls * 2], kd_vec[_ls * 2 + 1]),
                                T.cuda.make_float2(k_raw_vec[_ls * 2], k_raw_vec[_ls * 2 + 1]),
                            ),
                        )
                        T.buffer_store(kd_vec, T.cuda.float2_x(_pk), [_ls * 2])
                        T.buffer_store(kd_vec, T.cuda.float2_y(_pk), [_ls * 2 + 1])
                    packed_1 = _builder_name("packed_1", T.alloc_local((4,), "uint32", align=4))
                    with T.unroll(4) as _lp:
                        T.buffer_store(
                            packed_1,
                            T.cuda.float22bfloat162_rn(
                                qd_vec[_lp * 2 + 0], qd_vec[_lp * 2 + 1 + 0]
                            ),
                            [_lp],
                        )
                    with T.unroll(4) as word:
                        _builder_emit(
                            _st_shared_b32(
                                smem_raw,
                                T.cast(smem, "int32"),
                                smem_qd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    segment_1 * 8 // 64 * 4096
                                    + row_1 * 128
                                    + segment_1 * 8 % 64 * 2
                                    ^ (
                                        segment_1 * 8 // 64 * 4096
                                        + row_1 * 128
                                        + segment_1 * 8 % 64 * 2
                                        >> 7
                                        & 7
                                    )
                                    << 4
                                )
                                + word * 4,
                                packed_1[word],
                            )
                        )
                    packed_0_1 = _builder_name("packed_0_1", T.alloc_local((4,), "uint32", align=4))
                    with T.unroll(4) as _lp:
                        T.buffer_store(
                            packed_0_1,
                            T.cuda.float22bfloat162_rn(
                                kd_vec[_lp * 2 + 0], kd_vec[_lp * 2 + 1 + 0]
                            ),
                            [_lp],
                        )
                    with T.unroll(4) as word_1:
                        _builder_emit(
                            _st_shared_b32(
                                smem_raw,
                                T.cast(smem, "int32"),
                                smem_kd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    segment_1 * 8 // 64 * 4096
                                    + row_1 * 128
                                    + segment_1 * 8 % 64 * 2
                                    ^ (
                                        segment_1 * 8 // 64 * 4096
                                        + row_1 * 128
                                        + segment_1 * 8 % 64 * 2
                                        >> 7
                                        & 7
                                    )
                                    << 4
                                )
                                + word_1 * 4,
                                packed_0_1[word_1],
                            )
                        )
                    packed_1_1 = _builder_name("packed_1_1", T.alloc_local((4,), "uint32", align=4))
                    with T.unroll(4) as _lp:
                        T.buffer_store(
                            packed_1_1,
                            T.cuda.float22bfloat162_rn(
                                ki_vec[_lp * 2 + 0], ki_vec[_lp * 2 + 1 + 0]
                            ),
                            [_lp],
                        )
                    with T.unroll(4) as word_2:
                        _builder_emit(
                            _st_shared_b32(
                                smem_raw,
                                T.cast(smem, "int32"),
                                smem_ki_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    segment_1 * 8 // 64 * 4096
                                    + row_1 * 128
                                    + segment_1 * 8 % 64 * 2
                                    ^ (
                                        segment_1 * 8 // 64 * 4096
                                        + row_1 * 128
                                        + segment_1 * 8 % 64 * 2
                                        >> 7
                                        & 7
                                    )
                                    << 4
                                )
                                + word_2 * 4,
                                packed_1_1[word_2],
                            )
                        )
                _builder_if_1373_12 = _builder_scope_enter(T.If(prep_instance == 0))
                _builder_then_1373_12 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(11), T.uint32(128)))
                _builder_scope_exit(_builder_then_1373_12)
                _builder_else_1373_12 = _builder_scope_enter(T.Else())
                _builder_if_1375_12 = _builder_scope_enter(T.If(prep_instance == 1))
                _builder_then_1375_12 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(12), T.uint32(128)))
                _builder_scope_exit(_builder_then_1375_12)
                _builder_else_1375_12 = _builder_scope_enter(T.Else())
                _builder_if_1378_16 = _builder_scope_enter(T.If(prep_instance == 2))
                _builder_then_1378_16 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(13), T.uint32(128)))
                _builder_scope_exit(_builder_then_1378_16)
                _builder_else_1378_16 = _builder_scope_enter(T.Else())
                _builder_if_1380_16 = _builder_scope_enter(T.If(prep_instance == 3))
                _builder_then_1380_16 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(14), T.uint32(128)))
                _builder_scope_exit(_builder_then_1380_16)
                _builder_else_1380_16 = _builder_scope_enter(T.Else())
                _builder_emit(T.ptx.bar.sync(T.uint32(15), T.uint32(128)))
                _builder_scope_exit(_builder_else_1380_16)
                _builder_scope_exit(_builder_if_1380_16)
                _builder_scope_exit(_builder_else_1378_16)
                _builder_scope_exit(_builder_if_1378_16)
                _builder_scope_exit(_builder_else_1375_12)
                _builder_scope_exit(_builder_if_1375_12)
                _builder_scope_exit(_builder_else_1373_12)
                _builder_scope_exit(_builder_if_1373_12)
                pair_row_base = _builder_scalar(
                    "pair_row_base", prep_local_warp // 2 * 16, dtype="int32"
                )
                pair_col_base = _builder_scalar(
                    "pair_col_base", prep_local_warp % 2 * 16, dtype="int32"
                )
                a_frag = _builder_name("a_frag", T.alloc_local((4,), "uint32", align=4))
                b_frag = _builder_name("b_frag", T.alloc_local((4,), "uint32", align=4))
                acc = _builder_name("acc", T.alloc_local((8,), "float32", align=4))
                _builder_if_1389_12 = _builder_scope_enter(T.If(pair_row_base >= pair_col_base))
                _builder_then_1389_12 = _builder_scope_enter(T.Then())
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_kd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    lane // 16 // 8 * 256
                                    + (pair_row_base + lane % 16) * 8
                                    + (lane // 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4)
                                    // 16
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        b_frag[0],
                        b_frag[1],
                        b_frag[2],
                        b_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_ki_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    lane % 16 // 8 // 8 * 256
                                    + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8
                                    + (
                                        lane % 16 // 8 % 8 * 16
                                        ^ (pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4
                                    )
                                    // 16
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(_mma_m16n8k16_bf16_zero(acc, a_frag, b_frag))
                _builder_emit(_mma_m16n8k16_bf16_zero_off4(acc, a_frag, b_frag))
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_kd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    lane // 16 // 8 * 256
                                    + (pair_row_base + lane % 16) * 8
                                    + (lane // 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4)
                                    // 16
                                    ^ 2
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        b_frag[0],
                        b_frag[1],
                        b_frag[2],
                        b_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_ki_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        lane % 16 // 8 // 8 * 256
                                        + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8
                                        + (
                                            lane % 16 // 8 % 8 * 16
                                            ^ (pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4
                                        )
                                        // 16
                                        + 256
                                        ^ 2
                                    )
                                    - 256
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(_mma_m16n8k16_bf16_acc(acc, a_frag, b_frag))
                _builder_emit(_mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag))
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_kd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    lane // 16 // 8 * 256
                                    + (pair_row_base + lane % 16) * 8
                                    + (lane // 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4)
                                    // 16
                                    ^ 2
                                    ^ 6
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        b_frag[0],
                        b_frag[1],
                        b_frag[2],
                        b_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_ki_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        (
                                            lane % 16 // 8 // 8 * 256
                                            + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8
                                            + (
                                                lane % 16 // 8 % 8 * 16
                                                ^ (pair_col_base + 8 * (lane // 16) + lane % 8 & 7)
                                                << 4
                                            )
                                            // 16
                                            + 256
                                            ^ 2
                                        )
                                        - 256
                                        + 256
                                        ^ 6
                                    )
                                    - 256
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(_mma_m16n8k16_bf16_acc(acc, a_frag, b_frag))
                _builder_emit(_mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag))
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_kd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    lane // 16 // 8 * 256
                                    + (pair_row_base + lane % 16) * 8
                                    + (lane // 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4)
                                    // 16
                                    ^ 2
                                    ^ 6
                                    ^ 2
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        b_frag[0],
                        b_frag[1],
                        b_frag[2],
                        b_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_ki_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        (
                                            (
                                                lane % 16 // 8 // 8 * 256
                                                + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8
                                                + (
                                                    lane % 16 // 8 % 8 * 16
                                                    ^ (
                                                        pair_col_base + 8 * (lane // 16) + lane % 8
                                                        & 7
                                                    )
                                                    << 4
                                                )
                                                // 16
                                                + 256
                                                ^ 2
                                            )
                                            - 256
                                            + 256
                                            ^ 6
                                        )
                                        - 256
                                        + 256
                                        ^ 2
                                    )
                                    - 256
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(_mma_m16n8k16_bf16_acc(acc, a_frag, b_frag))
                _builder_emit(_mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag))
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_kd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        lane // 16 // 8 * 256
                                        + (pair_row_base + lane % 16) * 8
                                        + (
                                            lane // 16 % 8 * 16
                                            ^ (pair_row_base + lane % 16 & 7) << 4
                                        )
                                        // 16
                                        ^ 2
                                        ^ 6
                                        ^ 2
                                        ^ 6
                                    )
                                    + 256
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        b_frag[0],
                        b_frag[1],
                        b_frag[2],
                        b_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_ki_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        (
                                            (
                                                (
                                                    lane % 16 // 8 // 8 * 256
                                                    + (pair_col_base + 8 * (lane // 16) + lane % 8)
                                                    * 8
                                                    + (
                                                        lane % 16 // 8 % 8 * 16
                                                        ^ (
                                                            pair_col_base
                                                            + 8 * (lane // 16)
                                                            + lane % 8
                                                            & 7
                                                        )
                                                        << 4
                                                    )
                                                    // 16
                                                    + 256
                                                    ^ 2
                                                )
                                                - 256
                                                + 256
                                                ^ 6
                                            )
                                            - 256
                                            + 256
                                            ^ 2
                                        )
                                        - 256
                                        + 256
                                        ^ 6
                                    )
                                    + 256
                                    - 256
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(_mma_m16n8k16_bf16_acc(acc, a_frag, b_frag))
                _builder_emit(_mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag))
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_kd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        lane // 16 // 8 * 256
                                        + (pair_row_base + lane % 16) * 8
                                        + (
                                            lane // 16 % 8 * 16
                                            ^ (pair_row_base + lane % 16 & 7) << 4
                                        )
                                        // 16
                                        ^ 2
                                        ^ 6
                                        ^ 2
                                        ^ 6
                                    )
                                    + 256
                                    ^ 2
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        b_frag[0],
                        b_frag[1],
                        b_frag[2],
                        b_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_ki_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        lane % 16 // 8 // 8 * 256
                                                        + (
                                                            pair_col_base
                                                            + 8 * (lane // 16)
                                                            + lane % 8
                                                        )
                                                        * 8
                                                        + (
                                                            lane % 16 // 8 % 8 * 16
                                                            ^ (
                                                                pair_col_base
                                                                + 8 * (lane // 16)
                                                                + lane % 8
                                                                & 7
                                                            )
                                                            << 4
                                                        )
                                                        // 16
                                                        + 256
                                                        ^ 2
                                                    )
                                                    - 256
                                                    + 256
                                                    ^ 6
                                                )
                                                - 256
                                                + 256
                                                ^ 2
                                            )
                                            - 256
                                            + 256
                                            ^ 6
                                        )
                                        + 256
                                        - 256
                                        + 256
                                        ^ 2
                                    )
                                    - 256
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(_mma_m16n8k16_bf16_acc(acc, a_frag, b_frag))
                _builder_emit(_mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag))
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_kd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        lane // 16 // 8 * 256
                                        + (pair_row_base + lane % 16) * 8
                                        + (
                                            lane // 16 % 8 * 16
                                            ^ (pair_row_base + lane % 16 & 7) << 4
                                        )
                                        // 16
                                        ^ 2
                                        ^ 6
                                        ^ 2
                                        ^ 6
                                    )
                                    + 256
                                    ^ 2
                                    ^ 6
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        b_frag[0],
                        b_frag[1],
                        b_frag[2],
                        b_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_ki_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            lane % 16 // 8 // 8 * 256
                                                            + (
                                                                pair_col_base
                                                                + 8 * (lane // 16)
                                                                + lane % 8
                                                            )
                                                            * 8
                                                            + (
                                                                lane % 16 // 8 % 8 * 16
                                                                ^ (
                                                                    pair_col_base
                                                                    + 8 * (lane // 16)
                                                                    + lane % 8
                                                                    & 7
                                                                )
                                                                << 4
                                                            )
                                                            // 16
                                                            + 256
                                                            ^ 2
                                                        )
                                                        - 256
                                                        + 256
                                                        ^ 6
                                                    )
                                                    - 256
                                                    + 256
                                                    ^ 2
                                                )
                                                - 256
                                                + 256
                                                ^ 6
                                            )
                                            + 256
                                            - 256
                                            + 256
                                            ^ 2
                                        )
                                        - 256
                                        + 256
                                        ^ 6
                                    )
                                    - 256
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(_mma_m16n8k16_bf16_acc(acc, a_frag, b_frag))
                _builder_emit(_mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag))
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_kd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        lane // 16 // 8 * 256
                                        + (pair_row_base + lane % 16) * 8
                                        + (
                                            lane // 16 % 8 * 16
                                            ^ (pair_row_base + lane % 16 & 7) << 4
                                        )
                                        // 16
                                        ^ 2
                                        ^ 6
                                        ^ 2
                                        ^ 6
                                    )
                                    + 256
                                    ^ 2
                                    ^ 6
                                    ^ 2
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        b_frag[0],
                        b_frag[1],
                        b_frag[2],
                        b_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_ki_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                lane % 16 // 8 // 8 * 256
                                                                + (
                                                                    pair_col_base
                                                                    + 8 * (lane // 16)
                                                                    + lane % 8
                                                                )
                                                                * 8
                                                                + (
                                                                    lane % 16 // 8 % 8 * 16
                                                                    ^ (
                                                                        pair_col_base
                                                                        + 8 * (lane // 16)
                                                                        + lane % 8
                                                                        & 7
                                                                    )
                                                                    << 4
                                                                )
                                                                // 16
                                                                + 256
                                                                ^ 2
                                                            )
                                                            - 256
                                                            + 256
                                                            ^ 6
                                                        )
                                                        - 256
                                                        + 256
                                                        ^ 2
                                                    )
                                                    - 256
                                                    + 256
                                                    ^ 6
                                                )
                                                + 256
                                                - 256
                                                + 256
                                                ^ 2
                                            )
                                            - 256
                                            + 256
                                            ^ 6
                                        )
                                        - 256
                                        + 256
                                        ^ 2
                                    )
                                    - 256
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(_mma_m16n8k16_bf16_acc(acc, a_frag, b_frag))
                _builder_emit(_mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag))
                row0 = _builder_scalar("row0", pair_row_base + lane // 4, dtype="int32")
                row1 = _builder_scalar("row1", row0 + 8, dtype="int32")
                col0 = _builder_scalar("col0", pair_col_base + lane % 4 * 2, dtype="int32")
                beta0 = _builder_scalar(
                    "beta0", _ld_shared_f32(smem_prep_beta_all, stage_f32 + row0), dtype="f32"
                )
                beta1 = _builder_scalar(
                    "beta1", _ld_shared_f32(smem_prep_beta_all, stage_f32 + row1), dtype="f32"
                )
                seed = _builder_name("seed", T.alloc_local((8,), "float32"))
                with T.unroll(8) as _zi:
                    T.buffer_store(seed, T.float32(0.0), [_zi])
                _builder_if_1534_16 = _builder_scope_enter(T.If(row0 > col0))
                _builder_then_1534_16 = _builder_scope_enter(T.Then())
                T.buffer_store(seed, acc[0] * beta0, [0])
                _builder_scope_exit(_builder_then_1534_16)
                _builder_scope_exit(_builder_if_1534_16)
                _builder_if_1536_16 = _builder_scope_enter(T.If(row0 > col0 + 1))
                _builder_then_1536_16 = _builder_scope_enter(T.Then())
                T.buffer_store(seed, acc[1] * beta0, [1])
                _builder_scope_exit(_builder_then_1536_16)
                _builder_scope_exit(_builder_if_1536_16)
                _builder_if_1538_16 = _builder_scope_enter(T.If(row1 > col0))
                _builder_then_1538_16 = _builder_scope_enter(T.Then())
                T.buffer_store(seed, acc[2] * beta1, [2])
                _builder_scope_exit(_builder_then_1538_16)
                _builder_scope_exit(_builder_if_1538_16)
                _builder_if_1540_16 = _builder_scope_enter(T.If(row1 > col0 + 1))
                _builder_then_1540_16 = _builder_scope_enter(T.Then())
                T.buffer_store(seed, acc[3] * beta1, [3])
                _builder_scope_exit(_builder_then_1540_16)
                _builder_scope_exit(_builder_if_1540_16)
                _builder_if_1542_16 = _builder_scope_enter(T.If(row0 > col0 + 8))
                _builder_then_1542_16 = _builder_scope_enter(T.Then())
                T.buffer_store(seed, acc[4] * beta0, [4])
                _builder_scope_exit(_builder_then_1542_16)
                _builder_scope_exit(_builder_if_1542_16)
                _builder_if_1544_16 = _builder_scope_enter(T.If(row0 > col0 + 9))
                _builder_then_1544_16 = _builder_scope_enter(T.Then())
                T.buffer_store(seed, acc[5] * beta0, [5])
                _builder_scope_exit(_builder_then_1544_16)
                _builder_scope_exit(_builder_if_1544_16)
                _builder_if_1546_16 = _builder_scope_enter(T.If(row1 > col0 + 8))
                _builder_then_1546_16 = _builder_scope_enter(T.Then())
                T.buffer_store(seed, acc[6] * beta1, [6])
                _builder_scope_exit(_builder_then_1546_16)
                _builder_scope_exit(_builder_if_1546_16)
                _builder_if_1548_16 = _builder_scope_enter(T.If(row1 > col0 + 9))
                _builder_then_1548_16 = _builder_scope_enter(T.Then())
                T.buffer_store(seed, acc[7] * beta1, [7])
                _builder_scope_exit(_builder_then_1548_16)
                _builder_scope_exit(_builder_if_1548_16)
                seed_packed = _builder_name("seed_packed", T.alloc_local((4,), "uint32", align=4))
                with T.unroll(4) as _lp:
                    T.buffer_store(
                        seed_packed,
                        T.cuda.float22bfloat162_rn(seed[_lp * 2 + 0], seed[_lp * 2 + 1 + 0]),
                        [_lp],
                    )
                seed_lane_row = _builder_scalar("seed_lane_row", lane % 16, dtype="int32")
                seed_lane_col = _builder_scalar("seed_lane_col", lane // 16 * 8, dtype="int32")
                byte_off = _builder_scalar(
                    "byte_off",
                    (pair_row_base + seed_lane_row) * 128 + (pair_col_base + seed_lane_col) * 2,
                    dtype="int32",
                )
                swizzled_off = _builder_scalar(
                    "swizzled_off", byte_off ^ (byte_off >> 7 & 7) << 4, dtype="int32"
                )
                seed_addr = _builder_scalar(
                    "seed_addr",
                    smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off,
                    dtype="int32",
                )
                _builder_emit(
                    T.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                        smem_raw.ptr_to([seed_addr - T.cast(smem, "int32")]),
                        seed_packed[0],
                        seed_packed[1],
                        seed_packed[2],
                        seed_packed[3],
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_qd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    lane // 16 // 8 * 256
                                    + (pair_row_base + lane % 16) * 8
                                    + (lane // 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4)
                                    // 16
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        b_frag[0],
                        b_frag[1],
                        b_frag[2],
                        b_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_ki_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    lane % 16 // 8 // 8 * 256
                                    + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8
                                    + (
                                        lane % 16 // 8 % 8 * 16
                                        ^ (pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4
                                    )
                                    // 16
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(_mma_m16n8k16_bf16_zero(acc, a_frag, b_frag))
                _builder_emit(_mma_m16n8k16_bf16_zero_off4(acc, a_frag, b_frag))
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_qd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    lane // 16 // 8 * 256
                                    + (pair_row_base + lane % 16) * 8
                                    + (lane // 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4)
                                    // 16
                                    ^ 2
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        b_frag[0],
                        b_frag[1],
                        b_frag[2],
                        b_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_ki_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        lane % 16 // 8 // 8 * 256
                                        + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8
                                        + (
                                            lane % 16 // 8 % 8 * 16
                                            ^ (pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4
                                        )
                                        // 16
                                        + 256
                                        ^ 2
                                    )
                                    - 256
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(_mma_m16n8k16_bf16_acc(acc, a_frag, b_frag))
                _builder_emit(_mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag))
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_qd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    lane // 16 // 8 * 256
                                    + (pair_row_base + lane % 16) * 8
                                    + (lane // 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4)
                                    // 16
                                    ^ 2
                                    ^ 6
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        b_frag[0],
                        b_frag[1],
                        b_frag[2],
                        b_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_ki_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        (
                                            lane % 16 // 8 // 8 * 256
                                            + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8
                                            + (
                                                lane % 16 // 8 % 8 * 16
                                                ^ (pair_col_base + 8 * (lane // 16) + lane % 8 & 7)
                                                << 4
                                            )
                                            // 16
                                            + 256
                                            ^ 2
                                        )
                                        - 256
                                        + 256
                                        ^ 6
                                    )
                                    - 256
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(_mma_m16n8k16_bf16_acc(acc, a_frag, b_frag))
                _builder_emit(_mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag))
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_qd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    lane // 16 // 8 * 256
                                    + (pair_row_base + lane % 16) * 8
                                    + (lane // 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4)
                                    // 16
                                    ^ 2
                                    ^ 6
                                    ^ 2
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        b_frag[0],
                        b_frag[1],
                        b_frag[2],
                        b_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_ki_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        (
                                            (
                                                lane % 16 // 8 // 8 * 256
                                                + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8
                                                + (
                                                    lane % 16 // 8 % 8 * 16
                                                    ^ (
                                                        pair_col_base + 8 * (lane // 16) + lane % 8
                                                        & 7
                                                    )
                                                    << 4
                                                )
                                                // 16
                                                + 256
                                                ^ 2
                                            )
                                            - 256
                                            + 256
                                            ^ 6
                                        )
                                        - 256
                                        + 256
                                        ^ 2
                                    )
                                    - 256
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(_mma_m16n8k16_bf16_acc(acc, a_frag, b_frag))
                _builder_emit(_mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag))
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_qd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        lane // 16 // 8 * 256
                                        + (pair_row_base + lane % 16) * 8
                                        + (
                                            lane // 16 % 8 * 16
                                            ^ (pair_row_base + lane % 16 & 7) << 4
                                        )
                                        // 16
                                        ^ 2
                                        ^ 6
                                        ^ 2
                                        ^ 6
                                    )
                                    + 256
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        b_frag[0],
                        b_frag[1],
                        b_frag[2],
                        b_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_ki_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        (
                                            (
                                                (
                                                    lane % 16 // 8 // 8 * 256
                                                    + (pair_col_base + 8 * (lane // 16) + lane % 8)
                                                    * 8
                                                    + (
                                                        lane % 16 // 8 % 8 * 16
                                                        ^ (
                                                            pair_col_base
                                                            + 8 * (lane // 16)
                                                            + lane % 8
                                                            & 7
                                                        )
                                                        << 4
                                                    )
                                                    // 16
                                                    + 256
                                                    ^ 2
                                                )
                                                - 256
                                                + 256
                                                ^ 6
                                            )
                                            - 256
                                            + 256
                                            ^ 2
                                        )
                                        - 256
                                        + 256
                                        ^ 6
                                    )
                                    + 256
                                    - 256
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(_mma_m16n8k16_bf16_acc(acc, a_frag, b_frag))
                _builder_emit(_mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag))
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_qd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        lane // 16 // 8 * 256
                                        + (pair_row_base + lane % 16) * 8
                                        + (
                                            lane // 16 % 8 * 16
                                            ^ (pair_row_base + lane % 16 & 7) << 4
                                        )
                                        // 16
                                        ^ 2
                                        ^ 6
                                        ^ 2
                                        ^ 6
                                    )
                                    + 256
                                    ^ 2
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        b_frag[0],
                        b_frag[1],
                        b_frag[2],
                        b_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_ki_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        lane % 16 // 8 // 8 * 256
                                                        + (
                                                            pair_col_base
                                                            + 8 * (lane // 16)
                                                            + lane % 8
                                                        )
                                                        * 8
                                                        + (
                                                            lane % 16 // 8 % 8 * 16
                                                            ^ (
                                                                pair_col_base
                                                                + 8 * (lane // 16)
                                                                + lane % 8
                                                                & 7
                                                            )
                                                            << 4
                                                        )
                                                        // 16
                                                        + 256
                                                        ^ 2
                                                    )
                                                    - 256
                                                    + 256
                                                    ^ 6
                                                )
                                                - 256
                                                + 256
                                                ^ 2
                                            )
                                            - 256
                                            + 256
                                            ^ 6
                                        )
                                        + 256
                                        - 256
                                        + 256
                                        ^ 2
                                    )
                                    - 256
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(_mma_m16n8k16_bf16_acc(acc, a_frag, b_frag))
                _builder_emit(_mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag))
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_qd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        lane // 16 // 8 * 256
                                        + (pair_row_base + lane % 16) * 8
                                        + (
                                            lane // 16 % 8 * 16
                                            ^ (pair_row_base + lane % 16 & 7) << 4
                                        )
                                        // 16
                                        ^ 2
                                        ^ 6
                                        ^ 2
                                        ^ 6
                                    )
                                    + 256
                                    ^ 2
                                    ^ 6
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        b_frag[0],
                        b_frag[1],
                        b_frag[2],
                        b_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_ki_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            lane % 16 // 8 // 8 * 256
                                                            + (
                                                                pair_col_base
                                                                + 8 * (lane // 16)
                                                                + lane % 8
                                                            )
                                                            * 8
                                                            + (
                                                                lane % 16 // 8 % 8 * 16
                                                                ^ (
                                                                    pair_col_base
                                                                    + 8 * (lane // 16)
                                                                    + lane % 8
                                                                    & 7
                                                                )
                                                                << 4
                                                            )
                                                            // 16
                                                            + 256
                                                            ^ 2
                                                        )
                                                        - 256
                                                        + 256
                                                        ^ 6
                                                    )
                                                    - 256
                                                    + 256
                                                    ^ 2
                                                )
                                                - 256
                                                + 256
                                                ^ 6
                                            )
                                            + 256
                                            - 256
                                            + 256
                                            ^ 2
                                        )
                                        - 256
                                        + 256
                                        ^ 6
                                    )
                                    - 256
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(_mma_m16n8k16_bf16_acc(acc, a_frag, b_frag))
                _builder_emit(_mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag))
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_qd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        lane // 16 // 8 * 256
                                        + (pair_row_base + lane % 16) * 8
                                        + (
                                            lane // 16 % 8 * 16
                                            ^ (pair_row_base + lane % 16 & 7) << 4
                                        )
                                        // 16
                                        ^ 2
                                        ^ 6
                                        ^ 2
                                        ^ 6
                                    )
                                    + 256
                                    ^ 2
                                    ^ 6
                                    ^ 2
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        b_frag[0],
                        b_frag[1],
                        b_frag[2],
                        b_frag[3],
                        smem_raw.ptr_to(
                            [
                                smem_ki_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                lane % 16 // 8 // 8 * 256
                                                                + (
                                                                    pair_col_base
                                                                    + 8 * (lane // 16)
                                                                    + lane % 8
                                                                )
                                                                * 8
                                                                + (
                                                                    lane % 16 // 8 % 8 * 16
                                                                    ^ (
                                                                        pair_col_base
                                                                        + 8 * (lane // 16)
                                                                        + lane % 8
                                                                        & 7
                                                                    )
                                                                    << 4
                                                                )
                                                                // 16
                                                                + 256
                                                                ^ 2
                                                            )
                                                            - 256
                                                            + 256
                                                            ^ 6
                                                        )
                                                        - 256
                                                        + 256
                                                        ^ 2
                                                    )
                                                    - 256
                                                    + 256
                                                    ^ 6
                                                )
                                                + 256
                                                - 256
                                                + 256
                                                ^ 2
                                            )
                                            - 256
                                            + 256
                                            ^ 6
                                        )
                                        - 256
                                        + 256
                                        ^ 2
                                    )
                                    - 256
                                )
                                * 16
                                - T.cast(smem, "int32")
                            ]
                        ),
                    )
                )
                _builder_emit(_mma_m16n8k16_bf16_acc(acc, a_frag, b_frag))
                _builder_emit(_mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag))
                _builder_scope_exit(_builder_then_1389_12)
                _builder_else_1389_12 = _builder_scope_enter(T.Else())
                with T.unroll(8) as _zi:
                    T.buffer_store(acc, T.float32(0.0), [_zi])
                _builder_scope_exit(_builder_else_1389_12)
                _builder_scope_exit(_builder_if_1389_12)
                row0_1 = _builder_scalar("row0_1", pair_row_base + lane // 4, dtype="int32")
                row1_1 = _builder_scalar("row1_1", row0_1 + 8, dtype="int32")
                col0_1 = _builder_scalar("col0_1", pair_col_base + lane % 4 * 2, dtype="int32")
                mqk = _builder_name("mqk", T.alloc_local((8,), "float32"))
                with T.unroll(8) as _zi:
                    T.buffer_store(mqk, T.float32(0.0), [_zi])
                _builder_if_1710_12 = _builder_scope_enter(T.If(row0_1 >= col0_1))
                _builder_then_1710_12 = _builder_scope_enter(T.Then())
                T.buffer_store(mqk, acc[0], [0])
                _builder_scope_exit(_builder_then_1710_12)
                _builder_scope_exit(_builder_if_1710_12)
                _builder_if_1712_12 = _builder_scope_enter(T.If(row0_1 >= col0_1 + 1))
                _builder_then_1712_12 = _builder_scope_enter(T.Then())
                T.buffer_store(mqk, acc[1], [1])
                _builder_scope_exit(_builder_then_1712_12)
                _builder_scope_exit(_builder_if_1712_12)
                _builder_if_1714_12 = _builder_scope_enter(T.If(row1_1 >= col0_1))
                _builder_then_1714_12 = _builder_scope_enter(T.Then())
                T.buffer_store(mqk, acc[2], [2])
                _builder_scope_exit(_builder_then_1714_12)
                _builder_scope_exit(_builder_if_1714_12)
                _builder_if_1716_12 = _builder_scope_enter(T.If(row1_1 >= col0_1 + 1))
                _builder_then_1716_12 = _builder_scope_enter(T.Then())
                T.buffer_store(mqk, acc[3], [3])
                _builder_scope_exit(_builder_then_1716_12)
                _builder_scope_exit(_builder_if_1716_12)
                _builder_if_1718_12 = _builder_scope_enter(T.If(row0_1 >= col0_1 + 8))
                _builder_then_1718_12 = _builder_scope_enter(T.Then())
                T.buffer_store(mqk, acc[4], [4])
                _builder_scope_exit(_builder_then_1718_12)
                _builder_scope_exit(_builder_if_1718_12)
                _builder_if_1720_12 = _builder_scope_enter(T.If(row0_1 >= col0_1 + 9))
                _builder_then_1720_12 = _builder_scope_enter(T.Then())
                T.buffer_store(mqk, acc[5], [5])
                _builder_scope_exit(_builder_then_1720_12)
                _builder_scope_exit(_builder_if_1720_12)
                _builder_if_1722_12 = _builder_scope_enter(T.If(row1_1 >= col0_1 + 8))
                _builder_then_1722_12 = _builder_scope_enter(T.Then())
                T.buffer_store(mqk, acc[6], [6])
                _builder_scope_exit(_builder_then_1722_12)
                _builder_scope_exit(_builder_if_1722_12)
                _builder_if_1724_12 = _builder_scope_enter(T.If(row1_1 >= col0_1 + 9))
                _builder_then_1724_12 = _builder_scope_enter(T.Then())
                T.buffer_store(mqk, acc[7], [7])
                _builder_scope_exit(_builder_then_1724_12)
                _builder_scope_exit(_builder_if_1724_12)
                mqk_packed = _builder_name("mqk_packed", T.alloc_local((4,), "uint32", align=4))
                with T.unroll(4) as _lp:
                    T.buffer_store(
                        mqk_packed,
                        T.cuda.float22bfloat162_rn(mqk[_lp * 2 + 0], mqk[_lp * 2 + 1 + 0]),
                        [_lp],
                    )
                with T.unroll(2) as publish_pair:
                    publish_row = _builder_scalar(
                        "publish_row", pair_col_base + publish_pair * 8 + (lane & 7), dtype="int32"
                    )
                    publish_col = _builder_scalar(
                        "publish_col", 128 + pair_row_base + lane // 8 * 8, dtype="int32"
                    )
                    _pub_base = _builder_scalar(
                        "_pub_base",
                        publish_col // 64 * 4096 + publish_row * 128 + publish_col % 64 * 2,
                        dtype="int32",
                    )
                    _pub_addr = _builder_scalar(
                        "_pub_addr",
                        smem_final_trans_addr
                        + T.cast(prep_stage, "int32") * 41984
                        + (_pub_base ^ (_pub_base >> 7 & 7) << 4),
                        dtype="int32",
                    )
                    _builder_emit(
                        T.ptx.stmatrix.sync.aligned.m8n8.x2.trans.shared.b16(
                            smem_raw.ptr_to([_pub_addr - T.cast(smem, "int32")]),
                            mqk_packed[publish_pair * 2],
                            mqk_packed[publish_pair * 2 + 1],
                        )
                    )
                _builder_if_1746_12 = _builder_scope_enter(T.If(prep_instance == 0))
                _builder_then_1746_12 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(11), T.uint32(128)))
                _builder_scope_exit(_builder_then_1746_12)
                _builder_else_1746_12 = _builder_scope_enter(T.Else())
                _builder_if_1748_12 = _builder_scope_enter(T.If(prep_instance == 1))
                _builder_then_1748_12 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(12), T.uint32(128)))
                _builder_scope_exit(_builder_then_1748_12)
                _builder_else_1748_12 = _builder_scope_enter(T.Else())
                _builder_if_1751_16 = _builder_scope_enter(T.If(prep_instance == 2))
                _builder_then_1751_16 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(13), T.uint32(128)))
                _builder_scope_exit(_builder_then_1751_16)
                _builder_else_1751_16 = _builder_scope_enter(T.Else())
                _builder_if_1753_16 = _builder_scope_enter(T.If(prep_instance == 3))
                _builder_then_1753_16 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(14), T.uint32(128)))
                _builder_scope_exit(_builder_then_1753_16)
                _builder_else_1753_16 = _builder_scope_enter(T.Else())
                _builder_emit(T.ptx.bar.sync(T.uint32(15), T.uint32(128)))
                _builder_scope_exit(_builder_else_1753_16)
                _builder_scope_exit(_builder_if_1753_16)
                _builder_scope_exit(_builder_else_1751_16)
                _builder_scope_exit(_builder_if_1751_16)
                _builder_scope_exit(_builder_else_1748_12)
                _builder_scope_exit(_builder_if_1748_12)
                _builder_scope_exit(_builder_else_1746_12)
                _builder_scope_exit(_builder_if_1746_12)
                _builder_if_1757_12 = _builder_scope_enter(T.If(prep_tid < 128))
                _builder_then_1757_12 = _builder_scope_enter(T.Then())
                total_log2_1 = _builder_scalar(
                    "total_log2_1",
                    _ld_shared_f32(smem_gt_prefix_all, stage_f32 + prep_tid),
                    dtype="f32",
                )
                _builder_emit(
                    _st_shared_f32(smem_gt_all, stage_f32 + prep_tid, _approx_exp2(total_log2_1))
                )
                _builder_scope_exit(_builder_then_1757_12)
                _builder_scope_exit(_builder_if_1757_12)
                _builder_if_1764_12 = _builder_scope_enter(T.If(prep_local_warp >= 2))
                _builder_then_1764_12 = _builder_scope_enter(T.Then())
                stage_f32_0 = _builder_scalar(
                    "stage_f32_0", T.cast(prep_stage, "int32") * 10496, dtype="int32"
                )
                restore_scale = _builder_scalar(
                    "restore_scale",
                    _ld_shared_f32(smem_restore_factor_all, stage_f32_0 + 128),
                    dtype="f32",
                )
                restore_factor = _builder_name("restore_factor", T.alloc_local((8,), "float32"))
                restore_segment = _builder_scalar("restore_segment", lane & 15, dtype="int32")
                with T.unroll(2) as restore_vec:
                    _builder_emit(
                        _ld_shared_v4_f32(
                            restore_factor,
                            restore_vec * 4,
                            smem_restore_factor_all,
                            stage_f32_0 + restore_segment * 8 + restore_vec * 4,
                        )
                    )
                with T.serial(0, 6, unroll=False) as restore_pass:
                    restore_row = _builder_scalar(
                        "restore_row",
                        8 + (prep_local_warp - 2) * 12 + restore_pass * 2 + (lane >> 4),
                        dtype="int32",
                    )
                    restore_qd_values = _builder_name(
                        "restore_qd_values", T.alloc_local((8,), "float32")
                    )
                    restore_kd_values = _builder_name(
                        "restore_kd_values", T.alloc_local((8,), "float32")
                    )
                    restore_ki_values = _builder_name(
                        "restore_ki_values", T.alloc_local((8,), "float32")
                    )
                    packed_2 = _builder_name("packed_2", T.alloc_local((4,), "uint32", align=16))
                    _builder_emit(
                        _ld_shared_v4(
                            smem_raw,
                            T.cast(smem, "int32"),
                            packed_2,
                            smem_qd_addr
                            + T.cast(prep_stage, "int32") * 41984
                            + (
                                restore_segment * 8 // 64 * 4096
                                + restore_row * 128
                                + restore_segment * 8 % 64 * 2
                                ^ (
                                    restore_segment * 8 // 64 * 4096
                                    + restore_row * 128
                                    + restore_segment * 8 % 64 * 2
                                    >> 7
                                    & 7
                                )
                                << 4
                            ),
                        )
                    )
                    packed_fp32_1 = _builder_name("packed_fp32_1", T.alloc_local((8,), "float32"))
                    with T.unroll(4) as _pair:
                        T.buffer_store(
                            packed_fp32_1,
                            T.cuda.uint_as_float(packed_2[_pair + 0] << T.uint32(16)),
                            [_pair * 2],
                        )
                        T.buffer_store(
                            packed_fp32_1,
                            T.cuda.uint_as_float(packed_2[_pair + 0] & T.uint32(4294901760)),
                            [_pair * 2 + 1],
                        )
                    with T.unroll(8) as value_idx_2:
                        T.buffer_store(restore_qd_values, packed_fp32_1[value_idx_2], [value_idx_2])
                    packed_0_2 = _builder_name(
                        "packed_0_2", T.alloc_local((4,), "uint32", align=16)
                    )
                    _builder_emit(
                        _ld_shared_v4(
                            smem_raw,
                            T.cast(smem, "int32"),
                            packed_0_2,
                            smem_kd_addr
                            + T.cast(prep_stage, "int32") * 41984
                            + (
                                restore_segment * 8 // 64 * 4096
                                + restore_row * 128
                                + restore_segment * 8 % 64 * 2
                                ^ (
                                    restore_segment * 8 // 64 * 4096
                                    + restore_row * 128
                                    + restore_segment * 8 % 64 * 2
                                    >> 7
                                    & 7
                                )
                                << 4
                            ),
                        )
                    )
                    packed_0_fp32_1 = _builder_name(
                        "packed_0_fp32_1", T.alloc_local((8,), "float32")
                    )
                    with T.unroll(4) as _pair:
                        T.buffer_store(
                            packed_0_fp32_1,
                            T.cuda.uint_as_float(packed_0_2[_pair + 0] << T.uint32(16)),
                            [_pair * 2],
                        )
                        T.buffer_store(
                            packed_0_fp32_1,
                            T.cuda.uint_as_float(packed_0_2[_pair + 0] & T.uint32(4294901760)),
                            [_pair * 2 + 1],
                        )
                    with T.unroll(8) as value_idx_3:
                        T.buffer_store(
                            restore_kd_values, packed_0_fp32_1[value_idx_3], [value_idx_3]
                        )
                    packed_1_2 = _builder_name(
                        "packed_1_2", T.alloc_local((4,), "uint32", align=16)
                    )
                    _builder_emit(
                        _ld_shared_v4(
                            smem_raw,
                            T.cast(smem, "int32"),
                            packed_1_2,
                            smem_ki_addr
                            + T.cast(prep_stage, "int32") * 41984
                            + (
                                restore_segment * 8 // 64 * 4096
                                + restore_row * 128
                                + restore_segment * 8 % 64 * 2
                                ^ (
                                    restore_segment * 8 // 64 * 4096
                                    + restore_row * 128
                                    + restore_segment * 8 % 64 * 2
                                    >> 7
                                    & 7
                                )
                                << 4
                            ),
                        )
                    )
                    packed_1_fp32 = _builder_name("packed_1_fp32", T.alloc_local((8,), "float32"))
                    with T.unroll(4) as _pair:
                        T.buffer_store(
                            packed_1_fp32,
                            T.cuda.uint_as_float(packed_1_2[_pair + 0] << T.uint32(16)),
                            [_pair * 2],
                        )
                        T.buffer_store(
                            packed_1_fp32,
                            T.cuda.uint_as_float(packed_1_2[_pair + 0] & T.uint32(4294901760)),
                            [_pair * 2 + 1],
                        )
                    with T.unroll(8) as value_idx_4:
                        T.buffer_store(restore_ki_values, packed_1_fp32[value_idx_4], [value_idx_4])
                    restore_kr_values = _builder_name(
                        "restore_kr_values", T.alloc_local((8,), "float32")
                    )
                    with T.unroll(8) as restore_elem_1:
                        T.buffer_store(
                            restore_kr_values,
                            restore_ki_values[restore_elem_1] * restore_factor[restore_elem_1],
                            [restore_elem_1],
                        )
                    with T.unroll(4) as _ls:
                        _pk = _builder_scalar(
                            "_pk",
                            _mul_f32x2_inplace(
                                T.cuda.make_float2(
                                    restore_qd_values[_ls * 2], restore_qd_values[_ls * 2 + 1]
                                ),
                                T.cuda.make_float2(restore_scale, restore_scale),
                            ),
                        )
                        T.buffer_store(restore_qd_values, T.cuda.float2_x(_pk), [_ls * 2])
                        T.buffer_store(restore_qd_values, T.cuda.float2_y(_pk), [_ls * 2 + 1])
                    with T.unroll(4) as _ls:
                        _pk = _builder_scalar(
                            "_pk",
                            _mul_f32x2_inplace(
                                T.cuda.make_float2(
                                    restore_kd_values[_ls * 2], restore_kd_values[_ls * 2 + 1]
                                ),
                                T.cuda.make_float2(restore_scale, restore_scale),
                            ),
                        )
                        T.buffer_store(restore_kd_values, T.cuda.float2_x(_pk), [_ls * 2])
                        T.buffer_store(restore_kd_values, T.cuda.float2_y(_pk), [_ls * 2 + 1])
                    packed_2_1 = _builder_name("packed_2_1", T.alloc_local((4,), "uint32", align=4))
                    with T.unroll(4) as _lp:
                        T.buffer_store(
                            packed_2_1,
                            T.cuda.float22bfloat162_rn(
                                restore_qd_values[_lp * 2 + 0], restore_qd_values[_lp * 2 + 1 + 0]
                            ),
                            [_lp],
                        )
                    with T.unroll(4) as word_3:
                        _builder_emit(
                            _st_shared_b32(
                                smem_raw,
                                T.cast(smem, "int32"),
                                smem_qd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    restore_segment * 8 // 64 * 4096
                                    + restore_row * 128
                                    + restore_segment * 8 % 64 * 2
                                    ^ (
                                        restore_segment * 8 // 64 * 4096
                                        + restore_row * 128
                                        + restore_segment * 8 % 64 * 2
                                        >> 7
                                        & 7
                                    )
                                    << 4
                                )
                                + word_3 * 4,
                                packed_2_1[word_3],
                            )
                        )
                    packed_3 = _builder_name("packed_3", T.alloc_local((4,), "uint32", align=4))
                    with T.unroll(4) as _lp:
                        T.buffer_store(
                            packed_3,
                            T.cuda.float22bfloat162_rn(
                                restore_kd_values[_lp * 2 + 0], restore_kd_values[_lp * 2 + 1 + 0]
                            ),
                            [_lp],
                        )
                    with T.unroll(4) as word_4:
                        _builder_emit(
                            _st_shared_b32(
                                smem_raw,
                                T.cast(smem, "int32"),
                                smem_kd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    restore_segment * 8 // 64 * 4096
                                    + restore_row * 128
                                    + restore_segment * 8 % 64 * 2
                                    ^ (
                                        restore_segment * 8 // 64 * 4096
                                        + restore_row * 128
                                        + restore_segment * 8 % 64 * 2
                                        >> 7
                                        & 7
                                    )
                                    << 4
                                )
                                + word_4 * 4,
                                packed_3[word_4],
                            )
                        )
                    packed_4 = _builder_name("packed_4", T.alloc_local((4,), "uint32", align=4))
                    with T.unroll(4) as _lp:
                        T.buffer_store(
                            packed_4,
                            T.cuda.float22bfloat162_rn(
                                restore_kr_values[_lp * 2 + 0], restore_kr_values[_lp * 2 + 1 + 0]
                            ),
                            [_lp],
                        )
                    with T.unroll(4) as word_5:
                        _builder_emit(
                            _st_shared_b32(
                                smem_raw,
                                T.cast(smem, "int32"),
                                smem_kr_trans_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    restore_segment * 8 // 64 * 4096
                                    + restore_row * 128
                                    + restore_segment * 8 % 64 * 2
                                    ^ (
                                        restore_segment * 8 // 64 * 4096
                                        + restore_row * 128
                                        + restore_segment * 8 % 64 * 2
                                        >> 7
                                        & 7
                                    )
                                    << 4
                                )
                                + word_5 * 4,
                                packed_4[word_5],
                            )
                        )
                _builder_scope_exit(_builder_then_1764_12)
                _builder_scope_exit(_builder_if_1764_12)
                _builder_if_1897_12 = _builder_scope_enter(T.If(prep_local_warp == 0))
                _builder_then_1897_12 = _builder_scope_enter(T.Then())
                inverse_row = _builder_scalar("inverse_row", lane, dtype="int32")
                diag_block = _builder_scalar("diag_block", inverse_row // 8, dtype="int32")
                lane_in_diag = _builder_scalar("lane_in_diag", lane & 7, dtype="int32")
                inv_row = _builder_name("inv_row", T.alloc_local((8,), "float32"))
                packed_5 = _builder_name("packed_5", T.alloc_local((4,), "uint32", align=16))
                byte_off_1 = _builder_scalar(
                    "byte_off_1", inverse_row * 128 + diag_block * 8 * 2, dtype="int32"
                )
                swizzled_off_1 = _builder_scalar(
                    "swizzled_off_1", byte_off_1 ^ (byte_off_1 >> 7 & 7) << 4, dtype="int32"
                )
                _builder_emit(
                    _ld_shared_v4(
                        smem_raw,
                        T.cast(smem, "int32"),
                        packed_5,
                        smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off_1,
                    )
                )
                packed_fp32_2 = _builder_name("packed_fp32_2", T.alloc_local((8,), "float32"))
                with T.unroll(4) as _pair:
                    T.buffer_store(
                        packed_fp32_2,
                        T.cuda.uint_as_float(packed_5[_pair + 0] << T.uint32(16)),
                        [_pair * 2],
                    )
                    T.buffer_store(
                        packed_fp32_2,
                        T.cuda.uint_as_float(packed_5[_pair + 0] & T.uint32(4294901760)),
                        [_pair * 2 + 1],
                    )
                with T.unroll(8) as value_idx_5:
                    T.buffer_store(inv_row, packed_fp32_2[value_idx_5], [value_idx_5])
                with T.unroll(8) as diag_elem:
                    _builder_if_1922_20 = _builder_scope_enter(T.If(lane_in_diag == diag_elem))
                    _builder_then_1922_20 = _builder_scope_enter(T.Then())
                    T.buffer_store(inv_row, T.float32(1.0), [diag_elem])
                    _builder_scope_exit(_builder_then_1922_20)
                    _builder_scope_exit(_builder_if_1922_20)
                diag_group_base = _builder_scalar(
                    "diag_group_base", lane - lane_in_diag, dtype="int32"
                )
                row_scale = _builder_scalar("row_scale", -inv_row[0], dtype="f32")
                _builder_if_1926_16 = _builder_scope_enter(T.If(lane_in_diag > 0))
                _builder_then_1926_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, row_scale, [0])
                _builder_scope_exit(_builder_then_1926_16)
                _builder_scope_exit(_builder_if_1926_16)
                row_scale_1 = _builder_scalar("row_scale_1", -inv_row[1], dtype="f32")
                pivot_lane_1_0 = _builder_scalar(
                    "pivot_lane_1_0", diag_group_base + 1, dtype="int32"
                )
                pivot_1_0 = _builder_scalar(
                    "pivot_1_0",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[0], pivot_lane_1_0, 32),
                    dtype="f32",
                )
                _builder_if_1933_16 = _builder_scope_enter(T.If(lane_in_diag > 1))
                _builder_then_1933_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_1, pivot_1_0, inv_row[0]), [0])
                _builder_scope_exit(_builder_then_1933_16)
                _builder_scope_exit(_builder_if_1933_16)
                _builder_if_1935_16 = _builder_scope_enter(T.If(lane_in_diag > 1))
                _builder_then_1935_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, row_scale_1, [1])
                _builder_scope_exit(_builder_then_1935_16)
                _builder_scope_exit(_builder_if_1935_16)
                row_scale_2 = _builder_scalar("row_scale_2", -inv_row[2], dtype="f32")
                pivot_lane_2_0 = _builder_scalar(
                    "pivot_lane_2_0", diag_group_base + 2, dtype="int32"
                )
                pivot_2_0 = _builder_scalar(
                    "pivot_2_0",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[0], pivot_lane_2_0, 32),
                    dtype="f32",
                )
                _builder_if_1942_16 = _builder_scope_enter(T.If(lane_in_diag > 2))
                _builder_then_1942_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_2, pivot_2_0, inv_row[0]), [0])
                _builder_scope_exit(_builder_then_1942_16)
                _builder_scope_exit(_builder_if_1942_16)
                pivot_lane_2_1 = _builder_scalar(
                    "pivot_lane_2_1", diag_group_base + 2, dtype="int32"
                )
                pivot_2_1 = _builder_scalar(
                    "pivot_2_1",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[1], pivot_lane_2_1, 32),
                    dtype="f32",
                )
                _builder_if_1948_16 = _builder_scope_enter(T.If(lane_in_diag > 2))
                _builder_then_1948_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_2, pivot_2_1, inv_row[1]), [1])
                _builder_scope_exit(_builder_then_1948_16)
                _builder_scope_exit(_builder_if_1948_16)
                _builder_if_1950_16 = _builder_scope_enter(T.If(lane_in_diag > 2))
                _builder_then_1950_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, row_scale_2, [2])
                _builder_scope_exit(_builder_then_1950_16)
                _builder_scope_exit(_builder_if_1950_16)
                row_scale_3 = _builder_scalar("row_scale_3", -inv_row[3], dtype="f32")
                pivot_lane_3_0 = _builder_scalar(
                    "pivot_lane_3_0", diag_group_base + 3, dtype="int32"
                )
                pivot_3_0 = _builder_scalar(
                    "pivot_3_0",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[0], pivot_lane_3_0, 32),
                    dtype="f32",
                )
                _builder_if_1957_16 = _builder_scope_enter(T.If(lane_in_diag > 3))
                _builder_then_1957_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_3, pivot_3_0, inv_row[0]), [0])
                _builder_scope_exit(_builder_then_1957_16)
                _builder_scope_exit(_builder_if_1957_16)
                pivot_lane_3_1 = _builder_scalar(
                    "pivot_lane_3_1", diag_group_base + 3, dtype="int32"
                )
                pivot_3_1 = _builder_scalar(
                    "pivot_3_1",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[1], pivot_lane_3_1, 32),
                    dtype="f32",
                )
                _builder_if_1963_16 = _builder_scope_enter(T.If(lane_in_diag > 3))
                _builder_then_1963_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_3, pivot_3_1, inv_row[1]), [1])
                _builder_scope_exit(_builder_then_1963_16)
                _builder_scope_exit(_builder_if_1963_16)
                pivot_lane_3_2 = _builder_scalar(
                    "pivot_lane_3_2", diag_group_base + 3, dtype="int32"
                )
                pivot_3_2 = _builder_scalar(
                    "pivot_3_2",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[2], pivot_lane_3_2, 32),
                    dtype="f32",
                )
                _builder_if_1969_16 = _builder_scope_enter(T.If(lane_in_diag > 3))
                _builder_then_1969_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_3, pivot_3_2, inv_row[2]), [2])
                _builder_scope_exit(_builder_then_1969_16)
                _builder_scope_exit(_builder_if_1969_16)
                _builder_if_1971_16 = _builder_scope_enter(T.If(lane_in_diag > 3))
                _builder_then_1971_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, row_scale_3, [3])
                _builder_scope_exit(_builder_then_1971_16)
                _builder_scope_exit(_builder_if_1971_16)
                row_scale_4 = _builder_scalar("row_scale_4", -inv_row[4], dtype="f32")
                pivot_lane_4_0 = _builder_scalar(
                    "pivot_lane_4_0", diag_group_base + 4, dtype="int32"
                )
                pivot_4_0 = _builder_scalar(
                    "pivot_4_0",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[0], pivot_lane_4_0, 32),
                    dtype="f32",
                )
                _builder_if_1978_16 = _builder_scope_enter(T.If(lane_in_diag > 4))
                _builder_then_1978_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_4, pivot_4_0, inv_row[0]), [0])
                _builder_scope_exit(_builder_then_1978_16)
                _builder_scope_exit(_builder_if_1978_16)
                pivot_lane_4_1 = _builder_scalar(
                    "pivot_lane_4_1", diag_group_base + 4, dtype="int32"
                )
                pivot_4_1 = _builder_scalar(
                    "pivot_4_1",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[1], pivot_lane_4_1, 32),
                    dtype="f32",
                )
                _builder_if_1984_16 = _builder_scope_enter(T.If(lane_in_diag > 4))
                _builder_then_1984_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_4, pivot_4_1, inv_row[1]), [1])
                _builder_scope_exit(_builder_then_1984_16)
                _builder_scope_exit(_builder_if_1984_16)
                pivot_lane_4_2 = _builder_scalar(
                    "pivot_lane_4_2", diag_group_base + 4, dtype="int32"
                )
                pivot_4_2 = _builder_scalar(
                    "pivot_4_2",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[2], pivot_lane_4_2, 32),
                    dtype="f32",
                )
                _builder_if_1990_16 = _builder_scope_enter(T.If(lane_in_diag > 4))
                _builder_then_1990_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_4, pivot_4_2, inv_row[2]), [2])
                _builder_scope_exit(_builder_then_1990_16)
                _builder_scope_exit(_builder_if_1990_16)
                pivot_lane_4_3 = _builder_scalar(
                    "pivot_lane_4_3", diag_group_base + 4, dtype="int32"
                )
                pivot_4_3 = _builder_scalar(
                    "pivot_4_3",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[3], pivot_lane_4_3, 32),
                    dtype="f32",
                )
                _builder_if_1996_16 = _builder_scope_enter(T.If(lane_in_diag > 4))
                _builder_then_1996_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_4, pivot_4_3, inv_row[3]), [3])
                _builder_scope_exit(_builder_then_1996_16)
                _builder_scope_exit(_builder_if_1996_16)
                _builder_if_1998_16 = _builder_scope_enter(T.If(lane_in_diag > 4))
                _builder_then_1998_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, row_scale_4, [4])
                _builder_scope_exit(_builder_then_1998_16)
                _builder_scope_exit(_builder_if_1998_16)
                row_scale_5 = _builder_scalar("row_scale_5", -inv_row[5], dtype="f32")
                pivot_lane_5_0 = _builder_scalar(
                    "pivot_lane_5_0", diag_group_base + 5, dtype="int32"
                )
                pivot_5_0 = _builder_scalar(
                    "pivot_5_0",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[0], pivot_lane_5_0, 32),
                    dtype="f32",
                )
                _builder_if_2005_16 = _builder_scope_enter(T.If(lane_in_diag > 5))
                _builder_then_2005_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_5, pivot_5_0, inv_row[0]), [0])
                _builder_scope_exit(_builder_then_2005_16)
                _builder_scope_exit(_builder_if_2005_16)
                pivot_lane_5_1 = _builder_scalar(
                    "pivot_lane_5_1", diag_group_base + 5, dtype="int32"
                )
                pivot_5_1 = _builder_scalar(
                    "pivot_5_1",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[1], pivot_lane_5_1, 32),
                    dtype="f32",
                )
                _builder_if_2011_16 = _builder_scope_enter(T.If(lane_in_diag > 5))
                _builder_then_2011_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_5, pivot_5_1, inv_row[1]), [1])
                _builder_scope_exit(_builder_then_2011_16)
                _builder_scope_exit(_builder_if_2011_16)
                pivot_lane_5_2 = _builder_scalar(
                    "pivot_lane_5_2", diag_group_base + 5, dtype="int32"
                )
                pivot_5_2 = _builder_scalar(
                    "pivot_5_2",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[2], pivot_lane_5_2, 32),
                    dtype="f32",
                )
                _builder_if_2017_16 = _builder_scope_enter(T.If(lane_in_diag > 5))
                _builder_then_2017_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_5, pivot_5_2, inv_row[2]), [2])
                _builder_scope_exit(_builder_then_2017_16)
                _builder_scope_exit(_builder_if_2017_16)
                pivot_lane_5_3 = _builder_scalar(
                    "pivot_lane_5_3", diag_group_base + 5, dtype="int32"
                )
                pivot_5_3 = _builder_scalar(
                    "pivot_5_3",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[3], pivot_lane_5_3, 32),
                    dtype="f32",
                )
                _builder_if_2023_16 = _builder_scope_enter(T.If(lane_in_diag > 5))
                _builder_then_2023_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_5, pivot_5_3, inv_row[3]), [3])
                _builder_scope_exit(_builder_then_2023_16)
                _builder_scope_exit(_builder_if_2023_16)
                pivot_lane_5_4 = _builder_scalar(
                    "pivot_lane_5_4", diag_group_base + 5, dtype="int32"
                )
                pivot_5_4 = _builder_scalar(
                    "pivot_5_4",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[4], pivot_lane_5_4, 32),
                    dtype="f32",
                )
                _builder_if_2029_16 = _builder_scope_enter(T.If(lane_in_diag > 5))
                _builder_then_2029_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_5, pivot_5_4, inv_row[4]), [4])
                _builder_scope_exit(_builder_then_2029_16)
                _builder_scope_exit(_builder_if_2029_16)
                _builder_if_2031_16 = _builder_scope_enter(T.If(lane_in_diag > 5))
                _builder_then_2031_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, row_scale_5, [5])
                _builder_scope_exit(_builder_then_2031_16)
                _builder_scope_exit(_builder_if_2031_16)
                row_scale_6 = _builder_scalar("row_scale_6", -inv_row[6], dtype="f32")
                pivot_lane_6_0 = _builder_scalar(
                    "pivot_lane_6_0", diag_group_base + 6, dtype="int32"
                )
                pivot_6_0 = _builder_scalar(
                    "pivot_6_0",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[0], pivot_lane_6_0, 32),
                    dtype="f32",
                )
                _builder_if_2038_16 = _builder_scope_enter(T.If(lane_in_diag > 6))
                _builder_then_2038_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_6, pivot_6_0, inv_row[0]), [0])
                _builder_scope_exit(_builder_then_2038_16)
                _builder_scope_exit(_builder_if_2038_16)
                pivot_lane_6_1 = _builder_scalar(
                    "pivot_lane_6_1", diag_group_base + 6, dtype="int32"
                )
                pivot_6_1 = _builder_scalar(
                    "pivot_6_1",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[1], pivot_lane_6_1, 32),
                    dtype="f32",
                )
                _builder_if_2044_16 = _builder_scope_enter(T.If(lane_in_diag > 6))
                _builder_then_2044_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_6, pivot_6_1, inv_row[1]), [1])
                _builder_scope_exit(_builder_then_2044_16)
                _builder_scope_exit(_builder_if_2044_16)
                pivot_lane_6_2 = _builder_scalar(
                    "pivot_lane_6_2", diag_group_base + 6, dtype="int32"
                )
                pivot_6_2 = _builder_scalar(
                    "pivot_6_2",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[2], pivot_lane_6_2, 32),
                    dtype="f32",
                )
                _builder_if_2050_16 = _builder_scope_enter(T.If(lane_in_diag > 6))
                _builder_then_2050_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_6, pivot_6_2, inv_row[2]), [2])
                _builder_scope_exit(_builder_then_2050_16)
                _builder_scope_exit(_builder_if_2050_16)
                pivot_lane_6_3 = _builder_scalar(
                    "pivot_lane_6_3", diag_group_base + 6, dtype="int32"
                )
                pivot_6_3 = _builder_scalar(
                    "pivot_6_3",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[3], pivot_lane_6_3, 32),
                    dtype="f32",
                )
                _builder_if_2056_16 = _builder_scope_enter(T.If(lane_in_diag > 6))
                _builder_then_2056_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_6, pivot_6_3, inv_row[3]), [3])
                _builder_scope_exit(_builder_then_2056_16)
                _builder_scope_exit(_builder_if_2056_16)
                pivot_lane_6_4 = _builder_scalar(
                    "pivot_lane_6_4", diag_group_base + 6, dtype="int32"
                )
                pivot_6_4 = _builder_scalar(
                    "pivot_6_4",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[4], pivot_lane_6_4, 32),
                    dtype="f32",
                )
                _builder_if_2062_16 = _builder_scope_enter(T.If(lane_in_diag > 6))
                _builder_then_2062_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_6, pivot_6_4, inv_row[4]), [4])
                _builder_scope_exit(_builder_then_2062_16)
                _builder_scope_exit(_builder_if_2062_16)
                pivot_lane_6_5 = _builder_scalar(
                    "pivot_lane_6_5", diag_group_base + 6, dtype="int32"
                )
                pivot_6_5 = _builder_scalar(
                    "pivot_6_5",
                    T.cuda._shfl_sync(T.uint32(4294967295), inv_row[5], pivot_lane_6_5, 32),
                    dtype="f32",
                )
                _builder_if_2068_16 = _builder_scope_enter(T.If(lane_in_diag > 6))
                _builder_then_2068_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, _fmaf_rn(row_scale_6, pivot_6_5, inv_row[5]), [5])
                _builder_scope_exit(_builder_then_2068_16)
                _builder_scope_exit(_builder_if_2068_16)
                _builder_if_2070_16 = _builder_scope_enter(T.If(lane_in_diag > 6))
                _builder_then_2070_16 = _builder_scope_enter(T.Then())
                T.buffer_store(inv_row, row_scale_6, [6])
                _builder_scope_exit(_builder_then_2070_16)
                _builder_scope_exit(_builder_if_2070_16)
                packed_0_3 = _builder_name("packed_0_3", T.alloc_local((4,), "uint32", align=4))
                with T.unroll(4) as _lp:
                    T.buffer_store(
                        packed_0_3,
                        T.cuda.float22bfloat162_rn(inv_row[_lp * 2 + 0], inv_row[_lp * 2 + 1 + 0]),
                        [_lp],
                    )
                byte_off_1_1 = _builder_scalar(
                    "byte_off_1_1", inverse_row * 128 + diag_block * 8 * 2, dtype="int32"
                )
                swizzled_off_2 = _builder_scalar(
                    "swizzled_off_2", byte_off_1_1 ^ (byte_off_1_1 >> 7 & 7) << 4, dtype="int32"
                )
                with T.unroll(4) as word_6:
                    _builder_emit(
                        _st_shared_b32(
                            smem_raw,
                            T.cast(smem, "int32"),
                            smem_inv_work_addr
                            + T.cast(prep_stage, "int32") * 41984
                            + swizzled_off_2
                            + word_6 * 4,
                            packed_0_3[word_6],
                        )
                    )
                _builder_scope_exit(_builder_then_1897_12)
                _builder_scope_exit(_builder_if_1897_12)
                _builder_if_2086_12 = _builder_scope_enter(T.If(prep_local_warp < 2))
                _builder_then_2086_12 = _builder_scope_enter(T.Then())
                _builder_if_2087_16 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                _builder_then_2087_16 = _builder_scope_enter(T.Then())
                _builder_emit(
                    _mbarrier_arrive(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_PREP_DIAG_READY_OFF + prep_stage * 8,
                    )
                )
                _builder_scope_exit(_builder_then_2087_16)
                _builder_scope_exit(_builder_if_2087_16)
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_PREP_DIAG_READY_OFF + prep_stage * 8,
                        _phase_prep_diag_ready,
                    )
                )
                _builder_scope_exit(_builder_then_2086_12)
                _builder_scope_exit(_builder_if_2086_12)
                _builder_if_2099_12 = _builder_scope_enter(T.If(prep_local_warp < 2))
                _builder_then_2099_12 = _builder_scope_enter(T.Then())
                lane_row = _builder_scalar("lane_row", lane & 7, dtype="int32")
                byte_off_2 = _builder_scalar(
                    "byte_off_2",
                    (prep_local_warp * 16 + 8 + lane_row) * 128 + (prep_local_warp * 16 + 8) * 2,
                    dtype="int32",
                )
                swizzled_off_3 = _builder_scalar(
                    "swizzled_off_3", byte_off_2 ^ (byte_off_2 >> 7 & 7) << 4, dtype="int32"
                )
                d_addr = _builder_scalar(
                    "d_addr",
                    smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off_3,
                    dtype="int32",
                )
                byte_off_0 = _builder_scalar(
                    "byte_off_0",
                    (prep_local_warp * 16 + 8 + lane_row) * 128 + prep_local_warp * 16 * 2,
                    dtype="int32",
                )
                swizzled_off_1_1 = _builder_scalar(
                    "swizzled_off_1_1", byte_off_0 ^ (byte_off_0 >> 7 & 7) << 4, dtype="int32"
                )
                c_addr = _builder_scalar(
                    "c_addr",
                    smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off_1_1,
                    dtype="int32",
                )
                byte_off_2_1 = _builder_scalar(
                    "byte_off_2_1",
                    (prep_local_warp * 16 + lane_row) * 128 + prep_local_warp * 16 * 2,
                    dtype="int32",
                )
                swizzled_off_3_1 = _builder_scalar(
                    "swizzled_off_3_1", byte_off_2_1 ^ (byte_off_2_1 >> 7 & 7) << 4, dtype="int32"
                )
                a_addr = _builder_scalar(
                    "a_addr",
                    smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off_3_1,
                    dtype="int32",
                )
                d_frag = _builder_name("d_frag", T.alloc_local((2,), "uint32", align=4))
                c_frag = _builder_name("c_frag", T.alloc_local((1,), "uint32", align=4))
                dc_acc = _builder_name("dc_acc", T.alloc_local((4,), "float32", align=4))
                dc_bf16 = _builder_name("dc_bf16", T.alloc_local((2,), "uint32", align=4))
                inv_a_frag = _builder_name("inv_a_frag", T.alloc_local((1,), "uint32", align=4))
                o_acc = _builder_name("o_acc", T.alloc_local((4,), "float32", align=4))
                o_bf16 = _builder_name("o_bf16", T.alloc_local((2,), "uint32", align=4))
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x1.shared.b16(
                        d_frag[0], smem_raw.ptr_to([d_addr - T.cast(smem, "int32")])
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x1.shared.b16(
                        d_frag[1], smem_raw.ptr_to([d_addr - T.cast(smem, "int32")])
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16(
                        c_frag[0], smem_raw.ptr_to([c_addr - T.cast(smem, "int32")])
                    )
                )
                _builder_emit(_mma_m16n8k8_bf16_zero(dc_acc, d_frag, c_frag))
                with T.unroll(2) as _ls:
                    _pk = _builder_scalar(
                        "_pk",
                        _mul_f32x2_inplace(
                            T.cuda.make_float2(dc_acc[_ls * 2], dc_acc[_ls * 2 + 1]),
                            T.cuda.make_float2(T.float32(-1.0), T.float32(-1.0)),
                        ),
                    )
                    T.buffer_store(dc_acc, T.cuda.float2_x(_pk), [_ls * 2])
                    T.buffer_store(dc_acc, T.cuda.float2_y(_pk), [_ls * 2 + 1])
                with T.unroll(2) as _lp:
                    T.buffer_store(
                        dc_bf16,
                        T.cuda.float22bfloat162_rn(dc_acc[_lp * 2 + 0], dc_acc[_lp * 2 + 1 + 0]),
                        [_lp],
                    )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16(
                        inv_a_frag[0], smem_raw.ptr_to([a_addr - T.cast(smem, "int32")])
                    )
                )
                _builder_emit(_mma_m16n8k8_bf16_zero(o_acc, dc_bf16, inv_a_frag))
                with T.unroll(2) as _lp:
                    T.buffer_store(
                        o_bf16,
                        T.cuda.float22bfloat162_rn(o_acc[_lp * 2 + 0], o_acc[_lp * 2 + 1 + 0]),
                        [_lp],
                    )
                byte_off_4 = _builder_scalar(
                    "byte_off_4",
                    (prep_local_warp * 16 + 8 + lane_row) * 128 + prep_local_warp * 16 * 2,
                    dtype="int32",
                )
                swizzled_off_5 = _builder_scalar(
                    "swizzled_off_5", byte_off_4 ^ (byte_off_4 >> 7 & 7) << 4, dtype="int32"
                )
                o_addr = _builder_scalar(
                    "o_addr",
                    smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off_5,
                    dtype="int32",
                )
                _builder_emit(
                    T.ptx.stmatrix.sync.aligned.m8n8.x1.shared.b16(
                        smem_raw.ptr_to([o_addr - T.cast(smem, "int32")]), o_bf16[0]
                    )
                )
                _builder_if_2175_16 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                _builder_then_2175_16 = _builder_scope_enter(T.Then())
                _builder_emit(
                    _mbarrier_arrive(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_PREP_INV16_READY_OFF + prep_stage * 8,
                    )
                )
                _builder_scope_exit(_builder_then_2175_16)
                _builder_scope_exit(_builder_if_2175_16)
                _builder_emit(
                    _mbarrier_wait(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_PREP_INV16_READY_OFF + prep_stage * 8,
                        _phase_prep_inv16_ready,
                    )
                )
                _builder_scope_exit(_builder_then_2099_12)
                _builder_scope_exit(_builder_if_2099_12)
                _builder_if_2187_12 = _builder_scope_enter(T.If(prep_local_warp == 0))
                _builder_then_2187_12 = _builder_scope_enter(T.Then())
                lane_row_1 = _builder_scalar("lane_row_1", lane % 16, dtype="int32")
                lane_col = _builder_scalar("lane_col", lane // 16 * 8, dtype="int32")
                byte_off_3 = _builder_scalar(
                    "byte_off_3", (16 + lane_row_1) * 128 + (16 + lane_col) * 2, dtype="int32"
                )
                swizzled_off_4 = _builder_scalar(
                    "swizzled_off_4", byte_off_3 ^ (byte_off_3 >> 7 & 7) << 4, dtype="int32"
                )
                d_addr_1 = _builder_scalar(
                    "d_addr_1",
                    smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off_4,
                    dtype="int32",
                )
                byte_off_0_1 = _builder_scalar(
                    "byte_off_0_1", (16 + lane_row_1) * 128 + lane_col * 2, dtype="int32"
                )
                swizzled_off_1_2 = _builder_scalar(
                    "swizzled_off_1_2", byte_off_0_1 ^ (byte_off_0_1 >> 7 & 7) << 4, dtype="int32"
                )
                c_addr_1 = _builder_scalar(
                    "c_addr_1",
                    smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off_1_2,
                    dtype="int32",
                )
                byte_off_2_2 = _builder_scalar(
                    "byte_off_2_2", lane_row_1 * 128 + lane_col * 2, dtype="int32"
                )
                swizzled_off_3_2 = _builder_scalar(
                    "swizzled_off_3_2", byte_off_2_2 ^ (byte_off_2_2 >> 7 & 7) << 4, dtype="int32"
                )
                a_addr_1 = _builder_scalar(
                    "a_addr_1",
                    smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off_3_2,
                    dtype="int32",
                )
                d32_frag = _builder_name("d32_frag", T.alloc_local((4,), "uint32", align=4))
                c32_frag = _builder_name("c32_frag", T.alloc_local((4,), "uint32", align=4))
                dc32_acc = _builder_name("dc32_acc", T.alloc_local((8,), "float32", align=4))
                dc32_bf16 = _builder_name("dc32_bf16", T.alloc_local((4,), "uint32", align=4))
                a32_frag = _builder_name("a32_frag", T.alloc_local((4,), "uint32", align=4))
                o32_acc = _builder_name("o32_acc", T.alloc_local((8,), "float32", align=4))
                o32_bf16 = _builder_name("o32_bf16", T.alloc_local((4,), "uint32", align=4))
                zero32_bf16 = _builder_name("zero32_bf16", T.alloc_local((4,), "uint32", align=4))
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        d32_frag[0],
                        d32_frag[1],
                        d32_frag[2],
                        d32_frag[3],
                        smem_raw.ptr_to([d_addr_1 - T.cast(smem, "int32")]),
                    )
                )
                _dpb = _builder_scalar(
                    "_dpb",
                    (16 + lane_col) // 16 * 1024
                    + (16 + lane_row_1) * 32
                    + (16 + lane_col) % 16 * 2,
                    dtype="int32",
                )
                d_publish_addr = _builder_scalar(
                    "d_publish_addr",
                    smem_inv_addr
                    + T.cast(prep_stage, "int32") * 41984
                    + (_dpb ^ (_dpb >> 7 & 1) << 4),
                    dtype="int32",
                )
                _builder_emit(
                    T.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                        smem_raw.ptr_to([d_publish_addr - T.cast(smem, "int32")]),
                        d32_frag[0],
                        d32_frag[1],
                        d32_frag[2],
                        d32_frag[3],
                    )
                )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                        c32_frag[0],
                        c32_frag[1],
                        c32_frag[2],
                        c32_frag[3],
                        smem_raw.ptr_to([c_addr_1 - T.cast(smem, "int32")]),
                    )
                )
                _builder_emit(_mma_m16n8k16_bf16_zero(dc32_acc, d32_frag, c32_frag))
                _builder_emit(_mma_m16n8k16_bf16_zero_off4(dc32_acc, d32_frag, c32_frag))
                with T.unroll(4) as _ls:
                    _pk = _builder_scalar(
                        "_pk",
                        _mul_f32x2_inplace(
                            T.cuda.make_float2(dc32_acc[_ls * 2], dc32_acc[_ls * 2 + 1]),
                            T.cuda.make_float2(T.float32(-1.0), T.float32(-1.0)),
                        ),
                    )
                    T.buffer_store(dc32_acc, T.cuda.float2_x(_pk), [_ls * 2])
                    T.buffer_store(dc32_acc, T.cuda.float2_y(_pk), [_ls * 2 + 1])
                with T.unroll(4) as _lp:
                    T.buffer_store(
                        dc32_bf16,
                        T.cuda.float22bfloat162_rn(
                            dc32_acc[_lp * 2 + 0], dc32_acc[_lp * 2 + 1 + 0]
                        ),
                        [_lp],
                    )
                _builder_emit(
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                        a32_frag[0],
                        a32_frag[1],
                        a32_frag[2],
                        a32_frag[3],
                        smem_raw.ptr_to([a_addr_1 - T.cast(smem, "int32")]),
                    )
                )
                _apb = _builder_scalar(
                    "_apb",
                    lane_col // 16 * 1024 + lane_row_1 * 32 + lane_col % 16 * 2,
                    dtype="int32",
                )
                a_publish_addr = _builder_scalar(
                    "a_publish_addr",
                    smem_inv_addr
                    + T.cast(prep_stage, "int32") * 41984
                    + (_apb ^ (_apb >> 7 & 1) << 4),
                    dtype="int32",
                )
                _builder_emit(
                    T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                        smem_raw.ptr_to([a_publish_addr - T.cast(smem, "int32")]),
                        a32_frag[0],
                        a32_frag[1],
                        a32_frag[2],
                        a32_frag[3],
                    )
                )
                _builder_emit(_mma_m16n8k16_bf16_zero(o32_acc, dc32_bf16, a32_frag))
                _builder_emit(_mma_m16n8k16_bf16_zero_off4(o32_acc, dc32_bf16, a32_frag))
                with T.unroll(4) as _lp:
                    T.buffer_store(
                        o32_bf16,
                        T.cuda.float22bfloat162_rn(o32_acc[_lp * 2 + 0], o32_acc[_lp * 2 + 1 + 0]),
                        [_lp],
                    )
                _opb = _builder_scalar(
                    "_opb",
                    lane_col // 16 * 1024 + (16 + lane_row_1) * 32 + lane_col % 16 * 2,
                    dtype="int32",
                )
                o_publish_addr = _builder_scalar(
                    "o_publish_addr",
                    smem_inv_addr
                    + T.cast(prep_stage, "int32") * 41984
                    + (_opb ^ (_opb >> 7 & 1) << 4),
                    dtype="int32",
                )
                _builder_emit(
                    T.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                        smem_raw.ptr_to([o_publish_addr - T.cast(smem, "int32")]),
                        o32_bf16[0],
                        o32_bf16[1],
                        o32_bf16[2],
                        o32_bf16[3],
                    )
                )
                with T.unroll(4) as zero_word:
                    T.buffer_store(zero32_bf16, T.uint32(0), [zero_word])
                _zpb = _builder_scalar(
                    "_zpb",
                    (16 + lane_col) // 16 * 1024 + lane_row_1 * 32 + (16 + lane_col) % 16 * 2,
                    dtype="int32",
                )
                zero_publish_addr = _builder_scalar(
                    "zero_publish_addr",
                    smem_inv_addr
                    + T.cast(prep_stage, "int32") * 41984
                    + (_zpb ^ (_zpb >> 7 & 1) << 4),
                    dtype="int32",
                )
                _builder_emit(
                    T.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                        smem_raw.ptr_to([zero_publish_addr - T.cast(smem, "int32")]),
                        zero32_bf16[0],
                        zero32_bf16[1],
                        zero32_bf16[2],
                        zero32_bf16[3],
                    )
                )
                _builder_scope_exit(_builder_then_2187_12)
                _builder_else_2187_12 = _builder_scope_enter(T.Else())
                _builder_if_2322_12 = _builder_scope_enter(T.If(prep_local_warp == 1))
                _builder_then_2322_12 = _builder_scope_enter(T.Then())
                stage_f32_0_1 = _builder_scalar(
                    "stage_f32_0_1", T.cast(prep_stage, "int32") * 10496, dtype="int32"
                )
                restore_scale_1 = _builder_scalar(
                    "restore_scale_1",
                    _ld_shared_f32(smem_restore_factor_all, stage_f32_0_1 + 128),
                    dtype="f32",
                )
                restore_factor_1 = _builder_name("restore_factor_1", T.alloc_local((8,), "float32"))
                restore_segment_1 = _builder_scalar("restore_segment_1", lane & 15, dtype="int32")
                with T.unroll(2) as restore_vec_1:
                    _builder_emit(
                        _ld_shared_v4_f32(
                            restore_factor_1,
                            restore_vec_1 * 4,
                            smem_restore_factor_all,
                            stage_f32_0_1 + restore_segment_1 * 8 + restore_vec_1 * 4,
                        )
                    )
                with T.serial(0, 4, unroll=False) as restore_pass_1:
                    restore_row_1 = _builder_scalar(
                        "restore_row_1", restore_pass_1 * 2 + (lane >> 4), dtype="int32"
                    )
                    restore_qd_values_1 = _builder_name(
                        "restore_qd_values_1", T.alloc_local((8,), "float32")
                    )
                    restore_kd_values_1 = _builder_name(
                        "restore_kd_values_1", T.alloc_local((8,), "float32")
                    )
                    restore_ki_values_1 = _builder_name(
                        "restore_ki_values_1", T.alloc_local((8,), "float32")
                    )
                    packed_6 = _builder_name("packed_6", T.alloc_local((4,), "uint32", align=16))
                    _builder_emit(
                        _ld_shared_v4(
                            smem_raw,
                            T.cast(smem, "int32"),
                            packed_6,
                            smem_qd_addr
                            + T.cast(prep_stage, "int32") * 41984
                            + (
                                restore_segment_1 * 8 // 64 * 4096
                                + restore_row_1 * 128
                                + restore_segment_1 * 8 % 64 * 2
                                ^ (
                                    restore_segment_1 * 8 // 64 * 4096
                                    + restore_row_1 * 128
                                    + restore_segment_1 * 8 % 64 * 2
                                    >> 7
                                    & 7
                                )
                                << 4
                            ),
                        )
                    )
                    packed_fp32_3 = _builder_name("packed_fp32_3", T.alloc_local((8,), "float32"))
                    with T.unroll(4) as _pair:
                        T.buffer_store(
                            packed_fp32_3,
                            T.cuda.uint_as_float(packed_6[_pair + 0] << T.uint32(16)),
                            [_pair * 2],
                        )
                        T.buffer_store(
                            packed_fp32_3,
                            T.cuda.uint_as_float(packed_6[_pair + 0] & T.uint32(4294901760)),
                            [_pair * 2 + 1],
                        )
                    with T.unroll(8) as value_idx_6:
                        T.buffer_store(
                            restore_qd_values_1, packed_fp32_3[value_idx_6], [value_idx_6]
                        )
                    packed_0_4 = _builder_name(
                        "packed_0_4", T.alloc_local((4,), "uint32", align=16)
                    )
                    _builder_emit(
                        _ld_shared_v4(
                            smem_raw,
                            T.cast(smem, "int32"),
                            packed_0_4,
                            smem_kd_addr
                            + T.cast(prep_stage, "int32") * 41984
                            + (
                                restore_segment_1 * 8 // 64 * 4096
                                + restore_row_1 * 128
                                + restore_segment_1 * 8 % 64 * 2
                                ^ (
                                    restore_segment_1 * 8 // 64 * 4096
                                    + restore_row_1 * 128
                                    + restore_segment_1 * 8 % 64 * 2
                                    >> 7
                                    & 7
                                )
                                << 4
                            ),
                        )
                    )
                    packed_0_fp32_2 = _builder_name(
                        "packed_0_fp32_2", T.alloc_local((8,), "float32")
                    )
                    with T.unroll(4) as _pair:
                        T.buffer_store(
                            packed_0_fp32_2,
                            T.cuda.uint_as_float(packed_0_4[_pair + 0] << T.uint32(16)),
                            [_pair * 2],
                        )
                        T.buffer_store(
                            packed_0_fp32_2,
                            T.cuda.uint_as_float(packed_0_4[_pair + 0] & T.uint32(4294901760)),
                            [_pair * 2 + 1],
                        )
                    with T.unroll(8) as value_idx_7:
                        T.buffer_store(
                            restore_kd_values_1, packed_0_fp32_2[value_idx_7], [value_idx_7]
                        )
                    packed_1_3 = _builder_name(
                        "packed_1_3", T.alloc_local((4,), "uint32", align=16)
                    )
                    _builder_emit(
                        _ld_shared_v4(
                            smem_raw,
                            T.cast(smem, "int32"),
                            packed_1_3,
                            smem_ki_addr
                            + T.cast(prep_stage, "int32") * 41984
                            + (
                                restore_segment_1 * 8 // 64 * 4096
                                + restore_row_1 * 128
                                + restore_segment_1 * 8 % 64 * 2
                                ^ (
                                    restore_segment_1 * 8 // 64 * 4096
                                    + restore_row_1 * 128
                                    + restore_segment_1 * 8 % 64 * 2
                                    >> 7
                                    & 7
                                )
                                << 4
                            ),
                        )
                    )
                    packed_1_fp32_1 = _builder_name(
                        "packed_1_fp32_1", T.alloc_local((8,), "float32")
                    )
                    with T.unroll(4) as _pair:
                        T.buffer_store(
                            packed_1_fp32_1,
                            T.cuda.uint_as_float(packed_1_3[_pair + 0] << T.uint32(16)),
                            [_pair * 2],
                        )
                        T.buffer_store(
                            packed_1_fp32_1,
                            T.cuda.uint_as_float(packed_1_3[_pair + 0] & T.uint32(4294901760)),
                            [_pair * 2 + 1],
                        )
                    with T.unroll(8) as value_idx_8:
                        T.buffer_store(
                            restore_ki_values_1, packed_1_fp32_1[value_idx_8], [value_idx_8]
                        )
                    restore_kr_values_1 = _builder_name(
                        "restore_kr_values_1", T.alloc_local((8,), "float32")
                    )
                    with T.unroll(8) as restore_elem_3:
                        T.buffer_store(
                            restore_kr_values_1,
                            restore_ki_values_1[restore_elem_3] * restore_factor_1[restore_elem_3],
                            [restore_elem_3],
                        )
                    with T.unroll(4) as _ls:
                        _pk = _builder_scalar(
                            "_pk",
                            _mul_f32x2_inplace(
                                T.cuda.make_float2(
                                    restore_qd_values_1[_ls * 2], restore_qd_values_1[_ls * 2 + 1]
                                ),
                                T.cuda.make_float2(restore_scale_1, restore_scale_1),
                            ),
                        )
                        T.buffer_store(restore_qd_values_1, T.cuda.float2_x(_pk), [_ls * 2])
                        T.buffer_store(restore_qd_values_1, T.cuda.float2_y(_pk), [_ls * 2 + 1])
                    with T.unroll(4) as _ls:
                        _pk = _builder_scalar(
                            "_pk",
                            _mul_f32x2_inplace(
                                T.cuda.make_float2(
                                    restore_kd_values_1[_ls * 2], restore_kd_values_1[_ls * 2 + 1]
                                ),
                                T.cuda.make_float2(restore_scale_1, restore_scale_1),
                            ),
                        )
                        T.buffer_store(restore_kd_values_1, T.cuda.float2_x(_pk), [_ls * 2])
                        T.buffer_store(restore_kd_values_1, T.cuda.float2_y(_pk), [_ls * 2 + 1])
                    packed_2_2 = _builder_name("packed_2_2", T.alloc_local((4,), "uint32", align=4))
                    with T.unroll(4) as _lp:
                        T.buffer_store(
                            packed_2_2,
                            T.cuda.float22bfloat162_rn(
                                restore_qd_values_1[_lp * 2 + 0],
                                restore_qd_values_1[_lp * 2 + 1 + 0],
                            ),
                            [_lp],
                        )
                    with T.unroll(4) as word_7:
                        _builder_emit(
                            _st_shared_b32(
                                smem_raw,
                                T.cast(smem, "int32"),
                                smem_qd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    restore_segment_1 * 8 // 64 * 4096
                                    + restore_row_1 * 128
                                    + restore_segment_1 * 8 % 64 * 2
                                    ^ (
                                        restore_segment_1 * 8 // 64 * 4096
                                        + restore_row_1 * 128
                                        + restore_segment_1 * 8 % 64 * 2
                                        >> 7
                                        & 7
                                    )
                                    << 4
                                )
                                + word_7 * 4,
                                packed_2_2[word_7],
                            )
                        )
                    packed_3_1 = _builder_name("packed_3_1", T.alloc_local((4,), "uint32", align=4))
                    with T.unroll(4) as _lp:
                        T.buffer_store(
                            packed_3_1,
                            T.cuda.float22bfloat162_rn(
                                restore_kd_values_1[_lp * 2 + 0],
                                restore_kd_values_1[_lp * 2 + 1 + 0],
                            ),
                            [_lp],
                        )
                    with T.unroll(4) as word_8:
                        _builder_emit(
                            _st_shared_b32(
                                smem_raw,
                                T.cast(smem, "int32"),
                                smem_kd_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    restore_segment_1 * 8 // 64 * 4096
                                    + restore_row_1 * 128
                                    + restore_segment_1 * 8 % 64 * 2
                                    ^ (
                                        restore_segment_1 * 8 // 64 * 4096
                                        + restore_row_1 * 128
                                        + restore_segment_1 * 8 % 64 * 2
                                        >> 7
                                        & 7
                                    )
                                    << 4
                                )
                                + word_8 * 4,
                                packed_3_1[word_8],
                            )
                        )
                    packed_4_1 = _builder_name("packed_4_1", T.alloc_local((4,), "uint32", align=4))
                    with T.unroll(4) as _lp:
                        T.buffer_store(
                            packed_4_1,
                            T.cuda.float22bfloat162_rn(
                                restore_kr_values_1[_lp * 2 + 0],
                                restore_kr_values_1[_lp * 2 + 1 + 0],
                            ),
                            [_lp],
                        )
                    with T.unroll(4) as word_9:
                        _builder_emit(
                            _st_shared_b32(
                                smem_raw,
                                T.cast(smem, "int32"),
                                smem_kr_trans_addr
                                + T.cast(prep_stage, "int32") * 41984
                                + (
                                    restore_segment_1 * 8 // 64 * 4096
                                    + restore_row_1 * 128
                                    + restore_segment_1 * 8 % 64 * 2
                                    ^ (
                                        restore_segment_1 * 8 // 64 * 4096
                                        + restore_row_1 * 128
                                        + restore_segment_1 * 8 % 64 * 2
                                        >> 7
                                        & 7
                                    )
                                    << 4
                                )
                                + word_9 * 4,
                                packed_4_1[word_9],
                            )
                        )
                _builder_scope_exit(_builder_then_2322_12)
                _builder_scope_exit(_builder_if_2322_12)
                _builder_scope_exit(_builder_else_2187_12)
                _builder_scope_exit(_builder_if_2187_12)
                _builder_emit(_fence_async_shared())
                _builder_if_2452_12 = _builder_scope_enter(T.If(prep_instance == 0))
                _builder_then_2452_12 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(11), T.uint32(128)))
                _builder_scope_exit(_builder_then_2452_12)
                _builder_else_2452_12 = _builder_scope_enter(T.Else())
                _builder_if_2454_12 = _builder_scope_enter(T.If(prep_instance == 1))
                _builder_then_2454_12 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(12), T.uint32(128)))
                _builder_scope_exit(_builder_then_2454_12)
                _builder_else_2454_12 = _builder_scope_enter(T.Else())
                _builder_if_2457_16 = _builder_scope_enter(T.If(prep_instance == 2))
                _builder_then_2457_16 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(13), T.uint32(128)))
                _builder_scope_exit(_builder_then_2457_16)
                _builder_else_2457_16 = _builder_scope_enter(T.Else())
                _builder_if_2459_16 = _builder_scope_enter(T.If(prep_instance == 3))
                _builder_then_2459_16 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.bar.sync(T.uint32(14), T.uint32(128)))
                _builder_scope_exit(_builder_then_2459_16)
                _builder_else_2459_16 = _builder_scope_enter(T.Else())
                _builder_emit(T.ptx.bar.sync(T.uint32(15), T.uint32(128)))
                _builder_scope_exit(_builder_else_2459_16)
                _builder_scope_exit(_builder_if_2459_16)
                _builder_scope_exit(_builder_else_2457_16)
                _builder_scope_exit(_builder_if_2457_16)
                _builder_scope_exit(_builder_else_2454_12)
                _builder_scope_exit(_builder_if_2454_12)
                _builder_scope_exit(_builder_else_2452_12)
                _builder_scope_exit(_builder_if_2452_12)
                _builder_if_2463_12 = _builder_scope_enter(T.If(prep_local_warp == 0))
                _builder_then_2463_12 = _builder_scope_enter(T.Then())
                _builder_if_2464_16 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                _builder_then_2464_16 = _builder_scope_enter(T.Then())
                _builder_emit(
                    _mbarrier_arrive(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_QK_FULL_OFF + prep_stage * 8,
                    )
                )
                _builder_scope_exit(_builder_then_2464_16)
                _builder_scope_exit(_builder_if_2464_16)
                _builder_scope_exit(_builder_then_2463_12)
                _builder_scope_exit(_builder_if_2463_12)
                with T.serial(0, 5) as _advance:
                    T.buffer_store(prep_stage.buffer, prep_stage + 1, [0])
                    _builder_if_2472_16 = _builder_scope_enter(T.If(prep_stage == 5))
                    _builder_then_2472_16 = _builder_scope_enter(T.Then())
                    T.buffer_store(prep_stage.buffer, T.uint32(0), [0])
                    T.buffer_store(
                        _phase_raw_inputs_free.buffer, _phase_raw_inputs_free ^ T.uint32(1), [0]
                    )
                    T.buffer_store(
                        _phase_gate_raw_full.buffer, _phase_gate_raw_full ^ T.uint32(1), [0]
                    )
                    T.buffer_store(_phase_smem_free.buffer, _phase_smem_free ^ T.uint32(1), [0])
                    T.buffer_store(_phase_qk_raw_full.buffer, _phase_qk_raw_full ^ T.uint32(1), [0])
                    T.buffer_store(
                        _phase_prep_diag_ready.buffer, _phase_prep_diag_ready ^ T.uint32(1), [0]
                    )
                    T.buffer_store(
                        _phase_prep_inv16_ready.buffer, _phase_prep_inv16_ready ^ T.uint32(1), [0]
                    )
                    _builder_scope_exit(_builder_then_2472_16)
                    _builder_scope_exit(_builder_if_2472_16)
            _builder_scope_exit(_builder_then_907_4)
            _builder_scope_exit(_builder_if_907_4)
            _builder_scope_exit(_builder_else_828_4)
            _builder_scope_exit(_builder_if_828_4)
            _builder_scope_exit(_builder_else_700_4)
            _builder_scope_exit(_builder_if_700_4)
            _builder_scope_exit(_builder_else_557_4)
            _builder_scope_exit(_builder_if_557_4)
            _builder_scope_exit(_builder_else_295_4)
            _builder_scope_exit(_builder_if_295_4)
    return builder.get()


def bf16_fused_m128(**kwargs: Any):
    cfg = _cfg(**kwargs)
    kernel = _build_kernel(
        total_tokens=cfg.total_tokens,
        h=cfg.num_heads,
        num_seqs=cfg.num_seqs,
        beta_tma_tokens=cfg.beta_tma_tokens,
        beta_tma_heads=cfg.beta_tma_heads,
        scale=1.0 / math.sqrt(D_HEAD),
        lower_bound=cfg.lower_bound,
        use_initial_state=cfg.use_initial_state,
        store_final_state=cfg.store_final_state,
    )
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

    ref_out, ref_state = _reference_torch(case)
    torch.testing.assert_close(case["out"], ref_out, rtol=4.01 / 128, atol=5e-3)
    if cfg.store_final_state:
        torch.testing.assert_close(case["final_state"], ref_state, rtol=4.01 / 128, atol=5e-3)

    flashinfer_out, flashinfer_state = _flashinfer_cuda_reference(case)
    torch.testing.assert_close(case["out"], flashinfer_out, rtol=4.01 / 128, atol=5e-3)
    if cfg.store_final_state and flashinfer_state is not None:
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
