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
import os
import sys
from dataclasses import dataclass, fields
from typing import Any
from unittest import SkipTest

import torch

from tvm.script import tirx as T

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
# Helpers whose PTX instruction form has an exact T.ptx/T.cuda wrapper use that
# wrapper.  Helpers with no exact wrapper (token-returning waits, ticks waits,
# raw-descriptor MMA, raw tensor-map TMA) keep the verbatim CUDA body behind a
# target-local T.cuda.func_call inline fallback (see tirx_dsl_improve.md).
# ---------------------------------------------------------------------------

_FLASHKDA_SRCS = {
    "flashkda_tensormap_acquire": r"""__device__ __forceinline__ void flashkda_tensormap_acquire(const void *tmap_ptr) {
    asm volatile(
        "fence.proxy.tensormap::generic.acquire.gpu [%0], 128;\n"
        :: "l"(tmap_ptr) : "memory");
}
""",
    "flashkda_tanh_approx": r"""// tanh.approx.f32 (sigmoid via tanh; used throughout the prep role)
__device__ __forceinline__ float flashkda_tanh_approx(float x) {
    float y;
    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
""",
    "flashkda_fmaf_rn": r"""// __fmaf_rn (value form of fma.rn.f32)
__device__ __forceinline__ float flashkda_fmaf_rn(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
}
""",
    "flashkda_rsqrtf": r"""// rsqrtf
__device__ __forceinline__ float flashkda_rsqrtf(float x) {
    return rsqrtf(x);
}
""",
}


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


def _tanh_approx(x):
    # tanh.approx.f32 (prep-role sigmoid)
    return T.cuda.func_call(
        "flashkda_tanh_approx",
        x,
        source_code=_FLASHKDA_SRCS["flashkda_tanh_approx"],
        return_type="float32",
    )


def _expf(x):
    # .cu:1317 __expf -> ex2.approx.ftz.f32(x * log2(e)) under -use_fast_math
    _out = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.ex2.approx.ftz.f32(_out[0], x * T.float32(1.4426950408889634)))
    return _out[0]


def _rsqrtf(x):
    # rsqrtf verbatim
    return T.cuda.func_call(
        "flashkda_rsqrtf", x, source_code=_FLASHKDA_SRCS["flashkda_rsqrtf"], return_type="float32"
    )


def _fmaf_rn(a, b, c):
    # __fmaf_rn verbatim (value form)
    return T.cuda.func_call(
        "flashkda_fmaf_rn",
        a,
        b,
        c,
        source_code=_FLASHKDA_SRCS["flashkda_fmaf_rn"],
        return_type="float32",
    )


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


def _encode_tma_descriptors(
    case: dict[str, Any],
    *,
    output_key: str = "out",
    descriptor_storage_key: str = "descriptor_storage",
) -> None:
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
        case[output_key].data_ptr(),
        4,
        [64, tokens, h, D_HEAD // 64],
        [h * D_HEAD * 2, D_HEAD * 2, 64 * 2],
        [64, 32, 1, 2],
        _CU_TENSOR_MAP_SWIZZLE_128B,
    )
    case[descriptor_storage_key].copy_(
        torch.frombuffer(maps, dtype=torch.uint8).to(case[descriptor_storage_key].device)
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

KERNEL_META = {"name": "flashkda_bf16_fused_m128", "category": "flashkda", "compute_capability": 10}


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
    final_state_tx_tile = torch.empty_like(final_state)
    out = torch.empty(shape, device=device, dtype=torch.bfloat16)
    out_tx_tile = torch.empty_like(out)
    # Six CUtensorMap descriptors, 128 B each, 64 B aligned (torch storages are
    # 512 B aligned); the reference kernel publishes descriptors here.
    descriptor_storage = torch.empty((768,), device=device, dtype=torch.uint8)
    descriptor_storage_tx_tile = torch.empty_like(descriptor_storage)

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
        "out_tx_tile": out_tx_tile,
        "final_state": final_state,
        "final_state_tx_tile": final_state_tx_tile,
        "descriptor_storage": descriptor_storage,
        "descriptor_storage_tx_tile": descriptor_storage_tx_tile,
        "scale": 1.0 / math.sqrt(D_HEAD),
        "dispatch_reason": _m128_dispatch_reason(cfg),
    }
    if device != "cpu" and torch.device(device).type != "cpu":
        _encode_tma_descriptors(case)
        _encode_tma_descriptors(
            case, output_key="out_tx_tile", descriptor_storage_key="descriptor_storage_tx_tile"
        )
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
    """Import flashinfer.recurrent_kda from the PR head worktree.

    The flashinfer-ai/flashinfer#4262 head worktree is located via the
    FLASHKDA_PR_WORKTREE environment variable.
    Handles the case where an installed flashinfer (main, without kda.py) was
    already imported by purging it and re-importing from the worktree.
    """
    worktree = os.environ.get("FLASHKDA_PR_WORKTREE")
    if worktree is None:
        try:
            from flashinfer.kda import recurrent_kda

            return recurrent_kda
        except ImportError as e:
            raise RuntimeError(
                "flashinfer.kda is unavailable; set FLASHKDA_PR_WORKTREE to the "
                "flashinfer-ai/flashinfer#4262 head worktree containing the m128 kernel"
            ) from e
    if worktree not in sys.path:
        sys.path.insert(0, worktree)
    import flashinfer

    if not str(getattr(flashinfer, "__file__", "")).startswith(worktree):
        for mod_name in [
            m for m in list(sys.modules) if m == "flashinfer" or m.startswith("flashinfer.")
        ]:
            del sys.modules[mod_name]
        import flashinfer

        if not str(getattr(flashinfer, "__file__", "")).startswith(worktree):
            raise RuntimeError(
                f"failed to import flashinfer from PR worktree {worktree}; "
                f"got {getattr(flashinfer, '__file__', None)}"
            )
    from flashinfer.kda import recurrent_kda

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


def _tirx_args(case: dict[str, Any], *, use_tx_tile: bool = False) -> tuple[Any, ...]:
    suffix = "_tx_tile" if use_tx_tile else ""
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
        case[f"out{suffix}"].reshape(-1),
        case[f"final_state{suffix}"].reshape(-1),
        case[f"descriptor_storage{suffix}"],
    )


@T.jit
def _kernel(
    q: T.Buffer((total_tokens * h * D_HEAD,), "bfloat16"),
    k: T.Buffer((total_tokens * h * D_HEAD,), "bfloat16"),
    v: T.Buffer((total_tokens * h * D_HEAD,), "bfloat16"),
    g: T.Buffer((total_tokens * h * D_HEAD,), "bfloat16"),
    beta: T.Buffer((total_tokens * h,), "bfloat16"),
    beta_tma: T.Buffer((beta_tma_tokens, beta_tma_heads), "bfloat16"),
    A_log: T.Buffer((h,), "float32"),
    dt_bias: T.Buffer((h * D_HEAD,), "float32"),
    cu_seqlens: T.Buffer((num_seqs + 1,), "int64"),
    seq_order: T.Buffer((num_seqs,), "int32"),
    initial_state: T.Buffer((num_seqs * h * D_HEAD * D_HEAD,), "bfloat16"),
    out: T.Buffer((total_tokens * h * D_HEAD,), "bfloat16"),
    final_state: T.Buffer((num_seqs * h * D_HEAD * D_HEAD,), "bfloat16"),
    descriptor_storage: T.Buffer((768,), "uint8"),
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
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    # CUDA_TRANSCRIBE_START: kernel_flashkda_bf16_fused_m128,
    # flashkda_bf16_fused_m128.cu:504 (tensor-map acquire fences), then source
    # order. Grid = num_seqs * h CTAs, 1024 threads, SMEM_TOTAL dynamic smem.
    block_idx = T.cta_id([num_seqs * h])
    thread_idx = T.thread_id([THREADS])
    T.warpgroup_id([THREADS // 128])
    T.warp_id_in_wg([4])
    T.lane_id([32])
    T.thread_id_in_wg([128])

    pool = T.SMEMPool()
    smem_raw = pool.alloc((SMEM_TOTAL,), "uint8", align=1024)
    tmem_addr_storage = T.decl_buffer(
        (1,),
        "int32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_TMEM_ADDR_STORAGE_OFF,
        align=4,
    )
    smem_g_raw_all = T.decl_buffer(
        (SMEM_SMEM_G_RAW_ALL_STAGE_BYTES // 2,),
        "bfloat16",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_SMEM_G_RAW_ALL_OFF,
        align=1024,
    )
    smem_v_all = T.decl_buffer(
        (SMEM_SMEM_V_ALL_STAGE_BYTES // 2,),
        "bfloat16",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_SMEM_V_ALL_OFF,
        align=1024,
    )
    smem_gate_all = T.decl_buffer(
        (SMEM_SMEM_GATE_ALL_STAGE_BYTES // 4,),
        "float32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_SMEM_GATE_ALL_OFF,
        align=1024,
    )
    smem_gt_all = T.decl_buffer(
        (SMEM_SMEM_GT_ALL_STAGE_BYTES // 4,),
        "float32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_SMEM_GT_ALL_OFF,
        align=1024,
    )
    smem_gt_prefix_all = T.decl_buffer(
        (SMEM_SMEM_GT_PREFIX_ALL_STAGE_BYTES // 4,),
        "float32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_SMEM_GT_PREFIX_ALL_OFF,
        align=1024,
    )
    smem_restore_factor_all = T.decl_buffer(
        (SMEM_SMEM_RESTORE_FACTOR_ALL_STAGE_BYTES // 4,),
        "float32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_SMEM_RESTORE_FACTOR_ALL_OFF,
        align=1024,
    )
    smem_prep_beta_all = T.decl_buffer(
        (SMEM_SMEM_PREP_BETA_ALL_STAGE_BYTES // 4,),
        "float32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_SMEM_PREP_BETA_ALL_OFF,
        align=1024,
    )
    smem_gate_rate_all = T.decl_buffer(
        (SMEM_SMEM_GATE_RATE_ALL_STAGE_BYTES // 4,),
        "float32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_SMEM_GATE_RATE_ALL_OFF,
        align=1024,
    )
    pool.commit()

    # Kernel TMA descriptor pointers (descriptor_storage slots, .cu:501 ABI).
    q_tma = descriptor_storage.ptr_to([TMA_SLOT_Q])
    k_tma = descriptor_storage.ptr_to([TMA_SLOT_K])
    v_tma = descriptor_storage.ptr_to([TMA_SLOT_V])
    g_tma = descriptor_storage.ptr_to([TMA_SLOT_G])
    beta_tma_tmap = descriptor_storage.ptr_to([TMA_SLOT_BETA])
    out_tma = descriptor_storage.ptr_to([TMA_SLOT_OUT])

    # .cu:504-521 (FLASHINFER INTEGRATION: acquire global tensor maps).
    if thread_idx == 0:
        T.cuda.func_call(
            "flashkda_tensormap_acquire",
            q_tma,
            source_code=_FLASHKDA_SRCS["flashkda_tensormap_acquire"],
            return_type="void",
        )
        T.cuda.func_call(
            "flashkda_tensormap_acquire",
            k_tma,
            source_code=_FLASHKDA_SRCS["flashkda_tensormap_acquire"],
            return_type="void",
        )
        T.cuda.func_call(
            "flashkda_tensormap_acquire",
            v_tma,
            source_code=_FLASHKDA_SRCS["flashkda_tensormap_acquire"],
            return_type="void",
        )
        T.cuda.func_call(
            "flashkda_tensormap_acquire",
            g_tma,
            source_code=_FLASHKDA_SRCS["flashkda_tensormap_acquire"],
            return_type="void",
        )
        T.cuda.func_call(
            "flashkda_tensormap_acquire",
            beta_tma_tmap,
            source_code=_FLASHKDA_SRCS["flashkda_tensormap_acquire"],
            return_type="void",
        )
        T.cuda.func_call(
            "flashkda_tensormap_acquire",
            out_tma,
            source_code=_FLASHKDA_SRCS["flashkda_tensormap_acquire"],
            return_type="void",
        )
    T.cuda.cta_sync()  # .cu:520 __syncthreads()

    # .cu:522-524
    tid: T.let = thread_idx
    warp = _make_warp_uniform(T.cast(tid, "uint32") // T.uint32(32))
    lane: T.let = tid % 32
    # .cu:526-528
    smem: T.uint32 = T.cuda.cvta_generic_to_shared(T.address_of(smem_raw[0]))
    # .cu:530-531
    bid: T.let = block_idx
    num_bids: T.let = num_seqs * h

    # .cu:533-577 Kernel setup ops (smem aliases; int addresses).
    smem_qd_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_QD_OFF
    smem_g_raw_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_G_RAW_OFF
    smem_g_raw_all_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_G_RAW_ALL_OFF
    smem_kd_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_KD_OFF
    smem_q_raw_prefetch_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_Q_RAW_PREFETCH_OFF
    smem_final_trans_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_FINAL_TRANS_OFF
    smem_kr_trans_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_KR_TRANS_OFF
    smem_mqk_trans_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_MQK_TRANS_OFF
    smem_inv_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_INV_OFF
    smem_v_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_V_OFF
    smem_ki_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_KI_OFF
    smem_gate_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_GATE_OFF
    smem_beta_raw_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_BETA_RAW_OFF
    smem_inv_work_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_INV_WORK_OFF
    smem_out_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_OUT_OFF
    smem_restore_factor_all_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_RESTORE_FACTOR_ALL_OFF
    smem_gt_prefix_all_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_GT_PREFIX_ALL_OFF
    smem_gt_all_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_GT_ALL_OFF
    smem_prep_beta_all_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_PREP_BETA_ALL_OFF
    smem_gate_rate_all_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_GATE_RATE_ALL_OFF
    smem_v_all_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_V_ALL_OFF
    smem_gate_all_addr: T.int32 = T.cast(smem, "int32") + SMEM_SMEM_GATE_ALL_OFF

    # .cu:579-682 Mbarrier init (17 groups, 77 barriers at smem_raw[0..616)).
    if warp == 0:
        leader = T.cuda.elect_sync()
        if leader:
            # qk_full: 5 barriers, init_count=1
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([8]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([16]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([24]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([32]), T.uint32(1))
            # gate_raw_full: 5 barriers, init_count=1
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([40]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([48]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([56]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([64]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([72]), T.uint32(1))
            # qk_raw_full: 5 barriers, init_count=1
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([80]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([88]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([96]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([104]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([112]), T.uint32(1))
            # v_full: 5 barriers, init_count=1
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([120]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([128]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([136]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([144]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([152]), T.uint32(1))
            # v_free: 5 barriers, init_count=4
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([160]), T.uint32(4))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([168]), T.uint32(4))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([176]), T.uint32(4))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([184]), T.uint32(4))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([192]), T.uint32(4))
            # smem_free: 5 barriers, init_count=1
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([200]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([208]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([216]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([224]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([232]), T.uint32(1))
            # raw_inputs_free: 5 barriers, init_count=1
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([240]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([248]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([256]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([264]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([272]), T.uint32(1))
            # state_inp_ready: 5 barriers, init_count=4
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([280]), T.uint32(4))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([288]), T.uint32(4))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([296]), T.uint32(4))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([304]), T.uint32(4))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([312]), T.uint32(4))
            # old_out_ready: 5 barriers, init_count=1
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([320]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([328]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([336]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([344]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([352]), T.uint32(1))
            # u_inp_ready: 5 barriers, init_count=4
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([360]), T.uint32(4))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([368]), T.uint32(4))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([376]), T.uint32(4))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([384]), T.uint32(4))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([392]), T.uint32(4))
            # u2_acc_ready: 5 barriers, init_count=1
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([400]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([408]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([416]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([424]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([432]), T.uint32(1))
            # u2_inp_ready: 5 barriers, init_count=4
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([440]), T.uint32(4))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([448]), T.uint32(4))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([456]), T.uint32(4))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([464]), T.uint32(4))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([472]), T.uint32(4))
            # final_ready: 5 barriers, init_count=1
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([480]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([488]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([496]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([504]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([512]), T.uint32(1))
            # out_empty: 1 barriers, init_count=1
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([520]), T.uint32(1))
            # tmem_dealloc_ready: 1 barriers, init_count=2
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([528]), T.uint32(2))
            # prep_diag_ready: 5 barriers, init_count=2
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([536]), T.uint32(2))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([544]), T.uint32(2))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([552]), T.uint32(2))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([560]), T.uint32(2))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([568]), T.uint32(2))
            # prep_inv16_ready: 5 barriers, init_count=2
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([576]), T.uint32(2))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([584]), T.uint32(2))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([592]), T.uint32(2))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([600]), T.uint32(2))
            T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([608]), T.uint32(2))
        T.ptx.fence.mbarrier_init.release.cluster()  # .cu:679 fence.mbarrier_init.release.cluster
    T.cuda.cta_sync()  # .cu:682 __syncthreads()

    # .cu:684-689 TMEM alloc (256 columns, 256 used).
    if warp == 0:
        T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
            T.address_of(tmem_addr_storage[0]), T.uint32(256)
        )
    T.cuda.cta_sync()  # .cu:691 __syncthreads()
    T.ptx.tcgen05.fence__after_thread_sync()  # .cu:692

    # .cu:694-712 mbarrier group bases + taddr read (volatile int load).
    mbar_base: T.int32 = T.cast(smem, "int32")
    taddr: T.int32
    T.ptx.ld.volatile.shared.s32(taddr, T.address_of(tmem_addr_storage[0]))

    # .cu:714-721 Kernel post-init ops (TMEM column offsets).
    tmem_tmem_state: T.int32 = taddr + TMEM_TMEM_STATE_OFFSET
    tmem_tmem_state_inp: T.int32 = taddr + TMEM_TMEM_STATE_INP_OFFSET
    tmem_tmem_u_acc: T.int32 = taddr + TMEM_TMEM_U_ACC_OFFSET
    tmem_tmem_u2_inp: T.int32 = taddr + TMEM_TMEM_U2_INP_OFFSET
    tmem_tmem_u2_acc: T.int32 = taddr + TMEM_TMEM_U2_ACC_OFFSET
    tmem_tmem_out: T.int32 = taddr + TMEM_TMEM_OUT_OFFSET
    tmem_tmem_state_out: T.int32 = taddr + TMEM_TMEM_STATE_OUT_OFFSET

    # .cu:723-727 Register redistribution: dec phase first (warps 8-11).
    if warp >= 8 and warp <= 11:
        T.ptx.setmaxnreg.dec.sync.aligned.u32(48)

    # ---- Role dispatch (.cu:729-2550); role bodies transcribed in source order.
    if warp <= 3:
        # ---- Role: compute (.cu:730-963) ----
        T.ptx.setmaxnreg.inc.sync.aligned.u32(168)  # .cu:731
        # .cu:732 compute_main
        task_idx: T.int32 = bid  # .cu:733
        seq_idx: T.int32 = seq_order[task_idx // h]  # .cu:734
        head_idx: T.int32 = task_idx % h  # .cu:735
        bos: T.int64 = cu_seqlens[seq_idx]  # .cu:736
        eos: T.int64 = cu_seqlens[seq_idx + 1]  # .cu:737
        seq_len: T.int32 = T.cast(eos - bos, "int32")  # .cu:738
        num_chunks: T.int32 = (seq_len + 32 - 1) // 32  # .cu:739
        warp_in_wg: T.int32 = warp % 4  # .cu:740
        tmem_row_base: T.int32 = (warp_in_wg * 32) << 16  # .cu:741
        state_row: T.int32 = warp_in_wg * 32 + lane  # .cu:742
        warp_id_in_role: T.int32 = warp - 0  # .cu:743
        compute_local_warp: T.int32 = warp_id_in_role  # .cu:744
        state_base: T.int64 = (
            (T.cast(seq_idx, "int64") * T.cast(h, "int64") + T.cast(head_idx, "int64")) * 128
            + T.cast(state_row, "int64")
        ) * 128  # .cu:745
        initial_state_u32 = T.decl_buffer(
            (num_seqs * h * D_HEAD * D_HEAD // 2,), "uint32", data=initial_state.data
        )
        final_state_u32 = T.decl_buffer(
            (num_seqs * h * D_HEAD * D_HEAD // 2,), "uint32", data=final_state.data
        )
        for state_col_block in T.unroll(4):  # .cu:746-822 (#pragma unroll)
            state_frag: T.f32[32]
            for _zi in T.unroll(32):  # .cu:749-780
                state_frag[_zi] = T.float32(0.0)
            if use_initial_state:  # .cu:781
                # .cu:783-800: 2x uint4 loads + bf16->f32 shl/and unpack (frag 0-15)
                for _blk in T.unroll(2):
                    _vld = T.alloc_local((4,), "uint32", align=16)
                    _ld_global_v4_u32(
                        _vld,
                        initial_state_u32.ptr_to(
                            [(state_base + state_col_block * 32) // 2 + _blk * 4]
                        ),
                    )
                    for _pair in T.unroll(4):
                        state_frag[0 + _blk * 8 + _pair * 2] = T.cuda.uint_as_float(
                            _vld[_pair] << T.uint32(16)
                        )
                        state_frag[0 + _blk * 8 + _pair * 2 + 1] = T.cuda.uint_as_float(
                            _vld[_pair] & T.uint32(0xFFFF0000)
                        )
                # .cu:801-819: same at +16 elements (frag 16-31)
                for _blk in T.unroll(2):
                    _vld1 = T.alloc_local((4,), "uint32", align=16)
                    _ld_global_v4_u32(
                        _vld1,
                        initial_state_u32.ptr_to(
                            [(state_base + state_col_block * 32 + 16) // 2 + _blk * 4]
                        ),
                    )
                    for _pair in T.unroll(4):
                        state_frag[16 + _blk * 8 + _pair * 2] = T.cuda.uint_as_float(
                            _vld1[_pair] << T.uint32(16)
                        )
                        state_frag[16 + _blk * 8 + _pair * 2 + 1] = T.cuda.uint_as_float(
                            _vld1[_pair] & T.uint32(0xFFFF0000)
                        )
            _tmem_st_x32_f32(
                taddr + 64 + tmem_row_base + state_col_block * 32, state_frag
            )  # .cu:821
        T.ptx.tcgen05.wait__st.sync.aligned()  # .cu:823 tcgen05.wait::st.sync.aligned
        compute_stage: T.uint32 = 0  # .cu:824
        _phase_qk_full: T.uint32 = 0  # .cu:825
        _phase_v_full: T.uint32 = 0  # .cu:826
        _phase_old_out_ready: T.uint32 = 0  # .cu:827
        _phase_u2_acc_ready: T.uint32 = 0  # .cu:828
        _phase_final_ready: T.uint32 = 0  # .cu:829
        for chunk_idx in T.serial(0, num_chunks, unroll=False):  # .cu:830-831 (#pragma unroll 1)
            _mbarrier_wait(
                smem_raw,
                T.cast(smem, "int32"),
                mbar_base + MBAR_QK_FULL_OFF + compute_stage * 8,
                _phase_qk_full,
            )  # .cu:832
            for state_col_block_1 in T.serial(0, 4, unroll=False):  # .cu:833-834 (#pragma unroll 1)
                state_addr: T.int32 = taddr + 64 + tmem_row_base + state_col_block_1 * 32  # .cu:835
                _tmem_load_0: T.f32[32]
                _tmem_ld_x32(_tmem_load_0, state_addr)  # .cu:836-837
                _tmem_load_0_bf16: T.uint32[16]
                for _lp in T.unroll(16):  # .cu:839-843
                    _tmem_load_0_bf16[_lp] = T.cuda.float22bfloat162_rn(
                        _tmem_load_0[_lp * 2 + 0], _tmem_load_0[_lp * 2 + 1 + 0]
                    )
                # .cu:844-848 (inline tcgen05.st x16 of packed bf16 pairs)
                T.ptx["tcgen05.st.sync.aligned.32x32b.x16.b32"](
                    T.cast(taddr + tmem_row_base + state_col_block_1 * 16, "uint32"),
                    *[_tmem_load_0_bf16[_j] for _j in range(16)],
                )
                state_scale: T.f32[16]
                for state_half in T.unroll(2):  # .cu:850-859
                    for state_col in T.unroll(16):
                        state_scale[state_col] = smem_gt_all[
                            compute_stage * 10496
                            + state_col_block_1 * 32
                            + state_half * 16
                            + state_col
                        ]  # .cu:854
                    for _ls in T.unroll(8):  # .cu:857-858
                        _pk = _mul_f32x2_inplace(
                            T.cuda.make_float2(
                                _tmem_load_0[state_half * 16 + _ls * 2],
                                _tmem_load_0[state_half * 16 + _ls * 2 + 1],
                            ),
                            T.cuda.make_float2(state_scale[_ls * 2], state_scale[_ls * 2 + 1]),
                        )
                        _tmem_load_0[state_half * 16 + _ls * 2] = T.cuda.float2_x(_pk)
                        _tmem_load_0[state_half * 16 + _ls * 2 + 1] = T.cuda.float2_y(_pk)
                _tmem_st_x32_f32(state_addr, _tmem_load_0)  # .cu:860
            T.ptx.tcgen05.wait__st.sync.aligned()  # .cu:862
            if T.cuda.elect_sync():  # .cu:863-865
                _mbarrier_arrive(
                    smem_raw,
                    T.cast(smem, "int32"),
                    mbar_base + MBAR_STATE_INP_READY_OFF + compute_stage * 8,
                )
            _mbarrier_wait(
                smem_raw,
                T.cast(smem, "int32"),
                mbar_base + MBAR_V_FULL_OFF + compute_stage * 8,
                _phase_v_full,
            )  # .cu:866
            _mbarrier_wait(
                smem_raw,
                T.cast(smem, "int32"),
                mbar_base + MBAR_OLD_OUT_READY_OFF + compute_stage * 8,
                _phase_old_out_ready,
            )  # .cu:867
            _tmem_load_1: T.f32[32]
            _tmem_ld_x32(_tmem_load_1, taddr + 224 + tmem_row_base)  # .cu:868-869
            for residual_half in T.unroll(2):  # .cu:870-895
                residual_v: T.f32[16]
                residual_beta: T.f32[16]
                for residual_col in T.unroll(16):  # .cu:874-881
                    token_col: T.int32 = residual_half * 16 + residual_col  # .cu:876
                    residual_v[residual_col] = T.cuda.bfloat162float(
                        smem_v_all[compute_stage * 20992 + token_col * 128 + state_row]
                    )  # .cu:877-879
                    residual_beta[residual_col] = smem_prep_beta_all[
                        compute_stage * 10496 + token_col
                    ]  # .cu:880
                for _ls in T.unroll(8):  # .cu:882-884
                    _pk = _sub_f32x2_inplace(
                        T.cuda.make_float2(residual_v[_ls * 2], residual_v[_ls * 2 + 1]),
                        T.cuda.make_float2(
                            _tmem_load_1[residual_half * 16 + _ls * 2],
                            _tmem_load_1[residual_half * 16 + _ls * 2 + 1],
                        ),
                    )
                    residual_v[_ls * 2] = T.cuda.float2_x(_pk)
                    residual_v[_ls * 2 + 1] = T.cuda.float2_y(_pk)
                for _ls in T.unroll(8):  # .cu:885-887
                    _pk = _mul_f32x2_inplace(
                        T.cuda.make_float2(residual_v[_ls * 2], residual_v[_ls * 2 + 1]),
                        T.cuda.make_float2(residual_beta[_ls * 2], residual_beta[_ls * 2 + 1]),
                    )
                    residual_v[_ls * 2] = T.cuda.float2_x(_pk)
                    residual_v[_ls * 2 + 1] = T.cuda.float2_y(_pk)
                residual_v_bf16: T.uint32[8]
                for _lp in T.unroll(8):  # .cu:888-893
                    residual_v_bf16[_lp] = T.cuda.float22bfloat162_rn(
                        residual_v[_lp * 2 + 0], residual_v[_lp * 2 + 1 + 0]
                    )
                _tmem_st_x8_u32(
                    taddr + 224 + tmem_row_base + residual_half * 8, residual_v_bf16
                )  # .cu:894
            T.ptx.tcgen05.wait__st.sync.aligned()  # .cu:896
            if T.cuda.elect_sync():  # .cu:897-900
                _mbarrier_arrive(
                    smem_raw, T.cast(smem, "int32"), mbar_base + MBAR_V_FREE_OFF + compute_stage * 8
                )
                _mbarrier_arrive(
                    smem_raw,
                    T.cast(smem, "int32"),
                    mbar_base + MBAR_U_INP_READY_OFF + compute_stage * 8,
                )
            _mbarrier_wait(
                smem_raw,
                T.cast(smem, "int32"),
                mbar_base + MBAR_U2_ACC_READY_OFF + compute_stage * 8,
                _phase_u2_acc_ready,
            )  # .cu:901
            _tmem_load_2: T.f32[32]
            _tmem_ld_x32(_tmem_load_2, taddr + tmem_row_base)  # .cu:902-903
            _tmem_load_2_bf16: T.uint32[16]
            for _lp in T.unroll(16):  # .cu:904-909
                _tmem_load_2_bf16[_lp] = T.cuda.float22bfloat162_rn(
                    _tmem_load_2[_lp * 2 + 0], _tmem_load_2[_lp * 2 + 1 + 0]
                )
            # .cu:910-914 (inline tcgen05.st x16)
            T.ptx["tcgen05.st.sync.aligned.32x32b.x16.b32"](
                T.cast(taddr + 224 + tmem_row_base, "uint32"),
                *[_tmem_load_2_bf16[_j] for _j in range(16)],
            )
            T.ptx.tcgen05.wait__st.sync.aligned()  # .cu:915
            if T.cuda.elect_sync():  # .cu:916-918
                _mbarrier_arrive(
                    smem_raw,
                    T.cast(smem, "int32"),
                    mbar_base + MBAR_U2_INP_READY_OFF + compute_stage * 8,
                )
            _mbarrier_wait(
                smem_raw,
                T.cast(smem, "int32"),
                mbar_base + MBAR_FINAL_READY_OFF + compute_stage * 8,
                _phase_final_ready,
            )  # .cu:919
            compute_stage += 1  # .cu:920
            if compute_stage == 5:  # .cu:921
                compute_stage = T.uint32(0)
                _phase_qk_full = _phase_qk_full ^ T.uint32(1)
                _phase_v_full = _phase_v_full ^ T.uint32(1)
                _phase_old_out_ready = _phase_old_out_ready ^ T.uint32(1)
                _phase_u2_acc_ready = _phase_u2_acc_ready ^ T.uint32(1)
                _phase_final_ready = _phase_final_ready ^ T.uint32(1)
        if store_final_state:  # .cu:923-955
            for state_col_block_2 in T.unroll(4):
                _tmem_load_3: T.f32[32]
                _tmem_ld_x32(
                    _tmem_load_3, taddr + 64 + tmem_row_base + state_col_block_2 * 32
                )  # .cu:926-927
                for _half2 in T.unroll(2):  # .cu:928-953 (two 16-float groups)
                    _pk: T.uint32[8]
                    for _pj in T.unroll(8):
                        _pk[_pj] = T.cuda.float22bfloat162_rn(
                            _tmem_load_3[_half2 * 16 + _pj * 2],
                            _tmem_load_3[_half2 * 16 + _pj * 2 + 1],
                        )
                    _st_global_v4_u32(
                        final_state_u32.ptr_to(
                            [(state_base + state_col_block_2 * 32 + _half2 * 16) // 2]
                        ),
                        _pk[0],
                        _pk[1],
                        _pk[2],
                        _pk[3],
                    )  # .cu:938/951 first uint4
                    _st_global_v4_u32(
                        final_state_u32.ptr_to(
                            [(state_base + state_col_block_2 * 32 + _half2 * 16) // 2 + 4]
                        ),
                        _pk[4],
                        _pk[5],
                        _pk[6],
                        _pk[7],
                    )  # .cu:939/952 second uint4
        T.ptx.bar.sync(T.uint32(10), T.uint32(128))  # .cu:956 barrier.sync 10, 128
        if compute_local_warp == 0:  # .cu:957-961
            if T.cuda.elect_sync():
                _mbarrier_arrive(
                    smem_raw, T.cast(smem, "int32"), mbar_base + MBAR_TMEM_DEALLOC_READY_OFF
                )
    elif warp >= 4 and warp <= 7:
        # ---- Role: epilogue (.cu:964-1090) ----
        T.ptx.setmaxnreg.dec.sync.aligned.u32(48)  # .cu:965
        # .cu:966 epilogue_main
        task_idx_1: T.int32 = bid  # .cu:967
        seq_idx_1: T.int32 = seq_order[task_idx_1 // h]  # .cu:968
        head_idx_1: T.int32 = task_idx_1 % h  # .cu:969
        bos_1: T.int64 = cu_seqlens[seq_idx_1]  # .cu:970
        eos_1: T.int64 = cu_seqlens[seq_idx_1 + 1]  # .cu:971
        seq_len_1: T.int32 = T.cast(eos_1 - bos_1, "int32")  # .cu:972
        num_chunks_1: T.int32 = (seq_len_1 + 32 - 1) // 32  # .cu:973
        warp_id_in_role_1: T.int32 = warp - 4  # .cu:974
        epilogue_local_warp: T.int32 = warp_id_in_role_1  # .cu:975
        warp_in_wg_1: T.int32 = warp % 4  # .cu:976
        tmem_row_base_1: T.int32 = (warp_in_wg_1 * 32) << 16  # .cu:977
        state_row_1: T.int32 = warp_in_wg_1 * 32 + lane  # .cu:978
        epilogue_stage: T.uint32 = 0  # .cu:979
        output_stage: T.uint32 = 0  # .cu:980
        _phase_final_ready_1: T.uint32 = 0  # .cu:981
        for chunk_idx_1 in T.serial(0, num_chunks_1, unroll=False):  # .cu:982-983
            _mbarrier_wait(
                smem_raw,
                T.cast(smem, "int32"),
                mbar_base + MBAR_FINAL_READY_OFF + epilogue_stage * 8,
                _phase_final_ready_1,
            )  # .cu:984
            chunk_is_full: T.int32 = T.if_then_else(
                seq_len_1 >= (chunk_idx_1 + 1) * 32, 1, 0
            )  # .cu:985
            if chunk_is_full != 0:  # .cu:986
                # .cu:987-993 (inline tcgen05.ld.16x256b.x4, 16 uint32 regs)
                _tmem_load_4: T.uint32[16]
                T.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                    *[_tmem_load_4[_j] for _j in range(16)],
                    T.cast(taddr + 192 + tmem_row_base_1, "uint32"),
                )
                # .cu:994-1000 (same at TMEM row +16)
                _tmem_load_5: T.uint32[16]
                T.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                    *[_tmem_load_5[_j] for _j in range(16)],
                    T.cast(taddr + 192 + tmem_row_base_1 + 1048576, "uint32"),
                )
                T.ptx.tcgen05.wait__ld.sync.aligned()  # .cu:1001
                T.ptx.bar.sync(T.uint32(9), T.uint32(128))  # .cu:1002 barrier.sync 9, 128
                if epilogue_local_warp == 0:  # .cu:1003-1007
                    if T.cuda.elect_sync():
                        _mbarrier_arrive(
                            smem_raw, T.cast(smem, "int32"), mbar_base + MBAR_OUT_EMPTY_OFF
                        )
                if epilogue_local_warp == 0:  # .cu:1008-1012
                    if chunk_idx_1 >= 2:
                        T.ptx.cp.async_.bulk.wait_group.read(1)
                T.ptx.bar.sync(T.uint32(9), T.uint32(128))  # .cu:1013
                out_stage_addr: T.int32 = (
                    smem_out_addr + T.cast(output_stage, "int32") * 8192
                )  # .cu:1014
                for dim_half in T.unroll(2):  # .cu:1015-1049
                    out_packed = T.alloc_local((8,), "uint32", align=4)  # .cu:1017
                    if dim_half == 0:  # .cu:1018-1023
                        for _lp in T.unroll(8):
                            out_packed[_lp] = T.cuda.float22bfloat162_rn(
                                T.cuda.uint_as_float(_tmem_load_4[_lp * 2 + 0]),
                                T.cuda.uint_as_float(_tmem_load_4[_lp * 2 + 1 + 0]),
                            )
                    else:  # .cu:1024-1030
                        for _lp in T.unroll(8):
                            out_packed[_lp] = T.cuda.float22bfloat162_rn(
                                T.cuda.uint_as_float(_tmem_load_5[_lp * 2 + 0]),
                                T.cuda.uint_as_float(_tmem_load_5[_lp * 2 + 1 + 0]),
                            )
                    for token_group in T.unroll(2):  # .cu:1031-1048
                        mtx_idx: T.int32 = lane // 8  # .cu:1033
                        row_addr: T.int32 = lane & 7  # .cu:1034
                        dim_base: T.int32 = (
                            epilogue_local_warp * 32 + dim_half * 16 + (mtx_idx & 1) * 8
                        )  # .cu:1035
                        token_base: T.int32 = token_group * 16 + mtx_idx // 2 * 8  # .cu:1036
                        token_addr: T.int32 = token_base + row_addr  # .cu:1037
                        token_pair: T.int32 = token_addr // 2  # .cu:1038
                        token_parity: T.int32 = token_addr & 1  # .cu:1039
                        raw_row: T.int32 = token_pair + dim_base // 64 * 16  # .cu:1040
                        raw_col: T.int32 = (
                            dim_base & 63 ^ (token_pair & 3) << 4 ^ token_parity << 3
                        ) + token_parity * 64  # .cu:1041
                        stsm_offset: T.int32 = (raw_row * 128 + raw_col) * 2  # .cu:1042
                        pack_base: T.int32 = token_group * 4  # .cu:1043
                        # .cu:1044-1047 stmatrix.x4.trans
                        T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                            smem_raw.ptr_to([T.cast(out_stage_addr, "uint32") - smem + T.cast(stsm_offset, "uint32")]),
                            out_packed[pack_base],
                            out_packed[pack_base + 1],
                            out_packed[pack_base + 2],
                            out_packed[pack_base + 3],
                        )  # fmt: skip
                _fence_async_shared()  # .cu:1050
                T.ptx.bar.sync(T.uint32(9), T.uint32(128))  # .cu:1051
                if epilogue_local_warp == 0:  # .cu:1052-1057
                    if T.cuda.elect_sync():
                        _tma_store_4d(
                            smem_raw,
                            T.cast(smem, "int32"),
                            out_tma,
                            0,
                            T.cast(bos_1 + T.cast(chunk_idx_1 * 32, "int64"), "int32"),
                            head_idx_1,
                            0,
                            T.cast(smem_out_addr + T.cast(output_stage, "int32") * 8192, "uint32"),
                        )  # .cu:1054
                    T.ptx.cp.async_.bulk.commit_group()  # .cu:1056
                output_stage = output_stage ^ T.uint32(1)  # .cu:1058
            else:  # .cu:1059-1077 (partial chunk: scalar out stores)
                _tmem_load_6: T.f32[32]
                _tmem_ld_x32(_tmem_load_6, taddr + 192 + tmem_row_base_1)  # .cu:1060-1061
                T.ptx.tcgen05.wait__ld.sync.aligned()  # .cu:1062
                T.ptx.bar.sync(T.uint32(9), T.uint32(128))  # .cu:1063
                if epilogue_local_warp == 0:  # .cu:1064-1068
                    if T.cuda.elect_sync():
                        _mbarrier_arrive(
                            smem_raw, T.cast(smem, "int32"), mbar_base + MBAR_OUT_EMPTY_OFF
                        )
                for token_col_1 in T.unroll(32):  # .cu:1069-1076
                    out_token: T.int64 = bos_1 + T.cast(
                        chunk_idx_1 * 32 + token_col_1, "int64"
                    )  # .cu:1071
                    if out_token < eos_1:  # .cu:1072
                        out_idx: T.int64 = (
                            out_token * T.cast(h, "int64") + T.cast(head_idx_1, "int64")
                        ) * 128 + T.cast(state_row_1, "int64")  # .cu:1073
                        out[out_idx] = T.cast(_tmem_load_6[token_col_1], "bfloat16")  # .cu:1074
            epilogue_stage += 1  # .cu:1078
            if epilogue_stage == 5:  # .cu:1079
                epilogue_stage = T.uint32(0)
                _phase_final_ready_1 = _phase_final_ready_1 ^ T.uint32(1)
        if epilogue_local_warp == 0:  # .cu:1081-1083
            T.ptx.cp.async_.bulk.wait_group(0)
        T.ptx.bar.sync(T.uint32(9), T.uint32(128))  # .cu:1084
        if epilogue_local_warp == 0:  # .cu:1085-1089
            if T.cuda.elect_sync():
                _mbarrier_arrive(
                    smem_raw, T.cast(smem, "int32"), mbar_base + MBAR_TMEM_DEALLOC_READY_OFF
                )
    elif warp == 9:
        # ---- Role: mma (.cu:1092-1247) ----
        # .cu:1093 mma_main
        task_idx_2: T.int32 = bid  # .cu:1094
        seq_idx_2: T.int32 = seq_order[task_idx_2 // h]  # .cu:1095
        bos_2: T.int64 = cu_seqlens[seq_idx_2]  # .cu:1096
        eos_2: T.int64 = cu_seqlens[seq_idx_2 + 1]  # .cu:1097
        seq_len_2: T.int32 = T.cast(eos_2 - bos_2, "int32")  # .cu:1098
        num_chunks_2: T.int32 = (seq_len_2 + 32 - 1) // 32  # .cu:1099
        mma_stage: T.uint32 = 0  # .cu:1100
        _phase_qk_full_1: T.uint32 = 0  # .cu:1101
        _phase_state_inp_ready: T.uint32 = 0  # .cu:1102
        _phase_out_empty_0: T.uint32 = 1  # .cu:1103
        _phase_u_inp_ready: T.uint32 = 0  # .cu:1104
        _phase_u2_inp_ready: T.uint32 = 0  # .cu:1105
        for _chunk_idx in T.serial(0, num_chunks_2, unroll=False):  # .cu:1106-1107
            _mbarrier_wait(
                smem_raw,
                T.cast(smem, "int32"),
                mbar_base + MBAR_QK_FULL_OFF + mma_stage * 8,
                _phase_qk_full_1,
            )  # .cu:1108
            _mbarrier_wait(
                smem_raw,
                T.cast(smem, "int32"),
                mbar_base + MBAR_STATE_INP_READY_OFF + mma_stage * 8,
                _phase_state_inp_ready,
            )  # .cu:1109
            _mbarrier_wait(
                smem_raw, T.cast(smem, "int32"), mbar_base + MBAR_OUT_EMPTY_OFF, _phase_out_empty_0
            )  # .cu:1110
            _phase_out_empty_0 = _phase_out_empty_0 ^ T.uint32(1)  # .cu:1111
            _mma_b_addr_0: T.int32 = smem_qd_addr + T.cast(mma_stage, "int32") * 41984  # .cu:1112
            _mma_b_lo_0 = _make_warp_uniform(
                T.cast((_mma_b_addr_0 >> 4) & 0x3FFF, "uint32")
            )  # .cu:1113
            _mma_qk_8step(
                tmem_tmem_out, T.cast(_mma_b_lo_0, "int32"), tmem_tmem_state_inp, 0
            )  # .cu:1114-1150
            _mma_b_addr_1: T.int32 = smem_kd_addr + T.cast(mma_stage, "int32") * 41984  # .cu:1151
            _mma_b_lo_1 = _make_warp_uniform(
                T.cast((_mma_b_addr_1 >> 4) & 0x3FFF, "uint32")
            )  # .cu:1152
            _mma_qk_8step(
                tmem_tmem_u_acc, T.cast(_mma_b_lo_1, "int32"), tmem_tmem_state_inp, 0
            )  # .cu:1153-1189
            # .cu:1190 elect_commit2(old_out_ready + stage*8, raw_inputs_free + stage*8)
            _leader_1190 = T.cuda.elect_sync()
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                smem_raw.ptr_to(
                    [(mbar_base + MBAR_OLD_OUT_READY_OFF + mma_stage * 8) - T.cast(smem, "int32")]
                ),
                pred=_leader_1190,
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                smem_raw.ptr_to(
                    [(mbar_base + MBAR_RAW_INPUTS_FREE_OFF + mma_stage * 8) - T.cast(smem, "int32")]
                ),
                pred=_leader_1190,
            )
            _mbarrier_wait(
                smem_raw,
                T.cast(smem, "int32"),
                mbar_base + MBAR_U_INP_READY_OFF + mma_stage * 8,
                _phase_u_inp_ready,
            )  # .cu:1191
            _mma_b_addr_2: T.int32 = smem_inv_addr + T.cast(mma_stage, "int32") * 41984  # .cu:1192
            _mma_b_lo_2 = _make_warp_uniform(
                T.cast((_mma_b_addr_2 >> 4) & 0x3FFF, "uint32")
            )  # .cu:1193
            _mma_inv_2step(
                tmem_tmem_u2_acc, T.cast(_mma_b_lo_2, "int32"), tmem_tmem_u2_inp, 0
            )  # .cu:1194-1212
            _elect_commit(
                smem_raw.ptr_to(
                    [(mbar_base + MBAR_U2_ACC_READY_OFF + mma_stage * 8) - T.cast(smem, "int32")]
                )
            )  # .cu:1213
            _mbarrier_wait(
                smem_raw,
                T.cast(smem, "int32"),
                mbar_base + MBAR_U2_INP_READY_OFF + mma_stage * 8,
                _phase_u2_inp_ready,
            )  # .cu:1214
            _mma_b_addr_3: T.int32 = (
                smem_final_trans_addr + T.cast(mma_stage, "int32") * 41984
            )  # .cu:1215
            _mma_b_lo_3 = _make_warp_uniform(
                T.cast(((_mma_b_addr_3 >> 4) & 0x3FFF) | 0x1000000, "uint32")
            )  # .cu:1216
            _mma_final_2step(
                tmem_tmem_state_out, T.cast(_mma_b_lo_3, "int32"), tmem_tmem_u2_inp, 1
            )  # .cu:1217-1235
            # .cu:1236 elect_commit2(final_ready + stage*8, smem_free + stage*8)
            _leader_1236 = T.cuda.elect_sync()
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                smem_raw.ptr_to(
                    [(mbar_base + MBAR_FINAL_READY_OFF + mma_stage * 8) - T.cast(smem, "int32")]
                ),
                pred=_leader_1236,
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                smem_raw.ptr_to(
                    [(mbar_base + MBAR_SMEM_FREE_OFF + mma_stage * 8) - T.cast(smem, "int32")]
                ),
                pred=_leader_1236,
            )
            mma_stage += 1  # .cu:1237
            if mma_stage == 5:  # .cu:1238
                mma_stage = T.uint32(0)
                _phase_qk_full_1 = _phase_qk_full_1 ^ T.uint32(1)
                _phase_state_inp_ready = _phase_state_inp_ready ^ T.uint32(1)
                _phase_u_inp_ready = _phase_u_inp_ready ^ T.uint32(1)
                _phase_u2_inp_ready = _phase_u2_inp_ready ^ T.uint32(1)
        _phase_tmem_dealloc_ready_0: T.uint32 = 0  # .cu:1240
        _mbarrier_wait(
            smem_raw,
            T.cast(smem, "int32"),
            mbar_base + MBAR_TMEM_DEALLOC_READY_OFF,
            _phase_tmem_dealloc_ready_0,
        )  # .cu:1241
        _phase_tmem_dealloc_ready_0 = _phase_tmem_dealloc_ready_0 ^ T.uint32(1)  # .cu:1242
        _tmem_dealloc_addr: T.int32  # .cu:1243
        T.ptx.ld.volatile.shared.s32(_tmem_dealloc_addr, T.address_of(tmem_addr_storage[0]))
        T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
            T.cast(_tmem_dealloc_addr, "uint32"), T.uint32(256)
        )  # .cu:1244
        T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()  # .cu:1245
    elif warp == 10:
        # ---- Role: load (.cu:1248-1296) ----
        # .cu:1249 load_main
        task_idx_3: T.int32 = bid  # .cu:1250
        seq_idx_3: T.int32 = seq_order[task_idx_3 // h]  # .cu:1251
        head_idx_2: T.int32 = task_idx_3 % h  # .cu:1252
        bos_3: T.int64 = cu_seqlens[seq_idx_3]  # .cu:1253
        eos_3: T.int64 = cu_seqlens[seq_idx_3 + 1]  # .cu:1254
        seq_len_3: T.int32 = T.cast(eos_3 - bos_3, "int32")  # .cu:1255
        num_chunks_3: T.int32 = (seq_len_3 + 32 - 1) // 32  # .cu:1256
        load_stage: T.uint32 = 0  # .cu:1257
        _phase_v_free: T.uint32 = 1  # .cu:1258
        _phase_qk_full_2: T.uint32 = 0  # .cu:1259
        for chunk_idx_2 in T.serial(0, num_chunks_3, unroll=False):  # .cu:1260-1261
            _mbarrier_wait(
                smem_raw,
                T.cast(smem, "int32"),
                mbar_base + MBAR_V_FREE_OFF + load_stage * 8,
                _phase_v_free,
            )  # .cu:1262
            _mbarrier_wait(
                smem_raw,
                T.cast(smem, "int32"),
                mbar_base + MBAR_QK_FULL_OFF + load_stage * 8,
                _phase_qk_full_2,
            )  # .cu:1263
            chunk_is_full_1: T.int32 = T.if_then_else(
                seq_len_3 >= (chunk_idx_2 + 1) * 32, 1, 0
            )  # .cu:1264
            if T.cuda.elect_sync():  # .cu:1265-1270
                if chunk_is_full_1 != 0:
                    _mbarrier_arrive_expect_tx(
                        smem_raw.ptr_to(
                            [(mbar_base + MBAR_V_FULL_OFF + load_stage * 8) - T.cast(smem, "int32")]
                        ),
                        8192,
                    )  # .cu:1267
                    _tma_3d_gmem2smem(
                        smem_raw,
                        T.cast(smem, "int32"),
                        smem_v_addr + T.cast(load_stage, "int32") * 41984,
                        v_tma,
                        0,
                        head_idx_2,
                        T.cast(bos_3 + T.cast(chunk_idx_2 * 32, "int64"), "int32"),
                        mbar_base + MBAR_V_FULL_OFF + load_stage * 8,
                    )  # .cu:1268
            if chunk_is_full_1 == 0:  # .cu:1271-1285
                for v_load_iter in T.unroll(16):  # .cu:1272-1282
                    v_item: T.int32 = v_load_iter * 32 + lane  # .cu:1274
                    row: T.int32 = v_item // 16  # .cu:1275
                    segment: T.int32 = v_item % 16  # .cu:1276
                    token: T.int64 = bos_3 + T.cast(chunk_idx_2 * 32 + row, "int64")  # .cu:1277
                    token_valid: T.int32 = T.if_then_else(token < eos_3, 1, 0)  # .cu:1278
                    v_src: T.int64 = (
                        token * T.cast(h, "int64") + T.cast(head_idx_2, "int64")
                    ) * 128 + T.cast(segment * 8, "int64")  # .cu:1279
                    T.ptx["cp.async.cg.shared.global"](
                        smem_raw.ptr_to([SMEM_SMEM_V_OFF + T.cast(load_stage, "int32") * 41984 + (row * 128 + segment * 8) * 2]),
                        v.ptr_to([v_src]),
                        16,
                        T.cast(T.if_then_else(token_valid != 0, 16, 0), "uint32"),
                    )  # .cu:1280-1281  # fmt: skip
                T.ptx.cp.async_.commit_group()  # .cu:1283
                T.ptx.cp.async_.wait_group(0)  # .cu:1284
            T.ptx.bar.sync(T.uint32(8), T.uint32(32))  # .cu:1286 barrier.sync 8, 32
            if T.cuda.elect_sync():  # .cu:1287-1292
                if chunk_is_full_1 == 0:
                    _fence_async_shared()  # .cu:1289
                    _mbarrier_arrive(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_V_FULL_OFF + load_stage * 8,
                    )  # .cu:1290
            load_stage += 1  # .cu:1293
            if load_stage == 5:  # .cu:1294
                load_stage = T.uint32(0)
                _phase_v_free = _phase_v_free ^ T.uint32(1)
                _phase_qk_full_2 = _phase_qk_full_2 ^ T.uint32(1)
    elif warp >= 12 and warp <= 31:
        # ---- Role: prep (.cu:1298-2550) ----
        T.ptx.setmaxnreg.dec.sync.aligned.u32(48)  # .cu:1299
        # .cu:1300 prep_main
        task_idx_4: T.int32 = bid  # .cu:1301
        seq_idx_4: T.int32 = seq_order[task_idx_4 // h]  # .cu:1302
        head_idx_3: T.int32 = task_idx_4 % h  # .cu:1303
        bos_4: T.int64 = cu_seqlens[seq_idx_4]  # .cu:1304
        eos_4: T.int64 = cu_seqlens[seq_idx_4 + 1]  # .cu:1305
        seq_len_4: T.int32 = T.cast(eos_4 - bos_4, "int32")  # .cu:1306
        num_chunks_4: T.int32 = (seq_len_4 + 32 - 1) // 32  # .cu:1307
        instance_id: T.int32 = (warp - 12) // 4  # .cu:1308
        prep_instance: T.int32 = instance_id  # .cu:1309
        warp_id_in_role_2: T.int32 = warp - 12  # .cu:1310
        prep_local_warp: T.int32 = warp_id_in_role_2 - prep_instance * 4  # .cu:1311
        prep_tid: T.int32 = prep_local_warp * 32 + lane  # .cu:1312
        num_prep_iters: T.int32 = (num_chunks_4 + 4 - prep_instance) // 5  # .cu:1313
        prep_stage: T.uint32 = T.cast(prep_instance, "uint32")  # .cu:1314
        gate_rate_stage_f32: T.int32 = prep_instance * 10496  # .cu:1315
        if prep_tid == 0:  # .cu:1316-1319
            smem_gate_rate_all[gate_rate_stage_f32] = _expf(A_log[head_idx_3])
        if prep_instance == 0:  # .cu:1320-1332
            T.ptx.bar.sync(T.uint32(11), T.uint32(128))
        elif prep_instance == 1:
            T.ptx.bar.sync(T.uint32(12), T.uint32(128))
        else:
            if prep_instance == 2:
                T.ptx.bar.sync(T.uint32(13), T.uint32(128))
            elif prep_instance == 3:
                T.ptx.bar.sync(T.uint32(14), T.uint32(128))
            else:
                T.ptx.bar.sync(T.uint32(15), T.uint32(128))
        _phase_raw_inputs_free: T.uint32 = 1  # .cu:1333
        _phase_gate_raw_full: T.uint32 = 0  # .cu:1334
        _phase_smem_free: T.uint32 = 1  # .cu:1335
        _phase_qk_raw_full: T.uint32 = 0  # .cu:1336
        _phase_prep_diag_ready: T.uint32 = 0  # .cu:1337
        _phase_prep_inv16_ready: T.uint32 = 0  # .cu:1338
        for prep_iter in T.serial(0, num_prep_iters, unroll=False):  # .cu:1339-1340
            chunk_idx_3: T.int32 = prep_iter * 5 + prep_instance  # .cu:1341
            stage_f32: T.int32 = T.cast(prep_stage, "int32") * 10496  # .cu:1342
            stage_bf16: T.int32 = T.cast(prep_stage, "int32") * 20992  # .cu:1343
            chunk_is_full_2: T.int32 = T.if_then_else(
                seq_len_4 >= (chunk_idx_3 + 1) * 32, 1, 0
            )  # .cu:1344
            early_beta_value: T.f32 = T.float32(0.0)  # .cu:1345
            early_gate0: T.f32 = T.float32(0.0)  # .cu:1346
            if chunk_is_full_2 != 0:  # .cu:1347-1392
                _mbarrier_wait(
                    smem_raw,
                    T.cast(smem, "int32"),
                    mbar_base + MBAR_RAW_INPUTS_FREE_OFF + prep_stage * 8,
                    _phase_raw_inputs_free,
                )  # .cu:1348
                if prep_local_warp == 0:  # .cu:1349-1357
                    if T.cuda.elect_sync():
                        _mbarrier_arrive_expect_tx(
                            smem_raw.ptr_to([(mbar_base + MBAR_GATE_RAW_FULL_OFF + prep_stage * 8) - T.cast(smem, "int32")]),
                            8704,
                        )  # .cu:1351  # fmt: skip
                        _tma_3d_gmem2smem(
                            smem_raw,
                            T.cast(smem, "int32"),
                            smem_g_raw_addr + T.cast(prep_stage, "int32") * 41984,
                            g_tma,
                            0,
                            head_idx_3,
                            T.cast(bos_4 + T.cast(chunk_idx_3 * 32, "int64"), "int32"),
                            mbar_base + MBAR_GATE_RAW_FULL_OFF + prep_stage * 8,
                        )  # .cu:1352
                        _tma_2d_gmem2smem(
                            smem_raw,
                            T.cast(smem, "int32"),
                            smem_beta_raw_addr + T.cast(prep_stage, "int32") * 41984,
                            beta_tma_tmap,
                            head_idx_3 // 8 * 8,
                            T.cast(bos_4 + T.cast(chunk_idx_3 * 32, "int64"), "int32"),
                            mbar_base + MBAR_GATE_RAW_FULL_OFF + prep_stage * 8,
                        )  # .cu:1353
                        _mbarrier_arrive_expect_tx(
                            smem_raw.ptr_to([(mbar_base + MBAR_QK_RAW_FULL_OFF + prep_stage * 8) - T.cast(smem, "int32")]),
                            16384,
                        )  # .cu:1354  # fmt: skip
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
                        )  # .cu:1355
                _mbarrier_wait(
                    smem_raw,
                    T.cast(smem, "int32"),
                    mbar_base + MBAR_GATE_RAW_FULL_OFF + prep_stage * 8,
                    _phase_gate_raw_full,
                )  # .cu:1358
                if prep_local_warp == 2 and lane < 32:  # .cu:1359-1380
                    beta_raw_pair = T.alloc_local((1,), "uint32", align=4)  # .cu:1360
                    beta_raw_pair[0] = _ld_shared_b32(
                        smem_raw,
                        T.cast(smem, "int32"),
                        smem_beta_raw_addr + T.cast(prep_stage, "int32") * 41984 + lane * 16 + head_idx_3 % 8 // 2 * 4,
                    )  # .cu:1361  # fmt: skip
                    beta_raw_pair_fp32: T.f32[2]  # .cu:1362
                    for _pair in T.unroll(1):  # .cu:1363-1372
                        beta_raw_pair_fp32[_pair * 2] = T.cuda.uint_as_float(
                            beta_raw_pair[_pair + 0] << T.uint32(16)
                        )
                        beta_raw_pair_fp32[_pair * 2 + 1] = T.cuda.uint_as_float(
                            beta_raw_pair[_pair + 0] & T.uint32(0xFFFF0000)
                        )
                    beta_logit: T.f32 = beta_raw_pair_fp32[0]  # .cu:1373
                    if head_idx_3 % 2 != 0:  # .cu:1374-1376
                        beta_logit = beta_raw_pair_fp32[1]
                    early_beta_value = _tanh_approx(beta_logit * T.float32(0.5)) * T.float32(
                        0.5
                    ) + T.float32(0.5)  # .cu:1377-1379
                if prep_tid < 128:  # .cu:1381-1391
                    early_gate_rate: T.f32 = smem_gate_rate_all[stage_f32]  # .cu:1382
                    early_gate_bias: T.f32 = dt_bias[head_idx_3 * 128 + prep_tid]  # .cu:1383
                    early_gate_raw: T.f32 = T.cuda.bfloat162float(
                        smem_g_raw_all[stage_bf16 + prep_tid]
                    )  # .cu:1384-1385
                    early_gate_arg: T.f32 = early_gate_rate * (
                        early_gate_raw + early_gate_bias
                    )  # .cu:1386
                    early_gate_sigmoid: T.f32 = _tanh_approx(
                        early_gate_arg * T.float32(0.5)
                    ) * T.float32(0.5) + T.float32(0.5)  # .cu:1387-1389
                    early_gate0 = (
                        T.float32(lower_bound) * T.float32(1.4426950408889634) * early_gate_sigmoid
                    )  # .cu:1390
            _mbarrier_wait(
                smem_raw,
                T.cast(smem, "int32"),
                mbar_base + MBAR_SMEM_FREE_OFF + prep_stage * 8,
                _phase_smem_free,
            )  # .cu:1393
            if chunk_is_full_2 != 0:  # .cu:1394-1400
                if prep_local_warp == 0:
                    if T.cuda.elect_sync():
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
                        )  # .cu:1397
            if chunk_is_full_2 == 0:  # .cu:1401-1412
                for gate_load_pass in T.unroll(4):
                    gate_load_item: T.int32 = gate_load_pass * 128 + prep_tid  # .cu:1404
                    gate_load_row: T.int32 = gate_load_item // 16  # .cu:1405
                    gate_load_segment: T.int32 = gate_load_item % 16  # .cu:1406
                    gate_load_token: T.int64 = bos_4 + T.cast(
                        chunk_idx_3 * 32 + gate_load_row, "int64"
                    )  # .cu:1407
                    gate_load_base: T.int64 = (
                        gate_load_token * T.cast(h, "int64") + T.cast(head_idx_3, "int64")
                    ) * 128 + T.cast(gate_load_segment * 8, "int64")  # .cu:1408
                    T.ptx["cp.async.cg.shared.global"](
                        smem_raw.ptr_to([SMEM_SMEM_G_RAW_OFF + T.cast(prep_stage, "int32") * 41984 + gate_load_item * 16]),
                        g.ptr_to([gate_load_base]),
                        16,
                        T.cast(T.if_then_else(T.if_then_else(gate_load_token < eos_4, 1, 0) != 0, 16, 0), "uint32"),
                    )  # .cu:1409-1410  # fmt: skip
            if chunk_is_full_2 == 0:  # .cu:1413-1429
                T.ptx.cp.async_.commit_group()  # .cu:1414
                T.ptx.cp.async_.wait_group(0)  # .cu:1415
                if prep_instance == 0:
                    T.ptx.bar.sync(T.uint32(11), T.uint32(128))
                elif prep_instance == 1:
                    T.ptx.bar.sync(T.uint32(12), T.uint32(128))
                else:
                    if prep_instance == 2:
                        T.ptx.bar.sync(T.uint32(13), T.uint32(128))
                    elif prep_instance == 3:
                        T.ptx.bar.sync(T.uint32(14), T.uint32(128))
                    else:
                        T.ptx.bar.sync(T.uint32(15), T.uint32(128))
            if prep_local_warp == 2 and lane < 32:  # .cu:1430-1442
                beta_value: T.f32 = early_beta_value  # .cu:1431
                if chunk_is_full_2 == 0:  # .cu:1432-1440
                    beta_token: T.int64 = bos_4 + T.cast(chunk_idx_3 * 32 + lane, "int64")
                    if beta_token < eos_4:
                        beta_logit_1: T.f32 = T.cast(
                            beta[beta_token * T.cast(h, "int64") + T.cast(head_idx_3, "int64")],
                            "float32",
                        )  # .cu:1435
                        beta_value = _tanh_approx(beta_logit_1 * T.float32(0.5)) * T.float32(
                            0.5
                        ) + T.float32(0.5)  # .cu:1436-1438
                smem_prep_beta_all[stage_f32 + lane] = beta_value  # .cu:1441
            if prep_tid < 128:  # .cu:1443-1472
                gate_col: T.int32 = prep_tid  # .cu:1444
                gate_rate: T.f32 = smem_gate_rate_all[stage_f32]  # .cu:1445
                gate_bias: T.f32 = dt_bias[head_idx_3 * 128 + gate_col]  # .cu:1446
                prefix_log2: T.f32 = T.float32(0.0)  # .cu:1447
                for gate_row in T.serial(0, 32):  # .cu:1448-1471
                    gate_token: T.int64 = bos_4 + T.cast(
                        chunk_idx_3 * 32 + gate_row, "int64"
                    )  # .cu:1449
                    gate_log2: T.f32 = T.float32(0.0)  # .cu:1450
                    gate_needs_compute: T.int32 = 1  # .cu:1451
                    if gate_row == 0:  # .cu:1452-1457
                        if chunk_is_full_2 != 0:
                            gate_log2 = early_gate0
                            gate_needs_compute = 0
                    if gate_needs_compute != 0:  # .cu:1458-1468
                        if gate_token < eos_4:
                            gate_raw: T.f32 = T.cuda.bfloat162float(
                                smem_g_raw_all[stage_bf16 + gate_row * 128 + gate_col]
                            )  # .cu:1460-1461
                            gate_arg: T.f32 = gate_rate * (gate_raw + gate_bias)  # .cu:1462
                            gate_sigmoid: T.f32 = _tanh_approx(
                                gate_arg * T.float32(0.5)
                            ) * T.float32(0.5) + T.float32(0.5)  # .cu:1463-1465
                            gate_log2 = (
                                T.float32(lower_bound)
                                * T.float32(1.4426950408889634)
                                * gate_sigmoid
                            )  # .cu:1466
                    prefix_log2 += gate_log2  # .cu:1469
                    smem_gate_all[stage_f32 + gate_row * 128 + gate_col] = prefix_log2  # .cu:1470
            if prep_instance == 0:  # .cu:1473-1485
                T.ptx.bar.sync(T.uint32(11), T.uint32(128))
            elif prep_instance == 1:
                T.ptx.bar.sync(T.uint32(12), T.uint32(128))
            else:
                if prep_instance == 2:
                    T.ptx.bar.sync(T.uint32(13), T.uint32(128))
                elif prep_instance == 3:
                    T.ptx.bar.sync(T.uint32(14), T.uint32(128))
                else:
                    T.ptx.bar.sync(T.uint32(15), T.uint32(128))
            if chunk_is_full_2 != 0:  # .cu:1486-1488
                _mbarrier_wait(
                    smem_raw,
                    T.cast(smem, "int32"),
                    mbar_base + MBAR_QK_RAW_FULL_OFF + prep_stage * 8,
                    _phase_qk_raw_full,
                )
            if prep_tid < 128:  # .cu:1489-1493
                total_log2: T.f32 = smem_gt_prefix_all[stage_f32 + prep_tid]  # .cu:1490
                smem_restore_factor_all[stage_f32 + prep_tid] = _approx_exp2(
                    total_log2
                    - T.float32(lower_bound) * T.float32(1.4426950408889634) * T.float32(16.0)
                )  # .cu:1491-1492
            if prep_tid == 0:  # .cu:1494-1497
                smem_restore_factor_all[stage_f32 + 128] = _approx_exp2(
                    T.float32(lower_bound) * T.float32(1.4426950408889634) * T.float32(16.0)
                )
            q_u32 = T.decl_buffer((total_tokens * h * D_HEAD // 2,), "uint32", data=q.data)
            k_u32 = T.decl_buffer((total_tokens * h * D_HEAD // 2,), "uint32", data=k.data)
            for work_pass in T.serial(0, 4, unroll=False):  # .cu:1498-1694 (#pragma unroll 1)
                work_item: T.int32 = work_pass * 128 + prep_tid  # .cu:1500
                row_1: T.int32 = work_item // 16  # .cu:1501
                segment_1: T.int32 = work_item % 16  # .cu:1502
                token_1: T.int64 = bos_4 + T.cast(chunk_idx_3 * 32 + row_1, "int64")  # .cu:1503
                token_valid_1: T.int32 = T.if_then_else(token_1 < eos_4, 1, 0)  # .cu:1504
                gmem_base: T.int64 = (
                    token_1 * T.cast(h, "int64") + T.cast(head_idx_3, "int64")
                ) * 128 + T.cast(segment_1 * 8, "int64")  # .cu:1505
                # q/k smem swizzle byte offset (spelled inline at each .cu use site:
                # 1528, 1547, 1672, 1682, 1692)
                q_raw_vec: T.f32[8]  # .cu:1506
                k_raw_vec: T.f32[8]  # .cu:1507
                for _zi in T.unroll(8):  # .cu:1508-1523
                    q_raw_vec[_zi] = T.float32(0.0)
                    k_raw_vec[_zi] = T.float32(0.0)
                if chunk_is_full_2 != 0:  # .cu:1524-1562
                    packed = T.alloc_local((4,), "uint32", align=16)  # .cu:1525
                    _ld_shared_v4(
                        smem_raw,
                        T.cast(smem, "int32"),
                        packed,
                        smem_q_raw_prefetch_addr + T.cast(prep_stage, "int32") * 41984 + (segment_1 * 8 // 64 * 4096 + row_1 * 128 + segment_1 * 8 % 64 * 2 ^ ((segment_1 * 8 // 64 * 4096 + row_1 * 128 + segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)),
                    )  # .cu:1526-1528  # fmt: skip
                    packed_fp32: T.f32[8]  # .cu:1529
                    for _pair in T.unroll(4):  # .cu:1530-1539
                        packed_fp32[_pair * 2] = T.cuda.uint_as_float(
                            packed[_pair + 0] << T.uint32(16)
                        )
                        packed_fp32[_pair * 2 + 1] = T.cuda.uint_as_float(
                            packed[_pair + 0] & T.uint32(0xFFFF0000)
                        )
                    for value_idx in T.unroll(8):  # .cu:1540-1543
                        q_raw_vec[value_idx] = packed_fp32[value_idx]
                    packed_0 = T.alloc_local((4,), "uint32", align=16)  # .cu:1544
                    _ld_shared_v4(
                        smem_raw,
                        T.cast(smem, "int32"),
                        packed_0,
                        smem_kd_addr + T.cast(prep_stage, "int32") * 41984 + (segment_1 * 8 // 64 * 4096 + row_1 * 128 + segment_1 * 8 % 64 * 2 ^ ((segment_1 * 8 // 64 * 4096 + row_1 * 128 + segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)),
                    )  # .cu:1545-1547  # fmt: skip
                    packed_0_fp32: T.f32[8]  # .cu:1548
                    for _pair in T.unroll(4):  # .cu:1549-1558
                        packed_0_fp32[_pair * 2] = T.cuda.uint_as_float(
                            packed_0[_pair + 0] << T.uint32(16)
                        )
                        packed_0_fp32[_pair * 2 + 1] = T.cuda.uint_as_float(
                            packed_0[_pair + 0] & T.uint32(0xFFFF0000)
                        )
                    for value_idx_1 in T.unroll(8):  # .cu:1559-1562
                        k_raw_vec[value_idx_1] = packed_0_fp32[value_idx_1]
                elif token_valid_1 != 0:  # .cu:1563-1602
                    for _blk in T.unroll(1):  # .cu:1564-1582
                        _vldq = T.alloc_local((4,), "uint32", align=16)
                        _ld_global_v4_u32(_vldq, q_u32.ptr_to([gmem_base // 2 + _blk * 4]))
                        for _pair in T.unroll(4):
                            q_raw_vec[0 + _blk * 8 + _pair * 2] = T.cuda.uint_as_float(
                                _vldq[_pair] << T.uint32(16)
                            )
                            q_raw_vec[0 + _blk * 8 + _pair * 2 + 1] = T.cuda.uint_as_float(
                                _vldq[_pair] & T.uint32(0xFFFF0000)
                            )
                    for _blk in T.unroll(1):  # .cu:1583-1601
                        _vldk = T.alloc_local((4,), "uint32", align=16)
                        _ld_global_v4_u32(_vldk, k_u32.ptr_to([gmem_base // 2 + _blk * 4]))
                        for _pair in T.unroll(4):
                            k_raw_vec[0 + _blk * 8 + _pair * 2] = T.cuda.uint_as_float(
                                _vldk[_pair] << T.uint32(16)
                            )
                            k_raw_vec[0 + _blk * 8 + _pair * 2 + 1] = T.cuda.uint_as_float(
                                _vldk[_pair] & T.uint32(0xFFFF0000)
                            )
                q_sum: T.f32 = T.float32(0.0)  # .cu:1603
                k_sum: T.f32 = T.float32(0.0)  # .cu:1604
                for elem_in_segment in T.serial(0, 8):  # .cu:1605-1612
                    q_sum = _fmaf_rn(
                        q_raw_vec[elem_in_segment], q_raw_vec[elem_in_segment], q_sum
                    )  # .cu:1608
                    k_sum = _fmaf_rn(
                        k_raw_vec[elem_in_segment], k_raw_vec[elem_in_segment], k_sum
                    )  # .cu:1610
                q_sum += T.cuda._shfl_xor_sync(T.uint32(0xFFFFFFFF), q_sum, 8, 32)  # .cu:1613-1614
                k_sum += T.cuda._shfl_xor_sync(T.uint32(0xFFFFFFFF), k_sum, 8, 32)  # .cu:1615-1616
                q_sum += T.cuda._shfl_xor_sync(T.uint32(0xFFFFFFFF), q_sum, 4, 32)  # .cu:1617-1618
                k_sum += T.cuda._shfl_xor_sync(T.uint32(0xFFFFFFFF), k_sum, 4, 32)  # .cu:1619-1620
                q_sum += T.cuda._shfl_xor_sync(T.uint32(0xFFFFFFFF), q_sum, 2, 32)  # .cu:1621-1622
                k_sum += T.cuda._shfl_xor_sync(T.uint32(0xFFFFFFFF), k_sum, 2, 32)  # .cu:1623-1624
                q_sum += T.cuda._shfl_xor_sync(T.uint32(0xFFFFFFFF), q_sum, 1, 32)  # .cu:1625-1626
                k_sum += T.cuda._shfl_xor_sync(T.uint32(0xFFFFFFFF), k_sum, 1, 32)  # .cu:1627-1628
                q_inv: T.f32 = _rsqrtf(q_sum + T.float32(1e-06))  # .cu:1629-1630
                k_inv: T.f32 = _rsqrtf(k_sum + T.float32(1e-06))  # .cu:1631-1632
                for _ls in T.unroll(4):  # .cu:1633-1636
                    _pk = _mul_f32x2_inplace(
                        T.cuda.make_float2(q_raw_vec[_ls * 2], q_raw_vec[_ls * 2 + 1]),
                        T.cuda.make_float2(q_inv, q_inv),
                    )
                    q_raw_vec[_ls * 2] = T.cuda.float2_x(_pk)
                    q_raw_vec[_ls * 2 + 1] = T.cuda.float2_y(_pk)
                for _ls in T.unroll(4):  # .cu:1637-1640
                    _pk = _mul_f32x2_inplace(
                        T.cuda.make_float2(k_raw_vec[_ls * 2], k_raw_vec[_ls * 2 + 1]),
                        T.cuda.make_float2(k_inv, k_inv),
                    )
                    k_raw_vec[_ls * 2] = T.cuda.float2_x(_pk)
                    k_raw_vec[_ls * 2 + 1] = T.cuda.float2_y(_pk)
                qd_vec: T.f32[8]  # .cu:1641
                kd_vec: T.f32[8]  # .cu:1642
                ki_vec: T.f32[8]  # .cu:1643
                for elem_in_segment_1 in T.serial(0, 8):  # .cu:1644-1653
                    col: T.int32 = segment_1 * 8 + elem_in_segment_1  # .cu:1645
                    prefix: T.f32 = smem_gate_all[stage_f32 + row_1 * 128 + col]  # .cu:1646
                    common_log2: T.f32 = (
                        T.float32(lower_bound) * T.float32(1.4426950408889634) * T.float32(16.0)
                    )  # .cu:1647
                    decay: T.f32 = _approx_exp2(prefix - common_log2)  # .cu:1648-1649
                    qd_vec[elem_in_segment_1] = decay  # .cu:1650
                    kd_vec[elem_in_segment_1] = decay  # .cu:1651
                    ki_vec[elem_in_segment_1] = T.cuda.fdividef(
                        k_raw_vec[elem_in_segment_1], decay
                    )  # .cu:1652 (-use_fast_math: / -> div.approx.f32)
                for _ls in T.unroll(4):  # .cu:1654-1656
                    _pk = _mul_f32x2_inplace(
                        T.cuda.make_float2(qd_vec[_ls * 2], qd_vec[_ls * 2 + 1]),
                        T.cuda.make_float2(q_raw_vec[_ls * 2], q_raw_vec[_ls * 2 + 1]),
                    )
                    qd_vec[_ls * 2] = T.cuda.float2_x(_pk)
                    qd_vec[_ls * 2 + 1] = T.cuda.float2_y(_pk)
                for _ls in T.unroll(4):  # .cu:1657-1660
                    _pk = _mul_f32x2_inplace(
                        T.cuda.make_float2(qd_vec[_ls * 2], qd_vec[_ls * 2 + 1]),
                        T.cuda.make_float2(T.float32(scale), T.float32(scale)),
                    )
                    qd_vec[_ls * 2] = T.cuda.float2_x(_pk)
                    qd_vec[_ls * 2 + 1] = T.cuda.float2_y(_pk)
                for _ls in T.unroll(4):  # .cu:1661-1663
                    _pk = _mul_f32x2_inplace(
                        T.cuda.make_float2(kd_vec[_ls * 2], kd_vec[_ls * 2 + 1]),
                        T.cuda.make_float2(k_raw_vec[_ls * 2], k_raw_vec[_ls * 2 + 1]),
                    )
                    kd_vec[_ls * 2] = T.cuda.float2_x(_pk)
                    kd_vec[_ls * 2 + 1] = T.cuda.float2_y(_pk)
                packed_1 = T.alloc_local((4,), "uint32", align=4)  # .cu:1664
                for _lp in T.unroll(4):  # .cu:1665-1669
                    packed_1[_lp] = T.cuda.float22bfloat162_rn(
                        qd_vec[_lp * 2 + 0], qd_vec[_lp * 2 + 1 + 0]
                    )
                for word in T.unroll(4):  # .cu:1670-1673
                    _st_shared_b32(
                        smem_raw,
                        T.cast(smem, "int32"),
                        smem_qd_addr + T.cast(prep_stage, "int32") * 41984 + (segment_1 * 8 // 64 * 4096 + row_1 * 128 + segment_1 * 8 % 64 * 2 ^ ((segment_1 * 8 // 64 * 4096 + row_1 * 128 + segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + word * 4,
                        packed_1[word],
                    )  # fmt: skip
                packed_0_1 = T.alloc_local((4,), "uint32", align=4)  # .cu:1674
                for _lp in T.unroll(4):  # .cu:1675-1679
                    packed_0_1[_lp] = T.cuda.float22bfloat162_rn(
                        kd_vec[_lp * 2 + 0], kd_vec[_lp * 2 + 1 + 0]
                    )
                for word_1 in T.unroll(4):  # .cu:1680-1683
                    _st_shared_b32(
                        smem_raw,
                        T.cast(smem, "int32"),
                        smem_kd_addr + T.cast(prep_stage, "int32") * 41984 + (segment_1 * 8 // 64 * 4096 + row_1 * 128 + segment_1 * 8 % 64 * 2 ^ ((segment_1 * 8 // 64 * 4096 + row_1 * 128 + segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + word_1 * 4,
                        packed_0_1[word_1],
                    )  # fmt: skip
                packed_1_1 = T.alloc_local((4,), "uint32", align=4)  # .cu:1684
                for _lp in T.unroll(4):  # .cu:1685-1689
                    packed_1_1[_lp] = T.cuda.float22bfloat162_rn(
                        ki_vec[_lp * 2 + 0], ki_vec[_lp * 2 + 1 + 0]
                    )
                for word_2 in T.unroll(4):  # .cu:1690-1693
                    _st_shared_b32(
                        smem_raw,
                        T.cast(smem, "int32"),
                        smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + (segment_1 * 8 // 64 * 4096 + row_1 * 128 + segment_1 * 8 % 64 * 2 ^ ((segment_1 * 8 // 64 * 4096 + row_1 * 128 + segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + word_2 * 4,
                        packed_1_1[word_2],
                    )  # fmt: skip
            if prep_instance == 0:  # .cu:1695-1707
                T.ptx.bar.sync(T.uint32(11), T.uint32(128))
            elif prep_instance == 1:
                T.ptx.bar.sync(T.uint32(12), T.uint32(128))
            else:
                if prep_instance == 2:
                    T.ptx.bar.sync(T.uint32(13), T.uint32(128))
                elif prep_instance == 3:
                    T.ptx.bar.sync(T.uint32(14), T.uint32(128))
                else:
                    T.ptx.bar.sync(T.uint32(15), T.uint32(128))
            pair_row_base: T.int32 = prep_local_warp // 2 * 16  # .cu:1708
            pair_col_base: T.int32 = prep_local_warp % 2 * 16  # .cu:1709
            a_frag = T.alloc_local((4,), "uint32", align=4)  # .cu:1710
            b_frag = T.alloc_local((4,), "uint32", align=4)  # .cu:1711
            acc = T.alloc_local((8,), "float32", align=4)  # .cu:1712
            if pair_row_base >= pair_col_base:  # .cu:1713-1990
                # .cu:1714-1727 chain step 0 (kd a, ki b, 2x zero-C mma)
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    a_frag[0],
                    a_frag[1],
                    a_frag[2],
                    a_frag[3],
                    smem_raw.ptr_to([(smem_kd_addr + T.cast(prep_stage, "int32") * 41984 + (lane // 16 // 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane // 16 % 8 * 16 ^ ((pair_row_base + lane % 16 & 7) << 4)) // 16) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    b_frag[0],
                    b_frag[1],
                    b_frag[2],
                    b_frag[3],
                    smem_raw.ptr_to([(smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + (lane % 16 // 8 // 8 * 256 + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8 + (lane % 16 // 8 % 8 * 16 ^ ((pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4)) // 16) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                _mma_m16n8k16_bf16_zero(acc, a_frag, b_frag)
                _mma_m16n8k16_bf16_zero_off4(acc, a_frag, b_frag)
                # .cu:1728-1741 chain step 1
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    a_frag[0],
                    a_frag[1],
                    a_frag[2],
                    a_frag[3],
                    smem_raw.ptr_to([(smem_kd_addr + T.cast(prep_stage, "int32") * 41984 + ((lane // 16 // 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane // 16 % 8 * 16 ^ ((pair_row_base + lane % 16 & 7) << 4)) // 16) ^ 2) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    b_frag[0],
                    b_frag[1],
                    b_frag[2],
                    b_frag[3],
                    smem_raw.ptr_to([(smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + (((lane % 16 // 8 // 8 * 256 + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8 + (lane % 16 // 8 % 8 * 16 ^ ((pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4)) // 16) + 256 ^ 2) - 256) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                _mma_m16n8k16_bf16_acc(acc, a_frag, b_frag)
                _mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag)
                # .cu:1742-1755 chain step 2
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    a_frag[0],
                    a_frag[1],
                    a_frag[2],
                    a_frag[3],
                    smem_raw.ptr_to([(smem_kd_addr + T.cast(prep_stage, "int32") * 41984 + ((lane // 16 // 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane // 16 % 8 * 16 ^ ((pair_row_base + lane % 16 & 7) << 4)) // 16) ^ 2 ^ 6) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    b_frag[0],
                    b_frag[1],
                    b_frag[2],
                    b_frag[3],
                    smem_raw.ptr_to([(smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + ((((lane % 16 // 8 // 8 * 256 + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8 + (lane % 16 // 8 % 8 * 16 ^ ((pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4)) // 16) + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                _mma_m16n8k16_bf16_acc(acc, a_frag, b_frag)
                _mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag)
                # .cu:1756-1769 chain step 3
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    a_frag[0],
                    a_frag[1],
                    a_frag[2],
                    a_frag[3],
                    smem_raw.ptr_to([(smem_kd_addr + T.cast(prep_stage, "int32") * 41984 + ((lane // 16 // 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane // 16 % 8 * 16 ^ ((pair_row_base + lane % 16 & 7) << 4)) // 16) ^ 2 ^ 6 ^ 2) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    b_frag[0],
                    b_frag[1],
                    b_frag[2],
                    b_frag[3],
                    smem_raw.ptr_to([(smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + (((((lane % 16 // 8 // 8 * 256 + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8 + (lane % 16 // 8 % 8 * 16 ^ ((pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4)) // 16) + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                _mma_m16n8k16_bf16_acc(acc, a_frag, b_frag)
                _mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag)
                # .cu:1770-1783 chain step 4
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    a_frag[0],
                    a_frag[1],
                    a_frag[2],
                    a_frag[3],
                    smem_raw.ptr_to([(smem_kd_addr + T.cast(prep_stage, "int32") * 41984 + (((lane // 16 // 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane // 16 % 8 * 16 ^ ((pair_row_base + lane % 16 & 7) << 4)) // 16) ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    b_frag[0],
                    b_frag[1],
                    b_frag[2],
                    b_frag[3],
                    smem_raw.ptr_to([(smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + ((((((lane % 16 // 8 // 8 * 256 + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8 + (lane % 16 // 8 % 8 * 16 ^ ((pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4)) // 16) + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                _mma_m16n8k16_bf16_acc(acc, a_frag, b_frag)
                _mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag)
                # .cu:1784-1797 chain step 5
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    a_frag[0],
                    a_frag[1],
                    a_frag[2],
                    a_frag[3],
                    smem_raw.ptr_to([(smem_kd_addr + T.cast(prep_stage, "int32") * 41984 + ((((lane // 16 // 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane // 16 % 8 * 16 ^ ((pair_row_base + lane % 16 & 7) << 4)) // 16) ^ 2 ^ 6 ^ 2 ^ 6) + 256) ^ 2) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    b_frag[0],
                    b_frag[1],
                    b_frag[2],
                    b_frag[3],
                    smem_raw.ptr_to([(smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + (((((((lane % 16 // 8 // 8 * 256 + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8 + (lane % 16 // 8 % 8 * 16 ^ ((pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4)) // 16) + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                _mma_m16n8k16_bf16_acc(acc, a_frag, b_frag)
                _mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag)
                # .cu:1798-1811 chain step 6
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    a_frag[0],
                    a_frag[1],
                    a_frag[2],
                    a_frag[3],
                    smem_raw.ptr_to([(smem_kd_addr + T.cast(prep_stage, "int32") * 41984 + ((((lane // 16 // 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane // 16 % 8 * 16 ^ ((pair_row_base + lane % 16 & 7) << 4)) // 16) ^ 2 ^ 6 ^ 2 ^ 6) + 256) ^ 2 ^ 6) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    b_frag[0],
                    b_frag[1],
                    b_frag[2],
                    b_frag[3],
                    smem_raw.ptr_to([(smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + ((((((((lane % 16 // 8 // 8 * 256 + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8 + (lane % 16 // 8 % 8 * 16 ^ ((pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4)) // 16) + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                _mma_m16n8k16_bf16_acc(acc, a_frag, b_frag)
                _mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag)
                # .cu:1812-1825 chain step 7
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    a_frag[0],
                    a_frag[1],
                    a_frag[2],
                    a_frag[3],
                    smem_raw.ptr_to([(smem_kd_addr + T.cast(prep_stage, "int32") * 41984 + ((((lane // 16 // 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane // 16 % 8 * 16 ^ ((pair_row_base + lane % 16 & 7) << 4)) // 16) ^ 2 ^ 6 ^ 2 ^ 6) + 256) ^ 2 ^ 6 ^ 2) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    b_frag[0],
                    b_frag[1],
                    b_frag[2],
                    b_frag[3],
                    smem_raw.ptr_to([(smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + (((((((((lane % 16 // 8 // 8 * 256 + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8 + (lane % 16 // 8 % 8 * 16 ^ ((pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4)) // 16) + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                _mma_m16n8k16_bf16_acc(acc, a_frag, b_frag)
                _mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag)
                row0: T.int32 = pair_row_base + lane // 4  # .cu:1826
                row1: T.int32 = row0 + 8  # .cu:1827
                col0: T.int32 = pair_col_base + lane % 4 * 2  # .cu:1828
                beta0: T.f32 = smem_prep_beta_all[stage_f32 + row0]  # .cu:1829
                beta1: T.f32 = smem_prep_beta_all[stage_f32 + row1]  # .cu:1830
                seed: T.f32[8]  # .cu:1831
                for _zi in T.unroll(8):  # .cu:1832-1839
                    seed[_zi] = T.float32(0.0)
                if row0 > col0:  # .cu:1840-1842
                    seed[0] = acc[0] * beta0
                if row0 > col0 + 1:  # .cu:1843-1845
                    seed[1] = acc[1] * beta0
                if row1 > col0:  # .cu:1846-1848
                    seed[2] = acc[2] * beta1
                if row1 > col0 + 1:  # .cu:1849-1851
                    seed[3] = acc[3] * beta1
                if row0 > col0 + 8:  # .cu:1852-1854
                    seed[4] = acc[4] * beta0
                if row0 > col0 + 9:  # .cu:1855-1857
                    seed[5] = acc[5] * beta0
                if row1 > col0 + 8:  # .cu:1858-1860
                    seed[6] = acc[6] * beta1
                if row1 > col0 + 9:  # .cu:1861-1863
                    seed[7] = acc[7] * beta1
                seed_packed = T.alloc_local((4,), "uint32", align=4)  # .cu:1864
                for _lp in T.unroll(4):  # .cu:1865-1869
                    seed_packed[_lp] = T.cuda.float22bfloat162_rn(
                        seed[_lp * 2 + 0], seed[_lp * 2 + 1 + 0]
                    )
                seed_lane_row: T.int32 = lane % 16  # .cu:1870
                seed_lane_col: T.int32 = lane // 16 * 8  # .cu:1871
                byte_off: T.int32 = (pair_row_base + seed_lane_row) * 128 + (
                    pair_col_base + seed_lane_col
                ) * 2  # .cu:1872
                swizzled_off: T.int32 = byte_off ^ ((byte_off >> 7 & 7) << 4)  # .cu:1873
                seed_addr: T.int32 = (
                    smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off
                )  # .cu:1874
                # .cu:1875-1878 stmatrix.x4 (seed into inv_work)
                T.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                    smem_raw.ptr_to([(seed_addr) - T.cast(smem, "int32")]),
                    seed_packed[0],
                    seed_packed[1],
                    seed_packed[2],
                    seed_packed[3],
                )
                # .cu:1879-1990 second chain (qd a-side, ki b-side), same address pattern
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    a_frag[0],
                    a_frag[1],
                    a_frag[2],
                    a_frag[3],
                    smem_raw.ptr_to([(smem_qd_addr + T.cast(prep_stage, "int32") * 41984 + (lane // 16 // 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane // 16 % 8 * 16 ^ ((pair_row_base + lane % 16 & 7) << 4)) // 16) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    b_frag[0],
                    b_frag[1],
                    b_frag[2],
                    b_frag[3],
                    smem_raw.ptr_to([(smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + (lane % 16 // 8 // 8 * 256 + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8 + (lane % 16 // 8 % 8 * 16 ^ ((pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4)) // 16) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                _mma_m16n8k16_bf16_zero(acc, a_frag, b_frag)
                _mma_m16n8k16_bf16_zero_off4(acc, a_frag, b_frag)
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    a_frag[0],
                    a_frag[1],
                    a_frag[2],
                    a_frag[3],
                    smem_raw.ptr_to([(smem_qd_addr + T.cast(prep_stage, "int32") * 41984 + ((lane // 16 // 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane // 16 % 8 * 16 ^ ((pair_row_base + lane % 16 & 7) << 4)) // 16) ^ 2) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    b_frag[0],
                    b_frag[1],
                    b_frag[2],
                    b_frag[3],
                    smem_raw.ptr_to([(smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + (((lane % 16 // 8 // 8 * 256 + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8 + (lane % 16 // 8 % 8 * 16 ^ ((pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4)) // 16) + 256 ^ 2) - 256) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                _mma_m16n8k16_bf16_acc(acc, a_frag, b_frag)
                _mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag)
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    a_frag[0],
                    a_frag[1],
                    a_frag[2],
                    a_frag[3],
                    smem_raw.ptr_to([(smem_qd_addr + T.cast(prep_stage, "int32") * 41984 + ((lane // 16 // 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane // 16 % 8 * 16 ^ ((pair_row_base + lane % 16 & 7) << 4)) // 16) ^ 2 ^ 6) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    b_frag[0],
                    b_frag[1],
                    b_frag[2],
                    b_frag[3],
                    smem_raw.ptr_to([(smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + ((((lane % 16 // 8 // 8 * 256 + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8 + (lane % 16 // 8 % 8 * 16 ^ ((pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4)) // 16) + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                _mma_m16n8k16_bf16_acc(acc, a_frag, b_frag)
                _mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag)
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    a_frag[0],
                    a_frag[1],
                    a_frag[2],
                    a_frag[3],
                    smem_raw.ptr_to([(smem_qd_addr + T.cast(prep_stage, "int32") * 41984 + ((lane // 16 // 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane // 16 % 8 * 16 ^ ((pair_row_base + lane % 16 & 7) << 4)) // 16) ^ 2 ^ 6 ^ 2) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    b_frag[0],
                    b_frag[1],
                    b_frag[2],
                    b_frag[3],
                    smem_raw.ptr_to([(smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + (((((lane % 16 // 8 // 8 * 256 + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8 + (lane % 16 // 8 % 8 * 16 ^ ((pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4)) // 16) + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                _mma_m16n8k16_bf16_acc(acc, a_frag, b_frag)
                _mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag)
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    a_frag[0],
                    a_frag[1],
                    a_frag[2],
                    a_frag[3],
                    smem_raw.ptr_to([(smem_qd_addr + T.cast(prep_stage, "int32") * 41984 + (((lane // 16 // 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane // 16 % 8 * 16 ^ ((pair_row_base + lane % 16 & 7) << 4)) // 16) ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    b_frag[0],
                    b_frag[1],
                    b_frag[2],
                    b_frag[3],
                    smem_raw.ptr_to([(smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + ((((((lane % 16 // 8 // 8 * 256 + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8 + (lane % 16 // 8 % 8 * 16 ^ ((pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4)) // 16) + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                _mma_m16n8k16_bf16_acc(acc, a_frag, b_frag)
                _mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag)
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    a_frag[0],
                    a_frag[1],
                    a_frag[2],
                    a_frag[3],
                    smem_raw.ptr_to([(smem_qd_addr + T.cast(prep_stage, "int32") * 41984 + ((((lane // 16 // 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane // 16 % 8 * 16 ^ ((pair_row_base + lane % 16 & 7) << 4)) // 16) ^ 2 ^ 6 ^ 2 ^ 6) + 256) ^ 2) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    b_frag[0],
                    b_frag[1],
                    b_frag[2],
                    b_frag[3],
                    smem_raw.ptr_to([(smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + (((((((lane % 16 // 8 // 8 * 256 + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8 + (lane % 16 // 8 % 8 * 16 ^ ((pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4)) // 16) + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                _mma_m16n8k16_bf16_acc(acc, a_frag, b_frag)
                _mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag)
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    a_frag[0],
                    a_frag[1],
                    a_frag[2],
                    a_frag[3],
                    smem_raw.ptr_to([(smem_qd_addr + T.cast(prep_stage, "int32") * 41984 + ((((lane // 16 // 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane // 16 % 8 * 16 ^ ((pair_row_base + lane % 16 & 7) << 4)) // 16) ^ 2 ^ 6 ^ 2 ^ 6) + 256) ^ 2 ^ 6) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    b_frag[0],
                    b_frag[1],
                    b_frag[2],
                    b_frag[3],
                    smem_raw.ptr_to([(smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + ((((((((lane % 16 // 8 // 8 * 256 + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8 + (lane % 16 // 8 % 8 * 16 ^ ((pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4)) // 16) + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                _mma_m16n8k16_bf16_acc(acc, a_frag, b_frag)
                _mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag)
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    a_frag[0],
                    a_frag[1],
                    a_frag[2],
                    a_frag[3],
                    smem_raw.ptr_to([(smem_qd_addr + T.cast(prep_stage, "int32") * 41984 + ((((lane // 16 // 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane // 16 % 8 * 16 ^ ((pair_row_base + lane % 16 & 7) << 4)) // 16) ^ 2 ^ 6 ^ 2 ^ 6) + 256) ^ 2 ^ 6 ^ 2) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    b_frag[0],
                    b_frag[1],
                    b_frag[2],
                    b_frag[3],
                    smem_raw.ptr_to([(smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + (((((((((lane % 16 // 8 // 8 * 256 + (pair_col_base + 8 * (lane // 16) + lane % 8) * 8 + (lane % 16 // 8 % 8 * 16 ^ ((pair_col_base + 8 * (lane // 16) + lane % 8 & 7) << 4)) // 16) + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16) - T.cast(smem, "int32")]),
                )  # fmt: skip
                _mma_m16n8k16_bf16_acc(acc, a_frag, b_frag)
                _mma_m16n8k16_bf16_acc_off4(acc, a_frag, b_frag)
            else:  # .cu:1991-2000
                for _zi in T.unroll(8):
                    acc[_zi] = T.float32(0.0)
            row0_1: T.int32 = pair_row_base + lane // 4  # .cu:2001
            row1_1: T.int32 = row0_1 + 8  # .cu:2002
            col0_1: T.int32 = pair_col_base + lane % 4 * 2  # .cu:2003
            mqk: T.f32[8]  # .cu:2004
            for _zi in T.unroll(8):  # .cu:2005-2012
                mqk[_zi] = T.float32(0.0)
            if row0_1 >= col0_1:  # .cu:2013-2015
                mqk[0] = acc[0]
            if row0_1 >= col0_1 + 1:  # .cu:2016-2018
                mqk[1] = acc[1]
            if row1_1 >= col0_1:  # .cu:2019-2021
                mqk[2] = acc[2]
            if row1_1 >= col0_1 + 1:  # .cu:2022-2024
                mqk[3] = acc[3]
            if row0_1 >= col0_1 + 8:  # .cu:2025-2027
                mqk[4] = acc[4]
            if row0_1 >= col0_1 + 9:  # .cu:2028-2030
                mqk[5] = acc[5]
            if row1_1 >= col0_1 + 8:  # .cu:2031-2033
                mqk[6] = acc[6]
            if row1_1 >= col0_1 + 9:  # .cu:2034-2036
                mqk[7] = acc[7]
            mqk_packed = T.alloc_local((4,), "uint32", align=4)  # .cu:2037
            for _lp in T.unroll(4):  # .cu:2038-2042
                mqk_packed[_lp] = T.cuda.float22bfloat162_rn(mqk[_lp * 2 + 0], mqk[_lp * 2 + 1 + 0])
            for publish_pair in T.unroll(2):  # .cu:2043-2051
                publish_row: T.int32 = pair_col_base + publish_pair * 8 + (lane & 7)  # .cu:2045
                publish_col: T.int32 = 128 + pair_row_base + lane // 8 * 8  # .cu:2046
                _pub_base: T.int32 = (
                    publish_col // 64 * 4096 + publish_row * 128 + publish_col % 64 * 2
                )
                _pub_addr: T.int32 = (
                    smem_final_trans_addr
                    + T.cast(prep_stage, "int32") * 41984
                    + (_pub_base ^ ((_pub_base >> 7 & 7) << 4))
                )  # .cu:2047
                # .cu:2048-2050 stmatrix.x2.trans
                T.ptx.stmatrix.sync.aligned.m8n8.x2.trans.shared.b16(
                    smem_raw.ptr_to([(_pub_addr) - T.cast(smem, "int32")]),
                    mqk_packed[publish_pair * 2],
                    mqk_packed[publish_pair * 2 + 1],
                )
            if prep_instance == 0:  # .cu:2052-2064
                T.ptx.bar.sync(T.uint32(11), T.uint32(128))
            elif prep_instance == 1:
                T.ptx.bar.sync(T.uint32(12), T.uint32(128))
            else:
                if prep_instance == 2:
                    T.ptx.bar.sync(T.uint32(13), T.uint32(128))
                elif prep_instance == 3:
                    T.ptx.bar.sync(T.uint32(14), T.uint32(128))
                else:
                    T.ptx.bar.sync(T.uint32(15), T.uint32(128))
            if prep_tid < 128:  # .cu:2065-2069
                total_log2_1: T.f32 = smem_gt_prefix_all[stage_f32 + prep_tid]  # .cu:2066
                smem_gt_all[stage_f32 + prep_tid] = _approx_exp2(total_log2_1)  # .cu:2067-2068
            if prep_local_warp >= 2:  # .cu:2070-2187
                stage_f32_0: T.int32 = T.cast(prep_stage, "int32") * 10496  # .cu:2071
                restore_scale: T.f32 = smem_restore_factor_all[stage_f32_0 + 128]  # .cu:2072
                restore_factor: T.f32[8]  # .cu:2073
                restore_segment: T.int32 = lane & 15  # .cu:2074
                for restore_elem in T.unroll(8):  # .cu:2075-2079
                    restore_col: T.int32 = restore_segment * 8 + restore_elem  # .cu:2077
                    restore_factor[restore_elem] = smem_restore_factor_all[
                        stage_f32_0 + restore_col
                    ]  # .cu:2078
                for restore_pass in T.serial(
                    0, 6, unroll=False
                ):  # .cu:2080-2081 (#pragma unroll 1)
                    restore_row: T.int32 = (
                        8 + (prep_local_warp - 2) * 12 + restore_pass * 2 + (lane >> 4)
                    )  # .cu:2082
                    restore_qd_values: T.f32[8]  # .cu:2083
                    restore_kd_values: T.f32[8]  # .cu:2084
                    restore_ki_values: T.f32[8]  # .cu:2085
                    packed_2 = T.alloc_local((4,), "uint32", align=16)  # .cu:2086
                    _ld_shared_v4(
                        smem_raw,
                        T.cast(smem, "int32"),
                        packed_2,
                        smem_qd_addr + T.cast(prep_stage, "int32") * 41984 + (restore_segment * 8 // 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ ((restore_segment * 8 // 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)),
                    )  # .cu:2087-2089  # fmt: skip
                    packed_fp32_1: T.f32[8]  # .cu:2090
                    for _pair in T.unroll(4):  # .cu:2091-2100
                        packed_fp32_1[_pair * 2] = T.cuda.uint_as_float(
                            packed_2[_pair + 0] << T.uint32(16)
                        )
                        packed_fp32_1[_pair * 2 + 1] = T.cuda.uint_as_float(
                            packed_2[_pair + 0] & T.uint32(0xFFFF0000)
                        )
                    for value_idx_2 in T.unroll(8):  # .cu:2101-2104
                        restore_qd_values[value_idx_2] = packed_fp32_1[value_idx_2]
                    packed_0_2 = T.alloc_local((4,), "uint32", align=16)  # .cu:2105
                    _ld_shared_v4(
                        smem_raw,
                        T.cast(smem, "int32"),
                        packed_0_2,
                        smem_kd_addr + T.cast(prep_stage, "int32") * 41984 + (restore_segment * 8 // 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ ((restore_segment * 8 // 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)),
                    )  # .cu:2106-2108  # fmt: skip
                    packed_0_fp32_1: T.f32[8]  # .cu:2109
                    for _pair in T.unroll(4):  # .cu:2110-2119
                        packed_0_fp32_1[_pair * 2] = T.cuda.uint_as_float(
                            packed_0_2[_pair + 0] << T.uint32(16)
                        )
                        packed_0_fp32_1[_pair * 2 + 1] = T.cuda.uint_as_float(
                            packed_0_2[_pair + 0] & T.uint32(0xFFFF0000)
                        )
                    for value_idx_3 in T.unroll(8):  # .cu:2120-2123
                        restore_kd_values[value_idx_3] = packed_0_fp32_1[value_idx_3]
                    packed_1_2 = T.alloc_local((4,), "uint32", align=16)  # .cu:2124
                    _ld_shared_v4(
                        smem_raw,
                        T.cast(smem, "int32"),
                        packed_1_2,
                        smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + (restore_segment * 8 // 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ ((restore_segment * 8 // 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)),
                    )  # .cu:2125-2127  # fmt: skip
                    packed_1_fp32: T.f32[8]  # .cu:2128
                    for _pair in T.unroll(4):  # .cu:2129-2138
                        packed_1_fp32[_pair * 2] = T.cuda.uint_as_float(
                            packed_1_2[_pair + 0] << T.uint32(16)
                        )
                        packed_1_fp32[_pair * 2 + 1] = T.cuda.uint_as_float(
                            packed_1_2[_pair + 0] & T.uint32(0xFFFF0000)
                        )
                    for value_idx_4 in T.unroll(8):  # .cu:2139-2142
                        restore_ki_values[value_idx_4] = packed_1_fp32[value_idx_4]
                    restore_kr_values: T.f32[8]  # .cu:2143
                    for restore_elem_1 in T.unroll(8):  # .cu:2144-2147
                        restore_kr_values[restore_elem_1] = (
                            restore_ki_values[restore_elem_1] * restore_factor[restore_elem_1]
                        )  # .cu:2146
                    for _ls in T.unroll(4):  # .cu:2148-2151
                        _pk = _mul_f32x2_inplace(
                            T.cuda.make_float2(
                                restore_qd_values[_ls * 2], restore_qd_values[_ls * 2 + 1]
                            ),
                            T.cuda.make_float2(restore_scale, restore_scale),
                        )
                        restore_qd_values[_ls * 2] = T.cuda.float2_x(_pk)
                        restore_qd_values[_ls * 2 + 1] = T.cuda.float2_y(_pk)
                    for _ls in T.unroll(4):  # .cu:2152-2155
                        _pk = _mul_f32x2_inplace(
                            T.cuda.make_float2(
                                restore_kd_values[_ls * 2], restore_kd_values[_ls * 2 + 1]
                            ),
                            T.cuda.make_float2(restore_scale, restore_scale),
                        )
                        restore_kd_values[_ls * 2] = T.cuda.float2_x(_pk)
                        restore_kd_values[_ls * 2 + 1] = T.cuda.float2_y(_pk)
                    packed_2_1 = T.alloc_local((4,), "uint32", align=4)  # .cu:2156
                    for _lp in T.unroll(4):  # .cu:2157-2161
                        packed_2_1[_lp] = T.cuda.float22bfloat162_rn(
                            restore_qd_values[_lp * 2 + 0], restore_qd_values[_lp * 2 + 1 + 0]
                        )
                    for word_3 in T.unroll(4):  # .cu:2162-2165
                        _st_shared_b32(
                            smem_raw,
                            T.cast(smem, "int32"),
                            smem_qd_addr + T.cast(prep_stage, "int32") * 41984 + (restore_segment * 8 // 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ ((restore_segment * 8 // 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + word_3 * 4,
                            packed_2_1[word_3],
                        )  # fmt: skip
                    packed_3 = T.alloc_local((4,), "uint32", align=4)  # .cu:2166
                    for _lp in T.unroll(4):  # .cu:2167-2171
                        packed_3[_lp] = T.cuda.float22bfloat162_rn(
                            restore_kd_values[_lp * 2 + 0], restore_kd_values[_lp * 2 + 1 + 0]
                        )
                    for word_4 in T.unroll(4):  # .cu:2172-2175
                        _st_shared_b32(
                            smem_raw,
                            T.cast(smem, "int32"),
                            smem_kd_addr + T.cast(prep_stage, "int32") * 41984 + (restore_segment * 8 // 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ ((restore_segment * 8 // 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + word_4 * 4,
                            packed_3[word_4],
                        )  # fmt: skip
                    packed_4 = T.alloc_local((4,), "uint32", align=4)  # .cu:2176
                    for _lp in T.unroll(4):  # .cu:2177-2181
                        packed_4[_lp] = T.cuda.float22bfloat162_rn(
                            restore_kr_values[_lp * 2 + 0], restore_kr_values[_lp * 2 + 1 + 0]
                        )
                    for word_5 in T.unroll(4):  # .cu:2182-2185
                        _st_shared_b32(
                            smem_raw,
                            T.cast(smem, "int32"),
                            smem_kr_trans_addr + T.cast(prep_stage, "int32") * 41984 + (restore_segment * 8 // 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ ((restore_segment * 8 // 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + word_5 * 4,
                            packed_4[word_5],
                        )  # fmt: skip
            if prep_local_warp == 0:  # .cu:2188-2250
                inverse_row: T.int32 = lane  # .cu:2189
                diag_block: T.int32 = inverse_row // 8  # .cu:2190
                lane_in_diag: T.int32 = lane & 7  # .cu:2191
                inv_row: T.f32[8]  # .cu:2192
                packed_5 = T.alloc_local((4,), "uint32", align=16)  # .cu:2193
                byte_off_1: T.int32 = inverse_row * 128 + diag_block * 8 * 2  # .cu:2194
                swizzled_off_1: T.int32 = byte_off_1 ^ ((byte_off_1 >> 7 & 7) << 4)  # .cu:2195
                _ld_shared_v4(
                    smem_raw,
                    T.cast(smem, "int32"),
                    packed_5,
                    smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off_1,
                )  # .cu:2196-2198
                packed_fp32_2: T.f32[8]  # .cu:2199
                for _pair in T.unroll(4):  # .cu:2200-2209
                    packed_fp32_2[_pair * 2] = T.cuda.uint_as_float(
                        packed_5[_pair + 0] << T.uint32(16)
                    )
                    packed_fp32_2[_pair * 2 + 1] = T.cuda.uint_as_float(
                        packed_5[_pair + 0] & T.uint32(0xFFFF0000)
                    )
                for value_idx_5 in T.unroll(8):  # .cu:2210-2213
                    inv_row[value_idx_5] = packed_fp32_2[value_idx_5]
                for diag_elem in T.unroll(8):  # .cu:2214-2219
                    if lane_in_diag == diag_elem:
                        inv_row[diag_elem] = T.float32(1.0)
                diag_group_base: T.int32 = lane - lane_in_diag  # .cu:2220
                row_scale: T.f32 = -inv_row[0]  # .cu:2223 (statically-expanded pivot row 0)
                if lane_in_diag > 0:  # .cu:2234-2236
                    inv_row[0] = row_scale
                row_scale_1: T.f32 = -inv_row[1]  # .cu:2223 (statically-expanded pivot row 1)
                pivot_lane_1_0: T.int32 = diag_group_base + 1  # .cu:2226
                pivot_1_0: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[0], pivot_lane_1_0, 32
                )  # .cu:2227-2228
                if lane_in_diag > 1:  # .cu:2229-2232
                    inv_row[0] = _fmaf_rn(row_scale_1, pivot_1_0, inv_row[0])  # .cu:2230-2231
                if lane_in_diag > 1:  # .cu:2234-2236
                    inv_row[1] = row_scale_1
                row_scale_2: T.f32 = -inv_row[2]  # .cu:2223 (statically-expanded pivot row 2)
                pivot_lane_2_0: T.int32 = diag_group_base + 2  # .cu:2226
                pivot_2_0: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[0], pivot_lane_2_0, 32
                )  # .cu:2227-2228
                if lane_in_diag > 2:  # .cu:2229-2232
                    inv_row[0] = _fmaf_rn(row_scale_2, pivot_2_0, inv_row[0])  # .cu:2230-2231
                pivot_lane_2_1: T.int32 = diag_group_base + 2  # .cu:2226
                pivot_2_1: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[1], pivot_lane_2_1, 32
                )  # .cu:2227-2228
                if lane_in_diag > 2:  # .cu:2229-2232
                    inv_row[1] = _fmaf_rn(row_scale_2, pivot_2_1, inv_row[1])  # .cu:2230-2231
                if lane_in_diag > 2:  # .cu:2234-2236
                    inv_row[2] = row_scale_2
                row_scale_3: T.f32 = -inv_row[3]  # .cu:2223 (statically-expanded pivot row 3)
                pivot_lane_3_0: T.int32 = diag_group_base + 3  # .cu:2226
                pivot_3_0: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[0], pivot_lane_3_0, 32
                )  # .cu:2227-2228
                if lane_in_diag > 3:  # .cu:2229-2232
                    inv_row[0] = _fmaf_rn(row_scale_3, pivot_3_0, inv_row[0])  # .cu:2230-2231
                pivot_lane_3_1: T.int32 = diag_group_base + 3  # .cu:2226
                pivot_3_1: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[1], pivot_lane_3_1, 32
                )  # .cu:2227-2228
                if lane_in_diag > 3:  # .cu:2229-2232
                    inv_row[1] = _fmaf_rn(row_scale_3, pivot_3_1, inv_row[1])  # .cu:2230-2231
                pivot_lane_3_2: T.int32 = diag_group_base + 3  # .cu:2226
                pivot_3_2: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[2], pivot_lane_3_2, 32
                )  # .cu:2227-2228
                if lane_in_diag > 3:  # .cu:2229-2232
                    inv_row[2] = _fmaf_rn(row_scale_3, pivot_3_2, inv_row[2])  # .cu:2230-2231
                if lane_in_diag > 3:  # .cu:2234-2236
                    inv_row[3] = row_scale_3
                row_scale_4: T.f32 = -inv_row[4]  # .cu:2223 (statically-expanded pivot row 4)
                pivot_lane_4_0: T.int32 = diag_group_base + 4  # .cu:2226
                pivot_4_0: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[0], pivot_lane_4_0, 32
                )  # .cu:2227-2228
                if lane_in_diag > 4:  # .cu:2229-2232
                    inv_row[0] = _fmaf_rn(row_scale_4, pivot_4_0, inv_row[0])  # .cu:2230-2231
                pivot_lane_4_1: T.int32 = diag_group_base + 4  # .cu:2226
                pivot_4_1: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[1], pivot_lane_4_1, 32
                )  # .cu:2227-2228
                if lane_in_diag > 4:  # .cu:2229-2232
                    inv_row[1] = _fmaf_rn(row_scale_4, pivot_4_1, inv_row[1])  # .cu:2230-2231
                pivot_lane_4_2: T.int32 = diag_group_base + 4  # .cu:2226
                pivot_4_2: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[2], pivot_lane_4_2, 32
                )  # .cu:2227-2228
                if lane_in_diag > 4:  # .cu:2229-2232
                    inv_row[2] = _fmaf_rn(row_scale_4, pivot_4_2, inv_row[2])  # .cu:2230-2231
                pivot_lane_4_3: T.int32 = diag_group_base + 4  # .cu:2226
                pivot_4_3: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[3], pivot_lane_4_3, 32
                )  # .cu:2227-2228
                if lane_in_diag > 4:  # .cu:2229-2232
                    inv_row[3] = _fmaf_rn(row_scale_4, pivot_4_3, inv_row[3])  # .cu:2230-2231
                if lane_in_diag > 4:  # .cu:2234-2236
                    inv_row[4] = row_scale_4
                row_scale_5: T.f32 = -inv_row[5]  # .cu:2223 (statically-expanded pivot row 5)
                pivot_lane_5_0: T.int32 = diag_group_base + 5  # .cu:2226
                pivot_5_0: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[0], pivot_lane_5_0, 32
                )  # .cu:2227-2228
                if lane_in_diag > 5:  # .cu:2229-2232
                    inv_row[0] = _fmaf_rn(row_scale_5, pivot_5_0, inv_row[0])  # .cu:2230-2231
                pivot_lane_5_1: T.int32 = diag_group_base + 5  # .cu:2226
                pivot_5_1: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[1], pivot_lane_5_1, 32
                )  # .cu:2227-2228
                if lane_in_diag > 5:  # .cu:2229-2232
                    inv_row[1] = _fmaf_rn(row_scale_5, pivot_5_1, inv_row[1])  # .cu:2230-2231
                pivot_lane_5_2: T.int32 = diag_group_base + 5  # .cu:2226
                pivot_5_2: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[2], pivot_lane_5_2, 32
                )  # .cu:2227-2228
                if lane_in_diag > 5:  # .cu:2229-2232
                    inv_row[2] = _fmaf_rn(row_scale_5, pivot_5_2, inv_row[2])  # .cu:2230-2231
                pivot_lane_5_3: T.int32 = diag_group_base + 5  # .cu:2226
                pivot_5_3: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[3], pivot_lane_5_3, 32
                )  # .cu:2227-2228
                if lane_in_diag > 5:  # .cu:2229-2232
                    inv_row[3] = _fmaf_rn(row_scale_5, pivot_5_3, inv_row[3])  # .cu:2230-2231
                pivot_lane_5_4: T.int32 = diag_group_base + 5  # .cu:2226
                pivot_5_4: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[4], pivot_lane_5_4, 32
                )  # .cu:2227-2228
                if lane_in_diag > 5:  # .cu:2229-2232
                    inv_row[4] = _fmaf_rn(row_scale_5, pivot_5_4, inv_row[4])  # .cu:2230-2231
                if lane_in_diag > 5:  # .cu:2234-2236
                    inv_row[5] = row_scale_5
                row_scale_6: T.f32 = -inv_row[6]  # .cu:2223 (statically-expanded pivot row 6)
                pivot_lane_6_0: T.int32 = diag_group_base + 6  # .cu:2226
                pivot_6_0: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[0], pivot_lane_6_0, 32
                )  # .cu:2227-2228
                if lane_in_diag > 6:  # .cu:2229-2232
                    inv_row[0] = _fmaf_rn(row_scale_6, pivot_6_0, inv_row[0])  # .cu:2230-2231
                pivot_lane_6_1: T.int32 = diag_group_base + 6  # .cu:2226
                pivot_6_1: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[1], pivot_lane_6_1, 32
                )  # .cu:2227-2228
                if lane_in_diag > 6:  # .cu:2229-2232
                    inv_row[1] = _fmaf_rn(row_scale_6, pivot_6_1, inv_row[1])  # .cu:2230-2231
                pivot_lane_6_2: T.int32 = diag_group_base + 6  # .cu:2226
                pivot_6_2: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[2], pivot_lane_6_2, 32
                )  # .cu:2227-2228
                if lane_in_diag > 6:  # .cu:2229-2232
                    inv_row[2] = _fmaf_rn(row_scale_6, pivot_6_2, inv_row[2])  # .cu:2230-2231
                pivot_lane_6_3: T.int32 = diag_group_base + 6  # .cu:2226
                pivot_6_3: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[3], pivot_lane_6_3, 32
                )  # .cu:2227-2228
                if lane_in_diag > 6:  # .cu:2229-2232
                    inv_row[3] = _fmaf_rn(row_scale_6, pivot_6_3, inv_row[3])  # .cu:2230-2231
                pivot_lane_6_4: T.int32 = diag_group_base + 6  # .cu:2226
                pivot_6_4: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[4], pivot_lane_6_4, 32
                )  # .cu:2227-2228
                if lane_in_diag > 6:  # .cu:2229-2232
                    inv_row[4] = _fmaf_rn(row_scale_6, pivot_6_4, inv_row[4])  # .cu:2230-2231
                pivot_lane_6_5: T.int32 = diag_group_base + 6  # .cu:2226
                pivot_6_5: T.f32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), inv_row[5], pivot_lane_6_5, 32
                )  # .cu:2227-2228
                if lane_in_diag > 6:  # .cu:2229-2232
                    inv_row[5] = _fmaf_rn(row_scale_6, pivot_6_5, inv_row[5])  # .cu:2230-2231
                if lane_in_diag > 6:  # .cu:2234-2236
                    inv_row[6] = row_scale_6
                packed_0_3 = T.alloc_local((4,), "uint32", align=4)  # .cu:2238
                for _lp in T.unroll(4):  # .cu:2239-2243
                    packed_0_3[_lp] = T.cuda.float22bfloat162_rn(
                        inv_row[_lp * 2 + 0], inv_row[_lp * 2 + 1 + 0]
                    )
                byte_off_1_1: T.int32 = inverse_row * 128 + diag_block * 8 * 2  # .cu:2244
                swizzled_off_2: T.int32 = byte_off_1_1 ^ ((byte_off_1_1 >> 7 & 7) << 4)  # .cu:2245
                for word_6 in T.unroll(4):  # .cu:2246-2249
                    _st_shared_b32(
                        smem_raw,
                        T.cast(smem, "int32"),
                        smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off_2 + word_6 * 4,
                        packed_0_3[word_6],
                    )  # fmt: skip
            if prep_local_warp < 2:  # .cu:2251-2256
                if T.cuda.elect_sync():
                    _mbarrier_arrive(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_PREP_DIAG_READY_OFF + prep_stage * 8,
                    )
                _mbarrier_wait(
                    smem_raw,
                    T.cast(smem, "int32"),
                    mbar_base + MBAR_PREP_DIAG_READY_OFF + prep_stage * 8,
                    _phase_prep_diag_ready,
                )
            if prep_local_warp < 2:  # .cu:2257-2322
                lane_row: T.int32 = lane & 7  # .cu:2258
                byte_off_2: T.int32 = (prep_local_warp * 16 + 8 + lane_row) * 128 + (
                    prep_local_warp * 16 + 8
                ) * 2  # .cu:2259
                swizzled_off_3: T.int32 = byte_off_2 ^ ((byte_off_2 >> 7 & 7) << 4)  # .cu:2260
                d_addr: T.int32 = (
                    smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off_3
                )  # .cu:2261
                byte_off_0: T.int32 = (prep_local_warp * 16 + 8 + lane_row) * 128 + (
                    prep_local_warp * 16 * 2
                )  # .cu:2262
                swizzled_off_1_1: T.int32 = byte_off_0 ^ ((byte_off_0 >> 7 & 7) << 4)  # .cu:2263
                c_addr: T.int32 = (
                    smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off_1_1
                )  # .cu:2264
                byte_off_2_1: T.int32 = (prep_local_warp * 16 + lane_row) * 128 + (
                    prep_local_warp * 16 * 2
                )  # .cu:2265
                swizzled_off_3_1: T.int32 = byte_off_2_1 ^ (
                    (byte_off_2_1 >> 7 & 7) << 4
                )  # .cu:2266
                a_addr: T.int32 = (
                    smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off_3_1
                )  # .cu:2267
                d_frag = T.alloc_local((2,), "uint32", align=4)  # .cu:2268
                c_frag = T.alloc_local((1,), "uint32", align=4)  # .cu:2269
                dc_acc = T.alloc_local((4,), "float32", align=4)  # .cu:2270
                dc_bf16 = T.alloc_local((2,), "uint32", align=4)  # .cu:2271
                inv_a_frag = T.alloc_local((1,), "uint32", align=4)  # .cu:2272
                o_acc = T.alloc_local((4,), "float32", align=4)  # .cu:2273
                o_bf16 = T.alloc_local((2,), "uint32", align=4)  # .cu:2274
                # .cu:2275-2278 ldmatrix.x1 d_frag[0]
                T.ptx.ldmatrix.sync.aligned.m8n8.x1.shared.b16(
                    d_frag[0], smem_raw.ptr_to([(d_addr) - T.cast(smem, "int32")])
                )
                # .cu:2279-2282 ldmatrix.x1 d_frag[1] (same address)
                T.ptx.ldmatrix.sync.aligned.m8n8.x1.shared.b16(
                    d_frag[1], smem_raw.ptr_to([(d_addr) - T.cast(smem, "int32")])
                )
                # .cu:2283-2286 ldmatrix.x1.trans c_frag
                T.ptx.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16(
                    c_frag[0], smem_raw.ptr_to([(c_addr) - T.cast(smem, "int32")])
                )
                _mma_m16n8k8_bf16_zero(dc_acc, d_frag, c_frag)  # .cu:2287-2289
                for _ls in T.unroll(2):  # .cu:2290-2293
                    _pk = _mul_f32x2_inplace(
                        T.cuda.make_float2(dc_acc[_ls * 2], dc_acc[_ls * 2 + 1]),
                        T.cuda.make_float2(T.float32(-1.0), T.float32(-1.0)),
                    )
                    dc_acc[_ls * 2] = T.cuda.float2_x(_pk)
                    dc_acc[_ls * 2 + 1] = T.cuda.float2_y(_pk)
                for _lp in T.unroll(2):  # .cu:2294-2298
                    dc_bf16[_lp] = T.cuda.float22bfloat162_rn(
                        dc_acc[_lp * 2 + 0], dc_acc[_lp * 2 + 1 + 0]
                    )
                # .cu:2299-2302 ldmatrix.x1.trans inv_a_frag
                T.ptx.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16(
                    inv_a_frag[0], smem_raw.ptr_to([(a_addr) - T.cast(smem, "int32")])
                )
                _mma_m16n8k8_bf16_zero(o_acc, dc_bf16, inv_a_frag)  # .cu:2303-2305
                for _lp in T.unroll(2):  # .cu:2306-2310
                    o_bf16[_lp] = T.cuda.float22bfloat162_rn(
                        o_acc[_lp * 2 + 0], o_acc[_lp * 2 + 1 + 0]
                    )
                byte_off_4: T.int32 = (prep_local_warp * 16 + 8 + lane_row) * 128 + (
                    prep_local_warp * 16 * 2
                )  # .cu:2311
                swizzled_off_5: T.int32 = byte_off_4 ^ ((byte_off_4 >> 7 & 7) << 4)  # .cu:2312
                o_addr: T.int32 = (
                    smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off_5
                )  # .cu:2313
                # .cu:2314-2317 stmatrix.x1
                T.ptx.stmatrix.sync.aligned.m8n8.x1.shared.b16(
                    smem_raw.ptr_to([(o_addr) - T.cast(smem, "int32")]), o_bf16[0]
                )
                if T.cuda.elect_sync():  # .cu:2318-2320
                    _mbarrier_arrive(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_PREP_INV16_READY_OFF + prep_stage * 8,
                    )
                _mbarrier_wait(
                    smem_raw,
                    T.cast(smem, "int32"),
                    mbar_base + MBAR_PREP_INV16_READY_OFF + prep_stage * 8,
                    _phase_prep_inv16_ready,
                )  # .cu:2321
            if prep_local_warp == 0:  # .cu:2323-2404
                lane_row_1: T.int32 = lane % 16  # .cu:2324
                lane_col: T.int32 = lane // 16 * 8  # .cu:2325
                byte_off_3: T.int32 = (16 + lane_row_1) * 128 + (16 + lane_col) * 2  # .cu:2326
                swizzled_off_4: T.int32 = byte_off_3 ^ ((byte_off_3 >> 7 & 7) << 4)  # .cu:2327
                d_addr_1: T.int32 = (
                    smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off_4
                )  # .cu:2328
                byte_off_0_1: T.int32 = (16 + lane_row_1) * 128 + lane_col * 2  # .cu:2329
                swizzled_off_1_2: T.int32 = byte_off_0_1 ^ (
                    (byte_off_0_1 >> 7 & 7) << 4
                )  # .cu:2330
                c_addr_1: T.int32 = (
                    smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off_1_2
                )  # .cu:2331
                byte_off_2_2: T.int32 = lane_row_1 * 128 + lane_col * 2  # .cu:2332
                swizzled_off_3_2: T.int32 = byte_off_2_2 ^ (
                    (byte_off_2_2 >> 7 & 7) << 4
                )  # .cu:2333
                a_addr_1: T.int32 = (
                    smem_inv_work_addr + T.cast(prep_stage, "int32") * 41984 + swizzled_off_3_2
                )  # .cu:2334
                d32_frag = T.alloc_local((4,), "uint32", align=4)  # .cu:2335
                c32_frag = T.alloc_local((4,), "uint32", align=4)  # .cu:2336
                dc32_acc = T.alloc_local((8,), "float32", align=4)  # .cu:2337
                dc32_bf16 = T.alloc_local((4,), "uint32", align=4)  # .cu:2338
                a32_frag = T.alloc_local((4,), "uint32", align=4)  # .cu:2339
                o32_acc = T.alloc_local((8,), "float32", align=4)  # .cu:2340
                o32_bf16 = T.alloc_local((4,), "uint32", align=4)  # .cu:2341
                zero32_bf16 = T.alloc_local((4,), "uint32", align=4)  # .cu:2342
                # .cu:2343-2346 ldmatrix.x4 d32
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                    d32_frag[0],
                    d32_frag[1],
                    d32_frag[2],
                    d32_frag[3],
                    smem_raw.ptr_to([(d_addr_1) - T.cast(smem, "int32")]),
                )
                _dpb: T.int32 = (
                    (16 + lane_col) // 16 * 1024 + (16 + lane_row_1) * 32 + (16 + lane_col) % 16 * 2
                )
                d_publish_addr: T.int32 = (
                    smem_inv_addr
                    + T.cast(prep_stage, "int32") * 41984
                    + (_dpb ^ ((_dpb >> 7 & 1) << 4))
                )  # .cu:2347
                # .cu:2348-2351 stmatrix.x4 d32 publish
                T.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                    smem_raw.ptr_to([(d_publish_addr) - T.cast(smem, "int32")]),
                    d32_frag[0],
                    d32_frag[1],
                    d32_frag[2],
                    d32_frag[3],
                )
                # .cu:2352-2355 ldmatrix.x4.trans c32
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                    c32_frag[0],
                    c32_frag[1],
                    c32_frag[2],
                    c32_frag[3],
                    smem_raw.ptr_to([(c_addr_1) - T.cast(smem, "int32")]),
                )
                _mma_m16n8k16_bf16_zero(dc32_acc, d32_frag, c32_frag)  # .cu:2356-2358
                _mma_m16n8k16_bf16_zero_off4(dc32_acc, d32_frag, c32_frag)  # .cu:2359-2361
                for _ls in T.unroll(4):  # .cu:2362-2365
                    _pk = _mul_f32x2_inplace(
                        T.cuda.make_float2(dc32_acc[_ls * 2], dc32_acc[_ls * 2 + 1]),
                        T.cuda.make_float2(T.float32(-1.0), T.float32(-1.0)),
                    )
                    dc32_acc[_ls * 2] = T.cuda.float2_x(_pk)
                    dc32_acc[_ls * 2 + 1] = T.cuda.float2_y(_pk)
                for _lp in T.unroll(4):  # .cu:2366-2370
                    dc32_bf16[_lp] = T.cuda.float22bfloat162_rn(
                        dc32_acc[_lp * 2 + 0], dc32_acc[_lp * 2 + 1 + 0]
                    )
                # .cu:2371-2374 ldmatrix.x4.trans a32
                T.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                    a32_frag[0],
                    a32_frag[1],
                    a32_frag[2],
                    a32_frag[3],
                    smem_raw.ptr_to([(a_addr_1) - T.cast(smem, "int32")]),
                )
                _apb: T.int32 = lane_col // 16 * 1024 + lane_row_1 * 32 + lane_col % 16 * 2
                a_publish_addr: T.int32 = (
                    smem_inv_addr
                    + T.cast(prep_stage, "int32") * 41984
                    + (_apb ^ ((_apb >> 7 & 1) << 4))
                )  # .cu:2375
                # .cu:2376-2379 stmatrix.x4.trans a32 publish
                T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                    smem_raw.ptr_to([(a_publish_addr) - T.cast(smem, "int32")]),
                    a32_frag[0],
                    a32_frag[1],
                    a32_frag[2],
                    a32_frag[3],
                )
                _mma_m16n8k16_bf16_zero(o32_acc, dc32_bf16, a32_frag)  # .cu:2380-2382
                _mma_m16n8k16_bf16_zero_off4(o32_acc, dc32_bf16, a32_frag)  # .cu:2383-2385
                for _lp in T.unroll(4):  # .cu:2386-2390
                    o32_bf16[_lp] = T.cuda.float22bfloat162_rn(
                        o32_acc[_lp * 2 + 0], o32_acc[_lp * 2 + 1 + 0]
                    )
                _opb: T.int32 = lane_col // 16 * 1024 + (16 + lane_row_1) * 32 + lane_col % 16 * 2
                o_publish_addr: T.int32 = (
                    smem_inv_addr
                    + T.cast(prep_stage, "int32") * 41984
                    + (_opb ^ ((_opb >> 7 & 1) << 4))
                )  # .cu:2391
                # .cu:2392-2395 stmatrix.x4 o32 publish
                T.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                    smem_raw.ptr_to([(o_publish_addr) - T.cast(smem, "int32")]),
                    o32_bf16[0],
                    o32_bf16[1],
                    o32_bf16[2],
                    o32_bf16[3],
                )
                for zero_word in T.unroll(4):  # .cu:2396-2399
                    zero32_bf16[zero_word] = T.uint32(0)
                _zpb: T.int32 = (
                    (16 + lane_col) // 16 * 1024 + lane_row_1 * 32 + (16 + lane_col) % 16 * 2
                )
                zero_publish_addr: T.int32 = (
                    smem_inv_addr
                    + T.cast(prep_stage, "int32") * 41984
                    + (_zpb ^ ((_zpb >> 7 & 1) << 4))
                )  # .cu:2400
                # .cu:2401-2404 stmatrix.x4 zero publish
                T.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                    smem_raw.ptr_to([(zero_publish_addr) - T.cast(smem, "int32")]),
                    zero32_bf16[0],
                    zero32_bf16[1],
                    zero32_bf16[2],
                    zero32_bf16[3],
                )
            elif prep_local_warp == 1:  # .cu:2405-2522
                stage_f32_0_1: T.int32 = T.cast(prep_stage, "int32") * 10496  # .cu:2406
                restore_scale_1: T.f32 = smem_restore_factor_all[stage_f32_0_1 + 128]  # .cu:2407
                restore_factor_1: T.f32[8]  # .cu:2408
                restore_segment_1: T.int32 = lane & 15  # .cu:2409
                for restore_elem_2 in T.unroll(8):  # .cu:2410-2414
                    restore_col_1: T.int32 = restore_segment_1 * 8 + restore_elem_2  # .cu:2412
                    restore_factor_1[restore_elem_2] = smem_restore_factor_all[
                        stage_f32_0_1 + restore_col_1
                    ]  # .cu:2413
                for restore_pass_1 in T.serial(0, 4, unroll=False):  # .cu:2415-2416
                    restore_row_1: T.int32 = restore_pass_1 * 2 + (lane >> 4)  # .cu:2417
                    restore_qd_values_1: T.f32[8]  # .cu:2418
                    restore_kd_values_1: T.f32[8]  # .cu:2419
                    restore_ki_values_1: T.f32[8]  # .cu:2420
                    packed_6 = T.alloc_local((4,), "uint32", align=16)  # .cu:2421
                    _ld_shared_v4(
                        smem_raw,
                        T.cast(smem, "int32"),
                        packed_6,
                        smem_qd_addr + T.cast(prep_stage, "int32") * 41984 + (restore_segment_1 * 8 // 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ ((restore_segment_1 * 8 // 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)),
                    )  # .cu:2422-2424  # fmt: skip
                    packed_fp32_3: T.f32[8]  # .cu:2425
                    for _pair in T.unroll(4):  # .cu:2426-2435
                        packed_fp32_3[_pair * 2] = T.cuda.uint_as_float(
                            packed_6[_pair + 0] << T.uint32(16)
                        )
                        packed_fp32_3[_pair * 2 + 1] = T.cuda.uint_as_float(
                            packed_6[_pair + 0] & T.uint32(0xFFFF0000)
                        )
                    for value_idx_6 in T.unroll(8):  # .cu:2436-2439
                        restore_qd_values_1[value_idx_6] = packed_fp32_3[value_idx_6]
                    packed_0_4 = T.alloc_local((4,), "uint32", align=16)  # .cu:2440
                    _ld_shared_v4(
                        smem_raw,
                        T.cast(smem, "int32"),
                        packed_0_4,
                        smem_kd_addr + T.cast(prep_stage, "int32") * 41984 + (restore_segment_1 * 8 // 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ ((restore_segment_1 * 8 // 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)),
                    )  # .cu:2441-2443  # fmt: skip
                    packed_0_fp32_2: T.f32[8]  # .cu:2444
                    for _pair in T.unroll(4):  # .cu:2445-2454
                        packed_0_fp32_2[_pair * 2] = T.cuda.uint_as_float(
                            packed_0_4[_pair + 0] << T.uint32(16)
                        )
                        packed_0_fp32_2[_pair * 2 + 1] = T.cuda.uint_as_float(
                            packed_0_4[_pair + 0] & T.uint32(0xFFFF0000)
                        )
                    for value_idx_7 in T.unroll(8):  # .cu:2455-2458
                        restore_kd_values_1[value_idx_7] = packed_0_fp32_2[value_idx_7]
                    packed_1_3 = T.alloc_local((4,), "uint32", align=16)  # .cu:2459
                    _ld_shared_v4(
                        smem_raw,
                        T.cast(smem, "int32"),
                        packed_1_3,
                        smem_ki_addr + T.cast(prep_stage, "int32") * 41984 + (restore_segment_1 * 8 // 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ ((restore_segment_1 * 8 // 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)),
                    )  # .cu:2460-2462  # fmt: skip
                    packed_1_fp32_1: T.f32[8]  # .cu:2463
                    for _pair in T.unroll(4):  # .cu:2464-2473
                        packed_1_fp32_1[_pair * 2] = T.cuda.uint_as_float(
                            packed_1_3[_pair + 0] << T.uint32(16)
                        )
                        packed_1_fp32_1[_pair * 2 + 1] = T.cuda.uint_as_float(
                            packed_1_3[_pair + 0] & T.uint32(0xFFFF0000)
                        )
                    for value_idx_8 in T.unroll(8):  # .cu:2474-2477
                        restore_ki_values_1[value_idx_8] = packed_1_fp32_1[value_idx_8]
                    restore_kr_values_1: T.f32[8]  # .cu:2478
                    for restore_elem_3 in T.unroll(8):  # .cu:2479-2482
                        restore_kr_values_1[restore_elem_3] = (
                            restore_ki_values_1[restore_elem_3] * restore_factor_1[restore_elem_3]
                        )  # .cu:2481
                    for _ls in T.unroll(4):  # .cu:2483-2486
                        _pk = _mul_f32x2_inplace(
                            T.cuda.make_float2(
                                restore_qd_values_1[_ls * 2], restore_qd_values_1[_ls * 2 + 1]
                            ),
                            T.cuda.make_float2(restore_scale_1, restore_scale_1),
                        )
                        restore_qd_values_1[_ls * 2] = T.cuda.float2_x(_pk)
                        restore_qd_values_1[_ls * 2 + 1] = T.cuda.float2_y(_pk)
                    for _ls in T.unroll(4):  # .cu:2487-2490
                        _pk = _mul_f32x2_inplace(
                            T.cuda.make_float2(
                                restore_kd_values_1[_ls * 2], restore_kd_values_1[_ls * 2 + 1]
                            ),
                            T.cuda.make_float2(restore_scale_1, restore_scale_1),
                        )
                        restore_kd_values_1[_ls * 2] = T.cuda.float2_x(_pk)
                        restore_kd_values_1[_ls * 2 + 1] = T.cuda.float2_y(_pk)
                    packed_2_2 = T.alloc_local((4,), "uint32", align=4)  # .cu:2491
                    for _lp in T.unroll(4):  # .cu:2492-2496
                        packed_2_2[_lp] = T.cuda.float22bfloat162_rn(
                            restore_qd_values_1[_lp * 2 + 0], restore_qd_values_1[_lp * 2 + 1 + 0]
                        )
                    for word_7 in T.unroll(4):  # .cu:2497-2500
                        _st_shared_b32(
                            smem_raw,
                            T.cast(smem, "int32"),
                            smem_qd_addr + T.cast(prep_stage, "int32") * 41984 + (restore_segment_1 * 8 // 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ ((restore_segment_1 * 8 // 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + word_7 * 4,
                            packed_2_2[word_7],
                        )  # fmt: skip
                    packed_3_1 = T.alloc_local((4,), "uint32", align=4)  # .cu:2501
                    for _lp in T.unroll(4):  # .cu:2502-2506
                        packed_3_1[_lp] = T.cuda.float22bfloat162_rn(
                            restore_kd_values_1[_lp * 2 + 0], restore_kd_values_1[_lp * 2 + 1 + 0]
                        )
                    for word_8 in T.unroll(4):  # .cu:2507-2510
                        _st_shared_b32(
                            smem_raw,
                            T.cast(smem, "int32"),
                            smem_kd_addr + T.cast(prep_stage, "int32") * 41984 + (restore_segment_1 * 8 // 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ ((restore_segment_1 * 8 // 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + word_8 * 4,
                            packed_3_1[word_8],
                        )  # fmt: skip
                    packed_4_1 = T.alloc_local((4,), "uint32", align=4)  # .cu:2511
                    for _lp in T.unroll(4):  # .cu:2512-2516
                        packed_4_1[_lp] = T.cuda.float22bfloat162_rn(
                            restore_kr_values_1[_lp * 2 + 0], restore_kr_values_1[_lp * 2 + 1 + 0]
                        )
                    for word_9 in T.unroll(4):  # .cu:2517-2520
                        _st_shared_b32(
                            smem_raw,
                            T.cast(smem, "int32"),
                            smem_kr_trans_addr + T.cast(prep_stage, "int32") * 41984 + (restore_segment_1 * 8 // 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ ((restore_segment_1 * 8 // 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + word_9 * 4,
                            packed_4_1[word_9],
                        )  # fmt: skip
            _fence_async_shared()  # .cu:2523
            if prep_instance == 0:  # .cu:2524-2536
                T.ptx.bar.sync(T.uint32(11), T.uint32(128))
            elif prep_instance == 1:
                T.ptx.bar.sync(T.uint32(12), T.uint32(128))
            else:
                if prep_instance == 2:
                    T.ptx.bar.sync(T.uint32(13), T.uint32(128))
                elif prep_instance == 3:
                    T.ptx.bar.sync(T.uint32(14), T.uint32(128))
                else:
                    T.ptx.bar.sync(T.uint32(15), T.uint32(128))
            if prep_local_warp == 0:  # .cu:2537-2541
                if T.cuda.elect_sync():
                    _mbarrier_arrive(
                        smem_raw,
                        T.cast(smem, "int32"),
                        mbar_base + MBAR_QK_FULL_OFF + prep_stage * 8,
                    )
            for _advance in T.serial(0, 5):  # .cu:2542-2545
                prep_stage += 1
                if prep_stage == 5:
                    prep_stage = T.uint32(0)
                    _phase_raw_inputs_free = _phase_raw_inputs_free ^ T.uint32(1)
                    _phase_gate_raw_full = _phase_gate_raw_full ^ T.uint32(1)
                    _phase_smem_free = _phase_smem_free ^ T.uint32(1)
                    _phase_qk_raw_full = _phase_qk_raw_full ^ T.uint32(1)
                    _phase_prep_diag_ready = _phase_prep_diag_ready ^ T.uint32(1)
                    _phase_prep_inv16_ready = _phase_prep_inv16_ready ^ T.uint32(1)


def bf16_fused_m128(**kwargs: Any):
    cfg = _cfg(**kwargs)
    kernel = _kernel.specialize(
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


def run_test(**kwargs: Any) -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for FlashKDA bf16 fused m128")

    from tirx_kernels.flashkda.bf16_fused_m128_tx_tile import bf16_fused_m128_tx_tile
    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    cfg: FlashKDABf16FusedM128Config = case["config"]
    if not case["dispatch_reason"].startswith("m128:"):
        raise SkipTest(case["dispatch_reason"])
    executables = (
        (compile_kernel(bf16_fused_m128(**kwargs)), _tirx_args(case)),
        (compile_kernel(bf16_fused_m128_tx_tile(**kwargs)), _tirx_args(case, use_tx_tile=True)),
    )
    for executable, args in executables:
        executable(*args)
    torch.cuda.synchronize()

    ref_out, ref_state = _reference_torch(case)
    for suffix in ("", "_tx_tile"):
        torch.testing.assert_close(case[f"out{suffix}"], ref_out, rtol=4.01 / 128, atol=5e-3)
        if cfg.store_final_state:
            torch.testing.assert_close(
                case[f"final_state{suffix}"], ref_state, rtol=4.01 / 128, atol=5e-3
            )

    flashinfer_out, flashinfer_state = _flashinfer_cuda_reference(case)
    for suffix in ("", "_tx_tile"):
        torch.testing.assert_close(case[f"out{suffix}"], flashinfer_out, rtol=4.01 / 128, atol=5e-3)
        if cfg.store_final_state and flashinfer_state is not None:
            torch.testing.assert_close(
                case[f"final_state{suffix}"],
                flashinfer_state.reshape(case[f"final_state{suffix}"].shape),
                rtol=4.01 / 128,
                atol=5e-3,
            )
    cfg.validate()


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    _rounds = kwargs.pop("rounds", 1)
    _cooldown_s = kwargs.pop("cooldown_s", 1.0)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for FlashKDA bf16 fused m128 benchmark")

    from tirx_kernels.flashkda.bf16_fused_m128_tx_tile import bf16_fused_m128_tx_tile
    from tirx_kernels.runner import compile_kernel
    from tvm.tirx.bench import bench

    case = prepare_data(**kwargs)
    cfg: FlashKDABf16FusedM128Config = case["config"]
    if not case["dispatch_reason"].startswith("m128:"):
        raise SkipTest(case["dispatch_reason"])
    args = _tirx_args(case)
    tx_tile_args = _tirx_args(case, use_tx_tile=True)
    ex = compile_kernel(bf16_fused_m128(**kwargs))
    tx_tile_ex = compile_kernel(bf16_fused_m128_tx_tile(**kwargs))
    funcs = {"tirx": lambda: ex(*args), "tirx_tx_tile": lambda: tx_tile_ex(*tx_tile_args)}

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
        from tirx_kernels.flashkda._flashkda_bench import prepare_flashkda_raw_reference

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
