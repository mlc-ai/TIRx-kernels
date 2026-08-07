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
"""SM100a FP16 GDN prefill transcribed from FlashInfer's frozen CuTeDSL kernel.

The implementation follows the frozen CuTeDSL source order and preserves its
CUDA/PTX details instead of reorganizing them into a separate implementation.
"""

from __future__ import annotations

import ctypes
import hashlib
import inspect
import math
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any
from unittest import SkipTest

import torch
import torch.nn.functional as F

from tvm.script import tirx as T

D_HEAD = 128
CHUNK = 64
PAIR_TOKENS = 128
THREADS = 384
NUM_WARPS = 12
SMEM_TOTAL = 226048
TMEM_COLUMNS = 512
DESCRIPTOR_SLOT_BYTES = 128
DESCRIPTOR_SLOTS = 4
DESCRIPTOR_BYTES_PER_CTA = DESCRIPTOR_SLOT_BYTES * DESCRIPTOR_SLOTS
FROZEN_FLASHINFER_ADAPTER_SHA256 = (
    "dba25861bb34245a344fc4d4fb443419ecf3f425feb9053823bcd7d7165efe01"
)
FROZEN_FLASHINFER_KERNEL_SHA256 = "faaf28508f419a255cc030678ef86e73f47153631cc9403176d058b6a0bd715b"

HEAD_PAIRS = ((2, 8), (4, 16), (8, 32), (16, 64), (16, 32), (16, 48), (16, 16), (32, 32))

SEQ_CASES = (
    ((65536,), "1x65536"),
    ((32768,), "1x32768"),
    ((16384,), "1x16384"),
    ((8192,), "1x8192"),
    ((4096,), "1x4096"),
    ((2048,), "1x2048"),
    ((6144, 2048), "6144+2048"),
    ((4096, 4096), "4096+4096"),
    ((2048, 6144), "2048+6144"),
    ((1024, 7168), "1024+7168"),
    ((2048,) * 4, "2048x4"),
    ((1024,) * 8, "1024x8"),
    ((8192,) * 8, "8192x8"),
    ((8192,) * 16, "8192x16"),
    ((8192,) * 32, "8192x32"),
)


@dataclass(frozen=True, slots=True)
class GDNPrefillSM100Config:
    label: str
    hq: int
    hv: int
    seq_lens: tuple[int, ...]
    seed: int = 0

    def validate(self) -> None:
        if (self.hq, self.hv) not in HEAD_PAIRS:
            raise ValueError(f"unsupported GDN head specialization {(self.hq, self.hv)}")
        if not self.seq_lens or any(length <= 0 for length in self.seq_lens):
            raise ValueError("seq_lens must be non-empty and positive")

    @property
    def num_sequences(self) -> int:
        return len(self.seq_lens)

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)


CONFIGS = [
    {
        "label": f"hq{hq}_hv{hv}_s{seq_label}",
        "hq": hq,
        "hv": hv,
        "seq_lens": seq_lens,
        "seed": 10000 + head_idx * len(SEQ_CASES) + seq_idx,
    }
    for head_idx, (hq, hv) in enumerate(HEAD_PAIRS)
    for seq_idx, (seq_lens, seq_label) in enumerate(SEQ_CASES)
]

KERNEL_META = {"name": "gdn_prefill_sm100", "category": "attention", "compute_capability": 10}

# Shared-memory backing views, in the exact approved-sketch order.
SMEM_TMEM_HOLDING_OFF = 560
SMEM_Q_OFF = 1024
SMEM_K_OFF = 33792
SMEM_V_OFF = 99328
SMEM_AINV_OFF = 148480
SMEM_QK_OFF = 173056
SMEM_O_OFF = 189440
SMEM_CUMSUMLOG_OFF = 222208
SMEM_CUMPROD_OFF = 223488
SMEM_BETA_OFF = 224768

# Sixteen full/empty rings.  Offsets are bytes from the dynamic-SMEM base.
PIPELINE_SPECS = (
    ("load_k", 4, 0, 32, 1, 2),
    ("load_q", 2, 64, 80, 1, 2),
    ("load_v", 3, 96, 120, 1, 4),
    ("load_gate", 5, 144, 184, 32, 256),
    ("load_beta", 5, 224, 264, 32, 128),
    ("q_state_acc", 1, 304, 312, 1, 128),
    ("kv_acc", 1, 320, 328, 1, 128),
    ("cg0_acc", 2, 336, 352, 1, 128),
    ("cg1_acc", 1, 368, 376, 1, 128),
    ("ainv_ready", 3, 384, 408, 128, 1),
    ("qk_ready", 2, 432, 448, 128, 1),
    ("state_inp_ready", 1, 464, 472, 128, 1),
    ("vks_ready", 1, 480, 488, 128, 1),
    ("nv_ready", 1, 496, 504, 128, 1),
    ("decay_v_ready", 1, 512, 520, 128, 1),
    ("o_store", 2, 528, 544, 128, 32),
)

TMEM_STATE_COL = 0
TMEM_Q_STATE_COL = 128
TMEM_STATE_INPUT_COL = 192
TMEM_CG0_ACC_COL = 256
TMEM_CG1_ACC_COL = 384
TMEM_SHARED_INPUT_COL = 448

TMEM_ALLOC_BARRIER = 1
INVERSE_BARRIER = 2
INITIAL_STATE_BARRIER = 4

LAUNCH_TAGS = ("blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory")

# Only source operations unavailable as exact native TIRx calls live here.
_GDN_SRCS = {
    "replace_global_address": r"""__device__ __forceinline__ void gdn_tensormap_replace_global_address(void *desc, const void *addr) {
    asm volatile("tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;"
                 :: "l"(desc), "l"(addr) : "memory");
}
""",
    **{
        f"replace_global_dim_{dim}": f"""__device__ __forceinline__ void gdn_tensormap_replace_global_dim_{dim}(void *desc, unsigned int value) {{
    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], {dim}, %1;"
                 :: "l"(desc), "r"(value) : "memory");
}}
"""
        for dim in range(5)
    },
    **{
        f"replace_global_stride_{dim}": f"""__device__ __forceinline__ void gdn_tensormap_replace_global_stride_{dim}(void *desc, unsigned long long value) {{
    asm volatile("tensormap.replace.tile.global_stride.global.b1024.b64 [%0], {dim}, %1;"
                 :: "l"(desc), "l"(value) : "memory");
}}
"""
        for dim in range(4)
    },
    "tensormap_release": r"""__device__ __forceinline__ void gdn_tensormap_release() {
    asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
}
""",
    "tensormap_acquire": r"""__device__ __forceinline__ void gdn_tensormap_acquire(const void *desc) {
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
                 :: "l"(desc) : "memory");
}
""",
    "lg2_approx_ftz": r"""__device__ __forceinline__ float gdn_lg2_approx_ftz(float value) {
    float out;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(value));
    return out;
}
""",
}


@T.inline
def _init_pipeline(smem_raw, full_off, empty_off, stages, producers, consumers):
    for stage in range(stages):
        T.ptx.mbarrier.init.shared.b64(smem_raw.ptr_to([full_off + stage * 8]), T.uint32(producers))
        T.ptx.mbarrier.init.shared.b64(
            smem_raw.ptr_to([empty_off + stage * 8]), T.uint32(consumers)
        )


def _init_all_pipelines(smem_raw):
    for _, stages, full_off, empty_off, producers, consumers in PIPELINE_SPECS:
        _init_pipeline(smem_raw, full_off, empty_off, stages, producers, consumers)


def _make_warp_uniform(value):
    return T.cuda._shfl_sync(T.uint32(0xFFFFFFFF), value, 0, 32)


def _udiv_u32_const(value, divisor):
    # CuTe FastDivmod uses an unsigned multiply-high sequence.  Spell the two
    # non-power-of-two divisors in this specialization as their exact 64-bit
    # reciprocal forms so CUDA never introduces signed div/rem correction.
    if divisor == 3:
        return T.cast(
            T.shift_right(T.cast(value, "uint64") * T.uint64(0xAAAAAAAB), T.uint64(33)), "uint32"
        )
    if divisor == 48:
        return T.cast(
            T.shift_right(T.cast(value, "uint64") * T.uint64(0xAAAAAAAB), T.uint64(37)), "uint32"
        )
    return T.cast(value, "uint32") // T.uint32(divisor)


def _byte_ptr(ptr, byte_offset):
    return T.reinterpret("handle", T.reinterpret("uint64", ptr) + T.cast(byte_offset, "uint64"))


def _pipe_stage(count, stages):
    return T.cast(T.cast(count, "uint32") % T.uint32(stages), "int32")


def _pipe_phase(count, stages, initial_phase):
    phase = T.cast(T.bitwise_and(T.cast(count, "uint32") // T.uint32(stages), T.uint32(1)), "int32")
    return T.bitwise_xor(phase, T.int32(initial_phase))


def _shared_addr(smem_base_addr, byte_offset):
    return smem_base_addr + T.cast(byte_offset, "uint32")


def _pipe_full_addr(smem_base_addr, full_off, count, stages):
    return _shared_addr(smem_base_addr, full_off + _pipe_stage(count, stages) * 8)


def _pipe_empty_addr(smem_base_addr, empty_off, count, stages):
    return _shared_addr(smem_base_addr, empty_off + _pipe_stage(count, stages) * 8)


def _producer_acquire(smem_base_addr, empty_off, count, stages):
    return T.cuda.mbarrier_wait(
        _pipe_empty_addr(smem_base_addr, empty_off, count, stages), _pipe_phase(count, stages, 1)
    )


def _consumer_wait(smem_base_addr, full_off, count, stages):
    return T.cuda.mbarrier_wait(
        _pipe_full_addr(smem_base_addr, full_off, count, stages), _pipe_phase(count, stages, 0)
    )


def _software_commit(smem_base_addr, full_off, count, stages):
    return T.ptx.mbarrier.arrive.shared.b64(
        _pipe_full_addr(smem_base_addr, full_off, count, stages), T.uint32(1)
    )


def _consumer_release(smem_base_addr, empty_off, count, stages):
    return T.ptx.mbarrier.arrive.shared.b64(
        _pipe_empty_addr(smem_base_addr, empty_off, count, stages), T.uint32(1)
    )


def _pipe_next_index(index, stages):
    next_index = index + 1
    return T.Select(next_index == stages, T.int32(0), next_index)


def _pipe_next_phase(index, phase, stages):
    return T.Select(index + 1 == stages, T.bitwise_xor(phase, T.int32(1)), phase)


def _consumer_wait_state(smem_base_addr, full_off, index, phase):
    return T.cuda.mbarrier_wait(_shared_addr(smem_base_addr, full_off + index * 8), phase)


def _producer_acquire_state(smem_base_addr, empty_off, index, phase):
    return T.cuda.mbarrier_wait(_shared_addr(smem_base_addr, empty_off + index * 8), phase)


def _software_commit_state(smem_base_addr, full_off, index):
    return T.ptx.mbarrier.arrive.shared.b64(
        _shared_addr(smem_base_addr, full_off + index * 8), T.uint32(1)
    )


def _consumer_release_state(smem_base_addr, empty_off, index):
    return T.ptx.mbarrier.arrive.shared.b64(
        _shared_addr(smem_base_addr, empty_off + index * 8), T.uint32(1)
    )


@T.inline
def _producer_tail(smem_base_addr, empty_off, count, stages):
    for tail_step in range(stages):
        tail_count: T.int32 = count + tail_step
        _producer_acquire(smem_base_addr, empty_off, tail_count, stages)


@T.inline
def _producer_tail_state(smem_base_addr, empty_off, index, phase, stages):
    tail_index: T.int32 = index
    tail_phase: T.int32 = phase
    for _ in range(stages):
        current_index: T.int32 = tail_index
        current_phase: T.int32 = tail_phase
        _producer_acquire_state(smem_base_addr, empty_off, current_index, current_phase)
        tail_index = _pipe_next_index(current_index, stages)
        tail_phase = _pipe_next_phase(current_index, current_phase, stages)


@T.inline
def _descriptor_copy_payload(src_map, dst_ptr):
    payload = T.decl_buffer(
        (8,), "uint64", data=_byte_ptr(T.address_of(src_map), 0), scope="param", align=16
    )
    T.ptx.st.global_.v4.b64(dst_ptr, payload[0], payload[1], payload[2], payload[3])
    T.ptx.st.global_.v4.b64(_byte_ptr(dst_ptr, 32), payload[4], payload[5], payload[6], payload[7])


def _replace_global_address(desc, address):
    return T.cuda.func_call(
        "gdn_tensormap_replace_global_address",
        desc,
        address,
        source_code=_GDN_SRCS["replace_global_address"],
        return_type="void",
    )


def _replace_global_dim(desc, dim, value):
    return T.cuda.func_call(
        f"gdn_tensormap_replace_global_dim_{dim}",
        desc,
        value,
        source_code=_GDN_SRCS[f"replace_global_dim_{dim}"],
        return_type="void",
    )


def _replace_global_stride(desc, dim, value):
    return T.cuda.func_call(
        f"gdn_tensormap_replace_global_stride_{dim}",
        desc,
        value,
        source_code=_GDN_SRCS[f"replace_global_stride_{dim}"],
        return_type="void",
    )


@T.inline
def _replace_descriptor(desc, address, dim1, dim2, dim3, stride0, stride1, stride2):
    # Frozen TensorMapManager order.  Dimensions/strides not supplied by a
    # specialization are the literal one/zero fields below.
    _replace_global_address(desc, address)
    _replace_global_dim(desc, 0, T.uint32(128))
    _replace_global_dim(desc, 1, T.cast(dim1, "uint32"))
    _replace_global_stride(desc, 0, T.cast(stride0, "uint64"))
    _replace_global_dim(desc, 2, T.cast(dim2, "uint32"))
    _replace_global_stride(desc, 1, T.cast(stride1, "uint64"))
    _replace_global_dim(desc, 3, T.cast(dim3, "uint32"))
    _replace_global_stride(desc, 2, T.cast(stride2, "uint64"))
    _replace_global_dim(desc, 4, T.uint32(1))
    _replace_global_stride(desc, 3, T.uint64(0))


def _tensormap_release():
    return T.cuda.func_call(
        "gdn_tensormap_release", source_code=_GDN_SRCS["tensormap_release"], return_type="void"
    )


def _tensormap_acquire(desc):
    return T.cuda.func_call(
        "gdn_tensormap_acquire",
        desc,
        source_code=_GDN_SRCS["tensormap_acquire"],
        return_type="void",
    )


def _lg2_approx_ftz(value):
    return T.cuda.func_call(
        "gdn_lg2_approx_ftz", value, source_code=_GDN_SRCS["lg2_approx_ftz"], return_type="float32"
    )


def _smem_desc_b128(smem_addr):
    """Position-independent 128B-swizzled matrix descriptor used by CuTe."""
    desc_lo = T.cast(
        T.bitwise_and(T.shift_right(smem_addr, T.uint32(4)), T.uint32(0x3FFF)), "uint64"
    )
    return T.bitwise_or(T.uint64(0x4000404000010000), desc_lo)


def _smem_desc_k_trans_b128(smem_addr):
    """Frozen MN-major K descriptor used only by the (128,128,64) KV MMA."""
    desc_lo = T.cast(
        T.bitwise_and(T.shift_right(smem_addr, T.uint32(4)), T.uint32(0x3FFF)), "uint64"
    )
    return T.bitwise_or(T.uint64(0x4000404002000000), desc_lo + T.uint64(0x02000000))


# tcgen05.mma spelling: kind::f16 from the (float32, float16, float16) dtypes.
# The A operand's dtype picks the syntax line: uint64 is the descriptor (ss)
# form, uint32 the TMEM (ts) form.  disable-output-lane is a mandatory operand
# vector of four zeros at cta_group::1.
_MMA_CHAIN = "tcgen05.mma.cta_group::1.kind::f16"
_MMA_ZERO_MASKS = [T.uint32(0)] * 4


def _mma_commit(barrier):
    return T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
        barrier, pred=T.cuda.elect_sync()
    )


def _b128_swizzle_byte(byte_offset):
    """CuTe ``Swizzle<3,4,3>`` byte address used by every 16-bit SMEM tile."""
    return T.bitwise_xor(
        byte_offset, T.bitwise_and(T.shift_right(byte_offset, T.int32(3)), T.int32(112))
    )


def _b128_ptr(smem_raw, byte_offset):
    return smem_raw.ptr_to([_b128_swizzle_byte(byte_offset)])


def _f16_tile_ptr(smem_raw, base, stage, row, col):
    return _b128_ptr(smem_raw, base + stage * 8192 + row * 128 + col * 2)


def _st_shared_v4_u32(ptr, values, start):
    return T.ptx.st.shared.v4.b32(
        ptr, values[start], values[start + 1], values[start + 2], values[start + 3]
    )


def _ld_shared_v4_u32(ptr, values):
    return T.ptx.ld.shared.v4.b32(values[0], values[1], values[2], values[3], ptr)


def _pack_f16x2(a, b):
    # PTX's two-source packed conversion places its second operand in the
    # low half (CUDA's float2->half2 helper emits the same b,a order).
    packed = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.cvt.rn.f16x2.f32(packed[0], b, a))
    return packed[0]


def _pack_f16x2_cg1(a, b):
    """CuTe CG1 rmem logical pair -> native f16x2 register order."""
    return _pack_f16x2(a, b)


@T.inline
def _sub_zero_pack_f16x2(a, b, dst, dst_index):
    # CK:L3604/L3759/L3918 lowers the fragment negation as one packed
    # sub.f32x2, not two scalar neg.ftz.f32 instructions.
    # Keep the packed result in one RMEM destination.  The value-returning
    # form is an expression, so independently extracting x/y duplicates that
    # expression during CUDA emission (and produces two PTX sub instructions).
    negated: T.uint64[1]
    T.ptx.sub.rn.f32x2(
        negated[0], T.cuda.make_float2(T.float32(0.0), T.float32(0.0)), T.cuda.make_float2(a, b)
    )
    dst[dst_index] = _pack_f16x2(T.cuda.float2_x(negated[0]), T.cuda.float2_y(negated[0]))


def _unpack_f16_lo(word):
    return T.cast(
        T.reinterpret("float16", T.cast(T.bitwise_and(word, T.uint32(0xFFFF)), "uint16")), "float32"
    )


def _unpack_f16_hi(word):
    return T.cast(
        T.reinterpret("float16", T.cast(T.shift_right(word, T.uint32(16)), "uint16")), "float32"
    )


# The ptx mma line always spells the C operand; the legacy "omit C" form fed
# a literal zero per accumulator slot, which is now written at the call site.
_MMA_ZERO_C = [T.float32(0.0)] * 4


def _mma_m16n8k8_f16_zero(acc, a, b):
    return T.ptx["mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"](
        *[acc[i] for i in range(4)], *[a[i] for i in range(2)], b[0], *_MMA_ZERO_C
    )


def _mma_m16n8k16_f16_zero(acc, a, b, acc_off, b_off):
    return T.ptx["mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"](
        *[acc[acc_off + i] for i in range(4)],
        *[a[i] for i in range(4)],
        *[b[b_off + i] for i in range(2)],
        *_MMA_ZERO_C,
    )


def _mma_m16n8k16_f16_acc(acc, a, b, acc_off, b_off):
    return T.ptx["mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"](
        *[acc[acc_off + i] for i in range(4)],
        *[a[i] for i in range(4)],
        *[b[b_off + i] for i in range(2)],
        *[acc[acc_off + i] for i in range(4)],
    )


@T.inline
def _cg0_tmem_ld(tmem_base, stage, thread, values):
    # CK:L2968-L2982.  Ld16x32bx2/Repetition16 is two native x16 loads.
    row_bits: T.int32 = T.bitwise_and(thread << 16, T.int32(0x600000))
    addr: T.int32 = tmem_base + TMEM_CG0_ACC_COL + stage * 64 + row_bits
    # `16` is immHalfSplitoff: the second half-access starts 16 tmem columns
    # after the first (ISA 9.7.17.8.3, "taddr+immHalfSplitoff").
    T.ptx["tcgen05.ld.sync.aligned.16x32bx2.x16.b32"](
        *[values[i] for i in range(16)], T.uint32(addr), 16
    )
    T.ptx["tcgen05.ld.sync.aligned.16x32bx2.x16.b32"](
        *[values[16 + i] for i in range(16)], T.uint32(addr + 32), 16
    )


@T.inline
def _cg0_store_fragment(smem_raw, base, stage, thread, values):
    # CK:L2995-L3014.  One logical row owns [c:c+16] and [c+32:c+48].
    row: T.int32 = T.bitwise_or(
        T.bitwise_and(thread >> 1, T.int32(48)), T.bitwise_and(thread, T.int32(15))
    )
    col: T.int32 = T.bitwise_and(thread, T.int32(16))
    packed: T.uint32[16]
    for pair in T.unroll(16):
        packed[pair] = _pack_f16x2(values[pair * 2], values[pair * 2 + 1])
    _st_shared_v4_u32(_f16_tile_ptr(smem_raw, base, stage, row, col), packed, 0)
    _st_shared_v4_u32(_f16_tile_ptr(smem_raw, base, stage, row, col + 8), packed, 4)
    _st_shared_v4_u32(_f16_tile_ptr(smem_raw, base, stage, row, col + 32), packed, 8)
    _st_shared_v4_u32(_f16_tile_ptr(smem_raw, base, stage, row, col + 40), packed, 12)


@T.inline
def _cg0_load_fragment(smem_raw, base, stage, thread, values):
    row: T.int32 = T.bitwise_or(
        T.bitwise_and(thread >> 1, T.int32(48)), T.bitwise_and(thread, T.int32(15))
    )
    col: T.int32 = T.bitwise_and(thread, T.int32(16))
    packed: T.uint32[16]
    _ld_shared_v4_u32(_f16_tile_ptr(smem_raw, base, stage, row, col), packed)
    loaded1: T.uint32[4]
    loaded2: T.uint32[4]
    loaded3: T.uint32[4]
    _ld_shared_v4_u32(_f16_tile_ptr(smem_raw, base, stage, row, col + 8), loaded1)
    _ld_shared_v4_u32(_f16_tile_ptr(smem_raw, base, stage, row, col + 32), loaded2)
    _ld_shared_v4_u32(_f16_tile_ptr(smem_raw, base, stage, row, col + 40), loaded3)
    for i in T.unroll(4):
        packed[4 + i] = loaded1[i]
        packed[8 + i] = loaded2[i]
        packed[12 + i] = loaded3[i]
    for pair in T.unroll(16):
        values[pair * 2] = _unpack_f16_lo(packed[pair])
        values[pair * 2 + 1] = _unpack_f16_hi(packed[pair])


@T.inline
def _invert_diagonal_8x8(smem_raw, stage, block8, lane):
    # CK:L3400-L3427.  Eight-lane groups each own one complete f16 row.
    row_in_group: T.int32 = lane & 7
    logical_row: T.int32 = block8 + row_in_group
    words: T.uint32[4]
    row_values: T.float32[8]
    _ld_shared_v4_u32(_f16_tile_ptr(smem_raw, SMEM_AINV_OFF, stage, logical_row, block8), words)
    for pair in T.unroll(4):
        row_values[pair * 2] = _unpack_f16_lo(words[pair])
        row_values[pair * 2 + 1] = _unpack_f16_hi(words[pair])
    for i in T.unroll(8):
        if row_in_group == i:
            row_values[i] = T.float32(1.0)
    for src_row in T.unroll(7):
        # CuTe emits exact ``neg.f32`` here.  ``neg.ftz.f32`` cannot be folded
        # into its select/conversion consumers by ptxas.
        row_scale: T.float32
        T.ptx.neg.f32(row_scale, row_values[src_row])
        for i in T.unroll(7):
            if i < src_row:
                pivot: T.float32 = T.cuda._shfl_sync(
                    T.uint32(0xFFFFFFFF), row_values[i], src_row, 8
                )
                if row_in_group > src_row:
                    row_values[i] = row_values[i] + row_scale * pivot
        if row_in_group > src_row:
            row_values[src_row] = row_scale
    for pair in T.unroll(4):
        words[pair] = _pack_f16x2(row_values[pair * 2], row_values[pair * 2 + 1])
    _st_shared_v4_u32(_f16_tile_ptr(smem_raw, SMEM_AINV_OFF, stage, logical_row, block8), words, 0)


@T.inline
def _inverse_8_to_16(smem_raw, stage, block16, lane):
    # CK:L3430-L3620: D*C followed by the accumulator-to-A bridge and *A.
    a: T.uint32[2]
    b: T.uint32[1]
    acc: T.float32[4]
    d_word: T.uint32[1]
    c_word: T.uint32[1]
    T.ptx.ldmatrix.sync.aligned.m8n8.x1.shared.b16(
        d_word[0],
        _f16_tile_ptr(smem_raw, SMEM_AINV_OFF, stage, block16 + 8 + (lane & 7), block16 + 8),
    )
    T.ptx.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16(
        c_word[0], _f16_tile_ptr(smem_raw, SMEM_AINV_OFF, stage, block16 + 8 + (lane & 7), block16)
    )
    a[0] = d_word[0]
    a[1] = d_word[0]
    b[0] = c_word[0]
    _mma_m16n8k8_f16_zero(acc, a, b)
    _sub_zero_pack_f16x2(acc[0], acc[1], a, 0)
    _sub_zero_pack_f16x2(acc[2], acc[3], a, 1)
    T.ptx.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16(
        b[0], _f16_tile_ptr(smem_raw, SMEM_AINV_OFF, stage, block16 + (lane & 7), block16)
    )
    _mma_m16n8k8_f16_zero(acc, a, b)
    d_word[0] = _pack_f16x2(acc[0], acc[1])
    T.ptx.stmatrix.sync.aligned.m8n8.x1.shared.b16(
        _f16_tile_ptr(smem_raw, SMEM_AINV_OFF, stage, block16 + 8 + (lane & 7), block16), d_word[0]
    )


def _ldmatrix_x4_ptr_coords(base_row, base_col, lane, transpose):
    # CK:L3738-L3739/L3897-L3898: the .trans instruction modifier changes
    # each matrix's load semantics, not the lane-group-to-matrix address map.
    # Both forms use g1=bottom-left and g2=top-right in the 16x16 tile.
    lane_matrix: T.int32 = lane >> 3
    row: T.int32 = base_row + (lane & 7) + (lane_matrix & 1) * 8
    col: T.int32 = base_col + (lane_matrix >> 1) * 8
    return row, col


@T.inline
def _ldmatrix_x4_tile(smem_raw, stage, base_row, base_col, lane, transpose, dst):
    row, col = _ldmatrix_x4_ptr_coords(base_row, base_col, lane, transpose)
    T.ptx[f"ldmatrix.sync.aligned.m8n8.x4{'.trans' if transpose else ''}.shared.b16"](
        *[dst[i] for i in range(4)], _f16_tile_ptr(smem_raw, SMEM_AINV_OFF, stage, row, col)
    )


@T.inline
def _stmatrix_x4_tile(smem_raw, stage, base_row, base_col, lane, src):
    # CK:L3772/L3932: O partitions the row-major destination through the
    # C-view.  The instruction itself is non-transpose, and the frozen PTX
    # reuses C's address register because x4 trans/non-trans address maps match.
    row, col = _ldmatrix_x4_ptr_coords(base_row, base_col, lane, False)
    T.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
        _f16_tile_ptr(smem_raw, SMEM_AINV_OFF, stage, row, col), *[src[i] for i in range(4)]
    )


@T.inline
def _inverse_16_to_32(smem_raw, stage, block32, lane):
    # CK:L3622-L3773.  Two n8 MMAs form each 16x16 product.
    a: T.uint32[4]
    b: T.uint32[4]
    acc: T.float32[8]
    packed: T.uint32[4]
    _ldmatrix_x4_tile(smem_raw, stage, block32 + 16, block32 + 16, lane, False, a)
    _ldmatrix_x4_tile(smem_raw, stage, block32 + 16, block32, lane, True, b)
    _mma_m16n8k16_f16_zero(acc, a, b, 0, 0)
    _mma_m16n8k16_f16_zero(acc, a, b, 4, 2)
    for pair in T.unroll(4):
        _sub_zero_pack_f16x2(acc[pair * 2], acc[pair * 2 + 1], a, pair)
    _ldmatrix_x4_tile(smem_raw, stage, block32, block32, lane, True, b)
    _mma_m16n8k16_f16_zero(acc, a, b, 0, 0)
    _mma_m16n8k16_f16_zero(acc, a, b, 4, 2)
    for pair in T.unroll(4):
        packed[pair] = _pack_f16x2(acc[pair * 2], acc[pair * 2 + 1])
    _stmatrix_x4_tile(smem_raw, stage, block32 + 16, block32, lane, packed)


@T.inline
def _inverse_32_to_64(smem_raw, stage, local_warp, lane):
    # CK:L3775-L3932.  Each of two warps owns one 16x32 row slice.
    row_base: T.int32 = 32 + local_warp * 16
    a0: T.uint32[4]
    a1: T.uint32[4]
    b00: T.uint32[4]
    b01: T.uint32[4]
    b10: T.uint32[4]
    b11: T.uint32[4]
    acc: T.float32[16]
    packed_a: T.uint32[8]
    packed_o0: T.uint32[4]
    packed_o1: T.uint32[4]
    _ldmatrix_x4_tile(smem_raw, stage, row_base, 32, lane, False, a0)
    _ldmatrix_x4_tile(smem_raw, stage, row_base, 48, lane, False, a1)
    _ldmatrix_x4_tile(smem_raw, stage, 32, 0, lane, True, b00)
    _ldmatrix_x4_tile(smem_raw, stage, 32, 16, lane, True, b01)
    _ldmatrix_x4_tile(smem_raw, stage, 48, 0, lane, True, b10)
    _ldmatrix_x4_tile(smem_raw, stage, 48, 16, lane, True, b11)
    _mma_m16n8k16_f16_zero(acc, a0, b00, 0, 0)
    _mma_m16n8k16_f16_zero(acc, a0, b00, 4, 2)
    _mma_m16n8k16_f16_zero(acc, a0, b01, 8, 0)
    _mma_m16n8k16_f16_zero(acc, a0, b01, 12, 2)
    _mma_m16n8k16_f16_acc(acc, a1, b10, 0, 0)
    _mma_m16n8k16_f16_acc(acc, a1, b10, 4, 2)
    _mma_m16n8k16_f16_acc(acc, a1, b11, 8, 0)
    _mma_m16n8k16_f16_acc(acc, a1, b11, 12, 2)
    for pair in T.unroll(8):
        _sub_zero_pack_f16x2(acc[pair * 2], acc[pair * 2 + 1], packed_a, pair)

    _ldmatrix_x4_tile(smem_raw, stage, 0, 0, lane, True, b00)
    _ldmatrix_x4_tile(smem_raw, stage, 0, 16, lane, True, b01)
    _ldmatrix_x4_tile(smem_raw, stage, 16, 0, lane, True, b10)
    _ldmatrix_x4_tile(smem_raw, stage, 16, 16, lane, True, b11)
    for i in T.unroll(4):
        a0[i] = packed_a[i]
        a1[i] = packed_a[4 + i]
    _mma_m16n8k16_f16_zero(acc, a0, b00, 0, 0)
    _mma_m16n8k16_f16_zero(acc, a0, b00, 4, 2)
    _mma_m16n8k16_f16_zero(acc, a0, b01, 8, 0)
    _mma_m16n8k16_f16_zero(acc, a0, b01, 12, 2)
    _mma_m16n8k16_f16_acc(acc, a1, b10, 0, 0)
    _mma_m16n8k16_f16_acc(acc, a1, b10, 4, 2)
    _mma_m16n8k16_f16_acc(acc, a1, b11, 8, 0)
    _mma_m16n8k16_f16_acc(acc, a1, b11, 12, 2)
    for pair in T.unroll(4):
        packed_o0[pair] = _pack_f16x2(acc[pair * 2], acc[pair * 2 + 1])
        packed_o1[pair] = _pack_f16x2(acc[8 + pair * 2], acc[8 + pair * 2 + 1])
    T.ptx.bar.sync(T.uint32(INVERSE_BARRIER), T.uint32(128))
    _stmatrix_x4_tile(smem_raw, stage, row_base, 0, lane, packed_o0)
    _stmatrix_x4_tile(smem_raw, stage, row_base, 16, lane, packed_o1)


def _cg1_tmem_row_bits(thread):
    return T.bitwise_and(thread << 16, T.int32(0x600000))


@T.inline
def _state_tmem_ld_sub(tmem_base, thread, sub, values, value_offset):
    # CK:L4073-L4077/L4424-L4428: Ld32x32b, repetition 32.
    addr: T.int32 = tmem_base + _cg1_tmem_row_bits(thread) + sub * 32
    T.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
        *[values[value_offset + i] for i in range(32)], T.cast(addr, "uint32")
    )


@T.inline
def _state_tmem_st_sub(tmem_base, thread, sub, values, value_offset):
    # CK:L3997-L4007/L4472-L4476: St32x32b, repetition 32.
    addr: T.int32 = tmem_base + _cg1_tmem_row_bits(thread) + sub * 32
    T.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
        T.cast(addr, "uint32"), *[values[value_offset + i] for i in range(32)]
    )


@T.inline
def _state_input_tmem_st_sub(tmem_base, thread, sub, values):
    # CK:L4432-L4436: four St32x32b.x16 stores of packed f16 pairs.
    addr: T.int32 = tmem_base + _cg1_tmem_row_bits(thread) + TMEM_STATE_INPUT_COL + sub * 16
    T.ptx["tcgen05.st.sync.aligned.32x32b.x16.b32"](
        T.cast(addr, "uint32"), *[values[sub * 16 + i] for i in range(16)]
    )


@T.inline
def _cg1_tmem_ld_f32(tmem_base, column, thread, values):
    # CK:L4275-L4292/L4346-L4361: two row halves, each Ld16x256b.x8.
    addr: T.int32 = tmem_base + _cg1_tmem_row_bits(thread) + column
    T.ptx["tcgen05.ld.sync.aligned.16x256b.x8.b32"](
        *[values[i] for i in range(32)], T.cast(addr, "uint32")
    )
    T.ptx["tcgen05.ld.sync.aligned.16x256b.x8.b32"](
        *[values[32 + i] for i in range(32)], T.cast(addr + T.int32(0x100000), "uint32")
    )


@T.inline
def _cg1_tmem_st_f32(tmem_base, column, thread, values):
    # CK:L4349-L4355: the QS accumulator is overwritten in its original layout.
    addr: T.int32 = tmem_base + _cg1_tmem_row_bits(thread) + column
    T.ptx["tcgen05.st.sync.aligned.16x256b.x8.b32"](
        T.cast(addr, "uint32"), *[values[i] for i in range(32)]
    )
    T.ptx["tcgen05.st.sync.aligned.16x256b.x8.b32"](
        T.cast(addr + T.int32(0x100000), "uint32"), *[values[32 + i] for i in range(32)]
    )


@T.inline
def _cg1_tmem_st_f16_half(tmem_base, column, thread, half, values):
    # CK:L4312-L4329: one St16x128b.x8 row half of packed f16x2.
    addr: T.int32 = tmem_base + _cg1_tmem_row_bits(thread) + column
    T.ptx["tcgen05.st.sync.aligned.16x128b.x8.b32"](
        T.cast(addr + half * T.int32(0x100000), "uint32"),
        *[values[half * 16 + i] for i in range(16)],
    )


@T.inline
def _cg1_tmem_st_f16(tmem_base, column, thread, values):
    _cg1_tmem_st_f16_half(tmem_base, column, thread, 0, values)
    _cg1_tmem_st_f16_half(tmem_base, column, thread, 1, values)


def _cg1_smem_lane_byte(thread):
    # Frozen transform_partitioned_tensor_layout address algebra (PTX r118).
    a: T.int32 = T.bitwise_and(thread << 6, T.int32(448))
    b: T.int32 = T.bitwise_and(thread, T.int32(40))
    c: T.int32 = T.bitwise_or(b, a)
    d: T.int32 = T.bitwise_and(thread << 5, T.int32(512))
    e: T.int32 = T.bitwise_and(thread << 6, T.int32(4096))
    f: T.int32 = T.bitwise_or(d, e)
    g: T.int32 = T.bitwise_xor(a >> 3, c)
    return T.bitwise_or(f, g) << 1


def _cg1_smem_second_half_delta(thread):
    c: T.int32 = T.bitwise_or(
        T.bitwise_and(thread, T.int32(40)), T.bitwise_and(thread << 6, T.int32(448))
    )
    return T.if_then_else(T.bitwise_and(c, T.int32(128)) == 0, T.int32(32), T.int32(-32))


@T.inline
def _load_v_fragment(smem_raw, stage, thread, values):
    # CK:L4363-L4372/L4506-L4516: eight x4-trans loads in physical order.
    lane_byte: T.int32 = _cg1_smem_lane_byte(thread)
    stage_byte: T.int32 = stage * 16384
    for band in range(4):
        T.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
            *[values[band * 4 + i] for i in range(4)],
            smem_raw.ptr_to([SMEM_V_OFF + stage_byte + lane_byte + band * 2048]),
        )
    lane_byte = lane_byte + _cg1_smem_second_half_delta(thread)
    for band in range(4):
        T.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
            *[values[16 + band * 4 + i] for i in range(4)],
            smem_raw.ptr_to([SMEM_V_OFF + stage_byte + lane_byte + band * 2048]),
        )


@T.inline
def _store_o_fragment(smem_raw, stage, thread, values):
    # CK:L4374-L4387/L4601-L4619: the matching eight x4-trans stores.
    lane_byte: T.int32 = _cg1_smem_lane_byte(thread)
    stage_byte: T.int32 = stage * 16384
    for band in range(4):
        T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
            smem_raw.ptr_to([SMEM_O_OFF + stage_byte + lane_byte + band * 2048]),
            *[values[band * 4 + i] for i in range(4)],
        )
    lane_byte = lane_byte + _cg1_smem_second_half_delta(thread)
    for band in range(4):
        T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
            smem_raw.ptr_to([SMEM_O_OFF + stage_byte + lane_byte + band * 2048]),
            *[values[16 + band * 4 + i] for i in range(4)],
        )


@T.inline
def _load_initial_state(
    smem_base_addr, initial_state, batch, output_head, tmem_base, thread, kv_count, *, HV
):
    # CK:L3935-L4017.  Each CG1 thread owns one value row and 128 contiguous keys.
    _producer_acquire(smem_base_addr, 328, kv_count, 1)
    cg1_thread: T.int64 = T.cast(T.bitwise_and(thread, T.int32(127)), "int64")
    state_base: T.int64 = (
        T.cast(batch, "int64") * T.cast(HV, "int64") + T.cast(output_head, "int64")
    ) * T.int64(D_HEAD * D_HEAD) + cg1_thread * T.int64(D_HEAD)
    state_sub: T.uint32[32]
    for sub in range(4):
        for vector in range(8):
            T.ptx["ld.global.L1::no_allocate.v4.b32"](
                state_sub[vector * 4],
                state_sub[vector * 4 + 1],
                state_sub[vector * 4 + 2],
                state_sub[vector * 4 + 3],
                initial_state.ptr_to([state_base + sub * 32 + vector * 4]),
            )
        _state_tmem_st_sub(tmem_base, thread, sub, state_sub, 0)
    T.ptx.tcgen05.wait__st.sync.aligned()
    T.ptx.bar.sync(T.uint32(INITIAL_STATE_BARRIER), T.uint32(128))
    if T.bitwise_and(thread, T.int32(127)) == 0:
        _software_commit(smem_base_addr, 320, kv_count, 1)


@T.inline
def _copy_empty_state(initial_state, final_state, batch, output_head, thread, *, HV):
    # CK:L4019-L4052.  The source deliberately keeps this 128-iteration loop rolled.
    cg1_thread: T.int64 = T.cast(T.bitwise_and(thread, T.int32(127)), "int64")
    state_base: T.int64 = (
        T.cast(batch, "int64") * T.cast(HV, "int64") + T.cast(output_head, "int64")
    ) * T.int64(D_HEAD * D_HEAD) + cg1_thread * T.int64(D_HEAD)
    for key in T.serial(0, D_HEAD):
        final_state[state_base + key] = initial_state[state_base + key]


@T.inline
def _store_final_state(
    smem_base_addr, final_state, batch, output_head, tmem_base, thread, kv_count, *, HV
):
    # CK:L4054-L4146.  Each x32 TMEM read is immediately drained by eight v4 stores.
    _consumer_wait(smem_base_addr, 320, kv_count, 1)
    cg1_thread: T.int64 = T.cast(T.bitwise_and(thread, T.int32(127)), "int64")
    state_base: T.int64 = (
        T.cast(batch, "int64") * T.cast(HV, "int64") + T.cast(output_head, "int64")
    ) * T.int64(D_HEAD * D_HEAD) + cg1_thread * T.int64(D_HEAD)
    state_sub: T.uint32[32]
    for sub in range(4):
        _state_tmem_ld_sub(tmem_base, thread, sub, state_sub, 0)
        for vector in range(8):
            T.ptx["st.global.L1::no_allocate.v4.b32"](
                final_state.ptr_to([state_base + sub * 32 + vector * 4]),
                state_sub[vector * 4],
                state_sub[vector * 4 + 1],
                state_sub[vector * 4 + 2],
                state_sub[vector * 4 + 3],
            )
    _consumer_release(smem_base_addr, 328, kv_count, 1)


@T.inline
def _mma_ss_64x64_k128(tmem_d, a_desc_base, b_desc_base, full_barrier):
    # CK:L2163-L2211.  B128 K-major phase offsets are in 16-byte units.
    for kphase in range(8):
        phase_off: T.uint64 = T.uint64((kphase % 4) * 2 + (kphase // 4) * 512)
        T.evaluate(
            T.ptx[_MMA_CHAIN](
                T.cast(tmem_d, "uint32"),
                a_desc_base + phase_off,
                b_desc_base + phase_off,
                T.uint32(0x04100010),
                *_MMA_ZERO_MASKS,
                T.ptx.pred(0 if kphase == 0 else 1),
                pred=T.cuda.elect_sync(),
            )
        )
    _mma_commit(full_barrier)


@T.inline
def _mma_ts_128x64_k128(tmem_d, tmem_a, b_desc_base, full_barrier):
    # CK:L2322-L2344.  A advances eight TMEM columns per 16-wide K phase.
    for kphase in range(8):
        phase_off: T.uint64 = T.uint64((kphase % 4) * 2 + (kphase // 4) * 512)
        T.evaluate(
            T.ptx[_MMA_CHAIN](
                T.cast(tmem_d, "uint32"),
                T.cast(tmem_a + kphase * 8, "uint32"),
                b_desc_base + phase_off,
                T.uint32(0x08100010),
                *_MMA_ZERO_MASKS,
                T.ptx.pred(0 if kphase == 0 else 1),
                pred=T.cuda.elect_sync(),
            )
        )
    _mma_commit(full_barrier)


@T.inline
def _mma_ts_128x64_k64(tmem_d, tmem_a, b_desc_base, accumulate):
    # CK:L2352-L2389.  NV overwrites on phase zero; QKV always accumulates QS.
    for kphase in range(4):
        phase_off: T.uint64 = T.uint64(kphase * 2)
        T.evaluate(
            T.ptx[_MMA_CHAIN](
                T.cast(tmem_d, "uint32"),
                T.cast(tmem_a + kphase * 8, "uint32"),
                b_desc_base + phase_off,
                T.uint32(0x08100010),
                *_MMA_ZERO_MASKS,
                T.ptx.pred(accumulate if kphase == 0 else 1),
                pred=T.cuda.elect_sync(),
            )
        )


@T.inline
def _mma_ts_128x128_k64(tmem_d, tmem_a, b_desc_base, full_barrier):
    # CK:L2396-L2406.  K is the only MN-major B operand; recurrent state is D.
    for kphase in range(4):
        T.evaluate(
            T.ptx[_MMA_CHAIN](
                T.cast(tmem_d, "uint32"),
                T.cast(tmem_a + kphase * 8, "uint32"),
                b_desc_base + T.uint64(kphase * 128),
                T.uint32(0x08210010),
                *_MMA_ZERO_MASKS,
                True,
                pred=T.cuda.elect_sync(),
            )
        )
    _mma_commit(full_barrier)


# TMA spellings, one per tensor rank.  The frozen sites pass cta_group=1 and an
# explicit (zero) cache policy, so both modifiers are written and the policy is
# a real trailing operand.
_TMA_G2S_CHAIN = {
    dim: (
        f"cp.async.bulk.tensor.{dim}d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
    )
    for dim in (3, 4)
}
_TMA_S2G_CHAIN = {
    dim: f"cp.async.bulk.tensor.{dim}d.global.shared::cta.tile.bulk_group.L2::cache_hint"
    for dim in (3, 4)
}


@T.inline
def _tma_qk_g2s(smem_base_addr, smem_off, descriptor, full_barrier, token, qk_head):
    # CK:L1835-L1908.  The 64x128 f16 tile is two 64x64 TMA boxes sharing
    # one 16384-byte transaction barrier.
    for d_coord in range(0, 128, 64):
        T.ptx[_TMA_G2S_CHAIN[3]](
            _shared_addr(smem_base_addr, smem_off + d_coord * 128),
            descriptor,
            T.int32(d_coord),
            T.cast(token, "int32"),
            qk_head,
            full_barrier,
            T.uint64(0),
        )


@T.inline
def _tma_v_g2s(
    smem_base_addr,
    smem_off,
    descriptor,
    full_barrier,
    token,
    output_head,
    qk_head,
    value_subhead,
    *,
    HQ,
    HV,
):
    # CK:L1910-L1976.  V is staged as logical [128,64]; the first TensorMap
    # coordinate is the value dimension and therefore advances by 64.
    for d_coord in range(0, 128, 64):
        if HQ == HV:
            T.ptx[_TMA_G2S_CHAIN[3]](
                _shared_addr(smem_base_addr, smem_off + d_coord * 128),
                descriptor,
                T.int32(d_coord),
                T.cast(token, "int32"),
                output_head,
                full_barrier,
                T.uint64(0),
            )
        else:
            T.ptx[_TMA_G2S_CHAIN[4]](
                _shared_addr(smem_base_addr, smem_off + d_coord * 128),
                descriptor,
                T.int32(d_coord),
                T.cast(token, "int32"),
                value_subhead,
                qk_head,
                full_barrier,
                T.uint64(0),
            )


@T.inline
def _load_qkv_chunk(
    smem_base_addr,
    descriptor_q,
    descriptor_k,
    descriptor_v,
    chunk_offset,
    chunk_idx,
    output_head,
    qk_head,
    value_subhead,
    q_count,
    q_phase,
    k_count,
    k_phase,
    v_count,
    v_phase,
    *,
    HQ,
    HV,
):
    # Approved sketch node load_qkv_chunk, in its literal K -> Q -> V order.
    _producer_acquire_state(smem_base_addr, 32, k_count, k_phase)
    k_full = _shared_addr(smem_base_addr, k_count * 8)
    if chunk_idx == 0:
        if T.cuda.elect_sync():
            _tensormap_acquire(descriptor_k)
    if T.cuda.elect_sync():
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(k_full, T.uint32(16384))
        _tma_qk_g2s(
            smem_base_addr,
            SMEM_K_OFF + k_count * 16384,
            descriptor_k,
            k_full,
            chunk_offset,
            qk_head,
        )

    _producer_acquire_state(smem_base_addr, 80, q_count, q_phase)
    q_full = _shared_addr(smem_base_addr, 64 + q_count * 8)
    if chunk_idx == 0:
        if T.cuda.elect_sync():
            _tensormap_acquire(descriptor_q)
    if T.cuda.elect_sync():
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(q_full, T.uint32(16384))
        _tma_qk_g2s(
            smem_base_addr,
            SMEM_Q_OFF + q_count * 16384,
            descriptor_q,
            q_full,
            chunk_offset,
            qk_head,
        )

    _producer_acquire_state(smem_base_addr, 120, v_count, v_phase)
    v_full = _shared_addr(smem_base_addr, 96 + v_count * 8)
    if chunk_idx == 0:
        if T.cuda.elect_sync():
            _tensormap_acquire(descriptor_v)
    if T.cuda.elect_sync():
        T.ptx.mbarrier.arrive.expect_tx.shared.b64(v_full, T.uint32(16384))
        _tma_v_g2s(
            smem_base_addr,
            SMEM_V_OFF + v_count * 16384,
            descriptor_v,
            v_full,
            chunk_offset,
            output_head,
            qk_head,
            value_subhead,
            HQ=HQ,
            HV=HV,
        )


@T.inline
def _load_gate_beta_chunk(
    smem_raw,
    smem_base_addr,
    s_cumsumlog,
    s_cumprod,
    s_beta,
    gate,
    beta,
    chunk_offset,
    output_head,
    is_last_tile,
    batch_end,
    lane,
    gate_count,
    gate_phase,
    beta_count,
    beta_phase,
    *,
    HV,
):
    # CK:L1979-L2124.  Both last-tile comparisons are intentionally confined
    # to the last-tile branch, matching the frozen predicate lifetime.
    pos0: T.int64 = chunk_offset + T.cast(lane, "int64")
    pos1: T.int64 = chunk_offset + T.cast(lane + 32, "int64")
    valid0: T.int32 = 1
    valid1: T.int32 = 1
    gate0: T.float32 = T.float32(1.0)
    gate1: T.float32 = T.float32(1.0)
    if is_last_tile:
        valid0 = T.cast(pos0 < batch_end, "int32")
        valid1 = T.cast(pos1 < batch_end, "int32")
        if valid0 != 0:
            gate0 = gate[pos0 * T.cast(HV, "int64") + T.cast(output_head, "int64")]
        if valid1 != 0:
            gate1 = gate[pos1 * T.cast(HV, "int64") + T.cast(output_head, "int64")]
    else:
        gate0 = gate[pos0 * T.cast(HV, "int64") + T.cast(output_head, "int64")]
        gate1 = gate[pos1 * T.cast(HV, "int64") + T.cast(output_head, "int64")]

    gate0 = _lg2_approx_ftz(gate0 + T.float32(1.0e-10))
    gate1 = _lg2_approx_ftz(gate1 + T.float32(1.0e-10))
    for scan_step in T.unroll(0, 5):
        scan_offset: T.int32 = 1 << scan_step
        prior0: T.float32 = T.tvm_warp_shuffle_up(T.uint32(0xFFFFFFFF), gate0, scan_offset, 32, 32)
        prior1: T.float32 = T.tvm_warp_shuffle_up(T.uint32(0xFFFFFFFF), gate1, scan_offset, 32, 32)
        if lane >= scan_offset:
            gate0 = gate0 + prior0
            gate1 = gate1 + prior1
    gate1 = gate1 + T.cuda._shfl_sync(T.uint32(0xFFFFFFFF), gate0, 31, 32)
    cumprod0: T.float32
    cumprod1: T.float32
    T.ptx.ex2.approx.ftz.f32(cumprod0, gate0)
    T.ptx.ex2.approx.ftz.f32(cumprod1, gate1)

    # The register scan precedes stage acquisition in the frozen source.
    _producer_acquire_state(smem_base_addr, 184, gate_count, gate_phase)
    gate_stage: T.int32 = gate_count
    s_cumsumlog[gate_stage * 64 + lane] = gate0
    s_cumsumlog[gate_stage * 64 + lane + 32] = gate1
    s_cumprod[gate_stage * 64 + lane] = cumprod0
    s_cumprod[gate_stage * 64 + lane + 32] = cumprod1
    _software_commit_state(smem_base_addr, 144, gate_count)

    _producer_acquire_state(smem_base_addr, 264, beta_count, beta_phase)
    beta_stage: T.int32 = beta_count
    beta_dst0: T.uint32 = _shared_addr(smem_base_addr, SMEM_BETA_OFF + (beta_stage * 64 + lane) * 4)
    beta_dst1: T.uint32 = _shared_addr(
        smem_base_addr, SMEM_BETA_OFF + (beta_stage * 64 + lane + 32) * 4
    )
    if is_last_tile:
        s_beta[beta_stage * 64 + lane] = T.float32(0.0)
        s_beta[beta_stage * 64 + lane + 32] = T.float32(0.0)
        T.ptx["cp.async.ca.shared.global"](
            beta_dst0,
            beta.ptr_to([pos0 * T.cast(HV, "int64") + T.cast(output_head, "int64")]),
            4,
            pred=valid0,
        )
        T.ptx["cp.async.ca.shared.global"](
            beta_dst1,
            beta.ptr_to([pos1 * T.cast(HV, "int64") + T.cast(output_head, "int64")]),
            4,
            pred=valid1,
        )
    else:
        T.ptx["cp.async.ca.shared.global"](
            beta_dst0, beta.ptr_to([pos0 * T.cast(HV, "int64") + T.cast(output_head, "int64")]), 4
        )
        T.ptx["cp.async.ca.shared.global"](
            beta_dst1, beta.ptr_to([pos1 * T.cast(HV, "int64") + T.cast(output_head, "int64")]), 4
        )
    T.ptx.cp.async_.mbarrier.arrive.noinc.shared__cta.b64(
        _shared_addr(smem_base_addr, 224 + beta_count * 8)
    )


@T.inline
def _store_o_chunk(
    smem_base_addr,
    descriptor_o,
    chunk_offset,
    output_head,
    qk_head,
    value_subhead,
    o_index,
    o_phase,
    *,
    HQ,
    HV,
):
    # CK:L4637-L4700.  One consumer wait covers the two 64x64 S2G boxes.
    _consumer_wait_state(smem_base_addr, 528, o_index, o_phase)
    stage: T.int32 = o_index
    if T.cuda.elect_sync():
        for d_coord in range(0, 128, 64):
            if HQ == HV:
                T.ptx[_TMA_S2G_CHAIN[3]](
                    descriptor_o,
                    T.int32(d_coord),
                    T.cast(chunk_offset, "int32"),
                    output_head,
                    _shared_addr(smem_base_addr, SMEM_O_OFF + stage * 16384 + d_coord * 128),
                    T.uint64(0),
                )
            else:
                T.ptx[_TMA_S2G_CHAIN[4]](
                    descriptor_o,
                    T.int32(d_coord),
                    T.cast(chunk_offset, "int32"),
                    value_subhead,
                    qk_head,
                    _shared_addr(smem_base_addr, SMEM_O_OFF + stage * 16384 + d_coord * 128),
                    T.uint64(0),
                )
        T.ptx.cp.async_.bulk.commit_group()
        # legacy T.ptx.cp_async.bulk.wait_group defaulted read=True: wait only
        # until the bulk ops have finished READING their source, which is all
        # the O epilogue needs before recycling the stage.
        T.ptx.cp.async_.bulk.wait_group.read(0)
    _consumer_release_state(smem_base_addr, 544, o_index)


@T.jit
def _kernel(
    q_h: T.handle,
    k_h: T.handle,
    v_h: T.handle,
    gate_h: T.handle,
    beta_h: T.handle,
    o_h: T.handle,
    cu_seqlens_h: T.handle,
    initial_state_h: T.handle,
    final_state_h: T.handle,
    q_map: T.TensorMap(),
    k_map: T.TensorMap(),
    v_map: T.TensorMap(),
    o_map: T.TensorMap(),
    descriptor_workspace_h: T.handle,
    total_tokens: T.int64,
    num_sequences: T.int32,
    num_sms: T.int32,
    scale: T.float32,
    *,
    HQ: T.constexpr,
    HV: T.constexpr,
):
    q = T.match_buffer(q_h, (total_tokens * HQ * D_HEAD,), "float16", scope="global")
    k = T.match_buffer(k_h, (total_tokens * HQ * D_HEAD,), "float16", scope="global")
    v = T.match_buffer(v_h, (total_tokens * HV * D_HEAD,), "float16", scope="global")
    gate = T.match_buffer(gate_h, (total_tokens * HV,), "float32", scope="global")
    beta = T.match_buffer(beta_h, (total_tokens * HV,), "float32", scope="global")
    o = T.match_buffer(o_h, (total_tokens * HV * D_HEAD,), "float16", scope="global")
    cu_seqlens = T.match_buffer(cu_seqlens_h, (num_sequences + 1,), "int32", scope="global")
    initial_state = T.match_buffer(
        initial_state_h,
        (T.cast(num_sequences, "int64") * HV * D_HEAD * D_HEAD,),
        "float32",
        scope="global",
    )
    final_state = T.match_buffer(
        final_state_h,
        (T.cast(num_sequences, "int64") * HV * D_HEAD * D_HEAD,),
        "float32",
        scope="global",
    )
    descriptor_workspace = T.match_buffer(
        descriptor_workspace_h, (num_sms * DESCRIPTOR_BYTES_PER_CTA,), "int8", scope="global"
    )

    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})

    # CUDA TRANSCRIPTION START
    # sketch: static ABI/launch -> CK:L895-L1003.
    block = T.cta_id([T.min(num_sequences * HV, num_sms)])
    thread = T.thread_id([THREADS])
    output_heads: T.let = HV
    total_work: T.let = num_sequences * output_heads
    grid_x: T.let = T.min(total_work, num_sms)
    warp = _make_warp_uniform(T.cast(thread // 32, "uint32"))
    lane: T.let = thread % 32

    # sketch: exact shared backing views and aliases -> CK:L1005-L1044.
    pool = T.SMEMPool()
    smem_raw = pool.alloc((SMEM_TOTAL,), "uint8", align=1024)
    tmem_holding = T.decl_buffer(
        (1,),
        "int32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_TMEM_HOLDING_OFF,
        align=4,
    )
    s_q = T.decl_buffer(
        (64 * 128 * 2,),
        "float16",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_Q_OFF,
        align=1024,
    )
    s_k = T.decl_buffer(
        (64 * 128 * 4,),
        "float16",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_K_OFF,
        align=1024,
    )
    s_v = T.decl_buffer(
        (128 * 64 * 3,),
        "float16",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_V_OFF,
        align=1024,
    )
    s_ainv = T.decl_buffer(
        (64 * 64 * 3,),
        "float16",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_AINV_OFF,
        align=1024,
    )
    s_qk = T.decl_buffer(
        (64 * 64 * 2,),
        "float16",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_QK_OFF,
        align=1024,
    )
    s_o = T.decl_buffer(
        (128 * 64 * 2,),
        "float16",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_O_OFF,
        align=1024,
    )
    s_cumsumlog = T.decl_buffer(
        (64 * 5,),
        "float32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_CUMSUMLOG_OFF,
        align=4,
    )
    s_cumprod = T.decl_buffer(
        (64 * 5,),
        "float32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_CUMPROD_OFF,
        align=4,
    )
    s_beta = T.decl_buffer(
        (64 * 5,),
        "float32",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=SMEM_BETA_OFF,
        align=4,
    )
    pool.commit()

    # Four 128-byte descriptor slots are CTA-private within the runtime workspace.
    descriptor_cta_base: T.int64 = T.cast(block, "int64") * DESCRIPTOR_BYTES_PER_CTA
    descriptor_q = descriptor_workspace.ptr_to([descriptor_cta_base + 0])
    descriptor_k = descriptor_workspace.ptr_to([descriptor_cta_base + 128])
    descriptor_v = descriptor_workspace.ptr_to([descriptor_cta_base + 256])
    descriptor_o = descriptor_workspace.ptr_to([descriptor_cta_base + 384])

    # sketch: construct all sixteen rings before role selection -> CK:L1060-L1238.
    if warp == 0:
        if T.cuda.elect_sync():
            _init_all_pipelines(smem_raw)
            T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cta_sync()

    # sketch: descriptor prefetch hints belong to warp 8 -> CK:L1046-L1050.
    if warp == 8:
        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(q_map)))
        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(k_map)))
        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(v_map)))
        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(o_map)))

    # sketch: original independent CG0 if, followed by the CG1/issuer/load elif chain.
    if warp <= 3:
        smem_addr_cg0: T.uint32 = T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0]))
        T.ptx.setmaxnreg.inc.sync.aligned.u32(224)
        T.ptx.bar.sync(T.uint32(TMEM_ALLOC_BARRIER), T.uint32(320))
        tmem_base_cg0: T.int32
        T.ptx.ld.volatile.shared.s32(tmem_base_cg0, T.address_of(tmem_holding[0]))
        gate_consumer_index_cg0: T.int32 = 0
        gate_consumer_phase_cg0: T.int32 = 0
        beta_consumer_index_cg0: T.int32 = 0
        beta_consumer_phase_cg0: T.int32 = 0
        shared_consumer_index_cg0: T.int32 = 0
        shared_consumer_phase_cg0: T.int32 = 0
        ainv_producer_index_cg0: T.int32 = 0
        ainv_producer_phase_cg0: T.int32 = 1
        qk_producer_index_cg0: T.int32 = 0
        qk_producer_phase_cg0: T.int32 = 1
        work_linear_cg0: T.int32 = block
        while work_linear_cg0 < total_work:
            work_u32_cg0: T.uint32 = T.cast(work_linear_cg0, "uint32")
            remain_u32_cg0: T.uint32 = _udiv_u32_const(work_u32_cg0, HV)
            head_cg0: T.int32 = T.cast(
                work_u32_cg0 - remain_u32_cg0 * T.uint32(output_heads), "int32"
            )
            remain_cg0: T.int32 = T.cast(remain_u32_cg0, "int32")
            # The valid-work guard proves remain_cg0 < num_sequences.  This is
            # the frozen fast-divmod remainder with its quotient known zero;
            # spelling another runtime modulus makes NVRTC emit rem.s32,
            # which is absent from the frozen PTX.
            batch_cg0: T.int32 = remain_cg0
            batch_start_cg0: T.int64 = T.cast(cu_seqlens[batch_cg0], "int64")
            batch_end_cg0: T.int64 = T.cast(cu_seqlens[batch_cg0 + 1], "int64")
            seqlen_cg0: T.int32 = T.cast(batch_end_cg0 - batch_start_cg0, "int32")
            num_pairs_cg0: T.int32 = (seqlen_cg0 + PAIR_TOKENS - 1) // PAIR_TOKENS

            # sketch compute_group_0_pair -> CK:L2917-L3352.  The local arrays
            # are the literal Ld16x32bx2/Repetition16 fragments from the source.
            for pair_cg0 in T.serial(0, num_pairs_cg0):
                row_cg0: T.int32 = T.bitwise_or(
                    T.bitwise_and(thread >> 1, T.int32(48)), T.bitwise_and(thread, T.int32(15))
                )
                col_base_cg0: T.int32 = T.bitwise_and(thread, T.int32(16))
                transfer0_cg0: T.float32[32]
                transfer1_cg0: T.float32[32]

                gate0_count_cg0: T.int32 = gate_consumer_index_cg0
                gate0_phase_cg0: T.int32 = gate_consumer_phase_cg0
                _consumer_wait_state(smem_addr_cg0, 144, gate0_count_cg0, gate0_phase_cg0)
                gate_consumer_index_cg0 = _pipe_next_index(gate0_count_cg0, 5)
                gate_consumer_phase_cg0 = _pipe_next_phase(gate0_count_cg0, gate0_phase_cg0, 5)
                gate1_count_cg0: T.int32 = gate_consumer_index_cg0
                gate1_phase_cg0: T.int32 = gate_consumer_phase_cg0
                _consumer_wait_state(smem_addr_cg0, 144, gate1_count_cg0, gate1_phase_cg0)
                gate_consumer_index_cg0 = _pipe_next_index(gate1_count_cg0, 5)
                gate_consumer_phase_cg0 = _pipe_next_phase(gate1_count_cg0, gate1_phase_cg0, 5)
                gate0_stage_cg0: T.int32 = gate0_count_cg0
                gate1_stage_cg0: T.int32 = gate1_count_cg0
                row_log0_cg0: T.float32 = s_cumsumlog[gate0_stage_cg0 * 64 + row_cg0]
                row_log1_cg0: T.float32 = s_cumsumlog[gate1_stage_cg0 * 64 + row_cg0]
                for frag_idx_cg0 in T.unroll(32):
                    col_cg0: T.int32 = col_base_cg0 + frag_idx_cg0
                    if frag_idx_cg0 >= 16:
                        col_cg0 = col_cg0 + 16
                    # The frozen source guards the LDS + MUFU behind the Select's
                    # short-circuit; ex2's DPS shape would otherwise hoist both
                    # out unconditionally, so keep them under the same guard.
                    transfer_exp0_cg0: T.float32 = T.float32(0.0)
                    transfer_exp1_cg0: T.float32 = T.float32(0.0)
                    if row_cg0 >= col_cg0:
                        T.ptx.ex2.approx.ftz.f32(
                            transfer_exp0_cg0,
                            row_log0_cg0 - s_cumsumlog[gate0_stage_cg0 * 64 + col_cg0],
                        )
                        T.ptx.ex2.approx.ftz.f32(
                            transfer_exp1_cg0,
                            row_log1_cg0 - s_cumsumlog[gate1_stage_cg0 * 64 + col_cg0],
                        )
                    transfer0_cg0[frag_idx_cg0] = transfer_exp0_cg0
                    transfer1_cg0[frag_idx_cg0] = transfer_exp1_cg0
                _consumer_release_state(smem_addr_cg0, 184, gate0_count_cg0)
                _consumer_release_state(smem_addr_cg0, 184, gate1_count_cg0)

                beta0_count_cg0: T.int32 = beta_consumer_index_cg0
                beta0_phase_cg0: T.int32 = beta_consumer_phase_cg0
                _consumer_wait_state(smem_addr_cg0, 224, beta0_count_cg0, beta0_phase_cg0)
                beta_consumer_index_cg0 = _pipe_next_index(beta0_count_cg0, 5)
                beta_consumer_phase_cg0 = _pipe_next_phase(beta0_count_cg0, beta0_phase_cg0, 5)
                beta1_count_cg0: T.int32 = beta_consumer_index_cg0
                beta1_phase_cg0: T.int32 = beta_consumer_phase_cg0
                _consumer_wait_state(smem_addr_cg0, 224, beta1_count_cg0, beta1_phase_cg0)
                beta_consumer_index_cg0 = _pipe_next_index(beta1_count_cg0, 5)
                beta_consumer_phase_cg0 = _pipe_next_phase(beta1_count_cg0, beta1_phase_cg0, 5)
                beta0_cg0: T.float32 = s_beta[beta0_count_cg0 * 64 + row_cg0]
                beta1_cg0: T.float32 = s_beta[beta1_count_cg0 * 64 + row_cg0]

                # KK0 seed: acquire Ainv, consume accumulator, then multiply
                # packed f32x2 by transfer and the beta ROW before narrowing.
                ainv0_count_cg0: T.int32 = ainv_producer_index_cg0
                ainv0_phase_cg0: T.int32 = ainv_producer_phase_cg0
                _producer_acquire_state(smem_addr_cg0, 408, ainv0_count_cg0, ainv0_phase_cg0)
                ainv_producer_index_cg0 = _pipe_next_index(ainv0_count_cg0, 3)
                ainv_producer_phase_cg0 = _pipe_next_phase(ainv0_count_cg0, ainv0_phase_cg0, 3)
                kk0_count_cg0: T.int32 = shared_consumer_index_cg0
                kk0_phase_cg0: T.int32 = shared_consumer_phase_cg0
                _consumer_wait_state(smem_addr_cg0, 336, kk0_count_cg0, kk0_phase_cg0)
                shared_consumer_index_cg0 = _pipe_next_index(kk0_count_cg0, 2)
                shared_consumer_phase_cg0 = _pipe_next_phase(kk0_count_cg0, kk0_phase_cg0, 2)
                kk_values_cg0: T.float32[32]
                _cg0_tmem_ld(tmem_base_cg0, kk0_count_cg0, thread, kk_values_cg0)
                T.ptx.tcgen05.wait__ld.sync.aligned()
                _consumer_release_state(smem_addr_cg0, 352, kk0_count_cg0)
                for pair_frag_cg0 in T.unroll(16):
                    packed_mul_cg0: T.uint64
                    T.ptx.mul.rn.f32x2(
                        packed_mul_cg0,
                        T.cuda.make_float2(
                            kk_values_cg0[pair_frag_cg0 * 2], kk_values_cg0[pair_frag_cg0 * 2 + 1]
                        ),
                        T.cuda.make_float2(
                            transfer0_cg0[pair_frag_cg0 * 2], transfer0_cg0[pair_frag_cg0 * 2 + 1]
                        ),
                    )
                    T.ptx.mul.rn.f32x2(
                        packed_mul_cg0, packed_mul_cg0, T.cuda.make_float2(beta0_cg0, beta0_cg0)
                    )
                    kk_values_cg0[pair_frag_cg0 * 2] = T.cuda.float2_x(packed_mul_cg0)
                    kk_values_cg0[pair_frag_cg0 * 2 + 1] = T.cuda.float2_y(packed_mul_cg0)
                _cg0_store_fragment(smem_raw, SMEM_AINV_OFF, ainv0_count_cg0, thread, kk_values_cg0)

                # KK1 follows KK0 before any inverse work.
                ainv1_count_cg0: T.int32 = ainv_producer_index_cg0
                ainv1_phase_cg0: T.int32 = ainv_producer_phase_cg0
                _producer_acquire_state(smem_addr_cg0, 408, ainv1_count_cg0, ainv1_phase_cg0)
                ainv_producer_index_cg0 = _pipe_next_index(ainv1_count_cg0, 3)
                ainv_producer_phase_cg0 = _pipe_next_phase(ainv1_count_cg0, ainv1_phase_cg0, 3)
                kk1_count_cg0: T.int32 = shared_consumer_index_cg0
                kk1_phase_cg0: T.int32 = shared_consumer_phase_cg0
                _consumer_wait_state(smem_addr_cg0, 336, kk1_count_cg0, kk1_phase_cg0)
                shared_consumer_index_cg0 = _pipe_next_index(kk1_count_cg0, 2)
                shared_consumer_phase_cg0 = _pipe_next_phase(kk1_count_cg0, kk1_phase_cg0, 2)
                _cg0_tmem_ld(tmem_base_cg0, kk1_count_cg0, thread, kk_values_cg0)
                T.ptx.tcgen05.wait__ld.sync.aligned()
                _consumer_release_state(smem_addr_cg0, 352, kk1_count_cg0)
                for pair_frag_cg0 in T.unroll(16):
                    packed_mul_cg0: T.uint64
                    T.ptx.mul.rn.f32x2(
                        packed_mul_cg0,
                        T.cuda.make_float2(
                            kk_values_cg0[pair_frag_cg0 * 2], kk_values_cg0[pair_frag_cg0 * 2 + 1]
                        ),
                        T.cuda.make_float2(
                            transfer1_cg0[pair_frag_cg0 * 2], transfer1_cg0[pair_frag_cg0 * 2 + 1]
                        ),
                    )
                    T.ptx.mul.rn.f32x2(
                        packed_mul_cg0, packed_mul_cg0, T.cuda.make_float2(beta1_cg0, beta1_cg0)
                    )
                    kk_values_cg0[pair_frag_cg0 * 2] = T.cuda.float2_x(packed_mul_cg0)
                    kk_values_cg0[pair_frag_cg0 * 2 + 1] = T.cuda.float2_y(packed_mul_cg0)
                _cg0_store_fragment(smem_raw, SMEM_AINV_OFF, ainv1_count_cg0, thread, kk_values_cg0)

                # Hierarchical f16 inverse: diagonal 8, then 16, 32 and 64.
                local_warp_cg0: T.int32 = (thread >> 5) & 3
                inverse_group_cg0: T.int32 = local_warp_cg0 >> 1
                inverse_local_warp_cg0: T.int32 = local_warp_cg0 & 1
                inverse_stage_cg0: T.int32 = ainv0_count_cg0
                if inverse_group_cg0 == 1:
                    inverse_stage_cg0 = ainv1_count_cg0
                diagonal_block_cg0: T.int32 = ((inverse_local_warp_cg0 * 32 + lane) >> 3) * 8
                T.ptx.bar.sync(T.uint32(INVERSE_BARRIER), T.uint32(128))
                _invert_diagonal_8x8(smem_raw, inverse_stage_cg0, diagonal_block_cg0, lane)
                T.ptx.bar.sync(T.uint32(INVERSE_BARRIER), T.uint32(128))
                _inverse_8_to_16(smem_raw, ainv0_count_cg0, local_warp_cg0 * 16, lane)
                _inverse_8_to_16(smem_raw, ainv1_count_cg0, local_warp_cg0 * 16, lane)
                T.ptx.bar.sync(T.uint32(INVERSE_BARRIER), T.uint32(128))
                _inverse_16_to_32(smem_raw, inverse_stage_cg0, inverse_local_warp_cg0 * 32, lane)
                T.ptx.bar.sync(T.uint32(INVERSE_BARRIER), T.uint32(128))
                _inverse_32_to_64(smem_raw, inverse_stage_cg0, inverse_local_warp_cg0, lane)
                T.ptx.bar.sync(T.uint32(INVERSE_BARRIER), T.uint32(128))

                # Publish stage 0, scaling the inverse by the beta COLUMN.
                inv_values_cg0: T.float32[32]
                ainv0_stage_cg0: T.int32 = ainv0_count_cg0
                _cg0_load_fragment(smem_raw, SMEM_AINV_OFF, ainv0_stage_cg0, thread, inv_values_cg0)
                for pair_frag_cg0 in T.unroll(16):
                    inv_col0_cg0: T.int32 = col_base_cg0 + pair_frag_cg0 * 2
                    if pair_frag_cg0 >= 8:
                        inv_col0_cg0 = inv_col0_cg0 + 16
                    inv_mul_cg0: T.uint64
                    T.ptx.mul.rn.f32x2(
                        inv_mul_cg0,
                        T.cuda.make_float2(
                            inv_values_cg0[pair_frag_cg0 * 2], inv_values_cg0[pair_frag_cg0 * 2 + 1]
                        ),
                        T.cuda.make_float2(
                            s_beta[beta0_count_cg0 * 64 + inv_col0_cg0],
                            s_beta[beta0_count_cg0 * 64 + inv_col0_cg0 + 1],
                        ),
                    )
                    inv_values_cg0[pair_frag_cg0 * 2] = T.cuda.float2_x(inv_mul_cg0)
                    inv_values_cg0[pair_frag_cg0 * 2 + 1] = T.cuda.float2_y(inv_mul_cg0)
                _cg0_store_fragment(
                    smem_raw, SMEM_AINV_OFF, ainv0_stage_cg0, thread, inv_values_cg0
                )
                T.ptx.fence.proxy.async_.shared__cta()
                _software_commit_state(smem_addr_cg0, 384, ainv0_count_cg0)
                _consumer_release_state(smem_addr_cg0, 264, beta0_count_cg0)

                # Stage 1 publication is intentionally ordered after stage 0.
                ainv1_stage_cg0: T.int32 = ainv1_count_cg0
                _cg0_load_fragment(smem_raw, SMEM_AINV_OFF, ainv1_stage_cg0, thread, inv_values_cg0)
                for pair_frag_cg0 in T.unroll(16):
                    inv_col1_cg0: T.int32 = col_base_cg0 + pair_frag_cg0 * 2
                    if pair_frag_cg0 >= 8:
                        inv_col1_cg0 = inv_col1_cg0 + 16
                    inv_mul_cg0: T.uint64
                    T.ptx.mul.rn.f32x2(
                        inv_mul_cg0,
                        T.cuda.make_float2(
                            inv_values_cg0[pair_frag_cg0 * 2], inv_values_cg0[pair_frag_cg0 * 2 + 1]
                        ),
                        T.cuda.make_float2(
                            s_beta[beta1_count_cg0 * 64 + inv_col1_cg0],
                            s_beta[beta1_count_cg0 * 64 + inv_col1_cg0 + 1],
                        ),
                    )
                    inv_values_cg0[pair_frag_cg0 * 2] = T.cuda.float2_x(inv_mul_cg0)
                    inv_values_cg0[pair_frag_cg0 * 2 + 1] = T.cuda.float2_y(inv_mul_cg0)
                _cg0_store_fragment(
                    smem_raw, SMEM_AINV_OFF, ainv1_stage_cg0, thread, inv_values_cg0
                )
                T.ptx.fence.proxy.async_.shared__cta()
                _software_commit_state(smem_addr_cg0, 384, ainv1_count_cg0)
                _consumer_release_state(smem_addr_cg0, 264, beta1_count_cg0)

                # QK0/QK1 epilogues reuse the same transfer registers.
                for qk_in_pair_cg0 in T.unroll(2):
                    qk_ready_count_cg0: T.int32 = qk_producer_index_cg0
                    qk_ready_phase_cg0: T.int32 = qk_producer_phase_cg0
                    _producer_acquire_state(
                        smem_addr_cg0, 448, qk_ready_count_cg0, qk_ready_phase_cg0
                    )
                    qk_producer_index_cg0 = _pipe_next_index(qk_ready_count_cg0, 2)
                    qk_producer_phase_cg0 = _pipe_next_phase(
                        qk_ready_count_cg0, qk_ready_phase_cg0, 2
                    )
                    qk_acc_count_cg0: T.int32 = shared_consumer_index_cg0
                    qk_acc_phase_cg0: T.int32 = shared_consumer_phase_cg0
                    _consumer_wait_state(smem_addr_cg0, 336, qk_acc_count_cg0, qk_acc_phase_cg0)
                    shared_consumer_index_cg0 = _pipe_next_index(qk_acc_count_cg0, 2)
                    shared_consumer_phase_cg0 = _pipe_next_phase(
                        qk_acc_count_cg0, qk_acc_phase_cg0, 2
                    )
                    _cg0_tmem_ld(tmem_base_cg0, qk_acc_count_cg0, thread, kk_values_cg0)
                    for pair_frag_cg0 in T.unroll(16):
                        qk_factor0_cg0: T.float32 = transfer0_cg0[pair_frag_cg0 * 2]
                        qk_factor1_cg0: T.float32 = transfer0_cg0[pair_frag_cg0 * 2 + 1]
                        if qk_in_pair_cg0 == 1:
                            qk_factor0_cg0 = transfer1_cg0[pair_frag_cg0 * 2]
                            qk_factor1_cg0 = transfer1_cg0[pair_frag_cg0 * 2 + 1]
                        qk_mul_cg0: T.uint64
                        T.ptx.mul.rn.f32x2(
                            qk_mul_cg0,
                            T.cuda.make_float2(
                                kk_values_cg0[pair_frag_cg0 * 2],
                                kk_values_cg0[pair_frag_cg0 * 2 + 1],
                            ),
                            T.cuda.make_float2(qk_factor0_cg0, qk_factor1_cg0),
                        )
                        T.ptx.mul.rn.f32x2(qk_mul_cg0, qk_mul_cg0, T.cuda.make_float2(scale, scale))
                        kk_values_cg0[pair_frag_cg0 * 2] = T.cuda.float2_x(qk_mul_cg0)
                        kk_values_cg0[pair_frag_cg0 * 2 + 1] = T.cuda.float2_y(qk_mul_cg0)
                    _cg0_store_fragment(
                        smem_raw, SMEM_QK_OFF, qk_ready_count_cg0, thread, kk_values_cg0
                    )
                    T.ptx.fence.proxy.async_.shared__cta()
                    T.ptx.tcgen05.wait__ld.sync.aligned()
                    _consumer_release_state(smem_addr_cg0, 352, qk_acc_count_cg0)
                    _software_commit_state(smem_addr_cg0, 432, qk_ready_count_cg0)
            work_linear_cg0 = work_linear_cg0 + grid_x
        _producer_tail_state(
            smem_addr_cg0, 408, ainv_producer_index_cg0, ainv_producer_phase_cg0, 3
        )
        _producer_tail_state(smem_addr_cg0, 448, qk_producer_index_cg0, qk_producer_phase_cg0, 2)

    if warp >= 4 and warp <= 7:
        smem_addr_cg1: T.uint32 = T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0]))
        T.ptx.setmaxnreg.inc.sync.aligned.u32(256)
        if warp == 4:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_holding[0]), T.uint32(TMEM_COLUMNS)
            )
        T.ptx.bar.sync(T.uint32(TMEM_ALLOC_BARRIER), T.uint32(320))
        tmem_base_cg1: T.int32
        T.ptx.ld.volatile.shared.s32(tmem_base_cg1, T.address_of(tmem_holding[0]))
        v_consumer_index_cg1: T.int32 = 0
        v_consumer_phase_cg1: T.int32 = 0
        gate_consumer_index_cg1: T.int32 = 0
        gate_consumer_phase_cg1: T.int32 = 0
        shared_consumer_count_cg1: T.int32 = 0
        kv_consumer_count_cg1: T.int32 = 0
        qstate_consumer_count_cg1: T.int32 = 0
        kv_producer_count_cg1: T.int32 = 0
        state_input_producer_count_cg1: T.int32 = 0
        vks_producer_count_cg1: T.int32 = 0
        nv_producer_count_cg1: T.int32 = 0
        decay_producer_count_cg1: T.int32 = 0
        o_producer_index_cg1: T.int32 = 0
        o_producer_phase_cg1: T.int32 = 1
        work_linear_cg1: T.int32 = block
        while work_linear_cg1 < total_work:
            work_u32_cg1: T.uint32 = T.cast(work_linear_cg1, "uint32")
            remain_u32_cg1: T.uint32 = _udiv_u32_const(work_u32_cg1, HV)
            head_cg1: T.int32 = T.cast(
                work_u32_cg1 - remain_u32_cg1 * T.uint32(output_heads), "int32"
            )
            remain_cg1: T.int32 = T.cast(remain_u32_cg1, "int32")
            batch_cg1: T.int32 = remain_cg1
            batch_start_cg1: T.int64 = T.cast(cu_seqlens[batch_cg1], "int64")
            batch_end_cg1: T.int64 = T.cast(cu_seqlens[batch_cg1 + 1], "int64")
            seqlen_cg1: T.int32 = T.cast(batch_end_cg1 - batch_start_cg1, "int32")
            num_valid_chunks_cg1: T.int32 = (seqlen_cg1 + CHUNK - 1) // CHUNK
            num_pairs_cg1: T.int32 = (seqlen_cg1 + PAIR_TOKENS - 1) // PAIR_TOKENS
            padded_chunks_cg1: T.int32 = num_pairs_cg1 * 2

            # sketch load_initial_state -> CK:L3935-L4017.  Its KV producer
            # cursor remains persistent across scheduler work items.
            if padded_chunks_cg1 > 0:
                initial_count_cg1: T.int32 = kv_producer_count_cg1
                _load_initial_state(
                    smem_addr_cg1,
                    initial_state,
                    batch_cg1,
                    head_cg1,
                    tmem_base_cg1,
                    thread,
                    initial_count_cg1,
                    HV=HV,
                )
                kv_producer_count_cg1 = kv_producer_count_cg1 + 1

                # sketch compute_group_1_chunk -> CK:L4149-L4634.  The loop
                # body below follows the helper source literally; fixed
                # USE_INITIAL_STATE=True eliminates only the false branches.
                for chunk_cg1 in T.serial(0, padded_chunks_cg1):
                    if T.bitwise_and(chunk_cg1, T.int32(1)) == 0:
                        kv_producer_count_cg1 = kv_producer_count_cg1 + 1
                        kv_producer_count_cg1 = kv_producer_count_cg1 + 1

                    gate_count_cg1: T.int32 = gate_consumer_index_cg1
                    gate_phase_cg1: T.int32 = gate_consumer_phase_cg1
                    _consumer_wait_state(smem_addr_cg1, 144, gate_count_cg1, gate_phase_cg1)
                    gate_consumer_index_cg1 = _pipe_next_index(gate_count_cg1, 5)
                    gate_consumer_phase_cg1 = _pipe_next_phase(gate_count_cg1, gate_phase_cg1, 5)
                    gate_stage_cg1: T.int32 = gate_count_cg1
                    cumprod_total_cg1: T.float32 = s_cumprod[gate_stage_cg1 * 64 + 63]

                    # Previous recurrent state -> f16 state-input TMEM, then
                    # the same f32 fragment is decayed in place and restored.
                    kv_previous_count_cg1: T.int32 = kv_consumer_count_cg1
                    _consumer_wait(smem_addr_cg1, 320, kv_previous_count_cg1, 1)
                    kv_consumer_count_cg1 = kv_consumer_count_cg1 + 1
                    state_input_count_cg1: T.int32 = state_input_producer_count_cg1
                    state_input_producer_count_cg1 = state_input_producer_count_cg1 + 1
                    state_values_cg1: T.float32[128]
                    state_input_words_cg1: T.uint32[64]
                    for state_sub_cg1 in range(4):
                        _state_tmem_ld_sub(
                            tmem_base_cg1,
                            thread,
                            state_sub_cg1,
                            state_values_cg1,
                            state_sub_cg1 * 32,
                        )
                    for state_pair_cg1 in T.unroll(64):
                        state_input_words_cg1[state_pair_cg1] = _pack_f16x2_cg1(
                            state_values_cg1[state_pair_cg1 * 2],
                            state_values_cg1[state_pair_cg1 * 2 + 1],
                        )
                    for state_sub_cg1 in range(4):
                        _state_input_tmem_st_sub(
                            tmem_base_cg1, thread, state_sub_cg1, state_input_words_cg1
                        )
                    T.ptx.tcgen05.wait__st.sync.aligned()
                    _software_commit(smem_addr_cg1, 464, state_input_count_cg1, 1)

                    for state_pair_cg1 in T.unroll(64):
                        state_mul_cg1: T.uint64
                        T.ptx.mul.rn.f32x2(
                            state_mul_cg1,
                            T.cuda.make_float2(
                                state_values_cg1[state_pair_cg1 * 2],
                                state_values_cg1[state_pair_cg1 * 2 + 1],
                            ),
                            T.cuda.make_float2(cumprod_total_cg1, cumprod_total_cg1),
                        )
                        state_values_cg1[state_pair_cg1 * 2] = T.cuda.float2_x(state_mul_cg1)
                        state_values_cg1[state_pair_cg1 * 2 + 1] = T.cuda.float2_y(state_mul_cg1)
                    for state_sub_cg1 in range(4):
                        _state_tmem_st_sub(
                            tmem_base_cg1,
                            thread,
                            state_sub_cg1,
                            state_values_cg1,
                            state_sub_cg1 * 32,
                        )
                    T.ptx.tcgen05.wait__st.sync.aligned()
                    _consumer_release(smem_addr_cg1, 328, kv_previous_count_cg1, 1)

                    # Register factors in the exact Ld16x256b fragment order:
                    # columns 2*(tid%4)+8*g, with each pair reused twice per row half.
                    cumprod_factor_cg1: T.float32[16]
                    decay_factor_cg1: T.float32[16]
                    factor_col_base_cg1: T.int32 = T.bitwise_and(thread << 1, T.int32(6))
                    last_log_cg1: T.float32 = s_cumsumlog[gate_stage_cg1 * 64 + 63]
                    for factor_group_cg1 in T.unroll(8):
                        factor_col_cg1: T.int32 = factor_col_base_cg1 + factor_group_cg1 * 8
                        cumprod_factor_cg1[factor_group_cg1 * 2] = s_cumprod[
                            gate_stage_cg1 * 64 + factor_col_cg1
                        ]
                        cumprod_factor_cg1[factor_group_cg1 * 2 + 1] = s_cumprod[
                            gate_stage_cg1 * 64 + factor_col_cg1 + 1
                        ]
                        # Frozen PTX spells this as scalar negations feeding
                        # add.rn.f32x2; ptxas folds those negations into the
                        # packed subtraction.  Express the resulting native
                        # operation directly so the CUDA C++ bridge does not
                        # materialize two standalone FADD negations.
                        decay_diff_cg1: T.uint64[1]
                        T.ptx.sub.rn.f32x2(
                            decay_diff_cg1[0],
                            T.cuda.make_float2(last_log_cg1, last_log_cg1),
                            T.cuda.make_float2(
                                s_cumsumlog[gate_stage_cg1 * 64 + factor_col_cg1],
                                s_cumsumlog[gate_stage_cg1 * 64 + factor_col_cg1 + 1],
                            ),
                        )
                        T.ptx.ex2.approx.ftz.f32(
                            decay_factor_cg1[factor_group_cg1 * 2],
                            T.cuda.float2_x(decay_diff_cg1[0]),
                        )
                        T.ptx.ex2.approx.ftz.f32(
                            decay_factor_cg1[factor_group_cg1 * 2 + 1],
                            T.cuda.float2_y(decay_diff_cg1[0]),
                        )
                    _consumer_release_state(smem_addr_cg1, 184, gate_count_cg1)

                    # V and KS -> VKS, published through fixed shared-input slot 0.
                    vks_count_cg1: T.int32 = vks_producer_count_cg1
                    vks_producer_count_cg1 = vks_producer_count_cg1 + 1
                    v_count_cg1: T.int32 = v_consumer_index_cg1
                    v_phase_cg1: T.int32 = v_consumer_phase_cg1
                    _consumer_wait_state(smem_addr_cg1, 96, v_count_cg1, v_phase_cg1)
                    v_consumer_index_cg1 = _pipe_next_index(v_count_cg1, 3)
                    v_consumer_phase_cg1 = _pipe_next_phase(v_count_cg1, v_phase_cg1, 3)
                    v_words_cg1: T.uint32[32]
                    _load_v_fragment(smem_raw, v_count_cg1, thread, v_words_cg1)

                    ks_count_cg1: T.int32 = shared_consumer_count_cg1
                    _consumer_wait(smem_addr_cg1, 368, ks_count_cg1, 1)
                    shared_consumer_count_cg1 = shared_consumer_count_cg1 + 1
                    fragment_cg1: T.float32[64]
                    _cg1_tmem_ld_f32(tmem_base_cg1, TMEM_CG1_ACC_COL, thread, fragment_cg1)
                    for row_half_cg1 in T.unroll(2):
                        for factor_group_cg1 in T.unroll(8):
                            for factor_repeat_cg1 in T.unroll(2):
                                pair_cg1: T.int32 = (
                                    row_half_cg1 * 16 + factor_group_cg1 * 2 + factor_repeat_cg1
                                )
                                ks_mul_cg1: T.uint64
                                T.ptx.mul.rn.f32x2(
                                    ks_mul_cg1,
                                    T.cuda.make_float2(
                                        fragment_cg1[pair_cg1 * 2], fragment_cg1[pair_cg1 * 2 + 1]
                                    ),
                                    T.cuda.make_float2(
                                        cumprod_factor_cg1[factor_group_cg1 * 2],
                                        cumprod_factor_cg1[factor_group_cg1 * 2 + 1],
                                    ),
                                )
                                fragment_cg1[pair_cg1 * 2] = T.cuda.float2_x(ks_mul_cg1)
                                fragment_cg1[pair_cg1 * 2 + 1] = T.cuda.float2_y(ks_mul_cg1)
                    _consumer_release(smem_addr_cg1, 376, ks_count_cg1, 1)
                    for pair_cg1 in T.unroll(32):
                        ks_word_cg1: T.uint32 = _pack_f16x2_cg1(
                            fragment_cg1[pair_cg1 * 2], fragment_cg1[pair_cg1 * 2 + 1]
                        )
                        T.ptx.sub.f16x2(v_words_cg1[pair_cg1], v_words_cg1[pair_cg1], ks_word_cg1)
                    _cg1_tmem_st_f16(tmem_base_cg1, TMEM_SHARED_INPUT_COL, thread, v_words_cg1)
                    T.ptx.tcgen05.wait__st.sync.aligned()
                    _software_commit(smem_addr_cg1, 480, vks_count_cg1, 1)

                    # QS is scaled in f32 and written back to the same q-state slot.
                    qs_count_cg1: T.int32 = qstate_consumer_count_cg1
                    _consumer_wait(smem_addr_cg1, 304, qs_count_cg1, 1)
                    qstate_consumer_count_cg1 = qstate_consumer_count_cg1 + 1
                    _cg1_tmem_ld_f32(tmem_base_cg1, TMEM_Q_STATE_COL, thread, fragment_cg1)
                    for row_half_cg1 in T.unroll(2):
                        for factor_group_cg1 in T.unroll(8):
                            for factor_repeat_cg1 in T.unroll(2):
                                pair_cg1: T.int32 = (
                                    row_half_cg1 * 16 + factor_group_cg1 * 2 + factor_repeat_cg1
                                )
                                qs_mul_cg1: T.uint64
                                T.ptx.mul.rn.f32x2(
                                    qs_mul_cg1,
                                    T.cuda.make_float2(
                                        fragment_cg1[pair_cg1 * 2], fragment_cg1[pair_cg1 * 2 + 1]
                                    ),
                                    T.cuda.make_float2(
                                        cumprod_factor_cg1[factor_group_cg1 * 2],
                                        cumprod_factor_cg1[factor_group_cg1 * 2 + 1],
                                    ),
                                )
                                T.ptx.mul.rn.f32x2(
                                    qs_mul_cg1, qs_mul_cg1, T.cuda.make_float2(scale, scale)
                                )
                                fragment_cg1[pair_cg1 * 2] = T.cuda.float2_x(qs_mul_cg1)
                                fragment_cg1[pair_cg1 * 2 + 1] = T.cuda.float2_y(qs_mul_cg1)
                    _cg1_tmem_st_f32(tmem_base_cg1, TMEM_Q_STATE_COL, thread, fragment_cg1)
                    T.ptx.tcgen05.wait__st.sync.aligned()
                    _consumer_release(smem_addr_cg1, 312, qs_count_cg1, 1)

                    # NV is consumed only after VKS and QS publication.  V is
                    # released by one signalling lane per CG1 warp before the read.
                    nv_acc_count_cg1: T.int32 = shared_consumer_count_cg1
                    _consumer_wait(smem_addr_cg1, 368, nv_acc_count_cg1, 1)
                    shared_consumer_count_cg1 = shared_consumer_count_cg1 + 1
                    if lane == 0:
                        _consumer_release_state(smem_addr_cg1, 120, v_count_cg1)
                    _cg1_tmem_ld_f32(tmem_base_cg1, TMEM_CG1_ACC_COL, thread, fragment_cg1)
                    nv_words_cg1: T.uint32[32]
                    for pair_cg1 in T.unroll(32):
                        nv_words_cg1[pair_cg1] = _pack_f16x2_cg1(
                            fragment_cg1[pair_cg1 * 2], fragment_cg1[pair_cg1 * 2 + 1]
                        )
                    _consumer_release(smem_addr_cg1, 376, nv_acc_count_cg1, 1)

                    for row_half_cg1 in T.unroll(2):
                        for factor_group_cg1 in T.unroll(8):
                            for factor_repeat_cg1 in T.unroll(2):
                                pair_cg1: T.int32 = (
                                    row_half_cg1 * 16 + factor_group_cg1 * 2 + factor_repeat_cg1
                                )
                                decay_mul_cg1: T.uint64
                                T.ptx.mul.rn.f32x2(
                                    decay_mul_cg1,
                                    T.cuda.make_float2(
                                        fragment_cg1[pair_cg1 * 2], fragment_cg1[pair_cg1 * 2 + 1]
                                    ),
                                    T.cuda.make_float2(
                                        decay_factor_cg1[factor_group_cg1 * 2],
                                        decay_factor_cg1[factor_group_cg1 * 2 + 1],
                                    ),
                                )
                                fragment_cg1[pair_cg1 * 2] = T.cuda.float2_x(decay_mul_cg1)
                                fragment_cg1[pair_cg1 * 2 + 1] = T.cuda.float2_y(decay_mul_cg1)

                    nv_count_cg1: T.int32 = nv_producer_count_cg1
                    nv_producer_count_cg1 = nv_producer_count_cg1 + 1
                    decay_count_cg1: T.int32 = decay_producer_count_cg1
                    decay_producer_count_cg1 = decay_producer_count_cg1 + 1
                    decay_words_cg1: T.uint32[32]
                    # The source loop interleaves NV and decay-V stores by row half.
                    for row_half_cg1 in range(2):
                        _cg1_tmem_st_f16_half(
                            tmem_base_cg1, TMEM_SHARED_INPUT_COL, thread, row_half_cg1, nv_words_cg1
                        )
                        for pair_in_half_cg1 in T.unroll(16):
                            pair_cg1: T.int32 = row_half_cg1 * 16 + pair_in_half_cg1
                            decay_words_cg1[pair_cg1] = _pack_f16x2_cg1(
                                fragment_cg1[pair_cg1 * 2], fragment_cg1[pair_cg1 * 2 + 1]
                            )
                        _cg1_tmem_st_f16_half(
                            tmem_base_cg1,
                            TMEM_SHARED_INPUT_COL + 32,
                            thread,
                            row_half_cg1,
                            decay_words_cg1,
                        )
                    T.ptx.tcgen05.wait__st.sync.aligned()
                    _software_commit(smem_addr_cg1, 496, nv_count_cg1, 1)
                    _software_commit(smem_addr_cg1, 512, decay_count_cg1, 1)

                    # Same-chunk QKV output, converted and staged through O's
                    # persistent two-stage SMEM ring.
                    o_count_cg1: T.int32 = o_producer_index_cg1
                    o_phase_cg1: T.int32 = o_producer_phase_cg1
                    _producer_acquire_state(smem_addr_cg1, 544, o_count_cg1, o_phase_cg1)
                    o_producer_index_cg1 = _pipe_next_index(o_count_cg1, 2)
                    o_producer_phase_cg1 = _pipe_next_phase(o_count_cg1, o_phase_cg1, 2)
                    qkv_count_cg1: T.int32 = qstate_consumer_count_cg1
                    _consumer_wait(smem_addr_cg1, 304, qkv_count_cg1, 1)
                    qstate_consumer_count_cg1 = qstate_consumer_count_cg1 + 1
                    _cg1_tmem_ld_f32(tmem_base_cg1, TMEM_Q_STATE_COL, thread, fragment_cg1)
                    o_words_cg1: T.uint32[32]
                    for pair_cg1 in T.unroll(32):
                        o_words_cg1[pair_cg1] = _pack_f16x2_cg1(
                            fragment_cg1[pair_cg1 * 2], fragment_cg1[pair_cg1 * 2 + 1]
                        )
                    _store_o_fragment(smem_raw, o_count_cg1, thread, o_words_cg1)
                    T.ptx.fence.proxy.async_.shared__cta()
                    _consumer_release(smem_addr_cg1, 312, qkv_count_cg1, 1)
                    _software_commit_state(smem_addr_cg1, 528, o_count_cg1)

                final_count_cg1: T.int32 = kv_consumer_count_cg1
                _store_final_state(
                    smem_addr_cg1,
                    final_state,
                    batch_cg1,
                    head_cg1,
                    tmem_base_cg1,
                    thread,
                    final_count_cg1,
                    HV=HV,
                )
                kv_consumer_count_cg1 = kv_consumer_count_cg1 + 1
            else:
                # Retained runtime empty-sequence branch from CK:L1422-L1432.
                _copy_empty_state(initial_state, final_state, batch_cg1, head_cg1, thread, HV=HV)
            work_linear_cg1 = work_linear_cg1 + grid_x
        if warp == 4:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                T.cast(tmem_base_cg1, "uint32"), T.uint32(TMEM_COLUMNS)
            )
        _producer_tail_state(smem_addr_cg1, 544, o_producer_index_cg1, o_producer_phase_cg1, 2)
        _producer_tail(smem_addr_cg1, 472, state_input_producer_count_cg1, 1)
    elif warp == 8:
        smem_addr_issuer0: T.uint32 = T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0]))
        T.ptx.setmaxnreg.dec.sync.aligned.u32(24)
        T.ptx.bar.sync(T.uint32(TMEM_ALLOC_BARRIER), T.uint32(320))
        tmem_base_issuer0: T.int32
        T.ptx.ld.volatile.shared.s32(tmem_base_issuer0, T.address_of(tmem_holding[0]))
        cg0_producer_index_i0: T.int32 = 0
        cg0_producer_phase_i0: T.int32 = 1
        k_consumer_index_i0: T.int32 = 0
        k_consumer_phase_i0: T.int32 = 0
        q_consumer_index_i0: T.int32 = 0
        q_consumer_phase_i0: T.int32 = 0
        work_linear_issuer0: T.int32 = block
        while work_linear_issuer0 < total_work:
            remain_issuer0: T.int32 = T.cast(
                _udiv_u32_const(T.cast(work_linear_issuer0, "uint32"), HV), "int32"
            )
            batch_issuer0: T.int32 = remain_issuer0
            batch_start_issuer0: T.int64 = T.cast(cu_seqlens[batch_issuer0], "int64")
            batch_end_issuer0: T.int64 = T.cast(cu_seqlens[batch_issuer0 + 1], "int64")
            seqlen_issuer0: T.int32 = T.cast(batch_end_issuer0 - batch_start_issuer0, "int32")
            num_pairs_issuer0: T.int32 = (seqlen_issuer0 + PAIR_TOKENS - 1) // PAIR_TOKENS

            # sketch mma_cg0_pair -> CK:L2161-L2216.  Pipeline states are
            # persistent; handle snapshots retain their stages until release.
            for _pair_issuer0 in T.serial(0, num_pairs_issuer0):
                kk0_count_i0: T.int32 = cg0_producer_index_i0
                kk0_phase_i0: T.int32 = cg0_producer_phase_i0
                _producer_acquire_state(smem_addr_issuer0, 352, kk0_count_i0, kk0_phase_i0)
                cg0_producer_index_i0 = _pipe_next_index(kk0_count_i0, 2)
                cg0_producer_phase_i0 = _pipe_next_phase(kk0_count_i0, kk0_phase_i0, 2)
                k0_count_i0: T.int32 = k_consumer_index_i0
                k0_phase_i0: T.int32 = k_consumer_phase_i0
                _consumer_wait_state(smem_addr_issuer0, 0, k0_count_i0, k0_phase_i0)
                k_consumer_index_i0 = _pipe_next_index(k0_count_i0, 4)
                k_consumer_phase_i0 = _pipe_next_phase(k0_count_i0, k0_phase_i0, 4)
                k0_desc_i0: T.uint64 = _smem_desc_b128(
                    _shared_addr(smem_addr_issuer0, SMEM_K_OFF + k0_count_i0 * 16384)
                )
                _mma_ss_64x64_k128(
                    tmem_base_issuer0 + TMEM_CG0_ACC_COL + kk0_count_i0 * 64,
                    k0_desc_i0,
                    k0_desc_i0,
                    _shared_addr(smem_addr_issuer0, 336 + kk0_count_i0 * 8),
                )

                kk1_count_i0: T.int32 = cg0_producer_index_i0
                kk1_phase_i0: T.int32 = cg0_producer_phase_i0
                _producer_acquire_state(smem_addr_issuer0, 352, kk1_count_i0, kk1_phase_i0)
                cg0_producer_index_i0 = _pipe_next_index(kk1_count_i0, 2)
                cg0_producer_phase_i0 = _pipe_next_phase(kk1_count_i0, kk1_phase_i0, 2)
                k1_count_i0: T.int32 = k_consumer_index_i0
                k1_phase_i0: T.int32 = k_consumer_phase_i0
                _consumer_wait_state(smem_addr_issuer0, 0, k1_count_i0, k1_phase_i0)
                k_consumer_index_i0 = _pipe_next_index(k1_count_i0, 4)
                k_consumer_phase_i0 = _pipe_next_phase(k1_count_i0, k1_phase_i0, 4)
                k1_desc_i0: T.uint64 = _smem_desc_b128(
                    _shared_addr(smem_addr_issuer0, SMEM_K_OFF + k1_count_i0 * 16384)
                )
                _mma_ss_64x64_k128(
                    tmem_base_issuer0 + TMEM_CG0_ACC_COL + kk1_count_i0 * 64,
                    k1_desc_i0,
                    k1_desc_i0,
                    _shared_addr(smem_addr_issuer0, 336 + kk1_count_i0 * 8),
                )

                q0_count_i0: T.int32 = q_consumer_index_i0
                q0_phase_i0: T.int32 = q_consumer_phase_i0
                _consumer_wait_state(smem_addr_issuer0, 64, q0_count_i0, q0_phase_i0)
                q_consumer_index_i0 = _pipe_next_index(q0_count_i0, 2)
                q_consumer_phase_i0 = _pipe_next_phase(q0_count_i0, q0_phase_i0, 2)
                qk0_count_i0: T.int32 = cg0_producer_index_i0
                qk0_phase_i0: T.int32 = cg0_producer_phase_i0
                _producer_acquire_state(smem_addr_issuer0, 352, qk0_count_i0, qk0_phase_i0)
                cg0_producer_index_i0 = _pipe_next_index(qk0_count_i0, 2)
                cg0_producer_phase_i0 = _pipe_next_phase(qk0_count_i0, qk0_phase_i0, 2)
                q0_desc_i0: T.uint64 = _smem_desc_b128(
                    _shared_addr(smem_addr_issuer0, SMEM_Q_OFF + q0_count_i0 * 16384)
                )
                _mma_ss_64x64_k128(
                    tmem_base_issuer0 + TMEM_CG0_ACC_COL + qk0_count_i0 * 64,
                    q0_desc_i0,
                    k0_desc_i0,
                    _shared_addr(smem_addr_issuer0, 336 + qk0_count_i0 * 8),
                )

                q1_count_i0: T.int32 = q_consumer_index_i0
                q1_phase_i0: T.int32 = q_consumer_phase_i0
                _consumer_wait_state(smem_addr_issuer0, 64, q1_count_i0, q1_phase_i0)
                q_consumer_index_i0 = _pipe_next_index(q1_count_i0, 2)
                q_consumer_phase_i0 = _pipe_next_phase(q1_count_i0, q1_phase_i0, 2)
                qk1_count_i0: T.int32 = cg0_producer_index_i0
                qk1_phase_i0: T.int32 = cg0_producer_phase_i0
                _producer_acquire_state(smem_addr_issuer0, 352, qk1_count_i0, qk1_phase_i0)
                cg0_producer_index_i0 = _pipe_next_index(qk1_count_i0, 2)
                cg0_producer_phase_i0 = _pipe_next_phase(qk1_count_i0, qk1_phase_i0, 2)
                q1_desc_i0: T.uint64 = _smem_desc_b128(
                    _shared_addr(smem_addr_issuer0, SMEM_Q_OFF + q1_count_i0 * 16384)
                )
                _mma_ss_64x64_k128(
                    tmem_base_issuer0 + TMEM_CG0_ACC_COL + qk1_count_i0 * 64,
                    q1_desc_i0,
                    k1_desc_i0,
                    _shared_addr(smem_addr_issuer0, 336 + qk1_count_i0 * 8),
                )

                # Matrix-engine releases are commits, in Q0,Q1,K0,K1 order.
                _mma_commit(_shared_addr(smem_addr_issuer0, 80 + q0_count_i0 * 8))
                _mma_commit(_shared_addr(smem_addr_issuer0, 80 + q1_count_i0 * 8))
                _mma_commit(_shared_addr(smem_addr_issuer0, 32 + k0_count_i0 * 8))
                _mma_commit(_shared_addr(smem_addr_issuer0, 32 + k1_count_i0 * 8))
            work_linear_issuer0 = work_linear_issuer0 + grid_x
        _producer_tail_state(
            smem_addr_issuer0, 352, cg0_producer_index_i0, cg0_producer_phase_i0, 2
        )
    elif warp == 10:
        smem_addr_issuer1: T.uint32 = T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0]))
        T.ptx.setmaxnreg.dec.sync.aligned.u32(24)
        T.ptx.bar.sync(T.uint32(TMEM_ALLOC_BARRIER), T.uint32(320))
        tmem_base_issuer1: T.int32
        T.ptx.ld.volatile.shared.s32(tmem_base_issuer1, T.address_of(tmem_holding[0]))
        cg1_producer_count_i1: T.int32 = 0
        qstate_producer_count_i1: T.int32 = 0
        kv_producer_count_i1: T.int32 = 0
        k_consumer_index_i1: T.int32 = 0
        k_consumer_phase_i1: T.int32 = 0
        q_consumer_index_i1: T.int32 = 0
        q_consumer_phase_i1: T.int32 = 0
        ainv_consumer_index_i1: T.int32 = 0
        ainv_consumer_phase_i1: T.int32 = 0
        qk_consumer_index_i1: T.int32 = 0
        qk_consumer_phase_i1: T.int32 = 0
        state_input_consumer_count_i1: T.int32 = 0
        vks_consumer_count_i1: T.int32 = 0
        nv_consumer_count_i1: T.int32 = 0
        decay_consumer_count_i1: T.int32 = 0
        work_linear_issuer1: T.int32 = block
        while work_linear_issuer1 < total_work:
            remain_issuer1: T.int32 = T.cast(
                _udiv_u32_const(T.cast(work_linear_issuer1, "uint32"), HV), "int32"
            )
            batch_issuer1: T.int32 = remain_issuer1
            batch_start_issuer1: T.int64 = T.cast(cu_seqlens[batch_issuer1], "int64")
            batch_end_issuer1: T.int64 = T.cast(cu_seqlens[batch_issuer1 + 1], "int64")
            seqlen_issuer1: T.int32 = T.cast(batch_end_issuer1 - batch_start_issuer1, "int32")
            num_pairs_issuer1: T.int32 = (seqlen_issuer1 + PAIR_TOKENS - 1) // PAIR_TOKENS
            padded_chunks_issuer1: T.int32 = num_pairs_issuer1 * 2

            # sketch mma_cg1_chunk -> CK:L2313-L2406.
            for chunk_issuer1 in T.serial(0, padded_chunks_issuer1):
                k_count_i1: T.int32 = k_consumer_index_i1
                k_phase_i1: T.int32 = k_consumer_phase_i1
                _consumer_wait_state(smem_addr_issuer1, 0, k_count_i1, k_phase_i1)
                k_desc_i1: T.uint64 = _smem_desc_b128(
                    _shared_addr(smem_addr_issuer1, SMEM_K_OFF + k_count_i1 * 16384)
                )
                q_count_i1: T.int32 = q_consumer_index_i1
                q_phase_i1: T.int32 = q_consumer_phase_i1
                _consumer_wait_state(smem_addr_issuer1, 64, q_count_i1, q_phase_i1)
                q_desc_i1: T.uint64 = _smem_desc_b128(
                    _shared_addr(smem_addr_issuer1, SMEM_Q_OFF + q_count_i1 * 16384)
                )
                k_consumer_index_i1 = _pipe_next_index(k_count_i1, 4)
                k_consumer_phase_i1 = _pipe_next_phase(k_count_i1, k_phase_i1, 4)
                q_consumer_index_i1 = _pipe_next_index(q_count_i1, 2)
                q_consumer_phase_i1 = _pipe_next_phase(q_count_i1, q_phase_i1, 2)

                ks_count_i1: T.int32 = cg1_producer_count_i1
                _producer_acquire(smem_addr_issuer1, 376, ks_count_i1, 1)
                state_input_count_i1: T.int32 = state_input_consumer_count_i1
                _consumer_wait(smem_addr_issuer1, 464, state_input_count_i1, 1)
                _mma_ts_128x64_k128(
                    tmem_base_issuer1 + TMEM_CG1_ACC_COL,
                    tmem_base_issuer1 + TMEM_STATE_INPUT_COL,
                    k_desc_i1,
                    _pipe_full_addr(smem_addr_issuer1, 368, ks_count_i1, 1),
                )
                cg1_producer_count_i1 = cg1_producer_count_i1 + 1
                state_input_consumer_count_i1 = state_input_consumer_count_i1 + 1

                qs_count_i1: T.int32 = qstate_producer_count_i1
                _producer_acquire(smem_addr_issuer1, 312, qs_count_i1, 1)
                _mma_ts_128x64_k128(
                    tmem_base_issuer1 + TMEM_Q_STATE_COL,
                    tmem_base_issuer1 + TMEM_STATE_INPUT_COL,
                    q_desc_i1,
                    _pipe_full_addr(smem_addr_issuer1, 304, qs_count_i1, 1),
                )
                qstate_producer_count_i1 = qstate_producer_count_i1 + 1
                _mma_commit(_pipe_empty_addr(smem_addr_issuer1, 472, state_input_count_i1, 1))
                _mma_commit(_shared_addr(smem_addr_issuer1, 80 + q_count_i1 * 8))

                nv_acc_count_i1: T.int32 = cg1_producer_count_i1
                _producer_acquire(smem_addr_issuer1, 376, nv_acc_count_i1, 1)
                _consumer_wait(smem_addr_issuer1, 480, vks_consumer_count_i1, 1)
                vks_consumer_count_i1 = vks_consumer_count_i1 + 1
                ainv_count_i1: T.int32 = ainv_consumer_index_i1
                ainv_phase_i1: T.int32 = ainv_consumer_phase_i1
                _consumer_wait_state(smem_addr_issuer1, 384, ainv_count_i1, ainv_phase_i1)
                ainv_desc_i1: T.uint64 = _smem_desc_b128(
                    _shared_addr(smem_addr_issuer1, SMEM_AINV_OFF + ainv_count_i1 * 8192)
                )
                _mma_ts_128x64_k64(
                    tmem_base_issuer1 + TMEM_CG1_ACC_COL,
                    tmem_base_issuer1 + TMEM_SHARED_INPUT_COL,
                    ainv_desc_i1,
                    0,
                )
                _mma_commit(_pipe_full_addr(smem_addr_issuer1, 368, nv_acc_count_i1, 1))
                cg1_producer_count_i1 = cg1_producer_count_i1 + 1
                ainv_consumer_index_i1 = _pipe_next_index(ainv_count_i1, 3)
                ainv_consumer_phase_i1 = _pipe_next_phase(ainv_count_i1, ainv_phase_i1, 3)
                _mma_commit(_shared_addr(smem_addr_issuer1, 408 + ainv_count_i1 * 8))

                qkv_count_i1: T.int32 = qstate_producer_count_i1
                _producer_acquire(smem_addr_issuer1, 312, qkv_count_i1, 1)
                qk_count_i1: T.int32 = qk_consumer_index_i1
                qk_phase_i1: T.int32 = qk_consumer_phase_i1
                _consumer_wait_state(smem_addr_issuer1, 432, qk_count_i1, qk_phase_i1)
                qk_desc_i1: T.uint64 = _smem_desc_b128(
                    _shared_addr(smem_addr_issuer1, SMEM_QK_OFF + qk_count_i1 * 8192)
                )
                _consumer_wait(smem_addr_issuer1, 496, nv_consumer_count_i1, 1)
                nv_consumer_count_i1 = nv_consumer_count_i1 + 1
                _mma_ts_128x64_k64(
                    tmem_base_issuer1 + TMEM_Q_STATE_COL,
                    tmem_base_issuer1 + TMEM_SHARED_INPUT_COL,
                    qk_desc_i1,
                    1,
                )
                qk_consumer_index_i1 = _pipe_next_index(qk_count_i1, 2)
                qk_consumer_phase_i1 = _pipe_next_phase(qk_count_i1, qk_phase_i1, 2)
                _mma_commit(_shared_addr(smem_addr_issuer1, 448 + qk_count_i1 * 8))
                _mma_commit(_pipe_full_addr(smem_addr_issuer1, 304, qkv_count_i1, 1))
                qstate_producer_count_i1 = qstate_producer_count_i1 + 1

                if chunk_issuer1 == 0:
                    kv_producer_count_i1 = kv_producer_count_i1 + 1
                kv_count_i1: T.int32 = kv_producer_count_i1
                _producer_acquire(smem_addr_issuer1, 328, kv_count_i1, 1)
                _consumer_wait(smem_addr_issuer1, 512, decay_consumer_count_i1, 1)
                decay_consumer_count_i1 = decay_consumer_count_i1 + 1
                kt_desc_i1: T.uint64 = _smem_desc_k_trans_b128(
                    _shared_addr(smem_addr_issuer1, SMEM_K_OFF + k_count_i1 * 16384)
                )
                _mma_ts_128x128_k64(
                    tmem_base_issuer1 + TMEM_STATE_COL,
                    tmem_base_issuer1 + TMEM_SHARED_INPUT_COL + 32,
                    kt_desc_i1,
                    _pipe_full_addr(smem_addr_issuer1, 320, kv_count_i1, 1),
                )
                kv_producer_count_i1 = kv_producer_count_i1 + 1
                _mma_commit(_shared_addr(smem_addr_issuer1, 32 + k_count_i1 * 8))
            work_linear_issuer1 = work_linear_issuer1 + grid_x
        _producer_tail(smem_addr_issuer1, 376, cg1_producer_count_i1, 1)
        _producer_tail(smem_addr_issuer1, 312, qstate_producer_count_i1, 1)
        _producer_tail(smem_addr_issuer1, 328, kv_producer_count_i1, 1)
    elif warp == 9:
        smem_addr_tma: T.uint32 = T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0]))
        T.ptx.setmaxnreg.dec.sync.aligned.u32(24)
        q_producer_index: T.int32 = 0
        q_producer_phase: T.int32 = 1
        k_producer_index: T.int32 = 0
        k_producer_phase: T.int32 = 1
        v_producer_index: T.int32 = 0
        v_producer_phase: T.int32 = 1
        work_linear_tma: T.int32 = block

        # CK:L1581-L1597: copy the four by-value TensorMap payloads into the
        # CTA-private 128-byte slots once, preserving Q -> K -> V ordering.
        if work_linear_tma < total_work:
            if T.cuda.elect_sync():
                _descriptor_copy_payload(q_map, descriptor_q)
            T.cuda.warp_sync()
            if T.cuda.elect_sync():
                _descriptor_copy_payload(k_map, descriptor_k)
            T.cuda.warp_sync()
            if T.cuda.elect_sync():
                _descriptor_copy_payload(v_map, descriptor_v)
            T.cuda.warp_sync()
            T.ptx.fence.acq_rel.cta()

        while work_linear_tma < total_work:
            work_u32_tma: T.uint32 = T.cast(work_linear_tma, "uint32")
            remain_u32_tma: T.uint32 = _udiv_u32_const(work_u32_tma, HV)
            head_tma: T.int32 = T.cast(
                work_u32_tma - remain_u32_tma * T.uint32(output_heads), "int32"
            )
            qk_head_tma: T.int32 = head_tma
            value_subhead_tma: T.int32 = 0
            if HQ != HV:
                qk_head_tma = T.cast(_udiv_u32_const(T.cast(head_tma, "uint32"), HV // HQ), "int32")
                value_subhead_tma = head_tma - qk_head_tma * (HV // HQ)
            remain_tma: T.int32 = T.cast(remain_u32_tma, "int32")
            batch_tma: T.int32 = remain_tma
            batch_start_tma: T.int64 = T.cast(cu_seqlens[batch_tma], "int64")
            batch_end_tma: T.int64 = T.cast(cu_seqlens[batch_tma + 1], "int64")
            seqlen_tma: T.int32 = T.cast(batch_end_tma - batch_start_tma, "int32")
            num_valid_chunks_tma: T.int32 = (seqlen_tma + CHUNK - 1) // CHUNK
            num_pairs_tma: T.int32 = (seqlen_tma + PAIR_TOKENS - 1) // PAIR_TOKENS
            padded_chunks_tma: T.int32 = num_pairs_tma * 2

            if padded_chunks_tma > 0:
                if T.cuda.elect_sync():
                    T.ptx.cp.async_.bulk.wait_group.read(0)
                T.cuda.warp_sync()
                if T.cuda.elect_sync():
                    _replace_descriptor(
                        descriptor_q, q.data, batch_end_tma, HQ, 1, 2 * D_HEAD * HQ, 2 * D_HEAD, 0
                    )
                    _replace_descriptor(
                        descriptor_k, k.data, batch_end_tma, HQ, 1, 2 * D_HEAD * HQ, 2 * D_HEAD, 0
                    )
                    if HQ == HV:
                        _replace_descriptor(
                            descriptor_v,
                            v.data,
                            batch_end_tma,
                            HV,
                            1,
                            2 * D_HEAD * HV,
                            2 * D_HEAD,
                            0,
                        )
                    else:
                        _replace_descriptor(
                            descriptor_v,
                            v.data,
                            batch_end_tma,
                            HV // HQ,
                            HQ,
                            2 * D_HEAD * HV,
                            2 * D_HEAD,
                            2 * D_HEAD * (HV // HQ),
                        )
                T.cuda.warp_sync()
                _tensormap_release()

                # CK:L1659-L1703: all-but-last valid, explicit last valid,
                # then even-pair padding; each call itself is K -> Q -> V.
                for chunk_idx_tma in T.serial(0, num_valid_chunks_tma - 1):
                    chunk_offset_tma: T.int64 = batch_start_tma + T.cast(
                        chunk_idx_tma * CHUNK, "int64"
                    )
                    _load_qkv_chunk(
                        smem_addr_tma,
                        descriptor_q,
                        descriptor_k,
                        descriptor_v,
                        chunk_offset_tma,
                        chunk_idx_tma,
                        head_tma,
                        qk_head_tma,
                        value_subhead_tma,
                        q_producer_index,
                        q_producer_phase,
                        k_producer_index,
                        k_producer_phase,
                        v_producer_index,
                        v_producer_phase,
                        HQ=HQ,
                        HV=HV,
                    )
                    q_producer_phase = _pipe_next_phase(q_producer_index, q_producer_phase, 2)
                    q_producer_index = _pipe_next_index(q_producer_index, 2)
                    k_producer_phase = _pipe_next_phase(k_producer_index, k_producer_phase, 4)
                    k_producer_index = _pipe_next_index(k_producer_index, 4)
                    previous_v_index_tma: T.int32 = v_producer_index
                    previous_v_phase_tma: T.int32 = v_producer_phase
                    v_producer_index = _pipe_next_index(previous_v_index_tma, 3)
                    v_producer_phase = _pipe_next_phase(
                        previous_v_index_tma, previous_v_phase_tma, 3
                    )

                last_chunk_tma: T.int32 = num_valid_chunks_tma - 1
                last_offset_tma: T.int64 = batch_start_tma + T.cast(last_chunk_tma * CHUNK, "int64")
                _load_qkv_chunk(
                    smem_addr_tma,
                    descriptor_q,
                    descriptor_k,
                    descriptor_v,
                    last_offset_tma,
                    last_chunk_tma,
                    head_tma,
                    qk_head_tma,
                    value_subhead_tma,
                    q_producer_index,
                    q_producer_phase,
                    k_producer_index,
                    k_producer_phase,
                    v_producer_index,
                    v_producer_phase,
                    HQ=HQ,
                    HV=HV,
                )
                q_producer_phase = _pipe_next_phase(q_producer_index, q_producer_phase, 2)
                q_producer_index = _pipe_next_index(q_producer_index, 2)
                k_producer_phase = _pipe_next_phase(k_producer_index, k_producer_phase, 4)
                k_producer_index = _pipe_next_index(k_producer_index, 4)
                previous_v_index_tma: T.int32 = v_producer_index
                previous_v_phase_tma: T.int32 = v_producer_phase
                v_producer_index = _pipe_next_index(previous_v_index_tma, 3)
                v_producer_phase = _pipe_next_phase(previous_v_index_tma, previous_v_phase_tma, 3)

                for chunk_idx_tma in T.serial(num_valid_chunks_tma, padded_chunks_tma):
                    chunk_offset_tma: T.int64 = batch_start_tma + T.cast(
                        chunk_idx_tma * CHUNK, "int64"
                    )
                    _load_qkv_chunk(
                        smem_addr_tma,
                        descriptor_q,
                        descriptor_k,
                        descriptor_v,
                        chunk_offset_tma,
                        chunk_idx_tma,
                        head_tma,
                        qk_head_tma,
                        value_subhead_tma,
                        q_producer_index,
                        q_producer_phase,
                        k_producer_index,
                        k_producer_phase,
                        v_producer_index,
                        v_producer_phase,
                        HQ=HQ,
                        HV=HV,
                    )
                    q_producer_phase = _pipe_next_phase(q_producer_index, q_producer_phase, 2)
                    q_producer_index = _pipe_next_index(q_producer_index, 2)
                    k_producer_phase = _pipe_next_phase(k_producer_index, k_producer_phase, 4)
                    k_producer_index = _pipe_next_index(k_producer_index, 4)
                    previous_v_index_tma: T.int32 = v_producer_index
                    previous_v_phase_tma: T.int32 = v_producer_phase
                    v_producer_index = _pipe_next_index(previous_v_index_tma, 3)
                    v_producer_phase = _pipe_next_phase(
                        previous_v_index_tma, previous_v_phase_tma, 3
                    )
            work_linear_tma = work_linear_tma + grid_x

        # CK:L1707-L1709: producer tails are Q, K, then V in source order.
        _producer_tail_state(smem_addr_tma, 80, q_producer_index, q_producer_phase, 2)
        _producer_tail_state(smem_addr_tma, 32, k_producer_index, k_producer_phase, 4)
        _producer_tail_state(smem_addr_tma, 120, v_producer_index, v_producer_phase, 3)

    # This is an independent source-level if, not part of the chain above.
    if warp == 11:
        smem_addr_epi: T.uint32 = T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0]))
        T.ptx.setmaxnreg.dec.sync.aligned.u32(24)
        gate_producer_index: T.int32 = 0
        gate_producer_phase: T.int32 = 1
        beta_producer_index: T.int32 = 0
        beta_producer_phase: T.int32 = 1
        o_consumer_index: T.int32 = 0
        o_consumer_phase: T.int32 = 0
        work_linear_epi: T.int32 = block

        # CK:L1721-L1728: O payload initialization is independent of Q/K/V.
        if work_linear_epi < total_work:
            if T.cuda.elect_sync():
                _descriptor_copy_payload(o_map, descriptor_o)
            T.cuda.warp_sync()
            T.ptx.fence.acq_rel.cta()

        while work_linear_epi < total_work:
            work_u32_epi: T.uint32 = T.cast(work_linear_epi, "uint32")
            remain_u32_epi: T.uint32 = _udiv_u32_const(work_u32_epi, HV)
            head_epi: T.int32 = T.cast(
                work_u32_epi - remain_u32_epi * T.uint32(output_heads), "int32"
            )
            qk_head_epi: T.int32 = head_epi
            value_subhead_epi: T.int32 = 0
            if HQ != HV:
                qk_head_epi = T.cast(_udiv_u32_const(T.cast(head_epi, "uint32"), HV // HQ), "int32")
                value_subhead_epi = head_epi - qk_head_epi * (HV // HQ)
            remain_epi: T.int32 = T.cast(remain_u32_epi, "int32")
            batch_epi: T.int32 = remain_epi
            batch_start_epi: T.int64 = T.cast(cu_seqlens[batch_epi], "int64")
            batch_end_epi: T.int64 = T.cast(cu_seqlens[batch_epi + 1], "int64")
            seqlen_epi: T.int32 = T.cast(batch_end_epi - batch_start_epi, "int32")
            num_valid_chunks_epi: T.int32 = (seqlen_epi + CHUNK - 1) // CHUNK
            num_pairs_epi: T.int32 = (seqlen_epi + PAIR_TOKENS - 1) // PAIR_TOKENS
            padded_chunks_epi: T.int32 = num_pairs_epi * 2

            if padded_chunks_epi > 0:
                if T.cuda.elect_sync():
                    T.ptx.cp.async_.bulk.wait_group.read(0)
                T.cuda.warp_sync()
                if T.cuda.elect_sync():
                    if HQ == HV:
                        _replace_descriptor(
                            descriptor_o,
                            o.data,
                            batch_end_epi,
                            HV,
                            1,
                            2 * D_HEAD * HV,
                            2 * D_HEAD,
                            0,
                        )
                    else:
                        _replace_descriptor(
                            descriptor_o,
                            o.data,
                            batch_end_epi,
                            HV // HQ,
                            HQ,
                            2 * D_HEAD * HV,
                            2 * D_HEAD,
                            2 * D_HEAD * (HV // HQ),
                        )
                T.cuda.warp_sync()
                _tensormap_release()
                if T.cuda.elect_sync():
                    _tensormap_acquire(descriptor_o)

                # CK:L1762-L1812: exact fixed lookahead (0,1), then (2,3),
                # followed by chunk+4 immediately before the current O store.
                for prefetch_idx_epi in T.unroll(0, 2):
                    prefetch_offset_epi: T.int64 = batch_start_epi + T.cast(
                        prefetch_idx_epi * CHUNK, "int64"
                    )
                    _load_gate_beta_chunk(
                        smem_raw,
                        smem_addr_epi,
                        s_cumsumlog,
                        s_cumprod,
                        s_beta,
                        gate,
                        beta,
                        prefetch_offset_epi,
                        head_epi,
                        prefetch_idx_epi >= num_valid_chunks_epi - 1,
                        batch_end_epi,
                        lane,
                        gate_producer_index,
                        gate_producer_phase,
                        beta_producer_index,
                        beta_producer_phase,
                        HV=HV,
                    )
                    gate_producer_phase = _pipe_next_phase(
                        gate_producer_index, gate_producer_phase, 5
                    )
                    gate_producer_index = _pipe_next_index(gate_producer_index, 5)
                    beta_producer_phase = _pipe_next_phase(
                        beta_producer_index, beta_producer_phase, 5
                    )
                    beta_producer_index = _pipe_next_index(beta_producer_index, 5)

                if padded_chunks_epi > 2:
                    for prefetch_idx_epi in T.unroll(2, 4):
                        prefetch_offset_epi: T.int64 = batch_start_epi + T.cast(
                            prefetch_idx_epi * CHUNK, "int64"
                        )
                        _load_gate_beta_chunk(
                            smem_raw,
                            smem_addr_epi,
                            s_cumsumlog,
                            s_cumprod,
                            s_beta,
                            gate,
                            beta,
                            prefetch_offset_epi,
                            head_epi,
                            prefetch_idx_epi >= num_valid_chunks_epi - 1,
                            batch_end_epi,
                            lane,
                            gate_producer_index,
                            gate_producer_phase,
                            beta_producer_index,
                            beta_producer_phase,
                            HV=HV,
                        )
                        gate_producer_phase = _pipe_next_phase(
                            gate_producer_index, gate_producer_phase, 5
                        )
                        gate_producer_index = _pipe_next_index(gate_producer_index, 5)
                        beta_producer_phase = _pipe_next_phase(
                            beta_producer_index, beta_producer_phase, 5
                        )
                        beta_producer_index = _pipe_next_index(beta_producer_index, 5)

                for chunk_idx_epi in T.serial(0, padded_chunks_epi):
                    chunk_offset_epi: T.int64 = batch_start_epi + T.cast(
                        chunk_idx_epi * CHUNK, "int64"
                    )
                    prefetch_idx_epi: T.int32 = chunk_idx_epi + 4
                    if prefetch_idx_epi < padded_chunks_epi:
                        prefetch_offset_epi: T.int64 = chunk_offset_epi + 4 * CHUNK
                        _load_gate_beta_chunk(
                            smem_raw,
                            smem_addr_epi,
                            s_cumsumlog,
                            s_cumprod,
                            s_beta,
                            gate,
                            beta,
                            prefetch_offset_epi,
                            head_epi,
                            prefetch_idx_epi >= num_valid_chunks_epi - 1,
                            batch_end_epi,
                            lane,
                            gate_producer_index,
                            gate_producer_phase,
                            beta_producer_index,
                            beta_producer_phase,
                            HV=HV,
                        )
                        gate_producer_phase = _pipe_next_phase(
                            gate_producer_index, gate_producer_phase, 5
                        )
                        gate_producer_index = _pipe_next_index(gate_producer_index, 5)
                        beta_producer_phase = _pipe_next_phase(
                            beta_producer_index, beta_producer_phase, 5
                        )
                        beta_producer_index = _pipe_next_index(beta_producer_index, 5)

                    _store_o_chunk(
                        smem_addr_epi,
                        descriptor_o,
                        chunk_offset_epi,
                        head_epi,
                        qk_head_epi,
                        value_subhead_epi,
                        o_consumer_index,
                        o_consumer_phase,
                        HQ=HQ,
                        HV=HV,
                    )
                    previous_o_index_epi: T.int32 = o_consumer_index
                    previous_o_phase_epi: T.int32 = o_consumer_phase
                    o_consumer_index = _pipe_next_index(previous_o_index_epi, 2)
                    o_consumer_phase = _pipe_next_phase(
                        previous_o_index_epi, previous_o_phase_epi, 2
                    )
            work_linear_epi = work_linear_epi + grid_x

        _producer_tail_state(smem_addr_epi, 184, gate_producer_index, gate_producer_phase, 5)
        _producer_tail_state(smem_addr_epi, 264, beta_producer_index, beta_producer_phase, 5)


def _cfg(**kwargs: Any) -> GDNPrefillSM100Config:
    cfg_fields = {field.name for field in fields(GDNPrefillSM100Config)}
    cfg_kwargs = {key: value for key, value in kwargs.items() if key in cfg_fields}
    if "seq_lens" in cfg_kwargs:
        cfg_kwargs["seq_lens"] = tuple(int(length) for length in cfg_kwargs["seq_lens"])
    if "label" not in cfg_kwargs:
        cfg_kwargs["label"] = "custom"
    cfg = GDNPrefillSM100Config(**cfg_kwargs)
    cfg.validate()
    return cfg


class _AlignedTensorMap:
    """Host storage for one 64-byte-aligned, 128-byte TensorMap payload."""

    def __init__(self) -> None:
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_tensor_map(
    tensor: torch.Tensor, *, global_dims: tuple[int, ...], global_strides_bytes: tuple[int, ...]
) -> _AlignedTensorMap:
    """Encode the frozen 64x64 FP16, B128-swizzled TMA tile."""
    import tvm

    rank = len(global_dims)
    if rank not in (3, 4):
        raise ValueError(f"GDN TensorMap rank must be 3 or 4, got {rank}")
    if len(global_strides_bytes) != rank - 1:
        raise ValueError("TensorMap global stride count must be rank - 1")

    descriptor = _AlignedTensorMap()
    box_dims = (64, 64, *((1,) * (rank - 2)))
    element_strides = (1,) * rank
    encode = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    encode(
        descriptor.ptr,
        "float16",
        rank,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *global_dims,
        *global_strides_bytes,
        *box_dims,
        *element_strides,
        0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
        3,  # CU_TENSOR_MAP_SWIZZLE_128B
        3,  # CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        0,  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    )
    return descriptor


def _build_tensor_maps(case: dict[str, Any]) -> dict[str, _AlignedTensorMap]:
    """Build Q/K/V/O descriptors with the exact frozen logical coordinates."""
    cfg: GDNPrefillSM100Config = case["config"]
    qk_dims = (D_HEAD, cfg.total_tokens, cfg.hq)
    qk_strides = (2 * D_HEAD * cfg.hq, 2 * D_HEAD)

    if cfg.hq == cfg.hv:
        vo_dims = (D_HEAD, cfg.total_tokens, cfg.hv)
        vo_strides = (2 * D_HEAD * cfg.hv, 2 * D_HEAD)
    else:
        value_heads_per_q_head = cfg.hv // cfg.hq
        vo_dims = (D_HEAD, cfg.total_tokens, value_heads_per_q_head, cfg.hq)
        vo_strides = (2 * D_HEAD * cfg.hv, 2 * D_HEAD, 2 * D_HEAD * value_heads_per_q_head)

    return {
        "q": _encode_tensor_map(case["q"], global_dims=qk_dims, global_strides_bytes=qk_strides),
        "k": _encode_tensor_map(case["k"], global_dims=qk_dims, global_strides_bytes=qk_strides),
        "v": _encode_tensor_map(case["v"], global_dims=vo_dims, global_strides_bytes=vo_strides),
        "o": _encode_tensor_map(case["o"], global_dims=vo_dims, global_strides_bytes=vo_strides),
    }


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    tensor_maps = case["tensor_maps"]
    return (
        case["q"].view(-1),
        case["k"].view(-1),
        case["v"].view(-1),
        case["gate"].view(-1),
        case["beta"].view(-1),
        case["o"].view(-1),
        case["cu_seqlens"],
        case["initial_state"].view(-1),
        case["final_state"].view(-1),
        tensor_maps["q"].ptr,
        tensor_maps["k"].ptr,
        tensor_maps["v"].ptr,
        tensor_maps["o"].ptr,
        case["descriptor_workspace"],
        case["total_tokens"],
        case["num_sequences"],
        case["num_sms"],
        case["scale"],
    )


@lru_cache(maxsize=1)
def _load_frozen_oracle():
    from flashinfer.gdn_kernels.blackwell.gdn_prefill import chunk_gated_delta_rule_sm100

    adapter_path = Path(inspect.getfile(chunk_gated_delta_rule_sm100)).resolve()
    kernel_path = adapter_path.with_name("gated_delta_net_chunked.py")
    adapter_sha256 = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    kernel_sha256 = hashlib.sha256(kernel_path.read_bytes()).hexdigest()
    if (
        adapter_sha256 != FROZEN_FLASHINFER_ADAPTER_SHA256
        or kernel_sha256 != FROZEN_FLASHINFER_KERNEL_SHA256
    ):
        raise RuntimeError(
            "GDN oracle does not match the frozen FlashInfer sources: "
            f"adapter={adapter_path} sha256={adapter_sha256}, "
            f"kernel={kernel_path} sha256={kernel_sha256}"
        )
    return chunk_gated_delta_rule_sm100


def _run_frozen_oracle(
    case: dict[str, Any], output: torch.Tensor, final_state: torch.Tensor
) -> None:
    oracle = _load_frozen_oracle()
    oracle(
        case["q"],
        case["k"],
        case["v"],
        case["gate"],
        case["beta"],
        output,
        case["cu_seqlens"],
        case["initial_state"],
        final_state,
        case["scale"],
    )


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    cfg = _cfg(**kwargs)
    device = kwargs.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for GDN prefill SM100")
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    q = torch.randn(
        (cfg.total_tokens, cfg.hq, D_HEAD), dtype=torch.float16, device=device, generator=generator
    )
    k = F.normalize(
        torch.randn(
            (cfg.total_tokens, cfg.hq, D_HEAD),
            dtype=torch.float32,
            device=device,
            generator=generator,
        ),
        p=2,
        dim=-1,
    ).to(torch.float16)
    v = torch.randn(
        (cfg.total_tokens, cfg.hv, D_HEAD), dtype=torch.float16, device=device, generator=generator
    )
    gate = torch.rand(
        (cfg.total_tokens, cfg.hv), dtype=torch.float32, device=device, generator=generator
    )
    beta = torch.rand(
        (cfg.total_tokens, cfg.hv), dtype=torch.float32, device=device, generator=generator
    ).sigmoid()
    initial_state = torch.randn(
        (cfg.num_sequences, cfg.hv, D_HEAD, D_HEAD),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    final_state = torch.zeros_like(initial_state)
    o = torch.empty_like(v)

    endpoints = [0]
    for length in cfg.seq_lens:
        endpoints.append(endpoints[-1] + length)
    cu_seqlens = torch.tensor(endpoints, dtype=torch.int32, device=device)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    descriptor_workspace = torch.empty(
        (num_sms * DESCRIPTOR_BYTES_PER_CTA,), dtype=torch.int8, device=device
    )

    case = {
        "config": cfg,
        "q": q,
        "k": k,
        "v": v,
        "gate": gate,
        "beta": beta,
        "o": o,
        "cu_seqlens": cu_seqlens,
        "initial_state": initial_state,
        "final_state": final_state,
        "descriptor_workspace": descriptor_workspace,
        "total_tokens": cfg.total_tokens,
        "num_sequences": cfg.num_sequences,
        "num_sms": num_sms,
        "scale": 1.0 / math.sqrt(D_HEAD),
    }
    case["tensor_maps"] = _build_tensor_maps(case)
    return case


def get_kernel(**kwargs: Any):
    cfg = _cfg(**kwargs)
    return _kernel.specialize(HQ=cfg.hq, HV=cfg.hv).with_attr(
        "tirx.kernel_launch_params", list(LAUNCH_TAGS)
    )


def run_test(**kwargs: Any) -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for GDN prefill SM100")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        raise SkipTest(f"GDN prefill SM100 requires compute capability 10.x, got {capability}")

    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    executable = compile_kernel(get_kernel(**kwargs))
    executable(*_tirx_args(case))
    torch.cuda.synchronize()

    reference_o = torch.empty_like(case["o"])
    reference_state = torch.empty_like(case["final_state"])
    _run_frozen_oracle(case, reference_o, reference_state)
    torch.cuda.synchronize()

    for name, tensor in (
        ("tirx output", case["o"]),
        ("tirx final state", case["final_state"]),
        ("frozen output", reference_o),
        ("frozen final state", reference_state),
    ):
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{name} contains non-finite values")

    torch.testing.assert_close(case["o"], reference_o, atol=2e-3, rtol=1e-3)
    torch.testing.assert_close(case["final_state"], reference_state, atol=1e-3, rtol=1e-4)
    case["config"].validate()


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    rounds = kwargs.pop("rounds", 5)
    cooldown_s = kwargs.pop("cooldown_s", 1.0)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for GDN prefill SM100 benchmark")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        raise SkipTest(f"GDN prefill SM100 requires compute capability 10.x, got {capability}")

    from tirx_kernels.runner import compile_kernel
    from tvm.tirx.bench import bench

    case = prepare_data(**kwargs)
    executable = compile_kernel(get_kernel(**kwargs))
    args = _tirx_args(case)

    def _flashinfer_cutedsl_builder():
        reference_o = torch.empty_like(case["o"])
        reference_state = torch.empty_like(case["final_state"])

        # First execution performs CuTeDSL JIT and initializes its persistent
        # workspace; subsequent executions are launch-only warmups.
        _run_frozen_oracle(case, reference_o, reference_state)
        for _ in range(2):
            _run_frozen_oracle(case, reference_o, reference_state)
        torch.cuda.synchronize()

        def launch():
            _run_frozen_oracle(case, reference_o, reference_state)

        return launch

    return bench(
        {"tirx": lambda: executable(*args)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashinfer_cutedsl": _flashinfer_cutedsl_builder},
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


__all__ = ["CONFIGS", "KERNEL_META", "get_kernel", "prepare_data", "run_bench", "run_test"]
