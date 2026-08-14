# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400)
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM100a FP16 GDN prefill transcribed from FlashInfer's frozen CuTeDSL kernel.

The implementation follows the frozen CuTeDSL source order and preserves its
CUDA/PTX details instead of reorganizing them into a separate implementation.

Upstream source: flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Any
from unittest import SkipTest

import torch
import torch.nn.functional as F

import tvm
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as T

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


def _builder_assign(name: str, value):
    """Match parser assignment binding for one tuple/list member."""
    if isinstance(value, tvm.ir.Var):
        try:
            IRBuilder.name(name, value)
        except (TypeError, ValueError):
            pass
        return value
    if isinstance(value, tvm.tirx.Buffer):
        return _builder_name(name, value)
    if not tvm.ir.is_prim_expr(value) and not isinstance(value, tvm.ir.Expr):
        return value
    return _builder_scalar(name, value)


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


KERNEL_META = {"name": "gdn_prefill_sm100", "category": "flashinfer", "compute_capability": 10}

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
CG1_TMEM_DEALLOC_BARRIER = 3
INITIAL_STATE_BARRIER = 4

LAUNCH_TAGS = ("blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory")


def _init_pipeline(smem_raw, full_off, empty_off, stages, producers, consumers):
    with T.serial(stages) as stage:
        _builder_emit(
            T.ptx.mbarrier.init.shared.b64(
                smem_raw.ptr_to([full_off + stage * 8]), T.uint32(producers)
            )
        )
        _builder_emit(
            T.ptx.mbarrier.init.shared.b64(
                smem_raw.ptr_to([empty_off + stage * 8]), T.uint32(consumers)
            )
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


def _load_sequence_bounds(cu_seqlens, batch, bounds):
    _builder_emit(T.ptx.ld.global_.s32(bounds[0], cu_seqlens.ptr_to([batch])))
    _builder_emit(T.ptx.ld.global_.s32(bounds[1], cu_seqlens.ptr_to([batch + 1])))


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


def _producer_tail(smem_base_addr, empty_off, count, stages):
    with T.serial(stages) as tail_step:
        tail_count = _builder_scalar("tail_count", count + tail_step, dtype="int32")
        _builder_emit(_producer_acquire(smem_base_addr, empty_off, tail_count, stages))


def _producer_tail_state(smem_base_addr, empty_off, index, phase, stages):
    tail_index = _builder_scalar("tail_index", index, dtype="int32")
    tail_phase = _builder_scalar("tail_phase", phase, dtype="int32")
    with T.serial(stages) as _:
        current_index = _builder_scalar("current_index", tail_index, dtype="int32")
        current_phase = _builder_scalar("current_phase", tail_phase, dtype="int32")
        _builder_emit(
            _producer_acquire_state(smem_base_addr, empty_off, current_index, current_phase)
        )
        T.buffer_store(tail_index.buffer, _pipe_next_index(current_index, stages), [0])
        T.buffer_store(
            tail_phase.buffer, _pipe_next_phase(current_index, current_phase, stages), [0]
        )


def _descriptor_copy_payload(src_map, dst_ptr):
    payload = _builder_name("payload", T.alloc_local((4,), "uint64"))
    _builder_emit(
        T.ptx.ld.global_.v4.b64(
            payload[0], payload[1], payload[2], payload[3], _byte_ptr(T.address_of(src_map), 0)
        )
    )
    _builder_emit(T.ptx.st.global_.v4.b64(dst_ptr, payload[0], payload[1], payload[2], payload[3]))
    _builder_emit(
        T.ptx.ld.global_.v4.b64(
            payload[0], payload[1], payload[2], payload[3], _byte_ptr(T.address_of(src_map), 32)
        )
    )
    _builder_emit(
        T.ptx.st.global_.v4.b64(
            _byte_ptr(dst_ptr, 32), payload[0], payload[1], payload[2], payload[3]
        )
    )


def _replace_global_address(desc, address):
    return T.ptx.tensormap_replace.tile.global_address.global_.b1024.b64(
        desc, T.reinterpret("uint64", address)
    )


def _replace_global_dim(desc, dim, value):
    return T.ptx.tensormap_replace.tile.global_dim.global_.b1024.b32(desc, dim, value)


def _replace_global_stride(desc, dim, value):
    return T.ptx.tensormap_replace.tile.global_stride.global_.b1024.b64(desc, dim, value)


def _replace_descriptor(desc, address, dim1, dim2, dim3, stride0, stride1, stride2):
    _builder_emit(_replace_global_address(desc, address))
    _builder_emit(_replace_global_dim(desc, 0, T.uint32(128)))
    _builder_emit(_replace_global_dim(desc, 1, T.cast(dim1, "uint32")))
    _builder_emit(_replace_global_stride(desc, 0, T.cast(stride0, "uint64")))
    _builder_emit(_replace_global_dim(desc, 2, T.cast(dim2, "uint32")))
    _builder_emit(_replace_global_stride(desc, 1, T.cast(stride1, "uint64")))
    _builder_emit(_replace_global_dim(desc, 3, T.cast(dim3, "uint32")))
    _builder_emit(_replace_global_stride(desc, 2, T.cast(stride2, "uint64")))
    _builder_emit(_replace_global_dim(desc, 4, T.uint32(1)))
    _builder_emit(_replace_global_stride(desc, 3, T.uint64(0)))


def _tensormap_release():
    return T.ptx.fence.proxy.tensormap__generic.release.gpu()


def _tensormap_acquire(desc):
    return T.ptx.fence.proxy.tensormap__generic.acquire.gpu(desc)


def _lg2_approx_ftz(value):
    result = T.alloc_local((1,), "float32")
    T.evaluate(T.ptx.lg2.approx.ftz.f32(result[0], value))
    return result[0]


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


def _sub_zero_pack_f16x2(a, b, dst, dst_index):
    negated = _builder_name("negated", T.alloc_local((1,), "uint64"))
    _builder_emit(
        T.ptx.sub.rn.f32x2(
            negated[0], T.cuda.make_float2(T.float32(0.0), T.float32(0.0)), T.cuda.make_float2(a, b)
        )
    )
    T.buffer_store(
        dst, _pack_f16x2(T.cuda.float2_x(negated[0]), T.cuda.float2_y(negated[0])), [dst_index]
    )


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


def _cg0_tmem_ld(tmem_base, stage, thread, values):
    row_bits = _builder_scalar(
        "row_bits", T.bitwise_and(thread << 16, T.int32(6291456)), dtype="int32"
    )
    addr = _builder_scalar(
        "addr", tmem_base + TMEM_CG0_ACC_COL + stage * 64 + row_bits, dtype="int32"
    )
    _builder_emit(
        T.ptx["tcgen05.ld.sync.aligned.16x32bx2.x16.b32"](
            *[values[i] for i in range(16)], T.uint32(addr), 16
        )
    )
    _builder_emit(
        T.ptx["tcgen05.ld.sync.aligned.16x32bx2.x16.b32"](
            *[values[16 + i] for i in range(16)], T.uint32(addr + 32), 16
        )
    )


def _cg0_store_fragment(smem_raw, base, stage, thread, values):
    row = _builder_scalar(
        "row",
        T.bitwise_or(T.bitwise_and(thread >> 1, T.int32(48)), T.bitwise_and(thread, T.int32(15))),
        dtype="int32",
    )
    col = _builder_scalar("col", T.bitwise_and(thread, T.int32(16)), dtype="int32")
    packed = _builder_name("packed", T.alloc_local((16,), "uint32"))
    with T.unroll(16) as pair:
        T.buffer_store(packed, _pack_f16x2(values[pair * 2], values[pair * 2 + 1]), [pair])
    _builder_emit(_st_shared_v4_u32(_f16_tile_ptr(smem_raw, base, stage, row, col), packed, 0))
    _builder_emit(_st_shared_v4_u32(_f16_tile_ptr(smem_raw, base, stage, row, col + 8), packed, 4))
    _builder_emit(_st_shared_v4_u32(_f16_tile_ptr(smem_raw, base, stage, row, col + 32), packed, 8))
    _builder_emit(
        _st_shared_v4_u32(_f16_tile_ptr(smem_raw, base, stage, row, col + 40), packed, 12)
    )


def _cg0_load_fragment(smem_raw, base, stage, thread, values):
    row = _builder_scalar(
        "row",
        T.bitwise_or(T.bitwise_and(thread >> 1, T.int32(48)), T.bitwise_and(thread, T.int32(15))),
        dtype="int32",
    )
    col = _builder_scalar("col", T.bitwise_and(thread, T.int32(16)), dtype="int32")
    packed = _builder_name("packed", T.alloc_local((16,), "uint32"))
    _builder_emit(_ld_shared_v4_u32(_f16_tile_ptr(smem_raw, base, stage, row, col), packed))
    loaded1 = _builder_name("loaded1", T.alloc_local((4,), "uint32"))
    loaded2 = _builder_name("loaded2", T.alloc_local((4,), "uint32"))
    loaded3 = _builder_name("loaded3", T.alloc_local((4,), "uint32"))
    _builder_emit(_ld_shared_v4_u32(_f16_tile_ptr(smem_raw, base, stage, row, col + 8), loaded1))
    _builder_emit(_ld_shared_v4_u32(_f16_tile_ptr(smem_raw, base, stage, row, col + 32), loaded2))
    _builder_emit(_ld_shared_v4_u32(_f16_tile_ptr(smem_raw, base, stage, row, col + 40), loaded3))
    with T.unroll(4) as i:
        T.buffer_store(packed, loaded1[i], [4 + i])
        T.buffer_store(packed, loaded2[i], [8 + i])
        T.buffer_store(packed, loaded3[i], [12 + i])
    with T.unroll(16) as pair:
        T.buffer_store(values, _unpack_f16_lo(packed[pair]), [pair * 2])
        T.buffer_store(values, _unpack_f16_hi(packed[pair]), [pair * 2 + 1])


def _invert_diagonal_8x8(smem_raw, stage, block8, lane):
    row_in_group = _builder_scalar("row_in_group", lane & 7, dtype="int32")
    logical_row = _builder_scalar("logical_row", block8 + row_in_group, dtype="int32")
    words = _builder_name("words", T.alloc_local((4,), "uint32"))
    row_values = _builder_name("row_values", T.alloc_local((8,), "float32"))
    _builder_emit(
        _ld_shared_v4_u32(_f16_tile_ptr(smem_raw, SMEM_AINV_OFF, stage, logical_row, block8), words)
    )
    with T.unroll(4) as pair:
        T.buffer_store(row_values, _unpack_f16_lo(words[pair]), [pair * 2])
        T.buffer_store(row_values, _unpack_f16_hi(words[pair]), [pair * 2 + 1])
    with T.unroll(8) as i:
        _builder_if_12_8 = _builder_scope_enter(T.If(row_in_group == i))
        _builder_then_12_8 = _builder_scope_enter(T.Then())
        T.buffer_store(row_values, T.float32(1.0), [i])
        _builder_scope_exit(_builder_then_12_8)
        _builder_scope_exit(_builder_if_12_8)
    with T.unroll(7) as src_row:
        row_scale = _builder_alloc_scalar("row_scale", "float32")
        _builder_emit(T.ptx.neg.f32(row_scale, row_values[src_row]))
        with T.unroll(7) as i:
            _builder_if_20_12 = _builder_scope_enter(T.If(i < src_row))
            _builder_then_20_12 = _builder_scope_enter(T.Then())
            pivot = _builder_scalar(
                "pivot",
                T.cuda._shfl_sync(T.uint32(4294967295), row_values[i], src_row, 8),
                dtype="float32",
            )
            _builder_if_24_16 = _builder_scope_enter(T.If(row_in_group > src_row))
            _builder_then_24_16 = _builder_scope_enter(T.Then())
            T.buffer_store(row_values, row_values[i] + row_scale * pivot, [i])
            _builder_scope_exit(_builder_then_24_16)
            _builder_scope_exit(_builder_if_24_16)
            _builder_scope_exit(_builder_then_20_12)
            _builder_scope_exit(_builder_if_20_12)
        _builder_if_26_8 = _builder_scope_enter(T.If(row_in_group > src_row))
        _builder_then_26_8 = _builder_scope_enter(T.Then())
        T.buffer_store(row_values, row_scale, [src_row])
        _builder_scope_exit(_builder_then_26_8)
        _builder_scope_exit(_builder_if_26_8)
    with T.unroll(4) as pair:
        T.buffer_store(words, _pack_f16x2(row_values[pair * 2], row_values[pair * 2 + 1]), [pair])
    _builder_emit(
        _st_shared_v4_u32(
            _f16_tile_ptr(smem_raw, SMEM_AINV_OFF, stage, logical_row, block8), words, 0
        )
    )


def _inverse_8_to_16(smem_raw, stage, block16, lane):
    a = _builder_name("a", T.alloc_local((2,), "uint32"))
    b = _builder_name("b", T.alloc_local((1,), "uint32"))
    acc = _builder_name("acc", T.alloc_local((4,), "float32"))
    d_word = _builder_name("d_word", T.alloc_local((1,), "uint32"))
    c_word = _builder_name("c_word", T.alloc_local((1,), "uint32"))
    _builder_emit(
        T.ptx.ldmatrix.sync.aligned.m8n8.x1.shared.b16(
            d_word[0],
            _f16_tile_ptr(smem_raw, SMEM_AINV_OFF, stage, block16 + 8 + (lane & 7), block16 + 8),
        )
    )
    _builder_emit(
        T.ptx.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16(
            c_word[0],
            _f16_tile_ptr(smem_raw, SMEM_AINV_OFF, stage, block16 + 8 + (lane & 7), block16),
        )
    )
    T.buffer_store(a, d_word[0], [0])
    T.buffer_store(a, d_word[0], [1])
    T.buffer_store(b, c_word[0], [0])
    _builder_emit(_mma_m16n8k8_f16_zero(acc, a, b))
    _builder_emit(_sub_zero_pack_f16x2(acc[0], acc[1], a, 0))
    _builder_emit(_sub_zero_pack_f16x2(acc[2], acc[3], a, 1))
    _builder_emit(
        T.ptx.ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16(
            b[0], _f16_tile_ptr(smem_raw, SMEM_AINV_OFF, stage, block16 + (lane & 7), block16)
        )
    )
    _builder_emit(_mma_m16n8k8_f16_zero(acc, a, b))
    T.buffer_store(d_word, _pack_f16x2(acc[0], acc[1]), [0])
    _builder_emit(
        T.ptx.stmatrix.sync.aligned.m8n8.x1.shared.b16(
            _f16_tile_ptr(smem_raw, SMEM_AINV_OFF, stage, block16 + 8 + (lane & 7), block16),
            d_word[0],
        )
    )


def _ldmatrix_x4_ptr_coords(base_row, base_col, lane, transpose):
    # CK:L3738-L3739/L3897-L3898: the .trans instruction modifier changes
    # each matrix's load semantics, not the lane-group-to-matrix address map.
    # Both forms use g1=bottom-left and g2=top-right in the 16x16 tile.
    lane_matrix: T.int32 = lane >> 3
    row: T.int32 = base_row + (lane & 7) + (lane_matrix & 1) * 8
    col: T.int32 = base_col + (lane_matrix >> 1) * 8
    return row, col


def _ldmatrix_x4_tile(smem_raw, stage, base_row, base_col, lane, transpose, dst):
    _builder_values_2 = _ldmatrix_x4_ptr_coords(base_row, base_col, lane, transpose)
    row, col = _builder_values_2
    row = _builder_assign("row", row)
    col = _builder_assign("col", col)
    _builder_emit(
        T.ptx[f"ldmatrix.sync.aligned.m8n8.x4{('.trans' if transpose else '')}.shared.b16"](
            *[dst[i] for i in range(4)], _f16_tile_ptr(smem_raw, SMEM_AINV_OFF, stage, row, col)
        )
    )


def _stmatrix_x4_tile(smem_raw, stage, base_row, base_col, lane, src):
    _builder_values_5 = _ldmatrix_x4_ptr_coords(base_row, base_col, lane, False)
    row, col = _builder_values_5
    row = _builder_assign("row", row)
    col = _builder_assign("col", col)
    _builder_emit(
        T.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
            _f16_tile_ptr(smem_raw, SMEM_AINV_OFF, stage, row, col), *[src[i] for i in range(4)]
        )
    )


def _inverse_16_to_32(smem_raw, stage, block32, lane):
    a = _builder_name("a", T.alloc_local((4,), "uint32"))
    b = _builder_name("b", T.alloc_local((4,), "uint32"))
    acc = _builder_name("acc", T.alloc_local((8,), "float32"))
    packed = _builder_name("packed", T.alloc_local((4,), "uint32"))
    _builder_emit(_ldmatrix_x4_tile(smem_raw, stage, block32 + 16, block32 + 16, lane, False, a))
    _builder_emit(_ldmatrix_x4_tile(smem_raw, stage, block32 + 16, block32, lane, True, b))
    _builder_emit(_mma_m16n8k16_f16_zero(acc, a, b, 0, 0))
    _builder_emit(_mma_m16n8k16_f16_zero(acc, a, b, 4, 2))
    with T.unroll(4) as pair:
        _builder_emit(_sub_zero_pack_f16x2(acc[pair * 2], acc[pair * 2 + 1], a, pair))
    _builder_emit(_ldmatrix_x4_tile(smem_raw, stage, block32, block32, lane, True, b))
    _builder_emit(_mma_m16n8k16_f16_zero(acc, a, b, 0, 0))
    _builder_emit(_mma_m16n8k16_f16_zero(acc, a, b, 4, 2))
    with T.unroll(4) as pair:
        T.buffer_store(packed, _pack_f16x2(acc[pair * 2], acc[pair * 2 + 1]), [pair])
    _builder_emit(_stmatrix_x4_tile(smem_raw, stage, block32 + 16, block32, lane, packed))


def _inverse_32_to_64(smem_raw, stage, local_warp, lane):
    row_base = _builder_scalar("row_base", 32 + local_warp * 16, dtype="int32")
    a0 = _builder_name("a0", T.alloc_local((4,), "uint32"))
    a1 = _builder_name("a1", T.alloc_local((4,), "uint32"))
    b00 = _builder_name("b00", T.alloc_local((4,), "uint32"))
    b01 = _builder_name("b01", T.alloc_local((4,), "uint32"))
    b10 = _builder_name("b10", T.alloc_local((4,), "uint32"))
    b11 = _builder_name("b11", T.alloc_local((4,), "uint32"))
    acc = _builder_name("acc", T.alloc_local((16,), "float32"))
    packed_a = _builder_name("packed_a", T.alloc_local((8,), "uint32"))
    packed_o0 = _builder_name("packed_o0", T.alloc_local((4,), "uint32"))
    packed_o1 = _builder_name("packed_o1", T.alloc_local((4,), "uint32"))
    _builder_emit(_ldmatrix_x4_tile(smem_raw, stage, row_base, 32, lane, False, a0))
    _builder_emit(_ldmatrix_x4_tile(smem_raw, stage, row_base, 48, lane, False, a1))
    _builder_emit(_ldmatrix_x4_tile(smem_raw, stage, 32, 0, lane, True, b00))
    _builder_emit(_ldmatrix_x4_tile(smem_raw, stage, 32, 16, lane, True, b01))
    _builder_emit(_ldmatrix_x4_tile(smem_raw, stage, 48, 0, lane, True, b10))
    _builder_emit(_ldmatrix_x4_tile(smem_raw, stage, 48, 16, lane, True, b11))
    _builder_emit(_mma_m16n8k16_f16_zero(acc, a0, b00, 0, 0))
    _builder_emit(_mma_m16n8k16_f16_zero(acc, a0, b00, 4, 2))
    _builder_emit(_mma_m16n8k16_f16_zero(acc, a0, b01, 8, 0))
    _builder_emit(_mma_m16n8k16_f16_zero(acc, a0, b01, 12, 2))
    _builder_emit(_mma_m16n8k16_f16_acc(acc, a1, b10, 0, 0))
    _builder_emit(_mma_m16n8k16_f16_acc(acc, a1, b10, 4, 2))
    _builder_emit(_mma_m16n8k16_f16_acc(acc, a1, b11, 8, 0))
    _builder_emit(_mma_m16n8k16_f16_acc(acc, a1, b11, 12, 2))
    with T.unroll(8) as pair:
        _builder_emit(_sub_zero_pack_f16x2(acc[pair * 2], acc[pair * 2 + 1], packed_a, pair))
    _builder_emit(_ldmatrix_x4_tile(smem_raw, stage, 0, 0, lane, True, b00))
    _builder_emit(_ldmatrix_x4_tile(smem_raw, stage, 0, 16, lane, True, b01))
    _builder_emit(_ldmatrix_x4_tile(smem_raw, stage, 16, 0, lane, True, b10))
    _builder_emit(_ldmatrix_x4_tile(smem_raw, stage, 16, 16, lane, True, b11))
    with T.unroll(4) as i:
        T.buffer_store(a0, packed_a[i], [i])
        T.buffer_store(a1, packed_a[4 + i], [i])
    _builder_emit(_mma_m16n8k16_f16_zero(acc, a0, b00, 0, 0))
    _builder_emit(_mma_m16n8k16_f16_zero(acc, a0, b00, 4, 2))
    _builder_emit(_mma_m16n8k16_f16_zero(acc, a0, b01, 8, 0))
    _builder_emit(_mma_m16n8k16_f16_zero(acc, a0, b01, 12, 2))
    _builder_emit(_mma_m16n8k16_f16_acc(acc, a1, b10, 0, 0))
    _builder_emit(_mma_m16n8k16_f16_acc(acc, a1, b10, 4, 2))
    _builder_emit(_mma_m16n8k16_f16_acc(acc, a1, b11, 8, 0))
    _builder_emit(_mma_m16n8k16_f16_acc(acc, a1, b11, 12, 2))
    with T.unroll(4) as pair:
        T.buffer_store(packed_o0, _pack_f16x2(acc[pair * 2], acc[pair * 2 + 1]), [pair])
        T.buffer_store(packed_o1, _pack_f16x2(acc[8 + pair * 2], acc[8 + pair * 2 + 1]), [pair])
    _builder_emit(T.ptx.bar.sync(T.uint32(INVERSE_BARRIER), T.uint32(128)))
    _builder_emit(_stmatrix_x4_tile(smem_raw, stage, row_base, 0, lane, packed_o0))
    _builder_emit(_stmatrix_x4_tile(smem_raw, stage, row_base, 16, lane, packed_o1))


def _cg1_tmem_row_bits(thread):
    return T.bitwise_and(thread << 16, T.int32(0x600000))


def _state_tmem_ld_sub(tmem_base, thread, sub, values, value_offset):
    addr = _builder_scalar("addr", tmem_base + _cg1_tmem_row_bits(thread) + sub * 32, dtype="int32")
    _builder_emit(
        T.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
            *[values[value_offset + i] for i in range(32)], T.cast(addr, "uint32")
        )
    )


def _state_tmem_st_sub(tmem_base, thread, sub, values, value_offset):
    addr = _builder_scalar("addr", tmem_base + _cg1_tmem_row_bits(thread) + sub * 32, dtype="int32")
    _builder_emit(
        T.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
            T.cast(addr, "uint32"), *[values[value_offset + i] for i in range(32)]
        )
    )


def _state_input_tmem_st_sub(tmem_base, thread, sub, values):
    addr = _builder_scalar(
        "addr",
        tmem_base + _cg1_tmem_row_bits(thread) + TMEM_STATE_INPUT_COL + sub * 16,
        dtype="int32",
    )
    _builder_emit(
        T.ptx["tcgen05.st.sync.aligned.32x32b.x16.b32"](
            T.cast(addr, "uint32"), *[values[sub * 16 + i] for i in range(16)]
        )
    )


def _cg1_tmem_ld_f32(tmem_base, column, thread, values):
    addr = _builder_scalar("addr", tmem_base + _cg1_tmem_row_bits(thread) + column, dtype="int32")
    _builder_emit(
        T.ptx["tcgen05.ld.sync.aligned.16x256b.x8.b32"](
            *[values[i] for i in range(32)], T.cast(addr, "uint32")
        )
    )
    _builder_emit(
        T.ptx["tcgen05.ld.sync.aligned.16x256b.x8.b32"](
            *[values[32 + i] for i in range(32)], T.cast(addr + T.int32(1048576), "uint32")
        )
    )


def _cg1_tmem_st_f32(tmem_base, column, thread, values):
    addr = _builder_scalar("addr", tmem_base + _cg1_tmem_row_bits(thread) + column, dtype="int32")
    _builder_emit(
        T.ptx["tcgen05.st.sync.aligned.16x256b.x8.b32"](
            T.cast(addr, "uint32"), *[values[i] for i in range(32)]
        )
    )
    _builder_emit(
        T.ptx["tcgen05.st.sync.aligned.16x256b.x8.b32"](
            T.cast(addr + T.int32(1048576), "uint32"), *[values[32 + i] for i in range(32)]
        )
    )


def _cg1_tmem_st_f16_half(tmem_base, column, thread, half, values):
    addr = _builder_scalar("addr", tmem_base + _cg1_tmem_row_bits(thread) + column, dtype="int32")
    _builder_emit(
        T.ptx["tcgen05.st.sync.aligned.16x128b.x8.b32"](
            T.cast(addr + half * T.int32(1048576), "uint32"),
            *[values[half * 16 + i] for i in range(16)],
        )
    )


def _cg1_tmem_st_f16(tmem_base, column, thread, values):
    _builder_emit(_cg1_tmem_st_f16_half(tmem_base, column, thread, 0, values))
    _builder_emit(_cg1_tmem_st_f16_half(tmem_base, column, thread, 1, values))


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


def _load_v_fragment(smem_raw, stage, thread, values):
    lane_byte = _builder_scalar("lane_byte", _cg1_smem_lane_byte(thread), dtype="int32")
    stage_byte = _builder_scalar("stage_byte", stage * 16384, dtype="int32")
    with T.serial(4) as band:
        _builder_emit(
            T.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                *[values[band * 4 + i] for i in range(4)],
                smem_raw.ptr_to([SMEM_V_OFF + stage_byte + lane_byte + band * 2048]),
            )
        )
    T.buffer_store(lane_byte.buffer, lane_byte + _cg1_smem_second_half_delta(thread), [0])
    with T.serial(4) as band:
        _builder_emit(
            T.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                *[values[16 + band * 4 + i] for i in range(4)],
                smem_raw.ptr_to([SMEM_V_OFF + stage_byte + lane_byte + band * 2048]),
            )
        )


def _store_o_fragment(smem_raw, stage, thread, values):
    lane_byte = _builder_scalar("lane_byte", _cg1_smem_lane_byte(thread), dtype="int32")
    stage_byte = _builder_scalar("stage_byte", stage * 16384, dtype="int32")
    with T.serial(4) as band:
        _builder_emit(
            T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                smem_raw.ptr_to([SMEM_O_OFF + stage_byte + lane_byte + band * 2048]),
                *[values[band * 4 + i] for i in range(4)],
            )
        )
    T.buffer_store(lane_byte.buffer, lane_byte + _cg1_smem_second_half_delta(thread), [0])
    with T.serial(4) as band:
        _builder_emit(
            T.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                smem_raw.ptr_to([SMEM_O_OFF + stage_byte + lane_byte + band * 2048]),
                *[values[16 + band * 4 + i] for i in range(4)],
            )
        )


def _load_initial_state(
    smem_base_addr, initial_state, batch, output_head, tmem_base, thread, kv_count, *, HV
):
    _builder_emit(_producer_acquire(smem_base_addr, 328, kv_count, 1))
    cg1_thread = _builder_scalar(
        "cg1_thread", T.cast(T.bitwise_and(thread, T.int32(127)), "int64"), dtype="int64"
    )
    state_base = _builder_scalar(
        "state_base",
        (T.cast(batch, "int64") * T.cast(HV, "int64") + T.cast(output_head, "int64"))
        * T.int64(D_HEAD * D_HEAD)
        + cg1_thread * T.int64(D_HEAD),
        dtype="int64",
    )
    state_sub = _builder_name("state_sub", T.alloc_local((32,), "uint32"))
    with T.serial(4) as sub:
        with T.serial(8) as vector:
            _builder_emit(
                T.ptx["ld.global.L1::no_allocate.v4.b32"](
                    state_sub[vector * 4],
                    state_sub[vector * 4 + 1],
                    state_sub[vector * 4 + 2],
                    state_sub[vector * 4 + 3],
                    initial_state.ptr_to([state_base + sub * 32 + vector * 4]),
                )
            )
        _builder_emit(_state_tmem_st_sub(tmem_base, thread, sub, state_sub, 0))
    _builder_emit(T.ptx.tcgen05.wait__st.sync.aligned())
    _builder_emit(T.ptx.bar.sync(T.uint32(INITIAL_STATE_BARRIER), T.uint32(128)))
    _builder_if_23_4 = _builder_scope_enter(T.If(T.bitwise_and(thread, T.int32(127)) == 0))
    _builder_then_23_4 = _builder_scope_enter(T.Then())
    _builder_emit(_software_commit(smem_base_addr, 320, kv_count, 1))
    _builder_scope_exit(_builder_then_23_4)
    _builder_scope_exit(_builder_if_23_4)


def _copy_empty_state(initial_state, final_state, batch, output_head, thread, *, HV):
    cg1_thread = _builder_scalar(
        "cg1_thread", T.cast(T.bitwise_and(thread, T.int32(127)), "int64"), dtype="int64"
    )
    state_base = _builder_scalar(
        "state_base",
        (T.cast(batch, "int64") * T.cast(HV, "int64") + T.cast(output_head, "int64"))
        * T.int64(D_HEAD * D_HEAD)
        + cg1_thread * T.int64(D_HEAD),
        dtype="int64",
    )
    with T.serial(0, D_HEAD) as key:
        state_value = _builder_alloc_scalar("state_value", "float32")
        _builder_emit(T.ptx.ld.global_.f32(state_value, initial_state.ptr_to([state_base + key])))
        _builder_emit(T.ptx.st.global_.f32(final_state.ptr_to([state_base + key]), state_value))


def _store_final_state(
    smem_base_addr, final_state, batch, output_head, tmem_base, thread, kv_count, *, HV
):
    _builder_emit(_consumer_wait(smem_base_addr, 320, kv_count, 1))
    cg1_thread = _builder_scalar(
        "cg1_thread", T.cast(T.bitwise_and(thread, T.int32(127)), "int64"), dtype="int64"
    )
    state_base = _builder_scalar(
        "state_base",
        (T.cast(batch, "int64") * T.cast(HV, "int64") + T.cast(output_head, "int64"))
        * T.int64(D_HEAD * D_HEAD)
        + cg1_thread * T.int64(D_HEAD),
        dtype="int64",
    )
    state_sub = _builder_name("state_sub", T.alloc_local((32,), "uint32"))
    with T.serial(4) as sub:
        _builder_emit(_state_tmem_ld_sub(tmem_base, thread, sub, state_sub, 0))
        with T.serial(8) as vector:
            _builder_emit(
                T.ptx["st.global.L1::no_allocate.v4.b32"](
                    final_state.ptr_to([state_base + sub * 32 + vector * 4]),
                    state_sub[vector * 4],
                    state_sub[vector * 4 + 1],
                    state_sub[vector * 4 + 2],
                    state_sub[vector * 4 + 3],
                )
            )
    _builder_emit(_consumer_release(smem_base_addr, 328, kv_count, 1))


def _mma_ss_64x64_k128(tmem_d, a_desc_base, b_desc_base, full_barrier):
    with T.serial(8) as kphase:
        phase_off = _builder_scalar(
            "phase_off", T.uint64(kphase % 4 * 2 + kphase // 4 * 512), dtype="uint64"
        )
        _builder_emit(
            T.evaluate(
                T.ptx[_MMA_CHAIN](
                    T.cast(tmem_d, "uint32"),
                    a_desc_base + phase_off,
                    b_desc_base + phase_off,
                    T.uint32(68157456),
                    *_MMA_ZERO_MASKS,
                    T.ptx.pred(T.if_then_else(kphase == 0, 0, 1)),
                    pred=T.cuda.elect_sync(),
                )
            )
        )
    _builder_emit(_mma_commit(full_barrier))


def _mma_ts_128x64_k128(tmem_d, tmem_a, b_desc_base, full_barrier):
    with T.serial(8) as kphase:
        phase_off = _builder_scalar(
            "phase_off", T.uint64(kphase % 4 * 2 + kphase // 4 * 512), dtype="uint64"
        )
        _builder_emit(
            T.evaluate(
                T.ptx[_MMA_CHAIN](
                    T.cast(tmem_d, "uint32"),
                    T.cast(tmem_a + kphase * 8, "uint32"),
                    b_desc_base + phase_off,
                    T.uint32(135266320),
                    *_MMA_ZERO_MASKS,
                    T.ptx.pred(T.if_then_else(kphase == 0, 0, 1)),
                    pred=T.cuda.elect_sync(),
                )
            )
        )
    _builder_emit(_mma_commit(full_barrier))


def _mma_ts_128x64_k64(tmem_d, tmem_a, b_desc_base, accumulate):
    with T.serial(4) as kphase:
        phase_off = _builder_scalar("phase_off", T.uint64(kphase * 2), dtype="uint64")
        _builder_emit(
            T.evaluate(
                T.ptx[_MMA_CHAIN](
                    T.cast(tmem_d, "uint32"),
                    T.cast(tmem_a + kphase * 8, "uint32"),
                    b_desc_base + phase_off,
                    T.uint32(135266320),
                    *_MMA_ZERO_MASKS,
                    T.ptx.pred(T.if_then_else(kphase == 0, accumulate, 1)),
                    pred=T.cuda.elect_sync(),
                )
            )
        )


def _mma_ts_128x128_k64(tmem_d, tmem_a, b_desc_base, full_barrier):
    with T.serial(4) as kphase:
        _builder_emit(
            T.evaluate(
                T.ptx[_MMA_CHAIN](
                    T.cast(tmem_d, "uint32"),
                    T.cast(tmem_a + kphase * 8, "uint32"),
                    b_desc_base + T.uint64(kphase * 128),
                    T.uint32(136380432),
                    *_MMA_ZERO_MASKS,
                    True,
                    pred=T.cuda.elect_sync(),
                )
            )
        )
    _builder_emit(_mma_commit(full_barrier))


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


def _tma_qk_g2s(smem_base_addr, smem_off, descriptor, full_barrier, token, qk_head):
    with T.serial(0, 128, step=64) as d_coord:
        _builder_emit(
            T.ptx[_TMA_G2S_CHAIN[3]](
                _shared_addr(smem_base_addr, smem_off + d_coord * 128),
                descriptor,
                T.int32(d_coord),
                T.cast(token, "int32"),
                qk_head,
                full_barrier,
                T.uint64(0),
            )
        )


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
    with T.serial(0, 128, step=64) as d_coord:
        if HQ == HV:
            _builder_emit(
                T.ptx[_TMA_G2S_CHAIN[3]](
                    _shared_addr(smem_base_addr, smem_off + d_coord * 128),
                    descriptor,
                    T.int32(d_coord),
                    T.cast(token, "int32"),
                    output_head,
                    full_barrier,
                    T.uint64(0),
                )
            )
        else:
            _builder_emit(
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
            )


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
    _builder_emit(_producer_acquire_state(smem_base_addr, 32, k_count, k_phase))
    k_full = _builder_scalar("k_full", _shared_addr(smem_base_addr, k_count * 8))
    _builder_if_24_4 = _builder_scope_enter(T.If(chunk_idx == 0))
    _builder_then_24_4 = _builder_scope_enter(T.Then())
    _builder_if_25_8 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
    _builder_then_25_8 = _builder_scope_enter(T.Then())
    _builder_emit(_tensormap_acquire(descriptor_k))
    _builder_scope_exit(_builder_then_25_8)
    _builder_scope_exit(_builder_if_25_8)
    _builder_scope_exit(_builder_then_24_4)
    _builder_scope_exit(_builder_if_24_4)
    _builder_if_27_4 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
    _builder_then_27_4 = _builder_scope_enter(T.Then())
    _builder_emit(T.ptx.mbarrier.arrive.expect_tx.shared.b64(k_full, T.uint32(16384)))
    _builder_emit(
        _tma_qk_g2s(
            smem_base_addr,
            SMEM_K_OFF + k_count * 16384,
            descriptor_k,
            k_full,
            chunk_offset,
            qk_head,
        )
    )
    _builder_scope_exit(_builder_then_27_4)
    _builder_scope_exit(_builder_if_27_4)
    _builder_emit(_producer_acquire_state(smem_base_addr, 80, q_count, q_phase))
    q_full = _builder_scalar("q_full", _shared_addr(smem_base_addr, 64 + q_count * 8))
    _builder_if_40_4 = _builder_scope_enter(T.If(chunk_idx == 0))
    _builder_then_40_4 = _builder_scope_enter(T.Then())
    _builder_if_41_8 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
    _builder_then_41_8 = _builder_scope_enter(T.Then())
    _builder_emit(_tensormap_acquire(descriptor_q))
    _builder_scope_exit(_builder_then_41_8)
    _builder_scope_exit(_builder_if_41_8)
    _builder_scope_exit(_builder_then_40_4)
    _builder_scope_exit(_builder_if_40_4)
    _builder_if_43_4 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
    _builder_then_43_4 = _builder_scope_enter(T.Then())
    _builder_emit(T.ptx.mbarrier.arrive.expect_tx.shared.b64(q_full, T.uint32(16384)))
    _builder_emit(
        _tma_qk_g2s(
            smem_base_addr,
            SMEM_Q_OFF + q_count * 16384,
            descriptor_q,
            q_full,
            chunk_offset,
            qk_head,
        )
    )
    _builder_scope_exit(_builder_then_43_4)
    _builder_scope_exit(_builder_if_43_4)
    _builder_emit(_producer_acquire_state(smem_base_addr, 120, v_count, v_phase))
    v_full = _builder_scalar("v_full", _shared_addr(smem_base_addr, 96 + v_count * 8))
    _builder_if_56_4 = _builder_scope_enter(T.If(chunk_idx == 0))
    _builder_then_56_4 = _builder_scope_enter(T.Then())
    _builder_if_57_8 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
    _builder_then_57_8 = _builder_scope_enter(T.Then())
    _builder_emit(_tensormap_acquire(descriptor_v))
    _builder_scope_exit(_builder_then_57_8)
    _builder_scope_exit(_builder_if_57_8)
    _builder_scope_exit(_builder_then_56_4)
    _builder_scope_exit(_builder_if_56_4)
    _builder_if_59_4 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
    _builder_then_59_4 = _builder_scope_enter(T.Then())
    _builder_emit(T.ptx.mbarrier.arrive.expect_tx.shared.b64(v_full, T.uint32(16384)))
    _builder_emit(
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
    )
    _builder_scope_exit(_builder_then_59_4)
    _builder_scope_exit(_builder_if_59_4)


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
    pos0 = _builder_scalar("pos0", chunk_offset + T.cast(lane, "int64"), dtype="int64")
    pos1 = _builder_scalar("pos1", chunk_offset + T.cast(lane + 32, "int64"), dtype="int64")
    valid0 = _builder_scalar("valid0", 1, dtype="int32")
    valid1 = _builder_scalar("valid1", 1, dtype="int32")
    gate0 = _builder_scalar("gate0", T.float32(1.0), dtype="float32")
    gate1 = _builder_scalar("gate1", T.float32(1.0), dtype="float32")
    _builder_if_29_4 = _builder_scope_enter(T.If(is_last_tile))
    _builder_then_29_4 = _builder_scope_enter(T.Then())
    T.buffer_store(valid0.buffer, T.cast(pos0 < batch_end, "int32"), [0])
    T.buffer_store(valid1.buffer, T.cast(pos1 < batch_end, "int32"), [0])
    _builder_if_32_8 = _builder_scope_enter(T.If(valid0 != 0))
    _builder_then_32_8 = _builder_scope_enter(T.Then())
    _builder_emit(
        T.ptx.ld.global_.f32(
            gate0, gate.ptr_to([pos0 * T.cast(HV, "int64") + T.cast(output_head, "int64")])
        )
    )
    _builder_scope_exit(_builder_then_32_8)
    _builder_scope_exit(_builder_if_32_8)
    _builder_if_36_8 = _builder_scope_enter(T.If(valid1 != 0))
    _builder_then_36_8 = _builder_scope_enter(T.Then())
    _builder_emit(
        T.ptx.ld.global_.f32(
            gate1, gate.ptr_to([pos1 * T.cast(HV, "int64") + T.cast(output_head, "int64")])
        )
    )
    _builder_scope_exit(_builder_then_36_8)
    _builder_scope_exit(_builder_if_36_8)
    _builder_scope_exit(_builder_then_29_4)
    _builder_else_29_4 = _builder_scope_enter(T.Else())
    _builder_emit(
        T.ptx.ld.global_.f32(
            gate0, gate.ptr_to([pos0 * T.cast(HV, "int64") + T.cast(output_head, "int64")])
        )
    )
    _builder_emit(
        T.ptx.ld.global_.f32(
            gate1, gate.ptr_to([pos1 * T.cast(HV, "int64") + T.cast(output_head, "int64")])
        )
    )
    _builder_scope_exit(_builder_else_29_4)
    _builder_scope_exit(_builder_if_29_4)
    T.buffer_store(gate0.buffer, _lg2_approx_ftz(gate0 + T.float32(1e-10)), [0])
    T.buffer_store(gate1.buffer, _lg2_approx_ftz(gate1 + T.float32(1e-10)), [0])
    with T.unroll(0, 5) as scan_step:
        scan_offset = _builder_scalar("scan_offset", 1 << scan_step, dtype="int32")
        prior0 = _builder_scalar(
            "prior0",
            T.tvm_warp_shuffle_up(T.uint32(4294967295), gate0, scan_offset, 32, 32),
            dtype="float32",
        )
        prior1 = _builder_scalar(
            "prior1",
            T.tvm_warp_shuffle_up(T.uint32(4294967295), gate1, scan_offset, 32, 32),
            dtype="float32",
        )
        _builder_if_54_8 = _builder_scope_enter(T.If(lane >= scan_offset))
        _builder_then_54_8 = _builder_scope_enter(T.Then())
        T.buffer_store(gate0.buffer, gate0 + prior0, [0])
        T.buffer_store(gate1.buffer, gate1 + prior1, [0])
        _builder_scope_exit(_builder_then_54_8)
        _builder_scope_exit(_builder_if_54_8)
    T.buffer_store(
        gate1.buffer, gate1 + T.cuda._shfl_sync(T.uint32(4294967295), gate0, 31, 32), [0]
    )
    cumprod0 = _builder_alloc_scalar("cumprod0", "float32")
    cumprod1 = _builder_alloc_scalar("cumprod1", "float32")
    _builder_emit(T.ptx.ex2.approx.ftz.f32(cumprod0, gate0))
    _builder_emit(T.ptx.ex2.approx.ftz.f32(cumprod1, gate1))
    _builder_emit(_producer_acquire_state(smem_base_addr, 184, gate_count, gate_phase))
    gate_stage = _builder_scalar("gate_stage", gate_count, dtype="int32")
    _builder_emit(T.ptx.st.shared.f32(s_cumsumlog.ptr_to([gate_stage * 64 + lane]), gate0))
    _builder_emit(T.ptx.st.shared.f32(s_cumsumlog.ptr_to([gate_stage * 64 + lane + 32]), gate1))
    _builder_emit(T.ptx.st.shared.f32(s_cumprod.ptr_to([gate_stage * 64 + lane]), cumprod0))
    _builder_emit(T.ptx.st.shared.f32(s_cumprod.ptr_to([gate_stage * 64 + lane + 32]), cumprod1))
    _builder_emit(_software_commit_state(smem_base_addr, 144, gate_count))
    _builder_emit(_producer_acquire_state(smem_base_addr, 264, beta_count, beta_phase))
    beta_stage = _builder_scalar("beta_stage", beta_count, dtype="int32")
    beta_dst0 = _builder_scalar(
        "beta_dst0",
        _shared_addr(smem_base_addr, SMEM_BETA_OFF + (beta_stage * 64 + lane) * 4),
        dtype="uint32",
    )
    beta_dst1 = _builder_scalar(
        "beta_dst1",
        _shared_addr(smem_base_addr, SMEM_BETA_OFF + (beta_stage * 64 + lane + 32) * 4),
        dtype="uint32",
    )
    _builder_if_78_4 = _builder_scope_enter(T.If(is_last_tile))
    _builder_then_78_4 = _builder_scope_enter(T.Then())
    _builder_emit(T.ptx.st.shared.f32(s_beta.ptr_to([beta_stage * 64 + lane]), T.float32(0.0)))
    _builder_emit(T.ptx.st.shared.f32(s_beta.ptr_to([beta_stage * 64 + lane + 32]), T.float32(0.0)))
    _builder_emit(
        T.ptx["cp.async.ca.shared.global"](
            beta_dst0,
            beta.ptr_to([pos0 * T.cast(HV, "int64") + T.cast(output_head, "int64")]),
            4,
            pred=valid0,
        )
    )
    _builder_emit(
        T.ptx["cp.async.ca.shared.global"](
            beta_dst1,
            beta.ptr_to([pos1 * T.cast(HV, "int64") + T.cast(output_head, "int64")]),
            4,
            pred=valid1,
        )
    )
    _builder_scope_exit(_builder_then_78_4)
    _builder_else_78_4 = _builder_scope_enter(T.Else())
    _builder_emit(
        T.ptx["cp.async.ca.shared.global"](
            beta_dst0, beta.ptr_to([pos0 * T.cast(HV, "int64") + T.cast(output_head, "int64")]), 4
        )
    )
    _builder_emit(
        T.ptx["cp.async.ca.shared.global"](
            beta_dst1, beta.ptr_to([pos1 * T.cast(HV, "int64") + T.cast(output_head, "int64")]), 4
        )
    )
    _builder_scope_exit(_builder_else_78_4)
    _builder_scope_exit(_builder_if_78_4)
    _builder_emit(
        T.ptx.cp.async_.mbarrier.arrive.noinc.shared__cta.b64(
            _shared_addr(smem_base_addr, 224 + beta_count * 8)
        )
    )


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
    _builder_emit(_consumer_wait_state(smem_base_addr, 528, o_index, o_phase))
    stage = _builder_scalar("stage", o_index, dtype="int32")
    _builder_if_17_4 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
    _builder_then_17_4 = _builder_scope_enter(T.Then())
    with T.serial(0, 128, step=64) as d_coord:
        if HQ == HV:
            _builder_emit(
                T.ptx[_TMA_S2G_CHAIN[3]](
                    descriptor_o,
                    T.int32(d_coord),
                    T.cast(chunk_offset, "int32"),
                    output_head,
                    _shared_addr(smem_base_addr, SMEM_O_OFF + stage * 16384 + d_coord * 128),
                    T.uint64(0),
                )
            )
        else:
            _builder_emit(
                T.ptx[_TMA_S2G_CHAIN[4]](
                    descriptor_o,
                    T.int32(d_coord),
                    T.cast(chunk_offset, "int32"),
                    value_subhead,
                    qk_head,
                    _shared_addr(smem_base_addr, SMEM_O_OFF + stage * 16384 + d_coord * 128),
                    T.uint64(0),
                )
            )
    _builder_emit(T.ptx.cp.async_.bulk.commit_group())
    _builder_emit(T.ptx.cp.async_.bulk.wait_group.read(0))
    _builder_scope_exit(_builder_then_17_4)
    _builder_scope_exit(_builder_if_17_4)
    _builder_emit(_consumer_release_state(smem_base_addr, 544, o_index))


def _build_kernel(*, HQ: T.constexpr, HV: T.constexpr):
    with IRBuilder() as builder:
        with T.prim_func():
            T.func_name("_kernel")
            q_h = T.arg("q_h", T.handle())
            k_h = T.arg("k_h", T.handle())
            v_h = T.arg("v_h", T.handle())
            gate_h = T.arg("gate_h", T.handle())
            beta_h = T.arg("beta_h", T.handle())
            o_h = T.arg("o_h", T.handle())
            cu_seqlens_h = T.arg("cu_seqlens_h", T.handle())
            initial_state_h = T.arg("initial_state_h", T.handle())
            final_state_h = T.arg("final_state_h", T.handle())
            q_map = T.arg("q_map", T.TensorMap())
            k_map = T.arg("k_map", T.TensorMap())
            v_map = T.arg("v_map", T.TensorMap())
            o_map = T.arg("o_map", T.TensorMap())
            descriptor_workspace_h = T.arg("descriptor_workspace_h", T.handle())
            total_tokens = T.arg("total_tokens", T.int64())
            num_sequences = T.arg("num_sequences", T.int32())
            num_sms = T.arg("num_sms", T.int32())
            scale = T.arg("scale", T.float32())
            q = _builder_name(
                "q", T.match_buffer(q_h, (total_tokens * HQ * D_HEAD,), "float16", scope="global")
            )
            k = _builder_name(
                "k", T.match_buffer(k_h, (total_tokens * HQ * D_HEAD,), "float16", scope="global")
            )
            v = _builder_name(
                "v", T.match_buffer(v_h, (total_tokens * HV * D_HEAD,), "float16", scope="global")
            )
            gate = _builder_name(
                "gate", T.match_buffer(gate_h, (total_tokens * HV,), "float32", scope="global")
            )
            beta = _builder_name(
                "beta", T.match_buffer(beta_h, (total_tokens * HV,), "float32", scope="global")
            )
            o = _builder_name(
                "o", T.match_buffer(o_h, (total_tokens * HV * D_HEAD,), "float16", scope="global")
            )
            cu_seqlens = _builder_name(
                "cu_seqlens",
                T.match_buffer(cu_seqlens_h, (num_sequences + 1,), "int32", scope="global"),
            )
            initial_state = _builder_name(
                "initial_state",
                T.match_buffer(
                    initial_state_h,
                    (T.cast(num_sequences, "int64") * HV * D_HEAD * D_HEAD,),
                    "float32",
                    scope="global",
                ),
            )
            final_state = _builder_name(
                "final_state",
                T.match_buffer(
                    final_state_h,
                    (T.cast(num_sequences, "int64") * HV * D_HEAD * D_HEAD,),
                    "float32",
                    scope="global",
                ),
            )
            descriptor_workspace = _builder_name(
                "descriptor_workspace",
                T.match_buffer(
                    descriptor_workspace_h,
                    (num_sms * DESCRIPTOR_BYTES_PER_CTA,),
                    "int8",
                    scope="global",
                ),
            )
            _builder_emit(T.device_entry())
            _builder_enter(T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1}))
            block = _builder_name("block", T.cta_id([T.min(num_sequences * HV, num_sms)]))
            thread = _builder_name("thread", T.thread_id([THREADS]))
            output_heads = _builder_bind("output_heads", HV)
            total_work = _builder_bind("total_work", num_sequences * output_heads)
            grid_x = _builder_bind("grid_x", T.min(total_work, num_sms))
            warp = _builder_scalar("warp", _make_warp_uniform(T.cast(thread // 32, "uint32")))
            lane = _builder_bind("lane", thread % 32)
            pool = _builder_meta("pool", T.SMEMPool())
            smem_raw = _builder_name("smem_raw", pool.alloc((SMEM_TOTAL,), "uint8", align=1024))
            tmem_holding = _builder_name(
                "tmem_holding",
                T.decl_buffer(
                    (1,),
                    "int32",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_TMEM_HOLDING_OFF,
                    align=4,
                ),
            )
            s_q = _builder_name(
                "s_q",
                T.decl_buffer(
                    (64 * 128 * 2,),
                    "float16",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_Q_OFF,
                    align=1024,
                ),
            )
            s_k = _builder_name(
                "s_k",
                T.decl_buffer(
                    (64 * 128 * 4,),
                    "float16",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_K_OFF,
                    align=1024,
                ),
            )
            s_v = _builder_name(
                "s_v",
                T.decl_buffer(
                    (128 * 64 * 3,),
                    "float16",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_V_OFF,
                    align=1024,
                ),
            )
            s_ainv = _builder_name(
                "s_ainv",
                T.decl_buffer(
                    (64 * 64 * 3,),
                    "float16",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_AINV_OFF,
                    align=1024,
                ),
            )
            s_qk = _builder_name(
                "s_qk",
                T.decl_buffer(
                    (64 * 64 * 2,),
                    "float16",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_QK_OFF,
                    align=1024,
                ),
            )
            s_o = _builder_name(
                "s_o",
                T.decl_buffer(
                    (128 * 64 * 2,),
                    "float16",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_O_OFF,
                    align=1024,
                ),
            )
            s_cumsumlog = _builder_name(
                "s_cumsumlog",
                T.decl_buffer(
                    (64 * 5,),
                    "float32",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_CUMSUMLOG_OFF,
                    align=4,
                ),
            )
            s_cumprod = _builder_name(
                "s_cumprod",
                T.decl_buffer(
                    (64 * 5,),
                    "float32",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_CUMPROD_OFF,
                    align=4,
                ),
            )
            s_beta = _builder_name(
                "s_beta",
                T.decl_buffer(
                    (64 * 5,),
                    "float32",
                    data=smem_raw.data,
                    scope="shared.dyn",
                    byte_offset=SMEM_BETA_OFF,
                    align=4,
                ),
            )
            _builder_emit(pool.commit())
            descriptor_cta_base = _builder_scalar(
                "descriptor_cta_base",
                T.cast(block, "int64") * DESCRIPTOR_BYTES_PER_CTA,
                dtype="int64",
            )
            descriptor_q = _builder_scalar(
                "descriptor_q", descriptor_workspace.ptr_to([descriptor_cta_base + 0])
            )
            descriptor_k = _builder_scalar(
                "descriptor_k", descriptor_workspace.ptr_to([descriptor_cta_base + 128])
            )
            descriptor_v = _builder_scalar(
                "descriptor_v", descriptor_workspace.ptr_to([descriptor_cta_base + 256])
            )
            descriptor_o = _builder_scalar(
                "descriptor_o", descriptor_workspace.ptr_to([descriptor_cta_base + 384])
            )
            _builder_if_153_4 = _builder_scope_enter(T.If(warp == 0))
            _builder_then_153_4 = _builder_scope_enter(T.Then())
            _builder_if_154_8 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
            _builder_then_154_8 = _builder_scope_enter(T.Then())
            _builder_emit(_init_all_pipelines(smem_raw))
            _builder_emit(T.ptx.fence.mbarrier_init.release.cluster())
            _builder_scope_exit(_builder_then_154_8)
            _builder_scope_exit(_builder_if_154_8)
            _builder_scope_exit(_builder_then_153_4)
            _builder_scope_exit(_builder_if_153_4)
            _builder_emit(T.cuda.cta_sync())
            _builder_if_160_4 = _builder_scope_enter(T.If(warp == 8))
            _builder_then_160_4 = _builder_scope_enter(T.Then())
            _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(q_map))))
            _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(k_map))))
            _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(v_map))))
            _builder_emit(T.evaluate(T.ptx.prefetch.tensormap(T.address_of(o_map))))
            _builder_scope_exit(_builder_then_160_4)
            _builder_scope_exit(_builder_if_160_4)
            _builder_if_167_4 = _builder_scope_enter(T.If(warp <= 3))
            _builder_then_167_4 = _builder_scope_enter(T.Then())
            smem_addr_cg0 = _builder_scalar(
                "smem_addr_cg0", T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0])), dtype="uint32"
            )
            _builder_emit(T.ptx.setmaxnreg.inc.sync.aligned.u32(224))
            _builder_emit(T.ptx.bar.sync(T.uint32(TMEM_ALLOC_BARRIER), T.uint32(320)))
            tmem_base_cg0 = _builder_alloc_scalar("tmem_base_cg0", "int32")
            _builder_emit(
                T.ptx.ld.volatile.shared.s32(tmem_base_cg0, T.address_of(tmem_holding[0]))
            )
            gate_consumer_index_cg0 = _builder_scalar("gate_consumer_index_cg0", 0, dtype="int32")
            gate_consumer_phase_cg0 = _builder_scalar("gate_consumer_phase_cg0", 0, dtype="int32")
            beta_consumer_index_cg0 = _builder_scalar("beta_consumer_index_cg0", 0, dtype="int32")
            beta_consumer_phase_cg0 = _builder_scalar("beta_consumer_phase_cg0", 0, dtype="int32")
            shared_consumer_index_cg0 = _builder_scalar(
                "shared_consumer_index_cg0", 0, dtype="int32"
            )
            shared_consumer_phase_cg0 = _builder_scalar(
                "shared_consumer_phase_cg0", 0, dtype="int32"
            )
            ainv_producer_index_cg0 = _builder_scalar("ainv_producer_index_cg0", 0, dtype="int32")
            ainv_producer_phase_cg0 = _builder_scalar("ainv_producer_phase_cg0", 1, dtype="int32")
            qk_producer_index_cg0 = _builder_scalar("qk_producer_index_cg0", 0, dtype="int32")
            qk_producer_phase_cg0 = _builder_scalar("qk_producer_phase_cg0", 1, dtype="int32")
            work_linear_cg0 = _builder_scalar("work_linear_cg0", block, dtype="int32")
            with T.While(work_linear_cg0 < total_work):
                work_u32_cg0 = _builder_scalar(
                    "work_u32_cg0", T.cast(work_linear_cg0, "uint32"), dtype="uint32"
                )
                remain_u32_cg0 = _builder_scalar(
                    "remain_u32_cg0", _udiv_u32_const(work_u32_cg0, HV), dtype="uint32"
                )
                head_cg0 = _builder_scalar(
                    "head_cg0",
                    T.cast(work_u32_cg0 - remain_u32_cg0 * T.uint32(output_heads), "int32"),
                    dtype="int32",
                )
                remain_cg0 = _builder_scalar(
                    "remain_cg0", T.cast(remain_u32_cg0, "int32"), dtype="int32"
                )
                batch_cg0 = _builder_scalar("batch_cg0", remain_cg0, dtype="int32")
                sequence_bounds_cg0 = _builder_name(
                    "sequence_bounds_cg0", T.alloc_local((2,), "int32")
                )
                _builder_emit(_load_sequence_bounds(cu_seqlens, batch_cg0, sequence_bounds_cg0))
                batch_start_cg0 = _builder_scalar(
                    "batch_start_cg0", T.cast(sequence_bounds_cg0[0], "int64"), dtype="int64"
                )
                batch_end_cg0 = _builder_scalar(
                    "batch_end_cg0", T.cast(sequence_bounds_cg0[1], "int64"), dtype="int64"
                )
                seqlen_cg0 = _builder_scalar(
                    "seqlen_cg0", T.cast(batch_end_cg0 - batch_start_cg0, "int32"), dtype="int32"
                )
                num_pairs_cg0 = _builder_scalar(
                    "num_pairs_cg0", (seqlen_cg0 + PAIR_TOKENS - 1) // PAIR_TOKENS, dtype="int32"
                )
                with T.serial(0, num_pairs_cg0) as pair_cg0:
                    row_cg0 = _builder_scalar(
                        "row_cg0",
                        T.bitwise_or(
                            T.bitwise_and(thread >> 1, T.int32(48)),
                            T.bitwise_and(thread, T.int32(15)),
                        ),
                        dtype="int32",
                    )
                    col_base_cg0 = _builder_scalar(
                        "col_base_cg0", T.bitwise_and(thread, T.int32(16)), dtype="int32"
                    )
                    transfer0_cg0 = _builder_name("transfer0_cg0", T.alloc_local((32,), "float32"))
                    transfer1_cg0 = _builder_name("transfer1_cg0", T.alloc_local((32,), "float32"))
                    gate0_count_cg0 = _builder_scalar(
                        "gate0_count_cg0", gate_consumer_index_cg0, dtype="int32"
                    )
                    gate0_phase_cg0 = _builder_scalar(
                        "gate0_phase_cg0", gate_consumer_phase_cg0, dtype="int32"
                    )
                    _builder_emit(
                        _consumer_wait_state(smem_addr_cg0, 144, gate0_count_cg0, gate0_phase_cg0)
                    )
                    T.buffer_store(
                        gate_consumer_index_cg0.buffer, _pipe_next_index(gate0_count_cg0, 5), [0]
                    )
                    T.buffer_store(
                        gate_consumer_phase_cg0.buffer,
                        _pipe_next_phase(gate0_count_cg0, gate0_phase_cg0, 5),
                        [0],
                    )
                    gate1_count_cg0 = _builder_scalar(
                        "gate1_count_cg0", gate_consumer_index_cg0, dtype="int32"
                    )
                    gate1_phase_cg0 = _builder_scalar(
                        "gate1_phase_cg0", gate_consumer_phase_cg0, dtype="int32"
                    )
                    _builder_emit(
                        _consumer_wait_state(smem_addr_cg0, 144, gate1_count_cg0, gate1_phase_cg0)
                    )
                    T.buffer_store(
                        gate_consumer_index_cg0.buffer, _pipe_next_index(gate1_count_cg0, 5), [0]
                    )
                    T.buffer_store(
                        gate_consumer_phase_cg0.buffer,
                        _pipe_next_phase(gate1_count_cg0, gate1_phase_cg0, 5),
                        [0],
                    )
                    gate0_stage_cg0 = _builder_scalar(
                        "gate0_stage_cg0", gate0_count_cg0, dtype="int32"
                    )
                    gate1_stage_cg0 = _builder_scalar(
                        "gate1_stage_cg0", gate1_count_cg0, dtype="int32"
                    )
                    row_log0_cg0 = _builder_alloc_scalar("row_log0_cg0", "float32")
                    row_log1_cg0 = _builder_alloc_scalar("row_log1_cg0", "float32")
                    _builder_emit(
                        T.ptx.ld.shared.f32(
                            row_log0_cg0, s_cumsumlog.ptr_to([gate0_stage_cg0 * 64 + row_cg0])
                        )
                    )
                    _builder_emit(
                        T.ptx.ld.shared.f32(
                            row_log1_cg0, s_cumsumlog.ptr_to([gate1_stage_cg0 * 64 + row_cg0])
                        )
                    )
                    with T.unroll(32) as frag_idx_cg0:
                        col_cg0 = _builder_scalar(
                            "col_cg0", col_base_cg0 + frag_idx_cg0, dtype="int32"
                        )
                        _builder_if_235_20 = _builder_scope_enter(T.If(frag_idx_cg0 >= 16))
                        _builder_then_235_20 = _builder_scope_enter(T.Then())
                        T.buffer_store(col_cg0.buffer, col_cg0 + 16, [0])
                        _builder_scope_exit(_builder_then_235_20)
                        _builder_scope_exit(_builder_if_235_20)
                        transfer_exp0_cg0 = _builder_scalar(
                            "transfer_exp0_cg0", T.float32(0.0), dtype="float32"
                        )
                        transfer_exp1_cg0 = _builder_scalar(
                            "transfer_exp1_cg0", T.float32(0.0), dtype="float32"
                        )
                        _builder_if_242_20 = _builder_scope_enter(T.If(row_cg0 >= col_cg0))
                        _builder_then_242_20 = _builder_scope_enter(T.Then())
                        col_log0_cg0 = _builder_alloc_scalar("col_log0_cg0", "float32")
                        col_log1_cg0 = _builder_alloc_scalar("col_log1_cg0", "float32")
                        _builder_emit(
                            T.ptx.ld.shared.f32(
                                col_log0_cg0, s_cumsumlog.ptr_to([gate0_stage_cg0 * 64 + col_cg0])
                            )
                        )
                        _builder_emit(
                            T.ptx.ld.shared.f32(
                                col_log1_cg0, s_cumsumlog.ptr_to([gate1_stage_cg0 * 64 + col_cg0])
                            )
                        )
                        _builder_emit(
                            T.ptx.ex2.approx.ftz.f32(transfer_exp0_cg0, row_log0_cg0 - col_log0_cg0)
                        )
                        _builder_emit(
                            T.ptx.ex2.approx.ftz.f32(transfer_exp1_cg0, row_log1_cg0 - col_log1_cg0)
                        )
                        _builder_scope_exit(_builder_then_242_20)
                        _builder_scope_exit(_builder_if_242_20)
                        T.buffer_store(transfer0_cg0, transfer_exp0_cg0, [frag_idx_cg0])
                        T.buffer_store(transfer1_cg0, transfer_exp1_cg0, [frag_idx_cg0])
                    _builder_emit(_consumer_release_state(smem_addr_cg0, 184, gate0_count_cg0))
                    _builder_emit(_consumer_release_state(smem_addr_cg0, 184, gate1_count_cg0))
                    beta0_count_cg0 = _builder_scalar(
                        "beta0_count_cg0", beta_consumer_index_cg0, dtype="int32"
                    )
                    beta0_phase_cg0 = _builder_scalar(
                        "beta0_phase_cg0", beta_consumer_phase_cg0, dtype="int32"
                    )
                    _builder_emit(
                        _consumer_wait_state(smem_addr_cg0, 224, beta0_count_cg0, beta0_phase_cg0)
                    )
                    T.buffer_store(
                        beta_consumer_index_cg0.buffer, _pipe_next_index(beta0_count_cg0, 5), [0]
                    )
                    T.buffer_store(
                        beta_consumer_phase_cg0.buffer,
                        _pipe_next_phase(beta0_count_cg0, beta0_phase_cg0, 5),
                        [0],
                    )
                    beta1_count_cg0 = _builder_scalar(
                        "beta1_count_cg0", beta_consumer_index_cg0, dtype="int32"
                    )
                    beta1_phase_cg0 = _builder_scalar(
                        "beta1_phase_cg0", beta_consumer_phase_cg0, dtype="int32"
                    )
                    _builder_emit(
                        _consumer_wait_state(smem_addr_cg0, 224, beta1_count_cg0, beta1_phase_cg0)
                    )
                    T.buffer_store(
                        beta_consumer_index_cg0.buffer, _pipe_next_index(beta1_count_cg0, 5), [0]
                    )
                    T.buffer_store(
                        beta_consumer_phase_cg0.buffer,
                        _pipe_next_phase(beta1_count_cg0, beta1_phase_cg0, 5),
                        [0],
                    )
                    beta0_cg0 = _builder_alloc_scalar("beta0_cg0", "float32")
                    beta1_cg0 = _builder_alloc_scalar("beta1_cg0", "float32")
                    _builder_emit(
                        T.ptx.ld.shared.f32(
                            beta0_cg0, s_beta.ptr_to([beta0_count_cg0 * 64 + row_cg0])
                        )
                    )
                    _builder_emit(
                        T.ptx.ld.shared.f32(
                            beta1_cg0, s_beta.ptr_to([beta1_count_cg0 * 64 + row_cg0])
                        )
                    )
                    ainv0_count_cg0 = _builder_scalar(
                        "ainv0_count_cg0", ainv_producer_index_cg0, dtype="int32"
                    )
                    ainv0_phase_cg0 = _builder_scalar(
                        "ainv0_phase_cg0", ainv_producer_phase_cg0, dtype="int32"
                    )
                    _builder_emit(
                        _producer_acquire_state(
                            smem_addr_cg0, 408, ainv0_count_cg0, ainv0_phase_cg0
                        )
                    )
                    T.buffer_store(
                        ainv_producer_index_cg0.buffer, _pipe_next_index(ainv0_count_cg0, 3), [0]
                    )
                    T.buffer_store(
                        ainv_producer_phase_cg0.buffer,
                        _pipe_next_phase(ainv0_count_cg0, ainv0_phase_cg0, 3),
                        [0],
                    )
                    kk0_count_cg0 = _builder_scalar(
                        "kk0_count_cg0", shared_consumer_index_cg0, dtype="int32"
                    )
                    kk0_phase_cg0 = _builder_scalar(
                        "kk0_phase_cg0", shared_consumer_phase_cg0, dtype="int32"
                    )
                    _builder_emit(
                        _consumer_wait_state(smem_addr_cg0, 336, kk0_count_cg0, kk0_phase_cg0)
                    )
                    T.buffer_store(
                        shared_consumer_index_cg0.buffer, _pipe_next_index(kk0_count_cg0, 2), [0]
                    )
                    T.buffer_store(
                        shared_consumer_phase_cg0.buffer,
                        _pipe_next_phase(kk0_count_cg0, kk0_phase_cg0, 2),
                        [0],
                    )
                    kk_values_cg0 = _builder_name("kk_values_cg0", T.alloc_local((32,), "float32"))
                    _builder_emit(_cg0_tmem_ld(tmem_base_cg0, kk0_count_cg0, thread, kk_values_cg0))
                    _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                    _builder_emit(_consumer_release_state(smem_addr_cg0, 352, kk0_count_cg0))
                    with T.unroll(16) as pair_frag_cg0:
                        packed_mul_cg0 = _builder_alloc_scalar("packed_mul_cg0", "uint64")
                        _builder_emit(
                            T.ptx.mul.rn.f32x2(
                                packed_mul_cg0,
                                T.cuda.make_float2(
                                    kk_values_cg0[pair_frag_cg0 * 2],
                                    kk_values_cg0[pair_frag_cg0 * 2 + 1],
                                ),
                                T.cuda.make_float2(
                                    transfer0_cg0[pair_frag_cg0 * 2],
                                    transfer0_cg0[pair_frag_cg0 * 2 + 1],
                                ),
                            )
                        )
                        _builder_emit(
                            T.ptx.mul.rn.f32x2(
                                packed_mul_cg0,
                                packed_mul_cg0,
                                T.cuda.make_float2(beta0_cg0, beta0_cg0),
                            )
                        )
                        T.buffer_store(
                            kk_values_cg0, T.cuda.float2_x(packed_mul_cg0), [pair_frag_cg0 * 2]
                        )
                        T.buffer_store(
                            kk_values_cg0, T.cuda.float2_y(packed_mul_cg0), [pair_frag_cg0 * 2 + 1]
                        )
                    _builder_emit(
                        _cg0_store_fragment(
                            smem_raw, SMEM_AINV_OFF, ainv0_count_cg0, thread, kk_values_cg0
                        )
                    )
                    ainv1_count_cg0 = _builder_scalar(
                        "ainv1_count_cg0", ainv_producer_index_cg0, dtype="int32"
                    )
                    ainv1_phase_cg0 = _builder_scalar(
                        "ainv1_phase_cg0", ainv_producer_phase_cg0, dtype="int32"
                    )
                    _builder_emit(
                        _producer_acquire_state(
                            smem_addr_cg0, 408, ainv1_count_cg0, ainv1_phase_cg0
                        )
                    )
                    T.buffer_store(
                        ainv_producer_index_cg0.buffer, _pipe_next_index(ainv1_count_cg0, 3), [0]
                    )
                    T.buffer_store(
                        ainv_producer_phase_cg0.buffer,
                        _pipe_next_phase(ainv1_count_cg0, ainv1_phase_cg0, 3),
                        [0],
                    )
                    kk1_count_cg0 = _builder_scalar(
                        "kk1_count_cg0", shared_consumer_index_cg0, dtype="int32"
                    )
                    kk1_phase_cg0 = _builder_scalar(
                        "kk1_phase_cg0", shared_consumer_phase_cg0, dtype="int32"
                    )
                    _builder_emit(
                        _consumer_wait_state(smem_addr_cg0, 336, kk1_count_cg0, kk1_phase_cg0)
                    )
                    T.buffer_store(
                        shared_consumer_index_cg0.buffer, _pipe_next_index(kk1_count_cg0, 2), [0]
                    )
                    T.buffer_store(
                        shared_consumer_phase_cg0.buffer,
                        _pipe_next_phase(kk1_count_cg0, kk1_phase_cg0, 2),
                        [0],
                    )
                    _builder_emit(_cg0_tmem_ld(tmem_base_cg0, kk1_count_cg0, thread, kk_values_cg0))
                    _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                    _builder_emit(_consumer_release_state(smem_addr_cg0, 352, kk1_count_cg0))
                    with T.unroll(16) as pair_frag_cg0:
                        packed_mul_cg0 = _builder_alloc_scalar("packed_mul_cg0", "uint64")
                        _builder_emit(
                            T.ptx.mul.rn.f32x2(
                                packed_mul_cg0,
                                T.cuda.make_float2(
                                    kk_values_cg0[pair_frag_cg0 * 2],
                                    kk_values_cg0[pair_frag_cg0 * 2 + 1],
                                ),
                                T.cuda.make_float2(
                                    transfer1_cg0[pair_frag_cg0 * 2],
                                    transfer1_cg0[pair_frag_cg0 * 2 + 1],
                                ),
                            )
                        )
                        _builder_emit(
                            T.ptx.mul.rn.f32x2(
                                packed_mul_cg0,
                                packed_mul_cg0,
                                T.cuda.make_float2(beta1_cg0, beta1_cg0),
                            )
                        )
                        T.buffer_store(
                            kk_values_cg0, T.cuda.float2_x(packed_mul_cg0), [pair_frag_cg0 * 2]
                        )
                        T.buffer_store(
                            kk_values_cg0, T.cuda.float2_y(packed_mul_cg0), [pair_frag_cg0 * 2 + 1]
                        )
                    _builder_emit(
                        _cg0_store_fragment(
                            smem_raw, SMEM_AINV_OFF, ainv1_count_cg0, thread, kk_values_cg0
                        )
                    )
                    local_warp_cg0 = _builder_scalar(
                        "local_warp_cg0", thread >> 5 & 3, dtype="int32"
                    )
                    inverse_group_cg0 = _builder_scalar(
                        "inverse_group_cg0", local_warp_cg0 >> 1, dtype="int32"
                    )
                    inverse_local_warp_cg0 = _builder_scalar(
                        "inverse_local_warp_cg0", local_warp_cg0 & 1, dtype="int32"
                    )
                    inverse_stage_cg0 = _builder_scalar(
                        "inverse_stage_cg0", ainv0_count_cg0, dtype="int32"
                    )
                    _builder_if_344_16 = _builder_scope_enter(T.If(inverse_group_cg0 == 1))
                    _builder_then_344_16 = _builder_scope_enter(T.Then())
                    T.buffer_store(inverse_stage_cg0.buffer, ainv1_count_cg0, [0])
                    _builder_scope_exit(_builder_then_344_16)
                    _builder_scope_exit(_builder_if_344_16)
                    diagonal_block_cg0 = _builder_scalar(
                        "diagonal_block_cg0",
                        (inverse_local_warp_cg0 * 32 + lane >> 3) * 8,
                        dtype="int32",
                    )
                    _builder_emit(T.ptx.bar.sync(T.uint32(INVERSE_BARRIER), T.uint32(128)))
                    _builder_emit(
                        _invert_diagonal_8x8(smem_raw, inverse_stage_cg0, diagonal_block_cg0, lane)
                    )
                    _builder_emit(T.ptx.bar.sync(T.uint32(INVERSE_BARRIER), T.uint32(128)))
                    _builder_emit(
                        _inverse_8_to_16(smem_raw, ainv0_count_cg0, local_warp_cg0 * 16, lane)
                    )
                    _builder_emit(
                        _inverse_8_to_16(smem_raw, ainv1_count_cg0, local_warp_cg0 * 16, lane)
                    )
                    _builder_emit(T.ptx.bar.sync(T.uint32(INVERSE_BARRIER), T.uint32(128)))
                    _builder_emit(
                        _inverse_16_to_32(
                            smem_raw, inverse_stage_cg0, inverse_local_warp_cg0 * 32, lane
                        )
                    )
                    _builder_emit(T.ptx.bar.sync(T.uint32(INVERSE_BARRIER), T.uint32(128)))
                    _builder_emit(
                        _inverse_32_to_64(smem_raw, inverse_stage_cg0, inverse_local_warp_cg0, lane)
                    )
                    _builder_emit(T.ptx.bar.sync(T.uint32(INVERSE_BARRIER), T.uint32(128)))
                    inv_values_cg0 = _builder_name(
                        "inv_values_cg0", T.alloc_local((32,), "float32")
                    )
                    ainv0_stage_cg0 = _builder_scalar(
                        "ainv0_stage_cg0", ainv0_count_cg0, dtype="int32"
                    )
                    _builder_emit(
                        _cg0_load_fragment(
                            smem_raw, SMEM_AINV_OFF, ainv0_stage_cg0, thread, inv_values_cg0
                        )
                    )
                    with T.unroll(16) as pair_frag_cg0:
                        inv_col0_cg0 = _builder_scalar(
                            "inv_col0_cg0", col_base_cg0 + pair_frag_cg0 * 2, dtype="int32"
                        )
                        _builder_if_364_20 = _builder_scope_enter(T.If(pair_frag_cg0 >= 8))
                        _builder_then_364_20 = _builder_scope_enter(T.Then())
                        T.buffer_store(inv_col0_cg0.buffer, inv_col0_cg0 + 16, [0])
                        _builder_scope_exit(_builder_then_364_20)
                        _builder_scope_exit(_builder_if_364_20)
                        beta_col0_cg0 = _builder_alloc_scalar("beta_col0_cg0", "uint64")
                        _builder_emit(
                            T.ptx.ld.shared.u64(
                                beta_col0_cg0, s_beta.ptr_to([beta0_count_cg0 * 64 + inv_col0_cg0])
                            )
                        )
                        inv_mul_cg0 = _builder_alloc_scalar("inv_mul_cg0", "uint64")
                        _builder_emit(
                            T.ptx.mul.rn.f32x2(
                                inv_mul_cg0,
                                T.cuda.make_float2(
                                    inv_values_cg0[pair_frag_cg0 * 2],
                                    inv_values_cg0[pair_frag_cg0 * 2 + 1],
                                ),
                                beta_col0_cg0,
                            )
                        )
                        T.buffer_store(
                            inv_values_cg0, T.cuda.float2_x(inv_mul_cg0), [pair_frag_cg0 * 2]
                        )
                        T.buffer_store(
                            inv_values_cg0, T.cuda.float2_y(inv_mul_cg0), [pair_frag_cg0 * 2 + 1]
                        )
                    _builder_emit(
                        _cg0_store_fragment(
                            smem_raw, SMEM_AINV_OFF, ainv0_stage_cg0, thread, inv_values_cg0
                        )
                    )
                    _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                    _builder_emit(_software_commit_state(smem_addr_cg0, 384, ainv0_count_cg0))
                    _builder_emit(_consumer_release_state(smem_addr_cg0, 264, beta0_count_cg0))
                    ainv1_stage_cg0 = _builder_scalar(
                        "ainv1_stage_cg0", ainv1_count_cg0, dtype="int32"
                    )
                    _builder_emit(
                        _cg0_load_fragment(
                            smem_raw, SMEM_AINV_OFF, ainv1_stage_cg0, thread, inv_values_cg0
                        )
                    )
                    with T.unroll(16) as pair_frag_cg0:
                        inv_col1_cg0 = _builder_scalar(
                            "inv_col1_cg0", col_base_cg0 + pair_frag_cg0 * 2, dtype="int32"
                        )
                        _builder_if_392_20 = _builder_scope_enter(T.If(pair_frag_cg0 >= 8))
                        _builder_then_392_20 = _builder_scope_enter(T.Then())
                        T.buffer_store(inv_col1_cg0.buffer, inv_col1_cg0 + 16, [0])
                        _builder_scope_exit(_builder_then_392_20)
                        _builder_scope_exit(_builder_if_392_20)
                        beta_col1_cg0 = _builder_alloc_scalar("beta_col1_cg0", "uint64")
                        _builder_emit(
                            T.ptx.ld.shared.u64(
                                beta_col1_cg0, s_beta.ptr_to([beta1_count_cg0 * 64 + inv_col1_cg0])
                            )
                        )
                        inv_mul_cg0 = _builder_alloc_scalar("inv_mul_cg0", "uint64")
                        _builder_emit(
                            T.ptx.mul.rn.f32x2(
                                inv_mul_cg0,
                                T.cuda.make_float2(
                                    inv_values_cg0[pair_frag_cg0 * 2],
                                    inv_values_cg0[pair_frag_cg0 * 2 + 1],
                                ),
                                beta_col1_cg0,
                            )
                        )
                        T.buffer_store(
                            inv_values_cg0, T.cuda.float2_x(inv_mul_cg0), [pair_frag_cg0 * 2]
                        )
                        T.buffer_store(
                            inv_values_cg0, T.cuda.float2_y(inv_mul_cg0), [pair_frag_cg0 * 2 + 1]
                        )
                    _builder_emit(
                        _cg0_store_fragment(
                            smem_raw, SMEM_AINV_OFF, ainv1_stage_cg0, thread, inv_values_cg0
                        )
                    )
                    _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                    _builder_emit(_software_commit_state(smem_addr_cg0, 384, ainv1_count_cg0))
                    _builder_emit(_consumer_release_state(smem_addr_cg0, 264, beta1_count_cg0))
                    with T.unroll(2) as qk_in_pair_cg0:
                        qk_ready_count_cg0 = _builder_scalar(
                            "qk_ready_count_cg0", qk_producer_index_cg0, dtype="int32"
                        )
                        qk_ready_phase_cg0 = _builder_scalar(
                            "qk_ready_phase_cg0", qk_producer_phase_cg0, dtype="int32"
                        )
                        _builder_emit(
                            _producer_acquire_state(
                                smem_addr_cg0, 448, qk_ready_count_cg0, qk_ready_phase_cg0
                            )
                        )
                        T.buffer_store(
                            qk_producer_index_cg0.buffer,
                            _pipe_next_index(qk_ready_count_cg0, 2),
                            [0],
                        )
                        T.buffer_store(
                            qk_producer_phase_cg0.buffer,
                            _pipe_next_phase(qk_ready_count_cg0, qk_ready_phase_cg0, 2),
                            [0],
                        )
                        qk_acc_count_cg0 = _builder_scalar(
                            "qk_acc_count_cg0", shared_consumer_index_cg0, dtype="int32"
                        )
                        qk_acc_phase_cg0 = _builder_scalar(
                            "qk_acc_phase_cg0", shared_consumer_phase_cg0, dtype="int32"
                        )
                        _builder_emit(
                            _consumer_wait_state(
                                smem_addr_cg0, 336, qk_acc_count_cg0, qk_acc_phase_cg0
                            )
                        )
                        T.buffer_store(
                            shared_consumer_index_cg0.buffer,
                            _pipe_next_index(qk_acc_count_cg0, 2),
                            [0],
                        )
                        T.buffer_store(
                            shared_consumer_phase_cg0.buffer,
                            _pipe_next_phase(qk_acc_count_cg0, qk_acc_phase_cg0, 2),
                            [0],
                        )
                        _builder_emit(
                            _cg0_tmem_ld(tmem_base_cg0, qk_acc_count_cg0, thread, kk_values_cg0)
                        )
                        with T.unroll(16) as pair_frag_cg0:
                            qk_factor0_cg0 = _builder_scalar(
                                "qk_factor0_cg0", transfer0_cg0[pair_frag_cg0 * 2], dtype="float32"
                            )
                            qk_factor1_cg0 = _builder_scalar(
                                "qk_factor1_cg0",
                                transfer0_cg0[pair_frag_cg0 * 2 + 1],
                                dtype="float32",
                            )
                            _builder_if_437_24 = _builder_scope_enter(T.If(qk_in_pair_cg0 == 1))
                            _builder_then_437_24 = _builder_scope_enter(T.Then())
                            T.buffer_store(
                                qk_factor0_cg0.buffer, transfer1_cg0[pair_frag_cg0 * 2], [0]
                            )
                            T.buffer_store(
                                qk_factor1_cg0.buffer, transfer1_cg0[pair_frag_cg0 * 2 + 1], [0]
                            )
                            _builder_scope_exit(_builder_then_437_24)
                            _builder_scope_exit(_builder_if_437_24)
                            qk_mul_cg0 = _builder_alloc_scalar("qk_mul_cg0", "uint64")
                            _builder_emit(
                                T.ptx.mul.rn.f32x2(
                                    qk_mul_cg0,
                                    T.cuda.make_float2(
                                        kk_values_cg0[pair_frag_cg0 * 2],
                                        kk_values_cg0[pair_frag_cg0 * 2 + 1],
                                    ),
                                    T.cuda.make_float2(qk_factor0_cg0, qk_factor1_cg0),
                                )
                            )
                            _builder_emit(
                                T.ptx.mul.rn.f32x2(
                                    qk_mul_cg0, qk_mul_cg0, T.cuda.make_float2(scale, scale)
                                )
                            )
                            T.buffer_store(
                                kk_values_cg0, T.cuda.float2_x(qk_mul_cg0), [pair_frag_cg0 * 2]
                            )
                            T.buffer_store(
                                kk_values_cg0, T.cuda.float2_y(qk_mul_cg0), [pair_frag_cg0 * 2 + 1]
                            )
                        _builder_emit(
                            _cg0_store_fragment(
                                smem_raw, SMEM_QK_OFF, qk_ready_count_cg0, thread, kk_values_cg0
                            )
                        )
                        _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                        _builder_emit(T.ptx.tcgen05.wait__ld.sync.aligned())
                        _builder_emit(_consumer_release_state(smem_addr_cg0, 352, qk_acc_count_cg0))
                        _builder_emit(
                            _software_commit_state(smem_addr_cg0, 432, qk_ready_count_cg0)
                        )
                T.buffer_store(work_linear_cg0.buffer, work_linear_cg0 + grid_x, [0])
            _builder_emit(
                _producer_tail_state(
                    smem_addr_cg0, 408, ainv_producer_index_cg0, ainv_producer_phase_cg0, 3
                )
            )
            _builder_emit(
                _producer_tail_state(
                    smem_addr_cg0, 448, qk_producer_index_cg0, qk_producer_phase_cg0, 2
                )
            )
            _builder_scope_exit(_builder_then_167_4)
            _builder_scope_exit(_builder_if_167_4)
            _builder_if_465_4 = _builder_scope_enter(T.If(T.And(warp >= 4, warp <= 7)))
            _builder_then_465_4 = _builder_scope_enter(T.Then())
            smem_addr_cg1 = _builder_scalar(
                "smem_addr_cg1", T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0])), dtype="uint32"
            )
            _builder_emit(T.ptx.setmaxnreg.inc.sync.aligned.u32(256))
            _builder_if_468_8 = _builder_scope_enter(T.If(warp == 4))
            _builder_then_468_8 = _builder_scope_enter(T.Then())
            _builder_emit(
                T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                    T.address_of(tmem_holding[0]), T.uint32(TMEM_COLUMNS)
                )
            )
            _builder_scope_exit(_builder_then_468_8)
            _builder_scope_exit(_builder_if_468_8)
            _builder_emit(T.ptx.bar.sync(T.uint32(TMEM_ALLOC_BARRIER), T.uint32(320)))
            tmem_base_cg1 = _builder_alloc_scalar("tmem_base_cg1", "int32")
            _builder_emit(
                T.ptx.ld.volatile.shared.s32(tmem_base_cg1, T.address_of(tmem_holding[0]))
            )
            v_consumer_index_cg1 = _builder_scalar("v_consumer_index_cg1", 0, dtype="int32")
            v_consumer_phase_cg1 = _builder_scalar("v_consumer_phase_cg1", 0, dtype="int32")
            gate_consumer_index_cg1 = _builder_scalar("gate_consumer_index_cg1", 0, dtype="int32")
            gate_consumer_phase_cg1 = _builder_scalar("gate_consumer_phase_cg1", 0, dtype="int32")
            shared_consumer_count_cg1 = _builder_scalar(
                "shared_consumer_count_cg1", 0, dtype="int32"
            )
            kv_consumer_count_cg1 = _builder_scalar("kv_consumer_count_cg1", 0, dtype="int32")
            qstate_consumer_count_cg1 = _builder_scalar(
                "qstate_consumer_count_cg1", 0, dtype="int32"
            )
            kv_producer_count_cg1 = _builder_scalar("kv_producer_count_cg1", 0, dtype="int32")
            state_input_producer_count_cg1 = _builder_scalar(
                "state_input_producer_count_cg1", 0, dtype="int32"
            )
            vks_producer_count_cg1 = _builder_scalar("vks_producer_count_cg1", 0, dtype="int32")
            nv_producer_count_cg1 = _builder_scalar("nv_producer_count_cg1", 0, dtype="int32")
            decay_producer_count_cg1 = _builder_scalar("decay_producer_count_cg1", 0, dtype="int32")
            o_producer_index_cg1 = _builder_scalar("o_producer_index_cg1", 0, dtype="int32")
            o_producer_phase_cg1 = _builder_scalar("o_producer_phase_cg1", 1, dtype="int32")
            work_linear_cg1 = _builder_scalar("work_linear_cg1", block, dtype="int32")
            with T.While(work_linear_cg1 < total_work):
                work_u32_cg1 = _builder_scalar(
                    "work_u32_cg1", T.cast(work_linear_cg1, "uint32"), dtype="uint32"
                )
                remain_u32_cg1 = _builder_scalar(
                    "remain_u32_cg1", _udiv_u32_const(work_u32_cg1, HV), dtype="uint32"
                )
                head_cg1 = _builder_scalar(
                    "head_cg1",
                    T.cast(work_u32_cg1 - remain_u32_cg1 * T.uint32(output_heads), "int32"),
                    dtype="int32",
                )
                remain_cg1 = _builder_scalar(
                    "remain_cg1", T.cast(remain_u32_cg1, "int32"), dtype="int32"
                )
                batch_cg1 = _builder_scalar("batch_cg1", remain_cg1, dtype="int32")
                sequence_bounds_cg1 = _builder_name(
                    "sequence_bounds_cg1", T.alloc_local((2,), "int32")
                )
                _builder_emit(_load_sequence_bounds(cu_seqlens, batch_cg1, sequence_bounds_cg1))
                batch_start_cg1 = _builder_scalar(
                    "batch_start_cg1", T.cast(sequence_bounds_cg1[0], "int64"), dtype="int64"
                )
                batch_end_cg1 = _builder_scalar(
                    "batch_end_cg1", T.cast(sequence_bounds_cg1[1], "int64"), dtype="int64"
                )
                seqlen_cg1 = _builder_scalar(
                    "seqlen_cg1", T.cast(batch_end_cg1 - batch_start_cg1, "int32"), dtype="int32"
                )
                num_valid_chunks_cg1 = _builder_scalar(
                    "num_valid_chunks_cg1", (seqlen_cg1 + CHUNK - 1) // CHUNK, dtype="int32"
                )
                num_pairs_cg1 = _builder_scalar(
                    "num_pairs_cg1", (seqlen_cg1 + PAIR_TOKENS - 1) // PAIR_TOKENS, dtype="int32"
                )
                padded_chunks_cg1 = _builder_scalar(
                    "padded_chunks_cg1", num_pairs_cg1 * 2, dtype="int32"
                )
                _builder_if_509_12 = _builder_scope_enter(T.If(padded_chunks_cg1 > 0))
                _builder_then_509_12 = _builder_scope_enter(T.Then())
                initial_count_cg1 = _builder_scalar(
                    "initial_count_cg1", kv_producer_count_cg1, dtype="int32"
                )
                _builder_emit(
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
                )
                T.buffer_store(kv_producer_count_cg1.buffer, kv_producer_count_cg1 + 1, [0])
                with T.serial(0, padded_chunks_cg1) as chunk_cg1:
                    _builder_if_527_20 = _builder_scope_enter(
                        T.If(T.bitwise_and(chunk_cg1, T.int32(1)) == 0)
                    )
                    _builder_then_527_20 = _builder_scope_enter(T.Then())
                    T.buffer_store(kv_producer_count_cg1.buffer, kv_producer_count_cg1 + 1, [0])
                    T.buffer_store(kv_producer_count_cg1.buffer, kv_producer_count_cg1 + 1, [0])
                    _builder_scope_exit(_builder_then_527_20)
                    _builder_scope_exit(_builder_if_527_20)
                    gate_count_cg1 = _builder_scalar(
                        "gate_count_cg1", gate_consumer_index_cg1, dtype="int32"
                    )
                    gate_phase_cg1 = _builder_scalar(
                        "gate_phase_cg1", gate_consumer_phase_cg1, dtype="int32"
                    )
                    _builder_emit(
                        _consumer_wait_state(smem_addr_cg1, 144, gate_count_cg1, gate_phase_cg1)
                    )
                    T.buffer_store(
                        gate_consumer_index_cg1.buffer, _pipe_next_index(gate_count_cg1, 5), [0]
                    )
                    T.buffer_store(
                        gate_consumer_phase_cg1.buffer,
                        _pipe_next_phase(gate_count_cg1, gate_phase_cg1, 5),
                        [0],
                    )
                    gate_stage_cg1 = _builder_scalar(
                        "gate_stage_cg1", gate_count_cg1, dtype="int32"
                    )
                    cumprod_total_cg1 = _builder_alloc_scalar("cumprod_total_cg1", "float32")
                    _builder_emit(
                        T.ptx.ld.shared.f32(
                            cumprod_total_cg1, s_cumprod.ptr_to([gate_stage_cg1 * 64 + 63])
                        )
                    )
                    kv_previous_count_cg1 = _builder_scalar(
                        "kv_previous_count_cg1", kv_consumer_count_cg1, dtype="int32"
                    )
                    _builder_emit(_consumer_wait(smem_addr_cg1, 320, kv_previous_count_cg1, 1))
                    T.buffer_store(kv_consumer_count_cg1.buffer, kv_consumer_count_cg1 + 1, [0])
                    state_input_count_cg1 = _builder_scalar(
                        "state_input_count_cg1", state_input_producer_count_cg1, dtype="int32"
                    )
                    _builder_emit(_producer_acquire(smem_addr_cg1, 472, state_input_count_cg1, 1))
                    T.buffer_store(
                        state_input_producer_count_cg1.buffer,
                        state_input_producer_count_cg1 + 1,
                        [0],
                    )
                    state_values_cg1 = _builder_name(
                        "state_values_cg1", T.alloc_local((128,), "float32")
                    )
                    state_input_words_cg1 = _builder_name(
                        "state_input_words_cg1", T.alloc_local((64,), "uint32")
                    )
                    with T.serial(4) as state_sub_cg1:
                        _builder_emit(
                            _state_tmem_ld_sub(
                                tmem_base_cg1,
                                thread,
                                state_sub_cg1,
                                state_values_cg1,
                                state_sub_cg1 * 32,
                            )
                        )
                    with T.unroll(64) as state_pair_cg1:
                        T.buffer_store(
                            state_input_words_cg1,
                            _pack_f16x2_cg1(
                                state_values_cg1[state_pair_cg1 * 2],
                                state_values_cg1[state_pair_cg1 * 2 + 1],
                            ),
                            [state_pair_cg1],
                        )
                    with T.serial(4) as state_sub_cg1:
                        _builder_emit(
                            _state_input_tmem_st_sub(
                                tmem_base_cg1, thread, state_sub_cg1, state_input_words_cg1
                            )
                        )
                    _builder_emit(T.ptx.tcgen05.wait__st.sync.aligned())
                    _builder_emit(_software_commit(smem_addr_cg1, 464, state_input_count_cg1, 1))
                    with T.unroll(64) as state_pair_cg1:
                        state_mul_cg1 = _builder_alloc_scalar("state_mul_cg1", "uint64")
                        _builder_emit(
                            T.ptx.mul.rn.f32x2(
                                state_mul_cg1,
                                T.cuda.make_float2(
                                    state_values_cg1[state_pair_cg1 * 2],
                                    state_values_cg1[state_pair_cg1 * 2 + 1],
                                ),
                                T.cuda.make_float2(cumprod_total_cg1, cumprod_total_cg1),
                            )
                        )
                        T.buffer_store(
                            state_values_cg1, T.cuda.float2_x(state_mul_cg1), [state_pair_cg1 * 2]
                        )
                        T.buffer_store(
                            state_values_cg1,
                            T.cuda.float2_y(state_mul_cg1),
                            [state_pair_cg1 * 2 + 1],
                        )
                    with T.serial(4) as state_sub_cg1:
                        _builder_emit(
                            _state_tmem_st_sub(
                                tmem_base_cg1,
                                thread,
                                state_sub_cg1,
                                state_values_cg1,
                                state_sub_cg1 * 32,
                            )
                        )
                    _builder_emit(T.ptx.tcgen05.wait__st.sync.aligned())
                    _builder_emit(_consumer_release(smem_addr_cg1, 328, kv_previous_count_cg1, 1))
                    cumprod_factor_cg1 = _builder_name(
                        "cumprod_factor_cg1", T.alloc_local((16,), "float32")
                    )
                    decay_factor_cg1 = _builder_name(
                        "decay_factor_cg1", T.alloc_local((16,), "float32")
                    )
                    factor_col_base_cg1 = _builder_scalar(
                        "factor_col_base_cg1", T.bitwise_and(thread << 1, T.int32(6)), dtype="int32"
                    )
                    last_log_cg1 = _builder_alloc_scalar("last_log_cg1", "float32")
                    _builder_emit(
                        T.ptx.ld.shared.f32(
                            last_log_cg1, s_cumsumlog.ptr_to([gate_stage_cg1 * 64 + 63])
                        )
                    )
                    with T.unroll(8) as factor_group_cg1:
                        factor_col_cg1 = _builder_scalar(
                            "factor_col_cg1",
                            factor_col_base_cg1 + factor_group_cg1 * 8,
                            dtype="int32",
                        )
                        _builder_emit(
                            T.ptx.ld.shared.v2.f32(
                                cumprod_factor_cg1[factor_group_cg1 * 2],
                                cumprod_factor_cg1[factor_group_cg1 * 2 + 1],
                                s_cumprod.ptr_to([gate_stage_cg1 * 64 + factor_col_cg1]),
                            )
                        )
                        decay_diff_cg1 = _builder_name(
                            "decay_diff_cg1", T.alloc_local((1,), "uint64")
                        )
                        cumsumlog_factor_cg1 = _builder_name(
                            "cumsumlog_factor_cg1", T.alloc_local((2,), "float32")
                        )
                        _builder_emit(
                            T.ptx.ld.shared.v2.f32(
                                cumsumlog_factor_cg1[0],
                                cumsumlog_factor_cg1[1],
                                s_cumsumlog.ptr_to([gate_stage_cg1 * 64 + factor_col_cg1]),
                            )
                        )
                        _builder_emit(
                            T.ptx.sub.rn.f32x2(
                                decay_diff_cg1[0],
                                T.cuda.make_float2(last_log_cg1, last_log_cg1),
                                T.cuda.make_float2(
                                    cumsumlog_factor_cg1[0], cumsumlog_factor_cg1[1]
                                ),
                            )
                        )
                        _builder_emit(
                            T.ptx.ex2.approx.ftz.f32(
                                decay_factor_cg1[factor_group_cg1 * 2],
                                T.cuda.float2_x(decay_diff_cg1[0]),
                            )
                        )
                        _builder_emit(
                            T.ptx.ex2.approx.ftz.f32(
                                decay_factor_cg1[factor_group_cg1 * 2 + 1],
                                T.cuda.float2_y(decay_diff_cg1[0]),
                            )
                        )
                    _builder_emit(_consumer_release_state(smem_addr_cg1, 184, gate_count_cg1))
                    vks_count_cg1 = _builder_scalar(
                        "vks_count_cg1", vks_producer_count_cg1, dtype="int32"
                    )
                    T.buffer_store(vks_producer_count_cg1.buffer, vks_producer_count_cg1 + 1, [0])
                    v_count_cg1 = _builder_scalar(
                        "v_count_cg1", v_consumer_index_cg1, dtype="int32"
                    )
                    v_phase_cg1 = _builder_scalar(
                        "v_phase_cg1", v_consumer_phase_cg1, dtype="int32"
                    )
                    _builder_emit(_consumer_wait_state(smem_addr_cg1, 96, v_count_cg1, v_phase_cg1))
                    T.buffer_store(
                        v_consumer_index_cg1.buffer, _pipe_next_index(v_count_cg1, 3), [0]
                    )
                    T.buffer_store(
                        v_consumer_phase_cg1.buffer,
                        _pipe_next_phase(v_count_cg1, v_phase_cg1, 3),
                        [0],
                    )
                    v_words_cg1 = _builder_name("v_words_cg1", T.alloc_local((32,), "uint32"))
                    _builder_emit(_load_v_fragment(smem_raw, v_count_cg1, thread, v_words_cg1))
                    ks_count_cg1 = _builder_scalar(
                        "ks_count_cg1", shared_consumer_count_cg1, dtype="int32"
                    )
                    _builder_emit(_consumer_wait(smem_addr_cg1, 368, ks_count_cg1, 1))
                    T.buffer_store(
                        shared_consumer_count_cg1.buffer, shared_consumer_count_cg1 + 1, [0]
                    )
                    fragment_cg1 = _builder_name("fragment_cg1", T.alloc_local((64,), "float32"))
                    _builder_emit(
                        _cg1_tmem_ld_f32(tmem_base_cg1, TMEM_CG1_ACC_COL, thread, fragment_cg1)
                    )
                    with T.unroll(2) as row_half_cg1:
                        with T.unroll(8) as factor_group_cg1:
                            with T.unroll(2) as factor_repeat_cg1:
                                pair_cg1 = _builder_scalar(
                                    "pair_cg1",
                                    row_half_cg1 * 16 + factor_group_cg1 * 2 + factor_repeat_cg1,
                                    dtype="int32",
                                )
                                ks_mul_cg1 = _builder_alloc_scalar("ks_mul_cg1", "uint64")
                                _builder_emit(
                                    T.ptx.mul.rn.f32x2(
                                        ks_mul_cg1,
                                        T.cuda.make_float2(
                                            fragment_cg1[pair_cg1 * 2],
                                            fragment_cg1[pair_cg1 * 2 + 1],
                                        ),
                                        T.cuda.make_float2(
                                            cumprod_factor_cg1[factor_group_cg1 * 2],
                                            cumprod_factor_cg1[factor_group_cg1 * 2 + 1],
                                        ),
                                    )
                                )
                                T.buffer_store(
                                    fragment_cg1, T.cuda.float2_x(ks_mul_cg1), [pair_cg1 * 2]
                                )
                                T.buffer_store(
                                    fragment_cg1, T.cuda.float2_y(ks_mul_cg1), [pair_cg1 * 2 + 1]
                                )
                    _builder_emit(_consumer_release(smem_addr_cg1, 376, ks_count_cg1, 1))
                    with T.unroll(32) as pair_cg1:
                        ks_word_cg1 = _builder_scalar(
                            "ks_word_cg1",
                            _pack_f16x2_cg1(
                                fragment_cg1[pair_cg1 * 2], fragment_cg1[pair_cg1 * 2 + 1]
                            ),
                            dtype="uint32",
                        )
                        _builder_emit(
                            T.ptx.sub.f16x2(
                                v_words_cg1[pair_cg1], v_words_cg1[pair_cg1], ks_word_cg1
                            )
                        )
                    _builder_emit(
                        _cg1_tmem_st_f16(tmem_base_cg1, TMEM_SHARED_INPUT_COL, thread, v_words_cg1)
                    )
                    _builder_emit(T.ptx.tcgen05.wait__st.sync.aligned())
                    _builder_emit(_software_commit(smem_addr_cg1, 480, vks_count_cg1, 1))
                    qs_count_cg1 = _builder_scalar(
                        "qs_count_cg1", qstate_consumer_count_cg1, dtype="int32"
                    )
                    _builder_emit(_consumer_wait(smem_addr_cg1, 304, qs_count_cg1, 1))
                    T.buffer_store(
                        qstate_consumer_count_cg1.buffer, qstate_consumer_count_cg1 + 1, [0]
                    )
                    _builder_emit(
                        _cg1_tmem_ld_f32(tmem_base_cg1, TMEM_Q_STATE_COL, thread, fragment_cg1)
                    )
                    with T.unroll(2) as row_half_cg1:
                        with T.unroll(8) as factor_group_cg1:
                            with T.unroll(2) as factor_repeat_cg1:
                                pair_cg1 = _builder_scalar(
                                    "pair_cg1",
                                    row_half_cg1 * 16 + factor_group_cg1 * 2 + factor_repeat_cg1,
                                    dtype="int32",
                                )
                                qs_mul_cg1 = _builder_alloc_scalar("qs_mul_cg1", "uint64")
                                _builder_emit(
                                    T.ptx.mul.rn.f32x2(
                                        qs_mul_cg1,
                                        T.cuda.make_float2(
                                            fragment_cg1[pair_cg1 * 2],
                                            fragment_cg1[pair_cg1 * 2 + 1],
                                        ),
                                        T.cuda.make_float2(
                                            cumprod_factor_cg1[factor_group_cg1 * 2],
                                            cumprod_factor_cg1[factor_group_cg1 * 2 + 1],
                                        ),
                                    )
                                )
                                _builder_emit(
                                    T.ptx.mul.rn.f32x2(
                                        qs_mul_cg1, qs_mul_cg1, T.cuda.make_float2(scale, scale)
                                    )
                                )
                                T.buffer_store(
                                    fragment_cg1, T.cuda.float2_x(qs_mul_cg1), [pair_cg1 * 2]
                                )
                                T.buffer_store(
                                    fragment_cg1, T.cuda.float2_y(qs_mul_cg1), [pair_cg1 * 2 + 1]
                                )
                    _builder_emit(
                        _cg1_tmem_st_f32(tmem_base_cg1, TMEM_Q_STATE_COL, thread, fragment_cg1)
                    )
                    _builder_emit(T.ptx.tcgen05.wait__st.sync.aligned())
                    _builder_emit(_consumer_release(smem_addr_cg1, 312, qs_count_cg1, 1))
                    nv_acc_count_cg1 = _builder_scalar(
                        "nv_acc_count_cg1", shared_consumer_count_cg1, dtype="int32"
                    )
                    _builder_emit(_consumer_wait(smem_addr_cg1, 368, nv_acc_count_cg1, 1))
                    T.buffer_store(
                        shared_consumer_count_cg1.buffer, shared_consumer_count_cg1 + 1, [0]
                    )
                    _builder_if_719_20 = _builder_scope_enter(T.If(lane == 0))
                    _builder_then_719_20 = _builder_scope_enter(T.Then())
                    _builder_emit(_consumer_release_state(smem_addr_cg1, 120, v_count_cg1))
                    _builder_scope_exit(_builder_then_719_20)
                    _builder_scope_exit(_builder_if_719_20)
                    _builder_emit(
                        _cg1_tmem_ld_f32(tmem_base_cg1, TMEM_CG1_ACC_COL, thread, fragment_cg1)
                    )
                    nv_words_cg1 = _builder_name("nv_words_cg1", T.alloc_local((32,), "uint32"))
                    with T.unroll(32) as pair_cg1:
                        T.buffer_store(
                            nv_words_cg1,
                            _pack_f16x2_cg1(
                                fragment_cg1[pair_cg1 * 2], fragment_cg1[pair_cg1 * 2 + 1]
                            ),
                            [pair_cg1],
                        )
                    _builder_emit(_consumer_release(smem_addr_cg1, 376, nv_acc_count_cg1, 1))
                    with T.unroll(2) as row_half_cg1:
                        with T.unroll(8) as factor_group_cg1:
                            with T.unroll(2) as factor_repeat_cg1:
                                pair_cg1 = _builder_scalar(
                                    "pair_cg1",
                                    row_half_cg1 * 16 + factor_group_cg1 * 2 + factor_repeat_cg1,
                                    dtype="int32",
                                )
                                decay_mul_cg1 = _builder_alloc_scalar("decay_mul_cg1", "uint64")
                                _builder_emit(
                                    T.ptx.mul.rn.f32x2(
                                        decay_mul_cg1,
                                        T.cuda.make_float2(
                                            fragment_cg1[pair_cg1 * 2],
                                            fragment_cg1[pair_cg1 * 2 + 1],
                                        ),
                                        T.cuda.make_float2(
                                            decay_factor_cg1[factor_group_cg1 * 2],
                                            decay_factor_cg1[factor_group_cg1 * 2 + 1],
                                        ),
                                    )
                                )
                                T.buffer_store(
                                    fragment_cg1, T.cuda.float2_x(decay_mul_cg1), [pair_cg1 * 2]
                                )
                                T.buffer_store(
                                    fragment_cg1, T.cuda.float2_y(decay_mul_cg1), [pair_cg1 * 2 + 1]
                                )
                    nv_count_cg1 = _builder_scalar(
                        "nv_count_cg1", nv_producer_count_cg1, dtype="int32"
                    )
                    T.buffer_store(nv_producer_count_cg1.buffer, nv_producer_count_cg1 + 1, [0])
                    decay_count_cg1 = _builder_scalar(
                        "decay_count_cg1", decay_producer_count_cg1, dtype="int32"
                    )
                    T.buffer_store(
                        decay_producer_count_cg1.buffer, decay_producer_count_cg1 + 1, [0]
                    )
                    decay_words_cg1 = _builder_name(
                        "decay_words_cg1", T.alloc_local((32,), "uint32")
                    )
                    with T.serial(2) as row_half_cg1:
                        _builder_emit(
                            _cg1_tmem_st_f16_half(
                                tmem_base_cg1,
                                TMEM_SHARED_INPUT_COL,
                                thread,
                                row_half_cg1,
                                nv_words_cg1,
                            )
                        )
                        with T.unroll(16) as pair_in_half_cg1:
                            pair_cg1 = _builder_scalar(
                                "pair_cg1", row_half_cg1 * 16 + pair_in_half_cg1, dtype="int32"
                            )
                            T.buffer_store(
                                decay_words_cg1,
                                _pack_f16x2_cg1(
                                    fragment_cg1[pair_cg1 * 2], fragment_cg1[pair_cg1 * 2 + 1]
                                ),
                                [pair_cg1],
                            )
                        _builder_emit(
                            _cg1_tmem_st_f16_half(
                                tmem_base_cg1,
                                TMEM_SHARED_INPUT_COL + 32,
                                thread,
                                row_half_cg1,
                                decay_words_cg1,
                            )
                        )
                    _builder_emit(T.ptx.tcgen05.wait__st.sync.aligned())
                    _builder_emit(_software_commit(smem_addr_cg1, 496, nv_count_cg1, 1))
                    _builder_emit(_software_commit(smem_addr_cg1, 512, decay_count_cg1, 1))
                    o_count_cg1 = _builder_scalar(
                        "o_count_cg1", o_producer_index_cg1, dtype="int32"
                    )
                    o_phase_cg1 = _builder_scalar(
                        "o_phase_cg1", o_producer_phase_cg1, dtype="int32"
                    )
                    _builder_emit(
                        _producer_acquire_state(smem_addr_cg1, 544, o_count_cg1, o_phase_cg1)
                    )
                    T.buffer_store(
                        o_producer_index_cg1.buffer, _pipe_next_index(o_count_cg1, 2), [0]
                    )
                    T.buffer_store(
                        o_producer_phase_cg1.buffer,
                        _pipe_next_phase(o_count_cg1, o_phase_cg1, 2),
                        [0],
                    )
                    qkv_count_cg1 = _builder_scalar(
                        "qkv_count_cg1", qstate_consumer_count_cg1, dtype="int32"
                    )
                    _builder_emit(_consumer_wait(smem_addr_cg1, 304, qkv_count_cg1, 1))
                    T.buffer_store(
                        qstate_consumer_count_cg1.buffer, qstate_consumer_count_cg1 + 1, [0]
                    )
                    _builder_emit(
                        _cg1_tmem_ld_f32(tmem_base_cg1, TMEM_Q_STATE_COL, thread, fragment_cg1)
                    )
                    o_words_cg1 = _builder_name("o_words_cg1", T.alloc_local((32,), "uint32"))
                    with T.unroll(32) as pair_cg1:
                        T.buffer_store(
                            o_words_cg1,
                            _pack_f16x2_cg1(
                                fragment_cg1[pair_cg1 * 2], fragment_cg1[pair_cg1 * 2 + 1]
                            ),
                            [pair_cg1],
                        )
                    _builder_emit(_store_o_fragment(smem_raw, o_count_cg1, thread, o_words_cg1))
                    _builder_emit(T.ptx.fence.proxy.async_.shared__cta())
                    _builder_emit(_consumer_release(smem_addr_cg1, 312, qkv_count_cg1, 1))
                    _builder_emit(_software_commit_state(smem_addr_cg1, 528, o_count_cg1))
                final_count_cg1 = _builder_scalar(
                    "final_count_cg1", kv_consumer_count_cg1, dtype="int32"
                )
                _builder_emit(
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
                )
                T.buffer_store(kv_consumer_count_cg1.buffer, kv_consumer_count_cg1 + 1, [0])
                _builder_scope_exit(_builder_then_509_12)
                _builder_else_509_12 = _builder_scope_enter(T.Else())
                _builder_emit(
                    _copy_empty_state(
                        initial_state, final_state, batch_cg1, head_cg1, thread, HV=HV
                    )
                )
                _builder_scope_exit(_builder_else_509_12)
                _builder_scope_exit(_builder_if_509_12)
                T.buffer_store(work_linear_cg1.buffer, work_linear_cg1 + grid_x, [0])
            _builder_emit(T.cuda.warpgroup_sync(CG1_TMEM_DEALLOC_BARRIER))
            _builder_if_813_8 = _builder_scope_enter(T.If(warp == 4))
            _builder_then_813_8 = _builder_scope_enter(T.Then())
            _builder_emit(T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned())
            _builder_emit(
                T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(
                    T.cast(tmem_base_cg1, "uint32"), T.uint32(TMEM_COLUMNS)
                )
            )
            _builder_scope_exit(_builder_then_813_8)
            _builder_scope_exit(_builder_if_813_8)
            _builder_emit(
                _producer_tail_state(
                    smem_addr_cg1, 544, o_producer_index_cg1, o_producer_phase_cg1, 2
                )
            )
            _builder_emit(_producer_tail(smem_addr_cg1, 472, state_input_producer_count_cg1, 1))
            _builder_scope_exit(_builder_then_465_4)
            _builder_else_465_4 = _builder_scope_enter(T.Else())
            _builder_if_820_4 = _builder_scope_enter(T.If(warp == 8))
            _builder_then_820_4 = _builder_scope_enter(T.Then())
            smem_addr_issuer0 = _builder_scalar(
                "smem_addr_issuer0",
                T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0])),
                dtype="uint32",
            )
            _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(24))
            _builder_emit(T.ptx.bar.sync(T.uint32(TMEM_ALLOC_BARRIER), T.uint32(320)))
            tmem_base_issuer0 = _builder_alloc_scalar("tmem_base_issuer0", "int32")
            _builder_emit(
                T.ptx.ld.volatile.shared.s32(tmem_base_issuer0, T.address_of(tmem_holding[0]))
            )
            cg0_producer_index_i0 = _builder_scalar("cg0_producer_index_i0", 0, dtype="int32")
            cg0_producer_phase_i0 = _builder_scalar("cg0_producer_phase_i0", 1, dtype="int32")
            k_consumer_index_i0 = _builder_scalar("k_consumer_index_i0", 0, dtype="int32")
            k_consumer_phase_i0 = _builder_scalar("k_consumer_phase_i0", 0, dtype="int32")
            q_consumer_index_i0 = _builder_scalar("q_consumer_index_i0", 0, dtype="int32")
            q_consumer_phase_i0 = _builder_scalar("q_consumer_phase_i0", 0, dtype="int32")
            work_linear_issuer0 = _builder_scalar("work_linear_issuer0", block, dtype="int32")
            with T.While(work_linear_issuer0 < total_work):
                remain_issuer0 = _builder_scalar(
                    "remain_issuer0",
                    T.cast(_udiv_u32_const(T.cast(work_linear_issuer0, "uint32"), HV), "int32"),
                    dtype="int32",
                )
                batch_issuer0 = _builder_scalar("batch_issuer0", remain_issuer0, dtype="int32")
                sequence_bounds_issuer0 = _builder_name(
                    "sequence_bounds_issuer0", T.alloc_local((2,), "int32")
                )
                _builder_emit(
                    _load_sequence_bounds(cu_seqlens, batch_issuer0, sequence_bounds_issuer0)
                )
                batch_start_issuer0 = _builder_scalar(
                    "batch_start_issuer0",
                    T.cast(sequence_bounds_issuer0[0], "int64"),
                    dtype="int64",
                )
                batch_end_issuer0 = _builder_scalar(
                    "batch_end_issuer0", T.cast(sequence_bounds_issuer0[1], "int64"), dtype="int64"
                )
                seqlen_issuer0 = _builder_scalar(
                    "seqlen_issuer0",
                    T.cast(batch_end_issuer0 - batch_start_issuer0, "int32"),
                    dtype="int32",
                )
                num_pairs_issuer0 = _builder_scalar(
                    "num_pairs_issuer0",
                    (seqlen_issuer0 + PAIR_TOKENS - 1) // PAIR_TOKENS,
                    dtype="int32",
                )
                with T.serial(0, num_pairs_issuer0) as _pair_issuer0:
                    kk0_count_i0 = _builder_scalar(
                        "kk0_count_i0", cg0_producer_index_i0, dtype="int32"
                    )
                    kk0_phase_i0 = _builder_scalar(
                        "kk0_phase_i0", cg0_producer_phase_i0, dtype="int32"
                    )
                    _builder_emit(
                        _producer_acquire_state(smem_addr_issuer0, 352, kk0_count_i0, kk0_phase_i0)
                    )
                    T.buffer_store(
                        cg0_producer_index_i0.buffer, _pipe_next_index(kk0_count_i0, 2), [0]
                    )
                    T.buffer_store(
                        cg0_producer_phase_i0.buffer,
                        _pipe_next_phase(kk0_count_i0, kk0_phase_i0, 2),
                        [0],
                    )
                    k0_count_i0 = _builder_scalar("k0_count_i0", k_consumer_index_i0, dtype="int32")
                    k0_phase_i0 = _builder_scalar("k0_phase_i0", k_consumer_phase_i0, dtype="int32")
                    _builder_emit(
                        _consumer_wait_state(smem_addr_issuer0, 0, k0_count_i0, k0_phase_i0)
                    )
                    T.buffer_store(
                        k_consumer_index_i0.buffer, _pipe_next_index(k0_count_i0, 4), [0]
                    )
                    T.buffer_store(
                        k_consumer_phase_i0.buffer,
                        _pipe_next_phase(k0_count_i0, k0_phase_i0, 4),
                        [0],
                    )
                    k0_desc_i0 = _builder_scalar(
                        "k0_desc_i0",
                        _smem_desc_b128(
                            _shared_addr(smem_addr_issuer0, SMEM_K_OFF + k0_count_i0 * 16384)
                        ),
                        dtype="uint64",
                    )
                    _builder_emit(
                        _mma_ss_64x64_k128(
                            tmem_base_issuer0 + TMEM_CG0_ACC_COL + kk0_count_i0 * 64,
                            k0_desc_i0,
                            k0_desc_i0,
                            _shared_addr(smem_addr_issuer0, 336 + kk0_count_i0 * 8),
                        )
                    )
                    kk1_count_i0 = _builder_scalar(
                        "kk1_count_i0", cg0_producer_index_i0, dtype="int32"
                    )
                    kk1_phase_i0 = _builder_scalar(
                        "kk1_phase_i0", cg0_producer_phase_i0, dtype="int32"
                    )
                    _builder_emit(
                        _producer_acquire_state(smem_addr_issuer0, 352, kk1_count_i0, kk1_phase_i0)
                    )
                    T.buffer_store(
                        cg0_producer_index_i0.buffer, _pipe_next_index(kk1_count_i0, 2), [0]
                    )
                    T.buffer_store(
                        cg0_producer_phase_i0.buffer,
                        _pipe_next_phase(kk1_count_i0, kk1_phase_i0, 2),
                        [0],
                    )
                    k1_count_i0 = _builder_scalar("k1_count_i0", k_consumer_index_i0, dtype="int32")
                    k1_phase_i0 = _builder_scalar("k1_phase_i0", k_consumer_phase_i0, dtype="int32")
                    _builder_emit(
                        _consumer_wait_state(smem_addr_issuer0, 0, k1_count_i0, k1_phase_i0)
                    )
                    T.buffer_store(
                        k_consumer_index_i0.buffer, _pipe_next_index(k1_count_i0, 4), [0]
                    )
                    T.buffer_store(
                        k_consumer_phase_i0.buffer,
                        _pipe_next_phase(k1_count_i0, k1_phase_i0, 4),
                        [0],
                    )
                    k1_desc_i0 = _builder_scalar(
                        "k1_desc_i0",
                        _smem_desc_b128(
                            _shared_addr(smem_addr_issuer0, SMEM_K_OFF + k1_count_i0 * 16384)
                        ),
                        dtype="uint64",
                    )
                    _builder_emit(
                        _mma_ss_64x64_k128(
                            tmem_base_issuer0 + TMEM_CG0_ACC_COL + kk1_count_i0 * 64,
                            k1_desc_i0,
                            k1_desc_i0,
                            _shared_addr(smem_addr_issuer0, 336 + kk1_count_i0 * 8),
                        )
                    )
                    q0_count_i0 = _builder_scalar("q0_count_i0", q_consumer_index_i0, dtype="int32")
                    q0_phase_i0 = _builder_scalar("q0_phase_i0", q_consumer_phase_i0, dtype="int32")
                    _builder_emit(
                        _consumer_wait_state(smem_addr_issuer0, 64, q0_count_i0, q0_phase_i0)
                    )
                    T.buffer_store(
                        q_consumer_index_i0.buffer, _pipe_next_index(q0_count_i0, 2), [0]
                    )
                    T.buffer_store(
                        q_consumer_phase_i0.buffer,
                        _pipe_next_phase(q0_count_i0, q0_phase_i0, 2),
                        [0],
                    )
                    qk0_count_i0 = _builder_scalar(
                        "qk0_count_i0", cg0_producer_index_i0, dtype="int32"
                    )
                    qk0_phase_i0 = _builder_scalar(
                        "qk0_phase_i0", cg0_producer_phase_i0, dtype="int32"
                    )
                    _builder_emit(
                        _producer_acquire_state(smem_addr_issuer0, 352, qk0_count_i0, qk0_phase_i0)
                    )
                    T.buffer_store(
                        cg0_producer_index_i0.buffer, _pipe_next_index(qk0_count_i0, 2), [0]
                    )
                    T.buffer_store(
                        cg0_producer_phase_i0.buffer,
                        _pipe_next_phase(qk0_count_i0, qk0_phase_i0, 2),
                        [0],
                    )
                    q0_desc_i0 = _builder_scalar(
                        "q0_desc_i0",
                        _smem_desc_b128(
                            _shared_addr(smem_addr_issuer0, SMEM_Q_OFF + q0_count_i0 * 16384)
                        ),
                        dtype="uint64",
                    )
                    _builder_emit(
                        _mma_ss_64x64_k128(
                            tmem_base_issuer0 + TMEM_CG0_ACC_COL + qk0_count_i0 * 64,
                            q0_desc_i0,
                            k0_desc_i0,
                            _shared_addr(smem_addr_issuer0, 336 + qk0_count_i0 * 8),
                        )
                    )
                    q1_count_i0 = _builder_scalar("q1_count_i0", q_consumer_index_i0, dtype="int32")
                    q1_phase_i0 = _builder_scalar("q1_phase_i0", q_consumer_phase_i0, dtype="int32")
                    _builder_emit(
                        _consumer_wait_state(smem_addr_issuer0, 64, q1_count_i0, q1_phase_i0)
                    )
                    T.buffer_store(
                        q_consumer_index_i0.buffer, _pipe_next_index(q1_count_i0, 2), [0]
                    )
                    T.buffer_store(
                        q_consumer_phase_i0.buffer,
                        _pipe_next_phase(q1_count_i0, q1_phase_i0, 2),
                        [0],
                    )
                    qk1_count_i0 = _builder_scalar(
                        "qk1_count_i0", cg0_producer_index_i0, dtype="int32"
                    )
                    qk1_phase_i0 = _builder_scalar(
                        "qk1_phase_i0", cg0_producer_phase_i0, dtype="int32"
                    )
                    _builder_emit(
                        _producer_acquire_state(smem_addr_issuer0, 352, qk1_count_i0, qk1_phase_i0)
                    )
                    T.buffer_store(
                        cg0_producer_index_i0.buffer, _pipe_next_index(qk1_count_i0, 2), [0]
                    )
                    T.buffer_store(
                        cg0_producer_phase_i0.buffer,
                        _pipe_next_phase(qk1_count_i0, qk1_phase_i0, 2),
                        [0],
                    )
                    q1_desc_i0 = _builder_scalar(
                        "q1_desc_i0",
                        _smem_desc_b128(
                            _shared_addr(smem_addr_issuer0, SMEM_Q_OFF + q1_count_i0 * 16384)
                        ),
                        dtype="uint64",
                    )
                    _builder_emit(
                        _mma_ss_64x64_k128(
                            tmem_base_issuer0 + TMEM_CG0_ACC_COL + qk1_count_i0 * 64,
                            q1_desc_i0,
                            k1_desc_i0,
                            _shared_addr(smem_addr_issuer0, 336 + qk1_count_i0 * 8),
                        )
                    )
                    _builder_emit(
                        _mma_commit(_shared_addr(smem_addr_issuer0, 80 + q0_count_i0 * 8))
                    )
                    _builder_emit(
                        _mma_commit(_shared_addr(smem_addr_issuer0, 80 + q1_count_i0 * 8))
                    )
                    _builder_emit(
                        _mma_commit(_shared_addr(smem_addr_issuer0, 32 + k0_count_i0 * 8))
                    )
                    _builder_emit(
                        _mma_commit(_shared_addr(smem_addr_issuer0, 32 + k1_count_i0 * 8))
                    )
                T.buffer_store(work_linear_issuer0.buffer, work_linear_issuer0 + grid_x, [0])
            _builder_emit(
                _producer_tail_state(
                    smem_addr_issuer0, 352, cg0_producer_index_i0, cg0_producer_phase_i0, 2
                )
            )
            _builder_scope_exit(_builder_then_820_4)
            _builder_else_820_4 = _builder_scope_enter(T.Else())
            _builder_if_937_4 = _builder_scope_enter(T.If(warp == 10))
            _builder_then_937_4 = _builder_scope_enter(T.Then())
            smem_addr_issuer1 = _builder_scalar(
                "smem_addr_issuer1",
                T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0])),
                dtype="uint32",
            )
            _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(24))
            _builder_emit(T.ptx.bar.sync(T.uint32(TMEM_ALLOC_BARRIER), T.uint32(320)))
            tmem_base_issuer1 = _builder_alloc_scalar("tmem_base_issuer1", "int32")
            _builder_emit(
                T.ptx.ld.volatile.shared.s32(tmem_base_issuer1, T.address_of(tmem_holding[0]))
            )
            cg1_producer_count_i1 = _builder_scalar("cg1_producer_count_i1", 0, dtype="int32")
            qstate_producer_count_i1 = _builder_scalar("qstate_producer_count_i1", 0, dtype="int32")
            kv_producer_count_i1 = _builder_scalar("kv_producer_count_i1", 0, dtype="int32")
            k_consumer_index_i1 = _builder_scalar("k_consumer_index_i1", 0, dtype="int32")
            k_consumer_phase_i1 = _builder_scalar("k_consumer_phase_i1", 0, dtype="int32")
            q_consumer_index_i1 = _builder_scalar("q_consumer_index_i1", 0, dtype="int32")
            q_consumer_phase_i1 = _builder_scalar("q_consumer_phase_i1", 0, dtype="int32")
            ainv_consumer_index_i1 = _builder_scalar("ainv_consumer_index_i1", 0, dtype="int32")
            ainv_consumer_phase_i1 = _builder_scalar("ainv_consumer_phase_i1", 0, dtype="int32")
            qk_consumer_index_i1 = _builder_scalar("qk_consumer_index_i1", 0, dtype="int32")
            qk_consumer_phase_i1 = _builder_scalar("qk_consumer_phase_i1", 0, dtype="int32")
            state_input_consumer_count_i1 = _builder_scalar(
                "state_input_consumer_count_i1", 0, dtype="int32"
            )
            vks_consumer_count_i1 = _builder_scalar("vks_consumer_count_i1", 0, dtype="int32")
            nv_consumer_count_i1 = _builder_scalar("nv_consumer_count_i1", 0, dtype="int32")
            decay_consumer_count_i1 = _builder_scalar("decay_consumer_count_i1", 0, dtype="int32")
            work_linear_issuer1 = _builder_scalar("work_linear_issuer1", block, dtype="int32")
            with T.While(work_linear_issuer1 < total_work):
                remain_issuer1 = _builder_scalar(
                    "remain_issuer1",
                    T.cast(_udiv_u32_const(T.cast(work_linear_issuer1, "uint32"), HV), "int32"),
                    dtype="int32",
                )
                batch_issuer1 = _builder_scalar("batch_issuer1", remain_issuer1, dtype="int32")
                sequence_bounds_issuer1 = _builder_name(
                    "sequence_bounds_issuer1", T.alloc_local((2,), "int32")
                )
                _builder_emit(
                    _load_sequence_bounds(cu_seqlens, batch_issuer1, sequence_bounds_issuer1)
                )
                batch_start_issuer1 = _builder_scalar(
                    "batch_start_issuer1",
                    T.cast(sequence_bounds_issuer1[0], "int64"),
                    dtype="int64",
                )
                batch_end_issuer1 = _builder_scalar(
                    "batch_end_issuer1", T.cast(sequence_bounds_issuer1[1], "int64"), dtype="int64"
                )
                seqlen_issuer1 = _builder_scalar(
                    "seqlen_issuer1",
                    T.cast(batch_end_issuer1 - batch_start_issuer1, "int32"),
                    dtype="int32",
                )
                num_pairs_issuer1 = _builder_scalar(
                    "num_pairs_issuer1",
                    (seqlen_issuer1 + PAIR_TOKENS - 1) // PAIR_TOKENS,
                    dtype="int32",
                )
                padded_chunks_issuer1 = _builder_scalar(
                    "padded_chunks_issuer1", num_pairs_issuer1 * 2, dtype="int32"
                )
                with T.serial(0, padded_chunks_issuer1) as chunk_issuer1:
                    k_count_i1 = _builder_scalar("k_count_i1", k_consumer_index_i1, dtype="int32")
                    k_phase_i1 = _builder_scalar("k_phase_i1", k_consumer_phase_i1, dtype="int32")
                    _builder_emit(
                        _consumer_wait_state(smem_addr_issuer1, 0, k_count_i1, k_phase_i1)
                    )
                    k_desc_i1 = _builder_scalar(
                        "k_desc_i1",
                        _smem_desc_b128(
                            _shared_addr(smem_addr_issuer1, SMEM_K_OFF + k_count_i1 * 16384)
                        ),
                        dtype="uint64",
                    )
                    q_count_i1 = _builder_scalar("q_count_i1", q_consumer_index_i1, dtype="int32")
                    q_phase_i1 = _builder_scalar("q_phase_i1", q_consumer_phase_i1, dtype="int32")
                    _builder_emit(
                        _consumer_wait_state(smem_addr_issuer1, 64, q_count_i1, q_phase_i1)
                    )
                    q_desc_i1 = _builder_scalar(
                        "q_desc_i1",
                        _smem_desc_b128(
                            _shared_addr(smem_addr_issuer1, SMEM_Q_OFF + q_count_i1 * 16384)
                        ),
                        dtype="uint64",
                    )
                    T.buffer_store(k_consumer_index_i1.buffer, _pipe_next_index(k_count_i1, 4), [0])
                    T.buffer_store(
                        k_consumer_phase_i1.buffer, _pipe_next_phase(k_count_i1, k_phase_i1, 4), [0]
                    )
                    T.buffer_store(q_consumer_index_i1.buffer, _pipe_next_index(q_count_i1, 2), [0])
                    T.buffer_store(
                        q_consumer_phase_i1.buffer, _pipe_next_phase(q_count_i1, q_phase_i1, 2), [0]
                    )
                    ks_count_i1 = _builder_scalar(
                        "ks_count_i1", cg1_producer_count_i1, dtype="int32"
                    )
                    _builder_emit(_producer_acquire(smem_addr_issuer1, 376, ks_count_i1, 1))
                    state_input_count_i1 = _builder_scalar(
                        "state_input_count_i1", state_input_consumer_count_i1, dtype="int32"
                    )
                    _builder_emit(_consumer_wait(smem_addr_issuer1, 464, state_input_count_i1, 1))
                    _builder_emit(
                        _mma_ts_128x64_k128(
                            tmem_base_issuer1 + TMEM_CG1_ACC_COL,
                            tmem_base_issuer1 + TMEM_STATE_INPUT_COL,
                            k_desc_i1,
                            _pipe_full_addr(smem_addr_issuer1, 368, ks_count_i1, 1),
                        )
                    )
                    T.buffer_store(cg1_producer_count_i1.buffer, cg1_producer_count_i1 + 1, [0])
                    T.buffer_store(
                        state_input_consumer_count_i1.buffer, state_input_consumer_count_i1 + 1, [0]
                    )
                    qs_count_i1 = _builder_scalar(
                        "qs_count_i1", qstate_producer_count_i1, dtype="int32"
                    )
                    _builder_emit(_producer_acquire(smem_addr_issuer1, 312, qs_count_i1, 1))
                    _builder_emit(
                        _mma_ts_128x64_k128(
                            tmem_base_issuer1 + TMEM_Q_STATE_COL,
                            tmem_base_issuer1 + TMEM_STATE_INPUT_COL,
                            q_desc_i1,
                            _pipe_full_addr(smem_addr_issuer1, 304, qs_count_i1, 1),
                        )
                    )
                    T.buffer_store(
                        qstate_producer_count_i1.buffer, qstate_producer_count_i1 + 1, [0]
                    )
                    _builder_emit(
                        _mma_commit(
                            _pipe_empty_addr(smem_addr_issuer1, 472, state_input_count_i1, 1)
                        )
                    )
                    _builder_emit(_mma_commit(_shared_addr(smem_addr_issuer1, 80 + q_count_i1 * 8)))
                    nv_acc_count_i1 = _builder_scalar(
                        "nv_acc_count_i1", cg1_producer_count_i1, dtype="int32"
                    )
                    _builder_emit(_producer_acquire(smem_addr_issuer1, 376, nv_acc_count_i1, 1))
                    _builder_emit(_consumer_wait(smem_addr_issuer1, 480, vks_consumer_count_i1, 1))
                    T.buffer_store(vks_consumer_count_i1.buffer, vks_consumer_count_i1 + 1, [0])
                    ainv_count_i1 = _builder_scalar(
                        "ainv_count_i1", ainv_consumer_index_i1, dtype="int32"
                    )
                    ainv_phase_i1 = _builder_scalar(
                        "ainv_phase_i1", ainv_consumer_phase_i1, dtype="int32"
                    )
                    _builder_emit(
                        _consumer_wait_state(smem_addr_issuer1, 384, ainv_count_i1, ainv_phase_i1)
                    )
                    ainv_desc_i1 = _builder_scalar(
                        "ainv_desc_i1",
                        _smem_desc_b128(
                            _shared_addr(smem_addr_issuer1, SMEM_AINV_OFF + ainv_count_i1 * 8192)
                        ),
                        dtype="uint64",
                    )
                    _builder_emit(
                        _mma_ts_128x64_k64(
                            tmem_base_issuer1 + TMEM_CG1_ACC_COL,
                            tmem_base_issuer1 + TMEM_SHARED_INPUT_COL,
                            ainv_desc_i1,
                            0,
                        )
                    )
                    _builder_emit(
                        _mma_commit(_pipe_full_addr(smem_addr_issuer1, 368, nv_acc_count_i1, 1))
                    )
                    T.buffer_store(cg1_producer_count_i1.buffer, cg1_producer_count_i1 + 1, [0])
                    T.buffer_store(
                        ainv_consumer_index_i1.buffer, _pipe_next_index(ainv_count_i1, 3), [0]
                    )
                    T.buffer_store(
                        ainv_consumer_phase_i1.buffer,
                        _pipe_next_phase(ainv_count_i1, ainv_phase_i1, 3),
                        [0],
                    )
                    _builder_emit(
                        _mma_commit(_shared_addr(smem_addr_issuer1, 408 + ainv_count_i1 * 8))
                    )
                    qkv_count_i1 = _builder_scalar(
                        "qkv_count_i1", qstate_producer_count_i1, dtype="int32"
                    )
                    _builder_emit(_producer_acquire(smem_addr_issuer1, 312, qkv_count_i1, 1))
                    qk_count_i1 = _builder_scalar(
                        "qk_count_i1", qk_consumer_index_i1, dtype="int32"
                    )
                    qk_phase_i1 = _builder_scalar(
                        "qk_phase_i1", qk_consumer_phase_i1, dtype="int32"
                    )
                    _builder_emit(
                        _consumer_wait_state(smem_addr_issuer1, 432, qk_count_i1, qk_phase_i1)
                    )
                    qk_desc_i1 = _builder_scalar(
                        "qk_desc_i1",
                        _smem_desc_b128(
                            _shared_addr(smem_addr_issuer1, SMEM_QK_OFF + qk_count_i1 * 8192)
                        ),
                        dtype="uint64",
                    )
                    _builder_emit(_consumer_wait(smem_addr_issuer1, 496, nv_consumer_count_i1, 1))
                    T.buffer_store(nv_consumer_count_i1.buffer, nv_consumer_count_i1 + 1, [0])
                    _builder_emit(
                        _mma_ts_128x64_k64(
                            tmem_base_issuer1 + TMEM_Q_STATE_COL,
                            tmem_base_issuer1 + TMEM_SHARED_INPUT_COL,
                            qk_desc_i1,
                            1,
                        )
                    )
                    T.buffer_store(
                        qk_consumer_index_i1.buffer, _pipe_next_index(qk_count_i1, 2), [0]
                    )
                    T.buffer_store(
                        qk_consumer_phase_i1.buffer,
                        _pipe_next_phase(qk_count_i1, qk_phase_i1, 2),
                        [0],
                    )
                    _builder_emit(
                        _mma_commit(_shared_addr(smem_addr_issuer1, 448 + qk_count_i1 * 8))
                    )
                    _builder_emit(
                        _mma_commit(_pipe_full_addr(smem_addr_issuer1, 304, qkv_count_i1, 1))
                    )
                    T.buffer_store(
                        qstate_producer_count_i1.buffer, qstate_producer_count_i1 + 1, [0]
                    )
                    _builder_if_1060_16 = _builder_scope_enter(T.If(chunk_issuer1 == 0))
                    _builder_then_1060_16 = _builder_scope_enter(T.Then())
                    T.buffer_store(kv_producer_count_i1.buffer, kv_producer_count_i1 + 1, [0])
                    _builder_scope_exit(_builder_then_1060_16)
                    _builder_scope_exit(_builder_if_1060_16)
                    kv_count_i1 = _builder_scalar(
                        "kv_count_i1", kv_producer_count_i1, dtype="int32"
                    )
                    _builder_emit(_producer_acquire(smem_addr_issuer1, 328, kv_count_i1, 1))
                    _builder_emit(
                        _consumer_wait(smem_addr_issuer1, 512, decay_consumer_count_i1, 1)
                    )
                    T.buffer_store(decay_consumer_count_i1.buffer, decay_consumer_count_i1 + 1, [0])
                    kt_desc_i1 = _builder_scalar(
                        "kt_desc_i1",
                        _smem_desc_k_trans_b128(
                            _shared_addr(smem_addr_issuer1, SMEM_K_OFF + k_count_i1 * 16384)
                        ),
                        dtype="uint64",
                    )
                    _builder_emit(
                        _mma_ts_128x128_k64(
                            tmem_base_issuer1 + TMEM_STATE_COL,
                            tmem_base_issuer1 + TMEM_SHARED_INPUT_COL + 32,
                            kt_desc_i1,
                            _pipe_full_addr(smem_addr_issuer1, 320, kv_count_i1, 1),
                        )
                    )
                    T.buffer_store(kv_producer_count_i1.buffer, kv_producer_count_i1 + 1, [0])
                    _builder_emit(_mma_commit(_shared_addr(smem_addr_issuer1, 32 + k_count_i1 * 8)))
                T.buffer_store(work_linear_issuer1.buffer, work_linear_issuer1 + grid_x, [0])
            _builder_emit(_producer_tail(smem_addr_issuer1, 376, cg1_producer_count_i1, 1))
            _builder_emit(_producer_tail(smem_addr_issuer1, 312, qstate_producer_count_i1, 1))
            _builder_emit(_producer_tail(smem_addr_issuer1, 328, kv_producer_count_i1, 1))
            _builder_scope_exit(_builder_then_937_4)
            _builder_else_937_4 = _builder_scope_enter(T.Else())
            _builder_if_1081_4 = _builder_scope_enter(T.If(warp == 9))
            _builder_then_1081_4 = _builder_scope_enter(T.Then())
            smem_addr_tma = _builder_scalar(
                "smem_addr_tma", T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0])), dtype="uint32"
            )
            _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(24))
            q_producer_index = _builder_scalar("q_producer_index", 0, dtype="int32")
            q_producer_phase = _builder_scalar("q_producer_phase", 1, dtype="int32")
            k_producer_index = _builder_scalar("k_producer_index", 0, dtype="int32")
            k_producer_phase = _builder_scalar("k_producer_phase", 1, dtype="int32")
            v_producer_index = _builder_scalar("v_producer_index", 0, dtype="int32")
            v_producer_phase = _builder_scalar("v_producer_phase", 1, dtype="int32")
            work_linear_tma = _builder_scalar("work_linear_tma", block, dtype="int32")
            _builder_if_1094_8 = _builder_scope_enter(T.If(work_linear_tma < total_work))
            _builder_then_1094_8 = _builder_scope_enter(T.Then())
            _builder_if_1095_12 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
            _builder_then_1095_12 = _builder_scope_enter(T.Then())
            _builder_emit(_descriptor_copy_payload(q_map, descriptor_q))
            _builder_scope_exit(_builder_then_1095_12)
            _builder_scope_exit(_builder_if_1095_12)
            _builder_emit(T.cuda.warp_sync())
            _builder_if_1098_12 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
            _builder_then_1098_12 = _builder_scope_enter(T.Then())
            _builder_emit(_descriptor_copy_payload(k_map, descriptor_k))
            _builder_scope_exit(_builder_then_1098_12)
            _builder_scope_exit(_builder_if_1098_12)
            _builder_emit(T.cuda.warp_sync())
            _builder_if_1101_12 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
            _builder_then_1101_12 = _builder_scope_enter(T.Then())
            _builder_emit(_descriptor_copy_payload(v_map, descriptor_v))
            _builder_scope_exit(_builder_then_1101_12)
            _builder_scope_exit(_builder_if_1101_12)
            _builder_emit(T.cuda.warp_sync())
            _builder_emit(T.ptx.fence.acq_rel.cta())
            _builder_scope_exit(_builder_then_1094_8)
            _builder_scope_exit(_builder_if_1094_8)
            with T.While(work_linear_tma < total_work):
                work_u32_tma = _builder_scalar(
                    "work_u32_tma", T.cast(work_linear_tma, "uint32"), dtype="uint32"
                )
                remain_u32_tma = _builder_scalar(
                    "remain_u32_tma", _udiv_u32_const(work_u32_tma, HV), dtype="uint32"
                )
                head_tma = _builder_scalar(
                    "head_tma",
                    T.cast(work_u32_tma - remain_u32_tma * T.uint32(output_heads), "int32"),
                    dtype="int32",
                )
                qk_head_tma = _builder_scalar("qk_head_tma", head_tma, dtype="int32")
                value_subhead_tma = _builder_scalar("value_subhead_tma", 0, dtype="int32")
                if HQ != HV:
                    T.buffer_store(
                        qk_head_tma.buffer,
                        T.cast(_udiv_u32_const(T.cast(head_tma, "uint32"), HV // HQ), "int32"),
                        [0],
                    )
                    T.buffer_store(
                        value_subhead_tma.buffer, head_tma - qk_head_tma * (HV // HQ), [0]
                    )
                remain_tma = _builder_scalar(
                    "remain_tma", T.cast(remain_u32_tma, "int32"), dtype="int32"
                )
                batch_tma = _builder_scalar("batch_tma", remain_tma, dtype="int32")
                sequence_bounds_tma = _builder_name(
                    "sequence_bounds_tma", T.alloc_local((2,), "int32")
                )
                _builder_emit(_load_sequence_bounds(cu_seqlens, batch_tma, sequence_bounds_tma))
                batch_start_tma = _builder_scalar(
                    "batch_start_tma", T.cast(sequence_bounds_tma[0], "int64"), dtype="int64"
                )
                batch_end_tma = _builder_scalar(
                    "batch_end_tma", T.cast(sequence_bounds_tma[1], "int64"), dtype="int64"
                )
                seqlen_tma = _builder_scalar(
                    "seqlen_tma", T.cast(batch_end_tma - batch_start_tma, "int32"), dtype="int32"
                )
                num_valid_chunks_tma = _builder_scalar(
                    "num_valid_chunks_tma", (seqlen_tma + CHUNK - 1) // CHUNK, dtype="int32"
                )
                num_pairs_tma = _builder_scalar(
                    "num_pairs_tma", (seqlen_tma + PAIR_TOKENS - 1) // PAIR_TOKENS, dtype="int32"
                )
                padded_chunks_tma = _builder_scalar(
                    "padded_chunks_tma", num_pairs_tma * 2, dtype="int32"
                )
                _builder_if_1128_12 = _builder_scope_enter(T.If(padded_chunks_tma > 0))
                _builder_then_1128_12 = _builder_scope_enter(T.Then())
                _builder_if_1129_16 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                _builder_then_1129_16 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.cp.async_.bulk.wait_group.read(0))
                _builder_scope_exit(_builder_then_1129_16)
                _builder_scope_exit(_builder_if_1129_16)
                _builder_emit(T.cuda.warp_sync())
                _builder_if_1132_16 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                _builder_then_1132_16 = _builder_scope_enter(T.Then())
                _builder_emit(
                    _replace_descriptor(
                        descriptor_q, q.data, batch_end_tma, HQ, 1, 2 * D_HEAD * HQ, 2 * D_HEAD, 0
                    )
                )
                _builder_emit(
                    _replace_descriptor(
                        descriptor_k, k.data, batch_end_tma, HQ, 1, 2 * D_HEAD * HQ, 2 * D_HEAD, 0
                    )
                )
                if HQ == HV:
                    _builder_emit(
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
                    )
                else:
                    _builder_emit(
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
                    )
                _builder_scope_exit(_builder_then_1132_16)
                _builder_scope_exit(_builder_if_1132_16)
                _builder_emit(T.cuda.warp_sync())
                _builder_emit(_tensormap_release())
                with T.serial(0, num_valid_chunks_tma - 1) as chunk_idx_tma:
                    chunk_offset_tma = _builder_scalar(
                        "chunk_offset_tma",
                        batch_start_tma + T.cast(chunk_idx_tma * CHUNK, "int64"),
                        dtype="int64",
                    )
                    _builder_emit(
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
                    )
                    T.buffer_store(
                        q_producer_phase.buffer,
                        _pipe_next_phase(q_producer_index, q_producer_phase, 2),
                        [0],
                    )
                    T.buffer_store(
                        q_producer_index.buffer, _pipe_next_index(q_producer_index, 2), [0]
                    )
                    T.buffer_store(
                        k_producer_phase.buffer,
                        _pipe_next_phase(k_producer_index, k_producer_phase, 4),
                        [0],
                    )
                    T.buffer_store(
                        k_producer_index.buffer, _pipe_next_index(k_producer_index, 4), [0]
                    )
                    previous_v_index_tma = _builder_scalar(
                        "previous_v_index_tma", v_producer_index, dtype="int32"
                    )
                    previous_v_phase_tma = _builder_scalar(
                        "previous_v_phase_tma", v_producer_phase, dtype="int32"
                    )
                    T.buffer_store(
                        v_producer_index.buffer, _pipe_next_index(previous_v_index_tma, 3), [0]
                    )
                    T.buffer_store(
                        v_producer_phase.buffer,
                        _pipe_next_phase(previous_v_index_tma, previous_v_phase_tma, 3),
                        [0],
                    )
                last_chunk_tma = _builder_scalar(
                    "last_chunk_tma", num_valid_chunks_tma - 1, dtype="int32"
                )
                last_offset_tma = _builder_scalar(
                    "last_offset_tma",
                    batch_start_tma + T.cast(last_chunk_tma * CHUNK, "int64"),
                    dtype="int64",
                )
                _builder_emit(
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
                )
                T.buffer_store(
                    q_producer_phase.buffer,
                    _pipe_next_phase(q_producer_index, q_producer_phase, 2),
                    [0],
                )
                T.buffer_store(q_producer_index.buffer, _pipe_next_index(q_producer_index, 2), [0])
                T.buffer_store(
                    k_producer_phase.buffer,
                    _pipe_next_phase(k_producer_index, k_producer_phase, 4),
                    [0],
                )
                T.buffer_store(k_producer_index.buffer, _pipe_next_index(k_producer_index, 4), [0])
                previous_v_index_tma = _builder_scalar(
                    "previous_v_index_tma", v_producer_index, dtype="int32"
                )
                previous_v_phase_tma = _builder_scalar(
                    "previous_v_phase_tma", v_producer_phase, dtype="int32"
                )
                T.buffer_store(
                    v_producer_index.buffer, _pipe_next_index(previous_v_index_tma, 3), [0]
                )
                T.buffer_store(
                    v_producer_phase.buffer,
                    _pipe_next_phase(previous_v_index_tma, previous_v_phase_tma, 3),
                    [0],
                )
                with T.serial(num_valid_chunks_tma, padded_chunks_tma) as chunk_idx_tma:
                    chunk_offset_tma = _builder_scalar(
                        "chunk_offset_tma",
                        batch_start_tma + T.cast(chunk_idx_tma * CHUNK, "int64"),
                        dtype="int64",
                    )
                    _builder_emit(
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
                    )
                    T.buffer_store(
                        q_producer_phase.buffer,
                        _pipe_next_phase(q_producer_index, q_producer_phase, 2),
                        [0],
                    )
                    T.buffer_store(
                        q_producer_index.buffer, _pipe_next_index(q_producer_index, 2), [0]
                    )
                    T.buffer_store(
                        k_producer_phase.buffer,
                        _pipe_next_phase(k_producer_index, k_producer_phase, 4),
                        [0],
                    )
                    T.buffer_store(
                        k_producer_index.buffer, _pipe_next_index(k_producer_index, 4), [0]
                    )
                    previous_v_index_tma = _builder_scalar(
                        "previous_v_index_tma", v_producer_index, dtype="int32"
                    )
                    previous_v_phase_tma = _builder_scalar(
                        "previous_v_phase_tma", v_producer_phase, dtype="int32"
                    )
                    T.buffer_store(
                        v_producer_index.buffer, _pipe_next_index(previous_v_index_tma, 3), [0]
                    )
                    T.buffer_store(
                        v_producer_phase.buffer,
                        _pipe_next_phase(previous_v_index_tma, previous_v_phase_tma, 3),
                        [0],
                    )
                _builder_scope_exit(_builder_then_1128_12)
                _builder_scope_exit(_builder_if_1128_12)
                T.buffer_store(work_linear_tma.buffer, work_linear_tma + grid_x, [0])
            _builder_emit(
                _producer_tail_state(smem_addr_tma, 80, q_producer_index, q_producer_phase, 2)
            )
            _builder_emit(
                _producer_tail_state(smem_addr_tma, 32, k_producer_index, k_producer_phase, 4)
            )
            _builder_emit(
                _producer_tail_state(smem_addr_tma, 120, v_producer_index, v_producer_phase, 3)
            )
            _builder_scope_exit(_builder_then_1081_4)
            _builder_scope_exit(_builder_if_1081_4)
            _builder_scope_exit(_builder_else_937_4)
            _builder_scope_exit(_builder_if_937_4)
            _builder_scope_exit(_builder_else_820_4)
            _builder_scope_exit(_builder_if_820_4)
            _builder_scope_exit(_builder_else_465_4)
            _builder_scope_exit(_builder_if_465_4)
            _builder_if_1271_4 = _builder_scope_enter(T.If(warp == 11))
            _builder_then_1271_4 = _builder_scope_enter(T.Then())
            smem_addr_epi = _builder_scalar(
                "smem_addr_epi", T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0])), dtype="uint32"
            )
            _builder_emit(T.ptx.setmaxnreg.dec.sync.aligned.u32(24))
            gate_producer_index = _builder_scalar("gate_producer_index", 0, dtype="int32")
            gate_producer_phase = _builder_scalar("gate_producer_phase", 1, dtype="int32")
            beta_producer_index = _builder_scalar("beta_producer_index", 0, dtype="int32")
            beta_producer_phase = _builder_scalar("beta_producer_phase", 1, dtype="int32")
            o_consumer_index = _builder_scalar("o_consumer_index", 0, dtype="int32")
            o_consumer_phase = _builder_scalar("o_consumer_phase", 0, dtype="int32")
            work_linear_epi = _builder_scalar("work_linear_epi", block, dtype="int32")
            _builder_if_1283_8 = _builder_scope_enter(T.If(work_linear_epi < total_work))
            _builder_then_1283_8 = _builder_scope_enter(T.Then())
            _builder_if_1284_12 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
            _builder_then_1284_12 = _builder_scope_enter(T.Then())
            _builder_emit(_descriptor_copy_payload(o_map, descriptor_o))
            _builder_scope_exit(_builder_then_1284_12)
            _builder_scope_exit(_builder_if_1284_12)
            _builder_emit(T.cuda.warp_sync())
            _builder_emit(T.ptx.fence.acq_rel.cta())
            _builder_scope_exit(_builder_then_1283_8)
            _builder_scope_exit(_builder_if_1283_8)
            with T.While(work_linear_epi < total_work):
                work_u32_epi = _builder_scalar(
                    "work_u32_epi", T.cast(work_linear_epi, "uint32"), dtype="uint32"
                )
                remain_u32_epi = _builder_scalar(
                    "remain_u32_epi", _udiv_u32_const(work_u32_epi, HV), dtype="uint32"
                )
                head_epi = _builder_scalar(
                    "head_epi",
                    T.cast(work_u32_epi - remain_u32_epi * T.uint32(output_heads), "int32"),
                    dtype="int32",
                )
                qk_head_epi = _builder_scalar("qk_head_epi", head_epi, dtype="int32")
                value_subhead_epi = _builder_scalar("value_subhead_epi", 0, dtype="int32")
                if HQ != HV:
                    T.buffer_store(
                        qk_head_epi.buffer,
                        T.cast(_udiv_u32_const(T.cast(head_epi, "uint32"), HV // HQ), "int32"),
                        [0],
                    )
                    T.buffer_store(
                        value_subhead_epi.buffer, head_epi - qk_head_epi * (HV // HQ), [0]
                    )
                remain_epi = _builder_scalar(
                    "remain_epi", T.cast(remain_u32_epi, "int32"), dtype="int32"
                )
                batch_epi = _builder_scalar("batch_epi", remain_epi, dtype="int32")
                sequence_bounds_epi = _builder_name(
                    "sequence_bounds_epi", T.alloc_local((2,), "int32")
                )
                _builder_emit(_load_sequence_bounds(cu_seqlens, batch_epi, sequence_bounds_epi))
                batch_start_epi = _builder_scalar(
                    "batch_start_epi", T.cast(sequence_bounds_epi[0], "int64"), dtype="int64"
                )
                batch_end_epi = _builder_scalar(
                    "batch_end_epi", T.cast(sequence_bounds_epi[1], "int64"), dtype="int64"
                )
                seqlen_epi = _builder_scalar(
                    "seqlen_epi", T.cast(batch_end_epi - batch_start_epi, "int32"), dtype="int32"
                )
                num_valid_chunks_epi = _builder_scalar(
                    "num_valid_chunks_epi", (seqlen_epi + CHUNK - 1) // CHUNK, dtype="int32"
                )
                num_pairs_epi = _builder_scalar(
                    "num_pairs_epi", (seqlen_epi + PAIR_TOKENS - 1) // PAIR_TOKENS, dtype="int32"
                )
                padded_chunks_epi = _builder_scalar(
                    "padded_chunks_epi", num_pairs_epi * 2, dtype="int32"
                )
                _builder_if_1311_12 = _builder_scope_enter(T.If(padded_chunks_epi > 0))
                _builder_then_1311_12 = _builder_scope_enter(T.Then())
                _builder_if_1312_16 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                _builder_then_1312_16 = _builder_scope_enter(T.Then())
                _builder_emit(T.ptx.cp.async_.bulk.wait_group.read(0))
                _builder_scope_exit(_builder_then_1312_16)
                _builder_scope_exit(_builder_if_1312_16)
                _builder_emit(T.cuda.warp_sync())
                _builder_if_1315_16 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                _builder_then_1315_16 = _builder_scope_enter(T.Then())
                if HQ == HV:
                    _builder_emit(
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
                    )
                else:
                    _builder_emit(
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
                    )
                _builder_scope_exit(_builder_then_1315_16)
                _builder_scope_exit(_builder_if_1315_16)
                _builder_emit(T.cuda.warp_sync())
                _builder_emit(_tensormap_release())
                _builder_if_1340_16 = _builder_scope_enter(T.If(T.cuda.elect_sync()))
                _builder_then_1340_16 = _builder_scope_enter(T.Then())
                _builder_emit(_tensormap_acquire(descriptor_o))
                _builder_scope_exit(_builder_then_1340_16)
                _builder_scope_exit(_builder_if_1340_16)
                with T.unroll(0, 2) as prefetch_idx_epi:
                    prefetch_offset_epi = _builder_scalar(
                        "prefetch_offset_epi",
                        batch_start_epi + T.cast(prefetch_idx_epi * CHUNK, "int64"),
                        dtype="int64",
                    )
                    _builder_emit(
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
                    )
                    T.buffer_store(
                        gate_producer_phase.buffer,
                        _pipe_next_phase(gate_producer_index, gate_producer_phase, 5),
                        [0],
                    )
                    T.buffer_store(
                        gate_producer_index.buffer, _pipe_next_index(gate_producer_index, 5), [0]
                    )
                    T.buffer_store(
                        beta_producer_phase.buffer,
                        _pipe_next_phase(beta_producer_index, beta_producer_phase, 5),
                        [0],
                    )
                    T.buffer_store(
                        beta_producer_index.buffer, _pipe_next_index(beta_producer_index, 5), [0]
                    )
                _builder_if_1377_16 = _builder_scope_enter(T.If(padded_chunks_epi > 2))
                _builder_then_1377_16 = _builder_scope_enter(T.Then())
                with T.unroll(2, 4) as prefetch_idx_epi:
                    prefetch_offset_epi = _builder_scalar(
                        "prefetch_offset_epi",
                        batch_start_epi + T.cast(prefetch_idx_epi * CHUNK, "int64"),
                        dtype="int64",
                    )
                    _builder_emit(
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
                    )
                    T.buffer_store(
                        gate_producer_phase.buffer,
                        _pipe_next_phase(gate_producer_index, gate_producer_phase, 5),
                        [0],
                    )
                    T.buffer_store(
                        gate_producer_index.buffer, _pipe_next_index(gate_producer_index, 5), [0]
                    )
                    T.buffer_store(
                        beta_producer_phase.buffer,
                        _pipe_next_phase(beta_producer_index, beta_producer_phase, 5),
                        [0],
                    )
                    T.buffer_store(
                        beta_producer_index.buffer, _pipe_next_index(beta_producer_index, 5), [0]
                    )
                _builder_scope_exit(_builder_then_1377_16)
                _builder_scope_exit(_builder_if_1377_16)
                with T.serial(0, padded_chunks_epi) as chunk_idx_epi:
                    chunk_offset_epi = _builder_scalar(
                        "chunk_offset_epi",
                        batch_start_epi + T.cast(chunk_idx_epi * CHUNK, "int64"),
                        dtype="int64",
                    )
                    prefetch_idx_epi = _builder_scalar(
                        "prefetch_idx_epi", chunk_idx_epi + 4, dtype="int32"
                    )
                    _builder_if_1415_20 = _builder_scope_enter(
                        T.If(prefetch_idx_epi < padded_chunks_epi)
                    )
                    _builder_then_1415_20 = _builder_scope_enter(T.Then())
                    prefetch_offset_epi = _builder_scalar(
                        "prefetch_offset_epi", chunk_offset_epi + 4 * CHUNK, dtype="int64"
                    )
                    _builder_emit(
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
                    )
                    T.buffer_store(
                        gate_producer_phase.buffer,
                        _pipe_next_phase(gate_producer_index, gate_producer_phase, 5),
                        [0],
                    )
                    T.buffer_store(
                        gate_producer_index.buffer, _pipe_next_index(gate_producer_index, 5), [0]
                    )
                    T.buffer_store(
                        beta_producer_phase.buffer,
                        _pipe_next_phase(beta_producer_index, beta_producer_phase, 5),
                        [0],
                    )
                    T.buffer_store(
                        beta_producer_index.buffer, _pipe_next_index(beta_producer_index, 5), [0]
                    )
                    _builder_scope_exit(_builder_then_1415_20)
                    _builder_scope_exit(_builder_if_1415_20)
                    _builder_emit(
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
                    )
                    previous_o_index_epi = _builder_scalar(
                        "previous_o_index_epi", o_consumer_index, dtype="int32"
                    )
                    previous_o_phase_epi = _builder_scalar(
                        "previous_o_phase_epi", o_consumer_phase, dtype="int32"
                    )
                    T.buffer_store(
                        o_consumer_index.buffer, _pipe_next_index(previous_o_index_epi, 2), [0]
                    )
                    T.buffer_store(
                        o_consumer_phase.buffer,
                        _pipe_next_phase(previous_o_index_epi, previous_o_phase_epi, 2),
                        [0],
                    )
                _builder_scope_exit(_builder_then_1311_12)
                _builder_scope_exit(_builder_if_1311_12)
                T.buffer_store(work_linear_epi.buffer, work_linear_epi + grid_x, [0])
            _builder_emit(
                _producer_tail_state(
                    smem_addr_epi, 184, gate_producer_index, gate_producer_phase, 5
                )
            )
            _builder_emit(
                _producer_tail_state(
                    smem_addr_epi, 264, beta_producer_index, beta_producer_phase, 5
                )
            )
            _builder_scope_exit(_builder_then_1271_4)
            _builder_scope_exit(_builder_if_1271_4)
    return builder.get()


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
def _load_oracle():
    from flashinfer.gdn_kernels.blackwell.gdn_prefill import chunk_gated_delta_rule_sm100

    return chunk_gated_delta_rule_sm100


def _run_oracle(case: dict[str, Any], output: torch.Tensor, final_state: torch.Tensor) -> None:
    oracle = _load_oracle()
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
    return _build_kernel(HQ=cfg.hq, HV=cfg.hv).with_attr(
        "tirx.kernel_launch_params", list(LAUNCH_TAGS)
    )


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


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
    _run_oracle(case, reference_o, reference_state)
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
    rounds = kwargs.pop("rounds", 5)
    cooldown_s = kwargs.pop("cooldown_s", 1.0)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for GDN prefill SM100 benchmark")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        raise SkipTest(f"GDN prefill SM100 requires compute capability 10.x, got {capability}")

    from tirx_kernels.runner import bench

    case = prepare_data(**kwargs)
    args = _tirx_args(case)

    def _flashinfer_cutedsl_builder():
        reference_o = torch.empty_like(case["o"])
        reference_state = torch.empty_like(case["final_state"])

        # First execution performs CuTeDSL JIT and initializes its persistent
        # workspace; subsequent executions are launch-only warmups.
        _run_oracle(case, reference_o, reference_state)
        for _ in range(2):
            _run_oracle(case, reference_o, reference_state)
        torch.cuda.synchronize()

        def launch():
            _run_oracle(case, reference_o, reference_state)

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


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    config = dict(kwargs)
    protocol = {name: config.pop(name) for name in ("rounds", "cooldown_s") if name in config}
    prepared = prepare_bench(**config)
    return prepared.run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)


__all__ = ["CONFIGS", "KERNEL_META", "get_kernel", "prepare_data", "run_bench", "run_test"]
