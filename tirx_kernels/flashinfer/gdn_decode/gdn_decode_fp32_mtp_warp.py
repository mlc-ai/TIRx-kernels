# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Integration scaffold for FlashInfer's SM100 FP32-state MTP warp kernel.

Upstream source: flashinfer/gdn_kernels/gdn_decode_mtp.py.
"""

import functools
import os
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as TK
from tirx_kernels.runner import bench

KERNEL_META = {
    "name": "gdn_decode_fp32_mtp_warp",
    "category": "flashinfer",
    "compute_capability": 10,
}

K = 128
V = 128
THREADS = 128
VEC_SIZE = 4
SCALE = K**-0.5
NUM_WARPS = 4
NUM_GROUPS = 4
LANES_PER_GROUP = 32
SOURCE_NUM_SMS = 148
LOG2_E = 1.4426950408889634
LN_2 = 0.6931471805599453
_HAS_NATIVE_PTX_ADDR = hasattr(TK.ptx, "addr")


def _shfl_bfly_f32(value, lane_xor):
    """``shfl.sync.bfly.b32`` at width 32: clamp/segmask 31, full member mask.

    DPS: the destination pins the warp collective to the call site, so the
    shuffle is emitted once here rather than re-emitted at every textual use
    of the returned value.
    """
    shfl_bfly = TK.local_scalar("uint32")
    TK.ptx.shfl_sync.bfly.b32(
        shfl_bfly,
        TK.reinterpret("uint32", value),
        TK.cast(lane_xor, "uint32"),
        TK.uint32(31),
        TK.uint32(4294967295),
    )
    return TK.reinterpret("float32", shfl_bfly)


def _local_scalar(dtype: str, value):
    out = TK.alloc_local((1,), dtype)
    TK.assign(out[0], value)
    return out


def _global_load_u16_ptr_offset(ptr, byte_offset: int, native_offset: bool = True):
    out = TK.local_scalar("uint16")
    if not native_offset:
        TK.ptx.ld.global_.b16(out, TK.ptr_byte_offset(ptr, byte_offset, "bfloat16"))
        return out
    TK.ptx.ld.global_.b16(out, TK.ptx.addr(ptr, byte_offset))
    return out


def _shared_store_f32_ptr_offset(ptr, byte_offset: int, value, native_offset: bool = True):
    if not native_offset:
        TK.ptx.st.shared.b32(
            TK.ptr_byte_offset(ptr, byte_offset, "float32"), TK.reinterpret("uint32", value)
        )
        return
    TK.ptx.st.shared.b32(TK.ptx.addr(ptr, byte_offset), TK.reinterpret("uint32", value))


def _shared_load_f32_ptr_offset(ptr, byte_offset: int, native_offset: bool = True):
    word = TK.local_scalar("uint32")
    if not native_offset:
        TK.ptx.ld.shared.b32(word, TK.ptr_byte_offset(ptr, byte_offset, "float32"))
        return TK.reinterpret("float32", word)
    TK.ptx.ld.shared.b32(word, TK.ptx.addr(ptr, byte_offset))
    return TK.reinterpret("float32", word)


def _shared_store_u16_ptr_offset(ptr, byte_offset: int, value, native_offset: bool = True):
    if not native_offset:
        TK.ptx.st.shared.b16(TK.ptr_byte_offset(ptr, byte_offset, "bfloat16"), value)
        return
    TK.ptx.st.shared.b16(TK.ptx.addr(ptr, byte_offset), value)


def _shared_load_u16_ptr_offset(ptr, byte_offset: int, native_offset: bool = True):
    out = TK.local_scalar("uint16")
    if not native_offset:
        TK.ptx.ld.shared.b16(out, TK.ptr_byte_offset(ptr, byte_offset, "bfloat16"))
        return out
    TK.ptx.ld.shared.b16(out, TK.ptx.addr(ptr, byte_offset))
    return out


def _shared_load_f32x4_ptr(ptr, values):
    words = TK.alloc_local((4,), "uint32", align=16)
    TK.ptx.ld.shared.v4.b32(words[0], words[1], words[2], words[3], ptr)
    with TK.unroll(4) as i:
        TK.ptx.mov.b32(values[i], TK.reinterpret("float32", words[i]))


def _shared_load_f32x4_ptr_offset(ptr, byte_offset: TK.int32, values, native_offset: bool = True):
    with TK.If(TK.Not(native_offset)):
        with TK.Then():
            _shared_load_f32x4_ptr(TK.ptr_byte_offset(ptr, byte_offset, "float32"), values)
        with TK.Else():
            words = TK.alloc_local((4,), "uint32", align=16)
            TK.ptx.ld.shared.v4.b32(
                words[0], words[1], words[2], words[3], TK.ptx.addr(ptr, byte_offset)
            )
            with TK.unroll(4) as i:
                TK.ptx.mov.b32(values[i], TK.reinterpret("float32", words[i]))


def _shared_load_f32x4_b64_ptr(ptr, values):
    pairs = TK.alloc_local((2,), "uint64", align=16)
    TK.ptx.ld.shared.v2.b64(pairs[0], pairs[1], ptr)
    TK.ptx.mov.b32(values[0], TK.cuda.float2_x(pairs[0]))
    TK.ptx.mov.b32(values[1], TK.cuda.float2_y(pairs[0]))
    TK.ptx.mov.b32(values[2], TK.cuda.float2_x(pairs[1]))
    TK.ptx.mov.b32(values[3], TK.cuda.float2_y(pairs[1]))


def _shared_load_f32x4_b64_ptr_offset(
    ptr, byte_offset: TK.int32, values, native_offset: bool = True
):
    with TK.If(TK.Not(native_offset)):
        with TK.Then():
            _shared_load_f32x4_b64_ptr(TK.ptr_byte_offset(ptr, byte_offset, "float32"), values)
        with TK.Else():
            pairs = TK.alloc_local((2,), "uint64", align=16)
            TK.ptx.ld.shared.v2.b64(pairs[0], pairs[1], TK.ptx.addr(ptr, byte_offset))
            TK.ptx.mov.b32(values[0], TK.cuda.float2_x(pairs[0]))
            TK.ptx.mov.b32(values[1], TK.cuda.float2_y(pairs[0]))
            TK.ptx.mov.b32(values[2], TK.cuda.float2_x(pairs[1]))
            TK.ptx.mov.b32(values[3], TK.cuda.float2_y(pairs[1]))


def _global_load_f32x4_ptr(ptr, values, value_offset: TK.int32):
    words = TK.alloc_local((4,), "uint32", align=16)
    TK.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], ptr)
    with TK.unroll(4) as i:
        TK.ptx.mov.b32(values[value_offset + i], TK.reinterpret("float32", words[i]))


def _global_load_f32x4_ptr_offset(
    ptr, byte_offset: TK.int32, values, value_offset: TK.int32, native_offset: bool = True
):
    with TK.If(TK.Not(native_offset)):
        with TK.Then():
            _global_load_f32x4_ptr(
                TK.ptr_byte_offset(ptr, byte_offset, "float32"), values, value_offset
            )
        with TK.Else():
            words = TK.alloc_local((4,), "uint32", align=16)
            TK.ptx.ld.global_.v4.b32(
                words[0], words[1], words[2], words[3], TK.ptx.addr(ptr, byte_offset)
            )
            with TK.unroll(4) as i:
                TK.ptx.mov.b32(values[value_offset + i], TK.reinterpret("float32", words[i]))


def _global_load_f32x4_b64_ptr(ptr, values, value_offset: TK.int32):
    pairs = TK.alloc_local((2,), "uint64", align=16)
    TK.ptx.ld.global_.v2.b64(pairs[0], pairs[1], ptr)
    TK.ptx.mov.b32(values[value_offset], TK.cuda.float2_x(pairs[0]))
    TK.ptx.mov.b32(values[value_offset + 1], TK.cuda.float2_y(pairs[0]))
    TK.ptx.mov.b32(values[value_offset + 2], TK.cuda.float2_x(pairs[1]))
    TK.ptx.mov.b32(values[value_offset + 3], TK.cuda.float2_y(pairs[1]))


def _global_load_f32x4_b64_ptr_offset(
    ptr, byte_offset: TK.int32, values, value_offset: TK.int32, native_offset: bool = True
):
    with TK.If(TK.Not(native_offset)):
        with TK.Then():
            _global_load_f32x4_b64_ptr(
                TK.ptr_byte_offset(ptr, byte_offset, "float32"), values, value_offset
            )
        with TK.Else():
            pairs = TK.alloc_local((2,), "uint64", align=16)
            TK.ptx.ld.global_.v2.b64(pairs[0], pairs[1], TK.ptx.addr(ptr, byte_offset))
            TK.ptx.mov.b32(values[value_offset], TK.cuda.float2_x(pairs[0]))
            TK.ptx.mov.b32(values[value_offset + 1], TK.cuda.float2_y(pairs[0]))
            TK.ptx.mov.b32(values[value_offset + 2], TK.cuda.float2_x(pairs[1]))
            TK.ptx.mov.b32(values[value_offset + 3], TK.cuda.float2_y(pairs[1]))


def _global_store_f32x4_ptr_offset(
    ptr, byte_offset: TK.int32, values, value_offset: TK.int32, native_offset: bool = True
):
    with TK.If(TK.Not(native_offset)):
        with TK.Then():
            TK.ptx.st.global_.v4.b32(
                TK.ptr_byte_offset(ptr, byte_offset, "float32"),
                TK.reinterpret("uint32", values[value_offset]),
                TK.reinterpret("uint32", values[value_offset + 1]),
                TK.reinterpret("uint32", values[value_offset + 2]),
                TK.reinterpret("uint32", values[value_offset + 3]),
            )
        with TK.Else():
            TK.ptx.st.global_.v4.b32(
                TK.ptx.addr(ptr, byte_offset),
                TK.reinterpret("uint32", values[value_offset]),
                TK.reinterpret("uint32", values[value_offset + 1]),
                TK.reinterpret("uint32", values[value_offset + 2]),
                TK.reinterpret("uint32", values[value_offset + 3]),
            )


def _global_store_f32x4_b64_ptr(ptr, values, value_offset: TK.int32):
    TK.ptx.st.global_.v2.b64(
        ptr,
        TK.cuda.make_float2(values[value_offset], values[value_offset + 1]),
        TK.cuda.make_float2(values[value_offset + 2], values[value_offset + 3]),
    )


def _global_store_f32x4_b64_ptr_offset(
    ptr, byte_offset: TK.int32, values, value_offset: TK.int32, native_offset: bool = True
):
    with TK.If(TK.Not(native_offset)):
        with TK.Then():
            _global_store_f32x4_b64_ptr(
                TK.ptr_byte_offset(ptr, byte_offset, "float32"), values, value_offset
            )
        with TK.Else():
            TK.ptx.st.global_.v2.b64(
                TK.ptx.addr(ptr, byte_offset),
                TK.cuda.make_float2(values[value_offset], values[value_offset + 1]),
                TK.cuda.make_float2(values[value_offset + 2], values[value_offset + 3]),
            )


def _global_store_u16_ptr_offset(ptr, byte_offset: int, value, native_offset: bool = True):
    if not native_offset:
        TK.ptx.st.global_.b16(TK.ptr_byte_offset(ptr, byte_offset, "bfloat16"), value)
        return
    TK.ptx.st.global_.b16(TK.ptx.addr(ptr, byte_offset), value)


def _global_store_u32_ptr_offset(ptr, byte_offset: int, value, native_offset: bool = True):
    if not native_offset:
        TK.ptx.st.global_.b32(TK.ptr_byte_offset(ptr, byte_offset, "bfloat16"), value)
        return
    TK.ptx.st.global_.b32(TK.ptx.addr(ptr, byte_offset), value)


def _shared_store_u32_ptr_offset(ptr, byte_offset: int, value, native_offset: bool = True):
    if not native_offset:
        TK.ptx.st.shared.b32(TK.ptr_byte_offset(ptr, byte_offset, "bfloat16"), value)
        return
    TK.ptx.st.shared.b32(TK.ptx.addr(ptr, byte_offset), value)


def _packed_fma_store(out, lhs0, lhs1, rhs0, rhs1, acc0, acc1):
    TK.ptx.fma.rn.f32x2(
        out[0],
        TK.cuda.make_float2(lhs0, lhs1),
        TK.cuda.make_float2(rhs0, rhs1),
        TK.cuda.make_float2(acc0, acc1),
    )


def _gate_pair_store(out, scratch, a_bits, b_bits, exp_A_value, dt_value):
    TK.ptx.cvt.f32.bf16(scratch[0], TK.cast(b_bits, "uint16"))
    TK.ptx.add.rn.f32.bf16(scratch[1], TK.cast(a_bits, "uint16"), dt_value)
    TK.ptx.mul.f32(scratch[2], scratch[1], TK.float32(LOG2_E))
    TK.ptx.ex2.approx.ftz.f32(scratch[2], scratch[2])
    TK.ptx.add.f32(scratch[2], TK.float32(1.0), scratch[2])
    TK.ptx.lg2.approx.ftz.f32(scratch[2], scratch[2])
    TK.ptx.mul.f32(scratch[2], scratch[2], TK.float32(LN_2))
    TK.ptx.mov.b32(
        scratch[3],
        TK.if_then_else(scratch[1] <= TK.float32(20.0), TK.float32(1.0), TK.float32(0.0)),
    )
    TK.ptx.sub.f32(scratch[4], TK.float32(1.0), scratch[3])
    TK.ptx.mul.f32(scratch[4], scratch[1], scratch[4])
    TK.ptx.fma.rn.f32(scratch[2], scratch[2], scratch[3], scratch[4])
    TK.ptx.mul.f32(scratch[4], exp_A_value, scratch[2])
    TK.ptx.mul.f32(scratch[0], scratch[0], TK.float32(-LOG2_E))
    TK.ptx.ex2.approx.ftz.f32(scratch[0], scratch[0])
    TK.ptx.add.f32(scratch[0], TK.float32(1.0), scratch[0])
    TK.ptx.rcp.rn.f32(scratch[0], scratch[0])
    TK.ptx.mul.f32(scratch[4], scratch[4], TK.float32(-LOG2_E))
    TK.ptx.ex2.approx.ftz.f32(scratch[4], scratch[4])
    TK.assign(out[0], TK.cuda.make_float2(scratch[4], scratch[0]))


def _make_warp_uniform(value):
    """Broadcast lane 0's ``value`` to the warp -- ``shfl.sync.idx.b32``.

    Width 32, so the clamp/segmask operand is 31 and the member mask is full.
    DPS: the destination pins the warp collective to the call site.
    """
    uniform = TK.local_scalar("uint32")
    TK.ptx.shfl_sync.idx.b32(
        uniform, TK.cast(value, "uint32"), TK.uint32(0), TK.uint32(31), TK.uint32(0xFFFFFFFF)
    )
    return TK.cast(uniform, "int32")


def _source_config(
    batch: int, seq_len: int, num_v_heads: int, *, disable_state_update: bool
) -> tuple[int, int, bool]:
    """Mirror the frozen source's reachable warp-kernel picker."""
    work_units = batch * num_v_heads
    if work_units <= 64:
        tile_v, ilp_rows, use_smem_v = 8, 2, False
    elif work_units <= 128:
        tile_v, ilp_rows, use_smem_v = 16, 4, False
    elif work_units <= 448:
        if seq_len <= 2:
            tile_v, ilp_rows, use_smem_v = 16, 2, False
        else:
            tile_v, ilp_rows, use_smem_v = 32, 4, False
    elif work_units <= 1024:
        tile_v, ilp_rows, use_smem_v = 32, 4, False
    else:
        tile_v, ilp_rows, use_smem_v = 64, 4, True
        if not disable_state_update and seq_len <= 2:
            ilp_rows = 8
            use_smem_v = False
    return min(tile_v, V), ilp_rows, use_smem_v


def _target_config(
    batch: int, seq_len: int, num_heads: int, num_v_heads: int, *, disable_state_update: bool
) -> tuple[int, int, bool]:
    tile_v, ilp_rows, use_smem_v = _source_config(
        batch, seq_len, num_v_heads, disable_state_update=disable_state_update
    )
    if 128 < batch * num_v_heads <= 448 and seq_len == 2:
        return 32, 4, False
    if num_heads == 16 and 448 < batch * num_v_heads <= 1024 and 3 <= seq_len <= 7:
        return 32, 4, True
    return tile_v, ilp_rows, use_smem_v


def _case(
    label: str,
    *,
    batch: int = 4,
    seq_len: int = 4,
    num_heads: int = 16,
    num_v_heads: int = 64,
    use_qk_l2norm: bool = True,
    disable_state_update: bool = False,
    cache_intermediate_states: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    tile_v, ilp_rows, use_smem_v = _source_config(
        batch, seq_len, num_v_heads, disable_state_update=disable_state_update
    )
    return {
        "label": label,
        "batch": batch,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "num_v_heads": num_v_heads,
        "tile_v": tile_v,
        "ilp_rows": ilp_rows,
        "use_smem_v": use_smem_v,
        "use_qk_l2norm": use_qk_l2norm,
        "disable_state_update": disable_state_update,
        "cache_intermediate_states": cache_intermediate_states,
        **kwargs,
    }


CONFIGS = [
    _case("t2_wu256_ilp2", seq_len=2),
    _case("t3_wu256_ilp4", seq_len=3),
    _case("t4_wu512", batch=8, seq_len=4),
    _case("t5_wu1024", batch=16, seq_len=5),
    _case("t6_wu256", seq_len=6),
    _case("t7_wu256", seq_len=7),
    _case("t8_wu256", seq_len=8),
    _case("t2_wu2048_ilp8", batch=32, seq_len=2),
    _case("t4_wu2048_smem_v", batch=32, seq_len=4),
    _case("t8_wu2048_smem_v", batch=32, seq_len=8),
    _case("t4_l2off", use_qk_l2norm=False),
    _case("t4_disable_update", disable_state_update=True),
    _case("t4_cache_update", cache_intermediate_states=True),
    _case("t4_cache_disable_update", disable_state_update=True, cache_intermediate_states=True),
    _case("t4_split_pool", same_pool=False),
    _case("t4_negative_read", negative_read_index=True),
    _case("t4_negative_write", same_pool=False, negative_write_index=True),
    _case("t4_padded_pool", padded_pool=True),
    _case("t4_padded_split", padded_pool=True, same_pool=False),
    _case("t4_packed_qkv", packed_qkv=True),
    _case("t4_scatter_flat", per_token_pool_scatter=True),
    _case("t4_scatter_padded", padded_pool=True, per_token_pool_scatter=True),
    _case(
        "t8_scatter_i64_stress",
        batch=128,
        seq_len=8,
        num_heads=16,
        num_v_heads=64,
        per_token_pool_scatter=True,
    ),
    _case("t4_tp2", batch=8, num_heads=8, num_v_heads=32),
    _case("t4_tp4", batch=16, num_heads=4, num_v_heads=16),
    _case("t4_tp8", batch=32, num_heads=2, num_v_heads=8),
]


def _bench_case(seq_len: int, batch: int, num_heads: int, num_v_heads: int) -> dict[str, Any]:
    tile_v, ilp_rows, use_smem_v = _source_config(
        batch, seq_len, num_v_heads, disable_state_update=False
    )
    label = (
        f"t{seq_len}_b{batch}_h{num_heads}_hv{num_v_heads}_"
        f"tv{tile_v}_ilp{ilp_rows}_sv{int(use_smem_v)}"
    )
    return _case(
        label,
        batch=batch,
        seq_len=seq_len,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        cache_intermediate_states=True,
    )


_TP1_BATCHES = (4, 8, 16, 32, 64, 128, 256)
_TP_BOUNDARY_WORK_UNITS = (256, 512, 1024, 2048)
_TP_BOUNDARY_SEQ_LENS = (2, 3, 4, 8)
_TP_HEAD_CONFIGS = ((8, 32), (4, 16), (2, 8))

BENCH_CONFIGS = [
    _bench_case(seq_len, batch, 16, 64) for seq_len in range(2, 9) for batch in _TP1_BATCHES
] + [
    _bench_case(seq_len, work_units // num_v_heads, num_heads, num_v_heads)
    for num_heads, num_v_heads in _TP_HEAD_CONFIGS
    for work_units in _TP_BOUNDARY_WORK_UNITS
    for seq_len in _TP_BOUNDARY_SEQ_LENS
]

assert len(BENCH_CONFIGS) == 97
assert len({config["label"] for config in BENCH_CONFIGS}) == len(BENCH_CONFIGS)


def _require_supported_config(config: dict[str, Any]) -> None:
    batch = int(config["batch"])
    seq_len = int(config["seq_len"])
    num_heads = int(config["num_heads"])
    num_v_heads = int(config["num_v_heads"])
    disable_state_update = bool(config.get("disable_state_update", False))
    if batch < 1:
        raise ValueError("batch must be positive")
    if not 2 <= seq_len <= 8:
        raise ValueError("FP32 MTP warp port requires seq_len in [2, 8]")
    if batch * num_v_heads <= 128:
        raise ValueError("FP32 MTP warp port requires batch * num_v_heads > 128")
    if (num_heads, num_v_heads) not in ((16, 64), (8, 32), (4, 16), (2, 8)):
        raise ValueError("unsupported Qwen3.5 TP head configuration")
    if num_v_heads % num_heads:
        raise ValueError("num_v_heads must be divisible by num_heads")
    expected = _source_config(
        batch, seq_len, num_v_heads, disable_state_update=disable_state_update
    )
    actual = (int(config["tile_v"]), int(config["ilp_rows"]), bool(config["use_smem_v"]))
    if actual != expected:
        raise ValueError(f"config does not match frozen source picker: {actual} != {expected}")
    scatter = bool(config.get("per_token_pool_scatter", False))
    cache = bool(config.get("cache_intermediate_states", False))
    if scatter and (cache or disable_state_update):
        raise ValueError("per-token scatter requires cache off and state update on")


def _make_gdn_decode_fp32_mtp_warp(
    *,
    SEQ_LEN,
    NUM_HEADS,
    NUM_V_HEADS,
    TILE_V,
    NUM_V_TILES,
    ILP_ROWS,
    USE_SMEM_V,
    USE_QK_L2NORM,
    USE_NATIVE_OFFSETS,
    RELOAD_K_FOR_OUTPUT,
    USE_CANONICAL_WARP_ID,
    USE_PACKED_OUTPUT,
    DISABLE_STATE_UPDATE,
    CACHE_INTERMEDIATE_STATES,
    SAME_POOL,
    PER_TOKEN_POOL_SCATTER,
    PER_TOKEN_POOL_SCATTER_FLAT,
    PADDED_POOL,
    PACKED_QKV,
    POOL_FACTOR,
    INTERMEDIATE_BATCH_STRIDE,
    INTERMEDIATE_DUMMY_ELEMENTS,
    SSM_BATCH_STRIDE,
    SSM_DUMMY_ELEMENTS,
    SHARED_BYTES,
    S_K_BYTE_OFFSET,
    S_G_BYTE_OFFSET,
    S_BETA_BYTE_OFFSET,
    S_V_BYTE_OFFSET,
    S_OUTPUT_BYTE_OFFSET,
    ROWS_PER_GROUP,
    ITERS_PER_GROUP,
    PREFETCH_ROWS,
):
    @TK.kernel(
        warps=NUM_WARPS, arch="sm_100a", grid=lambda p: p["batch"] * NUM_V_HEADS * NUM_V_TILES
    )
    def gdn_decode_fp32_mtp_warp(
        state: TK.gptr[TK.f32],
        intermediate: TK.gptr[TK.f32],
        A_log: TK.gptr[TK.f32],
        a: TK.gptr[TK.bf16],
        dt_bias: TK.gptr[TK.f32],
        q: TK.gptr[TK.bf16],
        k: TK.gptr[TK.bf16],
        v: TK.gptr[TK.bf16],
        b_gate: TK.gptr[TK.bf16],
        output: TK.gptr[TK.bf16],
        read_indices: TK.gptr[TK.i32],
        write_indices: TK.gptr[TK.i32],
        ssm_state_indices: TK.gptr[TK.i32],
        state_slot_stride: TK.i64,
        state_head_stride: TK.i64,
        q_batch_stride: TK.i64,
        k_batch_stride: TK.i64,
        v_batch_stride: TK.i64,
        batch: TK.i32,
    ):
        smem = TK.smem_pool()
        s_q = smem.alloc((S_K_BYTE_OFFSET // 4,), TK.f32, align=16)
        s_k = smem.alloc(((S_G_BYTE_OFFSET - S_K_BYTE_OFFSET) // 4,), TK.f32, align=16)
        s_g = smem.alloc(((S_BETA_BYTE_OFFSET - S_G_BYTE_OFFSET) // 4,), TK.f32, align=16)
        s_beta = smem.alloc(((S_V_BYTE_OFFSET - S_BETA_BYTE_OFFSET) // 4,), TK.f32, align=16)
        s_v = smem.alloc(((S_OUTPUT_BYTE_OFFSET - S_V_BYTE_OFFSET) // 4,), TK.f32, align=16)
        s_output = smem.alloc((SEQ_LEN * TILE_V,), TK.bf16, align=16)
        smem.commit(SHARED_BYTES)
        linear_cta = TK.cta_id()
        tid = TK.thread_id()
        canonical_warp = TK.warp_id()
        canonical_lane = TK.lane_id()
        roles = TK.specialize(chain_dispatch=False)
        producer = roles.role("producer", warps=[0])
        workers = roles.role("workers", warps=range(1, NUM_WARPS))
        warp = TK.local_scalar("int32")
        lane = TK.local_scalar("int32")
        if USE_CANONICAL_WARP_ID:
            TK.assign(warp, canonical_warp)
            TK.assign(lane, canonical_lane)
        else:
            warp_raw = _local_scalar("int32", tid // LANES_PER_GROUP)
            TK.assign(warp, _make_warp_uniform(warp_raw[0]))
            TK.assign(lane, tid % LANES_PER_GROUP)
        k_start = _local_scalar("int32", lane * VEC_SIZE)
        v_tile = linear_cta % NUM_V_TILES
        cta_head = linear_cta // NUM_V_TILES
        hv = cta_head % NUM_V_HEADS
        n = cta_head // NUM_V_HEADS
        h = hv // (NUM_V_HEADS // NUM_HEADS)
        effective_state_slot_stride = TK.if_then_else(
            PADDED_POOL, state_slot_stride, TK.int64(NUM_V_HEADS * V * K)
        )
        effective_state_head_stride = TK.if_then_else(
            PADDED_POOL, state_head_stride, TK.int64(V * K)
        )
        effective_q_batch_stride = TK.if_then_else(
            PACKED_QKV, q_batch_stride, TK.int64(SEQ_LEN * NUM_HEADS * K)
        )
        effective_k_batch_stride = TK.if_then_else(
            PACKED_QKV, k_batch_stride, TK.int64(SEQ_LEN * NUM_HEADS * K)
        )
        effective_v_batch_stride = TK.if_then_else(
            PACKED_QKV, v_batch_stride, TK.int64(SEQ_LEN * NUM_V_HEADS * V)
        )
        read_slot_raw = TK.local_scalar("int32")
        TK.ptx.ld.global_.s32(read_slot_raw, read_indices.ptr_to([n]))
        A_value = TK.local_scalar("float32")
        TK.ptx.ld.global_.b32(A_value, A_log.ptr_to([hv]))
        dt_value = TK.local_scalar("float32")
        TK.ptx.ld.global_.b32(dt_value, dt_bias.ptr_to([hv]))
        r_h = TK.alloc_local((ILP_ROWS * VEC_SIZE,), "float32", align=16)
        r_q = TK.alloc_local((VEC_SIZE,), "float32", align=16)
        r_k = TK.alloc_local((VEC_SIZE,), "float32", align=16)
        r_k_output = TK.alloc_local((VEC_SIZE,), "float32", align=16)
        r_q_bits = TK.alloc_local((VEC_SIZE,), "uint16")
        r_k_bits = TK.alloc_local((VEC_SIZE,), "uint16")
        gate_scratch = TK.alloc_local((5,), "float32")
        gate_pair_value = TK.alloc_local((1,), "uint64")
        with TK.If(read_slot_raw >= 0), TK.Then():
            write_slot_raw = _local_scalar("int32", read_slot_raw)
            if not SAME_POOL:
                TK.ptx.ld.global_.s32(write_slot_raw[0], write_indices.ptr_to([n]))
            write_slot = _local_scalar(
                "int32", TK.if_then_else(write_slot_raw[0] < 0, read_slot_raw, write_slot_raw[0])
            )
            read_state_base = (
                TK.cast(read_slot_raw, "int64") * effective_state_slot_stride
                + TK.cast(hv, "int64") * effective_state_head_stride
            )
            write_state_base = _local_scalar("int64", read_state_base)
            if not SAME_POOL:
                TK.assign(
                    write_state_base[0],
                    TK.cast(write_slot[0], "int64") * effective_state_slot_stride
                    + TK.cast(hv, "int64") * effective_state_head_stride,
                )
            with producer:
                _mul = TK.local_scalar("float32")
                TK.ptx["mul.f32"](_mul, A_value, TK.float32(LOG2_E))
                _exp2 = TK.local_scalar("float32")
                TK.ptx["ex2.approx.ftz.f32"](_exp2, _mul)
                exp_A_value = _local_scalar("float32", _exp2)
                for t in range(SEQ_LEN):
                    q_base = TK.cast(n, "int64") * effective_q_batch_stride + TK.cast(
                        (t * NUM_HEADS + h) * K + k_start[0], "int64"
                    )
                    k_base = TK.cast(n, "int64") * effective_k_batch_stride + TK.cast(
                        (t * NUM_HEADS + h) * K + k_start[0], "int64"
                    )
                    q_input_ptr = q.ptr_to([q_base])
                    k_input_ptr = k.ptr_to([k_base])
                    for elem in range(VEC_SIZE):
                        TK.ptx.mov.b16(
                            r_q_bits[elem],
                            _global_load_u16_ptr_offset(q_input_ptr, elem * 2, USE_NATIVE_OFFSETS),
                        )
                    for elem in range(VEC_SIZE):
                        TK.ptx.mov.b16(
                            r_k_bits[elem],
                            _global_load_u16_ptr_offset(k_input_ptr, elem * 2, USE_NATIVE_OFFSETS),
                        )
                    for elem in range(VEC_SIZE):
                        TK.ptx.cvt.f32.bf16(r_q[elem], TK.cast(r_q_bits[elem], "uint16"))
                        TK.ptx.cvt.f32.bf16(r_k[elem], TK.cast(r_k_bits[elem], "uint16"))
                    if USE_QK_L2NORM:
                        sum_q = _local_scalar("float32", TK.float32(0.0))
                        sum_k = _local_scalar("float32", TK.float32(0.0))
                        for elem in range(VEC_SIZE):
                            TK.ptx.fma.rn.f32.bf16(
                                sum_q[0],
                                TK.cast(r_q_bits[elem], "uint16"),
                                TK.cast(r_q_bits[elem], "uint16"),
                                sum_q[0],
                            )
                            TK.ptx.fma.rn.f32.bf16(
                                sum_k[0],
                                TK.cast(r_k_bits[elem], "uint16"),
                                TK.cast(r_k_bits[elem], "uint16"),
                                sum_k[0],
                            )
                        for delta_index in range(5):
                            delta = _local_scalar(
                                "int32", TK.shift_right(TK.int32(16), delta_index)
                            )
                            TK.ptx["add.f32"](
                                sum_q[0], sum_q[0], _shfl_bfly_f32(sum_q[0], delta[0])
                            )
                            TK.ptx["add.f32"](
                                sum_k[0], sum_k[0], _shfl_bfly_f32(sum_k[0], delta[0])
                            )
                        _add = TK.local_scalar("float32")
                        TK.ptx["add.f32"](_add, sum_q[0], TK.float32(1e-06))
                        _rsqrt = TK.local_scalar("float32")
                        TK.ptx["rsqrt.approx.ftz.f32"](_rsqrt, _add)
                        _mul2 = TK.local_scalar("float32")
                        TK.ptx["mul.f32"](_mul2, _rsqrt, TK.float32(SCALE))
                        q_factor = _local_scalar("float32", _mul2)
                        _add2 = TK.local_scalar("float32")
                        TK.ptx["add.f32"](_add2, sum_k[0], TK.float32(1e-06))
                        _rsqrt2 = TK.local_scalar("float32")
                        TK.ptx["rsqrt.approx.ftz.f32"](_rsqrt2, _add2)
                        k_factor = _local_scalar("float32", _rsqrt2)
                        for elem in range(VEC_SIZE):
                            TK.ptx["mul.f32"](r_q[elem], r_q[elem], q_factor[0])
                            TK.ptx["mul.f32"](r_k[elem], r_k[elem], k_factor[0])
                    else:
                        for elem in range(VEC_SIZE):
                            TK.ptx["mul.f32"](r_q[elem], r_q[elem], TK.float32(SCALE))
                    shared_base = t * (K + 8) + k_start[0]
                    shared_q_ptr = s_q.ptr_to([shared_base])
                    for elem in range(VEC_SIZE):
                        _shared_store_f32_ptr_offset(
                            shared_q_ptr, elem * 4, r_q[elem], USE_NATIVE_OFFSETS
                        )
                        _shared_store_f32_ptr_offset(
                            shared_q_ptr, S_K_BYTE_OFFSET + elem * 4, r_k[elem], USE_NATIVE_OFFSETS
                        )
                    gate_index = (n * SEQ_LEN + t) * NUM_V_HEADS + hv
                    a_bits = TK.local_scalar("uint16")
                    TK.ptx.ld.global_.b16(a_bits, a.ptr_to([gate_index]))
                    b_bits = TK.local_scalar("uint16")
                    TK.ptx.ld.global_.b16(b_bits, b_gate.ptr_to([gate_index]))
                    _gate_pair_store(
                        gate_pair_value, gate_scratch, a_bits, b_bits, exp_A_value[0], dt_value
                    )
                    shared_g_ptr = s_g.ptr_to([t])
                    TK.ptx.st.shared.b32(
                        shared_g_ptr, TK.reinterpret("uint32", TK.cuda.float2_x(gate_pair_value[0]))
                    )
                    _shared_store_f32_ptr_offset(
                        shared_g_ptr,
                        S_BETA_BYTE_OFFSET - S_G_BYTE_OFFSET,
                        TK.cuda.float2_y(gate_pair_value[0]),
                        USE_NATIVE_OFFSETS,
                    )
                    if USE_SMEM_V:
                        with TK.If(tid < TILE_V), TK.Then():
                            v_input_base = TK.cast(n, "int64") * effective_v_batch_stride + TK.cast(
                                (t * NUM_V_HEADS + hv) * V + v_tile * TILE_V + tid, "int64"
                            )
                            v_input_ptr = v.ptr_to([v_input_base])
                            v_bits = TK.alloc_local((1,), "uint16")
                            TK.ptx.ld.global_.b16(v_bits[0], v_input_ptr)
                            _f32 = TK.local_scalar("float32")
                            TK.ptx.cvt.f32.bf16(_f32, TK.cast(v_bits[0], "uint16"))
                            TK.ptx.st.shared.b32(
                                s_v.ptr_to([t * TILE_V + tid]), TK.reinterpret("uint32", _f32)
                            )
            with workers:
                if PREFETCH_ROWS > 0:
                    pre_v_base = v_tile * TILE_V + warp * ROWS_PER_GROUP
                    prefetch_base = read_state_base + TK.cast(pre_v_base * K + k_start[0], "int64")
                    prefetch_ptr = state.ptr_to([prefetch_base])
                    for row in range(PREFETCH_ROWS):
                        _global_load_f32x4_b64_ptr_offset(
                            prefetch_ptr, row * K * 4, r_h, row * VEC_SIZE, USE_NATIVE_OFFSETS
                        )
                if USE_SMEM_V:
                    for t in range(SEQ_LEN):
                        with TK.If(tid < TILE_V), TK.Then():
                            v_input_base = TK.cast(n, "int64") * effective_v_batch_stride + TK.cast(
                                (t * NUM_V_HEADS + hv) * V + v_tile * TILE_V + tid, "int64"
                            )
                            v_input_ptr = v.ptr_to([v_input_base])
                            v_bits = TK.alloc_local((1,), "uint16")
                            TK.ptx.ld.global_.b16(v_bits[0], v_input_ptr)
                            _f32_2 = TK.local_scalar("float32")
                            TK.ptx.cvt.f32.bf16(_f32_2, TK.cast(v_bits[0], "uint16"))
                            TK.ptx.st.shared.b32(
                                s_v.ptr_to([t * TILE_V + tid]), TK.reinterpret("uint32", _f32_2)
                            )
            TK.cuda.cta_sync()
            for iter_index in range(ITERS_PER_GROUP):
                v_base = v_tile * TILE_V + warp * ROWS_PER_GROUP + iter_index * ILP_ROWS
                read_offset = _local_scalar(
                    "int64", read_state_base + TK.cast(v_base * K + k_start[0], "int64")
                )
                if ILP_ROWS == 8 or iter_index > 0:
                    read_ptr = state.ptr_to([read_offset[0]])
                    for row in range(ILP_ROWS):
                        if ILP_ROWS < 8 and iter_index == 0:
                            _global_load_f32x4_b64_ptr_offset(
                                read_ptr, row * K * 4, r_h, row * VEC_SIZE, USE_NATIVE_OFFSETS
                            )
                        else:
                            _global_load_f32x4_ptr_offset(
                                read_ptr, row * K * 4, r_h, row * VEC_SIZE, USE_NATIVE_OFFSETS
                            )
                else:
                    with TK.If(warp == 0), TK.Then():
                        read_ptr = state.ptr_to([read_offset[0]])
                        for row in range(ILP_ROWS):
                            _global_load_f32x4_b64_ptr_offset(
                                read_ptr, row * K * 4, r_h, row * VEC_SIZE, USE_NATIVE_OFFSETS
                            )
                sums = TK.alloc_local((ILP_ROWS,), "float32")
                residuals = TK.alloc_local((ILP_ROWS,), "float32")
                output_sums = TK.alloc_local((ILP_ROWS,), "float32")
                sum_lo = TK.alloc_local((4,), "float32")
                sum_hi = TK.alloc_local((4,), "float32")
                output_lo = TK.alloc_local((4,), "float32")
                output_hi = TK.alloc_local((4,), "float32")
                packed_value = TK.alloc_local((1,), "uint64")
                output_pair_bits = TK.local_scalar("uint32")
                output_scalar_bits = TK.alloc_local((ILP_ROWS,), "uint16")
                for t in range(SEQ_LEN):
                    shared_q_ptr = s_q.ptr_to([t * (K + 8) + k_start[0]])
                    if ILP_ROWS == 4:
                        _shared_load_f32x4_b64_ptr_offset(
                            shared_q_ptr, S_K_BYTE_OFFSET, r_k, USE_NATIVE_OFFSETS
                        )
                    else:
                        _shared_load_f32x4_ptr(shared_q_ptr, r_q)
                        _shared_load_f32x4_ptr_offset(
                            shared_q_ptr, S_K_BYTE_OFFSET, r_k, USE_NATIVE_OFFSETS
                        )
                    shared_g_ptr = s_g.ptr_to([t])
                    _lds32 = TK.local_scalar("uint32")
                    TK.ptx.ld.shared.b32(_lds32, shared_g_ptr)
                    g_value = _local_scalar("float32", TK.reinterpret("float32", _lds32))
                    beta = _local_scalar(
                        "float32",
                        _shared_load_f32_ptr_offset(
                            shared_g_ptr, S_BETA_BYTE_OFFSET - S_G_BYTE_OFFSET, USE_NATIVE_OFFSETS
                        ),
                    )
                    if ILP_ROWS == 4:
                        for row in range(4):
                            TK.ptx.mov.b32(sum_lo[row], TK.float32(0.0))
                            TK.ptx.mov.b32(sum_hi[row], TK.float32(0.0))
                        for pair in range(2):
                            for row in range(4):
                                base = _local_scalar("int32", row * VEC_SIZE + pair * 2)
                                TK.ptx.mul.f32(r_h[base[0]], r_h[base[0]], g_value[0])
                                TK.ptx.mul.f32(r_h[base[0] + 1], r_h[base[0] + 1], g_value[0])
                                _packed_fma_store(
                                    packed_value,
                                    r_h[base[0]],
                                    r_h[base[0] + 1],
                                    r_k[pair * 2],
                                    r_k[pair * 2 + 1],
                                    sum_lo[row],
                                    sum_hi[row],
                                )
                                TK.ptx.mov.b32(sum_lo[row], TK.cuda.float2_x(packed_value[0]))
                                TK.ptx.mov.b32(sum_hi[row], TK.cuda.float2_y(packed_value[0]))
                        for row in range(4):
                            TK.ptx.add.f32(sum_lo[row], sum_lo[row], sum_hi[row])
                    else:
                        for row in range(ILP_ROWS):
                            TK.ptx.mov.b32(sums[row], TK.float32(0.0))
                        for elem in range(VEC_SIZE):
                            for row in range(ILP_ROWS):
                                index = _local_scalar("int32", row * VEC_SIZE + elem)
                                TK.ptx["mul.f32"](r_h[index[0]], r_h[index[0]], g_value[0])
                                TK.ptx["fma.rn.f32"](sums[row], r_h[index[0]], r_k[elem], sums[row])
                    for delta_index in range(5):
                        delta = _local_scalar("int32", TK.shift_right(TK.int32(16), delta_index))
                        for row in range(ILP_ROWS):
                            if ILP_ROWS == 4:
                                TK.ptx.add.f32(
                                    sum_lo[row], sum_lo[row], _shfl_bfly_f32(sum_lo[row], delta[0])
                                )
                            else:
                                TK.ptx["add.f32"](
                                    sums[row], sums[row], _shfl_bfly_f32(sums[row], delta[0])
                                )
                    v_input_base = TK.cast(n, "int64") * effective_v_batch_stride + TK.cast(
                        (t * NUM_V_HEADS + hv) * V + v_base, "int64"
                    )
                    v_input_ptr = v.ptr_to([v_input_base])
                    shared_v_ptr = s_v.ptr_to([t * TILE_V + v_base - v_tile * TILE_V])
                    for row in range(ILP_ROWS):
                        if USE_SMEM_V:
                            v_value = _local_scalar(
                                "float32",
                                _shared_load_f32_ptr_offset(
                                    shared_v_ptr, row * 4, USE_NATIVE_OFFSETS
                                ),
                            )
                            if ILP_ROWS == 4:
                                TK.ptx.sub.f32(residuals[row], v_value[0], sum_lo[row])
                                TK.ptx.mul.f32(residuals[row], residuals[row], beta[0])
                            else:
                                _sub = TK.local_scalar("float32")
                                TK.ptx["sub.f32"](_sub, v_value[0], sums[row])
                                TK.ptx["mul.f32"](residuals[row], _sub, beta[0])
                        else:
                            v_bits = _local_scalar(
                                "uint16",
                                _global_load_u16_ptr_offset(
                                    v_input_ptr, row * 2, USE_NATIVE_OFFSETS
                                ),
                            )
                            if ILP_ROWS == 4:
                                TK.ptx.sub.rn.f32.bf16(
                                    residuals[row], TK.cast(v_bits[0], "uint16"), sum_lo[row]
                                )
                                TK.ptx.mul.f32(residuals[row], residuals[row], beta[0])
                            else:
                                _subbf = TK.local_scalar("float32")
                                TK.ptx.sub.rn.f32.bf16(
                                    _subbf, TK.cast(v_bits[0], "uint16"), sums[row]
                                )
                                TK.ptx["mul.f32"](residuals[row], _subbf, beta[0])
                    if ILP_ROWS == 4:
                        _shared_load_f32x4_b64_ptr(shared_q_ptr, r_q)
                        if RELOAD_K_FOR_OUTPUT:
                            _shared_load_f32x4_b64_ptr_offset(
                                shared_q_ptr, S_K_BYTE_OFFSET, r_k_output, USE_NATIVE_OFFSETS
                            )
                        for row in range(4):
                            TK.ptx.mov.b32(output_lo[row], TK.float32(0.0))
                            TK.ptx.mov.b32(output_hi[row], TK.float32(0.0))
                        for pair in range(2):
                            for row in range(4):
                                base = _local_scalar("int32", row * VEC_SIZE + pair * 2)
                                if RELOAD_K_FOR_OUTPUT:
                                    _packed_fma_store(
                                        packed_value,
                                        r_k_output[pair * 2],
                                        r_k_output[pair * 2 + 1],
                                        residuals[row],
                                        residuals[row],
                                        r_h[base[0]],
                                        r_h[base[0] + 1],
                                    )
                                else:
                                    _packed_fma_store(
                                        packed_value,
                                        r_k[pair * 2],
                                        r_k[pair * 2 + 1],
                                        residuals[row],
                                        residuals[row],
                                        r_h[base[0]],
                                        r_h[base[0] + 1],
                                    )
                                TK.ptx.mov.b32(r_h[base[0]], TK.cuda.float2_x(packed_value[0]))
                                TK.ptx.mov.b32(r_h[base[0] + 1], TK.cuda.float2_y(packed_value[0]))
                                _packed_fma_store(
                                    packed_value,
                                    r_h[base[0]],
                                    r_h[base[0] + 1],
                                    r_q[pair * 2],
                                    r_q[pair * 2 + 1],
                                    output_lo[row],
                                    output_hi[row],
                                )
                                TK.ptx.mov.b32(output_lo[row], TK.cuda.float2_x(packed_value[0]))
                                TK.ptx.mov.b32(output_hi[row], TK.cuda.float2_y(packed_value[0]))
                        for row in range(4):
                            TK.ptx.add.f32(output_lo[row], output_lo[row], output_hi[row])
                    else:
                        for row in range(ILP_ROWS):
                            TK.ptx.mov.b32(output_sums[row], TK.float32(0.0))
                        for elem in range(VEC_SIZE):
                            for row in range(ILP_ROWS):
                                index = _local_scalar("int32", row * VEC_SIZE + elem)
                                TK.ptx["fma.rn.f32"](
                                    r_h[index[0]], r_k[elem], residuals[row], r_h[index[0]]
                                )
                                TK.ptx["fma.rn.f32"](
                                    output_sums[row], r_h[index[0]], r_q[elem], output_sums[row]
                                )
                    if CACHE_INTERMEDIATE_STATES and ILP_ROWS != 4:
                        intermediate_base = TK.cast(
                            ((n * SEQ_LEN + t) * NUM_V_HEADS + hv) * V * K
                            + v_base * K
                            + k_start[0],
                            "int64",
                        )
                        intermediate_ptr = intermediate.ptr_to([intermediate_base])
                        for row in range(ILP_ROWS):
                            if ILP_ROWS == 2 and t > 0:
                                _global_store_f32x4_b64_ptr_offset(
                                    intermediate_ptr,
                                    row * K * 4,
                                    r_h,
                                    row * VEC_SIZE,
                                    USE_NATIVE_OFFSETS,
                                )
                            else:
                                _global_store_f32x4_ptr_offset(
                                    intermediate_ptr,
                                    row * K * 4,
                                    r_h,
                                    row * VEC_SIZE,
                                    USE_NATIVE_OFFSETS,
                                )
                    if PER_TOKEN_POOL_SCATTER and ILP_ROWS != 4:
                        scatter_slot = TK.alloc_local((1,), "int32")
                        TK.ptx.ld.global_.s32(
                            scatter_slot[0], ssm_state_indices.ptr_to([n * SEQ_LEN + t])
                        )
                        scatter_base = (
                            TK.cast(scatter_slot[0], "int64") * effective_state_slot_stride
                            + TK.cast(hv, "int64") * effective_state_head_stride
                            + TK.cast(v_base * K + k_start[0], "int64")
                        )
                        scatter_ptr = state.ptr_to([scatter_base])
                        for row in range(ILP_ROWS):
                            _global_store_f32x4_ptr_offset(
                                scatter_ptr, row * K * 4, r_h, row * VEC_SIZE, USE_NATIVE_OFFSETS
                            )
                    for delta_index in range(5):
                        delta = _local_scalar("int32", TK.shift_right(TK.int32(16), delta_index))
                        for row in range(ILP_ROWS):
                            if ILP_ROWS == 4:
                                TK.ptx.add.f32(
                                    output_lo[row],
                                    output_lo[row],
                                    _shfl_bfly_f32(output_lo[row], delta[0]),
                                )
                            else:
                                TK.ptx["add.f32"](
                                    output_sums[row],
                                    output_sums[row],
                                    _shfl_bfly_f32(output_sums[row], delta[0]),
                                )
                    with TK.If(lane == 0), TK.Then():
                        output_base = TK.cast(
                            ((n * SEQ_LEN + t) * NUM_V_HEADS + hv) * V + v_base, "int64"
                        )
                        output_ptr = output.ptr_to([output_base])
                        shared_output_ptr = s_output.ptr_to([t * TILE_V + v_base - v_tile * TILE_V])
                        if ILP_ROWS == 2:
                            TK.ptx.cvt.rn.bf16x2.f32(
                                output_pair_bits, output_sums[1], output_sums[0]
                            )
                            TK.ptx.st.global_.b32(output_ptr, output_pair_bits)
                        elif USE_PACKED_OUTPUT:
                            for pair in range(ILP_ROWS // 2):
                                output_value_0 = _local_scalar("float32", output_sums[pair * 2])
                                output_value_1 = _local_scalar("float32", output_sums[pair * 2 + 1])
                                if ILP_ROWS == 4:
                                    TK.assign(output_value_0[0], output_lo[pair * 2])
                                    TK.assign(output_value_1[0], output_lo[pair * 2 + 1])
                                TK.ptx.cvt.rn.bf16x2.f32(
                                    output_pair_bits, output_value_1[0], output_value_0[0]
                                )
                                if USE_SMEM_V:
                                    _shared_store_u32_ptr_offset(
                                        shared_output_ptr,
                                        pair * 4,
                                        output_pair_bits,
                                        USE_NATIVE_OFFSETS,
                                    )
                                else:
                                    _global_store_u32_ptr_offset(
                                        output_ptr, pair * 4, output_pair_bits, USE_NATIVE_OFFSETS
                                    )
                        else:
                            for row in range(ILP_ROWS):
                                output_value = _local_scalar("float32", output_sums[row])
                                if ILP_ROWS == 4:
                                    TK.assign(output_value[0], output_lo[row])
                                TK.ptx.cvt.rn.bf16.f32(output_scalar_bits[row], output_value[0])
                                if USE_SMEM_V:
                                    _shared_store_u16_ptr_offset(
                                        shared_output_ptr,
                                        row * 2,
                                        output_scalar_bits[row],
                                        USE_NATIVE_OFFSETS,
                                    )
                                else:
                                    _global_store_u16_ptr_offset(
                                        output_ptr,
                                        row * 2,
                                        output_scalar_bits[row],
                                        USE_NATIVE_OFFSETS,
                                    )
                    if CACHE_INTERMEDIATE_STATES and ILP_ROWS == 4:
                        intermediate_base = TK.cast(
                            ((n * SEQ_LEN + t) * NUM_V_HEADS + hv) * V * K
                            + v_base * K
                            + k_start[0],
                            "int64",
                        )
                        intermediate_ptr = intermediate.ptr_to([intermediate_base])
                        for row in range(4):
                            _global_store_f32x4_b64_ptr_offset(
                                intermediate_ptr,
                                row * K * 4,
                                r_h,
                                row * VEC_SIZE,
                                USE_NATIVE_OFFSETS,
                            )
                    if PER_TOKEN_POOL_SCATTER and ILP_ROWS == 4:
                        scatter_slot = TK.alloc_local((1,), "int32")
                        TK.ptx.ld.global_.s32(
                            scatter_slot[0], ssm_state_indices.ptr_to([n * SEQ_LEN + t])
                        )
                        scatter_base = (
                            TK.cast(scatter_slot[0], "int64") * effective_state_slot_stride
                            + TK.cast(hv, "int64") * effective_state_head_stride
                            + TK.cast(v_base * K + k_start[0], "int64")
                        )
                        scatter_ptr = state.ptr_to([scatter_base])
                        for row in range(4):
                            _global_store_f32x4_b64_ptr_offset(
                                scatter_ptr, row * K * 4, r_h, row * VEC_SIZE, USE_NATIVE_OFFSETS
                            )
                if not DISABLE_STATE_UPDATE and (not PER_TOKEN_POOL_SCATTER):
                    with TK.If(write_slot_raw[0] >= 0), TK.Then():
                        write_offset = _local_scalar(
                            "int64", write_state_base[0] + TK.cast(v_base * K + k_start[0], "int64")
                        )
                        write_ptr = state.ptr_to([write_offset[0]])
                        for row in range(ILP_ROWS):
                            if ILP_ROWS == 4 or (ILP_ROWS == 2 and CACHE_INTERMEDIATE_STATES):
                                _global_store_f32x4_b64_ptr_offset(
                                    write_ptr, row * K * 4, r_h, row * VEC_SIZE, USE_NATIVE_OFFSETS
                                )
                            else:
                                _global_store_f32x4_ptr_offset(
                                    write_ptr, row * K * 4, r_h, row * VEC_SIZE, USE_NATIVE_OFFSETS
                                )
            if USE_SMEM_V:
                TK.cuda.cta_sync()
                output_tile_base = TK.cast(
                    (n * SEQ_LEN * NUM_V_HEADS + hv) * V + v_tile * TILE_V, "int64"
                )
                with TK.If(tid < TILE_V), TK.Then():
                    output_tile_ptr = output.ptr_to([output_tile_base + tid])
                    shared_output_ptr = s_output.ptr_to([tid])
                    for t in range(SEQ_LEN):
                        output_bits = _local_scalar(
                            "uint16",
                            _shared_load_u16_ptr_offset(
                                shared_output_ptr, t * TILE_V * 2, USE_NATIVE_OFFSETS
                            ),
                        )
                        _global_store_u16_ptr_offset(
                            output_tile_ptr,
                            t * NUM_V_HEADS * V * 2,
                            output_bits[0],
                            USE_NATIVE_OFFSETS,
                        )

    return gdn_decode_fp32_mtp_warp.func


def _pool_factor(config: dict[str, Any]) -> int:
    factor = 1
    if config.get("negative_read_index", False) or config.get("negative_write_index", False):
        factor += 1
    if config.get("per_token_pool_scatter", False):
        factor += int(config["seq_len"])
    if not config.get("same_pool", True):
        factor += 1
    return factor


def get_kernel(**kwargs: Any):
    """Return the source-specialized TIRx PrimFunc."""
    config = dict(kwargs)
    _require_supported_config(config)
    seq_len = int(config["seq_len"])
    num_v_heads = int(config["num_v_heads"])
    work_units = int(config["batch"]) * num_v_heads
    tile_v, ilp_rows, use_smem_v = _target_config(
        int(config["batch"]),
        seq_len,
        int(config["num_heads"]),
        num_v_heads,
        disable_state_update=bool(config.get("disable_state_update", False)),
    )
    scatter = bool(config.get("per_token_pool_scatter", False))
    scatter_flat = scatter and not bool(config.get("padded_pool", False))
    cache = bool(config.get("cache_intermediate_states", False))
    pool_factor = int(config.get("pool_factor_override", _pool_factor(config)))
    intermediate_batch_stride = 0
    if cache:
        intermediate_batch_stride = seq_len * num_v_heads * V * K
    qk_bytes = 4 * ((seq_len - 1) * (K + 8) + K)
    gate_bytes_aligned = ((4 * seq_len + 15) // 16) * 16
    s_g_byte_offset = 2 * qk_bytes
    s_v_byte_offset = s_g_byte_offset + 2 * gate_bytes_aligned
    return _make_gdn_decode_fp32_mtp_warp(
        SEQ_LEN=seq_len,
        NUM_HEADS=int(config["num_heads"]),
        NUM_V_HEADS=num_v_heads,
        TILE_V=tile_v,
        NUM_V_TILES=V // tile_v,
        ILP_ROWS=ilp_rows,
        USE_SMEM_V=use_smem_v,
        USE_QK_L2NORM=bool(config.get("use_qk_l2norm", True)),
        USE_NATIVE_OFFSETS=_HAS_NATIVE_PTX_ADDR
        and (seq_len == 2 or 3 < seq_len < 8)
        and not (seq_len == 4 and int(config["num_heads"]) <= 8 and work_units == 1024),
        RELOAD_K_FOR_OUTPUT=seq_len == 8 and not use_smem_v,
        USE_CANONICAL_WARP_ID=seq_len == 4,
        USE_PACKED_OUTPUT=not (
            (seq_len in (5, 7) and work_units == 512)
            or (seq_len == 8 and int(config["num_heads"]) <= 4)
        ),
        DISABLE_STATE_UPDATE=bool(config.get("disable_state_update", False)),
        CACHE_INTERMEDIATE_STATES=cache,
        SAME_POOL=bool(config.get("same_pool", True)),
        PER_TOKEN_POOL_SCATTER=scatter,
        PER_TOKEN_POOL_SCATTER_FLAT=scatter_flat,
        PADDED_POOL=bool(config.get("padded_pool", False)),
        PACKED_QKV=bool(config.get("packed_qkv", False)),
        POOL_FACTOR=pool_factor,
        INTERMEDIATE_BATCH_STRIDE=intermediate_batch_stride,
        INTERMEDIATE_DUMMY_ELEMENTS=0 if cache else 1,
        SSM_BATCH_STRIDE=seq_len if scatter else 0,
        SSM_DUMMY_ELEMENTS=0 if scatter else 1,
        SHARED_BYTES=8 * seq_len * (K + 8) + 8 * seq_len + 6 * seq_len * tile_v + 128,
        S_K_BYTE_OFFSET=qk_bytes,
        S_G_BYTE_OFFSET=s_g_byte_offset,
        S_BETA_BYTE_OFFSET=s_g_byte_offset + gate_bytes_aligned,
        S_V_BYTE_OFFSET=s_v_byte_offset,
        S_OUTPUT_BYTE_OFFSET=s_v_byte_offset + 4 * seq_len * tile_v,
        ROWS_PER_GROUP=tile_v // NUM_GROUPS,
        ITERS_PER_GROUP=(tile_v // NUM_GROUPS) // ilp_rows,
        PREFETCH_ROWS=0 if ilp_rows == 8 else ilp_rows,
    )


def _allocate_pool(
    pool_slots: int,
    num_v_heads: int,
    *,
    padded: bool,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if padded:
        backing = (
            torch.randn(
                (pool_slots, num_v_heads * 2 + 1, V, K),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 0.05
        )
        return backing[:, : num_v_heads * 2 : 2], backing
    backing = (
        torch.randn(
            (pool_slots, num_v_heads, V, K), dtype=torch.float32, device=device, generator=generator
        )
        * 0.05
    )
    return backing, backing


def _clone_pool_layout(pool: torch.Tensor, *, padded: bool) -> tuple[torch.Tensor, torch.Tensor]:
    pool_slots, num_v_heads = pool.shape[:2]
    backing = torch.empty(
        (pool_slots, num_v_heads * 2 + 1 if padded else num_v_heads, V, K),
        dtype=pool.dtype,
        device=pool.device,
    )
    view = backing[:, : num_v_heads * 2 : 2] if padded else backing
    view.copy_(pool)
    return view, backing


def _make_qkv(
    batch: int,
    seq_len: int,
    num_heads: int,
    num_v_heads: int,
    *,
    packed: bool,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not packed:
        q = (
            torch.randn(
                (batch, seq_len, num_heads, K),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            * 0.05
        )
        k = (
            torch.randn(
                (batch, seq_len, num_heads, K),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            * 0.05
        )
        v = (
            torch.randn(
                (batch, seq_len, num_v_heads, V),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            * 0.05
        )
        return q, k, v, None

    q_elements = seq_len * num_heads * K
    k_elements = seq_len * num_heads * K
    v_elements = seq_len * num_v_heads * V
    row_elements = q_elements + k_elements + v_elements + 128
    backing = (
        torch.randn((batch, row_elements), dtype=torch.bfloat16, device=device, generator=generator)
        * 0.05
    )
    q = backing.as_strided((batch, seq_len, num_heads, K), (row_elements, num_heads * K, K, 1), 0)
    k = backing.as_strided(
        (batch, seq_len, num_heads, K), (row_elements, num_heads * K, K, 1), q_elements
    )
    v = backing.as_strided(
        (batch, seq_len, num_v_heads, V),
        (row_elements, num_v_heads * V, V, 1),
        q_elements + k_elements,
    )
    return q, k, v, backing


def _device_from_config(config: dict[str, Any]) -> torch.device:
    configured_device = config.get("device")
    device = (
        torch.device(configured_device)
        if configured_device is not None
        else torch.device("cuda", torch.cuda.current_device())
    )
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SkipTest("CUDA is required for FP32 MTP warp GDN decode")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SkipTest(f"FP32 MTP warp GDN decode requires SM100, got {capability}")
    return device


@functools.cache
def _load_oracle():
    import flashinfer.gdn_decode as public

    return public.gated_delta_rule_mtp


@functools.cache
def _compile_tirx(
    work_units: int,
    seq_len: int,
    num_heads: int,
    num_v_heads: int,
    tile_v: int,
    ilp_rows: int,
    use_smem_v: bool,
    pool_factor: int,
    use_qk_l2norm: bool,
    disable_state_update: bool,
    cache_intermediate_states: bool,
    same_pool: bool,
    per_token_pool_scatter: bool,
    padded_pool: bool,
    packed_qkv: bool,
):
    from tirx_kernels.runner import compile_kernel

    representative_work_units = work_units
    config = {
        "seq_len": seq_len,
        "batch": representative_work_units // num_v_heads,
        "num_heads": num_heads,
        "num_v_heads": num_v_heads,
        "tile_v": tile_v,
        "ilp_rows": ilp_rows,
        "use_smem_v": use_smem_v,
        "pool_factor_override": pool_factor,
        "use_qk_l2norm": use_qk_l2norm,
        "disable_state_update": disable_state_update,
        "cache_intermediate_states": cache_intermediate_states,
        "same_pool": same_pool,
        "per_token_pool_scatter": per_token_pool_scatter,
        "padded_pool": padded_pool,
        "packed_qkv": packed_qkv,
    }
    reg_level: int | None = None
    if seq_len == 2:
        reg_level = 4 if tile_v == 16 or work_units == 2048 else 0
    elif seq_len == 4 and num_heads == 2 and tile_v == 32:
        reg_level = 4
    elif seq_len in (3, 4) and num_heads < 16 and tile_v == 32:
        reg_level = 4
    elif seq_len == 4 and num_heads == 4 and tile_v == 64:
        reg_level = 4
    elif seq_len == 5 and tile_v == 32:
        reg_level = 4
    elif seq_len == 7 and tile_v == 32:
        reg_level = 0
    elif seq_len == 8 and tile_v == 64 and num_heads >= 8:
        reg_level = 0
    elif seq_len == 8 and num_heads <= 4:
        reg_level = 4
    if reg_level is None:
        os.environ.pop("TVM_CUDA_PTXAS_REG_LEVEL", None)
    else:
        os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = str(reg_level)
    return compile_kernel(get_kernel(**config))


def _compile_tirx_for_config(config: dict[str, Any]):
    return _compile_tirx(
        int(config["batch"]) * int(config["num_v_heads"]),
        int(config["seq_len"]),
        int(config["num_heads"]),
        int(config["num_v_heads"]),
        int(config["tile_v"]),
        int(config["ilp_rows"]),
        bool(config["use_smem_v"]),
        _pool_factor(config),
        bool(config.get("use_qk_l2norm", True)),
        bool(config.get("disable_state_update", False)),
        bool(config.get("cache_intermediate_states", False)),
        bool(config.get("same_pool", True)),
        bool(config.get("per_token_pool_scatter", False)),
        bool(config.get("padded_pool", False)),
        bool(config.get("packed_qkv", False)),
    )


def _tirx_executable(case: dict[str, Any]):
    return _compile_tirx_for_config(case["config"])


def _storage_span(tensor: torch.Tensor, elements: int) -> torch.Tensor:
    return tensor.as_strided((elements,), (1,))


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    config = case["config"]
    state = case["tirx_state"]
    q = case["q"]
    k = case["k"]
    v = case["v"]
    batch = int(config["batch"])
    state_elements = int(state.stride(0)) * case["pool_slots"]
    return (
        _storage_span(state, state_elements),
        _storage_span(case["tirx_intermediate"], case["tirx_intermediate"].numel()),
        case["A_log"],
        case["a"].reshape(-1),
        case["dt_bias"],
        _storage_span(q, int(q.stride(0)) * (batch - 1) + int(q[0].numel())),
        _storage_span(k, int(k.stride(0)) * (batch - 1) + int(k[0].numel())),
        _storage_span(v, int(v.stride(0)) * (batch - 1) + int(v[0].numel())),
        case["b_gate"].reshape(-1),
        case["tirx_output"].reshape(-1),
        case["read_indices"],
        case["write_indices"],
        case["ssm_state_indices"].reshape(-1),
        int(state.stride(0)),
        int(state.stride(1)),
        int(q.stride(0)),
        int(k.stride(0)),
        int(v.stride(0)),
        batch,
    )


def _run_reference(case: dict[str, Any]) -> torch.Tensor:
    config = case["config"]
    if not case.get("source_cache_initialized", False):
        import flashinfer.gdn_kernels.gdn_decode_mtp as source_module

        # The frozen native-4D cache key records strides but not pool_size,
        # even though the compiled tensor keeps shape[0] static.  Isolate each
        # correctness/benchmark case once, then retain its compiled source for
        # all subsequent launches and timing rounds.
        source_module._get_compiled_mtp_kernel.cache_clear()
        source_module._get_compiled_mtp_kernel_inline.cache_clear()
        case["source_cache_initialized"] = True
    oracle = _load_oracle()
    output, _ = oracle(
        q=case["q"],
        k=case["k"],
        v=case["v"],
        initial_state=case["source_state"],
        initial_state_indices=case["read_indices"],
        A_log=case["A_log"],
        a=case["a"],
        dt_bias=case["dt_bias"],
        b=case["b_gate"],
        scale=SCALE,
        output=case["source_output"],
        intermediate_states_buffer=(
            case["source_intermediate"]
            if bool(config.get("cache_intermediate_states", False))
            else None
        ),
        ssm_state_indices=(
            case["ssm_state_indices"] if bool(config.get("per_token_pool_scatter", False)) else None
        ),
        disable_state_update=bool(config.get("disable_state_update", False)),
        use_qk_l2norm=bool(config.get("use_qk_l2norm", True)),
        output_state_indices=(
            None if bool(config.get("same_pool", True)) else case["write_indices"]
        ),
    )
    return output


def _assert_case_close(case: dict[str, Any]) -> None:
    config = case["config"]
    torch.testing.assert_close(
        case["tirx_output"].float(), case["source_output"].float(), atol=1.0e-3, rtol=5.0e-3
    )
    torch.testing.assert_close(case["tirx_state"], case["source_state"], atol=2.0e-5, rtol=2.0e-5)
    if bool(config.get("cache_intermediate_states", False)):
        torch.testing.assert_close(
            case["tirx_intermediate"], case["source_intermediate"], atol=2.0e-5, rtol=2.0e-5
        )
    if case["qkv_backing"] is not None:
        torch.testing.assert_close(case["qkv_backing"], case["qkv_snapshot"], atol=0, rtol=0)
    if bool(config.get("padded_pool", False)):
        backing = case["tirx_state_backing"]
        snapshot = case["tirx_state_backing_snapshot"]
        torch.testing.assert_close(backing[:, 1::2], snapshot[:, 1::2], atol=0, rtol=0)
        torch.testing.assert_close(backing[:, -1], snapshot[:, -1], atol=0, rtol=0)


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Create deterministic, independently mutable TIRx and FlashInfer cases."""
    config = dict(kwargs)
    _require_supported_config(config)
    device = _device_from_config(config)
    batch = int(config["batch"])
    seq_len = int(config["seq_len"])
    num_heads = int(config["num_heads"])
    num_v_heads = int(config["num_v_heads"])
    padded_pool = bool(config.get("padded_pool", False))
    same_pool = bool(config.get("same_pool", True))
    scatter = bool(config.get("per_token_pool_scatter", False))
    negative_read = bool(config.get("negative_read_index", False))
    negative_write = bool(config.get("negative_write_index", False))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config.get("seed", 0)) + 20260813)

    pool_factor = _pool_factor(config)
    pool_slots = batch * pool_factor
    initial_pool, initial_backing = _allocate_pool(
        pool_slots, num_v_heads, padded=padded_pool, device=device, generator=generator
    )
    tirx_state, tirx_state_backing = _clone_pool_layout(initial_pool, padded=padded_pool)
    if padded_pool and scatter:
        # The frozen source cannot construct native-4D + scatter.  Its flat
        # source specialization is algorithmically identical for the logical
        # pool and remains the trusted oracle for this combined target case.
        source_state = initial_pool.contiguous()
        source_state_backing = source_state
    else:
        source_state, source_state_backing = _clone_pool_layout(initial_pool, padded=padded_pool)

    slot_offset = 1 if (negative_read or negative_write) else 0
    read_indices = torch.arange(batch, dtype=torch.int32, device=device) + slot_offset
    if negative_read:
        read_indices[-1] = -1

    next_slot = slot_offset + batch
    if scatter:
        scatter_indices = torch.arange(
            next_slot, next_slot + batch * seq_len, dtype=torch.int32, device=device
        ).reshape(batch, seq_len)
        next_slot += batch * seq_len
    else:
        scatter_indices = None
    if same_pool:
        write_indices = read_indices.clone()
    else:
        write_indices = torch.arange(next_slot, next_slot + batch, dtype=torch.int32, device=device)
        if negative_write:
            write_indices[-1] = -1

    q, k, v, qkv_backing = _make_qkv(
        batch,
        seq_len,
        num_heads,
        num_v_heads,
        packed=bool(config.get("packed_qkv", False)),
        device=device,
        generator=generator,
    )
    A_log = (
        torch.randn((num_v_heads,), dtype=torch.float32, device=device, generator=generator) * 0.1
    )
    dt_bias = (
        torch.randn((num_v_heads,), dtype=torch.float32, device=device, generator=generator) * 0.1
    )
    a = (
        torch.randn(
            (batch, seq_len, num_v_heads), dtype=torch.bfloat16, device=device, generator=generator
        )
        * 0.05
    )
    b_gate = (
        torch.randn(
            (batch, seq_len, num_v_heads), dtype=torch.bfloat16, device=device, generator=generator
        )
        * 0.05
    )
    output_initial = torch.randn(
        (batch, seq_len, num_v_heads, V), dtype=torch.bfloat16, device=device, generator=generator
    )
    if bool(config.get("cache_intermediate_states", False)):
        intermediate_initial = (
            torch.randn(
                (batch, seq_len, num_v_heads, V, K),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 0.05
        )
        tirx_intermediate = intermediate_initial.clone()
        source_intermediate = intermediate_initial.clone()
    else:
        tirx_intermediate = torch.zeros((1,), dtype=torch.float32, device=device)
        source_intermediate = torch.zeros((1,), dtype=torch.float32, device=device)

    return {
        "config": config,
        "pool_slots": pool_slots,
        "initial_pool": initial_pool.clone(),
        "initial_backing": initial_backing,
        "tirx_state": tirx_state,
        "tirx_state_backing": tirx_state_backing,
        "tirx_state_backing_snapshot": tirx_state_backing.clone(),
        "source_state": source_state,
        "source_state_backing": source_state_backing,
        "read_indices": read_indices,
        "write_indices": write_indices,
        "ssm_state_indices": (
            scatter_indices
            if scatter_indices is not None
            else torch.zeros((1,), dtype=torch.int32, device=device)
        ),
        "q": q,
        "k": k,
        "v": v,
        "qkv_backing": qkv_backing,
        "qkv_snapshot": qkv_backing.clone() if qkv_backing is not None else None,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "a": a,
        "b_gate": b_gate,
        "tirx_output": output_initial.clone(),
        "source_output": output_initial.clone(),
        "tirx_intermediate": tirx_intermediate,
        "source_intermediate": source_intermediate,
    }


def run_test(**kwargs: Any) -> None:
    case = prepare_data(**kwargs)
    executable = _tirx_executable(case)
    executable(*_tirx_args(case))
    torch.cuda.synchronize(case["tirx_state"].device)
    _run_reference(case)
    torch.cuda.synchronize(case["tirx_state"].device)
    _assert_case_close(case)


def prepare_bench(**kwargs: Any):
    """Compile the selected FP32 MTP warp specialization before CUDA setup."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    config = dict(kwargs)
    _require_supported_config(config)
    executable = _compile_tirx_for_config(config)
    return prepared_gpu_benchmark(run_gpu, {"config": dict(kwargs), "executable": executable})


def run_gpu(
    prepared,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int = 1,
    cooldown_s: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    kwargs = {**prepared["config"], **kwargs}
    case = prepare_data(**kwargs)
    executable = prepared["executable"]
    args = _tirx_args(case)

    def source_builder():
        executable(*args)
        _run_reference(case)
        torch.cuda.synchronize(case["tirx_state"].device)
        _assert_case_close(case)
        for _ in range(2):
            _run_reference(case)
        torch.cuda.synchronize(case["source_state"].device)

        def launch():
            _run_reference(case)

        return launch

    return bench(
        {"tirx": lambda: executable(*args)},
        references={"flashinfer_cutedsl": source_builder},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int = 1,
    cooldown_s: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    return prepare_bench(**kwargs).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_test",
]
