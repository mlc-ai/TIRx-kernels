# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM100 BF16 wide-vector single-token GDN decode kernel.

Upstream source: flashinfer/gdn_kernels/gdn_decode_bf16_state.py.
"""

import functools
import math
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as TK
from tirx_kernels.runner import bench

KERNEL_META = {
    "name": "gdn_decode_bf16_wide_vec_t1",
    "category": "flashinfer",
    "compute_capability": 10,
}


SEQ_LEN = 1
K = 128
V = 128
THREADS = 128
NUM_WARPS = 4
NUM_GROUPS = 8
LANES_PER_GROUP = 16
ELEMS_PER_LANE = 8
ILP_ROWS = 4
SHARED_BYTES = 1356
LOG2_E = 1.4426950408889634
LN_2 = 0.6931471805599453
SCALE = 1.0 / math.sqrt(K)


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


def _load_state_bf16x8(buffer, index, values, value_offset):
    words = TK.alloc_local((4,), "uint32", align=16)
    TK.ptx["ld.global.L1::evict_first.v4.b32"](
        words[0], words[1], words[2], words[3], buffer.ptr_to([index])
    )
    with TK.unroll(4) as pair:
        TK.ptx.mov.b32(
            values[value_offset + pair * 2],
            TK.cuda.uint_as_float(TK.shift_left(words[pair], TK.uint32(16))),
        )
        TK.ptx.mov.b32(
            values[value_offset + pair * 2 + 1],
            TK.cuda.uint_as_float(TK.bitwise_and(words[pair], TK.uint32(0xFFFF0000))),
        )


def _load_state_bf16x8_vector_buffer(buffer, index, values, value_offset):
    words = TK.alloc_local((4,), "uint32", align=16)
    TK.ptx.ld.global_.v4.b32(
        words[0], words[1], words[2], words[3], buffer.ptr_to([index // ELEMS_PER_LANE])
    )
    with TK.unroll(4) as pair:
        TK.ptx.mov.b32(
            values[value_offset + pair * 2],
            TK.cuda.uint_as_float(TK.shift_left(words[pair], TK.uint32(16))),
        )
        TK.ptx.mov.b32(
            values[value_offset + pair * 2 + 1],
            TK.cuda.uint_as_float(TK.bitwise_and(words[pair], TK.uint32(0xFFFF0000))),
        )


def _load_bf16x8_bits(buffer, index, values):
    words = TK.alloc_local((4,), "uint32", align=16)
    TK.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
    with TK.unroll(4) as pair:
        TK.ptx.mov.b16(
            values[pair * 2], TK.cast(TK.bitwise_and(words[pair], TK.uint32(0xFFFF)), "uint16")
        )
        TK.ptx.mov.b16(
            values[pair * 2 + 1], TK.cast(TK.shift_right(words[pair], TK.uint32(16)), "uint16")
        )


def _load_bf16x4_bits(buffer, index, values):
    words = TK.alloc_local((2,), "uint32", align=8)
    TK.ptx.ld.global_.v2.b32(words[0], words[1], buffer.ptr_to([index]))
    with TK.unroll(2) as pair:
        TK.ptx.mov.b16(
            values[pair * 2], TK.cast(TK.bitwise_and(words[pair], TK.uint32(0xFFFF)), "uint16")
        )
        TK.ptx.mov.b16(
            values[pair * 2 + 1], TK.cast(TK.shift_right(words[pair], TK.uint32(16)), "uint16")
        )


def _store_state_f32x8(buffer, index, values, value_offset):
    words = TK.alloc_local((4,), "uint32", align=16)
    with TK.unroll(4) as pair:
        TK.ptx.mov.b32(
            words[pair],
            TK.cuda.float22bfloat162_rn(
                values[value_offset + pair * 2], values[value_offset + pair * 2 + 1]
            ),
        )
    TK.ptx.st.global_.v4.b32(buffer.ptr_to([index]), words[0], words[1], words[2], words[3])


def _store_state_f32x8_vector_buffer(buffer, index, values, value_offset):
    words = TK.alloc_local((4,), "uint32", align=16)
    with TK.unroll(4) as pair:
        TK.ptx.mov.b32(
            words[pair],
            TK.cuda.float22bfloat162_rn(
                values[value_offset + pair * 2], values[value_offset + pair * 2 + 1]
            ),
        )
    TK.ptx["st.global.L1::evict_first.v4.b32"](
        buffer.ptr_to([index // ELEMS_PER_LANE]), words[0], words[1], words[2], words[3]
    )


def _packed_fma(lhs0, lhs1, rhs0, rhs1, acc0, acc1):
    out = TK.local_scalar("uint64")
    TK.ptx.fma.rn.f32x2(
        out,
        TK.cuda.make_float2(lhs0, lhs1),
        TK.cuda.make_float2(rhs0, rhs1),
        TK.cuda.make_float2(acc0, acc1),
    )
    return out


def _case(label: str, **kwargs: Any) -> dict[str, Any]:
    return {"label": label, **kwargs}


_TP_HEADS = ((16, 32), (8, 16), (4, 8), (2, 4))
_BATCH_SWEEP = (1, 4, 8, 16, 32, 64, 128, 256, 512)


def _production_tile_v(batch: int, num_v_heads: int) -> int | None:
    work_units = batch * num_v_heads
    if work_units >= 1024:
        return 128
    if work_units >= 512:
        return 64
    return None


BENCH_CONFIGS = [
    _case(
        f"b{batch}_h{num_heads}_hv{num_v_heads}_tv{tile_v}",
        batch=batch,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        tile_v=tile_v,
        seed=10000 + head_idx * len(_BATCH_SWEEP) + batch_idx,
    )
    for head_idx, (num_heads, num_v_heads) in enumerate(_TP_HEADS)
    for batch_idx, batch in enumerate(_BATCH_SWEEP)
    if (tile_v := _production_tile_v(batch, num_v_heads)) is not None
]

_CORRECTNESS_MODES = (
    ("l2off", {"use_qk_l2norm": False}),
    ("split", {"same_pool": False}),
    ("split_l2off", {"same_pool": False, "use_qk_l2norm": False}),
    ("padded", {"padded_pool": True}),
    ("padded_split", {"padded_pool": True, "same_pool": False}),
    ("packed_qkv", {"packed_qkv": True}),
    ("negative_read", {"negative_read_index": True}),
    ("negative_write", {"same_pool": False, "negative_write_index": True}),
    ("disable_update", {"disable_state_update": True}),
    ("disable_update_split", {"same_pool": False, "disable_state_update": True}),
    ("cache", {"cache_intermediate_states": True}),
    ("cache_split", {"same_pool": False, "cache_intermediate_states": True}),
)

_BOUNDARY_CASES = ((16, 16, 32, 64), (32, 16, 32, 128))

CONFIGS = [dict(config) for config in BENCH_CONFIGS] + [
    _case(
        f"b{batch}_h{num_heads}_hv{num_v_heads}_tv{tile_v}_{mode}",
        batch=batch,
        num_heads=num_heads,
        num_v_heads=num_v_heads,
        tile_v=tile_v,
        seed=20000 + boundary_idx * len(_CORRECTNESS_MODES) + mode_idx,
        **mode_kwargs,
    )
    for boundary_idx, (batch, num_heads, num_v_heads, tile_v) in enumerate(_BOUNDARY_CASES)
    for mode_idx, (mode, mode_kwargs) in enumerate(_CORRECTNESS_MODES)
]


def _make_gdn_decode_bf16_wide_vec_t1(
    *,
    NUM_HEADS,
    NUM_V_HEADS,
    TILE_V,
    NUM_V_TILES,
    ROWS_PER_GROUP,
    ITERS_PER_GROUP,
    POOL_FACTOR,
    INTERMEDIATE_BATCH_STRIDE,
    INTERMEDIATE_DUMMY_ELEMENTS,
    USE_QK_L2NORM,
    DISABLE_STATE_UPDATE,
    CACHE_INTERMEDIATE_STATES,
    SAME_POOL,
):
    @TK.kernel(
        warps=NUM_WARPS, arch="sm_100a", grid=lambda p: p["batch"] * NUM_V_HEADS * NUM_V_TILES
    )
    def gdn_decode_bf16_wide_vec_t1(
        state: TK.gptr[TK.bf16],
        intermediate: TK.gptr[TK.bf16],
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
        state_slot_stride: TK.i64,
        q_batch_stride: TK.i64,
        k_batch_stride: TK.i64,
        v_batch_stride: TK.i64,
        batch: TK.i32,
    ):
        smem = TK.smem_pool()
        s_q = smem.alloc((K,), TK.f32, align=16)
        s_k = smem.alloc((K,), TK.f32, align=16)
        s_gb = smem.alloc((3,), TK.f32, align=16)
        smem.commit(SHARED_BYTES)
        linear_cta = TK.cta_id()
        tid = TK.thread_id()
        lane_in_warp = _local_scalar("int32", tid % 32)
        warp_raw = _local_scalar("int32", tid // 32)

        state_vector = TK.decl_buffer(
            (state_slot_stride * TK.cast(batch * POOL_FACTOR, "int64") // ELEMS_PER_LANE,),
            "uint32x4",
            data=state.data,
            scope="global",
            align=32,
        )
        group = _local_scalar("int32", tid // LANES_PER_GROUP)
        lane = _local_scalar("int32", tid % LANES_PER_GROUP)
        k_start = _local_scalar("int32", lane[0] * ELEMS_PER_LANE)

        v_tile = _local_scalar("int32", linear_cta % NUM_V_TILES)
        cta_head = _local_scalar("int32", linear_cta // NUM_V_TILES)
        hv = _local_scalar("int32", cta_head[0] % NUM_V_HEADS)
        n = _local_scalar("int32", cta_head[0] // NUM_V_HEADS)
        h = _local_scalar("int32", hv[0] // (NUM_V_HEADS // NUM_HEADS))

        read_slot_raw = TK.local_scalar("int32")
        TK.ptx.ld.global_.s32(read_slot_raw, read_indices.ptr_to([n[0]]))

        _max = TK.local_scalar("int32")
        TK.ptx["max.s32"](_max, read_slot_raw, TK.int32(0))
        read_slot = _local_scalar("int32", _max)
        write_slot = _local_scalar("int32", read_slot[0])
        if not SAME_POOL:
            _ldg32 = TK.local_scalar("int32")
            TK.ptx.ld.global_.s32(_ldg32, write_indices.ptr_to([n[0]]))
            TK.ptx["max.s32"](write_slot[0], _ldg32, TK.int32(0))

        read_state_base = _local_scalar(
            "int64",
            TK.cast(read_slot[0], "int64") * state_slot_stride + TK.cast(hv[0] * V * K, "int64"),
        )
        write_state_base = _local_scalar("int64", read_state_base[0])
        if not SAME_POOL:
            TK.assign(
                write_state_base[0],
                TK.cast(write_slot[0], "int64") * state_slot_stride
                + TK.cast(hv[0] * V * K, "int64"),
            )
        # Phase 0: with the T=1-only kq value dead, independent warps can publish
        # normalized Q, normalized K, and the scalar gates concurrently.
        with TK.If(warp_raw[0] == 0), TK.Then():
            member_pre = _local_scalar("int32", lane_in_warp[0] % LANES_PER_GROUP)
            k_pre = _local_scalar("int32", member_pre[0] * ELEMS_PER_LANE)
            q_bits = TK.alloc_local((ELEMS_PER_LANE,), "uint16")
            r_q = TK.alloc_local((ELEMS_PER_LANE,), "float32")
            q_base = _local_scalar(
                "int64", TK.cast(n[0], "int64") * q_batch_stride + h[0] * K + k_pre[0]
            )
            _load_bf16x8_bits(q, q_base[0], q_bits)
            for i in range(ELEMS_PER_LANE):
                TK.ptx.cvt.f32.bf16(r_q[i], TK.cast(q_bits[i], "uint16"))

            if USE_QK_L2NORM:
                sum_q = _local_scalar("float32", 0.0)
                for i in range(ELEMS_PER_LANE):
                    TK.ptx.fma.rn.f32.bf16(
                        sum_q[0],
                        TK.cast(q_bits[i], "uint16"),
                        TK.cast(q_bits[i], "uint16"),
                        sum_q[0],
                    )
                for delta_index in range(4):
                    delta = _local_scalar("int32", TK.shift_right(TK.int32(8), delta_index))
                    TK.ptx["add.f32"](sum_q[0], sum_q[0], _shfl_bfly_f32(sum_q[0], delta[0]))
                _add = TK.local_scalar("float32")
                TK.ptx["add.f32"](_add, sum_q[0], TK.float32(1e-06))
                _rsqrt = TK.local_scalar("float32")
                TK.ptx["rsqrt.approx.ftz.f32"](_rsqrt, _add)
                _mul = TK.local_scalar("float32")
                TK.ptx["mul.f32"](_mul, _rsqrt, TK.float32(SCALE))
                q_factor = _local_scalar("float32", _mul)
                for i in range(ELEMS_PER_LANE):
                    TK.ptx["mul.f32"](r_q[i], r_q[i], q_factor[0])
            else:
                for i in range(ELEMS_PER_LANE):
                    TK.ptx["mul.f32"](r_q[i], r_q[i], TK.float32(SCALE))

            for i in range(ELEMS_PER_LANE):
                TK.ptx.st.shared.b32(s_q.ptr_to([k_pre[0] + i]), TK.reinterpret("uint32", r_q[i]))

        with TK.If(warp_raw[0] == 1), TK.Then():
            member_pre = _local_scalar("int32", lane_in_warp[0] % LANES_PER_GROUP)
            k_pre = _local_scalar("int32", member_pre[0] * ELEMS_PER_LANE)
            k_bits = TK.alloc_local((ELEMS_PER_LANE,), "uint16")
            r_k = TK.alloc_local((ELEMS_PER_LANE,), "float32")
            k_base = _local_scalar(
                "int64", TK.cast(n[0], "int64") * k_batch_stride + h[0] * K + k_pre[0]
            )
            _load_bf16x8_bits(k, k_base[0], k_bits)
            for i in range(ELEMS_PER_LANE):
                TK.ptx.cvt.f32.bf16(r_k[i], TK.cast(k_bits[i], "uint16"))

            if USE_QK_L2NORM:
                sum_k = _local_scalar("float32", 0.0)
                for i in range(ELEMS_PER_LANE):
                    TK.ptx.fma.rn.f32.bf16(
                        sum_k[0],
                        TK.cast(k_bits[i], "uint16"),
                        TK.cast(k_bits[i], "uint16"),
                        sum_k[0],
                    )
                for delta_index in range(4):
                    delta = _local_scalar("int32", TK.shift_right(TK.int32(8), delta_index))
                    TK.ptx["add.f32"](sum_k[0], sum_k[0], _shfl_bfly_f32(sum_k[0], delta[0]))
                _add2 = TK.local_scalar("float32")
                TK.ptx["add.f32"](_add2, sum_k[0], TK.float32(1e-06))
                _rsqrt2 = TK.local_scalar("float32")
                TK.ptx["rsqrt.approx.ftz.f32"](_rsqrt2, _add2)
                k_factor = _local_scalar("float32", _rsqrt2)
                for i in range(ELEMS_PER_LANE):
                    TK.ptx["mul.f32"](r_k[i], r_k[i], k_factor[0])

            for i in range(ELEMS_PER_LANE):
                TK.ptx.st.shared.b32(s_k.ptr_to([k_pre[0] + i]), TK.reinterpret("uint32", r_k[i]))

        with TK.If(warp_raw[0] == 2), TK.Then():
            A_value = TK.local_scalar("float32")
            TK.ptx.ld.global_.b32(A_value, A_log.ptr_to([hv[0]]))
            dt_value = TK.local_scalar("float32")
            TK.ptx.ld.global_.b32(dt_value, dt_bias.ptr_to([hv[0]]))
            a_bits = TK.local_scalar("uint16")
            TK.ptx.ld.global_.b16(a_bits, a.ptr_to([n[0] * NUM_V_HEADS + hv[0]]))
            _addbf = TK.local_scalar("float32")
            TK.ptx.add.rn.f32.bf16(_addbf, TK.cast(a_bits, "uint16"), dt_value)
            x_value = _local_scalar("float32", _addbf)
            _mul2 = TK.local_scalar("float32")
            TK.ptx["mul.f32"](_mul2, x_value[0], TK.float32(LOG2_E))
            _exp2 = TK.local_scalar("float32")
            TK.ptx["ex2.approx.ftz.f32"](_exp2, _mul2)
            softplus_exp = _local_scalar("float32", _exp2)
            _add3 = TK.local_scalar("float32")
            TK.ptx["add.f32"](_add3, TK.float32(1.0), softplus_exp[0])
            _log2 = TK.local_scalar("float32")
            TK.ptx["lg2.approx.ftz.f32"](_log2, _add3)
            softplus_log2 = _local_scalar("float32", _log2)
            _mul3 = TK.local_scalar("float32")
            TK.ptx["mul.f32"](_mul3, softplus_log2[0], TK.float32(LN_2))
            softplus_value = _local_scalar("float32", _mul3)
            use_softplus = _local_scalar(
                "float32",
                TK.if_then_else(x_value[0] <= TK.float32(20.0), TK.float32(1.0), TK.float32(0.0)),
            )
            _sub = TK.local_scalar("float32")
            TK.ptx["sub.f32"](_sub, TK.float32(1.0), use_softplus[0])
            direct_weight = _local_scalar("float32", _sub)
            _mul4 = TK.local_scalar("float32")
            TK.ptx["mul.f32"](_mul4, x_value[0], direct_weight[0])
            _fma = TK.local_scalar("float32")
            TK.ptx["fma.rn.f32"](_fma, softplus_value[0], use_softplus[0], _mul4)
            softplus_x = _local_scalar("float32", _fma)
            _mul5 = TK.local_scalar("float32")
            TK.ptx["mul.f32"](_mul5, A_value, TK.float32(LOG2_E))
            _exp2_2 = TK.local_scalar("float32")
            TK.ptx["ex2.approx.ftz.f32"](_exp2_2, _mul5)
            exp_A = _local_scalar("float32", _exp2_2)
            _sub2 = TK.local_scalar("float32")
            TK.ptx["sub.f32"](_sub2, TK.float32(0.0), exp_A[0])
            _mul6 = TK.local_scalar("float32")
            TK.ptx["mul.f32"](_mul6, _sub2, softplus_x[0])
            gate_exponent = _local_scalar("float32", _mul6)

            with TK.If(lane_in_warp[0] == 0), TK.Then():
                _mul7 = TK.local_scalar("float32")
                TK.ptx["mul.f32"](_mul7, gate_exponent[0], TK.float32(LOG2_E))
                _exp2_3 = TK.local_scalar("float32")
                TK.ptx["ex2.approx.ftz.f32"](_exp2_3, _mul7)
                g = _local_scalar("float32", _exp2_3)
                TK.ptx.st.shared.b32(s_gb.ptr_to([0]), TK.reinterpret("uint32", g[0]))

        with TK.If(warp_raw[0] == 3), TK.Then():
            with TK.If(lane_in_warp[0] == 0), TK.Then():
                b_bits = TK.local_scalar("uint16")
                TK.ptx.ld.global_.b16(b_bits, b_gate.ptr_to([n[0] * NUM_V_HEADS + hv[0]]))
                b_value = TK.local_scalar("float32")
                TK.ptx.cvt.f32.bf16(b_value, TK.cast(b_bits, "uint16"))
                _mul8 = TK.local_scalar("float32")
                TK.ptx["mul.f32"](_mul8, b_value, TK.float32(-LOG2_E))
                _exp2_4 = TK.local_scalar("float32")
                TK.ptx["ex2.approx.ftz.f32"](_exp2_4, _mul8)
                exp_neg_b = _local_scalar("float32", _exp2_4)
                _add4 = TK.local_scalar("float32")
                TK.ptx["add.f32"](_add4, TK.float32(1.0), exp_neg_b[0])
                _rcp = TK.local_scalar("float32")
                TK.ptx["rcp.rn.f32"](_rcp, _add4)
                beta = _local_scalar("float32", _rcp)
                TK.ptx.st.shared.b32(s_gb.ptr_to([1]), TK.reinterpret("uint32", beta[0]))

        TK.cuda.cta_sync()

        # Phase 1: eight independent 16-lane groups.  Each source constexpr
        # iteration is physically unrolled and carries four state rows in registers.
        # The source compiler retains the same-token Q/K/g/beta shared values across
        # every unrolled V-row body, so materialize that physical register lifetime
        # explicitly: inline PTX shared loads are opaque to nvcc's CSE.
        _lds32 = TK.local_scalar("uint32")
        TK.ptx.ld.shared.b32(_lds32, s_gb.ptr_to([0]))
        g_value = _local_scalar("float32", TK.reinterpret("float32", _lds32))
        _lds32_2 = TK.local_scalar("uint32")
        TK.ptx.ld.shared.b32(_lds32_2, s_gb.ptr_to([1]))
        beta_value = _local_scalar("float32", TK.reinterpret("float32", _lds32_2))
        r_k_main = TK.alloc_local((ELEMS_PER_LANE,), "float32")
        r_q_main = TK.alloc_local((ELEMS_PER_LANE,), "float32")
        for i in range(ELEMS_PER_LANE):
            _lds32_3 = TK.local_scalar("uint32")
            TK.ptx.ld.shared.b32(_lds32_3, s_k.ptr_to([k_start[0] + i]))
            TK.ptx.mov.b32(r_k_main[i], TK.reinterpret("float32", _lds32_3))
            _lds32_4 = TK.local_scalar("uint32")
            TK.ptx.ld.shared.b32(_lds32_4, s_q.ptr_to([k_start[0] + i]))
            TK.ptx.mov.b32(r_q_main[i], TK.reinterpret("float32", _lds32_4))

        for iter_index in range(ITERS_PER_GROUP):
            v_base = _local_scalar(
                "int32", v_tile[0] * TILE_V + group[0] * ROWS_PER_GROUP + iter_index * ILP_ROWS
            )
            read_state_offset = _local_scalar(
                "int64", read_state_base[0] + TK.cast(v_base[0] * K + k_start[0], "int64")
            )
            r_h = TK.alloc_local((ILP_ROWS * ELEMS_PER_LANE,), "float32")
            state_bits = TK.alloc_local((ILP_ROWS * ELEMS_PER_LANE,), "uint16", align=16)
            for row in range(ILP_ROWS):
                if TILE_V == 64:
                    _load_state_bf16x8_vector_buffer(
                        state_vector, read_state_offset[0] + row * K, r_h, row * ELEMS_PER_LANE
                    )
                else:
                    _load_state_bf16x8(
                        state, read_state_offset[0] + row * K, r_h, row * ELEMS_PER_LANE
                    )

            sums = TK.alloc_local((ILP_ROWS,), "float32")
            for row in range(ILP_ROWS):
                TK.ptx.mov.b32(sums[row], TK.float32(0.0))

            for pair in range(ELEMS_PER_LANE // 2):
                for row in range(ILP_ROWS):
                    pair_value = _local_scalar(
                        "uint64",
                        _packed_fma(
                            r_h[row * ELEMS_PER_LANE + pair * 2],
                            r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                            g_value[0],
                            g_value[0],
                            TK.float32(0.0),
                            TK.float32(0.0),
                        ),
                    )
                    TK.ptx.mov.b32(
                        r_h[row * ELEMS_PER_LANE + pair * 2], TK.cuda.float2_x(pair_value[0])
                    )
                    TK.ptx.mov.b32(
                        r_h[row * ELEMS_PER_LANE + pair * 2 + 1], TK.cuda.float2_y(pair_value[0])
                    )
                    TK.ptx["fma.rn.f32"](
                        sums[row],
                        r_h[row * ELEMS_PER_LANE + pair * 2],
                        r_k_main[pair * 2],
                        sums[row],
                    )
                    TK.ptx["fma.rn.f32"](
                        sums[row],
                        r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                        r_k_main[pair * 2 + 1],
                        sums[row],
                    )

            for delta_index in range(4):
                delta = _local_scalar("int32", TK.shift_right(TK.int32(8), delta_index))
                for row in range(ILP_ROWS):
                    TK.ptx["add.f32"](sums[row], sums[row], _shfl_bfly_f32(sums[row], delta[0]))

            values = TK.alloc_local((ILP_ROWS,), "float32")
            value_bits = TK.alloc_local((ILP_ROWS,), "uint16")
            v_input_base = _local_scalar(
                "int64", TK.cast(n[0], "int64") * v_batch_stride + hv[0] * V + v_base[0]
            )
            if TILE_V == 128:
                _load_bf16x4_bits(v, v_input_base[0], value_bits)
            else:
                for row in range(ILP_ROWS):
                    TK.ptx.ld.global_.b16(value_bits[row], v.ptr_to([v_input_base[0] + row]))
            for row in range(ILP_ROWS):
                _subbf = TK.local_scalar("float32")
                TK.ptx.sub.rn.f32.bf16(_subbf, TK.cast(value_bits[row], "uint16"), sums[row])
                TK.ptx["mul.f32"](values[row], _subbf, beta_value[0])

            for row in range(ILP_ROWS):
                TK.ptx.mov.b32(sums[row], TK.float32(0.0))
            for pair in range(ELEMS_PER_LANE // 2):
                q0 = _local_scalar("float32", r_q_main[pair * 2])
                q1 = _local_scalar("float32", r_q_main[pair * 2 + 1])
                for row in range(ILP_ROWS):
                    pair_value = _local_scalar(
                        "uint64",
                        _packed_fma(
                            r_k_main[pair * 2],
                            r_k_main[pair * 2 + 1],
                            values[row],
                            values[row],
                            r_h[row * ELEMS_PER_LANE + pair * 2],
                            r_h[row * ELEMS_PER_LANE + pair * 2 + 1],
                        ),
                    )
                    TK.ptx.mov.b32(
                        r_h[row * ELEMS_PER_LANE + pair * 2], TK.cuda.float2_x(pair_value[0])
                    )
                    TK.ptx.mov.b32(
                        r_h[row * ELEMS_PER_LANE + pair * 2 + 1], TK.cuda.float2_y(pair_value[0])
                    )
                    TK.ptx["fma.rn.f32"](
                        sums[row], r_h[row * ELEMS_PER_LANE + pair * 2], q0[0], sums[row]
                    )
                    TK.ptx["fma.rn.f32"](
                        sums[row], r_h[row * ELEMS_PER_LANE + pair * 2 + 1], q1[0], sums[row]
                    )

            for delta_index in range(4):
                delta = _local_scalar("int32", TK.shift_right(TK.int32(8), delta_index))
                for row in range(ILP_ROWS):
                    TK.ptx["add.f32"](sums[row], sums[row], _shfl_bfly_f32(sums[row], delta[0]))

            with TK.If(lane[0] == 0), TK.Then():
                output_base = _local_scalar(
                    "int64", TK.cast((n[0] * NUM_V_HEADS + hv[0]) * V + v_base[0], "int64")
                )
                for row in range(ILP_ROWS):
                    _bf16 = TK.local_scalar("uint16")
                    TK.ptx.cvt.rn.bf16.f32(_bf16, sums[row])
                    TK.ptx.st.global_.b16(output.ptr_to([output_base[0] + row]), _bf16)

            if CACHE_INTERMEDIATE_STATES:
                intermediate_base = _local_scalar(
                    "int64",
                    TK.cast(
                        (n[0] * NUM_V_HEADS + hv[0]) * V * K + v_base[0] * K + k_start[0], "int64"
                    ),
                )
                for row in range(ILP_ROWS):
                    _store_state_f32x8(
                        intermediate, intermediate_base[0] + row * K, r_h, row * ELEMS_PER_LANE
                    )

            if not DISABLE_STATE_UPDATE and not CACHE_INTERMEDIATE_STATES:
                write_state_offset = _local_scalar(
                    "int64",
                    read_state_offset[0]
                    if SAME_POOL
                    else write_state_base[0] + TK.cast(v_base[0] * K + k_start[0], "int64"),
                )
                for row in range(ILP_ROWS):
                    if TILE_V == 64:
                        _store_state_f32x8_vector_buffer(
                            state_vector, write_state_offset[0] + row * K, r_h, row * ELEMS_PER_LANE
                        )
                    else:
                        _store_state_f32x8(
                            state, write_state_offset[0] + row * K, r_h, row * ELEMS_PER_LANE
                        )

    return gdn_decode_bf16_wide_vec_t1.func


def get_kernel(**kwargs: Any):
    """Return the source-specialized TIRx PrimFunc."""
    tile_v = int(kwargs["tile_v"])
    same_pool = bool(kwargs.get("same_pool", True))
    return _make_gdn_decode_bf16_wide_vec_t1(
        NUM_HEADS=kwargs["num_heads"],
        NUM_V_HEADS=kwargs["num_v_heads"],
        TILE_V=tile_v,
        NUM_V_TILES=V // tile_v,
        ROWS_PER_GROUP=tile_v // NUM_GROUPS,
        ITERS_PER_GROUP=(tile_v // NUM_GROUPS) // ILP_ROWS,
        POOL_FACTOR=1 if same_pool else 2,
        INTERMEDIATE_BATCH_STRIDE=(
            int(kwargs["num_v_heads"]) * V * K
            if kwargs.get("cache_intermediate_states", False)
            else 0
        ),
        INTERMEDIATE_DUMMY_ELEMENTS=(0 if kwargs.get("cache_intermediate_states", False) else 1),
        USE_QK_L2NORM=kwargs.get("use_qk_l2norm", True),
        DISABLE_STATE_UPDATE=kwargs.get("disable_state_update", False),
        CACHE_INTERMEDIATE_STATES=kwargs.get("cache_intermediate_states", False),
        SAME_POOL=same_pool,
    )


def _require_supported_config(config: dict[str, Any]) -> None:
    batch = int(config["batch"])
    num_heads = int(config["num_heads"])
    num_v_heads = int(config["num_v_heads"])
    tile_v = int(config["tile_v"])
    if batch < 1:
        raise ValueError("batch must be positive")
    if (num_heads, num_v_heads) not in _TP_HEADS:
        raise ValueError(f"unsupported Qwen3-Next TP head pair {(num_heads, num_v_heads)}")
    if num_v_heads % num_heads:
        raise ValueError("num_v_heads must be divisible by num_heads")
    if tile_v not in (64, 128) or V % tile_v:
        raise ValueError("wide-vector T1 port requires tile_v in {64, 128}")
    if (tile_v // NUM_GROUPS) % ILP_ROWS:
        raise ValueError("tile_v is incompatible with the 8-group, ILP4 mapping")


def _allocate_pool(
    pool_slots: int,
    num_v_heads: int,
    *,
    padded: bool,
    device: str | torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    padding_heads = 1 if padded else 0
    backing = (
        torch.randn(
            (pool_slots, num_v_heads + padding_heads, V, K),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.1
    )
    return backing[:, :num_v_heads], backing


def _clone_pool_layout(pool: torch.Tensor, *, padded: bool) -> tuple[torch.Tensor, torch.Tensor]:
    pool_slots, num_v_heads = pool.shape[:2]
    padding_heads = 1 if padded else 0
    backing = torch.empty(
        (pool_slots, num_v_heads + padding_heads, V, K), dtype=pool.dtype, device=pool.device
    )
    view = backing[:, :num_v_heads]
    view.copy_(pool)
    return view, backing


def _make_qkv(
    batch: int,
    num_heads: int,
    num_v_heads: int,
    *,
    packed: bool,
    device: str | torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not packed:
        q = (
            torch.randn(
                (batch, SEQ_LEN, num_heads, K),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            * 0.05
        )
        k = torch.randn_like(q)
        v = (
            torch.randn(
                (batch, SEQ_LEN, num_v_heads, V),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            * 0.05
        )
        return q, k, v, None

    q_elements = num_heads * K
    k_elements = num_heads * K
    v_elements = num_v_heads * V
    row_elements = q_elements + k_elements + v_elements + 128
    backing = (
        torch.randn((batch, row_elements), dtype=torch.bfloat16, device=device, generator=generator)
        * 0.05
    )
    q = backing.as_strided((batch, SEQ_LEN, num_heads, K), (row_elements, row_elements, K, 1), 0)
    k = backing.as_strided(
        (batch, SEQ_LEN, num_heads, K), (row_elements, row_elements, K, 1), q_elements
    )
    v = backing.as_strided(
        (batch, SEQ_LEN, num_v_heads, V),
        (row_elements, row_elements, V, 1),
        q_elements + k_elements,
    )
    return q, k, v, backing


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    """Create deterministic independent mutable TIRx and FlashInfer cases."""
    config = dict(kwargs)
    _require_supported_config(config)
    device = config.get("device", "cuda")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise SkipTest("CUDA is required for BF16 wide-vector GDN decode")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] != 10:
        raise SkipTest(f"BF16 wide-vector GDN decode requires SM100, got {capability}")

    batch = int(config["batch"])
    num_heads = int(config["num_heads"])
    num_v_heads = int(config["num_v_heads"])
    same_pool = bool(config.get("same_pool", True))
    padded_pool = bool(config.get("padded_pool", False))
    negative_read = bool(config.get("negative_read_index", False))
    negative_write = bool(config.get("negative_write_index", False))
    packed_qkv = bool(config.get("packed_qkv", False))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config.get("seed", 0)) + 20260811)

    extra_null_slot = negative_read or negative_write
    pool_slots = batch * (1 if same_pool else 2) + int(extra_null_slot)
    initial_pool, initial_backing = _allocate_pool(
        pool_slots, num_v_heads, padded=padded_pool, device=device, generator=generator
    )
    tirx_state, tirx_state_backing = _clone_pool_layout(initial_pool, padded=padded_pool)
    source_state, source_state_backing = _clone_pool_layout(initial_pool, padded=padded_pool)

    slot_offset = 1 if extra_null_slot else 0
    read_indices = torch.arange(batch, dtype=torch.int32, device=device) + slot_offset
    if negative_read:
        read_indices[-1] = -1
    if same_pool:
        write_indices = read_indices
    else:
        write_indices = torch.arange(batch, dtype=torch.int32, device=device) + slot_offset + batch
        if negative_write:
            write_indices[-1] = -1

    q, k, v, qkv_backing = _make_qkv(
        batch, num_heads, num_v_heads, packed=packed_qkv, device=device, generator=generator
    )
    A_log = (
        torch.randn((num_v_heads,), dtype=torch.float32, device=device, generator=generator) * 0.1
    )
    dt_bias = (
        torch.randn((num_v_heads,), dtype=torch.float32, device=device, generator=generator) * 0.1
    )
    a = (
        torch.randn(
            (batch, SEQ_LEN, num_v_heads), dtype=torch.bfloat16, device=device, generator=generator
        )
        * 0.05
    )
    b_gate = torch.randn_like(a) * 0.05

    cache = bool(config.get("cache_intermediate_states", False))
    cache_shape = (batch, SEQ_LEN, num_v_heads, V, K)
    tirx_intermediate = (
        torch.empty(cache_shape, dtype=torch.bfloat16, device=device)
        if cache
        else torch.zeros((1,), dtype=torch.bfloat16, device=device)
    )
    source_intermediate = (
        torch.empty(cache_shape, dtype=torch.bfloat16, device=device)
        if cache
        else torch.zeros((1,), dtype=torch.bfloat16, device=device)
    )

    return {
        "config": config,
        "initial_pool": initial_pool.clone(),
        "initial_backing": initial_backing,
        "tirx_state": tirx_state,
        "tirx_state_backing": tirx_state_backing,
        "source_state": source_state,
        "source_state_backing": source_state_backing,
        "read_indices": read_indices,
        "write_indices": write_indices,
        "q": q,
        "k": k,
        "v": v,
        "qkv_backing": qkv_backing,
        "qkv_snapshot": qkv_backing.clone() if qkv_backing is not None else None,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "a": a,
        "b_gate": b_gate,
        "tirx_output": torch.empty(
            (batch, SEQ_LEN, num_v_heads, V), dtype=torch.bfloat16, device=device
        ),
        "source_output": torch.empty(
            (batch, SEQ_LEN, num_v_heads, V), dtype=torch.bfloat16, device=device
        ),
        "tirx_intermediate": tirx_intermediate,
        "source_intermediate": source_intermediate,
    }


@functools.cache
def _load_oracle():
    import flashinfer.gdn_kernels.gdn_decode_bf16_state as source_module

    return source_module.gated_delta_rule_t1_wide_vec


@functools.cache
def _compile_tirx(
    num_heads: int,
    num_v_heads: int,
    tile_v: int,
    use_qk_l2norm: bool,
    disable_state_update: bool,
    cache_intermediate_states: bool,
    same_pool: bool,
):
    from tirx_kernels.runner import compile_kernel

    return compile_kernel(
        get_kernel(
            batch=1,
            num_heads=num_heads,
            num_v_heads=num_v_heads,
            tile_v=tile_v,
            use_qk_l2norm=use_qk_l2norm,
            disable_state_update=disable_state_update,
            cache_intermediate_states=cache_intermediate_states,
            same_pool=same_pool,
        )
    )


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    config = case["config"]
    state = case["tirx_state"]
    q = case["q"]
    k = case["k"]
    v = case["v"]
    batch = int(config["batch"])
    pool_factor = 1 if bool(config.get("same_pool", True)) else 2

    def storage_span(tensor: torch.Tensor, elements: int) -> torch.Tensor:
        return tensor.as_strided((elements,), (1,))

    return (
        # Negative-index cases own one extra physical null slot, but the kernel
        # ABI is deliberately the production B (or 2B) pool span.  All clamped
        # and shifted indices remain inside that declared span.
        storage_span(state, int(state.stride(0)) * batch * pool_factor),
        storage_span(case["tirx_intermediate"], case["tirx_intermediate"].numel()),
        case["A_log"],
        case["a"].reshape(-1),
        case["dt_bias"],
        storage_span(q, int(q.stride(0)) * (batch - 1) + int(q.shape[2] * q.shape[3])),
        storage_span(k, int(k.stride(0)) * (batch - 1) + int(k.shape[2] * k.shape[3])),
        storage_span(v, int(v.stride(0)) * (batch - 1) + int(v.shape[2] * v.shape[3])),
        case["b_gate"].reshape(-1),
        case["tirx_output"].reshape(-1),
        case["read_indices"],
        case["write_indices"],
        int(state.stride(0)),
        int(q.stride(0)),
        int(k.stride(0)),
        int(v.stride(0)),
        batch,
    )


def _tirx_executable(case: dict[str, Any]):
    config = case["config"]
    return _compile_tirx(
        int(config["num_heads"]),
        int(config["num_v_heads"]),
        int(config["tile_v"]),
        bool(config.get("use_qk_l2norm", True)),
        bool(config.get("disable_state_update", False)),
        bool(config.get("cache_intermediate_states", False)),
        bool(config.get("same_pool", True)),
    )


def _run_reference(case: dict[str, Any]) -> torch.Tensor:
    config = case["config"]
    cache = bool(config.get("cache_intermediate_states", False))
    same_pool = bool(config.get("same_pool", True))
    oracle = _load_oracle()
    result = oracle(
        A_log=case["A_log"],
        a=case["a"],
        dt_bias=case["dt_bias"],
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=case["q"],
        k=case["k"],
        v=case["v"],
        b=case["b_gate"],
        initial_state_source=case["source_state"],
        initial_state_indices=case["read_indices"],
        output_state_indices=(case["read_indices"] if same_pool else case["write_indices"]),
        intermediate_states_buffer=(case["source_intermediate"] if cache else None),
        disable_state_update=bool(config.get("disable_state_update", False)),
        use_qk_l2norm_in_kernel=bool(config.get("use_qk_l2norm", True)),
        scale=SCALE,
        output=case["source_output"],
        tile_v=int(config["tile_v"]),
    )
    return result


def _assert_untouched_slots(case: dict[str, Any], state_key: str) -> None:
    config = case["config"]
    pool = case[state_key]
    initial = case["initial_pool"]
    update_pool = not bool(config.get("disable_state_update", False)) and not bool(
        config.get("cache_intermediate_states", False)
    )
    touched = torch.zeros((pool.shape[0],), dtype=torch.bool, device=pool.device)
    if update_pool:
        indices = (
            case["read_indices"] if bool(config.get("same_pool", True)) else case["write_indices"]
        )
        touched[torch.clamp(indices, min=0).to(torch.int64)] = True
    torch.testing.assert_close(pool[~touched], initial[~touched], atol=0.0, rtol=0.0)


def _assert_case_close(case: dict[str, Any]) -> None:
    torch.testing.assert_close(
        case["tirx_output"].float(), case["source_output"].float(), atol=1.0e-3, rtol=5.0e-3
    )
    torch.testing.assert_close(
        case["tirx_state"].float(), case["source_state"].float(), atol=2.0e-2, rtol=1.0e-2
    )
    _assert_untouched_slots(case, "tirx_state")
    _assert_untouched_slots(case, "source_state")
    if bool(case["config"].get("cache_intermediate_states", False)):
        torch.testing.assert_close(
            case["tirx_intermediate"].float(),
            case["source_intermediate"].float(),
            atol=2.0e-2,
            rtol=1.0e-2,
        )
    if case["qkv_backing"] is not None:
        torch.testing.assert_close(case["qkv_backing"], case["qkv_snapshot"], atol=0.0, rtol=0.0)


def run_test(**kwargs: Any) -> None:
    case = prepare_data(**kwargs)
    executable = _tirx_executable(case)
    executable(*_tirx_args(case))
    torch.cuda.synchronize()
    _run_reference(case)
    torch.cuda.synchronize()
    _assert_case_close(case)


def prepare_bench(**kwargs: Any):
    """Compile the selected wide-vector specialization before CUDA setup."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    config = dict(kwargs)
    _require_supported_config(config)
    executable = _compile_tirx(
        int(config["num_heads"]),
        int(config["num_v_heads"]),
        int(config["tile_v"]),
        bool(config.get("use_qk_l2norm", True)),
        bool(config.get("disable_state_update", False)),
        bool(config.get("cache_intermediate_states", False)),
        bool(config.get("same_pool", True)),
    )
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
        torch.cuda.synchronize()
        _assert_case_close(case)
        for _ in range(2):
            _run_reference(case)
        torch.cuda.synchronize()

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
    "prepare_data",
    "run_bench",
    "run_test",
]
