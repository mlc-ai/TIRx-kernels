# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400e330fb2debe0bf8730d9424a1d37927f), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Plain-TIRx parse-time helpers shared by the parser-DSL norm ports.

This block was shared out of the parser-DSL ``rmsnorm.py`` before its kern
rewrite; ``fused_add_rmsnorm.py``, ``fused_add_rmsnorm_quant.py``, and
``add_rmsnorm_fp4quant.py`` still parse at ``@T.prim_func`` time and need the
parse-time spellings.  Kern-DSL kernels must not import this module — they
spell the same operations through ``tirx_kernels.kern``.
"""

from __future__ import annotations

from tvm.script import tirx as T

_INT32_MAX = 2**31 - 1
_DEFAULT_EPS = 1e-6
_OPTIN_SMEM_BYTES = 232448
_ELEM_BYTES = 2


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _threads_per_row(H_per_cta: int) -> int:
    if H_per_cta <= 64:
        return 8
    if H_per_cta <= 128:
        return 16
    if H_per_cta <= 3072:
        return 32
    if H_per_cta <= 6144:
        return 64
    if H_per_cta <= 16384:
        return 128
    return 256



def _ptx_unary(chain: str, value, dtype: str = "float32"):
    out = T.alloc_local((1,), dtype)
    T.evaluate(T.ptx[chain](out[0], value))
    return out[0]


def _ptx_binary(chain: str, lhs, rhs, dtype: str = "float32"):
    out = T.alloc_local((1,), dtype)
    T.evaluate(T.ptx[chain](out[0], lhs, rhs))
    return out[0]


def _ptx_ternary(chain: str, lhs, rhs, acc, dtype: str = "float32"):
    out = T.alloc_local((1,), dtype)
    T.evaluate(T.ptx[chain](out[0], lhs, rhs, acc))
    return out[0]


def _add_f32(lhs, rhs):
    return _ptx_binary("add.f32", lhs, rhs)


def _div_rn_f32(lhs, rhs):
    return _ptx_binary("div.rn.f32", lhs, rhs)


def _fma_rn_f32(lhs, rhs, acc):
    return _ptx_ternary("fma.rn.f32", lhs, rhs, acc)


def _rsqrt_approx_ftz(value):
    return _ptx_unary("rsqrt.approx.ftz.f32", value)


def _cvt_to_f32(bits, dtype: str):
    if dtype == "float16":
        return _ptx_unary("cvt.f32.f16", T.cast(bits, "uint16"))
    return _ptx_unary("cvt.f32.bf16", T.cast(bits, "uint16"))


def _cvt_from_f32(value, dtype: str):
    if dtype == "float16":
        return _ptx_unary("cvt.rn.f16.f32", value, dtype="uint16")
    return _ptx_unary("cvt.rn.bf16.f32", value, dtype="uint16")


def _cvt_pair_from_f32(high, low, dtype: str):
    if dtype == "float16":
        return _ptx_binary("cvt.rn.f16x2.f32", high, low, dtype="uint32")
    return _ptx_binary("cvt.rn.bf16x2.f32", high, low, dtype="uint32")


def _shfl_bfly_f32(value, lane_xor: int):
    out = T.alloc_local((1,), "uint32")
    T.evaluate(
        T.ptx.shfl_sync.bfly.b32(
            out[0],
            T.reinterpret("uint32", value),
            T.uint32(lane_xor),
            T.uint32(31),
            T.uint32(0xFFFFFFFF),
        )
    )
    return T.reinterpret("float32", out[0])


def _butterfly_sum_f32(value, lane_xors: tuple[int, ...]):
    for lane_xor in lane_xors:
        peer = _shfl_bfly_f32(value, lane_xor)
        value = _add_f32(value, peer)
    return value


def _mapa_u32(pointer, peer):
    mapped = T.alloc_local((1,), "uint32")
    T.evaluate(
        T.ptx.mapa.shared__cluster.u32(
            mapped[0], T.cuda.cvta_generic_to_shared(pointer), T.cast(peer, "uint32")
        )
    )
    return mapped[0]


def _cluster_mbarrier_wait(pointer):
    return T.cuda.mbarrier_wait(pointer, T.int32(0))


@T.inline
def _load_global_bits(buffer, index, values, value_offset, VEC: T.constexpr):
    if VEC == 1:
        T.ptx.ld.global_.b16(values[value_offset], buffer.ptr_to([index]))
    elif VEC == 2:
        T.ptx.ld.global_.v2.b16(
            values[value_offset], values[value_offset + 1], buffer.ptr_to([index])
        )
    else:
        words = T.alloc_local((VEC // 2,), "uint32")
        if VEC == 4:
            T.ptx.ld.global_.v2.b32(words[0], words[1], buffer.ptr_to([index]))
        else:
            T.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
        for pair in T.unroll(VEC // 2):
            values[value_offset + pair * 2] = T.cast(
                T.bitwise_and(words[pair], T.uint32(0xFFFF)), "uint16"
            )
            values[value_offset + pair * 2 + 1] = T.cast(
                T.shift_right(words[pair], T.uint32(16)), "uint16"
            )


@T.inline
def _load_shared_bits(shared_raw, byte_offset, values, value_offset, VEC: T.constexpr):
    if VEC == 2:
        T.ptx.ld.shared.v2.b16(
            values[value_offset], values[value_offset + 1], shared_raw.ptr_to([byte_offset])
        )
    else:
        words = T.alloc_local((VEC // 2,), "uint32")
        if VEC == 4:
            halves = T.alloc_local((4,), "uint16")
            T.ptx.ld.shared.v4.b16(
                halves[0], halves[1], halves[2], halves[3], shared_raw.ptr_to([byte_offset])
            )
            for value in T.unroll(4):
                values[value_offset + value] = halves[value]
        else:
            T.ptx.ld.shared.v4.b32(
                words[0], words[1], words[2], words[3], shared_raw.ptr_to([byte_offset])
            )
            for pair in T.unroll(4):
                values[value_offset + pair * 2] = T.cast(
                    T.bitwise_and(words[pair], T.uint32(0xFFFF)), "uint16"
                )
                values[value_offset + pair * 2 + 1] = T.cast(
                    T.shift_right(words[pair], T.uint32(16)), "uint16"
                )


@T.inline
def _store_global_fragment(
    buffer,
    index,
    bits,
    words,
    predicate,
    value_offset,
    word_offset,
    VEC: T.constexpr,
    PACKED_NARROW: T.constexpr,
):
    if VEC == 1:
        T.ptx.st.global_.b16(buffer.ptr_to([index]), bits[value_offset], pred=predicate)
    elif VEC == 2:
        if PACKED_NARROW:
            bits[value_offset] = T.cast(
                T.bitwise_and(words[word_offset], T.uint32(0xFFFF)), "uint16"
            )
            bits[value_offset + 1] = T.cast(
                T.shift_right(words[word_offset], T.uint32(16)), "uint16"
            )
        T.ptx.st.global_.v2.b16(
            buffer.ptr_to([index]), bits[value_offset], bits[value_offset + 1], pred=predicate
        )
    elif VEC == 4:
        T.ptx.st.global_.v2.b32(
            buffer.ptr_to([index]), words[word_offset], words[word_offset + 1], pred=predicate
        )
    else:
        T.ptx.st.global_.v4.b32(
            buffer.ptr_to([index]),
            words[word_offset],
            words[word_offset + 1],
            words[word_offset + 2],
            words[word_offset + 3],
            pred=predicate,
        )


