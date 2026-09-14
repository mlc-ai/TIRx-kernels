# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400e330fb2debe0bf8730d9424a1d37927f), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""PTX helpers shared by the tirx-lite norm kernel ports."""

import tirx_kernels.tirx_lite as txl

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
    out = txl.local_scalar(dtype)
    txl.ptx[chain](out, value)
    return out


def _ptx_binary(chain: str, lhs, rhs, dtype: str = "float32"):
    out = txl.local_scalar(dtype)
    txl.ptx[chain](out, lhs, rhs)
    return out


def _ptx_ternary(chain: str, lhs, rhs, acc, dtype: str = "float32"):
    out = txl.local_scalar(dtype)
    txl.ptx[chain](out, lhs, rhs, acc)
    return out


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
        return _ptx_unary("cvt.f32.f16", txl.cast(bits, txl.u16))
    return _ptx_unary("cvt.f32.bf16", txl.cast(bits, txl.u16))


def _cvt_from_f32(value, dtype: str):
    if dtype == "float16":
        return _ptx_unary("cvt.rn.f16.f32", value, dtype=txl.u16)
    return _ptx_unary("cvt.rn.bf16.f32", value, dtype=txl.u16)


def _cvt_pair_from_f32(high, low, dtype: str):
    if dtype == "float16":
        return _ptx_binary("cvt.rn.f16x2.f32", high, low, dtype=txl.u32)
    return _ptx_binary("cvt.rn.bf16x2.f32", high, low, dtype=txl.u32)


def _shfl_bfly_f32(value, lane_xor: int):
    out = txl.local_scalar(txl.u32)
    txl.ptx.shfl_sync.bfly.b32(
        out, txl.reinterpret(txl.u32, value), txl.uint32(lane_xor), txl.uint32(31), txl.uint32(0xFFFFFFFF)
    )
    return txl.reinterpret(txl.f32, out)


def _butterfly_sum_f32(value, lane_xors: tuple[int, ...]):
    for lane_xor in lane_xors:
        peer = _shfl_bfly_f32(value, lane_xor)
        value = _add_f32(value, peer)
    return value


def _mapa_u32(pointer, peer):
    mapped = txl.local_scalar(txl.u32)
    txl.ptx.mapa.shared__cluster.u32(
        mapped, txl.cuda.cvta_generic_to_shared(pointer), txl.cast(peer, txl.u32)
    )
    return mapped


def _cluster_mbarrier_wait(pointer):
    txl.cuda.mbarrier_wait(pointer, txl.int32(0))


def _load_global_bits(buffer, index, values, value_offset, VEC: int):
    if VEC == 1:
        txl.ptx.ld.global_.b16(values[value_offset], buffer.ptr_to([index]))
    elif VEC == 2:
        txl.ptx.ld.global_.v2.b16(
            values[value_offset], values[value_offset + 1], buffer.ptr_to([index])
        )
    else:
        words = txl.alloc_local([VEC // 2], txl.u32)
        if VEC == 4:
            txl.ptx.ld.global_.v2.b32(words[0], words[1], buffer.ptr_to([index]))
        else:
            txl.ptx.ld.global_.v4.b32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
        for pair in range(VEC // 2):
            txl.assign(
                values[value_offset + pair * 2],
                txl.cast(txl.bitwise_and(words[pair], txl.uint32(0xFFFF)), txl.u16),
            )
            txl.assign(
                values[value_offset + pair * 2 + 1],
                txl.cast(txl.shift_right(words[pair], txl.uint32(16)), txl.u16),
            )


def _load_shared_bits(shared_raw, byte_offset, values, value_offset, VEC: int):
    if VEC == 2:
        txl.ptx.ld.shared.v2.b16(
            values[value_offset], values[value_offset + 1], shared_raw.ptr_to([byte_offset])
        )
    else:
        words = txl.alloc_local([VEC // 2], txl.u32)
        if VEC == 4:
            halves = txl.alloc_local([4], txl.u16)
            txl.ptx.ld.shared.v4.b16(
                halves[0], halves[1], halves[2], halves[3], shared_raw.ptr_to([byte_offset])
            )
            for value in range(4):
                txl.assign(values[value_offset + value], halves[value])
        else:
            txl.ptx.ld.shared.v4.b32(
                words[0], words[1], words[2], words[3], shared_raw.ptr_to([byte_offset])
            )
            for pair in range(4):
                txl.assign(
                    values[value_offset + pair * 2],
                    txl.cast(txl.bitwise_and(words[pair], txl.uint32(0xFFFF)), txl.u16),
                )
                txl.assign(
                    values[value_offset + pair * 2 + 1],
                    txl.cast(txl.shift_right(words[pair], txl.uint32(16)), txl.u16),
                )


def _store_global_fragment(
    buffer, index, bits, words, predicate, value_offset, word_offset, VEC: int, PACKED_NARROW: bool
):
    if VEC == 1:
        txl.ptx.st.global_.b16(buffer.ptr_to([index]), bits[value_offset], pred=predicate)
    elif VEC == 2:
        if PACKED_NARROW:
            txl.assign(
                bits[value_offset],
                txl.cast(txl.bitwise_and(words[word_offset], txl.uint32(0xFFFF)), txl.u16),
            )
            txl.assign(
                bits[value_offset + 1],
                txl.cast(txl.shift_right(words[word_offset], txl.uint32(16)), txl.u16),
            )
        txl.ptx.st.global_.v2.b16(
            buffer.ptr_to([index]), bits[value_offset], bits[value_offset + 1], pred=predicate
        )
    elif VEC == 4:
        txl.ptx.st.global_.v2.b32(
            buffer.ptr_to([index]), words[word_offset], words[word_offset + 1], pred=predicate
        )
    else:
        txl.ptx.st.global_.v4.b32(
            buffer.ptr_to([index]),
            words[word_offset],
            words[word_offset + 1],
            words[word_offset + 2],
            words[word_offset + 3],
            pred=predicate,
        )
