# Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms,
# Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun,
# Yong Li, Wei Lin.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

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
# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 2497bee508e6337c086f0993f68b41cabce97436), Copyright (c) 2025, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM100/SM103 FlexAttention forward specialized for Dqk=Dv=256.

Upstream source:
``python/cudnn/flex_attention/kernels/sm100/fwd/forward_hd256.py``.

The module consumes the compact arbitrary-mask plan produced by cuDNN
Frontend's FlexAttention planner.  The target implementation preserves the
source kernel's one- and two-CTA execution variants.
"""

import math
import random
from array import array

import tirx_kernels.kern as K

KERNEL_META = {
    "name": "cudnn_sm100_flex_attention_forward_hd256",
    "category": "cudnn",
    "runtime_cuda_archs": ["sm_100a", "sm_103a"],
    "reference_requirements": (
        {
            "package": "nvidia-cudnn-frontend",
            "git": {
                "url": "https://github.com/NVIDIA/cudnn-frontend.git",
                "commit": "2497bee508e6337c086f0993f68b41cabce97436",
            },
            "import": "cudnn",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.7.0", "import": "cutlass"},
    ),
}

_MASKS = (
    "causal",
    "document_causal",
    "local",
    "sink_local",
    "tree_dfs",
    "tree_bfs",
    "longformer",
    "hstu",
)


def _config(
    label,
    *,
    mask="causal",
    dtype="bfloat16",
    cta_group_size=1,
    sequence_mode="fixed",
    num_q_heads=4,
    num_kv_heads=4,
    seqlen=129,
    return_lse=True,
    per_head_mask=False,
    seed=25600,
):
    return {
        "label": label,
        "mask": mask,
        "dtype": dtype,
        "cta_group_size": cta_group_size,
        "sequence_mode": sequence_mode,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "seqlen": seqlen,
        "return_lse": return_lse,
        "per_head_mask": per_head_mask,
        "seed": seed,
    }


CONFIGS = [
    _config("c_bf16_cta1_fixed_mha_causal_lse", seed=25601),
    _config(
        "c_fp16_cta2_fixed_gqa_local_lse",
        mask="local",
        dtype="float16",
        cta_group_size=2,
        num_kv_heads=1,
        seqlen=257,
        seed=25602,
    ),
    _config(
        "c_bf16_cta1_varlen_gqa_longformer_lse",
        mask="longformer",
        sequence_mode="varlen",
        num_kv_heads=1,
        seqlen=387,
        seed=25603,
    ),
    _config(
        "c_fp16_cta2_varlen_mha_hstu_no_lse",
        mask="hstu",
        dtype="float16",
        cta_group_size=2,
        sequence_mode="varlen",
        seqlen=259,
        return_lse=False,
        seed=25604,
    ),
    _config(
        "c_bf16_cta1_fixed_mha_tree_dfs_per_head",
        mask="tree_dfs",
        per_head_mask=True,
        seqlen=128,
        seed=25605,
    ),
    _config(
        "c_bf16_cta2_fixed_mha_document_causal_lse",
        mask="document_causal",
        cta_group_size=2,
        seqlen=8193,
        seed=25606,
    ),
    _config(
        "c_fp16_cta1_fixed_gqa_sink_local_lse",
        mask="sink_local",
        dtype="float16",
        num_kv_heads=1,
        seqlen=1025,
        seed=25607,
    ),
    _config(
        "c_bf16_cta2_varlen_mha_tree_bfs_lse",
        mask="tree_bfs",
        cta_group_size=2,
        sequence_mode="varlen",
        seqlen=383,
        seed=25608,
    ),
    _config(
        "c_fp16_cta2_fixed_gqa_empty_no_lse",
        mask="empty",
        dtype="float16",
        cta_group_size=2,
        num_kv_heads=1,
        seqlen=127,
        return_lse=False,
        seed=25606,
    ),
]


def _benchmark_configs():
    configs = []
    seed = 25700
    for mask in _MASKS:
        for dtype in ("bfloat16", "float16"):
            dtype_label = "bf16" if dtype == "bfloat16" else "fp16"
            for cta_group_size in (1, 2):
                for sequence_mode in ("fixed", "varlen"):
                    for head_mode, num_kv_heads in (("mha", 4), ("gqa4", 1)):
                        seed += 1
                        label = (
                            f"p_{mask}_{dtype_label}_cta{cta_group_size}_"
                            f"{sequence_mode}_{head_mode}_lse"
                        )
                        configs.append(
                            _config(
                                label,
                                mask=mask,
                                dtype=dtype,
                                cta_group_size=cta_group_size,
                                sequence_mode=sequence_mode,
                                num_kv_heads=num_kv_heads,
                                seqlen=131072,
                                seed=seed,
                            )
                        )
    assert len(configs) == 128
    return configs


BENCH_CONFIGS = _benchmark_configs()


_NEG_INF = float("-inf")
_LOG2_E = math.log2(math.e)
_LN2 = math.log(2.0)
_TMA_CACHE = K.uint64(0)
_TMA_CACHE_EVICT_FIRST = K.uint64(0x12F0000000000000)
_TMA_CACHE_EVICT_LAST = K.uint64(0x14F0000000000000)
_TMEM_LD32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
_TMEM_LD16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
_TMEM_ST16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
_TMEM_LD2 = "tcgen05.ld.sync.aligned.32x32b.x2.b32"
_TMEM_ST2 = "tcgen05.st.sync.aligned.32x32b.x2.b32"


def _load_i32(buffer, index):
    value = K.local_scalar("int32")
    K.ptx.ld.global_.s32(value, buffer.ptr_to([index]))
    return value


def _load_u32x4(buffer, index):
    words = K.alloc_local((4,), "uint32")
    K.ptx.ld.global_.v4.u32(words[0], words[1], words[2], words[3], buffer.ptr_to([index]))
    return words


def _ld_shared_f32(buffer, index):
    bits = K.local_scalar("uint32")
    K.ptx.ld.shared.b32(bits, buffer.ptr_to([index]))
    return K.reinterpret("float32", bits)


def _st_shared_f32(buffer, index, value):
    K.ptx.st.shared.b32(buffer.ptr_to([index]), K.reinterpret("uint32", value))


def _exp2(value):
    out = K.local_scalar("float32")
    K.ptx.ex2.approx.ftz.f32(out, value)
    return out


def _log2(value):
    out = K.local_scalar("float32")
    K.ptx.lg2.approx.ftz.f32(out, value)
    return out


def _rcp(value):
    out = K.local_scalar("float32")
    K.ptx.div.rn.f32(out, K.float32(1.0), value)
    return out


def _packed(op, dst, base, a0, a1, b0, b1, c0=None, c1=None):
    lhs = K.local_scalar("uint64")
    rhs = K.local_scalar("uint64")
    result = K.local_scalar("uint64")
    K.ptx.mov.b64(lhs, a0, a1)
    K.ptx.mov.b64(rhs, b0, b1)
    if c0 is None:
        K.ptx[op](result, lhs, rhs)
    else:
        addend = K.local_scalar("uint64")
        K.ptx.mov.b64(addend, c0, c1)
        K.ptx[op](result, lhs, rhs, addend)
    K.ptx.mov.b64(dst[base], dst[base + 1], result)


def _reduce_max_128(values, initial):
    acc = K.alloc_local((4,), "float32")
    K.ptx.max.f32(acc[0], initial, values[0], values[1])
    K.ptx.max.f32(acc[1], values[2], values[3])
    K.ptx.max.f32(acc[2], values[4], values[5])
    K.ptx.max.f32(acc[3], values[6], values[7])
    with K.unroll(1, 16) as group:
        base = group * 8
        K.ptx.max.f32(acc[0], acc[0], values[base], values[base + 1])
        K.ptx.max.f32(acc[1], acc[1], values[base + 2], values[base + 3])
        K.ptx.max.f32(acc[2], acc[2], values[base + 4], values[base + 5])
        K.ptx.max.f32(acc[3], acc[3], values[base + 6], values[base + 7])
    K.ptx.max.f32(acc[0], acc[0], acc[1])
    K.ptx.max.f32(acc[0], acc[0], acc[2], acc[3])
    return acc[0]


def _reduce_max_128_ternary(values, initial):
    """Reduce 129 inputs with four contiguous ternary chains."""
    acc = K.alloc_local((4,), "float32")
    K.assign(acc[0], initial)
    with K.unroll(16) as pair:
        j = pair * 2
        K.ptx.max.f32(acc[0], acc[0], values[j], values[j + 1])
    K.assign(acc[1], values[32])
    with K.unroll(16) as pair:
        j = pair * 2 + 33
        K.ptx.max.f32(acc[1], acc[1], values[j], values[j + 1])
    K.assign(acc[2], values[65])
    with K.unroll(15) as pair:
        j = pair * 2 + 66
        K.ptx.max.f32(acc[2], acc[2], values[j], values[j + 1])
    K.assign(acc[3], values[96])
    with K.unroll(15) as pair:
        j = pair * 2 + 97
        K.ptx.max.f32(acc[3], acc[3], values[j], values[j + 1])
    K.ptx.max.f32(acc[0], acc[0], acc[1], acc[2])
    K.ptx.max.f32(acc[0], acc[0], acc[3], values[127])
    return acc[0]


def _packed_sum_source_128(values, old_sum, old_scale):
    """Match the source's four contiguous 32-value packed quarters."""
    acc = K.alloc_local((8,), "float32")
    carried = K.local_scalar("float32", init=old_sum * old_scale)
    K.assign(acc[0], carried)
    K.assign(acc[1], carried)
    with K.unroll(2, 8) as j:
        K.assign(acc[j], K.float32(0.0))
    with K.unroll(16) as pair:
        j = pair * 2
        with K.unroll(4) as quarter:
            base = quarter * 2
            _packed(
                "add.rn.f32x2",
                acc,
                base,
                acc[base],
                acc[base + 1],
                values[quarter * 32 + j],
                values[quarter * 32 + j + 1],
            )
    for lo, hi in ((0, 2), (4, 6), (0, 4)):
        _packed("add.rn.f32x2", acc, lo, acc[lo], acc[lo + 1], acc[hi], acc[hi + 1])
    return acc[0] + acc[1]


def _wait(barrier, stage, phase):
    K.cuda.mbarrier_wait(barrier.ptr_to([stage]), phase)


def _tmem_load32(dst, address):
    K.ptx[_TMEM_LD32](*(dst[i] for i in range(32)), K.cast(address, "uint32"))


def _tmem_load2(dst, address):
    K.ptx[_TMEM_LD2](dst[0], dst[1], K.cast(address, "uint32"))


def _tmem_store2(src, address):
    K.ptx[_TMEM_ST2](K.cast(address, "uint32"), src[0], src[1])


def _tmem_load16(dst, address):
    K.ptx[_TMEM_LD16](*(dst[i] for i in range(16)), K.cast(address, "uint32"))


def _tmem_store16(src, address):
    K.ptx[_TMEM_ST16](K.cast(address, "uint32"), *(src[i] for i in range(16)))


def _make_kernel(**config):
    hq = int(config["num_q_heads"])
    hkv = int(config["num_kv_heads"])
    total_q = int(config["seqlen"])
    total_k = total_q
    dtype = config["dtype"]
    cta_group = int(config["cta_group_size"])
    varlen = config["sequence_mode"] == "varlen"
    has_lse = bool(config["return_lse"])
    per_head_mask = bool(config["per_head_mask"])
    analytic_causal_cta2_fixed = (
        config["mask"] == "causal"
        and cta_group == 2
        and not varlen
        and hq == 4
        and dtype in ("bfloat16", "float16")
        and hkv in (1, 4)
    )
    analytic_causal_cta1_fixed = (
        config["mask"] == "causal"
        and dtype in ("bfloat16", "float16")
        and cta_group == 1
        and not varlen
        and hq == 4
        and hkv in (1, 4)
    )
    contiguous_full_hstu_fixed = (
        config["mask"] == "hstu"
        and not varlen
        and hq == 4
        and total_q == 131072
        and (
            (dtype == "bfloat16" and hkv == 1)
            or (cta_group == 1 and dtype in ("bfloat16", "float16") and hkv == 4)
        )
    )
    batch = int(config.get("batch", 3 if varlen and total_q < 4096 else 1))
    if dtype not in ("bfloat16", "float16"):
        raise ValueError("dtype must be bfloat16 or float16")
    if cta_group not in (1, 2):
        raise ValueError("cta_group_size must be 1 or 2")
    if hq <= 0 or hkv <= 0 or hq % hkv:
        raise ValueError("num_q_heads must be a positive multiple of num_kv_heads")
    if total_q <= 0:
        raise ValueError("seqlen must be positive")

    ratio = hq // hkv
    elem_dtype = K.bf16 if dtype == "bfloat16" else K.f16
    elem_dtype_name = dtype
    hmask = hq if per_head_mask else 1
    if varlen and batch > 1:
        q_lengths = [total_q // batch] * batch
        for i in range(total_q % batch):
            q_lengths[i] += 1
        if batch >= 3 and min(q_lengths) > 2:
            q_lengths[0] -= 1
            q_lengths[-1] += 1
    else:
        q_lengths = [total_q] * batch
    logical_tile = 128 * cta_group
    causal_dense_plan = analytic_causal_cta2_fixed and all(
        length % logical_tile == 0 for length in q_lengths
    )
    causal_dense_cta1 = analytic_causal_cta1_fixed and all(
        length % logical_tile == 0 for length in q_lengths
    )
    total_m_blocks = sum((length + logical_tile - 1) // logical_tile for length in q_lengths)
    task_count = total_m_blocks * hq
    smem_bytes = {(1, False): 230272, (1, True): 197504, (2, False): 164736, (2, True): 131968}[
        (cta_group, varlen)
    ]
    q_stage_bytes = 32768
    kv_stage_bytes = 32768 if cta_group == 1 else 16384
    sq_offset = 384
    sk_offset = 65920
    sv_offset = 131456 if cta_group == 1 else 98688
    so_offset = (196992 if cta_group == 1 else 131456) if not varlen else -1
    sum_offset = (
        (229760 if cta_group == 1 else 164224)
        if not varlen
        else (196992 if cta_group == 1 else 131456)
    )
    qk_idesc = {
        (1, "bfloat16"): 0x08200490,
        (1, "float16"): 0x08200010,
        (2, "bfloat16"): 0x10200490,
        (2, "float16"): 0x10200010,
    }[(cta_group, dtype)]
    pv_idesc = {
        (1, "bfloat16"): 0x08210490,
        (1, "float16"): 0x08210010,
        (2, "bfloat16"): 0x10210490,
        (2, "float16"): 0x10210010,
    }[(cta_group, dtype)]
    tma_g2s_4d = (
        "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
        if cta_group == 1
        else "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
    )
    tma_g2s_5d = (
        "cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
        if cta_group == 1
        else "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
    )
    tma_s2g_5d = "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
    q_tma_cache = (
        _TMA_CACHE_EVICT_FIRST
        if hkv == 1 and (causal_dense_plan or causal_dense_cta1)
        else _TMA_CACHE
    )
    kv_tma_cache = _TMA_CACHE_EVICT_LAST if hkv == 1 else _TMA_CACHE
    mma_f16 = f"tcgen05.mma.cta_group::{cta_group}.kind::f16"
    tmem_alloc = f"tcgen05.alloc.cta_group::{cta_group}.sync.aligned.shared::cta.b32"
    tmem_dealloc = f"tcgen05.dealloc.cta_group::{cta_group}.sync.aligned.b32"
    tmem_relinquish = f"tcgen05.relinquish_alloc_permit.cta_group::{cta_group}.sync.aligned"

    def host_prelude(params):
        def encode(tensor, dims, strides, box):
            descriptor = K.stack_alloca("tensormap", 1)
            K.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                descriptor,
                elem_dtype_name,
                len(dims),
                tensor.data,
                *dims,
                *strides,
                *box,
                *((1,) * len(dims)),
                0,
                3,
                2,
                0,
            )
            return descriptor

        q = params["q"]
        k = params["k"]
        v = params["v"]
        out = params["out"]
        map_q_tokens = total_q
        map_k_tokens = total_k
        q_batch_stride = 0 if varlen else total_q * hq * 512
        kv_batch_stride = 0 if varlen else total_k * hkv * 512
        qo_dims = (256, ratio, map_q_tokens, hkv, batch)
        qo_strides = (512, hq * 512, ratio * 512, q_batch_stride)
        # Tensor-map boxes describe each CTA's share.  CTA-group::2 combines
        # both shares in the completion transaction: Q remains M128 per CTA,
        # while K contributes N64 per CTA.  V contributes D128 per CTA.
        qo_box = (64, 1, 128, 1, 1)
        kv_dims = (256, map_k_tokens, hkv, batch)
        kv_strides = (hkv * 512, 512, kv_batch_stride)
        k_box = (64, 128 // cta_group, 1, 1)
        v_box = (64, 128, 1, 1)
        maps = [
            encode(q, qo_dims, qo_strides, qo_box),
            encode(k, kv_dims, kv_strides, k_box),
            encode(v, kv_dims, kv_strides, v_box),
        ]
        if not varlen:
            maps.append(encode(out, qo_dims, qo_strides, (64, 1, 128, 1, 1)))
        return tuple(maps)

    def kernel_body(
        q,
        k,
        v,
        out,
        lse,
        cu_q,
        cu_k,
        sequence_desc,
        mask_block_cnt,
        mask_block_offset,
        mask_block_idx,
        full_block_cnt,
        full_block_offset,
        full_block_idx,
        mask_payload,
        fwd_work_desc,
        softmax_scale_log2,
        softmax_scale,
        host,
    ):
        del q, k, v
        if varlen:
            q_map, k_map, v_map = host
            o_map = None
        else:
            q_map, k_map, v_map, o_map = host

        if cta_group == 1:
            cta_rank = K.int32(0)
        else:
            cta_rank = K.cta_id_in_cluster([2], preferred=[2])
        block_x = K.cta_id()
        task = K.local_scalar("int32", init=block_x // cta_group)
        work_valid = K.local_scalar("int32", init=1)
        clc_consumer_phase = K.local_scalar("int32", init=0)
        logical_m = K.local_scalar("int32", init=0)
        head = K.local_scalar("int32", init=0)
        batch_idx = K.local_scalar("int32", init=0)
        q_valid = K.local_scalar("int32", init=0)
        q_begin = K.local_scalar("int32", init=0)
        k_begin = K.local_scalar("int32", init=0)
        partial_count = K.local_scalar("int32", init=0)
        full_count = K.local_scalar("int32", init=0)
        partial_base = K.local_scalar("int32", init=0)
        full_base = K.local_scalar("int32", init=0)

        def load_work_metadata():
            K.assign(logical_m, _load_i32(fwd_work_desc, task * 4))
            K.assign(head, _load_i32(fwd_work_desc, task * 4 + 1))
            K.assign(batch_idx, _load_i32(fwd_work_desc, task * 4 + 2))
            K.assign(q_valid, _load_i32(fwd_work_desc, task * 4 + 3))
            K.assign(q_begin, _load_i32(sequence_desc, batch_idx * 8))
            K.assign(k_begin, _load_i32(sequence_desc, batch_idx * 8 + 1))
            plan_outer = _load_i32(sequence_desc, batch_idx * 8 + 4) + logical_m
            plan_head = head if hmask != 1 else K.int32(0)
            plan_row = plan_head * total_m_blocks + plan_outer
            if causal_dense_plan:
                K.assign(partial_count, cta_group)
                K.assign(full_count, logical_m * cta_group)
                K.assign(partial_base, plan_row * cta_group)
            else:
                K.assign(partial_count, _load_i32(mask_block_cnt, plan_row))
                K.assign(full_count, _load_i32(full_block_cnt, plan_row))
                K.assign(partial_base, _load_i32(mask_block_offset, plan_row))
                K.assign(full_base, _load_i32(full_block_offset, plan_row))

        physical_m = logical_m * cta_group + cta_rank
        loop_blocks = K.max(partial_count + full_count, 1)
        kv_head = head // ratio
        head_in_kv = head - kv_head * ratio
        warp = K.warp_id()
        lane = K.lane_id()
        tid = K.thread_id()
        tid128 = tid & 127
        is_leader_cta = cta_rank == 0

        with K.If(warp == 9), K.Then():
            K.ptx.prefetch.tensormap(K.address_of(q_map))
            K.ptx.prefetch.tensormap(K.address_of(k_map))
        with K.If(warp == 11), K.Then():
            K.ptx.prefetch.tensormap(K.address_of(v_map))
        if not varlen:
            with K.If(warp == 4), K.Then():
                K.ptx.prefetch.tensormap(K.address_of(o_map))

        arena = K.alloc_buffer((smem_bytes,), K.u8, scope="shared.dyn", align=1024)
        smem = K.smem_pool(base=arena)
        pool = smem.pool
        q_full = K.TMABar(pool, 2)
        q_empty = K.TCGen05Bar(pool, 2)
        k_full = K.TMABar(pool, 2)
        k_empty = K.TCGen05Bar(pool, 2)
        v_full = K.TMABar(pool, 2)
        v_empty = K.TCGen05Bar(pool, 2)
        score_full = K.TCGen05Bar(pool, 2)
        score_empty = K.MBarrier(pool, 2)
        p_full = K.MBarrier(pool, 2)
        p_empty = K.TCGen05Bar(pool, 2)
        stats_full = K.MBarrier(pool, 2)
        stats_empty = K.MBarrier(pool, 2)
        sum_full = K.MBarrier(pool, 1)
        sum_empty = K.MBarrier(pool, 1)
        o_full = K.TCGen05Bar(pool, 1)
        o_empty = K.MBarrier(pool, 1)
        tmem_dealloc_bar = K.MBarrier(pool, 1, leader=(tid == 8 * 32))
        tmem_mailbox = pool.alloc((1,), "uint32", align=8)
        clc_full = K.TMABar(pool, 1)
        clc_empty = K.MBarrier(pool, 1)
        clc_response = pool.alloc((4,), "uint32", align=16)
        if pool.offset > sq_offset:
            raise AssertionError("shared protocol prefix exceeded source sQ offset")
        pool.alloc((sq_offset - pool.offset,), "uint8")
        pool.alloc((smem_bytes - pool.offset,), "uint8")
        if pool.offset != smem_bytes:
            raise AssertionError("shared arena size changed")

        q_smem = K.decl_buffer(
            (2 * q_stage_bytes // 2,),
            elem_dtype_name,
            data=arena.data,
            byte_offset=sq_offset,
            scope="shared.dyn",
            align=1024,
        )
        k_smem = K.decl_buffer(
            (2 * kv_stage_bytes // 2,),
            elem_dtype_name,
            data=arena.data,
            byte_offset=sk_offset,
            scope="shared.dyn",
            align=1024,
        )
        v_smem = K.decl_buffer(
            (2 * kv_stage_bytes // 2,),
            elem_dtype_name,
            data=arena.data,
            byte_offset=sv_offset,
            scope="shared.dyn",
            align=1024,
        )
        sum_smem = K.decl_buffer(
            (128,), "float32", data=arena.data, byte_offset=sum_offset, scope="shared.dyn", align=16
        )
        if not varlen:
            o_smem = K.decl_buffer(
                (128 * 256,),
                elem_dtype_name,
                data=arena.data,
                byte_offset=so_offset,
                scope="shared.dyn",
                align=1024,
            )

        q_full.init(1)
        q_empty.init(1)
        k_full.init(1)
        k_empty.init(1)
        v_full.init(1)
        v_empty.init(1)
        score_full.init(1)
        score_empty.init(128 * cta_group)
        p_full.init(128 * cta_group)
        p_empty.init(1)
        stats_full.init(128)
        stats_empty.init(128)
        sum_full.init(128)
        sum_empty.init(128)
        o_full.init(1)
        o_empty.init(128 * cta_group)
        if cta_group == 2:
            tmem_dealloc_bar.init(32)
        clc_full.init(1)
        clc_empty.init(384 * cta_group)
        K.ptx.fence.mbarrier_init.release.cluster()
        if cta_group == 2:
            K.ptx.barrier.cluster.arrive.relaxed()
        # The source scheduler warp returns its registers before the pipeline
        # init wait.  Delaying this transition can deadlock WG0's 240-register
        # acquisition in the two-CTA launch.
        with K.If(warp == 10), K.Then():
            K.ptx.setmaxnreg.dec.sync.aligned.u32(72)
        if cta_group == 2:
            K.ptx.barrier.cluster.wait()
        else:
            K.cuda.cta_sync()

        def advance2(stage, phase):
            K.assign(stage, stage + 1)
            with K.If(stage == 2), K.Then():
                K.assign(stage, 0)
                K.assign(phase, phase ^ 1)

        def tma_full_ptr(barrier, stage):
            local = K.cuda.cvta_generic_to_shared(barrier.ptr_to([stage]))
            if cta_group == 1:
                return local
            mapped = K.local_scalar("uint32")
            K.ptx.mapa.shared__cluster.u32(mapped, local, K.uint32(0))
            return mapped

        def tma_g2s_dst(pointer):
            if cta_group == 1:
                return pointer
            cluster_pointer = K.local_scalar("uint64")
            K.ptx.cvta.to.shared__cluster.u64(cluster_pointer, pointer)
            return K.cast(cluster_pointer, "uint32")

        def advance_work():
            _wait(clc_full, 0, clc_consumer_phase)
            payload = K.local_scalar("uint128")
            payload_lo = K.local_scalar("uint64")
            payload_hi = K.local_scalar("uint64")
            canceled = K.local_scalar("uint32")
            next_cta = K.local_scalar("uint32", init=K.uint32(0xFFFFFFFF))
            K.ptx.ld.shared.v2.b64(payload_lo, payload_hi, K.address_of(clc_response[0]))
            K.ptx.mov.b128(payload, payload_lo, payload_hi)
            K.ptx.clusterlaunchcontrol.query_cancel.is_canceled.pred.b128(canceled, payload)
            K.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__x.b32.b128(
                next_cta, payload, pred=canceled
            )
            K.assign(work_valid, K.cast(canceled, "int32"))
            with K.If(canceled != 0), K.Then():
                K.assign(task, K.cast(next_cta // K.uint32(cta_group), "int32"))
            K.ptx.fence.proxy.async_.shared__cta()
            clc_empty.arrive(0, remote=0, pred=K.bool(True), count=1)
            K.assign(clc_consumer_phase, clc_consumer_phase ^ 1)

        def selected_nblock(step):
            if causal_dense_plan:
                return K.if_then_else(
                    step < cta_group, logical_m * cta_group + step, step - cta_group
                )
            elif causal_dense_cta1:
                result = K.local_scalar("int32", init=K.int32(0))
                with K.If(partial_count + full_count > 0), K.Then():
                    with K.If(step < partial_count):
                        with K.Then():
                            K.assign(result, logical_m * cta_group + step)
                        with K.Else():
                            K.assign(result, step - partial_count)
                return result
            else:
                result = K.local_scalar("int32", init=K.int32(0))
                with K.If(partial_count + full_count > 0), K.Then():
                    with K.If(step < partial_count):
                        with K.Then():
                            K.assign(result, _load_i32(mask_block_idx, partial_base + step))
                        with K.Else():
                            K.assign(
                                result, _load_i32(full_block_idx, full_base + step - partial_count)
                            )
                return result

        def issue_mma(dst, a, b, idesc, accumulate):
            with K.If(K.cuda.elect_sync()), K.Then():
                if cta_group == 1:
                    K.ptx[mma_f16](
                        K.cast(dst, "uint32"),
                        a,
                        b,
                        K.uint32(idesc),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.cast(accumulate, "bool"),
                    )
                else:
                    K.ptx[mma_f16](
                        K.cast(dst, "uint32"),
                        a,
                        b,
                        K.uint32(idesc),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.cast(accumulate, "bool"),
                    )

        def commit_tcgen(bar, stage):
            with K.If(K.cuda.elect_sync()), K.Then():
                if cta_group == 1:
                    bar.arrive(stage, cta_group=1)
                else:
                    bar.arrive(stage, cta_group=2, cta_mask=K.uint16(3))

        roles = K.specialize(chain_dispatch=True)
        r_softmax = roles.role("softmax", warps=range(0, 4), regs=240)
        r_correction = roles.role("correction", warps=range(4, 8), regs=128)
        r_mma = roles.role("mma", warps=[8], regs=72)
        r_qk_load = roles.role("qk_load", warps=[9], regs=72)
        r_scheduler = roles.role("scheduler", warps=[10], regs=72)
        r_v_load = roles.role("v_load", warps=[11], regs=72)

        with r_qk_load:
            q_stage = K.local_scalar("int32", init=0)
            q_phase = K.local_scalar("int32", init=1)
            k_stage = K.local_scalar("int32", init=0)
            k_phase = K.local_scalar("int32", init=1)
            with K.While(work_valid != 0):
                load_work_metadata()
                for qk_iter in range(2):
                    _wait(q_empty, q_stage, q_phase)
                    leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                    with K.If((leader != 0) & is_leader_cta), K.Then():
                        q_full.arrive(q_stage, tx_count=32768 if cta_group == 1 else 65536)
                    for band in range(2):
                        q_token = q_begin + physical_m * 128 if varlen else physical_m * 128
                        K.ptx[tma_g2s_5d](
                            tma_g2s_dst(
                                q_smem.ptr_to([q_stage * (q_stage_bytes // 2) + band * 128 * 64])
                            ),
                            K.address_of(q_map),
                            K.int32(qk_iter * 128 + band * 64),
                            head_in_kv,
                            q_token,
                            kv_head,
                            batch_idx,
                            tma_full_ptr(q_full, q_stage),
                            q_tma_cache,
                            pred=leader,
                        )
                    advance2(q_stage, q_phase)

                def load_k_block(nblock):
                    for qk_iter in range(2):
                        _wait(k_empty, k_stage, k_phase)
                        leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                        with K.If((leader != 0) & is_leader_cta), K.Then():
                            k_full.arrive(k_stage, tx_count=32768)
                        for band in range(2):
                            local_band_elems = 128 * 64 if cta_group == 1 else 64 * 64
                            K.ptx[tma_g2s_4d](
                                tma_g2s_dst(
                                    k_smem.ptr_to(
                                        [k_stage * (kv_stage_bytes // 2) + band * local_band_elems]
                                    )
                                ),
                                K.address_of(k_map),
                                K.int32(qk_iter * 128 + band * 64),
                                k_begin + nblock * 128 + cta_rank * 64,
                                kv_head,
                                batch_idx,
                                tma_full_ptr(k_full, k_stage),
                                kv_tma_cache,
                                pred=leader,
                            )
                        advance2(k_stage, k_phase)

                step = K.local_scalar("int32", init=0)
                if causal_dense_plan and (hkv == 1 or dtype == "float16"):
                    with K.While(step < cta_group):
                        load_k_block(logical_m * cta_group + step)
                        K.assign(step, step + 1)
                    K.assign(step, 0)
                    with K.While(step < full_count):
                        load_k_block(step)
                        K.assign(step, step + 1)
                elif causal_dense_cta1:
                    with K.While(step < cta_group):
                        load_k_block(logical_m * cta_group + step)
                        K.assign(step, step + 1)
                    K.assign(step, 0)
                    with K.While(step < full_count):
                        load_k_block(step)
                        K.assign(step, step + 1)
                elif contiguous_full_hstu_fixed:
                    full_start = K.local_scalar("int32", init=0)
                    with K.If(full_count > 0), K.Then():
                        K.assign(full_start, _load_i32(full_block_idx, full_base))
                    with K.While(step < partial_count):
                        load_k_block(_load_i32(mask_block_idx, partial_base + step))
                        K.assign(step, step + 1)
                    K.assign(step, 0)
                    with K.While(step < full_count):
                        load_k_block(full_start + step)
                        K.assign(step, step + 1)
                else:
                    with K.While(step < loop_blocks):
                        load_k_block(selected_nblock(step))
                        K.assign(step, step + 1)
                advance_work()
            _wait(k_empty, k_stage, k_phase)
            _wait(q_empty, q_stage, q_phase)

        with r_v_load:
            v_stage = K.local_scalar("int32", init=0)
            v_phase = K.local_scalar("int32", init=1)
            with K.While(work_valid != 0):
                load_work_metadata()

                def load_v_block(nblock):
                    for dv_iter in range(2):
                        _wait(v_empty, v_stage, v_phase)
                        leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                        with K.If((leader != 0) & is_leader_cta), K.Then():
                            v_full.arrive(v_stage, tx_count=32768)
                        if cta_group == 1:
                            for band in range(2):
                                K.ptx[tma_g2s_4d](
                                    tma_g2s_dst(
                                        v_smem.ptr_to(
                                            [v_stage * (kv_stage_bytes // 2) + band * 128 * 64]
                                        )
                                    ),
                                    K.address_of(v_map),
                                    K.int32(dv_iter * 128 + band * 64),
                                    k_begin + nblock * 128,
                                    kv_head,
                                    batch_idx,
                                    tma_full_ptr(v_full, v_stage),
                                    kv_tma_cache,
                                    pred=leader,
                                )
                        else:
                            K.ptx[tma_g2s_4d](
                                tma_g2s_dst(v_smem.ptr_to([v_stage * (kv_stage_bytes // 2)])),
                                K.address_of(v_map),
                                K.int32(dv_iter * 128) + cta_rank * 64,
                                k_begin + nblock * 128,
                                kv_head,
                                batch_idx,
                                tma_full_ptr(v_full, v_stage),
                                kv_tma_cache,
                                pred=leader,
                            )
                        advance2(v_stage, v_phase)

                step = K.local_scalar("int32", init=0)
                if causal_dense_plan and (hkv == 1 or dtype == "float16"):
                    with K.While(step < cta_group):
                        load_v_block(logical_m * cta_group + step)
                        K.assign(step, step + 1)
                    K.assign(step, 0)
                    with K.While(step < full_count):
                        load_v_block(step)
                        K.assign(step, step + 1)
                elif causal_dense_cta1:
                    with K.While(step < cta_group):
                        load_v_block(logical_m * cta_group + step)
                        K.assign(step, step + 1)
                    K.assign(step, 0)
                    with K.While(step < full_count):
                        load_v_block(step)
                        K.assign(step, step + 1)
                elif contiguous_full_hstu_fixed:
                    full_start = K.local_scalar("int32", init=0)
                    with K.If(full_count > 0), K.Then():
                        K.assign(full_start, _load_i32(full_block_idx, full_base))
                    with K.While(step < partial_count):
                        load_v_block(_load_i32(mask_block_idx, partial_base + step))
                        K.assign(step, step + 1)
                    K.assign(step, 0)
                    with K.While(step < full_count):
                        load_v_block(full_start + step)
                        K.assign(step, step + 1)
                else:
                    with K.While(step < loop_blocks):
                        load_v_block(selected_nblock(step))
                        K.assign(step, step + 1)
                advance_work()
            _wait(v_empty, v_stage, v_phase)

        with r_mma:
            K.ptx[tmem_alloc](K.address_of(tmem_mailbox[0]), K.uint32(512))
            K.ptx.bar.sync(K.uint32(1), K.uint32(288))
            tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            q_stage = K.local_scalar("int32", init=0)
            q_phase = K.local_scalar("int32", init=0)
            k_stage = K.local_scalar("int32", init=0)
            k_phase = K.local_scalar("int32", init=0)
            v_stage = K.local_scalar("int32", init=0)
            v_phase = K.local_scalar("int32", init=0)
            score_stage = K.local_scalar("int32", init=0)
            score_phase = K.local_scalar("int32", init=1)
            p_stage = K.local_scalar("int32", init=0)
            p_phase = K.local_scalar("int32", init=0)
            o_phase = K.local_scalar("int32", init=1)
            accumulated = K.local_scalar("int32", init=0)
            step = K.local_scalar("int32", init=0)
            with K.If(is_leader_cta), K.Then():
                with K.While(work_valid != 0):
                    load_work_metadata()
                    K.assign(accumulated, 0)
                    K.assign(step, 0)
                    with K.While(step < loop_blocks):
                        _wait(score_empty, score_stage, score_phase)
                        for qk_iter in range(2):
                            K.assign(q_stage, qk_iter)
                            _wait(q_full, q_stage, q_phase)
                            _wait(k_full, k_stage, k_phase)
                            q_desc = K.SmemDescriptor()
                            q_desc.init(
                                q_smem.ptr_to([q_stage * (q_stage_bytes // 2)]),
                                ldo=1024,
                                sdo=64,
                                swizzle=3,
                            )
                            q_desc.make_lo_uniform()
                            k_desc = K.SmemDescriptor()
                            k_desc.init(
                                k_smem.ptr_to([k_stage * (kv_stage_bytes // 2)]),
                                ldo=1024,
                                sdo=64,
                                swizzle=3,
                            )
                            k_desc.make_lo_uniform()
                            for kphase in range(8):
                                qk_offset = (kphase & 3) * 2 + (kphase // 4) * 1024
                                k_qk_offset = (kphase & 3) * 2 + (kphase // 4) * (1024 // cta_group)
                                issue_mma(
                                    tmem_base + score_stage * 128,
                                    q_desc.add_16B_offset(qk_offset),
                                    k_desc.add_16B_offset(k_qk_offset),
                                    qk_idesc,
                                    (qk_iter != 0) or (kphase != 0),
                                )
                            commit_tcgen(k_empty, k_stage)
                            advance2(k_stage, k_phase)
                        commit_tcgen(score_full, score_stage)
                        advance2(score_stage, score_phase)

                        with K.If(step != 0), K.Then():
                            _wait(p_full, p_stage, p_phase)
                            _wait(o_empty, 0, o_phase)
                            for dv_iter in range(2):
                                _wait(v_full, v_stage, v_phase)
                                v_desc = K.SmemDescriptor()
                                v_desc.init(
                                    v_smem.ptr_to([v_stage * (kv_stage_bytes // 2)]),
                                    ldo=1024,
                                    sdo=64,
                                    swizzle=3,
                                )
                                v_desc.make_lo_uniform()
                                for kphase in range(8):
                                    issue_mma(
                                        tmem_base + 256 + dv_iter * 128,
                                        tmem_base + p_stage * 128 + kphase * 8,
                                        v_desc.add_16B_offset(kphase * 128),
                                        pv_idesc,
                                        (accumulated != 0) | (kphase != 0),
                                    )
                                commit_tcgen(v_empty, v_stage)
                                advance2(v_stage, v_phase)
                            commit_tcgen(o_full, 0)
                            K.assign(o_phase, o_phase ^ 1)
                            commit_tcgen(p_empty, p_stage)
                            advance2(p_stage, p_phase)
                            K.assign(accumulated, 1)
                        K.assign(step, step + 1)

                    commit_tcgen(q_empty, 0)
                    commit_tcgen(q_empty, 1)
                    _wait(p_full, p_stage, p_phase)
                    _wait(o_empty, 0, o_phase)
                    for dv_iter in range(2):
                        _wait(v_full, v_stage, v_phase)
                        v_desc = K.SmemDescriptor()
                        v_desc.init(
                            v_smem.ptr_to([v_stage * (kv_stage_bytes // 2)]),
                            ldo=1024,
                            sdo=64,
                            swizzle=3,
                        )
                        v_desc.make_lo_uniform()
                        for kphase in range(8):
                            issue_mma(
                                tmem_base + 256 + dv_iter * 128,
                                tmem_base + p_stage * 128 + kphase * 8,
                                v_desc.add_16B_offset(kphase * 128),
                                pv_idesc,
                                (accumulated != 0) | (kphase != 0),
                            )
                        commit_tcgen(v_empty, v_stage)
                        advance2(v_stage, v_phase)
                    commit_tcgen(o_full, 0)
                    K.assign(o_phase, o_phase ^ 1)
                    commit_tcgen(p_empty, p_stage)
                    advance2(p_stage, p_phase)
                    K.assign(q_phase, q_phase ^ 1)
                    advance_work()
                _wait(score_empty, score_stage, score_phase)
                _wait(o_empty, 0, o_phase)
            with K.If(cta_rank != 0), K.Then():
                with K.While(work_valid != 0):
                    advance_work()

            K.ptx[tmem_relinquish]()
            K.ptx.bar.sync(K.uint32(1), K.uint32(288))
            if cta_group == 2:
                tmem_dealloc_bar.arrive(0, remote=1 - cta_rank)
                _wait(tmem_dealloc_bar, 0, 0)
            K.ptx[tmem_dealloc](tmem_base, K.uint32(512))

        with r_softmax:
            K.ptx.bar.sync(K.uint32(1), K.uint32(288))
            tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            row_hi = K.shift_left(K.cast(tid128, "uint32"), K.uint32(16))
            score_stage = K.local_scalar("int32", init=0)
            score_phase = K.local_scalar("int32", init=0)
            p_stage = K.local_scalar("int32", init=0)
            p_phase = K.local_scalar("int32", init=1)
            stats_stage = K.local_scalar("int32", init=0)
            stats_phase = K.local_scalar("int32", init=1)
            sum_phase = K.local_scalar("int32", init=1)
            row_max = K.local_scalar("float32", init=K.float32(_NEG_INF))
            row_sum = K.local_scalar("float32", init=K.float32(0.0))
            step = K.local_scalar("int32", init=0)

            def softmax_step(mask_mode, step):
                _wait(score_full, score_stage, score_phase)
                score = K.alloc_local((128,), "float32")
                for fragment in range(4):
                    regs = K.alloc_local((32,), "float32")
                    _tmem_load32(regs, tmem_base + score_stage * 128 + fragment * 32 + row_hi)
                    with K.unroll(32) as j:
                        K.assign(score[fragment * 32 + j], regs[j])
                K.ptx.tcgen05.wait__ld.sync.aligned()
                score_empty.arrive(score_stage, remote=0, pred=K.bool(True))

                if mask_mode == "empty":
                    with K.unroll(128) as j:
                        K.assign(score[j], K.float32(_NEG_INF))
                elif mask_mode == "partial":
                    if causal_dense_cta1:
                        with K.unroll(128) as j:
                            K.assign(
                                score[j], K.if_then_else(j <= tid128, score[j], K.float32(_NEG_INF))
                            )
                    elif causal_dense_plan:
                        with K.If(step == cta_rank), K.Then():
                            with K.unroll(128) as j:
                                K.assign(
                                    score[j],
                                    K.if_then_else(j <= tid128, score[j], K.float32(_NEG_INF)),
                                )
                        with K.If(step > cta_rank), K.Then():
                            with K.unroll(128) as j:
                                K.assign(score[j], K.float32(_NEG_INF))
                    else:
                        payload_linear = (
                            (partial_base + step) * cta_group + cta_rank
                        ) * 128 * 4 + tid128 * 4
                        words = _load_u32x4(mask_payload, payload_linear)
                        with K.unroll(128) as j:
                            keep = (
                                K.bitwise_and(
                                    K.shift_right(words[j // 32], K.uint32(j & 31)), K.uint32(1)
                                )
                                != 0
                            )
                            K.assign(score[j], K.if_then_else(keep, score[j], K.float32(_NEG_INF)))

                old_max = K.local_scalar("float32", init=row_max)
                if hkv == 1 and (causal_dense_plan or causal_dense_cta1):
                    K.assign(row_max, _reduce_max_128_ternary(score, row_max))
                else:
                    K.assign(row_max, _reduce_max_128(score, row_max))
                safe_max = K.local_scalar(
                    "float32",
                    init=K.if_then_else(row_max != K.float32(_NEG_INF), row_max, K.float32(0.0)),
                )
                _wait(stats_empty, stats_stage, stats_phase)
                stats_values = K.alloc_local((2,), "float32")
                K.assign(stats_values[0], old_max)
                K.assign(stats_values[1], safe_max)
                _tmem_store2(stats_values, tmem_base + score_stage * 128 + 64 + row_hi)
                K.ptx.tcgen05.wait__st.sync.aligned()
                stats_full.arrive(stats_stage)
                advance2(stats_stage, stats_phase)

                _wait(p_empty, p_stage, p_phase)
                negative_max_scale = K.local_scalar("float32", init=-safe_max * softmax_scale_log2)
                with K.unroll(64) as pair:
                    base = pair * 2
                    _packed(
                        "fma.rn.f32x2",
                        score,
                        base,
                        score[base],
                        score[base + 1],
                        softmax_scale_log2,
                        softmax_scale_log2,
                        negative_max_scale,
                        negative_max_scale,
                    )
                with K.unroll(128) as j:
                    K.assign(score[j], _exp2(score[j]))
                packed_p = K.alloc_local((64,), "uint32")
                with K.unroll(64) as pair:
                    base = pair * 2
                    if dtype == "bfloat16":
                        K.ptx.cvt.rn.bf16x2.f32(packed_p[pair], score[base + 1], score[base])
                    else:
                        K.ptx.cvt.rn.f16x2.f32(packed_p[pair], score[base + 1], score[base])
                K.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
                    K.cast(tmem_base + p_stage * 128 + row_hi, "uint32"),
                    *(packed_p[i] for i in range(32)),
                )
                K.ptx["tcgen05.st.sync.aligned.32x32b.x32.b32"](
                    K.cast(tmem_base + p_stage * 128 + 32 + row_hi, "uint32"),
                    *(packed_p[32 + i] for i in range(32)),
                )
                K.ptx.tcgen05.wait__st.sync.aligned()
                p_full.arrive(p_stage, remote=0, pred=K.bool(True))
                advance2(p_stage, p_phase)
                acc_scale = K.local_scalar(
                    "float32",
                    init=_exp2(softmax_scale_log2 * (old_max - safe_max)) * K.float32(0.5),
                )
                K.assign(row_sum, _packed_sum_source_128(score, row_sum, acc_scale))
                advance2(score_stage, score_phase)

            with K.While(work_valid != 0):
                load_work_metadata()
                K.assign(row_max, K.float32(_NEG_INF))
                K.assign(row_sum, K.float32(0.0))
                K.assign(step, 0)
                with K.If(partial_count + full_count == 0), K.Then():
                    softmax_step("empty", K.int32(0))
                with K.If(partial_count + full_count != 0), K.Then():
                    K.assign(step, 0)
                    with K.While(step < partial_count):
                        softmax_step("partial", step)
                        K.assign(step, step + 1)
                    K.assign(step, partial_count)
                    with K.While(step < partial_count + full_count):
                        softmax_step("full", step)
                        K.assign(step, step + 1)
                _wait(sum_empty, 0, sum_phase)
                _st_shared_f32(sum_smem, tid128, row_sum)
                K.ptx.fence.proxy.async_.shared__cta()
                sum_full.arrive(0)
                if has_lse:
                    valid_row = cta_rank * 128 + tid128 < q_valid
                    with K.If(valid_row), K.Then():
                        value = K.local_scalar("float32", init=K.float32(_NEG_INF))
                        with K.If((row_sum != 0.0) & (row_sum == row_sum)), K.Then():
                            K.assign(
                                value, softmax_scale * row_max + _log2(row_sum) * K.float32(_LN2)
                            )
                        token = (
                            q_begin + physical_m * 128 + tid128
                            if varlen
                            else physical_m * 128 + tid128
                        )
                        lse_index = (
                            head * total_q + token
                            if varlen
                            else (batch_idx * hq + head) * total_q + token
                        )
                        K.ptx.st.global_.f32(lse.ptr_to([lse_index]), value)
                K.assign(sum_phase, sum_phase ^ 1)
                advance_work()
            _wait(p_empty, p_stage, p_phase)
            _wait(stats_empty, stats_stage, stats_phase)
            K.ptx.bar.arrive(K.uint32(1), K.uint32(288))

        with r_correction:
            K.ptx.bar.sync(K.uint32(1), K.uint32(288))
            tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            row_hi = K.shift_left(K.cast(tid128, "uint32"), K.uint32(16))
            stats_stage = K.local_scalar("int32", init=0)
            stats_phase = K.local_scalar("int32", init=0)
            o_phase = K.local_scalar("int32", init=0)
            sum_phase = K.local_scalar("int32", init=0)
            with K.While(work_valid != 0):
                load_work_metadata()
                _wait(stats_full, stats_stage, stats_phase)
                stats_empty.arrive(stats_stage)
                advance2(stats_stage, stats_phase)
                step = K.local_scalar("int32", init=1)
                with K.While(step < loop_blocks):
                    _wait(stats_full, stats_stage, stats_phase)
                    stats_values = K.alloc_local((2,), "float32")
                    _tmem_load2(stats_values, tmem_base + stats_stage * 128 + 64 + row_hi)
                    K.ptx.tcgen05.wait__ld.sync.aligned()
                    scale = K.local_scalar(
                        "float32",
                        init=_exp2(softmax_scale_log2 * (stats_values[0] - stats_values[1])),
                    )
                    stats_empty.arrive(stats_stage)
                    advance2(stats_stage, stats_phase)
                    _wait(o_full, 0, o_phase)
                    if causal_dense_plan or (causal_dense_cta1 and hkv == 1):
                        values0 = K.alloc_local((16,), "float32")
                        values1 = K.alloc_local((16,), "float32")
                        _tmem_load16(values0, tmem_base + 256 + row_hi)
                        for flat_fragment in range(1, 16):
                            current_values = values1 if flat_fragment & 1 else values0
                            previous_values = values0 if flat_fragment & 1 else values1
                            _tmem_load16(
                                current_values, tmem_base + 256 + flat_fragment * 16 + row_hi
                            )
                            with K.unroll(8) as pair:
                                base = pair * 2
                                _packed(
                                    "mul.rn.f32x2",
                                    previous_values,
                                    base,
                                    previous_values[base],
                                    previous_values[base + 1],
                                    scale,
                                    scale,
                                )
                            _tmem_store16(
                                previous_values, tmem_base + 256 + (flat_fragment - 1) * 16 + row_hi
                            )
                        with K.unroll(8) as pair:
                            base = pair * 2
                            _packed(
                                "mul.rn.f32x2",
                                values1,
                                base,
                                values1[base],
                                values1[base + 1],
                                scale,
                                scale,
                            )
                        _tmem_store16(values1, tmem_base + 256 + 15 * 16 + row_hi)
                    else:
                        for dv_iter in range(2):
                            for fragment in range(8):
                                values = K.alloc_local((16,), "float32")
                                address = tmem_base + 256 + dv_iter * 128 + fragment * 16 + row_hi
                                _tmem_load16(values, address)
                                with K.unroll(8) as pair:
                                    base = pair * 2
                                    _packed(
                                        "mul.rn.f32x2",
                                        values,
                                        base,
                                        values[base],
                                        values[base + 1],
                                        scale,
                                        scale,
                                    )
                                _tmem_store16(values, address)
                    K.ptx.tcgen05.wait__st.sync.aligned()
                    o_empty.arrive(0, remote=0, pred=K.bool(True))
                    K.assign(o_phase, o_phase ^ 1)
                    K.assign(step, step + 1)

                _wait(sum_full, 0, sum_phase)
                row_sum = _ld_shared_f32(sum_smem, tid128)
                K.ptx.fence.proxy.async_.shared__cta()
                sum_empty.arrive(0)
                valid_sum = (row_sum != 0.0) & (row_sum == row_sum)
                out_scale = K.local_scalar(
                    "float32", init=K.if_then_else(valid_sum, _rcp(row_sum), K.float32(0.0))
                )
                _wait(o_full, 0, o_phase)
                if not varlen:
                    for dv_iter in range(2):
                        for fragment in range(4):
                            values = K.alloc_local((32,), "float32")
                            _tmem_load32(
                                values, tmem_base + 256 + dv_iter * 128 + fragment * 32 + row_hi
                            )
                            with K.unroll(16) as pair:
                                base = pair * 2
                                _packed(
                                    "mul.rn.f32x2",
                                    values,
                                    base,
                                    values[base],
                                    values[base + 1],
                                    out_scale,
                                    out_scale,
                                )
                            packed_o = K.alloc_local((16,), "uint32")
                            with K.unroll(16) as pair:
                                base = pair * 2
                                if dtype == "bfloat16":
                                    K.ptx.cvt.rn.bf16x2.f32(
                                        packed_o[pair], values[base + 1], values[base]
                                    )
                                else:
                                    K.ptx.cvt.rn.f16x2.f32(
                                        packed_o[pair], values[base + 1], values[base]
                                    )
                            with K.unroll(4) as vec:
                                d0 = dv_iter * 128 + fragment * 32 + vec * 8
                                local_d = d0 - dv_iter * 128
                                band = local_d // 64
                                minor = local_d - band * 64
                                raw_byte = (
                                    so_offset + band * 128 * 64 * 2 + tid128 * 128 + minor * 2
                                )
                                swizzled = raw_byte ^ ((raw_byte >> 3) & 0x70)
                                K.ptx.st.shared.v4.u32(
                                    o_smem.ptr_to([(swizzled - so_offset) // 2]),
                                    packed_o[vec * 4],
                                    packed_o[vec * 4 + 1],
                                    packed_o[vec * 4 + 2],
                                    packed_o[vec * 4 + 3],
                                )
                        K.ptx.tcgen05.wait__ld.sync.aligned()
                        K.ptx.fence.proxy.async_.shared__cta()
                        K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                        with K.If(warp == 4), K.Then():
                            leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                            for band in range(2):
                                K.ptx[tma_s2g_5d](
                                    K.address_of(o_map),
                                    K.int32(dv_iter * 128 + band * 64),
                                    head_in_kv,
                                    physical_m * 128,
                                    kv_head,
                                    batch_idx,
                                    o_smem.ptr_to([band * 128 * 64]),
                                    _TMA_CACHE,
                                    pred=leader,
                                )
                            K.ptx.cp.async_.bulk.commit_group(pred=leader)
                            K.ptx.cp.async_.bulk.wait_group.read(0)
                        K.ptx.bar.sync(K.uint32(2), K.uint32(128))
                else:
                    for dv_iter in range(2):
                        for fragment in range(4):
                            values = K.alloc_local((32,), "float32")
                            _tmem_load32(
                                values, tmem_base + 256 + dv_iter * 128 + fragment * 32 + row_hi
                            )
                            with K.unroll(16) as pair:
                                base = pair * 2
                                _packed(
                                    "mul.rn.f32x2",
                                    values,
                                    base,
                                    values[base],
                                    values[base + 1],
                                    out_scale,
                                    out_scale,
                                )
                            packed_o = K.alloc_local((16,), "uint32")
                            with K.unroll(16) as pair:
                                base = pair * 2
                                if dtype == "bfloat16":
                                    K.ptx.cvt.rn.bf16x2.f32(
                                        packed_o[pair], values[base + 1], values[base]
                                    )
                                else:
                                    K.ptx.cvt.rn.f16x2.f32(
                                        packed_o[pair], values[base + 1], values[base]
                                    )
                            valid_row = cta_rank * 128 + tid128 < q_valid
                            token = q_begin + physical_m * 128 + tid128
                            with K.unroll(16) as pair:
                                d0 = dv_iter * 128 + fragment * 32 + pair * 2
                                out_base = (token * hq + head) * 256 + d0
                                K.ptx.st.global_.b32(
                                    out.ptr_to([out_base]), packed_o[pair], pred=valid_row
                                )
                        K.ptx.tcgen05.wait__ld.sync.aligned()
                o_empty.arrive(0, remote=0, pred=K.bool(True))
                K.assign(o_phase, o_phase ^ 1)
                K.assign(sum_phase, sum_phase ^ 1)
                advance_work()
            if not varlen:
                with K.If(warp == 4), K.Then():
                    K.ptx.cp.async_.bulk.wait_group.read(0)
            K.ptx.bar.arrive(K.uint32(1), K.uint32(288))

        with r_scheduler:
            clc_producer_phase = K.local_scalar("int32", init=1)
            with K.If(is_leader_cta), K.Then():
                with K.While(work_valid != 0):
                    _wait(clc_empty, 0, clc_producer_phase)
                    with K.If(lane < cta_group), K.Then():
                        clc_full.arrive(0, tx_count=16, remote=lane)
                    with K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx[
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        ](K.address_of(clc_response[0]), K.address_of(clc_full.buf[0]))
                    K.assign(clc_producer_phase, clc_producer_phase ^ 1)
                    advance_work()
                _wait(clc_empty, 0, clc_producer_phase)
            with K.If(cta_rank != 0), K.Then():
                with K.While(work_valid != 0):
                    advance_work()

    def kernel(
        q,
        k,
        v,
        out,
        lse,
        cu_q,
        cu_k,
        sequence_desc,
        mask_block_cnt,
        mask_block_offset,
        mask_block_idx,
        full_block_cnt,
        full_block_offset,
        full_block_idx,
        mask_payload,
        fwd_work_desc,
        softmax_scale_log2,
        softmax_scale,
        *,
        host,
    ):
        if cta_group == 2:
            with K.attr({"tirx.required_block_size": 1}):
                kernel_body(
                    q,
                    k,
                    v,
                    out,
                    lse,
                    cu_q,
                    cu_k,
                    sequence_desc,
                    mask_block_cnt,
                    mask_block_offset,
                    mask_block_idx,
                    full_block_cnt,
                    full_block_offset,
                    full_block_idx,
                    mask_payload,
                    fwd_work_desc,
                    softmax_scale_log2,
                    softmax_scale,
                    host,
                )
        else:
            kernel_body(
                q,
                k,
                v,
                out,
                lse,
                cu_q,
                cu_k,
                sequence_desc,
                mask_block_cnt,
                mask_block_offset,
                mask_block_idx,
                full_block_cnt,
                full_block_offset,
                full_block_idx,
                mask_payload,
                fwd_work_desc,
                softmax_scale_log2,
                softmax_scale,
                host,
            )

    kernel.__annotations__ = {
        "q": K.gptr[elem_dtype],
        "k": K.gptr[elem_dtype],
        "v": K.gptr[elem_dtype],
        "out": K.gptr[elem_dtype],
        "lse": K.gptr[K.f32],
        "cu_q": K.gptr[K.i32],
        "cu_k": K.gptr[K.i32],
        "sequence_desc": K.gptr[K.i32],
        "mask_block_cnt": K.gptr[K.i32],
        "mask_block_offset": K.gptr[K.i32],
        "mask_block_idx": K.gptr[K.i32],
        "full_block_cnt": K.gptr[K.i32],
        "full_block_offset": K.gptr[K.i32],
        "full_block_idx": K.gptr[K.i32],
        "mask_payload": K.gptr[K.u32],
        "fwd_work_desc": K.gptr[K.i32],
        "softmax_scale_log2": K.f32,
        "softmax_scale": K.f32,
    }
    return K.kernel(
        warps=12,
        arch="sm_100a",
        grid=[task_count * cta_group],
        min_blocks_per_sm=1,
        host_prelude=host_prelude,
    )(kernel)


def get_kernel(**config):
    return _make_kernel(**config).func


def _torch_dtype(torch, name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


_GUARD_ELEMS = 64
_OUTPUT_GUARD = 42.5
_LSE_GUARD = -54321.25
_LSE_SENTINEL = 12345.25


def _guarded_tensor(torch, shape, dtype, *, fill, guard):
    elements = math.prod(shape)
    storage = torch.full((elements + 2 * _GUARD_ELEMS,), guard, dtype=dtype, device="cuda")
    view = storage[_GUARD_ELEMS : _GUARD_ELEMS + elements].view(shape)
    view.fill_(fill)
    return view, storage


def _lengths_for_config(config):
    total = int(config["seqlen"])
    varlen = config["sequence_mode"] == "varlen"
    batch = int(config.get("batch", 3 if varlen and total < 4096 else 1))
    if not varlen:
        return [total] * batch
    lengths = [total // batch] * batch
    for i in range(total % batch):
        lengths[i] += 1
    if batch >= 3 and min(lengths) > 2:
        lengths[0] -= 1
        lengths[-1] += 1
    return lengths


def _jittered_lengths(total, count, *, seed, jitter):
    if count <= 0 or total < count:
        raise ValueError("length count must be positive and no greater than total")
    generator = random.Random(seed)
    weights = [generator.uniform(1.0 - jitter, 1.0 + jitter) for _ in range(count)]
    scaled = [weight * total / sum(weights) for weight in weights]
    lengths = [max(1, math.floor(value)) for value in scaled]
    difference = total - sum(lengths)
    fractional_order = sorted(
        range(count), key=lambda idx: scaled[idx] - math.floor(scaled[idx]), reverse=True
    )
    for index in range(difference):
        lengths[fractional_order[index % count]] += 1
    if sum(lengths) != total:
        raise AssertionError("failed to normalize jittered lengths")
    return lengths


def _bounded_partition(total, minimum, maximum, *, seed, prefer_multiple=False):
    if total < minimum:
        return [total]
    min_parts = math.ceil(total / maximum)
    max_parts = total // minimum
    count = round(total / ((minimum + maximum) / 2))
    count = min(max(count, min_parts), max_parts)
    if prefer_multiple and count == 1 and max_parts >= 2:
        count = 2
    generator = random.Random(seed)
    lengths = [generator.randint(minimum, maximum) for _ in range(count)]
    difference = total - sum(lengths)
    while difference:
        order = list(range(count))
        generator.shuffle(order)
        progressed = False
        for index in order:
            if difference > 0:
                delta = min(difference, maximum - lengths[index])
            else:
                delta = -min(-difference, lengths[index] - minimum)
            if delta:
                lengths[index] += delta
                difference -= delta
                progressed = True
            if difference == 0:
                break
        if not progressed:
            raise AssertionError("bounded partition cannot satisfy the requested total")
    return lengths


def _merge_intervals(intervals):
    merged = []
    for begin, end in sorted(intervals):
        if not 0 <= begin <= end:
            raise ValueError(f"invalid interval [{begin}, {end})")
        if begin == end:
            continue
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    return [(begin, end) for begin, end in merged]


def _encode_intervals(intervals, *, max_intervals, starts_at_zero):
    merged = _merge_intervals(intervals)
    if not merged or len(merged) > max_intervals:
        raise ValueError("interval union does not fit the selected nfunc")
    if starts_at_zero:
        if merged[0][0] != 0:
            raise ValueError("the first interval must start at zero")
        encoded = [merged[0][1]]
        encoded.extend(value for pair in merged[1:] for value in pair)
        expected = 2 * max_intervals - 1
    else:
        encoded = [0]
        encoded.extend(value for pair in merged for value in pair)
        expected = 2 * max_intervals + 1
    encoded.extend([encoded[-1]] * (expected - len(encoded)))
    return encoded


def _tree_order(depth, traversal):
    node_count = 2**depth - 1
    if traversal == "bfs":
        return list(range(node_count))
    order = []

    def visit(node):
        if node >= node_count:
            return
        order.append(node)
        visit(2 * node + 1)
        visit(2 * node + 2)

    visit(0)
    return order


def _tree_ancestors(node):
    result = []
    while True:
        result.append(node)
        if node == 0:
            break
        node = (node - 1) // 2
    return list(reversed(result))


def _make_single_mask_endpoints(torch, mask, seqlen):
    if mask == "causal":
        return torch.arange(1, seqlen + 1, dtype=torch.int32).view(1, -1)
    if mask == "empty":
        return torch.zeros((1, seqlen), dtype=torch.int32)
    if mask == "document_causal":
        standard_lengths = (
            11858,
            8270,
            12765,
            11038,
            4578,
            14018,
            11721,
            11988,
            4942,
            8393,
            7541,
            13495,
            10465,
        )
        lengths = (
            list(standard_lengths)
            if seqlen == 128 * 1024
            else _bounded_partition(seqlen, 4096, 16384, seed=42, prefer_multiple=True)
        )
        endpoints = torch.empty((3, seqlen), dtype=torch.int32)
        offset = 0
        for length in lengths:
            end = offset + length
            endpoints[0, offset:end] = 0
            endpoints[1, offset:end] = offset
            endpoints[2, offset:end] = torch.arange(offset + 1, end + 1, dtype=torch.int32)
            offset = end
        return endpoints
    if mask == "local":
        q_idx = torch.arange(seqlen, dtype=torch.int32)
        end = q_idx + 1
        begin = torch.clamp(q_idx - 512, min=0)
        return torch.stack((torch.zeros_like(begin), begin, end))
    if mask == "sink_local":
        q_idx = torch.arange(seqlen, dtype=torch.int32)
        end = q_idx + 1
        sink_end = torch.minimum(end, torch.scalar_tensor(4, dtype=torch.int32))
        local_begin = torch.clamp(q_idx - 512, min=4)
        local_begin = torch.minimum(torch.maximum(local_begin, sink_end), end)
        return torch.stack((sink_end, local_begin, end))
    if mask in ("tree_dfs", "tree_bfs"):
        traversal = mask.removeprefix("tree_")
        depth = 7
        node_count = 2**depth - 1
        lengths = _jittered_lengths(seqlen, node_count, seed=42, jitter=0.2)
        order = _tree_order(depth, traversal)
        starts = {}
        offset = 0
        for node in order:
            starts[node] = offset
            offset += lengths[node]
        endpoints = torch.empty((2 * depth - 1, seqlen), dtype=torch.int32)
        for node in order:
            begin = starts[node]
            end = begin + lengths[node]
            row_end = torch.arange(begin + 1, end + 1, dtype=torch.int32)
            ancestors = [(starts[a], starts[a] + lengths[a]) for a in _tree_ancestors(node)[:-1]]
            template = _merge_intervals((*ancestors, (begin, end)))
            encoded = _encode_intervals(template, max_intervals=depth, starts_at_zero=True)
            current_end_slot = 2 * len(template) - 2
            for slot, value in enumerate(encoded):
                endpoints[slot, begin:end] = row_end if slot >= current_end_slot else value
        return endpoints
    if mask == "longformer":
        radius = 256
        global_tokens = [int((index + 0.5) * seqlen / 8) for index in range(8)]
        global_tokens = sorted({min(token, seqlen - 1) for token in global_tokens})
        flat = array("i")
        global_set = set(global_tokens)
        for q_idx in range(seqlen):
            if q_idx in global_set:
                intervals = [(0, seqlen)]
            else:
                intervals = [(max(0, q_idx - radius), min(seqlen, q_idx + radius + 1))]
                intervals.extend((token, token + 1) for token in global_tokens)
            flat.extend(_encode_intervals(intervals, max_intervals=9, starts_at_zero=False))
        return torch.tensor(flat, dtype=torch.int32).view(seqlen, 19).t().contiguous()
    if mask == "hstu":
        contexts = _bounded_partition(seqlen // 2, 4096, 8192, seed=42)
        targets = contexts.copy()
        targets[-1] += seqlen % 2
        endpoints = torch.empty((5, seqlen), dtype=torch.int32)
        offset = 0
        for context, target in zip(contexts, targets):
            context_end = offset + context
            document_end = context_end + target
            context_rows = torch.arange(offset + 1, context_end + 1, dtype=torch.int32)
            endpoints[0, offset:context_end] = 0
            endpoints[1, offset:context_end] = offset
            endpoints[2, offset:context_end] = context_rows
            endpoints[3, offset:context_end] = context_rows
            endpoints[4, offset:context_end] = context_rows
            target_rows = torch.arange(context_end, document_end, dtype=torch.int32)
            endpoints[0, context_end:document_end] = 0
            endpoints[1, context_end:document_end] = offset
            endpoints[2, context_end:document_end] = context_end
            endpoints[3, context_end:document_end] = target_rows
            endpoints[4, context_end:document_end] = target_rows + 1
            offset = document_end
        return endpoints
    raise ValueError(f"unknown mask {mask!r}")


def _make_mask_endpoints(torch, mask, lengths):
    pieces = [_make_single_mask_endpoints(torch, mask, length) for length in lengths]
    nfunc = pieces[0].shape[0]
    if any(piece.shape[0] != nfunc for piece in pieces):
        raise AssertionError("ragged mask pieces disagree on nfunc")
    return torch.cat(pieces, dim=1).contiguous()


def prepare_data(**config):
    import torch
    from cudnn.flex_attention import create_mask_plan

    dtype = _torch_dtype(torch, config["dtype"])
    hq = int(config["num_q_heads"])
    hkv = int(config["num_kv_heads"])
    total = int(config["seqlen"])
    varlen = config["sequence_mode"] == "varlen"
    cta_group = int(config["cta_group_size"])
    lengths = _lengths_for_config(config)
    batch = len(lengths)
    generator = torch.Generator(device="cuda").manual_seed(int(config["seed"]))
    if varlen:
        q = torch.randn((total, hq, 256), dtype=dtype, generator=generator, device="cuda")
        k = torch.randn((total, hkv, 256), dtype=dtype, generator=generator, device="cuda")
        v = torch.randn((total, hkv, 256), dtype=dtype, generator=generator, device="cuda")
        prefix = [0]
        for length in lengths:
            prefix.append(prefix[-1] + length)
        cu_q = torch.tensor(prefix, dtype=torch.int32, device="cuda")
        cu_k = cu_q.clone()
        output_shape = q.shape
        lse_shape = (hq, total)
    else:
        q = torch.randn((batch, total, hq, 256), dtype=dtype, generator=generator, device="cuda")
        k = torch.randn((batch, total, hkv, 256), dtype=dtype, generator=generator, device="cuda")
        v = torch.randn((batch, total, hkv, 256), dtype=dtype, generator=generator, device="cuda")
        cu_q = None
        cu_k = None
        output_shape = q.shape
        lse_shape = (batch, hq, total)

    endpoints = _make_mask_endpoints(torch, config["mask"], lengths)
    if config["per_head_mask"]:
        endpoints = endpoints.unsqueeze(0).repeat(hq, 1, 1)
    else:
        endpoints = endpoints.unsqueeze(0)
    mask_func = endpoints.to(device="cuda", non_blocking=True).contiguous()
    plan = create_mask_plan(
        mask_func,
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(lengths) if varlen else None,
        max_seqlen_k=max(lengths) if varlen else None,
        pack_gqa=False,
        build_backward=False,
        _fwd_variant="qstage1_2cta" if cta_group == 2 else None,
    )
    packed, plan_cu_q, plan_cu_k = plan._runtime_args
    if packed.sequence_desc is None or packed.fwd_work_desc is None:
        raise AssertionError("pinned HD256 plan did not provide schedule descriptors")

    def outputs():
        out, out_storage = _guarded_tensor(
            torch, output_shape, dtype, fill=float("nan"), guard=_OUTPUT_GUARD
        )
        lse, lse_storage = _guarded_tensor(
            torch, lse_shape, torch.float32, fill=_LSE_SENTINEL, guard=_LSE_GUARD
        )
        return {"out": out, "lse": lse, "guards": {"out": out_storage, "lse": lse_storage}}

    dummy_cu = torch.zeros((batch + 1,), dtype=torch.int32, device="cuda")
    runtime_cu_q = plan_cu_q if plan_cu_q is not None else dummy_cu
    runtime_cu_k = plan_cu_k if plan_cu_k is not None else dummy_cu
    immutable = {
        "q": q.clone(),
        "k": k.clone(),
        "v": v.clone(),
        "mask_func": mask_func.clone(),
        "cu_q": runtime_cu_q.clone(),
        "cu_k": runtime_cu_k.clone(),
    }
    for name in (
        "sequence_desc",
        "mask_block_cnt",
        "mask_block_offset",
        "mask_block_idx",
        "full_block_cnt",
        "full_block_offset",
        "full_block_idx",
        "mask_block_masks",
        "fwd_work_desc",
    ):
        immutable[name] = getattr(packed, name).clone()
    return {
        "config": dict(config),
        "q": q,
        "k": k,
        "v": v,
        "mask_func": mask_func,
        "plan": plan,
        "packed": packed,
        "cu_q": runtime_cu_q,
        "cu_k": runtime_cu_k,
        "softmax_scale": 1.0 / 16.0,
        "immutable": immutable,
        "tirx": outputs(),
        "source": outputs(),
    }


def _tirx_launch(executable, data):
    packed = data["packed"]
    args = (
        data["q"].reshape(-1),
        data["k"].reshape(-1),
        data["v"].reshape(-1),
        data["tirx"]["out"].reshape(-1),
        data["tirx"]["lse"].reshape(-1),
        data["cu_q"].reshape(-1),
        data["cu_k"].reshape(-1),
        packed.sequence_desc.reshape(-1),
        packed.mask_block_cnt.reshape(-1),
        packed.mask_block_offset.reshape(-1),
        packed.mask_block_idx.reshape(-1),
        packed.full_block_cnt.reshape(-1),
        packed.full_block_offset.reshape(-1),
        packed.full_block_idx.reshape(-1),
        packed.mask_block_masks.reshape(-1),
        packed.fwd_work_desc.reshape(-1),
        data["softmax_scale"] * _LOG2_E,
        data["softmax_scale"],
    )

    def launch():
        executable(*args)

    launch._keep_alive = args
    return launch


def _compile_reference(data):
    from cudnn.flex_attention import FlexAttentionFwd

    lse = data["source"]["lse"] if data["config"]["return_lse"] else None
    api = FlexAttentionFwd(
        data["q"], data["k"], data["v"], data["source"]["out"], data["plan"], lse
    )
    api.check_support()
    api.compile()

    def launch():
        api.execute(
            data["q"],
            data["k"],
            data["v"],
            data["source"]["out"],
            data["plan"],
            lse,
            softmax_scale=data["softmax_scale"],
        )

    launch._keep_alive = (api, lse)
    return launch


def _bits(torch, tensor):
    if tensor.dtype in (torch.bfloat16, torch.float16):
        return tensor.view(torch.uint16)
    if tensor.dtype == torch.float32:
        return tensor.view(torch.int32)
    return tensor


def _mismatch_report(torch, name, actual, expected):
    actual_bits = _bits(torch, actual)
    expected_bits = _bits(torch, expected)
    equal = actual_bits == expected_bits
    if bool(equal.all()):
        return None
    flat_equal = equal.reshape(-1)
    index = int((~flat_equal).nonzero()[0].item())
    actual_value = float(actual.reshape(-1)[index].item())
    expected_value = float(expected.reshape(-1)[index].item())
    actual_bit = int(actual_bits.reshape(-1)[index].item())
    expected_bit = int(expected_bits.reshape(-1)[index].item())
    finite = torch.isfinite(actual.float()) & torch.isfinite(expected.float())
    max_abs = (
        float((actual.float()[finite] - expected.float()[finite]).abs().max().item())
        if bool(finite.any())
        else float("nan")
    )
    return (
        f"{name} is not bitwise equal at flat index {index}: "
        f"tirx={actual_value} (bits={actual_bit:#x}), "
        f"source={expected_value} (bits={expected_bit:#x}), "
        f"max_finite_abs={max_abs:.9g}"
    )


def _assert_guards(torch, slot, buffers):
    for name, storage in buffers["guards"].items():
        guard = _OUTPUT_GUARD if name == "out" else _LSE_GUARD
        expected = torch.full((_GUARD_ELEMS,), guard, dtype=storage.dtype, device=storage.device)
        if not torch.equal(_bits(torch, storage[:_GUARD_ELEMS]), _bits(torch, expected)):
            raise AssertionError(f"{slot}.{name} prefix guard was modified")
        if not torch.equal(_bits(torch, storage[-_GUARD_ELEMS:]), _bits(torch, expected)):
            raise AssertionError(f"{slot}.{name} suffix guard was modified")


def _assert_immutable(torch, data):
    current = {
        "q": data["q"],
        "k": data["k"],
        "v": data["v"],
        "mask_func": data["mask_func"],
        "cu_q": data["cu_q"],
        "cu_k": data["cu_k"],
    }
    packed = data["packed"]
    for name in (
        "sequence_desc",
        "mask_block_cnt",
        "mask_block_offset",
        "mask_block_idx",
        "full_block_cnt",
        "full_block_offset",
        "full_block_idx",
        "mask_block_masks",
        "fwd_work_desc",
    ):
        current[name] = getattr(packed, name)
    for name, tensor in current.items():
        if not torch.equal(_bits(torch, tensor), _bits(torch, data["immutable"][name])):
            raise AssertionError(f"immutable input {name} was modified")


def _validate_outputs(data):
    import torch

    out_error = _mismatch_report(torch, "O", data["tirx"]["out"], data["source"]["out"])
    if out_error is not None:
        raise AssertionError(out_error)
    if data["config"]["return_lse"]:
        lse_error = _mismatch_report(torch, "LSE", data["tirx"]["lse"], data["source"]["lse"])
        if lse_error is not None:
            raise AssertionError(lse_error)
    elif not bool((data["tirx"]["lse"] == _LSE_SENTINEL).all()):
        raise AssertionError("no-LSE specialization modified the LSE sentinel")
    _assert_guards(torch, "tirx", data["tirx"])
    _assert_guards(torch, "source", data["source"])
    _assert_immutable(torch, data)


def _ptxas_register_usage_level(config):
    # The exhaustive register-level scan found two spill-free SASS classes.
    # Bench-suite selects the shorter level-4 class only for these exact
    # dense-causal GQA4 and CTA2 MHA schedule families; all others retain level 0.
    if (
        config["mask"] == "causal"
        and config["sequence_mode"] == "fixed"
        and int(config["num_q_heads"]) == 4
        and (
            (
                int(config["cta_group_size"]) == 2
                and int(config["num_kv_heads"]) in (1, 4)
                and config["dtype"] in ("bfloat16", "float16")
            )
            or (
                int(config["cta_group_size"]) == 1
                and int(config["num_kv_heads"]) == 1
                and config["dtype"] == "float16"
            )
        )
    ):
        return 4
    return 0


def _compile_tirx(config):
    import os

    from tirx_kernels.runner import compile_kernel

    name = "TVM_CUDA_PTXAS_REG_LEVEL"
    previous = os.environ.get(name)
    os.environ[name] = str(_ptxas_register_usage_level(config))
    try:
        return compile_kernel(get_kernel(**config), arch="sm_103a")
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def run_test(**config):
    import torch

    data = prepare_data(**config)
    executable = _compile_tirx(config)
    tirx_launch = _tirx_launch(executable, data)
    source_launch = _compile_reference(data)
    source_launch()
    tirx_launch()
    torch.cuda.synchronize()
    _validate_outputs(data)

    deterministic_out = data["tirx"]["out"].clone()
    deterministic_lse = data["tirx"]["lse"].clone()
    tirx_launch()
    torch.cuda.synchronize()
    out_error = _mismatch_report(torch, "deterministic.O", data["tirx"]["out"], deterministic_out)
    if out_error is not None:
        raise AssertionError(out_error)
    lse_error = _mismatch_report(torch, "deterministic.LSE", data["tirx"]["lse"], deterministic_lse)
    if lse_error is not None:
        raise AssertionError(lse_error)
    _assert_guards(torch, "tirx", data["tirx"])
    _assert_immutable(torch, data)


def prepare_bench(**config):
    from tirx_kernels.runner import prepared_gpu_benchmark

    executable = _compile_tirx(config)
    return prepared_gpu_benchmark(run_gpu, {"config": dict(config), "executable": executable})


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs):
    from tirx_kernels.runner import bench, defer_gpu_interrupts, external_references_enabled

    with defer_gpu_interrupts():
        import torch

    config = {**prepared["config"], **kwargs}
    with_source = external_references_enabled()
    gpu_state = prepared.get("gpu_state")
    if gpu_state is None:
        data = prepare_data(**config)
        gpu_state = {
            "data": data,
            "tirx_launch": _tirx_launch(prepared["executable"], data),
            "source_launch": None,
            "validated": False,
            "with_source": with_source,
        }
        prepared["gpu_state"] = gpu_state
    elif gpu_state["with_source"] != with_source:
        raise RuntimeError("reference timing mode changed within one prepared benchmark")

    data = gpu_state["data"]
    tirx_launch = gpu_state["tirx_launch"]
    if not gpu_state["validated"]:
        tirx_launch()
        torch.cuda.synchronize()
        if with_source:
            with defer_gpu_interrupts():
                source_launch = _compile_reference(data)
                gpu_state["source_launch"] = source_launch
                source_launch()
                torch.cuda.synchronize()
            _validate_outputs(data)
        gpu_state["validated"] = True

    source_launch = gpu_state["source_launch"]
    references = {"cudnn_frontend": lambda: source_launch} if source_launch is not None else None
    return bench(
        {"tirx": tirx_launch},
        references=references,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
    return prepare_bench(**config).run_gpu(
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
    "run_gpu",
    "run_test",
]
