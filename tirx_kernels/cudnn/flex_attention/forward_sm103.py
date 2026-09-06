# Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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
# (https://github.com/NVIDIA/cudnn-frontend @ 2497bee508e6337c086f0993f68b41cabce97436), Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM103 generic arbitrary interval-mask FlexAttention forward pass.

Upstream sources:
``python/cudnn/flex_attention/kernels/sm100/fwd/forward.py``,
``forward_qstage1.py``, ``forward_qstage2.py``, and the packed-mask,
pipeline, softmax, and CLC scheduler helpers reachable from those kernels.
"""

import tirx_kernels.kern as K

KERNEL_META = {
    "name": "cudnn_sm103_flex_attention_forward",
    "category": "cudnn",
    "runtime_cuda_archs": ["sm_103a"],
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


def _config(label, **overrides):
    config = {
        "label": label,
        "batch": 1,
        "num_q_heads": 2,
        "num_kv_heads": 2,
        "seqlen_q": 128,
        "seqlen_kv": 256,
        "head_dim": 128,
        "head_dim_v": 128,
        "dtype": "bfloat16",
        "variant": "qstage1_1cta",
        "layout": "fixed",
        "mask_pattern": "full",
        "pack_gqa": False,
        "return_lse": True,
        "softmax_scale": None,
        "seed": 20260904,
        # Correctness-first qstage1 implementation parameters.
        "kv_blocks": 2,
        "tensor_layout": "bshd",
        "has_block_sizes": True,
        "block_count_mode": "fixed",
        "block_count_pattern": None,
        "use_int64_kv_strides": False,
        "seqlens_q": None,
        "seqlens_kv": None,
        "mask_heads": 1,
    }
    config.update(overrides)
    if config["variant"] not in ("qstage2_1cta", "qstage1_1cta", "qstage1_2cta"):
        raise ValueError(f"unknown FlexAttention forward variant {config['variant']!r}")
    if config["layout"] not in ("fixed", "varlen"):
        raise ValueError("layout must be fixed or varlen")
    if config["layout"] == "varlen":
        if config["seqlens_q"] is None or config["seqlens_kv"] is None:
            raise ValueError("varlen layout requires seqlens_q and seqlens_kv")
        config["seqlens_q"] = tuple(int(x) for x in config["seqlens_q"])
        config["seqlens_kv"] = tuple(int(x) for x in config["seqlens_kv"])
        if (
            len(config["seqlens_q"]) != config["batch"]
            or len(config["seqlens_kv"]) != config["batch"]
        ):
            raise ValueError("varlen sequence-length tuples must match batch")
        if min(config["seqlens_q"]) < 0 or min(config["seqlens_kv"]) < 0:
            raise ValueError("varlen sequence lengths must be nonnegative")
        config["seqlen_q"] = max(config["seqlens_q"], default=0)
        config["seqlen_kv"] = max(config["seqlens_kv"], default=0)
    if config["mask_pattern"] not in ("full", "causal", "split", "mixed", "empty"):
        raise ValueError(f"unknown interval mask pattern {config['mask_pattern']!r}")
    physical_blocks = (config["seqlen_kv"] + 127) // 128
    config["kv_blocks"] = physical_blocks + (physical_blocks & 1)
    config["has_block_sizes"] = False
    config["block_count_mode"] = "variable_empty"
    return config


# Start with one deterministic, debugger-friendly specialization. Boundary
# and branch-complete configs are added as each frozen-sketch path lands.
CONFIGS = [
    _config("c_qs1_1cta_bf16_d128_full_b1_sq128_sk256_h2"),
    _config(
        "c_qs1_1cta_bf16_d128_causal_tail_sq129_sk255_h2",
        seqlen_q=129,
        seqlen_kv=255,
        mask_pattern="causal",
        seed=20260906,
    ),
    _config(
        "c_qs1_1cta_fp16_d64_split_sq127_sk384_mha",
        dtype="float16",
        head_dim=64,
        head_dim_v=64,
        seqlen_q=127,
        seqlen_kv=384,
        mask_pattern="split",
        seed=20260907,
    ),
    _config(
        "c_qs1_1cta_bf16_d96_empty_sq128_sk256_mha",
        head_dim=96,
        head_dim_v=96,
        mask_pattern="empty",
        seed=20260908,
    ),
    _config(
        "c_qs1_1cta_fp16_d128_full_gqa_pack",
        dtype="float16",
        num_q_heads=4,
        num_kv_heads=2,
        pack_gqa=True,
        seed=20260909,
    ),
    _config(
        "c_qs1_1cta_bf16_d64_causal_mqa_nopack_nolse",
        head_dim=64,
        head_dim_v=64,
        num_q_heads=8,
        num_kv_heads=1,
        pack_gqa=False,
        mask_pattern="causal",
        return_lse=False,
        seed=20260910,
    ),
    _config("c_qs1_1cta_bf16_d128_dv64_full", head_dim=128, head_dim_v=64, seed=20260911),
    _config(
        "c_qs1_1cta_fp16_d80_dv48_split",
        dtype="float16",
        head_dim=80,
        head_dim_v=48,
        mask_pattern="split",
        seed=20260912,
    ),
    _config(
        "c_qs1_1cta_bf16_d8_dv8_causal",
        head_dim=8,
        head_dim_v=8,
        mask_pattern="causal",
        seed=20260913,
    ),
    _config("c_qs1_1cta_bf16_d192_dv128_full", head_dim=192, head_dim_v=128, seed=20260914),
    _config(
        "c_qs2_1cta_bf16_d128_full_sq256_sk512",
        variant="qstage2_1cta",
        seqlen_q=256,
        seqlen_kv=512,
        seed=20260915,
    ),
    _config("c_qs1_1cta_bf16_d128_full_wide_sk1024", seqlen_kv=1024, seed=20260916),
    _config(
        "c_qs1_2cta_fp16_d128_split_sq256_sk512",
        variant="qstage1_2cta",
        dtype="float16",
        seqlen_q=256,
        seqlen_kv=512,
        mask_pattern="split",
        seed=20260917,
    ),
    _config(
        "c_qs2_1cta_bf16_d192_dv128_split_sq256_sk384",
        variant="qstage2_1cta",
        head_dim=192,
        head_dim_v=128,
        seqlen_q=256,
        seqlen_kv=384,
        mask_pattern="split",
        seed=20260918,
    ),
    _config(
        "c_qs1_2cta_bf16_d192_dv128_full_sq256_sk384",
        variant="qstage1_2cta",
        head_dim=192,
        head_dim_v=128,
        seqlen_q=256,
        seqlen_kv=384,
        seed=20260919,
    ),
    _config(
        "c_qs1_1cta_fp16_d120_dv104_full_i64kv",
        dtype="float16",
        head_dim=120,
        head_dim_v=104,
        use_int64_kv_strides=True,
        seed=20260920,
    ),
    _config(
        "c_qs1_1cta_bf16_d128_split_gqa_unpack_perhead_scale",
        num_q_heads=4,
        num_kv_heads=2,
        mask_heads=4,
        pack_gqa=False,
        mask_pattern="split",
        softmax_scale=0.125,
        seed=20260921,
    ),
    _config(
        "c_qs2_1cta_bf16_d64_full_b2_sq256_sk384",
        batch=2,
        variant="qstage2_1cta",
        head_dim=64,
        head_dim_v=64,
        seqlen_q=256,
        seqlen_kv=384,
        seed=20260922,
    ),
    _config(
        "c_qs1_2cta_fp16_d64_split_gqa_pack",
        variant="qstage1_2cta",
        dtype="float16",
        head_dim=64,
        head_dim_v=64,
        num_q_heads=4,
        num_kv_heads=2,
        pack_gqa=True,
        seqlen_q=256,
        seqlen_kv=384,
        mask_pattern="split",
        seed=20260923,
    ),
    _config(
        "c_varlen_qs1_1cta_bf16_d64_split_b2",
        batch=2,
        layout="varlen",
        head_dim=64,
        head_dim_v=64,
        seqlens_q=(73, 129),
        seqlens_kv=(191, 257),
        mask_pattern="split",
        seed=20260924,
    ),
    _config(
        "c_varlen_qs2_1cta_fp16_d128_full_gqa_pack_b2",
        batch=2,
        layout="varlen",
        variant="qstage2_1cta",
        dtype="float16",
        num_q_heads=4,
        num_kv_heads=2,
        pack_gqa=True,
        seqlens_q=(65, 131),
        seqlens_kv=(193, 259),
        seed=20260925,
    ),
    _config(
        "c_varlen_qs1_2cta_bf16_d64_split_b2",
        batch=2,
        layout="varlen",
        variant="qstage1_2cta",
        head_dim=64,
        head_dim_v=64,
        seqlens_q=(127, 257),
        seqlens_kv=(129, 385),
        mask_pattern="split",
        seed=20260926,
    ),
    _config(
        "c_varlen_qs2_1cta_bf16_d192_dv128_split_b2",
        batch=2,
        layout="varlen",
        variant="qstage2_1cta",
        head_dim=192,
        head_dim_v=128,
        seqlens_q=(73, 129),
        seqlens_kv=(191, 257),
        mask_pattern="split",
        seed=20260927,
    ),
    _config(
        "c_qs2_1cta_bf16_d128_mixed_sq256_sk384",
        variant="qstage2_1cta",
        seqlen_q=256,
        seqlen_kv=384,
        mask_pattern="mixed",
        seed=20260928,
    ),
    _config("c_qs1_1cta_bf16_d128_full_single_kv_block", seqlen_kv=128, seed=20260929),
    _config(
        "c_qs1_2cta_fp16_d64_full_single_kv_block",
        variant="qstage1_2cta",
        dtype="float16",
        head_dim=64,
        head_dim_v=64,
        seqlen_q=256,
        seqlen_kv=128,
        seed=20260930,
    ),
    _config(
        "c_qs2_1cta_fp16_d64_split_sq256_sk384_nolse",
        variant="qstage2_1cta",
        dtype="float16",
        head_dim=64,
        head_dim_v=64,
        seqlen_q=256,
        seqlen_kv=384,
        mask_pattern="split",
        return_lse=False,
        seed=20260931,
    ),
    _config(
        "c_qs2_1cta_bf16_d64_empty_sq256_sk256",
        variant="qstage2_1cta",
        head_dim=64,
        head_dim_v=64,
        seqlen_q=256,
        seqlen_kv=256,
        mask_pattern="empty",
        seed=20260932,
    ),
]

BENCH_CONFIGS = [
    _config(
        "p_qs2_1cta_bf16_d64_full_b1_sq1024_sk384_h8",
        variant="qstage2_1cta",
        head_dim=64,
        head_dim_v=64,
        num_q_heads=8,
        num_kv_heads=8,
        seqlen_q=1024,
        seqlen_kv=384,
        seed=20261001,
    ),
    _config(
        "p_qs2_1cta_bf16_d128_full_b1_sq4096_sk512_h8",
        variant="qstage2_1cta",
        num_q_heads=8,
        num_kv_heads=8,
        seqlen_q=4096,
        seqlen_kv=512,
        seed=20261002,
    ),
    _config(
        "p_qs2_1cta_fp16_d80_dv48_causal_b2_sq2048_sk384_h8_nolse",
        batch=2,
        variant="qstage2_1cta",
        dtype="float16",
        head_dim=80,
        head_dim_v=48,
        num_q_heads=8,
        num_kv_heads=8,
        seqlen_q=2048,
        seqlen_kv=384,
        mask_pattern="causal",
        return_lse=False,
        seed=20261003,
    ),
    _config(
        "p_qs2_1cta_bf16_d192_dv128_split_b1_sq2048_sk384_h4",
        variant="qstage2_1cta",
        head_dim=192,
        head_dim_v=128,
        num_q_heads=4,
        num_kv_heads=4,
        seqlen_q=2048,
        seqlen_kv=384,
        mask_pattern="split",
        seed=20261004,
    ),
    _config(
        "p_varlen_qs2_1cta_bf16_d128_full_gqa_pack_b4_h8kv2",
        batch=4,
        layout="varlen",
        variant="qstage2_1cta",
        num_q_heads=8,
        num_kv_heads=2,
        pack_gqa=True,
        seqlens_q=(1024, 2048, 3072, 4096),
        seqlens_kv=(257, 384, 511, 512),
        seed=20261005,
    ),
    _config(
        "p_qs1_1cta_bf16_d64_causal_mqa_b1_sq2048_sk4096_h8kv1_nolse",
        head_dim=64,
        head_dim_v=64,
        num_q_heads=8,
        num_kv_heads=1,
        seqlen_q=2048,
        seqlen_kv=4096,
        mask_pattern="causal",
        return_lse=False,
        seed=20261006,
    ),
    _config(
        "p_qs1_1cta_bf16_d128_full_b1_sq4096_sk8192_h8",
        num_q_heads=8,
        num_kv_heads=8,
        seqlen_q=4096,
        seqlen_kv=8192,
        seed=20261007,
    ),
    _config(
        "p_qs1_1cta_bf16_d128_dv64_split_gqa_b1_sq4096_sk4096_h8kv2",
        head_dim_v=64,
        num_q_heads=8,
        num_kv_heads=2,
        mask_heads=8,
        seqlen_q=4096,
        seqlen_kv=4096,
        mask_pattern="split",
        softmax_scale=0.125,
        seed=20261008,
    ),
    _config(
        "p_qs1_1cta_fp16_d128_full_gqa_pack_b1_sq4096_sk8192_h8kv2",
        dtype="float16",
        num_q_heads=8,
        num_kv_heads=2,
        pack_gqa=True,
        seqlen_q=4096,
        seqlen_kv=8192,
        seed=20261009,
    ),
    _config(
        "p_qs1_1cta_bf16_d192_dv128_split_b1_sq2048_sk4096_h4",
        head_dim=192,
        head_dim_v=128,
        num_q_heads=4,
        num_kv_heads=4,
        seqlen_q=2048,
        seqlen_kv=4096,
        mask_pattern="split",
        seed=20261010,
    ),
    _config(
        "p_qs1_2cta_bf16_d64_full_b1_sq4096_sk4096_h8",
        variant="qstage1_2cta",
        head_dim=64,
        head_dim_v=64,
        num_q_heads=8,
        num_kv_heads=8,
        seqlen_q=4096,
        seqlen_kv=4096,
        seed=20261011,
    ),
    _config(
        "p_qs1_2cta_bf16_d128_full_b1_sq4096_sk8192_h8",
        variant="qstage1_2cta",
        num_q_heads=8,
        num_kv_heads=8,
        seqlen_q=4096,
        seqlen_kv=8192,
        seed=20261012,
    ),
    _config(
        "p_qs1_2cta_fp16_d64_split_gqa_pack_b1_sq4096_sk384_h8kv2",
        variant="qstage1_2cta",
        dtype="float16",
        head_dim=64,
        head_dim_v=64,
        num_q_heads=8,
        num_kv_heads=2,
        pack_gqa=True,
        seqlen_q=4096,
        seqlen_kv=384,
        mask_pattern="split",
        seed=20261013,
    ),
    _config(
        "p_qs1_2cta_bf16_d192_dv128_full_b1_sq2048_sk4096_h4",
        variant="qstage1_2cta",
        head_dim=192,
        head_dim_v=128,
        num_q_heads=4,
        num_kv_heads=4,
        seqlen_q=2048,
        seqlen_kv=4096,
        seed=20261014,
    ),
    _config(
        "p_varlen_qs1_2cta_bf16_d128_split_gqa_pack_b4_h8kv2",
        batch=4,
        layout="varlen",
        variant="qstage1_2cta",
        num_q_heads=8,
        num_kv_heads=2,
        pack_gqa=True,
        seqlens_q=(1024, 2048, 3072, 4096),
        seqlens_kv=(1025, 2048, 3073, 4096),
        mask_pattern="split",
        seed=20261015,
    ),
]


_M = 128
_N = 128
_WARPS = 16
_TMEM_COLS = 512
_SPLIT_P = 96
_NEG_INF = -float("inf")
_LOG2_E = 1.4426950408889634
_LN2 = 0.6931471805599453
_TRY_WAIT_TICKS = 10_000_000
_MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
_MMA_F16_2CTA = "tcgen05.mma.cta_group::2.kind::f16"
_TCGEN_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
_TCGEN_COMMIT_2CTA = (
    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
)
_TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
_TMEM_ALLOC_2CTA = "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"
_TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
_TMEM_DEALLOC_2CTA = "tcgen05.dealloc.cta_group::2.sync.aligned.b32"
_TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
_TMEM_RELINQUISH_2CTA = "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"
_TMEM_LD32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
_TMEM_LD32_RED_MAX = "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
_TMEM_LD16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
_TMEM_ST16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
_TMA_G2S_3D = (
    "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
)
_TMA_G2S_4D = (
    "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
)
_TMA_G2S_5D = (
    "cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
)
_TMA_G2S_3D_2CTA = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
)
_TMA_G2S_4D_2CTA = (
    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
)
_TMA_G2S_5D_2CTA = (
    "cp.async.bulk.tensor.5d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2"
)
_BULK_G2S_MASK = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
_TMA_S2G_4D = "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint"
_TMA_S2G_5D = "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
_TMA_CACHE = K.uint64(0)
_POLY_EX2_3 = (
    1.0,
    0.695146143436431884765625,
    0.227564394474029541015625,
    0.077119089663028717041015625,
)
_FP32_ROUND_INT = float(2**23 + 2**22)


def _load_i32(buffer, index):
    value = K.local_scalar("int32")
    K.ptx.ld.global_.s32(value, buffer.ptr_to([index]))
    return value


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


def _exp2_nonfast(value):
    out = K.local_scalar("float32")
    K.ptx.ex2.approx.f32(out, value)
    return out


def _log2(value):
    out = K.local_scalar("float32")
    K.ptx.lg2.approx.ftz.f32(out, value)
    return out


def _rcp(value):
    out = K.local_scalar("float32")
    K.ptx.rcp.approx.ftz.f32(out, value)
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


def _reduce_max_128(values, initial=None):
    acc = K.alloc_local((4,), "float32")
    if initial is None:
        K.ptx.max.f32(acc[0], values[0], values[1])
    else:
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


def _packed_sum_128(values, old_sum, old_scale, first):
    acc = K.alloc_local((8,), "float32")
    with K.unroll(8) as j:
        K.assign(acc[j], values[j])
    if not first:
        scaled_old = K.local_scalar("float32", init=old_sum * old_scale)
        _packed("add.rn.f32x2", acc, 0, values[0], values[1], scaled_old, K.float32(0.0))
    with K.unroll(1, 16) as group:
        base = group * 8
        with K.unroll(4) as pair:
            _packed(
                "add.rn.f32x2",
                acc,
                pair * 2,
                acc[pair * 2],
                acc[pair * 2 + 1],
                values[base + pair * 2],
                values[base + pair * 2 + 1],
            )
    for lo, hi in ((0, 2), (4, 6), (0, 4)):
        _packed("add.rn.f32x2", acc, lo, acc[lo], acc[lo + 1], acc[hi], acc[hi + 1])
    return acc[0] + acc[1]


def _combine_int_frac_ex2(x_rounded, frac_ex2):
    rounded_i = K.local_scalar("int32")
    frac_i = K.local_scalar("int32")
    exponent = K.local_scalar("int32")
    bits = K.local_scalar("int32")
    out = K.local_scalar("float32")
    K.ptx.mov.b32(rounded_i, x_rounded)
    K.ptx.mov.b32(frac_i, frac_ex2)
    K.ptx.shl.b32(exponent, rounded_i, K.uint32(23))
    K.ptx.add.s32(bits, exponent, frac_i)
    K.ptx.mov.b32(out, bits)
    return out


def _ex2_emulation_2(values, base):
    clamped = K.alloc_local((2,), "float32")
    K.ptx.max.f32(clamped[0], values[base], K.float32(-127.0))
    K.ptx.max.f32(clamped[1], values[base + 1], K.float32(-127.0))
    packed = K.local_scalar("uint64")
    rhs = K.local_scalar("uint64")
    addend = K.local_scalar("uint64")
    rounded = K.alloc_local((2,), "float32")
    K.ptx.mov.b64(packed, clamped[0], clamped[1])
    K.ptx.mov.b64(rhs, K.float32(_FP32_ROUND_INT), K.float32(_FP32_ROUND_INT))
    K.ptx.add.rm.f32x2(packed, packed, rhs)
    K.ptx.mov.b64(rounded[0], rounded[1], packed)
    rounded_back = K.alloc_local((2,), "float32")
    K.ptx.mov.b64(packed, rounded[0], rounded[1])
    K.ptx.sub.rn.f32x2(packed, packed, rhs)
    K.ptx.mov.b64(rounded_back[0], rounded_back[1], packed)
    frac = K.alloc_local((2,), "float32")
    K.ptx.mov.b64(packed, clamped[0], clamped[1])
    K.ptx.mov.b64(rhs, rounded_back[0], rounded_back[1])
    K.ptx.sub.rn.f32x2(packed, packed, rhs)
    K.ptx.mov.b64(frac[0], frac[1], packed)
    poly = K.alloc_local((2,), "float32")
    K.assign(poly[0], K.float32(_POLY_EX2_3[3]))
    K.assign(poly[1], K.float32(_POLY_EX2_3[3]))
    for coeff in (_POLY_EX2_3[2], _POLY_EX2_3[1], _POLY_EX2_3[0]):
        K.ptx.mov.b64(packed, poly[0], poly[1])
        K.ptx.mov.b64(rhs, frac[0], frac[1])
        K.ptx.mov.b64(addend, K.float32(coeff), K.float32(coeff))
        K.ptx.fma.rn.f32x2(packed, packed, rhs, addend)
        K.ptx.mov.b64(poly[0], poly[1], packed)
    K.assign(values[base], _combine_int_frac_ex2(rounded[0], poly[0]))
    K.assign(values[base + 1], _combine_int_frac_ex2(rounded[1], poly[1]))


def _apply_mask128(values, block_size):
    with K.If(block_size < 128), K.Then():
        with K.unroll(4) as quarter:
            shift = K.max((quarter + 1) * 32 - block_size, 0)
            mask = K.local_scalar("uint32")
            K.ptx.shr.u32(mask, K.uint32(0xFFFFFFFF), K.cast(shift, "uint32"))
            with K.unroll(32) as bit:
                live = K.bitwise_and(mask, K.shift_left(K.uint32(1), K.cast(bit, "uint32"))) != 0
                index = quarter * 32 + bit
                K.assign(values[index], K.if_then_else(live, values[index], K.float32(_NEG_INF)))


def _wait(barrier, stage, phase):
    K.cuda.mbarrier_wait(barrier.ptr_to([stage]), phase)


def _stats_arrive(stage, warp):
    K.ptx.bar.arrive(K.cast(3 + stage * 4 + warp, "uint32"), K.uint32(64))


def _stats_sync(stage, warp):
    K.ptx.bar.sync(K.cast(3 + stage * 4 + warp, "uint32"), K.uint32(64))


def _tmem_load32(dst, address):
    K.ptx[_TMEM_LD32](*(dst[i] for i in range(32)), K.cast(address, "uint32"))


def _tmem_load32_red_max(dst, reduction, address):
    K.ptx[_TMEM_LD32_RED_MAX](*(dst[i] for i in range(32)), reduction, K.cast(address, "uint32"))


def _tmem_load16(dst, address):
    K.ptx[_TMEM_LD16](*(dst[i] for i in range(16)), K.cast(address, "uint32"))


def _tmem_store16(src, address):
    K.ptx[_TMEM_ST16](K.cast(address, "uint32"), *(src[i] for i in range(16)))


def _query_cancel_response(response, next_cta_x, valid):
    payload = K.local_scalar("uint128")
    payload_lo = K.local_scalar("uint64")
    payload_hi = K.local_scalar("uint64")
    canceled = K.local_scalar("uint32")
    K.ptx.ld.shared.v2.b64(payload_lo, payload_hi, K.address_of(response[0]))
    K.ptx.mov.b128(payload, payload_lo, payload_hi)
    K.ptx.clusterlaunchcontrol.query_cancel.is_canceled.pred.b128(canceled, payload)
    K.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__x.b32.b128(
        next_cta_x, payload, pred=canceled
    )
    K.assign(valid, K.cast(canceled, "int32"))
    K.ptx.fence.proxy.async_.shared__cta()


def _make_kernel(**config):
    batch = int(config["batch"])
    hq = int(config["num_q_heads"])
    hkv = int(config["num_kv_heads"])
    seqlen_q = int(config["seqlen_q"])
    seqlen_kv = int(config["seqlen_kv"])
    logical_d = int(config["head_dim"])
    logical_dv = int(config["head_dim_v"])
    dp = ((logical_d + 15) // 16) * 16
    dvp = ((logical_dv + 15) // 16) * 16
    dtype = config["dtype"]
    max_blocks = int(config["kv_blocks"])
    has_sizes = bool(config["has_block_sizes"])
    has_nums = config["block_count_mode"] != "fixed"
    allow_empty = config["block_count_mode"] == "variable_empty"
    return_lse = bool(config["return_lse"])
    is_varlen = config["layout"] == "varlen"
    variant = config["variant"]
    q_stages = 2 if variant == "qstage2_1cta" else 1
    cta_group = 2 if variant == "qstage1_2cta" else 1
    # The source stages 2CTA mask payloads through shared memory, but that
    # path produced stale payloads on cold multi-tile NVRTC launches. Direct
    # 16-byte vector reads preserve its consumer-native payload layout.
    use_smem_mask_pipeline = False
    work_q_tiles = q_stages * cta_group
    output_stages = 1 if cta_group == 2 else 2
    use_i64_kv = bool(config["use_int64_kv_strides"])
    if hq <= 0 or hkv <= 0 or hq % hkv:
        raise ValueError("num_q_heads must be a positive multiple of num_kv_heads")
    if not (
        logical_d % 8 == 0
        and logical_dv % 8 == 0
        and 8 <= logical_d <= 128
        and 8 <= logical_dv <= 128
    ) and (logical_d, logical_dv) != (192, 128):
        raise ValueError("unsupported FlexAttention D/Dv specialization")
    if dtype not in ("bfloat16", "float16"):
        raise ValueError("dtype must be bfloat16 or float16")
    if config["block_count_mode"] == "fixed" and (max_blocks < 2 or max_blocks % 2):
        raise ValueError("fixed block count must be even and at least two")

    ratio = hq // hkv
    requested_pack = config["pack_gqa"]
    pack = (ratio > 1 if requested_pack == "auto" else bool(requested_pack)) and 128 % ratio == 0
    hmask = int(config["mask_heads"])
    if hmask not in (1, hq):
        raise ValueError("mask_heads must be one or num_q_heads")
    if pack and hmask != 1:
        raise ValueError("packed GQA requires one shared mask head")
    if is_varlen and use_i64_kv:
        raise ValueError("the fixed-layout Int64 KV stride probe is not a varlen mode")
    scheduled_heads = hkv if pack else hq
    q_lengths = tuple(int(x) for x in config["seqlens_q"]) if is_varlen else (seqlen_q,) * batch
    full_varlen_output_tiles = is_varlen and all(
        (length * (ratio if pack else 1)) % 128 == 0 for length in q_lengths
    )
    direct_full_varlen_ids = (
        is_varlen
        and variant == "qstage2_1cta"
        and config["mask_pattern"] == "full"
        and all((length * (ratio if pack else 1)) % (q_stages * 128) == 0 for length in q_lengths)
    )
    kv_lengths = tuple(int(x) for x in config["seqlens_kv"]) if is_varlen else (seqlen_kv,) * batch
    plan_tile_rows = work_q_tiles * 128
    q_blocks_per_sample = tuple(
        (length * (ratio if pack else 1) + plan_tile_rows - 1) // plan_tile_rows
        for length in q_lengths
    )
    q_blocks = max(q_blocks_per_sample)
    total_m_blocks = sum(q_blocks_per_sample)
    total_q = sum(q_lengths)
    total_k = sum(kv_lengths)
    num_work_records = total_m_blocks * scheduled_heads
    plan_row_capacity = hmask * total_m_blocks

    overlap_qo = dp == 192 and dvp >= 64
    q_bytes = q_stages * 128 * dp * 2
    output_bytes = output_stages * 128 * dvp * 2
    q_output_bytes = max(q_bytes, output_bytes) if overlap_qo else q_bytes + output_bytes
    k_stage_bytes = (128 // cta_group) * dp * 2
    v_stage_bytes = 128 * (dvp // cta_group) * 2
    kv_stage_bytes = max(k_stage_bytes, v_stage_bytes)
    kv_stages = max(2, (224 * 1024 - q_output_bytes) // kv_stage_bytes)
    if (dp, dvp) == (192, 128) and kv_stages == 2:
        kv_stages = 3
    if variant == "qstage1_1cta":
        kv_stages = min(kv_stages, 4)
        if (dp, dvp) == (192, 128):
            kv_stages = 2
    elif variant == "qstage1_2cta":
        kv_stages = min(kv_stages, 7)
    uneven_kv = (dp, dvp) == (192, 128) and kv_stages == 3
    kv_stage_stride_bytes = (k_stage_bytes + v_stage_bytes) // 2 if uneven_kv else kv_stage_bytes
    kv_storage_bytes = (
        (kv_stages - 1) * kv_stage_stride_bytes + kv_stage_bytes
        if uneven_kv
        else kv_stages * kv_stage_bytes
    )
    uneven_kv_offset = 128 * (dp - dvp) // 2 if uneven_kv else 0

    def align_up(value, alignment):
        return (value + alignment - 1) // alignment * alignment

    cursor = 0

    def place(size, alignment):
        nonlocal cursor
        cursor = align_up(cursor, alignment)
        result = cursor
        cursor += size
        return result

    q_barrier_offset = place(16 * q_stages, 8)
    kv_barrier_offset = place(16 * kv_stages, 8)
    mask0_barrier_offset = place(16 if use_smem_mask_pipeline else 0, 8)
    mask1_barrier_offset = place(16 if use_smem_mask_pipeline else 0, 8)
    spo_offset = place(32, 8)
    plast_offset = place(32, 8)
    oacc_offset = place(32, 8)
    stats_offset = place(32, 8)
    oepi_offset = place(16 * output_stages, 8)
    load_epi_offset = place(16 if overlap_qo else 0, 8)
    tmem_dealloc_offset = place(8, 8)
    tmem_offset = place(4, 4)
    scale_offset = place(2048, 4)
    mask_offset = place(4096 if use_smem_mask_pipeline else 0, 16)
    clc_offset = place(16, 8)
    response_offset = place(16, 16)
    prefix_offset = cursor
    so_offset = align_up(prefix_offset, 1024)
    sq_offset = so_offset if overlap_qo else so_offset + output_bytes
    skv_offset = so_offset + q_output_bytes
    smem_bytes = skv_offset + kv_storage_bytes
    offsets = dict(
        q=q_barrier_offset,
        kv=kv_barrier_offset,
        mask0=mask0_barrier_offset,
        mask1=mask1_barrier_offset,
        spo=spo_offset,
        plast=plast_offset,
        oacc=oacc_offset,
        load_epi=load_epi_offset,
        stats=stats_offset,
        oepi=oepi_offset,
        tmem_dealloc=tmem_dealloc_offset,
        tmem=tmem_offset,
        scale=scale_offset,
        mask=mask_offset,
        clc=clc_offset,
        response=response_offset,
        prefix=prefix_offset,
        so=so_offset,
        sq=sq_offset,
        skv=skv_offset,
    )

    def band_width(padded):
        if padded % 64 == 0:
            return 64
        if padded % 32 == 0:
            return 32
        return 16

    def smem_swizzle(contiguous):
        if contiguous % 64 == 0:
            return 3
        if contiguous % 32 == 0:
            return 2
        if contiguous % 16 == 0:
            return 1
        return 0

    q_band_width = band_width(dp)
    o_band_width = band_width(dvp)
    # Group-2 PV distributes the N-major V operand across the two CTAs.  Its
    # local contiguous dimension is Dvp/2 (including the interleaved Dvp=16
    # case), while each CTA still writes a complete Dvp-wide output tile.
    v_load_band_width = dvp // cta_group if cta_group == 2 else o_band_width
    q_num_bands = dp // q_band_width
    v_load_num_bands = 1 if cta_group == 2 else dvp // v_load_band_width
    o_num_bands = dvp // o_band_width
    q_smem_swizzle = smem_swizzle(q_band_width)
    v_load_smem_swizzle = smem_swizzle(v_load_band_width)
    o_smem_swizzle = smem_swizzle(o_band_width)
    elem_dtype = K.bf16 if dtype == "bfloat16" else K.f16
    elem_dtype_name = "bfloat16" if dtype == "bfloat16" else "float16"
    if cta_group == 2:
        qk_idesc = 0x10200490 if dtype == "bfloat16" else 0x10200010
        pv_idesc = 0x10010490 + (dvp // 16) * 0x40000
    else:
        qk_idesc = 0x08200490 if dtype == "bfloat16" else 0x08200010
        pv_idesc = 0x08010490 + (dvp // 16) * 0x40000
    if dtype == "float16":
        pv_idesc -= 0x480
    mma_f16 = _MMA_F16_2CTA if cta_group == 2 else _MMA_F16
    tmem_alloc = _TMEM_ALLOC_2CTA if cta_group == 2 else _TMEM_ALLOC
    tmem_dealloc = _TMEM_DEALLOC_2CTA if cta_group == 2 else _TMEM_DEALLOC
    tmem_relinquish = _TMEM_RELINQUISH_2CTA if cta_group == 2 else _TMEM_RELINQUISH
    tma_g2s_3d = _TMA_G2S_3D_2CTA if cta_group == 2 else _TMA_G2S_3D
    tma_g2s_4d = _TMA_G2S_4D_2CTA if cta_group == 2 else _TMA_G2S_4D
    tma_g2s_5d = _TMA_G2S_5D_2CTA if cta_group == 2 else _TMA_G2S_5D
    mma_disable_lanes = (K.uint32(0),) * (8 if cta_group == 2 else 4)
    regs_other = 48 if max(dp, dvp) <= 64 else 56
    regs_softmax = 200 if dp <= 64 else 184
    regs_correction = 64 if dvp <= 64 else 88
    if variant == "qstage2_1cta" and is_varlen and dp >= 96:
        regs_softmax = 200
        regs_correction = 64
        regs_other = 48
    if variant != "qstage2_1cta" and dp >= 96:
        if dp >= 192:
            regs_softmax = 176
            regs_correction = 104
            regs_other = 56
        else:
            regs_softmax = 176
            regs_correction = 104
            regs_other = 56
    physical_blocks = (seqlen_kv + 127) // 128
    overlap_pv_with_k_wait = config["variant"] == "qstage1_1cta" and physical_blocks >= 7

    def host_prelude(params):
        def encode(tensor, dims, strides, box, swizzle):
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
                swizzle,
                2,
                0,
            )
            return descriptor

        q = params["q"]
        k = params["k"]
        v = params["v"]
        out = params["out"]
        elem_bytes = 2
        if is_varlen:
            if pack:
                q_dims = (logical_d, ratio, total_q, hkv)
                q_strides = (
                    logical_d * elem_bytes,
                    hq * logical_d * elem_bytes,
                    ratio * logical_d * elem_bytes,
                )
                q_box = (q_band_width, ratio, 128 // ratio, 1)
            else:
                q_dims = (logical_d, total_q, hq)
                q_strides = (hq * logical_d * elem_bytes, logical_d * elem_bytes)
                q_box = (q_band_width, 128, 1)
            k_dims = (logical_d, total_k, hkv)
            k_strides = (hkv * logical_d * elem_bytes, logical_d * elem_bytes)
            v_dims = (logical_dv, total_k, hkv)
            v_strides = (hkv * logical_dv * elem_bytes, logical_dv * elem_bytes)
        else:
            if pack:
                q_dims = (logical_d, ratio, seqlen_q, hkv, batch)
                q_strides = (
                    seqlen_q * logical_d * elem_bytes,
                    logical_d * elem_bytes,
                    ratio * seqlen_q * logical_d * elem_bytes,
                    hq * seqlen_q * logical_d * elem_bytes,
                )
                q_box = (q_band_width, ratio, 128 // ratio, 1, 1)
                o_dims = (logical_dv, ratio, seqlen_q, hkv, batch)
                o_strides = (
                    seqlen_q * logical_dv * elem_bytes,
                    logical_dv * elem_bytes,
                    ratio * seqlen_q * logical_dv * elem_bytes,
                    hq * seqlen_q * logical_dv * elem_bytes,
                )
                o_box = (o_band_width, ratio, 128 // ratio, 1, 1)
            else:
                q_dims = (logical_d, seqlen_q, hq, batch)
                q_strides = (
                    logical_d * elem_bytes,
                    seqlen_q * logical_d * elem_bytes,
                    hq * seqlen_q * logical_d * elem_bytes,
                )
                q_box = (q_band_width, 128, 1, 1)
                o_dims = (logical_dv, seqlen_q, hq, batch)
                o_strides = (
                    logical_dv * elem_bytes,
                    seqlen_q * logical_dv * elem_bytes,
                    hq * seqlen_q * logical_dv * elem_bytes,
                )
                o_box = (o_band_width, 128, 1, 1)

            k_batch_stride = (2**31 if use_i64_kv else hkv * seqlen_kv * logical_d) * elem_bytes
            v_batch_stride = (2**31 if use_i64_kv else hkv * seqlen_kv * logical_dv) * elem_bytes
            k_dims = (logical_d, seqlen_kv, hkv, batch)
            k_strides = (logical_d * elem_bytes, seqlen_kv * logical_d * elem_bytes, k_batch_stride)
            v_dims = (logical_dv, seqlen_kv, hkv, batch)
            v_strides = (
                logical_dv * elem_bytes,
                seqlen_kv * logical_dv * elem_bytes,
                v_batch_stride,
            )

        maps = (
            encode(q, q_dims, q_strides, q_box, q_smem_swizzle),
            encode(
                k,
                k_dims,
                k_strides,
                (q_band_width, 128 // cta_group, *((1,) * (len(k_dims) - 2))),
                q_smem_swizzle,
            ),
            encode(
                v,
                v_dims,
                v_strides,
                (v_load_band_width, 128, *((1,) * (len(v_dims) - 2))),
                v_load_smem_swizzle,
            ),
        )
        if is_varlen:
            return maps
        return (*maps, encode(out, o_dims, o_strides, o_box, o_smem_swizzle))

    def kernel_body(
        q,
        k,
        v,
        out,
        lse,
        partial_count,
        partial_offset,
        partial_index,
        full_count,
        full_offset,
        full_index,
        mask_payload,
        sequence_desc,
        fwd_work_desc,
        softmax_scale_log2,
        host,
    ):
        del q, k, v
        if is_varlen:
            q_map, k_map, v_map = host
        else:
            q_map, k_map, v_map, o_map = host
        if cta_group == 2:
            cluster_rank, _cluster_y, _cluster_z = K.cta_id_in_cluster(
                [2, 1, 1], preferred=[2, 1, 1]
            )
        else:
            cluster_rank, _cluster_y, _cluster_z = K.cta_id_in_cluster([1, 1, 1])
        initial_cta_x, _initial_y, _initial_z = K.cta_id([num_work_records * cta_group, 1, 1])
        work_index = K.local_scalar("int32")
        q_block = K.local_scalar("int32")
        head = K.local_scalar("int32")
        batch_idx = K.local_scalar("int32")
        work_valid = K.local_scalar("int32", init=1)
        clc_consumer_phase = K.local_scalar("int32", init=0)
        warp = K.warp_id()
        lane = K.thread_id() & 31
        tid = K.thread_id()

        def load_work_descriptor(index):
            descriptor_base = index * 4
            K.assign(q_block, _load_i32(fwd_work_desc, descriptor_base))
            K.assign(head, _load_i32(fwd_work_desc, descriptor_base + 1))
            K.assign(batch_idx, _load_i32(fwd_work_desc, descriptor_base + 2))

        with K.If(warp == 0), K.Then():
            K.ptx.prefetch.tensormap(K.address_of(q_map))
            K.ptx.prefetch.tensormap(K.address_of(k_map))
            K.ptx.prefetch.tensormap(K.address_of(v_map))
            if not is_varlen:
                K.ptx.prefetch.tensormap(K.address_of(o_map))

        arena = K.alloc_buffer((smem_bytes,), K.u8, scope="shared.dyn", align=1024)
        smem = K.smem_pool(base=arena)
        pool = smem.pool
        q_full = K.TMABar(pool, q_stages)
        q_empty = K.TCGen05Bar(pool, q_stages)
        kv_full = K.TMABar(pool, kv_stages)
        kv_empty = K.TCGen05Bar(pool, kv_stages)
        if use_smem_mask_pipeline:
            mask0_full = K.TMABar(pool, 1)
            mask0_empty = K.MBarrier(pool, 1)
            mask1_full = K.TMABar(pool, 1)
            mask1_empty = K.MBarrier(pool, 1)
        spo_full = K.TCGen05Bar(pool, 2)
        spo_empty = K.MBarrier(pool, 2)
        plast_full = K.MBarrier(pool, 2)
        plast_empty = K.TCGen05Bar(pool, 2)
        oacc_full = K.TCGen05Bar(pool, 2)
        oacc_empty = K.MBarrier(pool, 2)
        stats_full = K.MBarrier(pool, 2)
        stats_empty = K.MBarrier(pool, 2)
        oepi_full = K.MBarrier(pool, output_stages)
        oepi_empty = K.MBarrier(pool, output_stages)
        if overlap_qo:
            load_epi_full = K.MBarrier(pool, 1)
            load_epi_empty = K.MBarrier(pool, 1)
        tmem_dealloc_barrier = K.MBarrier(pool, 1, leader=(warp == 12) & (lane == 0))
        tmem_mailbox = pool.alloc((1,), "uint32", align=4)
        stats_smem = pool.alloc((512,), "float32", align=4)
        if use_smem_mask_pipeline:
            pool.alloc((4096,), "uint8", align=16)
        clc_full = K.TMABar(pool, 1)
        clc_empty = K.MBarrier(pool, 1)
        clc_response = pool.alloc((4,), "uint32", align=16)
        if pool.offset != offsets["prefix"]:
            raise AssertionError("shared protocol prefix changed")
        pool.alloc((so_offset - pool.offset,), "uint8")
        if pool.offset != so_offset:
            raise AssertionError("shared payload alignment changed")
        pool.alloc((smem_bytes - pool.offset,), "uint8")
        if pool.offset != smem_bytes:
            raise AssertionError("shared arena size changed")

        q_smem = K.decl_buffer(
            (q_stages * 128 * dp,),
            elem_dtype_name,
            data=arena.data,
            byte_offset=offsets["sq"],
            scope="shared.dyn",
            align=1024,
        )
        kv_smem = K.decl_buffer(
            (kv_storage_bytes // 2,),
            elem_dtype_name,
            data=arena.data,
            byte_offset=offsets["skv"],
            scope="shared.dyn",
            align=1024,
        )
        o_smem = K.decl_buffer(
            (output_stages * 128 * dvp,),
            elem_dtype_name,
            data=arena.data,
            byte_offset=offsets["so"],
            scope="shared.dyn",
            align=1024,
        )
        if use_smem_mask_pipeline:
            mask_smem = K.decl_buffer(
                (2 * 128 * 4,),
                "uint32",
                data=arena.data,
                byte_offset=offsets["mask"],
                scope="shared.dyn",
                align=16,
            )

        if cta_group == 2:
            tmem_dealloc_barrier.init(32)
            K.ptx.fence.mbarrier_init.release.cluster()

        q_full.init(1)
        q_empty.init(1)
        kv_full.init(1)
        kv_empty.init(1)
        if use_smem_mask_pipeline:
            mask0_full.init(1)
            mask0_empty.init(4)
            mask1_full.init(1)
            mask1_empty.init(4)
        spo_full.init(1)
        spo_empty.init(256 * cta_group)
        plast_full.init(4 * cta_group)
        plast_empty.init(1)
        oacc_full.init(1)
        oacc_empty.init(128)
        stats_full.init(128)
        stats_empty.init(128)
        oepi_full.init(128)
        oepi_empty.init(32)
        if overlap_qo:
            load_epi_full.init(4 if is_varlen else 1)
            load_epi_empty.init(1)
        K.ptx.fence.mbarrier_init.release.cluster()
        if cta_group == 2:
            K.ptx.barrier.cluster.arrive.relaxed()
            K.ptx.barrier.cluster.wait()
        else:
            K.cuda.cta_sync()

        clc_full.init(1)
        clc_empty.init(512 * cta_group)
        K.ptx.fence.mbarrier_init.release.cluster()
        if cta_group == 2:
            K.ptx.barrier.cluster.arrive.relaxed()
            K.ptx.barrier.cluster.wait()
        else:
            K.cuda.cta_sync()

        raw_count = K.local_scalar("int32")
        partial_n = K.local_scalar("int32")
        full_n = K.local_scalar("int32")
        partial_base = K.local_scalar("int32")
        full_base = K.local_scalar("int32")
        q_offset = K.local_scalar("int32")
        k_offset = K.local_scalar("int32")
        q_length = K.local_scalar("int32")
        k_length = K.local_scalar("int32")
        m_block_offset = K.local_scalar("int32")

        def load_tile_metadata():
            if is_varlen:
                sequence_base = batch_idx * 8
                K.assign(q_offset, _load_i32(sequence_desc, sequence_base))
                K.assign(k_offset, _load_i32(sequence_desc, sequence_base + 1))
                K.assign(q_length, _load_i32(sequence_desc, sequence_base + 2))
                K.assign(k_length, _load_i32(sequence_desc, sequence_base + 3))
                K.assign(m_block_offset, _load_i32(sequence_desc, sequence_base + 4))
            else:
                K.assign(q_offset, 0)
                K.assign(k_offset, 0)
                K.assign(q_length, seqlen_q)
                K.assign(k_length, seqlen_kv)
                K.assign(m_block_offset, batch_idx * q_blocks)
            outer_row = m_block_offset + q_block
            plan_head = K.int32(0) if hmask == 1 else head
            plan_row = plan_head * total_m_blocks + outer_row
            K.assign(partial_n, _load_i32(partial_count, plan_row))
            K.assign(full_n, _load_i32(full_count, plan_row))
            K.assign(partial_base, _load_i32(partial_offset, plan_row))
            K.assign(full_base, _load_i32(full_offset, plan_row))
            K.assign(raw_count, partial_n + full_n)

        def advance_work(load_descriptor=True):
            _wait(clc_full, 0, clc_consumer_phase)
            next_cta_x = K.local_scalar("int32", init=0)
            _query_cancel_response(clc_response, next_cta_x, work_valid)
            K.assign(work_index, next_cta_x // cta_group)
            if load_descriptor:
                with K.If(work_valid != 0), K.Then():
                    load_work_descriptor(work_index)
            clc_empty.arrive(0, remote=0, pred=K.bool(True), count=1)
            K.assign(clc_consumer_phase, clc_consumer_phase ^ 1)

        def q_tile(q_stage):
            return (q_block * q_stages + q_stage) * cta_group + cluster_rank

        def is_partial_item(logical):
            return logical >= full_n

        def is_padding_item(_logical):
            return K.bool(False)

        def sparse_id(logical):
            if direct_full_varlen_ids:
                return logical
            item = logical
            result = K.local_scalar("int32", init=0)
            with K.If(item >= full_n):
                with K.Then():
                    K.assign(result, _load_i32(partial_index, partial_base + item - full_n))
                with K.Else():
                    K.assign(result, _load_i32(full_index, full_base + item))
            return result

        def payload_index(logical):
            return partial_base + logical - full_n

        def advance_ring(stage, phase):
            K.assign(stage, stage + 1)
            with K.If(stage == kv_stages), K.Then():
                K.assign(stage, 0)
                K.assign(phase, phase ^ 1)

        def kv_smem_base(stage, phase, producer=False):
            base = stage * (kv_stage_stride_bytes // 2)
            if uneven_kv:
                effective_phase = phase ^ 1 if producer else phase
                return base + K.if_then_else(
                    stage == 1, uneven_kv_offset * (1 - 2 * effective_phase), 0
                )
            return base

        def tma_completion_barrier(barrier):
            if cta_group == 2:
                mapped = K.local_scalar("uint64")
                K.ptx.mapa.u64(mapped, barrier, K.uint32(0))
                return K.reinterpret("handle", mapped)
            return K.cuda.cvta_generic_to_shared(barrier)

        def tcgen_commit(barrier):
            if cta_group == 2:
                K.ptx[_TCGEN_COMMIT_2CTA](barrier, K.uint16(3))
            else:
                K.ptx[_TCGEN_COMMIT](barrier)

        def arrive_spo(stage):
            spo_empty.arrive(stage, remote=0)

        def arrive_plast(stage):
            plast_full.arrive(stage, remote=0)

        has_work = raw_count > 0
        block_iter_count = raw_count

        roles = K.specialize(chain_dispatch=True)
        r_softmax = roles.role("softmax", warps=range(0, 8), regs=regs_softmax)
        r_correction = roles.role("correction", warps=range(8, 12), regs=regs_correction)
        r_mma = roles.role("mma", warps=[12], regs=regs_other)
        r_epilogue = roles.role("epilogue", warps=[13], regs=regs_other)
        r_load = roles.role("load", warps=[14], regs=regs_other)
        r_idle = roles.role("idle", warps=[15], regs=regs_other)

        with r_idle:
            clc_producer_phase = K.local_scalar("int32", init=1)
            with K.While(work_valid != 0):
                scheduler_cta = cluster_rank == 0 if cta_group == 2 else K.bool(True)
                with K.If(scheduler_cta), K.Then():
                    _wait(clc_empty, 0, clc_producer_phase)
                    with K.If(lane < cta_group), K.Then():
                        mapped_clc_full = K.local_scalar("uint32")
                        K.ptx.mapa.shared__cluster.u32(
                            mapped_clc_full,
                            K.cuda.cvta_generic_to_shared(clc_full.ptr_to([0])),
                            K.cast(lane, "uint32"),
                        )
                        K.ptx.mbarrier.arrive.expect_tx.shared__cluster.b64(
                            mapped_clc_full, K.uint32(16)
                        )
                    with K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx[
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        ](K.address_of(clc_response[0]), K.address_of(clc_full.buf[0]))
                    K.assign(clc_producer_phase, clc_producer_phase ^ 1)
                advance_work(load_descriptor=False)
            with K.If(cluster_rank == 0 if cta_group == 2 else K.bool(True)), K.Then():
                _wait(clc_empty, 0, clc_producer_phase)

        with r_load:
            load_work_descriptor(initial_cta_x // cta_group)
            q_producer_phase = K.local_scalar("int32", init=1)
            if overlap_qo:
                load_epi_consumer_phase = K.local_scalar("int32", init=0)
            kv_stage = K.local_scalar("int32", init=0)
            kv_phase = K.local_scalar("int32", init=1)
            if use_smem_mask_pipeline:
                mask_producer_phase = K.alloc_local((2,), "int32")
                K.assign(mask_producer_phase[0], 1)
                K.assign(mask_producer_phase[1], 1)

            def load_mask(logical, stream):
                if use_smem_mask_pipeline:
                    mask_full_cur = mask0_full if stream == 0 else mask1_full
                    mask_empty_cur = mask0_empty if stream == 0 else mask1_empty
                    mask_phase = mask_producer_phase[stream]
                    padding = is_padding_item(logical)
                    is_partial = is_partial_item(logical)
                    payload = payload_index(logical)
                    with K.If(is_partial), K.Then():
                        _wait(mask_empty_cur, 0, mask_phase)
                        with K.If(padding):
                            with K.Then():
                                with K.unroll(4) as row_group:
                                    K.ptx.st.shared.v4.u32(
                                        mask_smem.ptr_to(
                                            [stream * 128 * 4 + row_group * 128 + lane * 4]
                                        ),
                                        K.uint32(0),
                                        K.uint32(0),
                                        K.uint32(0),
                                        K.uint32(0),
                                    )
                                K.ptx.fence.proxy.async_.shared__cta()
                                with K.If(K.cuda.elect_sync()), K.Then():
                                    mask_full_cur.arrive(0)
                            with K.Else():
                                mask_leader = K.cuda.elect_sync()
                                with K.If(mask_leader), K.Then():
                                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                        mask_full_cur.ptr_to([0]), K.uint32(2048)
                                    )
                                    K.ptx[_BULK_G2S_MASK](
                                        mask_smem.ptr_to([stream * 128 * 4]),
                                        mask_payload.ptr_to(
                                            [(payload * work_q_tiles + cluster_rank) * 128 * 4]
                                        ),
                                        K.uint32(2048),
                                        K.cuda.cvta_generic_to_shared(mask_full_cur.ptr_to([0])),
                                    )
                        K.assign(mask_producer_phase[stream], mask_producer_phase[stream] ^ 1)

            def load_kv(kind, logical, do_advance=True):
                tma_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                barrier_leader = (tma_leader != K.uint32(0)) & (
                    cluster_rank == 0 if cta_group == 2 else K.bool(True)
                )
                _wait(kv_empty, kv_stage, kv_phase)
                if uneven_kv and kind == 0:
                    with K.If(kv_stage == 0), K.Then():
                        _wait(kv_empty, 1, kv_phase)
                sid = sparse_id(logical)
                if kind == 0:
                    transfer_width = dp
                    transfer_rows = 128 // cta_group
                    transfer_bands = q_num_bands
                    transfer_band_width = q_band_width
                    descriptor = K.address_of(k_map)
                else:
                    transfer_width = dvp // cta_group
                    transfer_rows = 128
                    transfer_bands = v_load_num_bands
                    transfer_band_width = v_load_band_width
                    descriptor = K.address_of(v_map)
                K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                    kv_full.ptr_to([kv_stage]),
                    K.uint32(transfer_rows * transfer_width * 2 * cta_group),
                    pred=barrier_leader,
                )
                with K.unroll(transfer_bands) as band:
                    destination = (
                        kv_smem_base(kv_stage, kv_phase, producer=True)
                        + band * transfer_rows * transfer_band_width
                    )
                    if kind == 0:
                        coord_d = band * transfer_band_width
                        coord_n = k_offset + sid * 128 + cluster_rank * transfer_rows
                    else:
                        coord_d = cluster_rank * transfer_width + band * transfer_band_width
                        coord_n = k_offset + sid * 128
                    if is_varlen:
                        K.ptx[tma_g2s_3d](
                            kv_smem.ptr_to([destination]),
                            descriptor,
                            K.cast(coord_d, "int32"),
                            coord_n,
                            head if pack else head // ratio,
                            tma_completion_barrier(kv_full.ptr_to([kv_stage])),
                            _TMA_CACHE,
                            pred=tma_leader,
                        )
                    else:
                        K.ptx[tma_g2s_4d](
                            kv_smem.ptr_to([destination]),
                            descriptor,
                            K.cast(coord_d, "int32"),
                            coord_n,
                            head if pack else head // ratio,
                            batch_idx,
                            tma_completion_barrier(kv_full.ptr_to([kv_stage])),
                            _TMA_CACHE,
                            pred=tma_leader,
                        )
                if do_advance:
                    advance_ring(kv_stage, kv_phase)

            def prefetch_partial_index(ordinal):
                with K.If(K.cuda.elect_sync()), K.Then():
                    K.ptx.prefetch.global_.L1(partial_index.ptr_to([partial_base + ordinal]))

            def prefetch_full_index(ordinal):
                with K.If(K.cuda.elect_sync()), K.Then():
                    K.ptx.prefetch.global_.L1(full_index.ptr_to([full_base + ordinal]))

            def prefetch_partial_mask(ordinal, q_stage):
                # One lane per 128-byte cache line covers the 2-KiB
                # consumer-native payload subtile, matching the source
                # packed-mask helper.
                with K.If(lane < K.int32(16)), K.Then():
                    K.ptx.prefetch.global_.L1(
                        mask_payload.ptr_to(
                            [
                                ((partial_base + ordinal) * work_q_tiles + q_stage) * 128 * 4
                                + lane * 32
                            ]
                        )
                    )

            def load_q(q_stage):
                _wait(q_empty, q_stage, q_producer_phase)
                q_tma_leader = K.cuda.elect_sync()
                q_barrier_leader = (q_tma_leader != K.uint32(0)) & (
                    cluster_rank == 0 if cta_group == 2 else K.bool(True)
                )
                with K.If(q_barrier_leader), K.Then():
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        q_full.ptr_to([q_stage]), K.uint32(128 * dp * 2 * cta_group)
                    )
                with K.If(q_tma_leader), K.Then():
                    with K.unroll(q_num_bands) as band:
                        destination = q_stage * 128 * dp + band * 128 * q_band_width
                        packed_row = q_tile(q_stage) * 128
                        if pack:
                            if is_varlen:
                                K.ptx[tma_g2s_4d](
                                    q_smem.ptr_to([destination]),
                                    K.address_of(q_map),
                                    K.cast(band * q_band_width, "int32"),
                                    K.int32(0),
                                    q_offset + packed_row // ratio,
                                    head,
                                    tma_completion_barrier(q_full.ptr_to([q_stage])),
                                    _TMA_CACHE,
                                )
                            else:
                                K.ptx[tma_g2s_5d](
                                    q_smem.ptr_to([destination]),
                                    K.address_of(q_map),
                                    K.cast(band * q_band_width, "int32"),
                                    K.int32(0),
                                    packed_row // ratio,
                                    head,
                                    batch_idx,
                                    tma_completion_barrier(q_full.ptr_to([q_stage])),
                                    _TMA_CACHE,
                                )
                        else:
                            if is_varlen:
                                K.ptx[tma_g2s_3d](
                                    q_smem.ptr_to([destination]),
                                    K.address_of(q_map),
                                    K.cast(band * q_band_width, "int32"),
                                    q_offset + packed_row,
                                    head,
                                    tma_completion_barrier(q_full.ptr_to([q_stage])),
                                    _TMA_CACHE,
                                )
                            else:
                                K.ptx[tma_g2s_4d](
                                    q_smem.ptr_to([destination]),
                                    K.address_of(q_map),
                                    K.cast(band * q_band_width, "int32"),
                                    packed_row,
                                    head,
                                    batch_idx,
                                    tma_completion_barrier(q_full.ptr_to([q_stage])),
                                    _TMA_CACHE,
                                )

            with K.While(work_valid != 0):
                load_tile_metadata()
                with K.If(has_work), K.Then():
                    for q_stage in range(q_stages):
                        load_q(q_stage)
                    K.assign(q_producer_phase, q_producer_phase ^ 1)
                    if variant == "qstage2_1cta":
                        # Preserve the source two-list traversal and its
                        # current/next indirect-index and mask prefetches.
                        with K.If(partial_n > 0), K.Then():
                            prefetch_partial_index(partial_n - 1)
                            for q_stage in range(2):
                                prefetch_partial_mask(partial_n - 1, q_stage)
                            offset = K.local_scalar("int32", init=0)
                            with K.While(offset < partial_n):
                                ordinal = partial_n - 1 - offset
                                with K.If(offset + 1 < partial_n), K.Then():
                                    prefetch_partial_index(ordinal - 1)
                                    for q_stage in range(2):
                                        prefetch_partial_mask(ordinal - 1, q_stage)
                                logical = full_n + ordinal
                                load_kv(0, logical)
                                load_kv(1, logical)
                                K.assign(offset, offset + 1)
                        with K.If(full_n > 0), K.Then():
                            prefetch_full_index(full_n - 1)
                            offset = K.local_scalar("int32", init=0)
                            with K.While(offset < full_n):
                                ordinal = full_n - 1 - offset
                                with K.If(offset + 1 < full_n), K.Then():
                                    prefetch_full_index(ordinal - 1)
                                load_kv(0, ordinal)
                                load_kv(1, ordinal)
                                K.assign(offset, offset + 1)
                    else:
                        if use_smem_mask_pipeline:
                            load_mask(block_iter_count - 1, 0)
                        load_kv(0, block_iter_count - 1, False)
                        advance_ring(kv_stage, kv_phase)
                        with K.If(block_iter_count == 1), K.Then():
                            load_kv(1, 0)
                        with K.If(block_iter_count > 1), K.Then():
                            if use_smem_mask_pipeline:
                                load_mask(block_iter_count - 2, 1)
                            load_kv(0, block_iter_count - 2)
                            i = K.local_scalar("int32", init=0)
                            with K.While(i < block_iter_count - 2):
                                load_kv(1, block_iter_count - 1 - i)
                                if use_smem_mask_pipeline:
                                    with K.If((i & 1) == 0), K.Then():
                                        load_mask(block_iter_count - 3 - i, 0)
                                    with K.If((i & 1) != 0), K.Then():
                                        load_mask(block_iter_count - 3 - i, 1)
                                load_kv(0, block_iter_count - 3 - i)
                                K.assign(i, i + 1)
                            load_kv(1, 1)
                            load_kv(1, 0)
                advance_work()
                if overlap_qo:
                    _wait(load_epi_full, 0, load_epi_consumer_phase)
                    with K.If(K.cuda.elect_sync()), K.Then():
                        load_epi_empty.arrive(0)
                    K.assign(load_epi_consumer_phase, load_epi_consumer_phase ^ 1)
            tail_stage = K.local_scalar("int32", init=kv_stage)
            tail_phase = K.local_scalar("int32", init=kv_phase)
            with K.unroll(kv_stages) as _tail:
                _wait(kv_empty, tail_stage, tail_phase)
                advance_ring(tail_stage, tail_phase)
            if use_smem_mask_pipeline:
                _wait(mask0_empty, 0, mask_producer_phase[0])
                _wait(mask1_empty, 0, mask_producer_phase[1])
            _wait(q_empty, q_stages - 1, q_producer_phase)

        with r_mma:
            load_work_descriptor(initial_cta_x // cta_group)
            K.ptx[tmem_alloc](K.address_of(tmem_mailbox[0]), K.uint32(_TMEM_COLS))
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            tmem_base = K.local_scalar("uint32", init=K.uint32(0))
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            q_consumer_phase = K.local_scalar("int32", init=0)
            kv_stage = K.local_scalar("int32", init=0)
            kv_phase = K.local_scalar("int32", init=0)
            spo_phase0 = K.local_scalar("int32", init=0)
            spo_phase1 = K.local_scalar("int32", init=0)
            acc0 = K.local_scalar("int32", init=0)
            acc1 = K.local_scalar("int32", init=0)

            def descriptor_offset(k16, rows):
                per_band = q_band_width // 16
                return (k16 % per_band) * 2 + (k16 // per_band) * (rows * q_band_width * 2 // 16)

            def issue_qk(score_stage, k_stage, k_phase):
                q_desc = K.SmemDescriptor()
                q_desc.init(
                    q_smem.ptr_to([(score_stage if variant == "qstage2_1cta" else 0) * 128 * dp]),
                    ldo=1 if dp % 64 else 1024,
                    sdo=q_band_width,
                    swizzle=q_smem_swizzle,
                )
                q_desc.make_lo_uniform()
                k_desc = K.SmemDescriptor()
                k_desc.init(
                    kv_smem.ptr_to([kv_smem_base(k_stage, k_phase)]),
                    ldo=1 if dp % 64 else 1024,
                    sdo=q_band_width,
                    swizzle=q_smem_swizzle,
                )
                k_desc.make_lo_uniform()
                for k16 in range(dp // 16):
                    with K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx[mma_f16](
                            K.cast(tmem_base + score_stage * 128, "uint32"),
                            q_desc.add_16B_offset(descriptor_offset(k16, 128)),
                            k_desc.add_16B_offset(descriptor_offset(k16, 128 // cta_group)),
                            K.uint32(qk_idesc),
                            *mma_disable_lanes,
                            K.cast(k16 != 0, "bool"),
                        )

            def issue_pv(score_stage, v_stage, v_phase, accumulate, spo_phase):
                v_desc = K.SmemDescriptor()
                v_desc.init(
                    kv_smem.ptr_to([kv_smem_base(v_stage, v_phase)]),
                    ldo=(v_load_band_width if v_load_smem_swizzle == 0 else 16 * v_load_band_width),
                    sdo=(0 if v_load_smem_swizzle == 0 else v_load_band_width),
                    swizzle=v_load_smem_swizzle,
                )
                v_desc.make_lo_uniform()
                for k16 in range(8):
                    if k16 == 6:
                        _wait(plast_full, score_stage, spo_phase)
                    with K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx[mma_f16](
                            K.cast(tmem_base + 256 + score_stage * dvp, "uint32"),
                            K.cast(tmem_base + 64 + score_stage * 128 + k16 * 8, "uint32"),
                            v_desc.add_16B_offset(k16 * 2 * v_load_band_width),
                            K.uint32(pv_idesc),
                            *mma_disable_lanes,
                            K.cast(accumulate != 0 if k16 == 0 else True, "bool"),
                        )

            with K.While(work_valid != 0):
                load_tile_metadata()
                K.assign(acc0, 0)
                K.assign(acc1, 0)
                if variant == "qstage2_1cta":
                    with K.If(has_work), K.Then():
                        for score_stage in range(2):
                            _wait(q_full, score_stage, q_consumer_phase)
                        K.assign(q_consumer_phase, q_consumer_phase ^ 1)

                        _wait(kv_full, kv_stage, kv_phase)
                        first_k_stage = K.local_scalar("int32", init=kv_stage)
                        first_k_phase = K.local_scalar("int32", init=kv_phase)
                        for score_stage in range(2):
                            issue_qk(score_stage, first_k_stage, first_k_phase)
                            with K.If(K.cuda.elect_sync()), K.Then():
                                tcgen_commit(spo_full.ptr_to([score_stage]))
                        with K.If(K.cuda.elect_sync()), K.Then():
                            tcgen_commit(kv_empty.ptr_to([first_k_stage]))
                        advance_ring(kv_stage, kv_phase)

                        block = K.local_scalar("int32", init=1)
                        with K.While(block < block_iter_count):
                            _wait(kv_full, kv_stage, kv_phase)
                            v_release_stage = K.local_scalar("int32", init=kv_stage)
                            v_release_phase = K.local_scalar("int32", init=kv_phase)
                            k_release_stage = K.local_scalar("int32", init=0)
                            k_release_phase = K.local_scalar("int32", init=0)
                            for score_stage in range(2):
                                _wait(spo_empty, score_stage, spo_phase0)
                                issue_pv(
                                    score_stage, v_release_stage, v_release_phase, acc0, spo_phase0
                                )
                                if score_stage == 1:
                                    with K.If(K.cuda.elect_sync()), K.Then():
                                        tcgen_commit(kv_empty.ptr_to([v_release_stage]))
                                if score_stage == 0:
                                    advance_ring(kv_stage, kv_phase)
                                    _wait(kv_full, kv_stage, kv_phase)
                                    K.assign(k_release_stage, kv_stage)
                                    K.assign(k_release_phase, kv_phase)
                                issue_qk(score_stage, k_release_stage, k_release_phase)
                                with K.If(K.cuda.elect_sync()), K.Then():
                                    tcgen_commit(spo_full.ptr_to([score_stage]))
                            with K.If(K.cuda.elect_sync()), K.Then():
                                tcgen_commit(kv_empty.ptr_to([k_release_stage]))
                            advance_ring(kv_stage, kv_phase)
                            K.assign(spo_phase0, spo_phase0 ^ 1)
                            K.assign(acc0, 1)
                            K.assign(block, block + 1)

                        for score_stage in range(2):
                            with K.If(K.cuda.elect_sync()), K.Then():
                                tcgen_commit(q_empty.ptr_to([score_stage]))
                        _wait(kv_full, kv_stage, kv_phase)
                        for score_stage in range(2):
                            _wait(spo_empty, score_stage, spo_phase0)
                            issue_pv(score_stage, kv_stage, kv_phase, acc0, spo_phase0)
                            with K.If(K.cuda.elect_sync()), K.Then():
                                tcgen_commit(oacc_full.ptr_to([score_stage]))
                        with K.If(K.cuda.elect_sync()), K.Then():
                            tcgen_commit(kv_empty.ptr_to([kv_stage]))
                        advance_ring(kv_stage, kv_phase)
                        K.assign(spo_phase0, spo_phase0 ^ 1)
                else:
                    K.assign(acc0, 0)
                    K.assign(acc1, 0)
                    mma_has_work = has_work & (cluster_rank == 0) if cta_group == 2 else has_work
                    with K.If(mma_has_work), K.Then():
                        _wait(q_full, 0, q_consumer_phase)
                        K.assign(q_consumer_phase, q_consumer_phase ^ 1)

                        def initial_qk(score_stage):
                            _wait(kv_full, kv_stage, kv_phase)
                            issue_qk(score_stage, kv_stage, kv_phase)
                            with K.If(K.cuda.elect_sync()), K.Then():
                                tcgen_commit(spo_full.ptr_to([score_stage]))
                                tcgen_commit(kv_empty.ptr_to([kv_stage]))
                            advance_ring(kv_stage, kv_phase)

                        initial_qk(0)
                        with K.If(block_iter_count > 1), K.Then():
                            initial_qk(1)

                        middle_count = K.max(block_iter_count - 2, 0)

                        def middle_step(score_stage):
                            phase = spo_phase0 if score_stage == 0 else spo_phase1
                            accumulated = acc0 if score_stage == 0 else acc1
                            _wait(kv_full, kv_stage, kv_phase)
                            v_release_stage = K.local_scalar("int32", init=kv_stage)
                            v_release_phase = K.local_scalar("int32", init=kv_phase)
                            advance_ring(kv_stage, kv_phase)
                            if not overlap_pv_with_k_wait:
                                _wait(kv_full, kv_stage, kv_phase)
                            _wait(spo_empty, score_stage, phase)
                            issue_pv(
                                score_stage, v_release_stage, v_release_phase, accumulated, phase
                            )
                            if overlap_pv_with_k_wait:
                                with K.If(K.cuda.elect_sync()), K.Then():
                                    tcgen_commit(kv_empty.ptr_to([v_release_stage]))
                                _wait(kv_full, kv_stage, kv_phase)
                            issue_qk(score_stage, kv_stage, kv_phase)
                            with K.If(K.cuda.elect_sync()), K.Then():
                                tcgen_commit(spo_full.ptr_to([score_stage]))
                                if not overlap_pv_with_k_wait:
                                    tcgen_commit(kv_empty.ptr_to([v_release_stage]))
                                tcgen_commit(kv_empty.ptr_to([kv_stage]))
                            if score_stage == 0:
                                K.assign(spo_phase0, spo_phase0 ^ 1)
                                K.assign(acc0, 1)
                            else:
                                K.assign(spo_phase1, spo_phase1 ^ 1)
                                K.assign(acc1, 1)
                            advance_ring(kv_stage, kv_phase)

                        pair_count = middle_count // 2
                        pair = K.local_scalar("int32", init=0)
                        with K.While(pair < pair_count):
                            middle_step(0)
                            middle_step(1)
                            K.assign(pair, pair + 1)
                        with K.If((middle_count & 1) != 0), K.Then():
                            middle_step(0)

                        with K.If(K.cuda.elect_sync()), K.Then():
                            tcgen_commit(q_empty.ptr_to([0]))

                        epilogue_count = K.min(block_iter_count, 2)
                        epilogue_ordinal = K.local_scalar(
                            "int32", init=block_iter_count - epilogue_count
                        )
                        with K.While(epilogue_ordinal < block_iter_count):
                            score_stage = epilogue_ordinal & 1
                            phase = K.if_then_else(score_stage == 0, spo_phase0, spo_phase1)
                            accumulated = K.if_then_else(score_stage == 0, acc0, acc1)
                            _wait(kv_full, kv_stage, kv_phase)
                            _wait(spo_empty, score_stage, phase)
                            issue_pv(score_stage, kv_stage, kv_phase, accumulated, phase)
                            with K.If(K.cuda.elect_sync()), K.Then():
                                tcgen_commit(oacc_full.ptr_to([score_stage]))
                                tcgen_commit(kv_empty.ptr_to([kv_stage]))
                            advance_ring(kv_stage, kv_phase)
                            with K.If(score_stage == 0), K.Then():
                                K.assign(spo_phase0, spo_phase0 ^ 1)
                            with K.If(score_stage != 0), K.Then():
                                K.assign(spo_phase1, spo_phase1 ^ 1)
                            K.assign(epilogue_ordinal, epilogue_ordinal + 1)
                advance_work()

            K.ptx[tmem_relinquish]()
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            allocated = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(allocated, tmem_mailbox.ptr_to([0]))
            if cta_group == 2:
                remote_dealloc = K.local_scalar("uint32")
                K.ptx.mapa.shared__cluster.u32(
                    remote_dealloc,
                    K.cuda.cvta_generic_to_shared(tmem_dealloc_barrier.ptr_to([0])),
                    K.cast(cluster_rank ^ 1, "uint32"),
                )
                K.ptx.mbarrier.arrive.shared__cluster.b64(remote_dealloc)
                _wait(tmem_dealloc_barrier, 0, 0)
            K.ptx[tmem_dealloc](allocated, K.uint32(_TMEM_COLS))

        with r_epilogue:
            if not is_varlen:
                load_work_descriptor(initial_cta_x // cta_group)
            oepi_consumer_phase = K.local_scalar("int32", init=0)
            if overlap_qo:
                load_epi_producer_phase = K.local_scalar("int32", init=1)
            output_stage = K.local_scalar("int32", init=0)
            with K.While(work_valid != 0):
                if not is_varlen:
                    load_tile_metadata()
                    if variant == "qstage2_1cta":
                        for output_stage_static in range(2):
                            _wait(oepi_full, output_stage_static, oepi_consumer_phase)
                            with K.unroll(o_num_bands) as band:
                                packed_row = q_tile(output_stage_static) * 128
                                source_offset = (
                                    output_stage_static * 128 * dvp + band * 128 * o_band_width
                                )
                                if pack:
                                    K.ptx[_TMA_S2G_5D](
                                        K.address_of(o_map),
                                        K.cast(band * o_band_width, "int32"),
                                        K.int32(0),
                                        packed_row // ratio,
                                        head,
                                        batch_idx,
                                        o_smem.ptr_to([source_offset]),
                                        _TMA_CACHE,
                                    )
                                else:
                                    K.ptx[_TMA_S2G_4D](
                                        K.address_of(o_map),
                                        K.cast(band * o_band_width, "int32"),
                                        packed_row,
                                        head,
                                        batch_idx,
                                        o_smem.ptr_to([source_offset]),
                                        _TMA_CACHE,
                                    )
                            K.ptx.cp.async_.bulk.commit_group()
                        for output_stage_static in range(2):
                            K.ptx.cp.async_.bulk.wait_group.read(1 - output_stage_static)
                            oepi_empty.arrive(output_stage_static)
                        K.assign(oepi_consumer_phase, oepi_consumer_phase ^ 1)
                    else:
                        _wait(oepi_full, output_stage, oepi_consumer_phase)
                        with K.unroll(o_num_bands) as band:
                            source_offset = output_stage * 128 * dvp + band * 128 * o_band_width
                            if pack:
                                K.ptx[_TMA_S2G_5D](
                                    K.address_of(o_map),
                                    K.cast(band * o_band_width, "int32"),
                                    K.int32(0),
                                    (q_tile(0) * 128) // ratio,
                                    head,
                                    batch_idx,
                                    o_smem.ptr_to([source_offset]),
                                    _TMA_CACHE,
                                )
                            else:
                                K.ptx[_TMA_S2G_4D](
                                    K.address_of(o_map),
                                    K.cast(band * o_band_width, "int32"),
                                    q_tile(0) * 128,
                                    head,
                                    batch_idx,
                                    o_smem.ptr_to([source_offset]),
                                    _TMA_CACHE,
                                )
                        K.ptx.cp.async_.bulk.commit_group()
                        K.ptx.cp.async_.bulk.wait_group.read(0)
                        oepi_empty.arrive(output_stage)
                        K.assign(output_stage, output_stage + 1)
                        with K.If(output_stage == output_stages), K.Then():
                            K.assign(output_stage, 0)
                            K.assign(oepi_consumer_phase, oepi_consumer_phase ^ 1)
                if overlap_qo and not is_varlen:
                    _wait(load_epi_empty, 0, load_epi_producer_phase)
                    with K.If(K.cuda.elect_sync()), K.Then():
                        load_epi_full.arrive(0)
                    K.assign(load_epi_producer_phase, load_epi_producer_phase ^ 1)
                advance_work(load_descriptor=not is_varlen)

        with r_softmax:
            load_work_descriptor(initial_cta_x // cta_group)
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            tmem_base = K.local_scalar("uint32", init=K.uint32(0))
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            score_stage = K.if_then_else(warp < 4, 0, 1)
            local_warp = warp & 3
            tid128 = tid & 127
            row_hi = K.shift_left(K.cast(local_warp * 32, "uint32"), K.uint32(16))
            score_phase = K.local_scalar("int32", init=0)
            stats_producer_phase = K.local_scalar("int32", init=1)
            if cta_group == 2:
                mask_consumer_phase = K.local_scalar("int32", init=0)
            row_max = K.local_scalar("float32", init=K.float32(_NEG_INF))
            row_sum = K.local_scalar("float32", init=K.float32(0.0))
            with K.While(work_valid != 0):
                load_tile_metadata()
                K.assign(row_max, K.float32(_NEG_INF))
                K.assign(row_sum, K.float32(0.0))
                _wait(stats_empty, score_stage, stats_producer_phase)
                K.assign(stats_producer_phase, stats_producer_phase ^ 1)
                stream_count = (
                    block_iter_count
                    if variant == "qstage2_1cta"
                    else (block_iter_count + 1 - score_stage) // 2
                )
                with K.If(stream_count > 0), K.Then():

                    def consume_score(logical, first, partial=None):
                        is_partial = (
                            K.bool(partial) if partial is not None else is_partial_item(logical)
                        )
                        padding = is_padding_item(logical)
                        payload = payload_index(logical)
                        words = K.alloc_local((4,), "uint32")
                        # Each consumer reads its 16-byte row payload directly.
                        # For group-2 MMA, cluster rank selects the matching
                        # consumer-native payload subtile.
                        with K.If(is_partial), K.Then():
                            if use_smem_mask_pipeline:
                                with K.If(score_stage == 0), K.Then():
                                    _wait(mask0_full, 0, mask_consumer_phase)
                                with K.If(score_stage != 0), K.Then():
                                    _wait(mask1_full, 0, mask_consumer_phase)
                                K.ptx.ld.shared.v4.u32(
                                    words[0],
                                    words[1],
                                    words[2],
                                    words[3],
                                    mask_smem.ptr_to([score_stage * 128 * 4 + tid128 * 4]),
                                )
                                with K.If(K.cuda.elect_sync()), K.Then():
                                    with K.If(score_stage == 0), K.Then():
                                        mask0_empty.arrive(0)
                                    with K.If(score_stage != 0), K.Then():
                                        mask1_empty.arrive(0)
                                K.assign(mask_consumer_phase, mask_consumer_phase ^ 1)
                            else:
                                with K.If(padding):
                                    with K.Then():
                                        with K.unroll(4) as quarter:
                                            K.assign(words[quarter], K.uint32(0))
                                    with K.Else():
                                        K.ptx.ld.global_.v4.b32(
                                            words[0],
                                            words[1],
                                            words[2],
                                            words[3],
                                            mask_payload.ptr_to(
                                                [
                                                    (
                                                        payload * work_q_tiles
                                                        + (
                                                            cluster_rank
                                                            if cta_group == 2
                                                            else (
                                                                score_stage
                                                                if variant == "qstage2_1cta"
                                                                else 0
                                                            )
                                                        )
                                                    )
                                                    * 128
                                                    * 4
                                                    + tid128 * 4
                                                ]
                                            ),
                                        )

                        _wait(spo_full, score_stage, score_phase)
                        score = K.alloc_local((128,), "float32")
                        reduced = K.alloc_local((4,), "float32")
                        software_max = K.bool(True) if variant == "qstage2_1cta" else is_partial
                        with K.If(software_max):
                            with K.Then():
                                with K.unroll(4) as chunk:
                                    regs = K.alloc_local((32,), "float32")
                                    _tmem_load32(
                                        regs, tmem_base + score_stage * 128 + chunk * 32 + row_hi
                                    )
                                    with K.unroll(32) as j:
                                        K.assign(score[chunk * 32 + j], regs[j])
                                with K.If(is_partial), K.Then():
                                    with K.unroll(4) as quarter:
                                        with K.unroll(32) as bit:
                                            index = quarter * 32 + bit
                                            live = (
                                                K.bitwise_and(
                                                    words[quarter],
                                                    K.shift_left(
                                                        K.uint32(1), K.cast(bit, "uint32")
                                                    ),
                                                )
                                                != 0
                                            )
                                            K.assign(
                                                score[index],
                                                K.if_then_else(
                                                    live, score[index], K.float32(_NEG_INF)
                                                ),
                                            )
                            with K.Else():
                                with K.unroll(4) as chunk:
                                    regs = K.alloc_local((32,), "float32")
                                    _tmem_load32_red_max(
                                        regs,
                                        reduced[chunk],
                                        tmem_base + score_stage * 128 + chunk * 32 + row_hi,
                                    )
                                    with K.unroll(32) as j:
                                        K.assign(score[chunk * 32 + j], regs[j])

                        tile_max = K.local_scalar("float32")
                        with K.If(software_max):
                            with K.Then():
                                K.assign(tile_max, _reduce_max_128(score))
                            with K.Else():
                                K.ptx.max.f32(tile_max, reduced[0], reduced[1])
                                K.ptx.max.f32(tile_max, tile_max, reduced[2], reduced[3])

                        old_scale = K.local_scalar("float32", init=K.float32(0.0))
                        new_max = K.local_scalar("float32")
                        max_safe = K.local_scalar("float32")
                        if first:
                            K.assign(new_max, tile_max)
                            K.assign(
                                max_safe,
                                K.if_then_else(tile_max != K.float32(_NEG_INF), tile_max, 0.0),
                            )
                        else:
                            K.ptx.max.f32(new_max, row_max, tile_max)
                            K.assign(
                                max_safe,
                                K.if_then_else(new_max != K.float32(_NEG_INF), new_max, 0.0),
                            )
                            delta = K.local_scalar("float32")
                            K.ptx.sub.f32(delta, row_max, max_safe)
                            delta_scaled = K.local_scalar("float32")
                            K.ptx.mul.f32(delta_scaled, delta, softmax_scale_log2)
                            with K.If(software_max):
                                with K.Then():
                                    K.assign(old_scale, _exp2(delta_scaled))
                                with K.Else():
                                    K.assign(old_scale, _exp2_nonfast(delta_scaled))
                            with K.If(delta_scaled >= K.float32(-8.0)), K.Then():
                                K.assign(new_max, row_max)
                                K.assign(max_safe, row_max)
                                K.assign(old_scale, K.float32(1.0))
                            _st_shared_f32(stats_smem, score_stage * 128 + tid128, old_scale)
                        _stats_arrive(score_stage, local_warp)
                        negative_max = K.local_scalar("float32")
                        K.ptx.mul.f32(negative_max, max_safe, -softmax_scale_log2)
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
                                negative_max,
                                negative_max,
                            )

                        for fragment in range(4):
                            for pair in range(16):
                                base = fragment * 32 + pair * 2
                                K.assign(score[base], _exp2(score[base]))
                                K.assign(score[base + 1], _exp2(score[base + 1]))
                            packed_p = K.alloc_local((16,), "uint32")
                            with K.unroll(16) as pair:
                                base = fragment * 32 + pair * 2
                                if dtype == "bfloat16":
                                    K.ptx.cvt.rn.bf16x2.f32(
                                        packed_p[pair], score[base + 1], score[base]
                                    )
                                else:
                                    K.ptx.cvt.rn.f16x2.f32(
                                        packed_p[pair], score[base + 1], score[base]
                                    )
                            _tmem_store16(
                                packed_p,
                                tmem_base + 64 + score_stage * 128 + fragment * 16 + row_hi,
                            )
                            if fragment == 2:
                                K.ptx.tcgen05.wait__st.sync.aligned()
                                arrive_spo(score_stage)
                        K.ptx.tcgen05.wait__st.sync.aligned()
                        K.cuda.warp_sync()
                        with K.If(K.cuda.elect_sync()), K.Then():
                            arrive_plast(score_stage)

                        _wait(stats_empty, score_stage, stats_producer_phase)
                        K.assign(row_sum, _packed_sum_128(score, row_sum, old_scale, first))
                        K.assign(row_max, new_max)
                        K.assign(stats_producer_phase, stats_producer_phase ^ 1)
                        K.assign(score_phase, score_phase ^ 1)

                    if variant == "qstage2_1cta":
                        with K.If(partial_n > 0), K.Then():
                            consume_score(full_n + partial_n - 1, True, True)
                            iteration = K.local_scalar("int32", init=1)
                            with K.While(iteration < partial_n):
                                consume_score(full_n + partial_n - 1 - iteration, False, True)
                                K.assign(iteration, iteration + 1)
                        with K.If(full_n > 0), K.Then():
                            iteration = K.local_scalar("int32", init=0)
                            with K.If(partial_n == 0), K.Then():
                                consume_score(full_n - 1, True, False)
                                K.assign(iteration, 1)
                            with K.While(iteration < full_n):
                                consume_score(full_n - 1 - iteration, False, False)
                                K.assign(iteration, iteration + 1)
                    else:
                        consume_score(block_iter_count - 1 - score_stage, True)
                        iteration = K.local_scalar("int32", init=1)
                        with K.While(iteration < stream_count):
                            consume_score(
                                block_iter_count - 1 - (iteration * 2 + score_stage), False
                            )
                            K.assign(iteration, iteration + 1)

                    _st_shared_f32(stats_smem, score_stage * 128 + tid128, row_sum)
                    if return_lse or variant != "qstage2_1cta":
                        _st_shared_f32(stats_smem, 256 + score_stage * 128 + tid128, row_max)
                    _stats_arrive(score_stage, local_warp)
                with K.If(stream_count == 0), K.Then():
                    _stats_arrive(score_stage, local_warp)
                advance_work()
            _wait(stats_empty, score_stage, stats_producer_phase)
            K.ptx.bar.arrive(K.uint32(2), K.uint32(416))

        with r_correction:
            load_work_descriptor(initial_cta_x // cta_group)
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            tmem_base = K.local_scalar("uint32", init=K.uint32(0))
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            corr_warp = warp - 8
            tid128 = tid & 127
            row_hi = K.shift_left(K.cast(corr_warp * 32, "uint32"), K.uint32(16))
            oacc_consumer_phase = K.local_scalar("int32", init=0)
            qstage1_o_phase0 = K.local_scalar("int32", init=0)
            qstage1_o_phase1 = K.local_scalar("int32", init=0)
            oepi_producer_phase = K.local_scalar("int32", init=1)
            if overlap_qo:
                load_epi_producer_phase = K.local_scalar("int32", init=1)
            output_stage = K.local_scalar("int32", init=0)
            arrive_spo(0)
            arrive_spo(1)

            def rescale_o(score_stage, scale):
                with K.unroll(dvp // 16) as chunk:
                    values = K.alloc_local((16,), "float32")
                    _tmem_load16(values, tmem_base + 256 + score_stage * dvp + chunk * 16 + row_hi)
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
                    _tmem_store16(values, tmem_base + 256 + score_stage * dvp + chunk * 16 + row_hi)
                K.ptx.tcgen05.wait__st.sync.aligned()

            def output_addresses(output_stage, chunk):
                chunks_per_band = o_band_width // 16
                band = chunk // chunks_per_band
                local_chunk = chunk - band * chunks_per_band
                raw_byte = (
                    offsets["so"]
                    + output_stage * 128 * dvp * 2
                    + band * 128 * o_band_width * 2
                    + tid128 * o_band_width * 2
                )
                xor_mask = {16: 16, 32: 48, 64: 112}[o_band_width]
                first_byte = K.bitwise_xor(
                    raw_byte, K.bitwise_and(K.shift_right(raw_byte, K.uint32(3)), xor_mask)
                )
                if o_band_width == 16:
                    sign16 = K.if_then_else(K.bitwise_and(tid128, 4) == 0, 16, -16)
                elif o_band_width == 32:
                    sign16 = K.if_then_else(K.bitwise_and(tid128, 2) == 0, 16, -16)
                    sign32 = K.if_then_else(K.bitwise_and(tid128, 4) == 0, 32, -32)
                    if local_chunk == 1:
                        first_byte = first_byte + sign32
                else:
                    sign16 = K.if_then_else(K.bitwise_and(tid128, 1) == 0, 16, -16)
                    sign32 = K.if_then_else(K.bitwise_and(tid128, 2) == 0, 32, -32)
                    sign64 = K.if_then_else(K.bitwise_and(tid128, 4) == 0, 64, -64)
                    if local_chunk & 1:
                        first_byte = first_byte + sign32
                    if local_chunk & 2:
                        first_byte = first_byte + sign64
                return first_byte, first_byte + sign16

            def store_combined(output_stage, scale0, scale1):
                for chunk in range(dvp // 16):
                    values0 = K.alloc_local((16,), "float32")
                    values1 = K.alloc_local((16,), "float32")
                    _tmem_load16(values0, tmem_base + 256 + chunk * 16 + row_hi)
                    _tmem_load16(values1, tmem_base + 256 + dvp + chunk * 16 + row_hi)
                    with K.If(scale0 != K.float32(0.0)):
                        with K.Then():
                            with K.unroll(8) as pair:
                                base = pair * 2
                                _packed(
                                    "mul.rn.f32x2",
                                    values0,
                                    base,
                                    values0[base],
                                    values0[base + 1],
                                    scale0,
                                    scale0,
                                )
                        with K.Else():
                            with K.unroll(16) as value:
                                K.assign(values0[value], K.float32(0.0))
                    with K.If(scale1 != K.float32(0.0)), K.Then():
                        with K.unroll(8) as pair:
                            base = pair * 2
                            _packed(
                                "mul.rn.f32x2",
                                values1,
                                base,
                                values1[base],
                                values1[base + 1],
                                scale1,
                                scale1,
                            )
                            _packed(
                                "add.rn.f32x2",
                                values0,
                                base,
                                values0[base],
                                values0[base + 1],
                                values1[base],
                                values1[base + 1],
                            )
                    words = K.alloc_local((8,), "uint32")
                    with K.unroll(8) as pair:
                        if dtype == "bfloat16":
                            K.ptx.cvt.rn.bf16x2.f32(
                                words[pair], values0[pair * 2 + 1], values0[pair * 2]
                            )
                        else:
                            K.ptx.cvt.rn.f16x2.f32(
                                words[pair], values0[pair * 2 + 1], values0[pair * 2]
                            )
                    first_byte, second_byte = output_addresses(output_stage, chunk)
                    K.ptx.st.shared.v4.u32(
                        arena.ptr_to([first_byte]), words[0], words[1], words[2], words[3]
                    )
                    K.ptx.st.shared.v4.u32(
                        arena.ptr_to([second_byte]), words[4], words[5], words[6], words[7]
                    )

            def store_stage(output_stage, inv_sum):
                for chunk in range(dvp // 16):
                    values = K.alloc_local((16,), "float32")
                    _tmem_load16(values, tmem_base + 256 + output_stage * dvp + chunk * 16 + row_hi)
                    with K.unroll(8) as pair:
                        base = pair * 2
                        _packed(
                            "mul.rn.f32x2",
                            values,
                            base,
                            values[base],
                            values[base + 1],
                            inv_sum,
                            inv_sum,
                        )
                    words = K.alloc_local((8,), "uint32")
                    with K.unroll(8) as pair:
                        if dtype == "bfloat16":
                            K.ptx.cvt.rn.bf16x2.f32(
                                words[pair], values[pair * 2 + 1], values[pair * 2]
                            )
                        else:
                            K.ptx.cvt.rn.f16x2.f32(
                                words[pair], values[pair * 2 + 1], values[pair * 2]
                            )
                    first_byte, second_byte = output_addresses(output_stage, chunk)
                    K.ptx.st.shared.v4.u32(
                        arena.ptr_to([first_byte]), words[0], words[1], words[2], words[3]
                    )
                    K.ptx.st.shared.v4.u32(
                        arena.ptr_to([second_byte]), words[4], words[5], words[6], words[7]
                    )

            def store_zero_stage(output_stage):
                for chunk in range(dvp // 16):
                    first_byte, second_byte = output_addresses(output_stage, chunk)
                    K.ptx.st.shared.v4.u32(
                        arena.ptr_to([first_byte]),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                    )
                    K.ptx.st.shared.v4.u32(
                        arena.ptr_to([second_byte]),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                    )

            def store_zero():
                for chunk in range(dvp // 16):
                    first_byte, second_byte = output_addresses(0, chunk)
                    K.ptx.st.shared.v4.u32(
                        arena.ptr_to([first_byte]),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                    )
                    K.ptx.st.shared.v4.u32(
                        arena.ptr_to([second_byte]),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                    )

            def output_vector_address(output_stage, row, vector):
                vectors_per_band = o_band_width // 8
                band = vector // vectors_per_band
                local_vector = vector - band * vectors_per_band
                raw_byte = (
                    offsets["so"]
                    + output_stage * 128 * dvp * 2
                    + band * 128 * o_band_width * 2
                    + row * o_band_width * 2
                )
                swizzled_byte = K.bitwise_xor(
                    raw_byte,
                    K.bitwise_and(
                        K.shift_right(raw_byte, K.uint32(3)),
                        {16: 16, 32: 48, 64: 112}[o_band_width],
                    ),
                )
                return K.bitwise_xor(swizzled_byte, local_vector * 16)

            def store_varlen(output_stage, tile_stage):
                K.ptx.bar.sync(K.uint32(1), K.uint32(128))
                if logical_dv == dvp:
                    vectors_per_row = dvp // 8
                    for store_iter in range(vectors_per_row):
                        linear_vector = store_iter * 128 + tid128
                        store_row = linear_vector // vectors_per_row
                        store_vector = linear_vector - store_row * vectors_per_row
                        packed_row = q_tile(tile_stage) * 128 + store_row
                        if pack:
                            token = packed_row // ratio
                            local_head = packed_row - token * ratio
                            q_head = head * ratio + local_head
                        else:
                            token = packed_row
                            q_head = head
                        words = K.alloc_local((4,), "uint32")
                        shared_byte = output_vector_address(output_stage, store_row, store_vector)
                        K.ptx.ld.shared.v4.u32(
                            words[0], words[1], words[2], words[3], arena.ptr_to([shared_byte])
                        )
                        output_index = (
                            (q_offset + token) * hq + q_head
                        ) * logical_dv + store_vector * 8
                        if full_varlen_output_tiles:
                            K.ptx.st.global_.v4.u32(
                                out.ptr_to([output_index]), words[0], words[1], words[2], words[3]
                            )
                        else:
                            with K.If(token < q_length), K.Then():
                                K.ptx.st.global_.v4.u32(
                                    out.ptr_to([output_index]),
                                    words[0],
                                    words[1],
                                    words[2],
                                    words[3],
                                )
                else:
                    packed_row = q_tile(tile_stage) * 128 + tid128
                    if pack:
                        token = packed_row // ratio
                        local_head = packed_row - token * ratio
                        q_head = head * ratio + local_head
                    else:
                        token = packed_row
                        q_head = head
                    row_valid = token < q_length
                    for chunk in range(dvp // 16):
                        words = K.alloc_local((8,), "uint32")
                        first_byte, second_byte = output_addresses(output_stage, chunk)
                        K.ptx.ld.shared.v4.u32(
                            words[0], words[1], words[2], words[3], arena.ptr_to([first_byte])
                        )
                        K.ptx.ld.shared.v4.u32(
                            words[4], words[5], words[6], words[7], arena.ptr_to([second_byte])
                        )
                        output_index = ((q_offset + token) * hq + q_head) * logical_dv + chunk * 16
                        if chunk * 16 < logical_dv:
                            with K.If(row_valid), K.Then():
                                K.ptx.st.global_.v4.u32(
                                    out.ptr_to([output_index]),
                                    words[0],
                                    words[1],
                                    words[2],
                                    words[3],
                                )
                        if chunk * 16 + 8 < logical_dv:
                            with K.If(row_valid), K.Then():
                                K.ptx.st.global_.v4.u32(
                                    out.ptr_to([output_index + 8]),
                                    words[4],
                                    words[5],
                                    words[6],
                                    words[7],
                                )

            def store_lse(tile_stage, value):
                packed_row = q_tile(tile_stage) * 128 + tid128
                if pack:
                    token = packed_row // ratio
                    local_head = packed_row - token * ratio
                    q_head = head * ratio + local_head
                else:
                    token = packed_row
                    q_head = head
                with K.If(token < q_length), K.Then():
                    if is_varlen:
                        lse_index = q_head * total_q + q_offset + token
                    else:
                        lse_index = (batch_idx * hq + q_head) * seqlen_q + token
                    K.ptx.st.global_.f32(lse.ptr_to([lse_index]), value)

            with K.While(work_valid != 0):
                load_tile_metadata()
                if variant == "qstage2_1cta":
                    with K.If(has_work):
                        with K.Then():
                            _stats_sync(0, corr_warp)
                            stats_empty.arrive(0)
                            _stats_sync(1, corr_warp)

                            block = K.local_scalar("int32", init=1)
                            with K.While(block < block_iter_count):
                                for score_stage in range(2):
                                    _stats_sync(score_stage, corr_warp)
                                    scale = _ld_shared_f32(stats_smem, score_stage * 128 + tid128)
                                    ballot = K.local_scalar("uint32")
                                    K.ptx.vote_sync.ballot.b32(
                                        ballot,
                                        K.ptx.pred(scale < K.float32(1.0)),
                                        K.uint32(0xFFFFFFFF),
                                    )
                                    with K.If(ballot != 0), K.Then():
                                        rescale_o(score_stage, scale)
                                    arrive_spo(score_stage)
                                    stats_empty.arrive(1 - score_stage)
                                K.assign(block, block + 1)
                            stats_empty.arrive(1)

                            for score_stage in range(2):
                                _stats_sync(score_stage, corr_warp)
                                row_sum = _ld_shared_f32(stats_smem, score_stage * 128 + tid128)
                                row_max = K.local_scalar("float32", init=K.float32(_NEG_INF))
                                if return_lse:
                                    K.assign(
                                        row_max,
                                        _ld_shared_f32(
                                            stats_smem, 256 + score_stage * 128 + tid128
                                        ),
                                    )
                                stats_empty.arrive(score_stage)
                                valid = (row_sum != K.float32(0.0)) & (row_sum == row_sum)
                                inv_sum = _rcp(K.if_then_else(valid, row_sum, K.float32(1.0)))
                                _wait(oacc_full, score_stage, oacc_consumer_phase)
                                if not is_varlen:
                                    _wait(oepi_empty, score_stage, oepi_producer_phase)
                                store_stage(score_stage, inv_sum)
                                K.ptx.fence.proxy.async_.shared__cta()
                                arrive_spo(score_stage)
                                if is_varlen:
                                    store_varlen(score_stage, score_stage)
                                else:
                                    oepi_full.arrive(score_stage)

                                if return_lse:
                                    lse_value = K.local_scalar("float32", init=K.float32(_NEG_INF))
                                    with K.If(valid), K.Then():
                                        K.assign(
                                            lse_value,
                                            (row_max * softmax_scale_log2 + _log2(row_sum))
                                            * K.float32(_LN2),
                                        )
                                    store_lse(score_stage, lse_value)
                            K.assign(oacc_consumer_phase, oacc_consumer_phase ^ 1)
                        with K.Else():
                            for score_stage in range(2):
                                empty_sum = K.local_scalar("float32", init=K.float32(1.0))
                                _st_shared_f32(stats_smem, score_stage * 128 + tid128, empty_sum)
                                if return_lse:
                                    empty_max = K.local_scalar("float32", init=K.float32(_NEG_INF))
                                    _st_shared_f32(
                                        stats_smem, 256 + score_stage * 128 + tid128, empty_max
                                    )
                                _stats_sync(score_stage, corr_warp)
                                stats_empty.arrive(score_stage)
                                if not is_varlen:
                                    _wait(oepi_empty, score_stage, oepi_producer_phase)
                                store_zero_stage(score_stage)
                                K.ptx.fence.proxy.async_.shared__cta()
                                if is_varlen:
                                    store_varlen(score_stage, score_stage)
                                else:
                                    oepi_full.arrive(score_stage)
                                if return_lse:
                                    store_lse(score_stage, K.float32(_NEG_INF))
                    if not is_varlen:
                        K.assign(oepi_producer_phase, oepi_producer_phase ^ 1)
                else:
                    count0 = (block_iter_count + 1) // 2
                    count1 = block_iter_count // 2
                    total_sum = K.local_scalar("float32", init=K.float32(0.0))
                    safe_max = K.local_scalar("float32", init=K.float32(0.0))
                    sum0 = K.local_scalar("float32", init=K.float32(0.0))
                    sum1 = K.local_scalar("float32", init=K.float32(0.0))
                    maximum0 = K.local_scalar("float32", init=K.float32(_NEG_INF))
                    maximum1 = K.local_scalar("float32", init=K.float32(_NEG_INF))

                    _stats_sync(0, corr_warp)
                    stats_empty.arrive(0)
                    _stats_sync(1, corr_warp)
                    stats_empty.arrive(1)

                    stream_ordinal = K.local_scalar("int32", init=1)
                    with K.While(stream_ordinal < count0):
                        with K.If(stream_ordinal < count0), K.Then():
                            _stats_sync(0, corr_warp)
                            scale = _ld_shared_f32(stats_smem, tid128)
                            ballot = K.local_scalar("uint32")
                            K.ptx.vote_sync.ballot.b32(
                                ballot, K.ptx.pred(scale < K.float32(1.0)), K.uint32(0xFFFFFFFF)
                            )
                            with K.If(ballot != 0), K.Then():
                                rescale_o(0, scale)
                            arrive_spo(0)
                            stats_empty.arrive(0)
                        with K.If(stream_ordinal < count1), K.Then():
                            _stats_sync(1, corr_warp)
                            scale = _ld_shared_f32(stats_smem, 128 + tid128)
                            ballot = K.local_scalar("uint32")
                            K.ptx.vote_sync.ballot.b32(
                                ballot, K.ptx.pred(scale < K.float32(1.0)), K.uint32(0xFFFFFFFF)
                            )
                            with K.If(ballot != 0), K.Then():
                                rescale_o(1, scale)
                            arrive_spo(1)
                            stats_empty.arrive(1)
                        K.assign(stream_ordinal, stream_ordinal + 1)

                    with K.If(count0 > 0), K.Then():
                        _stats_sync(0, corr_warp)
                        K.assign(sum0, _ld_shared_f32(stats_smem, tid128))
                        K.assign(maximum0, _ld_shared_f32(stats_smem, 256 + tid128))
                        stats_empty.arrive(0)
                    with K.If(count1 > 0), K.Then():
                        _stats_sync(1, corr_warp)
                        K.assign(sum1, _ld_shared_f32(stats_smem, 128 + tid128))
                        K.assign(maximum1, _ld_shared_f32(stats_smem, 384 + tid128))
                        stats_empty.arrive(1)

                    valid0 = (count0 > 0) & (sum0 > K.float32(0.0)) & (sum0 == sum0)
                    valid1 = (count1 > 0) & (sum1 > K.float32(0.0)) & (sum1 == sum1)
                    rm0 = K.if_then_else(valid0, maximum0, K.float32(_NEG_INF))
                    rm1 = K.if_then_else(valid1, maximum1, K.float32(_NEG_INF))
                    maximum = K.local_scalar("float32")
                    K.ptx.max.f32(maximum, rm0, rm1)
                    K.assign(safe_max, K.if_then_else(maximum != K.float32(_NEG_INF), maximum, 0.0))
                    scale0 = K.local_scalar("float32")
                    scale1 = K.local_scalar("float32")
                    K.assign(
                        scale0,
                        K.if_then_else(valid0, _exp2((rm0 - safe_max) * softmax_scale_log2), 0.0),
                    )
                    K.assign(
                        scale1,
                        K.if_then_else(valid1, _exp2((rm1 - safe_max) * softmax_scale_log2), 0.0),
                    )
                    K.assign(total_sum, sum0 * scale0 + sum1 * scale1)
                    valid_total = (total_sum > K.float32(0.0)) & (total_sum == total_sum)
                    inv_sum = _rcp(K.if_then_else(valid_total, total_sum, 1.0))
                    final_scale0 = scale0 * inv_sum
                    final_scale1 = scale1 * inv_sum

                    with K.If(count0 > 0), K.Then():
                        _wait(oacc_full, 0, qstage1_o_phase0)
                    with K.If(count1 > 0), K.Then():
                        _wait(oacc_full, 1, qstage1_o_phase1)
                    if not is_varlen:
                        _wait(oepi_empty, output_stage, oepi_producer_phase)
                    store_combined(output_stage, final_scale0, final_scale1)
                    K.ptx.fence.proxy.async_.shared__cta()
                    with K.If(count0 > 0), K.Then():
                        arrive_spo(0)
                        K.assign(qstage1_o_phase0, qstage1_o_phase0 ^ 1)
                    with K.If(count1 > 0), K.Then():
                        arrive_spo(1)
                        K.assign(qstage1_o_phase1, qstage1_o_phase1 ^ 1)

                    if is_varlen:
                        store_varlen(output_stage, 0)
                    else:
                        oepi_full.arrive(output_stage)
                        K.assign(output_stage, output_stage + 1)
                        with K.If(output_stage == output_stages), K.Then():
                            K.assign(output_stage, 0)
                            K.assign(oepi_producer_phase, oepi_producer_phase ^ 1)
                    if return_lse:
                        lse_value = K.local_scalar("float32", init=K.float32(_NEG_INF))
                        with K.If(valid_total), K.Then():
                            K.assign(
                                lse_value,
                                (safe_max * softmax_scale_log2 + _log2(total_sum))
                                * K.float32(_LN2),
                            )
                        store_lse(0, lse_value)
                if overlap_qo and is_varlen:
                    _wait(load_epi_empty, 0, load_epi_producer_phase)
                    with K.If(K.cuda.elect_sync()), K.Then():
                        load_epi_full.arrive(0)
                    K.assign(load_epi_producer_phase, load_epi_producer_phase ^ 1)
                advance_work()
            if not is_varlen:
                if variant == "qstage2_1cta":
                    _wait(oepi_empty, q_stages - 1, oepi_producer_phase)
                else:
                    _wait(oepi_empty, output_stage, oepi_producer_phase)
            K.ptx.bar.arrive(K.uint32(2), K.uint32(416))

    def kernel(
        q,
        k,
        v,
        out,
        lse,
        partial_count,
        partial_offset,
        partial_index,
        full_count,
        full_offset,
        full_index,
        mask_payload,
        sequence_desc,
        fwd_work_desc,
        softmax_scale_log2,
        *,
        host,
    ):
        with K.attr({"tirx.required_block_size": 1}):
            kernel_body(
                q,
                k,
                v,
                out,
                lse,
                partial_count,
                partial_offset,
                partial_index,
                full_count,
                full_offset,
                full_index,
                mask_payload,
                sequence_desc,
                fwd_work_desc,
                softmax_scale_log2,
                host,
            )

    plan_item_capacity = plan_row_capacity * max_blocks
    kernel.__annotations__ = {
        "q": K.gptr[elem_dtype, (total_q * hq * logical_d,)],
        "k": K.gptr[elem_dtype, (total_k * hkv * logical_d,)],
        "v": K.gptr[elem_dtype, (total_k * hkv * logical_dv,)],
        "out": K.gptr[elem_dtype, (total_q * hq * logical_dv,)],
        "lse": K.gptr[K.f32, (total_q * hq,)],
        "partial_count": K.gptr[K.i32, (plan_row_capacity,)],
        "partial_offset": K.gptr[K.i32, (plan_row_capacity + 1,)],
        "partial_index": K.gptr[K.i32, (plan_item_capacity,)],
        "full_count": K.gptr[K.i32, (plan_row_capacity,)],
        "full_offset": K.gptr[K.i32, (plan_row_capacity + 1,)],
        "full_index": K.gptr[K.i32, (plan_item_capacity,)],
        "mask_payload": K.gptr[K.u32, (plan_item_capacity * work_q_tiles * 128 * 4,)],
        "sequence_desc": K.gptr[K.i32, (batch * 8,)],
        "fwd_work_desc": K.gptr[K.i32, (num_work_records * 4,)],
        "softmax_scale_log2": K.f32,
    }
    return K.kernel(
        warps=_WARPS,
        arch="sm_103a",
        grid=[num_work_records * cta_group, 1, 1],
        min_blocks_per_sm=1,
        host_prelude=host_prelude,
    )(kernel)


def get_kernel(**config):
    return _make_kernel(**config).func


def _without_label(config):
    return {key: value for key, value in config.items() if key != "label"}


def _torch_dtype(torch, name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _pack_enabled(config):
    ratio = config["num_q_heads"] // config["num_kv_heads"]
    requested = config["pack_gqa"]
    enabled = ratio > 1 if requested == "auto" else bool(requested)
    return enabled and 128 % ratio == 0


def _strided_kv(torch, generator, shape, dtype, use_i64):
    tensor = torch.randn(shape, dtype=dtype, generator=generator, device="cuda")
    if not use_i64:
        return tensor
    if shape[0] != 1:
        raise ValueError("the Int64 KV stride probe requires batch=1")
    return torch.as_strided(tensor, shape, (2**31, *tensor.stride()[1:]))


def prepare_data(**config):
    import torch

    from tirx_kernels.cudnn._reference import load_reference_module

    dtype = _torch_dtype(torch, config["dtype"])
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    is_varlen = config["layout"] == "varlen"
    batch = config["batch"]
    hq = config["num_q_heads"]
    hkv = config["num_kv_heads"]
    d = config["head_dim"]
    dv = config["head_dim_v"]

    if is_varlen:
        q_lengths = tuple(config["seqlens_q"])
        kv_lengths = tuple(config["seqlens_kv"])
        total_q = sum(q_lengths)
        total_k = sum(kv_lengths)
        q_user = torch.randn(total_q, hq, d, dtype=dtype, generator=generator, device="cuda")
        k_user = torch.randn(total_k, hkv, d, dtype=dtype, generator=generator, device="cuda")
        v_user = torch.randn(total_k, hkv, dv, dtype=dtype, generator=generator, device="cuda")
        q, k, v = q_user, k_user, v_user
        cu_seqlens_q = torch.tensor(
            [0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
        )
        cu_seqlens_k = torch.tensor(
            [0, *torch.tensor(kv_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
        )
    else:
        q_lengths = (config["seqlen_q"],) * batch
        kv_lengths = (config["seqlen_kv"],) * batch
        total_q = batch * config["seqlen_q"]
        total_k = batch * config["seqlen_kv"]
        q = torch.randn(
            batch, hq, config["seqlen_q"], d, dtype=dtype, generator=generator, device="cuda"
        )
        k = _strided_kv(
            torch,
            generator,
            (batch, hkv, config["seqlen_kv"], d),
            dtype,
            config["use_int64_kv_strides"],
        )
        v = _strided_kv(
            torch,
            generator,
            (batch, hkv, config["seqlen_kv"], dv),
            dtype,
            config["use_int64_kv_strides"],
        )
        q_user = q.transpose(1, 2).contiguous()
        if config["use_int64_kv_strides"]:
            k_user = k.transpose(1, 2)
            v_user = v.transpose(1, 2)
        else:
            k_user = k.transpose(1, 2).contiguous()
            v_user = v.transpose(1, 2).contiguous()
            k = k_user.transpose(1, 2).contiguous()
            v = v_user.transpose(1, 2).contiguous()
        q = q_user.transpose(1, 2).contiguous()
        cu_seqlens_q = None
        cu_seqlens_k = None

    nfunc = 3 if config["mask_pattern"] in ("split", "mixed") else 1
    endpoints = torch.zeros(
        (config["mask_heads"], nfunc, total_q), dtype=torch.int32, device="cuda"
    )
    q_begin = 0
    for q_length, kv_length in zip(q_lengths, kv_lengths):
        for token in range(q_length):
            flat_q = q_begin + token
            if config["mask_pattern"] == "full":
                endpoints[:, 0, flat_q] = kv_length
            elif config["mask_pattern"] == "causal":
                endpoints[:, 0, flat_q] = min(kv_length, token + 1)
            elif config["mask_pattern"] == "mixed":
                endpoints[:, 0, flat_q] = min(kv_length, 128)
                endpoints[:, 1, flat_q] = min(kv_length, 256)
                endpoints[:, 2, flat_q] = min(kv_length, 320 + token % 17)
            elif config["mask_pattern"] == "split":
                first_end = min(kv_length, max(1, kv_length // 4 + token % 17))
                second_begin = min(kv_length, max(first_end, kv_length // 2))
                second_end = min(kv_length, max(second_begin, 3 * kv_length // 4 + token % 29))
                endpoints[:, 0, flat_q] = first_end
                endpoints[:, 1, flat_q] = second_begin
                endpoints[:, 2, flat_q] = second_end
        q_begin += q_length

    flex = load_reference_module("cudnn.flex_attention")
    plan_kwargs = {
        "pack_gqa": _pack_enabled(config),
        "build_backward": False,
        "_fwd_variant": config["variant"],
    }
    if is_varlen:
        plan_kwargs.update(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=config["seqlen_q"],
            max_seqlen_k=config["seqlen_kv"],
        )
    plan = flex.create_mask_plan(endpoints, q_user, k_user, v_user, **plan_kwargs)
    packed_plan, owned_cu_q, owned_cu_k = plan._runtime_args
    if is_varlen:
        if owned_cu_q is None or owned_cu_k is None:
            raise AssertionError("varlen source plan omitted cumulative sequence lengths")
    elif owned_cu_q is not None or owned_cu_k is not None:
        raise AssertionError("fixed source plan unexpectedly owns sequence lengths")
    if bool(packed_plan.pack_gqa) != _pack_enabled(config):
        raise AssertionError("source plan resolved a different pack_gqa specialization")
    if packed_plan.fwd_work_desc is None:
        raise AssertionError("SM103 source plan omitted its forward work queue")
    sequence_desc = packed_plan.sequence_desc
    if sequence_desc is None:
        sequence_desc = torch.zeros((batch, 8), dtype=torch.int32, device="cuda")

    plan_item_capacity = packed_plan.mask_block_cnt.numel() * config["kv_blocks"]
    q_stages = 2 if config["variant"] == "qstage2_1cta" else 1
    cta_group = 2 if config["variant"] == "qstage1_2cta" else 1
    work_q_tiles = q_stages * cta_group

    def padded_plan_tensor(tensor, elements):
        result = torch.zeros((elements,), dtype=tensor.dtype, device=tensor.device)
        if tensor.numel() != 0:
            result[: tensor.numel()].copy_(tensor.reshape(-1))
        return result

    scale = d**-0.5 if config["softmax_scale"] is None else float(config["softmax_scale"])

    def tirx_outputs():
        out_shape = (total_q, hq, dv) if is_varlen else (batch, hq, config["seqlen_q"], dv)
        lse_shape = (hq, total_q) if is_varlen else (batch, hq, config["seqlen_q"])
        return {
            "out": torch.empty(out_shape, dtype=dtype, device="cuda"),
            "lse": torch.full(lse_shape, 12345.25, dtype=torch.float32, device="cuda"),
        }

    return {
        "config": dict(config),
        "q": q,
        "k": k,
        "v": v,
        "q_user": q_user,
        "k_user": k_user,
        "v_user": v_user,
        "cu_seqlens_q": owned_cu_q,
        "cu_seqlens_k": owned_cu_k,
        "endpoints": endpoints,
        "source_module": flex,
        "source_plan": plan,
        "partial_count": packed_plan.mask_block_cnt,
        "partial_offset": packed_plan.mask_block_offset,
        "partial_index": padded_plan_tensor(packed_plan.mask_block_idx, plan_item_capacity),
        "full_count": packed_plan.full_block_cnt,
        "full_offset": packed_plan.full_block_offset,
        "full_index": padded_plan_tensor(packed_plan.full_block_idx, plan_item_capacity),
        "mask_payload": padded_plan_tensor(
            packed_plan.mask_block_masks, plan_item_capacity * work_q_tiles * 128 * 4
        ),
        "sequence_desc": sequence_desc,
        "fwd_work_desc": packed_plan.fwd_work_desc,
        "softmax_scale": scale,
        "tirx": tirx_outputs(),
        "source": {"out": None, "lse": None},
    }


def _tirx_launch(executable, data):
    arguments = (
        data["q"].reshape(-1),
        data["k"].reshape(-1),
        data["v"].reshape(-1),
        data["tirx"]["out"].reshape(-1),
        data["tirx"]["lse"].reshape(-1),
        data["partial_count"].reshape(-1),
        data["partial_offset"].reshape(-1),
        data["partial_index"].reshape(-1),
        data["full_count"].reshape(-1),
        data["full_offset"].reshape(-1),
        data["full_index"].reshape(-1),
        data["mask_payload"].reshape(-1),
        data["sequence_desc"].reshape(-1),
        data["fwd_work_desc"].reshape(-1),
        data["softmax_scale"] * _LOG2_E,
    )

    def launch():
        executable(*arguments)

    launch._keep_alive = arguments
    return launch


def _compile_reference(data):
    import torch

    config = data["config"]
    flex = data["source_module"]
    plan = data["source_plan"]
    if config["layout"] == "varlen":
        source_out = torch.empty(
            (sum(config["seqlens_q"]), config["num_q_heads"], config["head_dim_v"]),
            dtype=_torch_dtype(torch, config["dtype"]),
            device="cuda",
        )
        lse_shape = (config["num_q_heads"], sum(config["seqlens_q"]))
    else:
        source_out = torch.empty(
            (config["batch"], config["seqlen_q"], config["num_q_heads"], config["head_dim_v"]),
            dtype=_torch_dtype(torch, config["dtype"]),
            device="cuda",
        )
        lse_shape = (config["batch"], config["num_q_heads"], config["seqlen_q"])
    source_lse = (
        torch.empty(lse_shape, dtype=torch.float32, device="cuda") if config["return_lse"] else None
    )
    operation = flex.FlexAttentionFwd(
        data["q_user"], data["k_user"], data["v_user"], source_out, plan, source_lse
    )
    if not operation.check_support():
        raise RuntimeError(f"FlexAttention source rejected {config.get('label', '<config>')}")
    operation.compile()

    data["source"]["out"] = (
        source_out if config["layout"] == "varlen" else source_out.transpose(1, 2)
    )
    data["source"]["lse"] = source_lse

    def launch():
        operation.execute(
            data["q_user"],
            data["k_user"],
            data["v_user"],
            source_out,
            plan,
            source_lse,
            softmax_scale=data["softmax_scale"],
        )

    launch._keep_alive = (flex, operation, plan, source_out, source_lse)
    return launch


def _oracle(data):
    import torch

    config = data["config"]
    q, k, v = data["q"], data["k"], data["v"]
    is_varlen = config["layout"] == "varlen"
    if is_varlen:
        out = torch.zeros_like(data["tirx"]["out"])
        lse = torch.full_like(data["tirx"]["lse"], -float("inf"))
        q_lengths = tuple(config["seqlens_q"])
        kv_lengths = tuple(config["seqlens_kv"])
    else:
        out = torch.zeros_like(data["tirx"]["out"])
        lse = torch.full_like(data["tirx"]["lse"], -float("inf"))
        q_lengths = (config["seqlen_q"],) * config["batch"]
        kv_lengths = (config["seqlen_kv"],) * config["batch"]

    ratio = config["num_q_heads"] // config["num_kv_heads"]
    endpoints = data["endpoints"]
    q_begin = 0
    k_begin = 0
    for batch_idx, (q_length, kv_length) in enumerate(zip(q_lengths, kv_lengths)):
        key_positions = torch.arange(kv_length, device=q.device)
        for q_head in range(config["num_q_heads"]):
            endpoint_head = 0 if config["mask_heads"] == 1 else q_head
            endpoint_slice = endpoints[endpoint_head, :, q_begin : q_begin + q_length]
            allowed = key_positions[None, :] < endpoint_slice[0, :, None]
            if endpoints.shape[1] == 3:
                allowed |= (key_positions[None, :] >= endpoint_slice[1, :, None]) & (
                    key_positions[None, :] < endpoint_slice[2, :, None]
                )
            kv_head = q_head // ratio
            if is_varlen:
                q_values = q[q_begin : q_begin + q_length, q_head]
                k_values = k[k_begin : k_begin + kv_length, kv_head]
                v_values = v[k_begin : k_begin + kv_length, kv_head]
            else:
                q_values = q[batch_idx, q_head, :q_length]
                k_values = k[batch_idx, kv_head, :kv_length]
                v_values = v[batch_idx, kv_head, :kv_length]
            scores = q_values.float() @ k_values.float().T
            scores *= data["softmax_scale"]
            live_rows = allowed.any(dim=1)
            masked_scores = scores.masked_fill(~allowed, -float("inf"))
            if live_rows.any():
                probabilities = torch.softmax(masked_scores[live_rows], dim=-1)
                result = (probabilities @ v_values.float()).to(q.dtype)
                result_lse = torch.logsumexp(masked_scores[live_rows], dim=-1)
                if is_varlen:
                    out[q_begin : q_begin + q_length, q_head][live_rows] = result
                    lse[q_head, q_begin : q_begin + q_length][live_rows] = result_lse
                else:
                    out[batch_idx, q_head, :q_length][live_rows] = result
                    lse[batch_idx, q_head, :q_length][live_rows] = result_lse
        q_begin += q_length
        k_begin += kv_length
    return out, lse


def _validate_tensor_pair(torch, name, actual, expected, *, atol, rtol):
    if torch.isnan(actual).any():
        raise AssertionError(f"{name} contains NaN")
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def _validate_outputs(data, *, with_oracle):
    import torch

    config = data["config"]
    tirx = data["tirx"]
    source = data["source"]
    if source["out"] is None:
        raise AssertionError("pinned source did not produce O")
    _validate_tensor_pair(
        torch,
        "TIRx versus source O",
        tirx["out"].float(),
        source["out"].float(),
        atol=0.002,
        rtol=0.002,
    )
    if config["return_lse"]:
        if source["lse"] is None:
            raise AssertionError("pinned source did not produce requested LSE")
        if torch.isnan(tirx["lse"]).any() or torch.isnan(source["lse"]).any():
            raise AssertionError("LSE contains NaN")
        if not torch.equal(torch.isfinite(tirx["lse"]), torch.isfinite(source["lse"])):
            raise AssertionError("TIRx/source LSE finite mask mismatch")
        finite = torch.isfinite(source["lse"])
        _validate_tensor_pair(
            torch,
            "TIRx versus source finite LSE",
            tirx["lse"][finite],
            source["lse"][finite],
            atol=2e-6,
            rtol=2e-6,
        )
    else:
        if source["lse"] is not None:
            raise AssertionError("source returned LSE for return_lse=False")
        if not torch.equal(tirx["lse"], torch.full_like(tirx["lse"], 12345.25)):
            raise AssertionError("no-LSE specialization modified its sentinel")

    metrics = {}
    if with_oracle:
        oracle_out, oracle_lse = _oracle(data)
        for name, values in (("tirx", tirx), ("source", source)):
            _validate_tensor_pair(
                torch,
                f"{name} versus oracle O",
                values["out"].float(),
                oracle_out.float(),
                atol=0.004,
                rtol=0.008,
            )
            if config["return_lse"]:
                if not torch.equal(torch.isfinite(values["lse"]), torch.isfinite(oracle_lse)):
                    raise AssertionError(f"{name}/oracle LSE finite mask mismatch")
                finite = torch.isfinite(oracle_lse)
                _validate_tensor_pair(
                    torch,
                    f"{name} versus oracle finite LSE",
                    values["lse"][finite],
                    oracle_lse[finite],
                    atol=2e-6,
                    rtol=2e-6,
                )
        empty = ~torch.isfinite(oracle_lse)
        if empty.any():
            for name, values in (("tirx", tirx), ("source", source)):
                if not torch.equal(values["out"][empty], torch.zeros_like(values["out"][empty])):
                    raise AssertionError(f"{name} empty sparse rows are not bitwise zero")
                if config["return_lse"] and not torch.isneginf(values["lse"][empty]).all():
                    raise AssertionError(f"{name} empty sparse LSE is not exactly -inf")
        metrics = {
            "tirx_source_o_max_abs": float(
                (tirx["out"].float() - source["out"].float()).abs().max()
            ),
            "tirx_oracle_o_max_abs": float((tirx["out"].float() - oracle_out.float()).abs().max()),
            "source_oracle_o_max_abs": float(
                (source["out"].float() - oracle_out.float()).abs().max()
            ),
        }
        if config["return_lse"]:
            finite_source = torch.isfinite(source["lse"])
            finite_oracle = torch.isfinite(oracle_lse)
            metrics.update(
                tirx_source_lse_max_abs=float(
                    (tirx["lse"][finite_source] - source["lse"][finite_source]).abs().max()
                )
                if finite_source.any()
                else 0.0,
                tirx_oracle_lse_max_abs=float(
                    (tirx["lse"][finite_oracle] - oracle_lse[finite_oracle]).abs().max()
                )
                if finite_oracle.any()
                else 0.0,
                source_oracle_lse_max_abs=float(
                    (source["lse"][finite_oracle] - oracle_lse[finite_oracle]).abs().max()
                )
                if finite_oracle.any()
                else 0.0,
            )
    return metrics


def run_test(**config):
    import torch

    from tirx_kernels.runner import compile_kernel

    kernel_config = _without_label(config)
    data = prepare_data(**kernel_config)
    tirx_launch = _tirx_launch(compile_kernel(get_kernel(**kernel_config)), data)
    source_launch = _compile_reference(data)
    tirx_launch()
    torch.cuda.synchronize()
    source_launch()
    torch.cuda.synchronize()
    return _validate_outputs(data, with_oracle=True)


def prepare_bench(**config):
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    kernel_config = _without_label(config)
    state = {"config": kernel_config, "executable": compile_kernel(get_kernel(**kernel_config))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **kwargs):
    from tirx_kernels.runner import bench, defer_gpu_interrupts, external_references_enabled

    with defer_gpu_interrupts():
        import torch

    config = _without_label({**prepared["config"], **kwargs})
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
            _validate_outputs(data, with_oracle=False)
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


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **config):
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
