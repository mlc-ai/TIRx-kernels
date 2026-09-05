# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM100 blk128 block-sparse attention forward pass.

Upstream sources:
``python/cudnn/block_sparse_attention/csrc/fwd/sm100_blk128/bsa_fwd_sm100.py``,
``python/cudnn/block_sparse_attention/_interface.py``, and the block-sparse
scheduler, pipeline, softmax, and packed-GQA helpers imported by that kernel.
"""

import tirx_kernels.kern as K

KERNEL_META = {
    "name": "cudnn_sm100_bsa_forward_blk128",
    "category": "cudnn",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "reference_requirements": (
        {
            "package": "nvidia-cudnn-frontend",
            "git": {
                "url": "https://github.com/NVIDIA/cudnn-frontend.git",
                "commit": "aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5",
            },
            "import": "cudnn",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.8.0.dev0", "import": "cutlass"},
    ),
}


def _config(label, **overrides):
    config = {
        "label": label,
        "batch": 1,
        "num_q_heads": 2,
        "num_kv_heads": 2,
        "seqlen_q": 257,
        "seqlen_kv": 1023,
        "head_dim": 128,
        "dtype": "bfloat16",
        "kv_blocks": 4,
        "tensor_layout": "bhsd",
        "has_block_sizes": True,
        "block_count_mode": "fixed",
        "block_count_pattern": None,
        "pack_gqa": "auto",
        "return_lse": True,
        "softmax_scale": None,
        "use_int64_kv_strides": False,
        "seed": 12800,
    }
    config.update(overrides)
    return config


def _correctness_configs():
    configs = []
    seed = 12800

    def add(label, **kwargs):
        nonlocal seed
        seed += 1
        configs.append(_config(label, seed=seed, **kwargs))

    for dtype in ("bfloat16", "float16"):
        dtype_label = "bf16" if dtype == "bfloat16" else "fp16"
        for head_dim in (64, 96, 128):
            for mode, num_q_heads, num_kv_heads in (("mha", 2, 2), ("gqa", 4, 2), ("mqa", 8, 1)):
                add(
                    f"c_{dtype_label}_d{head_dim}_{mode}_core",
                    dtype=dtype,
                    head_dim=head_dim,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                )

    add("c_bf16_d128_mha_sq1_skv256_kv2_mask", seqlen_q=1, seqlen_kv=256, kv_blocks=2)
    add(
        "c_fp16_d64_mha_sq127_skv255_kv2_mask",
        dtype="float16",
        head_dim=64,
        seqlen_q=127,
        seqlen_kv=255,
        kv_blocks=2,
    )
    add(
        "c_bf16_d96_mha_sq128_skv512_kv4_nomask",
        head_dim=96,
        seqlen_q=128,
        seqlen_kv=512,
        has_block_sizes=False,
    )
    add(
        "c_fp16_d128_mha_sq129_skv512_var_1_3_4_mask",
        dtype="float16",
        seqlen_q=129,
        seqlen_kv=512,
        block_count_mode="variable",
        block_count_pattern="1,3,4",
    )
    add(
        "c_bf16_d64_mha_sq129_skv512_empty_0_1_4_mask",
        head_dim=64,
        seqlen_q=129,
        seqlen_kv=512,
        block_count_mode="variable_empty",
        block_count_pattern="0,1,4",
    )
    add(
        "c_fp16_d96_gqa_var_1_3_4_nomask_pack",
        dtype="float16",
        head_dim=96,
        num_q_heads=4,
        num_kv_heads=2,
        block_count_mode="variable",
        block_count_pattern="1,3,4",
        has_block_sizes=False,
    )
    add(
        "c_bf16_d128_mqa_empty_0_1_4_mask_pack",
        num_q_heads=8,
        num_kv_heads=1,
        block_count_mode="variable_empty",
        block_count_pattern="0,1,4",
    )
    add(
        "c_fp16_d64_gqa_fixed_mask_nopack",
        dtype="float16",
        head_dim=64,
        num_q_heads=4,
        num_kv_heads=2,
        pack_gqa=False,
    )
    add("c_bf16_d96_gqa_ratio3_auto_fallback", head_dim=96, num_q_heads=3, num_kv_heads=1)
    add(
        "c_fp16_d128_gqa_pack_bshd",
        dtype="float16",
        num_q_heads=4,
        num_kv_heads=2,
        pack_gqa=True,
        tensor_layout="bshd",
    )
    add("c_bf16_d64_mha_b2_h3", batch=2, num_q_heads=3, num_kv_heads=3, head_dim=64)
    add("c_fp16_d96_mha_no_lse", dtype="float16", head_dim=96, return_lse=False)
    add("c_bf16_d128_mha_scale0125_i64kv", softmax_scale=0.125, use_int64_kv_strides=True)
    add(
        "c_fp16_d64_mqa_all_empty",
        dtype="float16",
        head_dim=64,
        num_q_heads=8,
        num_kv_heads=1,
        block_count_mode="variable_empty",
        block_count_pattern="0",
    )
    assert len(configs) == 32
    return configs


def _benchmark_configs():
    configs = []
    seed = 12900

    def add(**kwargs):
        nonlocal seed
        seed += 1
        index = len(configs)
        dtype_label = "bf16" if kwargs["dtype"] == "bfloat16" else "fp16"
        mode = kwargs.pop("mode")
        label = f"p{index:02d}_{dtype_label}_d{kwargs['head_dim']}_{mode}"
        configs.append(_config(label, seed=seed, **kwargs))

    for dtype in ("bfloat16", "float16"):
        for head_dim in (64, 96, 128):
            for mode, num_kv_heads in (("mha", 8), ("gqa", 2), ("mqa", 1)):
                add(
                    mode=mode,
                    dtype=dtype,
                    head_dim=head_dim,
                    batch=1,
                    num_q_heads=8,
                    num_kv_heads=num_kv_heads,
                    seqlen_q=4096,
                    seqlen_kv=8192,
                    kv_blocks=32,
                )

    add(
        mode="mha_tail_var",
        dtype="bfloat16",
        head_dim=128,
        batch=1,
        num_q_heads=8,
        num_kv_heads=8,
        seqlen_q=4097,
        seqlen_kv=8191,
        kv_blocks=32,
        block_count_mode="variable",
        block_count_pattern="1,mid,max",
    )
    add(
        mode="gqa_tail_empty_nomask",
        dtype="float16",
        head_dim=96,
        batch=1,
        num_q_heads=8,
        num_kv_heads=2,
        seqlen_q=4097,
        seqlen_kv=8191,
        kv_blocks=32,
        block_count_mode="variable_empty",
        block_count_pattern="0,1,max",
        has_block_sizes=False,
    )
    add(
        mode="gqa_nopack_longkv",
        dtype="bfloat16",
        head_dim=64,
        batch=1,
        num_q_heads=8,
        num_kv_heads=2,
        seqlen_q=2048,
        seqlen_kv=16384,
        kv_blocks=128,
        pack_gqa=False,
    )
    add(
        mode="gqa_ratio3_fallback_nolse",
        dtype="float16",
        head_dim=128,
        batch=1,
        num_q_heads=6,
        num_kv_heads=2,
        seqlen_q=2048,
        seqlen_kv=8192,
        kv_blocks=32,
        return_lse=False,
    )
    add(
        mode="mha_b2_h4",
        dtype="bfloat16",
        head_dim=96,
        batch=2,
        num_q_heads=4,
        num_kv_heads=4,
        seqlen_q=4096,
        seqlen_kv=8192,
        kv_blocks=32,
    )
    add(
        mode="mha_i64kv",
        dtype="bfloat16",
        head_dim=128,
        batch=1,
        num_q_heads=4,
        num_kv_heads=4,
        seqlen_q=4096,
        seqlen_kv=8192,
        kv_blocks=32,
        use_int64_kv_strides=True,
    )
    assert len(configs) == 24
    return configs


CONFIGS = _correctness_configs()
BENCH_CONFIGS = _benchmark_configs()


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
_TCGEN_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
_TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
_TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
_TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
_TMEM_LD32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
_TMEM_LD16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
_TMEM_ST16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
_TMA_G2S_4D = (
    "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
)
_TMA_G2S_5D = (
    "cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
)
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


def _tmem_load16(dst, address):
    K.ptx[_TMEM_LD16](*(dst[i] for i in range(16)), K.cast(address, "uint32"))


def _tmem_store16(src, address):
    K.ptx[_TMEM_ST16](K.cast(address, "uint32"), *(src[i] for i in range(16)))


def _query_cancel_response(response, q_block, head, batch_idx, valid):
    payload = K.local_scalar("uint128")
    payload_lo = K.local_scalar("uint64")
    payload_hi = K.local_scalar("uint64")
    canceled = K.local_scalar("uint32")
    K.ptx.ld.shared.v2.b64(payload_lo, payload_hi, K.address_of(response[0]))
    K.ptx.mov.b128(payload, payload_lo, payload_hi)
    K.ptx.clusterlaunchcontrol.query_cancel.is_canceled.pred.b128(canceled, payload)
    K.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__x.b32.b128(
        q_block, payload, pred=canceled
    )
    K.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__y.b32.b128(
        head, payload, pred=canceled
    )
    K.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__z.b32.b128(
        batch_idx, payload, pred=canceled
    )
    K.assign(valid, K.cast(canceled, "int32"))
    K.ptx.fence.proxy.async_.shared__cta()


def _make_kernel(**config):
    batch = int(config["batch"])
    hq = int(config["num_q_heads"])
    hkv = int(config["num_kv_heads"])
    seqlen_q = int(config["seqlen_q"])
    seqlen_kv = int(config["seqlen_kv"])
    d = int(config["head_dim"])
    dtype = config["dtype"]
    max_blocks = int(config["kv_blocks"])
    has_sizes = bool(config["has_block_sizes"])
    has_nums = config["block_count_mode"] != "fixed"
    allow_empty = config["block_count_mode"] == "variable_empty"
    return_lse = bool(config["return_lse"])
    use_i64_kv = bool(config["use_int64_kv_strides"])
    if hq <= 0 or hkv <= 0 or hq % hkv:
        raise ValueError("num_q_heads must be a positive multiple of num_kv_heads")
    if d not in (64, 96, 128):
        raise ValueError("head_dim must be 64, 96, or 128")
    if dtype not in ("bfloat16", "float16"):
        raise ValueError("dtype must be bfloat16 or float16")
    if config["block_count_mode"] == "fixed" and (max_blocks < 2 or max_blocks % 2):
        raise ValueError("fixed block count must be even and at least two")

    ratio = hq // hkv
    requested_pack = config["pack_gqa"]
    pack = (ratio > 1 if requested_pack == "auto" else bool(requested_pack)) and 128 % ratio == 0
    scheduled_heads = hkv if pack else hq
    q_rows = seqlen_q * ratio if pack else seqlen_q
    q_blocks = (q_rows + 127) // 128
    kv_stages = {64: 11, 96: 6, 128: 4}[d]
    smem_bytes = {64: 232448, 96: 224256, 128: 232448}[d]
    offsets = {
        64: dict(
            q=0,
            kv=16,
            spo=192,
            plast=224,
            oacc=256,
            stats=288,
            oepi=320,
            tmem=352,
            scale=356,
            clc=2408,
            response=2432,
            prefix=2456,
            so=3072,
            sq=35840,
            skv=52224,
        ),
        96: dict(
            q=0,
            kv=16,
            spo=112,
            plast=144,
            oacc=176,
            stats=208,
            oepi=240,
            tmem=272,
            scale=276,
            clc=2328,
            response=2352,
            prefix=2376,
            so=3072,
            sq=52224,
            skv=76800,
        ),
        128: dict(
            q=0,
            kv=16,
            spo=80,
            plast=112,
            oacc=144,
            stats=176,
            oepi=208,
            tmem=240,
            scale=244,
            clc=2296,
            response=2320,
            prefix=2344,
            so=3072,
            sq=68608,
            skv=101376,
        ),
    }[d]
    band_width = 32 if d == 96 else 64
    num_bands = d // band_width
    smem_swizzle = 2 if d == 96 else 3
    elem_dtype = K.bf16 if dtype == "bfloat16" else K.f16
    elem_dtype_name = "bfloat16" if dtype == "bfloat16" else "float16"
    qk_idesc = 0x08200490 if dtype == "bfloat16" else 0x08200010
    pv_idesc = {
        (64, "bfloat16"): 0x08110490,
        (64, "float16"): 0x08110010,
        (96, "bfloat16"): 0x08190490,
        (96, "float16"): 0x08190010,
        (128, "bfloat16"): 0x08210490,
        (128, "float16"): 0x08210010,
    }[(d, dtype)]
    regs_other = 48 if d == 64 else 56
    regs_softmax = 200 if d == 64 else 184
    regs_correction = 64 if d == 64 else 88
    physical_blocks = (seqlen_kv + 127) // 128

    def host_prelude(params):
        def encode(name, tensor, dims, strides, box):
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
                smem_swizzle,
                2,
                0,
            )
            return descriptor

        q = params["q"]
        k = params["k"]
        v = params["v"]
        out = params["out"]
        elem_bytes = 2
        if pack:
            qo_dims = (d, ratio, seqlen_q, hkv, batch)
            qo_strides = (
                seqlen_q * d * elem_bytes,
                d * elem_bytes,
                ratio * seqlen_q * d * elem_bytes,
                hq * seqlen_q * d * elem_bytes,
            )
            qo_box = (band_width, ratio, 128 // ratio, 1, 1)
        else:
            qo_dims = (d, seqlen_q, hq, batch)
            qo_strides = (d * elem_bytes, seqlen_q * d * elem_bytes, hq * seqlen_q * d * elem_bytes)
            qo_box = (band_width, 128, 1, 1)
        kv_batch_stride = (2**31 if use_i64_kv else hkv * seqlen_kv * d) * elem_bytes
        kv_dims = (d, seqlen_kv, hkv, batch)
        kv_strides = (d * elem_bytes, seqlen_kv * d * elem_bytes, kv_batch_stride)
        kv_box = (band_width, 128, 1, 1)
        return (
            encode("q", q, qo_dims, qo_strides, qo_box),
            encode("k", k, kv_dims, kv_strides, kv_box),
            encode("v", v, kv_dims, kv_strides, kv_box),
            encode("o", out, qo_dims, qo_strides, qo_box),
        )

    def kernel_body(
        q,
        k,
        v,
        out,
        lse,
        block_index,
        block_sizes,
        block_nums,
        block_sparse_num,
        softmax_scale_log2,
        host,
    ):
        del q, k, v, out
        q_map, k_map, v_map, o_map = host
        _cluster_x, _cluster_y, _cluster_z = K.cta_id_in_cluster([1, 1, 1])
        initial_q_block, initial_head, initial_batch = K.cta_id([q_blocks, scheduled_heads, batch])
        q_block = K.local_scalar("int32", init=initial_q_block)
        head = K.local_scalar("int32", init=initial_head)
        batch_idx = K.local_scalar("int32", init=initial_batch)
        work_valid = K.local_scalar("int32", init=1)
        clc_consumer_phase = K.local_scalar("int32", init=0)
        warp = K.warp_id()
        lane = K.thread_id() & 31
        tid = K.thread_id()

        with K.If(warp == 0), K.Then():
            K.ptx.prefetch.tensormap(K.address_of(q_map))
            K.ptx.prefetch.tensormap(K.address_of(k_map))
            K.ptx.prefetch.tensormap(K.address_of(v_map))
            K.ptx.prefetch.tensormap(K.address_of(o_map))

        arena = K.alloc_buffer((smem_bytes,), K.u8, scope="shared.dyn", align=1024)
        smem = K.smem_pool(base=arena)
        pool = smem.pool
        q_full = K.TMABar(pool, 1)
        q_empty = K.TCGen05Bar(pool, 1)
        kv_full = K.TMABar(pool, kv_stages)
        kv_empty = K.TCGen05Bar(pool, kv_stages)
        spo_full = K.TCGen05Bar(pool, 2)
        spo_empty = K.MBarrier(pool, 2)
        plast_full = K.MBarrier(pool, 2)
        plast_empty = K.TCGen05Bar(pool, 2)
        oacc_full = K.TCGen05Bar(pool, 2)
        oacc_empty = K.MBarrier(pool, 2)
        stats_full = K.MBarrier(pool, 2)
        stats_empty = K.MBarrier(pool, 2)
        oepi_full = K.MBarrier(pool, 2)
        oepi_empty = K.MBarrier(pool, 2)
        tmem_mailbox = pool.alloc((1,), "uint32", align=4)
        stats_smem = pool.alloc((512,), "float32", align=4)
        clc_full = K.TMABar(pool, 1)
        clc_empty = K.MBarrier(pool, 1)
        pool.alloc((8,), "uint8")
        clc_response = pool.alloc((4,), "uint32", align=16)
        pool.alloc((8,), "uint8")
        if pool.offset != offsets["prefix"]:
            raise AssertionError("shared protocol prefix changed")
        pool.alloc((3072 - pool.offset,), "uint8")
        if pool.offset != 3072:
            raise AssertionError("shared payload alignment changed")
        pool.alloc((smem_bytes - pool.offset,), "uint8")
        if pool.offset != smem_bytes:
            raise AssertionError("shared arena size changed")

        q_smem = K.decl_buffer(
            (128 * d,),
            elem_dtype_name,
            data=arena.data,
            byte_offset=offsets["sq"],
            scope="shared.dyn",
            align=1024,
        )
        kv_smem = K.decl_buffer(
            (kv_stages * 128 * d,),
            elem_dtype_name,
            data=arena.data,
            byte_offset=offsets["skv"],
            scope="shared.dyn",
            align=1024,
        )
        o_smem = K.decl_buffer(
            (2 * 128 * d,),
            elem_dtype_name,
            data=arena.data,
            byte_offset=offsets["so"],
            scope="shared.dyn",
            align=1024,
        )

        q_full.init(1)
        q_empty.init(1)
        kv_full.init(1)
        kv_empty.init(1)
        spo_full.init(1)
        spo_empty.init(256)
        plast_full.init(4)
        plast_empty.init(1)
        oacc_full.init(1)
        oacc_empty.init(128)
        stats_full.init(128)
        stats_empty.init(128)
        oepi_full.init(128)
        oepi_empty.init(32)
        K.ptx.fence.mbarrier_init.release.cluster()
        clc_full.init(1)
        clc_empty.init(512)
        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cta_sync()
        K.cuda.cta_sync()

        raw_count = K.local_scalar("int32")

        def load_tile_metadata():
            count_index = (batch_idx * scheduled_heads + head) * q_blocks + q_block
            if has_nums:
                K.assign(raw_count, _load_i32(block_nums, count_index))
            else:
                K.assign(raw_count, block_sparse_num)

        def advance_work():
            K.cuda.cta_sync()
            _wait(clc_full, 0, clc_consumer_phase)
            _query_cancel_response(clc_response, q_block, head, batch_idx, work_valid)
            clc_empty.arrive(0, remote=0, pred=K.bool(True), count=1)
            K.assign(clc_consumer_phase, clc_consumer_phase ^ 1)

        def sparse_id(logical):
            count_index = (batch_idx * scheduled_heads + head) * q_blocks + q_block
            if has_nums:
                logical = K.min(logical, K.max(raw_count - 1, 0))
            return _load_i32(block_index, count_index * max_blocks + logical)

        def advance_ring(stage, phase):
            K.assign(stage, stage + 1)
            with K.If(stage == kv_stages), K.Then():
                K.assign(stage, 0)
                K.assign(phase, phase ^ 1)

        has_work = raw_count > 0 if allow_empty else K.bool(True)
        block_iter_count = (raw_count + 1) & -2 if has_nums else raw_count

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
                _wait(clc_empty, 0, clc_producer_phase)
                with K.If(lane == 0), K.Then():
                    clc_full.arrive(0, tx_count=16, remote=0)
                with K.If(K.cuda.elect_sync()), K.Then():
                    K.ptx[
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                        ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                    ](K.address_of(clc_response[0]), K.address_of(clc_full.buf[0]))
                K.assign(clc_producer_phase, clc_producer_phase ^ 1)
                advance_work()
            _wait(clc_empty, 0, clc_producer_phase)

        with r_load:
            q_producer_phase = K.local_scalar("int32", init=1)
            kv_stage = K.local_scalar("int32", init=0)
            kv_phase = K.local_scalar("int32", init=1)

            def load_kv(kind, logical, do_advance=True):
                tma_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                _wait(kv_empty, kv_stage, kv_phase)
                count_index = (batch_idx * scheduled_heads + head) * q_blocks + q_block
                logical_index = K.min(logical, K.max(raw_count - 1, 0)) if has_nums else logical
                sid = K.local_scalar("int32", init=0)
                K.ptx["ld.global.b32"](
                    sid,
                    block_index.ptr_to([count_index * max_blocks + logical_index]),
                    pred=tma_leader,
                )
                K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                    kv_full.ptr_to([kv_stage]), K.uint32(128 * d * 2), pred=tma_leader
                )
                with K.unroll(num_bands) as band:
                    destination = kv_stage * 128 * d + band * 128 * band_width
                    descriptor = K.address_of(k_map) if kind == 0 else K.address_of(v_map)
                    K.ptx[_TMA_G2S_4D](
                        kv_smem.ptr_to([destination]),
                        descriptor,
                        K.cast(band * band_width, "int32"),
                        sid * 128,
                        head if pack else head // ratio,
                        batch_idx,
                        K.cuda.cvta_generic_to_shared(kv_full.ptr_to([kv_stage])),
                        _TMA_CACHE,
                        pred=tma_leader,
                    )
                if do_advance:
                    advance_ring(kv_stage, kv_phase)

            with K.While(work_valid != 0):
                load_tile_metadata()
                with K.If(has_work), K.Then():
                    load_kv(0, block_iter_count - 1, False)
                    _wait(q_empty, 0, q_producer_phase)
                    with K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                            q_full.ptr_to([0]), K.uint32(128 * d * 2)
                        )
                        with K.unroll(num_bands) as band:
                            if pack:
                                K.ptx[_TMA_G2S_5D](
                                    q_smem.ptr_to([band * 128 * band_width]),
                                    K.address_of(q_map),
                                    K.cast(band * band_width, "int32"),
                                    K.int32(0),
                                    (q_block * 128) // ratio,
                                    head,
                                    batch_idx,
                                    K.cuda.cvta_generic_to_shared(q_full.ptr_to([0])),
                                    _TMA_CACHE,
                                )
                            else:
                                K.ptx[_TMA_G2S_4D](
                                    q_smem.ptr_to([band * 128 * band_width]),
                                    K.address_of(q_map),
                                    K.cast(band * band_width, "int32"),
                                    q_block * 128,
                                    head,
                                    batch_idx,
                                    K.cuda.cvta_generic_to_shared(q_full.ptr_to([0])),
                                    _TMA_CACHE,
                                )
                    advance_ring(kv_stage, kv_phase)
                    K.assign(q_producer_phase, q_producer_phase ^ 1)
                    load_kv(0, block_iter_count - 2)
                    i = K.local_scalar("int32", init=0)
                    with K.While(i < block_iter_count - 2):
                        load_kv(1, block_iter_count - 1 - i)
                        load_kv(0, block_iter_count - 3 - i)
                        K.assign(i, i + 1)
                    load_kv(1, 1)
                    load_kv(1, 0)
                advance_work()
            tail_stage = K.local_scalar("int32", init=kv_stage)
            tail_phase = K.local_scalar("int32", init=kv_phase)
            with K.unroll(kv_stages) as _tail:
                _wait(kv_empty, tail_stage, tail_phase)
                advance_ring(tail_stage, tail_phase)
            _wait(q_empty, 0, q_producer_phase)
            with K.If(K.cuda.elect_sync()), K.Then():
                K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                    q_full.ptr_to([0]), K.uint32(128 * d * 2)
                )

        with r_mma:
            K.ptx[_TMEM_ALLOC](K.address_of(tmem_mailbox[0]), K.uint32(_TMEM_COLS))
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            tmem_base = K.local_scalar("uint32", init=K.uint32(0))
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            q_desc = K.SmemDescriptor()
            q_desc.init(
                q_smem.ptr_to([0]), ldo=1 if d == 96 else 1024, sdo=band_width, swizzle=smem_swizzle
            )
            q_desc.make_lo_uniform()
            q_consumer_phase = K.local_scalar("int32", init=0)
            kv_stage = K.local_scalar("int32", init=0)
            kv_phase = K.local_scalar("int32", init=0)
            spo_phase0 = K.local_scalar("int32", init=0)
            spo_phase1 = K.local_scalar("int32", init=0)
            acc0 = K.local_scalar("int32", init=0)
            acc1 = K.local_scalar("int32", init=0)

            def descriptor_offset(k16):
                per_band = band_width // 16
                return (k16 % per_band) * 2 + (k16 // per_band) * (128 * band_width * 2 // 16)

            def issue_qk(score_stage, k_stage):
                k_desc = K.SmemDescriptor()
                k_desc.init(
                    kv_smem.ptr_to([k_stage * 128 * d]),
                    ldo=1 if d == 96 else 1024,
                    sdo=band_width,
                    swizzle=smem_swizzle,
                )
                k_desc.make_lo_uniform()
                for k16 in range(d // 16):
                    with K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx[_MMA_F16](
                            K.cast(tmem_base + score_stage * 128, "uint32"),
                            q_desc.add_16B_offset(descriptor_offset(k16)),
                            k_desc.add_16B_offset(descriptor_offset(k16)),
                            K.uint32(qk_idesc),
                            K.uint32(0),
                            K.uint32(0),
                            K.uint32(0),
                            K.uint32(0),
                            K.cast(k16 != 0, "bool"),
                        )

            def issue_pv(score_stage, v_stage, accumulate, phase):
                v_desc = K.SmemDescriptor()
                v_desc.init(
                    kv_smem.ptr_to([v_stage * 128 * d]),
                    ldo=512 if d == 96 else 1024,
                    sdo=band_width,
                    swizzle=smem_swizzle,
                )
                v_desc.make_lo_uniform()
                for k16 in range(8):
                    if k16 == 6:
                        _wait(plast_full, score_stage, phase)
                    with K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx[_MMA_F16](
                            K.cast(tmem_base + 256 + score_stage * d, "uint32"),
                            K.cast(tmem_base + 64 + score_stage * 128 + k16 * 8, "uint32"),
                            v_desc.add_16B_offset(k16 * 2 * band_width),
                            K.uint32(pv_idesc),
                            K.uint32(0),
                            K.uint32(0),
                            K.uint32(0),
                            K.uint32(0),
                            K.cast(accumulate != 0 if k16 == 0 else True, "bool"),
                        )

            with K.While(work_valid != 0):
                load_tile_metadata()
                K.assign(acc0, 0)
                K.assign(acc1, 0)
                with K.If(has_work), K.Then():
                    _wait(q_full, 0, q_consumer_phase)
                    K.assign(q_consumer_phase, q_consumer_phase ^ 1)
                    for score_stage in range(2):
                        _wait(kv_full, kv_stage, kv_phase)
                        issue_qk(score_stage, kv_stage)
                        with K.If(K.cuda.elect_sync()), K.Then():
                            K.ptx[_TCGEN_COMMIT](spo_full.ptr_to([score_stage]))
                            K.ptx[_TCGEN_COMMIT](kv_empty.ptr_to([kv_stage]))
                        advance_ring(kv_stage, kv_phase)

                    pair_count = (block_iter_count - 2) // 2
                    pair = K.local_scalar("int32", init=0)
                    with K.While(pair < pair_count):
                        for score_stage in range(2):
                            phase = K.if_then_else(score_stage == 0, spo_phase0, spo_phase1)
                            _wait(kv_full, kv_stage, kv_phase)
                            v_release_stage = K.local_scalar("int32", init=kv_stage)
                            advance_ring(kv_stage, kv_phase)
                            _wait(kv_full, kv_stage, kv_phase)
                            _wait(spo_empty, score_stage, phase)
                            issue_pv(
                                score_stage,
                                v_release_stage,
                                K.if_then_else(score_stage == 0, acc0, acc1),
                                phase,
                            )
                            issue_qk(score_stage, kv_stage)
                            with K.If(K.cuda.elect_sync()), K.Then():
                                K.ptx[_TCGEN_COMMIT](spo_full.ptr_to([score_stage]))
                            if score_stage == 0:
                                K.assign(spo_phase0, spo_phase0 ^ 1)
                                K.assign(acc0, 1)
                            else:
                                K.assign(spo_phase1, spo_phase1 ^ 1)
                                K.assign(acc1, 1)
                            with K.If(K.cuda.elect_sync()), K.Then():
                                K.ptx[_TCGEN_COMMIT](kv_empty.ptr_to([v_release_stage]))
                                K.ptx[_TCGEN_COMMIT](kv_empty.ptr_to([kv_stage]))
                            advance_ring(kv_stage, kv_phase)
                        K.assign(pair, pair + 1)
                    with K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx[_TCGEN_COMMIT](q_empty.ptr_to([0]))

                    for score_stage in range(2):
                        phase = K.if_then_else(score_stage == 0, spo_phase0, spo_phase1)
                        _wait(kv_full, kv_stage, kv_phase)
                        _wait(spo_empty, score_stage, phase)
                        issue_pv(
                            score_stage,
                            kv_stage,
                            K.if_then_else(score_stage == 0, acc0, acc1),
                            phase,
                        )
                        with K.If(K.cuda.elect_sync()), K.Then():
                            K.ptx[_TCGEN_COMMIT](oacc_full.ptr_to([score_stage]))
                            K.ptx[_TCGEN_COMMIT](kv_empty.ptr_to([kv_stage]))
                        advance_ring(kv_stage, kv_phase)
                    K.assign(spo_phase0, spo_phase0 ^ 1)
                    K.assign(spo_phase1, spo_phase1 ^ 1)
                advance_work()

            K.ptx[_TMEM_RELINQUISH]()
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            allocated = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(allocated, tmem_mailbox.ptr_to([0]))
            K.ptx[_TMEM_DEALLOC](allocated, K.uint32(_TMEM_COLS))

        with r_epilogue:
            oepi_consumer_phase = K.local_scalar("int32", init=0)
            with K.While(work_valid != 0):
                load_tile_metadata()
                _wait(oepi_full, 0, oepi_consumer_phase)
                with K.unroll(num_bands) as band:
                    if pack:
                        K.ptx[_TMA_S2G_5D](
                            K.address_of(o_map),
                            K.cast(band * band_width, "int32"),
                            K.int32(0),
                            (q_block * 128) // ratio,
                            head,
                            batch_idx,
                            o_smem.ptr_to([band * 128 * band_width]),
                            _TMA_CACHE,
                        )
                    else:
                        K.ptx[_TMA_S2G_4D](
                            K.address_of(o_map),
                            K.cast(band * band_width, "int32"),
                            q_block * 128,
                            head,
                            batch_idx,
                            o_smem.ptr_to([band * 128 * band_width]),
                            _TMA_CACHE,
                        )
                K.ptx.cp.async_.bulk.commit_group()
                K.ptx.cp.async_.bulk.wait_group.read(0)
                oepi_empty.arrive(0)
                K.assign(oepi_consumer_phase, oepi_consumer_phase ^ 1)
                advance_work()

        with r_softmax:
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            tmem_base = K.local_scalar("uint32", init=K.uint32(0))
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            score_stage = K.if_then_else(warp < 4, 0, 1)
            local_warp = warp & 3
            tid128 = tid & 127
            row_hi = K.shift_left(K.cast(local_warp * 32, "uint32"), K.uint32(16))
            score_phase = K.local_scalar("int32", init=0)
            stats_producer_phase = K.local_scalar("int32", init=1)
            row_max = K.local_scalar("float32", init=K.float32(_NEG_INF))
            row_sum = K.local_scalar("float32", init=K.float32(0.0))
            with K.While(work_valid != 0):
                load_tile_metadata()
                K.assign(row_max, K.float32(_NEG_INF))
                K.assign(row_sum, K.float32(0.0))
                _wait(stats_empty, score_stage, stats_producer_phase)
                K.assign(stats_producer_phase, stats_producer_phase ^ 1)
                with K.If(has_work), K.Then():
                    work_groups = block_iter_count // 2

                    def consume_score(logical, first):
                        _wait(spo_full, score_stage, score_phase)
                        score = K.alloc_local((128,), "float32")
                        with K.unroll(4) as chunk:
                            regs = K.alloc_local((32,), "float32")
                            _tmem_load32(regs, tmem_base + score_stage * 128 + chunk * 32 + row_hi)
                            with K.unroll(32) as j:
                                K.assign(score[chunk * 32 + j], regs[j])

                        if first and (has_sizes or has_nums):
                            block_size = K.local_scalar("int32")
                            if has_sizes:
                                K.assign(
                                    block_size,
                                    K.if_then_else(
                                        (logical < raw_count) if has_nums else K.bool(True),
                                        _load_i32(block_sizes, sparse_id(logical)),
                                        0,
                                    ),
                                )
                            else:
                                K.assign(block_size, K.if_then_else(logical < raw_count, 128, 0))
                            _apply_mask128(score, block_size)
                        elif has_sizes:
                            _apply_mask128(score, _load_i32(block_sizes, sparse_id(logical)))

                        old_scale = K.local_scalar("float32", init=K.float32(0.0))
                        new_max = K.local_scalar("float32")
                        max_safe = K.local_scalar("float32")
                        if first:
                            tile_max = _reduce_max_128(score)
                            K.assign(new_max, tile_max)
                            K.assign(
                                max_safe,
                                K.if_then_else(tile_max != K.float32(_NEG_INF), tile_max, 0.0),
                            )
                        else:
                            tile_max = _reduce_max_128(score, row_max)
                            K.assign(new_max, tile_max)
                            K.assign(
                                max_safe,
                                K.if_then_else(new_max != K.float32(_NEG_INF), new_max, 0.0),
                            )
                            delta = K.local_scalar("float32")
                            K.ptx.sub.f32(delta, row_max, max_safe)
                            delta_scaled = K.local_scalar("float32")
                            K.ptx.mul.f32(delta_scaled, delta, softmax_scale_log2)
                            K.assign(old_scale, _exp2(delta_scaled))
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
                                if (pair * 2) % 10 >= 6 and fragment < 3:
                                    _ex2_emulation_2(score, base)
                                else:
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
                                spo_empty.arrive(score_stage)
                        K.ptx.tcgen05.wait__st.sync.aligned()
                        K.cuda.warp_sync()
                        with K.If(K.cuda.elect_sync()), K.Then():
                            plast_full.arrive(score_stage)

                        _wait(stats_empty, score_stage, stats_producer_phase)
                        K.assign(row_sum, _packed_sum_128(score, row_sum, old_scale, first))
                        K.assign(row_max, new_max)
                        K.assign(stats_producer_phase, stats_producer_phase ^ 1)
                        K.assign(score_phase, score_phase ^ 1)

                    consume_score(block_iter_count - 1 - score_stage, True)
                    iteration = K.local_scalar("int32", init=1)
                    with K.While(iteration < work_groups):
                        consume_score(block_iter_count - 1 - (iteration * 2 + score_stage), False)
                        K.assign(iteration, iteration + 1)

                    _st_shared_f32(stats_smem, score_stage * 128 + tid128, row_sum)
                    _st_shared_f32(stats_smem, 256 + score_stage * 128 + tid128, row_max)
                    _stats_arrive(score_stage, local_warp)
                with K.If(has_work == K.bool(False)), K.Then():
                    _stats_arrive(score_stage, local_warp)
                advance_work()
            _wait(stats_empty, score_stage, stats_producer_phase)
            K.ptx.bar.arrive(K.uint32(2), K.uint32(416))

        with r_correction:
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            tmem_base = K.local_scalar("uint32", init=K.uint32(0))
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            corr_warp = warp - 8
            tid128 = tid & 127
            row_hi = K.shift_left(K.cast(corr_warp * 32, "uint32"), K.uint32(16))
            oacc_consumer_phase = K.local_scalar("int32", init=0)
            oepi_producer_phase = K.local_scalar("int32", init=1)
            spo_empty.arrive(0)
            spo_empty.arrive(1)

            def rescale_o(score_stage, scale):
                with K.unroll(d // 16) as chunk:
                    values = K.alloc_local((16,), "float32")
                    _tmem_load16(values, tmem_base + 256 + score_stage * d + chunk * 16 + row_hi)
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
                    _tmem_store16(values, tmem_base + 256 + score_stage * d + chunk * 16 + row_hi)
                K.ptx.tcgen05.wait__st.sync.aligned()

            def output_addresses(chunk):
                chunks_per_band = band_width // 16
                band = chunk // chunks_per_band
                local_chunk = chunk - band * chunks_per_band
                raw_byte = offsets["so"] + band * 128 * band_width * 2 + tid128 * band_width * 2
                xor_mask = 48 if band_width == 32 else 112
                first_byte = K.bitwise_xor(
                    raw_byte, K.bitwise_and(K.shift_right(raw_byte, K.uint32(3)), xor_mask)
                )
                if band_width == 32:
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

            def store_combined(scale0, scale1):
                for chunk in range(d // 16):
                    values0 = K.alloc_local((16,), "float32")
                    values1 = K.alloc_local((16,), "float32")
                    combined = K.alloc_local((16,), "float32")
                    _tmem_load16(values0, tmem_base + 256 + chunk * 16 + row_hi)
                    _tmem_load16(values1, tmem_base + 256 + d + chunk * 16 + row_hi)
                    with K.unroll(8) as pair:
                        base = pair * 2
                        scaled0 = K.alloc_local((2,), "float32")
                        scaled1 = K.alloc_local((2,), "float32")
                        _packed(
                            "mul.rn.f32x2",
                            scaled0,
                            0,
                            values0[base],
                            values0[base + 1],
                            scale0,
                            scale0,
                        )
                        _packed(
                            "mul.rn.f32x2",
                            scaled1,
                            0,
                            values1[base],
                            values1[base + 1],
                            scale1,
                            scale1,
                        )
                        _packed(
                            "add.rn.f32x2",
                            combined,
                            base,
                            scaled0[0],
                            scaled0[1],
                            scaled1[0],
                            scaled1[1],
                        )
                    words = K.alloc_local((8,), "uint32")
                    with K.unroll(8) as pair:
                        if dtype == "bfloat16":
                            K.ptx.cvt.rn.bf16x2.f32(
                                words[pair], combined[pair * 2 + 1], combined[pair * 2]
                            )
                        else:
                            K.ptx.cvt.rn.f16x2.f32(
                                words[pair], combined[pair * 2 + 1], combined[pair * 2]
                            )
                    first_byte, second_byte = output_addresses(chunk)
                    K.ptx.st.shared.v4.u32(
                        arena.ptr_to([first_byte]), words[0], words[1], words[2], words[3]
                    )
                    K.ptx.st.shared.v4.u32(
                        arena.ptr_to([second_byte]), words[4], words[5], words[6], words[7]
                    )

            def store_zero():
                for chunk in range(d // 16):
                    first_byte, second_byte = output_addresses(chunk)
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

            with K.While(work_valid != 0):
                load_tile_metadata()
                total_sum = K.local_scalar("float32", init=K.float32(0.0))
                safe_max = K.local_scalar("float32", init=K.float32(0.0))
                with K.If(has_work):
                    with K.Then():
                        _stats_sync(0, corr_warp)
                        stats_empty.arrive(0)
                        _stats_sync(1, corr_warp)
                        pair_count = (block_iter_count - 2) // 2
                        pair = K.local_scalar("int32", init=0)
                        with K.While(pair < pair_count):
                            for score_stage in range(2):
                                _stats_sync(score_stage, corr_warp)
                                scale = _ld_shared_f32(stats_smem, score_stage * 128 + tid128)
                                ballot = K.local_scalar("uint32")
                                K.ptx.vote_sync.ballot.b32(
                                    ballot, K.ptx.pred(scale < K.float32(1.0)), K.uint32(0xFFFFFFFF)
                                )
                                with K.If(ballot != 0), K.Then():
                                    rescale_o(score_stage, scale)
                                spo_empty.arrive(score_stage)
                                stats_empty.arrive(1 - score_stage)
                            K.assign(pair, pair + 1)
                        stats_empty.arrive(1)

                        sum0 = K.local_scalar("float32")
                        sum1 = K.local_scalar("float32")
                        maximum0 = K.local_scalar("float32")
                        maximum1 = K.local_scalar("float32")
                        for score_stage in range(2):
                            _stats_sync(score_stage, corr_warp)
                            if score_stage == 0:
                                K.assign(sum0, _ld_shared_f32(stats_smem, tid128))
                                K.assign(maximum0, _ld_shared_f32(stats_smem, 256 + tid128))
                            else:
                                K.assign(sum1, _ld_shared_f32(stats_smem, 128 + tid128))
                                K.assign(maximum1, _ld_shared_f32(stats_smem, 384 + tid128))
                            stats_empty.arrive(score_stage)
                        valid0 = (sum0 != K.float32(0.0)) & (sum0 == sum0)
                        valid1 = (sum1 != K.float32(0.0)) & (sum1 == sum1)
                        rm0 = K.if_then_else(valid0, maximum0, K.float32(_NEG_INF))
                        rm1 = K.if_then_else(valid1, maximum1, K.float32(_NEG_INF))
                        maximum = K.local_scalar("float32")
                        K.ptx.max.f32(maximum, rm0, rm1)
                        K.assign(
                            safe_max, K.if_then_else(maximum != K.float32(_NEG_INF), maximum, 0.0)
                        )
                        scale0 = K.local_scalar("float32")
                        scale1 = K.local_scalar("float32")
                        K.assign(
                            scale0,
                            K.if_then_else(
                                valid0, _exp2((rm0 - safe_max) * softmax_scale_log2), 0.0
                            ),
                        )
                        K.assign(
                            scale1,
                            K.if_then_else(
                                valid1, _exp2((rm1 - safe_max) * softmax_scale_log2), 0.0
                            ),
                        )
                        K.assign(total_sum, sum0 * scale0 + sum1 * scale1)
                        valid_total = (total_sum != K.float32(0.0)) & (total_sum == total_sum)
                        inv_sum = _rcp(K.if_then_else(valid_total, total_sum, 1.0))
                        final_scale0 = scale0 * inv_sum
                        final_scale1 = scale1 * inv_sum
                        _wait(oacc_full, 0, oacc_consumer_phase)
                        _wait(oacc_full, 1, oacc_consumer_phase)
                        _wait(oepi_empty, 0, oepi_producer_phase)
                        store_combined(final_scale0, final_scale1)
                        K.ptx.fence.proxy.async_.shared__cta()
                        spo_empty.arrive(0)
                        spo_empty.arrive(1)
                        K.assign(oacc_consumer_phase, oacc_consumer_phase ^ 1)
                    with K.Else():
                        _stats_sync(0, corr_warp)
                        stats_empty.arrive(0)
                        _stats_sync(1, corr_warp)
                        stats_empty.arrive(1)
                        _wait(oepi_empty, 0, oepi_producer_phase)
                        store_zero()
                        K.ptx.fence.proxy.async_.shared__cta()

                oepi_full.arrive(0)
                K.assign(oepi_producer_phase, oepi_producer_phase ^ 1)
                if return_lse:
                    packed_row = q_block * 128 + tid128
                    if pack:
                        token = packed_row // ratio
                        local_head = packed_row - token * ratio
                        q_head = head * ratio + local_head
                    else:
                        token = packed_row
                        q_head = head
                    with K.If(token < seqlen_q), K.Then():
                        lse_value = K.local_scalar("float32", init=K.float32(_NEG_INF))
                        with K.If(total_sum > K.float32(0.0)), K.Then():
                            K.assign(
                                lse_value,
                                (safe_max * softmax_scale_log2 + _log2(total_sum))
                                * K.float32(_LN2),
                            )
                        lse_index = (batch_idx * hq + q_head) * seqlen_q + token
                        K.ptx.st.global_.f32(lse.ptr_to([lse_index]), lse_value)
                advance_work()
            _wait(oepi_empty, 0, oepi_producer_phase)
            K.ptx.bar.arrive(K.uint32(2), K.uint32(416))

    def kernel(
        q,
        k,
        v,
        out,
        lse,
        block_index,
        block_sizes,
        block_nums,
        block_sparse_num,
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
                block_index,
                block_sizes,
                block_nums,
                block_sparse_num,
                softmax_scale_log2,
                host,
            )

    kernel.__annotations__ = {
        "q": K.gptr[elem_dtype, (batch * hq * seqlen_q * d,)],
        "k": K.gptr[elem_dtype, (batch * hkv * seqlen_kv * d,)],
        "v": K.gptr[elem_dtype, (batch * hkv * seqlen_kv * d,)],
        "out": K.gptr[elem_dtype, (batch * hq * seqlen_q * d,)],
        "lse": K.gptr[K.f32, (batch * hq * seqlen_q,)],
        "block_index": K.gptr[K.i32, (batch * scheduled_heads * q_blocks * max_blocks,)],
        "block_sizes": K.gptr[K.i32, (physical_blocks,)],
        "block_nums": K.gptr[K.i32, (batch * scheduled_heads * q_blocks,)],
        "block_sparse_num": K.i32,
        "softmax_scale_log2": K.f32,
    }
    return K.kernel(
        warps=_WARPS,
        arch="sm_100a",
        grid=[q_blocks, scheduled_heads, batch],
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


def _metadata_shape(config):
    import math

    ratio = config["num_q_heads"] // config["num_kv_heads"]
    if _pack_enabled(config):
        return config["num_kv_heads"], math.ceil(config["seqlen_q"] * ratio / 128)
    return config["num_q_heads"], math.ceil(config["seqlen_q"] / 128)


def _counts(torch, config, metadata_heads, q_blocks, device):
    maximum = config["kv_blocks"]
    mode = config["block_count_mode"]
    if mode == "fixed":
        return torch.full(
            (config["batch"], metadata_heads, q_blocks), maximum, dtype=torch.int32, device=device
        )
    values = []
    for item in config["block_count_pattern"].split(","):
        item = item.strip()
        if item == "max":
            values.append(maximum)
        elif item == "mid":
            values.append(max(1, maximum // 2))
        else:
            values.append(int(item))
    flat = torch.arange(config["batch"] * metadata_heads * q_blocks, device=device)
    choices = torch.tensor(values, dtype=torch.int32, device=device)
    return choices[flat % len(values)].reshape(config["batch"], metadata_heads, q_blocks)


def _metadata(torch, config, metadata_heads, q_blocks, device):
    import math

    maximum = config["kv_blocks"]
    physical_blocks = math.ceil(config["seqlen_kv"] / 128)
    usable_blocks = physical_blocks if config["has_block_sizes"] else config["seqlen_kv"] // 128
    if maximum > usable_blocks:
        raise ValueError(f"{config.get('label', '<config>')}: kv_blocks exceeds usable blocks")
    rows = torch.empty(
        (config["batch"], metadata_heads, q_blocks, maximum), dtype=torch.int32, device=device
    )
    base = torch.arange(maximum, dtype=torch.int64, device=device)
    stride = max(1, usable_blocks // maximum)
    for linear in range(config["batch"] * metadata_heads * q_blocks):
        values = (base * stride + linear * 7) % usable_blocks
        if maximum:
            values[-1] = usable_blocks - 1
        rows.reshape(-1, maximum)[linear] = values.to(torch.int32)
    return rows


def _strided_kv(torch, generator, shape, dtype, use_i64):
    tensor = torch.randn(shape, dtype=dtype, generator=generator, device="cuda")
    if not use_i64:
        return tensor
    if shape[0] != 1:
        raise ValueError("the Int64 KV stride probe requires batch=1")
    return torch.as_strided(tensor, shape, (2**31, *tensor.stride()[1:]))


def prepare_data(**config):
    import math

    import torch

    dtype = _torch_dtype(torch, config["dtype"])
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    q = torch.randn(
        config["batch"],
        config["num_q_heads"],
        config["seqlen_q"],
        config["head_dim"],
        dtype=dtype,
        generator=generator,
        device="cuda",
    )
    k = _strided_kv(
        torch,
        generator,
        (config["batch"], config["num_kv_heads"], config["seqlen_kv"], config["head_dim"]),
        dtype,
        config["use_int64_kv_strides"],
    )
    v = _strided_kv(
        torch,
        generator,
        (config["batch"], config["num_kv_heads"], config["seqlen_kv"], config["head_dim"]),
        dtype,
        config["use_int64_kv_strides"],
    )
    if config["tensor_layout"] == "bshd":
        q_user = q.transpose(1, 2).contiguous()
        k_user = k.transpose(1, 2).contiguous()
        v_user = v.transpose(1, 2).contiguous()
        q = q_user.transpose(1, 2).contiguous()
        k = k_user.transpose(1, 2).contiguous()
        v = v_user.transpose(1, 2).contiguous()
    else:
        q_user, k_user, v_user = q, k, v

    metadata_heads, q_blocks = _metadata_shape(config)
    block_index = _metadata(torch, config, metadata_heads, q_blocks, q.device)
    block_nums = _counts(torch, config, metadata_heads, q_blocks, q.device)
    physical_blocks = math.ceil(config["seqlen_kv"] / 128)
    block_sizes = torch.full((physical_blocks,), 128, dtype=torch.int32, device=q.device)
    block_sizes[-1] = config["seqlen_kv"] - (physical_blocks - 1) * 128
    scale = (
        config["head_dim"] ** -0.5
        if config["softmax_scale"] is None
        else float(config["softmax_scale"])
    )

    def outputs():
        return {
            "out": torch.empty_like(q),
            "lse": torch.full(
                (config["batch"], config["num_q_heads"], config["seqlen_q"]),
                12345.25,
                dtype=torch.float32,
                device="cuda",
            ),
        }

    return {
        "config": dict(config),
        "q": q,
        "k": k,
        "v": v,
        "q_user": q_user,
        "k_user": k_user,
        "v_user": v_user,
        "block_index": block_index,
        "block_nums": block_nums,
        "block_sizes": block_sizes,
        "softmax_scale": scale,
        "tirx": outputs(),
        "source": outputs(),
    }


def _tirx_launch(executable, data):
    arguments = (
        data["q"].reshape(-1),
        data["k"].reshape(-1),
        data["v"].reshape(-1),
        data["tirx"]["out"].reshape(-1),
        data["tirx"]["lse"].reshape(-1),
        data["block_index"].reshape(-1),
        data["block_sizes"].reshape(-1),
        data["block_nums"].reshape(-1),
        data["config"]["kv_blocks"],
        data["softmax_scale"] * _LOG2_E,
    )

    def launch():
        executable(*arguments)

    launch._keep_alive = arguments
    return launch


def _compile_reference(data):
    from tirx_kernels.cudnn._reference import load_reference_module

    interface = load_reference_module("cudnn.block_sparse_attention._interface")
    config = data["config"]
    source_out_shape = (
        (config["batch"], config["num_q_heads"], config["seqlen_q"], config["head_dim"])
        if config["tensor_layout"] == "bhsd"
        else (config["batch"], config["seqlen_q"], config["num_q_heads"], config["head_dim"])
    )
    import torch

    source_out_storage = torch.empty(
        source_out_shape, dtype=_torch_dtype(torch, config["dtype"]), device="cuda"
    )

    def call_reference(q, k, v, out):
        return interface.bsa_attn_fwd(
            q,
            k,
            v,
            data["block_index"],
            config["kv_blocks"] if config["block_count_mode"] == "fixed" else 0,
            block_sizes=data["block_sizes"] if config["has_block_sizes"] else None,
            q2k_block_nums=(data["block_nums"] if config["block_count_mode"] != "fixed" else None),
            allow_empty_block_nums=config["block_count_mode"] == "variable_empty",
            softmax_scale=config["softmax_scale"],
            pack_gqa=None if config["pack_gqa"] == "auto" else config["pack_gqa"],
            return_lse=config["return_lse"],
            out=out,
            layout=config["tensor_layout"],
        )

    priming_tensors = ()
    if config["use_int64_kv_strides"]:
        compact_k = torch.empty(
            data["k_user"].shape, dtype=data["k_user"].dtype, device=data["k_user"].device
        ).copy_(data["k_user"])
        compact_v = torch.empty(
            data["v_user"].shape, dtype=data["v_user"].dtype, device=data["v_user"].device
        ).copy_(data["v_user"])
        compact_out = torch.empty_like(source_out_storage)
        call_reference(data["q_user"], compact_k, compact_v, compact_out)
        torch.cuda.synchronize()
        priming_tensors = (compact_k, compact_v, compact_out)

    def launch():
        source_out, source_lse = call_reference(
            data["q_user"], data["k_user"], data["v_user"], source_out_storage
        )
        if config["tensor_layout"] == "bshd":
            source_out = source_out.transpose(1, 2).contiguous()
        data["source"]["out"] = source_out
        data["source"]["lse"] = source_lse

    launch._keep_alive = (interface, source_out_storage, priming_tensors)
    return launch


def _oracle(data):
    import torch

    config = data["config"]
    q, k, v = data["q"], data["k"], data["v"]
    out = torch.zeros_like(q)
    lse = torch.full(
        (config["batch"], config["num_q_heads"], config["seqlen_q"]),
        -float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    ratio = config["num_q_heads"] // config["num_kv_heads"]
    packed = _pack_enabled(config)
    positions = torch.arange(config["seqlen_q"], device=q.device)
    for batch_idx in range(config["batch"]):
        for q_head in range(config["num_q_heads"]):
            kv_head = q_head // ratio
            metadata_head = kv_head if packed else q_head
            local_head = q_head - kv_head * ratio
            tile_for_position = (
                (positions * ratio + local_head) // 128 if packed else positions // 128
            )
            for q_tile in tile_for_position.unique().tolist():
                query_positions = positions[tile_for_position == q_tile]
                count = int(data["block_nums"][batch_idx, metadata_head, q_tile])
                if count == 0:
                    continue
                block_ids = data["block_index"][batch_idx, metadata_head, q_tile, :count].tolist()
                key_parts = []
                value_parts = []
                for block_id in block_ids:
                    valid = int(data["block_sizes"][block_id]) if config["has_block_sizes"] else 128
                    start = block_id * 128
                    stop = min(start + valid, config["seqlen_kv"])
                    key_parts.append(k[batch_idx, kv_head, start:stop].float())
                    value_parts.append(v[batch_idx, kv_head, start:stop].float())
                keys = torch.cat(key_parts, dim=0)
                values = torch.cat(value_parts, dim=0)
                scores = q[batch_idx, q_head, query_positions].float() @ keys.T
                scores *= data["softmax_scale"]
                probabilities = torch.softmax(scores, dim=-1)
                out[batch_idx, q_head, query_positions] = (probabilities @ values).to(q.dtype)
                lse[batch_idx, q_head, query_positions] = torch.logsumexp(scores, dim=-1)
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
        atol=0.03,
        rtol=0.03,
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
            atol=0.002,
            rtol=0.002,
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
                atol=0.03,
                rtol=0.03,
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
                    atol=0.002,
                    rtol=0.002,
                )
        empty = ~torch.isfinite(oracle_lse)
        if empty.any():
            for name, values in (("tirx", tirx), ("source", source)):
                if not torch.equal(values["out"][empty], torch.zeros_like(values["out"][empty])):
                    raise AssertionError(f"{name} empty sparse rows are not bitwise zero")
                if config["return_lse"] and not torch.isneginf(values["lse"][empty]).all():
                    raise AssertionError(f"{name} empty sparse LSE is not exactly -inf")
        metrics = {
            "tirx_oracle_o_max_abs": float((tirx["out"].float() - oracle_out.float()).abs().max()),
            "source_oracle_o_max_abs": float(
                (source["out"].float() - oracle_out.float()).abs().max()
            ),
        }
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
