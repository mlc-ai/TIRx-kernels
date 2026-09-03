# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ cc6e8794), Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Cake VSA longseq block-sparse attention forward for SM103a.

Upstream sources (FlashInfer @ cc6e8794c49bf66172627bdb9742fcb17d18b839):

- csrc/cake_vsa/cake_vsa_longseq_sm_103a.cu supplies
  kernel_flashinfer_blackwell_vsa_longseq_warp_specialized_sm100;
- csrc/cake_vsa/cake_vsa_longseq_host.cpp encodes three rank-4,
  SWIZZLE_128B tensor maps and launches 512 threads with 201344 dynamic
  shared-memory bytes;
- flashinfer/cake_vsa.py selects this profile for BF16, D=128, MHA with
  eight heads, N >= 16384, and a uniform top-k no larger than 192 blocks.

Each CTA handles one 128-row query block and one head. Warps 0-7 perform two
online-softmax instances, warps 8-11 correct and store the output, warp 12
issues QK/PV tensor-core operations, warp 13 produces Q and the three-stage K
ring, warp 14 produces the two-stage V ring, and warp 15 is idle. Every shared
object occupies one rank-one byte arena and is addressed by explicit offsets;
no first-class layouts or tile primitives are used.
"""

import math
import os
from functools import lru_cache
from typing import Any

import tirx_kernels.kern as K

KERNEL_META = {
    "name": "cake_vsa_longseq_sm103",
    "category": "flashinfer",
    "runtime_cuda_archs": ["sm_103a"],
    "reference_requirements": (
        {
            "package": "flashinfer-python",
            "git": {
                "url": "https://github.com/flashinfer-ai/flashinfer.git",
                "commit": "cc6e8794c49bf66172627bdb9742fcb17d18b839",
            },
            "import": "flashinfer",
        },
    ),
}
# The upstream kernel is a prebuilt cubin loaded through FlashInfer's own tvm-ffi
# launcher (``flashinfer.cake_vsa._load_module``); it does not use the CuTe DSL, and
# tvm-ffi is a dependency of flashinfer-python itself, so only FlashInfer is pinned.

# ---------------------------------------------------------------------------
# Source contract constants (host.cpp / cake_vsa.py / the device kernel).
# ---------------------------------------------------------------------------
HEAD_DIM = 128
BLOCK = 128  # R == C == 128 query/KV block size of this profile
MAX_SELECTED_BLOCKS = 192  # ``union_blocks`` capacity per instance
THREADS = 512
NUM_WARPS = THREADS // 32
SMEM_TOTAL = 201344
TMEM_COLS = 512
PROFILE = "longseq"
CUDA_ARCH = "sm_103a"
_PTXAS_REGISTER_USAGE_LEVEL = "5"


def _config(
    label: str,
    *,
    M: int,
    N: int,
    selected: int,
    pattern: str,
    return_lse: bool = False,
    return_temperature_lse: bool = False,
    lse_temperature_scale: float = 1.0,
    q_amp: float = 1.0,
    sm_scale: float | None = None,
    seed: int = 0,
    oracle_atol: float = 1.5e-3,
    oracle_rtol: float = 6.0e-3,
    oracle_lse_atol: float = 2.0e-6,
    oracle_lse_rtol: float = 5.0e-7,
    ulp_max: int = 2,
    ulp_le1_frac: float = 0.9999,
    ulp_le2_frac: float = 1.0,
) -> dict[str, Any]:
    return {
        "label": label,
        "M": M,
        "N": N,
        "num_heads": 8,
        "selected": selected,
        "pattern": pattern,
        "return_lse": return_lse,
        "return_temperature_lse": return_temperature_lse,
        "lse_temperature_scale": lse_temperature_scale,
        "q_amp": q_amp,
        "sm_scale": sm_scale,
        "seed": seed,
        "oracle_atol": oracle_atol,
        "oracle_rtol": oracle_rtol,
        "oracle_lse_atol": oracle_lse_atol,
        "oracle_lse_rtol": oracle_lse_rtol,
        "ulp_max": ulp_max,
        "ulp_le1_frac": ulp_le1_frac,
        "ulp_le2_frac": ulp_le2_frac,
    }


# Correctness matrix: uniform selected counts are a hard ABI invariant of this profile.
CONFIGS = [
    _config("m128_n16384_h8_sel8_upstream", M=128, N=16384, selected=8, pattern="upstream"),
    _config("m256_n16512_h8_sel1_tail", M=256, N=16512, selected=1, pattern="tail", seed=1),
    _config("m512_n32768_h8_sel7_ring", M=512, N=32768, selected=7, pattern="ring", seed=2),
    _config(
        "m128_n24576_h8_sel192_capacity", M=128, N=24576, selected=192, pattern="dense", seed=3
    ),
    _config("m1024_n65536_h8_sel64_window", M=1024, N=65536, selected=64, pattern="window", seed=4),
    _config(
        "m256_n32768_h8_sel8_kramp",
        M=256,
        N=32768,
        selected=8,
        pattern="kramp",
        q_amp=6.0,
        sm_scale=0.05,
        seed=5,
        oracle_atol=6.5e-3,
        oracle_rtol=1.0e-2,
        ulp_max=8,
        ulp_le1_frac=0.995,
        ulp_le2_frac=0.9998,
    ),
    _config(
        "m128_n16384_h8_sel3_bothlse",
        M=128,
        N=16384,
        selected=3,
        pattern="ring",
        return_lse=True,
        return_temperature_lse=True,
        lse_temperature_scale=0.7,
        seed=6,
    ),
]


BENCH_CONFIGS = [
    _config("m2048_n16384_h8_sel8", M=2048, N=16384, selected=8, pattern="random", seed=100),
    _config("m8192_n32768_h8_sel64", M=8192, N=32768, selected=64, pattern="random", seed=101),
    _config(
        "m32768_n131072_h8_sel192", M=32768, N=131072, selected=192, pattern="random", seed=102
    ),
    _config("m2048_n131072_h8_sel8", M=2048, N=131072, selected=8, pattern="random", seed=103),
    _config(
        "m16384_n32768_h8_sel32_window", M=16384, N=32768, selected=32, pattern="window", seed=104
    ),
    _config(
        "m8192_n65536_h8_sel64_kramp",
        M=8192,
        N=65536,
        selected=64,
        pattern="kramp",
        q_amp=6.0,
        sm_scale=0.05,
        seed=105,
        oracle_atol=6.5e-3,
        oracle_rtol=1.0e-2,
        ulp_max=8,
        ulp_le1_frac=0.995,
        ulp_le2_frac=0.9998,
    ),
]


_LN2 = math.log(2.0)
_LSE_SENTINEL = 12345.25
_GUARD_ELEMS = 64
_OUT_GUARD = 42.5
_STATS_GUARD = -54321.25
_ORACLE_ULP_MIN_REF = 0.125
_ORACLE_TILES_PER_CHUNK = 1


def _without_label(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "label"}


def _validate_shape(M: int, N: int, num_heads: int, selected: int) -> tuple[int, int]:
    if M <= 0 or M % BLOCK != 0:
        raise ValueError(f"M={M} must be a positive multiple of {BLOCK}")
    if N < 16384 or N % BLOCK != 0:
        raise ValueError(f"N={N} must be at least 16384 and a multiple of {BLOCK}")
    if num_heads != 8:
        raise ValueError(f"longseq requires exactly 8 heads, got {num_heads}")
    nb = N // BLOCK
    if selected <= 0 or selected > min(nb, MAX_SELECTED_BLOCKS):
        raise ValueError(f"selected={selected} must be in [1, min(nb={nb}, {MAX_SELECTED_BLOCKS})]")
    return M // BLOCK, nb


def _make_block_mask(*, num_heads: int, mb: int, nb: int, selected: int, pattern: str, generator):
    """Build a deterministic (H, mb, nb) mask with exactly selected bits per row."""
    import torch

    mask = torch.zeros((num_heads, mb, nb), dtype=torch.bool)
    slots = torch.arange(selected, dtype=torch.int64)
    if pattern == "dense":
        if selected != nb:
            raise ValueError("dense pattern requires selected == nb")
        mask[:] = True
        return mask
    for head in range(num_heads):
        for row in range(mb):
            if pattern == "upstream":
                columns = (slots * 7 + row) % nb
            elif pattern == "tail":
                columns = torch.arange(nb - selected, nb)
            elif pattern == "ring":
                columns = (slots * 19 + row * 13 + head * 17) % nb
            elif pattern == "window":
                columns = (slots + row * 23 + head * 11) % nb
            elif pattern in {"random", "kramp"}:
                columns = torch.randperm(nb, generator=generator)[:selected]
            else:
                raise ValueError(f"unknown mask pattern {pattern!r}")
            if int(torch.unique(columns).numel()) != selected:
                raise ValueError(f"pattern {pattern!r} generated duplicate columns")
            mask[head, row, columns] = True
    counts = mask.sum(dim=-1)
    if not bool((counts == selected).all()):
        raise AssertionError("longseq mask must have a uniform selected-block count")
    return mask


def _assert_longseq_route(mask, *, N: int, num_heads: int, selected: int) -> None:
    counts = mask.sum(dim=-1)
    if N < 16384 or num_heads != 8:
        raise AssertionError("shape does not satisfy the public longseq predicate")
    if selected > MAX_SELECTED_BLOCKS:
        raise AssertionError("selected count exceeds longseq capacity")
    if not bool((counts == selected).all()):
        raise AssertionError("longseq requires a fixed top-k count in every row and head")


def _guarded_tensor(shape, dtype, *, fill: float, guard: float, device):
    """Return a contiguous interior view plus its guarded backing storage."""
    import math as _math

    import torch

    elements = _math.prod(shape)
    storage = torch.full((elements + 2 * _GUARD_ELEMS,), guard, dtype=dtype, device=device)
    view = storage[_GUARD_ELEMS : _GUARD_ELEMS + elements].view(shape)
    view.fill_(fill)
    return view, storage


def prepare_data(**config: Any) -> dict[str, Any]:
    """Allocate deterministic inputs and independently guarded output buffers."""
    import torch

    config = _without_label(config)
    M, N = int(config["M"]), int(config["N"])
    num_heads, selected = int(config["num_heads"]), int(config["selected"])
    mb, nb = _validate_shape(M, N, num_heads, selected)
    return_lse = bool(config["return_lse"])
    return_tlse = bool(config["return_temperature_lse"])
    sm_scale = config["sm_scale"]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    device = torch.device("cuda")

    q = torch.randn((M, num_heads, HEAD_DIM), generator=generator, dtype=torch.float32)
    q = (q * float(config["q_amp"])).to(torch.bfloat16).to(device)
    k = torch.randn((N, num_heads, HEAD_DIM), generator=generator, dtype=torch.float32)
    k = k.to(torch.bfloat16).to(device)
    v = torch.randn((N, num_heads, HEAD_DIM), generator=generator, dtype=torch.float32)
    v = v.to(torch.bfloat16).to(device)
    mask = _make_block_mask(
        num_heads=num_heads,
        mb=mb,
        nb=nb,
        selected=selected,
        pattern=config["pattern"],
        generator=generator,
    )
    _assert_longseq_route(mask, N=N, num_heads=num_heads, selected=selected)
    block_mask = mask.to(device)

    def outputs() -> dict[str, Any]:
        out, out_storage = _guarded_tensor(
            (M, num_heads, HEAD_DIM),
            torch.bfloat16,
            fill=float("nan"),
            guard=_OUT_GUARD,
            device=device,
        )
        lse_shape = (M, num_heads) if return_lse else (1,)
        lse_fill = float("nan") if return_lse else _LSE_SENTINEL
        lse, lse_storage = _guarded_tensor(
            lse_shape, torch.float32, fill=lse_fill, guard=_STATS_GUARD, device=device
        )
        if return_tlse:
            tlse, tlse_storage = _guarded_tensor(
                (M, num_heads), torch.float32, fill=float("nan"), guard=_STATS_GUARD, device=device
            )
        else:
            tlse, tlse_storage = lse, lse_storage
        return {
            "out": out,
            "lse": lse,
            "tlse": tlse,
            "guards": {"out": out_storage, "lse": lse_storage, "tlse": tlse_storage},
        }

    return {
        "config": config,
        "M": M,
        "N": N,
        "num_heads": num_heads,
        "mb": mb,
        "nb": nb,
        "selected_blocks": selected,
        "q": q,
        "k": k,
        "v": v,
        "block_mask": block_mask,
        "block_mask_u8": block_mask.view(torch.uint8),
        "selected_indices": mask.nonzero(as_tuple=False)[:, 2]
        .view(num_heads, mb, selected)
        .permute(1, 0, 2)
        .contiguous(),
        "sm_scale": float(sm_scale),
        "softmax_scale_log2": float(sm_scale) / _LN2,
        "lse_temperature_scale": float(config["lse_temperature_scale"]),
        "return_lse": return_lse,
        "return_temperature_lse": return_tlse,
        "tirx": outputs(),
        "source": outputs(),
    }


# ---------------------------------------------------------------------------
# TIRx kernel
# ---------------------------------------------------------------------------
# cuTensorMapEncodeTiled enum values (CUtensorMapInterleave / Swizzle /
# L2promotion / FloatOOBfill) as used by the upstream host launcher.
_TMA_INTERLEAVE_NONE = 0
_TMA_SWIZZLE_128B = 3
_TMA_L2_PROMOTION_NONE = 0
_TMA_OOB_FILL_NONE = 0


def _host_prelude(params):
    """Encode the three upstream tensor maps from the runtime shape scalars.

    Mirrors ``EncodeTma_q`` / ``EncodeTma_k`` / ``EncodeTma_v`` in the upstream
    host launcher: rank-4 bf16 maps, 128-byte swizzle, no L2 promotion, no OOB
    fill.  Q is ``{64, M, Hq, 2}`` with a ``{64, 64, 1, 2}`` box (one 16 KB
    transaction covers both head-dim halves of 64 rows); K and V are
    ``{64, N, 2, Hkv}`` with a ``{64, 64, 1, 1}`` box (8 KB per transaction).
    """
    mb = params["mb"]
    nb = params["nb"]
    num_q_heads = params["num_q_heads"]
    num_kv_heads = params["num_kv_heads"]
    elem_bytes = 2

    def encode(tensor, dims, strides_bytes, box):
        descriptor = K.stack_alloca("tensormap", 1)
        K.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            descriptor,
            "bfloat16",
            4,
            tensor.data,
            *dims,
            *strides_bytes,
            *box,
            1,
            1,
            1,
            1,
            _TMA_INTERLEAVE_NONE,
            _TMA_SWIZZLE_128B,
            _TMA_L2_PROMOTION_NONE,
            _TMA_OOB_FILL_NONE,
        )
        return descriptor

    q_map = encode(
        params["q"],
        (64, mb * BLOCK, num_q_heads, HEAD_DIM // 64),
        (num_q_heads * HEAD_DIM * elem_bytes, HEAD_DIM * elem_bytes, 64 * elem_bytes),
        (64, 64, 1, HEAD_DIM // 64),
    )
    kv_dims = (64, nb * BLOCK, HEAD_DIM // 64, num_kv_heads)
    kv_strides = (num_kv_heads * HEAD_DIM * elem_bytes, 64 * elem_bytes, HEAD_DIM * elem_bytes)
    kv_box = (64, 64, 1, 1)
    k_map = encode(params["k"], kv_dims, kv_strides, kv_box)
    v_map = encode(params["v"], kv_dims, kv_strides, kv_box)
    return (q_map, k_map, v_map)


# Byte offsets inside the 201344-byte dynamic SMEM arena (source macro table).
_MBAR_Q_FULL = 0
_MBAR_UNION_READY = 8
_MBAR_K_FULL = 16  # 3 stages
_MBAR_K_EMPTY = 40  # 3 stages
_MBAR_V_FULL = 64  # 2 stages
_MBAR_V_EMPTY = 80  # 2 stages
_MBAR_S_FULL = 96  # 2 instances
_MBAR_P_FULL = 112  # 2 instances
_MBAR_CORR_SIG = 128  # 2 instances
_MBAR_CORR_DONE = 144  # 2 instances
_MBAR_O_FULL = 160  # 2 instances
_MBAR_TMEM_DEALLOC = 176
_SMEM_TMEM_MAILBOX = 184
_SMEM_Q0 = 1024
_SMEM_Q1 = 17408
_SMEM_K = 33792  # 3 x 32768
_SMEM_V = 132096  # 2 x 32768
_SMEM_STAGE_BYTES = 32768
_SMEM_SCALE = 197632  # 512 f32: acc_scale / row_sum / row_max / temperature_sum
_SMEM_UNION_COUNT = 199680  # 2 i32
_SMEM_UNION_BLOCKS = 199688  # 2 x 192 i32
# (offset, arrival count) in exact source order.
_MBARRIER_INIT = (
    (_MBAR_Q_FULL, 1),
    (_MBAR_UNION_READY, 32),
    (_MBAR_K_FULL, 1),
    (_MBAR_K_FULL + 8, 1),
    (_MBAR_K_FULL + 16, 1),
    (_MBAR_K_EMPTY, 1),
    (_MBAR_K_EMPTY + 8, 1),
    (_MBAR_K_EMPTY + 16, 1),
    (_MBAR_V_FULL, 1),
    (_MBAR_V_FULL + 8, 1),
    (_MBAR_V_EMPTY, 1),
    (_MBAR_V_EMPTY + 8, 1),
    (_MBAR_S_FULL, 1),
    (_MBAR_S_FULL + 8, 1),
    (_MBAR_P_FULL, 256),
    (_MBAR_P_FULL + 8, 256),
    (_MBAR_CORR_SIG, 128),
    (_MBAR_CORR_SIG + 8, 128),
    (_MBAR_CORR_DONE, 128),
    (_MBAR_CORR_DONE + 8, 128),
    (_MBAR_O_FULL, 1),
    (_MBAR_O_FULL + 8, 1),
    (_MBAR_TMEM_DEALLOC, 128),
)
# TMEM column bases per softmax instance: S (128 columns; P overwrites the upper 64
# in place) and the O accumulator (128 columns).
_TMEM_SCORES = (0, 256)
_TMEM_OUTPUT = (128, 384)
_TMEM_P_OFFSET = 64
# UMMA descriptor words and instruction descriptors (source immediates).
_DESC_HI = 0x40004040  # SBO 1024 B, base-offset bit, 128 B swizzle
_QK_IDESC = 0x04200490  # bf16 x bf16 -> f32, M=64, N=128, K-major A and B
_PV_IDESC = 0x04210490  # same with MN-major B (V is token-major)
_QK_A_STEPS = (2, 2, 2, 506, 2, 2, 2)  # Q low-word walk over K=128 (bytes/16)
_QK_B_STEPS = (2, 2, 2, 1018, 2, 2, 2)  # K low-word walk (second dim half at +16384 B)
_V_LBO_BIT = 0x4000000  # LBO = 16384 B: dim-half stride of the V tile
_PV_B_STEP = 128  # 2048 B = 16 tokens x 128 B per K16 step
_PV_A_STEP = 8  # 16 bf16 P columns = 8 TMEM 32-bit columns per K16 step
_REG_SOFTMAX = 192
_REG_CORRECTION = 80
_REG_OTHER = 48
_WAIT_TICKS = 0x989680

_TMA_G2S_4D = "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes"
_MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
_TCGEN05_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
_TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
_TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
_TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
_TMEM_LD_X32 = "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
_TMEM_ST_X16 = "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
_TMEM_ST_X64 = "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
_LN2_F32 = 0.6931471805599453
_NEG_INF = float("-inf")
_FULL_MASK = 0xFFFFFFFF


def _u32(value):
    return K.uint32(value)


def _i32(value):
    return K.int32(value)


def _f32(value):
    return K.float32(value)


def _mbar_wait(addr, phase):
    """``mbarrier.try_wait.parity.acquire.cta`` retry loop with the source's suspend hint."""
    K.cuda.mbarrier_wait(addr, phase)


def _mbar_arrive(addr):
    K.ptx.mbarrier.arrive.release.cta.shared__cta.b64(addr)


def _mbar_expect_tx(addr, tx_bytes):
    K.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(addr, _u32(tx_bytes))


def _ld_shared_i32(addr):
    value = K.local_scalar("int32")
    K.ptx.ld.shared.b32(value, addr)
    return value


def _ld_shared_f32(addr):
    value = K.local_scalar("float32")
    K.ptx.ld.shared.b32(value, addr)
    return value


def _ld_shared_counts(addr):
    """Both instance counts in one ``ld.shared.v2.b32`` (the source's vectorized pair)."""
    count0 = K.local_scalar("int32")
    count1 = K.local_scalar("int32")
    K.ptx.ld.shared.v2.b32(count0, count1, addr)
    return count0, count1


def _flip(phase):
    K.assign(phase, phase ^ _i32(1))


def _advance_ring(stage, phase, stages):
    """Advance a K/V ring cursor and flip parity exactly at its stage count."""
    K.assign(stage, stage + _u32(1))
    with K.If(stage == _u32(stages)), K.Then():
        K.assign(stage, _u32(0))
        _flip(phase)


def _pack2(lo, hi):
    packed = K.local_scalar("uint64")
    K.ptx.mov.b64(packed, lo, hi)
    return packed


def _packed_fma_inplace(values, base, scale_pair, bias_pair):
    """``fma.rn.ftz.f32x2`` on values[base:base+2] with packed scale and bias."""
    result = K.local_scalar("uint64")
    K.ptx.fma.rn.ftz.f32x2(result, _pack2(values[base], values[base + 1]), scale_pair, bias_pair)
    K.ptx.mov.b64(values[base], values[base + 1], result)


def _packed_mul_inplace(values, base, scale_pair):
    """``mul.rn.ftz.f32x2`` on values[base:base+2] with a packed scale."""
    result = K.local_scalar("uint64")
    K.ptx.mul.rn.ftz.f32x2(result, _pack2(values[base], values[base + 1]), scale_pair)
    K.ptx.mov.b64(values[base], values[base + 1], result)


def _exp2_inplace(values, index):
    K.ptx.ex2.approx.ftz.f32(values[index], values[index])


def _max_f32(a, b):
    out = K.local_scalar("float32")
    K.ptx.max.f32(out, a, b)
    return out


def _shfl_xor_f32(value, lane_xor):
    out = K.local_scalar("uint32")
    K.ptx.shfl_sync.bfly.b32(
        out, K.reinterpret("uint32", value), _u32(lane_xor), _u32(31), _u32(_FULL_MASK)
    )
    return K.reinterpret("float32", out)


def _tmem_load_x32(dst, base, taddr):
    K.ptx[_TMEM_LD_X32](*(dst[base + i] for i in range(32)), taddr, 64)


def _block_sum(values):
    """``softmax_block_sum`` over 64 values: 32 packed ``add.f32x2`` into one accumulator."""
    acc = K.local_scalar("uint64")
    K.ptx.mov.b64(acc, _f32(0.0), _f32(0.0))
    for j in range(32):
        K.ptx.add.f32x2(acc, acc, _pack2(values[2 * j], values[2 * j + 1]))
    acc_x = K.local_scalar("float32")
    acc_y = K.local_scalar("float32")
    K.ptx.mov.b64(acc_x, acc_y, acc)
    half = K.local_scalar("float32")
    K.ptx.add.ftz.f32(half, acc_x, acc_y)
    total = K.local_scalar("float32")
    K.ptx.add.ftz.f32(total, half, _shfl_xor_f32(half, 16))
    return total


def _row_max(values):
    """``row_max_x32_accum`` x2 + ``row_max_reduce``: alternating accumulators, 65 ``max.f32``."""
    acc0 = K.local_scalar("float32", init=_f32(_NEG_INF))
    acc1 = K.local_scalar("float32", init=_f32(_NEG_INF))
    for half in range(2):
        for j in range(16):
            pair = _max_f32(values[half * 32 + 2 * j], values[half * 32 + 2 * j + 1])
            if j % 2 == 0:
                K.assign(acc0, _max_f32(acc0, pair))
            else:
                K.assign(acc1, _max_f32(acc1, pair))
    return _max_f32(acc0, acc1)


def _build_kernel():
    @K.kernel(
        warps=NUM_WARPS,
        arch=CUDA_ARCH,
        min_blocks_per_sm=1,
        grid=lambda p: [p["mb"], p["num_q_heads"]],
        host_prelude=_host_prelude,
    )
    def cake_vsa_longseq_sm103(
        q: K.gptr[K.bf16],
        k: K.gptr[K.bf16],
        v: K.gptr[K.bf16],
        out: K.gptr[K.bf16],
        lse: K.gptr[K.f32],
        temperature_lse: K.gptr[K.f32],
        block_mask: K.gptr[K.u8],
        mb: K.i32,
        nb: K.i32,
        selected_blocks: K.i32,
        num_q_heads: K.i32,
        num_kv_heads: K.i32,
        softmax_scale_log2: K.f32,
        lse_temperature_scale: K.f32,
        return_softmax_lse: K.i32,
        return_temperature_lse: K.i32,
        *,
        host,
    ):
        q_map, k_map, v_map = host
        del q, k, v, num_kv_heads  # reached only through the tensor maps / host prelude
        # >>> kernel_flashinfer_blackwell_vsa_longseq_warp_specialized_sm100 body starts here
        q_tile, q_head = K.cta_id()
        warp = K.warp_id()
        lane = K.lane_id()

        arena = K.alloc_buffer((SMEM_TOTAL,), K.u8, scope="shared.dyn", align=1024)
        smem = K.local_scalar("uint32", init=K.cuda.cvta_generic_to_shared(arena.ptr_to([0])))

        def bar(offset):
            return smem + _u32(offset)

        # Mbarrier init (12 groups, 23 barriers) by one elected lane of warp 0.
        with K.If(warp == 0), K.Then():
            init_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
            with K.If(init_leader != _u32(0)), K.Then():
                for offset, count in _MBARRIER_INIT:
                    K.ptx.mbarrier.init.shared__cta.b64(bar(offset), _u32(count))
                K.ptx.fence.mbarrier_init.release.cluster()
        K.ptx.bar.warp.sync(_u32(_FULL_MASK))

        # TMEM alloc (512 columns) by warp 0; the address lands in the shared mailbox.
        with K.If(warp == 0), K.Then():
            K.ptx[_TMEM_ALLOC](bar(_SMEM_TMEM_MAILBOX), _u32(TMEM_COLS))
            K.ptx[_TMEM_RELINQUISH]()
        K.cuda.cta_sync()
        K.ptx["tcgen05.fence::after_thread_sync"]()
        taddr = K.local_scalar("uint32")
        K.ptx.ld.volatile.shared.b32(taddr, bar(_SMEM_TMEM_MAILBOX))

        q_local_base = q_tile * 128
        kv_len = nb * 128

        # Sibling role guards, as in the source; folding them into an if/else-if chain
        # measured 1.2-2.3% slower on the two small shapes with no register pressure to relieve.
        roles = K.specialize(chain_dispatch=False)
        other_regs = roles.register_scope("other", warps=range(12, 16), regs=_REG_OTHER)
        r_softmax = roles.role("softmax", warps=range(0, 8), regs=_REG_SOFTMAX)
        r_correction = roles.role("correction", warps=range(8, 12), regs=_REG_CORRECTION)
        r_mma = roles.role("mma", warps=[12], register_scope=other_regs)
        r_qk_load = roles.role("qk_load", warps=[13], register_scope=other_regs)
        r_v_load = roles.role("v_load", warps=[14], register_scope=other_regs)
        r_idle = roles.role("idle", warps=[15], register_scope=other_regs)

        # Register redistribution: warps 12..15 decrease first so the softmax
        # increase can be granted.
        with K.If(K.And(warp >= 12, warp <= 15)), K.Then():
            other_regs.emit()

        # ---- Role: softmax (warps 0..7, two four-warp instances) ----
        with r_softmax:
            q_valid = K.local_scalar("int32", init=_i32(128))
            with K.If(q_tile >= mb), K.Then():
                K.assign(q_valid, _i32(0))
            union_ready_phase = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_UNION_READY), union_ready_phase)
            _flip(union_ready_phase)
            instance = K.uniform(warp // 4)
            instance_row_offset = K.uniform(instance * 64)
            instance_token_offset = K.uniform(instance * 64)
            instance_tmem_offset = K.uniform(instance * 256)
            union_count = selected_blocks
            warp_in_instance = warp % 4
            tmem_row_origin = warp_in_instance * 32
            my_row = warp_in_instance * 16 + lane % 16
            col_half = lane // 16
            instance_valid = K.local_scalar("int32", init=q_valid - instance_token_offset)
            with K.If(instance_valid > _i32(64)), K.Then():
                K.assign(instance_valid, _i32(64))
            with K.If(instance_valid < _i32(0)), K.Then():
                K.assign(instance_valid, _i32(0))
            row_valid = K.local_scalar("int32", init=K.cast(my_row < instance_valid, "int32"))
            row_max = K.local_scalar("float32", init=_f32(_NEG_INF))
            row_sum = K.local_scalar("float32", init=_f32(0.0))
            temperature_sum = K.local_scalar("float32", init=_f32(0.0))
            phase_s_full_0 = K.local_scalar("int32", init=_i32(0))
            phase_s_full_1 = K.local_scalar("int32", init=_i32(0))
            phase_corr_done_0 = K.local_scalar("int32", init=_i32(0))
            phase_corr_done_1 = K.local_scalar("int32", init=_i32(0))
            score_addr = K.local_scalar(
                "uint32",
                init=taddr
                + K.cast(instance_tmem_offset, "uint32")
                + K.shift_left(K.cast(tmem_row_origin, "uint32"), _u32(16)),
            )
            p_addr = K.local_scalar("uint32", init=score_addr + _u32(_TMEM_P_OFFSET))
            scale_row_addr = bar(_SMEM_SCALE) + K.cast(
                instance_row_offset + my_row, "uint32"
            ) * _u32(4)
            scores = K.alloc_local((64,), "float32")
            packed_p = K.alloc_local((32,), "uint32")
            block_sum = K.local_scalar("float32", init=_f32(0.0))
            block_temperature_sum = K.local_scalar("float32", init=_f32(0.0))
            acc_scale = K.local_scalar("float32", init=_f32(1.0))
            temperature_acc_scale = K.local_scalar("float32", init=_f32(1.0))
            selected_max = K.local_scalar("float32", init=_f32(_NEG_INF))
            new_max_scaled = K.local_scalar("float32", init=_f32(0.0))
            safe_max = K.local_scalar("float32", init=_f32(0.0))

            def probabilities_from(frag, bias):
                """fma, exp2, block sum, bf16 pack and the two P stores of one branch."""
                scale_pair = _pack2(softmax_scale_log2, softmax_scale_log2)
                bias_pair = _pack2(bias, bias)
                for j in range(32):
                    _packed_fma_inplace(frag, 2 * j, scale_pair, bias_pair)
                for j in range(64):
                    _exp2_inplace(frag, j)
                K.assign(block_sum, _block_sum(frag))
                for j in range(32):
                    K.ptx.cvt.rn.bf16x2.f32(packed_p[j], frag[2 * j + 1], frag[2 * j])
                K.ptx[_TMEM_ST_X16](p_addr, 32, *(packed_p[i] for i in range(16)))
                K.ptx[_TMEM_ST_X16](p_addr + _u32(16), 32, *(packed_p[16 + i] for i in range(16)))

            union_index = K.local_scalar("int32", init=_i32(0))
            with K.While(union_index < union_count):
                n_block = _ld_shared_i32(
                    bar(_SMEM_UNION_BLOCKS)
                    + K.cast(instance * 192 + union_index, "uint32") * _u32(4)
                )
                with K.If(instance == 0):
                    with K.Then():
                        _mbar_wait(bar(_MBAR_S_FULL), phase_s_full_0)
                        _flip(phase_s_full_0)
                    with K.Else():
                        _mbar_wait(bar(_MBAR_S_FULL + 8), phase_s_full_1)
                        _flip(phase_s_full_1)
                valid_cols = K.local_scalar("int32", init=_i32(0))
                with K.If(row_valid != _i32(0)), K.Then():
                    K.assign(valid_cols, kv_len - n_block * 128)
                    with K.If(valid_cols > _i32(128)), K.Then():
                        K.assign(valid_cols, _i32(128))
                    with K.If(valid_cols < _i32(0)), K.Then():
                        K.assign(valid_cols, _i32(0))
                _tmem_load_x32(scores, 0, score_addr)
                _tmem_load_x32(scores, 32, score_addr + _u32(32))
                half_valid = K.local_scalar("int32", init=valid_cols - col_half * 64)
                with K.If(half_valid < _i32(0)), K.Then():
                    K.assign(half_valid, _i32(0))
                with K.If(half_valid > _i32(64)), K.Then():
                    K.assign(half_valid, _i32(64))
                # The source's per-column -inf fill for 0 < half_valid < 64 is unreachable
                # (valid_cols is a multiple of 128) and nvcc emits no code for it.
                tile_max = K.local_scalar("float32", init=_row_max(scores))
                with K.If(half_valid <= _i32(0)), K.Then():
                    K.assign(tile_max, _f32(_NEG_INF))
                K.assign(tile_max, _max_f32(tile_max, _shfl_xor_f32(tile_max, 16)))
                new_max = _max_f32(tile_max, row_max)
                K.assign(safe_max, K.if_then_else(new_max == _f32(_NEG_INF), _f32(0.0), new_max))
                K.ptx.mul.ftz.f32(new_max_scaled, safe_max, softmax_scale_log2)
                neg_new_max_scaled = K.local_scalar("float32")
                K.ptx.neg.ftz.f32(neg_new_max_scaled, new_max_scaled)
                acc_scale_log2 = K.local_scalar("float32")
                K.ptx.fma.rn.ftz.f32(
                    acc_scale_log2, row_max, softmax_scale_log2, neg_new_max_scaled
                )
                with K.If(acc_scale_log2 >= _f32(-8.0)):
                    with K.Then():
                        # The running max moves by less than 2^8: keep the old max, skip the rescale.
                        K.assign(selected_max, row_max)
                        K.assign(
                            safe_max, K.if_then_else(row_max == _f32(_NEG_INF), _f32(0.0), row_max)
                        )
                        K.assign(acc_scale, _f32(1.0))
                        K.assign(temperature_acc_scale, _f32(1.0))
                        K.ptx.mul.ftz.f32(new_max_scaled, safe_max, softmax_scale_log2)
                    with K.Else():
                        K.assign(selected_max, new_max)
                        exp_scale = K.local_scalar("float32")
                        K.ptx.ex2.approx.ftz.f32(exp_scale, acc_scale_log2)
                        K.assign(
                            acc_scale,
                            K.if_then_else(row_max > _f32(_NEG_INF), exp_scale, _f32(1.0)),
                        )
                        tau_log2 = K.local_scalar("float32")
                        K.ptx.mul.ftz.f32(tau_log2, acc_scale_log2, lse_temperature_scale)
                        exp_tau = K.local_scalar("float32")
                        K.ptx.ex2.approx.ftz.f32(exp_tau, tau_log2)
                        K.assign(
                            temperature_acc_scale,
                            K.if_then_else(row_max > _f32(_NEG_INF), exp_tau, _f32(1.0)),
                        )
                K.assign(row_max, selected_max)
                with K.If(col_half == 0), K.Then():
                    K.ptx.st.shared.b32(scale_row_addr, acc_scale)
                K.ptx.fence.proxy.async_.shared__cta()
                with K.If(instance == 0):
                    with K.Then():
                        _mbar_arrive(bar(_MBAR_CORR_SIG))
                    with K.Else():
                        _mbar_arrive(bar(_MBAR_CORR_SIG + 8))
                # The bias negates the (possibly re-selected) new_max_scaled after the branch.
                neg_bias = K.local_scalar("float32")
                K.ptx.neg.ftz.f32(neg_bias, new_max_scaled)
                score_bias = K.local_scalar("float32")
                K.assign(score_bias, K.if_then_else(valid_cols > _i32(0), neg_bias, _f32(_NEG_INF)))
                K.assign(block_temperature_sum, _f32(0.0))
                K.assign(block_sum, _f32(0.0))
                with K.If(return_temperature_lse != _i32(0)):
                    with K.Then():
                        tau_scale = K.local_scalar("float32")
                        K.ptx.mul.ftz.f32(tau_scale, softmax_scale_log2, lse_temperature_scale)
                        tau_bias = K.local_scalar("float32")
                        K.ptx.mul.ftz.f32(tau_bias, score_bias, lse_temperature_scale)
                        tau_scale_pair = _pack2(tau_scale, tau_scale)
                        tau_bias_pair = _pack2(tau_bias, tau_bias)
                        for j in range(32):
                            _packed_fma_inplace(scores, 2 * j, tau_scale_pair, tau_bias_pair)
                        for j in range(64):
                            _exp2_inplace(scores, j)
                        K.assign(block_temperature_sum, _block_sum(scores))
                        # S is re-read because the temperature sum consumed the first copy.
                        _tmem_load_x32(scores, 0, score_addr)
                        _tmem_load_x32(scores, 32, score_addr + _u32(32))
                        probabilities_from(scores, score_bias)
                    with K.Else():
                        probabilities_from(scores, score_bias)
                K.ptx.tcgen05.wait__st.sync.aligned()
                with K.If(instance == 0):
                    with K.Then():
                        _mbar_arrive(bar(_MBAR_P_FULL))
                        _mbar_wait(bar(_MBAR_CORR_DONE), phase_corr_done_0)
                        _flip(phase_corr_done_0)
                    with K.Else():
                        _mbar_arrive(bar(_MBAR_P_FULL + 8))
                        _mbar_wait(bar(_MBAR_CORR_DONE + 8), phase_corr_done_1)
                        _flip(phase_corr_done_1)
                K.ptx.fma.rn.ftz.f32(row_sum, row_sum, acc_scale, block_sum)
                with K.If(return_temperature_lse != _i32(0)), K.Then():
                    K.ptx.fma.rn.ftz.f32(
                        temperature_sum,
                        temperature_sum,
                        temperature_acc_scale,
                        block_temperature_sum,
                    )
                K.assign(union_index, union_index + _i32(1))
            with K.If(col_half == 0), K.Then():
                K.ptx.st.shared.b32(scale_row_addr + _u32(128 * 4), row_sum)
                K.ptx.st.shared.b32(scale_row_addr + _u32(256 * 4), row_max)
                K.ptx.st.shared.b32(scale_row_addr + _u32(384 * 4), temperature_sum)
            K.ptx.fence.proxy.async_.shared__cta()
            with K.If(instance == 0):
                with K.Then():
                    _mbar_arrive(bar(_MBAR_CORR_SIG))
                with K.Else():
                    _mbar_arrive(bar(_MBAR_CORR_SIG + 8))

        # ---- Role: correction and epilogue (warps 8..11) ----
        with r_correction:
            q_valid_c = K.local_scalar("int32", init=_i32(128))
            with K.If(q_tile >= mb), K.Then():
                K.assign(q_valid_c, _i32(0))
            union_ready_phase_c = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_UNION_READY), union_ready_phase_c)
            _flip(union_ready_phase_c)
            max_union_count_c = selected_blocks
            warp_in_role = warp - 8
            tmem_row_origin_c = warp_in_role * 32
            my_row_c = warp_in_role * 16 + lane % 16
            col_half_c = lane // 16
            row_addr = K.local_scalar(
                "uint32", init=K.shift_left(K.cast(tmem_row_origin_c, "uint32"), _u32(16))
            )
            scale_base_c = bar(_SMEM_SCALE) + K.cast(my_row_c, "uint32") * _u32(4)
            _mbar_arrive(bar(_MBAR_P_FULL))
            _mbar_arrive(bar(_MBAR_P_FULL + 8))
            phase_corr_sig_0 = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_CORR_SIG), phase_corr_sig_0)
            _flip(phase_corr_sig_0)
            _mbar_arrive(bar(_MBAR_CORR_DONE))
            phase_corr_sig_1 = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_CORR_SIG + 8), phase_corr_sig_1)
            _flip(phase_corr_sig_1)
            _mbar_arrive(bar(_MBAR_CORR_DONE + 8))
            o_frag = K.alloc_local((64,), "float32")

            def rescale_instance(instance, phase_corr_sig, union_index_1):
                count = selected_blocks
                with K.If(count > union_index_1), K.Then():
                    _mbar_wait(bar(_MBAR_CORR_SIG + 8 * instance), phase_corr_sig)
                    _flip(phase_corr_sig)
                    acc_scale_c = _ld_shared_f32(scale_base_c + _u32(64 * 4 * instance))
                    any_rescale = K.local_scalar("uint32")
                    K.ptx.vote_sync.any.pred(
                        any_rescale,
                        K.ptx.pred(K.cast(acc_scale_c < _f32(1.0), "bool")),
                        _u32(_FULL_MASK),
                    )
                    with K.If(any_rescale != _u32(0)), K.Then():
                        o_addr = taddr + _u32(_TMEM_OUTPUT[instance]) + row_addr
                        _tmem_load_x32(o_frag, 0, o_addr)
                        _tmem_load_x32(o_frag, 32, o_addr + _u32(32))
                        scale_pair = _pack2(acc_scale_c, acc_scale_c)
                        for j in range(32):
                            _packed_mul_inplace(o_frag, 2 * j, scale_pair)
                        K.ptx[_TMEM_ST_X64](o_addr, 64, *(o_frag[i] for i in range(64)))
                        K.ptx.tcgen05.wait__st.sync.aligned()
                    _mbar_arrive(bar(_MBAR_P_FULL + 8 * instance))
                    _mbar_arrive(bar(_MBAR_CORR_DONE + 8 * instance))

            union_index_1 = K.local_scalar("int32", init=_i32(1))
            with K.While(union_index_1 < max_union_count_c):
                rescale_instance(0, phase_corr_sig_0, union_index_1)
                rescale_instance(1, phase_corr_sig_1, union_index_1)
                K.assign(union_index_1, union_index_1 + _i32(1))
            phase_o_full_0 = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_O_FULL), phase_o_full_0)
            _flip(phase_o_full_0)
            phase_o_full_1 = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_O_FULL + 8), phase_o_full_1)
            _flip(phase_o_full_1)
            _mbar_wait(bar(_MBAR_CORR_SIG), phase_corr_sig_0)
            _flip(phase_corr_sig_0)
            _mbar_wait(bar(_MBAR_CORR_SIG + 8), phase_corr_sig_1)
            _flip(phase_corr_sig_1)
            K.ptx["tcgen05.fence::after_thread_sync"]()
            for instance in range(2):
                stats_addr = scale_base_c + _u32(64 * 4 * instance)
                final_sum = _ld_shared_f32(stats_addr + _u32(128 * 4))
                final_max = _ld_shared_f32(stats_addr + _u32(256 * 4))
                final_temperature_sum = _ld_shared_f32(stats_addr + _u32(384 * 4))
                rcp_sum = K.local_scalar("float32")
                K.ptx.rcp.approx.ftz.f32(rcp_sum, final_sum)
                sum_positive = K.local_scalar("int32", init=K.cast(final_sum > _f32(0.0), "int32"))
                inv_sum = K.local_scalar(
                    "float32", init=K.if_then_else(sum_positive != _i32(0), rcp_sum, _f32(0.0))
                )
                instance_valid_c = K.local_scalar("int32", init=q_valid_c - _i32(64 * instance))
                with K.If(instance_valid_c > _i32(64)), K.Then():
                    K.assign(instance_valid_c, _i32(64))
                with K.If(instance_valid_c < _i32(0)), K.Then():
                    K.assign(instance_valid_c, _i32(0))
                query = q_local_base + 64 * instance + my_row_c
                output_row = (query * num_q_heads + q_head) * 128
                o_addr = taddr + _u32(_TMEM_OUTPUT[instance]) + row_addr
                _tmem_load_x32(o_frag, 0, o_addr)
                _tmem_load_x32(o_frag, 32, o_addr + _u32(32))
                col_base = col_half_c * 64
                with K.If(my_row_c < instance_valid_c), K.Then():
                    inv_pair = _pack2(inv_sum, inv_sum)
                    for chunk in range(8):
                        base = 8 * chunk
                        for j in range(4):
                            _packed_mul_inplace(o_frag, base + 2 * j, inv_pair)
                        words = K.alloc_local((4,), "uint32")
                        for j in range(4):
                            K.ptx.cvt.rn.bf16x2.f32(
                                words[j], o_frag[base + 2 * j + 1], o_frag[base + 2 * j]
                            )
                        K.ptx.st.global_.v4.b32(
                            out.ptr_to([output_row + col_base + base]),
                            words[0],
                            words[1],
                            words[2],
                            words[3],
                        )
                    with K.If(col_half_c == 0), K.Then():
                        stat_idx = query * num_q_heads + q_head
                        with K.If(return_softmax_lse != _i32(0)), K.Then():
                            log2_sum = K.local_scalar("float32")
                            K.ptx.lg2.approx.ftz.f32(log2_sum, final_sum)
                            max_scaled = K.local_scalar("float32")
                            K.ptx.mul.ftz.f32(max_scaled, final_max, softmax_scale_log2)
                            log_sum = K.local_scalar("float32")
                            K.ptx.mul.ftz.f32(log_sum, log2_sum, _f32(_LN2_F32))
                            lse_value = K.local_scalar("float32")
                            K.ptx.fma.rn.ftz.f32(lse_value, max_scaled, _f32(_LN2_F32), log_sum)
                            K.ptx.st.global_.b32(
                                lse.ptr_to([stat_idx]),
                                K.if_then_else(sum_positive != _i32(0), lse_value, _f32(_NEG_INF)),
                            )
                        with K.If(return_temperature_lse != _i32(0)), K.Then():
                            log2_tsum = K.local_scalar("float32")
                            K.ptx.lg2.approx.ftz.f32(log2_tsum, final_temperature_sum)
                            t_value = K.local_scalar("float32", init=_f32(_NEG_INF))
                            with K.If(final_temperature_sum > _f32(0.0)), K.Then():
                                max_scaled_t = K.local_scalar("float32")
                                K.ptx.mul.ftz.f32(max_scaled_t, final_max, softmax_scale_log2)
                                max_ln = K.local_scalar("float32")
                                K.ptx.mul.ftz.f32(max_ln, max_scaled_t, _f32(_LN2_F32))
                                log_tsum = K.local_scalar("float32")
                                K.ptx.mul.ftz.f32(log_tsum, log2_tsum, _f32(_LN2_F32))
                                K.ptx.fma.rn.ftz.f32(
                                    t_value, lse_temperature_scale, max_ln, log_tsum
                                )
                            K.ptx.st.global_.b32(temperature_lse.ptr_to([stat_idx]), t_value)
            K.ptx.tcgen05.wait__ld.sync.aligned()
            K.ptx["tcgen05.fence::before_thread_sync"]()
            _mbar_arrive(bar(_MBAR_TMEM_DEALLOC))

        # ---- Role: MMA issuer (warp 12) ----
        with r_mma:
            union_ready_phase_m = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_UNION_READY), union_ready_phase_m)
            _flip(union_ready_phase_m)
            max_union_count_m = selected_blocks
            k_stage = K.local_scalar("uint32", init=_u32(0))
            k_phase = K.local_scalar("int32", init=_i32(0))
            v_stage = K.local_scalar("uint32", init=_u32(0))
            v_phase = K.local_scalar("int32", init=_i32(0))
            q_full_phase = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_Q_FULL), q_full_phase)
            _flip(q_full_phase)
            first_pv = [K.local_scalar("int32", init=_i32(1)) for _ in range(2)]
            phase_p_full = [K.local_scalar("int32", init=_i32(0)) for _ in range(2)]
            q_addr = (bar(_SMEM_Q0), bar(_SMEM_Q1))
            desc_hi = _u32(_DESC_HI)

            def qk_chain(instance, k_stage_now):
                a_lo = K.local_scalar(
                    "uint32", init=K.uniform((q_addr[instance] >> 4) & _u32(0x3FFF))
                )
                b_lo = K.local_scalar(
                    "uint32",
                    init=K.uniform(((bar(_SMEM_K) >> 4) & _u32(0x3FFF)) + k_stage_now * _u32(2048)),
                )
                leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                d_tmem = taddr + _u32(_TMEM_SCORES[instance])
                for k16 in range(8):
                    K.ptx[_MMA_F16](
                        d_tmem,
                        _pack2(a_lo, desc_hi),
                        _pack2(b_lo, desc_hi),
                        _u32(_QK_IDESC),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        K.ptx.pred(_u32(0 if k16 == 0 else 1)),
                        pred=leader,
                    )
                    if k16 < 7:
                        K.assign(a_lo, a_lo + _u32(_QK_A_STEPS[k16]))
                        K.assign(b_lo, b_lo + _u32(_QK_B_STEPS[k16]))
                commit_s_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                K.ptx[_TCGEN05_COMMIT](bar(_MBAR_S_FULL + 8 * instance), pred=commit_s_leader)
                commit_k_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                K.ptx[_TCGEN05_COMMIT](
                    bar(_MBAR_K_EMPTY) + k_stage_now * _u32(8), pred=commit_k_leader
                )

            def pv_chain(instance, v_stage_now):
                b_lo = K.local_scalar(
                    "uint32",
                    init=K.uniform(
                        (((bar(_SMEM_V) >> 4) & _u32(0x3FFF)) | _u32(_V_LBO_BIT))
                        + v_stage_now * _u32(2048)
                    ),
                )
                leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                d_tmem = taddr + _u32(_TMEM_OUTPUT[instance])
                a_tmem = K.local_scalar(
                    "uint32", init=taddr + _u32(_TMEM_SCORES[instance] + _TMEM_P_OFFSET)
                )
                enable_first = K.local_scalar(
                    "uint32", init=K.if_then_else(first_pv[instance] != _i32(0), _u32(0), _u32(1))
                )
                for k16 in range(8):
                    K.ptx[_MMA_F16](
                        d_tmem,
                        a_tmem,
                        _pack2(b_lo, desc_hi),
                        _u32(_PV_IDESC),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        _u32(0),
                        K.ptx.pred(enable_first if k16 == 0 else _u32(1)),
                        pred=leader,
                    )
                    if k16 < 7:
                        K.assign(a_tmem, a_tmem + _u32(_PV_A_STEP))
                        K.assign(b_lo, b_lo + _u32(_PV_B_STEP))
                K.assign(first_pv[instance], _i32(0))

            # Prime QK for both 64-row instances before the PV pipeline loop.
            for instance in range(2):
                with K.If(selected_blocks > _i32(0)), K.Then():
                    k_stage_now = K.local_scalar("uint32", init=k_stage)
                    k_phase_now = K.local_scalar("int32", init=k_phase)
                    _advance_ring(k_stage, k_phase, 3)
                    _mbar_wait(bar(_MBAR_K_FULL) + k_stage_now * _u32(8), k_phase_now)
                    qk_chain(instance, k_stage_now)

            union_index_m = K.local_scalar("int32", init=_i32(0))
            with K.While(union_index_m < max_union_count_m):
                for instance in range(2):
                    with K.If(selected_blocks > union_index_m), K.Then():
                        v_stage_now = K.local_scalar("uint32", init=v_stage)
                        v_phase_now = K.local_scalar("int32", init=v_phase)
                        _advance_ring(v_stage, v_phase, 2)
                        _mbar_wait(bar(_MBAR_V_FULL) + v_stage_now * _u32(8), v_phase_now)
                        _mbar_wait(bar(_MBAR_P_FULL + 8 * instance), phase_p_full[instance])
                        _flip(phase_p_full[instance])
                        pv_chain(instance, v_stage_now)
                        with K.If(union_index_m + _i32(1) == selected_blocks), K.Then():
                            commit_o_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                            K.ptx[_TCGEN05_COMMIT](
                                bar(_MBAR_O_FULL + 8 * instance), pred=commit_o_leader
                            )
                        commit_v_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                        K.ptx[_TCGEN05_COMMIT](
                            bar(_MBAR_V_EMPTY) + v_stage_now * _u32(8), pred=commit_v_leader
                        )
                    next_union_index = union_index_m + _i32(1)
                    with K.If(next_union_index < selected_blocks), K.Then():
                        k_stage_now = K.local_scalar("uint32", init=k_stage)
                        k_phase_now = K.local_scalar("int32", init=k_phase)
                        _advance_ring(k_stage, k_phase, 3)
                        _mbar_wait(bar(_MBAR_K_FULL) + k_stage_now * _u32(8), k_phase_now)
                        qk_chain(instance, k_stage_now)
                K.assign(union_index_m, union_index_m + _i32(1))
            dealloc_phase = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_TMEM_DEALLOC), dealloc_phase)
            _flip(dealloc_phase)
            taddr_dealloc = K.local_scalar("uint32")
            K.ptx.ld.volatile.shared.b32(taddr_dealloc, bar(_SMEM_TMEM_MAILBOX))
            K.ptx[_TMEM_DEALLOC](taddr_dealloc, _u32(TMEM_COLS))

        # ---- Role: Q/mask/K producer (warp 13) ----
        with r_qk_load:
            kv_head = q_head
            query_base = q_local_base
            load_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
            with K.If(load_leader != _u32(0)), K.Then():
                _mbar_expect_tx(bar(_MBAR_Q_FULL), 32768)
                K.ptx[_TMA_G2S_4D](
                    bar(_SMEM_Q0),
                    K.address_of(q_map),
                    _i32(0),
                    query_base,
                    q_head,
                    _i32(0),
                    bar(_MBAR_Q_FULL),
                )
                K.ptx[_TMA_G2S_4D](
                    bar(_SMEM_Q1),
                    K.address_of(q_map),
                    _i32(0),
                    query_base + _i32(64),
                    q_head,
                    _i32(0),
                    bar(_MBAR_Q_FULL),
                )
            q_block = query_base // 128
            mask_base = (q_head * mb + q_block) * nb
            selected_count = K.local_scalar("int32", init=_i32(0))
            block_base = K.local_scalar("int32", init=_i32(0))
            with K.While(block_base < nb):
                n_block = block_base + lane
                selected = K.local_scalar("int32", init=_i32(0))
                with K.If(n_block < nb), K.Then():
                    mask_byte = K.local_scalar("uint16")
                    K.ptx.ld.global_.nc.b8(mask_byte, block_mask.ptr_to([mask_base + n_block]))
                    K.assign(selected, K.cast(mask_byte != K.uint16(0), "int32"))
                vote = K.local_scalar("uint32")
                K.ptx.vote_sync.ballot.b32(
                    vote, K.ptx.pred(K.cast(selected != _i32(0), "bool")), _u32(_FULL_MASK)
                )
                ballot_count = K.local_scalar("uint32")
                K.ptx.popc.b32(ballot_count, vote)
                lower_lanes = K.shift_left(_u32(1), K.cast(lane, "uint32")) - _u32(1)
                lower_count = K.local_scalar("uint32")
                K.ptx.popc.b32(lower_count, vote & lower_lanes)
                selected_slot = selected_count + K.cast(lower_count, "int32")
                with K.If(selected != _i32(0)), K.Then():
                    slot_addr = bar(_SMEM_UNION_BLOCKS) + K.cast(selected_slot, "uint32") * _u32(4)
                    K.ptx.st.shared.b32(slot_addr, n_block)
                    K.ptx.st.shared.b32(slot_addr + _u32(MAX_SELECTED_BLOCKS * 4), n_block)
                K.assign(selected_count, selected_count + K.cast(ballot_count, "int32"))
                K.assign(block_base, block_base + _i32(32))
            with K.If(selected_count == _i32(0)), K.Then():
                with K.If(lane == 0), K.Then():
                    K.ptx.st.shared.b32(bar(_SMEM_UNION_BLOCKS), _i32(0))
                    K.ptx.st.shared.b32(bar(_SMEM_UNION_BLOCKS + MAX_SELECTED_BLOCKS * 4), _i32(0))
                K.assign(selected_count, _i32(1))
            with K.If(lane < 2), K.Then():
                K.ptx.st.shared.b32(
                    bar(_SMEM_UNION_COUNT) + K.cast(lane, "uint32") * _u32(4), selected_count
                )
            K.ptx.barrier.sync(8, 32)
            K.ptx.fence.proxy.async_.shared__cta()
            _mbar_arrive(bar(_MBAR_UNION_READY))

            k_load_stage = K.local_scalar("uint32", init=_u32(0))
            phase_k_empty = K.local_scalar("int32", init=_i32(1))
            union_index_k = K.local_scalar("int32", init=_i32(0))
            with K.While(union_index_k < selected_blocks):
                for instance in range(2):
                    n_block_k = _ld_shared_i32(
                        bar(_SMEM_UNION_BLOCKS + MAX_SELECTED_BLOCKS * 4 * instance)
                        + K.cast(union_index_k, "uint32") * _u32(4)
                    )
                    token_base = n_block_k * 128
                    _mbar_wait(bar(_MBAR_K_EMPTY) + k_load_stage * _u32(8), phase_k_empty)
                    push_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                    with K.If(push_leader != _u32(0)), K.Then():
                        full_bar = bar(_MBAR_K_FULL) + k_load_stage * _u32(8)
                        _mbar_expect_tx(full_bar, 32768)
                        stage_addr = bar(_SMEM_K) + k_load_stage * _u32(_SMEM_STAGE_BYTES)
                        token1 = token_base + _i32(64)
                        for dim_half in range(2):
                            for token_half, token in ((0, token_base), (1, token1)):
                                K.ptx[_TMA_G2S_4D](
                                    stage_addr + _u32(8192 * token_half + 16384 * dim_half),
                                    K.address_of(k_map),
                                    _i32(0),
                                    token,
                                    _i32(dim_half),
                                    kv_head,
                                    full_bar,
                                )
                    _advance_ring(k_load_stage, phase_k_empty, 3)
                K.assign(union_index_k, union_index_k + _i32(1))

        # ---- Role: V producer (warp 14) ----
        with r_v_load:
            union_ready_phase_v = K.local_scalar("int32", init=_i32(0))
            _mbar_wait(bar(_MBAR_UNION_READY), union_ready_phase_v)
            _flip(union_ready_phase_v)
            kv_head_v = q_head
            v_load_stage = K.local_scalar("uint32", init=_u32(0))
            phase_v_empty = K.local_scalar("int32", init=_i32(1))
            union_index_v = K.local_scalar("int32", init=_i32(0))
            with K.While(union_index_v < selected_blocks):
                for instance in range(2):
                    n_block_v = _ld_shared_i32(
                        bar(_SMEM_UNION_BLOCKS + MAX_SELECTED_BLOCKS * 4 * instance)
                        + K.cast(union_index_v, "uint32") * _u32(4)
                    )
                    token_base_v = n_block_v * 128
                    _mbar_wait(bar(_MBAR_V_EMPTY) + v_load_stage * _u32(8), phase_v_empty)
                    push_leader_v = K.local_scalar("uint32", init=K.cuda.elect_sync())
                    with K.If(push_leader_v != _u32(0)), K.Then():
                        full_bar_v = bar(_MBAR_V_FULL) + v_load_stage * _u32(8)
                        _mbar_expect_tx(full_bar_v, 32768)
                        stage_addr_v = bar(_SMEM_V) + v_load_stage * _u32(_SMEM_STAGE_BYTES)
                        token1_v = token_base_v + _i32(64)
                        for dim_half in range(2):
                            for token_half, token in ((0, token_base_v), (1, token1_v)):
                                K.ptx[_TMA_G2S_4D](
                                    stage_addr_v + _u32(8192 * token_half + 16384 * dim_half),
                                    K.address_of(v_map),
                                    _i32(0),
                                    token,
                                    _i32(dim_half),
                                    kv_head_v,
                                    full_bar_v,
                                )
                    _advance_ring(v_load_stage, phase_v_empty, 2)
                K.assign(union_index_v, union_index_v + _i32(1))

        # ---- Role: idle (warp 15) ----
        with r_idle:
            pass

    return cake_vsa_longseq_sm103


@lru_cache(maxsize=1)
def _kernel():
    return _build_kernel()


def get_kernel(**config: Any):
    """Return the TIRx PrimFunc; the kernel is shape-generic at runtime."""
    if config:
        cfg = _without_label(config)
        _validate_shape(cfg["M"], cfg["N"], cfg["num_heads"], cfg["selected"])
    return _kernel().func


@lru_cache(maxsize=1)
def _compiled_kernel():
    from tirx_kernels.runner import compile_kernel

    # nvcc mode keeps the TIRx build on the CUDA toolkit on PATH (13.3, PTX ISA
    # 9.3) -- the same toolchain the upstream ``nvcc -cubin`` build uses -- so
    # both binaries share one PTX ISA for SASS and profiler comparisons.
    # Match the native ptxas register-usage level. The narrow 48-register
    # producer roles spill descriptor and uniform values at TVM level 10,
    # default even though the source build has no local-memory traffic.
    previous = os.environ.get("TVM_CUDA_PTXAS_REG_LEVEL")
    os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = _PTXAS_REGISTER_USAGE_LEVEL
    try:
        return compile_kernel(get_kernel(), cuda_compile_mode="nvcc")
    finally:
        if previous is None:
            os.environ.pop("TVM_CUDA_PTXAS_REG_LEVEL", None)
        else:
            os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = previous


def _tirx_args(data: dict[str, Any], slot: str = "tirx") -> tuple[Any, ...]:
    buffers = data[slot]
    return (
        data["q"].view(-1),
        data["k"].view(-1),
        data["v"].view(-1),
        buffers["out"].view(-1),
        buffers["lse"].view(-1),
        buffers["tlse"].view(-1),
        data["block_mask_u8"].view(-1),
        data["mb"],
        data["nb"],
        data["selected_blocks"],
        data["num_heads"],
        data["num_heads"],
        data["softmax_scale_log2"],
        data["lse_temperature_scale"],
        int(data["return_lse"]),
        int(data["return_temperature_lse"]),
    )


def _tirx_launch(executable, data: dict[str, Any]):
    arguments = _tirx_args(data)

    def launch():
        executable(*arguments)

    launch._keep_alive = arguments
    return launch


# ---------------------------------------------------------------------------
# Upstream source kernel (the reference for correctness and benchmarks)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _source_module():
    """Build the upstream cubin + tvm-ffi launcher exactly as FlashInfer does."""
    from flashinfer import cake_vsa

    return cake_vsa._load_module(PROFILE, CUDA_ARCH)


def _source_launch(data: dict[str, Any]):
    """Replicate ``flashinfer.cake_vsa._run_standard`` for ``longseq``."""
    import tvm_ffi

    module = _source_module()
    buffers = data["source"]
    arguments = (
        data["q"],
        data["k"],
        data["v"],
        buffers["out"],
        buffers["lse"],
        buffers["tlse"],
        data["block_mask_u8"],
        data["mb"],
        data["nb"],
        data["selected_blocks"],
        data["num_heads"],
        data["num_heads"],
        data["softmax_scale_log2"],
        data["lse_temperature_scale"],
        int(data["return_lse"]),
        int(data["return_temperature_lse"]),
        data["mb"],
        data["num_heads"],
        1,
    )

    def launch():
        with tvm_ffi.use_torch_stream():
            module.run(*arguments)

    launch._keep_alive = arguments
    return launch


def _run_public_api(data: dict[str, Any]):
    """Run the upstream public API (``plan_cake_vsa`` + ``run_cake_vsa``)."""
    import torch
    from flashinfer import cake_vsa

    plan = cake_vsa.plan_cake_vsa(
        None,
        None,
        data["block_mask"],
        None,
        None,
        None,
        M=data["M"],
        N=data["N"],
        R=BLOCK,
        C=BLOCK,
        num_qo_heads=data["num_heads"],
        num_kv_heads=data["num_heads"],
        head_dim=HEAD_DIM,
        q_data_type=torch.bfloat16,
        sm_scale=data["sm_scale"],
        device=data["q"].device,
    )
    if data["return_lse"]:
        raise AssertionError("the public longseq route rejects return_lse")
    if (
        plan["N"] < 16384
        or plan["num_qo_heads"] != 8
        or not plan["uniform_selected_blocks"]
        or plan["max_selected_blocks"] != data["selected_blocks"]
        or plan["max_selected_blocks"] > MAX_SELECTED_BLOCKS
    ):
        raise AssertionError("public API plan does not satisfy the longseq route")
    return cake_vsa.run_cake_vsa(
        plan, data["q"], data["k"], data["v"], out=None, lse=None, return_lse=False, backend="cake"
    )


# ---------------------------------------------------------------------------
# fp64 sparse oracle and validation
# ---------------------------------------------------------------------------
def _bf16_order_key(values):
    """Map bf16 bit patterns to monotonically ordered integers."""
    import torch

    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(bits < 0x8000, bits + 0x8000, 0x8000 - (bits & 0x7FFF))


def _assert_bitwise(name: str, ours, theirs) -> None:
    if ours.dtype == theirs.dtype and ours.dtype.is_floating_point:
        view_dtype = (
            __import__("torch").int16 if ours.dtype.itemsize == 2 else __import__("torch").int32
        )
        ours_bits, theirs_bits = (
            ours.contiguous().view(view_dtype),
            theirs.contiguous().view(view_dtype),
        )
    else:
        ours_bits, theirs_bits = ours, theirs
    mismatch = ours_bits != theirs_bits
    count = int(mismatch.sum())
    if count:
        index = int(mismatch.reshape(-1).nonzero()[0])
        raise AssertionError(
            f"{name}: {count} of {mismatch.numel()} elements differ bitwise "
            f"(first flat index {index}: tirx={ours.reshape(-1)[index].item()!r}, "
            f"source={theirs.reshape(-1)[index].item()!r})"
        )


def _assert_guards(name: str, buffers: dict[str, Any]) -> None:
    expected = {"out": _OUT_GUARD, "lse": _STATS_GUARD, "tlse": _STATS_GUARD}
    for key, storage in buffers["guards"].items():
        guard = expected[key]
        if not bool((storage[:_GUARD_ELEMS] == guard).all()):
            raise AssertionError(f"{name}: {key} prefix guard was overwritten")
        if not bool((storage[-_GUARD_ELEMS:] == guard).all()):
            raise AssertionError(f"{name}: {key} suffix guard was overwritten")


class _OracleStats:
    def __init__(self, cfg: dict[str, Any], *, return_lse: bool, return_tlse: bool):
        self.cfg = cfg
        self.return_lse = return_lse
        self.return_tlse = return_tlse
        self.out_max_abs_err = 0.0
        self.out_max_rel_excess = 0.0
        self.out_max_err_small_ref = 0.0
        self.out_max_rel_err_large_ref = 0.0
        self.ulp_max = 0
        self.ulp_total = 0
        self.ulp_le1 = 0
        self.ulp_le2 = 0
        self.lse_max_abs_err = 0.0
        self.lse_max_rel_excess = 0.0
        self.tlse_max_abs_err = 0.0
        self.tlse_max_rel_excess = 0.0

    def update(self, out_bf16, lse, tlse, ref_out, ref_lse, ref_tlse) -> None:
        cfg = self.cfg
        out = out_bf16.double()
        err = (out - ref_out).abs()
        bound = cfg["oracle_atol"] + cfg["oracle_rtol"] * ref_out.abs()
        self.out_max_abs_err = max(self.out_max_abs_err, float(err.max()))
        self.out_max_rel_excess = max(self.out_max_rel_excess, float((err / bound).max()))
        small = ref_out.abs() < _ORACLE_ULP_MIN_REF
        large = ref_out.abs() >= 0.5
        if bool(small.any()):
            self.out_max_err_small_ref = max(self.out_max_err_small_ref, float(err[small].max()))
        if bool(large.any()):
            self.out_max_rel_err_large_ref = max(
                self.out_max_rel_err_large_ref, float((err[large] / ref_out[large].abs()).max())
            )
        sizeable = ~small
        if bool(sizeable.any()):
            ulps = (_bf16_order_key(ref_out.to(out_bf16.dtype)) - _bf16_order_key(out_bf16)).abs()[
                sizeable
            ]
            self.ulp_max = max(self.ulp_max, int(ulps.max()))
            self.ulp_total += ulps.numel()
            self.ulp_le1 += int((ulps <= 1).sum())
            self.ulp_le2 += int((ulps <= 2).sum())
        for flag, values, ref, attr in (
            (self.return_lse, lse, ref_lse, "lse"),
            (self.return_tlse, tlse, ref_tlse, "tlse"),
        ):
            if not flag:
                continue
            stat_err = (values.double() - ref).abs()
            stat_bound = cfg["oracle_lse_atol"] + cfg["oracle_lse_rtol"] * ref.abs()
            setattr(
                self,
                f"{attr}_max_abs_err",
                max(getattr(self, f"{attr}_max_abs_err"), float(stat_err.max())),
            )
            setattr(
                self,
                f"{attr}_max_rel_excess",
                max(getattr(self, f"{attr}_max_rel_excess"), float((stat_err / stat_bound).max())),
            )

    def check(self, name: str, stats: dict[str, float]) -> None:
        cfg = self.cfg
        stats[f"{name}_out_max_abs_err"] = self.out_max_abs_err
        stats[f"{name}_out_max_rel_excess"] = self.out_max_rel_excess
        stats[f"{name}_out_max_err_small_ref"] = self.out_max_err_small_ref
        stats[f"{name}_out_max_rel_err_large_ref"] = self.out_max_rel_err_large_ref
        if self.out_max_rel_excess > 1.0:
            raise AssertionError(
                f"{name}: out mismatch vs fp64 oracle (max abs "
                f"{self.out_max_abs_err:.3e}, max err/bound "
                f"{self.out_max_rel_excess:.3f})"
            )
        if self.ulp_total:
            le1 = self.ulp_le1 / self.ulp_total
            le2 = self.ulp_le2 / self.ulp_total
            stats[f"{name}_out_ulp_max"] = float(self.ulp_max)
            stats[f"{name}_out_ulp_le1_frac"] = le1
            stats[f"{name}_out_ulp_le2_frac"] = le2
            if (
                self.ulp_max > cfg["ulp_max"]
                or le1 < cfg["ulp_le1_frac"]
                or le2 < cfg["ulp_le2_frac"]
            ):
                raise AssertionError(
                    f"{name}: bf16 ULP distribution vs fp64 oracle too wide "
                    f"(max {self.ulp_max}, <=1 {le1:.6f}, <=2 {le2:.6f})"
                )
        for flag, attr in ((self.return_lse, "lse"), (self.return_tlse, "tlse")):
            if not flag:
                continue
            max_abs = getattr(self, f"{attr}_max_abs_err")
            excess = getattr(self, f"{attr}_max_rel_excess")
            stats[f"{name}_{attr}_max_abs_err"] = max_abs
            stats[f"{name}_{attr}_max_rel_excess"] = excess
            if excess > 1.0:
                raise AssertionError(
                    f"{name}: {attr} mismatch vs fp64 oracle "
                    f"(max abs {max_abs:.3e}, max err/bound {excess:.3f})"
                )


def _oracle_chunk(data: dict[str, Any], t0: int, t1: int):
    """Block-gather fp64 reference for query tiles [t0, t1)."""
    import torch

    T = t1 - t0
    H, S = data["num_heads"], data["selected_blocks"]
    nb = data["nb"]
    idx = data["selected_indices"][t0:t1].to(device=data["q"].device, dtype=torch.int64)
    heads = torch.arange(H, device=data["q"].device)[None, :, None]
    q = data["q"][t0 * BLOCK : t1 * BLOCK].double().view(T, BLOCK, H, HEAD_DIM)
    q = q.permute(0, 2, 1, 3)
    k_by_head = data["k"].view(nb, BLOCK, H, HEAD_DIM).permute(2, 0, 1, 3)
    v_by_head = data["v"].view(nb, BLOCK, H, HEAD_DIM).permute(2, 0, 1, 3)
    k_g = k_by_head[heads, idx].reshape(T, H, S * BLOCK, HEAD_DIM).double()
    v_g = v_by_head[heads, idx].reshape(T, H, S * BLOCK, HEAD_DIM).double()
    scores = torch.matmul(q, k_g.transpose(-1, -2)) * data["sm_scale"]
    ref_lse = torch.logsumexp(scores, dim=-1)
    ref_tlse = torch.logsumexp(scores * data["lse_temperature_scale"], dim=-1)
    probs = torch.softmax(scores, dim=-1)
    ref_out = torch.matmul(probs, v_g)
    return ref_out, ref_lse, ref_tlse


def _validate_outputs(data: dict[str, Any], *, with_oracle: bool) -> dict[str, float]:
    import torch

    tirx, source = data["tirx"], data["source"]
    return_lse = data["return_lse"]
    return_tlse = data["return_temperature_lse"]
    for name, buffers in (("tirx", tirx), ("source", source)):
        _assert_guards(name, buffers)
        if not torch.isfinite(buffers["out"].float()).all():
            raise AssertionError(f"{name}: non-finite values in out")
        if return_lse and not torch.isfinite(buffers["lse"]).all():
            raise AssertionError(f"{name}: non-finite values in lse")
        if return_tlse and not torch.isfinite(buffers["tlse"]).all():
            raise AssertionError(f"{name}: non-finite values in temperature lse")
        if not return_lse and not bool((buffers["lse"] == _LSE_SENTINEL).all()):
            raise AssertionError(f"{name}: lse sentinel overwritten with return_lse=0")

    _assert_bitwise("out", tirx["out"], source["out"])
    _assert_bitwise("lse", tirx["lse"], source["lse"])
    if return_tlse:
        _assert_bitwise("temperature_lse", tirx["tlse"], source["tlse"])

    stats: dict[str, float] = {}
    if not with_oracle:
        return stats
    trackers = {
        name: _OracleStats(data["config"], return_lse=return_lse, return_tlse=return_tlse)
        for name in ("tirx", "source")
    }
    mb, H = data["mb"], data["num_heads"]
    for t0 in range(0, mb, _ORACLE_TILES_PER_CHUNK):
        t1 = min(mb, t0 + _ORACLE_TILES_PER_CHUNK)
        ref_out, ref_lse, ref_tlse = _oracle_chunk(data, t0, t1)
        T = t1 - t0
        for name, buffers in (("tirx", tirx), ("source", source)):
            out = buffers["out"][t0 * BLOCK : t1 * BLOCK].view(T, BLOCK, H, HEAD_DIM)
            out = out.permute(0, 2, 1, 3)
            lse = (
                buffers["lse"][t0 * BLOCK : t1 * BLOCK].view(T, BLOCK, H).permute(0, 2, 1)
                if return_lse
                else None
            )
            tlse = (
                buffers["tlse"][t0 * BLOCK : t1 * BLOCK].view(T, BLOCK, H).permute(0, 2, 1)
                if return_tlse
                else None
            )
            trackers[name].update(out, lse, tlse, ref_out, ref_lse, ref_tlse)
    for name, tracker in trackers.items():
        tracker.check(name, stats)
    return stats


def _skip_unless_supported() -> None:
    from unittest import SkipTest

    import torch

    if not torch.cuda.is_available():
        raise SkipTest("CUDA device required")
    if torch.cuda.get_device_capability() != (10, 3):
        raise SkipTest("cake_vsa longseq sm_103a requires compute capability 10.3")


def _assert_inputs_unchanged(data: dict[str, Any], snapshots: dict[str, Any]) -> None:
    import torch

    for name, before in snapshots.items():
        if not torch.equal(data[name], before):
            raise AssertionError(f"input {name} was modified by a kernel launch")


def run_test(**config: Any) -> dict[str, float]:
    """Run one config against source, fp64 oracle, guards, and determinism checks."""
    import torch

    _skip_unless_supported()
    data = prepare_data(**config)
    snapshots = {name: data[name].clone() for name in ("q", "k", "v", "block_mask")}
    tirx_launch = _tirx_launch(_compiled_kernel(), data)
    tirx_launch()
    torch.cuda.synchronize()
    first = {key: data["tirx"][key].clone() for key in ("out", "lse", "tlse")}

    source_launch = _source_launch(data)
    source_launch()
    torch.cuda.synchronize()
    stats = _validate_outputs(data, with_oracle=True)
    _assert_inputs_unchanged(data, snapshots)

    tirx_launch()
    torch.cuda.synchronize()
    for key in ("out", "lse", "tlse"):
        _assert_bitwise(f"{key} (repeat launch)", data["tirx"][key], first[key])
    _assert_guards("tirx repeat", data["tirx"])
    _assert_inputs_unchanged(data, snapshots)

    if data["config"]["pattern"] == "upstream":
        api_out = _run_public_api(data)
        _assert_bitwise("out (public API)", data["tirx"]["out"], api_out)
        _assert_inputs_unchanged(data, snapshots)
    return stats


# ---------------------------------------------------------------------------
# Benchmark entry points
# ---------------------------------------------------------------------------
def prepare_bench(**config: Any):
    from tirx_kernels.runner import prepared_gpu_benchmark

    kernel_config = _without_label(config)
    _validate_shape(
        kernel_config["M"],
        kernel_config["N"],
        kernel_config["num_heads"],
        kernel_config["selected"],
    )
    state = {"config": kernel_config, "executable": _compiled_kernel()}
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
                source_launch = _source_launch(data)
                gpu_state["source_launch"] = source_launch
                source_launch()
                torch.cuda.synchronize()
            _validate_outputs(data, with_oracle=False)
        gpu_state["validated"] = True

    source_launch = gpu_state["source_launch"]
    references = {"flashinfer": lambda: source_launch} if source_launch is not None else None
    return bench(
        {"tirx": tirx_launch},
        references=references,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **config: Any):
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
