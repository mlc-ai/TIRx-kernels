# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ c5365737), Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Cake VSA ``ultrasparse_bsr`` block-sparse attention forward for SM100a.

Upstream sources (FlashInfer @ c5365737570a2a156d7cae0c4070fa3770ecc670):

- ``csrc/cake_vsa/cake_vsa_ultrasparse_bsr_sm_100a.cu`` -- the device kernel
  ``kernel_flashinfer_blackwell_vsa_ultrasparse_bsr_sm100`` (Cake source export,
  last changed by upstream commit adc49a85, "perf(cake_vsa): refresh Blackwell
  block-sparse WS kernel (#4804)").
- ``csrc/cake_vsa/cake_vsa_ultrasparse_bsr_host.cpp`` -- the tvm-ffi launcher
  that encodes the three 4-D SWIZZLE_128B tensor maps, requires exactly six
  selected blocks per query block and ``total_tiles == mb * num_q_heads``, and
  launches 512 threads with 135424 bytes of dynamic SMEM.
- ``flashinfer/cake_vsa.py`` -- ``plan_cake_vsa``/``run_cake_vsa`` dispatch.
  The ``ultrasparse_bsr`` profile is the route for bf16, head_dim 128, R = C =
  128, ``num_qo_heads == num_kv_heads == 8``, at least 625 query blocks, one BSR
  pattern shared by all heads with exactly six blocks in every row, and
  ``return_lse=False``; it launches ``grid = (min(mb * 8, SM count), 1, 1)``.

Kernel structure (persistent: every CTA walks ``tile_idx = blockIdx.x; tile_idx <
total_tiles; tile_idx += gridDim.x`` with ``q_tile = tile_idx % mb`` and
``q_head = tile_idx / mb``):

- warp 15 waits for the previous tile's epilogue, loads the 128-row Q tile with
  one 32 KB TMA transaction, copies the six BSR column indices of the query block
  into shared memory, then streams K and V tiles through a three-stage 32 KB
  SMEM ring in the order K5 K4 V5 K3 V4 K2 V3 K1 V2 K0 V1 V0;
- warp 12 issues every ``tcgen05.mma.cta_group::1.kind::f16``: eight K=16
  steps for ``S_i = Q K^T`` (M=128, N=128) per instance into TMEM, then eight
  steps for ``O_i += P_i V`` with P read from TMEM;
- warps 0-7 form two four-warp softmax instances that both own all 128 query
  rows and alternate over the six KV blocks (instance i handles blocks
  5-i, 3-i, 1-i): each thread reads its full 128-column fp32 score row from
  TMEM, applies the ``2^-8`` rescale-skip decision, writes bf16 P back in place
  and keeps per-row running max and sum;
- warps 8-11 rescale the TMEM O accumulator of an instance when a warp vote says
  any row needs it, then merge the two instances' accumulators and statistics
  (combine scales and one reciprocal), store bf16 output rows and, when
  requested, the natural-log LSE (also duplicated into ``temperature_lse``).

Every SMEM object lives in one rank-one byte arena addressed by explicit scalar
offsets; no first-class layouts or tile primitives are used.
"""

import math
from functools import lru_cache
from typing import Any

import tirx_kernels.kern as K

KERNEL_META = {
    "name": "cake_vsa_ultrasparse_bsr_sm100",
    "category": "flashinfer",
    "runtime_cuda_archs": ["sm_100a"],
    "reference_requirements": (
        {
            "package": "flashinfer-python",
            "git": {
                "url": "https://github.com/flashinfer-ai/flashinfer.git",
                "commit": "c5365737570a2a156d7cae0c4070fa3770ecc670",
            },
            "import": "flashinfer",
        },
    ),
}
# The upstream kernel is a prebuilt cubin loaded through FlashInfer's own tvm-ffi
# launcher (``flashinfer.cake_vsa._load_module``); tvm-ffi is a dependency of
# flashinfer-python itself, so only FlashInfer is pinned.

# ---------------------------------------------------------------------------
# Source contract constants (host.cpp / cake_vsa.py / the device kernel).
# ---------------------------------------------------------------------------
HEAD_DIM = 128
BLOCK = 128  # R == C == 128 query/KV block size of this profile
SELECTED_BLOCKS = 6  # host.cpp: selected_blocks must equal 6
NUM_HEADS = 8  # cake_vsa.py: num_qo_heads == num_kv_heads == 8 on this route
MIN_QUERY_BLOCKS = 625  # cake_vsa.py: mb >= 625 on this route
THREADS = 512
NUM_WARPS = THREADS // 32
SMEM_TOTAL = 135424
TMEM_COLS = 512
PROFILE = "ultrasparse_bsr"
CUDA_ARCH = "sm_100a"
_LN2 = math.log(2.0)

# ``lse`` sentinel used when the caller does not request LSE: upstream passes a
# one-element ``stats`` tensor that the kernel must leave untouched.
_LSE_SENTINEL = 12345.25

# Oracle tolerances (fp64 block-gather reference), frozen from the errors measured
# on GB200 across CONFIGS with a 1.25-1.6x margin (the TIRx-vs-source comparison is
# bitwise, so both implementations carry the same algorithmic error: bf16 P and
# output rounding, ``ex2.approx``/``rcp.approx``/``lg2.approx``).  Measured worst
# cases: out error/bound 0.79 (m81920_n131072 unsorted), bf16 ULP max 2 with
# >= 0.9999996 of the sizeable elements within 1 ULP, LSE abs error 1.1e-6.
# ``max_rel_excess`` in the returned stats is the worst error / (atol + rtol*|ref|)
# ratio.  The bf16 ULP checks apply to |ref| >= 0.125, where one ULP is at least 2^-10.
_ORACLE_OUT_ATOL = 1.5e-3
_ORACLE_OUT_RTOL = 6e-3
_ORACLE_LSE_ATOL = 6e-7
_ORACLE_LSE_RTOL = 1.5e-7
_ORACLE_ULP_MAX = 2
_ORACLE_ULP_LE1_FRAC = 0.9999
_ORACLE_ULP_LE2_FRAC = 1.0
_ORACLE_ULP_MIN_REF = 0.125
_ORACLE_TILES_PER_CHUNK = 16

# Per-band K amplification of the ``kramp`` pattern, indexed by band (= slot).
# Slots 5,4 are consumed first (pair 0), then 3,2, then 1,0, so the running max
# grows by more than 2^8 (in the exp2 domain) at every pair boundary.
_KRAMP_BAND_AMPS = (4.0, 4.0, 2.0, 2.0, 1.0, 1.0)


def _config(
    label: str,
    *,
    M: int,
    N: int,
    pattern: str,
    return_lse: bool = False,
    return_temperature_lse: bool = False,
    q_amp: float = 1.0,
    sm_scale: float | None = None,
    lse_temperature_scale: float = 1.0,
    api_check: str | None = "block_mask",
    seed: int = 0,
    oracle_atol: float = _ORACLE_OUT_ATOL,
    oracle_rtol: float = _ORACLE_OUT_RTOL,
    oracle_lse_atol: float = _ORACLE_LSE_ATOL,
    oracle_lse_rtol: float = _ORACLE_LSE_RTOL,
    ulp_max: int = _ORACLE_ULP_MAX,
    ulp_le1_frac: float = _ORACLE_ULP_LE1_FRAC,
    ulp_le2_frac: float = _ORACLE_ULP_LE2_FRAC,
) -> dict[str, Any]:
    return {
        "label": label,
        "M": M,
        "N": N,
        "pattern": pattern,
        "return_lse": return_lse,
        "return_temperature_lse": return_temperature_lse,
        "q_amp": q_amp,
        "sm_scale": sm_scale,
        "lse_temperature_scale": lse_temperature_scale,
        "api_check": api_check,
        "seed": seed,
        "oracle_atol": oracle_atol,
        "oracle_rtol": oracle_rtol,
        "oracle_lse_atol": oracle_lse_atol,
        "oracle_lse_rtol": oracle_lse_rtol,
        "ulp_max": ulp_max,
        "ulp_le1_frac": ulp_le1_frac,
        "ulp_le2_frac": ulp_le2_frac,
    }


# Correctness matrix.  Every row stays inside the ``ultrasparse_bsr`` dispatch
# domain (asserted by ``_assert_ultrasparse_route``): 8 heads, at least 625 query
# blocks, exactly six distinct KV blocks per query block shared by all heads.  The
# public API refuses ``return_lse`` on this route; the device kernel implements the
# LSE and temperature-LSE stores, which the direct launch exercises.
CONFIGS = [
    # 5000 tiles over 152 CTAs: 136 CTAs run 33 tiles, 16 run 32 (uneven persistent
    # tail; barrier phases live across 33 tiles).  No LSE: the sentinel must survive.
    _config("m80000_n8192_h8_sel6_tail", M=80000, N=8192, pattern="random_sorted", seed=0),
    # nb == 6: every KV block is selected (K5..K0 covers the whole sequence); 627 query
    # blocks give exactly 33 tiles per CTA.  Kernel-ABI LSE store.
    _config(
        "m80256_n768_h8_sel6_dense_lse", M=80256, N=768, pattern="dense6", return_lse=True, seed=1
    ),
    # 1024 KV blocks (token coordinates up to 131071) in caller order (unsorted per
    # row), handed to the public API through its trusted indptr/indices path.
    _config(
        "m81920_n131072_h8_sel6_unsorted",
        M=81920,
        N=131072,
        pattern="random_unsorted",
        api_check="indices",
        seed=2,
    ),
    # Banded pattern with K rows amplified per band (x4, x4, x2, x2, x1, x1) and Q
    # amplified x4: the running max grows by far more than 2^8 in pairs 1 and 2 on
    # every row, so the O-rescale branch, the warp vote and unequal combine scales
    # all execute.  Skipped rescales leave bf16 P values up to 2^8, so this regime
    # carries a larger inherent error than the default tolerance (measured: error/bound
    # 0.76, ULP max 7, 0.99962 within 1 ULP, 0.999982 within 2 ULP).
    _config(
        "m80000_n4096_h8_sel6_kramp",
        M=80000,
        N=4096,
        pattern="kramp",
        q_amp=4.0,
        seed=3,
        oracle_atol=6.5e-3,
        oracle_rtol=1e-2,
        ulp_max=8,
        ulp_le1_frac=0.9995,
        ulp_le2_frac=0.9998,
    ),
    # 8192 tiles (54 waves) with six consecutive blocks around the diagonal:
    # adjacent persistent tiles reuse K/V.
    _config("m131072_n32768_h8_sel6_window", M=131072, N=32768, pattern="window", seed=4),
    # Blocks 0 and nb-1 always selected (first/last TMA coordinates); both LSE flags
    # with two distinct full-size buffers (the kernel writes the same value to both).
    _config(
        "m80000_n1024_h8_sel6_edges_bothlse",
        M=80000,
        N=1024,
        pattern="edges",
        return_lse=True,
        return_temperature_lse=True,
        seed=5,
    ),
]

# Benchmark matrix (no LSE; the upstream Python path never requests it here).
BENCH_CONFIGS = [
    _config("m80000_n8192_h8_sel6", M=80000, N=8192, pattern="random_sorted", seed=100),
    _config("m131072_n32768_h8_sel6", M=131072, N=32768, pattern="random_sorted", seed=101),
    _config("m262144_n131072_h8_sel6", M=262144, N=131072, pattern="random_sorted", seed=102),
    _config("m80256_n768_h8_sel6_dense", M=80256, N=768, pattern="dense6", seed=103),
    _config("m131072_n16384_h8_sel6_window", M=131072, N=16384, pattern="window", seed=104),
    _config("m80000_n4096_h8_sel6_kramp", M=80000, N=4096, pattern="kramp", q_amp=4.0, seed=105),
]


def _without_label(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "label"}


def _validate_shape(M: int, N: int) -> tuple[int, int]:
    if M <= 0 or M % BLOCK != 0:
        raise ValueError(f"M={M} must be a positive multiple of {BLOCK}")
    if N <= 0 or N % BLOCK != 0:
        raise ValueError(f"N={N} must be a positive multiple of {BLOCK}")
    mb, nb = M // BLOCK, N // BLOCK
    if nb < SELECTED_BLOCKS:
        raise ValueError(f"N={N} has {nb} KV blocks; the route selects {SELECTED_BLOCKS} per row")
    return mb, nb


def _assert_ultrasparse_route(indices, *, mb: int, nb: int) -> None:
    """Re-evaluate the upstream ``run_cake_vsa`` dispatch predicates for this pattern.

    bf16, D=128, R=C=128 and eight MHA heads are fixed by this module.  The route
    is taken when ``mb >= 625``, the BSR pattern is shared by all heads (true by
    construction: one ``(mb, 6)`` index table) and every row selects exactly six
    distinct blocks.
    """
    if mb < MIN_QUERY_BLOCKS:
        raise ValueError(f"mb={mb} < {MIN_QUERY_BLOCKS}: shape would dispatch to blk128_compact")
    if tuple(indices.shape) != (mb, SELECTED_BLOCKS):
        raise ValueError(f"indices must be ({mb}, {SELECTED_BLOCKS}), got {tuple(indices.shape)}")
    if int(indices.min()) < 0 or int(indices.max()) >= nb:
        raise ValueError("BSR column index out of range")
    sorted_rows = indices.sort(dim=1).values
    if bool((sorted_rows[:, 1:] == sorted_rows[:, :-1]).any()):
        raise ValueError("a row selects a KV block twice; the route needs six distinct blocks")


def _make_bsr(*, mb: int, nb: int, pattern: str, generator):
    """Build the ``(mb, 6)`` int32 table of selected KV blocks in kernel order."""
    import torch

    if pattern == "dense6":
        if nb != SELECTED_BLOCKS:
            raise ValueError("pattern 'dense6' needs exactly six KV blocks")
        return torch.arange(SELECTED_BLOCKS, dtype=torch.int32).repeat(mb, 1)
    if pattern in {"random_sorted", "random_unsorted"}:
        rows = torch.stack(
            [torch.randperm(nb, generator=generator)[:SELECTED_BLOCKS] for _ in range(mb)]
        )
        if pattern == "random_sorted":
            rows = rows.sort(dim=1).values
        elif bool((rows[:, 1:] > rows[:, :-1]).all(dim=1).all()):
            raise ValueError("pattern 'random_unsorted' produced only sorted rows")
        return rows.to(torch.int32)
    if pattern == "kramp":
        # Slot p picks one block inside band p; bands are six equal ranges of blocks.
        edges = [nb * p // SELECTED_BLOCKS for p in range(SELECTED_BLOCKS + 1)]
        columns = []
        for p in range(SELECTED_BLOCKS):
            lo, hi = edges[p], edges[p + 1]
            if hi <= lo:
                raise ValueError("pattern 'kramp' needs at least six KV blocks")
            columns.append(lo + torch.randint(0, hi - lo, (mb,), generator=generator))
        return torch.stack(columns, dim=1).to(torch.int32)
    if pattern == "window":
        centre = (torch.arange(mb) * nb) // mb
        start = (centre - SELECTED_BLOCKS // 2).clamp(0, nb - SELECTED_BLOCKS)
        return (start.unsqueeze(1) + torch.arange(SELECTED_BLOCKS)).to(torch.int32)
    if pattern == "edges":
        if nb < SELECTED_BLOCKS + 2:
            raise ValueError("pattern 'edges' needs interior blocks to choose from")
        rows = []
        for _ in range(mb):
            interior = 1 + torch.randperm(nb - 2, generator=generator)[: SELECTED_BLOCKS - 2]
            rows.append(torch.cat([torch.tensor([0, nb - 1]), interior]).sort().values)
        return torch.stack(rows).to(torch.int32)
    raise ValueError(f"unknown BSR pattern {pattern!r}")


def _kramp_row_amps(nb: int):
    """Per-KV-token amplification of the ``kramp`` pattern (band of block b = b*6//nb)."""
    import torch

    band = (torch.arange(nb) * SELECTED_BLOCKS) // nb
    amps = torch.tensor(_KRAMP_BAND_AMPS, dtype=torch.float32)[band]
    return amps.repeat_interleave(BLOCK)  # (N,)


def _driver_sm_count() -> int | None:
    """SM count of device 0 through the CUDA driver API (no runtime context is created)."""
    import ctypes

    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    if lib.cuInit(0) != 0:
        return None
    device = ctypes.c_int()
    if lib.cuDeviceGet(ctypes.byref(device), 0) != 0:
        return None
    value = ctypes.c_int()
    # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT == 16
    if lib.cuDeviceGetAttribute(ctypes.byref(value), 16, device) != 0:
        return None
    return int(value.value)


def _sm_count() -> int:
    """The ``multi_processor_count`` the upstream host uses for ``grid_x``.

    The bench suite's CPU prepare must not initialise CUDA, so the repository's
    ``hardware_num_sms`` (environment-provided compile profile, or torch once it is
    initialised) is consulted first; outside the suite the driver API answers.
    """
    import os
    import sys

    from tirx_kernels.runner import PREPARE_NUM_SMS_ENV, hardware_num_sms

    torch = sys.modules.get("torch")
    if os.environ.get(PREPARE_NUM_SMS_ENV) is not None or (
        torch is not None and torch.cuda.is_initialized()
    ):
        return hardware_num_sms()
    count = _driver_sm_count()
    return count if count else hardware_num_sms()


def _grid_x(total_tiles: int) -> int:
    """``min(total_tiles, sm_count)`` persistent CTAs, as ``_run_standard`` launches."""
    return min(total_tiles, _sm_count())


def prepare_data(**config: Any) -> dict[str, Any]:
    """Allocate logical inputs plus independent TIRx and source output buffers."""
    import torch

    config = _without_label(config)
    M, N = config["M"], config["N"]
    mb, nb = _validate_shape(M, N)
    num_heads = NUM_HEADS
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
    if config["pattern"] == "kramp":
        k = k * _kramp_row_amps(nb).view(N, 1, 1)
    k = k.to(torch.bfloat16).to(device)
    v = torch.randn((N, num_heads, HEAD_DIM), generator=generator, dtype=torch.float32)
    v = v.to(torch.bfloat16).to(device)
    indices = _make_bsr(mb=mb, nb=nb, pattern=config["pattern"], generator=generator)
    _assert_ultrasparse_route(indices, mb=mb, nb=nb)
    bsr_indices = indices.contiguous().to(device)
    total_tiles = mb * num_heads

    def outputs() -> dict[str, Any]:
        out = torch.full(
            (M, num_heads, HEAD_DIM), float("nan"), dtype=torch.bfloat16, device=device
        )
        if return_lse:
            lse = torch.full((M, num_heads), float("nan"), dtype=torch.float32, device=device)
        else:
            lse = torch.full((1,), _LSE_SENTINEL, dtype=torch.float32, device=device)
        if return_tlse:
            tlse = torch.full((M, num_heads), float("nan"), dtype=torch.float32, device=device)
        else:
            tlse = lse  # upstream passes the same ``stats`` tensor for both slots
        return {"out": out, "lse": lse, "tlse": tlse}

    return {
        "config": config,
        "M": M,
        "N": N,
        "num_heads": num_heads,
        "mb": mb,
        "nb": nb,
        "total_tiles": total_tiles,
        "grid_x": _grid_x(total_tiles),
        "q": q,
        "k": k,
        "v": v,
        "bsr_indices": bsr_indices,
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
    fill.  Q is ``{64, M, Hq, 2}`` with a ``{64, 128, 1, 2}`` box (one 32 KB
    transaction covers both head-dim halves of the 128-row tile); K and V are
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
        (64, BLOCK, 1, HEAD_DIM // 64),
    )
    kv_dims = (64, nb * BLOCK, HEAD_DIM // 64, num_kv_heads)
    kv_strides = (num_kv_heads * HEAD_DIM * elem_bytes, 64 * elem_bytes, HEAD_DIM * elem_bytes)
    kv_box = (64, 64, 1, 1)
    k_map = encode(params["k"], kv_dims, kv_strides, kv_box)
    v_map = encode(params["v"], kv_dims, kv_strides, kv_box)
    return (q_map, k_map, v_map)


# Byte offsets inside the 135424-byte dynamic SMEM arena (source macro table).
_MBAR_Q_FULL = 0
_MBAR_Q_EMPTY = 8
_MBAR_UNION_READY = 16
_MBAR_KV_FULL = 24  # 3 stages
_MBAR_KV_EMPTY = 48  # 3 stages
_MBAR_S_FULL = 72  # 2 instances
_MBAR_P_FULL = 88  # 2 instances
_MBAR_CORR_SIG = 104  # 2 instances
_MBAR_CORR_DONE = 120  # 2 instances
_MBAR_O_FULL = 136  # 2 instances
_MBAR_TILE_DONE = 152
_SMEM_TMEM_MAILBOX = 160
_SMEM_Q = 1024  # 128 rows x 128: two 16 KB dim halves
_SMEM_KV = 33792  # 3 x 32768, K and V share the ring
_SMEM_KV_STAGE_BYTES = 32768
_SMEM_SCALE = 132096  # 768 f32: acc_scale[256] / row_sum[256] / row_max[256]
_SMEM_UNION_BLOCKS = 135172  # the six selected KV block indices of the tile
# (offset, arrival count) in source order: q_full, q_empty, union_ready, kv_full x3,
# kv_empty x3, s_full x2, p_full x2, corr_sig x2, corr_done x2, o_full x2, tile_done.
_MBARRIER_INIT = (
    (_MBAR_Q_FULL, 1),
    (_MBAR_Q_EMPTY, 128),
    (_MBAR_UNION_READY, 32),
    (_MBAR_KV_FULL, 1),
    (_MBAR_KV_FULL + 8, 1),
    (_MBAR_KV_FULL + 16, 1),
    (_MBAR_KV_EMPTY, 1),
    (_MBAR_KV_EMPTY + 8, 1),
    (_MBAR_KV_EMPTY + 16, 1),
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
    (_MBAR_TILE_DONE, 128),
)
# TMEM column bases per softmax instance: S (128 columns; P overwrites the upper 64
# in place) and the O accumulator (128 columns).
_TMEM_SCORES = (0, 128)
_TMEM_OUTPUT = (256, 384)
_TMEM_P_OFFSET = 64
# UMMA descriptor words and instruction descriptors (source immediates).
_DESC_HI = 0x40004040  # SBO 1024 B, base-offset bit, 128 B swizzle
_QK_IDESC = 0x08200490  # bf16 x bf16 -> f32, M=128, N=128, K-major A and B
_PV_IDESC = 0x08210490  # same with MN-major B (V is token-major)
_QK_STEPS = (2, 2, 2, 1018, 2, 2, 2)  # Q and K low-word walk over K=128 (bytes/16)
_V_LBO_BIT = 0x4000000  # LBO = 16384 B: dim-half stride of the V tile
_PV_B_STEP = 128  # 2048 B = 16 tokens x 128 B per K16 step
_PV_A_STEP = 8  # 16 bf16 P columns = 8 TMEM 32-bit columns per K16 step
_KV_STAGES = 3
_PUSHES_PER_TILE = 12  # K5 K4 V5 K3 V4 K2 V3 K1 V2 K0 V1 V0
_REG_SOFTMAX = 192
_REG_CORRECTION = 80
_REG_OTHER = 48

_TMA_G2S_4D = "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes"
_MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
_TCGEN05_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
_TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
_TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
_TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
_TMEM_LD_X32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
_TMEM_LD_X16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
_TMEM_LD_X8 = "tcgen05.ld.sync.aligned.32x32b.x8.b32"
_TMEM_ST_X16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
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


def _flip(phase):
    K.assign(phase, phase ^ _i32(1))


def _ring_schedule(initial_parity: int):
    """(stage, parity) of the 12 ring waits of one tile; 12 pushes are four full turns."""
    return [
        (i % _KV_STAGES, ((i // _KV_STAGES) & 1) ^ initial_parity) for i in range(_PUSHES_PER_TILE)
    ]


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


def _max_f32(a, b):
    out = K.local_scalar("float32")
    K.ptx.max.f32(out, a, b)
    return out


def _tmem_load_x32(dst, base, taddr):
    K.ptx[_TMEM_LD_X32](*(dst[base + i] for i in range(32)), taddr)


def _tmem_load_x16(dst, taddr):
    K.ptx[_TMEM_LD_X16](*(dst[i] for i in range(16)), taddr)


def _tmem_load_x8(dst, taddr):
    K.ptx[_TMEM_LD_X8](*(dst[i] for i in range(8)), taddr)


def _tmem_store_x16(taddr, src):
    K.ptx[_TMEM_ST_X16](taddr, *(src[i] for i in range(16)))


def _row_max_128(values):
    """``row_max_x32_accum`` x4 + ``row_max_reduce``: alternating accumulators over 64 pairs."""
    acc0 = K.local_scalar("float32", init=_f32(_NEG_INF))
    acc1 = K.local_scalar("float32", init=_f32(_NEG_INF))
    for quarter in range(4):
        for j in range(16):
            pair = _max_f32(values[32 * quarter + 2 * j], values[32 * quarter + 2 * j + 1])
            if j % 2 == 0:
                K.assign(acc0, _max_f32(acc0, pair))
            else:
                K.assign(acc1, _max_f32(acc1, pair))
    return _max_f32(acc0, acc1)


def _block_sum_128(values):
    """``softmax_block_sum`` x4: 64 packed ``add.f32x2`` into one accumulator, then ``.x + .y``."""
    acc = K.local_scalar("uint64")
    K.ptx.mov.b64(acc, _f32(0.0), _f32(0.0))
    for j in range(64):
        K.ptx.add.f32x2(acc, acc, _pack2(values[2 * j], values[2 * j + 1]))
    acc_x = K.local_scalar("float32")
    acc_y = K.local_scalar("float32")
    K.ptx.mov.b64(acc_x, acc_y, acc)
    total = K.local_scalar("float32")
    K.ptx.add.ftz.f32(total, acc_x, acc_y)
    return total


def _build_kernel():
    sm_count = _sm_count()

    @K.kernel(
        warps=NUM_WARPS,
        arch=CUDA_ARCH,
        min_blocks_per_sm=1,
        grid=lambda p: K.min(p["total_tiles"], K.int32(sm_count)),
        host_prelude=_host_prelude,
    )
    def cake_vsa_ultrasparse_bsr_sm100(
        q: K.gptr[K.bf16],
        k: K.gptr[K.bf16],
        v: K.gptr[K.bf16],
        out: K.gptr[K.bf16],
        lse: K.gptr[K.f32],
        temperature_lse: K.gptr[K.f32],
        bsr_indices: K.gptr[K.i32],
        mb: K.i32,
        nb: K.i32,
        selected_blocks: K.i32,
        total_tiles: K.i32,
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
        # ``selected_blocks`` (host-checked to be 6), ``num_kv_heads`` (the CTA uses
        # ``kv_head = q_head``) and ``lse_temperature_scale`` are ABI arguments the
        # device never reads; q/k/v are reached only through the tensor maps.
        del q, k, v, selected_blocks, num_kv_heads, lse_temperature_scale
        # >>> kernel_flashinfer_blackwell_vsa_ultrasparse_bsr_sm100 body starts here
        bid = K.cta_id()
        warp = K.warp_id()
        lane = K.lane_id()

        def num_bids():
            # ``gridDim.x``; read at every tile increment so no long-lived register is
            # needed (the source's plain special-register read is rematerialised by ptxas).
            return K.cast(K.cuda.mov_sreg(32, "nctaid.x"), "uint32")

        arena = K.alloc_buffer((SMEM_TOTAL,), K.u8, scope="shared.dyn", align=1024)
        smem = K.local_scalar("uint32", init=K.cuda.cvta_generic_to_shared(arena.ptr_to([0])))

        def bar(offset):
            return smem + _u32(offset)

        # Mbarrier init (11 groups, 20 barriers) by one elected lane of warp 0.
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

        total_tiles_u = K.cast(total_tiles, "uint32")
        mb_u = K.cast(mb, "uint32")
        kv_len = nb * 128

        # Sibling role guards, as in the source.
        roles = K.specialize(chain_dispatch=False)
        other_regs = roles.register_scope("other", warps=range(12, 16), regs=_REG_OTHER)
        r_softmax = roles.role("softmax", warps=range(0, 8), regs=_REG_SOFTMAX)
        r_correction = roles.role("correction", warps=range(8, 12), regs=_REG_CORRECTION)
        r_mma = roles.role("mma", warps=[12], register_scope=other_regs)
        r_idle = roles.role("idle", warps=[13, 14], register_scope=other_regs)
        r_load = roles.role("load", warps=[15], register_scope=other_regs)

        # Register redistribution: warps 12..15 decrease first so the softmax
        # increase can be granted.
        with K.If(K.And(warp >= 12, warp <= 15)), K.Then():
            other_regs.emit()

        # ---- Role: softmax (warps 0..7, two four-warp instances over alternate blocks) ----
        with r_softmax:
            phase_union_ready = K.local_scalar("int32", init=_i32(0))
            phase_s_full = [K.local_scalar("int32", init=_i32(0)) for _ in range(2)]
            phase_corr_done = [K.local_scalar("int32", init=_i32(0)) for _ in range(2)]
            scores = K.alloc_local((128,), "float32")
            packed_p = K.alloc_local((16,), "uint32")
            tile_idx = K.local_scalar("uint32", init=K.cast(bid, "uint32"))
            with K.While(tile_idx < total_tiles_u):
                q_tile = K.local_scalar("int32", init=K.cast(tile_idx % mb_u, "int32"))
                q_valid = K.local_scalar("int32", init=_i32(128))
                with K.If(q_tile >= mb), K.Then():
                    K.assign(q_valid, _i32(0))
                _mbar_wait(bar(_MBAR_UNION_READY), phase_union_ready)
                _flip(phase_union_ready)
                instance = K.uniform(warp // 4)
                instance_tmem_offset = K.uniform(instance * 128)
                instance_row_offset = K.uniform(instance * 128)
                warp_in_instance = warp % 4
                tmem_row_origin = warp_in_instance * 32
                my_row = tmem_row_origin + lane
                row_addr = K.shift_left(K.cast(tmem_row_origin, "uint32"), _u32(16))
                row_valid = K.local_scalar("int32", init=K.cast(my_row < q_valid, "int32"))
                row_max = K.local_scalar("float32", init=_f32(_NEG_INF))
                row_sum = K.local_scalar("float32", init=_f32(0.0))
                score_addr = K.local_scalar(
                    "uint32", init=taddr + K.cast(instance_tmem_offset, "uint32") + row_addr
                )
                p_addr = K.local_scalar("uint32", init=score_addr + _u32(_TMEM_P_OFFSET))
                scale_row_addr = bar(_SMEM_SCALE) + K.cast(
                    instance_row_offset + my_row, "uint32"
                ) * _u32(4)
                for pair in range(3):
                    # Instance ``i`` consumes slot 5 - i - 2*pair of the tile's block list.
                    n_block = _ld_shared_i32(
                        bar(_SMEM_UNION_BLOCKS + 4 * (5 - 2 * pair))
                        - K.cast(instance, "uint32") * _u32(4)
                    )
                    with K.If(instance == 0):
                        with K.Then():
                            _mbar_wait(bar(_MBAR_S_FULL), phase_s_full[0])
                            _flip(phase_s_full[0])
                        with K.Else():
                            _mbar_wait(bar(_MBAR_S_FULL + 8), phase_s_full[1])
                            _flip(phase_s_full[1])
                    valid_cols = K.local_scalar("int32", init=_i32(0))
                    with K.If(row_valid != _i32(0)), K.Then():
                        K.assign(valid_cols, kv_len - n_block * 128)
                        with K.If(valid_cols > _i32(128)), K.Then():
                            K.assign(valid_cols, _i32(128))
                        with K.If(valid_cols < _i32(0)), K.Then():
                            K.assign(valid_cols, _i32(0))
                    for quarter in range(4):
                        _tmem_load_x32(scores, 32 * quarter, score_addr + _u32(32 * quarter))
                    # valid_cols is 0 or 128 (128*(nb - n_block) clamped), so the source's
                    # per-lane masks are all-zero whenever this branch is taken: the whole
                    # fragment becomes -inf, which is what nvcc emits for the source.
                    with K.If(valid_cols < _i32(128)), K.Then():
                        for j in range(128):
                            K.assign(scores[j], _f32(_NEG_INF))
                    tile_max = _row_max_128(scores)
                    new_max = _max_f32(tile_max, row_max)
                    safe_max = K.local_scalar(
                        "float32",
                        init=K.if_then_else(new_max == _f32(_NEG_INF), _f32(0.0), new_max),
                    )
                    new_max_scaled = K.local_scalar("float32")
                    K.ptx.mul.ftz.f32(new_max_scaled, safe_max, softmax_scale_log2)
                    neg_new_max_scaled = K.local_scalar("float32")
                    K.ptx.neg.ftz.f32(neg_new_max_scaled, new_max_scaled)
                    acc_scale_log2 = K.local_scalar("float32")
                    K.ptx.fma.rn.ftz.f32(
                        acc_scale_log2, row_max, softmax_scale_log2, neg_new_max_scaled
                    )
                    acc_scale = K.local_scalar("float32")
                    selected_max = K.local_scalar("float32")
                    with K.If(acc_scale_log2 >= _f32(-8.0)):
                        with K.Then():
                            # The running max moves by less than 2^8: keep the old max, skip the rescale.
                            K.assign(selected_max, row_max)
                            K.assign(
                                safe_max,
                                K.if_then_else(row_max == _f32(_NEG_INF), _f32(0.0), row_max),
                            )
                            K.assign(acc_scale, _f32(1.0))
                            K.ptx.mul.ftz.f32(new_max_scaled, safe_max, softmax_scale_log2)
                        with K.Else():
                            K.assign(selected_max, new_max)
                            exp_scale = K.local_scalar("float32")
                            K.ptx.ex2.approx.ftz.f32(exp_scale, acc_scale_log2)
                            K.assign(
                                acc_scale,
                                K.if_then_else(row_max > _f32(_NEG_INF), exp_scale, _f32(1.0)),
                            )
                    K.assign(row_max, selected_max)
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
                    scale_pair = _pack2(softmax_scale_log2, softmax_scale_log2)
                    bias_pair = _pack2(neg_bias, neg_bias)
                    for j in range(64):
                        _packed_fma_inplace(scores, 2 * j, scale_pair, bias_pair)
                    for quarter in range(4):
                        base = 32 * quarter
                        for j in range(32):
                            K.ptx.ex2.approx.ftz.f32(scores[base + j], scores[base + j])
                        for j in range(16):
                            K.ptx.cvt.rn.bf16x2.f32(
                                packed_p[j], scores[base + 2 * j + 1], scores[base + 2 * j]
                            )
                        _tmem_store_x16(p_addr + _u32(16 * quarter), packed_p)
                    K.ptx.tcgen05.wait__st.sync.aligned()
                    with K.If(instance == 0):
                        with K.Then():
                            _mbar_arrive(bar(_MBAR_P_FULL))
                            _mbar_wait(bar(_MBAR_CORR_DONE), phase_corr_done[0])
                            _flip(phase_corr_done[0])
                        with K.Else():
                            _mbar_arrive(bar(_MBAR_P_FULL + 8))
                            _mbar_wait(bar(_MBAR_CORR_DONE + 8), phase_corr_done[1])
                            _flip(phase_corr_done[1])
                    block_sum = _block_sum_128(scores)
                    K.ptx.fma.rn.ftz.f32(row_sum, row_sum, acc_scale, block_sum)
                K.ptx.st.shared.b32(scale_row_addr + _u32(256 * 4), row_sum)
                K.ptx.st.shared.b32(scale_row_addr + _u32(512 * 4), row_max)
                K.ptx.fence.proxy.async_.shared__cta()
                with K.If(instance == 0):
                    with K.Then():
                        _mbar_arrive(bar(_MBAR_CORR_SIG))
                    with K.Else():
                        _mbar_arrive(bar(_MBAR_CORR_SIG + 8))
                K.assign(tile_idx, tile_idx + num_bids())

        # ---- Role: correction and epilogue (warps 8..11) ----
        with r_correction:
            phase_union_ready_c = K.local_scalar("int32", init=_i32(0))
            phase_corr_sig = [K.local_scalar("int32", init=_i32(0)) for _ in range(2)]
            phase_o_full = [K.local_scalar("int32", init=_i32(0)) for _ in range(2)]
            o16 = K.alloc_local((16,), "float32")
            o0 = K.alloc_local((8,), "float32")
            o1 = K.alloc_local((8,), "float32")
            words = K.alloc_local((4,), "uint32")
            tile_idx_c = K.local_scalar("uint32", init=K.cast(bid, "uint32"))
            with K.While(tile_idx_c < total_tiles_u):
                q_head = K.local_scalar("int32", init=K.cast(tile_idx_c // mb_u, "int32"))
                q_tile = K.local_scalar("int32", init=K.cast(tile_idx_c, "int32") - q_head * mb)
                query_base = q_tile * 128
                q_valid = K.local_scalar("int32", init=_i32(128))
                with K.If(q_tile >= mb), K.Then():
                    K.assign(q_valid, _i32(0))
                _mbar_wait(bar(_MBAR_UNION_READY), phase_union_ready_c)
                _flip(phase_union_ready_c)
                warp_in_role = warp - 8
                tmem_row_origin = warp_in_role * 32
                my_row = tmem_row_origin + lane
                row_addr = K.local_scalar(
                    "uint32", init=K.shift_left(K.cast(tmem_row_origin, "uint32"), _u32(16))
                )
                scale_base = bar(_SMEM_SCALE) + K.cast(my_row, "uint32") * _u32(4)
                # Pair 0 needs no O rescale: the correction half of p_full arrives at once.
                _mbar_arrive(bar(_MBAR_P_FULL))
                _mbar_arrive(bar(_MBAR_P_FULL + 8))
                for instance in range(2):
                    _mbar_wait(bar(_MBAR_CORR_SIG + 8 * instance), phase_corr_sig[instance])
                    _flip(phase_corr_sig[instance])
                    _mbar_arrive(bar(_MBAR_CORR_DONE + 8 * instance))
                for _pair in range(1, 3):
                    for instance in range(2):
                        _mbar_wait(bar(_MBAR_CORR_SIG + 8 * instance), phase_corr_sig[instance])
                        _flip(phase_corr_sig[instance])
                        acc_scale_c = _ld_shared_f32(scale_base + _u32(128 * 4 * instance))
                        any_rescale = K.local_scalar("uint32")
                        K.ptx.vote_sync.any.pred(
                            any_rescale,
                            K.ptx.pred(K.cast(acc_scale_c < _f32(1.0), "bool")),
                            _u32(_FULL_MASK),
                        )
                        with K.If(any_rescale != _u32(0)), K.Then():
                            scale_pair = _pack2(acc_scale_c, acc_scale_c)
                            for chunk in range(8):
                                cr_addr = (
                                    taddr + _u32(_TMEM_OUTPUT[instance] + 16 * chunk) + row_addr
                                )
                                _tmem_load_x16(o16, cr_addr)
                                for j in range(8):
                                    _packed_mul_inplace(o16, 2 * j, scale_pair)
                                _tmem_store_x16(cr_addr, o16)
                            K.ptx.tcgen05.wait__st.sync.aligned()
                        _mbar_arrive(bar(_MBAR_P_FULL + 8 * instance))
                        _mbar_arrive(bar(_MBAR_CORR_DONE + 8 * instance))
                for instance in range(2):
                    _mbar_wait(bar(_MBAR_O_FULL + 8 * instance), phase_o_full[instance])
                    _flip(phase_o_full[instance])
                for instance in range(2):
                    _mbar_wait(bar(_MBAR_CORR_SIG + 8 * instance), phase_corr_sig[instance])
                    _flip(phase_corr_sig[instance])
                K.ptx["tcgen05.fence::after_thread_sync"]()
                final_sum0 = _ld_shared_f32(scale_base + _u32(256 * 4))
                final_sum1 = _ld_shared_f32(scale_base + _u32(384 * 4))
                final_max0 = _ld_shared_f32(scale_base + _u32(512 * 4))
                final_max1 = _ld_shared_f32(scale_base + _u32(640 * 4))
                valid0 = K.local_scalar("int32", init=K.cast(final_sum0 > _f32(0.0), "int32"))
                valid1 = K.local_scalar("int32", init=K.cast(final_sum1 > _f32(0.0), "int32"))
                max0 = K.local_scalar(
                    "float32", init=K.if_then_else(valid0 != _i32(0), final_max0, _f32(_NEG_INF))
                )
                max1 = K.local_scalar(
                    "float32", init=K.if_then_else(valid1 != _i32(0), final_max1, _f32(_NEG_INF))
                )
                final_max = _max_f32(max0, max1)
                safe_max_c = K.local_scalar(
                    "float32",
                    init=K.if_then_else(final_max == _f32(_NEG_INF), _f32(0.0), final_max),
                )
                combine = []
                for instance, (max_i, valid_i) in enumerate(((max0, valid0), (max1, valid1))):
                    diff = K.local_scalar("float32")
                    K.ptx.sub.ftz.f32(diff, max_i, safe_max_c)
                    scaled = K.local_scalar("float32")
                    K.ptx.mul.ftz.f32(scaled, diff, softmax_scale_log2)
                    exp_i = K.local_scalar("float32")
                    K.ptx.ex2.approx.ftz.f32(exp_i, scaled)
                    combine.append(
                        K.local_scalar(
                            "float32", init=K.if_then_else(valid_i != _i32(0), exp_i, _f32(0.0))
                        )
                    )
                prod1 = K.local_scalar("float32")
                K.ptx.mul.ftz.f32(prod1, final_sum1, combine[1])
                final_sum = K.local_scalar("float32")
                K.ptx.fma.rn.ftz.f32(final_sum, final_sum0, combine[0], prod1)
                rcp_sum = K.local_scalar("float32")
                K.ptx.rcp.approx.ftz.f32(rcp_sum, final_sum)
                sum_positive = K.local_scalar("int32", init=K.cast(final_sum > _f32(0.0), "int32"))
                inv_sum = K.local_scalar(
                    "float32", init=K.if_then_else(sum_positive != _i32(0), rcp_sum, _f32(0.0))
                )
                output_scale0 = K.local_scalar("float32")
                K.ptx.mul.ftz.f32(output_scale0, combine[0], inv_sum)
                output_scale1 = K.local_scalar("float32")
                K.ptx.mul.ftz.f32(output_scale1, combine[1], inv_sum)
                query = query_base + my_row
                output_row = (query * num_q_heads + q_head) * 128
                with K.If(my_row < q_valid), K.Then():
                    scale0_pair = _pack2(output_scale0, output_scale0)
                    scale1_pair = _pack2(output_scale1, output_scale1)
                    one_pair = _pack2(_f32(1.0), _f32(1.0))
                    for chunk in range(16):
                        out_addr0 = taddr + _u32(_TMEM_OUTPUT[0] + 8 * chunk) + row_addr
                        _tmem_load_x8(o0, out_addr0)
                        _tmem_load_x8(o1, out_addr0 + _u32(128))
                        for j in range(4):
                            _packed_mul_inplace(o0, 2 * j, scale0_pair)
                        for j in range(4):
                            _packed_mul_inplace(o1, 2 * j, scale1_pair)
                        for j in range(8):
                            K.ptx.add.ftz.f32(o0[j], o0[j], o1[j])
                        # The source multiplies the merged fragment by a packed 1.0 with
                        # ``mul.rn.ftz.f32x2``; the ftz flush is part of the result.
                        for j in range(4):
                            _packed_mul_inplace(o0, 2 * j, one_pair)
                        for j in range(4):
                            K.ptx.cvt.rn.bf16x2.f32(words[j], o0[2 * j + 1], o0[2 * j])
                        K.ptx.st.global_.v4.b32(
                            out.ptr_to([output_row + 8 * chunk]),
                            words[0],
                            words[1],
                            words[2],
                            words[3],
                        )
                    stat_idx = query * num_q_heads + q_head
                    log2_sum = K.local_scalar("float32")
                    K.ptx.lg2.approx.ftz.f32(log2_sum, final_sum)
                    max_scaled = K.local_scalar("float32")
                    K.ptx.mul.ftz.f32(max_scaled, final_max, softmax_scale_log2)
                    log_sum = K.local_scalar("float32")
                    K.ptx.mul.ftz.f32(log_sum, log2_sum, _f32(_LN2_F32))
                    lse_value = K.local_scalar("float32")
                    K.ptx.fma.rn.ftz.f32(lse_value, max_scaled, _f32(_LN2_F32), log_sum)
                    final_lse = K.local_scalar(
                        "float32",
                        init=K.if_then_else(sum_positive != _i32(0), lse_value, _f32(_NEG_INF)),
                    )
                    with K.If(return_softmax_lse != _i32(0)), K.Then():
                        K.ptx.st.global_.b32(lse.ptr_to([stat_idx]), final_lse)
                    with K.If(return_temperature_lse != _i32(0)), K.Then():
                        K.ptx.st.global_.b32(temperature_lse.ptr_to([stat_idx]), final_lse)
                K.ptx.tcgen05.wait__ld.sync.aligned()
                K.ptx["tcgen05.fence::before_thread_sync"]()
                _mbar_arrive(bar(_MBAR_Q_EMPTY))
                _mbar_arrive(bar(_MBAR_TILE_DONE))
                K.assign(tile_idx_c, tile_idx_c + num_bids())

        # ---- Role: MMA issuer (warp 12) ----
        with r_mma:
            phase_union_ready_m = K.local_scalar("int32", init=_i32(0))
            phase_q_full = K.local_scalar("int32", init=_i32(0))
            phase_p_full = [K.local_scalar("int32", init=_i32(0)) for _ in range(2)]
            phase_tile_done = K.local_scalar("int32", init=_i32(0))
            desc_hi = _u32(_DESC_HI)
            ring = _ring_schedule(0)
            tile_idx_m = K.local_scalar("uint32", init=K.cast(bid, "uint32"))
            with K.While(tile_idx_m < total_tiles_u):
                _mbar_wait(bar(_MBAR_UNION_READY), phase_union_ready_m)
                _flip(phase_union_ready_m)
                _mbar_wait(bar(_MBAR_Q_FULL), phase_q_full)
                _flip(phase_q_full)
                first_pv = [K.local_scalar("int32", init=_i32(1)) for _ in range(2)]

                def wait_kv_full(push):
                    stage, parity = ring[push]
                    _mbar_wait(bar(_MBAR_KV_FULL + 8 * stage), _i32(parity))
                    return stage

                def qk_chain(instance, k_stage):
                    a_lo = K.local_scalar(
                        "uint32", init=K.uniform((bar(_SMEM_Q) >> 4) & _u32(0x3FFF))
                    )
                    b_lo = K.local_scalar(
                        "uint32",
                        init=K.uniform(
                            ((bar(_SMEM_KV) >> 4) & _u32(0x3FFF)) + _u32(k_stage * 2048)
                        ),
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
                            K.assign(a_lo, a_lo + _u32(_QK_STEPS[k16]))
                            K.assign(b_lo, b_lo + _u32(_QK_STEPS[k16]))
                    K.ptx[_TCGEN05_COMMIT](bar(_MBAR_S_FULL + 8 * instance), pred=leader)
                    K.ptx[_TCGEN05_COMMIT](bar(_MBAR_KV_EMPTY + 8 * k_stage), pred=leader)

                def pv_chain(instance, v_stage, last_pair):
                    _mbar_wait(bar(_MBAR_P_FULL + 8 * instance), phase_p_full[instance])
                    _flip(phase_p_full[instance])
                    b_lo = K.local_scalar(
                        "uint32",
                        init=K.uniform(
                            (((bar(_SMEM_KV) >> 4) & _u32(0x3FFF)) | _u32(_V_LBO_BIT))
                            + _u32(v_stage * 2048)
                        ),
                    )
                    leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                    d_tmem = taddr + _u32(_TMEM_OUTPUT[instance])
                    a_tmem = K.local_scalar(
                        "uint32", init=taddr + _u32(_TMEM_SCORES[instance] + _TMEM_P_OFFSET)
                    )
                    enable_first = K.local_scalar(
                        "uint32",
                        init=K.if_then_else(first_pv[instance] != _i32(0), _u32(0), _u32(1)),
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
                    if last_pair:
                        K.ptx[_TCGEN05_COMMIT](bar(_MBAR_O_FULL + 8 * instance), pred=leader)
                    K.ptx[_TCGEN05_COMMIT](bar(_MBAR_KV_EMPTY + 8 * v_stage), pred=leader)

                push = 0
                for instance in range(2):  # S5, S4
                    k_stage = wait_kv_full(push)
                    push += 1
                    qk_chain(instance, k_stage)
                for pair in range(3):
                    for instance in range(2):
                        v_stage = wait_kv_full(push)
                        push += 1
                        pv_chain(instance, v_stage, last_pair=(pair == 2))
                        if pair < 2:  # S for slot 5 - instance - 2*(pair + 1)
                            k_stage = wait_kv_full(push)
                            push += 1
                            qk_chain(instance, k_stage)
                _mbar_wait(bar(_MBAR_TILE_DONE), phase_tile_done)
                _flip(phase_tile_done)
                K.assign(tile_idx_m, tile_idx_m + num_bids())
            taddr_dealloc = K.local_scalar("uint32")
            K.ptx.ld.volatile.shared.b32(taddr_dealloc, bar(_SMEM_TMEM_MAILBOX))
            K.ptx[_TMEM_DEALLOC](taddr_dealloc, _u32(TMEM_COLS))

        # ---- Role: idle (warps 13, 14) ----
        with r_idle:
            pass

        # ---- Role: load warp (warp 15) ----
        with r_load:
            phase_q_empty = K.local_scalar("int32", init=_i32(1))
            ring_l = _ring_schedule(1)
            tile_idx_l = K.local_scalar("uint32", init=K.cast(bid, "uint32"))
            with K.While(tile_idx_l < total_tiles_u):
                _mbar_wait(bar(_MBAR_Q_EMPTY), phase_q_empty)
                _flip(phase_q_empty)
                q_head = K.local_scalar("int32", init=K.cast(tile_idx_l // mb_u, "int32"))
                q_tile = K.local_scalar("int32", init=K.cast(tile_idx_l, "int32") - q_head * mb)
                query_base = q_tile * 128
                kv_head = q_head
                load_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                with K.If(load_leader != _u32(0)), K.Then():
                    _mbar_expect_tx(bar(_MBAR_Q_FULL), 32768)
                    K.ptx[_TMA_G2S_4D](
                        bar(_SMEM_Q),
                        K.address_of(q_map),
                        _i32(0),
                        query_base,
                        q_head,
                        _i32(0),
                        bar(_MBAR_Q_FULL),
                    )
                with K.If(lane < _i32(6)), K.Then():
                    block_index = K.local_scalar("int32")
                    K.ptx.ld.global_.nc.b32(block_index, bsr_indices.ptr_to([q_tile * 6 + lane]))
                    K.ptx.st.shared.b32(
                        bar(_SMEM_UNION_BLOCKS) + K.cast(lane, "uint32") * _u32(4), block_index
                    )
                K.ptx.barrier.sync(_u32(8), _u32(32))
                K.ptx.fence.proxy.async_.shared__cta()
                _mbar_arrive(bar(_MBAR_UNION_READY))

                def push_tile(tensor_map, slot, push):
                    stage, parity = ring_l[push]
                    n_block = _ld_shared_i32(bar(_SMEM_UNION_BLOCKS + 4 * slot))
                    token_base = n_block * 128
                    _mbar_wait(bar(_MBAR_KV_EMPTY + 8 * stage), _i32(parity))
                    push_leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                    with K.If(push_leader != _u32(0)), K.Then():
                        full_bar = bar(_MBAR_KV_FULL + 8 * stage)
                        _mbar_expect_tx(full_bar, 32768)
                        stage_addr = bar(_SMEM_KV + stage * _SMEM_KV_STAGE_BYTES)
                        token1 = token_base + _i32(64)
                        for dim_half in range(2):
                            for token_half, token in ((0, token_base), (1, token1)):
                                K.ptx[_TMA_G2S_4D](
                                    stage_addr + _u32(8192 * token_half + 16384 * dim_half),
                                    K.address_of(tensor_map),
                                    _i32(0),
                                    token,
                                    _i32(dim_half),
                                    kv_head,
                                    full_bar,
                                )

                push = 0
                for instance in range(2):  # K5, K4
                    push_tile(k_map, 5 - instance, push)
                    push += 1
                for pair in range(3):
                    for instance in range(2):
                        slot = 5 - instance - 2 * pair
                        push_tile(v_map, slot, push)
                        push += 1
                        if slot - 2 >= 0:
                            push_tile(k_map, slot - 2, push)
                            push += 1
                K.assign(tile_idx_l, tile_idx_l + num_bids())

    return cake_vsa_ultrasparse_bsr_sm100


@lru_cache(maxsize=1)
def _kernel():
    return _build_kernel()


def get_kernel(**config: Any):
    """Return the TIRx PrimFunc; the kernel is shape-generic at runtime."""
    if config:
        cfg = _without_label(config)
        _validate_shape(cfg["M"], cfg["N"])
    return _kernel().func


# ptxas ``--register-usage-level`` for this kernel.  The upstream cubin is built
# with ptxas' native default (5).  TIRx's nvcc path defaults to 10, which here
# schedules the 48-register MMA/load warps so aggressively that descriptor packs
# and uniform values spill (stack 336 B, 120 LDL / 99 STL, 268 uniform spill and
# fill pairs versus the source's 96 B / 48 / 34 / 23); level 5 reproduces the
# source's allocation (88 B, 30 LDL / 27 STL, 23 pairs) and measured 6.7-7.2%
# faster than level 10 on the two streaming benchmark shapes.
_PTXAS_REG_LEVEL = "5"


@lru_cache(maxsize=1)
def _compiled_kernel():
    import os

    from tirx_kernels.runner import compile_kernel

    # nvcc mode keeps the TIRx build on the CUDA toolkit on PATH (13.2, PTX ISA
    # 9.2) -- the same toolchain the upstream ``nvcc -cubin`` build uses -- so
    # both binaries share one PTX ISA for SASS and profiler comparisons.
    previous = os.environ.get("TVM_CUDA_PTXAS_REG_LEVEL")
    os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = _PTXAS_REG_LEVEL
    try:
        return compile_kernel(get_kernel(), cuda_compile_mode="nvcc")
    finally:
        if previous is None:
            del os.environ["TVM_CUDA_PTXAS_REG_LEVEL"]
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
        data["bsr_indices"].view(-1),
        data["mb"],
        data["nb"],
        SELECTED_BLOCKS,
        data["total_tiles"],
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


def _source_args(data: dict[str, Any]) -> tuple[Any, ...]:
    """The ``flashinfer.cake_vsa._run_standard`` argument list for ``ultrasparse_bsr``."""
    buffers = data["source"]
    return (
        data["q"],
        data["k"],
        data["v"],
        buffers["out"],
        buffers["lse"],
        buffers["tlse"],
        data["bsr_indices"],
        data["mb"],
        data["nb"],
        SELECTED_BLOCKS,
        data["total_tiles"],
        data["num_heads"],
        data["num_heads"],
        data["softmax_scale_log2"],
        data["lse_temperature_scale"],
        int(data["return_lse"]),
        int(data["return_temperature_lse"]),
        data["grid_x"],
        1,
        1,
    )


def _source_launch(data: dict[str, Any]):
    import tvm_ffi

    module = _source_module()
    arguments = _source_args(data)

    def launch():
        with tvm_ffi.use_torch_stream():
            module.run(*arguments)

    launch._keep_alive = arguments
    return launch


def _dense_block_mask(data: dict[str, Any]):
    """``(num_heads, mb, nb)`` boolean mask equivalent to the shared BSR table."""
    import torch

    mb, nb = data["mb"], data["nb"]
    shared = torch.zeros((mb, nb), dtype=torch.bool, device=data["q"].device)
    shared.scatter_(1, data["bsr_indices"].to(torch.int64), True)
    return shared.unsqueeze(0).expand(data["num_heads"], -1, -1).contiguous()


def _run_public_api(data: dict[str, Any], mode: str):
    """Run the upstream public API (``plan_cake_vsa`` + ``run_cake_vsa``) without LSE.

    ``mode == "block_mask"`` hands the dense mask over (the planner recompacts it
    into sorted BSR columns); ``mode == "indices"`` hands ``indptr``/``indices``
    over, which the planner trusts and forwards in caller order.
    """
    import torch
    from flashinfer import cake_vsa

    mb = data["mb"]
    if mode == "indices":
        indptr = torch.arange(0, (mb + 1) * SELECTED_BLOCKS, SELECTED_BLOCKS, dtype=torch.int32)
        plan_args = (indptr.to(data["q"].device), data["bsr_indices"].view(-1), None)
    elif mode == "block_mask":
        plan_args = (None, None, _dense_block_mask(data))
    else:
        raise ValueError(f"unknown api_check mode {mode!r}")
    plan = cake_vsa.plan_cake_vsa(
        *plan_args,
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
    if plan["indices"] is None or plan["max_selected_blocks"] != SELECTED_BLOCKS:
        raise AssertionError("public API plan does not describe a six-block shared BSR pattern")
    if not plan["uniform_selected_blocks"] or plan["mb"] < MIN_QUERY_BLOCKS:
        raise AssertionError("public API plan would not dispatch to ultrasparse_bsr")
    if not torch.equal(plan["indices"].view(mb, SELECTED_BLOCKS), data["bsr_indices"]):
        raise AssertionError(
            "public API plan reordered the BSR columns relative to the direct launch"
        )
    return cake_vsa.run_cake_vsa(
        plan, data["q"], data["k"], data["v"], out=None, lse=None, return_lse=False, backend="cake"
    )


# ---------------------------------------------------------------------------
# fp64 oracle and validation
# ---------------------------------------------------------------------------
def _bf16_order_key(values):
    """Map bf16 bit patterns to integers ordered like the reals (for ULP distances)."""
    import torch

    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(bits < 0x8000, bits + 0x8000, 0x8000 - (bits & 0x7FFF))


def _assert_bitwise(name: str, ours, theirs) -> None:
    import torch

    if ours.dtype == torch.bfloat16:
        ours_bits, theirs_bits = ours.view(torch.int16), theirs.view(torch.int16)
    else:
        ours_bits, theirs_bits = ours.view(torch.int32), theirs.view(torch.int32)
    mismatch = ours_bits != theirs_bits
    count = int(mismatch.sum())
    if count:
        index = mismatch.reshape(-1).nonzero()[0].item()
        raise AssertionError(
            f"{name}: {count} of {mismatch.numel()} elements differ bitwise from the source "
            f"kernel (first at flat index {index}: tirx={ours.reshape(-1)[index].item()!r} "
            f"source={theirs.reshape(-1)[index].item()!r})"
        )


class _OracleStats:
    """Running error statistics of one implementation against the fp64 oracle."""

    def __init__(self, cfg: dict[str, Any], *, return_lse: bool, return_tlse: bool):
        self.cfg = cfg
        self.return_lse = return_lse
        self.return_tlse = return_tlse
        self.out_max_abs_err = 0.0
        self.out_max_rel_excess = 0.0
        self.out_max_err_small_ref = 0.0
        self.out_max_rel_err_large_ref = 0.0
        self.ulp_max = 0.0
        self.ulp_total = 0
        self.ulp_le1 = 0
        self.ulp_le2 = 0
        self.lse_max_abs_err = 0.0
        self.lse_max_rel_excess = 0.0
        self.tlse_max_abs_err = 0.0
        self.tlse_max_rel_excess = 0.0

    def update(self, out_bf16, lse, tlse, ref_out, ref_lse):
        import torch

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
            ulps = (_bf16_order_key(ref_out.to(torch.bfloat16)) - _bf16_order_key(out_bf16)).abs()[
                sizeable
            ]
            self.ulp_max = max(self.ulp_max, float(ulps.max()))
            self.ulp_total += ulps.numel()
            self.ulp_le1 += int((ulps <= 1).sum())
            self.ulp_le2 += int((ulps <= 2).sum())
        for flag, values, attr in ((self.return_lse, lse, "lse"), (self.return_tlse, tlse, "tlse")):
            if not flag:
                continue
            err = (values.double() - ref_lse).abs()
            bound = cfg["oracle_lse_atol"] + cfg["oracle_lse_rtol"] * ref_lse.abs()
            setattr(
                self,
                f"{attr}_max_abs_err",
                max(getattr(self, f"{attr}_max_abs_err"), float(err.max())),
            )
            setattr(
                self,
                f"{attr}_max_rel_excess",
                max(getattr(self, f"{attr}_max_rel_excess"), float((err / bound).max())),
            )

    def check(self, name: str, stats: dict[str, float]) -> None:
        cfg = self.cfg
        stats[f"{name}_out_max_abs_err"] = self.out_max_abs_err
        stats[f"{name}_out_max_rel_excess"] = self.out_max_rel_excess
        stats[f"{name}_out_max_err_small_ref"] = self.out_max_err_small_ref
        stats[f"{name}_out_max_rel_err_large_ref"] = self.out_max_rel_err_large_ref
        if self.out_max_rel_excess > 1.0:
            raise AssertionError(
                f"{name}: out mismatch vs fp64 oracle (max abs err {self.out_max_abs_err:.3e}, "
                f"max err/bound {self.out_max_rel_excess:.3f})"
            )
        if self.ulp_total:
            le1 = self.ulp_le1 / self.ulp_total
            le2 = self.ulp_le2 / self.ulp_total
            stats[f"{name}_out_ulp_max"] = self.ulp_max
            stats[f"{name}_out_ulp_le1_frac"] = le1
            stats[f"{name}_out_ulp_le2_frac"] = le2
            if (
                self.ulp_max > cfg["ulp_max"]
                or le1 < cfg["ulp_le1_frac"]
                or le2 < cfg["ulp_le2_frac"]
            ):
                raise AssertionError(
                    f"{name}: bf16 ULP distribution vs oracle too wide (max {self.ulp_max:.0f}, "
                    f"<=1 ULP {le1:.5f}, <=2 ULP {le2:.5f})"
                )
        for flag, attr in ((self.return_lse, "lse"), (self.return_tlse, "tlse")):
            if not flag:
                continue
            stats[f"{name}_{attr}_max_abs_err"] = getattr(self, f"{attr}_max_abs_err")
            stats[f"{name}_{attr}_max_rel_excess"] = getattr(self, f"{attr}_max_rel_excess")
            if getattr(self, f"{attr}_max_rel_excess") > 1.0:
                raise AssertionError(
                    f"{name}: {attr} mismatch vs oracle (max abs err "
                    f"{getattr(self, f'{attr}_max_abs_err'):.3e})"
                )


def _oracle_chunk(data: dict[str, Any], t0: int, t1: int):
    """fp64 block-gather reference for query tiles ``[t0, t1)``.

    Returns ``(out, lse)`` shaped ``(T, num_heads, 128, 128)`` and ``(T, num_heads, 128)``.
    """
    import torch

    nb, num_heads = data["nb"], data["num_heads"]
    idx = data["bsr_indices"][t0:t1].to(torch.int64)  # (T, 6)
    T = t1 - t0
    q = data["q"][t0 * BLOCK : t1 * BLOCK].double().view(T, BLOCK, num_heads, HEAD_DIM)
    q = q.permute(0, 2, 1, 3)  # (T, H, 128, D)
    k_blocks = data["k"].view(nb, BLOCK, num_heads, HEAD_DIM)[idx]  # (T, 6, 128, H, D)
    v_blocks = data["v"].view(nb, BLOCK, num_heads, HEAD_DIM)[idx]
    k_g = k_blocks.permute(0, 3, 1, 2, 4).reshape(T, num_heads, SELECTED_BLOCKS * BLOCK, HEAD_DIM)
    v_g = v_blocks.permute(0, 3, 1, 2, 4).reshape(T, num_heads, SELECTED_BLOCKS * BLOCK, HEAD_DIM)
    scores = torch.matmul(q, k_g.double().transpose(-1, -2)) * data["sm_scale"]
    row_max = scores.amax(dim=-1, keepdim=True)
    probs = torch.exp(scores - row_max)
    sums = probs.sum(dim=-1, keepdim=True)
    out = torch.matmul(probs / sums, v_g.double())
    lse = (row_max + torch.log(sums)).squeeze(-1)
    return out, lse


def _validate_outputs(data: dict[str, Any], *, with_oracle: bool) -> dict[str, float]:
    import torch

    tirx, source = data["tirx"], data["source"]
    return_lse, return_tlse = data["return_lse"], data["return_temperature_lse"]

    for name, buffers in (("tirx", tirx), ("source", source)):
        if not torch.isfinite(buffers["out"].float()).all():
            raise AssertionError(f"{name}: non-finite values in out")
        if return_lse and not torch.isfinite(buffers["lse"]).all():
            raise AssertionError(f"{name}: non-finite values in lse")
        if return_tlse and not torch.isfinite(buffers["tlse"]).all():
            raise AssertionError(f"{name}: non-finite values in temperature lse")
        if not return_lse and not bool((buffers["lse"] == _LSE_SENTINEL).all()):
            raise AssertionError(f"{name}: lse sentinel was overwritten with return_lse=0")
        if (
            return_lse
            and return_tlse
            and not torch.equal(buffers["lse"].view(torch.int32), buffers["tlse"].view(torch.int32))
        ):
            raise AssertionError(f"{name}: temperature_lse differs from lse")

    _assert_bitwise("out", tirx["out"], source["out"])
    _assert_bitwise("lse", tirx["lse"], source["lse"])
    if return_tlse:
        _assert_bitwise("temperature_lse", tirx["tlse"], source["tlse"])

    stats: dict[str, float] = {}
    if not with_oracle:
        return stats
    cfg = data["config"]
    mb, num_heads = data["mb"], data["num_heads"]
    trackers = {
        name: _OracleStats(cfg, return_lse=return_lse, return_tlse=return_tlse)
        for name in ("tirx", "source")
    }
    for t0 in range(0, mb, _ORACLE_TILES_PER_CHUNK):
        t1 = min(mb, t0 + _ORACLE_TILES_PER_CHUNK)
        ref_out, ref_lse = _oracle_chunk(data, t0, t1)
        T = t1 - t0
        for name, buffers in (("tirx", tirx), ("source", source)):
            out = buffers["out"][t0 * BLOCK : t1 * BLOCK].view(T, BLOCK, num_heads, HEAD_DIM)
            out = out.permute(0, 2, 1, 3)
            lse = (
                buffers["lse"][t0 * BLOCK : t1 * BLOCK].view(T, BLOCK, num_heads).permute(0, 2, 1)
                if return_lse
                else None
            )
            tlse = (
                buffers["tlse"][t0 * BLOCK : t1 * BLOCK].view(T, BLOCK, num_heads).permute(0, 2, 1)
                if return_tlse
                else None
            )
            trackers[name].update(out, lse, tlse, ref_out, ref_lse)
    for name, tracker in trackers.items():
        tracker.check(name, stats)
    return stats


def _skip_unless_supported() -> None:
    from unittest import SkipTest

    import torch

    if not torch.cuda.is_available():
        raise SkipTest("CUDA device required")
    if torch.cuda.get_device_capability() != (10, 0):
        raise SkipTest("cake_vsa ultrasparse_bsr sm_100a requires compute capability 10.0")


def run_test(**config: Any) -> dict[str, float]:
    """Compile, launch TIRx and the upstream kernel, and validate one config."""
    import torch

    _skip_unless_supported()
    data = prepare_data(**config)
    executable = _compiled_kernel()
    tirx_launch = _tirx_launch(executable, data)
    tirx_launch()
    torch.cuda.synchronize()
    first_out = data["tirx"]["out"].clone()
    first_lse = data["tirx"]["lse"].clone()

    source_launch = _source_launch(data)
    source_launch()
    torch.cuda.synchronize()
    stats = _validate_outputs(data, with_oracle=True)

    # A second TIRx launch into the same buffers must reproduce the first bit for bit.
    tirx_launch()
    torch.cuda.synchronize()
    _assert_bitwise("out (repeat launch)", data["tirx"]["out"], first_out)
    _assert_bitwise("lse (repeat launch)", data["tirx"]["lse"], first_lse)

    api_mode = data["config"]["api_check"]
    if api_mode is not None:
        # The public API must dispatch this pattern to ultrasparse_bsr and agree bitwise
        # on the output (the LSE flags do not influence ``out``).
        api_out = _run_public_api(data, api_mode)
        _assert_bitwise("out (public API)", data["tirx"]["out"], api_out)
    return stats


# ---------------------------------------------------------------------------
# Benchmark entry points
# ---------------------------------------------------------------------------
def prepare_bench(**config: Any):
    from tirx_kernels.runner import prepared_gpu_benchmark

    kernel_config = _without_label(config)
    _validate_shape(kernel_config["M"], kernel_config["N"])
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
