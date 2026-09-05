# This file is a TIRx port of code from MSA
# (https://github.com/MiniMax-AI/MSA @ 80434d7f), Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""MSA sparse-attention flat-schedule preparation.

Ports ``SparseAttentionPrepareFlatScheduleSm100`` -- the kernel that turns the
uneven CSR k2q row fanout into the compact worklist the SM100 sparse-attention
forward kernels consume.  One warp owns one ``(csr row, head_kv)`` pair: lane 0
reads the row's CSR bounds, decodes the row's ``(batch, kv_block)`` coordinate,
and splits the row into ``ceil(row_count / target_q_per_cta)`` chunks; the warp's
lanes then stride over those chunks, reserving each work slot with a relaxed
device-scope atomic and writing the six-field work item.

Row order is level-major: ``row_linear`` enumerates kv-block level 0 of every
batch, then level 1 of every batch that has one, and so on, which is what the
kernel's binary search over ``cu_seqlens_k`` inverts.

Work items land in atomic-arrival order, so the output is defined as a multiset
of rows plus the final ``work_count``, not as a fixed row order.

Upstream source: python/fmha_sm100/cute/src/sm100/prepare_scheduler.py:124.
"""

from typing import Any

import tirx_kernels.kern as K
from tirx_kernels.msa.utils._scalar_ops import (
    atom_add_global_i32,
    ld_global_i32,
    shfl_idx_i32,
    st_global_i32,
    uceil_div_i32,
    udiv_i32,
)

KERNEL_META = {
    "name": "msa_sparse_prepare_flat_schedule_sm100",
    "category": "msa",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "reference_requirements": (
        {
            "package": "msa",
            "git": {
                "url": "https://github.com/MiniMax-AI/MSA.git",
                "commit": "80434d7f67877c6570ca19cac444b84bc9855dac",
            },
            "import": "fmha_sm100",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.5.3", "import": "cutlass"},
        {"package": "quack-kernels", "specifier": "==0.5.0", "import": "quack"},
    ),
}

# `__init__(num_threads=128)` (:127-135); the kernel takes no shared memory.
NUM_THREADS = 128
WARPS_PER_CTA = NUM_THREADS // 32
LAUNCH_TAGS = ("blockIdx.x", "threadIdx.x")

# `_emit_work` writes six int32 fields per work item (:151-156).
WORK_FIELDS = 6


# ---------------------------------------------------------------------------
# Host-side schedule sizing -- `SparseAttentionScheduleModel` (:48-118).
#
# These stay on the host in the source too: the sizes they produce are kernel
# arguments, not kernel work.  The SM count comes from the runner's prepare-safe
# probe rather than `torch.cuda.get_device_properties`, so sizing is identical
# whether or not CUDA is up when a config is materialized.
# ---------------------------------------------------------------------------
def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _round_up(x: int, y: int) -> int:
    return _ceil_div(x, y) * y


def target_q_per_cta(
    *,
    total_q: int,
    topk: int,
    blk_kv: int,
    head_kv: int,
    qhead_per_kv: int,
    num_sms: int,
    usable_SM_count: int = -1,
) -> int:
    """`balanced_target_q_per_cta` (:82-104), including `_target_q_per_cta` (:59-80)."""
    if usable_SM_count > 0:
        num_sms = min(int(usable_SM_count), num_sms)
    q_tokens_per_group = 128 // qhead_per_kv
    total_refs_upper = total_q * topk * head_kv
    desired_work_items = max(num_sms * 2, 1)
    total_groups_upper = _ceil_div(max(total_refs_upper, 1), q_tokens_per_group)
    target_groups_per_cta = min(512, max(1, _ceil_div(total_groups_upper, desired_work_items)))
    occupancy_target = target_groups_per_cta * q_tokens_per_group

    sink_balance_cap = max(q_tokens_per_group, int(topk) * int(blk_kv) * 2)
    target = min(max(occupancy_target, q_tokens_per_group), sink_balance_cap)
    return _round_up(target, q_tokens_per_group)


def flat_schedule_capacity(
    *, total_rows: int, total_q: int, topk: int, head_kv: int, target: int
) -> int:
    """`flat_schedule_capacity` (:106-118) -- the `scheduler_metadata` row count."""
    row_upper = max(total_rows, 0) * max(head_kv, 1)
    refs_upper = max(total_q, 0) * max(topk, 1) * max(head_kv, 1)
    split_upper = _ceil_div(max(refs_upper, 1), max(target, 1))
    return max(1, row_upper + split_upper)


# ---------------------------------------------------------------------------
# Scalar memory and cross-lane primitives.
#
# Shared with the other MSA prepare kernel; see `utils/_scalar_ops.py` for the
# instruction each one emits and why global memory is reached only through
# `K.ptx.*` on `ptr_to`.
#
# Call-site map for this kernel: the scalar loads are the CSR bounds (:307-308)
# and the `cu_seqlens_k` scans (:165); the store is one work-item field
# (:151-156); the atomic is the slot reservation (:326-331), whose explicit
# `sem="relaxed", scope="gpu"` are PTX's defaults and so leave no qualifier in
# the export; the shuffle is the warp broadcast (:319-322).
# ---------------------------------------------------------------------------
_ld_global_i32 = ld_global_i32
_st_global_i32 = st_global_i32
_atom_add_global_i32 = atom_add_global_i32
_udiv_i32 = udiv_i32
_uceil_div_i32 = uceil_div_i32
_shfl_idx_i32 = shfl_idx_i32


# ---------------------------------------------------------------------------
# Target entry.
# ---------------------------------------------------------------------------
@K.kernel(
    warps=WARPS_PER_CTA,
    arch="sm_100a",
    grid=lambda p: K.ceildiv(p["total_rows"] * p["num_heads_kv"], WARPS_PER_CTA),
)
def _kernel(
    k2q_row_ptr: K.gptr(K.i32, shape=lambda p: (p["num_heads_kv"] * (p["total_rows"] + 1),)),
    cu_seqlens_k: K.gptr(K.i32, shape=lambda p: (p["num_batches"] + 1,)),
    scheduler_metadata: K.gptr(K.i32, shape=lambda p: (p["work_capacity"] * WORK_FIELDS,)),
    work_count: K.gptr[K.i32, (1,)],
    total_rows: K.i32,
    num_batches: K.i32,
    target: K.i32,
    work_capacity: K.i32,
    num_heads_kv: K.i32,
    blk_kv: K.i32,
):
    # CUDA TRANSCRIPTION START
    # sketch: static ABI/launch, one warp per (row, head) -> :290-295.
    block = K.cta_id()
    tidx = K.thread_id()
    lane = K.local_scalar(K.i32, init=tidx % 32, name="lane")
    warp = K.local_scalar(K.i32, init=tidx // 32, name="warp")
    row_head_idx = K.local_scalar(K.i32, init=block * WARPS_PER_CTA + warp, name="row_head_idx")
    total_row_heads = K.local_scalar(K.i32, init=total_rows * num_heads_kv, name="total_row_heads")

    # sketch: the six scalars the warp publishes, zeroed BEFORE the grid-tail
    # guard so an empty tail warp still broadcasts defined values -> :297-302.
    row_count = K.local_scalar(K.i32, name="row_count")
    num_chunks = K.local_scalar(K.i32, name="num_chunks")
    batch_idx = K.local_scalar(K.i32, name="batch_idx")
    kv_block_idx = K.local_scalar(K.i32, name="kv_block_idx")
    head_kv_idx = K.local_scalar(K.i32, name="head_kv_idx")
    row_linear = K.local_scalar(K.i32, name="row_linear")
    K.assign(row_count, K.int32(0))
    K.assign(num_chunks, K.int32(0))
    K.assign(batch_idx, K.int32(0))
    K.assign(kv_block_idx, K.int32(0))
    K.assign(head_kv_idx, K.int32(0))
    K.assign(row_linear, K.int32(0))

    # sketch: grid tail, then the row/head split -> :303-305.
    with K.If(row_head_idx < total_row_heads), K.Then():
        K.assign(row_linear, _udiv_i32(row_head_idx, num_heads_kv))
        K.assign(head_kv_idx, row_head_idx - row_linear * num_heads_kv)

        # sketch: lane 0 owns the whole decode -> :306-318.
        with K.If(lane == 0), K.Then():
            row_base = K.local_scalar(
                K.i32, init=head_kv_idx * (total_rows + 1) + row_linear, name="row_base"
            )
            row_start = K.local_scalar(
                K.i32, init=_ld_global_i32(k2q_row_ptr, row_base), name="row_start"
            )
            row_end = K.local_scalar(
                K.i32, init=_ld_global_i32(k2q_row_ptr, row_base + 1), name="row_end"
            )
            K.assign(row_count, row_end - row_start)

            prev = K.local_scalar(K.i32, name="prev")
            # The source pins `unroll=1` on every batch scan (:177, :190, :222)
            # and its export carries `.pragma "nounroll"` on each.  A TIRx `For`
            # emits no unroll pragma, so nvcc unrolls these scans fourfold and
            # triples the decode's load and divide traffic; a `While` lowers with
            # `#pragma unroll 1`, which is the shape the source has.  Hence the
            # explicit counters.
            batch_cursor = K.local_scalar(K.i32, name="batch_cursor")

            # sketch 2a: _max_rows_per_batch -> :183-193.  The base element is
            # hoisted and rotated forward, so the body holds one load.
            max_rows = K.local_scalar(K.i32, init=K.int32(0), name="max_rows")
            K.assign(prev, _ld_global_i32(cu_seqlens_k, 0))
            K.assign(batch_cursor, K.int32(0))
            with K.While(batch_cursor < num_batches):
                next_seq = K.local_scalar(
                    K.i32, init=_ld_global_i32(cu_seqlens_k, batch_cursor + 1), name="next_seq"
                )
                K.assign(max_rows, K.max(max_rows, _uceil_div_i32(next_seq - prev, blk_kv)))
                K.assign(prev, next_seq)
                K.assign(batch_cursor, batch_cursor + 1)

            # sketch 2b: binary search over levels -> :202-215.  `probe_base` is
            # hoisted above the search, and each probe re-seeds `prev` from it.
            lo = K.local_scalar(K.i32, name="lo")
            hi = K.local_scalar(K.i32, name="hi")
            K.assign(lo, K.int32(0))
            K.assign(hi, max_rows)
            probe_base = K.local_scalar(
                K.i32, init=_ld_global_i32(cu_seqlens_k, 0), name="probe_base"
            )
            rows_before_next = K.local_scalar(K.i32, name="rows_before_next")
            with K.While(lo < hi):
                mid = K.local_scalar(K.i32, init=_udiv_i32(lo + hi, 2), name="mid")
                K.assign(rows_before_next, K.int32(0))
                K.assign(prev, probe_base)
                K.assign(batch_cursor, K.int32(0))
                with K.While(batch_cursor < num_batches):
                    probe_seq = K.local_scalar(
                        K.i32, init=_ld_global_i32(cu_seqlens_k, batch_cursor + 1), name="probe_seq"
                    )
                    K.assign(
                        rows_before_next,
                        rows_before_next + K.min(_uceil_div_i32(probe_seq - prev, blk_kv), mid + 1),
                    )
                    K.assign(prev, probe_seq)
                    K.assign(batch_cursor, batch_cursor + 1)
                with K.If(rows_before_next <= row_linear):
                    with K.Then():
                        K.assign(lo, mid + 1)
                    with K.Else():
                        K.assign(hi, mid)
            level = K.local_scalar(K.i32, init=lo, name="level")

            # sketch 2c: offset inside the level band -> :217.
            rows_before = K.local_scalar(K.i32, init=K.int32(0), name="rows_before")
            K.assign(prev, _ld_global_i32(cu_seqlens_k, 0))
            K.assign(batch_cursor, K.int32(0))
            with K.While(batch_cursor < num_batches):
                before_seq = K.local_scalar(
                    K.i32, init=_ld_global_i32(cu_seqlens_k, batch_cursor + 1), name="before_seq"
                )
                K.assign(
                    rows_before,
                    rows_before + K.min(_uceil_div_i32(before_seq - prev, blk_kv), level),
                )
                K.assign(prev, before_seq)
                K.assign(batch_cursor, batch_cursor + 1)
            offset = K.local_scalar(K.i32, init=row_linear - rows_before, name="offset")

            # sketch 2d: the offset-th batch above `level` -> :218-229.  The scan
            # runs to `num_batches` even after the batch is found, and its two
            # loads stay in the body because the predicate breaks the rotation.
            active_idx = K.local_scalar(K.i32, name="active_idx")
            found = K.local_scalar(K.i32, name="found")
            K.assign(active_idx, K.int32(0))
            K.assign(found, K.int32(0))
            K.assign(batch_cursor, K.int32(0))
            with K.While(batch_cursor < num_batches):
                with K.If(found == 0), K.Then():
                    scan_next = K.local_scalar(
                        K.i32, init=_ld_global_i32(cu_seqlens_k, batch_cursor + 1), name="scan_next"
                    )
                    scan_prev = K.local_scalar(
                        K.i32, init=_ld_global_i32(cu_seqlens_k, batch_cursor), name="scan_prev"
                    )
                    scan_rows = K.local_scalar(
                        K.i32, init=_uceil_div_i32(scan_next - scan_prev, blk_kv), name="scan_rows"
                    )
                    with K.If(scan_rows > level), K.Then():
                        with K.If(active_idx == offset), K.Then():
                            K.assign(batch_idx, batch_cursor)
                            K.assign(found, K.int32(1))
                        K.assign(active_idx, active_idx + 1)
                K.assign(batch_cursor, batch_cursor + 1)
            K.assign(kv_block_idx, level)

            # sketch 2e: a row with no references emits nothing -> :315-318.
            with K.If(row_count > 0), K.Then():
                K.assign(num_chunks, _uceil_div_i32(row_count, target))

    # sketch: publish lane 0's four scalars to the warp -> :319-322.  Executed by
    # every warp, including the ones the grid-tail guard emptied.
    K.assign(row_count, _shfl_idx_i32(row_count, 0))
    K.assign(num_chunks, _shfl_idx_i32(num_chunks, 0))
    K.assign(batch_idx, _shfl_idx_i32(batch_idx, 0))
    K.assign(kv_block_idx, _shfl_idx_i32(kv_block_idx, 0))

    # sketch: lane-strided chunk emission -> :324-345.
    chunk_idx = K.local_scalar(K.i32, init=lane, name="chunk_idx")
    with K.While(chunk_idx < num_chunks):
        work_idx = K.local_scalar(
            K.i32, init=_atom_add_global_i32(work_count, 0, K.uint32(1)), name="work_idx"
        )
        q_begin = K.local_scalar(K.i32, init=chunk_idx * target, name="q_begin")
        q_count = K.local_scalar(K.i32, init=K.min(target, row_count - q_begin), name="q_count")
        with K.If(work_idx < work_capacity), K.Then():
            work_base = K.local_scalar(K.i32, init=work_idx * WORK_FIELDS, name="work_base")
            _st_global_i32(scheduler_metadata, work_base, head_kv_idx)
            _st_global_i32(scheduler_metadata, work_base + 1, row_linear)
            _st_global_i32(scheduler_metadata, work_base + 2, q_begin)
            _st_global_i32(scheduler_metadata, work_base + 3, q_count)
            _st_global_i32(scheduler_metadata, work_base + 4, batch_idx)
            _st_global_i32(scheduler_metadata, work_base + 5, kv_block_idx)
        K.assign(chunk_idx, chunk_idx + 32)


def get_kernel(**config):
    """Return the TIRx specialization of `SparseAttentionPrepareFlatScheduleSm100`.

    Nothing about a config reaches the kernel as a compile-time constant: the
    source compiles one binary for every shape (:536-538), so the shape and
    schedule scalars stay runtime arguments here too.
    """
    config.pop("label", None)
    return _kernel.func.with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))


# ---------------------------------------------------------------------------
# Config matrix.
# ---------------------------------------------------------------------------
def _case(
    *,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    head_kv: int,
    qhead_per_kv: int,
    topk: int,
    blk_kv: int = 128,
    seqlen_pattern: str = "uniform",
    label: str,
) -> dict:
    return {
        "label": label,
        "batch": batch,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "head_kv": head_kv,
        "qhead_per_kv": qhead_per_kv,
        "topk": topk,
        "blk_kv": blk_kv,
        "seqlen_pattern": seqlen_pattern,
    }


# Shapes follow MSA's own coverage: the sparse-attention tests parametrize
# batch 1/3, head_kv 2, qhead_per_kv 1-16 and topk 4-32 over 1k-8k sequences
# (test_sparse_atten.py:1426-1490), and the benchmark sweeps prefill
# 8k-128k at batch 1 and decode batch 32/64/128 with q_len 8.
BENCH_CONFIGS = [
    _case(
        label="prefill_b1_k8192_h2",
        batch=1,
        seqlen_q=8192,
        seqlen_k=8192,
        head_kv=2,
        qhead_per_kv=8,
        topk=16,
    ),
    _case(
        label="prefill_b1_k32768_h2",
        batch=1,
        seqlen_q=32768,
        seqlen_k=32768,
        head_kv=2,
        qhead_per_kv=8,
        topk=16,
    ),
    _case(
        label="prefill_b1_k131072_h1",
        batch=1,
        seqlen_q=131072,
        seqlen_k=131072,
        head_kv=1,
        qhead_per_kv=16,
        topk=32,
    ),
    _case(
        label="prefill_b3_k8192_h2_varlen",
        batch=3,
        seqlen_q=8192,
        seqlen_k=8192,
        head_kv=2,
        qhead_per_kv=4,
        topk=16,
        seqlen_pattern="varlen",
    ),
    _case(
        label="decode_b32_k8192_h4",
        batch=32,
        seqlen_q=8,
        seqlen_k=8192,
        head_kv=4,
        qhead_per_kv=16,
        topk=32,
    ),
    _case(
        label="decode_b64_k16384_h4_varlen",
        batch=64,
        seqlen_q=8,
        seqlen_k=16384,
        head_kv=4,
        qhead_per_kv=16,
        topk=32,
        seqlen_pattern="varlen",
    ),
    _case(
        label="decode_b128_k65536_h4",
        batch=128,
        seqlen_q=8,
        seqlen_k=65536,
        head_kv=4,
        qhead_per_kv=16,
        topk=32,
    ),
    _case(
        label="edge_b1_k512_h1",
        batch=1,
        seqlen_q=256,
        seqlen_k=512,
        head_kv=1,
        qhead_per_kv=1,
        topk=4,
    ),
]

# Correctness adds the shapes that only change decode/tail behavior, plus the two
# that drive branches the benchmark matrix never reaches: every benchmark shape
# has `total_rows * head_kv` divisible by the four warps of a CTA, so no warp
# ever fails the grid-tail guard, and none splits a row past the 32-lane stride,
# so the chunk loop never takes its back edge.
CONFIGS = [
    *BENCH_CONFIGS,
    _case(
        # 5 row-heads: two CTAs, the second with one active warp and three that
        # fall out on `row_head_idx < total_row_heads` (:303).
        label="tail_b1_k640_h1",
        batch=1,
        seqlen_q=640,
        seqlen_k=640,
        head_kv=1,
        qhead_per_kv=1,
        topk=2,
    ),
    _case(
        # `target_q_per_cta` bottoms out at 128 against a sink row holding
        # thousands of references, so the hottest row splits into ~56 chunks and
        # every lane re-enters the emission loop (`chunk_idx += 32`, :345).
        label="split_b1_k8192_t4",
        batch=1,
        seqlen_q=8192,
        seqlen_k=8192,
        head_kv=1,
        qhead_per_kv=1,
        topk=4,
    ),
    _case(
        label="tiny_b2_k384_uneven",
        batch=2,
        seqlen_q=128,
        seqlen_k=384,
        head_kv=2,
        qhead_per_kv=2,
        topk=2,
        seqlen_pattern="varlen",
    ),
    _case(
        label="sparse_b4_k2048_h1_topk1",
        batch=4,
        seqlen_q=64,
        seqlen_k=2048,
        head_kv=1,
        qhead_per_kv=1,
        topk=1,
        seqlen_pattern="varlen",
    ),
]


# ---------------------------------------------------------------------------
# Data preparation.
# ---------------------------------------------------------------------------
def _seqlens_k(batch: int, seqlen_k: int, blk_kv: int, pattern: str) -> list[int]:
    """Per-batch K lengths; `varlen` makes each batch a different row count."""
    if pattern == "uniform":
        return [seqlen_k] * batch
    if pattern != "varlen":
        raise ValueError(f"unknown seqlen_pattern: {pattern!r}")
    # Cycling 1, 3/4, 1/2, 1/4 of the nominal length: batches then disagree on
    # `rows_in_batch`, which is what makes the level bands ragged and the row
    # decode's binary search do real work.
    fractions = (1.0, 0.75, 0.5, 0.25)
    out = []
    for b in range(batch):
        length = int(seqlen_k * fractions[b % len(fractions)])
        out.append(max(blk_kv, _round_up(length, blk_kv)))
    return out


def row_coords(seqlens_k: list[int], blk_kv: int) -> tuple[list[int], list[int]]:
    """Level-major `row_linear -> (batch_idx, kv_block_idx)`.

    This is the mapping `_decode_sparse_row_linear` (:196-230) inverts on the
    device: level band L holds one row for every batch with more than L blocks,
    batches in index order.
    """
    rows_in_batch = [_ceil_div(length, blk_kv) for length in seqlens_k]
    batch_of_row: list[int] = []
    level_of_row: list[int] = []
    for level in range(max(rows_in_batch, default=0)):
        for b, rows in enumerate(rows_in_batch):
            if rows > level:
                batch_of_row.append(b)
                level_of_row.append(level)
    return batch_of_row, level_of_row


def _seqlens_q(seqlens_k: list[int], seqlen_q: int) -> list[int]:
    """Per-batch Q length: a batch never carries more queries than it has keys.

    A prefill config passes ``seqlen_q == seqlen_k``, so under a varlen K sweep
    each batch prefills its own (shorter) sequence; a decode config passes a
    short Q that is independent of K.  ``min`` expresses both.
    """
    return [min(seqlen_q, length) for length in seqlens_k]


def _row_fanout(seqlen_q: int, seqlen_k: int, blk_kv: int, topk: int):
    """Expected per-kv-block reference count for one batch, as a torch vector.

    A causal top-k selector: query ``t`` sits at absolute position
    ``seqlen_k - seqlen_q + t``, sees blocks ``0..g_t``, and keeps
    ``min(topk, g_t + 1)`` of them.  Summing that selection probability over the
    queries that can see each block gives the fanout profile the CSR carries --
    front-loaded, since early blocks are visible to every later query.
    """
    import torch

    rows = _ceil_div(seqlen_k, blk_kv)
    device = "cuda"
    queries = torch.arange(seqlen_q, device=device, dtype=torch.float64)
    last_visible = ((seqlen_k - seqlen_q) + queries).div(blk_kv, rounding_mode="floor").long()
    last_visible = last_visible.clamp(min=0, max=rows - 1)
    visible = last_visible + 1
    keep = torch.clamp(visible, max=topk).to(torch.float64) / visible.to(torch.float64)

    # Histogram the per-query selection weight by its last visible block, then
    # take the suffix sum: block j is reachable by every query with g_t >= j.
    weight = torch.zeros(rows, device=device, dtype=torch.float64)
    weight.scatter_add_(0, last_visible, keep)
    return torch.flip(torch.cumsum(torch.flip(weight, (0,)), 0), (0,))


def prepare_data(*, seed: int = 0, **config) -> dict[str, Any]:
    """Build one CSR fanout plus the schedule sizing the host would compute."""
    import torch

    from tirx_kernels.runner import hardware_num_sms

    config.pop("label", None)
    batch = config["batch"]
    seqlen_q = config["seqlen_q"]
    blk_kv = config["blk_kv"]
    head_kv = config["head_kv"]
    topk = config["topk"]

    seqlens_k = _seqlens_k(batch, config["seqlen_k"], blk_kv, config["seqlen_pattern"])
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(1234 + seed)

    cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_seqlens_k[1:] = torch.tensor(seqlens_k, dtype=torch.int32, device=device).cumsum(0)

    seqlens_q = _seqlens_q(seqlens_k, seqlen_q)
    batch_of_row, level_of_row = row_coords(seqlens_k, blk_kv)
    total_rows = len(batch_of_row)
    total_q = sum(seqlens_q)

    # Per-batch fanout profiles, scattered into level-major row order.
    counts = torch.zeros(head_kv, total_rows, dtype=torch.float64, device=device)
    profiles = [
        _row_fanout(q_len, k_len, blk_kv, topk)
        for q_len, k_len in zip(seqlens_q, seqlens_k, strict=True)
    ]
    row_index = torch.arange(total_rows, device=device)
    batch_index = torch.tensor(batch_of_row, dtype=torch.long, device=device)
    level_index = torch.tensor(level_of_row, dtype=torch.long, device=device)
    for b, profile in enumerate(profiles):
        mask = batch_index == b
        counts[:, row_index[mask]] = profile[level_index[mask]].unsqueeze(0)

    # Draw integer counts per head rather than rounding the expectation.  In the
    # decode regime a block's expected fanout is well below one -- 8 queries
    # picking 32 of 512 blocks averages 0.5 -- and rounding would collapse the
    # whole CSR to zero instead of the sparse 0/1/2 pattern the real selector
    # produces.  Poisson keeps the mean and restores that structure.
    counts = torch.poisson(counts, generator=generator)

    # The attention sink: block 0 of a batch is selected by every query, so the
    # schedule's hottest rows are the ones that actually split into chunks.
    sink_rows = level_index == 0
    sink_q = torch.tensor(seqlens_q, dtype=torch.float64, device=device)[batch_index[sink_rows]]
    counts[:, sink_rows] = sink_q.unsqueeze(0).expand(head_kv, -1)
    counts = counts.clamp_(min=0)

    # Keep the fanout inside the reference upper bound the capacity model uses
    # (`total_q * topk` per head), so no work item is ever dropped by the
    # `work_idx < work_capacity` guard and the output stays a complete multiset.
    per_head_budget = float(total_q * topk)
    totals = counts.sum(dim=1, keepdim=True).clamp(min=1.0)
    scale = torch.clamp(per_head_budget / totals, max=1.0)
    counts = (counts * scale).floor().clamp_(min=0).to(torch.int64)

    k2q_row_ptr = torch.zeros(head_kv, total_rows + 1, dtype=torch.int32, device=device)
    k2q_row_ptr[:, 1:] = counts.cumsum(dim=1).to(torch.int32)

    target = target_q_per_cta(
        total_q=total_q,
        topk=topk,
        blk_kv=blk_kv,
        head_kv=head_kv,
        qhead_per_kv=config["qhead_per_kv"],
        num_sms=hardware_num_sms(),
    )
    capacity = flat_schedule_capacity(
        total_rows=total_rows, total_q=total_q, topk=topk, head_kv=head_kv, target=target
    )

    chunks = torch.where(
        counts > 0, torch.div(counts + target - 1, target, rounding_mode="floor"), 0
    )
    expected_work = int(chunks.sum().item())
    if expected_work > capacity:
        raise AssertionError(
            f"generated fanout needs {expected_work} work items but capacity is {capacity}"
        )

    return {
        "config": dict(config),
        "seqlens_k": seqlens_k,
        "cu_seqlens_k": cu_seqlens_k,
        "k2q_row_ptr": k2q_row_ptr.contiguous(),
        "counts": counts,
        "batch_of_row": batch_of_row,
        "level_of_row": level_of_row,
        "total_rows": total_rows,
        "total_q": total_q,
        "target_q_per_cta": target,
        "work_capacity": capacity,
        "expected_work": expected_work,
        "head_kv": head_kv,
        "blk_kv": blk_kv,
    }


def make_outputs(data: dict[str, Any]) -> dict[str, Any]:
    """Fresh zeroed `scheduler_metadata` / `work_count`, as the host wrapper allocates."""
    import torch

    return {
        "scheduler_metadata": torch.zeros(
            (data["work_capacity"], WORK_FIELDS), dtype=torch.int32, device="cuda"
        ),
        "work_count": torch.zeros((1,), dtype=torch.int32, device="cuda"),
    }


def tirx_args(data: dict[str, Any], outputs: dict[str, Any]) -> tuple:
    """The flat launch ABI, bound once outside any timed region."""
    return (
        data["k2q_row_ptr"].reshape(-1),
        data["cu_seqlens_k"],
        outputs["scheduler_metadata"].reshape(-1),
        outputs["work_count"],
        data["total_rows"],
        len(data["seqlens_k"]),
        data["target_q_per_cta"],
        data["work_capacity"],
        data["head_kv"],
        data["blk_kv"],
    )


# ---------------------------------------------------------------------------
# Correctness.
# ---------------------------------------------------------------------------
def expected_work_items(data: dict[str, Any]):
    """The work-item multiset the schedule must contain, as a sorted int32 table.

    Chunking follows the kernel: a row with no references emits nothing, and a
    row with `n` references emits `ceil(n / target)` items whose `q_begin` walks
    the row in `target`-sized steps (:315-345).
    """
    import torch

    counts = data["counts"]
    target = data["target_q_per_cta"]
    head_kv, total_rows = counts.shape

    chunks = torch.where(
        counts > 0, torch.div(counts + target - 1, target, rounding_mode="floor"), 0
    )
    head_idx, row_idx = torch.meshgrid(
        torch.arange(head_kv, device=counts.device),
        torch.arange(total_rows, device=counts.device),
        indexing="ij",
    )
    flat_chunks = chunks.reshape(-1)
    repeat_head = torch.repeat_interleave(head_idx.reshape(-1), flat_chunks)
    repeat_row = torch.repeat_interleave(row_idx.reshape(-1), flat_chunks)
    repeat_count = torch.repeat_interleave(counts.reshape(-1), flat_chunks)

    starts = torch.cumsum(flat_chunks, 0) - flat_chunks
    chunk_ordinal = torch.arange(int(flat_chunks.sum().item()), device=counts.device)
    chunk_ordinal = chunk_ordinal - torch.repeat_interleave(starts, flat_chunks)

    q_begin = chunk_ordinal * target
    q_count = torch.clamp(repeat_count - q_begin, max=target)

    batch_of_row = torch.tensor(data["batch_of_row"], dtype=torch.int64, device=counts.device)
    level_of_row = torch.tensor(data["level_of_row"], dtype=torch.int64, device=counts.device)

    return torch.stack(
        [
            repeat_head,
            repeat_row,
            q_begin,
            q_count,
            batch_of_row[repeat_row],
            level_of_row[repeat_row],
        ],
        dim=1,
    ).to(torch.int32)


def sorted_rows(table):
    """Canonical order for an arrival-ordered work table."""
    import torch

    if table.numel() == 0:
        return table
    keys = table.to(torch.int64)
    order = torch.arange(keys.shape[0], device=keys.device)
    for column in reversed(range(keys.shape[1])):
        order = order[torch.argsort(keys[order, column], stable=True)]
    return table[order]


def assert_schedule_matches(data: dict[str, Any], outputs: dict[str, Any]) -> None:
    """Compare the produced schedule with the expected multiset."""
    import torch

    produced_count = int(outputs["work_count"][0].item())
    expected = expected_work_items(data)
    if produced_count != expected.shape[0]:
        raise AssertionError(
            f"work_count is {produced_count}, expected {expected.shape[0]} work items"
        )
    produced = outputs["scheduler_metadata"][:produced_count]
    torch.testing.assert_close(sorted_rows(produced), sorted_rows(expected), rtol=0, atol=0)


def run_test(**config):
    """Compile, launch, and validate one config against MSA's own kernel."""
    import unittest

    import torch

    from tirx_kernels.runner import compile_kernel

    if not torch.cuda.is_available():  # pragma: no cover - environment dependent
        raise unittest.SkipTest("CUDA device unavailable")

    config.pop("label", None)
    data = prepare_data(**config)

    # The source kernel over the same inputs, as the primary oracle.
    try:
        from tirx_kernels.msa.utils._msa_bench import compiled_flat_schedule
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"MSA reference unavailable: {exc}") from exc
    reference_outputs = make_outputs(data)
    try:
        launch_reference(data, reference_outputs, compiled_flat_schedule)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"MSA reference unavailable: {exc}") from exc
    torch.cuda.synchronize()
    assert_schedule_matches(data, reference_outputs)

    executable = compile_kernel(get_kernel(**config))
    outputs = make_outputs(data)
    executable(*tirx_args(data, outputs))
    torch.cuda.synchronize()
    # Invariant checker retained by design: the schedule layout is
    # nondeterministic (atomic slot handout), so bitwise-vs-reference is
    # meaningless and the contract itself is what both sides must satisfy.
    assert_schedule_matches(data, outputs)


def launch_reference(data: dict[str, Any], outputs: dict[str, Any], compiled_factory) -> None:
    """One launch of the MSA kernel over the given buffers."""
    case = {
        "k2q_row_ptr": data["k2q_row_ptr"],
        "cu_seqlens_k": data["cu_seqlens_k"],
        "scheduler_metadata": outputs["scheduler_metadata"],
        "work_count": outputs["work_count"],
        "target_q_per_cta": data["target_q_per_cta"],
        "work_capacity": data["work_capacity"],
        "head_kv": data["head_kv"],
        "blk_kv": data["blk_kv"],
    }
    compiled = compiled_factory(case)
    compiled(
        case["k2q_row_ptr"],
        case["cu_seqlens_k"],
        case["scheduler_metadata"],
        case["work_count"],
        case["target_q_per_cta"],
        case["work_capacity"],
        case["head_kv"],
        case["blk_kv"],
    )


# ---------------------------------------------------------------------------
# Benchmark entry points.
# ---------------------------------------------------------------------------
# Every launch must see a zero `work_count`: the counter is monotone, so a
# second launch against the same counter reserves slots past `work_capacity`,
# the guarded stores stop firing, and the timed kernel loses its output path
# entirely (and eventually wraps int32 into negative indices).  Both
# implementations therefore rotate over the same pre-zeroed counter slots and
# re-zero the whole block once a full rotation is spent -- the reset is one
# small memset per COUNTER_SLOTS launches rather than one per launch.
COUNTER_SLOTS = 4096

# MSA marks its tensors `assume_tensor_aligned`, and the compiled entry rejects a
# data pointer that is not 16-byte aligned, so consecutive slots are four int32
# apart rather than adjacent.
COUNTER_STRIDE = 4


def _rotating_counters():
    """One zeroed, 16-byte-aligned `work_count` cell per rotation slot."""
    import torch

    return torch.zeros((COUNTER_SLOTS * COUNTER_STRIDE,), dtype=torch.int32, device="cuda")


def _counter_slots(counters):
    return [counters[i * COUNTER_STRIDE : i * COUNTER_STRIDE + 1] for i in range(COUNTER_SLOTS)]


def prepare_bench(**config):
    """Compile the TIRx specialization without initializing CUDA."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    config.pop("label", None)
    state = {"config": dict(config), "executable": compile_kernel(get_kernel(**config))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
    """Kernel-only comparison against MSA's compiled flat-schedule launch."""
    from tirx_kernels.runner import bench

    config = {**prepared["config"], **config}
    config.pop("label", None)
    data = prepare_data(**config)
    executable = prepared["executable"]

    tirx_outputs = make_outputs(data)
    tirx_counters = _rotating_counters()
    tirx_slots = _counter_slots(tirx_counters)
    tirx_base = tirx_args(data, tirx_outputs)
    tirx_step = [0]

    def tirx_launch():
        step = tirx_step[0]
        slot = step % COUNTER_SLOTS
        if slot == 0 and step:
            tirx_counters.zero_()
        tirx_step[0] = step + 1
        executable(tirx_base[0], tirx_base[1], tirx_base[2], tirx_slots[slot], *tirx_base[4:])

    def build_reference():
        from tirx_kernels.msa.utils._msa_bench import compiled_flat_schedule

        reference_outputs = make_outputs(data)
        counters = _rotating_counters()
        slots = _counter_slots(counters)
        case = {
            "k2q_row_ptr": data["k2q_row_ptr"],
            "cu_seqlens_k": data["cu_seqlens_k"],
            "scheduler_metadata": reference_outputs["scheduler_metadata"],
            "work_count": reference_outputs["work_count"],
            "target_q_per_cta": data["target_q_per_cta"],
            "work_capacity": data["work_capacity"],
            "head_kv": data["head_kv"],
            "blk_kv": data["blk_kv"],
        }
        compiled = compiled_flat_schedule(case)
        metadata = reference_outputs["scheduler_metadata"]
        step = [0]

        def reference_launch():
            index = step[0]
            slot = index % COUNTER_SLOTS
            if slot == 0 and index:
                counters.zero_()
            step[0] = index + 1
            compiled(
                data["k2q_row_ptr"],
                data["cu_seqlens_k"],
                metadata,
                slots[slot],
                data["target_q_per_cta"],
                data["work_capacity"],
                data["head_kv"],
                data["blk_kv"],
            )

        # Pay the CuTeDSL compile and first-launch cost before timing starts.
        reference_launch()
        step[0] = 0
        counters.zero_()
        return reference_launch

    return bench(
        {"tirx": tirx_launch},
        references={"msa": build_reference},
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
    "flat_schedule_capacity",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_test",
    "target_q_per_cta",
]
