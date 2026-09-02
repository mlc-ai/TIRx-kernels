# This file is a TIRx port of code from MSA
# (https://github.com/MiniMax-AI/MSA @ 80434d7f), Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright TIRx authors

"""MSA sparse-attention forward split-slot preparation.

Ports ``SparseAttentionPrepareFwdSplitAtomicSm100`` -- the stage that hands every
CSR edge a split slot, so the forward attention kernel can write its partial
output without a second round of atomics.

One CTA per work item. The work list comes from the flat-schedule kernel
(:mod:`tirx_kernels.msa.sparse_prepare_flat_schedule`), and the grid is sized by
that list's *capacity* rather than its length, so CTAs past ``work_count`` exit
immediately. A surviving CTA reads its work item's ``(head_kv, row, q_begin,
q_count, batch)``, thread 0 broadcasts the row's CSR base through shared memory,
and the 256 threads then stride over the row's edges: each edge reserves a slot
with a device-scope atomic on ``split_counts[q_abs, head_kv]`` and writes
``q_idx | (slot << 24)``.

Slots are handed out in atomic-arrival order, so which edge of a
``(q_abs, head_kv)`` group gets which slot is not reproducible. What is defined,
and what the consumer relies on, is that the group's slots are exactly
``{0 .. degree-1}`` with no gap and no duplicate: the forward kernel writes
``split_counts[q_abs, h]`` partials into ``O_partial[0..count-1]`` and the
combine kernel reduces exactly that many.

Upstream source: python/fmha_sm100/cute/src/sm100/prepare_scheduler.py:348.
"""

from typing import Any

import tirx_kernels.kern as K
from tirx_kernels.msa.sparse_prepare_flat_schedule import (
    _seqlens_k,
    _seqlens_q,
    flat_schedule_capacity,
    row_coords,
    target_q_per_cta,
)
from tirx_kernels.msa.utils._scalar_ops import (
    atom_add_global_i32,
    bar_sync,
    ld_global_i32,
    ld_shared_i32,
    st_global_i32,
    st_shared_i32,
)

KERNEL_META = {
    "name": "msa_sparse_prepare_fwd_split_atomic_sm100",
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

# `__init__(num_threads=256)` (:351-358); validated a multiple of 32.
NUM_THREADS = 256

# `SharedStorage.sRow: MemRange[Int32, 3]` (:360-364) -- 12 bytes.
SROW_FIELDS = 3

# `scheduler_metadata` columns; this kernel reads 0..4 and ignores 5.
WORK_FIELDS = 6

# `q_idx | ((split_slot & 0xFF) << 24)` (:478-480).
SLOT_SHIFT = 24
SLOT_MASK = 0xFF
Q_IDX_MASK = (1 << SLOT_SHIFT) - 1

# The source's `@cute.struct SharedStorage` reads static, but `SmemAllocator`
# takes it from the dynamic pool -- the export carries
# `.extern .shared .align 1024 .b8 __dynamic_shmem__0[]` -- so the matching TIRx
# form is a pool allocation plus this launch tag, not a parser allocation.
LAUNCH_TAGS = ("blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory")


# ---------------------------------------------------------------------------
# Target entry.
# ---------------------------------------------------------------------------
@K.kernel(warps=NUM_THREADS // 32, arch="sm_100a", grid=lambda p: p["work_capacity"])
def _kernel(
    k2q_row_ptr: K.gptr[K.i32],
    k2q_q_indices: K.gptr[K.i32],
    scheduler_metadata: K.gptr[K.i32],
    work_count: K.gptr[K.i32],
    k2q_qsplit_indices: K.gptr[K.i32],
    split_counts: K.gptr[K.i32],
    cu_seqlens_q: K.gptr[K.i32],
    total_rows: K.i32,
    num_batches: K.i32,
    work_capacity: K.i32,
    nnz_capacity: K.i32,
    total_q: K.i32,
    num_heads_kv: K.i32,
    max_seqlen_q: K.i32,
    topk: K.i32,
):
    # CUDA TRANSCRIPTION START
    # sketch: static ABI/launch, one CTA per work item -> :445-446.
    block = K.cta_id()
    tidx = K.thread_id()

    # sketch: the row published through shared memory -> :360-364, :448-450.
    # The source's struct is allocated from the dynamic pool, not a static
    # `__shared__` array; the export shows `.extern .shared __dynamic_shmem__0`.
    pool = K.smem_pool()
    srow = pool.alloc((SROW_FIELDS,), K.i32, align=16)

    # sketch: the oversized-grid early-out -> :447.  The grid is sized by the
    # work list's capacity because `work_count` is device-resident, so the tail
    # CTAs retire here.  It precedes every shared access, so the barrier below
    # is never reached by a partial CTA.
    with K.If(block < ld_global_i32(work_count, 0)), K.Then():
        # sketch: the one metadata field every thread needs -> :451.
        work_base = block * WORK_FIELDS
        head_kv_idx = ld_global_i32(scheduler_metadata, work_base)

        # sketch: thread 0 publishes the row -> :457-461.  The compiler sinks
        # the other four metadata loads into this block, which is what makes
        # all three shared slots load-bearing rather than redundant.
        with K.If(tidx == 0), K.Then():
            batch_idx_t0 = ld_global_i32(scheduler_metadata, work_base + 4)
            q_count_t0 = ld_global_i32(scheduler_metadata, work_base + 3)
            q_begin_t0 = ld_global_i32(scheduler_metadata, work_base + 2)
            row_linear_t0 = ld_global_i32(scheduler_metadata, work_base + 1)
            row_base_t0 = ld_global_i32(k2q_row_ptr, head_kv_idx * (total_rows + 1) + row_linear_t0)
            st_shared_i32(srow, 0, row_base_t0 + q_begin_t0)
            st_shared_i32(srow, 1, q_count_t0)
            st_shared_i32(srow, 2, batch_idx_t0)

        # sketch: the only synchronization in the kernel -> :462.
        bar_sync()

        # sketch: everyone reads the row back -> :463-465.  `q_count` first: it
        # is the loop guard, and the export reads it first for that reason.
        row_count = ld_shared_i32(srow, 1)
        row_start = ld_shared_i32(srow, 0)
        batch_idx = ld_shared_i32(srow, 2)

        # sketch: lane-strided edge emission -> :466-481.
        qi = K.local_scalar(K.i32, init=tidx)
        with K.While(qi < row_count):
            edge = row_start + qi
            q_idx = ld_global_i32(k2q_q_indices, head_kv_idx * nnz_capacity + edge)
            # sketch: the validity guard -> :470.  Well-formed input never takes
            # it -- a CSR row covers only filled entries -- but the guard exists
            # because the GPU index builder leaves its tail uninitialized and the
            # reference builder fills it with -1.
            with K.If(K.And(q_idx >= 0, q_idx < max_seqlen_q)), K.Then():
                # The address of `cu_seqlens_q[batch_idx]` is loop-invariant, but
                # the export re-loads the value every edge: the atomic's memory
                # clobber prevents hoisting it. The port leaves it in the body.
                q_abs = ld_global_i32(cu_seqlens_q, batch_idx) + q_idx
                split_slot = atom_add_global_i32(
                    split_counts, q_abs * num_heads_kv + head_kv_idx, K.uint32(1)
                )
                # sketch: the capacity guard -> :477.  Signed compare against the
                # value the atomic returned; also unreachable on well-formed
                # input, where a (q_abs, head) group holds at most `topk` edges.
                with K.If(split_slot < topk), K.Then():
                    st_global_i32(
                        k2q_qsplit_indices,
                        head_kv_idx * nnz_capacity + edge,
                        K.bitwise_or(
                            q_idx, K.shift_left(K.bitwise_and(split_slot, SLOT_MASK), SLOT_SHIFT)
                        ),
                    )
            K.assign(qi, qi + NUM_THREADS)


def get_kernel(**config):
    """Return the TIRx specialization of `SparseAttentionPrepareFwdSplitAtomicSm100`.

    Nothing about a config reaches the kernel as a compile-time constant: the
    source's compile cache key is a bare string tuple (:496-498), so one binary
    serves every shape and all extents and scalars stay runtime arguments.
    """
    config.pop("label", None)
    return _kernel.func.with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))


# ---------------------------------------------------------------------------
# Config matrix.
#
# The eight benchmark shapes are the ones the flat-schedule sibling uses, which
# already span this kernel's two cost regimes: the prefill shapes are edge-bound
# (up to ~4.2M edges) while the decode shapes launch a capacity-sized grid whose
# CTAs mostly exit at once. Their `target_q_per_cta` also falls either side of
# the 256-thread stride, so both the multi-iteration and single-iteration forms
# of the emission loop are covered.
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

# Correctness adds the sibling's small boundary shapes: a varlen pair whose rows
# fit in one level band, and a fanout sparse enough that some rows are empty.
CONFIGS = [
    *BENCH_CONFIGS,
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
#
# The flat-schedule sibling's generator is expectation-based and produces per-row
# counts only; this kernel needs the real edge payload, so the CSR is built
# edge-first here. Three properties of the real index builder have to hold, and
# each one the benchmark or the oracle depends on:
#
#   * degree per `(q_abs, head)` is `min(topk, visible) <= topk`, so
#     `split_slot < topk` always holds and every valid edge is written. Filling
#     pre-computed row slots by sampling `q_idx` instead would give an expected
#     degree of exactly `topk`, so about half the queries would overshoot and
#     half the stores would silently die.
#   * edges inside a row ascend by `q_idx`. Production's scatter is sorted
#     (`build_k2q_csr.cu:15, 341-349`), and it is what makes one iteration's 256
#     threads hit 256 consecutive `(q_abs, h)` cells rather than scattering
#     across the whole tensor.
#   * the tail past `nnz` is `-1`, as the reference builder writes
#     (`sparse_index_utils.py:250`) and the CUDA builder memsets -- which is what
#     the kernel's `q_idx >= 0` guard exists for.
# ---------------------------------------------------------------------------
def _select_blocks(q_len: int, rows_in_batch: int, topk: int, blk_kv: int, k_len: int, generator):
    """Per query, the kv blocks it attends: the sink plus a causal sample."""
    import torch

    device = "cuda"
    queries = torch.arange(q_len, device=device)
    last_visible = ((k_len - q_len) + queries).div(blk_kv, rounding_mode="floor")
    last_visible = last_visible.clamp(min=0, max=rows_in_batch - 1)
    keep = torch.clamp(last_visible + 1, max=topk)

    # Rank the visible blocks by a random priority, with block 0 pinned first:
    # it is the attention sink and every query keeps it.
    priority = torch.rand((q_len, rows_in_batch), device=device, generator=generator)
    block_ids = torch.arange(rows_in_batch, device=device).unsqueeze(0)
    priority = torch.where(block_ids == 0, torch.full_like(priority, -1.0), priority)
    priority = torch.where(
        block_ids <= last_visible.unsqueeze(1), priority, torch.full_like(priority, 2.0)
    )
    rank = priority.argsort(dim=1).argsort(dim=1)
    return torch.nonzero(rank < keep.unsqueeze(1), as_tuple=True)


def prepare_data(*, seed: int = 0, **config) -> dict[str, Any]:
    """Build the CSR edge payload, the work list, and the schedule sizing."""
    import torch

    from tirx_kernels.runner import hardware_num_sms

    config.pop("label", None)
    batch = config["batch"]
    blk_kv = config["blk_kv"]
    head_kv = config["head_kv"]
    topk = config["topk"]
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(1234 + seed)

    seqlens_k = _seqlens_k(batch, config["seqlen_k"], blk_kv, config["seqlen_pattern"])
    seqlens_q = _seqlens_q(seqlens_k, config["seqlen_q"])
    total_q = sum(seqlens_q)
    max_seqlen_q = max(seqlens_q)

    cu_seqlens_q = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.tensor(seqlens_q, dtype=torch.int32, device=device).cumsum(0)
    q_offsets = [0]
    for length in seqlens_q:
        q_offsets.append(q_offsets[-1] + length)

    batch_of_row, level_of_row = row_coords(seqlens_k, blk_kv)
    total_rows = len(batch_of_row)
    row_of_coord = {
        (b, level): r for r, (b, level) in enumerate(zip(batch_of_row, level_of_row, strict=True))
    }

    nnz_capacity = total_q * topk
    k2q_row_ptr = torch.zeros((head_kv, total_rows + 1), dtype=torch.int32, device=device)
    k2q_q_indices = torch.full((head_kv, nnz_capacity), -1, dtype=torch.int32, device=device)
    degrees = torch.zeros((total_q, head_kv), dtype=torch.int64, device=device)

    for h in range(head_kv):
        rows, q_idx, q_abs = [], [], []
        for b, (k_len, q_len) in enumerate(zip(seqlens_k, seqlens_q, strict=True)):
            rows_in_batch = (k_len + blk_kv - 1) // blk_kv
            qi, block = _select_blocks(q_len, rows_in_batch, topk, blk_kv, k_len, generator)
            lookup = torch.tensor(
                [row_of_coord[(b, level)] for level in range(rows_in_batch)],
                dtype=torch.int64,
                device=device,
            )
            rows.append(lookup[block])
            q_idx.append(qi.to(torch.int32))
            q_abs.append(qi.to(torch.int64) + q_offsets[b])
        rows_cat = torch.cat(rows)
        q_idx_cat = torch.cat(q_idx)
        q_abs_cat = torch.cat(q_abs)

        # Row-major, q ascending within a row -- production's sorted scatter.
        order = (rows_cat * (max_seqlen_q + 1) + q_idx_cat.to(torch.int64)).argsort(stable=True)
        rows_sorted = rows_cat[order]
        k2q_row_ptr[h, 1:] = (
            torch.bincount(rows_sorted, minlength=total_rows).cumsum(0).to(torch.int32)
        )
        k2q_q_indices[h, : order.numel()] = q_idx_cat[order]
        degrees[:, h] = torch.bincount(q_abs_cat, minlength=total_q)

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
    counts = (k2q_row_ptr[:, 1:] - k2q_row_ptr[:, :-1]).to(torch.int64)
    work = _work_items(counts, target, batch_of_row, head_kv, total_rows, device)

    scheduler_metadata = torch.zeros((capacity, WORK_FIELDS), dtype=torch.int32, device=device)
    scheduler_metadata[: work.shape[0]] = work
    work_count = torch.tensor([work.shape[0]], dtype=torch.int32, device=device)
    if work.shape[0] > capacity:
        raise AssertionError(f"work list {work.shape[0]} exceeds capacity {capacity}")

    return {
        "config": dict(config),
        "seqlens_k": seqlens_k,
        "seqlens_q": seqlens_q,
        "cu_seqlens_q": cu_seqlens_q,
        "k2q_row_ptr": k2q_row_ptr.contiguous(),
        "k2q_q_indices": k2q_q_indices.contiguous(),
        "scheduler_metadata": scheduler_metadata.contiguous(),
        "work_count": work_count,
        "degrees": degrees,
        "total_rows": total_rows,
        "total_q": total_q,
        "max_seqlen_q": max_seqlen_q,
        "nnz_capacity": nnz_capacity,
        "work_capacity": capacity,
        "head_kv": head_kv,
        "topk": topk,
    }


def _work_items(counts, target: int, batch_of_row, head_kv: int, total_rows: int, device):
    """The work list the flat-schedule kernel produces, in row-head order.

    Synthesized rather than produced by running that kernel: a device-produced
    list redraws its atomic-arrival permutation on every invocation, which would
    add run-to-run noise and couple this benchmark's input to another kernel.
    Row-head order is the flat-schedule kernel's own indexing (`row_head_idx =
    block * 4 + warp`), so it is the closest deterministic stand-in for real
    arrival order -- and it keeps the heavy level-0 sink rows at the front,
    which a uniform shuffle would not.
    """
    import torch

    chunks = torch.where(
        counts > 0, torch.div(counts + target - 1, target, rounding_mode="floor"), 0
    )
    head_idx, row_idx = torch.meshgrid(
        torch.arange(head_kv, device=device), torch.arange(total_rows, device=device), indexing="ij"
    )
    flat = chunks.reshape(-1)
    head_rep = torch.repeat_interleave(head_idx.reshape(-1), flat)
    row_rep = torch.repeat_interleave(row_idx.reshape(-1), flat)
    count_rep = torch.repeat_interleave(counts.reshape(-1), flat)

    starts = torch.cumsum(flat, 0) - flat
    ordinal = torch.arange(int(flat.sum().item()), device=device) - torch.repeat_interleave(
        starts, flat
    )
    q_begin = ordinal * target
    q_count = torch.clamp(count_rep - q_begin, max=target)
    batch_row = torch.tensor(batch_of_row, dtype=torch.int64, device=device)

    table = torch.stack(
        [
            head_rep,
            row_rep,
            q_begin,
            q_count,
            batch_row[row_rep],
            torch.zeros_like(row_rep),  # kv_block_idx: written by the producer, unread here
        ],
        dim=1,
    ).to(torch.int32)
    return table[(row_rep * head_kv + head_rep).argsort(stable=True)]


def make_outputs(data: dict[str, Any]) -> dict[str, Any]:
    """Fresh outputs: `qsplit` uninitialized and `split_counts` zeroed, as the host does."""
    import torch

    return {
        "k2q_qsplit_indices": torch.empty(
            (data["head_kv"], data["nnz_capacity"]), dtype=torch.int32, device="cuda"
        ),
        "split_counts": torch.zeros(
            (data["total_q"], data["head_kv"]), dtype=torch.int32, device="cuda"
        ),
    }


def tirx_args(data: dict[str, Any], outputs: dict[str, Any]) -> tuple:
    """The flat launch ABI, bound once outside any timed region."""
    return (
        data["k2q_row_ptr"].reshape(-1),
        data["k2q_q_indices"].reshape(-1),
        data["scheduler_metadata"].reshape(-1),
        data["work_count"],
        outputs["k2q_qsplit_indices"].reshape(-1),
        outputs["split_counts"].reshape(-1),
        data["cu_seqlens_q"],
        data["total_rows"],
        len(data["seqlens_k"]),
        data["work_capacity"],
        data["nnz_capacity"],
        data["total_q"],
        data["head_kv"],
        data["max_seqlen_q"],
        data["topk"],
    )


def launch_reference(data: dict[str, Any], outputs: dict[str, Any], compiled_factory) -> None:
    """One launch of the MSA kernel over the given buffers."""
    case = _reference_case(data, outputs)
    compiled_factory(case)(
        case["k2q_row_ptr"],
        case["k2q_q_indices"],
        case["scheduler_metadata"],
        case["work_count"],
        case["k2q_qsplit_indices"],
        case["split_counts"],
        case["cu_seqlens_q"],
        case["work_capacity"],
        case["max_seqlen_q"],
        case["topk"],
    )


def _reference_case(data: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "k2q_row_ptr": data["k2q_row_ptr"],
        "k2q_q_indices": data["k2q_q_indices"],
        "scheduler_metadata": data["scheduler_metadata"],
        "work_count": data["work_count"],
        "k2q_qsplit_indices": outputs["k2q_qsplit_indices"],
        "split_counts": outputs["split_counts"],
        "cu_seqlens_q": data["cu_seqlens_q"],
        "work_capacity": data["work_capacity"],
        "max_seqlen_q": data["max_seqlen_q"],
        "topk": data["topk"],
    }


# ---------------------------------------------------------------------------
# Correctness: invariants, not element-wise equality.
# ---------------------------------------------------------------------------
def assert_split_metadata(data: dict[str, Any], outputs: dict[str, Any]) -> None:
    """Check the three properties that survive the atomic's arrival order.

    Slot assignment is not reproducible, so a bitwise comparison against another
    run means nothing. What is defined, and what the forward/combine pair
    depends on, is: the per-`(q_abs, head)` counts; the low 24 bits of a written
    entry; and that a group's slots are exactly `{0..degree-1}`, gapless and
    without duplicates -- a gap would make combine reduce an uninitialized
    `O_partial` slice, a duplicate would double-count one and stale another.
    """
    import torch

    torch.testing.assert_close(
        outputs["split_counts"], data["degrees"].to(torch.int32), rtol=0, atol=0
    )

    row_ptr = data["k2q_row_ptr"]
    q_indices = data["k2q_q_indices"]
    packed = outputs["k2q_qsplit_indices"]
    total_rows = data["total_rows"]
    head_kv = data["head_kv"]
    cu_q = data["cu_seqlens_q"].to(torch.int64)
    batch_of_row = torch.tensor(
        [b for b, _ in zip(*row_coords(data["seqlens_k"], data["config"]["blk_kv"]), strict=True)],
        dtype=torch.int64,
        device=packed.device,
    )

    for h in range(head_kv):
        live = int(row_ptr[h, total_rows].item())
        if live == 0:
            continue
        edge_q = q_indices[h, :live].to(torch.int64)
        got = packed[h, :live]
        # low 24 bits are a pure passthrough of that edge's own q_idx
        torch.testing.assert_close(
            torch.bitwise_and(got, Q_IDX_MASK), edge_q.to(torch.int32), rtol=0, atol=0
        )
        slots = torch.bitwise_right_shift(got, SLOT_SHIFT).bitwise_and(SLOT_MASK).to(torch.int64)

        # group each edge by its (q_abs, head) and check the slots form range(n)
        starts = row_ptr[h, :total_rows].to(torch.int64)
        ends = row_ptr[h, 1 : total_rows + 1].to(torch.int64)
        row_of_edge = torch.repeat_interleave(
            torch.arange(total_rows, device=packed.device), (ends - starts)
        )
        q_abs = cu_q[batch_of_row[row_of_edge]] + edge_q
        order = (q_abs * (data["topk"] + 1) + slots).argsort(stable=True)
        q_sorted, slot_sorted = q_abs[order], slots[order]
        group_start = torch.cat(
            [torch.ones(1, dtype=torch.bool, device=packed.device), q_sorted[1:] != q_sorted[:-1]]
        )
        expected = torch.arange(q_sorted.numel(), device=packed.device)
        expected = (
            expected
            - torch.cummax(torch.where(group_start, expected, torch.zeros_like(expected)), 0).values
        )
        if not bool(torch.equal(slot_sorted, expected)):
            raise AssertionError(f"head {h}: slots are not a gapless permutation per (q_abs, head)")


def run_test(**config):
    """Compile, launch, and validate one config against MSA's own kernel."""
    import unittest

    import torch

    from tirx_kernels.runner import compile_kernel

    if not torch.cuda.is_available():  # pragma: no cover - environment dependent
        raise unittest.SkipTest("CUDA device unavailable")

    config.pop("label", None)
    data = prepare_data(**config)

    try:
        from tirx_kernels.msa.utils._msa_bench import compiled_fwd_split_atomic
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"MSA reference unavailable: {exc}") from exc
    reference_outputs = make_outputs(data)
    try:
        launch_reference(data, reference_outputs, compiled_fwd_split_atomic)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"MSA reference unavailable: {exc}") from exc
    torch.cuda.synchronize()
    assert_split_metadata(data, reference_outputs)

    executable = compile_kernel(get_kernel(**config))
    outputs = make_outputs(data)
    executable(*tirx_args(data, outputs))
    torch.cuda.synchronize()
    # Invariant checker retained by design: slot order is atomic-race
    # dependent, so bitwise-vs-reference is meaningless and the metadata
    # contract is what both sides must satisfy.
    assert_split_metadata(data, outputs)
    torch.testing.assert_close(
        outputs["split_counts"], reference_outputs["split_counts"], rtol=0, atol=0
    )


def prepare_bench(**config):
    """Compile the TIRx specialization without initializing CUDA."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    config.pop("label", None)
    state = {"config": dict(config), "executable": compile_kernel(get_kernel(**config))}
    return prepared_gpu_benchmark(run_gpu, state)


# ---------------------------------------------------------------------------
# Benchmark entry points.
#
# `split_counts` is read-modify-write and the kernel never resets it, so the
# first launch hands out slots `0..degree-1` and every later one starts at
# `degree` -- `split_slot < topk` then fails everywhere and the store path stops
# firing, leaving a strictly cheaper kernel being timed. The counters have to be
# zero at every launch.
#
# They cannot be zeroed per launch: that work would land inside the timed region,
# and the local bench API rejects a per-iteration `prepare` callback. So both
# sides rotate over pre-zeroed copies and re-zero the block once per rotation.
# Clearing N copies every N launches averages to clearing one copy per launch, so
# N does not change the cost -- it only bounds memory. The block is sized past
# this part's L2 so a wrap's fill evicts the slice it is about to hand out, and
# no slice is ever measured warmer than another.
#
# Do not pin `timer:` in the bench-suite config. `cudagraph_proton` captures the
# closure and replays it, so the rotation would advance only at capture and every
# replay would reuse one spent slice.
# ---------------------------------------------------------------------------
SPLIT_BLOCK_BYTES = 256 << 20


def _rotating_split_counts(data: dict[str, Any]):
    """A pre-zeroed `(slots, total_q, head_kv)` block plus its per-slot views."""
    import torch

    slice_elems = data["total_q"] * data["head_kv"]
    slots = max(8, min(4096, SPLIT_BLOCK_BYTES // max(slice_elems * 4, 1)))
    block = torch.zeros((slots, data["total_q"], data["head_kv"]), dtype=torch.int32, device="cuda")
    views = [block[i] for i in range(slots)]
    for view in views:
        if view.data_ptr() % 16 != 0:
            raise AssertionError("split_counts slice is not 16-byte aligned")
    return block, views, slots


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
    """Kernel-only comparison against MSA's compiled split-atomic launch."""
    from tirx_kernels.runner import bench

    config = {**prepared["config"], **config}
    config.pop("label", None)
    data = prepare_data(**config)
    executable = prepared["executable"]

    tirx_outputs = make_outputs(data)
    tirx_block, tirx_views, slots = _rotating_split_counts(data)
    tirx_base = tirx_args(data, tirx_outputs)
    tirx_step = [0]

    def tirx_launch():
        step = tirx_step[0]
        slot = step % slots
        if slot == 0 and step:
            tirx_block.zero_()
        tirx_step[0] = step + 1
        executable(*tirx_base[:5], tirx_views[slot].reshape(-1), *tirx_base[6:])

    def build_reference():
        from tirx_kernels.msa.utils._msa_bench import compiled_fwd_split_atomic

        reference_outputs = make_outputs(data)
        block, views, ref_slots = _rotating_split_counts(data)
        case = _reference_case(data, reference_outputs)
        compiled = compiled_fwd_split_atomic(case)
        step = [0]

        def reference_launch():
            index = step[0]
            slot = index % ref_slots
            if slot == 0 and index:
                block.zero_()
            step[0] = index + 1
            compiled(
                case["k2q_row_ptr"],
                case["k2q_q_indices"],
                case["scheduler_metadata"],
                case["work_count"],
                case["k2q_qsplit_indices"],
                views[slot],
                case["cu_seqlens_q"],
                case["work_capacity"],
                case["max_seqlen_q"],
                case["topk"],
            )

        # Pay the CuTeDSL compile and first-launch cost, then hand the timed run
        # a clean rotation -- otherwise slot 0 is already spent on entry.
        reference_launch()
        step[0] = 0
        block.zero_()
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
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_test",
]
