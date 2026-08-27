# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2024 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer clustered exact top-k port.

Ports the ``fast_topk_clusters_exact`` family
(``include/flashinfer/fast_topk_clusters_exact.cuh``): one device worker
``fast_topk_cuda_v4`` (``:80-405``) behind three thin ``__global__`` wrappers that
differ only in where the row length comes from and what the epilogue does to each
index -- plain (``:407-449``), page-table transform (``:451-494``) and ragged
transform (``:496-538``).

This is the path ``flashinfer.top_k`` takes on SM100 whenever the caller asks for
neither determinism nor a tie-break rule (``topk.py:502-507``).  The selection
happens in Python, *before* the FFI, so the C++ ``TopKDispatch`` that the four
sibling ports go through never reaches it.

The algorithm is an exact 256-bin MSB-first radix select over the monotone
ordered bits of the row -- four rounds for fp32, two for the 16-bit dtypes -- run
across a **thread-block cluster**.  Each CTA histograms its slice into shared
memory; the threshold bin is then chosen from the cluster-wide sum, which every
CTA assembles by reading its peers' histograms through distributed shared memory.
Candidates equal to the threshold bin are carried in a double-buffered shared
cache that spills to a per-CTA global ring, and are re-classified one byte at a
time until the deficit closes.  Two things are settled by DSMEM atomics into rank
0: which of the bit-equal boundary candidates make the cut, and where each CTA's
results land in the output row.

Three consequences shape the port.

* **The result set is exact; the result order is not.**  Output slots are handed
  out by atomic compaction, the per-CTA base comes from a racing atomic, and the
  boundary bin's survivors are whichever lanes arrive first (``:306-308``).
  Correctness therefore compares value multisets, never positions -- which loses
  nothing, because an exact radix select over the ordered bits admits exactly one
  multiset.
* **Peer shared memory is a different address space.**  An address produced by
  ``mapa`` lives in ``shared::cluster``; reading it with a plain ``ld.shared`` is
  an illegal instruction, not a wrong answer.
* **Two source hazards are reproduced, not repaired.**  The emit at ``:195-197``
  has its bounds check commented out, and ``shared_threshold_bin`` is never
  initialized or reset between rounds (``:166-170`` seeds only the other three
  scalars), so a round in which no lane satisfies the crossing test reads the
  previous round's value.

Out of scope, with the predicate that excludes each: ``PDL_ENABLED`` (``pdl``
defaults to ``False`` at every Python call site and no caller passes ``True``);
the pre-computed histogram branch (``pre_hist`` is always ``None``, ``:159-163``);
and ``num_clusters`` outside ``{1, 2, 4, 8}`` (the launcher degrades it to 1
before the kernel sees it, ``:607-611``).
"""

from typing import Any

import tirx_kernels.kern as K
from tirx_kernels.flashinfer.utils import topk_radix as R
from tirx_kernels.flashinfer.utils.filtered_topk_ops import st_global_bits
from tirx_kernels.flashinfer.utils.topk_harness import source_module, torch_dtype
from tirx_kernels.runner import bench

KERNEL_META = {"name": "fast_topk_clusters", "category": "flashinfer", "compute_capability": 10}


# `clusterCtaIdx.x` is requested only when the cluster is real; a one-CTA cluster
# takes the plain form, as the sibling cluster kernels do.  The tag alone does not
# produce a cluster launch -- the body must bind the scope.
# The source writes `__cluster_dims__(NClusters, 1, 1)` unconditionally, so its
# `NClusters == 1` entries still declare a one-CTA cluster: all 12 carry
# `.reqnctapercluster 1, 1, 1` and still read `%cluster_ctarank`.
#
# TIRx cannot reproduce that. An extent-1 `cta_id_in_cluster` binding folds away
# and the `clusterCtaIdx.x` tag then fails to resolve ("Cannot find thread var"),
# so the tag must be gated on `nc > 1` -- which is what every cluster kernel in
# this repo already does (`deepep/dispatch.py:192-196`,
# `flashinfer/norm/rmsnorm.py:845-848`). The port therefore diverges from the
# reference on exactly one launch attribute at `nc == 1`: no cluster dimension is
# declared where the reference declares a trivial one. Nothing in the algorithm
# depends on it -- at one CTA per cluster the peer sum and the epilogue range
# claim are both compiled out, and rank is statically 0.
def launch_tags(nc: int) -> list[str]:
    tags = ["blockIdx.x"]
    if nc > 1:
        tags.append("clusterCtaIdx.x")
    tags += ["threadIdx.x", "tirx.use_dyn_shared_memory"]
    return tags


# `__launch_bounds__(1024)` gives `.maxntid 1024` and nothing else: the export
# carries no `.minnctapersm` and no `.maxnreg`. Occupancy is already pinned at
# 2 blocks/SM by the 115188 B carve against a 232448 B optin, so a min-blocks
# bound would restate it and diverge.
LAUNCH_BOUNDS_MAX_THREADS = 1024

# --- source constants ------------------------------------------------------
# 256 bins, 8 bits per round (:93).
RADIX = 256
# `NRemainingRounds = sizeof(T) - 1`, `LShiftStart = 8 * sizeof(T) - 8` (:90-92).
BLOCK_THREADS = 1024
# The launcher instantiates exactly these cluster widths; anything else degrades
# to 1 before the kernel is selected (:598-611).
CLUSTER_WIDTHS = (1, 2, 4, 8)
# `get_fast_topk_clusters` (topk.py:390-400) plus the short-row clamp (:412-413).
SHORT_ROW_CLUSTER_LIMIT = 8192

DTYPES = ("float32", "float16", "bfloat16")
MODES = ("plain", "page_table", "ragged")
PATTERNS = ("unique", "tie_heavy", "all_equal", "neg", "rowvar")


def dtype_bytes(dtype: str) -> int:
    return 4 if dtype == "float32" else 2


def radix_rounds(dtype: str) -> int:
    """`NRemainingRounds + 1` -- the total number of 8-bit passes (:90)."""
    return dtype_bytes(dtype)


def shared_mem_bytes(top_k: int, num_cached: int) -> int:
    """`get_shared_mem_bytes` (:579-582)."""
    return 16 * num_cached + 4 * top_k + 3124


def num_cached_for(top_k: int, shared_optin: int) -> int:
    """`get_num_cached_for_topk` (topk.py:375-387).

    The host sizes the candidate cache to fill half the opt-in shared budget,
    which is what makes the dynamic request ~112 KiB independently of ``k``.
    """
    shared_per_block = shared_optin // 2
    buffers_used = (top_k + 5 + 3 * RADIX + 8) * 4
    return (shared_per_block - buffers_used - 1024) // 16


def clusters_for(batch_size: int, seq_len: int) -> int:
    """`get_fast_topk_clusters` + the short-row clamp (topk.py:390-400, 412-413)."""
    if seq_len < SHORT_ROW_CLUSTER_LIMIT:
        return 1
    if batch_size <= 32:
        return 8
    if batch_size < 128:
        return 4
    if batch_size < 256:
        return 2
    return 1


# Global memory goes through raw PTX, never BufferLoad/BufferStore: the repo's
# low-level IR contract rejects the latter outright (db entry
# `express-low-level-memory-access-through-raw-ptx`).
def _ld_nc32(buf, index):
    """One `ld.global.nc.b32`. `topk_radix.ld_global_u32` drops the `.nc`, and
    every read-only input here (`logits`, `page_table`, `offsets`, `seq_lens`) is
    `const __restrict__` on the source side, so the reference reads all of them
    non-coherently. Only the overflow ring is deliberately unqualified -- this
    kernel writes it, so its parameter cannot be `const`."""
    out = K.local_scalar("uint32")
    K.ptx.ld.global_.nc.b32(out, buf.ptr_to([index]))
    return out


def _ld_nc_bits(buf, index, is32):
    """The scalar mirror of `_ld_vec4`: `ld.global.nc.b32` | `ld.global.nc.b16`."""
    if is32:
        return _ld_nc32(buf, index)
    out = K.local_scalar("uint16")
    K.ptx.ld.global_.nc.b16(out, buf.ptr_to([index]))
    return out


def _ld_vec4(buf, elem_index, is32):
    """One `vec_t<T,4>` load, returned as raw 32-bit words.

    The reference emits `ld.global.nc.v4.b32` at f32 and `ld.global.nc.v2.b32` at
    16 bits (:227-236). `topk_radix.ld_global_words` spells the same widths
    WITHOUT `.nc`; `logits` is `const __restrict__`, so the qualified form is the
    faithful one, and it carries almost all of this kernel's memory traffic.
    """
    if is32:
        w = K.alloc_local([4], "uint32", align=16)
        K.ptx["ld.global.nc.v4.b32"](w[0], w[1], w[2], w[3], buf.ptr_to([elem_index]))
        return w
    w2 = K.alloc_local([2], "uint32", align=8)
    K.ptx["ld.global.nc.v2.b32"](w2[0], w2[1], buf.ptr_to([elem_index]))
    return w2


def _st_idx(buf, index, value, i64: bool):
    """One index store: `st.global.b64` on the i64 plain wrapper, else `b32`.

    A Python-int `value` (the `-1` pad) is converted at trace time. Two traps
    live here: reinterpreting a literal generates `*(uint *)(&(-1))`, which is
    not an lvalue, and `-1` as an unsigned 64-bit constant overflows a C long on
    the way in -- so the pad is written as a signed constant of the right width.
    """
    if isinstance(value, int):
        if i64:
            # `-1` cannot be written as a uint64 literal: the unsigned form
            # overflows a C long on the way in, and casting the signed constant
            # constant-folds into "cannot make uint from negative value". The
            # bit-complement of its magnitude produces the same pattern.
            imm = K.bitwise_not(K.uint64(-value - 1)) if value < 0 else K.uint64(value)
            K.ptx.st.global_.b64(buf.ptr_to([index]), imm)
        else:
            R.st_global_u32(buf, index, K.uint32(value & 0xFFFFFFFF))
    elif i64:
        K.ptx.st.global_.b64(buf.ptr_to([index]), K.cast(K.cast(value, "int64"), "uint64"))
    else:
        R.st_global_u32(buf, index, K.reinterpret("uint32", K.cast(value, "int32")))


def _ld_i32(buf, index):
    """One `ld.global.nc.b32` of an int32, returned as int32."""
    return K.cast(_ld_nc32(buf, index), "int32")


def get_kernel(
    dtype: str = "float32",
    batch: int = 16,
    seq_len: int = 16384,
    k: int = 256,
    mode: str = "plain",
    pattern: str = "unique",
    idx_dtype: str = "int32",
    **kwargs: Any,
):
    """One PrimFunc per reachable specialization of `fast_topk_cuda_v4`.

    `NClusters`, `TopK`, `num_cached` and the dtype are template parameters and
    launcher constants on the source side (:598-605, :555-576), so they are static
    here too; `mode` and the index width select the wrapper (:407/:451/:496).
    """
    is32 = dtype == "float32"
    rounds = 4 if is32 else 2  # NRemainingRounds + 1 (:90)
    lshift_start = 8 * (4 if is32 else 2) - 8  # (:91)
    nc = clusters_for(batch, seq_len)
    num_cached = num_cached_for_device(k)
    ovf_stride = seq_len // nc  # binding:47, from the row stride
    plain = mode == "plain"
    pt = mode == "page_table"
    ragged = mode == "ragged"
    i64 = plain and idx_dtype == "int64"
    idx_t = "int64" if i64 else "int32"
    val_t = dtype
    bits_t = "uint32" if is32 else "uint16"
    grid = batch * nc
    # Per-thread trip counts of the two element loops. The db entry
    # `scale-a-staged-load-unroll-to-the-trip-count` says a reference's unroll is
    # a statement about how many independent global loads it wants in flight, and
    # the factor must be derived from the trip count rather than copied.
    hist_trips = max(1, seq_len // (BLOCK_THREADS * nc))
    vec_trips = max(1, (seq_len // 4) // (BLOCK_THREADS * nc))
    # The unroll factor must SCALE with the per-thread trip count, not be a
    # single constant: measured on an idle GPU, 3-4 repeats each,
    #   trips  4 -> hu4 1.083/1.032   hu8 0.896/0.955   (small, medium)
    #   trips  8 -> hu4 0.977         hu8 0.995         (pt b200 l16384)
    #   trips 16 -> hu4 0.987/1.006   hu8 1.060/1.011   (rag b300, plain b64 l65536)
    #   trips 32 -> hu4 0.963         hu8 1.009         (plain b200 l65536)
    # hu6 was worse than both neighbours on every shape tested (0.959/0.924/0.969/
    # 0.970), so this is swept rather than interpolated -- the db entry
    # `scale-a-staged-load-unroll-to-the-trip-count` warns against assuming
    # monotonicity and that warning holds here.
    hist_unroll = int(kwargs.get("hist_unroll", 8 if hist_trips >= 8 else min(4, hist_trips)))
    vec_unroll = int(kwargs.get("vec_unroll", min(4, vec_trips)))
    # The trivial branch and the writeback are the two remaining element loops.
    # Both are bounded by TopK rather than seq_len, so their per-thread trip count
    # is k/(1024*nc) -- 4 on the `seq_len == k == 4096` shapes, which are exactly
    # the two worst rows and which never enter the worker at all.
    triv_trips = max(1, k // (BLOCK_THREADS * nc))
    triv_unroll = int(kwargs.get("triv_unroll", min(4, triv_trips)))
    wb_unroll = int(kwargs.get("wb_unroll", min(4, triv_trips)))

    # The 13 tail scalars, in the source's declaration order (:102-106):
    # final_idx_count[1], num_cached_count[2], threshold_bin[1],
    # cum_reduce_buf[8], k_remaining_counter[1].
    FINAL = 0
    NCACHED = 1
    THR = 3
    CUM0 = 4
    KREM = 12
    NSCAL = 13

    # The plain wrapper takes (logits, indices, values, overflow); the transforms
    # take (logits, indices, seq_lens, table_or_offsets, overflow). That arity
    # difference is the source's own (:410-413 vs :454-461 / :499-505), so the
    # entry points differ and the body is emitted by one shared function.
    def _threshold(hist, scal, tid, warp, lane, rank, bank, k_rem, sum_across):
        """`get_threshold_bin` (:116-154). Leaves the bin in `scal[THR]` and
        walks the caller's `k_rem` register down."""
        K.tvm_storage_sync("shared")

        v = K.local_scalar("int32", init=K.int32(0))
        with K.If(tid < RADIX), K.Then():
            K.assign(v, K.cast(R.ld_shared_u32(hist, bank * RADIX + tid), "int32"))
            # Five-step warp suffix scan (:21-31), unrolled. `.down` accumulates
            # toward lane 0, so the warp's total over its 32 bins lands THERE.
            # Each shuffle is materialized BEFORE its lane guard, so the
            # collective stays convergent across the whole warp.
            for delta, keep in ((1, 31), (2, 30), (4, 28), (8, 24), (16, 16)):
                peer = R.shfl_down_u32(K.reinterpret("uint32", v), delta)
                with K.If(lane < keep), K.Then():
                    K.assign(v, v + K.cast(peer, "int32"))
            with K.If(lane == 0), K.Then():
                R.st_shared_u32(scal, CUM0 + warp, K.reinterpret("uint32", v))
        K.tvm_storage_sync("shared")

        with K.If(warp == 0), K.Then():
            w = K.local_scalar("int32", init=K.int32(0))
            with K.If(lane < 8), K.Then():
                K.assign(w, K.cast(R.ld_shared_u32(scal, CUM0 + lane), "int32"))
            for delta, keep in ((1, 7), (2, 6), (4, 4)):
                peer = R.shfl_down_u32(K.reinterpret("uint32", w), delta)
                with K.If(lane < keep), K.Then():
                    K.assign(w, w + K.cast(peer, "int32"))
            # `__syncwarp` sits BETWEEN the scan and the write-back into the same
            # buffer (:48). After the store it would protect nothing.
            K.ptx.bar.warp.sync(K.uint32(0xFFFFFFFF))
            # The guard is load-bearing: cum_reduce_buf is 8 ints with
            # k_remaining_counter 32 B later, so an unguarded 32-lane store puts
            # 128 B where 32 are allocated (:49-51).
            with K.If(lane < 8), K.Then():
                R.st_shared_u32(scal, CUM0 + lane, K.reinterpret("uint32", w))
        K.tvm_storage_sync("shared")
        with K.If(warp < 7), K.Then():
            K.assign(v, v + K.cast(R.ld_shared_u32(scal, CUM0 + warp + 1), "int32"))

        # --- cluster-wide sum of the same bin (:120-138) --------------------
        if nc > 1 and sum_across:
            with K.If(tid < RADIX), K.Then():
                # This IS the DSMEM publication: peers read the PING-PONG bank.
                R.st_shared_u32(hist, bank * RADIX + tid, K.reinterpret("uint32", v))
            K.ptx.barrier.cluster.arrive()
            K.ptx.barrier.cluster.wait()
            with K.If(tid < RADIX), K.Then():
                with K.unroll(nc - 1) as c:
                    peer = K.alloc_local([1], "uint32")
                    K.ptx.mapa.shared__cluster.u32(
                        peer[0],
                        K.cuda.cvta_generic_to_shared(hist.ptr_to([bank * RADIX + tid])),
                        K.cast((c + rank + 1) % nc, "uint32"),
                    )
                    got = K.local_scalar("uint32")
                    K.ptx.ld.shared__cluster.b32(got, peer[0])
                    K.assign(v, v + K.cast(got, "int32"))
                R.st_shared_u32(hist, 2 * RADIX + tid, K.reinterpret("uint32", v))
        else:
            with K.If(tid < RADIX), K.Then():
                R.st_shared_u32(hist, 2 * RADIX + tid, K.reinterpret("uint32", v))
        K.tvm_storage_sync("shared")

        # --- pick the crossing bin (:142-152) --------------------------------
        nxt = K.local_scalar("int32", init=K.int32(0))
        with K.If(tid < RADIX - 1), K.Then():
            K.assign(nxt, K.cast(R.ld_shared_u32(hist, 2 * RADIX + tid + 1), "int32"))
        with K.If(tid < RADIX), K.Then():
            with K.If(K.And(v > k_rem, nxt <= k_rem)), K.Then():
                R.st_shared_u32(scal, THR, K.reinterpret("uint32", tid))
        K.tvm_storage_sync("shared")
        bin_ = K.local_scalar("int32", init=K.cast(R.ld_shared_u32(scal, THR), "int32"))
        with K.If(bin_ < RADIX - 1), K.Then():
            K.assign(k_rem, k_rem - K.cast(R.ld_shared_u32(hist, 2 * RADIX + bin_ + 1), "int32"))

    def _classify(
        hist,
        scal,
        topk_inds,
        cbits,
        cidx,
        ovf,
        bits,
        index,
        bin_,
        phase,
        shift,
        last,
        k_rem,
        exceeded,
        ring_base,
    ):
        """The three-way split of :189-217, shared by pass 1 and every round.

        The digit is snapshotted rather than re-emitted: it is read by both arms
        of the split, this runs once per element of a row that can be 512K long,
        and the source keeps it in a register too.
        """
        if is32:
            digit = K.bitwise_and(K.cast(K.shift_right(bits, K.uint32(shift)), "int32"), 0xFF)
        else:
            digit = K.bitwise_and(K.cast(K.shift_right(bits, K.uint16(shift)), "int32"), 0xFF)
        d = K.local_scalar("int32", init=digit)
        with K.If(d > bin_):
            with K.Then():
                # HAZARD: unguarded. The source's `if (topk_offset < TopK)` is
                # commented out at :195-197; the writeback's `offs < TopK` is what
                # bounds the global store. Reproduce, do not repair.
                slot = R.atom_shared_add_u32(scal, FINAL, K.uint32(1))
                R.st_shared_u32(topk_inds, K.cast(slot, "int32"), K.reinterpret("uint32", index))
            with K.Else():
                with K.If(d == bin_), K.Then():
                    if not last:
                        # Read five times below: the capacity test, both cache
                        # stores, and the ring offset.  Every boundary-bin
                        # element of the row takes this arm.
                        slot = K.local_scalar(
                            "int32",
                            init=K.cast(
                                R.atom_shared_add_u32(scal, NCACHED + phase, K.uint32(1)), "int32"
                            ),
                        )
                        keep = K.local_scalar("int32", init=K.int32(1))
                        with K.If(slot < num_cached):
                            with K.Then():
                                R.st_shared_u32(
                                    cidx, phase * num_cached + slot, K.reinterpret("uint32", index)
                                )
                                R.st_shared_u32(
                                    cbits, phase * num_cached + slot, K.cast(bits, "uint32")
                                )
                            with K.Else():
                                with K.If(slot - num_cached < ovf_stride):
                                    with K.Then():
                                        off = (
                                            ring_base
                                            + phase * ovf_stride * 2
                                            + (slot - num_cached) * 2
                                        )
                                        R.st_global_u32(ovf, off, K.cast(bits, "uint32"))
                                        R.st_global_u32(
                                            ovf, off + 1, K.reinterpret("uint32", index)
                                        )
                                    with K.Else():
                                        # ring full: dropped (:210-214)
                                        K.assign(keep, K.int32(0))
                        with K.If(keep == 1), K.Then():
                            if is32:
                                nb = K.bitwise_and(
                                    K.cast(K.shift_right(bits, K.uint32(shift - 8)), "int32"), 0xFF
                                )
                            else:
                                nb = K.bitwise_and(
                                    K.cast(K.shift_right(bits, K.uint16(shift - 8)), "int32"), 0xFF
                                )
                            R.atom_shared_add_u32(hist, (phase ^ 1) * RADIX + nb, K.uint32(1))
                    else:
                        # Final-round tie rationing on rank 0's counter (:300-319).
                        # NOT guarded by nc > 1. The source calls
                        # `map_shared_rank(s_k_remaining_counter, 0)`
                        # unconditionally (:309, :360) and its NC==1 export still
                        # carries the cluster atomic (2 at f32, 10 at 16-bit).
                        # Only the peer sum (:121-131) and the epilogue claim
                        # (:379-399) sit inside `if (NClusters > 1)`.
                        # probe_nc1_atom.py confirms the cluster-space atomic
                        # works with no `clusterCtaIdx.x` tag bound, so at one
                        # CTA per cluster rank 0 is this CTA and the mapping is
                        # the identity.
                        with K.If(K.And(k_rem > 0, exceeded == 0)), K.Then():
                            peer = K.local_scalar("uint32")
                            K.ptx.mapa.shared__cluster.u32(
                                peer,
                                K.cuda.cvta_generic_to_shared(scal.ptr_to([KREM])),
                                K.uint32(0),
                            )
                            got = K.local_scalar("uint32")
                            K.ptx.atom.shared__cluster.add.u32(got, peer, K.uint32(1))
                            with K.If(K.cast(got, "int32") < k_rem):
                                with K.Then():
                                    slot = R.atom_shared_add_u32(scal, FINAL, K.uint32(1))
                                    R.st_shared_u32(
                                        topk_inds,
                                        K.cast(slot, "int32"),
                                        K.reinterpret("uint32", index),
                                    )
                                with K.Else():
                                    K.assign(exceeded, K.int32(1))

    def _round(
        hist,
        scal,
        topk_inds,
        cached_bits,
        cached_idx,
        ovf,
        tid,
        warp,
        lane,
        rank,
        ring_base,
        k_rem,
        exceeded,
        t,
    ):
        """One refinement round (:241-373).

        `t` arrives as a PYTHON int, so `phase`, `shift` and `last` are
        compile-time constants and the round is emitted as straight-line code.
        A `K.serial` over the rounds instead would make `last` a runtime
        predicate, collapse the four static `threshold()` regions into two, and
        leave the per-round `digit` forms unreachable.
        """
        phase = t % 2
        K.assign(exceeded, K.int32(0))  # :268 -- per round, per thread
        if nc > 1:
            K.ptx.barrier.cluster.wait()
        with K.If(tid < RADIX), K.Then():
            R.st_shared_u32(hist, (phase ^ 1) * RADIX + tid, K.uint32(0))
        with K.If(tid == 0), K.Then():
            R.st_shared_u32(scal, NCACHED + phase, K.uint32(0))
        _threshold(hist, scal, tid, warp, lane, rank, phase, k_rem, True)
        bin_t = K.local_scalar("int32", init=K.cast(R.ld_shared_u32(scal, THR), "int32"))
        raw = K.local_scalar(
            "int32", init=K.cast(R.ld_shared_u32(scal, NCACHED + (phase ^ 1)), "int32")
        )
        n_sh = K.local_scalar("int32", init=K.min(K.int32(num_cached), raw))
        n_gl = K.local_scalar(
            "int32", init=K.min(K.int32(ovf_stride), K.max(K.int32(0), raw - num_cached))
        )
        shift = lshift_start - t * 8
        last = t == rounds - 1

        # Stride 1024, NOT 1024*nc: each CTA owns its own candidate slice and
        # the cluster does not re-partition here (:271-273).
        with K.serial(tid, n_sh, step=BLOCK_THREADS) as i:
            _classify(
                hist,
                scal,
                topk_inds,
                cached_bits,
                cached_idx,
                ovf,
                R.ld_shared_u32(cached_bits, (phase ^ 1) * num_cached + i),
                K.local_scalar(
                    "int32",
                    init=K.cast(R.ld_shared_u32(cached_idx, (phase ^ 1) * num_cached + i), "int32"),
                ),
                bin_t,
                phase,
                shift,
                last,
                k_rem,
                exceeded,
                ring_base,
            )
        with K.serial(tid, n_gl, step=BLOCK_THREADS) as i2:
            off = ring_base + (phase ^ 1) * ovf_stride * 2 + i2 * 2
            _classify(
                hist,
                scal,
                topk_inds,
                cached_bits,
                cached_idx,
                ovf,
                R.ld_global_u32(ovf, off),
                K.local_scalar("int32", init=K.cast(R.ld_global_u32(ovf, off + 1), "int32")),
                bin_t,
                phase,
                shift,
                last,
                k_rem,
                exceeded,
                ring_base,
            )

        if nc > 1 and t < rounds - 1:
            # Skipped on the final round: no further threshold() means no
            # further peer read of this bank (:260-263).  Publish only after
            # every thread has consumed the previous ping-pong bank: arriving
            # immediately after threshold selection lets the next round
            # overwrite that bank while another warp is still reading its
            # cached candidates.
            K.ptx.barrier.cluster.arrive()

    def _emit(logits, out_idx, out_val, seq_lens_g, aux, ovf):
        cta = K.cta_id()
        rank = K.cta_id_in_cluster([nc]) if nc > 1 else K.int32(0)
        tid = K.thread_id()
        row = cta // nc
        warp = tid >> 5
        lane = tid & 31

        # Per-CTA invariants that every element loop indexes off; the source
        # holds both in registers rather than recomputing `row * stride`.
        logit_base = K.local_scalar("int32", init=row * seq_len)
        ind_base = K.local_scalar("int32", init=row * k)

        # The worker's shared arena is declared for the whole entry rather than
        # inside the `row_len > TopK` arm: a traced kernel owns one pool, and the
        # dynamic request is a launch property either way.  Nothing in the
        # trivial branch touches it.
        pool = K.smem_pool()
        cached_bits = pool.alloc((2 * num_cached,), "uint32", align=128)
        cached_idx = pool.alloc((2 * num_cached,), "int32")
        topk_inds = pool.alloc((k,), "int32")
        hist = pool.alloc((3 * RADIX,), "int32")
        scal = pool.alloc((NSCAL,), "int32")

        row_len = K.local_scalar("int32", init=K.int32(seq_len))
        if not plain:
            K.assign(row_len, _ld_i32(seq_lens_g, row))
        rag_off = K.local_scalar("int32", init=K.int32(0))
        # `page_table_offset` is hoisted ONCE per CTA in the source (:465) and
        # reused by both the trivial branch (:471) and the writeback (:490).
        # Recomputing `row * pt_stride` inside those loops costs a multiply-add
        # per element on the one path that has no other per-element arithmetic,
        # which is exactly where the page_table trivial shape loses to ragged.
        pt_base = K.local_scalar("int32", init=K.int32(0))
        if pt:
            K.assign(pt_base, row * seq_len)
        if ragged:
            # Hoisted above the trivial branch exactly as the source does (:510)
            # and reused by both it and the writeback.
            K.assign(rag_off, _ld_i32(aux, row))

        # ---- trivial branch (:417-429 / :467-476 / :511-520) ---------------
        with K.If(row_len <= k):
            with K.Then():
                if pt and triv_trips * BLOCK_THREADS * nc == k:
                    # Stage the page-table loads into DISTINCT registers before
                    # any store. `unroll=N` on a real loop reuses one destination
                    # local across the unrolled copies, so the loads serialize:
                    # each waits for the previous store to consume the register.
                    # Paired NCU on this shape measured long_scoreboard 22.48 vs
                    # the reference's 7.02 at IDENTICAL global load sectors (4352
                    # both) -- exposed load latency, not extra traffic. A
                    # trace-time unrolled body with one register per load lets
                    # all `triv_trips` be in flight.
                    stage = K.alloc_local([triv_trips], "int32")
                    tbase = tid + (cta % nc) * BLOCK_THREADS
                    for u in range(triv_trips):
                        iu = tbase + u * BLOCK_THREADS * nc
                        with K.If(iu < row_len):
                            with K.Then():
                                K.assign(stage[u], _ld_i32(aux, pt_base + iu))
                            with K.Else():
                                K.assign(stage[u], K.int32(-1))
                    for u in range(triv_trips):
                        iu = tbase + u * BLOCK_THREADS * nc
                        _st_idx(out_idx, ind_base + iu, stage[u], i64)
                else:
                    with K.serial(
                        tid + (cta % nc) * BLOCK_THREADS,
                        k,
                        step=BLOCK_THREADS * nc,
                        unroll=triv_unroll,
                    ) as i:
                        with K.If(i < row_len):
                            with K.Then():
                                if plain:
                                    _st_idx(out_idx, ind_base + i, i, i64)
                                    st_global_bits(
                                        out_val,
                                        ind_base + i,
                                        _ld_nc_bits(logits, logit_base + i, is32),
                                        is32,
                                    )
                                elif pt:
                                    _st_idx(out_idx, ind_base + i, _ld_i32(aux, pt_base + i), i64)
                                else:
                                    _st_idx(out_idx, ind_base + i, i + rag_off, i64)
                            with K.Else():
                                # output_values is left UNTOUCHED over the pad
                                # region (:425-427) -- the port must not zero it.
                                _st_idx(out_idx, ind_base + i, -1, i64)
            with K.Else():
                # Per-CTA constant, but every spilling element reads it from
                # inside the classification loops.
                ring_base = K.local_scalar("int32", init=cta * ovf_stride * 4)
                exceeded = K.local_scalar("int32", init=K.int32(0))

                # ---- zero banks 0/1 and three scalars (:158-170) ------------
                with K.If(tid < RADIX), K.Then():
                    R.st_shared_u32(hist, tid, K.uint32(0))
                    R.st_shared_u32(hist, RADIX + tid, K.uint32(0))
                with K.If(tid == 0), K.Then():
                    R.st_shared_u32(scal, FINAL, K.uint32(0))
                    R.st_shared_u32(scal, KREM, K.uint32(0))
                    R.st_shared_u32(scal, NCACHED, K.uint32(0))
                    # threshold_bin is deliberately NOT initialized (:166-170): a
                    # round whose crossing test selects no lane reads the
                    # previous round's value. Reproduce, do not repair.
                K.tvm_storage_sync("shared")

                # ---- first histogram pass: SCALAR strided (:174-179) --------
                with K.serial(
                    tid + rank * BLOCK_THREADS, row_len, step=BLOCK_THREADS * nc, unroll=hist_unroll
                ) as i:
                    xb = _ld_nc_bits(logits, logit_base + i, is32)
                    ob = K.local_scalar(
                        bits_t, init=R.to_ordered_u32(xb) if is32 else R.to_ordered_u16(xb)
                    )
                    if is32:
                        d0 = K.bitwise_and(
                            K.cast(K.shift_right(ob, K.uint32(lshift_start)), "int32"), 0xFF
                        )
                    else:
                        d0 = K.bitwise_and(
                            K.cast(K.shift_right(ob, K.uint16(lshift_start)), "int32"), 0xFF
                        )
                    R.atom_shared_add_u32(hist, d0, K.uint32(1))

                k_rem = K.local_scalar("int32", init=K.int32(k))
                _threshold(hist, scal, tid, warp, lane, rank, 0, k_rem, True)
                bin0 = K.local_scalar("int32", init=K.cast(R.ld_shared_u32(scal, THR), "int32"))

                if nc > 1:
                    # SPLIT barrier: the matching wait heads round 1 (:218-225, :248).
                    K.ptx.barrier.cluster.arrive()

                # Vectorized classification (:227-236): four elements per issue,
                # then a scalar tail (:237-239). Only THIS pass is vectorized --
                # the first histogram pass above is a plain strided scalar read,
                # which is why the row is read from global twice in total.
                vbase = (rank * BLOCK_THREADS + tid) * 4
                # Snapshotted: as a loop bound it is re-evaluated every trip.
                vec_end = K.local_scalar("int32", init=(row_len // 4) * 4)
                with K.serial(vbase, vec_end, step=BLOCK_THREADS * nc * 4, unroll=vec_unroll) as i:
                    w = _ld_vec4(logits, logit_base + i, is32)
                    with K.unroll(4) as j:
                        if is32:
                            eb = w[j]
                        else:
                            eb = K.cast(
                                K.bitwise_and(
                                    K.shift_right(w[j // 2], K.uint32(16 * (j % 2))),
                                    K.uint32(0xFFFF),
                                ),
                                "uint16",
                            )
                        _classify(
                            hist,
                            scal,
                            topk_inds,
                            cached_bits,
                            cached_idx,
                            ovf,
                            K.local_scalar(
                                bits_t, init=R.to_ordered_u32(eb) if is32 else R.to_ordered_u16(eb)
                            ),
                            i + j,
                            bin0,
                            0,
                            lshift_start,
                            False,
                            k_rem,
                            exceeded,
                            ring_base,
                        )
                with K.serial(
                    vec_end + rank * BLOCK_THREADS + tid, row_len, step=BLOCK_THREADS * nc
                ) as i3:
                    xb3 = _ld_nc_bits(logits, logit_base + i3, is32)
                    _classify(
                        hist,
                        scal,
                        topk_inds,
                        cached_bits,
                        cached_idx,
                        ovf,
                        K.local_scalar(
                            bits_t, init=R.to_ordered_u32(xb3) if is32 else R.to_ordered_u16(xb3)
                        ),
                        i3,
                        bin0,
                        0,
                        lshift_start,
                        False,
                        k_rem,
                        exceeded,
                        ring_base,
                    )

                # ---- refinement rounds (:241-373), emitted as straight-line
                # code so each round's phase/shift/last stay compile-time,
                # matching the source's `#pragma unroll` (:241).
                for t in range(1, rounds):
                    _round(
                        hist,
                        scal,
                        topk_inds,
                        cached_bits,
                        cached_idx,
                        ovf,
                        tid,
                        warp,
                        lane,
                        rank,
                        ring_base,
                        k_rem,
                        exceeded,
                        t,
                    )

                # ---- epilogue: claim the output range (:374-404) ------------
                K.tvm_storage_sync("shared")
                n_out = K.local_scalar("int32", init=K.int32(k))
                out_start = K.local_scalar("int32", init=K.int32(0))
                if nc > 1:
                    # Read own count BEFORE rank 0's cursor absorbs peers (:379-386).
                    my_num = K.local_scalar(
                        "int32", init=K.cast(R.ld_shared_u32(scal, FINAL), "int32")
                    )
                    K.ptx.barrier.cluster.arrive()
                    K.ptx.barrier.cluster.wait()
                    with K.If(rank > 0), K.Then():
                        with K.If(tid == 0), K.Then():
                            peer = K.local_scalar("uint32")
                            K.ptx.mapa.shared__cluster.u32(
                                peer,
                                K.cuda.cvta_generic_to_shared(scal.ptr_to([FINAL])),
                                K.uint32(0),
                            )
                            old = K.local_scalar("uint32")
                            K.ptx.atom.shared__cluster.add.u32(
                                old, peer, K.reinterpret("uint32", my_num)
                            )
                            R.st_shared_u32(scal, FINAL, old)  # broadcast (:392)
                        K.tvm_storage_sync("shared")
                        K.assign(out_start, K.cast(R.ld_shared_u32(scal, FINAL), "int32"))
                    K.ptx.barrier.cluster.arrive()
                    K.ptx.barrier.cluster.wait()
                    K.assign(n_out, K.min(K.int32(k), my_num))

                # ---- writeback (:438-447 / :486-492 / :530-536) -------------
                with K.serial(tid, n_out, step=BLOCK_THREADS, unroll=wb_unroll) as i4:
                    offs = i4 + out_start
                    with K.If(offs < k), K.Then():
                        ind = K.cast(R.ld_shared_u32(topk_inds, i4), "int32")
                        if plain:
                            _st_idx(out_idx, ind_base + offs, ind, i64)
                            # The value is RE-GATHERED from global, never carried
                            # through shared (:444).
                            st_global_bits(
                                out_val,
                                ind_base + offs,
                                _ld_nc_bits(logits, logit_base + ind, is32),
                                is32,
                            )
                        elif pt:
                            _st_idx(out_idx, ind_base + offs, _ld_i32(aux, pt_base + ind), i64)
                        else:
                            _st_idx(out_idx, ind_base + offs, ind + rag_off, i64)

    if plain:

        @K.kernel(warps=BLOCK_THREADS // 32, arch="sm_100a", grid=grid)
        def fast_topk_clusters_kernel(
            logits: K.gptr[val_t, (batch * seq_len,)],
            indices: K.gptr[idx_t, (batch * k,)],
            values: K.gptr[val_t, (batch * k,)],
            overflow: K.gptr[K.i32, (batch * 4 * ovf_stride * nc,)],
        ):
            _emit(logits, indices, values, None, None, overflow)

    else:

        @K.kernel(warps=BLOCK_THREADS // 32, arch="sm_100a", grid=grid)
        def fast_topk_clusters_kernel(
            logits: K.gptr[val_t, (batch * seq_len,)],
            indices: K.gptr[idx_t, (batch * k,)],
            seq_lens: K.gptr[K.i32, (batch,)],
            aux: K.gptr[K.i32, (batch * seq_len if pt else batch,)],
            overflow: K.gptr[K.i32, (batch * 4 * ovf_stride * nc,)],
        ):
            _emit(logits, indices, None, seq_lens, aux, overflow)

    return fast_topk_clusters_kernel.func.with_attr("tirx.launch_tags", launch_tags(nc))


# ---------------------------------------------------------------------------
# Harness.
# ---------------------------------------------------------------------------
def _row_lengths(batch: int, seq_len: int, k: int, pattern: str, device):
    """Per-row `seq_lens` for the transform modes.

    `rowvar` deliberately mixes rows that take the trivial branch with rows that
    do not, in one launch: the branch is per row on the transforms (`:466`,
    `:511`) but per launch on plain.
    """
    import torch

    if pattern == "rowvar":
        lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
        lens[0::3] = min(k, seq_len)  # short enough for the trivial branch
        lens[1::3] = max(1, seq_len // 2)
        return lens
    return torch.full((batch,), seq_len, dtype=torch.int32, device=device)


def prepare_data(
    dtype: str = "float32",
    batch: int = 16,
    seq_len: int = 16384,
    k: int = 256,
    mode: str = "plain",
    pattern: str = "unique",
    idx_dtype: str = "int32",
    **kwargs: Any,
):
    """One launch's inputs, outputs, and scratch.

    Selection here is exact over the ordered bits, so the only data-dependent
    behavior is what lands in the boundary bin.  The patterns are chosen for that
    path: `tie_heavy` and `all_equal` are what push the candidate cache past
    `num_cached` into the global overflow ring and then into the cluster-wide tie
    race, and `neg` takes the other half of the monotone-flip sign branch.

    Every tensor is materialized here, including the overflow scratch, so that
    nothing is allocated inside a timed closure.
    """
    import torch

    device = "cuda"
    g = torch.Generator(device=device).manual_seed(1234)

    if pattern in ("unique", "rowvar"):
        logits = torch.randn(batch, seq_len, dtype=torch.float32, device=device, generator=g) * 4
    elif pattern == "tie_heavy":
        # Few distinct values: most of the row collides in one bin, so the
        # boundary bin overflows and the tie race decides the tail.
        logits = torch.randint(
            0, 8, (batch, seq_len), dtype=torch.int32, device=device, generator=g
        ).to(torch.float32)
    elif pattern == "all_equal":
        # The entire row is one bin: every selected element is a tie winner.
        logits = torch.full((batch, seq_len), 1.5, dtype=torch.float32, device=device)
    elif pattern == "neg":
        logits = -(
            torch.rand(batch, seq_len, dtype=torch.float32, device=device, generator=g) * 8 + 0.5
        )
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    logits = logits.to(torch_dtype(dtype)).contiguous()

    idx_torch = torch.int64 if idx_dtype == "int64" else torch.int32
    data: dict[str, Any] = {
        "logits": logits,
        "batch": batch,
        "seq_len": seq_len,
        "k": k,
        "mode": mode,
        "dtype": dtype,
        "idx_dtype": idx_dtype,
        "num_clusters": clusters_for(batch, seq_len),
        "idx_torch": idx_torch,
    }

    if mode != "plain":
        data["seq_lens"] = _row_lengths(batch, seq_len, k, pattern, device)
    if mode == "page_table":
        # A per-row permutation.  The source's own tests use `randint`, which
        # repeats entries; here the table must be invertible, because the only
        # way to check the selection is to map an output index back to the row
        # position it came from and look at the logit there.  A permutation keeps
        # the mapping non-identity -- so a missing lookup still fails -- while
        # making that inverse exact.
        data["page_table"] = (
            torch.argsort(torch.rand(batch, seq_len, device=device, generator=g), dim=1)
            .to(torch.int32)
            .contiguous()
        )
    if mode == "ragged":
        data["offsets"] = (
            torch.arange(batch, dtype=torch.int32, device=device) * seq_len
        ).contiguous()

    return data


# `shared_memory_per_block_optin` on every SM100 part. Hard-coded rather than
# queried because `get_kernel` runs in the bench suite's CPU prepare stage, which
# fails the workload outright if it initializes CUDA ("CPU prepare changed CUDA
# initialization state from False to True"). The value is asserted against the
# live device in `prepare_data`, so a wrong constant cannot pass silently.
SM100_SHARED_OPTIN = 232448


def num_cached_for_device(k: int) -> int:
    """`get_num_cached_for_topk` (topk.py:375-387), without touching the device."""
    return num_cached_for(k, SM100_SHARED_OPTIN)


def alloc_outputs(data: dict[str, Any]):
    """Outputs plus the overflow ring, sized exactly as the Python layer sizes it."""
    import torch

    assert (
        torch.cuda.get_device_properties(0).shared_memory_per_block_optin == SM100_SHARED_OPTIN
    ), "SM100_SHARED_OPTIN disagrees with the live device; num_cached would diverge"

    batch, k = data["batch"], data["k"]
    device = data["logits"].device
    nc = data["num_clusters"]
    out: dict[str, Any] = {"indices": torch.empty(batch, k, dtype=data["idx_torch"], device=device)}
    if data["mode"] == "plain":
        out["values"] = torch.empty(batch, k, dtype=data["logits"].dtype, device=device)
    # topk.py:414-420 -- `overflow_stride` is recovered by the binding from this
    # tensor's row stride, so the shape is load-bearing, not just a capacity.
    out["overflow"] = torch.empty(
        batch, 4 * (data["seq_len"] // nc) * nc, dtype=torch.int32, device=device
    )
    return out


def run_reference(data: dict[str, Any], out: dict[str, Any]) -> None:
    """Drive the source's own FFI for this mode.

    The entries are absent from `dir(module)`; they must be taken by name.
    `num_clusters` and `num_cached` are passed explicitly so both sides run the
    same specialization rather than whatever the Python heuristic would pick.
    """
    module = source_module()
    nc = data["num_clusters"]
    ncache = num_cached_for_device(data["k"])
    mode = data["mode"]

    if mode == "plain":
        fn = getattr(module, "fast_topk_clusters_exact")
        fn(
            data["logits"],
            out["indices"],
            out["values"],
            None,
            out["overflow"],
            data["k"],
            ncache,
            nc,
            False,
        )
    elif mode == "page_table":
        fn = getattr(module, "fast_topk_clusters_exact_page_table_transform")
        fn(
            data["logits"],
            out["indices"],
            data["seq_lens"],
            data["page_table"],
            None,
            out["overflow"],
            data["k"],
            ncache,
            nc,
            False,
        )
    else:
        fn = getattr(module, "fast_topk_clusters_exact_ragged_transform")
        fn(
            data["logits"],
            out["indices"],
            data["seq_lens"],
            data["offsets"],
            None,
            out["overflow"],
            data["k"],
            ncache,
            nc,
            False,
        )


def local_indices(data: dict[str, Any], indices, row: int):
    """Undo the mode's index transform, giving indices into the row itself."""
    mode = data["mode"]
    if mode == "plain":
        return indices
    if mode == "ragged":
        return indices - int(data["offsets"][row].item())
    # page_table: invert the permutation built in prepare_data
    import torch

    table = data["page_table"][row].long()
    inverse = torch.empty_like(table)
    inverse[table] = torch.arange(table.numel(), device=table.device)
    return inverse[indices.long()]


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        "dtype": "float32",
        "batch": 16,
        "seq_len": 16384,
        "k": 256,
        "mode": "plain",
        "pattern": "unique",
        "idx_dtype": "int32",
    }
    cfg.update({key: value for key, value in config.items() if key != "label"})
    return cfg


def build_tirx_args(data: dict[str, Any], out: dict[str, Any]):
    """Bind the flat launch ABI once, outside any timed region."""
    args = [data["logits"].reshape(-1), out["indices"].reshape(-1)]
    if data["mode"] == "plain":
        args.append(out["values"].reshape(-1))
    else:
        args.append(data["seq_lens"])
    if data["mode"] == "page_table":
        args.append(data["page_table"].reshape(-1))
    elif data["mode"] == "ragged":
        args.append(data["offsets"])
    args.append(out["overflow"].reshape(-1))
    return tuple(args)


def compare_outputs(data: dict[str, Any], mine: dict[str, Any], theirs: dict[str, Any]) -> None:
    """Both sides are nondeterministic in order; compare what is well defined.

    The selected *set* is exact on this path, so the comparison is bit-exact on
    value multisets rather than tolerant on positions.  The trivial branch is the
    exception: identity indices with ``-1`` padding are deterministic, so those
    rows are compared positionally against the reference.
    """
    import torch

    k, mode = data["k"], data["mode"]
    for row in range(data["batch"]):
        row_len = int(data["seq_lens"][row].item()) if mode != "plain" else data["seq_len"]
        if row_len <= k:
            assert torch.equal(mine["indices"][row], theirs["indices"][row]), (
                f"row {row}: the trivial branch is deterministic and must match positionally"
            )
            continue
        local = local_indices(data, mine["indices"][row], row).long()
        assert int(local.min()) >= 0 and int(local.max()) < row_len, (
            f"row {row}: index out of range [0, {row_len})"
        )
        assert len(set(local.tolist())) == local.numel(), f"row {row}: duplicate indices"
        ours = torch.sort(
            data["logits"][row, local_indices(data, mine["indices"][row], row).long()].float()
        ).values
        ref = torch.sort(
            data["logits"][row, local_indices(data, theirs["indices"][row], row).long()].float()
        ).values
        assert torch.equal(ours, ref), f"row {row}: value multiset differs from the reference"
        if mode == "plain":
            gathered = data["logits"][row, mine["indices"][row].long()]
            assert torch.equal(mine["values"][row], gathered), (
                f"row {row}: output_values is not the gather of its own indices"
            )


def run_test(**config: Any) -> None:
    import unittest

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise unittest.SkipTest(f"torch unavailable: {exc}") from exc
    if not torch.cuda.is_available():  # pragma: no cover
        raise unittest.SkipTest("CUDA unavailable")

    from tirx_kernels.runner import compile_kernel

    cfg = _normalize_config(config)
    data = prepare_data(**cfg)

    mine = alloc_outputs(data)
    ex = compile_kernel(get_kernel(**cfg))
    ex(*build_tirx_args(data, mine))
    torch.cuda.synchronize()

    theirs = alloc_outputs(data)
    run_reference(data, theirs)
    torch.cuda.synchronize()

    compare_outputs(data, mine, theirs)


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU.

    The reference is NOT built here. `source_module()` JITs the FlashInfer module,
    which initializes CUDA, and the suite fails any workload whose CPU prepare
    does that ("CPU prepare changed CUDA initialization state from False to
    True"). It is built by the lazy `references` builder in `run_gpu` instead,
    which is what the bench API expects and what the sibling ports do.
    """
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    cfg = _normalize_config(kwargs)
    return prepared_gpu_benchmark(
        run_gpu, {"config": cfg, "executable": compile_kernel(get_kernel(**cfg))}
    )


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs):
    """Kernel-only comparison against the source launch.

    Both implementations ALTERNATE over the same two working sets in opposite
    phase, so each spends half its calls on each buffer. This is the bench-suite
    README's cancellation for the per-implementation allocation draw, and it is
    not optional here: giving each side its own outputs made this kernel's
    smallest shape read 0.725x-0.900x across repeats with no code change, with
    the tirx side alone swinging 8.2us to 14.3us while the reference held steady.

    An earlier version of this function reasoned the effect could not bite
    because the input row is shared and only the outputs are per-side. That was
    wrong: the overflow ring is about 1 MB per side at the small shape, it is
    both written and read back during the refinement rounds, and where it lands
    dominates a kernel this short.

    Every tensor is allocated before the closures are built. Allocating inside
    one would enqueue fill kernels that a per-kernel timer charges here.
    """
    cfg = dict(prepared["config"])
    ex = prepared["executable"]

    data = prepare_data(**cfg)
    set_a, set_b = alloc_outputs(data), alloc_outputs(data)
    args_a, args_b = build_tirx_args(data, set_a), build_tirx_args(data, set_b)

    phase = [0]

    def tirx_launch():
        ex(*(args_a if phase[0] & 1 == 0 else args_b))
        phase[0] += 1

    def build_reference():
        module = source_module()
        nc = data["num_clusters"]
        ncache = num_cached_for_device(data["k"])
        k, mode = data["k"], data["mode"]

        def ref_args(out):
            if mode == "plain":
                return (
                    data["logits"],
                    out["indices"],
                    out["values"],
                    None,
                    out["overflow"],
                    k,
                    ncache,
                    nc,
                    False,
                )
            if mode == "page_table":
                return (
                    data["logits"],
                    out["indices"],
                    data["seq_lens"],
                    data["page_table"],
                    None,
                    out["overflow"],
                    k,
                    ncache,
                    nc,
                    False,
                )
            return (
                data["logits"],
                out["indices"],
                data["seq_lens"],
                data["offsets"],
                None,
                out["overflow"],
                k,
                ncache,
                nc,
                False,
            )

        name = {
            "plain": "fast_topk_clusters_exact",
            "page_table": "fast_topk_clusters_exact_page_table_transform",
            "ragged": "fast_topk_clusters_exact_ragged_transform",
        }[mode]
        fn = getattr(module, name)
        # OPPOSITE phase to tirx: b, a, b, a ... against tirx's a, b, a, b ...
        rargs_a, rargs_b = ref_args(set_a), ref_args(set_b)
        rphase = [0]

        def reference_launch():
            fn(*(rargs_b if rphase[0] & 1 == 0 else rargs_a))
            rphase[0] += 1

        return reference_launch

    return bench(
        {"tirx": tirx_launch},
        references={"flashinfer": build_reference},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(**config: Any):
    prepared = prepare_bench(**config)
    return prepared.run_gpu()


# ---------------------------------------------------------------------------
# Config matrix.
# ---------------------------------------------------------------------------
_DT_TAG = {"float32": "f32", "float16": "f16", "bfloat16": "bf16"}
_MODE_TAG = {"plain": "plain", "page_table": "pt", "ragged": "rag"}


def _cfg(dtype, batch, seq_len, k, mode="plain", pattern="unique", idx_dtype="int32"):
    label = f"{_DT_TAG[dtype]}_{_MODE_TAG[mode]}_b{batch}_l{seq_len}_k{k}"
    if pattern != "unique":
        label += f"_{pattern}"
    if idx_dtype != "int32":
        label += "_i64"
    return {
        "label": label,
        "dtype": dtype,
        "batch": batch,
        "seq_len": seq_len,
        "k": k,
        "mode": mode,
        "pattern": pattern,
        "idx_dtype": idx_dtype,
    }


def _build_configs():
    """Representative cover of the reachable domain.

    The axes are not crossed.  ``num_clusters`` is not a free parameter -- the
    host derives it from ``(batch, seq_len)`` -- so cluster coverage is obtained
    by choosing batch sizes that land on each rung of that heuristic, and the
    ``seq_len < 8192`` clamp is exercised on its own.

    Retained in full: every mode x dtype pair; every cluster width the launcher
    can instantiate; both index widths on the mode that supports them; the
    trivial ``seq_len <= k`` branch on every mode; and every input pattern that
    can change which elements are selected.
    """
    configs = []

    # 1. Every mode x dtype at one carrier shape (batch 16 -> 8 clusters).
    for mode in MODES:
        for dtype in DTYPES:
            configs.append(_cfg(dtype, 16, 16384, 256, mode=mode))

    # 2. Cluster width.  The heuristic is batch<=32 -> 8, <128 -> 4, <256 -> 2,
    #    else 1, and any seq_len < 8192 clamps to 1 regardless of batch.
    for batch in (16, 64, 200, 300):
        configs.append(_cfg("float32", batch, 16384, 256))
    configs.append(_cfg("float32", 16, 4096, 256))  # clamp to one cluster
    configs.append(_cfg("float16", 16, 4096, 256))

    # 3. k ladder, and the largest row the tests exercise.
    for k in (256, 1024, 2048):
        configs.append(_cfg("float32", 64, 16384, k))
    configs.append(_cfg("float32", 64, 65536, 1024))
    configs.append(_cfg("bfloat16", 64, 65536, 1024))

    # 4. int64 indices -- plain only, the width `flashinfer.top_k` itself uses.
    configs.append(_cfg("float32", 16, 16384, 256, idx_dtype="int64"))
    configs.append(_cfg("float16", 64, 65536, 2048, idx_dtype="int64"))

    # 5. The trivial branch (:417-429, :467-476, :511-520): seq_len <= k skips
    #    the worker entirely and is the one path with a deterministic output.
    for mode in MODES:
        configs.append(_cfg("float32", 8, 4096, 4096, mode=mode))

    # 6. Input patterns.  Ties are the interesting axis here: the boundary bin
    #    is what drives the overflow ring and the cluster-wide tie rationing,
    #    and `all_equal` puts the entire row in it.
    for pattern in ("tie_heavy", "all_equal", "neg"):
        configs.append(_cfg("float32", 16, 16384, 256, pattern=pattern))
    configs.append(_cfg("float32", 16, 16384, 2048, pattern="all_equal"))
    #    Per-row lengths, including rows short enough to take the trivial path
    #    beside full rows in the same launch.
    for mode in ("page_table", "ragged"):
        configs.append(_cfg("float32", 16, 16384, 256, mode=mode, pattern="rowvar"))

    #    Two clusters is otherwise reached by only one shape, and the transforms
    #    otherwise run at only one cluster width; both matter because the
    #    cluster-wide tie rationing and range allocation scale with the width.
    configs.append(_cfg("float32", 200, 65536, 1024))
    configs.append(_cfg("float32", 200, 16384, 256, mode="page_table"))
    configs.append(_cfg("bfloat16", 300, 16384, 256, mode="ragged"))

    # 7. Measurement control: float16 and bfloat16 differ only in the ordered-bit
    #    map, and compile to the same instruction sequence at a given rung, so a
    #    spread between this pair is measurement rather than kernel.
    for dtype in ("float16", "bfloat16"):
        configs.append(_cfg(dtype, 64, 16384, 1024, pattern="tie_heavy"))

    seen: dict[str, dict[str, Any]] = {}
    for cfg in configs:
        seen.setdefault(cfg["label"], cfg)
    deduped = list(seen.values())

    covered = {clusters_for(c["batch"], c["seq_len"]) for c in deduped}
    assert covered == set(CLUSTER_WIDTHS), f"cluster widths not covered: {covered}"
    assert {c["mode"] for c in deduped} == set(MODES)
    assert {c["dtype"] for c in deduped} == set(DTYPES)
    return deduped


CONFIGS = _build_configs()
BENCH_CONFIGS = CONFIGS

_ = (bench, dtype_bytes, radix_rounds, shared_mem_bytes, num_cached_for, clusters_for, PATTERNS)
