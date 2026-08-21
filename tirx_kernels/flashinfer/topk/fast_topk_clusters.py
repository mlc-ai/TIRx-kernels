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

from tirx_kernels.flashinfer.utils import topk_radix as R
from tirx_kernels.flashinfer.utils.filtered_topk_ops import st_global_bits
from tirx_kernels.flashinfer.utils.topk_harness import source_module, torch_dtype
from tirx_kernels.runner import bench
from tvm.script import tirx as T

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
    out = T.alloc_local((1,), "uint32")
    T.evaluate(T.ptx.ld.global_.nc.b32(out[0], buf.ptr_to([index])))
    return out[0]


def _ld_nc_bits(buf, index, is32):
    """The scalar mirror of `_ld_vec4`: `ld.global.nc.b32` | `ld.global.nc.b16`."""
    if is32:
        return _ld_nc32(buf, index)
    out = T.alloc_local((1,), "uint16")
    T.evaluate(T.ptx.ld.global_.nc.b16(out[0], buf.ptr_to([index])))
    return out[0]


def _ld_vec4(buf, elem_index, is32):
    """One `vec_t<T,4>` load, returned as raw 32-bit words.

    The reference emits `ld.global.nc.v4.b32` at f32 and `ld.global.nc.v2.b32` at
    16 bits (:227-236). `topk_radix.ld_global_words` spells the same widths
    WITHOUT `.nc`; `logits` is `const __restrict__`, so the qualified form is the
    faithful one, and it carries almost all of this kernel's memory traffic.
    """
    if is32:
        w = T.alloc_local((4,), "uint32", align=16)
        T.evaluate(T.ptx["ld.global.nc.v4.b32"](w[0], w[1], w[2], w[3], buf.ptr_to([elem_index])))
        return w
    w2 = T.alloc_local((2,), "uint32", align=8)
    T.evaluate(T.ptx["ld.global.nc.v2.b32"](w2[0], w2[1], buf.ptr_to([elem_index])))
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
            imm = T.bitwise_not(T.uint64(-value - 1)) if value < 0 else T.uint64(value)
            T.evaluate(T.ptx.st.global_.b64(buf.ptr_to([index]), imm))
        else:
            R.st_global_u32(buf, index, T.uint32(value & 0xFFFFFFFF))
    elif i64:
        T.evaluate(
            T.ptx.st.global_.b64(buf.ptr_to([index]), T.cast(T.cast(value, "int64"), "uint64"))
        )
    else:
        R.st_global_u32(buf, index, T.reinterpret("uint32", T.cast(value, "int32")))


def _ld_i32(buf, index):
    """One `ld.global.nc.b32` of an int32, returned as int32."""
    return T.cast(_ld_nc32(buf, index), "int32")


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
    @T.macro
    def _threshold(hist, scal, tid, warp, lane, rank, bank, k_rem, sum_across):
        """`get_threshold_bin` (:116-154). Leaves the bin in `scal[THR]` and
        updates the caller's `k_rem` local; a macro cannot return."""
        T.evaluate(T.tvm_storage_sync("shared"))

        v = T.alloc_local((1,), "int32")
        v[0] = T.int32(0)
        if tid < RADIX:
            v[0] = T.cast(R.ld_shared_u32(hist, bank * RADIX + tid), "int32")
            # Five-step warp suffix scan (:21-31), unrolled. `.down` accumulates
            # toward lane 0, so the warp's total over its 32 bins lands THERE.
            s1 = R.shfl_down_u32(T.reinterpret("uint32", v[0]), 1)
            if lane < 31:
                v[0] += T.cast(s1, "int32")
            s2 = R.shfl_down_u32(T.reinterpret("uint32", v[0]), 2)
            if lane < 30:
                v[0] += T.cast(s2, "int32")
            s4 = R.shfl_down_u32(T.reinterpret("uint32", v[0]), 4)
            if lane < 28:
                v[0] += T.cast(s4, "int32")
            s8 = R.shfl_down_u32(T.reinterpret("uint32", v[0]), 8)
            if lane < 24:
                v[0] += T.cast(s8, "int32")
            s16 = R.shfl_down_u32(T.reinterpret("uint32", v[0]), 16)
            if lane < 16:
                v[0] += T.cast(s16, "int32")
            if lane == 0:
                R.st_shared_u32(scal, CUM0 + warp, T.reinterpret("uint32", v[0]))
        T.evaluate(T.tvm_storage_sync("shared"))

        if warp == 0:
            w = T.alloc_local((1,), "int32")
            w[0] = T.int32(0)
            if lane < 8:
                w[0] = T.cast(R.ld_shared_u32(scal, CUM0 + lane), "int32")
            t1 = R.shfl_down_u32(T.reinterpret("uint32", w[0]), 1)
            if lane < 7:
                w[0] += T.cast(t1, "int32")
            t2 = R.shfl_down_u32(T.reinterpret("uint32", w[0]), 2)
            if lane < 6:
                w[0] += T.cast(t2, "int32")
            t4 = R.shfl_down_u32(T.reinterpret("uint32", w[0]), 4)
            if lane < 4:
                w[0] += T.cast(t4, "int32")
            # `__syncwarp` sits BETWEEN the scan and the write-back into the same
            # buffer (:48). After the store it would protect nothing.
            T.evaluate(T.ptx.bar.warp.sync(T.uint32(0xFFFFFFFF)))
            # The guard is load-bearing: cum_reduce_buf is 8 ints with
            # k_remaining_counter 32 B later, so an unguarded 32-lane store puts
            # 128 B where 32 are allocated (:49-51).
            if lane < 8:
                R.st_shared_u32(scal, CUM0 + lane, T.reinterpret("uint32", w[0]))
        T.evaluate(T.tvm_storage_sync("shared"))
        if warp < 7:
            v[0] += T.cast(R.ld_shared_u32(scal, CUM0 + warp + 1), "int32")

        # --- cluster-wide sum of the same bin (:120-138) --------------------
        if nc > 1 and sum_across:
            if tid < RADIX:
                # This IS the DSMEM publication: peers read the PING-PONG bank.
                R.st_shared_u32(hist, bank * RADIX + tid, T.reinterpret("uint32", v[0]))
            T.evaluate(T.ptx.barrier.cluster.arrive())
            T.evaluate(T.ptx.barrier.cluster.wait())
            if tid < RADIX:
                for c in T.unroll(nc - 1):
                    peer = T.alloc_local((1,), "uint32")
                    T.evaluate(
                        T.ptx.mapa.shared__cluster.u32(
                            peer[0],
                            T.cuda.cvta_generic_to_shared(hist.ptr_to([bank * RADIX + tid])),
                            T.cast((c + rank + 1) % nc, "uint32"),
                        )
                    )
                    got = T.alloc_local((1,), "uint32")
                    T.evaluate(T.ptx.ld.shared__cluster.b32(got[0], peer[0]))
                    v[0] += T.cast(got[0], "int32")
                R.st_shared_u32(hist, 2 * RADIX + tid, T.reinterpret("uint32", v[0]))
        else:
            if tid < RADIX:
                R.st_shared_u32(hist, 2 * RADIX + tid, T.reinterpret("uint32", v[0]))
        T.evaluate(T.tvm_storage_sync("shared"))

        # --- pick the crossing bin (:142-152) --------------------------------
        nxt = T.alloc_local((1,), "int32")
        nxt[0] = T.int32(0)
        if tid < RADIX - 1:
            nxt[0] = T.cast(R.ld_shared_u32(hist, 2 * RADIX + tid + 1), "int32")
        if tid < RADIX:
            if v[0] > k_rem[0] and nxt[0] <= k_rem[0]:
                R.st_shared_u32(scal, THR, T.reinterpret("uint32", tid))
        T.evaluate(T.tvm_storage_sync("shared"))
        bin_ = T.alloc_local((1,), "int32")
        bin_[0] = T.cast(R.ld_shared_u32(scal, THR), "int32")
        if bin_[0] < RADIX - 1:
            k_rem[0] -= T.cast(R.ld_shared_u32(hist, 2 * RADIX + bin_[0] + 1), "int32")

    @T.macro
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
        tid,
    ):
        """The three-way split of :189-217, shared by pass 1 and every round."""
        d = T.alloc_local((1,), "int32")
        if is32:
            d[0] = T.cast(T.shift_right(bits, T.uint32(shift)), "int32") & 0xFF
        else:
            d[0] = T.cast(T.shift_right(bits, T.uint16(shift)), "int32") & 0xFF
        if d[0] > bin_[0]:
            # HAZARD: unguarded. The source's `if (topk_offset < TopK)` is
            # commented out at :195-197; the writeback's `offs < TopK` is what
            # bounds the global store. Reproduce, do not repair.
            slot = R.atom_shared_add_u32(scal, FINAL, T.uint32(1))
            R.st_shared_u32(topk_inds, T.cast(slot, "int32"), T.reinterpret("uint32", index))
        else:
            if d[0] == bin_[0]:
                if not last:
                    slot = T.cast(
                        R.atom_shared_add_u32(scal, NCACHED + phase, T.uint32(1)), "int32"
                    )
                    keep = T.alloc_local((1,), "int32")
                    keep[0] = T.int32(1)
                    if slot < num_cached:
                        R.st_shared_u32(
                            cidx, phase * num_cached + slot, T.reinterpret("uint32", index)
                        )
                        R.st_shared_u32(cbits, phase * num_cached + slot, T.cast(bits, "uint32"))
                    else:
                        if slot - num_cached < ovf_stride:
                            off = ring_base + phase * ovf_stride * 2 + (slot - num_cached) * 2
                            R.st_global_u32(ovf, off, T.cast(bits, "uint32"))
                            R.st_global_u32(ovf, off + 1, T.reinterpret("uint32", index))
                        else:
                            keep[0] = T.int32(0)  # ring full: dropped (:210-214)
                    if keep[0] == 1:
                        nb = T.alloc_local((1,), "int32")
                        if is32:
                            nb[0] = T.cast(T.shift_right(bits, T.uint32(shift - 8)), "int32") & 0xFF
                        else:
                            nb[0] = T.cast(T.shift_right(bits, T.uint16(shift - 8)), "int32") & 0xFF
                        R.atom_shared_add_u32(hist, (phase ^ 1) * RADIX + nb[0], T.uint32(1))
                else:
                    # Final-round tie rationing on rank 0's counter (:300-319).
                    # NOT guarded by nc > 1: at one CTA per cluster rank 0 is
                    # this CTA, and the reference's NC==1 export still carries
                    # the cluster atomic.
                    if k_rem[0] > 0 and exceeded[0] == 0:
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
                        peer = T.alloc_local((1,), "uint32")
                        T.evaluate(
                            T.ptx.mapa.shared__cluster.u32(
                                peer[0],
                                T.cuda.cvta_generic_to_shared(scal.ptr_to([KREM])),
                                T.uint32(0),
                            )
                        )
                        got = T.alloc_local((1,), "uint32")
                        T.evaluate(T.ptx.atom.shared__cluster.add.u32(got[0], peer[0], T.uint32(1)))
                        if T.cast(got[0], "int32") < k_rem[0]:
                            slot = R.atom_shared_add_u32(scal, FINAL, T.uint32(1))
                            R.st_shared_u32(
                                topk_inds, T.cast(slot, "int32"), T.reinterpret("uint32", index)
                            )
                        else:
                            exceeded[0] = T.int32(1)

    @T.macro
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
        Writing `for t in range(1, rounds)` inside a parsed body instead makes
        TVMScript emit `T.serial` -- a real runtime loop -- which collapses the
        four static `threshold()` regions into two, turns `last` into a runtime
        predicate, and leaves the per-round `digit` forms unreachable.
        """
        phase = t % 2
        exceeded[0] = T.int32(0)  # :268 -- per round, per thread
        if nc > 1:
            T.evaluate(T.ptx.barrier.cluster.wait())
        if tid < RADIX:
            R.st_shared_u32(hist, (phase ^ 1) * RADIX + tid, T.uint32(0))
        if tid == 0:
            R.st_shared_u32(scal, NCACHED + phase, T.uint32(0))
        _threshold(hist, scal, tid, warp, lane, rank, phase, k_rem, True)
        bin_t = T.alloc_local((1,), "int32")
        bin_t[0] = T.cast(R.ld_shared_u32(scal, THR), "int32")
        raw = T.alloc_local((1,), "int32")
        raw[0] = T.cast(R.ld_shared_u32(scal, NCACHED + (phase ^ 1)), "int32")
        n_sh = T.alloc_local((1,), "int32")
        n_gl = T.alloc_local((1,), "int32")
        n_sh[0] = T.min(T.int32(num_cached), raw[0])
        n_gl[0] = T.min(T.int32(ovf_stride), T.max(T.int32(0), raw[0] - num_cached))
        shift = lshift_start - t * 8
        last = t == rounds - 1

        # Stride 1024, NOT 1024*nc: each CTA owns its own candidate slice and
        # the cluster does not re-partition here (:271-273).
        for i in T.serial(tid, n_sh[0], step=BLOCK_THREADS):
            b = R.ld_shared_u32(cached_bits, (phase ^ 1) * num_cached + i)
            ix = T.cast(R.ld_shared_u32(cached_idx, (phase ^ 1) * num_cached + i), "int32")
            _classify(
                hist,
                scal,
                topk_inds,
                cached_bits,
                cached_idx,
                ovf,
                b,
                ix,
                bin_t,
                phase,
                shift,
                last,
                k_rem,
                exceeded,
                ring_base,
                tid,
            )
        for i in T.serial(tid, n_gl[0], step=BLOCK_THREADS):
            off = ring_base + (phase ^ 1) * ovf_stride * 2 + i * 2
            b = R.ld_global_u32(ovf, off)
            ix = T.cast(R.ld_global_u32(ovf, off + 1), "int32")
            _classify(
                hist,
                scal,
                topk_inds,
                cached_bits,
                cached_idx,
                ovf,
                b,
                ix,
                bin_t,
                phase,
                shift,
                last,
                k_rem,
                exceeded,
                ring_base,
                tid,
            )

        if nc > 1 and t < rounds - 1:
            # Publish only after every thread has consumed the previous
            # ping-pong bank.  Arriving immediately after threshold selection
            # lets the next round overwrite that bank while another warp is
            # still reading its cached candidates.
            T.evaluate(T.ptx.barrier.cluster.arrive())

    @T.macro
    def _emit(logits, out_idx, out_val, seq_lens_g, aux, ovf):
        T.device_entry()

        cta = T.cta_id([grid])
        rank = T.cta_id_in_cluster([nc]) if nc > 1 else T.int32(0)
        tid = T.thread_id([BLOCK_THREADS])
        row = cta // nc
        warp = tid >> 5
        lane = tid & 31

        logit_base = row * seq_len
        ind_base = row * k

        row_len = T.alloc_local((1,), "int32")
        row_len[0] = T.int32(seq_len)
        if not plain:
            row_len[0] = _ld_i32(seq_lens_g, row)
        rag_off = T.alloc_local((1,), "int32")
        rag_off[0] = T.int32(0)
        # `page_table_offset` is hoisted ONCE per CTA in the source (:465) and
        # reused by both the trivial branch (:471) and the writeback (:490).
        # Recomputing `row * pt_stride` inside those loops costs a multiply-add
        # per element on the one path that has no other per-element arithmetic,
        # which is exactly where the page_table trivial shape loses to ragged.
        pt_base = T.alloc_local((1,), "int32")
        pt_base[0] = T.int32(0)
        if pt:
            pt_base[0] = row * seq_len
        if ragged:
            # Hoisted above the trivial branch exactly as the source does (:510)
            # and reused by both it and the writeback.
            rag_off[0] = _ld_i32(aux, row)

        # ---- trivial branch (:417-429 / :467-476 / :511-520) ---------------
        if row_len[0] <= k:
            if pt and triv_trips * BLOCK_THREADS * nc == k:
                # Stage the page-table loads into DISTINCT registers before any
                # store. `T.serial(..., unroll=N)` reuses one destination local
                # across the unrolled copies, so the loads serialize: each waits
                # for the previous store to consume the register. Paired NCU on
                # this shape measured long_scoreboard 22.48 vs the reference's
                # 7.02 at IDENTICAL global load sectors (4352 both) -- exposed
                # load latency, not extra traffic. An explicitly unrolled body
                # with one register per load lets all `triv_trips` be in flight.
                stage = T.alloc_local((triv_trips,), "int32")
                tbase = tid + (cta % nc) * BLOCK_THREADS
                for u in range(triv_trips):
                    iu = tbase + u * BLOCK_THREADS * nc
                    if iu < row_len[0]:
                        stage[u] = _ld_i32(aux, pt_base[0] + iu)
                    else:
                        stage[u] = T.int32(-1)
                for u in range(triv_trips):
                    iu = tbase + u * BLOCK_THREADS * nc
                    _st_idx(out_idx, ind_base + iu, stage[u], i64)
            else:
                for i in T.serial(
                    tid + (cta % nc) * BLOCK_THREADS, k, step=BLOCK_THREADS * nc, unroll=triv_unroll
                ):
                    if i < row_len[0]:
                        if plain:
                            _st_idx(out_idx, ind_base + i, i, i64)
                            st_global_bits(
                                out_val,
                                ind_base + i,
                                _ld_nc_bits(logits, logit_base + i, is32),
                                is32,
                            )
                        elif pt:
                            _st_idx(out_idx, ind_base + i, _ld_i32(aux, pt_base[0] + i), i64)
                        else:
                            _st_idx(out_idx, ind_base + i, i + rag_off[0], i64)
                    else:
                        # output_values is left UNTOUCHED over the pad region
                        # (:425-427) -- the port must not zero it.
                        _st_idx(out_idx, ind_base + i, -1, i64)
        else:
            pool = T.SMEMPool()
            cached_bits = pool.alloc((2 * num_cached,), "uint32", align=128)
            cached_idx = pool.alloc((2 * num_cached,), "int32")
            topk_inds = pool.alloc((k,), "int32")
            hist = pool.alloc((3 * RADIX,), "int32")
            scal = pool.alloc((NSCAL,), "int32")
            pool.commit()

            ring_base = cta * ovf_stride * 4
            exceeded = T.alloc_local((1,), "int32")
            exceeded[0] = T.int32(0)

            # ---- zero banks 0/1 and three scalars (:158-170) ---------------
            if tid < RADIX:
                R.st_shared_u32(hist, tid, T.uint32(0))
                R.st_shared_u32(hist, RADIX + tid, T.uint32(0))
            if tid == 0:
                R.st_shared_u32(scal, FINAL, T.uint32(0))
                R.st_shared_u32(scal, KREM, T.uint32(0))
                R.st_shared_u32(scal, NCACHED, T.uint32(0))
                # threshold_bin is deliberately NOT initialized (:166-170): a
                # round whose crossing test selects no lane reads the previous
                # round's value. Reproduce, do not repair.
            T.evaluate(T.tvm_storage_sync("shared"))

            # ---- first histogram pass: SCALAR strided (:174-179) -----------
            for i in T.serial(
                tid + rank * BLOCK_THREADS, row_len[0], step=BLOCK_THREADS * nc, unroll=hist_unroll
            ):
                xb = _ld_nc_bits(logits, logit_base + i, is32)
                ob = R.to_ordered_u32(xb) if is32 else R.to_ordered_u16(xb)
                if is32:
                    d0 = T.cast(T.shift_right(ob, T.uint32(lshift_start)), "int32") & 0xFF
                else:
                    d0 = T.cast(T.shift_right(ob, T.uint16(lshift_start)), "int32") & 0xFF
                R.atom_shared_add_u32(hist, d0, T.uint32(1))

            k_rem = T.alloc_local((1,), "int32")
            k_rem[0] = T.int32(k)
            _threshold(hist, scal, tid, warp, lane, rank, 0, k_rem, True)
            bin0 = T.alloc_local((1,), "int32")
            bin0[0] = T.cast(R.ld_shared_u32(scal, THR), "int32")

            if nc > 1:
                # SPLIT barrier: the matching wait heads round 1 (:218-225, :248).
                T.evaluate(T.ptx.barrier.cluster.arrive())

            # Vectorized classification (:227-236): four elements per issue,
            # then a scalar tail (:237-239). Only THIS pass is vectorized -- the
            # first histogram pass above is a plain strided scalar read, which is
            # why the row is read from global twice in total.
            vbase = (rank * BLOCK_THREADS + tid) * 4
            for i in T.serial(
                vbase, (row_len[0] // 4) * 4, step=BLOCK_THREADS * nc * 4, unroll=vec_unroll
            ):
                w = _ld_vec4(logits, logit_base + i, is32)
                for j in T.unroll(4):
                    if is32:
                        eb = w[j]
                    else:
                        eb = T.cast(
                            T.bitwise_and(
                                T.shift_right(w[j // 2], T.uint32(16 * (j % 2))), T.uint32(0xFFFF)
                            ),
                            "uint16",
                        )
                    ob = R.to_ordered_u32(eb) if is32 else R.to_ordered_u16(eb)
                    _classify(
                        hist,
                        scal,
                        topk_inds,
                        cached_bits,
                        cached_idx,
                        ovf,
                        ob,
                        i + j,
                        bin0,
                        0,
                        lshift_start,
                        False,
                        k_rem,
                        exceeded,
                        ring_base,
                        tid,
                    )
            for i in T.serial(
                (row_len[0] // 4) * 4 + rank * BLOCK_THREADS + tid,
                row_len[0],
                step=BLOCK_THREADS * nc,
            ):
                xb = _ld_nc_bits(logits, logit_base + i, is32)
                ob = R.to_ordered_u32(xb) if is32 else R.to_ordered_u16(xb)
                _classify(
                    hist,
                    scal,
                    topk_inds,
                    cached_bits,
                    cached_idx,
                    ovf,
                    ob,
                    i,
                    bin0,
                    0,
                    lshift_start,
                    False,
                    k_rem,
                    exceeded,
                    ring_base,
                    tid,
                )

            # ---- refinement rounds (:241-373), emitted as straight-line code
            # so each round's phase/shift/last stay compile-time, matching the
            # source's `#pragma unroll` (:241).
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
                1,
            )
            if rounds == 4:
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
                    2,
                )
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
                    3,
                )
            # ---- epilogue: claim the output range (:374-404) ----------------
            T.evaluate(T.tvm_storage_sync("shared"))
            n_out = T.alloc_local((1,), "int32")
            out_start = T.alloc_local((1,), "int32")
            n_out[0] = T.int32(k)
            out_start[0] = T.int32(0)
            if nc > 1:
                my_num = T.alloc_local((1,), "int32")
                # Read own count BEFORE rank 0's cursor absorbs peers (:379-386).
                my_num[0] = T.cast(R.ld_shared_u32(scal, FINAL), "int32")
                T.evaluate(T.ptx.barrier.cluster.arrive())
                T.evaluate(T.ptx.barrier.cluster.wait())
                if rank > 0:
                    if tid == 0:
                        peer = T.alloc_local((1,), "uint32")
                        T.evaluate(
                            T.ptx.mapa.shared__cluster.u32(
                                peer[0],
                                T.cuda.cvta_generic_to_shared(scal.ptr_to([FINAL])),
                                T.uint32(0),
                            )
                        )
                        old = T.alloc_local((1,), "uint32")
                        T.evaluate(
                            T.ptx.atom.shared__cluster.add.u32(
                                old[0], peer[0], T.reinterpret("uint32", my_num[0])
                            )
                        )
                        R.st_shared_u32(scal, FINAL, old[0])  # broadcast (:392)
                    T.evaluate(T.tvm_storage_sync("shared"))
                    out_start[0] = T.cast(R.ld_shared_u32(scal, FINAL), "int32")
                T.evaluate(T.ptx.barrier.cluster.arrive())
                T.evaluate(T.ptx.barrier.cluster.wait())
                n_out[0] = T.min(T.int32(k), my_num[0])

            # ---- writeback (:438-447 / :486-492 / :530-536) -----------------
            for i in T.serial(tid, n_out[0], step=BLOCK_THREADS, unroll=wb_unroll):
                offs = i + out_start[0]
                if offs < k:
                    ind = T.cast(R.ld_shared_u32(topk_inds, i), "int32")
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
                        _st_idx(out_idx, ind_base + offs, _ld_i32(aux, pt_base[0] + ind), i64)
                    else:
                        _st_idx(out_idx, ind_base + offs, ind + rag_off[0], i64)

    if plain:

        @T.prim_func
        def fast_topk_clusters_kernel(
            logits_h: T.handle, indices_h: T.handle, values_h: T.handle, overflow_h: T.handle
        ):
            T.func_attr(
                {"global_symbol": "fast_topk_clusters_kernel", "tirx.launch_tags": launch_tags(nc)}
            )
            _emit(
                T.match_buffer(logits_h, (batch * seq_len,), val_t, scope="global"),
                T.match_buffer(indices_h, (batch * k,), idx_t, scope="global"),
                T.match_buffer(values_h, (batch * k,), val_t, scope="global"),
                None,
                None,
                T.match_buffer(overflow_h, (batch * 4 * ovf_stride * nc,), "int32", scope="global"),
            )

    else:

        @T.prim_func
        def fast_topk_clusters_kernel(
            logits_h: T.handle,
            indices_h: T.handle,
            seq_lens_h: T.handle,
            aux_h: T.handle,
            overflow_h: T.handle,
        ):
            T.func_attr(
                {"global_symbol": "fast_topk_clusters_kernel", "tirx.launch_tags": launch_tags(nc)}
            )
            _emit(
                T.match_buffer(logits_h, (batch * seq_len,), val_t, scope="global"),
                T.match_buffer(indices_h, (batch * k,), idx_t, scope="global"),
                None,
                T.match_buffer(seq_lens_h, (batch,), "int32", scope="global"),
                T.match_buffer(aux_h, (batch * seq_len if pt else batch,), "int32", scope="global"),
                T.match_buffer(overflow_h, (batch * 4 * ovf_stride * nc,), "int32", scope="global"),
            )

    return fast_topk_clusters_kernel


@T.prim_func
def fast_topk_clusters_kernel() -> None:
    """Scaffold entry for the kernel-sketch stage."""
    T.func_attr({"global_symbol": "fast_topk_clusters_kernel"})
    # TIRX_TRANSCRIBE_START fast_topk_clusters
    T.evaluate(0)


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


def assert_selection_is_top_k(data: dict[str, Any], indices, row: int, row_len: int) -> None:
    """The oracle: the selected *value multiset* must equal `torch.topk`'s.

    Positions carry no meaning on this path -- output slots are handed out by
    atomic compaction, the per-CTA base comes from a racing atomic, and the
    boundary bin's survivors are whichever lanes arrive first (`:306-308`).  The
    multiset is still unique, because an exact radix select over the ordered bits
    admits exactly one answer, so this is a bit-exact check and not a tolerance.
    """
    import torch

    k = data["k"]
    row_logits = data["logits"][row, :row_len].float()
    local = local_indices(data, indices, row).long()

    assert int(local.min()) >= 0 and int(local.max()) < row_len, (
        f"row {row}: index out of range [0, {row_len})"
    )
    assert len(set(local.tolist())) == local.numel(), f"row {row}: duplicate indices"

    got = torch.sort(row_logits[local]).values
    want = torch.sort(torch.topk(row_logits, min(k, row_len)).values).values
    assert torch.equal(got, want), (
        f"row {row}: selected value multiset differs from torch.topk; "
        f"max|delta| = {(got - want).abs().max().item()}"
    )


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
        assert_selection_is_top_k(data, mine["indices"][row], row, row_len)
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
