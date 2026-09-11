# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Agent-evolved SM100a Video Sparse Attention (VSA) forward, blk128 bf16.

The supported contract is the pinned large VSA row
``vsa-pooled-blk128-s80000-h8-topk156``: S = 80000, H = 8, D = 128,
``block_size`` = 128, 625 KV blocks and top-k = 156, with contiguous bf16
q/k/v/output of shape [S, H, 128] and ``q2k_indices`` int32[H, S/128, topk]
holding a sorted list of distinct KV block ids per (head, query block).
``kv_block_lens`` is ``None``: every token of a selected block is valid. There
is no causal mask and no GQA.

The selected kernel is the ``group16-cluster-multicast`` frontier member of the
2026-09-10 VSA evolution run. Everything from the module docstring's mechanism
notes down to ``_encode_map`` is that candidate's source; this module adds the
registry interface, input generation, the independent oracle, and the
FlashInfer reference arm.

It is pure bf16: q/k/v and P are bf16 and both tcgen05 MMAs are ``kind::f16``
with FP32 TMEM accumulation. No operand is quantized.

Two kernels run per invocation: the fused scheduler that builds the pair map
and reordered block lists, then the 2-CTA cluster attention kernel.

Candidate mechanism notes, carried over from the evolution run:


Family: "group-scheduled cluster intersection multicast".  A 16-warp Kern
preprocessing kernel builds exact per-row block-id bitsets, greedily pairs query
blocks within fixed groups of sixteen by intersection popcount, and reorders
each chosen pair in the same launch.  The attention kernel assigns each pair
to a 2-CTA cluster.  At matching positions, one TMA request multicasts K or V
into both CTAs; at unique positions each CTA loads its own tile.  Computation
inside each CTA retains two independent online-softmax streams and uses a
5-stage KV ring for extra delivery slack.  An odd final query block is emitted
as a single-rank task without changing its mask.

Layout contract: q/k/v/out are contiguous bf16[S, H, 128]; q2k_indices is
int32[H, S/128, topk].  One rank-3 TMA box (64 cols, 128 rows, 2 bands) moves a
whole 128x128 tile into the SW128B band-major smem layout the MMA descriptors
expect.
"""

import ctypes
import math
from typing import Any
from unittest import SkipTest

import torch
import tvm

import tirx_kernels.kern as K

D = 128
BLK = 128
KV_STAGES = 5
TMEM_COLS = 512
WARPS = 16
NEG_INF = -float("inf")
LOG2_E = 1.4426950408889634
REGS_SOFTMAX = 184
REGS_CORRECTION = 88
REGS_OTHER = 56
POLY_EX2_3 = (1.0, 0.695146143436431884765625, 0.227564394474029541015625, 0.077119089663028717041015625)
FP32_ROUND_INT = float(2**23 + 2**22)
QK_IDESC = 0x08200490                                                     
PV_IDESC = 0x08210490                        
MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
TCGEN_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
TMEM_LD32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TMEM_LD16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
TMEM_ST16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
TMA_G2S = "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
TMA_G2S_MCAST = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
    "mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
)
TMA_S2G = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                                               
POLICY_EVICT_NORMAL = 0x1000000000000000
POLICY_EVICT_FIRST = 0x12F0000000000000
POLICY_EVICT_LAST = 0x14F0000000000000
TILE_BYTES = BLK * D * 2
STAGE16 = TILE_BYTES // 16                                         


DEFAULT_CFG = dict(
    emu="quarter",                                                                                       
    kv_stages=5,                                                                
    regs=(184, 88, 56),                              
    kv_policy="evict_last",
    q_policy="evict_first",
    o_policy="evict_first",
    chain=True,
    iket=False,
)
_POLICIES = {"evict_last": POLICY_EVICT_LAST, "evict_first": POLICY_EVICT_FIRST,
             "evict_normal": POLICY_EVICT_NORMAL, "none": 0}


def make_pair_kernel(S, H, topk):
    """Greedily pair query blocks within fixed groups of eight.

    Eight warps evaluate every remaining candidate for the highest-numbered
    unmatched row in parallel.  Each intersection is an exact shared-memory
    binary-search count, so this changes scheduling only, never the attention
    mask.
    """
    num_q_blocks = S // BLK
    pairs_per_head = (num_q_blocks + 1) // 2
    group_size = 8
    groups_per_head = (num_q_blocks + group_size - 1) // group_size
    num_groups = H * groups_per_head
    binary_steps = max(1, (topk - 1).bit_length())

    def pair_groups(
        q2k: K.gptr[K.i32, (H * num_q_blocks * topk,)],
        pair_qblocks: K.gptr[K.i32, (H * pairs_per_head * 2,)],
        reordered: K.gptr[K.i32, (H * pairs_per_head * 2 * topk,)],
        common_counts: K.gptr[K.i32, (H * pairs_per_head,)],
    ):
        group_task = K.cta_id()
        warp = K.warp_id()
        lane = K.lane_id()
        head = K.local_scalar("int32", init=group_task // groups_per_head)
        group = K.local_scalar(
            "int32", init=group_task - head * groups_per_head
        )
        group_start = K.local_scalar("int32", init=group * group_size)
        group_n = K.local_scalar(
            "int32", init=K.min(group_size, num_q_blocks - group_start)
        )

        smem = K.smem_pool()
        lists = smem.alloc((group_size, topk), K.i32, align=16)
        counts = smem.alloc((group_size,), K.i32, align=16)
        used_s = smem.alloc((1,), K.u32, align=4)
        first_s = smem.alloc((1,), K.i32, align=4)
        partners = smem.alloc((group_size,), K.i32, align=16)
        pair_tasks = smem.alloc((group_size,), K.i32, align=16)
        pair_ranks = smem.alloc((group_size,), K.i32, align=16)

        pos = K.local_scalar("int32", init=lane)
        with K.While(pos < topk):
            value = K.local_scalar("int32", init=0)
            with K.If(warp < group_n), K.Then():
                K.ptx.ld.global_.b32(
                    value,
                    q2k.ptr_to(
                        [
                            (
                                head * num_q_blocks
                                + group_start
                                + warp
                            )
                            * topk
                            + pos
                        ]
                    ),
                )
            K.ptx.st.shared.b32(lists.ptr_to([warp, pos]), value)
            K.assign(pos, pos + 32)
        with K.If((warp == 0) & (lane == 0)), K.Then():
            K.ptx.st.shared.u32(used_s.ptr_to([0]), K.uint32(0))
        K.cuda.cta_sync()

        group_pairs = K.local_scalar("int32", init=(group_n + 1) // 2)
        for pair_slot in range(group_size // 2):
            with K.If(pair_slot < group_pairs), K.Then():
                with K.If((warp == 0) & (lane == 0)), K.Then():
                    used = K.local_scalar("uint32")
                    K.ptx.ld.shared.u32(used, used_s.ptr_to([0]))
                    valid_mask = K.local_scalar(
                        "uint32",
                        init=K.shift_left(
                            K.uint32(1), K.cast(group_n, "uint32")
                        )
                        - K.uint32(1),
                    )
                    available = K.local_scalar(
                        "uint32", init=(~used) & valid_mask
                    )
                    first_u = K.local_scalar("uint32")
                    K.ptx.bfind.u32(first_u, available)
                    K.ptx.st.shared.s32(
                        first_s.ptr_to([0]), K.cast(first_u, "int32")
                    )
                K.cuda.cta_sync()

                first = K.local_scalar("int32")
                used = K.local_scalar("uint32")
                K.ptx.ld.shared.s32(first, first_s.ptr_to([0]))
                K.ptx.ld.shared.u32(used, used_s.ptr_to([0]))
                candidate = (
                    (warp < group_n)
                    & (warp != first)
                    & (((used >> K.cast(warp, "uint32")) & K.uint32(1)) == 0)
                )
                common = K.local_scalar("int32", init=0)
                for chunk_base in range(0, topk, 32):
                    index = K.local_scalar("int32", init=chunk_base + lane)
                    is_common = K.local_scalar("int32", init=0)
                    with K.If(candidate & (index < topk)), K.Then():
                        value = K.local_scalar("int32")
                        K.ptx.ld.shared.s32(
                            value, lists.ptr_to([first, index])
                        )
                        lo = K.local_scalar("int32", init=0)
                        hi = K.local_scalar("int32", init=topk)
                        for _ in range(binary_steps):
                            with K.If(lo < hi), K.Then():
                                mid = K.local_scalar(
                                    "int32", init=(lo + hi) // 2
                                )
                                probe = K.local_scalar("int32")
                                K.ptx.ld.shared.s32(
                                    probe, lists.ptr_to([warp, mid])
                                )
                                with K.If(probe < value):
                                    with K.Then():
                                        K.assign(lo, mid + 1)
                                    with K.Else():
                                        K.assign(hi, mid)
                        with K.If(lo < topk), K.Then():
                            probe = K.local_scalar("int32")
                            K.ptx.ld.shared.s32(
                                probe, lists.ptr_to([warp, lo])
                            )
                            K.assign(
                                is_common, K.cast(probe == value, "int32")
                            )
                    ballot = K.local_scalar("uint32")
                    K.ptx.vote_sync.ballot.b32(
                        ballot,
                        K.ptx.pred(is_common),
                        K.uint32(0xFFFFFFFF),
                    )
                    chunk_common = K.local_scalar("uint32")
                    K.ptx.popc.b32(chunk_common, ballot)
                    K.assign(common, common + K.cast(chunk_common, "int32"))
                with K.If(lane == 0), K.Then():
                    K.ptx.st.shared.s32(
                        counts.ptr_to([warp]),
                        K.if_then_else(candidate, common, -1),
                    )
                K.cuda.cta_sync()

                with K.If((warp == 0) & (lane == 0)), K.Then():
                    best = K.local_scalar("int32", init=first)
                    best_count = K.local_scalar("int32", init=-1)
                    for candidate_row in range(group_size):
                        value = K.local_scalar("int32")
                        K.ptx.ld.shared.s32(
                            value, counts.ptr_to([candidate_row])
                        )
                        with K.If(value > best_count), K.Then():
                            K.assign(best_count, value)
                            K.assign(best, candidate_row)
                    has_partner = best_count >= 0
                    next_used = K.local_scalar(
                        "uint32",
                        init=used
                        | (K.uint32(1) << K.cast(first, "uint32"))
                        | (K.uint32(1) << K.cast(best, "uint32")),
                    )
                    K.ptx.st.shared.u32(used_s.ptr_to([0]), next_used)
                    pair_task = K.local_scalar(
                        "int32",
                        init=head * pairs_per_head
                        + (group * group_size) // 2
                        + pair_slot,
                    )
                    K.ptx.st.shared.s32(
                        partners.ptr_to([first]),
                        K.if_then_else(has_partner, best, first),
                    )
                    K.ptx.st.shared.s32(
                        pair_tasks.ptr_to([first]), pair_task
                    )
                    K.ptx.st.shared.s32(pair_ranks.ptr_to([first]), 0)
                    with K.If(has_partner), K.Then():
                        K.ptx.st.shared.s32(partners.ptr_to([best]), first)
                        K.ptx.st.shared.s32(
                            pair_tasks.ptr_to([best]), pair_task
                        )
                        K.ptx.st.shared.s32(pair_ranks.ptr_to([best]), 1)
                    K.ptx.st.global_.s32(
                        pair_qblocks.ptr_to([pair_task * 2]),
                        group_start + first,
                    )
                    K.ptx.st.global_.s32(
                        pair_qblocks.ptr_to([pair_task * 2 + 1]),
                        K.if_then_else(
                            has_partner, group_start + best, -1
                        ),
                    )
                K.cuda.cta_sync()

    return K.kernel(warps=8, arch="sm_100a", grid=num_groups)(pair_groups)


def make_pair_kernel_bitset(S, H, topk):
    """Bitset/popcount form of the group-of-sixteen greedy matcher."""
    num_q_blocks = S // BLK
    pairs_per_head = (num_q_blocks + 1) // 2
    group_size = 16
    bit_words = (num_q_blocks + 31) // 32
    groups_per_head = (num_q_blocks + group_size - 1) // group_size
    num_groups = H * groups_per_head

    def pair_groups(
        q2k: K.gptr[K.i32, (H * num_q_blocks * topk,)],
        pair_qblocks: K.gptr[K.i32, (H * pairs_per_head * 2,)],
        reordered: K.gptr[K.i32, (H * pairs_per_head * 2 * topk,)],
        common_counts: K.gptr[K.i32, (H * pairs_per_head,)],
    ):
        group_task = K.cta_id()
        warp = K.warp_id()
        lane = K.lane_id()
        head = K.local_scalar("int32", init=group_task // groups_per_head)
        group = K.local_scalar(
            "int32", init=group_task - head * groups_per_head
        )
        group_start = K.local_scalar("int32", init=group * group_size)
        group_n = K.local_scalar(
            "int32", init=K.min(group_size, num_q_blocks - group_start)
        )

        smem = K.smem_pool()
        bits = smem.alloc((group_size, bit_words), K.u32, align=16)
        counts = smem.alloc((group_size,), K.i32, align=16)
        used_s = smem.alloc((1,), K.u32, align=4)
        first_s = smem.alloc((1,), K.i32, align=4)
        partners = smem.alloc((group_size,), K.i32, align=16)
        pair_tasks = smem.alloc((group_size,), K.i32, align=16)
        pair_ranks = smem.alloc((group_size,), K.i32, align=16)

        word = K.local_scalar("int32", init=lane)
        with K.While(word < bit_words):
            K.ptx.st.shared.u32(bits.ptr_to([warp, word]), K.uint32(0))
            K.assign(word, word + 32)
        K.cuda.cta_sync()
        pos = K.local_scalar("int32", init=lane)
        with K.While(pos < topk):
            with K.If(warp < group_n), K.Then():
                sid = K.local_scalar("int32")
                K.ptx.ld.global_.s32(
                    sid,
                    q2k.ptr_to(
                        [
                            (
                                head * num_q_blocks
                                + group_start
                                + warp
                            )
                            * topk
                            + pos
                        ]
                    ),
                )
                prior = K.local_scalar("uint32")
                K.ptx.atom.shared.or_.b32(
                    prior,
                    bits.ptr_to([warp, sid >> 5]),
                    K.uint32(1) << K.cast(sid & 31, "uint32"),
                )
            K.assign(pos, pos + 32)
        with K.If((warp == 0) & (lane == 0)), K.Then():
            K.ptx.st.shared.u32(used_s.ptr_to([0]), K.uint32(0))
        K.cuda.cta_sync()

        group_pairs = K.local_scalar("int32", init=(group_n + 1) // 2)
        for pair_slot in range(group_size // 2):
            with K.If(pair_slot < group_pairs), K.Then():
                with K.If((warp == 0) & (lane == 0)), K.Then():
                    used = K.local_scalar("uint32")
                    K.ptx.ld.shared.u32(used, used_s.ptr_to([0]))
                    valid_mask = K.local_scalar(
                        "uint32",
                        init=(K.uint32(1) << K.cast(group_n, "uint32"))
                        - K.uint32(1),
                    )
                    available = K.local_scalar(
                        "uint32", init=(~used) & valid_mask
                    )
                    first_u = K.local_scalar("uint32")
                    K.ptx.bfind.u32(first_u, available)
                    K.ptx.st.shared.s32(
                        first_s.ptr_to([0]), K.cast(first_u, "int32")
                    )
                K.cuda.cta_sync()

                first = K.local_scalar("int32")
                used = K.local_scalar("uint32")
                K.ptx.ld.shared.s32(first, first_s.ptr_to([0]))
                K.ptx.ld.shared.u32(used, used_s.ptr_to([0]))
                candidate = (
                    (warp < group_n)
                    & (warp != first)
                    & (((used >> K.cast(warp, "uint32")) & K.uint32(1)) == 0)
                )
                lane_count = K.local_scalar("uint32", init=K.uint32(0))
                with K.If(candidate & (lane < bit_words)), K.Then():
                    lhs = K.local_scalar("uint32")
                    rhs = K.local_scalar("uint32")
                    K.ptx.ld.shared.u32(lhs, bits.ptr_to([first, lane]))
                    K.ptx.ld.shared.u32(rhs, bits.ptr_to([warp, lane]))
                    K.ptx.popc.b32(lane_count, lhs & rhs)
                common_u = K.local_scalar("uint32")
                K.ptx.redux_sync.add.u32(
                    common_u, lane_count, K.uint32(0xFFFFFFFF)
                )
                with K.If(lane == 0), K.Then():
                    K.ptx.st.shared.s32(
                        counts.ptr_to([warp]),
                        K.if_then_else(
                            candidate, K.cast(common_u, "int32"), -1
                        ),
                    )
                K.cuda.cta_sync()

                with K.If((warp == 0) & (lane == 0)), K.Then():
                    best = K.local_scalar("int32", init=first)
                    best_count = K.local_scalar("int32", init=-1)
                    for candidate_row in range(group_size):
                        value = K.local_scalar("int32")
                        K.ptx.ld.shared.s32(
                            value, counts.ptr_to([candidate_row])
                        )
                        with K.If(value > best_count), K.Then():
                            K.assign(best_count, value)
                            K.assign(best, candidate_row)
                    has_partner = best_count >= 0
                    next_used = K.local_scalar(
                        "uint32",
                        init=used
                        | (K.uint32(1) << K.cast(first, "uint32"))
                        | (K.uint32(1) << K.cast(best, "uint32")),
                    )
                    K.ptx.st.shared.u32(used_s.ptr_to([0]), next_used)
                    pair_task = K.local_scalar(
                        "int32",
                        init=head * pairs_per_head
                        + (group * group_size) // 2
                        + pair_slot,
                    )
                    K.ptx.st.shared.s32(
                        partners.ptr_to([first]),
                        K.if_then_else(has_partner, best, first),
                    )
                    K.ptx.st.shared.s32(
                        pair_tasks.ptr_to([first]), pair_task
                    )
                    K.ptx.st.shared.s32(pair_ranks.ptr_to([first]), 0)
                    with K.If(has_partner), K.Then():
                        K.ptx.st.shared.s32(partners.ptr_to([best]), first)
                        K.ptx.st.shared.s32(
                            pair_tasks.ptr_to([best]), pair_task
                        )
                        K.ptx.st.shared.s32(pair_ranks.ptr_to([best]), 1)
                    K.ptx.st.global_.s32(
                        pair_qblocks.ptr_to([pair_task * 2]),
                        group_start + first,
                    )
                    K.ptx.st.global_.s32(
                        pair_qblocks.ptr_to([pair_task * 2 + 1]),
                        K.if_then_else(
                            has_partner, group_start + best, -1
                        ),
                    )
                K.cuda.cta_sync()

                                                                              
                                                                           
                                                                  
        with K.If(warp < group_n), K.Then():
            partner = K.local_scalar("int32")
            pair_task = K.local_scalar("int32")
            pair_rank = K.local_scalar("int32")
            K.ptx.ld.shared.s32(partner, partners.ptr_to([warp]))
            K.ptx.ld.shared.s32(pair_task, pair_tasks.ptr_to([warp]))
            K.ptx.ld.shared.s32(pair_rank, pair_ranks.ptr_to([warp]))
            in_base = K.local_scalar(
                "int32",
                init=(head * num_q_blocks + group_start + warp) * topk,
            )
            out_base = K.local_scalar(
                "int32", init=(pair_task * 2 + pair_rank) * topk
            )
            singleton = partner == warp
            common_base = K.local_scalar("int32", init=0)
            lower_lane_mask = K.local_scalar(
                "uint32",
                init=(K.uint32(1) << K.cast(lane, "uint32"))
                - K.uint32(1),
            )
            for chunk_base in range(0, topk, 32):
                index = K.local_scalar("int32", init=chunk_base + lane)
                value = K.local_scalar("int32", init=0)
                is_common = K.local_scalar("int32", init=0)
                with K.If(index < topk), K.Then():
                    K.ptx.ld.global_.s32(
                        value, q2k.ptr_to([in_base + index])
                    )
                    word_bits = K.local_scalar("uint32")
                    K.ptx.ld.shared.u32(
                        word_bits, bits.ptr_to([partner, value >> 5])
                    )
                    K.assign(
                        is_common,
                        K.cast(
                            ((word_bits >> K.cast(value & 31, "uint32")) & 1)
                            != 0,
                            "int32",
                        ),
                    )
                ballot = K.local_scalar("uint32")
                K.ptx.vote_sync.ballot.b32(
                    ballot,
                    K.ptx.pred(is_common),
                    K.uint32(0xFFFFFFFF),
                )
                common_before = K.local_scalar("uint32")
                common_chunk = K.local_scalar("uint32")
                K.ptx.popc.b32(common_before, ballot & lower_lane_mask)
                K.ptx.popc.b32(common_chunk, ballot)
                with K.If(index < topk), K.Then():
                    common_rank = K.local_scalar(
                        "int32",
                        init=common_base + K.cast(common_before, "int32"),
                    )
                    target = K.local_scalar(
                        "int32",
                        init=K.if_then_else(
                            is_common != 0,
                            topk - 1 - common_rank,
                            index - common_rank,
                        ),
                    )
                    K.ptx.st.global_.s32(
                        reordered.ptr_to([out_base + target]), value
                    )
                    with K.If(singleton), K.Then():
                        K.ptx.st.global_.s32(
                            reordered.ptr_to(
                                [(pair_task * 2 + 1) * topk + target]
                            ),
                            value,
                        )
                K.assign(
                    common_base,
                    common_base + K.cast(common_chunk, "int32"),
                )
            with K.If((lane == 0) & (pair_rank == 0)), K.Then():
                K.ptx.st.global_.s32(
                    common_counts.ptr_to([pair_task]), common_base
                )

    return K.kernel(warps=16, arch="sm_100a", grid=num_groups)(pair_groups)


def make_preprocess_kernel(S, H, topk):
    """Place uniques first and reversed intersection last with one warp.

    The attention consumer walks the result backwards, so the majority unique
    suffix of each original list retains its descending traversal order.  Each
    lane binary-searches one key from a 32-element chunk in the opposite list;
    ballots provide its common-key rank without a serial merge dependency.
    """
    num_q_blocks = S // BLK
    pairs_per_head = (num_q_blocks + 1) // 2
    num_pairs = H * pairs_per_head

    def preprocess(
        q2k: K.gptr[K.i32, (H * num_q_blocks * topk,)],
        pair_qblocks: K.gptr[K.i32, (num_pairs * 2,)],
        reordered: K.gptr[K.i32, (num_pairs * 2 * topk,)],
        common_counts: K.gptr[K.i32, (num_pairs,)],
    ):
        pair_task = K.cta_id()
        lane = K.lane_id()
        head = K.local_scalar("int32", init=pair_task // pairs_per_head)
        q0 = K.local_scalar("int32")
        q1_raw = K.local_scalar("int32")
        K.ptx.ld.global_.s32(q0, pair_qblocks.ptr_to([pair_task * 2]))
        K.ptx.ld.global_.s32(
            q1_raw, pair_qblocks.ptr_to([pair_task * 2 + 1])
        )
        q1 = K.local_scalar(
            "int32", init=K.if_then_else(q1_raw >= 0, q1_raw, q0)
        )
        in0 = K.local_scalar("int32", init=(head * num_q_blocks + q0) * topk)
        in1 = K.local_scalar("int32", init=(head * num_q_blocks + q1) * topk)
        out0 = K.local_scalar("int32", init=(pair_task * 2) * topk)
        out1 = K.local_scalar("int32", init=(pair_task * 2 + 1) * topk)

        smem = K.smem_pool()
        lists = smem.alloc((2, topk), K.i32, align=16)
        pos = K.local_scalar("int32", init=lane)
        with K.While(pos < topk):
            value0 = K.local_scalar("int32")
            value1 = K.local_scalar("int32")
            K.ptx.ld.global_.b32(value0, q2k.ptr_to([in0 + pos]))
            K.ptx.ld.global_.b32(value1, q2k.ptr_to([in1 + pos]))
            K.ptx.st.shared.b32(lists.ptr_to([0, pos]), value0)
            K.ptx.st.shared.b32(lists.ptr_to([1, pos]), value1)
            K.assign(pos, pos + 32)
        K.cuda.warp_sync()

        common0 = K.local_scalar("int32", init=0)
        lower_lane_mask = K.local_scalar(
            "uint32",
            init=K.shift_left(K.uint32(1), K.cast(lane, "uint32")) - K.uint32(1),
        )
        binary_steps = max(1, (topk - 1).bit_length())

                                                                            
        for side in range(2):
            out_base = out0 if side == 0 else out1
            common_base = K.local_scalar("int32", init=0)
            for chunk_base in range(0, topk, 32):
                index = K.local_scalar("int32", init=chunk_base + lane)
                value = K.local_scalar("int32", init=0)
                is_common = K.local_scalar("int32", init=0)
                with K.If(index < topk), K.Then():
                    K.ptx.ld.shared.b32(value, lists.ptr_to([side, index]))
                    lo = K.local_scalar("int32", init=0)
                    hi = K.local_scalar("int32", init=topk)
                    for _ in range(binary_steps):
                        with K.If(lo < hi), K.Then():
                            mid = K.local_scalar("int32", init=(lo + hi) // 2)
                            probe = K.local_scalar("int32")
                            K.ptx.ld.shared.b32(
                                probe, lists.ptr_to([1 - side, mid])
                            )
                            with K.If(probe < value):
                                with K.Then():
                                    K.assign(lo, mid + 1)
                                with K.Else():
                                    K.assign(hi, mid)
                    with K.If(lo < topk), K.Then():
                        probe = K.local_scalar("int32")
                        K.ptx.ld.shared.b32(
                            probe, lists.ptr_to([1 - side, lo])
                        )
                        K.assign(is_common, K.cast(probe == value, "int32"))

                ballot = K.local_scalar("uint32")
                K.ptx.vote_sync.ballot.b32(
                    ballot,
                    K.ptx.pred(is_common),
                    K.uint32(0xFFFFFFFF),
                )
                common_before = K.local_scalar("uint32")
                common_in_chunk = K.local_scalar("uint32")
                K.ptx.popc.b32(common_before, ballot & lower_lane_mask)
                K.ptx.popc.b32(common_in_chunk, ballot)

                with K.If(index < topk), K.Then():
                    common_rank = K.local_scalar(
                        "int32",
                        init=common_base + K.cast(common_before, "int32"),
                    )
                    with K.If(is_common != 0):
                        with K.Then():
                            K.ptx.st.global_.b32(
                                reordered.ptr_to(
                                    [
                                        out_base + topk - 1 - common_rank
                                    ]
                                ),
                                value,
                            )
                        with K.Else():
                            K.ptx.st.global_.b32(
                                reordered.ptr_to(
                                    [out_base + index - common_rank]
                                ),
                                value,
                            )
                K.assign(
                    common_base,
                    common_base + K.cast(common_in_chunk, "int32"),
                )
            if side == 0:
                K.assign(common0, common_base)

        with K.If(lane == 0), K.Then():
            K.ptx.st.global_.b32(common_counts.ptr_to([pair_task]), common0)

    return K.kernel(warps=1, arch="sm_100a", grid=num_pairs)(preprocess)


def _emu_pair(mode, fragment, pair):
    """Which (fragment, pair) exp2 lanes go to the FMA polynomial instead of MUFU."""
    if mode == "cudnn":
        return (pair * 2) % 10 >= 6 and fragment < 3
    if mode == "fa4":
        return (pair * 2) % 16 >= 12 and 1 <= fragment < 3
    if mode == "quarter":
        return pair % 4 == 3
    if mode == "half":
        return pair % 2 == 1
    if mode == "none":
        return False
    raise ValueError(mode)


def make_kernel(S, H, topk, num_ctas, cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    KV_STAGES = int(cfg["kv_stages"])
    REGS_SOFTMAX, REGS_CORRECTION, REGS_OTHER = cfg["regs"]
    kv_policy = _POLICIES[cfg["kv_policy"]]
    q_policy = _POLICIES[cfg["q_policy"]]
    o_policy = _POLICIES[cfg["o_policy"]]
    emu_mode = cfg["emu"]
    use_iket = bool(cfg["iket"])
    if S % BLK:
        raise ValueError("S must be a multiple of 128")
    num_q_blocks = S // BLK
    pairs_per_head = (num_q_blocks + 1) // 2
    num_pair_tasks = H * pairs_per_head
    if num_ctas % 2:
        raise ValueError("the 2-CTA cluster grid must be even")
    num_clusters = num_ctas // 2
    count = topk if topk % 2 == 0 else topk + 1                                   
    phantom = count != topk
    pair_count = (count - 2) // 2
    work_groups = count // 2
    if count < 2:
        raise ValueError("need at least one selected block")

    def kernel_body(
        q_map, k_map, v_map, o_map, q2k, common_counts, pair_qblocks, scale_log2
    ):
        cta = K.cta_id()
        cluster_rank = K.cta_id_in_cluster([2], preferred=[2])
        cluster = cta // 2
        warp = K.warp_id()
        tid = K.thread_id()
        lane = tid & 31

        with K.If(warp == 0), K.Then():
            K.ptx.prefetch.tensormap(K.address_of(q_map))
            K.ptx.prefetch.tensormap(K.address_of(k_map))
            K.ptx.prefetch.tensormap(K.address_of(v_map))
            K.ptx.prefetch.tensormap(K.address_of(o_map))

        smem = K.smem_pool()
        q_smem = smem.alloc((BLK, D), K.bf16, swizzle=K.SW128B)
        kv_smem = smem.alloc((KV_STAGES, BLK, D), K.bf16, swizzle=K.SW128B)
        o_smem = smem.alloc((BLK, D), K.bf16, swizzle=K.SW128B)
        stats_smem = smem.alloc((512,), K.f32, align=16)
        tmem_mailbox = smem.alloc((1,), K.u32, align=8)
        pool = smem.pool
        q_full = K.TMABar(pool, 1)
        q_empty = K.TCGen05Bar(pool, 1)
        kv_full = K.TMABar(pool, KV_STAGES)
        kv_empty = K.TCGen05Bar(pool, KV_STAGES)
        spo_full = K.TCGen05Bar(pool, 2)
        spo_empty = K.MBarrier(pool, 2)
        plast_full = K.MBarrier(pool, 2)
        oacc_full = K.TCGen05Bar(pool, 2)
        stats_empty = K.MBarrier(pool, 2)
        oepi_full = K.MBarrier(pool, 1)
        oepi_empty = K.MBarrier(pool, 1)
        peer_ready = K.MBarrier(pool, 1)
        peer_ack = K.MBarrier(pool, 1)

        q_full.init(1)
        q_empty.init(1)
        kv_full.init(1)
        kv_empty.init(1)
        spo_full.init(1)
        spo_empty.init(256)
        plast_full.init(4)
        oacc_full.init(1)
        stats_empty.init(128)
        oepi_full.init(128)
        oepi_empty.init(1)
        peer_ready.init(1)
        peer_ack.init(1)
        K.ptx.fence.mbarrier_init.release.cluster()
        K.cuda.cluster_sync()

        roles = K.specialize(chain_dispatch=bool(cfg["chain"]))
        r_softmax = roles.role("softmax", warps=range(0, 8), regs=REGS_SOFTMAX)
        r_correction = roles.role("correction", warps=range(8, 12), regs=REGS_CORRECTION)
        r_mma = roles.role("mma", warps=[12], regs=REGS_OTHER)
        r_epilogue = roles.role("epilogue", warps=[13], regs=REGS_OTHER)
        r_load = roles.role("load", warps=[14], regs=REGS_OTHER)
        r_idle = roles.role("idle", warps=[15], regs=REGS_OTHER)

                                                                                    
        def exp2(value):
            out = K.local_scalar("float32")
            K.ptx.ex2.approx.ftz.f32(out, value)
            return out

        def rcp(value):
            out = K.local_scalar("float32")
            K.ptx.rcp.approx.ftz.f32(out, value)
            return out

        def packed(op, dst, base, a0, a1, b0, b1, c0=None, c1=None):
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

        def reduce_max_128(values, initial=None):
            acc = K.alloc_local((4,), "float32")
            if initial is None:
                K.ptx.max.f32(acc[0], values[0], values[1])
            else:
                K.ptx.max.f32(acc[0], initial, values[0], values[1])
            K.ptx.max.f32(acc[1], values[2], values[3])
            K.ptx.max.f32(acc[2], values[4], values[5])
            K.ptx.max.f32(acc[3], values[6], values[7])
            for group in range(1, 16):
                base = group * 8
                K.ptx.max.f32(acc[0], acc[0], values[base], values[base + 1])
                K.ptx.max.f32(acc[1], acc[1], values[base + 2], values[base + 3])
                K.ptx.max.f32(acc[2], acc[2], values[base + 4], values[base + 5])
                K.ptx.max.f32(acc[3], acc[3], values[base + 6], values[base + 7])
            K.ptx.max.f32(acc[0], acc[0], acc[1])
            K.ptx.max.f32(acc[0], acc[0], acc[2], acc[3])
            return acc[0]

        def packed_sum_128(values, old_sum, old_scale, first):
            acc = K.alloc_local((8,), "float32")
            for j in range(8):
                K.assign(acc[j], values[j])
            if not first:
                scaled_old = K.local_scalar("float32", init=old_sum * old_scale)
                packed("add.rn.f32x2", acc, 0, values[0], values[1], scaled_old, K.float32(0.0))
            for group in range(1, 16):
                base = group * 8
                for pair in range(4):
                    packed(
                        "add.rn.f32x2",
                        acc,
                        pair * 2,
                        acc[pair * 2],
                        acc[pair * 2 + 1],
                        values[base + pair * 2],
                        values[base + pair * 2 + 1],
                    )
            for lo, hi in ((0, 2), (4, 6), (0, 4)):
                packed("add.rn.f32x2", acc, lo, acc[lo], acc[lo + 1], acc[hi], acc[hi + 1])
            return acc[0] + acc[1]

        def combine_int_frac_ex2(x_rounded, frac_ex2):
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

        def ex2_emulation_2(values, base):
            clamped = K.alloc_local((2,), "float32")
            K.ptx.max.f32(clamped[0], values[base], K.float32(-127.0))
            K.ptx.max.f32(clamped[1], values[base + 1], K.float32(-127.0))
            packed_v = K.local_scalar("uint64")
            rhs = K.local_scalar("uint64")
            addend = K.local_scalar("uint64")
            rounded = K.alloc_local((2,), "float32")
            K.ptx.mov.b64(packed_v, clamped[0], clamped[1])
            K.ptx.mov.b64(rhs, K.float32(FP32_ROUND_INT), K.float32(FP32_ROUND_INT))
            K.ptx.add.rm.f32x2(packed_v, packed_v, rhs)
            K.ptx.mov.b64(rounded[0], rounded[1], packed_v)
            rounded_back = K.alloc_local((2,), "float32")
            K.ptx.mov.b64(packed_v, rounded[0], rounded[1])
            K.ptx.sub.rn.f32x2(packed_v, packed_v, rhs)
            K.ptx.mov.b64(rounded_back[0], rounded_back[1], packed_v)
            frac = K.alloc_local((2,), "float32")
            K.ptx.mov.b64(packed_v, clamped[0], clamped[1])
            K.ptx.mov.b64(rhs, rounded_back[0], rounded_back[1])
            K.ptx.sub.rn.f32x2(packed_v, packed_v, rhs)
            K.ptx.mov.b64(frac[0], frac[1], packed_v)
            poly = K.alloc_local((2,), "float32")
            K.assign(poly[0], K.float32(POLY_EX2_3[3]))
            K.assign(poly[1], K.float32(POLY_EX2_3[3]))
            for coeff in (POLY_EX2_3[2], POLY_EX2_3[1], POLY_EX2_3[0]):
                K.ptx.mov.b64(packed_v, poly[0], poly[1])
                K.ptx.mov.b64(rhs, frac[0], frac[1])
                K.ptx.mov.b64(addend, K.float32(coeff), K.float32(coeff))
                K.ptx.fma.rn.f32x2(packed_v, packed_v, rhs, addend)
                K.ptx.mov.b64(poly[0], poly[1], packed_v)
            K.assign(values[base], combine_int_frac_ex2(rounded[0], poly[0]))
            K.assign(values[base + 1], combine_int_frac_ex2(rounded[1], poly[1]))

        def tmem_load32(dst, address):
            K.ptx[TMEM_LD32](*(dst[i] for i in range(32)), K.cast(address, "uint32"))

        def tmem_load16(dst, address):
            K.ptx[TMEM_LD16](*(dst[i] for i in range(16)), K.cast(address, "uint32"))

        def tmem_store16(src, address):
            K.ptx[TMEM_ST16](K.cast(address, "uint32"), *(src[i] for i in range(16)))

        def stats_arrive(stage, warp_in_group):
            K.ptx.bar.arrive(K.cast(3 + stage * 4 + warp_in_group, "uint32"), K.uint32(64))

        def stats_sync(stage, warp_in_group):
            K.ptx.bar.sync(K.cast(3 + stage * 4 + warp_in_group, "uint32"), K.uint32(64))

        def ld_stats(index):
            value = K.local_scalar("float32")
            K.ptx.ld.shared.f32(value, stats_smem.ptr_to([index]))
            return value

        def st_stats(index, value):
            K.ptx.st.shared.f32(stats_smem.ptr_to([index]), value)

                                                                             
                                                          
        def encode(view, major):
            desc, off16 = view.encode(major=major, mma_k=16)
            lo = K.alloc_local((1,), "uint32")
            hi = K.alloc_local((1,), "uint32")
            K.assign(lo[0], K.uniform(K.Cast("uint32", desc.value)))
            K.assign(hi[0], K.Cast("uint32", K.shift_right(desc.value, K.uint64(32))))
            return (lo, hi), off16

        def desc_at(desc, off16):
            lo, hi = desc
            low = lo[0] if isinstance(off16, int) and off16 == 0 else lo[0] + K.Cast("uint32", off16)
            packed_desc = K.alloc_local((1,), "uint64")
            K.assign(
                packed_desc[0],
                K.bitwise_or(K.shift_left(K.Cast("uint64", hi[0]), K.uint64(32)), K.Cast("uint64", low)),
            )
            return packed_desc[0]

        def iket_range(name):
            token = K.alloc_local([1], "uint32")
            if use_iket:
                K.assign(token[0], K.cuda.iket.range_start(name))
            return token

        def iket_end(token):
            if use_iket:
                K.cuda.iket.range_end(token[0])

        def iket_mark(name):
            if use_iket:
                K.cuda.iket.mark(name)

        def task_coords(task):
            head = K.local_scalar("int32", init=task // pairs_per_head)
            q_linear = K.local_scalar("int32")
            K.ptx.ld.global_.s32(
                q_linear,
                pair_qblocks.ptr_to([task * 2 + cluster_rank]),
            )
            valid = q_linear >= 0
            q0 = K.local_scalar("int32")
            K.ptx.ld.global_.s32(q0, pair_qblocks.ptr_to([task * 2]))
            q_block = K.local_scalar(
                "int32", init=K.if_then_else(valid, q_linear, q0)
            )
            return head, q_block, valid

                                                                                 
        with r_idle:
            pass

                                                                                 
        with r_load:
            leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
            q_producer_phase = K.local_scalar("int32", init=1)
            kv_stage = K.local_scalar("int32", init=0)
            kv_phase = K.local_scalar("int32", init=1)
            ready_phase = K.local_scalar("int32", init=0)
            ack_phase = K.local_scalar("int32", init=0)
            task = K.local_scalar("int32", init=cluster)
            cluster_kv_u64 = K.local_scalar("uint64")
            K.ptx.cvta.to.shared__cluster.u64(
                cluster_kv_u64, kv_smem[0].ptr_to(0, 0)
            )
            cluster_kv = K.local_scalar(
                "uint32", init=K.cast(cluster_kv_u64, "uint32")
            )

            def advance_kv():
                K.assign(kv_stage, kv_stage + 1)
                with K.If(kv_stage == KV_STAGES), K.Then():
                    K.assign(kv_stage, 0)
                    K.assign(kv_phase, kv_phase ^ 1)

            with K.While(task < num_pair_tasks):
                head, q_block, _valid = task_coords(task)
                idx_base = (task * 2 + cluster_rank) * topk
                head2 = head * 2
                common_count = K.local_scalar("int32")
                K.ptx["ld.global.b32"](
                    common_count, common_counts.ptr_to([task])
                )

                def load_kv(is_v, logical):
                    tok_w = iket_range("load-wait-empty")
                    kv_empty.wait(kv_stage, kv_phase)
                    iket_end(tok_w)
                    if phantom:
                        logical = K.min(logical, topk - 1)
                    sid = K.local_scalar("int32", init=0)
                    K.ptx["ld.global.b32"](sid, q2k.ptr_to([idx_base + logical]), pred=leader)
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        kv_full.ptr_to([kv_stage]), K.uint32(TILE_BYTES), pred=leader
                    )
                    with K.If(logical >= topk - common_count):
                        with K.Then():
                            with K.If(cluster_rank == 0):
                                with K.Then():
                                    peer_ready.wait(0, ready_phase)
                                    K.ptx[TMA_G2S_MCAST](
                                        cluster_kv + K.cast(kv_stage * TILE_BYTES, "uint32"),
                                        K.address_of(v_map if is_v else k_map),
                                        K.int32(0),
                                        sid * BLK,
                                        head2,
                                        K.cuda.cvta_generic_to_shared(
                                            kv_full.ptr_to([kv_stage])
                                        ),
                                        K.uint16(3),
                                        K.uint64(kv_policy),
                                        pred=leader,
                                    )
                                    peer_ack.arrive(0, remote=1, pred=leader)
                                    K.assign(ready_phase, ready_phase ^ 1)
                                with K.Else():
                                    peer_ready.arrive(0, remote=0, pred=leader)
                                    peer_ack.wait(0, ack_phase)
                                    K.assign(ack_phase, ack_phase ^ 1)
                        with K.Else():
                            K.ptx[TMA_G2S](
                                kv_smem[kv_stage].ptr_to(0, 0),
                                K.address_of(v_map if is_v else k_map),
                                K.int32(0),
                                sid * BLK,
                                head2,
                                K.cuda.cvta_generic_to_shared(
                                    kv_full.ptr_to([kv_stage])
                                ),
                                K.uint64(kv_policy),
                                pred=leader,
                            )
                    advance_kv()

                load_kv(False, count - 1)
                q_empty.wait(0, q_producer_phase)
                K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                    q_full.ptr_to([0]), K.uint32(TILE_BYTES), pred=leader
                )
                K.ptx[TMA_G2S](
                    q_smem.ptr_to(0, 0),
                    K.address_of(q_map),
                    K.int32(0),
                    q_block * BLK,
                    head2,
                    K.cuda.cvta_generic_to_shared(q_full.ptr_to([0])),
                    K.uint64(q_policy),
                    pred=leader,
                )
                K.assign(q_producer_phase, q_producer_phase ^ 1)
                load_kv(False, count - 2)
                i = K.local_scalar("int32", init=0)
                with K.While(i < count - 2):
                    load_kv(True, count - 1 - i)
                    load_kv(False, count - 3 - i)
                    K.assign(i, i + 1)
                load_kv(True, 1)
                load_kv(True, 0)
                K.assign(task, task + num_clusters)

                                                                                
        with r_mma:
            K.ptx[TMEM_ALLOC](K.address_of(tmem_mailbox[0]), K.uint32(TMEM_COLS))
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            tmem_base = K.local_scalar("uint32", init=K.uint32(0))
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
            q_desc, qoff = encode(q_smem, "k")
            k_desc, koff = encode(kv_smem[0], "k")
            v_desc, voff = encode(kv_smem[0], "mn")
            q_consumer_phase = K.local_scalar("int32", init=0)
            kv_stage = K.local_scalar("int32", init=0)
            kv_phase = K.local_scalar("int32", init=0)
            spo_phase0 = K.local_scalar("int32", init=0)
            spo_phase1 = K.local_scalar("int32", init=0)
            acc0 = K.local_scalar("int32", init=0)
            acc1 = K.local_scalar("int32", init=0)
            task = K.local_scalar("int32", init=cluster)

            def advance_kv():
                K.assign(kv_stage, kv_stage + 1)
                with K.If(kv_stage == KV_STAGES), K.Then():
                    K.assign(kv_stage, 0)
                    K.assign(kv_phase, kv_phase ^ 1)

            def commit(bar, stage):
                K.ptx[TCGEN_COMMIT](bar.ptr_to([stage]), pred=leader)

            def issue_qk(score_stage, k_stage):
                for k16 in range(D // 16):
                    K.ptx[MMA_F16](
                        K.cast(tmem_base + score_stage * 128, "uint32"),
                        desc_at(q_desc, qoff(k16)),
                        desc_at(k_desc, k_stage * STAGE16 + koff(k16)),
                        K.uint32(QK_IDESC),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.cast(k16 != 0, "bool"),
                        pred=leader,
                    )

            def issue_pv(score_stage, v_stage, accumulate, phase):
                for k16 in range(8):
                    if k16 == 6:
                        plast_full.wait(score_stage, phase)
                    K.ptx[MMA_F16](
                        K.cast(tmem_base + 256 + score_stage * 128, "uint32"),
                        K.cast(tmem_base + 64 + score_stage * 128 + k16 * 8, "uint32"),
                        desc_at(v_desc, v_stage * STAGE16 + voff(k16)),
                        K.uint32(PV_IDESC),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.uint32(0),
                        K.cast(accumulate != 0 if k16 == 0 else True, "bool"),
                        pred=leader,
                    )

            with K.While(task < num_pair_tasks):
                iket_mark("mma-task-begin")
                K.assign(acc0, 0)
                K.assign(acc1, 0)
                tok = iket_range("mma-wait-q")
                q_full.wait(0, q_consumer_phase)
                iket_end(tok)
                K.assign(q_consumer_phase, q_consumer_phase ^ 1)
                for score_stage in range(2):
                    kv_full.wait(kv_stage, kv_phase)
                    issue_qk(score_stage, kv_stage)
                    commit(spo_full, score_stage)
                    commit(kv_empty, kv_stage)
                    advance_kv()
                pair = K.local_scalar("int32", init=0)
                with K.While(pair < pair_count):
                    for score_stage in range(2):
                        phase = spo_phase0 if score_stage == 0 else spo_phase1
                        acc = acc0 if score_stage == 0 else acc1
                        tok_kv = iket_range("mma-wait-kv")
                        kv_full.wait(kv_stage, kv_phase)
                        v_stage = K.local_scalar("int32", init=kv_stage)
                        advance_kv()
                        kv_full.wait(kv_stage, kv_phase)
                        iket_end(tok_kv)
                        tok_p = iket_range("mma-wait-p")
                        spo_empty.wait(score_stage, phase)
                        iket_end(tok_p)
                        issue_pv(score_stage, v_stage, acc, phase)
                        issue_qk(score_stage, kv_stage)
                        commit(spo_full, score_stage)
                        K.assign(phase, phase ^ 1)
                        K.assign(acc, 1)
                        commit(kv_empty, v_stage)
                        commit(kv_empty, kv_stage)
                        advance_kv()
                    K.assign(pair, pair + 1)
                commit(q_empty, 0)
                for score_stage in range(2):
                    phase = spo_phase0 if score_stage == 0 else spo_phase1
                    acc = acc0 if score_stage == 0 else acc1
                    kv_full.wait(kv_stage, kv_phase)
                    spo_empty.wait(score_stage, phase)
                    issue_pv(score_stage, kv_stage, acc, phase)
                    commit(oacc_full, score_stage)
                    commit(kv_empty, kv_stage)
                    advance_kv()
                K.assign(spo_phase0, spo_phase0 ^ 1)
                K.assign(spo_phase1, spo_phase1 ^ 1)
                K.assign(task, task + num_clusters)

            K.ptx[TMEM_RELINQUISH]()
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            allocated = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(allocated, tmem_mailbox.ptr_to([0]))
            K.ptx[TMEM_DEALLOC](allocated, K.uint32(TMEM_COLS))

                                                                                     
        with r_epilogue:
            leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
            oepi_consumer_phase = K.local_scalar("int32", init=0)
            task = K.local_scalar("int32", init=cluster)
            with K.While(task < num_pair_tasks):
                head, q_block, valid = task_coords(task)
                tok_ep = iket_range("epi-wait-full")
                oepi_full.wait(0, oepi_consumer_phase)
                iket_end(tok_ep)
                K.ptx[TMA_S2G](
                    K.address_of(o_map),
                    K.int32(0),
                    q_block * BLK,
                    head * 2,
                    o_smem.ptr_to(0, 0),
                    K.uint64(o_policy),
                    pred=leader & valid,
                )
                K.ptx.cp.async_.bulk.commit_group()
                K.ptx.cp.async_.bulk.wait_group.read(0)
                K.ptx.mbarrier.arrive.shared.b64(oepi_empty.ptr_to([0]), K.uint32(1), pred=leader)
                K.assign(oepi_consumer_phase, oepi_consumer_phase ^ 1)
                K.assign(task, task + num_clusters)

                                                                                    
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
            row_max = K.local_scalar("float32", init=K.float32(NEG_INF))
            row_sum = K.local_scalar("float32", init=K.float32(0.0))
            task = K.local_scalar("int32", init=cluster)

            def consume_score(first, masked):
                tok_s = iket_range("sm-wait-s")
                spo_full.wait(score_stage, score_phase)
                iket_end(tok_s)
                tok_c = iket_range("sm-compute")
                score = K.alloc_local((128,), "float32")
                for chunk in range(4):
                    regs = K.alloc_local((32,), "float32")
                    tmem_load32(regs, tmem_base + score_stage * 128 + chunk * 32 + row_hi)
                    for j in range(32):
                        K.assign(score[chunk * 32 + j], regs[j])
                if masked:
                                                                                  
                    with K.If(score_stage == 0), K.Then():
                        for j in range(128):
                            K.assign(score[j], K.float32(NEG_INF))
                old_scale = K.local_scalar("float32", init=K.float32(0.0))
                new_max = K.local_scalar("float32")
                max_safe = K.local_scalar("float32")
                if first:
                    tile_max = reduce_max_128(score)
                    K.assign(new_max, tile_max)
                    K.assign(max_safe, K.if_then_else(tile_max != K.float32(NEG_INF), tile_max, 0.0))
                else:
                    tile_max = reduce_max_128(score, row_max)
                    K.assign(new_max, tile_max)
                    K.assign(max_safe, K.if_then_else(new_max != K.float32(NEG_INF), new_max, 0.0))
                    delta = K.local_scalar("float32")
                    K.ptx.sub.f32(delta, row_max, max_safe)
                    delta_scaled = K.local_scalar("float32")
                    K.ptx.mul.f32(delta_scaled, delta, scale_log2)
                    K.assign(old_scale, exp2(delta_scaled))
                    with K.If(delta_scaled >= K.float32(-8.0)), K.Then():
                        K.assign(new_max, row_max)
                        K.assign(max_safe, row_max)
                        K.assign(old_scale, K.float32(1.0))
                    st_stats(score_stage * 128 + tid128, old_scale)
                stats_arrive(score_stage, local_warp)

                negative_max = K.local_scalar("float32")
                K.ptx.mul.f32(negative_max, max_safe, -scale_log2)
                for pair in range(64):
                    base = pair * 2
                    packed(
                        "fma.rn.f32x2",
                        score,
                        base,
                        score[base],
                        score[base + 1],
                        scale_log2,
                        scale_log2,
                        negative_max,
                        negative_max,
                    )
                for fragment in range(4):
                    for pair in range(16):
                        base = fragment * 32 + pair * 2
                        if _emu_pair(emu_mode, fragment, pair):
                            ex2_emulation_2(score, base)
                        else:
                            K.assign(score[base], exp2(score[base]))
                            K.assign(score[base + 1], exp2(score[base + 1]))
                    packed_p = K.alloc_local((16,), "uint32")
                    for pair in range(16):
                        base = fragment * 32 + pair * 2
                        K.ptx.cvt.rn.bf16x2.f32(packed_p[pair], score[base + 1], score[base])
                    tmem_store16(packed_p, tmem_base + 64 + score_stage * 128 + fragment * 16 + row_hi)
                    if fragment == 2:
                        K.ptx.tcgen05.wait__st.sync.aligned()
                        spo_empty.arrive(score_stage)
                K.ptx.tcgen05.wait__st.sync.aligned()
                K.cuda.warp_sync()
                with K.If(K.cuda.elect_sync()), K.Then():
                    plast_full.arrive(score_stage)
                iket_end(tok_c)

                tok_e = iket_range("sm-wait-stats-empty")
                stats_empty.wait(score_stage, stats_producer_phase)
                iket_end(tok_e)
                K.assign(row_sum, packed_sum_128(score, row_sum, old_scale, first))
                K.assign(row_max, new_max)
                K.assign(stats_producer_phase, stats_producer_phase ^ 1)
                K.assign(score_phase, score_phase ^ 1)

            with K.While(task < num_pair_tasks):
                K.assign(row_max, K.float32(NEG_INF))
                K.assign(row_sum, K.float32(0.0))
                stats_empty.wait(score_stage, stats_producer_phase)
                K.assign(stats_producer_phase, stats_producer_phase ^ 1)
                consume_score(True, phantom)
                iteration = K.local_scalar("int32", init=1)
                with K.While(iteration < work_groups):
                    consume_score(False, False)
                    K.assign(iteration, iteration + 1)
                st_stats(score_stage * 128 + tid128, row_sum)
                st_stats(256 + score_stage * 128 + tid128, row_max)
                stats_arrive(score_stage, local_warp)
                K.assign(task, task + num_clusters)
            stats_empty.wait(score_stage, stats_producer_phase)
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
            task = K.local_scalar("int32", init=cluster)
            spo_empty.arrive(0)
            spo_empty.arrive(1)

            def rescale_o(score_stage, scale):
                for chunk in range(D // 16):
                    values = K.alloc_local((16,), "float32")
                    address = tmem_base + 256 + score_stage * 128 + chunk * 16 + row_hi
                    tmem_load16(values, address)
                    for pair in range(8):
                        base = pair * 2
                        packed("mul.rn.f32x2", values, base, values[base], values[base + 1], scale, scale)
                    tmem_store16(values, address)
                K.ptx.tcgen05.wait__st.sync.aligned()

            def store_combined(scale0, scale1):
                for chunk in range(D // 16):
                    values0 = K.alloc_local((16,), "float32")
                    values1 = K.alloc_local((16,), "float32")
                    combined = K.alloc_local((16,), "float32")
                    tmem_load16(values0, tmem_base + 256 + chunk * 16 + row_hi)
                    tmem_load16(values1, tmem_base + 384 + chunk * 16 + row_hi)
                    for pair in range(8):
                        base = pair * 2
                        scaled0 = K.alloc_local((2,), "float32")
                        scaled1 = K.alloc_local((2,), "float32")
                        packed("mul.rn.f32x2", scaled0, 0, values0[base], values0[base + 1], scale0, scale0)
                        packed("mul.rn.f32x2", scaled1, 0, values1[base], values1[base + 1], scale1, scale1)
                        packed("add.rn.f32x2", combined, base, scaled0[0], scaled0[1], scaled1[0], scaled1[1])
                    words = K.alloc_local((8,), "uint32")
                    for pair in range(8):
                        K.ptx.cvt.rn.bf16x2.f32(words[pair], combined[pair * 2 + 1], combined[pair * 2])
                    K.ptx.st.shared.v4.u32(
                        o_smem.ptr_to(tid128, chunk * 16), words[0], words[1], words[2], words[3]
                    )
                    K.ptx.st.shared.v4.u32(
                        o_smem.ptr_to(tid128, chunk * 16 + 8), words[4], words[5], words[6], words[7]
                    )

            with K.While(task < num_pair_tasks):
                                                                                                    
                                                                                                     
                                                                 
                tile = K.local_scalar("int32", init=0)
                with K.While(tile < work_groups):
                    for score_stage in range(2):
                        stats_sync(score_stage, corr_warp)
                        scale = K.local_scalar("float32", init=K.float32(1.0))
                        with K.If(tile > 0), K.Then():
                            K.assign(scale, ld_stats(score_stage * 128 + tid128))
                            ballot = K.local_scalar("uint32")
                            K.ptx.vote_sync.ballot.b32(ballot, K.ptx.pred(scale < K.float32(1.0)), K.uint32(0xFFFFFFFF))
                            with K.If(ballot != 0), K.Then():
                                tok_r = iket_range("corr-rescale")
                                rescale_o(score_stage, scale)
                                iket_end(tok_r)
                            spo_empty.arrive(score_stage)
                        stats_empty.arrive(score_stage)
                    K.assign(tile, tile + 1)

                sum0 = K.local_scalar("float32")
                sum1 = K.local_scalar("float32")
                maximum0 = K.local_scalar("float32")
                maximum1 = K.local_scalar("float32")
                for score_stage in range(2):
                    stats_sync(score_stage, corr_warp)
                    if score_stage == 0:
                        K.assign(sum0, ld_stats(tid128))
                        K.assign(maximum0, ld_stats(256 + tid128))
                    else:
                        K.assign(sum1, ld_stats(128 + tid128))
                        K.assign(maximum1, ld_stats(384 + tid128))
                    stats_empty.arrive(score_stage)
                valid0 = (sum0 != K.float32(0.0)) & (sum0 == sum0)
                valid1 = (sum1 != K.float32(0.0)) & (sum1 == sum1)
                rm0 = K.if_then_else(valid0, maximum0, K.float32(NEG_INF))
                rm1 = K.if_then_else(valid1, maximum1, K.float32(NEG_INF))
                maximum = K.local_scalar("float32")
                K.ptx.max.f32(maximum, rm0, rm1)
                safe_max = K.local_scalar("float32")
                K.assign(safe_max, K.if_then_else(maximum != K.float32(NEG_INF), maximum, 0.0))
                scale0 = K.local_scalar("float32")
                scale1 = K.local_scalar("float32")
                K.assign(scale0, K.if_then_else(valid0, exp2((rm0 - safe_max) * scale_log2), 0.0))
                K.assign(scale1, K.if_then_else(valid1, exp2((rm1 - safe_max) * scale_log2), 0.0))
                total_sum = K.local_scalar("float32", init=sum0 * scale0 + sum1 * scale1)
                valid_total = (total_sum != K.float32(0.0)) & (total_sum == total_sum)
                inv_sum = rcp(K.if_then_else(valid_total, total_sum, 1.0))
                final_scale0 = K.local_scalar("float32", init=scale0 * inv_sum)
                final_scale1 = K.local_scalar("float32", init=scale1 * inv_sum)
                tok_o = iket_range("corr-wait-oacc")
                oacc_full.wait(0, oacc_consumer_phase)
                oacc_full.wait(1, oacc_consumer_phase)
                oepi_empty.wait(0, oepi_producer_phase)
                iket_end(tok_o)
                tok_st = iket_range("corr-store")
                store_combined(final_scale0, final_scale1)
                iket_end(tok_st)
                K.ptx.fence.proxy.async_.shared__cta()
                spo_empty.arrive(0)
                spo_empty.arrive(1)
                oepi_full.arrive(0)
                K.assign(oacc_consumer_phase, oacc_consumer_phase ^ 1)
                K.assign(oepi_producer_phase, oepi_producer_phase ^ 1)
                K.assign(task, task + num_clusters)
            oepi_empty.wait(0, oepi_producer_phase)
            K.ptx.bar.arrive(K.uint32(2), K.uint32(416))

    def vsa_blk128(
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        o_map: K.TensorMap,
        q2k: K.gptr[K.i32, (num_pair_tasks * 2 * topk,)],
        common_counts: K.gptr[K.i32, (num_pair_tasks,)],
        pair_qblocks: K.gptr[K.i32, (num_pair_tasks * 2,)],
        scale_log2: K.f32,
    ):
        kernel_body(
            q_map,
            k_map,
            v_map,
            o_map,
            q2k,
            common_counts,
            pair_qblocks,
            scale_log2,
        )

    return K.kernel(warps=WARPS, arch="sm_100a", min_blocks_per_sm=1, grid=num_ctas)(vsa_blk128)


                                                                                  
class _AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode_map(tensor, S, H):
    """Rank-3 map over bf16[S, H, 128]: dim0 = 64-col band, dim1 = token, dim2 = head*2+band."""
    desc = _AlignedTensorMap()
    dims = (64, S, H * 2)
    strides = (H * D * 2, 128)                                    
    box = (64, BLK, 2)
    tvm.get_global_func("runtime.cuTensorMapEncodeTiled")(
        desc.ptr,
        "bfloat16",
        3,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *dims,
        *strides,
        *box,
        *((1,) * 3),
        0,                   
        3,                
        2,                     
        0,                 
    )
    return desc



# ---------------------------------------------------------------------------
# Registry interface.
# ---------------------------------------------------------------------------

KERNEL_META = {
    "name": "agent_evolved_vsa_s80000_h8_topk156",
    "category": "agent_evolved",
    "runtime_cuda_archs": ["sm_100a"],
    "reference_requirements": (
        {
            "package": "flashinfer-python",
            "specifier": ">=0.6.18",
            "import": "flashinfer",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.7.0", "import": "cutlass"},
    ),
    "provenance": {
        "generator": "hmz",
        "run": "vsa_s80000_h8_d128_blk128_topk156_bf16-20260910-221104",
        "selected_version": "frontier/group16-cluster-multicast",
    },
}

CONFIGS = [
    {
        "label": "s80000_h8_blk128_topk156",
        "seqlen": 80000,
        "num_heads": 8,
        "block_size": 128,
        "topk": 156,
        "seed": 0,
    }
]


def _config(**config: Any) -> dict[str, Any]:
    """Validate one config against the contract this kernel implements."""
    values = {key: value for key, value in config.items() if key != "label"}
    unknown = set(values) - {"seqlen", "num_heads", "block_size", "topk", "seed"}
    if unknown:
        raise ValueError(f"unsupported config keys: {sorted(unknown)}")
    resolved = {**CONFIGS[0], **values}
    resolved.pop("label", None)
    if int(resolved["block_size"]) != BLK:
        raise ValueError("this kernel implements block_size=128 only")
    seqlen = int(resolved["seqlen"])
    if seqlen % BLK != 0:
        raise ValueError(f"seqlen must be a multiple of {BLK}")
    num_blocks = seqlen // BLK
    if not 1 <= int(resolved["topk"]) <= num_blocks:
        raise ValueError(f"topk must be in [1, {num_blocks}]")
    return resolved


def _assert_supported_arch() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved VSA forward")
    capability = torch.cuda.get_device_capability()
    runtime_arch = f"sm_{capability[0]}{capability[1]}a"
    if runtime_arch not in KERNEL_META["runtime_cuda_archs"]:
        raise SkipTest(
            "agent-evolved VSA forward requires one of "
            f"{KERNEL_META['runtime_cuda_archs']}, got {runtime_arch}"
        )


def current_cfg() -> dict[str, Any]:
    """The frontier member's fixed schedule configuration (no environment overrides)."""
    return dict(DEFAULT_CFG)


def _num_ctas(num_pair_tasks: int) -> int:
    """One 2-CTA cluster per pair task, capped at half the SMs (clusters are pairs)."""
    from tirx_kernels.runner import hardware_num_sms

    return 2 * max(1, min(hardware_num_sms() // 2, num_pair_tasks))


def _pair_tasks(seqlen: int, num_heads: int) -> int:
    return num_heads * ((seqlen // BLK + 1) // 2)


def get_kernel(**config: Any):
    """Return both traced PrimFuncs: the pair scheduler and the attention body."""
    resolved = _config(**config)
    seqlen = int(resolved["seqlen"])
    num_heads = int(resolved["num_heads"])
    topk = int(resolved["topk"])
    num_ctas = _num_ctas(_pair_tasks(seqlen, num_heads))
    return {
        "pair": make_pair_kernel_bitset(seqlen, num_heads, topk).func,
        "attention": make_kernel(seqlen, num_heads, topk, num_ctas, current_cfg()).func,
    }


# ---------------------------------------------------------------------------
# Inputs.
#
# The draw order reproduces the packaged VSA benchmark row
# `vsa-pooled-blk128-s80000-h8-topk156`, which follows flashinfer PR #4612:
# q, k and v come from one bf16 `randn` generator sequence, then the selection
# is PR #4612's compress step, the top-k KV blocks per (head, query block) by
# softmax of the mean-pooled block scores, emitted as ascending block ids.
# ---------------------------------------------------------------------------


def _pooled_topk_indices(q, k, block_size, topk, sm_scale):
    seqlen, num_heads, head_dim = q.shape
    query_blocks = seqlen // block_size
    key_blocks = k.shape[0] // block_size
    q_pooled = q.view(query_blocks, block_size, num_heads, head_dim).float().mean(1).permute(1, 0, 2)
    k_pooled = k.view(key_blocks, block_size, num_heads, head_dim).float().mean(1).permute(1, 0, 2)
    scores = torch.softmax(q_pooled @ k_pooled.transpose(-1, -2) * sm_scale, dim=-1)
    selected = torch.topk(scores, topk, dim=-1).indices.sort(dim=-1).values
    return selected.to(torch.int32).contiguous()


def prepare_data(**config: Any) -> dict[str, Any]:
    """Build one row's contract tensors plus the preallocated output."""
    resolved = _config(**config)
    device = torch.device("cuda")
    seqlen = int(resolved["seqlen"])
    num_heads = int(resolved["num_heads"])
    block_size = int(resolved["block_size"])
    topk = int(resolved["topk"])
    sm_scale = 1.0 / math.sqrt(D)
    generator = torch.Generator(device=device).manual_seed(int(resolved["seed"]))

    def randn():
        return torch.randn(
            (seqlen, num_heads, D), dtype=torch.bfloat16, device=device, generator=generator
        )

    q, k, v = randn(), randn(), randn()
    q2k_indices = _pooled_topk_indices(q, k, block_size, topk, sm_scale)
    return {
        "config": resolved,
        "q": q,
        "k": k,
        "v": v,
        "q2k_indices": q2k_indices,
        "kv_block_lens": None,
        "block_size": block_size,
        "sm_scale": sm_scale,
        "output": torch.empty_like(q),
    }


def _launch_state(case: dict[str, Any], executables: dict[str, Any]):
    """Encode the tensor maps and allocate the scheduler's output buffers."""
    q, k, v = case["q"], case["k"], case["v"]
    out = case["output"]
    resolved = case["config"]
    seqlen = int(resolved["seqlen"])
    num_heads = int(resolved["num_heads"])
    topk = int(resolved["topk"])
    idx_flat = case["q2k_indices"].contiguous().reshape(-1)
    num_pair_tasks = _pair_tasks(seqlen, num_heads)
    maps = (
        _encode_map(q, seqlen, num_heads),
        _encode_map(k, seqlen, num_heads),
        _encode_map(v, seqlen, num_heads),
        _encode_map(out, seqlen, num_heads),
    )
    reordered = torch.empty(num_pair_tasks * 2 * topk, dtype=torch.int32, device=q.device)
    common_counts = torch.empty(num_pair_tasks, dtype=torch.int32, device=q.device)
    pair_qblocks = torch.empty(num_pair_tasks * 2, dtype=torch.int32, device=q.device)
    argv = (
        maps[0].ptr,
        maps[1].ptr,
        maps[2].ptr,
        maps[3].ptr,
        reordered,
        common_counts,
        pair_qblocks,
        float(case["sm_scale"]) * LOG2_E,
    )
    pair, attention = executables["pair"], executables["attention"]

    def run():
        pair(idx_flat, pair_qblocks, reordered, common_counts)
        attention(*argv)

    run._keep_alive = (maps, idx_flat, reordered, common_counts, pair_qblocks, q, k, v, out, argv)
    return run


def _compile_all(**config: Any) -> dict[str, Any]:
    from tirx_kernels.runner import compile_kernel

    kernels = get_kernel(**config)
    return {name: compile_kernel(func) for name, func in kernels.items()}


# ---------------------------------------------------------------------------
# Independent oracle.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _reference_output(case: dict[str, Any], *, head_chunk: int = 1) -> torch.Tensor:
    """Masked FP32 block-sparse attention, computed independently of the kernel.

    A query block attends exactly the KV blocks its ``q2k_indices`` row lists.
    There is no causal mask, and with ``kv_block_lens = None`` every token of a
    selected block is valid.
    """
    q, k, v = case["q"], case["k"], case["v"]
    q2k = case["q2k_indices"]
    scale = float(case["sm_scale"])
    block_size = int(case["block_size"])
    seqlen, num_heads, head_dim = q.shape
    query_blocks = seqlen // block_size
    out = torch.empty_like(q)
    for head in range(0, num_heads, head_chunk):
        heads = slice(head, min(head + head_chunk, num_heads))
        k_head = k[:, heads].float()
        v_head = v[:, heads].float()
        for block in range(query_blocks):
            rows = slice(block * block_size, (block + 1) * block_size)
            selected = q2k[heads, block].long()
            # Gather the selected KV blocks: [heads, topk*block_size, D].
            token_ids = (
                selected.unsqueeze(-1) * block_size
                + torch.arange(block_size, device=q.device)
            ).reshape(selected.shape[0], -1)
            k_sel = torch.gather(
                k_head.permute(1, 0, 2), 1, token_ids.unsqueeze(-1).expand(-1, -1, head_dim)
            )
            v_sel = torch.gather(
                v_head.permute(1, 0, 2), 1, token_ids.unsqueeze(-1).expand(-1, -1, head_dim)
            )
            q_blk = q[rows, heads].float().permute(1, 0, 2)
            scores = torch.bmm(q_blk, k_sel.transpose(1, 2)) * scale
            weights = torch.softmax(scores, dim=-1)
            out[rows, heads] = torch.bmm(weights, v_sel).permute(1, 0, 2).to(q.dtype)
    return out


def check_correctness(outputs: dict[str, Any], **config: Any) -> None:
    _config(**config)
    first, actual, reference = outputs["first"], outputs["actual"], outputs["reference"]
    for name, tensor in (("first", first), ("actual", actual), ("reference", reference)):
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{name} output contains non-finite values")
    if not torch.equal(first, actual):
        max_abs = float((first.float() - actual.float()).abs().max())
        raise AssertionError(
            f"identical launches are not exactly repeatable; max abs diff={max_abs}"
        )
    # The contract's acceptance criterion: elementwise atol + rtol*|reference|
    # with atol = rtol = 0.01, evaluated in the native bf16 dtype.
    torch.testing.assert_close(actual, reference, atol=1e-2, rtol=1e-2)
    diff_rms = torch.sqrt(torch.mean((actual.float() - reference.float()).square()))
    reference_rms = torch.sqrt(torch.mean(reference.float().square()))
    rms_ratio = float(diff_rms / (reference_rms + 1e-8))
    if rms_ratio >= 1e-2:
        raise AssertionError(f"normalized RMS error ratio {rms_ratio:.6e} must be below 1e-2")


def run_test(**config: Any) -> None:
    _assert_supported_arch()
    case = prepare_data(**config)
    run = _launch_state(case, _compile_all(**config))
    case["output"].fill_(float("nan"))
    run()
    torch.cuda.synchronize()
    first = case["output"].clone()
    # Poison the buffer so a kernel that skips rows cannot pass by leaving them.
    case["output"].fill_(42.0)
    run()
    torch.cuda.synchronize()
    actual = case["output"].clone()
    reference = _reference_output(case)
    torch.cuda.synchronize()
    check_correctness({"first": first, "actual": actual, "reference": reference}, **config)


# ---------------------------------------------------------------------------
# Benchmark: the FlashInfer reference arm and the timed dispatch.
# ---------------------------------------------------------------------------


def _flashinfer_reference(case: dict[str, Any]):
    """Capture FlashInfer's block-sparse forward in a CUDA graph and return its replay.

    This is the arm ``BlockSparseAttentionWrapper`` compiles: ``bsa_attn_fwd``
    with the library default ``allow_empty_block_nums=True``. Building the
    batched views, the block counts and the CuTe-DSL compile are prepare work,
    as the packaged VSA baseline does them. The capture removes the host gap
    inside the timed span; the replay is bitwise equal to the eager call.
    """
    from flashinfer.cute_dsl.sparse import bsa_attn_fwd

    q, k, v = case["q"], case["k"], case["v"]
    q2k = case["q2k_indices"].to(torch.int32).unsqueeze(0).contiguous()
    counts = torch.full(q2k.shape[:3], q2k.shape[-1], dtype=torch.int32, device=q.device)
    batched = (q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
    scale = float(case["sm_scale"])

    def launch():
        output, _lse = bsa_attn_fwd(
            *batched,
            q2k,
            block_sparse_num=2,
            block_sizes=None,
            q2k_block_nums=counts,
            softmax_scale=scale,
            return_lse=True,
        )
        return output[0]

    launch()  # CuTe-DSL compile and workspace allocation
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        launch()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    state = (graph, q2k, counts, batched)

    def replay(_state=state):
        _state[0].replay()

    replay._keep_alive = state
    return replay


def prepare_bench(**config: Any):
    """Trace and compile before bench-suite assigns a GPU."""
    from tirx_kernels.runner import prepared_gpu_benchmark

    state = {"config": dict(config), "executables": _compile_all(**config)}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(
    prepared,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    _assert_supported_arch()
    from tirx_kernels.runner import bench

    config = dict(prepared["config"])
    config.update(kwargs)
    rounds = config.pop("rounds", 5)
    cooldown_s = config.pop("cooldown_s", 1.0)
    case = prepare_data(**config)
    run = _launch_state(case, prepared["executables"])
    run()
    torch.cuda.synchronize()

    return bench(
        {"tirx": run},
        references={"flashinfer_bsa": lambda: _flashinfer_reference(case)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    **config: Any,
) -> dict[str, Any]:
    values = dict(config)
    protocol = {name: values.pop(name) for name in ("rounds", "cooldown_s") if name in values}
    prepared = prepare_bench(**values)
    return prepared.run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)


__all__ = [
    "CONFIGS",
    "KERNEL_META",
    "check_correctness",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_test",
]
