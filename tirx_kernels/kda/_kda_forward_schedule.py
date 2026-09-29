# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""CPU scheduling for the packed KDA forward kernel.

Derived from ``tirx/kda_fwd_varlen.py`` in
https://github.com/humanfia/kda-for-kda-release at commit
``ea37ebaff74c88a2545751dcb8ea8ef6c6251b67``. No device tensors are retained:
the bounded cache stores immutable packed bytes and item counts.
"""

import heapq
import itertools
from array import array
from dataclasses import dataclass
from functools import lru_cache

BT = 64
MAX_ITEMS = 160
BETA = 1
SNAP = 1
START_COST = 0.9
HANDOFF_COST = 0.4


def _tok_bits(tok0):
    return (tok0 & 0xFFFF) | ((tok0 >> 16) << 23)


def _host_item_table(cu_list, H, num_ctas, force_lpt):
    """Per-CTA (word0, word1) item lists.

    Sequences are ranked by length (ties by index).  When the longest sequence is short against the mean load
    the (head, sequence, chunk) walk is cut into num_ctas equal-cost ranges (chunk cost 1, BETA per sequence
    start, cuts snapped to a sequence end within SNAP chunks; a range starting inside a sequence continues
    the state handed off by the previous CTA); otherwise whole (sequence, head) items go to the least loaded
    CTA group (LPT).  word0 = tok0[15:0] | nvalid << 16 | tok0[20:16] << 23 | dst_hand << 28 | src_hand << 29 | first << 30 | last << 31,
    word1 = seq | head << 16."""
    nseq = len(cu_list) - 1
    start = [int(cu_list[n]) for n in range(nseq)]
    length = [int(cu_list[n + 1] - cu_list[n]) for n in range(nseq)]
    order = [0] * nseq
    for n in range(nseq):
        rank = sum(
            1 for m in range(nseq) if (length[m] > length[n]) or (length[m] == length[n] and m < n)
        )
        order[rank] = n
    wh, maxnch, off, nch = 0, 0, [], []
    for r in range(nseq):
        c_ = max((length[order[r]] + BT - 1) // BT, 1)
        off.append(wh)
        nch.append(c_)
        wh += c_
        maxnch = max(maxnch, c_)
    off.append(wh)
    ch_cost = wh + BETA * nseq
    c_tot = ch_cost * H
    l_min = c_tot // num_ctas
    items = [[] for _ in range(num_ctas)]
    if maxnch <= l_min - (2 * SNAP + BETA) and not force_lpt:
        # cut positions minimising the maximum modelled range cost (chunk 1, unit start or continuation START_COST, handoff HANDOFF_COST)
        cuts = _optimal_cuts(nch, H, num_ctas)

        def cut_pos(p):
            return cuts[min(max(p, 0), num_ctas)]

        for cta in range(num_ctas):
            lo, hi = cut_pos(cta), cut_pos(cta + 1)
            for pas in range(3):
                for hh in range(lo // wh, (hi - 1) // wh + 1):
                    for rk in range(nseq):
                        c_ = nch[rk]
                        ib = hh * wh + off[rk]
                        ie = ib + c_
                        cb, ce = max(0, lo - ib), min(c_, hi - ib)
                        cont_in, head_out = cb > 0, ce < c_
                        kind = 2 if cont_in else (0 if head_out else 1)
                        if ie > lo and ib < hi and kind == pas:
                            n = order[rk]
                            s_tok, e_tok = start[n], start[n] + length[n]
                            for ch in range(cb, ce):
                                tok0 = s_tok + ch * BT
                                nvalid = max(min(e_tok - tok0, BT), 0)
                                word0 = (
                                    _tok_bits(tok0)
                                    | (nvalid << 16)
                                    | ((1 << 30) if ch == cb else 0)
                                    | ((1 << 31) if ch + 1 == ce else 0)
                                    | ((1 << 29) if (ch == cb and cont_in) else 0)
                                    | ((1 << 28) if (ch + 1 == ce and head_out) else 0)
                                )
                                items[cta].append((word0 & 0xFFFFFFFF, n | (hh << 16)))
    else:
        assert num_ctas >= H, "whole-item scheduling needs at least one CTA per head"
        # unit-level LPT: every (sequence, head) chain is one unit of cost chunks + START_COST, placed longest-first on the least loaded CTA, then a move/swap local search on the most loaded CTA
        units = sorted(
            ((nch[r] + START_COST, r, hh) for r in range(nseq) for hh in range(H)),
            key=lambda u: (-u[0], u[1], u[2]),
        )
        per = [[] for _ in range(num_ctas)]
        load = [0.0] * num_ctas
        heap = [(0.0, c) for c in range(num_ctas)]
        for cost, r, hh in units:
            l, c = heapq.heappop(heap)
            per[c].append((r, hh))
            load[c] = l + cost
            heapq.heappush(heap, (load[c], c))
        _balance_units(per, load, nch, START_COST)
        for cta in range(num_ctas):
            for r, hh in per[cta]:
                n = order[r]
                s_tok, ln = start[n], length[n]
                for ch in range(nch[r]):
                    tok0 = s_tok + ch * BT
                    nvalid = max(min(s_tok + ln - tok0, BT), 0)
                    word0 = (
                        _tok_bits(tok0)
                        | (nvalid << 16)
                        | ((1 << 30) if ch == 0 else 0)
                        | ((1 << 31) if ch + 1 == nch[r] else 0)
                    )
                    items[cta].append((word0 & 0xFFFFFFFF, n | (hh << 16)))
    return items


def _optimal_cuts(nch, H, num_ctas):
    """num_ctas + 1 cut positions in the (head, sequence-rank, chunk) walk minimising the maximum range cost."""
    walk = []
    for _ in range(H):
        for n in nch:
            for ch in range(n):
                walk.append((ch == 0, ch == n - 1))
    total = len(walk)

    def greedy(bound):
        cuts, i = [0], 0
        while i < total:
            cost, j, best_j, boundary = START_COST, i, None, False
            while j < total:
                c = cost + 1.0 + (START_COST if (j > i and walk[j][0]) else 0.0)
                if c + (0.0 if walk[j][1] else HANDOFF_COST) > bound:
                    break
                cost = c
                boundary = boundary or walk[j][1]
                if boundary:
                    best_j = j
                j += 1
            if best_j is None:
                return None
            i = best_j + 1
            cuts.append(i)
            if len(cuts) - 1 > num_ctas:
                return None
        return cuts

    lo, hi, best = 0.0, float(total) + 2 * START_COST + HANDOFF_COST, None
    for _ in range(50):
        mid = (lo + hi) / 2
        cuts = greedy(mid)
        if cuts is not None:
            hi, best = mid, cuts
        else:
            lo = mid
    if best is None:  # cannot happen (bound = total cost is always feasible), kept for safety
        best = [0, total]

    def range_cost(a, b):
        return (
            START_COST
            + (b - a)
            + START_COST * sum(1 for q in range(a + 1, b) if walk[q][0])
            + (0.0 if walk[b - 1][1] else HANDOFF_COST)
        )

    # give every CTA work: split the costliest ranges after a unit's last chunk nearest their middle
    while len(best) - 1 < num_ctas:
        done = False
        for _, i in sorted(
            ((range_cost(best[i], best[i + 1]), i) for i in range(len(best) - 1)), reverse=True
        ):
            a, b = best[i], best[i + 1]
            cands = [q + 1 for q in range(a, b - 1) if walk[q][1]]
            if cands:
                mid = (a + b) // 2
                best = [*best[: i + 1], min(cands, key=lambda x: abs(x - mid)), *best[i + 1 :]]
                done = True
                break
        if not done:
            best = best + [total] * (num_ctas - (len(best) - 1))  # remaining CTAs get empty ranges
            break
    return best


def _balance_units(per, load, nch, start_cost, max_rounds=400):
    """Lower the makespan of a unit assignment: move one unit off the most loaded CTA, or swap it with a smaller
    unit of another CTA, whenever that strictly lowers the maximum load; stop when no such step exists."""
    num_ctas = len(per)
    ucost = lambda r: nch[r] + start_cost  # noqa: E731
    for _ in range(max_rounds):
        worst = max(range(num_ctas), key=load.__getitem__)
        wl = load[worst]
        best = None
        for ui, (r, _hh) in enumerate(per[worst]):
            cu = ucost(r)
            for c in range(num_ctas):
                if c == worst:
                    continue
                nl = load[c] + cu
                if nl < wl and (best is None or nl < best[0]):
                    best = (nl, ui, c, -1)
                for vi, (r2, _h2) in enumerate(per[c]):
                    cv = ucost(r2)
                    if cv < cu:
                        top = max(wl - cu + cv, load[c] - cv + cu)
                        if top < wl and (best is None or top < best[0]):
                            best = (top, ui, c, vi)
        if best is None:
            return
        _, ui, c, vi = best
        u = per[worst][ui]
        if vi < 0:
            per[worst].pop(ui)
            per[c].append(u)
            load[worst] -= ucost(u[0])
            load[c] += ucost(u[0])
        else:
            v = per[c][vi]
            per[worst][ui], per[c][vi] = v, u
            load[worst] += ucost(v[0]) - ucost(u[0])
            load[c] += ucost(u[0]) - ucost(v[0])


@dataclass(frozen=True)
class PackedSchedule:
    words: bytes
    counts: tuple[int, ...]
    max_items: int


@lru_cache(maxsize=32)
def packed_schedule(
    cu: tuple[int, ...], heads: int, num_ctas: int, force_lpt: bool
) -> PackedSchedule:
    """Pack once per layout; callers upload independent device copies.

    ``cu`` contains cumulative sequence boundaries as host integers. Reading a
    CUDA cu_seqlens tensor is intentionally the caller's responsibility, so an
    in-place update can never hit a cache keyed only by tensor identity.
    """
    if len(cu) < 2 or len(cu) >= 65537 or cu[0] != 0 or cu[-1] <= 0:
        raise ValueError("expected 1..65535 sequences starting at zero")
    if any(a > b for a, b in itertools.pairwise(cu)):
        raise ValueError("sequence boundaries must be nondecreasing")
    if heads <= 0 or heads % 8 or heads >= 65536:
        raise ValueError("head count must be a positive multiple of eight below 65536")
    if cu[-1] >= 1 << 21 or cu[-1] * heads * 128 >= 1 << 31:
        raise ValueError("KDA schedule exceeds packed token or element offset range")
    if not 1 <= num_ctas <= heads * (len(cu) - 1):
        raise ValueError("CTA count must be in 1..heads*num_sequences")
    lists = _host_item_table(cu, heads, num_ctas, force_lpt)
    counts = tuple(map(len, lists))
    max_items = max(MAX_ITEMS, ((max(counts) + 31) // 32) * 32)
    words = array("I", [0]) * (num_ctas * max_items * 2)
    assert words.itemsize == 4
    for cta, items in enumerate(lists):
        start = cta * max_items * 2
        words[start : start + len(items) * 2] = array("I", (w for item in items for w in item))
    return PackedSchedule(words.tobytes(), counts, max_items)
