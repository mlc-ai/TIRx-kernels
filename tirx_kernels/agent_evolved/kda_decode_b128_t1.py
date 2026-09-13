# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Agent-evolved SM100a recurrent Kimi Delta Attention (KDA) decode.

The supported contract is the standard single-token decode row
``kda-decode-d128-t1-b128-h16-hv32-standard_decode-precomputed``: N = B = 128
sequences, T = L = 1 token, H = 16 qk heads, HV = 32 value heads and
K = V = 128, with one state slot per sequence (P = 128). q/k are
bf16[B, 1, H, 128], v/g are bf16[B, 1, HV, 128] and beta is bf16[B, 1, HV],
already sigmoid-activated. ``initial_state`` is a bf16[P, HV, V, K] pool in
V-first layout; ``A_log``, ``dt_bias``, ``cu_seqlens``, ``ssm_state_indices``,
``num_spec_tokens``, ``num_accepted_tokens`` and ``lower_bound`` are all
``None``, so the gates are precomputed and ``decay = exp(g)``.

q and k are normalized as ``x / sqrt(sum(x * x) + 1e-6)``; value head ``hv``
uses qk head ``hv // 2``. With a conceptual K-first state S, each token
computes ``S_decay = S * decay[:, None]``, ``S = S_decay + beta * outer(k,
v - k @ S_decay)`` and ``output = scale * q @ S``. Sequence ``n`` reads
``initial_state[n]`` and writes ``final_state[n]``; ``initial_state`` is never
mutated.

The selected kernel is the ``clc-steal`` frontier member of the 2026-09-12
KDA-decode evolution run. Everything from ``import torch`` down to the end of
``build_kernel`` is that candidate's source unchanged; this module adds the
registry interface, input generation, the independent oracle, and the
FlashInfer ``recurrent_kda`` reference arm.

Candidate mechanism notes, carried over from the evolution run:

Family: **clc-steal**. The grid has one CTA per 64-row state unit (8192), but
only the resident CTAs ever run: each running CTA keeps a small TMA ring
(``nstage`` 16 KiB stages) and obtains its next unit by cancelling a
not-yet-launched CTA with ``clusterlaunchcontrol.try_cancel`` (Blackwell
cluster launch control). The cancellation response arrives asynchronously on
an mbarrier, so the next unit's bulk load is issued while the current unit is
being computed and no L2 atomic ever sits on the issue path. Load balancing
across SMs is therefore done by hardware at unit granularity with a lookahead
of ``lookahead`` units, while the pipelining of the persistent ring is
preserved. Thread 0 of the four consumer warps doubles as the I/O and
scheduling lane; the unit id is passed to the other warps through a per-stage
shared header word.

The workload moves 256 MiB of state under the harness's protocol, which
zero-fills a 2xL2 buffer before every launch and leaves L2 full of dirty
lines. A plain 128 MiB device copy measures 42.3-43.6 us under that protocol,
so this kernel is at the memory floor: NCU puts it at roughly 81% of DRAM
peak (82% read, 72% write), and the residual loss is die-to-die fabric
traffic -- about half of all sectors cross the fabric at 26% of its peak --
rather than anything the kernel schedules.

Measured negative results from that run, recorded so they are not retried:
dynamic atomic-ticket and hybrid static-then-dynamic tail schedulers (the
per-SM finish spread is bandwidth-share variance, not load imbalance);
32-row units (fixed work doubles per byte); mixed unit sizes; four-stage
lookahead; keeping one bulk store outstanding; delivering beta through the
stage's TMA auxiliary copy (its global load is already hidden behind the
prologue); and packed-BF16 updates, which SM100 cannot express -- its
mixed-precision ISA has only scalar BF16-input/FP32-accumulate FMA, and the
packed mixed form starts at the SM107 family.
"""

from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K

H = 16
HV = 32
D = 128
GROUP = HV // H
LOG2E = 1.4426950408889634
FULL = 0xFFFFFFFF
AUX_BYTES = 4 * D * 2

CONFIG = dict(
    consumer_warps=4,
    ilp=8,
    lanes_per_row=16,
    unit_rows=64,
    nstage=3,
    lookahead=2,
    min_blocks_per_sm=4,
    final_wait="read",
    clc=True,
    static_units=0,
    debug="",
    iket=False,
    tail_units=1216,
)

_BULK_G2S = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
_BULK_S2G = "cp.async.bulk.global.shared::cta.bulk_group"


def _u32s(n, align=16):
    return K.alloc_local((n,), "uint32", align=align)


def _shfl_bfly(value_f32, lane_xor):
    out = K.local_scalar("uint32")
    K.ptx.shfl_sync.bfly.b32(
        out, K.reinterpret("uint32", value_f32), K.uint32(lane_xor), K.uint32(31), K.uint32(FULL)
    )
    return K.reinterpret("float32", out)


def _reduce(vals, count, lanes):
    x = lanes // 2
    while x >= 1:
        for i in range(count):
            K.ptx["add.f32"](vals[i], vals[i], _shfl_bfly(vals[i], x))
        x //= 2


def _unpack_bf16(words, nwords, dst, off):
    for p in range(nwords):
        K.ptx.mov.b32(dst[off + 2 * p], K.cuda.uint_as_float(K.shift_left(words[p], K.uint32(16))))
        K.ptx.mov.b32(
            dst[off + 2 * p + 1],
            K.cuda.uint_as_float(K.bitwise_and(words[p], K.uint32(0xFFFF0000))),
        )


def _mul2(a0, a1, b0, b1):
    out = K.local_scalar("uint64")
    K.ptx.mul.rn.f32x2(out, K.cuda.make_float2(a0, a1), K.cuda.make_float2(b0, b1))
    return out


def _fma2(a0, a1, b0, b1, c0, c1):
    out = K.local_scalar("uint64")
    K.ptx.fma.rn.f32x2(
        out, K.cuda.make_float2(a0, a1), K.cuda.make_float2(b0, b1), K.cuda.make_float2(c0, c1)
    )
    return out


def _lds_words(words, off, n, ptr_fn):
    i = 0
    while i < n:
        if n - i >= 4:
            K.ptx.ld.shared.v4.b32(words[off + i], words[off + i + 1], words[off + i + 2], words[off + i + 3], ptr_fn(2 * i))
            i += 4
        elif n - i >= 2:
            K.ptx.ld.shared.v2.b32(words[off + i], words[off + i + 1], ptr_fn(2 * i))
            i += 2
        else:
            K.ptx.ld.shared.b32(words[off + i], ptr_fn(2 * i))
            i += 1


_TRY_CANCEL = "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128"
HDR_ELEMS = 8
SENTINEL = 0xFFFFFFFF


def build_kernel(num_ctas, **overrides):
    cfg = {**CONFIG, **overrides}
    num_ctas = int(num_ctas)
    nc, ilp, lpr = int(cfg["consumer_warps"]), int(cfg["ilp"]), int(cfg["lanes_per_row"])
    unit_rows, nstage, look = int(cfg["unit_rows"]), int(cfg["nstage"]), int(cfg["lookahead"])
    mb = cfg["min_blocks_per_sm"]
    final_wait, use_clc, debug = cfg["final_wait"], bool(cfg["clc"]), cfg["debug"]
    static_units = int(cfg["static_units"])
    use_iket = bool(cfg["iket"])
    uniform_unit = bool(cfg.get("uniform_unit", False))
    out_lane_last = bool(cfg.get("out_lane_last", True))
    out_after_fence = bool(cfg.get("out_after_fence", True))
    out_lane = (lpr - 1) if out_lane_last else 0
    tail_units = int(cfg.get("tail_units", 0))

    def iket_range(name):
        if not use_iket:
            return None
        token = K.alloc_local([1], "uint32")
        K.assign(token[0], K.cuda.iket.range_start(name))
        return token

    def iket_end(token):
        if token is not None:
            K.cuda.iket.range_end(token[0])
    assert not (use_clc and static_units > 0)
    sched_on = use_clc or static_units > 0
    assert final_wait in ("read", "full") and debug in ("", "no_compute")
    assert 1 <= look <= nstage - 1
    assert D % unit_rows == 0
    units_per_tile = D // unit_rows
    unit_state_bytes = unit_rows * D * 2
    unit_stage_bytes = unit_state_bytes + AUX_BYTES
    assert lpr in (8, 16)
    epl = D // lpr
    wpl = epl // 2
    gpw = 32 // lpr
    groups = nc * gpw
    rpg = unit_rows // groups
    assert unit_rows % groups == 0 and rpg % ilp == 0 and ilp in (2, 4, 8, 16)
    iters = rpg // ilp
    aux0 = unit_state_bytes // 2
    hdr0 = unit_stage_bytes // 2
    stage_elems = hdr0 + HDR_ELEMS
    pending_allowed = nstage - look - 1

    @K.kernel(warps=nc, arch="sm_100a", grid=num_ctas, min_blocks_per_sm=mb)
    def kda_decode_clc(
        q: K.gptr[K.bf16],
        k: K.gptr[K.bf16],
        v: K.gptr[K.bf16],
        g: K.gptr[K.bf16],
        beta: K.gptr[K.bf16],
        state_in: K.gptr[K.bf16],
        state_out: K.gptr[K.bf16],
        output: K.gptr[K.bf16],
        scale: K.f32,
    ):
        cta = K.cta_id()
        tid = K.thread_id()
        smem = K.smem_pool()
        ring = smem.alloc((nstage * stage_elems,), K.bf16, align=128)
        handle = smem.pool.alloc((4,), K.u32, align=16)
        full = K.MBarrier(smem, nstage)
        full.init(1)
        clcbar = K.MBarrier(smem, 1)
        clcbar.init(1)
        K.cuda.cta_sync()

        lane_g = K.local_scalar("int32", init=tid % lpr)
        grp = K.local_scalar("int32", init=tid // lpr)
        k0 = K.local_scalar("int32", init=lane_g * epl)
        row0 = K.local_scalar("int32", init=grp * rpg)

        def issue_unit(unit, stage):
            tile = unit // units_per_tile
            part = unit % units_per_tile
            n = tile // HV
            hv = tile % HV
            h = hv // GROUP
            sbase = stage * stage_elems
            bar = full.ptr_to([stage])
            K.ptx.fence.proxy.async_.shared__cta()
            K.ptx.st.shared.b32(ring.ptr_to([sbase + hdr0]), K.reinterpret("uint32", unit))
            K.ptx.mbarrier.arrive.expect_tx.shared.b64(bar, K.uint32(unit_stage_bytes))
            K.ptx[_BULK_G2S](
                ring.ptr_to([sbase]),
                state_in.ptr_to([K.cast(tile, "int64") * (D * D) + part * (unit_rows * D)]),
                K.uint32(unit_state_bytes),
                bar,
            )
            qk_off = K.cast(n, "int64") * (H * D) + h * D
            g_off = K.cast(tile, "int64") * D
            K.ptx[_BULK_G2S](ring.ptr_to([sbase + aux0]), q.ptr_to([qk_off]), K.uint32(D * 2), bar)
            K.ptx[_BULK_G2S](ring.ptr_to([sbase + aux0 + D]), k.ptr_to([qk_off]), K.uint32(D * 2), bar)
            K.ptx[_BULK_G2S](ring.ptr_to([sbase + aux0 + 2 * D]), g.ptr_to([g_off]), K.uint32(D * 2), bar)
            K.ptx[_BULK_G2S](ring.ptr_to([sbase + aux0 + 3 * D]), v.ptr_to([g_off]), K.uint32(D * 2), bar)

        def issue_sentinel(stage):
            K.ptx.st.shared.b32(ring.ptr_to([stage * stage_elems + hdr0]), K.uint32(SENTINEL))
            full.arrive(stage)

        def try_cancel():
            if static_units > 0:
                return
            K.ptx.mbarrier.arrive.expect_tx.shared.b64(clcbar.ptr_to([0]), K.uint32(16))
            K.ptx[_TRY_CANCEL](K.address_of(handle[0]), clcbar.ptr_to([0]))

        clc_alive = K.local_scalar("int32", init=0)
        clc_phase = K.local_scalar("int32", init=0)
        fetch_j = K.local_scalar("int32", init=0)
        nxt = K.local_scalar("uint32", init=K.uint32(SENTINEL))

        def fetch_next():
            """Thread 0: obtain the next unit id into ``nxt`` (SENTINEL when there is none)."""
            if static_units > 0:
                K.assign(fetch_j, fetch_j + K.int32(1))
                K.assign(nxt, K.uint32(SENTINEL))
                with K.If(fetch_j < static_units), K.Then():
                    K.assign(nxt, K.reinterpret("uint32", cta + fetch_j * num_ctas))
                return
            t_f = iket_range("io-clc-wait")
            clcbar.wait(0, clc_phase)
            iket_end(t_f)
            K.assign(clc_phase, clc_phase ^ K.int32(1))
            K.query_cancel_first_ctaid_x(nxt, K.address_of(handle[0]))

        def refill_or_finish(stage):
            """Thread 0: issue the next unit into ``stage`` or publish the sentinel there."""
            with K.If(clc_alive == 1):
                with K.Then():
                    fetch_next()
                    with K.If(nxt != K.uint32(SENTINEL)):
                        with K.Then():
                            t_r = iket_range("io-stage-wait")
                            K.ptx.cp.async_.bulk.wait_group.read(pending_allowed)
                            iket_end(t_r)
                            t_i = iket_range("io-issue")
                            issue_unit(K.reinterpret("int32", nxt), stage)
                            try_cancel()
                            iket_end(t_i)
                        with K.Else():
                            K.assign(clc_alive, K.int32(0))
                            issue_sentinel(stage)
                with K.Else():
                    issue_sentinel(stage)

        with K.If(tid == 0), K.Then():
            issue_unit(cta, 0)
            if sched_on:
                try_cancel()
                K.assign(clc_alive, K.int32(1))
            for s in range(1, look):
                refill_or_finish(s)

        pipe = K.PipelineState(nstage, phase=0)
        with K.While(True):
            t_w = iket_range("cons-wait-full")
            full.wait(pipe.stage, pipe.phase)
            iket_end(t_w)
            hdr = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(hdr, ring.ptr_to([pipe.stage * stage_elems + hdr0]))
            with K.If(hdr == K.uint32(SENTINEL)), K.Then():
                K.Break()
            if uniform_unit:
                unit = K.local_scalar("int32", init=K.reinterpret("int32", K.uniform(hdr)))
            else:
                unit = K.local_scalar("int32", init=K.reinterpret("int32", hdr))
            rstage = K.local_scalar("int32", init=(pipe.stage + look) % nstage)
            if tail_units > 0:
                late = K.local_scalar("int32", init=K.if_then_else(unit >= K.int32(num_ctas - tail_units), 1, 0))
                with K.If((tid == 0) & (late == 0)), K.Then():
                    refill_or_finish(rstage)
            else:
                with K.If(tid == 0), K.Then():
                    refill_or_finish(rstage)

            sbase = pipe.stage * stage_elems
            tile = unit // units_per_tile
            part = unit % units_per_tile
            tile64 = K.cast(tile, "int64")
            t_c = iket_range("cons-unit")
            if debug != "no_compute":
                beta_bits = K.local_scalar("uint16")
                K.ptx.ld.global_.nc.b16(beta_bits, beta.ptr_to([tile64]))
                qw = _u32s(wpl)
                kw = _u32s(wpl)
                gw = _u32s(wpl)
                _lds_words(qw, 0, wpl, lambda e: ring.ptr_to([sbase + aux0 + k0 + e]))
                _lds_words(kw, 0, wpl, lambda e: ring.ptr_to([sbase + aux0 + D + k0 + e]))
                _lds_words(gw, 0, wpl, lambda e: ring.ptr_to([sbase + aux0 + 2 * D + k0 + e]))
                vwords = _u32s(rpg // 2)
                _lds_words(vwords, 0, rpg // 2, lambda e: ring.ptr_to([sbase + aux0 + 3 * D + part * unit_rows + row0 + e]))
                all_rows = []
                for r in range(rpg):
                    words = _u32s(wpl)
                    _lds_words(words, 0, wpl, lambda e, r=r: ring.ptr_to([sbase + (row0 + r) * D + k0 + e]))
                    all_rows.append(words)
                r_q = K.alloc_local((epl,), "float32")
                r_k = K.alloc_local((epl,), "float32")
                r_d = K.alloc_local((epl,), "float32")
                _unpack_bf16(qw, wpl, r_q, 0)
                _unpack_bf16(kw, wpl, r_k, 0)
                _unpack_bf16(gw, wpl, r_d, 0)
                sums = K.alloc_local((2,), "float32")
                K.ptx.mov.b32(sums[0], K.float32(0.0))
                K.ptx.mov.b32(sums[1], K.float32(0.0))
                for e in range(epl):
                    K.ptx["fma.rn.f32"](sums[0], r_q[e], r_q[e], sums[0])
                    K.ptx["fma.rn.f32"](sums[1], r_k[e], r_k[e], sums[1])
                _reduce(sums, 2, lpr)
                inv_q = K.local_scalar("float32")
                inv_k = K.local_scalar("float32")
                tmp = K.local_scalar("float32")
                K.ptx["add.f32"](tmp, sums[0], K.float32(1e-6))
                K.ptx["rsqrt.approx.ftz.f32"](inv_q, tmp)
                K.ptx["mul.f32"](inv_q, inv_q, scale)
                K.ptx["add.f32"](tmp, sums[1], K.float32(1e-6))
                K.ptx["rsqrt.approx.ftz.f32"](inv_k, tmp)
                for e in range(epl):
                    K.ptx["mul.f32"](r_q[e], r_q[e], inv_q)
                    K.ptx["mul.f32"](r_k[e], r_k[e], inv_k)
                    K.ptx["mul.f32"](r_d[e], r_d[e], K.float32(LOG2E))
                    K.ptx["ex2.approx.ftz.f32"](r_d[e], r_d[e])
                beta_f = K.local_scalar("float32")
                K.ptx.cvt.f32.bf16(beta_f, K.cast(beta_bits, "uint16"))
                out_base = tile64 * D + part * unit_rows
                deferred_outputs = []
                for it in range(iters):
                    rb = it * ilp
                    rows = all_rows[rb:rb + ilp]
                    sd = K.alloc_local((ilp * epl,), "float32")
                    for r in range(ilp):
                        _unpack_bf16(rows[r], wpl, sd, r * epl)
                    acc = K.alloc_local((ilp,), "float32")
                    acc_b = K.alloc_local((ilp,), "float32")
                    for r in range(ilp):
                        K.ptx.mov.b32(acc[r], K.float32(0.0))
                        K.ptx.mov.b32(acc_b[r], K.float32(0.0))
                    for pp in range(wpl):
                        for r in range(ilp):
                            i0 = r * epl + 2 * pp
                            prod = _mul2(sd[i0], sd[i0 + 1], r_d[2 * pp], r_d[2 * pp + 1])
                            K.ptx.mov.b32(sd[i0], K.cuda.float2_x(prod))
                            K.ptx.mov.b32(sd[i0 + 1], K.cuda.float2_y(prod))
                            pr = _fma2(sd[i0], sd[i0 + 1], r_k[2 * pp], r_k[2 * pp + 1], acc[r], acc_b[r])
                            K.ptx.mov.b32(acc[r], K.cuda.float2_x(pr))
                            K.ptx.mov.b32(acc_b[r], K.cuda.float2_y(pr))
                    for r in range(ilp):
                        K.ptx["add.f32"](acc[r], acc[r], acc_b[r])
                    _reduce(acc, ilp, lpr)
                    coef = K.alloc_local((ilp,), "float32")
                    for r in range(ilp):
                        rr = rb + r
                        vb = K.local_scalar("uint16")
                        if rr % 2 == 0:
                            K.ptx.mov.b16(vb, K.cast(K.bitwise_and(vwords[rr // 2], K.uint32(0xFFFF)), "uint16"))
                        else:
                            K.ptx.mov.b16(vb, K.cast(K.shift_right(vwords[rr // 2], K.uint32(16)), "uint16"))
                        diff = K.local_scalar("float32")
                        K.ptx.sub.rn.f32.bf16(diff, K.cast(vb, "uint16"), acc[r])
                        K.ptx["mul.f32"](coef[r], diff, beta_f)
                    for r in range(ilp):
                        K.ptx.mov.b32(acc[r], K.float32(0.0))
                        K.ptx.mov.b32(acc_b[r], K.float32(0.0))
                    for pp in range(wpl):
                        for r in range(ilp):
                            i0 = r * epl + 2 * pp
                            upd = _fma2(r_k[2 * pp], r_k[2 * pp + 1], coef[r], coef[r], sd[i0], sd[i0 + 1])
                            K.ptx.mov.b32(sd[i0], K.cuda.float2_x(upd))
                            K.ptx.mov.b32(sd[i0 + 1], K.cuda.float2_y(upd))
                            pr = _fma2(sd[i0], sd[i0 + 1], r_q[2 * pp], r_q[2 * pp + 1], acc[r], acc_b[r])
                            K.ptx.mov.b32(acc[r], K.cuda.float2_x(pr))
                            K.ptx.mov.b32(acc_b[r], K.cuda.float2_y(pr))
                    for r in range(ilp):
                        ow = _u32s(wpl)
                        for pp in range(wpl):
                            i0 = r * epl + 2 * pp
                            K.ptx.mov.b32(ow[pp], K.cuda.float22bfloat162_rn(sd[i0], sd[i0 + 1]))
                        for c in range(wpl // 4):
                            K.ptx.st.shared.v4.b32(ring.ptr_to([sbase + (row0 + rb + r) * D + k0 + 8 * c]), ow[4 * c], ow[4 * c + 1], ow[4 * c + 2], ow[4 * c + 3])
                    for r in range(ilp):
                        K.ptx["add.f32"](acc[r], acc[r], acc_b[r])
                    _reduce(acc, ilp, lpr)
                    out_vals = [K.local_scalar("float32", init=acc[r]) for r in range(ilp)]
                    def emit_output(out_vals=out_vals, rb=rb, out_base=out_base):
                        with K.If(lane_g == out_lane), K.Then():
                            o_idx = out_base + K.cast(row0 + rb, "int64")
                            packed = [K.cuda.float22bfloat162_rn(out_vals[2 * j], out_vals[2 * j + 1]) for j in range(ilp // 2)]
                            j = 0
                            while j < ilp // 2:
                                if ilp // 2 - j >= 4:
                                    K.ptx.st.global_.v4.b32(output.ptr_to([o_idx + 2 * j]), packed[j], packed[j + 1], packed[j + 2], packed[j + 3])
                                    j += 4
                                elif ilp // 2 - j >= 2:
                                    K.ptx.st.global_.v2.b32(output.ptr_to([o_idx + 2 * j]), packed[j], packed[j + 1])
                                    j += 2
                                else:
                                    K.ptx.st.global_.b32(output.ptr_to([o_idx + 2 * j]), packed[j])
                                    j += 1

                    if not out_after_fence:
                        emit_output()
                    else:
                        deferred_outputs.append(emit_output)
                K.ptx.fence.proxy.async_.shared__cta()
                for emit in deferred_outputs:
                    emit()
            iket_end(t_c)
            t_s = iket_range("cons-sync")
            K.cuda.cta_sync()
            iket_end(t_s)
            with K.If(tid == 0), K.Then():
                if tail_units > 0:
                    with K.If(late == 1), K.Then():
                        refill_or_finish(rstage)
                t_st = iket_range("io-store")
                K.ptx[_BULK_S2G](
                    state_out.ptr_to([tile64 * (D * D) + part * (unit_rows * D)]),
                    ring.ptr_to([sbase]),
                    K.uint32(unit_state_bytes),
                )
                K.ptx.cp.async_.bulk.commit_group()
                iket_end(t_st)
            pipe.advance()
        with K.If(tid == 0), K.Then():
            if final_wait == "read":
                K.ptx.cp.async_.bulk.wait_group.read(0)
            else:
                K.ptx.cp.async_.bulk.wait_group(0)

    return kda_decode_clc



# ---------------------------------------------------------------------------
# Registry interface.
# ---------------------------------------------------------------------------

KERNEL_META = {
    "name": "agent_evolved_kda_decode_b128_t1",
    "category": "agent_evolved",
    "runtime_cuda_archs": ["sm_100a"],
    "reference_requirements": (
        {"package": "flashinfer-python", "specifier": ">=0.6.18", "import": "flashinfer"},
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.7.0", "import": "cutlass"},
    ),
    "provenance": {
        "generator": "hmz",
        "run": "kda-decode-b128-t1",
        "selected_version": "frontier/clc-steal",
    },
}

CONFIGS = [
    {
        "label": "b128_t1_h16_hv32_d128",
        "num_seqs": 128,
        "num_tokens": 1,
        "num_qk_heads": 16,
        "num_v_heads": 32,
        "head_dim": 128,
        "seed": 42,
    }
]

STATE_STD = 0.01


def _config(**config: Any) -> dict[str, Any]:
    """Validate one config against the contract this kernel implements."""
    values = {key: value for key, value in config.items() if key != "label"}
    unknown = set(values) - set(CONFIGS[0]) - {"label"}
    if unknown:
        raise ValueError(f"unsupported config keys: {sorted(unknown)}")
    resolved = {**CONFIGS[0], **values}
    resolved.pop("label", None)
    if int(resolved["num_tokens"]) != 1:
        raise ValueError("this kernel implements the standard T=1 decode contract")
    if int(resolved["num_qk_heads"]) != H:
        raise ValueError(f"num_qk_heads must be {H}")
    if int(resolved["num_v_heads"]) != HV:
        raise ValueError(f"num_v_heads must be {HV}")
    if int(resolved["head_dim"]) != D:
        raise ValueError(f"head_dim must be {D}")
    return resolved


def _assert_supported_arch() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for agent-evolved KDA decode")
    capability = torch.cuda.get_device_capability()
    runtime_arch = f"sm_{capability[0]}{capability[1]}a"
    if runtime_arch not in KERNEL_META["runtime_cuda_archs"]:
        raise SkipTest(
            "agent-evolved KDA decode requires one of "
            f"{KERNEL_META['runtime_cuda_archs']}, got {runtime_arch}"
        )


def _num_units(resolved: dict[str, Any]) -> int:
    """One CTA per 64-row state unit: N * HV * (D // unit_rows)."""
    return int(resolved["num_seqs"]) * HV * (D // int(CONFIG["unit_rows"]))


def get_kernel(**config: Any):
    """Return the traced Kern PrimFunc for one compile key."""
    resolved = _config(**config)
    return build_kernel(_num_units(resolved)).func


# ---------------------------------------------------------------------------
# Inputs.
#
# The draw order reproduces the packaged KDA-decode benchmark row
# `kda-decode-d128-t1-b128-h16-hv32-standard_decode-precomputed`, which follows
# flashinfer PR #4279's benchmark: uniform q/k/v, a sigmoid'd normal beta, a
# log-sigmoid precomputed gate, and a normal state pool scaled by 0.01.
# ---------------------------------------------------------------------------


def prepare_data(**config: Any) -> dict[str, Any]:
    """Build one row's contract inputs plus the preallocated outputs."""
    resolved = _config(**config)
    device = torch.device("cuda")
    num_seqs = int(resolved["num_seqs"])
    num_qk_heads = int(resolved["num_qk_heads"])
    num_v_heads = int(resolved["num_v_heads"])
    head_dim = int(resolved["head_dim"])
    generator = torch.Generator(device=device).manual_seed(int(resolved["seed"]))
    token_shape = (num_seqs, 1)

    def rand(shape, dtype=torch.bfloat16):
        return torch.rand(shape, dtype=dtype, device=device, generator=generator)

    def randn(shape, dtype=torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device, generator=generator)

    q = rand((*token_shape, num_qk_heads, head_dim))
    k = rand((*token_shape, num_qk_heads, head_dim))
    v = rand((*token_shape, num_v_heads, head_dim))
    beta = torch.sigmoid(randn((*token_shape, num_v_heads)))
    g = torch.nn.functional.logsigmoid(
        randn((*token_shape, num_v_heads, head_dim), torch.float32)
    ).to(torch.bfloat16)
    initial_state = (
        randn((num_seqs, num_v_heads, head_dim, head_dim), torch.float32) * STATE_STD
    ).to(torch.bfloat16)
    return {
        "config": resolved,
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": None,
        "dt_bias": None,
        "scale": float(head_dim**-0.5),
        "initial_state": initial_state,
        "cu_seqlens": None,
        "ssm_state_indices": None,
        "num_spec_tokens": None,
        "num_accepted_tokens": None,
        "lower_bound": None,
        "output": torch.empty_like(v),
        # Only the written slots change; the rest must carry over.
        "final_state": initial_state.clone(),
    }


def _tirx_args(case: dict[str, Any]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Bind one case to the kernel's argument list (the candidate's ``setup``)."""
    q, k, v, g, beta = case["q"], case["k"], case["v"], case["g"], case["beta"]
    state_in, state_out, out = case["initial_state"], case["final_state"], case["output"]
    num_seqs = int(case["config"]["num_seqs"])
    assert case["cu_seqlens"] is None and case["A_log"] is None
    assert q.shape == (num_seqs, 1, H, D)
    assert v.shape == (num_seqs, 1, HV, D) and g.shape == (num_seqs, 1, HV, D)
    assert beta.shape == (num_seqs, 1, HV) and state_in.shape == (num_seqs, HV, D, D)
    for tensor in (q, k, v, g, beta, state_in, state_out, out):
        assert tensor.is_contiguous() and tensor.dtype == torch.bfloat16
    views = (
        q.view(-1), k.view(-1), v.view(-1), g.view(-1), beta.view(-1),
        state_in.view(-1), state_out.view(-1), out.view(-1),
    )
    args = (*views, float(case["scale"]))
    keep = (q, k, v, g, beta, state_in, state_out, out, *views)
    return args, keep


# ---------------------------------------------------------------------------
# Independent oracle and correctness.
#
# This is the packaged task's reference: the fp32 recurrence, one sequence at
# a time, from a k-first working state. It shares no code with the kernel and
# checks the complete final-state pool as well as the output.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _reference_output(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, v = case["q"], case["k"], case["v"]
    g, beta = case["g"], case["beta"]
    scale = float(case["scale"])
    initial_state = case["initial_state"]
    num_qk_heads, head_dim = q.shape[-2], q.shape[-1]
    num_v_heads = v.shape[-2]
    group = num_v_heads // num_qk_heads
    num_seqs, num_tokens = q.shape[0], 1

    def normalize(x):
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)

    qf = normalize(q.float()).reshape(num_seqs, num_tokens, num_qk_heads, head_dim) * scale
    kf = normalize(k.float()).reshape(num_seqs, num_tokens, num_qk_heads, head_dim)
    vf = v.float().reshape(num_seqs, num_tokens, num_v_heads, head_dim)
    bf = beta.float().reshape(num_seqs, num_tokens, num_v_heads)
    decay = g.float().reshape(num_seqs, num_tokens, num_v_heads, head_dim).exp()

    output = torch.empty_like(v, dtype=torch.float32)
    output_view = output.reshape(num_seqs, num_tokens, num_v_heads, head_dim)
    pool = initial_state.clone()
    for n in range(num_seqs):
        S = pool[n].float().transpose(-1, -2).clone()
        for t in range(num_tokens):
            q_t = qf[n, t].repeat_interleave(group, dim=0)
            k_t = kf[n, t].repeat_interleave(group, dim=0)
            v_t, b_t = vf[n, t], bf[n, t]
            S = S * decay[n, t][:, :, None]
            kS = torch.einsum("hk,hkv->hv", k_t, S)
            S = S + torch.einsum("hk,hv->hkv", b_t[:, None] * k_t, v_t - kS)
            output_view[n, t] = torch.einsum("hk,hkv->hv", q_t, S)
            pool[n] = S.transpose(-1, -2).to(pool.dtype)
    return output.to(v.dtype), pool


def check_correctness(outputs: dict[str, Any], **config: Any) -> None:
    """Gate the output and the complete final-state pool against the oracle."""
    case = outputs["case"]
    expected_output, expected_state = _reference_output(case)
    torch.testing.assert_close(
        outputs["output"].float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        outputs["final_state"].float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )


def run_test(**config: Any) -> None:
    """Compile, run once and gate against the oracle."""
    _assert_supported_arch()
    from tirx_kernels.runner import compile_kernel as _compile

    case = prepare_data(**config)
    executable = _compile(get_kernel(**config))
    args, keep = _tirx_args(case)
    executable(*args)
    torch.cuda.synchronize()
    check_correctness(
        {"case": case, "output": case["output"], "final_state": case["final_state"]},
        **config,
    )
    del keep


# ---------------------------------------------------------------------------
# FlashInfer reference arm.
#
# The packaged baseline for this row is `flashinfer.kda_decode.recurrent_kda`
# on its CuTe-DSL backend. The kernel updates its state pool in place while
# the contract returns a new pool, so the state copy and the caller-owned
# output buffer are prepare work (PR #4279's upstream arm does the same) and
# the timed span is the kernel alone.
# ---------------------------------------------------------------------------


def _recurrent_kda_args(case: dict[str, Any]):
    return (
        case["q"], case["k"], case["v"], case["g"], case["beta"],
        float(case["scale"]), case["initial_state"].clone(),
        torch.empty_like(case["v"]),
    )


def _recurrent_kda_launch(q, k, v, g, beta, scale, state, output):
    from flashinfer.kda_decode import recurrent_kda

    output, _ = recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        A_log=None,
        dt_bias=None,
        scale=scale,
        initial_state=state,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        lower_bound=None,
        cu_seqlens=None,
        ssm_state_indices=None,
        num_spec_tokens=None,
        num_accepted_tokens=None,
        output=output,
        backend="cute-dsl",
    )
    return output, state


# ---------------------------------------------------------------------------
# Benchmark entry points.
# ---------------------------------------------------------------------------


def prepare_bench(**config: Any):
    """Trace and compile before bench-suite assigns a GPU."""
    from tirx_kernels.runner import compile_kernel as _compile
    from tirx_kernels.runner import prepared_gpu_benchmark

    state = {"config": dict(config), "executable": _compile(get_kernel(**config))}
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
    executable = prepared["executable"]
    args, keep = _tirx_args(case)
    executable(*args)
    torch.cuda.synchronize()

    def _recurrent_kda_builder():
        # The state-pool copy and the output allocation are prepare work, as
        # the packaged baseline's own prepare step does them outside timing.
        reference_args = _recurrent_kda_args(case)
        _recurrent_kda_launch(*reference_args)
        return lambda: _recurrent_kda_launch(*reference_args)

    results = bench(
        {"tirx": lambda: executable(*args)},
        references={"flashinfer_recurrent_kda": _recurrent_kda_builder},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )
    del keep
    return results


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
