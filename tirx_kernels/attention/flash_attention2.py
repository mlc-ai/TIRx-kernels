# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashAttention-2-style tensor-core GQA prefill attention over packed variable-length
requests for SM120 (mma.sync m16n8k16, ldmatrix, cp.async).

Evolved with TIRx-harness on an RTX 5090 against FlashInfer's variable-length prefill (1.25x
in geometric mean over the Llama-3.2-1B, Qwen3-0.6B and Qwen3.5-0.8B heads).

One CTA owns (request, KV head, block of query positions). The query heads of a
GQA group are packed into the M dimension (packed row = position * G + head), so a
staged K/V tile serves every sibling head. QK^T and PV both accumulate in FP32.
Tiles run heaviest position block first: in LPT order inside chunks of 8 requests for up
to 32 requests, request by request beyond that.

Wide heads (head dim 512) split the output dimension: ``ds`` warps share each block
of rows, each computes the full QK^T and accumulates its slice of O. A sliding window
starts the key loop at the first tile any row can see and masks only boundary tiles.

Contract, for ``b`` requests packed along the token axis:

- ``q [n, h, d]``, ``k`` and ``v [nk, hk, d]``: FP16. ``qo_indptr`` and ``kv_indptr
  [b + 1]``: int32 offsets of each request's rows in ``q`` and in ``k``/``v``.
- ``out [n, h, d]``: FP16. ``lse [n, h]``: FP32 base-2 log-sum-exp of the scaled scores.
  A row that sees no key gets a zero output and ``lse = -50000``.
- ``causal`` (int32): nonzero lets query ``i`` of a request with ``qlen`` queries and ``klen``
  keys see keys ``j <= klen - qlen + i`` (bottom-right aligned); zero lets it see every key.
- ``sm_scale`` (float32) scales ``q @ k^T``.
- ``window > 0`` (fixed at trace time) adds ``q_pos [n]`` and ``kv_pos [b]`` (int32): the
  position of each query and of each request's first key. A query at position ``p`` sees
  keys at positions ``> p - window``. Positions are consecutive within a request, so the
  kernel reads only the first query position of each request.

``n``, ``nk`` and ``b`` are symbolic. The kernel expects ``b >= 1``, row offsets that are
multiples of 8 elements, and fewer than 2**31 rows and CTAs.
"""

import inspect
import math
from typing import Any

import torch
import tvm_ffi

import tirx_kernels.tirx_lite as txl
from tirx_kernels.runner import bench, resolve_backend_config
from tvm_tirx import ir, tirx

KERNEL_META = {
    "name": "flash_attention2",
    "category": "attention",
    "runtime_cuda_archs": ["sm_120a"],
}

LOG2E = math.log2(math.e)
FULL = 0xFFFFFFFF
EMPTY_LSE = -50000.0

# Measured configurations by (query heads, KV heads, head dim): one config, or
# [(minimum CTA count, config), ...]. A config sets warps, m-tiles per warp (mt), the key
# tile (bn), whether Q stays in registers (qreg) and the cp.async stages; optional keys are
# rc (requests per L2 chunk of the LPT order, default 8) and ds (warps sharing a block of
# rows, each owning d / ds of O, default 1).
TUNED = {
    (32, 8, 64): dict(warps=4, mt=2, bn=64, qreg=True, stages=2),
    (16, 8, 128): [
        (1024, dict(warps=4, mt=1, bn=64, qreg=True, stages=1)),
        (0, dict(warps=4, mt=1, bn=64, qreg=True, stages=2)),
    ],
    (8, 2, 256): [
        (1024, dict(warps=8, mt=1, bn=32, qreg=False, stages=1)),
        (0, dict(warps=4, mt=1, bn=32, qreg=False, stages=2)),
    ],
}

_PARAMS = ("q", "qp", "k", "v", "kp", "out", "lse", "causal", "sm_scale")
_WINDOW_PARAMS = ("qpos", "kpos")


def variants(heads):
    """The configurations for ``heads = (query heads, KV heads, head dim)``, or None.

    Returns ``[(minimum CTA count, config), ...]`` by descending threshold; the last entry
    is unconditional. Geometries outside ``TUNED`` use a configuration measured on them:
    head dim 128 with GQA groups of 4 or 8 (one cp.async stage), head dim 256 with groups
    of 4 or 8 (the (8, 2, 256) list), and head dim 512 with groups of 4 or 8 (two warps per
    block of rows, each accumulating half of O), each with at most 8 KV heads.
    """
    heads = tuple(heads)
    found = TUNED.get(heads)
    if found is None:
        h, hk, d = heads
        group = h // hk if 0 < hk <= 8 and h % hk == 0 else 0
        if d == 128 and group in (4, 8):
            found = dict(warps=4, mt=1, bn=64, qreg=True, stages=1)
        elif d == 256 and group in (4, 8):
            found = TUNED[(8, 2, 256)]
        elif d == 512 and group in (4, 8):
            # Q (32 rows), K and V (32 keys) take 96 KB of shared memory.
            found = dict(warps=4, ds=2, mt=1, bn=32, qreg=False, stages=1)
        else:
            return None
    if isinstance(found, dict):
        found = [(0, found)]
    return sorted(found, key=lambda x: -x[0])


def rows_per_tile(heads, cfg):
    """Query positions per CTA tile."""
    h, hk, _ = heads
    return 16 * cfg["mt"] * (cfg["warps"] // cfg.get("ds", 1)) // (h // hk)


def param_types(heads, window=0):
    """Parameter types with symbolic token counts ``n``, ``nk`` and request count ``batch``."""
    h, hk, d = heads
    n, nk, batch = (tirx.Var(name, "int64") for name in ("n", "nk", "batch"))
    types = [
        txl.Tensor((n, h, d), "float16"),
        txl.Tensor((batch + 1,), "int32"),
        txl.Tensor((nk, hk, d), "float16"),
        txl.Tensor((nk, hk, d), "float16"),
        txl.Tensor((batch + 1,), "int32"),
        txl.Tensor((n, h, d), "float16"),
        txl.Tensor((n, h), "float32"),
        ir.PrimType("int32"),
        ir.PrimType("float32"),
    ]
    if window:
        types += [txl.Tensor((n,), "int32"), txl.Tensor((batch,), "int32")]
    return types


def _ld32(tensor, index):
    value = txl.local_scalar("int32")
    txl.ptx.ld.global_.b32(value, tensor.ptr_to([index]))
    return value


def _shfl_bfly(value, delta, ty="float32"):
    out = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.bfly.b32(
        out, txl.reinterpret(value, ty="uint32"), delta, 31, txl.uint32(FULL)
    )
    return txl.reinterpret(out, ty=ty)


def _shfl_up_i32(value, delta):
    out = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.up.b32(out, txl.reinterpret(value, ty="uint32"), delta, 0, txl.uint32(FULL))
    return txl.reinterpret(out, ty="int32")


def _shfl_idx_i32(value, lane):
    out = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.idx.b32(
        out, txl.reinterpret(value, ty="uint32"), txl.cast(lane, "uint32"), 31, txl.uint32(FULL)
    )
    return txl.reinterpret(out, ty="int32")


def _redux(op, value):
    out = txl.local_scalar("int32")
    getattr(txl.ptx.redux_sync, op).s32(out, value, txl.uint32(FULL))
    return out


def _ballot(cond):
    out = txl.local_scalar("uint32")
    txl.ptx.vote_sync.ballot.b32(out, txl.ptx.pred(cond), txl.uint32(FULL))
    return out


def _exp2(x):
    r = txl.local_scalar("float32")
    txl.ptx.ex2.approx.ftz.f32(r, x)
    return r


def _pack(lo, hi):
    r = txl.local_scalar("uint32")
    txl.ptx.cvt.rn.f16x2.f32(r, hi, lo)
    return r


def _mma(c, a, b0, b1):
    txl.ptx["mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"](
        c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b0, b1, c[0], c[1], c[2], c[3]
    )


def _ldsm(ptr, trans=False):
    w = [txl.local_scalar("uint32") for _ in range(4)]
    if trans:
        txl.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(*w, ptr)
    else:
        txl.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(*w, ptr)
    return w


def make_kernel(heads, cfg, *, window=0, types=None, backend_config=None):
    """Trace one configuration for ``heads = (query heads, KV heads, head dim)``.

    ``types`` overrides the parameter types of ``param_types`` (same order and arity), so a
    caller can trace the kernel against its own tensor layouts. ``backend_config`` selects the
    CUDA architecture; without one, it is resolved from the visible GPU.
    """
    h, hk, d = heads
    if types is None:
        types = param_types(heads, window)
    G = h // hk
    W, MT, BN, QREG = cfg["warps"], cfg["mt"], cfg["bn"], cfg["qreg"]
    S = cfg.get("stages", 2)
    RC = cfg.get("rc", 8)  # requests per L2 chunk for LPT ordering
    DS = cfg.get("ds", 1)  # warps sharing a block of rows, each owning d / DS of O
    RG = W // DS  # blocks of 16 * MT rows
    BM = 16 * MT * RG
    P = BM // G  # query positions per tile
    T = 32 * W
    CH = d // 8  # 16-byte chunks per row
    KC = d // 16  # k-chunks of QK^T
    NT = BN // 8  # n-tiles of S
    DT = d // 8  # n-tiles of O
    DTW = DT // DS  # n-tiles of O per warp

    def prefill(q, qp, k, v, kp, out, lse, causal, sm_scale, qpos=None, kpos=None):
        batch = qp.shape[0] - 1
        tiles = (q.shape[0] + P - 1) // P + batch
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=tiles * hk, block=T),
            backend_config=resolve_backend_config(backend_config),
        )
        cta = txl.cta_id()
        tile = (cta // hk).astype("int32")
        kh = (cta % hk).astype("int32")
        lane, warp, tid = txl.lane_id(), txl.warp_id(), txl.thread_id()

        pool = txl.smem_pool()
        sq = pool.alloc((BM, d), "float16", align=128)
        sk = pool.alloc((S * BN, d), "float16", align=128)
        sv = pool.alloc((S * BN, d), "float16", align=128)

        # ---- locate (request, position block) of this tile: request-major,
        # heaviest (last) position block first inside each request.
        fb = txl.local_scalar("int32", -1)
        fj = txl.local_scalar("int32", 0)
        cum = txl.local_scalar("int32", 0)
        with txl.if_(batch <= 32), txl.then_():
            # global LPT order: block index j descending, then request ascending
            nbl = txl.local_scalar("int32", 0)
            with txl.if_(lane < batch), txl.then_():
                qlb = _ld32(qp, lane + 1) - _ld32(qp, lane)
                txl.assign(nbl, (qlb + P - 1) // P)
            # L2-friendly: requests are processed in chunks of RC; LPT inside a chunk
            chunk = lane // RC
            cbase = txl.local_scalar("int32", 0)
            fc = txl.local_scalar("int32", -1)
            for cc in range(32 // RC):
                tc = _redux("add", tirx.if_then_else(chunk == cc, nbl, 0))
                with txl.if_((fc < 0) & (tile < cbase + tc)), txl.then_():
                    txl.assign(fc, cc)
                with txl.if_(fc < 0), txl.then_():
                    txl.assign(cbase, cbase + tc)
            with txl.if_(fc >= 0), txl.then_():
                nbc = txl.local_scalar("int32", tirx.if_then_else(chunk == fc, nbl, 0))
                rt = tile - cbase
                mx = _redux("max", nbc)
                jj = txl.local_scalar("int32", 0)
                clzv = txl.local_scalar("uint32")
                txl.ptx.clz.b32(clzv, txl.cast(mx, "uint32"))
                nbits = (32 - txl.cast(clzv, "int32")).astype("int32")
                with txl.serial(nbits) as bi:
                    x = jj + (txl.int32(1) << (nbits - 1 - bi.astype("int32")))
                    cx = _redux("add", tirx.max(nbc - x, 0))
                    with txl.if_(cx > rt), txl.then_():
                        txl.assign(jj, x)
                above = _redux("add", tirx.max(nbc - jj - 1, 0))
                rel0 = rt - above
                flag = nbc > jj
                mask = _ballot(flag)
                lt = mask & ((txl.uint32(1) << txl.cast(lane, "uint32")) - txl.uint32(1))
                pc = txl.local_scalar("uint32")
                txl.ptx.popc.b32(pc, lt)
                sel = _ballot(flag & (txl.cast(pc, "int32") == rel0))
                hb = txl.local_scalar("uint32")
                txl.ptx.bfind.u32(hb, sel)
                txl.assign(fb, txl.cast(hb, "int32"))
                txl.assign(fj, jj)

        # batch > 32: request-major, heaviest block first inside each request
        with txl.serial(tirx.if_then_else(batch > 32, (batch + 31) // 32, 0)) as c:
            with txl.if_(fb < 0), txl.then_():
                idx = (c * 32 + lane).astype("int32")
                nb = txl.local_scalar("int32", 0)
                with txl.if_(idx < batch), txl.then_():
                    ql = _ld32(qp, idx + 1) - _ld32(qp, idx)
                    txl.assign(nb, (ql + P - 1) // P)
                incl = txl.local_scalar("int32", nb)
                for delta in (1, 2, 4, 8, 16):
                    up = _shfl_up_i32(incl, delta)
                    txl.assign(incl, tirx.if_then_else(lane >= delta, incl + up, incl))
                total = _shfl_idx_i32(incl, 31)
                rel = tile - cum
                with txl.if_(rel < total), txl.then_():
                    cnt = txl.local_scalar("int32", tirx.if_then_else(incl <= rel, 1, 0))
                    for delta in (16, 8, 4, 2, 1):
                        txl.assign(cnt, cnt + _shfl_bfly(cnt, delta, "int32"))
                    inc_f = _shfl_idx_i32(incl, cnt)
                    nb_f = _shfl_idx_i32(nb, cnt)
                    txl.assign(fb, (c * 32).astype("int32") + cnt)
                    txl.assign(fj, nb_f - 1 - (rel - (inc_f - nb_f)))
                txl.assign(cum, cum + total)

        with txl.if_(fb >= 0), txl.then_():
            qs = txl.local_scalar("int64", _ld32(qp, fb).astype("int64"))
            qlen = txl.local_scalar("int32", _ld32(qp, fb + 1) - _ld32(qp, fb))
            ks = txl.local_scalar("int64", _ld32(kp, fb).astype("int64"))
            klen = txl.local_scalar("int32", _ld32(kp, fb + 1) - _ld32(kp, fb))
            p0 = txl.local_scalar("int32", fj * P)
            pend = txl.local_scalar("int32", tirx.min(p0 + P, qlen))
            shift = klen - qlen
            # keys [0, hi) are needed by some row; keys [0, lo) are visible to all rows
            hi = txl.local_scalar(
                "int32",
                tirx.if_then_else(causal > 0, tirx.max(0, tirx.min(klen, shift + pend)), klen),
            )
            lo = txl.local_scalar(
                "int32",
                tirx.if_then_else(causal > 0, tirx.max(0, tirx.min(klen, shift + p0 + 1)), klen),
            )
            ntile = txl.local_scalar("int32", (hi + BN - 1) // BN)
            nfull = txl.local_scalar("int32", lo // BN)
            # Tiles [t0, tb) and [tc, ntile) are masked, [tb, tc) visible to every row.
            t0, tb, tc = 0, 0, nfull
            if window:
                # A query at position p sees keys at positions > p - window; positions are
                # consecutive within a request, so row pos's first key is woff + pos.
                woff = txl.local_scalar("int32", _ld32(qpos, qs) - _ld32(kpos, fb) - window + 1)
                t0 = txl.local_scalar("int32", tirx.min(tirx.max(woff + p0, 0) // BN, ntile))
                tb = txl.local_scalar(
                    "int32",
                    tirx.min(tirx.max((tirx.max(woff + pend - 1, 0) + BN - 1) // BN, t0), ntile),
                )
                tc = txl.local_scalar("int32", tirx.min(tirx.max(nfull, tb), ntile))

            # ---- stage Q (zero-filled beyond the request)
            for off in range(0, BM * CH, T):
                vec = tid + off
                if (BM * CH) % T:
                    cond = vec < BM * CH
                else:
                    cond = None
                r = vec // CH
                ch = vec % CH
                pos = p0 + r // G
                valid = pos < pend
                row = qs + tirx.if_then_else(valid, pos, 0).astype("int64")
                src = q.ptr_to([row, kh * G + r % G, ch * 8])
                dst = sq.ptr_to([r, (ch ^ (r % 8)) * 8])

                def issue():
                    txl.ptx.cp.async_.cg.shared.global_(
                        dst, src, 16, txl.cast(tirx.if_then_else(valid, 16, 0), "uint32")
                    )

                if cond is None:
                    issue()
                else:
                    with txl.if_(cond), txl.then_():
                        issue()
            txl.ptx.cp.async_.commit_group()

            def load_kv(buf, src_t, it, slot):
                base = it * BN
                for off in range(0, BN * CH, T):
                    vec = tid + off
                    r = vec // CH
                    ch = vec % CH
                    key = base + r
                    valid = key < klen
                    row = ks + tirx.if_then_else(valid, key, 0).astype("int64")
                    src = src_t.ptr_to([row, kh, ch * 8])
                    dst = buf.ptr_to([slot * BN + r, (ch ^ (r % 8)) * 8])
                    txl.ptx.cp.async_.cg.shared.global_(
                        dst, src, 16, txl.cast(tirx.if_then_else(valid, 16, 0), "uint32")
                    )
                txl.ptx.cp.async_.commit_group()

            with txl.if_(ntile > t0), txl.then_():
                load_kv(sk, k, t0, 0)
                load_kv(sv, v, t0, 0)

            scale = sm_scale * LOG2E
            if DS == 1:
                rowbase = warp * (16 * MT)
                dsl = 0
            else:
                rowbase = (warp % RG) * (16 * MT)
                dsl = warp // RG
            g, t4 = lane // 4, lane % 4

            # per-thread state: for each m-tile two rows (g, g+8)
            m_i = [[txl.local_scalar("float32", -math.inf) for _ in range(2)] for _ in range(MT)]
            l_i = [[txl.local_scalar("float32", 0.0) for _ in range(2)] for _ in range(MT)]
            o = [
                [[txl.local_scalar("float32", 0.0) for _ in range(4)] for _ in range(DTW)]
                for _ in range(MT)
            ]
            # row limits (exclusive key bound) per thread row
            lim = []
            for mt in range(MT):
                pair = []
                for hr in range(2):
                    r = rowbase + mt * 16 + g + hr * 8
                    pos = p0 + r // G
                    pair.append(
                        txl.local_scalar(
                            "int32",
                            tirx.if_then_else(causal > 0, tirx.min(klen, shift + pos + 1), klen),
                        )
                    )
                lim.append(pair)
            # first visible key per thread row (sliding window)
            low = []
            if window:
                for mt in range(MT):
                    low.append(
                        [
                            txl.local_scalar(
                                "int32", woff + p0 + (rowbase + mt * 16 + g + hr * 8) // G
                            )
                            for hr in range(2)
                        ]
                    )

            txl.ptx.cp.async_.wait_group(0)
            txl.ptx.bar.sync(0)

            def q_frag(mt, kc):
                r = rowbase + mt * 16 + lane % 16
                ch = kc * 2 + lane // 16
                return _ldsm(sq.ptr_to([r, (ch ^ (r % 8)) * 8]))

            qf = None
            if QREG:
                qf = [[q_frag(mt, kc) for kc in range(KC)] for mt in range(MT)]

            def body(it, masked):
                if S == 2:
                    slot = (it - t0) % 2 if window else it % 2
                    # K[it], V[it] landed (Q already waited)
                    txl.ptx.cp.async_.wait_group(1)
                    txl.ptx.bar.sync(0)
                    with txl.if_(it + 1 < ntile), txl.then_():
                        load_kv(sk, k, it + 1, 1 - slot)
                    with txl.if_(it + 1 >= ntile), txl.then_():
                        txl.ptx.cp.async_.commit_group()
                else:
                    slot = 0
                    # single slot: K[it] landed; every warp finished PV[it-1] -> V slot free
                    txl.ptx.cp.async_.wait_group(0)
                    txl.ptx.bar.sync(0)
                    with txl.if_(it > t0), txl.then_():
                        load_kv(sv, v, it, 0)
                s = [
                    [[txl.local_scalar("float32", 0.0) for _ in range(4)] for _ in range(NT)]
                    for _ in range(MT)
                ]
                for kc in range(KC):
                    if QREG:
                        af = [qf[mt][kc] for mt in range(MT)]
                    else:
                        af = [q_frag(mt, kc) for mt in range(MT)]
                    for np_ in range(NT // 2):
                        key = slot * BN + np_ * 16 + (lane // 16) * 8 + lane % 8
                        ch = kc * 2 + (lane // 8) % 2
                        bw = _ldsm(sk.ptr_to([key, (ch ^ (key % 8)) * 8]))
                        for mt in range(MT):
                            _mma(s[mt][2 * np_], af[mt], bw[0], bw[1])
                            _mma(s[mt][2 * np_ + 1], af[mt], bw[2], bw[3])
                pf = []
                for mt in range(MT):
                    alphas = []
                    for hr in range(2):
                        if masked:
                            for nt in range(NT):
                                for e in range(2):
                                    col = it * BN + nt * 8 + t4 * 2 + e
                                    x = s[mt][nt][hr * 2 + e]
                                    seen = col < lim[mt][hr]
                                    if window:
                                        seen = seen & (col >= low[mt][hr])
                                    txl.assign(x, tirx.if_then_else(seen, x, -math.inf))
                        mx = txl.local_scalar("float32", s[mt][0][hr * 2])
                        for nt in range(NT):
                            for e in range(2):
                                if nt or e:
                                    txl.assign(mx, tirx.max(mx, s[mt][nt][hr * 2 + e]))
                        for delta in (1, 2):
                            txl.assign(mx, tirx.max(mx, _shfl_bfly(mx, delta)))
                        mnew = txl.local_scalar("float32", tirx.max(m_i[mt][hr], mx * scale))
                        muse = txl.local_scalar(
                            "float32", tirx.if_then_else(mnew == -math.inf, 0.0, mnew)
                        )
                        alpha = _exp2(m_i[mt][hr] - muse)
                        txl.assign(m_i[mt][hr], mnew)
                        rs = txl.local_scalar("float32", 0.0)
                        for nt in range(NT):
                            for e in range(2):
                                x = s[mt][nt][hr * 2 + e]
                                txl.assign(x, _exp2(x * scale - muse))
                                txl.assign(rs, rs + x)
                        txl.assign(l_i[mt][hr], l_i[mt][hr] * alpha + rs)
                        alphas.append(alpha)
                    for dt in range(DTW):
                        for e in range(4):
                            txl.assign(o[mt][dt][e], o[mt][dt][e] * alphas[e // 2])
                    pf.append(
                        [
                            [
                                _pack(s[mt][2 * kc][0], s[mt][2 * kc][1]),
                                _pack(s[mt][2 * kc][2], s[mt][2 * kc][3]),
                                _pack(s[mt][2 * kc + 1][0], s[mt][2 * kc + 1][1]),
                                _pack(s[mt][2 * kc + 1][2], s[mt][2 * kc + 1][3]),
                            ]
                            for kc in range(BN // 16)
                        ]
                    )
                if S == 2:
                    txl.ptx.cp.async_.wait_group(1)
                    txl.ptx.bar.sync(0)
                    with txl.if_(it + 1 < ntile), txl.then_():
                        load_kv(sv, v, it + 1, 1 - slot)
                    with txl.if_(it + 1 >= ntile), txl.then_():
                        txl.ptx.cp.async_.commit_group()
                else:
                    # V[it] landed; every warp finished QK[it] -> K slot free
                    txl.ptx.cp.async_.wait_group(0)
                    txl.ptx.bar.sync(0)
                    with txl.if_(it + 1 < ntile), txl.then_():
                        load_kv(sk, k, it + 1, 0)
                for kc in range(BN // 16):
                    for dp in range(DTW // 2):
                        key = slot * BN + kc * 16 + ((lane // 8) % 2) * 8 + lane % 8
                        ch = (dsl * (DTW // 2) + dp) * 2 + lane // 16
                        bw = _ldsm(sv.ptr_to([key, (ch ^ (key % 8)) * 8]), trans=True)
                        for mt in range(MT):
                            _mma(o[mt][2 * dp], pf[mt][kc], bw[0], bw[1])
                            _mma(o[mt][2 * dp + 1], pf[mt][kc], bw[2], bw[3])

            if window:
                with txl.serial(tb - t0) as it0:
                    body(it0 + t0, True)
                with txl.serial(tc - tb) as it:
                    body(it + tb, False)
                with txl.serial(ntile - tc) as it2:
                    body(it2 + tc, True)
            else:
                with txl.serial(nfull) as it:
                    body(it, False)
                with txl.serial(ntile - nfull) as it2:
                    body(it2 + nfull, True)

            # ---- epilogue: normalize, stage through shared memory, store 16B vectors
            txl.ptx.cp.async_.wait_group(0)
            txl.ptx.bar.sync(0)
            for mt in range(MT):
                for hr in range(2):
                    den = l_i[mt][hr]
                    for delta in (1, 2):
                        txl.assign(den, den + _shfl_bfly(den, delta))
                    inv = txl.local_scalar("float32", tirx.if_then_else(den > 0.0, 1.0 / den, 0.0))
                    r = rowbase + mt * 16 + g + hr * 8
                    for dt in range(DTW):
                        ch = dsl * DTW + dt
                        word = _pack(o[mt][dt][hr * 2] * inv, o[mt][dt][hr * 2 + 1] * inv)
                        txl.ptx.st.shared.b32(sq.ptr_to([r, (ch ^ (r % 8)) * 8 + t4 * 2]), word)
                    pos = p0 + r // G
                    first = (t4 == 0) & (pos < pend)
                    if DS > 1:
                        first = first & (dsl == 0)
                    with txl.if_(first), txl.then_():
                        val = tirx.if_then_else(den > 0.0, m_i[mt][hr] + tirx.log2(den), EMPTY_LSE)
                        txl.ptx.st.global_.f32(
                            lse.ptr_to([qs + pos.astype("int64"), kh * G + r % G]), val
                        )
            txl.ptx.bar.sync(0)
            for off in range(0, BM * CH, T):
                vec = tid + off
                r = vec // CH
                ch = vec % CH
                pos = p0 + r // G
                with txl.if_(pos < pend), txl.then_():
                    w = [txl.local_scalar("uint32") for _ in range(4)]
                    txl.ptx.ld.shared.v4.b32(*w, sq.ptr_to([r, (ch ^ (r % 8)) * 8]))
                    txl.ptx.st.global_.v4.b32(
                        out.ptr_to([qs + pos.astype("int64"), kh * G + r % G, ch * 8]), *w
                    )

    # The tracer reads the kernel parameters from the signature; a window adds positions.
    names = _PARAMS + (_WINDOW_PARAMS if window else ())
    prefill.__signature__ = inspect.Signature(
        [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=ty)
            for name, ty in zip(names, types, strict=True)
        ]
    )
    return txl.kernel()(prefill).func


def _body_over(fn, params):
    """``fn``'s body with its parameters replaced by ``params``."""
    subst = dict(zip(fn.params, params, strict=True))
    return tvm_ffi.structural_mutate(fn.body, [(ir.Var, lambda var, _: subst.get(var, var))])


def get_kernel(q_heads, kv_heads, head_dim, window=0, *, backend_config=None, **kwargs):
    """The tuned kernel for one head geometry, dispatching between its configurations.

    Parameters follow the module contract: ``(q, qo_indptr, k, v, kv_indptr, out, lse,
    causal, sm_scale[, q_pos, kv_pos])``. A geometry with several configurations picks one
    per launch by its CTA count.
    """
    heads = (q_heads, kv_heads, head_dim)
    chosen = variants(heads)
    if chosen is None:
        raise ValueError(f"no flash_attention2 configuration for heads {heads}")
    types = param_types(heads, window)
    base = make_kernel(
        heads, chosen[-1][1], window=window, types=types, backend_config=backend_config
    )
    if len(chosen) == 1:
        return base
    params = list(base.params)
    n, batch = params[0].ty.shape[0], params[1].ty.shape[0] - 1
    body = base.body
    for thresh, cfg in reversed(chosen[:-1]):
        tile = rows_per_tile(heads, cfg)
        ctas = ((n + tile - 1) // tile + batch) * kv_heads
        variant = make_kernel(heads, cfg, window=window, types=types, backend_config=backend_config)
        body = ir.If(ctas >= thresh, _body_over(variant, params), body)
    return tirx.Function(params, body, attrs=base.attrs)


# ---------------------------------------------------------------------------
# Correctness and benchmark configurations
# ---------------------------------------------------------------------------

# Prefill head geometries (query heads, KV heads, head dim) of Llama-3.2-1B, Qwen3-0.6B,
# Qwen3-4B, the full-attention layers of Qwen3.5-0.8B, and Gemma-4-E2B's sliding (window 512)
# and global layers.
_MODELS = (
    ("llama3", (32, 8, 64), 0),
    ("qwen3", (16, 8, 128), 0),
    ("qwen3_4b", (32, 8, 128), 0),
    ("qwen35", (8, 2, 256), 0),
    ("gemma4_sliding", (8, 1, 256), 512),
    ("gemma4_global", (8, 1, 512), 0),
)
# Request lengths around the 64-key tile and the 512-position window.
_MIXED = (1, 63, 64, 65, 511, 512, 513, 1500)
# More than 32 requests take the request-major tile order instead of the warp-level LPT one.
_MANY = (1, 63, 64, 65, 130) * 8
# Enough tokens to select the configuration for 1024 or more CTAs where a geometry has one.
_LONG = (1500,) * 11


def _config(model, heads, window, case, qo_lens, kv_lens=None, causal=True):
    h, hk, d = heads
    return {
        "label": f"{model}_h{h}_kv{hk}_d{d}{f'_w{window}' if window else ''}_{case}",
        "q_heads": h,
        "kv_heads": hk,
        "head_dim": d,
        "window": window,
        "qo_lens": qo_lens,
        "kv_lens": kv_lens,
        "causal": causal,
    }


CONFIGS = []
for _model, _heads, _window in _MODELS:
    CONFIGS += [
        _config(_model, _heads, _window, "mixed", _MIXED),
        _config(_model, _heads, _window, "many", _MANY),
        # queries over a longer prefix of keys (bottom-right causal alignment)
        _config(_model, _heads, _window, "prefix", (1, 64, 513), (100, 64, 1500)),
    ]
    if len(variants(_heads)) > 1:
        CONFIGS.append(_config(_model, _heads, _window, "long", _LONG))
    if _window:
        # the same geometry without the window
        CONFIGS.append(_config(_model, _heads, 0, "mixed", _MIXED))
    else:
        CONFIGS.append(_config(_model, _heads, 0, "mixed_noncausal", _MIXED, causal=False))
# fewer keys than queries: the first rows of a causal request see no key
CONFIGS.append(_config("qwen3", (16, 8, 128), 0, "short_kv", (64, 513), (1, 500)))

BENCH_CONFIGS = []
for _model, _heads, _window in _MODELS:
    BENCH_CONFIGS += [
        _config(_model, _heads, _window, "1x1024", (1024,)),
        _config(_model, _heads, _window, "1x4096", (4096,)),
        _config(_model, _heads, _window, "8x512", (512,) * 8),
        _config(_model, _heads, _window, "mixed", _MIXED),
    ]


def prepare_data(
    q_heads, kv_heads, head_dim, qo_lens, kv_lens=None, causal=True, window=0, **kwargs
):
    """Packed requests with random FP16 Q/K/V and NaN-filled outputs.

    Request ``b`` holds keys at positions ``kv_pos[b] + j`` and its queries at the last
    ``qlen`` of those positions.
    """
    torch.manual_seed(0)
    kv_lens = qo_lens if kv_lens is None else kv_lens
    n, nk = sum(qo_lens), sum(kv_lens)

    def indptr(lens):
        return torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)

    kv_pos = torch.arange(len(qo_lens), dtype=torch.int32) * 7
    q_pos = torch.cat(
        [
            torch.arange(kl - ql, kl, dtype=torch.int32) + p
            for ql, kl, p in zip(qo_lens, kv_lens, kv_pos.tolist(), strict=True)
        ]
    )
    data = {
        "q": torch.randn(n, q_heads, head_dim, dtype=torch.float16),
        "qo_indptr": indptr(qo_lens),
        "k": torch.randn(nk, kv_heads, head_dim, dtype=torch.float16),
        "v": torch.randn(nk, kv_heads, head_dim, dtype=torch.float16),
        "kv_indptr": indptr(kv_lens),
        "out": torch.full((n, q_heads, head_dim), math.nan, dtype=torch.float16),
        "lse": torch.full((n, q_heads), math.nan, dtype=torch.float32),
        "q_pos": q_pos,
        "kv_pos": kv_pos,
    }
    data = {name: value.cuda() for name, value in data.items()}
    data["causal"] = int(causal)
    data["sm_scale"] = head_dim**-0.5
    return data


def _kernel_args(data, window):
    args = [data[name] for name in ("q", "qo_indptr", "k", "v", "kv_indptr", "out", "lse")]
    args += [data["causal"], data["sm_scale"]]
    if window:
        args += [data["q_pos"], data["kv_pos"]]
    return args


def _visible(data, b, window):
    """[qlen, klen] mask of the keys each query of request ``b`` sees."""
    qo, kv = data["qo_indptr"].tolist(), data["kv_indptr"].tolist()
    qlen, klen = qo[b + 1] - qo[b], kv[b + 1] - kv[b]
    device = data["q"].device
    i = torch.arange(qlen, device=device)[:, None]
    j = torch.arange(klen, device=device)[None, :]
    visible = torch.ones(qlen, klen, dtype=torch.bool, device=device)
    if data["causal"]:
        visible &= j <= klen - qlen + i
    if window:
        q_pos = data["q_pos"][qo[b] : qo[b + 1]].long()[:, None]
        visible &= data["kv_pos"][b].long() + j > q_pos - window
    return visible


def reference(data, window=0):
    """FP32 SDPA per request: (out [n, h, d] FP32, base-2 lse [n, h])."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q, k, v = data["q"], data["k"], data["v"]
    group = q.shape[1] // k.shape[1]
    qo, kv = data["qo_indptr"].tolist(), data["kv_indptr"].tolist()
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    for b in range(len(qo) - 1):
        qb = q[qo[b] : qo[b + 1]].float().transpose(0, 1)
        kb, vb = (
            t[kv[b] : kv[b + 1]].float().repeat_interleave(group, 1).transpose(0, 1) for t in (k, v)
        )
        visible = _visible(data, b, window)
        with sdpa_kernel(SDPBackend.MATH):
            ob = torch.nn.functional.scaled_dot_product_attention(
                qb, kb, vb, attn_mask=visible, scale=data["sm_scale"]
            )
        scores = (qb @ kb.transpose(1, 2)) * data["sm_scale"]
        lb = scores.masked_fill(~visible, -math.inf).logsumexp(-1) * LOG2E
        empty = ~visible.any(-1)
        out[qo[b] : qo[b + 1]] = ob.transpose(0, 1).masked_fill(empty[:, None, None], 0.0)
        lse[qo[b] : qo[b + 1]] = lb.T.masked_fill(empty[:, None], EMPTY_LSE)
    return out, lse


def run_test(
    q_heads,
    kv_heads,
    head_dim,
    qo_lens,
    kv_lens=None,
    causal=True,
    window=0,
    *,
    backend_config=None,
    **kwargs,
):
    """Compile, run and check against FP32 SDPA."""
    from tirx_kernels.runner import compile_kernel

    data = prepare_data(q_heads, kv_heads, head_dim, qo_lens, kv_lens, causal, window)
    kernel = get_kernel(q_heads, kv_heads, head_dim, window, backend_config=backend_config)
    ex = compile_kernel(kernel, backend_config=backend_config)
    ex(*_kernel_args(data, window))
    torch.cuda.synchronize()
    ref_out, ref_lse = reference(data, window)
    torch.testing.assert_close(data["out"].float(), ref_out, rtol=0.002, atol=0.001)
    torch.testing.assert_close(data["lse"], ref_lse, rtol=0.0001, atol=0.00005)


def prepare_bench(*, backend_config=None, **kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    kernel = get_kernel(
        kwargs["q_heads"],
        kwargs["kv_heads"],
        kwargs["head_dim"],
        kwargs.get("window", 0),
        backend_config=backend_config,
    )
    state = {
        "config": dict(kwargs),
        "executable": compile_kernel(kernel, backend_config=backend_config),
    }
    return prepared_gpu_benchmark(run_gpu, state, backend_config=backend_config)


def _flashinfer(data, window):
    import flashinfer

    q, k, v = data["q"], data["k"], data["v"]
    workspace = torch.empty(256 << 20, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        data["qo_indptr"],
        data["kv_indptr"],
        q.shape[1],
        k.shape[1],
        q.shape[2],
        causal=bool(data["causal"]),
        sm_scale=data["sm_scale"],
        window_left=window - 1 if window else -1,
        q_data_type=q.dtype,
    )
    out, lse = torch.empty_like(q), torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    return lambda: wrapper.run(q, k, v, out=out, lse=lse)


def _torch_sdpa(data, window):
    """FP16 SDPA, one call per request (the masks are built here, outside the timing)."""
    q, k, v = data["q"], data["k"], data["v"]
    qo, kv = data["qo_indptr"].tolist(), data["kv_indptr"].tolist()
    calls = []
    for b in range(len(qo) - 1):
        qb = q[qo[b] : qo[b + 1]].transpose(0, 1)
        kb, vb = (t[kv[b] : kv[b + 1]].transpose(0, 1) for t in (k, v))
        mask = _visible(data, b, window)
        calls.append((qb, kb, vb, mask))

    def run():
        for qb, kb, vb, mask in calls:
            torch.nn.functional.scaled_dot_product_attention(
                qb, kb, vb, attn_mask=mask, scale=data["sm_scale"], enable_gqa=True
            )

    return run


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, backend_config=None, **kwargs):
    """Allocate and measure after GPU assignment."""
    cfg = prepared["config"]
    window = cfg.get("window", 0)
    data = prepare_data(**cfg)
    args = _kernel_args(data, window)
    ex = prepared["executable"]
    try:
        import flashinfer  # noqa: F401

        references = {"flashinfer": lambda: _flashinfer(data, window)}
    except ImportError:
        references = {"torch_sdpa": lambda: _torch_sdpa(data, window)}
    return bench(
        {"tirx": lambda: ex(*args)},
        references=references,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        **kwargs,
    )


def run_bench(
    q_heads,
    kv_heads,
    head_dim,
    qo_lens,
    kv_lens=None,
    causal=True,
    window=0,
    warmup=None,
    repeat=None,
    timer=None,
    *,
    backend_config=None,
    **kwargs,
):
    """Standalone wrapper over the same explicit prepare and GPU stages."""
    return prepare_bench(
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        qo_lens=qo_lens,
        kv_lens=kv_lens,
        causal=causal,
        window=window,
        backend_config=backend_config,
    ).run_gpu(warmup=warmup, repeat=repeat, timer=timer, **kwargs)
