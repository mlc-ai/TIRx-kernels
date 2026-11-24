# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM120 path of ``fp16_bf16_gemm``: TMA loads and mma.sync, FP16 or BF16.

D[M, N] = A[M, K] @ B[N, K].T with FP32 accumulation; M may be symbolic.

gemm_ws: one CTA per output tile. Thread 0 fills a TMA ring and a round-robin duty warp
refills each slot after its readers release it. Consumer teams take interleaved K chunks or
share every chunk and take disjoint k16 steps (kk-split), and reduce their FP32 partials
through shared memory. With cluster split-K, S CTAs of a cluster share one tile, each reducing
K / S, and rank 0 adds the other ranks' partials in rank order. The epilogue stages the tile
through shared memory (stmatrix) and writes it with TMA stores.

gemm_mw: one team and two column-tile widths in one launch (N = 4096, 6144, 7168).

``schedule(N, K, dtype)`` is the row-range dispatch of a shape: ``TABLE`` for tuned shapes,
with ``TABLE_BF16`` overriding it for BF16, and ``DEFAULT`` otherwise. The tables are tuned
for the RTX 5090. ``FALLBACK`` stores with st.global.b32 and needs only a 4-byte-aligned D;
every other configuration needs a 16-byte-aligned D.
"""

import contextlib

import tirx_kernels.tirx_lite as txl
from tirx_kernels.runner import resolve_backend_config

# Targets that take this path.
ARCHS = ("sm_120", "sm_120a")
# f32 pair -> packed 16-bit pair (the first operand lands in the upper half).
_CVT = {"float16": "cvt.rn.f16x2.f32", "bfloat16": "cvt.rn.bf16x2.f32"}


def _mma(dtype):
    if dtype == "float16":
        return txl.ptx.mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    return txl.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32


def gemm_ws(rows, cfg, n, k, *, dtype="float16", backend_config=None):
    assert dtype in _CVT, dtype
    bm, bn, bk, stages, wgm, wgn, teams = cfg[:7]
    # split=1: every team reads every chunk and takes bk/teams of its k columns.
    split = cfg[7] if len(cfg) > 7 else 0
    # lag=1: the duty warp refills the slot released one chunk earlier (normally free
    # already) instead of waiting for the chunk it just read.
    lag = cfg[8] if len(cfg) > 8 else 0
    # sub0=1: chunk 0 lands in ring slot 0 as two k32 boxes (64-byte swizzle), one mbarrier
    # each, so each kk team starts as soon as its half of the first chunk arrives.
    sub0 = cfg[14] if len(cfg) > 14 else 0
    # csplit=S: a cluster of S CTAs splits K for one output tile (DSMEM reduction).
    csplit = cfg[15] if len(cfg) > 15 else 1
    assert 1 <= csplit <= 4, csplit
    # direct=1 (teams == 1): store output pairs straight from registers with
    # st.global.b32 instead of the TMA-store epilogue; needs only a 4-byte-aligned D.
    direct = cfg[9] if len(cfg) > 9 else 0
    assert not direct or teams == 1
    # ucols=(c0, c1): two column groups of c0 and c1 n fragments (wgm=4, wgn=2, teams=1).
    ucols = cfg[11] if len(cfg) > 11 else None
    fm, fn = bm // (16 * wgm), bn // (8 * wgn)
    # wmap: per-warp (row frag, m frags, col frag, n frags) for uneven warp tiles, laid
    # out so every SM sub-partition (warp % 4) gets the same number of mma per k step.
    wmap = None
    # rot: with kk-split teams and uneven column groups (wgm=1), team t's warp q takes
    # column group (q + t * rot) % warps, so each SM sub-partition gets equal work.
    rot = 0
    if ucols:
        assert not direct
        if teams > 1:
            # kk-split teams: one map shared by every team, rotated by `rot` per team.
            assert split
            if isinstance(ucols[0], int):
                assert wgm == 1 and len(ucols) == wgn
                wmap = [(0, fm, sum(ucols[:w]), ucols[w]) for w in range(wgn)]
            else:
                wmap = list(ucols)
            rot = cfg[13] if len(cfg) > 13 else len(wmap) // 2
        elif isinstance(ucols[0], int):
            assert wgm == 4 and wgn == 2 and 8 * sum(ucols) == bn and fm * 64 == bm
            wmap = [(w % 4 * fm, fm, 0 if w < 4 else ucols[0], ucols[w // 4]) for w in range(8)]
        else:
            wmap = list(ucols)
        assert len(wmap) == wgm * wgn
        cells = set()
        for rf, mf, cf, nf in wmap:
            for i in range(mf):
                for j in range(nf):
                    assert (rf + i, cf + j) not in cells
                    cells.add((rf + i, cf + j))
        assert len(cells) == (bm // 16) * (bn // 8), "wmap must tile the CTA tile"
        nw = len(wmap)
        region = [(w % nw + (w // nw) * rot) % nw for w in range(nw * teams)]
        loads = [
            sum(wmap[region[w]][1] * wmap[region[w]][3] for w in range(nw * teams) if w % 4 == q)
            for q in range(4)
        ]
        assert len(set(loads)) == 1, loads
        fm = max(w[1] for w in wmap)
        fn = max(w[3] for w in wmap)
    else:
        assert fm * 16 * wgm == bm and fn * 8 * wgn == bn
    warps = wgm * wgn
    team_threads = warps * 32
    consumers = team_threads * teams
    assert (k // bk) % csplit == 0
    if csplit > 1:
        assert teams == 1 and not direct and not sub0
    chunks = k // bk // csplit
    if split:
        assert k % bk == 0 and chunks >= stages and bk % (16 * teams) == 0
    else:
        assert k % bk == 0 and chunks % teams == 0 and chunks >= stages and stages % teams == 0
    readers = warps * teams if split else warps
    kspan = bk // teams if split else bk
    swz = {32: 2, 64: 3}[bk]
    nfrag = fm * fn
    own = [(nfrag * t) // teams for t in range(teams + 1)]
    smem_bytes = stages * (bm + -(-bn // (8 * -(-bn // 256))) * 8 * -(-bn // 256)) * bk * 2
    assert bn % 8 == 0 and bm % 8 == 0 and n % 8 == 0
    # Output staging: 64-column panels with the 128-byte swizzle (full-line TMA stores),
    # then the remaining columns as 16-column (32-byte swizzle) or 8-column panels.
    wide = cfg[12] if len(cfg) > 12 else 0
    nwide = bn // 64 if wide and not direct else 0
    wcols = 64 * nwide
    rem = bn - wcols
    pw = 16 if rem % 16 == 0 else 8
    # flip: with an 8-column remainder and a tile width of 16k+8 columns, odd column
    # tiles start 16 bytes into a 32-byte sector. Those tiles store the remainder panel
    # first so every 64-column panel row is sector aligned.
    flip = bool(nwide) and bool(rem) and (bn * 2) % 32 != 0 and (bn * 2 + rem * 2) % 32 == 0
    # Epilogue reuse of the ring: (teams - 1) partial copies of the tile, then the
    # 16-bit staging tile. `extra` grows the pool when the ring alone is too small.
    stage_off = -(-(teams - 1) * nfrag * team_threads * 16 // 1024) * 1024
    extra = max(0, stage_off + bm * bn * 2 - smem_bytes)
    if csplit > 1:
        # The cluster exchange holds one FP32 tile in the ring.
        extra = max(extra, bm * bn * 4 - smem_bytes)
    # If the one-shot exchange does not fit, reduce one owner team per round through
    # a buffer X that the 16-bit staging tile later aliases.
    rounds = teams > 1 and smem_bytes + extra + 128 > 101376
    rgroups = []
    if rounds:
        # Owner teams reduced per round: as many as the pool holds (2 of 4 for 80x160).
        cap = 101376 - 1024
        sizes = [(teams - 1) * (own[t + 1] - own[t]) * team_threads * 16 for t in range(teams)]
        for t in range(teams):
            if rgroups and sum(sizes[u] for u in rgroups[-1]) + sizes[t] <= cap:
                rgroups[-1].append(t)
            else:
                rgroups.append([t])
        xmax = max(sum(sizes[u] for u in g) for g in rgroups)
        stage_off = 0
        extra = max(0, max(xmax, bm * bn * 2) - smem_bytes)
    assert smem_bytes + extra + 128 <= 101376, (smem_bytes, extra)
    # pairb=1: load B per k16 as x4 pairs of n fragments (half the B registers of the
    # k32 single-fragment loads), for wide warp tiles that would otherwise spill.
    pair_b = kspan % 32 != 0 or (len(cfg) > 10 and cfg[10])
    # TMA boxes are at most 256 rows: load a wider B tile as nbox boxes of bbox rows
    # (a multiple of 8 rows keeps the swizzle pattern continuous across boxes).
    nbox = -(-bn // 256)
    # (The last box may run past the tile: those extra B rows are loaded and unused.)
    bbox = -(-bn // (8 * nbox)) * 8
    bnl = bbox * nbox
    # Expected bytes: every box is fully credited (OOB rows are zero-filled).

    @txl.kernel()
    def gemm(
        x: txl.Tensor((rows, k), dtype, align=1),
        w: txl.Tensor((n, k), dtype, align=1),
        y: txl.Tensor((rows, n), dtype, align=1),
    ):
        x_map = txl.stack_alloca("tensormap", 1)
        w_map = txl.stack_alloca("tensormap", 1)
        y_map = None if direct or not rem else txl.stack_alloca("tensormap", 1)
        yw_map = txl.stack_alloca("tensormap", 1) if nwide else None
        for desc, src, height, box in ((x_map, x, rows, bm), (w_map, w, n, bbox)):
            txl.cu_tensor_map_encode_tiled(
                desc, dtype, 2, src.data, k, height, k * 2, bk, box, 1, 1, 0, swz, 2, 0
            )
        if sub0:
            assert bk == 64 and split and teams == 2
            x32_map = txl.stack_alloca("tensormap", 1)
            w32_map = txl.stack_alloca("tensormap", 1)
            for desc, src, height, box in ((x32_map, x, rows, bm), (w32_map, w, n, bbox)):
                txl.cu_tensor_map_encode_tiled(
                    desc, dtype, 2, src.data, k, height, k * 2, 32, box, 1, 1, 0, 2, 2, 0
                )
        # Output panels: 16 columns x BM rows, 32-byte swizzle.
        if nwide:
            txl.cu_tensor_map_encode_tiled(
                yw_map, dtype, 2, y.data, n, rows, n * 2, 64, bm, 1, 1, 0, 3, 2, 0
            )
        if y_map is not None:
            txl.cu_tensor_map_encode_tiled(
                y_map, dtype, 2, y.data, n, rows, n * 2, pw, bm, 1, 1, 0, 1 if pw == 16 else 0, 2, 0
            )
        m = txl.cast(rows, "int32")
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(
                grid=((n + bn - 1) // bn * csplit, (m + bm - 1) // bm),
                block=consumers,
                **({"cluster": (csplit, 1)} if csplit > 1 else {}),
            ),
            backend_config=resolve_backend_config(backend_config),
        )
        block_n, block_m = txl.cta_id()
        crank = 0
        if csplit > 1:
            crank = block_n % csplit
            block_n = block_n // csplit
        tid, lane = txl.thread_id(), txl.lane_id()
        warp = txl.warp_id()
        team = warp // warps
        # Region index of this warp within its team (rotated per team when rot != 0).
        wit = (warp % warps + team * rot) % warps if rot else warp % warps
        row = (wit // wgn) * (16 * fm)
        col = (wit % wgn) * (8 * fn)
        pool = txl.smem_pool()
        swizzle = (
            txl.SwizzleMode.SWIZZLE_64B_ATOM if bk == 32 else txl.SwizzleMode.SWIZZLE_128B_ATOM
        )
        sx = pool.alloc((stages, bm, bk), dtype, swizzle=swizzle, align=1024)
        sw = pool.alloc((stages, bnl, bk), dtype, swizzle=swizzle, align=1024)
        if extra:
            pool.alloc((extra,), "uint8", align=16)
        full = pool.alloc((stages,), "uint64", align=8)
        empty = pool.alloc((stages,), "uint64", align=8)
        if sub0:
            subb = pool.alloc((2,), "uint64", align=8)
            cursor = pool.pool.offset
            # Ring slot 0 viewed as two k32 tiles with the 64-byte swizzle (same bytes).
            pool.pool.move_base_to(pool.offset_of(sx))
            sx32 = pool.alloc(
                (2, bm, 32), dtype, swizzle=txl.SwizzleMode.SWIZZLE_64B_ATOM, align=1024
            )
            pool.pool.move_base_to(pool.offset_of(sw))
            sw32 = pool.alloc(
                (2, bnl, 32), dtype, swizzle=txl.SwizzleMode.SWIZZLE_64B_ATOM, align=1024
            )
            pool.pool.move_base_to(cursor)

        def regs(size, ty):
            return [txl.local_scalar(ty) for _ in range(size)]

        load = txl.ptx["cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"]

        def issue(step, slot):
            barrier = full.ptr_to([slot])
            txl.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier, txl.uint32(2 * (bm + bnl) * bk))
            bar_addr = txl.cuda.cvta_generic_to_shared(barrier)
            kc = txl.cast((step + crank * chunks if csplit > 1 else step) * bk, "int32")
            load(sx[slot].ptr_to(0, 0), txl.address_of(x_map), kc, block_m * bm, bar_addr)
            for h in range(nbox):
                load(
                    sw[slot].ptr_to(h * bbox, 0),
                    txl.address_of(w_map),
                    kc,
                    block_n * bn + h * bbox,
                    bar_addr,
                )

        # No producer warp: thread 0 fills the ring. Every warp arrives on empty[slot]
        # after reading a chunk; one duty warp per chunk (round-robin within the team)
        # waits for that release and refills the slot `stages` ahead.
        # stages % teams == 0 keeps every slot owned by one team, so a parity wait can
        # never alias a stale phase.
        with txl.if_(tid == 0), txl.then_():
            for desc in [d for d in (x_map, w_map, y_map, yw_map) if d is not None]:
                txl.ptx.prefetch.tensormap(txl.address_of(desc))
            for s in range(stages):
                txl.ptx.mbarrier.init.shared.b64(full.ptr_to([s]), txl.uint32(1))
                txl.ptx.mbarrier.init.shared.b64(empty.ptr_to([s]), txl.uint32(readers))
            txl.ptx.fence.mbarrier_init.release.cluster()
            if sub0:
                # Phase 0 of full[0] completes at once; chunk 0 uses the two sub barriers.
                txl.ptx.mbarrier.arrive.shared.b64(full.ptr_to([0]), txl.uint32(1))
                for q in range(2):
                    txl.ptx.mbarrier.init.shared.b64(subb.ptr_to([q]), txl.uint32(1))
                txl.ptx.fence.mbarrier_init.release.cluster()
                for q in range(2):
                    barrier = subb.ptr_to([q])
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        barrier, txl.uint32(2 * (bm + bnl) * 32)
                    )
                    bar_addr = txl.cuda.cvta_generic_to_shared(barrier)
                    load(
                        sx32[q].ptr_to(0, 0),
                        txl.address_of(x32_map),
                        q * 32,
                        block_m * bm,
                        bar_addr,
                    )
                    for h in range(nbox):
                        load(
                            sw32[q].ptr_to(h * bbox, 0),
                            txl.address_of(w32_map),
                            q * 32,
                            block_n * bn + h * bbox,
                            bar_addr,
                        )
                for s in range(1, stages):
                    issue(s, s)
            else:
                for s in range(stages):
                    issue(s, s)
        txl.ptx.bar.sync(0)

        def consumer(fn, row, col, fm=fm):
            nfrag = fm * fn
            own = [(nfrag * t) // teams for t in range(teams + 1)]
            a = [regs(4, "uint32") for _ in range(fm)]
            b = [regs(2, "uint32") for _ in range(fn)]
            later = [regs(2, "uint32") for _ in range(fn)]
            acc = [
                [[txl.local_scalar("float32", 0) for _ in range(4)] for _ in range(fn)]
                for _ in range(fm)
            ]

            def consume(slot):
                for kk0 in range(0, kspan, 16):
                    kk = kk0 + team * kspan if split else kk0
                    for i in range(fm):
                        txl.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                            *a[i], sx[slot].m8n8x4(row + i * 16, kk, lane)
                        )
                    if pair_b:
                        for j in range(0, fn - 1, 2):
                            addr = sw[slot].ptr_to(
                                col + j * 8 + (lane // 16) * 8 + lane % 8,
                                kk + ((lane // 8) % 2) * 8,
                            )
                            txl.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(*b[j], *b[j + 1], addr)
                        if fn % 2:
                            addr = sw[slot].ptr_to(
                                col + (fn - 1) * 8 + lane % 8, kk + ((lane // 8) % 2) * 8
                            )
                            txl.ptx.ldmatrix.sync.aligned.m8n8.x2.shared.b16(*b[fn - 1], addr)
                    elif kk0 % 32 == 0:
                        for j in range(fn):
                            addr = sw[slot].ptr_to(
                                col + j * 8 + lane % 8,
                                kk + (lane // 16) * 16 + ((lane // 8) % 2) * 8,
                            )
                            txl.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(*b[j], *later[j], addr)
                    ops = later if (kk0 % 32 and not pair_b) else b
                    for i in range(fm):
                        for j in range(fn):
                            c = acc[i][j]
                            _mma(dtype)(*c, *a[i], *ops[j], *c)

            def consume0():
                # Chunk 0: team t reads its k32 half from sub tile t once its barrier fires.
                # (Waiting on the pre-completed phase 0 of full[0] keeps its phases paired.)
                txl.cuda.mbarrier_wait(full.ptr_to([0]), txl.cast(0, "int32"))
                txl.cuda.mbarrier_wait(subb.ptr_to([team]), txl.cast(0, "int32"))
                for ki in range(0, 32, 16):
                    for i in range(fm):
                        txl.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                            *a[i], sx32[team].m8n8x4(row + i * 16, ki, lane)
                        )
                    for j in range(0, fn - 1, 2):
                        addr = sw32[team].ptr_to(
                            col + j * 8 + (lane // 16) * 8 + lane % 8, ki + ((lane // 8) % 2) * 8
                        )
                        txl.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(*b[j], *b[j + 1], addr)
                    if fn % 2:
                        addr = sw32[team].ptr_to(
                            col + (fn - 1) * 8 + lane % 8, ki + ((lane // 8) % 2) * 8
                        )
                        txl.ptx.ldmatrix.sync.aligned.m8n8.x2.shared.b16(*b[fn - 1], addr)
                    for i in range(fm):
                        for j in range(fn):
                            c = acc[i][j]
                            _mma(dtype)(*c, *a[i], *b[j], *c)

            def chunk_body(it, first):
                step = it if split else it * teams + team
                slot = step % stages
                if first:
                    consume0()
                else:
                    txl.cuda.mbarrier_wait(
                        full.ptr_to([slot]), txl.cast(step // stages % 2, "int32")
                    )
                    consume(slot)
                txl.cuda.warp_sync()
                with txl.if_(lane == 0), txl.then_():
                    txl.ptx.mbarrier.arrive.shared.b64(empty.ptr_to([slot]), txl.uint32(1))
                # Round-robin refill duty: the duty warp waits for its teammates' release
                # of this chunk, then refills the slot `stages` ahead.
                duty = (warp if split else wit) == it % readers
                if lag:
                    prev = step - (1 if (split or teams == 1) else teams)
                    with (
                        txl.if_(duty & (lane == 0) & (it > 0) & (prev + stages < chunks)),
                        txl.then_(),
                    ):
                        txl.cuda.mbarrier_wait(
                            empty.ptr_to([prev % stages]), txl.cast(prev // stages % 2, "int32")
                        )
                        txl.ptx["fence.proxy.async.shared::cta"]()
                        issue(prev + stages, prev % stages)
                else:
                    with txl.if_(duty & (lane == 0) & (step + stages < chunks)), txl.then_():
                        txl.cuda.mbarrier_wait(
                            empty.ptr_to([slot]), txl.cast(step // stages % 2, "int32")
                        )
                        txl.ptx["fence.proxy.async.shared::cta"]()
                        issue(step + stages, slot)

            nit = chunks if split else chunks // teams
            if sub0:
                chunk_body(txl.int32(0), True)
                with txl.serial(nit - 1) as it1:
                    chunk_body(it1 + 1, False)
            else:
                with txl.serial(nit) as it:
                    chunk_body(it, False)

            frags = [(i, j) for i in range(fm) for j in range(fn)]
            base = txl.reinterpret(sx.buf.data, ty=txl.Ptr("float32", "shared.dyn"))
            # All TMA writes have landed (every full barrier was waited on) and all
            # consumers are past their last ldmatrix after this barrier.
            txl.ptx.bar.sync(1, consumers)
            T = teams

            def unit(f, src):
                # Slot of team `src`'s partial of fragment f in its owner's region.
                t = max(u for u in range(T) if own[u] <= f)
                slot_ = src if src < t else src - 1
                return (T - 1) * own[t] + slot_ * (own[t + 1] - own[t]) + (f - own[t])

            tt = wit * 32 + lane if rot else tid % team_threads
            if teams > 1 and not rounds:
                part = txl.decl_tensor(
                    ((T - 1) * nfrag, team_threads, 4), "float32", data=base, align=16
                )
                for t in range(teams):
                    with txl.if_(team == t), txl.then_():
                        for f, (i, j) in enumerate(frags):
                            if not (own[t] <= f < own[t + 1]):
                                txl.ptx.st.shared.v4.f32(
                                    part.ptr_to([unit(f, t), tt, 0]), *acc[i][j]
                                )
                txl.ptx.bar.sync(1, consumers)
            if rounds:
                xpart = txl.decl_tensor(
                    ((T - 1) * nfrag, team_threads, 4), "float32", data=base, align=16
                )
                other_r = regs(4, "float32")
                for grp in rgroups:
                    base_of, off = {}, 0
                    for d in grp:
                        base_of[d] = off
                        off += (T - 1) * (own[d + 1] - own[d])
                    for u in range(T):
                        with txl.if_(team == u), txl.then_():
                            for d in grp:
                                if u == d:
                                    continue
                                nd = own[d + 1] - own[d]
                                sl = u if u < d else u - 1
                                for f in range(own[d], own[d + 1]):
                                    i, j = frags[f]
                                    txl.ptx.st.shared.v4.f32(
                                        xpart.ptr_to([base_of[d] + sl * nd + f - own[d], tt, 0]),
                                        *acc[i][j],
                                    )
                    txl.ptx.bar.sync(1, consumers)
                    for d in grp:
                        nd = own[d + 1] - own[d]
                        with txl.if_(team == d), txl.then_():
                            for f in range(own[d], own[d + 1]):
                                i, j = frags[f]
                                for sl in range(T - 1):
                                    txl.ptx.ld.shared.v4.f32(
                                        *other_r,
                                        xpart.ptr_to([base_of[d] + sl * nd + f - own[d], tt, 0]),
                                    )
                                    for h in range(4):
                                        txl.assign(acc[i][j][h], acc[i][j][h] + other_r[h])
                    txl.ptx.bar.sync(1, consumers)
            if csplit > 1:
                # Cluster split-K: CTA rank r of the cluster reduced K chunks
                # [r * chunks, (r + 1) * chunks); ranks 1.. push their FP32 tile into rank 0's
                # ring (free once every CTA of the cluster passed its mainloop), and rank 0
                # adds them in rank order (deterministic) before its epilogue.
                # One rank per round, so the exchange needs one FP32 tile of smem.
                cpart = txl.decl_tensor((nfrag, team_threads, 4), "float32", data=base, align=16)
                raddr = txl.local_scalar("uint32")
                cother = regs(4, "float32")
                for r in range(1, csplit):
                    txl.ptx.barrier.cluster.arrive.release()
                    txl.ptx.barrier.cluster.wait.acquire()
                    with txl.if_(crank == r), txl.then_():
                        for f, (i, j) in enumerate(frags):
                            laddr = txl.cuda.cvta_generic_to_shared(cpart.ptr_to([f, tt, 0]))
                            txl.ptx["mapa.shared::cluster.u32"](raddr, laddr, txl.uint32(0))
                            txl.ptx["st.shared::cluster.v4.f32"](raddr, *acc[i][j])
                    txl.ptx.barrier.cluster.arrive.release()
                    txl.ptx.barrier.cluster.wait.acquire()
                    with txl.if_(crank == 0), txl.then_():
                        for f, (i, j) in enumerate(frags):
                            txl.ptx.ld.shared.v4.f32(*cother, cpart.ptr_to([f, tt, 0]))
                            for h in range(4):
                                txl.assign(acc[i][j][h], acc[i][j][h] + cother[h])
                txl.ptx.bar.sync(1, consumers)
            # With cluster split-K only rank 0 stores the tile.
            with (
                txl.if_(crank == 0) if csplit > 1 else contextlib.nullcontext(),
                txl.then_() if csplit > 1 else contextlib.nullcontext(),
            ):
                stg = txl.decl_tensor(
                    (bm * bn,),
                    dtype,
                    data=txl.ptr_byte_offset(base, stage_off, ty=txl.Ptr(dtype, "shared.dyn")),
                    align=1024,
                )

                def stage_addr(r, c):
                    # Element offset of (row r, column c) of the tile in the staging panels.
                    wide_off = (c // 64) * (bm * 64) + r * 64 + (((c % 64) // 8) ^ (r % 8)) * 8
                    cr = c - wcols
                    if pw == 16:
                        rem_off = (
                            wcols * bm
                            + (cr // 16) * (bm * 16)
                            + r * 16
                            + (((cr // 8) % 2) ^ ((r // 4) % 2)) * 8
                        )
                    else:
                        rem_off = wcols * bm + (cr // 8) * (bm * 8) + r * 8
                    if not rem:
                        return stg.ptr_to([wide_off])
                    if not nwide:
                        return stg.ptr_to([rem_off])
                    even = txl.Select(c < wcols, wide_off, rem_off)
                    if not flip:
                        return stg.ptr_to([even])
                    c8 = c - rem
                    if pw == 16:
                        rem_odd = (
                            wcols * bm
                            + (c // 16) * (bm * 16)
                            + r * 16
                            + (((c // 8) % 2) ^ ((r // 4) % 2)) * 8
                        )
                    else:
                        rem_odd = wcols * bm + (c // 8) * (bm * 8) + r * 8
                    odd = txl.Select(
                        c < rem,
                        rem_odd,
                        (c8 // 64) * (bm * 64) + r * 64 + (((c8 % 64) // 8) ^ (r % 8)) * 8,
                    )
                    return stg.ptr_to([txl.Select(block_n % 2 == 1, odd, even)])

                def stage_out(i, j, pair):
                    c0 = acc[i][j]
                    packed = regs(4 if pair else 2, "uint32")
                    srcs = [c0] + ([acc[i][j + 1]] if pair else [])
                    for q, c in enumerate(srcs):
                        txl.ptx[_CVT[dtype]](packed[2 * q], c[1], c[0])
                        txl.ptx[_CVT[dtype]](packed[2 * q + 1], c[3], c[2])
                    r = row + i * 16 + lane % 16
                    cc = col + (j + (lane // 16 if pair else 0)) * 8
                    addr = stage_addr(r, cc)
                    if pair:
                        txl.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(addr, *packed)
                    else:
                        txl.ptx.stmatrix.sync.aligned.m8n8.x2.shared.b16(addr, *packed)

                def finish(fs):
                    done = set()
                    for f in fs:
                        i, j = frags[f]
                        if f in done:
                            continue
                        pair = fn % 2 == 0 and j % 2 == 0 and (f + 1) in fs
                        stage_out(i, j, pair)
                        done.add(f)
                        if pair:
                            done.add(f + 1)

                if direct:
                    for i in range(fm):
                        for h in range(2):
                            r = block_m * bm + row + i * 16 + h * 8 + lane // 4
                            with txl.if_(r < m), txl.then_():
                                for j in range(fn):
                                    c = acc[i][j]
                                    cc = block_n * bn + col + j * 8 + (lane % 4) * 2
                                    pk = txl.local_scalar("uint32")
                                    txl.ptx[_CVT[dtype]](pk, c[h * 2 + 1], c[h * 2])
                                    if n % bn:
                                        with txl.if_(cc < n), txl.then_():
                                            txl.ptx.st.global_.b32(y.ptr_to([r, cc]), pk)
                                    else:
                                        txl.ptx.st.global_.b32(y.ptr_to([r, cc]), pk)
                    return
                if teams == 1:
                    finish(list(range(nfrag)))
                elif rounds:
                    for t in range(teams):
                        with txl.if_(team == t), txl.then_():
                            finish(list(range(own[t], own[t + 1])))
                else:
                    other = regs(4, "float32")
                    for t in range(teams):
                        with txl.if_(team == t), txl.then_():
                            for f in range(own[t], own[t + 1]):
                                i, j = frags[f]
                                for u in range(T):
                                    if u == t:
                                        continue
                                    txl.ptx.ld.shared.v4.f32(
                                        *other, part.ptr_to([unit(f, u), tt, 0])
                                    )
                                    for h in range(4):
                                        txl.assign(acc[i][j][h], acc[i][j][h] + other[h])
                            finish(list(range(own[t], own[t + 1])))
                txl.ptx["fence.proxy.async.shared::cta"]()
                txl.ptx.bar.sync(1, consumers)
                with txl.if_(tid == 0), txl.then_():
                    store = txl.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"]
                    sh = (block_n % 2) * rem if flip else 0
                    for p in range(nwide):
                        store(
                            txl.address_of(yw_map),
                            block_n * bn + p * 64 + sh,
                            block_m * bm,
                            stg.ptr_to([p * bm * 64]),
                        )
                    for p in range(rem // pw):
                        store(
                            txl.address_of(y_map),
                            block_n * bn + wcols + p * pw - (wcols * (block_n % 2) if flip else 0),
                            block_m * bm,
                            stg.ptr_to([wcols * bm + p * bm * pw]),
                        )
                    txl.ptx.cp.async_.bulk.commit_group()
                    txl.ptx.cp.async_.bulk.wait_group.read(0)

        with txl.if_(tid < consumers), txl.then_():
            if rot:
                # Shape groups are contiguous ranges of wit; offsets affine in wit or Select.
                groups = []
                for w, (rf, mf, cf, nf) in enumerate(wmap):
                    if groups and groups[-1][1:] == (mf, nf):
                        groups[-1][0].append(w)
                    else:
                        groups.append(([w], mf, nf))

                def offr(ws, k_, unit):
                    vals = [wmap[w][k_] * unit for w in ws]
                    if len(set(vals)) == 1:
                        return vals[0]
                    d = (vals[1] - vals[0]) // (ws[1] - ws[0])
                    if all(v == vals[0] + (w - ws[0]) * d for v, w in zip(vals, ws)):
                        return (wit - ws[0]) * d + vals[0] if ws[0] or vals[0] else wit * d
                    expr = txl.int32(vals[0])
                    for w, v in zip(ws[1:], vals[1:]):
                        expr = txl.Select(wit == w, txl.int32(v), expr)
                    return expr

                def emit_r(gi):
                    ws, mf, nf = groups[gi]
                    args = (nf, offr(ws, 0, 16), offr(ws, 2, 8))
                    if gi == len(groups) - 1:
                        consumer(*args, fm=mf)
                        return
                    with txl.if_(wit < groups[gi + 1][0][0]):
                        with txl.then_():
                            consumer(*args, fm=mf)
                        with txl.else_():
                            emit_r(gi + 1)

                emit_r(0)
            elif wmap:
                # One code path per distinct warp-tile shape; offsets selected by warp id.
                shapes = list(dict.fromkeys((mf, nf) for _, mf, _, nf in wmap))

                def offs(ws, k_, unit):
                    vals = [wmap[w][k_] * unit for w in ws]
                    if len(set(vals)) == 1:
                        return vals[0]
                    # Closed forms in the warp id keep the address math cheap.
                    for f, g in (
                        (lambda w: w % 4, lambda: warp % 4),
                        (lambda w: w % 4 // 2, lambda: warp % 4 // 2),
                        (lambda w: w % 2, lambda: warp % 2),
                    ):
                        xs = [f(w) for w in ws]
                        if len(set(xs)) < 2:
                            continue
                        a0 = [(v, x) for v, x in zip(vals, xs)]
                        (v0, x0), (v1, x1) = a0[0], next(t for t in a0 if t[1] != a0[0][1])
                        if (v1 - v0) % (x1 - x0):
                            continue
                        d = (v1 - v0) // (x1 - x0)
                        if all(v == v0 + (x - x0) * d for v, x in a0):
                            return g() * d + (v0 - x0 * d)
                    expr = txl.int32(vals[0])
                    for w, v in zip(ws[1:], vals[1:]):
                        expr = txl.Select(warp == w, txl.int32(v), expr)
                    return expr

                def emit(rest, lo=0):
                    mf, nf = rest[0]
                    ws = [w for w in range(len(wmap)) if (wmap[w][1], wmap[w][3]) == (mf, nf)]
                    if len(rest) == 1:
                        consumer(nf, offs(ws, 0, 16), offs(ws, 2, 8), fm=mf)
                        return
                    # Shape groups must be contiguous warp ranges, in shape order.
                    assert ws == list(range(lo, lo + len(ws))), ws
                    with txl.if_(warp < lo + len(ws)):
                        with txl.then_():
                            consumer(nf, offs(ws, 0, 16), offs(ws, 2, 8), fm=mf)
                        with txl.else_():
                            emit(rest[1:], lo + len(ws))

                emit(shapes)
            else:
                consumer(fn, row, col)

    return gemm.func


def gemm_mw(rows, cfg, n, k, *, dtype="float16", backend_config=None):
    """Mixed-width column tiles, one team of 8 warps: the first `count0` column tiles are
    `bn0` wide, the rest `bn1` (narrower). The narrow tiles finish their mainloop early
    and their TMA stores drain while the wide tiles are still computing, so only the wide
    tiles' store is exposed at the end of the kernel.
    cfg = (BM, BK, stages, lag, ((bn, count, layout, pair_b), ...)[, s0wait]); layout is (wgm, wgn)
    for a regular warp grid or a list of (row frag, m frags, col frag, n frags) per warp."""
    assert dtype in _CVT, dtype
    bm, bk, stages, lag, classes = cfg[:5]
    # s0wait=1: issue chunk 0 alone and the rest of the ring only once it has landed, so
    # nothing competes with the cold first chunk's fan-out.
    s0wait = cfg[5] if len(cfg) > 5 else 0
    assert len(classes) == 2
    swz = {32: 2, 64: 3}[bk]
    chunks = k // bk
    assert k % bk == 0 and chunks >= stages
    C = []
    c0 = 0
    for bn, cnt, layout, pb in classes:
        nbox = -(-bn // 256)
        bbox = -(-bn // (8 * nbox)) * 8
        if isinstance(layout[0], int):
            wgm, wgn = layout
            fm, fn = bm // (16 * wgm), bn // (8 * wgn)
            assert fm * 16 * wgm == bm and fn * 8 * wgn == bn and wgm * wgn == 8
            wmap = [((w // wgn) * fm, fm, (w % wgn) * fn, fn) for w in range(8)]
        else:
            wmap = list(layout)
        cells = set()
        for rf, mf, cf, nf in wmap:
            for i in range(mf):
                for j in range(nf):
                    assert (rf + i, cf + j) not in cells
                    cells.add((rf + i, cf + j))
        assert len(cells) == (bm // 16) * (bn // 8)
        loads = [sum(wmap[w][1] * wmap[w][3] for w in range(8) if w % 4 == q) for q in range(4)]
        assert len(set(loads)) == 1, loads
        nwide = bn // 64
        rem = bn - 64 * nwide
        pw = 16 if rem % 16 == 0 else 8
        C.append(
            dict(
                bn=bn,
                cnt=cnt,
                wmap=wmap,
                pb=pb,
                nbox=nbox,
                bbox=bbox,
                bnl=bbox * nbox,
                nwide=nwide,
                wcols=64 * nwide,
                rem=rem,
                pw=pw,
                col0=c0,
                flip=bool(nwide) and rem == 8 and (bn * 2) % 32 == 16 and (c0 * 2) % 32 == 0,
            )
        )
        c0 += bn * cnt
    assert c0 >= n and c0 - max(c["bn"] for c in C) < n
    ntiles = sum(c["cnt"] for c in C)
    bnl_max = max(c["bnl"] for c in C)
    smem_bytes = stages * (bm + bnl_max) * bk * 2
    extra = max(0, max(bm * c["bn"] * 2 for c in C) - smem_bytes)
    assert smem_bytes + extra + 128 <= 101376, (smem_bytes, extra)
    pws = sorted(set(c["pw"] for c in C if c["rem"]))

    @txl.kernel()
    def gemm(
        x: txl.Tensor((rows, k), dtype, align=1),
        w: txl.Tensor((n, k), dtype, align=1),
        y: txl.Tensor((rows, n), dtype, align=1),
    ):
        x_map = txl.stack_alloca("tensormap", 1)
        w_maps = [txl.stack_alloca("tensormap", 1) for _ in C]
        yw_map = txl.stack_alloca("tensormap", 1)
        y_maps = {p: txl.stack_alloca("tensormap", 1) for p in pws}
        txl.cu_tensor_map_encode_tiled(
            x_map, dtype, 2, x.data, k, rows, k * 2, bk, bm, 1, 1, 0, swz, 2, 0
        )
        for c, wm in zip(C, w_maps):
            txl.cu_tensor_map_encode_tiled(
                wm, dtype, 2, w.data, k, n, k * 2, bk, c["bbox"], 1, 1, 0, swz, 2, 0
            )
        txl.cu_tensor_map_encode_tiled(
            yw_map, dtype, 2, y.data, n, rows, n * 2, 64, bm, 1, 1, 0, 3, 2, 0
        )
        for p, ym in y_maps.items():
            txl.cu_tensor_map_encode_tiled(
                ym, dtype, 2, y.data, n, rows, n * 2, p, bm, 1, 1, 0, 1 if p == 16 else 0, 2, 0
            )
        m = txl.cast(rows, "int32")
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=(ntiles, (m + bm - 1) // bm), block=256),
            backend_config=resolve_backend_config(backend_config),
        )
        block_n, block_m = txl.cta_id()
        tid, lane = txl.thread_id(), txl.lane_id()
        warp = txl.warp_id()
        pool = txl.smem_pool()
        swizzle = (
            txl.SwizzleMode.SWIZZLE_64B_ATOM if bk == 32 else txl.SwizzleMode.SWIZZLE_128B_ATOM
        )
        sx = pool.alloc((stages, bm, bk), dtype, swizzle=swizzle, align=1024)
        sw = pool.alloc((stages, bnl_max, bk), dtype, swizzle=swizzle, align=1024)
        if extra:
            pool.alloc((extra,), "uint8", align=16)
        full = pool.alloc((stages,), "uint64", align=8)
        empty = pool.alloc((stages,), "uint64", align=8)
        load = txl.ptx["cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"]
        store = txl.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"]

        def regs(size, ty):
            return [txl.local_scalar(ty) for _ in range(size)]

        def body(ci):
            c = C[ci]
            bn, bnl, nbox, bbox, wmap, pair_b = (
                c["bn"],
                c["bnl"],
                c["nbox"],
                c["bbox"],
                c["wmap"],
                c["pb"],
            )
            nwide, wcols, rem, pw, flip = c["nwide"], c["wcols"], c["rem"], c["pw"], c["flip"]
            lb = block_n - sum(C[u]["cnt"] for u in range(ci))
            n0 = c["col0"] + lb * bn

            def issue(step, slot):
                barrier = full.ptr_to([slot])
                txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                    barrier, txl.uint32(2 * (bm + bnl) * bk)
                )
                bar_addr = txl.cuda.cvta_generic_to_shared(barrier)
                kc = txl.cast(step * bk, "int32")
                load(sx[slot].ptr_to(0, 0), txl.address_of(x_map), kc, block_m * bm, bar_addr)
                for h in range(nbox):
                    load(
                        sw[slot].ptr_to(h * bbox, 0),
                        txl.address_of(w_maps[ci]),
                        kc,
                        n0 + h * bbox,
                        bar_addr,
                    )

            with txl.if_(tid == 0), txl.then_():
                for desc in [x_map, w_maps[ci], yw_map, *y_maps.values()]:
                    txl.ptx.prefetch.tensormap(txl.address_of(desc))
                for s in range(stages):
                    txl.ptx.mbarrier.init.shared.b64(full.ptr_to([s]), txl.uint32(1))
                    txl.ptx.mbarrier.init.shared.b64(empty.ptr_to([s]), txl.uint32(8))
                txl.ptx.fence.mbarrier_init.release.cluster()
                if s0wait:
                    issue(0, 0)
                    txl.cuda.mbarrier_wait(full.ptr_to([0]), txl.cast(0, "int32"))
                    for s in range(1, stages):
                        issue(s, s)
                else:
                    for s in range(stages):
                        issue(s, s)
            txl.ptx.bar.sync(0)
            stg = txl.decl_tensor(
                (bm * bn,),
                dtype,
                data=txl.reinterpret(sx.buf.data, ty=txl.Ptr(dtype, "shared.dyn")),
                align=1024,
            )

            def stage_addr(r, cc):
                wide_off = (cc // 64) * (bm * 64) + r * 64 + (((cc % 64) // 8) ^ (r % 8)) * 8
                cr = cc - wcols
                if pw == 16:
                    rem_off = (
                        wcols * bm
                        + (cr // 16) * (bm * 16)
                        + r * 16
                        + (((cr // 8) % 2) ^ ((r // 4) % 2)) * 8
                    )
                else:
                    rem_off = wcols * bm + (cr // 8) * (bm * 8) + r * 8
                if not rem:
                    return stg.ptr_to([wide_off])
                even = txl.Select(cc < wcols, wide_off, rem_off)
                if not flip:
                    return stg.ptr_to([even])
                c8 = cc - rem
                odd = txl.Select(
                    cc < rem,
                    wcols * bm + r * 8,
                    (c8 // 64) * (bm * 64) + r * 64 + (((c8 % 64) // 8) ^ (r % 8)) * 8,
                )
                return stg.ptr_to([txl.Select(lb % 2 == 1, odd, even)])

            def consumer(fm, fn, row, col):
                a = [regs(4, "uint32") for _ in range(fm)]
                b = [regs(2, "uint32") for _ in range(fn)]
                later = [regs(2, "uint32") for _ in range(fn)]
                acc = [
                    [[txl.local_scalar("float32", 0) for _ in range(4)] for _ in range(fn)]
                    for _ in range(fm)
                ]

                def consume(slot):
                    for kk in range(0, bk, 16):
                        for i in range(fm):
                            txl.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                                *a[i], sx[slot].m8n8x4(row + i * 16, kk, lane)
                            )
                        if pair_b:
                            for j in range(0, fn - 1, 2):
                                addr = sw[slot].ptr_to(
                                    col + j * 8 + (lane // 16) * 8 + lane % 8,
                                    kk + ((lane // 8) % 2) * 8,
                                )
                                txl.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                                    *b[j], *b[j + 1], addr
                                )
                            if fn % 2:
                                addr = sw[slot].ptr_to(
                                    col + (fn - 1) * 8 + lane % 8, kk + ((lane // 8) % 2) * 8
                                )
                                txl.ptx.ldmatrix.sync.aligned.m8n8.x2.shared.b16(*b[fn - 1], addr)
                        elif kk % 32 == 0:
                            for j in range(fn):
                                addr = sw[slot].ptr_to(
                                    col + j * 8 + lane % 8,
                                    kk + (lane // 16) * 16 + ((lane // 8) % 2) * 8,
                                )
                                txl.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                                    *b[j], *later[j], addr
                                )
                        ops = later if (kk % 32 and not pair_b) else b
                        for i in range(fm):
                            for j in range(fn):
                                cc = acc[i][j]
                                _mma(dtype)(*cc, *a[i], *ops[j], *cc)

                with txl.serial(chunks) as step:
                    slot = step % stages
                    txl.cuda.mbarrier_wait(
                        full.ptr_to([slot]), txl.cast(step // stages % 2, "int32")
                    )
                    consume(slot)
                    txl.cuda.warp_sync()
                    with txl.if_(lane == 0), txl.then_():
                        txl.ptx.mbarrier.arrive.shared.b64(empty.ptr_to([slot]), txl.uint32(1))
                    duty = warp == step % 8
                    if lag:
                        prev = step - 1
                        with (
                            txl.if_(duty & (lane == 0) & (step > 0) & (prev + stages < chunks)),
                            txl.then_(),
                        ):
                            txl.cuda.mbarrier_wait(
                                empty.ptr_to([prev % stages]), txl.cast(prev // stages % 2, "int32")
                            )
                            txl.ptx["fence.proxy.async.shared::cta"]()
                            issue(prev + stages, prev % stages)
                    else:
                        with txl.if_(duty & (lane == 0) & (step + stages < chunks)), txl.then_():
                            txl.cuda.mbarrier_wait(
                                empty.ptr_to([slot]), txl.cast(step // stages % 2, "int32")
                            )
                            txl.ptx["fence.proxy.async.shared::cta"]()
                            issue(step + stages, slot)
                txl.ptx.bar.sync(1, 256)
                done = set()
                for i in range(fm):
                    for j in range(fn):
                        if (i, j) in done:
                            continue
                        pair = fn % 2 == 0 and j % 2 == 0
                        srcs = [acc[i][j]] + ([acc[i][j + 1]] if pair else [])
                        packed = regs(2 * len(srcs), "uint32")
                        for q, cs in enumerate(srcs):
                            txl.ptx[_CVT[dtype]](packed[2 * q], cs[1], cs[0])
                            txl.ptx[_CVT[dtype]](packed[2 * q + 1], cs[3], cs[2])
                        r = row + i * 16 + lane % 16
                        cc = col + (j + (lane // 16 if pair else 0)) * 8
                        addr = stage_addr(r, cc)
                        if pair:
                            txl.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(addr, *packed)
                            done.add((i, j + 1))
                        else:
                            txl.ptx.stmatrix.sync.aligned.m8n8.x2.shared.b16(addr, *packed)
                        done.add((i, j))

            # One code path per warp-tile shape (shape groups are contiguous warp ranges).
            shapes = list(dict.fromkeys((mf, nf) for _, mf, _, nf in wmap))

            def offs(ws, k_, unit):
                vals = [wmap[u][k_] * unit for u in ws]
                if len(set(vals)) == 1:
                    return vals[0]
                for f, g in (
                    (lambda u: u % 4, lambda: warp % 4),
                    (lambda u: u % 4 // 2, lambda: warp % 4 // 2),
                    (lambda u: u % 2, lambda: warp % 2),
                    (lambda u: u // 4, lambda: warp // 4),
                    (lambda u: u, lambda: warp),
                ):
                    xs = [f(u) for u in ws]
                    if len(set(xs)) < 2:
                        continue
                    a0 = list(zip(vals, xs))
                    (v0, x0), (v1, x1) = a0[0], next(t for t in a0 if t[1] != a0[0][1])
                    if (v1 - v0) % (x1 - x0):
                        continue
                    d = (v1 - v0) // (x1 - x0)
                    if all(v == v0 + (xx - x0) * d for v, xx in a0):
                        return g() * d + (v0 - x0 * d)
                expr = txl.int32(vals[0])
                for u, v in zip(ws[1:], vals[1:]):
                    expr = txl.Select(warp == u, txl.int32(v), expr)
                return expr

            def emit(rest, lo=0):
                mf, nf = rest[0]
                ws = [u for u in range(8) if (wmap[u][1], wmap[u][3]) == (mf, nf)]
                if len(rest) == 1:
                    consumer(mf, nf, offs(ws, 0, 16), offs(ws, 2, 8))
                    return
                assert ws == list(range(lo, lo + len(ws))), ws
                with txl.if_(warp < lo + len(ws)):
                    with txl.then_():
                        consumer(mf, nf, offs(ws, 0, 16), offs(ws, 2, 8))
                    with txl.else_():
                        emit(rest[1:], lo + len(ws))

            emit(shapes)
            txl.ptx["fence.proxy.async.shared::cta"]()
            txl.ptx.bar.sync(1, 256)
            with txl.if_(tid == 0), txl.then_():
                sh = (lb % 2) * rem if flip else 0
                for p in range(nwide):
                    store(
                        txl.address_of(yw_map),
                        n0 + p * 64 + sh,
                        block_m * bm,
                        stg.ptr_to([p * bm * 64]),
                    )
                for p in range(rem // pw):
                    store(
                        txl.address_of(y_maps[pw]),
                        n0 + wcols + p * pw - (wcols * (lb % 2) if flip else 0),
                        block_m * bm,
                        stg.ptr_to([wcols * bm + p * bm * pw]),
                    )
                txl.ptx.cp.async_.bulk.commit_group()
                txl.ptx.cp.async_.bulk.wait_group.read(0)

        if not C[1]["cnt"]:
            body(0)
        else:
            with txl.if_(block_n < C[0]["cnt"]):
                with txl.then_():
                    body(0)
                with txl.else_():
                    body(1)

    return gemm.func


# Row-range schedules: ((largest M or None, factory, cfg), ...), the last range unbounded.
# gemm_ws cfg = (BM, BN, BK, stages, warp rows, warp cols, teams, kk-split, refill lag,
#                direct store, paired B loads, uneven warp map or column groups, wide output
#                panels, team rotation, split first chunk, cluster split-K size).
# gemm_mw cfg = (BM, BK, stages, refill lag, ((BN, column tiles, warp layout, paired B), x2)[, s0wait]).
DEFAULT_CFG = (128, 128, 32, 4, 2, 4, 1, 0, 1)
DEFAULT = ((None, gemm_ws, DEFAULT_CFG),)
# Direct-store kernel for a D that is not 16-byte aligned.
FALLBACK = (128, 128, 32, 4, 2, 4, 1, 0, 1, 1)
# 128x200 warp map: four 32-row groups x column groups of 13 and 12 n fragments.
W200 = tuple((w % 4 * 2, 2, 0 if w < 4 else 13, 13 if w < 4 else 12) for w in range(8))
# 128x336 warp map: 64x88 and 64x80 warp tiles; each sub-partition (warp % 4) gets one of each.
W336 = (
    (0, 4, 0, 11),
    (4, 4, 0, 11),
    (0, 4, 21, 11),
    (4, 4, 21, 11),
    (4, 4, 11, 10),
    (0, 4, 11, 10),
    (4, 4, 32, 10),
    (0, 4, 32, 10),
)
# 128x304: 64x80 and 64x72 warp tiles, paired the same way.
W304 = (
    (0, 4, 0, 10),
    (4, 4, 0, 10),
    (0, 4, 19, 10),
    (4, 4, 19, 10),
    (4, 4, 10, 9),
    (0, 4, 10, 9),
    (4, 4, 29, 9),
    (0, 4, 29, 9),
)
LAG0 = (128, 128, 32, 4, 2, 4, 1, 0, 0)
S3 = (128, 128, 32, 3, 2, 4, 1, 0, 1)
W42 = (128, 128, 32, 4, 4, 2, 1, 0, 1)
N160 = (128, 160, 32, 4, 2, 4, 1, 0, 1)
N192 = (128, 192, 32, 4, 2, 4, 1, 0, 1)
N256 = (128, 256, 32, 4, 2, 4, 1, 0, 1)
N256S3 = (128, 256, 32, 3, 2, 4, 1, 0, 1)
M256 = (256, 128, 32, 4, 4, 2, 1, 0, 1)
M256S3 = (256, 128, 32, 3, 4, 2, 1, 0, 1)
T32X64 = (32, 64, 64, 4, 1, 2, 4, 1, 0)
T48X128 = (48, 128, 64, 4, 1, 4, 2, 1, 0)
T48X128T4 = (48, 128, 64, 4, 1, 2, 4, 1, 0)
T64X64 = (64, 64, 64, 4, 1, 2, 4, 1, 0)
T64X64S5 = (64, 64, 64, 5, 1, 2, 4, 1, 0)
T64X64W22 = (64, 64, 64, 4, 2, 2, 4, 1, 0)
T64X128 = (64, 128, 64, 3, 1, 4, 2, 1, 0)
T64X128T4 = (64, 128, 64, 4, 1, 2, 4, 1, 0)
T80X64 = (80, 64, 64, 5, 1, 2, 4, 1, 0)
T80X80 = (80, 80, 64, 3, 1, 2, 4, 1, 0)
T80X128 = (80, 128, 64, 3, 1, 4, 2, 1, 0)
T80X160 = (80, 160, 64, 2, 1, 4, 2, 1, 0)
T80X160S3 = (80, 160, 64, 3, 1, 2, 4, 1, 0)
T80X240 = (80, 240, 64, 2, 1, 4, 2, 1, 0, 0, 0, (8, 8, 7, 7), 0, 2, 1)
T96X128 = (96, 128, 64, 3, 1, 2, 4, 1, 0)
T112X128 = (112, 128, 64, 3, 1, 4, 2, 1, 0)
T128X64T2 = (128, 64, 64, 4, 4, 2, 2, 1, 0)
T128X64T4 = (128, 64, 64, 4, 4, 1, 4, 1, 0)
T128X128T2 = (128, 128, 64, 3, 2, 2, 2, 1, 0)
W42N64 = (128, 64, 32, 4, 4, 2, 1, 0, 1)
C128X16K2 = (128, 16, 64, 4, 4, 1, 1, 0, 1, 0, 0, None, 0, 0, 0, 2)
C128X32K2 = (128, 32, 64, 4, 4, 1, 1, 0, 1, 0, 0, None, 0, 0, 0, 2)
C128X48K4 = (128, 48, 64, 4, 4, 1, 1, 0, 1, 0, 0, None, 0, 0, 0, 4)
C64X128K2 = (64, 128, 64, 4, 2, 2, 1, 0, 1, 0, 0, None, 0, 0, 0, 2)
C64X128K3 = (64, 128, 64, 4, 2, 2, 1, 0, 1, 0, 0, None, 0, 0, 0, 3)
C64X128K4 = (64, 128, 64, 4, 2, 2, 1, 0, 1, 0, 0, None, 0, 0, 0, 4)
MW4096 = (128, 64, 2, 0, ((200, 8, W200, 0), (192, 13, (4, 2), 0)), 1)
MW6144 = (128, 32, 3, 0, ((304, 6, W304, 1), (288, 15, (2, 4), 1)))
MW7168 = (128, 32, 2, 0, ((352, 7, (2, 4), 1), (336, 14, W336, 1)))


def _rows(*ranges):
    return tuple((rows, gemm_ws, cfg) for rows, cfg in ranges)


TABLE = {
    (1024, 2048): _rows(
        (63, T80X80), (128, C128X16K2), (256, T80X80), (640, T64X64W22), (None, T80X80)
    ),
    (1024, 3072): _rows(
        (63, T80X64),
        (128, C128X16K2),
        (256, T80X64),
        (512, T64X64W22),
        (784, T80X64),
        (None, T80X80),
    ),
    (1536, 256): _rows(
        (256, T32X64),
        (640, T48X128),
        (724, DEFAULT_CFG),
        (1024, T80X128),
        (2896, DEFAULT_CFG),
        (None, W42N64),
    ),
    (1536, 2048): _rows(
        (256, T32X64),
        (383, T64X64S5),
        (724, T80X64),
        (1024, T80X128),
        (1280, T80X80),
        (1792, DEFAULT_CFG),
        (3072, T112X128),
        (None, N160),
    ),
    (1536, 4096): _rows(
        (128, C128X32K2),
        (724, T80X64),
        (1024, T80X128),
        (1280, T80X80),
        (1792, DEFAULT_CFG),
        (2048, N160),
        (None, T112X128),
    ),
    (1536, 6144): _rows(
        (128, C128X48K4),
        (159, T80X64),
        (191, C64X128K4),
        (256, C64X128K3),
        (724, T80X64),
        (1024, T80X128),
        (1280, T80X80),
        (1792, DEFAULT_CFG),
        (3072, T80X64),
        (None, N160),
    ),
    (1536, 12288): _rows(
        (128, C128X48K4),
        (159, T80X64),
        (191, C64X128K4),
        (256, C64X128K3),
        (724, T80X64),
        (1024, T80X128),
        (1280, T80X80),
        (1792, DEFAULT_CFG),
        (3072, T80X64),
        (None, N160),
    ),
    (2048, 1024): _rows(
        (192, T32X64),
        (511, T80X64),
        (640, T64X128),
        (724, T80X64),
        (1024, T80X160),
        (1447, DEFAULT_CFG),
        (5792, T80X160),
        (None, N160),
    ),
    (2048, 1536): _rows(
        (192, T32X64),
        (511, T80X64),
        (640, T64X128),
        (784, T80X64),
        (895, T80X80),
        (1024, T80X160),
        (1280, DEFAULT_CFG),
        (1536, N160),
        (None, T80X160),
    ),
    (2048, 2048): _rows((256, T32X64), (640, T128X64T2), (5792, T80X160), (None, T112X128)),
    (2048, 8192): _rows((256, T32X64), (640, T128X64T4), (None, T80X160S3)),
    (2560, 1536): _rows(
        (128, T32X64),
        (383, T80X64),
        (512, T64X128),
        (1448, W42),
        (2048, N256S3),
        (3072, W42),
        (None, M256),
    ),
    (2560, 4096): _rows(
        (383, T80X64),
        (512, T128X64T4),
        (640, T80X64),
        (724, DEFAULT_CFG),
        (1024, LAG0),
        (1536, T80X64),
        (None, LAG0),
    ),
    (2560, 9728): _rows(
        (128, T32X64),
        (383, T80X64),
        (512, T128X64T4),
        (640, T80X64),
        (724, DEFAULT_CFG),
        (1024, LAG0),
        (1536, T80X160S3),
        (None, LAG0),
    ),
    (3072, 2048): _rows((256, T64X64), (512, T80X128), (784, T112X128), (None, T80X240)),
    (4096, 1024): ((256, gemm_ws, T48X128), (512, gemm_ws, T112X128), (None, gemm_mw, MW4096)),
    (4096, 1536): _rows(
        (192, T32X64), (724, T112X128), (1024, T80X160), (1792, DEFAULT_CFG), (None, T80X160)
    ),
    (4096, 4096): _rows(
        (127, DEFAULT_CFG),
        (191, T64X64S5),
        (320, T64X128T4),
        (383, DEFAULT_CFG),
        (512, T112X128),
        (2047, DEFAULT_CFG),
        (None, T80X160S3),
    ),
    (4096, 12288): _rows(
        (127, T112X128),
        (191, C64X128K2),
        (320, T64X128T4),
        (1024, T112X128),
        (1280, N256),
        (None, T80X160S3),
    ),
    (4096, 14336): _rows(
        (127, T112X128),
        (191, C64X128K2),
        (320, T64X128T4),
        (1024, T112X128),
        (1280, N256),
        (None, T80X160S3),
    ),
    (5120, 1024): _rows(
        (256, T48X128),
        (724, LAG0),
        (1024, N256S3),
        (2047, DEFAULT_CFG),
        (4096, N256S3),
        (None, DEFAULT_CFG),
    ),
    (5120, 1536): _rows((256, T48X128), (724, LAG0), (1024, M256S3), (1536, LAG0), (None, M256)),
    (5120, 8192): _rows(
        (127, DEFAULT_CFG),
        (191, C64X128K2),
        (256, T64X128T4),
        (383, DEFAULT_CFG),
        (767, T128X128T2),
        (None, DEFAULT_CFG),
    ),
    (5120, 25600): _rows(
        (127, DEFAULT_CFG),
        (191, C64X128K2),
        (256, T64X128T4),
        (383, DEFAULT_CFG),
        (767, T128X128T2),
        (None, DEFAULT_CFG),
    ),
    (6144, 1024): ((256, gemm_ws, T48X128), (512, gemm_ws, N160), (None, gemm_mw, MW6144)),
    (6144, 2560): _rows(
        (191, T48X128),
        (257, T96X128),
        (447, DEFAULT_CFG),
        (512, N160),
        (1280, DEFAULT_CFG),
        (1792, M256),
        (None, T80X240),
    ),
    (6144, 4096): _rows(
        (127, DEFAULT_CFG),
        (191, T48X128T4),
        (257, T96X128),
        (447, DEFAULT_CFG),
        (512, N160),
        (1534, DEFAULT_CFG),
        (None, T80X160S3),
    ),
    (7168, 1024): ((256, gemm_ws, T48X128), (512, gemm_ws, N192), (None, gemm_mw, MW7168)),
    (8960, 1536): _rows(
        (128, T64X128),
        (256, DEFAULT_CFG),
        (512, N256S3),
        (1023, DEFAULT_CFG),
        (1280, T80X240),
        (1792, DEFAULT_CFG),
        (2048, S3),
        (None, N256),
    ),
    (12288, 1536): _rows(
        (256, T80X160S3),
        (384, N256),
        (1023, DEFAULT_CFG),
        (2048, T80X240),
        (2896, N256S3),
        (4096, S3),
        (None, N256S3),
    ),
    (16384, 2048): _rows((63, DEFAULT_CFG), (256, LAG0), (1447, DEFAULT_CFG), (None, N160)),
    (19456, 2560): _rows(
        (128, LAG0),
        (256, N256),
        (384, N192),
        (895, N256S3),
        (1280, N192),
        (2047, N256S3),
        (2896, N192),
        (None, N256S3),
    ),
    (24576, 1536): _rows((511, N160), (725, S3), (768, DEFAULT_CFG), (None, N256)),
    (24576, 4096): _rows((768, N160), (None, M256)),
    (28672, 4096): _rows((384, DEFAULT_CFG), (None, M256S3)),
    (51200, 5120): _rows((384, DEFAULT_CFG), (None, M256S3)),
}

TABLE_BF16 = {
    (2048, 8192): _rows((256, T32X64), (640, T128X64T4), (5792, T80X160S3), (None, T112X128)),
    (8960, 1536): _rows(
        (128, T64X128),
        (256, DEFAULT_CFG),
        (512, N256S3),
        (1023, DEFAULT_CFG),
        (1280, T80X240),
        (1792, DEFAULT_CFG),
        (2048, S3),
        (5792, N256),
        (None, S3),
    ),
}

# Rows up to which narrow untuned outputs borrow the (1024, 3072) schedule.
NARROW_ROWS = {1536: 1280, 2048: 1024, 2560: 640}


def schedule(n, k, dtype="float16"):
    """The row-range dispatch for (n, k) and ``dtype``.

    Untuned shapes run DEFAULT; narrow ones first borrow the (1024, 3072) schedule, whose
    80-row tiles keep more SMs busy, up to the rows in NARROW_ROWS.
    """
    if dtype == "bfloat16" and (n, k) in TABLE_BF16:
        return TABLE_BF16[(n, k)]
    if (n, k) in TABLE:
        return TABLE[(n, k)]
    limit = NARROW_ROWS.get(n)
    borrowed = TABLE[(1024, 3072)]
    if limit is None or not all(_fits(cfg, k) for _, _, cfg in borrowed):
        return DEFAULT
    ranges = []
    for rows, make, cfg in borrowed:
        top = limit if rows is None else min(rows, limit)
        ranges.append((top, make, cfg))
        if top == limit:
            break
    return (*ranges, *DEFAULT)


def _fits(cfg, k):
    """Whether gemm_ws can run ``cfg`` over a reduction of ``k`` (its pipeline needs at least
    ``stages`` chunks; K=256 has too few for the borrowed 5-stage schedule)."""
    bk, stages, teams = cfg[2], cfg[3], cfg[6]
    split = cfg[7] if len(cfg) > 7 else 0
    chunks = k // bk
    if k % bk or chunks < stages:
        return False
    return bk % (16 * teams) == 0 if split else chunks % teams == 0 and stages % teams == 0


def config_for(M, N, K, *, dtype="float16"):
    """The (factory, cfg) that ``schedule(N, K, dtype)`` runs for ``M`` rows."""
    for limit, make, cfg in schedule(N, K, dtype):
        if limit is None or M <= limit:
            return make, cfg
    raise AssertionError("a schedule ends with an unbounded row range")


def get_kernel(dtype, M, N, K, *, config=None, direct_store=False, backend_config=None):
    """The PrimFunc computing ``D[M, N] = A[M, K] @ B[N, K].T`` for ``dtype`` ("float16" or
    "bfloat16").

    ``M`` may be a symbolic int64 expression. ``config`` is a (factory, cfg) pair from
    ``schedule(N, K, dtype)``; without one, ``M`` must be an int and ``config_for`` picks the
    entry for it. ``direct_store`` builds ``FALLBACK`` for a D that is only 4-byte aligned.
    """
    if direct_store:
        if config is not None:
            raise ValueError("direct_store selects FALLBACK; pass it without a config")
        config = (gemm_ws, FALLBACK)
    elif config is None:
        if not isinstance(M, int):
            raise ValueError("a symbolic M needs an explicit config; see schedule(N, K)")
        config = config_for(M, N, K, dtype=dtype)
    make, cfg = config
    return make(M, cfg, n=N, k=K, dtype=dtype, backend_config=backend_config)
