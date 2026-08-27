# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Source-shaped SM100 blk64 BSA producer.

The shared arena is one-dimensional.  Every swizzle, stage and alias below is
expressed by scalar byte/element arithmetic; no first-class layout is used.
"""

import math

import tirx_kernels.kern as K

M = 64
N = 256
D = 128
WARPS = 16
KV_STAGES = 3
STAGES = 2
TMEM_COLS = 512
LOG2_E = math.log2(math.e)
LN2 = math.log(2.0)
NEG_INF = -float("inf")

ID_QK = 0x04400490
ID_PV = 0x04410490
MMA_WS_F16 = "tcgen05.mma.ws.cta_group::1.kind::f16"
TCGEN_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
TMEM_LD32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TMEM_ST32 = "tcgen05.st.sync.aligned.32x32b.x32.b32"
TMEM_ST16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
TMA_G2S_4D = (
    "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
)
TMA_G2S_5D = (
    "cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
)
TMA_S2G_4D = "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint"
TMA_CACHE = K.uint64(0)


def _load_i32(buffer, index):
    out = K.local_scalar("int32")
    K.ptx.ld.global_.s32(out, buffer.ptr_to([index]))
    return out


def _ld_shared_f32(buffer, index):
    out = K.local_scalar("float32")
    K.ptx.ld.shared.f32(out, buffer.ptr_to([index]))
    return out


def _st_shared_f32(buffer, index, value):
    K.ptx.st.shared.f32(buffer.ptr_to([index]), value)


def _exp2(value):
    out = K.local_scalar("float32")
    K.ptx.ex2.approx.ftz.f32(out, value)
    return out


def _log2(value):
    out = K.local_scalar("float32")
    K.ptx.lg2.approx.ftz.f32(out, value)
    return out


def _rcp(value):
    out = K.local_scalar("float32")
    K.ptx.rcp.approx.ftz.f32(out, value)
    return out


def _packed(op, dst, base, a, b, scale0, scale1, add0=None, add1=None):
    lhs = K.local_scalar("uint64")
    rhs = K.local_scalar("uint64")
    result = K.local_scalar("uint64")
    K.ptx.mov.b64(lhs, a, b)
    K.ptx.mov.b64(rhs, scale0, scale1)
    if add0 is None:
        K.ptx[op](result, lhs, rhs)
    else:
        addend = K.local_scalar("uint64")
        K.ptx.mov.b64(addend, add0, add1)
        K.ptx[op](result, lhs, rhs, addend)
    K.ptx.mov.b64(dst[base], dst[base + 1], result)


def _max3(dst, a, b, c):
    K.ptx.max.f32(dst, a, b, c)


def _reduce_max_128(values):
    acc = K.alloc_local((4,), "float32")
    K.ptx.max.f32(acc[0], values[0], values[1])
    K.ptx.max.f32(acc[1], values[2], values[3])
    K.ptx.max.f32(acc[2], values[4], values[5])
    K.ptx.max.f32(acc[3], values[6], values[7])
    with K.unroll(1, 16) as group:
        base = group * 8
        _max3(acc[0], acc[0], values[base], values[base + 1])
        _max3(acc[1], acc[1], values[base + 2], values[base + 3])
        _max3(acc[2], acc[2], values[base + 4], values[base + 5])
        _max3(acc[3], acc[3], values[base + 6], values[base + 7])
    K.ptx.max.f32(acc[0], acc[0], acc[1])
    _max3(acc[0], acc[0], acc[2], acc[3])
    return acc[0]


def _reduce_sum_128(values, old_sum, old_scale, first):
    acc = K.alloc_local((8,), "float32")
    with K.unroll(8) as j:
        K.assign(acc[j], values[j])
    if not first:
        scaled = K.local_scalar("float32")
        K.ptx.mul.f32(scaled, old_sum, old_scale)
        K.assign(acc[0], acc[0] + scaled)
    with K.unroll(1, 16) as group:
        base = group * 8
        with K.unroll(4) as pair:
            _packed(
                "add.rn.f32x2",
                acc,
                pair * 2,
                acc[pair * 2],
                acc[pair * 2 + 1],
                values[base + pair * 2],
                values[base + pair * 2 + 1],
            )
    for lo, hi in ((0, 2), (4, 6), (0, 4)):
        _packed("add.rn.f32x2", acc, lo, acc[lo], acc[lo + 1], acc[hi], acc[hi + 1])
    return acc[0] + acc[1]


def _apply_mask64(values, base, block_size):
    with K.If(block_size < 64), K.Then():
        with K.unroll(2) as half:
            shift = K.max((half + 1) * 32 - block_size, 0)
            mask = K.local_scalar("uint32")
            K.ptx.shr.u32(mask, K.uint32(0xFFFFFFFF), K.cast(shift, "uint32"))
            with K.If(mask != K.uint32(0xFFFFFFFF)):
                with K.Then():
                    with K.If(mask == 0):
                        with K.Then():
                            with K.unroll(32) as bit:
                                K.assign(values[base + half * 32 + bit], K.float32(NEG_INF))
                        with K.Else():
                            with K.unroll(32) as bit:
                                bit_mask = K.shift_left(K.uint32(1), K.cast(bit, "uint32"))
                                live = K.bitwise_and(mask, bit_mask) != 0
                                K.assign(
                                    values[base + half * 32 + bit],
                                    K.if_then_else(
                                        live, values[base + half * 32 + bit], K.float32(NEG_INF)
                                    ),
                                )


def _tmem_load32(dst, base, address):
    K.ptx[TMEM_LD32](*(dst[base + i] for i in range(32)), K.cast(address, "uint32"))


def _tmem_store16(src, base, address):
    K.ptx[TMEM_ST16](K.cast(address, "uint32"), *(src[base + i] for i in range(16)))


def _tmem_rescale(address, scale):
    regs = K.alloc_local((32,), "float32")
    with K.unroll(4) as chunk:
        _tmem_load32(regs, 0, address + chunk * 32)
        with K.unroll(16) as pair:
            _packed(
                "mul.rn.f32x2", regs, pair * 2, regs[pair * 2], regs[pair * 2 + 1], scale, scale
            )
        K.ptx[TMEM_ST32](K.cast(address + chunk * 32, "uint32"), *(regs[i] for i in range(32)))
    K.ptx.tcgen05.wait__st.sync.aligned()


def _mbar_arrive_wait(barrier, stage, phase):
    barrier.arrive(stage)
    _wait(barrier, stage, phase)


def _wait(barrier, stage, phase):
    K.cuda.mbarrier_wait(barrier.ptr_to([stage]), phase)


def _stats_arrive(stage, warp):
    K.ptx.bar.arrive(K.cast(3 + stage * 4 + warp, "uint32"), K.uint32(64))


def _stats_sync(stage, warp):
    K.ptx.bar.sync(K.cast(3 + stage * 4 + warp, "uint32"), K.uint32(64))


def _query_cancel_response(response_buffer, q_block, head, batch_idx, work_valid):
    response = K.local_scalar("uint128")
    canceled = K.local_scalar("uint32")
    K.ptx.ld.acquire.cta.shared.b128(response, K.address_of(response_buffer[0]))
    K.ptx.clusterlaunchcontrol.query_cancel.is_canceled.pred.b128(canceled, response)
    K.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__x.b32.b128(
        q_block, response, pred=canceled
    )
    K.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__y.b32.b128(
        head, response, pred=canceled
    )
    K.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__z.b32.b128(
        batch_idx, response, pred=canceled
    )
    K.assign(work_valid, K.cast(canceled, "int32"))
    K.ptx.fence.proxy.async_.shared__cta()


def _exchange_store(exchange, warp, lane, tmem0, tmem1, scale0, scale1, zero):
    a = K.alloc_local((32,), "float32")
    b = K.alloc_local((32,), "float32")
    out = K.alloc_local((32,), "float32")
    with K.unroll(4) as chunk:
        if zero:
            with K.unroll(32) as j:
                K.assign(out[j], K.float32(0.0))
        else:
            _tmem_load32(a, 0, tmem0 + chunk * 32)
            _tmem_load32(b, 0, tmem1 + chunk * 32)
            with K.unroll(16) as pair:
                _packed("mul.rn.f32x2", out, pair * 2, b[pair * 2], b[pair * 2 + 1], scale1, scale1)
                _packed(
                    "fma.rn.f32x2",
                    out,
                    pair * 2,
                    a[pair * 2],
                    a[pair * 2 + 1],
                    scale0,
                    scale0,
                    out[pair * 2],
                    out[pair * 2 + 1],
                )
        with K.unroll(8) as group:
            off = warp * 4096 + chunk * 1024 + lane * 4 + group * 128
            K.ptx.st.shared.v4.f32(
                exchange.ptr_to([off]),
                out[group * 4],
                out[group * 4 + 1],
                out[group * 4 + 2],
                out[group * 4 + 3],
            )


def _exchange_reduce_store(exchange, o_raw, corr_warp, lane, split_output):
    partner = corr_warp ^ 2
    a = K.alloc_local((32,), "float32")
    b = K.alloc_local((32,), "float32")
    summed = K.alloc_local((32,), "float32")
    row = (corr_warp & 1) * 32 + lane
    lane_swizzle = lane & 7
    with K.unroll(4) as chunk:
        with K.unroll(8) as group:
            own = corr_warp * 4096 + chunk * 1024 + lane * 4 + group * 128
            peer = partner * 4096 + chunk * 1024 + lane * 4 + group * 128
            K.ptx.ld.shared.v4.f32(
                a[group * 4],
                a[group * 4 + 1],
                a[group * 4 + 2],
                a[group * 4 + 3],
                exchange.ptr_to([own]),
            )
            K.ptx.ld.shared.v4.f32(
                b[group * 4],
                b[group * 4 + 1],
                b[group * 4 + 2],
                b[group * 4 + 3],
                exchange.ptr_to([peer]),
            )
        with K.unroll(16) as pair:
            _packed(
                "add.rn.f32x2",
                summed,
                pair * 2,
                a[pair * 2],
                a[pair * 2 + 1],
                b[pair * 2],
                b[pair * 2 + 1],
            )
        if split_output:
            with K.unroll(8) as group:
                col = ((chunk * 8 + group) ^ lane_swizzle) * 4
                elem = 32 * row + 2048 * (col >> 5) + (col & 31)
                K.ptx.st.shared.v4.f32(
                    o_raw.ptr_to([elem]),
                    summed[group * 4],
                    summed[group * 4 + 1],
                    summed[group * 4 + 2],
                    summed[group * 4 + 3],
                )
        else:
            packed = K.alloc_local((16,), "uint32")
            with K.unroll(16) as pair:
                K.ptx.cvt.rn.satfinite.bf16x2.f32(
                    packed[pair], summed[pair * 2 + 1], summed[pair * 2]
                )
            with K.unroll(4) as group:
                col = ((chunk * 4 + group) ^ lane_swizzle) * 8
                elem = 64 * row + 4096 * (col >> 6) + (col & 63)
                K.ptx.st.shared.v4.u32(
                    o_raw.ptr_to([elem]),
                    packed[group * 4],
                    packed[group * 4 + 1],
                    packed[group * 4 + 2],
                    packed[group * 4 + 3],
                )


def _resolve_splits(value):
    return 2 if value == "auto" else int(value)


def make_forward_kernel(**config):
    batch = int(config["batch"])
    heads = int(config["num_heads"])
    seqlen_q = int(config["seqlen_q"])
    seqlen_kv = int(config["seqlen_kv"])
    max_blocks = int(config["kv_blocks"])
    splits = _resolve_splits(config["kv_splits"])
    use_clc = bool(config["use_clc"])
    has_sizes = bool(config["has_block_sizes"])
    has_nums = config["block_count_mode"] != "fixed"
    allow_empty = config["block_count_mode"] == "variable_empty"
    split_output = splits > 1
    fixed_unsplit = not has_nums and not split_output
    q_blocks = (seqlen_q + 63) // 64
    grid = (q_blocks, heads if use_clc else heads * splits, batch)

    @K.kernel(warps=WARPS, arch="sm_100a", min_blocks_per_sm=1, grid=grid)
    def forward(
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        o_map: K.TensorMap,
        lse: K.gptr[K.f32],
        block_index: K.gptr[K.i32],
        block_sizes: K.gptr[K.i32],
        block_nums: K.gptr[K.i32],
        split_offsets: K.gptr[K.i32],
        softmax_scale_log2: K.f32,
    ):
        if use_clc:
            # CLC requires cluster-launch semantics, but its work coordinates
            # are the global CTA ids.  The source launches singleton clusters;
            # keep that explicit contract independently from the grid shape.
            K.cta_id_in_cluster([1, 1, 1])
            initial_q_block, initial_head, initial_batch = K.cta_id([q_blocks, heads, batch])
            initial_split = K.int32(0)
        else:
            initial_q_block, initial_head_split, initial_batch = K.cta_id(
                [q_blocks, heads * splits, batch]
            )
            initial_split = initial_head_split // heads
            initial_head = initial_head_split - initial_split * heads

        q_block = K.local_scalar("int32", init=initial_q_block)
        head = K.local_scalar("int32", init=initial_head)
        batch_idx = K.local_scalar("int32", init=initial_batch)
        split = K.local_scalar("int32", init=initial_split)
        work_valid = K.local_scalar("int32", init=1)
        clc_consumer_phase = K.local_scalar("int32", init=0)

        warp = K.warp_id()
        lane = K.thread_id() & 31
        tid = K.thread_id()

        arena = K.alloc_buffer((217088,), K.u8, scope="shared.dyn", align=1024)
        smem = K.smem_pool(base=arena)
        pool = smem.pool
        # Exact generated SharedStorage prefix, in declaration order.
        q_full = K.TMABar(pool, 1)
        q_empty = K.TCGen05Bar(pool, 1)
        kv_full = K.TMABar(pool, 3)
        kv_empty = K.TCGen05Bar(pool, 3)
        spo_full = K.TCGen05Bar(pool, 2)
        spo_empty = K.MBarrier(pool, 2)
        plast_full = K.MBarrier(pool, 2)
        plast_empty = K.TCGen05Bar(pool, 2)
        oacc_full = K.TCGen05Bar(pool, 2)
        oacc_empty = K.MBarrier(pool, 2)
        stats_full = K.MBarrier(pool, 2)
        stats_empty = K.MBarrier(pool, 2)
        oepi_full = K.MBarrier(pool, 2)
        oepi_empty = K.MBarrier(pool, 2)
        tmem_mailbox = pool.alloc((1,), "uint32", align=4)
        pool.alloc((4,), "uint8")
        reduce_bar = K.MBarrier(pool, 2, leader=(warp == 15) & (K.cuda.elect_sync() != K.uint32(0)))
        stats_smem = pool.alloc((512,), "float32", align=8)
        pair_smem = pool.alloc((256,), "float32", align=8)
        if use_clc:
            clc_full = K.TMABar(pool, 1)
            clc_empty = K.MBarrier(pool, 1)
            pool.alloc((8,), "uint8")
            clc_response = pool.alloc((4,), "uint32", align=16)
        pool.alloc((4096 - pool.offset,), "uint8")
        q_smem = pool.alloc((8192,), "bfloat16", align=1024)
        kv_smem = pool.alloc((98304,), "bfloat16", align=1024)
        assert pool.offset == 217088

        exchange = K.decl_buffer(
            (16384,),
            "float32",
            data=kv_smem.data,
            byte_offset=20480,
            scope="shared.dyn",
            align=1024,
        )
        if split_output:
            o_smem = K.decl_buffer(
                (16384,),
                "float32",
                data=kv_smem.data,
                byte_offset=86016,
                scope="shared.dyn",
                align=1024,
            )
        else:
            o_smem = K.decl_buffer(
                (16384,),
                "bfloat16",
                data=kv_smem.data,
                byte_offset=86016,
                scope="shared.dyn",
                align=1024,
            )

        # The source initializes each protocol from thread 0, except the two
        # correction reduction barriers (warp 15's elected lane).
        q_full.init(1)
        q_empty.init(1)
        kv_full.init(1)
        kv_empty.init(1)
        spo_full.init(1)
        spo_empty.init(256)
        plast_full.init(4)
        plast_empty.init(1)
        oacc_full.init(1)
        oacc_empty.init(128)
        stats_full.init(128)
        stats_empty.init(128)
        oepi_full.init(128)
        oepi_empty.init(32)
        reduce_bar.init(64)

        with K.If(warp == 0), K.Then():
            with K.If(K.cuda.elect_sync()), K.Then():
                K.ptx.prefetch.tensormap(K.address_of(q_map))
                K.ptx.prefetch.tensormap(K.address_of(k_map))
                K.ptx.prefetch.tensormap(K.address_of(v_map))
                K.ptx.prefetch.tensormap(K.address_of(o_map))

        K.ptx.fence.mbarrier_init.release.cluster()
        if use_clc:
            clc_full.init(1)
            clc_empty.init(512)
            K.ptx.fence.mbarrier_init.release.cluster()
            K.cuda.cta_sync()
        K.cuda.cta_sync()

        raw_count = K.local_scalar("int32")
        split_start = K.local_scalar("int32", init=0)
        count_index = (batch_idx * heads + head) * q_blocks + q_block

        def load_tile_metadata():
            K.assign(split_start, 0)
            if split_output:
                split_base = count_index * (splits + 1)
                K.assign(split_start, _load_i32(split_offsets, split_base + split))
                split_end = _load_i32(split_offsets, split_base + split + 1)
                K.assign(raw_count, split_end - split_start)
            elif has_nums:
                K.assign(raw_count, _load_i32(block_nums, count_index))
            else:
                K.assign(raw_count, K.int32(max_blocks))

        def advance_work():
            if use_clc:
                K.cuda.cta_sync()
                _wait(clc_full, 0, clc_consumer_phase)
                _query_cancel_response(clc_response, q_block, head, batch_idx, work_valid)
                clc_empty.arrive(0, remote=0, pred=K.bool(True), count=1)
                K.assign(clc_consumer_phase, clc_consumer_phase ^ 1)
            else:
                K.assign(work_valid, 0)

        has_work = raw_count > 0 if allow_empty else K.bool(True)
        n_iter = ((raw_count + 7) & -8) // 4

        def sparse_id(logical):
            clamped = K.min(logical, K.max(raw_count - 1, 0))
            return _load_i32(block_index, count_index * max_blocks + split_start + clamped)

        sp = K.specialize(chain_dispatch=True)
        r_softmax = sp.role("softmax", warps=range(0, 8), regs=192 if fixed_unsplit else 184)
        r_correction = sp.role("correction", warps=range(8, 12), regs=88)
        r_mma = sp.role("mma", warps=[12], regs=40 if fixed_unsplit else 48)
        r_epilogue = sp.role("epilogue", warps=[13], regs=40 if fixed_unsplit else 48)
        r_load = sp.role("load", warps=[14], regs=40 if fixed_unsplit else 48)
        r_idle = sp.role("idle", warps=[15], regs=40 if fixed_unsplit else 48)

        # Source dispatch order: idle/CLC, load, MMA, epilogue, softmax,
        # correction. Static and split specializations own one work item.
        with r_idle:
            if use_clc:
                clc_producer_phase = K.local_scalar("int32", init=1)
                with K.While(work_valid != 0):
                    _wait(clc_empty, 0, clc_producer_phase)
                    with K.If(lane == 0), K.Then():
                        clc_full.arrive(0, tx_count=16, remote=0)
                    with K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx[
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        ](K.address_of(clc_response[0]), K.address_of(clc_full.buf[0]))
                    K.assign(clc_producer_phase, clc_producer_phase ^ 1)
                    advance_work()
                _wait(clc_empty, 0, clc_producer_phase)

        with r_load:
            q_prod_phase = K.local_scalar("int32", init=1)
            kv_stage = K.local_scalar("int32", init=0)
            kv_phase = K.local_scalar("int32", init=1)

            def advance_kv():
                K.assign(kv_stage, kv_stage + 1)
                with K.If(kv_stage == 3), K.Then():
                    K.assign(kv_stage, 0)
                    K.assign(kv_phase, kv_phase ^ 1)

            def load_group(kind, reverse_group):
                _wait(kv_empty, kv_stage, kv_phase)
                with K.If(K.cuda.elect_sync()), K.Then():
                    K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        kv_full.ptr_to([kv_stage]), K.uint32(65536)
                    )
                    with K.unroll(4) as sub:
                        sid = sparse_id(reverse_group * 4 + sub)
                        if kind == 0:
                            slot = K.if_then_else(
                                sub == 0,
                                0,
                                K.if_then_else(sub == 1, 2, K.if_then_else(sub == 2, 1, 3)),
                            )
                            with K.unroll(2) as half:
                                dst_elem = kv_stage * 32768 + slot * 4096 + half * 16384
                                K.ptx[TMA_G2S_5D](
                                    kv_smem.ptr_to([dst_elem]),
                                    K.address_of(k_map),
                                    K.int32(0),
                                    K.int32(0),
                                    K.cast(half, "int32"),
                                    sid,
                                    batch_idx * heads + head,
                                    K.cuda.cvta_generic_to_shared(kv_full.ptr_to([kv_stage])),
                                    TMA_CACHE,
                                )
                        else:
                            dst_elem = kv_stage * 32768 + sub * 8192
                            K.ptx[TMA_G2S_5D](
                                kv_smem.ptr_to([dst_elem]),
                                K.address_of(v_map),
                                K.int32(0),
                                K.int32(0),
                                K.int32(0),
                                sid,
                                batch_idx * heads + head,
                                K.cuda.cvta_generic_to_shared(kv_full.ptr_to([kv_stage])),
                                TMA_CACHE,
                            )
                advance_kv()

            with K.While(work_valid != 0):
                load_tile_metadata()
                with K.If(has_work), K.Then():
                    _wait(q_empty, 0, q_prod_phase)
                    with K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                            q_full.ptr_to([0]), K.uint32(16384)
                        )
                        with K.unroll(2) as half:
                            K.ptx[TMA_G2S_4D](
                                q_smem.ptr_to([half * 4096]),
                                K.address_of(q_map),
                                K.cast(half * 64, "int32"),
                                q_block * 64,
                                head,
                                batch_idx,
                                K.cuda.cvta_generic_to_shared(q_full.ptr_to([0])),
                                TMA_CACHE,
                            )
                    K.assign(q_prod_phase, q_prod_phase ^ 1)
                    load_group(0, n_iter - 1)
                    load_group(0, n_iter - 2)
                    i = K.local_scalar("int32", init=0)
                    with K.While(i < n_iter - 2):
                        load_group(1, n_iter - 1 - i)
                        load_group(0, n_iter - 3 - i)
                        K.assign(i, i + 1)
                    load_group(1, 1)
                    load_group(1, 0)
                advance_work()
            _wait(kv_empty, kv_stage, kv_phase)
            _wait(q_empty, 0, q_prod_phase)

        with r_mma:
            K.ptx[TMEM_ALLOC](K.address_of(tmem_mailbox[0]), K.uint32(TMEM_COLS))
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            tmem_base = K.local_scalar("uint32", init=K.uint32(0))
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            q_desc = K.SmemDescriptor()
            q_desc.init(q_smem.ptr_to([0]), ldo=1024, sdo=64, swizzle=3)
            q_desc.make_lo_uniform()
            q_phase = K.local_scalar("int32", init=0)
            kv_stage = K.local_scalar("int32", init=0)
            kv_phase = K.local_scalar("int32", init=0)
            spo_phase0 = K.local_scalar("int32", init=0)
            spo_phase1 = K.local_scalar("int32", init=0)
            acc0 = K.local_scalar("int32", init=0)
            acc1 = K.local_scalar("int32", init=0)

            def advance_kv_cons():
                K.assign(kv_stage, kv_stage + 1)
                with K.If(kv_stage == 3), K.Then():
                    K.assign(kv_stage, 0)
                    K.assign(kv_phase, kv_phase ^ 1)

            def issue_qk(stage):
                k_stage_desc = K.SmemDescriptor()
                k_stage_desc.init(kv_smem.ptr_to([kv_stage * 32768]), ldo=1024, sdo=64, swizzle=3)
                k_stage_desc.make_lo_uniform()
                for ki in range(8):
                    q_off = (ki & 3) * 2 + (ki // 4) * 512
                    k_off = (ki & 3) * 2 + (ki // 4) * 2048
                    with K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx[MMA_WS_F16](
                            K.cast(tmem_base + stage * 128, "uint32"),
                            q_desc.add_16B_offset(q_off),
                            k_stage_desc.add_16B_offset(k_off),
                            K.uint32(ID_QK),
                            K.cast(ki != 0, "bool"),
                            K.uint64(0),
                        )
                with K.If(K.cuda.elect_sync()), K.Then():
                    K.ptx[TCGEN_COMMIT](spo_full.ptr_to([stage]))

            def issue_pv(stage, accumulate, phase):
                v_stage_desc = K.SmemDescriptor()
                v_stage_desc.init(kv_smem.ptr_to([kv_stage * 32768]), ldo=512, sdo=64, swizzle=3)
                v_stage_desc.make_lo_uniform()
                for ki, v_offset in enumerate(
                    (0x000, 0x080, 0x100, 0x180, 0x800, 0x880, 0x900, 0x980)
                ):
                    if ki == 2:
                        _wait(plast_full, stage, phase)
                    with K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx[MMA_WS_F16](
                            K.uint32(256 + stage * 128),
                            K.uint32(stage * 128 + ki * 8),
                            v_stage_desc.add_16B_offset(v_offset),
                            K.uint32(ID_PV),
                            K.cast(accumulate != 0 if ki == 0 else True, "bool"),
                            K.uint64(0),
                        )

            with K.While(work_valid != 0):
                load_tile_metadata()
                K.assign(acc0, 0)
                K.assign(acc1, 0)
                with K.If(has_work), K.Then():
                    _wait(q_full, 0, q_phase)
                    K.ptx.tcgen05.fence__after_thread_sync()
                    with K.unroll(2) as stage:
                        _wait(kv_full, kv_stage, kv_phase)
                        K.ptx.tcgen05.fence__after_thread_sync()
                        issue_qk(stage)
                        with K.If(K.cuda.elect_sync()), K.Then():
                            K.ptx[TCGEN_COMMIT](kv_empty.ptr_to([kv_stage]))
                        advance_kv_cons()

                    pairs = (n_iter - 2) // 2
                    pair = K.local_scalar("int32", init=0)
                    with K.While(pair < pairs):
                        for stage in range(2):
                            phase = K.if_then_else(stage == 0, spo_phase0, spo_phase1)
                            _wait(spo_empty, stage, phase)
                            _wait(kv_full, kv_stage, kv_phase)
                            K.ptx.tcgen05.fence__after_thread_sync()
                            issue_pv(stage, K.if_then_else(stage == 0, acc0, acc1), phase)
                            with K.If(K.cuda.elect_sync()), K.Then():
                                K.ptx[TCGEN_COMMIT](kv_empty.ptr_to([kv_stage]))
                            advance_kv_cons()
                            _wait(kv_full, kv_stage, kv_phase)
                            K.ptx.tcgen05.fence__after_thread_sync()
                            issue_qk(stage)
                            with K.If(K.cuda.elect_sync()), K.Then():
                                K.ptx[TCGEN_COMMIT](kv_empty.ptr_to([kv_stage]))
                            advance_kv_cons()
                            if stage == 0:
                                K.assign(spo_phase0, spo_phase0 ^ 1)
                                K.assign(acc0, 1)
                            else:
                                K.assign(spo_phase1, spo_phase1 ^ 1)
                                K.assign(acc1, 1)
                        K.assign(pair, pair + 1)
                    with K.If(K.cuda.elect_sync()), K.Then():
                        K.ptx[TCGEN_COMMIT](q_empty.ptr_to([0]))

                    for stage in range(2):
                        phase = K.if_then_else(stage == 0, spo_phase0, spo_phase1)
                        _wait(spo_empty, stage, phase)
                        _wait(kv_full, kv_stage, kv_phase)
                        K.ptx.tcgen05.fence__after_thread_sync()
                        issue_pv(stage, K.if_then_else(stage == 0, acc0, acc1), phase)
                        with K.If(K.cuda.elect_sync()), K.Then():
                            K.ptx[TCGEN_COMMIT](oacc_full.ptr_to([stage]))
                            K.ptx[TCGEN_COMMIT](kv_empty.ptr_to([kv_stage]))
                        advance_kv_cons()
                        if stage == 0:
                            K.assign(spo_phase0, spo_phase0 ^ 1)
                        else:
                            K.assign(spo_phase1, spo_phase1 ^ 1)
                    K.assign(q_phase, q_phase ^ 1)
                advance_work()

            K.ptx[TMEM_RELINQUISH]()
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            allocated = K.local_scalar("uint32")
            K.ptx.ld.shared.u32(allocated, tmem_mailbox.ptr_to([0]))
            K.ptx[TMEM_DEALLOC](allocated, K.uint32(TMEM_COLS))

        with r_epilogue:
            oepi_phase = K.local_scalar("int32", init=0)
            with K.While(work_valid != 0):
                load_tile_metadata()
                _wait(oepi_full, 0, oepi_phase)
                with K.If(K.cuda.elect_sync()), K.Then():
                    out_head = split * heads + head if split_output else head
                    if split_output:
                        with K.unroll(4) as quarter:
                            K.ptx[TMA_S2G_4D](
                                K.address_of(o_map),
                                K.cast(quarter * 32, "int32"),
                                q_block * 64,
                                out_head,
                                batch_idx,
                                o_smem.ptr_to([quarter * 2048]),
                                TMA_CACHE,
                            )
                    else:
                        with K.unroll(2) as half:
                            K.ptx[TMA_S2G_4D](
                                K.address_of(o_map),
                                K.cast(half * 64, "int32"),
                                q_block * 64,
                                out_head,
                                batch_idx,
                                o_smem.ptr_to([half * 4096]),
                                TMA_CACHE,
                            )
                    K.ptx.cp.async_.bulk.commit_group()
                K.ptx.cp.async_.bulk.wait_group.read(0)
                oepi_empty.arrive(0)
                K.assign(oepi_phase, oepi_phase ^ 1)
                advance_work()

        with r_softmax:
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            tmem_base = K.local_scalar("uint32", init=K.uint32(0))
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            stage = K.if_then_else(warp < 4, 0, 1)
            local_warp = warp & 3
            tid128 = tid & 127
            score_phase = K.local_scalar("int32", init=0)
            stats_phase = K.local_scalar("int32", init=1)
            row_max = K.local_scalar("float32", init=K.float32(NEG_INF))
            row_sum = K.local_scalar("float32", init=K.float32(0.0))
            with K.While(work_valid != 0):
                load_tile_metadata()
                K.assign(row_max, K.float32(NEG_INF))
                K.assign(row_sum, K.float32(0.0))
                _wait(stats_empty, stage, stats_phase)
                K.assign(stats_phase, stats_phase ^ 1)

                with K.If(has_work), K.Then():
                    wg_count = n_iter // 2
                    iteration = K.local_scalar("int32", init=0)
                    with K.While(iteration < wg_count):
                        _wait(spo_full, stage, score_phase)
                        score = K.alloc_local((128,), "float32")
                        with K.unroll(4) as chunk:
                            _tmem_load32(score, chunk * 32, tmem_base + stage * 128 + chunk * 32)

                        reverse_group = n_iter - 1 - (iteration * 2 + stage)
                        warp_col = local_warp // 2
                        logical_lo = reverse_group * 4 + warp_col
                        logical_hi = logical_lo + 2
                        bs_lo = K.local_scalar("int32")
                        bs_hi = K.local_scalar("int32")
                        if has_sizes:
                            K.assign(
                                bs_lo,
                                K.if_then_else(
                                    logical_lo < raw_count,
                                    _load_i32(block_sizes, sparse_id(logical_lo)),
                                    0,
                                ),
                            )
                            K.assign(
                                bs_hi,
                                K.if_then_else(
                                    logical_hi < raw_count,
                                    _load_i32(block_sizes, sparse_id(logical_hi)),
                                    0,
                                ),
                            )
                        else:
                            K.assign(bs_lo, K.if_then_else(logical_lo < raw_count, 64, 0))
                            K.assign(bs_hi, K.if_then_else(logical_hi < raw_count, 64, 0))
                        _apply_mask64(score, 0, bs_lo)
                        _apply_mask64(score, 64, bs_hi)

                        tile_max = _reduce_max_128(score)
                        first = iteration == 0
                        old_scale = K.local_scalar("float32")
                        new_max = K.local_scalar("float32")
                        row_max_safe = K.local_scalar("float32")
                        with K.If(first):
                            with K.Then():
                                K.assign(new_max, tile_max)
                                K.assign(
                                    row_max_safe,
                                    K.if_then_else(tile_max != K.float32(NEG_INF), tile_max, 0.0),
                                )
                                K.assign(old_scale, K.float32(0.0))
                            with K.Else():
                                K.ptx.max.f32(new_max, row_max, tile_max)
                                K.assign(
                                    row_max_safe,
                                    K.if_then_else(new_max != K.float32(NEG_INF), new_max, 0.0),
                                )
                                delta = K.local_scalar("float32")
                                K.ptx.sub.f32(delta, row_max, row_max_safe)
                                delta_scaled = K.local_scalar("float32")
                                K.ptx.mul.f32(delta_scaled, delta, softmax_scale_log2)
                                K.assign(old_scale, _exp2(delta_scaled))
                                with K.If(delta_scaled >= K.float32(-8.0)), K.Then():
                                    K.assign(new_max, row_max)
                                    K.assign(row_max_safe, row_max)
                                    K.assign(old_scale, K.float32(1.0))
                        with K.If(first == K.bool(False)), K.Then():
                            _st_shared_f32(stats_smem, stage * 128 + tid128, old_scale)
                        _stats_arrive(stage, local_warp)

                        negative_scale = K.local_scalar("float32")
                        K.ptx.neg.f32(negative_scale, softmax_scale_log2)
                        negative_rowmax = K.local_scalar("float32")
                        K.ptx.mul.f32(negative_rowmax, row_max_safe, negative_scale)
                        sum_acc = K.alloc_local((8,), "float32")
                        scaled_old_sum = K.local_scalar("float32")
                        K.ptx.mul.f32(scaled_old_sum, row_sum, old_scale)
                        K.assign(sum_acc[0], scaled_old_sum)
                        with K.unroll(1, 8) as acc_idx:
                            K.assign(sum_acc[acc_idx], K.float32(0.0))
                        packed_p = K.alloc_local((16,), "uint32")
                        for chunk in range(4):
                            for subgroup in range(4):
                                with K.unroll(4) as acc_pair:
                                    pair = subgroup * 4 + acc_pair
                                    score_base = chunk * 32 + pair * 2
                                    scaled = K.alloc_local((2,), "float32")
                                    _packed(
                                        "fma.rn.f32x2",
                                        scaled,
                                        0,
                                        score[score_base],
                                        score[score_base + 1],
                                        softmax_scale_log2,
                                        softmax_scale_log2,
                                        negative_rowmax,
                                        negative_rowmax,
                                    )
                                    exp0 = _exp2(scaled[0])
                                    exp1 = _exp2(scaled[1])
                                    K.ptx.cvt.rn.satfinite.bf16x2.f32(packed_p[pair], exp1, exp0)
                                    _packed(
                                        "add.rn.f32x2",
                                        sum_acc,
                                        acc_pair * 2,
                                        sum_acc[acc_pair * 2],
                                        sum_acc[acc_pair * 2 + 1],
                                        exp0,
                                        exp1,
                                    )
                            _tmem_store16(packed_p, 0, tmem_base + stage * 128 + chunk * 16)
                            if chunk == 0:
                                K.ptx.tcgen05.wait__st.sync.aligned()
                                spo_empty.arrive(stage)
                        K.ptx.tcgen05.wait__st.sync.aligned()
                        K.cuda.warp_sync()
                        with K.If(K.cuda.elect_sync()), K.Then():
                            plast_full.arrive(stage)
                        for lo, hi in ((0, 2), (4, 6), (0, 4)):
                            _packed(
                                "add.rn.f32x2",
                                sum_acc,
                                lo,
                                sum_acc[lo],
                                sum_acc[lo + 1],
                                sum_acc[hi],
                                sum_acc[hi + 1],
                            )
                        K.assign(row_sum, sum_acc[0] + sum_acc[1])
                        K.assign(row_max, new_max)
                        _wait(stats_empty, stage, stats_phase)
                        K.assign(stats_phase, stats_phase ^ 1)
                        K.assign(score_phase, score_phase ^ 1)
                        K.assign(iteration, iteration + 1)

                    _st_shared_f32(stats_smem, stage * 128 + tid128, row_sum)
                    _st_shared_f32(stats_smem, 256 + stage * 128 + tid128, row_max)
                    _stats_arrive(stage, local_warp)
                with K.If(has_work == K.bool(False)), K.Then():
                    _stats_arrive(stage, local_warp)
                advance_work()

            _wait(stats_empty, stage, stats_phase)
            K.ptx.bar.arrive(K.uint32(2), K.uint32(416))

        with r_correction:
            K.ptx.bar.sync(K.uint32(2), K.uint32(416))
            tmem_base = K.local_scalar("uint32", init=K.uint32(0))
            K.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            corr_warp = warp - 8
            tid128 = tid & 127
            lane_corr = tid & 31
            oacc_phase = K.local_scalar("int32", init=0)
            oepi_phase = K.local_scalar("int32", init=1)
            scale0 = K.local_scalar("float32", init=K.float32(0.0))
            scale1 = K.local_scalar("float32", init=K.float32(0.0))
            sum_local = K.local_scalar("float32", init=K.float32(0.0))
            max_safe = K.local_scalar("float32", init=K.float32(0.0))
            spo_empty.arrive(0)
            spo_empty.arrive(1)
            with K.While(work_valid != 0):
                load_tile_metadata()
                K.assign(scale0, K.float32(0.0))
                K.assign(scale1, K.float32(0.0))
                K.assign(sum_local, K.float32(0.0))
                K.assign(max_safe, K.float32(0.0))
                with K.If(has_work), K.Then():
                    _stats_sync(0, corr_warp)
                    stats_empty.arrive(0)
                    _stats_sync(1, corr_warp)
                    corr_pairs = (n_iter - 2) // 2
                    pair = K.local_scalar("int32", init=0)
                    with K.While(pair < corr_pairs):
                        with K.unroll(2) as stage:
                            _stats_sync(stage, corr_warp)
                            scale = _ld_shared_f32(stats_smem, stage * 128 + tid128)
                            ballot = K.local_scalar("uint32")
                            K.ptx.vote_sync.ballot.b32(
                                ballot, K.ptx.pred(scale < K.float32(1.0)), K.uint32(0xFFFFFFFF)
                            )
                            with K.If(ballot != 0), K.Then():
                                _tmem_rescale(tmem_base + 256 + stage * 128, scale)
                            spo_empty.arrive(stage)
                            stats_empty.arrive(1 - stage)
                        K.assign(pair, pair + 1)
                    stats_empty.arrive(1)

                    sum0 = K.local_scalar("float32", init=K.float32(0.0))
                    sum1 = K.local_scalar("float32", init=K.float32(0.0))
                    maximum0 = K.local_scalar("float32", init=K.float32(NEG_INF))
                    maximum1 = K.local_scalar("float32", init=K.float32(NEG_INF))
                    for stage in range(2):
                        _stats_sync(stage, corr_warp)
                        if stage == 0:
                            K.assign(sum0, _ld_shared_f32(stats_smem, tid128))
                            K.assign(maximum0, _ld_shared_f32(stats_smem, 256 + tid128))
                        else:
                            K.assign(sum1, _ld_shared_f32(stats_smem, 128 + tid128))
                            K.assign(maximum1, _ld_shared_f32(stats_smem, 384 + tid128))
                        stats_empty.arrive(stage)
                    valid0 = sum0 > 0
                    valid1 = sum1 > 0
                    rm0 = K.if_then_else(valid0, maximum0, K.float32(NEG_INF))
                    rm1 = K.if_then_else(valid1, maximum1, K.float32(NEG_INF))
                    max_local = K.local_scalar("float32")
                    K.ptx.max.f32(max_local, rm0, rm1)
                    K.assign(
                        max_safe, K.if_then_else(max_local > K.float32(NEG_INF), max_local, 0.0)
                    )
                    K.assign(
                        scale0,
                        K.if_then_else(valid0, _exp2((rm0 - max_safe) * softmax_scale_log2), 0.0),
                    )
                    K.assign(
                        scale1,
                        K.if_then_else(valid1, _exp2((rm1 - max_safe) * softmax_scale_log2), 0.0),
                    )
                    K.assign(sum_local, sum0 * scale0 + sum1 * scale1)
                    for stage in range(2):
                        _wait(oacc_full, stage, oacc_phase)
                        K.ptx.tcgen05.fence__after_thread_sync()
                    K.ptx.fence.proxy.async_.shared__cta()
                    _wait(oepi_empty, 0, oepi_phase)
                with K.If(has_work == K.bool(False)), K.Then():
                    for stage in range(2):
                        _stats_sync(stage, corr_warp)
                        stats_empty.arrive(stage)
                    K.assign(scale0, K.float32(0.0))
                    K.assign(scale1, K.float32(0.0))
                    K.assign(sum_local, K.float32(0.0))
                    K.assign(max_safe, K.float32(0.0))
                    _wait(oepi_empty, 0, oepi_phase)

                partner = corr_warp ^ 2
                _st_shared_f32(pair_smem, partner * 64 + lane_corr * 2, sum_local)
                _st_shared_f32(pair_smem, partner * 64 + lane_corr * 2 + 1, max_safe)
                _mbar_arrive_wait(reduce_bar, corr_warp & 1, 0)
                peer_sum = _ld_shared_f32(pair_smem, corr_warp * 64 + lane_corr * 2)
                peer_max = _ld_shared_f32(pair_smem, corr_warp * 64 + lane_corr * 2 + 1)
                max_total = K.local_scalar("float32")
                K.ptx.max.f32(max_total, max_safe, peer_max)
                max_total_safe = K.if_then_else(max_total > K.float32(NEG_INF), max_total, 0.0)
                own_rescale = K.if_then_else(
                    sum_local > 0, _exp2((max_safe - max_total_safe) * softmax_scale_log2), 0.0
                )
                peer_rescale = K.if_then_else(
                    peer_sum > 0, _exp2((peer_max - max_total_safe) * softmax_scale_log2), 0.0
                )
                total_sum = sum_local * own_rescale + peer_sum * peer_rescale
                inv_total = K.if_then_else(total_sum > 0, _rcp(total_sum), 0.0)
                own_weight = own_rescale * inv_total
                own_scale0 = scale0 * own_weight
                own_scale1 = scale1 * own_weight
                zero = allow_empty and ((own_scale0 == 0) & (own_scale1 == 0))
                if allow_empty:
                    with K.If(zero):
                        with K.Then():
                            _exchange_store(
                                exchange,
                                corr_warp,
                                lane_corr,
                                tmem_base + 256,
                                tmem_base + 384,
                                own_scale0,
                                own_scale1,
                                True,
                            )
                        with K.Else():
                            _exchange_store(
                                exchange,
                                corr_warp,
                                lane_corr,
                                tmem_base + 256,
                                tmem_base + 384,
                                own_scale0,
                                own_scale1,
                                False,
                            )
                else:
                    _exchange_store(
                        exchange,
                        corr_warp,
                        lane_corr,
                        tmem_base + 256,
                        tmem_base + 384,
                        own_scale0,
                        own_scale1,
                        False,
                    )
                _mbar_arrive_wait(reduce_bar, corr_warp & 1, 1)
                with K.If(corr_warp < 2), K.Then():
                    _exchange_reduce_store(exchange, o_smem, corr_warp, lane_corr, split_output)
                K.ptx.fence.proxy.async_.shared__cta()
                out_row = (corr_warp & 1) * 32 + lane_corr
                with K.If((corr_warp < 2) & (q_block * 64 + out_row < seqlen_q)), K.Then():
                    out_head = split * heads + head if split_output else head
                    lse_index = (
                        (batch_idx * (heads * splits) + out_head) * seqlen_q
                        + q_block * 64
                        + out_row
                    )
                    lse_value = K.if_then_else(
                        total_sum > 0,
                        (max_total_safe * softmax_scale_log2 + _log2(total_sum)) * K.float32(LN2),
                        K.float32(NEG_INF),
                    )
                    K.ptx.st.global_.f32(lse.ptr_to([lse_index]), lse_value)
                with K.If(has_work), K.Then():
                    spo_empty.arrive(0)
                    spo_empty.arrive(1)
                oepi_full.arrive(0)
                K.assign(oepi_phase, oepi_phase ^ 1)
                with K.If(has_work), K.Then():
                    K.assign(oacc_phase, oacc_phase ^ 1)
                advance_work()
            _wait(oepi_empty, 0, oepi_phase)
            K.ptx.bar.arrive(K.uint32(2), K.uint32(416))

    return forward
