# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Source-shaped SM100 blk64 BSA producer.

The shared arena is one-dimensional.  Every swizzle, stage and alias below is
expressed by scalar byte/element arithmetic; no first-class layout is used.
"""

import math

import tirx_kernels.tirx_lite as txl

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
TMA_CACHE = txl.uint64(0)


def _load_i32(buffer, index):
    out = txl.local_scalar("int32")
    txl.ptx.ld.global_.s32(out, buffer.ptr_to([index]))
    return out


def _ld_shared_f32(buffer, index):
    out = txl.local_scalar("float32")
    txl.ptx.ld.shared.f32(out, buffer.ptr_to([index]))
    return out


def _st_shared_f32(buffer, index, value):
    txl.ptx.st.shared.f32(buffer.ptr_to([index]), value)


def _exp2(value):
    out = txl.local_scalar("float32")
    txl.ptx.ex2.approx.ftz.f32(out, value)
    return out


def _log2(value):
    out = txl.local_scalar("float32")
    txl.ptx.lg2.approx.ftz.f32(out, value)
    return out


def _rcp(value):
    out = txl.local_scalar("float32")
    txl.ptx.rcp.approx.ftz.f32(out, value)
    return out


def _packed(op, dst, base, a, b, scale0, scale1, add0=None, add1=None):
    lhs = txl.local_scalar("uint64")
    rhs = txl.local_scalar("uint64")
    result = txl.local_scalar("uint64")
    txl.ptx.mov.b64(lhs, a, b)
    txl.ptx.mov.b64(rhs, scale0, scale1)
    if add0 is None:
        txl.ptx[op](result, lhs, rhs)
    else:
        addend = txl.local_scalar("uint64")
        txl.ptx.mov.b64(addend, add0, add1)
        txl.ptx[op](result, lhs, rhs, addend)
    txl.ptx.mov.b64(dst[base], dst[base + 1], result)


def _max3(dst, a, b, c):
    txl.ptx.max.f32(dst, a, b, c)


def _reduce_max_128(values):
    acc = txl.alloc_local((4,), "float32")
    txl.ptx.max.f32(acc[0], values[0], values[1])
    txl.ptx.max.f32(acc[1], values[2], values[3])
    txl.ptx.max.f32(acc[2], values[4], values[5])
    txl.ptx.max.f32(acc[3], values[6], values[7])
    with txl.unroll(1, 16) as group:
        base = group * 8
        _max3(acc[0], acc[0], values[base], values[base + 1])
        _max3(acc[1], acc[1], values[base + 2], values[base + 3])
        _max3(acc[2], acc[2], values[base + 4], values[base + 5])
        _max3(acc[3], acc[3], values[base + 6], values[base + 7])
    txl.ptx.max.f32(acc[0], acc[0], acc[1])
    _max3(acc[0], acc[0], acc[2], acc[3])
    return acc[0]


def _reduce_sum_128(values, old_sum, old_scale, first):
    acc = txl.alloc_local((8,), "float32")
    with txl.unroll(8) as j:
        txl.assign(acc[j], values[j])
    if not first:
        scaled = txl.local_scalar("float32")
        txl.ptx.mul.f32(scaled, old_sum, old_scale)
        txl.assign(acc[0], acc[0] + scaled)
    with txl.unroll(1, 16) as group:
        base = group * 8
        with txl.unroll(4) as pair:
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
    with txl.If(block_size < 64), txl.Then():
        with txl.unroll(2) as half:
            shift = txl.max((half + 1) * 32 - block_size, 0)
            mask = txl.local_scalar("uint32")
            txl.ptx.shr.u32(mask, txl.uint32(0xFFFFFFFF), txl.cast(shift, "uint32"))
            with txl.If(mask != txl.uint32(0xFFFFFFFF)):
                with txl.Then():
                    with txl.If(mask == 0):
                        with txl.Then():
                            with txl.unroll(32) as bit:
                                txl.assign(values[base + half * 32 + bit], txl.float32(NEG_INF))
                        with txl.Else():
                            with txl.unroll(32) as bit:
                                bit_mask = txl.shift_left(txl.uint32(1), txl.cast(bit, "uint32"))
                                live = txl.bitwise_and(mask, bit_mask) != 0
                                txl.assign(
                                    values[base + half * 32 + bit],
                                    txl.if_then_else(
                                        live, values[base + half * 32 + bit], txl.float32(NEG_INF)
                                    ),
                                )


def _tmem_load32(dst, base, address):
    txl.ptx[TMEM_LD32](*(dst[base + i] for i in range(32)), txl.cast(address, "uint32"))


def _tmem_store16(src, base, address):
    txl.ptx[TMEM_ST16](txl.cast(address, "uint32"), *(src[base + i] for i in range(16)))


def _tmem_rescale(address, scale):
    regs = txl.alloc_local((32,), "float32")
    with txl.unroll(4) as chunk:
        _tmem_load32(regs, 0, address + chunk * 32)
        with txl.unroll(16) as pair:
            _packed(
                "mul.rn.f32x2", regs, pair * 2, regs[pair * 2], regs[pair * 2 + 1], scale, scale
            )
        txl.ptx[TMEM_ST32](txl.cast(address + chunk * 32, "uint32"), *(regs[i] for i in range(32)))
    txl.ptx.tcgen05.wait__st.sync.aligned()


def _mbar_arrive_wait(barrier, stage, phase):
    barrier.arrive(stage)
    _wait(barrier, stage, phase)


def _wait(barrier, stage, phase):
    txl.cuda.mbarrier_wait(barrier.ptr_to([stage]), phase)


def _stats_arrive(stage, warp):
    txl.ptx.bar.arrive(txl.cast(3 + stage * 4 + warp, "uint32"), txl.uint32(64))


def _stats_sync(stage, warp):
    txl.ptx.bar.sync(txl.cast(3 + stage * 4 + warp, "uint32"), txl.uint32(64))


def _query_cancel_response(response_buffer, q_block, head, batch_idx, work_valid):
    response = txl.local_scalar("uint128")
    canceled = txl.local_scalar("uint32")
    txl.ptx.ld.acquire.cta.shared.b128(response, txl.address_of(response_buffer[0]))
    txl.ptx.clusterlaunchcontrol.query_cancel.is_canceled.pred.b128(canceled, response)
    txl.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__x.b32.b128(
        q_block, response, pred=canceled
    )
    txl.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__y.b32.b128(
        head, response, pred=canceled
    )
    txl.ptx.clusterlaunchcontrol.query_cancel.get_first_ctaid__z.b32.b128(
        batch_idx, response, pred=canceled
    )
    txl.assign(work_valid, txl.cast(canceled, "int32"))
    txl.ptx.fence.proxy.async_.shared__cta()


def _exchange_store(exchange, warp, lane, tmem0, tmem1, scale0, scale1, zero):
    a = txl.alloc_local((32,), "float32")
    b = txl.alloc_local((32,), "float32")
    out = txl.alloc_local((32,), "float32")
    with txl.unroll(4) as chunk:
        if zero:
            with txl.unroll(32) as j:
                txl.assign(out[j], txl.float32(0.0))
        else:
            _tmem_load32(a, 0, tmem0 + chunk * 32)
            _tmem_load32(b, 0, tmem1 + chunk * 32)
            with txl.unroll(16) as pair:
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
        with txl.unroll(8) as group:
            off = warp * 4096 + chunk * 1024 + lane * 4 + group * 128
            txl.ptx.st.shared.v4.f32(
                exchange.ptr_to([off]),
                out[group * 4],
                out[group * 4 + 1],
                out[group * 4 + 2],
                out[group * 4 + 3],
            )


def _exchange_reduce_store(exchange, o_raw, corr_warp, lane, split_output):
    partner = corr_warp ^ 2
    a = txl.alloc_local((32,), "float32")
    b = txl.alloc_local((32,), "float32")
    summed = txl.alloc_local((32,), "float32")
    row = (corr_warp & 1) * 32 + lane
    lane_swizzle = lane & 7
    with txl.unroll(4) as chunk:
        with txl.unroll(8) as group:
            own = corr_warp * 4096 + chunk * 1024 + lane * 4 + group * 128
            peer = partner * 4096 + chunk * 1024 + lane * 4 + group * 128
            txl.ptx.ld.shared.v4.f32(
                a[group * 4],
                a[group * 4 + 1],
                a[group * 4 + 2],
                a[group * 4 + 3],
                exchange.ptr_to([own]),
            )
            txl.ptx.ld.shared.v4.f32(
                b[group * 4],
                b[group * 4 + 1],
                b[group * 4 + 2],
                b[group * 4 + 3],
                exchange.ptr_to([peer]),
            )
        with txl.unroll(16) as pair:
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
            with txl.unroll(8) as group:
                col = ((chunk * 8 + group) ^ lane_swizzle) * 4
                elem = 32 * row + 2048 * (col >> 5) + (col & 31)
                txl.ptx.st.shared.v4.f32(
                    o_raw.ptr_to([elem]),
                    summed[group * 4],
                    summed[group * 4 + 1],
                    summed[group * 4 + 2],
                    summed[group * 4 + 3],
                )
        else:
            packed = txl.alloc_local((16,), "uint32")
            with txl.unroll(16) as pair:
                txl.ptx.cvt.rn.satfinite.bf16x2.f32(
                    packed[pair], summed[pair * 2 + 1], summed[pair * 2]
                )
            with txl.unroll(4) as group:
                col = ((chunk * 4 + group) ^ lane_swizzle) * 8
                elem = 64 * row + 4096 * (col >> 6) + (col & 63)
                txl.ptx.st.shared.v4.u32(
                    o_raw.ptr_to([elem]),
                    packed[group * 4],
                    packed[group * 4 + 1],
                    packed[group * 4 + 2],
                    packed[group * 4 + 3],
                )


def _kv_head(head, ratio):
    """Map a Q head onto the KV head whose tile it reads.

    ``head`` comes from the grid or the CLC response and is never negative, so
    the division is done unsigned. That matters: on the split-KV and CLC paths
    the source's head index is a signed value of unproven sign and its division
    costs ten instructions, while the unsigned form stays at one.
    """
    if ratio == 1:
        return head
    if ratio & (ratio - 1) == 0:
        return txl.cast(
            txl.shift_right(txl.cast(head, "uint32"), txl.uint32(ratio.bit_length() - 1)), "int32"
        )
    return txl.cast(txl.cast(head, "uint32") // txl.uint32(ratio), "int32")


def _resolve_splits(value):
    return 2 if value == "auto" else int(value)


def make_forward_kernel(**config):
    batch = int(config["batch"])
    heads = int(config["num_q_heads"])
    kv_heads = int(config["num_kv_heads"])
    if heads % kv_heads != 0:
        raise ValueError("num_q_heads must be a multiple of num_kv_heads")
    head_ratio = heads // kv_heads
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

    @txl.kernel(warps=WARPS, arch="sm_100a", min_blocks_per_sm=1, grid=grid)
    def forward(
        q_map: txl.TensorMap,
        k_map: txl.TensorMap,
        v_map: txl.TensorMap,
        o_map: txl.TensorMap,
        lse: txl.gptr[txl.f32],
        block_index: txl.gptr[txl.i32],
        block_sizes: txl.gptr[txl.i32],
        block_nums: txl.gptr[txl.i32],
        split_offsets: txl.gptr[txl.i32],
        softmax_scale_log2: txl.f32,
    ):
        if use_clc:
            # CLC requires cluster-launch semantics, but its work coordinates
            # are the global CTA ids.  The source launches singleton clusters;
            # keep that explicit contract independently from the grid shape.
            txl.cta_id_in_cluster([1, 1, 1])
            initial_q_block, initial_head, initial_batch = txl.cta_id([q_blocks, heads, batch])
            initial_split = txl.int32(0)
        else:
            initial_q_block, initial_head_split, initial_batch = txl.cta_id(
                [q_blocks, heads * splits, batch]
            )
            initial_split = initial_head_split // heads
            initial_head = initial_head_split - initial_split * heads

        q_block = txl.local_scalar("int32", init=initial_q_block)
        head = txl.local_scalar("int32", init=initial_head)
        batch_idx = txl.local_scalar("int32", init=initial_batch)
        split = txl.local_scalar("int32", init=initial_split)
        work_valid = txl.local_scalar("int32", init=1)
        clc_consumer_phase = txl.local_scalar("int32", init=0)

        warp = txl.warp_id()
        lane = txl.thread_id() & 31
        tid = txl.thread_id()

        arena = txl.alloc_buffer((217088,), txl.u8, scope="shared.dyn", align=1024)
        smem = txl.smem_pool(base=arena)
        pool = smem.pool
        # Exact generated SharedStorage prefix, in declaration order.
        q_full = txl.TMABar(pool, 1)
        q_empty = txl.TCGen05Bar(pool, 1)
        kv_full = txl.TMABar(pool, 3)
        kv_empty = txl.TCGen05Bar(pool, 3)
        spo_full = txl.TCGen05Bar(pool, 2)
        spo_empty = txl.MBarrier(pool, 2)
        plast_full = txl.MBarrier(pool, 2)
        plast_empty = txl.TCGen05Bar(pool, 2)
        oacc_full = txl.TCGen05Bar(pool, 2)
        oacc_empty = txl.MBarrier(pool, 2)
        stats_full = txl.MBarrier(pool, 2)
        stats_empty = txl.MBarrier(pool, 2)
        oepi_full = txl.MBarrier(pool, 2)
        oepi_empty = txl.MBarrier(pool, 2)
        tmem_mailbox = pool.alloc((1,), "uint32", align=4)
        pool.alloc((4,), "uint8")
        reduce_bar = txl.MBarrier(pool, 2, leader=(warp == 15) & (txl.cuda.elect_sync() != txl.uint32(0)))
        stats_smem = pool.alloc((512,), "float32", align=8)
        pair_smem = pool.alloc((256,), "float32", align=8)
        if use_clc:
            clc_full = txl.TMABar(pool, 1)
            clc_empty = txl.MBarrier(pool, 1)
            pool.alloc((8,), "uint8")
            clc_response = pool.alloc((4,), "uint32", align=16)
        pool.alloc((4096 - pool.offset,), "uint8")
        q_smem = pool.alloc((8192,), "bfloat16", align=1024)
        kv_smem = pool.alloc((98304,), "bfloat16", align=1024)
        assert pool.offset == 217088

        exchange = txl.decl_buffer(
            (16384,),
            "float32",
            data=kv_smem.data,
            byte_offset=20480,
            scope="shared.dyn",
            align=1024,
        )
        if split_output:
            o_smem = txl.decl_buffer(
                (16384,),
                "float32",
                data=kv_smem.data,
                byte_offset=86016,
                scope="shared.dyn",
                align=1024,
            )
        else:
            o_smem = txl.decl_buffer(
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

        with txl.If(warp == 0), txl.Then():
            with txl.If(txl.cuda.elect_sync()), txl.Then():
                txl.ptx.prefetch.tensormap(txl.address_of(q_map))
                txl.ptx.prefetch.tensormap(txl.address_of(k_map))
                txl.ptx.prefetch.tensormap(txl.address_of(v_map))
                txl.ptx.prefetch.tensormap(txl.address_of(o_map))

        txl.ptx.fence.mbarrier_init.release.cluster()
        if use_clc:
            clc_full.init(1)
            clc_empty.init(512)
            txl.ptx.fence.mbarrier_init.release.cluster()
            txl.cuda.cta_sync()
        txl.cuda.cta_sync()

        raw_count = txl.local_scalar("int32")
        split_start = txl.local_scalar("int32", init=0)
        count_index = (batch_idx * heads + head) * q_blocks + q_block
        # The K/V TensorMap fuses (batch, KV head) into its outermost dimension,
        # so a Q head selects its group's KV tile here. Both operands read the
        # live work-item scalars, which is what a persistent CLC producer needs:
        # the coordinate follows `head` as the scheduler advances it.
        kv_slot = batch_idx * kv_heads + _kv_head(head, head_ratio)

        def load_tile_metadata():
            txl.assign(split_start, 0)
            if split_output:
                split_base = count_index * (splits + 1)
                txl.assign(split_start, _load_i32(split_offsets, split_base + split))
                split_end = _load_i32(split_offsets, split_base + split + 1)
                txl.assign(raw_count, split_end - split_start)
            elif has_nums:
                txl.assign(raw_count, _load_i32(block_nums, count_index))
            else:
                txl.assign(raw_count, txl.int32(max_blocks))

        def advance_work():
            if use_clc:
                txl.cuda.cta_sync()
                _wait(clc_full, 0, clc_consumer_phase)
                _query_cancel_response(clc_response, q_block, head, batch_idx, work_valid)
                clc_empty.arrive(0, remote=0, pred=txl.bool(True), count=1)
                txl.assign(clc_consumer_phase, clc_consumer_phase ^ 1)
            else:
                txl.assign(work_valid, 0)

        has_work = raw_count > 0 if allow_empty else txl.bool(True)
        n_iter = ((raw_count + 7) & -8) // 4

        def sparse_id(logical):
            clamped = txl.min(logical, txl.max(raw_count - 1, 0))
            return _load_i32(block_index, count_index * max_blocks + split_start + clamped)

        sp = txl.specialize(chain_dispatch=True)
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
                clc_producer_phase = txl.local_scalar("int32", init=1)
                with txl.While(work_valid != 0):
                    _wait(clc_empty, 0, clc_producer_phase)
                    with txl.If(lane == 0), txl.Then():
                        clc_full.arrive(0, tx_count=16, remote=0)
                    with txl.If(txl.cuda.elect_sync()), txl.Then():
                        txl.ptx[
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                        ](txl.address_of(clc_response[0]), txl.address_of(clc_full.buf[0]))
                    txl.assign(clc_producer_phase, clc_producer_phase ^ 1)
                    advance_work()
                _wait(clc_empty, 0, clc_producer_phase)

        with r_load:
            q_prod_phase = txl.local_scalar("int32", init=1)
            kv_stage = txl.local_scalar("int32", init=0)
            kv_phase = txl.local_scalar("int32", init=1)

            def advance_kv():
                txl.assign(kv_stage, kv_stage + 1)
                with txl.If(kv_stage == 3), txl.Then():
                    txl.assign(kv_stage, 0)
                    txl.assign(kv_phase, kv_phase ^ 1)

            def load_group(kind, reverse_group):
                _wait(kv_empty, kv_stage, kv_phase)
                with txl.If(txl.cuda.elect_sync()), txl.Then():
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        kv_full.ptr_to([kv_stage]), txl.uint32(65536)
                    )
                    with txl.unroll(4) as sub:
                        sid = sparse_id(reverse_group * 4 + sub)
                        if kind == 0:
                            slot = txl.if_then_else(
                                sub == 0,
                                0,
                                txl.if_then_else(sub == 1, 2, txl.if_then_else(sub == 2, 1, 3)),
                            )
                            with txl.unroll(2) as half:
                                dst_elem = kv_stage * 32768 + slot * 4096 + half * 16384
                                txl.ptx[TMA_G2S_5D](
                                    kv_smem.ptr_to([dst_elem]),
                                    txl.address_of(k_map),
                                    txl.int32(0),
                                    txl.int32(0),
                                    txl.cast(half, "int32"),
                                    sid,
                                    kv_slot,
                                    txl.cuda.cvta_generic_to_shared(kv_full.ptr_to([kv_stage])),
                                    TMA_CACHE,
                                )
                        else:
                            dst_elem = kv_stage * 32768 + sub * 8192
                            txl.ptx[TMA_G2S_5D](
                                kv_smem.ptr_to([dst_elem]),
                                txl.address_of(v_map),
                                txl.int32(0),
                                txl.int32(0),
                                txl.int32(0),
                                sid,
                                kv_slot,
                                txl.cuda.cvta_generic_to_shared(kv_full.ptr_to([kv_stage])),
                                TMA_CACHE,
                            )
                advance_kv()

            with txl.While(work_valid != 0):
                load_tile_metadata()
                with txl.If(has_work), txl.Then():
                    _wait(q_empty, 0, q_prod_phase)
                    with txl.If(txl.cuda.elect_sync()), txl.Then():
                        txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                            q_full.ptr_to([0]), txl.uint32(16384)
                        )
                        with txl.unroll(2) as half:
                            txl.ptx[TMA_G2S_4D](
                                q_smem.ptr_to([half * 4096]),
                                txl.address_of(q_map),
                                txl.cast(half * 64, "int32"),
                                q_block * 64,
                                head,
                                batch_idx,
                                txl.cuda.cvta_generic_to_shared(q_full.ptr_to([0])),
                                TMA_CACHE,
                            )
                    txl.assign(q_prod_phase, q_prod_phase ^ 1)
                    load_group(0, n_iter - 1)
                    load_group(0, n_iter - 2)
                    i = txl.local_scalar("int32", init=0)
                    with txl.While(i < n_iter - 2):
                        load_group(1, n_iter - 1 - i)
                        load_group(0, n_iter - 3 - i)
                        txl.assign(i, i + 1)
                    load_group(1, 1)
                    load_group(1, 0)
                advance_work()
            _wait(kv_empty, kv_stage, kv_phase)
            _wait(q_empty, 0, q_prod_phase)

        with r_mma:
            txl.ptx[TMEM_ALLOC](txl.address_of(tmem_mailbox[0]), txl.uint32(TMEM_COLS))
            txl.ptx.bar.sync(txl.uint32(2), txl.uint32(416))
            tmem_base = txl.local_scalar("uint32", init=txl.uint32(0))
            txl.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            q_desc = txl.SmemDescriptor()
            q_desc.init(q_smem.ptr_to([0]), ldo=1024, sdo=64, swizzle=3)
            q_desc.make_lo_uniform()
            q_phase = txl.local_scalar("int32", init=0)
            kv_stage = txl.local_scalar("int32", init=0)
            kv_phase = txl.local_scalar("int32", init=0)
            spo_phase0 = txl.local_scalar("int32", init=0)
            spo_phase1 = txl.local_scalar("int32", init=0)
            acc0 = txl.local_scalar("int32", init=0)
            acc1 = txl.local_scalar("int32", init=0)

            def advance_kv_cons():
                txl.assign(kv_stage, kv_stage + 1)
                with txl.If(kv_stage == 3), txl.Then():
                    txl.assign(kv_stage, 0)
                    txl.assign(kv_phase, kv_phase ^ 1)

            def issue_qk(stage):
                k_stage_desc = txl.SmemDescriptor()
                k_stage_desc.init(kv_smem.ptr_to([kv_stage * 32768]), ldo=1024, sdo=64, swizzle=3)
                k_stage_desc.make_lo_uniform()
                for ki in range(8):
                    q_off = (ki & 3) * 2 + (ki // 4) * 512
                    k_off = (ki & 3) * 2 + (ki // 4) * 2048
                    with txl.If(txl.cuda.elect_sync()), txl.Then():
                        txl.ptx[MMA_WS_F16](
                            txl.cast(tmem_base + stage * 128, "uint32"),
                            q_desc.add_16B_offset(q_off),
                            k_stage_desc.add_16B_offset(k_off),
                            txl.uint32(ID_QK),
                            txl.cast(ki != 0, "bool"),
                            txl.uint64(0),
                        )
                with txl.If(txl.cuda.elect_sync()), txl.Then():
                    txl.ptx[TCGEN_COMMIT](spo_full.ptr_to([stage]))

            def issue_pv(stage, accumulate, phase):
                v_stage_desc = txl.SmemDescriptor()
                v_stage_desc.init(kv_smem.ptr_to([kv_stage * 32768]), ldo=512, sdo=64, swizzle=3)
                v_stage_desc.make_lo_uniform()
                for ki, v_offset in enumerate(
                    (0x000, 0x080, 0x100, 0x180, 0x800, 0x880, 0x900, 0x980)
                ):
                    if ki == 2:
                        _wait(plast_full, stage, phase)
                    with txl.If(txl.cuda.elect_sync()), txl.Then():
                        txl.ptx[MMA_WS_F16](
                            txl.uint32(256 + stage * 128),
                            txl.uint32(stage * 128 + ki * 8),
                            v_stage_desc.add_16B_offset(v_offset),
                            txl.uint32(ID_PV),
                            txl.cast(accumulate != 0 if ki == 0 else True, "bool"),
                            txl.uint64(0),
                        )

            with txl.While(work_valid != 0):
                load_tile_metadata()
                txl.assign(acc0, 0)
                txl.assign(acc1, 0)
                with txl.If(has_work), txl.Then():
                    _wait(q_full, 0, q_phase)
                    txl.ptx.tcgen05.fence__after_thread_sync()
                    with txl.unroll(2) as stage:
                        _wait(kv_full, kv_stage, kv_phase)
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        issue_qk(stage)
                        with txl.If(txl.cuda.elect_sync()), txl.Then():
                            txl.ptx[TCGEN_COMMIT](kv_empty.ptr_to([kv_stage]))
                        advance_kv_cons()

                    pairs = (n_iter - 2) // 2
                    pair = txl.local_scalar("int32", init=0)
                    with txl.While(pair < pairs):
                        for stage in range(2):
                            phase = txl.if_then_else(stage == 0, spo_phase0, spo_phase1)
                            _wait(spo_empty, stage, phase)
                            _wait(kv_full, kv_stage, kv_phase)
                            txl.ptx.tcgen05.fence__after_thread_sync()
                            issue_pv(stage, txl.if_then_else(stage == 0, acc0, acc1), phase)
                            with txl.If(txl.cuda.elect_sync()), txl.Then():
                                txl.ptx[TCGEN_COMMIT](kv_empty.ptr_to([kv_stage]))
                            advance_kv_cons()
                            _wait(kv_full, kv_stage, kv_phase)
                            txl.ptx.tcgen05.fence__after_thread_sync()
                            issue_qk(stage)
                            with txl.If(txl.cuda.elect_sync()), txl.Then():
                                txl.ptx[TCGEN_COMMIT](kv_empty.ptr_to([kv_stage]))
                            advance_kv_cons()
                            if stage == 0:
                                txl.assign(spo_phase0, spo_phase0 ^ 1)
                                txl.assign(acc0, 1)
                            else:
                                txl.assign(spo_phase1, spo_phase1 ^ 1)
                                txl.assign(acc1, 1)
                        txl.assign(pair, pair + 1)
                    with txl.If(txl.cuda.elect_sync()), txl.Then():
                        txl.ptx[TCGEN_COMMIT](q_empty.ptr_to([0]))

                    for stage in range(2):
                        phase = txl.if_then_else(stage == 0, spo_phase0, spo_phase1)
                        _wait(spo_empty, stage, phase)
                        _wait(kv_full, kv_stage, kv_phase)
                        txl.ptx.tcgen05.fence__after_thread_sync()
                        issue_pv(stage, txl.if_then_else(stage == 0, acc0, acc1), phase)
                        with txl.If(txl.cuda.elect_sync()), txl.Then():
                            txl.ptx[TCGEN_COMMIT](oacc_full.ptr_to([stage]))
                            txl.ptx[TCGEN_COMMIT](kv_empty.ptr_to([kv_stage]))
                        advance_kv_cons()
                        if stage == 0:
                            txl.assign(spo_phase0, spo_phase0 ^ 1)
                        else:
                            txl.assign(spo_phase1, spo_phase1 ^ 1)
                    txl.assign(q_phase, q_phase ^ 1)
                advance_work()

            txl.ptx[TMEM_RELINQUISH]()
            txl.ptx.bar.sync(txl.uint32(2), txl.uint32(416))
            allocated = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(allocated, tmem_mailbox.ptr_to([0]))
            txl.ptx[TMEM_DEALLOC](allocated, txl.uint32(TMEM_COLS))

        with r_epilogue:
            oepi_phase = txl.local_scalar("int32", init=0)
            with txl.While(work_valid != 0):
                load_tile_metadata()
                _wait(oepi_full, 0, oepi_phase)
                with txl.If(txl.cuda.elect_sync()), txl.Then():
                    out_head = split * heads + head if split_output else head
                    if split_output:
                        with txl.unroll(4) as quarter:
                            txl.ptx[TMA_S2G_4D](
                                txl.address_of(o_map),
                                txl.cast(quarter * 32, "int32"),
                                q_block * 64,
                                out_head,
                                batch_idx,
                                o_smem.ptr_to([quarter * 2048]),
                                TMA_CACHE,
                            )
                    else:
                        with txl.unroll(2) as half:
                            txl.ptx[TMA_S2G_4D](
                                txl.address_of(o_map),
                                txl.cast(half * 64, "int32"),
                                q_block * 64,
                                out_head,
                                batch_idx,
                                o_smem.ptr_to([half * 4096]),
                                TMA_CACHE,
                            )
                    txl.ptx.cp.async_.bulk.commit_group()
                txl.ptx.cp.async_.bulk.wait_group.read(0)
                oepi_empty.arrive(0)
                txl.assign(oepi_phase, oepi_phase ^ 1)
                advance_work()

        with r_softmax:
            txl.ptx.bar.sync(txl.uint32(2), txl.uint32(416))
            tmem_base = txl.local_scalar("uint32", init=txl.uint32(0))
            txl.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            stage = txl.if_then_else(warp < 4, 0, 1)
            local_warp = warp & 3
            tid128 = tid & 127
            score_phase = txl.local_scalar("int32", init=0)
            stats_phase = txl.local_scalar("int32", init=1)
            row_max = txl.local_scalar("float32", init=txl.float32(NEG_INF))
            row_sum = txl.local_scalar("float32", init=txl.float32(0.0))
            with txl.While(work_valid != 0):
                load_tile_metadata()
                txl.assign(row_max, txl.float32(NEG_INF))
                txl.assign(row_sum, txl.float32(0.0))
                _wait(stats_empty, stage, stats_phase)
                txl.assign(stats_phase, stats_phase ^ 1)

                with txl.If(has_work), txl.Then():
                    wg_count = n_iter // 2
                    iteration = txl.local_scalar("int32", init=0)
                    with txl.While(iteration < wg_count):
                        _wait(spo_full, stage, score_phase)
                        score = txl.alloc_local((128,), "float32")
                        with txl.unroll(4) as chunk:
                            _tmem_load32(score, chunk * 32, tmem_base + stage * 128 + chunk * 32)

                        reverse_group = n_iter - 1 - (iteration * 2 + stage)
                        warp_col = local_warp // 2
                        logical_lo = reverse_group * 4 + warp_col
                        logical_hi = logical_lo + 2
                        bs_lo = txl.local_scalar("int32")
                        bs_hi = txl.local_scalar("int32")
                        if has_sizes:
                            txl.assign(
                                bs_lo,
                                txl.if_then_else(
                                    logical_lo < raw_count,
                                    _load_i32(block_sizes, sparse_id(logical_lo)),
                                    0,
                                ),
                            )
                            txl.assign(
                                bs_hi,
                                txl.if_then_else(
                                    logical_hi < raw_count,
                                    _load_i32(block_sizes, sparse_id(logical_hi)),
                                    0,
                                ),
                            )
                        else:
                            txl.assign(bs_lo, txl.if_then_else(logical_lo < raw_count, 64, 0))
                            txl.assign(bs_hi, txl.if_then_else(logical_hi < raw_count, 64, 0))
                        _apply_mask64(score, 0, bs_lo)
                        _apply_mask64(score, 64, bs_hi)

                        tile_max = _reduce_max_128(score)
                        first = iteration == 0
                        old_scale = txl.local_scalar("float32")
                        new_max = txl.local_scalar("float32")
                        row_max_safe = txl.local_scalar("float32")
                        with txl.If(first):
                            with txl.Then():
                                txl.assign(new_max, tile_max)
                                txl.assign(
                                    row_max_safe,
                                    txl.if_then_else(tile_max != txl.float32(NEG_INF), tile_max, 0.0),
                                )
                                txl.assign(old_scale, txl.float32(0.0))
                            with txl.Else():
                                txl.ptx.max.f32(new_max, row_max, tile_max)
                                txl.assign(
                                    row_max_safe,
                                    txl.if_then_else(new_max != txl.float32(NEG_INF), new_max, 0.0),
                                )
                                delta = txl.local_scalar("float32")
                                txl.ptx.sub.f32(delta, row_max, row_max_safe)
                                delta_scaled = txl.local_scalar("float32")
                                txl.ptx.mul.f32(delta_scaled, delta, softmax_scale_log2)
                                txl.assign(old_scale, _exp2(delta_scaled))
                                with txl.If(delta_scaled >= txl.float32(-8.0)), txl.Then():
                                    txl.assign(new_max, row_max)
                                    txl.assign(row_max_safe, row_max)
                                    txl.assign(old_scale, txl.float32(1.0))
                        with txl.If(first == txl.bool(False)), txl.Then():
                            _st_shared_f32(stats_smem, stage * 128 + tid128, old_scale)
                        _stats_arrive(stage, local_warp)

                        negative_scale = txl.local_scalar("float32")
                        txl.ptx.neg.f32(negative_scale, softmax_scale_log2)
                        negative_rowmax = txl.local_scalar("float32")
                        txl.ptx.mul.f32(negative_rowmax, row_max_safe, negative_scale)
                        sum_acc = txl.alloc_local((8,), "float32")
                        scaled_old_sum = txl.local_scalar("float32")
                        txl.ptx.mul.f32(scaled_old_sum, row_sum, old_scale)
                        txl.assign(sum_acc[0], scaled_old_sum)
                        with txl.unroll(1, 8) as acc_idx:
                            txl.assign(sum_acc[acc_idx], txl.float32(0.0))
                        packed_p = txl.alloc_local((16,), "uint32")
                        for chunk in range(4):
                            for subgroup in range(4):
                                with txl.unroll(4) as acc_pair:
                                    pair = subgroup * 4 + acc_pair
                                    score_base = chunk * 32 + pair * 2
                                    scaled = txl.alloc_local((2,), "float32")
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
                                    txl.ptx.cvt.rn.satfinite.bf16x2.f32(packed_p[pair], exp1, exp0)
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
                                txl.ptx.tcgen05.wait__st.sync.aligned()
                                spo_empty.arrive(stage)
                        txl.ptx.tcgen05.wait__st.sync.aligned()
                        txl.cuda.warp_sync()
                        with txl.If(txl.cuda.elect_sync()), txl.Then():
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
                        txl.assign(row_sum, sum_acc[0] + sum_acc[1])
                        txl.assign(row_max, new_max)
                        _wait(stats_empty, stage, stats_phase)
                        txl.assign(stats_phase, stats_phase ^ 1)
                        txl.assign(score_phase, score_phase ^ 1)
                        txl.assign(iteration, iteration + 1)

                    _st_shared_f32(stats_smem, stage * 128 + tid128, row_sum)
                    _st_shared_f32(stats_smem, 256 + stage * 128 + tid128, row_max)
                    _stats_arrive(stage, local_warp)
                with txl.If(has_work == txl.bool(False)), txl.Then():
                    _stats_arrive(stage, local_warp)
                advance_work()

            _wait(stats_empty, stage, stats_phase)
            txl.ptx.bar.arrive(txl.uint32(2), txl.uint32(416))

        with r_correction:
            txl.ptx.bar.sync(txl.uint32(2), txl.uint32(416))
            tmem_base = txl.local_scalar("uint32", init=txl.uint32(0))
            txl.ptx.ld.shared.u32(tmem_base, tmem_mailbox.ptr_to([0]))
            corr_warp = warp - 8
            tid128 = tid & 127
            lane_corr = tid & 31
            oacc_phase = txl.local_scalar("int32", init=0)
            oepi_phase = txl.local_scalar("int32", init=1)
            scale0 = txl.local_scalar("float32", init=txl.float32(0.0))
            scale1 = txl.local_scalar("float32", init=txl.float32(0.0))
            sum_local = txl.local_scalar("float32", init=txl.float32(0.0))
            max_safe = txl.local_scalar("float32", init=txl.float32(0.0))
            spo_empty.arrive(0)
            spo_empty.arrive(1)
            with txl.While(work_valid != 0):
                load_tile_metadata()
                txl.assign(scale0, txl.float32(0.0))
                txl.assign(scale1, txl.float32(0.0))
                txl.assign(sum_local, txl.float32(0.0))
                txl.assign(max_safe, txl.float32(0.0))
                with txl.If(has_work), txl.Then():
                    _stats_sync(0, corr_warp)
                    stats_empty.arrive(0)
                    _stats_sync(1, corr_warp)
                    corr_pairs = (n_iter - 2) // 2
                    pair = txl.local_scalar("int32", init=0)
                    with txl.While(pair < corr_pairs):
                        with txl.unroll(2) as stage:
                            _stats_sync(stage, corr_warp)
                            scale = _ld_shared_f32(stats_smem, stage * 128 + tid128)
                            ballot = txl.local_scalar("uint32")
                            txl.ptx.vote_sync.ballot.b32(
                                ballot, txl.ptx.pred(scale < txl.float32(1.0)), txl.uint32(0xFFFFFFFF)
                            )
                            with txl.If(ballot != 0), txl.Then():
                                _tmem_rescale(tmem_base + 256 + stage * 128, scale)
                            spo_empty.arrive(stage)
                            stats_empty.arrive(1 - stage)
                        txl.assign(pair, pair + 1)
                    stats_empty.arrive(1)

                    sum0 = txl.local_scalar("float32", init=txl.float32(0.0))
                    sum1 = txl.local_scalar("float32", init=txl.float32(0.0))
                    maximum0 = txl.local_scalar("float32", init=txl.float32(NEG_INF))
                    maximum1 = txl.local_scalar("float32", init=txl.float32(NEG_INF))
                    for stage in range(2):
                        _stats_sync(stage, corr_warp)
                        if stage == 0:
                            txl.assign(sum0, _ld_shared_f32(stats_smem, tid128))
                            txl.assign(maximum0, _ld_shared_f32(stats_smem, 256 + tid128))
                        else:
                            txl.assign(sum1, _ld_shared_f32(stats_smem, 128 + tid128))
                            txl.assign(maximum1, _ld_shared_f32(stats_smem, 384 + tid128))
                        stats_empty.arrive(stage)
                    valid0 = sum0 > 0
                    valid1 = sum1 > 0
                    rm0 = txl.if_then_else(valid0, maximum0, txl.float32(NEG_INF))
                    rm1 = txl.if_then_else(valid1, maximum1, txl.float32(NEG_INF))
                    max_local = txl.local_scalar("float32")
                    txl.ptx.max.f32(max_local, rm0, rm1)
                    txl.assign(
                        max_safe, txl.if_then_else(max_local > txl.float32(NEG_INF), max_local, 0.0)
                    )
                    txl.assign(
                        scale0,
                        txl.if_then_else(valid0, _exp2((rm0 - max_safe) * softmax_scale_log2), 0.0),
                    )
                    txl.assign(
                        scale1,
                        txl.if_then_else(valid1, _exp2((rm1 - max_safe) * softmax_scale_log2), 0.0),
                    )
                    txl.assign(sum_local, sum0 * scale0 + sum1 * scale1)
                    for stage in range(2):
                        _wait(oacc_full, stage, oacc_phase)
                        txl.ptx.tcgen05.fence__after_thread_sync()
                    txl.ptx.fence.proxy.async_.shared__cta()
                    _wait(oepi_empty, 0, oepi_phase)
                with txl.If(has_work == txl.bool(False)), txl.Then():
                    for stage in range(2):
                        _stats_sync(stage, corr_warp)
                        stats_empty.arrive(stage)
                    txl.assign(scale0, txl.float32(0.0))
                    txl.assign(scale1, txl.float32(0.0))
                    txl.assign(sum_local, txl.float32(0.0))
                    txl.assign(max_safe, txl.float32(0.0))
                    _wait(oepi_empty, 0, oepi_phase)

                partner = corr_warp ^ 2
                _st_shared_f32(pair_smem, partner * 64 + lane_corr * 2, sum_local)
                _st_shared_f32(pair_smem, partner * 64 + lane_corr * 2 + 1, max_safe)
                _mbar_arrive_wait(reduce_bar, corr_warp & 1, 0)
                peer_sum = _ld_shared_f32(pair_smem, corr_warp * 64 + lane_corr * 2)
                peer_max = _ld_shared_f32(pair_smem, corr_warp * 64 + lane_corr * 2 + 1)
                max_total = txl.local_scalar("float32")
                txl.ptx.max.f32(max_total, max_safe, peer_max)
                max_total_safe = txl.if_then_else(max_total > txl.float32(NEG_INF), max_total, 0.0)
                own_rescale = txl.if_then_else(
                    sum_local > 0, _exp2((max_safe - max_total_safe) * softmax_scale_log2), 0.0
                )
                peer_rescale = txl.if_then_else(
                    peer_sum > 0, _exp2((peer_max - max_total_safe) * softmax_scale_log2), 0.0
                )
                total_sum = sum_local * own_rescale + peer_sum * peer_rescale
                inv_total = txl.if_then_else(total_sum > 0, _rcp(total_sum), 0.0)
                own_weight = own_rescale * inv_total
                own_scale0 = scale0 * own_weight
                own_scale1 = scale1 * own_weight
                zero = allow_empty and ((own_scale0 == 0) & (own_scale1 == 0))
                if allow_empty:
                    with txl.If(zero):
                        with txl.Then():
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
                        with txl.Else():
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
                with txl.If(corr_warp < 2), txl.Then():
                    _exchange_reduce_store(exchange, o_smem, corr_warp, lane_corr, split_output)
                txl.ptx.fence.proxy.async_.shared__cta()
                out_row = (corr_warp & 1) * 32 + lane_corr
                with txl.If((corr_warp < 2) & (q_block * 64 + out_row < seqlen_q)), txl.Then():
                    out_head = split * heads + head if split_output else head
                    lse_index = (
                        (batch_idx * (heads * splits) + out_head) * seqlen_q
                        + q_block * 64
                        + out_row
                    )
                    lse_value = txl.if_then_else(
                        total_sum > 0,
                        (max_total_safe * softmax_scale_log2 + _log2(total_sum)) * txl.float32(LN2),
                        txl.float32(NEG_INF),
                    )
                    txl.ptx.st.global_.f32(lse.ptr_to([lse_index]), lse_value)
                with txl.If(has_work), txl.Then():
                    spo_empty.arrive(0)
                    spo_empty.arrive(1)
                oepi_full.arrive(0)
                txl.assign(oepi_phase, oepi_phase ^ 1)
                with txl.If(has_work), txl.Then():
                    txl.assign(oacc_phase, oacc_phase ^ 1)
                advance_work()
            _wait(oepi_empty, 0, oepi_phase)
            txl.ptx.bar.arrive(txl.uint32(2), txl.uint32(416))

    return forward
