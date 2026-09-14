# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Source-shaped SM100 blk64 block-sparse-attention backward kernels."""

import tirx_kernels.tirx_lite as txl

WARPS = 16
THREADS = 512
TMEM_COLUMNS = 512
BAR_CTA = (1, 512)
BAR_TMEM = (2, 416)
BAR_COMPUTE = (3, 256)
BAR_DEALLOC = (4, 256)
BAR_REDUCE = (5, 128)

OFF_SK = 1024
OFF_SV = 17408
OFF_SQ = 33792
OFF_SP = 99328
OFF_SDO = 115712
OFF_SDS = 148480
OFF_SDQ = 164864
OFF_SLSE = 197632
OFF_SSUM = 198656
SHARED_BYTES = 199680

TMEM_DK = 0
TMEM_DV = 64
TMEM_DQ = 128
TMEM_DP = 128
TMEM_S = 256

ID_QK = 0x08100490
ID_DKDV = 0x08118490
ID_DQ = 0x08210490

MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
TMEM_LD16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
TMEM_LD32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TMA_G2S = (
    "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
)
TMA_REDUCE = "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
TMA_CACHE = txl.uint64(0)


def _xor(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return a ^ b
    return txl.bitwise_xor(a, b)


def _tile_elem(row, dim):
    return (dim // 64) * 4096 + _xor(row * 64 + dim % 64, (row % 8) * 8)


def _tile_byte(base, row, dim):
    return base + 2 * _tile_elem(row, dim)


def _desc_base(ldo, sdo=64, swizzle=3):
    arrangement = {0: 0, 1: 6, 2: 4, 3: 2, 4: 1}[swizzle]
    value = ((ldo & 0x3FFF) << 16) | ((sdo & 0x3FFF) << 32) | (1 << 46)
    return (value | ((arrangement & 0x7) << 61)) & 0xFFFFFFFFFFFFFFFF


def _desc_at(base, shared_address):
    field = txl.cast(
        txl.bitwise_and(txl.shift_right(shared_address, txl.uint32(4)), txl.uint32(0x3FFF)), "uint64"
    )
    return txl.bitwise_or(txl.uint64(base), field)


def _desc_add16(desc, offset):
    if offset == 0:
        return desc
    lo = txl.local_scalar("uint32")
    hi = txl.local_scalar("uint32")
    out = txl.local_scalar("uint64")
    txl.ptx.mov.b64(lo, hi, desc)
    txl.ptx.add.u32(lo, lo, txl.uint32(offset))
    txl.ptx.mov.b64(out, lo, hi)
    return out


def _mma(dest, a_desc, b_desc, idesc, accumulate, pred):
    txl.ptx[MMA_F16](
        txl.cast(dest, "uint32"),
        a_desc,
        b_desc,
        txl.uint32(idesc),
        txl.uint32(0),
        txl.uint32(0),
        txl.uint32(0),
        txl.uint32(0),
        txl.ptx.pred(accumulate),
        pred=pred,
    )


def _mma_chain(dest, a_desc, b_desc, idesc, a_offsets, b_offsets, accumulate, pred):
    flag = txl.local_scalar("uint32", init=txl.cast(accumulate, "uint32"))
    for a_offset, b_offset in zip(a_offsets, b_offsets):
        _mma(dest, _desc_add16(a_desc, a_offset), _desc_add16(b_desc, b_offset), idesc, flag, pred)
        txl.assign(flag, txl.uint32(1))


def _shfl_bfly_f32(value, lane_xor, membermask):
    out = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.bfly.b32(
        out, txl.reinterpret(txl.u32, value), txl.uint32(lane_xor), txl.uint32(31), membermask
    )
    return txl.reinterpret(txl.f32, out)


def _butterfly_sum_f32(value):
    membermask = txl.local_scalar("uint32", init=txl.tvm_warp_activemask())
    for lane_xor in (1, 2, 4):
        peer = _shfl_bfly_f32(value, lane_xor, membermask)
        total = txl.local_scalar("float32")
        txl.ptx.add.f32(total, value, peer)
        value = total
    return value


def _packed_binary(op, a0, a1, b0, b1):
    a = txl.local_scalar("uint64")
    b = txl.local_scalar("uint64")
    out = txl.local_scalar("uint64")
    txl.ptx.mov.b64(a, a0, a1)
    txl.ptx.mov.b64(b, b0, b1)
    txl.ptx[op](out, a, b)
    lo = txl.local_scalar("float32")
    hi = txl.local_scalar("float32")
    txl.ptx.mov.b64(lo, hi, out)
    return lo, hi


def _packed_fma(a0, a1, b0, b1, c0, c1):
    a = txl.local_scalar("uint64")
    b = txl.local_scalar("uint64")
    c = txl.local_scalar("uint64")
    out = txl.local_scalar("uint64")
    txl.ptx.mov.b64(a, a0, a1)
    txl.ptx.mov.b64(b, b0, b1)
    txl.ptx.mov.b64(c, c0, c1)
    txl.ptx["fma.rn.f32x2"](out, a, b, c)
    lo = txl.local_scalar("float32")
    hi = txl.local_scalar("float32")
    txl.ptx.mov.b64(lo, hi, out)
    return lo, hi


def _load_i32(buffer, index):
    out = txl.local_scalar("int32")
    txl.ptx.ld.global_.s32(out, buffer.ptr_to([index]))
    return out


def _publish_pipeline_init():
    txl.ptx["fence.mbarrier_init.release.cluster"]()
    txl.ptx.bar.sync(txl.uint32(0), txl.uint32(THREADS))


def _issue_tma_tile(desc, arena, dst, seq0, head, batch_idx, barrier):
    for half in range(2):
        txl.ptx[TMA_G2S](
            arena.ptr_to([dst + half * 8192]),
            txl.address_of(desc),
            txl.int32(half * 64),
            seq0,
            head,
            batch_idx,
            txl.cuda.cvta_generic_to_shared(barrier),
            TMA_CACHE,
        )


def _issue_tma_pair_tile(desc, arena, dst, seq0, head, batch_idx, barrier):
    for half in range(2):
        txl.ptx[TMA_G2S](
            arena.ptr_to([dst + half * 16384]),
            txl.address_of(desc),
            txl.int32(half * 64),
            seq0,
            head,
            batch_idx,
            txl.cuda.cvta_generic_to_shared(barrier),
            TMA_CACHE,
        )


def _tmem_load16(dst, base, address):
    txl.ptx[TMEM_LD16](*(dst[base + i] for i in range(16)), txl.cast(address, "uint32"))


def _tmem_load32(dst, base, address):
    txl.ptx[TMEM_LD32](*(dst[base + i] for i in range(32)), txl.cast(address, "uint32"))


def get_kernel(**config):
    batch = int(config["batch"])
    heads = int(config["num_heads"])
    seqlen_q = int(config["seqlen_q"])
    seqlen_kv = int(config["seqlen_kv"])
    has_block_sizes = bool(config["has_block_sizes"])
    q_blocks = (seqlen_q + 63) // 64
    bucket = config.get("bucket_size_blocks")
    if bucket is None:
        bucket = 1024 if q_blocks >= 3000 else (1088 if q_blocks < 2048 else 1152)
    groups = (q_blocks + int(bucket) - 1) // int(bucket)
    tasks = (seqlen_kv + 63) // 64
    q8 = (seqlen_q + 7) // 8 * 8
    k8 = (seqlen_kv + 7) // 8 * 8
    bh_count = batch * heads
    sum_plane = bh_count * q8
    dq_base = 2 * sum_plane
    dk_base = dq_base + bh_count * q8 * 128
    dv_base = dk_base + bh_count * k8 * 128

    @txl.kernel(
        warps=4, arch="sm_100a", min_blocks_per_sm=1, grid=((seqlen_q + 15) // 16, heads, batch)
    )
    def sum_odo(
        o: txl.gptr[txl.bf16], do: txl.gptr[txl.bf16], lse: txl.gptr[txl.f32], workspace: txl.gptr[txl.f32]
    ):
        required_block_size = txl.attr({"tirx.required_block_size": 1})
        required_block_size.__enter__()
        q_tile, head, batch_idx = txl.cta_id()
        tid = txl.thread_id()
        tidx = tid % txl.int32(8)
        tidy = tid // txl.int32(8)
        q_idx = q_tile * txl.int32(16) + tidy
        with txl.If(q_idx < txl.int32(seqlen_q)), txl.Then():
            acc = txl.local_scalar("float32", init=txl.float32(0.0))
            row = txl.cast(
                ((batch_idx * txl.int32(heads) + head) * txl.int32(seqlen_q) + q_idx) * txl.int32(128),
                "int64",
            )
            for step in range(8):
                dim = (tidx + txl.int32(step * 8)) * txl.int32(2)
                ow = txl.local_scalar("uint32")
                dw = txl.local_scalar("uint32")
                txl.ptx.ld.global_.b32(ow, o.ptr_to([row + txl.cast(dim, "int64")]))
                txl.ptx.ld.global_.b32(dw, do.ptr_to([row + txl.cast(dim, "int64")]))
                product = txl.local_scalar("uint32")
                txl.ptx["mul.bf16x2"](product, ow, dw)
                lo = txl.local_scalar("uint16")
                hi = txl.local_scalar("uint16")
                txl.ptx.mov.b32(lo, hi, product)
                fragment = txl.local_scalar("float32")
                txl.ptx["cvt.f32.bf16"](fragment, lo)
                txl.ptx["add.rn.f32.bf16"](fragment, hi, fragment)
                txl.ptx.add.f32(acc, acc, fragment)
            total = _butterfly_sum_f32(acc)
            with txl.If(tidx == txl.int32(0)), txl.Then():
                value = txl.local_scalar("float32")
                txl.ptx.neg.f32(value, total)
                bhq = txl.cast((batch_idx * txl.int32(heads) + head) * txl.int32(q8) + q_idx, "int64")
                txl.ptx.st.global_.b32(workspace.ptr_to([bhq]), value)
                lse_value = txl.local_scalar("float32")
                lse_index = txl.cast(
                    (batch_idx * txl.int32(heads) + head) * txl.int32(seqlen_q) + q_idx, "int64"
                )
                txl.ptx.ld.global_.b32(lse_value, lse.ptr_to([lse_index]))
                scaled = txl.local_scalar("float32")
                txl.ptx.mul.f32(scaled, lse_value, txl.float32(-1.4426950408889634))
                txl.ptx.st.global_.b32(workspace.ptr_to([txl.int64(sum_plane) + bhq]), scaled)
        required_block_size.__exit__(None, None, None)

    @txl.kernel(warps=WARPS, arch="sm_100a", min_blocks_per_sm=1, grid=(tasks * groups, heads, batch))
    def bwd(
        q_map: txl.TensorMap,
        k_map: txl.TensorMap,
        v_map: txl.TensorMap,
        do_map: txl.TensorMap,
        dq_map: txl.TensorMap,
        bucketed_offsets: txl.gptr[txl.i32],
        bucketed_indices: txl.gptr[txl.i32],
        block_sizes: txl.gptr[txl.i32],
        workspace: txl.gptr[txl.f32],
        edge_stride: txl.i64,
        softmax_scale: txl.f32,
    ):
        required_block_size = txl.attr({"tirx.required_block_size": 1})
        required_block_size.__enter__()
        block, head, batch_idx = txl.cta_id()
        warp = txl.warp_id()
        with txl.If(warp == txl.int32(13)), txl.Then():
            txl.ptx.prefetch.tensormap(txl.address_of(q_map))
            txl.ptx.prefetch.tensormap(txl.address_of(k_map))
            txl.ptx.prefetch.tensormap(txl.address_of(v_map))
            txl.ptx.prefetch.tensormap(txl.address_of(do_map))

        arena = txl.alloc_buffer((SHARED_BYTES,), txl.u8, scope="shared.dyn", align=1024)
        pool = txl.smem_pool(base=arena).pool
        q_pipe = txl.Pipeline(pool, 2, full="tma", empty="tcgen05", init_full=1, init_empty=1)
        _publish_pipeline_init()
        do_pipe = txl.Pipeline(pool, 1, full="tma", empty="tcgen05", init_full=1, init_empty=1)
        _publish_pipeline_init()
        lse_pipe = txl.Pipeline(pool, 1, full="mbar", empty="mbar", init_full=32, init_empty=256)
        _publish_pipeline_init()
        sum_pipe = txl.Pipeline(pool, 1, full="mbar", empty="mbar", init_full=32, init_empty=256)
        _publish_pipeline_init()
        s_pipe = txl.Pipeline(pool, 1, full="tcgen05", empty="mbar", init_full=1, init_empty=256)
        _publish_pipeline_init()
        dp_pipe = txl.Pipeline(pool, 1, full="tcgen05", empty="mbar", init_full=1, init_empty=256)
        _publish_pipeline_init()
        dq_pipe = txl.Pipeline(pool, 1, full="tcgen05", empty="mbar", init_full=1, init_empty=128)
        _publish_pipeline_init()
        p_pipe = txl.Pipeline(pool, 1, full="mbar", empty="tcgen05", init_full=256, init_empty=1)
        _publish_pipeline_init()
        ds_pipe = txl.Pipeline(pool, 1, full="mbar", empty="tcgen05", init_full=256, init_empty=1)
        _publish_pipeline_init()
        dkdv_pipe = txl.Pipeline(pool, 2, full="tcgen05", empty="mbar", init_full=1, init_empty=256)
        _publish_pipeline_init()
        tmem_mailbox = pool.alloc((1,), "uint32", align=4)
        assert pool.offset == 196
        txl.ptx.bar.sync(txl.uint32(BAR_CTA[0]), txl.uint32(BAR_CTA[1]))

        q_group = block // txl.int32(tasks)
        task = block - q_group * txl.int32(tasks)
        offsets_base = ((batch_idx * txl.int32(heads) + head) * txl.int32(groups) + q_group) * txl.int32(
            tasks + 1
        )
        begin = _load_i32(bucketed_offsets, offsets_base + task)
        end = _load_i32(bucketed_offsets, offsets_base + task + txl.int32(1))
        count = end - begin
        work = (count > txl.int32(0)) & (task * txl.int32(64) < txl.int32(seqlen_kv))

        with txl.If(work), txl.Then():
            smem_base = txl.local_scalar("uint32")
            txl.assign(smem_base, txl.cuda.cvta_generic_to_shared(arena.ptr_to([0])))
            sp = txl.specialize(chain_dispatch=True)
            r_load = sp.role("load", warps=[13], regs=96)
            r_mma = sp.role("mma", warps=[12], regs=96)
            r_compute = sp.role("compute", warps=list(range(4, 12)))
            r_reduce = sp.role("reduce", warps=list(range(4)), regs=152)
            r_empty = sp.role("empty", warps=[14, 15], regs=96)

            with r_load:
                lane = txl.tid_in_role()
                leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
                q_prod = txl.PipelineState(2, phase=0)
                do_prod = txl.PipelineState(1, phase=0)
                lse_prod = txl.PipelineState(1, phase=0)
                sum_prod = txl.PipelineState(1, phase=0)
                remaining = txl.local_scalar("int32", init=count)
                edge = txl.local_scalar("int32", init=0)
                bh_edge = txl.cast(batch_idx * txl.int32(heads) + head, "int64") * edge_stride
                bh_q = txl.cast(batch_idx * txl.int32(heads) + head, "int64") * txl.int64(q8)

                def load_pair(first):
                    q0 = _load_i32(bucketed_indices, bh_edge + txl.cast(begin + edge, "int64"))
                    txl.assign(edge, edge + txl.int32(1))
                    # Use the first out-of-range query block as the pair sentinel.
                    # ``seqlen_q // 64`` aliases the last valid block whenever
                    # the query length is not a multiple of 64, duplicating
                    # its contribution in the odd-edge pair.
                    q1 = txl.local_scalar("int32", init=txl.int32((seqlen_q + 63) // 64))
                    with txl.If(edge < count), txl.Then():
                        txl.assign(
                            q1, _load_i32(bucketed_indices, bh_edge + txl.cast(begin + edge, "int64"))
                        )
                    txl.assign(edge, edge + txl.int32(1))

                    q_pipe.empty.wait(q_prod.stage, q_prod.phase ^ 1)
                    with txl.If(leader != txl.uint32(0)), txl.Then():
                        q_bar = q_pipe.full.buf.ptr_to([q_prod.stage])
                        txl.ptx.mbarrier.arrive.expect_tx.shared.b64(q_bar, txl.uint32(16384))
                        txl.ptx.mbarrier.expect_tx.relaxed.cta.shared__cta.b64(
                            q_bar, txl.uint32(32768 if first else 16384)
                        )
                        if first:
                            _issue_tma_tile(
                                k_map, arena, OFF_SK, task * txl.int32(64), head, batch_idx, q_bar
                            )
                        q_stage = OFF_SQ + q_prod.stage * txl.int32(32768)
                        _issue_tma_pair_tile(
                            q_map, arena, q_stage, q0 * txl.int32(64), head, batch_idx, q_bar
                        )
                        _issue_tma_pair_tile(
                            q_map,
                            arena,
                            q_stage + txl.int32(8192),
                            q1 * txl.int32(64),
                            head,
                            batch_idx,
                            q_bar,
                        )
                    q_prod.advance()

                    lse_stage = lse_prod.stage
                    lse_pipe.empty.wait(lse_stage, lse_prod.phase ^ 1)
                    for pair_slot in range(2):
                        q_block = q0 if pair_slot == 0 else q1
                        for item in range(2):
                            q_row = q_block * txl.int32(64) + lane * txl.int32(2) + txl.int32(item)
                            dst = OFF_SLSE + pair_slot * 256 + lane * txl.int32(8) + item * 4
                            with txl.If(q_row < txl.int32(seqlen_q)):
                                with txl.Then():
                                    txl.ptx["cp.async.ca.shared.global"](
                                        arena.ptr_to([dst]),
                                        workspace.ptr_to(
                                            [txl.int64(sum_plane) + bh_q + txl.cast(q_row, "int64")]
                                        ),
                                        4,
                                        4,
                                    )
                                with txl.Else():
                                    txl.ptx.st.shared.b32(arena.ptr_to([dst]), txl.uint32(0))
                    txl.ptx["cp.async.mbarrier.arrive.noinc.shared.b64"](
                        lse_pipe.full.buf.ptr_to([lse_stage])
                    )
                    lse_prod.advance()

                    do_pipe.empty.wait(do_prod.stage, do_prod.phase ^ 1)
                    with txl.If(leader != txl.uint32(0)), txl.Then():
                        do_bar = do_pipe.full.buf.ptr_to([do_prod.stage])
                        txl.ptx.mbarrier.arrive.expect_tx.shared.b64(do_bar, txl.uint32(16384))
                        txl.ptx.mbarrier.expect_tx.relaxed.cta.shared__cta.b64(
                            do_bar, txl.uint32(32768 if first else 16384)
                        )
                        if first:
                            _issue_tma_tile(
                                v_map, arena, OFF_SV, task * txl.int32(64), head, batch_idx, do_bar
                            )
                        _issue_tma_pair_tile(
                            do_map, arena, OFF_SDO, q0 * txl.int32(64), head, batch_idx, do_bar
                        )
                        _issue_tma_pair_tile(
                            do_map, arena, OFF_SDO + 8192, q1 * txl.int32(64), head, batch_idx, do_bar
                        )
                    do_prod.advance()

                    sum_stage = sum_prod.stage
                    sum_pipe.empty.wait(sum_stage, sum_prod.phase ^ 1)
                    for pair_slot in range(2):
                        q_block = q0 if pair_slot == 0 else q1
                        for item in range(2):
                            q_row = q_block * txl.int32(64) + lane * txl.int32(2) + txl.int32(item)
                            dst = OFF_SSUM + pair_slot * 256 + lane * txl.int32(8) + item * 4
                            with txl.If(q_row < txl.int32(seqlen_q)):
                                with txl.Then():
                                    txl.ptx["cp.async.ca.shared.global"](
                                        arena.ptr_to([dst]),
                                        workspace.ptr_to([bh_q + txl.cast(q_row, "int64")]),
                                        4,
                                        4,
                                    )
                                with txl.Else():
                                    txl.ptx.st.shared.b32(arena.ptr_to([dst]), txl.uint32(0))
                    txl.ptx["cp.async.mbarrier.arrive.noinc.shared.b64"](
                        sum_pipe.full.buf.ptr_to([sum_stage])
                    )
                    sum_prod.advance()
                    txl.assign(remaining, remaining - txl.int32(2))

                load_pair(True)
                with txl.While(remaining > txl.int32(0)):
                    load_pair(False)

            with r_mma:
                txl.ptx[TMEM_ALLOC](
                    txl.cuda.cvta_generic_to_shared(tmem_mailbox.ptr_to([0])), txl.uint32(TMEM_COLUMNS)
                )
                txl.ptx.bar.sync(txl.uint32(BAR_TMEM[0]), txl.uint32(BAR_TMEM[1]))
                tcol = txl.local_scalar("uint32")
                txl.ptx.ld.shared.b32(tcol, tmem_mailbox.ptr_to([0]))
                t_dk = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(TMEM_DK))
                t_dv = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(TMEM_DV))
                t_dp = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(TMEM_DP))
                t_s = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(TMEM_S))
                elected = txl.local_scalar("uint32", init=txl.cuda.elect_sync())

                desc_k = _desc_base(1)
                desc_mn_wide = _desc_base(1024)
                desc_mn_k = _desc_base(512)
                desc_mn_narrow = _desc_base(0)
                d_k_k = _desc_at(desc_k, smem_base + txl.uint32(OFF_SK))
                d_v_k = _desc_at(desc_k, smem_base + txl.uint32(OFF_SV))
                d_do_k = _desc_at(desc_k, smem_base + txl.uint32(OFF_SDO))
                d_do_mn = _desc_at(desc_mn_wide, smem_base + txl.uint32(OFF_SDO))
                d_p_mn = _desc_at(desc_mn_narrow, smem_base + txl.uint32(OFF_SP))
                d_ds_k = _desc_at(desc_k, smem_base + txl.uint32(OFF_SDS))
                d_ds_mn = _desc_at(desc_mn_narrow, smem_base + txl.uint32(OFF_SDS))
                d_k_mn = _desc_at(desc_mn_k, smem_base + txl.uint32(OFF_SK))

                q_cons = txl.PipelineState(2, phase=0)
                q_release = txl.PipelineState(2, phase=0)
                do_cons = txl.PipelineState(1, phase=0)
                s_prod = txl.PipelineState(1, phase=0)
                dp_prod = txl.PipelineState(1, phase=0)
                dq_prod = txl.PipelineState(1, phase=0)
                p_cons = txl.PipelineState(1, phase=0)
                ds_cons = txl.PipelineState(1, phase=0)
                dkdv_prod = txl.PipelineState(2, phase=0)

                q_pipe.full.wait(q_cons.stage, q_cons.phase)
                s_pipe.empty.wait(s_prod.stage, s_prod.phase ^ 1)
                d_q_k = _desc_at(
                    desc_k, smem_base + txl.uint32(OFF_SQ) + q_cons.stage * txl.uint32(32768)
                )
                _mma_chain(
                    t_s,
                    d_q_k,
                    d_k_k,
                    ID_QK,
                    (0, 2, 4, 6, 1024, 1026, 1028, 1030),
                    (0, 2, 4, 6, 512, 514, 516, 518),
                    txl.uint32(0),
                    elected,
                )
                q_cons.advance()
                s_pipe.full.arrive(s_prod.stage, pred=elected)
                s_prod.advance()

                do_pipe.full.wait(do_cons.stage, do_cons.phase)
                dp_pipe.empty.wait(dp_prod.stage, dp_prod.phase ^ 1)
                dq_pipe.empty.wait(dq_prod.stage, dq_prod.phase ^ 1)
                _mma_chain(
                    t_dp,
                    d_do_k,
                    d_v_k,
                    ID_QK,
                    (0, 2, 4, 6, 1024, 1026, 1028, 1030),
                    (0, 2, 4, 6, 512, 514, 516, 518),
                    txl.uint32(0),
                    elected,
                )
                dp_pipe.full.arrive(dp_prod.stage, pred=elected)
                dp_prod.advance()
                p_pipe.full.wait(p_cons.stage, p_cons.phase)
                _mma_chain(
                    t_dv,
                    d_do_mn,
                    d_p_mn,
                    ID_DKDV,
                    (0, 128, 256, 384, 512, 640, 768, 896),
                    (0, 128, 256, 384, 512, 640, 768, 896),
                    txl.uint32(0),
                    elected,
                )
                p_pipe.empty.arrive(p_cons.stage, pred=elected)
                p_cons.advance()
                do_pipe.empty.arrive(do_cons.stage, pred=elected)
                do_cons.advance()

                pairs = (count + txl.int32(1)) // txl.int32(2)
                pair = txl.local_scalar("int32", init=txl.int32(1))
                dk_accumulate = txl.local_scalar("uint32", init=txl.uint32(0))
                with txl.While(pair < pairs):
                    q_pipe.full.wait(q_cons.stage, q_cons.phase)
                    s_pipe.empty.wait(s_prod.stage, s_prod.phase ^ 1)
                    d_q_k = _desc_at(
                        desc_k, smem_base + txl.uint32(OFF_SQ) + q_cons.stage * txl.uint32(32768)
                    )
                    _mma_chain(
                        t_s,
                        d_q_k,
                        d_k_k,
                        ID_QK,
                        (0, 2, 4, 6, 1024, 1026, 1028, 1030),
                        (0, 2, 4, 6, 512, 514, 516, 518),
                        txl.uint32(0),
                        elected,
                    )
                    q_cons.advance()
                    s_pipe.full.arrive(s_prod.stage, pred=elected)
                    s_prod.advance()

                    ds_pipe.full.wait(ds_cons.stage, ds_cons.phase)
                    dp_pipe.empty.wait(dp_prod.stage, dp_prod.phase ^ 1)
                    _mma_chain(
                        txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(TMEM_DQ)),
                        d_ds_k,
                        d_k_mn,
                        ID_DQ,
                        (0, 2, 4, 6),
                        (0, 128, 256, 384),
                        txl.uint32(0),
                        elected,
                    )
                    dq_pipe.full.arrive(dq_prod.stage, pred=elected)
                    dq_prod.advance()

                    d_q_mn = _desc_at(
                        desc_mn_wide,
                        smem_base + txl.uint32(OFF_SQ) + q_release.stage * txl.uint32(32768),
                    )
                    _mma_chain(
                        t_dk,
                        d_q_mn,
                        d_ds_mn,
                        ID_DKDV,
                        (0, 128, 256, 384, 512, 640, 768, 896),
                        (0, 128, 256, 384, 512, 640, 768, 896),
                        dk_accumulate,
                        elected,
                    )
                    txl.assign(dk_accumulate, txl.uint32(1))
                    q_pipe.empty.arrive(q_release.stage, pred=elected)
                    q_release.advance()
                    ds_pipe.empty.arrive(ds_cons.stage, pred=elected)
                    ds_cons.advance()

                    dq_pipe.empty.wait(dq_prod.stage, dq_prod.phase ^ 1)
                    do_pipe.full.wait(do_cons.stage, do_cons.phase)
                    _mma_chain(
                        t_dp,
                        d_do_k,
                        d_v_k,
                        ID_QK,
                        (0, 2, 4, 6, 1024, 1026, 1028, 1030),
                        (0, 2, 4, 6, 512, 514, 516, 518),
                        txl.uint32(0),
                        elected,
                    )
                    dp_pipe.full.arrive(dp_prod.stage, pred=elected)
                    dp_prod.advance()
                    p_pipe.full.wait(p_cons.stage, p_cons.phase)
                    _mma_chain(
                        t_dv,
                        d_do_mn,
                        d_p_mn,
                        ID_DKDV,
                        (0, 128, 256, 384, 512, 640, 768, 896),
                        (0, 128, 256, 384, 512, 640, 768, 896),
                        txl.uint32(1),
                        elected,
                    )
                    p_pipe.empty.arrive(p_cons.stage, pred=elected)
                    p_cons.advance()
                    do_pipe.empty.arrive(do_cons.stage, pred=elected)
                    do_cons.advance()
                    txl.assign(pair, pair + txl.int32(1))

                dkdv_pipe.empty.wait(dkdv_prod.stage, dkdv_prod.phase ^ 1)
                dkdv_pipe.full.arrive(dkdv_prod.stage, pred=elected)
                dkdv_prod.advance()
                dkdv_pipe.empty.wait(dkdv_prod.stage, dkdv_prod.phase ^ 1)
                ds_pipe.full.wait(ds_cons.stage, ds_cons.phase)
                d_q_mn = _desc_at(
                    desc_mn_wide, smem_base + txl.uint32(OFF_SQ) + q_release.stage * txl.uint32(32768)
                )
                _mma_chain(
                    t_dk,
                    d_q_mn,
                    d_ds_mn,
                    ID_DKDV,
                    (0, 128, 256, 384, 512, 640, 768, 896),
                    (0, 128, 256, 384, 512, 640, 768, 896),
                    dk_accumulate,
                    elected,
                )
                dkdv_pipe.full.arrive(dkdv_prod.stage, pred=elected)
                dkdv_prod.advance()
                _mma_chain(
                    txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(TMEM_DQ)),
                    d_ds_k,
                    d_k_mn,
                    ID_DQ,
                    (0, 2, 4, 6),
                    (0, 128, 256, 384),
                    txl.uint32(0),
                    elected,
                )
                dq_pipe.full.arrive(dq_prod.stage, pred=elected)
                dq_prod.advance()
                q_pipe.empty.arrive(q_release.stage, pred=elected)
                q_release.advance()
                ds_pipe.empty.arrive(ds_cons.stage, pred=elected)
                ds_cons.advance()

            with r_compute:
                txl.ptx.setmaxnreg.inc.sync.aligned.u32(txl.uint32(128))
                txl.ptx.bar.sync(txl.uint32(BAR_TMEM[0]), txl.uint32(BAR_TMEM[1]))
                tcol = txl.local_scalar("uint32")
                txl.ptx.ld.shared.b32(tcol, tmem_mailbox.ptr_to([0]))
                ctid = txl.tid_in_role()
                crow = ctid % txl.int32(128)
                wg = ctid // txl.int32(128)
                row_group = (crow // txl.int32(32)) * txl.int32(32)
                scale_log2 = txl.local_scalar("float32")
                txl.ptx.mul.f32(scale_log2, softmax_scale, txl.float32(1.4426950408889634))
                block_size = txl.local_scalar("int32", init=txl.int32(64))
                if has_block_sizes:
                    txl.ptx.ld.global_.s32(
                        block_size, block_sizes.ptr_to([batch_idx * txl.int32(tasks) + task])
                    )

                s_cons = txl.PipelineState(1, phase=0)
                lse_cons = txl.PipelineState(1, phase=0)
                p_prod = txl.PipelineState(1, phase=0)
                sum_cons = txl.PipelineState(1, phase=0)
                dp_cons = txl.PipelineState(1, phase=0)
                ds_prod = txl.PipelineState(1, phase=0)
                dkdv_cons = txl.PipelineState(2, phase=0)
                pair = txl.local_scalar("int32", init=txl.int32(0))
                pairs = (count + txl.int32(1)) // txl.int32(2)
                with txl.While(pair < pairs):
                    s_pipe.full.wait(s_cons.stage, s_cons.phase)
                    lse_pipe.full.wait(lse_cons.stage, lse_cons.phase)
                    p_pipe.empty.wait(p_prod.stage, p_prod.phase ^ 1)
                    scores = txl.alloc_local((32,), "float32")
                    for issue in range(2):
                        address = txl.cuda.get_tmem_addr(
                            tcol, row_group, txl.int32(TMEM_S + issue * 32) + wg * txl.int32(16)
                        )
                        _tmem_load16(scores, issue * 16, address)
                    if has_block_sizes:
                        for j in range(32):
                            col = wg * txl.int32(16) + txl.int32((j // 16) * 32 + j % 16)
                            live = txl.local_scalar("uint32")
                            txl.ptx["setp.lt.s32"](live, col, block_size)
                            bits = txl.local_scalar("uint32")
                            txl.ptx.selp.b32(
                                bits,
                                txl.reinterpret(txl.u32, scores[j]),
                                txl.uint32(0xFF800000),
                                txl.ptx.pred(live),
                            )
                            txl.assign(scores[j], txl.reinterpret(txl.f32, bits))
                    lse_value = txl.local_scalar("float32")
                    txl.ptx.ld.shared_.b32(lse_value, arena.ptr_to([OFF_SLSE + crow * txl.int32(4)]))
                    for j in range(0, 32, 2):
                        lo, hi = _packed_fma(
                            scores[j], scores[j + 1], scale_log2, scale_log2, lse_value, lse_value
                        )
                        txl.ptx.ex2.approx.ftz.f32(scores[j], lo)
                        txl.ptx.ex2.approx.ftz.f32(scores[j + 1], hi)
                    packed_p = txl.alloc_local((16,), "uint32")
                    for j in range(16):
                        txl.ptx["cvt.rn.bf16x2.f32"](packed_p[j], scores[2 * j + 1], scores[2 * j])
                    txl.ptx["tcgen05.wait::ld.sync.aligned"]()
                    txl.ptx.bar.sync(txl.uint32(BAR_COMPUTE[0]), txl.uint32(BAR_COMPUTE[1]))
                    txl.ptx["tcgen05.wait::ld.sync.aligned"]()
                    for group in range(4):
                        col = wg * txl.int32(16) + txl.int32((group // 2) * 32 + (group % 2) * 8)
                        txl.ptx.st.shared.v4.b32(
                            arena.ptr_to([_tile_byte(OFF_SP, crow, col)]),
                            packed_p[group * 4],
                            packed_p[group * 4 + 1],
                            packed_p[group * 4 + 2],
                            packed_p[group * 4 + 3],
                        )
                    txl.ptx.fence.proxy.async_.shared__cta()
                    p_pipe.full.arrive(p_prod.stage)
                    p_prod.advance()
                    s_pipe.empty.arrive(s_cons.stage)
                    s_cons.advance()
                    lse_pipe.empty.arrive(lse_cons.stage)
                    lse_cons.advance()

                    sum_pipe.full.wait(sum_cons.stage, sum_cons.phase)
                    dp_pipe.full.wait(dp_cons.stage, dp_cons.phase)
                    ds_pipe.empty.wait(ds_prod.stage, ds_prod.phase ^ 1)
                    dp_values = txl.alloc_local((32,), "float32")
                    for issue in range(2):
                        address = txl.cuda.get_tmem_addr(
                            tcol, row_group, txl.int32(TMEM_DP + issue * 32) + wg * txl.int32(16)
                        )
                        _tmem_load16(dp_values, issue * 16, address)
                    sum_value = txl.local_scalar("float32")
                    txl.ptx.ld.shared_.b32(sum_value, arena.ptr_to([OFF_SSUM + crow * txl.int32(4)]))
                    for j in range(0, 32, 2):
                        lo, hi = _packed_binary(
                            "add.rn.f32x2", dp_values[j], dp_values[j + 1], sum_value, sum_value
                        )
                        lo, hi = _packed_binary("mul.rn.f32x2", lo, hi, scores[j], scores[j + 1])
                        txl.assign(dp_values[j], lo)
                        txl.assign(dp_values[j + 1], hi)
                    packed_ds = txl.alloc_local((16,), "uint32")
                    for j in range(16):
                        txl.ptx["cvt.rn.bf16x2.f32"](
                            packed_ds[j], dp_values[2 * j + 1], dp_values[2 * j]
                        )
                    txl.ptx["tcgen05.wait::ld.sync.aligned"]()
                    dp_pipe.empty.arrive(dp_cons.stage)
                    dp_cons.advance()
                    for group in range(4):
                        col = wg * txl.int32(16) + txl.int32((group // 2) * 32 + (group % 2) * 8)
                        txl.ptx.st.shared.v4.b32(
                            arena.ptr_to([_tile_byte(OFF_SDS, crow, col)]),
                            packed_ds[group * 4],
                            packed_ds[group * 4 + 1],
                            packed_ds[group * 4 + 2],
                            packed_ds[group * 4 + 3],
                        )
                    txl.ptx.fence.proxy.async_.shared__cta()
                    ds_pipe.full.arrive(ds_prod.stage)
                    ds_prod.advance()
                    sum_pipe.empty.arrive(sum_cons.stage)
                    sum_cons.advance()
                    txl.assign(pair, pair + txl.int32(1))

                bh = txl.cast(batch_idx * txl.int32(heads) + head, "int64")
                for is_dk in (False, True):
                    dkdv_pipe.full.wait(dkdv_cons.stage, dkdv_cons.phase)
                    values = txl.alloc_local((32,), "float32")
                    tmem_offset = TMEM_DK if is_dk else TMEM_DV
                    for issue in range(2):
                        address = txl.cuda.get_tmem_addr(
                            tcol, row_group, txl.int32(tmem_offset + issue * 32) + wg * txl.int32(16)
                        )
                        _tmem_load16(values, issue * 16, address)
                    txl.ptx["tcgen05.wait::ld.sync.aligned"]()
                    for j in range(32):
                        col = wg * txl.int32(16) + txl.int32((j // 16) * 32 + j % 16)
                        seq = task * txl.int32(64) + col
                        with txl.If(seq < txl.int32(seqlen_kv)), txl.Then():
                            base = dk_base if is_dk else dv_base
                            index = (
                                txl.int64(base)
                                + bh * txl.int64(k8 * 128)
                                + txl.cast(seq, "int64") * txl.int64(128)
                                + txl.cast(crow, "int64")
                            )
                            old = txl.local_scalar("float32")
                            txl.ptx.atom.global_.add.f32(old, workspace.ptr_to([index]), values[j])
                    dkdv_pipe.empty.arrive(dkdv_cons.stage)
                    dkdv_cons.advance()
                txl.ptx.bar.sync(txl.uint32(BAR_DEALLOC[0]), txl.uint32(BAR_DEALLOC[1]))
                with txl.If(txl.warp_id() == txl.int32(8)), txl.Then():
                    txl.ptx[TMEM_DEALLOC](tcol, txl.uint32(TMEM_COLUMNS))

            with r_reduce:
                txl.ptx.bar.sync(txl.uint32(BAR_TMEM[0]), txl.uint32(BAR_TMEM[1]))
                tcol = txl.local_scalar("uint32")
                txl.ptx.ld.shared.b32(tcol, tmem_mailbox.ptr_to([0]))
                rtid = txl.tid_in_role()
                row_group = (rtid // txl.int32(32)) * txl.int32(32)
                dq_cons = txl.PipelineState(1, phase=0)
                store_stage = txl.local_scalar("int32", init=txl.int32(0))
                edge = txl.local_scalar("int32", init=txl.int32(0))
                bh_edge = txl.cast(batch_idx * txl.int32(heads) + head, "int64") * edge_stride
                remaining = txl.local_scalar("int32", init=count)
                with txl.While(remaining > txl.int32(0)):
                    dq_pipe.full.wait(dq_cons.stage, dq_cons.phase)
                    q0 = _load_i32(bucketed_indices, bh_edge + txl.cast(begin + edge, "int64"))
                    txl.assign(edge, edge + txl.int32(1))
                    q1 = txl.local_scalar("int32", init=txl.int32((seqlen_q + 63) // 64))
                    with txl.If(edge < count), txl.Then():
                        txl.assign(
                            q1, _load_i32(bucketed_indices, bh_edge + txl.cast(begin + edge, "int64"))
                        )
                    txl.assign(edge, edge + txl.int32(1))
                    values = txl.alloc_local((128,), "float32")
                    for issue in range(4):
                        address = txl.cuda.get_tmem_addr(
                            tcol, row_group, txl.int32(TMEM_DQ + issue * 32)
                        )
                        _tmem_load32(values, issue * 32, address)
                    txl.ptx["tcgen05.wait::ld.sync.aligned"]()
                    dq_pipe.empty.arrive(dq_cons.stage)
                    dq_cons.advance()
                    for chunk in range(4):
                        with txl.If(txl.warp_id() == txl.int32(0)), txl.Then():
                            txl.ptx.cp.async_.bulk.wait_group.read(1)
                        txl.ptx.bar.sync(txl.uint32(BAR_REDUCE[0]), txl.uint32(BAR_REDUCE[1]))
                        pair_slot = rtid // txl.int32(64)
                        row = rtid % txl.int32(64)
                        for vec in range(8):
                            raw = (
                                txl.int32(OFF_SDQ)
                                + store_stage * txl.int32(16384)
                                + pair_slot * txl.int32(8192)
                                + row * txl.int32(128)
                                + txl.int32(16 * (7 - vec))
                            )
                            address = txl.bitwise_xor(
                                raw, txl.bitwise_and(txl.shift_right(raw, txl.int32(3)), txl.int32(0x70))
                            )
                            # The source fragment and its shared CopyAtom both
                            # walk the eight vectors in reverse.  Reversing the
                            # register vector together with the address keeps
                            # logical dQ dimensions in ascending order.
                            base = chunk * 32 + (7 - vec) * 4
                            txl.ptx.st.shared.v4.b32(
                                arena.ptr_to([address]),
                                values[base],
                                values[base + 1],
                                values[base + 2],
                                values[base + 3],
                            )
                        txl.ptx.fence.proxy.async_.shared__cta()
                        txl.ptx.bar.sync(txl.uint32(BAR_REDUCE[0]), txl.uint32(BAR_REDUCE[1]))
                        with txl.If(txl.warp_id() == txl.int32(0)), txl.Then():
                            leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
                            with txl.If(leader != txl.uint32(0)), txl.Then():
                                txl.ptx[TMA_REDUCE](
                                    txl.address_of(dq_map),
                                    txl.int32(chunk * 32),
                                    q0 * txl.int32(64),
                                    head,
                                    batch_idx,
                                    arena.ptr_to([OFF_SDQ + store_stage * txl.int32(16384)]),
                                    TMA_CACHE,
                                )
                                txl.ptx[TMA_REDUCE](
                                    txl.address_of(dq_map),
                                    txl.int32(chunk * 32),
                                    q1 * txl.int32(64),
                                    head,
                                    batch_idx,
                                    arena.ptr_to([OFF_SDQ + store_stage * txl.int32(16384) + 8192]),
                                    TMA_CACHE,
                                )
                            txl.ptx.cp.async_.bulk.commit_group()
                        txl.assign(store_stage, txl.Select(store_stage == txl.int32(1), 0, 1))
                    txl.assign(remaining, remaining - txl.int32(2))
                txl.ptx.cp.async_.bulk.wait_group.read(0)

            with r_empty:
                pass
        required_block_size.__exit__(None, None, None)

    @txl.kernel(
        warps=4,
        arch="sm_100a",
        min_blocks_per_sm=1,
        grid=((max(seqlen_q, seqlen_kv) + 7) // 8, heads, batch),
    )
    def convert(
        workspace: txl.gptr[txl.f32],
        dq: txl.gptr[txl.bf16],
        dk: txl.gptr[txl.bf16],
        dv: txl.gptr[txl.bf16],
        softmax_scale: txl.f32,
    ):
        required_block_size = txl.attr({"tirx.required_block_size": 1})
        required_block_size.__enter__()
        seq_tile, head, batch_idx = txl.cta_id()
        tid = txl.thread_id()
        tidx = tid % txl.int32(16)
        tidy = tid // txl.int32(16)
        seq = seq_tile * txl.int32(8) + tidy
        bh = txl.cast(batch_idx * txl.int32(heads) + head, "int64")
        for group in range(2):
            dim = (tidx + txl.int32(group * 16)) * txl.int32(4)
            with txl.If(seq < txl.int32(seqlen_q)), txl.Then():
                source = (
                    txl.int64(dq_base)
                    + bh * txl.int64(q8 * 128)
                    + txl.cast(seq, "int64") * txl.int64(128)
                    + txl.cast(dim, "int64")
                )
                values = txl.alloc_local((4,), "float32")
                txl.ptx.ld.global_.v4.b32(
                    values[0], values[1], values[2], values[3], workspace.ptr_to([source])
                )
                scaled = txl.alloc_local((4,), "float32")
                for pair in range(2):
                    lo, hi = _packed_binary(
                        "mul.rn.f32x2",
                        values[pair * 2],
                        values[pair * 2 + 1],
                        softmax_scale,
                        softmax_scale,
                    )
                    txl.assign(scaled[pair * 2], lo)
                    txl.assign(scaled[pair * 2 + 1], hi)
                out0 = txl.local_scalar("uint32")
                out1 = txl.local_scalar("uint32")
                txl.ptx["cvt.rn.bf16x2.f32"](out0, scaled[1], scaled[0])
                txl.ptx["cvt.rn.bf16x2.f32"](out1, scaled[3], scaled[2])
                output = (bh * txl.int64(seqlen_q) + txl.cast(seq, "int64")) * txl.int64(128) + txl.cast(
                    dim, "int64"
                )
                txl.ptx.st.global_.v2.b32(dq.ptr_to([output]), out0, out1)
            with txl.If(seq < txl.int32(seqlen_kv)), txl.Then():
                dk_source = (
                    txl.int64(dk_base)
                    + bh * txl.int64(k8 * 128)
                    + txl.cast(seq, "int64") * txl.int64(128)
                    + txl.cast(dim, "int64")
                )
                dv_source = (
                    txl.int64(dv_base)
                    + bh * txl.int64(k8 * 128)
                    + txl.cast(seq, "int64") * txl.int64(128)
                    + txl.cast(dim, "int64")
                )
                kvals = txl.alloc_local((4,), "float32")
                vvals = txl.alloc_local((4,), "float32")
                txl.ptx.ld.global_.v4.b32(
                    kvals[0], kvals[1], kvals[2], kvals[3], workspace.ptr_to([dk_source])
                )
                txl.ptx.ld.global_.v4.b32(
                    vvals[0], vvals[1], vvals[2], vvals[3], workspace.ptr_to([dv_source])
                )
                kscaled = txl.alloc_local((4,), "float32")
                for pair in range(2):
                    lo, hi = _packed_binary(
                        "mul.rn.f32x2",
                        kvals[pair * 2],
                        kvals[pair * 2 + 1],
                        softmax_scale,
                        softmax_scale,
                    )
                    txl.assign(kscaled[pair * 2], lo)
                    txl.assign(kscaled[pair * 2 + 1], hi)
                k0 = txl.local_scalar("uint32")
                k1 = txl.local_scalar("uint32")
                v0 = txl.local_scalar("uint32")
                v1 = txl.local_scalar("uint32")
                txl.ptx["cvt.rn.bf16x2.f32"](k0, kscaled[1], kscaled[0])
                txl.ptx["cvt.rn.bf16x2.f32"](k1, kscaled[3], kscaled[2])
                txl.ptx["cvt.rn.bf16x2.f32"](v0, vvals[1], vvals[0])
                txl.ptx["cvt.rn.bf16x2.f32"](v1, vvals[3], vvals[2])
                output = (bh * txl.int64(seqlen_kv) + txl.cast(seq, "int64")) * txl.int64(128) + txl.cast(
                    dim, "int64"
                )
                txl.ptx.st.global_.v2.b32(dk.ptr_to([output]), k0, k1)
                txl.ptx.st.global_.v2.b32(dv.ptr_to([output]), v0, v1)
        required_block_size.__exit__(None, None, None)

    return sum_odo.func, bwd.func, convert.func
