# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Source-shaped SM100 blk64 block-sparse-attention backward kernels."""

import tirx_kernels.kern as K

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
TMA_CACHE = K.uint64(0)


def _xor(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return a ^ b
    return K.bitwise_xor(a, b)


def _tile_elem(row, dim):
    return (dim // 64) * 4096 + _xor(row * 64 + dim % 64, (row % 8) * 8)


def _tile_byte(base, row, dim):
    return base + 2 * _tile_elem(row, dim)


def _desc_base(ldo, sdo=64, swizzle=3):
    arrangement = {0: 0, 1: 6, 2: 4, 3: 2, 4: 1}[swizzle]
    value = ((ldo & 0x3FFF) << 16) | ((sdo & 0x3FFF) << 32) | (1 << 46)
    return (value | ((arrangement & 0x7) << 61)) & 0xFFFFFFFFFFFFFFFF


def _desc_at(base, shared_address):
    field = K.cast(
        K.bitwise_and(K.shift_right(shared_address, K.uint32(4)), K.uint32(0x3FFF)), "uint64"
    )
    return K.bitwise_or(K.uint64(base), field)


def _desc_add16(desc, offset):
    if offset == 0:
        return desc
    lo = K.local_scalar("uint32")
    hi = K.local_scalar("uint32")
    out = K.local_scalar("uint64")
    K.ptx.mov.b64(lo, hi, desc)
    K.ptx.add.u32(lo, lo, K.uint32(offset))
    K.ptx.mov.b64(out, lo, hi)
    return out


def _mma(dest, a_desc, b_desc, idesc, accumulate, pred):
    K.ptx[MMA_F16](
        K.cast(dest, "uint32"),
        a_desc,
        b_desc,
        K.uint32(idesc),
        K.uint32(0),
        K.uint32(0),
        K.uint32(0),
        K.uint32(0),
        K.ptx.pred(accumulate),
        pred=pred,
    )


def _mma_chain(dest, a_desc, b_desc, idesc, a_offsets, b_offsets, accumulate, pred):
    flag = K.local_scalar("uint32", init=K.cast(accumulate, "uint32"))
    for a_offset, b_offset in zip(a_offsets, b_offsets):
        _mma(dest, _desc_add16(a_desc, a_offset), _desc_add16(b_desc, b_offset), idesc, flag, pred)
        K.assign(flag, K.uint32(1))


def _shfl_bfly_f32(value, lane_xor, membermask):
    out = K.local_scalar("uint32")
    K.ptx.shfl_sync.bfly.b32(
        out, K.reinterpret(K.u32, value), K.uint32(lane_xor), K.uint32(31), membermask
    )
    return K.reinterpret(K.f32, out)


def _butterfly_sum_f32(value):
    membermask = K.local_scalar("uint32", init=K.tvm_warp_activemask())
    for lane_xor in (1, 2, 4):
        peer = _shfl_bfly_f32(value, lane_xor, membermask)
        total = K.local_scalar("float32")
        K.ptx.add.f32(total, value, peer)
        value = total
    return value


def _packed_binary(op, a0, a1, b0, b1):
    a = K.local_scalar("uint64")
    b = K.local_scalar("uint64")
    out = K.local_scalar("uint64")
    K.ptx.mov.b64(a, a0, a1)
    K.ptx.mov.b64(b, b0, b1)
    K.ptx[op](out, a, b)
    lo = K.local_scalar("float32")
    hi = K.local_scalar("float32")
    K.ptx.mov.b64(lo, hi, out)
    return lo, hi


def _packed_fma(a0, a1, b0, b1, c0, c1):
    a = K.local_scalar("uint64")
    b = K.local_scalar("uint64")
    c = K.local_scalar("uint64")
    out = K.local_scalar("uint64")
    K.ptx.mov.b64(a, a0, a1)
    K.ptx.mov.b64(b, b0, b1)
    K.ptx.mov.b64(c, c0, c1)
    K.ptx["fma.rn.f32x2"](out, a, b, c)
    lo = K.local_scalar("float32")
    hi = K.local_scalar("float32")
    K.ptx.mov.b64(lo, hi, out)
    return lo, hi


def _load_i32(buffer, index):
    out = K.local_scalar("int32")
    K.ptx.ld.global_.s32(out, buffer.ptr_to([index]))
    return out


def _publish_pipeline_init():
    K.ptx["fence.mbarrier_init.release.cluster"]()
    K.ptx.bar.sync(K.uint32(0), K.uint32(THREADS))


def _issue_tma_tile(desc, arena, dst, seq0, head, batch_idx, barrier):
    for half in range(2):
        K.ptx[TMA_G2S](
            arena.ptr_to([dst + half * 8192]),
            K.address_of(desc),
            K.int32(half * 64),
            seq0,
            head,
            batch_idx,
            K.cuda.cvta_generic_to_shared(barrier),
            TMA_CACHE,
        )


def _issue_tma_pair_tile(desc, arena, dst, seq0, head, batch_idx, barrier):
    for half in range(2):
        K.ptx[TMA_G2S](
            arena.ptr_to([dst + half * 16384]),
            K.address_of(desc),
            K.int32(half * 64),
            seq0,
            head,
            batch_idx,
            K.cuda.cvta_generic_to_shared(barrier),
            TMA_CACHE,
        )


def _tmem_load16(dst, base, address):
    K.ptx[TMEM_LD16](*(dst[base + i] for i in range(16)), K.cast(address, "uint32"))


def _tmem_load32(dst, base, address):
    K.ptx[TMEM_LD32](*(dst[base + i] for i in range(32)), K.cast(address, "uint32"))


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

    @K.kernel(
        warps=4, arch="sm_100a", min_blocks_per_sm=1, grid=((seqlen_q + 15) // 16, heads, batch)
    )
    def sum_odo(
        o: K.gptr[K.bf16], do: K.gptr[K.bf16], lse: K.gptr[K.f32], workspace: K.gptr[K.f32]
    ):
        required_block_size = K.attr({"tirx.required_block_size": 1})
        required_block_size.__enter__()
        q_tile, head, batch_idx = K.cta_id()
        tid = K.thread_id()
        tidx = tid % K.int32(8)
        tidy = tid // K.int32(8)
        q_idx = q_tile * K.int32(16) + tidy
        with K.If(q_idx < K.int32(seqlen_q)), K.Then():
            acc = K.local_scalar("float32", init=K.float32(0.0))
            row = K.cast(
                ((batch_idx * K.int32(heads) + head) * K.int32(seqlen_q) + q_idx) * K.int32(128),
                "int64",
            )
            for step in range(8):
                dim = (tidx + K.int32(step * 8)) * K.int32(2)
                ow = K.local_scalar("uint32")
                dw = K.local_scalar("uint32")
                K.ptx.ld.global_.b32(ow, o.ptr_to([row + K.cast(dim, "int64")]))
                K.ptx.ld.global_.b32(dw, do.ptr_to([row + K.cast(dim, "int64")]))
                product = K.local_scalar("uint32")
                K.ptx["mul.bf16x2"](product, ow, dw)
                lo = K.local_scalar("uint16")
                hi = K.local_scalar("uint16")
                K.ptx.mov.b32(lo, hi, product)
                fragment = K.local_scalar("float32")
                K.ptx["cvt.f32.bf16"](fragment, lo)
                K.ptx["add.rn.f32.bf16"](fragment, hi, fragment)
                K.ptx.add.f32(acc, acc, fragment)
            total = _butterfly_sum_f32(acc)
            with K.If(tidx == K.int32(0)), K.Then():
                value = K.local_scalar("float32")
                K.ptx.neg.f32(value, total)
                bhq = K.cast((batch_idx * K.int32(heads) + head) * K.int32(q8) + q_idx, "int64")
                K.ptx.st.global_.b32(workspace.ptr_to([bhq]), value)
                lse_value = K.local_scalar("float32")
                lse_index = K.cast(
                    (batch_idx * K.int32(heads) + head) * K.int32(seqlen_q) + q_idx, "int64"
                )
                K.ptx.ld.global_.b32(lse_value, lse.ptr_to([lse_index]))
                scaled = K.local_scalar("float32")
                K.ptx.mul.f32(scaled, lse_value, K.float32(-1.4426950408889634))
                K.ptx.st.global_.b32(workspace.ptr_to([K.int64(sum_plane) + bhq]), scaled)
        required_block_size.__exit__(None, None, None)

    @K.kernel(warps=WARPS, arch="sm_100a", min_blocks_per_sm=1, grid=(tasks * groups, heads, batch))
    def bwd(
        q_map: K.TensorMap,
        k_map: K.TensorMap,
        v_map: K.TensorMap,
        do_map: K.TensorMap,
        dq_map: K.TensorMap,
        bucketed_offsets: K.gptr[K.i32],
        bucketed_indices: K.gptr[K.i32],
        block_sizes: K.gptr[K.i32],
        workspace: K.gptr[K.f32],
        edge_stride: K.i64,
        softmax_scale: K.f32,
    ):
        required_block_size = K.attr({"tirx.required_block_size": 1})
        required_block_size.__enter__()
        block, head, batch_idx = K.cta_id()
        warp = K.warp_id()
        with K.If(warp == K.int32(13)), K.Then():
            K.ptx.prefetch.tensormap(K.address_of(q_map))
            K.ptx.prefetch.tensormap(K.address_of(k_map))
            K.ptx.prefetch.tensormap(K.address_of(v_map))
            K.ptx.prefetch.tensormap(K.address_of(do_map))

        arena = K.alloc_buffer((SHARED_BYTES,), K.u8, scope="shared.dyn", align=1024)
        pool = K.smem_pool(base=arena).pool
        q_pipe = K.Pipeline(pool, 2, full="tma", empty="tcgen05", init_full=1, init_empty=1)
        _publish_pipeline_init()
        do_pipe = K.Pipeline(pool, 1, full="tma", empty="tcgen05", init_full=1, init_empty=1)
        _publish_pipeline_init()
        lse_pipe = K.Pipeline(pool, 1, full="mbar", empty="mbar", init_full=32, init_empty=256)
        _publish_pipeline_init()
        sum_pipe = K.Pipeline(pool, 1, full="mbar", empty="mbar", init_full=32, init_empty=256)
        _publish_pipeline_init()
        s_pipe = K.Pipeline(pool, 1, full="tcgen05", empty="mbar", init_full=1, init_empty=256)
        _publish_pipeline_init()
        dp_pipe = K.Pipeline(pool, 1, full="tcgen05", empty="mbar", init_full=1, init_empty=256)
        _publish_pipeline_init()
        dq_pipe = K.Pipeline(pool, 1, full="tcgen05", empty="mbar", init_full=1, init_empty=128)
        _publish_pipeline_init()
        p_pipe = K.Pipeline(pool, 1, full="mbar", empty="tcgen05", init_full=256, init_empty=1)
        _publish_pipeline_init()
        ds_pipe = K.Pipeline(pool, 1, full="mbar", empty="tcgen05", init_full=256, init_empty=1)
        _publish_pipeline_init()
        dkdv_pipe = K.Pipeline(pool, 2, full="tcgen05", empty="mbar", init_full=1, init_empty=256)
        _publish_pipeline_init()
        tmem_mailbox = pool.alloc((1,), "uint32", align=4)
        assert pool.offset == 196
        K.ptx.bar.sync(K.uint32(BAR_CTA[0]), K.uint32(BAR_CTA[1]))

        q_group = block // K.int32(tasks)
        task = block - q_group * K.int32(tasks)
        offsets_base = ((batch_idx * K.int32(heads) + head) * K.int32(groups) + q_group) * K.int32(
            tasks + 1
        )
        begin = _load_i32(bucketed_offsets, offsets_base + task)
        end = _load_i32(bucketed_offsets, offsets_base + task + K.int32(1))
        count = end - begin
        work = (count > K.int32(0)) & (task * K.int32(64) < K.int32(seqlen_kv))

        with K.If(work), K.Then():
            smem_base = K.local_scalar("uint32")
            K.assign(smem_base, K.cuda.cvta_generic_to_shared(arena.ptr_to([0])))
            sp = K.specialize(chain_dispatch=True)
            r_load = sp.role("load", warps=[13], regs=96)
            r_mma = sp.role("mma", warps=[12], regs=96)
            r_compute = sp.role("compute", warps=list(range(4, 12)))
            r_reduce = sp.role("reduce", warps=list(range(4)), regs=152)
            r_empty = sp.role("empty", warps=[14, 15], regs=96)

            with r_load:
                lane = K.tid_in_role()
                leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                q_prod = K.PipelineState(2, phase=0)
                do_prod = K.PipelineState(1, phase=0)
                lse_prod = K.PipelineState(1, phase=0)
                sum_prod = K.PipelineState(1, phase=0)
                remaining = K.local_scalar("int32", init=count)
                edge = K.local_scalar("int32", init=0)
                bh_edge = K.cast(batch_idx * K.int32(heads) + head, "int64") * edge_stride
                bh_q = K.cast(batch_idx * K.int32(heads) + head, "int64") * K.int64(q8)

                def load_pair(first):
                    q0 = _load_i32(bucketed_indices, bh_edge + K.cast(begin + edge, "int64"))
                    K.assign(edge, edge + K.int32(1))
                    # Use the first out-of-range query block as the pair sentinel.
                    # ``seqlen_q // 64`` aliases the last valid block whenever
                    # the query length is not a multiple of 64, duplicating
                    # its contribution in the odd-edge pair.
                    q1 = K.local_scalar("int32", init=K.int32((seqlen_q + 63) // 64))
                    with K.If(edge < count), K.Then():
                        K.assign(
                            q1, _load_i32(bucketed_indices, bh_edge + K.cast(begin + edge, "int64"))
                        )
                    K.assign(edge, edge + K.int32(1))

                    q_pipe.empty.wait(q_prod.stage, q_prod.phase ^ 1)
                    with K.If(leader != K.uint32(0)), K.Then():
                        q_bar = q_pipe.full.buf.ptr_to([q_prod.stage])
                        K.ptx.mbarrier.arrive.expect_tx.shared.b64(q_bar, K.uint32(16384))
                        K.ptx.mbarrier.expect_tx.relaxed.cta.shared__cta.b64(
                            q_bar, K.uint32(32768 if first else 16384)
                        )
                        if first:
                            _issue_tma_tile(
                                k_map, arena, OFF_SK, task * K.int32(64), head, batch_idx, q_bar
                            )
                        q_stage = OFF_SQ + q_prod.stage * K.int32(32768)
                        _issue_tma_pair_tile(
                            q_map, arena, q_stage, q0 * K.int32(64), head, batch_idx, q_bar
                        )
                        _issue_tma_pair_tile(
                            q_map,
                            arena,
                            q_stage + K.int32(8192),
                            q1 * K.int32(64),
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
                            q_row = q_block * K.int32(64) + lane * K.int32(2) + K.int32(item)
                            dst = OFF_SLSE + pair_slot * 256 + lane * K.int32(8) + item * 4
                            with K.If(q_row < K.int32(seqlen_q)):
                                with K.Then():
                                    K.ptx["cp.async.ca.shared.global"](
                                        arena.ptr_to([dst]),
                                        workspace.ptr_to(
                                            [K.int64(sum_plane) + bh_q + K.cast(q_row, "int64")]
                                        ),
                                        4,
                                        4,
                                    )
                                with K.Else():
                                    K.ptx.st.shared.b32(arena.ptr_to([dst]), K.uint32(0))
                    K.ptx["cp.async.mbarrier.arrive.noinc.shared.b64"](
                        lse_pipe.full.buf.ptr_to([lse_stage])
                    )
                    lse_prod.advance()

                    do_pipe.empty.wait(do_prod.stage, do_prod.phase ^ 1)
                    with K.If(leader != K.uint32(0)), K.Then():
                        do_bar = do_pipe.full.buf.ptr_to([do_prod.stage])
                        K.ptx.mbarrier.arrive.expect_tx.shared.b64(do_bar, K.uint32(16384))
                        K.ptx.mbarrier.expect_tx.relaxed.cta.shared__cta.b64(
                            do_bar, K.uint32(32768 if first else 16384)
                        )
                        if first:
                            _issue_tma_tile(
                                v_map, arena, OFF_SV, task * K.int32(64), head, batch_idx, do_bar
                            )
                        _issue_tma_pair_tile(
                            do_map, arena, OFF_SDO, q0 * K.int32(64), head, batch_idx, do_bar
                        )
                        _issue_tma_pair_tile(
                            do_map, arena, OFF_SDO + 8192, q1 * K.int32(64), head, batch_idx, do_bar
                        )
                    do_prod.advance()

                    sum_stage = sum_prod.stage
                    sum_pipe.empty.wait(sum_stage, sum_prod.phase ^ 1)
                    for pair_slot in range(2):
                        q_block = q0 if pair_slot == 0 else q1
                        for item in range(2):
                            q_row = q_block * K.int32(64) + lane * K.int32(2) + K.int32(item)
                            dst = OFF_SSUM + pair_slot * 256 + lane * K.int32(8) + item * 4
                            with K.If(q_row < K.int32(seqlen_q)):
                                with K.Then():
                                    K.ptx["cp.async.ca.shared.global"](
                                        arena.ptr_to([dst]),
                                        workspace.ptr_to([bh_q + K.cast(q_row, "int64")]),
                                        4,
                                        4,
                                    )
                                with K.Else():
                                    K.ptx.st.shared.b32(arena.ptr_to([dst]), K.uint32(0))
                    K.ptx["cp.async.mbarrier.arrive.noinc.shared.b64"](
                        sum_pipe.full.buf.ptr_to([sum_stage])
                    )
                    sum_prod.advance()
                    K.assign(remaining, remaining - K.int32(2))

                load_pair(True)
                with K.While(remaining > K.int32(0)):
                    load_pair(False)

            with r_mma:
                K.ptx[TMEM_ALLOC](
                    K.cuda.cvta_generic_to_shared(tmem_mailbox.ptr_to([0])), K.uint32(TMEM_COLUMNS)
                )
                K.ptx.bar.sync(K.uint32(BAR_TMEM[0]), K.uint32(BAR_TMEM[1]))
                tcol = K.local_scalar("uint32")
                K.ptx.ld.shared.b32(tcol, tmem_mailbox.ptr_to([0]))
                t_dk = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(TMEM_DK))
                t_dv = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(TMEM_DV))
                t_dp = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(TMEM_DP))
                t_s = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(TMEM_S))
                elected = K.local_scalar("uint32", init=K.cuda.elect_sync())

                desc_k = _desc_base(1)
                desc_mn_wide = _desc_base(1024)
                desc_mn_k = _desc_base(512)
                desc_mn_narrow = _desc_base(0)
                d_k_k = _desc_at(desc_k, smem_base + K.uint32(OFF_SK))
                d_v_k = _desc_at(desc_k, smem_base + K.uint32(OFF_SV))
                d_do_k = _desc_at(desc_k, smem_base + K.uint32(OFF_SDO))
                d_do_mn = _desc_at(desc_mn_wide, smem_base + K.uint32(OFF_SDO))
                d_p_mn = _desc_at(desc_mn_narrow, smem_base + K.uint32(OFF_SP))
                d_ds_k = _desc_at(desc_k, smem_base + K.uint32(OFF_SDS))
                d_ds_mn = _desc_at(desc_mn_narrow, smem_base + K.uint32(OFF_SDS))
                d_k_mn = _desc_at(desc_mn_k, smem_base + K.uint32(OFF_SK))

                q_cons = K.PipelineState(2, phase=0)
                q_release = K.PipelineState(2, phase=0)
                do_cons = K.PipelineState(1, phase=0)
                s_prod = K.PipelineState(1, phase=0)
                dp_prod = K.PipelineState(1, phase=0)
                dq_prod = K.PipelineState(1, phase=0)
                p_cons = K.PipelineState(1, phase=0)
                ds_cons = K.PipelineState(1, phase=0)
                dkdv_prod = K.PipelineState(2, phase=0)

                q_pipe.full.wait(q_cons.stage, q_cons.phase)
                s_pipe.empty.wait(s_prod.stage, s_prod.phase ^ 1)
                d_q_k = _desc_at(
                    desc_k, smem_base + K.uint32(OFF_SQ) + q_cons.stage * K.uint32(32768)
                )
                _mma_chain(
                    t_s,
                    d_q_k,
                    d_k_k,
                    ID_QK,
                    (0, 2, 4, 6, 1024, 1026, 1028, 1030),
                    (0, 2, 4, 6, 512, 514, 516, 518),
                    K.uint32(0),
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
                    K.uint32(0),
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
                    K.uint32(0),
                    elected,
                )
                p_pipe.empty.arrive(p_cons.stage, pred=elected)
                p_cons.advance()
                do_pipe.empty.arrive(do_cons.stage, pred=elected)
                do_cons.advance()

                pairs = (count + K.int32(1)) // K.int32(2)
                pair = K.local_scalar("int32", init=K.int32(1))
                dk_accumulate = K.local_scalar("uint32", init=K.uint32(0))
                with K.While(pair < pairs):
                    q_pipe.full.wait(q_cons.stage, q_cons.phase)
                    s_pipe.empty.wait(s_prod.stage, s_prod.phase ^ 1)
                    d_q_k = _desc_at(
                        desc_k, smem_base + K.uint32(OFF_SQ) + q_cons.stage * K.uint32(32768)
                    )
                    _mma_chain(
                        t_s,
                        d_q_k,
                        d_k_k,
                        ID_QK,
                        (0, 2, 4, 6, 1024, 1026, 1028, 1030),
                        (0, 2, 4, 6, 512, 514, 516, 518),
                        K.uint32(0),
                        elected,
                    )
                    q_cons.advance()
                    s_pipe.full.arrive(s_prod.stage, pred=elected)
                    s_prod.advance()

                    ds_pipe.full.wait(ds_cons.stage, ds_cons.phase)
                    dp_pipe.empty.wait(dp_prod.stage, dp_prod.phase ^ 1)
                    _mma_chain(
                        K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(TMEM_DQ)),
                        d_ds_k,
                        d_k_mn,
                        ID_DQ,
                        (0, 2, 4, 6),
                        (0, 128, 256, 384),
                        K.uint32(0),
                        elected,
                    )
                    dq_pipe.full.arrive(dq_prod.stage, pred=elected)
                    dq_prod.advance()

                    d_q_mn = _desc_at(
                        desc_mn_wide,
                        smem_base + K.uint32(OFF_SQ) + q_release.stage * K.uint32(32768),
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
                    K.assign(dk_accumulate, K.uint32(1))
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
                        K.uint32(0),
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
                        K.uint32(1),
                        elected,
                    )
                    p_pipe.empty.arrive(p_cons.stage, pred=elected)
                    p_cons.advance()
                    do_pipe.empty.arrive(do_cons.stage, pred=elected)
                    do_cons.advance()
                    K.assign(pair, pair + K.int32(1))

                dkdv_pipe.empty.wait(dkdv_prod.stage, dkdv_prod.phase ^ 1)
                dkdv_pipe.full.arrive(dkdv_prod.stage, pred=elected)
                dkdv_prod.advance()
                dkdv_pipe.empty.wait(dkdv_prod.stage, dkdv_prod.phase ^ 1)
                ds_pipe.full.wait(ds_cons.stage, ds_cons.phase)
                d_q_mn = _desc_at(
                    desc_mn_wide, smem_base + K.uint32(OFF_SQ) + q_release.stage * K.uint32(32768)
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
                    K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(TMEM_DQ)),
                    d_ds_k,
                    d_k_mn,
                    ID_DQ,
                    (0, 2, 4, 6),
                    (0, 128, 256, 384),
                    K.uint32(0),
                    elected,
                )
                dq_pipe.full.arrive(dq_prod.stage, pred=elected)
                dq_prod.advance()
                q_pipe.empty.arrive(q_release.stage, pred=elected)
                q_release.advance()
                ds_pipe.empty.arrive(ds_cons.stage, pred=elected)
                ds_cons.advance()

            with r_compute:
                K.ptx.setmaxnreg.inc.sync.aligned.u32(K.uint32(128))
                K.ptx.bar.sync(K.uint32(BAR_TMEM[0]), K.uint32(BAR_TMEM[1]))
                tcol = K.local_scalar("uint32")
                K.ptx.ld.shared.b32(tcol, tmem_mailbox.ptr_to([0]))
                ctid = K.tid_in_role()
                crow = ctid % K.int32(128)
                wg = ctid // K.int32(128)
                row_group = (crow // K.int32(32)) * K.int32(32)
                scale_log2 = K.local_scalar("float32")
                K.ptx.mul.f32(scale_log2, softmax_scale, K.float32(1.4426950408889634))
                block_size = K.local_scalar("int32", init=K.int32(64))
                if has_block_sizes:
                    K.ptx.ld.global_.s32(
                        block_size, block_sizes.ptr_to([batch_idx * K.int32(tasks) + task])
                    )

                s_cons = K.PipelineState(1, phase=0)
                lse_cons = K.PipelineState(1, phase=0)
                p_prod = K.PipelineState(1, phase=0)
                sum_cons = K.PipelineState(1, phase=0)
                dp_cons = K.PipelineState(1, phase=0)
                ds_prod = K.PipelineState(1, phase=0)
                dkdv_cons = K.PipelineState(2, phase=0)
                pair = K.local_scalar("int32", init=K.int32(0))
                pairs = (count + K.int32(1)) // K.int32(2)
                with K.While(pair < pairs):
                    s_pipe.full.wait(s_cons.stage, s_cons.phase)
                    lse_pipe.full.wait(lse_cons.stage, lse_cons.phase)
                    p_pipe.empty.wait(p_prod.stage, p_prod.phase ^ 1)
                    scores = K.alloc_local((32,), "float32")
                    for issue in range(2):
                        address = K.cuda.get_tmem_addr(
                            tcol, row_group, K.int32(TMEM_S + issue * 32) + wg * K.int32(16)
                        )
                        _tmem_load16(scores, issue * 16, address)
                    if has_block_sizes:
                        for j in range(32):
                            col = wg * K.int32(16) + K.int32((j // 16) * 32 + j % 16)
                            live = K.local_scalar("uint32")
                            K.ptx["setp.lt.s32"](live, col, block_size)
                            bits = K.local_scalar("uint32")
                            K.ptx.selp.b32(
                                bits,
                                K.reinterpret(K.u32, scores[j]),
                                K.uint32(0xFF800000),
                                K.ptx.pred(live),
                            )
                            K.assign(scores[j], K.reinterpret(K.f32, bits))
                    lse_value = K.local_scalar("float32")
                    K.ptx.ld.shared_.b32(lse_value, arena.ptr_to([OFF_SLSE + crow * K.int32(4)]))
                    for j in range(0, 32, 2):
                        lo, hi = _packed_fma(
                            scores[j], scores[j + 1], scale_log2, scale_log2, lse_value, lse_value
                        )
                        K.ptx.ex2.approx.ftz.f32(scores[j], lo)
                        K.ptx.ex2.approx.ftz.f32(scores[j + 1], hi)
                    packed_p = K.alloc_local((16,), "uint32")
                    for j in range(16):
                        K.ptx["cvt.rn.bf16x2.f32"](packed_p[j], scores[2 * j + 1], scores[2 * j])
                    K.ptx["tcgen05.wait::ld.sync.aligned"]()
                    K.ptx.bar.sync(K.uint32(BAR_COMPUTE[0]), K.uint32(BAR_COMPUTE[1]))
                    K.ptx["tcgen05.wait::ld.sync.aligned"]()
                    for group in range(4):
                        col = wg * K.int32(16) + K.int32((group // 2) * 32 + (group % 2) * 8)
                        K.ptx.st.shared.v4.b32(
                            arena.ptr_to([_tile_byte(OFF_SP, crow, col)]),
                            packed_p[group * 4],
                            packed_p[group * 4 + 1],
                            packed_p[group * 4 + 2],
                            packed_p[group * 4 + 3],
                        )
                    K.ptx.fence.proxy.async_.shared__cta()
                    p_pipe.full.arrive(p_prod.stage)
                    p_prod.advance()
                    s_pipe.empty.arrive(s_cons.stage)
                    s_cons.advance()
                    lse_pipe.empty.arrive(lse_cons.stage)
                    lse_cons.advance()

                    sum_pipe.full.wait(sum_cons.stage, sum_cons.phase)
                    dp_pipe.full.wait(dp_cons.stage, dp_cons.phase)
                    ds_pipe.empty.wait(ds_prod.stage, ds_prod.phase ^ 1)
                    dp_values = K.alloc_local((32,), "float32")
                    for issue in range(2):
                        address = K.cuda.get_tmem_addr(
                            tcol, row_group, K.int32(TMEM_DP + issue * 32) + wg * K.int32(16)
                        )
                        _tmem_load16(dp_values, issue * 16, address)
                    sum_value = K.local_scalar("float32")
                    K.ptx.ld.shared_.b32(sum_value, arena.ptr_to([OFF_SSUM + crow * K.int32(4)]))
                    for j in range(0, 32, 2):
                        lo, hi = _packed_binary(
                            "add.rn.f32x2", dp_values[j], dp_values[j + 1], sum_value, sum_value
                        )
                        lo, hi = _packed_binary("mul.rn.f32x2", lo, hi, scores[j], scores[j + 1])
                        K.assign(dp_values[j], lo)
                        K.assign(dp_values[j + 1], hi)
                    packed_ds = K.alloc_local((16,), "uint32")
                    for j in range(16):
                        K.ptx["cvt.rn.bf16x2.f32"](
                            packed_ds[j], dp_values[2 * j + 1], dp_values[2 * j]
                        )
                    K.ptx["tcgen05.wait::ld.sync.aligned"]()
                    dp_pipe.empty.arrive(dp_cons.stage)
                    dp_cons.advance()
                    for group in range(4):
                        col = wg * K.int32(16) + K.int32((group // 2) * 32 + (group % 2) * 8)
                        K.ptx.st.shared.v4.b32(
                            arena.ptr_to([_tile_byte(OFF_SDS, crow, col)]),
                            packed_ds[group * 4],
                            packed_ds[group * 4 + 1],
                            packed_ds[group * 4 + 2],
                            packed_ds[group * 4 + 3],
                        )
                    K.ptx.fence.proxy.async_.shared__cta()
                    ds_pipe.full.arrive(ds_prod.stage)
                    ds_prod.advance()
                    sum_pipe.empty.arrive(sum_cons.stage)
                    sum_cons.advance()
                    K.assign(pair, pair + K.int32(1))

                bh = K.cast(batch_idx * K.int32(heads) + head, "int64")
                for is_dk in (False, True):
                    dkdv_pipe.full.wait(dkdv_cons.stage, dkdv_cons.phase)
                    values = K.alloc_local((32,), "float32")
                    tmem_offset = TMEM_DK if is_dk else TMEM_DV
                    for issue in range(2):
                        address = K.cuda.get_tmem_addr(
                            tcol, row_group, K.int32(tmem_offset + issue * 32) + wg * K.int32(16)
                        )
                        _tmem_load16(values, issue * 16, address)
                    K.ptx["tcgen05.wait::ld.sync.aligned"]()
                    for j in range(32):
                        col = wg * K.int32(16) + K.int32((j // 16) * 32 + j % 16)
                        seq = task * K.int32(64) + col
                        with K.If(seq < K.int32(seqlen_kv)), K.Then():
                            base = dk_base if is_dk else dv_base
                            index = (
                                K.int64(base)
                                + bh * K.int64(k8 * 128)
                                + K.cast(seq, "int64") * K.int64(128)
                                + K.cast(crow, "int64")
                            )
                            old = K.local_scalar("float32")
                            K.ptx.atom.global_.add.f32(old, workspace.ptr_to([index]), values[j])
                    dkdv_pipe.empty.arrive(dkdv_cons.stage)
                    dkdv_cons.advance()
                K.ptx.bar.sync(K.uint32(BAR_DEALLOC[0]), K.uint32(BAR_DEALLOC[1]))
                with K.If(K.warp_id() == K.int32(8)), K.Then():
                    K.ptx[TMEM_DEALLOC](tcol, K.uint32(TMEM_COLUMNS))

            with r_reduce:
                K.ptx.bar.sync(K.uint32(BAR_TMEM[0]), K.uint32(BAR_TMEM[1]))
                tcol = K.local_scalar("uint32")
                K.ptx.ld.shared.b32(tcol, tmem_mailbox.ptr_to([0]))
                rtid = K.tid_in_role()
                row_group = (rtid // K.int32(32)) * K.int32(32)
                dq_cons = K.PipelineState(1, phase=0)
                store_stage = K.local_scalar("int32", init=K.int32(0))
                edge = K.local_scalar("int32", init=K.int32(0))
                bh_edge = K.cast(batch_idx * K.int32(heads) + head, "int64") * edge_stride
                remaining = K.local_scalar("int32", init=count)
                with K.While(remaining > K.int32(0)):
                    dq_pipe.full.wait(dq_cons.stage, dq_cons.phase)
                    q0 = _load_i32(bucketed_indices, bh_edge + K.cast(begin + edge, "int64"))
                    K.assign(edge, edge + K.int32(1))
                    q1 = K.local_scalar("int32", init=K.int32((seqlen_q + 63) // 64))
                    with K.If(edge < count), K.Then():
                        K.assign(
                            q1, _load_i32(bucketed_indices, bh_edge + K.cast(begin + edge, "int64"))
                        )
                    K.assign(edge, edge + K.int32(1))
                    values = K.alloc_local((128,), "float32")
                    for issue in range(4):
                        address = K.cuda.get_tmem_addr(
                            tcol, row_group, K.int32(TMEM_DQ + issue * 32)
                        )
                        _tmem_load32(values, issue * 32, address)
                    K.ptx["tcgen05.wait::ld.sync.aligned"]()
                    dq_pipe.empty.arrive(dq_cons.stage)
                    dq_cons.advance()
                    for chunk in range(4):
                        with K.If(K.warp_id() == K.int32(0)), K.Then():
                            K.ptx.cp.async_.bulk.wait_group.read(1)
                        K.ptx.bar.sync(K.uint32(BAR_REDUCE[0]), K.uint32(BAR_REDUCE[1]))
                        pair_slot = rtid // K.int32(64)
                        row = rtid % K.int32(64)
                        for vec in range(8):
                            raw = (
                                K.int32(OFF_SDQ)
                                + store_stage * K.int32(16384)
                                + pair_slot * K.int32(8192)
                                + row * K.int32(128)
                                + K.int32(16 * (7 - vec))
                            )
                            address = K.bitwise_xor(
                                raw, K.bitwise_and(K.shift_right(raw, K.int32(3)), K.int32(0x70))
                            )
                            # The source fragment and its shared CopyAtom both
                            # walk the eight vectors in reverse.  Reversing the
                            # register vector together with the address keeps
                            # logical dQ dimensions in ascending order.
                            base = chunk * 32 + (7 - vec) * 4
                            K.ptx.st.shared.v4.b32(
                                arena.ptr_to([address]),
                                values[base],
                                values[base + 1],
                                values[base + 2],
                                values[base + 3],
                            )
                        K.ptx.fence.proxy.async_.shared__cta()
                        K.ptx.bar.sync(K.uint32(BAR_REDUCE[0]), K.uint32(BAR_REDUCE[1]))
                        with K.If(K.warp_id() == K.int32(0)), K.Then():
                            leader = K.local_scalar("uint32", init=K.cuda.elect_sync())
                            with K.If(leader != K.uint32(0)), K.Then():
                                K.ptx[TMA_REDUCE](
                                    K.address_of(dq_map),
                                    K.int32(chunk * 32),
                                    q0 * K.int32(64),
                                    head,
                                    batch_idx,
                                    arena.ptr_to([OFF_SDQ + store_stage * K.int32(16384)]),
                                    TMA_CACHE,
                                )
                                K.ptx[TMA_REDUCE](
                                    K.address_of(dq_map),
                                    K.int32(chunk * 32),
                                    q1 * K.int32(64),
                                    head,
                                    batch_idx,
                                    arena.ptr_to([OFF_SDQ + store_stage * K.int32(16384) + 8192]),
                                    TMA_CACHE,
                                )
                            K.ptx.cp.async_.bulk.commit_group()
                        K.assign(store_stage, K.Select(store_stage == K.int32(1), 0, 1))
                    K.assign(remaining, remaining - K.int32(2))
                K.ptx.cp.async_.bulk.wait_group.read(0)

            with r_empty:
                pass
        required_block_size.__exit__(None, None, None)

    @K.kernel(
        warps=4,
        arch="sm_100a",
        min_blocks_per_sm=1,
        grid=((max(seqlen_q, seqlen_kv) + 7) // 8, heads, batch),
    )
    def convert(
        workspace: K.gptr[K.f32],
        dq: K.gptr[K.bf16],
        dk: K.gptr[K.bf16],
        dv: K.gptr[K.bf16],
        softmax_scale: K.f32,
    ):
        required_block_size = K.attr({"tirx.required_block_size": 1})
        required_block_size.__enter__()
        seq_tile, head, batch_idx = K.cta_id()
        tid = K.thread_id()
        tidx = tid % K.int32(16)
        tidy = tid // K.int32(16)
        seq = seq_tile * K.int32(8) + tidy
        bh = K.cast(batch_idx * K.int32(heads) + head, "int64")
        for group in range(2):
            dim = (tidx + K.int32(group * 16)) * K.int32(4)
            with K.If(seq < K.int32(seqlen_q)), K.Then():
                source = (
                    K.int64(dq_base)
                    + bh * K.int64(q8 * 128)
                    + K.cast(seq, "int64") * K.int64(128)
                    + K.cast(dim, "int64")
                )
                values = K.alloc_local((4,), "float32")
                K.ptx.ld.global_.v4.b32(
                    values[0], values[1], values[2], values[3], workspace.ptr_to([source])
                )
                scaled = K.alloc_local((4,), "float32")
                for pair in range(2):
                    lo, hi = _packed_binary(
                        "mul.rn.f32x2",
                        values[pair * 2],
                        values[pair * 2 + 1],
                        softmax_scale,
                        softmax_scale,
                    )
                    K.assign(scaled[pair * 2], lo)
                    K.assign(scaled[pair * 2 + 1], hi)
                out0 = K.local_scalar("uint32")
                out1 = K.local_scalar("uint32")
                K.ptx["cvt.rn.bf16x2.f32"](out0, scaled[1], scaled[0])
                K.ptx["cvt.rn.bf16x2.f32"](out1, scaled[3], scaled[2])
                output = (bh * K.int64(seqlen_q) + K.cast(seq, "int64")) * K.int64(128) + K.cast(
                    dim, "int64"
                )
                K.ptx.st.global_.v2.b32(dq.ptr_to([output]), out0, out1)
            with K.If(seq < K.int32(seqlen_kv)), K.Then():
                dk_source = (
                    K.int64(dk_base)
                    + bh * K.int64(k8 * 128)
                    + K.cast(seq, "int64") * K.int64(128)
                    + K.cast(dim, "int64")
                )
                dv_source = (
                    K.int64(dv_base)
                    + bh * K.int64(k8 * 128)
                    + K.cast(seq, "int64") * K.int64(128)
                    + K.cast(dim, "int64")
                )
                kvals = K.alloc_local((4,), "float32")
                vvals = K.alloc_local((4,), "float32")
                K.ptx.ld.global_.v4.b32(
                    kvals[0], kvals[1], kvals[2], kvals[3], workspace.ptr_to([dk_source])
                )
                K.ptx.ld.global_.v4.b32(
                    vvals[0], vvals[1], vvals[2], vvals[3], workspace.ptr_to([dv_source])
                )
                kscaled = K.alloc_local((4,), "float32")
                for pair in range(2):
                    lo, hi = _packed_binary(
                        "mul.rn.f32x2",
                        kvals[pair * 2],
                        kvals[pair * 2 + 1],
                        softmax_scale,
                        softmax_scale,
                    )
                    K.assign(kscaled[pair * 2], lo)
                    K.assign(kscaled[pair * 2 + 1], hi)
                k0 = K.local_scalar("uint32")
                k1 = K.local_scalar("uint32")
                v0 = K.local_scalar("uint32")
                v1 = K.local_scalar("uint32")
                K.ptx["cvt.rn.bf16x2.f32"](k0, kscaled[1], kscaled[0])
                K.ptx["cvt.rn.bf16x2.f32"](k1, kscaled[3], kscaled[2])
                K.ptx["cvt.rn.bf16x2.f32"](v0, vvals[1], vvals[0])
                K.ptx["cvt.rn.bf16x2.f32"](v1, vvals[3], vvals[2])
                output = (bh * K.int64(seqlen_kv) + K.cast(seq, "int64")) * K.int64(128) + K.cast(
                    dim, "int64"
                )
                K.ptx.st.global_.v2.b32(dk.ptr_to([output]), k0, k1)
                K.ptx.st.global_.v2.b32(dv.ptr_to([output]), v0, v1)
        required_block_size.__exit__(None, None, None)

    return sum_odo.func, bwd.func, convert.func
