# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Source-shaped SM100 blk128 block-sparse-attention backward kernels."""

import tirx_kernels.tirx_lite as txl

WARPS = 16
THREADS = 512
TMEM_COLUMNS = 512
BAR_EPILOGUE_0 = (1, 160)
BAR_EPILOGUE_1 = (2, 160)
BAR_COMPUTE = (3, 256)
BAR_REDUCE = (4, 128)
BAR_TMEM = (5, 416)

TMEM_S = 0
TMEM_DV = 128

MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
TMEM_LD16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
TMEM_LD32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TMEM_ST16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
TMEM_RELINQUISH = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
TMA_G2S = (
    "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
)
TMA_S2G = "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint"
TMA_REDUCE = "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
TMA_CACHE = txl.uint64(0)


def _xor(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return a ^ b
    return txl.bitwise_xor(a, b)


def _tile_elem(row, dim):
    return (dim // 64) * 8192 + _xor(row * 64 + dim % 64, (row % 8) * 8)


def _tile_byte(base, row, dim):
    return base + 2 * _tile_elem(row, dim)


def _epi_bf16_byte(base, wg, row, dim, columns):
    """Source make_smem_layout_epi address for one workgroup's BF16 tile."""
    linear = (
        base + wg * txl.int32(128 * columns * 2) + row * txl.int32(columns * 2) + txl.int32(dim * 2)
    )
    mask = txl.int32(columns * 2 - 16)
    return _xor(linear, txl.bitwise_and(linear // txl.int32(8), mask))


def _desc_base(ldo, sdo=64, swizzle=3):
    arrangement = {0: 0, 1: 6, 2: 4, 3: 2, 4: 1}[swizzle]
    # Cute's source layout reports semantic ldo=512. Its line-info PTX
    # encodes that as low-half field 0x04000000 (the BF16 byte step), while
    # sdo remains 64 in descriptor units.
    encoded_ldo = ldo * 2
    value = ((encoded_ldo & 0x3FFF) << 16) | ((sdo & 0x3FFF) << 32) | (1 << 46)
    return (value | ((arrangement & 0x7) << 61)) & 0xFFFFFFFFFFFFFFFF


def _desc_at(base, shared_address):
    field = txl.cast(
        txl.bitwise_and(txl.shift_right(shared_address, txl.uint32(4)), txl.uint32(0x3FFF)),
        "uint64",
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


def _mma_tmem_chain(dest, tmem_a, b_desc, idesc, accumulate, pred):
    flag = txl.local_scalar("uint32", init=txl.cast(accumulate, "uint32"))
    for phase in range(8):
        _mma(
            dest,
            txl.cast(tmem_a + txl.uint32(phase * 8), "uint32"),
            _desc_add16(b_desc, phase * 128),
            idesc,
            flag,
            pred,
        )
        txl.assign(flag, txl.uint32(1))


def _tmem_store16(src, base, address):
    txl.ptx[TMEM_ST16](txl.cast(address, "uint32"), *(src[base + i] for i in range(16)))


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


class _PipelinePair:
    """Pair public K barrier objects without imposing a constructor epoch."""

    def __init__(self, full, empty):
        self.full = full
        self.empty = empty


def _issue_tma_tile(desc, arena, dst, seq0, head, batch_idx, barrier, head_dim):
    for feature_half in range(head_dim // 64):
        txl.ptx[TMA_G2S](
            arena.ptr_to([dst + feature_half * 16384]),
            txl.address_of(desc),
            txl.int32(feature_half * 64),
            seq0,
            head,
            batch_idx,
            txl.cuda.cvta_generic_to_shared(barrier),
            TMA_CACHE,
        )


def _bulk_stats(arena, dst, src, barrier, pred):
    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(barrier, txl.uint32(512), pred=pred)
    txl.ptx["cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"](
        arena.ptr_to([dst]),
        src,
        txl.uint32(512),
        txl.cuda.cvta_generic_to_shared(barrier),
        pred=pred,
    )


def _tcgen_commit(barrier, pred):
    txl.ptx["tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"](
        barrier, pred=pred
    )


def _bar_arrive(barrier):
    txl.ptx.mbarrier.arrive.shared.b64(barrier, txl.uint32(1))


def _tmem_load32(dst, base, address):
    txl.ptx[TMEM_LD32](*(dst[base + i] for i in range(32)), txl.cast(address, "uint32"))


def get_kernel(**config):
    batch = int(config["batch"])
    heads = int(config["num_heads"])
    seqlen_q = int(config["seqlen_q"])
    seqlen_kv = int(config["seqlen_kv"])
    head_dim = int(config["head_dim"])
    bshd = config["tensor_layout"] == "bshd"
    assert head_dim in (64, 128)
    q_blocks = (seqlen_q + 127) // 128
    bucket = config.get("bucket_size_blocks")
    if bucket is None:
        bucket = 256 if q_blocks >= 4096 and heads <= 1 else (512 if q_blocks >= 2048 else 384)
    groups = (q_blocks + int(bucket) - 1) // int(bucket)
    tasks = (seqlen_kv + 127) // 128
    q128 = q_blocks * 128
    k128 = tasks * 128
    bh_count = batch * heads
    sum_plane = bh_count * q128
    dq_base = 2 * sum_plane
    dk_base = dq_base + bh_count * q128 * head_dim
    dv_base = dk_base + bh_count * k128 * head_dim

    direct_dkv = groups == 1
    off_sq = 1024
    if head_dim == 64:
        off_sk, off_sv, off_sdo = 33792, 50176, 66560
        off_sds = 82944 if direct_dkv else 99328
        off_slse = 115712 if direct_dkv else 132096
        off_ssum, off_sdq = off_slse + 1024, off_slse + 2048
        shared_bytes = 150528 if direct_dkv else 166912
    else:
        off_sk, off_sv, off_sdo = 66560, 99328, 132096
        off_sds, off_slse, off_ssum, off_sdq = 164864, 197632, 198656, 199680
        shared_bytes = 232448
    tmem_dp = 128 + head_dim
    tmem_dq = tmem_dp
    tmem_ds = tmem_dp
    tmem_dk = 256 + head_dim
    id_qk = 0x08200490
    id_dkdv = 0x08110490 if head_dim == 64 else 0x08210490
    id_dq = 0x08118490 if head_dim == 64 else 0x08218490

    @txl.kernel(
        warps=8, arch="sm_100a", min_blocks_per_sm=1, grid=((seqlen_q + 127) // 128, heads, batch)
    )
    def preprocess(
        o: txl.gptr[txl.bf16],
        do: txl.gptr[txl.bf16],
        lse: txl.gptr[txl.f32],
        workspace: txl.gptr[txl.f32],
    ):
        required_block_size = txl.attr({"tirx.required_block_size": 1})
        required_block_size.__enter__()
        txl.ptx.griddepcontrol.wait()
        q_tile, head, batch_idx = txl.cta_id()
        tid = txl.thread_id()
        threads_per_row = head_dim // 8
        rows_per_wave = 256 // threads_per_row
        row_lane = tid // txl.int32(threads_per_row)
        dim8 = (tid % txl.int32(threads_per_row)) * txl.int32(8)
        membermask = txl.local_scalar("uint32", init=txl.tvm_warp_activemask())
        for row_repeat in range(128 // rows_per_wave):
            q_idx = q_tile * txl.int32(128) + row_lane + txl.int32(row_repeat * rows_per_wave)
            acc = txl.local_scalar("float32", init=txl.float32(0.0))
            with txl.If(q_idx < txl.int32(seqlen_q)), txl.Then():
                row = (
                    (
                        (
                            txl.cast(batch_idx, "int64") * txl.int64(seqlen_q)
                            + txl.cast(q_idx, "int64")
                        )
                        * txl.int64(heads)
                        + txl.cast(head, "int64")
                    )
                    if bshd
                    else (
                        (txl.cast(batch_idx, "int64") * txl.int64(heads) + txl.cast(head, "int64"))
                        * txl.int64(seqlen_q)
                        + txl.cast(q_idx, "int64")
                    )
                ) * txl.int64(head_dim)
                o_words = txl.alloc_local((4,), "uint32")
                do_words = txl.alloc_local((4,), "uint32")
                offset = row + txl.cast(dim8, "int64")
                txl.ptx.ld.global_.v4.b32(
                    o_words[0], o_words[1], o_words[2], o_words[3], o.ptr_to([offset])
                )
                txl.ptx.ld.global_.v4.b32(
                    do_words[0], do_words[1], do_words[2], do_words[3], do.ptr_to([offset])
                )
                for pair in range(4):
                    o_lo = txl.local_scalar("uint16")
                    o_hi = txl.local_scalar("uint16")
                    d_lo = txl.local_scalar("uint16")
                    d_hi = txl.local_scalar("uint16")
                    txl.ptx.mov.b32(o_lo, o_hi, o_words[pair])
                    txl.ptx.mov.b32(d_lo, d_hi, do_words[pair])
                    o0 = txl.local_scalar("float32")
                    o1 = txl.local_scalar("float32")
                    d0 = txl.local_scalar("float32")
                    d1 = txl.local_scalar("float32")
                    txl.ptx["cvt.f32.bf16"](o0, o_lo)
                    txl.ptx["cvt.f32.bf16"](o1, o_hi)
                    txl.ptx["cvt.f32.bf16"](d0, d_lo)
                    txl.ptx["cvt.f32.bf16"](d1, d_hi)
                    p0, p1 = _packed_binary("mul.f32x2", o0, o1, d0, d1)
                    fragment = txl.local_scalar("float32")
                    txl.ptx.add.f32(fragment, p0, p1)
                    txl.ptx.add.f32(acc, acc, fragment)
            total = acc
            for lane_xor in (1, 2, 4):
                peer = _shfl_bfly_f32(total, lane_xor, membermask)
                combined = txl.local_scalar("float32")
                txl.ptx.add.f32(combined, total, peer)
                txl.assign(total, combined)
            if threads_per_row == 16:
                peer = _shfl_bfly_f32(total, 8, membermask)
                combined = txl.local_scalar("float32")
                txl.ptx.add.f32(combined, total, peer)
                txl.assign(total, combined)
            with txl.If((tid % txl.int32(threads_per_row)) == txl.int32(0)), txl.Then():
                bhq = (
                    txl.cast(batch_idx, "int64") * txl.int64(heads) + txl.cast(head, "int64")
                ) * txl.int64(q128) + txl.cast(q_idx, "int64")
                txl.ptx.st.global_.b32(workspace.ptr_to([bhq]), total)
                scaled = txl.local_scalar("float32", init=txl.float32(0.0))
                with txl.If(q_idx < txl.int32(seqlen_q)), txl.Then():
                    lse_value = txl.local_scalar("float32")
                    lse_index = (
                        txl.cast(batch_idx, "int64") * txl.int64(heads) + txl.cast(head, "int64")
                    ) * txl.int64(seqlen_q) + txl.cast(q_idx, "int64")
                    txl.ptx.ld.global_.b32(lse_value, lse.ptr_to([lse_index]))
                    txl.ptx.mul.f32(scaled, lse_value, txl.float32(1.4426950408889634))
                    with txl.If(lse_value == txl.float32(float("-inf"))), txl.Then():
                        txl.assign(scaled, txl.float32(0.0))
                txl.ptx.st.global_.b32(workspace.ptr_to([txl.int64(sum_plane) + bhq]), scaled)
        txl.ptx.griddepcontrol.launch_dependents()
        bh = txl.cast(batch_idx, "int64") * txl.int64(heads) + txl.cast(head, "int64")
        for vec in range(head_dim // 8):
            elem = tid * txl.int32(4) + txl.int32(vec * 1024)
            dst = (
                txl.int64(dq_base)
                + bh * txl.int64(q128 * head_dim)
                + txl.cast(q_tile, "int64") * txl.int64(128 * head_dim)
                + txl.cast(elem, "int64")
            )
            txl.ptx.st.global_.v4.b32(
                workspace.ptr_to([dst]), txl.uint32(0), txl.uint32(0), txl.uint32(0), txl.uint32(0)
            )
        required_block_size.__exit__(None, None, None)

    @txl.kernel(
        warps=WARPS, arch="sm_100a", min_blocks_per_sm=1, grid=(tasks * groups, heads, batch)
    )
    def bwd(
        q_map: txl.TensorMap,
        k_map: txl.TensorMap,
        v_map: txl.TensorMap,
        do_map: txl.TensorMap,
        dv_map: txl.TensorMap,
        dk_map: txl.TensorMap,
        dk_output: txl.gptr[txl.bf16],
        dv_output: txl.gptr[txl.bf16],
        bucketed_offsets: txl.gptr[txl.i32],
        bucketed_indices: txl.gptr[txl.i32],
        workspace: txl.gptr[txl.f32],
        edge_stride: txl.i64,
        softmax_scale: txl.f32,
    ):
        required_block_size = txl.attr({"tirx.required_block_size": 1})
        required_block_size.__enter__()
        block, head, batch_idx = txl.cta_id()
        warp = txl.warp_id()
        with txl.If(warp == txl.int32(13)), txl.Then():
            with txl.If(txl.cuda.elect_sync()), txl.Then():
                txl.ptx.prefetch.tensormap(txl.address_of(q_map))
                txl.ptx.prefetch.tensormap(txl.address_of(k_map))
                txl.ptx.prefetch.tensormap(txl.address_of(v_map))
                txl.ptx.prefetch.tensormap(txl.address_of(do_map))
                if direct_dkv:
                    txl.ptx.prefetch.tensormap(txl.address_of(dv_map))
                    txl.ptx.prefetch.tensormap(txl.address_of(dk_map))

        arena = txl.alloc_buffer((shared_bytes,), txl.u8, scope="shared.dyn", align=1024)
        pool = txl.smem_pool(base=arena).pool
        q_pipe = _PipelinePair(txl.TMABar(pool, 2), txl.TCGen05Bar(pool, 2))
        do_pipe = _PipelinePair(txl.TMABar(pool, 1), txl.TCGen05Bar(pool, 1))
        lse_pipe = _PipelinePair(txl.TMABar(pool, 2), txl.MBarrier(pool, 2))
        sum_pipe = _PipelinePair(txl.TMABar(pool, 1), txl.MBarrier(pool, 1))
        s_pipe = _PipelinePair(txl.TCGen05Bar(pool, 1), txl.MBarrier(pool, 1))
        dp_pipe = _PipelinePair(txl.TCGen05Bar(pool, 1), txl.MBarrier(pool, 1))
        ds_pipe = _PipelinePair(txl.MBarrier(pool, 1), txl.TCGen05Bar(pool, 1))
        dkdv_pipe = _PipelinePair(txl.TCGen05Bar(pool, 2), txl.MBarrier(pool, 2))
        dq_pipe = _PipelinePair(txl.TCGen05Bar(pool, 1), txl.MBarrier(pool, 1))

        s_pipe.full.init(1)
        s_pipe.empty.init(8)
        _publish_pipeline_init()
        dp_pipe.full.init(1)
        dp_pipe.empty.init(8)
        _publish_pipeline_init()
        dkdv_pipe.full.init(1)
        dkdv_pipe.empty.init(8)
        _publish_pipeline_init()
        dq_pipe.full.init(1)
        dq_pipe.empty.init(4)
        _publish_pipeline_init()
        ds_pipe.full.init(8)
        ds_pipe.empty.init(1)
        _publish_pipeline_init()
        lse_pipe.full.init(1)
        lse_pipe.empty.init(8)
        sum_pipe.full.init(1)
        sum_pipe.empty.init(8)
        q_pipe.full.init(1)
        q_pipe.empty.init(1)
        do_pipe.full.init(1)
        do_pipe.empty.init(1)
        _publish_pipeline_init()
        tmem_mailbox = pool.alloc((1,), "uint32", align=4)
        assert pool.offset == 196

        q_group = block // txl.int32(tasks)
        task = block - q_group * txl.int32(tasks)
        offsets_base = (
            (batch_idx * txl.int32(heads) + head) * txl.int32(groups) + q_group
        ) * txl.int32(tasks + 1)
        begin = _load_i32(bucketed_offsets, offsets_base + task)
        end = _load_i32(bucketed_offsets, offsets_base + task + txl.int32(1))
        count = end - begin
        work = (count > txl.int32(0)) & (task * txl.int32(128) < txl.int32(seqlen_kv))

        smem_base = txl.local_scalar("uint32")
        txl.assign(smem_base, txl.cuda.cvta_generic_to_shared(arena.ptr_to([0])))
        sp = txl.specialize(chain_dispatch=True)
        # Roles 12-15 share physical warpgroup 3, so all four must execute the
        # collective setmaxnreg transition with the same target.
        r_load = sp.role("load", warps=[13], regs=88)
        r_mma = sp.role("mma", warps=[12], regs=88)
        r_compute = sp.role("compute", warps=list(range(4, 12)), regs=136)
        r_reduce = sp.role("reduce", warps=list(range(4)), regs=152)
        r_idle = sp.role("idle", warps=[14], regs=88)
        r_empty = sp.role("empty", warps=[15], regs=88)

        with r_load:
            leader = txl.local_scalar("uint32", init=txl.cuda.elect_sync())
            q_prod = txl.PipelineState(2, phase=0)
            do_prod = txl.PipelineState(1, phase=0)
            edge = txl.local_scalar("int32", init=txl.int32(0))
            bh = txl.cast(batch_idx, "int64") * txl.int64(heads) + txl.cast(head, "int64")
            bh_edge = bh * edge_stride
            bh_q = bh * txl.int64(q128)

            with txl.While(edge < count):
                q_block = _load_i32(bucketed_indices, bh_edge + txl.cast(begin + edge, "int64"))
                q_block_safe = txl.Select(
                    q_block < txl.int32(q_blocks), q_block, txl.int32(q_blocks - 1)
                )
                first = edge == txl.int32(0)

                q_pipe.empty.wait(q_prod.stage, q_prod.phase ^ 1)
                with txl.If(leader != txl.uint32(0)), txl.Then():
                    q_bar = q_pipe.full.buf.ptr_to([q_prod.stage])
                    q_tx = txl.local_scalar("uint32", init=txl.uint32(head_dim * 256))
                    with txl.If(first), txl.Then():
                        txl.assign(q_tx, txl.uint32(head_dim * 512))
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(q_bar, q_tx)
                    with txl.If(first), txl.Then():
                        _issue_tma_tile(
                            k_map,
                            arena,
                            off_sk,
                            task * txl.int32(128),
                            head,
                            batch_idx,
                            q_bar,
                            head_dim,
                        )
                    _issue_tma_tile(
                        q_map,
                        arena,
                        off_sq + q_prod.stage * txl.int32(128 * head_dim * 2),
                        q_block_safe * txl.int32(128),
                        head,
                        batch_idx,
                        q_bar,
                        head_dim,
                    )
                lse_pipe.empty.wait(q_prod.stage, q_prod.phase ^ 1)
                with txl.If(leader != txl.uint32(0)), txl.Then():
                    lse_bar = lse_pipe.full.buf.ptr_to([q_prod.stage])
                    _bulk_stats(
                        arena,
                        off_slse + q_prod.stage * txl.int32(512),
                        workspace.ptr_to(
                            [
                                txl.int64(sum_plane)
                                + bh_q
                                + txl.cast(q_block_safe, "int64") * txl.int64(128)
                            ]
                        ),
                        lse_bar,
                        txl.bool(True),
                    )
                q_prod.advance()

                do_pipe.empty.wait(do_prod.stage, do_prod.phase ^ 1)
                with txl.If(leader != txl.uint32(0)), txl.Then():
                    do_bar = do_pipe.full.buf.ptr_to([do_prod.stage])
                    do_tx = txl.local_scalar("uint32", init=txl.uint32(head_dim * 256))
                    with txl.If(first), txl.Then():
                        txl.assign(do_tx, txl.uint32(head_dim * 512))
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(do_bar, do_tx)
                    with txl.If(first), txl.Then():
                        _issue_tma_tile(
                            v_map,
                            arena,
                            off_sv,
                            task * txl.int32(128),
                            head,
                            batch_idx,
                            do_bar,
                            head_dim,
                        )
                    _issue_tma_tile(
                        do_map,
                        arena,
                        off_sdo,
                        q_block_safe * txl.int32(128),
                        head,
                        batch_idx,
                        do_bar,
                        head_dim,
                    )
                sum_pipe.empty.wait(do_prod.stage, do_prod.phase ^ 1)
                with txl.If(leader != txl.uint32(0)), txl.Then():
                    sum_bar = sum_pipe.full.buf.ptr_to([do_prod.stage])
                    _bulk_stats(
                        arena,
                        off_ssum,
                        workspace.ptr_to([bh_q + txl.cast(q_block_safe, "int64") * txl.int64(128)]),
                        sum_bar,
                        txl.bool(True),
                    )
                do_prod.advance()
                txl.assign(edge, edge + txl.int32(1))

            with txl.If(work), txl.Then():
                # Source producer_tail drains both Q/LSE ring stages and the
                # single dO/dPsum stage only for a processed tile.
                q_pipe.empty.wait(q_prod.stage, q_prod.phase ^ 1)
                lse_pipe.empty.wait(q_prod.stage, q_prod.phase ^ 1)
                q_prod.advance()
                q_pipe.empty.wait(q_prod.stage, q_prod.phase ^ 1)
                lse_pipe.empty.wait(q_prod.stage, q_prod.phase ^ 1)
                do_pipe.empty.wait(do_prod.stage, do_prod.phase ^ 1)
                sum_pipe.empty.wait(do_prod.stage, do_prod.phase ^ 1)

        with r_mma:
            txl.ptx[TMEM_ALLOC](
                txl.cuda.cvta_generic_to_shared(tmem_mailbox.ptr_to([0])), txl.uint32(TMEM_COLUMNS)
            )
            txl.ptx.bar.sync(txl.uint32(BAR_TMEM[0]), txl.uint32(BAR_TMEM[1]))
            tcol = txl.local_scalar("uint32")
            txl.ptx.ld.shared.b32(tcol, tmem_mailbox.ptr_to([0]))
            t_s = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(TMEM_S))
            t_dv = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(TMEM_DV))
            t_dp = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(tmem_dp))
            t_dq = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(tmem_dq))
            t_ds = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(tmem_ds))
            t_dk = txl.cuda.get_tmem_addr(tcol, txl.uint32(0), txl.uint32(tmem_dk))
            elected = txl.local_scalar("uint32", init=txl.cuda.elect_sync())

            desc = _desc_base(512)
            d_k = _desc_at(desc, smem_base + txl.uint32(off_sk))
            d_v = _desc_at(desc, smem_base + txl.uint32(off_sv))
            d_do = _desc_at(desc, smem_base + txl.uint32(off_sdo))
            d_ds = _desc_at(desc, smem_base + txl.uint32(off_sds))
            k_offsets = (0, 2, 4, 6)
            if head_dim == 128:
                k_offsets = (0, 2, 4, 6, 1024, 1026, 1028, 1030)
            reduce_offsets = (0, 128, 256, 384, 512, 640, 768, 896)

            q_cons = txl.PipelineState(2, phase=0)
            q_release = txl.PipelineState(2, phase=0)
            do_cons = txl.PipelineState(1, phase=0)
            s_prod = txl.PipelineState(1, phase=0)
            dp_prod = txl.PipelineState(1, phase=0)
            dq_prod = txl.PipelineState(1, phase=0)
            ds_cons = txl.PipelineState(1, phase=0)
            dkdv_prod = txl.PipelineState(2, phase=0)

            def q_desc(state):
                return _desc_at(
                    desc,
                    smem_base + txl.uint32(off_sq) + state.stage * txl.uint32(128 * head_dim * 2),
                )

            def issue_qk(state):
                _mma_chain(
                    t_s, d_k, q_desc(state), id_qk, k_offsets, k_offsets, txl.uint32(0), elected
                )

            def issue_dp():
                _mma_chain(t_dp, d_v, d_do, id_qk, k_offsets, k_offsets, txl.uint32(0), elected)

            def issue_dq():
                _mma_chain(
                    t_dq, d_ds, d_k, id_dq, reduce_offsets, reduce_offsets, txl.uint32(0), elected
                )

            with txl.If(work), txl.Then():
                q_pipe.full.wait(q_cons.stage, q_cons.phase)
                s_pipe.empty.wait(s_prod.stage, s_prod.phase ^ 1)
                issue_qk(q_cons)
                q_cons.advance()
                s_pipe.full.arrive(s_prod.stage, pred=elected)
                s_prod.advance()

                do_pipe.full.wait(do_cons.stage, do_cons.phase)
                dp_pipe.empty.wait(dp_prod.stage, dp_prod.phase ^ 1)
                dq_pipe.empty.wait(dq_prod.stage, dq_prod.phase ^ 1)
                issue_dp()
                dp_pipe.full.arrive(dp_prod.stage, pred=elected)
                dp_prod.advance()

                s_pipe.empty.wait(s_prod.stage, s_prod.phase ^ 1)
                _mma_tmem_chain(t_dv, t_s, d_do, id_dkdv, txl.uint32(0), elected)
                do_pipe.empty.arrive(do_cons.stage, pred=elected)
                do_cons.advance()

                edge = txl.local_scalar("int32", init=txl.int32(1))
                dk_accumulate = txl.local_scalar("uint32", init=txl.uint32(0))
                with txl.While(edge < count):
                    q_pipe.full.wait(q_cons.stage, q_cons.phase)
                    issue_qk(q_cons)
                    q_cons.advance()
                    s_pipe.full.arrive(s_prod.stage, pred=elected)
                    s_prod.advance()

                    ds_pipe.full.wait(ds_cons.stage, ds_cons.phase)
                    _mma_tmem_chain(t_dk, t_ds, q_desc(q_release), id_dkdv, dk_accumulate, elected)
                    txl.assign(dk_accumulate, txl.uint32(1))
                    q_pipe.empty.arrive(q_release.stage, pred=elected)
                    q_release.advance()

                    issue_dq()
                    dq_pipe.full.arrive(dq_prod.stage, pred=elected)
                    dq_prod.advance()
                    ds_pipe.empty.arrive(ds_cons.stage, pred=elected)
                    ds_cons.advance()

                    do_pipe.full.wait(do_cons.stage, do_cons.phase)
                    dq_pipe.empty.wait(dq_prod.stage, dq_prod.phase ^ 1)
                    dp_pipe.empty.wait(dp_prod.stage, dp_prod.phase ^ 1)
                    issue_dp()
                    dp_pipe.full.arrive(dp_prod.stage, pred=elected)
                    dp_prod.advance()

                    s_pipe.empty.wait(s_prod.stage, s_prod.phase ^ 1)
                    _mma_tmem_chain(t_dv, t_s, d_do, id_dkdv, txl.uint32(1), elected)
                    do_pipe.empty.arrive(do_cons.stage, pred=elected)
                    do_cons.advance()
                    txl.assign(edge, edge + txl.int32(1))

                s_pipe.full.arrive(s_prod.stage, pred=elected)
                dkdv_pipe.empty.wait(dkdv_prod.stage, dkdv_prod.phase ^ 1)
                dkdv_pipe.full.arrive(dkdv_prod.stage, pred=elected)
                dkdv_prod.advance()
                dkdv_pipe.empty.wait(dkdv_prod.stage, dkdv_prod.phase ^ 1)

                ds_pipe.full.wait(ds_cons.stage, ds_cons.phase)
                _mma_tmem_chain(t_dk, t_ds, q_desc(q_release), id_dkdv, dk_accumulate, elected)
                dkdv_pipe.full.arrive(dkdv_prod.stage, pred=elected)
                dkdv_prod.advance()

                issue_dq()
                dq_pipe.full.arrive(dq_prod.stage, pred=elected)
                dq_prod.advance()
                q_pipe.empty.arrive(q_release.stage, pred=elected)
                q_release.advance()
                ds_pipe.empty.arrive(ds_cons.stage, pred=elected)
                ds_cons.advance()

            txl.ptx[TMEM_RELINQUISH]()
            txl.ptx.bar.sync(txl.uint32(BAR_TMEM[0]), txl.uint32(BAR_TMEM[1]))
            txl.ptx[TMEM_DEALLOC](tcol, txl.uint32(TMEM_COLUMNS))
        with r_compute:
            txl.ptx.bar.sync(txl.uint32(BAR_TMEM[0]), txl.uint32(BAR_TMEM[1]))
            tcol = txl.local_scalar("uint32")
            txl.ptx.ld.shared.b32(tcol, tmem_mailbox.ptr_to([0]))
            ctid = txl.tid_in_role()
            crow = ctid % txl.int32(128)
            # tid_in_role retains the physical CTA thread id.  Source maps
            # physical compute warpgroups 1 and 2 through bit 7, yielding
            # the required logical half order 1, 0.
            wg = txl.bitwise_and(ctid, txl.int32(128)) // txl.int32(128)
            row_group = (crow // txl.int32(32)) * txl.int32(32)
            scale_log2 = txl.local_scalar("float32")
            txl.ptx.mul.f32(scale_log2, softmax_scale, txl.float32(1.4426950408889634))

            s_cons = txl.PipelineState(1, phase=0)
            lse_cons = txl.PipelineState(2, phase=0)
            sum_cons = txl.PipelineState(1, phase=0)
            dp_cons = txl.PipelineState(1, phase=0)
            ds_prod = txl.PipelineState(1, phase=0)
            dkdv_cons = txl.PipelineState(2, phase=0)
            edge = txl.local_scalar("int32", init=txl.int32(0))
            with txl.While(edge < count):
                lse_pipe.full.wait(lse_cons.stage, lse_cons.phase)
                s_pipe.full.wait(s_cons.stage, s_cons.phase)
                scores = txl.alloc_local((64,), "float32")
                for rep in range(2):
                    address = txl.cuda.get_tmem_addr(
                        tcol, row_group, txl.int32(TMEM_S + wg * 32 + rep * 64)
                    )
                    _tmem_load32(scores, rep * 32, address)
                kv_live = task * txl.int32(128) + crow < txl.int32(seqlen_kv)
                for j in range(64):
                    with txl.If(kv_live == txl.bool(False)), txl.Then():
                        txl.assign(scores[j], txl.reinterpret(txl.f32, txl.uint32(0xFF800000)))
                for rep in range(2):
                    for pair in range(16):
                        j = rep * 32 + pair * 2
                        qcol = wg * txl.int32(32) + txl.int32(rep * 64 + pair * 2)
                        lse0 = txl.local_scalar("float32")
                        lse1 = txl.local_scalar("float32")
                        txl.ptx.ld.shared_.b32(
                            lse0,
                            arena.ptr_to(
                                [off_slse + lse_cons.stage * txl.int32(512) + qcol * txl.int32(4)]
                            ),
                        )
                        txl.ptx.ld.shared_.b32(
                            lse1,
                            arena.ptr_to(
                                [
                                    off_slse
                                    + lse_cons.stage * txl.int32(512)
                                    + (qcol + txl.int32(1)) * txl.int32(4)
                                ]
                            ),
                        )
                        neg0 = txl.local_scalar("float32")
                        neg1 = txl.local_scalar("float32")
                        txl.ptx.neg.f32(neg0, lse0)
                        txl.ptx.neg.f32(neg1, lse1)
                        lo, hi = _packed_fma(
                            scores[j], scores[j + 1], scale_log2, scale_log2, neg0, neg1
                        )
                        txl.ptx.ex2.approx.ftz.f32(scores[j], lo)
                        txl.ptx.ex2.approx.ftz.f32(scores[j + 1], hi)
                packed_p = txl.alloc_local((32,), "uint32")
                for j in range(32):
                    txl.ptx["cvt.rn.bf16x2.f32"](packed_p[j], scores[2 * j + 1], scores[2 * j])
                txl.ptx["tcgen05.wait::ld.sync.aligned"]()
                txl.ptx.bar.sync(txl.uint32(BAR_COMPUTE[0]), txl.uint32(BAR_COMPUTE[1]))
                for rep in range(2):
                    _tmem_store16(
                        packed_p,
                        rep * 16,
                        txl.cuda.get_tmem_addr(
                            tcol, row_group, txl.int32(TMEM_S + wg * 16 + rep * 32)
                        ),
                    )
                txl.ptx["tcgen05.wait::st.sync.aligned"]()
                txl.ptx.bar.sync(txl.uint32(BAR_COMPUTE[0]), txl.uint32(BAR_COMPUTE[1]))
                with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                    s_pipe.empty.arrive(s_cons.stage)
                s_cons.advance()
                with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                    lse_pipe.empty.arrive(lse_cons.stage)
                lse_cons.advance()

                sum_pipe.full.wait(sum_cons.stage, sum_cons.phase)
                dp_pipe.full.wait(dp_cons.stage, dp_cons.phase)
                for rep in range(2):
                    dp_values = txl.alloc_local((32,), "float32")
                    address = txl.cuda.get_tmem_addr(
                        tcol, row_group, txl.int32(tmem_dp + wg * 32 + rep * 64)
                    )
                    _tmem_load32(dp_values, 0, address)
                    txl.ptx["tcgen05.wait::ld.sync.aligned"]()
                    txl.ptx.bar.sync(txl.uint32(BAR_COMPUTE[0]), txl.uint32(BAR_COMPUTE[1]))
                    for pair in range(16):
                        j = pair * 2
                        qcol = wg * txl.int32(32) + txl.int32(rep * 64 + pair * 2)
                        sum0 = txl.local_scalar("float32")
                        sum1 = txl.local_scalar("float32")
                        txl.ptx.ld.shared_.b32(sum0, arena.ptr_to([off_ssum + qcol * txl.int32(4)]))
                        txl.ptx.ld.shared_.b32(
                            sum1, arena.ptr_to([off_ssum + (qcol + txl.int32(1)) * txl.int32(4)])
                        )
                        lo, hi = _packed_binary(
                            "sub.rn.f32x2", dp_values[j], dp_values[j + 1], sum0, sum1
                        )
                        lo, hi = _packed_binary(
                            "mul.rn.f32x2", lo, hi, scores[rep * 32 + j], scores[rep * 32 + j + 1]
                        )
                        txl.assign(dp_values[j], lo)
                        txl.assign(dp_values[j + 1], hi)
                    packed_ds = txl.alloc_local((16,), "uint32")
                    for j in range(16):
                        txl.ptx["cvt.rn.bf16x2.f32"](
                            packed_ds[j], dp_values[2 * j + 1], dp_values[2 * j]
                        )
                    if rep == 0:
                        ds_pipe.empty.wait(ds_prod.stage, ds_prod.phase ^ 1)
                    _tmem_store16(
                        packed_ds,
                        0,
                        txl.cuda.get_tmem_addr(
                            tcol, row_group, txl.int32(tmem_ds + wg * 16 + rep * 32)
                        ),
                    )
                    for group in range(4):
                        qcol = wg * txl.int32(32) + txl.int32(rep * 64 + group * 8)
                        base = group * 4
                        txl.ptx.st.shared.v4.b32(
                            arena.ptr_to([_tile_byte(off_sds, crow, qcol)]),
                            packed_ds[base],
                            packed_ds[base + 1],
                            packed_ds[base + 2],
                            packed_ds[base + 3],
                        )
                txl.ptx["tcgen05.wait::st.sync.aligned"]()
                with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                    dp_pipe.empty.arrive(dp_cons.stage)
                dp_cons.advance()
                txl.ptx.fence.proxy.async_.shared__cta()
                txl.ptx.bar.sync(txl.uint32(BAR_COMPUTE[0]), txl.uint32(BAR_COMPUTE[1]))
                with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                    sum_pipe.empty.arrive(sum_cons.stage)
                sum_cons.advance()
                with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                    ds_pipe.full.arrive(ds_prod.stage)
                ds_prod.advance()
                txl.assign(edge, edge + txl.int32(1))

            with txl.If(work), txl.Then():
                bh = txl.cast(batch_idx, "int64") * txl.int64(heads) + txl.cast(head, "int64")
                for is_dk in (False, True):
                    dkdv_pipe.full.wait(dkdv_cons.stage, dkdv_cons.phase)
                    # The completion wait orders TCGen's async shared-memory
                    # reads.  Bridge that proxy before the compute warps reuse
                    # the same Q/dO storage through generic shared stores.
                    txl.ptx.fence.proxy.async_.shared__cta()
                    tmem_offset = tmem_dk if is_dk else TMEM_DV
                    epi_base = off_sq if is_dk else off_sdo
                    if direct_dkv:
                        values = txl.alloc_local((head_dim // 2,), "float32")
                        for stage in range(head_dim // 64):
                            address = txl.cuda.get_tmem_addr(
                                tcol,
                                row_group,
                                txl.int32(tmem_offset + wg * (head_dim // 2) + stage * 32),
                            )
                            _tmem_load32(values, stage * 32, address)
                        txl.ptx["tcgen05.wait::ld.sync.aligned"]()
                        if is_dk:
                            for pair in range(head_dim // 4):
                                j = pair * 2
                                lo, hi = _packed_binary(
                                    "mul.rn.f32x2",
                                    values[j],
                                    values[j + 1],
                                    softmax_scale,
                                    softmax_scale,
                                )
                                txl.assign(values[j], lo)
                                txl.assign(values[j + 1], hi)
                        packed = txl.alloc_local((head_dim // 4,), "uint32")
                        for pair in range(head_dim // 4):
                            txl.ptx["cvt.rn.bf16x2.f32"](
                                packed[pair], values[pair * 2 + 1], values[pair * 2]
                            )
                        for group in range(head_dim // 16):
                            base = group * 4
                            txl.ptx.st.shared.v4.b32(
                                arena.ptr_to(
                                    [_epi_bf16_byte(epi_base, wg, crow, group * 8, head_dim // 2)]
                                ),
                                packed[base],
                                packed[base + 1],
                                packed[base + 2],
                                packed[base + 3],
                            )
                        txl.ptx.fence.proxy.async_.shared__cta()
                        txl.ptx.bar.sync(txl.cast(txl.int32(1) + wg, "uint32"), txl.uint32(128))
                        with txl.If(crow < txl.int32(32)), txl.Then():
                            with txl.If(txl.cuda.elect_sync()), txl.Then():
                                target_map = dk_map if is_dk else dv_map
                                txl.ptx[TMA_S2G](
                                    txl.address_of(target_map),
                                    wg * txl.int32(head_dim // 2),
                                    task * txl.int32(128),
                                    head,
                                    batch_idx,
                                    arena.ptr_to(
                                        [epi_base + wg * txl.int32(128 * (head_dim // 2) * 2)]
                                    ),
                                    TMA_CACHE,
                                )
                            txl.ptx.bar.arrive(
                                txl.cast(txl.int32(1) + wg, "uint32"), txl.uint32(160)
                            )
                        txl.ptx.fence.proxy.async_.shared__cta()
                        txl.ptx.bar.sync(txl.cast(txl.int32(1) + wg, "uint32"), txl.uint32(160))
                    else:
                        for stage in range(head_dim // 64):
                            values = txl.alloc_local((32,), "float32")
                            address = txl.cuda.get_tmem_addr(
                                tcol,
                                row_group,
                                txl.int32(tmem_offset + wg * (head_dim // 2) + stage * 32),
                            )
                            _tmem_load32(values, 0, address)
                            txl.ptx["tcgen05.wait::ld.sync.aligned"]()
                            for vec in range(8):
                                base = vec * 4
                                txl.ptx.st.shared.v4.b32(
                                    arena.ptr_to(
                                        [
                                            epi_base
                                            + wg * txl.int32(16384)
                                            + txl.int32(vec * 2048)
                                            + crow * txl.int32(16)
                                        ]
                                    ),
                                    values[base],
                                    values[base + 1],
                                    values[base + 2],
                                    values[base + 3],
                                )
                            txl.ptx.fence.proxy.async_.shared__cta()
                            txl.ptx.bar.sync(txl.cast(txl.int32(1) + wg, "uint32"), txl.uint32(128))
                            with txl.If(crow < txl.int32(32)), txl.Then():
                                with txl.If(txl.cuda.elect_sync()), txl.Then():
                                    base = dk_base if is_dk else dv_base
                                    dst = (
                                        txl.int64(base)
                                        + bh * txl.int64(k128 * head_dim)
                                        + txl.cast(task, "int64") * txl.int64(128 * head_dim)
                                        + txl.cast(wg, "int64") * txl.int64(128 * head_dim // 2)
                                        + txl.int64(stage * 4096)
                                    )
                                    txl.ptx[
                                        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                                    ](
                                        workspace.ptr_to([dst]),
                                        arena.ptr_to([epi_base + wg * txl.int32(16384)]),
                                        txl.uint32(16384),
                                    )
                                if stage < head_dim // 64 - 1:
                                    txl.ptx.cp.async_.bulk.commit_group()
                                    txl.ptx.cp.async_.bulk.wait_group.read(0)
                                txl.ptx.bar.arrive(
                                    txl.cast(txl.int32(1) + wg, "uint32"), txl.uint32(160)
                                )
                            txl.ptx.fence.proxy.async_.shared__cta()
                            txl.ptx.bar.sync(txl.cast(txl.int32(1) + wg, "uint32"), txl.uint32(160))
                    with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                        dkdv_pipe.empty.arrive(dkdv_cons.stage)
                    dkdv_cons.advance()
            if direct_dkv:
                tile_live = task * txl.int32(128) < txl.int32(seqlen_kv)
                with txl.If((work == txl.bool(False)) & tile_live), txl.Then():
                    row = ctid % txl.int32(128)
                    seq = task * txl.int32(128) + row
                    compute_half = ctid // txl.int32(128) - txl.int32(1)
                    destination = (
                        (
                            (
                                txl.cast(batch_idx, "int64") * txl.int64(seqlen_kv)
                                + txl.cast(seq, "int64")
                            )
                            * txl.int64(heads)
                            + txl.cast(head, "int64")
                        )
                        if bshd
                        else (
                            (
                                txl.cast(batch_idx, "int64") * txl.int64(heads)
                                + txl.cast(head, "int64")
                            )
                            * txl.int64(seqlen_kv)
                            + txl.cast(seq, "int64")
                        )
                    ) * txl.int64(head_dim)
                    with txl.If(seq < txl.int32(seqlen_kv)), txl.Then():
                        with txl.If(compute_half == txl.int32(0)), txl.Then():
                            for vec in range(head_dim // 8):
                                txl.ptx.st.global_.v4.b32(
                                    dk_output.ptr_to([destination + txl.int64(vec * 8)]),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                )
                        with txl.If(compute_half == txl.int32(1)), txl.Then():
                            for vec in range(head_dim // 8):
                                txl.ptx.st.global_.v4.b32(
                                    dv_output.ptr_to([destination + txl.int64(vec * 8)]),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                )
            txl.ptx.bar.arrive(txl.uint32(BAR_TMEM[0]), txl.uint32(BAR_TMEM[1]))
        with r_reduce:
            txl.ptx.bar.sync(txl.uint32(BAR_TMEM[0]), txl.uint32(BAR_TMEM[1]))
            tcol = txl.local_scalar("uint32")
            txl.ptx.ld.shared.b32(tcol, tmem_mailbox.ptr_to([0]))
            rtid = txl.tid_in_role()
            row_group = (rtid // txl.int32(32)) * txl.int32(32)
            dq_cons = txl.PipelineState(1, phase=0)
            store_stage = txl.local_scalar("int32", init=txl.int32(0))
            edge = txl.local_scalar("int32", init=txl.int32(0))
            bh = txl.cast(batch_idx, "int64") * txl.int64(heads) + txl.cast(head, "int64")
            bh_edge = bh * edge_stride
            with txl.While(edge < count):
                dq_pipe.full.wait(dq_cons.stage, dq_cons.phase)
                q_block = _load_i32(bucketed_indices, bh_edge + txl.cast(begin + edge, "int64"))
                q_block_safe = txl.Select(
                    q_block < txl.int32(q_blocks), q_block, txl.int32(q_blocks - 1)
                )
                values = txl.alloc_local((head_dim,), "float32")
                for chunk in range(head_dim // 32):
                    address = txl.cuda.get_tmem_addr(
                        tcol, row_group, txl.int32(tmem_dq + chunk * 32)
                    )
                    _tmem_load32(values, chunk * 32, address)
                txl.ptx["tcgen05.wait::ld.sync.aligned"]()
                txl.ptx.bar.warp.sync(txl.uint32(0xFFFFFFFF))
                with txl.If(txl.lane_id() == txl.int32(0)), txl.Then():
                    dq_pipe.empty.arrive(dq_cons.stage)
                dq_cons.advance()

                for chunk in range(head_dim // 32):
                    for vec in range(8):
                        base = chunk * 32 + vec * 4
                        address = (
                            off_sdq
                            + store_stage * txl.int32(16384)
                            + txl.int32(vec * 2048)
                            + rtid * txl.int32(16)
                        )
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
                        with txl.If(txl.cuda.elect_sync()), txl.Then():
                            dst = (
                                txl.int64(dq_base)
                                + bh * txl.int64(q128 * head_dim)
                                + txl.cast(q_block_safe, "int64") * txl.int64(128 * head_dim)
                                + txl.int64(chunk * 4096)
                            )
                            txl.ptx["cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"](
                                workspace.ptr_to([dst]),
                                arena.ptr_to([off_sdq + store_stage * txl.int32(16384)]),
                                txl.uint32(16384),
                            )
                        txl.ptx.cp.async_.bulk.commit_group()
                        txl.ptx.cp.async_.bulk.wait_group.read(1)
                    txl.ptx.bar.sync(txl.uint32(BAR_REDUCE[0]), txl.uint32(BAR_REDUCE[1]))
                    txl.assign(store_stage, txl.Select(store_stage == txl.int32(1), 0, 1))
                txl.assign(edge, edge + txl.int32(1))
            with txl.If(work), txl.Then():
                with txl.If(txl.warp_id() == txl.int32(0)), txl.Then():
                    txl.ptx.cp.async_.bulk.wait_group.read(0)
                txl.ptx.bar.sync(txl.uint32(BAR_REDUCE[0]), txl.uint32(BAR_REDUCE[1]))
            txl.ptx.cp.async_.bulk.wait_group.read(0)
            txl.ptx.bar.arrive(txl.uint32(BAR_TMEM[0]), txl.uint32(BAR_TMEM[1]))

        with r_idle:
            pass
        with r_empty:
            pass

        required_block_size.__exit__(None, None, None)

    def make_postprocess(seq_len, padded_len, source_base):
        @txl.kernel(
            warps=4,
            arch="sm_100a",
            min_blocks_per_sm=1,
            grid=((seq_len + 127) // 128, heads, batch),
        )
        def postprocess(
            workspace: txl.gptr[txl.f32], output: txl.gptr[txl.bf16], output_scale: txl.f32
        ):
            required_block_size = txl.attr({"tirx.required_block_size": 1})
            required_block_size.__enter__()
            seq_tile, head, batch_idx = txl.cta_id()
            tid = txl.thread_id()
            seq = seq_tile * txl.int32(128) + tid
            bh = txl.cast(batch_idx, "int64") * txl.int64(heads) + txl.cast(head, "int64")
            arena = txl.alloc_buffer((128 * head_dim * 4,), txl.u8, scope="shared.dyn", align=1024)
            source = (
                txl.int64(source_base)
                + bh * txl.int64(padded_len * head_dim)
                + txl.cast(seq_tile, "int64") * txl.int64(128 * head_dim)
            )
            for vec in range(head_dim // 4):
                element = tid * txl.int32(4) + txl.int32(vec * 512)
                txl.ptx["cp.async.cg.shared.global"](
                    arena.ptr_to([element * txl.int32(4)]),
                    workspace.ptr_to([source + txl.cast(element, "int64")]),
                    16,
                    16,
                )
            txl.ptx.cp.async_.commit_group()
            txl.ptx.cp.async_.wait_group(0)
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(128))

            values = txl.alloc_local((head_dim,), "float32")
            for vec in range(head_dim // 4):
                element = tid * txl.int32(4) + txl.int32(vec * 512)
                txl.ptx.ld.shared_.v4.b32(
                    values[vec * 4],
                    values[vec * 4 + 1],
                    values[vec * 4 + 2],
                    values[vec * 4 + 3],
                    arena.ptr_to([element * txl.int32(4)]),
                )
            scaled = txl.alloc_local((head_dim,), "float32")
            for pair in range(head_dim // 2):
                lo, hi = _packed_binary(
                    "mul.f32x2", values[pair * 2], values[pair * 2 + 1], output_scale, output_scale
                )
                txl.assign(scaled[pair * 2], lo)
                txl.assign(scaled[pair * 2 + 1], hi)
            packed = txl.alloc_local((head_dim // 2,), "uint32")
            for pair in range(head_dim // 2):
                txl.ptx["cvt.rn.bf16x2.f32"](packed[pair], scaled[pair * 2 + 1], scaled[pair * 2])
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(128))
            for group in range(head_dim // 8):
                base = group * 4
                byte = _tile_byte(0, tid, txl.int32(group * 8))
                txl.ptx.st.shared.v4.b32(
                    arena.ptr_to([byte]),
                    packed[base],
                    packed[base + 1],
                    packed[base + 2],
                    packed[base + 3],
                )
            txl.ptx.bar.sync(txl.uint32(0), txl.uint32(128))
            threads_per_row = txl.int32(head_dim // 8)
            row_step = txl.int32(128 // (head_dim // 8))
            row_base = tid // threads_per_row
            feature = (tid % threads_per_row) * txl.int32(8)
            for rep in range(head_dim // 8):
                row = row_base + txl.int32(rep) * row_step
                seq = seq_tile * txl.int32(128) + row
                with txl.If(seq < txl.int32(seq_len)), txl.Then():
                    words = txl.alloc_local((4,), "uint32")
                    byte = _tile_byte(0, row, feature)
                    txl.ptx.ld.shared_.v4.b32(
                        words[0], words[1], words[2], words[3], arena.ptr_to([byte])
                    )
                    destination = (
                        (
                            (
                                txl.cast(batch_idx, "int64") * txl.int64(seq_len)
                                + txl.cast(seq, "int64")
                            )
                            * txl.int64(heads)
                            + txl.cast(head, "int64")
                        )
                        if bshd
                        else (
                            (
                                txl.cast(batch_idx, "int64") * txl.int64(heads)
                                + txl.cast(head, "int64")
                            )
                            * txl.int64(seq_len)
                            + txl.cast(seq, "int64")
                        )
                    ) * txl.int64(head_dim)
                    txl.ptx.st.global_.v4.b32(
                        output.ptr_to([destination + txl.cast(feature, "int64")]),
                        words[0],
                        words[1],
                        words[2],
                        words[3],
                    )
            required_block_size.__exit__(None, None, None)

        return postprocess.func

    kernels = [preprocess.func, bwd.func]
    kernels.append(make_postprocess(seqlen_q, q128, dq_base))
    if not direct_dkv:
        kernels.append(make_postprocess(seqlen_kv, k128, dk_base))
        kernels.append(make_postprocess(seqlen_kv, k128, dv_base))
    return tuple(kernels)
