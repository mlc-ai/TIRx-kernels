# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Device kernels for the DSA sparse attention backward pass.

Upstream source:
``python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100.py``.

The upstream ``FlashAttentionDSABackwardSm100.__call__`` launches four kernels on
one stream, and this module builds the same four in the same order:

1. ``sum_OdO``   -- per ``(head, query)`` delta ``-sum_d(O * dO)`` and the
   sink-folded log2-domain LSE, written to the LSE/OdO workspace;
2. ``bwd``       -- the 20-warp main kernel: one CTA per query token, heads on
   the MMA M axis, top-k KV tiles walked in reverse, dQ stored through TMA and
   dKV accumulated into an FP32 workspace with global atomics;
3. ``convert``   -- FP32 dKV workspace to the element dtype, unscrambling the two
   store fragment layouts;
4. ``sum_dSink`` -- the attention-sink gradient, warp-reduced then accumulated
   atomically.

Shapes, allocation sizes and the compile key follow the upstream host wrapper
``_interface_sm100.py``.
"""

import tirx_kernels.kern as K

from . import spec

# ``dsa_bwd_sm100.py:64-76``: warp roles of the main kernel and the resulting
# 640-thread block.
LOAD_KV_WARPS = (0, 1, 2, 3)
COMPUTE_WARPS = (4, 5, 6, 7)
REDUCE_WARPS = (8, 9, 10, 11, 12, 13, 14, 15)
MMA_WARP = 16
LOAD_WARP = 17
EMPTY_WARP = 18
BWD_WARPS = 20
BWD_THREADS = BWD_WARPS * 32

# ``dsa_bwd_sm100.py:149-154``: declared per-role register budgets.
REGS_LOAD_KV = 40
REGS_COMPUTE = 128
REGS_REDUCE = 128
REGS_MMA = 40
REGS_LOAD = 40
REGS_EMPTY = 40

# ``dsa_bwd_sm100.py:82-131``: named barrier ids and their participant counts.
BAR_CTA_SYNC = (1, 640)
BAR_TMEM_ALLOC = (2, 416)
BAR_COMPUTE_SYNC = (3, 128)
BAR_LOAD_SYNC = (4, 32)
BAR_LOAD_KV_SYNC = (5, 128)
BAR_REDUCE_SYNC = (6, 256)
BAR_T2R_DKV01_DONE = (7, 288)
BAR_T2R_DKV4_DONE = (8, 288)
BAR_TMEM_DEALLOC = (9, 288)
BAR_T2R_DKV23_DONE = (10, 288)

# ``dsa_bwd_sm100.py:133-147`` with ``block_tile = 64``. dKV2/dKV3 alias
# dKV0/dKV1 and dKV4 aliases dKV0; the three ``t2r_dKV*_done`` barriers are what
# keeps those reuses safe.
TMEM_S_OFFSET = 0
TMEM_DP_OFFSET = 0
TMEM_DKV0_OFFSET = 64
TMEM_DKV1_OFFSET = 128
# dKV2 is the one entry whose column genuinely differs between the two builds
# (`:2684` vs `:2687`): at d512 it takes the slot dQ4 would have used, so only
# dKV3 aliases; at d576 that slot is dQ4's and dKV2 falls back onto dKV0. This
# is also why the WAR barrier assignment differs between the builds.
TMEM_DKV2_D512_OFFSET = 448
TMEM_DKV2_D576_OFFSET = TMEM_DKV0_OFFSET
TMEM_DKV3_OFFSET = TMEM_DKV1_OFFSET
TMEM_DQ0_OFFSET = 192
TMEM_DQ1_OFFSET = 256
TMEM_DQ2_OFFSET = 320
TMEM_DQ3_OFFSET = 384
TMEM_DQ4_OFFSET = 448
TMEM_DKV4_OFFSET = TMEM_DKV0_OFFSET
TMEM_ALLOC_COLUMNS = spec.TMEM_CAPACITY_COLUMNS


# Element dtype and the narrow-operand spellings that follow from it. The
# element dtype selects the descriptor and these mnemonics and nothing else.
_ELEM = {"bfloat16": K.bf16, "float16": K.f16}
_MUL_HALF = {"bfloat16": "mul.bf16x2", "float16": "mul.f16x2"}
_ADD_WIDE = {"bfloat16": "add.rn.f32.bf16", "float16": "add.rn.f32.f16"}
_CVT_WIDE = {"bfloat16": "cvt.f32.bf16", "float16": "cvt.f32.f16"}
_CVT_NARROW = {"bfloat16": "cvt.rn.bf16.f32", "float16": "cvt.rn.f16.f32"}
_CVT_PACK = {"bfloat16": "cvt.rn.bf16x2.f32", "float16": "cvt.rn.f16x2.f32"}


# ---------------------------------------------------------------------------
# Shared memory is one flat byte pool. Every logical coordinate, swizzle and
# alias below is explicit scalar arithmetic on a byte offset into it.
# ---------------------------------------------------------------------------


# 2 / (k * ln 2) for k = 1, 3, .., 11 -- the odd-power series for log2.
_LOG2_SERIES = (
    2.8853900817779268,
    0.9617966939259757,
    0.5770780163555853,
    0.41219858311113244,
    0.3205988979753252,
    0.2623081892525388,
)


def _log2_soft(x):
    """`log2(x)` for x in [1, 2] as a polynomial, not `lg2.approx`.

    The source calls `cute.math.log2` WITHOUT fastmath (`:700`), which lowers to
    a software polynomial; `lg2` appears nowhere in any export, so the hardware
    approximation would not be the same kernel. The argument here is
    `exp2(a - max) + exp2(b - max)`, so one term is exactly 1 and the other at
    most 1: the range is [1, 2], `r = (x - 1) / (x + 1)` is bounded by 1/3, and
    six odd terms land within 1.6e-7 of the true value across the range.
    """
    one = K.local_scalar(K.f32, init=K.float32(1.0))
    num = K.local_scalar(K.f32)
    den = K.local_scalar(K.f32)
    K.ptx["sub.f32"](num, x, one)
    K.ptx["add.f32"](den, x, one)
    r = K.local_scalar(K.f32)
    K.ptx["div.rn.f32"](r, num, den)
    r2 = K.local_scalar(K.f32)
    K.ptx["mul.f32"](r2, r, r)
    acc = K.local_scalar(K.f32, init=K.float32(_LOG2_SERIES[-1]))
    for c in reversed(_LOG2_SERIES[:-1]):
        cv = K.local_scalar(K.f32, init=K.float32(c))
        K.ptx["fma.rn.f32"](acc, acc, r2, cv)
    out = K.local_scalar(K.f32)
    K.ptx["mul.f32"](out, acc, r)
    return out


def _xor(a, b):
    """`a ^ b`, trace-time when both sides are Python ints."""
    if isinstance(a, int) and isinstance(b, int):
        return a ^ b
    return K.bitwise_xor(a, b)


def _tile_elem(row, dim):
    """Element offset of `(row, dim)` inside an operand region.

    An operand region is a stack of 64-wide blocks of 64 rows -- the layout the
    source builds for every A/B operand and epilogue tile (`:864-903`) -- with
    the 128-byte swizzle inside each block: a row spans eight 16-byte
    sub-blocks and they are XORed by `row % 8`.

    `row` is the 64-strided axis and `dim` the contiguous one, which is why the
    same function serves the K-major operands (row = head or KV row), the
    64-wide P/dS tiles, and the dQ epilogue tile (row = head, dim = the dim).
    """
    return (dim // 64) * 4096 + _xor(row * 64 + dim % 64, (row % 8) * 8)


def _tile_byte(base, row, dim):
    """Byte offset of `(row, dim)` in the region starting at `base`."""
    return base + 2 * _tile_elem(row, dim)


def _desc_base(ldo, sdo, swizzle=3):
    """Fold the static SM100 shared-descriptor fields except the 14-bit address.

    Only two combinations occur in this kernel: regions at least two swizzle
    atoms wide take `ldo = 512` (one atom is 4096 elements = 512 sixteen-byte
    units apart), and the 64-wide P/dS tiles take `ldo = 0` because they have
    no second atom to step to. `sdo = 64` is the eight-row stride in the same
    units. Both were read off the descriptors this kernel's operands produce.
    """
    arrangement = {0: 0, 1: 6, 2: 4, 3: 2, 4: 1}[swizzle]
    value = ((ldo & 0x3FFF) << 16) | ((sdo & 0x3FFF) << 32) | (1 << 46)
    return (value | ((arrangement & 0x7) << 61)) & 0xFFFFFFFFFFFFFFFF


def _desc_at(base, shared_address):
    """`base` with the shared address folded into its 14-bit address field."""
    field = K.cast(
        K.bitwise_and(K.shift_right(shared_address, K.uint32(4)), K.uint32(0x3FFF)), "uint64"
    )
    return K.bitwise_or(K.uint64(base), field)


def _desc_add16(desc, offset):
    """`desc` stepped by a 16-byte-unit offset, without touching its high half."""
    if isinstance(offset, int) and offset == 0:
        return desc
    lo = K.local_scalar("uint32")
    hi = K.local_scalar("uint32")
    out = K.local_scalar("uint64")
    K.ptx.mov.b64(lo, hi, desc)
    K.ptx["add.u32"](lo, lo, K.uint32(offset))
    K.ptx.mov.b64(out, lo, hi)
    return out


def _gemm(d, a, b, idesc, accumulate, n_iter, unroll):
    """One tcgen05 k-loop, ROLLED: `n_iter` iterations of `unroll` issues.

    The source's chains are rolled loops, not fully unrolled ones (`unroll=4`
    on the 32-phase S/dP chains, `unroll=2` on the 4-phase dKV and dQ chains),
    which is what keeps the kernel at 32 static `tcgen05.mma` at d512 and 36 at
    d576. Emitting one instruction per k-phase instead multiplies that by three
    or four.

    `a` and `b` are each `(descriptor, per-iteration step, per-issue step)` in
    sixteen-byte units. Both steps are linear in the loop counter because the
    non-linear part of a K-major walk -- the jump at each 64-element swizzle
    atom -- lands exactly on the iteration boundary when `unroll` divides the
    atom.

    `accumulate` rides the first issue only; the flag is then latched to 1,
    which is the `(kp != 0) or flag` disjunction the accumulator semantics ask
    for, built as the source builds it.
    """
    a_desc, a_it, a_u = a
    b_desc, b_it, b_u = b
    flag = K.local_scalar("uint32")
    K.assign(flag, K.cast(accumulate, "uint32"))
    it = K.local_scalar(K.i32, init=K.int32(0))
    with K.While(it < K.int32(n_iter)):
        for u in range(unroll):
            _mma(
                MMA_F16,
                d,
                _desc_add16(a_desc, it * K.int32(a_it) + K.int32(u * a_u)),
                _desc_add16(b_desc, it * K.int32(b_it) + K.int32(u * b_u)),
                idesc,
                flag,
            )
            K.assign(flag, K.uint32(1))
        K.assign(it, it + K.int32(1))


def _mma(mma, d, a_desc, b_desc, idesc, accumulate):
    """One `tcgen05.mma` in its dense non-ws operand form."""
    K.ptx[mma](
        K.Cast("uint32", d),
        a_desc,
        b_desc,
        K.uint32(idesc),
        K.uint32(0),
        K.uint32(0),
        K.uint32(0),
        K.uint32(0),
        K.ptx.pred(accumulate),
    )


# The instruction descriptors every tcgen05 MMA carries, read out of the
# line-info PTX export (probe/probe_idesc.py). The bf16 and fp16 columns differ
# by exactly the d/a/b format bits; M, N and both transpose bits are
# dtype-invariant. trans_a=1 is an MN-major A operand, trans_b=1 an MN-major B.
_IDESC = {
    "bfloat16": {
        "qk": 0x04100490,  # S and dP:      M=64  N=64 trans 0,0
        "dkv": 0x08108490,  # dKV0..3:       M=128 N=64 trans 1,0
        "dq": 0x08118490,  # dQ0..3:        M=128 N=64 trans 1,1
        "dkv4": 0x04108490,  # dKV4 (d576):   M=64  N=64 trans 1,0
        "dq4": 0x04118490,  # dQ4  (d576):   M=64  N=64 trans 1,1
    },
    "float16": {
        "qk": 0x04100010,
        "dkv": 0x08108010,
        "dq": 0x08118010,
        "dkv4": 0x04108010,
        "dq4": 0x04118010,
    },
}


MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"
TMA_G2S = (
    "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
)
TMA_S2G = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint"
TCGEN05_COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"


def _bwd_load(
    desc_q,
    desc_do,
    desc_dq,
    smem,
    off_q,
    off_do,
    off_lse,
    off_sum,
    ws,
    p_qdo,
    p_lse,
    p_sum,
    token,
    head_block,
    plane,
    lane,
    head_dim,
    head_dim_v,
    num_head,
    block,
):
    """Role `load` (warp 17), `:1113-1207`. Runs once -- a CTA owns one token.

    Q and dO are TMA'd into one barrier under a single `expect_tx` covering
    both; LSE and sum_OdO come in as one 64-bit cp.async each.
    """
    q_bytes = block * head_dim * 2
    do_bytes = block * head_dim_v * 2
    for desc in (desc_q, desc_do, desc_dq):
        K.ptx.prefetch.tensormap(K.address_of(desc))

    with K.If(lane == 0), K.Then():
        # The Q/dO pipeline is acquired before the transfer is announced
        # (:1152-1155); it runs once, so the wait is on the initial phase.
        p_qdo.empty.wait(0, 1)
        # ONE expect_tx covers BOTH transfers: they share a barrier upstream
        # (tx_count = tma_copy_QdO_bytes, :442-444).
        K.ptx["mbarrier.arrive.expect_tx.shared.b64"](
            p_qdo.full.buf.ptr_to([0]), K.uint32(q_bytes + do_bytes)
        )
        for b in range(head_dim // 64):
            # Box `b` is the 64-wide dim block at element offset b * 4096.
            K.ptx[TMA_G2S](
                smem.ptr_to([_tile_byte(off_q, 0, b * 64)]),
                K.address_of(desc_q),
                K.int32(b * 64),
                head_block * block,
                token,
                p_qdo.full.buf.ptr_to([0]),
                K.uint64(0),
            )
        for b in range(head_dim_v // 64):
            K.ptx[TMA_G2S](
                smem.ptr_to([_tile_byte(off_do, 0, b * 64)]),
                K.address_of(desc_do),
                K.int32(b * 64),
                head_block * block,
                token,
                p_qdo.full.buf.ptr_to([0]),
                K.uint64(0),
            )

    # LSE and sum_OdO: 32 lanes x 2 f32 = the 64 rows of this head block. The
    # workspace planes are head-contiguous, so a head block is 64 consecutive
    # f32 only when it starts at head 0; index each row explicitly instead.
    slot = token * num_head + head_block * block
    p_lse.empty.wait(0, 1)
    K.ptx["cp.async.ca.shared.global"](
        smem.ptr_to([off_lse + lane * 8]), ws.ptr_to([plane + slot + lane * 2]), 8, 8
    )
    K.ptx["cp.async.mbarrier.arrive.noinc.shared.b64"](p_lse.full.buf.ptr_to([0]))

    p_sum.empty.wait(0, 1)
    K.ptx["cp.async.ca.shared.global"](
        smem.ptr_to([off_sum + lane * 8]), ws.ptr_to([slot + lane * 2]), 8, 8
    )
    K.ptx["cp.async.mbarrier.arrive.noinc.shared.b64"](p_sum.full.buf.ptr_to([0]))


# The gather writes the KV region through `_tile_byte`, the same explicit
# offset function the MMA descriptors are built against, so the operand a GEMM
# reads is exactly what the gather wrote.


def _gather_or_zero(smem, addr, kv, row, col, src_row, head_dim, live):
    """One lane's 128-bit slot of one gathered KV row, or a zero fill.

    The source does not use cp.async's ignore-src form for invalid rows; it
    stores zeros separately (`_zero_kv_row`, :1246-1271). Keep that: ignore-src
    and whole-instruction predication leave different bytes behind, and the MMA
    reads them either way.
    """
    # `col` is the dim in the *global* KV row; `dst`/`dcol` are where it lands
    # in the staged operand, which only the caller can split (the stage index
    # comes from the 64-wide group, which is runtime for the eight main groups
    # and trace-time for d576's tail).
    if live is None:
        K.ptx["cp.async.cg.shared.global"](
            smem.ptr_to([addr]), kv.ptr_to([src_row * head_dim + col]), 16, 16
        )
        return
    # `K.If(cond)` opens the frame; Then/Else nest inside it. The
    # `with K.If(c), K.Then():` shorthand is if-without-else only.
    with K.If(live):
        with K.Then():
            K.ptx["cp.async.cg.shared.global"](
                smem.ptr_to([addr]), kv.ptr_to([src_row * head_dim + col]), 16, 16
            )
        with K.Else():
            K.ptx["st.shared.v4.b32"](
                smem.ptr_to([addr]), K.uint32(0), K.uint32(0), K.uint32(0), K.uint32(0)
            )


def _bwd_load_kv(
    kv,
    topk_idxs,
    smem,
    off_k,
    p_k,
    tile_count,
    topk,
    token,
    tid,
    lane,
    head_dim,
    seqlen_kv,
    max_topk,
    has_topk_length,
    block,
):
    """Role `load_KV` (warps 0-3), `:1316-1431`. Reverse tile walk.

    Lane 0 of each warp reads 16 top-k indices and broadcasts them; all 32 lanes
    then cp.async-gather the rows, 8 lanes per 64-element head-dim group.
    """
    lwarp = tid // 32  # 0..3 within the role
    rows_per_warp = block // 4
    groups = head_dim // 64
    full_tiles = (topk % K.int32(block)) == K.int32(0)  # :1344

    def read_indices(tile_index):
        """This warp's 16 top-k indices, read by lane 0 and broadcast."""
        r = K.alloc_local([16], "int32")
        for i in range(rows_per_warp):
            row = i * 4 + lwarp
            gidx = tile_index * K.int32(block) + K.int32(row)
            v = K.local_scalar(K.i32, init=K.int32(-1))
            # The bound is the allocation extent, not `topk`: rows past the
            # length are read and then discarded.
            with K.If((lane == 0) & (gidx < K.int32(max_topk))), K.Then():
                K.ptx["ld.global.b32"](v, topk_idxs.ptr_to([token * max_topk + gidx]))
            bc = K.local_scalar(K.i32)
            K.ptx.shfl_sync.idx.b32(bc, v, K.uint32(0), K.uint32(31), K.uint32(0xFFFFFFFF))
            K.assign(r[i], bc)
        return r

    def gather_rows(tile_index, r, is_first):
        """One tile's 64 rows. `is_first` is a trace-time flag, as upstream.

        The two builds run different validity programs (:1292-1310). With a
        compact top-k list every row of a non-ragged tile is valid, so the copy
        is unconditional and only the ragged tile zero-fills past `topk`;
        without one, every row is tested for both bounds and its own index.
        """
        for i in range(rows_per_warp):
            row = i * 4 + lwarp
            gidx = tile_index * K.int32(block) + K.int32(row)
            src_row = r[i]
            if has_topk_length:
                live = (gidx < topk) if is_first else None
            else:
                live = (gidx < topk) & (src_row >= K.int32(0))
            # The group is chosen by `lane // 8` and the slot inside it by
            # `lane % 8` (:1224, :1339), so each lane owns two of the eight
            # 64-wide groups and every address has exactly one writer. Indexing
            # the groups with a Python loop instead makes four lanes collide on
            # each address, which silently loses whole groups.
            for j in range(2):
                g = lane // K.int32(8) + K.int32(j * 4)
                slot = (lane % K.int32(8)) * K.int32(8)
                col = g * K.int32(64) + slot
                _gather_or_zero(
                    smem, _tile_byte(off_k, row, col), kv, row, col, src_row, head_dim, live
                )
            if groups == 9:
                # The ninth group is the d576 tail: 64 dims, so only 8 lanes.
                with K.If(lane < K.int32(8)), K.Then():
                    slot = (lane % K.int32(8)) * K.int32(8)
                    tcol = K.int32(8 * 64) + slot
                    _gather_or_zero(
                        smem, _tile_byte(off_k, row, tcol), kv, row, tcol, src_row, head_dim, live
                    )

    def publish():
        """Drain the gather and hand the tile to the MMA warp."""
        K.ptx["cp.async.commit_group"]()
        # Waits for ALL groups: the gather is not double-buffered.
        K.ptx["cp.async.wait_group"](K.int32(0))
        K.ptx["fence.proxy.async.shared::cta"]()
        # Makes the four warps' gathers jointly visible before any one of them
        # commits the pipeline.
        K.ptx.bar.sync(K.uint32(BAR_LOAD_KV_SYNC[0]), K.uint32(BAR_LOAD_KV_SYNC[1]))
        K.ptx["mbarrier.arrive.shared.b64"](p_k.full.buf.ptr_to([0]), K.uint32(1))

    st = K.PipelineState(1, phase=0)
    tile_index = K.local_scalar(K.i32, init=tile_count - K.int32(1))

    # The highest-index tile is the ragged one and is peeled, so the loop body
    # never carries the ragged-tile test (:1343-1393).
    r0 = read_indices(tile_index)
    p_k.empty.wait(0, st.phase ^ 1)
    with K.If(full_tiles):
        with K.Then():
            gather_rows(tile_index, r0, is_first=False)
        with K.Else():
            gather_rows(tile_index, r0, is_first=True)
    publish()
    st.advance()
    K.assign(tile_index, tile_index - K.int32(1))

    with K.While(tile_index >= K.int32(0)):
        r = read_indices(tile_index)
        p_k.empty.wait(0, st.phase ^ 1)
        gather_rows(tile_index, r, is_first=False)
        publish()
        st.advance()
        K.assign(tile_index, tile_index - K.int32(1))


def make_bwd_kernel(*, head_dim, num_head, dtype, max_topk, has_topk_length):
    """Trace the main backward kernel for one static specialization (``:740-1110``).

    One CTA per query token; the 64 attention heads form the MMA M dimension, so
    a token's top-k KV rows are gathered once and amortized across every head.
    """
    spec.check_dispatch(
        {
            "head_dim": head_dim,
            "dtype": dtype,
            "topk_mode": "full",
            "sink_mode": "normal",
            "has_topk_length": has_topk_length,
        }
    )
    head_dim_v = spec.head_dim_v_for(head_dim)
    elem = _ELEM[dtype]
    block = spec.BLOCK_TILE
    idesc = _IDESC[dtype]

    @K.kernel(
        warps=BWD_WARPS,
        arch="sm_100a",
        min_blocks_per_sm=1,
        grid=lambda p: [p["seqlen_q"], (num_head + block - 1) // block, 1],
    )
    def bwd(
        desc_q: K.TensorMap,
        desc_do: K.TensorMap,
        desc_dq: K.TensorMap,
        kv: K.gptr[elem],
        dq: K.gptr[elem],
        ws_dkv: K.gptr[K.f32],
        topk_idxs: K.gptr[K.i32],
        topk_length: K.gptr[K.i32],
        ws: K.gptr[K.f32],
        seqlen_q: K.i32,
        seqlen_kv: K.i32,
        plane: K.i32,
        scale: K.f32,
    ):
        # ---- kernel body starts here ----
        token, head_block, _z = K.cta_id()
        tid = K.thread_id()
        K.keep_alive(dq.data)

        # `topk` is CTA-uniform: one value per query token.
        if has_topk_length:
            topk = K.local_scalar(K.i32)
            K.ptx["ld.global.b32"](topk, topk_length.ptr_to([token]))
        else:
            K.keep_alive(topk_length.data)
            topk = K.local_scalar(K.i32, init=K.int32(max_topk))

        # An empty or malformed row contributes nothing to dKV and its dQ tile is
        # zero. This runs BEFORE any pipeline init or TMEM allocation, and that
        # ordering is load-bearing: a CTA that exited after arriving on a
        # pipeline init would strand the rest of the grid.
        with K.If(topk <= K.int32(0)), K.Then():
            i = K.local_scalar(K.i32, init=tid)
            with K.While(i < K.int32(head_dim * block)):
                head_offset = i // K.int32(head_dim)
                dim_idx = i % K.int32(head_dim)
                head_idx = head_block * block + head_offset
                with K.If(head_idx < K.int32(num_head)), K.Then():
                    zero = K.local_scalar("uint16", init=K.uint16(0))
                    K.ptx["st.global.b16"](
                        dq.ptr_to([token * (num_head * head_dim) + head_idx * head_dim + dim_idx]),
                        zero,
                    )
                K.assign(i, i + K.int32(BWD_THREADS))

        # The source spells the early-out as `nvvm.exit()`. K has no CTA-exit
        # instruction, so the rest of the kernel is guarded by the complement
        # instead. That is the same control flow: `topk` is CTA-uniform, so
        # every thread of a CTA takes the same arm and the collectives below are
        # either all executed or all skipped.
        with K.If(topk > K.int32(0)), K.Then():
            K.keep_alive(ws_dkv.data)
            tile_count = (topk + K.int32(block - 1)) // K.int32(block)

            # --- one linear SMEM pool ---------------------------------------
            # Every object below is a byte offset into a single flat buffer.
            # The order and the 1024-byte alignment of the operand regions
            # follow SharedStorage (:448-470); the totals are 214,528 B at d512
            # and 230,912 B at d576 against a 232,448 B cap.
            dkv2_offset = TMEM_DKV2_D576_OFFSET if head_dim > 512 else TMEM_DKV2_D512_OFFSET
            # Only ONE WAR barrier is skipped-once per build, and which one
            # differs: 8 at d512 (`:1546`), 10 at d576 (`:1543`). The loop tail
            # therefore carries exactly one compensating arrive (`:1752-1757`);
            # compensating a barrier that was never skipped, or missing the one
            # that was, hangs.
            war_gen_a = BAR_T2R_DKV23_DONE if head_dim > 512 else BAR_T2R_DKV4_DONE
            war_gen_b = BAR_T2R_DKV4_DONE if head_dim > 512 else BAR_T2R_DKV01_DONE
            tail = head_dim - 512

            q_bytes = block * head_dim * 2
            do_bytes = block * head_dim_v * 2
            p_bytes = block * block * 2
            OFF_Q = 1024
            OFF_K = OFF_Q + q_bytes
            OFF_DO = OFF_K + q_bytes
            OFF_P = OFF_DO + do_bytes
            OFF_DS = OFF_P + p_bytes
            OFF_LSE = OFF_DS + p_bytes
            OFF_SUM = OFF_LSE + block * 4
            shared_bytes = OFF_SUM + block * 4
            # sdQ and sdQ4 alias sK. They are NOT co-live with it: the dQ
            # epilogue runs only after the MMA warp's last release of the KV
            # pipeline (`:1683`) and the compute warp's dQ wait (`:2033`). The
            # two 64x64 dQ stages are exactly the two boxes one 128-wide round
            # stores.
            OFF_DQ = OFF_K
            OFF_DQ4 = OFF_K + 2 * p_bytes

            smem = K.alloc_buffer((shared_bytes,), K.u8, scope="shared.dyn", align=1024)
            pool = K.smem_pool(base=smem)
            smem_base = K.local_scalar("uint32")
            K.assign(smem_base, K.cuda.cvta_generic_to_shared(smem.ptr_to([0])))

            # --- pipelines --------------------------------------------------
            # Arrival counts are the upstream CooperativeGroup sizes
            # (:2711-2866). The 1s are single-Thread agents -- one elected lane
            # arrives -- not 32-thread warps; getting one wrong deadlocks.
            p_qdo = K.Pipeline(pool, 1, full="tma", empty="tcgen05", init_full=1, init_empty=1)
            p_k = K.Pipeline(pool, 1, full="mbar", empty="tcgen05", init_full=128, init_empty=1)
            p_lse = K.Pipeline(pool, 1, full="mbar", empty="mbar", init_full=32, init_empty=128)
            p_sum = K.Pipeline(pool, 1, full="mbar", empty="mbar", init_full=32, init_empty=128)
            p_s = K.Pipeline(pool, 1, full="tcgen05", empty="mbar", init_full=1, init_empty=128)
            p_dp = K.Pipeline(pool, 1, full="tcgen05", empty="mbar", init_full=1, init_empty=128)
            p_dq = K.Pipeline(pool, 1, full="tcgen05", empty="mbar", init_full=1, init_empty=128)
            p_p = K.Pipeline(pool, 1, full="mbar", empty="tcgen05", init_full=128, init_empty=1)
            p_ds = K.Pipeline(pool, 1, full="mbar", empty="tcgen05", init_full=128, init_empty=1)
            p_dkv = K.Pipeline(pool, 2, full="tcgen05", empty="mbar", init_full=1, init_empty=256)

            tmem_hold = pool.alloc((1,), K.i32, align=4)
            K.ptx["fence.mbarrier_init.release.cluster"]()
            K.ptx.bar.sync(K.uint32(0), K.uint32(BWD_THREADS))

            sp = K.specialize(chain_dispatch=True)
            # Declared in the source's dispatch order (:909-1110), which is
            # deliberately not ascending warp order.
            r_load = sp.role("load", warps=[LOAD_WARP], regs=REGS_LOAD)
            r_mma = sp.role("mma", warps=[MMA_WARP], regs=REGS_MMA)
            r_compute = sp.role("compute", warps=list(COMPUTE_WARPS), regs=REGS_COMPUTE)
            r_reduce = sp.role("reduce", warps=list(REDUCE_WARPS), regs=REGS_REDUCE)
            r_loadkv = sp.role("load_kv", warps=list(LOAD_KV_WARPS), regs=REGS_LOAD_KV)
            r_empty = sp.role("empty", warps=[EMPTY_WARP, EMPTY_WARP + 1], regs=REGS_EMPTY)

            lane = tid % 32

            with r_load:
                _bwd_load(
                    desc_q,
                    desc_do,
                    desc_dq,
                    smem,
                    OFF_Q,
                    OFF_DO,
                    OFF_LSE,
                    OFF_SUM,
                    ws,
                    p_qdo,
                    p_lse,
                    p_sum,
                    token,
                    head_block,
                    plane,
                    lane,
                    head_dim,
                    head_dim_v,
                    num_head,
                    block,
                )

            with r_mma:
                # TMEM is allocated by compute warp 0; every role that reads it
                # waits on barrier 2 (416 threads: compute + reduce + mma).
                K.ptx.bar.sync(K.uint32(BAR_TMEM_ALLOC[0]), K.uint32(BAR_TMEM_ALLOC[1]))
                tcol = K.local_scalar("uint32")
                K.ptx.ld.shared.b32(tcol, tmem_hold.ptr_to([0]))
                t_s = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(TMEM_S_OFFSET))
                # dP shares S's columns at a different lane range (:2675-2677).
                t_dp = K.cuda.get_tmem_addr(tcol, K.uint32(16), K.uint32(TMEM_DP_OFFSET))

                t_dkv = [
                    K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(off))
                    for off in (TMEM_DKV0_OFFSET, TMEM_DKV1_OFFSET)
                ]
                t_dkv4 = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(TMEM_DKV4_OFFSET))
                t_dkv23 = [
                    K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(off))
                    for off in (dkv2_offset, TMEM_DKV3_OFFSET)
                ]
                t_dq = [
                    K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(off))
                    for off in (TMEM_DQ0_OFFSET, TMEM_DQ1_OFFSET, TMEM_DQ2_OFFSET, TMEM_DQ3_OFFSET)
                ]
                t_dq4 = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(TMEM_DQ4_OFFSET))

                # Operand descriptors, encoded once. Only a 64-wide K-major
                # operand has no second swizzle atom to step to, so only those
                # take LBO 0; everything else steps one atom = 4096 elements =
                # 512 sixteen-byte units. SBO is the eight-row stride.
                D_WIDE = _desc_base(512, 64)
                D_NARROW_K = _desc_base(0, 64)
                d_q_k = _desc_at(D_WIDE, smem_base + K.uint32(OFF_Q))
                d_k_k = _desc_at(D_WIDE, smem_base + K.uint32(OFF_K))
                d_do_k = _desc_at(D_WIDE, smem_base + K.uint32(OFF_DO))
                d_p_k = _desc_at(D_NARROW_K, smem_base + K.uint32(OFF_P))
                d_ds_k = _desc_at(D_NARROW_K, smem_base + K.uint32(OFF_DS))
                d_ds_mn = _desc_at(D_WIDE, smem_base + K.uint32(OFF_DS))
                # M-block b of an MN-major operand is 128 dims on, which is two
                # 64-wide blocks = 16,384 bytes. Index 4 is d576's 64-wide tail
                # (dims 512:576), the source's "block 8".
                d_do_mn = [
                    _desc_at(D_WIDE, smem_base + K.uint32(OFF_DO + b * 16384)) for b in range(4)
                ]
                d_q_mn = [
                    _desc_at(D_WIDE, smem_base + K.uint32(OFF_Q + b * 16384)) for b in range(5)
                ]
                d_k_mn = [
                    _desc_at(D_WIDE, smem_base + K.uint32(OFF_K + b * 16384)) for b in range(5)
                ]

                st_k = K.PipelineState(1, phase=0)
                st_s = K.PipelineState(1, phase=0)
                st_dp = K.PipelineState(1, phase=0)
                st_p = K.PipelineState(1, phase=0)
                st_ds_m = K.PipelineState(1, phase=0)
                st_dkv = K.PipelineState(2, phase=0)
                # `accumulate` for dQ: 0 on the first processed tile, 1 after.
                acc_dq = K.local_scalar(K.i32, init=0)
                # The dKV generations per tile: 2 at d512, 3 at d576.
                # Q and dO arrive once: a CTA owns one query token.
                p_qdo.full.wait(0, 0)
                # dQ accumulates across every tile and is published once, so
                # its pipeline is acquired here and committed after the loop
                # (:1491, :1759) -- not per tile.
                p_dq.empty.wait(0, 1)

                idx = K.local_scalar(K.i32, init=tile_count - K.int32(1))
                with K.While(idx >= K.int32(0)):
                    p_k.full.wait(0, st_k.phase)
                    p_s.empty.wait(0, st_s.phase ^ 1)
                    with K.If(lane == K.int32(0)), K.Then():
                        # S = Q @ K^T over the whole head dimension: one
                        # chain, ACCUMULATE false on the first k-phase and true
                        # after. Q and K share the k walk because both are
                        # K-major over the same 64-wide block stride.
                        _gemm(
                            t_s,
                            (d_q_k, *(512, 2)),
                            (d_k_k, *(512, 2)),
                            idesc["qk"],
                            0,
                            head_dim // 64,
                            4,
                        )
                        K.ptx[TCGEN05_COMMIT](p_s.full.buf.ptr_to([0]))

                    # dP = dO @ V^T. V is columns 0:head_dim_v of the shared KV
                    # buffer, so this reads the same gathered tile with a
                    # shorter K extent -- no separate alias needed.
                    p_dp.empty.wait(0, st_dp.phase ^ 1)
                    with K.If(lane == K.int32(0)), K.Then():
                        _gemm(
                            t_dp,
                            (d_do_k, *(512, 2)),
                            (d_k_k, *(512, 2)),
                            idesc["qk"],
                            0,
                            head_dim_v // 64,
                            4,
                        )
                        K.ptx[TCGEN05_COMMIT](p_dp.full.buf.ptr_to([0]))

                    # --- dKV generation A: dKV0, dKV1 (:1531-1594) ----------
                    # dKV = dO^T @ P, then += Q^T @ dS into the SAME columns:
                    # that accumulation is what fuses dV and dK.
                    p_p.full.wait(0, st_p.phase)
                    p_dkv.empty.wait(st_dkv.stage, st_dkv.phase ^ 1)
                    with K.If(idx != tile_count - K.int32(1)), K.Then():
                        # WAR against the PREVIOUS generation's T2R reads.
                        # Skipped on the first processed tile and compensated
                        # by one unpaired arrive after the loop (:1752-1757).
                        K.ptx.bar.sync(K.uint32(war_gen_a[0]), K.uint32(war_gen_a[1]))
                    with K.If(lane == K.int32(0)), K.Then():
                        for j in range(2):
                            _gemm(
                                t_dkv[j],
                                (d_do_mn[j], *(256, 128)),
                                (d_p_k, *(4, 2)),
                                idesc["dkv"],
                                0,
                                2,
                                2,
                            )
                    p_ds.full.wait(0, st_ds_m.phase)
                    with K.If(lane == K.int32(0)), K.Then():
                        for j in range(2):
                            _gemm(
                                t_dkv[j],
                                (d_q_mn[j], *(256, 128)),
                                (d_ds_k, *(4, 2)),
                                idesc["dkv"],
                                1,
                                2,
                                2,
                            )
                        K.ptx[TCGEN05_COMMIT](p_dkv.full.buf.ptr_to([st_dkv.stage]))
                    st_dkv.advance()

                    if tail:
                        # dKV4 aliases dKV0, so barrier 7 orders its write
                        # against the dKV0/dKV1 reads. Unconditional (:1600).
                        K.ptx.bar.sync(
                            K.uint32(BAR_T2R_DKV01_DONE[0]), K.uint32(BAR_T2R_DKV01_DONE[1])
                        )
                        with K.If(lane == K.int32(0)), K.Then():
                            _gemm(
                                t_dkv4,
                                (d_q_mn[4], *(256, 128)),
                                (d_ds_k, *(4, 2)),
                                idesc["dkv4"],
                                0,
                                2,
                                2,
                            )
                            # No matching acquire upstream: the source leans on
                            # barrier 7 above for what an acquire would give.
                            K.ptx[TCGEN05_COMMIT](p_dkv.full.buf.ptr_to([st_dkv.stage]))
                        st_dkv.advance()

                    # --- dQ0..dQ3 (and dQ4) (:1622-1676) --------------------
                    # dQ accumulates across the WHOLE top-k axis in TMEM and is
                    # published only after the loop, so ACCUMULATE is False
                    # only on the first processed tile.
                    with K.If(lane == K.int32(0)), K.Then():
                        for j in range(4):
                            _gemm(
                                t_dq[j],
                                (d_k_mn[j], *(256, 128)),
                                (d_ds_mn, *(256, 128)),
                                idesc["dq"],
                                acc_dq,
                                2,
                                2,
                            )
                        if tail:
                            _gemm(
                                t_dq4,
                                (d_k_mn[4], *(256, 128)),
                                (d_ds_mn, *(256, 128)),
                                idesc["dq4"],
                                acc_dq,
                                2,
                                2,
                            )

                    # The KV tile's SMEM is reusable from here (:1683).
                    # `p_k`'s empty barrier is tcgen05-flavoured and expects
                    # ONE arrival, so it must be released by a single elected
                    # lane's commit (:1683) -- a plain 32-lane `arrive` lands
                    # 32 arrivals on an expect-1 barrier and walks its parity
                    # forward, which deadlocks the gather warps from the third
                    # tile on. The commit also carries the real ordering: sK is
                    # reusable only once the MMAs reading it have retired.
                    with K.If(lane == K.int32(0)), K.Then():
                        K.ptx[TCGEN05_COMMIT](p_k.empty.buf.ptr_to([0]))

                    # --- dKV generation B: dKV2, dKV3 (:1689-1746) ----------
                    # Unconditional, unlike generation A's WAR barrier.
                    K.ptx.bar.sync(K.uint32(war_gen_b[0]), K.uint32(war_gen_b[1]))
                    p_dkv.empty.wait(st_dkv.stage, st_dkv.phase ^ 1)
                    with K.If(lane == K.int32(0)), K.Then():
                        for j in range(2):
                            _gemm(
                                t_dkv23[j],
                                (d_do_mn[2 + j], *(256, 128)),
                                (d_p_k, *(4, 2)),
                                idesc["dkv"],
                                0,
                                2,
                                2,
                            )
                        K.ptx[TCGEN05_COMMIT](p_p.empty.buf.ptr_to([0]))
                        for j in range(2):
                            _gemm(
                                t_dkv23[j],
                                (d_q_mn[2 + j], *(256, 128)),
                                (d_ds_k, *(4, 2)),
                                idesc["dkv"],
                                1,
                                2,
                                2,
                            )
                        K.ptx[TCGEN05_COMMIT](p_dkv.full.buf.ptr_to([st_dkv.stage]))
                        K.ptx[TCGEN05_COMMIT](p_ds.empty.buf.ptr_to([0]))
                    st_dkv.advance()

                    K.assign(acc_dq, K.int32(1))
                    st_k.advance()
                    st_s.advance()
                    st_dp.advance()
                    st_p.advance()
                    st_ds_m.advance()
                    K.assign(idx, idx - K.int32(1))

                # The generation-A WAR barrier is skipped on the first
                # processed tile, so the reduce warps are one arrival short on
                # the last one. Exactly ONE compensating arrive closes that
                # (:1752-1757) -- one per WAR barrier would hang instead.
                K.ptx.bar.sync(K.uint32(war_gen_a[0]), K.uint32(war_gen_a[1]))
                with K.If(lane == K.int32(0)), K.Then():
                    K.ptx[TCGEN05_COMMIT](p_dq.full.buf.ptr_to([0]))
                    K.ptx[TCGEN05_COMMIT](p_qdo.empty.buf.ptr_to([0]))

            with r_compute:
                rwarp = K.warp_id_in_role()
                rtid = K.tid_in_role()
                with K.If(rwarp == K.int32(0)), K.Then():
                    K.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                        K.cuda.cvta_generic_to_shared(tmem_hold.ptr_to([0])),
                        K.uint32(TMEM_ALLOC_COLUMNS),
                    )
                K.ptx.bar.sync(K.uint32(BAR_TMEM_ALLOC[0]), K.uint32(BAR_TMEM_ALLOC[1]))
                tcol = K.local_scalar("uint32")
                K.ptx.ld.shared.b32(tcol, tmem_hold.ptr_to([0]))
                t_s = K.cuda.get_tmem_addr(tcol, K.uint32(0), K.uint32(TMEM_S_OFFSET))

                t_dp = K.cuda.get_tmem_addr(tcol, K.uint32(16), K.uint32(TMEM_DP_OFFSET))

                # This thread's slice of the 64x64 tile. Both elements of a
                # consecutive pair share a row, so only two sLSE rows are ever
                # read (probe/probe_s_row_coord.py).
                gsub = rtid // K.int32(4)
                base_row = (gsub // K.int32(8)) * K.int32(16) + gsub % K.int32(8)

                p_lse.full.wait(0, 0)
                p_sum.full.wait(0, 0)
                lse_lo = K.local_scalar(K.f32)
                lse_hi = K.local_scalar(K.f32)
                K.ptx["ld.shared.b32"](lse_lo, smem.ptr_to([OFF_LSE + base_row * K.int32(4)]))
                K.ptx["ld.shared.b32"](
                    lse_hi, smem.ptr_to([OFF_LSE + (base_row + K.int32(8)) * K.int32(4)])
                )
                sum_lo = K.local_scalar(K.f32)
                sum_hi = K.local_scalar(K.f32)
                K.ptx["ld.shared.b32"](sum_lo, smem.ptr_to([OFF_SUM + base_row * K.int32(4)]))
                K.ptx["ld.shared.b32"](
                    sum_hi, smem.ptr_to([OFF_SUM + (base_row + K.int32(8)) * K.int32(4)])
                )

                scale_log2e = K.local_scalar(K.f32)
                K.ptx["mul.f32"](scale_log2e, scale, K.float32(1.4426950408889634))

                st_s = K.PipelineState(1, phase=0)
                st_dp = K.PipelineState(1, phase=0)
                st_p = K.PipelineState(1, phase=0)
                st_ds = K.PipelineState(1, phase=0)
                idx = K.local_scalar(K.i32, init=tile_count - K.int32(1))
                with K.While(idx >= K.int32(0)):
                    p_s.full.wait(0, st_s.phase)
                    p_p.empty.wait(0, st_p.phase ^ 1)
                    rs = K.alloc_local([32], "float32")
                    K.ptx["tcgen05.ld.sync.aligned.16x256b.x8.b32"](
                        *[rs[i] for i in range(32)], t_s
                    )

                    # P = exp2(fma(S, scale*log2e, scaled_lse)), two lanes at a
                    # time; the pair shares a row so one lse value serves both.
                    scale2 = K.local_scalar("uint64")
                    K.ptx.mov.b64(scale2, scale_log2e, scale_log2e)
                    lse2_lo = K.local_scalar("uint64")
                    lse2_hi = K.local_scalar("uint64")
                    K.ptx.mov.b64(lse2_lo, lse_lo, lse_lo)
                    K.ptx.mov.b64(lse2_hi, lse_hi, lse_hi)
                    for i in range(0, 32, 2):
                        packed = K.local_scalar("uint64")
                        K.ptx.mov.b64(packed, rs[i], rs[i + 1])
                        addend = lse2_lo if (i % 4) < 2 else lse2_hi
                        K.ptx["fma.rn.f32x2"](packed, packed, scale2, addend)
                        K.ptx.mov.b64(rs[i], rs[i + 1], packed)
                        K.ptx["ex2.approx.ftz.f32"](rs[i], rs[i])
                        K.ptx["ex2.approx.ftz.f32"](rs[i + 1], rs[i + 1])

                    # Quantize and transpose P into sP as the dOP MMA's B
                    # operand. `.trans` is what converts the row-major
                    # (head, kv) accumulator fragment into the head-major
                    # operand layout, with no separate transpose pass.
                    rp = K.alloc_local([16], "uint32")
                    for i in range(16):
                        K.ptx[_CVT_PACK[dtype]](rp[i], rs[2 * i + 1], rs[2 * i])
                    K.ptx["tcgen05.wait::ld.sync.aligned"]()
                    K.ptx.bar.sync(K.uint32(BAR_COMPUTE_SYNC[0]), K.uint32(BAR_COMPUTE_SYNC[1]))
                    for b in range(4):
                        K.ptx["stmatrix.sync.aligned.m8n8.x4.trans.shared.b16"](
                            smem.ptr_to(
                                [
                                    _tile_byte(
                                        OFF_P,
                                        K.int32(16 * b) + rtid % K.int32(16),
                                        (rtid // K.int32(32)) * K.int32(16)
                                        + (rtid % K.int32(32)) // K.int32(16) * K.int32(8),
                                    )
                                ]
                            ),
                            # The four .x4 matrices tile the 16x16 region as
                            # (0,0),(1,0),(0,1),(1,1) while the T2R fragment
                            # walks columns first, so operands 1 and 2 trade
                            # places -- otherwise each region lands transposed.
                            rp[4 * b],
                            rp[4 * b + 2],
                            rp[4 * b + 1],
                            rp[4 * b + 3],
                        )
                    K.ptx["fence.proxy.async.shared::cta"]()
                    p_p.full.arrive(0)
                    st_p.advance()
                    p_s.empty.arrive(0)

                    p_dp.full.wait(0, st_dp.phase)
                    p_ds.empty.wait(0, st_ds.phase ^ 1)
                    rdp = K.alloc_local([32], "float32")
                    K.ptx["tcgen05.ld.sync.aligned.16x256b.x8.b32"](
                        *[rdp[i] for i in range(32)], t_dp
                    )

                    # dS = (dP + sum_OdO) * P * softmax_scale. `sum_OdO` is
                    # already negated upstream (sum_OdO_scale = -1.0, :488), so
                    # this add IS the `dP - delta` of the softmax backward, and
                    # rs still holds P.
                    sum2_lo = K.local_scalar("uint64")
                    sum2_hi = K.local_scalar("uint64")
                    K.ptx.mov.b64(sum2_lo, sum_lo, sum_lo)
                    K.ptx.mov.b64(sum2_hi, sum_hi, sum_hi)
                    scl2 = K.local_scalar("uint64")
                    K.ptx.mov.b64(scl2, scale, scale)
                    for i in range(0, 32, 2):
                        packed = K.local_scalar("uint64")
                        K.ptx.mov.b64(packed, rdp[i], rdp[i + 1])
                        addend = sum2_lo if (i % 4) < 2 else sum2_hi
                        K.ptx["add.rn.f32x2"](packed, packed, addend)
                        pv = K.local_scalar("uint64")
                        K.ptx.mov.b64(pv, rs[i], rs[i + 1])
                        K.ptx["mul.rn.f32x2"](packed, packed, pv)
                        # The `* softmax_scale` is folded into the quantize
                        # upstream (:1938); it is one packed multiply either way.
                        K.ptx["mul.f32x2"](packed, packed, scl2)
                        K.ptx.mov.b64(rdp[i], rdp[i + 1], packed)

                    # dS follows P through the identical transposing store; it
                    # is the dKV MMA's A operand.
                    rq = K.alloc_local([16], "uint32")
                    for i in range(16):
                        K.ptx[_CVT_PACK[dtype]](rq[i], rdp[2 * i + 1], rdp[2 * i])
                    K.ptx["tcgen05.wait::ld.sync.aligned"]()
                    # dP's TMEM is dead once the quantize above has read it, so
                    # it is released before the stores rather than after
                    # (:1943 precedes :1946).
                    p_dp.empty.arrive(0)
                    K.ptx.bar.sync(K.uint32(BAR_COMPUTE_SYNC[0]), K.uint32(BAR_COMPUTE_SYNC[1]))
                    for b in range(4):
                        K.ptx["stmatrix.sync.aligned.m8n8.x4.trans.shared.b16"](
                            smem.ptr_to(
                                [
                                    _tile_byte(
                                        OFF_DS,
                                        K.int32(16 * b) + rtid % K.int32(16),
                                        (rtid // K.int32(32)) * K.int32(16)
                                        + (rtid % K.int32(32)) // K.int32(16) * K.int32(8),
                                    )
                                ]
                            ),
                            rq[4 * b],
                            rq[4 * b + 2],
                            rq[4 * b + 1],
                            rq[4 * b + 3],
                        )
                    K.ptx["fence.proxy.async.shared::cta"]()
                    p_ds.full.arrive(0)
                    st_ds.advance()

                    st_s.advance()
                    st_dp.advance()
                    K.assign(idx, idx - K.int32(1))

                p_lse.empty.arrive(0)
                p_sum.empty.arrive(0)

                # --- dQ epilogue: 4 rounds (5 at d576) (:2033-2140) --------
                # One wait for the whole epilogue: dQ was committed once.
                p_dq.full.wait(0, 0)
                t_dq_e = [
                    (tcol, off)
                    for off in (TMEM_DQ0_OFFSET, TMEM_DQ1_OFFSET, TMEM_DQ2_OFFSET, TMEM_DQ3_OFFSET)
                ]
                for i in range(4):
                    with K.If(rwarp == K.int32(0)), K.Then():
                        # A TMA-store pipeline acquires on the bulk group; it
                        # has no smem barrier at all.
                        K.ptx["cp.async.bulk.wait_group.read"](K.int32(0))
                    K.ptx.bar.sync(K.uint32(BAR_COMPUTE_SYNC[0]), K.uint32(BAR_COMPUTE_SYNC[1]))
                    # 32x32b, NOT the dKV path's 16x256b: this shape is what
                    # makes the fragment match the 128x64 epilogue tile. Thread
                    # t owns dim t; its 64 registers are the 64 heads, eight
                    # heads per issue.
                    rdq = K.alloc_local([64], "float32")
                    for e in range(8):
                        a = K.cuda.get_tmem_addr(
                            t_dq_e[i][0], K.int32(0), K.int32(t_dq_e[i][1] + 8 * e)
                        )
                        K.ptx["tcgen05.ld.sync.aligned.32x32b.x8.b32"](
                            *[rdq[8 * e + k] for k in range(8)], a
                        )
                    K.ptx["tcgen05.wait::ld.sync.aligned"]()
                    # Scalar converts and scalar stores, deliberately: the
                    # source does not vectorize this and it is the largest
                    # instruction count in the kernel.
                    for j in range(64):
                        qv = K.local_scalar("uint16")
                        K.ptx[_CVT_NARROW[dtype]](qv, rdq[j])
                        K.ptx["st.shared.b16"](
                            smem.ptr_to([_tile_byte(OFF_DQ, K.int32(j), rtid)]), qv
                        )
                    # Two barriers around the fence: the first makes every
                    # thread's shared writes done, the second makes the fence
                    # observed before warp 4 issues the TMA (:2545-2552).
                    K.ptx.bar.sync(K.uint32(BAR_COMPUTE_SYNC[0]), K.uint32(BAR_COMPUTE_SYNC[1]))
                    K.ptx["fence.proxy.async.shared::cta"]()
                    K.ptx.bar.sync(K.uint32(BAR_COMPUTE_SYNC[0]), K.uint32(BAR_COMPUTE_SYNC[1]))
                    with K.If(rwarp == K.int32(0)), K.Then():
                        for h in range(2):
                            K.ptx[TMA_S2G](
                                K.address_of(desc_dq),
                                K.int32(i * 128 + h * 64),
                                head_block * K.int32(block),
                                token,
                                smem.ptr_to([OFF_DQ + h * 8192]),
                                K.uint64(0),
                            )
                        K.ptx["cp.async.bulk.commit_group"]()
                    K.ptx.bar.sync(K.uint32(BAR_COMPUTE_SYNC[0]), K.uint32(BAR_COMPUTE_SYNC[1]))
                if tail:
                    # The fifth round (:2116-2135). THIS is the only path by
                    # which d576's dQ columns 512:576 reach mdQ; omitting it
                    # leaves them whatever `torch.empty_like` left there, which
                    # a d512 test cannot detect.
                    with K.If(rwarp == K.int32(0)), K.Then():
                        K.ptx["cp.async.bulk.wait_group.read"](K.int32(0))
                    K.ptx.bar.sync(K.uint32(BAR_COMPUTE_SYNC[0]), K.uint32(BAR_COMPUTE_SYNC[1]))
                    # 16x256b.x2, four issues of 8: M=64 halves the fragment
                    # and the head steps by 16 per issue.
                    rq4 = K.alloc_local([32], "float32")
                    for e in range(4):
                        a4 = K.cuda.get_tmem_addr(
                            tcol, K.int32(0), K.int32(TMEM_DQ4_OFFSET + 16 * e)
                        )
                        K.ptx["tcgen05.ld.sync.aligned.16x256b.x2.b32"](
                            *[rq4[8 * e + k] for k in range(8)], a4
                        )
                    K.ptx["tcgen05.wait::ld.sync.aligned"]()
                    dim_b = (rtid // K.int32(32)) * K.int32(16) + (rtid % K.int32(32)) // K.int32(4)
                    head_b = (rtid % K.int32(4)) * K.int32(2)
                    for j in range(32):
                        qv = K.local_scalar("uint16")
                        K.ptx[_CVT_NARROW[dtype]](qv, rq4[j])
                        # Scalar, not vectorized: the tail's coordinates are
                        # not contiguous per thread (:2596).
                        K.ptx["st.shared.b16"](
                            smem.ptr_to(
                                [
                                    _tile_byte(
                                        OFF_DQ4,
                                        head_b + K.int32((j % 2) + (j // 4) * 8),
                                        dim_b + K.int32(((j // 2) % 2) * 8),
                                    )
                                ]
                            ),
                            qv,
                        )
                    K.ptx.bar.sync(K.uint32(BAR_COMPUTE_SYNC[0]), K.uint32(BAR_COMPUTE_SYNC[1]))
                    K.ptx["fence.proxy.async.shared::cta"]()
                    K.ptx.bar.sync(K.uint32(BAR_COMPUTE_SYNC[0]), K.uint32(BAR_COMPUTE_SYNC[1]))
                    with K.If(rwarp == K.int32(0)), K.Then():
                        K.ptx[TMA_S2G](
                            K.address_of(desc_dq),
                            K.int32(512),
                            head_block * K.int32(block),
                            token,
                            smem.ptr_to([OFF_DQ4]),
                            K.uint64(0),
                        )
                        K.ptx["cp.async.bulk.commit_group"]()
                    K.ptx.bar.sync(K.uint32(BAR_COMPUTE_SYNC[0]), K.uint32(BAR_COMPUTE_SYNC[1]))

                p_dq.empty.arrive(0)
                # producer_tail: drain every outstanding store before the CTA
                # may free sdQ or exit.
                K.ptx["cp.async.bulk.wait_group.read"](K.int32(0))

                with K.If(rwarp == K.int32(0)), K.Then():
                    # Barrier 9 is compute warp 4 plus the 8 reduce warps: the
                    # reduce warps must be drained out of TMEM before its
                    # columns are handed back (:120, :1095).
                    K.ptx.bar.sync(K.uint32(BAR_TMEM_DEALLOC[0]), K.uint32(BAR_TMEM_DEALLOC[1]))
                    K.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](
                        tcol, K.uint32(TMEM_ALLOC_COLUMNS)
                    )

            with r_reduce:
                K.ptx.bar.sync(K.uint32(BAR_TMEM_ALLOC[0]), K.uint32(BAR_TMEM_ALLOC[1]))
                rtcol = K.local_scalar("uint32")
                K.ptx.ld.shared.b32(rtcol, tmem_hold.ptr_to([0]))
                rtid = K.tid_in_role()
                wg = rtid // K.int32(128)  # two warpgroups split the 64 KV rows
                wtid = rtid % K.int32(128)

                def t2r(col):
                    """One sub-tile's 32 f32: two `16x256b.x4` issues.

                    The second issue sits 16 lanes above the first (`:5708`,
                    base + 1048640 = column + (16 << 16)); that lane step is
                    what advances the DIM. The warpgroup steps the COLUMN by
                    32, because columns are the KV rows and the two warpgroups
                    split those (`split_wg`, :2628).
                    """
                    regs = K.alloc_local([32], "float32")
                    for e in range(2):
                        a = K.cuda.get_tmem_addr(
                            rtcol, K.int32(16 * e), K.int32(col) + wg * K.int32(32)
                        )
                        K.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                            *[regs[16 * e + j] for j in range(16)], a
                        )
                    return regs

                def t2r1(col):
                    """The d576 tail: M=64, so ONE issue of 16 (`:2244`).

                    Its dim base steps by 16 per warp rather than 32, because
                    the accumulator is half as tall; reading it with the 128-
                    wide map walks off the end of the 64-dim tail.
                    """
                    regs = K.alloc_local([16], "float32")
                    a = K.cuda.get_tmem_addr(rtcol, K.int32(0), K.int32(col) + wg * K.int32(32))
                    K.ptx["tcgen05.ld.sync.aligned.16x256b.x4.b32"](
                        *[regs[j] for j in range(16)], a
                    )
                    return regs

                # The workspace column group is the thread's index within its
                # warpgroup, NOT a dim: threads 4k..4k+3 share the group and
                # differ only in row, which is what makes the four values of
                # one atomic contiguous (k_api_findings.md).
                col_group = (wtid // K.int32(4)) * K.int32(4)
                row_base = wg * K.int32(32) + (wtid % K.int32(4)) * K.int32(2)

                full_tiles = (topk % K.int32(block)) == K.int32(0)  # :2194

                def rows_for(tile_index):
                    """The 8 KV rows this thread reduces, or -1 where invalid."""
                    r = K.alloc_local([8], "int32")
                    for i in range(8):
                        g = tile_index * K.int32(block) + row_base + K.int32((i // 2) * 8 + (i % 2))
                        K.assign(r[i], K.int32(-1))
                        # With a whole number of tiles every row is in range,
                        # so the read needs no bound (:2202); the ragged case
                        # keeps it (:2205).
                        with K.If(full_tiles):
                            with K.Then():
                                K.ptx["ld.global.b32"](
                                    r[i], topk_idxs.ptr_to([token * K.int32(max_topk) + g])
                                )
                            with K.Else():
                                with K.If(g < topk), K.Then():
                                    K.ptx["ld.global.b32"](
                                        r[i], topk_idxs.ptr_to([token * K.int32(max_topk) + g])
                                    )
                    return r

                # `atom` rather than `red` because the vector forms live only
                # under `atom` in the table; the returned values are dead, which
                # is what the source's own `atom.global.add.v4.f32` does too.
                sink4 = K.alloc_local([4], "float32")

                def reduce4(regs, r, sub_tile):
                    """`atom.global.add.v4.f32` per row: four dims, one row."""
                    for i in range(8):
                        b = i * 2 - i % 2
                        with K.If(r[i] >= K.int32(0)), K.Then():
                            K.ptx["atom.global.add.v4.f32"](
                                sink4[0],
                                sink4[1],
                                sink4[2],
                                sink4[3],
                                ws_dkv.ptr_to(
                                    [r[i] * K.int32(head_dim) + K.int32(sub_tile * 128) + col_group]
                                ),
                                regs[b],
                                regs[b + 2],
                                regs[b + 16],
                                regs[b + 18],
                            )
                    K.ptx.bar.sync(K.uint32(BAR_REDUCE_SYNC[0]), K.uint32(BAR_REDUCE_SYNC[1]))

                st_r = K.PipelineState(2, phase=0)
                ridx = K.local_scalar(K.i32, init=tile_count - K.int32(1))
                with K.While(ridx >= K.int32(0)):
                    r = rows_for(ridx)

                    # --- generation A: dKV0, dKV1 (:2223-2235) --------------
                    p_dkv.full.wait(st_r.stage, st_r.phase)
                    g0 = t2r(TMEM_DKV0_OFFSET)
                    g1 = t2r(TMEM_DKV1_OFFSET)
                    K.ptx["tcgen05.wait::ld.sync.aligned"]()
                    # The registers are drained and the MMA warp released
                    # BEFORE the slow atomics -- that split is the whole point
                    # of the WAR barrier scheme (:2231).
                    K.ptx.bar.sync(K.uint32(BAR_T2R_DKV01_DONE[0]), K.uint32(BAR_T2R_DKV01_DONE[1]))
                    reduce4(g0, r, 0)
                    reduce4(g1, r, 1)
                    p_dkv.empty.arrive(st_r.stage)
                    st_r.advance()

                    if tail:
                        p_dkv.full.wait(st_r.stage, st_r.phase)
                        g4 = t2r1(TMEM_DKV4_OFFSET)
                        K.ptx["tcgen05.wait::ld.sync.aligned"]()
                        K.ptx.bar.sync(
                            K.uint32(BAR_T2R_DKV4_DONE[0]), K.uint32(BAR_T2R_DKV4_DONE[1])
                        )
                        # The tail is 64 dims wide, so a 2-wide atomic.
                        for i in range(8):
                            b = i * 2 - i % 2
                            with K.If(r[i] >= K.int32(0)), K.Then():
                                K.ptx["atom.global.add.v2.f32"](
                                    sink4[0],
                                    sink4[1],
                                    ws_dkv.ptr_to(
                                        [
                                            r[i] * K.int32(head_dim)
                                            + K.int32(512)
                                            + (wtid // K.int32(4)) * K.int32(2)
                                        ]
                                    ),
                                    g4[b],
                                    g4[b + 2],
                                )
                        K.ptx.bar.sync(K.uint32(BAR_REDUCE_SYNC[0]), K.uint32(BAR_REDUCE_SYNC[1]))
                        p_dkv.empty.arrive(st_r.stage)
                        st_r.advance()

                    # --- generation B: dKV2, dKV3 (:2252-2277) --------------
                    p_dkv.full.wait(st_r.stage, st_r.phase)
                    g2 = t2r(dkv2_offset)
                    g3 = t2r(TMEM_DKV3_OFFSET)
                    K.ptx["tcgen05.wait::ld.sync.aligned"]()
                    K.ptx.bar.sync(K.uint32(war_gen_a[0]), K.uint32(war_gen_a[1]))
                    reduce4(g2, r, 2)
                    reduce4(g3, r, 3)
                    p_dkv.empty.arrive(st_r.stage)
                    st_r.advance()

                    K.assign(ridx, ridx - K.int32(1))

                # The reduce warps must not stall here -- they are done (:1095).
                K.ptx["bar.arrive"](K.uint32(BAR_TMEM_DEALLOC[0]), K.uint32(BAR_TMEM_DEALLOC[1]))

            with r_loadkv:
                _bwd_load_kv(
                    kv,
                    topk_idxs,
                    smem,
                    OFF_K,
                    p_k,
                    tile_count,
                    topk,
                    token,
                    tid,
                    lane,
                    head_dim,
                    seqlen_kv,
                    max_topk,
                    has_topk_length,
                    block,
                )

            with r_empty:
                pass

            _unused = (seqlen_q, scale)

    return bwd


# ``dsa_bwd_sm100.py:57-59``: the preprocess block is 8 head-dim lanes by 16
# query lanes, and each thread loads four elements at a time.
SUM_ODO_THREADS_D = 8
SUM_ODO_THREADS_Q = 16
SUM_ODO_ELEM_PER_LOAD = 4


def _shfl_bfly_f32(value, lane_xor):
    """One ``shfl.sync.bfly.b32`` on an f32, reinterpreted through u32."""
    out = K.local_scalar(K.u32)
    K.ptx.shfl_sync.bfly.b32(
        out, K.reinterpret(K.u32, value), K.uint32(lane_xor), K.uint32(31), K.uint32(0xFFFFFFFF)
    )
    return K.reinterpret(K.f32, out)


def _butterfly_sum_f32(value, lane_xors):
    """Sum across the lanes an xor-mask set spans; clamp 31 keeps it in-group."""
    for lane_xor in lane_xors:
        peer = _shfl_bfly_f32(value, lane_xor)
        total = K.local_scalar(K.f32)
        K.ptx.add.f32(total, value, peer)
        value = total
    return value


def make_sum_odo_kernel(*, head_dim, num_head, dtype, max_topk):
    """Trace the delta / sink-folded-LSE preprocess kernel (``:648-707``).

    Writes, per ``(head, query)``, ``sum_OdO = -sum_d(O * dO)`` and
    ``scaled_lse = -(log2(e) * log(exp(lse) + exp(sink)))`` into the two f32
    planes of the LSE/OdO workspace.
    """
    head_dim_v = spec.head_dim_v_for(head_dim)
    block_q = 40 if max_topk == 1024 else 41  # :56, an odd tuned block height
    elem = _ELEM[dtype]
    mul_half = _MUL_HALF[dtype]
    add_wide = _ADD_WIDE[dtype]
    cvt_wide = _CVT_WIDE[dtype]
    d_steps = head_dim_v // SUM_ODO_ELEM_PER_LOAD // SUM_ODO_THREADS_D
    q_arms = (block_q + SUM_ODO_THREADS_Q - 1) // SUM_ODO_THREADS_Q

    @K.kernel(
        warps=(SUM_ODO_THREADS_D * SUM_ODO_THREADS_Q) // 32,
        arch="sm_100a",
        min_blocks_per_sm=1,
        grid=lambda p: [(p["seqlen_q"] + block_q - 1) // block_q, num_head, 1],
    )
    def sum_odo(
        out: K.gptr[elem],
        dout: K.gptr[elem],
        lse: K.gptr[K.f32],
        attn_sink: K.gptr[K.f32],
        ws: K.gptr[K.f32],
        seqlen_q: K.i32,
        plane: K.i32,
        sum_odo_scale: K.f32,
        lse_scale: K.f32,
    ):
        tid = K.thread_id()
        q_block, head, _b = K.cta_id()
        # blockDim was (8, 16, 1) upstream; the flat id splits the same way.
        tidx = tid % SUM_ODO_THREADS_D
        tidy = tid // SUM_ODO_THREADS_D

        log2_e = K.local_scalar(K.f32)
        K.ptx.neg.f32(log2_e, lse_scale)  # lse_scale is -log2(e) (:489)

        for arm in range(q_arms):
            idx_q_t = tidy + arm * SUM_ODO_THREADS_Q
            idx_q = idx_q_t + block_q * q_block
            with K.If((idx_q_t < K.int32(block_q)) & (idx_q < seqlen_q)), K.Then():
                acc = K.local_scalar(K.f32, init=K.float32(0.0))
                base_o = idx_q * (num_head * head_dim_v) + head * head_dim_v
                # A rolled walk over the head dimension, as upstream (`:678`):
                # one body, not `d_steps` copies of it.
                step = K.local_scalar(K.i32, init=K.int32(0))
                with K.While(step < K.int32(d_steps)):
                    d = (tidx + step * K.int32(SUM_ODO_THREADS_D)) * K.int32(SUM_ODO_ELEM_PER_LOAD)
                    ow = K.alloc_local([2], "uint32")
                    dw = K.alloc_local([2], "uint32")
                    K.ptx["ld.global.v2.b32"](ow[0], ow[1], out.ptr_to([base_o + d]))
                    K.ptx["ld.global.v2.b32"](dw[0], dw[1], dout.ptr_to([base_o + d]))
                    narrow = []
                    for half in range(2):
                        prod = K.local_scalar(K.u32)
                        K.ptx[mul_half](prod, ow[half], dw[half])
                        # The narrow multiply is load-bearing: forming the
                        # product in fp32 instead shifts every delta by ~1e-3
                        # relative, and delta feeds dS on every tile.
                        lo = K.local_scalar("uint16")
                        hi = K.local_scalar("uint16")
                        K.ptx["mov.b32"](lo, hi, prod)
                        narrow += [lo, hi]
                    # The 4-element fragment is reduced into f32 first -- one
                    # real widening convert and three widening adds -- and only
                    # the result joins the accumulator, with a plain add.f32
                    # (:682-683). Accumulating each element straight into `acc`
                    # would collapse all three of those into one add family.
                    frag = K.local_scalar(K.f32)
                    K.ptx[cvt_wide](frag, narrow[0])
                    for nv in narrow[1:]:
                        # `add.rn.f32.<narrow>` takes the narrow value first.
                        K.ptx[add_wide](frag, nv, frag)
                    K.ptx["add.f32"](acc, acc, frag)
                    K.assign(step, step + K.int32(1))

                total = _butterfly_sum_f32(acc, (1, 2, 4))

                with K.If(tidx == K.int32(0)), K.Then():
                    lse_v = K.local_scalar(K.f32)
                    sink_v = K.local_scalar(K.f32)
                    K.ptx["ld.global.b32"](lse_v, lse.ptr_to([idx_q * num_head + head]))
                    K.ptx["ld.global.b32"](sink_v, attn_sink.ptr_to([head]))

                    lse_log2 = K.local_scalar(K.f32)
                    sink_log2 = K.local_scalar(K.f32)
                    K.ptx["mul.f32"](lse_log2, lse_v, log2_e)
                    K.ptx["mul.f32"](sink_log2, sink_v, log2_e)
                    m = K.local_scalar(K.f32)
                    K.ptx["max.f32"](m, lse_log2, sink_log2)
                    # exp2 of each shifted term, summed, then log2: the
                    # numerically safe fold of the sink into the denominator.
                    a = K.local_scalar(K.f32)
                    b = K.local_scalar(K.f32)
                    K.ptx["sub.f32"](a, lse_log2, m)
                    K.ptx["sub.f32"](b, sink_log2, m)
                    ea = K.local_scalar(K.f32)
                    eb = K.local_scalar(K.f32)
                    K.ptx["ex2.approx.f32"](ea, a)
                    K.ptx["ex2.approx.f32"](eb, b)
                    ssum = K.local_scalar(K.f32)
                    K.ptx["add.f32"](ssum, ea, eb)
                    lg = _log2_soft(ssum)
                    with_sink = K.local_scalar(K.f32)
                    K.ptx["add.f32"](with_sink, m, lg)
                    scaled = K.local_scalar(K.f32)
                    K.ptx["neg.f32"](scaled, with_sink)

                    # lse == +inf means the row selected nothing; the sink must
                    # not resurrect it (:703-704).
                    is_inf = K.local_scalar("uint32")
                    K.ptx["setp.eq.f32"](is_inf, lse_v, K.float32(float("inf")))
                    out_lse = K.local_scalar(K.f32)
                    K.ptx["selp.f32"](out_lse, K.float32(float("-inf")), scaled, K.ptx.pred(is_inf))

                    scaled_odo = K.local_scalar(K.f32)
                    K.ptx["mul.f32"](scaled_odo, sum_odo_scale, total)
                    slot = idx_q * num_head + head
                    K.ptx["st.global.b32"](ws.ptr_to([slot]), scaled_odo)
                    K.ptx["st.global.b32"](ws.ptr_to([plane + slot]), out_lse)

    return sum_odo


def make_convert_kernel(*, head_dim, dtype, max_topk):
    """Trace the FP32 dKV workspace to element-dtype conversion (``:609-646``).

    Inverts the two fragment scrambles ``reduce_dKV`` wrote with: the 128-wide
    sub-tiles used a 4-wide atomic and the d576 tail a 2-wide one, so the two
    halves need different unscramble maps. Every access here is scalar --
    ``convert_elem_per_load = 4`` does not reach it.
    """
    elem = _ELEM[dtype]
    head_dim_main = (head_dim // 128) * 128
    has_tail = head_dim_main != head_dim
    # ``:568-570``: both keyed off max_topk.
    block_seq = 4 if max_topk == 2048 else 32
    threads_seq = 4 if max_topk == 2048 else block_seq
    threads_d = 32

    @K.kernel(
        warps=(threads_d * threads_seq) // 32,
        arch="sm_100a",
        grid=lambda p: [(p["seqlen_kv"] + block_seq - 1) // block_seq, 1, 1],
    )
    def convert(ws_dkv: K.gptr[K.f32], dkv: K.gptr[elem], seqlen_kv: K.i32):
        # ---- kernel body starts here ----
        seq_block, _y, _z = K.cta_id()
        tid = K.thread_id()
        tidx = tid % threads_d
        tidy = tid // threads_d

        seq_id = block_seq * seq_block + tidy
        with K.If(seq_id < seqlen_kv), K.Then():
            row = seq_id * head_dim
            for i in range(head_dim_main // 64):
                for j in range(2):
                    src = tidx + j * 32 + i * 64
                    v = K.local_scalar(K.f32)
                    K.ptx["ld.global.b32"](v, ws_dkv.ptr_to([row + src]))
                    # Inverse of the 128-wide store's 4-wide fragment gather:
                    # a 4x8 transpose inside each 32-element block.
                    dim = tidx // 4 + (tidx % 4) * 8 + j * 32 + i * 64
                    nv = K.local_scalar("uint16")
                    K.ptx[_CVT_NARROW[dtype]](nv, v)
                    K.ptx["st.global.b16"](dkv.ptr_to([row + dim]), nv)
            if has_tail:
                for j in range(2):
                    src = tidx + j * 32 + head_dim_main
                    v = K.local_scalar(K.f32)
                    K.ptx["ld.global.b32"](v, ws_dkv.ptr_to([row + src]))
                    # A DIFFERENT inverse: the 64-wide tail store used a 2-wide
                    # fragment, so its scramble is not the 128-wide one.
                    kk = tidx // 2 + j * 16
                    dim = head_dim_main + (kk // 8) * 16 + kk % 8 + (tidx % 2) * 8
                    nv = K.local_scalar("uint16")
                    K.ptx[_CVT_NARROW[dtype]](nv, v)
                    K.ptx["st.global.b16"](dkv.ptr_to([row + dim]), nv)

    return convert


# ``dsa_bwd_sm100.py:60-61``: the sink-gradient kernel is one warp per 256
# queries.
DSINK_BLOCK_Q = 256
DSINK_THREADS = 32


def make_sum_dsink_kernel(*, num_head):
    """Trace the attention-sink gradient reduction kernel (``:709-738``).

    ``d_sink[h] += sum_q exp2(sink*log2e + scaled_lse[h,q]) * sum_OdO[h,q]``,
    warp-reduced then accumulated with one scalar atomic per CTA.
    """

    @K.kernel(
        warps=DSINK_THREADS // 32,
        arch="sm_100a",
        min_blocks_per_sm=1,
        grid=lambda p: [(p["seqlen_q"] + DSINK_BLOCK_Q - 1) // DSINK_BLOCK_Q, num_head, 1],
    )
    def sum_dsink(
        ws: K.gptr[K.f32],
        attn_sink: K.gptr[K.f32],
        d_sink: K.gptr[K.f32],
        seqlen_q: K.i32,
        plane: K.i32,
    ):
        # ---- kernel body starts here ----
        q_block, head, _b = K.cta_id()
        tid = K.thread_id()

        log2_e = K.float32(1.4426950408889634)
        sink_v = K.local_scalar(K.f32)
        K.ptx["ld.global.b32"](sink_v, attn_sink.ptr_to([head]))
        sink_log2 = K.local_scalar(K.f32)
        K.ptx["mul.f32"](sink_log2, sink_v, log2_e)

        # `q_end` clamps the last block; the stride is the warp width, so a
        # thread walks its own column of the 256-query block.
        q_end = K.local_scalar(K.i32, init=(q_block + 1) * DSINK_BLOCK_Q)
        with K.If(q_end > seqlen_q), K.Then():
            K.assign(q_end, seqlen_q)

        acc = K.local_scalar(K.f32, init=K.float32(0.0))
        q_idx = K.local_scalar(K.i32, init=q_block * DSINK_BLOCK_Q + tid)
        with K.While(q_idx < q_end):
            slot = q_idx * num_head + head
            sl = K.local_scalar(K.f32)
            K.ptx["ld.global.b32"](sl, ws.ptr_to([plane + slot]))
            arg = K.local_scalar(K.f32)
            K.ptx["add.f32"](arg, sink_log2, sl)
            p_sink = K.local_scalar(K.f32)
            K.ptx["ex2.approx.f32"](p_sink, arg)
            odo = K.local_scalar(K.f32)
            K.ptx["ld.global.b32"](odo, ws.ptr_to([slot]))
            K.ptx["fma.rn.f32"](acc, p_sink, odo, acc)
            K.assign(q_idx, q_idx + DSINK_THREADS)

        total = _butterfly_sum_f32(acc, (1, 2, 4, 8, 16))

        with K.If(tid == 0), K.Then():
            prev = K.local_scalar(K.f32)
            K.ptx.atom.global_.add.f32(prev, d_sink.ptr_to([head]), total)

    return sum_dsink


def get_kernel(**config):
    """Return the four device functions in the upstream launch order."""
    head_dim = config["head_dim"]
    num_head = config["num_head"]
    dtype = config["dtype"]
    max_topk = config["max_topk"]
    has_topk_length = config["has_topk_length"]
    return [
        make_sum_odo_kernel(
            head_dim=head_dim, num_head=num_head, dtype=dtype, max_topk=max_topk
        ).func,
        make_bwd_kernel(
            head_dim=head_dim,
            num_head=num_head,
            dtype=dtype,
            max_topk=max_topk,
            has_topk_length=has_topk_length,
        ).func,
        make_convert_kernel(head_dim=head_dim, dtype=dtype, max_topk=max_topk).func,
        make_sum_dsink_kernel(num_head=num_head).func,
    ]
