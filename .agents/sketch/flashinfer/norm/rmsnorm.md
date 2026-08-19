<!--
Copyright (c) 2025 by FlashInfer team.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: Apache-2.0

This execution sketch documents a TIRx port of FlashInfer's CuTe-DSL
RMSNormKernel. See LICENSE, NOTICE, and licenses/ for applicable terms.
-->

# FlashInfer 2-D RMSNorm / Gemma RMSNorm SM100: execution sketch

This file is a non-executable execution sketch. It freezes the implementation
shape of FlashInfer's CuTe-DSL `RMSNormKernel` for the paired target
[`tirx_kernels/flashinfer/norm/rmsnorm.py`](../../../../tirx_kernels/flashinfer/norm/rmsnorm.py).
The target may become executable only after this sketch passes independent
source/PTX review; after the first PASS this file is permanently frozen.

The source is fixed at FlashInfer commit
`f2e04400e330fb2debe0bf8730d9424a1d37927f`:

- `flashinfer/norm/kernels/rmsnorm.py`, SHA256
  `b273fe5444aaf86a1600c196817b7a733b18f6f82030a16e1ef2731c784d48f0`;
- `flashinfer/norm/utils.py`, SHA256
  `3f44ac6727c58883420068bf0aa5b239b12d2e86819ad80e54bc1bc016ec881a`;
- public dispatch in `flashinfer/norm/__init__.py`, SHA256
  `226a88f5fb14e78e06e1be79020edcae01bfa9e53e677bd485373ea4d51cffcb`.

The shared family covers FP16/BF16, `weight_bias=0.0` RMSNorm and
`weight_bias=1.0` Gemma RMSNorm, compact and explicit Int64 row-strided
addressing, PDL off/on, synchronous and `cp.async` input traffic, sub-warp,
one-warp, multi-warp, and cluster reduction, and cluster sizes 1/2/4/8/16.
Only the 2-D source body is in scope. QK RMSNorm, fused-add, quantization,
legacy CUDA, and every tile primitive are out of scope.

## Source dispatch and resource formulae

These are compile-time formulae copied mechanically from `RMSNormKernel`.
They are part of the sketch, not tuning choices:

```python
ELEM_BYTES = 2
MAX_VEC = 128 // 8 // ELEM_BYTES                         # 8 elements

def threads_per_row(h_per_cta):
    if h_per_cta <= 64: return 8
    if h_per_cta <= 128: return 16
    if h_per_cta <= 3072: return 32
    if h_per_cta <= 6144: return 64
    if h_per_cta <= 16384: return 128
    return 256

def num_threads(h_per_cta):
    return 128 if h_per_cta <= 16384 else 256

def source_config(H):
    # Select the first feasible divisor exactly in this order. Source fallback
    # remains 16 even if no candidate was feasible.
    for candidate in (1, 2, 4, 8, 16):
        if H % candidate == 0 and estimate_smem(H, candidate) <= 232448:
            cluster_n = candidate
            break
    else:
        cluster_n = 16

    H_per_cta = H // cluster_n
    tpr = threads_per_row(H_per_cta)
    threads = num_threads(H_per_cta)
    rows = threads // tpr
    warps_per_row = max(tpr // 32, 1)
    vec = min(H_per_cta & -H_per_cta, MAX_VEC)
    copy_bits = 16 * vec
    vec_blocks = max(1, ceil_div(H_per_cta // vec, tpr))
    cols = vec * vec_blocks * tpr
    tile_bytes = rows * cols * ELEM_BYTES
    use_async = copy_bits >= 32 and tile_bytes <= 232448 // 2
    reduce_bytes = rows * warps_per_row * 4
    if cluster_n > 1:
        reduce_bytes *= cluster_n
    smem_bytes = (tile_bytes if use_async else 0) + reduce_bytes
    if cluster_n > 1:
        smem_bytes += 8
    return cluster_n, H_per_cta, tpr, threads, rows, warps_per_row, \
           vec, copy_bits, vec_blocks, cols, use_async, smem_bytes

def estimate_smem(H, cluster_n):
    # Same derived values as above, but the estimate always includes the X
    # tile even when the eventual body selects synchronous register loads.
    H_per_cta = H // cluster_n
    tpr = threads_per_row(H_per_cta)
    threads = num_threads(H_per_cta)
    rows = threads // tpr
    warps_per_row = max(tpr // 32, 1)
    vec = min(H_per_cta & -H_per_cta, MAX_VEC)
    vec_blocks = max(1, ceil_div(H_per_cta // vec, tpr))
    cols = vec * vec_blocks * tpr
    bytes_ = rows * cols * ELEM_BYTES
    bytes_ += rows * warps_per_row * cluster_n * 4
    return bytes_ + (8 if cluster_n > 1 else 0)
```

The host selects compact only when both logical tensors are contiguous and
`M*H <= 2**31-1`. Otherwise it passes explicit `x_row_stride:int64` and
`y_row_stride:int64` in elements. This predicate must not be weakened.

## Pipeline at a glance

| lanes / CTA role | source-owned work | publication or reuse edge |
| --- | --- | --- |
| each contiguous group of `threads_per_row` threads | one logical row; each thread owns `vec_size` adjacent columns in every vector block | no communication during X/W load |
| each sub-warp/full warp within a row | ordered local sum followed by butterfly sum at widths 8/16/32 | shuffle result is replicated within its reduction subgroup |
| lane 0 of every row warp, cluster 1 multi-warp path | publish one partial to `(row,warp_in_row)` | one CTA barrier before final warp load |
| lanes `<cluster_n` of every warp, clustered path | send that warp's partial to the same slot on every peer CTA | remote `st.async.shared::cluster...mbarrier::complete_tx` and one local mbarrier wait |
| every row warp after publication | redundantly load the row's `warps_per_row*cluster_n` partials and full-warp reduce | the replicated row sum in every row warp feeds FP32 mean/rsqrt |
| all threads after reduction | clustered arrive/wait or CTA barrier, reload staged X when async, scale and store | barrier protects shared X reuse and completes source reduction phase |

There is one `cp.async` group, not a multi-stage pipeline. Weight is loaded
synchronously between async commit and wait. PDL wait is the first executable
operation; PDL signal is the last. PDL-disabled specializations contain no PDL
instruction and no launch attribute.

## Primitive vocabulary

Structural operations do not move or compute values:

```python
specialize(...)       # compile-time family member
launch(...)           # exact grid, block, cluster, dynamic shared, PDL attrs
raw_shared(...)       # one dynamic shared allocation
view(...)             # typed view at an explicit byte offset
reg_tile(...)         # thread-private register fragment
address(...)          # compact or explicit Int64 backing-store address
pair_view(...)        # register-only adjacent-lane view; no instruction
scalar_pair(...)      # {scalar,scalar}, except {scalar,undefined} for final odd pair
```

Data movement, arithmetic, and synchronization remain primitive:

```python
copy_g2r(src, dst, bits, predicate=None)
copy_g2s_async_ca(src, dst, bits, source_bytes)
copy_s2r(src, dst, bits)
copy_r2s(src, dst)
copy_r2g(src, dst, bits, predicate=None)
cast(dtype, src, rounding=None)
mul(lhs, rhs, lanes=1)
add(lhs, rhs, lanes=1)
fma(lhs, rhs, acc)
div(lhs, rhs)
rsqrt_fast(src)
shuffle_xor(src, delta, width)
elect_one()
cp_async_commit()
cp_async_wait(group)
cta_barrier()
cluster_arrive_relaxed()
cluster_wait()
mbarrier_init(ptr, arrivals)
mbarrier_init_fence()
mbarrier_arrive_expect_tx(ptr, bytes)
map_shared_to_peer(ptr, peer)
remote_store_complete_tx(value, dst, mbarrier)
mbarrier_try_wait_parity(ptr, phase, timeout)
pdl_wait()
pdl_signal()
```

A stated vector copy is exactly one reviewed copy. The address-space-specific
table is:

| `VEC` | global X/W load and Y store | shared X load | reviewed narrowing |
| ---: | --- | --- | --- |
| 1 | scalar `ld/st.global.b16` | n/a in the reviewed sync-vec1 path | scalar `cvt.rn.{f16,bf16}.f32` |
| 2 | `ld/st.global.v2.b16` | `ld.shared.v2.b16` | scalar `cvt.rn.{f16,bf16}.f32` for all six values of the reviewed `(VEC=2,VEC_BLOCKS=3)` H66 profile |
| 4 | `ld/st.global.v2.b32` | `ld.shared.v4.b16` | `cvt.rn.{f16,bf16}x2.f32`, two issues per vector block |
| 8 | `ld/st.global.v4.b32` | `ld.shared.v4.b32` | `cvt.rn.{f16,bf16}x2.f32`, four issues per vector block |

Every async row copy is `cp.async.ca.shared.global`, with immediate copy size
`COPY_BITS/8` and the same runtime source-size operand for tail zero-fill.
`mul(..., lanes=2)` and `add(..., lanes=2)` name one native packed-FP32
instruction. There is deliberately no compound `rmsnorm`,
`row_reduce`, `block_reduce`, `cluster_reduce`, tile-copy, or tile-compute
primitive.

## Complete sketch

```python
@specialize(
    VARIANT_WEIGHT_BIAS=(("rmsnorm", 0.0), ("gemma_rmsnorm", 1.0)),
    DTYPE=("f16", "bf16"),
    H="positive compile-time integer",
    LAYOUT=("compact", "strided_i64"),
    ENABLE_PDL=(False, True),
    TARGET="sm_100a",
)
@launch(
    grid=(ceil_div(runtime_M, ROWS_PER_BLOCK), CLUSTER_N, 1),
    block=(NUM_THREADS, 1, 1),
    cluster=((1, CLUSTER_N, 1) if CLUSTER_N > 1 else None),
    dynamic_smem_bytes=SMEM_BYTES,
    use_programmatic_dependent_launch=ENABLE_PDL,
)
def flashinfer_rmsnorm(
    x_storage,             # DTYPE one-dimensional backing storage
    weight,                # DTYPE [H], compact
    y_storage,             # DTYPE one-dimensional backing storage
    runtime_M: i64,
    runtime_eps: f32,
    x_row_stride: i64 = H, # present only in strided_i64 specialization
    y_row_stride: i64 = H, # present only in strided_i64 specialization
):
    tid = thread_id_x(NUM_THREADS)
    block_x = block_id_x()
    block_y = 0                  # cluster-1 source specialization folds this
    cta_rank = 0
    if CLUSTER_N > 1:
        block_y = block_id_y()   # global H-tile coordinate, source blockIdx.y
        cta_rank = cta_rank_in_cluster()  # flat DSMEM source-slot rank

    if ENABLE_PDL:
        pdl_wait()
        # instruction_selection: griddepcontrol.wait; extent: one issue per CTA

    # CuTe TV layout:
    # shape=((TPR,ROWS),(VEC,VEC_BLOCKS))
    # stride=((VEC*ROWS,1),(ROWS,ROWS*VEC*TPR)).
    row_in_cta = tid // TPR
    thread_in_row = tid % TPR
    row_i32 = block_x * ROWS + row_in_cta
    row_i64 = cast_i64(row_i32)
    row_valid = row_i64 < runtime_M
    warp = tid // 32
    lane = tid % 32
    WARPS_PER_ROW = max(TPR // 32, 1)
    row_warp = warp // WARPS_PER_ROW
    warp_in_row = warp % WARPS_PER_ROW

    smem = raw_shared("u8", SMEM_BYTES, alignment=16)
    cursor = 0
    if USE_ASYNC:
        sX = view(smem, DTYPE, (ROWS, COLS_PER_TILE),
                  byte_offset=cursor, row_major=True, alignment=16)
        cursor += ROWS * COLS_PER_TILE * 2
    if CLUSTER_N == 1:
        reduction = view(smem, "f32", shape=(ROWS, WARPS_PER_ROW),
                         stride=(1, ROWS), byte_offset=cursor, alignment=4)
        cursor += ROWS * WARPS_PER_ROW * 4
    else:
        reduction = view(
            smem, "f32", shape=(ROWS, (WARPS_PER_ROW, CLUSTER_N)),
            stride=(1, (ROWS, ROWS * WARPS_PER_ROW)),
            byte_offset=cursor, alignment=4,
        )
        cursor += ROWS * WARPS_PER_ROW * CLUSTER_N * 4
        mbar = view(smem, "mbarrier", (1,), byte_offset=cursor, alignment=8)

    if CLUSTER_N > 1:
        if tid == 0:
            mbarrier_init(mbar, arrivals=1)
            # instruction_selection: mbarrier.init.shared.b64; extent: one per CTA
        mbarrier_init_fence()
        # instruction_selection: fence.mbarrier_init.release.cluster; extent: one per CTA
        cluster_arrive_relaxed()
        # instruction_selection: barrier.cluster.arrive.relaxed; extent: one per CTA
        cluster_wait()
        # instruction_selection: barrier.cluster.wait; extent: one per CTA;
        # the reviewed source PTX has no .acquire modifier

    # Source fragment layout is (VEC,VEC_BLOCKS):(1,VEC). Flattened adjacent
    # values therefore use flat = v + VEC*vb, including pairs spanning vector
    # block boundaries when VEC==1.
    x_frag = reg_tile(DTYPE, shape=(VEC,VEC_BLOCKS), stride=(1,VEC))
    w_frag = reg_tile(DTYPE, shape=(VEC,VEC_BLOCKS), stride=(1,VEC))
    x_f32 = reg_tile("f32", shape=(VEC,VEC_BLOCKS), stride=(1,VEC))
    w_f32 = reg_tile("f32", shape=(VEC,VEC_BLOCKS), stride=(1,VEC))
    TOTAL_VALUES = VEC * VEC_BLOCKS
    PACKED_PAIRS = ceil_div(TOTAL_VALUES, 2)

    # ----------------------------------------------------------------------
    # Pass 1: X and W traffic. Every address in strided_i64 uses i64 from
    # row-stride multiplication through final pointer addition.
    # ----------------------------------------------------------------------
    if not USE_ASYNC:
        for vb in static_range(VEC_BLOCKS):
            for v in static_range(VEC):
                x_frag[v,vb] = zero(DTYPE)
                # instruction_selection: mov.b16 zero initialization propagated
                # across the complete fragment; reviewed sync-vec1 H111 emits
                # seven mov.b16 issues for its seven physical values

    for vb in static_range(VEC_BLOCKS):
        local_col = (thread_in_row + vb * TPR) * VEC
        absolute_col = block_y * COLS_PER_TILE + local_col
        col_valid = absolute_col < H

        if USE_ASYNC:
            if row_valid:
                x_offset = (row_i32 * H + absolute_col) if COMPACT else \
                           (row_i64 * x_row_stride + absolute_col)
                source_bytes = (COPY_BITS // 8) if col_valid else 0
                copy_g2s_async_ca(
                    address(x_storage, x_offset),
                    sX[row_in_cta, local_col:local_col+VEC],
                    bits=COPY_BITS,
                    source_bytes=source_bytes,
                )
                # instruction_selection: cp.async.ca.shared.global with
                # immediate size COPY_BITS/8 and runtime source-size operand;
                # extent: one issue per vb for an in-bounds row, using zero
                # source bytes to zero-fill an invalid column tail
        else:
            if row_valid and col_valid:
                x_offset = (row_i32 * H + absolute_col) if COMPACT else \
                           (row_i64 * x_row_stride + absolute_col)
                copy_g2r(address(x_storage, x_offset), x_frag[:,vb], COPY_BITS)
                # instruction_selection: exact global load from the VEC table;
                # extent: one row-and-column-predicated vector per vb

    if USE_ASYNC:
        cp_async_commit()
        # instruction_selection: cp.async.commit_group; extent: one per CTA thread

    # Weight is source-ordered after X issue/commit and before async wait.
    for vb in static_range(VEC_BLOCKS):
        local_col = (thread_in_row + vb * TPR) * VEC
        absolute_col = block_y * COLS_PER_TILE + local_col
        if absolute_col < H:
            copy_g2r(weight[absolute_col:absolute_col+VEC], w_frag[:,vb], COPY_BITS)
            # instruction_selection: exact global load from the VEC table;
            # extent: one column-predicated vector per vb

    if USE_ASYNC:
        cp_async_wait(0)
        # instruction_selection: cp.async.wait_group 0; extent: one per CTA thread
        for vb in static_range(VEC_BLOCKS):
            local_col = (thread_in_row + vb * TPR) * VEC
            copy_s2r(sX[row_in_cta,local_col:local_col+VEC], x_frag[:,vb], COPY_BITS)
            # instruction_selection: exact shared load from the VEC table;
            # extent: one vector per vb

    # Preserve source fragment order: widen all loaded X values, form squares,
    # then reduce the complete fragment in its source profile order.
    # Phase 1a: widen the complete fragment before any square.
    for vb in static_range(VEC_BLOCKS):
        for v in static_range(VEC):
            x_f32[v,vb] = cast("f32", x_frag[v,vb])
            # instruction_selection: scalar f16/bf16-to-f32 conversion;
            # extent: every physical fragment value, source value order

    # Phase 1b: square every adjacent flattened pair. The last high lane when
    # TOTAL_VALUES is odd is an emitted but semantically unused register lane;
    # its square is not consumed by Phase 1c.
    x_sq = reg_tile("f32", shape=(TOTAL_VALUES,))
    for pair in static_range(PACKED_PAIRS):
        square_pair = mul(
            pair_view(x_f32, pair, unused_high_if_odd=True),
            pair_view(x_f32, pair, unused_high_if_odd=True), lanes=2)
        # instruction_selection: mul.f32x2; extent: ceil(TOTAL_VALUES/2)
        # packed issues, including four issues for the seven-value H111 case
        for packed_lane in static_range(2):
            flat = pair * 2 + packed_lane
            if flat < TOTAL_VALUES:
                x_sq[flat] = square_pair[packed_lane]

    # Phase 1c: only valid flattened values participate, in source order.
    local_sum = 0.0
    for flat in static_range(TOTAL_VALUES):
        local_sum = add(local_sum, x_sq[flat])
        # instruction_selection: add.f32; extent: one ordered issue per valid
        # fragment value (seven, not eight, for reviewed H111)

    # ----------------------------------------------------------------------
    # row_reduce_sum_multirow, expanded without a compound reduction helper.
    # ----------------------------------------------------------------------
    WARP_WIDTH = min(TPR, 32)
    for delta in powers_of_two_below(WARP_WIDTH):   # 1,2,4,(8),(16)
        peer = shuffle_xor(local_sum, delta, width=32)
        # instruction_selection: shfl.sync.bfly.b32; extent: one scalar at
        # every explicit stage, member mask -1 and clamp operand 31; subgroup
        # width limits only the number of offsets
        local_sum = add(local_sum, peer)
        # instruction_selection: add.f32; extent: one scalar per stage
    warp_sum = local_sum

    if WARPS_PER_ROW > 1 and CLUSTER_N == 1:
        if lane == 0:
            copy_r2s(warp_sum, reduction[row_warp,warp_in_row])
            # instruction_selection: st.shared.b32; extent: one per row warp
        cta_barrier()
        # instruction_selection: bar.sync; extent: one CTA-wide publication edge
        final = 0.0
        if lane < WARPS_PER_ROW:
            final = copy_s2r(reduction[row_warp,lane])
            # instruction_selection: ld.shared.b32; extent: one scalar per
            # participating lane of every row warp
        for delta in (1,2,4,8,16):
            peer = shuffle_xor(final, delta, width=32)
            # instruction_selection: shfl.sync.bfly.b32, mask -1, clamp 31;
            # extent: one scalar at each full-warp stage in every row warp
            final = add(final, peer)
            # instruction_selection: add.f32; extent: one scalar per stage
        sum_sq = final

    elif CLUSTER_N > 1:
        if warp == 0:
            if elect_one():
                # instruction_selection: elect.sync; extent: one election in warp 0
                EXPECTED_BYTES = ROWS * WARPS_PER_ROW * CLUSTER_N * 4
                mbarrier_arrive_expect_tx(mbar, EXPECTED_BYTES)
                # instruction_selection: mbarrier.arrive.expect_tx.shared.b64;
                # extent: one elected active-lane issue per CTA
        if lane < CLUSTER_N:
            peer_reduction = map_shared_to_peer(
                reduction[row_warp,(warp_in_row,cta_rank)], lane)
            peer_mbar = map_shared_to_peer(mbar, lane)
            # instruction_selection: mapa.shared::cluster for destination and
            # barrier; extent: two addresses per sending lane
            remote_store_complete_tx(warp_sum, peer_reduction, peer_mbar)
            # instruction_selection:
            # st.async.shared::cluster.mbarrier::complete_tx::bytes.f32;
            # extent: one partial from every warp to every peer CTA
        wait_done = False
        while not wait_done:
            wait_done = mbarrier_try_wait_parity(mbar, phase=0, timeout=10000000)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 with
            # parity 0 and timeout 10000000, followed by bra.uni retry;
            # extent: one uniform retry loop per CTA thread

        total_partials = WARPS_PER_ROW * CLUSTER_N
        final = 0.0
        for i in static_range(ceil_div(total_partials, 32)):
            partial = lane + i * 32
            if partial < total_partials:
                partial_value = copy_s2r(reduction[row_warp,partial])
                # instruction_selection: ld.shared.b32; extent: one scalar
                # for each owned partial in every row warp
                final = add(final, partial_value)
                # instruction_selection: add.f32; extent: one per loaded partial
        for delta in (1,2,4,8,16):
            peer = shuffle_xor(final, delta, width=32)
            # instruction_selection: shfl.sync.bfly.b32, mask -1, clamp 31;
            # extent: one scalar at each full-warp stage in every row warp
            final = add(final, peer)
            # instruction_selection: add.f32; extent: one scalar per stage
        sum_sq = final

    else:
        sum_sq = warp_sum

    if is_power_of_two(H):
        shifted = fma(sum_sq, exact_float32_reciprocal(H), runtime_eps)
        # instruction_selection: fma.rn.f32; extent: one scalar per thread in
        # reviewed H64/H4096 and cluster H131072/H262144/H524288/H1048576
    else:
        mean_sq = div(sum_sq, float32(H))
        # instruction_selection: div.rn.f32; extent: one scalar per thread in
        # reviewed H66/H111/H500
        shifted = add(mean_sq, runtime_eps)
        # instruction_selection: add.f32; extent: one scalar per thread
    rstd = rsqrt_fast(shifted)
    # instruction_selection: rsqrt.approx.ftz.f32; extent: one scalar per thread

    if CLUSTER_N > 1:
        cluster_arrive_relaxed()
        cluster_wait()
        # instruction_selection: barrier.cluster.arrive.relaxed followed by
        # barrier.cluster.wait (no .acquire); extent: one pair per CTA
    else:
        cta_barrier()
        # instruction_selection: bar.sync; extent: one post-reduction CTA edge

    # ----------------------------------------------------------------------
    # Pass 2: async source reload, weight bias, scale, cast, predicated store.
    # The synchronous path keeps x_f32 live across reduction exactly as source.
    # ----------------------------------------------------------------------
    if USE_ASYNC:
        # Phase 2a: reload the entire input fragment before any conversion.
        for vb in static_range(VEC_BLOCKS):
            local_col = (thread_in_row + vb * TPR) * VEC
            copy_s2r(sX[row_in_cta,local_col:local_col+VEC], x_frag[:,vb], COPY_BITS)
            # instruction_selection: exact shared load from the VEC table;
            # extent: one vector per vb after the reduction barrier

        # Phase 2b: convert the complete reloaded X fragment.
        for flat in static_range(TOTAL_VALUES):
            x_f32.flat[flat] = cast("f32", x_frag.flat[flat])
            # instruction_selection: scalar f16/bf16-to-f32 conversion;
            # extent: every physical X value

    # Phase 2c: convert the complete W fragment. Invalid column-tail registers
    # are semantically dead but still follow the source conversion/dataflow.
    for flat in static_range(TOTAL_VALUES):
        w_f32.flat[flat] = cast("f32", w_frag.flat[flat])
        # instruction_selection: scalar f16/bf16-to-f32 conversion;
        # extent: every physical W value

    # Phase 2d: first packed multiply over the complete flattened fragment.
    normalized = reg_tile("f32", shape=(TOTAL_VALUES,))
    for pair in static_range(PACKED_PAIRS):
        normalized_pair = mul(
            pair_view(x_f32, pair, unused_high_if_odd=True),
            scalar_pair(rstd, pair, TOTAL_VALUES,
                        undefined_high_if_odd=True), lanes=2)
        # instruction_selection: mul.f32x2; extent: PACKED_PAIRS issues
        for packed_lane in static_range(2):
            flat = pair * 2 + packed_lane
            if flat < TOTAL_VALUES:
                normalized[flat] = normalized_pair[packed_lane]

    # Phase 2e: bias add is physically present for both public variants,
    # including packed +0.0 in ordinary RMSNorm.
    biased_w = reg_tile("f32", shape=(TOTAL_VALUES,))
    for pair in static_range(PACKED_PAIRS):
        biased_pair = add(
            pair_view(w_f32, pair, unused_high_if_odd=True),
            scalar_pair(WEIGHT_BIAS, pair, TOTAL_VALUES,
                        undefined_high_if_odd=True), lanes=2)
        # instruction_selection: add.f32x2; extent: PACKED_PAIRS issues for
        # both WEIGHT_BIAS=0.0 and WEIGHT_BIAS=1.0
        for packed_lane in static_range(2):
            flat = pair * 2 + packed_lane
            if flat < TOTAL_VALUES:
                biased_w[flat] = biased_pair[packed_lane]

    # Phase 2f: second packed multiply, still for the full fragment.
    y_f32 = reg_tile("f32", shape=(TOTAL_VALUES,))
    for pair in static_range(PACKED_PAIRS):
        y_pair = mul(
            pair_view(normalized, pair, unused_high_if_odd=True),
            pair_view(biased_w, pair, unused_high_if_odd=True), lanes=2)
        # instruction_selection: mul.f32x2; extent: PACKED_PAIRS issues,
        # source-ordered after every bias add
        for packed_lane in static_range(2):
            flat = pair * 2 + packed_lane
            if flat < TOTAL_VALUES:
                y_f32[flat] = y_pair[packed_lane]

    # Phase 2g: narrow the complete valid fragment before any global store.
    y_frag = reg_tile(DTYPE, shape=(VEC,VEC_BLOCKS), stride=(1,VEC))
    if VEC == 1 or (VEC == 2 and VEC_BLOCKS == 3):
        for flat in static_range(TOTAL_VALUES):
            y_frag.flat[flat] = cast(DTYPE, y_f32[flat], rounding="rn")
            # instruction_selection: cvt.rn.{f16,bf16}.f32; extent:
            # TOTAL_VALUES scalar issues (seven for reviewed H111, six for
            # reviewed H66), independently of FP16 versus BF16
    else:
        for pair in static_range(PACKED_PAIRS):
            narrowed_pair = cast(
                DTYPE + "x2", pair_view(y_f32,pair,unused_high_if_odd=True),
                rounding="rn")
            # instruction_selection: cvt.rn.{f16,bf16}x2.f32; extent:
            # PACKED_PAIRS issues for reviewed VEC 4/8 branches
            for packed_lane in static_range(2):
                flat = pair * 2 + packed_lane
                if flat < TOTAL_VALUES:
                    y_frag.flat[flat] = narrowed_pair[packed_lane]

    # Phase 2h: only the final copy is row/column predicated.
    for vb in static_range(VEC_BLOCKS):
        local_col = (thread_in_row + vb * TPR) * VEC
        absolute_col = block_y * COLS_PER_TILE + local_col
        col_valid = absolute_col < H
        if row_valid and col_valid:
            y_offset = (row_i32 * H + absolute_col) if COMPACT else \
                       (row_i64 * y_row_stride + absolute_col)
            copy_r2g(y_frag[:,vb], address(y_storage,y_offset), COPY_BITS)
            # instruction_selection: exact global store from the VEC table;
            # extent: one row-and-column-predicated vector per vb

    if ENABLE_PDL:
        pdl_signal()
        # instruction_selection: griddepcontrol.launch_dependents; extent: one issue per CTA
```

## Addressing, predicates, and tail behavior

- Compact coordinates and `row_i32*H + col` offsets remain Int32 exactly as
  the source compiler path; the host has proved `M*H` fits signed Int32. Only
  the row bound comparison widens `row_i32` to the runtime `M:int64` ABI.
- Strided rows use `row_i64*x_row_stride + col` and
  `row_i64*y_row_stride + col` entirely in Int64 (`mad.lo.s64` in reviewed
  PTX). The buffers are one-dimensional backing storage; no dynamic-stride
  2-D `match_buffer` is allowed.
- The column predicate is the source `predicate_k`: one predicate per vector
  block, testing the first element's absolute hidden coordinate against `H`.
  Source vector alignment makes an accepted vector wholly in bounds.
- The row predicate gates async/sync X issue and final Y store. For an in-range
  async row the column predicate becomes a zero source-size, not skipped work.
  Weight loads remain independent of row validity. Out-of-range rows still
  execute every reduction and synchronization operation.
- Grid `block_y` owns disjoint `[block_y*COLS_PER_TILE, ...]` global ranges;
  the separate flat `cta_rank` selects the source-CTA slot in DSMEM. The
  registered cluster branch closure uses exact partitions; the `H` predicate
  remains present and is not replaced by a shape whitelist.

## Storage ownership and lifetimes

| storage | owner | lifetime / reuse rule |
| --- | --- | --- |
| `sX` | each thread writes and reads its own `(row,local_col)` vectors | async issue through the post-reduction reload; the final CTA/cluster barrier is the reuse edge |
| cluster-1 reduction | lane 0 of each row warp publishes one scalar | publication barrier through the redundant final reduction in every row warp |
| clustered reduction | every source warp sends its partial to every peer; slot records source CTA rank | remote completion through local mbarrier wait and final reduction |
| cluster mbarrier | thread 0 initializes; one elected lane of warp 0 arrives with the byte expectation; all remote stores complete it | cluster initialization through the one reduction transaction epoch |
| X fragment | one thread | synchronous path: load through pass 2; async path: first load through square/reduction then discarded and reloaded after barrier |
| W fragment | one thread | global load before the async X wait (or at the same source-ordered point in the sync path) through pass 2 |
| output fragment | one thread | first FP32 multiply, bias add, second FP32 multiply, narrowing, then one predicated vector store |

No race depends on scheduling. Every row subgroup owns one row, every vector
has one thread owner, the cluster transaction byte count includes exactly all
warp-to-peer scalar stores, all participants execute the wait, and output
columns are disjoint across threads and cluster CTAs.

## Module and verification contract

- Registry name is `flashinfer_rmsnorm`, category `flashinfer`, compute
  capability 10; `get_kernel` is the only registered factory.
- `CONFIGS` contains exactly 279 unique executable configurations and
  `BENCH_CONFIGS` contains exactly the five frozen BF16 performance workloads.
- Each upstream matrix config executes both implicit FlashInfer output and the
  caller-provided `out` API without duplicating the compile label.
- Correctness compares the public FlashInfer CuTe-DSL path and an independent
  FP32 oracle at `rtol=atol=1e-3`, asserts output object identity, immutable
  inputs/weight, finite output, and sampled Int64 boundary rows for huge cases.
- Static verification rejects `TilePrimitiveCall`, `tirx.tile.*`, imports of
  `tvm.script.tirx.tile`, and tile primitives in every generated PrimFunc.
- The only performance authority is a complete five-workload
  `python -m tirx_kernels.bench_suite` run. Each exact
  `flashinfer_cutedsl_time / tirx_time` ratio must be strictly greater than
  0.99; diagnostic PTX/SASS/NCU results never replace that gate.

## Instruction selection is a lowering consequence

The port first preserves the source thread-value layout, vector width, copy
path, ordered fragment reduction, shuffle topology, shared offsets, cluster
transaction protocol, register lifetime, and barriers. Plain TIRx and local
typed PTX may then express the reviewed opcodes. It may not use a tile
primitive, change row ownership, substitute a different block/cluster
reduction, scalarize reviewed vector traffic, keep async X live instead of
reloading it, fuse or reassociate the two scale multiplies, replace
`rsqrt.approx.ftz.f32`, or introduce PDL operations into a disabled
specialization.
