<!--
Copyright (c) 2025 by FlashInfer team.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: Apache-2.0

This execution sketch documents a TIRx port of FlashInfer's CuTe-DSL
FusedAddRMSNormKernel. See LICENSE, NOTICE, and licenses/ for applicable terms.
-->

# FlashInfer fused add + RMSNorm / Gemma RMSNorm SM100: execution sketch

This file is a non-executable execution sketch. It freezes the implementation
shape of FlashInfer's CuTe-DSL `FusedAddRMSNormKernel` for the paired target
[`tirx_kernels/flashinfer/norm/fused_add_rmsnorm.py`](../../../../tirx_kernels/flashinfer/norm/fused_add_rmsnorm.py).
The target may become executable only after this sketch passes independent
source/PTX review; after the first PASS this file is permanently frozen.

The source is fixed at FlashInfer commit
`f2e04400e330fb2debe0bf8730d9424a1d37927f`:

- `flashinfer/norm/kernels/fused_add_rmsnorm.py`, SHA256
  `f6f4a7d9c88996f26c33ed2661f82026e4ee3e747bf12c57febe9f9e68e61435`;
- inherited launch/layout helpers in `flashinfer/norm/kernels/rmsnorm.py`,
  SHA256 `b273fe5444aaf86a1600c196817b7a733b18f6f82030a16e1ef2731c784d48f0`;
- reduction and predicate helpers in `flashinfer/norm/utils.py`, SHA256
  `3f44ac6727c58883420068bf0aa5b239b12d2e86819ad80e54bc1bc016ec881a`;
- public dispatch in `flashinfer/norm/__init__.py`, SHA256
  `226a88f5fb14e78e06e1be79020edcae01bfa9e53e677bd485373ea4d51cffcb`.

The family covers FP16/BF16, ordinary fused-add RMSNorm with
`weight_bias=0.0` and Gemma fused-add RMSNorm with `weight_bias=1.0`, compact
and explicit independent Int64 row strides, PDL off/on, synchronous and paired
`cp.async` input/residual traffic, sub-warp, one-warp, multi-warp, and cluster
reduction, and cluster sizes 1/2/4/8/16. Only the two-dimensional in-place
fused-add body is in scope. Quantization, plain RMSNorm, legacy CUDA, and every
tile primitive are out of scope.

## Source dispatch and resource formulae

These compile-time formulae are the source policy, not tuning choices:

```python
ELEM_BYTES = 2
OPTIN_SMEM_BYTES = 232448
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

def derived(H, cluster_n):
    H_PER_CTA = H // cluster_n
    TPR = threads_per_row(H_PER_CTA)
    THREADS = num_threads(H_PER_CTA)
    ROWS = THREADS // TPR
    WARPS_PER_ROW = max(TPR // 32, 1)
    VEC = min(H_PER_CTA & -H_PER_CTA, MAX_VEC)
    COPY_BITS = 16 * VEC
    VEC_BLOCKS = max(1, ceil_div(H_PER_CTA // VEC, TPR))
    COLS = VEC * VEC_BLOCKS * TPR
    TWO_TILE_BYTES = 2 * ROWS * COLS * ELEM_BYTES
    USE_ASYNC = COPY_BITS >= 32 and TWO_TILE_BYTES <= OPTIN_SMEM_BYTES // 2
    REDUCE_BYTES = ROWS * WARPS_PER_ROW * cluster_n * 4
    SMEM_BYTES = (TWO_TILE_BYTES if USE_ASYNC else 0) + REDUCE_BYTES
    if cluster_n > 1: SMEM_BYTES += 8
    return (...)

def estimate_smem(H, cluster_n):
    # The estimate always includes both X and residual tiles, even if the
    # eventual body uses synchronous register loads.
    d = derived_without_async_elision(H, cluster_n)
    return 2 * d.ROWS * d.COLS * ELEM_BYTES \
           + d.ROWS * d.WARPS_PER_ROW * cluster_n * 4 \
           + (8 if cluster_n > 1 else 0)

def source_config(H):
    best_fit = 1
    for candidate in (1, 2, 4, 8, 16):
        if H % candidate != 0:
            continue
        required = estimate_smem(H, candidate)
        if required <= OPTIN_SMEM_BYTES // 2:
            return derived(H, candidate)
        if required <= OPTIN_SMEM_BYTES and best_fit == 1:
            best_fit = candidate
    return derived(H, best_fit)
```

The host chooses the compact ABI only when both mutable tensors are contiguous
and `M*H <= 2**31-1`. Otherwise it passes independent
`x_row_stride:int64` and `residual_row_stride:int64`, both in elements. The
weight is always compact. Compact X/R fake tensors carry
`assumed_align=gcd(128,H*ELEM_BYTES)` bytes. Strided X/R bases each carry
`assumed_align=16` bytes and their independent symbolic row strides each carry
`divisibility=VEC`; weight carries `assumed_align=16` bytes. This predicate,
alignment, stride-divisibility, and stride independence are fixed.

## Pipeline at a glance

| lanes / CTA role | source-owned work | publication or reuse edge |
| --- | --- | --- |
| each contiguous group of `TPR` threads | one logical row and one cluster H partition; each thread owns `VEC` adjacent columns per vector block | X, residual, and weight fragments remain thread-private |
| every thread before reduction | widen X/R, pairwise add to FP32 `h`, narrow and publish `h` to residual global memory, then square `h` | residual update is source-ordered before reduction and remains independent of the later input store |
| each sub-warp/full warp within a row | ordered local sum followed by butterfly stages at widths 8/16/32 | subgroup-replicated partial |
| lane 0 of every row warp, cluster-1 multi-warp branch | publish one row-warp partial to shared | one CTA barrier before redundant final loads |
| lanes `<CLUSTER_N` of every warp, clustered branch | send that warp's partial to the corresponding slot on every peer CTA | remote async shared store completes the peer mbarrier transaction |
| every row warp after publication | load and full-warp reduce `WARPS_PER_ROW*CLUSTER_N` partials | replicated row sum feeds mean and approximate rsqrt |
| all threads after reduction | cluster arrive/wait or CTA barrier, use still-live FP32 `h` and weight, narrow and write input | final barrier is the reduction completion edge, not an X reload edge |

There is one paired `cp.async` group, not a staged pipeline. Each thread issues
all X copies then all residual copies, commits once, loads weight synchronously,
waits once, and loads both shared fragments. PDL wait is the first body-side
dependency instruction and PDL signal is the last. Disabled specializations
contain neither instruction nor launch attribute.

## Primitive vocabulary

Structural operations do not move or compute values:

```python
specialize(...)       # compile-time family member
launch(...)           # exact grid, block, cluster, dynamic shared, PDL attrs
raw_shared(...)       # one dynamic shared allocation
view(...)             # typed view at an explicit byte offset
reg_tile(...)         # thread-private register fragment
address(...)          # compact or explicit Int64 backing-store address
pair_view(...)        # adjacent flattened f32 register values
scalar_pair(...)      # duplicated scalar, with unused high lane for an odd tail
```

Data movement, arithmetic, and synchronization remain primitive:

```python
copy_g2r(src, dst, bits, predicate=None)
copy_g2s_async_ca(src, dst, bits, source_bytes)
copy_s2r(src, dst, bits)
copy_r2g(src, dst, bits, predicate=None)
cast(dtype, src, rounding=None)
add(lhs, rhs, lanes=1)
mul(lhs, rhs, lanes=1)
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

One vector copy is one reviewed instruction. The source exports select:

| `VEC` | global X/R/W load and X/R store | shared X/R load | reviewed narrowing |
| ---: | --- | --- | --- |
| 1 | scalar `ld/st.global.b16` | synchronous reviewed profile | scalar `cvt.rn.{f16,bf16}.f32` |
| 2 | `ld/st.global.v2.b16` | `ld.shared.v2.b16` | scalar converts for reviewed `(VEC=2,VEC_BLOCKS=3)` H66 profile |
| 4 | `ld/st.global.v2.b32` | `ld.shared.v4.b16` | `cvt.rn.{f16,bf16}x2.f32` |
| 8 | `ld/st.global.v4.b32` | `ld.shared.v4.b32` | `cvt.rn.{f16,bf16}x2.f32` |

Every async copy is `cp.async.ca.shared.global`, with immediate size
`COPY_BITS/8` and a runtime source-size operand for column-tail zero fill.
`add(...,lanes=2)` and `mul(...,lanes=2)` each denote one native packed-FP32
instruction. There is no compound fused-add, RMSNorm, row/CTA/cluster
reduction, tile-copy, or tile-compute operation.

## Complete sketch

```python
@specialize(
    VARIANT_WEIGHT_BIAS=(("fused_add_rmsnorm", 0.0),
                         ("gemma_fused_add_rmsnorm", 1.0)),
    DTYPE=("f16", "bf16"),
    H="positive compile-time integer",
    LAYOUT=("compact", "strided_i64"),
    ENABLE_PDL=(False, True),
    TARGET="sm_100a",
)
@launch(
    grid=(ceil_div(runtime_M, ROWS), CLUSTER_N, 1),
    block=(THREADS, 1, 1),
    cluster=((1, CLUSTER_N, 1) if CLUSTER_N > 1 else None),
    dynamic_smem_bytes=SMEM_BYTES,
    use_programmatic_dependent_launch=ENABLE_PDL,
)
def flashinfer_fused_add_rmsnorm(
    input_storage,                    # mutable DTYPE backing storage; compact
                                      # align=gcd(128,H*2), strided align=16
    residual_storage,                 # same base-alignment contract as input
    weight,                           # compact DTYPE [H], align=16
    runtime_M: i64,
    runtime_eps: f32,
    x_row_stride: i64 = H,            # strided_i64 only; divisible by VEC
    residual_row_stride: i64 = H,     # strided_i64 only; independently
                                      # divisible by VEC
):
    tid = thread_id_x(THREADS)
    block_x = block_id_x()
    block_y = 0
    cta_rank = 0
    if CLUSTER_N > 1:
        block_y = block_id_y()
        cta_rank = cta_rank_in_cluster()

    if ENABLE_PDL:
        pdl_wait()
        # instruction_selection: griddepcontrol.wait; extent: one issue per CTA

    # CuTe TV layout shape=((TPR,ROWS),(VEC,VEC_BLOCKS)) and
    # stride=((VEC*ROWS,1),(ROWS,ROWS*VEC*TPR)).
    row_in_cta = tid // TPR
    thread_in_row = tid % TPR
    row_i32 = block_x * ROWS + row_in_cta
    row_i64 = cast_i64(row_i32)
    row_valid = row_i64 < runtime_M
    warp = tid // 32
    lane = tid % 32
    row_warp = warp // WARPS_PER_ROW
    warp_in_row = warp % WARPS_PER_ROW

    smem = raw_shared("u8", SMEM_BYTES, alignment=16)
    cursor = 0
    if USE_ASYNC:
        sX = view(smem, DTYPE, (ROWS,COLS), byte_offset=cursor,
                  row_major=True, alignment=16)
        cursor += ROWS * COLS * ELEM_BYTES
        sR = view(smem, DTYPE, (ROWS,COLS), byte_offset=cursor,
                  row_major=True, alignment=16)
        cursor += ROWS * COLS * ELEM_BYTES
    if CLUSTER_N == 1:
        reduction = view(smem, "f32", (ROWS,WARPS_PER_ROW),
                         stride=(1,ROWS), byte_offset=cursor, alignment=4)
        cursor += ROWS * WARPS_PER_ROW * 4
    else:
        reduction = view(smem, "f32", (ROWS,(WARPS_PER_ROW,CLUSTER_N)),
                         stride=(1,(ROWS,ROWS*WARPS_PER_ROW)),
                         byte_offset=cursor, alignment=4)
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
        # instruction_selection: barrier.cluster.wait; extent: one per CTA

    x_frag = reg_tile(DTYPE, (VEC,VEC_BLOCKS), stride=(1,VEC))
    r_frag = reg_tile(DTYPE, (VEC,VEC_BLOCKS), stride=(1,VEC))
    w_frag = reg_tile(DTYPE, (VEC,VEC_BLOCKS), stride=(1,VEC))
    TOTAL_VALUES = VEC * VEC_BLOCKS
    PAIRS = ceil_div(TOTAL_VALUES, 2)

    if not USE_ASYNC:
        for flat in static_range(TOTAL_VALUES):
            x_frag.flat[flat] = zero(DTYPE)
            r_frag.flat[flat] = zero(DTYPE)
            # instruction_selection: mov.b16 zero initialization; extent: two
            # complete fragments before any predicated synchronous load

    # Input copies precede residual copies in the one source async group.
    for vb in static_range(VEC_BLOCKS):
        local_col = (thread_in_row + vb * TPR) * VEC
        absolute_col = block_y * COLS + local_col
        col_valid = absolute_col < H
        x_offset = (row_i32 * H + absolute_col) if COMPACT else \
                   (row_i64 * x_row_stride + absolute_col)
        if USE_ASYNC:
            if row_valid:
                copy_g2s_async_ca(address(input_storage,x_offset),
                                  sX[row_in_cta,local_col:local_col+VEC],
                                  COPY_BITS, (COPY_BITS//8) if col_valid else 0)
                # instruction_selection: cp.async.ca.shared.global; extent:
                # one row-gated issue per vb with runtime tail source size
        else:
            if row_valid and col_valid:
                copy_g2r(address(input_storage,x_offset), x_frag[:,vb], COPY_BITS)
                # instruction_selection: exact global load from the VEC table;
                # extent: one row-and-column-predicated vector per vb

    for vb in static_range(VEC_BLOCKS):
        local_col = (thread_in_row + vb * TPR) * VEC
        absolute_col = block_y * COLS + local_col
        col_valid = absolute_col < H
        r_offset = (row_i32 * H + absolute_col) if COMPACT else \
                   (row_i64 * residual_row_stride + absolute_col)
        if USE_ASYNC:
            if row_valid:
                copy_g2s_async_ca(address(residual_storage,r_offset),
                                  sR[row_in_cta,local_col:local_col+VEC],
                                  COPY_BITS, (COPY_BITS//8) if col_valid else 0)
                # instruction_selection: cp.async.ca.shared.global; extent:
                # one row-gated issue per vb after every input issue
        else:
            if row_valid and col_valid:
                copy_g2r(address(residual_storage,r_offset), r_frag[:,vb], COPY_BITS)
                # instruction_selection: exact global load from the VEC table;
                # extent: one row-and-column-predicated vector per vb

    if USE_ASYNC:
        cp_async_commit()
        # instruction_selection: cp.async.commit_group; extent: one per CTA thread

    for vb in static_range(VEC_BLOCKS):
        local_col = (thread_in_row + vb * TPR) * VEC
        absolute_col = block_y * COLS + local_col
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
        for vb in static_range(VEC_BLOCKS):
            local_col = (thread_in_row + vb * TPR) * VEC
            copy_s2r(sR[row_in_cta,local_col:local_col+VEC], r_frag[:,vb], COPY_BITS)
            # instruction_selection: exact shared load from the VEC table;
            # extent: one vector per vb after every input shared load

    x_f32 = reg_tile("f32", (TOTAL_VALUES,))
    r_f32 = reg_tile("f32", (TOTAL_VALUES,))
    w_f32 = reg_tile("f32", (TOTAL_VALUES,))
    for flat in static_range(TOTAL_VALUES):
        x_f32[flat] = cast("f32", x_frag.flat[flat])
        # instruction_selection: cvt.f32.{f16,bf16}; extent: one per X value
    for flat in static_range(TOTAL_VALUES):
        r_f32[flat] = cast("f32", r_frag.flat[flat])
        # instruction_selection: cvt.f32.{f16,bf16}; extent: one per residual value

    h = reg_tile("f32", (TOTAL_VALUES,))
    for pair in static_range(PAIRS):
        h_pair = add(pair_view(x_f32,pair,unused_high_if_odd=True),
                     pair_view(r_f32,pair,unused_high_if_odd=True), lanes=2)
        # instruction_selection: add.f32x2; extent: PAIRS packed issues
        h.store_pair(pair, h_pair)

    # The source narrows and publishes residual before forming h squared.
    h_narrow = reg_tile(DTYPE, (TOTAL_VALUES,))
    if VEC == 1 or (VEC == 2 and VEC_BLOCKS == 3):
        for flat in static_range(TOTAL_VALUES):
            h_narrow[flat] = cast(DTYPE, h[flat], rounding="rn")
            # instruction_selection: cvt.rn.{f16,bf16}.f32; extent: scalar
    else:
        for pair in static_range(PAIRS):
            h_narrow.store_pair(pair, cast(DTYPE+"x2", pair_view(h,pair), rounding="rn"))
            # instruction_selection: cvt.rn.{f16,bf16}x2.f32; extent: one per pair
    for vb in static_range(VEC_BLOCKS):
        local_col = (thread_in_row + vb * TPR) * VEC
        absolute_col = block_y * COLS + local_col
        col_valid = absolute_col < H
        r_offset = (row_i32 * H + absolute_col) if COMPACT else \
                   (row_i64 * residual_row_stride + absolute_col)
        if row_valid and col_valid:
            copy_r2g(h_narrow.slice(vb,VEC), address(residual_storage,r_offset), COPY_BITS)
            # instruction_selection: exact global store from the VEC table;
            # extent: one predicated residual vector per vb

    h_sq = reg_tile("f32", (TOTAL_VALUES,))
    for pair in static_range(PAIRS):
        h_sq.store_pair(pair, mul(pair_view(h,pair), pair_view(h,pair), lanes=2))
        # instruction_selection: mul.f32x2; extent: PAIRS packed issues
    local_sum = 0.0
    for flat in static_range(TOTAL_VALUES):
        local_sum = add(local_sum, h_sq[flat])
        # instruction_selection: add.f32; extent: TOTAL_VALUES ordered scalar
        # issues per thread, including zero-filled column-tail fragment entries

    WARP_WIDTH = min(TPR, 32)
    for delta in powers_of_two_below(WARP_WIDTH):
        peer = shuffle_xor(local_sum, delta, width=32)
        # instruction_selection: shfl.sync.bfly.b32; extent: one scalar per stage
        local_sum = add(local_sum, peer)
        # instruction_selection: add.f32; extent: one scalar per stage
    warp_sum = local_sum

    if WARPS_PER_ROW > 1 and CLUSTER_N == 1:
        if lane == 0:
            reduction[row_warp,warp_in_row] = warp_sum
            # instruction_selection: st.shared.b32; extent: one per row warp
        cta_barrier()
        # instruction_selection: bar.sync 0; extent: one CTA publication edge
        final = 0.0
        if lane < WARPS_PER_ROW:
            final = reduction[row_warp,lane]
            # instruction_selection: ld.shared.b32; extent: one scalar per participating lane
        for delta in (1,2,4,8,16):
            peer = shuffle_xor(final, delta, width=32)
            # instruction_selection: shfl.sync.bfly.b32; extent: one scalar per stage
            final = add(final, peer)
            # instruction_selection: add.f32; extent: one scalar per stage
        sum_sq = final

    elif CLUSTER_N > 1:
        if warp == 0 and elect_one():
            # instruction_selection: elect.sync; extent: one election in warp 0
            mbarrier_arrive_expect_tx(
                mbar, ROWS * WARPS_PER_ROW * CLUSTER_N * 4)
            # instruction_selection: mbarrier.arrive.expect_tx.shared.b64;
            # extent: one elected issue per CTA
        if lane < CLUSTER_N:
            peer_reduce = map_shared_to_peer(
                reduction[row_warp,(warp_in_row,cta_rank)], lane)
            peer_mbar = map_shared_to_peer(mbar, lane)
            # instruction_selection: mapa.shared::cluster.u32; extent: two mappings
            remote_store_complete_tx(warp_sum, peer_reduce, peer_mbar)
            # instruction_selection:
            # st.async.shared::cluster.mbarrier::complete_tx::bytes.f32;
            # extent: one partial to each peer selected by an active lane
        done = False
        while not done:
            done = mbarrier_try_wait_parity(mbar, phase=0, timeout=10000000)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 plus
            # bra.uni retry; extent: one uniform wait loop per CTA thread
        total_partials = WARPS_PER_ROW * CLUSTER_N
        final = 0.0
        for i in static_range(ceil_div(total_partials,32)):
            partial = lane + i * 32
            if partial < total_partials:
                value = reduction[row_warp,partial]
                # instruction_selection: ld.shared.b32; extent: one owned partial
                final = add(final, value)
                # instruction_selection: add.f32; extent: one per loaded partial
        for delta in (1,2,4,8,16):
            peer = shuffle_xor(final, delta, width=32)
            # instruction_selection: shfl.sync.bfly.b32; extent: one scalar per stage
            final = add(final, peer)
            # instruction_selection: add.f32; extent: one scalar per stage
        sum_sq = final
    else:
        sum_sq = warp_sum

    if is_power_of_two(H):
        shifted = fma(sum_sq, exact_float32_reciprocal(H), runtime_eps)
        # instruction_selection: fma.rn.f32; extent: one scalar per thread
    else:
        mean_sq = div(sum_sq, float32(H))
        # instruction_selection: div.rn.f32; extent: one scalar per thread
        shifted = add(mean_sq, runtime_eps)
        # instruction_selection: add.f32; extent: one scalar per thread
    rstd = rsqrt_fast(shifted)
    # instruction_selection: rsqrt.approx.ftz.f32; extent: one scalar per thread

    if CLUSTER_N > 1:
        cluster_arrive_relaxed()
        # instruction_selection: barrier.cluster.arrive.relaxed; extent: one per CTA
        cluster_wait()
        # instruction_selection: barrier.cluster.wait; extent: one per CTA
    else:
        cta_barrier()
        # instruction_selection: bar.sync 0; extent: one post-reduction CTA edge

    for flat in static_range(TOTAL_VALUES):
        w_f32[flat] = cast("f32", w_frag.flat[flat])
        # instruction_selection: cvt.f32.{f16,bf16}; extent: one per weight value
    normalized = reg_tile("f32", (TOTAL_VALUES,))
    for pair in static_range(PAIRS):
        normalized.store_pair(
            pair, mul(pair_view(h,pair), scalar_pair(rstd,pair,TOTAL_VALUES), lanes=2))
        # instruction_selection: mul.f32x2; extent: PAIRS packed issues
    biased_w = reg_tile("f32", (TOTAL_VALUES,))
    for pair in static_range(PAIRS):
        biased_w.store_pair(
            pair, add(pair_view(w_f32,pair),
                      scalar_pair(WEIGHT_BIAS,pair,TOTAL_VALUES), lanes=2))
        # instruction_selection: add.f32x2; extent: PAIRS packed issues for both variants
    y = reg_tile("f32", (TOTAL_VALUES,))
    for pair in static_range(PAIRS):
        y.store_pair(pair, mul(pair_view(normalized,pair),
                               pair_view(biased_w,pair), lanes=2))
        # instruction_selection: mul.f32x2; extent: PAIRS packed issues

    y_narrow = reg_tile(DTYPE, (TOTAL_VALUES,))
    if VEC == 1 or (VEC == 2 and VEC_BLOCKS == 3):
        for flat in static_range(TOTAL_VALUES):
            y_narrow[flat] = cast(DTYPE, y[flat], rounding="rn")
            # instruction_selection: cvt.rn.{f16,bf16}.f32; extent: scalar
    else:
        for pair in static_range(PAIRS):
            y_narrow.store_pair(pair, cast(DTYPE+"x2", pair_view(y,pair), rounding="rn"))
            # instruction_selection: cvt.rn.{f16,bf16}x2.f32; extent: one per pair
    for vb in static_range(VEC_BLOCKS):
        local_col = (thread_in_row + vb * TPR) * VEC
        absolute_col = block_y * COLS + local_col
        col_valid = absolute_col < H
        x_offset = (row_i32 * H + absolute_col) if COMPACT else \
                   (row_i64 * x_row_stride + absolute_col)
        if row_valid and col_valid:
            copy_r2g(y_narrow.slice(vb,VEC), address(input_storage,x_offset), COPY_BITS)
            # instruction_selection: exact global store from the VEC table;
            # extent: one predicated input vector per vb

    if ENABLE_PDL:
        pdl_signal()
        # instruction_selection: griddepcontrol.launch_dependents; extent: one issue per CTA
```

## Export reconciliation and reviewed profiles

- Dynamic shared memory is one raw allocation; the generated PTX broadcasts
  its base with `shfl.sync.idx.b32`. Async H4096 places `sX` at byte 0, `sR`
  after 16384 bytes, and reduction after 32768 bytes.
- BF16 M32 H4096 emits 16 `cp.async` issues (eight X then eight residual), one
  commit, eight weight `ld.global.v4.b32`, one wait, 16
  `ld.shared.v4.b32`, eight residual and eight input `st.global.v4.b32`.
- BF16 H66 emits six 4-byte async issues, three weight loads, six shared loads,
  and three vector stores to each mutable tensor. FP16 H111 selects the fully
  synchronous scalar branch: seven loads from each of X/R/W and seven stores
  to each mutable tensor.
- FP16 strided M19 H500 emits `mad.lo.s64` addressing for independent X and
  residual rows; its four X plus four residual async issues have tail source
  sizes and its stores remain separately predicated.
- Cluster H32768/H131072/H262144 and sync H524288 exports contain the explicit
  mbarrier init/fence, cluster arrive/wait pairs, two peer address mappings,
  one remote complete-tx store site, uniform parity wait, and ten butterfly
  stages. H1048576 follows the source fallback to cluster 1 and synchronous
  loads. PDL-enabled profiles contain exactly one wait and one signal.

## Addressing, predicates, mutation, and tails

- Compact row offsets are Int32 after the host proves `M*H` fits signed Int32;
  only comparison with `runtime_M:i64` widens the row. Strided row products and
  pointer additions remain Int64 independently for input and residual.
- One source `predicate_k` predicate covers each vector block by testing its
  first absolute hidden coordinate. Alignment and `VEC` selection make an
  accepted vector wholly in range.
- Row validity gates X/R traffic and both mutable stores. In an async valid row,
  an invalid column becomes a zero source size rather than a skipped async
  instruction. Weight traffic depends only on the column predicate.
- `h` is computed in FP32. Residual observes `cast_DTYPE(h)` before the
  reduction; input observes `cast_DTYPE(h*rstd*(weight+WEIGHT_BIAS))` after the
  reduction. The normalization uses FP32 `h`, not the narrowed residual value.
- Out-of-range row groups still execute weight loads, reduction, mbarrier, and
  CTA/cluster synchronization. Cluster CTAs own disjoint H ranges and publish
  partials to every peer before any input result is stored.

## Storage ownership and lifetimes

| storage | owner | lifetime / reuse rule |
| --- | --- | --- |
| `sX`, `sR` | each thread writes and reads its own `(row,local_col)` vectors | one async issue/commit/wait/load epoch; dead after FP32 `h` is formed |
| cluster-1 reduction | lane 0 of each row warp publishes one scalar | CTA publication barrier through final shared load |
| clustered reduction | every source warp sends a partial to every peer; slot includes source CTA rank | complete-tx remote store through local mbarrier wait and final load |
| cluster mbarrier | thread 0 initializes; one elected lane expects all peer-store bytes | initialization barrier through the single reduction transaction |
| X/R fragments | one thread | load through FP32 add; source fragments die after `h` is formed |
| FP32 `h` | one thread | add through early residual store, square/reduction, barrier, and final input multiply |
| weight fragment | one thread | global load before async wait through post-reduction bias/multiply |
| residual output fragment | one thread | `h` narrowing through the early predicated residual store |
| input output fragment | one thread | post-reduction multiplies and narrowing through final predicated input store |

No race depends on scheduling: each vector has one owner, residual publication
does not feed the reduction, every source partial completes exactly one byte
counted cluster transaction per peer, all participants execute the wait, and
cluster CTAs write disjoint hidden-column ranges.

## Module and verification contract

- Registry name is `flashinfer_fused_add_rmsnorm`, category `flashinfer`, and
  compute capability 10; `get_kernel` is the only registered factory.
- `CONFIGS` contains exactly 281 unique configurations and `BENCH_CONFIGS`
  contains the six frozen BF16 source workloads, including PDL and both public
  variants.
- Correctness checks both in-place outputs against the direct public
  FlashInfer CuTe-DSL path and an independent FP32 oracle at
  `rtol=atol=1e-3`, checks finite values, mutation/object behavior, independent
  row padding and guard regions, and sampled Int64 overflow boundary rows.
- Static verification rejects `TilePrimitiveCall`, `tirx.tile.*`, imports of
  `tvm.script.tirx.tile`, and tile primitives in every generated PrimFunc.
- The only final performance authority is `python -m
  tirx_kernels.bench_suite`; every exact `flashinfer_cutedsl_time / tirx_time`
  ratio in the complete six-shape matrix must be strictly greater than 0.99.
  PTX, SASS, NCU, and any other timing API are diagnostic only.

## Instruction selection is a lowering consequence

The port preserves source row ownership, TV fragment order, vector width,
paired copy issue order, early residual publication, ordered FP32 reduction,
shared offsets, cluster transaction protocol, live `h` lifetime, and barriers
before relying on local typed PTX. It may not use a tile primitive, substitute a
different reduction, scalarize reviewed vector traffic, reload narrowed
residual for normalization, move residual publication after reduction, merge or
reassociate the two post-reduction multiplies, replace
`rsqrt.approx.ftz.f32`, couple the two row strides, or emit PDL operations in a
disabled specialization.
