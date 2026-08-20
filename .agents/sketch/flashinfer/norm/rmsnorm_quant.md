<!--
Copyright (c) 2025 by FlashInfer team.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: Apache-2.0

This execution sketch documents a TIRx port of FlashInfer's CuTe-DSL
RMSNormQuantKernel. See LICENSE, NOTICE, and licenses/ for applicable terms.
-->

# FlashInfer 2-D RMSNorm FP8 quantization SM100: execution sketch

This file is a non-executable execution sketch. It freezes the implementation
shape of FlashInfer's CuTe-DSL `RMSNormQuantKernel` for the paired target
[`tirx_kernels/flashinfer/norm/rmsnorm_quant.py`](../../../../tirx_kernels/flashinfer/norm/rmsnorm_quant.py).
The target becomes executable only after this sketch passes independent
source/line-info-PTX review; after the first PASS this file is permanently
frozen.

The source is fixed at FlashInfer commit
`f2e04400e330fb2debe0bf8730d9424a1d37927f`:

- `flashinfer/norm/kernels/rmsnorm.py`, SHA256
  `b273fe5444aaf86a1600c196817b7a733b18f6f82030a16e1ef2731c784d48f0`;
- `flashinfer/norm/utils.py`, SHA256
  `3f44ac6727c58883420068bf0aa5b239b12d2e86819ad80e54bc1bc016ec881a`;
- public dispatch in `flashinfer/norm/__init__.py`, SHA256
  `226a88f5fb14e78e06e1be79020edcae01bfa9e53e677bd485373ea4d51cffcb`.

The family covers FP16/BF16 X and W, E4M3FN/E5M2 caller-provided Y,
`weight_bias=0.0`, compact and explicit Int64 row-strided addressing, PDL
off/on, synchronous and one-group `cp.async` input traffic, sub-warp through
cluster reduction, and cluster sizes 1/2/4/8/16. SM100 always uses hardware
FP8 conversion. Gemma, fused-add, FP4, legacy CUDA, the pre-SM89 software-FP8
path, and every tile primitive are out of scope.

## Source dispatch and resource formulae

These compile-time formulae are copied mechanically from
`RMSNormQuantKernel` and the inherited `RMSNormKernel` helpers. They are not
tuning choices:

```python
ELEM_BYTES = 2
MAX_VEC = 128 // 8 // ELEM_BYTES                         # 8 input elements

def threads_per_row(h_per_cta):
    if h_per_cta <= 64: return 8
    if h_per_cta <= 128: return 16
    if h_per_cta <= 3072: return 32
    if h_per_cta <= 6144: return 64
    if h_per_cta <= 16384: return 128
    return 256

def rmsnorm_num_threads(h_per_cta):
    return 128 if h_per_cta <= 16384 else 256

def estimate_smem(H, cluster_n):
    # Cluster selection deliberately uses RMSNormKernel's estimate, before the
    # quant-only 256-thread override. It always includes the complete X tile.
    h_per_cta = H // cluster_n
    tpr = threads_per_row(h_per_cta)
    threads = rmsnorm_num_threads(h_per_cta)
    rows = threads // tpr
    warps_per_row = max(tpr // 32, 1)
    vec = min(h_per_cta & -h_per_cta, MAX_VEC)
    vec_blocks = max(1, ceil_div(h_per_cta // vec, tpr))
    cols = vec * vec_blocks * tpr
    bytes_ = rows * cols * ELEM_BYTES
    bytes_ += rows * warps_per_row * cluster_n * 4
    return bytes_ + (8 if cluster_n > 1 else 0)

def source_config(H):
    for candidate in (1, 2, 4, 8, 16):
        if H % candidate == 0 and estimate_smem(H, candidate) <= 232448:
            cluster_n = candidate
            break
    else:
        cluster_n = 16

    h_per_cta = H // cluster_n
    tpr = threads_per_row(h_per_cta)
    threads = rmsnorm_num_threads(h_per_cta)
    if h_per_cta > 8192 and threads < 256:
        threads = 256                                # quant-only override
    rows = threads // tpr
    warps_per_row = max(tpr // 32, 1)
    vec = min(h_per_cta & -h_per_cta, MAX_VEC)
    copy_bits = 16 * vec
    vec_blocks = max(1, ceil_div(h_per_cta // vec, tpr))
    cols = vec * vec_blocks * tpr
    tile_bytes = rows * cols * ELEM_BYTES
    use_async = copy_bits >= 32 and tile_bytes <= 232448 // 2
    reduce_bytes = rows * warps_per_row * 4
    if cluster_n > 1:
        reduce_bytes *= cluster_n
    smem_bytes = (tile_bytes if use_async else 0) + reduce_bytes
    if cluster_n > 1:
        smem_bytes += 8
    return cluster_n, h_per_cta, tpr, threads, rows, warps_per_row, \
           vec, copy_bits, vec_blocks, cols, use_async, smem_bytes
```

The public host selects compact only when X and Y are both contiguous and
`M*H <= 2**31-1`. Otherwise it compiles and passes symbolic
`x_row_stride:int64` and `y_row_stride:int64`, in elements and divisible by
the input `VEC`. The runtime scale remains a device FP32 tensor of shape one.

## Pipeline at a glance

| lanes / CTA role | source-owned work | publication or reuse edge |
| --- | --- | --- |
| every CTA thread at entry | optional PDL wait, then load the one device scale and form its FTZ approximate reciprocal | replicated `inv_scale` remains live through the FP8 epilogue |
| each contiguous `TPR`-thread group | one logical row; each thread owns `VEC` adjacent H columns in each vector block | X and W load without communication |
| each row sub-warp/full warp | ordered fragment sum then width-8/16/32 butterfly reduction | row-warp partial is replicated |
| lane 0 of each row warp, non-cluster multi-warp path | publish one partial to shared `(row,warp_in_row)` | CTA barrier before final shared loads |
| lanes `<CLUSTER_N` of each warp, clustered path | remotely publish the warp partial to the matching slot on every peer CTA | two `mapa`, remote async store, and mbarrier completion |
| every row warp after publication | redundantly load and reduce all row partials | replicated sum feeds FP32 divide/add/rsqrt |
| all threads after reduction | cluster arrive/wait or CTA barrier, then reload staged X on async profiles | X reload intentionally shortens FP32 register lifetime |
| each row thread in the epilogue | apply X*rstd, W+bias, their product, and inverse-scale product | each complete FP32 fragment feeds one packed-or-scalar FP8 store loop |

There is exactly one `cp.async` group, not a multi-stage pipeline. W loads are
issued between X async commit and wait. PDL wait follows CTA/thread-ID and any
compiler-hoisted pure-local initialization, but precedes the scale load and all
dependency-sensitive global traffic; PDL signal follows every output store.
PDL-disabled specializations contain neither PDL instruction nor launch
attribute.

## Primitive vocabulary

Structural operations do not move or compute values:

```python
specialize(...)
launch(...)
raw_shared(...)
view(...)
reg_tile(...)
address(...)
pair_view(...)
scalar_pair(...)
```

Every data, arithmetic, conversion, and synchronization operation remains
primitive:

```python
copy_g2r(src, dst, bits, predicate=None)
copy_g2s_async_ca(src, dst, bits, source_bytes)
copy_s2r(src, dst, bits)
copy_r2s(src, dst)
copy_r2g(src, dst, bits, predicate=None)
fill(dst, value)
cast(dtype, src)
mul(lhs, rhs, lanes=1)
add(lhs, rhs, lanes=1)
fma(lhs, rhs, acc)
fma_half_inputs_to_f32(lhs, rhs, acc)
div(lhs, rhs)
maximum(lhs, rhs)
minimum(lhs, rhs)
rcp_fast(src)
rsqrt_fast(src)
shuffle_xor(src, delta, width)
convert_fp8_pair(lo, hi, dtype)
pack_two_b16(lo, hi)
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

Input traffic uses the source 128-bit-copy layout:

| `VEC` | global X/W load | shared X load |
| ---: | --- | --- |
| 1 | `ld.global.b16` | async disabled |
| 2 | `ld.global.v2.b16` | `ld.shared.v2.b16` |
| 4 | `ld.global.v2.b32` | `ld.shared.v4.b16` |
| 8 | `ld.global.v4.b32` | `ld.shared.v4.b32` |

Every async X issue is `cp.async.ca.shared.global`, with immediate copy size
`COPY_BITS/8` and the same runtime source-size operand for tail zero-fill.
`mul(...,lanes=2)` and `add(...,lanes=2)` each denote one native packed-FP32
instruction. There is no compound RMSNorm, row/block/cluster reduction,
quantization, tile-copy, or tile-compute primitive.

The hardware FP8 store families are fixed by the source inline PTX:

| path | conversions | packing | store |
| --- | --- | --- | --- |
| full vec8 | four `cvt.rn.satfinite.{e4m3,e5m2}x2.f32` | two `mov.b32` from adjacent b16 pairs | one `st.global.v2.b32` |
| full vec4 | two pair converts | one `mov.b32` | one `st.global.b32` |
| full vec2 | one pair convert | none | one `st.global.b16` |
| vec1 or any tail | explicit scalar max/min clamp, one pair convert with zero in the unused high byte | none | one `st.global.b8` per valid value |

## Complete sketch

```python
@specialize(
    INPUT_DTYPE=("f16", "bf16"),
    OUTPUT_DTYPE=("e4m3fn", "e5m2"),
    H="positive compile-time integer",
    LAYOUT=("compact", "strided_i64"),
    ENABLE_PDL=(False, True),
    WEIGHT_BIAS=0.0,
    USE_HW_FP8=True,
    TARGET="sm_100a",
)
@launch(
    grid=(ceil_div(runtime_M, ROWS), CLUSTER_N, 1),
    block=(NUM_THREADS, 1, 1),
    cluster=((1, CLUSTER_N, 1) if CLUSTER_N > 1 else None),
    dynamic_smem_bytes=SMEM_BYTES,
    use_programmatic_dependent_launch=ENABLE_PDL,
)
def flashinfer_rmsnorm_quant(
    x_storage,                 # INPUT_DTYPE one-dimensional backing storage
    weight,                    # INPUT_DTYPE [H], compact
    out_storage,               # OUTPUT_DTYPE backing storage, caller provided
    runtime_M: i64,
    scale,                     # f32 [1], device resident
    runtime_eps: f32,
    x_row_stride: i64 = H,     # appended only for strided_i64
    y_row_stride: i64 = H,     # appended only for strided_i64
):
    tid = thread_id_x(NUM_THREADS)
    block_x = block_id_x()
    block_y = 0
    cta_rank = 0
    if CLUSTER_N > 1:
        block_y = block_id_y()
        cta_rank = cta_rank_in_cluster()

    if ENABLE_PDL:
        pdl_wait()
        # instruction_selection: griddepcontrol.wait; extent: one issue per CTA

    scale_value = copy_g2r(scale[0], bits=32)
    # instruction_selection: ld.global.b32; extent: one scalar load per thread
    inv_scale = rcp_fast(scale_value)
    # instruction_selection: rcp.approx.ftz.f32; extent: one scalar per thread

    # CuTe TV layout:
    # shape=((TPR,ROWS),(VEC,VEC_BLOCKS))
    # stride=((VEC*ROWS,1),(ROWS,ROWS*VEC*TPR)).
    row_in_cta = tid // TPR
    thread_in_row = tid % TPR
    row_i64 = cast_i64(block_x) * ROWS + cast_i64(row_in_cta)
    row_valid = row_i64 < runtime_M
    warp = tid // 32
    lane = tid % 32
    WARPS_PER_ROW = max(TPR // 32, 1)
    row_warp = warp // WARPS_PER_ROW
    warp_in_row = warp % WARPS_PER_ROW

    smem = raw_shared("u8", SMEM_BYTES, alignment=16)
    cursor = 0
    if USE_ASYNC:
        sX = view(smem, INPUT_DTYPE, (ROWS,COLS_PER_TILE),
                  byte_offset=cursor, row_major=True, alignment=16)
        cursor += ROWS * COLS_PER_TILE * 2
    if CLUSTER_N == 1:
        reduction = view(smem, "f32", (ROWS,WARPS_PER_ROW),
                         stride=(1,ROWS), byte_offset=cursor, alignment=4)
        cursor += ROWS * WARPS_PER_ROW * 4
    else:
        reduction = view(
            smem, "f32", (ROWS,(WARPS_PER_ROW,CLUSTER_N)),
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

    # Source fragment layout is (VEC,VEC_BLOCKS):(1,VEC); flat=v+VEC*vb.
    x_frag = reg_tile(INPUT_DTYPE, (VEC,VEC_BLOCKS), stride=(1,VEC))
    w_frag = reg_tile(INPUT_DTYPE, (VEC,VEC_BLOCKS), stride=(1,VEC))
    x_f32 = reg_tile("f32", (VEC,VEC_BLOCKS), stride=(1,VEC))
    w_f32 = reg_tile("f32", (VEC,VEC_BLOCKS), stride=(1,VEC))
    TOTAL_VALUES = VEC * VEC_BLOCKS
    PACKED_PAIRS = ceil_div(TOTAL_VALUES, 2)

    # ------------------------------------------------------------------
    # Pass 1: X issue, optional async commit, W issue, optional wait/reload.
    # ------------------------------------------------------------------
    if not USE_ASYNC:
        for flat in static_range(TOTAL_VALUES):
            fill(x_frag[flat], zero(INPUT_DTYPE))
            # instruction_selection: mov.b16 zero materialization; extent:
            # one logical value, subject only to reviewed backend coalescing

    for vb in static_range(VEC_BLOCKS):
        local_col = (thread_in_row + vb * TPR) * VEC
        absolute_col = block_y * COLS_PER_TILE + local_col
        col_valid = absolute_col < H
        # Compact flat-offset narrowing is legal only under the host proof
        # runtime_M*H<=INT32_MAX; the logical row itself remains i64.
        x_offset = compact_safe_offset(row_i64,H,absolute_col) if COMPACT else \
                   (row_i64 * x_row_stride + cast_i64(absolute_col))
        if USE_ASYNC:
            if row_valid:
                source_bytes = (COPY_BITS // 8) if col_valid else 0
                copy_g2s_async_ca(
                    address(x_storage,x_offset),
                    sX[row_in_cta,local_col:local_col+VEC],
                    bits=COPY_BITS,
                    source_bytes=source_bytes)
                # instruction_selection: cp.async.ca.shared.global with
                # immediate COPY_BITS/8 and runtime source-size; extent: one
                # issue per vb for each in-bounds row
        else:
            if row_valid and col_valid:
                copy_g2r(address(x_storage,x_offset), x_frag[:,vb], COPY_BITS)
                # instruction_selection: ld.global from the VEC table;
                # extent: one predicated vector per vb

    if USE_ASYNC:
        cp_async_commit()
        # instruction_selection: cp.async.commit_group; extent: one per thread

    for vb in static_range(VEC_BLOCKS):
        local_col = (thread_in_row + vb * TPR) * VEC
        absolute_col = block_y * COLS_PER_TILE + local_col
        if absolute_col < H:
            copy_g2r(weight[absolute_col:absolute_col+VEC], w_frag[:,vb], COPY_BITS)
            # instruction_selection: ld.global from the VEC table; extent:
            # one column-predicated vector per vb

    if USE_ASYNC:
        cp_async_wait(0)
        # instruction_selection: cp.async.wait_group 0; extent: one per thread
        for vb in static_range(VEC_BLOCKS):
            local_col = (thread_in_row + vb * TPR) * VEC
            copy_s2r(sX[row_in_cta,local_col:local_col+VEC], x_frag[:,vb], COPY_BITS)
            # instruction_selection: shared load from the VEC table; extent:
            # one vector per vb

    for flat in static_range(TOTAL_VALUES):
        x_f32[flat] = cast("f32", x_frag[flat])
        # instruction_selection: cvt.f32.{f16,bf16}; extent: one scalar per
        # physical fragment value in flattened source order

    if TOTAL_VALUES == 1:
        local_sum = fma_half_inputs_to_f32(x_frag[0],x_frag[0],0.0)
        # instruction_selection: fma.rn.f32.{f16,bf16}; extent: exactly one
        # fused square/initial-reduction issue. The earlier scalar X-to-FP32
        # cast remains live for the later epilogue.
    else:
        x_sq = reg_tile("f32", (TOTAL_VALUES,))
        for pair in static_range(PACKED_PAIRS):
            square_pair = mul(
                pair_view(x_f32,pair,unused_high_if_odd=True),
                pair_view(x_f32,pair,unused_high_if_odd=True), lanes=2)
            # instruction_selection: mul.f32x2; extent: PACKED_PAIRS issues,
            # including an unused high lane for an odd fragment
            for packed_lane in static_range(2):
                flat = pair * 2 + packed_lane
                if flat < TOTAL_VALUES:
                    x_sq[flat] = square_pair[packed_lane]

        local_sum = 0.0
        for flat in static_range(TOTAL_VALUES):
            local_sum = add(local_sum, x_sq[flat])
            # instruction_selection: add.f32; extent: one ordered scalar issue
            # per valid physical fragment value

    # ------------------------------------------------------------------
    # Expanded row_reduce_sum_multirow.
    # ------------------------------------------------------------------
    WARP_WIDTH = min(TPR, 32)
    for delta in powers_of_two_below(WARP_WIDTH):
        peer = shuffle_xor(local_sum, delta, width=32)
        # instruction_selection: shfl.sync.bfly.b32 with full mask and clamp
        # 31; extent: one scalar per explicit subgroup stage
        local_sum = add(local_sum, peer)
        # instruction_selection: add.f32; extent: one scalar per stage
    warp_sum = local_sum

    if WARPS_PER_ROW > 1 and CLUSTER_N == 1:
        if lane == 0:
            copy_r2s(warp_sum, reduction[row_warp,warp_in_row])
            # instruction_selection: st.shared.b32; extent: one per row warp
        cta_barrier()
        # instruction_selection: bar.sync; extent: one CTA publication edge
        final = 0.0
        if lane < WARPS_PER_ROW:
            final = copy_s2r(reduction[row_warp,lane])
            # instruction_selection: ld.shared.b32; extent: one scalar per
            # participating lane in every row warp
        for delta in (1,2,4,8,16):
            peer = shuffle_xor(final, delta, width=32)
            # instruction_selection: shfl.sync.bfly.b32; extent: one scalar
            # at each full-warp stage
            final = add(final, peer)
            # instruction_selection: add.f32; extent: one scalar per stage
        sum_sq = final

    elif CLUSTER_N > 1:
        if warp == 0 and elect_one():
            # instruction_selection: elect.sync; extent: one election in warp 0
            EXPECTED_BYTES = ROWS * WARPS_PER_ROW * CLUSTER_N * 4
            mbarrier_arrive_expect_tx(mbar, EXPECTED_BYTES)
            # instruction_selection: mbarrier.arrive.expect_tx.shared.b64;
            # extent: one elected issue per CTA
        if lane < CLUSTER_N:
            peer_reduction = map_shared_to_peer(
                reduction[row_warp,(warp_in_row,cta_rank)], lane)
            peer_mbar = map_shared_to_peer(mbar, lane)
            # instruction_selection: mapa.shared::cluster.u32; extent: two
            # address maps per sending lane
            remote_store_complete_tx(warp_sum, peer_reduction, peer_mbar)
            # instruction_selection:
            # st.async.shared::cluster.mbarrier::complete_tx::bytes.f32;
            # extent: one partial from every warp to every peer CTA
        wait_done = False
        while not wait_done:
            wait_done = mbarrier_try_wait_parity(mbar, phase=0, timeout=10000000)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 plus
            # uniform retry branch; extent: one retry loop per CTA thread
        total_partials = WARPS_PER_ROW * CLUSTER_N
        final = 0.0
        for i in static_range(ceil_div(total_partials,32)):
            partial = lane + i * 32
            if partial < total_partials:
                partial_value = copy_s2r(reduction[row_warp,partial])
                # instruction_selection: ld.shared.b32; extent: one scalar
                # for every partial owned by this lane
                final = add(final, partial_value)
                # instruction_selection: add.f32; extent: one scalar per load
        for delta in (1,2,4,8,16):
            peer = shuffle_xor(final, delta, width=32)
            # instruction_selection: shfl.sync.bfly.b32; extent: one scalar
            # at each full-warp stage
            final = add(final, peer)
            # instruction_selection: add.f32; extent: one scalar per stage
        sum_sq = final
    else:
        sum_sq = warp_sum

    if H == 1:
        shifted = add(sum_sq, runtime_eps)
        # instruction_selection: add.f32 after multiply-by-one folding;
        # extent: one scalar per thread
    elif is_power_of_two(H):
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
        # instruction_selection: bar.sync; extent: one post-reduction CTA edge

    # ------------------------------------------------------------------
    # Pass 2: source async reload, FP32 multiply phases, FP8 output paths.
    # ------------------------------------------------------------------
    if USE_ASYNC:
        for vb in static_range(VEC_BLOCKS):
            local_col = (thread_in_row + vb * TPR) * VEC
            copy_s2r(sX[row_in_cta,local_col:local_col+VEC], x_frag[:,vb], COPY_BITS)
            # instruction_selection: shared load from the VEC table; extent:
            # one vector per vb after the post-reduction synchronization
        for flat in static_range(TOTAL_VALUES):
            x_f32[flat] = cast("f32", x_frag[flat])
            # instruction_selection: cvt.f32.{f16,bf16}; extent: every X value

    for flat in static_range(TOTAL_VALUES):
        w_f32[flat] = cast("f32", w_frag[flat])
        # instruction_selection: cvt.f32.{f16,bf16}; extent: every W value

    normalized = reg_tile("f32", (TOTAL_VALUES,))
    biased_w = reg_tile("f32", (TOTAL_VALUES,))
    weighted = reg_tile("f32", (TOTAL_VALUES,))
    y_f32 = reg_tile("f32", (TOTAL_VALUES,))
    if TOTAL_VALUES == 1:
        normalized[0] = mul(x_f32[0],rstd)
        # instruction_selection: mul.f32; extent: one scalar normalization
        biased_w[0] = add(w_f32[0],0.0)
        # instruction_selection: add.f32; extent: one scalar W-bias operation
        weighted[0] = mul(normalized[0],biased_w[0])
        # instruction_selection: mul.f32; extent: one scalar weighted product
        y_f32[0] = mul(weighted[0],inv_scale)
        # instruction_selection: mul.f32; extent: one scalar inverse-scale product
    else:
        # Phase 2a: materialize the complete normalized fragment.
        for pair in static_range(PACKED_PAIRS):
            normalized_pair = mul(
                pair_view(x_f32,pair,unused_high_if_odd=True),
                scalar_pair(rstd,pair,TOTAL_VALUES,undefined_high_if_odd=True),
                lanes=2)
            # instruction_selection: mul.f32x2; extent: PACKED_PAIRS issues
            # over the complete fragment before any W-bias operation
            for packed_lane in static_range(2):
                flat = pair * 2 + packed_lane
                if flat < TOTAL_VALUES:
                    normalized[flat] = normalized_pair[packed_lane]

        # Phase 2b: materialize the complete biased-W fragment.
        for pair in static_range(PACKED_PAIRS):
            biased_pair = add(
                pair_view(w_f32,pair,unused_high_if_odd=True),
                scalar_pair(0.0,pair,TOTAL_VALUES,undefined_high_if_odd=True),
                lanes=2)
            # instruction_selection: add.f32x2; extent: PACKED_PAIRS issues
            # for the source's explicit weight_bias=0.0 expression
            for packed_lane in static_range(2):
                flat = pair * 2 + packed_lane
                if flat < TOTAL_VALUES:
                    biased_w[flat] = biased_pair[packed_lane]

        # Phase 2c: consume normalized and biased_w into the complete weighted
        # fragment; their lifetimes end after this loop.
        for pair in static_range(PACKED_PAIRS):
            weighted_pair = mul(
                pair_view(normalized,pair,unused_high_if_odd=True),
                pair_view(biased_w,pair,unused_high_if_odd=True), lanes=2)
            # instruction_selection: mul.f32x2; extent: PACKED_PAIRS issues
            # after the complete bias phase and before inverse-scale work
            for packed_lane in static_range(2):
                flat = pair * 2 + packed_lane
                if flat < TOTAL_VALUES:
                    weighted[flat] = weighted_pair[packed_lane]

        # Phase 2d: consume weighted into the final output fragment; only
        # y_f32 remains live for FP8 conversion and stores.
        for pair in static_range(PACKED_PAIRS):
            y_pair = mul(
                pair_view(weighted,pair,unused_high_if_odd=True),
                scalar_pair(inv_scale,pair,TOTAL_VALUES,undefined_high_if_odd=True),
                lanes=2)
            # instruction_selection: mul.f32x2; extent: PACKED_PAIRS issues
            # after every weighted-fragment issue
            for packed_lane in static_range(2):
                flat = pair * 2 + packed_lane
                if flat < TOTAL_VALUES:
                    y_f32[flat] = y_pair[packed_lane]

    actual_row = row_i64
    col_offset = thread_in_row * VEC
    FP8_MAX = 448.0 if OUTPUT_DTYPE == "e4m3fn" else 57344.0

    for vb in static_range(VEC_BLOCKS):
        local_col = col_offset + vb * TPR * VEC
        absolute_col = block_y * COLS_PER_TILE + local_col
        y_offset = (actual_row * H + absolute_col) if COMPACT else \
                   (actual_row * y_row_stride + cast_i64(absolute_col))

        if VEC == 8 and absolute_col + 8 <= H and actual_row < runtime_M:
            p01 = convert_fp8_pair(y_f32[vb*8+0], y_f32[vb*8+1], OUTPUT_DTYPE)
            p23 = convert_fp8_pair(y_f32[vb*8+2], y_f32[vb*8+3], OUTPUT_DTYPE)
            p45 = convert_fp8_pair(y_f32[vb*8+4], y_f32[vb*8+5], OUTPUT_DTYPE)
            p67 = convert_fp8_pair(y_f32[vb*8+6], y_f32[vb*8+7], OUTPUT_DTYPE)
            # instruction_selection: four
            # cvt.rn.satfinite.{e4m3,e5m2}x2.f32 issues, each PTX pair ordered
            # high operand then low operand; extent: one complete vec8 fragment
            lo = pack_two_b16(p01,p23)
            hi = pack_two_b16(p45,p67)
            # instruction_selection: mov.b32 {b16,b16}; extent: two packed words
            copy_r2g((lo,hi), address(out_storage,y_offset), bits=64)
            # instruction_selection: st.global.v2.b32; extent: one vec8 store

        elif VEC == 4 and absolute_col + 4 <= H and actual_row < runtime_M:
            p01 = convert_fp8_pair(y_f32[vb*4+0], y_f32[vb*4+1], OUTPUT_DTYPE)
            p23 = convert_fp8_pair(y_f32[vb*4+2], y_f32[vb*4+3], OUTPUT_DTYPE)
            # instruction_selection: two
            # cvt.rn.satfinite.{e4m3,e5m2}x2.f32 issues; extent: one vec4 fragment
            packed = pack_two_b16(p01,p23)
            # instruction_selection: mov.b32 {b16,b16}; extent: one packed word
            copy_r2g(packed, address(out_storage,y_offset), bits=32)
            # instruction_selection: st.global.b32; extent: one vec4 store

        elif VEC == 2 and absolute_col + 2 <= H and actual_row < runtime_M:
            p01 = convert_fp8_pair(y_f32[vb*2+0], y_f32[vb*2+1], OUTPUT_DTYPE)
            # instruction_selection:
            # cvt.rn.satfinite.{e4m3,e5m2}x2.f32; extent: one vec2 fragment
            copy_r2g(p01, address(out_storage,y_offset), bits=16)
            # instruction_selection: st.global.b16; extent: one vec2 store

        else:
            for e in static_range(VEC):
                scalar_col = absolute_col + e
                if scalar_col < H and actual_row < runtime_M:
                    clamped_low = maximum(y_f32[vb*VEC+e], -FP8_MAX)
                    # instruction_selection: setp.le.f32 + selp.f32; extent:
                    # one compare-select pair per valid scalar tail value
                    clamped = minimum(clamped_low, FP8_MAX)
                    # instruction_selection: setp.ge.f32 + selp.f32; extent:
                    # one compare-select pair per valid scalar tail value
                    pair = convert_fp8_pair(clamped, 0.0, OUTPUT_DTYPE)
                    # instruction_selection:
                    # cvt.rn.satfinite.{e4m3,e5m2}x2.f32 with PTX operands
                    # zero,clamped so the valid result occupies the stored byte;
                    # extent: one scalar tail value
                    scalar_offset = (actual_row * H + scalar_col) if COMPACT else \
                                    (actual_row * y_row_stride + cast_i64(scalar_col))
                    copy_r2g(pair.low_byte, address(out_storage,scalar_offset), bits=8)
                    # instruction_selection: st.global.b8; extent: one valid scalar

    if ENABLE_PDL:
        pdl_signal()
        # instruction_selection: griddepcontrol.launch_dependents; extent:
        # one issue per CTA after all stores
```

## Addressing, predicates, tails, and excluded software conversion

- Compact X addressing may use the source constexpr row stride only because
  the host proves `M*H <= INT32_MAX`. The strided ABI keeps the row, stride
  product, H-column addition, and final pointer offset Int64. Output addressing
  begins from the source's explicit `actual_row:Int64` on both paths.
- X and Y are one-dimensional backing buffers in TIRx. Strides are independent
  and measured in elements; an FP8 Y element is one byte.
- `predicate_k` tests the first column of each input vector. Legal H partitions
  and stride divisibility make a true predicate cover the complete vector.
  Async false columns issue zero source bytes rather than skip the instruction.
- Output full-vector checks use `absolute_col+VEC<=H` and Int64 row bounds.
  Every failed vec8/4/2 check enters the scalar loop, where row and each column
  are checked again before clamp/convert/store.
- Out-of-range rows still execute scale load/reciprocal, W loads, async
  commit/wait, reduction, every barrier, and PDL signal; synchronization is not
  nested under the row predicate.
- The source software E4M3/E5M2 converters are recorded as an unreachable
  pre-SM89 alternative. They are not transcribed because SM100 fixes
  `USE_HW_FP8=True`.

## Storage ownership and lifetimes

| storage | owner | lifetime / reuse rule |
| --- | --- | --- |
| device scale scalar / `inv_scale` | every thread loads the same element | immediately after PDL wait through the final FP8 conversion phase |
| `sX` | each row-thread writes and reads its `(row,local_col)` vectors | async issue through the post-reduction reload |
| reduction buffer | row warps; clustered copies are remotely published | publication through replicated final reduction and post-reduction sync |
| cluster mbarrier | one per CTA, initialized by thread 0 | cluster initialization through remote-store completion wait |
| X fragment | one thread | sync: load through epilogue; async: first load through square, then post-sync reload |
| W fragment | one thread | global load before async wait through complete weighted normalization |
| widened X / squares | one thread | X widening through ordered local sum; async X widening is recreated after sync |
| widened W | one thread | W widening through completion of the biased-W phase |
| normalized fragment | one thread | complete normalization phase through its consumption by the weighted phase |
| biased-W fragment | one thread | complete bias phase through its consumption by the weighted phase |
| weighted fragment | one thread | complete weighted phase through its consumption by inverse-scale phase |
| final `y_f32` fragment | one thread | complete inverse-scale phase through FP8 conversion/store |
| FP8 b16 pairs / b32 words | one thread | one vector block's conversion through its single packed store |

Each row group owns one output row and each `(cluster_y,thread,vb,e)` owns a
disjoint H element after predicates. Cluster CTAs communicate only reduction
partials; X, W, scale, and Y traffic is otherwise independent.

## Module and verification contract

- Registry name is `flashinfer_rmsnorm_quant`, category `flashinfer`, compute
  capability 10; the factory supports every positive-H source specialization.
- Compact ABI is `x,weight,out,M:int64,scale:f32[1],eps:f32`; strided ABI appends
  `x_row_stride:int64,y_row_stride:int64`. The public API returns `None` and
  preserves the caller-provided output object and stride.
- `CONFIGS` contains exactly 1552 unique rows; the eight `BENCH_CONFIGS` are
  direct selections from that same closure and the suite YAML lists all eight.
- Correctness must prove public `rmsnorm_quant_cute` dispatch, compare raw FP8
  bytes against FlashInfer, compare dequantized values against an independent
  FP32 oracle with `rtol=atol=1`, preserve X/W/scale, and protect output padding
  and tails. Large-H inputs are periodic and non-constant.
- Every specialization undergoes low-level IR rejection of `TilePrimitiveCall`,
  `tirx.tile.*`, tile primitives, `T.cuda.func_call`, and `cuda_func_call`.
- Performance passes only when all eight suite rows have both `tirx` and
  `flashinfer_cutedsl`, no unresolved error/interference, and each individual
  `flashinfer_cutedsl_time/tirx_time` ratio is strictly above 0.99.

## Instruction selection is a lowering consequence

The port preserves source cluster selection, quant-only thread override, TV
layout, vector width, copy branch, fragment order, scale-load position,
reduction publication, post-reduction synchronization, async-X reload, four
FP32 epilogue phases, FP8 pair operand order, store width, and PDL boundaries.
Plain TIRx and typed/local PTX may express reviewed instructions. They may not
scalarize a full packed FP8 path, retain X FP32 registers instead of reloading
shared X, reassociate inverse-scale multiplication ahead of weighting, weaken
Int64 addressing or predicates, replace FTZ reciprocal/rsqrt, or emit any tile
or CUDA function-call primitive.
