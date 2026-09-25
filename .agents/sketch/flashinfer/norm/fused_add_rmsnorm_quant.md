<!--
Copyright (c) 2025 by FlashInfer team.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: Apache-2.0

This execution sketch documents a TIRx port of FlashInfer's CuTe-DSL
FusedAddRMSNormQuantKernel. See LICENSE, NOTICE, and licenses/ for applicable
terms.
-->

# FlashInfer fused add + RMSNorm + FP8 quantization SM100: execution sketch

This file is a non-executable execution sketch. It freezes the source-shaped
execution of FlashInfer's CuTe-DSL `FusedAddRMSNormQuantKernel` for
[`tirx_kernels/flashinfer/norm/fused_add_rmsnorm_quant.py`](../../../../tirx_kernels/flashinfer/norm/fused_add_rmsnorm_quant.py).
The target becomes executable only after independent source/line-info-PTX
review; after the first reviewer PASS this file is permanently frozen.

The source is fixed at FlashInfer commit
`f2e04400e330fb2debe0bf8730d9424a1d37927f`:

- `flashinfer/norm/kernels/fused_add_rmsnorm.py`, SHA256
  `f6f4a7d9c88996f26c33ed2661f82026e4ee3e747bf12c57febe9f9e68e61435`;
- inherited launch/layout helpers in `flashinfer/norm/kernels/rmsnorm.py`,
  SHA256 `b273fe5444aaf86a1600c196817b7a733b18f6f82030a16e1ef2731c784d48f0`;
- reduction, reciprocal, and FP8 helpers in `flashinfer/norm/utils.py`, SHA256
  `3f44ac6727c58883420068bf0aa5b239b12d2e86819ad80e54bc1bc016ec881a`;
- public dispatch in `flashinfer/norm/__init__.py`, SHA256
  `226a88f5fb14e78e06e1be79020edcae01bfa9e53e677bd485373ea4d51cffcb`.

The family covers FP16/BF16 input, residual, and weight; E4M3FN/E5M2
caller-provided output; `weight_bias=0.0`; compact and three independently
row-strided Int64 tensors; PDL off/on; synchronous and one-group paired
`cp.async` traffic; sub-warp through cluster reduction; cluster sizes
1/2/4/8/16; early in-place residual publication; and SM100 hardware FP8
conversion. Gemma bias, non-quant fused add, plain RMSNormQuant, legacy CUDA,
pre-SM89 software FP8 conversion, and every tile primitive are out of scope.

## Source dispatch and resource formulae

Cluster selection is inherited from `FusedAddRMSNormKernel` and therefore uses
the two-tile estimate with the inherited thread count. Only after that choice
does the quant kernel apply its `H_PER_CTA > 8192` 256-thread override. That
order is load-bearing: H12288 remains async with two rows, while H16384 and
clustered H32768 use two rows but become synchronous.

```python
ELEM_BYTES = 2
OPTIN_SMEM_BYTES = 232448
MAX_VEC = 8

def threads_per_row(h):
    if h <= 64: return 8
    if h <= 128: return 16
    if h <= 3072: return 32
    if h <= 6144: return 64
    if h <= 16384: return 128
    return 256

def fused_base(H, cluster_n):
    h = H // cluster_n
    tpr = threads_per_row(h)
    threads = 128 if h <= 16384 else 256
    rows = threads // tpr
    warps = max(tpr // 32, 1)
    vec = min(h & -h, MAX_VEC)
    blocks = max(1, ceil_div(h // vec, tpr))
    cols = vec * blocks * tpr
    estimate = 2 * rows * cols * ELEM_BYTES \
             + rows * warps * cluster_n * 4 \
             + (8 if cluster_n > 1 else 0)
    return estimate

def choose_cluster(H):
    best_fit = 1
    for c in (1, 2, 4, 8, 16):
        if H % c: continue
        required = fused_base(H, c)
        if required <= OPTIN_SMEM_BYTES // 2: return c
        if required <= OPTIN_SMEM_BYTES and best_fit == 1: best_fit = c
    return best_fit

def source_config(H):
    cluster_n = choose_cluster(H)
    h = H // cluster_n
    tpr = threads_per_row(h)
    threads = 128 if h <= 16384 else 256
    if h > 8192 and threads < 256: threads = 256
    rows = threads // tpr
    warps = max(tpr // 32, 1)
    vec = min(h & -h, MAX_VEC)
    copy_bits = 16 * vec
    blocks = max(1, ceil_div(h // vec, tpr))
    cols = vec * blocks * tpr
    two_tile_bytes = 2 * rows * cols * ELEM_BYTES
    use_async = copy_bits >= 32 and two_tile_bytes <= OPTIN_SMEM_BYTES // 2
    reduce_bytes = rows * warps * cluster_n * 4
    smem_bytes = (two_tile_bytes if use_async else 0) + reduce_bytes \
                 + (8 if cluster_n > 1 else 0)
    rolled = vec * blocks > 512
    return cluster_n, h, tpr, threads, rows, warps, vec, blocks, cols, \
           use_async, smem_bytes, rolled
```

The host selects compact only when Y, X, and residual are all contiguous and
`M*H <= 2**31-1`. Otherwise the runtime ABI appends independent
`y_row_stride`, `x_row_stride`, and `residual_row_stride` Int64 element
strides, each divisible by VEC. Weight stays compact and scale is a compact
device FP32 tensor of shape one. Every tensor has unit inner stride. Compact Y
has assumed alignment `gcd(128,H*1)` bytes, compact X/residual have assumed
alignment `gcd(128,H*2)` bytes, strided Y/X/residual each have assumed
alignment 16 bytes, weight has assumed alignment 16 bytes, and scale has
assumed alignment 4 bytes.

`TOTAL_VALUES=VEC*VEC_BLOCKS` is source-constexpr. TIRx uses static unrolling
through 512 values and a serial `unroll=False` loop above 512. This changes
only physical expansion, not ownership, source order, association, predicates,
or instruction family. H131070 and H131073 freeze the 512/513 boundary;
H1048576 is the rolled cluster-1 synchronous fallback guard.

## Pipeline at a glance

| lanes / CTA role | source-owned work | publication or reuse edge |
| --- | --- | --- |
| every thread at entry | optional PDL wait, load `scale[0]`, form FTZ approximate reciprocal | `inv_scale` remains live through FP8 stores |
| each contiguous TPR-thread group | one logical row and one cluster H partition; each thread owns VEC adjacent columns per vector block | X, residual, W, and FP32 h remain thread-private |
| every thread before reduction | load X/R, widen, packed-add to h, narrow/store h to residual, square h | early residual store is source-ordered before reduction |
| each row sub-warp/full warp | ordered local sum then width-8/16/32 butterfly stages | row-warp partial is replicated |
| lane 0 of each row warp, cluster-1 multi-warp path | publish one partial in shared memory | CTA barrier before final shared loads |
| lanes `<CLUSTER_N` of each warp, cluster path | remotely publish the warp partial to every peer CTA | two `mapa`, complete-tx store, and peer mbarrier |
| every row warp after publication | redundantly load and reduce all row partials | replicated sum feeds divide/add/rsqrt |
| all threads after reduction | cluster arrive/wait or CTA barrier, retain FP32 h and W | unlike RMSNormQuant, there is no post-reduction X reload |
| each row thread in epilogue | materialize h*rstd, W+0, their product, inverse-scale product, then packed/scalar FP8 stores | output fragment dies after its owned stores |

There is exactly one async group: all X issues precede all residual issues,
then one commit; W loads occur before one wait; shared X loads precede shared
residual loads. PDL wait precedes scale and dependency-sensitive global
traffic; PDL signal follows residual and output stores. Disabled PDL profiles
contain neither instruction nor launch attribute.

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

Key data, arithmetic, conversion, and synchronization operations remain
primitive:

```python
copy_g2r(src, dst, bits, predicate=None)
copy_g2s_async_ca(src, dst, bits, source_bytes)
copy_s2r(src, dst, bits)
copy_r2s(src, dst)
copy_r2g(src, dst, bits, predicate=None)
fill(dst, value)
cast(dtype, src, rounding=None)
add(lhs, rhs, lanes=1)
mul(lhs, rhs, lanes=1)
fma(lhs, rhs, acc)
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

Reviewed source PTX fixes vector traffic and output width:

| VEC | X/R/W global and residual store | shared X/R | FP8 output |
| ---: | --- | --- | --- |
| 1 | scalar `ld/st.global.b16` | sync only | clamp + pair-convert + `st.global.b8` |
| 2 | `ld/st.global.v2.b16` | `ld.shared.v2.b16` | pair-convert + `st.global.b16`; scalar tail fallback |
| 4 | `ld/st.global.v2.b32` | `ld.shared.v4.b16` | two pair-converts + pack + `st.global.b32`; scalar tail fallback |
| 8 | `ld/st.global.v4.b32` | `ld.shared.v4.b32` | four pair-converts + two packs + `st.global.v2.b32`; scalar tail fallback |

Every async copy is `cp.async.ca.shared.global` with immediate size
`COPY_BITS/8` and the source-size operand used for column-tail zero fill.
`add(...,lanes=2)` and `mul(...,lanes=2)` are native packed FP32 operations.
There is no compound fused-add, RMSNorm, reduction, quantization, tile-copy, or
tile-compute primitive.

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
    block=(THREADS, 1, 1),
    cluster=((1, CLUSTER_N, 1) if CLUSTER_N > 1 else None),
    dynamic_smem_bytes=SMEM_BYTES,
    use_programmatic_dependent_launch=ENABLE_PDL,
)
def flashinfer_fused_add_rmsnorm_quant(
    out_storage,                 # mutable OUTPUT_DTYPE, inner stride 1;
                                 # compact align=gcd(128,H), strided align=16
    input_storage,               # read-only INPUT_DTYPE, inner stride 1;
                                 # compact align=gcd(128,2*H), strided align=16
    residual_storage,            # mutable INPUT_DTYPE, same stride/alignment as X
    weight,                      # compact INPUT_DTYPE [H], stride 1, align=16
    runtime_M: i64,
    scale,                       # compact f32 [1], stride 1, align=4
    runtime_eps: f32,
    y_row_stride: i64 = H,       # appended only for strided_i64
    x_row_stride: i64 = H,
    residual_row_stride: i64 = H,
):
    tid = thread_id_x(THREADS)
    block_x = block_id_x()
    block_y = block_id_y() if CLUSTER_N > 1 else 0
    cta_rank = cta_rank_in_cluster() if CLUSTER_N > 1 else 0

    if ENABLE_PDL:
        pdl_wait()
        # instruction_selection: griddepcontrol.wait; extent: one static site,
        # executed by every CTA thread

    scale_value = copy_g2r(scale[0], bits=32)
    # instruction_selection: ld.global.b32; extent: one scalar per thread
    inv_scale = rcp_fast(scale_value)
    # instruction_selection: rcp.approx.ftz.f32; extent: one scalar per thread

    row_in_cta = tid // TPR
    thread_in_row = tid % TPR
    actual_row = cast_i64(block_x) * cast_i64(ROWS) + cast_i64(row_in_cta)
    compact_row_i32 = block_x * ROWS + row_in_cta if COMPACT else 0
    # compact_row_i32 exists only under the host M*H<=INT32_MAX proof.
    row_valid = actual_row < runtime_M
    warp = tid // 32
    lane = tid % 32
    row_warp = warp // WARPS_PER_ROW
    warp_in_row = warp % WARPS_PER_ROW

    smem = raw_shared("u8", SMEM_BYTES, alignment=16)
    cursor = 0
    if USE_ASYNC:
        sX = view(smem, INPUT_DTYPE, (ROWS,COLS), byte_offset=cursor,
                  row_major=True, alignment=16)
        cursor += ROWS * COLS * 2
        sR = view(smem, INPUT_DTYPE, (ROWS,COLS), byte_offset=cursor,
                  row_major=True, alignment=16)
        cursor += ROWS * COLS * 2
    if CLUSTER_N == 1:
        reduction = view(smem, "f32", (ROWS,WARPS_PER_ROW),
                         stride=(1,ROWS), byte_offset=cursor, alignment=4)
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

    x_frag = reg_tile(INPUT_DTYPE, (VEC,VEC_BLOCKS), stride=(1,VEC))
    r_frag = reg_tile(INPUT_DTYPE, (VEC,VEC_BLOCKS), stride=(1,VEC))
    w_frag = reg_tile(INPUT_DTYPE, (VEC,VEC_BLOCKS), stride=(1,VEC))
    TOTAL_VALUES = VEC * VEC_BLOCKS
    PAIRS = ceil_div(TOTAL_VALUES, 2)
    FRAGMENT_RANGE = serial_range(unroll=False) if TOTAL_VALUES > 512 else static_range

    if not USE_ASYNC:
        for flat in FRAGMENT_RANGE(TOTAL_VALUES):
            fill(x_frag.flat[flat], zero(INPUT_DTYPE))
            fill(r_frag.flat[flat], zero(INPUT_DTYPE))
            # instruction_selection: mov.b16 zero materialization; extent: two values

    # All X issues precede all residual issues.
    for vb in FRAGMENT_RANGE(VEC_BLOCKS):
        local_col = (thread_in_row + vb * TPR) * VEC
        absolute_col = block_y * COLS + local_col
        col_valid = absolute_col < H
        x_offset = (compact_row_i32 * H + absolute_col) if COMPACT else \
                   (actual_row * x_row_stride + cast_i64(absolute_col))
        if USE_ASYNC:
            if row_valid:
                copy_g2s_async_ca(address(input_storage,x_offset),
                                  sX[row_in_cta,local_col:local_col+VEC],
                                  COPY_BITS, (COPY_BITS//8) if col_valid else 0)
                # instruction_selection: cp.async.ca.shared.global; extent:
                # one row-gated issue per vb with runtime source size
        elif row_valid and col_valid:
            copy_g2r(address(input_storage,x_offset), x_frag[:,vb], COPY_BITS)
            # instruction_selection: global load from VEC table; extent: one vector

    for vb in FRAGMENT_RANGE(VEC_BLOCKS):
        local_col = (thread_in_row + vb * TPR) * VEC
        absolute_col = block_y * COLS + local_col
        col_valid = absolute_col < H
        r_offset = (compact_row_i32 * H + absolute_col) if COMPACT else \
                   (actual_row * residual_row_stride + cast_i64(absolute_col))
        if USE_ASYNC:
            if row_valid:
                copy_g2s_async_ca(address(residual_storage,r_offset),
                                  sR[row_in_cta,local_col:local_col+VEC],
                                  COPY_BITS, (COPY_BITS//8) if col_valid else 0)
                # instruction_selection: cp.async.ca.shared.global; extent:
                # one row-gated issue per vb after every X issue
        elif row_valid and col_valid:
            copy_g2r(address(residual_storage,r_offset), r_frag[:,vb], COPY_BITS)
            # instruction_selection: global load from VEC table; extent: one vector

    if USE_ASYNC:
        cp_async_commit()
        # instruction_selection: cp.async.commit_group; extent: one per thread

    for vb in FRAGMENT_RANGE(VEC_BLOCKS):
        local_col = (thread_in_row + vb * TPR) * VEC
        absolute_col = block_y * COLS + local_col
        if absolute_col < H:
            copy_g2r(weight[absolute_col:absolute_col+VEC], w_frag[:,vb], COPY_BITS)
            # instruction_selection: global load from VEC table; extent: one vector

    if USE_ASYNC:
        cp_async_wait(0)
        # instruction_selection: cp.async.wait_group 0; extent: one per thread
        for vb in FRAGMENT_RANGE(VEC_BLOCKS):
            local_col = (thread_in_row + vb * TPR) * VEC
            copy_s2r(sX[row_in_cta,local_col:local_col+VEC], x_frag[:,vb], COPY_BITS)
            # instruction_selection: shared load from VEC table; extent: one vector
        for vb in FRAGMENT_RANGE(VEC_BLOCKS):
            local_col = (thread_in_row + vb * TPR) * VEC
            copy_s2r(sR[row_in_cta,local_col:local_col+VEC], r_frag[:,vb], COPY_BITS)
            # instruction_selection: shared load from VEC table; extent: one vector

    x_f32 = reg_tile("f32", (TOTAL_VALUES,))
    r_f32 = reg_tile("f32", (TOTAL_VALUES,))
    for flat in FRAGMENT_RANGE(TOTAL_VALUES):
        x_f32[flat] = cast("f32", x_frag.flat[flat])
        # instruction_selection: cvt.f32.{f16,bf16}; extent: one X value
    for flat in FRAGMENT_RANGE(TOTAL_VALUES):
        r_f32[flat] = cast("f32", r_frag.flat[flat])
        # instruction_selection: cvt.f32.{f16,bf16}; extent: one residual value

    h = reg_tile("f32", (TOTAL_VALUES,))
    for pair in FRAGMENT_RANGE(PAIRS):
        h.store_pair(pair, add(pair_view(x_f32,pair), pair_view(r_f32,pair), lanes=2))
        # instruction_selection: add.f32x2; extent: PAIRS packed issues

    h_narrow = reg_tile(INPUT_DTYPE, (TOTAL_VALUES,))
    if VEC == 1 or (VEC == 2 and VEC_BLOCKS == 3):
        for flat in FRAGMENT_RANGE(TOTAL_VALUES):
            h_narrow[flat] = cast(INPUT_DTYPE, h[flat], rounding="rn")
            # instruction_selection: cvt.rn.{f16,bf16}.f32; extent: one scalar
    else:
        for pair in FRAGMENT_RANGE(PAIRS):
            h_narrow.store_pair(pair, cast(INPUT_DTYPE+"x2", pair_view(h,pair), rounding="rn"))
            # instruction_selection: cvt.rn.{f16,bf16}x2.f32; extent: one pair
    for vb in FRAGMENT_RANGE(VEC_BLOCKS):
        local_col = (thread_in_row + vb * TPR) * VEC
        absolute_col = block_y * COLS + local_col
        r_offset = (compact_row_i32 * H + absolute_col) if COMPACT else \
                   (actual_row * residual_row_stride + cast_i64(absolute_col))
        if row_valid and absolute_col < H:
            copy_r2g(h_narrow.slice(vb,VEC), address(residual_storage,r_offset), COPY_BITS)
            # instruction_selection: global store from VEC table; extent: one vector

    h_sq = reg_tile("f32", (TOTAL_VALUES,))
    for pair in FRAGMENT_RANGE(PAIRS):
        h_sq.store_pair(pair, mul(pair_view(h,pair), pair_view(h,pair), lanes=2))
        # instruction_selection: mul.f32x2; extent: PAIRS packed issues
    local_sum = 0.0
    for flat in FRAGMENT_RANGE(TOTAL_VALUES):
        local_sum = add(local_sum, h_sq[flat])
        # instruction_selection: add.f32; extent: ordered TOTAL_VALUES chain

    for delta in powers_of_two_below(min(TPR,32)):
        peer = shuffle_xor(local_sum, delta, width=32)
        # instruction_selection: shfl.sync.bfly.b32; extent: one subgroup stage
        local_sum = add(local_sum, peer)
        # instruction_selection: add.f32; extent: one subgroup stage
    warp_sum = local_sum

    if WARPS_PER_ROW > 1 and CLUSTER_N == 1:
        if lane == 0:
            copy_r2s(warp_sum, reduction[row_warp,warp_in_row])
            # instruction_selection: st.shared.b32; extent: one per row warp
        cta_barrier()
        # instruction_selection: bar.sync 0; extent: one publication edge
        final = 0.0
        if lane < WARPS_PER_ROW:
            final = copy_s2r(reduction[row_warp,lane])
            # instruction_selection: ld.shared.b32; extent: one participating lane
        for delta in (1,2,4,8,16):
            peer = shuffle_xor(final, delta, width=32)
            # instruction_selection: shfl.sync.bfly.b32; extent: one full-warp stage
            final = add(final, peer)
            # instruction_selection: add.f32; extent: one full-warp stage
        sum_sq = final
    elif CLUSTER_N > 1:
        if warp == 0 and elect_one():
            # instruction_selection: elect.sync; extent: one election in warp 0
            mbarrier_arrive_expect_tx(mbar, ROWS*WARPS_PER_ROW*CLUSTER_N*4)
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
            # extent: one partial to every peer CTA
        done = False
        while not done:
            done = mbarrier_try_wait_parity(mbar, phase=0, timeout=10000000)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 plus
            # uniform retry branch; extent: one wait loop per CTA thread
        final = 0.0
        for i in static_range(ceil_div(WARPS_PER_ROW*CLUSTER_N,32)):
            partial = lane + i*32
            if partial < WARPS_PER_ROW*CLUSTER_N:
                value = copy_s2r(reduction[row_warp,partial])
                # instruction_selection: ld.shared.b32; extent: one owned partial
                final = add(final, value)
                # instruction_selection: add.f32; extent: one loaded partial
        for delta in (1,2,4,8,16):
            peer = shuffle_xor(final, delta, width=32)
            # instruction_selection: shfl.sync.bfly.b32; extent: one full-warp stage
            final = add(final, peer)
            # instruction_selection: add.f32; extent: one full-warp stage
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
        # instruction_selection: bar.sync 0; extent: one post-reduction edge

    w_f32 = reg_tile("f32", (TOTAL_VALUES,))
    for flat in FRAGMENT_RANGE(TOTAL_VALUES):
        w_f32[flat] = cast("f32", w_frag.flat[flat])
        # instruction_selection: cvt.f32.{f16,bf16}; extent: one W value

    normalized = reg_tile("f32", (TOTAL_VALUES,))
    biased_w = reg_tile("f32", (TOTAL_VALUES,))
    weighted = reg_tile("f32", (TOTAL_VALUES,))
    y_f32 = reg_tile("f32", (TOTAL_VALUES,))
    for pair in FRAGMENT_RANGE(PAIRS):
        normalized.store_pair(pair, mul(pair_view(h,pair), scalar_pair(rstd,pair), lanes=2))
        # instruction_selection: mul.f32x2; extent: complete normalized phase
    for pair in FRAGMENT_RANGE(PAIRS):
        biased_w.store_pair(pair, add(pair_view(w_f32,pair), scalar_pair(0.0,pair), lanes=2))
        # instruction_selection: add.f32x2; extent: complete W-bias phase
    for pair in FRAGMENT_RANGE(PAIRS):
        weighted.store_pair(pair, mul(pair_view(normalized,pair), pair_view(biased_w,pair), lanes=2))
        # instruction_selection: mul.f32x2; extent: complete weighted phase
    for pair in FRAGMENT_RANGE(PAIRS):
        y_f32.store_pair(pair, mul(pair_view(weighted,pair), scalar_pair(inv_scale,pair), lanes=2))
        # instruction_selection: mul.f32x2; extent: complete inverse-scale phase

    FP8_MAX = 448.0 if OUTPUT_DTYPE == "e4m3fn" else 57344.0
    for vb in FRAGMENT_RANGE(VEC_BLOCKS):
        local_col = (thread_in_row + vb * TPR) * VEC
        absolute_col = block_y * COLS + local_col
        y_offset = (actual_row * H + absolute_col) if COMPACT else \
                   (actual_row * y_row_stride + cast_i64(absolute_col))
        if VEC == 8 and absolute_col + 8 <= H and actual_row < runtime_M:
            p01 = convert_fp8_pair(y_f32[vb*8+0], y_f32[vb*8+1], OUTPUT_DTYPE)
            p23 = convert_fp8_pair(y_f32[vb*8+2], y_f32[vb*8+3], OUTPUT_DTYPE)
            p45 = convert_fp8_pair(y_f32[vb*8+4], y_f32[vb*8+5], OUTPUT_DTYPE)
            p67 = convert_fp8_pair(y_f32[vb*8+6], y_f32[vb*8+7], OUTPUT_DTYPE)
            # instruction_selection: four
            # cvt.rn.satfinite.{e4m3,e5m2}x2.f32; extent: one vec8
            lo = pack_two_b16(p01,p23)
            hi = pack_two_b16(p45,p67)
            # instruction_selection: mov.b32 {b16,b16}; extent: two words
            copy_r2g((lo,hi), address(out_storage,y_offset), bits=64)
            # instruction_selection: st.global.v2.b32; extent: one vec8 store
        elif VEC == 4 and absolute_col + 4 <= H and actual_row < runtime_M:
            p01 = convert_fp8_pair(y_f32[vb*4+0], y_f32[vb*4+1], OUTPUT_DTYPE)
            p23 = convert_fp8_pair(y_f32[vb*4+2], y_f32[vb*4+3], OUTPUT_DTYPE)
            # instruction_selection: two pair converts; extent: one vec4
            packed = pack_two_b16(p01,p23)
            # instruction_selection: mov.b32 {b16,b16}; extent: one word
            copy_r2g(packed, address(out_storage,y_offset), bits=32)
            # instruction_selection: st.global.b32; extent: one vec4 store
        elif VEC == 2 and absolute_col + 2 <= H and actual_row < runtime_M:
            pair = convert_fp8_pair(y_f32[vb*2+0], y_f32[vb*2+1], OUTPUT_DTYPE)
            # instruction_selection: one pair convert; extent: one vec2
            copy_r2g(pair, address(out_storage,y_offset), bits=16)
            # instruction_selection: st.global.b16; extent: one vec2 store
        else:
            for e in static_range(VEC):
                scalar_col = absolute_col + e
                if scalar_col < H and actual_row < runtime_M:
                    low = maximum(y_f32[vb*VEC+e], -FP8_MAX)
                    # instruction_selection: setp.le.f32 + selp.f32; extent: one scalar
                    clamped = minimum(low, FP8_MAX)
                    # instruction_selection: setp.ge.f32 + selp.f32; extent: one scalar
                    pair = convert_fp8_pair(clamped, 0.0, OUTPUT_DTYPE)
                    # instruction_selection:
                    # cvt.rn.satfinite.{e4m3,e5m2}x2.f32 with zero high operand;
                    # extent: one scalar tail value
                    scalar_offset = (actual_row*H + scalar_col) if COMPACT else \
                                    (actual_row*y_row_stride + cast_i64(scalar_col))
                    copy_r2g(pair.low_byte, address(out_storage,scalar_offset), bits=8)
                    # instruction_selection: st.global.b8; extent: one valid scalar

    if ENABLE_PDL:
        pdl_signal()
        # instruction_selection: griddepcontrol.launch_dependents; extent: one
        # static site, executed by every CTA thread
```

## Export reconciliation and reviewed profiles

- Dynamic shared storage is one raw allocation. H4096 BF16 uses 32784 bytes:
  sX at 0, sR at 16384, then four FP32 row/warp partials. Its PTX has 16
  async issues (eight X then eight residual), one commit, eight W vector loads,
  one wait, 16 shared vector loads, eight residual vectors, and eight FP8 vec8
  stores.
- H64 selects VEC8; H66 selects VEC2 with three vector blocks and a source-size
  tail; H111 selects VEC1 synchronous scalar traffic and scalar FP8 stores;
  strided H500 selects VEC4 and independent Int64 X/R/Y row products.
- The quant override makes H12288 two-row async but H16384 two-row sync.
  H32768/65536/131072/262144 choose cluster 2/4/8/16 and sync after the same
  override. H524288 remains cluster16 sync; H1048576 falls back to cluster1
  sync.
- Cluster PTX contains mbarrier init/fence, two cluster arrive/wait pairs, two
  `mapa.shared::cluster`, one remote complete-tx store site, and the uniform
  parity-wait loop. Because `blockIdx.y` is runtime, clustered vec8 PTX retains
  both packed and scalar-tail output arms even though legal launch coordinates
  own complete partitions.
- H131070 has exactly 512 physical values; H131073 has 513 and the source PTX
  expands all 513. TIRx deliberately rolls the latter and every larger
  fragment without changing the ordered loop body.
- PDL-enabled profiles contain exactly one wait and one signal. E4M3 and E5M2
  exports use their corresponding hardware pair conversion with PTX high/low
  operands reversed relative to logical `(lo,hi)` helper arguments.

## Addressing, predicates, mutation, and tails

- Runtime ABI order is `(out,input,residual,weight,M:i64,scale:f32[1],eps)`;
  the strided specialization appends Y/X/residual Int64 row strides. Only out
  and residual mutate. Input, weight, and scale are read-only.
- Compact X/R offsets may narrow under the host `M*H<=INT32_MAX` proof.
  Output begins from explicit `actual_row:Int64`; every strided row product and
  pointer addition remains Int64 and independent.
- One column predicate tests each input vector's first coordinate. Async false
  columns issue zero source bytes; synchronous false columns skip the load.
  Row validity gates X/R traffic and residual stores but not W, reduction, or
  synchronization.
- Residual observes `cast_INPUT_DTYPE(fp32(input)+fp32(residual))` before the
  reduction. RMSNorm and quantization consume the original FP32 h, never the
  narrowed residual value.
- FP8 full vectors rely on `satfinite`; every failed vec8/4/2 guard enters the
  scalar loop with explicit finite-range max/min clamp and independent row and
  column guards.

## Storage ownership and lifetimes

| storage | owner | lifetime / reuse rule |
| --- | --- | --- |
| compact Y / X / residual | global, unit inner stride | assumed alignment `gcd(128,H) / gcd(128,2H) / gcd(128,2H)` bytes |
| strided Y / X / residual | global, unit inner stride | independent Int64 row strides divisible by VEC; assumed alignment 16 bytes |
| weight / scale | global, compact, unit stride | assumed alignment 16 / 4 bytes |
| scale / `inv_scale` | every thread loads the same element | after PDL wait through FP8 output |
| `sX`, `sR` | one row-thread vector owner | async issue through one shared reload; dead after h |
| reduction buffer | row warps; cluster copies remotely publish | warp partial publication through final reduction |
| cluster mbarrier | thread 0 initializes; elected warp-0 lane expects bytes | init through remote completion wait |
| X/R fragments | one thread | copy through widening/add |
| FP32 h | one thread | add through residual store, reduction barrier, and quant epilogue |
| W fragment | one thread | W load before async wait through biased-W phase |
| residual output fragment | one thread | h narrowing through early residual store |
| normalized / biased-W / weighted | one thread | four source-ordered complete epilogue phases |
| final y FP32 fragment | one thread | inverse-scale phase through FP8 stores |
| FP8 pairs / packed words | one thread | one vector block's conversion through store |

## Module and verification contract

- Registry name is `flashinfer_fused_add_rmsnorm_quant`, category
  `flashinfer`, compute capability 10; `get_kernel` is the registered factory.
- `CONFIGS` contains exactly 1556 unique configurations. The three
  `BENCH_CONFIGS` are exactly FlashInfer's current BF16→E4M3 source benchmark
  rows: M32/H4096/PDL0, M64/H8192/PDL0, and M32/H4096/PDL1.
- Correctness calls the direct public CuTe-DSL path, compares raw FP8 bytes and
  an independent FP32 quant oracle, compares residual at `rtol=atol=1e-3`,
  checks finite values, object/stride identity, independent padding and guard
  regions, Int64 overflow paths, and input/weight/scale immutability.
- Every specialization undergoes low-level rejection of `TilePrimitiveCall`,
  `tirx.tile.*`, tile imports/primitives, `T.cuda.func_call`, and
  `cuda_func_call`.
- Final performance authority is only `python -m tirx_kernels.bench_suite`.
  Every one of the three exact `flashinfer_cutedsl_time/tirx_time` ratios must
  be strictly greater than 0.99 in one complete valid reference-enabled run.

## Instruction selection is a lowering consequence

The port preserves source cluster choice, quant thread override, TV fragment
order, X-before-residual async issue order, early residual publication,
ordered FP32 reduction, cluster transaction protocol, live h lifetime, four
FP32 output phases, FP8 operand order/store width, independent Int64 strides,
and PDL boundaries. Plain TIRx and typed local PTX may express reviewed
instructions. They may not use tile or CUDA function-call primitives,
scalarize a reviewed packed path, reload narrowed residual, reassociate scale
or weight multiplication, replace FTZ reciprocal/rsqrt, couple strides, move a
store across the reduction edge, or unroll fragments above the frozen 512-value
lowering boundary.
