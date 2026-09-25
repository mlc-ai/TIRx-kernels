<!--
This file is a design sketch for a TIRx port of code from FlashInfer
(https://github.com/flashinfer-ai/flashinfer @ f2e04400e330fb2debe0bf8730d9424a1d37927f), Copyright (c) 2025 by FlashInfer team.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# FlashInfer 3-D QK RMSNorm / Gemma RMSNorm SM100: execution sketch

This file is a non-executable execution sketch. It freezes the implementation
shape of FlashInfer's CuTe-DSL `QKRMSNormKernel` for the paired target
[`tirx_kernels/flashinfer/norm/qk_rmsnorm.py`](../../../../tirx_kernels/flashinfer/norm/qk_rmsnorm.py).
The target module is the source of truth once this sketch has passed its first
independent source/PTX review.

The source is fixed at FlashInfer commit
`f2e04400e330fb2debe0bf8730d9424a1d37927f`:

- `flashinfer/norm/kernels/rmsnorm.py`, SHA256
  `b273fe5444aaf86a1600c196817b7a733b18f6f82030a16e1ef2731c784d48f0`;
- `flashinfer/norm/utils.py`, SHA256
  `3f44ac6727c58883420068bf0aa5b239b12d2e86819ad80e54bc1bc016ec881a`;
- public dispatch in `flashinfer/norm/__init__.py`, SHA256
  `226a88f5fb14e78e06e1be79020edcae01bfa9e53e677bd485373ea4d51cffcb`.

The shared family covers FP16/BF16, `weight_bias=0.0` RMSNorm and
`weight_bias=1.0` Gemma RMSNorm, arbitrary legal 3-D X/Y batch/head strides,
PDL off/on, synchronous and `cp.async` X traffic, and sub-warp, one-warp, and
multi-warp row reduction. `QKRMSNormKernel` fixes `cluster_n=1`; cluster
launches, DSMEM, mbarriers, 2-D RMSNorm, fused-add, quantization, legacy CUDA,
and every tile primitive are out of scope.

## Source dispatch and resource formulae

These compile-time formulae are copied from `QKRMSNormKernel` and its inherited
`RMSNormKernel` helpers. They are not tuning choices:

```python
ELEM_BYTES = 2
MAX_VEC = 128 // 8 // ELEM_BYTES                         # 8 elements

def threads_per_row(H):
    if H <= 64: return 8
    if H <= 128: return 16
    if H <= 3072: return 32
    if H <= 6144: return 64
    if H <= 16384: return 128
    return 256

def source_config(H):
    tpr = threads_per_row(H)
    threads = 128 if H <= 16384 else 256
    rows = threads // tpr
    warps_per_row = max(tpr // 32, 1)
    vec = min(H & -H, MAX_VEC)
    copy_bits = 16 * vec
    vec_blocks = max(1, ceil_div(H // vec, tpr))
    cols = vec * vec_blocks * tpr
    tile_bytes = rows * cols * ELEM_BYTES
    use_async = copy_bits >= 32 and tile_bytes <= 232448 // 2
    smem_bytes = (tile_bytes if use_async else 0) + rows * warps_per_row * 4
    return tpr, threads, rows, warps_per_row, vec, copy_bits, \
           vec_blocks, cols, use_async, smem_bytes
```

The source always compiles symbolic `B:int64`, `N:int64`, and four symbolic
`int64` X/Y batch/head strides. Each stride is divisible by `VEC`; last-dimension
stride is one and each fake tensor has assumed 16-byte base alignment. Compact
and strided tensors therefore use the same compiled specialization.

## Pipeline at a glance

| lanes / CTA role | source-owned work | publication or reuse edge |
| --- | --- | --- |
| each contiguous group of `TPR` threads | one flattened `(batch,head)` row; each thread owns `VEC` adjacent columns in each vector block | no communication during X/W load |
| each sub-warp/full warp in a row | ordered fragment sum followed by butterfly sum at widths 8/16/32 | replicated partial within the row warp |
| lane 0 of each row warp when `WARPS_PER_ROW>1` | publish one partial to shared `(row,warp_in_row)` | CTA barrier before redundant final reduction |
| every row warp after publication | load the row's `WARPS_PER_ROW` partials and full-warp reduce | replicated row sum feeds mean/rsqrt |
| all CTA threads | post-reduction CTA barrier; async profiles then reload X from shared | barrier protects the reduction buffer and staged-X lifetime |

There is one `cp.async` group rather than a multi-stage pipeline. W is loaded
synchronously after async X commit and before X wait. PDL wait is the first
executable operation and PDL signal follows every output store. A PDL-disabled
specialization contains neither instruction nor launch attribute.

## Primitive vocabulary

Structural declarations and views do not emit data movement or arithmetic:

```python
specialize(...)
launch(...)
raw_shared(...)
view(...)
reg_tile(...)
address_3d(...)
pair_view(...)
scalar_pair(...)
```

Key data movement, arithmetic, and schedule operations remain primitive:

```python
copy_g2r(src, dst, bits, predicate=None)
copy_g2s_async_ca(src, dst, bits, source_bytes)
copy_s2r(src, dst, bits)
copy_r2s(src, dst)
copy_r2g(src, dst, bits, predicate=None)
fill(dst, value)
cast(dtype, src, rounding=None)
mul(lhs, rhs, lanes=1)
add(lhs, rhs, lanes=1)
fma(lhs, rhs, acc)
fma_half_inputs_to_f32(lhs, rhs, acc)
div(lhs, rhs)
rsqrt_fast(src)
shuffle_xor(src, delta, width)
cp_async_commit()
cp_async_wait(group)
cta_barrier()
pdl_wait()
pdl_signal()
```

The reviewed element (`E`) and packed (`P`) global-copy families are:

| `VEC` | element family `E` | packed family `P` | async shared load |
| ---: | --- | --- | --- |
| 1 | `.b16` | none | async is disabled |
| 2 | `.v2.b16` | `.b32` | `ld.shared.v2.b16` |
| 4 | `.v4.b16` | `.v2.b32` | `ld.shared.v4.b16` |
| 8 | none | `.v4.b32` | `ld.shared.v4.b32` |

The source emits distinct selectors for synchronous X, W, and Y rather than
one family based only on `VEC`. Define:

```python
FULL_TILE = H == COLS
VB_POW2 = VEC_BLOCKS > 0 and is_power_of_two(VEC_BLOCKS)
PACKED_NARROW = VEC > 1 and (VB_POW2 or FULL_TILE)

def sync_x_family():
    if VEC == 1: return E
    if VEC == 8: return P
    return P if VB_POW2 else E

def weight_profile():
    if VEC == 1: return [(E, VEC_BLOCKS)]
    if VEC == 8: return [(P, VEC_BLOCKS)]
    if VEC_BLOCKS == 1: return [(E, 1)]
    if VB_POW2: return [(P, VEC_BLOCKS)]
    if TOTAL_VALUES < 28: return [(E, VEC_BLOCKS)]
    return [(P, VEC_BLOCKS - 1), (E, 1)]

def output_family():
    if VEC == 1: return E
    if VEC == 8: return P
    return P if VB_POW2 else E

def sync_zero_fill_profile():
    if FULL_TILE and VB_POW2:
        return [("mov.b32 f32-zero seed", 1),
                ("mov.b64 pair materialization", PACKED_PAIRS)]
    if FULL_TILE:
        return [("mov.b32 f32-zero materialization", TOTAL_VALUES)]
    if VB_POW2 and TOTAL_VALUES > 1:
        return [("mov.b32 packed-b16-zero materialization",
                 TOTAL_VALUES // 2)]
    if TOTAL_VALUES == 1:
        return [("mov.b16 CFG-separated zero materialization", 2)]
    return [("mov.b16 zero materialization", TOTAL_VALUES)]
```

`ld.global` uses `sync_x_family()` and `weight_profile()`; `st.global` uses
`output_family()`. A scalar-narrowed VEC8 fragment is re-packed with `mov.b32`
before its `.v4.b32` store. `PACKED_NARROW` emits `TOTAL_VALUES/2`
`cvt.rn.{f16,bf16}x2.f32` instructions; otherwise it emits `TOTAL_VALUES`
scalar `cvt.rn.{f16,bf16}.f32` instructions. These are current SM100a NVPTX
lowering consequences of the source fragment vector shape and tail predicate,
not explicit branches in FlashInfer's Python source.

Every async X issue is `cp.async.ca.shared.global` with immediate copy size
`COPY_BITS/8` and the source predicate represented by the source-size operand.
`mul(..., lanes=2)` and `add(..., lanes=2)` each denote one native packed-FP32
instruction. No operation hides RMS normalization, a row reduction, a tile
copy, a role transition, or a synchronization edge.

## Complete sketch

```python
@specialize(
    VARIANT_WEIGHT_BIAS=(("rmsnorm", 0.0), ("gemma_rmsnorm", 1.0)),
    DTYPE=("f16", "bf16"),
    H="positive compile-time integer",
    ENABLE_PDL=(False, True),
    TARGET="sm_100a",
)
@launch(
    grid=(ceil_div(runtime_B * runtime_N, ROWS), 1, 1),
    block=(NUM_THREADS, 1, 1),
    cluster=None,
    dynamic_smem_bytes=SMEM_BYTES,
    use_programmatic_dependent_launch=ENABLE_PDL,
)
def flashinfer_qk_rmsnorm(
    x_storage,                 # DTYPE one-dimensional backing storage
    weight,                    # DTYPE [H], compact
    y_storage,                 # DTYPE one-dimensional backing storage
    runtime_B: i64,
    runtime_N: i64,
    runtime_eps: f32,
    x_batch_stride: i64,
    x_head_stride: i64,
    y_batch_stride: i64,
    y_head_stride: i64,
):
    tid = thread_id_x(NUM_THREADS)
    block = block_id_x()

    if ENABLE_PDL:
        pdl_wait()
        # instruction_selection: griddepcontrol.wait; extent: one issue per CTA

    # Source make_tv_layout:
    # shape=((TPR,1),(VEC,VEC_BLOCKS))
    # stride=((VEC,1),(1,VEC*TPR)).
    lane_in_row = tid % TPR
    row_in_cta = tid // TPR
    actual_row_i64 = cast_i64(block * ROWS + row_in_cta)
    runtime_M = runtime_B * runtime_N
    row_valid = actual_row_i64 < runtime_M
    batch_idx = actual_row_i64 // runtime_N
    head_idx = actual_row_i64 % runtime_N
    warp = tid // 32
    lane = tid % 32
    WARPS_PER_ROW = max(TPR // 32, 1)
    row_warp = warp // WARPS_PER_ROW
    warp_in_row = warp % WARPS_PER_ROW

    smem = raw_shared("u8", SMEM_BYTES, alignment=16)
    cursor = 0
    if USE_ASYNC:
        sX = view(smem, DTYPE, (ROWS,COLS), byte_offset=cursor,
                  row_major=True, alignment=16)
        cursor += ROWS * COLS * 2
    reduction = view(smem, "f32", (ROWS,WARPS_PER_ROW),
                     stride=(1,ROWS), byte_offset=cursor, alignment=4)

    # CuTe fragment layout is (VEC,VEC_BLOCKS):(1,VEC). Adjacent flattened
    # values therefore use flat=v+VEC*vb.
    x_frag = reg_tile(DTYPE, (VEC,VEC_BLOCKS), stride=(1,VEC))
    w_frag = reg_tile(DTYPE, (VEC,VEC_BLOCKS), stride=(1,VEC))
    x_f32 = reg_tile("f32", (VEC,VEC_BLOCKS), stride=(1,VEC))
    w_f32 = reg_tile("f32", (VEC,VEC_BLOCKS), stride=(1,VEC))
    TOTAL_VALUES = VEC * VEC_BLOCKS
    PACKED_PAIRS = ceil_div(TOTAL_VALUES, 2)
    FULL_TILE = H == COLS
    VB_POW2 = VEC_BLOCKS > 0 and is_power_of_two(VEC_BLOCKS)
    PACKED_NARROW = VEC > 1 and (VB_POW2 or FULL_TILE)

    # ----------------------------------------------------------------------
    # Pass 1: source-ordered X and W traffic.
    # ----------------------------------------------------------------------
    if not USE_ASYNC:
        for flat in static_range(TOTAL_VALUES):
            fill(x_frag[flat], zero(DTYPE))
            # instruction_selection: sync_zero_fill_profile(), independently
            # of sync_x_family(): FULL_TILE&&VB_POW2 emits one mov.b32 f32
            # zero seed then PACKED_PAIRS mov.b64; other FULL_TILE emits
            # TOTAL_VALUES mov.b32 f32-zero; partial VB_POW2 with more than
            # one value emits TOTAL_VALUES/2 mov.b32 packed-b16-zero;
            # TOTAL_VALUES==1 emits two CFG-separated mov.b16 zeros;
            # otherwise emits TOTAL_VALUES mov.b16. All precede sync loads.

    for vb in static_range(VEC_BLOCKS):
        local_col = (lane_in_row + vb * TPR) * VEC
        col_valid = local_col < H
        x_offset = batch_idx * x_batch_stride \
                 + head_idx * x_head_stride + cast_i64(local_col)

        if USE_ASYNC:
            if row_valid:
                source_bytes = (COPY_BITS // 8) if col_valid else 0
                copy_g2s_async_ca(
                    address_3d(x_storage, x_offset),
                    sX[row_in_cta,local_col:local_col+VEC],
                    bits=COPY_BITS,
                    source_bytes=source_bytes,
                )
                # instruction_selection: cp.async.ca.shared.global with
                # immediate COPY_BITS/8 and runtime source-size operand;
                # extent: one issue per vb for an in-bounds row
        else:
            if row_valid and col_valid:
                copy_g2r(address_3d(x_storage,x_offset), x_frag[:,vb], COPY_BITS)
                # instruction_selection: ld.global using sync_x_family():
                # VEC1 E, VEC8 P, VEC2/4 P iff VB_POW2 else E; extent:
                # VEC_BLOCKS row-and-column-predicated issues

    if USE_ASYNC:
        cp_async_commit()
        # instruction_selection: cp.async.commit_group; extent: one per CTA thread

    # W issue is after X issue/commit and before the async wait.
    for vb in static_range(VEC_BLOCKS):
        local_col = (lane_in_row + vb * TPR) * VEC
        if local_col < H:
            copy_g2r(weight[local_col:local_col+VEC], w_frag[:,vb], COPY_BITS)
            # instruction_selection: ld.global using weight_profile():
            # VEC1 all E, VEC8 all P; VEC2/4 use one E when VB==1,
            # all P when VB_POW2, all E when TOTAL_VALUES<28, otherwise
            # (VEC_BLOCKS-1) P plus one final predicated E tail

    if USE_ASYNC:
        cp_async_wait(0)
        # instruction_selection: cp.async.wait_group 0; extent: one per CTA thread
        for vb in static_range(VEC_BLOCKS):
            local_col = (lane_in_row + vb * TPR) * VEC
            copy_s2r(sX[row_in_cta,local_col:local_col+VEC], x_frag[:,vb], COPY_BITS)
            # instruction_selection: ld.shared.v2.b16 / v4.b16 / v4.b32 for
            # VEC 2/4/8 respectively; extent: one vector per vb

    # Widen the complete X fragment before forming any square.
    for flat in static_range(TOTAL_VALUES):
        x_f32[flat] = cast("f32", x_frag[flat])
        # instruction_selection: cvt.f32.{f16,bf16}; extent: one scalar per
        # physical fragment value in flattened source order

    if TOTAL_VALUES == 1:
        local_sum = fma_half_inputs_to_f32(x_frag[0], x_frag[0], 0.0)
        # instruction_selection: fma.rn.f32.{f16,bf16}; extent: exactly one.
        # Current NVPTX folds the source square and ordered reduction seed
        # together, while the earlier scalar x_f32 widening remains live for
        # the later epilogue.
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
            # instruction_selection: add.f32; extent: one ordered issue per
            # valid physical fragment value

    # ----------------------------------------------------------------------
    # row_reduce_sum_multirow with source cluster_n=1.
    # ----------------------------------------------------------------------
    WARP_WIDTH = min(TPR, 32)
    for delta in powers_of_two_below(WARP_WIDTH):
        peer = shuffle_xor(local_sum, delta, width=32)
        # instruction_selection: shfl.sync.bfly.b32 with full member mask and
        # clamp 31; extent: one scalar per explicit subgroup stage
        local_sum = add(local_sum, peer)
        # instruction_selection: add.f32; extent: one scalar per stage
    warp_sum = local_sum

    if WARPS_PER_ROW > 1:
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
            # instruction_selection: shfl.sync.bfly.b32 with full member mask
            # and clamp 31; extent: one scalar at each full-warp stage
            final = add(final, peer)
            # instruction_selection: add.f32; extent: one scalar per stage
        sum_sq = final
    else:
        sum_sq = warp_sum

    if H == 1:
        shifted = add(sum_sq, runtime_eps)
        # instruction_selection: add.f32 after the multiply-by-one fold;
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

    cta_barrier()
    # instruction_selection: bar.sync; extent: one source post-reduction CTA edge

    # ----------------------------------------------------------------------
    # Pass 2: async reload, W conversion, packed epilogue, and final stores.
    # ----------------------------------------------------------------------
    if USE_ASYNC:
        for vb in static_range(VEC_BLOCKS):
            local_col = (lane_in_row + vb * TPR) * VEC
            copy_s2r(sX[row_in_cta,local_col:local_col+VEC], x_frag[:,vb], COPY_BITS)
            # instruction_selection: exact shared load from the VEC table;
            # extent: one vector per vb after the CTA barrier
        for flat in static_range(TOTAL_VALUES):
            x_f32[flat] = cast("f32", x_frag[flat])
            # instruction_selection: cvt.f32.{f16,bf16}; extent: every X value

    for flat in static_range(TOTAL_VALUES):
        w_f32[flat] = cast("f32", w_frag[flat])
        # instruction_selection: cvt.f32.{f16,bf16}; extent: every W value

    normalized = reg_tile("f32", (TOTAL_VALUES,))
    biased_w = reg_tile("f32", (TOTAL_VALUES,))
    y_f32 = reg_tile("f32", (TOTAL_VALUES,))
    if TOTAL_VALUES == 1:
        normalized[0] = mul(x_f32[0], rstd)
        # instruction_selection: mul.f32; extent: one before the bias add
        biased_w[0] = add(w_f32[0], WEIGHT_BIAS)
        # instruction_selection: add.f32; extent: one before the second mul
        y_f32[0] = mul(normalized[0], biased_w[0])
        # instruction_selection: mul.f32; extent: one before scalar narrowing
    else:
        for pair in static_range(PACKED_PAIRS):
            normalized_pair = mul(
                pair_view(x_f32,pair,unused_high_if_odd=True),
                scalar_pair(rstd,pair,TOTAL_VALUES,undefined_high_if_odd=True),
                lanes=2,
            )
            # instruction_selection: mul.f32x2; extent: PACKED_PAIRS issues
            # over the complete fragment before any bias add
            for packed_lane in static_range(2):
                flat = pair * 2 + packed_lane
                if flat < TOTAL_VALUES:
                    normalized[flat] = normalized_pair[packed_lane]

        for pair in static_range(PACKED_PAIRS):
            biased_pair = add(
                pair_view(w_f32,pair,unused_high_if_odd=True),
                scalar_pair(WEIGHT_BIAS,pair,TOTAL_VALUES,undefined_high_if_odd=True),
                lanes=2,
            )
            # instruction_selection: add.f32x2; extent: PACKED_PAIRS issues
            # for both bias 0.0 and bias 1.0 before any second multiply
            for packed_lane in static_range(2):
                flat = pair * 2 + packed_lane
                if flat < TOTAL_VALUES:
                    biased_w[flat] = biased_pair[packed_lane]

        for pair in static_range(PACKED_PAIRS):
            y_pair = mul(
                pair_view(normalized,pair,unused_high_if_odd=True),
                pair_view(biased_w,pair,unused_high_if_odd=True), lanes=2)
            # instruction_selection: mul.f32x2; extent: PACKED_PAIRS issues
            # after every bias add and before any narrowing
            for packed_lane in static_range(2):
                flat = pair * 2 + packed_lane
                if flat < TOTAL_VALUES:
                    y_f32[flat] = y_pair[packed_lane]

    y_frag = reg_tile(DTYPE, (VEC,VEC_BLOCKS), stride=(1,VEC))
    if not PACKED_NARROW:
        for flat in static_range(TOTAL_VALUES):
            y_frag[flat] = cast(DTYPE, y_f32[flat], rounding="rn")
            # instruction_selection: cvt.rn.{f16,bf16}.f32; extent:
            # TOTAL_VALUES scalar issues before any store
    else:
        for pair in static_range(PACKED_PAIRS):
            narrowed_pair = cast(
                DTYPE + "x2", pair_view(y_f32,pair,unused_high_if_odd=True),
                rounding="rn")
            # instruction_selection: cvt.rn.{f16,bf16}x2.f32; extent:
            # PACKED_PAIRS issues before any store
            for packed_lane in static_range(2):
                flat = pair * 2 + packed_lane
                if flat < TOTAL_VALUES:
                    y_frag[flat] = narrowed_pair[packed_lane]

    for vb in static_range(VEC_BLOCKS):
        local_col = (lane_in_row + vb * TPR) * VEC
        col_valid = local_col < H
        y_offset = batch_idx * y_batch_stride \
                 + head_idx * y_head_stride + cast_i64(local_col)
        if row_valid and col_valid:
            copy_r2g(y_frag[:,vb], address_3d(y_storage,y_offset), COPY_BITS)
            # instruction_selection: predicated st.global using output_family():
            # VEC1 E, VEC8 P, VEC2/4 P iff VB_POW2 else E; extent:
            # VEC_BLOCKS row-and-column-predicated issues. Scalar-narrowed
            # VEC8 values are first re-packed with mov.b32 for the P store.

    if ENABLE_PDL:
        pdl_signal()
        # instruction_selection: griddepcontrol.launch_dependents; extent:
        # one issue per CTA after all stores
```

## Addressing, predicates, and tails

- `runtime_M=B*N`, flattened row bounds, quotient/remainder, and all four
  batch/head stride products remain Int64. X and Y are one-dimensional backing
  buffers; no dynamic-stride 3-D `match_buffer` binding is used.
- The source `predicate_k` tests the first column of each vector block against
  H. Because H and all row starts are divisible by VEC, a true predicate makes
  the complete vector legal.
- The row predicate gates X issue and final Y store. For an in-range async row,
  a false column predicate produces zero source bytes rather than skipping the
  async instruction. W loads are independent of row validity.
- Out-of-range rows still execute commit/wait, W loads, reduction, both CTA
  barriers, and PDL signal so no synchronization depends on a divergent row
  branch.
- X and Y strides are independent. Compact, source batch-gap, independently
  padded head strides, and an unused `2**31` batch stride all use this one ABI.

## Storage ownership and lifetimes

| storage | owner | lifetime / reuse rule |
| --- | --- | --- |
| `sX` | each row-thread writes and reads its own `(row,local_col)` vectors | async issue through the post-reduction reload |
| reduction buffer | lane 0 of each row warp publishes one scalar | publication barrier through final redundant reduction |
| X fragment | one thread | sync: load through epilogue; async: first load through square then reload after barrier |
| W fragment | one thread | global load before async wait through the complete epilogue |
| FP32 epilogue fragments | one thread | full first-multiply phase, full bias phase, full second-multiply phase, full narrow phase, then stores |
| output fragment | one thread | complete narrowing through predicated vector stores |

Each row group owns one `(batch,head)` row, each vector has one thread owner,
and all global output regions are disjoint under the source stride preconditions.

## Module and verification contract

- Registry name is `flashinfer_qk_rmsnorm`, category `flashinfer`, compute
  capability 10; `get_kernel` is the only PrimFunc factory.
- `CONFIGS` contains exactly 400 unique executable configurations and
  `BENCH_CONFIGS` contains exactly the three fixed upstream 3-D workloads.
- Every main-matrix config checks both implicit FlashInfer output and a
  caller-provided output without duplicating compilation labels.
- Correctness uses the public API with a proven `qk_rmsnorm_cute` dispatch and
  an independent FP32 oracle at `rtol=atol=1e-3`; it checks finite output,
  immutable inputs/weight, output identity, strided padding/guards, and sampled
  Int64 overflow rows.
- Static verification rejects `TilePrimitiveCall`, `tirx.tile.*`, tile imports,
  and tile primitives in every generated PrimFunc.
- Performance is accepted only from a complete three-workload bench-suite run
  in which each `flashinfer_cutedsl_time/tirx_time` ratio is strictly above
  0.99.

## Instruction selection is a lowering consequence

The port preserves the source TV layout, vector width, copy branch, fragment
order, shuffle topology, reduction-buffer layout, barrier placement, async-X
reload, and PDL boundaries before considering tuning. Plain TIRx and local PTX
may express reviewed instructions; they may not scalarize vector traffic,
change row ownership, replace the shared multi-warp reduction, keep async X
live instead of reloading it, reassociate the epilogue, move any store before
the full-fragment narrow phase, replace `rsqrt.approx.ftz.f32`, or emit PDL
instructions in a disabled specialization.
