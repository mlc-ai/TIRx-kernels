<!--
Copyright (c) 2025 by FlashInfer team.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: Apache-2.0

This execution sketch documents a TIRx port of FlashInfer's CuTe-DSL
LayerNormKernel. See LICENSE, NOTICE, and licenses/ for applicable terms.
-->

# FlashInfer LayerNorm SM100: execution sketch

This file is a non-executable execution sketch. It freezes the implementation
shape of FlashInfer's CuTe-DSL `LayerNormKernel` for the paired target
[`tirx_kernels/flashinfer/norm/layernorm.py`](../../../../tirx_kernels/flashinfer/norm/layernorm.py).
The target may become executable only after this sketch passes independent
source/PTX review; after the first PASS this file is permanently frozen.

The source is fixed at FlashInfer commit
`f2e04400e330fb2debe0bf8730d9424a1d37927f`:

- `flashinfer/norm/kernels/layernorm.py`, SHA256
  `123dd99aa37202ee499654e80c39c2edef5517b5b8b1e46c70885619c885fff0`;
- reduction, predicate, vector, and layout helpers in
  `flashinfer/norm/utils.py`, SHA256
  `3f44ac6727c58883420068bf0aa5b239b12d2e86819ad80e54bc1bc016ec881a`;
- public dispatch in `flashinfer/norm/__init__.py`, SHA256
  `226a88f5fb14e78e06e1be79020edcae01bfa9e53e677bd485373ea4d51cffcb`.

The port covers the source's public two-dimensional BF16 input/output and
FP32 gamma/beta specialization, independent Int64 input/output row strides,
runtime FP32 epsilon, vector widths 1/2/4/8, one- and multi-warp rows, one or
more vector blocks per thread, column tails, and the internal PDL off/on
branch. FP16 template instantiations, non-FP32 affine parameters, legacy CUDA,
RMSNorm, quantization, clusters, asynchronous copies, and every tile primitive
are out of scope.

## Source dispatch and resource formulae

These compile-time formulae are source policy rather than tuning choices:

```python
ELEM_BITS = 16
MAX_VEC = 128 // ELEM_BITS                             # 8 BF16 values

def source_vec(H):
    for candidate in (8, 4, 2, 1):
        if H % candidate == 0 and H // candidate >= 32:
            return candidate
    return gcd(8, H)

def source_config(H):
    VEC = source_vec(H)
    TPR = min(1024, max(32, pow2_ceil(ceil_div(H, VEC))))
    WARPS = max(TPR // 32, 1)
    VEC_BLOCKS = max(1, ceil_div(H // VEC, TPR))
    COLS = VEC * VEC_BLOCKS * TPR
    SMEM_BYTES = 2 * WARPS * 4
    TOTAL_VALUES = VEC * VEC_BLOCKS
    # Fresh PTX selects the mixed BF16 accumulator chain for a complete
    # physical row tile or a >8-value fragment. Predicated short fragments are
    # first widened and use ordinary f32 adds.
    MIXED_LOCAL_SUM = TOTAL_VALUES > 1 and (COLS == H or TOTAL_VALUES > 8)
    return VEC, TPR, WARPS, VEC_BLOCKS, COLS, SMEM_BYTES, MIXED_LOCAL_SUM
```

The compiled fake-tensor ABI always retains independent symbolic Int64 X and Y
row strides, including compact callers. Both row strides are divisible by
`VEC`; X/Y bases and compact FP32 gamma/beta bases have 16-byte assumed
alignment. There is no separate compact PrimFunc and no Int32 flat-offset
shortcut.

Reviewed anchor specializations are:

| `H` | `VEC` | `TPR` | `WARPS` | `VEC_BLOCKS` | `COLS` | relevant branch |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 66 | 2 | 64 | 2 | 1 | 128 | vector tail, multi-warp, PDL |
| 128 | 4 | 32 | 1 | 1 | 128 | full vector, one warp |
| 129 | 1 | 256 | 8 | 1 | scalar tail, multi-warp |
| 500 | 4 | 128 | 4 | 1 | vector tail, independent strides |
| 1024 | 8 | 128 | 4 | 1 | full 128-bit vector |
| 12288 | 8 | 1024 | 32 | 2 | two vector blocks plus tail |
| 16384 | 8 | 1024 | 32 | 2 | two full vector blocks |

## Pipeline at a glance

There is no producer/consumer pipeline. Every CTA is one row, and every thread
performs all three source phases in order.

| CTA/lane role | source-owned work | publication or reuse edge |
| --- | --- | --- |
| every thread | synchronously load its disjoint BF16 column vectors into logically zero-initialized registers | X feeds the first reduction and difference construction; the resulting FP32 difference stays live through the second reduction and affine epilogue |
| every thread | ordered local BF16 sum, then warp butterfly sum | one scalar partial per warp |
| lane 0 of each warp when `WARPS>1` | publish first-pass partial to `sum_smem[warp]` | CTA barrier publishes all warp sums |
| lanes `<WARPS` when `WARPS>1` | load one first-pass partial and full-warp reduce | replicated row sum feeds mean |
| every thread | subtract mean, square, and explicitly zero invalid physical fragment values | masked difference-square stays private until variance reduction |
| lane 0 / lanes `<WARPS` when `WARPS>1` | repeat publication/load using distinct `var_smem` | second CTA barrier publishes variance partials |
| every thread | form variance and approximate reciprocal standard deviation, then execute the source's explicit CTA barrier | barrier separates both reductions from affine loads/stores |
| every thread | scalar-load its FP32 gamma/beta values, apply packed/scalar affine math, narrow, and store its BF16 vectors | each valid output element has one owner |

The only publications are the two multi-warp reductions. The two shared
buffers never alias. PDL wait precedes all data traffic and PDL signal follows
all stores; disabled specializations contain neither instruction nor launch
attribute.

## Primitive vocabulary

Structural operations do not move or compute values:

```python
specialize(...)       # compile-time H and PDL family member
launch(...)           # exact grid, block, dynamic shared, and PDL attribute
raw_shared(...)       # dynamic shared allocation
view(...)             # typed linear view at a byte offset
reg_tile(...)         # thread-private fragment
address(...)          # logical tensor element under an Int64 row stride
pair_view(...)        # adjacent f32 registers consumed as one f32x2 value
```

Data movement, arithmetic, and synchronization remain primitive:

```python
fill(dst, value)
copy_g2r(src, dst, bits, predicate=None)
copy_r2s(src, dst)
copy_s2r(src, dst, predicate=None)
copy_r2g(src, dst, bits, predicate=None)
cast(dtype, src, rounding=None)
add(lhs, rhs, lanes=1, mixed_bf16=False)
sub(lhs, rhs, lanes=1)
mul(lhs, rhs, lanes=1)
fma(lhs, rhs, acc, lanes=1)
div(lhs, rhs)
rsqrt_fast(src)
shuffle_xor(src, delta)
cta_barrier()
pdl_wait()
pdl_signal()
```

One vector copy is one reviewed instruction. Fresh SM100a line-info exports
select:

| `VEC` | X load | Y store | narrowing |
| ---: | --- | --- | --- |
| 1 | `ld.global.b16` | `st.global.b16` | `cvt.rn.bf16.f32` |
| 2 | `ld.global.v2.b16` | `st.global.b32` | `cvt.rn.bf16x2.f32` |
| 4 | `ld.global.v4.b16` | `st.global.v2.b32` | two `cvt.rn.bf16x2.f32` |
| 8 | `ld.global.v4.b32` | `st.global.v4.b32` | four `cvt.rn.bf16x2.f32` |

Every gamma/beta element is a scalar `ld.global.b32`, including full-vector
profiles. Packed difference, square, normalize, scale, and bias operations use
the reviewed `*.f32x2` forms. No op below hides a row reduction, normalization,
affine epilogue, tile copy, or tile computation.

## Complete sketch

```python
@specialize(
    INPUT_DTYPE="bf16",
    AFFINE_DTYPE="f32",
    H="positive compile-time integer",
    ENABLE_PDL=(False, True),
    TARGET="sm_100a",
)
@launch(
    grid=(runtime_M, 1, 1),
    block=(TPR, 1, 1),
    dynamic_smem_bytes=SMEM_BYTES,
    use_programmatic_dependent_launch=ENABLE_PDL,
)
def flashinfer_layernorm(
    out_storage,                  # BF16, row stride y_row_stride
    input_storage,                # BF16, row stride x_row_stride
    gamma,                        # compact FP32 [H]
    beta,                         # compact FP32 [H]
    runtime_M: i64,
    runtime_eps: f32,
    y_row_stride: i64,            # elements, divisible by VEC
    x_row_stride: i64,            # elements, independently divisible by VEC
):
    tid = thread_id_x(TPR)
    row = cta_id_x(runtime_M)
    lane = tid % 32
    warp = tid // 32

    smem = raw_shared("u8", SMEM_BYTES, alignment=1024)
    sum_smem = view(smem, "f32", (WARPS,), byte_offset=0, alignment=4)
    var_smem = view(smem, "f32", (WARPS,),
                    byte_offset=WARPS*4, alignment=4)
    # For WARPS>1, fresh PTX materializes and broadcasts the dynamic-shared
    # base in the mechanical prologue before the PDL wait.
    # instruction_selection: shfl.sync.idx.b32; extent: one shared-base
    # broadcast per CTA thread. One-warp profiles eliminate both shared views.

    if ENABLE_PDL:
        pdl_wait()
        # instruction_selection: griddepcontrol.wait; extent: one issue per CTA,
        # after shared-address setup but before every global-memory/data operation

    TOTAL_VALUES = VEC * VEC_BLOCKS
    x_bits = reg_tile("bf16", (TOTAL_VALUES,))
    x_f32 = reg_tile("f32", (TOTAL_VALUES,))
    fill(x_bits, 0)
    # instruction_selection: source-logical fill; full-column profiles DCE it
    # completely, while a possibly partial vector block materializes only its
    # zero seed as mov.b32/mov.b64 before the predicated global load

    # Source TV layout: each thread owns VEC consecutive columns in each block;
    # successive vector blocks are VEC*TPR columns apart.
    for vb in static_range(VEC_BLOCKS):
        col = (tid + vb * TPR) * VEC
        col_valid = col < H
        x_offset = cast_i64(row) * x_row_stride + cast_i64(col)
        if col_valid:
            copy_g2r(address(input_storage,x_offset),
                     x_bits[vb*VEC:(vb+1)*VEC], 16*VEC)
            # instruction_selection: exact X load from the VEC table; extent:
            # one first-column-predicated vector per vector block

    for value in static_range(TOTAL_VALUES):
        x_f32[value] = cast("f32", x_bits[value])
        # instruction_selection: cvt.f32.bf16; extent: one physical value

    # The source reduction starts from the first widened value. Fresh BF16 PTX
    # folds later conversions into mixed-input adds for complete physical row
    # tiles and >8-value fragments. Predicated short fragments (reviewed H66,
    # H320, H500, and H768) keep ordinary f32 adds instead.
    local_sum = add(x_f32[0], 0.0)
    # instruction_selection: add.f32; extent: one initial accumulator issue
    for value in static_range(1, TOTAL_VALUES):
        if MIXED_LOCAL_SUM:
            local_sum = add(local_sum, x_bits[value], mixed_bf16=True)
            # instruction_selection: add.rn.f32.bf16; extent: one ordered issue
            # per remaining physical BF16 value
        else:
            local_sum = add(local_sum, x_f32[value])
            # instruction_selection: add.f32; extent: one ordered issue per
            # remaining widened value in a predicated short fragment

    for delta in (1, 2, 4, 8, 16):
        peer = shuffle_xor(local_sum, delta)
        # instruction_selection: shfl.sync.bfly.b32, clamp 31, full mask;
        # extent: one scalar issue per stage
        local_sum = add(local_sum, peer)
        # instruction_selection: add.f32; extent: one scalar issue per stage
    warp_sum = local_sum

    if WARPS > 1:
        if lane == 0:
            copy_r2s(warp_sum, sum_smem[warp])
            # instruction_selection: st.shared.b32; extent: one scalar per warp
        cta_barrier()
        # instruction_selection: bar.sync 0; extent: first publication edge
        block_sum = 0.0
        if lane < WARPS:
            block_sum = copy_s2r(sum_smem[lane])
            # instruction_selection: ld.shared.b32; extent: one participating
            # lane in every warp
        for delta in (1, 2, 4, 8, 16):
            peer = shuffle_xor(block_sum, delta)
            # instruction_selection: shfl.sync.bfly.b32, clamp 31, full mask;
            # extent: one scalar issue per stage
            block_sum = add(block_sum, peer)
            # instruction_selection: add.f32; extent: one scalar issue per stage
        sum_x = block_sum
    else:
        sum_x = warp_sum

    # Preserve the source/compiler split. Power-of-two H uses a reciprocal
    # multiply. General H with a packed fragment keeps positive mean and packed
    # subtraction; the scalar-fragment branch folds the sign into the divisor
    # so its subsequent difference is an add.
    if is_power_of_two(H):
        mean = mul(sum_x, exact_float32_reciprocal(H))
        # instruction_selection: mul.f32; extent: one scalar per thread
    elif TOTAL_VALUES > 1:
        mean = div(sum_x, float32(H))
        # instruction_selection: div.rn.f32; extent: one scalar per thread
    else:
        negative_mean = div(sum_x, float32(-H))
        # instruction_selection: div.rn.f32; extent: one scalar per thread

    diff = reg_tile("f32", (TOTAL_VALUES,))
    diff_sq = reg_tile("f32", (TOTAL_VALUES,))
    if TOTAL_VALUES == 1:
        if is_power_of_two(H):
            diff[0] = sub(x_f32[0], mean)
            # instruction_selection: sub.f32; extent: one scalar value
        else:
            diff[0] = add(x_f32[0], negative_mean)
            # instruction_selection: add.f32; extent: one scalar value
        diff_sq[0] = mul(diff[0], diff[0])
        # instruction_selection: mul.f32; extent: one scalar value
    else:
        for pair in static_range(ceil_div(TOTAL_VALUES,2)):
            diff_pair = sub(pair_view(x_f32,pair), pair(mean,mean), lanes=2)
            # instruction_selection: sub.f32x2; extent: one packed pair
            diff.store_pair(pair, diff_pair)
        for pair in static_range(ceil_div(TOTAL_VALUES,2)):
            square_pair = mul(pair_view(diff,pair), pair_view(diff,pair), lanes=2)
            # instruction_selection: mul.f32x2; extent: one packed pair
            diff_sq.store_pair(pair, square_pair)

    # A zero-filled invalid X lane has diff=-mean, so source semantics require
    # this second mask before the variance reduction.
    for vb in static_range(VEC_BLOCKS):
        vector_col = (tid + vb * TPR) * VEC
        if vector_col >= H:
            for e in static_range(VEC):
                diff_sq[vb*VEC+e] = 0.0
            # instruction_selection: one CSE'd setp.gt.u32 per possibly partial
            # vector block, then scalar selp.f32 for VEC=1 or shared-predicate
            # selp.b64/mov sequences for VEC=2/4/8; extent: the complete
            # invalid physical vector, not one independent predicate per value

    local_var = add(diff_sq[0], 0.0)
    # instruction_selection: add.f32; extent: one initial accumulator issue
    for value in static_range(1, TOTAL_VALUES):
        local_var = add(local_var, diff_sq[value])
        # instruction_selection: add.f32; extent: one ordered issue per value

    for delta in (1, 2, 4, 8, 16):
        peer = shuffle_xor(local_var, delta)
        # instruction_selection: shfl.sync.bfly.b32, clamp 31, full mask;
        # extent: one scalar issue per stage
        local_var = add(local_var, peer)
        # instruction_selection: add.f32; extent: one scalar issue per stage
    warp_var = local_var

    if WARPS > 1:
        if lane == 0:
            copy_r2s(warp_var, var_smem[warp])
            # instruction_selection: st.shared.b32; extent: one scalar per warp
        cta_barrier()
        # instruction_selection: bar.sync 0; extent: second publication edge
        block_var = 0.0
        if lane < WARPS:
            block_var = copy_s2r(var_smem[lane])
            # instruction_selection: ld.shared.b32; extent: one participating
            # lane in every warp
        for delta in (1, 2, 4, 8, 16):
            peer = shuffle_xor(block_var, delta)
            # instruction_selection: shfl.sync.bfly.b32, clamp 31, full mask;
            # extent: one scalar issue per stage
            block_var = add(block_var, peer)
            # instruction_selection: add.f32; extent: one scalar issue per stage
        sum_diff_sq = block_var
    else:
        sum_diff_sq = warp_var

    if is_power_of_two(H):
        shifted_var = fma(sum_diff_sq, exact_float32_reciprocal(H), runtime_eps)
        # instruction_selection: fma.rn.f32; extent: one scalar per thread
    else:
        variance = div(sum_diff_sq, float32(H))
        # instruction_selection: div.rn.f32; extent: one scalar per thread
        shifted_var = add(variance, runtime_eps)
        # instruction_selection: add.f32; extent: one scalar per thread
    rstd = rsqrt_fast(shifted_var)
    # instruction_selection: rsqrt.approx.ftz.f32; extent: one scalar per thread

    cta_barrier()
    # instruction_selection: bar.sync 0; extent: one source-ordered edge after
    # the reciprocal standard deviation and before affine loads

    gamma_frag = reg_tile("f32", (TOTAL_VALUES,))
    beta_frag = reg_tile("f32", (TOTAL_VALUES,))
    fill(gamma_frag, 0.0)
    fill(beta_frag, 0.0)
    # instruction_selection: source-logical fills; full-column profiles DCE
    # both fills, while possibly partial vector blocks materialize only the
    # required mov.b32/mov.b64 zero seeds before predicated scalar loads
    for vb in static_range(VEC_BLOCKS):
        for e in static_range(VEC):
            value = vb * VEC + e
            col = tid * VEC + vb * TPR * VEC + e
            if col < H:
                gamma_frag[value] = copy_g2r(gamma[col])
                # instruction_selection: ld.global.b32; extent: one scalar f32
                beta_frag[value] = copy_g2r(beta[col])
                # instruction_selection: ld.global.b32; extent: one scalar f32

    y_f32 = reg_tile("f32", (TOTAL_VALUES,))
    if TOTAL_VALUES == 1:
        normalized = mul(diff[0], rstd)
        # instruction_selection: mul.f32; extent: one scalar value
        y_f32[0] = fma(normalized, gamma_frag[0], beta_frag[0])
        # instruction_selection: fma.rn.f32; extent: one scalar value
    else:
        normalized = reg_tile("f32", (TOTAL_VALUES,))
        scaled = reg_tile("f32", (TOTAL_VALUES,))
        for pair in static_range(ceil_div(TOTAL_VALUES,2)):
            normalized_pair = mul(pair_view(diff,pair), pair(rstd,rstd), lanes=2)
            # instruction_selection: mul.f32x2; extent: one packed pair
            normalized.store_pair(pair, normalized_pair)
        for pair in static_range(ceil_div(TOTAL_VALUES,2)):
            scaled_pair = mul(pair_view(normalized,pair),
                              pair_view(gamma_frag,pair), lanes=2)
            # instruction_selection: mul.f32x2; extent: one packed pair
            scaled.store_pair(pair, scaled_pair)
        for pair in static_range(ceil_div(TOTAL_VALUES,2)):
            y_pair = add(pair_view(scaled,pair), pair_view(beta_frag,pair), lanes=2)
            # instruction_selection: add.f32x2; extent: one packed pair
            y_f32.store_pair(pair, y_pair)

    y_bits = reg_tile("bf16", (TOTAL_VALUES,))
    if TOTAL_VALUES == 1:
        y_bits[0] = cast("bf16", y_f32[0], rounding="rn")
        # instruction_selection: cvt.rn.bf16.f32; extent: one scalar value
    else:
        for pair in static_range(ceil_div(TOTAL_VALUES,2)):
            y_bits.store_pair(pair,
                cast("bf16x2",
                     high=y_f32[pair*2+1], low=y_f32[pair*2], rounding="rn"))
            # instruction_selection: cvt.rn.bf16x2.f32 with PTX operands
            # high-address value then low-address value; extent: one packed
            # pair whose low 16 bits belong at the lower output address

    for vb in static_range(VEC_BLOCKS):
        col = (tid + vb * TPR) * VEC
        col_valid = col < H
        y_offset = cast_i64(row) * y_row_stride + cast_i64(col)
        if col_valid:
            copy_r2g(y_bits[vb*VEC:(vb+1)*VEC],
                     address(out_storage,y_offset), 16*VEC)
            # instruction_selection: exact Y store from the VEC table; extent:
            # one first-column-predicated vector per vector block

    if ENABLE_PDL:
        pdl_signal()
        # instruction_selection: griddepcontrol.launch_dependents; extent:
        # one issue per CTA after all output stores
```

## Addressing, predicates, and tails

- Grid extent equals runtime M, so no row predicate exists. `row` is widened
  before either row-stride multiplication; both element offsets remain Int64.
- The source TV layout owns `VEC` adjacent values and strides successive vector
  blocks by `VEC*TPR`. All supported vector widths divide H, so testing the
  first column proves an entire in-range vector.
- X invalid physical values remain zero because the fragment is filled before
  predicated loads. They participate harmlessly in the first sum.
- Invalid difference squares must be zeroed explicitly before the second sum.
  This is a distinct predicate and may not be inferred from the X fill.
- Gamma/beta loads are scalar and individually predicated. Output reuses the
  vector first-column predicate and never touches row padding or the trailing
  backing-store guard.
- Every fragment loop is a compile-time unrolled loop in reviewed required
  profiles, including the 16-value-per-thread H12288/H16384 branches. No
  runtime or source-size loop is introduced.

## Storage ownership and lifetimes

| storage | owner | lifetime / reuse rule |
| --- | --- | --- |
| X/Y/gamma/beta GMEM | caller | X/gamma/beta read-only; Y is the only mutable tensor |
| `sum_smem[WARPS]` | lane 0 writers, lanes `<WARPS` readers | first reduction publication through its full-warp completion |
| `var_smem[WARPS]` | lane 0 writers, lanes `<WARPS` readers | second reduction publication through its full-warp completion; never aliases sum buffer |
| BF16 X fragment | one thread | logical zero-fill/load through widening and the optional mixed-BF16 first-sum chain; dead after difference inputs are available |
| FP32 X fragment | one thread | widening through difference construction; the affine path consumes the retained difference instead |
| difference / square fragment | one thread | mean through masked second local sum; difference remains live for affine |
| gamma/beta fragments | one thread | post-reduction scalar loads through affine add |
| normalized/scaled fragments | one thread | three explicit packed affine phases; lifetimes end at consumption |
| BF16 Y fragment | one thread | narrowing through one predicated vector store per block |

## Module and verification contract

- Registry name is `flashinfer_layernorm`, category `flashinfer`, compute
  capability 10. Runtime ABI is
  `out,input,gamma,beta,M:int64,eps:f32,y_row_stride:int64,x_row_stride:int64`.
- The supported tensor domain is BF16 X/Y, FP32 gamma/beta, positive M/H,
  inner stride one, independent row strides divisible by the selected vector.
- `CONFIGS` includes the exact upstream 16-case matrix, trace/example cases,
  Int64 overflow regression, PDL/vec2 tail, independent-stride/full-ABI guard,
  input/parameter immutability, output padding, and finite-value checks.
- The complete required performance matrix is the upstream 16-case matrix
  `M in {1,2,3,128}` by `H in {128,129,1024,16384}`. Each row must satisfy
  `flashinfer_cutedsl_time / tirx_time > 0.99` in one valid reference-enabled
  `bench_suite` artifact; no other timer is a retention or PASS criterion.
- Every specialization undergoes low-level IR rejection of
  `TilePrimitiveCall`, `tirx.tile.*`, tile primitives, `T.cuda.func_call`, and
  `cuda_func_call`.

## Instruction selection is a lowering consequence

The port preserves source vector selection, one-CTA-per-row topology, symbolic
Int64 strides, zero-fill and both column predicates, mixed BF16 local summation,
two non-aliasing shared reductions, power-of-two versus general-H arithmetic,
fast reciprocal square root, explicit post-reduction barrier, scalar affine
loads, packed affine phase order, BF16 store width, full compile-time unrolling,
and PDL boundaries. Plain TIRx and typed/local PTX may express reviewed
instructions; they may not use tile primitives, replace the two-pass
difference-of-squares with `E[x^2]-mean^2`, reuse one shared buffer, hoist
gamma/beta loads before the source barrier, vectorize those scalar FP32 loads,
contract packed multiply/add into a different arithmetic order, or weaken the
Int64 addressing and tail predicates.
