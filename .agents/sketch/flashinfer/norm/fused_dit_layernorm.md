<!--
Copyright (c) 2026 by FlashInfer team.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: Apache-2.0

This execution sketch documents a TIRx port of FlashInfer's CUDA
meta_fused_layernorm kernel. See LICENSE, NOTICE, and licenses/ for applicable
terms.
-->

# FlashInfer fused DIT LayerNorm SM100: execution sketch

This file is a non-executable execution sketch. It freezes the implementation
shape of FlashInfer's CUDA `meta_fused_layernorm` for
[`tirx_kernels/flashinfer/norm/fused_dit_layernorm.py`](../../../../tirx_kernels/flashinfer/norm/fused_dit_layernorm.py).
The target may become executable only after independent source/PTX review; this
file is permanently frozen after its first reviewer PASS.

The source is fixed at FlashInfer commit
`f2e04400e330fb2debe0bf8730d9424a1d37927f`:

- `include/flashinfer/norm/fused_dit_layernorm.cuh`, SHA256
  `f8d8af27f6715e282858547a3e8185a2cc71eaaad00d70923e9f02aad076e2f4`;
- host dispatch in `csrc/norm.cu`, SHA256
  `c4f86ef2046fdd13a5cd79f7ffc93c7522d228ffb53f3b2c8a77d29a3dc20e0f`;
- public wrappers, tests, and benchmark copied under
  `.porting/flashinfer_fused_dit_layernorm/source_export/source_artifacts/`.

The writer-owned fresh export used the normal FlashInfer JIT path for
`sm_100a`, `-O3 -DNDEBUG -lineinfo`, with no debug flags. Its PTX SHA256 is
`549696308d8c16dad94ab13faed07e3e5708044e531b84ff7e739dcf3ac69607`;
its compile-command SHA256 is
`8562bd99cda718dad977d8b13a0f486ba130b26f498b723d53dcefca7c8591bc`.
The export contains 54,062 `.loc` directives and all eighteen unique
`meta_fused_layernorm` entries. The reviewer must independently export the
same eighteen branches through the normal JIT path.

## Frozen specialization and launch

The complete family is the Cartesian product below. No runtime dispatch is
allowed inside one generated member except the batch loop and residual-present
branch.

```text
MODE = GRGB | RSS | GRSS
OUTPUT = BF16 | NVFP4 | MXFP8
USE_INPUT_SF_SCALE = false | true
HIDDEN_SIZE = 3072 BF16 values
BLOCK_SIZE = 384 threads = 12 warps
GRID = (runtime_num_rows, 1, 1)
DYNAMIC_SHARED = 0
CLUSTER = none
PDL = none
```

`blockIdx.x` is exactly the row within a batch. There is no row guard because
the grid exactly covers `runtime_num_rows`. Every CTA executes a real runtime
serial loop over `runtime_batch_size`; each of its 384 threads owns four
adjacent BF16x2 pairs, or eight hidden values. Therefore `384*8=3072` and no
column guard exists. Gate, scale, and shift always use the source WAN row
stride `6*3072`, even when the public tensor is presented as a 2-D view.

The mode traits are fixed:

| mode | residual expression | LayerNorm epilogue |
| --- | --- | --- |
| GRGB | `input * (gate + gate_bias) + residual` | `centered * (rstd * gamma) + beta` |
| RSS | `input + residual`, with residual optionally zero | normalized, then `(scale+scale_bias+1)` and `(shift+shift_bias)` |
| GRSS | GRGB residual expression | RSS scale/shift epilogue |

When enabled, input global scale multiplies the gate expression in GRGB/GRSS
and the input directly in RSS before the residual FMA. The scale is loaded once
per thread before all bias fragments and before the batch loop. The residual
presence test is runtime-uniform and remains inside every batch iteration. A
legal dummy pointer represents CUDA null only at the TIRx ABI boundary; it does
not remove this branch.

## Fixed device ABI

All specializations have the same nineteen arguments. Unused pointers receive
valid dummy allocations. Mode, output format, and input-scale presence are
compile-time configuration, while batch, rows, epsilon, and residual presence
remain runtime values.

| position | argument | device dtype / role |
| ---: | --- | --- |
| 0 | `input_ptr` | read-only BF16 compact input |
| 1 | `residual_ptr` | read-only BF16 compact residual or legal dummy |
| 2 | `gate_ptr` | read-only BF16 stride-6 auxiliary or dummy |
| 3 | `gate_bias_ptr` | read-only FP32 `[3072]` or dummy |
| 4 | `gamma_ptr` | read-only FP32 `[3072]` or dummy |
| 5 | `beta_ptr` | read-only FP32 `[3072]` or dummy |
| 6 | `scale_ptr` | read-only BF16 stride-6 auxiliary or dummy |
| 7 | `scale_bias_ptr` | read-only FP32 `[3072]` or dummy |
| 8 | `shift_ptr` | read-only BF16 stride-6 auxiliary or dummy |
| 9 | `shift_bias_ptr` | read-only FP32 `[3072]` or dummy |
| 10 | `residual_output_ptr` | mutable BF16 compact residual output |
| 11 | `norm_output_ptr` | mutable BF16, packed E2M1, or packed E4M3 output |
| 12 | `sf_output_ptr` | mutable swizzled uint8 SF storage or dummy |
| 13 | `output_sf_scale_ptr` | read-only FP32 NVFP4 global scale or dummy |
| 14 | `input_sf_scale_ptr` | read-only FP32 input global scale or dummy |
| 15 | `runtime_batch_size` | uniform int32 batch-loop extent |
| 16 | `runtime_num_rows` | uniform int32 row extent and grid-x extent |
| 17 | `runtime_epsilon` | uniform FP32 epsilon |
| 18 | `runtime_has_residual` | uniform int32 CUDA-null semantic |

Every FP32 bias buffer used by its specialization is non-null in the supported
production and official-test domain. Although the public API types these biases
as optional, the source device kernel unconditionally dereferences each used
bias; this port does not invent a replacement algorithm for that upstream
inconsistency.

## Storage, ownership, and lifetime

The kernel uses exactly 136 bytes of static shared storage in three separately
aligned objects and no dynamic shared memory:

1. `reduceStore`: 120 bytes, alignment 8, CUB
   `BlockReduce<float2,384>::TempStorage`;
2. `s_mean`: 8 bytes, alignment 8, one duplicated FP32 pair;
3. `s_inv_std`: 8 bytes, alignment 8, one duplicated FP32 pair.

Within `reduceStore`, lane zero of warp `w` writes its sum/sumsq pair to byte
offset `16+8*w`, for `w=0..11`. Bytes 0..15 and 112..119 remain unused but are
part of the source CUB ABI. After the first CTA barrier, thread zero keeps its
live warp-zero result and serially loads the eleven pairs at byte offsets
24,32,...,104. `s_mean` and `s_inv_std` are written only by thread zero, then
published by the second CTA barrier and loaded by every thread.

The optional input scale and every used gamma/beta/bias 32-byte fragment are
register-resident across all runtime batch iterations. Input, residual, gate,
scale, and shift packed fragments are reloaded inside each iteration. The
unrounded FP32 residual fragment survives from its residual expression through
sum/sumsq, centering, and epilogue. The BF16 residual fragment exists only for
the early residual output store. The final BF16 fragment exists before all
three output-format branches; both quantizers therefore consume BF16-rounded,
not the preceding FP32, values.

The mutable outputs are independent: the early BF16 residual store precedes
the reduction; the norm payload and quant SF stores occur only after the
epilogue. All input tensors, auxiliary backing allocations, FP32 parameters,
and global scales are read-only for the complete launch.

## Primitive vocabulary

Structural notation does not move values or perform arithmetic:

```text
specialize(...)  launch(...)  reg_pair4(...)  shared_object(...)
address(...)     wan_stride6_address(...)     sf_128x4_address(...)
```

All execution work stays in primitive TIRx/PTX operations:

```text
load_global_v4_b32  load_global_v4_b64  load_global_b32
store_global_v4_b32 store_global_b32    store_global_b64 store_global_b8
store_generic_v4_b32 store_generic_b32  store_generic_b64
load_shared_b64     store_shared_v2_b32
bf16x2_to_f32x2     f32x2_to_bf16x2_rn
bf16_to_f32          max_f32_ftz
neg_f32_ftz          compare_gt_u32      compare_eq_ftz_f32
add_f32_ftz_rn      mul_f32_ftz         fma_f32_ftz_rn
add_f32x2_ftz_rn    mul_f32x2_ftz_rn    fma_f32x2_ftz_rn
abs_bf16x2          max_bf16x2          max_bf16
shuffle_down_b32    shuffle_xor_b32
cvt_e4m3x2_f32_rn_satfinite  cvt_f16x2_e4m3x2_rn  cvt_f32_f16
cvt_ue8m0x2_f32_rp_satfinite cvt_bf16x2_ue8m0x2_rn
cvt_e2m1x2_f32_rn_satfinite  mov_b32_bytes
rcp_approx_ftz      rsqrt_approx_ftz
and_b16 and_b32 or_b32 add_s32 sub_s32 shift_left_b32 shift_right_s32
cvt_u16_u32 cvt_u64_u16 shift_left_b64 or_b64 branch
reinterpret_b32 reinterpret_f32 low_byte low_half low_bf16_bits
cta_barrier
```

There is no LayerNorm, block-reduce, quantize, tile-copy, tile-compute, or
other compound primitive. There is no tile primitive and no external FuncCall
escape. The exact CUB reduction and accurate-rsqrt expansion are written from
the primitive vocabulary below.

## Complete source-order execution sketch

```text
specialize(MODE, OUTPUT, USE_INPUT_SF_SCALE, HIDDEN_SIZE=3072)
launch(grid=(runtime_num_rows,1,1), block=(384,1,1), static_shared=136)
kernel(fixed nineteen-argument ABI):
    tid = threadIdx.x                         # 0..383
    lane = tid & 31                           # 0..31
    warp = tid >> 5                           # 0..11
    row = blockIdx.x                          # no row predicate
    pair_base = tid * 4                       # BF16x2 units

    shared reduceStore[120] align 8
    shared s_mean[8] align 8
    shared s_inv_std[8] align 8

    if USE_INPUT_SF_SCALE:
        input_sf = load_global_b32(input_sf_scale_ptr[0])
        # selection: ld.global.b32; extent: one scalar/thread before batch loop
    else:
        input_sf = 1.0

    if MODE == GRGB:
        gamma[0:4] = load_global_v4_b64(gamma_ptr + 8*tid)
        beta[0:4] = load_global_v4_b64(beta_ptr + 8*tid)
        # selection: ld.global.v4.b64 per parameter; extent: 32 B/thread
    if MODE in (GRGB,GRSS):
        gate_bias[0:4] = load_global_v4_b64(gate_bias_ptr + 8*tid)
        # selection: ld.global.v4.b64; extent: 32 B/thread
    if MODE in (RSS,GRSS):
        scale_bias[0:4] = load_global_v4_b64(scale_bias_ptr + 8*tid)
        shift_bias[0:4] = load_global_v4_b64(shift_bias_ptr + 8*tid)
        # selection: ld.global.v4.b64 per parameter; extent: 32 B/thread

    for batch in serial_runtime_range(runtime_batch_size):
        dense_pair = ((batch*runtime_num_rows + row)*1536 + pair_base)
        auxiliary_pair = ((batch*runtime_num_rows + row)*9216 + pair_base)

        input_bf16[0:4] = load_global_v4_b32(input_ptr + 2*dense_pair)
        input[0:4] = bf16x2_to_f32x2(input_bf16[0:4])
        # selection: ld.global.v4.b32, then eight scalar cvt.f32.bf16
        # extent: one 16 B load and eight scalar widens/thread/iteration

        if runtime_has_residual != 0:          # CTA-uniform branch
            residual_bf16[0:4] = load_global_v4_b32(
                residual_ptr + 2*dense_pair)
            residual[0:4] = bf16x2_to_f32x2(residual_bf16[0:4])
            # selection: ld.global.v4.b32 plus eight widens
        else:
            residual[0:4] = zero_f32x2
            # extent: four zero register pairs, no residual global load

        if MODE in (GRGB,GRSS):
            gate_bf16[0:4] = load_global_v4_b32(
                gate_ptr + 2*auxiliary_pair)
            gate[0:4] = bf16x2_to_f32x2(gate_bf16[0:4])
            # selection: ld.global.v4.b32 plus eight widens
            if USE_INPUT_SF_SCALE:
                gate[k] = add_f32x2_ftz_rn(gate[k],gate_bias[k])  # k=0..3
                gate[k] = mul_f32x2_ftz_rn(gate[k],input_sf)      # k=0..3
                input[k] = fma_f32x2_ftz_rn(input[k],gate[k],residual[k])
                # selection: add.rn.ftz.f32x2 x4, mul... x4, fma... x4
            else:
                input[k] = fma_f32x2_ftz_rn(
                    input[k],add_f32x2_ftz_rn(gate[k],gate_bias[k]),residual[k])
                # selection: add.rn.ftz.f32x2 x4, fma.rn.ftz.f32x2 x4
        else:                                  # RSS
            if USE_INPUT_SF_SCALE:
                input[k] = fma_f32x2_ftz_rn(input[k],input_sf,residual[k])
                # selection: fma.rn.ftz.f32x2; extent: four pairs
            else:
                input[k] = add_f32x2_ftz_rn(input[k],residual[k])
                # selection: add.rn.ftz.f32x2; extent: four pairs

        residual_out_bf16[k] = f32x2_to_bf16x2_rn(input[k])      # k=0..3
        store_global_v4_b32(residual_output_ptr + 2*dense_pair,
                            residual_out_bf16[0:4])
        # selection: cvt.rn.bf16x2.f32 x4, st.global.v4.b32 x1
        # The reduction below still consumes unrounded input[0:4].

        sum = 0.0
        sum_sq = 0.0
        for k in static_range(4):
            sum = add_f32_ftz_rn(add_f32_ftz_rn(sum,input[k].x),input[k].y)
            sum_sq = fma_f32_ftz_rn(
                input[k].y,input[k].y,
                fma_f32_ftz_rn(input[k].x,input[k].x,sum_sq))
        # selection: ordered scalar add.rn.ftz.f32 x8 and
        # fma.rn.ftz.f32 x8; extent: eight unrounded values/thread

        pair = (sum,sum_sq)
        for delta in (1,2,4,8,16):
            peer_x, valid_x = shuffle_down_b32(pair.x,delta,mask=0xffffffff)
            peer_y, valid_y = shuffle_down_b32(pair.y,delta,mask=0xffffffff)
            if valid_x and valid_y:
                pair = add_f32x2_ftz_rn(pair,(peer_x,peer_y))
            # selection: paired shfl.sync.down.b32 and one
            # add.rn.ftz.f32x2 for a valid source lane; extent: five stages

        if lane == 0:
            store_shared_v2_b32(reduceStore + 16 + 8*warp,pair)
            # selection: st.shared.v2.b32; extent: one per warp, twelve total
        cta_barrier()
        # selection: bar.sync 0; extent: exactly one CTA barrier here

        if tid == 0:
            block_pair = pair                 # live warp-zero result
            for byte_offset in (24,32,40,48,56,64,72,80,88,96,104):
                partial = load_shared_b64(reduceStore + byte_offset)
                block_pair = add_f32x2_ftz_rn(block_pair,partial)
            # selection: ld.shared.b64 x11 and serial
            # add.rn.ftz.f32x2 x11; only thread zero

            mean = mul_f32_ftz(block_pair.x,0x1.555556p-12)  # 1/3072
            mean_sq = mul_f32_ftz(block_pair.y,0x1.555556p-12)
            neg_mean = neg_f32_ftz(mean)
            variance = fma_f32_ftz_rn(neg_mean,mean,mean_sq)
            variance = max_f32_ftz(variance,0.0)
            variance_eps = add_f32_ftz_rn(variance,runtime_epsilon)
            # selection: mul.ftz x2, neg.ftz.f32 x1, fma.rn.ftz,
            # max.ftz,
            # add.rn.ftz; extent: one scalar chain/CTA/iteration

            bits = reinterpret_b32(variance_eps)
            normal_test = compare_gt_u32(
                add_s32(bits,-0x00800000),0x7effffff)
            # selection: add.s32 x1 and setp.gt.u32 x1
            if not normal_test:
                normalized_bits = or_b32(and_b32(bits,0x00ffffff),0x3f000000)
                exponent_adjust = sub_s32(normalized_bits,bits)
                seed = rsqrt_approx_ftz(reinterpret_f32(normalized_bits))
                seed_sq = mul_f32_ftz(seed,seed)
                neg_seed_sq = neg_f32_ftz(seed_sq)
                correction0 = fma_f32_ftz_rn(seed,seed,neg_seed_sq)
                neg_normalized = neg_f32_ftz(reinterpret_f32(normalized_bits))
                correction1 = fma_f32_ftz_rn(
                    seed_sq,neg_normalized,1.0)
                correction2 = fma_f32_ftz_rn(
                    correction0,neg_normalized,correction1)
                correction3 = fma_f32_ftz_rn(
                    correction2,0.375,0.5)
                correction4 = mul_f32_ftz(seed,correction2)
                refined = fma_f32_ftz_rn(correction3,correction4,seed)
                rstd_bits = add_s32(shift_right_s32(exponent_adjust,1),
                                    reinterpret_b32(refined))
                rstd = reinterpret_f32(rstd_bits)
                # exact __frsqrt_rn normal-range PTX expansion, including
                # neg.ftz.f32 x2 for seed_sq and normalized mantissa
            else:
                rstd = rsqrt_approx_ftz(variance_eps)
                # exact __frsqrt_rn exceptional-range fallback

            store_shared_v2_b32(s_mean,(mean,mean))
            store_shared_v2_b32(s_inv_std,(rstd,rstd))
            # selection: st.shared.v2.b32 x2; one thread

        cta_barrier()
        # selection: bar.sync 0; second and final barrier/iteration
        mean2 = load_shared_b64(s_mean)
        rstd2 = load_shared_b64(s_inv_std)
        # selection: ld.shared.b64 x2; extent: every thread

        for k in static_range(4):
            input[k] = fma_f32x2_ftz_rn((-1.0,-1.0),mean2,input[k])
        # selection: fma.rn.ftz.f32x2 x4

        if MODE == GRGB:
            for k in static_range(4):
                scaled_inv = mul_f32x2_ftz_rn(rstd2,gamma[k])
                input[k] = fma_f32x2_ftz_rn(input[k],scaled_inv,beta[k])
            # selection: mul.rn.ftz.f32x2 x4, fma.rn.ftz.f32x2 x4
        else:
            input[k] = mul_f32x2_ftz_rn(input[k],rstd2)           # k=0..3
            # selection: mul.rn.ftz.f32x2 x4

        if MODE in (RSS,GRSS):
            scale_bf16[0:4] = load_global_v4_b32(
                scale_ptr + 2*auxiliary_pair)
            scale[0:4] = bf16x2_to_f32x2(scale_bf16[0:4])
            # selection: ld.global.v4.b32, then eight scalar widens
            shift_bf16[0:4] = load_global_v4_b32(
                shift_ptr + 2*auxiliary_pair)
            shift[0:4] = bf16x2_to_f32x2(shift_bf16[0:4])
            # selection: ld.global.v4.b32, then eight scalar widens
            # source order is complete scale load/widen before shift load/widen
            for k in static_range(4):
                affine_scale = add_f32x2_ftz_rn(
                    (1.0,1.0),add_f32x2_ftz_rn(scale[k],scale_bias[k]))
                affine_shift = add_f32x2_ftz_rn(shift[k],shift_bias[k])
                input[k] = fma_f32x2_ftz_rn(
                    input[k],affine_scale,affine_shift)
            # selection: add.rn.ftz.f32x2 x12, fma... x4

        output_bf16[k] = f32x2_to_bf16x2_rn(input[k])            # k=0..3
        # selection: cvt.rn.bf16x2.f32 x4 before format dispatch

        if OUTPUT == BF16:
            store_generic_v4_b32(norm_output_ptr + 2*dense_pair,output_bf16)
            # selection: generic st.v4.b32; extent: 16 B/thread/iteration
        elif OUTPUT == NVFP4:
            execute the frozen NVFP4 tail below
        else:                                      # MXFP8
            execute the frozen MXFP8 tail below

    # Fresh PTX retains a backward branch to advance batch and repeats all
    # per-iteration loads, barriers, reduction, epilogue, and stores.
```

## Frozen NVFP4 tail

The converter operates on the four final BF16x2 pairs. Two adjacent threads
form one 16-value scale group. The optimized PTX emits a byte store from every
thread, not only the source-level even-thread predicate: paired threads have
the same reduced maximum and compute the same address, so they redundantly
store the same byte. This instruction behavior is preserved.

```text
global_scale = load_global_b32(output_sf_scale_ptr[0])
# selection: ld.global.b32; extent: one/thread/batch iteration

max2 = abs_bf16x2(output_bf16[0])
next2 = abs_bf16x2(output_bf16[1])
local2 = max_bf16x2(max2,next2)
next2 = abs_bf16x2(output_bf16[2])
local2 = max_bf16x2(local2,next2)
next2 = abs_bf16x2(output_bf16[3])
local2 = max_bf16x2(local2,next2)
peer2 = shuffle_xor_b32(local2,1,mask=0xffffffff)
group2 = max_bf16x2(peer2,local2)
vec_max_bf16 = max_bf16(group2.low,group2.high)
vec_max = bf16_to_f32(vec_max_bf16)
# selection: source-ordered abs.bf16x2 x4 and max.bf16x2 x4 total,
# shfl.sync.bfly.b32 xor 1 x1, max.bf16 x1, cvt.f32.bf16 x1

sf_value = mul_f32_ftz(global_scale,
                       mul_f32_ftz(vec_max,rcp_approx_ftz(6.0)))
sf_e4m3x2 = cvt_e4m3x2_f32_rn_satfinite(0.0,sf_value)
sf_byte = low_byte(sf_e4m3x2)
# selection: rcp.approx.ftz.f32; mul.ftz.f32 x2;
# cvt.rn.satfinite.e4m3x2.f32 with zero high input

k = tid // 2
num_k_tiles = 48
sf_offset = batch*ceil_div(runtime_num_rows,128)*(num_k_tiles*512) \
          + (row//128)*(num_k_tiles*512) + (k//4)*512 \
          + (row%32)*16 + ((row%128)//32)*4 + (k%4)
store_global_b8(sf_output_ptr + sf_offset,sf_byte)
# selection: st.global.b8; extent: one/thread/iteration, intentionally
# redundant within each two-thread group

is_zero = compare_eq_ftz_f32(vec_max,0.0)
output_scale = 0.0
if not is_zero:
    sf_low_b16 = and_b16(sf_e4m3x2,0x00ff)
    decoded_f16x2 = cvt_f16x2_e4m3x2_rn(sf_low_b16)
    decoded_low_u16 = cvt_u16_u32(decoded_f16x2)
    decoded_sf = cvt_f32_f16(decoded_low_u16)
    output_scale = rcp_approx_ftz(
        mul_f32_ftz(decoded_sf,rcp_approx_ftz(global_scale)))
# selection after SF store: setp.eq.ftz.f32, zero mov, predicated branch;
# nonzero path has and.b16 x1, cvt.rn.f16x2.e4m3x2 x1,
# cvt.u16.u32 x1, cvt.f32.f16 x1, rcp.approx.ftz x2, mul.ftz x1

q[k].x = mul_f32_ftz(bf16_to_f32(output_bf16[k].x),output_scale)
q[k].y = mul_f32_ftz(bf16_to_f32(output_bf16[k].y),output_scale) # k=0..3
byte[k] = cvt_e2m1x2_f32_rn_satfinite(q[k].y,q[k].x)             # k=0..3
packed = mov_b32_bytes(byte[0],byte[1],byte[2],byte[3])
store_generic_b32(norm_output_ptr +
                  ((batch*runtime_num_rows+row)*384 + tid),packed)
# selection: eight BF16 widens, mul.ftz x8,
# cvt.rn.satfinite.e2m1x2.f32 x4, exact mov.b32 {b0,b1,b2,b3},
# generic st.b32 x1
```

## Frozen MXFP8 tail

Four adjacent threads form one 32-value scale group. As with NVFP4, optimized
PTX emits one identical-address byte store per thread in the group.

```text
max2 = abs_bf16x2(output_bf16[0])
next2 = abs_bf16x2(output_bf16[1])
local2 = max_bf16x2(max2,next2)
next2 = abs_bf16x2(output_bf16[2])
local2 = max_bf16x2(local2,next2)
next2 = abs_bf16x2(output_bf16[3])
local2 = max_bf16x2(local2,next2)
peer2 = shuffle_xor_b32(local2,1,0xffffffff)
local2 = max_bf16x2(peer2,local2)
peer4 = shuffle_xor_b32(local2,2,0xffffffff)
group4 = max_bf16x2(peer4,local2)
vec_max_bf16 = max_bf16(group4.low,group4.high)
vec_max = bf16_to_f32(vec_max_bf16)
# selection: source-ordered abs.bf16x2 x4 and max.bf16x2 x5 total,
# shfl.sync.bfly.b32 xor 1 and xor 2, max.bf16, cvt.f32.bf16

sf_value = mul_f32_ftz(vec_max,rcp_approx_ftz(448.0))
sf_ue8m0x2 = cvt_ue8m0x2_f32_rp_satfinite(0.0,sf_value)
sf_byte = low_byte(sf_ue8m0x2)
is_zero = compare_eq_ftz_f32(vec_max,0.0)
output_scale = 0.0
if not is_zero:
    sf_low_b16 = and_b16(sf_ue8m0x2,0x00ff)
    decoded_bf16x2 = cvt_bf16x2_ue8m0x2_rn(sf_low_b16)
    decoded_low_bits = low_bf16_bits(decoded_bf16x2)
    decoded_sf = reinterpret_f32(shift_left_b32(decoded_low_bits,16))
    output_scale = rcp_approx_ftz(decoded_sf)
# selection: rcp.approx.ftz, mul.ftz,
# cvt.rp.satfinite.ue8m0x2.f32 with zero high input,
# then setp.eq.ftz.f32, zero mov, predicated branch; nonzero path has
# and.b16 x1, cvt.rn.bf16x2.ue8m0x2 x1, shl.b32 16 x1, and
# rcp.approx.ftz x1

k = tid // 4
num_k_tiles = 24
sf_offset = batch*ceil_div(runtime_num_rows,128)*(num_k_tiles*512) \
          + (row//128)*(num_k_tiles*512) + (k//4)*512 \
          + (row%32)*16 + ((row%128)//32)*4 + (k%4)
store_global_b8(sf_output_ptr + sf_offset,sf_byte)
# selection: st.global.b8; extent: one/thread/iteration, intentionally
# redundant within each four-thread group; PTX places this after the
# zero-test and conditional UE8M0 decode/output-scale chain

q[k].x = mul_f32_ftz(bf16_to_f32(output_bf16[k].x),output_scale)
q[k].y = mul_f32_ftz(bf16_to_f32(output_bf16[k].y),output_scale) # k=0..3
packed_pair[k] = cvt_e4m3x2_f32_rn_satfinite(q[k].y,q[k].x)       # k=0..3
wide[k] = cvt_u64_u16(packed_pair[k])                             # k=0..3
packed = or_b64(wide[0],shift_left_b64(wide[1],16))
packed = or_b64(packed,shift_left_b64(wide[2],32))
packed = or_b64(packed,shift_left_b64(wide[3],48))
store_generic_b64(norm_output_ptr +
                  ((batch*runtime_num_rows+row)*768 + 2*tid),packed)
# selection: eight BF16 widens, mul.ftz x8,
# cvt.rn.satfinite.e4m3x2.f32 x4, cvt.u64.u16 x4,
# shl.b64 x3, or.b64 x3, generic st.b64 x1; little-endian pair order
```

## Output layouts and validation invariants

BF16 norm output is compact `[B,R,3072]`. NVFP4 payload is compact packed
E2M1, exposed as int32 `[B,R,384]`; MXFP8 payload is compact packed E4M3,
exposed as int32 `[B,R,768]`.

For a scale-group column `k`, both quant formats use the exact 128x4 layout
above. NVFP4 has 48 K tiles and one logical SF for 16 hidden values; MXFP8 has
24 K tiles and one logical SF for 32 values. The batch stride includes
`ceil_div(R,128)` M tiles, so padding rows exist. Only source-addressed bytes
may change; all padded SF bytes and external guards remain untouched.

Correctness must cover all eighteen compile-time branches and the fixed
43-config matrix. CUDA and TIRx residual BF16 bytes, packed quant bytes, and
logical SF bytes must match exactly. BF16 norm values also compare to the FP32
oracle at upstream `rtol=1.6e-2, atol=1e-5`; dequantized values use
`rtol=0.5, atol=2.0`. Inputs, residual input, every auxiliary backing tensor,
FP32 parameters, and both global scales remain unchanged. Destination-passing
identity, 1-D/2-D bias views, 2-D/3-D auxiliary views, residual-none, one and
odd row counts, source/TIRx output independence, SF padding, and guard bytes
are part of the contract.

Any implementation using a different reduction topology, computing statistics
from BF16-rounded residuals, replacing accurate `__frsqrt_rn` with a lone
approximate instruction, predicating away the PTX-observed redundant SF byte
stores, moving bias loads into the batch loop, adding a row guard, PDL,
clusters, tile primitives, or a FuncCall escape does not implement this sketch.
