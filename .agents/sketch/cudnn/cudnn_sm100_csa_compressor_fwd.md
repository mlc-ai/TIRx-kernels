<!--
This file is a design sketch for a TIRx port of code from cuDNN Frontend
(https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5),
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# cuDNN SM100 CSA compressor forward: coarse execution sketch

This is a non-executable execution sketch for
[`tirx_kernels/cudnn/csa/compressor_fwd_sm100.py`](../../../tirx_kernels/cudnn/csa/compressor_fwd_sm100.py).
It freezes the source kernel's one-row/one-column-group CTA mapping, scalar or
paired lane ownership, APE hoist, ragged segment scan, overlap-window branches,
serial FP32 reduction order, and BF16 traffic. After the initial sketch review,
the linked module becomes the executable source of truth and this file remains
frozen.

The source is
`python/cudnn/csa/compressor/compressor_sm100.py::_compressor_fwd_kernel` at
commit `aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5`. Ratio 4 with
`coff in {1,2}` and `d in {65,128,512}` is in scope. The separate ratio-128
kernel, backward kernel, and head dimensions outside this validated dispatch
domain are out of scope.

## Line-info evidence

The writer exports are under
`.porting/compressor_fwd_sm100/writer_source_export/`; every artifact targets
`sm_100a`, declares `.reqntid 64,1,1`, and carries usable `.loc` directives into
the source file.

| specialization | vec | PTX SHA256 | lines | `.loc` | branch evidence |
| --- | ---: | --- | ---: | ---: | --- |
| d65_c1 | 1 | `ad4a5aef93ca7bed6a2883d648965ef960310dc1bd39a3310c20e2e9e9b90b46` | 480 | 125 | scalar, own-block window |
| d65_c2 | 1 | `7ff03b62976ef50640ba8bbd658df02aa0f15036cfc3d4d8984102793aeb9fdd` | 662 | 186 | scalar, overlap window |
| d128_c1 | 2 | `f5a8b961321df26d554d5b47693bf3f8368451bc1d879cbb862075e88f169510` | 613 | 168 | paired, own-block window |
| d128_c2 | 2 | `19e9c7b32a4500d0c8222334626a678eb2ef4b021c92e8bcca6202bfb4005b28` | 1077 | 335 | paired, overlap window |
| d512_c1 | 2 | `7587c4f17344400da14a3cd7c58564027005d9bb102ca9f33ffef16c27adc81c` | 613 | 168 | paired, wider grid.y |
| d512_c2 | 2 | `991dd7b1c4030931986ee098b0a17bfe09b675a2e315685800487ae9b18a958e` | 1089 | 338 | paired, wider grid.y |

Instruction counts below use instruction lines minus predicated lines.

## Execution roles at a glance

| role | threads | owned work | communication |
| --- | --- | --- | --- |
| compressor lane | all 64 CTA threads | one output row `bb=blockIdx.x`, and `vec` adjacent head columns selected by `blockIdx.y*64+threadIdx.x` | none; every output element is thread-local |

There is no producer/consumer split, shared memory, TMEM, barrier, pipe, stage,
phase, warp exchange, or CTA exchange. Both warps execute the same independent
lane program. A thread whose column is outside `d/vec` exits; a CTA whose row is
outside `nb_total` exits.

## Primitive vocabulary

- `g2r_ape(dst, address, vec)`: copy one or two FP32 APE values from GMEM to registers.
  `# instruction_selection: ld.global.b32 or ld.global.v2.b32; extent: one (k,lane-group) slice`
- `g2r_previous_bf16(dst, address, vec)`: conditionally copy a coff-2 previous-half score/KV slice.
  `# instruction_selection: ld.global.b16 for vec1; packed ld.global.b32 followed by mov.b32 lane extraction for vec2; extent: one (k,lane-group) slice`
- `g2r_own_bf16(dst, address, vec)`: copy an own-block score/KV slice.
  `# instruction_selection: ld.global.b16 or ld.global.v2.b16; extent: one (k,lane-group) slice`
- `bf16_add_ape(dst, score_bits, ape)`: widen score and add APE with the source rounding.
  `# instruction_selection: add.rn.f32.bf16; extent: one window lane`
- `bf16_to_f32(dst, bits)`: widen a KV lane.
  `# instruction_selection: cvt.f32.bf16; extent: one window lane`
- `ordered_max(dst, candidate)`: replace the running maximum only when candidate is greater.
  `# instruction_selection: setp.gt.f32 + selp.f32; extent: one serial comparison`
- `source_exp(dst, x)`: one CuTe exponential after subtracting the maximum.
  `# instruction_selection: 11 arithmetic instructions: sub.f32; fma.rn.f32; cvt.sat.f32.f32; fma.rm.f32; add.f32; neg.f32; two fma.rn.f32; shl.b32; ex2.approx.ftz.f32; mul.f32; extent: one scalar. Two mov.b32 constant initializations are shared across every exponent and lane of the thread.`
- `ordered_add(dst, value)`: append one term to a serial FP32 accumulation.
  `# instruction_selection: add.f32; extent: one scalar term`
- `rounded_weight(dst, exponent, denominator)`: form one probability.
  `# instruction_selection: div.rn.f32; extent: one scalar term`
- `rounded_product(dst, left, right)`: form the source-pinned weighted KV product.
  `# instruction_selection: mul.rn.f32; extent: one scalar term`
- `r2g_bf16(out, address, vec)`: round and store one or two output lanes.
  `# instruction_selection: cvt.rn.bf16.f32 + st.global.b16, or cvt.rn.bf16x2.f32 + st.global.b32; extent: one lane-group`

No primitive above contains another algorithmic phase or changes thread ownership.

## Complete sketch

```python
P = specialize(ratio=4, d in {65, 128, 512}, coff in {1, 2})
vec = 2 if d % 2 == 0 else 1
W = coff * d
win = 8 if coff == 2 else 4
ncol = d // vec

ABI = (
    kv_bf16_flat, score_bf16_flat, ape_f32_flat,
    cu_seqlens_i32, cu_seqlens_comp_i32, out_bf16_flat,
    nb_total_i32, n_seq_i32,
)
launch(grid=(nb_total, ceil_div(ncol, 64), 1), block=(64, 1, 1), arch="sm_100a")
# instruction_selection: `.reqntid 64,1,1`; extent: one specialization

bb = block_idx_x
col = block_idx_y * 64 + thread_idx_x
if col < ncol:
    cvec = col * vec

    # All APE traffic is before nb_valid and before the row/segment work. The
    # k order remains 0..win-1 even though coff=2 alternates physical halves.
    ape_k = registers_f32(win * vec)
    for k in static_range(win):
        colbase = cvec if coff == 1 or k < 4 else d + cvec
        g2r_ape(ape_k[k*vec:(k+1)*vec],
                ape + ((k % 4) * W + colbase), vec)
        # instruction_selection: `ld.global.b32` (vec1) or
        #   `ld.global.v2.b32` (vec2); extent: win static slices

    nb_valid = g2r_i32(cu_seqlens_comp[n_seq])
    # instruction_selection: `ld.global.b32`; extent: one scalar per active thread

    if bb < nb_total:
        # Padding rows deliberately keep the initial (seq_idx,bis)=(0,0).
        seq_idx = 0
        bis = 0
        if bb < nb_valid:
            bis = bb
            for s in dynamic_range(n_seq):
                cs = g2r_i32(cu_seqlens_comp[s])
                ce = g2r_i32(cu_seqlens_comp[s + 1])
                # instruction_selection: scalar `ld.global.b32` loads; extent:
                #   source-order segment scan. Generated PTX uses a nounroll loop
                #   with integer comparisons/selects over groups of boundaries.
                if bb >= cs and bb < ce:
                    seq_idx = s
                    bis = bb - cs

        token_base = g2r_i32(cu_seqlens[seq_idx])
        # instruction_selection: `ld.global.b32`; extent: one selected segment base
        tok0 = token_base + bis * 4

        scores = registers_f32(win * vec)
        values = registers_f32(win * vec)
        for k in static_range(win):
            if coff == 2 and k < 4:
                if bis > 0:
                    offset = (tok0 - 4 + k) * W + cvec
                    score_bits = registers_bf16(vec)
                    value_bits = registers_bf16(vec)
                    g2r_previous_bf16(score_bits, score + offset, vec)
                    # instruction_selection: vec1 `ld.global.b16`; vec2 packed
                    #   `ld.global.b32` plus `mov.b32` extraction; extent: one
                    #   previous-block score slice
                    g2r_previous_bf16(value_bits, kv + offset, vec)
                    # instruction_selection: same conditional packed/scalar family;
                    #   extent: one previous-block KV slice
                for lane in static_range(vec):
                    # Preserve source/PTX order: defaults are constructed for both
                    # paths after the guarded slice loads.
                    scores[k*vec+lane] = -inf
                    values[k*vec+lane] = 0.0
                    # instruction_selection: immediate/move construction; extent:
                    #   one unconditional pair per lane
                    if bis > 0:
                        bf16_add_ape(scores[k*vec+lane], score_bits[lane], ape_k[k*vec+lane])
                        # instruction_selection: `add.rn.f32.bf16`; extent: one lane
                        bf16_to_f32(values[k*vec+lane], value_bits[lane])
                        # instruction_selection: `cvt.f32.bf16`; extent: one lane
                    # The overwrite and APE add stay guarded, so non-finite APE
                    # cannot affect the invalid half of a segment's first block.
            else:
                offset = ((tok0 + k - 4) * W + d + cvec) if coff == 2 else ((tok0 + k) * W + cvec)
                score_bits = registers_bf16(vec)
                value_bits = registers_bf16(vec)
                g2r_own_bf16(score_bits, score + offset, vec)
                # instruction_selection: `ld.global.b16` or `ld.global.v2.b16`;
                #   extent: one own-block score slice
                g2r_own_bf16(value_bits, kv + offset, vec)
                # instruction_selection: same width; extent: one own-block KV slice
                for lane in static_range(vec):
                    bf16_add_ape(scores[k*vec+lane], score_bits[lane], ape_k[k*vec+lane])
                    # instruction_selection: `add.rn.f32.bf16`; extent: one lane
                    bf16_to_f32(values[k*vec+lane], value_bits[lane])
                    # instruction_selection: `cvt.f32.bf16`; extent: one lane

        output = registers_f32(vec)
        for lane in static_range(vec):
            maximum = scores[lane]
            for k in static_range(1, win):
                ordered_max(maximum, scores[k*vec+lane])
                # instruction_selection: `setp.gt.f32` + `selp.f32`; extent: win-1 serial comparisons

            denominator = 0.0
            exponent = registers_f32(win)
            # Constants 0f4B400001 and 0f437C0000 are initialized by two shared
            # mov.b32 instructions once per thread; the other constants below are
            # instruction immediates.
            for k in static_range(win):
                source_exp(exponent[k], scores[k*vec+lane] - maximum)
                # instruction_selection: the fixed 11-instruction arithmetic
                #   range-reduction/FMA/`ex2` sequence listed above; extent: one
                #   scalar exponent, plus the two thread-shared constant moves
                ordered_add(denominator, exponent[k])
                # instruction_selection: `add.f32`; extent: one serial denominator term

            accumulator = 0.0
            for k in static_range(win):
                probability = rounded_weight(exponent[k], denominator)
                # instruction_selection: `div.rn.f32`; extent: one scalar probability
                product = rounded_product(values[k*vec+lane], probability)
                # instruction_selection: source-pinned `mul.rn.f32`; extent: one scalar product
                ordered_add(accumulator, product)
                # instruction_selection: `add.f32`; extent: one serial output term
            output[lane] = accumulator

        r2g_bf16(out + bb*d + cvec, output, vec)
        # instruction_selection: scalar BF16 conversion/store or packed BF16x2
        #   conversion plus 32-bit store; extent: one thread-owned lane-group
```

## Static specialization boundary

| axis | accepted values | emitted-code consequence |
| --- | --- | --- |
| ratio | 4 | fixes a four-row APE and windows of 4 or 8 |
| coff | 1, 2 | selects own-block-only versus overlap window, W, APE half, and register count |
| d | 65, 128, 512 | selects scalar versus paired traffic and `grid.y`; d512 changes address constants but not the lane program |
| vec | derived 1 or 2 | selects scalar/v2 loads and scalar/packed output conversion/store |
| rows_per_cta | 1 | one `blockIdx.x` row, including padding rows |
| threads | 64 | two K warps and source-identical column grouping |

## TIRx and benchmark contract

- Registry name `cudnn_sm100_csa_compressor_fwd`, category `cudnn`, compute capability 10.
- Device implementation imports only `tirx_kernels.kern as K`; all memory and key arithmetic instructions use plain K expressions or `K.ptx`.
- Correctness includes the source ratio-4 matrix, static-capacity padding, empty output, determinism, and an independent FP32 eager oracle.
- The performance matrix is exactly `(batch,d) in {(1,128),(3,128),(1,512),(3,512)}` with every sequence length 8192 and coff=2.
- The only performance gate is a complete bench-suite run with the pinned cuDNN Frontend source reference; every row requires `mean(source_us)/mean(tirx_us) > 0.99`.

## Instruction-selection summary

| source decision | generated consequence |
| --- | --- |
| APE list built before row work | all 4/8 APE loads are hoisted above `nb_valid` and the segment scan |
| `vec=1` | scalar `ld.global.b32` APE, scalar `ld.global.b16` score/KV, scalar BF16 store |
| `vec=2` | `ld.global.v2.b32` APE; packed `ld.global.b32` + extraction for conditional previous-half score/KV; `ld.global.v2.b16` for own-block score/KV; `cvt.rn.bf16x2.f32` and `st.global.b32` |
| BF16 score plus FP32 APE | fused `add.rn.f32.bf16`, not separate conversion and add |
| invalid previous block | immediate `-inf`/zero with neither source load nor APE add |
| serial maximum | ordered `setp.gt.f32`/`selp.f32` chain, no reduction primitive |
| `cute_math.exp` | 11 line-associated arithmetic instructions around `ex2.approx.ftz.f32`, repeated 4/8 times per lane; two constant `mov.b32` instructions are shared by the thread, and the denominator `add.f32` is a separate operation |
| `ex[k] / den` | `div.rn.f32`, not reciprocal approximation |
| `_fmul_rn` | inline `mul.rn.f32`, opaque to contraction |
| serial weighted sum | rounded product followed by `add.f32` in k order |

For the paired exports, coff=1 contains 8 each of the exp-expansion arithmetic,
division, and rounded multiply instructions; coff=2 contains 16 each because
both adjacent lanes evaluate an 8-element window. The scalar exports contain 4
or 8 respectively. These counts describe static instructions, not dynamic
executions.
