<!--
Copyright (c) 2026 The TIRx Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the
specific language governing permissions and limitations
under the License.

This design sketch documents a TIRx port of FlashInfer's
flashinfer/quantization/kernels/nvfp4_quantize.py
(NVFP4QuantizeLinearKernel / NVFP4QuantizeSwizzledKernel, including the
silu_and_mul=True compile variant), the CuTe-DSL SM100 kernels behind
flashinfer.quantization.nvfp4_quantize(backend="cute-dsl") and
flashinfer.quantization.silu_and_mul_nvfp4_quantize.
-->

# nvfp4_quantize SM100: coarse WASP pipeline sketch

This non-executable design sketch describes the thread roles, control flow,
storage placement, and PTX-level operations of
[`tirx_kernels/flashinfer/quantization/nvfp4_quantize.py`](../../tirx_kernels/flashinfer/quantization/nvfp4_quantize.py).
That TIRx module is the authoritative implementation.

The instantiations are `DTYPE in {f16, bf16}` crossed with
`SF_LAYOUT in {linear, 128x4, 8x4}` crossed with `SILU in {false, true}` of
the source CuTe-DSL classes `NVFP4QuantizeLinearKernel` /
`NVFP4QuantizeSwizzledKernel`, compiled for sm_100a with
`disable_fp4_quant_fast_math=False`, `nvfp4_4over6_config=None`, and
`global_scale_is_tensor=True`. Accepted target is SM100/B200. Grid and block
extents are static per config (the source host computes them once per call
from the same formulas, mirrored in the module); `M`, `total_sf_blocks`
(linear) / `M`, `padded_M` (swizzled) and all pointers stay runtime ABI
values, exactly like the source kernel signatures. The `enable_pdl=True`
variant is ported as the same compile-time knob (entry `griddepcontrol.wait`,
exit `griddepcontrol.launch_dependents`) but is off in all default configs:
TVM launches do not carry the PDL launch attribute, so test/bench parity pins
`enable_pdl=False` on both sides. Out of scope (deferred variants):
`NVFP4QuantizeTMAKernel` (env `FLASHINFER_NVFP4_QUANTIZE_USE_TMA`, off by
default), the 4over6 dual-scale path (env-gated), the
`FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH` exact-math path, the fp8e4m3 input
path, the per-token kernel (ported separately as `nvfp4_quantize_per_token`),
the host-scalar global-scale compile variant, and tile (`Tx`) primitives.

## Pipeline at a glance

| Warps | Role-local program | Publication/reuse edges |
| --- | --- | --- |
| all (uniform) | Every thread runs the same single-role program: one global-scale broadcast load at kernel start, then a flat (linear) or row-based (swizzled) loop over 16-element SF blocks: two 16-byte loads (four with SwiGLU: gate lo/hi + up lo/hi), optional silu(gate)*up in f32 rounded back to packed pairs, 8-word absmax tree, pair-max to f32, E4M3 scale factor from `global_scale * (amax * rcp.approx.ftz(6.0))`, output scale `rcp(SF_f32 * rcp(global_scale))`, 16 unpack-scale pairs, 2x e2m1x8 converts, one u64 pack, one unpredicated SF byte store, then one 8-byte output store (source order). | none — no SMEM, no mbarriers, no cross-thread data at all (1 thread per SF block; no shuffle reduction); the only ordering edges are the optional `griddepcontrol` pair |

There is no warp specialization and no producer/consumer split. The branches
are the runtime grid-stride loop guards, the compile-time `needs_col_loop` /
multi-row split (swizzled), the padding-row / padding-column zero-fill paths
(swizzled), and the zero-scale select inside `nvfp4_compute_output_scale`.

## Primitive vocabulary

Structural operations declare placement without moving data:

```python
specialize(...)       # compile-time variant selection
launch(...)           # compile-time launch topology and attributes
reg_tile(...)         # per-thread register tile
```

Copies state their direction and width:

```python
copy_g2r_v4_u32(src_addr, dst_b32x4)  # one 16-byte global -> register vector load
copy_g2r_f32(src_addr, dst)           # one scalar 32-bit global load (global scale broadcast; emitted ld.global.b32)
copy_r2g_u64(src_u64, dst_addr)       # one 8-byte register -> global store
copy_r2g_u8(src_u8, dst_addr)         # one byte register -> global store (emitted st.global.b8)
```

The compute vocabulary is deliberately primitive. Multi-instruction source asm
blocks are written as subroutines whose bodies annotate each instruction; a
call site counts as one op tile of the subroutine's dominant PTX family:

```python
abs_h2(dst_b32, src_b32)         # and.b32 with 0x7FFF7FFF (same for f16 and bf16)
max_h2(dst_b32, lhs, rhs)        # max.f16x2 (f16) | max.bf16x2 (bf16)
rcp_ftz(dst, src)                # rcp.approx.ftz.f32
mul(dst, lhs, rhs)               # mul.f32 (non-ftz, as the source asm)
fdiv(dst, lhs, rhs)              # div.rn.f32 (fdiv_rn; the silu path is NOT fast-div)
exp_fast(dst, src)               # cute.math.exp(fastmath=True) lowering
cast_e4m3x2(dst_u16, hi, lo)     # cvt.rn.satfinite.e4m3x2.f32 (hi operand first)
cast_f16x2_e4m3x2(dst_u32, src)  # cvt.rn.f16x2.e4m3x2 (SF decode path)
cast_e2m1x2(dst_u8, hi, lo)      # cvt.rn.satfinite.e2m1x2.f32 (hi operand first)
cast2x(dst_b32, lo, hi)          # 2x cvt.rn.f16.f32 + mov.b32 {h0,h1} (bf16: 2x cvt.rn.bf16.f32 + mov.b32); silu repack
select(dst, pred, a, b)          # selp family (UE8M0-style selects, output_scale zero)
setp(dst_pred, lhs, rhs)         # setp.eq.f32 / setp.eq.u32 inside subroutines
move / bitops                    # mov, and, or, shl, shr, add, sub on b32/s32/b64
idiv / imod(dst, lhs, rhs)       # integer div/mod by compile-time constants
```

Subroutines (bodies annotated per instruction in the sketch):

```python
ABSMAX_8(v[8]) -> b32                # half2_max_abs_8 / bfloat2_max_abs_8
HMAX_REDUCE_F32(x_b32) -> f32        # pair-max extraction to f32
FLOAT2_SCALED(h2_b32, scale) -> (f32, f32)  # half2/bfloat2_to_float2_scaled
E2M1X8(s[8]) -> u32                  # cvt_e2m1x8_f32: 4x e2m1x2 + byte pack
NVFP4_SCALE(amax, gs) -> (sf_u8, output_scale)  # _nvfp4_standard_quant_from_amax
SILU_HALF2(g_h2, u_h2) -> b32        # _silu_and_mul_half2 / _bfloat2
PROCESS_BLOCK(row, col) -> (sf_u8, packed64)  # process_nvfp4_block_[half|bfloat] /
                                     # process_nvfp4_silu_block_[half|bfloat]
```

`thread_id`, `cta_id`, `grid_dim`, `pdl_wait`, and `pdl_launch_dependents` are
schedule operations. Address expressions, loop bounds, and guards are shown
directly; they do not hide copies, computation, role changes, or
synchronization. All global addresses are 64-bit (`get_ptr_as_int64` in the
source); integer division/modulo by the compile-time constant
`num_sf_blocks_per_row = K/16` lowers to mul-shift or shr/and families.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

variant = specialize(DTYPE=("f16", "bf16"), SF_LAYOUT=("linear", "128x4", "8x4"),
                     SILU=(False, True), ENABLE_PDL=False, target="sm_100a")
# instruction_selection: none; extent: 2x3x2 compile-time instantiations (x PDL knob)

SF_VEC = 16                       # NVFP4_SF_VEC_SIZE
IN_COLS = 2 * K if SILU else K    # input row width (silu: gate || up)
GRID_X, BLOCK_X = host_launch_shape(M, K, SF_LAYOUT)  # static per config
# instruction_selection: none; extent: static launch metadata

launch_config = launch(
    grid=(GRID_X, 1, 1),
    block=(BLOCK_X, 1, 1),
    launch_bounds=(1024, 4),        # declared by the source launch (max_number_threads,
                                    # min_blocks_per_mp); see the static-boundary table for
                                    # the realized-occupancy note
    dynamic_smem_bytes=0,
)
# instruction_selection: none; extent: static launch metadata

# ===========================================================================
# LINEAR kernel (nvfp4_quantize.py:298-363)
# BLOCK_X = 512 (16 warps); SF_BLOCKS_PER_TB = 512; one SF block per thread
# ===========================================================================

def nvfp4_quantize_linear(
    input,          # DTYPE [M*IN_COLS], direct global pointer
    output,         # u8 [M*K/2], direct global pointer
    scales,         # u8 [total_sf_blocks], direct global pointer
    M,              # runtime i32 (unused by the body; source ABI parity)
    total_sf_blocks,  # runtime i32 (= M * K/16)
    global_scale,   # f32 [1], direct global pointer — LAST in the source ABI
):
    bx = cta_id(axis="x", extent=GRID_X)
    # instruction_selection: mov.u32 from %ctaid.x; extent: scalar per thread
    tx = thread_id(extent=BLOCK_X, dtype="uint32")
    # instruction_selection: mov.u32 from %tid.x; extent: scalar per thread
    gd = grid_dim(axis="x")
    # instruction_selection: mov.u32 from %nctaid.x; extent: scalar per thread

    pdl_wait()                        # only when ENABLE_PDL
    # instruction_selection: griddepcontrol.wait; extent: every thread, kernel entry

    gs = copy_g2r_f32(global_scale + 0)   # Float32(global_scale[0]) — once per thread
    # instruction_selection: ld.global.b32; extent: one scalar load, kernel start
    row_amax = 0.0                        # only consumed by the out-of-scope 4over6 path

    sf_idx = bx * 512 + tx
    # instruction_selection: mad.lo.s32 family; extent: scalar per thread
    while sf_idx < total_sf_blocks:
        # instruction_selection: setp.lt.s32 + bra; extent: loop control, no unroll
        row_idx = idiv(sf_idx, K // 16)
        col_idx = imod(sf_idx, K // 16)
        # instruction_selection: constant-divisor mul-shift family; extent: scalar per thread per iteration

        scale_fp8, packed64 = PROCESS_BLOCK(row_idx, col_idx, gs)
        # instruction_selection: see PROCESS_BLOCK; extent: one 16-element SF block

        # SF store first (linear index == sf_idx), then the output store (:351 then :357)
        copy_r2g_u8(scale_fp8, scales + sf_idx)
        # instruction_selection: st.global.b8; extent: one byte store per iteration
        out_base = i64(row_idx) * (K // 2) + col_idx * 8
        # instruction_selection: mad.lo.s64 family; extent: scalar per iteration
        copy_r2g_u64(packed64, i64(output) + out_base)
        # instruction_selection: st.global.u64; extent: one 8-byte store

        sf_idx = sf_idx + gd * 512
        # instruction_selection: mad.lo.s32; extent: loop induction update

    pdl_launch_dependents()           # only when ENABLE_PDL
    # instruction_selection: griddepcontrol.launch_dependents; extent: every thread, kernel exit

# ===========================================================================
# SWIZZLED kernel (nvfp4_quantize.py:487-646)
# BLOCK_X = _compute_optimal_threads(K) in [128, 512];
# compile-time split: needs_col_loop (num_sf_blocks_per_row > 512)
# ===========================================================================

def nvfp4_quantize_swizzled(
    input, output, scales,      # as linear
    M, padded_M,                # runtime i32
    global_scale,               # f32 [1] — LAST in the source ABI
):
    bx = cta_id(axis="x", extent=GRID_X)
    # instruction_selection: mov.u32 from %ctaid.x; extent: scalar per thread
    tx = thread_id(extent=BLOCK_X, dtype="uint32")
    # instruction_selection: mov.u32 from %tid.x; extent: scalar per thread
    gd = grid_dim(axis="x")
    # instruction_selection: mov.u32 from %nctaid.x; extent: scalar per thread

    pdl_wait()                        # only when ENABLE_PDL
    # instruction_selection: griddepcontrol.wait; extent: every thread, kernel entry

    gs = copy_g2r_f32(global_scale + 0)
    # instruction_selection: ld.global.b32; extent: one scalar load, kernel start

    NSB = K // 16                     # num_sf_blocks_per_row (compile-time)
    PAD_COLS = (NSB + 3) // 4 * 4     # padded_sf_cols (compile-time)

    if needs_col_loop:                # compile-time branch (K/16 > 512)
        # 1 block per row; threads stride columns (:523-576)
        row_idx = bx
        while row_idx < padded_M:
            # instruction_selection: setp.lt.s32 + bra; extent: row loop control
            if row_idx >= M:
                # padding row: zero-fill by ALL threads (no lane predicate), stride BLOCK_X
                sf_col = tx
                while sf_col < PAD_COLS:
                    # instruction_selection: setp/bra; extent: zero-fill loop control
                    copy_r2g_u8(0, scales + SF_OFFSET(row_idx, sf_col, PAD_COLS))
                    # instruction_selection: st.global.b8; extent: one byte store
                    sf_col = sf_col + BLOCK_X
            else:
                sf_col = tx
                while sf_col < NSB:
                    # instruction_selection: setp/bra; extent: column loop control
                    scale_fp8, packed64 = PROCESS_BLOCK(row_idx, sf_col, gs)
                    # instruction_selection: see PROCESS_BLOCK; extent: one SF block
                    # source order: SF byte store BEFORE the output store (:557 then :563)
                    copy_r2g_u8(scale_fp8, scales + SF_OFFSET(row_idx, sf_col, PAD_COLS))
                    # instruction_selection: st.global.b8; extent: one byte store per iteration
                    out_base = i64(row_idx) * (K // 2) + sf_col * 8
                    # instruction_selection: mad.lo.s64 family; extent: scalar per iteration
                    copy_r2g_u64(packed64, i64(output) + out_base)
                    # instruction_selection: st.global.u64; extent: one 8-byte store
                    sf_col = sf_col + BLOCK_X
                sf_col2 = NSB + tx
                while sf_col2 < PAD_COLS:      # padding columns of a data row
                    # instruction_selection: setp/bra; extent: zero-fill loop control
                    copy_r2g_u8(0, scales + SF_OFFSET(row_idx, sf_col2, PAD_COLS))
                    # instruction_selection: st.global.b8; extent: one byte store
                    sf_col2 = sf_col2 + BLOCK_X
            row_idx = row_idx + gd
            # instruction_selection: add.s32; extent: row induction update
    else:
        # multi-row path: threads_per_row == NSB (1 thread per SF block)
        row_in_block = idiv(tx, NSB)
        sf_idx_in_row = imod(tx, NSB)
        # instruction_selection: constant-divisor shr/and or mul-shift family; extent: scalar per thread
        ROWS_PER_BLOCK = BLOCK_X // NSB

        row_batch_idx = bx
        row_idx = row_batch_idx * ROWS_PER_BLOCK + row_in_block
        while row_batch_idx * ROWS_PER_BLOCK < padded_M:
            # instruction_selection: setp.lt.s32 + bra; extent: batch loop control
            if row_idx < padded_M:
                # instruction_selection: setp.ge.s32 + bra (emitted polarity: skips body); extent: per iteration
                if row_idx >= M:
                    # padding row: zero ALL padded SF columns; stride threads_per_row == NSB (:597-603)
                    local_sf = sf_idx_in_row
                    while local_sf < PAD_COLS:
                        # instruction_selection: setp/bra; extent: zero-fill loop control
                        copy_r2g_u8(0, scales + SF_OFFSET(row_idx, local_sf, PAD_COLS))
                        # instruction_selection: st.global.b8; extent: one byte store
                        local_sf = local_sf + NSB
                else:
                    if sf_idx_in_row < NSB:
                        # provably true (sf_idx_in_row = tidx % NSB < NSB); compile-time folded
                        scale_fp8, packed64 = PROCESS_BLOCK(row_idx, sf_idx_in_row, gs)
                        # instruction_selection: see PROCESS_BLOCK; extent: one SF block
                        # source order: SF byte store BEFORE the output store (:619 then :625)
                        copy_r2g_u8(scale_fp8,
                                    scales + SF_OFFSET(row_idx, sf_idx_in_row, PAD_COLS))
                        # instruction_selection: st.global.b8; extent: one byte store
                        out_base = i64(row_idx) * (K // 2) + sf_idx_in_row * 8
                        # instruction_selection: mad.lo.s64 family; extent: scalar per iteration
                        copy_r2g_u64(packed64, i64(output) + out_base)
                        # instruction_selection: st.global.u64; extent: one 8-byte store
                    if NSB != PAD_COLS:       # compile-time; padding SF columns of data row
                        pad_col = NSB + sf_idx_in_row
                        while pad_col < PAD_COLS:
                            # instruction_selection: setp/bra; extent: zero-fill loop control
                            copy_r2g_u8(0, scales + SF_OFFSET(row_idx, pad_col, PAD_COLS))
                            # instruction_selection: st.global.b8; extent: one byte store
                            pad_col = pad_col + NSB   # stride threads_per_row == NSB (:638)
            row_batch_idx = row_batch_idx + gd
            row_idx = row_batch_idx * ROWS_PER_BLOCK + row_in_block
            # instruction_selection: mad.lo.s32; extent: batch induction update

    pdl_launch_dependents()           # only when ENABLE_PDL
    # instruction_selection: griddepcontrol.launch_dependents; extent: every thread, kernel exit

# ===========================================================================
# PROCESS_BLOCK — process_nvfp4_block_half (utils:1870) / _bfloat (:1895), or
# process_nvfp4_silu_block_half (:1948) / _bfloat (:2004) when SILU.  Loads the
# block, optionally applies SwiGLU, computes the E4M3 SF and output scale, and
# packs to one u64.  Like the source helper it performs NO stores.
# ===========================================================================

def PROCESS_BLOCK(row_idx, col_idx, gs):
    elem_base = col_idx * SF_VEC
    # instruction_selection: shl by constant; extent: scalar
    base = i64(row_idx) * IN_COLS + elem_base
    # instruction_selection: mul.wide.s32/mad.lo.s64 family; extent: scalar
    h = reg_tile("b32", [8])          # gate (or only) words
    # instruction_selection: none; extent: eight b32 registers per thread
    copy_g2r_v4_u32(i64(input) + (base + 0) * 2, h[0:4])
    # instruction_selection: ld.global.v4.u32; extent: one 16-byte vector load
    copy_g2r_v4_u32(i64(input) + (base + 8) * 2, h[4:8])
    # instruction_selection: ld.global.v4.u32; extent: one 16-byte vector load
    if SILU:
        u = reg_tile("b32", [8])      # up words at +K elements (:1967-1972)
        # instruction_selection: none; extent: eight b32 registers per thread
        copy_g2r_v4_u32(i64(input) + (base + K + 0) * 2, u[0:4])
        # instruction_selection: ld.global.v4.u32; extent: one 16-byte vector load
        copy_g2r_v4_u32(i64(input) + (base + K + 8) * 2, u[4:8])
        # instruction_selection: ld.global.v4.u32; extent: one 16-byte vector load
        for i in static_range(8):
            h[i] = SILU_HALF2(h[i], u[i])
            # instruction_selection: see SILU_HALF2; extent: one packed pair per call

    block_max_h2 = ABSMAX_8(h)
    # instruction_selection: see ABSMAX_8; extent: 8x and.b32 + 7x max.{f16,bf16}x2
    block_max = HMAX_REDUCE_F32(block_max_h2)
    # instruction_selection: see HMAX_REDUCE_F32; extent: one pair to one f32

    scale_fp8, output_scale = NVFP4_SCALE(block_max, gs)
    # instruction_selection: see NVFP4_SCALE; extent: one scalar scale pair

    s = reg_tile("f32", [16])
    # instruction_selection: none; extent: per-thread registers
    for i in static_range(8):
        s[2*i], s[2*i+1] = FLOAT2_SCALED(h[i], output_scale)
        # instruction_selection: see FLOAT2_SCALED; extent: one packed pair -> two f32
    packed_lo = E2M1X8(s[0:8])
    # instruction_selection: see E2M1X8; extent: 4x cvt.rn.satfinite.e2m1x2.f32 + byte pack
    packed_hi = E2M1X8(s[8:16])
    # instruction_selection: see E2M1X8; extent: second 8-element group
    packed64 = (u64(packed_hi) << 32) | u64(packed_lo)
    # instruction_selection: shl.b64 + or.b64 (source: plain CuTe-DSL u64 shift/or); extent: one u64
    return scale_fp8, packed64

# ===========================================================================
# NVFP4_SCALE — _nvfp4_standard_quant_from_amax (utils:1588, fast-math path)
# + nvfp4_compute_output_scale (fp4_common.py:973).  Returns (sf_u8, out_scale).
# ===========================================================================

def NVFP4_SCALE(amax, gs):
    fp4_max_rcp = rcp_ftz(6.0)
    # instruction_selection: rcp.approx.ftz.f32; extent: one scalar
    scale_float = mul(gs, mul(amax, fp4_max_rcp))   # source order: gs * (amax * rcp)
    # instruction_selection: mul.f32 x2; extent: two scalars
    sf_bits = CVT_F32_TO_E4M3(scale_float)          # fp4_common.py:811
    # instruction_selection: mov.f32 0 + cvt.rn.satfinite.e4m3x2.f32 + cvt.u32.u16; extent: one scalar
    scale_fp8 = low_u8(sf_bits)
    # instruction_selection: none — the u8 truncation folds into the st.global.b8 low-byte store; extent: one byte
    # nvfp4_compute_output_scale (fp4_common.py:973):
    fp8_pair = u16(sf_bits)
    # instruction_selection: cvt.u16.u32; extent: one scalar
    h2 = cast_f16x2_e4m3x2(fp8_pair)
    # instruction_selection: cvt.rn.f16x2.e4m3x2; extent: one packed pair
    sf_f32 = cast_f32_h(low_b16(h2))
    # instruction_selection: mov.b32 {h_lo,h_hi} + cvt.f32.f16; extent: one scalar
    product = mul(sf_f32, rcp_ftz(gs))
    # instruction_selection: rcp.approx.ftz.f32 + mul.f32; extent: one scalar each
    result = rcp_ftz(product)
    # instruction_selection: rcp.approx.ftz.f32; extent: one scalar
    output_scale = select(sf_f32 == 0.0, 0.0, result)
    # instruction_selection: setp.eq.f32 + selp.f32; extent: one scalar
    return scale_fp8, output_scale

# ===========================================================================
# SILU_HALF2 — _silu_and_mul_half2 (utils:1740) / _bfloat2 (:1752):
# unpack both pairs with scale 1.0, silu(g)*u per scalar in f32, repack.
# ===========================================================================

def SILU_HALF2(g2, u2):
    g0, g1 = FLOAT2_SCALED(g2, 1.0)
    # instruction_selection: see FLOAT2_SCALED (scale constant 1.0); extent: one pair
    u0, u1 = FLOAT2_SCALED(u2, 1.0)
    # instruction_selection: see FLOAT2_SCALED (scale constant 1.0); extent: one pair
    a0 = mul(SILU_ONE(g0), u0)
    # instruction_selection: mul.f32 (after the silu sequence); extent: one scalar
    a1 = mul(SILU_ONE(g1), u1)
    # instruction_selection: mul.f32; extent: one scalar
    return cast2x(a0, a1)
    # instruction_selection: cvt.rn.f16.f32 x2 + mov.b32 {h0,h1} (bf16: cvt.rn.bf16.f32 x2 + mov.b32 {h0,h1}); extent: one packed pair

def SILU_ONE(g):                      # _silu_f32 (utils:1731): g / (1 + exp(-g)) via fdiv_rn
    e = exp_fast(-g)                  # cute.math.exp(-g, fastmath=True)
    # instruction_selection: mul.f32 (x -log2e, folds the negation) + ex2.approx.ftz.f32
    # + add.f32 (+1.0); extent: one scalar
    return fdiv(g, 1.0 + e)           # fdiv_rn: div.rn.f32 (NOT approximate)
    # instruction_selection: div.rn.f32; extent: one scalar

# ===========================================================================
# ABSMAX_8, HMAX_REDUCE_F32, FLOAT2_SCALED, E2M1X8, SF_OFFSET — the SAME source
# helpers reviewed in .agents/sketch/flashinfer/quantization/mxfp8_quantize.md / mxfp4_quantize.md
# (utils:691/:728, :95/:122, :376/:406, :442, :535/:569) with identical
# instruction selections; not repeated here.
# ===========================================================================
```

## Host wrapper and validation

The Python module performs host-only work; none of it emits device PTX:

```python
def host_launch_shape(m, k, sf_layout):
    # linear (nvfp4_quantize.py:1941-1951): 512 threads; SF_BLOCKS_PER_TB = 512
    #   grid = min(ceildiv(m*(k/16), 512), SM_COUNT * 4)
    # swizzled (:1967-1980): threads = _compute_optimal_threads(k) in [128, 512]
    #   (largest multiple of k/16); needs_col_loop when k/16 > 512;
    #   grid = min(ceildiv(padded_m, rows_per_block), SM_COUNT * 4)
    # instruction_selection: none; extent: host-only arithmetic (SM_COUNT=148 on B200)

def prepare_data(dtype, m, k, sf_layout, fuse_silu):
    host_assert(dtype in ("float16", "bfloat16") and k % 16 == 0 and m >= 1)
    a = contiguous_seeded_randn(shape=(m, k * (2 if fuse_silu else 1)), dtype=dtype)
    global_scale = seeded_uniform_fp32(shape=(1,)) + 0.5   # device f32 [1]
    # instruction_selection: none; extent: tensor constructions

# run_test compares the packed FP4 output and the full SF buffer (source
# zero-fills padding rows/columns, so the whole padded buffer is compared)
# against flashinfer.quantization.nvfp4_quantize(backend="cute-dsl",
# enable_pdl=False) or silu_and_mul_nvfp4_quantize(enable_pdl=False), exact
# byte equality (rtol=0, atol=0).
# run_bench times the primfunc launch (tirx) against the cached compiled
# source kernel (_get_compiled_kernel_nvfp4, device-tensor global scale) with
# preallocated outputs, enable_pdl=False; both closures are no-argument.
```

## Static specialization boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| `DTYPE` | static per config | selects f16/bf16 unpack and `max.f16x2` vs `max.bf16x2`, `cvt.rn.f16.f32` vs `cvt.rn.bf16.f32` (silu repack) |
| `SILU` | static per config | doubles input loads (gate+up), adds the silu f32 sequence and repack; input row width 2K |
| `SF_LAYOUT` | static per config | selects linear vs swizzled kernel body and the SF index math (128x4 vs 8x4) |
| `K` | static per config | `NSB = K/16`, `PAD_COLS`, swizzled threads/rows_per_block/needs_col_loop constant-fold |
| grid/block extents | static per config | host formulas baked in; kernel reads `%ctaid.x`/`%tid.x`/`%nctaid.x` only |
| `M`, `total_sf_blocks`, `padded_M` | runtime i32 ABI | loop bounds and row predicates stay runtime like the source (linear `M` is ABI-only) |
| `global_scale` device tensor | runtime pointer; value read once per thread | one `ld.global.b32` broadcast at kernel start |
| `ENABLE_PDL` | static per config (default False) | griddepcontrol pair present only in the PDL instantiation |
| launch bounds | static | source declares (1024, 4) but ptxas does not enforce the implied cap for these kernels (the mxfp4-class source binary realizes ~50 regs / 2 blocks per SM); the TIRx build sets min_blocks_per_sm=2 to realize the source binary's occupancy instead of the unenforced hint |
| `disable_fast_math=False`, `4over6=None`, `global_scale_is_tensor=True`, `use_tma=False` | static | default-environment code path only |

## TIRx module and benchmark contract

- `KERNEL_META = {"name": "nvfp4_quantize", "category": "flashinfer",
  "compute_capability": 10}`.
- The executable kernel is expressed entirely in plain TIRx: explicit `while`
  grid-stride loops, runtime scalar ABI, register tiles, and native `T.ptx.*`
  forms for every non-trivial instruction (`ld.global.v4.u32`,
  `ld.global.b32`, `st.global.u64`/`st.global.b8`, `max.f16x2`/`max.bf16x2`,
  `max.f32`, `mul.f32` non-ftz, `div.rn.f32`, the exp-fastmath lowering
  (`mul.f32` + `ex2.approx.ftz.f32`),
  `rcp.approx.ftz.f32`, `cvt.rn.satfinite.e4m3x2.f32`,
  `cvt.rn.f16x2.e4m3x2`, `cvt.rn.satfinite.e2m1x2.f32`,
  `cvt.rn.f16.f32`/`cvt.rn.bf16.f32` + `mov.b32` pack, `setp`/`selp` families,
  `griddepcontrol`). Integer bit math is plain TIRx b32 ops. There is no
  `T.cuda.func_call` and no `Tx` tile primitives anywhere in the pre-dispatch
  IR.
- `get_kernel(dtype, m, k, sf_layout, fuse_silu, enable_pdl)` returns the
  specialized primfunc with static grid/block; `prepare_data`, `run_test`,
  `run_bench` follow the repository contract.
- The timed implementation is named `tirx`; the reference is the cached
  compiled CuTe-DSL source kernel launched with `enable_pdl=False` and
  preallocated outputs. Allocation, compilation, and correctness checks stay
  outside timing.
- Correctness compares the full packed output and the full padded SF buffer
  against the source wrapper, exact byte equality.

## Instruction selection is a lowering consequence

The sketch above never requests a hardware instruction beyond the documented
PTX helpers. The following lowering families follow from storage direction,
shape, dtype, and schedule. Per-iteration counts below are the expected values
the sketch-reviewer verifies against a fresh line-info PTX export of the exact
source specializations (`CUTE_DSL_KEEP=ptx CUTE_DSL_LINEINFO=1`,
enable_pdl=False, use_tma unset, default env).

| Primitive/schedule pattern | PTX family (expected, per SF block, non-silu) |
| --- | --- |
| global scale broadcast | `ld.global.b32` x1 (kernel start, per thread) |
| `copy_g2r_v4_u32` loads | `ld.global.v4.u32` x2 (silu: x4; no `.nc`) |
| silu per pair (SILU only) | unpack (mov/cvt or shift/mask family) + `mul.f32`(-log2e) + `ex2.approx.ftz.f32` + `add.f32` + `div.rn.f32` + `mul.f32` + `cvt.rn.f16.f32` x2 + `mov.b32 {h0,h1}` (bf16: `cvt.rn.bf16.f32` x2) (x8 pairs) |
| absmax tree | `and.b32` x8 + `max.f16x2` x7 (bf16: `max.bf16x2`) |
| pair-max to f32 | f16: `mov.b32 {h0,h1}` + `cvt.f32.f16` x2 + `max.f32`; bf16: `and.b32` + `shr.b32` + `shl.b32` x2 + `mov.b32` x2 + `max.f32` |
| scale normalize | `rcp.approx.ftz.f32` x1 + `mul.f32` x2 |
| E4M3 SF convert | `mov.f32` 0 + `cvt.rn.satfinite.e4m3x2.f32` + `cvt.u32.u16` (u8 narrowing folds into the byte store) |
| output scale | `cvt.u16.u32` + `cvt.rn.f16x2.e4m3x2` + `mov.b32 {h_lo,h_hi}` + `cvt.f32.f16` + `rcp.approx.ftz.f32` x2 + `mul.f32` + `setp.eq.f32` + `selp.f32` |
| unpack + scale (16 elements) | f16: `mov.b32 {h0,h1}` x8 + `cvt.f32.f16` x16; bf16 (per pair): `and.b32` + `shr.b32` + `shl.b32` x2 + `mov.b32` x2 (x8 pairs); `mul.f32` x16 |
| e2m1 convert + pack | `cvt.rn.satfinite.e2m1x2.f32` x8 + byte-pack movs (source `mov.b32 {b0..b3}` x2; TIRx b16-pair `shl`/`or` + `mov.b32` x2) |
| u64 combine | `cvt.u64.u32` + `shl.b64` + `or.b64` (source: CuTe u64 shift/or, not mov.b64) |
| output store | `st.global.u64` x1 (AFTER the SF store) |
| SF store | `st.global.b8` x1 (unpredicated, BEFORE the output store) |
| SF swizzle address | `and.b32`/`shr.b32` by constants + `mad.lo.s32` family (swizzled layouts only) |
| row/col decode + addresses | constant-divisor mul-shift family, `mul.wide.s32`/`mad.lo.s64` 64-bit address math |
| loop control | `setp` + `bra`, `add.s32`/`mad.lo.s32` induction |
| PDL (ENABLE_PDL only) | `griddepcontrol.wait` x1, `griddepcontrol.launch_dependents` x1 (whole kernel) |

The bf16 instantiations differ only in the unpack sequences and the
`max.bf16x2`/`cvt.rn.bf16.f32` opcodes. The 8x4 instantiation differs from
128x4 only in the SF_OFFSET arithmetic. The silu instantiation adds the
gate/up loads, the exp/div.rn f32 sequence, and the scalar-cvt repack before
the shared quantization pipeline.
