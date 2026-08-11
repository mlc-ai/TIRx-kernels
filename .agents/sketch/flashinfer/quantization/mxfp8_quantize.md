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
flashinfer/quantization/kernels/mxfp8_quantize.py
(MXFP8QuantizeLinearKernel / MXFP8QuantizeSwizzledKernel), the
CuTe-DSL SM100 kernels behind
flashinfer.quantization.mxfp8_quantize(backend="cute-dsl").
-->

# mxfp8_quantize SM100: coarse WASP pipeline sketch

This non-executable design sketch describes the thread roles, control flow,
storage placement, and PTX-level operations of
[`tirx_kernels/flashinfer/quantization/mxfp8_quantize.py`](../../tirx_kernels/flashinfer/quantization/mxfp8_quantize.py).
That TIRx module is the authoritative implementation.

The instantiations are `DTYPE in {f16, bf16}` crossed with
`THREADS_PER_SF in {2, 4}` (host dispatch `m * (K/32) >= 65536`) crossed with
`SF_LAYOUT in {linear, 128x4, 8x4}` of the source CuTe-DSL classes
`MXFP8QuantizeLinearKernel` / `MXFP8QuantizeSwizzledKernel`, compiled for
sm_100a. Grid and block extents are static per config (the source host
computes them once per call from the same formulas, mirrored in the module);
`total_sf_blocks` (linear) / `M`, `padded_M` (swizzled) and all pointers stay
runtime ABI values, exactly like the source kernel signatures. Accepted target
is SM100/B200. The `enable_pdl=True` variant is ported as the same
compile-time knob (entry `griddepcontrol.wait`, exit
`griddepcontrol.launch_dependents`) but is off in all default configs: TVM
launches do not carry the PDL launch attribute, so test/bench parity pins
`enable_pdl=False` on both sides. The M-agnostic single-compile caching of the
source (one cubin per K) is a host-side JIT property and out of scope; tile
(`Tx`) primitives are out of scope. The source has no TMA/smem/mbarrier path
for MXFP8 — nothing is deferred.

## Pipeline at a glance

| Warps | Role-local program | Publication/reuse edges |
| --- | --- | --- |
| all (uniform) | **linear**: every thread runs the same single-role program: flat SF-block grid-stride loop — per 32-element SF block: 2x (2T) or 1x (4T) 16-byte loads, half2/bf16x2 absmax tree, 2/4-lane butterfly max, UE8M0 encode, inverse-scale bit build, 8 (2T) or 4 (4T) e4m3 pair-converts with packing, 2x (2T) or 1x (4T) 8-byte stores, one conditional SF byte store. **swizzled**: same per-block program, embedded in a row-based iteration with a compile-time `needs_col_loop` split and padding row/column SF zero-fill paths. | none — no SMEM, no mbarriers, no cross-CTA data; the only cross-thread edges are the 1-2 `shfl.sync.bfly.b32` max reduction inside each 2/4-lane SF group and the optional `griddepcontrol` pair |

There is no warp specialization and no producer/consumer split. The branches
are the runtime grid-stride loop guards, the `thread_in_sf == 0` SF-store
predicate, the compile-time `needs_col_loop` / multi-row split (swizzled), the
padding-row / padding-column zero-fill paths (swizzled), and the zero-input
selects inside the UE8M0 helpers.

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
copy_r2g_u64(src_u64, dst_addr)       # one 8-byte register -> global store
copy_r2g_u8(src_u8, dst_addr)         # one byte register -> global store (emitted st.global.b8)
```

The compute vocabulary is deliberately primitive. Multi-instruction source asm
blocks are written as subroutines whose bodies annotate each instruction; a
call site counts as one op tile of the subroutine's dominant PTX family:

```python
abs_h2(dst_b32, src_b32)         # and.b32 with 0x7FFF7FFF (source habs2/bfloat2_habs2
                                 #  clear both sign bits; same for f16 and bf16)
max_h2(dst_b32, lhs, rhs)        # max.f16x2 (f16) | max.bf16x2 (bf16)
shfl_bfly_f32(dst, src, lane_xor) # shfl.sync.bfly.b32, full membermask
fmax(dst, lhs, rhs)              # max.f32
mul(dst, lhs, rhs)               # mul.f32
cast_f32_h(dst, src)             # cvt.f32.f16 (unpack path for f16 pairs)
idiv / imod(dst, lhs, rhs)       # integer div/mod; constant divisors lower to
                                 #  mul-shift or shr/and families
move / bitops                    # mov, and, or, shl, shr, add, sub on b32/s32
select(dst, pred, a, b)          # selp family inside UE8M0 helpers
```

Subroutines (bodies annotated per instruction in the sketch):

```python
HMAX_REDUCE_F32(x_b32) -> f32        # pair-max extraction to f32
UE8M0_ENCODE(f) -> u32               # float_to_ue8m0_fast: IEEE bit math
UE8M0_INV_SCALE(u) -> f32            # ue8m0_to_inv_scale_fast: (254-u)<<23 bits
FP8X2_SCALED(h2_b32, inv_f32) -> u32 # unpack + scale + cvt.rn.satfinite.e4m3x2.f32
PACK_U64(b01, b23, b45, b67) -> u64  # pack_fp8x8_to_u64 byte gather
```

`thread_id`, `cta_id`, `grid_dim`, `pdl_wait`, and `pdl_launch_dependents` are
schedule operations. Address expressions, loop bounds, and guards are shown
directly; they do not hide copies, computation, role changes, or
synchronization. All global addresses are 64-bit (`get_ptr_as_int64` in the
source); integer division/modulo by the compile-time constant
`num_sf_blocks_per_row = K/32` lowers to mul-shift or shr/and families.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

variant = specialize(DTYPE=("f16", "bf16"), THREADS_PER_SF=(2, 4),
                     SF_LAYOUT=("linear", "128x4", "8x4"), ENABLE_PDL=False,
                     target="sm_100a")
# instruction_selection: none; extent: 2x2x3 compile-time instantiations (x PDL knob)

SF_VEC = 32                       # SF_VEC_SIZE (utils:37)
ELTS = 16 if THREADS_PER_SF == 2 else 8        # 2T: ELTS_PER_THREAD; 4T: _SMALL
SFBPW = 16 if THREADS_PER_SF == 2 else 8       # SF blocks per warp
GRID_X, BLOCK_X = host_launch_shape(M, K, SF_LAYOUT, THREADS_PER_SF)  # static per config
# instruction_selection: none; extent: static launch metadata

launch_config = launch(
    grid=(GRID_X, 1, 1),
    block=(BLOCK_X, 1, 1),
    launch_bounds=(1024, 4),        # max_number_threads=1024, min_blocks_per_mp=4
    dynamic_smem_bytes=0,
)
# instruction_selection: none; extent: static launch metadata

# ===========================================================================
# LINEAR kernel (mxfp8_quantize.py:220-330) — flat SF-block iteration
# BLOCK_X = 512 (16 warps) always; SF_BLOCKS_PER_TB = 16 * SFBPW
# ===========================================================================

def mxfp8_quantize_linear(
    input,          # DTYPE [M*K], direct global pointer
    output,         # u8 [M*K], direct global pointer
    scales,         # u8 [total_sf_blocks], direct global pointer
    total_sf_blocks,  # runtime i32 (= M * K/32)
):
    bx = cta_id(axis="x", extent=GRID_X)
    # instruction_selection: mov.u32 from %ctaid.x; extent: scalar per thread
    tx = thread_id(extent=BLOCK_X, dtype="uint32")
    # instruction_selection: mov.u32 from %tid.x; extent: scalar per thread
    gd = grid_dim(axis="x")
    # instruction_selection: mov.u32 from %nctaid.x; extent: scalar per thread

    pdl_wait()                        # only when ENABLE_PDL
    # instruction_selection: griddepcontrol.wait; extent: every thread, kernel entry

    warp_idx = idiv(tx, 32)
    lane_idx = imod(tx, 32)
    # instruction_selection: shr.b32/and.b32 (constant 32); extent: scalar per thread
    sf_idx_in_warp = idiv(lane_idx, THREADS_PER_SF)
    thread_in_sf = imod(lane_idx, THREADS_PER_SF)
    # instruction_selection: shr/and (constant 2|4); extent: scalar per thread

    sf_idx = bx * (16 * SFBPW) + warp_idx * SFBPW + sf_idx_in_warp
    # instruction_selection: mad.lo.s32 family; extent: scalar per thread
    while sf_idx < total_sf_blocks:
        # instruction_selection: setp.lt.s32 + bra; extent: loop control, no unroll
        row_idx = idiv(sf_idx, K // 32)
        col_idx = imod(sf_idx, K // 32)
        # instruction_selection: constant-divisor mul-shift family; extent: scalar per thread per iteration
        elem_idx = col_idx * SF_VEC + thread_in_sf * ELTS
        # instruction_selection: shl/mad by constants; extent: scalar per thread per iteration
        in_addr = i64(input) + (i64(row_idx) * K + elem_idx) * 2
        # instruction_selection: mul.wide.s32/mad.lo.s64 family; extent: scalar per thread per iteration

        v = reg_tile("b32", [8])      # 2T: 8 words live; 4T: 4 words used
        # instruction_selection: none; extent: per-thread registers
        copy_g2r_v4_u32(in_addr, v[0:4])
        # instruction_selection: ld.global.v4.u32; extent: one 16-byte vector load
        if THREADS_PER_SF == 2:
            copy_g2r_v4_u32(in_addr + 16, v[4:8])
            # instruction_selection: ld.global.v4.u32; extent: one 16-byte vector load
            max_all = ABSMAX_8(v[0:8])            # tree below; f16/bf16 forms
            # instruction_selection: see ABSMAX_8; extent: 8x and.b32 + 7x max.{f16,bf16}x2
            local_max = HMAX_REDUCE_F32(max_all)
            # instruction_selection: see HMAX_REDUCE_F32; extent: one pair to one f32
            global_max = fmax(local_max, shfl_bfly_f32(local_max, 1))
            # instruction_selection: shfl.sync.bfly.b32 + max.f32; extent: one butterfly round
        else:
            max_all = ABSMAX_4(v[0:4])
            # instruction_selection: see ABSMAX_4; extent: 4x and.b32 + 3x max.{f16,bf16}x2
            local_max = HMAX_REDUCE_F32(max_all)
            # instruction_selection: see HMAX_REDUCE_F32; extent: one pair to one f32
            t = fmax(local_max, shfl_bfly_f32(local_max, 1))
            # instruction_selection: shfl.sync.bfly.b32 + max.f32; extent: butterfly round 1
            global_max = fmax(t, shfl_bfly_f32(t, 2))
            # instruction_selection: shfl.sync.bfly.b32 + max.f32; extent: butterfly round 2

        scale_ue8m0_u32 = UE8M0_ENCODE(mul(global_max, 1.0 / 448.0))
        # instruction_selection: mul.f32 then see UE8M0_ENCODE; extent: one scalar
        inv_scale = UE8M0_INV_SCALE(scale_ue8m0_u32)
        # instruction_selection: see UE8M0_INV_SCALE; extent: one scalar

        out_addr = i64(output) + (i64(row_idx) * K + elem_idx)
        # instruction_selection: mad.lo.s64 family; extent: scalar per thread per iteration
        lo = FP8X8(v[0:4], inv_scale)
        # instruction_selection: see FP8X8; extent: 4x FP8X2_SCALED + PACK_U64 (8 elements)
        copy_r2g_u64(lo, out_addr)
        # instruction_selection: st.global.u64; extent: one 8-byte store
        if THREADS_PER_SF == 2:
            hi = FP8X8(v[4:8], inv_scale)
            # instruction_selection: see FP8X8; extent: second 8-element group
            copy_r2g_u64(hi, out_addr + 8)
            # instruction_selection: st.global.u64; extent: one 8-byte store

        if thread_in_sf == 0:
            # instruction_selection: setp.ne.s32 + bra (emitted polarity: skips the store); extent: per iteration
            copy_r2g_u8(low_u8(scale_ue8m0_u32), scales + sf_idx)
            # instruction_selection: st.global.b8; extent: one byte store per iteration

        sf_idx = sf_idx + gd * (16 * SFBPW)
        # instruction_selection: mad.lo.s32; extent: loop induction update

    pdl_launch_dependents()           # only when ENABLE_PDL
    # instruction_selection: griddepcontrol.launch_dependents; extent: every thread, kernel exit

# ===========================================================================
# SWIZZLED kernel (mxfp8_quantize.py:438-731) — row-based iteration
# BLOCK_X = _compute_optimal_warps(K, SFBPW) * 32 in [128, 1024];
# compile-time split: needs_col_loop (num_sf_blocks_per_row > col_units)
# ===========================================================================

def mxfp8_quantize_swizzled(
    input,          # DTYPE [M*K], direct global pointer
    output,         # u8 [M*K], direct global pointer
    scales,         # u8 [padded_M * padded_sf_cols], direct global pointer
    M,              # runtime i32
    padded_M,       # runtime i32 (round_up(M, 128|8))
):
    bx = cta_id(axis="x", extent=GRID_X)
    # instruction_selection: mov.u32 from %ctaid.x; extent: scalar per thread
    tx = thread_id(extent=BLOCK_X, dtype="uint32")
    # instruction_selection: mov.u32 from %tid.x; extent: scalar per thread
    gd = grid_dim(axis="x")
    # instruction_selection: mov.u32 from %nctaid.x; extent: scalar per thread

    pdl_wait()                        # only when ENABLE_PDL
    # instruction_selection: griddepcontrol.wait; extent: every thread, kernel entry

    NSB = K // 32                     # num_sf_blocks_per_row (compile-time)
    PAD_COLS = (NSB + 3) // 4 * 4     # padded_sf_cols (compile-time)

    if needs_col_loop:                # compile-time branch (large K)
        col_unit_idx = idiv(tx, THREADS_PER_SF)
        thread_in_unit = imod(tx, THREADS_PER_SF)
        # instruction_selection: shr/and by constants; extent: scalar per thread
        col_units_per_block = BLOCK_X // THREADS_PER_SF

        row_idx = bx
        while row_idx < padded_M:
            # instruction_selection: setp.lt.s32 + bra; extent: row loop control
            if row_idx >= M:
                # instruction_selection: setp.lt.s32 + bra (padding-row arm); extent: per row iteration
                sf_col = col_unit_idx
                while sf_col < PAD_COLS:
                    # instruction_selection: setp/bra; extent: zero-fill loop control
                    if thread_in_unit == 0:
                        copy_r2g_u8(0, scales + SF_OFFSET(row_idx, sf_col, PAD_COLS))
                        # instruction_selection: st.global.b8; extent: one byte store (address math per SF_OFFSET)
                    sf_col = sf_col + col_units_per_block
            else:
                sf_col = col_unit_idx
                while sf_col < NSB:
                    # instruction_selection: setp/bra; extent: column loop control
                    QUANTIZE_BLOCK(row_idx, sf_col, thread_in_unit)   # same body as linear per-block program
                    # instruction_selection: see linear kernel body; extent: one SF block
                    if thread_in_unit == 0:
                        copy_r2g_u8(low_u8(scale_ue8m0_u32),
                                    scales + SF_OFFSET(row_idx, sf_col, PAD_COLS))
                        # instruction_selection: st.global.b8; extent: one byte store per iteration
                    sf_col = sf_col + col_units_per_block
                sf_col = NSB + col_unit_idx
                while sf_col < PAD_COLS:      # padding columns of a data row
                    # instruction_selection: setp/bra; extent: zero-fill loop control
                    if thread_in_unit == 0:
                        copy_r2g_u8(0, scales + SF_OFFSET(row_idx, sf_col, PAD_COLS))
                        # instruction_selection: st.global.b8; extent: one byte store
                    sf_col = sf_col + col_units_per_block
            row_idx = row_idx + gd
            # instruction_selection: add.s32; extent: row induction update
    else:
        # multi-row path (small K): ROWS_PER_BLOCK = col_units // NSB rows per block iteration
        threads_per_row = NSB * THREADS_PER_SF
        row_in_block = idiv(tx, threads_per_row)
        local_tidx = imod(tx, threads_per_row)
        sf_col_idx = idiv(local_tidx, THREADS_PER_SF)
        thread_in_unit = imod(local_tidx, THREADS_PER_SF)
        # instruction_selection: constant-divisor shr/and or mul-shift family; extent: scalar per thread

        row_batch_idx = bx
        row_idx = row_batch_idx * ROWS_PER_BLOCK + row_in_block
        while row_batch_idx * ROWS_PER_BLOCK < padded_M:
            # instruction_selection: setp.lt.s32 + bra; extent: batch loop control
            if row_idx < padded_M:
                # instruction_selection: setp.ge.s32 + bra (emitted polarity: skips body); extent: per iteration
                if row_idx >= M:
                    # padding row: zero ALL padded SF columns; stride is NSB (source :620)
                    if thread_in_unit == 0:
                        pad_col = sf_col_idx
                        while pad_col < PAD_COLS:
                            # instruction_selection: setp/bra; extent: zero-fill loop control
                            copy_r2g_u8(0, scales + SF_OFFSET(row_idx, pad_col, PAD_COLS))
                            # instruction_selection: st.global.b8; extent: one byte store
                            pad_col = pad_col + NSB
                else:
                    if sf_col_idx < NSB:
                        # provably true in the multi-row path
                        # (sf_col_idx = (tidx % (NSB*TPS)) // TPS < NSB by construction);
                        # compile-time folded — no setp/bra emitted in the source PTX
                        QUANTIZE_BLOCK(row_idx, sf_col_idx, thread_in_unit)
                        # instruction_selection: see linear kernel body; extent: one SF block
                        if thread_in_unit == 0:
                            copy_r2g_u8(low_u8(scale_ue8m0_u32),
                                        scales + SF_OFFSET(row_idx, sf_col_idx, PAD_COLS))
                            # instruction_selection: st.global.b8; extent: one byte store
                    if NSB != PAD_COLS:       # compile-time; padding SF columns of data row
                        if thread_in_unit == 0:
                            pad_col = NSB + sf_col_idx
                            while pad_col < PAD_COLS:
                                # instruction_selection: setp/bra; extent: zero-fill loop control
                                copy_r2g_u8(0, scales + SF_OFFSET(row_idx, pad_col, PAD_COLS))
                                # instruction_selection: st.global.b8; extent: one byte store
                                pad_col = pad_col + NSB
            row_batch_idx = row_batch_idx + gd
            row_idx = row_batch_idx * ROWS_PER_BLOCK + row_in_block
            # instruction_selection: mad.lo.s32; extent: batch induction update

    pdl_launch_dependents()           # only when ENABLE_PDL
    # instruction_selection: griddepcontrol.launch_dependents; extent: every thread, kernel exit

# ===========================================================================
# QUANTIZE_BLOCK — the per-SF-block program shared by both kernels
# (linear :261-324, swizzled col-loop :494-575, multi-row :624-709).
# Loads ELTS elements at (row_idx, sf_col*32 + thread_in_unit*ELTS), quantizes
# against the 2/4-lane reduced UE8M0 scale, stores 8-byte group(s) and leaves
# scale_ue8m0_u32 live for the caller's conditional SF store.
# ===========================================================================

def QUANTIZE_BLOCK(row_idx, sf_col, thread_in_unit):
    elem_idx = sf_col * SF_VEC + thread_in_unit * ELTS
    # instruction_selection: shl/mad by constants; extent: scalar
    in_addr = i64(input) + (i64(row_idx) * K + elem_idx) * 2
    # instruction_selection: mul.wide.s32/mad.lo.s64 family; extent: scalar
    v = reg_tile("b32", [8])
    # instruction_selection: none; extent: per-thread registers
    copy_g2r_v4_u32(in_addr, v[0:4])
    # instruction_selection: ld.global.v4.u32; extent: one 16-byte vector load
    if THREADS_PER_SF == 2:
        copy_g2r_v4_u32(in_addr + 16, v[4:8])
        # instruction_selection: ld.global.v4.u32; extent: one 16-byte vector load
        max_all = ABSMAX_8(v[0:8])
        # instruction_selection: see ABSMAX_8; extent: 8x and.b32 + 7x max.{f16,bf16}x2
        global_max = REDUCE_MAX_2T(HMAX_REDUCE_F32(max_all))
        # instruction_selection: shfl.sync.bfly.b32 + max.f32; extent: one butterfly round
    else:
        max_all = ABSMAX_4(v[0:4])
        # instruction_selection: see ABSMAX_4; extent: 4x and.b32 + 3x max.{f16,bf16}x2
        global_max = REDUCE_MAX_4T(HMAX_REDUCE_F32(max_all))
        # instruction_selection: 2x (shfl.sync.bfly.b32 + max.f32); extent: two butterfly rounds
    scale_ue8m0_u32 = UE8M0_ENCODE(mul(global_max, 1.0 / 448.0))
    # instruction_selection: mul.f32 then see UE8M0_ENCODE; extent: one scalar
    inv_scale = UE8M0_INV_SCALE(scale_ue8m0_u32)
    # instruction_selection: see UE8M0_INV_SCALE; extent: one scalar
    out_addr = i64(output) + (i64(row_idx) * K + elem_idx)
    # instruction_selection: mad.lo.s64 family; extent: scalar
    copy_r2g_u64(FP8X8(v[0:4], inv_scale), out_addr)
    # instruction_selection: see FP8X8 + st.global.u64; extent: one 8-byte store
    if THREADS_PER_SF == 2:
        copy_r2g_u64(FP8X8(v[4:8], inv_scale), out_addr + 8)
        # instruction_selection: see FP8X8 + st.global.u64; extent: one 8-byte store
    # scale_ue8m0_u32 remains live; caller stores it when thread_in_unit == 0

# ===========================================================================
# Subroutines — bodies in source order, one annotation per instruction
# ===========================================================================

def ABSMAX_8(v):                      # half2_max_abs_8 (utils:691) / bfloat2_max_abs_8 (utils:728)
    a = reg_tile("b32", [8])
    for i in static_range(8):
        a[i] = abs_h2(v[i])
        # instruction_selection: and.b32 (mask 0x7FFF7FFF); extent: 8 packed pairs
    m01 = max_h2(a[0], a[1]); m23 = max_h2(a[2], a[3])
    m45 = max_h2(a[4], a[5]); m67 = max_h2(a[6], a[7])
    # instruction_selection: max.f16x2 | max.bf16x2; extent: 4 packed-pair maxima
    m03 = max_h2(m01, m23); m47 = max_h2(m45, m67)
    # instruction_selection: max.f16x2 | max.bf16x2; extent: 2 packed-pair maxima
    return max_h2(m03, m47)
    # instruction_selection: max.f16x2 | max.bf16x2; extent: 1 packed-pair maximum

def ABSMAX_4(v):                      # half2_max_abs_4 (utils:616) / bfloat2_max_abs_4 (utils:633)
    a0 = abs_h2(v[0]); a1 = abs_h2(v[1]); a2 = abs_h2(v[2]); a3 = abs_h2(v[3])
    # instruction_selection: and.b32 (mask 0x7FFF7FFF); extent: 4 packed pairs
    m01 = max_h2(a0, a1); m23 = max_h2(a2, a3)
    # instruction_selection: max.f16x2 | max.bf16x2; extent: 2 packed-pair maxima
    return max_h2(m01, m23)
    # instruction_selection: max.f16x2 | max.bf16x2; extent: 1 packed-pair maximum

def HMAX_REDUCE_F32(x):               # hmax_reduce_to_f32 (utils:95) / bfloat2_hmax_reduce_to_f32 (utils:122)
    if DTYPE == "f16":
        h0, h1 = unpack_b16x2(x)
        # instruction_selection: mov.b32 {h0,h1}, x; extent: one pair unpack
        f0 = cast_f32_h(h0); f1 = cast_f32_h(h1)
        # instruction_selection: cvt.f32.f16 x2; extent: two scalars
    else:
        lo = (x & 0xFFFF) << 16; hi = (x >> 16) << 16
        # instruction_selection: and.b32 (mask 0xFFFF) + shr.b32 (16) + shl.b32 x2 (16); extent: one pair unpack
        f0 = b32_to_f32(lo); f1 = b32_to_f32(hi)
        # instruction_selection: mov.b32 x2; extent: two scalars
    return fmax(f0, f1)
    # instruction_selection: max.f32; extent: one scalar

def UE8M0_ENCODE(f):                  # float_to_ue8m0_fast (utils:157)
    p_zero = f <= 0.0
    # instruction_selection: setp.le.f32; extent: one predicate
    bits = b32(f); exp = (bits >> 23) & 255; mant = bits & 0x7FFFFF
    # instruction_selection: mov.b32 + shr.b32 + and.b32 x2; extent: one scalar
    bump = select(mant != 0, 1, 0)
    # instruction_selection: setp.ne.u32 + selp.u32; extent: one scalar
    tiny_sub = (exp == 0) & (mant <= 0x400000)
    # instruction_selection: setp.eq.u32 + setp.le.u32 + and.pred; extent: one scalar
    if tiny_sub:
        bump = 0
    # instruction_selection: @pred mov.u32; extent: one predicated move
    result = exp + bump
    # instruction_selection: add.u32; extent: one scalar
    result = select(result > 254, 254, result)
    # instruction_selection: setp.gt.u32 + selp.u32; extent: one scalar
    return select(p_zero, 0, result)
    # instruction_selection: selp.u32; extent: one scalar

def UE8M0_INV_SCALE(u):               # ue8m0_to_inv_scale_fast (utils:207)
    new_exp = max(254 - u, 0)
    # instruction_selection: sub.s32 + max.s32; extent: one scalar
    f_bits = new_exp << 23
    # instruction_selection: shl.b32 + mov.b32; extent: one scalar
    if u == 0:
        f_bits = 0
    # instruction_selection: setp.eq.u32 + @pred mov.b32; extent: one predicated move
    return f32(f_bits)
    # instruction_selection: none (register reinterpret); extent: one scalar

def FP8X2_SCALED(h2, inv):            # half2_to_fp8x2_scaled (utils:249) / bfloat2_to_fp8x2_scaled (utils:286)
    if DTYPE == "f16":
        h0, h1 = unpack_b16x2(h2)
        # instruction_selection: mov.b32 {h0,h1}, h2; extent: one pair unpack
        f0 = cast_f32_h(h0); f1 = cast_f32_h(h1)
        # instruction_selection: cvt.f32.f16 x2; extent: two scalars
    else:
        lo = (h2 & 0xFFFF) << 16; hi = (h2 >> 16) << 16
        # instruction_selection: and.b32 (mask 0xFFFF) + shr.b32 (16) + shl.b32 x2 (16); extent: one pair unpack
        f0 = b32_to_f32(lo); f1 = b32_to_f32(hi)
        # instruction_selection: mov.b32 x2; extent: two scalars
    f0 = mul(f0, inv); f1 = mul(f1, inv)
    # instruction_selection: mul.f32 x2; extent: two scalars
    pair = cast_e4m3x2(f1, f0)        # hi operand first (source operand order)
    # instruction_selection: cvt.rn.satfinite.e4m3x2.f32; extent: one packed pair
    return zext_u16_to_u32(pair)
    # instruction_selection: cvt.u32.u16; extent: one scalar

def FP8X8(v4, inv):                   # half2x4_to_fp8x8_packed (utils:650) / bfloat2x4 (:668)
    b01 = FP8X2_SCALED(v4[0], inv)    # + 3 more identical calls on v4[1..3]
    # instruction_selection: see FP8X2_SCALED; extent: 4 packed pairs (8 elements)
    return PACK_U64(b01, b23, b45, b67)

def PACK_U64(b01, b23, b45, b67):     # pack_fp8x8_to_u64 (utils:326)
    lo = (b01 & 0xFFFF) | ((b23 & 0xFFFF) << 16)
    hi = (b45 & 0xFFFF) | ((b67 & 0xFFFF) << 16)
    # instruction_selection: and.b32 x4 + shl.b32 x2 + or.b32 x2; extent: two words
    return pack_b32x2(lo, hi)
    # instruction_selection: mov.b64 {lo, hi}; extent: one u64

def SF_OFFSET(row, col, padded_cols): # 128x4: compute_sf_index_swizzled_128x4_gpu (utils:535)
                                      # 8x4:  compute_sf_index_swizzled_8x4_gpu (utils:569)
    if SF_LAYOUT == "128x4":
        return (col % 4 + (col // 4) * 512 + (row % 32) * 16
                + ((row % 128) // 32) * 4 + (row // 128) * (128 * padded_cols))
        # instruction_selection: and/shr by constants + mad.lo.s32 family; extent: one scalar per SF store
    else:  # 8x4: [mTiles, kTiles, 8, 4] tiles of 32 bytes
        num_k_tiles = (padded_cols + 3) // 4
        return ((row // 8) * (num_k_tiles * 32) + (col // 4) * 32
                + (row % 8) * 4 + col % 4)
        # instruction_selection: and/shr by constants + mad.lo.s32 family; extent: one scalar per SF store
```

## Host wrapper and validation

The Python module performs host-only work; none of it emits device PTX:

```python
def host_launch_shape(m, k, sf_layout, threads_per_sf):
    # linear (mxfp8_quantize.py:957-967): 512 threads; SF_BLOCKS_PER_TB = 256|128
    #   grid = min(ceildiv(m*(k/32), SF_BLOCKS_PER_TB), SM_COUNT * 4)
    # swizzled (:936-948): warps = _compute_optimal_warps(k, sfbpw) in [4..32]
    #   rows_per_block = col_units // (k/32) or 1 (needs_col_loop);
    #   grid = min(ceildiv(padded_m, rows_per_block), SM_COUNT * 4)
    # instruction_selection: none; extent: host-only arithmetic (SM_COUNT=148 on B200)

use_2t = m * (k // 32) >= 65536       # source dispatch (:933-934)

def prepare_data(dtype, m, k, sf_layout, enable_pdl):
    host_assert(dtype in ("float16", "bfloat16") and k % 32 == 0 and m >= 1)
    a = contiguous_seeded_randn(shape=(m, k), dtype=dtype)   # seed 42
    # instruction_selection: none; extent: tensor constructions

# run_test compares the packed FP8 output and the full SF buffer (source
# zero-fills padding rows/columns, so the whole padded buffer is compared)
# against flashinfer.quantization.mxfp8_quantize(backend="cute-dsl",
# enable_pdl=False), exact byte equality (rtol=0, atol=0).
# run_bench times the primfunc launch (tirx) against the cached compiled
# source kernel (_get_compiled_kernel_mxfp8_{linear,swizzled}) with
# preallocated outputs, enable_pdl=False; both closures are no-argument.
```

## Static specialization boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| `DTYPE` | static per config | selects f16/bf16 unpack (`mov.b32 {h0,h1}` + `cvt.f32.f16` vs `and/shr/shl` + `mov.b32`) and `max.f16x2` vs `max.bf16x2` |
| `THREADS_PER_SF` (2T/4T) | static per config (host dispatch `m*(K/32) >= 65536`) | 16 vs 8 elements per thread, 1 vs 2 butterfly rounds, 2 vs 1 output stores, SF blocks per warp 16 vs 8 |
| `SF_LAYOUT` | static per config | selects linear vs swizzled kernel body and the SF index math (128x4 vs 8x4) |
| `K` | static per config | `NSB = K/32`, `PAD_COLS`, swizzled warps/rows_per_block/needs_col_loop constant-fold |
| grid/block extents | static per config | host formulas baked in; kernel reads `%ctaid.x`/`%tid.x`/`%nctaid.x` only |
| `total_sf_blocks` / `M`, `padded_M` | runtime i32 ABI | loop bounds and row predicates stay runtime like the source |
| `ENABLE_PDL` | static per config (default False) | griddepcontrol pair present only in the PDL instantiation; TVM launch carries no PDL attribute |
| `launch_bounds (1024, 4)` | static | register budget contract (max_number_threads, min_blocks_per_mp) |

## TIRx module and benchmark contract

- `KERNEL_META = {"name": "mxfp8_quantize", "category": "flashinfer",
  "compute_capability": 10}`.
- The executable kernel is expressed entirely in plain TIRx: explicit `while`
  grid-stride loops, runtime scalar ABI, register tiles, and native `T.ptx.*`
  forms for every non-trivial instruction (`ld.global.v4.u32`,
  `st.global.u64`/`st.global.b8`, `max.f16x2`/`max.bf16x2`, `max.f32`,
  `shfl.sync.bfly.b32`, `cvt.rn.satfinite.e4m3x2.f32`, `cvt.u32.u16`,
  `mov.b32`/`mov.b64` packs, `setp`/`selp` families, `griddepcontrol`).
  Integer bit math of the UE8M0 helpers is plain TIRx b32 ops, matching the
  source asm instruction for instruction. There is no `T.cuda.func_call` and
  no `Tx` tile primitives anywhere in the pre-dispatch IR.
- `get_kernel(dtype, m, k, sf_layout, enable_pdl)` returns the specialized
  primfunc with static grid/block; `prepare_data`, `run_test`, `run_bench`
  follow the repository contract.
- The timed implementation is named `tirx`; the reference is the cached
  compiled CuTe-DSL source kernel launched with `enable_pdl=False` and
  preallocated outputs. Allocation, compilation, and correctness checks stay
  outside timing.
- Correctness compares the full packed output and the full padded SF buffer
  against the source wrapper, exact byte equality.

## Instruction selection is a lowering consequence

The sketch above never requests a hardware instruction beyond the documented
PTX helpers. The following lowering families follow from storage direction,
shape, dtype, and schedule. PTX names are taken from the source inline-asm
blocks (`ld.global.v4.u32`, `st.global.u64`, `cvt.rn.satfinite.e4m3x2.f32`,
`shfl.sync.bfly.b32`, `max.f32`); per-iteration counts below are the expected
values the sketch-reviewer verifies against a fresh line-info PTX export of
the exact source specializations (`CUTE_DSL_KEEP=ptx CUTE_DSL_LINEINFO=1`,
enable_pdl=False).

| Primitive/schedule pattern | PTX family (expected, per SF block, 2T path) |
| --- | --- |
| `copy_g2r_v4_u32` loads | `ld.global.v4.u32` x2 (4T: x1; no `.nc`) |
| absmax tree | `and.b32` x8 + `max.f16x2` x7 (bf16: `max.bf16x2`; 4T: x4 + x3) |
| pair-max to f32 | f16: `mov.b32 {h0,h1}` + `cvt.f32.f16` x2 + `max.f32`; bf16: `and.b32` + `shr.b32` + `shl.b32` x2 + `mov.b32` x2 + `max.f32` |
| cross-lane max | `shfl.sync.bfly.b32` + `max.f32` (2T: x1 each; 4T: x2 each) |
| UE8M0 encode | `mul.f32` x1 + `setp.le.f32` + `mov.b32` + `shr.b32` + `and.b32` x2 + `setp.ne.u32` + `selp.u32` + `setp.eq.u32` + `setp.le.u32` + `and.pred` + `@p mov.u32` + `add.u32` + `setp.gt.u32` + `selp.u32` x2 |
| inverse scale | `sub.s32` + `max.s32` + `shl.b32` + `mov.b32` + `setp.eq.u32` + `@p mov.b32` |
| FP8 convert + scale (per 8 elements) | f16: `mov.b32 {h0,h1}` x4 + `cvt.f32.f16` x8; bf16 (per pair): `and.b32` + `shr.b32` + `shl.b32` x2 + `mov.b32` x2 (x4 pairs); `mul.f32` x8 + `cvt.rn.satfinite.e4m3x2.f32` x4 + `cvt.u32.u16` x4 (2T: two groups) |
| u64 pack (per 8 elements) | `and.b32` x4 + `shl.b32` x2 + `or.b32` x2 + `mov.b64` |
| output stores | `st.global.u64` x2 (4T: x1) |
| SF store | `st.global.b8` x1 (predicated on `thread_in_sf == 0`) |
| SF swizzle address | `and.b32`/`shr.b32` by constants + `mad.lo.s32` family (swizzled layouts only) |
| row/col decode + addresses | constant-divisor mul-shift family, `mul.wide.s32`/`mad.lo.s64` 64-bit address math |
| loop control | `setp` + `bra`, `add.s32`/`mad.lo.s32` induction |
| PDL (ENABLE_PDL only) | `griddepcontrol.wait` x1, `griddepcontrol.launch_dependents` x1 (whole kernel) |

The bf16 instantiations differ only in the unpack sequences and the
`max.bf16x2` opcode. The 8x4 instantiation differs from 128x4 only in the
SF_OFFSET arithmetic. The 4T instantiations halve the per-thread element
count, add the second butterfly round, and use one output store.
