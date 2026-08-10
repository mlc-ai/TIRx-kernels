<!--
Copyright (c) 2026 The TIRX Authors

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
flashinfer/quantization/kernels/mxfp4_quantize.py
(MXFP4QuantizeLinearKernel / MXFP4QuantizeSwizzledKernel), the
CuTe-DSL SM100 kernels behind
flashinfer.quantization.mxfp4_quantize(backend="cute-dsl").
-->

# mxfp4_quantize SM100: coarse WASP pipeline sketch

This non-executable design sketch describes the thread roles, control flow,
storage placement, and PTX-level operations of
[`tirx_kernels/flashinfer/mxfp4_quantize.py`](../../tirx_kernels/flashinfer/mxfp4_quantize.py).
That TIRx module is the authoritative implementation.

The instantiations are `DTYPE in {f16, bf16}` crossed with
`SF_LAYOUT in {linear, 128x4, 8x4}` of the source CuTe-DSL classes
`MXFP4QuantizeLinearKernel` / `MXFP4QuantizeSwizzledKernel` in the 1T/SF
thread configuration (`use_4t_per_sf=False`), compiled for sm_100a. The source
dispatches 4T/SF only when `num_sm <= 80` (mxfp4_quantize.py:956); the
accepted target is SM100/B200 (148 SMs), so 4T/SF is unreachable and out of
scope. Grid and block extents are static per config (the source host computes
them once per call from the same formulas, mirrored in the module); `M`,
`total_sf_blocks` (linear) / `M`, `padded_M` (swizzled) and all pointers stay
runtime ABI values, exactly like the source kernel signatures. The
`enable_pdl=True` variant is ported as the same compile-time knob (entry
`griddepcontrol.wait`, exit `griddepcontrol.launch_dependents`) but is off in
all default configs: TVM launches do not carry the PDL launch attribute, so
test/bench parity pins `enable_pdl=False` on both sides. The M-agnostic
single-compile caching of the source is a host-side JIT property and out of
scope; tile (`Tx`) primitives are out of scope. The source has no
TMA/smem/mbarrier path for MXFP4 — nothing else is deferred.

## Pipeline at a glance

| Warps | Role-local program | Publication/reuse edges |
| --- | --- | --- |
| all (uniform) | **linear**: every thread runs the same single-role program: flat SF-block grid-stride loop — per 32-element SF block: four 16-byte loads, a 16-word half2/bf16x2 absmax tree, pair-max to f32, UE8M0 encode against `rcp.approx.ftz(6.0)`, inverse-scale bit build, 16 unpack-scale pairs, 4x e2m1x8 converts, 2x u64 pack, one unpredicated SF byte store, then two 8-byte output stores (source order). **swizzled**: same per-block program, embedded in row-based iteration with a compile-time `needs_col_loop` split and padding row/column SF zero-fill paths (no lane predicate in 1T/SF). | none — no SMEM, no mbarriers, no cross-thread data at all (1T/SF has no shuffle reduction); the only ordering edges are the optional `griddepcontrol` pair |

There is no warp specialization and no producer/consumer split. The branches
are the runtime grid-stride loop guards, the compile-time `needs_col_loop` /
multi-row split (swizzled), the padding-row / padding-column zero-fill paths
(swizzled), and the zero-input selects inside the UE8M0 helpers.

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
abs_h2(dst_b32, src_b32)         # and.b32 with 0x7FFF7FFF (same for f16 and bf16)
max_h2(dst_b32, lhs, rhs)        # max.f16x2 (f16) | max.bf16x2 (bf16)
rcp_ftz(dst, src)                # rcp.approx.ftz.f32
mul(dst, lhs, rhs)               # mul.f32 (non-ftz, as the source asm)
cast_e2m1x2(dst_u8, hi, lo)      # cvt.rn.satfinite.e2m1x2.f32 (hi operand first)
select(dst, pred, a, b)          # selp family inside UE8M0 helpers
setp(dst_pred, lhs, rhs)         # setp.le.f32 / setp.{ne,eq,le,gt}.u32
move / bitops                    # mov, and, or, shl, shr, add, sub on b32/s32/b64
idiv / imod(dst, lhs, rhs)       # integer div/mod; constant divisors lower to
                                 #  mul-shift or shr/and families
```

Subroutines (bodies annotated per instruction in the sketch):

```python
ABSMAX_8(v[8]) -> b32                # half2_max_abs_8 / bfloat2_max_abs_8
HMAX_REDUCE_F32(x_b32) -> f32        # pair-max extraction to f32
UE8M0_ENCODE(f) -> u32               # float_to_ue8m0_fast: IEEE bit math
UE8M0_INV_SCALE(u) -> f32            # ue8m0_to_inv_scale_fast: (254-u)<<23 bits
FLOAT2_SCALED(h2_b32, inv_f32) -> (f32, f32)  # half2/bfloat2_to_float2_scaled
E2M1X8(s[8]) -> u32                  # cvt_e2m1x8_f32: 4x e2m1x2 + byte pack
PROCESS_BLOCK(row, col) -> (ue_u32, u64, u64)  # process_mxfp4_block_half/bfloat (1T/SF; no stores)
```

`thread_id`, `cta_id`, `grid_dim`, `pdl_wait`, and `pdl_launch_dependents` are schedule
operations. Address expressions, loop bounds, and guards are shown directly;
they do not hide copies, computation, role changes, or synchronization. All
global addresses are 64-bit (`get_ptr_as_int64` in the source); integer
division/modulo by the compile-time constant `num_sf_blocks_per_row = K/32`
lowers to mul-shift or shr/and families.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

variant = specialize(DTYPE=("f16", "bf16"), SF_LAYOUT=("linear", "128x4", "8x4"),
                     THREADS_PER_SF=1, ENABLE_PDL=False, target="sm_100a")
# instruction_selection: none; extent: 2x3 compile-time instantiations (x PDL knob)

SF_VEC = 32                       # MXFP4_SF_VEC_SIZE
GRID_X, BLOCK_X = host_launch_shape(M, K, SF_LAYOUT)  # static per config
# instruction_selection: none; extent: static launch metadata

launch_config = launch(
    grid=(GRID_X, 1, 1),
    block=(BLOCK_X, 1, 1),
    launch_bounds=(1024, 4),        # max_number_threads=1024, min_blocks_per_mp=4
    dynamic_smem_bytes=0,
)
# instruction_selection: none; extent: static launch metadata

# ===========================================================================
# LINEAR kernel (mxfp4_quantize.py:328-375, 1T/SF arm)
# BLOCK_X = 512 (16 warps) always; SF_BLOCKS_PER_TB = 512; one SF block/thread
# ===========================================================================

def mxfp4_quantize_linear(
    input,          # DTYPE [M*K], direct global pointer
    output,         # u8 [M*K/2], direct global pointer
    scales,         # u8 [total_sf_blocks], direct global pointer
    M,              # runtime i32 (unused by the 1T body; source ABI parity)
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

    sf_idx = bx * 512 + tx
    # instruction_selection: mad.lo.s32 family; extent: scalar per thread
    while sf_idx < total_sf_blocks:
        # instruction_selection: setp.lt.s32 + bra; extent: loop control, no unroll
        row_idx = idiv(sf_idx, K // 32)
        col_idx = imod(sf_idx, K // 32)
        # instruction_selection: constant-divisor mul-shift family; extent: scalar per thread per iteration

        scale_ue8m0_u32, packed64_0, packed64_1 = PROCESS_BLOCK(row_idx, col_idx)
        # instruction_selection: see PROCESS_BLOCK; extent: one 32-element SF block

        # SF store: linear index == sf_idx (compute_sf_index_linear_gpu), unpredicated;
        # source order: SF byte store BEFORE the two output stores (:365 then :372-373)
        copy_r2g_u8(low_u8(scale_ue8m0_u32), scales + sf_idx)
        # instruction_selection: st.global.b8; extent: one byte store per iteration

        out_base = i64(row_idx) * (K // 2) + col_idx * 16
        # instruction_selection: mad.lo.s64 family; extent: scalar per thread per iteration
        copy_r2g_u64(packed64_0, i64(output) + out_base)
        # instruction_selection: st.global.u64; extent: one 8-byte store
        copy_r2g_u64(packed64_1, i64(output) + out_base + 8)
        # instruction_selection: st.global.u64; extent: one 8-byte store

        sf_idx = sf_idx + gd * 512
        # instruction_selection: mad.lo.s32; extent: loop induction update

    pdl_launch_dependents()           # only when ENABLE_PDL
    # instruction_selection: griddepcontrol.launch_dependents; extent: every thread, kernel exit

# ===========================================================================
# SWIZZLED kernel (mxfp4_quantize.py:476-798, 1T/SF arms)
# BLOCK_X = _compute_optimal_threads(K) in [128, 512];
# compile-time split: needs_col_loop (num_sf_blocks_per_row > 512)
# ===========================================================================

def mxfp4_quantize_swizzled(
    input,          # DTYPE [M*K], direct global pointer
    output,         # u8 [M*K/2], direct global pointer
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

    if needs_col_loop:                # compile-time branch (K/32 > 512)
        # 1T: col_unit == tidx (mxfp4_quantize.py:515-517)
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
                    scale_ue8m0_u32, packed64_0, packed64_1 = PROCESS_BLOCK(row_idx, sf_col)
                    # instruction_selection: see PROCESS_BLOCK; extent: one SF block
                    # source order: SF byte store BEFORE the two output stores (:624 then :629-630)
                    copy_r2g_u8(low_u8(scale_ue8m0_u32),
                                scales + SF_OFFSET(row_idx, sf_col, PAD_COLS))
                    # instruction_selection: st.global.b8; extent: one byte store per iteration
                    out_base = i64(row_idx) * (K // 2) + sf_col * 16
                    # instruction_selection: mad.lo.s64 family; extent: scalar per iteration
                    copy_r2g_u64(packed64_0, i64(output) + out_base)
                    # instruction_selection: st.global.u64; extent: one 8-byte store
                    copy_r2g_u64(packed64_1, i64(output) + out_base + 8)
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
        # multi-row path: 1T -> threads_per_row == NSB, thread_in_sf == 0 always
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
                    # padding row: zero ALL padded SF columns; stride is threads_per_row == NSB (:676-682)
                    local_sf = sf_idx_in_row
                    while local_sf < PAD_COLS:
                        # instruction_selection: setp/bra; extent: zero-fill loop control
                        copy_r2g_u8(0, scales + SF_OFFSET(row_idx, local_sf, PAD_COLS))
                        # instruction_selection: st.global.b8; extent: one byte store
                        local_sf = local_sf + NSB
                else:
                    if sf_idx_in_row < NSB:
                        # provably true in the 1T multi-row path
                        # (sf_idx_in_row = tidx % NSB < NSB by construction);
                        # compile-time folded — no setp/bra emitted in the source PTX
                        scale_ue8m0_u32, packed64_0, packed64_1 = PROCESS_BLOCK(row_idx, sf_idx_in_row)
                        # instruction_selection: see PROCESS_BLOCK; extent: one SF block
                        # source order: SF byte store BEFORE the two output stores (:771 then :778-779)
                        copy_r2g_u8(low_u8(scale_ue8m0_u32),
                                    scales + SF_OFFSET(row_idx, sf_idx_in_row, PAD_COLS))
                        # instruction_selection: st.global.b8; extent: one byte store
                        out_base = i64(row_idx) * (K // 2) + sf_idx_in_row * 16
                        # instruction_selection: mad.lo.s64 family; extent: scalar per iteration
                        copy_r2g_u64(packed64_0, i64(output) + out_base)
                        # instruction_selection: st.global.u64; extent: one 8-byte store
                        copy_r2g_u64(packed64_1, i64(output) + out_base + 8)
                        # instruction_selection: st.global.u64; extent: one 8-byte store
                    if NSB != PAD_COLS:       # compile-time; padding SF columns of data row
                        pad_col = NSB + sf_idx_in_row
                        while pad_col < PAD_COLS:
                            # instruction_selection: setp/bra; extent: zero-fill loop control
                            copy_r2g_u8(0, scales + SF_OFFSET(row_idx, pad_col, PAD_COLS))
                            # instruction_selection: st.global.b8; extent: one byte store
                            pad_col = pad_col + NSB   # stride threads_per_row == NSB (:791)
            row_batch_idx = row_batch_idx + gd
            row_idx = row_batch_idx * ROWS_PER_BLOCK + row_in_block
            # instruction_selection: mad.lo.s32; extent: batch induction update

    pdl_launch_dependents()           # only when ENABLE_PDL
    # instruction_selection: griddepcontrol.launch_dependents; extent: every thread, kernel exit

# ===========================================================================
# PROCESS_BLOCK — process_mxfp4_block_half (utils:765) / _bfloat (utils:839),
# the 1T/SF per-block program shared by both kernels. Loads 32 elements at
# (row_idx, col*32), quantizes against the UE8M0 scale, and packs to two u64.
# Like the source helper it performs NO stores; it returns
# (scale_ue8m0_u32, packed64_0, packed64_1) and the caller stores the SF byte
# first, then the two 8-byte output groups.
# ===========================================================================

def PROCESS_BLOCK(row_idx, col_idx):
    elem_base = col_idx * SF_VEC
    # instruction_selection: shl by constant; extent: scalar
    base = i64(row_idx) * K + elem_base
    # instruction_selection: mul.wide.s32/mad.lo.s64 family; extent: scalar
    h = reg_tile("b32", [16])
    # instruction_selection: none; extent: sixteen b32 registers per thread
    copy_g2r_v4_u32(i64(input) + (base + 0) * 2, h[0:4])
    # instruction_selection: ld.global.v4.u32; extent: one 16-byte vector load
    copy_g2r_v4_u32(i64(input) + (base + 8) * 2, h[4:8])
    # instruction_selection: ld.global.v4.u32; extent: one 16-byte vector load
    copy_g2r_v4_u32(i64(input) + (base + 16) * 2, h[8:12])
    # instruction_selection: ld.global.v4.u32; extent: one 16-byte vector load
    copy_g2r_v4_u32(i64(input) + (base + 24) * 2, h[12:16])
    # instruction_selection: ld.global.v4.u32; extent: one 16-byte vector load

    max_first = ABSMAX_8(h[0:8])
    # instruction_selection: see ABSMAX_8; extent: 8x and.b32 + 7x max.{f16,bf16}x2
    max_second = ABSMAX_8(h[8:16])
    # instruction_selection: see ABSMAX_8; extent: 8x and.b32 + 7x max.{f16,bf16}x2
    block_max_h2 = max_h2(max_first, max_second)
    # instruction_selection: max.f16x2 | max.bf16x2; extent: one packed pair
    block_max = HMAX_REDUCE_F32(block_max_h2)
    # instruction_selection: see HMAX_REDUCE_F32; extent: one pair to one f32

    normalized = mul(block_max, rcp_ftz(6.0))
    # instruction_selection: rcp.approx.ftz.f32 + mul.f32; extent: one scalar each
    scale_ue8m0_u32 = UE8M0_ENCODE(normalized)
    # instruction_selection: see UE8M0_ENCODE; extent: one scalar
    inv_scale = UE8M0_INV_SCALE(scale_ue8m0_u32)
    # instruction_selection: see UE8M0_INV_SCALE; extent: one scalar

    s = reg_tile("f32", [32])
    # instruction_selection: none; extent: per-thread registers
    for i in static_range(16):
        s[2*i], s[2*i+1] = FLOAT2_SCALED(h[i], inv_scale)
        # instruction_selection: see FLOAT2_SCALED; extent: one packed pair -> two f32
    packed = reg_tile("b32", [4])
    for j in static_range(4):
        packed[j] = E2M1X8(s[8*j:8*j+8])
        # instruction_selection: see E2M1X8; extent: 4x cvt.rn.satfinite.e2m1x2.f32 + byte pack
    packed64_0 = (u64(packed[1]) << 32) | u64(packed[0])
    # instruction_selection: shl.b64 + or.b64 (source: plain CuTe-DSL u64 shift/or, NOT mov.b64); extent: one u64
    packed64_1 = (u64(packed[3]) << 32) | u64(packed[2])
    # instruction_selection: shl.b64 + or.b64; extent: one u64
    return scale_ue8m0_u32, packed64_0, packed64_1   # no stores here (source helper)

# ===========================================================================
# Subroutines — bodies in source order, one annotation per instruction.
# ABSMAX_8, HMAX_REDUCE_F32, UE8M0_ENCODE, UE8M0_INV_SCALE are the SAME source
# helpers reviewed in .agents/sketch/mxfp8_quantize.md (utils:691/:728, :95/
# :122, :157, :207) with identical instruction selections; reproduced here in
# compact form.
# ===========================================================================

def ABSMAX_8(v):                      # utils:691 half2_max_abs_8 / :728 bf16
    a = [abs_h2(v[i]) for i in range(8)]
    # instruction_selection: and.b32 (mask 0x7FFF7FFF); extent: 8 packed pairs
    m01 = max_h2(a[0], a[1]); m23 = max_h2(a[2], a[3])
    m45 = max_h2(a[4], a[5]); m67 = max_h2(a[6], a[7])
    # instruction_selection: max.f16x2 | max.bf16x2; extent: 4 packed-pair maxima
    m03 = max_h2(m01, m23); m47 = max_h2(m45, m67)
    # instruction_selection: max.f16x2 | max.bf16x2; extent: 2 packed-pair maxima
    return max_h2(m03, m47)
    # instruction_selection: max.f16x2 | max.bf16x2; extent: 1 packed-pair maximum

def HMAX_REDUCE_F32(x):               # utils:95 / :122
    if DTYPE == "f16":
        h0, h1 = unpack_b16x2(x)
        # instruction_selection: mov.b32 {h0,h1}, x; extent: one pair unpack
        f0 = cast_f32_h(h0); f1 = cast_f32_h(h1)
        # instruction_selection: cvt.f32.f16 x2; extent: two scalars
    else:
        lo = (x & 0xFFFF) << 16; hi = (x >> 16) << 16
        # instruction_selection: and.b32 (0xFFFF) + shr.b32 (16) + shl.b32 x2 (16); extent: one pair unpack
        f0 = b32_to_f32(lo); f1 = b32_to_f32(hi)
        # instruction_selection: mov.b32 x2; extent: two scalars
    return fmax(f0, f1)
    # instruction_selection: max.f32; extent: one scalar

def UE8M0_ENCODE(f):                  # utils:157 float_to_ue8m0_fast
    p_zero = setp_le_f32(f, 0.0)
    # instruction_selection: setp.le.f32; extent: one predicate
    bits = b32(f); exp = (bits >> 23) & 255; mant = bits & 0x7FFFFF
    # instruction_selection: mov.b32 + shr.b32 + and.b32 x2; extent: one scalar
    bump = select(mant != 0, 1, 0)
    # instruction_selection: setp.ne.u32 + selp.u32; extent: one scalar
    tiny_sub = (exp == 0) & (mant <= 0x400000)
    # instruction_selection: setp.eq.u32 + setp.le.u32 + and.pred; extent: one scalar
    bump = select(tiny_sub, 0, bump)
    # instruction_selection: @pred mov.u32 (selp form); extent: one predicated move
    result = exp + bump
    # instruction_selection: add.u32; extent: one scalar
    result = select(result > 254, 254, result)
    # instruction_selection: setp.gt.u32 + selp.u32; extent: one scalar
    return select(p_zero, 0, result)
    # instruction_selection: selp.u32; extent: one scalar

def UE8M0_INV_SCALE(u):               # utils:207 ue8m0_to_inv_scale_fast
    new_exp = max(254 - u, 0)
    # instruction_selection: sub.s32 + max.s32; extent: one scalar
    f_bits = new_exp << 23
    # instruction_selection: shl.b32 + mov.b32; extent: one scalar
    return select(u == 0, 0.0, f32(f_bits))
    # instruction_selection: setp.eq.u32 + @pred mov.b32 (selp form); extent: one predicated move

def FLOAT2_SCALED(h2, inv):           # utils:376 half2_to_float2_scaled / :406 bfloat2_to_float2_scaled
    if DTYPE == "f16":
        h0, h1 = unpack_b16x2(h2)
        # instruction_selection: mov.b32 {h0,h1}, h2; extent: one pair unpack
        f0 = cast_f32_h(h0); f1 = cast_f32_h(h1)
        # instruction_selection: cvt.f32.f16 x2; extent: two scalars
    else:
        lo = (h2 & 0xFFFF) << 16; hi = (h2 >> 16) << 16
        # instruction_selection: and.b32 (0xFFFF) + shr.b32 (16) + shl.b32 x2 (16); extent: one pair unpack
        f0 = b32_to_f32(lo); f1 = b32_to_f32(hi)
        # instruction_selection: mov.b32 x2; extent: two scalars
    return mul(f0, inv), mul(f1, inv)
    # instruction_selection: mul.f32 x2; extent: two scalars

def E2M1X8(s8):                       # utils:442 cvt_e2m1x8_f32
    for i in static_range(4):
        byte[i] = cast_e2m1x2(s8[2*i + 1], s8[2*i])   # hi operand first (source order)
        # instruction_selection: cvt.rn.satfinite.e2m1x2.f32; extent: one packed pair
    return pack_bytes4(byte[0], byte[1], byte[2], byte[3])
    # instruction_selection: source mov.b32 {b0,b1,b2,b3} (4 x b8 form not registered in
    # the TIRx dialect) -> native b16-pair shl/or + mov.b32 (2 x b16), the form proven
    # by silu_and_mul_nvfp4_experts_quantize; same SASS pack family; extent: one u32

def SF_OFFSET(row, col, padded_cols): # 128x4: utils:535; 8x4: utils:569
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
def host_launch_shape(m, k, sf_layout):
    # linear (mxfp4_quantize.py:963-971): 512 threads; SF_BLOCKS_PER_TB = 512
    #   grid = min(ceildiv(m*(k/32), 512), SM_COUNT * 4)
    # swizzled (:979-986): threads = _compute_optimal_threads(k) in [128, 512]
    #   (largest multiple of k/32); needs_col_loop when k/32 > 512;
    #   grid = min(ceildiv(padded_m, rows_per_block), SM_COUNT * 4)
    # instruction_selection: none; extent: host-only arithmetic (SM_COUNT=148 on B200)

use_4t = SM_COUNT <= 80               # always False on B200 (out of scope)

def prepare_data(dtype, m, k, sf_layout, enable_pdl):
    host_assert(dtype in ("float16", "bfloat16") and k % 32 == 0 and m >= 1)
    a = contiguous_seeded_randn(shape=(m, k), dtype=dtype)   # seed 42
    # instruction_selection: none; extent: tensor constructions

# run_test compares the packed FP4 output and the full SF buffer (source
# zero-fills padding rows/columns, so the whole padded buffer is compared)
# against flashinfer.quantization.mxfp4_quantize(backend="cute-dsl",
# enable_pdl=False), exact byte equality (rtol=0, atol=0).
# run_bench times the primfunc launch (tirx) against the cached compiled
# source kernel (_get_compiled_kernel_mxfp4, use_4t_per_sf=False) with
# preallocated outputs, enable_pdl=False; both closures are no-argument.
```

## Static specialization boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| `DTYPE` | static per config | selects f16/bf16 unpack (`mov.b32 {h0,h1}` + `cvt.f32.f16` vs `and/shr/shl` + `mov.b32`) and `max.f16x2` vs `max.bf16x2` |
| `THREADS_PER_SF == 1` | static (B200 num_sm=148 > 80) | one SF block per thread; no `shfl.sync` reduction; 4T/SF out of scope |
| `SF_LAYOUT` | static per config | selects linear vs swizzled kernel body and the SF index math (128x4 vs 8x4) |
| `K` | static per config | `NSB = K/32`, `PAD_COLS`, swizzled threads/rows_per_block/needs_col_loop constant-fold |
| grid/block extents | static per config | host formulas baked in; kernel reads `%ctaid.x`/`%tid.x` only |
| `M`, `total_sf_blocks`, `padded_M` | runtime i32 ABI | loop bounds and row predicates stay runtime like the source (linear `M` is ABI-only, unread by the 1T body) |
| `ENABLE_PDL` | static per config (default False) | griddepcontrol pair present only in the PDL instantiation; TVM launch carries no PDL attribute |
| `launch_bounds` | static | source declares max_number_threads=1024 + min_blocks_per_mp=4, but ptxas does not enforce the implied 16-register cap (source binary uses 50 registers, 2 blocks/SM); the TIRx build therefore sets min_blocks_per_sm=2 (`__launch_bounds__(block, 2)`, 64-register headroom, nvcc picks 46), matching the source binary's realized occupancy instead of the unenforced hint |

## TIRx module and benchmark contract

- `KERNEL_META = {"name": "mxfp4_quantize", "category": "flashinfer",
  "compute_capability": 10}`.
- The executable kernel is expressed entirely in plain TIRx: explicit `while`
  grid-stride loops, runtime scalar ABI, register tiles, and native `T.ptx.*`
  forms for every non-trivial instruction (`ld.global.v4.u32`,
  `st.global.u64`/`st.global.b8`, `max.f16x2`/`max.bf16x2`, `max.f32`,
  `mul.f32` non-ftz, `rcp.approx.ftz.f32`, `cvt.rn.satfinite.e2m1x2.f32`,
  `mov.b32` pack, `setp`/`selp` families, `griddepcontrol`). Integer bit math
  of the UE8M0 helpers is plain TIRx b32 ops, matching the source asm
  instruction for instruction; all float muls and the float compare go through
  explicit `T.ptx` forms so the FTZ-flagged host compiler cannot reinterpret
  them. There is no `T.cuda.func_call` and no `Tx` tile primitives anywhere in
  the pre-dispatch IR.
- `get_kernel(dtype, m, k, sf_layout, enable_pdl)` returns the specialized
  primfunc with static grid/block; `prepare_data`, `run_test`, `run_bench`
  follow the repository contract.
- The timed implementation is named `tirx`; the reference is the cached
  compiled CuTe-DSL source kernel launched with `enable_pdl=False`,
  `use_4t_per_sf=False`, and preallocated outputs. Allocation, compilation,
  and correctness checks stay outside timing.
- Correctness compares the full packed output and the full padded SF buffer
  against the source wrapper, exact byte equality.

## Instruction selection is a lowering consequence

The sketch above never requests a hardware instruction beyond the documented
PTX helpers. The following lowering families follow from storage direction,
shape, dtype, and schedule. PTX names are taken from the source inline-asm
blocks; per-iteration counts below are the expected values the sketch-reviewer
verifies against a fresh line-info PTX export of the exact source
specializations (`CUTE_DSL_KEEP=ptx CUTE_DSL_LINEINFO=1`, enable_pdl=False,
use_4t_per_sf=False).

| Primitive/schedule pattern | PTX family (expected, per SF block, 1T/SF) |
| --- | --- |
| `copy_g2r_v4_u32` loads | `ld.global.v4.u32` x4 (no `.nc`) |
| absmax tree | `and.b32` x16 + `max.f16x2` x15 (bf16: `max.bf16x2`) — two ABSMAX_8 + one final max |
| pair-max to f32 | f16: `mov.b32 {h0,h1}` + `cvt.f32.f16` x2 + `max.f32`; bf16: `and.b32` + `shr.b32` + `shl.b32` x2 + `mov.b32` x2 + `max.f32` |
| scale normalize | `rcp.approx.ftz.f32` x1 + `mul.f32` x1 |
| UE8M0 encode | `setp.le.f32` + `mov.b32` + `shr.b32` + `and.b32` x2 + `setp.ne.u32` + `selp.u32` + `setp.eq.u32` + `setp.le.u32` + `and.pred` + `@p mov.u32` + `add.u32` + `setp.gt.u32` + `selp.u32` x2 |
| inverse scale | `setp.eq.u32` + `sub.s32` + `max.s32` + `shl.b32` + `mov.b32` + `@p mov.b32` |
| unpack + scale (per 32 elements) | f16: `mov.b32 {h0,h1}` x16 + `cvt.f32.f16` x32; bf16 (per pair): `and.b32` + `shr.b32` + `shl.b32` x2 + `mov.b32` x2 (x16 pairs); `mul.f32` x32 |
| e2m1 convert + pack | `cvt.rn.satfinite.e2m1x2.f32` x16 + byte-pack movs (source `mov.b32 {b0..b3}` x4; TIRx b16-pair `shl`/`or` + `mov.b32` x4) |
| u64 combine | `shl.b64` x2 + `or.b64` x2 (source: CuTe u64 shift/or, not mov.b64) |
| output stores | `st.global.u64` x2 |
| SF store | `st.global.b8` x1 (unpredicated in 1T/SF) |
| SF swizzle address | `and.b32`/`shr.b32` by constants + `mad.lo.s32` family (swizzled layouts only) |
| row/col decode + addresses | constant-divisor mul-shift family, `mul.wide.s32`/`mad.lo.s64` 64-bit address math |
| loop control | `setp` + `bra`, `add.s32`/`mad.lo.s32` induction |
| PDL (ENABLE_PDL only) | `griddepcontrol.wait` x1, `griddepcontrol.launch_dependents` x1 (whole kernel) |

The bf16 instantiations differ only in the unpack sequences and the
`max.bf16x2` opcode. The 8x4 instantiation differs from 128x4 only in the
SF_OFFSET arithmetic. Relative to the mxfp8 port, the 1T/SF structure removes
all shuffle reductions and the SF-store lane predicate, and the e2m1 nibble
pack replaces the e4m3 byte pack.
