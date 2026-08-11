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
(NVFP4QuantizePerTokenKernel), the CuTe-DSL SM100 kernel behind
flashinfer.quantization.nvfp4_quantize(..., per_token_activation=True,
backend="cute-dsl").
-->

# nvfp4_quantize_per_token SM100: coarse WASP pipeline sketch

This non-executable design sketch describes the thread roles, control flow,
storage placement, and PTX-level operations of
[`tirx_kernels/flashinfer/quantization/nvfp4_quantize_per_token.py`](../../tirx_kernels/flashinfer/quantization/nvfp4_quantize_per_token.py).
That TIRx module is the authoritative implementation.

The instantiations are `DTYPE in {f16, bf16}` crossed with
`SF_LAYOUT in {linear, 128x4, 8x4}` of the source CuTe-DSL class
`NVFP4QuantizePerTokenKernel`, compiled for sm_100a with
`disable_fp4_quant_fast_math=False` and `nvfp4_4over6_config=None`. Accepted
target is SM100/B200. The grid is `M` blocks of 128 threads (one CTA per token
row); `M` is the runtime ABI scalar exactly like the source signature, and the
grid extent is static per config (the source host launches `grid=[M,1,1]`).
The `enable_pdl=True` variant is ported as the same compile-time knob (entry
`griddepcontrol.wait`, exit `griddepcontrol.launch_dependents`) but is off in
all default configs: TVM launches do not carry the PDL launch attribute, so
test/bench parity pins `enable_pdl=False` on both sides. Out of scope
(deferred variants): the `FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH` exact-math
path, the 4over6 dual-scale path, fp8 input (rejected by the source), and tile
(`Tx`) primitives.

## Pipeline at a glance

| Warps | Role-local program | Publication/reuse edges |
| --- | --- | --- |
| all 4 warps (uniform) | Every thread runs the same single-role program in two passes. Pass 1: column-stride loop over the row's 16-element SF blocks — two 16-byte loads, absmax tree, pair-max to f32, fmax-accumulate. Then a warp+block max reduction. Then row scales, a one-lane per-token-scale store, a barrier. Pass 2: the same column-stride loop re-running the regular NVFP4 block quantizer with the row's encode scale, storing the SF byte (layout offset) and the packed u64 output per block. Swizzled layouts finish with a padding-column SF zero-fill loop. | one 16-byte SMEM reduction buffer + two `bar.sync` (inside `block_reduce`; after the per-token-scale store); butterfly `shfl.sync.bfly.b32` reductions; optional `griddepcontrol` pair |

There is no warp specialization: all four warps run the same program and
cooperate only in the reduction. The branches are the column-loop guards, the
`row_amax == 0` select, the `tidx == 0` store predicate, the `lane < 4` smem
read predicate, and the padding-column loop.

## Primitive vocabulary

Structural operations declare placement without moving data:

```python
specialize(...)       # compile-time variant selection
launch(...)           # compile-time launch topology and attributes
reg_tile(...)         # per-thread register tile
smem_tile(...)        # per-CTA shared tile
```

Copies state their direction and width:

```python
copy_g2r_v4_u32(src_addr, dst_b32x4)  # one 16-byte global -> register vector load
copy_g2r_f32(src_addr, dst)           # one scalar 32-bit global load (emitted ld.global.b32)
copy_r2g_u64(src_u64, dst_addr)       # one 8-byte register -> global store
copy_r2g_u8(src_u8, dst_addr)         # one byte register -> global store (emitted st.global.b8)
copy_r2g_f32(src_f32, dst_addr)       # one f32 register -> global store (per-token scale)
copy_r2s_f32(src_f32, dst_addr)       # one f32 register -> shared store
copy_s2r_f32(src_addr, dst)           # one f32 shared load
```

Compute and sync vocabulary (subroutines annotate per instruction):

```python
abs_h2 / max_h2                # and.b32 0x7FFF7FFF; max.f16x2 | max.bf16x2
shfl_bfly_f32(dst, src, xor)   # shfl.sync.bfly.b32, full membermask
fmax(dst, lhs, rhs)            # max.f32
rcp_ftz(dst, src)              # rcp.approx.ftz.f32
mul(dst, lhs, rhs)             # mul.f32 (non-ftz, as the source asm)
bar_sync()                     # bar.sync 0 (cute.arch.barrier())
select / setp                  # selp/setp families
```

Subroutines (bodies annotated per instruction in the sketch; ABSMAX_8,
HMAX_REDUCE_F32, FLOAT2_SCALED, E2M1X8, NVFP4_SCALE are the same source
helpers reviewed in .agents/sketch/flashinfer/quantization/nvfp4_quantize.md with identical
instruction selections and are not repeated):

```python
WARP_REDUCE_MAX(v) -> f32      # warp_reduce: 5 butterfly rounds (fp4_common:1356)
BLOCK_REDUCE_MAX(v) -> f32     # block_reduce: smem + bar.sync + warp (fp4_common:1371)
ROW_SCALES(row_amax, gs_inv) -> (encode, token)  # _row_scales fast-math (:729-738)
PROCESS_BLOCK(col, encode_scale) -> (sf_u8, packed64)  # process_nvfp4_block_*
```

`thread_id`, `cta_id`, `pdl_wait`, and `pdl_launch_dependents` are schedule
operations. Address expressions, loop bounds, and guards are shown directly;
they do not hide copies, computation, role changes, or synchronization. Row
base addresses use explicit 64-bit arithmetic (the source's documented fix for
int32 row-slice overflow, nvfp4_quantize.py:789-817); column indices stay
32-bit.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

variant = specialize(DTYPE=("f16", "bf16"), SF_LAYOUT=("linear", "128x4", "8x4"),
                     ENABLE_PDL=False, target="sm_100a")
# instruction_selection: none; extent: 2x3 compile-time instantiations (x PDL knob)

SF_VEC = 16                       # NVFP4_SF_VEC_SIZE
GRID_X, BLOCK_X = M, 128          # one CTA per row; 4 warps
# instruction_selection: none; extent: static launch metadata

launch_config = launch(
    grid=(GRID_X, 1, 1),
    block=(BLOCK_X, 1, 1),
    launch_bounds=(1024, 4),        # declared by the source; see boundary table note
    dynamic_smem_bytes=1024,        # the source's SmemAllocator carves the 16B reduction
                                    # buffer from a dynamic window padded to a 1024B
                                    # alignment (PTX: .extern .shared __dynamic_shmem__0);
                                    # the TIRx port uses an equivalent static 16B tile
)
# instruction_selection: none; extent: static launch metadata

def nvfp4_quantize_per_token(
    input,          # DTYPE [M*K], direct global pointer
    output,         # u8 [M*K/2], direct global pointer
    scales,         # u8 [padded_M * padded_sf_cols], direct global pointer
    per_token_scale,# f32 [M], direct global pointer
    M,              # runtime i32 (source ABI; the grid extent carries the same value)
    global_scale_inv,  # f32 [1], direct global pointer
):
    bx = cta_id(axis="x", extent=GRID_X)
    # instruction_selection: mov.u32 from %ctaid.x; extent: scalar per thread
    tx = thread_id(extent=BLOCK_X, dtype="uint32")
    # instruction_selection: mov.u32 from %tid.x; extent: scalar per thread

    pdl_wait()                        # only when ENABLE_PDL
    # instruction_selection: griddepcontrol.wait; extent: every thread, kernel entry

    red_buf = smem_tile("f32", [1, 4])    # block_reduce buffer (kernel:779-784)
    # instruction_selection: none; extent: 16 bytes of smem per CTA (dynamic window in
    # the source binary; a static tile in the TIRx port — same storage class/lifetime)

    row_idx = bx                      # one CTA per token row (kernel:786)
    NSB = K // 16                     # num_sf_blocks_per_row (compile-time)
    PAD_COLS = NSB if SF_LAYOUT == "linear" else (NSB + 3) // 4 * 4

    # 64-bit row bases (kernel:794-817: int32 row slicing wraps; use int64)
    in_row = i64(input) + i64(row_idx) * (K * 2)        # byte address
    # instruction_selection: mul.wide.s32/mad.lo.s64 family; extent: scalar per thread
    out_row = i64(output) + i64(row_idx) * (K // 2)
    # instruction_selection: mad.lo.s64 family; extent: scalar per thread

    # =======================================================================
    # Pass 1: row amax (kernel:819-834)
    # =======================================================================
    local_amax = 0.0
    sf_col = tx
    while sf_col < NSB:
        # instruction_selection: setp.lt.s32 + bra; extent: column loop control
        h = reg_tile("b32", [8])
        # instruction_selection: none; extent: eight b32 registers per thread
        copy_g2r_v4_u32(in_row + sf_col * (SF_VEC * 2), h[0:4])
        # instruction_selection: ld.global.v4.u32; extent: one 16-byte vector load
        copy_g2r_v4_u32(in_row + sf_col * (SF_VEC * 2) + 16, h[4:8])
        # instruction_selection: ld.global.v4.u32; extent: one 16-byte vector load
        block_max = HMAX_REDUCE_F32(ABSMAX_8(h))
        # instruction_selection: see ABSMAX_8 + HMAX_REDUCE_F32; extent: one SF block
        local_amax = fmax(local_amax, block_max)
        # instruction_selection: max.f32; extent: one scalar per iteration
        sf_col = sf_col + 128
        # instruction_selection: add.s32; extent: loop induction update

    # =======================================================================
    # Reduction (kernel:836-837; warp_reduce + block_reduce, fp4_common:1356-1390)
    # =======================================================================
    warp_amax = WARP_REDUCE_MAX(local_amax)
    # instruction_selection: 5x (shfl.sync.bfly.b32 + max.f32), offsets 1,2,4,8,16; extent: warp allreduce
    row_amax = BLOCK_REDUCE_MAX(warp_amax)
    # instruction_selection: see BLOCK_REDUCE_MAX; extent: block allreduce via red_buf + bar.sync

    gs_inv = copy_g2r_f32(global_scale_inv + 0)   # read once per thread (kernel:838)
    # instruction_selection: ld.global.b32; extent: one scalar load
    global_encode_scale, token_scale = ROW_SCALES(row_amax, gs_inv)
    # instruction_selection: see ROW_SCALES; extent: one scalar pair

    if tx == 0:
        # instruction_selection: setp.ne.s32 + bra (emitted polarity: skips the store); extent: once per CTA
        copy_r2g_f32(token_scale, per_token_scale + row_idx)
        # instruction_selection: st.global.f32 (emitted st.global.b32); extent: one scalar store per CTA
    bar_sync()
    # instruction_selection: bar.sync 0; extent: whole CTA (kernel:844)

    # =======================================================================
    # Pass 2: quantize with the row encode scale (kernel:846-875)
    # =======================================================================
    sf_col = tx
    while sf_col < NSB:
        # instruction_selection: setp.lt.s32 + bra; extent: column loop control
        scale_fp8, packed64 = PROCESS_BLOCK(sf_col, global_encode_scale)
        # instruction_selection: see PROCESS_BLOCK; extent: one 16-element SF block
        # source order: SF byte store BEFORE the output store (:869 then :873)
        copy_r2g_u8(scale_fp8, scales + SF_OFFSET(row_idx, sf_col, PAD_COLS))
        # instruction_selection: st.global.b8; extent: one byte store per iteration
        copy_r2g_u64(packed64, out_row + sf_col * 8)
        # instruction_selection: st.global.u64; extent: one 8-byte store per iteration
        sf_col = sf_col + 128
        # instruction_selection: add.s32; extent: loop induction update

    if SF_LAYOUT != "linear":         # compile-time; padding columns (:877-882)
        sf_col2 = NSB + tx
        while sf_col2 < PAD_COLS:
            # instruction_selection: setp/bra; extent: zero-fill loop control
            copy_r2g_u8(0, scales + SF_OFFSET(row_idx, sf_col2, PAD_COLS))
            # instruction_selection: st.global.b8; extent: one byte store
            sf_col2 = sf_col2 + 128

    pdl_launch_dependents()           # only when ENABLE_PDL
    # instruction_selection: griddepcontrol.launch_dependents; extent: every thread, kernel exit

# ===========================================================================
# WARP_REDUCE_MAX — warp_reduce (fp4_common:1356): 5 butterfly rounds
# ===========================================================================

def WARP_REDUCE_MAX(v):
    for i in static_range(5):
        v = fmax(v, shfl_bfly_f32(v, 1 << i))
        # instruction_selection: shfl.sync.bfly.b32 + max.f32; extent: one butterfly round
    return v

# ===========================================================================
# BLOCK_REDUCE_MAX — block_reduce (fp4_common:1371): smem exchange + warp round
# red_buf is the (1,4) f32 smem tile; init value 0.0
# ===========================================================================

def BLOCK_REDUCE_MAX(v):
    lane = tx % 32
    warp = tx // 32
    # instruction_selection: and/shr by constants; extent: scalar per thread
    if lane == 0:
        # instruction_selection: setp.ne + bra (emitted polarity); extent: per warp
        copy_r2s_f32(v, red_buf[0, warp])
        # instruction_selection: st.shared.f32 (emitted st.shared.b32); extent: one scalar store per warp
    bar_sync()
    # instruction_selection: bar.sync 0; extent: whole CTA
    block_val = 0.0
    if lane < 4:
        # instruction_selection: setp.ge + bra (emitted polarity: skips the read); extent: per thread
        block_val = copy_s2r_f32(red_buf[0, lane])
        # instruction_selection: ld.shared.f32 (emitted ld.shared.b32); extent: one scalar load
    return WARP_REDUCE_MAX(block_val)
    # instruction_selection: 5x (shfl.sync.bfly.b32 + max.f32); extent: one warp allreduce

# ===========================================================================
# ROW_SCALES — _row_scales fast-math path (nvfp4_quantize.py:729-738)
# ===========================================================================

def ROW_SCALES(row_amax, gs_inv):
    if row_amax == 0.0:
        # instruction_selection: setp.eq.f32 + bra/selp family; extent: one select
        return FLOAT32_MAX, 0.0
    token_scale = mul(row_amax, gs_inv)
    # instruction_selection: mul.f32; extent: one scalar
    encode_scale = rcp_ftz(token_scale)
    # instruction_selection: rcp.approx.ftz.f32; extent: one scalar
    return encode_scale, token_scale

# ===========================================================================
# PROCESS_BLOCK — process_nvfp4_block_half (utils:1870) / _bfloat (:1895) with
# global_scale = global_encode_scale: the SAME block program reviewed in
# .agents/sketch/flashinfer/quantization/nvfp4_quantize.md (2 loads, ABSMAX_8, HMAX_REDUCE_F32,
# NVFP4_SCALE, 8x FLOAT2_SCALED, 2x E2M1X8, shl.b64/or.b64 combine; no stores).
# Loads at in_row + sf_col*32 bytes; returns (scale_fp8_u8, packed64).
# ===========================================================================
```

## Host wrapper and validation

The Python module performs host-only work; none of it emits device PTX:

```python
def prepare_data(dtype, m, k, sf_layout, zero_row):
    host_assert(dtype in ("float16", "bfloat16") and k % 16 == 0 and m >= 1)
    a = contiguous_seeded_randn(shape=(m, k), dtype=dtype)   # seed 42
    if zero_row:
        a[0].zero_()              # covers the row_amax == 0 path
    gs_inv = torch_tensor([1.0 / (448.0 * 6.0)], dtype="float32")  # device [1]
    # instruction_selection: none; extent: tensor constructions

# The SF buffer is zero-filled at allocation (source: torch.zeros) so swizzled
# padding rows carry zeros written by the host; padding columns are zeroed by
# the kernel. run_test compares the packed output, the full SF buffer, and the
# per-token scale against flashinfer.quantization.nvfp4_quantize(...,
# per_token_activation=True, backend="cute-dsl", enable_pdl=False), exact byte
# equality (rtol=0, atol=0) on all three.
# run_bench times the primfunc launch (tirx) against the cached compiled
# source kernel (_get_compiled_kernel_nvfp4_per_token) with preallocated
# outputs, enable_pdl=False; both closures are no-argument.
```

## Static specialization boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| `DTYPE` | static per config | selects f16/bf16 unpack and `max.f16x2` vs `max.bf16x2` |
| `SF_LAYOUT` | static per config | SF index math (linear/128x4/8x4) and the padding-column loop presence |
| `K` | static per config | `NSB = K/16`, `PAD_COLS`, column-loop trip counts constant-fold |
| grid extent `M` | static per config | one CTA per row; `M` also rides the ABI like the source |
| `global_scale_inv` device tensor | runtime pointer; value read once per thread | one `ld.global.b32` broadcast per thread |
| `ENABLE_PDL` | static per config (default False) | griddepcontrol pair present only in the PDL instantiation |
| launch bounds | static | source declares (1024, 4) but ptxas does not enforce the implied cap; the TIRx build sets min_blocks_per_sm=2 to realize the source binary's occupancy (as in the sibling ports) |
| `disable_fast_math=False`, `4over6=None` | static | default-environment code path only |

## TIRx module and benchmark contract

- `KERNEL_META = {"name": "nvfp4_quantize_per_token", "category": "flashinfer",
  "compute_capability": 10}`.
- The executable kernel is expressed entirely in plain TIRx: explicit `while`
  column-stride loops, runtime scalar ABI, register tiles, one static 16-byte
  smem buffer, `bar.sync` barriers, and native `T.ptx.*` forms for every
  non-trivial instruction (`ld.global.v4.u32`, `st.global.u64`/`st.global.b8`,
  `st.shared`/`ld.shared`, `shfl.sync.bfly.b32`, `max.f32`,
  `max.f16x2`/`max.bf16x2`, `mul.f32` non-ftz, `rcp.approx.ftz.f32`,
  `cvt.rn.satfinite.e4m3x2.f32`, `cvt.rn.satfinite.e2m1x2.f32`, `setp`/`selp`
  families, `griddepcontrol`). There is no `T.cuda.func_call` and no `Tx` tile
  primitives anywhere in the pre-dispatch IR.
- `get_kernel(dtype, m, k, sf_layout, enable_pdl)` returns the specialized
  primfunc with static grid/block; `prepare_data`, `run_test`, `run_bench`
  follow the repository contract.
- The timed implementation is named `tirx`; the reference is the cached
  compiled CuTe-DSL source kernel launched with `enable_pdl=False` and
  preallocated outputs. Allocation, compilation, and correctness checks stay
  outside timing.
- Correctness compares the packed output, the full padded SF buffer, and the
  per-token scale against the source wrapper, exact byte equality.

## Instruction selection is a lowering consequence

The sketch above never requests a hardware instruction beyond the documented
PTX helpers. The following lowering families follow from storage direction,
shape, dtype, and schedule. Per-iteration counts below are the expected values
the sketch-reviewer verifies against a fresh line-info PTX export of the exact
source specializations (`CUTE_DSL_KEEP=ptx CUTE_DSL_LINEINFO=1`,
enable_pdl=False, default env).

| Primitive/schedule pattern | PTX family (expected) |
| --- | --- |
| pass-1 loads | `ld.global.v4.u32` x2 per SF block (no `.nc`) |
| pass-1 absmax | `and.b32` x8 + `max.f16x2` x7 (bf16: `max.bf16x2`) + pair-max sequence + `max.f32` accumulate |
| warp reduction | `shfl.sync.bfly.b32` + `max.f32`, x5 rounds (offsets 1,2,4,8,16), full membermask |
| block reduction | `st.shared.b32` (lane0) + `bar.sync` + `ld.shared.b32` (lanes<4) + 5 shuffle rounds |
| row scales | `setp.eq.f32` + select + `mul.f32` + `rcp.approx.ftz.f32` |
| per-token store | `st.global.b32` x1 per CTA (tidx==0) + `bar.sync` |
| pass-2 block | the nvfp4_quantize block inventory: 2 loads, absmax tree, `rcp.approx.ftz.f32`+`mul.f32` normalize, E4M3 convert, 9-instr output scale, 8x unpack+2x`mul.f32` each, 8x `cvt.rn.satfinite.e2m1x2.f32` + pack, `shl.b64`+`or.b64` combine |
| stores (pass 2) | `st.global.b8` x1 (SF, unpredicated, first) + `st.global.u64` x1 (output) |
| zero fill (swizzled) | `st.global.b8` x1 per padding column |
| 64-bit row bases | `mul.wide.s32`/`mad.lo.s64` family (once per kernel) |
| loop control | `setp` + `bra`, `add.s32` induction |
| PDL (ENABLE_PDL only) | `griddepcontrol.wait` x1, `griddepcontrol.launch_dependents` x1 (whole kernel) |

The bf16 instantiations differ only in the unpack sequences and the
`max.bf16x2` opcode. The 8x4/linear instantiations differ from 128x4 only in
the SF_OFFSET arithmetic and the padding-column loop.
