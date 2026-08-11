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
csrc/nv_internal/tensorrt_llm/kernels/quantization.cuh
(cvt_fp16_to_fp4_expert), the SM100 kernel behind
flashinfer.activation.silu_and_mul_scaled_nvfp4_experts_quantize,
itself ported from NVIDIA TensorRT-LLM. See NOTICE and
licenses/ for upstream attribution.
-->

# silu_and_mul_nvfp4_experts_quantize SM100: coarse WASP pipeline sketch

This non-executable design sketch describes the thread roles, control flow,
storage placement, and PTX-level operations of
[`tirx_kernels/flashinfer/activation/silu_and_mul_nvfp4_experts_quantize.py`](../../tirx_kernels/flashinfer/activation/silu_and_mul_nvfp4_experts_quantize.py).
That TIRx module is the authoritative implementation.

The two instantiations are `DTYPE in {f16, bf16}` of the default-environment
source specialization `cvt_fp16_to_fp4_expert<T, UE8M0_SF=false,
DISABLE_FP4_QUANT_FAST_MATH=false, NVFP4_4OVER6_CONFIG=std::false_type>`,
compiled for sm_100a with CUDA >= 12.9 (device `ELTS_PER_THREAD = 16`). Grid
and block extents are static per config (the source host computes them once
per call from the same formulas); `num_rows`, `num_cols`, `n_experts`,
`use_silu_and_mul`, and all pointers stay runtime ABI values, exactly like the
source kernel signature. Accepted target is SM100/B200. The 4over6 refinement
path, the UE8M0 SF path, the fastmath-disabled path, the TMA quantization
kernel, `scaled_fp4_grouped_quant_sm100`, SM90/SM120, and tile (`Tx`)
primitives are out of scope.

## Pipeline at a glance

| Warps | Role-local program | Publication/reuse edges |
| --- | --- | --- |
| all (uniform) | Every thread runs the same single-role program: expert-partition decode, then a grid-stride loop over this expert's 16-element chunks: mask early-exit, two 32-byte packed loads, fused silu*mul rounded back to fp16/bf16, local abs-max, per-16-element NVFP4 quantization (E4M3 scale factor + 16 x e2m1 packed in one uint64), one swizzled SF byte store, one uint64 store. | none — no SMEM, no mbarriers, no cross-thread data; each thread owns a whole 16-element SF block (`CVT_FP4_NUM_THREADS_PER_SF == 1`, the `__shfl_xor_sync` reductions are compiled out) |

There is no warp specialization and no producer/consumer split; the only
branches are the expert-partition select, the mask early-exit, the
`use_silu_and_mul` / `use_mask` / `SFScale` runtime predicates, and the
`vecMax == 0` selects.

## Primitive vocabulary

Structural operations declare placement without moving data:

```python
specialize(...)       # compile-time variant selection
launch(...)           # compile-time launch topology and attributes
reg_tile(...)         # per-thread register tile
```

Copies state their direction and width:

```python
copy_g2r_v4_b64(src_addr, dst_b64x4)  # one 32-byte global -> register vector load
copy_g2r_scalar(src_addr, dst)        # one scalar (b32) global -> register load
copy_r2g_b64(src_u64, dst_addr)       # one 8-byte register -> global store
copy_r2g_b8(src_u8, dst_addr)         # one byte register -> global store
```

The compute vocabulary is deliberately primitive:

```python
cast(dst, src)                     # scalar cvt (f16/bf16 <-> f32, s/u int casts)
cast2x(dst_b32, lo, hi)            # pack two f32 into one f16x2/bf16x2 b32 register
abs_h2(dst_h2, src_h2)             # packed fp16x2/bf16x2 absolute value (__habs2)
max_h2(dst_h2, lhs, rhs)           # packed fp16x2/bf16x2 maximum (__hmax2)
max_h(dst_h, lhs, rhs)             # scalar fp16/bf16 maximum (__hmax)
mul(dst, lhs, rhs)
add(dst, lhs, rhs)
fdiv(dst, lhs, rhs)                # fp32 division; production -use_fast_math lowers
                                   # `/` to div.approx.ftz.f32 (MUFU.RCP + FMUL in
                                   # SASS), a plain -O3 build to div.rn.f32
rcp_ftz(dst, src)                  # rcp.approx.ftz.f32
exp2_fast(dst, src)                # ex2 approximation of __expf; production build
                                   # (-use_fast_math) lowers it to ex2.approx.ftz.f32
                                   # (bare MUFU.EX2 in SASS); a plain -O3 build emits
                                   # ex2.approx.f32, whose SASS adds a predicated
                                   # FSETP.GEU subnormal guard around each MUFU.EX2
cast_e4m3x2(dst_u16, hi, lo)       # cvt.rn.satfinite.e4m3x2.f32 pair conversion
cast_f16x2_e4m3x2(dst_u32, src)    # cvt.rn.f16x2.e4m3x2 pair decode
fp32_vec_to_e2m1(dst_u64, f32x8)   # 8x cvt.rn.satfinite.e2m1x2.f32 + byte pack
                                   # (source: one asm block with 4 x b8 mov.b32 packs;
                                   #  TIRx: b16-pair shl/or + mov.b32/mov.b64 packs, since
                                   #  the dialect does not register the 4 x b8 form)
select(dst, pred, a, b)            # selp family (b16/u16/u32 forms; nvcc lowers
                                   # f32 ternaries to branches with preloaded
                                   # defaults instead — branch-lowered sites are
                                   # annotated as such at the op)
idiv(dst, lhs, rhs)                # integer division family (div.s32/u32)
imod(dst, lhs, rhs)                # integer modulo family
move(dst, src)
```

All f32 arithmetic ops (`mul`/`add`/`fdiv`, the `vec_max` setp) are annotated
with their production `-use_fast_math` forms (`.ftz` modifiers, approximate
division); a plain `-O3` build emits the same opcodes without `.ftz` and with
round-to-nearest division. The audit evidence below comes from two fresh
exports of the exact instantiation: production
(`nvcc -ptx -lineinfo -arch=sm_100a -O3 -use_fast_math`, the flags the shipped
JIT cubins are built with — `build.ninja` evidence) and plain -O3.

`thread_id`, `cta_id`, `break_`, and pointer-null predicates are schedule
operations. Address expressions, loop bounds, and guards are shown directly;
they do not hide copies, computation, role changes, or synchronization.
Integer division/modulo by runtime values is kept as explicit `idiv`/`imod`
ops because the source emits `div.s32`/`div.u32` on runtime operands;
divisions by the compile-time constants 16/128/32/4 lower to shift/mask
families like the source's `shr.s32`/`and.b32` sites. Branch annotations use
the emitted polarity: nvcc compiles a C++ `if (c)` to the complementary
`setp` that selects the else/skip arm, so the annotated predicate is the
inverse of the C++ condition at that site.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

variant = specialize(DTYPE=("f16", "bf16"), target="sm_100a")
# instruction_selection: none; extent: two compile-time instantiations

ELTS = 16                  # CVT_FP16_TO_FP4_ELTS_PER_THREAD (device, CUDA >= 12.9)
SF_VEC = 16                # CVT_FP4_SF_VEC_SIZE
THREADS_PER_SF = 1         # SF_VEC / ELTS: each thread owns one SF block
GRID_X, BLOCK_X = host_launch_shape(N_EXPERTS, M, K)  # static per config; see host section
# instruction_selection: none; extent: static launch metadata

launch_config = launch(
    grid=(GRID_X, 1, 1),
    block=(BLOCK_X, 1, 1),
    launch_bounds=(512, 4),        # source __launch_bounds__(512, 4)
    dynamic_smem_bytes=0,
)
# instruction_selection: none; extent: static launch metadata

def silu_and_mul_nvfp4_experts_quantize(
    input,         # DTYPE [num_rows, 2*num_cols], direct global pointer
    sf_scale,      # f32 [n_experts], direct global pointer
    output,        # u64 [num_rows, num_cols/16], direct global pointer
    sf_out,        # u8 [n_experts * padded_m * padded_k_sf], direct global pointer
    mask,          # i32 [n_experts], direct global pointer
    num_rows,      # runtime i32 (= B * M)
    num_cols,      # runtime i32 (= K)
    n_experts,     # runtime i32 (= B)
    use_silu_and_mul,  # runtime bool (always true for this API)
):
    bx = cta_id(axis="x", extent=GRID_X)
    # instruction_selection: mov.u32 from %ctaid.x; extent: scalar per thread
    tx = thread_id(extent=BLOCK_X, dtype="uint32")
    # instruction_selection: mov.u32 from %tid.x; extent: scalar per thread

    # =======================================================================
    # Expert partition (quantization.cuh:642-663)
    # =======================================================================

    tid32 = bx * BLOCK_X + tx                 # source: blockIdx.x * blockDim.x + threadIdx.x
    # instruction_selection: mad.lo.s32; extent: scalar per thread
    stride = idiv(GRID_X * BLOCK_X, n_experts)
    # instruction_selection: div.u32 (unsigned gridDim.x*blockDim.x product); extent: scalar per thread
    part_rem = imod(GRID_X * BLOCK_X, n_experts)
    # instruction_selection: div.u32 + mul/sub pair (u32 modulo expansion); extent: scalar per thread
    if part_rem > 0:
        # instruction_selection: setp.lt.s32 + bra (emitted polarity: selects the else arm when part_rem < 1); extent: one branch
        bound = part_rem * (stride + 1)
        # instruction_selection: add.s32 + mul.lo.s32; extent: scalar per thread
        if tid32 < bound:
            # instruction_selection: setp.ge.s32 + bra (emitted polarity: selects the else arm when tid32 >= bound); extent: one branch
            expert_idx = idiv(tid32, stride + 1)
            # instruction_selection: div.s32; extent: scalar per thread
            tid_in_expert = imod(tid32, stride + 1)
            # instruction_selection: div.s32 + mul/sub pair; extent: scalar per thread
            actual_stride = stride + 1
            # instruction_selection: add.s32; extent: scalar per thread
        else:
            expert_idx = part_rem + idiv(tid32 - bound, stride)
            # instruction_selection: sub.s32 + div.s32 + add.s32; extent: scalar per thread
            tid_in_expert = imod(tid32 - bound, stride)
            # instruction_selection: sub.s32 + div.s32 + mul/sub pair; extent: scalar per thread
            actual_stride = stride
            # instruction_selection: none; extent: register move
    else:
        expert_idx = idiv(tid32, stride)
        # instruction_selection: div.s32; extent: scalar per thread
        tid_in_expert = imod(tid32, stride)
        # instruction_selection: div.s32 + mul/sub pair; extent: scalar per thread
        actual_stride = stride
        # instruction_selection: none; extent: register move

    m_rows = idiv(num_rows, n_experts)
    # instruction_selection: div.s32; extent: scalar per thread
    padded_m = (m_rows + 127) // 128 * 128
    # instruction_selection: add.s32 + shr/mask + mul family; extent: scalar per thread
    cols_per_row = idiv(num_cols, ELTS)
    # instruction_selection: shr.s32 (divisor is constant 16); extent: scalar per thread
    use_mask = mask != nullptr
    # instruction_selection: setp.eq.b64 (emitted polarity: selects the no-mask arm when null); extent: scalar per thread
    actual_cols_per_row = cols_per_row * 2 if use_silu_and_mul else cols_per_row
    # instruction_selection: shl.b32 x1 (silu branch) | none; extent: scalar per thread

    # =======================================================================
    # Grid-stride loop over this expert's chunks (quantization.cuh:675-720)
    # =======================================================================

    global_idx = tid_in_expert + expert_idx * m_rows * cols_per_row
    # instruction_selection: mul.lo.s32 x2 + add.s32; extent: scalar per thread
    loop_bound = (expert_idx + 1) * m_rows * cols_per_row
    # instruction_selection: add.s32 + mul.lo.s32 x2; extent: scalar per thread
    while global_idx < loop_bound:
        # instruction_selection: setp.lt.s32 + bra; extent: loop control, unroll hint 1
        row_idx = idiv(global_idx, cols_per_row)
        # instruction_selection: div.s32; extent: scalar per thread per iteration
        col_idx = imod(global_idx, cols_per_row)
        # instruction_selection: div.s32 + mul/sub pair; extent: scalar per thread per iteration
        row_idx_in_expert = row_idx - expert_idx * m_rows
        # instruction_selection: mul.lo.s32 + sub.s32; extent: scalar per thread per iteration

        if use_mask:
            # instruction_selection: predicated block on the null predicate; extent: per iteration
            mask_val = copy_g2r_scalar(mask[expert_idx])
            # instruction_selection: ld.global.b32; extent: one scalar load per iteration
            if row_idx_in_expert >= mask_val:
                # instruction_selection: setp.ge.s32 + bra (loop exit); extent: per iteration
                break_()

        in_offset = i64(row_idx) * actual_cols_per_row + col_idx
        # instruction_selection: mul.wide.s32 + add.s64 family; extent: scalar per thread per iteration

        x_bits = reg_tile("b64", [4])
        # instruction_selection: none; extent: four b64 registers per thread
        copy_g2r_v4_b64(input + in_offset * ELTS, x_bits)
        # instruction_selection: ld.global.v4.b64; extent: one 32-byte vector load (x chunk)

        if use_silu_and_mul:
            # instruction_selection: ld.param.s8-derived predicate + bra; extent: per iteration
            y_bits = reg_tile("b64", [4])
            # instruction_selection: none; extent: four b64 registers per thread
            copy_g2r_v4_b64(input + (in_offset + cols_per_row) * ELTS, y_bits)
            # instruction_selection: ld.global.v4.b64; extent: one 32-byte vector load (y chunk)
            SILU_AND_MUL(x_bits, y_bits)     # sequence below; overwrites x_bits
            # instruction_selection: see silu block; extent: 16 elements

        out_offset = i64(row_idx) * cols_per_row + col_idx
        # instruction_selection: mul.wide.s32 + add.s64 family; extent: scalar per thread per iteration

        sfscale_val = select(sf_scale != nullptr, copy_g2r_scalar(sf_scale[expert_idx]), 1.0)
        # instruction_selection: branch-lowered: setp.eq.b64 + bra over the guarded ld.global.b32 with preloaded 1.0f default (no selp.f32 exists); extent: one scalar load + select per iteration

        # ===================================================================
        # SF swizzled output address (utils:1096-1140 + quantization.cuh:706-714)
        # ===================================================================

        num_cols_padded = (num_cols + SF_VEC * 4 - 1) // (SF_VEC * 4) * (SF_VEC * 4)
        # instruction_selection: add.s32 + shr/mask family; extent: scalar per thread per iteration (loop-invariant in practice)
        num_cols_sfout = num_cols_padded // SF_VEC // 4
        # instruction_selection: shr.s32 family; extent: scalar per thread
        sf_expert_base = expert_idx * padded_m * num_cols_sfout
        # instruction_selection: mul.lo.s32 x2 (uint32 pointer units); extent: scalar per thread
        num_k_tiles = (num_cols + SF_VEC * 4 - 1) // (SF_VEC * 4)
        # instruction_selection: add.s32 + shr.s32 family; extent: scalar per thread
        sf_off = ((row_idx_in_expert // 128) * (num_k_tiles * 512)
                  + (col_idx // 4) * 512
                  + (row_idx_in_expert % 32) * 16
                  + ((row_idx_in_expert % 128) // 32) * 4
                  + (col_idx % 4))
        # instruction_selection: div/mod by constants (shr.s32, and.b32) + mul/add 64-bit accumulation (mad.lo.s64 family); extent: scalar per thread per iteration
        sf_addr = sf_out + (sf_expert_base * 4) + sf_off      # byte address
        # instruction_selection: add.s64 family; extent: scalar per thread per iteration

        # ===================================================================
        # cvt_warp_fp16_to_fp4 (utils:447-508, 660-683), THREADS_PER_SF == 1
        # ===================================================================

        # --- local abs-max over the 8 packed pairs (silu-rounded values) ---
        local_max = abs_h2(x_bits.pair[0])
        # instruction_selection: abs.f16x2 (f16) | abs.bf16x2 (bf16); extent: one packed pair
        for i in static_range(1, 8):
            local_max = max_h2(local_max, abs_h2(x_bits.pair[i]))
            # instruction_selection: abs.f16x2 + max.f16x2 (bf16x2 for bf16); extent: 7 rounds
        vec_max = cast("f32", max_h(local_max.lo, local_max.hi))
        # instruction_selection: setp.gt.f16 + selp.b16 (__hmax lowering) + cvt.f32.f16 (bf16: setp.gt.bf16 + selp + cvt.f32.bf16); extent: one scalar

        # --- SF computation (default env: fast-math rcp, E4M3) ---
        sf_value = mul(sfscale_val, mul(vec_max, rcp_ftz(6.0)))
        # instruction_selection: rcp.approx.ftz.f32 + mul.ftz.f32 x2 (production form); extent: one scalar
        sf_bits_u16 = cast_e4m3x2(0.0, sf_value)          # lo = e4m3(sf_value)
        # instruction_selection: cvt.rn.satfinite.e4m3x2.f32; extent: one packed pair (lo byte used)
        fp8_sf_val = low_u8(sf_bits_u16)
        # instruction_selection: mov/cvt.u16.u32 byte extract family; extent: one byte
        sf_f16x2 = cast_f16x2_e4m3x2(sf_bits_u16)
        # instruction_selection: cvt.rn.f16x2.e4m3x2; extent: one packed pair
        sf_value_r = cast("f32", sf_f16x2.lo)
        # instruction_selection: mov.b32 {lo,hi} + cvt.f32.f16; extent: one scalar
        output_scale = select(vec_max != 0.0,
                              rcp_ftz(mul(sf_value_r, rcp_ftz(sfscale_val))),
                              0.0)
        # instruction_selection: branch-lowered: setp.eq.ftz.f32 + bra with preloaded 0.0f default (production form; plain -O3 shows setp.eq.f32), guarding rcp.approx.ftz.f32 x2 + mul.ftz.f32 in the non-zero arm; extent: one scalar

        # --- SF byte store (STG.8, per thread; predicate is the always-true
        #     THREADS_PER_SF==1 slot plus the SFout non-null check) ---
        if sf_addr != nullptr:
            # instruction_selection: setp.eq.b64 + bra (emitted polarity: skips the store when null); extent: per iteration
            copy_r2g_b8(fp8_sf_val, sf_addr)
            # instruction_selection: st.global.b8; extent: one byte store per iteration

        # --- scale to e2m1 and pack (fp32_vec_to_e2m1; source asm block, native TIRx pack) ---
        for i in static_range(8):
            f_lo = cast("f32", x_bits.pair[i].lo)
            # instruction_selection: cvt.f32.f16 (bf16: cvt.f32.bf16); extent: one scalar
            f_hi = cast("f32", x_bits.pair[i].hi)
            # instruction_selection: cvt.f32.f16 (bf16: cvt.f32.bf16); extent: one scalar
            fp2[i].lo = mul(f_lo, output_scale)
            # instruction_selection: mul.ftz.f32; extent: one scalar (production form)
            fp2[i].hi = mul(f_hi, output_scale)
            # instruction_selection: mul.ftz.f32; extent: one scalar (production form)
        e2m1_vec = fp32_vec_to_e2m1(fp2)
        # instruction_selection: cvt.rn.satfinite.e2m1x2.f32 x8 + mov.b32 {b0..b3} x2 + mov.b64 {v0,v1} byte pack; extent: one uint64
        copy_r2g_b64(e2m1_vec, output + out_offset)
        # instruction_selection: st.global.b64; extent: one 8-byte store per iteration

        global_idx += actual_stride
        # instruction_selection: add.s32; extent: loop induction update

# ===========================================================================
# SILU_AND_MUL (utils:1142-1166) — silu(x) * y per element, rounded back to
# DTYPE pairs; overwrites the x register pairs in place
# ===========================================================================

def SILU_AND_MUL(x_bits, y_bits):              # 8 packed pairs each
    for i in static_range(8):
        x_lo = cast("f32", x_bits.pair[i].lo)
        # instruction_selection: cvt.f32.f16 (bf16: cvt.f32.bf16); extent: one scalar
        x_hi = cast("f32", x_bits.pair[i].hi)
        # instruction_selection: cvt.f32.f16 (bf16: cvt.f32.bf16); extent: one scalar
        y_lo = cast("f32", y_bits.pair[i].lo)
        # instruction_selection: cvt.f32.f16 (bf16: cvt.f32.bf16); extent: one scalar
        y_hi = cast("f32", y_bits.pair[i].hi)
        # instruction_selection: cvt.f32.f16 (bf16: cvt.f32.bf16); extent: one scalar
        out_lo = SILU_ONE(x_lo) * y_lo
        # instruction_selection: mul.ftz.f32 (production form, after the silu sequence below); extent: one scalar
        out_hi = SILU_ONE(x_hi) * y_hi
        # instruction_selection: mul.ftz.f32 (production form); extent: one scalar
        x_bits.pair[i] = cast2x(out_lo, out_hi)
        # instruction_selection: cvt.rn.f16x2.f32 (bf16: cvt.rn.bf16x2.f32); extent: one packed pair

def SILU_ONE(v):                               # source: v / (1.0f + __expf(-v))
    t = mul(v, -1.4426950408889634)
    # instruction_selection: mul.ftz.f32; extent: one scalar (production -use_fast_math form)
    e = exp2_fast(t)
    # instruction_selection: ex2.approx.ftz.f32; extent: one scalar (production -use_fast_math lowering of __expf; bare MUFU.EX2 in production SASS)
    d1 = add(1.0, e)
    # instruction_selection: add.ftz.f32; extent: one scalar (production -use_fast_math form)
    return fdiv(v, d1)
    # instruction_selection: div.approx.ftz.f32; extent: one scalar (production lowering of `/`: MUFU.RCP + FMUL in SASS; plain -O3 export shows div.rn.f32)
```

## Host wrapper and validation

The Python module performs host-only work; none of it emits device PTX:

```python
def host_launch_shape(n_experts, m, k):
    # quantization.cu:729-745, with the host TU's ELTS_PER_THREAD == 8
    wspr = max(1, k // 8)
    block = min(wspr, 512)
    blocks_per_sm = 2048 // block
    grid = min(ceildiv(n_experts * m * wspr, block), SM_COUNT * blocks_per_sm)
    while grid <= SM_COUNT and block > 64:
        grid *= 2
        block = (block + 1) // 2
    grid = round_up(grid, n_experts)
    # instruction_selection: none; extent: host-only arithmetic (SM_COUNT=148 on B200)
    return grid, block

def prepare_data(dtype, n_experts, m, k, mask_mode):
    host_assert(k % 16 == 0 and k > 0 and m >= 1 and n_experts >= 1)
    a = contiguous_seeded_randn(shape=(n_experts, m, 2 * k), dtype=dtype)
    mask = seeded_randint(low=1, high=m + 1, shape=(n_experts,), dtype="int32")  # or full m
    global_scale = seeded_uniform_fp32(shape=(n_experts,)) * 1.0 + 0.5
    # instruction_selection: none; extent: tensor constructions

# run_test compares packed outputs and swizzled SF bytes against the source
# API, restricted to slots the kernel actually writes (rows < mask[e],
# kIdx < k/16); both sides leave the rest as uninitialized torch.empty bytes.
# run_bench times the primfunc launch (tirx) against the source thop
# silu_and_mul_scaled_nvfp4_experts_quantize with preallocated outputs.
```

## Static specialization boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| `DTYPE` | static per config | selects f16/bf16 cvt opcode families (`cvt.f32.f16` vs `cvt.f32.bf16`, `cvt.rn.f16x2.f32` vs `cvt.rn.bf16x2.f32`, `abs/max` pair opcodes) |
| grid/block extents | static per config | host formula baked in; kernel reads `%ctaid.x`/`%tid.x` only |
| `num_rows`, `num_cols`, `n_experts` | runtime i32 ABI | partition div/mod, row/col decode, SF strides stay runtime `div.s32`/`shr.s32` like the source |
| `use_silu_and_mul` | runtime bool ABI (always true) | runtime branch preserved; the non-silu path exists in the binary like the source |
| `use_mask`, `SFScale` null checks | runtime predicates | source predicates preserved even though both are non-null for this API |
| `UE8M0_SF=false`, `DISABLE_FP4_QUANT_FAST_MATH=false`, `4OVER6=false_type` | static | default-env code path only |
| `ELTS=16`, `SF_VEC=16`, `THREADS_PER_SF=1` | static | no `__shfl_xor_sync` reduction; one SF byte per thread |
| `__launch_bounds__(512, 4)` | static | register budget contract |

## TIRx module and benchmark contract

- `KERNEL_META = {"name": "silu_and_mul_nvfp4_experts_quantize", "category":
  "flashinfer", "compute_capability": 10}`.
- The executable kernel is expressed entirely in plain TIRx: explicit `while`
  grid-stride loop, runtime-shape scalar ABI, register tiles, and native
  `T.ptx.*` forms for every non-trivial instruction (`ld.global.v4.b64`,
  `ex2.approx.ftz.f32`, `abs.f16x2`/`max.f16x2`, `setp.gt.f16` + `selp.b16`
  for scalar `__hmax`, the `cvt.rn.satfinite.*` conversions, `rcp.approx.ftz.f32`).
  The `fp32_vec_to_e2m1` pack is native too: eight
  `cvt.rn.satfinite.e2m1x2.f32` into b8 locals, then b16-pair shifts plus the
  registered `mov.b32` (2 x b16) and `mov.b64` (2 x b32) packs — the dialect
  deliberately does not register the source asm's 4 x b8 `mov.b32` form.
  There is no `T.cuda.func_call` and no `Tx` tile primitives.
- `get_kernel(dtype, n_experts, m, k, mask_mode)` returns the specialized
  primfunc with static grid/block; `prepare_data`, `run_test`, `run_bench`
  follow the repository contract.
- The timed implementation is named `tirx`; the reference is the source thop
  `silu_and_mul_scaled_nvfp4_experts_quantize` (kernel-only, preallocated
  outputs, `use_silu_and_mul=True`). Allocation, compilation, and correctness
  checks stay outside timing.
- Correctness compares only slots the kernel writes (rows `< mask[e]`, SF
  bytes for `kIdx < k/16`), exact byte equality, against the source API.

## Instruction selection is a lowering consequence

The sketch above never requests a hardware instruction beyond the documented
PTX helpers. The following lowering families follow from storage direction,
shape, dtype, and schedule. PTX names and static counts are taken from fresh
line-info PTX exports of the exact source instantiation
(`cvt_fp16_to_fp4_expert<half, false, false, std::false_type>`, `nvcc -ptx
-lineinfo -arch=sm_100a -O3 -use_fast_math -DENABLE_BF16 -DENABLE_FP8
-DENABLE_FP4`, the flags the shipped JIT cubins are built with — `build.ninja`
evidence; a plain `-O3` export without `-use_fast_math` was also made for the
modifier-only deltas); they are audit evidence, not operands.

| Primitive/schedule pattern | PTX family (fresh SM100a exports) |
| --- | --- |
| `copy_g2r_v4_b64` x/y loads | `ld.global.v4.b64` (2 per iteration, no `.nc`) |
| `copy_g2r_scalar` mask / SFScale | `ld.global.b32` (2 per iteration) |
| silu per element | `mul.ftz.f32` + `ex2.approx.ftz.f32` + `add.ftz.f32` + `div.approx.ftz.f32` + `mul.ftz.f32` (production, x16 each; plain -O3 drops the `.ftz` modifiers, uses `div.rn.f32`, and its SASS adds a predicated FSETP.GEU subnormal guard around each MUFU.EX2) |
| silu result rounding | `cvt.rn.f16x2.f32` / `cvt.rn.bf16x2.f32` (8 per iteration) |
| packed abs/max chain | `abs.f16x2` x8 + `max.f16x2` x7 (bf16: bf16x2 forms) |
| scalar `__hmax` + promotion | `setp.gt.f16` + `selp.b16` + `cvt.f32.f16` |
| SF scale ops | `rcp.approx.ftz.f32` x3, `mul.ftz.f32` x3 (production form), branch-lowered selects (`setp.eq.b64`/`setp.eq.ftz.f32` + `bra`, preloaded defaults) |
| E4M3 SF convert | `cvt.rn.satfinite.e4m3x2.f32` x1 + `cvt.rn.f16x2.e4m3x2` x1 + byte extract |
| quantize scale | `cvt.f32.f16` x16 + `mul.ftz.f32` x16 (production form) |
| e2m1 pack | `cvt.rn.satfinite.e2m1x2.f32` x8 + byte-pack movs |
| stores | `st.global.b8` x1 (SF), `st.global.b64` x1 (output) |
| partition / decode integer ops | `div.s32` x5, `div.u32` x1, `shr.s32`/`and.b32` for constant divisors, `mul.wide.s32`/`add.s64` for 64-bit offsets |
| loop control | `setp` + `bra`, `add.s32` induction |

Static PTX opcode histogram of the exported fp16 instantiation (production
flags; whole kernel, one loop body): `mul.ftz.f32` 51, `cvt.f32.f16` 50,
`add.ftz.f32`/`div.approx.ftz.f32`/`ex2.approx.ftz.f32` 16 each,
`setp.eq.ftz.f32` 1, `abs.f16x2` 8, `max.f16x2` 7,
`cvt.rn.satfinite.e2m1x2.f32` 8, `cvt.rn.f16x2.f32` 8, `ld.global.v4.b64` 2,
`ld.global.b32` 2, `rcp.approx.ftz.f32` 3, `st.global.b8` 1,
`st.global.b64` 1, `cvt.rn.satfinite.e4m3x2.f32` 1, `cvt.rn.f16x2.e4m3x2` 1,
`div.s32` 5, `div.u32` 1. The bf16 instantiation differs only in the
f16/bf16 cvt and packed-op suffixes.
