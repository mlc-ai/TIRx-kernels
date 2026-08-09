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
include/flashinfer/activation.cuh (act_and_mul_kernel), the single
template kernel behind flashinfer.activation.silu_and_mul,
gelu_and_mul, and gelu_tanh_and_mul.
-->

# act_and_mul SM100: coarse WASP pipeline sketch

This non-executable design sketch describes the storage layout, thread roles,
control flow, and PTX-level operations of
[`tirx_kernels/flashinfer/act_and_mul.py`](../../tirx_kernels/flashinfer/act_and_mul.py).
That TIRx module is the authoritative implementation.

The six instantiations are `ACT in {silu, gelu, gelu_tanh}` crossed with
`DTYPE in {f16, bf16}`; each mirrors one `act_and_mul_kernel<T, Activation>`
template instantiation of the source. `num_tokens`, `d`, and the derived block
size are static per config, exactly like the per-shape JIT launch the source
host wrapper computes. The accepted target is SM100/B200. Inputs outside the
source's vectorized dispatch domain (`d % 8 != 0` or `d < 8`), FP32 or FP8
dtypes, clusters, shared-memory staging, and tile (`Tx`) primitives are out of
scope. `enable_pdl` is a host-side launch attribute: the kernel body always
contains both `griddepcontrol` instructions on SM90+, and the attribute itself
selects no code.

## Pipeline at a glance

| Warps | Role-local program | Publication/reuse edges |
| --- | --- | --- |
| all (uniform) | Every thread of CTA `t` runs the same single-role program: PDL wait, grid-stride vector loop over `d/8` 16-byte chunks (two vector loads, 8 fp32 activations-times-mul, one vector store), then a scalar remainder loop over `d % (stride*8)` elements, then PDL launch-dependents. | none — no SMEM, no mbarriers, no cross-thread data; the only ordering edges are `griddepcontrol.wait` (entry) and `griddepcontrol.launch_dependents` (exit) |

There is one CTA per token (`blockIdx.x == token_idx`) and no warp
specialization; the source-order loop structure below is the whole kernel.

## Primitive vocabulary

Structural operations declare placement without moving data:

```python
specialize(...)       # compile-time variant selection
launch(...)           # compile-time launch topology and attributes
reg_tile(...)         # per-thread register tile
```

Copies state their direction and width:

```python
copy_g2r_v4(src_addr, dst_b32x4)  # one 16-byte global -> register vector load
copy_g2r_scalar(src_addr, dst)    # one scalar (b16) global -> register load
copy_r2g_v4(src_b32x4, dst_addr)  # one 16-byte register -> global vector store
copy_r2g_scalar(src, dst_addr)    # one scalar (b16) register -> global store
```

The compute vocabulary is deliberately primitive:

```python
cast(dst, src)              # scalar cvt between f32 and f16/bf16
cast2x(dst_b32, lo, hi)     # pack two f32 into one f16x2/bf16x2 b32 register
mul(dst, lhs, rhs)
add(dst, lhs, rhs)
fma(dst, lhs, rhs, acc)
div(dst, lhs, rhs)          # fp32 division; production -use_fast_math lowers
                            # `/` to div.approx.ftz.f32 (MUFU.RCP + FMUL in
                            # SASS), a plain -O3 build to div.rn.f32
exp2_fast(dst, src)         # ex2 approximation of __expf; production build
                            # (-use_fast_math) lowers it to ex2.approx.ftz.f32
                            # (bare MUFU.EX2 in SASS); a plain -O3 build emits
                            # ex2.approx.f32, whose SASS adds a predicated
                            # FSETP.GEU subnormal guard around each MUFU.EX2
tanh_fast(dst, src)         # tanh approximation instruction
erf_libdevice(dst, src)     # libdevice erff inline expansion (see note)
move(dst, src)
```

All f32 arithmetic ops (`mul`/`add`/`fma`/`div`, `setp` sites) are annotated
with their production `-use_fast_math` forms (`.ftz` modifiers, approximate
division); a plain `-O3` build emits the same opcodes without `.ftz` and with
round-to-nearest division. The audit evidence below comes from two fresh
exports of the exact instantiation: production
(`nvcc -ptx -lineinfo -arch=sm_100a -O3 -use_fast_math`, the flags the shipped
JIT cubins are built with — `build.ninja` evidence) and plain -O3.

`erf_libdevice` denotes the inlined `::erf(float)` device-library expansion
emitted by nvcc at every call site; its exact mixed sequence (abs/setp/selp/mul/
fma plus a predicated small-argument branch around the neg/ex2/sub tail and a
final copysign) is fixed by libdevice, is identical for every element, and is
recorded in the instruction-selection summary from line-info PTX evidence. It
is used as one op because the source calls `::erf` as one unmodifiable library
call; splitting it would invent structure the source does not have.

`thread_id`, `cta_id`, `pdl_wait`, and `pdl_launch_dependents` are schedule
operations. Address expressions, loop bounds, and guards are shown directly;
they do not hide copies, computation, role changes, or synchronization.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

variant = specialize(ACT=("silu", "gelu", "gelu_tanh"), DTYPE=("f16", "bf16"),
                     target="sm_100a")
# instruction_selection: none; extent: six compile-time instantiations

VEC = 8                     # 16 bytes / 2 bytes per element
BLOCK = min(D // VEC, 1024) # static per config; source host formula
STRIDE = BLOCK              # source names blockDim.x as stride
N_VEC = D // VEC            # vectorized chunk count
REM = D % (STRIDE * VEC)    # remainder element count; 0 unless D/8 > 1024 and D % 8192 != 0
REM_OFF = D - REM           # remainder start offset within a row

launch_config = launch(
    grid=(NUM_TOKENS, 1, 1),
    block=(BLOCK, 1, 1),
    dynamic_smem_bytes=0,
    programmatic_dependent_launch=HOST_ENABLE_PDL,  # host attribute; no code effect
)
# instruction_selection: none; extent: static launch metadata

def act_and_mul(input,      # DTYPE [NUM_TOKENS, 2*D], direct global pointer
                output,     # DTYPE [NUM_TOKENS, D], direct global pointer
                ):
    token = cta_id(axis="x", extent=NUM_TOKENS)
    # instruction_selection: mov.u32 from %ctaid.x; extent: scalar per thread
    tid = thread_id(extent=BLOCK, dtype="uint32")
    # instruction_selection: mov.u32 from %tid.x; extent: scalar per thread

    pdl_wait()
    # instruction_selection: griddepcontrol.wait; extent: every thread, kernel entry

    # =======================================================================
    # Main vector loop (source: #pragma unroll 1 grid-stride loop)
    # =======================================================================

    idx = tid                      # uint32 chunk index; per-thread start
    while idx < N_VEC:             # per-thread trip count ceil((N_VEC - tid)/STRIDE)
        # instruction_selection: none; extent: loop control only (setp/bra), unroll hint 1

        x_addr = input + token * (2 * D) + idx * VEC          # 16B-aligned
        y_addr = input + token * (2 * D) + D + idx * VEC      # 16B-aligned
        o_addr = output + token * D + idx * VEC               # 16B-aligned
        # instruction_selection: 64-bit integer address family (mul.lo.s64, mad.lo.s64, shl.b64, add.s64, or.b64; cvta.to.global.u64 is entry-only, not per iteration); extent: per iteration

        x_bits = reg_tile("b32", [4])
        # instruction_selection: none; extent: four b32 registers per thread
        copy_g2r_v4(x_addr, x_bits)
        # instruction_selection: ld.global.nc.v4.b32; extent: one 16-byte vector load
        y_bits = reg_tile("b32", [4])
        # instruction_selection: none; extent: four b32 registers per thread
        copy_g2r_v4(y_addr, y_bits)
        # instruction_selection: ld.global.nc.v4.b32; extent: one 16-byte vector load

        x_vec = reg_tile("f32", [8])
        y_vec = reg_tile("f32", [8])
        # instruction_selection: none; extent: sixteen f32 registers per thread
        for i in static_range(8):
            cast(x_vec[i], x_bits.f16_pair[i])
            # instruction_selection: cvt.f32.f16 (DTYPE=f16) | cvt.f32.bf16 (DTYPE=bf16); extent: one scalar per element, 8 total
        for i in static_range(8):
            cast(y_vec[i], y_bits.f16_pair[i])
            # instruction_selection: cvt.f32.f16 (DTYPE=f16) | cvt.f32.bf16 (DTYPE=bf16); extent: one scalar per element, 8 total

        out_vec = reg_tile("f32", [8])
        # instruction_selection: none; extent: eight f32 registers per thread
        for i in static_range(8):
            ACTIVATE(out_vec[i], x_vec[i], y_vec[i])          # ACT-specific sequence below
        o_bits = reg_tile("b32", [4])
        # instruction_selection: none; extent: four packed output registers per thread
        for p in static_range(4):
            cast2x(o_bits[p], out_vec[2 * p], out_vec[2 * p + 1])
            # instruction_selection: cvt.rn.f16x2.f32 (DTYPE=f16) | cvt.rn.bf16x2.f32 (DTYPE=bf16); extent: one packed pair conversion, 4 total
        copy_r2g_v4(o_bits, o_addr)
        # instruction_selection: st.global.v4.b32; extent: one 16-byte vector store
        idx = idx + STRIDE

    # =======================================================================
    # Scalar remainder loop (source: #pragma unroll 1; dead when REM == 0)
    # =======================================================================

    ridx = tid
    while ridx < REM:              # statically unreachable when REM == 0
        # instruction_selection: none; extent: loop control only (setp/bra), unroll hint 1
        xr = reg_tile("f32", [1])
        yr = reg_tile("f32", [1])
        # instruction_selection: none; extent: two f32 registers per thread
        xr_b = reg_tile("b16", [1])
        yr_b = reg_tile("b16", [1])
        # instruction_selection: none; extent: two b16 registers per thread
        copy_g2r_scalar(input + token * (2 * D) + REM_OFF + ridx, xr_b)
        # instruction_selection: ld.global.nc.b16; extent: one scalar load
        copy_g2r_scalar(input + token * (2 * D) + REM_OFF + D + ridx, yr_b)
        # instruction_selection: ld.global.nc.b16; extent: one scalar load
        cast(xr[0], xr_b[0])
        # instruction_selection: cvt.f32.f16 | cvt.f32.bf16; extent: one scalar
        cast(yr[0], yr_b[0])
        # instruction_selection: cvt.f32.f16 | cvt.f32.bf16; extent: one scalar
        or_ = reg_tile("f32", [1])
        # instruction_selection: none; extent: one f32 register per thread
        ACTIVATE(or_[0], xr[0], yr[0])                        # same ACT sequence as vector path
        or_b = reg_tile("b16", [1])
        # instruction_selection: none; extent: one b16 register per thread
        cast(or_b[0], or_[0])
        # instruction_selection: cvt.rn.f16.f32 (DTYPE=f16) | cvt.rn.bf16.f32 (DTYPE=bf16); extent: one scalar
        copy_r2g_scalar(or_b[0], output + token * D + REM_OFF + ridx)
        # instruction_selection: st.global.b16; extent: one scalar store
        ridx = ridx + STRIDE

    pdl_launch_dependents()
    # instruction_selection: griddepcontrol.launch_dependents; extent: every thread, kernel exit

# ===========================================================================
# ACT = silu (source: val / (1.0f + __expf(-val)))
# ===========================================================================

def ACTIVATE_silu(o, x, y):      # per scalar element
    t = mul(x, -1.4426950408889634)
    # instruction_selection: mul.ftz.f32; extent: one scalar (production -use_fast_math form; plain -O3 export shows mul.f32)
    e = exp2_fast(t)
    # instruction_selection: ex2.approx.ftz.f32; extent: one scalar (production -use_fast_math lowering of __expf; bare MUFU.EX2 in production SASS)
    d1 = add(1.0, e)
    # instruction_selection: add.ftz.f32; extent: one scalar (production -use_fast_math form)
    s = div(x, d1)
    # instruction_selection: div.approx.ftz.f32; extent: one scalar (production -use_fast_math lowering of `/`: MUFU.RCP + FMUL in SASS; plain -O3 export shows div.rn.f32)
    mul(o, s, y)
    # instruction_selection: mul.ftz.f32; extent: one scalar (production form)

# ===========================================================================
# ACT = gelu (source: val * 0.5f * (1.0f + ::erf(val * M_SQRT1_2)))
# ===========================================================================

def ACTIVATE_gelu(o, x, y):      # per scalar element
    t = mul(x, 0.7071067811865476)
    # instruction_selection: mul.ftz.f32; extent: one scalar (production form)
    e = erf_libdevice(t)
    # instruction_selection: libdevice erff inline expansion per call (production line-info PTX evidence): abs.ftz.f32 x2, setp.ltu.ftz.f32 x1, setp.ge.ftz.f32 x1, mul.ftz.f32 x1, selp.f32 x8, fma.rn.ftz.f32 x7, one predicated bra (small-|t| path skips the exp tail), neg.ftz.f32 x1, ex2.approx.ftz.f32 x1, sub.ftz.f32 x1, copysign.f32 x1; extent: one scalar erf
    a = add(1.0, e)
    # instruction_selection: add.ftz.f32; extent: one scalar (production form)
    h = mul(x, 0.5)
    # instruction_selection: mul.ftz.f32; extent: one scalar (production form)
    g = mul(h, a)
    # instruction_selection: mul.ftz.f32; extent: one scalar (production form)
    mul(o, g, y)
    # instruction_selection: mul.ftz.f32; extent: one scalar (production form)

# ===========================================================================
# ACT = gelu_tanh (source: val * 0.5f * (1.0f + tanh(0.7978845608028654f *
#                    (val + 0.044715f * val * val * val))))
# ===========================================================================

def ACTIVATE_gelu_tanh(o, x, y):  # per scalar element
    t1 = mul(x, 0.044715)
    # instruction_selection: mul.ftz.f32; extent: one scalar (production form; source order: 0.044715f*val first)
    t2 = mul(x, t1)
    # instruction_selection: mul.ftz.f32; extent: one scalar (production form; 0.044715f*val*val)
    u = fma(x, t2, x)
    # instruction_selection: fma.rn.ftz.f32; extent: one scalar (production form; val + 0.044715f*val*val*val)
    w = mul(u, 0.7978845608028654)
    # instruction_selection: mul.ftz.f32; extent: one scalar (production form)
    h = tanh_fast(w)
    # instruction_selection: tanh.approx.f32; extent: one scalar
    a = add(1.0, h)
    # instruction_selection: add.ftz.f32; extent: one scalar (production form)
    c = mul(a, 0.5)
    # instruction_selection: mul.ftz.f32; extent: one scalar (production form; cdf = 0.5f*(1.0f+tanh), source order)
    g = mul(x, c)
    # instruction_selection: mul.ftz.f32; extent: one scalar (production form; val*cdf)
    mul(o, g, y)
    # instruction_selection: mul.ftz.f32; extent: one scalar (production form)
```

## Host wrapper and validation

The Python module performs host-only work; none of it emits device PTX:

```python
def prepare_data(act, dtype, num_tokens, d):
    host_assert(act in ("silu", "gelu", "gelu_tanh"))
    host_assert(dtype in ("float16", "bfloat16"))
    host_assert(d >= 8 and d % 8 == 0)   # source vectorized dispatch domain
    input = contiguous_seeded_randn(shape=(num_tokens, 2 * d), dtype=dtype)
    # instruction_selection: none; extent: one tensor construction

launch_args = (input, output)
# instruction_selection: none; extent: flat two-pointer launch ABI

# run_test compares against flashinfer.activation.<act>_and_mul with the
# source test tolerance rtol=1e-3, atol=1e-3.
# run_bench times the primfunc launch (tirx) against the flashinfer JIT
# module launched with enable_pdl=False; both closures are no-argument.
```

## Static specialization boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| `ACT` | static per config | selects one of three device-function bodies (separate cubins in the source JIT, separate primfunc instantiations here) |
| `DTYPE` | static per config | selects f16/bf16 cvt opcode families; same layout |
| `num_tokens`, `d` | static per config | grid extent, `BLOCK = min(d/8, 1024)`, `N_VEC`, `REM`, `REM_OFF` constant-fold |
| `REM == 0` | static per config | remainder loop is compile-time eliminated for most configs; live only when `d/8 > 1024 and d % 8192 != 0` (e.g. d=11008) |
| PDL attribute | host-side | `griddepcontrol.wait`/`launch_dependents` always emitted on SM100; attribute selects no code |
| unroll hints | static | both loops carry the source's `#pragma unroll 1`; the 8-wide element loop is fully unrolled |
| dtype widths, 16-byte vector width, SM100a | static | preserves the 128-bit memory contract |

Automatic dispatch is outside the kernel: the source host wrapper computes
`d`, `num_tokens`, and the block size from tensor shape at every call; this
module bakes the same values per config.

## TIRx module and benchmark contract

- `KERNEL_META = {"name": "act_and_mul", "category": "flashinfer",
  "compute_capability": 10}`.
- The executable kernel is expressed entirely in plain TIRx: explicit
  `while` grid-stride loops, scalar/vector register buffers, explicit
  global loads/stores, and native `T.ptx.*` forms for every non-trivial
  instruction (`griddepcontrol`, `ld.global.nc.v4.b32`, `ex2.approx.ftz.f32`,
  `tanh.approx.f32`, `fma.rn.ftz.f32`, the cvt packs). There is no
  `T.cuda.func_call` and no `Tx` tile primitives anywhere in the
  pre-dispatch IR.
- `get_kernel(act, dtype, num_tokens, d)` returns the specialized primfunc;
  `prepare_data`, `run_test`, `run_bench` follow the repository contract.
- The timed implementation is named `tirx`; flashinfer is a lazy reference
  builder launched with `enable_pdl=False`. Allocation, compilation, and
  correctness checks stay outside timing.
- Correctness reference is the source implementation itself
  (`flashinfer.activation.{silu,gelu,gelu_tanh}_and_mul`), tolerance
  `rtol=1e-3, atol=1e-3` as in the source test suite.

## Instruction selection is a lowering consequence

The sketch above never requests a hardware instruction beyond the two PDL
intrinsics and the documented fast-math wrappers. The following lowering
families follow from storage direction, shape, dtype, and schedule. PTX names
and static counts are taken from fresh line-info PTX exports of the exact
source instantiations (explicit template instantiation of
`act_and_mul_kernel<T, Activation>`) under two flag sets: the production JIT
flags (`nvcc -ptx -lineinfo -arch=sm_100a -O3 -use_fast_math`, matching the
shipped JIT cubins' `build.ninja`) and plain `-O3` (no fast math); they are
audit evidence, not operands.

| Primitive/schedule pattern | PTX family (fresh SM100a exports) |
| --- | --- |
| `copy_g2r_v4` x/y loads | `ld.global.nc.v4.b32` (2 per vector iteration) |
| element casts f16/bf16 -> f32 | `cvt.f32.f16` / `cvt.f32.bf16` (16 per vector iteration, 2 per remainder iteration) |
| silu `exp2_fast` (`__expf`) | `ex2.approx.ftz.f32` (production); `ex2.approx.f32` (plain -O3, whose SASS adds a predicated FSETP.GEU subnormal guard) |
| silu division | `div.approx.ftz.f32` (production); `div.rn.f32` (plain -O3) |
| silu mul/add | `mul.ftz.f32` / `add.ftz.f32` (production); unmodified forms under plain -O3 |
| gelu `erf_libdevice` (`::erf`) | inline libdevice sequence per call (production): `abs.ftz.f32` x2, `setp.ltu.ftz.f32`/`setp.ge.ftz.f32` x1 each, `mul.ftz.f32` x1, `selp.f32` x8, `fma.rn.ftz.f32` x7, one predicated `bra` (small-|t| path skips the exp tail), `neg.ftz.f32`/`ex2.approx.ftz.f32`/`sub.ftz.f32`/`copysign.f32` x1 each; plain -O3 drops the `.ftz` modifiers |
| gelu/gelu_tanh mul/add/fma | `mul.ftz.f32` / `add.ftz.f32` / `fma.rn.ftz.f32` (production); unmodified forms under plain -O3 |
| gelu_tanh `tanh_fast` | `tanh.approx.f32` |
| output pack f32x2 -> f16x2/bf16x2 | `cvt.rn.f16x2.f32` / `cvt.rn.bf16x2.f32` (4 per vector iteration) |
| remainder output cast | `cvt.rn.f16.f32` / `cvt.rn.bf16.f32` (1 per remainder iteration) |
| `copy_r2g_v4` store | `st.global.v4.b32` (1 per vector iteration) |
| remainder loads/store | `ld.global.nc.b16` x2, `st.global.b16` x1 per remainder iteration |
| PDL schedule | `griddepcontrol.wait` x1, `griddepcontrol.launch_dependents` x1 (whole kernel) |
| address/loop arithmetic | `mul.lo.s64`, `mad.lo.s64`, `shl.b64`, `add.s64`, `or.b64`, `rem.s32`; `cvta.to.global.u64` entry-only; `setp`/`bra` loop control |

Static PTX opcode counts per exported instantiation (fp16 entries, one vector
iteration + one remainder iteration of straight-line code each). Rows tagged
[O3] are from the plain -O3 export; rows tagged [prod] from the production
`-use_fast_math` export; untagged rows are identical in both:

| Family | silu | gelu | gelu_tanh |
| --- | ---: | ---: | ---: |
| `ld.global.nc.v4.b32` | 2 | 2 | 2 |
| `ld.global.nc.b16` | 2 | 2 | 2 |
| `st.global.v4.b32` | 1 | 1 | 1 |
| `st.global.b16` | 1 | 1 | 1 |
| `cvt.f32.f16` | 16+ | 16+ | 16+ |
| `cvt.rn.f16x2.f32` | 4 | 4 | 4 |
| `mul.f32` [O3] | 18 | 45 | 54 |
| `mul.ftz.f32` [prod] | 18 | 45 | 54 |
| `fma.rn.f32` [O3] | 0 | 63 | 9 |
| `fma.rn.ftz.f32` [prod] | 0 | 63 | 9 |
| `add.f32` [O3] | 9 | 9 | 9 |
| `add.ftz.f32` [prod] | 9 | 9 | 9 |
| `div.rn.f32` [O3] | 9 | 0 | 0 |
| `div.approx.ftz.f32` [prod] | 9 | 0 | 0 |
| `ex2.approx.f32` [O3] | 9 | 0 | 0 |
| `ex2.approx.ftz.f32` [prod] | 9 | 9 | 0 |
| `tanh.approx.f32` | 0 | 0 | 9 |
| `neg.f32`/`neg.ftz.f32` [O3/prod] (gelu erf internals) | 0 | 9 | 0 |
| `sub.f32`/`sub.ftz.f32` [O3/prod] (gelu erf internals) | 0 | 9 | 0 |
| `abs.f32`/`abs.ftz.f32` [O3/prod] (gelu erf internals) | 0 | 18 | 0 |
| `selp.f32` (gelu erf internals) | 0 | 72 | 0 |
| `setp.ltu(.ftz).f32` / `setp.ge(.ftz).f32` (gelu erf internals) | 0 | 9 / 9 | 0 |
| `copysign.f32` (gelu erf internals) | 0 | 9 | 0 |
| `griddepcontrol.*` | 2 | 2 | 2 |

Per-element sequences are identical across the vector loop and the remainder
loop; the vector loop simply runs them 8-wide between 16-byte accesses. The
bf16 entries differ only in cvt opcode suffixes (`cvt.f32.bf16` x18,
`cvt.rn.bf16x2.f32` x4, `cvt.rn.bf16.f32` x1). In gelu's plain -O3 export the
libdevice-internal ex2 already carries `.ftz`; every other erf instruction
takes the modifiers of its build.
