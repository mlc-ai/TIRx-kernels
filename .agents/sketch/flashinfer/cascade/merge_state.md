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
include/flashinfer/attention/cascade.cuh (MergeStateKernel), the kernel
behind flashinfer.cascade.merge_state.
-->

# merge_state SM100: coarse WASP pipeline sketch

This non-executable design sketch describes the thread roles, control flow,
register storage, and PTX-level operations of
[`tirx_kernels/flashinfer/cascade/merge_state.py`](../../../../tirx_kernels/flashinfer/cascade/merge_state.py).
That TIRx module is the authoritative implementation.

The source is fixed at FlashInfer commit
`f2e04400e330fb2debe0bf8730d9424a1d37927f`:
`include/flashinfer/attention/cascade.cuh` L45-71 (`MergeStateKernel`) and
L572-587 (host `MergeState`, the launch formulae), with `vec_t` load/cast
helpers from `include/flashinfer/vec_dtypes.cuh` and `math::ptx_exp2` /
`math::ptx_log2` from `include/flashinfer/math.cuh`.

The four instantiations are `DTYPE in {f16, bf16}` crossed with
`VEC in {8, 16}`, each mirroring one `MergeStateKernel<vec_size, DTypeIn,
DTypeO>` template instantiation with `DTypeIn == DTypeO` (the only pairing the
`csrc/cascade.cu` binding produces). `VEC` is the source's
`max(16 / sizeof(DTypeIn), HEAD_DIM / 32)`: 8 for `head_dim` 64/128/256 and 16
for `head_dim` 512. `seq_len`, `num_heads`, and `head_dim` are static per
config; the source passes `num_heads`/`head_dim` as runtime `uint32_t` kernel
arguments, and folding them changes only address arithmetic. Out of scope:
`MergeStateInPlaceKernel` (in-place, optional mask), `MergeStatesKernel` and
the persistent variable-length kernels (many index sets), f32/fp8 dtypes
(rejected by `DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16`), `head_dim` outside
`DISPATCH_HEAD_DIM`, shared-memory staging, PDL (the source launches with no
attribute and emits no `griddepcontrol`), and tile (`Tx`) primitives.

## Pipeline at a glance

| Warps | Role-local program | Publication/reuse edges |
| --- | --- | --- |
| all (uniform) | Every thread `(tx, ty)` of CTA `pos` runs the same single-role straight-line program: two scalar lse loads, `max`/`sub`/`ex2`/`add`/two approximate divisions for the two blend weights, `VEC/8` 16-byte loads of `v_a` and of `v_b`, `VEC` widen-multiply-fma steps, `VEC/2` pack conversions, `VEC/8` 16-byte stores, then a pointer-null-guarded `lg2`/`add`/scalar store of the merged lse. | none — no SMEM, no mbarriers, no cross-thread data; the kernel has no synchronization at all |

There is one CTA per position (`blockIdx.x == pos`), one thread row per head
(`threadIdx.y == head_idx`), and `BDX = head_dim / VEC` threads per head
(`threadIdx.x == tx`), each owning one `VEC`-element slice. Every loop (the
`VEC`-wide element loop and the `vec_t` load/store and `vec_cast` helper loops)
is fully unrolled; the only branch is the `s_merged != nullptr` guard.

## Primitive vocabulary

Structural operations declare placement without moving data:

```python
specialize(...)       # compile-time variant selection
launch(...)           # compile-time launch topology
reg_tile(...)         # per-thread register tile
```

Copies state their direction and width:

```python
copy_g2r_b32(src_addr, dst)        # one scalar 32-bit global -> register load
copy_g2r_v4(src_addr, dst_b32x4)   # one 16-byte global -> register vector load
copy_r2g_v4(src_b32x4, dst_addr)   # one 16-byte register -> global vector store
copy_r2g_b32(src, dst_addr)        # one scalar 32-bit register -> global store
```

The compute vocabulary is deliberately primitive; every f32 arithmetic op is
annotated with its production `-use_fast_math` form (`.ftz` modifiers,
approximate division), the flags the FlashInfer JIT builds the shipped cubin
with (`flashinfer/jit/core.py`: `-std=c++17 -use_fast_math -DNDEBUG -O3`):

```python
max(dst, lhs, rhs)          # f32 maximum
sub(dst, lhs, rhs)
add(dst, lhs, rhs)
mul(dst, lhs, rhs)
fma(dst, lhs, rhs, acc)
exp2_fast(dst, src)         # math::ptx_exp2 -> ex2.approx.ftz.f32 (inline asm in the source)
log2_fast(dst, src)         # math::ptx_log2 -> lg2.approx.ftz.f32 (inline asm in the source)
div_fast(dst, lhs, rhs)     # fp32 `/` under -use_fast_math -> div.approx.ftz.f32
unpack2(lo_b16, hi_b16, word_b32)   # mov.b32 {lo, hi}, word
cast(dst_f32, src_b16)      # cvt.f32.f16 (DTYPE=f16) | cvt.f32.bf16 (DTYPE=bf16)
cast2x(dst_b32, lo, hi)     # cvt.rn.f16x2.f32 | cvt.rn.bf16x2.f32 (hi operand first)
isnull(pred, ptr)           # setp.eq.s64 pred, ptr, 0 (+ predicated branch)
```

`thread_id`, `cta_id` are schedule operations. Address expressions are shown
directly; they do not hide copies, computation, role changes, or
synchronization.

The audit evidence below comes from a fresh line-info export of the exact four
instantiations (explicit template instantiation of `MergeStateKernel` inside
`namespace flashinfer`, `nvcc -std=c++17 -use_fast_math -DNDEBUG -O3
-DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 ... -arch=sm_103a -ptx
-lineinfo`, CUDA 13.1 V13.1.80), preserved at
`.porting/merge_state/ptx/merge_state_sm103a_prod.ptx` (plus the identical
`-arch=sm_100a` export, the `sm_103a` SASS, and `ptxas -v`). The per-entry
slices are `entry_v8_half.ptx`, `entry_v8_bf16.ptx`, `entry_v16_half.ptx`,
`entry_v16_bf16.ptx` in the same directory.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

variant = specialize(DTYPE=("f16", "bf16"), VEC=(8, 16), target="sm_100a")
# instruction_selection: none; extent: four compile-time instantiations

VEC = max(8, HEAD_DIM // 32)      # source: max(16 / sizeof(DTypeIn), HEAD_DIM / 32)
BDX = HEAD_DIM // VEC             # 8 | 16 | 32 | 32 for head_dim 64 | 128 | 256 | 512
BDY = NUM_HEADS                   # one thread row per head
NWORDS = VEC // 2                 # packed b32 words per slice: 4 | 8

launch_config = launch(
    grid=(SEQ_LEN, 1, 1),
    block=(BDX, BDY, 1),          # BDX * BDY <= 1024
    dynamic_smem_bytes=0,
)
# instruction_selection: none; extent: static launch metadata

def merge_state(v_a,        # DTYPE [SEQ_LEN, NUM_HEADS, HEAD_DIM], direct global pointer
                s_a,        # f32   [SEQ_LEN, NUM_HEADS]
                v_b,        # DTYPE [SEQ_LEN, NUM_HEADS, HEAD_DIM]
                s_b,        # f32   [SEQ_LEN, NUM_HEADS]
                v_merged,   # DTYPE [SEQ_LEN, NUM_HEADS, HEAD_DIM]
                s_merged,   # f32   [SEQ_LEN, NUM_HEADS]; may be null in the source ABI
                ):
    pos = cta_id(axis="x", extent=SEQ_LEN)
    # instruction_selection: mov.u32 from %ctaid.x; extent: scalar per thread
    tx = thread_id(axis="x", extent=BDX)          # vector-slice index
    ty = thread_id(axis="y", extent=BDY)          # head_idx
    # instruction_selection: mov.u32 from %tid.x, mov.u32 from %tid.y; extent: scalar per thread
    # (the port owns one flat thread axis and derives tx = tid % BDX, ty = tid / BDX;
    #  same thread order tid = ty * BDX + tx, so slice and head ownership are unchanged)

    # =======================================================================
    # Blend weights from the two log2-sum-exp values (source L53-59)
    # =======================================================================

    row = pos * NUM_HEADS + ty                    # (pos, head) scalar index
    # instruction_selection: mad.lo.s32 + mul.wide.u32 + add.s64 address family; extent: per thread

    s_a_val = reg_tile("f32", [1])
    s_b_val = reg_tile("f32", [1])
    # instruction_selection: none; extent: two f32 registers per thread
    copy_g2r_b32(s_a + row, s_a_val)
    # instruction_selection: ld.global.nc.b32; extent: one scalar load
    copy_g2r_b32(s_b + row, s_b_val)
    # instruction_selection: ld.global.nc.b32; extent: one scalar load

    s_max = max(s_a_val, s_b_val)
    # instruction_selection: max.ftz.f32; extent: one scalar
    e_a = exp2_fast(sub(s_a_val, s_max))
    # instruction_selection: sub.ftz.f32 + ex2.approx.ftz.f32; extent: one scalar each
    e_b = exp2_fast(sub(s_b_val, s_max))
    # instruction_selection: sub.ftz.f32 + ex2.approx.ftz.f32; extent: one scalar each
    denom = add(e_a, e_b)
    # instruction_selection: add.ftz.f32; extent: one scalar (shared by both divisions and the lse)
    a_scale = div_fast(e_a, denom)
    # instruction_selection: div.approx.ftz.f32; extent: one scalar
    b_scale = div_fast(e_b, denom)
    # instruction_selection: div.approx.ftz.f32; extent: one scalar (ptxas realizes both as one MUFU.RCP + two FMUL.FTZ)

    # =======================================================================
    # Vector loads: vec_t<float, VEC>::cast_load of v_a (L61) then v_b (L62) (source L60-62)
    # =======================================================================

    slice = row * HEAD_DIM + tx * VEC             # first element of this thread's slice
    # instruction_selection: mul.lo.s32, cvt.u64.u32, mul.wide.u32, add.s64, shl.b64, add.s64 family; extent: per thread

    # cast_load = one 16-byte load per 8 elements (vec_t::load), then the pairwise
    # widen (vec_cast<float, DTYPE>); the source issues it for v_a, then for v_b.
    a_bits = reg_tile("b32", [NWORDS])
    a_vec = reg_tile("f32", [VEC])
    # instruction_selection: none; extent: NWORDS b32 + VEC f32 registers per thread
    for k in static_range(VEC // 8):
        copy_g2r_v4(v_a + slice + 8 * k, a_bits[4 * k : 4 * k + 4])
        # instruction_selection: ld.global.nc.v4.b32; extent: one 16-byte vector load (VEC//8 total; the second at [addr+16])
    for w in static_range(NWORDS):                # vec_cast<float, DTYPE>: pairwise widen
        lo, hi = unpack2(a_bits[w])
        # instruction_selection: mov.b32 {lo, hi}; extent: one per word (the source's __half22float2 asm re-issues it per half: 2 per word for f16, 1 per word for bf16)
        cast(a_vec[2 * w], lo); cast(a_vec[2 * w + 1], hi)
        # instruction_selection: cvt.f32.f16 (f16) | cvt.f32.bf16 (bf16); extent: two scalars per word, VEC total

    b_bits = reg_tile("b32", [NWORDS])
    b_vec = reg_tile("f32", [VEC])
    # instruction_selection: none; extent: NWORDS b32 + VEC f32 registers per thread
    for k in static_range(VEC // 8):
        copy_g2r_v4(v_b + slice + 8 * k, b_bits[4 * k : 4 * k + 4])
        # instruction_selection: ld.global.nc.v4.b32; extent: one 16-byte vector load (VEC//8 total; the second at [addr+16])
    for w in static_range(NWORDS):
        lo, hi = unpack2(b_bits[w])
        # instruction_selection: mov.b32 {lo, hi}; extent: one per word
        cast(b_vec[2 * w], lo); cast(b_vec[2 * w + 1], hi)
        # instruction_selection: cvt.f32.f16 (f16) | cvt.f32.bf16 (bf16); extent: two scalars per word, VEC total

    # =======================================================================
    # Blend: v_merged[i] = a_scale * v_a[i] + b_scale * v_b[i] (source L63-66)
    # =======================================================================

    o_vec = reg_tile("f32", [VEC])
    # instruction_selection: none; extent: VEC f32 registers per thread
    for i in static_range(VEC):                   # #pragma unroll, fully unrolled
        t = mul(b_scale, b_vec[i])
        # instruction_selection: mul.ftz.f32; extent: one scalar per element, VEC total
        fma(o_vec[i], a_scale, a_vec[i], t)
        # instruction_selection: fma.rn.ftz.f32; extent: one scalar per element, VEC total (nvcc contracts a*va + (b*vb) into mul + fma in this order)

    # =======================================================================
    # Vector store: vec_t<float, VEC>::cast_store into v_merged (source L67)
    # =======================================================================

    o_bits = reg_tile("b32", [NWORDS])
    # instruction_selection: none; extent: NWORDS packed output registers per thread
    for w in static_range(NWORDS):                # vec_cast<DTYPE, float>: pairwise narrow
        cast2x(o_bits[w], o_vec[2 * w], o_vec[2 * w + 1])
        # instruction_selection: cvt.rn.f16x2.f32 (f16) | cvt.rn.bf16x2.f32 (bf16), hi operand first; extent: one packed pair conversion, NWORDS total
    for k in static_range(VEC // 8):
        copy_r2g_v4(o_bits[4 * k : 4 * k + 4], v_merged + slice + 8 * k)
        # instruction_selection: st.global.v4.b32; extent: one 16-byte vector store (VEC//8 total; the second at [addr+16])

    # =======================================================================
    # Merged log2-sum-exp, guarded by the source's null-pointer test (source L68-70)
    # =======================================================================

    if not isnull(s_merged):
        # instruction_selection: setp.eq.s64 %p, s_merged, 0 + @%p bra (branch over the tail); extent: one predicate, kernel-uniform
        lse = add(s_max, log2_fast(denom))
        # instruction_selection: lg2.approx.ftz.f32 + add.ftz.f32; extent: one scalar each (log2(e_a + e_b) + s_max; the export commutes the add operands: add.ftz.f32 %r, s_max, lg2)
        copy_r2g_b32(lse, s_merged + row)
        # instruction_selection: st.global.b32; extent: one scalar store (every tx of the head writes the same value)
```

## Host wrapper and validation

The Python module performs host-only work; none of it emits device PTX:

```python
def prepare_data(dtype, seq_len, num_heads, head_dim):
    host_assert(dtype in ("float16", "bfloat16"))            # DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16
    host_assert(head_dim in (64, 128, 256, 512))            # DISPATCH_HEAD_DIM
    host_assert(head_dim // VEC * num_heads <= 1024)        # block limit (unchecked by the source)
    v_a, v_b = seeded_randn((seq_len, num_heads, head_dim), dtype)
    s_a, s_b = seeded_randn((seq_len, num_heads), f32)
    # instruction_selection: none; extent: four tensor constructions

launch_args = (v_a, s_a, v_b, s_b, v_merged, s_merged)     # flat six-pointer launch ABI
# instruction_selection: none; extent: matches the source kernel's six pointer params;
# the source's trailing (num_heads, head_dim) scalars are static in the port

# run_test compares against flashinfer.merge_state(v_a, s_a, v_b, s_b) with the
# source test tolerance rtol=1e-3, atol=1e-3 on both v_merged and s_merged.
# run_bench times the primfunc launch (tirx) against the source module op
# get_cascade_module().merge_state with preallocated outputs; both closures are
# no-argument.
```

## Static specialization boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| `DTYPE` | static per config | selects f16/bf16 cvt opcode families; same layout |
| `head_dim` | static per config | fixes `VEC` (8 or 16), `BDX`, the number of 16-byte loads/stores per operand (1 or 2), and the unrolled element count |
| `num_heads` | static per config | `BDY`; folds into the `(pos, head)` and slice address arithmetic (a runtime `uint32_t` argument in the source) |
| `seq_len` | static per config | grid extent |
| `s_merged != nullptr` | runtime pointer predicate | kept as in the source (`setp.eq.s64` + branch); the Python path always passes a non-null pointer, so the tail always executes |
| `-use_fast_math` | build flag | `.ftz` arithmetic, `div.approx.ftz.f32` division; `ex2`/`lg2` are already explicit `.approx.ftz` inline asm in the source |
| unroll hints | static | the `VEC`-wide element loop carries `#pragma unroll` and is fully unrolled; there is no other loop |

Automatic dispatch is outside the kernel: the source host wrapper reads
`seq_len`, `num_heads`, `head_dim` from tensor shapes at every call and
switches on `head_dim`; this module bakes the same values per config.

## TIRx module and benchmark contract

- `KERNEL_META = {"name": "merge_state", "category": "flashinfer",
  "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a"]}`.
- The executable kernel is expressed entirely in plain TIRx: register buffers,
  explicit global loads/stores, and native `K.ptx.*` forms for every non-trivial
  instruction (`ld.global.nc.b32`, `ld.global.nc.v4.b32`, `max.ftz.f32`,
  `sub.ftz.f32`, `ex2.approx.ftz.f32`, `add.ftz.f32`, `div.approx.ftz.f32`,
  `mov.b32` unpack, `cvt.f32.f16`/`cvt.f32.bf16`, `mul.ftz.f32`,
  `fma.rn.ftz.f32`, `cvt.rn.f16x2.f32`/`cvt.rn.bf16x2.f32`, `st.global.v4.b32`,
  `lg2.approx.ftz.f32`, `st.global.b32`). The null test is the codegen's
  pointer comparison (`K.isnullptr`), which lowers to `setp.eq.s64`. There is
  no `T.cuda.func_call`, no shared memory, and no `Tx` tile primitives.
- `get_kernel(dtype, seq_len, num_heads, head_dim)` returns the specialized
  primfunc; `prepare_data`, `run_test`, `run_bench` follow the repository
  contract.
- The timed implementation is named `tirx`; flashinfer is a lazy reference
  builder. Allocation, compilation, and correctness checks stay outside timing.
- Correctness reference is the source implementation itself
  (`flashinfer.merge_state`), tolerance `rtol=1e-3, atol=1e-3` as in the source
  test suite (`tests/attention/test_shared_prefix_kernels.py`).

## Instruction selection is a lowering consequence

The sketch above never requests a hardware instruction beyond the two
`math::` inline-asm wrappers the source itself spells (`ex2`, `lg2`). The
following lowering families follow from storage direction, shape, dtype, and
the production build flags. PTX names and static counts are taken from the
fresh line-info export cited above (`sm_103a`; the `sm_100a` export differs
only in its `.target` line and in re-loading the `s_merged` parameter inside
the guarded tail instead of reusing the entry register — register-allocation
noise with the same instruction families); they are audit evidence, not
operands.

| Primitive/schedule pattern | PTX family (fresh export) |
| --- | --- |
| `copy_g2r_b32` lse loads | `ld.global.nc.b32` (2) |
| `copy_g2r_v4` v_a/v_b loads | `ld.global.nc.v4.b32` (2 for VEC=8, 4 for VEC=16; the second of a pair addresses `[base+16]`) |
| `max` | `max.ftz.f32` (1) |
| `sub` before each `ex2` | `sub.ftz.f32` (2) |
| `exp2_fast` (`math::ptx_exp2`) | `ex2.approx.ftz.f32` (2), SASS `MUFU.EX2` x2 |
| `add` denominator, `add` lse | `add.ftz.f32` (2) |
| `div_fast` | `div.approx.ftz.f32` (2); SASS: one `MUFU.RCP` + `FMUL.FTZ` x2 |
| `unpack2` | `mov.b32 {lo, hi}` (f16: 2 per word = 2*VEC total because `__half22float2` re-unpacks per half; bf16: 1 per word = VEC total) |
| `cast` widen | `cvt.f32.f16` (VEC*2 total) for f16, SASS `HADD2.F32`; `cvt.f32.bf16` (VEC*2 total) for bf16, SASS `PRMT` + `SHF.L.U32` / `IMAD.U32 ..., 0x10000` (both shift-left-16 forms) |
| blend `mul`/`fma` | `mul.ftz.f32` (VEC) + `fma.rn.ftz.f32` (VEC) |
| `cast2x` narrow | `cvt.rn.f16x2.f32` / `cvt.rn.bf16x2.f32` (NWORDS), SASS `F2FP.{F16,BF16}.F32.PACK_AB` |
| `copy_r2g_v4` store | `st.global.v4.b32` (1 for VEC=8, 2 for VEC=16) |
| `isnull` guard | `setp.eq.s64` (1) + `@%p bra` (1); SASS `ISETP.NE.U32.AND` + `ISETP.NE.AND.EX` + `@!P0 EXIT` |
| `log2_fast` (`math::ptx_log2`) | `lg2.approx.ftz.f32` (1), SASS `MUFU.LG2` |
| `copy_r2g_b32` lse store | `st.global.b32` (1) |
| address/index arithmetic | `mad.lo.s32`, `mul.lo.s32`, `mul.wide.u32`, `cvt.u64.u32`, `shl.b64`, `add.s64`; `cvta.to.global.u64` entry-only |

Static PTX opcode counts per exported instantiation (straight-line kernel, no
loops; instruction lines, none predicated except the one `bra`):

| Family | f16 VEC=8 | bf16 VEC=8 | f16 VEC=16 | bf16 VEC=16 |
| --- | ---: | ---: | ---: | ---: |
| `ld.global.nc.b32` | 2 | 2 | 2 | 2 |
| `ld.global.nc.v4.b32` | 2 | 2 | 4 | 4 |
| `max.ftz.f32` | 1 | 1 | 1 | 1 |
| `sub.ftz.f32` | 2 | 2 | 2 | 2 |
| `ex2.approx.ftz.f32` | 2 | 2 | 2 | 2 |
| `add.ftz.f32` | 2 | 2 | 2 | 2 |
| `div.approx.ftz.f32` | 2 | 2 | 2 | 2 |
| `mov.b32` (unpack) | 16 | 8 | 32 | 16 |
| `cvt.f32.f16` / `cvt.f32.bf16` | 16 | 16 | 32 | 32 |
| `mul.ftz.f32` | 8 | 8 | 16 | 16 |
| `fma.rn.ftz.f32` | 8 | 8 | 16 | 16 |
| `cvt.rn.f16x2.f32` / `cvt.rn.bf16x2.f32` | 4 | 4 | 8 | 8 |
| `st.global.v4.b32` | 1 | 1 | 2 | 2 |
| `setp.eq.s64` / `bra` | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| `lg2.approx.ftz.f32` | 1 | 1 | 1 | 1 |
| `st.global.b32` | 1 | 1 | 1 | 1 |
| ptxas registers (`-v`, sm_103a) | 31 | 28 | 32 | 32 |

The bf16 entries differ from f16 only in the unpack/widen and pack opcode
suffixes (`cvt.f32.bf16`, `cvt.rn.bf16x2.f32`) and in issuing one `mov.b32`
per word instead of two; the SASS realizes the bf16 widen as `PRMT` plus
`SHF.L.U32` / `IMAD.U32` shift-left-16 forms where f16 uses `HADD2.F32`. The VEC=16 entries run the same
per-element sequence 16-wide between two 16-byte loads per operand and two
16-byte stores.
