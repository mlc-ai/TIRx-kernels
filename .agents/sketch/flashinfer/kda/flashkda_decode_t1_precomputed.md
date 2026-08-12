<!--
This file is a design sketch for a TIRx port of code from FlashInfer
(https://github.com/flashinfer-ai/flashinfer @ f2e04400),
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# FlashKDA cake T=1 precomputed decode: coarse execution sketch

**Non-executable.** This is the semantic execution skeleton, not runnable code.
The source of truth for the implementation is
`tirx_kernels/flashinfer/kda/flashkda_decode_t1_precomputed.py`.

Transcribed from FlashInfer's frozen generated export
`csrc/kda/flashkda_decode_d128_t1_precomputed_direct_split16.cu`, symbol
`kernel_flashinfer_recurrent_kda_t1_direct` (flashinfer-ai/flashinfer @
`f2e04400`). Line references below are into that file unless stated otherwise.

**Fixed specialization.** `HEAD_DIM = 128`, `TOKENS = 1`, `GATE_KIND = 0`
(precomputed log-gate), `LAUNCH_THREADS = 32`, `DIRECT_IMPL = 1`. The only free
knob is `VALUE_SPLIT ∈ {16, 8}`, chosen by the host from
`work = num_seqs * num_value_heads` against `2 * sm_count`
(`recurrent_kda.py:1169-1177`). The `_split16` and `_split8` bodies differ in
exactly three constants, so this sketch covers both.

**Out of scope.** `T ∈ {2,3,4,5,6}`; the `GATE_KIND = 1` lower-bound variant
(`d128_t3_lower_bound_split4`); the non-direct `d128_t1_precomputed_split{1,2,4,8}`
exports (256 threads, shared memory, `ldmatrix` + `mma`, WY transform), which
`run_recurrent_kda` never selects at T=1 (`recurrent_kda.py:1443-1447`).

## Pipeline at a glance

This kernel has **no asynchronous pipeline, no shared memory, no mbarrier, no
TMA, no `cp.async`, and no barrier of any kind**. One warp per CTA holds the
entire working set in registers and exchanges everything by shuffle.

The value split is a **partition, not a reduction**: CTA `(hv, value_tile)` owns
`TILE_ROW_STRIDE = 128 / VALUE_SPLIT` consecutive value rows of the `[V, K]`
state, and both the state rows it rewrites and the output elements it produces
are disjoint from every other CTA's. That is why there is no atomic, no
semaphore, no workspace and no second kernel.

| Role | Ownership | Publication/reuse edges |
| --- | --- | --- |
| CTA `(hv, value_tile, n)` | value head `hv`, sequence row `n`, value rows `[value_tile*TILE_ROW_STRIDE, +TILE_ROW_STRIDE)` | independent; no cross-CTA communication |
| warp (the whole CTA) | one `q`/`k`/`g` vector triple, one `beta`, the full 128-wide K axis | the two L2 norms and `k_dot_q` are warp-wide reductions |
| lane, phase 1 (`elem_start = lane*4`) | 4 consecutive elements of q, k, g | consumed by the 32-lane L2 butterflies, then redistributed |
| lane, phase 2 (`k_lane = lane & 15`) | 8 consecutive K elements `k_lane*8 .. +7` of q̂, k̂, γ | 16 lanes tile the 128-wide K axis |
| `v_lane = lane / 16` | one of two **interleaved** value rows in flight | `row_in_tile = v_lane + 2*(row_block*4 + row_local)` |
| `k_lane == 0` lane pair | the scalar `v[row]` load and the scalar `out[row]` store | `v` is broadcast to the other 15 lanes by one `shfl.idx` |

The lane split changes once, between the vector loads and the row loop
(`:153-164`). That redistribution is a pure shuffle stage, not a barrier: it
exists because the 128-wide vectors are loaded 4-per-lane across 32 lanes but
consumed 8-per-lane across 16 lanes.

Dependency chain per CTA: load q/k/g -> L2 norms -> redistribute + gate exp ->
`k_dot_q` -> [per row: load state row -> decay -> `pred`/`base` -> `delta` ->
rank-1 update -> store state, store out].

## Primitive vocabulary

Structural operations do not move or compute data:

```python
reg_tile(dtype, shape)              # per-thread register array
view(tensor[...])                   # logical slice, no instruction
```

Data movement:

```python
copy_g2r(gmem_slice, reg, hint=None)   # global -> register
copy_r2g(reg, gmem_slice, pred=None)   # register -> global
bcast(value, src_lane)                 # one lane's value to the whole warp
bfly(value, lane_xor)                  # butterfly exchange
```

Compute:

```python
fill(reg, c);  cast(dst, src);  add(a,b);  sub(a,b);  mul(a,b);  fma(a,b,c)
exp(x);  rsqrt(x);  dot(a, b)          # `dot` is one explicit FMA chain, see below
```

Control/scalar state only: `row_block`, `row_local`, `active`, `has_token`.

**`dot` is not a compound op here.** Every `dot` below expands to one explicit
chain of `fma.rn.ftz.f32` in a *fixed association order* copied from the source;
the order is written out at each site because it is load-bearing for bit-level
agreement.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================
# All from the JIT macro block (flash_kda_decode.py:231-257) plus the host
# LaunchContext (flashkda_decode_binding_common.cuh:50-59, :336-345).
HEAD_DIM     = 128
THREADS      = 32                       # __launch_bounds__(32); one warp
VALUE_SPLIT                             # 16 or 8; the only free knob
TILE_ROW_STRIDE = 128 // VALUE_SPLIT    # 8 or 16 value rows per CTA  (:79)
ROW_BLOCKS      = TILE_ROW_STRIDE // 8  # 1 or 2                      (:182)
K_LANES, V_LANES = 16, 2
H, HV, HEAD_RATIO                       # HV % H == 0, HV >= H
GATE_TOKEN_STRIDE                       # g.stride(1); the only non-compact input
STATE_SLOT_STRIDE                       # state.stride(0)

# Runtime: scale (f32). `lower_bound`, `A_log`, `dt_bias` exist in the ABI but
# GATE_KIND == 0 ignores them; `num_accepted_tokens` is in the kernel signature
# and never dereferenced at T=1 (:52).

grid  = (HV * VALUE_SPLIT, num_seqs)    # binding_direct_impl.cuh:57
block = (32,)                           # :58 -- dynamic smem is 0

# ===========================================================================
# Work decomposition and lane roles  (:63-79)
# ===========================================================================
work       = cta_id.x
value_tile = work & (VALUE_SPLIT - 1)
hv         = work // VALUE_SPLIT
n          = cta_id.y
query_head = hv // HEAD_RATIO           # the GQA fold
lane       = thread_id.x
k_lane     = lane & 15                  # 16 lanes x 8 elements = 128 K
v_lane     = lane // 16                 # selects one of two interleaved rows

# ===========================================================================
# Row activity, resolved before any load  (:72-79)
# ===========================================================================
raw_token_pos = cu_seqlens[n]
seq_len       = cu_seqlens[n + 1] - raw_token_pos
# instruction_selection: ld.global.nc.b32; extent: scalar (x2)
has_token  = raw_token_pos >= 0 and raw_token_pos < num_seqs and seq_len > 0
token_pos  = raw_token_pos if has_token else 0
raw_slot   = ssm_state_indices[n]
# instruction_selection: ld.global.nc.b32; extent: scalar
initial_slot = 0 if raw_slot < 0 else raw_slot     # inactive rows clamp to slot 0
active       = raw_slot >= 0 and has_token
tile_row_base = value_tile * TILE_ROW_STRIDE

# `has_token`'s `raw_token_pos < gridDim.y` term is why the body only accepts an
# identity cu_seqlens; the Python layer forbids a user-supplied T=1 cu_seqlens
# (recurrent_kda.py:1722-1724).

# ===========================================================================
# Phase 1: q, k, g vector loads, 4 elements per lane  (:88-132)
# ===========================================================================
q_src = reg_tile(f32, [4]); k_src = reg_tile(f32, [4]); gate_src = reg_tile(f32, [4])
elem_start = lane * 4

copy_g2r(q[token_pos, query_head, elem_start : elem_start+4], q_src)
# instruction_selection: ld.global.nc.v2.b32; extent: one 8-byte vector (4 bf16)
copy_g2r(k[token_pos, query_head, elem_start : elem_start+4], k_src)
# instruction_selection: ld.global.nc.v2.b32; extent: one 8-byte vector
copy_g2r(g[token_pos * GATE_TOKEN_STRIDE + hv*128 + elem_start : +4], gate_src)
# instruction_selection: ld.global.nc.v2.b32; extent: one 8-byte vector

# The bf16 -> f32 widening is NOT cvt. Each loaded 32-bit word carries two bf16
# and is split by an integer pair, which is what the source spells inline
# (:96-102) and what survives to PTX:
#   lo = word << 16 ; hi = word & 0xffff0000
# instruction_selection: shl.b32 + and.b32; extent: 2 words per tensor, 3 tensors

# ===========================================================================
# Phase 2: QK L2 norms -- full 32-lane butterflies  (:133-152)
# ===========================================================================
# The association order is fixed by the source (:136) and is copied verbatim:
#   sum = a0*a0 + a1*a1 + (a2*a2 + a3*a3)
q_sum_sq = dot(q_src, q_src)
k_sum_sq = dot(k_src, k_src)
# instruction_selection: mul.ftz.f32 + fma.rn.ftz.f32 + add.ftz.f32; extent: 4-term chain each

for offset in (16, 8, 4, 2, 1):        # full-warp, unlike the 16-lane dot reduce
    q_sum_sq = add(q_sum_sq, bfly(q_sum_sq, offset))
    k_sum_sq = add(k_sum_sq, bfly(k_sum_sq, offset))
# instruction_selection: shfl.sync.bfly.b32 + add.ftz.f32; extent: 5 rounds x 2 values

q_scale = mul(rsqrt(add(q_sum_sq, 1e-6)), scale)   # `scale` multiplies q ONLY
k_scale = rsqrt(add(k_sum_sq, 1e-6))
# instruction_selection: rsqrt.approx.ftz.f32; extent: scalar (x2)
# eps is hardcoded 1e-6f in the source (:148,:150), not a runtime argument.

# ===========================================================================
# Phase 3: lane redistribution 4-per-lane -> 8-per-lane, and the gate  (:153-164)
# ===========================================================================
q_reg = reg_tile(f32, [8]); k_reg = reg_tile(f32, [8]); gate_reg = reg_tile(f32, [8])
for i in range(8):                      # constexpr unroll
    src_lane = 2 * k_lane + i // 4      # the 4-per-lane -> 8-per-lane remap
    q_reg[i]    = mul(bcast(q_src[i & 3], src_lane), q_scale)
    k_reg[i]    = mul(bcast(k_src[i & 3], src_lane), k_scale)
    gate_reg[i] = exp(bcast(gate_src[i & 3], src_lane))
    # instruction_selection: shfl.sync.idx.b32 x3 + mul.ftz.f32 x2 + ex2.approx.ftz.f32
    # extent: 8 iterations; the exp is __expf, i.e. a mul by log2(e) folded into
    # the ex2 operand -- GATE_KIND == 0 means `g` arrives as a LOG-gate.

# ===========================================================================
# Phase 4: k_dot_q -- 16-lane butterfly  (:165-176)
# ===========================================================================
# Source association (:167), copied verbatim:
#   (k0q0 + k1q1 + (k2q2 + k3q3)) + (k4q4 + k5q5 + (k6q6 + k7q7))
k_dot_q = dot(k_reg, q_reg)
# instruction_selection: mul.ftz.f32 + fma.rn.ftz.f32 + add.ftz.f32; extent: 8-term chain

for offset in (8, 4, 2, 1):            # 16-lane group only, NOT 32
    k_dot_q = add(k_dot_q, bfly(k_dot_q, offset))
# instruction_selection: shfl.sync.bfly.b32 + add.ftz.f32; extent: 4 rounds
# Both half-warps end up holding the value they need; that is why the width is
# 16 here and 32 for the L2 norms above.

beta_value = cast(f32, beta[token_pos, hv])     # already sigmoided by the caller
# instruction_selection: ld.global.nc.b16 + cvt.f32.bf16; extent: scalar

# ===========================================================================
# Phase 5: the row loop  (:181-268)
# ===========================================================================
state_rows = reg_tile(f32, [4, 8])     # 4 value rows x 8 K elements; 32 registers

for row_block in range(ROW_BLOCKS):    # `#pragma unroll 1` in the source (:182)
    # ---- issue all four state-row loads before any math (:184-208) ----------
    # Prefetching the whole tile first is the source's own structure; the decay
    # math below depends on every row, so interleaving would serialise the
    # misses.
    for row_local in range(4):         # constexpr unroll
        row = tile_row_base + v_lane + 2 * (row_block * 4 + row_local)
        copy_g2r(state[initial_slot, hv, row, k_lane*8 : +8],
                 state_rows[row_local], hint="L1::no_allocate")
        # instruction_selection: ld.global.L1::no_allocate.v4.b32; extent: one
        # 16-byte tile (8 bf16) per row, 4 rows
        # The eviction hint is the source's (:193): the recurrent state streams
        # through exactly once, so L1 is left for q/k/g/v.
        # Widened by the same integer pair as phase 1, not by cvt:
        # instruction_selection: shl.b32 + and.b32; extent: 4 words per row

    # ---- delta rule, one value row at a time (:210-267) ---------------------
    for row_local in range(4):         # constexpr unroll
        row = tile_row_base + v_lane + 2 * (row_block * 4 + row_local)

        # Decay and both projections share one pass over the row. The source
        # accumulates sequentially (:212-216), not as a tree:
        pred = 0.0; base = 0.0
        for i in range(8):
            h_decay = mul(state_rows[row_local][i], gate_reg[i])
            pred = fma(h_decay, k_reg[i], pred)
            base = fma(h_decay, q_reg[i], base)
        # instruction_selection: mul.ftz.f32 + fma.rn.ftz.f32 x2; extent: 8 iterations

        for offset in (8, 4, 2, 1):
            pred = add(pred, bfly(pred, offset))
        for offset in (8, 4, 2, 1):
            base = add(base, bfly(base, offset))
        # instruction_selection: shfl.sync.bfly.b32 + add.ftz.f32; extent: 4 rounds x 2

        # ---- v: one scalar load, then a broadcast (:242-245) ----------------
        v_value = 0.0
        if k_lane == 0:
            v_value = cast(f32, v[token_pos, hv, row])
            # instruction_selection: ld.global.nc.b16 + cvt.f32.bf16; extent: scalar
        v_value = bcast(v_value, v_lane * 16)
        # instruction_selection: shfl.sync.idx.b32; extent: scalar
        # One load per row instead of 16 redundant ones; the source lane index is
        # v_lane*16 so each half-warp picks up its own row's value.

        delta = mul(sub(v_value, pred), beta_value)
        # instruction_selection: sub.ftz.f32 + mul.ftz.f32; extent: scalar

        # ---- rank-1 update, in registers (:246-248) ------------------------
        for i in range(8):
            state_rows[row_local][i] = fma(delta, k_reg[i],
                                           mul(state_rows[row_local][i], gate_reg[i]))
        # instruction_selection: mul.ftz.f32 + fma.rn.ftz.f32; extent: 8 iterations
        # The decayed value is recomputed rather than reused from the pred/base
        # pass; the source does the same, and it is what keeps the register tile
        # at 32.

        # ---- publish (:249-266) --------------------------------------------
        if active:
            words = reg_tile(u32, [4])
            for p in range(4):
                words[p] = cast(bf16x2, (state_rows[row_local][2*p],
                                         state_rows[row_local][2*p+1]))
                # instruction_selection: cvt.rn.bf16x2.f32; extent: 4 pairs
            copy_r2g(words, state[initial_slot, hv, row, k_lane*8 : +8])
            # instruction_selection: st.global.v4.b32; extent: one 16-byte tile
            if k_lane == 0:
                copy_r2g(cast(bf16, add(base, mul(delta, k_dot_q))),
                         out[token_pos, hv, row])
                # instruction_selection: fma.rn.ftz.f32 + cvt.rn.bf16.f32 +
                # st.global.b16; extent: scalar
                # o = q.S_new is folded as base + delta*k_dot_q, so the updated
                # state is never re-read.
        elif has_token and k_lane == 0:
            copy_r2g(cast(bf16, 0.0), out[token_pos, hv, row])
            # instruction_selection: st.global.b16; extent: scalar
            # An in-row but inactive sequence zeroes its output and leaves the
            # state slot untouched.
```

## Inactive-row contract

Three states, and the sketch must keep them distinct because the correctness
gate tests all three:

| `raw_slot` | `has_token` | state | out |
| --- | --- | --- | --- |
| `>= 0` | true | rewritten at `raw_slot` | `base + delta*k_dot_q` |
| `< 0` | true | untouched (`initial_slot` clamps to 0 but `active` gates the store) | zeroed |
| any | false | untouched | not written at all |

## Static specialization and launch boundary

| value | where it is fixed | note |
| --- | --- | --- |
| `VALUE_SPLIT` | host, from `work` vs `2*sm_count` | the only knob; 16 and 8 are the only reachable values at T=1 |
| `TILE_ROW_STRIDE`, `ROW_BLOCKS` | derived from `VALUE_SPLIT` | the three-line diff between the two exports |
| `H`, `HV`, `HEAD_RATIO` | host, per call | constexpr in the port; runtime ints in the source |
| `GATE_TOKEN_STRIDE`, `STATE_SLOT_STRIDE` | host, from tensor strides | constexpr in the port |
| `scale` | runtime | stays a runtime argument, as in the source |
| eps `1e-6` | hardcoded in the body | not an argument |

## Instruction-selection summary

Read off the exported line-info PTX for both variants
(`.porting/flashkda_decode_t1_precomputed/ptx/split{16,8}/kernel.ptx`), which is
the evidence for every annotation above. Body totals, split16 / split8:

| form | 16 | 8 | where |
| --- | ---: | ---: | --- |
| `fma.rn.ftz.f32` | 108 | 108 | every dot chain and the rank-1 update |
| `mul.ftz.f32` | 69 | 69 | scales, decay, `delta`, the `__expf` log2(e) fold |
| `add.ftz.f32` | 53 | 53 | butterflies and the dot tails |
| `shfl.sync.bfly.b32` | 46 | 46 | 2x5 L2 rounds + 4 rounds x (k_dot_q, pred, base) |
| `shfl.sync.idx.b32` | 29 | 29 | the 3x8 redistribution + one v broadcast per row |
| `shl.b32` / `and.b32` | 31/25 | 34/25 | bf16 -> f32 widening, vectors and state rows |
| `cvt.rn.bf16x2.f32` | 16 | 16 | 4 pairs x 4 rows, the state store |
| `ex2.approx.ftz.f32` | 8 | 8 | the gate, one per K element per lane |
| `cvt.rn.bf16.f32` + `st.global.b16` | 8 + 8 | 8 + 8 | the scalar out stores, both branches |
| `cvt.f32.bf16` | 5 | 5 | `beta` (1) and `v` (4 rows) -- the scalar loads |
| `ld.global.nc.v2.b32` | 3 | 3 | q, k, g |
| `ld.global.L1::no_allocate.v4.b32` | 4 | 4 | the state rows |
| `st.global.v4.b32` | 4 | 4 | the state rows |
| `rsqrt.approx.ftz.f32` | 2 | 2 | the two L2 norms |
| total | 557 | 571 | |

Three consequences the port must honour:

1. **Everything float is `.ftz`.** `-use_fast_math` (`jit/core.py:545`) plus plain
   CUDA operators give `mul.ftz.f32` / `add.ftz.f32` / `sub.ftz.f32` /
   `fma.rn.ftz.f32`, and `__expf` / `rsqrtf` give `ex2.approx.ftz.f32` /
   `rsqrt.approx.ftz.f32`. This is the **opposite** of the CuTe-DSL KDA decode
   siblings, whose source emitted no `.ftz` and which therefore needed explicit
   non-FTZ helpers. Reusing those helpers here would be a divergence.
2. **`__restrict__` buys `.nc`.** q, k, g, v and beta all load through
   `ld.global.nc.*`; only the state, which is written, does not. The state's
   `L1::no_allocate` hint is separate and explicit in the source.
3. **The widening asymmetry is deliberate.** Vector loads widen with the
   `shl`/`and` integer pair; the two scalar loads (`beta`, `v`) use
   `cvt.f32.bf16`. Both forms appear in the PTX and both must be reproduced.

The two variants compile to the same instruction stream; the split8 body differs
only by the extra `row_block` trip (+3 `shl.b32`, +3 `add.s32`, +1 `bra`, +4
`mov.pred`), because `#pragma unroll 1` keeps a single copy of the row body.

## TIRx module and benchmark contract

- Module `tirx_kernels/flashinfer/kda/flashkda_decode_t1_precomputed.py`,
  `KERNEL_META["name"] = "flashkda_decode_t1_precomputed"`, cc 10.
- Reference: the frozen export itself, through the JIT module's direct ABI
  (`flashinfer.jit.flash_kda_decode.get_flash_kda_decode_module`), so the
  comparison is kernel-only and inactive rows are reachable.
- 11 bench rows cover both split sides at `HV = H = 16` and `12` plus GQA
  `HV = 32, H = 16`; `verify_dispatch.py` asserts each row's split against
  FlashInfer's own selector.
