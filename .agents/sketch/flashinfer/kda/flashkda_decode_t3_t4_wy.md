<!--
This file is a design sketch for a TIRx port of code from FlashInfer
(https://github.com/flashinfer-ai/flashinfer @ f2e04400),
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# FlashKDA cake T=4 precomputed and T=3 lower-bound decode: coarse WY execution sketch

**Non-executable.** This is the semantic execution skeleton, not runnable code.
The source of truth for the implementations is
`tirx_kernels/flashinfer/kda/flashkda_decode_t4_precomputed.py` and
`tirx_kernels/flashinfer/kda/flashkda_decode_t3_lower_bound.py`.

Transcribed from two frozen generated exports, both with symbol
`kernel_flashinfer_recurrent_kda_wy_vtile_short` (flashinfer-ai/flashinfer @
`f2e04400`):

- `csrc/kda/flashkda_decode_d128_t4_precomputed_split2.cu` (701 lines) — `T4:NNN`
- `csrc/kda/flashkda_decode_d128_t3_lower_bound_split4.cu` (711 lines) — `T3:NNN`

**One sketch, two kernels.** Both are the same Loom template as the already-ported
T=2 body, and the structural distance is far smaller than the line counts suggest.
Stripping every integer literal and diffing phase by phase:

| phase | T=2 → T=4 | T=4 → T=3 |
| --- | --- | --- |
| A preprocess | **identical** | **+9 lines**: the `GATE_KIND == 1` chain |
| B state gather | identical | identical |
| C sVec / WY coefficients | identical¹ | identical¹ |
| D MMA chain | identical | identical |
| E quad broadcast + solve | identical | identical |
| F outputs | **rewritten**, 48 → 49 lines | **identical** |
| G publish `sU` | identical | identical |
| H recurrence + checkpoints | identical | identical |

¹ the only textual difference is the Loom-generated temporary's name hash
(`_bval_1100091392` vs `_bval_1101446352`), which carries no semantics.

So the whole port is: the ported T=2 skeleton re-parameterized by `T` and the
geometry, plus **exactly two** structural deltas — phase F's generic rewrite
(shared by both new bodies, byte-identical between them) and T=3's gate insert.
This sketch is written against that decomposition: the common skeleton is given
once with the knobs symbolic, and the two deltas get their own sections.

**Fixed specializations.**

| knob | T=4 | T=3 |
| --- | --- | --- |
| `TOKENS` | 4 | 3 |
| `GATE_KIND` | 0 (precomputed log-gate) | **1 (in-kernel lower-bound gate)** |
| `VALUE_SPLIT` | 2 | 4 |
| `LAUNCH_THREADS` | 128 (4 warps) | 96 (3 warps) |
| `ROWS_PER_CTA` | 64 (`T4:159`) | 32 (`T3:159`) |
| `SMEM_TOTAL` | 25856 | 15872 |
| grid | `(HV*2, N)` | `(HV*4, N)` |
| nat clamp | `[0, 3]` | `[0, 2]` |
| warp/group guards | **all statically true** | **all live** — see below |
| shape domain | any `H`, `HV` with `HV % H == 0` | `H == HV == 16`, `N ∈ {1,2,4,8,16}` |
| dispatch | `_select_flash_kda_decode_value_split` returns 2 at T=4 on sm100a, no shape dependence | no split selector at all: `recurrent_kda.py:1437-1438` early-returns the variant |

`HEAD_DIM = 128`, `DIRECT_IMPL` unset, `DIRECT_PREFIX_CHECKPOINT = 0`,
`BLOCK_CHECKPOINT_MMA = 0`, `NUM_MAIN_STAGES = 1` for both.

**Out of scope.** T ∈ {1,2,5,6}; the `t1_direct` one-warp schedule (a different
kernel symbol); `VALUE_SPLIT = 8`, reachable only on GB300 and not measurable
here; T=3 with precomputed gates, which upstream does not export at all and has
a dedicated rejection test.

## Pipeline at a glance

No warp specialization, no asynchronous pipeline: no `cp.async`, no mbarrier, no
TMA, no atomics, nothing double-buffered. The same warps play every role in
sequence, separated by two CTA barriers and one warp barrier.

The value split is a **partition**: CTA `(hv, value_tile)` owns value rows
`[ROWS_PER_CTA·value_tile, +ROWS_PER_CTA)` of the `[V,K]` state, and every CTA
redundantly recomputes the whole token preprocessing for its own `(n, hv)`. No
CTA combines anything with any other.

### T=4 — 4 warps, fully occupied

Every guard in the body (`warp_0 < 4`, `group < 8`) is statically true at 128
threads. This is the same "everyone does everything" shape as the T=2 port.

| Role | Ownership | Publication/reuse edges |
| --- | --- | --- |
| CTA `(hv, value_tile, n)` | value head `hv`, sequence `n`, 64 value rows | independent |
| warp `w`, phases A and C | **token `w`** (4 warps ⇄ 4 tokens) | publishes `sK`/`sD`/`sBeta`/`sSlot`/`sToken`, and (warp 0) `sInit`, across barrier #1; publishes `sVec`/`sL`/`sR` across barrier #2 |
| thread `tid`, phases B and H | 8 value rows (`group = tid/16`, rows `group*8 + r`) × 8 keys | gathers state into registers **and** stages it to `sState`; later writes all four tokens' checkpoints |
| warp `w`, phase D | value rows `w*16 .. +15` of the 64-row tile | consumes `sState` (all 64 rows, staged by all 4 warps) and `sVec` |
| lane with `lane_quad == 2` | the WY solve for rows `w*16 + frag_row` and `+8`, and the `sU` publish | the only lanes that load `v` |
| lane with `lane_quad >= 2` | **two output tokens each**: lane `4f+2` writes tokens 0,1 and lane `4f+3` writes tokens 2,3 | the only lanes that store `out` |

The last row is the new shape. At T=2 the solver lane and the only output lane
were the same lane; at T=4 the output work is split across **two** lanes of each
quad while the solve still happens on one, so the residuals must cross the quad.

### T=3 — 3 warps, and warp 2 is a token-only warp

The warp count is `max(tokens, value_rows/16, (value_rows/8+1)/2) = max(3,2,2)`
(`flash_kda_decode.py:115-137`): the **token** term wins. So the third warp
exists only to preprocess the third token, and every guard is live:

| guard | site | effect at 96 threads |
| --- | --- | --- |
| `warp_0 < 3` | `T3:180` (A), `T3:339` (C) | all three warps |
| `group < 4` | `T3:303` (B), `T3:669` (H) | **excludes groups 4,5 — warp 2** |
| `warp_0 < 2` | `T3:415` (D), `:493` (F), `:542` (G) | **excludes warp 2** |

`warp_0 < 2` is `value_rows/16`; `group < 4` is `value_rows/8`. Both select the
same two warps, and neither depends on `T`.

| Role | Ownership | Publication/reuse edges |
| --- | --- | --- |
| CTA `(hv, value_tile, n)` | value head `hv`, sequence `n`, 32 value rows | independent |
| warp 0,1, phases A and C | tokens 0,1 | as T=4 |
| **warp 2** | **token 2 only**: its q/k/g, its L2 norms, its gate, its `sVec` columns 2 and 6, its `sL`/`sR` row | publishes across **both** CTA barriers, then does nothing — it never gathers state, never issues an MMA, never stores |
| thread `tid` with `group < 4`, phases B and H | 8 value rows × 8 keys | as T=4, but only warps 0,1 participate |
| warp 0,1, phase D | value rows `w*16 .. +15` of the 32-row tile | consumes all 32 staged rows |
| lane with `lane_quad == 2` (warps 0,1) | the WY solve, depth 3 | loads `v` |
| lane with `lane_quad >= 2` (warps 0,1) | lane `4f+2` writes tokens 0,1; lane `4f+3` writes token 2 **and computes a discarded token 3** | see the hazard section |

Both `__syncthreads()` are therefore **real 3-warp cross-warp edges** here, where
in the T=2 and T=4 bodies every warp reaches every phase. `sU` stays intra-warp,
so the warp barrier in phase G is still sufficient.

Warp 2 also allocates the full `hist[64]` register array and never fills it —
that is the source's behaviour and the port reproduces it rather than trying to
shrink the allocation.

Dependency chain (both bodies): preprocess → **barrier** → state gather+stage →
sVec/L/R → **barrier** → MMA → WY solve → outputs → sU → **warp barrier** →
recurrence and checkpoints.

## Primitive vocabulary

Structural operations do not move or compute data:

```python
reg_tile(dtype, shape)                 # per-thread register array
smem_arena(bytes, align)               # the single shared allocation
smem_view(name, offset, dtype, shape)  # a named region of that arena
swz(byte_off)                          # byte_off ^ ((byte_off >> 7 & 7) << 4)
widen(word)                            # one u32 -> two f32, as shl.b32 / and.b32
```

Data movement:

```python
copy_g2r(gmem_slice, reg)              # global -> register
copy_r2g(reg, gmem_slice, pred=None)   # register -> global
copy_r2s(reg, smem_slice)              # register -> shared
copy_s2r(smem_slice, reg)              # shared -> register
ldm(smem_addr, frag, trans=False)      # ldmatrix, one x4 group
bcast(v, src_lane, 31, 0xFFFFFFFF)     # shfl.idx
bfly(v, lane_xor, 31, 0xFFFFFFFF)      # shfl.bfly
```

Compute:

```python
fill(reg, c); cast(dst, src); add(a,b); sub(a,b); mul(a,b); fma(a,b,c)
exp(x); rsqrt(x); neg(x); div(a,b)     # neg and div are T=3-only
dot(a, b)                              # `dot` = one explicit FMA chain
mma(acc, a_frag, b_frag, init=False)   # one mma.sync.m16n8k16 issue
```

Sync: `cta_sync()`, `warp_sync()`.

**Every shuffle is width-32 with a full member mask** — the operands are always
`(31, 0xFFFFFFFF)`. A reduction that wants a narrower group restricts its *xor
offset set*, never the instruction width. **`dot` is not a compound op**: each one
expands to an explicit FMA chain in a fixed association order, written out at its
site because the order is load-bearing.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================
HEAD_DIM = 128
TOKENS, VALUE_SPLIT, THREADS = (4, 2, 128)   # T=4
TOKENS, VALUE_SPLIT, THREADS = (3, 4,  96)   # T=3
ROWS_PER_CTA = 128 // VALUE_SPLIT             # 64 / 32        (T4:159, T3:159)
H, HV, HEAD_RATIO                             # T=3: pinned at 16, 16, 1
GATE_TOKEN_STRIDE, STATE_SLOT_STRIDE          # host-derived, constexpr here

grid  = (HV * VALUE_SPLIT, num_seqs)          # binding_impl.cuh:64
block = (THREADS,)

# Runtime scalars: `scale` in both; T=3 additionally takes `lower_bound` and the
# A_log / dt_bias buffers, which the precomputed bodies never dereference.

# The source uses dynamic smem (extern __shared__ __align__(1024), the launcher
# does cudaFuncSetAttribute + <<<..., SMEM_TOTAL, ...>>>). Both totals fit the
# static limit, so the port declares one static arena at the same alignment and
# carves the same byte offsets -- what the swizzled ldmatrix addressing depends
# on is the offsets and the alignment, both preserved.
#
#            T=4 (25856 B)                    T=3 (15872 B)
arena     = smem_arena(SMEM_TOTAL, align=1024)
sState0   = smem_view(arena,     0, bf16, [ROWS_PER_CTA, 64])   #     0 /     0
sState1   = smem_view(arena,  ..., bf16, [ROWS_PER_CTA, 64])    #  8192 /  4096
sVec      = smem_view(arena,  ..., bf16, [128, 16])             # 16384 /  8192
sK        = smem_view(arena,  ..., f32,  [TOKENS, 128])         # 20480 / 12288
sD        = smem_view(arena,  ..., f32,  [TOKENS, 128])         # 22528 / 13824
sBeta     = smem_view(arena,  ..., f32,  [TOKENS])              # 24576 / 15360
sSlot     = smem_view(arena,  ..., i32,  [TOKENS])              # 24592 / 15372
sToken    = smem_view(arena,  ..., i32,  [TOKENS])              # 24608 / 15384
sInit     = smem_view(arena,  ..., i32,  [1])                   # 24624 / 15396
sL        = smem_view(arena,  ..., f32,  [TOKENS, TOKENS])      # 24640 / 15412
sR        = smem_view(arena,  ..., f32,  [TOKENS, TOKENS])      # 24704 / 15448
sU        = smem_view(arena,  ..., f32,  [TOKENS, ROWS_PER_CTA])# 24768 / 15484
# sGramA0/1 alias sVec and are t5/t6 machinery: dead in both.
# sVec keeps 16 columns at every T -- the region is sized for the largest T, so
# only columns 0..TOKENS-1 (k side) and 4..4+TOKENS-1 (q side) are ever written.

# ===========================================================================
# Work decomposition and lane roles  (T4/T3:100-179 -- identical)
# ===========================================================================
work       = cta_id.x
value_tile = work % VALUE_SPLIT
hv         = work // VALUE_SPLIT
n          = cta_id.y
query_head = hv // HEAD_RATIO
tid        = thread_id.x
warp       = tid // 32          # == token index in phases A and C
lane       = tid % 32
lane_quad  = lane & 3           # the MMA accumulator's column pair
frag_row   = lane // 4          # the MMA accumulator's row
quad_base  = lane - lane_quad
group      = tid // 16          # phases B and H: 8 groups (T=4) / 6, of which 4 run (T=3)
lane_group = tid % 16
k_start    = lane_group * 8     # phases B and H: 8 keys per thread
elem_start = lane * 4           # phases A and C: 4 keys per lane
tile_row_base  = value_tile * ROWS_PER_CTA
owned_row_base = group * 8
token_base = cu_seqlens[n];  seq_len = cu_seqlens[n+1] - token_base
# instruction_selection: ld.global.nc.b32; extent: scalar (x2)  (.loc :147,:148)
# cu_seqlens must step by exactly TOKENS per row. The body does not bound-check
# token_base -- deliberately unvalidated for CUDA-graph capture. Padding is
# signalled ONLY by ssm_state_indices < 0.
#
# `warp` is make_warp_uniform(tid/32) at :101, one shfl.sync.idx.b32 at .loc :39
# in both bodies. Semantically the identity -- a uniformity hint -- but its
# result IS `warp` and is live throughout, so the port spells it `tid // 32`.

# ===========================================================================
# Phase A: token preprocess, warp <-> token  (T4:180-290, T3:180-299)
# ===========================================================================
if warp < TOKENS:                       # T=4: always true. T=3: LIVE, warp 2 is
    token         = warp                # the token-only warp.
    active_token  = token < seq_len
    token_pos     = token_base + token if active_token else 0
    qk_base   = (token_pos * H + query_head) * 128 + elem_start
    gate_base = token_pos * GATE_TOKEN_STRIDE + hv * 128 + elem_start

    r_q = reg_tile(f32, [4]); r_k = reg_tile(f32, [4]); r_d = reg_tile(f32, [4])
    copy_g2r(q[qk_base : +4],   r_q)
    copy_g2r(k[qk_base : +4],   r_k)
    copy_g2r(g[gate_base : +4], r_d)
    # instruction_selection: ld.global.nc.v2.b32; extent: one 8-byte vector
    # (4 bf16) each, 3 total in both bodies
    # Each 32-bit word holds two bf16 and is split by an integer pair, not by
    # cvt: lo = word << 16, hi = word & 0xffff0000
    # instruction_selection: shl.b32 + and.b32; extent: 2 words per tensor, 3 tensors

    # ---- L2 norms: two independent full-warp butterflies -------------------
    q_sq = dot(r_q, r_q);  k_sq = dot(r_k, r_k)
    # instruction_selection: fma.rn.ftz.f32; extent: 4-term chain each, the first
    # with C = 0f00000000 -- there is no mul at these sites
    for offset in (16, 8, 4, 2, 1):
        q_sq = add(q_sq, bfly(q_sq, offset, 31, 0xFFFFFFFF))
    for offset in (16, 8, 4, 2, 1):
        k_sq = add(k_sq, bfly(k_sq, offset, 31, 0xFFFFFFFF))
    # instruction_selection: shfl.sync.bfly.b32 + add.ftz.f32; extent: 5 rounds x 2
    # = 10 shfl in phase A, both bodies. The two loops are sequential, not
    # interleaved.
    q_norm = mul(rsqrt(add(q_sq, 1e-6)), scale)   # scale folds into q only
    k_norm = rsqrt(add(k_sq, 1e-6))
    # instruction_selection: rsqrt.approx.ftz.f32; extent: scalar (x2)
    # eps is hardcoded inside the rsqrt argument.

    # ---- the gate: the ONE place the two bodies diverge --------------------
    # T=4 (GATE_KIND == 0): g already holds log(gamma).
    for i in range(4):
        r_q[i] = mul(r_q[i], q_norm)
        r_k[i] = mul(r_k[i], k_norm)
        r_d[i] = exp(r_d[i])
        # instruction_selection: mul.ftz.f32 x3 + ex2.approx.ftz.f32
        # extent: 4 iterations -> 4 ex2 total in the T=4 body

    # T=3 (GATE_KIND == 1): g is the RAW pre-gate; the gate is derived here.
    # Hoisted above the element loop, once per lane, every lane (T3:259-263):
    gate_a     = exp(A_log[query_head])
    # instruction_selection: ld.global.nc.b32 (scalar; NOT broadcast from lane 0)
    # + mul.ftz.f32 (x log2e) + ex2.approx.ftz.f32; extent: scalar
    neg_gate_a = neg(gate_a)
    # instruction_selection: neg.ftz.f32; extent: scalar -- exactly ONE in the
    # whole body. The source writes `(-gate_a)` inside the element loop
    # (T3:274); it is loop-invariant and is hoisted, so the port hoists it too.
    for i in range(4):
        r_q[i] = mul(r_q[i], q_norm)
        r_k[i] = mul(r_k[i], k_norm)
        biased = add(r_d[i], dt_bias[query_head * 128 + elem_start + i])
        # instruction_selection: ld.global.nc.b32 + add.ftz.f32; extent: scalar.
        # FOUR SEPARATE SCALAR LOADS at [%rd30], +4, +8, +12 -- the four fp32 are
        # contiguous but nvcc does NOT vectorize them. No ld.global.nc.v4.b32
        # appears anywhere in this body.
        sig    = exp(mul(biased, neg_gate_a))
        # instruction_selection: mul.ftz.f32 (biased, -gate_a; that operand
        # order) + mul.ftz.f32 (x log2e) + ex2.approx.ftz.f32; extent: scalar
        r_d[i] = exp(div(lower_bound, add(sig, 1.0)))
        # instruction_selection: add.ftz.f32 (sig first, 0f3F800000 second)
        # + div.approx.ftz.f32 (numerator = the loop-invariant lower_bound
        # register) + mul.ftz.f32 (x log2e) + ex2.approx.ftz.f32; extent: scalar
        #
        # The division lowers to div.approx.ftz.f32, NOT rcp.approx + mul: 4
        # div.approx.ftz.f32 in the body, one per element, and zero rcp.
        # ex2 total in the T=3 body is 9 = 4 elements x 2 + the hoisted gate_a,
        # against 4 in the T=4 body. That 9-vs-4 count is the arithmetic proof
        # that gate_a is hoisted and the per-element chain has two exp sites.

    copy_r2s(r_k, sK[token, elem_start : +4])
    copy_r2s(r_d, sD[token, elem_start : +4])
    # instruction_selection: st.shared.v4.b32; extent: one 16-byte tile each

    if lane == 0:
        sSlot[token]  = ssm_state_indices[n*TOKENS + token] if active_token else -1
        sToken[token] = token_pos
        sBeta[token]  = cast(f32, beta[token_pos * HV + hv])
        # instruction_selection: ld.global.nc.b32 + ld.global.nc.b16
        # + cvt.f32.bf16 + st.shared.b32 x3; extent: scalar each
        if token == 0:
            accepted     = clamp(num_accepted_tokens[n] - 1, 0, TOKENS - 1)
            initial_slot = ssm_state_indices[n*TOKENS + accepted]
            sInit[0]     = 0 if initial_slot < 0 else initial_slot
            # instruction_selection: ld.global.nc.b32 x2 + st.shared.b32; extent: scalar
            # The clamp ceiling is TOKENS-1: 3 at T=4, 2 at T=3. Both edges are
            # reachable and both are covered by CONFIGS.

cta_sync()
# instruction_selection: bar.sync 0; extent: CTA
# Orders sInit -> the gather base below, and sK/sD/sBeta/sSlot/sToken -> their
# cross-warp readers in phases C, E, F and H. At T=3 this is a genuine 3-warp
# edge: warp 2 publishes token 2's sK/sD/sBeta/sSlot/sToken here and warps 0,1
# consume them in C and H.

# ===========================================================================
# Phase B: state gather  (T4:292-329 all 128 threads, T3:301-338 groups 0..3)
# ===========================================================================
if group < ROWS_PER_CTA // 8:           # T=4: `group < 8`, statically true.
                                        # T=3: `group < 4`, LIVE -- warp 2 sits out.
    initial_head_base = sInit[0] * STATE_SLOT_STRIDE + hv * 128 * 128
    hist = reg_tile(f32, [8, 8])
    for row_local in range(8):          # constexpr unroll
        row  = owned_row_base + row_local
        pack = reg_tile(u32, [4])
        copy_g2r(state[initial_head_base + (tile_row_base + row)*128 + k_start : +8], pack)
        # instruction_selection: ld.global.v4.b32; extent: one 16-byte tile
        # (8 bf16), 8 rows. NOT .nc: `state` is written later in this kernel.
        for p in range(4):
            hist[row_local][2*p], hist[row_local][2*p+1] = widen(pack[p])
        # instruction_selection: shl.b32 + and.b32; extent: 4 words per row
        half = sState0 if lane_group < 8 else sState1
        copy_r2s(pack, half[row, (k_start % 64) : +8] @ swz)
        # instruction_selection: st.shared.v4.b32; extent: one 16-byte tile, 8 rows
        # The bf16 bits go to shared unmodified; the swizzle is on the byte
        # offset: swz(row*128 + (k_start % 64)*2).
# Identical in all three bodies: 8 ld.global.v4.b32 + 16 st.shared.v4.b32
# STATIC regardless of T (both arms of the `lane_group < 8` select are emitted;
# a thread executes 8 of the stores). What changes is how many threads run it (128 vs 64)
# and therefore how many of the ROWS_PER_CTA rows get staged.

# ===========================================================================
# Phase C: sVec columns and the WY coefficients  (T4:330-404, T3:339-413)
# ===========================================================================
if warp < TOKENS:                       # T=3: warp 2 IS here -- it owns token 2's
                                        # sVec columns 2 and 6 and sL/sR row 2.
    for i in range(4):                  # constexpr unroll
        k_idx  = elem_start + i
        prefix = 1.0
        for j in range(TOKENS):         # prefix gate: product over j <= token
            if token >= j:
                prefix = mul(prefix, sD[j, k_idx])
        # instruction_selection: ld.shared.b32 + mul.ftz.f32; extent: <= TOKENS per element
        sVec[k_idx, token]     = cast(bf16, mul(prefix, r_k[i]))
        sVec[k_idx, 4 + token] = cast(bf16, mul(prefix, r_q[i]))
        # instruction_selection: mul.ftz.f32 + cvt.rn.bf16.f32 + st.shared.b16
        # extent: 2 scalars per element, 4 elements -> 8 st.shared.b16 and 8 of
        # the 14 cvt.rn.bf16.f32, identical in both bodies
        # The prefix multiply is FP32; the bf16 rounding happens on the way into
        # sVec, which is what the MMA consumes as its B operand.

    # ---- L and R: the WY coefficients --------------------------------------
    # ratio_scan accumulates sD of the *source* token between offsets, so the
    # dots see the decay between source and target.
    ratio = reg_tile(f32, [4]); fill(ratio, 1.0)
    for source_offset in range(TOKENS):
        source_token = token - source_offset
        if source_token >= 0:
            dot_kk = 0.0; dot_qk = 0.0
            for i in range(4):
                sk = sK[source_token, elem_start + i]
                dot_kk = fma(mul(r_k[i], sk), ratio[i], dot_kk)
                dot_qk = fma(mul(r_q[i], sk), ratio[i], dot_qk)
            # instruction_selection: ld.shared.v4.b32 (the 4 contiguous sK floats)
            # + mul.ftz.f32 + fma.rn.ftz.f32; extent: 4 iterations
            # Source order is `r * source_k * ratio`. At source_offset == 0 the
            # ratio is still 1.0 and folds away, so that offset emits 4 bare fma
            # (the first with C = 0f00000000) and the later offsets emit the
            # mul+fma pair.
            for offset in (16, 8, 4, 2, 1):
                dot_kk = add(dot_kk, bfly(dot_kk, offset, 31, 0xFFFFFFFF))
            for offset in (16, 8, 4, 2, 1):
                dot_qk = add(dot_qk, bfly(dot_qk, offset, 31, 0xFFFFFFFF))
            # instruction_selection: shfl.sync.bfly.b32 + add.ftz.f32
            # extent: 5 rounds x 2 dots x TOKENS source offsets
            # = 40 shfl at T=4, 30 at T=3 (20 at T=2) -- this is the single
            # biggest shuffle site in either body.
            if lane == 0:
                beta_source = sBeta[source_token]
                if source_token < token:
                    sL[token, source_token] = mul(beta_source, dot_kk)
                sR[token, source_token] = mul(beta_source, dot_qk)
                # instruction_selection: ld.shared.b32 + mul.ftz.f32
                # + st.shared.b32; extent: scalar
                # Live entries: sL = strict lower triangle (6 at T=4, 3 at T=3),
                # sR = lower triangle incl. diagonal (10 at T=4, 6 at T=3).
            if source_token > 0:
                for i in range(4):
                    ratio[i] = mul(ratio[i], sD[source_token, elem_start + i])
                # instruction_selection: ld.shared.v4.b32 + mul.ftz.f32; extent: 4

cta_sync()
# instruction_selection: bar.sync 0; extent: CTA
# The genuinely cross-warp edges are sVec (warp w writes columns w and 4+w, and
# every MMA warp's ldmatrix reads columns 0..7) and sL/sR (written by one warp's
# lane 0, read by all MMA warps in phases E and F). At T=3 warp 2 writes sVec
# columns 2 and 6, sL row 2 and sR row 2 here and then leaves -- warps 0,1 depend
# on that through this barrier and nothing else.
# The sState edge is intra-warp by the row-set argument in phase G and is ordered
# here only as a side effect. A port narrowing this barrier must reason from
# sVec/sL/sR, not from sState.

# ===========================================================================
# Phase D: the MMA chain  (T4:406-438 warps 0..3, T3:415-447 warps 0,1)
# ===========================================================================
if warp < ROWS_PER_CTA // 16:           # T=4: `warp_0 < 4`, statically true.
                                        # T=3: `warp_0 < 2`, LIVE.
    acc = reg_tile(f32, [4])
    for state_half in range(2):         # keys 0..63, then 64..127
        for mma_k in (0, 16, 32, 48):   # constexpr unroll -> 8 issues
            global_k = state_half * 64 + mma_k

            # B operand: sVec rows `global_k .. +15`, cols 0..15, transposed so
            # that n = sVec column and k = key.
            vec_frag = reg_tile(u32, [4])
            ldm(sVec @ swz((global_k + lane % 16) * 32 + (lane // 16) * 16),
                vec_frag, trans=True)
            # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16
            # extent: one x4 group per issue, 8 issues
            # vec_frag[2],[3] are cols 8..15 -- loaded by the x4 and never used.

            # A operand: this warp's 16 state rows, keys `mma_k .. +15`.
            state_frag = reg_tile(u32, [4])
            half = sState0 if state_half == 0 else sState1
            ldm(half @ swz((warp*16 + lane % 16) * 128 + (mma_k + (lane//16)*8) * 2),
                state_frag)
            # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.shared.b16
            # extent: one x4 group per issue, 8 issues

            mma(acc, state_frag, vec_frag, init=(state_half == 0 and mma_k == 0))
            # instruction_selection: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
            # extent: one issue; the first has an explicit zero C operand
            # ({0f00000000 x4}), the rest tie C to D

# The MMA chain is byte-identical at T=2, T=3 and T=4: 8 ldmatrix.x4 +
# 8 ldmatrix.x4.trans + 8 mma in every body. n = 8 was always 8 columns wide;
# what changes with T is only how many of those columns carry live data.
# C[16 rows][8 cols] += state[row][k] * sVec[k][col] over all 128 keys:
#   cols 0..TOKENS-1     -> (prefix_t (*) k_t) . h0   the delta-rule history term
#   cols 4..4+TOKENS-1   -> (prefix_t (*) q_t) . h0   the output history term
#   the rest              -> garbage (sVec columns never written)
# acc layout: acc[0],[1] = row `frag_row`, cols 2*lane_quad, +1;
#             acc[2],[3] = row `frag_row + 8`, same cols. So:
#   lane_quad 0 -> k-side tokens 0,1     lane_quad 2 -> q-side tokens 0,1
#   lane_quad 1 -> k-side tokens 2,3     lane_quad 3 -> q-side tokens 2,3
# At T=2 only lane_quad 0 and 2 carried live data; at T=4 all four do, and at
# T=3 all but the token-3 column do.

# ===========================================================================
# Phase E: quad broadcast and the WY forward substitution  (T4:439-483, T3:448-492)
# ===========================================================================
if warp < ROWS_PER_CTA // 16:
    ha_lo = reg_tile(f32, [4]); ha_hi = reg_tile(f32, [4])
    u_lo  = reg_tile(f32, [TOKENS]); u_hi = reg_tile(f32, [TOKENS])
    for t in range(4):                  # NOT range(TOKENS) -- see below
        ha_lo[t] = bcast(acc[t % 2],     quad_base + t // 2, 31, 0xFFFFFFFF)
    for t in range(4):                  # the source runs the two loops in this
        ha_hi[t] = bcast(acc[2 + t % 2], quad_base + t // 2, 31, 0xFFFFFFFF)
    # order (all four ha_lo, then all four ha_hi), not interleaved.
    # instruction_selection: shfl.sync.idx.b32; extent: scalar, 8 broadcasts in
    # BOTH bodies -- the source issues all 8 unconditionally.
    # The `quad_base + 1` half (ha_*[2],[3]) was dead at T=2. At T=4 both are
    # live. At T=3, ha_*[2] is live (token 2) and ha_*[3] is dead -- but its two
    # shuffles still execute, so the count is 8 there too. This is a place where
    # a "clean" port that skips the dead broadcast would emit fewer instructions
    # than the source; the port issues all 8.

    if lane_quad == 2:
        row_lo = warp * 16 + frag_row;  row_hi = row_lo + 8
        for t in range(TOKENS):         # serial forward substitution, depth TOKENS
            base_t = (sToken[t] * HV + hv) * 128
            solved_lo = sub(cast(f32, v[base_t + tile_row_base + row_lo]), ha_lo[t])
            solved_hi = sub(cast(f32, v[base_t + tile_row_base + row_hi]), ha_hi[t])
            # instruction_selection: ld.global.nc.b16 + cvt.f32.bf16 + sub.ftz.f32
            # extent: 2 scalars per token -> 8 ld.global.nc.b16 at T=4, 6 at T=3
            for s in range(TOKENS):
                if s < t:
                    solved_lo = sub(solved_lo, mul(sL[t, s], u_lo[s]))
                    solved_hi = sub(solved_hi, mul(sL[t, s], u_hi[s]))
                # instruction_selection: mul.ftz.f32 + sub.ftz.f32; extent: scalar,
                # taken once per (t,s) with s < t -> 6 taken pairs at T=4, 3 at
                # T=3; total sub.ftz.f32 in phase E is 20 at T=4 and 12 at T=3.
                # The sL COEFFICIENTS ARE VECTORIZED, not loaded one scalar per
                # pair: T=4 fetches its 6 live entries with three instructions --
                # ld.shared.b32 (sL[1][0], +24656), ld.shared.v2.b32 (sL[2][0..1],
                # +24672), ld.shared.v4.b32 (sL[3][0..3], +24688, over-fetching
                # the never-written sL[3][3]); T=3 fetches its 3 with two --
                # ld.shared.v4.b32 (+15424, spanning sL[1][0..2] and sL[2][0],
                # over-fetching the dead sL[1][1..2]) and ld.shared.b32
                # (sL[2][1], +15440). The T=2 sibling's "scalar per pair" reads
                # correctly there because exactly one pair is taken; it does not
                # carry over.
            u_lo[t] = solved_lo;  u_hi[t] = solved_hi

    # ---- the quad broadcast that came back to life -------------------------
    for t in range(TOKENS):
        u_lo[t] = bcast(u_lo[t], quad_base + 2, 31, 0xFFFFFFFF)
        u_hi[t] = bcast(u_hi[t], quad_base + 2, 31, 0xFFFFFFFF)
    # instruction_selection: shfl.sync.idx.b32; extent: scalar, 2*TOKENS
    # -> 8 at T=4, 6 at T=3
    # At T=2 these broadcasts were an IDENTITY: all 32 lanes execute them, but
    # the only lane that consumes the result is lane_quad == 2, which is the
    # broadcast's own source lane -- so the consumer received its own value and
    # no other lane read one. The T=2 port dropped them on that basis.
    # At T=3/T=4 phase F runs on lane_quad >= 2, so lane 4f+3 needs the residuals
    # that lane 4f+2 solved, and the broadcast stops being an identity. Dropping
    # it silently corrupts every output token >= 2.
    # Total shfl.sync.idx.b32 in the body: 17 at T=4 (1 + 8 + 8), 15 at T=3
    # (1 + 8 + 6).

# ===========================================================================
# Phase F: the outputs -- the one rewritten phase  (T4:484-532, T3:493-541)
# ===========================================================================
# T=2 had one block per token under `lane_quad == 2`. Both new bodies replace
# that with ONE generic block under `lane_quad >= 2`, where the lane index picks
# the token pair. The two bodies' text here is identical apart from the literal
# TOKENS; this is not a re-parameterization of the T=2 code, it is different code.
if warp < ROWS_PER_CTA // 16 and lane_quad >= 2:
    token0 = (lane_quad - 2) * 2        # lane 4f+2 -> 0, lane 4f+3 -> 2
    token1 = token0 + 1                 #             1,              3
    row_lo = warp * 16 + frag_row;  row_hi = row_lo + 8

    out0_lo, out1_lo = acc[0], acc[1]   # the q-side columns for THIS lane_quad:
    out0_hi, out1_hi = acc[2], acc[3]   # cols 4,5 at lane_quad 2; 6,7 at 3
    # No shuffle is needed to get the right column pair -- the accumulator layout
    # already keys the column pair off lane_quad, which is why the rewrite works.

    for s in range(TOKENS):
        residual_lo = u_lo[s];  residual_hi = u_hi[s]
        coef0 = sR[token0, s] if token0 >= s else 0.0
        coef1 = sR[token1, s] if (token1 < TOKENS and token1 >= s) else 0.0
        #                         ^^^^^^^^^^^^^^^^ DEVIATION, T=3 only: the
        # source has no such term and reads sR[9..11] at lane_quad == 3. See
        # "The T=3 lane_quad == 3 hazard" below. At T=4 the added term is
        # statically true and changes nothing.
        # instruction_selection: ld.shared.b32; extent: scalar per taken mask
        out0_lo = fma(coef0, residual_lo, out0_lo)
        out1_lo = fma(coef1, residual_lo, out1_lo)
        out0_hi = fma(coef0, residual_hi, out0_hi)
        out1_hi = fma(coef1, residual_hi, out1_hi)
        # instruction_selection: fma.rn.ftz.f32; extent: 4 per source token
        # -> 16 fma in phase F at T=4, 12 at T=3. The masked-out coefficient is
        # a real zero-operand fma, not a skipped iteration.

    slot0 = sSlot[token0]
    slot1 = sSlot[token1] if token1 < TOKENS else -1   # DEVIATION, T=3 only
    for (t, slot, o_lo, o_hi) in ((token0, slot0, out0_lo, out0_hi),
                                  (token1, slot1, out1_lo, out1_hi)):
        if t < TOKENS:                  # T=4: statically true. T=3: LIVE for token1.
            base_t = (sToken[t] * HV + hv) * 128 + tile_row_base
            active = slot >= 0
            copy_r2g(cast(bf16, o_lo if active else 0.0), out[base_t + row_lo])
            copy_r2g(cast(bf16, o_hi if active else 0.0), out[base_t + row_hi])
            # instruction_selection: st.global.b16 (7 in each body) +
            # cvt.rn.bf16.f32 (6 in each body -- the zero branch converts one 0.0
            # and reuses it for both rows)
            # A padded row writes EXPLICIT zeros; the upstream test asserts them
            # bit-exactly, so this is not an "unwritten" path.

# ===========================================================================
# Phase G: publish sU  (T4:533-546, T3:542-555)
# ===========================================================================
if warp < ROWS_PER_CTA // 16:
    if lane_quad == 2:
        for t in range(TOKENS):
            sU[t, warp*16 + frag_row]     = u_lo[t]
            sU[t, warp*16 + frag_row + 8] = u_hi[t]
        # instruction_selection: st.shared.b32; extent: 2 scalars per token
        # -> 8 at T=4, 6 at T=3
    warp_sync()
    # instruction_selection: bar.warp.sync; extent: warp
    # A warp barrier suffices because producer and consumer are the same warp:
    # writer rows are warp*16 + frag_row{,+8}, i.e. [16w, 16w+16); reader rows
    # below are group*8 + r with group = tid/16, i.e. warp w owns groups 2w and
    # 2w+1 -> rows [16w, 16w+16). This holds at 64, 96 and 128 threads alike.
    # Any port that changes the row-to-thread mapping must re-derive it or
    # promote to cta_sync.

# The BLOCK_CHECKPOINT_MMA != 0 block (T4:547-659, T3:556-668), including the
# source's third mma.sync site, is statically dead: the macro is 0 and the
# binding static-asserts it. Not transcribed. It emits no instructions at all in
# either exported PTX.

# ===========================================================================
# Phase H: recurrence and checkpoints  (T4:660-696, T3:669-705)
# ===========================================================================
if group < ROWS_PER_CTA // 8:           # T=4: `group < 8`, statically true.
                                        # T=3: `group < 4`, LIVE.
    for t in range(TOKENS):             # serial over tokens
        slot_t = sSlot[t]
        beta_t = sBeta[t]
        for row_local in range(8):      # constexpr unroll
            row    = owned_row_base + row_local
            update = mul(sU[t, row], beta_t)
            # instruction_selection (slot_t, beta_t), PER BODY:
            #   T=4: ld.shared.b32 x2 per token (8 total) -- sBeta is 16 B at
            #        24576 and sSlot 16 B at 24592, so sBeta[t] and sSlot[t] are
            #        16 B apart and can never merge.
            #   T=3: ONE ld.shared.v4.b32 at 15360 covering sBeta[0..2] AND
            #        sSlot[0] -- sBeta is only 12 B, so those four words are
            #        contiguous at a 16-byte-aligned base -- plus ld.shared.b32
            #        for sSlot[1] (15376) and sSlot[2] (15380). Three
            #        instructions for the whole phase; sBeta is never
            #        scalar-loaded IN THIS PHASE at T=3 (phase C still reads it
            #        scalar, three times).
            # instruction_selection (the 8 consecutive sU rows), PER BODY:
            #   T=4: ld.shared.v4.b32 x2 per token (+24768, +24784) -- sU's base
            #        is 16-byte aligned and the row stride is 256 B, so every
            #        thread's 8-float run is 0 mod 16.
            #   T=3: ld.shared.b32 + ld.shared.v4.b32 x2 per token (+15484,
            #        +15488, +15504) -- sU's base 15484 is 12 mod 16 and every
            #        run inherits that, so nvcc peels one scalar and OVER-FETCHES
            #        one float (9 fetched for the 8 needed).
            # plus mul.ftz.f32; extent: scalar per row
            #
            # Every shared-load form that differs between the two bodies differs
            # for one reason, and it cuts both ways. T=3's 12- and 36-byte
            # metadata regions (sBeta, sSlot, sToken, sL, sR) push everything
            # after them OFF the 16-byte grid -- sL to 15412 (4 mod 16) and sU to
            # 15484 (12 mod 16), even though sU is itself 384 B -- so nvcc peels
            # a scalar off runs that are clean v4s at T=4. In one place it runs
            # the other way: sBeta sits at 15360, which IS 16-byte aligned, and
            # its 12-byte size puts sBeta[0..2] + sSlot[0] in a single aligned
            # quad, giving T=3 a merged v4 that T=4 cannot form. Either
            # direction, the port must take its load forms per body, not from the
            # shared skeleton.
            words = reg_tile(u32, [4])
            for i in range(8):
                hist[row_local][i] = fma(hist[row_local][i], sD[t, k_start + i],
                                         mul(update, sK[t, k_start + i]))
                # instruction_selection: ld.shared.v4.b32 x2 (sD) + x2 (sK) PER
                # TOKEN -- the two slices are invariant across `row_local`, so
                # nvcc hoists them out of the 8-row loop -- plus mul.ftz.f32 +
                # fma.rn.ftz.f32; extent: 8 iterations of the arithmetic.
                # Dynamically that is 4 shared loads per token (16 per thread at
                # T=4, 12 at T=3); statically the census shows 28/20 because the
                # whole block is replicated 2*TOKENS-1 times (see below).
                # The source writes `hist*sD + update*sK` and the compiler
                # contracts the FIRST product: `update*sK` is the rounded one and
                # `hist*sD` is fused into the FMA. Inverting this rounds the wrong
                # product at the kernel's only stateful accumulation, which feeds
                # both the checkpoint and every later token's history.
                # The recurrence advances UNCONDITIONALLY -- only the store below
                # is predicated -- and it stays FP32, so token t+1 consumes the
                # un-rounded token-t state rather than the bf16 checkpoint.
            for p in range(4):
                words[p] = cast(bf16x2, (hist[row_local][2*p], hist[row_local][2*p+1]))
            # instruction_selection: cvt.rn.bf16x2.f32; extent: 4 pairs per row
            # -> 128 at T=4 (4 tokens x 8 rows x 4), 96 at T=3
            if slot_t >= 0:
                copy_r2g(words, state[slot_t*STATE_SLOT_STRIDE + hv*16384
                                      + (tile_row_base + row)*128 + k_start : +8])
                # instruction_selection: st.global.v4.b32; extent: one 16-byte
                # tile -> 32 at T=4 (8 rows x 4 tokens), 24 at T=3
# EVERY token's slot receives a full ROWS_PER_CTA-row write: these kernels
# checkpoint per token, they do not write one final state.
```

## The T=3 `lane_quad == 3` hazard, and the port's deliberate deviation

Phase F is byte-identical between the two bodies, but `TOKENS = 3` is odd, so at
`lane_quad == 3` the generic block computes `token0 = 2` (valid) and
`token1 = 3` (**out of range**). Before the `token1 < 3` mask discards the
results, the source has already:

| read | lands in | why it is safe in the source |
| --- | --- | --- |
| `sR[token1*3 + s]` for `s = 0,1,2`, i.e. `sR[9..11]` | `sR` is `[3][3]` = 36 B at offset 15448, so `sR[9..11]` is the **first 12 bytes of `sU`** (offset 15484) | in-bounds shared memory; the value only feeds `out1_*`, which is never stored |
| `out1_lo/out1_hi` from `acc[1]`/`acc[3]` | MMA column 7 = `sVec` column 7, never written | garbage in, garbage discarded |
| `sSlot[token1]` = `sSlot[3]` | `sSlot` is `[3]` = 12 B at 15372, so this is `sToken[0]` | only used as `slot1`, which the mask discards |

**The port predicates those reads on `token1 < TOKENS` instead of issuing them.**
This is a deliberate deviation from a mechanical transcription, and it is the
only one in either kernel. The justification:

1. It is **numerically identical** — every value read is provably dead.
2. TIRx has no equivalent of "read past a `T.decl_buffer` view into the next
   region and rely on the arena layout". Reproducing the source literally would
   mean hand-computing byte offsets that deliberately leave the declared region,
   which `check_low_level_ir` and every reviewer would read as a bug.
3. The cost is four `ld.shared.b32` (three `sR` plus one `sSlot`) and one
   predicate, on the 16 threads of 96 with `lane_quad == 3` in warps 0 and 1,
   in one phase — measurable against the perf gate, not against intuition.

Any port that instead lets `token1 = 3` reach `sR`/`sSlot` **must** keep the
`token1 < TOKENS` guard on the store, or it writes token 3 of a 3-token sequence
over another sequence's output.

## Was dead at T=2, alive now

| element | T=2 | T=4 | T=3 |
| --- | --- | --- | --- |
| MMA acc columns 2,3,6,7 | garbage | live | live except cols 3 and 7 (token 3) |
| `quad_base+1` broadcasts → `ha_*[2],[3]` | dead (8 issued) | live (8 issued) | `[2]` live, `[3]` dead (8 still issued) |
| **`u_lo`/`u_hi` quad broadcasts** | **identity (dropped)** | **live, 8** | **live, 6** |
| `sL` live entries | 1 | 6 | 3 |
| `sR` live entries | 3 | 10 | 6 |
| solve depth | 2 | 4 | 3 |
| phase-F writer lanes per quad | 1 | 2 | 2 (one with a discarded half) |

Nothing that was live at T=2 died. Still dead in both: `vec_frag[2],[3]`, `sVec`
columns 8..15, `sGramA0`/`sGramA1`, and the whole `BLOCK_CHECKPOINT_MMA` block.

## Instruction-selection summary

Read off line-info PTX exported by re-running FlashInfer's own nvcc command line
(`.porting/flashkda_decode_t{4,3}_*/ptx/<variant>/kernel.ptx`), counted with
`ptx_census.py`, which drops the `--source-in-ptx` comment echo and peels the
inline-asm braces before matching. Body totals: **2639** instructions at T=4 and
**2167** at T=3, of which 56 and 48 are predicated (`@%pN`) branches; the
unpredicated remainder is 2583 / 2119. None of the forms tabulated below is ever
predicated in these bodies, so every row is a straight count.

| form | T=4 | T=3 | where |
| --- | ---: | ---: | --- |
| `mul.ftz.f32` | 588 | 429 | H 504/360, C 59/41, A 13/22, E 12/6 |
| `fma.rn.ftz.f32` | 504 | 364 | H 448/320, C 32/24, F 16/12, A 8/8 |
| `shl.b32` / `and.b32` | 182 / 161 | 156 / 141 | the bf16 widening pair plus index and swizzle arithmetic |
| `cvt.rn.bf16x2.f32` | 128 | 96 | phase H, `TOKENS × 8 rows × 4 pairs` |
| `add.ftz.f32` | 51 | 49 | butterfly tails, the eps adds, and (T=3) the `1 +` in the gate |
| `ld.shared.v4.b32` | 51 | 37 | H 42/31 — T=4: 28 sD/sK + 14 sU; T=3: 20 sD/sK + 10 sU + **1 sBeta/sSlot merge**. C 7/5, E 2/1 |
| `shfl.sync.bfly.b32` | 50 | 40 | C 40/30 (2 dots × 5 rounds × TOKENS), A 10/10 |
| `ld.shared.b32` | 41 | 37 | scalars: prefix gate, sBeta, sL, sR, sSlot, sToken, sInit |
| `ld.shared.v2.b32` | 2 | 1 | T=4: the phase-E `sL[2][0..1]` pair and the phase-F `sSlot[token0..1]` pair. T=3: the phase-E `sToken[0..1]` pair |
| `xor.b32` | 34 | 34 | the shared swizzle — identical, it is a per-access constant folding |
| `st.global.v4.b32` | 32 | 24 | the checkpoints, `8 rows × TOKENS` |
| `sub.ftz.f32` | 20 | 12 | the WY residuals |
| `st.shared.b32` | 19 | 15 | G 8/6, C 7/5, A 4/4 |
| `st.shared.v4.b32` | 18 | 18 | B 16 + A 2 — **identical**, phase B is T-independent |
| `shfl.sync.idx.b32` | 17 | 15 | E 16/14 + 1 `make_warp_uniform` |
| `cvt.rn.bf16.f32` | 14 | 14 | 8 sVec column stores + 6 output conversions |
| `ld.global.nc.b16` | 9 | 7 | E 8/6 (the `v` loads) + 1 `beta` |
| `cvt.f32.bf16` | 9 | 7 | same sites |
| `ldmatrix…x4.shared.b16` | **8** | **8** | the A operand, one per MMA issue |
| `ldmatrix…x4.trans.shared.b16` | **8** | **8** | the B operand, one per MMA issue |
| `mma.sync…m16n8k16.f32.bf16.bf16.f32` | **8** | **8** | 1 zeroing + 7 accumulating |
| `ld.global.v4.b32` | 8 | 8 | the state gather — **not** `.nc` |
| `st.shared.b16` | 8 | 8 | the sVec columns |
| `st.global.b16` | 7 | 7 | the outputs |
| `ld.global.nc.b32` | 5 | **10** | T=3's extra 5 = 4 `dt_bias` + 1 `A_log` |
| `ld.global.nc.v2.b32` | 3 | 3 | q, k, g |
| `ex2.approx.ftz.f32` | 4 | **9** | T=4: one per element. T=3: 2 per element + the hoisted `gate_a` |
| `rsqrt.approx.ftz.f32` | 2 | 2 | the two L2 norms |
| **`div.approx.ftz.f32`** | — | **4** | the gate, one per element |
| **`neg.ftz.f32`** | — | **1** | `-gate_a`, hoisted out of the element loop |
| `bar.sync` / `bar.warp.sync` | 2 / 1 | 2 / 1 | the two CTA barriers and the sU warp barrier |

Consequences the port must honour:

1. **Every float operation is `.ftz`.** `-use_fast_math` plus plain CUDA
   operators give `mul/add/sub.ftz.f32` and `fma.rn.ftz.f32`; `__expf`,
   `rsqrtf` and the division give `ex2.approx.ftz.f32`,
   `rsqrt.approx.ftz.f32` and `div.approx.ftz.f32`. There is **no plain-`.f32`
   arithmetic anywhere in either body** — including inside T=3's gate. This
   matches the ported T=1/T=2 siblings and is the opposite of the two CuTe-DSL
   KDA decode ports, whose helpers are deliberately non-FTZ.
2. **`.nc` follows read-only-ness, not `__restrict__` alone.** q, k, g, v, beta,
   the metadata and (T=3) A_log/dt_bias load through `ld.global.nc.*`; the state
   gather is a plain `ld.global.v4.b32` because the same kernel writes that
   buffer.
3. **`dt_bias` is NOT vectorized.** The four fp32 a lane needs are contiguous
   (`[%rd30]`, `+4`, `+8`, `+12`) and nvcc still emits four scalar
   `ld.global.nc.b32`. There is no `ld.global.nc.v4.b32` anywhere in the T=3
   body. A port that "improves" this to one vector load diverges from the source
   in the one phase every thread executes.
4. **The MMA operands are bf16 and the accumulator is FP32**, so the kernels lose
   precision an FP32 oracle does not: the measured gap to a sequential FP32
   oracle is 6.1e-05 (T=4) and 3.1e-05 (T=3) on the output, against 6e-08 for the
   register-only t1_direct sibling.

### The `2T−1` static inflation in phase H

Phase H's recurrence appears **`2·TOKENS − 1` times** in the compiled body — 7
copies at T=4, 5 at T=3, 3 at T=2 — each an exact 64-instruction block
(8 rows × 8 keys). Verified by basic-block attribution: the `fma.rn.ftz.f32` at
the recurrence line are spread over exactly 7 / 5 / 3 blocks with 64 in each.

This is **nvcc tail duplication around the `slot_t >= 0` store branch**, not
extra arithmetic: token 0's recurrence has one incoming path, and every later
token has two (the taken and not-taken successors of the previous token's store
predicate), giving `1 + 2(T−1)`. The stores and the bf16 packing are *not*
duplicated — they sit inside the predicate — which is why `st.global.v4.b32` is
exactly `8 × TOKENS` and `cvt.rn.bf16x2.f32` exactly `32 × TOKENS`.

At runtime each thread still executes `TOKENS × 64` FMAs. The consequence for
this port is a measurement one: a **static** PTX-count comparison between the
port and the source will show phase H inflated by up to 1.75× unless TIRx's
codegen happens to make the same duplication choice. Compare dynamic work, or
compare the store and packing counts, which are duplication-free.

## TIRx module and benchmark contract

- Modules `tirx_kernels/flashinfer/kda/flashkda_decode_t4_precomputed.py` and
  `flashkda_decode_t3_lower_bound.py`, `KERNEL_META["name"]` matching, cc 10.
  Two kernels ⇒ two modules; the registry allows one `KERNEL_META` per module.
  T=3 sibling-imports every PTX/math/shuffle/global/shared helper from the T=2
  module, as does T=4; only geometry constants, the data recipe, the oracles and
  the config matrices are per-module.
- Reference for both: the frozen export itself through the JIT module's direct
  ABI, so the comparison is kernel-only and every metadata combination —
  negative slots, arbitrary `num_accepted_tokens` — is reachable. T=3's call
  passes **real** `A_log`, `dt_bias` and `lower_bound`; T=4's passes fp32 dummies
  and `0.0`, which that body never dereferences.
- **T=4: 9 bench rows** — `hv32h16` × N{8,16,32,64,128} (parity with FlashInfer's
  own export bench), plus `hv16h16` and `hv12h12` at N{8,64}. All split2.
- **T=3: 5 bench rows** — `hv16h16` × N{1,2,4,8,16}. That is the kernel's
  *entire* legal domain: the dispatcher enforces `H == HV == 16` and
  `N ∈ {1,2,4,8,16}`, so there is no shape outside these five.
- `verify_dispatch.py` asserts every row reaches its variant, and for T=3 also
  that 10 near-miss contracts (wrong `num_spec_tokens`, absent or non-negative
  `lower_bound`, missing/bf16/too-short `A_log`/`dt_bias`, `use_gate_in_kernel`
  off) and 4 out-of-domain shapes are refused — the negative space is most of
  T=3's contract.
