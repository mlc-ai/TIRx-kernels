<!--
This file is a design sketch for a TIRx port of code from FlashInfer
(https://github.com/flashinfer-ai/flashinfer @ f2e04400),
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# FlashKDA cake T=5 coefficient-gram decode: coarse execution sketch

**Non-executable.** This is the semantic execution skeleton, not runnable code.
The source of truth for the implementation is
`tirx_kernels/flashinfer/kda/flashkda_decode_t5_gram.py`.

Transcribed from four frozen generated exports, all with symbol
`kernel_flashinfer_recurrent_kda_wy_vtile_short` (flashinfer-ai/flashinfer @
`f2e04400`), 787 body lines each:

- `csrc/kda/flashkda_decode_d128_t5_precomputed_gram_split1.cu` — `S1:NNN`
- `csrc/kda/flashkda_decode_d128_t5_precomputed_gram_split2.cu` — `S2:NNN`, and
  the default for a bare `:NNN`
- `csrc/kda/flashkda_decode_d128_t5_precomputed_gram_split4.cu` — `S4:NNN`
- `csrc/kda/flashkda_decode_d128_t5_precomputed_gram_split8.cu` — `S8:NNN`

**Four exports, one kernel.** All four are the same body at four value splits.
Unlike T=2 (always split4) and T=4 (always split2), **the T=5 split is chosen by
shape** — `recurrent_kda.py:1181-1191`, with `W = N·HV` and `S` SMs:

| band | `8W ≤ 3S` | `2W ≤ S` | `4W ≤ 3S` | `2W ≤ 3S` | else |
| --- | --- | --- | --- | --- | --- |
| split | 8 | 2 | 4 | 2 | 1 |

The split4 island sits *between* two split2 bands, so the policy is not monotonic
in `W`. All four exports are production-reachable, so all four are in scope and
the split is a constexpr, as in the two-split T=1 port.

## What changed relative to the ported T=4 body

Stripping integer literals and diffing phase by phase against
`flashkda_decode_d128_t4_precomputed_split2.cu` (the already-ported sibling):

| phase | T=4 → T=5 |
| --- | --- |
| prologue, registers | **identical** modulo constants; `ratio_scan[4]` is now dead |
| A preprocess | **identical** modulo `T` (nat clamp `[0,3]` → `[0,4]`, `ssm_idx` stride 4 → 5) |
| **order** | **B and C swap**: T=4 ran gather-then-coefficients, T=5 runs coefficients-then-gather |
| C sVec publish | **changed**: q columns move `4..7` → **`8..12`**; a new `k / prefix` copy is published to `sGramA0/1` |
| C butterflies | **deleted** — the whole `ratio_scan` + per-source dot-product block is gone |
| C″ gram | **new**: a named barrier and 16 tensor-core issues that produce `sL`/`sR` directly |
| B state gather | identical modulo the split's rows-per-group |
| D MMA chain | **doubled**: `mma_acc_c` goes live, a second `mma.sync` per step over sVec columns 8..15 |
| E broadcasts + solve | **extended**: 8 → 20 accumulator broadcasts, solve depth 4 → 5 |
| F outputs | **rewritten again**: bases come from `hc_*` rather than `mma_acc`, plus a fifth-token tail |
| G publish `sU` | identical modulo `T` (split8 swaps the warp barrier for a CTA barrier) |
| H recurrence | identical modulo `T` |

So the port is the ported T=4 skeleton re-parameterized by `T` and the split,
with **three** structural deltas: the gram block (with its barrier and its new
shared regions), D's second MMA and the phase-E broadcasts that feed on it, and
phase F's `hc_*` rewrite. This sketch gives the common skeleton once and each
delta its own section.

**Fixed specializations.**

| knob | split1 | split2 | split4 | split8 |
| --- | --- | --- | --- | --- |
| `TOKENS` | 5 | 5 | 5 | 5 |
| `GATE_KIND` | 0 (precomputed log-gate) | 0 | 0 | 0 |
| `LAUNCH_THREADS` | **256** (8 warps) | 160 (5 warps) | 160 | 160 |
| `ROWS_PER_CTA` | 128 | 64 | 32 | 16 |
| `ROWS_PER_GROUP` | 8 | 8 | 8 | **2** |
| `ROW_GROUPS` (B, H) | 16 | 8 | 4 | 8 |
| `MMA_WARPS` (D–G) | 8 | 4 | 2 | **1** |
| `GRAM_WARP` | 0 | 0 | **4** | **4** |
| gram barrier shape | sync/arrive | sync/arrive | **all five sync** | sync/arrive |
| `sU` publish barrier | `__syncwarp` | `__syncwarp` | `__syncwarp` | **`__syncthreads`** |
| `SMEM_TOTAL` | 49024 | 31360 | 22528 | 18048 |
| grid | `(HV·1, N)` | `(HV·2, N)` | `(HV·4, N)` | `(HV·8, N)` |

`HEAD_DIM = 128`, `DIRECT_PREFIX_CHECKPOINT = 0`, `BLOCK_CHECKPOINT_MMA = 0`,
`NUM_MAIN_STAGES = 1`, `coefficient_gram = true` for all four. `#define THREADS
256` appears in every body but is vestigial — it only feeds
`static_assert(THREADS == 256)` at `binding_impl.cuh:40`; the real block size is
`FLASHKDA_DECODE_LAUNCH_THREADS`.

**Out of scope.** T ∈ {1,2,3,4} (ported separately); **T=6**, which is the same
generator family but has its own four bodies with different launch threads; the
`BLOCK_CHECKPOINT_MMA` schedule, dead in every export.

## Pipeline at a glance

No warp specialization, no asynchronous pipeline: no `cp.async`, no mbarrier, no
TMA, no atomics, nothing double-buffered. What is new at T=5 is a **partial
barrier**: one warp waits on the other four inside a phase, rather than the whole
CTA meeting at a `__syncthreads`.

The value split is a **partition**: CTA `(hv, value_tile)` owns value rows
`[ROWS_PER_CTA·value_tile, +ROWS_PER_CTA)` of the `[V,K]` state, and every CTA
redundantly recomputes the whole token preprocessing for its own `(n, hv)`. No
CTA combines anything with any other — which is why the split is free to move
with shape. Measured in `characterize_source.py`: all four splits agree on both
the output and the whole state pool, with `max|Δ| = 0` observed against an
asserted tolerance of 1e-2.

### Roles

| Role | Ownership | Publication/reuse edges |
| --- | --- | --- |
| CTA `(hv, value_tile, n)` | value head `hv`, sequence `n`, `ROWS_PER_CTA` value rows | independent |
| warp `w < 5`, phases A and C′ | **token `w`** (5 warps ⇄ 5 tokens) | publishes `sK`/`sD`/`sBeta`/`sSlot`/`sToken`/(warp 0)`sInit` across barrier #1; publishes `sGramA*` row `w` and `sVec` columns `w`, `8+w` across **the named barrier** to the gram warp; the same `sVec` columns reach every *other* MMA warp only across barrier #2, since those warps merely arrive at the named barrier (split4 excepted -- there all five block on it) |
| warp `GRAM_WARP` | **all 25 WY coefficients**: a 16×8×128 Gram product per side (16×16×128 for both sides together) | consumes every token warp's `sGramA*` row and `sVec` column; publishes `sL`/`sR` across barrier #2 |
| thread `tid`, `group < ROW_GROUPS`, phases B and H | `ROWS_PER_GROUP` value rows × 8 keys | gathers state into `hist` **and** stages it to `sState`; later writes all five tokens' checkpoints |
| warp `w < MMA_WARPS`, phase D | value rows `w·16 .. +15` | consumes all staged `sState` rows and all of `sVec` |
| lane with `lane_quad == 2` | the depth-5 WY solve for rows `w·16 + frag_row` and `+8`; the `sU` publish; **and token 4's output** | the only lanes that load `v` |
| lane with `lane_quad >= 2` | tokens 0,1 on lane `4f+2`; tokens 2,3 on lane `4f+3` | the only lanes that store `out` |

Dependency chain: preprocess → **CTA barrier** → sVec/sGramA publish →
**named barrier** → gram MMA → state gather → **CTA barrier** → main MMA → WY
solve → outputs → `sU` → **warp barrier** → recurrence and checkpoints.

Note the order: the gram warp's tensor-core work sits *between* the two CTA
barriers, overlapping the state gather that every other warp is doing. That is
the reason for a named barrier rather than a third `__syncthreads`: only the five
token warps have to meet, and only at the moment `sGramA*` becomes readable.

### Who participates in the named barrier

The gram block is nested inside `if (warp_0 < 5)` (`:293` opens, `:409` closes),
so the participant set is **exactly the five token warps = 160 threads in every
split**, including split1's 256-thread launch, where warps 5..7 never touch the
named barrier. (They do execute both `__syncthreads()` — "barrier #1" and
"barrier #2" below always mean those two CTA barriers; the named barrier is
always called by name, never "barrier 1", even though its id is 1.) Reading that nesting is what rules out the alternative — and wrong —
reading in which 256 threads arrive at a 160-count barrier and the gram warp is
released after a single producer has published.

| split | gram warp | who executes `barrier.sync 1,160` | who executes `barrier.arrive 1,160` |
| --- | --- | --- | --- |
| 1, 2 | 0 | warp 0 | warps 1..4 (warps 5..7 are outside the guard) |
| 8 | 4 | warp 4 | warps 0..3 |
| **4** | 4 | **all five token warps** (unconditional) | nobody — the arm is `else if (0)`, dead |

The PTX confirms the split4 shape independently: its body contains **one**
`barrier.*` instruction, the other three contain two.

### Guards, per split

| guard | site | split1 | split2 | split4 | split8 |
| --- | --- | --- | --- | --- | --- |
| `warp_0 < 5` (A, C′, gram) | `:180`, `:293` | live (8 warps) | statically true | statically true | statically true |
| `group < ROW_GROUPS` (B, H) | `:411`, `:768` | 16 of 16 | 8 of 10 | 4 of 10 | 8 of 10 |
| `warp_0 < MMA_WARPS` (D–G) | `:448`, `:562`, `:641` | 8 of 8 | 4 of 5 | 2 of 5 | 1 of 5 |

At split1 the token guard is the live one (warps 5..7 skip A/C′/gram entirely);
at the other three the value guards are live and warp 4 is a **token-only warp** —
it preprocesses token 4, participates in the named barrier, and at split4/split8
also runs the gram, but never gathers state, never issues the main MMA, and never
stores output.

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
exp(x); rsqrt(x); div(a,b)             # div is new at T=5
dot(a, b)                              # `dot` = one explicit FMA chain
mma(acc, a_frag, b_frag, init=False)   # one mma.sync.m16n8k16 issue
```

Sync: `cta_sync()`, `warp_sync()`, and new at T=5:

```python
named_bar_sync(bar_id, threads)        # barrier.sync  1, 160 -- blocks
named_bar_arrive(bar_id, threads)      # barrier.arrive 1, 160 -- does not block
```

**Every shuffle is width-32 with a full member mask** — the operands are always
`(31, 0xFFFFFFFF)`. **`dot` is not a compound op**: each one expands to an
explicit FMA chain in a fixed association order, written out at its site because
the order is load-bearing.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================
HEAD_DIM = 128
TOKENS = 5
VALUE_SPLIT   in (1, 2, 4, 8)                 # chosen by shape, constexpr here
THREADS       = 256 if VALUE_SPLIT == 1 else 160
ROWS_PER_CTA  = 128 // VALUE_SPLIT            # 128 / 64 / 32 / 16   (:159)
ROWS_PER_GROUP= 2 if VALUE_SPLIT == 8 else 8  # (:160, S8:160)
ROW_GROUPS    = ROWS_PER_CTA // ROWS_PER_GROUP
MMA_WARPS     = ROWS_PER_CTA // 16            # 8 / 4 / 2 / 1
GRAM_WARP     = 4 if VALUE_SPLIT in (4, 8) else 0
H, HV, HEAD_RATIO, GATE_TOKEN_STRIDE, STATE_SLOT_STRIDE   # host-derived

grid  = (HV * VALUE_SPLIT, num_seqs)          # binding_impl.cuh:64
block = (THREADS,)

# Runtime scalars: `scale`. `lower_bound` is passed as 0.0 and the A_log/dt_bias
# pointers are 1-element dummies -- GATE_KIND 0 never dereferences them, which is
# measured, not assumed (characterize_source.py feeds 1e3 / -7.5 dummies and gets
# bit-identical output).

# The source uses dynamic smem; every total fits the 49152 B static limit (split1
# with 128 B to spare), so the port declares one static arena at the same
# alignment and carves the same byte offsets -- the swizzled ldmatrix addressing
# depends on both.
#
#                                                  s1 /    s2 /    s4 /    s8
arena   = smem_arena(SMEM_TOTAL, align=1024)
sState0 = smem_view(arena, ..., bf16, [ROWS_PER_CTA, 64])  #     0/     0/    0/    0
sState1 = smem_view(arena, ..., bf16, [ROWS_PER_CTA, 64])  # 16384/  8192/ 4096/ 2048
sVec    = smem_view(arena, ..., bf16, [128, 16])           # 32768/ 16384/ 8192/ 4096
sK      = smem_view(arena, ..., f32,  [5, 128])            # 36864/ 20480/12288/ 8192
sD      = smem_view(arena, ..., f32,  [5, 128])            # 39424/ 23040/14848/10752
sBeta   = smem_view(arena, ..., f32,  [5])                 # 41984/ 25600/17408/13312
sSlot   = smem_view(arena, ..., i32,  [5])                 # 42004/ 25620/17428/13332
sToken  = smem_view(arena, ..., i32,  [5])                 # 42024/ 25640/17448/13352
sInit   = smem_view(arena, ..., i32,  [1])                 # 42044/ 25660/17468/13372
sL      = smem_view(arena, ..., f32,  [5, 5])              # 42060/ 25676/17484/13388
sR      = smem_view(arena, ..., f32,  [5, 5])              # 42160/ 25776/17584/13488
sU      = smem_view(arena, ..., f32,  [5, ROWS_PER_CTA])   # 42260/ 25876/17684/13588
sGramA0 = smem_view(arena, ..., bf16, [16, 64])            # 44928/ 27264/18432/13952
sGramA1 = smem_view(arena, ..., bf16, [16, 64])            # 46976/ 29312/20480/16000

# sGramA0/1 are the layout news: in the T<=4 bodies those #defines aliased sVec
# -- sGramA0 at sVec's base and sGramA1 at sVec+2048 (16384 and 18432 at T=4
# split2), both inside sVec's 4096 B region -- and were dead. Here they are real 2048 B regions appended past
# sU. They are 4096 B of the +5504 B delta versus T=4's split2 arena; the other
# 1408 B is the T 4->5 growth of sK/sD (+512 each), sU (+256), sL/sR (+36 each),
# sBeta/sSlot/sToken (+4 each) and 44 B of realignment. Rows 5..15 are never
# written.
# sVec still has 16 columns; only 0..4 (k) and 8..12 (q) are ever written.

# ===========================================================================
# Work decomposition and lane roles  (:100-178 -- identical to T=4)
# ===========================================================================
work       = cta_id.x
value_tile = work % VALUE_SPLIT              # `work & (VALUE_SPLIT-1)` in source
hv         = work // VALUE_SPLIT
n          = cta_id.y
query_head = hv // HEAD_RATIO                # the body's only div.s32
tid        = thread_id.x
warp       = tid // 32                       # == token index in phases A and C'
lane       = tid % 32
lane_quad  = lane % 4
frag_row   = lane // 4                       # the m16n8k16 M index
quad_base  = lane - lane_quad
group      = tid // 16
lane_group = tid % 16
k_start    = lane_group * 8
elem_start = lane * 4
tile_row_base  = value_tile * ROWS_PER_CTA
owned_row_base = group * ROWS_PER_GROUP

# Register declarations (:161-177), byte-identical to the T=2/T=3/T=4 block.
r_q, r_k, r_d = (reg_tile(f32, [4]) for _ in range(3))
ratio_scan    = reg_tile(f32, [4])            # DEAD at T=5: the butterfly path is gone
r_state       = reg_tile(f32, [8])
hist          = reg_tile(f32, [ROWS_PER_GROUP, 8])   # [8,8] / [2,8] at split8
state_pack    = reg_tile(u32, [4])
state_frag, vec_frag = reg_tile(u32, [4]), reg_tile(u32, [4])
mma_acc, mma_acc_c   = reg_tile(f32, [4]), reg_tile(f32, [4])
ha_lo, ha_hi, hc_lo, hc_hi = (reg_tile(f32, [TOKENS]) for _ in range(4))
u_lo, u_hi    = reg_tile(f32, [TOKENS]), reg_tile(f32, [TOKENS])
# `hist` is the one split-dependent tile: the source declares `float hist[64]` at
# splits 1/2/4 and `float hist[16]` at split8 (S8:166). Warps that skip phases B
# and H still allocate it -- the port reproduces that rather than shrinking it.

token_base = load_i32(cu_seqlens[n])
seq_len    = load_i32(cu_seqlens[n+1]) - token_base
# instruction_selection: ld.global.nc.b32; extent: scalar (x2)
# cu_seqlens must step by exactly 5 per row; the body does not bound-check it.
# Padding is signalled ONLY by ssm_state_indices < 0.

# `warp` is make_warp_uniform(tid/32) at :101 -- one shfl.sync.idx.b32 whose
# result IS `warp`. Semantically the identity; the port spells it `tid // 32`.

# ===========================================================================
# Phase A: token preprocess, warp <-> token  (:180-290)
# Identical to the ported T=4 phase A except for the two constants noted.
# ===========================================================================
if warp < TOKENS:
    token     = warp
    active    = token < seq_len
    token_pos = token_base + token if active else 0

    copy_g2r(q[(token_pos*H + query_head)*128 + elem_start : +4], r_q)
    copy_g2r(k[(token_pos*H + query_head)*128 + elem_start : +4], r_k)
    copy_g2r(g[token_pos*GATE_TOKEN_STRIDE + hv*128 + elem_start : +4], r_d)
    # instruction_selection: ld.global.nc.v2.b32 (one per tensor) + widen;
    #                        extent: vector
    # Four contiguous bf16 = 8 bytes = ONE v2.b32 per tensor; each 32-bit word is
    # then split into two f32 by shl.b32 / and.b32, exactly as at T=2/T=4.

    q_sum = dot(r_q, r_q); k_sum = dot(r_k, r_k)
    # instruction_selection: fma.rn.ftz.f32; extent: two 4-term chains (8 per
    #                        body), each seeded with C = 0f00000000
    # There is no `mul` at these sites, and the two chains are emitted
    # INTERLEAVED (q[0], k[0], q[1], k[1], ...) because the source accumulates
    # both in one fused loop (:241-244).
    for off in (16, 8, 4, 2, 1):
        q_sum = add(q_sum, bfly(q_sum, off, 31, 0xFFFFFFFF))
    for off in (16, 8, 4, 2, 1):
        k_sum = add(k_sum, bfly(k_sum, off, 31, 0xFFFFFFFF))
    # instruction_selection: shfl.sync.bfly.b32 + add.ftz.f32;
    #                        extent: 5 rounds x 2 sequential reductions
    #                        = 10 shfl (+10 add) per body
    # The two reductions are SEQUENTIAL, not interleaved: `_warp_reduce_0`
    # completes all five rounds (:245-249) before `_warp_reduce_1` starts
    # (:250-254), and the PTX emits five shfl at .loc 248 then five at .loc 253.
    # This is the opposite of the dot chains directly above, which do interleave
    # -- the distinction comes from the source's loop structure, not from a
    # scheduling preference, so the port has to reproduce both.
    q_norm = mul(rsqrt(add(q_sum, 1e-6)), scale)
    k_norm = rsqrt(add(k_sum, 1e-6))
    # instruction_selection: rsqrt.approx.ftz.f32; extent: scalar (x2)
    # `scale` folds into q only. FTZ is not incidental: the export is built with
    # -use_fast_math, so every f32 op in this body is the .ftz line.

    for i in range(4):
        r_q[i] = mul(r_q[i], q_norm)
        r_k[i] = mul(r_k[i], k_norm)
        r_d[i] = exp(r_d[i])
        # instruction_selection: ex2.approx.ftz.f32; extent: scalar (x4)
        # GATE_KIND 0: g arrives in log space and this is the only transcendental
        # applied to it. (The T=3 sibling's three-transcendental chain has no
        # analogue here -- A_log/dt_bias are dummies.)

    copy_r2s(r_k, sK[token, elem_start : +4])
    copy_r2s(r_d, sD[token, elem_start : +4])
    # instruction_selection: st.shared.v4.b32; extent: vector (one each)
    # Four contiguous f32 per lane, so nvcc emits ONE 16-byte store per region --
    # 2 st.shared.v4.b32 in phase A, not 8 scalar ones. The port has to vectorize
    # these from the start; issuing scalars here is the exact mistake the T=2
    # port made and paid for.

    if lane == 0:
        raw_slot = load_i32(ssm_state_indices[n*5 + token])     # stride 5
        copy_r2s(raw_slot if active else -1, sSlot[token])      # source order:
        copy_r2s(token_pos, sToken[token])                     # sSlot, sToken,
        copy_r2s(beta[(token_pos*HV) + hv], sBeta[token])      # then sBeta
        # instruction_selection: ld.global.nc.b32 (ssm_state_indices)
        #                        + ld.global.nc.b16 + cvt.f32.bf16 (beta)
        #                        + st.shared.b32 x3; extent: scalar
    if warp == 0 and lane == 0:
        accepted = clamp(load_i32(num_accepted_tokens[n]) - 1, 0, 4)   # [0,4] at T=5
        init_slot = load_i32(ssm_state_indices[n*5 + accepted])
        # instruction_selection: ld.global.nc.b32 x2 + max.s32 + min.s32;
        #                        extent: scalar
        copy_r2s(0 if init_slot < 0 else init_slot, sInit[0])
        # instruction_selection: st.shared.b32; extent: scalar
        # sSlot/sToken/sBeta/sInit are the body's only scalar shared stores in
        # phase A -- 4 of them, all on lane 0.
        # Both clamp arms are exercised upstream and locally (nat 0/1 agree, 5/12
        # agree, 1 and 5 differ).

cta_sync()
# instruction_selection: bar.sync 0; extent: CTA
# Orders sInit -> the gather base, and sK/sD/sBeta/sSlot/sToken -> their readers
# in C', the gram block, E, F and H.

# ===========================================================================
# Phase C': gate prefix, sVec publish, sGramA publish  (:293-339)
# CHANGED versus T=4: q columns move to 8+token, and sGramA* is new.
# ===========================================================================
if warp < TOKENS:
    token = warp
    for i in range(4):
        k_idx  = elem_start + i
        prefix = 1.0
        for p in range(TOKENS):
            if token >= p:
                prefix = mul(prefix, copy_s2r(sD[p, k_idx]))
        # instruction_selection: ld.shared.b32 + mul.ftz.f32; extent: loop (<=5)
        # `prefix` = product of the gates of tokens 0..token at this key. These
        # stay SCALAR (20 ld.shared.b32 in the phase): the walk is across tokens
        # at a fixed key, so consecutive iterations are 512 B apart.

        copy_r2s(cast(bf16, mul(prefix, r_k[i])), sVec[swz(k_idx*32 + token*2)])
        # instruction_selection: cvt.rn.bf16.f32 + st.shared.b16; extent: scalar
        copy_r2s(cast(bf16, mul(prefix, r_q[i])), sVec[swz(k_idx*32 + (8+token)*2)])
        # instruction_selection: cvt.rn.bf16.f32 + st.shared.b16; extent: scalar
        # The q column is 8+token, NOT 4+token. The generator emits
        # `c_col = 4 + token;` and then immediately overrides it with
        # `{ c_col = 8 + token; }` (:311-314) -- a dead store the port drops.
        # Columns 5,6,7 and 13,14,15 are never written by anyone.

        half = sGramA0 if k_idx < 64 else sGramA1
        copy_r2s(cast(bf16, div(r_k[i], prefix)),
                 half[swz(token*128 + (k_idx % 64)*2)])
        # instruction_selection: div.approx.ftz.f32 + cvt.rn.bf16.f32
        #                        + st.shared.b16; extent: scalar
        # 8 div.approx.ftz.f32 per body (4 elements x the two branches), read off
        # the exported PTX -- NOT rcp+mul and not the full-range sequence. This is
        # the gate-DEFLATED key; sVec holds the gate-INFLATED one. Their product
        # is exactly T=4's ratio_scan factor prefix_t/prefix_s, but with both
        # operands rounded to bf16 and with a division T<=4 never performed. It is
        # the numerically delicate step of the body when gates are small.

# ===========================================================================
# Phase C'': the coefficient Gram block  (:340-408)  -- NEW AT T=5
# One warp replaces T=4's entire butterfly path with two tensor-core products.
# ===========================================================================
if warp < TOKENS:
    if VALUE_SPLIT == 4:
        named_bar_sync(1, 160)         # S4:340-344 -- all five token warps block
        # instruction_selection: barrier.sync 1, 160; extent: 5 warps
        is_gram = (warp == GRAM_WARP)
    else:
        is_gram = (warp == GRAM_WARP)
        if not is_gram:
            named_bar_arrive(1, 160)   # :406 -- publish and go
            # instruction_selection: barrier.arrive 1, 160; extent: 5 warps

    if is_gram:
        # Declared inside the gram warp's block, as the source does (:345-348).
        gram_a_frag, gram_b_frag = reg_tile(u32, [4]), reg_tile(u32, [4])
        gram_k_acc, gram_q_acc   = reg_tile(f32, [4]), reg_tile(f32, [4])

        if VALUE_SPLIT != 4:
            named_bar_sync(1, 160)     # :343 -- wait for the other four
            # instruction_selection: barrier.sync 1, 160; extent: 5 warps

        # A = sGramA (rows = SOURCE tokens), B = sVec (cols = TARGET tokens).
        for gram_half in range(2):                    # k 0..63 / 64..127
            for gram_k in range(0, 64, 16):
                a_addr = swz((lane % 16 * 64 + gram_k + lane // 16 * 8) * 2)
                b_addr = swz(((gram_half*64 + gram_k + lane % 16) * 16
                              + lane // 16 * 8) * 2)
                ldm(sGramA0 if gram_half == 0 else sGramA1, a_addr, gram_a_frag)
                # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.shared.b16;
                #                        extent: tile (8 per body)
                ldm(sVec, b_addr, gram_b_frag, trans=True)
                # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16;
                #                        extent: tile (8 per body)
                # b_addr is the SAME formula phase D uses for its sVec operand
                # (:360-361 vs :454-455); frags [0],[1] cover columns 0..7 (the k
                # side), frags [2],[3] cover 8..15 (the q side).

                init = (gram_half == 0 and gram_k == 0)
                mma(gram_k_acc, gram_a_frag, gram_b_frag[0:2], init=init)
                mma(gram_q_acc, gram_a_frag, gram_b_frag[2:4], init=init)
                # instruction_selection: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32;
                #                        extent: loop (2 x 4 x 2 = 16 issues)
                # The first step uses the zero-C form, the other seven accumulate.

        # Result mapping -- the transpose of T=4's roles. At T=4 the warp index was
        # the TARGET token and it looped over sources; here the MMA M index is the
        # SOURCE token and the N index (the sVec column) is the TARGET.
        source = frag_row                 # M index, 0..7, masked to < 5
        target0, target1 = lane_quad*2, lane_quad*2 + 1     # N indices
        if source < TOKENS:
            beta_source = copy_s2r(sBeta[source])
            if source <  target0 and target0 < TOKENS:
                copy_r2s(mul(beta_source, gram_k_acc[0]), sL[target0, source])
            if source <  target1 and target1 < TOKENS:
                copy_r2s(mul(beta_source, gram_k_acc[1]), sL[target1, source])
            if source <= target0 and target0 < TOKENS:
                copy_r2s(mul(beta_source, gram_q_acc[0]), sR[target0, source])
            if source <= target1 and target1 < TOKENS:
                copy_r2s(mul(beta_source, gram_q_acc[1]), sR[target1, source])
            # instruction_selection: mul.ftz.f32 + st.shared.b32; extent: scalar (x4)
            # Only acc[0],[1] are ever read: acc[2],[3] hold M rows 8..15, which
            # are past the five real tokens. sL gets the strict lower triangle
            # (10 live entries), sR the lower triangle including the diagonal (15).
            # lane_quad == 3 writes nothing; lane_quad == 2 writes only target0=4.

# ===========================================================================
# Phase B: gather the initial checkpoint  (:410-446)
# Identical to T=4 except that split8 owns 2 rows per group instead of 8.
# ===========================================================================
if group < ROW_GROUPS:
    init_slot = copy_s2r(sInit[0])
    for row_local in range(ROWS_PER_GROUP):
        value_row_local = owned_row_base + row_local       # CTA-local, indexes sState
        value_row       = tile_row_base + value_row_local  # global, indexes `state`
        copy_g2r(state[init_slot*STATE_SLOT_STRIDE + hv*128*128
                       + value_row*128 + k_start : +8], state_pack)
        # instruction_selection: ld.global.v4.b32; extent: vector
        #                        (8 per body; 2 at split8)
        for i in range(8):
            hist[row_local*8 + i] = widen(state_pack)[i]
        # instruction_selection: shl.b32 / and.b32; extent: scalar
        half, k_off = (sState0, k_start) if lane_group < 8 else (sState1, k_start - 64)
        copy_r2s(state_pack, half[swz(value_row_local*128 + k_off*2)])
        # instruction_selection: st.shared.v4.b32; extent: vector
        #                        (16 per body; 4 at split8)
        # This is an if/ELSE selecting the destination half (:438-442), not a
        # `lane_group < 8` guard: lanes 8..15 stage keys 64..127 into sState1.
        # Treating it as a guard would leave sState1 unwritten and phase D's
        # `state_half == 1` ldmatrix would read uninitialized shared memory.

cta_sync()
# instruction_selection: bar.sync 0; extent: CTA
# Orders sState (staged by every group) and sL/sR (written by the gram warp)
# against their readers in D, E and F -- and, at splits 1, 2 and 8, sVec.
# That last edge is easy to get wrong: `barrier.arrive` is non-blocking, so an
# arriving warp releases its own writes but ACQUIRES NOTHING. Every MMA warp
# except the gram warp therefore reaches phase D having synchronized with the
# other token warps only here. At split4 -- the one body where all five token
# warps execute the blocking `barrier.sync` -- sVec is already visible before
# this barrier, which then carries only sState and sL/sR. Split1 adds a second
# sub-case: warps 5..7 are outside the `warp_0 < 5` guard entirely and never
# touch the named barrier at all.

# ===========================================================================
# Phase D: the main MMA chain  (:448-490)  -- TWO issues per step at T=5
# ===========================================================================
if warp < MMA_WARPS:
    for state_half in range(2):
        for mma_k in range(0, 64, 16):
            vec_addr   = swz(((state_half*64 + mma_k + lane % 16)*16
                              + lane // 16 * 8) * 2)
            state_addr = swz(((warp*16 + lane % 16)*64
                              + (mma_k + lane // 16 * 8)) * 2)
            ldm(sVec, vec_addr, vec_frag, trans=True)
            # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16;
            #                        extent: tile (8 per body)
            ldm(sState0 if state_half == 0 else sState1, state_addr, state_frag)
            # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.shared.b16;
            #                        extent: tile (8 per body)

            init = (state_half == 0 and mma_k == 0)
            mma(mma_acc,   state_frag, vec_frag[0:2], init=init)   # columns 0..7
            mma(mma_acc_c, state_frag, vec_frag[2:4], init=init)   # columns 8..15
            # instruction_selection: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32;
            #                        extent: loop (2 x 4 x 2 = 16 issues)
            # `mma_acc_c` and `vec_frag[2],[3]` were DEAD at T<=4, where all four
            # live columns fit one n=8 tile. At T=5 the k side needs columns 0..4
            # and the q side 8..12, so the body pays a second issue per step.

# ===========================================================================
# Phase E: quad broadcasts and the WY solve  (:491-560)
# ===========================================================================
if warp < MMA_WARPS:
    # m16n8k16 puts columns 2q, 2q+1 of row `frag_row` in lane `quad_base+q`'s
    # acc[0],[1], and rows +8 in acc[2],[3]. Token t therefore lives at
    # (lane quad_base + t//2, acc index t%2).
    # Emission order is the source's (:491-531), not a tidy nest: the T<=4 block
    # first, then everything T=5 made live.
    for t in range(4):
        ha_lo[t] = bcast(mma_acc[t % 2],     quad_base + t // 2, 31, 0xFFFFFFFF)
    for t in range(4):
        ha_hi[t] = bcast(mma_acc[2 + t % 2], quad_base + t // 2, 31, 0xFFFFFFFF)
    ha_lo[4] = bcast(mma_acc[0], quad_base + 2, 31, 0xFFFFFFFF)
    ha_hi[4] = bcast(mma_acc[2], quad_base + 2, 31, 0xFFFFFFFF)
    for t in range(TOKENS):
        hc_lo[t] = bcast(mma_acc_c[t % 2],     quad_base + t // 2, 31, 0xFFFFFFFF)
    for t in range(TOKENS):
        hc_hi[t] = bcast(mma_acc_c[2 + t % 2], quad_base + t // 2, 31, 0xFFFFFFFF)
    # instruction_selection: shfl.sync.idx.b32; extent: loop (20 per body)
    # ha_* carries the k side (used by the solve), hc_* the q side (used by the
    # output rewrite). At T<=4 hc_* was dead and ha_* stopped at index 3.

    if lane_quad == 2:
        for t in range(TOKENS):                       # depth 5
            base = (copy_s2r(sToken[t]) * HV + hv) * 128
            # instruction_selection: ld.shared.v2.b32 + ld.shared.v4.b32;
            #                        extent: vector (the five sToken words at once,
            #                        over-reading sInit[0] -- in region, never used)
            solved_lo = sub(cast(f32, v[base + tile_row_base + warp*16 + frag_row]),
                            ha_lo[t])
            solved_hi = sub(cast(f32, v[base + tile_row_base + warp*16 + frag_row + 8]),
                            ha_hi[t])
            # instruction_selection: ld.global.nc.b16 + sub.ftz.f32; extent: scalar
            for p in range(TOKENS):
                if p < t:
                    coef = copy_s2r(sL[t, p])
                    solved_lo = sub(solved_lo, mul(coef, u_lo[p]))
                    solved_hi = sub(solved_hi, mul(coef, u_hi[p]))
            # instruction_selection: 5 ld.shared.b32 + 1 ld.shared.v2.b32
            #                        + 1 ld.shared.v4.b32, then mul.ftz.f32 +
            #                        sub.ftz.f32; extent: loop (10 live (t,p) pairs)
            # The ten live sL entries are read as a mixed scalar/v2/v4 set; the v4
            # over-reads sL[24], which nobody writes -- in region, never used.
            u_lo[t], u_hi[t] = solved_lo, solved_hi

    for t in range(TOKENS):
        u_lo[t] = bcast(u_lo[t], quad_base + 2, 31, 0xFFFFFFFF)
        u_hi[t] = bcast(u_hi[t], quad_base + 2, 31, 0xFFFFFFFF)
    # instruction_selection: shfl.sync.idx.b32; extent: loop (10 per body)
    # The solve runs on lane_quad == 2 but lane_quad == 3 also writes output, so
    # the residuals have to cross the quad -- live since T=4.

# ===========================================================================
# Phase F: output rewrite  (:562-640)  -- REWRITTEN AGAIN AT T=5
# ===========================================================================
if warp < MMA_WARPS and lane_quad >= 2:
    token0 = (lane_quad - 2) * 2      # 0 or 2
    token1 = token0 + 1               # 1 or 3
    row_lo = tile_row_base + warp*16 + frag_row
    row_hi = row_lo + 8

    # The bases come from hc_* (the q-side MMA), not from mma_acc. The source
    # first assigns mma_acc[0..3] and then unconditionally overwrites with
    # hc_lo/hc_hi (:567-582) -- another dead generator store the port drops.
    out0_lo, out1_lo = hc_lo[token0], hc_lo[token1]
    out0_hi, out1_hi = hc_hi[token0], hc_hi[token1]

    for s in range(TOKENS):
        coef0 = copy_s2r(sR[token0, s]) if token0 >= s else 0.0
        coef1 = copy_s2r(sR[token1, s]) if token1 >= s else 0.0
        # instruction_selection: ld.shared.b32; extent: scalar (masked)
        out0_lo = fma(coef0, u_lo[s], out0_lo)
        out1_lo = fma(coef1, u_lo[s], out1_lo)
        out0_hi = fma(coef0, u_hi[s], out0_hi)
        out1_hi = fma(coef1, u_hi[s], out1_hi)
        # instruction_selection: fma.rn.ftz.f32; extent: loop (5 x 4 = 20)

    for (token, lo, hi) in ((token0, out0_lo, out0_hi), (token1, out1_lo, out1_hi)):
        slot = copy_s2r(sSlot[token])
        base = (copy_s2r(sToken[token]) * HV + hv) * 128
        copy_r2g(cast(bf16, lo) if slot >= 0 else 0.0, out[base + row_lo])
        copy_r2g(cast(bf16, hi) if slot >= 0 else 0.0, out[base + row_hi])
        # instruction_selection: cvt.rn.bf16.f32 + st.global.b16; extent: scalar
        # Scalar stores, not vectorized: 10 static st.global.b16 in the body.
        # A padded row (slot < 0) writes an EXPLICIT zero -- it is not skipped.

    if lane_quad == 2:                                  # the fifth-token tail
        out4_lo, out4_hi = hc_lo[4], hc_hi[4]
        for s in range(TOKENS):
            coef4 = copy_s2r(sR[4, s])                  # no mask: 4 >= every s
            out4_lo = fma(coef4, u_lo[s], out4_lo)
            out4_hi = fma(coef4, u_hi[s], out4_hi)
            # instruction_selection: ld.shared.v4.b32 (sR[20..23]) + ld.shared.b32
            #                        (sR[24]) + fma.rn.ftz.f32; extent: 10 fma
            # Row 4 of sR is contiguous AND unmasked -- the only sR read in the
            # body that vectorizes. The masked pair loop above cannot.
        slot4 = copy_s2r(sSlot[4])
        base4 = (copy_s2r(sToken[4]) * HV + hv) * 128
        copy_r2g(cast(bf16, out4_lo) if slot4 >= 0 else 0.0, out[base4 + row_lo])
        copy_r2g(cast(bf16, out4_hi) if slot4 >= 0 else 0.0, out[base4 + row_hi])
        # instruction_selection: st.global.b16; extent: scalar

    # NO out-of-region read at T=5. lane_quad in {2,3} gives token0 in {0,2} and
    # token1 in {1,3}, so every sSlot/sToken index is < 5 and every sR index is
    # <= 19; the tail's sR[4,s] tops out at 24 < 25. The source's `token0 < 5` /
    # `token1 < 5` guards are statically true. This is the opposite of the T=3
    # sibling, where lane_quad == 3 computed token1 = 3 and read past a 3-entry
    # region before masking -- that port had to predicate those reads; this one
    # must not, because predicating here would only add instructions.

# ===========================================================================
# Phase G: publish sU  (:641-654)
# ===========================================================================
if warp < MMA_WARPS:
    if lane_quad == 2:
        for t in range(TOKENS):
            copy_r2s(u_lo[t], sU[t, warp*16 + frag_row])
            copy_r2s(u_hi[t], sU[t, warp*16 + frag_row + 8])
            # instruction_selection: st.shared.b32; extent: scalar (10 per body)
    if VALUE_SPLIT != 8:
        warp_sync()         # :651-653, INSIDE the guard
        # instruction_selection: bar.warp.sync -1; extent: warp
        # Sufficient because warp w produces rows w*16 .. +15 and consumes exactly
        # the same rows as groups 2w, 2w+1.

if VALUE_SPLIT == 8:
    cta_sync()              # S8:652-654, at OUTER scope -- all 160 threads
    # instruction_selection: bar.sync 0; extent: CTA
    # NOTE THE NESTING, it is not cosmetic: at split8 the `warp_0 < 1` guard
    # CLOSES at S8:651 and the barrier sits outside it, because the sU producer
    # is warp 0 alone while the consumers are groups 0..7 = warps 0..3. Keeping
    # it inside the guard -- the shape the other three splits use for their
    # __syncwarp -- would have one warp of five arrive at a CTA barrier and the
    # kernel would hang. PTX: 3 bar.sync and 0 bar.warp.sync at split8, 2 and 1
    # elsewhere.

# ===========================================================================
# Phase H: FP32 recurrence and checkpoints  (:768-804)
# Identical to T=4 except for the token count and the split's rows per group.
# ===========================================================================
if group < ROW_GROUPS:
    for t in range(TOKENS):
        slot_t = copy_s2r(sSlot[t])
        for row_local in range(ROWS_PER_GROUP):
            value_row_local = owned_row_base + row_local
            beta_t = copy_s2r(sBeta[t])
            update = mul(copy_s2r(sU[t, value_row_local]), beta_t)
            for i in range(8):
                k_idx = k_start + i
                hist[row_local*8 + i] = fma(hist[row_local*8 + i],
                                            copy_s2r(sD[t, k_idx]),
                                            mul(update, copy_s2r(sK[t, k_idx])))
                r_state[i] = hist[row_local*8 + i]
            # instruction_selection: ld.shared.v4.b32 + mul.ftz.f32 + fma.rn.ftz.f32;
            #                        extent: loop (5 x ROWS_PER_GROUP x 8)
            # The sD and sK slices are invariant across `row_local`, so they are
            # hoisted OUT of the row loop: 4 v4 loads per token (2 sD + 2 sK)
            # dynamically, 36 statically because the token block is replicated
            # 2*TOKENS-1 = 9 times by the `slot_t >= 0` store split -- the same
            # nvcc tail duplication the T=4 sibling documents. The decisive
            # evidence is that split8 carries the SAME 36 at .loc 782 while its
            # row count drops 8 -> 2; a per-(token,row) issue would be 160 and 40.
            # The contraction order is `hist*sD + (sU*beta)*sK`, unchanged since T=2.
            if slot_t >= 0:
                copy_r2g(pack_bf16x2(r_state),
                         state[slot_t*STATE_SLOT_STRIDE + hv*128*128
                               + (tile_row_base + value_row_local)*128 + k_start])
                # instruction_selection: cvt.rn.bf16x2.f32 x4 + st.global.v4.b32;
                #                        extent: vector (40 stores; 10 at split8)
                # A padded token advances `hist` but stores nothing.
```

## Instruction-selection summary

Counted from the exported `--source-in-ptx` bodies with `ptx_census.py`
(comment-, brace- and predicate-aware) and attributed to phases with
`ptx_phase_census.py` via `.loc`, restricted to the body file — `__shfl_sync` and
the bf16 conversions carry `sm_30_intrinsics.hpp` / `cuda_bf16.hpp` line numbers
and are only attributable through their `inlined_at` field.

| form | s1 | s2 | s4 | s8 | where |
| --- | --- | --- | --- | --- | --- |
| `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | 32 | 32 | 32 | 32 | 16 gram + 16 main |
| `ldmatrix...x4.shared.b16` | 16 | 16 | 16 | 16 | 8 gram (A) + 8 main (sState) |
| `ldmatrix...x4.trans.shared.b16` | 16 | 16 | 16 | 16 | 8 gram (sVec) + 8 main (sVec) |
| `shfl.sync.idx.b32` | 31 | 31 | 31 | 31 | 1 warp-uniform + 20 acc + 10 u |
| `shfl.sync.bfly.b32` | 10 | 10 | 10 | 10 | phase A's two L2 reductions |
| `div.approx.ftz.f32` | 8 | 8 | 8 | 8 | `k / prefix`, 4 elements x 2 halves |
| `ex2.approx.ftz.f32` | 4 | 4 | 4 | 4 | the gate, phase A |
| `fma.rn.ftz.f32` | 614 | 614 | 614 | **182** | 576 in H, 30 in F, 8 in A's dot chains |
| `mul.ftz.f32` | 709 | 709 | 709 | **223** | 648 in H, 24 in C′'s prefix walk, 20 in E's solve, 13 in A, 4 in the gram result map |
| `add.ftz.f32` / `sub.ftz.f32` | 12 / 30 | 12 / 30 | 12 / 30 | 12 / 30 | 10 reduction adds + 2 rsqrt epsilons in A / E's solve |
| `rsqrt.approx.ftz.f32` | 2 | 2 | 2 | 2 | the q and k L2 norms |
| `bar.sync 0` | 2 | 2 | 2 | **3** | split8 replaces the warp barrier |
| `bar.warp.sync` | 1 | 1 | 1 | **0** | ditto |
| `barrier.sync/arrive 1, 160` | 2 | 2 | **1** | 2 | split4's arrive arm is dead |
| `st.shared.b16` | 16 | 16 | 16 | 16 | 4 sVec-k + 4 sVec-q + 8 sGramA |
| `ld.global.nc.v2.b32` | 3 | 3 | 3 | 3 | phase A's q, k and g slices |
| `ld.global.nc.b32` | 5 | 5 | 5 | 5 | 2 cu_seqlens, 2 ssm_state_indices, 1 num_accepted_tokens |
| `ld.global.nc.b16` | 11 | 11 | 11 | 11 | 1 beta in A + 10 `v` in the phase-E solve |
| `cvt.f32.bf16` | 11 | 11 | 11 | 11 | the same 11 bf16 scalars widened |
| `st.shared.v4.b32` | 18 | 18 | 18 | **6** | 2 phase-A sK/sD + phase-B staging (16; 4 at split8) |
| `st.shared.b32` | 18 | 18 | 18 | 18 | 4 phase-A scalars + 4 gram sL/sR + 10 sU |
| `ld.shared.b32` | 72 | 72 | 72 | 72 | 20 in C′ (gate prefix); 17 in F (8 sR + 9 sSlot/sToken); **28** in H (sSlot, sBeta, sU run ends — one `sBeta[4]` load carries no body-line attribution); 5 in E; 1 sInit at `:292`; 1 sBeta in the gram block |
| `ld.shared.v4.b32` | 48 | 48 | 48 | **39** | 45 in H (36 sD/sK hoisted per token + 9 sU, 4 rows each); 2 in E; 1 in F |
| `ld.shared.v2.b32` | 11 | 11 | 11 | **2** | 9 in H (sU, 2 rows each — sU's base is 4 mod 16, so an 8-row run peels b32+v2+v4+b32), 2 in E (sToken, sL) |
| `ld.global.v4.b32` | 8 | 8 | 8 | **2** | phase B, one per owned row |
| `st.global.b16` | 10 | 10 | 10 | 10 | phase F outputs, scalar |
| `st.global.v4.b32` | 40 | 40 | 40 | **10** | phase H checkpoints |
| `cvt.rn.bf16x2.f32` | 160 | 160 | 160 | **40** | phase H packing |

Per-phase totals for split2, out of 3064 body instructions: prologue 173, A 123,
C′ 249, gram 132, B 211, D 93, E 142, F 226, G 19, **H 1624**. The
`BLOCK_CHECKPOINT_MMA` block contributes **0** — it is eliminated entirely, not
merely unexecuted. The remaining 72 are the two barrier regions and the
parameter loads, which carry no body line attribution; the list above is a
breakdown, not a partition. Phase H is 53% of the body, exactly as at T=4 — the
whole gram block is 4%.

The three splits with `ROWS_PER_GROUP = 8` (1, 2, 4) have identical
**key-operation** mixes — every row of the table above — apart from split4's
missing `barrier.arrive`. (They do *not* share a launch geometry: split1 runs 8
warps over 128 rows, splits 2 and 4 run 5 warps over 64 and 32. What makes their
per-thread instruction mix identical is the per-group row count, which is why
split8 — the only body with `ROWS_PER_GROUP = 2` — is the only one that differs.) Their full bodies still differ (2969 / 3064 / 3062
instructions) purely in address arithmetic, because split1 folds `value_tile` to
a constant 0. Split8 differs only where `ROWS_PER_GROUP` does. Nothing in the body is `T`-shaped except the
loop trip counts, which is why one parameterized TIRx body covers all four.

## Was dead at T≤4, live at T=5

| element | T≤4 | T=5 |
| --- | --- | --- |
| `sVec` columns | 0..3 k, 4..7 q | **0..4 k, 8..12 q** (5,6,7,13,14,15 never written) |
| `vec_frag[2],[3]` | dead | **live** — B operand of both the q-side gram MMA and the main `mma_acc_c` |
| `mma_acc_c[4]` | dead | **live** — the second main MMA |
| `hc_lo/hc_hi[0..4]` | dead | **live** — the phase-F output bases |
| `ha_lo/ha_hi[4]` | dead | **live** — token 4's solve input |
| `sGramA0`, `sGramA1` | dead aliases of `sVec` | **live**, separate 2048 B regions |
| `sL` / `sR` live entries | 6 / 10 | **10 / 15** |
| solve depth | 4 | **5** |
| named barrier | none | **`barrier.sync 1, 160` / `barrier.arrive 1, 160`** |
| `ratio_scan[4]` | live (the butterfly path) | **dead** — the path is gone |

Still dead at T=5: `sVec` columns 5,6,7,13,14,15; `sGramA*` rows 5..15;
`gram_*_acc[2],[3]`; `mma_acc[1]` on `lane_quad == 2` and all of `mma_acc_c` on
`lane_quad == 3`; the entire `BLOCK_CHECKPOINT_MMA` block; and — as in every
sibling — the **three** `elem_start < 128` guards (`:194`, `:260`, `:294`), statically true
at `HEAD_DIM = 128` because `elem_start = lane*4 <= 124`, together with the
`r_q/r_k/r_d = 0.0f` zero-init the first one guards (`:188-193`) and the
`gate_a = 1.0f` initializer at `:259` (GATE_KIND 0 never reads it). No
instruction in any export carries `.loc 188-193`, `:259`, `:194`, `:260` or
`:294`; the port drops them. The neighbouring `qk_base`/`gate_base` at `:186-187`
are **live** address arithmetic (3 instructions at `.loc 187`), folded into the
phase-A copies rather than dropped.

**Uninitialized-but-in-region reads are safe and must not be predicated.** Both
`ldmatrix` sites read a full 16-row / 16-column tile, so they touch `sGramA*`
rows 5..15 and `sVec` columns 5,6,7,13,14,15, which nobody writes. Those land in
MMA output rows 8..15 (`acc[2],[3]`, never read) and in accumulator columns that
no broadcast ever selects — `m16n8k16` columns are independent, so no garbage,
NaN or Inf can reach a live value. Predicating them would change the instruction
mix for no numerical gain.
