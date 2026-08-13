<!--
This file is a design sketch for a TIRx port of code from FlashInfer
(https://github.com/flashinfer-ai/flashinfer @ f2e04400),
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# FlashKDA cake T=6 coefficient-gram decode: coarse execution sketch

**Non-executable.** This is the semantic execution skeleton, not runnable code.
The source of truth for the implementation is
`tirx_kernels/flashinfer/kda/flashkda_decode_t6_gram.py`.

Transcribed from four frozen generated exports, all with symbol
`kernel_flashinfer_recurrent_kda_wy_vtile_short` (flashinfer-ai/flashinfer @
`f2e04400`), 816 body lines each:

- `csrc/kda/flashkda_decode_d128_t6_precomputed_gram_split1.cu` — `S1:NNN`
- `csrc/kda/flashkda_decode_d128_t6_precomputed_gram_split2.cu` — `S2:NNN`, and
  the default for a bare `:NNN`
- `csrc/kda/flashkda_decode_d128_t6_precomputed_gram_split4.cu` — `S4:NNN`
- `csrc/kda/flashkda_decode_d128_t6_precomputed_gram_split8.cu` — `S8:NNN`

T=6 is the last variant the cake family exports. It shares one arm of the
sm100a value-split policy with T=5 (`recurrent_kda.py:1181-1191` is a single
`if num_tokens in (5, 6):`), so the split is shape-dependent, all four exports
are production-reachable, and the split is a constexpr here as it was there.

## What changed relative to the ported T=5 body

Line numbering is **1:1 identical to T=5 through line 531**; the first
insertion is the +6-line block at `:532-537`. Phase by phase:

| phase | T=5 → T=6 |
| --- | --- |
| prologue | identical but for six array sizes `[5]→[6]` (`:172-177`) |
| A preprocess | identical modulo T: `warp_0 < 6`, `ssm_idx[n*6 + …]`, nat clamp **`[0,5]`** |
| C′ publish | identical modulo T: prefix loop 6, sVec k cols **0..5**, q cols **8..13** |
| C″ gram | identical modulo the barrier count (**192**) and `GRAM_WARP` (**5** at splits 4/8) |
| B gather | **byte-identical** — no T dependence |
| D MMA | **byte-identical** — no T dependence |
| E broadcasts + solve | **+6 lines**: token 5's broadcast is *elided* (below); solve depth 6; u redistribution 6 |
| F outputs | **changed**: a **second** tail, and a new coefficient clamp (below) |
| G publish `sU` | identical modulo T (12 stores) |
| H recurrence | identical modulo T (6 tokens) |
| the arena | **static → dynamic**: split1 needs 50560 B (below) |

So four deltas carry the whole port. Everything else is the ported T=5
skeleton re-parameterized by `T`.

**Fixed specializations.**

| knob | split1 | split2 | split4 | split8 |
| --- | --- | --- | --- | --- |
| `TOKENS` | 6 | 6 | 6 | 6 |
| `GATE_KIND` | 0 (precomputed log-gate) | 0 | 0 | 0 |
| `LAUNCH_THREADS` | **256** (8 warps) | **192** (6 warps) | 192 | 192 |
| `ROWS_PER_CTA` | 128 | 64 | 32 | 16 |
| `ROWS_PER_GROUP` | 8 | 8 | 8 | **2** |
| `ROW_GROUPS` (B, H) | 16 | 8 | 4 | 8 |
| `MMA_WARPS` (D–G) | 8 | 4 | 2 | **1** |
| `GRAM_WARP` | 0 | 0 | **5** | **5** |
| gram barrier shape | sync/arrive | sync/arrive | **all six sync** | sync/arrive |
| `sU` publish barrier | `__syncwarp` | `__syncwarp` | `__syncwarp` | **`__syncthreads`, outer scope** |
| `SMEM_TOTAL` | **50560** | 32640 | 23680 | 19200 |
| grid | `(HV·1, N)` | `(HV·2, N)` | `(HV·4, N)` | `(HV·8, N)` |

`HEAD_DIM = 128`, `DIRECT_PREFIX_CHECKPOINT = 0`, `BLOCK_CHECKPOINT_MMA = 0`,
`NUM_MAIN_STAGES = 1`, `coefficient_gram = true`. `#define THREADS 256` is
vestigial (`static_assert` at `binding_impl.cuh:40`); the real block size is
`FLASHKDA_DECODE_LAUNCH_THREADS`, and `tokens=6` is what makes it 192 at
splits 2/4/8 (`flash_kda_decode.py:115-136`).

**Out of scope.** T ∈ {1,2,3,4,5} (ported separately); the sm103a T=6 policy
(`recurrent_kda.py:1220-1225`), a separate three-band rule under which split4
is unreachable — the port targets sm100a and replicates only `_current`.

## Delta 1 — the arena is dynamic shared memory

split1's `SMEM_TOTAL` is **50560 B**, past the 49152 B static ceiling every
earlier cake port fit inside. The source has always been dynamic
(`extern __shared__ __align__(1024) char smem_raw[]` at `:104`, sized by
`cudaFuncSetAttribute(MaxDynamicSharedMemorySize, SMEM_TOTAL)`,
`binding_impl.cuh:59-66`), so the port stops approximating it with a static
buffer and allocates through `T.SMEMPool` **for all four splits** — uniform,
and closer to the source than T=5's static arena was.

Two things about this are not optional:

- the kernel must declare `tirx.use_dyn_shared_memory` in
  `tirx.kernel_launch_params`, or the launch reserves **zero** dynamic bytes
  and every arena access faults at runtime — it is not a compile error;
- the byte offsets and the 1024-byte alignment carry over unchanged, because
  the swizzled `ldmatrix` addressing depends on both.

Probed before this sketch was written: all four arenas compile, launch and
round-trip data through the body's swizzled `st.shared.b16` + `ldmatrix`,
while a 4 MB arena is rejected with a precise diagnostic.

## Delta 2 — token 5's quad broadcast is elided

The m16n8k16 accumulator map from T=5 holds for all six tokens: token `t`
lives at lane `quad_base + t//2`, accumulator index `t%2`. For `t = 5` that
source lane is `quad_base + 2` — which is exactly the lane that consumes it,
since both consumers (the WY solve at `:539-559` and phase F's token-5 tail at
`:651-667`) run under `lane_quad == 2`. So the generator does not shuffle:

```
ha_lo[5] = mma_acc[1];    ha_hi[5] = mma_acc[3];      // :533-534
hc_lo[5] = mma_acc_c[1];  hc_hi[5] = mma_acc_c[3];    // :535-536
```

assigned **outside** the `lane_quad == 2` guard (every lane executes them; only
lane_quad 2's values are meaningful). The broadcast count therefore stays at
**20**, not 24 — confirmed in the PTX, which carries 33 `shfl.sync.idx.b32`
total = 1 warp-uniform + 20 broadcasts + 12 u-redistribution.

## Delta 3 — phase F has two tails, and token 4's row needs a clamp

The pair structure is unchanged: `token0 = (lane_quad-2)*2`, `token1 = +1`, so
`lane_quad == 2` writes tokens 0,1 and `lane_quad == 3` writes 2,3, with bases
`hc_*[0..3]` (and the same dead `mma_acc` pre-init the port drops). Tokens 4
and 5 have no partner lane, so **both tails run on `lane_quad == 2`**:

- token 4 (`:629-650`), base `hc_*[4]`, coefficients `sR[24 + s]` — but
  **clamped**: `coef4 = 0.0f` unless `s <= 4`. This is load-bearing.
  `sR[29]` is (target 4, source 5), which the gram block never writes — the
  gram guard is `source <= target`. It is in-region but uninitialized, so the
  clamp is what keeps garbage out of the result. (The source's own dead
  initializer at `:633` reads `sR[24+s]` unconditionally before the clamp
  overwrites it; that read is harmless and the port drops it.)
- token 5 (`:651-667`), base `hc_*[5]`, coefficients `sR[30 + s]` with **no**
  clamp — target 5 accepts all six sources, so `sR[30..35]` are all live.

Writer-lane stores: **8** on `lane_quad == 2` (tokens 0,1,4,5 × lo/hi), 4 on
`lane_quad == 3`.

**No out-of-region read at T=6.** Every `sSlot`/`sToken` index is < 6 and every
`sR` index ≤ 35 < 36. This is not the T=3 hazard class, and unlike T=3 nothing
needs predicating — only the source's own clamp reproducing.

## Delta 4 — six token warps

`barrier.sync 1, 192` / `barrier.arrive 1, 192` in every split, including
split1's 256-thread launch, because the gram region is nested inside
`if (warp_0 < 6)` (`:293` opens, `:409` closes) — split1's warps 6,7 never
reach it. `GRAM_WARP` is `(flag) ? 5 : 0`, i.e. **0** at splits 1/2 and **5**
at splits 4/8. split4 again hoists the wait unconditional across all six token
warps and leaves its arrive arm dead (`else if (0)`) — its PTX carries **one**
`barrier.*` where the others carry two.

## Pipeline at a glance

No warp specialization, no async pipeline: no `cp.async`, no mbarrier, no TMA,
no atomics. The value split is a partition — CTA `(hv, value_tile)` owns
`ROWS_PER_CTA` value rows and every CTA redundantly preprocesses its own
`(n, hv)` tokens, which is why all four splits produce bit-identical results
(measured: `max|Δ| = 0` on output and the whole state pool).

| Role | Ownership | Publication/reuse edges |
| --- | --- | --- |
| CTA `(hv, value_tile, n)` | value head, sequence, `ROWS_PER_CTA` rows | independent |
| warp `w < 6`, phases A and C′ | **token `w`** | publishes `sK`/`sD`/`sBeta`/`sSlot`/`sToken`/(warp 0)`sInit` across barrier #1; publishes `sGramA*` row `w` and `sVec` columns `w`, `8+w` across the named barrier to the gram warp; the same `sVec` columns reach every *other* MMA warp only across barrier #2, since those warps merely arrive at the named barrier (split4 excepted — there all six block on it) |
| warp `GRAM_WARP` | **all 36 WY coefficient slots**: a 16×8×128 Gram product per side | consumes every token warp's `sGramA*` row and `sVec` column; publishes `sL`/`sR` across barrier #2 |
| thread `tid`, `group < ROW_GROUPS` (B, H) | `ROWS_PER_GROUP` rows × 8 keys | gathers state into `hist` and stages it to `sState`; later writes all six tokens' checkpoints |
| warp `w < MMA_WARPS` (D) | value rows `w·16 .. +15` | consumes all staged `sState` and all of `sVec` |
| lane `lane_quad == 2` | the depth-6 WY solve, the `sU` publish, **and tokens 4 and 5's output** | the only lanes that load `v` |
| lane `lane_quad >= 2` | tokens 0,1 on `4f+2`; 2,3 on `4f+3` | the only lanes that store `out` |

Guards per split — at split1 the token guard is the live one (warps 6,7 skip
A/C′/gram); at the other three the value guards are live and warp 5 is a
token-only warp (it preprocesses token 5, joins the named barrier, and at
splits 4/8 also runs the gram, but never gathers state, never issues the main
MMA and never stores output):

| guard | site | split1 | split2 | split4 | split8 |
| --- | --- | --- | --- | --- | --- |
| `warp_0 < 6` (A, C′, gram) | `:180`, `:293` | live (8 warps) | statically true | statically true | statically true |
| `group < ROW_GROUPS` (B, H) | `:411`, `:798` | 16 of 16 | 8 of 12 | 4 of 12 | 8 of 12 |
| `warp_0 < MMA_WARPS` (D–G) | `:448`, `:568`, `:671` | 8 of 8 | 4 of 6 | 2 of 6 | 1 of 6 |

Dependency chain: preprocess → **CTA barrier** → sVec/sGramA publish →
**named barrier** → gram MMA → state gather → **CTA barrier** → main MMA → WY
solve → outputs → `sU` → **warp barrier** → recurrence and checkpoints.

## Primitive vocabulary

Structural operations do not move or compute data:

```python
reg_tile(dtype, shape)                 # per-thread register array
smem_pool()                            # T.SMEMPool -- the dynamic arena
pool.alloc(bytes, align)               # the single shared allocation
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

Compute: `fill cast add sub mul fma exp rsqrt div dot mma`.
Sync: `cta_sync()`, `warp_sync()`, `named_bar_sync(id, threads)`,
`named_bar_arrive(id, threads)`.

**Every shuffle is width-32 with a full member mask.** **`dot` is not a
compound op**: each expands to an explicit FMA chain in a fixed association
order.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================
HEAD_DIM = 128
TOKENS = 6
VALUE_SPLIT   in (1, 2, 4, 8)                 # chosen by shape, constexpr here
THREADS       = 256 if VALUE_SPLIT == 1 else 192
ROWS_PER_CTA  = 128 // VALUE_SPLIT
ROWS_PER_GROUP= 2 if VALUE_SPLIT == 8 else 8
ROW_GROUPS    = ROWS_PER_CTA // ROWS_PER_GROUP
MMA_WARPS     = ROWS_PER_CTA // 16
GRAM_WARP     = 5 if VALUE_SPLIT in (4, 8) else 0

grid  = (HV * VALUE_SPLIT, num_seqs)          # binding_impl.cuh:64
block = (THREADS,)
launch_params += ["tirx.use_dyn_shared_memory"]   # or the arena is 0 bytes

# Runtime scalars: `scale`. `lower_bound` is 0.0 and A_log/dt_bias are 1-element
# dummies -- GATE_KIND 0 never dereferences them (measured, not assumed).

#                                                  s1 /    s2 /    s4 /    s8
pool    = smem_pool()
arena   = pool.alloc(SMEM_TOTAL, align=1024)               # 50560/32640/23680/19200
sState0 = smem_view(arena, ..., bf16, [ROWS_PER_CTA, 64])  #     0/     0/    0/    0
sState1 = smem_view(arena, ..., bf16, [ROWS_PER_CTA, 64])  # 16384/  8192/ 4096/ 2048
sVec    = smem_view(arena, ..., bf16, [128, 16])           # 32768/ 16384/ 8192/ 4096
sK      = smem_view(arena, ..., f32,  [6, 128])            # 36864/ 20480/12288/ 8192
sD      = smem_view(arena, ..., f32,  [6, 128])            # 39936/ 23552/15360/11264
sBeta   = smem_view(arena, ..., f32,  [6])                 # 43008/ 26624/18432/14336
sSlot   = smem_view(arena, ..., i32,  [6])                 # 43032/ 26648/18456/14360
sToken  = smem_view(arena, ..., i32,  [6])                 # 43056/ 26672/18480/14384
sInit   = smem_view(arena, ..., i32,  [1])                 # 43080/ 26696/18504/14408
sL      = smem_view(arena, ..., f32,  [6, 6])              # 43096/ 26712/18520/14424
sR      = smem_view(arena, ..., f32,  [6, 6])              # 43240/ 26856/18664/14568
sU      = smem_view(arena, ..., f32,  [6, ROWS_PER_CTA])   # 43384/ 27000/18808/14712
sGramA0 = smem_view(arena, ..., bf16, [16, 64])            # 46464/ 28544/19584/15104
sGramA1 = smem_view(arena, ..., bf16, [16, 64])            # 48512/ 30592/21632/17152

# sVec keeps 16 columns; only 0..5 (k) and 8..13 (q) are ever written.
# sGramA* rows 0..5 are written; 6..15 are read by ldmatrix and discarded.

# ===========================================================================
# Work decomposition and lane roles  (:100-178)
# ===========================================================================
work, n    = cta_id.x, cta_id.y
value_tile = work % VALUE_SPLIT
hv         = work // VALUE_SPLIT
query_head = hv // HEAD_RATIO                # the body's only div.s32
tid        = thread_id.x
warp       = make_warp_uniform(tid // 32)    # == token index in A and C'
# instruction_selection: shfl.sync.idx.b32 (lane 0, mask 0xFFFFFFFF); extent: scalar
# Semantically the identity, and NOT droppable: it is the hint that lets ptxas
# prove `warp_0 < 6` is warp-uniform. split1 is the geometry where that guard is
# live, and without it phase A's reductions land in a WARPSYNC.COLLECTIVE retry
# region -- the T=5 port paid for this and the fix is carried in from the start.
lane, lane_quad, frag_row = tid % 32, lane % 4, lane // 4
quad_base  = lane - lane_quad
group, lane_group = tid // 16, tid % 16
k_start, elem_start = lane_group * 8, lane * 4
tile_row_base, owned_row_base = value_tile * ROWS_PER_CTA, group * ROWS_PER_GROUP

# Register declarations (:161-177). Only the ha/hc/u arrays differ from T=5.
r_q, r_k, r_d = (reg_tile(f32, [4]) for _ in range(3))
ratio_scan    = reg_tile(f32, [4])                   # DEAD -- the butterfly path is gone
r_state       = reg_tile(f32, [8])
hist          = reg_tile(f32, [ROWS_PER_GROUP, 8])   # [8,8] / [2,8] at split8
state_pack, state_frag, vec_frag = (reg_tile(u32, [4]) for _ in range(3))
mma_acc, mma_acc_c = reg_tile(f32, [4]), reg_tile(f32, [4])
ha_lo, ha_hi, hc_lo, hc_hi, u_lo, u_hi = (reg_tile(f32, [TOKENS]) for _ in range(6))

token_base = load_i32(cu_seqlens[n])
seq_len    = load_i32(cu_seqlens[n+1]) - token_base
# instruction_selection: ld.global.nc.b32; extent: scalar (x2)
# cu_seqlens must step by exactly 6; the body does not bound-check it. Padding
# is signalled ONLY by ssm_state_indices < 0.

# ===========================================================================
# Phase A: token preprocess, warp <-> token  (:180-290)
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
    # Four contiguous bf16 = 8 bytes = ONE v2.b32; each word splits into two f32
    # by shl.b32 / and.b32.

    q_sum = dot(r_q, r_q); k_sum = dot(r_k, r_k)
    # instruction_selection: fma.rn.ftz.f32; extent: two 4-term chains (8 per
    #                        body), each seeded with C = 0f00000000
    # No `mul` at these sites, and the two chains are emitted INTERLEAVED,
    # because the source accumulates both in one fused loop (:241-244).
    for off in (16, 8, 4, 2, 1):
        q_sum = add(q_sum, bfly(q_sum, off, 31, 0xFFFFFFFF))
    for off in (16, 8, 4, 2, 1):
        k_sum = add(k_sum, bfly(k_sum, off, 31, 0xFFFFFFFF))
    # instruction_selection: shfl.sync.bfly.b32 + add.ftz.f32;
    #                        extent: 5 rounds x 2 SEQUENTIAL reductions = 10 shfl
    # `_warp_reduce_0` completes all five rounds (:245-249) before
    # `_warp_reduce_1` starts (:250-254) -- the opposite of the dot chains above.

    q_norm = mul(rsqrt(add(q_sum, 1e-6)), scale)
    k_norm = rsqrt(add(k_sum, 1e-6))
    # instruction_selection: rsqrt.approx.ftz.f32; extent: scalar (x2)
    # `scale` folds into q only. FTZ throughout: the export is -use_fast_math.

    for i in range(4):
        r_q[i] = mul(r_q[i], q_norm)
        r_k[i] = mul(r_k[i], k_norm)
        r_d[i] = exp(r_d[i])
        # instruction_selection: ex2.approx.ftz.f32; extent: scalar (x4)
        # GATE_KIND 0: g arrives in log space; this is its only transcendental.

    copy_r2s(r_k, sK[token, elem_start : +4])
    copy_r2s(r_d, sD[token, elem_start : +4])
    # instruction_selection: st.shared.v4.b32; extent: vector (one each)
    # Four contiguous f32 per lane -> ONE 16-byte store per region, 2 in the
    # phase. Issuing scalars here is a measurable regression, not a style choice.

    if lane == 0:
        raw_slot = load_i32(ssm_state_indices[n*6 + token])     # stride 6
        copy_r2s(raw_slot if active else -1, sSlot[token])      # source order:
        copy_r2s(token_pos, sToken[token])                     # sSlot, sToken,
        copy_r2s(beta[token_pos*HV + hv], sBeta[token])        # then sBeta
        # instruction_selection: ld.global.nc.b32 (ssm_state_indices)
        #                        + ld.global.nc.b16 + cvt.f32.bf16 (beta)
        #                        + st.shared.b32 x3; extent: scalar
    if warp == 0 and lane == 0:
        accepted  = clamp(load_i32(num_accepted_tokens[n]) - 1, 0, 5)   # [0,5]
        init_slot = load_i32(ssm_state_indices[n*6 + accepted])
        copy_r2s(0 if init_slot < 0 else init_slot, sInit[0])
        # instruction_selection: ld.global.nc.b32 x2 + max.s32 x2 + min.s32
        #                        + st.shared.b32; extent: scalar
        # Two independent clamps emit two max.s32: `accepted < 0` (:280) and
        # `initial_slot < 0` (:287).
        # Both clamp arms are exercised: nat 0/1 agree, 6/13 agree, 1 and 6 differ.

cta_sync()
# instruction_selection: bar.sync 0; extent: CTA

# ===========================================================================
# Phase C': gate prefix, sVec publish, sGramA publish  (:293-339)
# ===========================================================================
if warp < TOKENS:
    token = warp
    for i in range(4):
        k_idx  = elem_start + i
        prefix = 1.0
        for p in range(TOKENS):
            if token >= p:
                prefix = mul(prefix, copy_s2r(sD[p, k_idx]))
        # instruction_selection: ld.shared.b32 + mul.ftz.f32; extent: loop (<=6)
        # SCALAR (24 in the phase): the walk is across tokens at a fixed key, so
        # consecutive iterations are 512 B apart.

        copy_r2s(cast(bf16, mul(prefix, r_k[i])), sVec[swz(k_idx*32 + token*2)])
        copy_r2s(cast(bf16, mul(prefix, r_q[i])), sVec[swz(k_idx*32 + (8+token)*2)])
        # instruction_selection: cvt.rn.bf16.f32 + st.shared.b16; extent: scalar
        # The q column is 8+token; the generator's `c_col = 4 + token` at :311 is
        # a dead store overwritten at :313, and the port drops it.

        half = sGramA0 if k_idx < 64 else sGramA1
        copy_r2s(cast(bf16, div(r_k[i], prefix)), half[swz(token*128 + (k_idx%64)*2)])
        # instruction_selection: div.approx.ftz.f32 + cvt.rn.bf16.f32
        #                        + st.shared.b16; extent: scalar (8 div per body)
        # The gate-DEFLATED key. sVec holds the inflated one; their product is
        # the T<=4 ratio factor, with both operands rounded to bf16 and a
        # division the WY path never performed.

# ===========================================================================
# Phase C'': the coefficient Gram block  (:340-408)
# ===========================================================================
if warp < TOKENS:
    if VALUE_SPLIT == 4:
        named_bar_sync(1, TOKENS * 32)     # S4:340-343, all six token warps block
        # instruction_selection: barrier.sync 1, 192; extent: 6 warps
        is_gram = (warp == GRAM_WARP)
    else:
        is_gram = (warp == GRAM_WARP)
        if not is_gram:
            named_bar_arrive(1, TOKENS * 32)   # :406 -- publish and go
            # instruction_selection: barrier.arrive 1, 192; extent: 6 warps

    if is_gram:
        if VALUE_SPLIT != 4:
            named_bar_sync(1, TOKENS * 32)     # :343 -- wait for the other five
            # instruction_selection: barrier.sync 1, 192; extent: 6 warps
        gram_a, gram_b = reg_tile(u32, [4]), reg_tile(u32, [4])
        gram_k_acc, gram_q_acc = reg_tile(f32, [4]), reg_tile(f32, [4])

        for gram_half in range(2):                    # k 0..63 / 64..127
            for gram_k in range(0, 64, 16):
                a_addr = swz((lane % 16 * 64 + gram_k + lane // 16 * 8) * 2)
                b_addr = swz(((gram_half*64 + gram_k + lane % 16) * 16
                              + lane // 16 * 8) * 2)
                ldm(sGramA0 if gram_half == 0 else sGramA1, a_addr, gram_a)
                # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.shared.b16;
                #                        extent: tile (8 per body)
                ldm(sVec, b_addr, gram_b, trans=True)
                # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16;
                #                        extent: tile (8 per body)
                # b_addr is phase D's sVec formula; frags [0],[1] are columns
                # 0..7 (k side), [2],[3] are 8..15 (q side).

                init = (gram_half == 0 and gram_k == 0)
                mma(gram_k_acc, gram_a, gram_b[0:2], init=init)
                mma(gram_q_acc, gram_a, gram_b[2:4], init=init)
                # instruction_selection: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32;
                #                        extent: loop (2 x 4 x 2 = 16 issues)

        # M index = SOURCE token, N index = TARGET token.
        source = frag_row                 # 0..7, masked to < 6
        target0, target1 = lane_quad*2, lane_quad*2 + 1
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
            # Only acc[0],[1] are read; [2],[3] hold M rows 8..15. lane_quads 0..2
            # cover targets 0..5 -- at T=6 lane_quad 2's SECOND target (5) is live
            # too -- and lane_quad 3 (targets 6,7) is entirely masked out.
            # sL gets 15 live entries (strict lower), sR gets 21 (lower + diagonal).

# ===========================================================================
# Phase B: gather the initial checkpoint  (:410-446)  -- byte-identical to T=5
# ===========================================================================
if group < ROW_GROUPS:
    init_slot = copy_s2r(sInit[0])
    for row_local in range(ROWS_PER_GROUP):
        value_row_local = owned_row_base + row_local        # CTA-local, sState
        value_row       = tile_row_base + value_row_local   # global, `state`
        copy_g2r(state[init_slot*STATE_SLOT_STRIDE + hv*128*128
                       + value_row*128 + k_start : +8], state_pack)
        # instruction_selection: ld.global.v4.b32 (no .nc -- state is written
        #                        later in this kernel); extent: vector (8; 2 at split8)
        for i in range(8):
            hist[row_local*8 + i] = widen(state_pack)[i]
        # instruction_selection: shl.b32 / and.b32; extent: scalar
        half, k_off = (sState0, k_start) if lane_group < 8 else (sState1, k_start - 64)
        copy_r2s(state_pack, half[swz(value_row_local*128 + k_off*2)])
        # instruction_selection: st.shared.v4.b32; extent: vector (16; 4 at split8)
        # An if/ELSE selecting the destination half (:438-442), NOT a guard:
        # lanes 8..15 stage keys 64..127 into sState1.

cta_sync()
# instruction_selection: bar.sync 0; extent: CTA
# Orders sState and sL/sR -- and, at splits 1, 2 and 8, sVec: `barrier.arrive`
# releases but ACQUIRES NOTHING, so every MMA warp except the gram warp first
# synchronizes with the other token warps here. At split4 all six token warps
# block on the named barrier, so sVec is already visible.

# ===========================================================================
# Phase D: the main MMA chain  (:448-490)  -- byte-identical to T=5
# ===========================================================================
if warp < MMA_WARPS:
    for state_half in range(2):
        for mma_k in range(0, 64, 16):
            ldm(sVec, swz(((state_half*64 + mma_k + lane % 16)*16
                           + lane // 16 * 8) * 2), vec_frag, trans=True)
            # instruction_selection: ldmatrix...x4.trans.shared.b16; extent: tile (8)
            ldm(sState0 if state_half == 0 else sState1,
                swz(((warp*16 + lane % 16)*64 + (mma_k + lane // 16 * 8)) * 2),
                state_frag)
            # instruction_selection: ldmatrix...x4.shared.b16; extent: tile (8)
            init = (state_half == 0 and mma_k == 0)
            mma(mma_acc,   state_frag, vec_frag[0:2], init=init)   # columns 0..7
            mma(mma_acc_c, state_frag, vec_frag[2:4], init=init)   # columns 8..15
            # instruction_selection: mma.sync.aligned.m16n8k16...; extent: loop (16)
            # Two issues per step: the k side needs sVec columns 0..5 and the q
            # side 8..13, which no single n=8 tile covers.

# ===========================================================================
# Phase E: quad broadcasts and the WY solve  (:491-566)
# ===========================================================================
if warp < MMA_WARPS:
    # Source emission order (:491-530), then the elided token-5 reads.
    for t in range(4):
        ha_lo[t] = bcast(mma_acc[t % 2],     quad_base + t // 2, 31, 0xFFFFFFFF)
    for t in range(4):
        ha_hi[t] = bcast(mma_acc[2 + t % 2], quad_base + t // 2, 31, 0xFFFFFFFF)
    ha_lo[4] = bcast(mma_acc[0], quad_base + 2, 31, 0xFFFFFFFF)
    ha_hi[4] = bcast(mma_acc[2], quad_base + 2, 31, 0xFFFFFFFF)
    for t in range(5):
        hc_lo[t] = bcast(mma_acc_c[t % 2],     quad_base + t // 2, 31, 0xFFFFFFFF)
    for t in range(5):
        hc_hi[t] = bcast(mma_acc_c[2 + t % 2], quad_base + t // 2, 31, 0xFFFFFFFF)
    # instruction_selection: shfl.sync.idx.b32; extent: loop (20 per body)

    ha_lo[5], ha_hi[5] = mma_acc[1],   mma_acc[3]        # :533-534, NO shuffle
    hc_lo[5], hc_hi[5] = mma_acc_c[1], mma_acc_c[3]      # :535-536
    # instruction_selection: none -- register moves, folded away
    # Token 5's source lane IS quad_base + 2, the only lane that consumes it, so
    # the source reads the local registers. Assigned outside the lane_quad == 2
    # guard: every lane executes them, only lane_quad 2's values are meaningful.

    if lane_quad == 2:
        row_lo, row_hi = warp*16 + frag_row, warp*16 + frag_row + 8
        for t in range(TOKENS):                       # depth 6
            base = (copy_s2r(sToken[t]) * HV + hv) * 128
            solved_lo = sub(cast(f32, v[base + tile_row_base + row_lo]), ha_lo[t])
            solved_hi = sub(cast(f32, v[base + tile_row_base + row_hi]), ha_hi[t])
            # instruction_selection: ld.global.nc.b16 + cvt.f32.bf16 + sub.ftz.f32;
            #                        extent: scalar (12 v loads in the phase)
            for p in range(TOKENS):
                if p < t:
                    coef = copy_s2r(sL[t, p])
                    solved_lo = sub(solved_lo, mul(coef, u_lo[p]))
                    solved_hi = sub(solved_hi, mul(coef, u_hi[p]))
            # instruction_selection: ld.shared.{b32,v2.b32,v4.b32} (the 15 live
            #                        sL entries are read as a mixed scalar/vector
            #                        set) + mul.ftz.f32 + sub.ftz.f32;
            #                        extent: loop (15 live (t,p) pairs)
            u_lo[t], u_hi[t] = solved_lo, solved_hi

    for t in range(TOKENS):
        u_lo[t] = bcast(u_lo[t], quad_base + 2, 31, 0xFFFFFFFF)
        u_hi[t] = bcast(u_hi[t], quad_base + 2, 31, 0xFFFFFFFF)
    # instruction_selection: shfl.sync.idx.b32; extent: loop (12 per body)
    # The solve runs on lane_quad == 2 but lane_quad == 3 also writes output.

# ===========================================================================
# Phase F: the outputs  (:568-670)
# ===========================================================================
if warp < MMA_WARPS and lane_quad >= 2:
    token0, token1 = (lane_quad - 2) * 2, (lane_quad - 2) * 2 + 1
    row_lo_f, row_hi_f = warp*16 + frag_row, warp*16 + frag_row + 8
    # `tile_row_base` is folded into the store BASE below, exactly as the
    # merged T=5 port does: the source stores to `global_row_lo =
    # tile_row_base + value_row_lo_1` (:606-607). Dropping it is invisible at
    # split1 -- `value_tile = work & 0` folds it to 0 -- and silently wrong at
    # splits 2/4/8, where every value tile would write head rows [0, ROWS).
    # STATIC indices under a lane_quad branch, as the source does -- indexing
    # hc_* by a runtime token would spill the register array to local memory.
    out0_lo, out1_lo, out0_hi, out1_hi = hc_lo[0], hc_lo[1], hc_hi[0], hc_hi[1]
    if lane_quad == 3:
        out0_lo, out1_lo, out0_hi, out1_hi = hc_lo[2], hc_lo[3], hc_hi[2], hc_hi[3]

    for s in range(TOKENS):
        coef0 = copy_s2r(sR[token0, s]) if token0 >= s else 0.0
        coef1 = copy_s2r(sR[token1, s]) if token1 >= s else 0.0
        # instruction_selection: ld.shared.b32; extent: scalar (masked)
        out0_lo = fma(coef0, u_lo[s], out0_lo);  out1_lo = fma(coef1, u_lo[s], out1_lo)
        out0_hi = fma(coef0, u_hi[s], out0_hi);  out1_hi = fma(coef1, u_hi[s], out1_hi)
        # instruction_selection: fma.rn.ftz.f32; extent: loop (6 x 4 = 24)

    for (token, lo, hi) in ((token0, out0_lo, out0_hi), (token1, out1_lo, out1_hi)):
        slot = copy_s2r(sSlot[token])
        base = (copy_s2r(sToken[token])*HV + hv)*128 + tile_row_base
        copy_r2g(cast(bf16, lo) if slot >= 0 else 0.0, out[base + row_lo_f])
        copy_r2g(cast(bf16, hi) if slot >= 0 else 0.0, out[base + row_hi_f])
        # instruction_selection: cvt.rn.bf16.f32 + st.global.b16; extent: scalar
        # A padded row writes an EXPLICIT zero -- it is not skipped.

    if lane_quad == 2:                        # BOTH tails live here at T=6
        out4_lo, out4_hi = hc_lo[4], hc_hi[4]
        for s in range(TOKENS):
            coef4 = copy_s2r(sR[4, s]) if s <= 4 else 0.0
            # instruction_selection: ld.shared.v2.b32 + ld.shared.v4.b32
            #                        (sR[24..29] in one pair of loads);
            #                        extent: vector
            # Row 4 of sR is contiguous, so the whole six-entry row is loaded --
            # INCLUDING sR[29] -- and the clamp discards element 5 afterwards.
            # That clamp is load-bearing: sR[29] = (target 4, source 5) is never
            # written by the gram block, whose predicate is `source <= target`.
            # The source's own unconditional read at :633 is dead; the port
            # drops it and keeps the clamp.
            out4_lo = fma(coef4, u_lo[s], out4_lo)
            out4_hi = fma(coef4, u_hi[s], out4_hi)
        store token 4 exactly as above (same `... + tile_row_base` base),
            predicated on sSlot[4]

        out5_lo, out5_hi = hc_lo[5], hc_hi[5]
        for s in range(TOKENS):
            coef5 = copy_s2r(sR[5, s])        # NO clamp: target 5 accepts all six
            # instruction_selection: ld.shared.v4.b32 + ld.shared.v2.b32
            #                        (sR[30..35]); extent: vector
            out5_lo = fma(coef5, u_lo[s], out5_lo)
            out5_hi = fma(coef5, u_hi[s], out5_hi)
        store token 5 exactly as above (same `... + tile_row_base` base),
            predicated on sSlot[5]
        # instruction_selection: fma.rn.ftz.f32 + cvt.rn.bf16.f32
        #                        + st.global.b16, plus 6 scalar ld.shared.b32 for
        #                        the tails' sSlot[4]/sToken[4]/sSlot[5]/sToken[5];
        #                        extent: 24 fma, 8 stores
        # Both tails' coefficient rows vectorize. At T=5 only one tail existed
        # and the sketch there called it "the only sR read in the body that
        # vectorizes"; at T=6 both do, while the masked pair loop above still
        # cannot.
    # No out-of-region read anywhere: sSlot/sToken indices < 6, sR indices <= 35.

# ===========================================================================
# Phase G: publish sU  (:671-684)
# ===========================================================================
if warp < MMA_WARPS:
    if lane_quad == 2:
        for t in range(TOKENS):
            copy_r2s(u_lo[t], sU[t, warp*16 + frag_row])
            copy_r2s(u_hi[t], sU[t, warp*16 + frag_row + 8])
            # instruction_selection: st.shared.b32; extent: scalar (12 per body)
    if VALUE_SPLIT != 8:
        warp_sync()        # :681-683, INSIDE the guard
        # instruction_selection: bar.warp.sync -1; extent: warp
        # Warp w produces rows [16w, 16w+16), exactly what its own groups consume.

if VALUE_SPLIT == 8:
    cta_sync()             # S8:682-684, at OUTER scope -- all 192 threads
    # instruction_selection: bar.sync 0; extent: CTA
    # NOTE THE NESTING: split8's `warp_0 < 1` guard CLOSES at S8:681. One MMA
    # warp produces sU for eight consuming groups, so the barrier must be
    # unguarded; keeping it inside would hang the kernel.

# ===========================================================================
# Phase H: FP32 recurrence and checkpoints  (:798-834)
# ===========================================================================
if group < ROW_GROUPS:
    for t in range(TOKENS):
        slot_t = copy_s2r(sSlot[t]);  beta_t = copy_s2r(sBeta[t])
        sd_t, sk_t = copy_s2r(sD[t, k_start:+8]), copy_s2r(sK[t, k_start:+8])
        # instruction_selection: ld.shared.v4.b32 x4 (2 per region); extent: vector
        # Hoisted OUT of the row loop -- invariant across rows. 44 static v4 at
        # .loc 812 = 4 per token x 2*TOKENS-1 = 11 replications, the same nvcc
        # tail duplication the T=5 export shows (9 there).

        # The store predicate is hoisted out of the row loop and the loop
        # duplicated -- the shape nvcc produces for the source's per-row
        # `if (slot_t >= 0)` (:818). Keeping the branch per row costs DRAM
        # utilisation at the largest split1 shapes; the T=5 port measured it.
        if slot_t >= 0:
            for row_local in range(ROWS_PER_GROUP):
                row_h  = owned_row_base + row_local
                update = mul(copy_s2r(sU[t, row_h]), beta_t)
                for i in range(8):
                    hist[row_local*8 + i] = fma(hist[row_local*8 + i], sd_t[i],
                                                mul(update, sk_t[i]))
                # instruction_selection: mul.ftz.f32 + fma.rn.ftz.f32;
                #                        extent: loop (6 x ROWS_PER_GROUP x 8)
                # `hist*sD + (sU*beta)*sK`: the compiler contracts the FIRST
                # product and rounds update*sK.
                copy_r2g(pack_bf16x2(hist[row_local]),
                         state[slot_t*STATE_SLOT_STRIDE + hv*128*128
                               + (tile_row_base + row_h)*128 + k_start])
                # instruction_selection: cvt.rn.bf16x2.f32 x4 + st.global.v4.b32;
                #                        extent: vector (48 stores; 12 at split8)
        else:
            for row_local in range(ROWS_PER_GROUP):   # advance, store nothing
                same recurrence, no store
```

## Instruction-selection summary

Counted from the exported `--source-in-ptx` bodies with `ptx_census.py`
(comment-, brace- and predicate-aware) and attributed with
`ptx_phase_census.py` via `.loc`, restricted to the body file — `__shfl_sync`
and the bf16 conversions carry header line numbers and are only attributable
through `inlined_at`.

| form | s1 | s2 | s4 | s8 | where |
| --- | --- | --- | --- | --- | --- |
| `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | 32 | 32 | 32 | 32 | 16 gram + 16 main |
| `ldmatrix...x4.shared.b16` | 16 | 16 | 16 | 16 | 8 gram (A) + 8 main (sState) |
| `ldmatrix...x4.trans.shared.b16` | 16 | 16 | 16 | 16 | 8 gram (sVec) + 8 main (sVec) |
| `shfl.sync.idx.b32` | 33 | 33 | 33 | 33 | 1 warp-uniform + **20** broadcasts + **12** u |
| `shfl.sync.bfly.b32` | 10 | 10 | 10 | 10 | phase A's two L2 reductions |
| `div.approx.ftz.f32` | 8 | 8 | 8 | 8 | `k / prefix`, 4 elements x 2 halves |
| `ex2.approx.ftz.f32` | 4 | 4 | 4 | 4 | the gate, phase A |
| `rsqrt.approx.ftz.f32` | 2 | 2 | 2 | 2 | the q and k L2 norms |
| `bar.sync 0` | 2 | 2 | 2 | **3** | split8 replaces the warp barrier |
| `bar.warp.sync` | 1 | 1 | 1 | **0** | ditto |
| `barrier.sync/arrive 1, 192` | 2 | 2 | **1** | 2 | split4's arrive arm is dead |
| `st.shared.b16` | 16 | 16 | 16 | 16 | 4 sVec-k + 4 sVec-q + 8 sGramA |
| `st.shared.v4.b32` | 18 | 18 | 18 | **6** | 2 phase-A sK/sD + phase-B staging |
| `st.shared.b32` | 20 | 20 | 20 | 20 | 4 phase-A scalars + 4 gram sL/sR + 12 sU |
| `ld.shared.b32` | 57 | 57 | 57 | 57 | 24 in C′ (gate prefix), 17 in F (sR/sSlot/sToken), 11 in H, 2 in E, 1 sInit (`:292`), 1 in the gram block, 1 hoisted `sBeta[5]` with no `.loc` |
| `ld.shared.v4.b32` | 60 | 60 | 60 | **49** | 55 in H (44 sD/sK hoisted per token + 11 sU), 3 in E, 2 in F |
| `ld.shared.v2.b32` | 29 | 29 | 29 | **18** | 22 in H (sU peel), 4 in E (sToken/sL), 3 in F |
| `fma.rn.ftz.f32` | 760 | 760 | 760 | **226** | phase H dominates; 48 in F, 8 in A's dot chains |
| `mul.ftz.f32` | 867 | 867 | 867 | **275** | phase H dominates; C′'s prefix walk, E's solve, 4 in the gram result map |
| `ld.global.nc.v2.b32` | 3 | 3 | 3 | 3 | phase A's q, k and g slices |
| `ld.global.v4.b32` | 8 | 8 | 8 | **2** | phase B, one per owned row |
| `st.global.b16` | 14 | 14 | 14 | 14 | phase F outputs, scalar |
| `st.global.v4.b32` | 48 | 48 | 48 | **12** | phase H checkpoints (6 tokens x rows) |
| `cvt.rn.bf16x2.f32` | 192 | 192 | 192 | **48** | phase H packing |

Per-phase totals for split2, out of **3553** counted body instructions:
prologue 197, A 123, C′ 265, gram 127, B 211, D 93, E 175, F 291,
**H 1969**. The `BLOCK_CHECKPOINT_MMA` block contributes 0 — eliminated,
not merely unexecuted. Phase H is **55%** of the body; the gram block is
3.6%. (3553 is `ptx_census.py`'s counted-instruction figure; its
*statement line* count for the same body is 5112 and includes `.loc`
directives, declarations and labels.)

Splits 1, 2 and 4 share `ROWS_PER_GROUP = 8` and therefore identical
key-operation mixes apart from split4's missing `barrier.arrive`; their full
bodies differ only in address arithmetic (**3430 / 3553 / 3551** counted
instructions, split1 folding `value_tile` to 0). split8 has **1721**, and
differs only where `ROWS_PER_GROUP` does.

## Live/dead inventory

Live at T=6 that was dead at T≤4 (and stays live from T=5): `mma_acc_c`,
`vec_frag[2],[3]`, `hc_*`, `sGramA0/1`, the named barrier. Newly live at T=6
versus T=5: `ha_*[5]`/`hc_*[5]` (via the elided broadcast), sVec columns 5 and
13, `sGramA*` row 5, `sL` row 5 / `sR` row 5, and lane_quad 2's second gram
target.

Still dead: `ratio_scan[4]`; sVec columns 6,7,14,15; `sGramA*` rows 6..15;
`gram_*_acc[2],[3]`; the whole `BLOCK_CHECKPOINT_MMA` block; the three
`elem_start < 128` guards (`:194`, `:260`, `:294`) and the `r_q/r_k/r_d`
zero-init at `:188-193` that precedes the first of them, plus
`gate_a = 1.0f` (`:259`); and phase F's `token0 < 6` / `token1 < 6` guards
(`:610`, `:619`), statically true under `lane_quad >= 2` because token0 is
0 or 2 and token1 is 1 or 3 — this is exactly the guard that is LIVE at T=3
and that forced the predication that sibling needed. All of these carry
zero instructions in every export.

**Uninitialized-but-in-region reads are safe.** Both `ldmatrix` sites read full
16-row / 16-column tiles, touching `sGramA*` rows 6..15 and `sVec` columns
6,7,14,15 that nobody writes; those land in MMA outputs that no broadcast ever
selects (`m16n8k16` columns are independent). The one place an uninitialized
value could reach a result is `sR[29]` in phase F's token-4 tail, and the
source clamps it — which the port reproduces rather than predicating away.
