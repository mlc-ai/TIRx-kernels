<!--
This file is a design sketch for a TIRx port of code from FlashInfer
(https://github.com/flashinfer-ai/flashinfer @ f2e04400),
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# FlashKDA cake T=2 precomputed decode: coarse WY execution sketch

**Non-executable.** This is the semantic execution skeleton, not runnable code.
The source of truth for the implementation is
`tirx_kernels/flashinfer/kda/flashkda_decode_t2_precomputed.py`.

Transcribed from FlashInfer's frozen generated export
`csrc/kda/flashkda_decode_d128_t2_precomputed_split4.cu`, symbol
`kernel_flashinfer_recurrent_kda_wy_vtile_short` (flashinfer-ai/flashinfer @
`f2e04400`). Bare `:NNN` line references are into that file.

**Fixed specialization.** `HEAD_DIM = 128`, `TOKENS = 2`, `GATE_KIND = 0`
(precomputed log-gate), `VALUE_SPLIT = 4`, `LAUNCH_THREADS = 64`,
`DIRECT_IMPL` unset. The sm100a selector returns 4 unconditionally at T=2
(`recurrent_kda.py:1177-1178`), so split4 is the entire dispatch surface — there
is no split knob in this port.

**Out of scope.** T ∈ {1,3,4,5,6}; the `GATE_KIND = 1` lower-bound variant; the
`t1_direct` one-warp schedule (a different kernel symbol, ported separately);
`VALUE_SPLIT = 8`, reachable only on GB300 and not measurable here.

## Pipeline at a glance

Two warps, **no warp specialization**: the value-warp count, the state-warp
count and the token count all collapse to 2 at this specialization
(`flash_kda_decode.py:115-137`), so the same two warps play every role in
sequence. There is **no asynchronous pipeline** — no `cp.async`, no mbarrier, no
TMA, no atomics, `NUM_MAIN_STAGES 1`, nothing double-buffered.

The value split is a **partition**: CTA `(hv, value_tile)` owns value rows
`[32·value_tile, +32)` of the `[V,K]` state, and every CTA redundantly recomputes
the whole token preprocessing for its own `(n, hv)`. No CTA combines anything
with any other.

| Role | Ownership | Publication/reuse edges |
| --- | --- | --- |
| CTA `(hv, value_tile, n)` | value head `hv`, sequence `n`, 32 value rows | independent |
| warp `w`, phases A and C | **token `w`**: its q/k/g vectors, its L2 norms, its sVec columns and L/R row | publishes `sK`/`sD`/`sBeta`/`sSlot`/`sToken`, and (warp 0) `sInit`, across barrier #1; publishes `sVec`/`sL`/`sR` across barrier #2 |
| thread `tid`, phases B and H | 8 value rows (`group = tid/16`, rows `group*8 + r`) × 8 keys (`k_start = (tid%16)*8`) | gathers the state into registers **and** stages it to `sState` for the MMA; later writes both tokens' checkpoints |
| warp `w`, phase D | **value rows `w*16 .. +15`** of the 32-row tile | consumes `sState` (all 32 rows staged by both warps) and `sVec` |
| lane with `lane_quad == 2` | the WY solve, both tokens' outputs, and the `sU` publish for rows `w*16 + lane/4` and `+8` | the only lanes that load `v` and store `out` |

The lane→data mapping changes twice, and both barriers exist for that: `tid`-major
in phases A/C, `group`-major in phase B, MMA-fragment-major in phase D.

Dependency chain: preprocess → **barrier** → state gather+stage → sVec/L/R →
**barrier** → MMA → WY solve → outputs → sU → **warp barrier** → recurrence and
checkpoints.

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
exp(x); rsqrt(x); dot(a, b)            # `dot` = one explicit FMA chain
mma(acc, a_frag, b_frag, init=False)   # one mma.sync.m16n8k16 issue
```

Sync: `cta_sync()`, `warp_sync()`.

**Every shuffle is width-32 with a full member mask** — the operands are always
`(31, 0xFFFFFFFF)`. A reduction that wants a narrower group restricts its *xor
offset set*, never the instruction width. **`dot` is not a compound op**: each
one expands to an explicit FMA chain in a fixed association order, written out
at its site because the order is load-bearing.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================
HEAD_DIM, TOKENS, VALUE_SPLIT = 128, 2, 4
ROWS_PER_CTA    = 128 // VALUE_SPLIT      # 32                          (:159)
H, HV, HEAD_RATIO                          # HV % H == 0, HV >= H
GATE_TOKEN_STRIDE, STATE_SLOT_STRIDE       # host-derived, constexpr here

grid  = (HV * VALUE_SPLIT, num_seqs)       # binding_impl.cuh:64
block = (64,)                              # 2 warps

# The source uses dynamic smem (extern __shared__ __align__(1024), the launcher
# does cudaFuncSetAttribute + <<<..., SMEM_TOTAL, ...>>>). 14720 B fits the
# static limit, so the port declares one static arena at the same alignment and
# carves the same byte offsets -- what the swizzled ldmatrix addressing depends
# on is the offsets and the alignment, both preserved.
arena     = smem_arena(14720, align=1024)
sState0   = smem_view(arena,     0, bf16, [32, 64])   # keys   0..63, swizzled
sState1   = smem_view(arena,  4096, bf16, [32, 64])   # keys  64..127, swizzled
sVec      = smem_view(arena,  8192, bf16, [128, 16])  # row stride 32 B, swizzled
sK        = smem_view(arena, 12288, f32,  [2, 128])   # L2-normalized k
sD        = smem_view(arena, 13312, f32,  [2, 128])   # exp(g)
sBeta     = smem_view(arena, 14336, f32,  [2])
sSlot     = smem_view(arena, 14344, i32,  [2])
sToken    = smem_view(arena, 14352, i32,  [2])
sInit     = smem_view(arena, 14360, i32,  [1])
sL        = smem_view(arena, 14376, f32,  [2, 2])     # only sL[1][0] is live
sR        = smem_view(arena, 14392, f32,  [2, 2])     # sR[0][0], sR[1][0], sR[1][1]
sU        = smem_view(arena, 14408, f32,  [2, 32])
# sGramA0/1 (offsets 8192/10240) alias sVec and are t5/t6 machinery: dead here.

# ===========================================================================
# Work decomposition and lane roles  (:100-179)
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
group      = tid // 16          # phases B and H: 4 groups of 8 rows
lane_group = tid % 16
k_start    = lane_group * 8     # phases B and H: 8 keys per thread
elem_start = lane * 4           # phases A and C: 4 keys per lane
tile_row_base  = value_tile * ROWS_PER_CTA
owned_row_base = group * 8
token_base = cu_seqlens[n];  seq_len = cu_seqlens[n+1] - token_base
# instruction_selection: ld.global.nc.b32; extent: scalar (x2)   (.loc :147,:148)
# cu_seqlens must step by exactly TOKENS per row. The body does not bound-check
# token_base -- the contract is the caller's and is deliberately unvalidated for
# CUDA-graph capture. Padding is signalled ONLY by ssm_state_indices < 0.
#
# The source computes `warp` as make_warp_uniform(tid/32) at :101, which emits a
# shfl.sync.idx.b32 at .loc :39. It is semantically the identity -- a uniformity
# hint -- but its result IS `warp` and is live throughout, so it is not dead
# code; the port spells the same value as `tid // 32`. (The one-warp t1_direct
# sibling emits the same instruction with an unused result, which is where the
# "dead" characterization comes from; it does not carry over.)

# ===========================================================================
# Phase A: token preprocess, warp <-> token  (:180-290)
# ===========================================================================
if warp < TOKENS:
    token         = warp
    active_token  = token < seq_len
    token_pos     = token_base + token if active_token else 0
    qk_base   = (token_pos * H + query_head) * 128 + elem_start
    gate_base = token_pos * GATE_TOKEN_STRIDE + hv * 128 + elem_start

    r_q = reg_tile(f32, [4]); r_k = reg_tile(f32, [4]); r_d = reg_tile(f32, [4])
    copy_g2r(q[qk_base : +4],   r_q)
    # instruction_selection: ld.global.nc.v2.b32; extent: one 8-byte vector (4 bf16)
    copy_g2r(k[qk_base : +4],   r_k)
    # instruction_selection: ld.global.nc.v2.b32; extent: one 8-byte vector
    copy_g2r(g[gate_base : +4], r_d)
    # instruction_selection: ld.global.nc.v2.b32; extent: one 8-byte vector
    # Each 32-bit word holds two bf16 and is split by an integer pair, not by
    # cvt: lo = word << 16, hi = word & 0xffff0000  (:200-206)
    # instruction_selection: shl.b32 + and.b32; extent: 2 words per tensor, 3 tensors

    # ---- L2 norms: two independent full-warp butterflies ------------------
    # Association is index-ordered accumulation over i = 0..3  (:241-244)
    q_sq = dot(r_q, r_q);  k_sq = dot(r_k, r_k)
    # instruction_selection: fma.rn.ftz.f32; extent: 4-term chain each, the first
    # with C = 0f00000000 -- there is no mul at these sites
    for offset in (16, 8, 4, 2, 1):
        q_sq = add(q_sq, bfly(q_sq, offset, 31, 0xFFFFFFFF))
    # instruction_selection: shfl.sync.bfly.b32 + add.ftz.f32; extent: 5 rounds
    for offset in (16, 8, 4, 2, 1):
        k_sq = add(k_sq, bfly(k_sq, offset, 31, 0xFFFFFFFF))
    # instruction_selection: shfl.sync.bfly.b32 + add.ftz.f32; extent: 5 rounds
    # The two loops are sequential, not interleaved (.loc :248 then :253).
    q_norm = mul(rsqrt(add(q_sq, 1e-6)), scale)   # scale folds into q only
    k_norm = rsqrt(add(k_sq, 1e-6))
    # instruction_selection: rsqrt.approx.ftz.f32; extent: scalar (x2)
    # eps is hardcoded inside the rsqrt argument (:255,:257).

    # ---- gate and the shared publish  (:260-271) --------------------------
    for i in range(4):
        r_q[i] = mul(r_q[i], q_norm)
        r_k[i] = mul(r_k[i], k_norm)
        r_d[i] = exp(r_d[i])       # __expf: mul by log2(e), then ex2
        # instruction_selection: mul.ftz.f32 x3 + ex2.approx.ftz.f32; extent: 4 iterations
    copy_r2s(r_k, sK[token, elem_start : +4])
    copy_r2s(r_d, sD[token, elem_start : +4])
    # instruction_selection: st.shared.v4.b32; extent: one 16-byte tile each

    if lane == 0:
        sSlot[token]  = ssm_state_indices[n*2 + token] if active_token else -1
        sToken[token] = token_pos
        sBeta[token]  = cast(f32, beta[token_pos * HV + hv])
        # instruction_selection: ld.global.nc.b32 + ld.global.nc.b16 + cvt.f32.bf16
        #   + st.shared.b32 x3; extent: scalar each
        if token == 0:
            # num_accepted_tokens is LIVE at T=2 (it is not read at all by the
            # t1_direct sibling). It selects which checkpoint the recurrence
            # starts from, with both clamp edges reachable.  (:279-287)
            accepted     = clamp(num_accepted_tokens[n] - 1, 0, TOKENS - 1)
            initial_slot = ssm_state_indices[n*2 + accepted]
            sInit[0]     = 0 if initial_slot < 0 else initial_slot
            # instruction_selection: ld.global.nc.b32 x2 + st.shared.b32; extent: scalar

cta_sync()
# instruction_selection: bar.sync 0; extent: CTA
# Orders sInit -> the gather base below, and sK/sD/sBeta/sSlot/sToken -> their
# cross-warp readers in phases C, E, F and H.

# ===========================================================================
# Phase B: state gather, all 64 threads  (:292-329)
# ===========================================================================
# Every thread owns 8 rows x 8 keys and does two things with each row: keeps it
# as the FP32 recurrent carry, and stages the untouched bf16 bits into sState
# for the MMA to read back.
initial_head_base = sInit[0] * STATE_SLOT_STRIDE + hv * 128 * 128
hist = reg_tile(f32, [8, 8])
for row_local in range(8):                # constexpr unroll
    row  = owned_row_base + row_local
    pack = reg_tile(u32, [4])
    copy_g2r(state[initial_head_base + (tile_row_base + row)*128 + k_start : +8], pack)
    # instruction_selection: ld.global.v4.b32; extent: one 16-byte tile (8 bf16), 8 rows
    # NOT .nc: `state` is written later in this kernel.
    for p in range(4):
        hist[row_local][2*p], hist[row_local][2*p+1] = widen(pack[p])
    # instruction_selection: shl.b32 + and.b32; extent: 4 words per row
    half = sState0 if lane_group < 8 else sState1
    copy_r2s(pack, half[row, (k_start % 64) : +8] @ swz)
    # instruction_selection: st.shared.v4.b32; extent: one 16-byte tile, 8 rows
    # The bf16 bits go to shared unmodified; the swizzle is on the byte offset:
    # swz(row*128 + (k_start % 64)*2)  (:322,:324)

# ===========================================================================
# Phase C: sVec columns and the WY coefficients, warp <-> token  (:330-404)
# ===========================================================================
if warp < TOKENS:
    for i in range(4):                    # constexpr unroll
        k_idx  = elem_start + i
        prefix = 1.0
        for j in range(TOKENS):           # prefix gate: product over j <= token
            if token >= j:
                prefix = mul(prefix, sD[j, k_idx])
        # instruction_selection: ld.shared.b32 + mul.ftz.f32; extent: <= 2 per element
        sVec[k_idx, token]     = cast(bf16, mul(prefix, r_k[i]))
        sVec[k_idx, 4 + token] = cast(bf16, mul(prefix, r_q[i]))
        # instruction_selection: mul.ftz.f32 + cvt.rn.bf16.f32 + st.shared.b16
        # extent: 2 scalars per element, 4 elements  (.loc :346,:353)
        # The prefix multiply is FP32; the bf16 rounding happens on the way into
        # sVec, which is what the MMA will consume as its B operand.

    # ---- L and R: the WY coefficients  (:357-403) -------------------------
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
            # Source order is `r * source_k * ratio` (:372-375). At
            # source_offset == 0 the ratio is still 1.0 and folds away, so that
            # offset emits 4 bare fma (the first with C = 0f00000000) and only
            # source_offset == 1 emits the mul+fma pair -- 4 mul + 8 fma per dot
            # line across the two offsets.
            for offset in (16, 8, 4, 2, 1):
                dot_kk = add(dot_kk, bfly(dot_kk, offset, 31, 0xFFFFFFFF))
            for offset in (16, 8, 4, 2, 1):
                dot_qk = add(dot_qk, bfly(dot_qk, offset, 31, 0xFFFFFFFF))
            # instruction_selection: shfl.sync.bfly.b32 + add.ftz.f32; extent: 5 rounds x 2
            if lane == 0:
                beta_source = sBeta[source_token]
                if source_token < token:
                    sL[token, source_token] = mul(beta_source, dot_kk)
                sR[token, source_token] = mul(beta_source, dot_qk)
                # instruction_selection: ld.shared.b32 + mul.ftz.f32 + st.shared.b32
                # extent: scalar  (.loc :390,:392)
            if source_token > 0:
                for i in range(4):
                    ratio[i] = mul(ratio[i], sD[source_token, elem_start + i])
                # instruction_selection: ld.shared.v4.b32 + mul.ftz.f32; extent: 4

cta_sync()
# instruction_selection: bar.sync 0; extent: CTA
# The genuinely cross-warp edges are sVec (warp 0 writes columns 0 and 4, warp 1
# writes 1 and 5, and both warps' ldmatrix read columns 0..7) and sL/sR (written
# by one warp's lane 0, read by both warps in phases E and F). The sState edge is
# intra-warp by the same row-set argument as the phase-G warp barrier -- groups
# 0,1 are warp 0 and write rows 0..15, which is exactly what warp 0 reads back --
# and is ordered here only as a side effect. A port narrowing this barrier must
# reason from sVec/sL/sR, not from sState.

# ===========================================================================
# Phase D: the MMA chain, warp <-> 16 value rows  (:406-438)
# ===========================================================================
if warp < TOKENS:
    acc = reg_tile(f32, [4])
    for state_half in range(2):           # keys 0..63, then 64..127
        for mma_k in (0, 16, 32, 48):     # constexpr unroll -> 8 issues
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
            # ({0f00000000 x4}), the rest tie C to D  (.loc :431 x1, :435 x7)

# One chain, two projections. C[16 rows][8 cols] += state[row][k] * sVec[k][col]
# over all 128 keys, so with the sVec column assignment from phase C:
#   col 0,1 -> (prefix_t (*) k_t) . h0   the delta-rule history term
#   col 4,5 -> (prefix_t (*) q_t) . h0   the output history term
#   col 2,3,6,7 -> garbage (sVec columns never written; live at T=4)
# acc layout: acc[0],[1] = row `frag_row`, cols 2*lane_quad, +1;
#             acc[2],[3] = row `frag_row + 8`, same cols.
# So lane_quad == 0 holds the k-side and lane_quad == 2 holds the q-side.

# ===========================================================================
# Phase E: quad broadcast and the WY forward substitution  (:439-482)
# ===========================================================================
if warp < TOKENS:
    ha_lo = reg_tile(f32, [2]); ha_hi = reg_tile(f32, [2])
    for t in range(TOKENS):
        ha_lo[t] = bcast(acc[t],     quad_base, 31, 0xFFFFFFFF)
        ha_hi[t] = bcast(acc[2 + t], quad_base, 31, 0xFFFFFFFF)
    # instruction_selection: shfl.sync.idx.b32; extent: scalar, 4 live broadcasts
    # The source also broadcasts from quad_base+1 into ha_lo[2],[3] / ha_hi[2],[3]
    # (:443-446,:451-454). Those are MMA columns 2,3 -- T=4 territory, dead here.

    if lane_quad == 2:
        u_lo = reg_tile(f32, [2]); u_hi = reg_tile(f32, [2])
        row_lo = warp * 16 + frag_row;  row_hi = row_lo + 8
        for t in range(TOKENS):         # serial forward substitution
            base_t = (sToken[t] * HV + hv) * 128
            solved_lo = sub(cast(f32, v[base_t + tile_row_base + row_lo]), ha_lo[t])
            solved_hi = sub(cast(f32, v[base_t + tile_row_base + row_hi]), ha_hi[t])
            # instruction_selection: ld.global.nc.b16 + cvt.f32.bf16 + sub.ftz.f32
            # extent: 2 scalars per token  (.loc :463,:464)
            for s in range(TOKENS):
                if s < t:
                    solved_lo = sub(solved_lo, mul(sL[t, s], u_lo[s]))
                    solved_hi = sub(solved_hi, mul(sL[t, s], u_hi[s]))
                # instruction_selection: ld.shared.b32 + mul.ftz.f32 + sub.ftz.f32
                # extent: scalar, taken once (only s=0 < t=1)
            u_lo[t] = solved_lo;  u_hi[t] = solved_hi
    # The source then broadcasts u_lo/u_hi from quad_base+2 to the whole quad
    # (:476-482). Its only consumer is the dead BLOCK_CHECKPOINT_MMA block;
    # every live consumer below is already lane_quad == 2. Dead here.

# ===========================================================================
# Phase F: both tokens' outputs  (:484-531)
# ===========================================================================
if warp < TOKENS and lane_quad == 2:
    row_lo = warp * 16 + frag_row;  row_hi = row_lo + 8
    for t in range(TOKENS):
        out_lo = acc[t];  out_hi = acc[2 + t]     # the q-side MMA columns
        for s in range(TOKENS):
            coef = sR[t, s] if s <= t else 0.0    # sR[0][1] is never written
            out_lo = fma(coef, u_lo[s], out_lo)
            out_hi = fma(coef, u_hi[s], out_hi)
            # instruction_selection: ld.shared.b32 + fma.rn.ftz.f32; extent: scalar x2
        base_t = (sToken[t] * HV + hv) * 128 + tile_row_base
        active = sSlot[t] >= 0
        copy_r2g(cast(bf16, out_lo if active else 0.0), out[base_t + row_lo])
        copy_r2g(cast(bf16, out_hi if active else 0.0), out[base_t + row_hi])
        # instruction_selection: st.global.b16, 2 per token per branch (8 total);
        # cvt.rn.bf16.f32, 2 per token in the active branch but only 1 per token
        # in the zero branch, where both stores reuse the single converted zero
        # (6 total)  (.loc :503,:504,:506,:507 and :523,:524,:526,:527)
        # A padded row writes EXPLICIT zeros -- the upstream test asserts them
        # bit-exactly, so this is not an "unwritten" path.

# ===========================================================================
# Phase G: publish sU  (:532-545)
# ===========================================================================
if warp < TOKENS:
    if lane_quad == 2:
        for t in range(TOKENS):
            sU[t, warp*16 + frag_row]     = u_lo[t]
            sU[t, warp*16 + frag_row + 8] = u_hi[t]
        # instruction_selection: st.shared.b32; extent: 2 scalars per token
    warp_sync()
    # instruction_selection: bar.warp.sync; extent: warp
    # A warp barrier suffices because producer and consumer are the same warp:
    # writer rows are warp*16 + frag_row{,+8}, i.e. [0,16) for warp 0 and
    # [16,32) for warp 1; reader rows below are group*8 + r with group = tid/16,
    # i.e. groups 0,1 -> rows 0..15 and groups 2,3 -> rows 16..31. Any port that
    # changes the row-to-thread mapping must re-derive this or promote to
    # cta_sync.

# The BLOCK_CHECKPOINT_MMA != 0 block (:546-658), including the source's third
# mma.sync site, is statically dead: the macro is 0 and the binding
# static-asserts it. Not transcribed.

# ===========================================================================
# Phase H: recurrence and checkpoints, all 64 threads  (:659-695)
# ===========================================================================
for t in range(TOKENS):                   # serial over tokens
    slot_t = sSlot[t]
    beta_t = sBeta[t]
    for row_local in range(8):            # constexpr unroll
        row    = owned_row_base + row_local
        update = mul(sU[t, row], beta_t)
        # instruction_selection: ld.shared.v4.b32 (the hoisted sBeta/sSlot block)
        # + ld.shared.v2.b32/v4.b32 over the 8 consecutive sU rows + mul.ftz.f32
        # extent: scalar per row; phase H emits no scalar ld.shared.b32 at all
        words = reg_tile(u32, [4])
        for i in range(8):
            hist[row_local][i] = fma(hist[row_local][i], sD[t, k_start + i],
                                     mul(update, sK[t, k_start + i]))
            # instruction_selection: ld.shared.v4.b32 x2 (sD) + x2 (sK) per token
            # + mul.ftz.f32 + fma.rn.ftz.f32; extent: 8 iterations
            # The source writes `hist*sD + update*sK` (:673) and the compiler
            # contracts the FIRST product: `update*sK` is the rounded one and
            # `hist*sD` is fused into the FMA. Inverting this rounds the wrong
            # product at the kernel's only stateful accumulation, which feeds
            # both the checkpoint and token 1's history.
            # The recurrence advances UNCONDITIONALLY -- only the store below is
            # predicated -- and it stays FP32, so token 1 consumes the un-rounded
            # token-0 state rather than the bf16 checkpoint.
        for p in range(4):
            words[p] = cast(bf16x2, (hist[row_local][2*p], hist[row_local][2*p+1]))
        # instruction_selection: cvt.rn.bf16x2.f32; extent: 4 pairs per row
        if slot_t >= 0:
            copy_r2g(words, state[slot_t*STATE_SLOT_STRIDE + hv*16384
                                  + (tile_row_base + row)*128 + k_start : +8])
            # instruction_selection: st.global.v4.b32; extent: one 16-byte tile,
            # 8 rows x 2 tokens = 16 stores  (.loc :689)
# Both tokens' slots receive a full 32-row write: this kernel checkpoints per
# token, it does not write one final state.
```

## Inactive-row contract

| `sSlot[t]` | state | `out` for token t |
| --- | --- | --- |
| `>= 0` | rewritten at that slot | `acc_q + Σ sR·u` |
| `< 0` | untouched (the FP32 recurrence still advances) | **explicit 0.0** |

`cu_seqlens` plays no part in this: `active_token` gates only `sSlot`, and under
the stride-T caller contract it is always true.

## What is deliberately absent

The source is a Loom-generated template covering T=1..6; at T=2/split4 much of it
is statically dead, and the port carries none of it:

| dead item | site | why |
| --- | --- | --- |
| `sGramA0`, `sGramA1` | `:81-86`, `:136-139` | t5/t6 coefficient-gram; alias sVec, never touched |
| the whole `BLOCK_CHECKPOINT_MMA` block, incl. the 3rd `mma.sync` | `:546-658` | macro is 0, static-asserted |
| `ha_lo[2],[3]`, `ha_hi[2],[3]` + their 4 broadcasts | `:443-446`, `:451-454` | MMA columns 2,3 — T=4 territory |
| the `u_lo`/`u_hi` quad broadcasts | `:476-482` | only the dead block consumes them |
| `vec_frag[2],[3]` | `:415` | the x4 loads sVec cols 8..15; the MMA reads 2 registers |
| sVec columns 2,3 and 6..15 | — | never written; feed dead accumulator lanes. **Not zeroed** — zeroing is numerically safe but adds stores the source does not issue |
| `sR[0][1]`, `sL[0][*]`, `sL[1][1]` | `:390-392` | of the two 2×2 matrices only `sL[1][0]` and `sR[{0,0},{1,0},{1,1}]` are written and read |
| `elem_start < 128` guards, and the `r_q/r_k/r_d = 0.0f` prologue they guard | `:188-194` etc. | statically true at D=128, so the zero-init is unreachable |
| `group < 4` guards around phases B and H | `:294`, `:659` | statically true at 64 threads (`group = tid/16 ∈ {0..3}`) |
| `A_log`, `dt_bias`, `lower_bound` | `:98` | `GATE_KIND 0`; never dereferenced |

## Instruction-selection summary

Read off line-info PTX exported by re-running FlashInfer's own nvcc command line
(`.porting/flashkda_decode_t2_precomputed/ptx/d128_t2_precomputed_split4/`).
Body total **1570 instructions**; the counts every annotation above rests on:

| form | count | where |
| --- | ---: | --- |
| `mul.ftz.f32` | 254 | normalization, gate, prefix, dots, decay, coefficients |
| `fma.rn.ftz.f32` | 224 | every dot chain, the WY solve, the recurrence |
| `shl.b32` / `and.b32` | 131 / 116 | **38 each** are the bf16 widening pair (`<<16`, `& 0xffff0000`); the rest is index and swizzle arithmetic |
| `cvt.rn.bf16x2.f32` | 64 | 2 tokens × 8 rows × 4 pairs, the checkpoint packing |
| `xor.b32` | 34 | the shared swizzle |
| `add.ftz.f32` | 31 | butterfly tails and the eps adds |
| `shfl.sync.bfly.b32` | 30 | 2 L2 norms + 2 coefficient dots per token, 5 rounds each |
| `st.shared.v4.b32` | 18 | 16 sState stages + 2 sK/sD publishes |
| `st.global.v4.b32` | 16 | the checkpoints, 8 rows × 2 tokens |
| `cvt.rn.bf16.f32` | 14 | 8 sVec column stores + 8 output stores (both branches) |
| `shfl.sync.idx.b32` | 13 | 12 quad broadcasts + 1 `make_warp_uniform` (identity, but its result is `warp`) |
| `st.shared.b32` | 11 | scalars: sBeta/sSlot/sToken/sInit/sL/sR/sU |
| `ldmatrix…x4.shared.b16` | **8** | the A operand, one per MMA issue |
| `ldmatrix…x4.trans.shared.b16` | **8** | the B operand, one per MMA issue |
| `mma.sync…m16n8k16.f32.bf16.bf16.f32` | **8** | 1 zeroing (`.loc :431`) + 7 accumulating (`.loc :435`) |
| `ld.global.v4.b32` | 8 | the state gather — **not** `.nc`, since state is written |
| `st.shared.b16` | 8 | the sVec columns |
| `st.global.b16` | 8 | the outputs, both branches |
| `ld.global.nc.{b16,b32,v2.b32}` | 5 / 5 / 3 | v+beta, metadata, q/k/g |
| `ld.shared.v4.b32` | 19 | the sK/sD slices in phases C and H, the sBeta/sSlot block, sU runs |
| `ld.shared.b32` | 19 | the scalars only: sInit, prefix gate, sBeta, sL, sR, sSlot, sToken |
| `ld.shared.v2.b32` | 8 | sToken pairs and the sU row runs |
| `sub.ftz.f32` | 6 | the WY residuals |
| `cvt.f32.bf16` | 5 | the scalar v and beta loads |
| `ex2.approx.ftz.f32` | 4 | the gate, one per lane element |
| `rsqrt.approx.ftz.f32` | 2 | the two L2 norms |
| `bar.sync` / `bar.warp.sync` | 2 / 1 | the two CTA barriers and the sU warp barrier |

Three consequences the port must honour:

1. **Every float operation is `.ftz`.** `-use_fast_math` plus plain CUDA
   operators give `mul/add/sub.ftz.f32` and `fma.rn.ftz.f32`; `__expf` and
   `rsqrtf` give `ex2.approx.ftz.f32` / `rsqrt.approx.ftz.f32`. There is **no
   plain-`.f32` arithmetic anywhere in the body**. This matches the t1_direct
   sibling and is the opposite of the two CuTe-DSL KDA decode ports, whose
   helpers are deliberately non-FTZ.
2. **`.nc` follows read-only-ness, not `__restrict__` alone.** q, k, g, v, beta
   and the metadata load through `ld.global.nc.*`; the state gather is a plain
   `ld.global.v4.b32` because the same kernel writes that buffer.
3. **The MMA operands are bf16 and the accumulator is FP32.** `prefix⊙k` and
   `prefix⊙q` are rounded to bf16 on the way into sVec and the state A operand
   is bf16 straight from global, so the kernel loses precision an FP32 oracle
   does not — the port's measured gap to a sequential FP32 oracle is 6.1e-05 on
   the output, versus 6e-08 for the register-only t1_direct sibling.

## TIRx module and benchmark contract

- Module `tirx_kernels/flashinfer/kda/flashkda_decode_t2_precomputed.py`,
  `KERNEL_META["name"] = "flashkda_decode_t2_precomputed"`, cc 10.
- Reference: the frozen export itself through the JIT module's direct ABI, so
  the comparison is kernel-only and every metadata combination — negative slots,
  arbitrary `num_accepted_tokens` — is reachable.
- 9 bench rows: `hv32h16` × N{8,16,32,64,128} (parity with FlashInfer's own
  export bench), plus `hv16h16` and `hv12h12` at N{8,64}. `verify_dispatch.py`
  asserts every row reaches `d128_t2_precomputed_split4`.
