<!--
Copyright (c) 2025 by FlashInfer team.
Copyright (c) 2026 The TIRx Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Recurrent-KDA grouped-CTA decode SM100: coarse WASP execution sketch

This is a non-executable sketch of FlashInfer's CuTe DSL `_grouped_kda_kernel`
(`flashinfer/kda_kernels/recurrent_kda.py:482`, host `_grouped_kda_host:729`,
compile cache `_get_grouped_compiled:836`). It records the column/part thread
split, the register-resident `(8, G)` state granule, the four shared-memory
staging buffers and their per-thread strided views, the single CTA barrier that
separates the two thread-index mappings, the reduction and butterfly orders, and
the predicated output/checkpoint/orphan paths that the TIRx port must preserve.
The implementation represented by this sketch is maintained in
[`tirx_kernels/flashinfer/kda/recurrent_kda_decode_grouped.py`](../../../../tirx_kernels/flashinfer/kda/recurrent_kda_decode_grouped.py),
which becomes the source of truth after this sketch passes review.

The target is SM100a/B200. `D = 128` (`K == V`), `VSPLIT = 4`, `RATIO = 1`,
`USE_L2 = 1`, `HAS_DT_BIAS = 1`, `BETA_LOGIT = 0`, `USE_SRC = 0`. Q/K/V/G/beta/
state/out are BF16; `A_log`, `dt_bias`, `scale`, `lower_bound` are FP32;
`cu_seqlens`, `ssm_state_indices`, `num_accepted_tokens` are int32.

The host fixes the tiling with a hard rule, `KS = 4 if T == 1 else 2` and
`VSPLIT = 4` (`:859-860`), so **exactly two tile shapes are in scope**:

| variant | `T` | `KS` | `NT` threads | warps | `KC` | `G` | `CPB` | `EPT` | `SW` | smem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode | 1 | 4 | 128 | 4 | 32 | 4 | 32 | 1 | 4 | 1600 B |
| verify | 2/4/8 | 2 | 64 | 2 | 64 | 8 | 32 | 2 | 2 | 3200/6400/12800 B |

Both in-kernel gate modes are in scope: `GATE_MODE = 2` (lower bound, Kimi K3)
and `GATE_MODE = 1` (softplus, Kimi Linear). Out of scope: `GATE_MODE = 0`
(gate precomputed outside), `USE_SRC = 1`, `BETA_LOGIT = 1`, `USE_L2 = 0`,
`D = 64`, GQA (`RATIO != 1`), the dense no-`cu_seqlens` path, and the one-warp
`recurrent_kda_decode_kernel`, which owns `T == 1 and sequence_heads >= 128`
(`:1798`) and is sketched separately in
[`recurrent_kda_decode_one_warp.md`](recurrent_kda_decode_one_warp.md).

## Pipeline at a glance

This kernel has **no asynchronous pipeline, no mbarrier, no TMA, and no
`cp.async`**. It has shared memory and exactly **one** CTA barrier
(`cute.arch.sync_threads()` at `:641`). That barrier exists for a specific
reason: the two phases index threads differently, and the sketch must keep that
visible.

* **Phase A** maps a thread to gate/query/key *elements* by `d = tid % D`, and
  each thread handles `EPT` elements strided by `NT`. This mapping covers all
  `D = 128` gate columns of a token.
* **Phase B** maps a thread to a *state column* `v_idx = vz*CPB + tid//KS` and a
  *K-slice* `part = tid % KS` of that column. Each thread owns a `(8, G)` FP32
  granule of the `[V, K]` state and reads the whole `D`-length gate/key/query
  vectors for its own K-slice.

A thread's phase-A elements and its phase-B slice belong to different threads'
data, so the staged vectors must be complete before phase B starts. That is the
entire purpose of the barrier; there is no producer/consumer warp specialization.

| Role | Ownership | Publication/reuse edges |
| --- | --- | --- |
| CTA `(hv, n, vz)` | value head `hv`, sequence row `n`, column block `[vz*CPB, (vz+1)*CPB)` | independent; no cross-CTA communication |
| thread `tid`, phase A | gate/q/k elements `d + e*NT` for `e in [0, EPT)` of every token `t` | writes `s_eg/s_kr/s_qr`; consumed by *all* threads after the barrier |
| thread `tid`, phase B | state granule `(8, G)` = K-slice `part` of column `v_idx` | reads all of `s_eg/s_kr/s_qr` for its slice; carries `s` across tokens |
| lane `tid % 32`, warp `tid // 32` | one L2 partial pair per token | `lane == 0 and wid < SW` publishes to `s_red`; all threads read it back |
| `part` group of `KS` threads | the `KS` K-slices of one column | joined by butterfly shuffles for the two dot products |
| `part == 0` thread | the output element `out[pidx, hv, v_idx]` | sole writer of that element |
| CTA `(hv, 0, 0)`, `tid < D` | the orphan packed suffix `[cu[n_seq], q_total)` | decode-only tail; see the orphan note below |

The dependency chain per token is: staged gate/key/query (phase A) → L2 factors
→ decay + prediction dot → butterfly join → `deltak` → rank-1 update + output
dot → butterfly join → predicated output store → checkpoint store. Tokens are
strictly sequential in phase B because `s` carries between them; the `T` tokens
are `constexpr`-unrolled, not looped.

## Primitive vocabulary

Structural operations do not move or compute data:

```python
specialize(...)       # static D, T, KS, VSPLIT, GATE_MODE, ...
launch(...)           # grid/block metadata
tile(...)             # GMEM storage declaration
smem_tile(...)        # SMEM storage declaration
view(...)             # typed/strided view without a copy
slice(...)            # fix one coordinate of a view
reg_tile(...)         # thread-private registers
```

Copies always name their storage direction:

```python
copy_g2r(src, dst, evict=None, predicate=None)
copy_r2g(src, dst, evict=None, predicate=None)
copy_r2s(src, dst, predicate=None)
copy_s2r(src, dst)
```

There is no `copy_g2s`, TMA, `cp.async`, `ldmatrix`, `stmatrix`, or TCGEN05
operation in this kernel: every global↔shared transfer goes through registers.

The computation vocabulary is deliberately primitive. `vmul`/`vadd` are the
8-wide packed-FP32 forms; the scalar forms carry no `v` prefix:

```python
fill(dst, value)
cast(dst, src, rounding=None)
cast_f32(dst, src)                            # BF16 -> FP32 widening
add(dst, lhs, rhs);   vadd(dst, lhs, rhs)     # 8-wide FP32
mul(dst, lhs, rhs);   vmul(dst, lhs, rhs)     # 8-wide FP32
sub(dst, lhs, rhs)
neg(dst, src)
fma(dst, lhs, rhs, acc)
exp2(dst, src)
log1p(dst, src)
rcp(dst, src)
rsqrt(dst, src)
select(dst, predicate, true_value, false_value)
shuffle_bfly(src, lane_xor, clamp, member_mask) -> dst
vmov(dst, src)                                # register move, no PTX of its own
cta_sync()
```

There is no compound `l2norm`, `gate`, `sigmoid`, `softplus`, `delta_rule`,
`update_state`, `reduce`, or `checkpoint` operation: every one of those paths is
expanded below.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

variant = specialize(
    D=128,
    T=(1, 2, 4, 8),
    KS=(4, 2),            # tied to T by the host: KS = 4 if T == 1 else 2
    VSPLIT=4,
    RATIO=1,
    GATE_MODE=(1, 2),
    HAS_DT_BIAS=True,
    BETA_LOGIT=False,
    USE_L2=True,
    USE_SRC=False,
    target="sm_100a",
)

# recurrent_kda.py:515-524 -- every tile constant is derived from (D, KS, VSPLIT).
KC  = D // KS                 # state elements per thread     32 | 64
G   = KC // 8                 # 16B granules per thread        4 | 8
CPB = D // VSPLIT             # columns per CTA               32
NT  = (D * KS) // VSPLIT      # threads per CTA              128 | 64
EPT = max(D // NT, 1)         # phase-A elements per thread    1 | 2
SW  = min(D, NT) // 32        # warp partials per reduction    4 | 2

# The host owns the tiling rule and the dispatch domain
# (recurrent_kda.py:859-860, :1798).
host_assert(KS == (4 if T == 1 else 2) and VSPLIT == 4)
host_assert(not (T == 1 and N * HV >= 128))   # else the one-warp kernel wins
host_assert(HV % RATIO == 0)
host_assert(state_slot_stride % 8 == 0)       # cute.assume(divby=8), :766-767

# Runtime ABI. `q_total` may exceed `cu[n_seq]` only in the T == 1 decode
# layout; see the orphan note at the end.
q, k        = tile(gmem, bf16, [q_total, H,  D])      # indexed by h = hv // RATIO
v, g, out   = tile(gmem, bf16, [q_total, HV, D])      # indexed by hv
beta        = tile(gmem, bf16, [q_total, HV])
a_log       = tile(gmem, f32,  [H])
dt_bias     = tile(gmem, f32,  [H * D])
cu          = tile(gmem, i32,  [n_seq + 1])
ssm_idx     = tile(gmem, i32,  [n_seq, T])            # -1 == CUDA-graph pad slot
nat         = tile(gmem, i32,  [n_seq])
state       = tile(gmem, bf16, [n_slots, HV, D, D])   # read AND written in place

launch(grid=(HV, n_seq, VSPLIT), block=(NT, 1, 1),
       smem_bytes=4 * (3 * T * D + 16 * T),
       preferred_smem_carveout=25)

hv, n, vz = cta_id()
tid       = thread_id()
lane, wid = tid % 32, tid // 32
h         = hv // RATIO
v_idx     = vz * CPB + tid // KS      # the state column this thread owns
part      = tid % KS                  # which K-slice of that column
d         = tid % D                   # phase-A element base

# ===========================================================================
# Shared memory: three T-batched staging buffers + one reduction scratch
# ===========================================================================
# recurrent_kda.py:526-541. Declared as one 16B-aligned arena and carved into
# four views, so the TIRx port computes byte offsets at specialization time.

s_eg  = smem_tile(f32, [T * D])   # exp2(gate)          offset 0
s_kr  = smem_tile(f32, [T * D])   # un-normalized key   offset 4*T*D
s_qr  = smem_tile(f32, [T * D])   # un-normalized query offset 8*T*D
s_red = smem_tile(f32, [T * 16])  # L2 warp partials    offset 12*T*D

# Per-thread strided views: (8, G, KS, T) with stride (1, KS*8, 8, D), sliced at
# `part`. The KS threads of a column interleave at 8-element granularity, which
# is what makes each thread's 8-wide read contiguous.
eg_v = slice(view(s_eg, shape=(8, G, KS, T), stride=(1, KS * 8, 8, D)), part)
kr_v = slice(view(s_kr, shape=(8, G, KS, T), stride=(1, KS * 8, 8, D)), part)
qr_v = slice(view(s_qr, shape=(8, G, KS, T), stride=(1, KS * 8, 8, D)), part)

# ===========================================================================
# Row bounds and the initial checkpoint slot
# ===========================================================================

token_base = cu[n]
seq_len    = cu[n + 1] - token_base

# recurrent_kda.py:552-560. `nat` is an unconditional argument but is read only
# for T > 1; SGLang never supplies it, so the host substitutes a cached ones
# vector and `ic` collapses to 0.
ic = 0
if T > 1:
    ic = clamp(nat[n] - 1, 0, T - 1)
slot0 = ssm_idx[n, ic]
if slot0 < 0:
    slot0 = 0                     # pad rows still load (a defined address), never store

# ===========================================================================
# State load: one (8, G) BF16 granule per thread, eviction-hinted
# ===========================================================================

s   = reg_tile(f32,  (8, G))      # THE recurrent carry; lives the whole kernel
sb0 = reg_tile(bf16, (8, G))

gv0 = slice(view(state[slot0, hv, v_idx, :], shape=(8, G, KS), stride=(1, KS * 8, 8)), part)
copy_g2r(gv0, sb0, evict="no_allocate")
# instruction_selection: ld.global.L1::no_allocate.v4.b32; extent: G x 16B vectors
# The widening to FP32 is NOT written here; it is written after the barrier.
# See "state widening" below. The load itself does not move.

# ===========================================================================
# Loop-invariant gate constants
# ===========================================================================

av = 1.0
if GATE_MODE != 0:
    mul(t0, a_log[h], LOG2_E)
    # instruction_selection: mul.f32; extent: scalar
    exp2(av, t0)
    # instruction_selection: ex2.approx.ftz.f32; extent: scalar

dtb = reg_tile(f32, (EPT,))
for e in range(EPT):              # constexpr
    dtb[e] = dt_bias[h * D + d + e * NT]
    # instruction_selection: ld.global.b32; extent: scalar

# ===========================================================================
# Phase A: stage every token's gate/key/query, then one barrier
# ===========================================================================
# Thread mapping here is `d + e*NT`, NOT the (v_idx, part) mapping of phase B.

slots, ves = [], []               # constexpr python lists, one entry per token
q_bits, k_bits, g_bits, b_bits = [], [], [], []

# --- issue: every token's global loads, before any of them is consumed ------
# The source loads and consumes one token at a time.  Splitting the loop is the
# one deliberate deviation from its dataflow order: the values, their types and
# their consumers are unchanged, only the issue point moves.  bench_suite
# measures a cold L2 (it zeroes a 256 MB buffer before every timed iteration),
# which is also what an inference server sees, and under a cold L2 this kernel
# is bound by how many DRAM misses are in flight rather than by instruction
# count.  Per-token issue keeps 3*EPT+3 requests outstanding; issuing all T
# tokens keeps T times as many.  Cost is T*(3*EPT+1) BF16 staging registers.
for t in range(T):                # constexpr unroll
    slots.append(ssm_idx[n, t])
    # instruction_selection: ld.global.b32; extent: scalar

    # Out-of-row tokens clamp their token index to 0 so the loads below stay in
    # bounds; the value is discarded by the `active` predicate in phase B.
    pidx = token_base + t
    if t >= seq_len:
        pidx = 0
    # instruction_selection: setp.lt.s32 + selp.b32; extent: scalar

    ves.append(v[pidx, hv, v_idx])            # stays BF16 until :681
    # instruction_selection: ld.global.b16; extent: scalar
    b_bits.append(beta[pidx, hv])              # BETA_LOGIT == 0: already sigmoid'd
    # instruction_selection: ld.global.b16; extent: scalar
    for e in range(EPT):          # constexpr
        de = d + e * NT
        q_bits.append(q[pidx, h,  de])
        # instruction_selection: ld.global.b16; extent: scalar
        k_bits.append(k[pidx, h,  de])
        # instruction_selection: ld.global.b16; extent: scalar
        g_bits.append(g[pidx, hv, de])
        # instruction_selection: ld.global.b16; extent: scalar

# --- consume: gate math and the L2 partials, off the staged registers -------
bbs = []
for t in range(T):                # constexpr unroll
    bbs.append(cast_f32(b_bits[t]))
    # instruction_selection: cvt.f32.bf16; extent: scalar

    fill(sqp, 0.0); fill(skp, 0.0)
    for e in range(EPT):          # constexpr
        qe = q_bits[t * EPT + e]
        ke = k_bits[t * EPT + e]
        ge = g_bits[t * EPT + e]

        # The L2 partials accumulate the BF16 loads directly into FP32.
        fma(sqp, qe, qe, sqp)
        # instruction_selection: fma.rn.f32.bf16; extent: scalar
        fma(skp, ke, ke, skp)
        # instruction_selection: fma.rn.f32.bf16; extent: scalar

        # ---- gate branch (recurrent_kda.py:613-627) ----
        add(x, ge, dtb[e])
        # instruction_selection: add.rn.f32.bf16; extent: scalar
        if GATE_MODE == 1:                     # softplus, Kimi Linear
            mul(u, x, LOG2_E)
            # instruction_selection: mul.f32; extent: scalar
            exp2(u, u)
            # instruction_selection: ex2.approx.ftz.f32; extent: scalar
            log1p(sp, u)
            # instruction_selection: inlined __nv_log1pf -- add.rz.f32 exponent
            # split, 8-term fma.rn.f32 minimax chain, ln2 rescale, plus an
            # inf/zero fixup branch; extent: ~12 fma.rn.f32 per site
            select(sp, x > 20.0, x, sp)        # linear guard, :620
            # instruction_selection: setp.gt.f32 + selp.f32; extent: scalar
            mul(gate, sp, -av)
            # instruction_selection: mul.f32; extent: scalar
        else:                                  # GATE_MODE == 2, lower bound
            neg(u, x)
            # instruction_selection: neg.f32; extent: scalar
            mul(u, av, u)
            # instruction_selection: mul.f32; extent: scalar
            mul(u, u, LOG2_E)
            # instruction_selection: mul.f32; extent: scalar
            # The negation is its own instruction on `x`; it is NOT folded into
            # a negative LOG2_E constant.
            exp2(u, u)
            # instruction_selection: ex2.approx.ftz.f32; extent: scalar
            add(u, u, 1.0)
            # instruction_selection: add.f32; extent: scalar
            rcp(sig, u)
            # instruction_selection: rcp.rn.f32; extent: scalar  (NOT rcp.approx)
            mul(gate, lower_bound, sig)
            # instruction_selection: mul.f32; extent: scalar

        mul(eg, gate, LOG2_E)
        # instruction_selection: mul.f32; extent: scalar
        exp2(eg, eg)
        # instruction_selection: ex2.approx.ftz.f32; extent: scalar

        # Note: the staged key/query are the RAW values. L2 normalization is
        # applied in phase B as a scalar factor, not here.
        copy_r2s(eg, s_eg[t * D + de])
        # instruction_selection: st.shared.b32; extent: scalar
        # The staging tiles are FP32, so q/k convert here; the L2 partials
        # above consumed the raw BF16 registers instead.
        cast_f32(ke_f, ke)
        # instruction_selection: cvt.f32.bf16; extent: scalar
        cast_f32(qe_f, qe)
        # instruction_selection: cvt.f32.bf16; extent: scalar
        copy_r2s(ke_f, s_kr[t * D + de])
        # instruction_selection: st.shared.b32; extent: scalar
        copy_r2s(qe_f, s_qr[t * D + de])
        # instruction_selection: st.shared.b32; extent: scalar

    # ---- L2 partial reduction (recurrent_kda.py:633-640) ----
    # A full 32-lane butterfly, five rounds, even though only SW warp results
    # are consumed. The offsets DESCEND: `warp_reduction_sum` halves a group
    # width, so the emission order is 16, 8, 4, 2, 1. FP32 addition is not
    # associative and the checkpoint must be exact, so the port must not
    # reverse this.
    for off in (16, 8, 4, 2, 1):  # constexpr
        add(sqp, sqp, shuffle_bfly(sqp, off, 31, 0xffffffff))
        # instruction_selection: shfl.sync.bfly.b32 + add.f32; extent: scalar
        add(skp, skp, shuffle_bfly(skp, off, 31, 0xffffffff))
        # instruction_selection: shfl.sync.bfly.b32 + add.f32; extent: scalar

    copy_r2s(sqp, s_red[t * 16 + wid],     predicate=(lane == 0) & (wid < SW))
    # instruction_selection: setp.ne.b32 + branch-guarded st.shared.b32; extent:
    # scalar  (ptxas if-converts this site to a predicated STS in SASS)
    copy_r2s(skp, s_red[t * 16 + 8 + wid], predicate=(lane == 0) & (wid < SW))
    # instruction_selection: setp.ne.b32 + branch-guarded st.shared.b32; extent: scalar

cta_sync()
# instruction_selection: bar.sync 0; extent: CTA
# The ONLY barrier in the kernel. It separates the `d = tid % D` element mapping
# above from the `(v_idx, part)` granule mapping below.

# ---- state widening, deferred to here ----
# Nothing in phase A reads `s`, so this placement reaches only scheduling: same
# op, same extent, same instruction, same data-dependence order as writing it at
# the load site. ptxas hoists this form partway back anyway, so the two
# spellings land within a few instructions of each other on every landmark.
#
# Load-to-consumer distance in scheduled SASS:
#
#            T=1   T=2   T=4   T=8
#   source    60   199   393   874
#   deferred  71   216   462   970
#   at load   19    18    21    16
#
# It is worth 15-20% on the T = 1 decode shapes and is within run-to-run noise
# on every T > 1 shape, so it is unconditional. Measured on full bench_suite
# matrices, with the tight row cross-checked five times on a clock-verified GPU:
#
#                    deferred        at load site
#   dec_hv16_b4      1.062-1.090     0.855
#   dec_hv12_b8      1.045-1.087     1.019
#   dec_hv12_b4      1.064-1.068     0.995
#   ver_t8_hv16_b1   1.645           1.671
#   ver_t8_hv16_b4   1.652           1.642
#   ver_t2_hv16_b8   1.092           1.108
#
# This used to sit behind a `T <= 2` constexpr whose mechanism the sketch
# admitted it could not explain: before phase A issued all T tokens' loads up
# front, deferring cost 2-3% at T = 8. That cost is gone -- the deferral and the
# hoisting were two ways of moving work off the load's critical path, and the
# hoisting subsumes it -- so the boundary is gone too.
cast(s, sb0)
# instruction_selection: cvt.f32.bf16; extent: 8*G element loop

# ===========================================================================
# Phase B: barrier-free sequential recurrence over the T tokens
# ===========================================================================

pf   = reg_tile(f32, (8,))
kreg = reg_tile(f32, (8, G))      # staged keys, loaded in pass 1, reused in pass 2

for t in range(T):                # constexpr unroll; `s` carries across tokens
    slot   = slots[t]
    in_row = t < seq_len
    active = in_row & (slot >= 0)

    if active:
        pidx = token_base + t
        ve, bb = ves[t], bbs[t]

        # ---- L2 factors from the staged partials (:657-664) ----
        fill(sqt, 0.0); fill(skt, 0.0)
        for w in range(SW):       # constexpr
            copy_s2r(s_red[t * 16 + w], r0)
            # instruction_selection: the SW loop is fully vectorized in both
            # in-scope shapes -- ld.shared.v2.b32 at SW == 2 (T > 1) and
            # ld.shared.v4.b32 at SW == 4 (T == 1); extent: SW f32
            copy_s2r(s_red[t * 16 + 8 + w], r1)
            # instruction_selection: ld.shared.v2.b32 / ld.shared.v4.b32; extent: SW f32
            add(sqt, sqt, r0)
            # instruction_selection: add.f32; extent: scalar
            add(skt, skt, r1)
            # instruction_selection: add.f32; extent: scalar
        add(skt, skt, 1e-6)       # eps is hardcoded in the source, not an arg
        # instruction_selection: add.f32; extent: scalar
        rsqrt(rk, skt)
        # instruction_selection: rsqrt.approx.f32; extent: scalar  (NOT .ftz)
        add(sqt, sqt, 1e-6)
        # instruction_selection: add.f32; extent: scalar
        rsqrt(rq, sqt)
        # instruction_selection: rsqrt.approx.f32; extent: scalar
        mul(rq, rq, scale)
        # instruction_selection: mul.f32; extent: scalar

        # ---- pass 1: decay the state, accumulate the raw prediction ----
        # gi == 0 is peeled so `pvec` is initialized by a mul, not a fill+fma.
        copy_s2r(eg_v[:, 0, t], e0)
        # instruction_selection: ld.shared.v2.b64; extent: 8 f32 (2 x 16B)
        vmul(svec, s[:, 0], e0)
        # instruction_selection: mul.f32x2; extent: 8 f32 (4 packed pairs)
        vmov(s[:, 0], svec)
        copy_s2r(kr_v[:, 0, t], kreg[:, 0])
        # instruction_selection: ld.shared.v2.b64; extent: 8 f32
        vmul(pvec, kreg[:, 0], svec)
        # instruction_selection: mul.f32x2; extent: 8 f32
        for gi in range(1, G):    # constexpr
            copy_s2r(eg_v[:, gi, t], ei)
            # instruction_selection: ld.shared.v2.b64; extent: 8 f32
            vmul(svec, s[:, gi], ei)
            # instruction_selection: mul.f32x2; extent: 8 f32
            vmov(s[:, gi], svec)
            copy_s2r(kr_v[:, gi, t], kreg[:, gi])
            # instruction_selection: ld.shared.v2.b64; extent: 8 f32
            vmul(tmp, kreg[:, gi], svec)
            # instruction_selection: mul.f32x2; extent: 8 f32
            vadd(pvec, pvec, tmp)
            # instruction_selection: add.f32x2; extent: 8 f32
            # NOTE: mul + add, NOT fma.f32x2. The source keeps them separate and
            # the PTX confirms zero fma.*.f32x2 in either variant.

        # ---- balanced 8-term tree, then the KS butterfly join (:675-681) ----
        vmov(pf, pvec)
        add(pred, add(add(pf[0], pf[1]), add(pf[2], pf[3])),
                  add(add(pf[4], pf[5]), add(pf[6], pf[7])))
        # instruction_selection: add.f32; extent: 7-add balanced tree
        for off_i in range(bit_length(KS - 1)):   # KS=2 -> {1}; KS=4 -> {1, 2}
            add(pred, pred, shuffle_bfly(pred, 1 << off_i, 31, 0xffffffff))
            # instruction_selection: shfl.sync.bfly.b32 + add.f32; extent: scalar

        # ---- delta rule (:681) -- `rk` appears TWICE, once inside and once
        # outside the parenthesis. This is not a typo in the source.
        mul(t1, rk, pred)
        # instruction_selection: mul.f32; extent: scalar
        sub(t1, ve, t1)
        # instruction_selection: sub.rn.f32.bf16; extent: scalar -- `ve` is
        # never converted; the mixed-precision subtract takes the raw BF16.
        mul(deltak, rk, bb)
        # instruction_selection: mul.f32; extent: scalar
        mul(deltak, deltak, t1)
        # instruction_selection: mul.f32; extent: scalar

        # ---- pass 2: rank-1 update, accumulate the raw output ----
        # The source re-reads `kr_v` here, but the compiler keeps the pass-1
        # registers live -- the PTX shows 3*G shared loads per token, not 4*G,
        # which is why `kreg` is carried rather than reloaded.
        vmul(tmp, kreg[:, 0], deltak)
        # instruction_selection: mul.f32x2; extent: 8 f32
        vadd(svec, s[:, 0], tmp)
        # instruction_selection: add.f32x2; extent: 8 f32
        vmov(s[:, 0], svec)
        copy_s2r(qr_v[:, 0, t], q0)
        # instruction_selection: ld.shared.v2.b64; extent: 8 f32
        vmul(ovec, q0, svec)
        # instruction_selection: mul.f32x2; extent: 8 f32
        for gi in range(1, G):    # constexpr
            vmul(tmp, kreg[:, gi], deltak)
            # instruction_selection: mul.f32x2; extent: 8 f32
            vadd(svec, s[:, gi], tmp)
            # instruction_selection: add.f32x2; extent: 8 f32
            vmov(s[:, gi], svec)
            copy_s2r(qr_v[:, gi, t], qi)
            # instruction_selection: ld.shared.v2.b64; extent: 8 f32
            vmul(tmp, qi, svec)
            # instruction_selection: mul.f32x2; extent: 8 f32
            vadd(ovec, ovec, tmp)
            # instruction_selection: add.f32x2; extent: 8 f32

        vmov(pf, ovec)
        add(o, add(add(pf[0], pf[1]), add(pf[2], pf[3])),
               add(add(pf[4], pf[5]), add(pf[6], pf[7])))
        # instruction_selection: add.f32; extent: 7-add balanced tree
        for off_i in range(bit_length(KS - 1)):
            add(o, o, shuffle_bfly(o, 1 << off_i, 31, 0xffffffff))
            # instruction_selection: shfl.sync.bfly.b32 + add.f32; extent: scalar

        # ---- output store, owned by part == 0 (:698-700) ----
        mul(o, o, rq)
        # instruction_selection: mul.f32; extent: scalar
        cast(o_bf, o)
        # instruction_selection: cvt.rn.bf16.f32; extent: scalar
        copy_r2g(o_bf, out[pidx, hv, v_idx], predicate=(part == 0))
        # instruction_selection: setp.ne.b32 + branch-guarded st.global.b16;
        # extent: scalar
        # DELIBERATE DEVIATION: the source leaves this one branch-guarded and
        # ptxas does NOT if-convert it. The TIRx port emits `@p st.global.b16`
        # instead, because inline asm is opaque to ptxas -- an `if` around an asm
        # store can never be if-converted, and in the one-warp sibling that cost
        # extra BRA/BSYNC plus divergence-induced shuffle bank conflicts.

        # ---- BF16 checkpoint write, eviction-hinted (:702-713) ----
        sb1 = reg_tile(bf16, (8, G))
        cast(sb1, s, rounding="rn")
        # instruction_selection: cvt.rn.bf16x2.f32; extent: 4*G packed pairs
        gw = slice(view(state[slot, hv, v_idx, :], shape=(8, G, KS),
                        stride=(1, KS * 8, 8)), part)
        copy_r2g(sb1, gw, evict="no_allocate")
        # instruction_selection: st.global.L1::no_allocate.v4.b32; extent: G x 16B

    else:
        # Pad rows (slot < 0) still own their output element and must zero it,
        # because the host allocates `out` uninitialized (:1828-1830, :1848).
        fill(zero_bf, 0.0)
        # instruction_selection: none -- folded into the store's immediate operand
        copy_r2g(zero_bf, out[token_base + t, hv, v_idx],
                 predicate=in_row & (part == 0))
        # instruction_selection: branch-guarded st.global.b16 [addr], 0x0000;
        # extent: scalar  (ptxas if-converts this site to a predicated STG)

# ===========================================================================
# Orphan packed suffix (:719-725)
# ===========================================================================
# Carrier tokens beyond the last row are owned by no sequence. One CTA per value
# head zeroes them, covering all D columns via `tid + e*NT`.
#
# Reachable only in the T == 1 decode layout: spec mode sizes `out` as
# N*NUM_TOKENS while `cu_seqlens` must step by T (:1658-1661, :1822), so
# `q_total == cu[n_seq]` there and this loop is empty.

if (n == 0) & (vz == 0) & (tid < D):
    covered = cu[n_seq]
    for pos in range(covered, q_total):        # RUNTIME loop, not constexpr
        for e in range(EPT):                   # constexpr
            copy_r2g(zero_bf, out[pos, hv, tid + e * NT])
            # instruction_selection: st.global.b16; extent: scalar
```

## Static specialization boundary

| value | kind | why |
| --- | --- | --- |
| `D`, `T`, `KS`, `VSPLIT`, `RATIO`, `GATE_MODE`, `HAS_DT_BIAS`, `BETA_LOGIT`, `USE_L2`, `USE_SRC` | constexpr | the source's `cutlass.Constexpr` list (`:499-510`) |
| `KC`, `G`, `CPB`, `NT`, `EPT`, `SW` | constexpr | derived from `(D, KS, VSPLIT)` at `:515-521` |
| `n_seq` | constexpr in TIRx, runtime in the source | the source reads it from `grid_dim()`; every TIRx config pins a batch size, so specializing is exact and removes one launch scalar |
| `q_total`, `g` token stride, state slot stride, `scale`, `lower_bound` | runtime scalars | passed by the host (`:2185-2196`) |
| `source` stride, `source_indices` stride | dropped | dead under `USE_SRC = 0`; the host even aliases `source = state` (`:2143-2145`) |
| `stream` | dropped | supplied by the TIRx runtime |

## TIRx module and benchmark contract

* Module: `tirx_kernels/flashinfer/kda/recurrent_kda_decode_grouped.py`,
  `KERNEL_META["name"] = "recurrent_kda_decode_grouped"`, `compute_capability = 10`.
* Helpers (`_ptx_*`, non-FTZ scalar math, BF16 pack/convert, shuffles, guarded
  stores) are imported from the one-warp sibling module rather than duplicated.
* Every global **and shared** access must go through raw PTX with `ptr_to`:
  `low_level_ir.py:26` forbids `BufferLoad`/`BufferStore` on `global`, `shared`,
  and `shared.dyn` alike.
* TVM compiles with `--use_fast_math`, so native FP32 arithmetic lowers to
  `.ftz` forms. The source's only `.ftz` instruction is `ex2.approx.ftz.f32`;
  `add`/`mul`/`sub`/`fma`/`rsqrt`/`rcp` are all non-FTZ and must be emitted
  explicitly.
* Correctness configs cover both gate modes and four branch families (pad rows,
  orphan suffix, zero-length row, scratch stride `S > T`), each verified live
  against the source in `.porting/.../characterize_source.py`.
* Benchmark rows are `GATE_MODE = 2` only, so the softplus `log1p` polynomial is
  a correctness obligation rather than a performance-alignment target.

## Instruction-selection summary

Counts below are from the exported line-info PTX for the production verify shape
`T = 8, KS = 2, G = 8, GATE_MODE = 2` (`.porting/recurrent_kda_decode_grouped/ptx/t8_ks2_lb`,
`CUTE_DSL_LINEINFO=1`). They are evidence for the selections above, not hidden
computation. Per-token multipliers assume the 8 unrolled tokens.

| PTX instruction | count | origin |
| --- | --- | --- |
| `mul.f32x2` | 1024 | 8 tokens x 128 = the four 8-wide multiplies per granule (`s*eg`, `kr*svec`, `kr*deltak`, `qr*svec`) |
| `add.f32x2` | 704 | 8 tokens x 88 = `pvec` and `ovec` accumulation plus the rank-1 add |
| `ld.shared.v2.b64` | 384 | 8 tokens x 3G x 2 = the `eg`/`kr`/`qr` 8-wide reads; `kr` is loaded once and reused across both passes |
| `add.f32` | 272 | the two 7-add trees per token, the butterfly joins, the `1.0 +` of the sigmoid, and the L2/eps scalars |
| `cvt.rn.bf16x2.f32` | 256 | 8 tokens x 4G = the checkpoint cast |
| `cvt.f32.bf16` | 168 | the initial state granule plus the q/k staging converts and `beta` |
| `mul.f32` | 105 | gate scalars, `rq`, `deltak` |
| `shfl.sync.bfly.b32` | 96 | 80 = 8 tokens x 2 reductions x 5 rounds (`warp_reduction_sum`), 16 = 8 tokens x 2 KS joins x 1 round (`KS = 2`) |
| `ld.global.b16` | 64 | 8 tokens x (q, k, g at `EPT = 2`, plus `v` and `beta`) |
| `st.global.L1::no_allocate.v4.b32` | 64 | 8 tokens x G checkpoint granules |
| `st.shared.b32` | 64 | the phase-A staging writes plus the `s_red` publications |
| `ex2.approx.ftz.f32` | 33 | 1 for `av` + 8 tokens x 2 elements x 2 (sigmoid, gate) |
| `fma.rn.f32.bf16` | 32 | 8 tokens x 2 elements x 2 L2 partials |
| `add.rn.f32.bf16` | 16 | 8 tokens x 2 elements: `g + dt_bias` |
| `rcp.rn.f32` | 16 | 8 tokens x 2 elements: the `GATE_MODE = 2` sigmoid |
| `rsqrt.approx.f32` | 16 | 8 tokens x 2: the key and query L2 factors |
| `ld.shared.v2.b32` | 16 | 8 tokens x 2: the `SW = 2` `s_red` pair reads (the `T = 1` shape uses 2 x `ld.shared.v4.b32` instead, `SW = 4`) |
| `neg.f32` | 16 | 8 tokens x 2 elements: the `GATE_MODE = 2` sigmoid negation |
| `sub.rn.f32.bf16` | 8 | one per token: `ve - rk*pred`, consuming the raw BF16 `v` |
| `ld.global.L1::no_allocate.v4.b32` | 8 | G initial-state granules |

Three selections are load-bearing and easy to get wrong:

1. **Packed FP32.** The 8-wide granule arithmetic is `mul.f32x2` / `add.f32x2`,
   not scalar `mul.f32`, and not `fma.f32x2` — the source writes `a + b * c` but
   the compiler emits a separate multiply and add, with zero `fma.*.f32x2` in
   either variant. TIRx exposes `.f32x2` on `add`/`sub`/`mul`/`fma` for `sm_100`
   with `uint64` packed operands (`backend/cuda/ptx/table.py:2927-2985`).
2. **Non-FTZ everything except `ex2`.** `ex2.approx.ftz.f32` is the only `.ftz`
   instruction in the whole kernel. In particular `rsqrt.approx.f32` keeps
   denormals, and the `GATE_MODE = 2` sigmoid is `rcp.rn.f32` — a
   correctly-rounded reciprocal, not `rcp.approx.f32` and not `div.rn.f32`.
3. **Guarded stores, and one deliberate deviation.** The source's PTX contains
   **zero** predicated stores -- all three guarded sites are `setp` plus a branch.
   ptxas then if-converts two of them (the `s_red` publication and the pad-row
   zero-fill) into predicated SASS stores, but leaves the `part == 0` output store
   branch-guarded. The TIRx port emits `@p st.global.b16` at all three sites
   anyway. That is an intentional improvement, not a transcription: inline asm is
   opaque to ptxas, so an `if` wrapped around an asm store can never be
   if-converted, and in the one-warp sibling that cost extra `BRA`/`BSYNC` and
   turned the resulting divergence into shared-memory bank conflicts.
