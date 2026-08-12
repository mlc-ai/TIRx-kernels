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

# Recurrent-KDA one-warp decode SM100: coarse WASP execution sketch

This is a non-executable sketch of FlashInfer's CuTe DSL
`recurrent_kda_decode_kernel`
(`flashinfer/kda_kernels/recurrent_kda.py:190`, helpers `compute_gate_value:88`,
`_dot8_row:154`, `_reduce_k_group:174`). It records the exact single-warp lane
split, register-resident state tile, shuffle-only cross-lane protocol, gate
branch structure, reduction order, and predicated output/state paths that the
TIRx port must preserve. The implementation represented by this sketch is
maintained in
[`tirx_kernels/flashinfer/kda/recurrent_kda_decode_one_warp.py`](../../../../tirx_kernels/flashinfer/kda/recurrent_kda_decode_one_warp.py),
which becomes the source of truth after this sketch passes review.

The target is SM100a/B200. `HEAD_DIM = 128` (`K == V`), `NUM_TOKENS = 1`,
`USE_CU_SEQLENS = 1`, `USE_QK_L2NORM = 1`, `USE_GATE_IN_KERNEL = 1`,
`HAS_DT_BIAS = 1`, `BETA_IS_LOGIT = 0`, `HAS_INITIAL_STATE_SOURCE = 0`,
`HAS_NUM_ACCEPTED_TOKENS = 0`, `ZERO_PADDED_OUTPUT = 1`. Q/K/V/G/beta/state/out
are BF16; `A_log`, `dt_bias`, `scale`, `eps`, `lower_bound` are FP32;
`cu_seqlens` and `ssm_state_indices` are int32. Two schedule points are in
scope, `TILE_ROWS = 16` with `DOT_REDUCTION_DUAL_ACCUM` and `TILE_ROWS = 8`
with `DOT_REDUCTION_TREE`, together with both in-kernel gate modes
(`USE_LOWER_BOUND` 1 and 0). `TILE_ROWS = 32`, `HEAD_DIM = 64`, `NUM_TOKENS > 1`
and the whole speculative-decode path, the `sequence_heads < 128` grouped-CTA
kernel, GQA (`HV != H`), the dense no-`cu_seqlens` path, and the precomputed-gate
mode are out of scope.

## Pipeline at a glance

This kernel has **no asynchronous pipeline, no shared memory, no mbarrier, and
no barrier of any kind**. One CTA is exactly one warp (`.reqntid 32, 1, 1`).
Every lane runs the same program; the "roles" are the two lane coordinates that
partition the `[TILE_ROWS, 128]` state tile the warp owns in registers. All
cross-lane movement is a warp shuffle with a full `0xFFFFFFFF` member mask.

| Lane coordinate | Register ownership | Publication/reuse edges |
| --- | --- | --- |
| `k_lane = tidx % 16` | K slice `[8*k_lane, 8*k_lane + 8)` of every owned V row | supplies `_reduce_k_group` butterflies; `k_lane == j` selects the single lane that stores output row `j` |
| `v_lane = tidx // 16` | V rows `v_offset + v_lane + 2*j` for `j in [0, TILE_ROWS/2)` | receives `v_loaded` by indexed shuffle from the lane that loaded that row |
| `tidx` (flat, load view) | the `[4*tidx, 4*tidx + 4)` slice of q/k/g for the vector loads | redistributed to the `k_lane` view by 24 indexed shuffles per token |
| `tidx < TILE_ROWS` | one `v` element and one zero-pad output element | `v` published to all lanes by indexed shuffle at consume time |

The single dependency chain per CTA is: state load → gate → L2 norm → shuffle
broadcast → `TILE_ROWS/2` sequential rank-1 recurrence steps → output store →
state store. There is no producer/consumer split and nothing to overlap across
lanes beyond ILP inside one lane.

## Primitive vocabulary

Structural operations do not move or compute data:

```python
specialize(...)       # static HEAD_DIM, TILE_ROWS, reduction schedule, gate mode
launch(...)           # grid/block metadata
tile(...)             # GMEM storage declaration
view(...)             # typed view without a copy
reg_tile(...)         # lane-private scalar/vector registers
```

Copies always name their storage direction:

```python
copy_g2r(src, dst, predicate=None)
copy_r2g(src, dst, predicate=None)
```

There is no `copy_g2s`, `copy_s2r`, `copy_r2s`, `copy_s2g`, TMA, `cp.async`,
`ldmatrix`, `stmatrix`, or TCGEN05 operation in this kernel: it never touches
shared memory or tensor memory.

The computation vocabulary is deliberately primitive:

```python
fill(dst, value)
cast(dst, src, rounding=None)
add(dst, lhs, rhs)
sub(dst, lhs, rhs)
neg(dst, src)
mul(dst, lhs, rhs)
fma(dst, lhs, rhs, acc)
exp2(dst, src)
log2(dst, src)
div(dst, lhs, rhs)
rsqrt(dst, src)
select(dst, predicate, true_value, false_value)
shuffle_bfly(src, lane_xor, clamp, member_mask) -> dst
shuffle_index(src, source_lane, clamp, member_mask) -> dst
```

Predicates, loop indices, and static address expressions are control
operations. There is no compound `l2norm`, `gate`, `sigmoid`, `softplus`,
`delta_rule`, `update_state`, or `reduce` operation: every one of those paths is
expanded below.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

variant = specialize(
    HEAD_DIM=128,
    TILE_ROWS=(8, 16),
    DOT_REDUCTION_SCHEDULE=("TREE", "DUAL_ACCUM"),
    USE_LOWER_BOUND=(False, True),
    NUM_TOKENS=1,
    USE_CU_SEQLENS=True,
    USE_QK_L2NORM=True,
    USE_GATE_IN_KERNEL=True,
    HAS_DT_BIAS=True,
    BETA_IS_LOGIT=False,
    HAS_INITIAL_STATE_SOURCE=False,
    HAS_NUM_ACCEPTED_TOKENS=False,
    ZERO_PADDED_OUTPUT=True,
    target="sm_100a",
)

# recurrent_kda.py:242-246 -- the lane partition is fully determined by HEAD_DIM.
K_LANES = HEAD_DIM // 8               # 16
V_LANES = 32 // K_LANES               # 2
VALUES_PER_THREAD = HEAD_DIM // 32    # 4
NUM_V_TILES = HEAD_DIM // TILE_ROWS   # 8 or 16
ROWS = TILE_ROWS // V_LANES           # 8 or 4 register rows per lane

# The host picks (TILE_ROWS, DOT_REDUCTION_SCHEDULE) from sequence_heads and
# guarantees the one-warp domain (recurrent_kda.py:1108-1149, :1798).
host_assert(sequence_heads == N * HV)
host_assert(sequence_heads >= 128)
host_assert((TILE_ROWS, DOT_REDUCTION_SCHEDULE) ==
            select_kernel_schedule(HEAD_DIM, 1, True, sequence_heads))
host_assert(HV % H == 0)
host_assert(state_slot_stride % 16 == 0)   # symbolic outer stride, divisibility 16
host_assert(address(state) % 32 == 0)      # assumed_align=32 on every tensor
# gG alone among the token-indexed tensors is compiled with runtime batch and
# token strides (recurrent_kda.py:1035-1047) so a gate sliced out of a fused
# projection needs no copy; its head and K axes stay compact, and q/k/v/o are
# fully compact by contrast (:1050-1058).
host_assert(gate_batch_stride % 16 == 0)
host_assert(gate_token_stride % 16 == 0)

launch_config = launch(
    grid=(N * HV * NUM_V_TILES, 1, 1),   # recurrent_kda.py:986
    block=(32, 1, 1),                    # recurrent_kda.py:987 -- one warp
    threads=32,
    dynamic_smem_bytes=0,
    mbarriers=0,
)

def recurrent_kda_decode_one_warp(
    gQ, gK, gV, gG, gBeta, gH, gO, gALog, gDtBias,
    gCuSeqlens, gSsmStateIndices,
    scale, eps, lower_bound,
):
    # -----------------------------------------------------------------------
    # Lane and CTA coordinates (recurrent_kda.py:229-246)
    # -----------------------------------------------------------------------
    tidx = thread_id(axis="x", extent=32)
    # instruction_selection: mov.u32 from %tid.x; extent: one thread coordinate
    bidx = cta_id(axis="x", extent=N * HV * NUM_V_TILES)
    # instruction_selection: mov.u32 from %ctaid.x; extent: one CTA coordinate
    # The source flattens all three work axes into %ctaid.x and re-derives them
    # here; the division order fixes which CTA owns which (v_tile, head, seq).

    v_tile_idx = bidx % NUM_V_TILES
    bh = bidx // NUM_V_TILES
    value_head_idx = bh % HV
    batch_idx = bh // HV
    query_head_idx = value_head_idx // (HV // H)
    v_offset = v_tile_idx * TILE_ROWS
    k_lane = tidx % K_LANES
    v_lane = tidx // K_LANES

    # -----------------------------------------------------------------------
    # Sequence metadata (recurrent_kda.py:248-252)
    # -----------------------------------------------------------------------
    token_base_offset = copy_g2r(gCuSeqlens[batch_idx])
    # instruction_selection: ld.global.b32; extent: one scalar per lane
    seq_end = copy_g2r(gCuSeqlens[batch_idx + 1])
    # instruction_selection: ld.global.b32; extent: one scalar per lane
    seq_len = seq_end - token_base_offset

    # -----------------------------------------------------------------------
    # Zero-padded output prefill (recurrent_kda.py:253-255)
    # Runs BEFORE the state load so inactive rows are defined even when the
    # token loop skips every store. Valid only because NUM_TOKENS == 1 makes
    # the token axis index equal to batch_idx.
    # -----------------------------------------------------------------------
    if tidx < TILE_ROWS:
        zero_bf16 = cast("bf16", 0.0)
        # instruction_selection: no standalone instruction -- the bf16 zero is folded into the st.global.b16 immediate operand
        copy_r2g(zero_bf16, gO[0, batch_idx, value_head_idx, v_offset + tidx])
        # instruction_selection: st.global.b16 under a setp.gt.u32/@%p guard; extent: one predicated scalar store

    # -----------------------------------------------------------------------
    # Initial-state slot (recurrent_kda.py:257-272)
    # -----------------------------------------------------------------------
    init_raw_slot = copy_g2r(gSsmStateIndices[batch_idx * NUM_TOKENS])
    # instruction_selection: ld.global.b32; extent: one scalar per lane
    init_seq_idx = select(init_raw_slot < 0, 0, init_raw_slot)
    # instruction_selection: max.s32 against immediate 0; extent: one scalar. At
    # NUM_TOKENS == 1 this is the same expression on the same CSE'd slot load as
    # the seq_idx clamp below, so the pair emits a single max.s32.
    h_read = view(gH[init_seq_idx, value_head_idx, :, :])

    # -----------------------------------------------------------------------
    # Register storage (recurrent_kda.py:280-293). No SMEM, no mbarrier.
    # -----------------------------------------------------------------------
    h_reg = reg_tile("f32", [ROWS, 8])        # the recurrent state, layout stride (8, 1)
    h_bf16 = reg_tile("bf16", [8])            # one 16-byte state row in flight
    q_src = reg_tile("f32", [VALUES_PER_THREAD])
    k_src = reg_tile("f32", [VALUES_PER_THREAD])
    gate_src = reg_tile("f32", [VALUES_PER_THREAD])
    q_bf16 = reg_tile("bf16", [VALUES_PER_THREAD])
    k_bf16 = reg_tile("bf16", [VALUES_PER_THREAD])
    gate_bf16 = reg_tile("bf16", [VALUES_PER_THREAD])
    q_reg = reg_tile("f32", [8])              # this lane's K slice after broadcast
    k_reg = reg_tile("f32", [8])
    gate_reg = reg_tile("f32", [8])

    # -----------------------------------------------------------------------
    # State load (recurrent_kda.py:295-300)
    # Row j of this lane is V row (v_offset + v_lane + V_LANES*j); the K slice
    # is the 8 contiguous elements at 8*k_lane, which is exactly 16 bytes.
    # -----------------------------------------------------------------------
    # Every row's load is issued before any of them is widened. The source
    # interleaves the two, and this is the one deliberate deviation from its
    # order here: same loads, same widening, same values, only the issue point
    # moves. bench_suite times a cold L2 (it zeroes a 256 MB buffer before every
    # timed iteration), so the figure of merit is how many DRAM misses are in
    # flight; interleaved, each cvt chain sits on the critical path of the next
    # load. Worth 1.7-3.7% -- see the perf notes.
    #
    # Only the loads move. Hoisting the dt_bias loads the same way was measured
    # and rejected: 4 more f32 live across the token loop costs 6.6% on
    # hv16_b16_tr16_lb, in a kernel that already holds ROWS*8 f32 of state.
    for j in static_range(ROWS):
        v_idx = v_offset + v_lane + V_LANES * j
        copy_g2r(h_read[v_idx, 8 * k_lane : 8 * k_lane + 8], h_words[j])
        # instruction_selection: ld.global.v4.b32; extent: one 16-byte tile (8 bf16) per j, ROWS issues
    for j in static_range(ROWS):
        for i in static_range(8):
            h_reg[j, i] = cast("f32", h_words[j][i])
            # instruction_selection: cvt.f32.bf16; extent: one scalar, 8 per j

    # -----------------------------------------------------------------------
    # Per-head gate constants (recurrent_kda.py:302-305)
    # -----------------------------------------------------------------------
    h_K_offset = query_head_idx * HEAD_DIM
    a_log_raw = copy_g2r(gALog[query_head_idx])
    # instruction_selection: ld.global.b32; extent: one scalar per lane
    A_log_val = exp2(mul(a_log_raw, LOG2_E))
    # instruction_selection: mul.f32 then ex2.approx.ftz.f32; extent: one scalar pair, hoisted above the token loop

    # -----------------------------------------------------------------------
    # Loop-invariant lower-bound gate constants (recurrent_kda.py:124-127).
    # These live inside compute_gate_value's per-i loop in the source; both are
    # loop-invariant, so the compiler hoists them here.
    # -----------------------------------------------------------------------
    if USE_LOWER_BOUND:
        neg_A_log2e = mul(neg(A_log_val), LOG2_E)
        # instruction_selection: one mul.f32 against the negated immediate 0fBFB8AA3B (-log2 e); the neg is constant-folded and emits NO neg.f32 -- the lower-bound PTX contains zero neg.f32, unlike the softplus branch which emits exactly one; extent: one scalar, loop-invariant
        lb_log2e = mul(lower_bound, LOG2_E)
        # instruction_selection: mul.f32 against the immediate 0f3FB8AA3B; extent: one scalar, loop-invariant

    # =======================================================================
    # Token loop. NUM_TOKENS == 1, so this executes exactly once; the source
    # keeps it a real loop and the sketch keeps the same structure.
    # =======================================================================
    for token_t in range(NUM_TOKENS):
        # --- slot / activity resolution (recurrent_kda.py:321-332) ---------
        raw_slot = copy_g2r(gSsmStateIndices[batch_idx * NUM_TOKENS + token_t])
        # instruction_selection: ld.global.b32; extent: one scalar (CSE'd with the init_raw_slot load at NUM_TOKENS == 1)
        has_token = token_t < seq_len
        is_active = (raw_slot >= 0) and has_token
        token_offset = select(has_token, token_base_offset + token_t, 0)
        # instruction_selection: setp.gt.s32 for has_token then selp.b32; extent: one scalar
        # The false arm is literally 0 (recurrent_kda.py:328), not the sequence
        # base. Every q/k/gate/v/beta load below is unguarded, so an inactive or
        # CUDA-graph-padded row -- whose token_base_offset equals the total token
        # count -- would read out of bounds without this clamp.
        seq_idx = select(raw_slot < 0, 0, raw_slot)
        # instruction_selection: no standalone instruction -- CSE'd with the
        # init_seq_idx max.s32 above, since NUM_TOKENS == 1 makes raw_slot and
        # init_raw_slot the same load; extent: zero additional issues

        # --- per-head views and beta (recurrent_kda.py:333-346) ------------
        q_head = view(gQ[0, token_offset, query_head_idx, :])
        k_head = view(gK[0, token_offset, query_head_idx, :])
        gate_head = view(gG[0, token_offset, value_head_idx, :])
        v_head = view(gV[0, token_offset, value_head_idx, :])
        o_head = view(gO[0, token_offset, value_head_idx, :])
        beta_raw = copy_g2r(gBeta[0, token_offset, value_head_idx])
        # instruction_selection: ld.global.b16; extent: one scalar, identical address in every lane
        beta = cast("f32", beta_raw)
        # instruction_selection: cvt.f32.bf16; extent: one scalar
        # BETA_IS_LOGIT == 0: the source sigmoid at :348-351 is compile-time
        # eliminated and emits no PTX.

        # --- q/k/gate vector loads (recurrent_kda.py:353-358) --------------
        # Load view: thread tidx owns the contiguous 4-element slice at 4*tidx,
        # which is a different partition from the k_lane compute view below.
        copy_g2r(q_head[4 * tidx : 4 * tidx + 4], q_bf16)
        # instruction_selection: ld.global.v4.b16; extent: one 8-byte tile (4 bf16)
        copy_g2r(k_head[4 * tidx : 4 * tidx + 4], k_bf16)
        # instruction_selection: ld.global.v4.b16; extent: one 8-byte tile (4 bf16)
        copy_g2r(gate_head[4 * tidx : 4 * tidx + 4], gate_bf16)
        # instruction_selection: ld.global.v4.b16; extent: one 8-byte tile (4 bf16)

        # --- V load, hoisted only at TILE_ROWS == 16 (recurrent_kda.py:359-364) ---
        v_loaded = 0.0
        # instruction_selection: mov.b32 zero; extent: one scalar initializer
        if TILE_ROWS == 16:
            if tidx < TILE_ROWS:
                v_raw = copy_g2r(v_head[v_offset + tidx])
                # instruction_selection: ld.global.b16 under the tidx < TILE_ROWS guard; extent: one predicated scalar, issued here to hide its latency behind the gate and L2-norm block
                v_loaded = cast("f32", v_raw)
                # instruction_selection: cvt.f32.bf16; extent: one scalar

        # --- gate + q/k conversion (recurrent_kda.py:366-380, :88-148) -----
        for i in static_range(VALUES_PER_THREAD):
            k_idx = tidx * VALUES_PER_THREAD + i
            q_src[i] = cast("f32", q_bf16[i])
            # instruction_selection: cvt.f32.bf16; extent: one scalar
            k_src[i] = cast("f32", k_bf16[i])
            # instruction_selection: cvt.f32.bf16; extent: one scalar

            dt = copy_g2r(gDtBias[h_K_offset + k_idx])
            # instruction_selection: ld.global.b32; extent: one scalar, VALUES_PER_THREAD issues
            g_val = add(dt, gate_bf16[i])
            # instruction_selection: add.rn.f32.bf16; extent: one scalar -- the bf16 operand feeds the add directly, so no separate cvt.f32.bf16 is emitted for the gate input

            if USE_LOWER_BOUND:
                ag = mul(neg_A_log2e, g_val)
                # instruction_selection: mul.f32; extent: one scalar
                exp_neg = exp2(ag)
                # instruction_selection: ex2.approx.ftz.f32; extent: one scalar
                denom = add(1.0, exp_neg)
                # instruction_selection: add.f32; extent: one scalar
                ls = div(lb_log2e, denom)
                # instruction_selection: div.rn.f32 with lb_log2e as the numerator and (1 + exp_neg) as the denominator; extent: one scalar, VALUES_PER_THREAD issues.
                # The source writes sig = 1.0/denom followed by ls = lb_log2e * sig
                # (:136-137); those fold into this single divide -- no mul.f32 is
                # emitted for the lower_bound scale. The divide is full-precision
                # div.rn.f32, NOT rcp.approx.
                gate_src[i] = exp2(ls)
                # instruction_selection: ex2.approx.ftz.f32; extent: one scalar
            else:
                exp_g = exp2(mul(g_val, LOG2_E))
                # instruction_selection: mul.f32 then ex2.approx.ftz.f32; extent: one scalar pair
                log2_v = log2(add(1.0, exp_g))
                # instruction_selection: add.f32 then lg2.approx.ftz.f32; extent: one scalar pair
                gate_src[i] = exp2(mul(neg(A_log_val), log2_v))
                # instruction_selection: mul.f32 then ex2.approx.ftz.f32, with the neg.f32 hoisted out of the loop; extent: one scalar pair

        # --- q/k sum of squares (recurrent_kda.py:382-404) ----------------
        # The reduction schedule changes the dependency graph, not the result.
        if DOT_REDUCTION_SCHEDULE == "DUAL_ACCUM":
            q_sum_even = mul(q_src[0], q_src[0])
            # instruction_selection: mul.f32; extent: one scalar
            q_sum_odd = mul(q_src[1], q_src[1])
            # instruction_selection: mul.f32; extent: one scalar
            q_sum_even = fma(q_src[2], q_src[2], q_sum_even)
            # instruction_selection: fma.rn.f32.bf16 -- both multiplicands are still the bf16 load registers; extent: one scalar
            q_sum_odd = fma(q_src[3], q_src[3], q_sum_odd)
            # instruction_selection: fma.rn.f32.bf16; extent: one scalar
            q_sum_sq = add(q_sum_even, q_sum_odd)
            # instruction_selection: add.f32; extent: one scalar
            k_sum_even = mul(k_src[0], k_src[0])
            # instruction_selection: mul.f32; extent: one scalar
            k_sum_odd = mul(k_src[1], k_src[1])
            # instruction_selection: mul.f32; extent: one scalar
            k_sum_even = fma(k_src[2], k_src[2], k_sum_even)
            # instruction_selection: fma.rn.f32.bf16; extent: one scalar
            k_sum_odd = fma(k_src[3], k_src[3], k_sum_odd)
            # instruction_selection: fma.rn.f32.bf16; extent: one scalar
            k_sum_sq = add(k_sum_even, k_sum_odd)
            # instruction_selection: add.f32; extent: one scalar
        else:
            q_sum_sq = add(fma(q_src[0], q_src[0], mul(q_src[1], q_src[1])),
                           fma(q_src[2], q_src[2], mul(q_src[3], q_src[3])))
            # instruction_selection: two mul.f32, two fma.rn.f32.bf16 taking the raw bf16 load registers, one add.f32; extent: one vector -- two of the source's three adds contract into the fmas
            k_sum_sq = add(fma(k_src[0], k_src[0], mul(k_src[1], k_src[1])),
                           fma(k_src[2], k_src[2], mul(k_src[3], k_src[3])))
            # instruction_selection: two mul.f32, two fma.rn.f32.bf16 taking the raw bf16 load registers, one add.f32; extent: one vector

        # --- full-warp butterfly over all 32 lanes (recurrent_kda.py:405-411) ---
        # Both reductions span the whole warp because the load view partitions
        # K across all 32 lanes, not across the 16 k_lanes.
        for offset in [16, 8, 4, 2, 1]:
            q_sum_sq = add(q_sum_sq, shuffle_bfly(q_sum_sq, offset, 31, 0xFFFFFFFF))
            # instruction_selection: shfl.sync.bfly.b32 then add.f32; extent: one round, five rounds total
            k_sum_sq = add(k_sum_sq, shuffle_bfly(k_sum_sq, offset, 31, 0xFFFFFFFF))
            # instruction_selection: shfl.sync.bfly.b32 then add.f32; extent: one round, five rounds total

        # --- scale factors (recurrent_kda.py:413-417) ---------------------
        q_scale_factor = mul(rsqrt(add(q_sum_sq, eps)), scale)
        # instruction_selection: add.f32, rsqrt.approx.ftz.f32, mul.f32; extent: one scalar chain
        k_scale_factor = rsqrt(add(k_sum_sq, eps))
        # instruction_selection: add.f32, rsqrt.approx.ftz.f32; extent: one scalar chain

        # --- broadcast the load view into the k_lane compute view ---------
        # (recurrent_kda.py:418-436) Lane tidx ends up holding K elements
        # [8*k_lane, 8*k_lane + 8). source_lane walks the two load-view lanes
        # 2*k_lane and 2*k_lane + 1 that together cover that slice.
        for i in static_range(8):
            source_lane = V_LANES * k_lane + i // VALUES_PER_THREAD
            source_value = i % VALUES_PER_THREAD
            q_reg[i] = mul(shuffle_index(q_src[source_value], source_lane, 31, 0xFFFFFFFF),
                           q_scale_factor)
            # instruction_selection: shfl.sync.idx.b32 then mul.f32; extent: one element, 8 issues
            k_reg[i] = mul(shuffle_index(k_src[source_value], source_lane, 31, 0xFFFFFFFF),
                           k_scale_factor)
            # instruction_selection: shfl.sync.idx.b32 then mul.f32; extent: one element, 8 issues
            gate_reg[i] = shuffle_index(gate_src[source_value], source_lane, 31, 0xFFFFFFFF)
            # instruction_selection: shfl.sync.idx.b32; extent: one element, 8 issues -- no scale multiply on the gate

        # --- late V load for the non-TILE_ROWS-16 schedules ---------------
        # (recurrent_kda.py:437-439)
        if TILE_ROWS != 16:
            if tidx < TILE_ROWS:
                v_loaded = cast("f32", copy_g2r(v_head[v_offset + tidx]))
                # instruction_selection: ld.global.b16 under the tidx < TILE_ROWS guard then cvt.f32.bf16; extent: one predicated scalar, placed next to its consumer

        # --- sequential rank-1 recurrence (recurrent_kda.py:440-460) ------
        # Each j is one V row of the warp's tile. The two dots inside a j are
        # separated by the state update, so the loop is a true serial chain.
        for j in static_range(ROWS):
            for i in static_range(8):
                h_reg[j, i] = mul(h_reg[j, i], gate_reg[i])
                # instruction_selection: mul.f32; extent: one scalar, 8 per j

            # pred = (S @ k) for this row, reduced across the 16 k_lanes
            if DOT_REDUCTION_SCHEDULE == "DUAL_ACCUM":
                even = mul(h_reg[j, 0], k_reg[0])
                # instruction_selection: mul.f32; extent: one scalar
                odd = mul(h_reg[j, 1], k_reg[1])
                # instruction_selection: mul.f32; extent: one scalar
                for pair in static_range(1, 4):
                    even = fma(h_reg[j, 2 * pair], k_reg[2 * pair], even)
                    # instruction_selection: fma.rn.f32; extent: one scalar, three issues
                    odd = fma(h_reg[j, 2 * pair + 1], k_reg[2 * pair + 1], odd)
                    # instruction_selection: fma.rn.f32; extent: one scalar, three issues
                pred = add(even, odd)
                # instruction_selection: add.f32; extent: one scalar
            else:
                pred = add(add(fma(h_reg[j, 0], k_reg[0], mul(h_reg[j, 1], k_reg[1])),
                               fma(h_reg[j, 2], k_reg[2], mul(h_reg[j, 3], k_reg[3]))),
                           add(fma(h_reg[j, 4], k_reg[4], mul(h_reg[j, 5], k_reg[5])),
                               fma(h_reg[j, 6], k_reg[6], mul(h_reg[j, 7], k_reg[7]))))
                # instruction_selection: balanced mul.f32 / fma.rn.f32 / add.f32 tree; extent: four mul, four fma, three add

            for offset in [8, 4, 2, 1]:
                pred = add(pred, shuffle_bfly(pred, offset, 31, 0xFFFFFFFF))
                # instruction_selection: shfl.sync.bfly.b32 then add.f32; extent: one round, four rounds reduce the 16-lane k group

            v_idx = v_offset + v_lane + V_LANES * j
            v_val = shuffle_index(v_loaded, v_lane + V_LANES * j, 31, 0xFFFFFFFF)
            # instruction_selection: shfl.sync.idx.b32; extent: one element -- pulls row j's V from the lane that loaded it

            delta = mul(sub(v_val, pred), beta)
            # instruction_selection: sub.f32 then mul.f32; extent: one scalar pair

            for i in static_range(8):
                h_reg[j, i] = fma(k_reg[i], delta, h_reg[j, i])
                # instruction_selection: fma.rn.f32; extent: one scalar, 8 per j -- the rank-1 state update

            # out = (S @ q) for this row, same reduction structure as pred
            if DOT_REDUCTION_SCHEDULE == "DUAL_ACCUM":
                even = mul(h_reg[j, 0], q_reg[0])
                # instruction_selection: mul.f32; extent: one scalar
                odd = mul(h_reg[j, 1], q_reg[1])
                # instruction_selection: mul.f32; extent: one scalar
                for pair in static_range(1, 4):
                    even = fma(h_reg[j, 2 * pair], q_reg[2 * pair], even)
                    # instruction_selection: fma.rn.f32; extent: one scalar, three issues
                    odd = fma(h_reg[j, 2 * pair + 1], q_reg[2 * pair + 1], odd)
                    # instruction_selection: fma.rn.f32; extent: one scalar, three issues
                out = add(even, odd)
                # instruction_selection: add.f32; extent: one scalar
            else:
                out = add(add(fma(h_reg[j, 0], q_reg[0], mul(h_reg[j, 1], q_reg[1])),
                              fma(h_reg[j, 2], q_reg[2], mul(h_reg[j, 3], q_reg[3]))),
                          add(fma(h_reg[j, 4], q_reg[4], mul(h_reg[j, 5], q_reg[5])),
                              fma(h_reg[j, 6], q_reg[6], mul(h_reg[j, 7], q_reg[7]))))
                # instruction_selection: balanced mul.f32 / fma.rn.f32 / add.f32 tree; extent: four mul, four fma, three add

            for offset in [8, 4, 2, 1]:
                out = add(out, shuffle_bfly(out, offset, 31, 0xFFFFFFFF))
                # instruction_selection: shfl.sync.bfly.b32 then add.f32; extent: one round, four rounds

            if is_active:
                if k_lane == j:
                    out_bf16 = cast("bf16", out, rounding="rn")
                    # instruction_selection: cvt.rn.bf16.f32; extent: one scalar, one per j
                    copy_r2g(out_bf16, o_head[v_idx])
                    # instruction_selection: st.global.b16 under the is_active and k_lane == j guards; extent: one predicated scalar store per j

        # --- state writeback (recurrent_kda.py:462-469) -------------------
        # The source rebinds h_out to seq_idx at :463. At NUM_TOKENS == 1 that is
        # the SAME slot the prologue loaded from: init_raw_slot (:259) and
        # raw_slot (:323) index the same element, so init_seq_idx and seq_idx
        # share one max.s32 and the store reuses the load's base address. A
        # negative slot cannot reach here at all, because it clears is_active
        # (:327). The port must keep the rebind structurally without modelling it
        # as a second, distinct slot.
        if is_active:
            h_out = view(gH[seq_idx, value_head_idx, :, :])
            for j in static_range(ROWS):
                v_idx = v_offset + v_lane + V_LANES * j
                for i in static_range(8):
                    h_bf16[i] = cast("bf16", h_reg[j, i], rounding="rn")
                    # instruction_selection: cvt.rn.bf16x2.f32 packing two elements at a time; extent: four issues per j
                copy_r2g(h_bf16, h_out[v_idx, 8 * k_lane : 8 * k_lane + 8])
                # instruction_selection: st.global.v4.b32; extent: one 16-byte tile per j, ROWS issues
```

## Static specialization and launch boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| `HEAD_DIM = 128` | static | fixes `K_LANES`/`V_LANES`/`VALUES_PER_THREAD`, the 16-byte state vector, and the four-round k-group butterfly |
| `TILE_ROWS` (8 or 16) | static | register-tile row count, the `NUM_V_TILES` factor of the flat 1-D grid extent (`:986`), and whether the V load is hoisted above the gate block |
| `DOT_REDUCTION_SCHEDULE` | static | dependency graph of both 8-element dots and of the q/k sum-of-squares |
| `USE_LOWER_BOUND` | static | selects the `div.rn.f32` sigmoid chain or the `lg2.approx` softplus chain |
| `NUM_TOKENS = 1`, `ZERO_PADDED_OUTPUT = 1`, `HAS_DT_BIAS = 1`, `BETA_IS_LOGIT = 0`, `HAS_INITIAL_STATE_SOURCE = 0`, `HAS_NUM_ACCEPTED_TOKENS = 0`, `USE_QK_L2NORM = 1`, `USE_GATE_IN_KERNEL = 1`, `USE_CU_SEQLENS = 1` | static | eliminates the spec-decode slot search, the beta sigmoid, the separate initial-state pool, and the dense addressing path |
| `N`, `H`, `HV`, pool size | static for the TIRx launch | match the selected benchmark shape exactly |
| state slot stride | runtime `int64` | preserves the source's symbolic outer stride (divisibility 16), so envelope-strided pools work |
| gate batch and token strides | runtime `int64` | `gG` alone is compiled non-compact on those two axes (divisibility 16, head/K compact, `:1035-1047`) so a gate sliced from a fused projection needs no copy; q/k/v/o are fully compact |
| `scale`, `eps`, `lower_bound` | runtime `f32` kernel parameters | the source passes all three as kernel arguments, not immediates |
| `cu_seqlens`, `ssm_state_indices` values | runtime | preserve the source's activity predicates and negative-slot handling |

The host picks `(TILE_ROWS, DOT_REDUCTION_SCHEDULE)` from `sequence_heads`
exactly as `_select_kernel_schedule` does, and every correctness and benchmark
call asserts that the predicate selected the intended specialization and that
`sequence_heads >= 128` keeps the call inside the one-warp domain.

## TIRx module and benchmark contract

- `KERNEL_META` names `recurrent_kda_decode_one_warp`, category `flashinfer`,
  compute capability 10.
- All optional dependencies are lazy; `flashinfer` is imported only inside the
  reference builder and the correctness oracle.
- `CONFIGS` covers both tile schedules, both gate modes, the schedule-band
  boundaries at `sequence_heads` 176/192, negative-slot padding rows, and an
  envelope-strided state pool. `BENCH_CONFIGS` is the seven-row SGLang-parity
  plus Kimi-K3 matrix, all with `lower_bound = -5.0`.
- TIRx and the reference receive independent mutable state pools and output
  buffers; the state pool is updated in place, so each implementation gets its
  own clone of the same initial contents.
- Compilation, allocation, flashinfer JIT, warmup, and preflight validation all
  happen outside the timed closures.
- The implementation and every pre-dispatch specialization contain no tile
  primitives.

## Instruction-selection summary

- Each CTA is one warp with `.reqntid 32, 1, 1`, zero shared memory, zero
  mbarriers, and no `bar.sync` of any kind. Every cross-lane exchange is
  `shfl.sync.bfly.b32` or `shfl.sync.idx.b32` with member mask `0xFFFFFFFF` and
  clamp `31`.
- **All FP32 arithmetic is non-flush-to-zero**: `mul.f32`, `add.f32`,
  `sub.f32`, `fma.rn.f32`. Only the transcendentals are FTZ
  (`ex2.approx.ftz.f32`, `lg2.approx.ftz.f32`, `rsqrt.approx.ftz.f32`), and the
  sigmoid reciprocal is the full-precision `div.rn.f32`. Reproducing this split
  is required; substituting `.ftz` arithmetic or `rcp.approx` for the divide
  changes both numerics and instruction selection.
- Mixed-precision forms appear where a bf16 load register feeds an FP32 op
  directly: `add.rn.f32.bf16` for the `dt_bias + gate` add, and
  `fma.rn.f32.bf16` for the two q/k sum-of-squares terms each schedule
  contracts -- `q_src[2]`/`q_src[3]` under `DUAL_ACCUM` (`:390-391`) and
  `q_src[0]`/`q_src[2]` under `TREE` (`:399-400`). The complementary two terms
  stay `mul.f32` on the converted f32 values.
- Shuffle counts per token at `TILE_ROWS = 16`: 10 butterflies for the two
  full-warp L2 norms, 24 indexed shuffles for the q/k/gate broadcast, 8 indexed
  shuffles for the V redistribution, and 64 butterflies for the sixteen
  k-group dot reductions -- 74 `shfl.sync.bfly.b32` and 32
  `shfl.sync.idx.b32` in total. At `TILE_ROWS = 8` only the per-row groups
  halve: the k-group butterflies drop to 32 (42 total) and the V
  redistribution drops to 4, while the 24-shuffle q/k/gate broadcast does not
  scale with `TILE_ROWS` -- giving 42 `shfl.sync.bfly.b32` and 28
  `shfl.sync.idx.b32` (24 broadcast + 4 V).
- Memory traffic per CTA at `TILE_ROWS = 16`: 8 `ld.global.v4.b32` state loads,
  3 `ld.global.v4.b16` q/k/gate loads, 4 `ld.global.b32` dt_bias loads,
  4 more `ld.global.b32` for `A_log`/`cu_seqlens`/`ssm_state_indices`,
  2 `ld.global.b16` for beta and V, 9 `st.global.b16` (8 output rows plus the
  zero-pad prefill), and 8 `st.global.v4.b32` state stores.
- Conversions at `TILE_ROWS = 16`: 74 `cvt.f32.bf16` on the load side, 32
  `cvt.rn.bf16x2.f32` packing the state writeback two elements at a time, and 8
  `cvt.rn.bf16.f32` for the scalar output stores. At `TILE_ROWS = 8` these
  scale with `ROWS`: 42, 16, and 4.
