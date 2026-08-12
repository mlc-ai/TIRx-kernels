<!--
Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0

This design sketch documents a modified TIRx port of FlashInfer's
gdn_decode_bf16_state.py. See LICENSE, NOTICE, and licenses/ for the
applicable terms.
-->

# GDN decode BF16 wide-vector MTP SM100: execution sketch

This file is a non-executable execution sketch. It freezes the lane ownership,
shared publication, token-serial register recurrence, packed-FP32x2 arithmetic,
and 128-bit state traffic of FlashInfer's CuTeDSL `gdn_wide_vec_kernel` for the
multi-token path. The corresponding TIRx module is
[`tirx_kernels/flashinfer/gdn_decode/gdn_decode_bf16_wide_vec_mtp.py`](../../../../tirx_kernels/flashinfer/gdn_decode/gdn_decode_bf16_wide_vec_mtp.py).
That module may become executable only after this sketch passes independent
review.

The frozen FlashInfer commit is
`f2e04400e330fb2debe0bf8730d9424a1d37927f`; the source SHA256 is
`61de9ffa703962cb1ddb73823100550138708bbcbb535a3efcac608940e67e61`.
The exact source chain is
`gated_delta_rule_mtp -> gated_delta_rule_mtp_wide_vec -> _run_wide_vec ->
gdn_wide_vec_kernel`.

The target is SM100a/B200 and fixes K=V=128, T>=2, BF16 Q/K/V/state/output,
FP32 A_log/dt_bias/accumulators, 128 threads, four warps, eight 16-lane
subgroups, eight adjacent K elements per lane, four V rows of ILP, and packed
`fma.rn.f32x2`. T=1, K/V other than 128, FP32 state, the generic MTP ILP4
kernel, and tile primitives are out of scope.

Correctness retains tile_v in {32,64,128}; Q/K normalization on/off; output
on/off; state update on/off; dense batch-scoped intermediate caching; same and
split pools; contiguous and padded pool slots; negative read/write indices;
scalar recovery; per-request accepted steps; and flat or padded per-token pool
scatter. The production performance mode fixes Q/K L2 normalization on,
output on, state update off, dense intermediate caching on, same contiguous
pool, no recovery, and no accepted/scatter mode.

## Pipeline at a glance

There is no asynchronous copy pipeline, TensorMap, TMEM, or mbarrier. Dynamic
shared memory is published with one CTA barrier after each four-token
precompute pass. State is loaded once per four-row body, remains in registers
while tokens execute serially, and is written only by the selected semantic
mode.

| Physical lanes | Role-local program | Publication/reuse edge |
| --- | --- | --- |
| every warp, both 16-lane halves | pass `p` owns token `4*p+warp`; both halves redundantly load and transform the same eight Q/K elements, publish sQ/sK, and compute g/beta | scalar shared Q/K stores and one lane-zero packed g/beta store, then CTA `bar.sync 0` after every pass |
| all four warps when a pass is wholly inside scalar recovery | omit Q load, Q normalization, and sQ publication; retain K and gate publication | same per-pass CTA barrier; skipped sQ is never consumed by Phase A |
| groups 0..7 after the final publication barrier | each 16-lane group owns `tile_v/8` V rows; each lane owns K `[8*lane,8*lane+8)`; each static body carries four V rows | sQ/sK/sGB are immutable; four state rows remain in registers across every token in Phase A and Phase B |
| lane 0 of each 16-lane group | converts and writes four contiguous output rows for each Phase-B token | four scalar BF16 global stores in reviewed source PTX; disjoint V ownership |
| every lane | optionally writes four 16-byte state vectors after each token, at the recovery boundary, or at final writeback | disjoint `(V-row,K-segment)` ownership; dense cache, flat scatter, padded scatter, and final pool write are mutually selected |

## Primitive vocabulary

Structural operations do not move or compute values:

```python
specialize(...)       # compile-time source variant
launch(...)           # physical grid/block/shared metadata
raw_shared(...)       # one dynamic shared allocation
view(...)             # typed storage view without a copy
alias(...)            # exact read/write storage alias
reg_tile(...)         # lane-private register storage
```

Copies expose storage direction:

```python
copy_g2r(src, dst=None, cache=None, predicate=None)
copy_s2r(src, dst=None)
copy_r2s(src, dst, predicate=None)
copy_r2g(src, dst, predicate=None)
```

Computation and synchronization stay primitive:

```python
cast(dtype, src, rounding=None)
add(lhs, rhs)
sub(lhs, rhs)
mul(lhs, rhs)
fma(lhs, rhs, acc, lanes=1)
exp2(src)
log2(src)
reciprocal(src)
rsqrt(src)
setp_le(lhs, rhs)
select(predicate, true_value, false_value)
shuffle_xor(src, lane_delta, member_mask, clamp)
warp_uniform(src, source_lane, member_mask, clamp)
cta_barrier()
```

`fma(..., lanes=2)` is one packed two-lane FP32 instruction with two ordered
results. A stated eight-element copy is one explicit loop of the same scalar
instruction family; a four-row state copy is four distinct 128-bit operations.
Address arithmetic and view construction are not expanded unless they change a
runtime stride, pool slot, mode, or owner. There is deliberately no compound
`normalize`, `softplus`, `sigmoid`, `reduce`, `delta_rule`, `update_state`, or
`scatter_state` primitive.

## Legal source-specialization contract

The mode flags below describe only combinations accepted by the direct source
wrapper; the Cartesian product in `@specialize` is constrained by the
`@require` clauses and is not a promise that invalid products compile.

- `0 <= RECOVERY_STEPS <= T`. A positive recovery count requires state
  writeback and forbids dense intermediate caching.
- Per-token scatter requires `T>=2`, state writeback, no dense intermediate
  cache, and `RECOVERY_STEPS==0`. Its index tensor must be device-local int32
  with shape `[B,T]`.
- Per-request accepted steps require a device-local int32 tensor with shape
  `[B]`. They may coexist with scatter, but the per-request-fused boundary mode
  exists only when output and state writeback are enabled and scatter is off.
- `PER_TOKEN_POOL_SCATTER_FLAT` is selected only for a contiguous state pool;
  padded/strided pools select the 4-D `state` slot route and pass a dummy
  `intermediate` argument. Dense cache and flat scatter are the only modes that
  use `intermediate` storage.
- `SAME_POOL` is true only when write indices are omitted or alias read
  indices. Otherwise the source loads and clamps the split-pool write index.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, launch, and storage
# ===========================================================================

@specialize(
    T=(2..8),
    H=(16,8,4,2),
    HV=(32,16,8,4),
    K=128,
    V=128,
    TILE_V=(32,64,128),
    USE_QK_L2NORM=(False,True),
    DISABLE_STATE_UPDATE=(False,True),
    CACHE_INTERMEDIATE_STATES=(False,True),
    USE_PACKED_FMA=True,
    SAME_POOL=(False,True),
    DISABLE_OUTPUT=(False,True),
    RECOVERY_STEPS=(0..T),
    PER_REQUEST_ACCEPTED_STEPS=(False,True),
    PER_TOKEN_POOL_SCATTER=(False,True),
    PER_TOKEN_POOL_SCATTER_FLAT=(False,True),
    SOFTPLUS_BETA=1.0,
    SOFTPLUS_THRESHOLD=20.0,
    SCALE=1.0/sqrt(128.0),
    target="sm_100a",
)
@require(0 <= RECOVERY_STEPS <= T)
@require(
    RECOVERY_STEPS == 0
    or (not CACHE_INTERMEDIATE_STATES and not DISABLE_STATE_UPDATE))
@require(
    not PER_TOKEN_POOL_SCATTER
    or (
        T >= 2
        and not CACHE_INTERMEDIATE_STATES
        and not DISABLE_STATE_UPDATE
        and RECOVERY_STEPS == 0
    ))
@require(PER_TOKEN_POOL_SCATTER_FLAT implies PER_TOKEN_POOL_SCATTER)
@launch(
    grid=(batch * HV * (128 // TILE_V), 1, 1),
    block=(128, 1, 1),
    num_warps=4,
    dynamic_smem_bytes=1096*T + 256,
)
def gdn_decode_bf16_wide_vec_mtp(
    state,                  # bf16 [pool,HV,128,128], mutable, arbitrary slot stride
    intermediate,           # bf16 [B,T,HV,128,128], contiguous flat-pool alias, or dummy
    A_log,                  # f32  [HV]
    a,                      # bf16 [B,T,HV]
    dt_bias,                # f32  [HV]
    q, k,                   # bf16 [B,T,H,128], independent runtime batch strides
    v, b_gate,              # bf16 [B,T,HV,128], bf16 [B,T,HV]
    output,                 # bf16 [B,T,HV,128] or dummy
    read_indices,           # i32 [B]
    write_indices,          # i32 [B], dummy when SAME_POOL
    accepted_steps,         # i32 [B], dummy unless PER_REQUEST_ACCEPTED_STEPS
    ssm_state_indices,      # i32 [B,T], dummy unless PER_TOKEN_POOL_SCATTER
    state_slot_stride,      # i64 BF16 elements; permits padded 4-D pools
    q_batch_stride,         # i64 BF16 elements; permits packed QKV views
    k_batch_stride,         # i64 BF16 elements
    v_batch_stride,         # i64 BF16 elements
    batch,                  # i32 runtime launch extent
):
    NUM_WARPS = 4
    NUM_GROUPS = 8
    LANES_PER_GROUP = 16
    ELEMS_PER_LANE = 8
    ILP_ROWS = 4
    ROWS_PER_GROUP = TILE_V // NUM_GROUPS        # 4, 8, 16
    ITERS_PER_GROUP = ROWS_PER_GROUP // ILP_ROWS # 1, 2, 4
    NUM_V_TILES = 128 // TILE_V                   # 4, 2, 1
    PRECOMPUTE_PASSES = ceil_div(T, NUM_WARPS)

    tid = thread_id(axis="x", extent=128)
    warp_raw = tid // 32
    warp = warp_uniform(
        warp_raw, source_lane=0, member_mask=0xffffffff, clamp=31)
    # instruction_selection: shfl.sync.idx.b32; extent: one warp-index
    # broadcast from source lane 0 with full mask and clamp 31
    lane_in_warp = tid % 32
    group = tid // 16
    lane = tid % 16
    k_start = lane * 8

    linear_cta = cta_id(axis="x")
    v_tile = linear_cta % NUM_V_TILES
    tmp = linear_cta // NUM_V_TILES
    hv = tmp % HV
    n = tmp // HV
    h = hv // (HV // H)

    read_slot_raw = copy_g2r(read_indices[n])
    # instruction_selection: ld.global.b32; extent: one i32 pool index per CTA thread, compiler broadcast/hoisting permitted
    A_value = copy_g2r(A_log[hv])
    # instruction_selection: ld.global.b32; extent: one FP32 head scalar loaded
    # once during CTA setup and retained across every precompute pass
    dt_value = copy_g2r(dt_bias[hv])
    # instruction_selection: ld.global.b32; extent: one FP32 head scalar loaded
    # once during CTA setup and retained across every precompute pass
    read_slot = select(read_slot_raw < 0, 0, read_slot_raw)
    # instruction_selection: max.s32; extent: one null-slot redirect

    if SAME_POOL:
        write_slot = read_slot
        write_state = alias(state[read_slot], state[read_slot])
    else:
        write_slot_raw = copy_g2r(write_indices[n])
        # instruction_selection: ld.global.b32; extent: one i32 split-pool index when a surviving pool write needs it
        write_slot = select(write_slot_raw < 0, 0, write_slot_raw)
        # instruction_selection: max.s32; extent: one null-slot redirect
        write_state = state[cast("i64", write_slot), hv, :, :]

    read_state = state[cast("i64", read_slot), hv, :, :]
    # i64 slot arithmetic preserves caller-supplied padded state_slot_stride.

    if PER_REQUEST_ACCEPTED_STEPS:
        accepted = copy_g2r(accepted_steps[n])
        # instruction_selection: ld.global.b32; extent: one runtime accepted-step bound per CTA thread
    else:
        accepted = T - 1

    PER_REQUEST_FUSED = (
        PER_REQUEST_ACCEPTED_STEPS
        and not DISABLE_OUTPUT
        and not DISABLE_STATE_UPDATE
        and not PER_TOKEN_POOL_SCATTER
    )
    if PER_REQUEST_FUSED:
        phase_a_bound = accepted + 1
        phase_b_begin = accepted + 1
        phase_b_bound = T - accepted - 1
    elif PER_REQUEST_ACCEPTED_STEPS:
        phase_a_bound = RECOVERY_STEPS
        phase_b_begin = RECOVERY_STEPS
        phase_b_bound = accepted + 1 - RECOVERY_STEPS
    else:
        phase_a_bound = RECOVERY_STEPS
        phase_b_begin = RECOVERY_STEPS
        phase_b_bound = T - RECOVERY_STEPS

    # CuTe allocates two stride-(136,1) FP32 arrays followed by a dense (T,2)
    # array, all 16-byte aligned. cosize((T,128),(136,1))*4 = 544*T-32.
    smem = raw_shared(dtype="u8", bytes=1096*T+256, alignment=16)
    sQ = view(
        smem[0 : 544*T-32], dtype="f32", shape=(T,128),
        stride=(136,1), alignment=16)
    sK = view(
        smem[544*T-32 : 1088*T-64], dtype="f32", shape=(T,128),
        stride=(136,1), alignment=16)
    sGB = view(
        smem[1088*T-64 : 1096*T-64], dtype="f32", shape=(T,2),
        stride=(2,1), alignment=16)
    # The remaining 320 bytes are allocator/launcher reservation tail.

    r_q_bf16 = reg_tile(dtype="bf16", shape=(8,))
    r_k_bf16 = reg_tile(dtype="bf16", shape=(8,))
    r_q = reg_tile(dtype="f32", shape=(8,))
    r_k = reg_tile(dtype="f32", shape=(8,))
    r_h_bf16 = reg_tile(dtype="bf16", shape=(4,8))
    r_h = reg_tile(dtype="f32", shape=(4,8))
    r_o_bf16 = reg_tile(dtype="bf16", shape=(4,)) if not DISABLE_OUTPUT else None

    # =======================================================================
    # Optional tile-32 state prefetch before Phase 0
    # =======================================================================

    if TILE_V == 32:
        pre_v_base = v_tile*TILE_V + group*ROWS_PER_GROUP
        for row in static_range(4):
            copy_g2r(
                read_state[pre_v_base+row, k_start:k_start+8],
                r_h_bf16[row], cache="L1::evict_first")
            # instruction_selection: ld.global.L1::evict_first.v4.b32;
            # extent: one 16-byte BF16 state vector, four explicit rows

    # =======================================================================
    # Phase 0: all four warps precompute token-shared Q/K/g/beta
    # =======================================================================

    member_pre = lane_in_warp % 16
    k_pre = member_pre * 8
    for pass_index in static_range(PRECOMPUTE_PASSES):
        t_pre = pass_index*4 + warp
        all_recovery_pass = (pass_index+1)*4 <= RECOVERY_STEPS
        do_q_pass = (not DISABLE_OUTPUT) and (not all_recovery_pass)

        if t_pre < T:
            if do_q_pass:
                for i in static_range(8):
                    r_q_bf16[i] = copy_g2r(q[n,t_pre,h,k_pre+i])
                    # instruction_selection: ld.global.b16; extent: one scalar Q load, eight explicit loop instances per half-warp
            for i in static_range(8):
                r_k_bf16[i] = copy_g2r(k[n,t_pre,h,k_pre+i])
                # instruction_selection: ld.global.b16; extent: one scalar K load, eight explicit loop instances per half-warp

            if do_q_pass:
                for i in static_range(8):
                    r_q[i] = cast("f32", r_q_bf16[i])
                    # instruction_selection: cvt.f32.bf16; extent: one scalar, eight loop instances
            for i in static_range(8):
                r_k[i] = cast("f32", r_k_bf16[i])
                # instruction_selection: cvt.f32.bf16; extent: one scalar, eight loop instances

            if USE_QK_L2NORM:
                sum_k = 0.0
                for i in static_range(8):
                    sum_k = fma(r_k_bf16[i], r_k_bf16[i], sum_k)
                    # instruction_selection: fma.rn.f32.bf16; extent: one ordered scalar square-accumulate, eight loop instances
                for delta in (8,4,2,1):
                    peer_k = shuffle_xor(sum_k, delta, member_mask=-1, clamp=31)
                    # instruction_selection: shfl.sync.bfly.b32; extent: one scalar at each of four explicit stages
                    sum_k = add(sum_k, peer_k)
                    # instruction_selection: add.f32; extent: one scalar at each stage
                k_factor = rsqrt(add(sum_k, 1.0e-6))
                # instruction_selection: add.f32 then rsqrt.approx.ftz.f32; extent: one scalar each
                for i in static_range(8):
                    r_k[i] = mul(r_k[i], k_factor)
                    # instruction_selection: mul.f32; extent: one scalar, eight loop instances

                if do_q_pass:
                    sum_q = 0.0
                    for i in static_range(8):
                        sum_q = fma(r_q_bf16[i], r_q_bf16[i], sum_q)
                        # instruction_selection: fma.rn.f32.bf16; extent: one ordered scalar square-accumulate, eight loop instances
                    for delta in (8,4,2,1):
                        peer_q = shuffle_xor(sum_q, delta, member_mask=-1, clamp=31)
                        # instruction_selection: shfl.sync.bfly.b32; extent: one scalar at each of four explicit stages
                        sum_q = add(sum_q, peer_q)
                        # instruction_selection: add.f32; extent: one scalar at each stage
                    q_factor = mul(rsqrt(add(sum_q, 1.0e-6)), SCALE)
                    # instruction_selection: add.f32, rsqrt.approx.ftz.f32, mul.f32; extent: one scalar each
                    for i in static_range(8):
                        r_q[i] = mul(r_q[i], q_factor)
                        # instruction_selection: mul.f32; extent: one scalar, eight loop instances
            elif do_q_pass:
                for i in static_range(8):
                    r_q[i] = mul(r_q[i], SCALE)
                    # instruction_selection: mul.f32; extent: one scalar, eight loop instances

            if do_q_pass:
                for i in static_range(8):
                    copy_r2s(r_q[i], sQ[t_pre,k_pre+i])
                    # instruction_selection: st.shared.b32; extent: one scalar, eight explicit loop instances; duplicate half-warps are idempotent
            for i in static_range(8):
                copy_r2s(r_k[i], sK[t_pre,k_pre+i])
                # instruction_selection: st.shared.b32; extent: one scalar, eight explicit loop instances; duplicate half-warps are idempotent

            a_value = copy_g2r(a[n,t_pre,hv])
            # instruction_selection: ld.global.b16; extent: one scalar BF16 gate input per active thread
            x = add(cast("f32", a_value), dt_value)
            # instruction_selection: add.rn.f32.bf16; extent: one mixed scalar add
            beta_x = mul(SOFTPLUS_BETA, x)
            # instruction_selection: constexpr identity at beta=1; otherwise mul.f32; extent: one scalar
            exp_beta_x = exp2(mul(beta_x, LOG2E))
            # instruction_selection: mul.f32 then ex2.approx.ftz.f32; extent: one scalar each
            softplus_log = log2(add(1.0, exp_beta_x))
            # instruction_selection: add.f32 then lg2.approx.ftz.f32; extent: one scalar each
            softplus_value = mul(mul(reciprocal(SOFTPLUS_BETA), softplus_log), LN2)
            # instruction_selection: constexpr fold at beta=1, otherwise rcp/mul; extent: scalar chain
            softplus_pred = setp_le(beta_x, SOFTPLUS_THRESHOLD)
            # instruction_selection: setp.le.f32; extent: one scalar predicate
            softplus_weight = select(softplus_pred, 1.0, 0.0)
            # instruction_selection: selp.f32; extent: one scalar
            direct_weight = sub(1.0, softplus_weight)
            # instruction_selection: sub.f32; extent: one scalar
            softplus_x = fma(
                softplus_value, softplus_weight, mul(x, direct_weight))
            # instruction_selection: mul.f32 then fma.rn.f32; extent: one scalar each

            exp_A = exp2(mul(A_value, LOG2E))
            # instruction_selection: mul.f32 then ex2.approx.ftz.f32; extent: one scalar each
            gate_exponent = mul(sub(0.0, exp_A), softplus_x)
            # instruction_selection: neg.f32/mul.f32; extent: one scalar each

            if lane_in_warp == 0:
                b_value = cast("f32", copy_g2r(b_gate[n,t_pre,hv]))
                # instruction_selection: ld.global.b16 then cvt.f32.bf16; extent: one scalar each
                exp_neg_b = exp2(mul(b_value, -LOG2E))
                # instruction_selection: mul.f32 then ex2.approx.ftz.f32; extent: one scalar each
                beta = reciprocal(add(1.0, exp_neg_b))
                # instruction_selection: add.f32 then rcp.rn.f32; extent: one scalar each
                g = exp2(mul(gate_exponent, LOG2E))
                # instruction_selection: mul.f32 then ex2.approx.ftz.f32; extent: one scalar each
                copy_r2s((g,beta), sGB[t_pre,0:2])
                # instruction_selection: st.shared.v2.b32; extent: one 8-byte pair by physical lane 0

        cta_barrier()
        # instruction_selection: bar.sync 0; extent: one CTA-wide publication point per precompute pass

    # =======================================================================
    # Phase 1: static four-row bodies, token-serial recurrence in registers
    # =======================================================================

    for iter_index in static_range(ITERS_PER_GROUP):
        v_base = v_tile*TILE_V + group*ROWS_PER_GROUP + iter_index*4
        v_rows = (v_base+0, v_base+1, v_base+2, v_base+3)

        if not (TILE_V == 32 and iter_index == 0):
            for row in static_range(4):
                copy_g2r(
                    read_state[v_rows[row],k_start:k_start+8],
                    r_h_bf16[row], cache="L1::evict_first")
                # instruction_selection: ld.global.L1::evict_first.v4.b32;
                # extent: one 16-byte BF16 state vector, four explicit rows
        for element in static_range(8):
            for row in static_range(4):
                r_h[row,element] = cast("f32", r_h_bf16[row,element])
                # instruction_selection: cvt.f32.bf16; extent: one scalar, 32 element-major loop instances

        # -------------------------------------------------------------------
        # Phase A token body: recurrence only, no Q load and no output
        # -------------------------------------------------------------------
        for t in runtime_range(phase_a_bound, unroll=1):
            g, beta = copy_s2r(sGB[t,0:2])
            # instruction_selection: ld.shared.v2.b32; extent: one 8-byte
            # vector containing the token's g and beta

            dot_hk = reg_tile(dtype="f32", shape=(4,), init=0.0)
            k_pairs = reg_tile(dtype="f32", shape=(4,2))
            for pair in static_range(4):
                k0 = copy_s2r(sK[t,k_start+2*pair+0])
                # instruction_selection: ld.shared.b32; extent: one scalar, four explicit pair instances
                k1 = copy_s2r(sK[t,k_start+2*pair+1])
                # instruction_selection: ld.shared.b32; extent: one scalar, four explicit pair instances
                k_pairs[pair] = (k0,k1)
                for row in static_range(4):
                    r_h[row,2*pair:2*pair+2] = fma(
                        r_h[row,2*pair:2*pair+2], (g,g), (0.0,0.0), lanes=2)
                    # instruction_selection: fma.rn.f32x2; extent: one packed decay, four rows per pair
                    dot_hk[row] = fma(r_h[row,2*pair+0], k0, dot_hk[row])
                    # instruction_selection: fma.rn.f32; extent: one ordered scalar h-k accumulate
                    dot_hk[row] = fma(r_h[row,2*pair+1], k1, dot_hk[row])
                    # instruction_selection: fma.rn.f32; extent: one ordered scalar h-k accumulate

            for delta in (8,4,2,1):
                for row in static_range(4):
                    peer = shuffle_xor(dot_hk[row], delta, member_mask=-1, clamp=31)
                    # instruction_selection: shfl.sync.bfly.b32; extent: one scalar, four rows at each explicit stage
                    dot_hk[row] = add(dot_hk[row], peer)
                    # instruction_selection: add.f32; extent: one scalar, four rows at each stage

            value = reg_tile(dtype="f32", shape=(4,))
            for row in static_range(4):
                v_bf16 = copy_g2r(v[n,t,hv,v_rows[row]])
                # instruction_selection: ld.global.b16; extent: one scalar BF16 V value, four rows
                residual = sub(v_bf16, dot_hk[row])
                # instruction_selection: sub.rn.f32.bf16; extent: one mixed BF16-FP32 scalar subtract, four rows
                value[row] = mul(residual, beta)
                # instruction_selection: mul.f32; extent: one scalar, four rows

            for pair in static_range(4):
                k0, k1 = k_pairs[pair]  # register aliases; no second shared load
                for row in static_range(4):
                    r_h[row,2*pair:2*pair+2] = fma(
                        (k0,k1), (value[row],value[row]),
                        r_h[row,2*pair:2*pair+2], lanes=2)
                    # instruction_selection: fma.rn.f32x2; extent: one packed rank-one update, four rows per pair

            # The source contains a Phase-A dense-cache branch, but it is dead
            # under every legal direct-wrapper specialization: positive
            # recovery forbids caching, and caching disables the state update
            # needed to select per-request-fused Phase A. It emits no normal-
            # path PTX and is intentionally absent from this execution body.

        # -------------------------------------------------------------------
        # Recovery/accepted boundary write of h_K before Phase B
        # -------------------------------------------------------------------
        if ((RECOVERY_STEPS > 0 or PER_REQUEST_FUSED)
                and not DISABLE_STATE_UPDATE
                and not CACHE_INTERMEDIATE_STATES):
            for row in static_range(4):
                r_h_bf16[row] = cast("bf16x8", r_h[row], rounding="rn")
                # instruction_selection: cvt.rn.bf16x2.f32; extent: four pair conversions for one row
            for row in static_range(4):
                copy_r2g(r_h_bf16[row], write_state[v_rows[row],k_start:k_start+8])
                # instruction_selection: st.global.v4.b32; extent: one fire-and-forget 16-byte state vector, four rows

        # -------------------------------------------------------------------
        # Phase B token body: recurrence, output, and per-token state route
        # -------------------------------------------------------------------
        for t_offset in runtime_range(phase_b_bound, unroll=1):
            t = phase_b_begin + t_offset
            g, beta = copy_s2r(sGB[t,0:2])
            # instruction_selection: ld.shared.v2.b32; extent: one 8-byte
            # vector containing the token's g and beta

            dot_hk = reg_tile(dtype="f32", shape=(4,), init=0.0)
            k_pairs = reg_tile(dtype="f32", shape=(4,2))
            for pair in static_range(4):
                k0 = copy_s2r(sK[t,k_start+2*pair+0])
                # instruction_selection: ld.shared.b32; extent: one scalar, four explicit pair instances
                k1 = copy_s2r(sK[t,k_start+2*pair+1])
                # instruction_selection: ld.shared.b32; extent: one scalar, four explicit pair instances
                k_pairs[pair] = (k0,k1)
                for row in static_range(4):
                    r_h[row,2*pair:2*pair+2] = fma(
                        r_h[row,2*pair:2*pair+2], (g,g), (0.0,0.0), lanes=2)
                    # instruction_selection: fma.rn.f32x2; extent: one packed decay, four rows per pair
                    dot_hk[row] = fma(r_h[row,2*pair+0], k0, dot_hk[row])
                    # instruction_selection: fma.rn.f32; extent: one ordered scalar h-k accumulate
                    dot_hk[row] = fma(r_h[row,2*pair+1], k1, dot_hk[row])
                    # instruction_selection: fma.rn.f32; extent: one ordered scalar h-k accumulate

            for delta in (8,4,2,1):
                for row in static_range(4):
                    peer = shuffle_xor(dot_hk[row], delta, member_mask=-1, clamp=31)
                    # instruction_selection: shfl.sync.bfly.b32; extent: one scalar, four rows at each explicit stage
                    dot_hk[row] = add(dot_hk[row], peer)
                    # instruction_selection: add.f32; extent: one scalar, four rows at each stage

            value = reg_tile(dtype="f32", shape=(4,))
            for row in static_range(4):
                v_bf16 = copy_g2r(v[n,t,hv,v_rows[row]])
                # instruction_selection: ld.global.b16; extent: one scalar BF16 V value, four rows
                residual = sub(v_bf16, dot_hk[row])
                # instruction_selection: sub.rn.f32.bf16; extent: one mixed BF16-FP32 scalar subtract, four rows
                value[row] = mul(residual, beta)
                # instruction_selection: mul.f32; extent: one scalar, four rows

            dot_hq = reg_tile(dtype="f32", shape=(4,), init=0.0)
            for pair in static_range(4):
                k0, k1 = k_pairs[pair]  # register aliases; no second shared load
                if not DISABLE_OUTPUT:
                    q0 = copy_s2r(sQ[t,k_start+2*pair+0])
                    # instruction_selection: ld.shared.b32; extent: one scalar, four explicit pair instances
                    q1 = copy_s2r(sQ[t,k_start+2*pair+1])
                    # instruction_selection: ld.shared.b32; extent: one scalar, four explicit pair instances
                for row in static_range(4):
                    r_h[row,2*pair:2*pair+2] = fma(
                        (k0,k1), (value[row],value[row]),
                        r_h[row,2*pair:2*pair+2], lanes=2)
                    # instruction_selection: fma.rn.f32x2; extent: one packed rank-one update, four rows per pair
                    if not DISABLE_OUTPUT:
                        dot_hq[row] = fma(r_h[row,2*pair+0], q0, dot_hq[row])
                        # instruction_selection: fma.rn.f32; extent: one ordered scalar h-q accumulate
                        dot_hq[row] = fma(r_h[row,2*pair+1], q1, dot_hq[row])
                        # instruction_selection: fma.rn.f32; extent: one ordered scalar h-q accumulate

            if not DISABLE_OUTPUT:
                for delta in (8,4,2,1):
                    for row in static_range(4):
                        peer = shuffle_xor(dot_hq[row], delta, member_mask=-1, clamp=31)
                        # instruction_selection: shfl.sync.bfly.b32; extent: one scalar, four rows at each explicit stage
                        dot_hq[row] = add(dot_hq[row], peer)
                        # instruction_selection: add.f32; extent: one scalar, four rows at each stage
                if lane == 0:
                    for row in static_range(4):
                        r_o_bf16[row] = cast("bf16", dot_hq[row], rounding="rn")
                        # instruction_selection: cvt.rn.bf16.f32; extent: one scalar, four rows
                    for row in static_range(4):
                        copy_r2g(r_o_bf16[row], output[n,t,hv,v_rows[row]])
                        # instruction_selection: st.global.b16; extent: one scalar BF16 output, four rows

            if CACHE_INTERMEDIATE_STATES or PER_TOKEN_POOL_SCATTER:
                for row in static_range(4):
                    r_h_bf16[row] = cast("bf16x8", r_h[row], rounding="rn")
                    # instruction_selection: cvt.rn.bf16x2.f32; extent: four pair conversions for one row

            if PER_TOKEN_POOL_SCATTER:
                pool_slot_t = copy_g2r(ssm_state_indices[n,t])
                # instruction_selection: ld.global.b32; extent: one per-token pool slot per CTA thread
                if PER_TOKEN_POOL_SCATTER_FLAT:
                    flat_index = cast("i64", pool_slot_t)*HV + hv
                    for row in static_range(4):
                        copy_r2g(
                            r_h_bf16[row],
                            intermediate[flat_index,v_rows[row],k_start:k_start+8])
                        # instruction_selection: st.global.v4.b32; extent: one 16-byte state vector, four rows, i64 flat index
                else:
                    scatter_state = state[cast("i64", pool_slot_t),hv,:,:]
                    for row in static_range(4):
                        copy_r2g(
                            r_h_bf16[row],
                            scatter_state[v_rows[row],k_start:k_start+8])
                        # instruction_selection: st.global.v4.b32; extent: one 16-byte state vector, four rows, padded slot stride retained
            elif CACHE_INTERMEDIATE_STATES:
                dense_index = cast("i64", n*T*HV + t*HV + hv)
                for row in static_range(4):
                    copy_r2g(
                        r_h_bf16[row],
                        intermediate[dense_index,v_rows[row],k_start:k_start+8])
                    # instruction_selection: st.global.v4.b32; extent: one 16-byte state vector, four rows, batch-scoped i64 flat index

        # -------------------------------------------------------------------
        # Final pool write, suppressed by cache/scatter/disable semantics
        # -------------------------------------------------------------------
        if (not DISABLE_STATE_UPDATE
                and not CACHE_INTERMEDIATE_STATES
                and RECOVERY_STEPS == 0
                and not PER_REQUEST_FUSED
                and not (PER_TOKEN_POOL_SCATTER and SAME_POOL)):
            for row in static_range(4):
                r_h_bf16[row] = cast("bf16x8", r_h[row], rounding="rn")
                # instruction_selection: cvt.rn.bf16x2.f32; extent: four pair conversions for one row
            for row in static_range(4):
                copy_r2g(r_h_bf16[row], write_state[v_rows[row],k_start:k_start+8])
                # instruction_selection: st.global.v4.b32; extent: one 16-byte final state vector, four rows
```

## Storage ownership and lifetimes

| storage | logical owner | lifetime and alias rule |
| --- | --- | --- |
| sQ `[T,128]`, stride 136 | both 16-lane halves of token-owning warp publish identical values | a token is live after its pass barrier through every Phase-B body; all-recovery passes leave sQ dead and uninitialized by design |
| sK `[T,128]`, stride 136 | both halves of token-owning warp publish identical values | every token is live after its pass barrier through all Phase-A/Phase-B bodies |
| sGB `[T,2]` | lane 0 of the token-owning warp publishes `(g,beta)` | immutable after the pass barrier and consumed by all groups |
| r_h `[4,8]` | one lane in one 16-lane group | one static four-row body; loaded once and carried across all serial tokens before any final write |
| r_h_bf16 `[4,8]` | exact source family `r_hb0..3`, one lane in one group | receives the initial four state vectors, is overwritten by all-four-row BF16 packing, and is reused as the sole source for boundary, dense cache, flat/padded scatter, and final pool stores |
| read/write pool views | one CTA, disjoint V tile and K segment | exact alias under SAME_POOL; otherwise distinct i64 slot bases; negative indices redirect to slot zero |
| dense intermediate | one `(n,t,hv,V,K)` region | batch-scoped and indexed by `n`, never the read pool slot; every token state may be written |
| flat per-token scatter | one `(pool_slot_t,hv,V,K)` region | uses i64 `(pool_slot_t*HV+hv)` and aliases a contiguous pool view |
| padded per-token scatter | one caller-strided pool slot | preserves the 4-D slot stride and does not flatten through padding |

No data race depends on scheduling. Duplicate Phase-0 half-warps write
bit-identical Q/K values, lane zero uniquely writes each g/beta pair, each pass
ends in a CTA barrier, and Phase 1 assigns every `(V,K)` state element to one
lane. Only subgroup lane zero writes each output V element.

## Control-flow modes

| mode | Phase A | Phase B | state destination |
| --- | --- | --- | --- |
| production cache | empty | all T tokens, output on | dense batch-scoped intermediate after every token; pool unchanged |
| state-only | empty | all T tokens, output compile-time removed | final same/split pool state unless cache enabled |
| scalar recovery K | first K tokens, no output | remaining T-K tokens | h_K boundary write is the only pool write; rejected suffix affects output but does not overwrite h_K |
| per-request fused accepted | runtime first `accepted[n]+1` tokens, no output | rejected suffix | h_K boundary write per request is the only pool write; rejected suffix does not overwrite h_K |
| per-request non-fused | scalar recovery prefix | runtime accepted suffix only | ordinary final/cache route |
| per-token pool scatter | empty scalar phase by contract | accepted token range | every h_{t+1} to `ssm_state_indices[n,t]`; same-pool final write suppressed |

## Frozen source-PTX evidence

Fresh source builds are preserved under
`.porting/gdn_decode_bf16_wide_vec_mtp/source_ptx_stage2/`; its
[`manifest.md`](../../../../.porting/gdn_decode_bf16_wide_vec_mtp/source_ptx_stage2/manifest.md)
records the exact variants, toolchain, artifact hashes, register declarations,
and source-to-PTX mapping. All PTX declares `.target sm_100a`,
`.reqntid 128,1,1`, dynamic shared memory, and `.file 1` pointing to the frozen
CuTeDSL source.

| evidence variant | property frozen by reviewed PTX |
| --- | --- |
| T2/tile32 production | one precompute barrier, state hoist before Phase 0, one static four-row body, output plus dense per-token state writes |
| T4/tile64 production | one precompute barrier and two static four-row bodies |
| T8/tile128 production | two precompute barriers and four static four-row bodies; sK begins at byte 4320 and sGB at byte 8640 |
| recovery4/split/tile128 | Q producer absent for the all-recovery pass, Phase-A recurrence, boundary 128-bit state writes, then output Phase B |
| mixed accepted/tile64 | runtime per-request Phase-A and Phase-B bounds plus boundary write |
| flat scatter/tile32 | per-token i32 slot loads widened into flat i64 pool addresses and 128-bit writes |
| L2-off/state-only/tile32 | Q path and both output dot/store families removed; no rsqrt normalization |

## TIRx module and benchmark contract

- Registry name: `gdn_decode_bf16_wide_vec_mtp`; supported compute capability
  is exactly 10.
- Correctness calls FlashInfer's direct
  `gated_delta_rule_mtp_wide_vec(..., tile_v=...)` oracle so all semantic
  branches and all three tile sizes execute this body.
- Correctness covers T=2..8, each Qwen3-Next `(H,HV)` pair, tile thresholds,
  same/split and contiguous/padded pools, negative indices, L2 on/off, output
  and update suppression, dense cache, scalar recovery, mixed accepted steps,
  and flat/padded scatter.
- Performance contains exactly 78 production workloads: T in {2,4,8},
  `(H,HV)` in {(16,32),(8,16),(4,8),(2,4)}, and B in
  {1,4,8,16,32,64,128,256,512}, filtered by `B*HV>=128`. Tile selection is
  32 for `[128,512)`, 64 for `[512,1024)`, and 128 for `>=1024`.
- Performance mode fixes Q/K L2 on, output on, state update disabled, dense
  batch-scoped intermediate cache on, same contiguous pool, no recovery,
  no accepted/scatter mode.
- Acceptance requires all 78 `source_time / tirx_time > 0.99` in each of five
  complete filtered bench-suite rounds.

## Instruction selection is a lowering consequence

The port must state the same placement, lane layout, vector width, static body
count, serial token recurrence, and synchronization before relying on opcodes:

- Q/K use eight scalar `ld.global.b16` operations per active Phase-0 thread;
  V uses four scalar BF16 loads per four-row/token body. State uses four
  `ld.global.L1::evict_first.v4.b32` loads and mode-selected
  `st.global.v4.b32` stores.
- Shared Q/K publication uses scalar `st.shared.b32`; g/beta publication uses
  `st.shared.v2.b32`; each precompute pass ends in `bar.sync 0`.
- All 16-lane reductions use `shfl.sync.bfly.b32` at offsets 8,4,2,1 with clamp
  31, intentionally creating two independent reductions per physical warp.
- Decay and rank-one update use `fma.rn.f32x2`; dot products remain ordered
  scalar `fma.rn.f32` chains.
- L2 and gate math use `rsqrt.approx.ftz.f32`, `ex2.approx.ftz.f32`,
  `lg2.approx.ftz.f32`, and `rcp.rn.f32`.
- Output is four lane-zero scalar BF16 conversions/stores in fresh source PTX;
  state is four BF16x2 conversions followed by one 128-bit store per row.
- `ITERS_PER_GROUP` is physically static: one body at tile 32, two at tile 64,
  and four at tile 128. A serial runtime V-row loop is not a faithful port.

The implementation may express these primitives through plain TIRx helpers,
but may not use tile primitives or substitute a different thread mapping,
reduction topology, memory width, fast-math sequence, packed-FMA schedule,
shared-publication protocol, or state lifetime.
