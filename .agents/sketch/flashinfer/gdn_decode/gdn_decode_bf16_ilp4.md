<!--
Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0

This design sketch documents a modified TIRx port of FlashInfer's
gdn_decode_bf16_state.py. See LICENSE, NOTICE, and licenses/ for the
applicable terms.
-->

# GDN decode BF16 ILP4 SM100: execution sketch

This file is a non-executable execution sketch. It freezes the four-warp
ownership, the T-dependent shared precompute, the four-row register recurrence,
the packed-FP32x2 dot/update arithmetic, and the 64-bit state addressing of
FlashInfer's CuTeDSL `gdn_decode_bf16state_mtp_ilp4_kernel`. The corresponding
TIRx module is
[`tirx_kernels/flashinfer/gdn_decode/gdn_decode_bf16_ilp4.py`](../../../../tirx_kernels/flashinfer/gdn_decode/gdn_decode_bf16_ilp4.py).
That module may become executable only after this sketch passes independent
review.

The frozen FlashInfer commit is
`f2e04400e330fb2debe0bf8730d9424a1d37927f`; the source SHA256 is
`61de9ffa703962cb1ddb73823100550138708bbcbb535a3efcac608940e67e61`.
The exact source chain is the T=1 fallback or T>=2 low-work-unit fallback into
`gated_delta_rule_mtp -> run_gdn_decode_bf16state_mtp_ilp4 ->
gdn_decode_bf16state_mtp_ilp4_kernel`.

The target is SM100a/B200 and fixes K=V=128, BF16 Q/K/V/state/output, FP32
head scalars/state/reductions, 128 threads, four warps, one 32-lane group per
warp, four adjacent K elements per lane, four V rows of ILP, and packed
`fma.rn.f32x2`. The production domain is T=1 with `B*HV<512` and T>=2 with
`B*HV<128`; the 48 power-of-two-batch benchmark cases use tile 16 or 32, while
the reachable T=1 fallback also uses tile 64 once its source occupancy picker
sees at least four SM waves (for example B=96, HV=4 on the 148-SM target).
Recovery and tile primitives are out of scope.

Correctness retains T=1 and T>1 branches; Q/K normalization on/off; output and
state update on/off; same/split and contiguous/padded pools; negative indices;
dense intermediate caching; per-request accepted steps; and flat/padded
per-token scatter.

## Pipeline at a glance

There is no asynchronous copy pipeline, TensorMap, TMEM, or mbarrier. For T>1,
the four warps publish token-shared Q/K and optionally g/beta into dynamic
shared memory, with one CTA barrier after each four-token pass. T=1 allocates
no typed shared tensors and performs Q/K/gate work inside every four-row body.

| Physical lanes | Role-local program | Publication/reuse edge |
| --- | --- | --- |
| warp `w` for T>1 | precompute tokens `4*p+w`; load four Q/K values per lane, normalize/scale, publish sQ/sK, and for T>2 publish g/beta from lane 0 | scalar shared Q/K stores, one lane-zero packed g/beta store, then `bar.sync 0` after every pass |
| all four warps for T=2 | publish only Q/K in Phase 0; recompute g/beta inside every four-row/token body | the same per-pass CTA barrier protects sQ/sK |
| all four warps for T=1 | skip Phase 0 and all shared traffic; every four-row body loads and transforms Q/K and g/beta directly | no publication edge |
| warp/group 0..3 in the row-quad loop | each warp owns `tile_v/4` V rows; each lane owns K `[4*lane,4*lane+4)`; every body carries four V rows across its serial token loop | register state is private; no post-precompute synchronization |
| lane 0 of each warp | writes four scalar BF16 output values per processed token/body | disjoint V ownership |
| every lane | writes four 8-byte BF16 state vectors to dense cache, scatter pool, or final pool as selected | disjoint `(V-row,K-segment)` ownership |

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

Copies expose storage direction and one PTX family per occurrence:

```python
copy_g2r(src, dst=None)
copy_s2r(src, dst=None)
copy_r2s(src, dst)
copy_r2g(src, dst)
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

`fma(..., lanes=2)` denotes one packed two-result FP32 instruction. A copy is
written as an explicit scalar loop when PTX emits four scalar operations; a
four-element copy remains one operation only where line-info PTX proves a
single vector instruction family. Address arithmetic is omitted unless it changes runtime
stride, slot selection, ownership, or the dense/flat/padded destination. There
is deliberately no compound `normalize`, `softplus`, `sigmoid`, `reduce`,
`delta_rule`, `update_state`, or `scatter_state` primitive.

## Legal source-specialization contract

- `T>=1`, `K=V=128`, and `tile_v` is a multiple of 16 dividing 128; the target
  production matrix uses tile 16/32 and the reachable T=1 fallback also uses
  tile 64.
- Per-token scatter requires T>=2, state update enabled, no dense cache, and no
  recovery. Flat scatter is selected only for a contiguous pool.
- Per-request accepted steps use a device-local int32 `[B]` tensor and make the
  token loop bound `accepted[n]+1`; without it the loop bound is constexpr T.
- Dense cache is batch-scoped and indexed by `n`, never by a pool slot.
- SAME_POOL aliases write addressing to read addressing; a split write index is
  loaded and clamped only when SAME_POOL is false.
- `recovery_steps>0` is rejected before this source kernel is selected.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, launch, and storage
# ===========================================================================

@specialize(
    T=(1..8),
    H=(16,8,4,2),
    HV=(32,16,8,4),
    K=128,
    V=128,
    TILE_V=(16,32,64),
    USE_QK_L2NORM=(False,True),
    DISABLE_STATE_UPDATE=(False,True),
    CACHE_INTERMEDIATE_STATES=(False,True),
    USE_PACKED_FMA=True,
    SAME_POOL=(False,True),
    DISABLE_OUTPUT=(False,True),
    PER_REQUEST_ACCEPTED_STEPS=(False,True),
    PER_TOKEN_POOL_SCATTER=(False,True),
    PER_TOKEN_POOL_SCATTER_FLAT=(False,True),
    SOFTPLUS_BETA=1.0,
    SOFTPLUS_THRESHOLD=20.0,
    SCALE=1.0/sqrt(128.0),
    target="sm_100a",
)
@require(not PER_TOKEN_POOL_SCATTER or
         (T >= 2 and not CACHE_INTERMEDIATE_STATES and
          not DISABLE_STATE_UPDATE))
@require(PER_TOKEN_POOL_SCATTER_FLAT implies PER_TOKEN_POOL_SCATTER)
@launch(
    grid=(batch * HV * (128 // TILE_V), 1, 1),
    block=(128, 1, 1),
    num_warps=4,
    dynamic_smem_bytes=(128 if T == 1 else 1096*T + 128),
)
def gdn_decode_bf16_ilp4(
    state,                  # bf16 [pool,HV,128,128], arbitrary slot stride
    intermediate,           # bf16 [B,T,HV,128,128], flat-pool alias, or dummy
    A_log,                  # f32 [HV]
    a,                      # bf16 [B,T,HV]
    dt_bias,                # f32 [HV]
    q, k,                   # bf16 [B,T,H,128], independent batch strides
    v, b_gate,              # bf16 [B,T,HV,128], bf16 [B,T,HV]
    output,                 # bf16 [B,T,HV,128] or dummy
    read_indices,           # i32 [B]
    write_indices,          # i32 [B], dummy when SAME_POOL
    accepted_steps,         # i32 [B], dummy unless PER_REQUEST_ACCEPTED_STEPS
    ssm_state_indices,      # i32 [B,T], dummy unless PER_TOKEN_POOL_SCATTER
    state_slot_stride,      # i64 BF16 elements
    q_batch_stride,         # i64 BF16 elements
    k_batch_stride,         # i64 BF16 elements
    v_batch_stride,         # i64 BF16 elements
    batch,                  # i32 runtime launch extent
):
    NUM_GROUPS = 4
    LANES_PER_GROUP = 32
    ELEMS_PER_LANE = 4
    ILP_ROWS = 4
    ROWS_PER_GROUP = TILE_V // NUM_GROUPS
    ROW_QUADS = ROWS_PER_GROUP // ILP_ROWS
    NUM_V_TILES = 128 // TILE_V

    tid = thread_id(axis="x", extent=128)
    warp_raw = tid // 32
    warp = warp_uniform(
        warp_raw, source_lane=0, member_mask=0xffffffff, clamp=31)
    # instruction_selection: shfl.sync.idx.b32; extent: one warp-index broadcast
    lane = tid % 32
    group = warp
    k_start = lane * 4

    linear_cta = cta_id(axis="x")
    v_tile = linear_cta % NUM_V_TILES
    tmp = linear_cta // NUM_V_TILES
    hv = tmp % HV
    n = tmp // HV
    h = hv // (HV // H)

    if PER_REQUEST_ACCEPTED_STEPS:
        accepted = copy_g2r(accepted_steps[n])
        # instruction_selection: ld.global.b32; extent: one i32 runtime bound per CTA thread
        token_bound = accepted + 1
    else:
        token_bound = T

    read_slot_raw = copy_g2r(read_indices[n])
    # instruction_selection: ld.global.b32; extent: one i32 pool index per CTA thread
    read_slot = select(read_slot_raw < 0, 0, read_slot_raw)
    # instruction_selection: max.s32; extent: one null-slot redirect
    if SAME_POOL:
        write_slot = read_slot
        write_state = alias(state[read_slot], state[read_slot])
    else:
        write_slot_raw = copy_g2r(write_indices[n])
        # instruction_selection: ld.global.b32; extent: one split-pool index per CTA thread
        write_slot = select(write_slot_raw < 0, 0, write_slot_raw)
        # instruction_selection: max.s32; extent: one null-slot redirect
        write_state = state[cast("i64", write_slot),hv,:,:]
    read_state = state[cast("i64", read_slot),hv,:,:]

    A_value = copy_g2r(A_log[hv])
    # instruction_selection: ld.global.b32; extent: one FP32 head scalar per CTA thread
    dt_value = copy_g2r(dt_bias[hv])
    # instruction_selection: ld.global.b32; extent: one FP32 head scalar per CTA thread

    if T > 1:
        smem = raw_shared(dtype="u8", bytes=1096*T+128, alignment=16)
        sQ = view(smem[0:544*T-32], dtype="f32", shape=(T,128),
                  stride=(136,1), alignment=16)
        sK = view(smem[544*T-32:1088*T-64], dtype="f32", shape=(T,128),
                  stride=(136,1), alignment=16)
        sGB = view(smem[1088*T-64:1096*T-64], dtype="f32", shape=(T,2),
                   stride=(2,1), alignment=16)
        # The remaining 192 bytes are source allocator/launcher reservation.
    else:
        smem = raw_shared(dtype="u8", bytes=128, alignment=16)
        # No typed shared view is allocated or accessed at T=1.

    r_q_bf16 = reg_tile(dtype="bf16", shape=(4,))
    r_k_bf16 = reg_tile(dtype="bf16", shape=(4,))
    r_q = reg_tile(dtype="f32", shape=(4,))
    r_k = reg_tile(dtype="f32", shape=(4,))
    r_h_bf16 = reg_tile(dtype="bf16", shape=(4,4))
    r_h = reg_tile(dtype="f32", shape=(4,4))
    r_v_bf16 = reg_tile(dtype="bf16", shape=(4,))
    r_o_bf16 = reg_tile(dtype="bf16", shape=(4,))

    # =======================================================================
    # T>1 token-shared precompute: one source pass per four tokens
    # =======================================================================

    if T > 1:
        for pass_index in static_range(ceil_div(T,4)):
            t_pre = pass_index*4 + group
            if t_pre < T:
                if not DISABLE_OUTPUT:
                    for i in static_range(4):
                        r_q_bf16[i] = copy_g2r(q[n,t_pre,h,k_start+i])
                        # instruction_selection: ld.global.b16; extent: one scalar, four instances
                for i in static_range(4):
                    r_k_bf16[i] = copy_g2r(k[n,t_pre,h,k_start+i])
                    # instruction_selection: ld.global.b16; extent: one scalar, four instances

                if not DISABLE_OUTPUT:
                    for i in static_range(4):
                        r_q[i] = cast("f32", r_q_bf16[i])
                        # instruction_selection: cvt.f32.bf16; extent: one scalar, four instances
                for i in static_range(4):
                    r_k[i] = cast("f32", r_k_bf16[i])
                    # instruction_selection: cvt.f32.bf16; extent: one scalar, four instances

                if USE_QK_L2NORM:
                    sum_k = 0.0
                    for i in static_range(4):
                        sum_k = fma(r_k_bf16[i], r_k_bf16[i], sum_k)
                        # instruction_selection: fma.rn.f32.bf16; extent: one ordered square-accumulate, four instances
                    for delta in (16,8,4,2,1):
                        peer_k = shuffle_xor(sum_k, delta, member_mask=-1, clamp=31)
                        # instruction_selection: shfl.sync.bfly.b32; extent: one scalar at each explicit full-warp stage
                        sum_k = add(sum_k, peer_k)
                        # instruction_selection: add.f32; extent: one scalar at each stage
                    k_factor = rsqrt(add(sum_k,1.0e-6))
                    # instruction_selection: add.f32 then rsqrt.approx.ftz.f32; extent: one scalar each
                    for i in static_range(4):
                        r_k[i] = mul(r_k[i], k_factor)
                        # instruction_selection: mul.f32; extent: one scalar, four instances

                    if not DISABLE_OUTPUT:
                        sum_q = 0.0
                        for i in static_range(4):
                            sum_q = fma(r_q_bf16[i], r_q_bf16[i], sum_q)
                            # instruction_selection: fma.rn.f32.bf16; extent: one ordered square-accumulate, four instances
                        for delta in (16,8,4,2,1):
                            peer_q = shuffle_xor(sum_q, delta, member_mask=-1, clamp=31)
                            # instruction_selection: shfl.sync.bfly.b32; extent: one scalar at each explicit full-warp stage
                            sum_q = add(sum_q, peer_q)
                            # instruction_selection: add.f32; extent: one scalar at each stage
                        q_factor = mul(rsqrt(add(sum_q,1.0e-6)), SCALE)
                        # instruction_selection: add.f32, rsqrt.approx.ftz.f32, mul.f32; extent: one scalar each
                        for i in static_range(4):
                            r_q[i] = mul(r_q[i], q_factor)
                            # instruction_selection: mul.f32; extent: one scalar, four instances
                elif not DISABLE_OUTPUT:
                    for i in static_range(4):
                        r_q[i] = mul(r_q[i], SCALE)
                        # instruction_selection: mul.f32; extent: one scalar, four instances

                if not DISABLE_OUTPUT:
                    for i in static_range(4):
                        copy_r2s(r_q[i], sQ[t_pre,k_start+i])
                        # instruction_selection: st.shared.b32; extent: one scalar, four instances
                for i in static_range(4):
                    copy_r2s(r_k[i], sK[t_pre,k_start+i])
                    # instruction_selection: st.shared.b32; extent: one scalar, four instances

                if T > 2:
                    a_value = copy_g2r(a[n,t_pre,hv])
                    # instruction_selection: ld.global.b16; extent: one BF16 gate scalar
                    b_bf16 = copy_g2r(b_gate[n,t_pre,hv])
                    # instruction_selection: ld.global.b16; extent: one BF16 gate scalar
                    b_value = cast("f32",b_bf16)
                    # instruction_selection: cvt.f32.bf16; extent: one scalar
                    x = add(a_value, dt_value)
                    # instruction_selection: add.rn.f32.bf16; extent: one mixed scalar add
                    beta_x = mul(SOFTPLUS_BETA, x)
                    # instruction_selection: constexpr identity at beta=1; otherwise mul.f32; extent: scalar
                    exp_beta_x = exp2(mul(beta_x, LOG2E))
                    # instruction_selection: mul.f32 then ex2.approx.ftz.f32; extent: one scalar each
                    softplus_log = log2(add(1.0, exp_beta_x))
                    # instruction_selection: add.f32 then lg2.approx.ftz.f32; extent: one scalar each
                    softplus_value = mul(
                        mul(reciprocal(SOFTPLUS_BETA), softplus_log), LN2)
                    # instruction_selection: at beta=1 reciprocal and inner mul fold, then exactly one scalar mul.f32 by LN2; non-unit beta additionally emits rcp/mul
                    use_softplus = select(
                        setp_le(beta_x,SOFTPLUS_THRESHOLD), 1.0, 0.0)
                    # instruction_selection: setp.le.f32 then selp.f32; extent: one scalar each
                    direct = sub(1.0, use_softplus)
                    # instruction_selection: sub.f32; extent: one scalar
                    softplus_x = fma(softplus_value, use_softplus, mul(x,direct))
                    # instruction_selection: mul.f32 then fma.rn.f32; extent: one scalar each
                    exp_A = exp2(mul(A_value,LOG2E))
                    # instruction_selection: mul.f32 then ex2.approx.ftz.f32; extent: one scalar each
                    gate_exponent = mul(sub(0.0,exp_A), softplus_x)
                    # instruction_selection: neg.f32 then mul.f32; extent: one scalar each
                    beta = reciprocal(add(1.0,exp2(mul(b_value,-LOG2E))))
                    # instruction_selection: mul.f32, ex2.approx.ftz.f32, add.f32, rcp.rn.f32; extent: one scalar each
                    g = exp2(mul(gate_exponent,LOG2E))
                    # instruction_selection: mul.f32 then ex2.approx.ftz.f32; extent: one scalar each
                    if lane == 0:
                        copy_r2s((g,beta), sGB[t_pre,0:2])
                        # instruction_selection: st.shared.v2.b32; extent: one 8-byte pair by lane 0

            cta_barrier()
            # instruction_selection: bar.sync 0; extent: one CTA-wide publication per pass

    # =======================================================================
    # Four-row bodies: T=1 fully unrolled, T>1 source loop with unroll=1
    # =======================================================================

    for row_quad in source_range(
            ROW_QUADS, unroll=1, unroll_full=(T <= 1)):
        v_base = v_tile*TILE_V + group*ROWS_PER_GROUP + row_quad*4
        v_rows = (v_base+0,v_base+1,v_base+2,v_base+3)

        for row in static_range(4):
            copy_g2r(
                read_state[v_rows[row],k_start:k_start+4], r_h_bf16[row])
            # instruction_selection: ld.global.v4.b16 in fixed-bound T1/T2/T3/T4 variants, ld.global.v2.b32 in the evidenced accepted-step T5 variant; extent: one 8-byte BF16 state vector, four explicit rows
        for i in static_range(4):
            for row in static_range(4):
                r_h[row,i] = cast("f32", r_h_bf16[row,i])
                # instruction_selection: cvt.f32.bf16; extent: one scalar, 16 element-major instances

        for t in runtime_or_static_range(
                token_bound, unroll=1,
                unroll_full=(T <= 1 and not PER_REQUEST_ACCEPTED_STEPS)):
            if T > 1:
                if not DISABLE_OUTPUT:
                    copy_s2r(sQ[t,k_start:k_start+4], r_q)
                    # instruction_selection: ld.shared.v2.b64; extent: one 16-byte FP32 Q vector
                copy_s2r(sK[t,k_start:k_start+4], r_k)
                # instruction_selection: ld.shared.v2.b64; extent: one 16-byte FP32 K vector
                if T > 2:
                    g,beta = copy_s2r(sGB[t,0:2])
                    # instruction_selection: ld.shared.v2.b32; extent: one 8-byte g/beta pair
                else:
                    g,beta = INLINE_GATE(t)
                    # instruction_selection: see INLINE_GATE; extent: one scalar chain per row-quad/token/thread
            else:
                for i in static_range(4):
                    r_q_bf16[i] = copy_g2r(q[n,t,h,k_start+i])
                    # instruction_selection: ld.global.b16; extent: one scalar, four instances per row body
                for i in static_range(4):
                    r_k_bf16[i] = copy_g2r(k[n,t,h,k_start+i])
                    # instruction_selection: ld.global.b16; extent: one scalar, four instances per row body
                for i in static_range(4):
                    r_q[i] = cast("f32", r_q_bf16[i])
                    # instruction_selection: cvt.f32.bf16; extent: one scalar, four instances
                    r_k[i] = cast("f32", r_k_bf16[i])
                    # instruction_selection: cvt.f32.bf16; extent: one scalar, four instances
                if USE_QK_L2NORM:
                    sum_q = 0.0
                    sum_k = 0.0
                    for i in static_range(4):
                        sum_q = fma(r_q_bf16[i],r_q_bf16[i],sum_q)
                        # instruction_selection: fma.rn.f32.bf16; extent: one ordered scalar, four instances
                        sum_k = fma(r_k_bf16[i],r_k_bf16[i],sum_k)
                        # instruction_selection: fma.rn.f32.bf16; extent: one ordered scalar, four instances
                    for delta in (16,8,4,2,1):
                        sum_q = add(sum_q,shuffle_xor(
                            sum_q,delta,member_mask=-1,clamp=31))
                        # instruction_selection: shfl.sync.bfly.b32 then add.f32; extent: one scalar at each explicit stage
                        sum_k = add(sum_k,shuffle_xor(
                            sum_k,delta,member_mask=-1,clamp=31))
                        # instruction_selection: shfl.sync.bfly.b32 then add.f32; extent: one scalar at each explicit stage
                    q_factor = mul(rsqrt(add(sum_q,1.0e-6)),SCALE)
                    # instruction_selection: add.f32, rsqrt.approx.ftz.f32, mul.f32; extent: one scalar each
                    k_factor = rsqrt(add(sum_k,1.0e-6))
                    # instruction_selection: add.f32 then rsqrt.approx.ftz.f32; extent: one scalar each
                    for i in static_range(4):
                        r_q[i] = mul(r_q[i],q_factor)
                        # instruction_selection: mul.f32; extent: one scalar, four instances
                        r_k[i] = mul(r_k[i],k_factor)
                        # instruction_selection: mul.f32; extent: one scalar, four instances
                else:
                    for i in static_range(4):
                        r_q[i] = mul(r_q[i],SCALE)
                        # instruction_selection: mul.f32; extent: one scalar, four instances
                g,beta = INLINE_GATE(t)
                # instruction_selection: see INLINE_GATE; extent: one scalar chain per row body/thread

            dot_hk_lo = reg_tile(dtype="f32", shape=(4,), init=0.0)
            dot_hk_hi = reg_tile(dtype="f32", shape=(4,), init=0.0)
            for pair in static_range(2):
                for row in static_range(4):
                    r_h[row,2*pair+0] = mul(r_h[row,2*pair+0],g)
                    # instruction_selection: mul.f32; extent: one scalar decay
                    r_h[row,2*pair+1] = mul(r_h[row,2*pair+1],g)
                    # instruction_selection: mul.f32; extent: one scalar decay
                for row in static_range(4):
                    dot_hk_lo[row],dot_hk_hi[row] = fma(
                        r_h[row,2*pair:2*pair+2],
                        r_k[2*pair:2*pair+2],
                        (dot_hk_lo[row],dot_hk_hi[row]), lanes=2)
                    # instruction_selection: fma.rn.f32x2; extent: one packed h-k pair, four rows per pair

            dot_hk = reg_tile(dtype="f32", shape=(4,))
            for row in static_range(4):
                dot_hk[row] = add(dot_hk_lo[row],dot_hk_hi[row])
                # instruction_selection: add.f32; extent: one scalar pair fold per row
            for delta in (16,8,4,2,1):
                for row in static_range(4):
                    peer = shuffle_xor(dot_hk[row],delta,member_mask=-1,clamp=31)
                    # instruction_selection: shfl.sync.bfly.b32; extent: one scalar, four rows at each explicit stage
                    dot_hk[row] = add(dot_hk[row],peer)
                    # instruction_selection: add.f32; extent: one scalar, four rows at each stage

            for row in static_range(4):
                r_v_bf16[row] = copy_g2r(v[n,t,hv,v_base+row])
                # instruction_selection: ld.global.b16; extent: one scalar, four instances
            residual = reg_tile(dtype="f32", shape=(4,))
            for row in static_range(4):
                residual[row] = sub(r_v_bf16[row],dot_hk[row])
                # instruction_selection: sub.rn.f32.bf16; extent: one mixed BF16-FP32 scalar subtract
                residual[row] = mul(residual[row],beta)
                # instruction_selection: mul.f32; extent: one scalar

            dot_hq_lo = reg_tile(dtype="f32", shape=(4,), init=0.0)
            dot_hq_hi = reg_tile(dtype="f32", shape=(4,), init=0.0)
            for pair in static_range(2):
                for row in static_range(4):
                    r_h[row,2*pair:2*pair+2] = fma(
                        r_k[2*pair:2*pair+2],
                        (residual[row],residual[row]),
                        r_h[row,2*pair:2*pair+2], lanes=2)
                    # instruction_selection: fma.rn.f32x2; extent: one packed rank-one update
                if not DISABLE_OUTPUT:
                    for row in static_range(4):
                        dot_hq_lo[row],dot_hq_hi[row] = fma(
                            r_h[row,2*pair:2*pair+2],
                            r_q[2*pair:2*pair+2],
                            (dot_hq_lo[row],dot_hq_hi[row]), lanes=2)
                        # instruction_selection: fma.rn.f32x2; extent: one packed h-q pair

            if not DISABLE_OUTPUT:
                dot_hq = reg_tile(dtype="f32", shape=(4,))
                for row in static_range(4):
                    dot_hq[row] = add(dot_hq_lo[row],dot_hq_hi[row])
                    # instruction_selection: add.f32; extent: one scalar pair fold per row, before any state conversion/store

            if CACHE_INTERMEDIATE_STATES or PER_TOKEN_POOL_SCATTER:
                if PER_TOKEN_POOL_SCATTER and not PER_TOKEN_POOL_SCATTER_FLAT and not SAME_POOL:
                    for i in static_range(4):
                        for row in static_range(4):
                            r_h_bf16[row,i] = cast(
                                "bf16",r_h[row,i],rounding="rn")
                            # instruction_selection: cvt.rn.bf16.f32; extent: four row conversions per element, 16 element-major instances
                else:
                    for row in static_range(4):
                        r_h_bf16[row] = cast(
                            "bf16x4", r_h[row], rounding="rn")
                        # instruction_selection: cvt.rn.bf16x2.f32; extent: two pair conversions per row in packed variants

            if PER_TOKEN_POOL_SCATTER:
                pool_slot_t = copy_g2r(ssm_state_indices[n,t])
                # instruction_selection: ld.global.b32; extent: one per-token slot per CTA thread
                if PER_TOKEN_POOL_SCATTER_FLAT:
                    flat = cast("i64",pool_slot_t)*HV + hv
                    for row in static_range(4):
                        copy_r2g(
                            r_h_bf16[row],
                            intermediate[flat,v_rows[row],k_start:k_start+4])
                        # instruction_selection: st.global.v2.b32; extent: one 8-byte state vector per row
                else:
                    scatter_state = state[cast("i64",pool_slot_t),hv,:,:]
                    for row in static_range(4):
                        copy_r2g(
                            r_h_bf16[row],
                            scatter_state[v_rows[row],k_start:k_start+4])
                        # instruction_selection: st.global.v2.b32; extent: one 8-byte padded-pool state vector per row
            elif CACHE_INTERMEDIATE_STATES:
                dense = cast("i64",n*T*HV + t*HV + hv)
                for row in static_range(4):
                    copy_r2g(
                        r_h_bf16[row],
                        intermediate[dense,v_rows[row],k_start:k_start+4])
                    # instruction_selection: st.global.v2.b32; extent: one 8-byte dense-cache state vector per row

            if not DISABLE_OUTPUT:
                for delta in (16,8,4,2,1):
                    for row in static_range(4):
                        peer = shuffle_xor(
                            dot_hq[row],delta,member_mask=-1,clamp=31)
                        # instruction_selection: shfl.sync.bfly.b32; extent: one scalar, four rows at each explicit stage
                        dot_hq[row] = add(dot_hq[row],peer)
                        # instruction_selection: add.f32; extent: one scalar, four rows at each stage
                if lane == 0:
                    for row in static_range(4):
                        r_o_bf16[row] = cast(
                            "bf16",dot_hq[row],rounding="rn")
                        # instruction_selection: cvt.rn.bf16.f32; extent: one scalar, four rows
                    for row in static_range(4):
                        copy_r2g(r_o_bf16[row],output[n,t,hv,v_base+row])
                        # instruction_selection: st.global.b16; extent: one scalar, four instances by lane 0

        # Source final write is independent of dense caching. In cache/scatter
        # modes r_h_bf16 already contains the last processed token state.
        if not DISABLE_STATE_UPDATE:
            if not (PER_TOKEN_POOL_SCATTER and SAME_POOL):
                if not CACHE_INTERMEDIATE_STATES and not PER_TOKEN_POOL_SCATTER:
                    for row in static_range(4):
                        r_h_bf16[row] = cast(
                            "bf16x4",r_h[row],rounding="rn")
                        # instruction_selection: cvt.rn.bf16x2.f32; extent: two pair conversions per row
                for row in static_range(4):
                    copy_r2g(
                        r_h_bf16[row],
                        write_state[v_rows[row],k_start:k_start+4])
                    # instruction_selection: st.global.v4.b16 in padded split-scatter, otherwise st.global.v2.b32; extent: one 8-byte final-state vector per row

# ===========================================================================
# INLINE_GATE — source gate chain for T=1 and T=2
# ===========================================================================

def INLINE_GATE(t):
    a_value = copy_g2r(a[n,t,hv])
    # instruction_selection: ld.global.b16; extent: one BF16 gate scalar
    b_bf16 = copy_g2r(b_gate[n,t,hv])
    # instruction_selection: ld.global.b16; extent: one BF16 gate scalar
    b_value = cast("f32",b_bf16)
    # instruction_selection: cvt.f32.bf16; extent: one scalar
    x = add(a_value,dt_value)
    # instruction_selection: add.rn.f32.bf16; extent: one mixed scalar add
    beta_x = mul(SOFTPLUS_BETA,x)
    # instruction_selection: constexpr identity at beta=1; otherwise mul.f32; extent: scalar
    exp_beta_x = exp2(mul(beta_x,LOG2E))
    # instruction_selection: mul.f32 then ex2.approx.ftz.f32; extent: one scalar each
    softplus_log = log2(add(1.0,exp_beta_x))
    # instruction_selection: add.f32 then lg2.approx.ftz.f32; extent: one scalar each
    softplus_value = mul(
        mul(reciprocal(SOFTPLUS_BETA),softplus_log),LN2)
    # instruction_selection: at beta=1 reciprocal and inner mul fold, then exactly one scalar mul.f32 by LN2; non-unit beta additionally emits rcp/mul
    use_softplus = select(setp_le(beta_x,SOFTPLUS_THRESHOLD),1.0,0.0)
    # instruction_selection: setp.le.f32 then selp.f32; extent: one scalar each
    direct = sub(1.0,use_softplus)
    # instruction_selection: sub.f32; extent: one scalar
    softplus_x = fma(softplus_value,use_softplus,mul(x,direct))
    # instruction_selection: mul.f32 then fma.rn.f32; extent: one scalar each
    exp_A = exp2(mul(A_value,LOG2E))
    # instruction_selection: mul.f32 then ex2.approx.ftz.f32; extent: one scalar each
    gate_exponent = mul(sub(0.0,exp_A),softplus_x)
    # instruction_selection: neg.f32 then mul.f32; extent: one scalar each
    beta = reciprocal(add(1.0,exp2(mul(b_value,-LOG2E))))
    # instruction_selection: mul.f32, ex2.approx.ftz.f32, add.f32, rcp.rn.f32; extent: one scalar each
    g = exp2(mul(gate_exponent,LOG2E))
    # instruction_selection: mul.f32 then ex2.approx.ftz.f32; extent: one scalar each
    return g,beta
```

## Storage ownership and lifetimes

| storage | logical owner | lifetime and alias rule |
| --- | --- | --- |
| sQ `[T,128]`, stride 136 | token-owning warp publishes its four-element segment | T>1 only; live after the pass barrier through every row-quad/token body; dead when output is disabled |
| sK `[T,128]`, stride 136 | token-owning warp publishes its four-element segment | T>1 only; live after the pass barrier through all row bodies |
| sGB `[T,2]` | lane 0 of the token-owning warp | T>2 only; immutable after the pass barrier |
| r_h `[4,4]` | one lane in one physical warp/group | one row-quad body; loaded once and carried across its accepted token loop |
| r_h_bf16 `[4,4]` | same lane | initial loads, then reused for cache/scatter/final stores after explicit repacking |
| read/write state views | one CTA and disjoint V/K segments | exact alias under SAME_POOL; otherwise distinct i64 slot bases |
| dense intermediate | one `(n,t,hv,V,K)` region | batch-scoped, written after each processed token |
| flat/padded scatter | one caller-selected slot per `(n,t)` | flat route uses i64 `(slot*HV+hv)`; padded route retains 4-D slot stride |

No data race depends on scheduling. Each warp owns disjoint V rows, each lane
owns disjoint K segments, lane zero uniquely owns the four output values, and
every T>1 precompute pass completes at a CTA barrier before shared consumption.

## Control-flow modes

| mode | token loop | state destinations |
| --- | --- | --- |
| T=1 production | one fully unrolled token inside each fully unrolled row body | final same/split pool state |
| T>1 production cache | constexpr T tokens | dense cache after every token; pool remains unchanged because production disables state update |
| per-request accepted | runtime `accepted[n]+1` tokens | output/cache/scatter only for processed tokens; optional final pool contains h_K |
| state-only | same token bound, output dot/reduction/store removed | final/cache/scatter routes remain |
| per-token scatter | same token bound | every h_{t+1} to `ssm_state_indices[n,t]`; same-pool final write suppressed, split final write retained |

## TIRx module and benchmark contract

- Registry name: `gdn_decode_bf16_ilp4`; supported compute capability is 10.
- Correctness uses the frozen FlashInfer ILP4 path, not a mathematical oracle,
  and covers all compile-time modes in the legal contract plus T=1/T=2/T>2
  and odd precompute tails.
- Performance contains exactly 48 production workloads: T in {1,2,4,8}, the
  four Qwen3-Next `(H,HV)` pairs, and only batch sizes reaching this fallback.
- T=1 performance enables final state update without dense cache. T>1
  performance enables output and dense cache while disabling final pool update.
- Only a complete target-filtered `python -m tirx_kernels.bench_suite` run may
  accept performance, and every `flashinfer_cutedsl_us / tirx_us` ratio must be
  strictly greater than 0.99.

## Instruction selection is a lowering consequence

- T=1 and T>1 Q/K use four scalar `ld.global.b16` operations per segment; T>1
  shared Q/K consumption uses one `ld.shared.v2.b64`. State loads are
  `ld.global.v4.b16` in the evidenced fixed-bound variants and `ld.global.v2.b32`
  in the accepted T5 variant. V and output use four scalar `ld.global.b16` and
  `st.global.b16` operations per row body/token.
- Q/K shared publication uses scalar `st.shared.b32`; T>2 g/beta publication
  uses lane-zero `st.shared.v2.b32`; every pass ends with `bar.sync 0`.
- Every reduction uses full-warp `shfl.sync.bfly.b32` at offsets
  16,8,4,2,1 with clamp 31.
- State decay uses scalar `mul.f32`. h-k and h-q pair accumulation, and the
  rank-one update, use `fma.rn.f32x2`; pair halves are explicitly folded with
  `add.f32` before the shuffle tree.
- L2 and gate math use `rsqrt.approx.ftz.f32`, `ex2.approx.ftz.f32`,
  `lg2.approx.ftz.f32`, and `rcp.rn.f32`. The BF16 V residual selects
  `sub.rn.f32.bf16`.
- Output is four lane-zero scalar `st.global.b16` operations. Packed state modes
  use two `cvt.rn.bf16x2.f32` conversions per row; padded split-scatter uses four
  scalar conversions and `st.global.v4.b16` for its final row, while its
  per-token scatter and other final modes retain `st.global.v2.b32`.
- The source row-quad loop is fully unrolled only at T=1. T>1 retains the
  source `unroll=1` loop shape; replacing it with a generic output-element loop
  or a different warp/lane mapping is not faithful.
