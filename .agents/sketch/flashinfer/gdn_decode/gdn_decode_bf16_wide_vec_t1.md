<!--
Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0

This design sketch documents a modified TIRx port of FlashInfer's
gdn_decode_bf16_state.py. See LICENSE, NOTICE, and licenses/ for the
applicable terms.
-->

# GDN decode BF16 wide-vector T1 SM100: execution sketch

This file is a non-executable execution sketch.  It freezes the lane ownership,
shared publication, register-resident state update, packed-FP32x2 arithmetic,
and vector state traffic of FlashInfer's CuTeDSL
`gdn_wide_vec_kernel_t1`.  The target implementation is
[`tirx_kernels/flashinfer/gdn_decode/gdn_decode_bf16_wide_vec_t1.py`](../../tirx_kernels/flashinfer/gdn_decode/gdn_decode_bf16_wide_vec_t1.py),
which may become executable only after this sketch passes independent review.

The frozen FlashInfer commit is
`f2e04400e330fb2debe0bf8730d9424a1d37927f`; the source SHA256 is
`61de9ffa703962cb1ddb73823100550138708bbcbb535a3efcac608940e67e61`.
The target is SM100a/B200 and fixes T=1, K=V=128, BF16 Q/K/V/state/output,
FP32 A_log/dt_bias/accumulators, and packed `fma.rn.f32x2`.  `tile_v` is 64
or 128.  The direct-source branches for Q/K L2 normalization, same/split state
pool, padded slot stride, packed QKV batch strides, negative slot indices,
disabled state update, and intermediate-state caching remain in scope.

The production dispatcher reaches this body only when `B*HV >= 512`: it uses
`tile_v=64` for `[512,1024)` and `tile_v=128` at `>=1024`.  T>1, K/V other
than 128, FP32 state, the ILP4 kernel, and the source's direct-only
`tile_v=32` specialization are out of scope.

## Pipeline at a glance

There is no asynchronous pipeline and no mbarrier.  A single CTA barrier is
the only publication edge.  T=1 specializes the source's four-warps-over-T
precompute loop to one active logical warp: warp 0 performs the complete Q,
K, K-Q dot, g, and beta producer chain while warps 1..3 wait at the barrier.
After the barrier all eight 16-lane groups run independently.

| Physical lanes | Role-local program | Publication/reuse edge |
| --- | --- | --- |
| warp 0, lanes 0..31 | two redundant 16-lane halves issue eight scalar Q and eight scalar K loads, optionally normalize both vectors, publish sQ/sK, reduce K-Q, and form the g/beta chains | scalar shared stores for Q/K and three lane-0 g/beta/kq stores, then CTA `bar.sync 0` |
| warps 1..3 | no Phase-0 producer work for T=1 | wait at the CTA `bar.sync 0` |
| groups 0..7 after barrier, 16 lanes each | each lane owns eight adjacent K values; each group owns 8 V rows at tile 64 or 16 V rows at tile 128 and processes four rows per fully-unrolled iteration | shared Q/K/g/beta are read-only; state remains in registers for each four-row iteration |
| lane 0 of each 16-lane group | writes four output rows per iteration | no further synchronization; each group owns disjoint V rows |
| every lane after each iteration | optionally writes four 16-byte state vectors either to intermediate storage or the selected output-pool slot | disjoint `(V-row,K-segment)` ownership |

## Primitive vocabulary

Structural operations do not move or compute values:

```python
specialize(...)       # compile-time source variant
launch(...)           # physical grid/block/shared metadata
raw_shared(...)       # one dynamic shared allocation
view(...)             # typed storage view without a copy
alias(...)            # exact read/write storage alias
reg_tile(...)         # lane-private register tile
```

Copies expose storage direction:

```python
copy_g2r(src, dst=None, predicate=None)
copy_s2r(src, dst=None)
copy_r2s(src, dst)
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
select(predicate, true_value, false_value)
setp_le(lhs, rhs)
selp(predicate, true_value, false_value)
unpack_b32_pair(src)
shuffle_xor(src, lane_delta, member_mask, clamp)
warp_uniform(src, source_lane, member_mask, clamp)
cta_barrier()
```

`fma(..., lanes=2)` is one packed two-lane FP32 operation with two ordered
results.  A copy of a stated eight-element tile is one explicit loop of the
same scalar PTX family; a four-row state copy is four separately stated
128-bit operations.  Address arithmetic and static view construction are not
expanded because they do not change ownership, masking, control flow, or
instruction selection.  There is deliberately no `normalize`, `softplus`,
`sigmoid`, `reduce`, `delta_rule`, or `update_state` compound primitive.

## Complete sketch

```python
# ==========================================================================
# Static specialization, runtime ABI, launch, and storage
# ==========================================================================

@specialize(
    T=1,
    H=(16, 8, 4, 2),
    HV=(32, 16, 8, 4),
    K=128,
    V=128,
    TILE_V=(64, 128),
    USE_QK_L2NORM=(False, True),
    DISABLE_STATE_UPDATE=(False, True),
    CACHE_INTERMEDIATE_STATES=(False, True),
    USE_PACKED_FMA=True,
    SAME_POOL=(False, True),
    SOFTPLUS_BETA=1.0,
    SOFTPLUS_THRESHOLD=20.0,
    SCALE=1.0/sqrt(128.0),
    target="sm_100a",
)
@launch(
    grid=(batch * HV * (128 // TILE_V), 1, 1),
    block=(128, 1, 1),
    num_warps=4,
    dynamic_smem_bytes=1356,
)
def gdn_decode_bf16_wide_vec_t1(
    state,                  # bf16 [pool,HV,128,128], mutable
    intermediate,           # bf16 [B,1,HV,128,128] or dummy
    A_log,                  # f32  [HV]
    a,                      # bf16 [B,1,HV]
    dt_bias,                # f32  [HV]
    q, k,                   # bf16 [B,1,H,128]
    v, b_gate,              # bf16 [B,1,HV,128], bf16 [B,1,HV]
    output,                 # bf16 [B,1,HV,128]
    read_indices,           # i32  [B]
    write_indices,          # i32  [B], ignored when SAME_POOL or no final pool write survives
    batch,                  # i32 runtime launch extent
    state_slot_stride,      # i64 BF16 elements, permits padded 4-D pool
    q_batch_stride,         # i64 BF16 elements, permits packed QKV view
    k_batch_stride,         # i64 BF16 elements
    v_batch_stride,         # i64 BF16 elements
):
    NUM_GROUPS = 8
    LANES_PER_GROUP = 16
    ELEMS_PER_LANE = 8
    ILP_ROWS = 4
    ROWS_PER_GROUP = TILE_V // NUM_GROUPS       # 8 or 16
    ITERS_PER_GROUP = ROWS_PER_GROUP // ILP_ROWS # 2 or 4
    NUM_V_TILES = 128 // TILE_V                  # 2 or 1

    tid = thread_id(axis="x", extent=128)
    warp_raw = tid // 32
    warp = warp_uniform(
        warp_raw, source_lane=0, member_mask=0xffffffff, clamp=32)
    # This mirrors cute.arch.make_warp_uniform(cute.arch.warp_idx()).
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

    read_slot = copy_g2r(read_indices[n])
    # instruction_selection: ld.global.b32; extent: one i32 slot index per CTA thread, with compiler hoisting/sinking permitted
    # The source requests 1356 dynamic bytes and gives each typed allocation
    # 16-byte alignment.  For T=1 the logical cosize of each (1,128),
    # stride-(136,1) view is 128 floats, so reviewed PTX uses offsets 0, 512,
    # and 1024.  The remaining 340 bytes are reservation tail.
    smem = raw_shared(dtype="u8", bytes=1356, alignment=16)
    sQ = view(
        smem[0:512], dtype="f32", shape=(1,128), stride=(136,1), alignment=16)
    sK = view(
        smem[512:1024], dtype="f32", shape=(1,128), stride=(136,1), alignment=16)
    sGB = view(
        smem[1024:1036], dtype="f32", shape=(1,3), stride=(3,1), alignment=16)
    # Slot 2 holds the source's published K-Q dot.  Phase 1 does not consume
    # it for T=1, but the frozen generated source retains the producer/store.

    r_q_bf16 = reg_tile(dtype="bf16", shape=(8,))
    r_k_bf16 = reg_tile(dtype="bf16", shape=(8,))
    r_q = reg_tile(dtype="f32", shape=(8,))
    r_k = reg_tile(dtype="f32", shape=(8,))
    r_h_words = reg_tile(dtype="u32", shape=(4,4))
    r_h_bf16 = reg_tile(dtype="bf16", shape=(4,8))
    r_h = reg_tile(dtype="f32", shape=(4,8))
    r_k_main = reg_tile(dtype="f32", shape=(4,2))
    r_q_main = reg_tile(dtype="f32", shape=(4,2))

    read_slot = select(read_slot < 0, 0, read_slot)
    # instruction_selection: max.s32(index,0); extent: one scalar null-slot redirect

    if SAME_POOL:
        write_slot = read_slot
        write_state = alias(state[read_slot], state[read_slot])
    else:
        write_slot = copy_g2r(write_indices[n])
        # instruction_selection: ld.global.b32; extent: one i32 split-pool write slot per CTA thread only when not DISABLE_STATE_UPDATE and not CACHE_INTERMEDIATE_STATES; zero instructions after constexpr DCE when no final pool write survives
        write_slot = select(write_slot < 0, 0, write_slot)
        # instruction_selection: max.s32(index,0); extent: one scalar null-slot redirect only when not DISABLE_STATE_UPDATE and not CACHE_INTERMEDIATE_STATES; zero instructions after constexpr DCE when no final pool write survives
        write_state = state[cast("i64", write_slot), hv, :, :]

    read_state = state[cast("i64", read_slot), hv, :, :]
    # cast to i64 participates in wide state-slot offset arithmetic; the
    # runtime state_slot_stride is retained and may exceed HV*128*128.

    # ==========================================================================
    # Phase 0: the only T=1 precompute pass belongs to logical warp 0.
    # Warps 1..3 execute no producer instructions before the CTA barrier.
    # ==========================================================================

    if warp == 0:
        member_pre = lane_in_warp % 16
        k_pre = member_pre * 8

        # The frozen PTX sinks the two uniform FP32 loads into warp 0 before
        # the Q/K tile traffic, physically issuing dt_bias and then A_log.
        dt_value = copy_g2r(dt_bias[hv])
        A_value = copy_g2r(A_log[hv])

        for i in static_range(8):
            r_q_bf16[i] = copy_g2r(q[n, 0, h, k_pre+i])
            # instruction_selection: ld.global.b16; extent: one scalar Q load
            # per loop instance, using the runtime batch stride
        for i in static_range(8):
            r_k_bf16[i] = copy_g2r(k[n, 0, h, k_pre+i])
            # instruction_selection: ld.global.b16; extent: one scalar K load
            # per loop instance, using the runtime batch stride
        for i in static_range(8):
            r_q[i] = cast("f32", r_q_bf16[i])
            r_k[i] = cast("f32", r_k_bf16[i])
            # instruction_selection: cvt.f32.bf16; extent: Q then K scalar
            # conversions for each element after both complete load tiles

        if USE_QK_L2NORM:
            sum_q = 0.0
            sum_k = 0.0
            for i in static_range(8):
                sum_q = fma(r_q_bf16[i], r_q_bf16[i], sum_q)
                sum_k = fma(r_k_bf16[i], r_k_bf16[i], sum_k)
                # instruction_selection: fma.rn.f32.bf16; extent: two scalar
                # square-accumulates, eight loop instances

            for delta in (8, 4, 2, 1):
                peer_q = shuffle_xor(sum_q, delta, member_mask=-1, clamp=31)
                # instruction_selection: shfl.sync.bfly.b32; extent: one scalar, four explicit stages
                sum_q = add(sum_q, peer_q)
                peer_k = shuffle_xor(sum_k, delta, member_mask=-1, clamp=31)
                sum_k = add(sum_k, peer_k)
                # instruction_selection: add.f32; extent: two scalars, four explicit stages

            q_eps = add(sum_q, 1.0e-6)
            # instruction_selection: add.f32; extent: one scalar
            inv_q = rsqrt(q_eps)
            # instruction_selection: rsqrt.approx.ftz.f32; extent: one scalar
            q_factor = mul(inv_q, SCALE)
            # instruction_selection: mul.f32; extent: one scalar
            k_factor = rsqrt(add(sum_k, 1.0e-6))
            for i in static_range(8):
                r_q[i] = mul(r_q[i], q_factor)
                r_k[i] = mul(r_k[i], k_factor)
                # instruction_selection: mul.f32; extent: two scalars, eight loop instances
        else:
            for i in static_range(8):
                r_q[i] = mul(r_q[i], SCALE)
                # instruction_selection: mul.f32; extent: one scalar, eight loop instances

        copy_r2s(r_q, sQ[0, k_pre:k_pre+8])
        copy_r2s(r_k, sK[0, k_pre:k_pre+8])
        # instruction_selection: st.shared.b32; extent: explicit unrolled loop
        # of eight Q and eight K scalar stores; both half-warps write identical locations

        kq = 0.0
        for i in static_range(8):
            kq = fma(r_k[i], r_q[i], kq)
            # instruction_selection: fma.rn.f32; extent: eight ordered scalar FMAs
        for delta in (8, 4, 2, 1):
            kq = add(kq, shuffle_xor(kq, delta, member_mask=-1, clamp=31))
            # instruction_selection: shfl.sync.bfly.b32 plus add.f32;
            # extent: one scalar at each of four explicit stages

        a_value = copy_g2r(a[n, 0, hv])
        # instruction_selection: ld.global.b16; extent: one scalar BF16 gate input per active thread
        x = add(cast("f32", a_value), dt_value)
        # instruction_selection: add.rn.f32.bf16 in reviewed PTX; extent: one scalar mixed BF16/FP32 add
        beta_x = x  # constexpr identity because SOFTPLUS_BETA is 1.0; zero instructions
        beta_x_log2 = mul(beta_x, LOG2E)
        # instruction_selection: mul.f32; extent: one scalar base conversion
        exp_beta_x = exp2(beta_x_log2)
        # instruction_selection: ex2.approx.ftz.f32; extent: one scalar
        one_plus_exp = add(1.0, exp_beta_x)
        # instruction_selection: add.f32; extent: one scalar
        log_softplus = log2(one_plus_exp)
        # instruction_selection: lg2.approx.ftz.f32; extent: one scalar
        log_softplus = mul(log_softplus, LN2)
        # instruction_selection: mul.f32; extent: one scalar base conversion
        softplus_value = log_softplus  # constexpr reciprocal/multiply fold; zero instructions
        use_softplus_pred = setp_le(beta_x, SOFTPLUS_THRESHOLD)
        use_softplus = selp(use_softplus_pred, 1.0, 0.0)
        # instruction_selection: setp.le.f32 plus selp.f32; extent: one scalar threshold branch without control divergence
        direct_weight = sub(1.0, use_softplus)
        # instruction_selection: sub.f32; extent: one scalar
        softplus_x = mul(x, direct_weight)
        # instruction_selection: mul.f32; extent: one scalar direct branch
        softplus_x = fma(softplus_value, use_softplus, softplus_x)
        # instruction_selection: fma.rn.f32; extent: one scalar selected result

        A_log2 = mul(A_value, LOG2E)
        # instruction_selection: mul.f32; extent: one scalar base conversion
        exp_A = exp2(A_log2)
        # instruction_selection: ex2.approx.ftz.f32; extent: one scalar
        gate_exponent = mul(-exp_A, softplus_x)
        # instruction_selection: neg.f32 plus mul.f32; extent: one scalar each
        if lane_in_warp == 0:
            b_value = copy_g2r(b_gate[n, 0, hv])
            neg_b_log2 = mul(cast("f32", b_value), -LOG2E)
            exp_neg_b = exp2(neg_b_log2)
            beta = reciprocal(add(1.0, exp_neg_b))
            # The b load and complete beta chain are physically lane-0-only.
            gate_log2 = mul(gate_exponent, LOG2E)
            # instruction_selection: mul.f32; extent: one scalar base conversion
            g = exp2(gate_log2)
            # instruction_selection: ex2.approx.ftz.f32; extent: one scalar
            copy_r2s(g, sGB[0, 0])
            copy_r2s(beta, sGB[0, 1])
            copy_r2s(kq, sGB[0, 2])
            # instruction_selection: st.shared.b32; extent: three scalar
            # stores by physical warp-0 lane 0

    cta_barrier()
    # instruction_selection: bar.sync 0; extent: one CTA-wide publication point reached by all 128 threads

    # ==========================================================================
    # Phase 1: eight independent 16-lane groups, four V rows at a time
    # Source lines 2163-2410.  The source constexpr loop and reviewed PTX are
    # fully unrolled to 2 bodies for TILE_V=64 or 4 for TILE_V=128.
    # ==========================================================================

    for iter_idx in static_range(ITERS_PER_GROUP):
        v_base = v_tile * TILE_V + group * ROWS_PER_GROUP + iter_idx * 4
        v_rows = (v_base + 0, v_base + 1, v_base + 2, v_base + 3)

        copy_g2r(read_state[v_rows[0], k_start:k_start+8], r_h_words[0])
        # instruction_selection: one 16-byte vector containing eight adjacent
        # BF16 K values; ordinary global cache qualifier for both tile sizes
        for pair in (3, 2, 1, 0):
            r_h_bf16[0, 2*pair:2*pair+2] = unpack_b32_pair(r_h_words[0, pair])
        copy_g2r(read_state[v_rows[1], k_start:k_start+8], r_h_words[1])
        # instruction_selection: ld.global.v4.b32; extent: one independent 16-byte vector
        for pair in (3, 2, 1, 0):
            r_h_bf16[1, 2*pair:2*pair+2] = unpack_b32_pair(r_h_words[1, pair])
        copy_g2r(read_state[v_rows[2], k_start:k_start+8], r_h_words[2])
        # instruction_selection: ld.global.v4.b32; extent: one independent 16-byte vector
        for pair in (3, 2, 1, 0):
            r_h_bf16[2, 2*pair:2*pair+2] = unpack_b32_pair(r_h_words[2, pair])
        copy_g2r(read_state[v_rows[3], k_start:k_start+8], r_h_words[3])
        # instruction_selection: ld.global.v4.b32; extent: one independent 16-byte vector
        for pair in (3, 2, 1, 0):
            r_h_bf16[3, 2*pair:2*pair+2] = unpack_b32_pair(r_h_words[3, pair])
        # Each load is immediately followed by its four mov.b32 pair unpacks
        # in reverse word order; all 16 unpacks precede any conversion.
        for element in static_range(8):
            for row in static_range(4):
                r_h[row, element] = cast("f32", r_h_bf16[row, element])
        # instruction_selection: cvt.f32.bf16; extent: 32 scalar conversions,
        # element-major across the four rows exactly as the frozen PTX

        if iter_idx == 0:
            g = copy_s2r(sGB[0, 0])
            beta = copy_s2r(sGB[0, 1])
            # The source loads g/beta here on first use and retains them across
            # every later constexpr body.

        s = (0.0, 0.0, 0.0, 0.0)
        for pair in static_range(4):
            if iter_idx == 0:
                r_k_main[pair] = copy_s2r(
                    sK[0, k_start+2*pair:k_start+2*pair+2])
            k_pair = r_k_main[pair]
            # instruction_selection: ld.shared.b32; extent: explicit loop of two scalar loads per K pair, four pairs
            for row in static_range(4):
                r_h[row, 2*pair:2*pair+2] = fma(
                    r_h[row, 2*pair:2*pair+2], (g, g), (0.0, 0.0), lanes=2)
                # instruction_selection: fma.rn.f32x2; extent: one packed two-lane decay, four rows for each of four K pairs
                s[row] = fma(r_h[row, 2*pair+0], k_pair[0], s[row])
                # instruction_selection: fma.rn.f32; extent: one scalar h-k accumulate, four rows for each K pair
                s[row] = fma(r_h[row, 2*pair+1], k_pair[1], s[row])
                # instruction_selection: fma.rn.f32; extent: second scalar h-k accumulate, four rows for each K pair

        for delta in (8, 4, 2, 1):
            for row in static_range(4):
                peer = shuffle_xor(s[row], delta, member_mask=-1, clamp=31)
                # instruction_selection: shfl.sync.bfly.b32; extent: one scalar, four rows at each of four explicit stages
                s[row] = add(s[row], peer)
                # instruction_selection: add.f32; extent: one scalar, four rows at each of four explicit stages

        value = reg_tile(dtype="f32", shape=(4,))
        for row in static_range(4):
            value_bf16 = copy_g2r(v[n, 0, hv, v_rows[row]])
            # instruction_selection: ld.global.b16; extent: four independent
            # scalar loads for both tile sizes
            residual = sub(cast("f32", value_bf16), s[row])
            # instruction_selection: sub.rn.f32.bf16 in reviewed PTX; extent: one mixed BF16/FP32 scalar subtract per row
            value[row] = mul(residual, beta)
            # instruction_selection: mul.f32; extent: one scalar per row

        out_acc = (0.0, 0.0, 0.0, 0.0)
        for pair in static_range(4):
            if iter_idx == 0:
                r_q_main[pair] = copy_s2r(
                    sQ[0, k_start+2*pair:k_start+2*pair+2])
            q_pair = r_q_main[pair]
            # instruction_selection: ld.shared.b32; extent: explicit loop of two scalar loads per K pair, four pairs
            k_pair = r_k_main[pair]
            # The reviewed PTX retains K from the decay loop and emits no
            # redundant reload; Q/K/g/beta all remain live in registers across
            # later constexpr bodies.
            for row in static_range(4):
                r_h[row, 2*pair:2*pair+2] = fma(
                    k_pair, (value[row], value[row]),
                    r_h[row, 2*pair:2*pair+2], lanes=2)
                # instruction_selection: fma.rn.f32x2; extent: one packed rank-one update, four rows for each of four K pairs
                out_acc[row] = fma(r_h[row, 2*pair+0], q_pair[0], out_acc[row])
                # instruction_selection: fma.rn.f32; extent: one scalar h-q accumulate, four rows for each K pair
                out_acc[row] = fma(r_h[row, 2*pair+1], q_pair[1], out_acc[row])
                # instruction_selection: fma.rn.f32; extent: second scalar h-q accumulate, four rows for each K pair

        for delta in (8, 4, 2, 1):
            for row in static_range(4):
                peer = shuffle_xor(out_acc[row], delta, member_mask=-1, clamp=31)
                # instruction_selection: shfl.sync.bfly.b32; extent: one scalar, four rows at each of four explicit stages
                out_acc[row] = add(out_acc[row], peer)
                # instruction_selection: add.f32; extent: one scalar, four rows at each of four explicit stages

        if lane == 0:
            for row in static_range(4):
                out_bf16 = cast("bf16", out_acc[row], rounding="rn")
                # instruction_selection: cvt.rn.bf16.f32; extent: one scalar per owned output row
                copy_r2g(out_bf16, output[n, 0, hv, v_rows[row]])
                # instruction_selection: st.global.b16; extent: one scalar per owned output row, subgroup lane 0 only

        if CACHE_INTERMEDIATE_STATES:
            for row in static_range(4):
                packed_state = cast("bf16x8", r_h[row], rounding="rn")
                # instruction_selection: cvt.rn.bf16x2.f32; extent: four pair conversions for one eight-element row
                copy_r2g(
                    packed_state,
                    intermediate[n, 0, hv, v_rows[row], k_start:k_start+8])
                # instruction_selection: st.global.v4.b32; extent: one 16-byte vector per row, indexed by batch n rather than pool slot

        if not DISABLE_STATE_UPDATE and not CACHE_INTERMEDIATE_STATES:
            for row in static_range(4):
                packed_state = cast("bf16x8", r_h[row], rounding="rn")
                # instruction_selection: cvt.rn.bf16x2.f32; extent: four pair conversions for one eight-element row
                copy_r2g(
                    packed_state,
                    write_state[v_rows[row], k_start:k_start+8])
                # instruction_selection: st.global.v4.b32; extent: one
                # ordinary 16-byte vector store per row at the aliased or
                # split-pool slot
```

## Storage ownership and lifetimes

| storage | logical owner | lifetime and alias rule |
| --- | --- | --- |
| sQ/sK | warp 0 publishes both; all groups consume | live from Phase-0 stores through every Phase-1 iteration; immutable after the barrier |
| sGB | warp-0 lane 0 publishes g, beta, and kq | g/beta are consumed by every Phase-1 thread; kq is stored but not loaded for T=1 |
| r_h `[4,8]` | one lane in one 16-lane group | loaded, decayed, updated, used for output, and optionally stored within one four-row iteration; no cross-iteration state |
| state read/write views | one CTA and disjoint V/K segments | exact alias when SAME_POOL; distinct Int64 slot bases otherwise |
| intermediate | one `(n,hv,V,K)` region | batch-scoped, never indexed by a pool slot; when enabled it owns the final state and suppresses pool writeback |

No data race depends on scheduling: the duplicate Phase-0 half-warps store
bit-identical values; the CTA barrier completes those stores; Phase 1 assigns
each `(V,K)` state element to exactly one lane and each V output to exactly one
subgroup lane 0.

## Frozen source-PTX evidence

The mandatory fresh source build is preserved under
`.porting/gdn_decode_bf16_wide_vec_t1/source_ptx_stage2/`; its
[`manifest.md`](../../../.porting/gdn_decode_bf16_wide_vec_t1/source_ptx_stage2/manifest.md)
records the exact static arguments, toolchain, artifacts, and hashes.  The PTX
maps `.file 1` to the frozen CuTeDSL source and declares `.reqntid 128,1,1`.

| source region | reviewed line-info PTX consequence |
| --- | --- |
| 2088-2120 | sixteen scalar BF16 Q/K loads, BF16/F32 conversion, optional mixed-BF16 square FMAs, 16-lane butterfly normalization, then sixteen scalar shared stores |
| 2122-2161 | eight scalar K-Q FMAs, four butterfly steps, fast exp/log/reciprocal gate arithmetic, three scalar shared stores by lane 0, and one CTA barrier |
| 2195-2203 | exactly four `ld.global.v4.b32` and 32 BF16-to-FP32 conversions per unrolled iteration |
| 2210-2273 | packed two-lane decay, scalar h-k FMA chains, and four independent 16-lane butterfly reductions |
| 2275-2355 | four scalar V loads, four scalar residual/gate products, packed two-lane rank-one updates, scalar h-q FMA chains, four butterfly reductions, and four lane-zero scalar BF16 output stores |
| 2403-2410 | four BF16x2 packs and one 128-bit state store for each of four V rows |

For the reviewed tile-64 specialization, the source compiler physically
duplicates the complete Phase-1 body twice.  Tile 128 must select the same body
four times; changing the row/group mapping or introducing a serial runtime loop
would not be a faithful port.

## TIRx module and benchmark contract

- Registry name: `gdn_decode_bf16_wide_vec_t1`; supported compute capability
  is exactly 10.
- Correctness uses the direct FlashInfer
  `gated_delta_rule_t1_wide_vec(..., tile_v=...)` oracle, not the top-level
  dispatcher, so every semantic branch and both tile sizes execute this body.
- Performance uses the source production domain only, same-pool contiguous
  state, Q/K L2 normalization, state update enabled, cache disabled, and the 18
  Qwen3-Next TP shapes recorded in the module's `BENCH_CONFIGS`.
- The final performance result is accepted only from a complete unfiltered
  `python -m tirx_kernels.bench_suite` run and only when every required
  `source_us / tirx_us` ratio is strictly greater than 0.99.

## Instruction selection is a lowering consequence

The port must state the same placement, lane layout, vector width, unrolled
iteration count, and synchronization before relying on these opcodes:

- Q/K use eight scalar `ld.global.b16` operations each per active Phase-0
  thread. V uses four scalar `ld.global.b16` operations at both tile sizes.
  State uses four ordinary 16-byte vector loads and, when enabled, four
  ordinary 16-byte vector stores per body.
- shared publication/consumption uses scalar `st.shared.b32` and
  `ld.shared.b32`; publication is one `bar.sync 0`.
- both reductions use `shfl.sync.bfly.b32` with offsets 8, 4, 2, 1 and clamp
  31, intentionally forming two independent 16-lane reductions per warp.
- decay and rank-one update use `fma.rn.f32x2`; dot products remain ordered
  scalar `fma.rn.f32` chains.
- L2 and gate math use `rsqrt.approx.ftz.f32`, `ex2.approx.ftz.f32`,
  `lg2.approx.ftz.f32`, and `rcp.rn.f32`.
- output uses scalar `cvt.rn.bf16.f32`/`st.global.b16`; state uses four
  `cvt.rn.bf16x2.f32` followed by one `st.global.v4.b32` per row.

The implementation may express these primitives through plain TIRx helpers,
but may not use tile primitives or substitute a mathematically equivalent
thread mapping, reduction topology, memory width, fast-math sequence, or
packed-FMA schedule.
