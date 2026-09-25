<!--
Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: Apache-2.0

This execution sketch documents a TIRx port of FlashInfer's
gdn_decode_mtp.py. See LICENSE, NOTICE, and licenses/ for applicable terms.
-->

# GDN decode FP32 MTP warp SM100: execution sketch

This is a non-executable execution sketch for FlashInfer's CuTeDSL
`gdn_verify_kernel_mtp`.  The target is
[`tirx_kernels/flashinfer/gdn_decode/gdn_decode_fp32_mtp_warp.py`](../../../tirx_kernels/flashinfer/gdn_decode/gdn_decode_fp32_mtp_warp.py).
The frozen source commit is
`f2e04400e330fb2debe0bf8730d9424a1d37927f`; the source SHA256 is
`657d99af36ace6dffb05f4ff31061ce9c80419468ff948873098c42d6b7ecf50`.

The port is the warp-specialized production body only: `B*HV > 128`, T in
2..8, K=V=128, BF16 Q/K/V/a/b/output, FP32 state/intermediate/A_log/dt_bias,
four full warps, four adjacent K values per lane, and SM100 packed FP32x2 FMA
only in the ILP4 body.  Inline MTP, varlen, FP16 activation, recovery,
accepted-step truncation, and disabled output are out of scope.

## Frozen picker and launch

`work_units = B*HV`; the source selects exactly:

| reachable warp domain | TILE_V | ILP_ROWS | USE_SMEM_V |
| --- | ---: | ---: | --- |
| 129..448 and T=2 | 16 | 2 | false |
| 129..448 and T>=3 | 32 | 4 | false |
| 449..1024 | 32 | 4 | false |
| >1024, state update enabled, T=2 | 64 | 8 | false |
| >1024, all other reachable modes | 64 | 4 | true |

The launch is `grid=(B*HV*(128/TILE_V),1,1)`, `block=(128,1,1)`, and
dynamic shared bytes are
`8*T*(128+8) + 8*T + 6*T*TILE_V + 128`.  The source reserves sVdata and
sOutput even when `USE_SMEM_V=false`.

## Pipeline at a glance

There is one producer/publication barrier in every specialization and one
additional output-drain barrier only with SMEM-V.

| physical lanes | role | publication/reuse edge |
| --- | --- | --- |
| warp 0, lanes 0..31 | for every t: load and normalize Q/K, publish sQ/sK, form g and beta, publish sG/sBeta; with SMEM-V, lanes `<TILE_V` also publish V | first CTA barrier |
| warps 1..3 | prefetch the first 2 or 4 state rows while warp 0 computes; with SMEM-V, lanes `<TILE_V` publish V for every t | first CTA barrier |
| all four warps | each warp owns `TILE_V/4` V rows; it runs the selected ILP2/4/8 body across all T with state resident in registers | sQ/sK/sG/sBeta are immutable after first barrier |
| warp lane 0 | direct BF16 output stores, or BF16 sOutput publication | direct path has no second barrier |
| all lanes with SMEM-V | after second barrier, lanes `<TILE_V` cooperatively drain one BF16 output per t | second CTA barrier |

## Primitive vocabulary

Structural primitives do not copy or compute values:

```python
specialize(...)       # compile-time source variant
launch(...)           # grid, block, launch bounds, dynamic shared bytes
raw_shared(...)       # single dynamic shared allocation
view(...)             # typed view with explicit offset/shape/stride
alias(...)            # identical read/write pool view
reg_tile(...)         # lane-private registers
```

Copies and primitive arithmetic expose direction and instruction family:

```python
copy_g2r(src, dst=None, predicate=None)
copy_s2r(src, dst=None)
copy_r2s(src, dst)
copy_r2g(src, dst, predicate=None)
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
select(pred, true_value, false_value)
shuffle_xor(src, delta, member_mask, clamp)
warp_uniform(src, source_lane, member_mask, clamp)
cta_barrier()
```

`fma(..., lanes=2)` is exactly one `fma.rn.f32x2`.  No `normalize`,
`softplus`, `sigmoid`, `reduce`, `delta_rule`, state-update, or tile primitive
is permitted.

## Complete sketch

```python
@specialize(
    T=(2,3,4,5,6,7,8), H=(16,8,4,2), HV=(64,32,16,8), K=128, V=128,
    TILE_V=(16,32,64), ILP_ROWS=(2,4,8), USE_SMEM_V=(False,True),
    USE_QK_L2NORM=(False,True), DISABLE_STATE_UPDATE=(False,True),
    CACHE_INTERMEDIATE_STATES=(False,True), SAME_POOL=(False,True),
    PER_TOKEN_POOL_SCATTER=(False,True),
    USE_PACKED_FMA=True, SOFTPLUS_BETA=1.0,
    SOFTPLUS_THRESHOLD=20.0, SCALE=1.0/sqrt(128), target="sm_100a",
)
@require(B*HV > 128, K == 128, V == 128, HV % H == 0)
@require((TILE_V,ILP_ROWS,USE_SMEM_V) == frozen_picker(B,T,HV,DISABLE_STATE_UPDATE))
@launch(
    grid=(B*HV*(128//TILE_V),1,1), block=(128,1,1), num_warps=4,
    dynamic_smem_bytes=8*T*136 + 8*T + 6*T*TILE_V + 128,
)
def gdn_decode_fp32_mtp_warp(
    state, intermediate, A_log, a, dt_bias, q, k, v, b_gate, output,
    read_indices, write_indices, ssm_state_indices,
    state_slot_stride, state_head_stride,
    q_batch_stride, k_batch_stride, v_batch_stride, B,
):
    tid = thread_id(axis="x", extent=128)
    lane = tid % 32
    warp_raw = tid // 32
    warp = warp_uniform(warp_raw, source_lane=0,
                        member_mask=0xffffffff, clamp=31)
    # instruction_selection: mov.u32 %tid.x, shr.u32, then
    # shfl.sync.idx.b32(dst,src,0,31,0xffffffff); extent: one warp-uniform
    # warp index and one lane index per thread
    k_start = lane * 4

    linear_cta = cta_id(axis="x")
    v_tile = linear_cta % (128 // TILE_V)
    tmp = linear_cta // (128 // TILE_V)
    hv = tmp % HV
    n = tmp // HV
    h = hv // (HV // H)

    cache_idx = copy_g2r(read_indices[n])
    # instruction_selection: ld.global.b32; extent: one scalar per CTA thread
    A_value = copy_g2r(A_log[hv])
    dt_value = copy_g2r(dt_bias[hv])
    # instruction_selection: ld.global.b32; extent: A_log then dt_bias,
    # two uniform FP32 loads before the negative-read predicate

    smem = raw_shared(dtype="u8", bytes=dynamic_smem_bytes, alignment=16)
    QK_BYTES = 4*((T-1)*136 + 128)
    GATE_BYTES_ALIGNED = align_up(4*T, 16)
    sQ = view(smem, "f32", (T,128), stride=(136,1), byte_offset=0)
    sK = view(smem, "f32", (T,128), stride=(136,1),
              byte_offset=QK_BYTES)
    sG = view(smem, "f32", (T,), byte_offset=2*QK_BYTES)
    sBeta = view(smem, "f32", (T,),
                 byte_offset=2*QK_BYTES+GATE_BYTES_ALIGNED)
    sV = view(smem, "f32", (T,TILE_V),
              byte_offset=2*QK_BYTES+2*GATE_BYTES_ALIGNED)
    sOutput = view(smem, "bf16", (T,TILE_V),
                   byte_offset=2*QK_BYTES+2*GATE_BYTES_ALIGNED
                               +4*T*TILE_V)

    r_h = reg_tile("f32", (8,4))
    r_q = reg_tile("f32", (4,))
    r_k = reg_tile("f32", (4,))
    r_q_bf16 = reg_tile("bf16", (4,))
    r_k_bf16 = reg_tile("bf16", (4,))

    if cache_idx >= 0:
        write_raw = copy_g2r(write_indices[n])
        # instruction_selection: ld.global.b32; extent: one scalar when final
        # pool write addressing survives specialization
        write_idx = select(write_raw < 0, cache_idx, write_raw)
        # instruction_selection: setp.lt.s32 + selp.b32; extent: one safe-view clamp
        read_state = state_view(
            cache_idx, hv, state_slot_stride, state_head_stride)
        if SAME_POOL:
            write_state = alias(read_state, read_state)
        else:
            write_state = state_view(
                write_idx, hv, state_slot_stride, state_head_stride)
        # Compact flat, slot-padded, and head-strided native 4-D pools share
        # one complete runtime addressing representation.  Both independent
        # strides participate in Int64 address arithmetic;
        # instruction_selection: cvt.s64.s32 plus mul/mad.wide address family

        # ================================================================
        # Phase 1: warp-specialized producer and state prefetch
        # ================================================================
        if warp == 0:
            for t in static_range(T):
                for i in static_range(4):
                    r_q_bf16[i] = copy_g2r(q[n,t,h,k_start+i])
                    # instruction_selection: ld.global.b16; extent: four scalar Q loads
                for i in static_range(4):
                    r_k_bf16[i] = copy_g2r(k[n,t,h,k_start+i])
                    # instruction_selection: ld.global.b16; extent: four scalar K loads
                for i in static_range(4):
                    r_q[i] = cast("f32", r_q_bf16[i])
                    r_k[i] = cast("f32", r_k_bf16[i])
                    # instruction_selection: cvt.f32.bf16; extent: Q then K,
                    # four of each for every t

                if USE_QK_L2NORM:
                    sum_q = 0.0
                    sum_k = 0.0
                    for i in static_range(4):
                        sum_q = fma(r_q_bf16[i], r_q_bf16[i], sum_q)
                        sum_k = fma(r_k_bf16[i], r_k_bf16[i], sum_k)
                        # instruction_selection: fma.rn.f32.bf16; extent:
                        # two scalar square accumulates, four loop instances
                    for delta in (16,8,4,2,1):
                        peer_q = shuffle_xor(
                            sum_q, delta, member_mask=-1, clamp=31)
                        peer_k = shuffle_xor(
                            sum_k, delta, member_mask=-1, clamp=31)
                        # instruction_selection: shfl.sync.bfly.b32; extent:
                        # two scalar shuffles at each of five full-warp stages
                        sum_q = add(sum_q, peer_q)
                        sum_k = add(sum_k, peer_k)
                        # instruction_selection: add.f32; extent: two scalar
                        # adds at each of five full-warp stages
                    q_eps = add(sum_q, 1.0e-6)
                    k_eps = add(sum_k, 1.0e-6)
                    # instruction_selection: add.f32; extent: two scalars per t
                    q_inv = rsqrt(q_eps)
                    k_inv = rsqrt(k_eps)
                    # instruction_selection: rsqrt.approx.ftz.f32; extent:
                    # two scalars per t
                    q_factor = mul(q_inv, SCALE)
                    # instruction_selection: mul.f32; extent: once per t
                    for i in static_range(4):
                        r_q[i] = mul(r_q[i], q_factor)
                        r_k[i] = mul(r_k[i], k_inv)
                        # instruction_selection: mul.f32 x2; extent: four pairs
                else:
                    for i in static_range(4):
                        r_q[i] = mul(r_q[i], SCALE)
                        # instruction_selection: mul.f32; extent: four per t

                for i in static_range(4):
                    copy_r2s(r_q[i], sQ[t,k_start+i])
                    copy_r2s(r_k[i], sK[t,k_start+i])
                    # instruction_selection: st.shared.b32; extent: eight
                    # scalar stores per t in both L2-on and L2-off variants

                a_bits = copy_g2r(a[n,t,hv])
                b_bits = copy_g2r(b_gate[n,t,hv])
                # instruction_selection: ld.global.b16; extent: two scalar loads per t
                b_value = cast("f32", b_bits)
                # instruction_selection: cvt.f32.bf16; extent: one scalar,
                # before the mixed a/dt add
                x = add(a_bits, dt_value)
                # instruction_selection: add.rn.f32.bf16; extent: one mixed add

                x_log2 = mul(x, LOG2E)
                # instruction_selection: mul.f32; extent: one base conversion
                exp_x = exp2(x_log2)
                # instruction_selection: ex2.approx.ftz.f32; extent: one scalar
                one_plus = add(1.0, exp_x)
                # instruction_selection: add.f32; extent: one scalar
                softplus_log2 = log2(one_plus)
                # instruction_selection: lg2.approx.ftz.f32; extent: one scalar
                softplus = mul(softplus_log2, LN2)
                # instruction_selection: mul.f32; extent: one natural-log conversion
                pred = setp_le(x, 20.0)
                use_softplus = select(pred, 1.0, 0.0)
                # instruction_selection: setp.le.f32 + selp.f32; extent: one scalar
                direct_weight = sub(1.0, use_softplus)
                # instruction_selection: sub.f32; extent: one scalar
                direct = mul(direct_weight, x)
                # instruction_selection: mul.f32; extent: one scalar
                softplus_x = fma(use_softplus, softplus, direct)
                # instruction_selection: fma.rn.f32; extent: one selected result

                A_log2 = mul(A_value, LOG2E)
                # instruction_selection: mul.f32; extent: one scalar
                exp_A = exp2(A_log2)
                # instruction_selection: ex2.approx.ftz.f32; extent: one scalar
                decay_positive = mul(exp_A, softplus_x)
                # instruction_selection: mul.f32; extent: one positive scalar;
                # the source negation is folded into the final base conversion
                neg_b_log2 = mul(b_value, -LOG2E)
                # instruction_selection: mul.f32; extent: one scalar
                exp_neg_b = exp2(neg_b_log2)
                # instruction_selection: ex2.approx.ftz.f32; extent: one scalar
                beta_denom = add(1.0, exp_neg_b)
                # instruction_selection: add.f32; extent: one scalar
                beta = reciprocal(beta_denom)
                # instruction_selection: rcp.rn.f32; extent: one scalar
                g_log2 = mul(decay_positive, -LOG2E)
                # instruction_selection: mul.f32; extent: one scalar with the
                # folded negative sign, after beta construction
                g = exp2(g_log2)
                # instruction_selection: ex2.approx.ftz.f32; extent: one scalar
                copy_r2s(g, sG[t])
                copy_r2s(beta, sBeta[t])
                # instruction_selection: L2-on uses st.shared.b32 x2 per t;
                # reviewed L2-off PTX combines paired adjacent t values as one
                # st.shared.v2.b32 for sG and one for sBeta, with scalar
                # st.shared.b32 tails when T is odd.  Every warp-0 lane writes
                # the same warp-uniform values exactly as source.

                if USE_SMEM_V and tid < TILE_V:
                    v_bits = copy_g2r(v[n,t,hv,v_tile*TILE_V+tid])
                    # instruction_selection: ld.global.b16; extent: one per t
                    v_value = cast("f32", v_bits)
                    # instruction_selection: cvt.f32.bf16; extent: one per t
                    copy_r2s(v_value, sV[t,tid])
                    # instruction_selection: st.shared.b32; extent: one per t
        else:
            v_prefetch = v_tile*TILE_V + warp*(TILE_V//4)
            for row in static_range(min(ILP_ROWS,4)):
                copy_g2r(read_state[v_prefetch+row,k_start:k_start+4], r_h[row])
                # instruction_selection: ld.global.v2.b64; extent: two rows
                # for ILP2 or four rows for ILP4.  ILP8 emits zero prefetch
                # loads after DCE because its body unconditionally reloads all
                # eight rows.
            if USE_SMEM_V:
                for t in static_range(T):
                    if tid < TILE_V:
                        v_bits = copy_g2r(v[n,t,hv,v_tile*TILE_V+tid])
                        # instruction_selection: ld.global.b16; extent: once per t
                        v_value = cast("f32", v_bits)
                        # instruction_selection: cvt.f32.bf16; extent: once per t
                        copy_r2s(v_value, sV[t,tid])
                        # instruction_selection: st.shared.b32; extent: once per t

        cta_barrier()
        # instruction_selection: bar.sync 0; extent: one CTA publication edge

        # ================================================================
        # Phase 2: constexpr-selected ILP2 / ILP4 / ILP8 body
        # ================================================================
        for row_chunk in static_range((TILE_V//4)//ILP_ROWS):
            v_base = v_tile*TILE_V + warp*(TILE_V//4) + row_chunk*ILP_ROWS
            if ILP_ROWS == 8 or warp == 0 or row_chunk > 0:
                for row in static_range(ILP_ROWS):
                    copy_g2r(read_state[v_base+row,k_start:k_start+4], r_h[row])
                    # instruction_selection: ILP8 uses ld.global.v4.b32 for
                    # every row; ILP2/4 use ld.global.v2.b64 in the first
                    # emitted row chunk and ld.global.v4.b32 in later unrolled
                    # chunks.  The first prefetched chunk in warps 1..3 emits
                    # zero duplicate loads for ILP2/4.  ILP8 deliberately
                    # reloads all eight rows after the barrier.

            for t in static_range(T):
                copy_s2r(sQ[t,k_start:k_start+4], r_q)
                copy_s2r(sK[t,k_start:k_start+4], r_k)
                # instruction_selection: ILP4 uses ld.shared.v2.b64 for each
                # four-float vector; ILP2/8 reviewed PTX uses
                # ld.shared.v4.b32; extent: one Q and one K vector per t/body
                g = copy_s2r(sG[t])
                beta = copy_s2r(sBeta[t])
                # instruction_selection: ld.shared.b32 x2; extent: once per t/body

                sums = reg_tile("f32", (ILP_ROWS,))
                if ILP_ROWS == 4:
                    sums_lo = zeros("f32", 4)
                    sums_hi = zeros("f32", 4)
                    for pair in static_range(2):
                        for row in static_range(4):
                            r_h[row,2*pair] = mul(r_h[row,2*pair], g)
                            r_h[row,2*pair+1] = mul(r_h[row,2*pair+1], g)
                            # instruction_selection: mul.f32 x2; extent:
                            # two pairs by four rows
                            sums_lo[row], sums_hi[row] = fma(
                                r_h[row,2*pair:2*pair+2],
                                r_k[2*pair:2*pair+2],
                                (sums_lo[row],sums_hi[row]), lanes=2)
                            # instruction_selection: fma.rn.f32x2; extent:
                            # one per pair per row
                    for row in static_range(4):
                        sums[row] = add(sums_lo[row], sums_hi[row])
                        # instruction_selection: add.f32; extent: one per row
                else:
                    for row in static_range(ILP_ROWS):
                        sums[row] = 0.0
                    for i in static_range(4):
                        for row in static_range(ILP_ROWS):
                            r_h[row,i] = mul(r_h[row,i], g)
                            # instruction_selection: mul.f32; extent: one per element
                            sums[row] = fma(r_h[row,i], r_k[i], sums[row])
                            # instruction_selection: fma.rn.f32; extent:
                            # one per element; ILP2 and ILP8 are deliberately scalar

                for delta in (16,8,4,2,1):
                    for row in static_range(ILP_ROWS):
                        peer = shuffle_xor(
                            sums[row], delta, member_mask=-1, clamp=31)
                        # instruction_selection: shfl.sync.bfly.b32; extent:
                        # every row at five full-warp stages
                        sums[row] = add(sums[row], peer)
                        # instruction_selection: add.f32; extent: every row at
                        # five full-warp stages

                residual = reg_tile("f32", (ILP_ROWS,))
                for row in static_range(ILP_ROWS):
                    if USE_SMEM_V:
                        v_value = copy_s2r(sV[t,v_base-v_tile*TILE_V+row])
                        # instruction_selection: ld.shared.b32; extent: one per row
                        delta = sub(v_value, sums[row])
                        # instruction_selection: sub.f32; extent: one per row
                    else:
                        v_bits = copy_g2r(v[n,t,hv,v_base+row])
                        # instruction_selection: ld.global.b16; extent: one per row
                        delta = sub(cast("f32", v_bits), sums[row])
                        # instruction_selection: sub.rn.f32.bf16; extent: one
                        # mixed BF16/FP32 subtract per row
                    residual[row] = mul(delta, beta)
                    # instruction_selection: mul.f32; extent: one per row

                if ILP_ROWS != 4:
                    for i in static_range(4):
                        for row in static_range(ILP_ROWS):
                            r_h[row,i] = fma(r_k[i], residual[row], r_h[row,i])
                            # instruction_selection: fma.rn.f32; extent:
                            # scalar rank-one update for every ILP2/8 element
                    if CACHE_INTERMEDIATE_STATES:
                        for row in static_range(ILP_ROWS):
                            copy_r2g(r_h[row], intermediate[n,t,hv,v_base+row,k_start:k_start+4])
                            # instruction_selection: ILP8 uses st.global.v4.b32
                            # for every row/t; ILP2/T2 uses st.global.v4.b32 at
                            # t=0 and st.global.v2.b64 at t=1; extent: one
                            # 16-byte store per row before the output dot
                    if PER_TOKEN_POOL_SCATTER:
                        scatter_slot = copy_g2r(ssm_state_indices[n,t])
                        # instruction_selection: ld.global.b32; extent: one per t/body
                        scatter_base = cast("i64", scatter_slot) * state_slot_stride
                        # instruction_selection: cvt.s64.s32 plus
                        # mad.wide/mul.lo.s64 address family; Int64 is mandatory
                        # for B128/T8/HV64
                        for row in static_range(ILP_ROWS):
                            copy_r2g(r_h[row], scatter_state(
                                scatter_base,hv,state_head_stride,v_base+row,k_start))
                            # instruction_selection: st.global.v4.b32; extent:
                            # one 16-byte store per row for both ILP2 and ILP8,
                            # before the output dot product

                out = reg_tile("f32", (ILP_ROWS,))
                if ILP_ROWS == 4:
                    out_lo = zeros("f32", 4)
                    out_hi = zeros("f32", 4)
                    for pair in static_range(2):
                        for row in static_range(4):
                            r_h[row,2*pair:2*pair+2] = fma(
                                r_k[2*pair:2*pair+2],
                                (residual[row],residual[row]),
                                r_h[row,2*pair:2*pair+2], lanes=2)
                            # instruction_selection: fma.rn.f32x2; extent:
                            # one rank-one update per pair per row
                            out_lo[row], out_hi[row] = fma(
                                r_h[row,2*pair:2*pair+2],
                                r_q[2*pair:2*pair+2],
                                (out_lo[row],out_hi[row]), lanes=2)
                            # instruction_selection: fma.rn.f32x2; extent:
                            # one output accumulate per pair per row
                    for row in static_range(4):
                        out[row] = add(out_lo[row], out_hi[row])
                        # instruction_selection: add.f32; extent: one per row
                else:
                    for row in static_range(ILP_ROWS):
                        out[row] = 0.0
                    for i in static_range(4):
                        for row in static_range(ILP_ROWS):
                            out[row] = fma(r_h[row,i], r_q[i], out[row])
                            # instruction_selection: fma.rn.f32; extent:
                            # scalar output accumulate for ILP2/8

                for delta in (16,8,4,2,1):
                    for row in static_range(ILP_ROWS):
                        peer = shuffle_xor(
                            out[row], delta, member_mask=-1, clamp=31)
                        # instruction_selection: shfl.sync.bfly.b32; extent:
                        # every row at five full-warp stages
                        out[row] = add(out[row], peer)
                        # instruction_selection: add.f32; extent: every row at
                        # five full-warp stages
                if lane == 0:
                    for row in static_range(ILP_ROWS):
                        out_bits = cast("bf16", out[row], rounding="rn")
                        # instruction_selection: cvt.rn.bf16.f32; extent: one per row
                        if USE_SMEM_V:
                            copy_r2s(out_bits, sOutput[t,v_base-v_tile*TILE_V+row])
                            # instruction_selection: st.shared.b16; extent: one per row
                        else:
                            copy_r2g(out_bits, output[n,t,hv,v_base+row])
                            # instruction_selection: st.global.b16; extent: one per row

                if ILP_ROWS == 4:
                    if CACHE_INTERMEDIATE_STATES:
                        for row in static_range(4):
                            copy_r2g(r_h[row], intermediate[n,t,hv,v_base+row,k_start:k_start+4])
                            # instruction_selection: st.global.v2.b64; extent:
                            # one 16-byte store per row after the ILP4 output
                    if PER_TOKEN_POOL_SCATTER:
                        scatter_slot = copy_g2r(ssm_state_indices[n,t])
                        # instruction_selection: ld.global.b32; extent: one per t/body
                        scatter_base = cast("i64", scatter_slot) * state_slot_stride
                        # instruction_selection: cvt.s64.s32 plus
                        # mad.wide/mul.lo.s64 address family; Int64 is mandatory
                        # for B128/T8/HV64
                        for row in static_range(4):
                            copy_r2g(r_h[row], scatter_state(
                                scatter_base,hv,state_head_stride,v_base+row,k_start))
                            # instruction_selection: st.global.v2.b64; extent:
                            # one 16-byte store per row after the ILP4 output

            if not DISABLE_STATE_UPDATE and not PER_TOKEN_POOL_SCATTER:
                if write_raw >= 0:
                    for row in static_range(ILP_ROWS):
                        copy_r2g(r_h[row], write_state[v_base+row,k_start:k_start+4])
                        # instruction_selection: ILP2 cache-off uses
                        # st.global.v4.b32 while ILP2 cache-on uses
                        # st.global.v2.b64; ILP4 uses st.global.v2.b64; ILP8
                        # uses st.global.v4.b32.  Extent: one 16-byte store per
                        # row after all T, guarded by the raw write index.

        if USE_SMEM_V:
            cta_barrier()
            # instruction_selection: bar.sync 0; extent: second CTA barrier
            for t in static_range(T):
                if tid < TILE_V:
                    out_bits = copy_s2r(sOutput[t,tid])
                    # instruction_selection: ld.shared.b16; extent: one per t
                    copy_r2g(out_bits, output[n,t,hv,v_tile*TILE_V+tid])
                    # instruction_selection: st.global.b16; extent: one per t
```

The entire producer, first barrier, ILP body, optional second barrier, and
output drain are inside `cache_idx >= 0`.  A negative read index therefore
leaves output and every mutable state buffer untouched.  A negative write
index still permits output and dense intermediate caching but suppresses only
the final pool write.  Scatter suppresses the final pool write because its
last per-token store is already h_T.

## Ownership and lifetimes

| storage | owner | lifetime / alias rule |
| --- | --- | --- |
| sQ/sK | warp 0 publishes, all warps consume | all T rows live across the first barrier and entire state loop |
| sG/sBeta | every warp-0 lane writes identical values, all warps consume | all T scalars live across the first barrier |
| sV | all warps cooperatively publish when enabled | immutable after first barrier |
| sOutput | lane 0 of each warp publishes disjoint V rows | live until second barrier and cooperative drain |
| r_h `[8,4]` | one lane owns four K values of up to eight V rows | fixed source allocation outside the read-index branch; the selected prefix remains FP32 and register-resident across all T |
| read/write pool views | one CTA owns a V tile and each lane owns four K values | exact alias for same-pool; Int64 independent roots for split/padded |
| intermediate | batch-indexed `(n,t,hv,V,K)` | FP32 dense cache; never pool-slot indexed |

No race relies on scheduling.  Warp 0 completes all producer stores before
the first barrier; each `(V,K)` state element has one lane owner; only warp
lane 0 writes each scalar output; and the SMEM output path drains only after
the second barrier.

## Fresh source-PTX evidence

Writer-side cache-bypassed line-info builds live under
`.porting/gdn_decode_fp32_mtp_warp/source_ptx_stage2/`.  They cover ILP2/T2,
ILP4/T4, ILP8/T2, SMEM-V/T8, L2-off split/negative, and flat scatter.  PTX
SHA256 values are respectively:

- `e8755b988ce40bc54dde914fd70246073a2bd69a7ee1e2534c1d7934492ac462`
- `b1203a510126d38b3ef35ccd010d18822aefd4ba5acd80aa74c645d97d147078`
- `cda5f318d814de09e5d50c6c8e3dd6355f5dcce53124f4bc990b000d2e7306ba`
- `2b2ed71f4015276040f86c18d6addd495b468ad172af386af4cf8977a87410fd`
- `073d8f9f099b07ff7f4587fc2b5ea30681e51e77bb68cc3f9180bfaefcc9e244`
- `6b6ffdc4381e9c7fce57d6926af3710c22d95ae35a8074cc917b16930f91c63d`

The SMEM-V build has two `bar.sync`, BF16 `st.shared.b16`/`ld.shared.b16`,
and a cooperative scalar BF16 global drain.  Every other build has exactly
one `bar.sync`.  ILP4 contains native `fma.rn.f32x2`; ILP2 and ILP8 contain
only scalar `fma.rn.f32` in the state body.  State cache and pool traffic is
16-byte FP32 vector traffic; output is scalar BF16 traffic.

The frozen source currently fails CuTeDSL construction for the combined
native-stride 4-D pool plus per-token scatter specialization because its
scatter site builds a three-coordinate tile against a four-dimensional
tensor.  The port still specifies the intended padded scatter address rule;
correctness will compare it to the same frozen algorithm on an equivalent
contiguous reference pool while separately testing the frozen native-stride
read/final-write path.

## TIRx integration and acceptance contract

- Registry name is `gdn_decode_fp32_mtp_warp`, category `flashinfer`, compute
  capability 10.
- `CONFIGS` covers every T, all source picker branches, L2 on/off, cache and
  update modes, same/split/negative indices, contiguous and padded pools,
  packed QKV, flat/padded scatter, the Int64 stress shape, and TP1/2/4/8.
- `BENCH_CONFIGS` contains 97 source-reachable production cases: the 49 TP1
  cases plus 48 TP2/4/8 boundary cases.  Every performance case has dense
  cache and final state update enabled.
- Correctness uses the frozen `flashinfer.gdn_decode.gated_delta_rule_mtp`
  source body and independently mutable inputs.
- Performance is accepted only from a complete target-filtered
  `python -m tirx_kernels.bench_suite` run, with every exact
  `flashinfer_cutedsl_us / tirx_us > 0.99`; direct timing is diagnostic only.

## Instruction selection is a lowering consequence

The implementation must first preserve lane ownership, load/store width,
warp-specialized overlap, source-order gate math, reduction topology,
register lifetimes, and synchronization.  It may then use plain TIRx PTX
helpers for the named instructions.  It may not use tile primitives, change
full-warp butterfly offsets `(16,8,4,2,1)`, replace the fast exp/log chain,
pack ILP2/8 arithmetic, scalarize FP32 state vectors, move ILP4 cache stores
ahead of the source's output chain, or replace SMEM-V's two-barrier protocol.
