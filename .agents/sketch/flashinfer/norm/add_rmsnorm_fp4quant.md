<!--
Copyright (c) 2025 by FlashInfer team.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: Apache-2.0

This execution sketch documents a TIRx port of FlashInfer's CuTe-DSL
AddRMSNormFP4QuantKernel. See LICENSE, NOTICE, and licenses/ for applicable terms.
-->

# FlashInfer Add RMSNorm FP4 quantization SM100: execution sketch

This non-executable sketch freezes the implementation shape of FlashInfer's
CuTe-DSL `AddRMSNormFP4QuantKernel` for
[`tirx_kernels/flashinfer/norm/add_rmsnorm_fp4quant.py`](../../../../tirx_kernels/flashinfer/norm/add_rmsnorm_fp4quant.py).
It records the source execution skeleton rather than a mathematically equivalent
replacement and is frozen after its first independent reviewer PASS.

The source is fixed at FlashInfer commit
`f2e04400e330fb2debe0bf8730d9424a1d37927f`:

- `flashinfer/cute_dsl/add_rmsnorm_fp4quant.py`, SHA256
  `1d034042a6b31c4f7301626a3d6ee7cff7cfbd67715d95ae3f0918f70b62d881`;
- `flashinfer/cute_dsl/fp4_common.py`, SHA256
  `ff490c11853603634fd19c4558021b03db08b409dc54d24146f1bdc66a6b784f`;
- public alias/wrapper in `flashinfer/norm/__init__.py`, SHA256
  `226a88f5fb14e78e06e1be79020edcae01bfa9e53e677bd485373ea4d51cffcb`.

Writer-owned fresh line-info PTX and clean MLIR are under
`.porting/flashinfer_add_rmsnorm_fp4quant/source_exports/writer/`. The accepted
profiles explicitly pass PDL false unless their name contains `pdl`; directories
ending `_invalid_auto_pdl` are preserved dispatch-diagnostic artifacts and are not
instruction-selection evidence. The accepted matrix covers FP16/BF16, 2-D/3-D,
column and row tails, linear/swizzled/dual scale layouts, optional YNorm, NV/MX
scale paths, the Add-specific rolled-loop boundaries, cluster2/16, and PDL.

The supported device family is compact FP16/BF16 X/residual/W, packed E2M1 Y,
block size 16/32, E4M3/UE8M0 scale bytes, row-major or padded 128x4 swizzled
scales, optional simultaneous swizzled+row-major scales, optional input-dtype
YNorm, PDL off/on, host-flattened 3-D, and cluster sizes 1/2/4/8/16. There is no
strided device ABI, tile primitive, TMA, additional pipeline stage, max-register
cap, or launch bound.

## Source dispatch and launch formulae

```python
VEC = 8
ELEM_BYTES = 2
MAX_SMEM = 232448  # B200 opt-in value of the reviewed environment

def threads_per_row(hcta):
    if hcta <= 64: return 8
    if hcta <= 128: return 16
    if hcta <= 3072: return 32
    if hcta <= 6144: return 64
    if hcta <= 16384: return 128
    return 256

def source_config(H, cluster_n):
    hcta = H // cluster_n
    tpr = threads_per_row(hcta)
    threads = 128 if hcta <= 16384 else 256
    rows = threads // tpr
    warps_per_row = max(tpr // 32, 1)
    vec_blocks = max(1, ceil_div(hcta // VEC, tpr))
    cols = VEC * vec_blocks * tpr
    tile_bytes = rows * cols * ELEM_BYTES
    if cluster_n == 1:
        smem = 4 * tile_bytes + rows * warps_per_row * 4
    else:
        smem = 2 * tile_bytes + rows * warps_per_row * cluster_n * 4 + 8
    return hcta,tpr,threads,rows,warps_per_row,vec_blocks,cols,tile_bytes,smem

CLUSTER_N = next(
    (c for c in (1,2,4,8,16)
     if H % c == 0 and source_config(H,c).smem <= MAX_SMEM),
    16,
)
```

Launch is `grid=(ceil_div(runtime_M,ROWS),CLUSTER_N,1)`,
`block=(THREADS,1,1)`, and cluster `(1,CLUSTER_N,1)` iff clustered. Runtime ABI
is `(X,R,W,Y,S,S_linear,YNorm,global_scale,M:i32,eps:f32)`. Disabled optional
outputs still occupy fixed dummy-buffer slots but are never accessed. PTX has
`.reqntid`, `.extern .shared .align 1024`, and no `.maxnreg`/minimum-CTA directive.

On B200, H16384 is cluster1, H32768 is cluster2, and H524288 is cluster16.
H1048576 falls back to cluster16 but its Add-specific shared allocation exceeds
the opt-in limit, so it is not a launchable validation profile.

## Source-proven loop shape

| phase | specialization | PTX structure |
| --- | --- | --- |
| X/R/W async issue and X/R shared load | `VEC_BLOCKS` | always statically expanded |
| H add, narrow, square, local reduction | `8*VEC_BLOCKS` values | statically expanded; add/square use `*.f32x2` per adjacent pair |
| block16, no YNorm | `SF_PER_THREAD <= 2` | completely expanded |
| block16, no YNorm, cluster1 shared path | `SF_PER_THREAD >= 3` | one-SF rolled body, cursor `+=1` |
| block16 with YNorm | `SF_PER_THREAD == 1` | expanded |
| block16 with YNorm, cluster1 shared path | `SF_PER_THREAD >= 2` | one-SF rolled body, cursor `+=1` |
| block32 | `SF_PER_THREAD == 1` | expanded |
| block32, cluster1 shared path | `SF_PER_THREAD >= 2` | one-SF rolled body, cursor `+=1` |
| any clustered global-reload path | reachable rolled trips | two-SF static body, cursor `+=2`, second SF tail-guarded |

Fresh adjacent evidence is H1024/H1536 for block16 without YNorm,
H512/H1024 with YNorm, and H1024/H2048 for block32. The cluster1 backedges map
to source line 647/648 and contain one scale store and one 16-value FP4 store.
Dual-layout alone leaves the block16 trip-two case expanded; dual-layout plus
YNorm follows the YNorm threshold. Independently exported cluster2 H30720
(odd trip 15) and H32768 (trip 16) show the distinct global-reload body: two
scale blocks are statically emitted, the cursor increments by two, and the
second block is guarded for the odd tail. The same two-SF clustered body was
observed for block16 core, dual-layout, YNorm, and block32 UE8M0. These are
path-and-trip rules, not hard-coded H cases, and must not be replaced with the
RMSNormFP4 loop rule.

## CTA roles, publication edges, and storage

| owner | work | publication/reuse edge |
| --- | --- | --- |
| every CTA thread | optional PDL wait; assigned X/R/W vector blocks; H add/square; assigned scale blocks | one contiguous TPR-lane group owns one row |
| each row warp | ordered local sum plus subwarp/full-warp butterfly | scalar warp partial |
| lane 0 per row warp, cluster1 multiwarp | publish partial to reduction shared | CTA barrier, then lanes `<WARPS_PER_ROW` reload |
| lanes `<CLUSTER_N`, clustered CTA | remote-store every warp partial to every peer | mapa + complete-tx mbarrier protocol |
| every clustered CTA | publish its residual H slice, cluster fence/barrier, then quantize the complete row | deliberate identical racing Y/S stores |

One 1024-aligned dynamic shared arena contains:

1. `sX` then `sR`, input dtype `[ROWS,COLS]` each;
2. cluster1 only: `sW` then `sH`, input dtype `[ROWS,COLS]` each;
3. FP32 reduction `[ROWS,WARPS_PER_ROW]` or
   `[ROWS,(WARPS_PER_ROW,CLUSTER_N)]`;
4. clustered only: one 8-byte mbarrier.

`sH` stores the input-dtype-rounded H values, not FP32. Cluster1 phase 3 scalar
loads and widens `sH/sW`; clustered phase 3 uses packed global residual/W loads.
Global scale is read before the first post-reduction barrier in E4M3 branches;
canonical UE8M0 dead-code-eliminates that load. Residual is written after the
first post-reduction barrier. Clustered execution then fences and performs a
second cluster barrier before any full-row global reload.

## Primitive vocabulary

Structural vocabulary does not move or compute data:

```python
specialize(...)  launch(...)  raw_shared(...)  view(...)  reg_fragment(...)
address(...)     swizzled_scale_offset(...)    pair_view(...)
each_static_replica(...)
```

Primitive data/schedule vocabulary:

```python
pdl_wait()  pdl_signal()
cp_async_ca_128(src,dst,source_bytes)  cp_async_commit()  cp_async_wait(0)
load_shared_128(src)  load_shared_b16(src)  load_shared_b32(src)
load_global_128(src)  load_global_b32(src)
store_shared_128(dst,v)  store_shared_b32(dst,v)
store_global_128(dst,v)  store_global_b8(dst,v)  store_global_u64(dst,v)
widen_input_scalar(v,dtype)  cvt_input_pair(hi,lo,dtype)
add_f32(a,b)  add_f32x2(a,b)  mul_f32(a,b)  mul_f32x2(a,b)
fma_rn_f32(a,b,c)
mul_input2(a,b,dtype)  abs_input2(v,dtype)  max_input2(a,b,dtype)
abs_f32(v)  max_f32(a,b)  div_f32(a,b)
rsqrt_fast(v)  rcp_fast(v)  min_f32(a,b)  lg2_fast(v)  ex2_fast(v)
shuffle_xor(v,delta,width)
cvt_e4m3_pair(hi,lo)  cvt_e2m1_pair(hi,lo)  cvt_f16x2_e4m3x2(v)
cvt_f32_f16(v)  cvt_f16_f32(v)  cvt_f32_bf16(v)
cvt_s32_f32_rpi(v)  cvt_f32_s32_rn(v)
and_b32(a,mask)  shift_left_b32(v,n)  shift_right_b32(v,n)
mov_b32_pair(v)  mov_b32_from_b16(lo,hi)
mov_b32_bytes(b0,b1,b2,b3)  mov_b32_as_f32(v)
setp(...)  select(...)  add_s32(...)  sub_s32(...)  cvt_u32_u16(...)
cvt_u16_u32(...)  cvt_u64_u32(...)  shift_left_b64(v,n)  or_b64(a,b)
cta_barrier()  cluster_arrive_relaxed()  cluster_wait()
mbarrier_init(...)  mbarrier_init_fence()  mbarrier_arrive_expect_tx(...)
map_shared_to_peer(...)  remote_store_complete_tx(...)  mbarrier_try_wait(...)
fence_cluster_acq_rel()
elect_one()
```

There is no compound add-RMSNorm, reduction, scale conversion, FP4 quantize,
dual-output, or YNorm primitive.

## Complete execution sketch

```python
@specialize(
    INPUT_DTYPE=("f16","bf16"), H="compile-time multiple of BLOCK_SIZE, >=64",
    BLOCK_SIZE=(16,32), SCALE_FORMAT=("e4m3","ue8m0"),
    SWIZZLED=(False,True), BOTH_SF=(False,True), YNORM=(False,True),
    ENABLE_PDL=(False,True), TARGET="sm_100a",
)
@launch(
    grid=(ceil_div(runtime_M,ROWS),CLUSTER_N,1), block=(THREADS,1,1),
    cluster=((1,CLUSTER_N,1) if CLUSTER_N > 1 else None),
    dynamic_smem_bytes=SMEM_BYTES,
    use_programmatic_dependent_launch=ENABLE_PDL,
)
def flashinfer_add_rmsnorm_fp4quant(
    X,R,W,Y_u8,S_u8,S_linear_u8,YNorm,global_scale,
    runtime_M:i32,runtime_eps:f32,
):
    tid = thread_id_x()

    if ENABLE_PDL:
        pdl_wait()
        # instruction_selection: griddepcontrol.wait; extent: one CTA issue,
        # immediately after tid acquisition and before every CTA-ID read

    block_x = block_id_x()
    cluster_y = block_id_y() if CLUSTER_N > 1 else 0
    lane_in_row = tid % TPR
    row_in_cta = tid // TPR
    lane = tid % 32
    warp = tid // 32
    row_warp = warp // WARPS_PER_ROW
    warp_in_row = warp % WARPS_PER_ROW
    actual_row = block_x * ROWS + row_in_cta
    row_valid = actual_row < runtime_M

    fp4_max_rcp = rcp_fast(6.0)
    # instruction_selection: rcp.approx.ftz.f32; extent: one scalar/thread

    smem = raw_shared("u8",SMEM_BYTES,alignment=1024)
    sX = view(smem,INPUT_DTYPE,(ROWS,COLS),offset=0,alignment=16)
    sR = view(smem,INPUT_DTYPE,(ROWS,COLS),offset=TILE_BYTES,alignment=16)
    if CLUSTER_N == 1:
        sW = view(smem,INPUT_DTYPE,(ROWS,COLS),offset=2*TILE_BYTES,alignment=16)
        sH = view(smem,INPUT_DTYPE,(ROWS,COLS),offset=3*TILE_BYTES,alignment=16)
        reduction = view(smem,"f32",(ROWS,WARPS_PER_ROW),
                         offset=4*TILE_BYTES,alignment=4)
    else:
        reduction = view(smem,"f32",(ROWS,(WARPS_PER_ROW,CLUSTER_N)),
                         offset=2*TILE_BYTES,alignment=4)
        mbar = view(smem,"mbarrier",(1,),
                    offset=2*TILE_BYTES+ROWS*WARPS_PER_ROW*CLUSTER_N*4,
                    alignment=8)

    if CLUSTER_N > 1:
        if tid == 0:
            mbarrier_init(mbar,arrivals=1)
            # instruction_selection: mbarrier.init.shared.b64; extent: one/CTA
        mbarrier_init_fence()
        # instruction_selection: fence.mbarrier_init.release.cluster; one/CTA
        cluster_arrive_relaxed()
        # instruction_selection: barrier.cluster.arrive.relaxed; one/CTA
        cluster_wait()
        # instruction_selection: barrier.cluster.wait; one/CTA

    # Phase 1: source TV layout owns eight adjacent values per lane per vb.
    for vb in static_unroll(VEC_BLOCKS):
        local_col = (lane_in_row + vb*TPR)*8
        absolute_col = cluster_y*COLS + local_col
        col_valid = absolute_col < H
        if row_valid:
            cp_async_ca_128(X[actual_row,absolute_col],
                            sX[row_in_cta,local_col],16 if col_valid else 0)
            # instruction_selection: cp.async.ca.shared.global 16 bytes with
            # source-size 16/0; extent: one X issue/vb/thread
            cp_async_ca_128(R[actual_row,absolute_col],
                            sR[row_in_cta,local_col],16 if col_valid else 0)
            # instruction_selection: cp.async.ca.shared.global 16 bytes with
            # source-size 16/0; extent: one residual issue/vb/thread
        if CLUSTER_N == 1:
            cp_async_ca_128(W[absolute_col],sW[row_in_cta,local_col],
                            16 if col_valid else 0)
            # instruction_selection: cp.async.ca.shared.global 16 bytes with
            # source-size 16/0; extent: one W issue/vb/thread, outside row guard
    cp_async_commit()
    # instruction_selection: cp.async.commit_group; extent: one/thread
    cp_async_wait(0)
    # instruction_selection: cp.async.wait_group 0; extent: one/thread

    x_bits = reg_fragment(INPUT_DTYPE,8*VEC_BLOCKS)
    r_bits = reg_fragment(INPUT_DTYPE,8*VEC_BLOCKS)
    for vb in static_unroll(VEC_BLOCKS):
        local_col = (lane_in_row + vb*TPR)*8
        x_bits[8*vb:8*vb+8] = load_shared_128(
            sX[row_in_cta,local_col])
        # instruction_selection: ld.shared.v4.b32; one 128-bit X issue/vb/thread
        r_bits[8*vb:8*vb+8] = load_shared_128(
            sR[row_in_cta,local_col])
        # instruction_selection: ld.shared.v4.b32; one 128-bit R issue/vb/thread

    x_f32 = reg_fragment("f32",8*VEC_BLOCKS)
    r_f32 = reg_fragment("f32",8*VEC_BLOCKS)
    for value in static_unroll(8*VEC_BLOCKS):
        x_f32[value] = widen_input_scalar(x_bits[value],INPUT_DTYPE)
        # instruction_selection: cvt.f32.{f16,bf16}; extent: one value
        r_f32[value] = widen_input_scalar(r_bits[value],INPUT_DTYPE)
        # instruction_selection: cvt.f32.{f16,bf16}; extent: one value

    h_f32 = reg_fragment("f32",8*VEC_BLOCKS)
    h_sq = reg_fragment("f32",8*VEC_BLOCKS)
    h_bits = reg_fragment(INPUT_DTYPE,8*VEC_BLOCKS)
    for pair in static_unroll(4*VEC_BLOCKS):
        pair_view(h_f32,pair) = add_f32x2(
            pair_view(x_f32,pair),pair_view(r_f32,pair))
        # instruction_selection: add.f32x2; extent: one adjacent pair
        pair_view(h_sq,pair) = mul_f32x2(
            pair_view(h_f32,pair),pair_view(h_f32,pair))
        # instruction_selection: mul.f32x2; extent: one adjacent pair
        h_bits[2*pair:2*pair+2] = cvt_input_pair(
            h_f32[2*pair+1],h_f32[2*pair],INPUT_DTYPE)
        # instruction_selection: cvt.rn.{f16x2,bf16x2}.f32; one adjacent pair

    if CLUSTER_N == 1:
        for vb in static_unroll(VEC_BLOCKS):
            local_col = (lane_in_row + vb*TPR)*8
            store_shared_128(
                sH[row_in_cta,local_col],h_bits[8*vb:8*vb+8])
            # instruction_selection: st.shared.v4.b32; one 128-bit issue/vb/thread

    local_sum = 0.0
    for value in static_unroll(8*VEC_BLOCKS):
        local_sum = add_f32(local_sum,h_sq[value])
        # instruction_selection: add.f32; ordered extent one/value
    for delta in powers_of_two_below(min(TPR,32)):
        peer = shuffle_xor(local_sum,delta)
        # instruction_selection: shfl.sync.bfly.b32; one/subwarp stage
        local_sum = add_f32(local_sum,peer)
        # instruction_selection: add.f32; one/subwarp stage
    warp_sum = local_sum

    if WARPS_PER_ROW > 1 and CLUSTER_N == 1:
        if lane == 0:
            store_shared_b32(reduction[row_warp,warp_in_row],warp_sum)
            # instruction_selection: st.shared.b32; one/row warp
        cta_barrier()
        # instruction_selection: bar.sync 0; one reduction publication edge
        final = 0.0
        if lane < WARPS_PER_ROW:
            final = load_shared_b32(reduction[row_warp,lane])
            # instruction_selection: ld.shared.b32; one participating lane
        for delta in (1,2,4,8,16):
            peer = shuffle_xor(final,delta)
            # instruction_selection: shfl.sync.bfly.b32; one/full-warp stage
            final = add_f32(final,peer)
            # instruction_selection: add.f32; one/full-warp stage
        sum_sq = final
    elif CLUSTER_N > 1:
        if warp == 0:
            elected = elect_one()
            # instruction_selection: elect.sync; one invocation in warp 0,
            # followed by its predicate branch
            if elected:
                mbarrier_arrive_expect_tx(
                    mbar,ROWS*WARPS_PER_ROW*CLUSTER_N*4)
                # instruction_selection: mbarrier.arrive.expect_tx.shared.b64;
                # one elected issue/CTA with exact byte count
        if lane < CLUSTER_N:
            peer_value = map_shared_to_peer(
                reduction[row_warp,(warp_in_row,cluster_rank())],lane)
            # instruction_selection: mapa.shared::cluster.u32; one value map/lane
            peer_bar = map_shared_to_peer(mbar,lane)
            # instruction_selection: mapa.shared::cluster.u32; one barrier map/lane
            remote_store_complete_tx(warp_sum,peer_value,peer_bar)
            # instruction_selection:
            # st.async.shared::cluster.mbarrier::complete_tx::bytes.f32;
            # one partial from every row warp to each peer
        while not mbarrier_try_wait(mbar,phase=0,timeout=10000000):
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 plus
            # uniform retry branch; one completion loop/thread
            pass
        final = 0.0
        for i in static_unroll(ceil_div(WARPS_PER_ROW*CLUSTER_N,32)):
            idx = lane + i*32
            if idx < WARPS_PER_ROW*CLUSTER_N:
                partial = load_shared_b32(reduction[row_warp,idx])
                # instruction_selection: ld.shared.b32; one owned partial
                final = add_f32(final,partial)
                # instruction_selection: add.f32; one loaded partial
        for delta in (1,2,4,8,16):
            peer = shuffle_xor(final,delta)
            # instruction_selection: shfl.sync.bfly.b32; one/full-warp stage
            final = add_f32(final,peer)
            # instruction_selection: add.f32; one/full-warp stage
        sum_sq = final
    else:
        sum_sq = warp_sum

    if H is a power of two:
        shifted = fma_rn_f32(sum_sq,float32(1.0/H),runtime_eps)
        # instruction_selection: fma.rn.f32; exactly one/thread
    else:
        mean_sq = div_f32(sum_sq,float32(H))
        # instruction_selection: div.rn.f32; exactly one/thread
        shifted = add_f32(mean_sq,runtime_eps)
        # instruction_selection: add.f32; exactly one/thread after the divide
    rstd = rsqrt_fast(shifted)
    # instruction_selection: rsqrt.approx.ftz.f32; one/thread

    if BLOCK_SIZE == 16 or SCALE_FORMAT == "e4m3":
        global_scale_value = load_global_b32(global_scale[0])
        # instruction_selection: ld.global.b32; one scalar/thread in E4M3;
        # canonical UE8M0 eliminates it

    if CLUSTER_N > 1:
        cluster_arrive_relaxed()
        # instruction_selection: barrier.cluster.arrive.relaxed; one/CTA
        cluster_wait()
        # instruction_selection: barrier.cluster.wait; one/CTA
    else:
        cta_barrier()
        # instruction_selection: bar.sync 0; one post-reduction shared edge

    if row_valid:
        for vb in static_unroll(VEC_BLOCKS):
            local_col = (lane_in_row + vb*TPR)*8
            absolute_col = cluster_y*COLS + local_col
            if absolute_col < H:
                store_global_128(R[actual_row,absolute_col],
                                 h_bits[8*vb:8*vb+8])
                # instruction_selection: st.global.v4.b32 with column predicate;
                # one 128-bit residual issue/vb/thread

    if CLUSTER_N > 1:
        fence_cluster_acq_rel()
        # instruction_selection: fence.acq_rel.cluster; one/CTA thread
        cluster_arrive_relaxed()
        # instruction_selection: barrier.cluster.arrive.relaxed; one/CTA
        cluster_wait()
        # instruction_selection: barrier.cluster.wait; one/CTA

    # Phase 3 uses a static sequence only below the reviewed trip boundary;
    # otherwise cursor is a true while-loop scalar. Cluster1 instantiates one
    # lexical scale-block body; clustered global-reload paths instantiate the
    # exact same primitive sequence twice before the backedge.
    if row_valid:
        BODY_SF = 2 if CLUSTER_N > 1 else 1
        cursor = 0
        while cursor < SF_PER_THREAD:
            body_slot = each_static_replica(range(BODY_SF))
            # structure: compile-time lexical replication only; it emits no op
            sf_iter = cursor + body_slot
            sf_idx = lane_in_row + sf_iter*TPR
            if sf_idx < H//BLOCK_SIZE:
                block_start = sf_idx*BLOCK_SIZE

                if CLUSTER_N == 1:
                    # Shared source helper loads all H scalars, then all W scalars.
                    h0 = reg_fragment("f32",16)
                    w0 = reg_fragment("f32",16)
                    for value in static_unroll(16):
                        bits = load_shared_b16(sH[row_in_cta,block_start+value])
                        # instruction_selection: ld.shared.b16; one H value
                        h0[value] = widen_input_scalar(bits,INPUT_DTYPE)
                        # instruction_selection: cvt.f32.{f16,bf16}; one H value
                    for value in static_unroll(16):
                        bits = load_shared_b16(sW[row_in_cta,block_start+value])
                        # instruction_selection: ld.shared.b16; one W value
                        w0[value] = widen_input_scalar(bits,INPUT_DTYPE)
                        # instruction_selection: cvt.f32.{f16,bf16}; one W value
                    y0 = reg_fragment("f32",16)
                    y0[0] = mul_f32(mul_f32(h0[0],rstd),w0[0])
                    # instruction_selection: two mul.f32; first value
                    max0 = abs_f32(y0[0])
                    # instruction_selection: abs.f32; first value
                    for value in static_unroll_range(1,16):
                        y0[value] = mul_f32(mul_f32(h0[value],rstd),w0[value])
                        # instruction_selection: two mul.f32; one value
                        max0 = max_f32(max0,abs_f32(y0[value]))
                        # instruction_selection: abs.f32 then max.f32; ordered fold

                    if BLOCK_SIZE == 32:
                        # Source completes chunk0 before beginning chunk1.
                        h1 = reg_fragment("f32",16)
                        w1 = reg_fragment("f32",16)
                        for value in static_unroll(16):
                            bits = load_shared_b16(
                                sH[row_in_cta,block_start+16+value])
                            # instruction_selection: ld.shared.b16; one H1 value
                            h1[value] = widen_input_scalar(bits,INPUT_DTYPE)
                            # instruction_selection: one scalar widen
                        for value in static_unroll(16):
                            bits = load_shared_b16(
                                sW[row_in_cta,block_start+16+value])
                            # instruction_selection: ld.shared.b16; one W1 value
                            w1[value] = widen_input_scalar(bits,INPUT_DTYPE)
                            # instruction_selection: one scalar widen
                        y1 = reg_fragment("f32",16)
                        y1[0] = mul_f32(mul_f32(h1[0],rstd),w1[0])
                        # instruction_selection: two mul.f32; first chunk1 value
                        max1 = abs_f32(y1[0])
                        # instruction_selection: abs.f32; first chunk1 value
                        for value in static_unroll_range(1,16):
                            y1[value] = mul_f32(
                                mul_f32(h1[value],rstd),w1[value])
                            # instruction_selection: two mul.f32; one chunk1 value
                            max1 = max_f32(max1,abs_f32(y1[value]))
                            # instruction_selection: abs then ordered max; one/value
                        max_abs = max_f32(max0,max1)
                        # instruction_selection: max.f32; one cross-chunk issue
                    else:
                        max_abs = max0
                else:
                    # Cluster path issues every 128-bit H/W load before arithmetic.
                    h0 = reg_fragment("u32",8)
                    w0 = reg_fragment("u32",8)
                    h0[0:4] = load_global_128(R[actual_row,block_start])
                    h0[4:8] = load_global_128(R[actual_row,block_start+8])
                    w0[0:4] = load_global_128(W[block_start])
                    w0[4:8] = load_global_128(W[block_start+8])
                    # instruction_selection: four ld.global.v4.u32 for chunk0,
                    # ordered H0,H0,W0,W0
                    if BLOCK_SIZE == 32:
                        h1 = reg_fragment("u32",8)
                        w1 = reg_fragment("u32",8)
                        h1[0:4] = load_global_128(R[actual_row,block_start+16])
                        h1[4:8] = load_global_128(R[actual_row,block_start+24])
                        w1[0:4] = load_global_128(W[block_start+16])
                        w1[4:8] = load_global_128(W[block_start+24])
                        # instruction_selection: four more ld.global.v4.u32,
                        # ordered H1,H1,W1,W1 before any packed multiply

                    xw0 = reg_fragment("u32",8)
                    for pair in static_unroll(8):
                        xw0[pair] = mul_input2(h0[pair],w0[pair],INPUT_DTYPE)
                        # instruction_selection: mul.{f16x2,bf16x2}, no .rn;
                        # eight chunk0 issues
                    if BLOCK_SIZE == 32:
                        xw1 = reg_fragment("u32",8)
                        for pair in static_unroll(8):
                            xw1[pair] = mul_input2(h1[pair],w1[pair],INPUT_DTYPE)
                            # instruction_selection: eight chunk1 packed issues

                    abs0 = reg_fragment("u32",8)
                    for pair in static_unroll(8):
                        abs0[pair] = abs_input2(xw0[pair],INPUT_DTYPE)
                        # instruction_selection: and.b32 0x7fff7fff; eight issues
                    m0_01 = max_input2(abs0[0],abs0[1],INPUT_DTYPE)
                    m0_23 = max_input2(abs0[2],abs0[3],INPUT_DTYPE)
                    m0_45 = max_input2(abs0[4],abs0[5],INPUT_DTYPE)
                    m0_67 = max_input2(abs0[6],abs0[7],INPUT_DTYPE)
                    m0_03 = max_input2(m0_01,m0_23,INPUT_DTYPE)
                    m0_47 = max_input2(m0_45,m0_67,INPUT_DTYPE)
                    max_pair = max_input2(m0_03,m0_47,INPUT_DTYPE)
                    # instruction_selection: seven packed max issues for chunk0
                    if BLOCK_SIZE == 32:
                        abs1 = reg_fragment("u32",8)
                        for pair in static_unroll(8):
                            abs1[pair] = abs_input2(xw1[pair],INPUT_DTYPE)
                            # instruction_selection: eight chunk1 abs issues,
                            # all after the complete chunk0 max tree
                        m1_01 = max_input2(abs1[0],abs1[1],INPUT_DTYPE)
                        m1_23 = max_input2(abs1[2],abs1[3],INPUT_DTYPE)
                        m1_45 = max_input2(abs1[4],abs1[5],INPUT_DTYPE)
                        m1_67 = max_input2(abs1[6],abs1[7],INPUT_DTYPE)
                        m1_03 = max_input2(m1_01,m1_23,INPUT_DTYPE)
                        m1_47 = max_input2(m1_45,m1_67,INPUT_DTYPE)
                        max1_pair = max_input2(m1_03,m1_47,INPUT_DTYPE)
                        max_pair = max_input2(max_pair,max1_pair,INPUT_DTYPE)
                        # instruction_selection: eight packed max issues total
                        # after chunk1 abs, including one cross-chunk maximum

                    if INPUT_DTYPE == "f16":
                        max_lo16,max_hi16 = mov_b32_pair(max_pair)
                        # instruction_selection: mov.b32 pair; one/SF
                        max_lo = cvt_f32_f16(max_lo16)
                        max_hi = cvt_f32_f16(max_hi16)
                        # instruction_selection: cvt.f32.f16; two/SF
                    else:
                        max_lo32 = and_b32(max_pair,0xffff)
                        max_hi32 = shift_right_b32(max_pair,16)
                        max_lo32 = shift_left_b32(max_lo32,16)
                        max_hi32 = shift_left_b32(max_hi32,16)
                        # instruction_selection: and, shift-right, two
                        # shift-left issues for the BF16 maximum pair
                        max_lo = mov_b32_as_f32(max_lo32)
                        max_hi = mov_b32_as_f32(max_hi32)
                        # instruction_selection: mov.b32 to f32; two/SF
                    max_xw = max_f32(max_lo,max_hi)
                    # instruction_selection: max.f32; one horizontal issue

                    y0 = reg_fragment("f32",16)
                    for pair in static_unroll(8):
                        if INPUT_DTYPE == "f16":
                            lo16,hi16 = mov_b32_pair(xw0[pair])
                            lo = cvt_f32_f16(lo16)
                            hi = cvt_f32_f16(hi16)
                            # instruction_selection: mov pair + two cvt; one pair
                        else:
                            lo32 = shift_left_b32(and_b32(xw0[pair],0xffff),16)
                            hi32 = shift_left_b32(shift_right_b32(xw0[pair],16),16)
                            lo = mov_b32_as_f32(lo32)
                            hi = mov_b32_as_f32(hi32)
                            # instruction_selection: BF16 unpack/mov chain; one pair
                        y0[2*pair] = mul_f32(lo,rstd)
                        y0[2*pair+1] = mul_f32(hi,rstd)
                        # instruction_selection: two mul.f32; one pair
                    if BLOCK_SIZE == 32:
                        y1 = reg_fragment("f32",16)
                        for pair in static_unroll(8):
                            if INPUT_DTYPE == "f16":
                                lo16,hi16 = mov_b32_pair(xw1[pair])
                                lo = cvt_f32_f16(lo16)
                                hi = cvt_f32_f16(hi16)
                                # instruction_selection: mov pair + two cvt
                            else:
                                lo32 = shift_left_b32(and_b32(xw1[pair],0xffff),16)
                                hi32 = shift_left_b32(
                                    shift_right_b32(xw1[pair],16),16)
                                lo = mov_b32_as_f32(lo32)
                                hi = mov_b32_as_f32(hi32)
                                # instruction_selection: BF16 unpack/mov chain
                            y1[2*pair] = mul_f32(lo,rstd)
                            y1[2*pair+1] = mul_f32(hi,rstd)
                            # instruction_selection: two mul.f32; one pair
                    max_abs = mul_f32(max_xw,rstd)
                    # instruction_selection: mul.f32; one scale-block maximum

                if BLOCK_SIZE == 16 or SCALE_FORMAT == "e4m3":
                    scale_f = mul_f32(global_scale_value,max_abs)
                    # instruction_selection: mul.f32; one scalar
                    scale_f = mul_f32(scale_f,fp4_max_rcp)
                    # instruction_selection: mul.f32; one scalar
                    scale_f = min_f32(scale_f,448.0)
                    # instruction_selection: min.f32; one scalar
                    scale_pair16 = cvt_e4m3_pair(0.0,scale_f)
                    # instruction_selection: cvt.rn.satfinite.e4m3x2.f32;
                    # zero is the high mate and scale the low mate, one/SF
                    scale_word = cvt_u32_u16(scale_pair16)
                    # instruction_selection: cvt.u32.u16; one/SF
                    scale_byte = scale_word
                    # lowering: the source `& 0xff` view is DCE; st.global.b8
                    # consumes the low byte with no standalone and.b32
                    decode_pair16 = cvt_u16_u32(scale_word)
                    # instruction_selection: cvt.u16.u32; one/SF
                    decoded_h2 = cvt_f16x2_e4m3x2(decode_pair16)
                    # instruction_selection: cvt.rn.f16x2.e4m3x2; one/SF
                    decoded_lo16,_ = mov_b32_pair(decoded_h2)
                    # instruction_selection: mov.b32 {b16,b16}; one/SF
                    decoded = cvt_f32_f16(decoded_lo16)
                    # instruction_selection: cvt.f32.f16; one/SF
                    reciprocal = rcp_fast(decoded)
                    # instruction_selection: rcp.approx.ftz.f32; one/SF
                    decoded_zero = setp(decoded == 0.0)
                    # instruction_selection: setp.eq.f32; one/SF
                    decoded_rcp = select(decoded_zero,0.0,reciprocal)
                    # instruction_selection: selp.f32; one/SF
                    inv_scale = mul_f32(decoded_rcp,global_scale_value)
                    # instruction_selection: mul.f32; one scale
                else:
                    scale_f = mul_f32(max_abs,fp4_max_rcp)
                    # instruction_selection: mul.f32; one scalar
                    nonpositive = setp(scale_f <= 0.0)
                    # instruction_selection: setp.le.f32; one/SF
                    log_scale = lg2_fast(scale_f)
                    # instruction_selection: lg2.approx.f32; one scale
                    exponent = cvt_s32_f32_rpi(log_scale)
                    # instruction_selection: cvt.rpi.s32.f32; one/SF
                    biased = add_s32(exponent,127)
                    # instruction_selection: add.s32; one/SF
                    negative = setp(biased < 0)
                    overflow = setp(biased > 255)
                    # instruction_selection: setp.lt.s32 and setp.gt.s32
                    biased = select(negative,0,biased)
                    biased = select(overflow,255,biased)
                    scale_word = select(nonpositive,0,biased)
                    # instruction_selection: three ordered selp.s32 issues
                    scale_byte = scale_word
                    # lowering: low-byte view is DCE before st.global.b8
                    scale_zero = setp(scale_word == 0)
                    # instruction_selection: setp.eq.u32; one/SF
                    negative_exponent = sub_s32(127,scale_word)
                    # instruction_selection: sub.s32; one/SF
                    inv_exponent = cvt_f32_s32_rn(negative_exponent)
                    # instruction_selection: cvt.rn.f32.s32; one scale
                    inv_candidate = ex2_fast(inv_exponent)
                    # instruction_selection: ex2.approx.f32; one/SF
                    inv_scale = select(scale_zero,0.0,inv_candidate)
                    # instruction_selection: selp.f32; one/SF

                store_global_b8(S_u8[primary_offset],scale_byte)
                # instruction_selection: st.global.b8; one primary scale byte;
                # no preceding `and.b32 0xff` survives PTX
                if BOTH_SF:
                    store_global_b8(S_linear_u8[actual_row,sf_idx],scale_byte)
                    # instruction_selection: st.global.b8; one linear scale byte

                for value in static_unroll(16):
                    q0[value] = mul_f32(y0[value],inv_scale)
                    # instruction_selection: mul.f32; one value
                pair0 = reg_fragment("u8",8)
                for pair in static_unroll(8):
                    pair0[pair] = cvt_e2m1_pair(q0[2*pair+1],q0[2*pair])
                    # instruction_selection:
                    # cvt.rn.satfinite.e2m1x2.f32; eight/SF, PTX high then low
                lo32 = mov_b32_bytes(pair0[0],pair0[1],pair0[2],pair0[3])
                hi32 = mov_b32_bytes(pair0[4],pair0[5],pair0[6],pair0[7])
                # instruction_selection: mov.b32 {b8,b8,b8,b8}; two/SF
                lo64 = cvt_u64_u32(lo32)
                hi64 = shift_left_b64(cvt_u64_u32(hi32),32)
                # instruction_selection: two cvt.u64.u32 and one shl.b64
                packed0 = or_b64(hi64,lo64)
                # instruction_selection: or.b64; one/SF
                if BLOCK_SIZE == 32:
                    for value in static_unroll(16):
                        q1[value] = mul_f32(y1[value],inv_scale)
                        # instruction_selection: mul.f32; one value
                    pair1 = reg_fragment("u8",8)
                    for pair in static_unroll(8):
                        pair1[pair] = cvt_e2m1_pair(q1[2*pair+1],q1[2*pair])
                        # instruction_selection: eight E2M1 pair conversions
                    lo32 = mov_b32_bytes(pair1[0],pair1[1],pair1[2],pair1[3])
                    hi32 = mov_b32_bytes(pair1[4],pair1[5],pair1[6],pair1[7])
                    lo64 = cvt_u64_u32(lo32)
                    hi64 = shift_left_b64(cvt_u64_u32(hi32),32)
                    packed1 = or_b64(hi64,lo64)
                    # instruction_selection: two byte-pack moves, two widens,
                    # one shift, one or
                # Source finishes every block32 conversion/pack before either
                # payload store; block16 reaches this store immediately.
                store_global_u64(Y_u8[actual_row,block_start//2],packed0)
                # instruction_selection: st.global.u64; first FP4 payload
                if BLOCK_SIZE == 32:
                    store_global_u64(Y_u8[actual_row,block_start//2+8],packed1)
                    # instruction_selection: st.global.u64; second payload

                if YNORM:
                    for group in static_unroll(4):
                        if INPUT_DTYPE == "f16":
                            h0 = cvt_f16_f32(y0[group*4])
                            h1 = cvt_f16_f32(y0[group*4+1])
                            # instruction_selection: cvt.rn.f16.f32;
                            # two scalar conversions for the low pair
                            norm_lo32 = mov_b32_from_b16(h0,h1)
                            # instruction_selection: mov.b32 {b16,b16}; one
                            h2 = cvt_f16_f32(y0[group*4+2])
                            h3 = cvt_f16_f32(y0[group*4+3])
                            # instruction_selection: cvt.rn.f16.f32;
                            # two scalar conversions for the high pair
                            norm_hi32 = mov_b32_from_b16(h2,h3)
                            # instruction_selection: mov.b32 {b16,b16}; one
                        else:
                            b0 = cvt_f32_bf16(y0[group*4])
                            b1 = cvt_f32_bf16(y0[group*4+1])
                            # instruction_selection: cvt.rn.bf16.f32;
                            # two scalar conversions for the low pair
                            norm_lo32 = mov_b32_from_b16(b0,b1)
                            # instruction_selection: mov.b32 {b16,b16}; one
                            b2 = cvt_f32_bf16(y0[group*4+2])
                            b3 = cvt_f32_bf16(y0[group*4+3])
                            # instruction_selection: cvt.rn.bf16.f32;
                            # two scalar conversions for the high pair
                            norm_hi32 = mov_b32_from_b16(b2,b3)
                            # instruction_selection: mov.b32 {b16,b16}; one
                        norm_lo64 = cvt_u64_u32(norm_lo32)
                        norm_hi64 = shift_left_b64(cvt_u64_u32(norm_hi32),32)
                        ynorm_word = or_b64(norm_hi64,norm_lo64)
                        # instruction_selection: two widens, shift, or
                        store_global_u64(
                            YNorm[actual_row,block_start+group*4],ynorm_word)
                        # instruction_selection: st.global.u64; four input-dtype values
                    if BLOCK_SIZE == 32:
                        for group in static_unroll(4):
                            if INPUT_DTYPE == "f16":
                                h0 = cvt_f16_f32(y1[group*4])
                                h1 = cvt_f16_f32(y1[group*4+1])
                                # instruction_selection: two scalar FP16 cvt
                                norm_lo32 = mov_b32_from_b16(h0,h1)
                                # instruction_selection: one mov.b32 pack
                                h2 = cvt_f16_f32(y1[group*4+2])
                                h3 = cvt_f16_f32(y1[group*4+3])
                                # instruction_selection: two scalar FP16 cvt
                                norm_hi32 = mov_b32_from_b16(h2,h3)
                                # instruction_selection: one mov.b32 pack
                            else:
                                b0 = cvt_f32_bf16(y1[group*4])
                                b1 = cvt_f32_bf16(y1[group*4+1])
                                # instruction_selection: two scalar BF16 cvt
                                norm_lo32 = mov_b32_from_b16(b0,b1)
                                # instruction_selection: one mov.b32 pack
                                b2 = cvt_f32_bf16(y1[group*4+2])
                                b3 = cvt_f32_bf16(y1[group*4+3])
                                # instruction_selection: two scalar BF16 cvt
                                norm_hi32 = mov_b32_from_b16(b2,b3)
                                # instruction_selection: one mov.b32 pack
                            norm_lo64 = cvt_u64_u32(norm_lo32)
                            norm_hi64 = shift_left_b64(
                                cvt_u64_u32(norm_hi32),32)
                            ynorm_word = or_b64(norm_hi64,norm_lo64)
                            # instruction_selection: same conversion/pack family
                            store_global_u64(
                                YNorm[actual_row,block_start+16+group*4],
                                ynorm_word)
                            # instruction_selection: st.global.u64; four values
            cursor += BODY_SF
            # instruction_selection: scalar loop-index add in rolled variants;
            # immediate 1 for cluster1, 2 for clustered paths; absent when the
            # reviewed trip threshold selects complete static expansion

    if ENABLE_PDL:
        pdl_signal()
        # instruction_selection: griddepcontrol.launch_dependents; one CTA issue
```

The pseudocode writes a `while` for the general phase-3 form. The implementation
must statically emit its body below the reviewed cluster1 threshold, retain the
single-SF cluster1 while exactly at/above it, and retain the two-replica
clustered body with a guard on replica one. `primary_offset` is row-major unless
`SWIZZLED or BOTH_SF`, in which case it is
`(row//128)*(ceil_div(SF_COLS,4)*512)+(sf//4)*512+(row%32)*16+((row%128)//32)*4+(sf%4)`.

## Storage ownership and lifetimes

| storage | owner | lifetime |
| --- | --- | --- |
| X/W/global-scale | caller | read-only for the whole launch |
| residual | caller | initial add input, then source-order in-place output; clustered phase 3 reloads it after fence/barrier |
| Y/S/S-linear/YNorm | caller | independently mutable outputs; optional buffers remain untouched when disabled |
| sX/sR | each CTA | async publication through phase-1 shared load |
| sW/sH | each cluster1 CTA | async W/narrowed H publication through complete phase-3 scale loop |
| reduction/mbarrier | CTA/cluster | row partial publication through rstd completion only |
| H FP32 and H-square fragments | one thread | add through row reduction; H narrows before publication |
| packed H/W or scalar FP32 scale block | one thread | one static or rolled phase-3 block |

## Module and verification contract

- Registry name `flashinfer_add_rmsnorm_fp4quant`, category `flashinfer`, compute
  capability 10. PrimFunc returns `None` and exposes the fixed ten-argument ABI.
- `CONFIGS` contains 1222 positive upstream device/public cases and 23 structural
  guards. Three source-wrapper-only negative/trace cases are documented but are
  not represented as device executions.
- `BENCH_CONFIGS` is exactly the ten active unified source samples. Every row in
  one valid reference-enabled `bench_suite` run must satisfy
  `flashinfer_cutedsl_time / tirx_time > 0.99`; no other timer or API is a PASS
  criterion.
- Every specialization is rejected if low-level IR contains any tile primitive,
  `tirx.tile.*`, `TilePrimitiveCall`, or function call.

## Instruction selection is a lowering consequence

The port preserves the three-input async group in cluster1 and two-input group in
clusters, input-dtype H publication, ordered FP32 sum, exact row/cluster
reduction, global-scale load placement, residual-store placement, cluster
visibility fence, source's redundant full-row clustered quantization, Add-specific
rolled-loop thresholds, scalar shared phase-3 loads versus packed global loads,
chunk ordering, E4M3/UE8M0 rounding chains, byte stores without synthetic masks,
packed E2M1 ordering, dual scale writes, YNorm conversion/store width, and PDL
boundaries. Plain TIRx/PTX may express these reviewed instructions; it may not use
tile primitives, change arithmetic precision/order, deduplicate cluster stores,
reload a different value, move the residual store/fence, adopt RMSNormFP4's loop
threshold, or fold optional outputs into extra launches.
