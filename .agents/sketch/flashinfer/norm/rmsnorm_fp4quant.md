<!--
Copyright (c) 2025 by FlashInfer team.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: Apache-2.0

This execution sketch documents a TIRx port of FlashInfer's CuTe-DSL
RMSNormFP4QuantKernel. See LICENSE, NOTICE, and licenses/ for applicable terms.
-->

# FlashInfer RMSNorm FP4 quantization SM100: execution sketch

This non-executable sketch freezes the implementation shape of FlashInfer's
CuTe-DSL `RMSNormFP4QuantKernel` for
[`tirx_kernels/flashinfer/norm/rmsnorm_fp4quant.py`](../../../../tirx_kernels/flashinfer/norm/rmsnorm_fp4quant.py).
It records the source execution skeleton, not a mathematically equivalent
replacement. The target is implemented only after independent source/PTX
review, and this file is permanently frozen after its first reviewer PASS.

The source is fixed at FlashInfer commit
`f2e04400e330fb2debe0bf8730d9424a1d37927f`:

- `flashinfer/cute_dsl/rmsnorm_fp4quant.py`, SHA256
  `ec32fae9254adb9b888c0affd99822c89806a5881de04db0b0a755d81f6f90a3`;
- `flashinfer/cute_dsl/fp4_common.py`, SHA256
  `ff490c11853603634fd19c4558021b03db08b409dc54d24146f1bdc66a6b784f`;
- public alias and wrapper in `flashinfer/norm/__init__.py`, SHA256
  `226a88f5fb14e78e06e1be79020edcae01bfa9e53e677bd485373ea4d51cffcb`.

Writer-owned fresh line-info exports are under
`.porting/flashinfer_rmsnorm_fp4quant/source_exports/`; their opcode and loop
audit is `.porting/flashinfer_rmsnorm_fp4quant/source_ptx_audit.md`. They cover
FP16/BF16, NVFP4/MXFP4, linear/swizzled scales, PDL, every launch threshold,
the 7-to-8 scale-loop boundary, and cluster sizes 2/16. The reviewer must
independently export through the normal public CuTeDSL build path.

The supported family is compact FP16/BF16 X and W, packed E2M1 Y bytes,
block size 16 or 32, E4M3 or UE8M0 scale bytes, linear or 128x4 swizzled scale
layout, 2-D or host-flattened 3-D input, PDL off/on, and clusters 1/2/4/8/16.
There is no strided device specialization. Tile primitives, TMA, W shared
staging, a second `cp.async` stage, and post-reduction X shared reload are out
of scope.

## Source dispatch and resource formulae

These are source formulae, not tuning choices. `MAX_SMEM=232448` is the B200
opt-in value used by the pinned export and validation environment.

```python
COPY_BITS = 128
ELEM_BYTES = 2
VEC = 8

def threads_per_row(h_per_cta):
    if h_per_cta <= 64: return 8
    if h_per_cta <= 128: return 16
    if h_per_cta <= 3072: return 32
    if h_per_cta <= 6144: return 64
    if h_per_cta <= 16384: return 128
    return 256

def num_threads(h_per_cta):
    return 128 if h_per_cta <= 16384 else 256

def estimate_smem(H, cluster_n):
    hcta = H // cluster_n
    tpr = threads_per_row(hcta)
    threads = num_threads(hcta)
    rows = threads // tpr
    warps_per_row = max(tpr // 32, 1)
    vec_blocks = max(1, ceil_div(hcta // VEC, tpr))
    cols = VEC * vec_blocks * tpr
    tile_bytes = rows * cols * ELEM_BYTES
    if cluster_n == 1:
        # Deliberately conservative selection estimate: source counts sX+sW.
        return 2 * tile_bytes + rows * warps_per_row * 4
    return tile_bytes + rows * warps_per_row * cluster_n * 4 + 8

def source_config(H):
    cluster_n = next(
        (c for c in (1,2,4,8,16)
         if H % c == 0 and estimate_smem(H,c) <= MAX_SMEM),
        16,
    )
    hcta = H // cluster_n
    tpr = threads_per_row(hcta)
    threads = num_threads(hcta)
    rows = threads // tpr
    warps_per_row = max(tpr // 32, 1)
    vec_blocks = max(1, ceil_div(hcta // VEC, tpr))
    cols = VEC * vec_blocks * tpr
    tile_bytes = rows * cols * ELEM_BYTES
    reduction_bytes = rows * warps_per_row * 4
    if cluster_n > 1:
        reduction_bytes *= cluster_n
    smem_bytes = tile_bytes + reduction_bytes + (8 if cluster_n > 1 else 0)
    num_sf = H // BLOCK_SIZE
    sf_per_thread = ceil_div(num_sf, tpr)
    return cluster_n,hcta,tpr,threads,rows,warps_per_row,vec_blocks,cols, \
           smem_bytes,num_sf,sf_per_thread
```

The actual shared allocation always contains one X tile, not the conservative
two-tile cluster-selection estimate. Launch is
`grid=(ceil_div(M,ROWS),CLUSTER_N,1)`, `block=(THREADS,1,1)`, optional cluster
`(1,CLUSTER_N,1)`, with exact dynamic shared size above. The PTX has
`.reqntid` and `.extern .shared .align 1024`, and no `.maxnreg` or launch-bound
directive.

Runtime device ABI is `(X,W,Y,S,global_scale,M:i32,eps:f32)`. X/W are compact
input dtype; Y is compact packed bytes `[M,H/2]`; S is `[M,H/BLOCK]` linear or
the padded swizzled backing allocation; global scale is device FP32 `[1]`.
The public 3-D case flattens its leading two axes into runtime M before launch.

## Source-proven loop shape

| phase | compile-time extent | source/PTX structure |
| --- | ---: | --- |
| X `cp.async` and shared load | `VEC_BLOCKS` | always statically expanded, including large H |
| row fragment square/sum | `8*VEC_BLOCKS` | statically expanded |
| phase-3 SF loop | `SF_PER_THREAD <= 7` | completely expanded |
| phase-3 SF loop | `SF_PER_THREAD >= 8` | backward `while` loop, two SF blocks per body, cursor `+=2`, second block guarded |
| block16 payload | 16 values | one source-exact group, no inner runtime loop |
| block32 payload | 2 x 16 values | both groups loaded before either group is consumed |

Fresh adjacent evidence is H14336/block16 (`SF_PER_THREAD=7`, seven static
stores, no backedge) and H16384/block16 (`SF_PER_THREAD=8`, two static blocks
inside a loop whose backedge maps to source line 498). This is a loop-trip
boundary, not a hard-coded H boundary.

## CTA roles, publication edges, and storage

| owner | source-owned work | publication/reuse edge |
| --- | --- | --- |
| every CTA thread | optional PDL wait; phase-1 X issue, square contribution; phase-3 assigned SF blocks | one row group per contiguous TPR lanes |
| each TPR lane group | one logical row, one CTA column slice during reduction | replicated rstd after row/cluster reduction |
| lane 0 of each row warp, non-cluster multiwarp | publish one warp partial to CTA shared | CTA barrier, then all row warps redundantly finish the reduction |
| lanes `<CLUSTER_N`, clustered CTA | remotely publish each warp partial to every peer CTA | two `mapa`, remote complete-tx store, mbarrier wait |
| every cluster CTA in phase 3 | process the complete row's scale-block space | intentionally redundant full-H Y/S stores; no `cluster_y` offset |

Raw shared storage is aligned to 1024 bytes:

1. `sX`: input dtype `[ROWS,COLS]`, row-major/order `(1,0)`, alignment 16;
2. reduction: FP32 `[ROWS,WARPS_PER_ROW]` for one CTA, or
   `[ROWS,(WARPS_PER_ROW,CLUSTER_N)]` for clustered execution, alignment 4;
3. one 8-byte mbarrier only for `CLUSTER_N>1`.

X shared lifetime starts at async issue and ends after the single phase-2
shared load. The FP32 X fragment lives only through square/reduction. Phase 3
reloads X and W directly from global into packed register words. The reduction
buffer is reused only by the row-reduction protocol; clustered phase 3 begins
after the second cluster barrier. Scale is stored before its Y payload for each
SF block. Inputs/global scale are read-only; Y/S are the two mutable outputs.

## Primitive vocabulary

Structural vocabulary does not move or compute values:

```python
specialize(...)  launch(...)  raw_shared(...)  view(...)  reg_fragment(...)
address(...)     swizzled_scale_offset(...)     logical_pair_view(...)
```

All copy, compute, conversion, and synchronization work remains primitive:

```python
pdl_wait()  pdl_signal()
cp_async_ca_128(src,dst,source_bytes)  cp_async_commit()  cp_async_wait(0)
load_shared_128(src)  load_global_128(src)  store_global_b8(dst,v)
store_global_u64(dst,v)
mul_input2(a,b,dtype)  abs_input2(v,dtype)  max_input2(a,b,dtype)
widen_input_scalar(v,dtype)  mul_f32(a,b)  mul_f32x2(a,b)
add_f32(a,b)  div_f32(a,b)  max_f32(a,b)
rsqrt_fast(v)  rcp_fast(v)  min_f32(a,b)  lg2_fast(v)  ex2_fast(v)
shuffle_xor(v,delta,width)  copy_r2s(v,dst)  copy_s2r(src)
cvt_e4m3_pair(hi,lo)  cvt_e2m1_pair(hi,lo)  cvt_f16x2_e4m3x2(v)
cvt_f32_f16(v)  cvt_s32_f32_rpi(v)  cvt_f32_s32_rn(v)
and_b32(a,mask)  shift_left_b32(v,n)  shift_right_b32(v,n)
mov_b32_pair(lo,hi)  mov_b32_bytes(b0,b1,b2,b3)
setp(...)  select(...)  add_s32(...)  sub_s32(...)  cvt_u64_u32(...)
shift_left_b64(v,n)  or_b64(a,b)
cta_barrier()  cluster_arrive_relaxed()  cluster_wait()
mbarrier_init(...)  mbarrier_init_fence()  mbarrier_arrive_expect_tx(...)
map_shared_to_peer(...)  remote_store_complete_tx(...)  mbarrier_try_wait(...)
```

There is no compound RMSNorm, reduction, scale, FP4 quantize, tiled copy, or
tile compute primitive.

## Complete execution sketch

```python
@specialize(
    INPUT_DTYPE=("f16","bf16"),
    H="compile-time positive multiple of BLOCK_SIZE, at least 64",
    BLOCK_SIZE=(16,32), SCALE_FORMAT=("e4m3","ue8m0"),
    SWIZZLED=(False,True), ENABLE_PDL=(False,True), TARGET="sm_100a",
)
@launch(
    grid=(ceil_div(runtime_M,ROWS),CLUSTER_N,1),
    block=(THREADS,1,1),
    cluster=((1,CLUSTER_N,1) if CLUSTER_N > 1 else None),
    dynamic_smem_bytes=SMEM_BYTES,
    use_programmatic_dependent_launch=ENABLE_PDL,
)
def flashinfer_rmsnorm_fp4quant(
    X, W, Y_u8, S_u8, global_scale, runtime_M:i32, runtime_eps:f32,
):
    tid = thread_id_x()
    block_x = block_id_x()
    cluster_y = block_id_y() if CLUSTER_N > 1 else 0
    lane_in_row = tid % TPR
    row_in_cta = tid // TPR
    warp = tid // 32
    lane = tid % 32
    WARPS_PER_ROW = max(TPR // 32,1)
    row_warp = warp // WARPS_PER_ROW
    warp_in_row = warp % WARPS_PER_ROW

    if ENABLE_PDL:
        pdl_wait()
        # instruction_selection: griddepcontrol.wait; extent: one CTA issue

    fp4_max_rcp = rcp_fast(6.0)
    # instruction_selection: rcp.approx.ftz.f32; extent: one scalar/thread

    smem = raw_shared("u8",SMEM_BYTES,alignment=1024)
    sX = view(smem,INPUT_DTYPE,(ROWS,COLS),row_major=True,
              byte_offset=0,alignment=16)
    reduce_offset = ROWS * COLS * 2
    if CLUSTER_N == 1:
        reduction = view(smem,"f32",(ROWS,WARPS_PER_ROW),
                         byte_offset=reduce_offset,alignment=4)
    else:
        reduction = view(smem,"f32",(ROWS,(WARPS_PER_ROW,CLUSTER_N)),
                         byte_offset=reduce_offset,alignment=4)
        mbar = view(smem,"mbarrier",(1,),
                    byte_offset=reduce_offset + ROWS*WARPS_PER_ROW*CLUSTER_N*4,
                    alignment=8)

    if CLUSTER_N > 1:
        if tid == 0:
            mbarrier_init(mbar,arrivals=1)
            # instruction_selection: mbarrier.init.shared.b64; one/CTA
        mbarrier_init_fence()
        # instruction_selection: fence.mbarrier_init.release.cluster; one/CTA
        cluster_arrive_relaxed()
        # instruction_selection: barrier.cluster.arrive.relaxed; one/CTA
        cluster_wait()
        # instruction_selection: barrier.cluster.wait; one/CTA

    # TV layout shape ((TPR,ROWS),(8,VEC_BLOCKS)); stride
    # ((8*ROWS,1),(ROWS,ROWS*8*TPR)). Each lane owns eight adjacent values
    # at every vb, and cluster_y selects the reduction column slice.
    actual_row = block_x * ROWS + row_in_cta
    row_valid = actual_row < runtime_M
    x_frag = reg_fragment(INPUT_DTYPE,8*VEC_BLOCKS)

    # --------------------------------------------------------------
    # Phase 1: one async group, X global -> shared.
    # --------------------------------------------------------------
    for vb in static_unroll(VEC_BLOCKS):
        local_col = (lane_in_row + vb*TPR) * 8
        absolute_col = cluster_y * COLS + local_col
        col_valid = absolute_col < H
        if row_valid:
            source_bytes = 16 if col_valid else 0
            cp_async_ca_128(
                X + actual_row*H + absolute_col,
                sX[row_in_cta,local_col], source_bytes)
            # instruction_selection: cp.async.ca.shared.global, immediate 16
            # bytes plus source-size 16/0; extent: one issue/vb/thread
    cp_async_commit()
    # instruction_selection: cp.async.commit_group; one/thread
    cp_async_wait(0)
    # instruction_selection: cp.async.wait_group 0; one/thread

    for vb in static_unroll(VEC_BLOCKS):
        local_col = (lane_in_row + vb*TPR) * 8
        x_frag[vb] = load_shared_128(sX[row_in_cta,local_col])
        # instruction_selection: ld.shared.v4.b32; one 128-bit issue/vb/thread

    # Source widens all physical values, squares adjacent pairs with packed
    # FP32 instructions, then performs an ordered scalar ADD over the fragment.
    x_f32 = reg_fragment("f32",8*VEC_BLOCKS)
    x_sq = reg_fragment("f32",8*VEC_BLOCKS)
    for flat in static_unroll(8*VEC_BLOCKS):
        x_f32[flat] = widen_input_scalar(x_frag,flat,INPUT_DTYPE)
        # instruction_selection: cvt.f32.{f16,bf16}; one scalar/value
    for pair in static_unroll(4*VEC_BLOCKS):
        x_sq[2*pair:2*pair+2] = mul_f32x2(
            x_f32[2*pair:2*pair+2],x_f32[2*pair:2*pair+2])
        # instruction_selection: mul.f32x2; exactly 4*VEC_BLOCKS issues
    local_sum = 0.0
    for flat in static_unroll(8*VEC_BLOCKS):
        local_sum = add_f32(local_sum,x_sq[flat])
        # instruction_selection: add.f32; ordered extent one/value

    # --------------------------------------------------------------
    # Expanded source row_reduce protocol.
    # --------------------------------------------------------------
    WARP_WIDTH = min(TPR,32)
    for delta in powers_of_two_below(WARP_WIDTH):
        peer = shuffle_xor(local_sum,delta,width=32)
        # instruction_selection: shfl.sync.bfly.b32 full mask/clamp31;
        # one issue per explicit subgroup stage
        local_sum = add_f32(local_sum,peer)
        # instruction_selection: add.f32; one/stage
    warp_sum = local_sum

    if WARPS_PER_ROW > 1 and CLUSTER_N == 1:
        if lane == 0:
            copy_r2s(warp_sum,reduction[row_warp,warp_in_row])
            # instruction_selection: st.shared.b32; one/row warp
        cta_barrier()
        # instruction_selection: bar.sync; one publication edge/CTA
        final = 0.0
        if lane < WARPS_PER_ROW:
            final = copy_s2r(reduction[row_warp,lane])
            # instruction_selection: ld.shared.b32; one participating lane
        for delta in (1,2,4,8,16):
            peer = shuffle_xor(final,delta,width=32)
            # instruction_selection: shfl.sync.bfly.b32; one/stage
            final = add_f32(final,peer)
            # instruction_selection: add.f32; one/stage
        sum_sq = final

    elif CLUSTER_N > 1:
        if warp == 0 and elect_one():
            # instruction_selection: elect.sync; one election/CTA
            expected = ROWS*WARPS_PER_ROW*CLUSTER_N*4
            mbarrier_arrive_expect_tx(mbar,expected)
            # instruction_selection: mbarrier.arrive.expect_tx.shared.b64;
            # one elected issue/CTA with exact expected byte count
        if lane < CLUSTER_N:
            peer_value = map_shared_to_peer(
                reduction[row_warp,(warp_in_row,cluster_rank())],lane)
            peer_bar = map_shared_to_peer(mbar,lane)
            # instruction_selection: mapa.shared::cluster.u32; two maps/lane
            remote_store_complete_tx(warp_sum,peer_value,peer_bar)
            # instruction_selection:
            # st.async.shared::cluster.mbarrier::complete_tx::bytes.f32;
            # one partial from every row warp to every peer CTA
        done = False
        while not done:
            done = mbarrier_try_wait(mbar,phase=0,timeout=10000000)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 plus
            # uniform retry branch; one completion loop/thread
        total = WARPS_PER_ROW*CLUSTER_N
        final = 0.0
        for i in static_unroll(ceil_div(total,32)):
            partial = lane + i*32
            if partial < total:
                v = copy_s2r(reduction[row_warp,partial])
                # instruction_selection: ld.shared.b32; one owned partial
                final = add_f32(final,v)
                # instruction_selection: add.f32; one loaded partial
        for delta in (1,2,4,8,16):
            peer = shuffle_xor(final,delta,width=32)
            # instruction_selection: shfl.sync.bfly.b32; one/stage
            final = add_f32(final,peer)
            # instruction_selection: add.f32; one/stage
        sum_sq = final
    else:
        sum_sq = warp_sum

    mean_sq = div_f32(sum_sq,float32(H))
    # instruction_selection: div.rn.f32, or source-compiler folded reciprocal
    # multiply for exactly representable power-of-two H; one/thread
    shifted = add_f32(mean_sq,runtime_eps)
    # instruction_selection: add.f32, possibly fused with a folded reciprocal
    # as fma.rn.f32 for a power-of-two H; one/thread
    rstd = rsqrt_fast(shifted)
    # instruction_selection: rsqrt.approx.ftz.f32; one/thread

    # The source expression reads global_scale before this edge. In canonical
    # UE8M0 PTX the unused load is DCE; E4M3 branches retain ld.global.b32.
    if BLOCK_SIZE == 16 or SCALE_FORMAT == "e4m3":
        global_scale_value = load_global_f32(global_scale[0])
        # instruction_selection: ld.global.b32; one scalar/thread in E4M3
        # specializations, absent after DCE in canonical UE8M0

    if CLUSTER_N > 1:
        cluster_arrive_relaxed()
        # instruction_selection: barrier.cluster.arrive.relaxed; one/CTA
        cluster_wait()
        # instruction_selection: barrier.cluster.wait; one/CTA
    else:
        cta_barrier()
        # instruction_selection: bar.sync; one post-reduction edge/CTA

    # --------------------------------------------------------------
    # Phase 3: all cluster CTAs intentionally revisit the complete H row.
    # SF cursor never includes cluster_y.
    # --------------------------------------------------------------
    if row_valid:
        SF_PER_THREAD = ceil_div(H//BLOCK_SIZE,TPR)

        def process_one_sf(sf_idx):
            if sf_idx < H//BLOCK_SIZE:
                block_start = sf_idx*BLOCK_SIZE

                if BLOCK_SIZE == 16:
                    # Issue all four 128-bit loads before packed arithmetic.
                    x_words = reg_fragment("u32",8)
                    w_words = reg_fragment("u32",8)
                    x_words[0:4] = load_global_128(
                        X+actual_row*H+block_start)
                    # instruction_selection: ld.global.v4.u32; X values 0..7
                    x_words[4:8] = load_global_128(
                        X+actual_row*H+block_start+8)
                    # instruction_selection: ld.global.v4.u32; X values 8..15
                    w_words[0:4] = load_global_128(W+block_start)
                    # instruction_selection: ld.global.v4.u32; W values 0..7
                    w_words[4:8] = load_global_128(W+block_start+8)
                    # instruction_selection: ld.global.v4.u32; W values 8..15
                    # Extent is exactly two X then two W 128-bit issues/SF.

                    xw = reg_fragment(INPUT_DTYPE+"x2",8)
                    for pair in static_unroll(8):
                        xw[pair] = mul_input2(x_words,w_words,pair,INPUT_DTYPE)
                        # instruction_selection: mul.f16x2 or mul.bf16x2;
                        # eight packed issues, preserving input-dtype rounding
                    abs_xw = reg_fragment("u32",8)
                    for pair in static_unroll(8):
                        abs_xw[pair] = and_b32(xw[pair],0x7fff7fff)
                        # instruction_selection: and.b32 with 0x7fff7fff;
                        # exactly eight packed absolute-value issues/SF
                    max01 = max_input2(abs_xw[0],abs_xw[1],INPUT_DTYPE)
                    max23 = max_input2(abs_xw[2],abs_xw[3],INPUT_DTYPE)
                    max45 = max_input2(abs_xw[4],abs_xw[5],INPUT_DTYPE)
                    max67 = max_input2(abs_xw[6],abs_xw[7],INPUT_DTYPE)
                    max03 = max_input2(max01,max23,INPUT_DTYPE)
                    max47 = max_input2(max45,max67,INPUT_DTYPE)
                    pair_max = max_input2(max03,max47,INPUT_DTYPE)
                    # instruction_selection: max.f16x2 or max.bf16x2;
                    # exactly seven issues in the source 8->4->2->1 tree
                    if INPUT_DTYPE == "f16":
                        max_lo16,max_hi16 = mov_b32_pair(pair_max)
                        # instruction_selection: mov.b32 {b16,b16}; one/SF
                        max_lo = cvt_f32_f16(max_lo16)
                        max_hi = cvt_f32_f16(max_hi16)
                        # instruction_selection: cvt.f32.f16; two/SF
                    else:
                        max_lo32 = and_b32(pair_max,0xffff)
                        max_hi32 = shift_right_b32(pair_max,16)
                        # instruction_selection: and.b32 then shr.b32; one each/SF
                        max_lo32 = shift_left_b32(max_lo32,16)
                        max_hi32 = shift_left_b32(max_hi32,16)
                        # instruction_selection: shl.b32; two/SF
                        max_lo = mov_b32_as_f32(max_lo32)
                        max_hi = mov_b32_as_f32(max_hi32)
                        # instruction_selection: mov.b32 to f32; two/SF
                    max_xw = max_f32(max_lo,max_hi)
                    # instruction_selection: max.f32; one horizontal max/SF
                    y_f32 = reg_fragment("f32",16)
                    for pair in static_unroll(8):
                        if INPUT_DTYPE == "f16":
                            lo16,hi16 = mov_b32_pair(xw[pair])
                            # instruction_selection: mov.b32 {b16,b16}; one/pair
                            lo = cvt_f32_f16(lo16)
                            hi = cvt_f32_f16(hi16)
                            # instruction_selection: cvt.f32.f16; two/pair
                        else:
                            lo32 = and_b32(xw[pair],0xffff)
                            hi32 = shift_right_b32(xw[pair],16)
                            lo32 = shift_left_b32(lo32,16)
                            hi32 = shift_left_b32(hi32,16)
                            # instruction_selection: and.b32, shr.b32, and two
                            # shl.b32 issues per packed BF16 pair
                            lo = mov_b32_as_f32(lo32)
                            hi = mov_b32_as_f32(hi32)
                            # instruction_selection: mov.b32 to f32; two/pair
                        y_f32[2*pair] = mul_f32(lo,rstd)
                        y_f32[2*pair+1] = mul_f32(hi,rstd)
                        # instruction_selection: mul.f32; sixteen values/SF
                    max_abs = mul_f32(max_xw,rstd)
                    # instruction_selection: mul.f32; one/SF

                    scale_float = mul_f32(
                        mul_f32(global_scale_value,max_abs),fp4_max_rcp)
                    # instruction_selection: mul.f32 chain; two issues/SF
                    scale_float = min_f32(scale_float,448.0)
                    # instruction_selection: min.f32; one/SF
                    fp8_zero = mov_f32(0.0)
                    # instruction_selection: mov.f32 zero; one/SF
                    scale_pair16 = cvt_e4m3_pair(fp8_zero,scale_float)
                    # instruction_selection: cvt.rn.satfinite.e4m3x2.f32;
                    # zero is high mate and scale is low mate, one/SF
                    scale_word = cvt_u32_u16(scale_pair16)
                    # instruction_selection: cvt.u32.u16; one/SF
                    scale_byte = scale_word & 0xff
                    # instruction_selection: source low-byte view is DCE;
                    # zero standalone instructions before st.global.b8
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
                    decoded_zero = setp_eq_f32(decoded,0.0)
                    # instruction_selection: setp.eq.f32; one/SF
                    decoded_rcp = select_f32(0.0,reciprocal,decoded_zero)
                    # instruction_selection: selp.f32; one/SF, zero stays zero
                    inv_scale = mul_f32(decoded_rcp,global_scale_value)
                    # instruction_selection: mul.f32; one/SF

                    s_offset = swizzled_scale_offset(actual_row,sf_idx) \
                               if SWIZZLED else actual_row*(H//16)+sf_idx
                    store_global_b8(S_u8+s_offset,scale_byte)
                    # instruction_selection: st.global.b8; one/SF, before Y

                    scaled = reg_fragment("f32",16)
                    for value in static_unroll(16):
                        scaled[value] = mul_f32(y_f32[value],inv_scale)
                        # instruction_selection: mul.f32; sixteen values/SF
                    pair_bytes = reg_fragment("u8",8)
                    for pair in static_unroll(8):
                        pair_bytes[pair] = cvt_e2m1_pair(
                            scaled[2*pair+1],scaled[2*pair])
                        # instruction_selection:
                        # cvt.rn.satfinite.e2m1x2.f32 to b8; eight/SF, PTX high then
                        # low so logical element 0 occupies the low nibble
                    packed_lo32 = mov_b32_bytes(
                        pair_bytes[0],pair_bytes[1],pair_bytes[2],pair_bytes[3])
                    packed_hi32 = mov_b32_bytes(
                        pair_bytes[4],pair_bytes[5],pair_bytes[6],pair_bytes[7])
                    # instruction_selection: mov.b32 {b8,b8,b8,b8}; two/SF
                    packed_lo64 = cvt_u64_u32(packed_lo32)
                    packed_hi64 = cvt_u64_u32(packed_hi32)
                    # instruction_selection: cvt.u64.u32; two/SF
                    packed_hi64 = shift_left_b64(packed_hi64,32)
                    # instruction_selection: shl.b64; one/SF
                    packed64 = or_b64(packed_hi64,packed_lo64)
                    # instruction_selection: or.b64; one/SF
                    store_global_u64(
                        Y_u8 + actual_row*(H//2) + block_start//2,packed64)
                    # instruction_selection: st.global.u64; one/SF

                else:  # BLOCK_SIZE == 32
                    # Source issues both 16-value chunks' X/W loads before
                    # consuming either chunk.
                    x0 = reg_fragment("u32",8)
                    w0 = reg_fragment("u32",8)
                    x1 = reg_fragment("u32",8)
                    w1 = reg_fragment("u32",8)
                    x0[0:4] = load_global_128(X+actual_row*H+block_start)
                    x0[4:8] = load_global_128(X+actual_row*H+block_start+8)
                    w0[0:4] = load_global_128(W+block_start)
                    w0[4:8] = load_global_128(W+block_start+8)
                    x1[0:4] = load_global_128(X+actual_row*H+block_start+16)
                    x1[4:8] = load_global_128(X+actual_row*H+block_start+24)
                    w1[0:4] = load_global_128(W+block_start+16)
                    w1[4:8] = load_global_128(W+block_start+24)
                    # instruction_selection: ld.global.v4.u32; exactly eight
                    # issues ordered X0,X0,W0,W0,X1,X1,W1,W1 before arithmetic
                    xw0 = reg_fragment("u32",8)
                    xw1 = reg_fragment("u32",8)
                    for pair in static_unroll(8):
                        xw0[pair] = mul_input2(x0[pair],w0[pair],INPUT_DTYPE)
                        # instruction_selection: mul.f16x2 or mul.bf16x2;
                        # eight chunk0 packed issues before chunk1 begins
                    for pair in static_unroll(8):
                        xw1[pair] = mul_input2(x1[pair],w1[pair],INPUT_DTYPE)
                        # instruction_selection: mul.f16x2 or mul.bf16x2;
                        # eight chunk1 packed issues after all chunk0 multiplies
                    abs0 = reg_fragment("u32",8)
                    abs1 = reg_fragment("u32",8)
                    for pair in static_unroll(8):
                        abs0[pair] = and_b32(xw0[pair],0x7fff7fff)
                        # instruction_selection: and.b32 0x7fff7fff;
                        # eight chunk0 packed absolute-value issues
                    m0_01 = max_input2(abs0[0],abs0[1],INPUT_DTYPE)
                    m0_23 = max_input2(abs0[2],abs0[3],INPUT_DTYPE)
                    m0_45 = max_input2(abs0[4],abs0[5],INPUT_DTYPE)
                    m0_67 = max_input2(abs0[6],abs0[7],INPUT_DTYPE)
                    m0_03 = max_input2(m0_01,m0_23,INPUT_DTYPE)
                    m0_47 = max_input2(m0_45,m0_67,INPUT_DTYPE)
                    max0 = max_input2(m0_03,m0_47,INPUT_DTYPE)
                    # instruction_selection: max.f16x2 or max.bf16x2;
                    # complete seven-issue chunk0 tree before chunk1 abs work
                    for pair in static_unroll(8):
                        abs1[pair] = and_b32(xw1[pair],0x7fff7fff)
                        # instruction_selection: and.b32 0x7fff7fff;
                        # eight chunk1 packed absolute-value issues
                    m1_01 = max_input2(abs1[0],abs1[1],INPUT_DTYPE)
                    m1_23 = max_input2(abs1[2],abs1[3],INPUT_DTYPE)
                    m1_45 = max_input2(abs1[4],abs1[5],INPUT_DTYPE)
                    m1_67 = max_input2(abs1[6],abs1[7],INPUT_DTYPE)
                    m1_03 = max_input2(m1_01,m1_23,INPUT_DTYPE)
                    m1_47 = max_input2(m1_45,m1_67,INPUT_DTYPE)
                    max1 = max_input2(m1_03,m1_47,INPUT_DTYPE)
                    # instruction_selection: max.f16x2 or max.bf16x2;
                    # complete seven-issue chunk1 tree after all chunk1 abs work
                    max_pair = max_input2(max0,max1,INPUT_DTYPE)
                    # instruction_selection: max.f16x2 or max.bf16x2;
                    # one final cross-chunk issue; fifteen max issues total
                    if INPUT_DTYPE == "f16":
                        max_lo16,max_hi16 = mov_b32_pair(max_pair)
                        # instruction_selection: mov.b32 {b16,b16}; one/SF
                        max_lo = cvt_f32_f16(max_lo16)
                        max_hi = cvt_f32_f16(max_hi16)
                        # instruction_selection: cvt.f32.f16; two/SF
                    else:
                        max_lo32 = and_b32(max_pair,0xffff)
                        max_hi32 = shift_right_b32(max_pair,16)
                        max_lo32 = shift_left_b32(max_lo32,16)
                        max_hi32 = shift_left_b32(max_hi32,16)
                        # instruction_selection: and.b32, shr.b32, then two
                        # shl.b32 issues/SF
                        max_lo = mov_b32_as_f32(max_lo32)
                        max_hi = mov_b32_as_f32(max_hi32)
                        # instruction_selection: mov.b32 to f32; two/SF
                    max_xw = max_f32(max_lo,max_hi)
                    # instruction_selection: max.f32; one horizontal max/SF
                    y = reg_fragment("f32",(2,16))
                    for chunk in static_unroll(2):
                        words = xw0 if chunk == 0 else xw1
                        for pair in static_unroll(8):
                            if INPUT_DTYPE == "f16":
                                lo16,hi16 = mov_b32_pair(words[pair])
                                # instruction_selection: mov.b32 pair; one/pair
                                lo = cvt_f32_f16(lo16)
                                hi = cvt_f32_f16(hi16)
                                # instruction_selection: cvt.f32.f16; two/pair
                            else:
                                lo32 = and_b32(words[pair],0xffff)
                                hi32 = shift_right_b32(words[pair],16)
                                lo32 = shift_left_b32(lo32,16)
                                hi32 = shift_left_b32(hi32,16)
                                # instruction_selection: and.b32, shr.b32,
                                # two shl.b32 issues/pair
                                lo = mov_b32_as_f32(lo32)
                                hi = mov_b32_as_f32(hi32)
                                # instruction_selection: mov.b32 to f32; two/pair
                            y[chunk,2*pair] = mul_f32(lo,rstd)
                            y[chunk,2*pair+1] = mul_f32(hi,rstd)
                            # instruction_selection: mul.f32; 32 values/SF
                    max_abs = mul_f32(max_xw,rstd)
                    # instruction_selection: mul.f32; one/SF

                    if SCALE_FORMAT == "ue8m0":
                        scale_float = mul_f32(max_abs,fp4_max_rcp)
                        # instruction_selection: mul.f32; one/SF
                        nonpositive = setp_le_f32(scale_float,0.0)
                        # instruction_selection: setp.le.f32; one/SF
                        log2_value = lg2_fast(scale_float)
                        # instruction_selection: lg2.approx.f32; one/SF
                        exponent = cvt_s32_f32_rpi(log2_value)
                        # instruction_selection: cvt.rpi.s32.f32; one/SF
                        biased = add_s32(exponent,127)
                        # instruction_selection: add.s32; one/SF
                        negative = setp_lt_s32(biased,0)
                        overflow = setp_gt_s32(biased,255)
                        # instruction_selection: setp.lt.s32 and setp.gt.s32;
                        # one each/SF
                        biased = select_s32(0,biased,negative)
                        biased = select_s32(255,biased,overflow)
                        scale_word = select_s32(0,biased,nonpositive)
                        # instruction_selection: selp.s32; three/SF in this order
                        scale_byte = scale_word & 0xff
                        # instruction_selection: source low-byte view is DCE;
                        # zero standalone instructions before st.global.b8
                        scale_zero = setp_eq_u32(scale_word,0)
                        # instruction_selection: setp.eq.u32; one/SF
                        negative_exponent = sub_s32(127,scale_word)
                        # instruction_selection: sub.s32; one/SF
                        negative_exponent_f32 = cvt_f32_s32_rn(negative_exponent)
                        # instruction_selection: cvt.rn.f32.s32; one/SF
                        inverse_candidate = ex2_fast(negative_exponent_f32)
                        # instruction_selection: ex2.approx.f32; one/SF
                        inv_scale = select_f32(0.0,inverse_candidate,scale_zero)
                        # instruction_selection: selp.f32; one/SF
                    else:
                        scale_float = mul_f32(
                            mul_f32(global_scale_value,max_abs),fp4_max_rcp)
                        # instruction_selection: mul.f32 chain; two/SF
                        scale_float = min_f32(scale_float,448.0)
                        # instruction_selection: min.f32; one/SF
                        fp8_zero = mov_f32(0.0)
                        # instruction_selection: mov.f32 zero; one/SF
                        scale_pair16 = cvt_e4m3_pair(fp8_zero,scale_float)
                        # instruction_selection:
                        # cvt.rn.satfinite.e4m3x2.f32; one/SF
                        scale_word = cvt_u32_u16(scale_pair16)
                        # instruction_selection: cvt.u32.u16; one/SF
                        scale_byte = scale_word & 0xff
                        # instruction_selection: source low-byte view is DCE;
                        # zero standalone instructions before st.global.b8
                        decode_pair16 = cvt_u16_u32(scale_word)
                        # instruction_selection: cvt.u16.u32; one/SF
                        decoded_h2 = cvt_f16x2_e4m3x2(decode_pair16)
                        # instruction_selection: cvt.rn.f16x2.e4m3x2; one/SF
                        decoded_lo16,_ = mov_b32_pair(decoded_h2)
                        # instruction_selection: mov.b32 pair; one/SF
                        decoded = cvt_f32_f16(decoded_lo16)
                        # instruction_selection: cvt.f32.f16; one/SF
                        reciprocal = rcp_fast(decoded)
                        # instruction_selection: rcp.approx.ftz.f32; one/SF
                        decoded_zero = setp_eq_f32(decoded,0.0)
                        # instruction_selection: setp.eq.f32; one/SF
                        decoded_rcp = select_f32(0.0,reciprocal,decoded_zero)
                        # instruction_selection: selp.f32; one/SF
                        inv_scale = mul_f32(decoded_rcp,global_scale_value)
                        # instruction_selection: mul.f32; one/SF

                    s_offset = swizzled_scale_offset(actual_row,sf_idx) \
                               if SWIZZLED else actual_row*(H//32)+sf_idx
                    store_global_b8(S_u8+s_offset,scale_byte)
                    # instruction_selection: st.global.b8; one/SF, before Y
                    for chunk in static_unroll(2):
                        scaled = reg_fragment("f32",16)
                        for value in static_unroll(16):
                            scaled[value] = mul_f32(y[chunk,value],inv_scale)
                            # instruction_selection: mul.f32; sixteen/chunk
                        pair_bytes = reg_fragment("u8",8)
                        for pair in static_unroll(8):
                            pair_bytes[pair] = cvt_e2m1_pair(
                                scaled[2*pair+1],scaled[2*pair])
                            # instruction_selection:
                            # cvt.rn.satfinite.e2m1x2.f32 to b8; eight/chunk
                        packed_lo32 = mov_b32_bytes(
                            pair_bytes[0],pair_bytes[1],
                            pair_bytes[2],pair_bytes[3])
                        packed_hi32 = mov_b32_bytes(
                            pair_bytes[4],pair_bytes[5],
                            pair_bytes[6],pair_bytes[7])
                        # instruction_selection: mov.b32 {b8,b8,b8,b8};
                        # two/chunk
                        packed_lo64 = cvt_u64_u32(packed_lo32)
                        packed_hi64 = cvt_u64_u32(packed_hi32)
                        # instruction_selection: cvt.u64.u32; two/chunk
                        packed_hi64 = shift_left_b64(packed_hi64,32)
                        # instruction_selection: shl.b64; one/chunk
                        packed64 = or_b64(packed_hi64,packed_lo64)
                        # instruction_selection: or.b64; one/chunk
                        store_global_u64(
                            Y_u8+actual_row*(H//2)
                            +(block_start+chunk*16)//2,packed64)
                        # instruction_selection: st.global.u64; chunk0 then chunk1

        if SF_PER_THREAD <= 7:
            for sf_iter in static_unroll(SF_PER_THREAD):
                process_one_sf(lane_in_row + sf_iter*TPR)
        else:
            sf_iter = 0
            while sf_iter < SF_PER_THREAD:
                process_one_sf(lane_in_row + sf_iter*TPR)
                process_one_sf(lane_in_row + (sf_iter+1)*TPR)
                # second call retains its sf_idx < H/BLOCK_SIZE predicate
                sf_iter += 2
            # instruction_selection: add.s32 cursor by 2, setp.ne.b32 against
            # SF_PER_THREAD, then predicated `@p bra` backedge; two static SF
            # payloads/body and the second payload retains its sf bound guard

    if ENABLE_PDL:
        pdl_signal()
        # instruction_selection: griddepcontrol.launch_dependents; one CTA issue
```

## Scale addressing and literal source behavior

Linear S offset is `row*(H/BLOCK_SIZE)+sf`. Swizzled offset is exactly

```text
(row//128) * (ceil((H/BLOCK_SIZE)/4) * 512)
+ (sf//4) * 512
+ (row%32) * 16
+ ((row%128)//32) * 4
+ (sf%4)
```

The swizzled allocation is padded to
`ceil(M/128)*ceil((H/BLOCK_SIZE)/4)*512` bytes; only valid logical positions
are written. Phase 3 uses full-H `sf_idx` in every cluster CTA. The redundant
cluster stores are literal pinned-source behavior and are not deduplicated.

Block16 always executes E4M3 scaling even if the cache key says UE8M0.
Block32 chooses UE8M0 or E4M3. The public canonical pairs are block16/E4M3
and block32/UE8M0, but the reachable block32/E4M3 class specialization remains
in the correctness domain. The global scale is part of the E4M3 scale and
inverse-scale topology; it is not applied as a separate normalization factor.
UE8M0 ignores it, so its load may disappear from emitted PTX.

The packed multiply happens in FP16x2/BF16x2 before widening; replacing it
with FP32 multiplication changes both max-abs and raw output bytes. E2M1
conversion uses native saturating round-to-nearest pair instructions. Scale
bytes are computed and stored before packed Y bytes, preserving source order.

## Verification locked into the module

The module's validation manifest covers the full upstream 1007-case logical
matrix plus structural guards for both input dtypes, both canonical formats,
reachable block32/E4M3, zero and explicit global scales, block/tile/thread
thresholds, row tails, 2-D/3-D flattening, swizzled M/K padding, PDL, the
7-to-8 loop boundary, and cluster 2/16. Every specialization passes the
low-level no-tile/no-call gate before execution. Correctness compares raw Y
bytes and logical S bytes with the CuTeDSL kernel, checks dequantized RMSNorm
against an independent FP32 oracle, verifies input/weight/global-scale
immutability, output identity and untouched padding/guards.

The required performance matrix is exactly the six pinned FlashInfer sample
rows. Only one reference-enabled `tirx_kernels.bench_suite` artifact may decide
performance retention; diagnostic timing or profiler duration is not a gate.
