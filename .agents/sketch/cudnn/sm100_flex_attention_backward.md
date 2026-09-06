<!--
Copyright (c) 2026 The TIRx Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This design sketch documents a TIRx port of cuDNN Frontend's
python/cudnn/flex_attention/kernels/sm100/bwd/backward.py at
edc7c2833327d699d70b616c6f264b6ae92599b2.
-->

# cudnn_sm100_flex_attention_backward: coarse WASP pipeline sketch

This is a non-executable semantic execution sketch for the direct SM100 Flex
Attention backward main kernel. The executable source of truth becomes
[`tirx_kernels/cudnn/flex_attention/flex_attention_backward_sm100.py`](../../../tirx_kernels/cudnn/flex_attention/flex_attention_backward_sm100.py)
and its private package after this sketch passes review. Planning, forward/LSE
generation, dPsum preprocessing, accumulator clears, and final gradient casts
are host/harness responsibilities and are outside the timed kernel.

The port follows the exact requested source at cuDNN Frontend commit
`edc7c2833327d699d70b616c6f264b6ae92599b2` with
`nvidia-cutlass-dsl==4.8.0.dev0`. It uses the direct
`FlexAttentionBackwardSm100.__call__` main-launch ABI, not the HD256 sibling.

## Scope and specialization boundary

The static domain is:

| axis | values |
| --- | --- |
| input dtype | FP16 or BF16; Q/K/V/dO share it |
| `(Dqk,Dv)` | independent multiples of 8 in `[8,128]`, plus `(192,128)` |
| padded dimensions | `TD=ceil(Dqk/16)*16`, `TDV=ceil(Dv/16)*16` |
| topology | CTA group 2 only for exact `(128,128)` and `(192,128)`; otherwise CTA group 1 |
| sequence mode | fixed BSHD or packed varlen THD with both cu-seqlens arrays |
| heads | MHA or GQA; `Hq/Hkv` is a positive integer |
| reduction order | nondeterministic bulk reduction, or deterministic semaphore-ordered |
| plan | arbitrary compact K2Q plan with partial/full rows, native packed masks, and dQ write ranks |

Out of scope: `(256,256)` (the separately dispatched `backward_hd256.py`),
FP8 inputs, dropout, SM90, SM103/SM107 publication, mask planning, forward,
preprocess, and postprocess. No code beneath `tirx_kernels/kern/` and no inline
CUDA function-call exemption belongs to this port.

Static source constants are `M=N=128`, `Q_STAGE=1` for CTA group 2 and `2`
otherwise, `DO_STAGE=1`, `DKV_STAGE=2`, and 512 TMEM columns. The launch has
512 threads, `.minnctapersm 1`, and either no cluster or cluster `(2,1,1)`.

## Writer line-info evidence

The writer exports are retained under
`.porting/flex_attention_backward_sm100/source_export/`; exact generation,
normalization, shared maps, and opcode evidence are recorded in
`source_export/findings.md`. Each case retains `generated.mlir`, raw
`source.generated_9_4.ptx`, and canonical `source.ptx`. CUTLASS 4.8's current
CUDA-13 wheel labeled the raw output 9.4; canonical files alter only the PTX
version directive to 9.2 and all assemble with CUDA 13.2 `ptxas -arch=sm_100a`.
The final source and TIRx benchmark executables must both use PTX 9.2.

| export | `.loc` | SMEM | topology |
| --- | ---: | ---: | --- |
| `fp16_d8_dv8_fixed_mha` | 1907 | 93184 | 1CTA |
| `bf16_d8_dv128_fixed_det_gqa4` | 2156 | 146432 | 1CTA |
| `bf16_d72_dv40_varlen_det_gqa4` | 3291 | 154624 | 1CTA |
| `fp16_d128_dv8_varlen_mha` | 2968 | 179200 | 1CTA |
| `bf16_d128_dv128_fixed_mha` | 2568 | 231424 | 2CTA |
| `bf16_d128_dv128_fixed_det_gqa4` | 2490 | 231424 | 2CTA |
| `bf16_d192_dv128_fixed_mha` | 2387 | 231424 | 2CTA |
| `fp16_d192_dv128_varlen_det_gqa4` | 3581 | 231424 | 2CTA |

Instruction counts use the corpus convention: instruction lines minus
predicated instruction lines. Counts are supporting evidence; the source-order
loop bodies below are the porting contract.

## Pipeline at a glance

Source role dispatch is a flat sequence in this exact order: empty, relay,
load, MMA, compute, reduce. The TIRx specialization must preserve the same
physical warp ownership and register transitions.

| warps | role | program | publication / reuse edges |
| --- | --- | --- | --- |
| 15 | empty | register decrease, no data work | participates only in CTA/cluster initialization before dispatch |
| 14 | relay | 2CTA only: one wait and one leader arrival for each active K2Q iteration | peer dS full -> leader dS-ready |
| 13 | load | scheduler/K2Q decode; stream Q/K/V/dO/LSE/dPsum, plus Qt/dOt/Kt in cooperative variants | TMA full barriers -> MMA or compute; consumers release empty barriers |
| 12 | MMA | allocate TMEM; issue all five tensor-core products in topology-specific order; relinquish and free | consumes Q/K/V/dO/dS; publishes S,dP,dQ,dK,dV |
| 4..11 | compute | T2R S/dP, apply packed mask, reconstruct P, form dS, R2T/R2S publication, dK/dV stores/reductions | consumes S,dP,LSE,dPsum,dKV; produces P/dS and global dK/dV |
| 0..3 | reduce | T2R dQ, stage FP32, source-ordered global reduce-add, deterministic semaphore protocol | consumes dQ; produces global dQ accumulator |

There is one scheduler item per physical N CTA. A 2CTA pair shares one logical
256-row sparse K tile. For each scheduler item, the K2Q plan splits work into
partial rows and full rows. The 1CTA order is partial then full; both cooperative
paths use full then partial. Every role traverses exactly the same counts and
indices even when its local program does not need the actual Q block number.

## Primitive vocabulary

Structural declarations below are linear and do no work:

```python
linear_smem(bytes, alignment)
linear_regs(dtype, count)
linear_tmem(base_column, row_band, column_count)
alias_bytes(parent, byte_offset, byte_count)
slice_bytes(parent, byte_offset, byte_count)
tensormap(name, rank, dtype, global_dims, strides, box, swizzle, l2)
mbarrier_pair(byte_offset, stages)
pipeline_state(stages, stage, phase)
```

No first-class layout appears. Named scalar offset functions below encode every
swizzle, stage stride, transpose, and ownership mapping that changes selected
bytes or instruction descriptors.

Data movement and computation primitives are deliberately small:

```python
copy_g2s_tma(desc, coords, smem_byte, mbarrier, multicast=None)
copy_g2s_bulk(src, smem_byte, bytes, mbarrier)
copy_s2cluster(src_byte, peer_byte, bytes, peer, mbarrier)
copy_t2r(tmem_addr, registers, repetition)
copy_r2t(registers, tmem_addr, repetition)
copy_r2s(registers, smem_byte, width)
copy_s2g_tma(desc, coords, smem_byte)
reduce_s2g_add_f32(dst, smem_byte, bytes)
load_global(ptr, index, width=1)
store_global(ptr, index, value, width=1)
load_shared(smem_byte, width=1)
store_shared(smem_byte, value, width=1)
prefetch_descriptor(desc)

gemm(dst_tmem, a_desc_or_tmem, b_desc, instruction_desc, accumulate)
fill(dst, value)
cast(dst, src)
fma(dst, a, b, c, lanes=1)
sub(dst, a, b, lanes=1)
mul(dst, a, b, lanes=1)
exp2(dst, src)
select(dst, predicate, true_value, false_value)
```

Schedule primitives are `init`, `acquire`, `wait`, `expect_tx`, `commit`,
`arrive`, `release`, `advance`, `fence_mbarrier_init`, `fence_tmem_load`,
`fence_tmem_store`, `fence_shared_async`, `fence_proxy_shared`,
`named_barrier`, `cluster_arrive`, `cluster_wait`, `tma_store_commit`,
`tma_store_wait`, `tmem_alloc`, `tmem_relinquish`, `tmem_free`, `elect`, and
`set_register_budget`.

## Complete sketch

```python
# ========================================================================
# Static specialization, direct runtime ABI, and launch
# ========================================================================

@specialize(ELEM, D, DV, HQRATIO, DETERMINISTIC, VARLEN,
            CTA_GROUP=(2 if (D,DV) in ((128,128),(192,128)) else 1))
TD = ceil_div(D, 16) * 16
TDV = ceil_div(DV, 16) * 16
Q_STAGE = 1 if CTA_GROUP == 2 else 2
DQ_NCOL = (24 if CTA_GROUP == 2 and TD == 192 else
           (16 if DETERMINISTIC else 8) if CTA_GROUP == 2 else gcd(32, TD))
DQ_SMEM_STAGES = ((2 if TD == 192 or DETERMINISTIC else 4)
                  if CTA_GROUP == 2 else 64 // DQ_NCOL)
DK_NCOL = gcd(32, TD // 2)
DV_NCOL = gcd(32, TDV // 2)
DKV_POSTPROCESS = HQRATIO > 1 or D != TD or DV != TDV
USE_TMA_STORE = DKV_POSTPROCESS or not (HQRATIO == 1 and VARLEN)
DIRECT_RAW_DKV = not USE_TMA_STORE       # exact-dimension varlen MHA only
EPI_K_COLS = gcd((32 if DKV_POSTPROCESS else 64), TD // 2)
EPI_V_COLS = gcd((32 if DKV_POSTPROCESS else 64), TDV // 2)
EPI_K_STAGES = max(1, (TD // 2) // EPI_K_COLS)
EPI_V_STAGES = max(1, (TDV // 2) // EPI_V_COLS)

def kernel(
    q_map, qt_map, k_map, kt_map, v_map, do_map, dot_map,
    lse_log2, dpsum, dq_accum, dk_out_or_accum, dv_out_or_accum,
    dq_sem, dk_sem, dv_sem, cu_q, cu_k,
    partial_count, partial_index, full_count, full_index,
    cu_k_blocks, dq_order_partial, dq_order_full,
    partial_offset, full_offset, packed_mask, sequence_desc, work_desc,
    softmax_scale,
):
    launch(block=(512,1,1), grid=scheduler_grid,
           cluster=((2,1,1) if CTA_GROUP == 2 else None),
           min_blocks_per_sm=1, dynamic_smem=SMEM_BYTES)
    # instruction_selection: `.reqntid 512,1,1`, `.minnctapersm 1`,
    # `.extern .shared .align 1024`; extent: one specialized entry. The
    # cooperative entries use CTA-group-two instructions and cluster launch
    # metadata; writer PTX `source.ptx` entry directives.

    warp = warp_uniform(thread_id() // 32)
    # instruction_selection: `shfl.sync.idx.b32`; extent: one warp-uniform
    # broadcast of the physical warp id (source :943).
    cta_rank = block_id_in_cluster()
    logical_n = scheduler_n // CTA_GROUP
    leader_cta = (cta_rank == 0)

    if warp == 13 and elect():
        for desc in present(q_map, qt_map, k_map, kt_map, v_map,
                            do_map, dot_map, direct_dk_map, direct_dv_map):
            prefetch_descriptor(desc)
            # instruction_selection: `prefetch.tensormap`; extent: one elected
            # issue per present descriptor. Static counts are 4 for 1CTA
            # accumulation or raw-varlen-direct cases, 6 for 1CTA fixed direct,
            # 7 for cooperative accumulation/raw-varlen-direct, and 9 for
            # cooperative fixed direct.

    # ====================================================================
    # One rank-one dynamic shared arena and explicit field offsets
    # ====================================================================
    smem = linear_smem(SMEM_BYTES, alignment=1024)
    # instruction_selection: dynamic `.extern .shared .align 1024`; extent:
    # one allocation. There is no static shared object.

    # 1CTA prefix. Q/LSE have 2 stages; all other listed pairs have the stage
    # count in their names. In 2CTA, Q/LSE shrink to 1 stage and cooperative
    # pairs occupy 176..239 as documented below.
    Q_BAR       = 0
    DO_BAR      = 32 if CTA_GROUP == 1 else 16
    LSE_BAR     = 48 if CTA_GROUP == 1 else 32
    DPSUM_BAR   = 80 if CTA_GROUP == 1 else 48
    SP_BAR      = 96 if CTA_GROUP == 1 else 64
    DP_BAR      = 112 if CTA_GROUP == 1 else 80
    DS_BAR      = 128 if CTA_GROUP == 1 else 96
    DKV_BAR     = 144 if CTA_GROUP == 1 else 112
    DQ_BAR      = 176 if CTA_GROUP == 1 else 144
    TMEM_WORD   = 192 if CTA_GROUP == 1 else 160
    TMEM_FREE_B = 200 if CTA_GROUP == 1 else 168
    QT_BAR      = 176                         # 2CTA only
    KT_BAR      = 192                         # 2CTA only
    DS_EMPTY_C  = 208                         # 2CTA only
    DS_FULL_C   = 216                         # 2CTA only
    DS_LEADER_C = 224                         # 2CTA only
    DQ_EMPTY_C  = 232                         # 2CTA D192 alias guard

    # Tile bases/sizes come from specialization constants. These anchor maps
    # bound the piecewise formulas used by the executable module:
    # D8/DV8 1CTA:
    #   sQ 1024+8192, sK 9216+4096, sV 13312+4096, sdO 17408+8192,
    #   sdS 25600+32768, sLSE 58368+1024, sdPsum 59392+512,
    #   sdQacc 60416+32768, total 93184.
    # D128/DV128 2CTA:
    #   sQ 1024+16384, sK 17408+32768, sV 50176+32768,
    #   sdO 82944+16384, sQt 99328+16384, sdOt 115712+16384,
    #   sdSx 132096+16384, sKt 148480+32768, sdS 181248+32768,
    #   sLSE 214016+512, sdPsum 214528+512, sdQacc 215040+16384,
    #   total 231424.
    # D192/DV128 2CTA:
    #   sQ 1024+24576, sK 25600+49152, sV 74752+32768,
    #   sdO 107520+16384, sQt/sdOt/sdSx zero-size at 123904,
    #   sKt 123904+49152, sdS 173056+32768, sLSE 205824+512,
    #   sdPsum 206336+512, sdQacc 206848+24576, total 231424.
    sQ      = alias_bytes(smem, SQ_BASE, SQ_BYTES)
    sK      = alias_bytes(smem, SK_BASE, SK_BYTES)
    sV      = alias_bytes(smem, SV_BASE, SV_BYTES)
    sdO     = alias_bytes(smem, SDO_BASE, SDO_BYTES)
    # D128 2CTA owns separate transpose fields.  In 1CTA and D192 the
    # transpose names are distinct descriptor views over the same bytes.
    sQt     = (alias_bytes(smem, SQT_BASE, SQT_BYTES)
               if CTA_GROUP == 2 and TD == 128 else sQ)
    sdOt    = (alias_bytes(smem, SDOT_BASE, SDOT_BYTES)
               if CTA_GROUP == 2 and TD == 128 else sdO)
    sKt     = (alias_bytes(smem, SKT_BASE, SKT_BYTES)
               if CTA_GROUP == 2 else sK)
    sdSx    = (alias_bytes(smem, SDSX_BASE, SDSX_BYTES)
               if CTA_GROUP == 2 and TD == 128 else None)
    sdS     = alias_bytes(smem, SDS_BASE, SDS_BYTES)
    sLSE    = alias_bytes(smem, SLSE_BASE, SLSE_BYTES)
    sdPsum  = alias_bytes(smem, SDPSUM_BASE, SDPSUM_BYTES)
    sdQacc  = alias_bytes(smem, SDQ_BASE, SDQ_BYTES)
    sdK = alias_bytes(sK if CTA_GROUP == 2 else sQ, 0, SDK_BYTES)
    sdV = alias_bytes(sV if CTA_GROUP == 2 else sdO, 0, SDV_BYTES)
    # D192's sdSx is a narrow transpose/exchange view of the first 16,384
    # bytes of sdQacc; D128's is a separate physical field.
    if CTA_GROUP == 2 and TD == 192:
        sdSx = alias_bytes(sdQacc, 0, SDS_EXCHANGE_BYTES)

    # Exact scalar maps transcribed from the generated native maps.  CuTe's
    # S<B,4,3> acts on *element* indices, hence the final multiply by two.
    # These functions are integer formulas, not first-class layout objects.
    def swz(B, element):
        return element ^ (((element >> 7) & ((1 << B) - 1)) << 4)
    def narrow_byte(base, B, element):
        return base + 2 * swz(B, element)

    if CTA_GROUP == 1:
        G, GV, B = TD // 16, TDV // 16, 1
        # Semantic domains are m,n in [0,128), d in [0,TD), v in
        # [0,TDV), qstage in [0,2).  The transpose views alias their named
        # parents but intentionally use different scalar maps.
        def sQ_byte(m,d,qstage):
            return narrow_byte(SQ_BASE,B,16*m+(d%16)+2048*(d//16)+128*TD*qstage)
        def sK_byte(n,d):
            return narrow_byte(SK_BASE,B,16*n+(d%16)+2048*(d//16))
        def sV_byte(n,v):
            return narrow_byte(SV_BASE,B,16*n+(v%16)+2048*(v//16))
        def sdO_byte(m,v):
            return narrow_byte(SDO_BASE,B,(v%16)+2048*(v//16)+16*(m%16)+256*(m//16))
        def sQt_byte(d,m,qstage):
            return narrow_byte(SQ_BASE,B,(d%16)+2048*(d//16)+16*(m%16)+256*(m//16)+128*TD*qstage)
        def sdOt_byte(v,m):
            return narrow_byte(SDO_BASE,B,16*m+(v%16)+2048*(v//16))
        def sKt_byte(d,n):
            return narrow_byte(SK_BASE,B,(d%16)+2048*(d//16)+16*(n%16)+256*(n//16))
    else:
        G, GV, B = TD // 16, TDV // 16, 3
        # Cooperative resident K/V rows are CTA-local n in [0,128); Q/dO
        # rows are multicast-local m in [0,64).  b=d//16 and bv=v//16.
        def sQ_byte(m,d):
            b=d//16
            return narrow_byte(SQ_BASE,B,64*m+(d%16)+16*(b%4)+4096*(b//4))
        def sK_byte(n,d):
            b=d//16
            return narrow_byte(SK_BASE,B,64*n+(d%16)+16*(b%4)+8192*(b//4))
        def sV_byte(n,v):
            b=v//16
            return narrow_byte(SV_BASE,B,64*n+(v%16)+16*(b%4)+8192*(b//4))
        def sdO_byte(m,v):
            return narrow_byte(SDO_BASE,B,m+64*(v%16)+1024*(v//16))

        if TD == 128:
            def sQt_byte(d,m):
                return narrow_byte(SQT_BASE,3,m+64*(d%16)+1024*(d//16))
            def sdOt_byte(v,m):
                b=v//16
                return narrow_byte(SDOT_BASE,3,64*m+(v%16)+16*(b%4)+4096*(b//4))
            def sKt_byte(d,n):
                return narrow_byte(SKT_BASE,3,(d%64)+64*(n%16)+1024*(n//16)+8192*(d//64))
        else:
            # D192 native transpose domains are explicit because the
            # CTA-group-two atom interleaves semantic axes.  sQt aliases sQ:
            # (a0,a1,a2,c) in [32,3,16,8], raw strides [1,4096,32,512].
            # sKt is separate: (a0,a1,a2,c) in [32,3,16,16], strides
            # [1,8192,32,512].  These are exactly the Q/Qt and Kt descriptor
            # coordinates supplied by the source's MMA/TMA partitions.
            def sQt_native_byte(a0,a1,a2,c):
                return narrow_byte(SQ_BASE,2,a0+4096*a1+32*a2+512*c)
            def sdOt_byte(v,m):
                b=v//16
                return narrow_byte(SDO_BASE,3,64*m+(v%16)+16*(b%4)+4096*(b//4))
            def sKt_native_byte(a0,a1,a2,c):
                return narrow_byte(SKT_BASE,2,a0+8192*a1+32*a2+512*c)

    # dS has two descriptor views over one allocation in every topology.
    # Both domains are row,col in [0,128).  The first is dQ's MN-major A;
    # the second is dK's K-major transposed A.
    def sdS_byte(row,col):
        return narrow_byte(SDS_BASE,3,(row%64)+8192*(row//64)+64*(col%16)+1024*(col//16))
    def sdSt_byte(row,col):
        b=col//16
        return narrow_byte(SDS_BASE,3,64*row+(col%16)+16*(b%4)+8192*(b//4))
    def sdSx_byte(row,col):               # row [0,128), col [0,64)
        return (SDSX_BASE if TD == 128 else SDQ_BASE) + 2*(row+128*col)
    def sLSE_byte(row,stage):
        return SLSE_BASE + 4*(row+128*stage)
    def sdPsum_byte(row):
        return SDPSUM_BASE + 4*row
    def sdQacc_byte(row,col,stage):
        return SDQ_BASE + 4*(row+128*col+128*DQ_NCOL*stage)

    # Staged dK/dV output aliases. Accumulation uses FP32 linear slices;
    # exact fixed-shape MHA uses narrow linear TMA tiles. Exact varlen MHA
    # has no sdK/sdV access because DIRECT_RAW_DKV stores registers directly.
    def dkv_accum_byte(base,row,col,wg,ncol):
        return base + 4*(row+128*col+128*ncol*wg)
    def dkv_narrow_byte(base,element,wg,elements_per_wg):
        return base + 2*(element+elements_per_wg*wg)

    # Range proof against the exported field sizes: swz only XOR-permutes
    # bits inside each listed span.  In 1CTA the narrow spans are sQ=2*128*TD*2,
    # sK=2*128*TD, sV=2*128*TDV, sdO/sdOt<=2*128*TDV, and all transpose
    # aliases have the same bound as their parent.  In 2CTA they are
    # sQ/sQt=2*64*TD, sK/sKt=2*128*TD, sV=2*128*TDV,
    # sdO/sdOt=2*64*TDV.  sdS/sdSt and sdSx span 32768 and 16384 bytes.
    # The linear FP32 spans are LSE=4*128*Q_STAGE, dPsum=512, and
    # dQ=4*128*DQ_NCOL*DQ_SMEM_STAGES.  Accumulation dK/dV spans
    # 4*128*{DK,DV}_NCOL*2; narrow TMA staging spans
    # 2*(128*gcd(64,{TD,TDV}//2))*2.  Each expression is <= its physical
    # field size above for every supported specialization; D192's 16,384-byte
    # sdSx subspan is < its 24,576-byte sdQ parent.
    def packed_mask_word(payload, subtile, consumer_thread, word):
        return (((payload * CTA_GROUP + subtile) * 256 + consumer_thread) * 2 + word)

    # TMEM aliases. Columns are exact specialization scalars, never objects.
    if CTA_GROUP == 2 and TD == 192:
        TV=0; TK=TDV; TS=TK+TD; TP=TS; TDP=512-128; TDS=TDP; TDQ=512-TD//2
    else:
        TS=0; TP=0; TV=128; TDP=TV+TDV; TDS=TDP
        TDQ=(TD//2 if CTA_GROUP == 2 else TDP); TK=TDP+128
    tS  = linear_tmem(TS,  cta_rank*128, 128)
    tP  = linear_tmem(TP,  cta_rank*128, TDV//2)
    tdP = linear_tmem(TDP, cta_rank*128, 128)
    tdS = linear_tmem(TDS, cta_rank*128, TD//2)
    tdV = linear_tmem(TV,  cta_rank*128, TDV)
    tdK = linear_tmem(TK,  cta_rank*128, TD)
    tdQ = linear_tmem(TDQ, 0, TD//CTA_GROUP)

    # Pipeline barrier initialization. Counts are source logical agents (one
    # elected arrival per participating warp), not thread counts.
    init(Q_BAR, Q_STAGE, full=1, empty=CTA_GROUP)
    init(DO_BAR, 1, full=1, empty=CTA_GROUP)
    init(LSE_BAR, Q_STAGE, full=1, empty=8)
    init(DPSUM_BAR, 1, full=1, empty=8)
    init(SP_BAR, 1, full=1, empty=8*CTA_GROUP)
    init(DP_BAR, 1, full=1, empty=8*CTA_GROUP)
    init(DS_BAR, 1, full=8*CTA_GROUP, empty=1)
    init(DKV_BAR, 2, full=1, empty=8*CTA_GROUP)
    init(DQ_BAR, 1, full=1, empty=4*CTA_GROUP)
    if CTA_GROUP == 2:
        if TD == 128: init(QT_BAR, Q_STAGE, full=1, empty=2)
        init(KT_BAR, 1, full=1, empty=2)
        init(DS_EMPTY_C, 1, full=1)
        init(DS_FULL_C, 1, full=1)
        init(DS_LEADER_C, 1, full=2)
        if TD == 192: init(DQ_EMPTY_C, 1, full=4)
        init_tmem_deallocation(TMEM_FREE_B, arrivals=4)
        # The allocator owns this separate two-CTA deallocation mbarrier.
    # instruction_selection: `mbarrier.init.shared.b64`; exact static extent:
    # 24 sites for 1CTA, 28 for D128 2CTA, and 27 for D192 2CTA. D192 aliases
    # Q/Qt and therefore has no QT_BAR pair; the one deallocation site above
    # is included in each cooperative total.

    # Exact initial states. P(stage,phase) starts producer objects at (0,1);
    # C(stage,phase) starts consumers at (0,0). `advance` increments stage and
    # toggles phase on wrap. Single phase variables below start at one because
    # they address producer-side empty barriers directly.
    Q_LSE_p=P(0,1); dO_dPsum_p=P(0,1)
    Qt_p=P(0,1); Kt_p=P(0,1)
    Q_Qt_p=P(0,1); O_Ot_p=P(0,1); LSE_p=P(0,1); dPsum_p=P(0,1)
    Q_c=C(0,0); Qt_c=C(0,0); Kt_c=C(0,0); dO_c=C(0,0)
    LSE_c=C(0,0); dPsum_c=C(0,0); SP_dP_c=C(0,0)
    dS_p=P(0,1); dS_c=C(0,0); dKV_c=C(0,0); dQ_c=C(0,0)
    dQ_smem_p=P(0,1)
    producer_phase_acc=1; producer_phase_dQ=1; producer_phase_dKV=1
    relay_phase=0; dS_cluster_phase=0

    fence_mbarrier_init()
    # instruction_selection: `fence.mbarrier_init.release.cluster`; extent:
    # publication epochs emitted by the deferred pipeline constructors.
    if CTA_GROUP == 1:
        named_barrier(id=0, threads=512)
        # instruction_selection: `bar.sync 0`; extent: CTA publication epoch.
    else:
        cluster_arrive(relaxed=True); cluster_wait()
        # instruction_selection: `barrier.cluster.arrive.relaxed` then
        # `barrier.cluster.wait`; extent: cluster publication epoch.

    # Every wait below is a retry loop around
    # `mbarrier.try_wait.parity.shared[::cta].b64 ...,10000000`; every arrive
    # is `mbarrier.arrive.shared[::cluster].b64` or a matching tcgen05 commit.

    # Source-order register redistribution. Cooperative overrides are applied
    # after base setup and are identical for D128 and D192:
    #   1CTA: reduce +152, compute +136, load/MMA -88, relay/empty -24
    #   2CTA MHA nondet: reduce +128, compute +144, load/MMA/relay -96
    #   2CTA MHA det:    reduce +128, compute +136, load/relay -96, MMA -112
    #   2CTA GQA:        reduce +128, compute +136, load/MMA/relay -112
    # Empty remains -24. Relay always takes the selected MMA budget in 2CTA.
    set_register_budget_by_role()
    # instruction_selection: `setmaxnreg.inc.sync.aligned.u32` for reduce and
    # compute; `setmaxnreg.dec.sync.aligned.u32` for load/MMA/relay/empty;
    # extent: one collective transition per physical warpgroup.

    # ====================================================================
    # Warp 15: empty
    # ====================================================================
    if warp == 15:
        pass

    # ====================================================================
    # Warp 14: cooperative dS relay
    # ====================================================================
    if warp == 14 and CTA_GROUP == 2:
        for work in scheduler_items():
            count = k2q_partial_count(work) + k2q_full_count(work)
            for iteration in range(count):
                wait(DS_FULL_C, relay_phase)
                # instruction_selection: retry loop containing
                # `mbarrier.try_wait.parity.shared.b64`; extent: one per active
                # K2Q iteration.
                if elect(): arrive(DS_LEADER_C, 0)
                # instruction_selection: `mbarrier.arrive.shared.b64`; extent:
                # one elected arrival per iteration, with two-CTA count 2.
                relay_phase ^= 1

    # ====================================================================
    # Warp 13: TMA producer
    # ====================================================================
    if warp == 13:
        for work in scheduler_items():
            n, hq, batch = work.n, work.head, work.batch
            hkv = hq // HQRATIO
            sparse_n = n // CTA_GROUP
            partial_n, partial_base = load_plan_count_and_base(partial_count, partial_offset, sparse_n, hq, batch)
            full_n, full_base = load_plan_count_and_base(full_count, full_offset, sparse_n, hq, batch)
            # instruction_selection: `ld.global.b32`; extent: scalar plan
            # counts, offsets, and compact indices. Varlen uses cu_k_blocks to
            # rebase the outer row.
            group_order = (full, partial) if CTA_GROUP == 2 else (partial, full)
            if CTA_GROUP == 2 and TD == 192:
                # D192 has two Q-family and two dO-family transactions for
                # every edge. pipeline_Qt aliases pipeline_Q, and sQt/sdOt
                # are descriptor aliases of sQ/sdO.
                iteration=0
                for group in group_order:
                    for i in range(group.count):
                        qblock=load_global(group.index,group.base+i)
                        if iteration == 0:
                            acquire(Q_BAR,Q_Qt_p,extra_tx=K_BYTES)
                            copy_g2s_tma(k_map,k_coords(sparse_n,hkv,batch),SK_BASE,
                                         full(Q_BAR,Q_Qt_p))
                            copy_g2s_tma(q_map,q_coords(qblock,hq,batch),SQ_BASE,
                                         full(Q_BAR,Q_Qt_p),multicast=q_mcast)
                            commit(Q_BAR,Q_Qt_p); advance(Q_Qt_p)

                            acquire(LSE_BAR,LSE_p)
                            copy_g2s_bulk(lse_ptr(qblock,hq,batch),SLSE_BASE,512,
                                          full(LSE_BAR,LSE_p)); advance(LSE_p)

                            acquire(DO_BAR,O_Ot_p,extra_tx=V_BYTES)
                            copy_g2s_tma(v_map,v_coords(sparse_n,hkv,batch),SV_BASE,
                                         full(DO_BAR,O_Ot_p))
                            copy_g2s_tma(dot_map,do_transpose_coords(qblock,hq,batch),
                                         SDO_BASE,full(DO_BAR,O_Ot_p),multicast=do_mcast)
                            commit(DO_BAR,O_Ot_p); advance(O_Ot_p)

                            acquire(DPSUM_BAR,dPsum_p)
                            copy_g2s_bulk(dpsum_ptr(qblock,hq,batch),SDPSUM_BASE,512,
                                          full(DPSUM_BAR,dPsum_p)); advance(dPsum_p)

                            acquire(Q_BAR,Q_Qt_p,extra_tx=K_BYTES)
                            copy_g2s_tma(qt_map,qt_coords(qblock,hq,batch),SQ_BASE,
                                         full(Q_BAR,Q_Qt_p),multicast=q_mcast)
                            copy_g2s_tma(kt_map,kt_coords(sparse_n,hkv,batch),SKT_BASE,
                                         full(Q_BAR,Q_Qt_p),multicast=k_mcast)
                            commit(Q_BAR,Q_Qt_p); advance(Q_Qt_p)

                            acquire(DO_BAR,O_Ot_p)
                            copy_g2s_tma(do_map,do_coords(qblock,hq,batch),SDO_BASE,
                                         full(DO_BAR,O_Ot_p),multicast=do_mcast)
                            commit(DO_BAR,O_Ot_p); advance(O_Ot_p)
                        else:
                            acquire(LSE_BAR,LSE_p)
                            copy_g2s_bulk(lse_ptr(qblock,hq,batch),SLSE_BASE,512,
                                          full(LSE_BAR,LSE_p)); advance(LSE_p)
                            acquire(Q_BAR,Q_Qt_p)
                            copy_g2s_tma(q_map,q_coords(qblock,hq,batch),SQ_BASE,
                                         full(Q_BAR,Q_Qt_p),multicast=q_mcast)
                            commit(Q_BAR,Q_Qt_p); advance(Q_Qt_p)
                            acquire(DPSUM_BAR,dPsum_p)
                            copy_g2s_bulk(dpsum_ptr(qblock,hq,batch),SDPSUM_BASE,512,
                                          full(DPSUM_BAR,dPsum_p)); advance(dPsum_p)
                            acquire(DO_BAR,O_Ot_p)
                            copy_g2s_tma(dot_map,do_transpose_coords(qblock,hq,batch),
                                         SDO_BASE,full(DO_BAR,O_Ot_p),multicast=do_mcast)
                            commit(DO_BAR,O_Ot_p); advance(O_Ot_p)
                            acquire(Q_BAR,Q_Qt_p)
                            copy_g2s_tma(qt_map,qt_coords(qblock,hq,batch),SQ_BASE,
                                         full(Q_BAR,Q_Qt_p),multicast=q_mcast)
                            commit(Q_BAR,Q_Qt_p); advance(Q_Qt_p)
                            acquire(DO_BAR,O_Ot_p)
                            copy_g2s_tma(do_map,do_coords(qblock,hq,batch),SDO_BASE,
                                         full(DO_BAR,O_Ot_p),multicast=do_mcast)
                            commit(DO_BAR,O_Ot_p); advance(O_Ot_p)
                        iteration += 1
                tail(Q_BAR,Q_Qt_p); tail(LSE_BAR,LSE_p)
                tail(DO_BAR,O_Ot_p); tail(DPSUM_BAR,dPsum_p)

            elif CTA_GROUP == 2:             # D128 delayed-Qt producer
                iteration=0; previous_q=0
                for group in group_order:
                    for i in range(group.count):
                        qblock=load_global(group.index,group.base+i)
                        if iteration:
                            acquire(QT_BAR,Qt_p)
                            copy_g2s_tma(qt_map,qt_coords(previous_q,hq,batch),SQT_BASE,
                                         full(QT_BAR,Qt_p),multicast=q_mcast)
                            commit(QT_BAR,Qt_p); advance(Qt_p)
                        acquire(Q_BAR,Q_LSE_p,extra_tx=(K_BYTES if not iteration else 0))
                        if not iteration:
                            copy_g2s_tma(k_map,k_coords(sparse_n,hkv,batch),SK_BASE,
                                         full(Q_BAR,Q_LSE_p))
                        copy_g2s_tma(q_map,q_coords(qblock,hq,batch),SQ_BASE,
                                     full(Q_BAR,Q_LSE_p),multicast=q_mcast)
                        commit(Q_BAR,Q_LSE_p)
                        acquire(LSE_BAR,Q_LSE_p)
                        copy_g2s_bulk(lse_ptr(qblock,hq,batch),SLSE_BASE,512,
                                      full(LSE_BAR,Q_LSE_p)); advance(Q_LSE_p)

                        acquire(DO_BAR,dO_dPsum_p,
                                extra_tx=(V_BYTES+(DO_BYTES if not iteration else 0)))
                        if not iteration:
                            copy_g2s_tma(v_map,v_coords(sparse_n,hkv,batch),SV_BASE,
                                         full(DO_BAR,dO_dPsum_p))
                        copy_g2s_tma(do_map,do_coords(qblock,hq,batch),SDO_BASE,
                                     full(DO_BAR,dO_dPsum_p),multicast=do_mcast)
                        copy_g2s_tma(dot_map,do_transpose_coords(qblock,hq,batch),SDOT_BASE,
                                     full(DO_BAR,dO_dPsum_p),multicast=do_mcast)
                        commit(DO_BAR,dO_dPsum_p)
                        acquire(DPSUM_BAR,dO_dPsum_p)
                        copy_g2s_bulk(dpsum_ptr(qblock,hq,batch),SDPSUM_BASE,512,
                                      full(DPSUM_BAR,dO_dPsum_p)); advance(dO_dPsum_p)
                        if not iteration:
                            acquire(KT_BAR,Kt_p)
                            copy_g2s_tma(kt_map,kt_coords(sparse_n,hkv,batch),SKT_BASE,
                                         full(KT_BAR,Kt_p),multicast=k_mcast)
                            commit(KT_BAR,Kt_p); advance(Kt_p)
                        previous_q=qblock; iteration += 1
                if iteration:
                    acquire(QT_BAR,Qt_p)
                    copy_g2s_tma(qt_map,qt_coords(previous_q,hq,batch),SQT_BASE,
                                 full(QT_BAR,Qt_p),multicast=q_mcast)
                    commit(QT_BAR,Qt_p); advance(Qt_p)
                tail(Q_BAR,Q_LSE_p.clone()); tail(LSE_BAR,Q_LSE_p)
                tail(QT_BAR,Qt_p); tail(DO_BAR,dO_dPsum_p.clone())
                tail(DPSUM_BAR,dO_dPsum_p)

            else:                            # 1CTA partial-first producer
                iteration=0
                for group in group_order:
                    for i in range(group.count):
                        qblock=load_global(group.index,group.base+i)
                        acquire(Q_BAR,Q_LSE_p,extra_tx=(K_BYTES if not iteration else 0))
                        if not iteration:
                            copy_g2s_tma(k_map,k_coords(sparse_n,hkv,batch),SK_BASE,
                                         full(Q_BAR,Q_LSE_p))
                        copy_g2s_tma(q_map,q_coords(qblock,hq,batch),
                                     SQ_BASE+Q_LSE_p.stage*SQ_STAGE_BYTES,
                                     full(Q_BAR,Q_LSE_p))
                        commit(Q_BAR,Q_LSE_p)
                        acquire(LSE_BAR,Q_LSE_p)
                        copy_g2s_bulk(lse_ptr(qblock,hq,batch),
                                      SLSE_BASE+Q_LSE_p.stage*512,512,
                                      full(LSE_BAR,Q_LSE_p)); advance(Q_LSE_p)
                        acquire(DO_BAR,dO_dPsum_p,extra_tx=(V_BYTES if not iteration else 0))
                        if not iteration:
                            copy_g2s_tma(v_map,v_coords(sparse_n,hkv,batch),SV_BASE,
                                         full(DO_BAR,dO_dPsum_p))
                        copy_g2s_tma(do_map,do_coords(qblock,hq,batch),SDO_BASE,
                                     full(DO_BAR,dO_dPsum_p))
                        commit(DO_BAR,dO_dPsum_p)
                        acquire(DPSUM_BAR,dO_dPsum_p)
                        copy_g2s_bulk(dpsum_ptr(qblock,hq,batch),SDPSUM_BASE,512,
                                      full(DPSUM_BAR,dO_dPsum_p)); advance(dO_dPsum_p)
                        iteration += 1
                tail(Q_BAR,Q_LSE_p.clone()); tail(LSE_BAR,Q_LSE_p)
                tail(DO_BAR,dO_dPsum_p.clone()); tail(DPSUM_BAR,dO_dPsum_p)

            # instruction_selection: rank-3/4
            # `cp.async.bulk.tensor.*d.shared::cluster.global.tile` and
            # 512-byte `cp.async.bulk.shared::cluster.global...`; extent and
            # state advances are exactly the branch bodies above. Every tail
            # is a parity-wait retry loop for the final live producer state.

    # ====================================================================
    # Warp 12: all five tcgen05 GEMM families
    # ====================================================================
    if warp == 12:
        tmem_alloc(TMEM_WORD, 512, CTA_GROUP)
        # instruction_selection:
        # `tcgen05.alloc.cta_group::{1|2}.sync.aligned.shared::cta.b32`;
        # extent: one 512-column allocation.
        named_or_unaligned_tmem_barrier()
        # instruction_selection: `bar.sync` for 1CTA or source unaligned
        # cluster-aware mbarrier protocol for 2CTA; extent: one allocation
        # publication epoch.

        for work in scheduler_items():
            iters=k2q_partial_count(work)+k2q_full_count(work)
            process_tile=iters>0
            # Both CTA warp-12 roles participate in allocation and teardown,
            # but only rank 0 executes or publishes cooperative MMA work.
            execute_work=process_tile and (CTA_GROUP == 1 or leader_cta)
            if not execute_work: continue

            if CTA_GROUP == 2 and TD == 192:
                accumulate_dK=False; accumulate_dV=False
                for i in range(iters):
                    wait(Q_BAR,Q_c); wait(empty(DQ_BAR,0),producer_phase_acc)
                    gemm(tS,desc(sK,sK_byte),desc(sQ,sQ_byte),ID_S,False)
                    arrive_tcgen(full(SP_BAR,0)); release(Q_BAR,Q_c); advance(Q_c)
                    producer_phase_acc ^= 1

                    wait(DO_BAR,dO_c); wait(empty(SP_BAR,0),producer_phase_acc)
                    gemm(tdP,desc(sV,sV_byte),desc(sdOt,sdOt_byte),ID_DP,False)
                    arrive_tcgen(full(DP_BAR,0)); release(DO_BAR,dO_c); advance(dO_c)

                    wait(Q_BAR,Q_c); wait(empty(DP_BAR,0),producer_phase_acc)
                    gemm(tdK,tdS,desc(sQt,sQt_native_byte),ID_DK,accumulate_dK)
                    release(Q_BAR,Q_c); advance(Q_c); accumulate_dK=True

                    wait(DO_BAR,dO_c)
                    gemm(tdV,tP,desc(sdO,sdO_byte),ID_DV,accumulate_dV)
                    release(DO_BAR,dO_c); advance(dO_c); accumulate_dV=True

                    wait(DS_BAR,dS_c); wait(DS_LEADER_C,dS_cluster_phase)
                    gemm(tdQ,desc(sdS,sdS_byte),desc(sKt,sKt_native_byte),ID_DQ,False)
                    arrive_tcgen(full(DQ_BAR,0)); release(DS_BAR,dS_c); advance(dS_c)
                    dS_cluster_phase ^= 1
                wait(empty(DKV_BAR,0),producer_phase_dKV); arrive_tcgen(full(DKV_BAR,0))
                wait(empty(DKV_BAR,1),producer_phase_dKV); arrive_tcgen(full(DKV_BAR,1))
                producer_phase_dKV ^= 1

            elif CTA_GROUP == 2:             # D128
                accumulate_dK=False
                wait(Q_BAR,Q_c); wait(empty(SP_BAR,0),producer_phase_acc)
                gemm(tS,desc(sK,sK_byte),desc(sQ,sQ_byte),ID_S,False)
                arrive_tcgen(full(SP_BAR,0)); release(Q_BAR,Q_c); advance(Q_c)
                wait(DO_BAR,dO_c); wait(empty(DP_BAR,0),producer_phase_acc)
                gemm(tdP,desc(sV,sV_byte),desc(sdOt,sdOt_byte),ID_DP,False)
                arrive_tcgen(full(DP_BAR,0))
                producer_phase_acc ^= 1
                wait(empty(SP_BAR,0),producer_phase_acc)
                gemm(tdV,tP,desc(sdO,sdO_byte),ID_DV,False)
                release(DO_BAR,dO_c); advance(dO_c); wait(KT_BAR,Kt_c)

                for i in range(iters-1):
                    wait(Q_BAR,Q_c); wait(empty(DQ_BAR,0),producer_phase_dQ)
                    gemm(tS,desc(sK,sK_byte),desc(sQ,sQ_byte),ID_S,False)
                    arrive_tcgen(full(SP_BAR,0)); release(Q_BAR,Q_c); advance(Q_c)
                    wait(QT_BAR,Qt_c); wait(empty(DP_BAR,0),producer_phase_acc)
                    gemm(tdK,tdS,desc(sQt,sQt_byte,Qt_c.stage),ID_DK,accumulate_dK)
                    release(QT_BAR,Qt_c); advance(Qt_c); accumulate_dK=True
                    wait(DO_BAR,dO_c)
                    gemm(tdP,desc(sV,sV_byte),desc(sdOt,sdOt_byte),ID_DP,False)
                    arrive_tcgen(full(DP_BAR,0))
                    wait(DS_BAR,dS_c); wait(DS_LEADER_C,dS_cluster_phase)
                    gemm(tdQ,desc(sdS,sdS_byte),desc(sKt,sKt_byte),ID_DQ,False)
                    arrive_tcgen(full(DQ_BAR,0)); release(DS_BAR,dS_c); advance(dS_c)
                    dS_cluster_phase ^= 1; producer_phase_dQ ^= 1
                    producer_phase_acc ^= 1; wait(empty(SP_BAR,0),producer_phase_acc)
                    gemm(tdV,tP,desc(sdO,sdO_byte),ID_DV,True)
                    release(DO_BAR,dO_c); advance(dO_c)

                arrive_tcgen(full(SP_BAR,0))
                wait(empty(DKV_BAR,0),producer_phase_dKV); arrive_tcgen(full(DKV_BAR,0))
                wait(empty(DKV_BAR,1),producer_phase_dKV)
                wait(QT_BAR,Qt_c); wait(empty(DP_BAR,0),producer_phase_acc)
                gemm(tdK,tdS,desc(sQt,sQt_byte,Qt_c.stage),ID_DK,accumulate_dK)
                release(QT_BAR,Qt_c); advance(Qt_c); arrive_tcgen(full(DKV_BAR,1))
                producer_phase_dKV ^= 1
                wait(DS_BAR,dS_c); wait(DS_LEADER_C,dS_cluster_phase)
                wait(empty(DQ_BAR,0),producer_phase_dQ)
                gemm(tdQ,desc(sdS,sdS_byte),desc(sKt,sKt_byte),ID_DQ,False)
                arrive_tcgen(full(DQ_BAR,0)); release(DS_BAR,dS_c); advance(dS_c)
                release(KT_BAR,Kt_c); advance(Kt_c)
                dS_cluster_phase ^= 1; producer_phase_dQ ^= 1; producer_phase_acc ^= 1

            else:                            # 1CTA
                accumulate_dK=False
                q0=wait_and_advance(Q_BAR,Q_c)
                wait(empty(SP_BAR,0),producer_phase_acc)
                gemm(tS,desc(sK,sK_byte),desc(sQ,sQ_byte,q0.stage),ID_S,False)
                arrive_tcgen(full(SP_BAR,0))
                wait(DO_BAR,dO_c); wait(empty(DP_BAR,0),producer_phase_acc)
                wait(empty(DQ_BAR,0),producer_phase_acc)
                gemm(tdP,desc(sV,sV_byte),desc(sdOt,sdOt_byte,dO_c.stage),ID_DP,False)
                arrive_tcgen(full(DP_BAR,0)); producer_phase_acc ^= 1
                wait(empty(SP_BAR,0),producer_phase_acc)
                gemm(tdV,tP,desc(sdO,sdO_byte,dO_c.stage),ID_DV,False)
                release(DO_BAR,dO_c); advance(dO_c)
                for i in range(iters-1):
                    qnext=wait_and_advance(Q_BAR,Q_c)
                    gemm(tS,desc(sK,sK_byte),desc(sQ,sQ_byte,qnext.stage),ID_S,False)
                    arrive_tcgen(full(SP_BAR,0)); wait(DS_BAR,dS_c)
                    gemm(tdK,tdS,desc(sQt,sQt_byte,q0.stage),ID_DK,accumulate_dK)
                    q0.release(); accumulate_dK=True
                    gemm(tdQ,desc(sdS,sdS_byte),desc(sKt,sKt_byte),ID_DQ,False)
                    arrive_tcgen(full(DQ_BAR,0)); release(DS_BAR,dS_c); advance(dS_c)
                    wait(DO_BAR,dO_c); wait(empty(DQ_BAR,0),producer_phase_acc)
                    gemm(tdP,desc(sV,sV_byte),desc(sdOt,sdOt_byte,dO_c.stage),ID_DP,False)
                    arrive_tcgen(full(DP_BAR,0)); producer_phase_acc ^= 1
                    wait(empty(SP_BAR,0),producer_phase_acc)
                    gemm(tdV,tP,desc(sdO,sdO_byte,dO_c.stage),ID_DV,True)
                    release(DO_BAR,dO_c); advance(dO_c); q0=qnext
                arrive_tcgen(full(SP_BAR,0))
                wait(empty(DKV_BAR,0),producer_phase_dKV); arrive_tcgen(full(DKV_BAR,0))
                wait(empty(DKV_BAR,1),producer_phase_dKV); wait(DS_BAR,dS_c)
                gemm(tdK,tdS,desc(sQt,sQt_byte,q0.stage),ID_DK,accumulate_dK)
                arrive_tcgen(full(DKV_BAR,1)); producer_phase_dKV ^= 1
                gemm(tdQ,desc(sdS,sdS_byte),desc(sKt,sKt_byte),ID_DQ,False)
                arrive_tcgen(full(DQ_BAR,0)); q0.release()
                release(DS_BAR,dS_c); advance(dS_c); producer_phase_acc ^= 1

            # instruction_selection: every gemm above expands to
            # `tcgen05.mma.cta_group::{1|2}.kind::f16`; S/dP/dQ/dK/dV
            # publications use matching tcgen05 mbarrier commits. The branch
            # explicitly advances each Q/Qt/Kt/dO/dS consumer and each raw
            # accumulator/dQ/dKV phase at its source edge.

        tmem_relinquish(CTA_GROUP)
        # instruction_selection:
        # `tcgen05.relinquish_alloc_permit.cta_group::{1|2}.sync.aligned`;
        # extent: one teardown issue.
        named_or_unaligned_tmem_barrier()
        tmem_free(tmem_base,512,CTA_GROUP)
        # instruction_selection:
        # `tcgen05.dealloc.cta_group::{1|2}.sync.aligned.b32`; extent: one
        # 512-column deallocation.

    # ====================================================================
    # Warps 4..11: P/dS computation and dK/dV output
    # ====================================================================
    if 4 <= warp <= 11:
        wait_for_tmem_allocation()
        compute_tid = thread_id() % 256
        compute_group = compute_tid // 128
        consumer_group = compute_tid
        for work in scheduler_items():
            partial_n, partial_base = plan_partial_row(work)
            full_n, full_base = plan_full_row(work)
            group_order = (full,partial) if CTA_GROUP == 2 else (partial,full)
            iter_index=0
            for group in group_order:
                for gi in range(group.count):
                    wait(LSE_BAR,LSE_c); wait(SP_BAR,SP_dP_c)
                    copy_t2r(tS_for(consumer_group), scores[0:64], repetition=32)
                    # instruction_selection:
                    # `tcgen05.ld.sync.aligned.32x32b.x32.b32`; extent: score
                    # fragment for one compute thread and sparse iteration.
                    if TD == 192:
                        fence_tmem_load(); elect_release(SP_BAR,SP_dP_c)
                        # instruction_selection:
                        # `tcgen05.wait::ld.sync.aligned` plus async TMEM-load
                        # fence and elected mbarrier arrival; extent: one S
                        # read-complete publication.
                    elif CTA_GROUP == 2 and iter_index > 0:
                        fence_tmem_load(); elect_commit(DS_BAR,dS_p); advance(dS_p)
                        # instruction_selection: TMEM wait/fence and elected
                        # mbarrier arrival; extent: delayed dS/S-alias release.

                    if group is partial:
                        w0=load_global(packed_mask, packed_mask_word(partial_base+gi,cta_rank,consumer_group,0))
                        w1=load_global(packed_mask, packed_mask_word(partial_base+gi,cta_rank,consumer_group,1))
                        # instruction_selection: one `ld.global.v2.b32`; extent:
                        # two native packed mask words per compute thread.
                        for bit in range(64):
                            scores[bit]=select((w[bit//32]>>(bit%32))&1,
                                               scores[bit], -inf)
                            # instruction_selection: integer shift/test plus
                            # `selp.f32`; extent: one score lane.

                    for score_stage in range(2):
                        lse_pair=load_shared(
                            SLSE_BASE+LSE_c.stage*512+lse_byte(consumer_group,score_stage))
                        # instruction_selection: `ld.shared.v2.b32`; extent:
                        # one two-row LSE-log2 pair for this score stage.
                        for pair in range(32):
                            scores[score_stage,2*pair:2*pair+2]=fma(
                                scores[score_stage,pair],(scale_log2,scale_log2),
                                (-lse_pair0,-lse_pair1),lanes=2)
                            # instruction_selection: `fma.rn.f32x2`; extent:
                            # 32 pairs per stage, 64 static instructions total.
                            exp2(scores[score_stage,2*pair],scores[score_stage,2*pair])
                            exp2(scores[score_stage,2*pair+1],scores[score_stage,2*pair+1])
                            # instruction_selection: `ex2.approx.ftz.f32`;
                            # extent: 64 scalars per stage, 128 static total.
                        cast(packed_p[score_stage],scores[score_stage])
                        # instruction_selection: `cvt.rn.{f16|bf16}x2.f32`;
                        # extent: 32 packed conversions per stage.
                        if score_stage == 0:
                            fence_tmem_load(); named_barrier(3,256)
                            # instruction_selection: `tcgen05.wait::ld` then
                            # `bar.sync 3,256`; extent: the first-stage
                            # overwrite-protection rendezvous.
                        copy_r2t(packed_p[score_stage],
                                 tP_for(consumer_group,score_stage),repetition=16)
                        # instruction_selection:
                        # `tcgen05.st.sync.aligned.32x32b.x16.b32`; extent:
                        # one P fragment for each of two stages.
                    fence_tmem_store(); fence_shared_async(); named_barrier(3,256)
                    # instruction_selection: `tcgen05.wait::st.sync.aligned`,
                    # async-shared fence, `bar.sync 3,256`; extent: P publish.
                    if TD != 192: elect_release(SP_BAR,SP_dP_c)
                    release(LSE_BAR,LSE_c); advance(LSE_c)

                    wait(DPSUM_BAR,dPsum_c); wait(DP_BAR,SP_dP_c)
                    for ds_stage in range(2):
                        copy_t2r(tdP_for(consumer_group,ds_stage),
                                 dp[ds_stage,0:64],repetition=32)
                        # instruction_selection:
                        # `tcgen05.ld.sync.aligned.32x32b.x32.b32`; extent:
                        # one dP fragment for each of two stages.
                        fence_tmem_load(); named_barrier(3,256)
                        # instruction_selection: TMEM wait/fence then
                        # `bar.sync 3,256`; extent: once per stage, protecting
                        # the in-place dP/dS alias.
                        dsum_pair=load_shared(
                            SDPSUM_BASE+dpsum_byte(consumer_group,ds_stage))
                        # instruction_selection: `ld.shared.v2.b32`; extent:
                        # one two-row dPsum pair for this stage.
                        for pair in range(32):
                            dp_pair=sub(dp[ds_stage,pair],dsum_pair,lanes=2)
                            # instruction_selection: `sub.rn.f32x2`; extent:
                            # 32 pairs per stage, 64 static total.
                            ds_pair=mul(scores[ds_stage,pair],dp_pair,lanes=2)
                            # instruction_selection: `mul.rn.f32x2`; extent:
                            # 32 pairs per stage, 64 static total.
                        cast(packed_ds[ds_stage],ds_pair)
                        # instruction_selection: `cvt.rn.{f16|bf16}x2.f32`;
                        # extent: 32 packed conversions per stage.
                        if ds_stage == 0:
                            acquire(DS_BAR,dS_p)
                            if CTA_GROUP == 2: allocate(exchange_regs)
                        copy_r2t(packed_ds[ds_stage],
                                 tdS_for(consumer_group,ds_stage),repetition=16)
                        # instruction_selection:
                        # `tcgen05.st.sync.aligned.32x32b.x16.b32`; extent:
                        # one dS fragment for each of two stages.
                        if CTA_GROUP == 2 and ds_stage == exchange_stage:
                            copy_regs(packed_ds[ds_stage],exchange_regs)
                        else:
                            copy_r2s(packed_ds[ds_stage],
                                     sdS_byte(consumer_group,ds_stage),16)
                            # instruction_selection: `st.shared.v4.b32`;
                            # extent: the complete non-exchange stage, or both
                            # stages in 1CTA.
                    fence_tmem_store()
                    # instruction_selection: `tcgen05.wait::st.sync.aligned`;
                    # extent: dS TMEM publication.
                    if CTA_GROUP == 2: elect_release(DP_BAR,SP_dP_c)
                    advance(SP_dP_c)
                    if CTA_GROUP == 2 and TD == 192:
                        wait(DQ_EMPTY_C,dS_p.phase)
                        # instruction_selection: mbarrier parity-wait retry
                        # loop; extent: protect sdQacc/sdSx alias.
                    if CTA_GROUP == 2:
                        copy_r2s(exchange_regs,SDSX_BASE+exchange_byte,16)
                        # instruction_selection: `st.shared.v4.b32`; extent:
                        # loop of exchange-half stores.
                    fence_shared_async(); named_barrier(3,256)
                    # instruction_selection: async-shared fence then
                    # `bar.sync 3,256`; extent: compute-WG publication.
                    release(DPSUM_BAR,dPsum_c); advance(dPsum_c)
                    if not (CTA_GROUP == 2 and TD == 128):
                        elect_commit(DS_BAR,dS_p); advance(dS_p)
                        # instruction_selection: elected
                        # `mbarrier.arrive.shared.b64`; extent: one dS publish.
                    if CTA_GROUP == 2 and compute_tid == 0:
                        peer=cta_rank^1
                        expect_tx(peer.DS_FULL_C,SDS_EXCHANGE_BYTES)
                        copy_s2cluster(SDSX_BASE,peer.SDS_BASE+cta_rank*SDS_EXCHANGE_BYTES,
                                        SDS_EXCHANGE_BYTES,peer,peer.DS_FULL_C)
                        # instruction_selection:
                        # `mbarrier.arrive.expect_tx.shared::cluster.b64`,
                        # `mapa.shared::cluster.u32`, and
                        # `cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes`;
                        # extent: one half-dS DSMEM copy.
                    iter_index += 1
            if CTA_GROUP == 2 and TD == 128 and iter_index:
                elect_commit(DS_BAR,dS_p); advance(dS_p)
                # instruction_selection: elected mbarrier arrival; extent:
                # final delayed dS publication.

            if iter_index and DIRECT_RAW_DKV:
                # Exact-dimension varlen MHA bypasses SMEM and TensorMaps.
                # Each 128-thread compute WG handles its source partition.
                wait(DKV_BAR,dKV_c)                       # dV stage
                copy_t2r(tdV,dv_regs,repetition=16)
                fence_tmem_load(); cast(dv_narrow,dv_regs)
                store_global(dv_out,predicated_dv_coords,dv_narrow,width=16)
                # instruction_selection: `tcgen05.ld...x16.b32`,
                # `cvt.rn.{f16|bf16}x2.f32`, predicated `st.global.v4.b32`;
                # extent: the complete WG-owned dV fragment.
                warp_sync(); elect_release(DKV_BAR,dKV_c); advance(dKV_c)

                wait(DKV_BAR,dKV_c)                       # dK stage
                copy_t2r(tdK,dk_regs,repetition=16); fence_tmem_load()
                mul(dk_regs,dk_regs,softmax_scale,lanes=2)
                # instruction_selection: exact opcode `mul.f32x2` (without
                # `.rn`) over every packed dK pair.
                cast(dk_narrow,dk_regs)
                store_global(dk_out,predicated_dk_coords,dk_narrow,width=16)
                # instruction_selection: predicated `st.global.v4.b32`;
                # extent: the complete WG-owned dK fragment.
                warp_sync(); elect_release(DKV_BAR,dKV_c)
                # Source returns this state without advancing after dK.

            elif iter_index:
                # Fixed exact MHA uses narrow SMEM -> TensorMap stores.
                # GQA or padded dimensions use FP32 SMEM -> bulk reductions.
                for which in (dv,dk):
                    stages=EPI_V_STAGES if which is dv else EPI_K_STAGES
                    ncol=DV_NCOL if which is dv else DK_NCOL
                    base=SV_BASE if CTA_GROUP==2 and which is dv else (
                         SK_BASE if CTA_GROUP==2 else (SDO_BASE if which is dv else SQ_BASE))
                    wait(DKV_BAR,dKV_c)
                    det_kv=DETERMINISTIC and DKV_POSTPROCESS and HQRATIO>1
                    if det_kv:
                        wait_global_sem(which_sem,head_rank)
                        named_barrier(epi_bar+wg,128)
                        # One acquire protects the complete output, not a stage.
                    for epi_stage in range(stages):
                        copy_t2r(which_tmem_stage(epi_stage),epi_regs,repetition=ncol)
                        fence_tmem_load()
                        if which is dk and not DKV_POSTPROCESS:
                            mul(epi_regs,epi_regs,softmax_scale,lanes=2)
                            # instruction_selection: `mul.rn.f32x2` for the
                            # staged fixed-MHA dK path.
                        cast(epi_value,epi_regs)
                        copy_r2s(epi_value,dkv_stage_byte(base,epi_stage,wg),16)
                        # instruction_selection:
                        # `tcgen05.ld.sync.aligned.32x32b.x{32|16|8}.b32`
                        # followed by `st.shared.v4.b32`; extent: one complete
                        # EPI_{K,V}_COLS slice per loop iteration.
                        fence_proxy_shared(); named_barrier(epi_bar+wg,128)
                        if DKV_POSTPROCESS:
                            reduce_s2g_add_f32(which_accum_stage(epi_stage),
                                               dkv_stage_byte(base,epi_stage,wg),
                                               128*ncol*4)
                            # instruction_selection:
                            # `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32`.
                        else:
                            copy_s2g_tma(which_map,kv_stage_coords(epi_stage),
                                         dkv_stage_byte(base,epi_stage,wg))
                            # instruction_selection:
                            # `cp.async.bulk.tensor.*d.global.shared::cta.tile.bulk_group`.
                        if epi_stage < stages-1:
                            tma_store_commit(); tma_store_wait(0,read=not det_kv)
                        leader_arrive(epi_bar+wg,160)
                        fence_proxy_shared(); named_barrier(epi_bar+wg,160)
                    if det_kv:
                        leader(tma_store_commit(); tma_store_wait(0,read=False))
                        named_barrier(epi_bar+wg,128)
                        increment_global_sem(which_sem)
                        # One release follows the complete staged output.
                    warp_sync(); elect_release(DKV_BAR,dKV_c); advance(dKV_c)

            if not DKV_POSTPROCESS and not iter_index and leader_cta:
                fill(zero,0); store_global(dk_out,empty_tile,zero,width=16)
                # instruction_selection: `st.global.v4.b32`; extent: vector
                # loop over live D elements of an empty K tile.
                store_global(dv_out,empty_tile,zero,width=16)
                # instruction_selection: `st.global.v4.b32`; extent: vector
                # loop over live DV elements.
            if DETERMINISTIC and DKV_POSTPROCESS and HQRATIO>1 and not iter_index:
                pass_zero_tokens_in_source_order()
                # instruction_selection: global semaphore wait/increment
                # loops; extent: dV then dK, both compute workgroups.
        arrive_tmem_teardown()
        # instruction_selection: named/unaligned barrier arrival; extent: one
        # per compute warp in the TMEM lifetime group.

    # ====================================================================
    # Warps 0..3: dQ drain and source-order reduction
    # ====================================================================
    if 0 <= warp <= 3:
        wait_for_tmem_allocation()
        reduce_tid=thread_id()%128
        for work in scheduler_items():
            partial_n, partial_base = plan_partial_row(work)
            full_n, full_base = plan_full_row(work)
            group_order=(full,partial) if CTA_GROUP==2 else (partial,full)
            for group in group_order:
                for gi in range(group.count):
                    qblock=load_global(group.index,group.base+gi)
                    # instruction_selection: `ld.global.b32`; extent: one Q
                    # block index per dQ contribution.
                    wait(DQ_BAR,dQ_c)
                    # instruction_selection: mbarrier parity-wait retry loop.
                    copy_t2r(tdQ_for(reduce_tid),dq_regs,repetition=dq_t2r_rep)
                    # instruction_selection:
                    # `tcgen05.ld.sync.aligned.32x32b.x{32|16|8}.b32`;
                    # extent: TD/CTA_GROUP columns in dimension-selected chunks.
                    fence_tmem_load(); warp_sync()
                    elect_release(DQ_BAR,dQ_c); advance(dQ_c)
                    # instruction_selection: `tcgen05.wait::ld.sync.aligned`
                    # then `bar.warp.sync -1` and elected mbarrier arrival;
                    # extent: one dQ generation.
                    for stage in range(dq_column_slice_count):
                        smem_stage=dQ_smem_p.stage
                        copy_r2s(dq_regs[stage],sdQacc_byte(0,0,smem_stage),16)
                        # instruction_selection: `st.shared.v4.b32`; extent:
                        # vector loop over one DQ_NCOL slice.
                        fence_proxy_shared(); named_barrier(4,128)
                        # instruction_selection: async proxy fence then
                        # `bar.sync 4,128`; extent: reduce-WG publication.
                        if DETERMINISTIC and stage == 0 and not qblock_oob:
                            write_rank=load_global(group.dq_order,group.base+gi)
                            # instruction_selection: deterministic-only
                            # `ld.global.b32`; extent: one source-planned rank.
                            wait_global_sem(dq_sem,write_rank)
                            # instruction_selection: global acquire polling
                            # loop once at stage zero for this contribution.
                        reduce_s2g_add_f32(dq_accum_ptr(qblock,stage),
                                           sdQacc_byte(0,0,smem_stage),128*DQ_NCOL*4)
                        # instruction_selection:
                        # `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32`;
                        # extent: one DQ_NCOL slice.
                        tma_store_commit(); tma_store_wait(pipeline_limit)
                        # instruction_selection: `cp.async.bulk.commit_group`
                        # and `cp.async.bulk.wait_group.read`; extent: each
                        # slice, then final limit zero.
                        named_barrier(4,128)
                        # instruction_selection: `bar.sync 4,128`; extent:
                        # protect staged-buffer reuse.
                        advance(dQ_smem_p)

                    if CTA_GROUP==2 and TD==192:
                        if DQ_SMEM_STAGES > 1:
                            elected_tma_warp(tma_store_wait(0,read=not DETERMINISTIC))
                            named_barrier(4,128)
                        elect_arrive(DQ_EMPTY_C,0)
                        # instruction_selection: elected
                        # `mbarrier.arrive.shared.b64` only after every slice,
                        # final bulk drain, and reduce-WG barrier; extent: one
                        # safe release of the sdQacc/sdSx alias epoch.
                    if DETERMINISTIC:
                        if DQ_SMEM_STAGES > 1 and TD != 192:
                            elected_tma_warp(tma_store_wait(0,read=False))
                            named_barrier(4,128)
                        if not qblock_oob: increment_global_sem(dq_sem)
                        # instruction_selection: one release atomic after all
                        # slices and required pending-copy waits.
            if partial_n+full_n:
                elected_tma_warp(tma_store_wait(0,read=not DETERMINISTIC))
                named_barrier(4,128)
        if not DETERMINISTIC: tma_store_wait(0,read=True)
        arrive_tmem_teardown()
        # instruction_selection: final `cp.async.bulk.wait_group.read 0`,
        # reduce-WG barrier, and named/unaligned TMEM barrier arrival; extent:
        # the source's per-work drain and reduce-role teardown.
```

## Logical GEMMs and owners

| result | logical product | owner | accumulator | operand forms |
| --- | --- | --- | --- | --- |
| S.T | `K[CTA_GROUP*128,TD] @ Q[128,TD].T` | warp 12 | FP32 TMEM at `TS` | SMEM/SMEM, K/K major |
| dP.T | `V[CTA_GROUP*128,TDV] @ dO[128,TDV].T` | warp 12 | FP32 TMEM at `TDP` | SMEM/SMEM, K/K major |
| dV | `P.T[CTA_GROUP*128,128] @ dO[128,TDV]` | warp 12 | FP32 TMEM at `TV` | TMEM/SMEM, K/MN major |
| dK | `dS.T[CTA_GROUP*128,128] @ Q[128,TD]` | warp 12 | FP32 TMEM at `TK` | TMEM/SMEM, K/MN major |
| dQ | `dS[128,CTA_GROUP*128] @ K[CTA_GROUP*128,TD]` | warp 12 | FP32 TMEM at `TDQ` | SMEM/SMEM, MN/MN major |

All five select `tcgen05.mma.cta_group::{1|2}.kind::f16`. Dtype changes the
narrow conversion opcodes, not the MMA kind. The instruction descriptor and
matrix-descriptor base bits are compile-time scalar integers; only the 14-bit
shared address and the accumulate predicate vary at runtime.

## Storage aliases and lifetimes

| bytes / columns | first lifetime | later alias | ordering edge |
| --- | --- | --- | --- |
| `sQ` | Q stages consumed by score/dK | FP32 `sdK` staging in 1CTA | dK begins only after all Q consumption |
| `sdO` | dO stages consumed by dP/dV | FP32 `sdV` staging in 1CTA | dV publication follows final dO release |
| `sK` | resident K | FP32 `sdK` staging in 2CTA | dQ consumes resident K/Kt before dK output |
| `sV` | resident V | FP32 `sdV` staging in 2CTA | final dP/dV MMA precedes dV output |
| `sdQacc` D192 | `sdSx` peer-exchange source | dQ reduction stage | `DQ_EMPTY_C` releases only after the dQ drain, before the next exchange |
| TMEM `TS==TP` | score FP32 | narrow P | compute-WG read fence/barrier then R2T store |
| TMEM `TDP==TDS` | dP FP32 | narrow dS | dP read fence/barrier then R2T store |
| topology-dependent overlaps | S or dP columns | dQ/dK/dV accumulators | explicit empty/full pipeline phases in MMA order |

## Packed-plan and TensorMap ABI

The plan is twelve runtime tensor slots in source tuple order:
`mask_block_cnt`, `mask_block_idx`, `full_block_cnt`, `full_block_idx`,
`cu_total_m_blocks`, `dq_write_order`, `dq_write_order_full`,
`mask_block_offset`, `full_block_offset`, `mask_block_masks`, `sequence_desc`,
and `fwd_work_desc`. Optional pointers are omitted from specializations exactly
as in the source. `mask_block_masks` is uint32 `[partial_edge, physical_subtile,
256 compute threads, 2 words]` in the native TMEM-load value order.

Q/K/V/dO TensorMaps describe fixed rank-4 or varlen rank-3 input tensors after
source transposition. Both D128 and D192 cooperative specializations construct
Qt, Kt, and dOt TensorMaps: D128 targets separate sQt/sdOt fields, while D192
targets the aliased sQ/sdO descriptor views. LSE-log2 and dPsum are raw aligned
global pointers copied in 512-byte bulk transactions. Exact fixed-shape MHA
carries dK/dV output TensorMaps; exact varlen MHA carries raw output tensors and
uses predicated register-to-global stores with no output TensorMaps. Padded or
GQA specializations carry flat FP32 accumulation pointers. All TensorMap
stride/box/swizzle fields are encoded on the host and remain opaque by-value
kernel arguments.

## Correctness and benchmark contract

- The public module exposes all 257 supported dimension pairs plus 32
  interaction cases: 289 correctness configs total. It uses raw direct-main
  source/TIRx comparisons, small FP64 analytic oracles, input immutability,
  output redzones, deterministic exact checks where source-stable, and a
  five-run source envelope for nondeterministic FP32 reductions.
- Comparisons use `rtol=0`; absolute tolerances are derived from source repeat
  spread plus the smallest justified dtype/accumulation floor. NaN/Inf and
  silent all-zero outputs fail independently.
- The benchmark contains 48 cases. Planning, forward/LSE, dPsum preprocess,
  zeroing, semaphore reset/ring selection, and postprocess are outside the
  measurement. The Proton scope contains exactly one direct main launch.
- Deterministic timing uses a preallocated 16,384-slot semaphore/workspace ring,
  warmup zero, one-millisecond repeat windows, and hard failure before reuse.
- `bench_suite` is the sole final performance authority. Every row must satisfy
  `source_time / tirx_time > 0.99`, with both sides compiled as PTX 9.2 on GB200.

## Instruction-selection summary

- 1CTA/2CTA topology selects every `tcgen05` MMA, commit, alloc, relinquish,
  dealloc, TMA address space, cluster barrier, and dS exchange form.
- Linear SMEM offsets plus integer swizzle functions select matrix descriptor
  base addresses; no layout object exists in the sketch or target program.
- Q/K/V/dO and cooperative Qt/Kt/dOt are TensorMap bulk loads; LSE/dPsum are
  raw bulk copies; P/dS are TMEM stores; dQ and accumulation dK/dV are bulk
  FP32 reductions. Fixed direct dK/dV are TensorMap stores, while varlen direct
  dK/dV are predicated register-to-global vector stores.
- The compute body is a fixed chain of 32-word TMEM loads, packed FP32x2 FMA,
  FTZ exp2, packed subtract/multiply, narrow conversions, and 16-word TMEM
  stores. The exported absence of `stmatrix` and global atomic instructions is
  intentional.
- Pipeline phases, partial/full group order, one-iteration-delayed Qt load,
  cooperative dS exchange, D192 alias barrier, and deterministic semaphore
  ranks are semantic schedule constraints, not optional tuning choices.
