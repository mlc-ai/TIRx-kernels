<!--
This file is a design sketch for a TIRx port of code from cuDNN Frontend
(https://github.com/NVIDIA/cudnn-frontend @ 2497bee508e6337c086f0993f68b41cabce97436),
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# cuDNN SM103 generic flex-attention forward: coarse WASP pipeline sketch

This is a non-executable, operation-level transcription of
python/cudnn/flex_attention/kernels/sm100/fwd/forward.py together with the
reachable qstage1, qstage2, packed-mask, softmax, pipeline, and CLC scheduler
helpers at cuDNN Frontend commit
2497bee508e6337c086f0993f68b41cabce97436. The executable source of truth will
be tirx_kernels/cudnn/flex_attention/forward_sm103.py.

The transcription freezes the source's sixteen-warp persistent schedule,
qstage2 one-CTA and qstage1 one-/two-CTA issue orders, compact partial/full
interval-mask traversal, Q and aliased K/V TMA rings, two score/P/output TMEM
streams, native SM103 row-max/exp2 behavior, correction protocols, fixed TMA
and true-varlen vector output paths, packed/unpacked GQA addressing, D192
Q/output alias, and the narrow-workset PV/K-wait choice.

The module uses only import tirx_kernels.kern as K for device construction.
It uses scalar control flow, opaque TensorMaps, raw K.ptx instructions,
rank-one storage, and scalar byte/address expressions. It uses no tile
primitive, first-class layout object, inline CUDA device call, IR-check
exemption, or modification below tirx_kernels/kern/.

After the independent sketch reviewer first returns PASS, this file is
immutable.

## Scope and static specialization boundary

In scope:

- target SM103a and PTX ISA 9.3;
- Q/K/V/O all FP16 or all BF16;
- logical D and Dv divisible by 8 in [8,128], plus D=192,Dv=128; physical
  Dp=ceil(D/16)*16 and Dvp=ceil(Dv/16)*16;
- fixed BSHD or true-varlen THD, optional FP32 LSE;
- MHA, GQA, and MQA, with planner-selected packed or explicit unpacked GQA;
- reusable arbitrary interval plans whose odd endpoint count encodes
  [0,F0) union [F1,F2) union ...;
- qstage2_1cta, qstage1_1cta with the narrow/overlap compile-time branch, and
  qstage1_2cta with its shared packed-mask pipeline.

Out of scope: SM100/SM107, D=256, backward, mixed input dtypes, unplanned masks,
a mathematically equivalent alternate attention schedule, and any source
dispatch outside the generic forward routes above.

Static values and source-derived consequences:

| specialization | Q stages | score/TMEM O streams | shared output stages | CTA group / cluster | KV cap | special branch |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| qstage2_1cta | 2 | 2 | 2 | 1 / singleton | capacity, max 32; D192/Dv128 raw 2 promoted to uneven 3 | Q0/Q1 share every K/V |
| qstage1_1cta | 1 | 2 | 2 | 1 / singleton | max 4 | alternating N blocks; overlap only for non-narrow worksets |
| qstage1_2cta | 1 | 2 | 1 | 2 / (2,1,1) | max 7 | group-2 QK/PV and two 2-KiB mask slots |
| qstage1_1cta D192,Dv128 | 1 | 2 | 2 | 1 / singleton | forced 2 | shared output aliases Q and has a one-stage reuse handshake |

The generic capacity formula first resolves to
floor((224KiB-Q/output bytes)/per-CTA KV-stage bytes), capped as above. For
D192,Dv128 one-CTA this raw capacity is two; qstage2 then promotes it to a
three-stage uneven [49152,32768,49152]-byte K/V ring. Qstage1 one-CTA
deliberately forces two regular stages because K1 and V0 overlap. The uneven
ring has 40960-byte average stage stride, 131072-byte cosize, and a signed
8192-byte stage-1 displacement selected by the pipeline phase.

## Writer line-info evidence

The normal installed FlexAttentionFwd compile/execute route produced and
executed these writer exports under
.porting/cudnn_sm103_flex_attention_forward/source_exports. Every target entry
declares .version 9.3, .target sm_103a, .reqntid 512,1,1,
.minnctapersm 1, a 1024-byte-aligned dynamic shared arena, and source .file/.loc
records. The installed files are byte-identical to the user-specified checkout.

| export | target PTX glob beneath export directory | SHA256 | .loc count |
| --- | --- | --- | ---: |
| qstage2_1cta | qstage2_1cta/*FlexAttentionForwardSm100*.ptx | 177633db5d6f06a3f54d2690fd8cf5686b01880fe3d1fbdf3b18c1d0bbc40b7c | 3814 |
| qstage1_1cta narrow | qstage1_1cta_narrow/*FlexAttentionForwardQStage1Sm100*.ptx | 2b92192ae9da5ef7380a1707e0c4ae661caeb00bd83195771938a377fccc3da9 | 5341 |
| qstage1_1cta overlap | qstage1_1cta_wide/*FlexAttentionForwardQStage1Sm100*.ptx | 73d23964c469b71a047c38f816c2c4a9c08a0df01ada02b33f8fb532439fb2b2 | 5343 |
| qstage1_2cta | qstage1_2cta/*FlexAttentionForwardQStage1Sm100*.ptx | a4fcd8d5960599deab88e46aedf2003bb01a399722faa04b1505986fdb5ab6d7 | 5584 |
| qstage1_1cta D192,Dv128 | qstage1_1cta_d192/*FlexAttentionForwardQStage1Sm100*.ptx | 1f7c426cf5866d38af88d75f856aa66e8035822a061a293a147c3271e8fc8bb6 | 5432 |

The qstage1 two-CTA anchor contains 80 group-2 f16-kind MMAs, 17 group-2
cluster-multicast TMEM commits, 16 x32 row-max reduction loads, 16 ordinary
x32 score loads, 32 x16 correction loads, 48 x16 TMEM stores, 1028 native
ftz exp2 operations, eleven rank-4 group-2 tensor loads, four raw bulk mask
copies, one group-2 TMEM allocation/deallocation pair, and the CLC
try-cancel/query family. The one-CTA anchors contain the corresponding group-1
forms. These are static emitted counts, not runtime iteration counts.

## Pipeline at a glance

| warps | role | per-work-record program | publication / reuse edges |
| --- | --- | --- | --- |
| 0-3 | score stream 0 | consume S0, partial-mask or full-row max, online max/sum, native exp2, write P0 | S/P/O[0] full -> stats[0] and first-96 release -> P-last[0] |
| 4-7 | score stream 1 | qstage2 owns Q stage 1; qstage1 owns odd N traversals | corresponding stream-1 edges |
| 8-11 | correction/output | rescale O streams, finalize qstage2 independently or merge qstage1 streams, normalize, stage output, store varlen/LSE | stats and O-full -> S/P/O empty; output-full; optional load/output-alias release |
| 12 | MMA | allocate TMEM; source-exact QK/PV chains; relinquish/deallocate TMEM | Q/KV full and P releases -> S/O full and Q/KV empty |
| 13 | fixed epilogue | consume shared output stage(s), issue TMA S2G | output-full -> output-empty; optional Q/output alias release |
| 14 elected lane | TMA/mask producer | consume work descriptors, issue Q/K/V and optional mask traffic in planned order | Q/KV empty -> Q/KV full; mask empty -> mask full |
| 15 | CLC scheduler on leader CTA | asynchronously fetch next 16-byte work descriptor and cancel tail | one-stage CLC full/empty ring to every warp |
| 15 nonleader and moved warp 13 in varlen | empty consumers | advance the same CLC stream without tile work | preserve all-consumer CLC arrival accounting |

Role branches occur in source order: CLC/empty, load, MMA, fixed epilogue,
score warps, then correction. The roles execute concurrently; this order fixes
only construction and per-role bodies.

## Primitive vocabulary and representation

Every physical storage object is a scalar or a region inside one rank-one byte
arena. Names ending in desc are opaque hardware descriptors assembled by scalar
integer operations; they are not mapping objects.

~~~python
linear_smem_u8(name, byte_count, alignment)
linear_reg(name, dtype, element_count)
smem_byte(arena, scalar_byte_offset)
tmem_address(base_column, scalar_row_band)
tensor_map(name, rank, scalar_extents, scalar_byte_strides, scalar_box, swizzle_code)

copy_g2s(tensor_map, coordinates, smem_byte, completion_barrier, optional_multicast)
copy_s2g(tensor_map, coordinates, smem_byte)
copy_g2r(global_pointer, registers)
copy_s2r(smem_byte, registers)
copy_r2s(registers, smem_byte)
copy_t2r(tmem_address, registers)
copy_r2t(registers, tmem_address)
copy_r2g(registers, global_pointer, predicate)
gemm(tmem_destination, a_operand, b_descriptor, instruction_descriptor, accumulate)
fill(registers, scalar)
cast(registers, destination_dtype, rounding)
max(dst, values); add(dst, lhs, rhs); mul(dst, lhs, rhs); exp2(dst, src)
init / acquire / wait / expect_bytes / arrive / commit / release / fence / barrier / tail
~~~

One copy or gemm below denotes one instruction or one explicit loop of the same
instruction family. Phase and stage arithmetic is scalar bookkeeping.

## Complete sketch

~~~python
P = specialize(
    variant, dtype, D, Dv, qhead_per_kvhead, pack_gqa,
    is_varlen_q, return_lse, overlap_pv_with_k_wait
)
M = 128
N = 128
Dp = ceil_div(D,16)*16
Dvp = ceil_div(Dv,16)*16
QST = 2 if variant == qstage2_1cta else 1
SST = 2
OST = 2 if (variant != qstage1_2cta) else 1
CG = 2 if variant == qstage1_2cta else 1
CLUSTER = (2,1,1) if CG == 2 else singleton
USE_MASK_SMEM = CG == 2
USE_TMA_O = not is_varlen_q
Q_RANK = 5 if (not is_varlen_q and pack_gqa) else (4 if (not is_varlen_q or pack_gqa) else 3)
KV_RANK = 4 if not is_varlen_q else 3
O_RANK = 5 if pack_gqa else 4
OVERLAP_QO = Dp == 192 and Dvp >= 64
RAW_KVST = min(floor((224*1024-Q_OUTPUT_BYTES)/REGULAR_KV_STAGE_BYTES),32)
KVST = 3 if (Dp == 192 and Dvp == 128 and RAW_KVST == 2) else RAW_KVST
if variant == qstage1_1cta:
    KVST = min(KVST,4)
if variant == qstage1_2cta:
    KVST = min(KVST,7)
if variant == qstage1_1cta and Dp == 192 and Dvp == 128:
    KVST = 2
UNEVEN_KV = Dp == 192 and Dvp == 128 and KVST == 3

launch(
    target=sm_103a,
    grid=(num_fwd_work_records*CG,1,1),
    block=(512,1,1),
    cluster=CLUSTER,
    min_blocks_per_sm=1,
    dynamic_smem_bytes=ARENA_BYTES
)
# instruction_selection: .reqntid 512,1,1; .minnctapersm 1; singleton or
# two-CTA cluster launch; extent: one statically specialized entry.

ABI = (
    Q, K, V, O, optional_LSE, Qmap, Kmap, Vmap, optional_Omap,
    softmax_scale_log2,
    partial_count, partial_index, full_count, full_index,
    optional_cu_total_m_blocks, q_write_order, q_write_order_full,
    partial_offset, full_offset, mask_payload,
    sequence_desc, fwd_work_desc
)

# The plan work record is exactly four int32 values.
# [m_block, head_idx, batch_idx, q_valid_rows].
# Per-plan-row lookup yields partial_count/indices in reverse traversal order,
# then full_count/indices in reverse traversal order. Packed GQA selects a KV
# head and folds local query-head position into the 128-row coordinate.

# One rank-one shared byte arena; fields are placed in source struct order.
arena = linear_smem_u8(ARENA_BYTES, alignment=1024)
cursor = 0
def place(bytes,align):
    cursor = align_up(cursor,align)
    old = cursor
    cursor += bytes
    return old

OFF_QBAR     = place(16*QST,8)
OFF_KVBAR    = place(16*KVST,8)
OFF_MASK0    = place(16 if USE_MASK_SMEM else 0,8)
OFF_MASK1    = place(16 if USE_MASK_SMEM else 0,8)
OFF_SPO      = place(16*SST,8)
OFF_PLAST    = place(16*SST,8)
OFF_OACC     = place(16*SST,8)
OFF_STATSBAR = place(16*SST,8)
OFF_OEPI     = place(16*OST,8)
OFF_LOADEPI  = place(16 if OVERLAP_QO else 0,8)
OFF_TMEM_DEALLOC = place(8,8)
OFF_TMEM_MAILBOX = place(4,4)
OFF_SCALE = place(SST*M*2*4,4)
OFF_MASK = place(SST*M*4*4 if USE_MASK_SMEM else 0,16)
OFF_CLC_BAR = place(16,8)
OFF_CLC_RESPONSE = place(16,16)
SO_BYTES = 0 if OVERLAP_QO else OST*M*Dvp*2
SQ_BYTES = max(QST*M*Dp*2, OST*M*Dvp*2) if OVERLAP_QO else QST*M*Dp*2
K_STAGE_BYTES = N*Dp*2//CG
V_STAGE_BYTES = N*Dvp*2//CG
KV_STAGE_BYTES = max(K_STAGE_BYTES,V_STAGE_BYTES)
KV_ARENA_BYTES = ((KVST-1)*40960 + 49152) if UNEVEN_KV else KVST*KV_STAGE_BYTES
OFF_SO = place(SO_BYTES,1024)
OFF_SQ = place(SQ_BYTES,1024)
OFF_SKV = place(KV_ARENA_BYTES,1024)
ARENA_BYTES = cursor

# Every two-sided n-stage pipe stores full barriers first, then empty barriers.
def pipe_full(base,n,s):  return base + 8*s
def pipe_empty(base,n,s): return base + 8*n + 8*s

# Source K-major atom selection for a padded 16-bit contiguous dimension.
def band_width(padded_d):
    return 64 if padded_d%64 == 0 else (32 if padded_d%32 == 0 else 16)
def swizzle_code(padded_d):
    return 3 if padded_d%64 == 0 else (2 if padded_d%32 == 0 else 1)
def swizzle_element(e,bw):
    bits = 3 if bw == 64 else (2 if bw == 32 else 1)
    return e ^ (((e>>7)&((1<<bits)-1))<<4)
def matrix_element(row,col,padded_d):
    bw = band_width(padded_d)
    band = col//bw
    local = col-band*bw
    return band*(128*bw)+swizzle_element(row*bw+local,bw)
def q_byte(stage,row,col):
    return OFF_SQ + 2*(stage*M*Dp + matrix_element(row,col,Dp))
def uneven_kv_stage_byte(stage,phase):
    # Source offset_kv_smem: stage 1 moves right for phase 0, left for phase 1.
    # Producer pipeline phases start inverted, so producer calls with phase^1;
    # MMA consumers call with their current consumer phase.
    return OFF_SKV + stage*40960 + (8192*(1-2*phase) if stage == 1 else 0)
def kv_byte(stage,phase,row,col,padded_d,is_producer):
    base = (uneven_kv_stage_byte(stage,phase ^ 1 if is_producer else phase)
            if UNEVEN_KV else OFF_SKV + stage*KV_STAGE_BYTES)
    return base + 2*matrix_element(row,col,padded_d)
def o_byte(stage,row,col):
    base = OFF_SQ if OVERLAP_QO else OFF_SO
    return base + 2*(stage*M*Dvp + matrix_element(row,col,Dvp))
def scale_byte(stream,row,is_max):
    return OFF_SCALE + 4*(stream*M + row + (SST*M if is_max else 0))
def mask_byte(stream,row,word):
    return OFF_MASK + 4*(stream*M*4 + row*4 + word)

# Raw shared descriptors encode the scalar start in 16-byte units, the source
# leading/stride fields, and swizzle_code. Q/K are K-major; V is MN-major.
# O uses the epilogue descriptor matching fixed BSHD output.
def q_desc(stage,k16): return source_q_descriptor(q_byte(stage,0,0),k16,Dp)
def k_desc(stage,phase,k16):
    return source_k_descriptor(kv_byte(stage,phase,0,0,Dp,false),k16,Dp)
def v_desc(stage,phase,k16):
    return source_v_descriptor(kv_byte(stage,phase,0,0,Dvp,false),k16,Dvp)

# TMEM is always a 512-column allocation. S streams own columns 0 and 128.
# P overlays the upper half of each S stream at S+64 in 32-bit column units.
# O streams begin at 256 and 256+Dvp.
S_COL = [0,128]
P_COL = [64,192]
O_COL = [256,256+Dvp]

# Pipelines. Full/empty phase starts match the source Pipeline constructors.
q_pipe       = two_sided_tma_umma(OFF_QBAR,QST,producer=one_tma_thread,consumer=warp12,tx=Q_BYTES*CG)
kv_pipe      = two_sided_tma_umma(OFF_KVBAR,KVST,producer=one_tma_thread,consumer=warp12,tx=K_BYTES*CG)
spo_pipe     = umma_async(OFF_SPO,SST,producer=warp12,consumer=(score_warps_plus_correction)*CG)
plast_pipe   = async_umma(OFF_PLAST,SST,producer=score_warps*CG,consumer=warp12)
oacc_pipe    = umma_async(OFF_OACC,SST,producer=warp12,consumer=correction_threads*CG)
stats_pipe   = two_sided_async(OFF_STATSBAR,SST,producer=128_score_threads,consumer=128_correction_threads)
oepi_pipe    = two_sided_async(OFF_OEPI,OST,producer=correction_threads,consumer=warp13) if USE_TMA_O else none
load_epi_pipe= two_sided_async(OFF_LOADEPI,1,producer=output_warps,consumer=warp14) if OVERLAP_QO else none
mask0_pipe   = one_stage_tma(OFF_MASK0,producer=one_tma_thread,consumer=warps0_3,tx=2048) if USE_MASK_SMEM else none
mask1_pipe   = one_stage_tma(OFF_MASK1,producer=one_tma_thread,consumer=warps4_7,tx=2048) if USE_MASK_SMEM else none
clc_pipe     = one_stage_clc(OFF_CLC_BAR,producer=one_thread,consumer=all_16_warps*CG,tx=16)

# Named barriers: 1 joins output warps before direct stores; 2 joins the MMA,
# eight score, and four correction warps for TMEM pointer publication/dealloc;
# ids 3..10 pair one score warp with its matching correction warp.
EPILOGUE_NAMED = 1
TMEM_PTR_NAMED = 2
SOFTMAX_STATS_NAMED_BASE = 3

if warp_id == 0:
    for map in Qmap,Kmap,Vmap,optional_Omap:
        if map exists:
            prefetch(map)
            # instruction_selection: prefetch.tensormap; extent: one descriptor.

# There are three distinct source initialization protocols; they are not merged.
if CG == 2 and warp_id == 12 and elected_lane:
    init(OFF_TMEM_DEALLOC,arrival_count=32)
    # instruction_selection: mbarrier.init.shared.b64; extent: one group-2 TMEM-dealloc barrier.
    fence_init()
    # instruction_selection: fence.mbarrier_init.release.cluster; extent: warp-12 elected lane.
# The singleton variant has no TMEM-dealloc mbarrier.

if elected main_pipeline_initializer:
    for barrier in q,kv,mask,spo,plast,oacc,stats,oepi,load_epi live barriers:
        init(smem_byte(arena,barrier.offset),barrier.source_arrival_count)
        # instruction_selection: mbarrier.init.shared.b64; extent: each deferred main-pipeline barrier.
    fence_init()
    # instruction_selection: fence.mbarrier_init.release.cluster; extent: main-pipeline initializer.
if CG == 2:
    arrive_cluster_relaxed()
    # instruction_selection: barrier.cluster.arrive.relaxed; extent: one CTA.
    wait_cluster()
    # instruction_selection: barrier.cluster.wait; extent: one CTA.
else:
    cta_sync()
    # instruction_selection: bar.sync 0; extent: singleton CTA.

# CLC is created non-deferred and initializes only its full/empty barriers.
if elected_clc_initializer:
    init(clc_pipe.full,clc_full_arrivals); init(clc_pipe.empty,clc_empty_arrivals)
    # instruction_selection: mbarrier.init.shared.b64; extent: two CLC barriers.
    fence_init()
    # instruction_selection: fence.mbarrier_init.release.cluster; extent: CLC initializer.
if CG == 2:
    arrive_cluster_relaxed(); wait_cluster()
    # instruction_selection: barrier.cluster.arrive.relaxed then barrier.cluster.wait; extent: CTA.
else:
    cta_sync()
    # instruction_selection: bar.sync 0; extent: singleton CTA.

# The leader CTA's warp 15 owns the CLC producer and also consumes the
# resulting work. Every other role owns an identical consumer sequence.
if warp_id == 15 and cta_rank == 0:
    clc_producer_stage=0; clc_producer_phase=1
    clc_consumer_stage=0; clc_consumer_phase=0
    initial_hw=hardware_initial_work()
    initial_task=initial_hw.cta_x//CG
    initial_valid=initial_hw.valid and initial_task<num_fwd_work_records
    if initial_valid:
        m_block=load_global_i32(fwd_work_desc,initial_task,0)
        head_idx=load_global_i32(fwd_work_desc,initial_task,1)
        batch_idx=load_global_i32(fwd_work_desc,initial_task,2)
        # instruction_selection: ld.global.b32; extent: three live initial descriptor fields.
        # Source q_valid_rows field 3 is dead and eliminated from target PTX.
    work=(m_block,head_idx,batch_idx,source_field3_dead,initial_valid)
    while work.valid:
        acquire(clc_pipe.empty,clc_producer_stage,clc_producer_phase)
        # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64; extent: one producer poll.
        try_cancel_next_grid_cta_into_16B_response(clc_pipe.full_barrier)
        # instruction_selection: clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128; extent: one response.
        advance clc producer state

        wait(clc_pipe.full,clc_consumer_stage,clc_consumer_phase)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one consumer poll.
        response=load_shared_u128(OFF_CLC_RESPONSE)
        # instruction_selection: ld.shared.v2.b64; extent: one 128-bit response.
        canceled=query_cancel_is_canceled(response)
        next_x=query_cancel_get_first_ctaid_x(response)
        # instruction_selection: clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 and get_first_ctaid::x.b32.b128; extent: one response.
        fence_async_shared()
        # instruction_selection: fence.proxy.async.shared::cta; extent: one response.
        task_idx=next_x//CG; valid=(not canceled) and task_idx<num_fwd_work_records
        if valid:
            m_block=load_global_i32(fwd_work_desc,task_idx,0)
            head_idx=load_global_i32(fwd_work_desc,task_idx,1)
            batch_idx=load_global_i32(fwd_work_desc,task_idx,2)
            # instruction_selection: ld.global.b32; extent: three live descriptor fields; source field 3 is PTX-DCE.
        release(clc_pipe.empty,clc_consumer_stage)
        # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: one consumer warp.
        advance clc consumer state
        work=(m_block,head_idx,batch_idx,source_field3_dead,valid)
    tail(clc_pipe)
    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: producer's outstanding empty-stage drain only.
else if this is an empty warp:
    clc_consumer_stage=0; clc_consumer_phase=0
    initial_hw=hardware_initial_work()
    initial_task=initial_hw.cta_x//CG
    initial_valid=initial_hw.valid and initial_task<num_fwd_work_records
    if initial_valid:
        m_block=load_global_i32(fwd_work_desc,initial_task,0)
        head_idx=load_global_i32(fwd_work_desc,initial_task,1)
        batch_idx=load_global_i32(fwd_work_desc,initial_task,2)
        # instruction_selection: ld.global.b32; extent: three live initial descriptor fields.
        # Source q_valid_rows field 3 is dead and eliminated from target PTX.
    work=(m_block,head_idx,batch_idx,source_field3_dead,initial_valid)
    while work.valid:
        wait(clc_pipe.full,clc_consumer_stage,clc_consumer_phase)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one consumer poll.
        response=load_shared_u128(OFF_CLC_RESPONSE)
        # instruction_selection: ld.shared.v2.b64; extent: one 128-bit response.
        canceled=query_cancel_is_canceled(response)
        next_x=query_cancel_get_first_ctaid_x(response)
        # instruction_selection: clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 and get_first_ctaid::x.b32.b128; extent: one response.
        fence_async_shared()
        # instruction_selection: fence.proxy.async.shared::cta; extent: one response.
        task_idx=next_x//CG; valid=(not canceled) and task_idx<num_fwd_work_records
        if valid:
            m_block=load_global_i32(fwd_work_desc,task_idx,0)
            head_idx=load_global_i32(fwd_work_desc,task_idx,1)
            batch_idx=load_global_i32(fwd_work_desc,task_idx,2)
            # instruction_selection: ld.global.b32; extent: three live descriptor fields; source field 3 is PTX-DCE.
        release(clc_pipe.empty,clc_consumer_stage)
        # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: one consumer warp.
        advance clc consumer state
        work=(m_block,head_idx,batch_idx,source_field3_dead,valid)

# Warp 14: TMA and optional packed-mask producer.
if warp_id == 14:
    q_producer_phase=1
    kv_producer_stage=0; kv_producer_phase=1
    mask0_producer_stage=0; mask0_producer_phase=1
    mask1_producer_stage=0; mask1_producer_phase=1
    load_epi_consumer_stage=0; load_epi_consumer_phase=0
    clc_consumer_stage=0; clc_consumer_phase=0

    initial_hw=hardware_initial_work()
    initial_task=initial_hw.cta_x//CG
    initial_valid=initial_hw.valid and initial_task<num_fwd_work_records
    if initial_valid:
        m_block=load_global_i32(fwd_work_desc,initial_task,0)
        head_idx=load_global_i32(fwd_work_desc,initial_task,1)
        batch_idx=load_global_i32(fwd_work_desc,initial_task,2)
        # instruction_selection: ld.global.b32; extent: three live descriptor fields.
        # Source field 3 is q_valid_rows; it is dead and PTX-DCE in all specializations.
    work=(m_block,head_idx,batch_idx,source_field3_dead,initial_valid)

    while work.valid:
        decode sequence offsets and packed/unpacked head map
        partial/full lists select reverse partial order then reverse full order
        total=partial_count+full_count

        if variant == qstage2_1cta:
            q_was_produced=false
            if partial_count>0:
                for qs in 0,1:
                    acquire(q_pipe.empty[qs],q_producer_phase)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one Q producer acquire.
                    expect_bytes(q_pipe.full[qs],Q_BYTES*CG)
                    # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one Q transaction.
                    copy_g2s(Qmap,q_coordinates(qs),q_byte(qs,0,0),q_pipe.full[qs])
                    # instruction_selection: cp.async.bulk.tensor.<Q_RANK>d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint; extent: one qstage2 Q box.
                q_was_produced=true
                current=partial_count-1
                prefetch(partial_indices[current])
                for qs in 0,1: prefetch(mask_payload[payload_base+current],qstage=qs)
                # instruction_selection: prefetch.global.L1; extent: current index and both Q-stage payload subtiles.

                for offset in range(partial_count):
                    ordinal=partial_count-1-offset
                    n_block=load_global_i32(partial_indices,ordinal)
                    # instruction_selection: ld.global.b32; extent: one partial index.
                    if offset+1<partial_count:
                        prefetch(partial_indices[ordinal-1])
                        for qs in 0,1: prefetch(mask_payload[payload_base+ordinal-1],qstage=qs)
                        # instruction_selection: prefetch.global.L1; extent: next index and both Q-stage payload subtiles.

                    acquire(kv_pipe.empty[kv_producer_stage],kv_producer_phase)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: K producer acquire.
                    if UNEVEN_KV and kv_producer_stage==0:
                        wait(kv_pipe.empty[1],kv_producer_phase)
                        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: uneven-ring neighbor reuse wait.
                    expect_bytes(kv_pipe.full[kv_producer_stage],K_BYTES*CG)
                    # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one K transaction.
                    copy_g2s(Kmap,k_coordinates(n_block),kv_byte(kv_producer_stage,kv_producer_phase,0,0,Dp,true),kv_pipe.full[kv_producer_stage])
                    # instruction_selection: cp.async.bulk.tensor.<KV_RANK>d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint; extent: one qstage2 K box.
                    advance kv producer state

                    acquire(kv_pipe.empty[kv_producer_stage],kv_producer_phase)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: V producer acquire.
                    expect_bytes(kv_pipe.full[kv_producer_stage],V_BYTES*CG)
                    # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one V transaction.
                    copy_g2s(Vmap,v_coordinates(n_block),kv_byte(kv_producer_stage,kv_producer_phase,0,0,Dvp,true),kv_pipe.full[kv_producer_stage])
                    # instruction_selection: cp.async.bulk.tensor.<KV_RANK>d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint; extent: one qstage2 V box.
                    advance kv producer state
            elif full_count>0:
                for qs in 0,1:
                    acquire(q_pipe.empty[qs],q_producer_phase)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one Q producer acquire.
                    expect_bytes(q_pipe.full[qs],Q_BYTES*CG)
                    # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one Q transaction.
                    copy_g2s(Qmap,q_coordinates(qs),q_byte(qs,0,0),q_pipe.full[qs])
                    # instruction_selection: cp.async.bulk.tensor.<Q_RANK>d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint; extent: one qstage2 Q box.
                q_was_produced=true

            if full_count>0:
                current=full_count-1
                prefetch(full_indices[current])
                # instruction_selection: prefetch.global.L1; extent: current full index.
                for offset in range(full_count):
                    ordinal=full_count-1-offset
                    n_block=load_global_i32(full_indices,ordinal)
                    # instruction_selection: ld.global.b32; extent: one full index.
                    if offset+1<full_count:
                        prefetch(full_indices[ordinal-1])
                        # instruction_selection: prefetch.global.L1; extent: next full index.

                    acquire(kv_pipe.empty[kv_producer_stage],kv_producer_phase)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: K producer acquire.
                    if UNEVEN_KV and kv_producer_stage==0:
                        wait(kv_pipe.empty[1],kv_producer_phase)
                        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: uneven-ring neighbor reuse wait.
                    expect_bytes(kv_pipe.full[kv_producer_stage],K_BYTES*CG)
                    # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one K transaction.
                    copy_g2s(Kmap,k_coordinates(n_block),kv_byte(kv_producer_stage,kv_producer_phase,0,0,Dp,true),kv_pipe.full[kv_producer_stage])
                    # instruction_selection: cp.async.bulk.tensor.<KV_RANK>d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint; extent: one qstage2 K box.
                    advance kv producer state

                    acquire(kv_pipe.empty[kv_producer_stage],kv_producer_phase)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: V producer acquire.
                    expect_bytes(kv_pipe.full[kv_producer_stage],V_BYTES*CG)
                    # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one V transaction.
                    copy_g2s(Vmap,v_coordinates(n_block),kv_byte(kv_producer_stage,kv_producer_phase,0,0,Dvp,true),kv_pipe.full[kv_producer_stage])
                    # instruction_selection: cp.async.bulk.tensor.<KV_RANK>d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint; extent: one qstage2 V box.
                    advance kv producer state
            if q_was_produced: q_producer_phase^=1
        else:
            # Source-exact Q -> mask0 -> K0 -> mask1 -> K1 ->
            # V(i-2) -> mask(i) -> K(i) -> final two Vs.
            if total>0:
                acquire(q_pipe.empty[0],q_producer_phase)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: Q0 producer acquire.
                expect_bytes(q_pipe.full[0],Q_BYTES*CG)
                # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: Q0 transaction.
                copy_g2s(Qmap,q_coordinates(0),q_byte(0,0,0),q_pipe.full[0])
                # instruction_selection: cp.async.bulk.tensor.<Q_RANK>d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint for CG=1, or cp.async.bulk.tensor.<Q_RANK>d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2 for CG=2; extent: one Q0 box.

                if USE_MASK_SMEM and ordinal0 is partial:
                    acquire(mask0_pipe.empty,mask0_producer_phase)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: mask0 acquire.
                    expect_bytes(mask0_pipe.full,2048)
                    # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: 2048 bytes.
                    copy_g2s(mask_payload,payload0,mask_byte(0,0,0),mask0_pipe.full)
                    # instruction_selection: cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes; extent: one 2048-byte payload.
                    advance mask0 producer state

                n0=block_by_traversal_ordinal(0)
                acquire(kv_pipe.empty[kv_producer_stage],kv_producer_phase)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: K0 acquire.
                expect_bytes(kv_pipe.full[kv_producer_stage],K_BYTES*CG)
                # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: K0 transaction.
                copy_g2s(Kmap,k_coordinates(n0),kv_byte(kv_producer_stage,kv_producer_phase,0,0,Dp,true),kv_pipe.full[kv_producer_stage])
                # instruction_selection: cp.async.bulk.tensor.<KV_RANK>d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint for CG=1, or cp.async.bulk.tensor.<KV_RANK>d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2 for CG=2; extent: one K0 box.
                advance kv producer state

                if total==1:
                    acquire(kv_pipe.empty[kv_producer_stage],kv_producer_phase)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: V0 acquire.
                    expect_bytes(kv_pipe.full[kv_producer_stage],V_BYTES*CG)
                    # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: V0 transaction.
                    copy_g2s(Vmap,v_coordinates(n0),kv_byte(kv_producer_stage,kv_producer_phase,0,0,Dvp,true),kv_pipe.full[kv_producer_stage])
                    # instruction_selection: cp.async.bulk.tensor.<KV_RANK>d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint for CG=1, or cp.async.bulk.tensor.<KV_RANK>d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2 for CG=2; extent: one V0 box.
                    advance kv producer state
                else:
                    if USE_MASK_SMEM and ordinal1 is partial:
                        acquire(mask1_pipe.empty,mask1_producer_phase)
                        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: mask1 acquire.
                        expect_bytes(mask1_pipe.full,2048)
                        # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: 2048 bytes.
                        copy_g2s(mask_payload,payload1,mask_byte(1,0,0),mask1_pipe.full)
                        # instruction_selection: cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes; extent: one 2048-byte payload.
                        advance mask1 producer state

                    n1=block_by_traversal_ordinal(1)
                    acquire(kv_pipe.empty[kv_producer_stage],kv_producer_phase)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: K1 acquire.
                    expect_bytes(kv_pipe.full[kv_producer_stage],K_BYTES*CG)
                    # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: K1 transaction.
                    copy_g2s(Kmap,k_coordinates(n1),kv_byte(kv_producer_stage,kv_producer_phase,0,0,Dp,true),kv_pipe.full[kv_producer_stage])
                    # instruction_selection: cp.async.bulk.tensor.<KV_RANK>d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint for CG=1, or cp.async.bulk.tensor.<KV_RANK>d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2 for CG=2; extent: one K1 box.
                    advance kv producer state

                    for ordinal in range(2,total):
                        nv=block_by_traversal_ordinal(ordinal-2)
                        acquire(kv_pipe.empty[kv_producer_stage],kv_producer_phase)
                        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one steady V acquire.
                        expect_bytes(kv_pipe.full[kv_producer_stage],V_BYTES*CG)
                        # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one steady V transaction.
                        copy_g2s(Vmap,v_coordinates(nv),kv_byte(kv_producer_stage,kv_producer_phase,0,0,Dvp,true),kv_pipe.full[kv_producer_stage])
                        # instruction_selection: cp.async.bulk.tensor.<KV_RANK>d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint for CG=1, or cp.async.bulk.tensor.<KV_RANK>d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2 for CG=2; extent: one steady V box.
                        advance kv producer state

                        if USE_MASK_SMEM and ordinal is partial:
                            acquire(mask_pipe[ordinal&1].empty,mask_producer_phase[ordinal&1])
                            # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one parity mask acquire.
                            expect_bytes(mask_pipe[ordinal&1].full,2048)
                            # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: 2048 bytes.
                            copy_g2s(mask_payload,payload_ordinal,mask_byte(ordinal&1,0,0),mask_pipe[ordinal&1].full)
                            # instruction_selection: cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes; extent: one 2048-byte payload.
                            advance selected mask producer state

                        nk=block_by_traversal_ordinal(ordinal)
                        acquire(kv_pipe.empty[kv_producer_stage],kv_producer_phase)
                        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one steady K acquire.
                        expect_bytes(kv_pipe.full[kv_producer_stage],K_BYTES*CG)
                        # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one steady K transaction.
                        copy_g2s(Kmap,k_coordinates(nk),kv_byte(kv_producer_stage,kv_producer_phase,0,0,Dp,true),kv_pipe.full[kv_producer_stage])
                        # instruction_selection: cp.async.bulk.tensor.<KV_RANK>d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint for CG=1, or cp.async.bulk.tensor.<KV_RANK>d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2 for CG=2; extent: one steady K box.
                        advance kv producer state

                    for ordinal in total-2,total-1:
                        nv=block_by_traversal_ordinal(ordinal)
                        acquire(kv_pipe.empty[kv_producer_stage],kv_producer_phase)
                        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one tail V acquire.
                        expect_bytes(kv_pipe.full[kv_producer_stage],V_BYTES*CG)
                        # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one tail V transaction.
                        copy_g2s(Vmap,v_coordinates(nv),kv_byte(kv_producer_stage,kv_producer_phase,0,0,Dvp,true),kv_pipe.full[kv_producer_stage])
                        # instruction_selection: cp.async.bulk.tensor.<KV_RANK>d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint for CG=1, or cp.async.bulk.tensor.<KV_RANK>d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2 for CG=2; extent: one tail V box.
                        advance kv producer state
                q_producer_phase^=1

        # Load-role next-work protocol, before the optional D192 alias wait.
        wait(clc_pipe.full,clc_consumer_stage,clc_consumer_phase)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one response wait.
        response=load_shared_u128(OFF_CLC_RESPONSE)
        # instruction_selection: ld.shared.v2.b64; extent: one 128-bit response.
        canceled=query_cancel_is_canceled(response)
        next_x=query_cancel_get_first_ctaid_x(response)
        # instruction_selection: clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 and get_first_ctaid::x.b32.b128; extent: one response.
        fence_async_shared()
        # instruction_selection: fence.proxy.async.shared::cta; extent: one response.
        task_idx=next_x//CG; valid=(not canceled) and task_idx<num_fwd_work_records
        if valid:
            m_block=load_global_i32(fwd_work_desc,task_idx,0)
            head_idx=load_global_i32(fwd_work_desc,task_idx,1)
            batch_idx=load_global_i32(fwd_work_desc,task_idx,2)
            # instruction_selection: ld.global.b32; extent: three live descriptor fields; source field 3 is PTX-DCE.
        release(clc_pipe.empty,clc_consumer_stage)
        # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: one consumer warp.
        advance clc consumer state
        work=(m_block,head_idx,batch_idx,source_field3_dead,valid)

        if OVERLAP_QO:
            wait(load_epi_pipe.full,load_epi_consumer_stage,load_epi_consumer_phase)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: no pre-production first-tile wait; first wait is after first-tile loads and CLC advance.
            release(load_epi_pipe.empty,load_epi_consumer_stage)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: one alias release.
            advance load-epi consumer state

    tail every KV and mask producer ring
    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one drain poll per live stage.
    acquire(q_pipe.empty[QST-1],q_producer_phase)
    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: final Q reuse poll.

# Warp 12: TMEM allocation and MMA issue.
if warp_id == 12:
    mma_q_consumer_phase=0
    kv_consumer_stage=0; kv_consumer_phase=0
    qstage2_spo_producer_phase=0
    qstage1_spo_phase0=0; qstage1_spo_phase1=0
    clc_consumer_stage=0; clc_consumer_phase=0

    allocate_tmem(smem_byte(arena,OFF_TMEM_MAILBOX),512,CG)
    # instruction_selection: tcgen05.alloc.cta_group::1 or ::2.sync.aligned.shared::cta.b32; extent: elected allocator.
    named_barrier_sync(TMEM_PTR_NAMED,416)
    # instruction_selection: bar.sync; extent: thirteen participating warps.
    tmem_base=load_smem_u32(OFF_TMEM_MAILBOX)
    # instruction_selection: ld.shared.u32; extent: scalar.

    initial_hw=hardware_initial_work()
    initial_task=initial_hw.cta_x//CG
    initial_valid=initial_hw.valid and initial_task<num_fwd_work_records
    if initial_valid:
        m_block=load_global_i32(fwd_work_desc,initial_task,0)
        head_idx=load_global_i32(fwd_work_desc,initial_task,1)
        batch_idx=load_global_i32(fwd_work_desc,initial_task,2)
        # instruction_selection: ld.global.b32; extent: three live initial descriptor fields; source field 3 is PTX-DCE.
    work=(m_block,head_idx,batch_idx,source_field3_dead,initial_valid)

    while work.valid:
        total=partial_count+full_count
        if total>0 and cta_rank==0:
            if variant == qstage2_1cta:
                for stage in 0,1:
                    wait(q_pipe.full[stage],mma_q_consumer_phase)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one Q stage.
                    if stage==0:
                        wait(kv_pipe.full,kv_consumer_state)
                        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: shared K0 wait once.
                        Kstage=kv_consumer_stage; Kphase=kv_consumer_phase
                    for k16 in range(Dp//16):
                        gemm(S_COL[stage],q_desc(stage,k16),k_desc(Kstage,Kphase,k16),QK_IDESC,accumulate=(k16!=0))
                        # instruction_selection: tcgen05.mma.cta_group::1.kind::f16; extent: Dp/16 QK issues.
                    commit(spo_pipe.full[stage])
                    # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64; extent: one S stage.
                mma_q_consumer_phase^=1
                release(kv_pipe.empty,Kstage)
                # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: shared K0 released once.
                advance kv consumer state

                accumulated=false
                for block in range(total-1):
                    wait(kv_pipe.full,kv_consumer_state)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one V stage.
                    Vrelease=clone(kv_consumer_state)
                    Vstage=kv_consumer_stage; Vphase=kv_consumer_phase
                    for stage in 0,1:
                        acquire(spo_pipe.empty[stage],qstage2_spo_producer_phase)
                        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one SPO stage.
                        for k16 in range(8):
                            if k16==6:
                                wait(plast_pipe.full[stage],qstage2_spo_producer_phase)
                                # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: P last-split before issue six.
                            gemm(O_COL[stage],P_COL[stage]+8*k16,v_desc(Vstage,Vphase,k16),PV_IDESC,accumulate=(accumulated or k16!=0))
                            # instruction_selection: tcgen05.mma.cta_group::1.kind::f16; extent: one of eight PV issues.
                        if stage==1:
                            release(kv_pipe.empty,Vrelease.index)
                            # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: old V released once after stream 1.
                            advance Vrelease
                        if stage==0:
                            advance kv consumer state
                            wait(kv_pipe.full,kv_consumer_state)
                            # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: next shared K wait once.
                            Kstage=kv_consumer_stage; Kphase=kv_consumer_phase
                        for k16 in range(Dp//16):
                            gemm(S_COL[stage],q_desc(stage,k16),k_desc(Kstage,Kphase,k16),QK_IDESC,accumulate=(k16!=0))
                            # instruction_selection: tcgen05.mma.cta_group::1.kind::f16; extent: Dp/16 steady QK issues.
                        commit(spo_pipe.full[stage])
                        # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64; extent: one steady S stage.
                    release(kv_pipe.empty,Kstage)
                    # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: shared K released once after stream 1.
                    advance kv consumer state
                    qstage2_spo_producer_phase^=1
                    accumulated=true

                for stage in 0,1:
                    release(q_pipe.empty[stage])
                    # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: one Q stage.
                wait(kv_pipe.full,kv_consumer_state)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: final V wait once.
                Vstage=kv_consumer_stage; Vphase=kv_consumer_phase
                for stage in 0,1:
                    acquire(spo_pipe.empty[stage],qstage2_spo_producer_phase)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: final SPO stage.
                    for k16 in range(8):
                        if k16==6:
                            wait(plast_pipe.full[stage],qstage2_spo_producer_phase)
                            # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: P last-split.
                        gemm(O_COL[stage],P_COL[stage]+8*k16,v_desc(Vstage,Vphase,k16),PV_IDESC,accumulate=(accumulated or k16!=0))
                        # instruction_selection: tcgen05.mma.cta_group::1.kind::f16; extent: one of eight final PV issues.
                    commit(oacc_pipe.full[stage])
                    # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64; extent: one final O stage.
                qstage2_spo_producer_phase^=1
                release(kv_pipe.empty,Vstage)
                # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: final V.
                advance kv consumer state
            else:
                # QK prologue stream 0.
                wait(q_pipe.full[0],mma_q_consumer_phase)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: Q0.
                mma_q_consumer_phase^=1
                wait(kv_pipe.full,kv_consumer_state)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: K0.
                Kstage=kv_consumer_stage; Kphase=kv_consumer_phase
                for k16 in range(Dp//16):
                    gemm(S_COL[0],q_desc(0,k16),k_desc(Kstage,Kphase,k16),QK_IDESC,accumulate=(k16!=0))
                    # instruction_selection: tcgen05.mma.cta_group::1.kind::f16 for CG=1 or tcgen05.mma.cta_group::2.kind::f16 for CG=2; extent: Dp/16 QK issues.
                commit(spo_pipe.full[0])
                # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 for CG=1 or tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 for CG=2; extent: S0.
                release(kv_pipe.empty,Kstage)
                # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: K0.
                advance kv consumer state

                if total>1:
                    wait(kv_pipe.full,kv_consumer_state)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: K1.
                    Kstage=kv_consumer_stage; Kphase=kv_consumer_phase
                    for k16 in range(Dp//16):
                        gemm(S_COL[1],q_desc(0,k16),k_desc(Kstage,Kphase,k16),QK_IDESC,accumulate=(k16!=0))
                        # instruction_selection: tcgen05.mma.cta_group::1.kind::f16 for CG=1 or tcgen05.mma.cta_group::2.kind::f16 for CG=2; extent: Dp/16 QK issues.
                    commit(spo_pipe.full[1])
                    # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 for CG=1 or tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 for CG=2; extent: S1.
                    release(kv_pipe.empty,Kstage)
                    # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: K1.
                    advance kv consumer state

                o_acc0=false; o_acc1=false
                middle_count=max(total-2,0)
                for stream in each paired middle 0,1, then optional odd stream 0:
                    phase_cur=qstage1_spo_phase0 if stream==0 else qstage1_spo_phase1
                    wait(kv_pipe.full,kv_consumer_state)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: current V.
                    Vrelease=clone(kv_consumer_state)
                    Vstage=kv_consumer_stage; Vphase=kv_consumer_phase
                    advance kv consumer state
                    Kstage=kv_consumer_stage; Kphase=kv_consumer_phase
                    if not overlap_pv_with_k_wait:
                        wait(kv_pipe.full,kv_consumer_state)
                        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: K before PV.
                    acquire(spo_pipe.empty[stream],phase_cur)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: selected SPO stream.
                    for k16 in range(8):
                        if k16==6:
                            wait(plast_pipe.full[stream],phase_cur)
                            # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: P last-split.
                        gemm(O_COL[stream],P_COL[stream]+8*k16,v_desc(Vstage,Vphase,k16),PV_IDESC,accumulate=((o_acc0 if stream==0 else o_acc1) or k16!=0))
                        # instruction_selection: tcgen05.mma.cta_group::1.kind::f16 for CG=1 or tcgen05.mma.cta_group::2.kind::f16 for CG=2; extent: one of eight PV issues.
                    if overlap_pv_with_k_wait:
                        release(kv_pipe.empty,Vrelease.index)
                        # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: old V before K wait.
                        wait(kv_pipe.full,kv_consumer_state)
                        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: overlapped K wait.
                    for k16 in range(Dp//16):
                        gemm(S_COL[stream],q_desc(0,k16),k_desc(Kstage,Kphase,k16),QK_IDESC,accumulate=(k16!=0))
                        # instruction_selection: tcgen05.mma.cta_group::1.kind::f16 for CG=1 or tcgen05.mma.cta_group::2.kind::f16 for CG=2; extent: Dp/16 QK issues.
                    commit(spo_pipe.full[stream])
                    # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 for CG=1 or tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 for CG=2; extent: selected S stream.
                    if stream==0: qstage1_spo_phase0^=1; o_acc0=true
                    else: qstage1_spo_phase1^=1; o_acc1=true
                    if not overlap_pv_with_k_wait:
                        release(kv_pipe.empty,Vrelease.index)
                        # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: old V after QK.
                    release(kv_pipe.empty,Kstage)
                    # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: current K.
                    advance kv consumer state

                release(q_pipe.empty[0])
                # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: Q0 after the last QK.
                for epilogue_ordinal in final min(total,2) traversal ordinals:
                    stream=epilogue_ordinal&1
                    phase_cur=qstage1_spo_phase0 if stream==0 else qstage1_spo_phase1
                    wait(kv_pipe.full,kv_consumer_state)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: final V.
                    Vstage=kv_consumer_stage; Vphase=kv_consumer_phase
                    acquire(spo_pipe.empty[stream],phase_cur)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: final selected SPO stream.
                    for k16 in range(8):
                        if k16==6:
                            wait(plast_pipe.full[stream],phase_cur)
                            # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: P last-split.
                        gemm(O_COL[stream],P_COL[stream]+8*k16,v_desc(Vstage,Vphase,k16),PV_IDESC,accumulate=((o_acc0 if stream==0 else o_acc1) or k16!=0))
                        # instruction_selection: tcgen05.mma.cta_group::1.kind::f16 for CG=1 or tcgen05.mma.cta_group::2.kind::f16 for CG=2; extent: one of eight final PV issues.
                    commit(oacc_pipe.full[stream])
                    # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 for CG=1 or tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 for CG=2; extent: one final O stream.
                    if stream==0: qstage1_spo_phase0^=1; o_acc0=true
                    else: qstage1_spo_phase1^=1; o_acc1=true
                    release(kv_pipe.empty,Vstage)
                    # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: final V.
                    advance kv consumer state

        # MMA role-local next-work protocol.
        wait(clc_pipe.full,clc_consumer_stage,clc_consumer_phase)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one response wait.
        response=load_shared_u128(OFF_CLC_RESPONSE)
        # instruction_selection: ld.shared.v2.b64; extent: one 128-bit response.
        canceled=query_cancel_is_canceled(response)
        next_x=query_cancel_get_first_ctaid_x(response)
        # instruction_selection: clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 and get_first_ctaid::x.b32.b128; extent: one response.
        fence_async_shared()
        # instruction_selection: fence.proxy.async.shared::cta; extent: one response.
        task_idx=next_x//CG; valid=(not canceled) and task_idx<num_fwd_work_records
        if valid:
            m_block=load_global_i32(fwd_work_desc,task_idx,0)
            head_idx=load_global_i32(fwd_work_desc,task_idx,1)
            batch_idx=load_global_i32(fwd_work_desc,task_idx,2)
            # instruction_selection: ld.global.b32; extent: three live descriptor fields; source field 3 is PTX-DCE.
        release(clc_pipe.empty,clc_consumer_stage)
        # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: one consumer warp.
        advance clc consumer state
        work=(m_block,head_idx,batch_idx,source_field3_dead,valid)

    relinquish_tmem(CG)
    # instruction_selection: tcgen05.relinquish_alloc_permit.cta_group::1 or ::2.sync.aligned; extent: allocator warp.
    named_barrier_sync(TMEM_PTR_NAMED,416)
    # instruction_selection: bar.sync; extent: thirteen participating warps.
    deallocate_tmem(tmem_base,512,CG)
    # instruction_selection: tcgen05.dealloc.cta_group::1 or ::2.sync.aligned.b32; extent: elected allocator.

# Warps 0..7: two independent score streams, four warps each.
if warp_id <= 7:
    stream=0 if warp_id<4 else 1
    local_warp=warp_id&3
    score_consumer_phase=0
    stats_producer_phase=1
    mask_consumer_stage=0; mask_consumer_phase=0
    clc_consumer_stage=0; clc_consumer_phase=0
    named_barrier_sync(TMEM_PTR_NAMED,416)
    # instruction_selection: bar.sync; extent: thirteen participating warps.

    initial_hw=hardware_initial_work()
    initial_task=initial_hw.cta_x//CG
    initial_valid=initial_hw.valid and initial_task<num_fwd_work_records
    if initial_valid:
        m_block=load_global_i32(fwd_work_desc,initial_task,0)
        head_idx=load_global_i32(fwd_work_desc,initial_task,1)
        batch_idx=load_global_i32(fwd_work_desc,initial_task,2)
        # instruction_selection: ld.global.b32; extent: three live initial descriptor fields; source field 3 is PTX-DCE.
    work=(m_block,head_idx,batch_idx,source_field3_dead,initial_valid)
    while work.valid:
        row_max=-inf; row_sum=0
        # This acquire is unconditional, including an empty tile, and its phase
        # flips before traversal. Each score step acquires the next stats slot.
        acquire(stats_pipe.empty[stream],stats_producer_phase)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one initial stats slot.
        stats_producer_phase ^= 1

        traversal = all entries for qstage2, or parity ordinals stream,stream+2,... for qstage1
        partial entries precede full entries; both compact lists run in reverse
        if no entry belongs to stream:
            named_barrier_arrive(SOFTMAX_STATS_NAMED_BASE+stream*4+local_warp)
            # instruction_selection: bar.arrive; extent: one empty score warp.
        for each selected entry with is_first=(local_stream_ordinal==0):
            if entry is partial:
                if USE_MASK_SMEM:
                    wait(mask_pipe[stream].full,mask_consumer_phase)
                    # instruction_selection: mbarrier.try_wait.parity.shared::cta.b64; extent: one consumer poll.
                    copy_s2r(mask_byte(stream,row,0),four_u32_mask_words)
                    # instruction_selection: ld.shared.v4.b32; extent: four words per row.
                    release(mask_pipe[stream].empty)
                    # instruction_selection: mbarrier.arrive.shared.b64; extent: one consumer warp.
                    advance mask consumer state
                else:
                    copy_g2r(mask_payload_pointer,four_u32_mask_words)
                    # instruction_selection: ld.global.v4.b32; extent: four words per row.

            wait(spo_pipe.full[stream],score_consumer_phase)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one S stage.
            if variant starts qstage1 and entry is full:
                copy_t2r(S_COL[stream],score_regs,hw_row_max)
                # instruction_selection: tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32; extent: one score band and reduction register.
                row_max_new=max_f32(hw_row_max,row_max) if not is_first else hw_row_max
                # instruction_selection: max.f32; extent: row fragments, no ftz.
                row_max_safe=row_max_new if row_max_new!=-inf else 0
                if is_first:
                    acc_scale=0
                else:
                    acc_scale_log2=(row_max-row_max_safe)*softmax_scale_log2
                    acc_scale=exp2_nonfast(acc_scale_log2)
                    # instruction_selection: ex2.approx.f32; extent: one row.
                    if acc_scale_log2>=-8:
                        row_max_new=row_max; row_max_safe=row_max; acc_scale=1
            else:
                copy_t2r(S_COL[stream],score_regs)
                # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x32.b32; extent: one score band.
                if entry is partial:
                    replace masked values with -inf using four mask words
                    # instruction_selection: setp and selp.b32; extent: 128 columns.
                local_max=software_warp_reduce_max_f32(score_regs,initial=(row_max if not is_first else -inf))
                # instruction_selection: max.f32 plus shfl.sync; extent: row fragments, no ftz.
                row_max_new=local_max; row_max_safe=row_max_new if row_max_new!=-inf else 0
                if is_first:
                    acc_scale=0
                else:
                    acc_scale_log2=(row_max-row_max_safe)*softmax_scale_log2
                    acc_scale=exp2_fast(acc_scale_log2)
                    # instruction_selection: ex2.approx.ftz.f32; extent: one row.
                    if acc_scale_log2>=-8:
                        row_max_new=row_max; row_max_safe=row_max; acc_scale=1
            row_max=row_max_new

            if not is_first:
                store_shared(scale_byte(stream,row,false),acc_scale)
                # instruction_selection: st.shared.b32; extent: one row scale.
            named_barrier_arrive(SOFTMAX_STATS_NAMED_BASE+stream*4+local_warp)
            # instruction_selection: bar.arrive; extent: one score warp.

            score_pairs=fma_packed_f32x2(
                score_pairs,
                (softmax_scale_log2,softmax_scale_log2),
                (-row_max_safe*softmax_scale_log2,-row_max_safe*softmax_scale_log2),
            )
            # instruction_selection: fma.rn.f32x2; extent: every packed score pair.
            exp2(score_regs,score_regs)
            # instruction_selection: ex2.approx.ftz.f32; extent: score fragments.
            cast(probability_pairs,score_regs,dtype)
            # instruction_selection: cvt.rn.bf16x2.f32 or cvt.rn.f16x2.f32; extent: packed pairs.
            for probability band in source order:
                copy_r2t(probability_pairs,P_COL[stream]+band)
                # instruction_selection: tcgen05.st.sync.aligned.32x32b.x16.b32; extent: one x16 band.
                if first 96 columns have just completed:
                    fence_tmem_store()
                    # instruction_selection: fence.proxy.async.shared::cta; extent: first 96 P columns.
                    release(spo_pipe.empty[stream])
                    # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: first P split.
            fence_tmem_store()
            # instruction_selection: fence.proxy.async.shared::cta; extent: final 32 P columns.
            sync_warp()
            # instruction_selection: bar.warp.sync; extent: one score warp.
            commit(plast_pipe.full[stream])
            # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: final P split.
            acquire(stats_pipe.empty[stream],stats_producer_phase)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: next stats slot for this score step.
            stats_producer_phase ^= 1
            row_sum=(0 if is_first else row_sum*acc_scale)+reduce_add(score_regs)
            # instruction_selection: mul/add plus shfl.sync; extent: one row.
            score_consumer_phase^=1

        if stream has work:
            store_shared(scale_byte(stream,row,false),row_sum)
            if return_lse or variant starts qstage1:
                store_shared(scale_byte(stream,row,true),row_max)
            # instruction_selection: st.shared.b32; extent: final sum and optional max.
            named_barrier_arrive(SOFTMAX_STATS_NAMED_BASE+stream*4+local_warp)
            # instruction_selection: bar.arrive; extent: one final score warp.

        # Score role-local next-work protocol.
        wait(clc_pipe.full,clc_consumer_stage,clc_consumer_phase)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one response wait.
        response=load_shared_u128(OFF_CLC_RESPONSE)
        # instruction_selection: ld.shared.v2.b64; extent: one 128-bit response.
        canceled=query_cancel_is_canceled(response)
        next_x=query_cancel_get_first_ctaid_x(response)
        # instruction_selection: clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 and get_first_ctaid::x.b32.b128; extent: one response.
        fence_async_shared()
        # instruction_selection: fence.proxy.async.shared::cta; extent: one response.
        task_idx=next_x//CG; valid=(not canceled) and task_idx<num_fwd_work_records
        if valid:
            m_block=load_global_i32(fwd_work_desc,task_idx,0)
            head_idx=load_global_i32(fwd_work_desc,task_idx,1)
            batch_idx=load_global_i32(fwd_work_desc,task_idx,2)
            # instruction_selection: ld.global.b32; extent: three live descriptor fields; source field 3 is PTX-DCE.
        release(clc_pipe.empty,clc_consumer_stage)
        # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: one consumer warp.
        advance clc consumer state
        work=(m_block,head_idx,batch_idx,source_field3_dead,valid)

    acquire final stats empty phase
    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: producer tail equivalent.
    named_barrier_arrive(TMEM_PTR_NAMED)
    # instruction_selection: bar.arrive; extent: score warp.

# Warps 8..11: correction and output production.
if 8 <= warp_id < 12:
    corr_tid=thread_id&127; local_warp=warp_id&3
    stats_consumer_phase=0
    qstage2_o_consumer_phase=0
    qstage2_epi_producer_phase=1
    qstage1_o_consumer_phase0=0; qstage1_o_consumer_phase1=0
    qstage1_epi_producer_stage=0; qstage1_epi_producer_phase=1
    load_epi_producer_stage=0; load_epi_producer_phase=1
    clc_consumer_stage=0; clc_consumer_phase=0
    for stream in 0,1:
        release(spo_pipe.empty[stream])
        # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: initial no-rescale publication for one stream.

    initial_hw=hardware_initial_work()
    initial_task=initial_hw.cta_x//CG
    initial_valid=initial_hw.valid and initial_task<num_fwd_work_records
    if initial_valid:
        m_block=load_global_i32(fwd_work_desc,initial_task,0)
        head_idx=load_global_i32(fwd_work_desc,initial_task,1)
        batch_idx=load_global_i32(fwd_work_desc,initial_task,2)
        # instruction_selection: ld.global.b32; extent: three live initial descriptor fields; source field 3 is PTX-DCE.
    work=(m_block,head_idx,batch_idx,source_field3_dead,initial_valid)

    while work.valid:
        total=partial_count+full_count
        if variant == qstage2_1cta:
            if total>0:
                named_barrier_sync(SOFTMAX_STATS_NAMED_BASE+local_warp,64)
                # instruction_selection: bar.sync; extent: initial stream-0 score/correction pair.
                release(stats_pipe.empty[0])
                # instruction_selection: mbarrier.arrive.shared.b64; extent: initial stream-0 stats slot.
                named_barrier_sync(SOFTMAX_STATS_NAMED_BASE+4+local_warp,64)
                # instruction_selection: bar.sync; extent: initial stream-1 pair, with its stats slot held.
                stats_consumer_phase^=1

                for block in range(total-1):
                    for stage in 0,1:
                        named_barrier_sync(SOFTMAX_STATS_NAMED_BASE+stage*4+local_warp,64)
                        # instruction_selection: bar.sync; extent: one score/correction pair.
                        scale=load_shared(scale_byte(stage,row,false))
                        # instruction_selection: ld.shared.b32; extent: one row scale.
                        if vote_any(scale<1):
                            for band in each 16-column output band:
                                copy_t2r(O_COL[stage]+band,output_regs)
                                # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x16.b32; extent: one band.
                                mul_packed_f32x2(output_regs,scale)
                                # instruction_selection: mul.rn.f32x2; extent: one band.
                                copy_r2t(output_regs,O_COL[stage]+band)
                                # instruction_selection: tcgen05.st.sync.aligned.32x32b.x16.b32; extent: one band.
                            fence_tmem_store()
                            # instruction_selection: fence.proxy.async.shared::cta; extent: one rescaled O stream.
                        release(spo_pipe.empty[stage])
                        # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: one SPO stream.
                        release(stats_pipe.empty[1-stage])
                        # instruction_selection: mbarrier.arrive.shared.b64; extent: reversed-index stats slot.
                    stats_consumer_phase^=1
                release(stats_pipe.empty[1])
                # instruction_selection: mbarrier.arrive.shared.b64; extent: initially held/final stream-1 slot.

                for stage in 0,1:
                    named_barrier_sync(SOFTMAX_STATS_NAMED_BASE+stage*4+local_warp,64)
                    # instruction_selection: bar.sync; extent: final score/correction pair.
                    row_sum=load_shared(scale_byte(stage,row,false))
                    row_max=load_shared(scale_byte(stage,row,true)) if return_lse else none
                    # instruction_selection: ld.shared.b32; extent: final sum and optional max.
                    release(stats_pipe.empty[stage])
                    # instruction_selection: mbarrier.arrive.shared.b64; extent: final stats slot.
                    invalid=(row_sum==0) or isnan(row_sum)
                    inv_sum=reciprocal(row_sum if not invalid else 1)
                    # instruction_selection: rcp.approx.ftz.f32; extent: one row.
                    wait(oacc_pipe.full[stage],qstage2_o_consumer_phase)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one final O stream.
                    if USE_TMA_O:
                        acquire(oepi_pipe.empty[stage],qstage2_epi_producer_phase)
                        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one output stage.
                    for band in each 16-column output band:
                        copy_t2r(O_COL[stage]+band,output_regs)
                        # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x16.b32; extent: one band.
                        mul_packed_f32x2(output_regs,inv_sum)
                        # instruction_selection: mul.rn.f32x2; extent: one band.
                        cast(output_pairs,output_regs,dtype)
                        # instruction_selection: cvt.rn.bf16x2.f32 or cvt.rn.f16x2.f32; extent: packed pairs.
                        copy_r2s(output_pairs,o_byte(stage,row,band))
                        # instruction_selection: st.shared.v4.b32; extent: one 128-bit output vector.
                    fence_shared()
                    # instruction_selection: fence.proxy.async.shared::cta; extent: one output stream.
                    release(spo_pipe.empty[stage])
                    # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: one O reuse signal.
                    if USE_TMA_O:
                        commit(oepi_pipe.full[stage])
                        # instruction_selection: mbarrier.arrive.shared.b64; extent: one fixed-output stage.
                    else:
                        named_barrier_sync(EPILOGUE_NAMED,128)
                        # instruction_selection: bar.sync; extent: four correction warps.
                        copy_s2r(output_vector,output_regs)
                        # instruction_selection: ld.shared.v4.b32; extent: one 128-bit vector.
                        copy_r2g(output_regs,varlen_output_pointer,row_and_Dv_predicate)
                        # instruction_selection: st.global.v4.b32; extent: one predicated vector.
                qstage2_o_consumer_phase^=1
                stats_consumer_phase^=1
                qstage2_epi_producer_phase^=1
            else:
                for stage in 0,1:
                    store_shared(row_sum[stage],1)
                    # instruction_selection: st.shared.b32; extent: synthetic row sum.
                    if return_lse:
                        store_shared(row_max[stage],-inf)
                        # instruction_selection: st.shared.b32; extent: synthetic row max.
                    stats[stage]=(1,-inf,false)
                    named_barrier_sync(SOFTMAX_STATS_NAMED_BASE+stage*4+local_warp,64)
                    # instruction_selection: bar.sync; extent: synthetic score/correction pair.
                    release(stats_pipe.empty[stage])
                    # instruction_selection: mbarrier.arrive.shared.b64; extent: synthetic stats slot.
                    if USE_TMA_O:
                        acquire(oepi_pipe.empty[stage],qstage2_epi_producer_phase)
                        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one empty-tile output stage.
                    for band in each 16-column output band:
                        copy_t2r(O_COL[stage]+band,output_regs)
                        # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x16.b32; extent: one band.
                        mul_packed_f32x2(output_regs,0)
                        # instruction_selection: mul.rn.f32x2; extent: zero-scale one band.
                        cast(output_pairs,output_regs,dtype)
                        # instruction_selection: cvt.rn.bf16x2.f32 or cvt.rn.f16x2.f32; extent: packed pairs.
                        copy_r2s(output_pairs,o_byte(stage,row,band))
                        # instruction_selection: st.shared.v4.b32; extent: one zero vector.
                    fence_shared()
                    # instruction_selection: fence.proxy.async.shared::cta; extent: one empty output stream.
                    if USE_TMA_O:
                        commit(oepi_pipe.full[stage])
                        # instruction_selection: mbarrier.arrive.shared.b64; extent: one empty fixed-output stage.
                    else:
                        named_barrier_sync(EPILOGUE_NAMED,128)
                        # instruction_selection: bar.sync; extent: four correction warps.
                        copy_s2r(output_vector,output_regs)
                        # instruction_selection: ld.shared.v4.b32; extent: one 128-bit vector.
                        copy_r2g(output_regs,varlen_output_pointer,row_and_Dv_predicate)
                        # instruction_selection: st.global.v4.b32; extent: one predicated zero vector.
                stats_consumer_phase^=1
                qstage2_epi_producer_phase^=1
                # qstage2_o_consumer_phase is unchanged; no P or O-acc operation occurs.

            if return_lse:
                for stage in 0,1:
                    lse=((row_max*softmax_scale_log2+log2_fast(row_sum))*ln2 if valid_stats else -inf)
                    # instruction_selection: lg2.approx.ftz.f32 plus fma/mul; extent: one row.
                    copy_r2g(lse,LSE_pointer,valid_row)
                    # instruction_selection: st.global.b32; extent: one valid row.
        else:
            count0=(total+1)//2; count1=total//2
            stats=[(0,-inf,true),(0,-inf,true)]
            for stream in 0,1:
                named_barrier_sync(SOFTMAX_STATS_NAMED_BASE+stream*4+local_warp,64)
                # instruction_selection: bar.sync; extent: initial stream pair.
                release(stats_pipe.empty[stream])
                # instruction_selection: mbarrier.arrive.shared.b64; extent: initial stats slot.
                if (count0 if stream==0 else count1)==0:
                    stats[stream]=(0,-inf,true)

            for ordinal in range(1,count0):
                if ordinal<count0:
                    named_barrier_sync(SOFTMAX_STATS_NAMED_BASE+local_warp,64)
                    # instruction_selection: bar.sync; extent: stream-0 pair.
                    scale=load_shared(scale_byte(0,row,false))
                    # instruction_selection: ld.shared.b32; extent: stream-0 scale.
                    if vote_any(scale<1):
                        for band in each 16-column output band:
                            copy_t2r(O_COL[0]+band,output_regs)
                            # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x16.b32; extent: one band.
                            mul_packed_f32x2(output_regs,scale)
                            # instruction_selection: mul.rn.f32x2; extent: one band.
                            copy_r2t(output_regs,O_COL[0]+band)
                            # instruction_selection: tcgen05.st.sync.aligned.32x32b.x16.b32; extent: one band.
                        fence_tmem_store()
                        # instruction_selection: fence.proxy.async.shared::cta; extent: stream 0.
                    release(spo_pipe.empty[0])
                    # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: stream-0 SPO.
                    release(stats_pipe.empty[0])
                    # instruction_selection: mbarrier.arrive.shared.b64; extent: stream-0 stats.
                if ordinal<count1:
                    named_barrier_sync(SOFTMAX_STATS_NAMED_BASE+4+local_warp,64)
                    # instruction_selection: bar.sync; extent: stream-1 pair.
                    scale=load_shared(scale_byte(1,row,false))
                    # instruction_selection: ld.shared.b32; extent: stream-1 scale.
                    if vote_any(scale<1):
                        for band in each 16-column output band:
                            copy_t2r(O_COL[1]+band,output_regs)
                            # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x16.b32; extent: one band.
                            mul_packed_f32x2(output_regs,scale)
                            # instruction_selection: mul.rn.f32x2; extent: one band.
                            copy_r2t(output_regs,O_COL[1]+band)
                            # instruction_selection: tcgen05.st.sync.aligned.32x32b.x16.b32; extent: one band.
                        fence_tmem_store()
                        # instruction_selection: fence.proxy.async.shared::cta; extent: stream 1.
                    release(spo_pipe.empty[1])
                    # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: stream-1 SPO.
                    release(stats_pipe.empty[1])
                    # instruction_selection: mbarrier.arrive.shared.b64; extent: stream-1 stats.

            for stream in 0,1:
                count=count0 if stream==0 else count1
                if count>0:
                    named_barrier_sync(SOFTMAX_STATS_NAMED_BASE+stream*4+local_warp,64)
                    # instruction_selection: bar.sync; extent: final stream pair.
                    row_sum=load_shared(scale_byte(stream,row,false))
                    row_max=load_shared(scale_byte(stream,row,true))
                    # instruction_selection: ld.shared.b32; extent: final sum and max.
                    release(stats_pipe.empty[stream])
                    # instruction_selection: mbarrier.arrive.shared.b64; extent: final stats slot.
                    invalid=(row_sum<=0) or isnan(row_sum)
                    stats[stream]=(row_sum,row_max,invalid)

            row_sum0,row_max0,invalid0=stats[0]
            row_sum1,row_max1,invalid1=stats[1]
            row_max0=row_max0 if not invalid0 else -inf
            row_max1=row_max1 if not invalid1 else -inf
            combined_max=max_f32(row_max0,row_max1)
            # instruction_selection: max.f32; extent: one row.
            safe_max=combined_max if combined_max!=-inf else 0
            scale0=exp2_fast((row_max0-safe_max)*softmax_scale_log2) if not invalid0 else 0
            scale1=exp2_fast((row_max1-safe_max)*softmax_scale_log2) if not invalid1 else 0
            # instruction_selection: ex2.approx.ftz.f32; extent: each valid stream.
            combined_sum=row_sum0*scale0+row_sum1*scale1
            # instruction_selection: fma.rn.f32 plus mul.rn.f32; extent: one row.
            combined_invalid=(combined_sum<=0) or isnan(combined_sum)
            inv_sum=reciprocal(combined_sum if not combined_invalid else 1)
            # instruction_selection: rcp.approx.ftz.f32; extent: one row.
            final_scale0=scale0*inv_sum; final_scale1=scale1*inv_sum
            # instruction_selection: mul.rn.f32; extent: two row scalars.

            if count0>0:
                wait(oacc_pipe.full[0],qstage1_o_consumer_phase0)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: O stream 0.
            if count1>0:
                wait(oacc_pipe.full[1],qstage1_o_consumer_phase1)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: O stream 1.
            output_stage=0
            if USE_TMA_O:
                acquire(oepi_pipe.empty[qstage1_epi_producer_stage],qstage1_epi_producer_phase)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: current qstage1 output stage.
                output_stage=qstage1_epi_producer_stage

            for band in each 16-column output band:
                copy_t2r(O_COL[0]+band,regs0)
                # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x16.b32; extent: stream-0 band.
                copy_t2r(O_COL[1]+band,regs1)
                # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x16.b32; extent: stream-1 band.
                if final_scale0!=0:
                    out=mul_packed_f32x2(regs0,final_scale0)
                    # instruction_selection: mul.rn.f32x2; extent: stream-0 band.
                else:
                    fill(out,0)
                    # instruction_selection: mov.b32; extent: output registers, avoiding NaN*0.
                if final_scale1!=0:
                    tmp=mul_packed_f32x2(regs1,final_scale1)
                    # instruction_selection: mul.rn.f32x2; extent: stream-1 band.
                    out=add_packed_f32x2(out,tmp)
                    # instruction_selection: add.rn.f32x2; extent: combined band.
                cast(output_pairs,out,dtype)
                # instruction_selection: cvt.rn.bf16x2.f32 or cvt.rn.f16x2.f32; extent: packed pairs.
                copy_r2s(output_pairs,o_byte(output_stage,row,band))
                # instruction_selection: st.shared.v4.b32; extent: one 128-bit vector.
            fence_shared()
            # instruction_selection: fence.proxy.async.shared::cta; extent: combined output.
            if count0>0:
                release(spo_pipe.empty[0])
                # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: stream-0 O reuse.
                qstage1_o_consumer_phase0^=1
            if count1>0:
                release(spo_pipe.empty[1])
                # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: stream-1 O reuse.
                qstage1_o_consumer_phase1^=1

            if USE_TMA_O:
                commit(oepi_pipe.full[qstage1_epi_producer_stage])
                # instruction_selection: mbarrier.arrive.shared.b64; extent: current qstage1 output stage.
                advance qstage1 output producer state
            else:
                named_barrier_sync(EPILOGUE_NAMED,128)
                # instruction_selection: bar.sync; extent: four correction warps.
                copy_s2r(output_vector,output_regs)
                # instruction_selection: ld.shared.v4.b32; extent: one 128-bit vector.
                copy_r2g(output_regs,varlen_output_pointer,row_and_Dv_predicate)
                # instruction_selection: st.global.v4.b32; extent: one predicated vector.

            if return_lse:
                lse=((safe_max*softmax_scale_log2+log2_fast(combined_sum))*ln2 if not combined_invalid else -inf)
                # instruction_selection: lg2.approx.ftz.f32 plus fma/mul; extent: one row.
                copy_r2g(lse,LSE_pointer,valid_row)
                # instruction_selection: st.global.b32; extent: one valid row.

        if OVERLAP_QO and direct output:
            acquire(load_epi_pipe.empty[load_epi_producer_stage],load_epi_producer_phase)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: direct-output alias stage.
            commit(load_epi_pipe.full[load_epi_producer_stage])
            # instruction_selection: mbarrier.arrive.shared.b64; extent: completed direct output.
            advance load-epi producer state

        # Correction role-local next-work protocol.
        wait(clc_pipe.full,clc_consumer_stage,clc_consumer_phase)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one response wait.
        response=load_shared_u128(OFF_CLC_RESPONSE)
        # instruction_selection: ld.shared.v2.b64; extent: one 128-bit response.
        canceled=query_cancel_is_canceled(response)
        next_x=query_cancel_get_first_ctaid_x(response)
        # instruction_selection: clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 and get_first_ctaid::x.b32.b128; extent: one response.
        fence_async_shared()
        # instruction_selection: fence.proxy.async.shared::cta; extent: one response.
        task_idx=next_x//CG; valid=(not canceled) and task_idx<num_fwd_work_records
        if valid:
            m_block=load_global_i32(fwd_work_desc,task_idx,0)
            head_idx=load_global_i32(fwd_work_desc,task_idx,1)
            batch_idx=load_global_i32(fwd_work_desc,task_idx,2)
            # instruction_selection: ld.global.b32; extent: three live descriptor fields; source field 3 is PTX-DCE.
        release(clc_pipe.empty,clc_consumer_stage)
        # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: one consumer warp.
        advance clc consumer state
        work=(m_block,head_idx,batch_idx,source_field3_dead,valid)

    if USE_TMA_O and variant==qstage2_1cta:
        acquire(oepi_pipe.empty[1],qstage2_epi_producer_phase)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: qstage2 output-producer tail.
    if USE_TMA_O and variant starts qstage1:
        acquire(oepi_pipe.empty[qstage1_epi_producer_stage],qstage1_epi_producer_phase)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: qstage1 output-producer tail.
    named_barrier_arrive(TMEM_PTR_NAMED)
    # instruction_selection: bar.arrive; extent: correction warp.

# Warp 13: fixed-length shared-to-global output.
if warp_id == 13 and USE_TMA_O:
    qstage2_epi_consumer_phase=0
    qstage1_epi_consumer_stage=0; qstage1_epi_consumer_phase=0
    load_epi_producer_stage=0; load_epi_producer_phase=1
    clc_consumer_stage=0; clc_consumer_phase=0

    initial_hw=hardware_initial_work()
    initial_task=initial_hw.cta_x//CG
    initial_valid=initial_hw.valid and initial_task<num_fwd_work_records
    if initial_valid:
        m_block=load_global_i32(fwd_work_desc,initial_task,0)
        head_idx=load_global_i32(fwd_work_desc,initial_task,1)
        batch_idx=load_global_i32(fwd_work_desc,initial_task,2)
        # instruction_selection: ld.global.b32; extent: three live initial descriptor fields; source field 3 is PTX-DCE.
    work=(m_block,head_idx,batch_idx,source_field3_dead,initial_valid)

    while work.valid:
        if variant == qstage2_1cta:
            for stream in 0,1:
                wait(oepi_pipe.full[stream],qstage2_epi_consumer_phase)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one qstage2 output stage.
                copy_s2g(Omap,o_coordinates(stream),o_byte(stream,0,0))
                # instruction_selection: cp.async.bulk.tensor.<O_RANK>d.global.shared::cta.tile.bulk_group.L2::cache_hint; extent: one output box.
                commit_bulk_group()
                # instruction_selection: cp.async.bulk.commit_group; extent: one output box.
            for stream in 0,1:
                wait_bulk_group_read(1-stream)
                # instruction_selection: cp.async.bulk.wait_group.read; extent: source overlap depth for one stream.
                release(oepi_pipe.empty[stream])
                # instruction_selection: mbarrier.arrive.shared.b64; extent: one qstage2 output stage.
            qstage2_epi_consumer_phase^=1
        else:
            wait(oepi_pipe.full[qstage1_epi_consumer_stage],qstage1_epi_consumer_phase)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: current qstage1 output stage.
            copy_s2g(Omap,o_coordinates(0),o_byte(qstage1_epi_consumer_stage,0,0))
            # instruction_selection: cp.async.bulk.tensor.<O_RANK>d.global.shared::cta.tile.bulk_group.L2::cache_hint; extent: one output box.
            commit_bulk_group()
            # instruction_selection: cp.async.bulk.commit_group; extent: one output box.
            wait_bulk_group_read(0)
            # instruction_selection: cp.async.bulk.wait_group.read; extent: current output box.
            release(oepi_pipe.empty[qstage1_epi_consumer_stage])
            # instruction_selection: mbarrier.arrive.shared.b64; extent: current qstage1 output stage.
            advance qstage1 output consumer state

        if OVERLAP_QO:
            acquire(load_epi_pipe.empty[load_epi_producer_stage],load_epi_producer_phase)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: fixed-output alias stage.
            commit(load_epi_pipe.full[load_epi_producer_stage])
            # instruction_selection: mbarrier.arrive.shared.b64; extent: completed fixed output.
            advance load-epi producer state

        # Epilogue role-local next-work protocol.
        wait(clc_pipe.full,clc_consumer_stage,clc_consumer_phase)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent: one response wait.
        response=load_shared_u128(OFF_CLC_RESPONSE)
        # instruction_selection: ld.shared.v2.b64; extent: one 128-bit response.
        canceled=query_cancel_is_canceled(response)
        next_x=query_cancel_get_first_ctaid_x(response)
        # instruction_selection: clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 and get_first_ctaid::x.b32.b128; extent: one response.
        fence_async_shared()
        # instruction_selection: fence.proxy.async.shared::cta; extent: one response.
        task_idx=next_x//CG; valid=(not canceled) and task_idx<num_fwd_work_records
        if valid:
            m_block=load_global_i32(fwd_work_desc,task_idx,0)
            head_idx=load_global_i32(fwd_work_desc,task_idx,1)
            batch_idx=load_global_i32(fwd_work_desc,task_idx,2)
            # instruction_selection: ld.global.b32; extent: three live descriptor fields; source field 3 is PTX-DCE.
        release(clc_pipe.empty,clc_consumer_stage)
        # instruction_selection: mbarrier.arrive.shared::cluster.b64; extent: one consumer warp.
        advance clc consumer state
        work=(m_block,head_idx,batch_idx,source_field3_dead,valid)
~~~

## TensorMap fields and global scalar mappings

TensorMap ranks are specialization-constant and match the source exactly:
fixed unpacked Q/K/V/O are rank 4; fixed packed Q/O are rank 5 while K/V remain
rank 4; true-varlen unpacked Q/K/V are rank 3; true-varlen packed Q is rank 4
while K/V remain rank 3. True-varlen output never constructs an O TensorMap.
Packing adds the local query-head ratio dimension only to Q/O; the sparse plan
remains indexed by its planned mask-head rule. V is feature-major for the PV B
operand. Boxes cover one logical 128-row tile and one physical feature band;
group-2 copies add cluster addressing/multicast. TensorMap OOB fill supplies
zero for D/Dv padding and boundary rows, fixed output drops padding, and
true-varlen output uses row and Dv predicates on 128-bit stores.

Global element addresses are ordinary scalar equations:

~~~python
fixed_q(b,s,h,d) = (((b*Sq+s)*Hq+h)*D+d)*2
fixed_k(b,s,h,d) = (((b*Sk+s)*Hkv+h)*D+d)*2
fixed_v(b,s,h,dv)= (((b*Sk+s)*Hkv+h)*Dv+dv)*2
fixed_o(b,s,h,dv)= (((b*Sq+s)*Hq+h)*Dv+dv)*2
var_q(t,h,d)     = ((t*Hq+h)*D+d)*2
var_k(t,h,d)     = ((t*Hkv+h)*D+d)*2
var_v(t,h,dv)    = ((t*Hkv+h)*Dv+dv)*2
var_o(t,h,dv)    = ((t*Hq+h)*Dv+dv)*2
kv_head = planned_head if pack_gqa else query_head//qhead_per_kvhead
packed_row = token*qhead_per_kvhead + local_query_head
~~~

## Pipeline and barrier inventory

| object | stages | initial producer/consumer ownership | transaction / arrival meaning |
| --- | ---: | --- | --- |
| Q full/empty | QST | warp14 / warp12 | exact Q TensorMap box, multiplied by CG |
| K/V full/empty | KVST | warp14 / warp12 | K bytes are base expectation; V adjusts expected bytes when Dvp differs |
| mask0,mask1 | 1 each, group2 only | warp14 / four score warps | one 2048-byte payload per partial traversal |
| S/P/O full/empty | 2 | warp12 / four score plus four correction warps per stream and per CTA | S full; empty only after first 96 P columns and prior O rescale/read |
| P-last full/empty | 2 | four score warps per CTA / warp12 | remaining 32 P columns visible before PV issue six |
| O-acc full/empty | 2 | warp12 / correction warps across cluster | final PV accumulator ready |
| stats full/empty | 2 | 128 score threads / 128 correction threads | row max/sum and rescale coefficient |
| output full/empty | OST, fixed only | correction / warp13 | shared output safe for TMA and reuse |
| load/output alias | 1, D192 overlap only | output side / warp14 | output no longer occupies aliased Q bytes |
| CLC | 1 | leader warp15 lane / all warps across cluster | one 16-byte work response |
| TMEM pointer/dealloc | named id 2 plus optional cluster mbarrier | warp12 / 13 participating warps | 512-column allocation publication and teardown |
| paired stats | named ids 3..10 | one score warp / matching correction warp | per-stream per-local-warp online-stat handoff |

Every stage/phase advances exactly once per corresponding source event.
Producer tails drain only the source rings that expose tails. The MMA output
rings deliberately have no invented tail acquire.

## Source coverage map

| source region | sketch region | semantics |
| --- | --- | --- |
| forward.py:49-146 | scope/static values, roles, TMEM map | padding, variant topology, warp ids, register budgets, TMEM columns |
| forward.py:148-246 | specialization and rank-one arena | capacity-derived Q/KV/O sizes, caps, D192 exception |
| forward.py:248-418 | ABI and TensorMap section | fixed/varlen transforms, pack GQA, TensorMaps, output path, CLC grid |
| forward.py:420-513 | arena and launch | source struct order, alignments, dynamic size, 512-thread launch |
| forward.py:515-799 | prologue, arena, pipes, CLC | descriptor prefetch, barrier groups, aliases, cluster and scheduler construction |
| forward.py:801-960 | source-order role selection | scheduler/load/MMA/epilogue/score/correction ownership, TMEM teardown |
| forward.py:963-1101 | warp14 body | work coordinates, Q/K/V TMA construction, tails, Q/output reuse |
| forward.py:1105-1284 | score warp body | per-stream traversal, final sum/max publication, empty stream |
| forward.py:1287-1386 | inner score entry loop | S load/ld.red, mask, max, exp2, P split, row sum |
| forward.py:1389-1433 | correction rescale loop | x16 TMEM load/multiply/store and fence |
| forward.py:1436-1558 | correction output band and direct output | normalize/cast/shared staging, packed/varlen predicates |
| forward.py:1561-1643 | warp13 body | qstage1 one-at-a-time and qstage2 batched TMA stores |
| forward.py:1646-1721 | CLC and TMA helpers | next-work prefetch, empty consumption, acquire/expect/copy, uneven stage |
| forward_qstage1.py:61-151 | qstage1 static branch and warp14 feed | group1/group2 tuning, mask-pipe enable, N-direction producer |
| forward_qstage1.py:154-368 | qstage1 correction | stream counts, rescale, max/sum merge, LSE and output |
| forward_qstage1.py:370-453 | qstage1 output band loop | two collective O loads, scaled add, cast/store |
| forward_qstage1.py:456-705 | qstage1 MMA | K0/K1 prologue, paired/odd middle, overlap choice, final PV/O commits |
| forward_qstage2.py:30-84 | qstage2 static branch and producer | two-Q-stage compact-plan producer |
| forward_qstage2.py:86-293 | qstage2 MMA | two QK streams per K and two PV streams per V |
| forward_qstage2.py:295-487 | qstage2 correction | independent stream rescale/finalize and empty tile |
| packed_mask.py:851-965 | qstage2 warp14 traversal | reverse partial then reverse full K/V interleave |
| packed_mask.py:968-1188 | qstage1 warp14 traversal | parity mapping, packed-mask copies, Q/K/K then V/K tail |
| packed_mask.py:1193-1300 | count and empty handling | total block count and zero-output/-inf-LSE protocol |
| packed_mask.py:1303-1426 | qstage2 score traversal | partial masked entries then full entries |
| packed_mask.py:1430-1560 | qstage1 score traversal | even/odd streams, SMEM/global masks, full-block continuation |
| pipeline.py and tile_scheduler.py reachable CLC helpers | pipeline inventory and warp15 loop | full/empty phase protocol and 16-byte work-record scheduling |
| softmax.py reachable SoftmaxSm100 methods | explicit score operations | online max/scale, native exp2, row-sum recurrence |

Every semantic sketch region maps back to at least one row above. Incidental
pointer calculation, descriptor bit assembly, and temporary scalar plumbing
remain inside the corresponding movement or compute row.

## TIRx integration, correctness, and benchmark contract

The target module exposes one uniquely named kernel family through get_kernel,
CONFIGS, prepare_data, run_test, and run_bench. It constructs the production
kernel only through K. The harness creates reusable plans outside the timed
closure, gives source and TIRx separate output/workspace state, and uses the
same logical Q/K/V, plan arrays, scale, and optional LSE mode.

Correctness includes both dtypes; D/Dv padding boundaries; D192/Dv128 aliasing;
qstage2, qstage1 narrow and overlap, and qstage1 group2; zero, one, odd, and
even selected-block counts; all-partial, all-full, and mixed multi-interval
plans; empty rows; sequence tails; MHA/GQA/MQA; packed and unpacked GQA;
fixed/varlen; and LSE on/off. Comparisons use exact finiteness and -inf checks,
zero tolerance for masked/empty invariants where representable, and the
tightest empirically defensible FP16/BF16 ULP/absolute bounds against the
same source execution.

The curated bench-suite matrix is the sole final performance authority. Source
and TIRx both compile as PTX 9.3 on GB300. The timed closures include only the
production source forward launch or the production TIRx launch; plan building,
TensorMap encoding, compilation, allocation, initialization, validation, and
synchronization setup remain outside. Completion requires one final complete
bench-suite run with source_time/tirx_time greater than 0.99 for every required
configuration.

## Instruction-selection summary

- TensorMap loads are rank-4/rank-5 fixed or rank-3/rank-4 varlen
  cp.async.bulk.tensor transfers; group2 adds cta_group::2 and cluster
  multicast semantics. Fixed output is tensor bulk S2G; group2 mask payload
  uses raw 2048-byte bulk copies.
- QK and PV remain tcgen05.mma.cta_group::1/::2.kind::f16. QK issues Dp/16 K16
  phases; PV issues eight K16 phases and waits at the source's 96-column split.
  BF16 versus FP16 and Dvp change instruction-descriptor bits, not the source
  role split or issue ordering.
- qstage1 full blocks use
  tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32 on SM103; partial blocks and
  all qstage2 blocks use ordinary x32 score loads. Correction and epilogue use
  x16 TMEM loads and x16 stores only. Output staging uses st.shared.v4.b32;
  direct output uses ld.shared.v4.b32 and st.global.v4.b32.
- Online row maxima use max.f32 (no ftz). Non-fast full/ld-red rescaling uses
  ex2.approx.f32; the masked/software path and score exponentials use
  ex2.approx.ftz.f32. Final normalization uses rcp.approx.ftz.f32; optional LSE
  uses lg2.approx.ftz.f32.
- All asynchronous visibility and reuse points remain explicit
  mbarrier.try_wait, arrive/expect, tcgen05 commit, proxy fence, named barrier,
  bulk commit/wait, cluster barrier, CLC query, and TMEM allocate/relinquish/
  deallocate instructions.
