<!--
This file is a design sketch for a TIRx port of code from cuDNN Frontend
(https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5),
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 AND MIT
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# cuDNN SM100 blk128 block-sparse attention forward: coarse WASP pipeline sketch

This is the non-executable, mechanically translatable sketch for
[`tirx_kernels/cudnn/bsa/block_sparse_attention_forward_sm100_blk128.py`](../../../tirx_kernels/cudnn/bsa/block_sparse_attention_forward_sm100_blk128.py).
The authoritative source is
`python/cudnn/block_sparse_attention/csrc/fwd/sm100_blk128/bsa_fwd_sm100.py`
at commit `aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5`, SHA256
`d991ad23d01bcf0fa28c5982f3aa16c159ac84bb72b33beff9089ce77984cca9`.
The wrapper and scheduler/softmax/pack helpers are MIT; the main file is
Apache-2.0. The port targets SM100 only.

First-class layouts are forbidden. Every shared-memory object below is a byte
range in one rank-one `u8` arena. Every alias, swizzle, stage, descriptor and
TMEM fragment is selected by scalar integer arithmetic. “Tile” means a logical
rectangle only; it is not a TIRx tile primitive or a layout-bearing value.

## Scope and specialization boundary

| axis | in-scope values | emitted-code consequence |
| --- | --- | --- |
| input/output dtype | BF16, FP16 | conversion opcode and QK/PV instruction descriptor |
| head dimension | 64, 96, 128 | 11/6/4 KV stages, 4/6/8 QK instructions, PV descriptor and SMEM band width |
| attention family | MHA, GQA, MQA | Q-head/KV-head mapping; ratios dividing 128 default to packed GQA |
| packed GQA | true/false | packed row is `token*ratio+local_qhead`; only Q/O TensorMaps become rank 5; sparse metadata keeps its rank |
| sparse count | fixed even `>=2`; variable odd/even; optionally zero | odd counts round up to even; a phantom entry clamps to the last live sparse id and receives block size zero |
| block sizes | present/absent | 128-column predicate mask or a constant full block |
| LSE | present/absent | correction emits or omits the scalar FP32 store |
| Q/O presentation | BHSD/BSHD at wrapper | wrapper normalizes to direct BHSD before the timed kernel |
| KV address width | i32/i64 | TensorMap stride/address arithmetic only |

Out of scope: causal attention, head dimensions other than 64/96/128, mixed
input dtypes, Dv different from D, split-KV, block sizes other than 128,
multi-CTA MMA, non-singleton clusters, and wrapper-level transposed device
storage.

## Writer line-info evidence

Sixteen independently compiled and executed source variants are preserved at
`.porting/block_sparse_attention_forward_sm100_blk128/writer_exports/`; the
manifest is `writer_exports_manifest.json`. Every export targets `sm_100a`,
declares `.reqntid 512,1,1` and `.minnctapersm 1`, has usable `.file/.loc`, and
passes the independent FP32 oracle.

| family | total SMEM | KV stages | QK idesc BF16/FP16 | PV idesc BF16/FP16 | descriptor high word |
| --- | ---: | ---: | --- | --- | --- |
| D64 | 232448 | 11 | `0x08200490` / `0x08200010` | `0x08110490` / `0x08110010` | `0x40004040` |
| D96 | 224256 | 6 | same | `0x08190490` / `0x08190010` | `0x80004020` |
| D128 | 232448 | 4 | same | `0x08210490` / `0x08210010` | `0x40004040` |

MHA uses rank-4 Q/O tensor-bulk instructions. Packed GQA/MQA uses rank-5 Q/O;
ratio-3 automatic fallback is non-packed and returns to rank 4. K/V are rank 4.
The D128 anchor contains 64 `tcgen05.mma...kind::f16`, eight x32 score loads,
32 x16 correction loads, 24 x16 TMEM stores, 15 TMEM commits, and the raw CLC
query/try-cancel/coordinate-extraction family. Counts are static emitted counts.

## Fixed storage and integer maps

```python
P = specialize(dtype in {bf16,f16}, D in {64,96,128}, ratio>=1,
               PACK=(ratio>1 and 128%ratio==0 unless explicitly disabled),
               HAS_SIZES, HAS_NUMS, ALLOW_EMPTY, RETURN_LSE, I64_KV)
M=N=128; Q_STAGES=1; S_STAGES=2
KV_STAGES={64:11,96:6,128:4}[D]
THREADS=512; WARPS=16; SPLIT_P=96; TMEM_COLS=512
SMEM_BYTES={64:232448,96:224256,128:232448}[D]
REG_OTHER={64:48,96:56,128:56}[D]
REG_SOFTMAX={64:200,96:184,128:184}[D]
REG_CORRECTION={64:64,96:88,128:88}[D]

SCHED_HEADS = Hkv if P.PACK else Hq
Q_ROWS = SQ*ratio if P.PACK else SQ
Q_BLOCKS = ceil_div(Q_ROWS,128)
launch(grid=(Q_BLOCKS,SCHED_HEADS,B), block=(512,1,1), cluster=(1,1,1),
       min_blocks_per_sm=1, dynamic_smem_bytes=SMEM_BYTES, target="sm_100a")
# instruction_selection: `.reqntid 512,1,1`, `.minnctapersm 1`, singleton
# cluster launch, `.extern .shared .align 1024`; extent: one specialization.

ABI=(Q_dtype[B,Hq,SQ,D], K_dtype[B,Hkv,SKV,D], V_dtype[B,Hkv,SKV,D],
     O_dtype[B,Hq,SQ,D], optional_LSE_f32[B,Hq,SQ], block_index_i32,
     optional_block_sizes_i32, fixed_count_i32, optional_block_nums_i32,
     softmax_scale_log2_f32, Qmap, Kmap, Vmap, Omap)

arena = linear_smem_u8(SMEM_BYTES, alignment=1024)
# D64 / D96 / D128 exact byte offsets from the fresh exports.
OFF = {
  64:  dict(Q=0,KV=16,SPO=192,PLAST=224,OACC=256,STATS=288,OEPI=320,
            TMEM=352,SCALE=356,CLC=2408,RESP=2432,sO=3072,sQ=35840,sKV=52224),
  96:  dict(Q=0,KV=16,SPO=112,PLAST=144,OACC=176,STATS=208,OEPI=240,
            TMEM=272,SCALE=276,CLC=2328,RESP=2352,sO=3072,sQ=52224,sKV=76800),
  128: dict(Q=0,KV=16,SPO=80, PLAST=112,OACC=144,STATS=176,OEPI=208,
            TMEM=240,SCALE=244,CLC=2296,RESP=2320,sO=3072,sQ=68608,sKV=101376),
}[D]

# A two-sided n-stage pipeline stores full[stage] at base+8*stage and
# empty[stage] at base+8*n+8*stage. This is byte arithmetic in `arena`.
def full(base,n,stage):  return base + 8*stage
def empty(base,n,stage): return base + 8*n + 8*stage

BW={64:64,96:32,128:64}[D]
NBAND=D//BW
def swizzle343(element): return element ^ (((element>>7)&7)<<4)
def matrix_elem(row,col):
    band=col//BW; local_col=col-band*BW
    return band*(128*BW)+swizzle343(row*BW+local_col)
def q_byte(row,col):          return OFF.sQ + 2*matrix_elem(row,col)
def kv_byte(stage,row,col):   return OFF.sKV + 2*(stage*128*D+matrix_elem(row,col))
def o_byte(stage,row,col):    return OFF.sO + 2*(stage*128*D+matrix_elem(row,col))
# D96 therefore has physical band bases +0,+8192,+16384 bytes and global-D
# coordinates 0,32,64. D64/D128 use 64-column bands. Q/K use the bytes as
# K-major operands; V uses `row=K token,col=output feature` as an MN-major B
# operand; O uses the same band-local physical map for the epilogue TensorMap.

def stat_byte(stage,row,is_max):
    return OFF.SCALE + 4*(stage*128 + row + (256 if is_max else 0))
# instruction_selection: scalar `ld.shared.b32`/`st.shared.b32`; extent: one
# rank-one byte offset, never a multidimensional shared tensor.

DESC_HI={64:0x40004040,96:0x80004020,128:0x40004040}[D]
def desc_start(byte): return (byte & 0x3ffff) >> 4
def desc64(byte): return (u64(DESC_HI)<<32) | desc_start(byte)
def qk_desc(byte,k16):
    band=k16//(BW//16); within=k16%(BW//16)
    return desc64(byte) + band*(128*BW*2//16) + 2*within
def v_desc(stage,k16): return desc64(kv_byte(stage,0,0)) + k16*(16*BW*2//16)
# Thus Q/K low-word deltas are [0,2,4,6]+0x400*band for BW64 and
# [0,2]+0x200*band for BW32. V deltas are 0x80*k16 for BW64 and 0x40*k16
# for BW32. The descriptor high word encodes leading/stride/version/swizzle;
# Q/K idesc mark K-major, PV idesc marks its B operand MN-major.

S_COL=(0,128); P_COL=(64,192); O_COL=(256,256+D)
def tmem_row_addr(row,col): return col | ((row&0x60)<<16)
# The instruction lane supplies row&31. For softmax/correction
# row=(warp%4)*32+lane. Score x32 addresses use col=stage*128+{0,32,64,96};
# P x16 stores use col=P_COL[stage]+{0,16,32,48}. PV treats packed 16-bit P
# as tmem_a=P_COL[stage]+8*k16 while its FP32 O accumulator is O_COL[stage].
```

TensorMaps use 128-byte swizzle and zero OOB fill. Q/O are rank 4 in the
non-packed path and rank 5 in packed GQA; K/V are rank 4. Each D band transfers
`128*BW*2` bytes to/from the corresponding integer byte base. Packed row
decoding is `packed=q_block*128+row`, `token=packed//ratio`,
`q_head=kv_head*ratio+packed%ratio`; non-packed is
`token=q_block*128+row,q_head=head,kv_head=head//ratio`. Metadata is indexed by
`(batch,kv_head,q_block)` when packed and `(batch,q_head,q_block)` otherwise,
but its physical ABI never changes rank: `block_index` remains rank 4
`[B,scheduled_head,Q_block,slot]` and `block_nums` remains rank 3
`[B,scheduled_head,Q_block]`. Packing changes index ownership, not metadata rank.

## Exact prologue and pipeline state

```python
tid=thread_id(); warp=warp_uniform(tid//32); lane=tid&31

# Source order: warp 0, all 32 lanes, before shared allocation/initialization.
if warp==0:
    prefetch_tensormap(Qmap); prefetch_tensormap(Kmap)
    prefetch_tensormap(Vmap); prefetch_tensormap(Omap)
# instruction_selection: four consecutive `prefetch.tensormap`; extent: all
# lanes of warp 0 execute each live descriptor prefetch, with no `elect.sync`.

# One elected lane initializes the ordinary pipelines.
init(full(OFF.Q,1,0),1);       init(empty(OFF.Q,1,0),1)
for s in static_range(KV_STAGES):
    init(full(OFF.KV,KV_STAGES,s),1); init(empty(OFF.KV,KV_STAGES,s),1)
for s in static_range(2):
    init(full(OFF.SPO,2,s),1);    init(empty(OFF.SPO,2,s),256)
    init(full(OFF.PLAST,2,s),4);  init(empty(OFF.PLAST,2,s),1)
    init(full(OFF.OACC,2,s),1);   init(empty(OFF.OACC,2,s),128)
    init(full(OFF.STATS,2,s),128);init(empty(OFF.STATS,2,s),128)
    init(full(OFF.OEPI,2,s),128); init(empty(OFF.OEPI,2,s),32)
fence_mbarrier_init_release_cluster()
# instruction_selection: `mbarrier.init.shared.b64` exactly
# 22+2*KV_STAGES times, then the first init fence; no CTA sync yet.

init(full(OFF.CLC,1,0),1); init(empty(OFF.CLC,1,0),512)
fence_mbarrier_init_release_cluster(); sync_cta()
# instruction_selection: two CLC `mbarrier.init.shared.b64`, second init fence,
# then first `bar.sync 0`; extent: CLC create protocol.
sync_cta()
# instruction_selection: second `bar.sync 0`; extent: `pipeline_init_wait`.
# Total init count is 22+2*KV_STAGES+2: D64=46,D96=36,D128=32.

Q_PROD_PHASE=1; Q_CONS_PHASE=0
KV_PROD=(index=0,phase=1); KV_CONS=(index=0,phase=0)
SCORE_PHASE=[0,0]; SPO_ACQUIRE_PHASE=[0,0]
STATS_PROD_PHASE=[1,1]; STATS_CONS_PHASE=0
OACC_CONS_PHASE=0; OEPI_PROD_PHASE=1; OEPI_CONS_PHASE=0
CLC_PROD=(index=0,phase=1); CLC_CONS=(index=0,phase=0)
# `advance` increments index and flips parity on ring wrap (one-stage state
# toggles parity every advance). All states above live outside role loops.
```

## Source-order role programs

The branch order below is source-exact:
`CLC/empty → load → MMA → epilogue → softmax → correction`. Each role owns its
own `initial_work_tile_info`, persistent loop and CLC consumer advance. The six
consumer CFGs are intentionally written separately; they are not a common tail.

```python
# 1. Warp 15: CLC producer and its own consumer path.
if warp==15:
    setmaxnreg_decrease(REG_OTHER)
    # instruction_selection: D64 `setmaxnreg.dec...48`; D96/D128 `...56`.
    work=initial_work_tile_info()
    while work.valid:
        wait(empty(OFF.CLC,1,CLC_PROD.index),CLC_PROD.phase)
        arrive_expect_tx_cluster(full(OFF.CLC,1,CLC_PROD.index),16)
        try_cancel_async_b128(OFF.RESP,full(OFF.CLC,1,CLC_PROD.index))
        advance(CLC_PROD)
        sync_cta(); wait(full(OFF.CLC,1,CLC_CONS.index),CLC_CONS.phase)
        response128=load_shared_v2_b64(OFF.RESP)
        valid=query_cancel_is_canceled(response128)
        next_x=query_cancel_get_first_ctaid_x(response128)
        next_y=query_cancel_get_first_ctaid_y(response128)
        next_z=query_cancel_get_first_ctaid_z(response128)
        fence_async_shared_cta(); release_cluster(empty(OFF.CLC,1,CLC_CONS.index))
        advance(CLC_CONS); work=(next_x,next_y,next_z,valid)
        # instruction_selection: parity wait, cluster expect-tx 16 bytes,
        # try-cancel, `bar.sync`, full wait, `ld.shared.v2.b64`, predicate/x/y/z
        # decode, async-shared fence and cluster-scope empty release.
    wait(empty(OFF.CLC,1,CLC_PROD.index),CLC_PROD.phase)
    # instruction_selection: producer-tail parity wait; extent: one final slot.

# 2. Warp 14: Q plus reverse K/V TensorMap producer.
if warp==14:
    setmaxnreg_decrease(REG_OTHER)
    # instruction_selection: D64 `setmaxnreg.dec.sync.aligned.u32 48`;
    # D96/D128 immediate 56; extent: load-warp branch entry.
    work=initial_work_tile_info()
    while work.valid:
        q_block,head,batch=work.xyz
        if P.PACK:
            token0=(q_block*128)//ratio; head_kv=head
        else:
            token0=q_block*128; q_head=head; head_kv=head//ratio
        raw=block_nums[batch,head,q_block] if P.HAS_NUMS else fixed_count
        process=(raw>0) if P.ALLOW_EMPTY else True
        count=(raw+1)&~1 if P.HAS_NUMS else raw
        def sparse_id(i):
            j=min(i,max(raw-1,0)) if P.HAS_NUMS else i
            return load_global_i32(block_index,meta_offset(batch,head,q_block,j))
        if process:
            wait(empty(OFF.KV,KV_STAGES,KV_PROD.index),KV_PROD.phase)
            # instruction_selection: `mbarrier.try_wait.parity.shared.b64`;
            # extent: all 32 lanes of warp 14, one current KV ring stage.
            arrive_expect_tx(full(OFF.KV,KV_STAGES,KV_PROD.index),128*D*2)
            # instruction_selection: elected
            # `mbarrier.arrive.expect_tx.shared.b64`; transaction extent:
            # one 128xD dtype tile = 128*D*2 bytes across NBAND issues.
            sid=sparse_id(count-1)
            for band in static_range(NBAND):
                copy_g2s_tma(Kmap,(band*BW,sid*128,head_kv,batch),
                             kv_byte(KV_PROD.index,0,band*BW),
                             full(OFF.KV,KV_STAGES,KV_PROD.index))
                # instruction_selection: elected rank-4
                # `cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint`;
                # extent: one 128xBW band, NBAND={1,3,2}[D].
            wait(empty(OFF.Q,1,0),Q_PROD_PHASE)
            # instruction_selection: shared parity wait by all 32 lanes of warp
            # 14; extent: the singleton Q stage. Expect/TMA below remain elected.
            arrive_expect_tx(full(OFF.Q,1,0),128*D*2)
            # instruction_selection: elected `mbarrier.arrive.expect_tx.shared.b64`;
            # transaction extent: 128*D*2 bytes.
            for band in static_range(NBAND):
                if P.PACK:
                    copy_g2s_tma(Qmap_rank5,(band*BW,0,token0,head_kv,batch),
                                 q_byte(0,band*BW),full(OFF.Q,1,0))
                    # instruction_selection: elected rank-5
                    # `cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint`;
                    # extent: one 128xBW packed-Q band, NBAND per D.
                else:
                    copy_g2s_tma(Qmap_rank4,(band*BW,token0,q_head,batch),
                                 q_byte(0,band*BW),full(OFF.Q,1,0))
                    # instruction_selection: elected rank-4 G2S tensor-bulk
                    # family with shared::cta/global.tile, mbarrier completion
                    # and L2 cache hint; extent: one 128xBW band.
            advance(KV_PROD); Q_PROD_PHASE^=1
            # K[N-1] is issued before Q, and KV advances only after Q.
            for kind,logical in [(K,count-2)] + flatten(
                    [(V,count-1-i),(K,count-3-i)] for i in serial_range(count-2)
                  ) + [(V,1),(V,0)]:
                wait(empty(OFF.KV,KV_STAGES,KV_PROD.index),KV_PROD.phase)
                # instruction_selection: shared parity wait by all 32 lanes of
                # warp 14; extent: one KV stage. Expect/TMA remain elected.
                arrive_expect_tx(full(OFF.KV,KV_STAGES,KV_PROD.index),128*D*2)
                # instruction_selection: elected shared expect-tx; transaction
                # extent: 128*D*2 bytes across NBAND tensor-bulk operations.
                sid=sparse_id(logical)
                for band in static_range(NBAND):
                    copy_g2s_tma(Kmap if kind==K else Vmap,
                                 (band*BW,sid*128,head_kv,batch),
                                 kv_byte(KV_PROD.index,0,band*BW),
                                 full(OFF.KV,KV_STAGES,KV_PROD.index))
                    # instruction_selection: elected rank-4
                    # `cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint`;
                    # extent: NBAND={1,3,2}[D] 128xBW bands for this K or V.
                advance(KV_PROD)
        prefetch_next_work()
        sync_cta(); wait(full(OFF.CLC,1,CLC_CONS.index),CLC_CONS.phase)
        response128=load_shared_v2_b64(OFF.RESP)
        valid=query_cancel_is_canceled(response128)
        nx=query_cancel_get_first_ctaid_x(response128)
        ny=query_cancel_get_first_ctaid_y(response128)
        nz=query_cancel_get_first_ctaid_z(response128); work=(nx,ny,nz,valid)
        fence_async_shared_cta(); release_cluster(empty(OFF.CLC,1,CLC_CONS.index))
        advance(CLC_CONS)
        # instruction_selection: load-role `bar.sync 0`, shared full parity
        # wait, `ld.shared.v2.b64`, CLC is-canceled/get-first-ctaid x/y/z,
        # `fence.proxy.async.shared::cta`, and
        # `mbarrier.arrive.shared::cluster.b64`; extent: one CLC response.
    KV_TAIL=clone(KV_PROD)
    for tail_slot in static_range(KV_STAGES):
        wait(empty(OFF.KV,KV_STAGES,KV_TAIL.index),KV_TAIL.phase)
        # instruction_selection: all 32 load-warp lanes execute
        # `mbarrier.try_wait.parity.shared.b64`; extent: exactly KV_STAGES
        # waits (11/6/4 for D64/D96/D128), beginning at the current ring state.
        advance(KV_TAIL)
    wait(empty(OFF.Q,1,0),Q_PROD_PHASE)
    # instruction_selection: all 32 load-warp lanes execute one shared parity
    # wait on Q empty at the persistent Q phase.
    elected_arrive_expect_tx(full(OFF.Q,1,0),128*D*2)
    # instruction_selection: elected
    # `mbarrier.arrive.expect_tx.shared.b64`; extent: one Q-full arm for
    # 128*D*2 bytes. Tail emits no Q TMA and does not change Q_PROD_PHASE.

# 3. Warp 12: TMEM allocation and sole QK/PV issuer.
if warp==12:
    setmaxnreg_decrease(REG_OTHER)
    # instruction_selection: other-role `setmaxnreg.dec.sync.aligned.u32`,
    # immediate 48/56 for D64/D96-or-128; extent: MMA-warp branch entry.
    allocate_tmem(OFF.TMEM,512); wait_for_alloc(); retrieve_tmem_pointer()
    # instruction_selection:
    # `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32`, followed by
    # allocator named-barrier retrieval; extent: one 512-column allocation.
    work=initial_work_tile_info()
    while work.valid:
        raw=work_sparse_count(); process=(raw>0) if P.ALLOW_EMPTY else True
        count=(raw+1)&~1 if P.HAS_NUMS else raw
        if process:
            wait(full(OFF.Q,1,0),Q_CONS_PHASE); Q_CONS_PHASE^=1
            # instruction_selection: ordinary shared parity wait for the sole Q
            # stage; no `tcgen05.fence::after_thread_sync` follows it.
            for stage in static_range(2):
                wait(full(OFF.KV,KV_STAGES,KV_CONS.index),KV_CONS.phase)
                # instruction_selection: shared parity wait for one K stage.
                for k16 in static_range(D//16):
                    mma_qk(S_COL[stage],qk_desc(q_byte(0,0),k16),
                           qk_desc(kv_byte(KV_CONS.index,0,0),k16),
                           QK_IDESC[dtype],accumulate=(k16!=0))
                    # instruction_selection: elected SS
                    # `tcgen05.mma.cta_group::1.kind::f16`; QK idesc is the
                    # dtype-specific 0x08200..., first issue zero-initializes and
                    # later issues accumulate; extent: 4/6/8 issues for D64/96/128.
                tmem_commit_cluster(full(OFF.SPO,2,stage))
                # instruction_selection: elected `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
                # extent: one score generation.
                tmem_commit_cluster(empty(OFF.KV,KV_STAGES,KV_CONS.index))
                # instruction_selection: the same cluster-scope TMEM commit,
                # not an ordinary mbarrier arrive; extent: one K release.
                advance(KV_CONS)
            acc=[False,False]
            for pair in serial_range((count-2)//2):
                for stage in static_range(2):
                    wait(full(OFF.KV,KV_STAGES,KV_CONS.index),KV_CONS.phase)
                    # instruction_selection: shared parity wait for V.
                    VREL=clone(KV_CONS); vstage=KV_CONS.index; advance(KV_CONS)
                    wait(full(OFF.KV,KV_STAGES,KV_CONS.index),KV_CONS.phase)
                    # instruction_selection: shared parity wait for the following K.
                    kstage=KV_CONS.index
                    wait(empty(OFF.SPO,2,stage),SPO_ACQUIRE_PHASE[stage])
                    # instruction_selection: shared parity wait; this one stage
                    # phase also drives the P-last wait below.
                    for k16 in static_range(8):
                        if k16==6:
                            wait_cta(full(OFF.PLAST,2,stage),SPO_ACQUIRE_PHASE[stage])
                            # instruction_selection:
                            # `mbarrier.try_wait.parity.shared::cta.b64`; extent:
                            # after six 16-column slices (96 P columns), before slices 6/7.
                        mma_pv(O_COL[stage],P_COL[stage]+8*k16,v_desc(vstage,k16),
                               PV_IDESC[dtype,D],accumulate=(acc[stage] or k16!=0))
                        # instruction_selection: elected TS
                        # `tcgen05.mma.cta_group::1.kind::f16`, tmem A and
                        # MN-major shared V B, D/dtype PV idesc; first issue uses
                        # zero predicate iff this stage has no prior O, later
                        # issues accumulate; extent: exactly eight K16 issues.
                    for k16 in static_range(D//16):
                        mma_qk(S_COL[stage],qk_desc(q_byte(0,0),k16),
                               qk_desc(kv_byte(kstage,0,0),k16),QK_IDESC[dtype],
                               accumulate=(k16!=0))
                        # instruction_selection: elected SS QK, zero then
                        # accumulate, 4/6/8 issues for D64/96/128.
                    tmem_commit_cluster(full(OFF.SPO,2,stage))
                    # Source updates the single stage phase/accumulator state
                    # after score commit and before either KV release.
                    SPO_ACQUIRE_PHASE[stage]^=1; acc[stage]=True
                    tmem_commit_cluster(empty(OFF.KV,KV_STAGES,VREL.index))
                    tmem_commit_cluster(empty(OFF.KV,KV_STAGES,KV_CONS.index))
                    # instruction_selection: two elected cluster-scope
                    # `tcgen05.commit...mbarrier::arrive::one` releases, V then K.
                    advance(KV_CONS)
            tmem_commit_cluster(empty(OFF.Q,1,0))
            # instruction_selection: elected cluster-scope `tcgen05.commit` Q release.
            for stage in static_range(2):
                wait(full(OFF.KV,KV_STAGES,KV_CONS.index),KV_CONS.phase)
                # instruction_selection: shared parity wait for epilogue V.
                vstage=KV_CONS.index
                wait(empty(OFF.SPO,2,stage),SPO_ACQUIRE_PHASE[stage])
                # instruction_selection: shared SPO-empty wait using the stage phase.
                for k16 in static_range(8):
                    if k16==6:
                        wait_cta(full(OFF.PLAST,2,stage),SPO_ACQUIRE_PHASE[stage])
                        # instruction_selection: CTA-scope P-last parity wait
                        # after the first six PV issues.
                    mma_pv(O_COL[stage],P_COL[stage]+8*k16,v_desc(vstage,k16),
                           PV_IDESC[dtype,D],accumulate=(acc[stage] or k16!=0))
                    # instruction_selection: elected TS `tcgen05.mma`, exact
                    # eight-issue PV chain with zero/acc predicate as above.
                tmem_commit_cluster(full(OFF.OACC,2,stage))
                # instruction_selection: elected
                # `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
                # extent: one final O accumulator stage.
                tmem_commit_cluster(empty(OFF.KV,KV_STAGES,KV_CONS.index))
                # instruction_selection: elected cluster-scope TMEM commits for
                # O-acc publication then V-empty release.
                advance(KV_CONS)
            SPO_ACQUIRE_PHASE[0]^=1; SPO_ACQUIRE_PHASE[1]^=1
            # Epilogue flips both stage phases only after both PV stages finish.
        sync_cta(); wait(full(OFF.CLC,1,CLC_CONS.index),CLC_CONS.phase)
        response128=load_shared_v2_b64(OFF.RESP)
        valid=query_cancel_is_canceled(response128)
        nx=query_cancel_get_first_ctaid_x(response128)
        ny=query_cancel_get_first_ctaid_y(response128)
        nz=query_cancel_get_first_ctaid_z(response128); work=(nx,ny,nz,valid)
        fence_async_shared_cta(); release_cluster(empty(OFF.CLC,1,CLC_CONS.index))
        advance(CLC_CONS)
        # instruction_selection: MMA-role `bar.sync 0`, shared full parity
        # wait, `ld.shared.v2.b64`, CLC predicate/x/y/z decode, async-shared
        # fence and cluster-scope empty arrive; extent: one CLC response.
    relinquish_tmem_alloc_permit(); named_barrier_sync(2,416); free_tmem(512)
    # instruction_selection:
    # `tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned` ->
    # `bar.sync 2,416` -> `tcgen05.dealloc.cta_group::1.sync.aligned.b32`;
    # extent: 512 columns, warp 12 arrives exactly once through the sync.

# 4. Warp 13: corrected O TensorMap epilogue.
if warp==13:
    setmaxnreg_decrease(REG_OTHER)
    # instruction_selection: `setmaxnreg.dec.sync.aligned.u32` immediate
    # 48/56 for D64/D96-or-128; extent: epilogue-warp branch entry.
    work=initial_work_tile_info()
    while work.valid:
        q_block,head,batch=work.xyz
        if P.PACK:
            token0=(q_block*128)//ratio; head_kv=head
        else:
            token0=q_block*128; q_head=head; head_kv=head//ratio
        wait(full(OFF.OEPI,2,0),OEPI_CONS_PHASE)
        # instruction_selection: `mbarrier.try_wait.parity.shared.b64`;
        # extent: epilogue warp, stage 0.
        for band in static_range(NBAND):
            if P.PACK:
                copy_s2g_tma(Omap_rank5,(band*BW,0,token0,head_kv,batch),
                             o_byte(0,0,band*BW))
                # instruction_selection: all 32 lanes of warp 13 execute rank-5
                # `cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint`;
                # extent: one 128xBW packed-O band, NBAND per D.
            else:
                copy_s2g_tma(Omap_rank4,(band*BW,token0,q_head,batch),
                             o_byte(0,0,band*BW))
                # instruction_selection: all 32 lanes of warp 13 execute the
                # rank-4 S2G tensor-bulk with global/shared::cta tile,
                # bulk-group and L2 cache hint; one 128xBW band, NBAND per D.
        commit_bulk_group(); wait_bulk_group_read(0)
        # instruction_selection: `cp.async.bulk.commit_group` then
        # `cp.async.bulk.wait_group.read 0`; extent: all NBAND O stores.
        release(empty(OFF.OEPI,2,0)); OEPI_CONS_PHASE^=1
        # instruction_selection: ordinary `mbarrier.arrive.shared.b64` by all
        # 32 lanes of warp 13; extent matches the empty-barrier arrival count 32.
        sync_cta(); wait(full(OFF.CLC,1,CLC_CONS.index),CLC_CONS.phase)
        response128=load_shared_v2_b64(OFF.RESP)
        valid=query_cancel_is_canceled(response128)
        nx=query_cancel_get_first_ctaid_x(response128)
        ny=query_cancel_get_first_ctaid_y(response128)
        nz=query_cancel_get_first_ctaid_z(response128); work=(nx,ny,nz,valid)
        fence_async_shared_cta(); release_cluster(empty(OFF.CLC,1,CLC_CONS.index))
        advance(CLC_CONS)
        # instruction_selection: epilogue-role `bar.sync 0`, shared full
        # parity wait, `ld.shared.v2.b64`, CLC predicate/x/y/z decode,
        # async-shared fence and cluster-scope empty arrive; one response.

# 5. Warps 0..7: one persistent softmax loop for its fixed stage.
if warp<8:
    setmaxnreg_increase(REG_SOFTMAX)
    # instruction_selection: `setmaxnreg.inc.sync.aligned.u32` immediate
    # 200 for D64 and 184 for D96/D128; extent: each softmax warp.
    wait_for_tmem_alloc(); retrieve_tmem_pointer()
    # instruction_selection: allocator named `bar.sync 2,416` participation
    # used to retrieve the shared TMEM pointer; extent: warps 0..7.
    stage=0 if warp<4 else 1; local_warp=warp&3; row=local_warp*32+lane
    score_phase=SCORE_PHASE[stage]; stats_prod=STATS_PROD_PHASE[stage]
    work=initial_work_tile_info()
    while work.valid:
        raw=work_sparse_count(); has_work=(raw>0) if P.ALLOW_EMPTY else True
        wait(empty(OFF.STATS,2,stage),stats_prod); stats_prod^=1
        # instruction_selection: shared parity wait on the 128-arrival stats
        # empty barrier; extent: one fixed stage and 128 softmax threads.
        row_max=-inf; row_sum=0
        if has_work:
            count=(raw+1)&~1 if P.HAS_NUMS else raw
            wg_count=count//2

            # First block is a distinct static body. Only it can be phantom.
            logical_first=count-1-stage; sid=sparse_id_clamped(logical_first)
            wait(full(OFF.SPO,2,stage),score_phase)
            # instruction_selection: `mbarrier.try_wait.parity.shared.b64`;
            # no `tcgen05.fence::after_thread_sync`; extent: first score generation.
            for chunk in static_range(4):
                copy_t2r(tmem_row_addr(row,S_COL[stage]+32*chunk),
                         score[32*chunk:32*chunk+32])
                # instruction_selection:
                # `tcgen05.ld.sync.aligned.32x32b.x32.b32`; extent: four x32
                # loads, row=(warp%4)*32+lane, columns 0/32/64/96.
            if P.HAS_SIZES or P.HAS_NUMS:
                first_size=(0 if P.HAS_NUMS and logical_first>=raw else
                            load_global_i32(block_sizes,sid)) if P.HAS_SIZES else \
                           (0 if logical_first>=raw else 128)
                for col in static_range(128):
                    if col>=first_size: score[col]=-inf
                # instruction_selection: i32 block-size load when present plus
                # integer compare and `selp.f32` to -inf; extent: first block
                # only. Fixed/no-size specializations emit no mask operations.
            mx=[max(score[0],score[1]),max(score[2],score[3]),
                max(score[4],score[5]),max(score[6],score[7])]
            for i in static_range(8,128,8):
                for j in static_range(4): mx[j]=max(mx[j],score[i+2*j],score[i+2*j+1])
            row_max=max(max(mx[0],mx[1]),mx[2],mx[3])
            max_safe=0 if row_max==-inf else row_max; old_scale=0
            # instruction_selection: first node uses two-input `max.f32`; all
            # later groups use three-input max, ending max01 then max(max01,m2,m3).
            named_barrier_arrive(stage*4+local_warp,64)
            # instruction_selection: `bar.arrive` with 64 participants;
            # extent: the paired first softmax/correction warp.
            for pair in static_range(64):
                score[2*pair:2*pair+2]=fma_packed(
                    score[2*pair:2*pair+2],(softmax_scale_log2,softmax_scale_log2),
                    (-max_safe*softmax_scale_log2,-max_safe*softmax_scale_log2))
            # instruction_selection: `fma.rn.f32x2`; extent: 64 packed pairs.
            for frag in static_range(4):
                for pair in static_range(16):
                    idx=frag*32+2*pair
                    if (2*pair)%10>=6 and frag<3:
                        xy=(max(score[idx],-127),max(score[idx+1],-127))
                        xy_round=add_packed_rm(xy,(12582912.0,12582912.0))
                        xy_back=sub_packed_rn(xy_round,(12582912.0,12582912.0))
                        frac=sub_packed_rn(xy,xy_back)
                        poly=(POLY_EX2_3,POLY_EX2_3)
                        poly=fma_packed(poly,frac,(POLY_EX2_2,POLY_EX2_2))
                        poly=fma_packed(poly,frac,(POLY_EX2_1,POLY_EX2_1))
                        poly=fma_packed(poly,frac,(POLY_EX2_0,POLY_EX2_0))
                        ix=bitcast_s32(xy_round[0]); iy=bitcast_s32(xy_round[1])
                        px=bitcast_s32(poly[0]); py=bitcast_s32(poly[1])
                        p[idx]=bitcast_f32((ix<<23)+px)
                        p[idx+1]=bitcast_f32((iy<<23)+py)
                        # instruction_selection: two `max.f32`, one
                        # `add.rm.f32x2`, two `sub.rn.f32x2`, three
                        # `fma.rn.f32x2`, then per element `shl.b32` +
                        # `add.s32` exponent-bit combine; extent: 18 emulated
                        # pairs per static softmax step.
                    else:
                        p[idx]=exp2(score[idx]); p[idx+1]=exp2(score[idx+1])
                        # instruction_selection: two `ex2.approx.ftz.f32`;
                        # extent: 46 hardware pairs / 92 values per step.
                for pair in static_range(16):
                    idx=frag*32+2*pair
                    packed_p[idx//2]=convert_pair_rn(dtype,p[idx],p[idx+1])
                    # instruction_selection: after all exp in this fragment,
                    # `cvt.rn.bf16x2.f32` or `cvt.rn.f16x2.f32`; extent: 16 pairs.
            for chunk in static_range(4):
                copy_r2t(packed_p[16*chunk:16*chunk+16],
                         tmem_row_addr(row,P_COL[stage]+16*chunk))
                # instruction_selection:
                # `tcgen05.st.sync.aligned.32x32b.x16.b32`; extent: one of four
                # P chunks at 64/80/96/112 or 192/208/224/240.
                if chunk==2:
                    fence_tmem_store(); mbarrier_arrive(empty(OFF.SPO,2,stage))
                    # instruction_selection: `tcgen05.wait::st.sync.aligned`
                    # then ordinary `mbarrier.arrive.shared.b64`; 96-column split.
            fence_tmem_store(); sync_warp(); elect_mbarrier_arrive(full(OFF.PLAST,2,stage))
            # instruction_selection: TMEM-store wait, `bar.warp.sync`, elected
            # ordinary mbarrier arrive; extent: final 32 P columns.
            wait(empty(OFF.STATS,2,stage),stats_prod)
            # instruction_selection: stats-empty shared parity wait occurs after
            # all P stores/publications and before row-sum arithmetic.
            acc=[(p[0],p[1]),(p[2],p[3]),(p[4],p[5]),(p[6],p[7])]
            for i in static_range(8,128,8):
                for j in static_range(4): acc[j]=add_packed(acc[j],(p[i+2*j],p[i+2*j+1]))
            acc[0]=add_packed(acc[0],acc[1]); acc[2]=add_packed(acc[2],acc[3])
            acc[0]=add_packed(acc[0],acc[2]); row_sum=acc[0][0]+acc[0][1]
            # instruction_selection: `add.rn.f32x2` four-way tree then scalar
            # `add.f32`; extent: first complete 128-value probability row.
            stats_prod^=1; score_phase^=1
            # Both phase returns move only after the row-sum tree completes.

            # Remaining blocks are a separate loop and are statically non-first.
            for n_tile in serial_range(wg_count-1):
                logical=count-1-stage-2*(n_tile+1); sid=sparse_id_clamped(logical)
                wait(full(OFF.SPO,2,stage),score_phase)
                # instruction_selection: shared parity wait, no TMEM fence.
                for chunk in static_range(4):
                    copy_t2r(tmem_row_addr(row,S_COL[stage]+32*chunk),
                             score[32*chunk:32*chunk+32])
                    # instruction_selection: x32 T2R; four chunks per row.
                if P.HAS_SIZES:
                    live_size=load_global_i32(block_sizes,sid)
                    for col in static_range(128):
                        if col>=live_size: score[col]=-inf
                    # instruction_selection: live block-size load and i32
                    # predicate/select; no phantom `logical>=raw` predicate here.
                old=row_max
                mx=[max(old,score[0],score[1]),max(score[2],score[3]),
                    max(score[4],score[5]),max(score[6],score[7])]
                for i in static_range(8,128,8):
                    for j in static_range(4): mx[j]=max(mx[j],score[i+2*j],score[i+2*j+1])
                row_max=max(max(mx[0],mx[1]),mx[2],mx[3])
                # instruction_selection: the old row max is injected into the
                # first three-input `max.f32`, then the exact four-way source tree.
                max_safe=0 if row_max==-inf else row_max
                delta=(old-max_safe)*softmax_scale_log2; old_scale=exp2(delta)
                if delta>=-8: row_max=old; max_safe=old; old_scale=1
                # instruction_selection: sub/mul, `ex2.approx.ftz.f32`, compare
                # and selects for the source threshold-8 rescale path.
                store_shared_f32(stat_byte(stage,row,False),old_scale)
                # instruction_selection: scalar `st.shared.b32`; extent: one row.
                named_barrier_arrive(stage*4+local_warp,64)
                # instruction_selection: `bar.arrive`, paired 64-thread rendezvous.
                for pair in static_range(64):
                    score[2*pair:2*pair+2]=fma_packed(
                        score[2*pair:2*pair+2],(softmax_scale_log2,softmax_scale_log2),
                        (-max_safe*softmax_scale_log2,-max_safe*softmax_scale_log2))
                # instruction_selection: 64 packed `fma.rn.f32x2` pairs.
                for frag in static_range(4):
                    for pair in static_range(16):
                        idx=frag*32+2*pair
                        if (2*pair)%10>=6 and frag<3:
                            xy=(max(score[idx],-127),max(score[idx+1],-127))
                            xy_round=add_packed_rm(xy,(12582912.0,12582912.0))
                            xy_back=sub_packed_rn(xy_round,(12582912.0,12582912.0))
                            frac=sub_packed_rn(xy,xy_back)
                            poly=(POLY_EX2_3,POLY_EX2_3)
                            poly=fma_packed(poly,frac,(POLY_EX2_2,POLY_EX2_2))
                            poly=fma_packed(poly,frac,(POLY_EX2_1,POLY_EX2_1))
                            poly=fma_packed(poly,frac,(POLY_EX2_0,POLY_EX2_0))
                            ix=bitcast_s32(xy_round[0]); iy=bitcast_s32(xy_round[1])
                            px=bitcast_s32(poly[0]); py=bitcast_s32(poly[1])
                            p[idx]=bitcast_f32((ix<<23)+px)
                            p[idx+1]=bitcast_f32((iy<<23)+py)
                            # instruction_selection: two `max.f32`, one
                            # `add.rm.f32x2`, two `sub.rn.f32x2`, three
                            # `fma.rn.f32x2`, then two per-element
                            # `shl.b32`/`add.s32` combines; extent: 18
                            # emulated pairs per static softmax step.
                        else:
                            p[idx]=exp2(score[idx]); p[idx+1]=exp2(score[idx+1])
                            # instruction_selection: two
                            # `ex2.approx.ftz.f32`; extent: 46 hardware pairs /
                            # 92 hardware values per static step.
                    for pair in static_range(16):
                        idx=frag*32+2*pair
                        packed_p[idx//2]=convert_pair_rn(dtype,p[idx],p[idx+1])
                        # instruction_selection: fragment-final BF16x2/F16x2
                        # round-to-nearest conversion; extent: 16 pairs.
                for chunk in static_range(4):
                    copy_r2t(packed_p[16*chunk:16*chunk+16],
                             tmem_row_addr(row,P_COL[stage]+16*chunk))
                    # instruction_selection: one x16 R2T P store.
                    if chunk==2:
                        fence_tmem_store(); mbarrier_arrive(empty(OFF.SPO,2,stage))
                        # instruction_selection: TMEM wait then ordinary SPO
                        # empty arrive after the first 96 columns.
                fence_tmem_store(); sync_warp(); elect_mbarrier_arrive(full(OFF.PLAST,2,stage))
                # instruction_selection: TMEM wait, warp sync and elected
                # ordinary P-last arrive for final 32 columns.
                wait(empty(OFF.STATS,2,stage),stats_prod)
                # instruction_selection: stats-empty wait before row-sum tree.
                init=row_sum*old_scale
                acc=[add_packed((init,0),(p[0],p[1])),(p[2],p[3]),
                     (p[4],p[5]),(p[6],p[7])]
                for i in static_range(8,128,8):
                    for j in static_range(4): acc[j]=add_packed(acc[j],(p[i+2*j],p[i+2*j+1]))
                acc[0]=add_packed(acc[0],acc[1]); acc[2]=add_packed(acc[2],acc[3])
                acc[0]=add_packed(acc[0],acc[2]); row_sum=acc[0][0]+acc[0][1]
                # instruction_selection: old sum enters only the first packed
                # pair; packed add tree then final scalar add, exact source order.
                stats_prod^=1; score_phase^=1
                # Both phase returns move only after the row-sum tree completes.
            store_shared_f32(stat_byte(stage,row,False),row_sum)
            # instruction_selection: scalar `st.shared.b32`; final row sum.
            store_shared_f32(stat_byte(stage,row,True),row_max)
            # instruction_selection: scalar `st.shared.b32`; final row max.
            named_barrier_arrive(stage*4+local_warp,64)
            # instruction_selection: final `bar.arrive`, 64 participants.
        else:
            named_barrier_arrive(stage*4+local_warp,64)
            # instruction_selection: synthetic empty `bar.arrive`, 64 participants.
        sync_cta(); wait(full(OFF.CLC,1,CLC_CONS.index),CLC_CONS.phase)
        response128=load_shared_v2_b64(OFF.RESP)
        valid=query_cancel_is_canceled(response128)
        nx=query_cancel_get_first_ctaid_x(response128)
        ny=query_cancel_get_first_ctaid_y(response128)
        nz=query_cancel_get_first_ctaid_z(response128); work=(nx,ny,nz,valid)
        fence_async_shared_cta(); release_cluster(empty(OFF.CLC,1,CLC_CONS.index))
        advance(CLC_CONS)
        # instruction_selection: softmax-role `bar.sync 0`, shared full parity
        # wait, `ld.shared.v2.b64`, CLC predicate/x/y/z decode, async-shared
        # fence and cluster-scope empty arrive; extent: one response.
    wait(empty(OFF.STATS,2,stage),stats_prod); named_barrier_arrive(2,416)
    # instruction_selection: `mbarrier.try_wait.parity.shared.b64` stats
    # producer-tail, then `bar.arrive 2,416`; extent: warps 0..7.

# 6. Warps 8..11: correction and normalized output.
if 8<=warp<12:
    setmaxnreg_decrease(REG_CORRECTION)
    # instruction_selection: `setmaxnreg.dec.sync.aligned.u32` immediate
    # 64 for D64 and 88 for D96/D128; extent: each correction warp.
    wait_for_tmem_alloc(); retrieve_tmem_pointer()
    # instruction_selection: named allocator rendezvous/retrieval; extent:
    # warps 8..11, part of the 416-thread TMEM lifetime group.
    cw=warp-8; row=cw*32+lane
    for stage in static_range(2): mbarrier_arrive(empty(OFF.SPO,2,stage))
    # Exactly two priming releases before, never inside, the persistent loop.
    # instruction_selection: two ordinary `mbarrier.arrive.shared.b64` per
    # correction thread; extent: 128 threads and two stages, once per kernel.
    stats_cons=STATS_CONS_PHASE; oacc_cons=OACC_CONS_PHASE; oepi_prod=OEPI_PROD_PHASE
    work=initial_work_tile_info()
    while work.valid:
        raw=work_sparse_count(); has_work=(raw>0) if P.ALLOW_EMPTY else True
        stats=[(0,-inf,True),(0,-inf,True)]
        if has_work:
            named_barrier_sync(0*4+cw,64); mbarrier_arrive(empty(OFF.STATS,2,0))
            # instruction_selection: `bar.sync id,64` followed by ordinary
            # `mbarrier.arrive.shared.b64`; first stat has no O correction.
            named_barrier_sync(1*4+cw,64); stats_cons^=1
            # instruction_selection: paired `bar.sync id,64`; extent: stage 1.
            count=(raw+1)&~1 if P.HAS_NUMS else raw
            for pair in serial_range((count-2)//2):
                for stage in static_range(2):
                    named_barrier_sync(stage*4+cw,64)
                    # instruction_selection: `bar.sync id,64`; one paired warp.
                    scale=load_shared_f32(stat_byte(stage,row,False))
                    # instruction_selection: scalar `ld.shared.b32`; one row scale.
                    if warp_vote_any(scale<1):
                        # instruction_selection: `vote.sync.ballot.b32`; one warp vote.
                        for chunk in static_range(D//16):
                            copy_t2r(tmem_row_addr(row,O_COL[stage]+16*chunk),o16)
                            # instruction_selection:
                            # `tcgen05.ld.sync.aligned.32x32b.x16.b32`; extent:
                            # D/16 chunks for this stage and row.
                            for j in static_range(0,16,2):
                                o16[j:j+2]=mul_packed(o16[j:j+2],(scale,scale))
                                # instruction_selection: `mul.rn.f32x2`;
                                # extent: eight packed pairs per chunk.
                            copy_r2t(o16,tmem_row_addr(row,O_COL[stage]+16*chunk))
                            # instruction_selection:
                            # `tcgen05.st.sync.aligned.32x32b.x16.b32`;
                            # extent: D/16 chunks.
                        fence_tmem_store()
                        # instruction_selection: `tcgen05.wait::st.sync.aligned`.
                    mbarrier_arrive(empty(OFF.SPO,2,stage))
                    mbarrier_arrive(empty(OFF.STATS,2,1-stage))
                    # instruction_selection: two ordinary shared mbarrier
                    # arrives, O-rescaled then crossed stats reuse.
                stats_cons^=1
            mbarrier_arrive(empty(OFF.STATS,2,1))
            # instruction_selection: final ordinary stats-empty arrive.
            for stage in static_range(2):
                named_barrier_sync(stage*4+cw,64)
                # instruction_selection: final paired `bar.sync id,64`.
                sm=load_shared_f32(stat_byte(stage,row,False))
                mx=load_shared_f32(stat_byte(stage,row,True))
                # instruction_selection: two scalar `ld.shared.b32` per stage.
                mbarrier_arrive(empty(OFF.STATS,2,stage)); bad=(sm==0 or sm!=sm)
                # instruction_selection: ordinary stats-empty arrive plus f32
                # zero/NaN predicates; extent: one row/stage.
                stats[stage]=(sm,mx,bad)
            rm0=-inf if stats[0].bad else stats[0].max
            rm1=-inf if stats[1].bad else stats[1].max
            maxc=max(rm0,rm1); safe_max=0 if maxc==-inf else maxc
            scale0=0 if stats[0].bad else exp2((rm0-safe_max)*softmax_scale_log2)
            scale1=0 if stats[1].bad else exp2((rm1-safe_max)*softmax_scale_log2)
            # instruction_selection: `max.f32`, f32 sub/mul and two
            # `ex2.approx.ftz.f32` selected by valid-stage predicates.
            total=stats[0].sum*scale0+stats[1].sum*scale1
            # instruction_selection: one `mul.f32` for the stage-1 term then
            # one `fma.rn.f32` for stage 0 plus stage 1.
            inv=rcp(total if total!=0 and total==total else 1)
            final0=scale0*inv; final1=scale1*inv
            # instruction_selection: f32 mul/add and
            # `rcp.approx.ftz.f32`; extent: one output row.
            wait(full(OFF.OACC,2,0),oacc_cons); wait(full(OFF.OACC,2,1),oacc_cons)
            # instruction_selection: two shared parity waits; extent: both O stages.
            wait(empty(OFF.OEPI,2,0),oepi_prod)
            # instruction_selection: shared parity wait; extent: corrected-O buffer.
            for chunk in static_range(D//16):
                copy_t2r(tmem_row_addr(row,O_COL[0]+16*chunk),o0)
                copy_t2r(tmem_row_addr(row,O_COL[1]+16*chunk),o1)
                # instruction_selection: two
                # `tcgen05.ld.sync.aligned.32x32b.x16.b32`; extent: D/16 chunks.
                for j in static_range(0,16,2):
                    z0=mul_packed(o0[j:j+2],(final0,final0)); z1=mul_packed(o1[j:j+2],(final1,final1))
                    out[j:j+2]=add_packed(z0,z1)
                    # instruction_selection: two `mul.rn.f32x2` and one
                    # `add.rn.f32x2`; extent: eight packed pairs per chunk.
                for pair in static_range(8):
                    narrow[pair]=convert_pair_rn(dtype,out[2*pair],out[2*pair+1])
                    # instruction_selection: `cvt.rn.bf16x2.f32` or
                    # `cvt.rn.f16x2.f32`; extent: eight conversions per chunk,
                    # total D64/D96/D128 = 32/48/64 conversions per role body.
                store_shared_v4_b32(o_byte(0,row,16*chunk)+0,narrow[0:4])
                store_shared_v4_b32(o_byte(0,row,16*chunk)+16,narrow[4:8])
                # instruction_selection: two `st.shared.v4.b32` per chunk;
                # total D64/D96/D128 = 8/12/16 vector stores.
            fence_async_shared_cta()
            # instruction_selection: `fence.proxy.async.shared::cta` immediately
            # after the shared stores and before either SPO release.
            for stage in static_range(2): mbarrier_arrive(empty(OFF.SPO,2,stage))
            # instruction_selection: two ordinary shared mbarrier arrives.
            oacc_cons^=1; stats_cons^=1
        else:
            for stage in static_range(2):
                named_barrier_sync(stage*4+cw,64)
                mbarrier_arrive(empty(OFF.STATS,2,stage))
                # instruction_selection: paired `bar.sync id,64` and ordinary
                # stats-empty arrive; extent: both synthetic empty stages.
            stats_cons^=1; wait(empty(OFF.OEPI,2,0),oepi_prod)
            # instruction_selection: shared O-epi-empty parity wait.
            for chunk in static_range(D//16):
                zero4=(0,0,0,0)
                store_shared_v4_b32(o_byte(0,row,16*chunk)+0,zero4)
                store_shared_v4_b32(o_byte(0,row,16*chunk)+16,zero4)
                # instruction_selection: two direct zero `st.shared.v4.b32`
                # words per chunk; D96 emits 12. No conversion and no TMEM load.
            fence_async_shared_cta()
            # instruction_selection: `fence.proxy.async.shared::cta` immediately
            # after the empty-path shared stores.
            total=0; safe_max=0
            # No OACC wait/read and no SPO release in an empty work item.
        mbarrier_arrive(full(OFF.OEPI,2,0)); oepi_prod^=1
        # instruction_selection: ordinary `mbarrier.arrive.shared.b64` from all
        # 128 correction threads; extent: one O-epi full generation.
        if P.RETURN_LSE and logical_row_is_live(work,row):
            lse=(safe_max*softmax_scale_log2+log2(total))*ln2 if total>0 else -inf
            # instruction_selection: `lg2.approx.ftz.f32`, `fma.rn.f32`, final
            # `mul.f32` by ln2 and valid-row predicates; extent: one live row.
            store_global_f32(LSE[logical_row(work,row)],lse)
            # instruction_selection: scalar `st.global.b32`; omitted entirely
            # when RETURN_LSE is false.
        sync_cta(); wait(full(OFF.CLC,1,CLC_CONS.index),CLC_CONS.phase)
        response128=load_shared_v2_b64(OFF.RESP)
        valid=query_cancel_is_canceled(response128)
        nx=query_cancel_get_first_ctaid_x(response128)
        ny=query_cancel_get_first_ctaid_y(response128)
        nz=query_cancel_get_first_ctaid_z(response128); work=(nx,ny,nz,valid)
        fence_async_shared_cta(); release_cluster(empty(OFF.CLC,1,CLC_CONS.index))
        advance(CLC_CONS)
        # instruction_selection: correction-role `bar.sync 0`, shared full
        # parity wait, `ld.shared.v2.b64`, CLC predicate/x/y/z decode,
        # async-shared fence and cluster-scope empty arrive; one response.
    wait(empty(OFF.OEPI,2,0),oepi_prod); named_barrier_arrive(2,416)
    # instruction_selection: shared parity producer-tail wait then
    # `bar.arrive 2,416`; warps 0..11 contribute 384 arrivals. Warp 12's later
    # sync contributes the remaining 32 exactly once, after relinquishing.
```

Every work item retains role-local states across CLC advance. An odd variable
count rounds even; a phantom sparse id is clamped but its score block is fully
`-inf` masked. An empty tile issues no Q/K/V or MMA, touches no OACC generation,
writes bitwise-zero O and exact `-inf` LSE. `RETURN_LSE=False` emits no LSE store.

## Bidirectional source / sketch / PTX map

| semantic edge | source lines | sketch section | exported `.loc` / opcode evidence |
| --- | --- | --- | --- |
| ABI, specialization, layouts | 214–460 | scope/storage | wrapper path; rank-4/5 TensorMaps in MHA/packed/fallback exports |
| prefetch and two-stage init | 541–701 | exact prologue | main `.loc 547/631/686/701`; four prefetches, 32 D128 inits, two fences/two syncs |
| role order and register budgets | 703–852 | source-order roles | main `.loc 708/723/750/781/797/829`; exact `setmaxnreg` immediates |
| CLC producer/consumer/tail | 856–883; scheduler 138–188 | role 1 and six advances | scheduler `.loc 150/172/181–185`; query, expect-16, v2.b64, six sync/release paths |
| reverse Q/K/V producer | 885–1015,1888–1940 | role 2 | main `.loc 977–1006/1012–1015`; K-before-Q and producer tails |
| QK/PV order/descriptors | 1018–1240,1989–2305 | maps/role 3 | generated `.loc 214–265`; P 64/192, V/K wait order and cloned release |
| score load/mask/max | 1244–1459 | role 5 | main `.loc 1444–1457`; four x32 loads and helper `.loc 155–188` tree |
| exp/convert/sum/P split | 1460–1486; softmax 168–194; kernel_utils 192–210 | role 5 | frequency-10 exp2, x16 stores, 96/32 publication and packed sum tree |
| correction/stat phases | 1489–1675 | role 6 | main `.loc 1515/1536–1675`; priming, phases and producer tail |
| rescale/combine/output | 1675–1887 | roles 6/4 | main `.loc 1713–1784/1858–1863`; x16 TMEM, packed math, rank-4/5 TMA |
| TMEM lifetime | 752–775,798–852 | roles 3/5/6 | alloc; warps 0–11 arrive; warp 12 relinquish→sync→free |

Reading in reverse, every tensor-bulk copy, TMEM load/store, MMA, conversion,
transcendental, vector shared/global access, mbarrier, named barrier, bulk-group,
CLC and TMEM-lifetime instruction in the 16 fresh exports maps to a concrete
occurrence above. Address arithmetic maps to integer functions, never a layout.

## TIRx, correctness and performance contract

The public module exports `KERNEL_META`, `CONFIGS`, `BENCH_CONFIGS`,
`get_kernel`, `prepare_data`, `run_test`, `prepare_bench`, `run_gpu`, and
`run_bench`. Device code imports only `tirx_kernels.kern as K`; low-level
instruction families use `K.ptx[...]`. There is no inline CUDA function call,
tile primitive, first-class layout or multidimensional SMEM allocation.

The frozen correctness matrix has 32 rows spanning dtype/D/family plus minimal
and ragged Q/KV tails, fixed/odd/empty counts, packed and non-packed GQA,
ratio-3 fallback, BSHD normalization, no-LSE, custom scale and i64 KV address
arithmetic. TIRx and pinned source share immutable inputs and independent
outputs. Both are checked against the same FP32 block-sparse oracle: O uses
`atol=rtol=0.03`; finite LSE uses `atol=rtol=0.002`; finite/`-inf` masks are
exact; NaN is forbidden; empty O is bitwise zero and empty LSE exactly `-inf`;
no-LSE sentinels remain untouched. No match fraction or skip is valid.

Performance truth is exclusively `python -m tirx_kernels.bench_suite` with the
frozen 24-row matrix and pinned cuDNN Frontend reference. Each row must have five
finite positive Proton samples for both implementations and strict
`mean(cudnn_frontend)/mean(tirx) > 0.99`.
