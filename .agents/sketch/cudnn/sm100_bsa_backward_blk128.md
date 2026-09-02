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

This sketch documents a TIRx port of NVIDIA cuDNN Frontend commit
aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5, file
python/cudnn/block_sparse_attention/csrc/bwd/sm100_blk128/bsa_bwd_sm100.py,
plus its bsa_bwd_preprocess.py and bsa_bwd_postprocess.py launch helpers.
-->

# cudnn_sm100_bsa_backward_blk128: source-shaped backward sketch

This is the non-executable design for
`tirx_kernels/cudnn/bsa/block_sparse_attention_backward_sm100_blk128.py`.
The implementation imports `tirx_kernels.kern as K` and nothing else from a
kernel language.  It has no tile primitive, first-class layout, rank-greater-
than-one shared allocation, CUDA function call, inline CUDA source, or change
under `tirx_kernels/kern`.

The public operation preserves the source launch graph:

1. one 256-thread preprocess writes FP32 `dPsum`, `LSE_log2`, and clears the
   FP32 `dQaccum` plane;
2. one 512-thread SM100a main program consumes bucketed inverse CSR and writes
   `dQaccum`; it writes BF16 dK/dV directly for one Q group and FP32 dK/dV
   accumulation planes for more than one group;
3. one 128-thread postprocess converts/scales `dQaccum` to BF16 dQ;
4. only when `q_groups>1`, two more instances of that postprocess convert dK
   with softmax scale and dV with scale one.

CSR construction, tensor normalization, output/workspace allocation, zeroing
of multi-group dK/dV accumulators, compilation and validation are host work.
Only these source-shaped launches are timed.

## Fixed scope and public ABI

All specializations are BF16, head dimension `D in {64,128}`, sparse block
128, singleton cluster and SM100a.  Public Q/K/V/O/dO/dQ/dK/dV may be BHSD or
BSHD and are normalized to the source BSHD views without a copy.  LSE and
workspaces are FP32.  Offsets and indices are i32.  Every outer tensor and
workspace address is computed in i64; c11/p10 explicitly guard the KV and
large-Q i64 paths.

```text
preprocess(O:bf16*, dO:bf16*, dPsum:f32*, LSE:f32*,
           LSE_log2:f32*, dQaccum:f32*)
main(Q,K,V,dO:bf16*, LSE_log2,dPsum,dQaccum:f32*,
     dK,dV:(bf16* if groups==1 else f32*),
     offsets,indices:i32*, softmax_scale:f32)
postprocess(accum:f32*, output:bf16*, scale:f32)
```

With `Q128=ceil_div(SQ,128)*128`, `KV128=ceil_div(SKV,128)*128`:

```text
dPsum    [B,H,Q128]
LSE_log2 [B,H,Q128]
dQaccum  [B,H,Q128,D]
dKaccum  [B,H,KV128,D]  only groups>1, zero before timed closure
dVaccum  [B,H,KV128,D]  only groups>1, zero before timed closure
```

Preprocess arithmetic is exactly source arithmetic:

```text
dPsum[q]    = sum_d(float(O[q,d]) * float(dO[q,d]))
LSE_log2[q] = (LSE[q] == -inf ? 0 : LSE[q]*log2(e))
dQaccum[:]  = 0
```

Main recomputes `S=K@Q^T`, then
`P=exp2(S*softmax_scale*log2(e)-LSE_log2)` and
`dS=(dP-dPsum)*P`.  Postprocess multiplies dQ and dK by softmax scale,
does not multiply dV, and rounds once to BF16.

## Independent retained source evidence

Writer exports are retained under
`.porting/cudnn_sm100_bsa_backward_blk128/source_exports/` for all four
specializations `(D64,D128) x (group1,group2)`.  AOT and cache were disabled;
PTX retention and line info were enabled.  Each export executed with real
nonzero GPU inputs and asserted finite, nonzero dQ/dK/dV.  Referenced source
and helper files are copied under `.porting/.../source_snapshot/`.

| main specialization | `.loc` | MMA | T2R x32 | prefetch | static dQ reduce | static dKV epilogue | total raw reduce |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| D64, group1 | 1353 | 64 | 8 | 6 | 2 | 2 rank-4 S2G | 2 |
| D64, group2 | 1252 | 64 | 8 | 4 | 2 | 2 raw reduce-add | 4 |
| D128, group1 | 1465 | 80 | 12 | 6 | 4 | 2 rank-4 S2G | 4 |
| D128, group2 | 1403 | 80 | 12 | 4 | 4 | 4 raw reduce-add | 8 |

The dQ counts are one elected static instruction per 32-D chunk.  Each static
dKV epilogue instruction is executed by both 128-thread compute workgroups;
that runtime workgroup multiplicity is separate from the static PTX count.

All main exports have `.reqntid 512,1,1`, 24 mbarrier initializations, six
initialization fences and six `bar.sync 0`.  D128 loads twelve rank-4 G2S
boxes in the statically visible first/steady bodies; D64 loads six.  The
preprocess and postprocess exports also retain nonzero `.loc` coverage.  Every
`.file` entry resolves to the pinned source/helper snapshot, never generated
CUDA.

Static MMA counts are explained, rather than merely matched:

```text
QK:  two source sites * (D/16) issues
dP:  two source sites * (D/16) issues
dV:  prologue and steady sites * 8 K phases
dK:  steady and tail sites * 8 K phases
dQ:  steady and tail sites * 8 K phases
total D64  = 2*4 + 2*4 + 2*8 + 2*8 + 2*8 = 64
total D128 = 2*8 + 2*8 + 2*8 + 2*8 + 2*8 = 80
```

The exported instruction descriptors are:

```text
QK and dP, M128 N128 K16, shared/shared: 0x08200490
dV/dK, M128 N64  K16, tmem/shared:       0x08110490 (D64)
dV/dK, M128 N128 K16, tmem/shared:       0x08210490 (D128)
dQ,    M128 N64  K16, shared/shared:     0x08118490 (D64)
dQ,    M128 N128 K16, shared/shared:     0x08218490 (D128)
```

## Launches and inverse-CSR task decode

```python
q_blocks = ceil_div(SQ,128)
kv_blocks = ceil_div(SKV,128)
bucket = explicit_bucket or (256 if q_blocks>=4096 and H<=1 else
                             512 if q_blocks>=2048 else 384)
groups = ceil_div(q_blocks,bucket)

pre_grid  = (ceil_div(SQ,128), H, B)       # 256 threads
main_grid = (kv_blocks*groups, H, B)        # 512 threads
postQ     = (ceil_div(SQ,128), H, B)        # 128 threads
postKV    = (ceil_div(SKV,128), H, B)       # 128 threads, groups>1

sched_n = blockIdx.x
q_group = sched_n // kv_blocks
kv_block = sched_n - q_group*kv_blocks
begin = offsets[B,H,q_group,kv_block]
end   = offsets[B,H,q_group,kv_block+1]
count = end-begin
q_block(iter) = indices[B,H,begin+iter]
```

One loop iteration is exactly one 128-row Q block.  There is no pairing or
dummy second edge.  `m_block_safe=min(q_block,q_blocks-1)` protects a malformed
last index in the same way as the source, while valid inputs always supply an
in-range Q block.

All six pipeline-constructor epochs and the TMEM allocation rendezvous occur
for every CTA, including `count==0`.  Role loops do no edge work when empty.
For direct group1, the compute warps explicitly zero the empty task's physical
dK/dV output tile.  For multi-group, host-zeroed FP32 accumulation remains
unchanged for an empty task.

## Main CTA roles and named barriers

Dispatch is source order:

| warps | registers | role |
| --- | --- | --- |
| 15 | 24 decrease | empty |
| 14 | 24 decrease | idle |
| 13 | 88 decrease | descriptor prefetch and Q/K/V/dO/stats producer |
| 12 | 88 decrease | TMEM owner and every MMA issue |
| 4-11 | 136 increase | P/dS compute and dK/dV epilogue |
| 0-3 | 152 increase | dQ TMEM reduction to global FP32 |

Named barriers preserve the source IDs and counts:

```text
bar 1/2: 128 compute threads per epilogue half; leader-warp arrive uses 160
bar 3:   256 compute threads, TMEM read/write and shared handoff
bar 4:   128 reduce threads, every sdQ/reduce-add handoff plus tail
bar 5:   416 threads, MMA + compute + reduce TMEM allocation/retrieval/free
```

Barrier 0 is reserved for the six constructor-local CTA synchronization
epochs below.  `instruction_selection` includes
`setmaxnreg.{inc,dec}.sync.aligned.u32`, `bar.sync`,
`fence.mbarrier_init.release.cluster`,
`tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32`, and matching
deallocation.

## One rank-1 shared arena

The implementation allocates one `u8 arena[SMEM_BYTES] align=1024`.  All
subobjects are byte offsets and all address transforms are integer formulas.
Barrier/header offsets are identical in all specializations:

| object | offset | bytes |
| --- | ---: | ---: |
| Q full[2],empty[2] | 0 | 32 |
| dO full,empty | 32 | 16 |
| LSE full[2],empty[2] | 48 | 32 |
| dPsum full,empty | 80 | 16 |
| S/P full,empty | 96 | 16 |
| dP full,empty | 112 | 16 |
| dS full,empty | 128 | 16 |
| dKV full[2],empty[2] | 144 | 32 |
| dQ full,empty | 176 | 16 |
| TMEM holding word | 192 | 4 |
| alignment gap | 196 | 828 |

Data regions are source struct offsets measured by compiling the actual four
specializations and reading the static struct metadata:

| specialization | sQ / bytes | sK | sV | sdO / bytes | sdS | sLSE | dPsum | sdQ | total |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| D64 g1 | 1024 / 32768 | 33792 | 50176 | 66560 / 16384 | 82944 | 115712 | 116736 | 117760 | 150528 |
| D64 gN | 1024 / 32768 | 33792 | 50176 | 66560 / 32768 | 99328 | 132096 | 133120 | 134144 | 166912 |
| D128 g1/gN | 1024 / 65536 | 66560 | 99328 | 132096 / 32768 | 164864 | 197632 | 198656 | 199680 | 232448 |

Region extents not explicit in the table are:

```text
sK,sV: D*128*2 bytes each
sdS:   128*128*2 = 32768 bytes
sLSE:  2 stages * 128 f32 = 1024 bytes
dPsum: 1 stage  * 128 f32 = 512 bytes, then alignment padding
sdQ:   2 stages * 128*32 f32 = 32768 bytes
```

`sQ` is reused by sdK; `sdO` is reused by sdV.  D64 group1 needs only a
16384-byte sdV view, while D64 multi-group needs two FP32 128x32 views and
therefore a 32768-byte sdO allocation.  D128 needs 32768 bytes either way.

The 128-byte swizzle is expressed directly.  For any 128-row BF16 operand and
64-column band:

```python
band_elem(row,dim64) = (row*64 + dim64) ^ ((row & 7)*8)
bf16_addr(base,row,dim,D,stage=0) = (
  base + stage*(128*D*2) + (dim//64)*(128*64*2)
       + 2*band_elem(row,dim%64)
)
```

This names sQ, sK, sV, sdO and sdS views, including their transposed descriptor
interpretations.  Scalar stats and reduction storage are:

```python
stat_addr(base,row,stage,stride128=True) = base + 4*(stage*128 + row)
sdq_addr(stage,row,col32) = SDQ + 4*(stage*128*32 + row*32 + col32)
tmem_cell(col,row_group32) = col + (row_group32 << 16)
```

The implementation must preserve the exported vector-store ordering and any
swizzled R2S addresses used by direct dKV.  It may spell those as integer XOR
functions, never as a layout object.

## TMEM allocation and MMA descriptors

Every CTA allocates 512 TMEM columns.  Logical regions are:

```text
S/P  = base + 0
dV   = base + 128
dP/dQ/dS = base + 128 + D
dK   = base + 256 + D
```

Thus D64 uses `(S=0,dV=128,dP/dQ=192,dK=320)` and D128 uses
`(0,128,256,384)`.  P aliases S; dS and dQ alias dP by alternating pipeline
lifetimes.

Shared descriptors use swizzle mode 3, `sdo=64`, `ldo=512`, and address field
`(shared_addr>>4)&0x3fff`.  The QK/dP shared/shared helpers advance the low
descriptor by `0,2,4,6` and, for D128, `0x400,0x402,0x404,0x406`.  The
TMEM-A dV/dK helpers advance TMEM-A by 8 and shared-B by 0x80 for eight K
phases.  dQ emits eight elected raw `tcgen05.mma` issues because its source
helper uses a shared/shared partition whose accumulator coordinate must be
spelled explicitly in K.

Every elected issue is
`tcgen05.mma.cta_group::1.kind::f16`; publication is
`tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`.

## Pipeline construction and state

```text
S/P:   stage1 UMMA->compute producer arrivals1, consumer arrivals8
dP:    stage1 UMMA->compute producer arrivals1, consumer arrivals8
dKV:   stage2 UMMA->compute producer arrivals1, consumer arrivals8
dQ:    stage1 UMMA->reduce  producer arrivals1, consumer arrivals4
dS:    stage1 compute->UMMA producer arrivals8, consumer arrivals1
LSE:   stage2 TMA->compute  producer arrivals1, consumer arrivals8, 512B/stage
dPsum: stage1 TMA->compute  producer arrivals1, consumer arrivals8, 512B/stage
Q:     stage2 TMA->UMMA     producer arrivals1, consumer arrivals1, D*256B/stage
dO:    stage1 TMA->UMMA     producer arrivals1, consumer arrivals1, D*256B/stage
```

Initialization follows source construction order, not shared declaration
order.  The first five families sync independently; deferred TMA families
share the sixth epoch completed by non-deferred dO:

```text
epoch 1 S/P:  full96, empty104                         -> fence; bar.sync 0
epoch 2 dP:   full112,empty120                         -> fence; bar.sync 0
epoch 3 dKV:  full144,152; empty160,168                -> fence; bar.sync 0
epoch 4 dQ:   full176,empty184                         -> fence; bar.sync 0
epoch 5 dS:   full128,empty136                         -> fence; bar.sync 0
epoch 6 LSE:  full48,56; empty64,72
        dPsum full80,empty88
        Q:    full0,8; empty16,24
        dO:   full32,empty40                           -> fence; bar.sync 0
```

That is exactly 24 `mbarrier.init`, six release fences and six `bar.sync 0`.
Producer states start `(stage=0,phase=1)`, consumer states `(0,0)`.  One-stage
advance toggles phase.  Two-stage Q/LSE/dKV advances stage and toggles phase
only on wrap.  The source clones one Q consumer handle: the original advances
after QK, while release is delayed until dK has consumed that Q stage.

Role-owned states are:

```text
load:    Q_LSE_prod(depth2), dO_dPsum_prod(depth1)
mma:     Q_cons+delayed_handles, dO_cons, dS_cons;
         S/P_prod,dP_prod,dQ_prod(depth1), dKV_prod(depth2)
compute: S/P_cons,dP_cons,dS_prod,dKV_cons,
         LSE_cons(depth2),dPsum_cons(depth1)
reduce:  dQ_cons(depth1), bulk_store_prod(depth2)
```

## Complete main role program

The pseudocode below is operation-level and every collective line is a required
source instruction family, not permission to call an out-of-line helper.

```python
if warp==13 and elected:
  prefetch(desc_Q,desc_K,desc_V,desc_dO)
  if groups==1: prefetch(desc_dV,desc_dK)
for epoch in six_epochs_above:
  elected_init_all_mbarriers(epoch)
  fence_mbarrier_init_release_cluster(); bar_sync_0()

set_role_register_budget()
if warp==12: tmem_alloc(512)
if warp in {0..12 except idle}: named_barrier(5,416); retrieve_tmem_base()

role load_warp_13:
  for each scheduler task:
    if count>0:
      for edge in 0..count-1:
        qb = indices[begin+edge]
        acquire(Q_prod)
        expect(Q_full, D*256 + (D*256 if edge==0 else 0))
        if edge==0: rank4_tma(K[kv_block,0:D] -> sK)
        rank4_tma(Q[qb*128,0:D] -> sQ[Q_stage])
        commit Q
        acquire(LSE_prod sharing the Q cursor)
        bulk_g2s_512B(LSE_log2[qb*128:q+128] -> sLSE[stage])
        advance Q_LSE_prod

        acquire(dO_prod)
        expect(dO_full, D*256 + (D*256 if edge==0 else 0))
        if edge==0: rank4_tma(V[kv_block,0:D] -> sV)
        rank4_tma(dO[qb*128,0:D] -> sdO)
        commit dO
        acquire(dPsum_prod sharing the dO cursor)
        bulk_g2s_512B(dPsum[qb*128:q+128] -> shared)
        advance dO_dPsum_prod
      producer_tail(Q,LSE,dO,dPsum)

role mma_warp_12:
  for each task with count>0:
    zero_init_dK = True
    Q0 = wait_advance_Q(); wait_empty(S/P)
    issue QK(Q0,zero=True); publish S
    wait dO0; wait_empty(dP); wait_empty(dQ)
    issue V@dO0.T(zero=True); publish dP
    toggle shared S/P,dP,dQ producer phase
    wait_empty(S/P) as the compute-produced P-ready handoff
    issue P.T@dO0(zero=True); release dO0

    current_Q = Q0
    for edge in 1..count-1:
      next_Q = wait_advance_Q()
      issue QK(next_Q,zero=True); publish S
      wait dS
      issue dS.T@current_Q(zero=zero_init_dK); zero_init_dK=False
      release current_Q
      issue dS@K(zero=True); publish dQ
      release dS
      wait next dO; wait next empty dQ
      issue V@dO.T(zero=True); publish dP
      toggle shared producer phase
      wait_empty(S/P) as P-ready; issue P.T@dO(zero=False); release dO
      current_Q=next_Q

    publish final S/P full generation after final dV
    publish dV generation on dKV stage0
    wait dKV stage1 empty; wait final dS
    issue dS.T@current_Q(zero=zero_init_dK); publish dK generation
    issue final dS@K(zero=True); publish dQ
    release current_Q,dS; toggle producer phases

role compute_warps_4_11:
  local_ct = threadIdx.x % 256
  dp_idx = local_ct % 128
  wg = local_ct // 128
  for each task:
    for edge in 0..count-1:
      wait LSE; wait S
      T2R_x32(S)                         # two row fragments per thread
      if physical KV row >= SKV: S=-inf # whole-row tail predicate only
      packed_fma(S,scale*log2e,-LSE); exp2_fast; BF16 pack P
      on first repetition: fence_tmem_load; bar3_256
      R2T_x16(P)
      fence_tmem_store; fence_shared; bar3_256
      elected release S; release LSE; advance

      wait dPsum; wait dP
      T2R_x32(dP); fence_tmem_load; bar3_256
      packed_sub(dP,dPsum); packed_mul(P,dP); BF16 pack dS
      acquire dS; R2T_x16(dS)
      R2S BF16 dS with st.shared.v4.b32
      # eight 128-bit stores/thread: two x32 repetitions, four stores each;
      # 256 threads cover the full [KV-row128,Q-row128] BF16 tile using the
      # bf16_addr(sdS,kv_row,q_row,D=128) integer swizzle
      fence_tmem_store; advance combined S/P/dP consumer
      fence_shared; bar3_256
      release dPsum; elected publish dS; advance

    if count>0:
      wait dV generation; epilogue(V,scale=None); release/advance dKV
      wait dK generation; epilogue(K,scale=(softmax_scale if groups==1 else None))
      release/advance dKV
    elif groups==1:
      wg0 first 128 compute threads zero direct dK tile
      wg1 second 128 compute threads zero direct dV tile

  epilogue(kind):
    split 256 compute threads into two 128-thread workgroups
    each workgroup owns half the output feature columns
    for epi_stage in (1 stage except D128 multi-group has 2):
      T2R_x32 FP32; fence_tmem_load
      if direct K: packed multiply by softmax_scale
      convert to destination dtype (BF16 direct, FP32 multi-group)
      R2S with st.shared.v4.b32 into sdK=sQ or sdV=sdO
      # group1 BF16: 4 stores/thread for D64 or 8 for D128 per kind;
      # one stage, and each workgroup covers [128,D/2]
      # groupN FP32: 8 stores/thread/stage over [128,32]; one stage for D64,
      # two stages for D128, repeated independently for K and V
      fence_shared; bar_sync(1+wg,128)
      leader warp:
        if groups==1: rank4 TMA S2G one half-tile
        else: raw cp.reduce.async.bulk.global.shared.add.f32 16384B
        for a following stage: bulk_commit; bulk_wait_group_read(0)
        barrier_arrive(1+wg,160)
      fence_shared; bar_sync(1+wg,160)

role reduce_warps_0_3:
  local_rt = threadIdx.x % 128
  for each task:
    for edge in 0..count-1:
      wait dQ
      T2R_x32 dQ for D/32 repetitions # 2 for D64, 4 for D128
      fence_tmem_load; sync_warp; elected release dQ; advance
      for chunk in 0,32,..,D-32:
        R2S 128x32 FP32 into sdQ[bulk_stage]
        fence_shared; bar4_128
        if reduce warp0 elected:
          cp.reduce.async.bulk.global.shared.add.f32 16384B
          bulk_commit(); bulk_wait_group_read(1)
        bar4_128; advance bulk_stage modulo2
    if count>0:
      warp0 bulk_wait_group_read(0); bar4_128
  all reduce warps execute final bulk_wait_group_read(0)

mma owner after all tasks:
  relinquish allocation permit; bar5_416; free 512 TMEM columns
compute/reduce roles after all tasks:
  bar5 arrival
```

The score tail mask is deliberately whole-row only.  There is no block-size
input and no per-column sparse mask.  TMA tensor-map clipping handles Q/KV
sequence tails; explicit global predicates guard direct empty-tile zeroing and
postprocess stores.

## Preprocess program

The preprocess launch is 256 threads per Q block.  It uses source 128-bit
loads and FP32 multiply/reduction, not BF16 packed multiplication:

```python
tile q0 = blockIdx.x*128
each thread owns a source 128-bit vector and strided repetitions over [128,D]
acc = sum(float(O)*float(dO))
warp_reduce within threads_per_row = (128 if D%128==0 else 64)/8
column-zero owner writes dPsum[row], or zero for padding
all 256 threads vector-zero dQaccum[128,D]
thread row<rounded tail writes LSE_log2 = 0 for -inf else LSE*log2(e)
```

`griddepcontrol_wait` precedes O/dO/LSE reads and launch-dependents follows
their loads on SM100, matching the source launch dependency protocol.

## Postprocess program

Each 128-thread CTA owns one complete `[128,D]` accumulator tile.  It uses a
rank-1 shared arena of `max(128*D*4,128*D*2)=128*D*4` bytes: 32768 bytes for
D64 and 65536 bytes for D128.  There is no D/32 outer pipeline loop.

```python
each thread issues D/4 cp.async.cg.shared.global 128-bit loads
cp.async.commit_group; cp.async.wait_group 0; bar.sync 0
S2R D FP32 values/thread
D/2 mul.f32x2 and D/2 cvt.rn.bf16x2.f32 per thread
bar.sync 0
D/8 st.shared.v4.b32 BF16 stores/thread
bar.sync 0
D/8 ld.shared.v4.b32 and predicated st.global.v4.b32 per thread
```

The same K kernel is instantiated for dQ, dK and dV; dV receives `scale=1`.

## Fixed correctness and benchmark domain

Correctness is the repository-derived 18-case matrix in `spec.py`, including:
D64/D128; BHSD/BSHD; B/H>1; SQ/SKV tails at 1/127/129/255/257/641;
fixed, variable and all-empty inverse CSR; scale 0.125; i64 KV strides;
the 384->385 Q-block group transition; cancellation and sharp-softmax data.
No case may be skipped.

Performance is the 13-case matrix p00-p12 in `spec.py`: D64/D128, small and
large KV working sets, B/H parallelism, variable/empty CSR, Q-block counts
384/385/2048/4096, group counts 1/2/4/8/16, i64 strides, BSHD and explicit
scale.  Only full `bench_suite` source/TIRx ratios are authoritative.

## Correctness boundary

The source-capability tolerance is calibrated before judging TIRx.  Five fresh
source executions per numerical class are compared to the chunked sparse FP64
analytic oracle over this ordered grid:

```text
{0,2^-20,2^-18,2^-16,2^-14,2^-12,2^-10,2^-9,2^-8,2^-7,
 1e-2,2e-2,3e-2} x itself
```

Choose the lexicographically smallest passing `(max_error,sum_error,atol,rtol)`
pair for each of dQ/dK/dV and each required intermediate.  TIRx must pass the
same selected tolerance; it may never widen it.  A required tolerance above
0.03 is a blocking correctness failure.

Every check also requires exact NaN/+Inf/-Inf classification, exact padding
zeros, redzones, input immutability, preallocated-output pointer identity, CSR
immutability and deterministic repeat behavior.  Final dQ/dK/dV, dPsum,
LSE_log2, dQaccum, and multi-group dK/dV accumulation planes are compared.
The final gate includes memcheck, initcheck, racecheck and synccheck on c00,
c08 and c13.

The low-level representation contract is immutable:
`inspect_low_level_ir(...).ok`, zero function calls, only rank-1 shared
allocation, no tile primitive/layout, no inline CUDA source/call, no exemptions,
and no changes under `tirx_kernels/kern`, TVM or low-level-IR policy code.
