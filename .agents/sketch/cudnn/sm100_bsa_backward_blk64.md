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
python/cudnn/block_sparse_attention/csrc/bwd/sm100_blk64/bsa_bwd_sm100.py.
-->

# cudnn_sm100_bsa_backward_blk64: source-shaped three-kernel sketch

This is the frozen, non-executable design for
`tirx_kernels/cudnn/bsa/block_sparse_attention_backward_sm100_blk64.py`.
The public operation is exactly the source class's direct three-launch ABI:

1. `sum_OdO`: BF16 `O,dO` to two FP32 workspace planes;
2. `bwd`: bucketed inverse-CSR block-sparse backward into FP32 `dQ/dK/dV`
   workspace planes;
3. `convert`: FP32 planes to BF16 `dQ,dK,dV`.

Bucketed-CSR construction, the forward O/LSE producer, allocation, zero fill,
compilation and validation are host work outside the timed closure.  The
kernel imports only `tirx_kernels.kern as K`; it uses no tile primitive, no
first-class layout, no multidimensional shared buffer and no CUDA function
call.

## Fixed scope and runtime ABI

Every specialization is BF16, head dimension 128, sparse block 64, singleton
cluster and SM100a.  Static keys are `(B,H,SQ,SKV,has_block_sizes,bucket)`.
The direct tensors are compact BHSD except the explicit i64-stride guard; all
workspace plane offsets and outer tensor offsets are i64.

```text
sum_OdO(O:bf16*, dO:bf16*, workspace:f32*)
bwd(dO,O,Q,K,V:bf16*, LSE:f32*, offsets:i32*, indices:i32*,
    block_sizes:i32*, workspace:f32*, softmax_scale:f32)
convert(workspace:f32*, dQ,dK,dV:bf16*, softmax_scale:f32)
```

Workspace, in FP32 elements with `Q8=roundup(SQ,8)`, `K8=roundup(SKV,8)`:

```text
sum_OdO   [B,H,Q8]                 @ 0
scaledLSE [B,H,Q8]                 @ B*H*Q8
dQ_acc    [B,H,Q8,128]             @ 2*B*H*Q8
dK_acc    [B,H,K8,128]             @ previous + B*H*Q8*128
dV_acc    [B,H,K8,128]             @ previous + B*H*K8*128
```

`sum_OdO=-sum_d(BF16(O*dO))`, matching the source's narrow packed multiply;
`scaledLSE=-log2(e)*LSE`.  Main recomputes
`P=exp2(S*softmax_scale*log2(e)+scaledLSE)` and
`dS=(dP+sum_OdO)*P`.  `convert` multiplies dQ and dK by softmax scale and does
not multiply dV.

## Independent line-info evidence

Writer exports are under
`.porting/block_sparse_attention_backward_sm100_blk64/source_export/` and are
indexed by `source_export_manifest.json`.  They were produced with AOT disabled,
cache disabled, PTX retention and line info enabled.  The unmasked build has
1589 `.loc` directives; the masked build has 1631.  Both contain three entries:

| source entry | `.reqntid` | static instruction evidence |
| --- | --- | --- |
| `sum_OdO`, source 619-658 | `(8,16,1)` | 31 `ld.global.b32`, 15 `mul.bf16x2`, 15 `cvt.f32.bf16`, 15 `add.rn.f32.bf16`, 3 shuffles, 2 stores |
| `bwd`, source 659-960 / 1023-2140 | `(512,1,1)` | 72 `tcgen05.mma...f16`, 24 mbarrier init, 20 rank-4 TMA G2S, 8 TMA reduce-add S2G, 8 x16 + 4 x32 T2R, 64 scalar f32 atomics |
| `convert`, source 966-1022 | `(16,8,1)` | 44 global loads, 22 packed BF16 converts, 16 packed f32 multiplies, 11 vector stores |

The masked and unmasked source programs differ only in the K-column mask arm.
The c07 export is the primary op-level anchor; c01 proves the compile-time
absence of that arm.  c09 proves empty inverse-CSR tasks, c15 the 1088 bucket
two-group branch, c18 i64 workspace offsets beyond 2 GiB, and c19 grid.x 86881.

## Launch and task decode

```python
sum_grid     = (ceil_div(SQ,16), H, B)          # 128 threads / 4 warps
tasks        = ceil_div(SKV,64)
groups       = ceil_div(ceil_div(SQ,64), bucket)
bwd_grid     = (tasks*groups, H, B)             # 512 threads / 16 warps
convert_grid = (ceil_div(max(SQ,SKV),8), H, B)  # 128 threads / 4 warps

q_group = blockIdx.x // tasks
task     = blockIdx.x - q_group*tasks
kv_block = task
begin    = offsets[((B,H,q_group)*(tasks+1))+task]
end      = offsets[((B,H,q_group)*(tasks+1))+task+1]
count    = end-begin
work     = count>0 and kv_block*64<SKV
```

Before this decode, warp 13 unconditionally prefetches the four TensorMaps and
the whole CTA unconditionally constructs all ten pipelines.  The `work` guard
begins only after those constructors and named barrier 1.  A no-work CTA thus
participates in every initialization rendezvous but executes no role body,
register-budget branch, named barrier 2--5 or TMEM allocation.

## Main CTA roles and rendezvous

The dispatch order is source order, not numeric warp order:

| warps | registers | role |
| --- | --- | --- |
| 13 | 96 decrease | descriptor prefetch and paired Q/dO/LSE/sum/K/V load |
| 12 | 96 decrease | allocate TMEM and issue every MMA |
| 4-11 | 128 increase | T2R S/dP, P/dS recompute, BF16 shared stores, dK/dV epilogue |
| 0-3 | 152 increase | T2R dQ and staged TMA reduce-add |
| 14-15 | 96 decrease | empty |

Named barriers preserve the source counts exactly:

```text
bar 1: 512 threads, after all 24 mbarrier.init and ten constructor epochs
bar 2: 416 threads, MMA + compute + reduce TMEM allocation/retrieval
bar 3: 256 compute threads, paired T2R/shared-store rendezvous
bar 4: 256 compute threads, TMEM deallocation rendezvous
bar 5: 128 reduce threads, each sdQ store/TMA handoff rendezvous
```

Barrier 1 follows ten constructor-local release fences and ten distinct
`bar.sync 0`, not one shared initialization epoch.

`instruction_selection`: `setmaxnreg.{inc,dec}.sync.aligned.u32`,
`bar.sync`, `fence.mbarrier_init.release.cluster`,
`tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32`, and
`tcgen05.dealloc.cta_group::1.sync.aligned.b32` exactly as exported at source
`:783-958`.

## One rank-1 shared arena

The only shared allocation is `u8 arena[199680] align=1024`.  Every following
name is an integer byte offset; swizzles are explicit address functions.

| object | offset | bytes | logical extent / lifetime |
| --- | ---: | ---: | --- |
| Q full[2], Q empty[2] | 0 | 32 | 2-stage TMA→MMA |
| dO full,empty | 32 | 16 | 1-stage TMA→MMA |
| LSE full,empty | 48 | 16 | 1-stage cp.async→compute |
| sum full,empty | 64 | 16 | 1-stage cp.async→compute |
| S full,empty | 80 | 16 | 1-stage MMA→compute |
| dP full,empty | 96 | 16 | 1-stage MMA→compute |
| dQ full,empty | 112 | 16 | 1-stage MMA→reduce |
| P full,empty | 128 | 16 | 1-stage compute→MMA |
| dS full,empty | 144 | 16 | 1-stage compute→MMA |
| dKdV full[2],empty[2] | 160 | 32 | 2-stage MMA→compute |
| TMEM holding word | 192 | 4 | allocation mailbox |
| alignment gap | 196 | 828 | unused |
| sK | 1024 | 16384 | BF16 [64,128] |
| sV | 17408 | 16384 | BF16 [64,128] |
| sQ | 33792 | 65536 | BF16 [stage2, pair2,64,128] |
| sP | 99328 | 16384 | BF16 [128,64] |
| sdO | 115712 | 32768 | BF16 [pair2,64,128] |
| sdS | 148480 | 16384 | BF16 [128,64] |
| sdQ | 164864 | 32768 | FP32 [stage2,pair2,64,32] |
| sLSE | 197632 | 512 | FP32 [128] |
| sSum | 198656 | 512 | FP32 [128] |
| tail padding | 199168 | 512 | source struct extent to 199680 |

For every BF16 MMA operand region:

```python
tile_elem(row, dim) = (dim//64)*4096 + ((row*64 + dim%64) ^ ((row%8)*8))
tile_byte(base,row,dim) = base + 2*tile_elem(row,dim)
```

All remaining selected scalar addresses are integer functions as well:

```python
linear_f32(base,pair,row) = base + 4*(pair*64 + row)  # sLSE/sSum
tmem_cell(base,row,col) = base + col + (row << 16)    # tcgen05 address word

# compute/epilogue local thread ct in 0..255, issue i in 0..1, reg r in 0..15
c_row(ct) = ct % 128
c_col(ct,i,r) = (ct//128)*16 + i*32 + r
c_tmem(base,ct,i) = tmem_cell(base,(c_row(ct)//32)*32,(ct//128)*16+i*32)
# The same map denotes output (kv_row=c_col, dim=c_row) for dK/dV.

# reduce local thread rt in 0..127, issue i in 0..3, reg r in 0..31
q_row(rt) = rt
q_dim(i,r) = i*32 + r
q_tmem(base,rt,i) = tmem_cell(base,(q_row(rt)//32)*32,i*32)

# register vector j=0..7 for a fixed 32-D reduce chunk; p=rt//64,
# t=rt%64.  Source CopyAtom visits vectors in reverse shared order.
sdq_raw(stage,p,t,j) = 164864 + stage*16384 + p*8192 + t*128 + 16*(7-j)
sdq_vec(stage,p,t,j) = sdq_raw(stage,p,t,j) ^ ((sdq_raw(stage,p,t,j)>>3)&0x70)
```

Thus P/dS use `tile_byte(99328 or 148480,c_row,c_col)`, the dK/dV
atomics use the transposed logical coordinate stated above, and every dQ
register is `dQ[q_row,q_dim]` before its explicit `sdq_vec` store.  No
first-class layout survives into the implementation.

The full `c_row`/`q_row` remains the scalar register-to-shared/global
coordinate.  Only a T2R instruction operand uses its 32-row group base; the
individual lane within that group is implicit in `tcgen05.ld`.

## TMEM and descriptors

TMEM always allocates 512 columns:

```text
dK = base+0       [128,64]
dV = base+64      [128,64]
dQ = base+128     [128,128] (aliases dP [128,64] while lifetimes alternate)
S  = base+256     [128,64]
```

Shared descriptors use 128B swizzle 3, `sdo=64`, and `ldo=512` for a
128-wide region or `ldo=0` for a 64-wide region.  The address field is
`(shared_addr>>4)&0x3fff`.  The exported BF16 instruction descriptors are:

```text
Q@K^T, dO@V^T:             0x08100490  # M128,N64,K16, K/K major
dO^T@P, Q^T@dS:            0x08118490  # M128,N64,K16, MN/MN major
dS@K:                      0x08210490  # M128,N128,K16, K/MN major
```

`instruction_selection`: each static issue is
`tcgen05.mma.cta_group::1.kind::f16`; elected lane commits with
`tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`.
The source export contains 72 static MMA instructions: 8 QK + 8 dP + 8 dV in
the prologue; each steady pair adds 8 QK + 4 dQ + 8 dK + 8 dP + 8 dV; the
tail adds 8 dK + 4 dQ.  Rolled runtime repetition preserves this source
program rather than cloning the static body per sparse edge.

## Pipeline contracts

```text
Q:     stages2 full=TMA(1)      empty=tcgen05(1)
dO:    stages1 full=TMA(1)      empty=tcgen05(1)
LSE:   stages1 full=mbar(32)    empty=mbar(256)
sum:   stages1 full=mbar(32)    empty=mbar(256)
S:     stages1 full=tcgen05(1)  empty=mbar(256)
dP:    stages1 full=tcgen05(1)  empty=mbar(256)
dQ:    stages1 full=tcgen05(1)  empty=mbar(128)
P:     stages1 full=mbar(256)   empty=tcgen05(1)
dS:    stages1 full=mbar(256)   empty=tcgen05(1)
dKdV:  stages2 full=tcgen05(1)  empty=mbar(256)
```

Initialization is source declaration order.  Each constructor initializes all
of its full/empty barriers and then executes its own
`fence.mbarrier_init.release.cluster; bar.sync 0` pair:

```text
Q:     offsets 0,8,16,24       -> fence; bar.sync 0
dO:    offsets 32,40           -> fence; bar.sync 0
LSE:   offsets 48,56           -> fence; bar.sync 0
sum:   offsets 64,72           -> fence; bar.sync 0
S:     offsets 80,88           -> fence; bar.sync 0
dP:    offsets 96,104          -> fence; bar.sync 0
dQ:    offsets 112,120         -> fence; bar.sync 0
P:     offsets 128,136         -> fence; bar.sync 0
dS:    offsets 144,152         -> fence; bar.sync 0
dKdV:  offsets 160,168,176,184 -> fence; bar.sync 0
```

`instruction_selection`: 24 `mbarrier.init.shared.b64`; ten separate
`fence.mbarrier_init.release.cluster`; ten separate `bar.sync 0`; then one
`bar.sync 1,512`.  TMA-store has no stored mbarrier: two bulk-group stages are
tracked by commit/wait counters.

Every runtime cursor is the integer pair `(stage,phase)`.  A producer cursor
starts `(0,1)` because it waits on the initial empty generation; a consumer
cursor starts `(0,0)` because it waits on the initial full generation.  For a
one-stage edge `advance` toggles phase.  For Q and dKdV, which are two-stage,
it increments stage and only on `stage==2` wraps stage to zero and toggles
phase.  The reduce bulk-group cursor is `(stage=0)` modulo two and has no
mbarrier phase.

```text
load:    Q_prod,LSE_prod,dO_prod,sum_prod                         = producer
mma:     Q_cons,Q_release=clone(Q_cons),dO_cons,P_cons,dS_cons   = consumer
         S_prod,dP_prod,dQ_prod,dKdV_prod                         = producer
compute: S_cons,LSE_cons,sum_cons,dP_cons,dKdV_cons               = consumer
         P_prod,dS_prod                                           = producer
reduce:  dQ_cons                                                  = consumer
         reduce_bulk_prod                                         = stage 0
```

Each operation below names the cursor advanced at that exact handoff.  The Q
consumer advances immediately after its S read, while `Q_release`, cloned at
the same initial `(0,0)`, advances only after the corresponding dK read.  This
is the source's deliberately delayed Q release.

## Complete role program

The following is operation-level pseudocode.  Each load/store/collective line
is immediately followed by its required instruction selection and extent.

```python
if warp == 13:
  prefetch(desc_Q,desc_K,desc_V,desc_dO)
  # instruction_selection: four prefetch.tensormap, unconditional, source 703-713
for constructor in [Q,dO,LSE,sum,S,dP,dQ,P,dS,dKdV]:
  init_full_and_empty_barriers(constructor)
  # instruction_selection: mbarrier.init.shared.b64, respectively 4,2,2,2,2,2,2,2,2,4 sites
  init_fence(); cta_barrier_zero()
  # instruction_selection: one fence.mbarrier_init.release.cluster then one bar.sync 0
named_barrier(1, 512)
# instruction_selection: bar.sync 1,512, unconditional and after the tenth constructor epoch

decode_task_and_work()
if work:
  apply_source_register_budget_for_role()
  # instruction_selection: one setmaxnreg.inc/dec.sync.aligned.u32 per nonempty role

  role load_warp_13:
    Q_prod=producer_state(depth=2,stage=0,phase=1)
    LSE_prod,dO_prod,sum_prod=producer_state(depth=1,stage=0,phase=1)
    for pair in ceil_div(count,2):
      q0 = indices[begin+2*pair]
      q1 = indices[begin+2*pair+1] if live else floor(SQ/64)
      acquire(Q_prod.stage,Q_prod.phase)
      # instruction_selection: mbarrier.try_wait.parity.shared.b64; one Q-empty wait/pair,
      # initial stage 0 phase 1 and the current two-stage producer cursor thereafter
      expect_tx_base(Q_full[stage], 16384)
      # instruction_selection: mbarrier.arrive.expect_tx.shared.b64, 16384-byte base/pair
      if pair == 0:
        add_expected_bytes(Q_full[stage],32768)  # total 49152
        # instruction_selection: mbarrier.expect_tx.relaxed.cta.shared.b64, +32768 bytes
        tma_g2s_2x64x64(K[kv_block],sK,Q_full[stage])
        # instruction_selection: two elected cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint;
        # extent two [64,64] BF16 boxes = one [64,128] tile = 16384 bytes
      else:
        add_expected_bytes(Q_full[stage],16384)  # total 32768
        # instruction_selection: mbarrier.expect_tx.relaxed.cta.shared.b64, +16384 bytes
      tma_g2s_2x64x64(Q[q0*64],sQ[stage,0],Q_full[stage])
      # instruction_selection: two elected cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint, 2*8192 bytes
      tma_g2s_2x64x64(Q[q1*64],sQ[stage,1],Q_full[stage])
      # instruction_selection: two elected cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint, 2*8192 bytes;
      # dummy q1 is still issued and clipped by its source TensorMap
      advance(Q_prod)
      # state_selection: increment stage; on stage 2 wrap to 0 and toggle phase

      acquire(LSE_prod.stage,LSE_prod.phase)
      # instruction_selection: mbarrier.try_wait.parity.shared.b64, one LSE-empty wait/pair
      for lane in 0..31: cp_async_or_st_zero(two_q0_and_two_q1_f32)
      # instruction_selection: four scalar cp.async.ca.shared.global or st.shared.b32 zero sites/lane;
      # extent 32 lanes*4 = 128 f32 including the dummy/partial-row zero arms
      commit(LSE); advance(LSE_prod)
      # instruction_selection: one cp.async.mbarrier.arrive.noinc.shared.b64/pair
      # state_selection: one-stage producer phase toggle after publication

      acquire(dO_prod.stage,dO_prod.phase)
      # instruction_selection: mbarrier.try_wait.parity.shared.b64, one dO-empty wait/pair
      expect_tx_base(dO_full,16384)
      # instruction_selection: mbarrier.arrive.expect_tx.shared.b64, 16384-byte base/pair
      if pair == 0:
        add_expected_bytes(dO_full,32768)  # total 49152
        # instruction_selection: mbarrier.expect_tx.relaxed.cta.shared.b64, +32768 bytes
        tma_g2s_2x64x64(V[kv_block],sV,dO_full)
        # instruction_selection: two elected cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint, 2*8192 bytes
      else:
        add_expected_bytes(dO_full,16384)  # total 32768
        # instruction_selection: mbarrier.expect_tx.relaxed.cta.shared.b64, +16384 bytes
      tma_g2s_2x64x64(dO[q0],sdO[0],dO_full)
      # instruction_selection: two elected cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint, 2*8192 bytes
      tma_g2s_2x64x64(dO[q1],sdO[1],dO_full)
      # instruction_selection: two elected cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint, 2*8192 bytes
      advance(dO_prod)
      # state_selection: one-stage producer phase toggle after all dO/V issues

      acquire(sum_prod.stage,sum_prod.phase)
      # instruction_selection: mbarrier.try_wait.parity.shared.b64, one sum-empty wait/pair
      for lane in 0..31: cp_async_or_st_zero(two_q0_and_two_q1_f32)
      # instruction_selection: four scalar cp.async.ca.shared.global or st.shared.b32 zero sites/lane;
      # extent 128 f32 including dummy/partial-row zero arms
      commit(sum); advance(sum_prod)
      # instruction_selection: one cp.async.mbarrier.arrive.noinc.shared.b64/pair
      # state_selection: one-stage producer phase toggle after publication

  role mma_warp_12:
    Q_cons=consumer_state(depth=2,stage=0,phase=0)
    Q_release=clone(Q_cons)
    dO_cons,P_cons,dS_cons=consumer_state(depth=1,stage=0,phase=0)
    S_prod,dP_prod,dQ_prod=producer_state(depth=1,stage=0,phase=1)
    dKdV_prod=producer_state(depth=2,stage=0,phase=1)
    tmem_alloc(512)
    # instruction_selection: tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32
    named_barrier(2,416)
    # instruction_selection: bar.sync 2,416
    retrieve_base()
    # instruction_selection: ld.shared.b32
    elected = elect_sync()
    wait(Q_cons.stage,Q_cons.phase); acquire(S_prod.stage,S_prod.phase)
    # instruction_selection: one Q-full mbarrier.try_wait.parity and one S-empty wait
    mma S0=Qpair@K; commit(S); advance(Q_cons); advance(S_prod)
    # instruction_selection: 8 tcgen05.mma f16 plus one tcgen05.commit...mbarrier::arrive::one
    # state_selection: Q consumer and S producer advance at this handoff
    wait(dO_cons.stage,dO_cons.phase); acquire(dP_prod.stage,dP_prod.phase)
    acquire(dQ_prod.stage,dQ_prod.phase)
    # instruction_selection: dO-full, dP-empty and dQ-empty mbarrier.try_wait operations
    mma dP0=dOpair@V; commit(dP); advance(dP_prod)
    # instruction_selection: 8 tcgen05.mma f16 plus one tcgen05.commit
    # state_selection: dP producer phase toggle after publication; dQ producer stays acquired
    wait(P_cons.stage,P_cons.phase); mma dV=dO^T@P accumulate=false
    release(P_cons); advance(P_cons); release(dO_cons); advance(dO_cons)
    # instruction_selection: one P-full wait, 8 tcgen05.mma f16, then two independent
    # mbarrier.arrive.shared.b64 releases
    # state_selection: P and dO one-stage consumer phase toggles after release
    dk_accumulate = False
    for remaining_pair:
      wait(Q_cons.stage,Q_cons.phase); acquire(S_prod.stage,S_prod.phase)
      mma S; commit(S); advance(Q_cons); advance(S_prod)
      # instruction_selection: Q/S waits, 8 tcgen05.mma f16, one tcgen05 commit
      # state_selection: Q two-stage consumer and S producer advance
      wait(dS_cons.stage,dS_cons.phase); acquire(dP_prod.stage,dP_prod.phase)
      # instruction_selection: dS-full and dP-empty mbarrier.try_wait operations
      mma dQ=dS@K accumulate=false; commit(dQ); advance(dQ_prod)
      # instruction_selection: 4 tcgen05.mma f16 for K=64,N=128 plus one tcgen05 commit
      # state_selection: dQ producer phase toggle, then acquire its next empty generation below
      mma dK=Q^T@dS with first K phase accumulate=dk_accumulate,
          remaining seven phases accumulate=True
      dk_accumulate = True
      # instruction_selection: 8 tcgen05.mma f16; first-ever issue uses false predicate,
      # every later first phase and all remaining phases use true
      release(Q_release); advance(Q_release); release(dS_cons); advance(dS_cons)
      # instruction_selection: two independent mbarrier.arrive.shared.b64
      # state_selection: delayed cloned Q-release cursor and dS consumer advance here
      acquire(dQ_prod.stage,dQ_prod.phase); wait(dO_cons.stage,dO_cons.phase)
      mma dP; commit(dP); advance(dP_prod)
      # instruction_selection: dQ-empty/dO-full waits, 8 tcgen05.mma, one tcgen05 commit
      # state_selection: dP producer phase toggle
      wait(P_cons.stage,P_cons.phase); mma dV+=dO^T@P
      release(P_cons); advance(P_cons); release(dO_cons); advance(dO_cons)
      # instruction_selection: P-full wait, 8 tcgen05.mma, two independent releases
      # state_selection: P and dO consumer phase toggles
    acquire(dKdV_prod.stage,dKdV_prod.phase); publish_dV_generation(); advance(dKdV_prod)
    # instruction_selection: dKdV-empty wait then tcgen05.commit...mbarrier::arrive::one
    # state_selection: two-stage producer increments from generation dV to dK
    acquire(dKdV_prod.stage,dKdV_prod.phase); wait(dS_cons.stage,dS_cons.phase)
    # instruction_selection: dKdV-empty and dS-full waits
    mma final_dK with first phase accumulate=dk_accumulate then seven true
    dk_accumulate = True
    # instruction_selection: 8 tcgen05.mma f16, including tail-only false first issue
    publish_dK_generation(); advance(dKdV_prod)
    # instruction_selection: tcgen05.commit...mbarrier::arrive::one
    # state_selection: dKdV two-stage producer wraps/toggles only if stage reaches 2
    mma final_dQ accumulate=false; commit(dQ); advance(dQ_prod)
    # instruction_selection: exactly 4 tcgen05.mma f16 plus one tcgen05 commit
    # state_selection: dQ producer phase toggle
    release(Q_release); advance(Q_release); release(dS_cons); advance(dS_cons)
    # instruction_selection: two independent mbarrier.arrive.shared.b64
    # state_selection: delayed Q-release and dS consumer advance

  role compute_warps_4_11:
    S_cons,LSE_cons,sum_cons,dP_cons=consumer_state(depth=1,stage=0,phase=0)
    P_prod,dS_prod=producer_state(depth=1,stage=0,phase=1)
    dKdV_cons=consumer_state(depth=2,stage=0,phase=0)
    named_barrier(2,416); retrieve_base()
    # instruction_selection: bar.sync 2,416 + ld.shared.b32
    block_size = block_sizes[B,kv_block] if masked else 64
    # instruction_selection: one ld.global.b32/compute thread in masked build;
    # compile-time absent with the whole mask arm in unmasked c01
    for pair in ceil_div(count,2):
      wait(S_cons.stage,S_cons.phase); wait(LSE_cons.stage,LSE_cons.phase)
      acquire(P_prod.stage,P_prod.phase)
      # instruction_selection: three independent mbarrier.try_wait.parity.shared.b64
      t2r S at c_tmem(S_base,ct,0) and c_tmem(S_base,ct,1)
      # instruction_selection: two tcgen05.ld.sync.aligned.32x32b.x16.b32;
      # 16 f32/issue, 32 f32/compute thread, mapped by c_row/c_col
      mask columns >= block_size to -inf if masked
      # instruction_selection: 32 setp.lt.s32 plus 32 selp.b32/compute thread;
      # entire arm is absent from unmasked c01
      packed_fma(S, scale*log2e, scaledLSE); exp2 each lane
      # instruction_selection: 16 fma.rn.f32x2 plus 32 ex2.approx.ftz.f32/thread
      pack BF16x2
      # instruction_selection: 16 cvt.rn.bf16x2.f32/compute thread
      tmem_load_wait(); named_barrier(3,256); tmem_load_wait()
      # instruction_selection: tcgen05.wait::ld.sync.aligned; bar.sync 3,256;
      # second tcgen05.wait::ld.sync.aligned before every store
      store 32 BF16 values/thread to tile_byte(sP,c_row,c_col)
      # instruction_selection: four st.shared.v4.b32/compute thread
      fence_proxy_async_shared()
      # instruction_selection: fence.proxy.async.shared::cta
      commit(P); advance(P_prod)
      release(S_cons); advance(S_cons); release(LSE_cons); advance(LSE_cons)
      # instruction_selection: three independent mbarrier.arrive.shared.b64
      # state_selection: P producer, S consumer and LSE consumer each toggle phase

      wait(sum_cons.stage,sum_cons.phase); wait(dP_cons.stage,dP_cons.phase)
      acquire(dS_prod.stage,dS_prod.phase)
      # instruction_selection: three independent mbarrier.try_wait.parity.shared.b64
      t2r dP at c_tmem(dP_base,ct,0) and c_tmem(dP_base,ct,1)
      # instruction_selection: two tcgen05.ld.sync.aligned.32x32b.x16.b32;
      # 32 f32/compute thread
      packed_add(dP,sum); packed_mul(result,P); pack BF16x2
      # instruction_selection: 16 add.rn.f32x2, 16 mul.rn.f32x2 and
      # 16 cvt.rn.bf16x2.f32/compute thread
      tmem_load_wait(); release(dP_cons); advance(dP_cons)
      # instruction_selection: tcgen05.wait::ld.sync.aligned then one
      # mbarrier.arrive.shared.b64 before all sdS stores
      store 32 BF16 values/thread to tile_byte(sdS,c_row,c_col)
      # instruction_selection: four st.shared.v4.b32/compute thread
      fence_proxy_async_shared()
      # instruction_selection: fence.proxy.async.shared::cta
      commit(dS); advance(dS_prod)
      # instruction_selection: one mbarrier.arrive.shared.b64 dS publication
      # state_selection: dS producer phase toggle
      release(sum_cons); advance(sum_cons)
      # instruction_selection: a distinct final mbarrier.arrive.shared.b64 sum release
      # state_selection: sum consumer phase toggle

    wait(dKdV_cons.stage,dKdV_cons.phase)  # generation dV
    # instruction_selection: one mbarrier.try_wait.parity.shared.b64
    t2r dV at c_tmem(dV_base,ct,0/1); tmem_load_wait()
    # instruction_selection: two tcgen05.ld.sync.aligned.32x32b.x16.b32 and
    # one tcgen05.wait::ld.sync.aligned; 32 f32/compute thread
    atomic_add workspace.dV[kv_block*64+c_col,c_row] when row<SKV
    # instruction_selection: 32 predicated scalar atom.global.add.f32/thread
    release(dKdV_cons); advance(dKdV_cons)
    # instruction_selection: one mbarrier.arrive.shared.b64
    # state_selection: two-stage consumer increments from dV to dK
    wait(dKdV_cons.stage,dKdV_cons.phase)  # generation dK
    # instruction_selection: one mbarrier.try_wait.parity.shared.b64
    t2r dK at c_tmem(dK_base,ct,0/1); tmem_load_wait()
    # instruction_selection: two tcgen05.ld.sync.aligned.32x32b.x16.b32 and
    # one tcgen05.wait::ld.sync.aligned; 32 f32/compute thread
    atomic_add workspace.dK[kv_block*64+c_col,c_row] when row<SKV
    # instruction_selection: 32 predicated scalar atom.global.add.f32/thread
    release(dKdV_cons); advance(dKdV_cons)
    # instruction_selection: one mbarrier.arrive.shared.b64
    # state_selection: two-stage consumer wraps/toggles only on stage 2
    named_barrier(4,256)
    # instruction_selection: bar.sync 4,256
    if global_warp == 8: deallocate 512 columns
    # instruction_selection: tcgen05.dealloc.cta_group::1.sync.aligned.b32;
    # owner is (global_warp & 7)==0 within global compute warps 4..11, hence exactly warp 8

  role reduce_warps_0_3:
    dQ_cons=consumer_state(depth=1,stage=0,phase=0)
    reduce_bulk_prod=bulk_state(depth=2,stage=0)
    named_barrier(2,416); retrieve_base()
    # instruction_selection: bar.sync 2,416 + ld.shared.b32
    for pair in ceil_div(count,2):
      wait(dQ_cons.stage,dQ_cons.phase)
      q0/q1 = inverse CSR pair, with q1=dummy floor(SQ/64) if odd
      # instruction_selection: one dQ-full mbarrier.try_wait.parity.shared.b64
      t2r dQ at q_tmem(dQ_base,rt,issue) for issue in 0..3
      # instruction_selection: four tcgen05.ld.sync.aligned.32x32b.x32.b32;
      # 32 f32/issue = 128 f32/reduce thread, exact q_row/q_dim map
      tmem_load_wait(); release(dQ_cons); advance(dQ_cons)
      # instruction_selection: one tcgen05.wait::ld.sync.aligned then one
      # mbarrier.arrive.shared.b64 dQ release
      # state_selection: dQ consumer phase toggle after release
      for dim_chunk in 0,32,64,96:
        if global_warp == 0: bulk_wait_for_two_stage_slot()
        # instruction_selection: cp.async.bulk.wait_group.read 1, once/chunk and
        # only in global warp 0
        named_barrier(5,128)
        # instruction_selection: bar.sync 5,128, first rendezvous/chunk
        for j in 0..7: store_v4(regs[4*j:4*j+4],sdq_vec(stage,p,t,j))
        # instruction_selection: eight st.shared.v4.b32/reduce thread/chunk;
        # 128 threads cover the paired 128x32 FP32 tile
        fence_proxy_async_shared()
        # instruction_selection: fence.proxy.async.shared::cta, once/chunk
        named_barrier(5,128)
        # instruction_selection: bar.sync 5,128, second rendezvous/chunk
        if global_warp == 0:
          elected lane issues reduce-add for q0 and q1
          # instruction_selection:
          # two unconditional elected cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group.L2::cache_hint;
          # two [64,32] FP32 boxes/chunk. Dummy/OOB q1 is always issued and TensorMap-clipped
          bulk_commit()
          # instruction_selection: one cp.async.bulk.commit_group/chunk, only in global warp 0
        advance(reduce_bulk_prod)  # all four reduce warps, modulo-two stage
        # state_selection: groupwide stage advance is outside the global-warp-0 guard
    bulk_wait_group_read(0)
    # instruction_selection: groupwide cp.async.bulk.wait_group.read 0, unguarded tail
```

The dK/dV stores use scalar atomics because the pinned source export lowers
`CopyUniversalOp` to 64 `atom.global.add.f32` sites, not vector atomics.
Out-of-range final-K rows are suppressed by the coordinate predicate;
`block_sizes` affects score/P/dS but does not shrink the physical workspace
row loop beyond `SKV`.

## `sum_OdO` program

```python
thread_x = tid % 8; thread_y = tid // 8
q = blockIdx.x*16 + thread_y
if q < SQ:
  acc = 0
  for d_pair = thread_x; d_pair < 64; d_pair += 8:
    O2,dO2 = ld.global.b32
    prod2 = mul.bf16x2(O2,dO2)
    acc += cvt.f32.bf16(prod.lo); acc = add.rn.f32.bf16(prod.hi,acc)
  acc = butterfly_sum(acc, xor=1,2,4)
  if thread_x==0:
    workspace.sum[q] = -acc
    workspace.scaled_lse[q] = -log2(e)*LSE[q]
```

`instruction_selection`: the loop is fully unrolled to eight packed loads per
thread; `ld.global.b32`, `mul.bf16x2`, `mov.b32`, `cvt.f32.bf16`,
`add.rn.f32.bf16`, three `shfl.sync.bfly.b32`, `mul.f32` and
`st.global.b32`, source `.loc` 631-657.  Padding beyond SQ remains untouched
zero from the host reset.

## `convert` program

```python
tidx=tid%16; tidy=tid//16; seq=blockIdx.x*8+tidy
for d4=tidx; d4<32; d4+=16:
  if seq<SQ: load v4 f32 dQacc; packed_mul(scale); pack two BF16x2; store v2.b32 dQ
  if seq<SKV:
    load v4 f32 dKacc,dVacc
    packed_mul dK by scale; pack dK and dV; store v2.b32 each
```

`instruction_selection`: `ld.global.v4.b32`, `mov.b64`,
`mul.rn.f32x2`, `cvt.rn.bf16x2.f32`, `st.global.v2.b32`; source `.loc`
982-1021.  All outer element offsets use i64 multiplication.

## Correctness boundary

All 20 fixed configurations must pass against both the pinned source and the
sparse FP64 analytic oracle.  The smallest source-capability tolerances are:

```text
sum_OdO/scaledLSE: 2^-11 / 2^-15
dQacc,dKacc,dVacc: 0.03 / 2^-7 / 0.03
dQ,dK,dV:          2^-9 / 2^-9 / 0.03
```

Each comparison uses the stated atol and rtol simultaneously, exact NaN/+Inf/
-Inf classification, exact zero padding, and redzones.  Empty tasks, partial Q
and K blocks, variable inverse-CSR lengths 0/1/max, masked/unmasked code, BSHD
normalization, batch-dependent block sizes, bucket transitions, i64 strides,
>2GiB workspace offsets and grid.x>65535 are required; none may be skipped.

The frozen representation contract is: `inspect_low_level_ir(...).ok`, no
function calls, only rank-1 shared allocation, no tile primitive/layout, no
inline CUDA source/call, and zero changes under `tirx_kernels/kern`, TVM or
low-level-IR exemptions.
