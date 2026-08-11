# FlashInfer MTP horizontal selective-state-update — SM100 kernel sketch

## Scope, source identity, and transcription contract

This sketch freezes the implementation structure of FlashInfer commit
`f2e04400e330fb2debe0bf8730d9424a1d37927f` for the explicit
`algorithm="horizontal"` path on SM100a. The primary source is
`include/flashinfer/mamba/kernel_selective_state_update_mtp_horizontal.cuh`
with SHA-256
`7d07b8ef9faa9cfac7a24071ed235a48a65bd7842579c612f2c56e949440bdfa`.
The launch/descriptor path is
`include/flashinfer/mamba/invoke_selective_state_update_mtp.cuh` with
SHA-256
`5be5da574adc8b148064ee1214a9eda99881ec5163ef244e462cb53ae54c4bf8`.

Reachable helpers are frozen in `.porting/selective_state_update_mtp_horizontal/`
and include `ssu_mtp_common.cuh`, `common.cuh`, `conversion.cuh`,
`create_tensor_map.cuh`, and `vec_dtypes.cuh`. The TIRx transcription must use
ordinary control flow, buffers, CUDA intrinsics, and native `T.ptx` only. It
must not use a tile primitive or reproduce the legacy async-horizontal body.

The current source contains an `xor_swizzle` helper and an old high-level
comment about XOR swizzling, but the reachable horizontal kernel never invokes
that helper and all four TensorMaps use `CU_TENSOR_MAP_SWIZZLE_NONE`. Therefore
this frozen code shape has **no emitted XOR-address transformation**. Padding
to a 128-byte bank cycle is the only shared-layout transformation in the
reachable body. A transcription must not invent an XOR instruction absent
from the frozen CUDA/PTX evidence.

The 36 correctness configurations plus 40 benchmark configurations collapse
to 14 unique JIT module keys. Options represented only by runtime nullable
pointers or runtime flags (`D`, bias, z, update, intermediate) do not create a
new module. The independent reviewer must enumerate the exact keys rather
than compile once per public workload.

## Host dispatch and descriptor gates

The public API accepts `auto`, `simple`, `vertical`, `horizontal`, and
`async_horizontal`. Before dispatch, `async_horizontal` is normalized to
`simple`. This port enters with `horizontal` explicitly; it does not use
`auto`.

Common host gates run before the horizontal branch:

```text
input_t = bf16
matrixA_t = f32
state_t in {bf16, f16, f32}
weight_t in {bf16, f32}
stateIndex_t in {i32, i64}

x, optional z, B, C bases and batch strides satisfy PackedAligned<input_t>
state and optional intermediate bases satisfy the selected packed-state alignment
output satisfies PackedAligned<input_t, OUTPUT_VECTOR>
```

The horizontal branch then enforces:

```text
nheads % ngroups == 0
DIM % 32 == 0
state_scale is absent
cu_seqlens is absent
PHILOX_ROUNDS in {0,10}
PHILOX_ROUNDS == 10 implies state_t == f16
NUM_IN_STAGES = 2
TMA_STATE_ROWS = 2 * ROWS_PER_PASS = 32
HEADS_PER_CTA = 1
HEADS_PER_GROUP in {1,2,4,8,16,32,64}
```

The represented dispatch domain is `DIM={64,128}`, `DSTATE={64,96,128}`,
`NTOKENS={1,2,4,6,8}`. The module key also includes state, weight, index,
cu-seqlens/accepted-token dtypes, and Philox rounds even though the latter two
input families are absent from the actual horizontal launch.

Four rank-4 TensorMaps are constructed and passed as independent
`__grid_constant__ CUtensorMap` kernel parameters in the exact order state,
B, C, x:

```text
state:
  shape   = (DSTATE, DIM, nheads, state_cache_size)
  strides = (1, DSTATE, DSTATE*DIM, state_stride_batch)
  tile    = (DSTATE_PAD, 32, 1, 1)

B/C:
  shape   = (DSTATE, ngroups, NTOKENS, batch)
  strides = (1, DSTATE, {B,C}_stride_mtp, {B,C}_stride_batch)
  tile    = (DSTATE_PAD, 1, NTOKENS, 1)

x:
  shape   = (DIM, nheads, NTOKENS, batch)
  strides = (1, DIM, x_stride_mtp, x_stride_batch)
  tile    = (DIM, 1, NTOKENS, 1)
```

All bases are 128-byte aligned. Descriptor mode is interleave none, swizzle
none, L2 promotion 128B, OOB fill none, and element box strides all one.
Strides after dimension zero are encoded in bytes. A tile width larger than
the global DSTATE is legal: TMA skips out-of-bounds columns, and compute
guards them in registers.

The launch is:

```text
grid  = (batch, nheads, 1)       # HEADS_PER_CTA is frozen to one
block = (32, 5, 1)               # four compute warps plus one TMA warp
dynamic shared = sizeof(GroupStorageHorizontal)
launch bounds = __launch_bounds__(160, 6)
```

## Compile-time geometry

```text
NUM_COMPUTE_WARPS = 4
NUM_TMA_WARPS = 1
NUM_WARPS = 5
LANES_PER_ROW = 8
ROWS_PER_WARP = 32 / 8 = 4
ROWS_PER_PASS = 4 * 4 = 16
NUM_IN_STAGES = 2
TMA_STATE_ROWS = 32
NUM_TMA_LOADS = DIM / 32                 # two for DIM64, four for DIM128
SUBPASSES_PER_TMA = 32 / 16 = 2
HEADS_PER_CTA = 1

DSTATE_PAD = round_up(DSTATE*sizeof(state_t), 128) / sizeof(state_t)
STATE_VALUES_PER_THREAD = DSTATE_PAD / 8
STATE_VALUES_PER_BANK = 4 / sizeof(state_t)
SMEM_READS_PER_THREAD_PER_TILE = 32 / 8 = 4
ELEMS_PER_TILE_MEMBER = 4 * STATE_VALUES_PER_BANK
ELEMS_PER_TILE = 8 * ELEMS_PER_TILE_MEMBER
NUM_TILES = STATE_VALUES_PER_THREAD / ELEMS_PER_TILE_MEMBER
PAIRS_PER_TILE_MEMBER = ELEMS_PER_TILE_MEMBER / 2
```

Consequences by frozen state shape:

| state / DSTATE | DSTATE_PAD | elems/member | tiles/thread | pairs/tile |
|---|---:|---:|---:|---:|
| bf16/f16, 64 | 64 | 8 | 1 | 4 |
| bf16/f16, 96 | 128 | 8 | 2, second tile members 4..7 inactive | 4 |
| bf16/f16, 128 | 128 | 8 | 2 | 4 |
| f32, 128 | 128 | 4 | 4 | 2 |

Within a compute warp:

```text
member = lane % 8                  # DSTATE partition 0..7
group  = lane / 8                  # one of four DIM rows
baseCol(tile,e) = tile*ELEMS_PER_TILE + member*ELEMS_PER_TILE_MEMBER + e
sram_row = subpass*16 + compute_warp*4 + group
dd = tma_load*32 + sram_row
```

Every active lane owns float2 recurrence registers for its in-bounds adjacent
state pairs. For DSTATE96 the second tile is present in the padded shared row,
but only members 0..3 execute its loads/math/stores; members 4..7 initialize
or retain no logical state for that tile.

## Exact shared-memory ABI

One CTA owns one `GroupStorageHorizontal`; there is no per-head array because
`HEADS_PER_CTA=1`. Offsets must be mechanically computed with C++ struct
alignment:

```text
OFF_B = 0, alignment 128
SIZE_B = NTOKENS * DSTATE_PAD * 2

OFF_C = align_up(OFF_B + SIZE_B, 128)
SIZE_C = NTOKENS * DSTATE_PAD * 2

OFF_STATE = align_up(OFF_C + SIZE_C, 128)
SIZE_STATE_STAGE = 32 * DSTATE_PAD * sizeof(state_t)
SIZE_STATE = 2 * SIZE_STATE_STAGE

OFF_X = align_up(OFF_STATE + SIZE_STATE, 128)
SIZE_X = NTOKENS * DIM * 2

OFF_DT = OFF_X + SIZE_X
SIZE_DT = NTOKENS * 4

OFF_OUT = OFF_DT + SIZE_DT
SIZE_OUT = NTOKENS * DIM * 4

OFF_EMPTY = align_up(OFF_OUT + SIZE_OUT, 8)      # two uint64 mbarriers
OFF_FULL = OFF_EMPTY + 16                        # two uint64 mbarriers
OFF_OUT_READY = OFF_FULL + 16                    # one uint64 mbarrier
SHARED_BYTES = align_up(OFF_OUT_READY + 8, 128)  # struct alignment
```

B, C, state stages, and x are TMA destinations. dt is cooperatively published
by compute lanes. out is published by `member==0` lanes and later read by the
same compute warps during the inline epilogue. There is no aliasing between
regions or stages.

For the representative BF16 state, DSTATE128, DIM64, NTOKENS4 shape the
mechanical offsets are B=0, C=1024, state=2048, x=18432, dt=18944,
out=18960, empty=19984, full=20000, out-ready=20016, total=20096 bytes.
The reviewer must confirm this representative calculation and every other
frozen shape against line-associated PTX shared offsets.

## Entry, index selection, and barrier initialization

```text
batch, head_cta = cta_id((BATCH,NHEADS))
base_head = head_cta
lane, warp = thread_id((32,5))
```

If `state_batch_indices` is present, every thread reads the **flat** element
`state_batch_indices[batch]`; it does not multiply by the public batch stride
and it does not inspect a token dimension. i32 is sign-extended to i64. If the
pointer is absent, batch is widened to i64. `dst_state_batch_indices` is not
read; final state stores use the selected source `state_batch`. The runtime
pad predicate compares the i64 state slot with sign-extended `pad_slot_id` and
dispatches separate compile-time `IS_PAD=true/false` bodies.

Warp 0 lane 0 initializes:

```text
for stage in static_range(2):
  mbarrier.init.shared.b64(empty[stage], 160)   # 128 compute + 32 TMA lanes
  mbarrier.init.shared.b64(full[stage], 129)    # 128 compute + one tx arrival
mbarrier.init.shared.b64(out_ready, 128)        # four compute warps only
bar.sync 0                                      # all 160 threads
```

No inactive head tail exists because grid.y is exactly nheads and
`HEADS_PER_CTA=1`.

## Raw parity-wait primitive and frozen parity operands

All pipeline waits use the reachable `arrive_and_wait_parity` helper rather
than the standard CUDA barrier wait with nanosleep backoff:

```text
bar_addr = cvta.to.shared.u32(native_barrier_handle)
mbarrier.arrive.shared::cta.b64 _, [bar_addr]
ready = 0
while ready == 0:
  pred = mbarrier.try_wait.parity.shared::cta.b64 [bar_addr], parity
  ready = selp.b32(1,0,pred)
```

The arrive has no consumed token. The loop is a tight retry loop with no
`nanosleep`. At source level TMA lanes own `parity_empty[2]`, compute lanes own
`parity_full[2]` and `parity_out_ready`, and each helper call advances the
corresponding phase. Frozen `-O3` PTX statically expands the callers,
constant-propagates the phase into each `try_wait`, and DCEs every source-level
`parity ^= 1`; it emits no parity-update `xor.b32`.

The exact emitted parity operands are:

| shape | stage sequence | empty/full wait parity | out-ready parity |
| --- | --- | --- | --- |
| DIM64 | `0,1` | `0,0` | `0` |
| DIM128 | `0,1,0,1` | `0,0,1,1` | `0` |

These constants apply independently to producer empty waits and consumer full
waits. Non-Philox entries contain no `xor.b32` at all. Philox entry XORs are
PRNG round operations described separately below; none implements parity or
shared-address swizzling.

## Warp 4: TMA producer and two-stage state pipeline

Before waiting on any empty stage, lane zero issues B, C, and the single x
TensorMap loads. All three target `full[0]`:

```text
cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    B <- tensor_B(0, kv_group, 0, batch), full[0]
cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    C <- tensor_C(0, kv_group, 0, batch), full[0]
cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    x <- tensor_x(0, base_head, 0, batch), full[0]
```

The semantic producer traversal has `NUM_TMA_LOADS` loads, but frozen PTX has
no runtime `tl` loop or backedge. It emits two static regions for DIM64 and
four for DIM128. Their exact topology is:

```text
STATE_STAGE_BYTES = 32*DSTATE_PAD*sizeof(state_t)
BCX_BYTES = 2*NTOKENS*DSTATE_PAD*2 + NTOKENS*DIM*2

DIM64:  tl       = [0,1]
        stage    = [0,1]
        parity   = [0,0]
        non-pad  = [BCX_BYTES+STATE_STAGE_BYTES, STATE_STAGE_BYTES]
        pad      = [BCX_BYTES, 0]

DIM128: tl       = [0,1,2,3]
        stage    = [0,1,0,1]
        parity   = [0,0,1,1]
        non-pad  = [BCX_BYTES+STATE_STAGE_BYTES,
                    STATE_STAGE_BYTES, STATE_STAGE_BYTES, STATE_STAGE_BYTES]
        pad      = [BCX_BYTES, 0, 0, 0]
```

In each static region all 32 TMA lanes arrive/wait on the selected
`empty[stage]`; lane zero then acts:

```text
if not IS_PAD:
  state[stage] <- tensor_state(0, tl*32, base_head, state_batch), full[stage]
  transaction_bytes = STATE_STAGE_BYTES
  if tl == 0:
    transaction_bytes += 2*NTOKENS*DSTATE_PAD*2 + NTOKENS*DIM*2
else:
  omit the state TMA
  transaction_bytes = first_BCX_bytes if tl == 0 else 0

mbarrier.arrive.expect_tx.release.cta.shared::cta.b64
    full[stage], transaction_bytes
```

Thus B/C/x and the first state tile form one full-barrier transaction. Later
state tiles carry only one 32-row state byte count. Pad still loads B/C/x so
the output remains valid, but it does not touch the state TensorMap.

The producer is allowed to advance into stage 1 while compute uses stage 0.
On wraparound it cannot reuse a stage until all four compute warps have
released that stage and all 32 producer lanes have completed the empty-stage
parity wait. The transcription must statically emit these two/four regions and
must not reintroduce a runtime outer `tl` loop.

## Warps 0..3: prologue and cooperative dt publication

Each compute lane pre-arrives once at both empty stages. These 128 arrivals,
together with warp 4's 32 arrivals, complete the initial empty phases.

Each compute thread has `flat_thread=warp*32+lane`. It processes
`idx=flat_thread; idx<NTOKENS; idx+=128`. Since `HEADS_PER_CTA=1` and
`NTOKENS<=8`, only the first NTOKENS flat threads execute one dt episode:

```text
step = idx
head = base_head
dt_bias = optional weight load at head, else zero
dt = weight load at batch*dt_stride_batch + step*dt_stride_mtp + head
dt = add.ftz.f32(dt, dt_bias)
if runtime dt_softplus and dt <= 20:
  exp_arg = mul.ftz.f32(dt, log2(e))
  exp_value = ex2.approx.ftz.f32(exp_arg)
  biased = add.ftz.f32(1.0, exp_value)
  log_value = lg2.approx.ftz.f32(biased)
  dt = mul.ftz.f32(log_value, ln(2))
st.shared.b32 dt[step]
```

The five softplus instructions above are emitted once in each compile-time
pad/non-pad prologue body, with SASS `MUFU.EX2` and `MUFU.LG2`; no generic
`exp`, `log`, or exact-division alternative is permitted.

There is no separate dt barrier. Every compute lane subsequently enters the
first full-stage parity barrier, whose 128 compute arrivals plus TMA
transaction arrival establish visibility of dt, B, C, x, and state stage 0.

The pre-first-full-wait placement is fixed, not scheduler-permissive:

| compile-time body | intermediate-index selection | seed | A / optional D |
| --- | --- | --- | --- |
| pad, PHILOX0 | absent | absent | loaded before the first full wait |
| non-pad, PHILOX0 | runtime-nullable index selection/load | absent | loaded before the first full wait |
| pad, PHILOX10 | absent | absent | loaded before the first full wait |
| non-pad, PHILOX10 | runtime-nullable index selection/load | conditional `ld.global.b64` | loaded before the first full wait |

The intermediate index uses flat `intermediate_state_indices[batch]` and falls
back to `state_batch`. PHILOX0 has no seed or PRNG instruction. The state-head
base arithmetic is also established before entering the retained subpass/token
work.

## Compute traversal and stage lifecycle

The head loop is one static iteration. As on the producer side, the compute
`tl` traversal is two static regions for DIM64 and four for DIM128, with stage
sequence `0,1[,0,1]` and full-wait parity `0,0[,1,1]`. Only `subpass` remains a
runtime extent-two loop in each static region:

```text
arrive_and_wait_parity(full[stage], parity_full[stage])

for subpass in runtime_range(2):       # source explicitly prevents unrolling
  sram_row = subpass*16 + warp*4 + lane//8
  dd = tl*32 + sram_row
  load one logical DSTATE row into float2 recurrence registers
  execute the NTOKENS recurrence loop and optional stores

mbarrier.arrive.shared::cta.b64 empty[stage]  # every compute lane
```

The empty-stage release occurs only after both 16-row subpasses and all token
work for the 32-row state tile are complete. It is an arrive without wait;
warp 4 consumes completion through its next parity wait. Frozen DIM64 has four
`.pragma "nounroll"` subpass regions across pad/non-pad bodies and DIM128 has
eight; there is no emitted outer-TMA-loop backedge.

## State, B, C, and x vector loads

State load address is `state_stage + sram_row*DSTATE_PAD + c0` where `c0` is
pair aligned. Pad or `c0>=DSTATE` initializes the pair to `(0,0)` without a
shared load. Logical pair ownership is not the physical PTX width: nvcc
coalesces adjacent lane/member loads. The following counts are per HPG kernel
entry and already include its compile-time pad and non-pad bodies. “State/B/C
LDS” counts only the line-associated conversion-helper regions; x, dt, and the
epilogue are listed separately.

| frozen module shape | State/B/C LDS sites | conversion after LDS |
| --- | --- | --- |
| BF16, weight BF16, i64, D64/S128/T4 | 20 `ld.shared.v4.b32` | BF16 shift/mask unpack |
| BF16, weight F32, i32, D64/S128/T4 | 20 `ld.shared.v4.b32` | BF16 shift/mask unpack |
| BF16, D128/S128/T4 | 40 `ld.shared.v4.b32` | BF16 shift/mask unpack |
| BF16, D64/S128/T1 | 18 `ld.shared.v4.b32` + 2 `ld.shared.v2.b32` + 4 scalar `ld.shared.b32` | BF16 shift/mask unpack |
| BF16, D64/S128/T2 | 20 `ld.shared.v4.b32` | BF16 shift/mask unpack |
| BF16, D64/S128/T4 | 20 `ld.shared.v4.b32` | BF16 shift/mask unpack |
| BF16, D64/S128/T6 | 20 `ld.shared.v4.b32` | BF16 shift/mask unpack |
| BF16, D64/S128/T8 | 20 `ld.shared.v4.b32` | BF16 shift/mask unpack |
| BF16, D64/S64/T4 | 10 `ld.shared.v4.b32` | BF16 shift/mask unpack |
| BF16, D64/S96/T4 | 10 `ld.shared.v4.b32` + 40 predicated scalar `ld.shared.b32` | BF16 shift/mask unpack; second-tile members 4..7 inactive |
| FP16, D64/S128/T4, Philox10 | 4 state + 16 B/C `ld.shared.v4.b32` | 32 scalar `cvt.f32.f16` for state; BF16 unpack for B/C |
| FP16, D64/S128/T4, Philox0 | 4 state + 16 B/C `ld.shared.v4.b32` | 32 scalar `cvt.f32.f16` for state; BF16 unpack for B/C |
| FP32, D64/S128/T4 | 8 state `ld.shared.v4.b32` + 32 B/C `ld.shared.v2.b32` | no state conversion; BF16 unpack for B/C |
| FP32, D64/S128/T6 | 8 state `ld.shared.v4.b32` + 32 B/C `ld.shared.v2.b32` | no state conversion; BF16 unpack for B/C |

The DIM64 bodies additionally contain four scalar `ld.shared.b16` x sites and
four scalar `ld.shared.b32` dt sites per entry; DIM128 doubles each extent to
eight. DSTATE96 scalar LDS is guarded by the active-member predicate, so padded
columns are never consumed. These physical vector widths must be expressed
directly in TIRx even though each lane still receives the same logical pairs.

The source-defined `xor_swizzle` helper is not called at any of these
addresses. Expected physical addressing is direct row-major DSTATE_PAD with
ordinary add/mad/shift strength reduction selected by nvcc.

## Token recurrence and packed SM100 arithmetic

At the start of each subpass there are five logical views of B token zero, C
token zero, x token zero, dt token zero, and out token zero, but only four
physical shared cursors. Frozen PTX folds B/C into one moving base and addresses
C through a static separation from B. Consequently there is exactly one B/C
row-stride add per emitted token backedge and no instruction associated with
source `.loc 351`:

| shape | B/C add | x add | dt add | out add |
| --- | ---: | ---: | ---: | ---: |
| DSTATE64, DIM64 | `+128` | `+128` | `+4` | `+256` |
| DSTATE96/128, DIM64 | `+256` | `+128` | `+4` | `+256` |
| DSTATE128, DIM128 | `+256` | `+256` | `+4` | `+512` |

These are byte immediates. Let `L=DIM/32` be the number of static `tl` regions.
For every NTOKENS>1 specialization, all `L` non-pad regions and pad regions
`0..L-2` emit the complete B/C+x+dt+out group. The last pad region retains only
the dt `+4`; its B/C, x, and out advances are DCE. Therefore a full entry has
`2L-1` complete groups but `2L` dt advances: `3/4` for DIM64 and `7/8` for
DIM128. NTOKENS1 statically eliminates every advance at source lines 350--354.
The conditional intermediate-state global pointer has its own typed advance
of `nheads*DIM*DSTATE` elements and is not part of the folded shared-cursor
group.

For each token:

```text
dt = ld.shared.b32(*dt_cursor)
dA = ex2.approx.ftz.f32((A*dt)*log2(e))
x = cvt.f32.bf16(ld.shared.b16(x_cursor[dd]))
dtx = mul.ftz.f32(dt,x)
dA2 = (dA,dA)
dtx2 = (dtx,dtx)
out2 = (0,0)

for every in-bounds float2 state pair:
  B2 = BF16x2 shared load/conversion
  C2 = BF16x2 shared load/conversion
  dBx = mul.f32x2(B2,dtx2)
  state2 = fma.rn.f32x2(dA2,state2,dBx)
  out2 = fma.rn.f32x2(state2,C2,out2)

out = add.ftz.f32(out2.x,out2.y)
for offset in (4,2,1):
  peer = shfl.sync.down.b32(mask=-1, value=out, delta=offset, clamp=31)
  out = add.ftz.f32(out,peer)

if member == 0:
  out = fma.rn.ftz.f32(D,x,out)
  st.shared.b32 out_cursor[dd], out
```

`mul.f32x2` and `fma.rn.f32x2` are mandatory native 64-bit-register PTX
instructions on SM100; four scalar replacements are not equivalent for the
transcription. The reduction is only over the eight adjacent lanes belonging
to one row even though `shfl.sync.down` uses the full-warp mask: offsets are
4, 2, 1 and only `member==0` publishes the completed row sum.

`D*x+out` is always one `fma.rn.ftz.f32` (`FFMA.FTZ` in SASS), never a
separate multiply/add alternative. It has one static instruction site in each
pad/non-pad recurrence region: four sites per DIM64 entry and eight per DIM128
entry.

After compute, physical cursor advancement follows the exact complete-group
and final-pad-region exception above. No separate C addition is emitted.

## Direct intermediate and final state stores

Only real DSTATE columns store. Each active tile member owns one
`packed_tile_t`:

- BF16/F16 state: eight elements / 16 bytes per tile member.
- FP32 state: four elements / 16 bytes per tile member.

When an intermediate buffer exists and `IS_PAD=false`, every token converts
and writes each active tile. The destination base is:

```text
icache_idx*intermediate_state_stride_batch
+ step*nheads*DIM*DSTATE
+ head*DIM*DSTATE + dd*DSTATE + col0
```

The source strength-reduces the step term to a moving typed pointer. At the
last token, if runtime `update_state` is true and not pad, it also writes:

```text
state_batch*state_stride_batch + head*DIM*DSTATE + dd*DSTATE + col0
```

The destination-index pointer is intentionally ignored.

For the TIRx registry, `update_state` is a compile-time property of each
`CONFIGS`/`BENCH_CONFIGS` entry.  The transcription partially evaluates the
source runtime predicate at specialization time: `UPDATE_STATE=false` emits no
last-token final-store branch, while `UPDATE_STATE=true` emits the same
last-token non-pad store body without re-testing the fixed flag.  Both shapes
remain independently discoverable and benchmarked; this changes only the
selected instruction shape, not the address, conversion, or store sequence of
the enabled source path.

Let `L=DIM/32` be the number of statically emitted `tl` regions. Let `S` be the
active 16-byte tiles owned by one lane in a state row:

| dtype | DSTATE64 | DSTATE96 | DSTATE128 |
| --- | ---: | ---: | ---: |
| BF16/F16 | 1 | 2 (second tile predicated by active member) | 2 |
| FP32 | 2 | 3 (last padded tile absent) | 4 |

The ordinary direct-store group is mechanically fixed:

| dtype | one branch in one static `tl` | one branch in the full entry |
| --- | --- | --- |
| BF16 | `8*S` scalar `cvt.rn.bf16.f32`; `4*S` `mov.b32 d,{lo,hi}`; `S` `st.global.v4.b32` | multiply every count by `L` |
| F16 | `8*S` scalar `cvt.rn.f16.f32`; `4*S` `mov.b32 d,{lo,hi}`; `S` `st.global.v4.b32` | multiply every count by `L` |
| FP32 | no conversion or pair-pack; `S` `st.global.v4.b32` | multiply the store count by `L` |

Thus the frozen ordinary BF16/F16 DSTATE128/DIM64 entry has, in each branch,
32 scalar conversions, 16 pair packs, and four 128-bit stores; DSTATE64 halves
those values, DSTATE96 retains the DSTATE128 static sites under its predicate,
and DIM128 doubles them. The frozen FP32 DSTATE128/DIM64 entry has eight
`st.global.v4.b32` per branch and no conversion. ptxas selects
`F2FP.{BF16,F16}.F32.PACK_AB` and `STG.E.128` for the corresponding PTX.

Intermediate and final groups are separate emitted CFG regions and ordinary
paths do not CSE their conversion or packing:

| runtime outcome, non-pad | executed groups |
| --- | --- |
| intermediate only | intermediate group at every token; no final group |
| final only | final group only on the last token when `update_state!=0` |
| both | intermediate group at every token and a distinct final group on the last token |
| neither | no ordinary conversion, pack, or state store |

Pad executes none of the index, conversion, packing, or state-store regions.

Philox10 is a separate frozen FP16 path. Only its non-pad body conditionally
loads the 64-bit seed, splits it into low/high 32-bit keys, and emits PRNG. In
each static `tl`, inside each dynamic `sp` iteration but before the token loop,
nvcc unconditionally hoists four Philox4x32 invocations for signed-narrowed i32
counters at:

```text
state_ptr_offset + dd*DSTATE + {col0+0, col0+4, col0+64, col0+68}
```

The helper boundary narrows `state_ptr_offset` to signed i32; counter low/high
construction therefore uses the narrowed low word and sign-high `shr.s32`, not
an invented 64-bit address calculation. One static region contains 69
`mul.hi.u32`, 61 PRNG `mul.lo.s32`, 136 `xor.b32`, 27 `add.s32`, and four
sign-high `shr.s32` sites. The two DIM64 regions total
`138/122/272/54/8`; these are the only 272 XOR sites in this module.

Each intermediate branch and each final branch separately emits eight
`cvt.rs.f16x2.f32` and two `st.global.v4.b32` per static `tl`, reusing the same
four hoisted random-result groups. Across both branches and both static regions
the entry has 32 conversions and eight stores. Intermediate-only, final-only,
and both follow the same branch outcomes as the ordinary table, but the
non-pad neither-store path still executes the unconditionally hoisted PRNG.
The compile-time pad body and every PHILOX0 module contain no seed load, PRNG
round, stochastic conversion, or PRNG XOR.

## Inline epilogue and output barrier

After all state stages are released, every compute lane performs one
`arrive_and_wait_parity(out_ready, parity_out_ready)`. The barrier count is
128, so this synchronizes the four compute warps and makes all shared out rows
visible. There is no TMA-warp participation.

Each compute warp semantically owns `step=warp; step<NTOKENS; step+=4`, but the
frozen epilogue has no runtime step loop. T1/T2/T4 emit one static episode per
pad/non-pad body (two source episodes per entry); T6/T8 emit `step` and
`step+4` episodes in each body (four per entry). Inactive warps are predicated.
For each episode:

```text
out_base = batch*out_stride_batch + step*out_stride_mtp + head*DIM
z_base   = batch*z_stride_batch   + step*z_stride_mtp   + head*DIM
OUTPUT_VECTOR = 2 for DIM64, 4 for DIM128
d = lane*OUTPUT_VECTOR
if DIM64 and NTOKENS == 1:
  out_values = two scalar ld.shared.b32 out[step,d:d+2]
elif DIM64:
  out_values = ld.shared.v2.b32 out[step,d]
else:
  out_values = ld.shared.v4.b32 out[step,d]
if z is present:
  z_values = ld.global.v{2,4}.b16 z[z_base+d]
for k in static_range(OUTPUT_VECTOR):
  if z:
    zf = cvt.f32.bf16(z[k])
    exp_neg = ex2.approx.ftz.f32((0-zf)*log2(e))
    sigmoid = div.approx.ftz.f32(1,1+exp_neg)
    out[k] = out[k] * (zf*sigmoid)
  output[k] = cvt.rn.bf16.f32(out[k])
st.global.v{2,4}.b16 output[out_base+d]
```

The global widths do not share the T1 exception: DIM64 z/output remain
`ld.global.v2.b16` and `st.global.v2.b16`, while DIM128 remains v4. Across the
two pad/non-pad bodies T1/T2/T4 contain two global output-store sites and
T6/T8 contain four. T1 has four scalar shared-output loads (two values per
body); DIM64/T>1 uses the frozen `ld.shared.v2.b32` family, and DIM128 uses
`ld.shared.v4.b32`. The transcription emits these static episodes directly.

The source contains a second out-ready parity wait only between heads. Since
`HEADS_PER_CTA=1`, `h < HEADS_PER_CTA-1` is compile-time false and that second
wait is absent from every current specialization.

## Synchronization ledger

For one non-pad DIM64 CTA:

1. One thread initializes five barriers; all 160 threads execute one CTA sync.
2. Each compute lane pre-arrives empty stage 0 and 1.
3. TMA lane zero issues B/C/x to full stage 0.
4. All TMA lanes arrive/wait empty stage 0 with parity 0; lane zero issues state rows 0..31
   and full-stage transaction bytes including B/C/x.
5. TMA proceeds to empty stage 1 with parity 0 and issues state rows 32..63
   while compute arrives/waits full stage 0 parity 0 and processes its two
   subpasses.
6. Compute releases empty stage 0, waits/processes full stage 1 parity 0, then
   releases empty stage 1. DIM128 statically emits two additional regions after
   wraparound, using parity 1 for stage 0 then stage 1.
7. All compute lanes arrive/wait out-ready, execute the distributed epilogue,
   and exit. Warp 4 has already exited after its final TMA transaction.

Pad follows the same barrier phases and byte-accounting structure but omits
all state TMA and initializes state registers to zero. No thread may return
early before satisfying its role's barrier ledger.

## Copy / compute / synchronization schedule

```text
host: build four descriptors, set dynamic-smem attribute, launch 160 threads

entry: select flat state index; initialize 2 empty + 2 full + out barrier; CTA sync

warp4 lane0: issue B/C/x -> full0
compute: arrive empty0/empty1; cooperatively publish dt

warp4 all: wait empty0 parity=0
warp4 lane0: state chunk0 -> full0; arrive.expect_tx(BCX + state bytes)
warp4 all: wait empty1 parity=0
warp4 lane0: state chunk1 -> full1; arrive.expect_tx(state bytes)

compute all: wait full0 parity=0; runtime subpass0/1; release empty0
compute all: wait full1 parity=0; runtime subpass0/1; release empty1

DIM128 only:
  statically emit chunks2/3 on stages0/1 with parity=1
  statically emit compute full0/full1 waits with parity=1 and release again

compute all: wait out_ready parity=0; static z/SILU conversion/output episodes
```

## Reviewer evidence requirements

The independent reviewer must use normal FlashInfer SM100a JIT with
`-DNDEBUG -O3 -lineinfo`, not a debug build. For each of the 14 unique module
keys it must freeze line-associated PTX plus horizontal-only SASS functions,
record toolchain/source/config hashes, and verify both directions:

```text
CUDA source/helper operation -> sketch statement -> PTX/SASS evidence
PTX/SASS instruction region -> source/helper line -> sketch statement
```

The review has five independent checks:

1. roles and launch geometry;
2. storage placement, shared ABI, descriptors, and grid-constant parameters;
3. barriers, phases, byte counts, parity, and stage lifetime;
4. operation dataflow, pointer strength reduction, pad/output/store topology;
5. exact instruction selection, vector width, modifiers, conversions, Philox,
   f32x2, shuffle, and fast math.

All five must PASS with no findings before TIRx implementation begins. Any
ambiguous `may`, helper shorthand without expanded PTX, or claim of an
unemitted XOR swizzle is a required fix, not a documentation preference.
