<!--
Copyright (c) 2026 The TIRx Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the
specific language governing permissions and limitations
under the License.

This design sketch documents a TIRx port of MSA's
python/fmha_sm100/cute/src/sm100/fwd/combine.py
SparseAttentionForwardCombine.
-->

# msa_sparse_atten_fwd_combine_sm100: coarse WASP pipeline sketch

This non-executable design sketch describes the storage layout, thread roles,
control flow, and PTX-level operations of
[`tirx_kernels/msa/sparse_atten_fwd_combine.py`](../../../tirx_kernels/msa/sparse_atten_fwd_combine.py).
That TIRx module is the authoritative implementation.

The port covers **one kernel**: `SparseAttentionForwardCombine.kernel`
(`combine.py:466-1125`) together with its pipeline-stage loader `load_O_partial`
(`:1128-1161`) and the launcher `__call__` (`:243-455`) -- K2 of the sparse
attention forward, which reduces the per-split partials K1 wrote and restores
the real head-dim column order K1 permuted away.

Instantiations: the host's compile key (`:1339-1355`) is
`(D, k_block, tile_m, topk, partial_dtype, out_dtype, has_cu_seqlens,
has_seqused, has_lse, return_temperature_lse, has_split_counts,
has_output_scale, use_pdl, min_blocks_per_mp)`. Production pins most of it:
`D=128` so `k_block_size=128`, `tile_m=64`, `stages=2`, `num_threads=256`
(the constructor default at `:46`; the host never passes it)
(`:1327-1337`, `:1365-1381`), out dtype bf16 (`interface.py:920, :1481`),
varlen, `split_counts` present, `lse_out` present, `use_pdl=True`
(`interface.py:972, :1532`). The axes that remain are **topk** (4, 8, 16 or 32 --
`can_implement` demands `(tile_m*topk) % num_threads == 0`, `:106`), **partial
dtype**, **temperature**, **output_scale**, and **seqused**. Only partial dtype
changes device code structurally: it selects one of three STG.128 fake-column
maps, the staging layout, and the copy width.

Fixed specialization for this sketch: **fp32 partials, topk 16, no temperature,
no output scale, no seqused**. Line info comes from
`.porting/sparse_atten_fwd_combine/ptx_lineinfo/fp32p_t16/` (585 `.loc`; **1327
instruction lines, 73 predicated, 1254 net** by the "instruction lines minus
predicated lines" convention). The filter is: every semicolon-terminated line
that is not a directive, comment or label, counting a leading `@%p` predicate as
part of its instruction. A stricter filter that also requires the first token to
match an opcode pattern reaches 1319/1246 on the same bytes; both numbers are
recorded so a recount does not read the delta as an error. Six sibling exports
under `ptx_lineinfo/` cover the other axes.

Accepted target SM100/B200. Out of scope, with the predicate that excludes each:
the batched 4-D mode and its `PackGQAComb.load_LSE` staging (`:563-616`), which
`cu_seqlens is not None` excludes; the `use_pdl=False` scalar-scatter epilogue
(`:967-985`), which `use_pdl=True` excludes; out dtypes fp16 and fp32, which the
interface excludes by allocating bf16; `num_splits_dynamic_ptr` (rejected
outright at `:527-529`), `varlen_batch_idx` and `semaphore_to_reset`, all three
pinned `None` by the host (`:1465-1473, :1492-1494`), which also compiles out the
semaphore-reset block at `:518-526`; and every `not is_even_k` predication path
-- the `tOpO` k-mask at `:716-721`, `tOpO_store` at `:1099-1105`, and the
`tOpO[k]` term at `:1154` -- which `head_dim % k_block_size == 0` excludes, so
the export emits none of them. Tile (`Tx`) primitives are out of scope
everywhere.

## Pipeline at a glance

There is **no warp specialization**: all 256 threads of a CTA run the same
program. The roles below are the phases that program moves through, not
different warps.

| Phase / role | Program | Publication / reuse edges |
| --- | --- | --- |
| **prologue, all 256 threads** (`:497-548`) | decode `(m_block, k_block, batch)`; take the dynamic SMEM base; build `SeqlenInfo`; compute `max_idx = seqlen * num_head`; guard the whole body on `m_block * 64 < max_idx`; wait on K1 | `griddepcontrol.wait` is the only cross-kernel edge, and the export carries exactly one (`.loc 1 550`) |
| **step 1, all 256 threads** (`:550-673`) | stage `LSE_partial` for 64 flat rows x `topk` splits into `sLSE`, filling `-inf` for a split at or past the row's count and for a row past `max_idx` | Publishes `sLSE`. One `cp.async.commit_group` closes the group. |
| **step 2, all 256 threads** (`:675-740`) | precompute per-row `(m_idx, head_idx, split_count, raw GMEM element pointer)`; prime the partial pipeline `stages-1` deep | Publishes `sO[stage 0]`; one commit group per primed stage |
| **step 3, all 256 threads** (`:741-765`) | `cp.async.wait_group 1`, `bar.sync`, then read `sLSE` back transposed over the split axis | Consumes `sLSE`; the barrier is what makes the staged tile visible |
| **step 4, all 256 threads** (`:767-873`) | per row: max over splits, index of the last live split, `exp2` scales, sum, `final_lse`, normalized scales | **Overwrites `sLSE` in place with the normalized scales** (`:822`) and publishes `sMaxValidSplit` (`:872`). Both publications cross a partition boundary: they are written on the **s2r** row map (`tidx >> 2`) and read by step 6 on the **O-partial** row map (`(tidx >> 5) + 8m`), so `:907` is load-bearing twice over -- either one alone would require it. |
| **step 5, `k_block == 0` only** (`:875-902`) | one thread per row stores `LSE_out` (and the temperature LSE) | Publishes to global memory. This is the authoritative public LSE. |
| **step 6, all 256 threads** (`:904-953`) | `bar.sync`, read `sMaxValidSplit`, then walk live splits: issue the next stage, commit, wait, copy stage to registers, fp32 FMA into the accumulator | Consumes `sLSE` scales and `sO`. **No barrier inside the loop** (`:937`) -- but the reason is narrower than "each thread reads its own data": that holds for `sO`, which the thread's own `cp.async` wrote, and it is what the source's comment at `:937` is scoped to. The `sLSE` scale loads at `:926` do cross threads; they are safe because `:907` already published them, before the loop. |
| **step 7, all 256 threads** (`:955-1125`) | 7a scatter the accumulator into `sO_perm` in real column order; `bar.sync`; 7b read `sO_perm` back with the store partitioning and issue 128-bit global stores | `sO_perm` occupies `[73984, 92416)` and `sO` `[8448, 73984)` -- disjoint, so 7a overwrites nothing the pipeline still owns; the `wait_group 0` at `:952` retires the in-flight prefetches rather than protecting against aliasing |

**One CTA owns 64 flat `(q, head)` rows crossed with one 128-column k-block.**
The grid flattens `(seqlen, num_head)` with the head axis innermost, so a CTA's
64 rows are 64 consecutive `(q, head)` pairs, not 64 queries.

## Primitive vocabulary

```python
# structural
smem_pool.alloc(shape, dtype)            # dynamic shared allocation
tile[stage]                              # stage view of the staging buffer
identity(shape)                          # coordinate tile, for thread->element maps

# directional copies
cp_async_ca(smem, gmem, bytes)           # cp.async.ca.shared.global (LSE, 4 B)
cp_async_cg(smem, gmem, bytes)           # cp.async.cg.shared.global (partials, 16 B)
store_shared(smem, reg, width)           # st.shared.u32 | st.shared.f32 | st.shared.v4.f32
load_shared(smem, width)  -> reg         # ld.shared.u32 | ld.shared.f32 | ld.shared.v4.{f32,b32}
load_global(gmem)         -> reg         # ld.global.u32
store_global(gmem, reg, width)           # st.global.f32 | st.global.v4.b32

# compute
exp2(x)                                  # ex2.approx.ftz.f32
log2(x)                                  # lg2.approx.ftz.f32
recip(x)                                 # rcp.rn.f32
fma(a, b, c)                             # fma.rn.f32
maxf(a, b)                               # max.f32
max_nan(a, b)                            # max.NaN.f32  (propagates NaN)
max_i32(a, b)                            # max.s32
cvt_bf16x2(a, b)                         # cvt.rn.bf16x2.f32
divmod_fast(idx, dm)                     # mul.hi.u32 + sub + shr + add + shr
                                         #   + mul.lo + sub  (7 instructions)
div_i32(a, b)                            # div.s32, dividend known non-negative
floordiv_i32(a, b)                       # div.s32 + a 7-instruction sign fixup

# schedule
commit_group()                           # cp.async.commit_group
wait_group(n)                            # cp.async.wait_group n
barrier()                                # bar.sync 0
grid_dep_wait()                          # griddepcontrol.wait
warp_reduce_max(x, 4) / warp_reduce_sum(x, 4)
                                         # shfl.sync.bfly.b32 laneMask 2 then 1, c=31
last_live_split(regs)                    # setp.neu.f32 + or.b32 + selp.b32 chain
                                         #   over the thread's 4 split slots

# FIVE thread->coordinate maps. Conflating any two is the error this sketch
# is most at risk of, so every shared-memory index below names the one it uses.
lse_row(tidx)      = tidx & 63           # step 1 LSE stage,  order=(1,0) (:179-186)
lse_split(tidx, i) = (tidx >> 6) + 4*i   #   .loc 1 627
s2r_row(tidx)      = tidx >> 2           # step 3 read-back,  order=(0,1) (:196-203)
s2r_split(tidx, i) = (tidx & 3) + 4*i    #   .loc 1 755
o_row(tidx, m)     = (tidx >> 5) + 8*m   # O-partial staging, m in 0..7 (:120-134)
o_col(tidx, v)     = (tidx & 31) * 4 + v #   .loc 1 681, 682
store_row(tidx, m) = (tidx >> 4) + 16*m  # O output store,    m in 0..3 (:136-163)
store_col(tidx)    = (tidx & 15) * 8     #   .loc 1 1092
# Step 6 indexes sLSE by an ABSOLUTE split `s` crossed with o_row.
#
# Four edges cross a partition boundary, each covered by exactly one barrier:
#   sLSE staged (lse_*) -> read back (s2r_*)              :747
#   sLSE scales (s2r_*) -> step 6 (absolute s, o_row)     :907
#   sMaxValidSplit (s2r_row) -> step 6 (o_row)            :907
#   sO_perm (o_*) -> 7b (store_*)                         :1088
# Everything else a thread touches it wrote itself -- which is why the
# accumulate loop needs no barrier of its own.
```

### The reductions are 4-lane butterflies, in laneMask 2 then 1 order

`smem_threads_per_col_lse = num_threads // m_block_smem = 256 // 64 = 4`
(`:193`), so every reduction is over four lanes and the export carries exactly
two shuffles per reduction:

```
shfl.sync.bfly.b32 %f681, %f679, 2, 31, -1;
shfl.sync.bfly.b32 %f685, %f684, 1, 31, -1;
```

`laneMask` descends 2 then 1, the membermask is `-1`, and `c = 31` (full-warp
clamp). Three reductions use this shape: the LSE max (`.loc 1 782`), the
`max_valid_split` index (`.loc 1 795`, on `b32` integers), and the scale sum
(`.loc 1 810`).

### The row divmod is a FastDivmod, and the head divmod is a real division

`decode_flat_row_idx` (`:457-464`) divides the flat row index by `num_head`
through a host-built `FastDivmodDivisor`, and the export lowers it to the
classic two-shift magic sequence (`.loc 1 463 28`):

```
mul.hi.u32 %r105, %r9,   %r86      // idx * multiplier
sub.s32    %r106, %r9,   %r105
shr.u32    %r107, %r106, %r102     // shift_1  (a .u8 parameter)
add.s32    %r108, %r107, %r105
shr.u32    %r109, %r108, %r104     // shift_2  (a .u8 parameter)
mul.lo.s32 %r110, %r109, %r85      // quot * divisor
sub.s32    %r111, %r9,   %r110     // remainder
```

The multiplier, divisor and the two shifts arrive as kernel parameters
(`ld.param.u32` for the first two, `ld.param.v2.u8` for the shifts), so the port
computes the same four values on the host and emits the same seven
instructions. `head_idx // qhead_per_kvhead` is **not** given that treatment: it
stays a genuine `div.s32` against a runtime parameter (`.loc 1 642 53`,
`.loc 1 709`), because `qhead_per_kvhead` reaches the kernel as a plain `Int32`.

### The shared storage is dynamic, and `sLSETemperature` is always allocated

`SharedStorage` (`:383-397`) reads like a static struct but
`cutlass.utils.SmemAllocator` takes it from the dynamic pool, so the TIRx form
is a `K.smem_pool()` allocation plus the `tirx.use_dyn_shared_memory` launch
tag. `sLSETemperature` is a field of that struct unconditionally -- the
temperature axis gates the *use*, not the allocation -- and the export proves
it: with temperature off, `sMaxValidSplit` still starts at byte 8192
(`st.shared.u32 [%r318+8192]`, `.loc 1 872`), which is `sLSE` (4096) plus an
equally sized `sLSETemperature` (4096). A port that allocated it conditionally
would shift every later offset by 4 KB.

Byte map for the fixed specialization:

| field | offset | size | derivation |
| --- | --- | --- | --- |
| `sLSE` | 0 | 4096 | `cosize(smem_layout_lse) = topk * tile_m = 1024` fp32 |
| `sLSETemperature` | 4096 | 4096 | same layout, allocated unconditionally |
| `sMaxValidSplit` | 8192 | 256 | `tile_m` int32 |
| `sO` | 8448 | 65536 | `tile_m * k_block * stages = 64*128*2` fp32 |
| `sO_perm` | 73984 | 18400 | `cosize(smem_layout_perm) = 9200` bf16; the sketch's `(64, 144)` rectangle is a 32-byte superset with the same base and total |
| total | | 92416 | ~90.25 KB, which is why `stages=2` and not 4 |

### `.file` map of the export

`.file 1` = `combine.py`, `.file 2` = `src/common/seqlen_info.py` (the two
`cu_seqlens` loads at `:35` and `:45`, plus `:66` for `offset_batch`'s
`domain_offset`), `.file 3` = `src/common/utils.py` (`elem_pointer`), `.file 4`
= `src/common/copy_utils.py`, which is where the three fake-column maps are
defined (`:861`, `:878`, `:897`) -- `tma_utils.py` only re-exports them. Every
`.loc` cited below is from `ptx_lineinfo/fp32p_t16/`.

## Thread and value partitioning

Every per-thread count below is a static consequence of the tiled-copy layouts
in `_setup_attributes` (`:110-240`), and each is confirmed by an opcode count in
the export.

| partition | derivation | per thread | export evidence |
| --- | --- | --- | --- |
| LSE staging (`gmem_tiled_copy_LSE`, `:164-185`) | `m_block_smem = 64`, threads `(4, 64)`, `order=(1,0)`: row `tidx & 63`, split base `tidx >> 6` | 4 splits x 1 row | `cp.async.ca x4` (`.loc 1 653`); map at `.loc 1 627` |
| LSE s2r (`s2r_tiled_copy_LSE`, `:196-203`) | `smem_threads_per_col_lse = 4`, atom `(4, 64)`, `order=(0,1)`: row `tidx >> 2`, split base `tidx & 3` -- **transposed against staging** | 4 splits x 1 row | `ld.shared.f32 x4` (`.loc 1 757`); map at `.loc 1 755` |
| partial staging (`gmem_tiled_copy_O_partial`, `:120-134`) | `async_copy_elems = 128/32 = 4`, `gmem_threads_per_row = 128/4 = 32`, threads `(8, 32)` | `num_rows = 8` rows x 1 k-tile x 4 values | `ld.global.u32 x8` (`.loc 1 708`), `cp.async.cg x16` = 8 rows x 2 call sites (`.loc 1 1155`) |
| output store (`gmem_tiled_copy_O`, `:136-163`) | `output_copy_elems = 128/16 = 8`, `gmem_threads_per_row_o = 16`, threads `(16, 16)` | 4 rows x 1 k-tile x 8 values | `ld.shared.v4.b32 x4` (`.loc 1 1111`), `st.global.v4.b32 x4` (`.loc 1 1125`) |

The two output partitionings differ on purpose (`:136-140`): a 128-bit
transaction is 4 fp32 partials but 8 bf16 outputs, so 7a writes `sO_perm` with
the *partial* partitioning and 7b reads it back with the *output* one. The
`bar.sync` at `:1088` is what makes that re-partition legal.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================
variant = specialize(HEAD_DIM=128, K_BLOCK=128, TILE_M=64, STAGES=2,
                     NUM_THREADS=256, TOPK=16, partial="f32", out="bf16",
                     temperature=False, output_scale=False, seqused=False,
                     use_pdl=True, target="sm_100a")
# instruction_selection: none; extent: the compile-key constants (:1339-1355)
NUM_ROWS   = 8      # partial-staging rows per thread
NUM_VALS   = 4      # fp32 elements per 128-bit copy
SPLITS_PT  = 4      # LSE splits per thread  (topk / (256 // 64))
LANES_COL  = 4      # threads per LSE column -> reduction width
LOG2_E     = 0f3FB8AA3B
LN_2       = 0f3F317218

# Runtime ABI. Extents and strides are explicit scalars; the buffers are flat.
#   o_partial     [topk, total_q, head_q, 128]  f32  read-only
#   lse_partial   [topk, total_q, head_q]       f32  read-only
#   o_out         [total_q, head_q, 128]        bf16 write-only
#   lse_out       [total_q, head_q]             f32  write-only
#   cu_seqlens    [batch+1]                     i32  read-only
#   split_counts  [total_q, head_kv]            i32  read-only
#   (lse_temperature_partial / _out, seqused, output_scale: present in the
#    signature, unread in this specialization)
# Scalars: stride_op_split/q/h, stride_lp_split/q, stride_o_q/h, stride_l_q,
#          stride_sc_q, qhead_per_kv, total_q, head_q, num_batches,
#          head_div_mul, head_div_div, head_div_s1, head_div_s2

launch_config = launch(grid=(ceil_div(total_q * head_q, 64), 1, num_batches),
                       block=(256, 1, 1), dynamic_smem_bytes=92416,
                       pdl=True, min_blocks_per_sm=None)
# instruction_selection: none; extent: static launch metadata (:414-454).
#   grid.y is `ceil_div(head_dim, k_block)` = 1 here, kept as a real axis so the
#   `k_block == 0` guard on the LSE store stays a guard and not a constant.
#   `min_blocks_per_mp` is 0 unless an output scale is present (:1337).

# ===========================================================================
# Storage.  One dynamic allocation, five fields, byte offsets above.
# ===========================================================================
sLSE        = smem_pool.alloc((TOPK, TILE_M), "float32")      # :385-387
sLSETemp    = smem_pool.alloc((TOPK, TILE_M), "float32")      # :388-390 (unread here)
sMaxValid   = smem_pool.alloc((TILE_M,), "int32")             # :391
sO          = smem_pool.alloc((STAGES, TILE_M, K_BLOCK), "float32")   # :392-394
sO_perm     = smem_pool.alloc((TILE_M, K_BLOCK + 16), "bfloat16")     # :395-397
# instruction_selection: none (declarations); extent: one dynamic pool per CTA.
#   The +16 row pad on sO_perm is the bank-conflict break for 7a (:374-382).
#   sLSE's swizzle is `(3,2,3)` composed over an atom of (min(topk,8), 64)
#   (:208-222); the port reproduces the resulting element order, not the
#   swizzle object.

def kernel():
    tidx = thread_id(extent=256)                    # :497
    # instruction_selection: mov.u32 %tid.x; extent: scalar
    m_block, k_block, batch = cta_id(axis=(x, y, z))  # :498
    # instruction_selection: mov.u32 %ctaid.{x,y,z}; extent: scalar

    # -----------------------------------------------------------------------
    # Prologue: varlen bounds and the K1 dependency (:531-550)
    # -----------------------------------------------------------------------
    q_offset = load_global(cu_seqlens, batch)          # seqlen_info.py:35
    # instruction_selection: ld.global.u32 (.loc 2 35); extent: one scalar
    q_end    = load_global(cu_seqlens, batch + 1)      # seqlen_info.py:45
    # instruction_selection: ld.global.u32 [addr+4] (.loc 2 45); extent: one
    #   scalar.  The two loads are adjacent, so the second reuses the first
    #   address with a +4 displacement.
    seqlen  = q_end - q_offset
    max_idx = seqlen * head_q                          # :542
    if m_block * TILE_M >= max_idx:                    # :547
        return
    # instruction_selection: shl.b32 by 6 then setp.le.s32 with the operands
    #   swapped -- the compare emitted is `max_idx <= m_block*64` (.loc 1 547)
    #   -- and a predicated branch; extent:
    #   CTA-uniform.  Guards the entire body, so a skipped CTA reaches no
    #   barrier -- which is what makes the four bar.syncs below safe.
    grid_dep_wait()                                    # :550
    # instruction_selection: griddepcontrol.wait (.loc 1 550); extent: one
    #   instruction, CTA-wide.  The export carries exactly ONE griddepcontrol
    #   and NO launch_dependents: this kernel waits on K1 and signals nothing.

    # -----------------------------------------------------------------------
    # Step 1: stage LSE_partial, -inf past a row's split count (:618-673)
    # -----------------------------------------------------------------------
    for m in range(1):                    # rows per thread; unrolled (:636)
        idx = m_block * TILE_M + lse_row(tidx)
        if idx < max_idx:                                        # :639
            # instruction_selection: setp.ge.s32 + a branch taken to the
            #   whole-row -inf fill (.loc 1 639) -- the compare is emitted in
            #   the inverted sense; extent: scalar
            m_idx, head_idx = divmod_fast(idx, head_divmod)      # :463
            # instruction_selection: mul.hi.u32 + sub.s32 + shr.u32 + add.s32 +
            #   shr.u32 + mul.lo.s32 + sub.s32 (.loc 1 463 28); extent: seven
            #   scalar instructions, the FastDivmod expansion
            row_count = load_global(split_counts,
                                    (q_offset + m_idx) * stride_sc_q
                                    + div_i32(head_idx, qhead_per_kv))
            # instruction_selection: div.s32 for the head-group index
            #   (.loc 1 642 53) + cvt.s64.s32 + mul.lo.s64 + add.s64 + shl.b64 +
            #   ld.global.u32 (.loc 1 642); extent: one scalar per row.
            #   The division is real: qhead_per_kvhead is a runtime parameter,
            #   so it gets no magic-number treatment.
            for s in range(SPLITS_PT):                           # :650, unrolled
                si = lse_split(tidx, s)
                if si < TOPK and si < row_count:                 # :652
                    # instruction_selection: one compare per split and a branch
                    #   -- setp.ge.s32 for the first slot and setp.lt.s32 for
                    #   the other three (.loc 1 652); extent: 4 per thread.  The
                    #   `si < TOPK` half folds away at compile time: a thread's
                    #   splits are {c, c+4, c+8, c+12} with c < 4, so si <= 15 is
                    #   statically below topk = 16.  There is no and.pred here;
                    #   the module's eight all belong to step 2.
                    cp_async_ca(sLSE[si, lse_row(tidx)],
                                lse_partial + (q_offset + m_idx) * stride_lp_q
                                            + head_idx
                                            + si * stride_lp_split, 4)
                    # instruction_selection:
                    #   cp.async.ca.shared.global [smem], [gmem], 4, 4
                    #   (.loc 1 653); extent: 4 per thread, one per split slot.
                    #   CA is forced here, not chosen: `cp.async.cg` exists
                    #   only at a 16-byte cp-size, so a 4-byte element has no
                    #   other spelling.  The partial rows below are 16 bytes and
                    #   do take `cg`.
                else:
                    store_shared(sLSE[si, lse_row(tidx)], -inf)
                    # instruction_selection: st.shared.u32 (.loc 1 665);
                    #   extent: one per masked split.  A shared store, not a
                    #   copy: the value is an immediate, so no memory is read.
        else:
            for s in range(SPLITS_PT):                           # :669
                store_shared(sLSE[lse_split(tidx, s), lse_row(tidx)], -inf)
                # instruction_selection: st.shared.u32 (.loc 1 670); extent:
                #   4 per thread.  A separate site from :665 -- this is the
                #   whole-row-out-of-range fill, and the export keeps them
                #   distinct.
    commit_group()                                               # :673
    # instruction_selection: cp.async.commit_group (.loc 1 673); extent: one.
    #   Closes the LSE group so the partial stages below queue behind it.

    # -----------------------------------------------------------------------
    # Step 2: per-row indices and the pipeline prime (:679-740)
    # -----------------------------------------------------------------------
    for m in range(NUM_ROWS):             # unrolled (:691)
        idx = m_block * TILE_M + o_row(tidx, m)
        if idx >= max_idx:                                       # :694
            hidx[m], midx[m], split_count[m], row_ptr[m] = -1, 0, 0, 0
            # instruction_selection: setp.ge.s32 x8 with a predicated branch per
            #   row, each preceded by mov.u64/mov.pred/mov.b32 seeding the
            #   0/false/0 defaults (.loc 1 694, 32 instructions); extent: 8 per
            #   thread.  A real branch per row, not an if-conversion: the
            #   defaults are materialized first and the else arm is jumped over.
        else:
            midx[m], hidx[m] = divmod_fast(idx, head_divmod)     # :700
            split_count[m] = load_global(split_counts,
                                         (q_offset + midx[m]) * stride_sc_q
                                         + floordiv_i32(hidx[m], qhead_per_kv))
            # instruction_selection: the SIGNED FLOOR division expands to eight
            #   instructions per row -- div.s32 + mul.lo.s32 + setp.ne.s32 +
            #   xor.b32 + setp.lt.s32 + and.pred + selp.s32 + add.s32
            #   (.loc 1 709 44, 64 instructions over 8 rows; the line's other
            #   8 add.s32 at column 24 are the `offset + m_idx` row index) --
            #   followed by
            #   ld.global.u32 (.loc 1 708); extent: 8 loads and 8 expansions per
            #   thread.  Step 1 divides the same way in source and gets a BARE
            #   div.s32 (.loc 1 642 53) because its dividend is a fresh divmod
            #   remainder, known non-negative; here the dividend is read back
            #   from an rmem tensor, so the sign correction survives.
            row_ptr[m] = elem_pointer(o_partial,
                                      (midx[m], k_block * K_BLOCK, 0, hidx[m]))
            # instruction_selection: cvt.s64.s32 + cvt.u64.u32 + mul.lo.s64 +
            #   add.s64 for the crd2idx arithmetic (.loc 1 711), then add.s64 +
            #   shl.b64 for the pointer add and element width (.loc 3 371,
            #   `elem_pointer`); extent: 10 instructions per row, 80 static.
            #   Held in registers for the whole kernel, but the loop re-derives
            #   BOTH the split displacement and this thread's column from it --
            #   `(tidx & 31) * 4` is rematerialized inside the loop body every
            #   iteration (`shl.b32` + `and.b32 ..., 124` at .loc 1 681 15) and
            #   added into the source address at .loc 1 1142.  Dropping the
            #   column term would make all 32 threads of a column group read
            #   the same 16 bytes.
    for stage in range(STAGES - 1):                              # :736, unrolled
        if stage < TOPK:
            load_O_partial(split=stage, stage=stage)
        commit_group()                                           # :739
        # instruction_selection: cp.async.commit_group (.loc 1 739); extent:
        #   one per primed stage.  One group per stage is what makes
        #   `wait_group(STAGES-1)` below mean "the oldest stage has landed".

    # -----------------------------------------------------------------------
    # Step 3: publish the staged LSE and read it back transposed (:746-765)
    # -----------------------------------------------------------------------
    wait_group(STAGES - 1)                                       # :746
    # instruction_selection: cp.async.wait_group 1 (.loc 1 746); extent: one.
    #   Leaves the newest partial stage in flight and retires the LSE group.
    barrier()                                                    # :747
    # instruction_selection: bar.sync 0 (.loc 1 747); extent: CTA-wide.  The
    #   only reason it is needed: step 1 partitions sLSE by (split, row) and
    #   step 3 reads it by (row, split), so a thread reads what other threads
    #   wrote.
    for s in range(SPLITS_PT):
        lse_reg[s] = load_shared(sLSE[s2r_split(tidx, s), s2r_row(tidx)])
        # instruction_selection: ld.shared.f32 x4 at +0/1024/2048/3072
        #   (.loc 1 757); extent: 4 per thread -- and they are NOT the elements
        #   this thread staged.  The two tiled copies over sLSE carry
        #   deliberately transposed thread maps: `order=(1,0)` for the gmem
        #   stage (:179-186) against `order=(0,1)` for the read-back
        #   (:196-203).  Staging owns row `tidx & 63` and split base
        #   `tidx >> 6` (.loc 1 627: `shr.u32 %r6,%r1,6`, `and.b32 %r91,%r1,63`);
        #   the read-back owns row `tidx >> 2` and split base `tidx & 3`
        #   (.loc 1 755: `shr.u32 %r41,%r1,2`, with `%r7 = tidx & 3`).  The two
        #   coincide for exactly FOUR threads -- tidx 0, 85, 170 and 255, the
        #   alternating bit patterns, where `tidx & 63 == tidx >> 2` and
        #   `tidx >> 6 == tidx & 3` hold together.  For the other 252 the two
        #   element sets are disjoint.  This is what the :747 barrier publishes,
        #   and it is what puts the four split-groups of ONE row on lanes
        #   t..t^3 so the 4-lane butterfly below reduces a row rather than four
        #   unrelated rows.  A port that re-read its own staged elements would
        #   make :747 look removable and would silently reduce the wrong set.

    # -----------------------------------------------------------------------
    # Step 4: the split reduction (:779-873)
    # -----------------------------------------------------------------------
    for m in range(1):                    # rows per thread
        lse_max = warp_reduce_max(max_nan(lse_reg[0..3]), LANES_COL)   # :782-787
        # instruction_selection: max.NaN.f32 x3 over the 4 register values
        #   (.loc 1 785), then the butterfly at .loc 1 782, whose two steps are
        #   NOT the same combine: laneMask 2 is shfl.sync.bfly.b32 + setp.le.f32
        #   + setp.nan.f32 + selp.f32 x2 -- a NaN-propagating max written out --
        #   and laneMask 1 is shfl.sync.bfly.b32 + a plain max.f32; c=31,
        #   membermask -1.  extent: 3 maxima + 2 shuffles + 5 combine
        #   instructions.  The NaN-propagating form is load-bearing: it is what
        #   keeps a NaN partial from being dropped by the -inf identity.
        max_valid = warp_reduce_max(last_live_split(lse_reg), LANES_COL)  # :790-797
        # instruction_selection: setp.neu.f32 x4 against 0fFF800000
        #   (.loc 1 792) + or.b32 x3 building the split coordinates 4/8/12
        #   (.loc 1 793) + selp.b32 x4 (.loc 1 242), then
        #   shfl.sync.bfly.b32 x2 with max.s32 x2 (.loc 1 795); extent: 2
        #   shuffles and 2 integer maxima.  The same butterfly shape as the
        #   float max, but a plain signed max -- no NaN handling on an index.
        lse_max_cur = 0.0 if lse_max == -inf else lse_max          # :799-800
        # instruction_selection: setp.eq.f32 against 0fFF800000 (.loc 1 800)
        #   then selp.f32 (.loc 1 242); extent: scalar.  What the select
        #   returns is the ALREADY-SCALED lse_max * LOG2_E -- NVVM hoists that
        #   multiply above the select (the export is NVVM output, so this is
        #   not a ptxas effect the port inherits for free), which is why the
        #   exp2 site below carries five multiplies and not four.
        acc = 0.0
        for s in range(SPLITS_PT):                                 # :806, unrolled
            scale = exp2(lse_reg[s] * LOG2_E - lse_max_cur * LOG2_E)
            # instruction_selection: mul.f32 by 0f3FB8AA3B x5 -- four per-split
            #   plus the one hoisted lse_max*LOG2_E -- and sub.f32 x4
            #   (.loc 1 806), then ex2.approx.ftz.f32 x4 (.loc 1 805); extent:
            #   4 per thread.  A sub, not an fma: the scaled maximum is a
            #   loop-invariant operand, so there is nothing to fuse.
            #   `fastmath=True` is what selects ex2.approx.ftz over a
            #   correctly-rounded expansion, and it is load-bearing for a
            #   bitwise match.
            acc = acc + scale                                      # :808
            # instruction_selection: add.f32 x4 (.loc 1 808), the first of
            #   which adds the literal 0f00000000 -- the zero initializer is
            #   NOT folded away; extent: 4 per thread
            lse_reg[s] = scale                                     # :809
        lse_sum = warp_reduce_sum(acc, LANES_COL)                  # :810
        # instruction_selection: add.f32 + shfl.sync.bfly.b32 x2 (.loc 1 810);
        #   extent: 2 shuffles, 2 adds
        if max_valid < 0 or lse_sum == 0.0 or lse_sum != lse_sum:  # :815
            final_lse[m], inv_sum = -inf, 0.0
            # instruction_selection: setp.lt.s32 (.loc 1 815) + setp.equ.f32
            #   against 0f00000000 + or.pred + mov.f32 x2 seeding the -inf and
            #   0.0 defaults + a predicated branch around the else arm
            #   (.loc 1 242); extent: scalar.  The source's three conditions
            #   become TWO compares: `sum == 0.0` and `sum != sum` fuse into the
            #   single unordered-equal setp.equ.f32, true for both zero and NaN.
            #   A port emitting a separate self-comparison would carry an
            #   instruction the reference does not.
        else:
            final_lse[m] = log2(lse_sum) * LN_2 + lse_max          # :818
            # instruction_selection: lg2.approx.ftz.f32 (.loc 1 818) then
            #   fma.rn.f32 with 0f3F317218 (.loc 1 818); extent: 2.  `log` under
            #   fastmath becomes lg2 folded with ln(2) into a single FMA that
            #   also adds lse_max -- one instruction, not two.
            inv_sum = recip(lse_sum)                               # :819
            # instruction_selection: rcp.rn.f32 (.loc 1 819); extent: one.
            #   NOT div.rn.f32: the reciprocal is computed once and multiplied
            #   into four scales below.
        for s in range(SPLITS_PT):
            lse_reg[s] = lse_reg[s] * inv_sum                      # :820
            # instruction_selection: mul.f32 x4 (.loc 1 820); extent: 4
    for s in range(SPLITS_PT):
        store_shared(sLSE[s2r_split(tidx, s), s2r_row(tidx)], lse_reg[s])   # :822
        # instruction_selection: st.shared.f32 x4 (.loc 1 822); extent: 4.
        #   sLSE is REUSED here: the staged log-sum-exps are overwritten in
        #   place by the normalized scales step 6 consumes.
    if s2r_split(tidx, 0) == 0 and s2r_row(tidx) < TILE_M:         # :869-873
        store_shared(sMaxValid[s2r_row(tidx)], max_valid)
        # instruction_selection: shl.b32 of `tidx >> 2` then st.shared.u32
        #   [base+8192] (.loc 1 872); extent: one, from the threads whose s2r
        #   split coordinate is 0 -- i.e. `tidx & 3 == 0`, which the export
        #   spells `setp.ne.s32 %p89,%r7,0` at :869.
        #   THIS IS A CROSS-PARTITION EDGE -- one of the two that `:907`
        #   publishes, alongside the sLSE scales.  The write indexes by the s2r
        #   row `tidx >> 2` (`%r41`); step 6 reads the same array by the
        #   O-partial row `(tidx >> 5) + 8m` (`%r14`, .loc 1 910).  The two maps
        #   differ, so the `:907` barrier is what makes the array readable --
        #   exactly as `:747` does for the staged sLSE.  The +8192
        #   displacement is the proof that sLSETemperature is allocated even
        #   with the temperature axis off.

    # -----------------------------------------------------------------------
    # Step 5: the authoritative LSE_out store (:889-902)
    # -----------------------------------------------------------------------
    if k_block == 0 and s2r_split(tidx, 0) == 0:                   # :891, :893
        # instruction_selection: the two conditions are OR-FUSED -- or.b32 of
        #   the split coordinate with ctaid.y, then one setp.ne.s32 against 0
        #   and one branch (.loc 1 242); extent: scalar.  The same trick as the
        #   degenerate guard at :815: two zero-tests become one.  A port
        #   emitting two compares and an and.pred would diverge.  Only the first
        #   k-block writes LSE, and only one thread per row within it.
        idx = m_block * TILE_M + s2r_row(tidx)
        if idx < max_idx:                                          # :896
            m_idx, head_idx = divmod_fast(idx, head_divmod)
            store_global(lse_out, (q_offset + m_idx) * stride_l_q + head_idx,
                         final_lse[0])
            # instruction_selection: st.global.f32 (.loc 1 898); extent: one
            #   scalar per owning thread
```

```python
    # -----------------------------------------------------------------------
    # Step 6: the accumulation loop over live splits (:907-953)
    # -----------------------------------------------------------------------
    barrier()                                                      # :907
    # instruction_selection: bar.sync 0 (.loc 1 907); extent: CTA-wide.  Makes
    #   both step-4 publications visible: the scales in sLSE and sMaxValid.
    thr_max_valid = load_shared(sMaxValid[o_row(tidx, 0)])          # :910
    # instruction_selection: ld.shared.u32 [base+8192] (.loc 1 910); extent: one
    for m in range(1, NUM_ROWS):                                    # :911, unrolled
        thr_max_valid = max_i32(thr_max_valid, load_shared(sMaxValid[o_row(tidx, m)]))
        # instruction_selection: ld.shared.u32 [base+8192+32*m] (.loc 1 912) +
        #   max.s32; extent: 7 loads and 7 maxima.  The 32-byte stride is this
        #   thread's row stride (8 rows apart) in the int32 array.
    fill(acc, 0.0)                                                  # :916
    # instruction_selection: mov.f32 x32 (.loc 1 242) -- one materializing the
    #   literal 0f00000000 and 31 copying it; extent: 8 rows x 4 values.  They
    #   sit in the loop PREHEADER, ahead of the zero-trip branch, so a thread
    #   whose rows are all short still carries zeroed accumulators into the
    #   epilogue instead of undefined registers.
    stage_load, stage_compute = STAGES - 1, 0
    for s in range(thr_max_valid + 1):                              # :922
        # instruction_selection: an unsigned zero-trip test in the preheader,
        #   setp.gt.u32 against 2147483646 -- which is how `thr_max_valid == -1`
        #   is detected after the +1 -- and a setp.ne.s32 latch against the
        #   bound; extent: a data-dependent trip count, ONE iteration per
        #   body.  The source asks for `unroll=4` at :922 and the export does
        #   not grant it: the body is a MULTI-BLOCK region -- 33 internal
        #   forward branches from the inlined loader rows, the accumulate
        #   guards and the prefetch test, closed by the single backward latch --
        #   holding exactly one iteration's work: 8 ld.shared.f32, 8
        #   cp.async.cg plus 8 st.shared.v4.f32 from the inlined
        #   `load_O_partial`, 8 ld.shared.v4.f32, 32 fma.rn.f32, one commit and
        #   one wait.  The port must not unroll by four to match a hint the
        #   reference itself did not realize.
        #   The bound is per THREAD: a thread whose eight rows are all short
        #   exits early while its neighbours keep going, which is the point of
        #   sMaxValid.
        for m in range(NUM_ROWS):
            scale[m] = load_shared(sLSE[s, o_row(tidx, m)])         # :926
            # instruction_selection: ld.shared.f32 x8 (.loc 1 926); extent: 8
            #   per iteration -- and, like the :757 read-back, these are NOT the
            #   scales this thread computed.  Step 4 wrote them on the s2r map,
            #   row `tidx >> 2` (.loc 1 822, base %r42 from `shr.u32 %r41,%r1,2`);
            #   step 6 reads them on the O-partial map, rows
            #   `(tidx >> 5) + 8m` (.loc 1 926, base from `shr.u32 %r351,%r350,5`).
            #   Thread 0 writes row 0 and reads rows 0, 8, ... 56.  This is the
            #   FOURTH cross-partition edge and the second one `:907` publishes
        if s + STAGES - 1 <= thr_max_valid:                         # :930
            load_O_partial(split=s + STAGES - 1, stage=stage_load)
        commit_group()                                              # :932
        # instruction_selection: cp.async.commit_group (.loc 1 932); extent: one
        #   per iteration -- issued even when no copy was made, so the group
        #   count the wait below reasons about stays exact.
        stage_load = 0 if stage_load == STAGES - 1 else stage_load + 1
        wait_group(STAGES - 1)                                      # :936
        # instruction_selection: cp.async.wait_group 1 (.loc 1 936); extent: one
        #   per iteration.  No barrier follows it: the stage data this wait
        #   covers is what the thread's own cp.async wrote, which is the claim
        #   the source's comment at :937 makes and the only one it makes.  The
        #   loop's other shared read, the sLSE scales at :926, does cross
        #   threads -- it is `:907`, before the loop, that makes it safe.
        for m in range(NUM_ROWS):
            part[m][0..3] = load_shared_v4(sO[stage_compute, o_row(tidx, m),
                                              o_col(tidx)])
            # instruction_selection: ld.shared.v4.f32 x8 (.loc 1 939); extent: 8
            #   vector loads per iteration, one per staged row -- but only seven
            #   are unconditional.  The compiler SINKS the m = 0 load into the
            #   guarded block below, so the port must place that one load inside
            #   the guard and the other seven ahead of it.
        stage_compute = 0 if stage_compute == STAGES - 1 else stage_compute + 1
        for m in range(NUM_ROWS):                                   # :943
            if hidx[m] >= 0 and scale[m] > 0.0:                     # :944
                # instruction_selection: setp.leu.f32 against 0f00000000
                #   (.loc 1 944, x8) then or.pred against the hoisted row
                #   predicate from :1144 (.loc 1 242, x8), driving a predicated
                #   skip; extent: 8 per iteration.  The `hidx >= 0` half is not
                #   a second compare -- it is the predicate step 2 already
                #   produced.  `leu` is unordered, so a NaN scale skips the row
                #   too, which is what makes a dead split's contribution
                #   unreachable.
                for v in range(NUM_VALS):
                    acc[m][v] = fma(part[m][v], scale[m], acc[m][v])
                    # instruction_selection: fma.rn.f32 x32 (.loc 1 946);
                    #   extent: 8 rows x 4 values per iteration.  fp32
                    #   accumulate in a fixed per-thread order -- no atomics, no
                    #   cross-thread combining, which is what makes the whole
                    #   kernel bitwise reproducible.
    wait_group(0)                                                   # :952
    # instruction_selection: cp.async.wait_group 0 (.loc 1 952); extent: one.
    #   Drains every outstanding group before the epilogue.  The source's own
    #   comment (:950-951) frames this as protecting the permutation buffer,
    #   but the export shows sO at [8448, 73984) and sO_perm at [73984, 92416),
    #   so there is no aliasing to protect against.  What the drain does retire
    #   is the loop's last prefetch: the body issues a commit every iteration
    #   whether or not it copied, so groups can still be outstanding when the
    #   loop exits.
    barrier()                                                       # :953
    # instruction_selection: bar.sync 0 (.loc 1 953); extent: CTA-wide

    # -----------------------------------------------------------------------
    # Step 7a: scatter the accumulator into sO_perm in REAL column order
    #          (:986-1088)
    # -----------------------------------------------------------------------
    for m in range(NUM_ROWS):                                       # :1011
        row_local = o_row(tidx, m)
        if hidx[m] >= 0:                                            # :1013
            # instruction_selection: no compare is emitted here.  The row
            #   predicate is computed ONCE per row in step 2 as
            #   setp.gt.s32 ..., -1 (.loc 1 1144, x8) and reused by all four
            #   consumers; this site spends only a predicated branch per row
            #   (`@%pNN bra`, x8) and emits nothing else.  The x8 not.pred that
            #   invert the row predicate are themselves hoisted into the prime
            #   `load_O_partial` block (.loc 1 242), and the inverted predicate
            #   is shared with the :944 or.pred and both loader call sites;
            #   extent: 8 branches per thread.  A port that re-compared per
            #   site would carry compares the reference does not.
            for v_pair in range(NUM_VALS // 2):                     # :1015
                fake_col = o_col(tidx, v_pair * 2)
                real_col = stg128_fake_to_real(fake_col)            # copy_utils.py:861
                # instruction_selection: an and.b32 / shr.u32 / and.b32 / or.b32
                #   chain and no multiply at all (.loc 4 863, 864, 866; 8 each,
                #   16 for the shr/and pair); extent: address arithmetic only,
                #   no memory op.  It is NOT hoisted: the chain is re-emitted
                #   inside each of the eight predicated row blocks, so a port
                #   that lifts it out of the loop undercounts the reference.
                #   The only multiply on this path is the sO_perm row address,
                #   mul.lo.s32 by 72 plus seven mad.lo.s32 (.loc 1 1037).
                #   This is the inverse of the permutation K1's STG.128
                #   epilogue applied.
                word = cvt_bf16x2(acc[m][2*v_pair + 1], acc[m][2*v_pair])
                # instruction_selection: cvt.rn.bf16x2.f32 x16 (.loc 1 242);
                #   extent: 8 rows x 2 pairs.  Two fp32 accumulators become one
                #   32-bit word in ONE instruction -- the reason 7a stores pairs
                #   rather than scalars.  Operand order is (hi, lo) = (v+1, v).
                store_shared(sO_perm_i32[row_local * ((K_BLOCK + 16) // 2)
                                         + real_col // 2], word)
                # instruction_selection: st.shared.u32 x16 (.loc 1 1049);
                #   extent: 16 per thread.  The row stride is
                #   (k_block + 16)/2 = 72 words, which is the padded stride --
                #   an unpadded 64 would put every thread of a quad in one bank.
    barrier()                                                       # :1088
    # instruction_selection: bar.sync 0 (.loc 1 1088); extent: CTA-wide.  This
    #   is the re-partition point: 7a wrote sO_perm with the PARTIAL copy
    #   partitioning (8 rows x 4 values), 7b reads it with the OUTPUT one
    #   (4 rows x 8 values), so every thread reads other threads' words.

    # -----------------------------------------------------------------------
    # Step 7b: sO_perm -> registers -> GMEM, 128 bits at a time (:1090-1125)
    # -----------------------------------------------------------------------
    for m in range(4):                    # store rows per thread, unrolled
        out_reg[m][0..7] = load_shared_v4(sO_perm[store_row(tidx, m), store_col(tidx)])
        # instruction_selection: ld.shared.v4.b32 x4 (.loc 1 1111); extent: 4
        #   per thread, 8 bf16 values each.  As in step 6, the m = 0 load is
        #   sunk into the store guard below and only three are unconditional.
    for m in range(4):                                              # :1114
        idx = m_block * TILE_M + store_row(tidx, m)
        if idx < max_idx:                                           # :1117
            # instruction_selection: setp.ge.s32 x4 + branch (.loc 1 1117),
            #   again the inverted sense; extent: 4 per thread.  The
            #   tail rows of the last m_block are dropped here, which is why a
            #   seqused launch leaves them at their prior contents.
            m_idx, head_idx = divmod_fast(idx, head_divmod)         # :1118
            store_global_v4(o_out, (q_offset + m_idx) * stride_o_q
                                   + head_idx * stride_o_h
                                   + k_block * K_BLOCK + store_col(tidx),
                            out_reg[m])
            # instruction_selection: st.global.v4.b32 x4 (.loc 1 1125); extent:
            #   4 per thread, 16 bytes each.  This is the store the whole
            #   fake-column detour exists to make contiguous.

# ===========================================================================
# load_O_partial: one pipeline stage (:1128-1161)
# ===========================================================================
def load_O_partial(split, stage):
    for m in range(NUM_ROWS):                                       # :1143, unrolled
        if hidx[m] >= 0:                                            # :1144
            for k in range(1):            # one k-tile at K_BLOCK = 128
                if split < split_count[m]:                          # :1154
                    # instruction_selection: three compare forms across the two
                    #   call sites, 16 in total (.loc 1 1154) -- the prime site
                    #   emits setp.lt.s32 against the immediate 1 for row 0 and
                    #   setp.gt.s32 against 0 for rows 1-7, because its split is
                    #   the constant 0; the loop site emits setp.ge.s32 for row
                    #   0 and setp.lt.s32 for rows 1-7.  The polarity flips with
                    #   which arm the branch is taken to; extent: 8 per call
                    #   site.
                    cp_async_cg(sO[stage, o_row(tidx, m), o_col(tidx)],
                                row_ptr[m] + split * stride_op_split + o_col(tidx),
                                16)
                    # instruction_selection:
                    #   cp.async.cg.shared.global [smem], [gmem], 16, 16
                    #   (.loc 1 1155); extent: 16 static = 8 rows x 2 call
                    #   sites (the prime and the loop).  CG bypasses L1: a
                    #   partial row is read exactly once.
                else:
                    store_shared_v4(sO[stage, o_row(tidx, m), o_col(tidx)], 0.0)
                    # instruction_selection: st.shared.v4.f32 with a splatted
                    #   zero register (.loc 1 1161); extent: 16 static.  The
                    #   zero fill is what makes a ragged split count safe: a
                    #   dead slot contributes 0 * scale, and the accumulate
                    #   guard drops it anyway.
```

## Control-flow map

| Region | Source | Predicate | Who executes |
| --- | --- | --- | --- |
| varlen row bound | `:547` | `m_block * 64 < seqlen * head_q` | CTA-uniform; precedes every barrier, so no CTA reaches a partial barrier |
| K1 dependency | `:550` | `use_pdl` (compile time) | CTA-wide, one `griddepcontrol.wait` |
| LSE row valid | `:639` | `idx < max_idx` | per staged row |
| LSE split live | `:652` | `si < topk and si < row_count` | per staged split; the else stores `-inf` |
| partial row valid | `:694` | `idx >= max_idx` | per staged row; a real predicated branch each, with the `-1/0/0/0` defaults seeded by `mov.u64`/`mov.pred`/`mov.b32` before the branch |
| accumulate loop | `:922` | `s <= thr_max_valid` | per thread, data-dependent trip count; the source's `unroll=4` is not realized |
| stage prefetch | `:930` | `s + stages - 1 <= thr_max_valid` | per iteration |
| accumulate guard | `:944` | `hidx[m] >= 0 and scale[m] > 0` | per row per iteration; `setp.leu` so NaN also skips |
| copy vs zero fill | `:1154` | `split < split_count[m]` | per row per staged split |
| LSE writeback | `:880-894` | `k_block == 0` and split coord `== 0` | one thread per row of the first k-block |
| store row valid | `:1117` | `idx < max_idx` | per store row; the source of unwritten tail rows under `seqused` |

## Instruction-selection summary

| decision | selects | why |
| --- | --- | --- |
| LSE staged with a 4-byte copy, partials with 16 | `cp.async.ca` vs `cp.async.cg` | not a caching preference but the ISA: `cp.async.cg` is defined only for a 16-byte cp-size, so a 4-byte LSE element has to use `ca`. The partial rows are 16 bytes and take `cg`, which is the cache-bypassing choice a row read exactly once wants |
| `stages = 2` | two `sO` slices, `wait_group 1` | 92416 B total SMEM at 2 stages against roughly 158 KB at 4; the host's own note records 1 -> 2 blocks/SM and DRAM throughput 76% -> 89% (`:1374-1379`). That note's "168 KB -> 103 KB" predates this specialization -- the export-derived 92416 B is what the port must budget |
| commit group per stage, never per copy | `wait_group(stages-1)` means "oldest stage landed" | a single group for all stages would collapse the pipeline into a barrier |
| no barrier in the accumulate loop | nothing | the loop's two shared reads are covered without one: `sO` is what the thread's own `cp.async` wrote (which is what `:937` is scoped to), and the `sLSE` scales, though written by other threads on the s2r map, were published by `:907` before the loop began. So no edge *arises inside* the loop, and a barrier there would only add a CTA-wide wait to every iteration of a loop whose trip count already varies per thread |
| `fastmath=True` on exp and log | `ex2.approx.ftz.f32`, `lg2.approx.ftz.f32` | correctly-rounded forms would be multi-instruction expansions and would not match K1's own softmax scaling |
| one reciprocal, four multiplies | `rcp.rn.f32` + `mul.f32 x4` | four `div.rn.f32` would be four multi-instruction sequences |
| `log(sum) + lse_max` written as one expression | `lg2.approx` + a single `fma.rn.f32` with `ln 2` | the add folds into the FMA; emitting `mul` then `add` would be one instruction more and a different rounding |
| reductions over exactly four lanes | `shfl.sync.bfly.b32` laneMask 2 then 1 | `smem_threads_per_col_lse = 4`; a full-warp reduction would be five shuffles and would mix unrelated rows |
| pairs converted before the SMEM store | `cvt.rn.bf16x2.f32` + `st.shared.u32` | two accumulators per instruction and per store; the scalar path costs twice as many of both |
| `sO_perm` row stride padded by 16 | conflict-free `st.shared.u32` | at stride 64 the quad writing one row lands in one bank |
| separate copy partitionings for load and store | `ld.shared.v4.b32` reading what `st.shared.u32` wrote | 128 bits is 4 fp32 partials but 8 bf16 outputs; the `bar.sync` at `:1088` legalizes the re-partition |
| flat row index divided by `num_head` | FastDivmod: `mul.hi.u32` + 2 `shr` + `add` + `mul.lo` + `sub` | the divisor is a launch invariant, so the host precomputes the magic; the port passes the same four scalars |
| `head_idx // qhead_per_kvhead` left alone | `div.s32` | `qhead_per_kvhead` is a runtime `Int32` with no host-side magic |
