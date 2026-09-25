<!--
This sketch is a design artifact, not source. It is written once, reviewed
once, and then frozen: the implementation follows it, and where the two
disagree the TIRx module linked below is what runs.
-->

# msa_sparse_atten_fwd_nvfp4_kv_sm100: coarse WASP pipeline sketch

This non-executable design sketch describes the storage layout, warp roles,
pipelines, control flow, and PTX-level operations of
[`tirx_kernels/msa/sparse_atten_fwd_nvfp4_kv.py`](../../../tirx_kernels/msa/sparse_atten_fwd_nvfp4_kv.py).
That TIRx module is the authoritative implementation.

The port covers **one kernel**: `SparseAttentionForwardNvfp4KvSm100.kernel`
(`atten_fwd_nvfp4_kv.py:603-1191`) with its trace-time launcher `__call__`
(`:295-597`) and every warp-role helper it dispatches to. It computes the same
attention as this package's already-ported BF16/FP8 forward, over K and V held
as packed NVFP4, and writes the same split partials that the combine kernel (a
separate class, separate compile key, **out of scope**) reduces. It consumes the
work list and the packed `q_idx | slot<<24` metadata produced by the two
preparation kernels already ported into this package.

**No tensor core in this kernel ever sees FP4.** TMA stages packed bytes into
shared memory; the two softmax warpgroups dequantize them into ordinary BF16 or
FP8 tiles before the MMA reads them. The dequant target is Q's dtype, so a
single `mma_kind` serves both GEMMs -- unlike the sibling, which can mix a
`f8f6f4` QK with an `f16` PV.

This kernel resembles that already-ported sibling closely enough that carrying a
fact across from it is the main hazard in reading this sketch. Three places
where the two genuinely differ, and where the sibling's answer is wrong here:
the pipeline mbarriers are initialised by **warp 0** (the sibling's port moved
them to warp 1 as a performance change, which is not what this source does); the
causal-diagonal search is a **fixed 32-trip rolled loop**, not a `while`; and
only the gather4 Q path sets an **L2 eviction policy**, while the KV and TMA-Q
loads pass a zero policy.

## Scope and instantiations

The upstream compile key is a 15-tuple (`interface.py:1910-1926`). Fixed for
every specialization in this port:

| axis | value | why it is fixed |
| --- | --- | --- |
| `head_dim` | 128 | `__init__` raises otherwise (`:78-81`) |
| `n_block_size` | 128 | the KV block width the schedule is built around |
| `m_block_size` | 128 | 128 packed Q heads per MMA tile |
| `use_prepare_scheduler` | `True` | `__init__` raises otherwise (`:100-102`) |
| `fp8_pair_dequant` | `True` | the shipped default (`interface.py:1860`); the port pins it rather than reading the environment |
| `has_v_global_scale` | `False` | **unreachable**: the interface pins it (`interface.py:1862-1866`) and applies V's tensor scale once in the combine kernel |

In scope, i.e. the specializations this module compiles:

| axis | values | what changes in device code |
| --- | --- | --- |
| Q dtype | `bf16`, `fp8_e4m3` | the largest axis. It fixes the **dequant target program** (a BF16 conversion chain vs an FP8 pair chain), the single `mma_kind` (`"f16"` vs `"f8f6f4"`, `:347`), `tmem_s_to_p_offset` (64 vs 96, `:352-354`), the **whole gather4 Q body** (`:2014-2056`: FP8 issues one copy per gather with no prefetch, BF16 issues two half-row copies plus a prefetch), and **where the K tensor scale is applied**. |
| `qheadperkv` | 1, 2, 4 / 8, 16 | selects the whole Q-load warpgroup program: `use_q_gather4` (`:84`) picks a raw gather4 descriptor path for 1/2/4 and a CUTE-managed TMA path for 8/16. Also sets `q_tokens_per_group = 128 // qheadperkv` (`:112`) and `tokens_per_gather4 = 4 // qheadperkv` (`:90`). |
| `partial_dtype` | fp32, bf16, **fp16**, fp8 e4m3 | three store WIDTHS -- 4-lane, 8-lane and 16-lane 128-bit stores with three different swizzle-inverse column remaps (`:3041-3269`) -- but FOUR programs: bf16 and fp16 share the 8-lane width and the remap yet take different converters, `cvt.rn.bf16.f32` against `cvt.rn.f16.f32`, dispatched by `const_expr` at `:2868-2884`. Saying fp16 "shares the bf16 path" is true of the width and false of the arithmetic. |
| temperature LSE | on / off | `mLSE_temperature_partial is not None` adds one scaled row-sum reduction, one `sScaleTemperature` publish and one extra LSE store |
| `has_k_global_scale` | `True` / `False` | not a scalar difference but two different device paths, one per Q dtype: BF16 Q folds the tensor scale into the dequantized values, FP8 Q multiplies the FP32 S accumulator instead |
| `paged_kv` | `False` / `True` | `page_table is not None` at this kernel's host entry (`interface.py:1867`; `page_size = int(k.shape[2])` at `:1871`, compile key `:1910-1926`). `page_size == blk_kv` (`:103-111`) makes a CTA's KV block exactly one page, so paging swaps one TMA coordinate rather than gathering: K/V become `[num_pages, head_kv, page_size, 64]`, the descriptors go rank 4, `k_batch_offset` becomes 0, and thread 0 resolves the page into `sPagedKvIdx` (`:864-867`). The one NVFP4-specific part is the block-scale row, `_paged_kv_scale_row` (`:1328-1336`), threaded to four dequant call sites through the two dequant wrappers. |
| `has_seqused_k` | `False` / `True` | paged-only at the host entry (`interface.py:367-369`); returns `mSeqUsedK[batch]` ahead of the paged-capacity and `cu_seqlens_k` fallbacks (`:205-216`). Under causal masking it can drive `causal_q_offset` negative, which leaves the leading `seqlen_q - seqlen_k` queries with no legal key and a legitimately `-inf` LSE. |
| `causal` | `False` / `True` | `num_regs_softmax` 176 -> 192, `num_regs_store` 112 -> 80, `num_regs_other` derived as `512 - softmax*2 - store` and landing on 48 either way, `ex2_emu_freq` 16 -> 0 (`:177-185`); the diagonal binary search is skipped and `causal_q_offset` is pinned to 0. |

Out of scope, with the predicate that excludes each:

- **`fp8_pair_dequant=False`** -- environment-gated off the default
  (`interface.py:1860`); the port pins it True rather than reading the
  environment, and under the pinned toolchain its body is the same instruction
  family as the pair path.
- **`page_size != blk_kv`** -- rejected at `:103-111`; the equality is what makes
  a CTA's KV block exactly one page.
- **the BF16/FP8-KV sibling** `SparseAttentionForwardSm100` (`atten_fwd.py`) --
  a separate class, already ported.
- **the combine kernel** `fwd/combine.py` -- a downstream consumer, and the
  only place V's tensor scale is applied.
- Tile (`Tx`) primitives are out of scope everywhere.

## The line-info export this sketch is annotated from

Every `instruction_selection` annotation below is read out of a line-info PTX
export, not out of the source text. Eight exports are preserved under
`.porting/sparse_atten_fwd_nvfp4_kv/ptx_lineinfo/<name>/`;
`.porting/sparse_atten_fwd_nvfp4_kv/export_findings.md` records what they
settle. Six cover the flat compile keys; the seventh and eighth are the two
paged builds, `paged_bf16q_tmaq_qh16_fp32` (BF16-Q) and
`paged_fp8q_gather4_qh4_bf16partial` (FP8-Q). A paged annotation cites whichever
matches its Q dtype -- the two dequant programs have different source sites, so
the BF16-Q scalar arm's `.loc 1 1450` / `:1625` occur only in the seventh and
the FP8-Q pair arm's `.loc 1 1384` / `:1558` only in the eighth.

Three newly in-scope axes have no export of THIS kernel, each for a stated
reason rather than by omission:

- `causal=False` -- the axis is computed by character-identical code in the
  BF16 sibling (`atten_fwd.py:173-181` against `atten_fwd_nvfp4_kv.py:177-185`,
  both deriving `num_regs_other = 48`), and the sibling's causal/non-causal
  export pair measures it: `setmaxnreg.inc` 176 -> 192, `dec` 112 -> 80,
  `dec 48` x3 in both, `ex2.approx` 224 -> 256, static instructions
  5571 -> 5082. That identity settles the CONSTANTS; the control-flow
  consequences annotated here -- the deleted diagonal search, the deleted
  `mCuSeqlensQ[batch+1]` load, the compiled-out causal-mask arm, the
  zero-frequency exp2 arm -- come from that same sibling pair.
- `partial_dtype=fp16` -- both classes call the same
  `common/copy_utils.stg_128_f16_cs`, exported in the sibling as
  `bf16_tmaq_qh16_fp16partial`.
- `has_seqused_k` -- its entire device delta is one `ld.global.u32` of
  `mSeqUsedK[batch]` replacing the paged-capacity multiply at `:212-215`,
  plus the negative-`causal_q_offset` consequence recorded in the scope table.
  Both are visible in the sibling's `paged_seqused_bf16_tmaq_qh16_fp32`.

There are FOUR `.file` numbering families across these eight exports, measured
from their tails:

| build | 3 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- |
| BF16-Q flat | `quack/copy_utils.py` | `utils.py` | `mask.py` | `softmax.py` |
| FP8-Q flat | `quack/copy_utils.py` | `mask.py` | `utils.py` | `softmax.py` |
| BF16-Q paged | `paged_kv.py` | `blackwell_helpers.py` | `utils.py` | `mask.py` |
| FP8-Q paged | `paged_kv.py` | `mask.py` | `utils.py` | `softmax.py` |

The paged FP8-Q build is NOT the paged BF16-Q shift applied to the FP8-Q order:
`paged_kv.py` enters as file 3, but `quack/copy_utils.py` drops out of that
build entirely, and the two exactly cancel -- so its numbering coincides with
flat FP8-Q rather than shifting by one. Applying the BF16-Q paged shift to it
would read its `.loc 6` region as `blackwell_helpers.py` when it is `mask.py`. Unqualified counts below are from `bf16q_tmaq_qh16_fp32` (5922
instructions, 161 predicated). Counting convention: instruction lines minus
predicated lines.

Reproduce any export with:

```bash
mkdir -p .porting/sparse_atten_fwd_nvfp4_kv/ptx_lineinfo/<name>
MM_SPARSE_ATTN_AOT_DISABLE=1 CUTE_DSL_NO_CACHE=1 CUTE_DSL_KEEP=ptx \
CUTE_DSL_LINEINFO=1 CUTE_DSL_DUMP_DIR=.porting/sparse_atten_fwd_nvfp4_kv/ptx_lineinfo/<name> \
python .porting/sparse_atten_fwd_nvfp4_kv/export_driver.py <name>
```

Two properties of these exports drive the whole dequant region and must not be
re-derived from the source text:

1. **The pinned toolchain reports CUDA 12.9**, baked into its native library
   rather than taken from the host (which is 13.2). Both dequant helpers gate
   their fast paths on `>= 13.2` (`utils.py:652-693`, `:697-771`), so the pin
   takes the **fallbacks**, and the export confirms it: `mul.e4m3x4.e2m1x4`
   count is **0 in all eight exports**, both paged ones included. The port transcribes the fallback chains.
2. **Re-read the `.file` id table for every export.** The ids are assigned per
   export and they are *not* stable even across this kernel's own
   specializations: see the four-family table above. In particular the two
   paged builds do NOT share a numbering -- `paged_kv.py` enters as file 3 in
   both, but `quack/copy_utils.py` drops out of the FP8-Q paged build and
   cancels the insertion. The table sits at the
   very end of the PTX and its lines are tab-indented, so a `^\.file` grep finds
   nothing -- read the tail. Assuming one export's numbering for another
   silently reattributes a whole region.

The dequant helpers are `llvm.inline_asm` blocks, so their instructions inherit
the caller's `.loc` and are attributed to the kernel entry line rather than to
`utils.py`. Their identity is settled by opcode counts, which are unambiguous:
see the instruction-selection summary.

## Pipeline at a glance

16 warps, 512 threads, one CTA per work item. Role selection is a flat sequence
of independent `if` blocks -- **not** an `if/elif` chain -- each ANDed with
`cta_valid_work`. A warp falls through the blocks it does not match and returns.

| Warps | Role-local tile program | Main publication / reuse edges |
| --- | --- | --- |
| 0..3 | softmax warpgroup 0 **with the epilogue fused in**; it first dequantizes the whole K tile from `sKFp4` into `sK`. Takes the **even** Q groups. | waits `mbar_k_tma`; publishes `mbar_k` and `sK`; consumes `mbar_s`; publishes `mbar_p` + `mbar_p_lastsplit` and `sScale`; consumes `mbar_o`; acquires/releases `mbar_sm_stats` |
| 4..7 | softmax warpgroup 1, same body, **odd** Q groups; dequantizes V from `sVFp4` into `sV` first | waits `mbar_v_tma`; publishes `mbar_v` and `sV`; same pipeline edges, other slot parity |
| 8..11 | Q-load warpgroup, running one of **two structurally different programs** (see `q_load_warpgroup()`): the gather4 program publishes `sQIdxMeta` only, from all four warps, and has three `bar_load_wg` sites; the TMA program publishes `sQIdxMeta` **and** `sQLoadMIdx` from one thread per token and has two | produces `mbar_q`; publishes `sQIdxMeta` (read later by both softmax WGs *and* the epilogue) on both programs; `sQLoadMIdx` exists **only on the TMA program**, where it is the TMA's row addressing -- the gather4 program takes its four row coordinates by decoding `sQIdxMeta` |
| 12 | the single MMA-issue warp; also the TMEM allocator warp | waits `mbar_k`/`mbar_v`, consumes `mbar_q`, produces `mbar_s` and `mbar_o`, consumes `mbar_p`/`mbar_p_lastsplit` |
| 13 | K load: one TMA of the packed FP4 tile for this CTA's single KV block | produces `mbar_k_tma` |
| 14 | V load: one TMA of the packed FP4 tile for the same block | produces `mbar_v_tma` |
| 15 | idle; executes only `setmaxnreg.dec 48` and falls out | none |

Dispatch order in the source is warp 15, then the Q-load warpgroup, then the
KV-load warps, then the MMA warp, then softmax WG0 and WG1 (`:943-1191`).

**The dequant handoff is unconditional here.** The sibling gates its
`mbar_k_tma`/`mbar_v_tma` and its `KvDequantK`/`KvDequantV` named barriers on an
fp8-staging predicate; this kernel always stages, so both barrier pairs and both
named barriers always exist (`:511-512`, `:923-934`). There is no
"TMA lands straight in MMA layout" branch at all -- the packed tile is never a
legal MMA operand.

**KV is not pipelined.** One CTA owns exactly one KV block, so the K/V barriers
are single-shot, with `expect_tx` set once by thread 0 in the prologue, and the
load warps issue one TMA each and retire. All pipelining runs along the
**Q-group axis**: `num_q_groups = ceil(count_raw / q_tokens_per_group)`, with
`q_stage = 2`, `s_stage = 2`, `o_stage = 2` and a 16-deep `sQIdxMeta` ring.

**There is no online rescale.** Because a Q group sees exactly one KV block,
every softmax step is the first-and-only step: one row max, one exponentiation,
one row sum, no correction loop, no running accumulator rescale.

**The epilogue is fused into the softmax warpgroups**, and the export shows the
whole softmax+epilogue body is **emitted twice**, once per warpgroup, not shared
behind a runtime `stage`.

## Primitive vocabulary

Structural operations do not compute values:

```python
tile(...)        # declare storage, dtype, logical shape, placement
view(...)        # change logical indexing without moving values
alias(...)       # declare exact storage aliasing and non-overlap lifetime
slice(...)       # select a logical interval
reg_tile(...)    # declare a role-local register tile
tensormap(...)   # declare a TMA descriptor and its encode fields
```

Copies always state their storage direction:

```python
copy_g2s(src, dst, bar=None, hint=None)        # global -> shared (TMA)
copy_g2s_gather4(dst, dst_byte_off, desc, col, r0, r1, r2, r3, bar, hint)
prefetch_g2s_gather4(desc, col, r0, r1, r2, r3, hint)
copy_s2r(src, dst)                             # shared -> register
copy_r2s(src, dst)                             # register -> shared
copy_t2r(src, dst)                             # tensor memory -> register
copy_r2t(src, dst)                             # register -> tensor memory
copy_r2g(ptr, values, cache=None)              # register -> global
load_shared(smem, idx) -> reg
store_shared(smem, idx, reg)
load_global(buf, idx) -> reg
store_global(buf, idx, reg)
prefetch_descriptor(desc)
```

The computational vocabulary:

```python
fill(dst, value)
cast(dst, src)                       # includes fp4 -> f16, f16 -> bf16, f32 -> narrow
add(dst, lhs, rhs, lanes=1)          # lanes=2 is one packed two-lane op
sub(dst, lhs, rhs, lanes=1)
mul(dst, lhs, rhs, lanes=1)
fma(dst, a, b, c, lanes=1)
max(dst, lhs, rhs)                   # also the SM100 three-source form
fmax(a, b, c) -> reg                 # the same instruction used as an expression
exp2(dst, src)                       # MUFU
exp2_poly_pair(dst0, dst1)           # ex2_emulation_2: one degree-3 FFMA
                                     # polynomial over a PAIR of values
log2(dst, src)
rcp(dst, src)
select(dst, predicate, a, b)
clamp(x, lo, hi) -> reg              # min then max
shift_right_u32(dst, src)            # shr.u32, which saturates shifts >= 32
permute_bytes(dst, a, b, selector)   # prmt
gemm(dst, lhs, rhs, accumulate=False)
```

Small index decodes, written as expressions rather than named ops so that no
primitive hides a computation:

```python
word & 0xFFFFFF                      # q_idx out of a packed qsplit word
(word >> 24) & 0xFF                  # split index out of the same word
decode_m_idx(word)                   # ((q_batch_offset + (word & 0xFFFFFF))
                                     #  * num_heads_kv + head_kv_idx) * qheadperkv
dst_view(sDst, row, col, v, is_v)    # the shared destination sub-tile a dequant
                                     # task writes.  FOUR forms, one per
                                     # (K or V) x (FP8 pair or BF16):
                                     #   K FP8 : sK[(row,None), 0, col, 0]
                                     #           col = pair_col               :1419
                                     #   K BF16: sK[(row,None), 0,
                                     #              (col % 4, col // 4), 0]
                                     #           col = scale_col         :1491-1497
                                     #   V     : sV[(None, row % G), 0,
                                     #              row // G, 0]  then
                                     #           local_tile(..., (n,), (col,))
                                     #           G = 32 FP8 :1593-1594
                                     #           G = 16 BF16 :1666-1669
                                     # The BF16 K form indexes a HIERARCHICAL
                                     # mode: the eight scale columns land as
                                     # (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),
                                     # (2,1),(3,1), not 0..7.  Substituting a
                                     # plain scale_col there permutes the K
                                     # tile -- QK still produces plausible
                                     # numbers, so only the bitwise check
                                     # catches it.  The store instruction is
                                     # the same in all four; only the address
                                     # differs, which is why the export cannot
                                     # tell the arms apart and the source has
                                     # to be the authority here.
frag_coord(r, cN)                    # the (row, col) an epilogue fragment slot
                                     # maps to, from the TMEM load's own layout
```

There is deliberately no `tree_add`, `exp2_population`, `broadcast`, or any
other primitive that would fold a whole reduction or population into one name --
those loops are written out where they occur, so their instruction counts and
their emulation predicates stay visible.

Schedule operations: `pipe`, `init_pipe`, `acquire`, `wait`, `commit`,
`release`, `expect_tx`, `fence`, `barrier`, `named_barrier_arrive`,
`named_barrier_arrive_and_wait`, `elect`, `set_register_budget`,
`allocate_tmem`, `relinquish_tmem_alloc_permit`, `free_tmem`, `wait_tmem_ld`,
`griddepcontrol_launch_dependents`, and cursor (`slot`, `phase`) updates.

`add(..., lanes=2)` is one packed two-lane operation with two ordered results,
not shorthand for two scalar adds. There are deliberately no primitives named
`attention`, `softmax`, `mask`, `online_update`, `TMA`, `TCGEN05`,
`dequantize`, `unpack_nvfp4`, or `epilogue`.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================
HEAD_DIM        = 128
M_BLOCK         = 128          # packed Q heads per MMA tile
N_BLOCK         = 128          # KV block width
PACKED_HEAD_DIM = 64           # HEAD_DIM // 2: two E2M1 values per byte
SCALE_BLOCK     = 16           # head-dim elements per E4M3 scale byte
SCALE_COLS      = HEAD_DIM // SCALE_BLOCK          # 8
Q_STAGE = S_STAGE = O_STAGE = 2
QIDX_META_STAGES = 16
SPLIT_P_ARRIVE   = 96          # of 128 P columns published early (:169-172)
EX2_EMU_FREQ     = 16          # causal only; 0 when causal=False (:184)
EX2_EMU_RES      = 4           # softmax.py default, never overridden
EX2_EMU_START_FRG = 1          # class attr (:185)
NUM_Q_LOAD_WARPS = 4          # warps 8..11, the Q producer
NUM_DEQUANT_WARPS = 4         # `warps_per_group`: the SOFTMAX warpgroup's
                              # own four warps, which run the K/V dequant.
                              # Same number, different constant -- do not
                              # substitute one for the other.
K_TILE           = 64          # class attr (:61); BF16 gather4 splits on it

qheadperkv        # 1,2,4 -> gather4 Q program; 8,16 -> TMA Q program
q_tokens_per_group = M_BLOCK // qheadperkv
tokens_per_gather4 = 4 // qheadperkv                # gather4 program only
gathers_per_warp   = M_BLOCK // (NUM_Q_LOAD_WARPS * 4)          # 8 (:1894)
q_dtype           # "bf16" or "fp8e4m3"
mma_dtype = q_dtype                                 # K/V dequant target (:333-334)
mma_kind  = "f8f6f4" if q_dtype is fp8 else "f16"   # ONE kind, both GEMMs (:347)
tmem_s_to_p = N_BLOCK - N_BLOCK * width(q_dtype) // 32    # 64 bf16 / 96 fp8 (:352-354)
tokens_per_warp   = ceil(q_tokens_per_group / NUM_Q_LOAD_WARPS)  # TMA program
subtiles_per_token = 1 if q_dtype is fp8 else 2      # BF16 Q splits on K_TILE
partial_dtype     # fp32 | bf16 | fp16 | fp8e4m3
has_temperature   # LSE_temperature_partial is not None
has_k_global      # k_global_scale is not None
# instruction_selection: none; extent: compile-time constants only.
#   Unlike the sibling there is no separate qk/pv dtype: the dequant target is
#   Q's dtype, so one `mma_kind` covers both GEMMs.

# Runtime ABI, in the source's parameter order (:296-324).
mK, mV                    # global uint8. FLAT: (total_k, PACKED_HEAD_DIM,
                          # head_kv), host permute [0,2,1] / [1,0,2] (:365-372).
                          # PAGED: (num_pages, head_kv, page_size,
                          # PACKED_HEAD_DIM), host permute [2,3,1,0] and the
                          # extra [1,0,2,3] for V (:373-379) -> rank-4
                          # descriptors. V is MN-major either way.
mKScale, mVScale          # global uint8, E4M3 bytes in cuBLAS 128x4 tiled order
mKGlobalScale             # global f32[1] or absent
mVGlobalScale             # always absent (interface pins has_v_global_scale off)
mK2qIndices               # global i32 (head_kv, nnz)      CSR payload, q-ascending
mK2qQSplitIndices         # global i32 (head_kv, nnz)      q_idx | slot<<24
mK2qCounts                # global i32 (head_kv, rows+1)   CSR row_ptr
mSchedulerMetadata        # global i32 (work_capacity, 6)
mWorkCount                # global i32 (1,)
mO_partial                # global partial_dtype (topk*total_q*head_q, HEAD_DIM)
mLSE_partial              # global f32 (topk, total_q, head_q)
mLSE_temperature_partial  # optional, same shape
mQ_flat                   # global q_dtype (total_q*head_q, HEAD_DIM)
mQ_gather4_desc           # raw uint8[128] TensorMap, gather4 program only
mPageTable                # optional i32 (batch, pages_per_seq); PAGED_KV only.
                          # pages_per_seq = mPageTable.shape[1] (:211 analogue)
mSeqUsedK                 # optional i32 (batch,); HAS_SEQUSED_K only, and the
                          # host accepts it only alongside mPageTable
mCuSeqlensQ, mCuSeqlensK  # global i32 (batch+1,)
softmax_scale_log2        # f32, host-computed: softmax_scale * log2(e)  (:487)
lse_temperature_scale_log2 # f32, host-computed, chained off the above  (:488)
lse_temperature_scale     # f32
num_kv_blocks, num_heads_kv, seq_len_q, work_capacity   # i32
# instruction_selection: none; extent: ABI only.
#   All three scale scalars arrive as `.param .f32` -- the export shows params
#   15-17 and no device-side recomputation.  Recomputing either log2 product in
#   the kernel would be an ABI change, and `lse_temperature_scale_log2` is
#   chained off the already-folded `softmax_scale_log2`, so the two cannot be
#   derived independently from the raw scales.

launch(grid=(work_capacity,), block=(512, 1, 1),
       smem=max(sizeof(SharedStorage), 49152), min_blocks_per_mp=1)
# instruction_selection: none; extent: launch geometry (:590-596), confirmed by
#   the export.  The grid is sized by schedule CAPACITY, so tail CTAs retire.

# ===========================================================================
# Storage
# ===========================================================================
# Packed FP4 staging tiles -- the TMA destinations. Plain (no MMA swizzle):
# nothing reads them as an MMA operand, only the dequant loop reads them back.
sKFp4 = tile("shared", "uint8", (N_BLOCK, PACKED_HEAD_DIM), align=1024)   # 8 KiB
sVFp4 = tile("shared", "uint8", (PACKED_HEAD_DIM, N_BLOCK), align=1024)   # 8 KiB

# The MMA operand tiles the dequant loop fills. 128-byte swizzled.
sK = tile("shared", mma_dtype, (N_BLOCK, HEAD_DIM), align=1024)
sV = tile("shared", mma_dtype, (HEAD_DIM, N_BLOCK), align=1024)   # MN-major
sQ = tile("shared", mma_dtype, (Q_STAGE, M_BLOCK, HEAD_DIM), align=1024)
sQ_load = alias(sQ)        # same bytes, load-program indexing

# Softmax/epilogue metadata.
sScale       = tile("shared", "f32", (O_STAGE, 2 * M_BLOCK))   # [0:M) sum, [M:2M) max
sScaleTemp   = tile("shared", "f32", (O_STAGE, M_BLOCK))       # temperature row_sum
sSplitIdx    = tile("shared", "i32", (O_STAGE, q_tokens_per_group))
sQIdx        = tile("shared", "i32", (O_STAGE, q_tokens_per_group))
sQIdxMeta    = tile("shared", "i32", (QIDX_META_STAGES, q_tokens_per_group))
sQLoadMIdx   = tile("shared", "i32", (Q_STAGE, q_tokens_per_group))
#   Declared UNCONDITIONALLY (:553-554, :722-723) and therefore shifts every
#   later SMEM offset in both programs -- the qh4 gather4 export still reserves
#   its 2*32*4 bytes (sPagedKvIdx at 3888, sQIdxMeta at 4148 = 3892 + 256).
#   Only the TMA program writes or reads it.
sRowMeta     = tile("shared", "i32", (8,))   # batch, kv_block, row_start, count,
                                             # valid_cols, q_off, k_off, causal_off
sDiagQCount  = tile("shared", "i32", (1,))
sPagedKvIdx  = tile("shared", "i32", (1,))   # written by thread 0 from the page
                                             # table under the same publish
                                             # fence as sRowMeta; read warp-wide
                                             # once per KV-load warp, and ONCE
                                             # PER ITERATION in each of the two
                                             # dequant task loops.  Allocated
                                             # unconditionally, so flat and
                                             # paged share every later offset
tmem_addr    = tile("shared", "u32", (1,))
tmem_dealloc_mbar = mbar(1)                  # 8 B, likewise unconditional

# TMEM: 512 columns total.
S0 = tmem_tile("f32", (M_BLOCK, N_BLOCK), col=0)
S1 = tmem_tile("f32", (M_BLOCK, N_BLOCK), col=128)
O0 = tmem_tile("f32", (M_BLOCK, HEAD_DIM), col=256)
O1 = tmem_tile("f32", (M_BLOCK, HEAD_DIM), col=384)
P  = view(S_stage_base + tmem_s_to_p)      # P overlays S starting at column
                                           # tmem_s_to_p: 64 for BF16 (the
                                           # upper half) but 96 for FP8, whose
                                           # narrower elements need only a
                                           # quarter of the S tile

# mbarriers. Note both TMA pairs are UNCONDITIONAL here (:511-512); the sibling
# gates its equivalents on an fp8-staging predicate.
mbar_k_tma, mbar_v_tma = mbar(2), mbar(2)   # packed FP4 bytes landed
mbar_k,     mbar_v     = mbar(2), mbar(2)   # dequantized tile ready for the MMA
mbar_q            = pipe(Q_STAGE)           # full/empty pairs
mbar_s            = pipe(S_STAGE)
mbar_p            = pipe(S_STAGE)
mbar_p_lastsplit  = pipe(S_STAGE)
mbar_o            = pipe(O_STAGE)
mbar_sm_stats     = pipe(O_STAGE)           # ONLY the empty pair is used: the
                                            # producer acquire (:2595) waits it
                                            # and the epilogue's consumer_release
                                            # (:3304) arrives it, both at byte
                                            # offset 240.  The full pair (offset
                                            # 224) is initialized like every
                                            # other barrier but is never arrived
                                            # and never waited -- no other
                                            # instruction in the kernel names
                                            # that address.

# Named barriers, renumbered from 8 under the CUTLASS user-id convention. The
# source's own ids collide (`StoreEpilogue + stage` reaches KvLoad's id), which
# is benign only by timing; the port renumbers and records the deviation.
bar_tmem_alloc   = named_barrier(id=8,  threads=32 * (4 * 2 + 1))   # 288
bar_load_wg      = named_barrier(id=9,  threads=128)
bar_kv_load      = named_barrier(id=10, threads=64)
bar_dequant_k    = named_barrier(id=11, threads=128)
bar_dequant_v    = named_barrier(id=12, threads=128)
bar_epilogue     = named_barrier(id=13, threads=128)   # +stage -> 13, 14
# There is NO softmax-stats named barrier.  The source declares
# `SoftmaxStatsW0..W7`, but `_wg_softmax` passes `signal_stats_barrier=False`
# (:2784-2786, :2795-2797) and `_epilogue_step` gets `use_stats_barrier=False`
# (:2853), so both guarded sites are compiled out: no `bar.*` operand in any of
# the eight exports names an id in 3..10, and the softmax-to-epilogue stats edge is
# carried by the `mbar_sm_stats` PIPE alone.  Dropping the eight dead ids is also
# what lets this renumbering fit: `bar_epilogue + stage` occupies 13 AND 14, so
# the scheme uses SEVEN ids, 8..14 -- inside the sixteen the hardware has, which
# keeping the eight dead ids would not be.
# instruction_selection: bar.sync (arrive-and-wait) or bar.arrive (arrive-only)
#   with the id and the thread count in REGISTERS, materialized by two `mov.b32`
#   immediately before the barrier (e.g. `mov.b32 %r,11; mov.b32 %r,128;
#   bar.sync %r,%r;`); extent: scalar per site.  Two sites differ: the CTA-wide
#   `:809` init barrier is a literal `bar.sync 0`, and `:892` is the ONE-operand
#   form `bar.sync %r` with no thread count.  Which of arrive-and-wait or
#   arrive-only a site uses is load-bearing and is stated at each site below.

# EVERY mbarrier wait in this kernel -- all 28 of them, pipeline waits and bare
# `mbarrier_wait` alike -- is the SAME four-instruction inline-asm retry loop,
# never a single instruction:
#
#   LAB_WAIT: mbarrier.try_wait.parity.shared[::cta].b64 P1, [addr], phase, 10000000;
#             @P1 bra.uni DONE;
#             bra.uni LAB_WAIT;
#   DONE:
#
# The `10000000` is a cycle TIMEOUT operand, and the loop re-tries on expiry. The
# annotations below write this form as `THE_WAIT`, and give the extent as "one
# retry loop".  Only the state space varies: 26 sites are `.shared`, and the two
# inside `gemm_ptx_partial` are `.shared::cta`.

# ===========================================================================
# Prologue (:603-960)
# ===========================================================================
block = cta_id()
tid   = thread_id()
warp  = warp_id()

work_count = load_global(mWorkCount, 0)
# instruction_selection: ld.global.u32 (:669); extent: scalar.
cta_valid_work = block < work_count

# CTA-WIDE, not thread 0: every thread reads all six scheduler columns, which
# is why head_kv_idx / batch_idx / kv_block_idx are available to every role
# without going through sRowMeta.  The export issues these before `%tid.x` is
# even read.
if cta_valid_work:
    head_kv_idx  = load_global(mSchedulerMetadata, block * 6 + 0)
    row_linear   = load_global(mSchedulerMetadata, block * 6 + 1)
    work_q_begin = load_global(mSchedulerMetadata, block * 6 + 2)
    work_q_count = load_global(mSchedulerMetadata, block * 6 + 3)
    batch_idx    = load_global(mSchedulerMetadata, block * 6 + 4)
    kv_block_idx = load_global(mSchedulerMetadata, block * 6 + 5)
    # instruction_selection: ld.global.u32 x5 and ld.global.s32 x1 (:671-676);
    #   extent: scalar each.

if warp == 0:
    if not use_q_gather4:
        prefetch_descriptor(q_tma_atom)
        # instruction_selection: prefetch.tensormap (:693), issued by the whole
        #   warp; extent: scalar.
    else:
        with elect():
            prefetch_descriptor(q_gather4_desc)
            # instruction_selection: prefetch.tensormap (:696) under elect.sync;
            #   extent: scalar.  The two Q programs differ here: the CUTE atom
            #   path is unelected, the raw-descriptor path is elected.

    # Pipeline mbarrier init: WARP 0, all lanes, NO elect, and BEFORE the
    # thread-0 metadata block.  (The sibling port moved this to another warp as
    # a performance change; that is not this source and not this sketch.)
    # One pipeline at a time -- ALL of a pipeline's barriers are initialized
    # before the next pipeline starts, in the order q, s, p, p_lastsplit, o,
    # sm_stats.  The export emits them in exactly that grouping, not interleaved
    # by stage.
    for stage in range(Q_STAGE): init(mbar_q_full[stage], 1); init(mbar_q_empty[stage], 1)
    for stage in range(S_STAGE): init(mbar_s_full[stage], 1); init(mbar_s_empty[stage], 128)
    for stage in range(S_STAGE): init(mbar_p_full[stage], 128); init(mbar_p_empty[stage], 1)
    for stage in range(S_STAGE): init(mbar_p_last_full[stage], 128); init(mbar_p_last_empty[stage], 1)
    for stage in range(O_STAGE): init(mbar_o_full[stage], 1); init(mbar_o_empty[stage], 128)
    for stage in range(O_STAGE): init(mbar_sm_stats_full[stage], 128); init(mbar_sm_stats_empty[stage], 128)
    # instruction_selection: mbarrier.init.shared.b64 x24 (pipeline.py:62,:169,
    #   :269,:324); extent: scalar each.  The arrive counts 1/1/1/128/128/1/128/
    #   128 are confirmed against the export operand by operand.

fence("mbarrier_init.release.cluster")
barrier()                     # CTA-wide, pair #1
# instruction_selection: fence.mbarrier_init.release.cluster (:808) then
#   bar.sync 0 (:809); extent: scalar.  This pair brackets the PIPELINE init.

if tid == 0:
    base_row_start = load_global(mK2qCounts, head_kv_idx * (rows + 1) + row_linear)
    row_start = base_row_start + work_q_begin
    count_raw = work_q_count
    # The logical K length, in the source's own priority order (:205-216).
    # `seqused` is tested FIRST; checking paged first would silently substitute
    # the paged capacity for a shorter supplied length.
    if HAS_SEQUSED_K:
        seqlen_k = load_global(mSeqUsedK, batch_idx)
        # instruction_selection: ld.global.u32 (1 static); extent: scalar.
    elif PAGED_KV:
        seqlen_k = pages_per_seq * N_BLOCK
        # instruction_selection: integer multiply, no load; extent: scalar.
        #   The FULL paged capacity, zero-padded tail pages included.
    else:
        seqlen_k = load_global(mCuSeqlensK, batch_idx + 1) - load_global(mCuSeqlensK, batch_idx)
    q_batch_offset = load_global(mCuSeqlensQ, batch_idx)
    k_batch_offset = 0 if PAGED_KV else load_global(mCuSeqlensK, batch_idx)
    # instruction_selection: ld.global.u32, and the COUNT is per axis, not a
    #   constant.  Each cu_seqlens length is a difference, and both the paged
    #   and non-causal arms delete loads: flat+causal 4, flat+non-causal 3,
    #   paged+causal 2 (both from mCuSeqlensQ -- the paged export carries no
    #   mCuSeqlensK load at all, since `_logical_seqlen_k` takes the multiply
    #   arm and `k_batch_offset` is the immediate 0 at :841-845),
    #   paged+non-causal 1.
    kv_valid_cols  = clamp(seqlen_k - kv_block_idx * N_BLOCK, 0, N_BLOCK)
    if CAUSAL:
        seqlen_q = load_global(mCuSeqlensQ, batch_idx + 1) - q_batch_offset
        # instruction_selection: ld.global.u32 (.loc 1 855 23), 1 static;
        #   extent: scalar.  ONLY under causal -- my non-causal export has zero
        #   `.loc 1 855` and zero `.loc 1 862`, so hoisting this out of the
        #   branch gives the non-causal build a load the source does not have.
        causal_q_offset = seqlen_k - seqlen_q
        # May be NEGATIVE under `seqused_k`, which is legal: the leading
        # `seqlen_q - seqlen_k` queries then have no valid key, run the
        # all-masked path, and store a neutral partial whose LSE is exactly
        # -inf. Clamping it to a finite value diverges from the source.
    else:
        causal_q_offset = 0                       # (:893-901 analogue)
    if PAGED_KV:
        page_idx = load_global(mPageTable, batch_idx * pages_per_seq + kv_block_idx)
        # instruction_selection: ld.global.u32 (1 static); extent: scalar.
        store_shared(sPagedKvIdx, 0, page_idx)
        # instruction_selection: st.shared.u32 (1 static); extent: scalar.  A
        #   separate scalar store -- it targets a different array than the two
        #   sRowMeta vector stores -- riding the same thread-0 region and the
        #   same publish fence, so paging adds no barrier (:864-867, :890-891).
    store_shared(sRowMeta, 0..7, those values)
    # instruction_selection: st.shared.v4.u32 (:849, :863) -- the eight words go
    #   out as two four-word stores; extent: 4 words each.

    # Causal diagonal split point over the CSR row, which is sorted by q_idx.
    # The whole search sits inside `const_expr(self.causal)` AND, within it,
    # behind a runtime guard: a row with no work or no visible column skips it
    # (:874-889).  Under `causal=False` the region is compiled out entirely --
    # my non-causal export carries zero `.loc 1 277/278/285` and zero
    # `.loc 1 243`, against 1/2/1 and 2 in the causal build.
    diag_q_count = 0
    if CAUSAL:
        row_has_visible_cols = count_raw > 0 and kv_valid_cols > 0
        if row_has_visible_cols:
            # The first q_idx at or past the diagonal, where the threshold is
            # the row's last visible column mapped back into q space.
            q_threshold = kv_block_idx * N_BLOCK + kv_valid_cols - causal_q_offset
            # A FIXED 32-trip loop with `unroll=1` and a predicated body -- not
            # a data-dependent `while`.  The export keeps it rolled
            # ($L__BB0_7, one probe load in the body), so 32 is an upper bound
            # on int32-sized rows rather than a trip count run to convergence.
            left, right = 0, count_raw
            for _ in range(32):                       # rolled, unroll=1
                if left < right:
                    mid = (left + right) // 2
                    probe = load_global(mK2qIndices,
                                        head_kv_idx * nnz + row_start + mid)
                    # instruction_selection: ld.global.u32 (:243), ONE per
                    #   iteration inside the rolled loop; extent: scalar.
                    left, right = (mid + 1, right) if probe < q_threshold else (left, mid)
            diag_q_count = left
    store_shared(sDiagQCount, 0, diag_q_count)
    # instruction_selection: st.shared.u32 (:890); extent: scalar.  The one
    #   scalar the metadata block publishes outside the two sRowMeta vector stores.

    # K/V single-shot barriers: init and expect_tx, both unconditional.
    init(mbar_k_tma[0], 1); expect_tx(mbar_k_tma[0], N_BLOCK * PACKED_HEAD_DIM)
    init(mbar_v_tma[0], 1); expect_tx(mbar_v_tma[0], N_BLOCK * PACKED_HEAD_DIM)
    init(mbar_k[0], 1)
    init(mbar_v[0], 1)
    # instruction_selection: mbarrier.init.shared.b64 x4 (:868-872) and
    #   mbarrier.expect_tx.relaxed.cta.shared::cta.b64 x2 (:871,:873); extent:
    #   scalar each.  The transaction count is 8192, the PACKED byte count --
    #   half the sibling's, because the tile is FP4.

fence("mbarrier_init.release.cluster")
barrier()                     # CTA-wide, pair #2
# instruction_selection: fence.mbarrier_init.release.cluster (:891) then
#   bar.sync 0 (:892); extent: scalar.  This second pair brackets the K/V
#   mbarrier init and the metadata publish.  Both pairs are reached by ALL
#   warps, including ones with no work.

# ===========================================================================
# Role dispatch -- a flat sequence of independent `if`s, in source order
# ===========================================================================
if warp == 15:
    set_register_budget(dec=48); return
    # instruction_selection: setmaxnreg.dec.sync.aligned.u32 (:947); extent: scalar.

if 8 <= warp < 12 and cta_valid_work:
    set_register_budget(dec=112); q_load_warpgroup()
    # instruction_selection: setmaxnreg.dec.sync.aligned.u32 (:956); extent:
    #   scalar.  There is NO matching increase: the export carries only two
    #   `setmaxnreg.inc`, both in the softmax warpgroups.
if 13 <= warp < 15 and cta_valid_work:
    set_register_budget(dec=48);  kv_load_warp(is_v=(warp == 14))
    # instruction_selection: setmaxnreg.dec.sync.aligned.u32 (:1014); extent:
    #   scalar.
if warp == 12 and cta_valid_work:
    set_register_budget(dec=48);  mma_issue_warp()
    # instruction_selection: setmaxnreg.dec.sync.aligned.u32 (:1035); extent:
    #   scalar.
if warp < 4 and cta_valid_work:
    set_register_budget(inc=176); softmax_warpgroup(stage=0)
if 4 <= warp < 8 and cta_valid_work:
    set_register_budget(inc=176); softmax_warpgroup(stage=1)
    # instruction_selection: setmaxnreg.inc.sync.aligned.u32 (:1064,:1131);
    #   extent: scalar each -- the only two increases in the kernel.

# ===========================================================================
# KV load warps 13/14 (:1676-1777)
# ===========================================================================
def kv_load_warp(is_v):
    if has_work:                     # no-work CTAs do nothing at all here
        if PAGED_KV:
            page_idx = load_shared(sPagedKvIdx, 0)
            # instruction_selection: ld.shared.u32 (paged export,
            #   .loc 1 1709 53 for K / .loc 1 1744 53 for V, 1 static each);
            #   extent: scalar, WARP-WIDE.  It sits OUTSIDE the elected region:
            #   the export shows this load immediately before the copy helper's
            #   `elect.sync` (.loc 1 1734 / :1769), so all 32 lanes execute it.
            #   Once per KV-load warp and reused across the issue -- a claim
            #   that holds HERE and not in the dequant loops, where the same
            #   cell is re-read every iteration.
        with elect():                # region 1 (:1734/:1769) -- the copy only
            copy_g2s(mV if is_v else mK, sVFp4 if is_v else sKFp4,
                     bar=mbar_v_tma if is_v else mbar_k_tma, hint=0)
            # instruction_selection, PAGED: cp.async.bulk.tensor.**4d**
            #   .shared::cluster.global.tile.mbarrier::complete_tx::bytes
            #   .L2::cache_hint (paged export, 2 static -- one for K, one for V,
            #   where the BF16 sibling needs two each because its row is two
            #   swizzle atoms wide and the packed FP4 row is one).  Coordinates
            #   are fastest-first `(head_dim_offset, token_in_page, head_kv,
            #   page)` with the token coordinate pinned to 0.  Load mode,
            #   completion and cache hint are UNCHANGED from the flat form.
            # instruction_selection, FLAT: cp.async.bulk.tensor.3d.shared::cluster
            #   .global.tile.mbarrier::complete_tx::bytes.L2::cache_hint
            #   (:1734,:1769, 2 static); extent: a (PACKED_HEAD_DIM, 1, N_BLOCK)
            #   box = 8192 bytes.  Half the sibling's transaction size, because
            #   the tile is packed FP4.
            #   The `.L2::cache_hint` qualifier is present but the policy operand
            #   is ZERO (`mov.u64 %rd, 0`): these go through quack's
            #   `tma_get_copy_fn`, not MSA's own cached helper.  Only the
            #   gather4 Q path sets a real policy.
        with elect():                # region 2 -- a SEPARATE elect.sync
            commit(mbar_v_tma if is_v else mbar_k_tma)
            # instruction_selection: elect.sync then
            #   mbarrier.arrive.release.cta.shared::cta.b64 (:1740,:1775);
            #   extent: scalar.  The export has two distinct `elect.sync`
            #   regions here, not one spanning both operations -- the arrive
            #   carries `.loc 1 295`, the copy `.loc 1 1734`.  expect_tx was set
            #   in the prologue.
        named_barrier_arrive_and_wait(bar_kv_load)
        # instruction_selection: bar.sync id=10, 64 threads (:1777); extent:
        #   warp pair.  Inside the has_work region -- a no-work CTA contributes
        #   no arrival here.
```

```python
# ===========================================================================
# The dequantization pass (:1338-1674 bodies, :1779-1865 wrappers)
#
# This is the region with no counterpart in the sibling. It runs ONCE per CTA,
# before the softmax warpgroup enters its Q-group loop, and it is what turns
# the packed FP4 staging tile into a legal MMA operand. WG0 takes K, WG1 V.
# ===========================================================================

# The scale tensor is stored in cuBLAS 128x4 tiled order, so a (row, col) pair
# has to be mapped before the byte can be read. Kept in the sketch, against the
# usual rule about address arithmetic, because getting it wrong produces
# plausible values rather than a fault.
def scale_128x4_offset(row, col):
    tiles_n = (SCALE_COLS + 3) // 4                  # 2 for SCALE_COLS = 8
    return ((row // 128) * tiles_n + col // 4) * 512 \
         + (row % 128 % 32) * 16 \
         + (row % 128 // 32) * 4 \
         + (col % 4)
# instruction_selection: integer shift/mask/mad chain, no memory op (:1204-1211);
#   extent: scalar.

# THE one genuine NVFP4 paged difference. `scale_128x4_offset` above is
# layout-only and identical either way; what moves is the logical row fed into
# it. Both forms are the row-major flattening of the tensor the kernel actually
# reads, which is why the host quantizer needs no paged-specific helper -- it
# flattens `prod(shape[:-1])` and the order falls out (quantize.py:174-239).
#
# The two arms take DIFFERENT arguments, and that is the whole point. The flat
# arm takes the absolute token `k_batch_offset + kv_block_idx*N_BLOCK + row`;
# the paged arm takes the INTRA-PAGE row `row`, 0..N_BLOCK-1, and never
# computes the absolute token at all. Feeding the absolute token to the paged
# form yields `(page*H+h)*N_BLOCK + k_batch_offset + kv_block_idx*N_BLOCK + row`
# -- a different, in-range E4M3 byte for every task. It does not fault and the
# values look plausible; only the bitwise gate catches it.

def paged_kv_scale_row(row, head_kv_idx, page_idx):    # (:1328-1336)
    return (page_idx * num_heads_kv + head_kv_idx) * N_BLOCK + row
    # instruction_selection: mad.lo.s32 then shl.b32 by 7 then add.s32
    #   (paged export, .loc 1 1336 12 and .loc 1 1336 11); extent: scalar.
    #   Against flat the whole-module counts move `cvt.s64.s32` 44 -> 46,
    #   `mul.lo.s64` 40 -> 41, `shl.b64` 38 -> 39 -- one extra multiply and
    #   add, nothing structural.

def flat_kv_scale_row(token, head_kv_idx):             # (:1319-1326)
    return token * num_heads_kv + head_kv_idx
    # instruction_selection: integer mad, no memory op; extent: scalar.

def dequant_kv(sFp4, sDst, mbar_tma, mbar_ready, dequant_bar, mScale, is_v):
    if not has_work:
        return
    wait(mbar_tma[0], phase=0)
    # instruction_selection: THE_WAIT
    #   (:1803,:1847); extent: one retry loop.  This waits for the PACKED bytes;
    #   `mbar_ready` below is what means "MMA-legal tile".

    # ---------------------------------------------------------------- BF16 Q
    # 1024 sixteen-element chunks over 128 threads, rolled (`unroll=1`).
    if mma_dtype is bf16:
        TOTAL_TASKS = N_BLOCK * SCALE_COLS           # 128 * 8 = 1024
        task = warp_in_wg * 32 + lane
        for task_idx in range(task, TOTAL_TASKS, NUM_DEQUANT_WARPS * 32):  # rolled
            # NUM_DEQUANT_WARPS is `warps_per_group` = 4, the SOFTMAX
            # warpgroup's own warps -- the ones running this body.  Numerically
            # 128 threads, same as the Q-load warpgroup, but a different
            # constant in the source and it must not be conflated.
            row       = task_idx // SCALE_COLS
            scale_col = task_idx - row * SCALE_COLS
            if PAGED_KV:
                page_idx  = load_shared(sPagedKvIdx, 0)
                # instruction_selection: ld.shared.u32, ONE PER ITERATION
                #   (paged export: .loc 1 1450 19 at the K site inside the
                #   `.pragma "nounroll"` loop $L__BB0_89, .loc 1 1625 19 at the
                #   V site inside $L__BB0_120 -- `dequant_kv` is parameterised
                #   by `is_v`, so this one arm covers both); extent: scalar.
                #   The backend does NOT hoist it out of the task loop, so a
                #   port that lifts it changes the hottest loop's instruction
                #   count.  This is a different read from the KV-load warps'
                #   one -- that one really is once per warp; this one is not.
                scale_row = paged_kv_scale_row(row, head_kv_idx, page_idx)
            else:
                token     = k_batch_offset + token_in_block_base + row
                scale_row = flat_kv_scale_row(token, head_kv_idx)

            r_words = reg_tile([2], "u32")           # 8 bytes = 16 FP4 values
            copy_s2r(sFp4[row * PACKED_HEAD_DIM + scale_col * 8 ..+ 8], r_words)
            # All four dequant sites compute the SAME linear byte address,
            # `row * PACKED_HEAD_DIM + byte_col` (:1398, :1464, :1572, :1639),
            # even though the two staging tiles are declared with opposite
            # majorness: sKFp4 is (N_BLOCK, PACKED_HEAD_DIM) stride
            # (PACKED_HEAD_DIM, 1) and sVFp4 is (PACKED_HEAD_DIM, N_BLOCK)
            # stride (1, PACKED_HEAD_DIM) (:418-431).  Column-major V makes
            # that linear byte element (byte_col, row), i.e. the same
            # token-major packed bytes K reads from (row, byte_col).  The
            # tiles differ in declared layout but not in addressing.
            #
            # instruction_selection: ld.shared.v2.u32 (:1472 K, :1647 V), ONE per
            #   iteration; extent: 16 FP4 values as two 32-bit words.

            scale_byte = load_global(mScale,
                                     scale_128x4_offset(scale_row, scale_col))
            # instruction_selection: ld.global.u8 (:1233), ONE per task, issued
            #   inside the loop; extent: scalar.  ZERO-extending here; the FP8
            #   path's loader is sign-extending instead, which is a real operand-
            #   form difference and not a transcription liberty.
            #   The scale tensor is never staged in shared memory -- each task
            #   reads its own byte straight from global.  A 512-byte scale tile
            #   serves a whole 128-row block, so these hit L1.

            r_scale = reg_tile([1], "u32")           # bf16x2, the byte replicated
            cast(r_scale, scale_byte)
            # instruction_selection: a pure BIT-MANIPULATION chain, no float
            #   conversion instruction at all (utils.py:830
            #   `cvt_fp8_e4m3_to_bf16x2_replicated` -> :540-567
            #   `cvt_fp8x4_e4m3_bf16x4`); extent: scalar.  The export shows, at
            #   `.loc 6 834` (bf16q_tmaq_qh16_fp32.ptx:1663-1678):
            #     mul.lo.s32  t, byte, 16843009      // 0x01010101, replicate
            #     prmt.b32    q, t, t, 0x1302
            #     and.b32     out,  q, 0x80008000    // sign
            #     and.b32     m,    q, 0x7f007f00    // exponent+mantissa
            #     shr.u32     m,    m, 4
            #     or.b32      out,  out, m
            #     fma.rn.bf16x2 r_scale, out, 0x7b807b80, 0
            #   Only the `mul.lo.s32` carries `.loc 6 834`; the rest is one
            #   `llvm.inline_asm` block inheriting the caller's line.  The byte
            #   replication is that `mul.lo.s32` plus the `prmt.b32` -- it is NOT
            #   a broadcast operand inside a conversion.  `cvt.rn.f16x2.e4m3x2`
            #   is ZERO whole-kernel in every BF16-Q export (it is 8 in the FP8-Q
            #   ones, from a different helper), so an annotation naming it here
            #   would name an instruction this build never emits.  The E4M3 byte
            #   lands in both bf16 lanes so one packed multiply scales a value
            #   pair, and the `0x7b807b80` constant is the E4M3->BF16 exponent
            #   bias folded into the fma.

            if has_k_global and not is_v:
                r_global = reg_tile([1], "u32")
                cast(r_global, load_global(mKGlobalScale, 0))
                # instruction_selection: ld.global.f32 (:1479) INSIDE the loop --
                #   not hoisted -- then cvt.rn.bf16x2.f32 broadcasting the FP32
                #   tensor scale into both bf16 lanes; extent: scalar.  The whole
                #   kernel emits zero cvt.rn.f16x2.f32, so this is the bf16x2
                #   form despite the helper's name.
                mul(r_scale, r_scale, r_global, lanes=2)
                # instruction_selection: mul.rn.bf16x2 (:1480-1483), ONE per task;
                #   extent: scalar.  This single instruction is the whole
                #   `has_k_global_scale` difference on the BF16 path, and the
                #   export counts it: whole-kernel `mul.rn.bf16x2` is 17 with the
                #   tensor scale and 16 without, and `cvt.rn.bf16x2.f32` is 145
                #   against 144, on otherwise identical builds.

            r_vals = reg_tile([16], "bf16")
            cast(r_vals, r_words)
            # instruction_selection: cvt.rn.f16x2.e2m1x2 x8 per 16-element tile,
            #   then per pair cvt.f32.f16 x16 + cvt.rn.bf16x2.f32 x8
            #   (utils.py:620-648 then :775-800); extent: a 16-element register
            #   tile.  NOT the single-instruction cvt.rn.bf16x2.e2m1x2: the
            #   pinned toolchain reports CUDA 12.9 and takes the fallback, and
            #   the export shows 16 cvt.rn.f16x2.e2m1x2 across the two dequant
            #   bodies with zero cvt.rn.bf16x2.e2m1x2.
            mul(r_vals, r_vals, r_scale, lanes=2)
            # instruction_selection: mul.rn.bf16x2 x8, one per value pair
            #   (utils.py:804); extent: a 16-element register tile.

            for v in range(2):                       # constexpr, unrolled
                copy_r2s(r_vals[v * 8 : v * 8 + 8], dst_view(sDst, row, scale_col, v, is_v))
                # instruction_selection: st.shared.v4.b32 (:1509 K, :1673 V), 2
                #   per task; extent: 8 bf16 values per instruction.  `is_v`
                #   selects the MN-major destination view -- an indexing
                #   difference only, same instruction.

    # ----------------------------------------------------------------- FP8 Q
    # Pair form: 32 FP4 values and TWO scale bytes per task, 512 tasks.
    else:
        TOTAL_PAIRS = N_BLOCK * (SCALE_COLS // 2)    # 128 * 4 = 512
        task = warp_in_wg * 32 + lane
        for pair_idx in range(task, TOTAL_PAIRS, NUM_DEQUANT_WARPS * 32):  # rolled
            row       = pair_idx // (SCALE_COLS // 2)
            pair_col  = pair_idx - row * (SCALE_COLS // 2)
            if PAGED_KV:
                page_idx  = load_shared(sPagedKvIdx, 0)
                # instruction_selection: ld.shared.u32, ONE PER ITERATION, at
                #   this program's own sites -- .loc 1 1384 (K) and .loc 1 1558
                #   (V); extent: scalar.  Not hoisted, same as the BF16-Q path.
                #   Measured directly in `paged_fp8q_gather4_qh4_bf16partial`:
                #   the cell sits at +3888 in that layout and carries FOUR
                #   reads, whose nearest preceding `.loc` are 1709 53 and
                #   1744 53 (the two KV-load warps) and 1384 23 / 1558 23
                #   (these two pair sites).  Same four-read shape as the BF16-Q
                #   paged build at its own offset +3504.
                scale_row = paged_kv_scale_row(row, head_kv_idx, page_idx)
            else:
                token     = k_batch_offset + token_in_block_base + row
                scale_row = flat_kv_scale_row(token, head_kv_idx)

            r_words = reg_tile([4], "u32")           # 16 bytes = 32 FP4 values
            copy_s2r(sFp4[row * PACKED_HEAD_DIM + pair_col * 16 ..+ 16], r_words)
            # instruction_selection: ld.shared.v4.u32, ONE per iteration; extent:
            #   32 FP4 values.

            scale_lo = load_global(mScale, scale_128x4_offset(scale_row, 2 * pair_col))
            scale_hi = load_global(mScale, scale_128x4_offset(scale_row, 2 * pair_col + 1))
            # instruction_selection: ld.global.s8 x2 (:1254); extent: scalar each.
            #   SIGN-extending, unlike the BF16 path's ld.global.u8 -- the byte is
            #   consumed as an Int32 here.  The two blocks a pair task covers are
            #   adjacent in `col`, but the 128x4 map does not make their bytes
            #   adjacent, so this is two loads rather than one 16-bit load.

            r_vals = reg_tile([32], "fp8e4m3")
            cast(r_vals, r_words, scale_lo, scale_hi)
            # instruction_selection: per 8 values -- prmt.b32 to broadcast the
            #   scale byte, cvt.rn.f16x2.e4m3x2 on the scale, cvt.rn.f16x2.e2m1x2
            #   x4, mul.rn.f16x2 x4, then cvt.rn.satfinite.e4m3x2.f16x2 x4
            #   (utils.py:734-763); extent: a 32-element register tile.  Per
            #   dequant body the export counts 16 cvt.rn.f16x2.e2m1x2, 4
            #   prmt.b32, 4 cvt.rn.f16x2.e4m3x2, 16 mul.rn.f16x2, 16
            #   cvt.rn.satfinite.e4m3x2.f16x2, and ZERO mul.e4m3x4.
            #   The K TENSOR scale is absent here on purpose: it cannot be folded
            #   into E4M3 without overflowing, so it is applied later on the FP32
            #   S accumulator instead.

            for v in range(2):                       # constexpr, unrolled
                copy_r2s(r_vals[v * 16 : v * 16 + 16], dst_view(sDst, row, pair_col, v, is_v))
                # instruction_selection: st.shared.v4.u32 (:1423,:1598); extent:
                #   16 fp8 values per instruction.

    fence("view_async_shared")
    # instruction_selection: fence.proxy.async.shared::cta (:1510,:1674, 2
    #   static); extent: scalar.
    named_barrier_arrive_and_wait(dequant_bar)
    # instruction_selection: bar.sync id=11 or 12, 128 threads (:1818,:1862);
    #   extent: warpgroup.  Unconditional here; the sibling's equivalent exists
    #   only under its fp8-staging predicate.
    if warp_in_wg == 0:
        with elect():
            commit(mbar_ready[0])
            # instruction_selection: mbarrier.arrive.release.cta.shared::cta.b64
            #   (:1821,:1865); extent: scalar.  This is what makes the MMA warp's
            #   `wait(mbar_k)` mean "the MMA-legal tile is ready" rather than
            #   "the packed bytes landed".

# ===========================================================================
# Q-load warpgroup, warps 8..11 (:1867-2188)
# ===========================================================================
def q_load_warpgroup():
    """Two structurally different programs, not one body with a branch at the end.

    They differ in how many `bar_load_wg` sites they have, which buffers they
    publish, which threads do the publishing, and whether the issue block is
    elected.
    """
    if use_q_gather4:
        for group in range(num_q_groups):            # rolled, unroll=1
            if warp_in_wg == 0:
                acquire(mbar_q_empty[stage], phase ^ 1)
                expect_tx(mbar_q_full[stage],
                          M_BLOCK * HEAD_DIM * width_bytes(q_dtype))
                # instruction_selection: THE_WAIT then
                #   mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 under
                #   elect.sync (:1904); extent: one retry loop then one scalar
                #   arrive carrying 32768 bytes for BF16 Q, 16384 for FP8 Q.
                #   BOTH programs go through the same
                #   `pipeline_q.producer_acquire_w_index_phase`, so the
                #   transaction count is declared here too -- whole-kernel
                #   `mbarrier.arrive.expect_tx` is 2 in EVERY build, gather4 and
                #   TMA alike.
            named_barrier_arrive_and_wait(bar_load_wg)
            # instruction_selection: bar.sync id=9, 128 threads (:1906); extent:
            #   warpgroup.  Site 1 of THREE on this path.

            # The publish is spread over all four load warps, not done by warp 0
            # alone, and it writes sQIdxMeta ONLY -- the gather4 program has no
            # sQLoadMIdx (:1918-1937).
            for meta_iter in range(meta_iters):      # constexpr
                tok = (meta_iter * NUM_Q_LOAD_WARPS + warp_in_wg) * 32 + lane
                if tok < q_tokens_per_group:
                    qi = group * q_tokens_per_group + tok
                    if qi < count_raw:
                        qsplit = load_global(mK2qQSplitIndices,
                                             head_kv_idx * nnz + row_start + qi)
                        # instruction_selection: ld.global.u32; extent: scalar.
                        store_shared(sQIdxMeta, meta_slot + tok, qsplit)
                        # instruction_selection: st.shared.u32 (:1933); extent:
                        #   scalar.
                    else:
                        store_shared(sQIdxMeta, meta_slot + tok, 0)
                        # instruction_selection: st.shared.u32 (:1937); extent:
                        #   scalar.  Zero marks the slot dead for the row decode
                        #   below and for the epilogue.
            named_barrier_arrive_and_wait(bar_load_wg)
            # instruction_selection: bar.sync id=9, 128 threads (:1938); extent:
            #   warpgroup.  Site 2: the row decode below reads sQIdxMeta.

            with elect():                            # (:1940) one lane per warp
                for g in range(gathers_per_warp):    # 8, constexpr
                    gather_idx = g * NUM_Q_LOAD_WARPS + warp_in_wg
                    # A gather always covers FOUR rows, but how many TOKENS
                    # those rows come from is `tokens_per_gather4 = 4 //
                    # qheadperkv`, and the source has three constexpr arms for it
                    # (:1948-2010).  One token contributes `qheadperkv`
                    # consecutive rows, because the flat Q view is
                    # ((q_abs * num_heads_kv + head_kv) * qheadperkv + h).
                    tok_base = gather_idx * tokens_per_gather4
                    rows = []
                    for t in range(tokens_per_gather4):
                        qi = group * q_tokens_per_group + tok_base + t
                        if qi < count_raw:
                            q_idx = load_shared(sQIdxMeta, meta_slot + tok_base + t) & 0xFFFFFF
                            base = ((q_batch_offset + q_idx) * num_heads_kv
                                    + head_kv_idx) * qheadperkv
                        else:
                            base = q_oob_m_idx * qheadperkv
                        rows += [base + h for h in range(qheadperkv)]
                    r0, r1, r2, r3 = rows
                    # instruction_selection: ld.shared.u32 then `and.b32
                    #   0xFFFFFF`, `tokens_per_gather4 * gathers_per_warp` per
                    #   warp; extent: scalar each.  The count and the `.loc` move
                    #   with qheadperkv: 8 at `.loc 1 2002` for qh=4 (one word
                    #   per gather), 8 at `:1981` plus 8 at `:1988` for qh=2 (two
                    #   words), and four decodes at `:1960/:1964/:1968/:1972` for
                    #   qh=1.  The `qheadperkv == 1` arm is the only one where
                    #   `q_oob_m_idx` is used unmultiplied.  The row coordinates
                    #   come from sQIdxMeta via the qsplit decode -- this path
                    #   never reads sQLoadMIdx.
                    if q_dtype is fp8:
                        copy_g2s_gather4(sQ, byte_off, q_desc, col=0,
                                         r0, r1, r2, r3,
                                         bar=mbar_q_full[stage], hint=EVICT_LAST)
                        # instruction_selection: cp.async.bulk.tensor.2d
                        #   .shared::cta.global.tile::gather4
                        #   .mbarrier::complete_tx::bytes.cta_group::1
                        #   .L2::cache_hint (:2045), ONE per gather, 8 whole-
                        #   kernel; extent: four rows of box_x=128 elements.
                        #   NO prefetch on this arm -- the FP8 gather4 build
                        #   carries zero cp.async.bulk.prefetch.
                    else:
                        for ks_c in range(2):        # k_stages, constexpr
                            if ks_c + 1 < 2:
                                prefetch_g2s_gather4(q_desc, col=(ks_c + 1) * K_TILE,
                                                     r0, r1, r2, r3, hint=EVICT_LAST)
                                # instruction_selection: cp.async.bulk.prefetch
                                #   .tensor.2d.L2.global.tile::gather4
                                #   .L2::cache_hint (:2036), 8 whole-kernel;
                                #   extent: four rows.
                            copy_g2s_gather4(sQ, byte_off + ks_c * k_tile_stride,
                                             q_desc, col=ks_c * K_TILE,
                                             r0, r1, r2, r3,
                                             bar=mbar_q_full[stage], hint=EVICT_LAST)
                            # instruction_selection: cp.async.bulk.tensor.2d
                            #   .shared::cta.global.tile::gather4...
                            #   .L2::cache_hint (:2045), TWO per gather, 16
                            #   whole-kernel; extent: four rows of box_x=64.
                            #   BF16 Q splits the row into two k-tiles; FP8 does
                            #   not. The policy IS set on this path: EVICT_LAST
                            #   (0x14F0000000000000), which suits a row every
                            #   topK CTA re-reads.  The export carries that
                            #   constant eight times in a gather4 build and zero
                            #   times in a TMA-Q one.
            named_barrier_arrive_and_wait(bar_load_wg)
            # instruction_selection: bar.sync id=9, 128 threads (:2057); extent:
            #   warpgroup.  Site 3, which the TMA program does not have: it
            #   closes the elected issue block before the next group reuses the
            #   slot.

        if do_final_acquire and warp_in_wg == 0:
            acquire(mbar_q_empty[next_slot], next_phase)
            expect_tx(mbar_q_full[next_slot], ...)
            # instruction_selection: THE_WAIT then
            #   mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 (:2065);
            #   extent: one retry loop then one scalar arrive.  The gather4
            #   program has the SAME `do_final_acquire` tail as the TMA one
            #   (:2059-2065) -- it is outside the group loop and leaves the pipe
            #   in the state the next CTA-level user expects.
    else:
        for group in range(num_q_groups):            # rolled, unroll=1
            if warp_in_wg == 0:
                acquire(mbar_q_empty[stage], phase ^ 1)
                expect_tx(mbar_q_full[stage],
                          M_BLOCK * HEAD_DIM * width_bytes(q_dtype))
                # instruction_selection: THE_WAIT then
                #   mbarrier.arrive.expect_tx.release.cta.shared::cta.b64
                #   (:2118) under elect.sync; extent: one retry loop then one
                #   scalar arrive carrying 32768 bytes for BF16 Q, 16384 for FP8
                #   Q.  The WAIT is warp-wide -- all 32 lanes spin -- and ONLY
                #   the arrive sits behind `elect.sync`; the source's guard here
                #   is just `if warp_idx_in_wg == 0` (:2117).  Same shape as the
                #   gather4 arm above, which goes through the same
                #   `producer_acquire_w_index_phase`.  The Q pipe takes its
                #   transaction count at producer acquire, unlike K/V which took
                #   theirs once in the prologue.
            named_barrier_arrive_and_wait(bar_load_wg)
            # instruction_selection: bar.sync id=9, 128 threads (:2120); extent:
            #   warpgroup.  Site 1 of TWO on this path.

            if group_tidx < q_tokens_per_group:
                qi = group * q_tokens_per_group + group_tidx
                if qi < count_raw:
                    qsplit = load_global(mK2qQSplitIndices,
                                         head_kv_idx * nnz + row_start + qi)
                    # instruction_selection: ld.global.u32; extent: scalar.
                    store_shared(sQIdxMeta, meta_slot + group_tidx, qsplit)
                    store_shared(sQLoadMIdx, stage_slot + group_tidx, decode_m_idx(qsplit))
                    # instruction_selection: st.shared.u32 x2 (:2143,:2144);
                    #   extent: scalar each.
                else:
                    store_shared(sQIdxMeta, meta_slot + group_tidx, 0)
                    store_shared(sQLoadMIdx, stage_slot + group_tidx, q_oob_m_idx)
                    # instruction_selection: st.shared.u32 x2 (:2148,:2149);
                    #   extent: scalar each.  The out-of-range arm steers the TMA
                    #   to a safe row instead of a foreign token.
            named_barrier_arrive_and_wait(bar_load_wg)
            # instruction_selection: bar.sync id=9, 128 threads (:2150); extent:
            #   warpgroup.  Site 2: the TMA below reads sQLoadMIdx.

            # The four load warps take DISJOINT token ranges, the same way the
            # gather4 arm partitions by `gather_idx` (:2152-2157).
            for qi_slot in range(tokens_per_warp):   # constexpr
                tok_idx = warp_in_wg * tokens_per_warp + qi_slot
                if tok_idx < q_tokens_per_group:     # :2157
                    # Never false for qheadperkv 8/16, where
                    # `NUM_Q_LOAD_WARPS * tokens_per_warp == q_tokens_per_group`
                    # exactly; the guard is what makes the ceil-division safe in
                    # general.
                    m_tile_idx = load_shared(sQLoadMIdx, stage_slot + tok_idx)
                    # instruction_selection: ld.shared.u32 (:2158), 2 per warp at
                    #   qh16 and 4 at qh8; extent: scalar.  The address carries
                    #   this warp's own `warp_in_wg * tokens_per_warp` stride --
                    #   without it all four warps would fetch the same rows into
                    #   the same subtile slots and three quarters of the Q tile
                    #   would never be loaded.  This is the OTHER half of the
                    #   publish two barriers above: the TMA's row coordinate is
                    #   read back out of sQLoadMIdx, which is why that ring
                    #   exists at all and why the OOB arm writes `q_oob_m_idx`
                    #   into it rather than leaving it stale.
                    for sub in range(subtiles_per_token):
                        copy_g2s(mQ_flat, sQ_load,
                                 row=m_tile_idx,
                                 dst=sub_stage_base + sub * q_tokens_per_group + tok_idx,
                                 bar=mbar_q_full[stage], hint=0)
                        # instruction_selection: cp.async.bulk.tensor.2d
                        #   .shared::cluster.global.tile
                        #   .mbarrier::complete_tx::bytes.L2::cache_hint
                        #   (:2166,:2171); extent: a (swizzle_elems, qheadperkv)
                        #   box.  The static count is `tokens_per_warp *
                        #   subtiles_per_token`, where `subtiles_per_token` is 1
                        #   for FP8 Q and 2 for BF16 Q -- so it tracks
                        #   qheadperkv, NOT dtype alone: the export has 2 copies
                        #   at qh=16 and 4 at qh=8 for FP8 Q, and 4 / 8
                        #   respectively for BF16 Q.  The second BF16 k-tile
                        #   lands `q_tokens_per_group` slots further on.
                        #   Policy operand is ZERO here, like the KV loads --
                        #   this path also goes through quack's copy helper.

        if do_final_acquire and warp_in_wg == 0:
            acquire(mbar_q_empty[next_slot], next_phase)
            expect_tx(mbar_q_full[next_slot], ...)
            # instruction_selection: THE_WAIT then
            #   mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 (:2187)
            #   under elect.sync; extent: one retry loop then one scalar arrive.
            #   Warp-wide wait, elected arrive, exactly as in the loop.  The tail
            #   acquire (`do_final_acquire`) leaves the pipe in the state the
            #   next CTA-level user expects; it is a real second site, not a
            #   repeat of the in-loop one.

# ===========================================================================
# MMA-issue warp 12, also the TMEM allocator (:2189-2436)
# ===========================================================================
def mma_issue_warp():
    allocate_tmem(512, out=tmem_addr)
    # instruction_selection: tcgen05.alloc.cta_group::1.sync.aligned
    #   .shared::cta.b32 (:1041); extent: scalar.
    named_barrier_arrive_and_wait(bar_tmem_alloc)
    # instruction_selection: bar.sync id=8, 288 threads (:1042); extent: the two
    #   softmax warpgroups plus this warp.  It also gates the MMA warp behind
    #   the two dequant passes.

    wait(mbar_k[0], phase=0)
    # instruction_selection: THE_WAIT (:2321);
    #   extent: one retry loop.  The DEQUANT-ready barrier, not the TMA one.

    # Prologue: QK for group 0, and for group 1 when there is one.
    wait(mbar_q_full[0], phase=0)
    acquire(mbar_s_empty[0], phase=1)
    # instruction_selection: THE_WAIT x2
    #   (:2333,:2334); extent: one retry loop each.  The Q wait precedes the S
    #   acquire.
    for ki in range(HEAD_DIM // mma_k):
        gemm(S[0], sQ[0], sK, accumulate=(ki != 0))
        # instruction_selection: tcgen05.mma.cta_group::1.kind::<mma_kind>,
        #   every instance `@leader_thread`-predicated, in a chain of 8 (f16,
        #   mma_k=16) or 4 (f8f6f4, mma_k=32); extent: one 128x128xmma_k tile
        #   each.  48 kind::f16 / 24 kind::f8f6f4 whole-kernel, in six chains of
        #   eight.  Every `gemm(...)` below carries the same predication.
    commit(mbar_s_full[0]); release(mbar_q_empty[0])
    # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one
    #   .shared::cluster.b64 [%r] x2 (:2336,:2337) under elect.sync; extent:
    #   scalar each.  The mnemonic is `mbarrier::arrive::one` with the barrier
    #   as a single register operand.

    if num_q_groups > 1:
        wait(mbar_q_full[1], phase=0); acquire(mbar_s_empty[1], phase=1)
        # instruction_selection: THE_WAIT x2 (:2340,:2341); extent: one retry
        #   loop each.  Same Q-then-S order as the steady loop.  `:2336,:2337`
        #   are the GROUP-0 commit/release pair above, a different instruction
        #   family in the same function.
        for ki in range(HEAD_DIM // mma_k):
            gemm(S[1], sQ[1], sK, accumulate=(ki != 0))
            # instruction_selection: tcgen05.mma...kind::<mma_kind>, every
            #   instance `@leader_thread`-predicated (:2342); extent: one tile
            #   each.
        commit(mbar_s_full[1]); release(mbar_q_empty[1])
        # instruction_selection: tcgen05.commit.cta_group::1
        #   .mbarrier::arrive::one.shared::cluster.b64 [%r] x2 (:2343,:2344)
        #   under elect.sync; extent: scalar each.

    wait(mbar_v[0], phase=0)
    # instruction_selection: THE_WAIT (:2346);
    #   extent: one retry loop.  V is waited AFTER the prologue QKs -- the first PV
    #   is still two groups away, so the dequant of V overlaps them.

    # Steady state: PV of group qi-2, then QK of group qi, reusing the slot the
    # PV just freed.
    for qi in range(2, num_q_groups):                # rolled, unroll=1
        pv_qi = qi - 2
        wait(mbar_p_full[pv_slot], pv_phase)
        acquire(mbar_o_empty[pv_slot], pv_phase ^ 1)
        # instruction_selection: THE_WAIT x2
        #   (:2356,:2357); extent: one retry loop each.
        for ki in range(SPLIT_P_ARRIVE // mma_k):
            gemm(O[pv_slot], P[pv_slot], sV, accumulate=(ki != 0))
            # instruction_selection: tcgen05.mma...kind::<mma_kind> with the A
            #   operand in TMEM; extent: one tile each.
        wait(mbar_p_last_full[pv_slot], pv_phase)
        # instruction_selection: THE_WAIT with `.shared::cta` instead of
        #   `.shared` (PTX 1129 and 1473); extent: one retry loop.  The state
        #   space is the ONLY thing that distinguishes this wait -- 2 of the
        #   kernel's 28 waits are `::cta`, 26 are `.shared`, and all 28 are the
        #   same four-instruction retry loop.  What is special about this site is
        #   WHERE it sits: hand-written inline asm inside `gemm_ptx_partial`,
        #   emitted BETWEEN two `tcgen05.mma` instructions, because the producer
        #   publishes 3/4 of P early and the last quarter separately, so this
        #   wait must land MID-chain where a pipeline wait cannot go.
        for ki in range(SPLIT_P_ARRIVE // mma_k, HEAD_DIM // mma_k):
            gemm(O[pv_slot], P[pv_slot], sV, accumulate=True)
        commit(mbar_o_full[pv_slot])
        release(mbar_p_last_empty[pv_slot]); release(mbar_p_empty[pv_slot])
        # instruction_selection: tcgen05.commit... x3 -- the O commit at :2379,
        #   the two P releases at :2381 and :2382; extent: scalar each.

        wait(mbar_q_full[q_slot], q_phase)
        acquire(mbar_s_empty[s_slot], s_phase ^ 1)
        # instruction_selection: THE_WAIT x2 (:2388,:2389); extent: one retry
        #   loop each.  Q wait BEFORE S acquire, as in the source.
        for ki in range(HEAD_DIM // mma_k):
            gemm(S[s_slot], sQ[q_slot], sK, accumulate=(ki != 0))
        commit(mbar_s_full[s_slot]); release(mbar_q_empty[q_slot])
        # instruction_selection: tcgen05.commit... x2 (:2402); extent: scalar each.

    # Drain the remaining one or two PV tiles.
    drain_begin = 0 if num_q_groups == 1 else num_q_groups - 2
    for pv_qi in range(drain_begin, num_q_groups):   # rolled, unroll=1
        wait(mbar_p_full[pv_slot], pv_phase); acquire(mbar_o_empty[pv_slot], pv_phase ^ 1)
        # instruction_selection: THE_WAIT x2 (:2409,:2410); extent: one retry
        #   loop each.
        ... same split PV chain, including the mid-chain `::cta` wait ...
        commit(mbar_o_full[pv_slot])
        release(mbar_p_last_empty[pv_slot]); release(mbar_p_empty[pv_slot])
        # instruction_selection: tcgen05.commit... (:2434,:2435); extent: scalar
        #   each.

    relinquish_tmem_alloc_permit()
    # instruction_selection: tcgen05.relinquish_alloc_permit.cta_group::1.sync
    #   .aligned (:1054); extent: scalar.  AFTER the MMA body, not right after
    #   the alloc.
    named_barrier_arrive_and_wait(bar_tmem_alloc)
    # instruction_selection: bar.sync id=8, 288 threads (:1055); extent: 288
    #   threads.  The MMA warp arrives AND waits here; the softmax warpgroups
    #   only arrive.
    free_tmem(tmem_addr, 512)
    # instruction_selection: tcgen05.dealloc.cta_group::1.sync.aligned.b32
    #   (:1056); extent: scalar.
    griddepcontrol_launch_dependents()
    # instruction_selection: griddepcontrol.launch_dependents (:1057); extent:
    #   scalar.  Releases the dependent grid once this CTA's TMEM is returned.
```

```python
# ===========================================================================
# Softmax warpgroup, warps 0..3 (stage 0) and 4..7 (stage 1) (:2618-2856)
#
# The whole body below is emitted TWICE, once per warpgroup, with `stage` a
# compile-time constant -- the export shows every single-site operation in it
# appearing exactly twice.
# ===========================================================================
def softmax_warpgroup(stage):
    group_tidx = tid - stage * 128          # 0..127; this thread's M row

    # The dequant pass runs first, before this warpgroup enters its Q-group
    # loop. Unconditional: WG0 always takes K, WG1 always V.
    if stage == 0:
        dequant_kv(sKFp4, sK, mbar_k_tma, mbar_k, bar_dequant_k, mKScale, is_v=False)
    else:
        dequant_kv(sVFp4, sV, mbar_v_tma, mbar_v, bar_dequant_v, mVScale, is_v=True)

    named_barrier_arrive_and_wait(bar_tmem_alloc)
    # instruction_selection: bar.sync id=8, 288 threads (:1089,:1156); extent:
    #   warpgroup.  Entry side -- arrive AND wait.

    # The source const_expr-splits the `_softmax_step` call site itself
    # (:2744-2752 causal vs :2788-2821 non-causal). The causal arm derives the
    # two quantities the mask needs; the non-causal arm passes literal
    # `Int32(0)` for both and never computes `kv_block_col_start` (:2727-2729).
    if CAUSAL:
        kv_block_col_start = kv_block_idx * N_BLOCK          # (:2727-2729)
        diag_q_count = load_shared(sDiagQCount, 0)
        # instruction_selection: ld.shared.u32; extent: scalar.
    else:
        kv_block_col_start = 0
        diag_q_count = 0

    # WG0 takes even Q groups, WG1 odd.
    for qi_iter in range((num_q_groups + (1 - stage)) // 2):
        qi_group = qi_iter * 2 + stage
        if CAUSAL:
            # How many of this group's tokens still sit on the diagonal.
            masked_tok_count = clamp(diag_q_count - qi_group * q_tokens_per_group,
                                     0, q_tokens_per_group)
        else:
            masked_tok_count = 0                             # literal (:2788-2821)
        softmax_step(stage, qi_group)
        named_barrier_arrive_and_wait(bar_epilogue + stage)
        # instruction_selection: bar.sync id=13 or 14, 128 threads (:2822);
        #   extent: warpgroup.  Separates the P publish from the epilogue.
        epilogue_step(stage, qi_group)

    named_barrier_arrive(bar_tmem_alloc)
    # instruction_selection: bar.arrive id=8, 288 threads (:1124,:1191); extent:
    #   warpgroup.  ARRIVE-ONLY on the exit side: the MMA warp is the one that
    #   waits, and making this an arrive-and-wait would park both warpgroups
    #   behind the TMEM teardown for no ordering they need.

def softmax_step(stage, qi_group):
    wait(mbar_s_full[s_slot], phase)
    # instruction_selection: THE_WAIT; extent: one retry loop.
    r_s = reg_tile([128], "f32")
    copy_t2r(S[s_slot], r_s)
    # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x32.b32 x8 whole-
    #   kernel, 4 per warpgroup (:2480); extent: 128 f32 per thread.  32x32b maps
    #   thread -> M row, register -> column.
    #   NOTE: there is NO tcgen05.wait::ld here -- the export's only two are in
    #   the epilogue.  The S read is consumed without a TMEM-load fence.

    if q_dtype is fp8 and has_k_global:
        kg = load_global(mKGlobalScale, 0)
        for j in range(64):
            mul(r_s[2*j : 2*j+2], r_s[2*j : 2*j+2], (kg, kg), lanes=2)
        # instruction_selection: mul.rn.f32x2 x64 per warpgroup, 128 whole-kernel
        #   (:2488), and ZERO in a kgs=False build; extent: the 128-element S
        #   fragment.  THE FP8-Q-ONLY HOOK.  It is emitted between the
        #   tcgen05.ld at :2480 and the mask region, so it feeds row-max and
        #   every exp2 -- its placement is bitwise-load-bearing, not a rescale
        #   that commutes.  On the BF16 path this multiply does not exist: the
        #   tensor scale was folded into the dequantized K values instead.

    # Causal mask. The ENTIRE construct below is inside `const_expr(self.causal
    # and apply_causal_mask)` (:2510-2546): under `causal=False` the caller
    # passes `mask_causal=False` and the runtime branch, the q_idx read and the
    # diagonal column limit are all compiled out, leaving only `kv_valid_cols`.
    # Measured: `.loc 1 2511` 2 -> 0 and `.loc 1 2515` 4 -> 0 between the causal
    # and non-causal exports, with the mask region shrinking 522 -> 498 lines.
    if CAUSAL:
        # A runtime branch on whether this Q group straddles the diagonal. Both
        # arms end in the same bit-test body, but they compute different column
        # limits, and only the diagonal arm reads q_idx.
        need_causal_mask = masked_tok_count > 0
        if need_causal_mask:
            tok = group_tidx // qheadperkv          # :2513
            q_idx = load_shared(sQIdxMeta, meta_slot + tok) & 0xFFFFFF
            # instruction_selection: ld.shared.u32 x2 whole-kernel (:2515);
            #   extent: scalar.  Present only on this arm, and only in a causal
            #   build.
            col_limit = min(kv_valid_cols,
                            q_idx + causal_q_offset - kv_block_col_start + 1)
        else:
            col_limit = kv_valid_cols
    else:
        col_limit = kv_valid_cols               # (:2537-2546)
    if col_limit < N_BLOCK:                 # mask.py:114
        # A fully-visible tile skips the bit-test body entirely -- the mask
        # region below is reached only under this runtime guard, which is why
        # its `.loc 7` instructions sit inside a predicated block.
        for chunk in range(4):
            bits = shift_right_u32(0xFFFFFFFF, col_limit_in_chunk)
            # instruction_selection: shr.u32 (mask.py:19-21 `r2p_bitmask_below`),
            #   13 static; extent: scalar.  `shr.u32` CLAMPS shifts >= 32 to zero
            #   and the mask relies on that; a shift with undefined behaviour there
            #   leaves every chunk past the column limit unmasked.
            select(r_s[chunk * 32 : chunk * 32 + 32], bits, r_s[...], -inf)
            # instruction_selection: and.b32 + setp.eq.s32 + selp.b32 per element
            #   (mask.py:36-46 `mask_r2p_lambda`); extent: a 32-element chunk.  This
            #   is the kernel's densest instruction group -- 248 and.b32, 240
            #   setp.eq.s32 in the mask region's own `.loc 7` instructions (263
            #   selp.b32 is the whole-kernel count).

    # `_compute_row_max` for arch 100 (utils.py:258-276): four accumulators
    # SEEDED from the first eight elements, then a three-source loop from i=8,
    # then a two-step fold.  There is no -inf initialization -- the seed IS the
    # initialization, which is why the loop starts at 8 and runs 15 trips.
    local_max = reg_tile([4], "f32")
    for a in range(4):
        max(local_max[a], r_s[2*a], r_s[2*a + 1])
        # instruction_selection: max.f32, TWO-source form (utils.py:262 for the
        #   first, :266/:267/:268 for the rest), 2 each whole-kernel; extent:
        #   scalar.
    for i in range(8, 128, 8):              # 15 trips
        for a in range(4):
            max(local_max[a], local_max[a], r_s[i + 2*a], r_s[i + 2*a + 1])
        # instruction_selection: max.f32 with THREE source operands, an SM100
        #   form (utils.py:271-274), 30 each whole-kernel = 15 trips x 2
        #   warpgroups; extent: scalar.
    max(local_max[0], local_max[0], local_max[1])
    # instruction_selection: max.f32 two-source (utils.py:275); extent: scalar.
    #   This fold is not optional -- without it local_max[1] is computed and
    #   discarded, and the row max would ignore every element at index congruent
    #   to 2 or 3 mod 8.
    row_max = fmax(local_max[0], local_max[2], local_max[3])
    # instruction_selection: max.f32 three-source (utils.py:276); extent: scalar.
    #   Whole-kernel total 4*2 + 4*30 + 2 + 2 = 132, i.e. 66 per warpgroup --
    #   against 127 for a two-input tree.  The four independent accumulators are
    #   what make that count reachable; one accumulator would serialize the tree.
    row_max_safe = select(row_max == -inf, 0.0, row_max)

    for j in range(64):
        fma(r_s[2*j:2*j+2], r_s[2*j:2*j+2], scale_log2, -row_max_safe*scale_log2, lanes=2)
        # instruction_selection: fma.rn.f32x2, 128 whole-kernel (softmax.py:354);
        #   extent: the S fragment.  Scale and row-max subtract fused.
    # The temperature row sum is computed HERE, before the P handoff, from a
    # SECOND exp2 population over the same fragment (:2551-2555).  This is why a
    # temperature build carries 480 ex2 against 224.
    if has_temperature:
        acc_t = reg_tile([4, 2], "f32")      # four packed accumulators
        fill(acc_t, 0.0)
        for i in range(0, 128, 8):
            for a in range(4):
                mul(r_t[2*a:2*a+2], r_s[i+2*a : i+2*a+2], temp_scale_log2, lanes=2)
                exp2(r_t[2*a]); exp2(r_t[2*a+1])
                add(acc_t[a], acc_t[a], r_t[2*a:2*a+2], lanes=2)
        add(acc_t[0], acc_t[0], acc_t[1], lanes=2)      # utils.py:346
        add(acc_t[2], acc_t[2], acc_t[3], lanes=2)      # utils.py:347
        add(acc_t[0], acc_t[0], acc_t[2], lanes=2)      # utils.py:348
        row_sum_t = acc_t[0][0] + acc_t[0][1]
        # instruction_selection: mul.rn.f32x2 x64, ex2.approx.ftz.f32 x128 and
        #   add.rn.f32x2 x64+3 per warpgroup (`utils.fadd_exp2_scaled_reduce`,
        #   utils.py:308-350); extent: the fragment.  NOTE this population has
        #   NO emulation -- all 128 are real `ex2`, which is exactly why a
        #   temperature build carries 480 `ex2` (224 emulated-main + 256
        #   all-real-temperature) rather than 448.  It reads r_s but does not
        #   write it back, so the P population below still sees the
        #   scale-subtracted values.

    # THE P HANDOFF COMES NEXT, BEFORE ANY STATS WORK.  Both acquires precede
    # the exp2/convert that fills the register the stores read (:2557-2562), and
    # the whole publish completes before the sm_stats acquire.  Ordering it the
    # other way would put the MMA warp's PV wait behind a stats pipe it has no
    # dependence on, and the split-P protocol below assumes the 3/4 arrive is
    # the earliest thing the MMA warp can see after S.
    if SPLIT_P_ARRIVE > 0:
        acquire(mbar_p_last_empty[p_slot], phase)
    acquire(mbar_p_empty[p_slot], phase)
    # instruction_selection: THE_WAIT x2 each
    #   (:2561,:2563); extent: one retry loop each.  The last-split slot is acquired
    #   FIRST -- it is the one gemm_ptx_partial waits on latest, so taking it
    #   first cannot deadlock against the other order.

    # exp2 and the convert to the MMA dtype are ONE pass (`apply_exp2_convert`,
    # :2571-2576), not an exp2 block followed by a cast block: the emulated
    # elements and the converted output come out of the same loop.
    # The emulation predicate is a TWO-dimensional constexpr test over
    # (fragment, position-in-fragment), not a stride over element index. The
    # fragment is 32 elements wide, so the 128-element row is four fragments,
    # and the loop steps in PAIRS (softmax.py:378-396).
    for j in range(4):                       # frg_cnt = 128 // 32
        for k in range(0, 32, 2):
            if EX2_EMU_FREQ == 0:
                # The non-causal build. `apply_exp2_convert` has its own
                # `const_expr(ex2_emu_freq == 0)` arm (softmax.py:381-383) that
                # takes real exp2 for BOTH elements of every pair -- there is no
                # emulation and no `fmax` clamp. Reaching the predicate below
                # with a zero frequency would be a modulo by zero.
                exp2(r_s[j*32+k]); exp2(r_s[j*32+k+1])
                continue
            emulate = (k % EX2_EMU_FREQ >= EX2_EMU_FREQ - EX2_EMU_RES
                       and EX2_EMU_START_FRG <= j < 4 - 1)
            if emulate: exp2_poly_pair(r_s[j*32+k], r_s[j*32+k+1])
            else:       exp2(r_s[j*32+k]); exp2(r_s[j*32+k+1])
        # instruction_selection, NON-CAUSAL (EX2_EMU_FREQ = 0): every element
        #   takes real MUFU -- `ex2.approx.ftz.f32` 224 -> 256 whole-module,
        #   the polynomial `fma.rn.f32x2` 176 -> 128 (all 24 per warpgroup
        #   gone), and `max.f32` 164 -> 132 as the emulation's 32 `fmax(x,-127)`
        #   clamps disappear with it.  Measured against the causal build.
        # instruction_selection, CAUSAL: ex2.approx.ftz.f32 for 112 of every 128
        #   (softmax.py:389-390, 112 at each of two sites), and
        #   `utils.ex2_emulation_2` -- a degree-3 FFMA polynomial evaluated on a
        #   PAIR (`utils.evaluate_polynomial_2`, 3 fma.rn.f32x2 at `.loc 6 924`,
        #   24 per warpgroup = 8 pairs x 3) -- for the other 16; extent: the
        #   fragment. With EX2_EMU_FREQ=16, EX2_EMU_RES=4 (the default) and
        #   EX2_EMU_START_FRG=1, the predicate fires at k in {12,14,28,30} and
        #   j in {1,2}: absolute elements 44-47, 60-63, 76-79, 92-95.  A stride
        #   test like `j % 16 == 0` would emulate elements 0,16,32,... instead --
        #   a DIFFERENT eight pairs, and therefore a different bitwise result,
        #   even though the 112/16 split is the same.  The emulated elements are
        #   deliberately kept off the first and last fragments so the polynomial
        #   never lands on a row edge.
        #   `ex2_emulation_2` also opens with `fmax(x, -127.0)` (utils.py:993),
        #   which the export carries as 32 `max.f32` -- the reason a raw
        #   whole-kernel `max.f32` count is 164 while the row-max tree above is
        #   only 132.  Do not attribute those 32 to the reduction.
        cast(r_p_packed[j*32 : j*32+32], r_s[j*32 : j*32+32])
        # instruction_selection: cvt.rn.bf16x2.f32, 128 whole-kernel
        #   (softmax.py:396-398) for BF16 P, or cvt.rn.satfinite.e4m3x2.f32 for
        #   FP8 P; extent: one 32-element fragment.  The convert is INSIDE the
        #   fragment loop -- `acc_S_row_converted_frg[None, j].store(...)` runs
        #   once per j -- and the exp2 results STAY in r_s, so the row sum below
        #   reduces them.  This is a convert, not a move.

    for k in range(P_SUBTILES):
        copy_r2t(r_p_packed_sub[k], P[p_slot])
        # instruction_selection: tcgen05.st.sync.aligned.32x32b.x16.b32 (BF16 P)
        #   or .x8 (FP8 P), 8 whole-kernel, 4 per warpgroup (:2579); extent: one
        #   subtile.  Four matches the decomposition above: three subtiles
        #   before the split arrive and one after.
        if SPLIT_P_ARRIVE > 0 and k + 1 == split_idx:
            fence("tmem_store")
            commit(mbar_p_full[p_slot])
            # instruction_selection: tcgen05.wait::st.sync.aligned x2 (:2588)
            #   then mbarrier.arrive.release.cta.shared::cta.b64 (:2589); extent:
            #   scalar each.  `split_idx = P_SUBTILES * SPLIT_P_ARRIVE //
            #   n_block_size` -- the 3/4 publish lands on a subtile edge, so the
            #   early arrive never splits a single tcgen05.st.
    fence("tmem_store")
    if SPLIT_P_ARRIVE == 0: commit(mbar_p_full[p_slot])
    else:                   commit(mbar_p_last_full[p_slot])
    # instruction_selection: tcgen05.wait::st.sync.aligned x2 (:2590) then
    #   mbarrier.arrive.release.cta.shared::cta.b64 (:2594); extent: scalar each.
    #   A TMEM-store fence precedes EACH commit -- the MMA warp must not read a
    #   P quarter whose stores are still in flight.

    acquire(mbar_sm_stats_empty[o_slot], phase)
    # instruction_selection: THE_WAIT (:2595);
    #   extent: one retry loop.  The stats pipe's EMPTY side is a real producer
    #   acquire; only its full side is never consumer-waited.
    # `_compute_row_sum` for arch 100 (utils.py:288-304): the same four-packed-
    # accumulator shape as the row max, seeded from the first eight elements.
    acc = reg_tile([4, 2], "f32")
    for a in range(4):
        acc[a] = (r_s[2*a], r_s[2*a + 1])       # seed, no adds
    for i in range(8, 128, 8):                  # 15 trips
        for a in range(4):
            add(acc[a], acc[a], r_s[i + 2*a : i + 2*a + 2], lanes=2)
    add(acc[0], acc[0], acc[1], lanes=2)
    add(acc[2], acc[2], acc[3], lanes=2)
    add(acc[0], acc[0], acc[2], lanes=2)
    row_sum = acc[0][0] + acc[0][1]
    # instruction_selection: add.rn.f32x2, 60 loop + 3 fold = 63 per warpgroup,
    #   126 whole-kernel (:2597); extent: a packed pair each, then one scalar
    #   add.f32 for the final horizontal.  The reduction runs over the EXP2'D
    #   fragment, which is why it sits after the convert and not before it.
    store_shared(sScale, slot * 256 + group_tidx, row_sum)
    store_shared(sScale, slot * 256 + 128 + group_tidx, row_max)
    if has_temperature:
        store_shared(sScaleTemp, slot * 128 + group_tidx, row_sum_t)
    # instruction_selection: st.shared.f32 (:2603,:2604, and :2611 for the
    #   temperature store); extent: scalar each.
    fence("view_async_shared")
    # instruction_selection: fence.proxy.async.shared::cta x2 (:2612); extent:
    #   scalar.  Publishes sScale to the epilogue's reader.
    # `:2614-2615` guards a `sm_stats_barrier.arrive_w_index` on
    # `signal_stats_barrier`, and `_wg_softmax` passes False (:2784-2786,
    # :2795-2797), so NOTHING is emitted here.  The export's only two `bar.arrive`
    # are the TMEM-allocator's `2, 288` at `.loc 1 295`.  The softmax-to-epilogue
    # stats edge rides `mbar_sm_stats` plus the `bar_epilogue` arrive-and-wait
    # between the two bodies, not a stats named barrier.
    release(mbar_s_empty[s_slot])
    # instruction_selection: mbarrier.arrive.release.cta.shared::cta.b64 (:2616);
    #   extent: scalar.  Frees the S slot the next QK reuses.

def epilogue_step(stage, qi_group):
    wait(mbar_o_full[o_slot], phase)
    # instruction_selection: THE_WAIT x2 (:2986);
    #   extent: one retry loop.  The PV of this group has landed in O.

    if group_tidx < q_tokens_per_group:
        word = load_shared(sQIdxMeta, meta_slot + group_tidx)
        store_shared(sQIdx,     slot * q_tokens_per_group + group_tidx, word & 0xFFFFFF)
        store_shared(sSplitIdx, slot * q_tokens_per_group + group_tidx, (word >> 24) & 0xFF)
        # instruction_selection: ld.shared.u32 then and.b32 / shr.u32 +
        #   st.shared.u32 (:2995-2998); extent: scalar each.
    named_barrier_arrive_and_wait(bar_epilogue + stage)
    # instruction_selection: bar.sync id=13 or 14, 128 threads (:2999); extent:
    #   warpgroup.

    # Two column passes over O, ROLLED (`unroll=1`), each reading its half of
    # the accumulator and storing it before the next pass runs.
    #
    # The read shape is 16x256b, not 32x32b, and that choice is what makes the
    # 128-bit stores below contiguous: 16x256b puts four columns of ONE row in
    # lanes 0-3, so a store covers eight rows in whole 64-byte runs.  Under
    # 32x32b the thread IS the row and every lane of a store lands on a
    # different row -- identical bytes at twice the sectors.
    for col_pass in range(2):                        # rolled
        copy_t2r(O[o_slot] col-half col_pass, r_o)
        # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x8.b32, TWO per
        #   pass (:3028) -- one per row half, reached through the LANE field of
        #   the TMEM address rather than the repetition suffix; extent: 32
        #   registers each.

        for r in range(num_rows):                    # fragment rows
            for cN in range(num_cols // lanes_per_store):
                row, col = frag_coord(r, cN)
                if row < M_BLOCK:
                    tok = row // qheadperkv
                    qi  = qi_group * q_tokens_per_group + tok
                    if qi < count_raw:
                        q_idx = load_shared(sQIdx,     slot * q_tokens_per_group + tok)
                        split = load_shared(sSplitIdx, slot * q_tokens_per_group + tok)
                        # instruction_selection: ld.shared.u32 x2 (:3055,:3056),
                        #   32 of each whole-kernel; extent: scalar each.  These
                        #   are the reads that consume the decode published at
                        #   the top of the epilogue: `split` picks the O_partial
                        #   slot and `q_idx` the query row, so the store address
                        #   is (split, q_batch_offset + q_idx, head).
                        row_sum = load_shared(sScale, slot * 256 + row)
                        safe    = select(row_sum == 0 or isnan(row_sum), 1.0, row_sum)
                        rcp(row_scale, safe)
                        # instruction_selection: rcp.approx.ftz.f32, 32 whole-
                        #   kernel, 16 per warpgroup (:3070), one per 128-bit
                        #   store; extent: scalar.  The zero/NaN guard is BEFORE
                        #   the reciprocal.  Its whole-kernel count follows the store
                        #   width: 32 for fp32, 16 for the 8-lane half paths,
                        #   8 for fp8.
                        mul(scaled, r_o[cols], row_scale, lanes=2)
                        # instruction_selection: mul.rn.f32x2, two per fp32 store
                        #   group (:3086,:3088); extent: the group.
                        copy_r2g(o_partial_ptr + fake_col(partial_dtype, col),
                                 scaled, cache="cs")
                        # instruction_selection: one 128-bit store per group,
                        #   but the CONVERTER in front of it is what the partial
                        #   dtype selects, and the two half formats differ:
                        #     fp32 -- st.global.cs.v4.f32, no convert (32
                        #       whole-kernel, 16 per warpgroup, :2866);
                        #     bf16 -- 8 x cvt.rn.bf16.f32 packed into
                        #       st.global.cs.v4.b32 (:2882, stg_128_bf16_cs);
                        #     fp16 -- 8 x cvt.rn.f16.f32 packed into
                        #       st.global.cs.v4.b32 (:2884, stg_128_f16_cs).
                        #       Measured on the fp16 export: 128 cvt.rn.f16.f32,
                        #       16 st.global.cs.v4.b32, `.loc 1 2884` 16 with
                        #       `.loc 1 2882` 0, against 32 st.global.cs.v4.f32
                        #       and 0 converts in the fp32 build.  The lane
                        #       count, the fake-column remap
                        #       (real_col_to_stg128_half_fake_col, :3160) and
                        #       the control flow are shared with bf16 -- only
                        #       the convert opcode is not;
                        #     fp8  -- sixteen packed bytes in st.global.cs.v4.b32.
                        #   extent: 128 bits.  The address goes through
                        #   real_col_to_stg128*_fake_col: O_partial is stored in
                        #   fake-column order, the combine kernel reads it back
                        #   that way, and the permutation is what makes these
                        #   stores whole-sector.
                        #   The `qi < count_raw` guard is what keeps a partial
                        #   final group from writing another query's rows.

    wait_tmem_ld()
    # instruction_selection: tcgen05.wait::ld.sync.aligned x2 (:3270); extent:
    #   scalar.  AFTER the whole store loop -- it releases the O TMEM slot, and
    #   it is the only TMEM-load wait in the kernel.

    # LSE: one row per thread, and the WHOLE block is guarded (:3275) exactly as
    # the O-store loop above is -- without it a partial final Q group writes
    # another query's LSE rows.
    tok_local = group_tidx // qheadperkv
    h_local   = group_tidx - tok_local * qheadperkv
    qi_lse    = qi_group * q_tokens_per_group + tok_local
    if qi_lse < count_raw:
        # instruction_selection: setp.ge.s32 (:3275) branching over the whole
        #   block; extent: scalar.  A live predicate, one per warpgroup.
        row_sum = load_shared(sScale, slot * 256 + group_tidx)
        row_max = load_shared(sScale, slot * 256 + 128 + group_tidx)
        log2(lg, row_sum)
        # instruction_selection: lg2.approx.ftz.f32 (:3284), 2 whole-kernel
        #   without temperature and 4 with it -- the temperature arm takes its
        #   own log2 of its own row sum; extent: scalar.
        lse = select(row_sum == 0 or isnan(row_sum), -inf,
                     (row_max * softmax_scale_log2 + lg) * ln2)
        q_idx_lse = load_shared(sQIdx,     slot * q_tokens_per_group + tok_local)
        split_lse = load_shared(sSplitIdx, slot * q_tokens_per_group + tok_local)
        # instruction_selection: ld.shared.u32 (:3286) and ld.shared.s32 (:3288);
        #   extent: scalar each.  This is where the sQIdx / sSplitIdx pair
        #   published at the top of the epilogue is consumed -- the split index
        #   picks the O_partial/LSE_partial slot this CTA owns.
        h_abs = head_kv_idx * qheadperkv + h_local
        store_global(mLSE_partial, (split_lse, q_batch_offset + q_idx_lse, h_abs), lse)
        # instruction_selection: st.global.f32 x2 (:3290); extent: scalar.
        #   `softmax_scale_log2` is the host-computed `.param .f32`.
        if has_temperature:
            row_sum_t = load_shared(sScaleTemp, slot * 128 + group_tidx)
            log2(lg_t, row_sum_t)
            store_global(mLSE_temperature_partial,
                         (split_lse, q_batch_offset + q_idx_lse, h_abs),
                         select(row_sum_t == 0 or isnan(row_sum_t), -inf,
                                (row_max * lse_temperature_scale_log2 + lg_t) * ln2))
            # instruction_selection: lg2.approx.ftz.f32 + st.global.f32; extent:
            #   scalar each.  It reuses `row_max`, not a second maximum, and it
            #   carries its own zero/NaN guard.

    named_barrier_arrive_and_wait(bar_epilogue + stage)
    # instruction_selection: bar.sync id=13 or 14, 128 threads (:3302); extent:
    #   warpgroup.  The third and last epilogue barrier for this group.
    release(mbar_sm_stats_empty[o_slot])
    release(mbar_o_empty[o_slot])
    # instruction_selection: mbarrier.arrive.release.cta.shared::cta.b64 x2 each
    #   (:3304,:3305); extent: scalar each.  Returning the O slot is what lets
    #   the MMA warp start the next PV.  The stats release arrives byte offset
    #   240 -- the SAME address the producer acquire at :2595 waits on, i.e. the
    #   empty barrier.  The full pair at 224 is initialized in the prologue and
    #   then never named again.
```

## Logical GEMM ownership

| GEMM | A | B | D | owner | kind | chain |
| --- | --- | --- | --- | --- | --- | --- |
| QK | `sQ[stage]` (SMEM, K-major) | `sK` (SMEM, K-major) | `S[slot]` (TMEM f32) | warp 12 | `mma_kind` | 8 x mma_k=16 (BF16 Q) or 4 x mma_k=32 (FP8 Q) |
| PV | `P[slot]` (TMEM, overlays S) | `sV` (SMEM, MN-major) | `O[slot]` (TMEM f32) | warp 12 | `mma_kind` | same length, split 3/4 + 1/4 around `mbar_p_lastsplit` |

Both GEMMs take the **same** kind, because both operands were dequantized to
Q's dtype. The sibling can mix kinds; this kernel cannot. Whole-kernel counts:
48 `kind::f16` (BF16 Q) or 24 `kind::f8f6f4` (FP8 Q).

## TensorMap ABI

Each build carries **three** `.param .align 64 .b8 [...128]` descriptors: K, V,
and exactly one of TMA-Q / gather4-Q.

| descriptor | rank | element | box | swizzle | policy | notes |
| --- | --- | --- | --- | --- | --- | --- |
| K, flat | 3 | uint8 | `(PACKED_HEAD_DIM, 1, N_BLOCK)` | 128 B | zero | 8192 transaction bytes -- half the sibling's |
| V, flat | 3 | uint8 | `(PACKED_HEAD_DIM, 1, N_BLOCK)` | 128 B | zero | MN-major source permute done host-side |
| K, **paged** | **4** | uint8 | `(PACKED_HEAD_DIM, N_BLOCK, 1, 1)` | 128 B | zero | global dims fastest-first `(PACKED_HEAD_DIM, page_size, head_kv, num_pages)` from the `[2,3,1,0]` host permute (`:373-379`); coordinate tuple `(0, 0, head_kv, page)`; same 8192 transaction bytes and same `.tile` / `mbarrier::complete_tx::bytes` / `L2::cache_hint` as flat |
| V, **paged** | **4** | uint8 | `(PACKED_HEAD_DIM, N_BLOCK, 1, 1)` | 128 B | zero | same dims, with the extra `[1,0,2,3]` MN-major permute; coordinate tuple `(0, 0, head_kv, page)` |
| Q (TMA program) | 2 | q_dtype | `(swizzle_elems, qheadperkv)` | 128 B | zero | present only for qheadperkv 8/16 |
| Q (gather4 program) | 2 | q_dtype | `(box_x, 1)` | 128 B | EVICT_LAST | `box_x` = 128 for FP8 Q, 64 for BF16 Q; the instruction supplies four row coordinates |

The port encodes each in the PrimFunc host prologue via
`runtime.cuTensorMapEncodeTiled`, rather than taking the source's prebuilt
`uint8[128]` descriptor tensor as a kernel argument. Sanctioned ABI deviation,
same as the sibling.

## Storage aliases and lifetimes

| alias | backing | live |
| --- | --- | --- |
| `sQ_load` | `sQ` | the TMA Q program's indexing view of the same bytes; the gather4 program addresses `sQ` directly (:978), so this alias is the TMA program's only |
| `P` | `S` TMEM stage starting at column `tmem_s_to_p` (64 for BF16 Q, 96 for FP8) | from the softmax `copy_r2t` to the PV chain that consumes it |
| `sKFp4` / `sVFp4` | own storage | from the TMA to the end of the dequant pass; dead for the rest of the kernel |
| `sK` / `sV` | own storage | from the dequant pass to the last MMA of the CTA |

`sKFp4`/`sVFp4` are the only tiles whose lifetime ends early. They are not
aliased onto `sK`/`sV`: the dequant reads the packed tile while writing the
unpacked one, so the two must coexist.

## Static specialization boundary

Resolved at trace time, so each appears as straight-line code rather than a
branch: `qheadperkv` and the Q-load program it selects, `q_dtype` and with it
the dequant program / `mma_kind` / `tmem_s_to_p` / gather4 box width and k-tile
split, `partial_dtype` and its store width, `has_temperature`, `has_k_global`,
`causal`, `paged_kv`, `has_seqused_k`.

The last three are `const_expr` in the source exactly like the others, and each
deletes code rather than selecting between equal-cost arms: `causal=False`
removes the diagonal search, the `mCuSeqlensQ[batch+1]` load, the whole
causal-mask arm and the exp2 emulation; `paged_kv` swaps the KV descriptors to
rank 4, pins `k_batch_offset` to 0 and changes the block-scale row;
`has_seqused_k` replaces the paged-capacity multiply with one `ld.global.u32`
of `mSeqUsedK[batch]`.

Runtime branches that survive: `cta_valid_work`, `has_work`, `qi < count_raw`,
`qi_lse < count_raw` (the LSE block's own guard), `row < M_BLOCK`,
`need_causal_mask` (**causal builds only** -- it does not survive a
non-causal specialization), `col_limit < N_BLOCK` (the mask body runs only on a partly
visible tile), `group_tidx < q_tokens_per_group` (the metadata publish and the
epilogue decode), `tok_idx < q_tokens_per_group` (the TMA per-warp partition),
`num_q_groups > 1`, the 32-trip search's `left < right` predicate, and the
`num_q_groups` loop bounds.

## TIRx module and benchmark contract

`KERNEL_META["name"] = "msa_sparse_atten_fwd_nvfp4_kv_sm100"`, category `msa`,
`runtime_cuda_archs = ["sm_100a"]`.

`prepare_data` builds packed FP4 K/V, random E4M3 scale bytes in a positive
finite band, the CSR payload, the work list and a frozen qsplit assignment, and
keeps dequantized BF16 twins. `run_test` checks the partials **bitwise**
(`rtol=atol=0`) against the compiled MSA source kernel on identical frozen
inputs, masked to the live split slots; a secondary check on the BF16-Q configs
runs the already-ported sibling on the dequantized twins, which catches
data-plumbing errors the bitwise check cannot see because both sides would
consume the same wrong bytes.

`run_bench` times one forward launch against MSA's compiled NVFP4 forward.
Nothing is rotated: the kernel reads its inputs without touching them and
overwrites the partial slots it owns.

## Instruction-selection summary

Placement, layout, shape and schedule -- not arithmetic -- select nearly every
instruction in this kernel:

- **Packing selects the TMA transaction size.** K/V descriptors are `uint8`
  with a half-width head-dim extent, so each tile is 8192 bytes against the
  sibling's 16384. The `expect_tx` counts in the prologue must match, and they
  are set once because KV is single-shot.
- **The toolchain version selects the dequant chain.** The pinned CuTe-DSL
  reports CUDA 12.9, so both helpers take their fallbacks: BF16 goes
  `cvt.rn.f16x2.e2m1x2` x8 per tile then the f16->bf16 route then
  `mul.rn.bf16x2`; FP8 goes `prmt` + `cvt.rn.f16x2.e4m3x2` +
  `cvt.rn.f16x2.e2m1x2` + `mul.rn.f16x2` + `cvt.rn.satfinite.e4m3x2.f16x2`. The
  single-instruction forms (`cvt.rn.bf16x2.e2m1x2`, `mul.e4m3x4.e2m1x4`) have
  count **zero** in every export.
- **Q's dtype selects where the K tensor scale lands**, and the export counts
  it: `mul.rn.bf16x2` is 17 with `has_k_global_scale` and 16 without on BF16 Q,
  while FP8 Q instead carries `mul.rn.f32x2` x128 on the S accumulator and zero
  when the scale is absent.
- **Q's dtype also selects the scale loader's extension.** BF16 uses
  `ld.global.u8`, FP8 `ld.global.s8` -- the operand-form trap, visible only in
  the export.
- **The TMEM load shape selects store coalescing.** `16x256b` in the epilogue
  puts four columns of one row in lanes 0-3; `32x32b` would make the thread the
  row and double the sectors for identical bytes.
- **Only the gather4 Q path sets an L2 policy.** It takes EVICT_LAST, which
  suits a row every topK CTA re-reads. The KV loads and the TMA-Q loads carry
  the `.L2::cache_hint` qualifier with a ZERO policy operand, because they go
  through quack's copy helper rather than MSA's own cached one. The sibling
  kernel routes its KV loads differently and does set a policy there, so this
  is a place where copying the sibling would diverge from this reference.
- **Q's dtype selects the gather4 body's shape.** BF16 splits each row into two
  `K_TILE` halves and prefetches the second; FP8 issues one full-row copy and no
  prefetch at all.
- **The mask relies on `shr.u32` clamping** shifts >= 32 to zero, and lowers to
  `and.b32` + `setp.eq.s32` + `selp.b32` per element -- the densest instruction
  group in the kernel.
- **Three-input `max.f32`** halves the row-max tree to 66 instructions.
- **`ex2_emu_freq = 16` counts elements**, giving 112 MUFU `ex2.approx.ftz.f32`
  and 8 pair-polynomial evaluations, covering 16 elements, per 128-element row.
- **The softmax and epilogue bodies are emitted twice**, once per warpgroup,
  from a compile-time `stage`.
