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
python/fmha_sm100/cute/src/sm100/fwd/atten_fwd.py
SparseAttentionForwardSm100.
-->

# msa_sparse_atten_fwd_sm100: coarse WASP pipeline sketch

This non-executable design sketch describes the storage layout, warp roles,
pipelines, control flow, and PTX-level operations of
[`tirx_kernels/msa/sparse_atten_fwd.py`](../../../tirx_kernels/msa/sparse_atten_fwd.py).
That TIRx module is the authoritative implementation.

The port covers **one kernel**: `SparseAttentionForwardSm100.kernel`
(`atten_fwd.py:637-1235`) with its trace-time launcher `__call__` (`:292-631`)
and every warp-role helper it dispatches to. This is the attention kernel
itself: it consumes the work list produced by
`SparseAttentionPrepareFlatScheduleSm100` and the packed `q_idx | slot<<24`
metadata produced by `SparseAttentionPrepareFwdSplitAtomicSm100` -- both
already ported into this package -- and writes the split partials that the
combine kernel (a separate class, separate compile key, **out of scope**)
reduces.

## Scope and instantiations

The upstream compile key is a 16-tuple (`interface.py:1714-1731`). Fixed for
every specialization in this port:

| axis | value | why it is fixed |
| --- | --- | --- |
| `head_dim` | 128 | `__init__` raises otherwise (`:75-79`) |
| `n_block_size` | 128 | the KV block width the schedule is built around |
| `m_block_size` | 128 | 128 packed Q heads per MMA tile |
| `causal` | `True` | every upstream test and benchmark sets it; it is a real codegen axis (`:173-181`) |
| `paged_kv` | `False` | out of scope, see below |
| `use_prepare_scheduler` | `True` | `__init__` raises otherwise (`:96-98`) |

In scope, i.e. the specializations this module compiles:

| axis | values | what changes in device code |
| --- | --- | --- |
| `qheadperkv` | 1, 2, 4 / 8, 16 | selects **the whole Q-load warpgroup program**: `use_q_gather4` (`:81`) picks a raw gather4 descriptor path for 1/2/4 and a CUTE-managed TMA path for 8/16. Also sets `q_tokens_per_group = 128 // qheadperkv` (`:108`) and `tokens_per_gather4 = 4 // qheadperkv` (`:87`). |
| dtype combination | `bf16`, `fp8`, `bf16q_fp8kv`, `fp8_pvbf16` | resolved at trace time (`:318-364`) into `qk_dtype`/`pv_dtype`, the two `mma_kind` strings, and `k_fp8_to_bf16`/`v_fp8_to_bf16`, which add a shared-memory dequantization pass to the softmax warpgroups |
| `partial_dtype` | fp32, bf16, fp8 e4m3 | three distinct epilogue store paths: 4-lane, 8-lane, 16-lane 128-bit stores with three different swizzle-inverse column remaps (`:2772-2984`) |
| temperature LSE | on / off | `mLSE_temperature_partial is not None` adds one scaled row-sum reduction, one `sScaleTemperature` publish and one extra LSE store |

Out of scope, with the predicate that excludes each:

- **paged KV** -- `page_table is not None` at the host entry; requires
  `page_size == blk_kv` (`:99-105`) and swaps in a rank-4 descriptor plus
  `PagedKVManager` block indirection.
- **`seqused_k`** -- `seqused_k is not None`; replaces `_logical_seqlen_k`'s
  `cu_seqlens_k` difference with a per-batch lookup (`:208-212`).
- **`causal=False`** -- never exercised upstream; it would change
  `num_regs_softmax` 176 -> 192, `num_regs_store` 112 -> 80, and
  `ex2_emu_freq` 16 -> 0, i.e. remove the polynomial exp2 mixing entirely.
- **fp16 partials** -- accepted by the dtype check at `:372-377` but never
  exercised upstream; it shares the bf16 8-lane store path.
- **the NVFP4-KV sibling** `SparseAttentionForwardNvfp4KvSm100`
  (`atten_fwd_nvfp4_kv.py`) -- a separate class with its own compile key.
- **the combine kernel** `fwd/combine.py` -- a downstream consumer.
- Tile (`Tx`) primitives are out of scope everywhere.

## The line-info export this sketch is annotated from

Every `instruction_selection` annotation below is read out of a line-info PTX
export, not out of the source text. Six exports, one per structurally distinct
in-scope compile key, are preserved under
`.porting/sparse_atten_fwd/ptx_lineinfo/<name>/` together with the
`analysis.txt` each was summarized into; `.porting/sparse_atten_fwd/export_findings.md`
records the `.loc` file-id table and the eight places where the export
contradicts the source text. Unqualified counts below are from
`bf16_tmaq_qh16_fp32`. Counting convention: instruction lines minus predicated
lines.

Reproduce any export with:

```bash
mkdir -p .porting/sparse_atten_fwd/ptx_lineinfo/<name>
MM_SPARSE_ATTN_AOT_DISABLE=1 CUTE_DSL_NO_CACHE=1 CUTE_DSL_KEEP=ptx \
CUTE_DSL_LINEINFO=1 CUTE_DSL_DUMP_DIR=.porting/sparse_atten_fwd/ptx_lineinfo/<name> \
python .porting/sparse_atten_fwd/export_driver.py <name>
```

The reference only compiles under a pinned CuTe-DSL (4.5.3 + `quack-kernels`
0.5.0, installed at `.reference-deps/msa-cutedsl`). The ambient 4.6.0.dev0
fails NVVM serialization on every gather4 specialization while compiling the
TMA-Q ones fine.

## Pipeline at a glance

16 warps, 512 threads, one CTA per work item. Role selection is a flat sequence
of independent `if` blocks -- **not** an `if/elif` chain -- each ANDed with
`cta_valid_work`. A warp falls through the blocks it does not match and returns.

| Warps | Role-local tile program | Main publication / reuse edges |
| --- | --- | --- |
| 0..3 | softmax warpgroup 0 **with the epilogue fused in**; on `k_fp8_to_bf16` it first dequantizes the whole K tile from `sKFp8` into `sK`. Takes the **even** Q groups. | consumes `mbar_s`; publishes `mbar_p` + `mbar_p_lastsplit` and `sScale`; consumes `mbar_o`; releases `mbar_sm_stats` |
| 4..7 | softmax warpgroup 1, same body, **odd** Q groups; on `v_fp8_to_bf16` it dequantizes V into `sV` first | same edges, other slot parity |
| 8..11 | Q-load warpgroup: publishes the packed `sQIdxMeta` ring, then issues either 8 gather4 TMAs per warp per group, or one/two plain TMAs per token | produces `mbar_q`; publishes `sQIdxMeta` (read later by both softmax WGs *and* the epilogue) |
| 12 | the single MMA-issue warp; also the TMEM allocator warp | waits `mbar_k`/`mbar_v`, consumes `mbar_q`, produces `mbar_s` and `mbar_o`, consumes `mbar_p`/`mbar_p_lastsplit` |
| 13 | K load: one TMA for this CTA's single KV block | produces `mbar_k` (or `mbar_k_tma` on the fp8 path) |
| 14 | V load: one TMA for the same block | produces `mbar_v` (or `mbar_v_tma`) |
| 15 | idle; executes only `setmaxnreg.dec 48` and falls out | none |

**KV is not pipelined.** One CTA owns exactly one KV block, so `mbar_k` and
`mbar_v` are single-shot barriers with `expect_tx` set once by thread 0 in the
prologue, and the load warps issue one TMA each and retire. All pipelining runs
along the **Q-group axis**: `num_q_groups = ceil(count_raw / q_tokens_per_group)`,
with `q_stage = 2`, `s_stage = 2`, `o_stage = 2` and a 16-deep `sQIdxMeta` ring.

**There is no online rescale.** Because a Q group sees exactly one KV block,
every softmax step is the first-and-only step: one row max, one exponentiation,
one row sum, no correction loop, no running accumulator rescale
(`:2284-2287`).

**The epilogue is fused into the softmax warpgroups**, and the export shows the
whole softmax+epilogue body is **emitted twice**, once per warpgroup, not
shared behind a runtime `stage`.

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
cast(dst, src)                       # includes fp8 -> bf16 and f32 -> narrow
add(dst, lhs, rhs, lanes=1)          # lanes=2 is one packed f32x2 op
sub(dst, lhs, rhs, lanes=1)
mul(dst, lhs, rhs, lanes=1)
fma(dst, a, b, c, lanes=1)
max(dst, lhs, rhs)
exp2(dst, src)                       # MUFU
exp2_poly(dst, src)                  # degree-3 FFMA emulation
log2(dst, src)
rcp(dst, src)
select(dst, predicate, a, b)
shuffle_index(dst, src, lane, mask, clamp)
gemm(dst, lhs, rhs, accumulate=False)
```

Schedule operations: `pipe`, `init_pipe`, `acquire`, `wait`, `commit`,
`release`, `expect_tx`, `fence`, `barrier`, `named_barrier`, `elect`,
`set_register_budget`, TMEM `allocate`/`relinquish`/`free`, and cursor
(`slot`, `phase`) updates.

`add(..., lanes=2)` is one packed two-lane f32 operation with two ordered
results, not shorthand for two scalar adds. There are deliberately no
primitives named `attention`, `softmax`, `mask`, `online_update`, `TMA`,
`TCGEN05`, `dequantize`, or `epilogue`.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

@specialize(
    HEAD_DIM=128,
    M_BLOCK=128,               # 128 packed Q heads per MMA tile
    N_BLOCK=128,               # KV block width
    K_TILE=64,                 # UTCMMA bf16 K-tile (:59)
    QHEADPERKV=(1, 2, 4, 8, 16),
    USE_Q_GATHER4=QHEADPERKV in (1, 2, 4),          # :81
    Q_TOKENS_PER_GROUP=128 // QHEADPERKV,           # :108
    TOKENS_PER_GATHER4=(4 // QHEADPERKV) if USE_Q_GATHER4 else 0,   # :87
    DTYPE_MODE=("bf16", "fp8", "bf16q_fp8kv", "fp8_pvbf16"),
    PARTIAL_DTYPE=("f32", "bf16", "fp8e4m3"),
    RETURN_TEMPERATURE_LSE=(False, True),
    CAUSAL=True,
    PAGED_KV=False,
    HAS_SEQUSED_K=False,
    Q_STAGE=2, S_STAGE=2, O_STAGE=2, KV_STAGE=1,    # :116-119
    K_STAGES=2,                                     # Q k-subtiles, :125
    QIDX_META_STAGES=16,                            # :123
    SPLIT_P_ARRIVE=96,                              # 128//4*3, floored to 32, :165-167
    EX2_EMU_FREQ=16,                                # causal only, :180
    EX2_EMU_START_FRG=1,                            # :181
    NUM_REGS_SOFTMAX=176, NUM_REGS_STORE=112, NUM_REGS_OTHER=48,   # :173-178
    CTA_GROUP=1, CLUSTER=(1, 1, 1),                 # :184-188
    target="sm_100a",
)
# instruction_selection: none; extent: compile-time constants only.
#   QK/PV operand dtypes and the two MMA kind strings are derived, not free:
#   qk_dtype = q_dtype unless overridden; pv_dtype = bf16 for the legacy
#   bf16-Q/fp8-KV case, else v_dtype (:321-335). mma_kind is "f8f6f4" when the
#   operand width is 8, else "f16" (:363-364).
K_FP8_TO_BF16 = (k_storage is fp8) and (qk_dtype is bf16)    # :354-357
V_FP8_TO_BF16 = (v_storage is fp8) and (pv_dtype is bf16)    # :358-361

@kernel(
    grid=(work_capacity, 1, 1),        # :611, :626 -- a RUNTIME extent
    block=(512, 1, 1),                 # 16 warps, :143
    num_warps=16,
    cluster=(1, 1, 1),                 # cta_group::1; no cluster launch attribute
    dynamic_smem_bytes=max(SharedStorage.size_in_bytes(), 49152),   # :628
    tmem_columns=512,                  # :161, power-of-two rounded
    min_blocks_per_sm=1,               # :630
    target="sm_100a",
)
def msa_sparse_atten_fwd_sm100(
    k,                    # bf16|fp8 [total_k, head_kv, 128], input
    v,                    # bf16|fp8 [total_k, head_kv, 128], input
    k2q_q_indices,        # i32 [head_kv, nnz], input; q ascending within a CSR row, -1 past nnz
    k2q_qsplit_indices,   # i32 [head_kv, nnz], input; q_idx | (slot << 24)
    k2q_row_ptr,          # i32 [head_kv, total_rows + 1], input
    scheduler_metadata,   # i32 [work_capacity, 6], input; all six columns read
    work_count,           # i32 [1], input; the early-out bound
    o_partial,            # f32|bf16|fp8 [topk*total_q*head_q, 128], OUTPUT, uninitialized
    lse_partial,          # f32 [topk, total_q, head_q], OUTPUT, uninitialized
    lse_temperature_partial,   # f32 [topk, total_q, head_q], OUTPUT, optional
    q_flat,               # bf16|fp8 [total_q*head_q, 128], input
    cu_seqlens_q,         # i32 [num_batches + 1], input
    cu_seqlens_k,         # i32 [num_batches + 1], input
    softmax_scale,        # f32; the kernel forms softmax_scale_log2 at trace time (:514)
    lse_temperature_inv_scale,   # f32; 1 / lse_temperature_scale
    num_kv_blocks,        # i32 == total_rows -- DEAD inside the kernel (:681)
    num_heads_kv,         # i32
    seq_len_q,            # i32 == max_seqlen_q
    work_capacity,        # i32 == grid.x
    total_k, total_q, head_q, nnz, total_rows, num_batches, topk,   # i32 extents
):
    # Upstream additionally passes `mQ_gather4_desc`, a host-built uint8[128]
    # raw CUtensorMap (`interface.py:1682-1688`). The port DROPS that ABI
    # argument and encodes the identical descriptor in its own launcher
    # prologue -- see "TensorMap ABI" below. This is the sketch's one
    # deliberate ABI deviation.

    tid = thread_id()
    warp = warp_uniform(tid // 32)
    # instruction_selection: shfl.sync.idx.b32 (:706, 1 static); extent: warp
    #   broadcast.  `make_warp_uniform` is a real instruction, not a no-op: it
    #   is what lets ptxas treat every role predicate below as warp-uniform.
    #   It is emitted at PTX 113, BEFORE the three descriptor prefetches and
    #   long before either prologue barrier.
    lane = tid % 32
    block = block_id_x()

    # -----------------------------------------------------------------------
    # Work-item decode and the CTA-level early-out
    # -----------------------------------------------------------------------
    work_idx = block
    cta_valid_work = work_idx < load_global(work_count, 0)
    # instruction_selection: ld.global.u32 (atten_fwd.py:698, 1 static);
    #   extent: scalar, executed by every thread.
    # The grid is sized by the work list's CAPACITY, not its length --
    # work_count lives on the device and is never read back -- so the launch is
    # deliberately oversized and the tail CTAs fall straight through.

    head_kv_idx = row_linear = work_q_begin = work_q_count = i32(0)
    batch_idx = kv_block_idx = i32(0)
    if cta_valid_work:
        head_kv_idx  = load_global(scheduler_metadata, (work_idx, 0))
        row_linear   = load_global(scheduler_metadata, (work_idx, 1))
        work_q_begin = load_global(scheduler_metadata, (work_idx, 2))
        work_q_count = load_global(scheduler_metadata, (work_idx, 3))
        batch_idx    = load_global(scheduler_metadata, (work_idx, 4))
        kv_block_idx = load_global(scheduler_metadata, (work_idx, 5))
    # instruction_selection: ld.global.u32 x5 (:700, :702, :703, :704, :705)
    #   plus ld.global.**s32** for `row_linear` (:701); extent: scalar, all
    #   threads.  That signed load is the module's only one, and it is the same
    #   operand-form point the epilogue's :3003 makes: the width and signedness
    #   follow how the value is consumed downstream, not how it is declared.
    #   All six columns are consumed, unlike the split-atomic sibling which
    #   reads only 0..4.

    # =======================================================================
    # GMEM tiles
    # =======================================================================

    K = tile("gmem", k, KV_DTYPE, [total_k, num_heads_kv, 128], alignment=16)
    V = tile("gmem", v, KV_DTYPE, [total_k, num_heads_kv, 128], alignment=16)
    Qf = tile("gmem", q_flat, Q_DTYPE, [total_q * head_q, 128], alignment=16)
    # The K/V descriptors see permuted views, built at trace time (:378-387):
    #   K: [total_k, h, d] -> [total_k, d, h]      (K-major B operand)
    #   V: [total_k, h, d] -> [d, total_k, h]      (MN-major B operand)
    K_desc_view = view(K, KV_DTYPE, [total_k, 128, num_heads_kv])
    V_desc_view = view(V, KV_DTYPE, [128, total_k, num_heads_kv])

    Idx   = tile("gmem", k2q_q_indices,      "i32", [num_heads_kv, nnz])
    QSpl  = tile("gmem", k2q_qsplit_indices, "i32", [num_heads_kv, nnz])
    RowPtr= tile("gmem", k2q_row_ptr,        "i32", [num_heads_kv, total_rows + 1])
    Op    = tile("gmem", o_partial,   PARTIAL_DTYPE, [topk * total_q * head_q, 128])
    Lp    = tile("gmem", lse_partial, "f32", [topk, total_q, head_q])
    LpT   = tile("gmem", lse_temperature_partial, "f32", [topk, total_q, head_q])
    CuQ   = tile("gmem", cu_seqlens_q, "i32", [num_batches + 1])
    CuK   = tile("gmem", cu_seqlens_k, "i32", [num_batches + 1])

    # =======================================================================
    # Shared memory: one dynamic pool.  Sizes for BF16 / QHEADPERKV=16.
    #
    # The source declares this as `@cute.struct SharedStorage` and allocates it
    # with `cutlass.utils.SmemAllocator` (:734-735), which reads like a static
    # __shared__ block.  It is not one: every export carries
    #   .extern .shared .align 1024 .b8 __dynamic_shmem__0[]
    # The matching TIRx form is T.SMEMPool() + pool.commit() under the
    # `tirx.use_dyn_shared_memory` launch tag, never T.alloc_shared.
    #
    # Measured SharedStorage.size_in_bytes() per in-scope specialization:
    #   bf16 TMA-Q  qheadperkv=16 -> 135168     <- the sizes below
    #   bf16 gather4 qheadperkv=4 -> 138240
    #   bf16 gather4 qheadperkv=1 -> 146432
    #   bf16 Q + fp8 KV, gather4, qheadperkv=4 -> 171008  (+16384 +16384 staging)
    #   all-fp8 TMA-Q qheadperkv=16 -> 69632
    #   fp8 pv=bf16 TMA-Q qheadperkv=8 -> 103424
    # The metadata half scales with Q_TOKENS_PER_GROUP = 128 // QHEADPERKV, so
    # the footprint GROWS as the head group shrinks.  The 49152 floor at :628
    # never binds.
    # =======================================================================

    smem = tile("smem", "u8", [dynamic_smem_bytes], byte_offset=0, alignment=1024)

    # --- mbarrier words (:537-548) ---------------------------------------
    mbar_k        = view(smem, "i64", [2])      # single-shot; only slot 0 used
    mbar_v        = view(smem, "i64", [2])
    mbar_k_tma    = view(smem, "i64", [2])      # present only if K_FP8_TO_BF16
    mbar_v_tma    = view(smem, "i64", [2])      # present only if V_FP8_TO_BF16
    mbar_q        = view(smem, "i64", [Q_STAGE, 2])     # full/empty
    mbar_s        = view(smem, "i64", [S_STAGE, 2])
    mbar_p        = view(smem, "i64", [S_STAGE, 2])
    mbar_p_last   = view(smem, "i64", [S_STAGE, 2])
    mbar_o        = view(smem, "i64", [O_STAGE, 2])
    mbar_sm_stats = view(smem, "i64", [O_STAGE, 2])
    tmem_dealloc_mbar = view(smem, "i64", [1])
    tmem_holding      = view(smem, "i32", [1])

    # --- scalar metadata (:554-587) --------------------------------------
    sScale       = view(smem, "f32", [O_STAGE, 256])
    #   slot s: [0:128) row_sum per M row, [128:256) row_max per M row
    sScaleTemp   = view(smem, "f32", [O_STAGE, 128])          # temperature only
    sSplitIdx    = view(smem, "i32", [O_STAGE, Q_TOKENS_PER_GROUP])
    sQIdx        = view(smem, "i32", [O_STAGE, Q_TOKENS_PER_GROUP])
    sDiagQCount  = view(smem, "i32", [1])
    sRowMeta     = view(smem, "i32", [8])
    #   0 batch, 1 kv_block, 2 row_start, 3 count_raw, 4 kv_valid_cols,
    #   5 q_batch_off, 6 k_batch_off, 7 causal_q_offset
    sPagedKvIdx  = view(smem, "i32", [1])                    # unused, PAGED_KV=False
    sQLoadMIdx   = view(smem, "i32", [Q_STAGE, Q_TOKENS_PER_GROUP])
    #   Allocated unconditionally (:582-583) even though only the TMA-Q body
    #   reads it; its bytes are part of every footprint above.
    sQIdxMeta    = view(smem, "i32", [QIDX_META_STAGES, Q_TOKENS_PER_GROUP])
    #   The packed qsplit ring.  Deliberately deeper (16) than the in-flight
    #   group distance so the epilogue can re-read q_idx/split without touching
    #   global memory again (:120-123).

    # --- the big tiles, 1024-byte aligned (:588-608) ----------------------
    sK = view(smem, MMA_KV_DTYPE, [128, 128], alignment=1024,
              layout="SM100-B-K-major", lifetime="one KV block, whole CTA life")
    sV = view(smem, MMA_KV_DTYPE, [128, 128], alignment=1024,
              layout="SM100-B-MN-major", lifetime="one KV block, whole CTA life")
    sKFp8 = view(smem, "fp8e4m3", [128, 128], alignment=1024)   # K_FP8_TO_BF16 only
    sVFp8 = view(smem, "fp8e4m3", [128, 128], alignment=1024)   # V_FP8_TO_BF16 only
    sQ = view(smem, Q_DTYPE, [Q_STAGE, 128, 128], alignment=1024,
              layout="SM100-A-K-major", lifetime="Q load until MMA release")
    sQ_load = alias(sQ, Q_DTYPE,
                    [Q_STAGE * Q_TOKENS_PER_GROUP * (128 // Q_LOAD_TILE),
                     QHEADPERKV, Q_LOAD_TILE],
                    layout="per-token sub-tile view of the same bytes",
                    lifetime="identical to sQ")
    #   Q_LOAD_TILE = 128 for fp8 Q, else K_TILE=64 (:427-429).
    #   sQ and sQ_load are THE SAME BYTES with two layouts (:744-745): the MMA
    #   sees one monolithic A tile, the loader sees q_tokens x k_subtiles boxes.

    # =======================================================================
    # TMEM: 512 columns.  P OVERLAYS the upper part of each S tile.
    # =======================================================================

    S = tile("tmem", "f32", [S_STAGE, 128, 128], base_col=0, columns=256)
    #   S0 -> cols [0,128), S1 -> cols [128,256); stage stride = N_BLOCK = 128.
    O = tile("tmem", "f32", [O_STAGE, 128, 128], base_col=256, columns=256)
    #   O0 -> cols [256,384), O1 -> cols [384,512); stage stride = 128.
    TMEM_S_TO_P = 128 - 128 * P_DTYPE_WIDTH // 32      # 64 for bf16 P, 96 for fp8 P (:369-371)
    P = alias(S, P_DTYPE, [S_STAGE, 128, 128],
              base_col=TMEM_S_TO_P,
              lifetime="written after row_max/row_sum are extracted from the "
                       "same columns; the tail of S is destroyed by the P store "
                       "and that is safe only because of that ordering")
    assert tmem_column_end(O) == 512

    # =======================================================================
    # Pipes.  Slot/phase arithmetic is explicit everywhere in this kernel:
    #   slot = i % stages;  phase = (i // stages) & 1;  producer phase = phase ^ 1
    # =======================================================================

    q_pipe = pipe("q", stages=Q_STAGE, words=mbar_q,
                  producers=1, consumers=1, tx_count=q_tma_bytes,
                  kind="tma->umma")
    #   expect_tx is set at PRODUCER ACQUIRE, and the several sub-tile TMAs the
    #   Q-load warpgroup issues for that stage sum to exactly q_tma_bytes.
    s_pipe = pipe("s", stages=S_STAGE, words=mbar_s,
                  producers=1, consumers=128, kind="umma->async")
    p_pipe = pipe("p", stages=S_STAGE, words=mbar_p,
                  producers=128, consumers=1, kind="async->umma")
    p_last_pipe = pipe("p_last", stages=S_STAGE, words=mbar_p_last,
                       producers=128, consumers=1, kind="async->umma")
    #   p_pipe carries the FIRST 3/4 of P; p_last_pipe carries the last quarter
    #   and is consumed from INSIDE the PV MMA instruction sequence.
    o_pipe = pipe("o", stages=O_STAGE, words=mbar_o,
                  producers=1, consumers=128, kind="umma->async")
    sm_stats_pipe = pipe("sm_stats", stages=O_STAGE, words=mbar_sm_stats,
                         producers=128, consumers=128, kind="async")
    #   Only the EMPTY half of sm_stats is live: the softmax warpgroup acquires
    #   it (:2331) and the epilogue releases it (:3019); nothing ever commits or
    #   waits on its full half.  Its sole job is to stop softmax from
    #   overwriting sScale_slot before the epilogue two groups back has drained
    #   it.  The port keeps the pipe object and both halves' storage, because
    #   removing the dead half would change the barrier word layout.

    # Named barriers.  MSA's NamedBarrierFwdSm100 emits its raw enum values with
    # no user-barrier bias -- the export's physical ids are 2 (TmemPtr), 11
    # (LoadWG), 12/13 (StoreEpilogue + stage), 13 (KvLoad), 14/15 (dequant) --
    # which produces a real collision (see below).  The port renumbers from 8
    # upward, the convention this repository documents at
    # `tirx_kernels/flashmla/sparse_decode_head64.py:36-39`.
    tmem_alloc_bar   = named_barrier(id=8,  threads=32 * (4 * 2 + 1))   # 288: both softmax WGs + MMA warp
    load_wg_bar      = named_barrier(id=9,  threads=32 * 4)             # the Q-load warpgroup
    kv_load_bar      = named_barrier(id=10, threads=32 * 2)             # fp8 paths only
    kv_dequant_k_bar = named_barrier(id=11, threads=32 * 4)
    kv_dequant_v_bar = named_barrier(id=12, threads=32 * 4)
    epilogue_bar     = named_barrier(id=13, threads=32 * 4, indexed=2)  # id 13 + stage
    #   DEVIATION, recorded: upstream uses StoreEpilogue=12 with `barrier_id +
    #   stage`, so stage 1 resolves to id 13 -- which is also KvLoad's id.  The
    #   export shows the collision physically: `bar.sync 13, 128` at :2554 (the
    #   stage-1 epilogue) and `bar.sync 13, 64` at :1485 (KvLoad), the same
    #   barrier id issued with two different participant counts in one kernel.
    #   Upstream is safe only by accident of timing: kv_load_bar fires once,
    #   very early, before any epilogue runs.  The port keeps the identical
    #   synchronization structure and gives the indexed epilogue barrier its own
    #   pair of ids.
    sm_stats_bar     = named_barrier(id=15, threads=32 * 2)
    #   DEAD: `signal_stats_barrier=False` at both `_softmax_step` call sites
    #   (:2518, :2552) and `use_stats_barrier=False` in the epilogue (:2584),
    #   so no arrive is ever emitted.  Declared to keep the id space aligned.
```

```python
    # =======================================================================
    # Prologue.  Descriptor prefetch, thread-0 metadata publish, mbarrier init.
    # =======================================================================

    if warp == 0:
        prefetch_descriptor(K_map)
        # instruction_selection: prefetch.tensormap (:721, 1 static);
        #   extent: scalar, warp-wide (no elect -- the whole warp issues it).
        prefetch_descriptor(V_map)
        # instruction_selection: prefetch.tensormap (:722, 1 static); extent: scalar.
        if not USE_Q_GATHER4:
            prefetch_descriptor(Q_map)
            # instruction_selection: prefetch.tensormap (:724, 1 static); extent: scalar.
        else:
            with elect():
                prefetch_descriptor(Q_gather4_map)
                # instruction_selection: prefetch.tensormap under elect.sync
                #   (:727); extent: scalar, one lane.

    # The cluster handshake.  The source comments it "no-op for 1CTA cluster";
    # the export disagrees and emits a real fence and a real CTA barrier.
    fence("mbarrier_init_release_cluster")
    # instruction_selection: fence.mbarrier_init.release.cluster (:841, 1 static);
    #   extent: scalar.
    barrier()
    # instruction_selection: bar.sync (:842, 1 static); extent: CTA-wide.

    if tid == 0:
        base_row_start = load_global(RowPtr, (head_kv_idx, row_linear))
        # instruction_selection: ld.global.u32 (:860, 1 static); extent: scalar,
        #   thread 0 only.  The source also computes `count_raw` from
        #   RowPtr[.., row_linear+1] (:862-865) and then immediately overwrites
        #   both with the work item's own slice (:866-867), so that second CSR
        #   load is dead and the export carries no instruction for it.
        row_start = base_row_start + work_q_begin
        count_raw = work_q_count

        # kv_valid_cols = clamp(seqlen_k - kv_block*128, 0, 128)   (:215-229)
        seqlen_k = load_global(CuK, batch_idx + 1) - load_global(CuK, batch_idx)
        # instruction_selection: ld.global.u32 x2 (:212, 2 static); extent: scalar.
        kv_valid_cols = min(max(seqlen_k - kv_block_idx * 128, 0), 128)
        q_batch_offset = load_global(CuQ, batch_idx)
        # instruction_selection: ld.global.u32 (:198, 1 static); extent: scalar.
        k_batch_offset = load_global(CuK, batch_idx)
        # instruction_selection: none -- CSE'd into the :212 pair above, which
        #   already loads CuK[batch_idx].  The module's ld.global count is
        #   exactly 14 with no instruction for :883.

        store_shared(sRowMeta, 0..6,
                     [batch_idx, kv_block_idx, row_start, count_raw,
                      kv_valid_cols, q_batch_offset, k_batch_offset])
        # instruction_selection: st.shared.v4.u32 (:888, 1 static); extent: one
        #   4-word vector store.  The source writes seven separate scalar
        #   `sRowMeta[i] = ...` statements (:885-891); the backend merges them.

        # causal_q_offset = seqlen_k - seqlen_q, needed by the mask (:892-901)
        seqlen_q = load_global(CuQ, batch_idx + 1) - q_batch_offset
        # instruction_selection: ld.global.u32 (:894, 1 static); extent: scalar.
        causal_q_offset = seqlen_k - seqlen_q

        store_shared(sRowMeta, 7, causal_q_offset)
        # instruction_selection: st.shared.v4.u32 (:902, 1 static); extent: a
        #   second 4-word vector store, ORDERED AFTER the :894 global load --
        #   the two v4 stores are separated by it, so a port that merges all
        #   eight fields into one store would have to sink the seqlen_q load
        #   above them and change the dependence order.

        init_pipe(mbar_k[0], arrivals=1)
        # instruction_selection: mbarrier.init.shared.b64 (:907, 1 static); extent: scalar.
        init_pipe(mbar_v[0], arrivals=1)
        # instruction_selection: mbarrier.init.shared.b64 (:908, 1 static); extent: scalar.
        if K_FP8_TO_BF16:
            init_pipe(mbar_k_tma[0], arrivals=1)
            expect_tx(mbar_k_tma[0], k_tma_bytes)
        else:
            expect_tx(mbar_k[0], k_tma_bytes)
            # instruction_selection: mbarrier.expect_tx.relaxed.cta.shared::cta.b64
            #   (:913, 1 static); extent: scalar.  Note the K/V barriers take
            #   their expect_tx HERE, in the prologue, not at the load warp --
            #   the opposite of the Q pipe, which sets it at producer acquire.
        if V_FP8_TO_BF16:
            init_pipe(mbar_v_tma[0], arrivals=1)
            expect_tx(mbar_v_tma[0], v_tma_bytes)
        else:
            expect_tx(mbar_v[0], v_tma_bytes)
            # instruction_selection: mbarrier.expect_tx.relaxed.cta.shared::cta.b64
            #   (:918, 1 static); extent: scalar.

        # The causal diagonal split point: how many of this row's tokens still
        # need per-column causal masking.  A 32-step binary search over the CSR
        # row, which is sorted by q_idx (:259-285).
        diag_q_count = 0
        if count_raw > 0 and kv_valid_cols > 0:
            q_threshold = (kv_block_idx * 128 + kv_valid_cols) - causal_q_offset
            left, right = 0, count_raw
            for _ in range(32):                    # `unroll=1`, and it stays rolled
                if left < right:
                    mid = (left + right) // 2
                    probe = load_global(Idx, (head_kv_idx, row_start + mid))
                    # instruction_selection: ld.global.u32 (:239, 1 static);
                    #   extent: scalar, ONE instruction inside a rolled loop.
                    #   The single static occurrence is the evidence the loop
                    #   was not unrolled and the probe was not hoisted: the port
                    #   must express this as a rolled loop with the load in the
                    #   body, not as 32 straight-line probes.
                    left, right = (mid + 1, right) if probe < q_threshold else (left, mid)
            diag_q_count = left
        store_shared(sDiagQCount, 0, diag_q_count)
        # instruction_selection: st.shared.u32 (:935, 1 static); extent: scalar.

    fence("mbarrier_init_release_cluster")
    # instruction_selection: fence.mbarrier_init.release.cluster (:936, 1 static);
    #   extent: scalar.  This is the SECOND such fence in the prologue; the
    #   first (:841) belongs to the cluster handshake.
    barrier()
    # instruction_selection: bar.sync (:937, 1 static); extent: CTA-wide.
    #   Every warp reaches this, including the CTAs whose `cta_valid_work` is
    #   false and every warp that matches no role below.

    # The pipes' own barrier words are initialized by the pipeline objects
    # rather than by this thread-0 block:
    # instruction_selection: mbarrier.init.shared.b64 (pipeline.py:62 x4,
    #   :169 x4, :269 x8, :324 x8; 24 static total); extent: one per stage per
    #   half across the six pipes.

    if warp == 15:
        set_register_budget("decrease", NUM_REGS_OTHER)     # 48
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32 (:992, 1 static);
        #   extent: warp-wide.  NOT gated on cta_valid_work.
        return

    # =======================================================================
    # ROLE: Q-load warpgroup, warps 8..11.  Independent `if` on a THREAD range.
    # =======================================================================
    if 8 * 32 <= tid < 12 * 32 and cta_valid_work:
        set_register_budget("decrease", NUM_REGS_STORE)     # 112 on the causal path
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32 (:1001, 1 static);
        #   extent: warpgroup.  `store_reg_decrease` is True whenever
        #   NUM_REGS_STORE <= 128 (:179), which the causal specialization is.

        row_start, count_raw = load_shared(sRowMeta, 2..3)
        # instruction_selection: ld.shared.v2.u32 (:1004, 1 static); extent: two
        #   adjacent words in one instruction.  The source reads sRowMeta[2] and
        #   sRowMeta[3] as separate statements; the backend merges them.
        q_batch_offset = load_shared(sRowMeta, 5)
        # instruction_selection: ld.shared.u32 (:1006, 1 static); extent: scalar.
        has_work = count_raw > 0
        num_q_groups = ceil_div(count_raw, Q_TOKENS_PER_GROUP)
        # Deliberately NOT gated on KV validity: a sparse entry past seqused_k
        # still has to run the all-masked path so its partial is neutral
        # (:1007-1009).

        if USE_Q_GATHER4:
            q_load_gather4(...)      # warps 8..11, body below
        else:
            q_load_tma(...)          # warps 8..11, body below

    # -----------------------------------------------------------------------
    # Q-load body A: the raw gather4 path (QHEADPERKV in {1, 2, 4})
    # -----------------------------------------------------------------------
    def q_load_gather4():
        warp_in_wg = shuffle_index(local_warp, lane=0, mask=31, clamp=-1)
        lane_idx = (tid - 8 * 32) % 32
        # mQ_2d.shape[0] // QHEADPERKV = total_q * num_heads_kv: one past the
        # last Q *tile*, not the last Q row.  The gather arm below scales it by
        # QHEADPERKV to get one past the last Q *row*, so the TMA lands out of
        # bounds and the descriptor's OOB fill supplies the data.  `total_q`
        # alone is an IN-RANGE row whenever num_heads_kv > 1 (:1647).
        q_oob_m_idx = total_q * num_heads_kv
        GATHERS_PER_WARP = 128 // (4 * 4)           # = 8 (:1648-1649)

        if not has_work:
            return
        for qi_group in range(num_q_groups):        # rolled
            slot = qi_group % Q_STAGE
            phase = (qi_group // Q_STAGE) & 1
            producer_phase = phase ^ 1
            if warp_in_wg == 0:
                acquire(q_pipe.producer, slot, producer_phase)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 then
                #   mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 (the
                #   TMA-Q variant of this same call is at :1867, 1 static each);
                #   extent: scalar, one thread.  The expect_tx is issued HERE,
                #   at acquire, and the gather4 TMAs below sum to exactly
                #   q_tma_bytes for the stage.
            named_barrier_arrive_and_wait(load_wg_bar)
            # instruction_selection: bar.sync with a named id (the TMA-Q twin is
            #   at :1869, 1 static); extent: 128 threads.

            qidx_meta_slot = (qi_group & (QIDX_META_STAGES - 1)) * Q_TOKENS_PER_GROUP
            # Publish this group's packed qsplit words.  Q_TOKENS_PER_GROUP is
            # 32/64/128 on this path, so it takes `meta_iters` sweeps of the
            # whole warpgroup (:1671-1690).
            for meta_iter in range(meta_iters):     # constexpr, fully unrolled
                tok = (meta_iter * 4 + warp_in_wg) * 32 + lane_idx
                if tok < Q_TOKENS_PER_GROUP:
                    qi = qi_group * Q_TOKENS_PER_GROUP + tok
                    if qi < count_raw:
                        word = load_global(QSpl, (head_kv_idx, row_start + qi))
                        # instruction_selection: ld.global.u32 (:249, 1 static);
                        #   extent: scalar.
                        store_shared(sQIdxMeta, qidx_meta_slot + tok, word)
                        # instruction_selection: st.shared.u32 (:1686, 1 static);
                        #   extent: scalar.
                    else:
                        store_shared(sQIdxMeta, qidx_meta_slot + tok, 0)
                        # instruction_selection: st.shared.u32 (:1690, 1 static);
                        #   extent: scalar.  The two arms stay two separate
                        #   stores in the export, exactly as on the TMA-Q path --
                        #   they are not folded into a select plus one store.
            named_barrier_arrive_and_wait(load_wg_bar)
            # instruction_selection: bar.sync, named; extent: 128 threads.
            #   Two barriers per group bracket the metadata publish: the ring
            #   slot must be acquired before anyone writes it, and every writer
            #   must be done before any lane reads a neighbour's token.

            with elect():
                for gather_slot in range(GATHERS_PER_WARP):   # constexpr, unrolled
                    gather_idx = gather_slot * 4 + warp_in_wg
                    tok_base = gather_idx * TOKENS_PER_GATHER4

                    # Resolve the four GMEM row indices this gather4 pulls.
                    # The three QHEADPERKV cases differ only in how many
                    # distinct tokens the four rows come from (:1702-1763):
                    #   1 -> four tokens, one head row each
                    #   2 -> two tokens, two consecutive head rows each
                    #   4 -> one token, four consecutive head rows
                    # A token past `count_raw` uses q_oob_m_idx, so the gather
                    # lands on an out-of-range row instead of being predicated
                    # off -- the mask makes its contribution neutral later.
                    q_idx = decode_q_idx(load_shared(sQIdxMeta, qidx_meta_slot + tok_base))
                    # instruction_selection: ld.shared.u32 (:1755, 8 static on
                    #   the QHEADPERKV=4 export); extent: scalar, one per gather
                    #   slot -- the metadata word is re-read per gather, not
                    #   hoisted across the eight.
                    r0 = (q_batch_offset + q_idx) * num_heads_kv + head_kv_idx
                    r0, r1, r2, r3 = expand_rows(r0, QHEADPERKV)

                    if Q_DTYPE is fp8:
                        copy_g2s_gather4(sQ, slot * Q_STAGE_STRIDE_BYTES
                                             + gather_idx * 4 * 128 * 1,
                                         Q_gather4_map, col=0,
                                         r0, r1, r2, r3,
                                         bar=q_pipe.full[slot],
                                         hint=TMA_CACHE_EVICT_LAST)
                        # instruction_selection:
                        #   cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4
                        #   .mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint;
                        #   extent: one 4-row gather per slot, 8 per warp.
                    else:
                        for ks in range(K_STAGES):        # constexpr, unrolled
                            if ks + 1 < K_STAGES:
                                prefetch_g2s_gather4(Q_gather4_map,
                                                     col=(ks + 1) * K_TILE,
                                                     r0, r1, r2, r3,
                                                     hint=TMA_CACHE_EVICT_LAST)
                                # instruction_selection:
                                #   cp.async.bulk.prefetch.tensor.2d.L2.global
                                #   .tile::gather4.L2::cache_hint (:1789, 8
                                #   static); extent: one per gather slot -- only
                                #   ks=0 satisfies the guard, so the second
                                #   k-subtile is prefetched while the first is
                                #   still in flight.
                            copy_g2s_gather4(sQ,
                                             (slot * K_STAGES + ks) * K_TILE_STRIDE_BYTES
                                                 + gather_idx * 4 * K_TILE * 2,
                                             Q_gather4_map, col=ks * K_TILE,
                                             r0, r1, r2, r3,
                                             bar=q_pipe.full[slot],
                                             hint=TMA_CACHE_EVICT_LAST)
                            # instruction_selection:
                            #   cp.async.bulk.tensor.2d.shared::cta.global
                            #   .tile::gather4.mbarrier::complete_tx::bytes
                            #   .cta_group::1.L2::cache_hint (:1798, 16 static);
                            #   extent: 8 gather slots x 2 k-subtiles per warp.
            named_barrier_arrive_and_wait(load_wg_bar)
            # instruction_selection: bar.sync, named; extent: 128 threads.

        if warp_in_wg == 0:
            # One extra acquire past the end so the ring's empty half is left in
            # the state the next work item would expect (:1812-1817).
            acquire(q_pipe.producer,
                    num_q_groups % Q_STAGE,
                    ((num_q_groups // Q_STAGE) & 1) ^ 1)

    # -----------------------------------------------------------------------
    # Q-load body B: the CUTE-managed TMA path (QHEADPERKV in {8, 16})
    # -----------------------------------------------------------------------
    def q_load_tma():
        warp_in_wg = warp - 8
        q_oob_m_idx = total_q * num_heads_kv    # = mQ_2d.shape[0] // QHEADPERKV (:1855)
        TOKENS_PER_WARP = ceil_div(Q_TOKENS_PER_GROUP, 4)
        SUBTILES_PER_TOKEN = 1 if Q_DTYPE is fp8 else K_STAGES

        if not has_work:
            return
        for qi_group in range(num_q_groups):        # rolled
            slot = qi_group % Q_STAGE
            phase = (qi_group // Q_STAGE) & 1
            if warp_in_wg == 0:
                acquire(q_pipe.producer, slot, phase ^ 1)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 +
                #   mbarrier.arrive.expect_tx.release.cta.shared::cta.b64
                #   (:1867, 1 static each); extent: scalar, one thread.
            named_barrier_arrive_and_wait(load_wg_bar)
            # instruction_selection: bar.sync, named (:1869, 1 static);
            #   extent: 128 threads.

            load_meta_slot  = slot * Q_TOKENS_PER_GROUP
            qidx_meta_slot  = (qi_group & (QIDX_META_STAGES - 1)) * Q_TOKENS_PER_GROUP
            sub_stage_base  = slot * Q_TOKENS_PER_GROUP * SUBTILES_PER_TOKEN

            # With QHEADPERKV >= 8 the group is at most 16 tokens, so ONE warp's
            # low lanes publish the whole group (:1883-1898).
            if warp_in_wg == 0 and lane < Q_TOKENS_PER_GROUP:
                qi = qi_group * Q_TOKENS_PER_GROUP + lane
                if qi < count_raw:
                    word = load_global(QSpl, (head_kv_idx, row_start + qi))
                    # instruction_selection: ld.global.u32 (:249, 1 static); extent: scalar.
                    store_shared(sQIdxMeta, qidx_meta_slot + lane, word)
                    # instruction_selection: st.shared.u32 (:1892, 1 static); extent: scalar.
                    store_shared(sQLoadMIdx, load_meta_slot + lane,
                                 (q_batch_offset + decode_q_idx(word)) * num_heads_kv
                                     + head_kv_idx)
                    # instruction_selection: st.shared.u32 (:1893, 1 static); extent: scalar.
                else:
                    store_shared(sQIdxMeta, qidx_meta_slot + lane, 0)
                    # instruction_selection: st.shared.u32 (:1897, 1 static); extent: scalar.
                    store_shared(sQLoadMIdx, load_meta_slot + lane, q_oob_m_idx)
                    # instruction_selection: st.shared.u32 (:1898, 1 static); extent: scalar.
                    #   The two arms stay separate stores in the export; they are
                    #   not merged into a select plus one store.
            named_barrier_arrive_and_wait(load_wg_bar)
            # instruction_selection: bar.sync, named (:1899, 1 static); extent: 128 threads.

            for qi_slot in range(TOKENS_PER_WARP):        # constexpr, unrolled
                tok = warp_in_wg * TOKENS_PER_WARP + qi_slot
                if tok < Q_TOKENS_PER_GROUP:
                    m_tile = load_shared(sQLoadMIdx, load_meta_slot + tok)
                    # instruction_selection: ld.shared.u32 (:1907, 2 static);
                    #   extent: scalar, once per token slot.
                    if Q_DTYPE is fp8:
                        copy_g2s(Qf[m_tile], sQ_load[sub_stage_base + tok],
                                 bar=q_pipe.full[slot])
                    else:
                        copy_g2s(Qf_k0[m_tile], sQ_load[sub_stage_base + tok],
                                 bar=q_pipe.full[slot])
                        # instruction_selection: cp.async.bulk.tensor.2d
                        #   .shared::cluster.global.tile
                        #   .mbarrier::complete_tx::bytes.L2::cache_hint under
                        #   elect.sync (:1915, 2 static each); extent: one
                        #   (QHEADPERKV x 64) box.
                        copy_g2s(Qf_k1[m_tile],
                                 sQ_load[sub_stage_base + Q_TOKENS_PER_GROUP + tok],
                                 bar=q_pipe.full[slot])
                        # instruction_selection: cp.async.bulk.tensor.2d
                        #   .shared::cluster.global.tile
                        #   .mbarrier::complete_tx::bytes.L2::cache_hint under
                        #   elect.sync (:1920, 2 static each); extent: the second
                        #   64-column k-subtile of the same token.

        if warp_in_wg == 0:
            acquire(q_pipe.producer,
                    num_q_groups % Q_STAGE,
                    ((num_q_groups // Q_STAGE) & 1) ^ 1)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 +
            #   mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 (:1935,
            #   1 static each); extent: scalar.

    # =======================================================================
    # ROLE: KV load, warps 13 and 14.  One TMA each; there is no KV ring.
    # =======================================================================
    if 13 <= warp < 15 and cta_valid_work:
        set_register_budget("decrease", NUM_REGS_OTHER)     # 48
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32 (:1054, 1 static);
        #   extent: warp-wide.
        kv_block = load_shared(sRowMeta, 1)
        # instruction_selection: ld.shared.u32 (:1055, 1 static); extent: scalar.
        k_batch_offset = load_shared(sRowMeta, 6)
        # instruction_selection: ld.shared.u32 (:1056, 1 static); extent: scalar.
        has_work = load_shared(sRowMeta, 3) > 0
        # instruction_selection: ld.shared.u32 (:1057, 1 static); extent: scalar.
        #   These three stay three separate scalar loads -- unlike the Q-load
        #   warpgroup's sRowMeta[2..3], which the backend merged into a v2.

        if not has_work:
            return
        warp_in_role = warp - 13
        if warp_in_role == 0:
            if K_FP8_TO_BF16:
                copy_g2s(K_desc_view[kv_block, :, head_kv_idx], sKFp8,
                         bar=mbar_k_tma[0])
                with elect():
                    commit(mbar_k_tma[0])
            else:
                copy_g2s(K_desc_view[kv_block, :, head_kv_idx], sK, bar=mbar_k[0])
                # instruction_selection: cp.async.bulk.tensor.**3d**
                #   .shared::cluster.global.tile.mbarrier::complete_tx::bytes
                #   .L2::cache_hint under elect.sync (:1571, 2 static each);
                #   extent: the CTA's single 128x128 K tile, issued as TWO
                #   instructions.  Rank 3, not 2: the KV-head axis stays in the
                #   descriptor rather than being folded into the base address.
                with elect():
                    commit(mbar_k[0])
                    # instruction_selection: mbarrier.arrive.release.cta
                    #   .shared::cta.b64 (:1580, 1 static); extent: scalar, one
                    #   lane.  The transaction-byte count was already promised
                    #   by thread 0's prologue expect_tx, so this is a plain
                    #   arrive, not an arrive-and-expect.
        if warp_in_role == 1:
            if V_FP8_TO_BF16:
                copy_g2s(V_desc_view[:, kv_block, head_kv_idx], sVFp8,
                         bar=mbar_v_tma[0])
                with elect():
                    commit(mbar_v_tma[0])
            else:
                copy_g2s(V_desc_view[:, kv_block, head_kv_idx], sV, bar=mbar_v[0])
                # instruction_selection: cp.async.bulk.tensor.3d
                #   .shared::cluster.global.tile.mbarrier::complete_tx::bytes
                #   .L2::cache_hint under elect.sync (:1612, 2 static each);
                #   extent: the CTA's single 128x128 V tile, MN-major.
                with elect():
                    commit(mbar_v[0])
                    # instruction_selection: mbarrier.arrive.release.cta
                    #   .shared::cta.b64 (:1621, 1 static); extent: scalar.
        if K_FP8_TO_BF16 or V_FP8_TO_BF16:
            named_barrier_arrive_and_wait(kv_load_bar)
            # instruction_selection: bar.sync, named, 64 threads (:1485);
            #   extent: the two KV load warps.  Present only on the fp8 paths.
```

```python
    # =======================================================================
    # ROLE: the single MMA-issue warp, warp 12.  Also the TMEM allocator warp.
    # =======================================================================
    if warp == 12 and cta_valid_work:
        set_register_budget("decrease", NUM_REGS_OTHER)     # 48
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32 (:1089, 1 static);
        #   extent: warp-wide.
        count_raw = load_shared(sRowMeta, 3)
        has_work = count_raw > 0
        num_q_groups = ceil_div(count_raw, Q_TOKENS_PER_GROUP)

        tmem.allocate(columns=512)
        # instruction_selection: tcgen05.alloc.cta_group::1.sync.aligned
        #   .shared::cta.b32 (:1095, 1 static); extent: scalar.
        tmem.wait_for_alloc()
        # instruction_selection: bar.sync, named, 288 threads (:1096, 1 static);
        #   extent: the TMEM retrieve barrier.
        # The retrieve barrier spans 288 threads = both softmax warpgroups plus
        # this warp (:767-772).  On the fp8 paths that means the MMA warp is
        # gated on the softmax warpgroups having finished dequantizing K and V
        # into sK/sV -- the dequant is upstream of the first QK by construction,
        # not by an extra edge.

        # Descriptor state, hoisted into named PTX registers before the loop.
        # `declare_ptx_smem_desc` materializes the Q A-operand descriptors as
        # persistent `.reg .b64 lean_q_desc_<k>` and `declare_ptx_idesc` the
        # instruction descriptor as `.reg .b32 lean_qk_idesc` (:1982-1989).
        # Four QK partials are pre-bound: {S slot 0, S slot 1} x {wrap, advance}
        # (:1996-2043), where `advance` adds +sQ_stage_stride to the A
        # descriptor and `wrap` adds -(Q_STAGE-1)*stride.  The Q ring is walked
        # inside PTX registers, never re-derived per iteration.
        q_desc  = reg_tile([], "u64", name="lean_q_desc_k", persistent=True)
        qk_idesc = reg_tile([], "u32", name="lean_qk_idesc", persistent=True)

        if not has_work:
            return

        wait(mbar_k[0], phase=0)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:2069,
        #   1 static); extent: scalar.

        # Issue order (:2070-2077):  Q0K, Q1K, P0V, Q2K, P1V, Q3K, ...
        #   QK(qi) consumes S slot qi&1; PV(qi-2) frees that same slot before
        #   QK(qi) reuses it, so a 2-slot S ring is safe and the phase toggles
        #   every two groups per slot.

        # --- prologue: up to two QK tiles ---------------------------------
        wait(q_pipe.full, slot=0, phase=0)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:2081, 1 static);
        #   extent: scalar.
        acquire(s_pipe.producer, slot=0, phase=1)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:2082, 1 static);
        #   extent: scalar.
        gemm(S[0], sQ[0], sK, accumulate=False)
        # instruction_selection: tcgen05.mma.cta_group::1.kind::f16 (48 static
        #   across the whole kernel, all reported at :291 because the chain is
        #   built inside blackwell_helpers' PTX string builder); extent: one
        #   128x128x128 QK tile issued as a K-loop of same-family instructions
        #   over the pre-bound descriptors.  `mma_kind` becomes "f8f6f4" when
        #   the QK operand width is 8.
        commit(s_pipe, slot=0)
        # instruction_selection: tcgen05.commit.cta_group::1
        #   .mbarrier::arrive::one.shared::cluster.b64 (:2084, 1 static);
        #   extent: scalar.
        release(q_pipe.consumer, slot=0)
        # instruction_selection: tcgen05.commit.cta_group::1
        #   .mbarrier::arrive::one.shared::cluster.b64 (:2085, 1 static);
        #   extent: scalar.  A TMA->UMMA pipe releases its empty half with a
        #   tcgen05 commit, not a plain mbarrier arrive, so the release is
        #   ordered behind the MMA that consumed the stage.

        if num_q_groups > 1:
            wait(q_pipe.full, slot=1, phase=0)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:2088, 1 static).
            acquire(s_pipe.producer, slot=1, phase=1)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:2089, 1 static).
            gemm(S[1], sQ[1], sK, accumulate=False)
            # instruction_selection: tcgen05.mma.cta_group::1.kind::f16; extent:
            #   one QK tile, the `advance` partial (A descriptor +stride).
            commit(s_pipe, slot=1)
            # instruction_selection: tcgen05.commit... (:2091, 1 static).
            release(q_pipe.consumer, slot=1)
            # instruction_selection: tcgen05.commit... (:2092, 1 static).

        wait(mbar_v[0], phase=0)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:2094, 1 static);
        #   extent: scalar.  V is waited AFTER the two prologue QKs, so the V
        #   TMA overlaps them.

        # --- steady state: PV(qi-2) then QK(qi) ---------------------------
        for qi in range(2, num_q_groups):           # rolled (`unroll=1`)
            pv_qi   = qi - 2
            pv_slot = pv_qi & 1
            pv_phase = (pv_qi // 2) & 1
            wait(p_pipe.full, pv_slot, pv_phase)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:2104,
            #   1 static); extent: scalar.  This is the EARLY-P edge: softmax
            #   has published only the first 3/4 of P.
            acquire(o_pipe.producer, pv_slot, pv_phase ^ 1)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:2105,
            #   1 static); extent: scalar.
            gemm(O[pv_slot], P[pv_slot], sV, accumulate=False)
            # instruction_selection: an 8-instruction
            #   tcgen05.mma.cta_group::1.kind::f16 chain (PTX 1120-1147) with an
            #   embedded mbarrier.try_wait.parity.shared::cta.b64 (2 static,
            #   reported at :291) partway through it; extent: one 128x128x128 PV
            #   tile, issued as two batches of the same instruction family with
            #   the late-P wait between them.  `split_arrive = 96` means the
            #   first 3/4 of the K extent is issued against the early-P barrier
            #   and the sequence then blocks on p_last before the final quarter.
            #   Unlike the QK issue below, the `pv_slot` test does NOT duplicate
            #   the chain: it collapses to an address select feeding one chain,
            #   because the two arms differ only in their TMEM addresses.  The A
            #   operand lives in TMEM, so its address is passed explicitly
            #   (tA_addr = tmem_p_offset) rather than read from the tensor.
            commit(o_pipe, pv_slot)
            # instruction_selection: tcgen05.commit.cta_group::1
            #   .mbarrier::arrive::one.shared::cluster.b64 (:2127, 1 static);
            #   extent: scalar.
            release(p_last_pipe.consumer, pv_slot)
            # instruction_selection: tcgen05.commit.cta_group::1
            #   .mbarrier::arrive::one.shared::cluster.b64 (:2129, 1 static);
            #   extent: scalar.
            release(p_pipe.consumer, pv_slot)
            # instruction_selection: tcgen05.commit.cta_group::1
            #   .mbarrier::arrive::one.shared::cluster.b64 (:2130, 1 static);
            #   extent: scalar.  Note that p_pipe and p_last_pipe are
            #   softmax-PRODUCED pipes, yet their consumer release is still a
            #   tcgen05 commit -- the rule is about who issues the arrival, not
            #   who produced the data.

            q_slot  = qi % Q_STAGE
            q_phase = (qi // Q_STAGE) & 1
            s_slot  = qi & 1
            s_phase = (qi // 2) & 1
            wait(q_pipe.full, q_slot, q_phase)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:2136,
            #   1 static); extent: scalar.
            acquire(s_pipe.producer, s_slot, s_phase ^ 1)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:2137,
            #   1 static); extent: scalar.
            if s_slot == 0:
                gemm(S[0], sQ[q_slot], sK, accumulate=False)
            else:
                gemm(S[1], sQ[q_slot], sK, accumulate=False)
            # instruction_selection: two separate 8-instruction
            #   tcgen05.mma.cta_group::1.kind::f16 chains at PTX 1264-1271 and
            #   1335-1342, selected by a RUNTIME branch --
            #   `setp.ne.s32 %p90, %r62, 0` at PTX 1181 (.loc 1 2107) and
            #   `@%p90 bra $L__BB0_74` at PTX 1209; extent: one QK tile per arm.
            #   `s_slot` and `q_slot` are runtime values in a rolled loop, so
            #   this is NOT a compile-time choice: the S-slot test costs a real
            #   branch and duplicates the chain, while the (q_slot 0 -> `wrap`,
            #   q_slot 1 -> `advance`) descriptor choice collapses into an
            #   address select feeding the chain.  Whole-kernel census: 48
            #   tcgen05.mma = 6 chains x 8 instructions (2 prologue QK, 2 steady
            #   QK behind this branch, 1 steady PV, 1 drain PV); an f8f6f4
            #   specialization emits 24 = 6 x 4.
            commit(s_pipe, s_slot)
            # instruction_selection: tcgen05.commit.cta_group::1
            #   .mbarrier::arrive::one.shared::cluster.b64 (:2149, 1 static);
            #   extent: scalar.
            release(q_pipe.consumer, q_slot)
            # instruction_selection: tcgen05.commit.cta_group::1
            #   .mbarrier::arrive::one.shared::cluster.b64 (:2150, 1 static);
            #   extent: scalar.

        # --- drain the last one or two PV tiles ---------------------------
        drain_begin = 0 if num_q_groups == 1 else num_q_groups - 2
        for pv_qi in range(drain_begin, num_q_groups):     # rolled
            pv_slot  = pv_qi & 1
            pv_phase = (pv_qi // 2) & 1
            wait(p_pipe.full, pv_slot, pv_phase)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:2157,
            #   1 static); extent: scalar.
            acquire(o_pipe.producer, pv_slot, pv_phase ^ 1)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:2158,
            #   1 static); extent: scalar.
            gemm(O[pv_slot], P[pv_slot], sV, accumulate=False)
            # instruction_selection: the sixth and last 8-instruction
            #   tcgen05.mma.cta_group::1.kind::f16 chain, with the same embedded
            #   late-P wait; extent: one PV tile.
            commit(o_pipe, pv_slot)
            # instruction_selection: tcgen05.commit... (:2180, 1 static); extent: scalar.
            release(p_last_pipe.consumer, pv_slot)
            # instruction_selection: tcgen05.commit... (:2182, 1 static); extent: scalar.
            release(p_pipe.consumer, pv_slot)
            # instruction_selection: tcgen05.commit... (:2183, 1 static); extent: scalar.

        tmem.relinquish_alloc_permit()
        # instruction_selection: tcgen05.relinquish_alloc_permit.cta_group::1
        #   .sync.aligned (:1108, 1 static); extent: scalar.
        named_barrier_arrive_and_wait(tmem_alloc_bar)
        # instruction_selection: bar.sync, named, 288 threads (:1109, 1 static);
        #   extent: both softmax warpgroups plus this warp.
        tmem.free(columns=512)
        # instruction_selection: tcgen05.dealloc.cta_group::1.sync.aligned.b32
        #   (:1110, 1 static); extent: scalar.
        griddepcontrol_launch_dependents()
        # instruction_selection: griddepcontrol.launch_dependents (:1111, 1 static);
        #   extent: scalar.  There is no matching griddepcontrol.wait anywhere in
        #   this kernel, so no launch attribute is required -- only a kernel that
        #   waits needs `tirx.use_programtic_dependent_launch`.

    # =======================================================================
    # ROLE: softmax warpgroup 0 (warps 0..3) and 1 (warps 4..7).
    #
    # The export shows this whole body -- softmax step AND fused epilogue -- is
    # emitted TWICE, once per warpgroup, not shared behind a runtime `stage`.
    # Every single-site operation inside it has exactly 2 static occurrences.
    # =======================================================================
    if (0 <= warp < 4 or 4 <= warp < 8) and cta_valid_work:
        stage = 0 if warp < 4 else 1                # a COMPILE-TIME constant
        set_register_budget("increase", NUM_REGS_SOFTMAX)   # 176 on the causal path
        # instruction_selection: setmaxnreg.inc.sync.aligned.u32 (:1118 for WG0,
        #   :1180 for WG1, 1 static each); extent: warpgroup.

        kv_block_idx, count_raw = load_shared(sRowMeta, 1..3)
        # instruction_selection: ld.shared.v4.u32 (:1119 / :1181, 1 static each);
        #   extent: four adjacent metadata words in one instruction.  The source
        #   reads sRowMeta[1] and sRowMeta[3] as separate statements.
        kv_valid_cols     = load_shared(sRowMeta, 4)
        # instruction_selection: ld.shared.u32 (:1121 / :1183, 1 static); extent: scalar.
        causal_q_offset   = load_shared(sRowMeta, 7)
        # instruction_selection: ld.shared.u32 (:1122 / :1184, 1 static); extent: scalar.
        diag_q_count      = load_shared(sDiagQCount, 0)
        # instruction_selection: ld.shared.u32 (:1127 / :1189, 1 static); extent: scalar.
        q_batch_offset    = load_shared(sRowMeta, 5)
        # instruction_selection: ld.shared.u32 (:1170 / :1232, 1 static); extent: scalar.
        has_work = count_raw > 0
        num_q_groups = ceil_div(count_raw, Q_TOKENS_PER_GROUP)

        # -------------------------------------------------------------------
        # FP8 staging, before the main loop.  WG0 dequantizes K, WG1 V.
        # -------------------------------------------------------------------
        if stage == 0 and K_FP8_TO_BF16:
            dequant_kv(sKFp8, sK, mbar_k_tma, mbar_k, kv_dequant_k_bar, is_v=False)
        if stage == 1 and V_FP8_TO_BF16:
            dequant_kv(sVFp8, sV, mbar_v_tma, mbar_v, kv_dequant_v_bar, is_v=True)

        tmem.wait_for_alloc()
        # instruction_selection: bar.sync, named, 288 threads (:1140 / :1202,
        #   1 static each); extent: the TMEM retrieve barrier.

        # ... the per-group loop, below ...

        named_barrier_arrive(tmem_alloc_bar)
        # instruction_selection: bar.arrive, named (reported at :291, 2 static);
        #   extent: warpgroup.  Arrive-only: the MMA warp is the one that waits
        #   before freeing TMEM.

    # -----------------------------------------------------------------------
    # FP8 -> BF16 shared-memory staging (:1255-1303, :1488-1516)
    #
    # This is NOT in the load warps.  The KV load warps only land the fp8 bytes
    # in sKFp8/sVFp8; a whole softmax warpgroup then converts them into the bf16
    # tile the MMA reads, before entering its own loop.
    # -----------------------------------------------------------------------
    def dequant_kv(sFp8, sBf16, mbar_tma, mbar_ready, dequant_bar, is_v):
        if not has_work:
            return
        wait(mbar_tma[0], phase=0)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:1504,
        #   2 static across both warpgroups); extent: scalar.

        CHUNKS_PER_ROW = 128 // 16
        TOTAL_TASKS = 128 * CHUNKS_PER_ROW          # 1024 sixteen-element chunks
        task = warp_in_wg * 32 + lane
        for task_idx in range(task, TOTAL_TASKS, 4 * 32):     # rolled (`unroll=1`)
            row   = task_idx // CHUNKS_PER_ROW
            chunk = task_idx - row * CHUNKS_PER_ROW
            r_fp8 = reg_tile([16], "fp8e4m3")
            copy_s2r(sFp8[row, chunk * 16 : chunk * 16 + 16], r_fp8)
            # instruction_selection: ld.shared.v4.u32 (:1287, 2 static -- one per
            #   warpgroup copy); extent: 16 fp8 values as four 32-bit words in
            #   ONE instruction, inside a rolled loop.
            r_bf16 = reg_tile([16], "bf16")
            cast(r_bf16, r_fp8)
            # instruction_selection: prmt.b32 + fma.rn.bf16x2 (utils.py:540's
            #   cvt_fp8x4_e4m3_bf16x4, four words per task); extent: a 16-element
            #   register tile.  MSA does NOT use the hardware
            #   cvt.rn.bf16x2.e4m3x2 here; it uses the byte-permute plus packed
            #   FMA trick, and the port has to select the same one.
            for v in range(2):                      # constexpr, unrolled
                copy_r2s(r_bf16[v * 8 : v * 8 + 8], sBf16_dst_view(row, chunk, v, is_v))
                # instruction_selection: st.shared.v4.u32 (:1302, 4 static = 2
                #   per warpgroup copy); extent: 8 bf16 values per instruction.
                #   `is_v` selects the MN-major destination view (:1289-1298),
                #   which is an indexing difference only -- the instruction is
                #   the same.
        fence("view_async_shared")
        # instruction_selection: fence.proxy.async.shared::cta (:1303, 2 static);
        #   extent: scalar.
        named_barrier_arrive_and_wait(dequant_bar)
        # instruction_selection: bar.sync, named, 128 threads (:1513, 2 static);
        #   extent: warpgroup.
        if warp_in_wg == 0:
            with elect():
                commit(mbar_ready[0])
                # instruction_selection: mbarrier.arrive.release.cta
                #   .shared::cta.b64 (:1516, 2 static); extent: scalar.  This is
                #   what makes the MMA warp's `wait(mbar_k)` mean "the bf16 tile
                #   is ready" rather than "the fp8 bytes landed".

    # -----------------------------------------------------------------------
    # The per-group loop inside a softmax warpgroup (:2460-2586)
    # -----------------------------------------------------------------------
    def softmax_wg_loop(stage):
        group_tidx = tid - stage * 128              # 0..127; this thread's M row
        kv_block_col_start = kv_block_idx * 128     # causal only
        num_stage_groups = (num_q_groups + (1 - stage)) // 2
        # WG0 takes Q groups 0, 2, 4, ...; WG1 takes 1, 3, 5, ...

        for qi_iter in range(num_stage_groups):     # rolled (`unroll=1`)
            qi_group = qi_iter * 2 + stage
            phase = qi_iter & 1
            producer_phase = phase ^ 1
            qidx_meta_slot = (qi_group & (QIDX_META_STAGES - 1)) * Q_TOKENS_PER_GROUP

            softmax_reset()                         # row_max = -inf, row_sum = 0

            # How many of this group's tokens still sit on the causal diagonal.
            qi_group_start = qi_group * Q_TOKENS_PER_GROUP
            masked_tok_count = clamp(diag_q_count - qi_group_start,
                                     0, Q_TOKENS_PER_GROUP)

            softmax_step(stage, phase, producer_phase, masked_tok_count, ...)

            named_barrier_arrive_and_wait(epilogue_bar, index=stage)
            # instruction_selection: bar.sync, named, id = base + stage, 128
            #   threads (:2554); extent: warpgroup.  This is the barrier whose
            #   upstream id collides with KvLoad -- see the declaration above.

            epilogue_step(qi_group, stage, ...)

    # -----------------------------------------------------------------------
    # One softmax step (:2186-2352).  One M row per thread, 128 S columns.
    # -----------------------------------------------------------------------
    def softmax_step(stage, phase, producer_phase, masked_tok_count, ...):
        wait(s_pipe.full, slot=stage, phase=phase)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:2222,
        #   2 static); extent: scalar.

        rS = reg_tile([128], "f32")
        copy_t2r(S[stage][group_tidx, :], rS)
        # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x32.b32 (:2227,
        #   8 static = 4 per warpgroup copy); extent: this thread's 128 fp32 S
        #   elements arrive as four 32-register loads.

        # --- masking (mask.py:71-121) -------------------------------------
        # Column-limit masking always runs: any column >= kv_valid_cols is
        # forced to -inf through an r2p bitmask built from 32-bit chunks.
        # Causal masking runs only for the first `masked_tok_count` tokens of
        # the group, and it needs this thread's SPARSE q_idx, not its row index.
        if masked_tok_count > 0:
            tok = group_tidx // QHEADPERKV
            q_idx = decode_q_idx(load_shared(sQIdxMeta, qidx_meta_slot + tok))
            # instruction_selection: ld.shared.u32 (:2251, 2 static); extent: scalar.
            causal_col_limit = q_idx + causal_q_offset - kv_block_col_start + 1
            select(rS, col < min(kv_valid_cols, causal_col_limit), rS, -inf)
            # instruction_selection: shl.b32 to build the r2p chunk bitmask, then
            #   and.b32 per element (mask.py:40, 124 per warpgroup copy = four
            #   32-element r2p chunks x 31 emitted bit tests -- the one whose
            #   `1 << i` folds is free; 248 across the two copies), then
            #   selp.b32/selp.f32 for the -inf substitution; extent: a
            #   128-element register tile.  `and.b32` at mask.py:40 is the
            #   largest single .loc-attributed opcode group in the whole module
            #   -- the mask, not the softmax, dominates this step's static count.
        else:
            select(rS, col < kv_valid_cols, rS, -inf)
            # instruction_selection: none of its own -- this arm SHARES the r2p
            #   body above.  `const_expr(self.causal and apply_causal_mask)` at
            #   :2246 is a compile-time true, so `need_causal_mask` at :2247 is a
            #   runtime branch (`@%p121 bra $L__BB0_96`, PTX 1649): the causal
            #   arm falls through into the body at `$L__BB0_95` (PTX 1678) and
            #   this arm branches back into the same body (PTX 2480), both
            #   joining at `$L__BB0_97`.  Only the `col_limit` computation and
            #   its own `< 128` guard are duplicated per arm -- the bit-test and
            #   select tile is emitted ONCE per warpgroup copy.  A port that
            #   emits the masking tile per arm doubles the module's hottest
            #   instruction group.

        # --- row max, then scale-and-subtract ------------------------------
        # Always is_first=True: a Q group sees exactly one KV block, so this is
        # the first and only online-softmax step and there is no rescale of a
        # running accumulator (:2284-2287).
        row_max = max_reduce(rS)
        # instruction_selection: max.f32 (utils.py:262-276, 132 static = 66 per
        #   warpgroup copy); extent: an intra-thread binary tree over 128
        #   elements -- NO shuffles: each thread owns a whole M row, so the
        #   reduction never crosses lanes.  The module's other 32 max.f32 are
        #   the exp2 polynomial's clamp at utils.py:993, counted with exp2_poly
        #   below, not here.
        fma(rS, rS, softmax_scale_log2, -row_max * softmax_scale_log2, lanes=2)
        # instruction_selection: fma.rn.f32x2 (softmax.py:354, 128 static = 64
        #   per warpgroup copy); extent: 64 packed pairs.  The scale and the
        #   row-max subtraction are one fused packed FMA, not a multiply
        #   followed by a subtract.  The module's other 48 fma.rn.f32x2 belong
        #   to the exp2 polynomial at utils.py:924.

        if RETURN_TEMPERATURE_LSE:
            temp_row_sum = scaled_exp2_row_sum(rS, lse_temperature_scale)
            # instruction_selection: ex2.approx.ftz.f32 plus add.rn.f32x2;
            #   extent: a second full pass over the 128-element tile, needed
            #   because the temperature LSE is a differently-scaled sum, not a
            #   rescaling of the main one.

        # --- publish P in two pieces ---------------------------------------
        if SPLIT_P_ARRIVE > 0:
            acquire(p_last_pipe.producer, stage, producer_phase)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:2297,
            #   2 static); extent: scalar.
        acquire(p_pipe.producer, stage, producer_phase)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:2299,
        #   2 static); extent: scalar.

        rP = reg_tile([128], P_DTYPE)
        exp2(rP, rS)
        # instruction_selection: ex2.approx.ftz.f32 (softmax.py:389 and :390,
        #   112 static each = 112 per warpgroup copy) MIXED WITH exp2_poly
        #   (utils.py:924 fma.rn.f32x2 x48, :993 max.f32 x32, :995
        #   add.rm.f32x2 x16, :998/:1001 sub.rn.f32x2 x16 each); extent: 128 elements
        #   per thread, of which 112 go through MUFU and 16 through the
        #   degree-3 FFMA polynomial.  That ratio is `ex2_emu_freq = 16` counted
        #   in ELEMENTS -- one emulated pair per sixteen -- and it exists only
        #   because this specialization is causal.  A port that sends all 128
        #   through MUFU has a different instruction mix and a different MUFU
        #   pressure profile.
        cast(rP, rS)
        # instruction_selection: cvt.rn.bf16x2.f32 (softmax.py:398, 128 static
        #   = 64 per warpgroup copy); extent: 128 elements as 64 packed
        #   conversions.  On an fp8 P the same site becomes the fp8 packed
        #   conversion instead.

        for k in range(4):                          # constexpr, unrolled
            copy_r2t(rP[k], P[stage][group_tidx, k])
            # instruction_selection: tcgen05.st.sync.aligned.32x32b.x16.b32
            #   (:2315, 8 static = 4 per warpgroup copy); extent: one quarter of
            #   this thread's P row.  Repetition is dtype-aware: 16 for bf16 P,
            #   8 for fp8 P, chosen so that k==split_idx lands exactly on the
            #   3/4 column boundary (:2429-2439).
            if k + 1 == SPLIT_IDX:                  # SPLIT_IDX = 4 * 96 // 128 = 3
                fence("view_async_tmem_store")
                # instruction_selection: tcgen05.wait::st.sync.aligned (:2324,
                #   2 static); extent: scalar.
                commit(p_pipe, stage)
                # instruction_selection: mbarrier.arrive.release.cta
                #   .shared::cta.b64 (:2325, 2 static); extent: scalar.  This is
                #   the EARLY publish: the MMA warp starts PV on 3/4 of P.
        fence("view_async_tmem_store")
        # instruction_selection: tcgen05.wait::st.sync.aligned (:2326, 2 static);
        #   extent: scalar.
        commit(p_last_pipe, stage)
        # instruction_selection: mbarrier.arrive.release.cta.shared::cta.b64
        #   (:2330, 2 static); extent: scalar.  The PV instruction sequence is
        #   already running and blocks on this barrier partway through.

        acquire(sm_stats_pipe.producer, stage, producer_phase)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:2331,
        #   2 static); extent: scalar.  Only the EMPTY half of this pipe is
        #   live; it is a credit on sScale_slot, nothing more.

        row_sum = add_reduce(rP_as_f32)
        # instruction_selection: add.rn.f32x2 (126 static); extent: an
        #   intra-thread packed tree over 128 elements.
        store_shared(sScale, stage * 256 + group_tidx, row_sum)
        # instruction_selection: st.shared.f32 (:2339, 2 static); extent: scalar.
        store_shared(sScale, stage * 256 + 128 + group_tidx, row_max)
        # instruction_selection: st.shared.f32 (:2340, 2 static); extent: scalar.
        if RETURN_TEMPERATURE_LSE:
            store_shared(sScaleTemp, stage * 128 + group_tidx, temp_row_sum)
            # instruction_selection: st.shared.f32 (:2347); extent: scalar.
        fence("view_async_shared")
        # instruction_selection: fence.proxy.async.shared::cta (:2348, 2 static);
        #   extent: scalar.
        release(s_pipe.consumer, stage)
        # instruction_selection: mbarrier.arrive.release.cta.shared::cta.b64
        #   (:2352, 2 static); extent: scalar.
```

```python
    # -----------------------------------------------------------------------
    # The fused epilogue (:2659-3020), running in the same warpgroup.
    # -----------------------------------------------------------------------
    def epilogue_step(qi_group, stage, ...):
        slot = qi_group & 1
        phase = (qi_group // 2) & 1
        wait(o_pipe.full, slot, phase)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:2716,
        #   2 static); extent: scalar.  The sibling acquire at :2692 emits
        #   nothing -- both static occurrences report :2716.

        # --- decode the packed qsplit words into the 2-deep caches ---------
        # Threads 0..Q_TOKENS_PER_GROUP-1 split each word once; everyone else
        # reads the result back after the barrier.
        if group_tidx < Q_TOKENS_PER_GROUP:
            word = load_shared(sQIdxMeta, qidx_meta_slot + group_tidx)
            # instruction_selection: ld.shared.u32 (:2725, 2 static); extent: scalar.
            store_shared(sQIdx,     slot * Q_TOKENS_PER_GROUP + group_tidx, word & 0x00FFFFFF)
            # instruction_selection: and.b32 (`_decode_q_idx_from_qsplit`, :253,
            #   5 static module-wide, 2 of them here) then st.shared.u32 (:2727,
            #   2 static); extent: scalar.
            store_shared(sSplitIdx, slot * Q_TOKENS_PER_GROUP + group_tidx, (word >> 24) & 0xFF)
            # instruction_selection: shr.u32 (`_decode_split_idx_from_qsplit`,
            #   :257, 2 static -- the `& 0xFF` folds into it) then st.shared.u32
            #   (:2728, 2 static); extent: scalar.  The decode
            #   happens once per group here; every per-store read below comes out
            #   of these two caches, never out of sQIdxMeta again.
        named_barrier_arrive_and_wait(epilogue_bar, index=stage)
        # instruction_selection: bar.sync, named, id = base + stage (:2729,
        #   2 static); extent: warpgroup.

        # --- read O out of TMEM in two 64-column passes -------------------
        rO = reg_tile([64], "f32")
        for col_pass in range(2):                   # written `unroll=1`; emitted STRAIGHT-LINE
            copy_t2r(O[slot][:, col_pass * 64 : col_pass * 64 + 64], rO[col_pass])
            # instruction_selection: the TMEM copy-partition address math at
            #   :2751 (and.b32 x8, add.s32 x8, shr.u32 x4, shl.b32 x4,
            #   cvt.u64.u32 x2, or.b64 x2, or.b32 x2), then
            #   tcgen05.ld.sync.aligned.16x256b.x8.b32
            #   (:2758, 4 static = 2 per warpgroup copy), 32 registers each;
            #   extent: half of this thread's 64 fp32 O values.  The two loads
            #   are ADJACENT in the emitted code -- the source's `unroll=1`
            #   annotation did not survive, and a port that keeps the loop
            #   rolled would issue one load in a loop instead of two in a row.

        # --- scale and store, four values at a time ------------------------
        for r in range(NUM_ROWS):                   # constexpr, unrolled
            for c4 in range(NUM_COLS // 4):         # constexpr, unrolled
                row = frag_row(r, c4)
                col = frag_col(r, c4) + col_pass * 64
                if row < 128:
                    tok        = row // QHEADPERKV
                    row_in_tok = row % QHEADPERKV
                    qi = qi_group * Q_TOKENS_PER_GROUP + tok
                    if qi < count_raw:
                        q_idx = load_shared(sQIdx, slot * Q_TOKENS_PER_GROUP + tok)
                        # instruction_selection: ld.shared.u32 (:2785, 32 static
                        #   = 16 per warpgroup copy); extent: scalar, ONE PER
                        #   128-BIT STORE.  This is not hoisted out of the c4
                        #   loop, and the port must not hoist it either.
                        split = load_shared(sSplitIdx, slot * Q_TOKENS_PER_GROUP + tok)
                        # instruction_selection: ld.shared.u32 (:2786, 32 static);
                        #   extent: scalar, one per store.
                        q_abs = q_batch_offset + q_idx
                        flat_row = (split * total_q * head_q
                                    + q_abs * head_q
                                    + head_kv_idx * QHEADPERKV
                                    + row_in_tok)
                        row_sum = load_shared(sScale, slot * 256 + row)
                        # instruction_selection: ld.shared.f32 (:2796, 32 static);
                        #   extent: scalar, one per store.
                        row_scale = rcp(select(row_sum == 0 or isnan(row_sum),
                                               1.0, row_sum))
                        # instruction_selection: rcp.approx.ftz.f32 (:2800, 32
                        #   static = 16 per warpgroup copy) preceded by
                        #   setp.ne.f32 (34 static) and selp.f32 (36 static);
                        #   extent: scalar, one reciprocal per 128-bit store.
                        #   The guard substitutes 1.0 for a zero or NaN row sum
                        #   -- an all-masked row -- before the reciprocal, not
                        #   after.
                        mul(o0..o3, o0..o3, row_scale, lanes=2)
                        # instruction_selection: mul.rn.f32x2 (64 static);
                        #   extent: two packed pairs per store.
                        fake_col = real_col_to_stg128_fake_col(col)
                        # instruction_selection: and.b32 + or.b32 (copy_utils
                        #   :854 and :858, 32 static each); extent: scalar index
                        #   math.  This is the INVERSE of the shared-memory
                        #   swizzle: it is what makes the four values contiguous
                        #   in global memory, and an oracle that ignores it sees
                        #   a shuffled row rather than a wrong one.
                        copy_r2g(Op[flat_row * 128 + fake_col], (o0, o1, o2, o3),
                                 cache="streaming")
                        # instruction_selection: st.global.cs.v4.f32 (:2597, 32
                        #   static = 16 per warpgroup copy); extent: 4 fp32 in
                        #   one 128-bit store.  The bf16/f16 partial path becomes
                        #   st.global.cs.v4.b32 of 8 packed halves (:2613-2615)
                        #   with `real_col_to_stg128_half_fake_col`; the fp8 path
                        #   becomes st.global.cs.v4.b32 of 16 packed bytes
                        #   (:2638, 8 static on the fp8-partial export) with
                        #   `real_col_to_stg128_fp8_fake_col`.  Three different
                        #   remaps, three different lane counts, one instruction
                        #   family.

        fence("view_async_tmem_load")
        # instruction_selection: tcgen05.wait::ld.sync.aligned (:2985, 2 static);
        #   extent: scalar.

        # --- LSE: one row per thread --------------------------------------
        tok_local = group_tidx // QHEADPERKV
        h_local   = group_tidx % QHEADPERKV
        if qi_group * Q_TOKENS_PER_GROUP + tok_local < count_raw:
            row_sum = load_shared(sScale, slot * 256 + group_tidx)
            # instruction_selection: ld.shared.f32 (:2991, 2 static); extent: scalar.
            row_max = load_shared(sScale, slot * 256 + 128 + group_tidx)
            # instruction_selection: ld.shared.f32 (:2992, 2 static); extent: scalar.
            lse = (row_max * softmax_scale_log2 + log2(row_sum)) * LN2
            # instruction_selection: lg2.approx.ftz.f32 (:2999, 2 static) plus
            #   fma.rn.f32 and mul; extent: scalar.  `-inf` replaces it when the
            #   row sum is zero or NaN.
            q_idx = load_shared(sQIdx, slot * Q_TOKENS_PER_GROUP + tok_local)
            # instruction_selection: ld.shared.u32 (:3001, 2 static); extent: scalar.
            split = load_shared(sSplitIdx, slot * Q_TOKENS_PER_GROUP + tok_local)
            # instruction_selection: ld.shared.s32 (:3003, 2 static); extent:
            #   scalar.  Note the SIGNED load here against the unsigned load of
            #   the same cache at :2786 -- the operand form follows how the
            #   value is consumed downstream, and the port has to match both.
            h_abs = head_kv_idx * QHEADPERKV + h_local
            store_global(Lp, (split, q_batch_offset + q_idx, h_abs), lse)
            # instruction_selection: st.global.f32 (:3005, 2 static); extent:
            #   scalar.  LSE is indexed in three dimensions, unlike O_partial,
            #   which is indexed through the flattened row.
            if RETURN_TEMPERATURE_LSE:
                temp_sum = load_shared(sScaleTemp, slot * 128 + group_tidx)
                lse_t = (row_max * lse_temperature_scale_log2 + log2(temp_sum)) * LN2
                store_global(LpT, (split, q_batch_offset + q_idx, h_abs), lse_t)
                # instruction_selection: lg2.approx.ftz.f32 + st.global.f32
                #   (:3013, :3015); extent: scalar.

        named_barrier_arrive_and_wait(epilogue_bar, index=stage)
        # instruction_selection: bar.sync, named, id = base + stage (:3017);
        #   extent: warpgroup.
        release(sm_stats_pipe.consumer, slot)
        # instruction_selection: mbarrier.arrive.release.cta.shared::cta.b64
        #   (:3019); extent: scalar.  This is the empty-half credit that lets
        #   softmax overwrite sScale_slot two groups later.
        release(o_pipe.consumer, slot)
        # instruction_selection: mbarrier.arrive.release.cta.shared::cta.b64
        #   (:3020); extent: scalar.
```

## Logical GEMM ownership

| GEMM | shape | A operand | B operand | accumulator | issued by | consumed by |
| --- | --- | --- | --- | --- | --- | --- |
| QK | 128 x 128 x 128 | `sQ[q_slot]`, SMEM, K-major, descriptor walked in PTX registers | `sK`, SMEM, K-major | `S[s_slot]`, TMEM f32 | warp 12 | softmax WG `s_slot & 1` |
| PV | 128 x 128 x 128 | `P[pv_slot]`, **TMEM**, address passed explicitly because a TMEM iterator reports 0 | `sV`, SMEM, MN-major | `O[pv_slot]`, TMEM f32 | warp 12 | the epilogue in the same softmax WG |

Both are `tcgen05.mma.cta_group::1`, `kind::f16` for bf16 operands and
`kind::f8f6f4` when the operand width is 8. 48 static MMA instructions in the
bf16/TMA-Q export. Neither GEMM ever accumulates across calls: `zero_init=True`
at every issue site, because a CTA owns one KV block and each Q group is
computed once.

## TensorMap ABI

| map | rank | box | swizzle | built where | notes |
| --- | --- | --- | --- | --- | --- |
| K | 3 | `(128, 128)` over `[total_k, 128, head_kv]` | 128B | launcher prologue | `cp.async.bulk.tensor.3d`; the KV-head axis stays in the descriptor |
| V | 3 | `(128, 128)` MN-major | 128B | launcher prologue | same, after the `[1,0,2]` swap |
| Q (TMA path) | 2 | `(QHEADPERKV, Q_LOAD_TILE)` | 128B | launcher prologue | `Q_LOAD_TILE` = 128 for fp8 Q, else 64 |
| Q (gather4 path) | 2 | `(box_x, 1)`, `box_x` = 128 for fp8 Q else 64 | 128B | **launcher prologue in the port; a host-built `uint8[128]` kernel argument upstream** | `INTERLEAVE_NONE`, `L2_PROMOTION_256B`, `OOB_FILL_NONE` |

The gather4 descriptor deserves the note that it is **a plain tiled map**:
`create_q_gather4_tma_desc` (`tma_utils.py:332-407`) is one
`cuTensorMapEncodeTiled` call with no gather-specific field. The gather-ness
lives entirely in the PTX instruction
(`cp.async.bulk.tensor.2d...tile::gather4`), which takes four independent row
coordinates against an ordinary rank-2 map. That is why the port can drop
upstream's descriptor-tensor ABI argument and encode the identical descriptor
in its own prologue like every other TIRx kernel in this repository, and why
`prepare_bench` never needs a device.

## Storage aliases and lifetimes

| alias | over | why it is safe |
| --- | --- | --- |
| `sQ_load` over `sQ` | the same bytes, two layouts | the MMA reads one monolithic K-major A tile while the loader writes `q_tokens x k_subtiles` boxes; the two views never disagree because the loader completes a stage before the MMA acquires it |
| `P` over the upper columns of `S` | `TMEM_S_TO_P = 128 - 128 * P_WIDTH / 32` columns in | the P store destroys the tail of S, which is safe **only because** `row_max` and `row_sum` have already been extracted from the register copy of S; the ordering is load-bearing, not incidental |
| `sKFp8` / `sVFp8` beside `sK` / `sV` | separate allocations, not aliases | the fp8 staging tiles and their bf16 destinations are live at the same time during the dequantization pass, so they cannot overlap; this is what takes the `bf16 Q + fp8 KV` gather4 build from 138240 to 171008 bytes -- exactly 16384 + 16384 more |

## Static specialization boundary

| decided at compile time | decided at runtime |
| --- | --- |
| `QHEADPERKV` and therefore the entire Q-load program, `Q_TOKENS_PER_GROUP`, `TOKENS_PER_GATHER4`, `GATHERS_PER_WARP`, `meta_iters`, `TOKENS_PER_WARP` | `work_capacity` (the grid extent), `num_heads_kv`, `seq_len_q`, all tensor extents and strides |
| every dtype: Q/K/V storage, `qk_dtype`, `pv_dtype`, `p_dtype`, `o_dtype`, and from them `K_FP8_TO_BF16`, `V_FP8_TO_BF16`, both `mma_kind` strings, `store_rep`, `TMEM_S_TO_P` | `head_kv_idx`, `row_linear`, `work_q_begin`, `work_q_count`, `batch_idx`, `kv_block_idx` -- the whole work item |
| `CAUSAL`, and through it the register budgets and `EX2_EMU_FREQ` | `count_raw`, `num_q_groups`, `diag_q_count`, `masked_tok_count`, `kv_valid_cols`, `causal_q_offset` |
| `RETURN_TEMPERATURE_LSE` (presence of the third output buffer) | `softmax_scale_log2`, `lse_temperature_scale_log2` |
| every stage count, `SPLIT_P_ARRIVE`, `SPLIT_IDX`, the TMEM column map, every named-barrier id and participant count | slot and phase cursors, which are derived from runtime group indices |

Shapes are **not** in the compile key upstream and are not in the port's
either: `to_cute_tensor` marks every layout dynamic, so a single binary per
compile key serves every shape.

## TIRx module and benchmark contract

- `get_kernel(**config)` specializes on `(qhead_per_kv, causal, dtype_mode,
  partial_dtype)` and removes the temperature output handle when it is absent,
  then attaches `("blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory")`.
- `prepare_data` reuses the split-atomic module's CSR and work-list builders and
  adds Q/K/V plus a **frozen** qsplit assignment: slots `0..degree-1` in CSR
  order per `(q_abs, head_kv)` group. Upstream assigns those slots with a device
  atomic in arrival order, so re-running the producer would hand this kernel a
  different permutation every call. Freezing an equivalent instance of the same
  contract is what makes the forward reproducible and therefore comparable
  element by element.
- Correctness compares `O_partial` and `LSE_partial` against MSA's own compiled
  kernel on identical frozen inputs, **masked to `split < degrees[q_abs, h]`**
  because both buffers arrive uninitialized and a query of degree `d` never
  touches slots `d..topk-1`.
- The benchmark is kernel-only against the compiled CuTe-DSL callable. Unlike
  the two preparation kernels, this one needs no counter rotation: it reads its
  inputs without mutating them and overwrites -- never accumulates into -- the
  partial slots it owns, so the hundredth launch does exactly the work the first
  one did.

## Instruction-selection summary

How placement, layout, shape and schedule select the emitted instruction:

- **Placement selects the copy family.** GMEM to SMEM is always TMA, never
  `cp.async`: `cp.async.bulk.tensor.3d` for the K and V tiles (the head axis is
  a descriptor mode), `cp.async.bulk.tensor.2d` for the TMA-Q sub-tiles, and
  `cp.async.bulk.tensor.2d...tile::gather4` when four independent rows have to
  be pulled into one box. TMEM to registers is `tcgen05.ld`, registers to TMEM
  `tcgen05.st`; SMEM to registers in the dequantization pass is a plain
  `ld.shared.v4.u32` because 16 fp8 values are exactly four words.
- **Layout selects the vector width, and the swizzle selects the index math.**
  The epilogue stores 128 bits per instruction in every partial dtype -- 4 fp32,
  8 halves or 16 fp8 bytes -- and each width needs its own inverse-swizzle
  column remap so the packed values land contiguously. The same 128-bit budget
  is why the shared metadata stores merge into `st.shared.v4.u32` and the
  `sRowMeta` reads into `ld.shared.v4.u32` / `ld.shared.v2.u32`.
- **Shape selects the TMEM access shape.** 128 fp32 S elements per thread arrive
  as four `32x32b.x32` loads; P leaves as four `32x32b.x16` stores with a
  repetition chosen so the 3/4 publish boundary falls on an instruction edge;
  64 fp32 O values arrive as two `16x256b.x8` loads of 32 registers each.
- **Schedule selects where the synchronization instruction goes.** The Q pipe
  sets `expect_tx` at producer acquire (`mbarrier.arrive.expect_tx`) because a
  stage is filled by several TMAs; the single-shot K and V barriers take their
  `expect_tx` in the prologue instead and their load warps issue a bare
  `mbarrier.arrive`. Any pipe half **whose arrival the MMA warp issues** is
  released with `tcgen05.commit` rather than `mbarrier.arrive`, so the release
  is ordered behind the MMA -- that covers the UMMA-produced `s`/`o` pipes and
  equally the softmax-produced `p`/`p_last` pipes, whose consumer release at
  :2129/:2130/:2182/:2183 is also a commit. The split-P
  handoff puts an `mbarrier.try_wait.parity` **inside** the PV instruction
  sequence, which is what lets PV start on three quarters of P.
- **`causal` selects the arithmetic mix.** `EX2_EMU_FREQ = 16` sends one element
  pair in sixteen through a degree-3 FFMA polynomial instead of MUFU: 112
  `ex2.approx.ftz.f32` and 48 `fma.rn.f32x2` of polynomial per warpgroup copy,
  per 128 elements. Turning causal off removes the polynomial entirely and
  changes three register budgets.
- **Counts are evidence, not derivation.** For `bf16_tmaq_qh16_fp32`, as
  counted by `.porting/sparse_atten_fwd/analyze_export.py`: 5737 instruction
  lines, 153 of them predicated, 5584 net. 248 `and.b32` at `mask.py:40`
  (the largest single `.loc` group), 48 `tcgen05.mma` in six 8-instruction
  chains, 224 `ex2.approx.ftz.f32`, 176 `fma.rn.f32x2` (128 softmax + 48
  polynomial), 164 `max.f32` (132 reduction + 32 polynomial clamp), 128
  `cvt.rn.bf16x2.f32`, 126 `add.rn.f32x2`, 64 `mul.rn.f32x2`, 32
  `st.global.cs.v4.f32`, 32 `rcp.approx.ftz.f32`, 26 `mbarrier.init.shared.b64`
  (24 from the six pipes plus `:907` and `:908`), 8
  `tcgen05.ld...32x32b.x32`, 8 `tcgen05.st...32x32b.x16`, 4
  `tcgen05.ld...16x256b.x8`. The
  `bf16_gather4_qh4_fp32` export runs 5952 instruction lines / 157 predicated /
  5795 net against 5737 / 153 / 5584 -- a delta of 215 lines -- and differs in
  about 25 opcodes -- 16 `...tile::gather4...cache_hint` and 8
  `cp.async.bulk.prefetch...gather4` replacing the four 2-D Q TMAs, one extra
  `bar.sync` per group (the gather4 body takes `load_wg_bar` three times per
  group at :1659/:1691/:1810 against the TMA path's two), a net +6
  `ld.shared.u32` (+8 at :1755, -2 at :1907), and roughly 180 more
  address-arithmetic instructions (`add.s32` 347->393, `mad.lo.s32` 3->10,
  `cvt.u32.u64` 0->32, and so on).

  What **is** identical between the two exports is the softmax and epilogue
  opcode profile, to the instruction: `ex2.approx` 224, `fma.rn.f32x2` 176,
  `max.f32` 164, `cvt.rn.bf16x2.f32` 128, `add.rn.f32x2` 126, `mul.rn.f32x2`
  64, `rcp.approx` 32, `st.global.cs.v4.f32` 32, `selp.b32` 263, `selp.f32` 36,
  `ld.shared.f32` 36, `sub.rn.f32x2` 32, `add.rm.f32x2` 16, `st.global.f32` 2,
  `lg2.approx` 2, and all 89 `tcgen05.*`. That is the measurement that says the
  two Q paths differ only in the load warpgroup -- stated over the subset it
  actually covers, not over the whole module.
