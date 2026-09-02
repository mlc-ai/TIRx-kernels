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

This design sketch documents a TIRx port of cuDNN Frontend's
python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100.py
FlashAttentionDSABackwardSm100.
-->

# cudnn_sm100_dsa_sparse_attention_backward: coarse WASP pipeline sketch

This non-executable design sketch describes the storage layout, warp roles,
pipelines, control flow, and PTX-level operations of
[`tirx_kernels/cudnn/dsa/sparse_attention_backward.py`](../../../tirx_kernels/cudnn/dsa/sparse_attention_backward.py)
and its private implementation package. That TIRx module is the authoritative
implementation.

The port covers **four device kernels**, all launched from one host entry
(`FlashAttentionDSABackwardSm100.__call__`, `dsa_bwd_sm100.py:254-607`) onto one
stream, in this order:

| # | kernel | source | threads | what it does |
| --- | --- | --- | --- | --- |
| 1 | `sum_OdO` | `:648-707` | 128 | per `(head, query)`: the delta `-sum_d(O*dO)` and the sink-folded log2-domain LSE |
| 2 | `bwd` | `:740-1110` | **640 (20 warps)** | the attention backward itself |
| 3 | `convert` | `:609-646` | `32 * num_threads_seq` | FP32 dKV workspace to element dtype |
| 4 | `sum_dSink` | `:709-738` | 32 | the attention-sink gradient |

`bwd` is where essentially all the time goes and where the whole warp
specialization lives; the other three are small and are sketched at the same
op level but much more briefly.

## Scope and instantiations

The upstream compile key is a 7-tuple (`_interface_sm100.py:170`):
`(dtype, head_dim, head_dim_v, num_head, block_tile, max_topk, has_topk_length)`.

Fixed for every specialization in this port:

| axis | value | why it is fixed |
| --- | --- | --- |
| `block_tile` | 64 | pinned at `_interface_sm100.py:94`; the public API's `block_tile` argument is ignored on SM100 |
| `head_dim_v` | 512 | derived, `512 if head_dim == 576 else head_dim` (`_interface_sm100.py:64`) |
| `batch_size` | 1 | the interface is flat/varlen; batch is folded into the token axis (`:96`) |
| cluster | `(1,1,1)` | `cluster_layout_vmnk` is all zeros (`:357`); `cta_group = ONE` (`:319`). No multicast, no DSMEM. |

In scope, i.e. the specializations this module compiles:

| axis | values | what changes in device code |
| --- | --- | --- |
| `head_dim` | 512 / 576 | `same_hdim_kv = head_dim == head_dim_v` (`:31`) gates **five** distinct code changes: the 64-wide `dKV4`/`dQ4` tail MMAs, the `sK_tail`/`sQT_tail` SMEM views, the second TMA-store atom `tma_atom_dQ_64`, a different TMEM alias map (below), and a different `t2r_dKV*_done` barrier assignment in both the MMA and reduce loops. It also changes which pipeline generation count the `mma_reduce_dKV` pipeline runs at: 2 commits per tile at 512, **3** at 576. |
| `num_head` | 64 (with d512) / 32 (with d576) | only the grid's `y` extent, `ceil_div(num_head, 64)`; the MMA M tile is always 64 heads |
| `dtype` | bf16 / fp16 | `element_dtype` for every SMEM operand and both quantize sites. `acc_dtype` stays fp32 unconditionally (`:52`). Changes `cvt.rn.bf16x2.f32` to `cvt.rn.f16x2.f32` and `mul.bf16x2` to `mul.f16x2`. |
| `has_topk_length` | True / False | selects the whole KV-validity program. **True (compact)**: `topk = mTopkLength[token_idx]` (`:796`), rows past `topk` are zero-filled in the ragged tile only. **False (non-compact)**: `topk = max_topk` statically, and every row tests `topk_idx >= 0` (`:1307-1313`). |
| `max_topk` | any | `tile_count = ceil_div(topk, 64)`. Also selects three launch constants by pure Python comparison: `sum_OdO_block_q = 40 if max_topk == 1024 else 41` (`:56`), and `block_seq`/`num_threads_seq = 4 if max_topk == 2048 else 32` (`:568-570`). |

Out of scope, with the predicate that excludes each:

- **`head_dim not in (512, 576)`** -- asserted at `_interface_sm100.py:63`.
- **fp8 / mxfp8 element dtypes** -- `__init__` raises (`:47-49`).
- **`batch_size > 1`** -- the host wrapper always passes 1.
- **the SM90 sibling** `FlashAttentionDSABackwardSm90` (`dsa_bwd_sm90.py`) -- a
  separate class with a different structure (three kernel objects, two MMA
  warpgroups) selected by `api.py:173-206`.
- **the forward pass** -- production is FlashMLA, C++, outside this repo. The
  kernel only consumes its `out` and `lse`.
- Tile (`Tx`) primitives are out of scope everywhere.

## The line-info export this sketch is annotated from

Every `instruction_selection` annotation below is read out of a line-info PTX
export, not out of the source text. Three exports, one per structurally distinct
in-scope compile key, are preserved under
`.porting/dsa_bwd_sm100/writer_source_export/<name>/`:

| name | key | `.loc` lines | key-op notes |
| --- | --- | --- | --- |
| `d512_bf16_len` | d512, h64, bf16, compact | 2084 | the unqualified counts below |
| `d576_bf16_len` | d576, h32, bf16, compact | 2852 | adds the tail MMAs and the third dKV generation |
| `d512_bf16_nolen` | d512, h64, bf16, non-compact | 2674 | the `topk_idx >= 0` arm on every row, not just the ragged tile |

Each export contains all four kernels as four `.visible .entry` points, because
`cute.compile` compiles the whole `__call__`, and the four carry `.reqntid`
`8,16,1` / `640,1,1` / `32,32,1` / `32,1,1` respectively.

Approximate instruction-line counts for `d512_bf16_len`: `sum_OdO` ~720,
`bwd` ~4900-5100, `convert` ~70, `sum_dSink` ~255. These are order-of-magnitude
context only, not evidence: the count moves by a few percent depending on whether
`ret`, `.pragma`, labels and multi-line operand continuations are counted, and
nothing in this sketch depends on them. **Every per-op count below is a count of
one named instruction at one `.loc`, which is exact and recheckable**; those are
the figures to verify.

Reproduce any export with:

```bash
mkdir -p .porting/dsa_bwd_sm100/writer_source_export/<name>
CUDNN_FRONTEND_PATH=<cudnn-frontend checkout> \
CUTE_DSL_NO_CACHE=1 CUTE_DSL_KEEP=ptx CUTE_DSL_LINEINFO=1 \
CUTE_DSL_DUMP_DIR=.porting/dsa_bwd_sm100/writer_source_export/<name> \
python .porting/dsa_bwd_sm100/export_ptx.py <head_dim> <bf16|fp16> <len|nolen>
```

The DSA path has no AOT or on-disk cache, so `CUTE_DSL_NO_CACHE=1` plus a fresh
process is enough. The export's single `.file` directive names the **checkout**
source, which is what makes these annotations describe the revision this port
follows rather than the older `cudnn` wheel that is also installed (see
`.porting/dsa_bwd_sm100/source_overview.md`).

Two attribution quirks to read the tables with:

- Some inlined code is attributed to **`:254`**, the `__call__` decorator line,
  rather than to its own source line -- but not all of it, so this is a fact to
  check per site rather than a rule. Landing at `:254`: the `atom.global.add.*`
  sites, the `exit`, and the `cvt` inside `quantize`/`element_dtype()` when
  reached from `convert` or `store_dQ_64`. Keeping their own `.loc` despite
  being inlined helpers: `t2r_dKV` (`:2305`), `store_dQ` (`:2533`, `:2543`),
  `_copy_kv_row` (`:1229`) and `quantize` (`:2624`, `:2625`).
- `is_first`/`full_tiles` specialization duplicates whole helper bodies, so a
  static count is often `call_sites x per_call`, not a dynamic count. The KV
  gather's 96 `cp.async.cg` is exactly `3 call sites x 16 rows x 2 groups`.

## Pipeline at a glance

`bwd`: 20 warps, 640 threads, **one CTA per query token**. The 64 attention heads
are the MMA `M` dimension, so a token's top-k KV rows are gathered once and
amortized across every head. Role selection is an `if/elif` chain
(`:814`, `:909-1110`) whose **order is not warp-id order**: `17, 16, 4..7,
8..15, 0..3, else`. The port preserves that order, because it is what the
dispatch compare sequence lowers from.

| Warps | Role-local tile program | Main publication / reuse edges |
| --- | --- | --- |
| 0..3 | `load_KV`: per tile, lane 0 of each warp reads 16 top-k indices and broadcasts them; all 32 lanes then cp.async-gather 64 KV rows into `sK` | produces `load_mma_K` |
| 4..7 | `compute`: T2R of `S` and `dP`, the softmax recompute, stmatrix-transpose of `P` and `dS` into SMEM, and after the loop the whole dQ TMA-store epilogue. Warp 4 is also the TMEM allocator and the only TMA-store issuer. | consumes `mma_compute_S`, `mma_compute_dP`, `load_compute_LSE`, `load_compute_sum_OdO`, `mma_compute_dQ`; produces `compute_mma_P`, `compute_mma_dS`, `compute_tmastore_dQ` |
| 8..15 | `reduce_dKV`, two warpgroups splitting the 64 KV rows: T2R each dKV sub-tile out of TMEM, **release the WAR barrier**, then add into the FP32 dKV workspace with 4-wide global atomics | consumes `mma_reduce_dKV`; arrives on `t2r_dKV01_done`, `t2r_dKV4_done`, `t2r_dKV23_done`, `tmem_dealloc` |
| 16 | `mma`: issues **every** tcgen05 MMA in the kernel | consumes `load_mma_QdO`, `load_mma_K`, `compute_mma_P`, `compute_mma_dS`; produces `mma_compute_S`, `mma_compute_dP`, `mma_compute_dQ`, `mma_reduce_dKV` |
| 17 | `load`: prefetches the three TMA descriptors, then one TMA for `Q`, one for `dO`, and one cp.async each for `LSE` and `sum_OdO`. **No loop** -- a CTA owns one query token. | produces `load_mma_QdO`, `load_compute_LSE`, `load_compute_sum_OdO` |
| 18..19 | idle; execute only `setmaxnreg.dec 40` and fall out | none |

**Q/dO are not pipelined.** One CTA owns one query token, so `load_mma_QdO` is a
single-shot barrier acquired once before the loop (`:1490`) and released once
after it (`:1763`); the load warp issues its copies and retires. All pipelining
runs along the **top-k tile axis**.

**The tile loop runs backwards**, `tile_index = tile_count - 1` down to `0`, in
all four looping roles (`:1343`, `:1493`, `:1870`, `:2190`). The
**first processed** tile is therefore the highest-index one, which is the ragged
one, and `full_tiles = (topk % 64) == 0` (`:1344`, `:2194`) is what decides
whether that first tile needs the bounds-checked program at all. Reversing this
loop would move the ragged tile to the wrong end.

**Every pipeline is single-stage except `mma_reduce_dKV`, which has 2**
(`:159-172`). That one is the only place two dKV generations are in flight, and
it is the reason the three `t2r_dKV*_done` named barriers exist: dKV2/dKV3 alias
dKV0/dKV1 in TMEM, and a 2-deep pipeline does not order the next generation's
writes against the previous generation's reads.

**dKV never reaches global memory as element dtype from this kernel.** It is
accumulated in FP32 into a workspace by atomics and converted by kernel 3. That
also makes the result order-nondeterministic, which the correctness contract
below depends on.

## Primitive vocabulary

Structural operations do not compute values:

```python
tile(...)        # declare storage, dtype, logical shape, placement
view(...)        # change logical indexing without moving values
alias(...)       # declare exact storage aliasing and non-overlap lifetime
slice(...)       # select a logical interval
reg_tile(...)    # declare a role-local register tile
tensormap(...)   # declare a TMA descriptor and its encode fields
tmem_cols(...)   # name a tensor-memory column range
```

Every SMEM object below is a **byte offset into one linear pool**. There are no
layout objects anywhere in this sketch; where the source carries a CuTe swizzled
`ComposedLayout`, this sketch names the swizzle as an instruction/descriptor
immediate and the addressing as an explicit index function.

Copies always state their storage direction:

```python
copy_g2s_tma(desc, coords, dst_off, bar)     # global -> shared, TMA
copy_s2g_tma(desc, coords, src_off)          # shared -> global, TMA
copy_g2s_async(dst_off, src_ptr, bytes, pred_bytes)  # global -> shared, cp.async
copy_t2r(src_cols, dst_reg, shape, rep)      # tensor memory -> register
copy_r2s_stmatrix(src_reg, dst_off)          # register -> shared, transposing
copy_r2s(src_reg, dst_off)                   # register -> shared, plain
load_global(buf, idx) -> reg
store_global(buf, idx, reg)
load_shared(off) -> reg
store_shared(off, reg)
atomic_add_global(ptr, values)               # values is a 1/2/4-wide f32 vector
prefetch_descriptor(desc)
zero_shared(dst_off, bytes)
```

The computational vocabulary:

```python
fill(dst, value)
cast(dst, src)                       # f32 -> bf16/f16, bf16 -> f32
add(dst, lhs, rhs, lanes=1)          # lanes=2 is one packed f32x2 op
mul(dst, lhs, rhs, lanes=1)
fma(dst, a, b, c, lanes=1)
max(dst, lhs, rhs)
exp2(dst, src, ftz=False)
log2(dst, src)
gemm(dst, a, b, accumulate)          # one tcgen05 MMA issue
warp_reduce_add(dst, src, group)     # butterfly shuffles
shuffle_broadcast(dst, src, lane)
```

Schedule operations: `pipe`, `init_pipe`, `acquire`, `wait`, `commit`,
`release`, `expect_tx`, `fence_tmem_load`, `fence_proxy_shared`,
`fence_view_async_shared`, `cp_async_commit`, `cp_async_wait`,
`tma_store_commit`, `tma_store_wait`, `named_barrier`, `elect`,
`set_register_budget`, `tmem_alloc`, `tmem_retrieve`, `tmem_dealloc`,
`exit_cta`, and cursor (`slot`, `phase`) updates.

`add(..., lanes=2)` is one packed two-lane f32 operation with two ordered
results, not shorthand for two scalar adds. There are deliberately no primitives
named `attention`, `softmax`, `backward`, `epilogue`, `reduce_dkv`, or
`store_dq`.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

@specialize(
    HEAD_DIM,                  # 512 or 576
    HEAD_DIM_V=512,
    NUM_HEAD,                  # 64 with d512, 32 with d576
    BLOCK_TILE=64,             # KV rows per tile; also the MMA M/N tile
    MAX_TOPK,
    HAS_TOPK_LENGTH,           # bool
    ELEM,                      # "bf16" | "f16"
    ACC="f32",
)
# Derived at trace time, exactly as `__init__` derives them (:29-61):
SAME_HDIM   = (HEAD_DIM == HEAD_DIM_V)      # :31  -- True only at d512
HEAD_DIM_MAIN = (HEAD_DIM // 128) * 128     # :36  -- always 512
N_MAIN_TILES  = HEAD_DIM_MAIN // 128        # 4 dKV/dQ sub-tiles of 128
HAS_TAIL      = not SAME_HDIM               # the fifth, 64-wide sub-tile
SUM_ODO_BLOCK_Q = 40 if MAX_TOPK == 1024 else 41       # :56
BLOCK_SEQ       = 4  if MAX_TOPK == 2048 else 32       # :568
NUM_THREADS_SEQ = 4  if MAX_TOPK == 2048 else BLOCK_SEQ  # :570

def host_entry(
    q,            # ELEM [S_q, H, D],   input
    kv,           # ELEM [S_kv, D],     input -- K and V share this buffer
    out,          # ELEM [S_q, H, D_v], input  (the forward's output)
    dout,         # ELEM [S_q, H, D_v], input
    lse,          # f32  [S_q, H],      input -- KV-only LSE, sink EXCLUDED
    attn_sink,    # f32  [H],           input -- -inf disables the sink
    topk_idxs,    # i32  [S_q, MAX_TOPK], input -- global KV rows, -1 invalid
    topk_length,  # i32  [S_q],         input, HAS_TOPK_LENGTH only
    dq,           # ELEM [S_q, H, D],   output -- fully overwritten
    dkv,          # ELEM [S_kv, D],     output -- MUST be zero on entry
    d_sink,       # f32  [H],           output -- MUST be zero on entry
    ws_lse_odo,   # u8 [1, H, roundup8(S_q), 8],  workspace, MUST be zero
    ws_dkv,       # u8 [1, 1, roundup8(S_kv), roundup8(D)*4], workspace, zero
    softmax_scale,  # f32, defaults to 1/sqrt(D)
):
    # The four launches, one stream, in this order (:493-607).
    launch(sum_OdO,   grid=(ceil_div(S_q, SUM_ODO_BLOCK_Q), H, 1), block=128)
    launch(bwd,       grid=(S_q, ceil_div(H, 64), 1),              block=640,
           smem=SHARED_BYTES, min_blocks_per_sm=1)
    launch(convert,   grid=(ceil_div(S_kv, BLOCK_SEQ), 1, 1),
           block=32 * NUM_THREADS_SEQ)
    launch(sum_dSink, grid=(ceil_div(S_q, 256), H, 1),             block=32)
    # instruction_selection: none; host-side launch only.
    # Only sum_OdO -> bwd and bwd -> {convert, sum_dSink} are real
    # dependencies; the stream order supplies both.

# The two workspace views (:217-240). Both are re-typed windows into one u8
# allocation; `scaled_lse` starts exactly `H * roundup8(S_q) * 4` bytes after
# `sum_OdO`, so they are one allocation holding two f32 planes, NOT an
# interleaved pair despite the 8-byte-per-entry shape the host allocates.
sum_OdO_ws    = view(ws_lse_odo, "f32", [H, S_q], stride=[1, H])   # head-contiguous
scaled_lse_ws = view(ws_lse_odo, "f32", [H, S_q], stride=[1, H],
                     byte_offset=H * roundup8(S_q) * 4)
dkv_acc_ws    = view(ws_dkv, "f32", [S_kv, D], stride=[D, 1])      # D-contiguous

# ===========================================================================
# Kernel 1 of 4: sum_OdO  (:648-707)
#   grid (ceil_div(S_q, SUM_ODO_BLOCK_Q), H, 1), block [8, 16, 1] = 128 threads
#   tidx indexes head-dim groups (8 of them), tidy indexes queries (16).
# ===========================================================================

def sum_OdO():
    bidx, bidy, bidz = block_id()        # q block, head, batch
    tidx, tidy, _ = thread_id_3d()

    for idx_q_t in range(tidy, SUM_ODO_BLOCK_Q, 16):      # :667, 3 iterations
        idx_q = idx_q_t + SUM_ODO_BLOCK_Q * bidx
        if idx_q >= S_q:
            continue

        # Each thread strides over the head dim in 4-element groups.
        acc = f32(0.0)
        for idx_d in range(tidx, HEAD_DIM_V // 4, 8):     # :678, 16 iterations
            o_frg  = load_global(out,  (bidy, idx_d * 4, idx_q), width=4)
            do_frg = load_global(dout, (bidy, idx_d * 4, idx_q), width=4)
            # instruction_selection: ld.global.v2.b32 (:679, :680; 42 static
            #   across the whole kernel); extent: vector of 4 ELEM = 64 bits.
            #   `sum_OdO_elem_per_load = 4` (:59) is what makes this a v2.b32
            #   and not four scalar loads.
            prod = mul(o_frg, do_frg)
            # instruction_selection: mul.bf16x2 (:681, 42 static); extent:
            #   2 packed lanes x 2 per 4-element fragment. fp16 emits
            #   mul.f16x2.
            acc = add(acc, reduce_add(cast(prod, "f32")))
            # instruction_selection: cvt.f32.bf16 (:682, 21 static), then BOTH
            #   add.rn.f32.bf16 (:683, 63 static) and add.f32 (:683, 42 static);
            #   extent: scalar each, over the 4-element fragment.  The site is a
            #   real cvt plus two add families, not one fused widening add.

        acc = warp_reduce_add(acc, group=8)
        # instruction_selection: shfl.sync.bfly.b32 (:685, 9 static = 3 outer
        #   iterations x log2(8)); extent: 3-step butterfly over 8 lanes.

        if tidx == 0:
            lse_bhq = load_global(lse, (bidy, idx_q))
            # instruction_selection: ld.global.b32 (:688, 3 static); extent:
            #   scalar, lane 0 of each 8-lane group only.  `.b32`, not `.f32`:
            #   the export never widens or types these loads.
            sink_bh = load_global(attn_sink, (bidy,))
            # instruction_selection: ld.global.b32 (:693, 3 static); extent:
            #   scalar, lane 0 of each 8-lane group.

            # Fold the sink into the denominator with a max shift, in log2
            # units. `LSE_scale = -log2(e)` and `sum_OdO_scale = -1.0` are
            # passed in as f32 kernel arguments (:488-489), not folded at
            # trace time.
            lse_log2  = mul(lse_bhq, log2_e)
            # instruction_selection: mul.f32 (:696, 3 static); extent: scalar.
            sink_log2 = mul(sink_bh, log2_e)
            # instruction_selection: mul.f32 (:697, 3 static); extent: scalar.
            m         = max(lse_log2, sink_log2)
            # instruction_selection: max.f32 (:698, 3 static); extent: scalar.
            s         = add(exp2(sub(lse_log2, m)), exp2(sub(sink_log2, m)))
            # instruction_selection: sub.f32 (:699, 6 static) and add.f32 (:699,
            #   3 static) around the exponentials; extent: scalar each.
            # instruction_selection: ex2.approx.f32 (:699, 6 static = 3 outer
            #   iterations x 2 calls); extent: scalar.  NOTE the absence of
            #   `.ftz`: this call site does not pass fastmath, unlike the
            #   compute warp's exp2 in `bwd`, which does and emits
            #   ex2.approx.ftz.f32.  The port must not unify the two.
            scaled = neg(add(m, log2(s)))
            # instruction_selection: neg.f32 (:701, 3 static); extent: scalar.
            # instruction_selection: a SOFTWARE log2 polynomial (:700) --
            #   fma.rn.f32 x36, add.f32 x9, mul.f32 x9, cvt.rn.f32.s32 x3;
            #   extent: ~19 FP32 instructions per dynamic call, 57 static over
            #   the 3 peeled outer arms.  `lg2` appears ZERO times in any
            #   export: `cute.math.log2` without fastmath does NOT lower to
            #   lg2.approx.f32, and the port must not substitute a fast log2
            #   here.  Contrast :699 above, which does get a hardware ex2.
            if lse_bhq == +inf:
                scaled = -inf                       # :703-704
            # instruction_selection: setp.eq.f32 (:703, 3 static) then selp.f32
            #   (:254, 3 static); extent: scalar, one per peeled outer arm --
            #   the compiler predicates this, it is not a branch.  The other 12
            #   selp.f32 in this kernel belong to the log2 polynomial at :700
            #   (its denormal rescale and zero/-inf special cases), not here.

            store_global(sum_OdO_ws,    (bidy, idx_q), mul(sum_OdO_scale, acc))
            # instruction_selection: mul.f32 (:689, 3 static); extent: scalar --
            #   the `sum_OdO_scale = -1.0` negation is a real multiply, not a
            #   sign flip folded into the store.
            store_global(scaled_lse_ws, (bidy, idx_q), scaled)
            # instruction_selection: st.global.b32 (:706, 3 static) and
            #   st.global.b32 (:707, 3 static); extent: scalar.  `.b32`, not
            #   `.f32`.

# ===========================================================================
# Kernel 2 of 4: bwd  (:740-1110)
# ===========================================================================

def bwd():
    token_idx, head_block_idx, _ = block_id()
    tidx = thread_id()
    batch_idx = thread_id_z()      # :790 shadows the block_id z with the
                                   # thread_id z.  blockDim.z == 1 and
                                   # batch_size == 1, so both are 0; the port
                                   # keeps the value 0 rather than the quirk.
    warp = warp_uniform(tidx // 32)
    # instruction_selection: shfl.sync.idx.b32 (:791, 1 static); extent: warp
    #   broadcast.  This is a real instruction and it is what lets every role
    #   predicate below lower as warp-uniform.

    # -----------------------------------------------------------------------
    # Per-token top-k length, and the empty-row early exit
    # -----------------------------------------------------------------------
    if HAS_TOPK_LENGTH:
        topk = load_global(topk_length, (token_idx,))
        # instruction_selection: ld.global.b32 (:796, 1 static); extent: scalar, all
        #   threads.  CTA-uniform by construction: one value per query token.
    else:
        topk = MAX_TOPK                                   # :798, compile-time

    if topk <= 0:                                         # :805
        # Zero this token's whole dQ tile with all 640 threads and leave.
        # This runs BEFORE the SMEM allocator, before every pipeline init and
        # before the TMEM allocation, and that ordering is load-bearing: a CTA
        # that exited after arriving on a pipeline init would strand the rest.
        for linear in range(tidx, HEAD_DIM * BLOCK_TILE, 640):   # :806
            head = head_block_idx * BLOCK_TILE + linear // HEAD_DIM
            if head < NUM_HEAD:
                store_global(dq, (linear % HEAD_DIM, head, token_idx), ELEM(0))
                # instruction_selection: st.global.b16 (:811); extent: scalar,
                #   strided over 640 threads.  Not vectorized.
        exit_cta()
        # instruction_selection: exit (attributed to :254, 1 static); extent:
        #   whole CTA.

    # -----------------------------------------------------------------------
    # Descriptor prefetch, storage, pipelines
    # -----------------------------------------------------------------------
    if warp == 17:                                        # :814
        prefetch_descriptor(desc_Q)
        prefetch_descriptor(desc_dO)
        prefetch_descriptor(desc_dQ)
        # instruction_selection: prefetch.tensormap (:815-817); extent: 3
        #   scalar issues, load warp only.  Note desc_dQ_64 is NOT prefetched
        #   even in the d576 build.

    # --- one linear SMEM pool -------------------------------------------
    # The source declares this as `@cute.struct SharedStorage` (:448-470) and
    # allocates it with `utils.SmemAllocator` (:819-820), which reads like a
    # static __shared__ block.  It is not: the export carries
    #   .extern .shared .align 1024 .b8 __dynamic_shmem__0[]
    # The matching TIRx form is a dynamic pool, never a static shared array.
    #
    # Field order is declaration order.  `sQ`, `sK`, `sdO` are 1024-byte aligned
    # (`buffer_align_bytes`, :156) and the rest 128-byte (`non_tma_align_bytes`,
    # :157), and the struct itself pads up to its own 1024-byte alignment.
    #
    # Exact byte map, measured (probe/probe_smem_layouts.py), d512 / d576:
    #   mbarriers (22 x i64)         0        176 B
    #   tmem_holding_buf (i32)     176          4 B
    #   sQ                        1024   /   1024      65536 /  73728 B
    #   sK                       66560   /  74752      65536 /  73728 B
    #   sdO                     132096   / 148480      65536 /  65536 B
    #   sP                      197632   / 214016       8192 /   8192 B
    #   sdS                     205824   / 222208       8192 /   8192 B
    #   sLSE                    214016   / 230400        256 /    256 B
    #   sSum_OdO                214272   / 230656        256 /    256 B
    #   raw end                 214528   / 230912
    #   padded to 1024         215040   / 231424   <- SharedStorage.size_in_bytes()
    #
    # The source's assert is `227 * 1024`, which IS 232,448 -- the same number as
    # the SM100 hardware cap, not a second tighter bound (:446, :472-474).
    # Headroom: 17,408 B at d512 but only **1,024 B** at d576.  The d576 build is
    # one 1 KiB buffer away from not launching, so nothing may be added to this
    # pool without re-measuring.
    smem = tile("smem", "u8", [SHARED_BYTES], byte_offset=0, alignment=1024)

    # 22 mbarrier slots: (stage x 2) per pipeline, 9 pipelines at 1 stage plus
    # mma_reduce_dKV at 2.  full/empty pairs.
    mbar_load_mma_QdO        = view(smem, "i64", [1, 2])     # :450
    mbar_load_mma_K          = view(smem, "i64", [1, 2])     # :451
    mbar_load_compute_LSE    = view(smem, "i64", [1, 2])     # :452
    mbar_load_compute_sumOdO = view(smem, "i64", [1, 2])     # :453
    mbar_mma_compute_S       = view(smem, "i64", [1, 2])     # :454
    mbar_mma_compute_dP      = view(smem, "i64", [1, 2])     # :455
    mbar_mma_compute_dQ      = view(smem, "i64", [1, 2])     # :456
    mbar_compute_mma_P       = view(smem, "i64", [1, 2])     # :457
    mbar_compute_mma_dS      = view(smem, "i64", [1, 2])     # :458
    mbar_mma_reduce_dKV      = view(smem, "i64", [2, 2])     # :459  <- 2 stages
    tmem_holding             = view(smem, "i32", [1])        # :460

    sQ       = view(smem, ELEM, [64 * HEAD_DIM],   alignment=1024)  # :461
    sK       = view(smem, ELEM, [64 * HEAD_DIM],   alignment=1024)  # :465
    sdO      = view(smem, ELEM, [64 * HEAD_DIM_V], alignment=1024)  # :466
    sP       = view(smem, ELEM, [64 * 64],         alignment=128)   # :467
    sdS      = view(smem, ELEM, [64 * 64],         alignment=128)   # :468
    sLSE     = view(smem, "f32", [64],             alignment=128)   # :469
    sSum_OdO = view(smem, "f32", [64],             alignment=128)   # :470

    # --- aliases: the same bytes under a different operand mapping -------
    # The source spells each of these as a `recast_ptr` + `make_tensor` pair
    # (:862-903) carrying a CuTe swizzled layout.  This port carries the same
    # bytes with an explicit base offset, element shape and stride tuple, plus
    # the swizzle immediate the matrix descriptor needs.
    #
    # EVERY buffer and every alias in this kernel uses the SAME swizzle,
    # S<3,4,3> -- the 128-byte swizzle, 16-byte atoms, 8-row period.  That is
    # measured, not assumed (probe/probe_smem_layouts.py builds all sixteen
    # layouts and prints the swizzle of each).  So the swizzle immediate is a
    # constant of this kernel and the only thing that varies per alias is the
    # base offset and the shape:stride below.
    #
    # `outer` shapes are in ELEMENTS, from the same probe.  The two shapes that
    # matter are the K-major form ((64,16),1,(4,8),1):((64,1),0,(16,4096),0)
    # and the MN-major form (((64,2),16),4,4,1):(((1,4096),64),8192,1024,0):
    # they differ in which axis carries stride 1, which is exactly what selects
    # the descriptor's leading/stride byte offsets.
    #
    # alias      base            shape:stride (elements)                    used as
    sQT      = alias(sQ,  0)     # (((64,2),16),4,4,1):(((1,4096),64),8192,1024,0)
                                 #   A of dKV += Q^T @ dS, MN-major      # :880-881
    sQT_tail = alias(sQ,  0)     # ((64,16),9,4,1):((1,64),4096,1024,0), block 8
                                 #   = elements 8*4096.. ; A of dKV4     # :896-899
    sV       = alias(sK,  0)     # K-major form; identical to sK when SAME_HDIM,
                                 #   so :864 is a no-op alias there      # :864
    sK_2     = alias(sK,  0)     # MN-major form; A of dQ = K @ dS^T     # :886-887
    sK_tail  = alias(sK,  0)     # ((64,16),9,4,1) block 8; A of dQ4     # :892-894
    sdQ      = alias(sK,  0)     # ((64,2),(8,8),(1,1)):((1,4096),(64,512),(0,0))
                                 #   epilogue 128x64                     # :871-872
    sdQ4     = alias(sK,  0)     # ((64,1),(8,8),(1,1)):((1,0),(64,512),(0,0))
                                 #   epilogue 64x64 -- same form as       # :902-903
                                 #   sP_store / sdS_store
    sdOT     = alias(sdO, 0)     # MN-major form; A of dKV += dO^T @ P   # :883-884
    sP_store = alias(sP,  0)     # ((64,1),(8,8),(1,1)):((1,0),(64,512),(0,0))
                                 #   stmatrix destination, column-major  # :866
    sdST     = alias(sdS, 0)     # ((64,16),1,4,1):((1,64),0,1024,0)
                                 #   B of dQ = K @ dS^T, MN-major        # :877-878
    sdS_store= alias(sdS, 0)     # ((64,1),(8,8),(1,1)):((1,0),(64,512),(0,0))
                                 #   stmatrix destination, column-major  # :869
    # Every alias starts at its parent's base (offset 0 within it): the source's
    # `recast_ptr` reinterprets the same pointer, and the tail views reach their
    # 64-wide block through the block index (8), not through a base offset.

    # --- the index functions used by the role bodies below ---------------
    # Named here so the bodies can stay readable; each is plain arithmetic over
    # the linear pool, which is what replaces the source's layout objects.
    # UNITS: every offset function below returns ELEMENTS, and every TMA
    # coordinate is in ELEMENTS.  Mixing element offsets with byte offsets, or
    # element coordinates with tile indices, is the easiest way to get a
    # plausible-looking wrong address here, so the unit is restated at each use.
    #
    # `sK_off(row, group, lane)` -- the gathered KV row map -- is defined with
    # the gather itself, below, because its strides are the non-obvious part.
    def row(i):                      # the S/dP fragment's row coordinate
        return tTR_c[i].mode0        #   :1880-1881, :1926-1931
    def row_coord(i):                # the dKV fragment's row coordinate
        return tTR_cdKV[i].mode1     #   :2199, :2213
    def sdQ_elem(dp, j):             # ELEMENTS into the linear sdQ region
        return (dp % 64) + (dp // 64) * 4096 + 64 * j
                                     #   :2540-2543.  Two compositions, so state
                                     #   the RESULT rather than either half:
                                     #   `make_ordered_layout((128,64),(0,1))`
                                     #   is (128,64):(1,128), and sdQ itself is
                                     #   ((64,2),(8,8),(1,1)):((1,4096),(64,512),
                                     #   (0,0)) -- COL_MAJOR, because
                                     #   LayoutEnum.from_tensor(mdQ) reads mdQ's
                                     #   mode-0 stride of 1 (:296-297).  Their
                                     #   composition measures ((64,2),64):
                                     #   ((1,4096),64), which is the formula
                                     #   above.  Verified against crd2idx by
                                     #   probe/probe_dq_store_map.py.  Deriving
                                     #   it from a ROW_MAJOR sdQ instead gives
                                     #   64*dp + j, which is wrong in both
                                     #   terms.
    def coords_i(i):                 # the dQ TMA store's coordinate, ELEMENTS
        return (i * 128, head_block_idx * 64, token_idx)   # :1977-1980
                                     #   mode 0 steps 128 head dims per round;
                                     #   mode 1 is head_block_idx * 64, an
                                     #   element offset, not a tile index

    # --- pipelines (:822-852) --------------------------------------------
    # producer_arrivals -> consumer_arrivals, in THREADS.  These are the
    # CooperativeGroup sizes at :2711-2866, and they are exactly the count
    # operands of the 22 `mbarrier.init.shared.b64` in the export; getting one
    # wrong deadlocks silently.
    #
    # Read the 1s carefully: the source writes `len([self.mma_warp_id])`, which
    # is `len([16])` == **1**, a single-Thread agent -- one elected lane
    # arrives, not a 32-thread warp.  Every UMMA side is 1 because
    # `tcgen05.commit` is one arrival.  32 appears only where a whole warp
    # arrives thread-by-thread (the load warp's two cp.async sites); 128 and
    # 256 are the 4- and 8-warp consumers arriving per thread.
    pipe_load_mma_QdO  = pipe(mbar_load_mma_QdO, 1, tma(tx=Q_BYTES+dO_BYTES),
                              producer=1,   consumer=1)     # :2714 TmaUmma
    pipe_load_mma_K    = pipe(mbar_load_mma_K, 1, async_umma,
                              producer=128, consumer=1)     # :2726 AsyncUmma
    pipe_load_cmp_LSE  = pipe(mbar_load_compute_LSE, 1, cp_async,
                              producer=32,  consumer=128)   # :2743 CpAsync
    pipe_load_cmp_sOdO = pipe(mbar_load_compute_sumOdO, 1, cp_async,
                              producer=32,  consumer=128)   # :2760 CpAsync
    pipe_mma_cmp_S     = pipe(mbar_mma_compute_S, 1, umma_async,
                              producer=1,   consumer=128)   # :2774 UmmaAsync
    pipe_mma_cmp_dP    = pipe(mbar_mma_compute_dP, 1, umma_async,
                              producer=1,   consumer=128)   # :2802
    pipe_mma_cmp_dQ    = pipe(mbar_mma_compute_dQ, 1, umma_async,
                              producer=1,   consumer=128)   # :2788
    pipe_cmp_mma_P     = pipe(mbar_compute_mma_P, 1, async_umma,
                              producer=128, consumer=1)     # :2819
    pipe_cmp_mma_dS    = pipe(mbar_compute_mma_dS, 1, async_umma,
                              producer=128, consumer=1)     # :2836
    pipe_mma_red_dKV   = pipe(mbar_mma_reduce_dKV, 2, umma_async,
                              producer=1,   consumer=256)   # :2850  <- 2 stages
    pipe_cmp_tma_dQ    = pipe(None, 1, tma_store, producer=128)  # :2863
    # instruction_selection: mbarrier.init.shared.b64 x22 (:2714..:2850);
    #   extent: one per stage per direction.  pipe_cmp_tma_dQ has NO smem
    #   barrier -- a TMA-store pipeline is `cp.async.bulk.commit_group` plus
    #   `cp.async.bulk.wait_group.read`, nothing else.

    init_pipe_arrive(relaxed=True)          # :860
    # instruction_selection: fence.mbarrier_init.release.cluster (:860, 1
    #   static); extent: CTA.
    init_pipe_wait()                        # :905
    # instruction_selection: bar.sync 0 (:905, 1 static); extent: whole CTA.
    #   NOT a cluster barrier -- `barrier.cluster` appears zero times in the
    #   export, because the cluster is (1,1,1).
    #   Note this sits AFTER every alias declaration (:862-903), so a port that
    #   hoists the wait changes when the aliases become legible.

    tile_count = ceil_div(topk, 64)         # :907

    # -----------------------------------------------------------------------
    # Role dispatch -- source order, NOT warp-id order (:909-1110)
    # -----------------------------------------------------------------------
    if   warp == 17:            set_register_budget(dec, 40);  role_load()
    elif warp == 16:            set_register_budget(dec, 40);  role_mma()
    elif warp in (4,5,6,7):     set_register_budget(inc, 128); role_compute()
    elif warp in (8..15):       set_register_budget(inc, 128); role_reduce()
    elif warp in (0,1,2,3):     set_register_budget(dec, 40);  role_load_KV()
    else:                       set_register_budget(dec, 40)   # warps 18,19
    # instruction_selection: setmaxnreg.dec.sync.aligned.u32 40 x4 (:910, :928,
    #   :1098, :1110) and setmaxnreg.inc.sync.aligned.u32 128 x2 (:1026,
    #   :1074); extent: scalar per role branch.  The export settles a question
    #   the source text raises: warps 16, 17 and 18..19 share warpgroup 4 yet
    #   each issues its OWN setmaxnreg.  Six instructions, one per role, not
    #   one per warpgroup.

# ---------------------------------------------------------------------------
# Role: load  (warp 17)   (:1113-1207)   -- runs once, no loop
# ---------------------------------------------------------------------------

def role_load():
    acquire(pipe_load_mma_QdO, slot=0)
    # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:1155, 1
    #   static) on the empty barrier, then mbarrier.arrive.expect_tx.shared.b64
    #   (:1152, 1 static) on the full one; extent: scalar, elected lane.  A
    #   TMA/UMMA producer_acquire is BOTH, which is why it is one of the 23
    #   try_waits the summary counts.  ONE expect_tx covers BOTH the
    #   Q and dO transfers -- `tx_count = tma_copy_QdO_bytes` is their sum
    #   (:442-444, :2719) and both copies below post to the same barrier
    #   (:1153).  Splitting it into two barriers changes the protocol.

    copy_g2s_tma(desc_Q,  (0, head_block_idx * 64, token_idx), sQ,
                 bar=mbar_load_mma_QdO)   # coordinates in ELEMENTS
    # instruction_selection: cp.async.bulk.tensor.3d.shared::cta.global.tile.
    #   mbarrier::complete_tx::bytes**.L2::cache_hint** -- the qualifier is part
    #   of the emitted mnemonic and brings a 64-bit cache-policy operand, a
    #   constant 0 in this build.  (:1154, **HEAD_DIM // 64 static** -- 8 at
    #   d512, **9 at d576**); extent: that many issues covering the
    #   64 x HEAD_DIM tile.  One `cute.copy` is eight or nine TMA instructions
    #   here, not one: the descriptor's box splits the head-dim axis into
    #   64-wide boxes, so the count follows HEAD_DIM.  Rank is 3.
    copy_g2s_tma(desc_dO, (0, head_block_idx * 64, token_idx), sdO,
                 bar=mbar_load_mma_QdO)   # coordinates in ELEMENTS
    # instruction_selection: same opcode, also `.L2::cache_hint` (:1162, 8
    #   static in BOTH builds);
    #   extent: 8 issues covering 64 x 512 -- dO is HEAD_DIM_V wide, which is
    #   512 either way, so this one does not follow HEAD_DIM.

    # LSE and sum_OdO: 32 lanes x 2 f32 = the 64 rows of this head block.
    acquire(pipe_load_cmp_LSE, slot=0)                         # :1181
    # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:1184);
    #   extent: scalar spin on the LSE empty barrier.
    copy_g2s_async(sLSE, addr(scaled_lse_ws, head_block_idx, token_idx), bytes=8)
    # instruction_selection: cp.async.ca.shared.global (:1186, 1 static);
    #   extent: 64 bits per lane, 32 lanes.  `.ca` not `.cg`: the atom is built
    #   with LoadCacheMode.ALWAYS (:1170).  num_bits_per_copy=64 with
    #   thr_layout 32 / val_layout 2 (:1170-1173).
    commit(pipe_load_cmp_LSE)
    # instruction_selection: cp.async.mbarrier.arrive.noinc.shared.b64 (:1191);
    #   extent: scalar.  This is how a cp.async pipeline commits -- no
    #   separate commit_group.

    acquire(pipe_load_cmp_sOdO, slot=0)                        # :1195
    # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:1198);
    #   extent: scalar spin on the sum_OdO empty barrier.
    copy_g2s_async(sSum_OdO, addr(sum_OdO_ws, head_block_idx, token_idx), bytes=8)
    # instruction_selection: cp.async.ca.shared.global (:1200, 1 static);
    #   extent: 64 bits per lane, 32 lanes.
    commit(pipe_load_cmp_sOdO)
    # instruction_selection: cp.async.mbarrier.arrive.noinc.shared.b64 (:1206).
    # The warp retires here.  There is no producer_commit for
    # pipe_load_mma_QdO: a TMA pipeline's completion IS the expect_tx.

# ---------------------------------------------------------------------------
# Role: load_KV  (warps 0..3)   (:1316-1431)
#   rows_per_warp = 64 / 4 = 16.  Warp w owns rows w, w+4, w+8, ...
# ---------------------------------------------------------------------------

def role_load_KV():
    lane = tidx % 32
    lwarp = tidx // 32                     # 0..3
    full_tiles = (topk % 64) == 0          # :1344

    def read_indices(tile_index) -> reg_tile[16]:
        r = reg_tile("i32", [16])
        for i in range(16):                                    # :1348, unrolled
            row = i * 4 + lwarp
            idx = tile_index * 64 + row
            v = i32(-1)
            if lane == 0 and idx < MAX_TOPK:
                v = load_global(topk_idxs, (idx, token_idx))
                # instruction_selection: ld.global.b32 (:1354 and :1404, 16
                #   static each); extent: scalar, lane 0 only.  The bound is
                #   MAX_TOPK, the allocation extent, not `topk`: rows past the
                #   length are read but discarded.
            r[i] = shuffle_broadcast(v, lane=0)
            # instruction_selection: shfl.sync.idx.b32 (:1355 and :1405, 16
            #   static each, 32 total); extent: warp broadcast, 16 per tile.
        return r

    def gather_rows(tile_index, r, is_first):
        # is_first is a COMPILE-TIME flag; the source instantiates this helper
        # twice from two call sites (:1361 / :1375) plus once in the loop
        # (:1411), which is why the gather's static counts are 3x per-call.
        for i in range(16):                                    # :1292, unrolled
            row = i * 4 + lwarp
            idx = tile_index * 64 + row
            dst = row                         # a row index; sK_off() maps it
            if HAS_TOPK_LENGTH:
                if is_first and idx >= topk:
                    zero_kv_row(dst)
                else:
                    copy_kv_row(dst, r[i])
            else:
                if idx < topk and r[i] >= 0:
                    copy_kv_row(dst, r[i])
                else:
                    zero_kv_row(dst)

    # `sK_slice` is composition(sK[...], (64, HEAD_DIM)) (:1357-1358), which
    # measures (64,(64,8)):(64,(1,4096)) -- NOT a plain row-major (64, HEAD_DIM).
    # The row stride is 64 elements (128 B) and the head-dim GROUP stride is
    # 4096 elements (8192 B), with 64 contiguous elements inside a group, all
    # under swizzle S<3,4,3>.  The export settles it: stepping `j` by one moves
    # the group index by 4 and the SMEM address by +32768 B while the GMEM
    # address moves by only +512 B (`cp.async.cg.shared.global [%r289+32768],
    # [%rd152+512]`, :1229).  A row-major reading -- row stride 1024 B, group
    # stride 128 B -- is wrong in both strides.
    def sK_off(row, group, lane):        # ELEMENTS into the linear sK region
        return row * 64 + group * 4096 + (lane % 8) * 8

    def copy_kv_row(row, topk_idx):
        # 8 lanes cover one 64-element group; lanes are grouped by `lane % 8`
        # (:1339) and the group index by `lane // 8` (:1224).
        for j in range(2):                                     # :1223, unrolled
            group = j * 4 + lane // 8
            copy_g2s_async(sK_off(row, group, lane),
                           addr(kv, topk_idx, group * 64 + (lane % 8) * 8),
                           bytes=16)
            # Both sides in ELEMENTS, and BOTH carry the lane term: the 8 lanes
            # of a group read 8 disjoint 8-element runs (`get_slice(lane % 8)`
            # with val_layout (8,), :1339).  Dropping it from the source side
            # makes all 8 lanes read the same 16 bytes and leaves 56 of every 64
            # elements never loaded.
            # instruction_selection: cp.async.cg.shared.global (:1229 at d512,
            #   **:1237 at d576**, 96 static = 3 call sites x 16 rows x 2
            #   groups); extent: 128 bits per lane.  `.cg` not `.ca`:
            #   LoadCacheMode.GLOBAL (:1332), num_bits_per_copy=128, thr/val
            #   layouts 8/8 (:1334-1338).
        if HAS_TAIL:
            if lane < 8:                                       # :1243
                copy_g2s_async(sK_off(row, 8, lane),
                               addr(kv, topk_idx, 512 + (lane % 8) * 8),
                               bytes=16)
                # instruction_selection: cp.async.cg.shared.global (:1244, 48
                #   static); extent: 128 bits, first 8 lanes only.  The 9th
                #   group is the 64-wide d576 tail.

    def zero_kv_row(row):
        for j in range(2):                                     # :1253
            group = j * 4 + lane // 8
            zero_shared(sK_off(row, group, lane), bytes=16)
            # instruction_selection: st.shared.v4.b32 (:1258 at d512, **:1265 at
            #   d576**, 32 static; 192 static in the non-compact build, where
            #   every row can be invalid); extent: 128 bits per lane.  A zero
            #   fill, not a copy: an invalid row must contribute nothing to
            #   either GEMM that reads sK.
        if HAS_TAIL and lane < 8:
            zero_shared(sK_off(row, 8, lane), bytes=16)
            # instruction_selection: st.shared.v4.b32 (:1271, 16 static);
            #   extent: 128 bits, first 8 lanes only.  d576 only.

    # --- first (highest-index, ragged) tile, then the rest ---------------
    tile_index = tile_count - 1                                # :1343
    r = read_indices(tile_index)
    acquire(pipe_load_mma_K, slot=0)                           # :1356
    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent:
    #   scalar spin on the K empty barrier, every load_KV thread.
    if full_tiles:
        gather_rows(tile_index, r, is_first=False)             # :1361
    else:
        gather_rows(tile_index, r, is_first=True)              # :1375
    cp_async_commit()
    # instruction_selection: cp.async.commit_group (:1389, 2 static -- one here
    #   and one in the loop); extent: scalar.
    cp_async_wait(0)
    # instruction_selection: cp.async.wait_group 0 (:1390, 2 static); extent:
    #   scalar.  Waits for ALL groups: the gather is not double-buffered.
    fence_view_async_shared()                                  # :1391
    # instruction_selection: fence.proxy.async.shared::cta (:1391, 1 static);
    #   extent: CTA.
    named_barrier(5, 128)                                      # :1392
    # instruction_selection: bar.sync 5, 128 (:1392, 1 static); extent: the 4
    #   load_KV warps.  This is what makes the 4 warps' gathers jointly visible
    #   before any one of them commits the pipeline.
    commit(pipe_load_mma_K, slot=0)                            # :1393
    # instruction_selection: mbarrier.arrive.shared.b64 (:1393, 1 static);
    #   extent: scalar per producer thread -- the pipeline's full barrier
    #   expects 128 arrivals, one per load_KV thread.
    tile_index -= 1

    while tile_index >= 0:                                     # :1397
        r = read_indices(tile_index)
        acquire(pipe_load_mma_K, slot=0)                       # :1407
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent:
        #   scalar spin on the empty barrier.
        gather_rows(tile_index, r, is_first=False)             # :1411
        cp_async_commit()                                      # :1425
        # instruction_selection: cp.async.commit_group (:1425, 1 static).
        cp_async_wait(0)                                       # :1426
        # instruction_selection: cp.async.wait_group 0 (:1426, 1 static).
        fence_view_async_shared()                              # :1427
        # instruction_selection: fence.proxy.async.shared::cta (:1427, 1 static).
        named_barrier(5, 128)                                  # :1428
        # instruction_selection: bar.sync 5, 128 (:1428, 1 static).
        commit(pipe_load_mma_K, slot=0)                        # :1429
        # instruction_selection: mbarrier.arrive.shared.b64 (:1429, 1 static).
        tile_index -= 1

# ---------------------------------------------------------------------------
# Role: mma  (warp 16)   (:1434-1765)
#   Issues every tcgen05 MMA.  32 static MMA instructions in the d512 build.
# ---------------------------------------------------------------------------

def role_mma():
    tmem_wait_alloc(); base = tmem_retrieve()   # :929-930
    # instruction_selection: bar.sync 2, 416 then ld.shared.b32 of
    #   tmem_holding; extent: the compute+reduce+mma participants.

    wait(pipe_load_mma_QdO, slot=0)             # :1490 -- once, before the loop
    # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:1493, 1 of
    #   23 static); extent: scalar spin.  Waits for BOTH TMAs: they share one
    #   barrier and one expect_tx.
    acquire(pipe_mma_cmp_dQ, slot=0)            # :1491 -- once, before the loop
    # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:1494);
    #   extent: scalar spin on the dQ empty barrier.  Acquired ONCE for the
    #   whole kernel: dQ accumulates across every tile.

    tile_index = tile_count - 1                 # :1493
    is_first_mma = True                         # compile-time per unrolled arm
    while tile_index >= 0:                      # :1495
        wait(pipe_load_mma_K, slot=0)           # :1497
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:1500);
        #   extent: scalar spin.
        acquire(pipe_mma_cmp_S, slot=0)         # :1498
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:1501);
        #   extent: scalar spin.

        # --- S = Q @ K^T ------------------------------------------------
        gemm(tmem[S], sQ, sK, accumulate=False_then_True)      # :1500-1509
        # instruction_selection: tcgen05.mma.cta_group::1.kind::f16 (:1502, 4
        #   static); extent: ONE LOOP of 32 K-phases with a 4-wide unrolled
        #   body.  The export has a real backedge over the four issues, so
        #   `unroll=4` at :1501 produced a rolled loop, not full unrolling.
        #   ACCUMULATE is False on the first K-phase and True thereafter
        #   (:1500, :1509) -- the flag is part of the instruction, not a
        #   separate zeroing pass.
        commit(pipe_mma_cmp_S, slot=0)
        # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::
        #   arrive::one.shared::cluster.b64 (:1511, 1 of 9 static); extent:
        #   scalar.  A UMMA pipeline commits with tcgen05.commit, never with
        #   a plain mbarrier.arrive.

        # --- dP = dO @ V^T ----------------------------------------------
        acquire(pipe_mma_cmp_dP, slot=0)                       # :1515
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:1518);
        #   extent: scalar spin.
        gemm(tmem[dP], sdO, sV, accumulate=False_then_True)    # :1516-1525
        # instruction_selection: tcgen05.mma...kind::f16 (:1518, 4 static);
        #   extent: loop of 32 K-phases, unrolled 4.
        commit(pipe_mma_cmp_dP, slot=0)                        # :1527
        # instruction_selection: tcgen05.commit...b64 (:1527, 1 static).

        # --- dKV generation A: dKV0, dKV1 -------------------------------
        wait(pipe_cmp_mma_P, slot=0)                           # :1531
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent:
        #   scalar spin on the P full barrier (128 compute-thread arrivals).
        acquire(pipe_mma_red_dKV, slot=stage_a)                # :1532
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent:
        #   scalar spin on the 2-stage dKV empty barrier.
        if not is_first_mma:
            # WAR against the PREVIOUS generation's T2R reads.  Which barrier
            # depends on the head dim, because the alias maps differ.
            named_barrier(10 if HAS_TAIL else 8, 288)          # :1541-1546
            # instruction_selection: bar.sync 10, 288 (:1543) at d576 /
            #   bar.sync 8, 288 (:1546) at d512; extent: the 8 reduce warps
            #   plus this one.  Skipped on the first processed tile and
            #   compensated by an unpaired arrive after the loop (:1752-1757).
        gemm(tmem[dKV0], sdOT[0:128],   sP, accumulate=False_then_True)  # :1547
        # instruction_selection: tcgen05.mma...kind::f16 (:1549, 2 static);
        #   extent: loop of 4 K-phases, unrolled 2 (:1548).
        gemm(tmem[dKV1], sdOT[128:256], sP, accumulate=False_then_True)  # :1559
        # instruction_selection: same (:1561, 2 static); extent: same.

        wait(pipe_cmp_mma_dS, slot=0)                          # :1570
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent:
        #   scalar spin on the dS full barrier.
        gemm(tmem[dKV0], sQT[0:128],   sdS, accumulate=True)   # :1574
        # instruction_selection: same (:1576, 2 static); extent: loop of 4
        #   K-phases, unrolled 2.  ACCUMULATE stays True for the whole loop
        #   (:1574 sets it once): this ADDS dK onto the dV already in these
        #   columns, which is what fuses the two gradients.
        gemm(tmem[dKV1], sQT[128:256], sdS, accumulate=True)   # :1584
        # instruction_selection: same (:1585, 2 static).
        commit(pipe_mma_red_dKV, slot=stage_a)                 # :1594
        # instruction_selection: tcgen05.commit...b64 (:1594, 1 static).

        if HAS_TAIL:
            named_barrier(7, 288)                              # :1600-1602
            # instruction_selection: bar.sync 7, 288 (:1602, 1 static); extent:
            #   the 8 reduce warps plus this one.  Orders the dKV4 write
            #   against the dKV0/dKV1 reads, because dKV4 aliases dKV0.
            #   UNCONDITIONAL -- barrier 7 is never skipped in either build.
            gemm(tmem[dKV4], sQT_tail, sdS, accumulate=False_then_True)
            # instruction_selection: tcgen05.mma...kind::f16 (:1606, 2 static);
            #   extent: loop of 4 K-phases.  d576 only.
            commit(pipe_mma_red_dKV, slot=stage_b)             # :1616
            # instruction_selection: tcgen05.commit...b64 (**:1616**, 1 static).
            #   :1617 is the producer_state.advance() and emits nothing.
            # NOTE: this commit has NO matching acquire.  The source relies on
            # t2r_dKV01_done above for the safety an acquire would provide.
            # The generation count per tile is therefore 3 at d576 and 2 at
            # d512, and the reduce role's wait count matches.

        # --- dQ0..dQ3 (and dQ4) -----------------------------------------
        for i in range(4):                                     # :1622-1667
            gemm(tmem[dQ_i], sK_2[i*128:(i+1)*128], sdST,
                 accumulate=not is_first_mma)
            # instruction_selection: tcgen05.mma...kind::f16 (:1624, :1636,
            #   ..., 2 static each); extent: loop of 4 K-phases, unrolled 2.
            #   ACCUMULATE is False only on the first PROCESSED tile: dQ
            #   accumulates across the whole top-k axis in TMEM and is never
            #   published until the loop ends.
        if HAS_TAIL:
            gemm(tmem[dQ4], sK_tail, sdST, accumulate=not is_first_mma)  # :1670
            # instruction_selection: tcgen05.mma...kind::f16 (:1673, 2
            #   static); extent: loop of 4 K-phases, unrolled 2.  d576 only,
            #   which is why that build has 36 MMA issues against d512's 32.

        release(pipe_load_mma_K, slot=0)                       # :1683
        # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::
        #   arrive::one.shared::cluster.b64 (:1683, 1 static); extent: scalar.
        # The KV tile's SMEM is reusable from here; the gather warps may
        # overwrite sK for the next tile.

        # --- dKV generation B: dKV2, dKV3 -------------------------------
        # UNCONDITIONAL, unlike the generation-A WAR barrier above: the export
        # shows this arrive outside any is_first-predicated region.
        named_barrier(8 if HAS_TAIL else 7, 288)               # :1689-1692
        # instruction_selection: bar.sync 7, 288 (:1692, d512) / bar.sync 8, 288
        #   (:1690, d576), 1 static; extent: the 8 reduce warps plus this one.
        acquire(pipe_mma_red_dKV, slot=stage_b_or_c)           # :1694
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent:
        #   scalar spin.
        gemm(tmem[dKV2], sdOT[256:384], sP, accumulate=False_then_True)  # :1695
        # instruction_selection: tcgen05.mma...kind::f16 (:1697, 2 static);
        #   extent: loop of 4 K-phases, unrolled 2.
        gemm(tmem[dKV3], sdOT[384:512], sP, accumulate=False_then_True)  # :1706
        # instruction_selection: same (:1709, 2 static).
        release(pipe_cmp_mma_P, slot=0)                        # :1719
        # instruction_selection: tcgen05.commit...b64 (:1719, 1 static).
        gemm(tmem[dKV2], sQT[256:384], sdS, accumulate=True)   # :1724
        # instruction_selection: same (:1725, 2 static).
        gemm(tmem[dKV3], sQT[384:512], sdS, accumulate=True)   # :1732
        # instruction_selection: same (:1734, 2 static).
        commit(pipe_mma_red_dKV, slot=stage_b_or_c)            # :1742
        # instruction_selection: tcgen05.commit...b64 (:1742, 1 static).
        release(pipe_cmp_mma_dS, slot=0)                       # :1746
        # instruction_selection: tcgen05.commit...b64 (:1746, 1 static).
        is_first_mma = False
        tile_index -= 1

    # --- loop tail: balance the ONE skipped first-iteration arrive -------
    # Exactly one WAR barrier is skipped-once per build -- the generation-A one
    # at :1541-1546, which is id 10 at d576 and id 8 at d512.  Barrier 7 is
    # unconditional in both builds (:1602 / :1692), so it needs no compensation.
    named_barrier(10 if HAS_TAIL else 8, 288)                  # :1752-1757
    # instruction_selection: bar.sync 8, 288 (:1753, d512) / bar.sync 10, 288
    #   (:1757, d576), 1 static; extent: the 8 reduce warps plus this one.
    #   The export has exactly one tail barrier per build, not two.

    commit(pipe_mma_cmp_dQ, slot=0)                            # :1759
    # instruction_selection: tcgen05.commit...b64 (:1759, 1 static); extent:
    #   scalar.  dQ is published ONCE, after the whole top-k axis is
    #   accumulated.
    release(pipe_load_mma_QdO, slot=0)                         # :1763
    # instruction_selection: tcgen05.commit...b64 (:1763, 1 static).

# ---------------------------------------------------------------------------
# Role: compute  (warps 4..7)   (:1767-2140)
# ---------------------------------------------------------------------------

def role_compute():
    if warp == 4:
        tmem_alloc(512)                                        # :1028
        # instruction_selection: tcgen05.alloc.cta_group::1.sync.aligned.
        #   shared::cta.b32 (:1028, 1 static); extent: scalar, warp 4 only.
        #   512 columns = all of tensor memory (:79-80).
    tmem_wait_alloc(); base = tmem_retrieve()                  # :1029-1030
    # instruction_selection: bar.sync 2, 416 (:1029, 1 static) then
    #   ld.shared.b32 of tmem_holding; extent: compute + reduce + mma.

    wait(pipe_load_cmp_LSE, slot=0)                            # :1848
    # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:1851);
    #   extent: scalar spin, every compute thread.
    wait(pipe_load_cmp_sOdO, slot=0)                           # :1849
    # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:1852).
    softmax_scale_log2e = softmax_scale * log2_e               # :1852
    # instruction_selection: mul.f32 (:1852, 1 static); extent: scalar, every
    #   compute thread.  NOT a trace-time fold: `scale_softmax` arrives as a
    #   runtime f32 kernel parameter (`.param .f32 ..._param_25`, source :768),
    #   so this multiply executes.  It is also the only scalar mul.f32 in `bwd`
    #   -- everything else in the softmax path is packed.

    tile_index = tile_count - 1                                # :1870
    while tile_index >= 0:                                     # :1871
        # --- P = exp2(fma(S, scale*log2e, scaled_lse)) ------------------
        wait(pipe_mma_cmp_S, slot=0)                           # :1872
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent:
        #   scalar spin on the S full barrier (one tcgen05.commit arrival).
        acquire(pipe_cmp_mma_P, slot=0)                        # :1873
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent:
        #   scalar spin on the P empty barrier.
        rS = copy_t2r(tmem[S], shape="16x256b", rep=8)         # :1875
        # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x8.b32
        #   (:1875, 1 static); extent: one issue per thread covering this
        #   thread's slice of the 64x64 S tile.  Ld16x256bOp(Repetition(8))
        #   at :1816-1819.

        for i in range(0, len(rS), 2):                         # :1877
            lse2 = (load_shared(sLSE, row(i)), load_shared(sLSE, row(i+1)))
            # instruction_selection: ld.shared.b32 (:1880, **2 static**);
            #   extent: scalar.  Not 32: the loop runs 16 iterations x 2 reads,
            #   but each thread's 16x256b fragment spans only TWO rows of the
            #   64-row tile, so every iteration re-reads the same two values and
            #   the compiler keeps one load each.  A port that hoists these by
            #   hand is matching what the compiler already does; one that
            #   indexes them per-iteration emits 32.
            rS[i], rS[i+1] = fma((rS[i], rS[i+1]),
                                 (softmax_scale_log2e,) * 2, lse2, lanes=2)
            # instruction_selection: fma.rn.f32x2 (:1884, 16 static); extent:
            #   one packed two-lane FMA per pair.  Not two scalar FMAs: the
            #   source calls fma_packed_f32x2 and the export confirms the
            #   packed form.
            rS[i]   = exp2(rS[i],   ftz=True)
            rS[i+1] = exp2(rS[i+1], ftz=True)
            # instruction_selection: ex2.approx.ftz.f32 (:1889-:1890, 32
            #   static = 16 pairs x 2); extent: scalar each.  `.ftz` because
            #   this call passes fastmath=True (:1889) -- contrast sum_OdO's
            #   plain ex2.approx.f32.
        rP = cast(rS, ELEM)                                    # :1892
        # instruction_selection: cvt.rn.bf16x2.f32 (via quantize, :2625);
        #   extent: 4-element fragments (frg_cnt=4 at :1892).

        fence_tmem_load()                                      # :1894
        # instruction_selection: tcgen05.wait::ld.sync.aligned (:1894, 1 of 8
        #   static); extent: scalar.  Must precede the barrier: it is what
        #   makes the T2R results architecturally visible.
        named_barrier(3, 128)                                  # :1895
        # instruction_selection: bar.sync 3, 128 (:1895, 1 static); extent:
        #   the 4 compute warps.

        copy_r2s_stmatrix(rP, sP_store)                        # :1898-:1900
        # instruction_selection: stmatrix.sync.aligned.m8n8.x4.trans.shared.b16
        #   (:1900, 4 static); extent: 4 matrices per issue, transposing.
        #   The `.trans` is the whole point: it converts the row-major
        #   (head, kv) accumulator fragment into the head-major B-operand
        #   layout the dOP MMA needs, with no separate transpose pass.
        fence_proxy_shared()                                   # :1905
        # instruction_selection: fence.proxy.async.shared::cta (:1905, 1 of 8
        #   static); extent: CTA.  Required before a UMMA reads what a
        #   generic-proxy store wrote.
        commit(pipe_cmp_mma_P, slot=0)                         # :1910
        # instruction_selection: mbarrier.arrive.shared.b64 (:1910); extent:
        #   scalar.  An async->UMMA pipeline commits with a plain arrive.
        release(pipe_mma_cmp_S, slot=0)                        # :1913
        # instruction_selection: mbarrier.arrive.shared.b64 (:1913, 1 static);
        #   extent: scalar per compute thread -- the empty barrier expects 128.

        # --- dS = (dP + sum_OdO) * P * softmax_scale --------------------
        wait(pipe_mma_cmp_dP, slot=0)                          # :1916
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent:
        #   scalar spin.
        acquire(pipe_cmp_mma_dS, slot=0)                       # :1917
        # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent:
        #   scalar spin.
        rdP = copy_t2r(tmem[dP], shape="16x256b", rep=8)       # :1919
        # instruction_selection: tcgen05.ld...16x256b.x8.b32 (:1919, 1 static);
        #   extent: one issue.  Reads the SAME columns as S but at lane offset
        #   16<<16 (:2675-2677), which is what lets dP and S share columns.

        for i in range(0, len(rdP), 2):                        # :1921
            sums = (load_shared(sSum_OdO, row(i)), load_shared(sSum_OdO, row(i+1)))
            # instruction_selection: ld.shared.b32 (:1925, **2 static**);
            #   extent: scalar -- same two-rows-per-fragment CSE as sLSE above.
            rdP[i], rdP[i+1] = add((rdP[i], rdP[i+1]), sums, lanes=2)
            # instruction_selection: add.rn.f32x2 (:1922, 16 static); extent:
            #   one packed two-lane add.  `sum_OdO` is already negated
            #   (sum_OdO_scale = -1.0 at :488), so this add IS the
            #   `dP - delta` of the softmax backward.
            rdP[i], rdP[i+1] = mul((rdP[i], rdP[i+1]), (rS[i], rS[i+1]), lanes=2)
            # instruction_selection: mul.rn.f32x2 (:1936, 16 static); extent:
            #   one packed two-lane multiply.  rS still holds P here, so this
            #   is the `* P` factor; rS must stay live across both loops.
        rdS = cast(rdP * softmax_scale, ELEM)                  # :1938
        # instruction_selection: mul.f32x2 (:2624, 16 static) then
        #   cvt.rn.bf16x2.f32 (:2625, 16 static); extent: one packed two-lane
        #   multiply per pair, then one packed convert per pair.  The scale is
        #   PACKED like the two loops above it, not scalar -- the only plain
        #   `mul.f32` in `bwd` is the single runtime scalar one at :1852.  The
        #   `* softmax_scale` is folded into the quantize call (:1938 passes
        #   it), not a separate pass.

        fence_tmem_load()                                      # :1940
        # instruction_selection: tcgen05.wait::ld.sync.aligned (:1940, 1 of 8
        #   static); extent: scalar.
        named_barrier(3, 128)                                  # :1941
        # instruction_selection: bar.sync 3, 128 (:1941, 1 static).
        release(pipe_mma_cmp_dP, slot=0)                       # :1943
        # instruction_selection: mbarrier.arrive.shared.b64 (:1943, 1 static).
        copy_r2s_stmatrix(rdS, sdS_store)                      # :1946-:1948
        # instruction_selection: stmatrix.sync.aligned.m8n8.x4.trans.shared.b16
        #   (:1948, 4 static); extent: 4 matrices, transposing.
        fence_proxy_shared()                                   # :1955
        # instruction_selection: fence.proxy.async.shared::cta (:1955, 1 static).
        commit(pipe_cmp_mma_dS, slot=0)                        # :1960
        # instruction_selection: mbarrier.arrive.shared.b64 (:1960, 1 static).
        tile_index -= 1

    release(pipe_load_cmp_LSE, slot=0)                         # :1965
    # instruction_selection: mbarrier.arrive.shared.b64 (:1965, 1 static).
    release(pipe_load_cmp_sOdO, slot=0)                        # :1966
    # instruction_selection: mbarrier.arrive.shared.b64 (:1966, 1 static).

    # --- dQ epilogue: 4 rounds (5 at d576) --------------------------------
    # sdQ ALIASES sK, so this may only run after the KV loop has drained.
    wait(pipe_mma_cmp_dQ, slot=0)                              # :2033
    # instruction_selection: mbarrier.try_wait.parity.shared.b64; extent:
    #   scalar spin.  One wait for the whole epilogue: dQ was committed once.
    for i in range(4):                                         # :2035-2113
        if warp == 4:
            acquire(pipe_cmp_tma_dQ)                           # :2036
            # instruction_selection: cp.async.bulk.wait_group.read (:2036,
            #   :2057, :2077, :2097, :2140 -- 5 static); extent: scalar.  A
            #   TMA-store pipeline acquires by waiting on the bulk group, not
            #   on an mbarrier: it has no smem barrier at all.
        named_barrier(3, 128)                                  # :2038
        # instruction_selection: bar.sync 3, 128 (:2038, :2058, :2078, :2098 --
        #   4 static, one per unrolled round); extent: the 4 compute warps.
        #   This is what makes warp 4's acquire visible to the other three.
        #   The round-CLOSING barrier is a separate set of four (:2053, :2073,
        #   :2093, :2112); the eight are disjoint pairs, not one pool.
        store_dQ(i)
        if warp == 4:
            commit(pipe_cmp_tma_dQ)                            # :2051
            # instruction_selection: cp.async.bulk.commit_group (:2051, :2071,
            #   :2091, :2111 -- 4 static, one per unrolled round); extent:
            #   scalar.
        named_barrier(3, 128)                                  # :2053
        # instruction_selection: bar.sync 3, 128 (:2053, :2073, :2093, :2112
        #   -- 4 static, one per unrolled round).
    if HAS_TAIL:                                               # :2116-2135
        if warp == 4:
            acquire(pipe_cmp_tma_dQ)                           # :2118
            # instruction_selection: cp.async.bulk.wait_group.read (:2118, 1
            #   static); extent: scalar.  d576 only -- this is the fifth round.
        named_barrier(3, 128)                                  # :2119
        # instruction_selection: bar.sync 3, 128 (:2119, 1 static).
        store_dQ_64()
        if warp == 4:
            commit(pipe_cmp_tma_dQ)                            # :2133
            # instruction_selection: cp.async.bulk.commit_group (:2133, 1
            #   static).
        named_barrier(3, 128)                                  # :2134
        # instruction_selection: bar.sync 3, 128 (:2134, 1 static).
    release(pipe_mma_cmp_dQ, slot=0)                           # :2137
    # instruction_selection: mbarrier.arrive.shared.b64 (:2137, 1 static).
    tma_store_wait(0)                                          # :2140
    # instruction_selection: cp.async.bulk.wait_group.read (:2140, the last of
    #   the 5 static); extent: scalar.  `producer_tail`: drains every
    #   outstanding dQ store before the CTA may free sdQ / exit.

    if warp == 4:
        named_barrier(9, 288)                                  # :1070
        # instruction_selection: bar.sync 9, 288 (:1070, 1 static); extent: the
        #   8 reduce warps plus compute warp 4.  arrive_and_wait here, against
        #   the reduce side's arrive-only at :1095.
        tmem_dealloc(base, 512)                                # :1071
        # instruction_selection: tcgen05.dealloc.cta_group::1.sync.aligned.b32
        #   (:1071, 1 static); extent: scalar.
        #   The barrier first: the reduce warps must have drained their final
        #   T2R reads or the dealloc races them inside the CTA.

def store_dQ(i):                                               # :2507-2555
    rdQ = copy_t2r(tmem[dQ_i], shape="32x32b", rep=8)          # :2533
    # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x8.b32 (:2533, 32
    #   static = 4 sub-tiles x 8); extent: 8 issues per sub-tile.  A DIFFERENT
    #   TMEM load shape from the dKV path's 16x256b: 32x32b is what makes the
    #   register fragment match the 128x64 epilogue tile the TMA store wants.
    rq = cast(rdQ, ELEM)                                       # :2535
    # instruction_selection: cvt.rn.bf16.f32 (:2625, **256 static**); extent:
    #   SCALAR, one per element.  NOT the packed cvt.rn.bf16x2.f32 that the
    #   other two quantize sites get: :2625 emits 32 packed converts (the P and
    #   dS sites, 16 pairs each) AND 256 scalar ones, and the 256 belong here.
    fence_tmem_load()                                          # :2537
    # instruction_selection: tcgen05.wait::ld.sync.aligned (:2537, 4 static).
    copy_r2s(rq, sdQ + sdQ_elem(dp_idx, j))                    # :2543
    # instruction_selection: st.shared.b16 (:2543, **256 static** = 4 rounds x
    #   64 elements per thread); extent: SCALAR.  `cute.autovec_copy` does NOT
    #   vectorize this access -- there is not one `st.shared.v4.b32` at :2543,
    #   and the only 32 in `bwd` are the KV zero-fill at :1258.  This is the
    #   single largest instruction count in the kernel and the difference
    #   between a ~64- and a ~512-instruction epilogue, so a port that
    #   "helpfully" vectorizes it is not transcribing this kernel.
    named_barrier(3, 128)                                      # :2545
    # instruction_selection: bar.sync 3, 128 (:2545, 4 static -- one per round).
    fence_proxy_shared()                                       # :2547
    # instruction_selection: fence.proxy.async.shared::cta (:2547, 4 static).
    named_barrier(3, 128)                                      # :2552
    # instruction_selection: bar.sync 3, 128 (:2552, 4 static).  Two barriers
    #   around the fence: the first makes every thread's shared writes done,
    #   the second makes the fence itself observed before warp 4 issues the TMA.
    if warp == 4:
        copy_s2g_tma(desc_dQ, coords_i, sdQ)                   # :2555
        # instruction_selection: cp.async.bulk.tensor.3d.global.shared::cta.
        #   tile.bulk_group**.L2::cache_hint** (:2555, 8 static = 4 rounds x 2
        #   issues); extent: 2 issues per 128x64 tile, with the 64-bit
        #   cache-policy operand.  Warp 4 alone issues every dQ store.

def store_dQ_64():                                             # :2558-2608
    rdQ = copy_t2r(tmem[dQ4], shape="16x256b", rep=2)          # :2588
    # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x2.b32 (:2588,
    #   **4 static**); extent: 4 issues.  d576 only.
    fence_tmem_load()                                          # :2590
    # instruction_selection: tcgen05.wait::ld.sync.aligned (:2590, 1 static).
    for i in range(len(rdQ)):                                  # :2593 loop header
        store_shared(sdQ4, (row_i, col_i), cast(rdQ[i], ELEM))
        # instruction_selection: cvt.rn.bf16.f32 (:254, 32 static -- the cast is
        #   attributed to the outer entry, not to :2596) then st.shared.b16
        #   (**:2596**, 32 static); extent: SCALAR, element by element.
        #   Deliberately NOT vectorized: the 64-wide tail's coordinate mapping
        #   is not contiguous per thread, so the source writes through a
        #   coordinate tensor.  `convert` carries the matching unscramble.
    named_barrier(3, 128)                                      # :2598
    # instruction_selection: bar.sync 3, 128 (:2598, 1 static).
    fence_proxy_shared()                                       # :2600
    # instruction_selection: fence.proxy.async.shared::cta (:2600, 1 static).
    named_barrier(3, 128)                                      # :2605
    # instruction_selection: bar.sync 3, 128 (:2605, 1 static).
    if warp == 4:
        copy_s2g_tma(desc_dQ_64, (512, head_block_idx * 64, token_idx), sdQ4)
        # instruction_selection: cp.async.bulk.tensor.3d.global.shared::cta.
        #   tile.bulk_group**.L2::cache_hint** (:2608, 1 static); extent: 1
        #   issue for the 64x64
        #   tile.  THIS is the only path by which the d576 dQ tail columns
        #   512:576 reach mdQ -- omitting it leaves those columns whatever
        #   `torch.empty_like` left there, which is not zero and not detected by
        #   a d512 test.

# ---------------------------------------------------------------------------
# Role: reduce_dKV  (warps 8..15, two warpgroups)   (:2143-2280)
#   Each warpgroup owns half of the 64 KV rows (`split_wg`, :2628).
# ---------------------------------------------------------------------------

def role_reduce():
    tmem_wait_alloc(); base = tmem_retrieve()                  # :1075-1076
    # instruction_selection: bar.sync 2, 416 (:1075, 1 static) then
    #   ld.shared.b32 of tmem_holding; extent: compute + reduce + mma.
    full_tiles = (topk % 64) == 0                              # :2194
    tile_index = tile_count - 1                                # :2190

    while tile_index >= 0:                                     # :2195
        # Preload this tile's 8 row indices into registers, once, shared by
        # all four sub-tile reductions.
        r = reg_tile("i32", [8])
        for i in range(8):                                     # :2197
            row = row_coord(i * 2 - i % 2)                     # :2198-2199
            g = tile_index * 64 + row
            r[i] = load_global(topk_idxs, (g, token_idx)) if (full_tiles or g < topk) else -1
            # instruction_selection: ld.global.b32 (:2202/:2205, 8 static each); extent:
            #   scalar per thread.  Read again here rather than shared from
            #   the gather warps: different roles, no channel between them.
        if HAS_TAIL:
            r64 = ... same, over the dKV4 row coords ...       # :2211-2221

        # --- generation A: dKV0, dKV1 -----------------------------------
        wait(pipe_mma_red_dKV, slot=a)                         # :2223
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 (:2226);
        #   extent: scalar spin, every reduce thread.
        rdKV0 = copy_t2r(tmem[dKV0], shape="16x256b", rep=4)   # :2228
        rdKV1 = copy_t2r(tmem[dKV1], shape="16x256b", rep=4)   # :2229
        # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x4.b32
        #   (:2305, 8 static across the four t2r_dKV call sites); extent: **2
        #   issues per sub-tile**.  The fragment is a 128x64 f32 tile split by
        #   warpgroup = 32 f32 per thread, and Repetition(4) delivers 16
        #   registers per issue (:2171-2174), so one call is two issues.
        fence_tmem_load()                                      # :2230
        # instruction_selection: tcgen05.wait::ld.sync.aligned (:2230, 1 static).
        named_barrier(7, 288)                                  # :2231
        # instruction_selection: bar.sync 7, 288 (:2231, 1 static); extent: the
        #   8 reduce warps plus mma.
        #   THIS IS THE SPLIT that the whole barrier scheme exists for: the
        #   registers are drained and the MMA warp released BEFORE the slow
        #   global atomics below.  Folding the atomics back before this
        #   barrier (as the dead `store_dKV` at :2400 does) serializes the
        #   MMA warp behind them.  Three helpers in the source are dead and
        #   reach no PTX: `store_dKV` (:2399), `store_dKV_64` (:2453) and
        #   `interleave_wg` (:2644).  None has a call site anywhere in the file,
        #   so the port should not transcribe any of them.
        atomic_reduce(rdKV0, r, sub_tile=0)                    # :2232
        atomic_reduce(rdKV1, r, sub_tile=1)                    # :2233
        release(pipe_mma_red_dKV, slot=a)                      # :2235
        # instruction_selection: mbarrier.arrive.shared.b64 (:2235, 1 static);
        #   extent: scalar per reduce thread -- the empty barrier expects 256.

        # --- generation A.5: dKV4 (d576 only) ---------------------------
        if HAS_TAIL:
            wait(pipe_mma_red_dKV, slot=b)                     # :2241
            # instruction_selection: mbarrier.try_wait.parity.shared.b64.
            rdKV4 = copy_t2r(tmem[dKV4], shape="16x256b", rep=4)  # :2244
            # instruction_selection: tcgen05.ld...16x256b.x4.b32 (:2365, 1
            #   static); extent: 1 issue.
            fence_tmem_load()                                  # :2245
            # instruction_selection: tcgen05.wait::ld.sync.aligned (:2245).
            named_barrier(8, 288)                              # :2246
            # instruction_selection: bar.sync 8, 288 (:2246, 1 static).
            atomic_reduce_64(rdKV4, r64)                       # :2247
            release(pipe_mma_red_dKV, slot=b)                  # :2249
            # instruction_selection: mbarrier.arrive.shared.b64 (:2249).

        # --- generation B: dKV2, dKV3 -----------------------------------
        wait(pipe_mma_red_dKV, slot=c)                         # :2252
        # instruction_selection: mbarrier.try_wait.parity.shared.b64.
        rdKV2 = copy_t2r(tmem[dKV2], shape="16x256b", rep=4)   # :2257/:2270
        rdKV3 = copy_t2r(tmem[dKV3], shape="16x256b", rep=4)   # :2258/:2271
        # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x4.b32 (:2305,
        #   part of the 8 static); extent: 2 issues per sub-tile.
        fence_tmem_load()                                      # :2259/:2272
        # instruction_selection: tcgen05.wait::ld.sync.aligned (:2259, 1 static).
        named_barrier(8 if SAME_HDIM else 10, 288)             # :2260/:2273
        # instruction_selection: bar.sync 8, 288 (:2260, d512) / bar.sync 10,
        #   288 (:2273, d576), 1 static; extent: the 8 reduce warps plus mma.
        atomic_reduce(rdKV2, r, sub_tile=2)                    # :2261/:2274
        atomic_reduce(rdKV3, r, sub_tile=3)                    # :2262/:2275
        release(pipe_mma_red_dKV, slot=c)                      # :2277
        # instruction_selection: mbarrier.arrive.shared.b64 (:2277, 1 static).
        tile_index -= 1

    named_barrier_arrive(9, 288)                               # :1095
    # instruction_selection: bar.arrive 9, 288; extent: arrive only, no wait --
    #   the reduce warps must not stall here, they are done.

def atomic_reduce(rdKV, r, sub_tile):                          # :2309-2340
    for i in range(8):                                         # :2322
        base_c = i * 2 - i % 2                                 # :2323
        frg = (rdKV[base_c], rdKV[base_c+2], rdKV[base_c+16], rdKV[base_c+18])
        # The 4 values a thread owns are NOT contiguous in the fragment; this
        # gather is what makes them contiguous in the workspace row, and it is
        # the mapping `convert` later inverts.
        if r[i] >= 0:
            atomic_add_global(addr(dkv_acc_ws, r[i], sub_tile*128 + (dp_idx//4)*4),
                              frg)
            # instruction_selection: atom.global.add.v4.f32 (attributed to
            #   :254 by inlining, **32 static** = 4 sub-tiles x 8); extent: 4
            #   f32 lanes per issue.  A VECTOR atomic, not four scalar ones.
    named_barrier(6, 256)                                      # :2340
    # instruction_selection: bar.sync 6, 256 (4 static at :2340); extent: the
    #   8 reduce warps.

def atomic_reduce_64(rdKV, r):                                 # :2369-2397
    for i in range(8):
        frg = (rdKV[base_c], rdKV[base_c+2])                   # 2-wide
        if r[i] >= 0:
            atomic_add_global(addr(dkv_acc_ws, r[i], 512 + (dp_idx//4)*2), frg)
            # instruction_selection: atom.global.add.v2.f32 (attributed to
            #   :254 by inlining, 8 static); extent: 2 f32 lanes.  d576 only.
    named_barrier(6, 256)                                      # :2397
    # instruction_selection: bar.sync 6, 256 (:2397, 1 static); extent: the 8
    #   reduce warps.

# ===========================================================================
# Kernel 3 of 4: convert  (:609-646)
#   grid (ceil_div(S_kv, BLOCK_SEQ), 1, 1), block [32, NUM_THREADS_SEQ, 1]
# ===========================================================================

def convert():
    seq_id = BLOCK_SEQ * block_id_x() + thread_id_y()          # :623
    if seq_id >= S_kv:
        return
    for i in range(HEAD_DIM_MAIN // 64):                       # :632, unrolled
        for j in range(2):                                     # :633, unrolled
            v = load_global(dkv_acc_ws, (seq_id, tidx + j*32 + i*64))
            # instruction_selection: ld.global.b32 (:634, 16 static); extent:
            #   SCALAR.  Despite `convert_elem_per_load = 4` (:571) the export
            #   shows 16 scalar b32 loads, not four v4 loads -- the source
            #   constant does not reach this access.
            dim = tidx//4 + (tidx%4)*8 + j*32 + i*64           # :635
            # The inverse of the 128-wide store's fragment gather.
            store_global(dkv, (seq_id, dim), cast(v, ELEM))
            # instruction_selection: cvt.rn.bf16.f32 (**:254**, 16 static -- the
            #   cast is attributed to the outer entry, not to :636) then
            #   st.global.b16 (:636, 16 static); extent: scalar.
    if HAS_TAIL:
        for j in range(2):                                     # :642
            v = load_global(dkv_acc_ws, (seq_id, tidx + j*32 + HEAD_DIM_MAIN))
            # instruction_selection: ld.global.b32 (:643, 2 static); extent:
            #   scalar.  d576 only.
            k = tidx//2 + j*16                                 # :644
            dim = HEAD_DIM_MAIN + (k//8)*16 + k%8 + (tidx%2)*8  # :645
            # A DIFFERENT inverse: the 64-wide store used a 2-wide fragment,
            # so its scramble is not the 128-wide one.
            store_global(dkv, (seq_id, dim), cast(v, ELEM))
            # instruction_selection: cvt.rn.bf16.f32 (:254) then st.global.b16
            #   (:646, 2 static); extent: scalar.  d576 only.

# ===========================================================================
# Kernel 4 of 4: sum_dSink  (:709-738)
#   grid (ceil_div(S_q, 256), H, 1), block [32, 1, 1]
# ===========================================================================

def sum_dSink():
    q_block, head, _ = block_id()
    q_end = min(S_q, (q_block + 1) * 256)                      # :722
    q_idx = q_block * 256 + tidx                               # :723
    sink_log2 = load_global(attn_sink, (head,)) * log2_e       # :726
    # instruction_selection: ld.global.b32 (:726, 1 static) then mul.f32
    #   (:726, 1 static); extent: scalar, once per CTA.
    acc = f32(0.0)
    while q_idx < q_end:                                       # :729
        p_sink = exp2(sink_log2 + load_global(scaled_lse_ws, (head, q_idx)))
        # instruction_selection: ld.global.b32 (:730, 15 static) then add.f32
        #   (:730, 15 static) feeding
        # instruction_selection: ex2.approx.f32 (:730, 15 static); extent:
        #   scalar.  No `.ftz` here either.  `scaled_lse` is already
        #   `-lse_with_sink` in log2 units, so this one exp2 is the whole
        #   sink probability.
        acc += p_sink * load_global(sum_OdO_ws, (head, q_idx))
        # instruction_selection: ld.global.b32 (:731, 15 static) then
        #   fma.rn.f32 (:731, 15 static); extent: scalar.
        q_idx += 32                                            # :732
    acc = warp_reduce_add(acc, group=32)                       # :734
    # instruction_selection: shfl.sync.bfly.b32 (:734, 5 static = log2(32));
    #   extent: 5-step butterfly.
    if tidx == 0:
        atomic_add_global(addr(d_sink, head), acc)             # :738
        # instruction_selection: atom.global.add.f32 (attributed to :254, 1
        #   static); extent: SCALAR -- one lane, one f32.  Contrast the dKV
        #   reduction's v4 form.
```

## Logical GEMM ownership

Every GEMM is issued by warp 16 and lands in TMEM. `M` is always the axis the
accumulator is indexed by.

| accumulator | formula | A operand (smem) | B operand (smem) | mma tiler | cta tiler | K-phases | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `S` | `Q @ K^T` | `sQ`, K-major | `sK`, K-major | `(64,64,D)` | -- | 32, unroll 4 | `:322`, `:1502` |
| `dP` | `dO @ V^T` | `sdO`, K-major | `sV`, K-major | `(64,64,512)` | -- | 32, unroll 4 | `:327`, `:1518` |
| `dKV0..3` | `dO^T @ P` then `+= Q^T @ dS` | `sdOT` / `sQT`, MN-major | `sP` / `sdS`, K-major | `(128,64,64)` | `(512,64,64)` | 4, unroll 2 | `:332`, `:336` |
| `dQ0..3` | `K @ dS^T` | `sK_2`, MN-major | `sdST`, MN-major | `(128,64,64)` | `(512,64,64)` | 4, unroll 2 | `:340`, `:1624` |
| `dKV4` | `Q^T[512:576] @ dS` | `sQT_tail`, MN-major | `sdS`, K-major | `(64,64,64)` | -- | 4 | `:346`, `:1610` |
| `dQ4` | `K[512:576] @ dS^T` | `sK_tail`, MN-major | `sdST`, MN-major | `(64,64,64)` | -- | 4 | `:350`, `:1673` |

All six use `tcgen05.mma.cta_group::1.kind::f16` with an instruction descriptor
in a register; the operand major mode is carried by the SMEM matrix descriptor,
not by a different opcode. 32 static MMA instructions in the d512 build.

The dV/dK fusion is the `accumulate=True` on the second GEMM into the same
columns: `dKV[:, 0:512]` receives `dV + dK` and `dKV[:, 512:576]` receives `dK`
alone, because `V` is only the leading 512 columns of the shared `KV` buffer.

## TMEM column map

512 columns, allocated whole by warp 4. `block_tile = 64` columns per tile.

| tensor | d512 column | d576 column | aliases | source |
| --- | --- | --- | --- | --- |
| `S` | 0 | 0 | -- | `:133` |
| `dP` | 0 **+ lane offset `16 << 16`** | same | shares S's columns at a different lane range | `:134`, `:2675-2677` |
| `dKV0` | 64 | 64 | -- | `:135` |
| `dKV1` | 128 | 128 | -- | `:136` |
| `dKV2` | **448** | 64 | d512: the free dQ4 slot. d576: aliases `dKV0`. | `:2684` / `:2687` |
| `dKV3` | 128 | 128 | aliases `dKV1` in both | `:2685` / `:2688` |
| `dQ0..dQ3` | 192, 256, 320, 384 | same | live across every tile | `:139-142` |
| `dQ4` | -- | 448 | d576 only | `:143` |
| `dKV4` | -- | 64 | aliases `dKV0` | `:144` |

The two maps are genuinely different, not just a renaming: at d512 `dKV2` gets
its own column range because `dQ4` does not exist, so only `dKV3` aliases. That
is why the barrier assignment at `:1544-1546` and `:2260` differs from
`:1541-1543` and `:2273`.

## Named barriers

| id | threads | participants | purpose | source |
| --- | --- | --- | --- | --- |
| 1 | 640 | -- | declared, never used | `:82` |
| 2 | 416 | compute + reduce + mma | TMEM allocation retrieve | `:86`, `:856` |
| 3 | 128 | compute warps | intra-role ordering and the dQ store rounds | `:90` |
| 4 | 32 | -- | declared, never used | `:94` |
| 5 | 128 | load_KV warps | joint gather visibility before the pipeline commit | `:98`, `:1392` |
| 6 | 256 | reduce warps | intra-role ordering after each atomic batch | `:102`, `:2340` |
| 7 | 288 | reduce + mma | WAR: dKV0/dKV1 reads before their columns are rewritten. **Unconditional in both builds** (`:1602` at d576, `:1692` at d512; `:2231`) | `:106` |
| 8 | 288 | reduce + mma | d512: the generation-A WAR, **skipped once** (`:1546`, `:2260`, tail `:1753`). d576: the dKV4 WAR, unconditional (`:1690`, `:2246`) | `:110` |
| 9 | 288 | reduce + compute warp 4 | reduce warps drained before TMEM dealloc | `:120` |
| 10 | 288 | reduce + mma | d576 only: the generation-A WAR, **skipped once** (`:1543`, `:2273`, tail `:1757`). **Entirely unused at d512.** | `:128` |

Barriers 7, 8 and 10 are the WAR set, but they are not symmetric and only **one**
of them is skipped-once per build:

| build | skipped once, compensated at the tail | unconditional | unused |
| --- | --- | --- | --- |
| d512 | **8** | 7 | 10 |
| d576 | **10** | 7, 8 | -- |

So the loop tail carries exactly one arrive (`bar.sync 8` at d512, `bar.sync 10`
at d576), not one per WAR barrier. A port that compensates a barrier that was
never skipped, or fails to compensate the one that was, hangs -- and only for
some `tile_count`, which is what makes it expensive to find later.

## Storage aliases and lifetimes

Base offsets, element shapes and swizzle are given at the declaration site in the
complete sketch above; this table carries only the lifetime rules, which are the
part a reader has to hold in their head while reading the role bodies.

All sixteen layouts share swizzle **S<3,4,3>** (128-byte). The measured element
shape:stride per alias, from `probe/probe_smem_layouts.py`:

| form | shape:stride (elements) | who uses it |
| --- | --- | --- |
| K-major operand, `sQ`/`sK` | `((64,16),1,(4, HEAD_DIM//64),1):((64,1),0,(16,4096),0)` -- `(4,8)` at d512, **`(4,9)`** at d576 (cosize 32768 / 36864) | `sQ`, `sK` |
| K-major operand, `sdO`/`sV` | `((64,16),1,(4,8),1):((64,1),0,(16,4096),0)` -- always `(4,8)`, since `HEAD_DIM_V` is 512 in both builds | `sdO`, `sV` |
| MN-major operand | `(((64,2),16),4,4,1):(((1,4096),64),8192,1024,0)` | `sQT`, `sK_2`, `sdOT` |
| 64-wide B, K-major | `((64,16),1,4,1):((64,1),0,16,0)` | `sP`, `sdS` |
| 64-wide B, MN-major | `((64,16),1,4,1):((1,64),0,1024,0)` | `sdST` |
| epilogue COL_MAJOR | `((64,1),(8,8),(1,1)):((1,0),(64,512),(0,0))` | `sP_store`, `sdS_store` |
| epilogue 128x64, COL_MAJOR | `((64,2),(8,8),(1,1)):((1,4096),(64,512),(0,0))` | `sdQ` |
| epilogue 64x64, COL_MAJOR | `((64,1),(8,8),(1,1)):((1,0),(64,512),(0,0))` -- the same form as `sP_store`/`sdS_store` | `sdQ4` |
| 9x64 tail split | `((64,16),9,4,1):((1,64),4096,1024,0)`, block 8 | `sK_tail`, `sQT_tail` |

| alias | over | lifetime constraint |
| --- | --- | --- |
| `sQT`, `sQT_tail` | `sQ` | co-live with `sQ`; different operand mapping of the same bytes, never a different value |
| `sV`, `sK_2`, `sK_tail` | `sK` | co-live with `sK` for the whole KV loop |
| `sdQ`, `sdQ4` | `sK` | **not** co-live: only legal after the KV loop drains, i.e. after the MMA warp's last `release(pipe_load_mma_K)` at `:1683` and the compute warp's `wait(pipe_mma_cmp_dQ)` at `:2033` |
| `sdOT` | `sdO` | co-live |
| `sP_store` | `sP` | the stmatrix destination view of the same bytes the dOP MMA reads as `sP` |
| `sdST`, `sdS_store` | `sdS` | same, for dS |

## TensorMap ABI

Four descriptors, all rank 3, all built host-side by the source's
`make_tiled_tma_atom` calls:

| descriptor | direction | tile | built at | issues per copy |
| --- | --- | --- | --- | --- |
| `desc_Q` | G2S | `(64, D)` | `:409` | `D // 64` -- 8 at d512, 9 at d576 |
| `desc_dO` | G2S | `(64, 512)` | `:414` | 8 |
| `desc_dQ` | S2G | `(128, 64)` | `:419` | 2 |
| `desc_dQ_64` | S2G | `(64, 64)` | `:431` | 1, d576 only |

The port encodes all four in its own launcher prologue rather than receiving
them as an ABI argument. `desc_dQ_64` is never prefetched, matching `:815-817`.

## Static specialization boundary

Resolved at trace time, never at runtime: `HEAD_DIM`, `NUM_HEAD`, `ELEM`,
`MAX_TOPK`, `HAS_TOPK_LENGTH`, `SAME_HDIM` and everything derived from it
(`HAS_TAIL`, the TMEM map, the barrier assignment, the tail MMAs and views),
`SUM_ODO_BLOCK_Q`, `BLOCK_SEQ`, `NUM_THREADS_SEQ`, every SMEM byte offset, all
pipeline stage counts and CooperativeGroup sizes, and the `is_first`/`full_tiles`
helper instantiations.

Runtime values: `token_idx`, `head_block_idx`, `topk`, `tile_count`,
`tile_index`, the gathered `topk_idxs`, and every tensor base pointer.

`S_q` and `S_kv` are runtime kernel arguments, so the compile key matches the
source's 7-tuple and one build serves every sequence length.

## TIRx module and benchmark contract

- `KERNEL_META["name"] = "cudnn_sm100_dsa_sparse_attention_backward"`,
  category `cudnn`, `runtime_cuda_archs = ["sm_100a"]`.
- `get_kernel` returns the four device functions **in launch order**; the launch
  closure preserves that order on one stream.
- `dkv`, `d_sink` and both workspaces are accumulated into, so the timed closure
  re-zeroes them every repetition. The upstream wrapper does the same zeroing
  inside its own timed call (`_interface_sm100.py:105-162`), which is what keeps
  the two timing scopes comparable.
- Correctness is checked against a gathered FP32 oracle at `atol = rtol = 5e-2`,
  the upstream test tolerance. **No bitwise comparison against the source is
  available**: `dkv` is accumulated by global FP32 atomics whose order varies
  run to run, upstream included.
- The reference is loaded from a `CUDNN_FRONTEND_PATH` checkout, not the
  installed `cudnn` wheel, which carries an older revision of this kernel.

## Instruction-selection summary

Static counts from `d512_bf16_len`, `bwd` entry only, instruction lines minus
predicated lines.

| count | instruction | what selects it |
| --- | --- | --- |
| **256** | `st.shared.b16` | the dQ epilogue's r2s (`:2543`); `autovec_copy` does **not** vectorize it -- the largest single count in the kernel |
| **256** | `cvt.rn.bf16.f32` | the dQ epilogue's quantize (`:2625`), scalar, one per element |
| 96 | `cp.async.cg.shared.global` | the KV gather; `.cg` from `LoadCacheMode.GLOBAL`, 128 bits from `num_bits_per_copy=128`. 3 call sites x 16 rows x 2 groups. |
| 34 | `shfl.sync.idx.b32` | 32 top-k broadcasts (16 per tile x 2 call sites) + 2 `make_warp_uniform` |
| 34 | `bar.sync` | the ten named barriers, dominated by the dQ store rounds and the reduce batches |
| 32 | `tcgen05.mma.cta_group::1.kind::f16` | every GEMM (36 at d576); operand major mode rides the matrix descriptor, not the opcode |
| 32 | `ex2.approx.ftz.f32` | `P = exp2(...)`; `.ftz` from `fastmath=True` |
| 32 | `cvt.rn.bf16x2.f32` | the P and dS quantize sites only (2 x 16 pairs); the dQ site is the 256 scalar converts above. `f16x2` in the fp16 build |
| 32 | `tcgen05.ld...32x32b.x8.b32` | the dQ epilogue T2R; a different shape from the dKV path |
| 32 | `atom.global.add.v4.f32` | the dKV reduction; 4-wide vector, not 4 scalars |
| 32 | `st.shared.v4.b32` | the KV zero-fill at `:1258` -- the kernel's **only** vectorized shared store |
| 23 | `mbarrier.try_wait.parity.shared.b64` | every pipeline wait and acquire |
| 22 | `mbarrier.init.shared.b64` | 9 pipelines x 1 stage + 1 pipeline x 2 stages, full/empty each |
| 16 | `cp.async.bulk.tensor.3d...mbarrier::complete_tx::bytes` | the Q and dO TMA loads; dO is always 8, Q is `D // 64` (8 at d512, 9 at d576) |
| 16 | `mul.f32x2` | the dS quantize's `* softmax_scale` (`:2624`) |
| 11 | `mbarrier.arrive.shared.b64` | the async-side pipeline commits and releases |
| 16 / 16 / 16 | `fma.rn.f32x2`, `add.rn.f32x2`, `mul.rn.f32x2` | the packed softmax recompute |
| 9 | `tcgen05.commit...b64` | UMMA pipeline commits |
| 8 | `stmatrix...m8n8.x4.trans.shared.b16` | the P and dS transposes, 4 each |
| 8 | `tcgen05.ld...16x256b.x4.b32` | the dKV T2R, 1 per sub-tile per generation |
| 8 | `cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group` | the dQ TMA stores, 2 per round |
| 8 | `tcgen05.wait::ld.sync.aligned` | after every T2R group |
| 8 | `fence.proxy.async.shared::cta` | before every UMMA read of a generic-proxy write |
| 6 | `setmaxnreg.{dec 40, inc 128}` | one per role branch, not one per warpgroup |
| 2 | `tcgen05.ld...16x256b.x8.b32` | the S and dP T2R |
| 2 | `cp.async.ca.shared.global` | LSE and sum_OdO; `.ca` from `LoadCacheMode.ALWAYS` |
| 1 | `tcgen05.alloc...b32` | warp 4 claims all 512 columns |
| 1 | `exit` | the `topk <= 0` early out |

`SYNC` and `NANOSLEEP` in a later profile come from `mbarrier.try_wait` spin
loops and are not standalone targets.
