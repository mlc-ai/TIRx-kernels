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

This design sketch documents a TIRx port of FlashInfer's
include/flashinfer/topk.cuh StableSortTopKByValueKernel.
-->

# stable_sort_topk_by_value SM100: coarse WASP pipeline sketch

This non-executable design sketch describes the storage layout, thread roles,
control flow, and PTX-level operations of
[`tirx_kernels/flashinfer/topk/stable_sort_topk_by_value.py`](../../../../tirx_kernels/flashinfer/topk/stable_sort_topk_by_value.py).
That TIRx module is the authoritative implementation.

The port covers **one kernel**: `StableSortTopKByValueKernel<BLOCK_THREADS,
ITEMS_PER_THREAD, int32_t, DType>` (`topk.cuh:3090-3133`), the `sorted_output`
epilogue of the top-k pipeline. `TopKDispatch` appends it after **both** the
filtered and the radix branch (`:3470-3472`), so it is not owned by either
algorithm; the Python layer requests it as `sorted_cuda = sorted && deterministic
&& k <= 2048` (`flashinfer/topk.py:634`).

Instantiations: `DType in {f32, f16, bf16}` x `(BLOCK_THREADS, ITEMS_PER_THREAD)
in {(32,4), (32,8), (64,8), (64,9), (128,8), (256,8)}` = **18 reachable
specializations**, all exported in `.porting/stable_sort_topk_by_value/ptx/`.
`IdType` is always `int32_t`. `k`, the rung, and `end_bit` are static per config
here, exactly the values `StableSortTopKByValue` (`:3135`) picks per call.

Accepted target SM100/B200. Out of scope, with the launcher predicate that
excludes each: `k > 2048` -> `cudaErrorInvalidValue` (`:3139-3141`); `k <= 1` ->
`cudaSuccess` **with no launch at all** (`:3142-3144`). Tile (`Tx`) primitives are
out of scope everywhere.

## Pipeline at a glance

| Kernel / role | Program | Publication / reuse edges |
| --- | --- | --- |
| all `BLOCK_THREADS` threads of the single CTA (uniform, no warp specialization) | blocked load of one row's `(value, index)` pairs into registers, complementing the monotone value key; `P` digit passes of cub `BlockRadixSort` (rank + exchange); blocked writeback of the same `k` slots | Everything is intra-CTA. The rank counters and the two exchange buffers **alias one shared union**; `bar.sync 0` separates every rank phase from every exchange phase. There are **no atomics, no global synchronization, no workspace, and no cross-CTA edges** — one CTA owns one row and the kernel is fully in place. |

The kernel is a single role. There is no producer/consumer split, no pipe, no
mbarrier, and no asynchronous copy: every global access is a plain synchronous
`ld.global`/`st.global`, and every shared access is a plain `ld.shared`/
`st.shared`. The only synchronization primitive in the whole kernel is
`bar.sync 0`.

**The two tensors are simultaneously input and output.** The kernel reads `k`
slots of a row, sorts them in registers, and writes the same `k` slots back.

## Primitive vocabulary

```python
copy_g2r(reg, gmem)               # ld.global.b32 | ld.global.b16 (NOT .nc: the
                                  #   pointers are not __restrict__ const)
copy_r2g(gmem, reg)               # st.global.b32 | st.global.b16
copy_r2s(smem, reg) / copy_s2r    # shared traffic, always through T.ptx
sort_key(bits)                    # the fused monotone-flip + complement (below)
digit(key, begin, nbits)          # shr + and  (NOT bfe on SM >= 70).  The mask
                                  #   is materialized once per kernel by an
                                  #   inline-asm `bmsk.clamp.b32 (0, 4)` hoisted
                                  #   above the pass loop (`cuda::bitmask<uint32>`
                                  #   is not constant-folded); it yields 0xF, so a
                                  #   TIRx `and` against the immediate 0xF is the
                                  #   sanctioned equivalent.  This is the only
                                  #   opcode family in the export not otherwise
                                  #   accounted for here.
rank_keys(keys) -> ranks          # cub BlockRadixRank, one digit pass
scatter_to_blocked(items, ranks)  # cub BlockExchange, blocked -> blocked
barrier()                         # bar.sync 0
```

### `sort_key` is one fused operation, and it is an involution

The source writes two nested transforms at `:3112-3113`:

```cpp
OrderedType ordered = Traits::ToOrdered(row_values[pos]);   // topk_common.cuh:35-38
keys[i] = static_cast<uint32_t>(static_cast<OrderedType>(~ordered));
```

`ToOrdered` is `sign ? ~bits : bits ^ SIGN_BIT`. Complementing that result
collapses the pair into **a single XOR against an inverted mask**, and the export
confirms nvcc does exactly that — `not.b32` and `not.b16` are **0** on all 18
entries:

```
.loc 2 3113   setp.lt.s32  %p, bits, 0;                # sign bit set?
              selp.b32     %m, 0, 2147483647, %p;      # mask = sign ? 0 : 0x7FFFFFFF
              xor.b32      key, %m, bits;              # key = bits ^ mask
```

so `sort_key(bits) = bits ^ (sign(bits) ? 0 : 0x7FFF_FFFF)` (16-bit: mask
`{0x0000, 0x7FFF}`, ops `setp.lt.s16` / `selp.b16` / `xor.b16`).

**`sort_key` is its own inverse.** The writeback's `FromOrdered(~key)`
(`:3129-3130`) lowers to the *same three instructions with the same constants*,
attributed to `topk_common.cuh` -- `:42` for f32, `:67` for f16, `:93` for bf16
(three separate `RadixTopKTraits` specializations, one per dtype):

```
.loc 19 67     setp.lt.s16  %p, key16, 0;      # f16; bf16 is .loc 19 93
              selp.b16     %m, 0, 32767, %p;
              xor.b16      val, %m, key16;
```

Proof: if the sign bit is set the mask is 0 and the map is the identity; if it is
clear the map flips only bits `[0, width-1)`, leaving the sign bit clear, so
applying it twice cancels. The port therefore needs **one** helper, used in both
directions — not `to_ordered_*` composed with a bitwise-not, which would emit the
`not` the export does not contain.

This map is **not** `utils/topk_radix.py`'s `to_ordered_u32`/`from_ordered_u32`
(mask `{0x80000000, 0xFFFFFFFF}`); those are the plain monotone flip used by the
radix ports. Do not reuse them here.

### `.file` map of the export

`.file 2` = `topk.cuh`, `.file 19` = `topk_common.cuh`, `.file 3` =
`block_radix_sort.cuh`, `.file 4` = `block_exchange.cuh`, `.file 5` =
`block_radix_rank.cuh`, `.file 13` = `block_scan_warp_scans.cuh`, `.file 16` =
`warp_scan_shfl.cuh`, `.file 7` = `cuda/__bit/bitfield.h`. Every `.loc` cited
below uses these indices.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================
variant = specialize(DTYPE=("f32","f16","bf16"),
                     BT_IPT=((32,4),(32,8),(64,8),(64,9),(128,8),(256,8)),
                     target="sm_100a")
# instruction_selection: none; extent: compile-time instantiations

BT, IPT   = finalize_block_config(k)   # the launcher's k ladder (:3154-3166),
                                       # identical to LaunchFinalizeTopKIndices
END_BIT   = 32 if DTYPE == "f32" else 16      # 8 * sizeof(OrderedType) (:3121)
RADIX_BITS = 4                                # cub default
P         = END_BIT // RADIX_BITS             # 8 digit passes (f32) / 4 (16-bit)
SIGN      = 0x80000000 if DTYPE == "f32" else 0x8000
MASK_HI   = 0x7FFFFFFF if DTYPE == "f32" else 0x7FFF

launch_config = launch(grid=(NUM_ROWS,1,1), block=(BT,1,1),
                       dynamic_smem_bytes=0, launch_bounds=BT)
# instruction_selection: none; extent: static launch metadata.  Note there is NO
#   cudaFuncSetAttribute in this launcher -- the smem union is small enough to
#   need no optin (contrast the filtered unified kernel's 128 KiB request).

# ===========================================================================
# Storage.  The ONLY shared memory is cub's TempStorage union (:3097).
# ===========================================================================
# BlockRadixSort::_TempStorage (block_radix_sort.cuh:268-274) is a union of the
# ranking storage with the key and value exchange buffers.  The rank grid
# dominates at every rung, so the realized union is 36 * BT bytes.
s_counters32 = shared((9 * BT,), "uint32", align=16)   # raking view
s_counters16 = view(s_counters32, "uint16")            # per-digit RMW view
s_xchg_keys  = alias(s_counters32, (BT*IPT + pad,), "uint32")
s_xchg_vals  = alias(s_counters32, (BT*IPT + pad,), "uint32")
s_scan       = shared((WARPS + 1,), "uint32", align=32)  # BlockScan scratch:
                                  # warp_aggregates[WARPS] + block_prefix, the
                                  # struct __align__(32) (block_scan_warp_scans.cuh:69-79),
                                  # so 32 B for BT <= 128 and 64 B for BT = 256.
                                  # Outside BlockRadixRank's inner `aliasable`
                                  # union, still inside the outer BlockRadixSort
                                  # union; never overlaps the exchange because
                                  # 4*(BT*IPT+pad) <= 36*BT at every rung.
# instruction_selection: none; extent: static shared layout.
#   pad = (BT*IPT) >> 5 iff IPT == 8 (INSERT_PADDING, block_exchange.cuh:139-140).

keys   = alloc_local((IPT,), "uint32")     # (:3105)
values = alloc_local((IPT,), "uint32")     # (:3106) the indices, carried as satellites
ranks  = alloc_local((IPT,), "int32")

def stable_sort_topk_by_value(out_idx, out_val):
    row = cta_id(axis="x", extent=NUM_ROWS)   # instruction_selection: mov.u32 %ctaid.x
    tx  = thread_id(extent=BT)                # instruction_selection: mov.u32 %tid.x
    row_idx_base = out_idx + row * k          # (:3102)
    row_val_base = out_val + row * k          # (:3103)

    # -----------------------------------------------------------------------
    # Stage 1: blocked load (:3108-3119)
    # -----------------------------------------------------------------------
    # Blocked arrangement: thread tx owns the CONTIGUOUS run [tx*IPT, tx*IPT+IPT).
    # The export shows the per-thread stores at consecutive addresses
    # ([%rd3], +2, +4, +6 for 16-bit; +4/+8/+12 for f32), which is the blocked
    # layout showing through.
    for i in unroll(IPT):                                   # #pragma unroll (:3108)
        pos = tx * IPT + i
        if pos < k:                                          # (:3111)
            v = load_global(row_val_base[pos])
            # instruction_selection: ld.global.b32 (f32) | ld.global.b16 (16-bit)
            #   (.loc 2 3112); extent: one scalar per item.  NOT .nc -- these
            #   pointers are not __restrict__ const, unlike the filtered unified
            #   kernel's inputs.
            keys[i] = sort_key(v)
            # instruction_selection: setp.lt.s32 + selp.b32 + xor.b32 (f32,
            #   .loc 2 3113) | setp.lt.s16 + selp.b16 + xor.b16 + cvt.u32.u16
            #   (16-bit: the key is the zero-extended 16-bit result); extent: one
            #   per item.  ONE fused map, no `not` -- see the vocabulary section.
            values[i] = load_global(row_idx_base[pos])
            # instruction_selection: ld.global.b32 (.loc 2 3114); extent: one
            #   scalar per item.  The index is reinterpreted, never converted.
        else:                                                # (:3115-3118)
            keys[i]   = 0xFFFFFFFF
            values[i] = 0xFFFFFFFF
            # instruction_selection: mov.b32 of the immediate; extent: two
            #   registers.  PADDING INVARIANT: ~0u is maximal within [0, END_BIT),
            #   and every real key fits in END_BIT bits, so padding sorts to the
            #   tail of the ascending order and is never written back.

    # -----------------------------------------------------------------------
    # Stage 2: cub BlockRadixSort, ascending, blocked -> blocked (:3121-3122)
    # -----------------------------------------------------------------------
    # `Sort`, not `SortDescending`: descending-by-value already lives in the
    # complemented key.  cub therefore instantiates DESCENDING=false, so
    # DescendingBlockRadixRank (block_radix_rank.cuh:462-467) is never reached,
    # and because KeyT is uint32 its float twiddling is the identity.
    for (begin_bit, nbits) in static_passes(END_BIT):   # P passes, RADIX_BITS=4
        rank_keys(keys, ranks, begin_bit, nbits)
        # rank = reset the 9 padded counter lanes; per-item uint16 counter RMW
        #   (each thread owns its tid column, NO atomics); memoized 9-word raking
        #   upsweep over a CONTIGUOUS per-thread segment; BlockScan ExclusiveSum
        #   with the `block_prefix = aggregate << 16` packing trick; exclusive
        #   downsweep; ranks[i] = thread_prefix[i] + reloaded counter.
        # instruction_selection: st.shared.b32 x9 (reset) + ld.shared.b16 x2*IPT
        #   + st.shared.b16 x IPT (counter RMW and rank reload) + ld/st.shared.b32
        #   (raking) + shfl.sync.up.b32 x5 (one warp scan) + **bar.sync 0 x4**;
        #   extent: one per digit pass.  The four barriers are
        #   `block_radix_rank.cuh:479` (separates the per-item uint16 counter RMW
        #   from the raking upsweep), `block_scan_warp_scans.cuh:174` and `:400`
        #   (inside ExclusiveSum), and `block_radix_rank.cuh:484` (separates the
        #   raking downsweep from the rank reload).  None is incidental: dropping
        #   :479 races the RMW against the upsweep, dropping :484 races the
        #   downsweep against the reload.
        #   Digit extraction is shr + and -- cuda::bitfield_extract falls through
        #   to shift+mask on SM >= 70, and bfe.u32 is 0 in the export; the `and`
        #   mask itself is produced by one loop-invariant inline-asm
        #   `bmsk.clamp.b32 (0, 4)` per entry (see the digit() vocabulary note).
        #   No match.any, no ballot, no PRMT, no atom.
        barrier()
        # instruction_selection: bar.sync 0; extent: CTA-wide
        scatter_to_blocked(keys, ranks, s_xchg_keys)
        # instruction_selection: st.shared.b32 to pad(rank) + bar.sync 0 +
        #   ld.shared.b32 from pad(tx*IPT+i); pad(x) = x + (x>>5) iff IPT == 8;
        #   extent: one per digit pass.  At IPT == 4 the gather is instead a
        #   single `ld.shared.v4.b32` per exchange (.loc 4 627) -- which is
        #   precisely why cub omits padding at that rung ("otherwise we can
        #   typically use 128b loads", block_exchange.cuh:138).
        barrier()
        # instruction_selection: bar.sync 0; extent: CTA-wide
        scatter_to_blocked(values, ranks, s_xchg_vals)
        # instruction_selection: same family as the key scatter; extent: one per
        #   digit pass.  The indices ride as satellite data -- this is what makes
        #   the sort carry them, and it is the whole reason ValueT is non-null.
        if not last_pass:
            barrier()
            # instruction_selection: bar.sync 0; extent: CTA-wide
    # SYNC ACCOUNTING: 9 bar.sync per key+value pass, minus the trailing one on
    # the final pass = cub's 9P - 1 (71 for f32, 35 for the 16-bit dtypes).  The
    # export shows bar.sync = 9 STATIC on every instantiation, i.e. the pass loop
    # is ROLLED (one backward branch), exactly like the merged finalize kernel.

    # -----------------------------------------------------------------------
    # Stage 3: blocked writeback (:3124-3132)
    # -----------------------------------------------------------------------
    for i in unroll(IPT):                                   # #pragma unroll (:3124)
        pos = tx * IPT + i
        if pos < k:                                          # (:3127)
            store_global(row_idx_base[pos], values[i])
            # instruction_selection: st.global.b32 (.loc 2 3128); extent: one
            #   scalar per item.  Padding lanes are simply never written; there is
            #   no -1 fixup here (contrast the finalize kernel's max.s32).
            store_global(row_val_base[pos], sort_key(keys[i]))
            # instruction_selection: (16-bit only) cvt.u16.u32 to narrow the key
            #   (.loc 2 3129), then setp.lt.s16 + selp.b16 + xor.b16
            #   (.loc 19 67 for f16, .loc 19 93 for bf16)
            #   + st.global.b16 (.loc 2 3130); f32 is setp.lt.s32 + selp.b32 +
            #   xor.b32 + st.global.b32; extent: one scalar per item.
            #   THE SAME `sort_key` AS THE LOAD -- the map is an involution, and
            #   the export shows identical opcodes and identical mask constants on
            #   both sides.
```

## Launcher ladder and pass count

| k range | BLOCK_THREADS | ITEMS_PER_THREAD | capacity | f32 passes | 16-bit passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| <= 128 | 32 | 4 | 128 | 8 | 4 |
| <= 256 | 32 | 8 | 256 | 8 | 4 |
| <= 512 | 64 | 8 | 512 | 8 | 4 |
| <= 576 | 64 | 9 | 576 | 8 | 4 |
| <= 1024 | 128 | 8 | 1024 | 8 | 4 |
| <= 2048 | 256 | 8 | 2048 | 8 | 4 |

Identical to `LaunchFinalizeTopKIndices` (`:3060-3079`), so
`finalize_block_config` is imported rather than duplicated.

## Shared-memory budget

| (BT, IPT) | rank grid | key exchange | value exchange | realized union | scan block | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (32, 4) | 1152 | 512 | 512 | **1152** | 32 | **1184** |
| (32, 8) | 1152 | 1056 | 1056 | **1152** | 32 | **1184** |
| (64, 8) | 2304 | 2112 | 2112 | **2304** | 32 | **2336** |
| (64, 9) | 2304 | 2304 | 2304 | **2304** | 32 | **2336** |
| (128, 8) | 4608 | 4224 | 4224 | **4608** | 32 | **4640** |
| (256, 8) | 9216 | 8448 | 8448 | **9216** | 64 | **9280** |

Bytes. The rank grid (`36 * BT`) dominates the union at every rung. The totals are
the realized `.shared .align 16 .b8 temp_storage[N]` of the export (1184 / 2336 /
4640 / 9280, identical across dtypes), not a derived figure.

The scan block is `BlockScanWarpScans::_TempStorage`
(`block_scan_warp_scans.cuh:69-79`): `warp_aggregates[WARPS]` +
`WarpScanT::TempStorage warp_scan[WARPS]` (empty for the shuffle scan) +
`block_prefix`, the whole struct `__align__(32)` -- so it rounds to 32 bytes for
BT <= 128 and 64 for BT = 256, **not** the raw `(WARPS + 1) * 4` payload.

Placement: the scan block sits outside `BlockRadixRank`'s inner `aliasable` union
but still inside the outer `BlockRadixSort` union, exactly as in cub. It never
overlaps the exchange buffers because `4 * (BT*IPT + pad) <= 36 * BT` at every
rung, with equality at (64, 9).

Dynamic shared memory is **0** in the source; TIRx pools are `shared.dyn`, so the
port allocates through the pool and the launch carries
`tirx.use_dyn_shared_memory` -- the same sanctioned mechanism deviation the merged
finalize kernel already carries. The byte total must match the column above.

## Static specialization boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| `(BLOCK_THREADS, ITEMS_PER_THREAD)` | static per config | register array extents, exchange padding, warp count |
| `END_BIT` (32 / 16) | static from DType | pass count P = 8 / 4 |
| `k` | static per config here, a kernel arg in the source | turns the guard operand into an immediate. The `pos < k` predicate itself **must remain**: `pos = tx * IPT + i` is a runtime value, so the branch survives at every rung where `k < BT * IPT`, and only disappears at the six exact capacities (128/256/512/576/1024/2048). Dropping it would read and write outside the k-element row and feed unpadded garbage into the sort. |
| `max_len` | **dead** | accepted at `:3093`, present in the launch arg pack (`:3147`), never read — dropped in the port |
| `IdType = int32` | static | the satellite payload is always a 32-bit index |
| padding value `~0u` | static | sorts last under the restricted `END_BIT` |
| tie order | **input-dependent** | stability preserves the incoming order; it does not create index-ascending order |

## TIRx module and benchmark contract

- `KERNEL_META = {"name": "stable_sort_topk_by_value", "category": "flashinfer",
  "compute_capability": 10}`; device symbol `stable_sort_topk_by_value`.
- Plain TIRx only: a `T.SMEMPool` arena, explicit loops, and native `T.ptx.*`
  forms for every key operation. Global and shared memory are reached exclusively
  through `T.ptx.ld/st.*` on `buffer.ptr_to([...])` — never a native
  `BufferLoad`/`BufferStore`, which the repository's low-level IR contract
  forbids. No `T.cuda.func_call`. No `Tx` tile primitives anywhere.
- The cub collective is the **already-merged** `utils/block_radix_sort.py`
  (`alloc_sort_smem`, `emit_block_radix_sort`), reused unchanged: it already
  sorts `uint32` keys with `uint32` satellites over an arbitrary `end_bit`, with
  contiguous raking segments and the `uint16` counter view.
- The source exposes **no FFI or Python entry for this kernel alone** — the
  pipeline FFI launches main(+finalize)+sort. Correctness and the benchmark
  reference therefore both call the source's own `StableSortTopKByValue<DType,
  int32_t>` launcher, compiled through `torch.utils.cpp_extension` with the JIT's
  own flags (precedent: `tirx_kernels/basic/nvfp4_gemm.py:72-273`), built in
  `prepare_bench` before the workload owns a GPU. An independent
  `torch.sort(..., descending=True, stable=True)` oracle checks the reference
  itself.
- Comparison is **bit-exact positional equality on both tensors**: the kernel is
  fully deterministic, and stability makes the output a function of the input
  order alone.

## Instruction selection is a lowering consequence

| Primitive / pattern | PTX family (fresh sm_100a export, 18 entries) |
| --- | --- |
| value load | `ld.global.b32` (f32) / `ld.global.b16` (16-bit) — **not** `.nc` |
| index load / store | `ld.global.b32` / `st.global.b32` |
| `sort_key`, f32 (`.loc 2 3113`, `.loc 19 42`) | `setp.lt.s32` + `selp.b32 {0, 0x7FFFFFFF}` + `xor.b32` |
| `sort_key`, 16-bit (`.loc 2 3113`; back: `.loc 19 67` f16 / `.loc 19 93` bf16) | `setp.lt.s16` + `selp.b16 {0, 0x7FFF}` + `xor.b16` |
| key widen / narrow (16-bit only) | `cvt.u32.u16` on load, `cvt.u16.u32` on writeback |
| complement `~ordered` | **fused into the XOR mask** — `not.b32` = `not.b16` = 0 |
| digit extraction | `shr` + `and` (shift+mask on SM >= 70); `bfe.u32` = 0 |
| rank counter RMW | `ld.shared.b16` x2*IPT + `st.shared.b16` xIPT, **no atomics** |
| rank raking / downsweep | `ld.shared.b32` / `st.shared.b32` |
| block scan | `shfl.sync.up.b32` x5 (one warp scan; `exclusive = inclusive - input`) |
| exchange | `st.shared.b32` -> `bar.sync 0` -> `ld.shared.b32`, padded iff IPT == 8; at IPT == 4 the gather is one `ld.shared.v4.b32` (`.loc 4 627`) |
| block-scan aggregate read | `ld.shared.v2.b32` at BT = 64, `ld.shared.v4.b32` at BT = 128/256 (`.loc 13 178`, `.loc 13 130`) |
| digit mask | one hoisted inline-asm `bmsk.clamp.b32 (0, 4)` per entry -> constant `0xF` |
| synchronization | `bar.sync 0` only — 9 static, dynamic `9P - 1` |
| absent everywhere | `match.any`, `vote.sync`, `prmt`, `popc`, `atom`, `clz`, `ld.global.nc.*` |

Static opcode counts per exported instantiation (full table in
`.porting/stable_sort_topk_by_value/ptx/opcode_table.md`):

| op | f32 (32,4) | f32 (32,8) | f32 (256,8) | f16 (32,4) | f16 (32,8) | bf16 (64,9) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bar.sync` | 9 | 9 | 9 | 9 | 9 | 9 |
| `shfl.sync.up.b32` | 5 | 5 | 5 | 5 | 5 | 5 |
| `ld.shared.b16` | 8 | 16 | 16 | 8 | 16 | 18 |
| `st.shared.b16` | 4 | 8 | 8 | 4 | 8 | 9 |
| `xor.b32` | 8 | 16 | 16 | 0 | 0 | 0 |
| `xor.b16` | 0 | 0 | 0 | 8 | 16 | 18 |
| `not.b32` / `not.b16` | 0 | 0 | 0 | 0 | 0 | 0 |
| `bfe.u32` | 0 | 0 | 0 | 0 | 0 | 0 |

`xor` counts at exactly `2 * IPT` on every entry — one application of `sort_key`
on the load and one on the writeback, in the dtype's own width — while `not` is
uniformly 0. That pair of counts is the check that the port fused the map rather
than composing a monotone flip with a complement.

`bar.sync` holding at 9 static across every rung and dtype is the rolled pass
loop: the dynamic count varies with `P`, the static count does not.
