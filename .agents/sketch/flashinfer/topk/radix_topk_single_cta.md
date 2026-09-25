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
include/flashinfer/topk.cuh RadixTopKKernel_Unified, restricted to the
SINGLE_CTA=true specialization the host launchers select when one CTA
can stage a whole row in shared memory.
-->

# radix_topk_single_cta SM100: coarse WASP pipeline sketch

This non-executable design sketch describes the storage layout, thread roles,
control flow, and PTX-level operations of
[`tirx_kernels/flashinfer/topk/radix_topk_single_cta.py`](../../../../tirx_kernels/flashinfer/topk/radix_topk_single_cta.py).
That TIRx module is the authoritative implementation.

The instantiations are `DTYPE in {f32, f16, bf16}` x `MODE in {Basic,
PageTableTransform, RaggedTransform}` x `DETERMINISTIC in {false, true}` x
`VEC_SIZE in {4,2,1}` (f32) / `{8,4,2,1}` (f16, bf16), each mirroring one
`RadixTopKKernel_Unified<1024, VEC_SIZE, true, DETERMINISTIC, MODE, DType,
int32_t>` template instantiation. `IdType` is always `int32_t` (`csrc/topk.cu`).
`num_rows`, `length`, `k`, and the derived `chunk_size` / grid / dynamic-smem
size are static per config, exactly like the per-launch values
`RadixTopKMultiCTA` (:2172), `RadixTopKPageTableTransformMultiCTA` (:1939) and
`RadixTopKRaggedTransformMultiCTA` (:2055) compute at every call.

The accepted target is SM100/B200. Out of scope, because the source itself
routes them elsewhere: `SINGLE_CTA=false` (the cross-CTA `RadixRowState`
barrier/histogram protocol), per-row `top_k_arr` (`csrc/topk.cu` always passes
`nullptr`), and the FilteredTopK / cluster algorithms that `TopKDispatch`
(:3446) selects for other shapes, tie-break modes, and `dsa_graph_safe`.
Tile (`Tx`) primitives are out of scope everywhere.

Because `ctas_per_group == 1`, `group_id == blockIdx.x`, `cta_in_group == 0`,
`chunk_start == 0`, `actual_chunk_size == length`, and every
`AdvanceRadixGroupBarrier` / `RadixGroupResetStateLastCTA` /
`state->histogram` / `det_scratch` reference is compile-time dead
(`state == nullptr`). The sketch shows only what survives.

## Pipeline at a glance

| Warps | Role-local program | Publication/reuse edges |
| --- | --- | --- |
| all 32 warps (uniform) | Every thread of CTA `blockIdx.x` runs the same single-role program: persistent row loop; per-row header loads; mode trivial-branch early-out; block-strided vectorized row load + monotone key flip into `s_ordered`; `NUM_ROUNDS` radix rounds over `s_hist`/`s_suffix`; block-strided gt/eq count with a warp shuffle reduction; one collect pass; mode epilogue. | all edges are CTA-internal: `s_ordered` published once per row by the load loop, re-read by every round, the counting pass and the collect pass; `s_hist`/`s_suffix`/`s_scalars` are republished each round. Every publication point is one `bar.sync 0`. |

There is no warp specialization, no producer/consumer split, no mbarrier, no
async copy and no cluster. The only cross-thread mechanisms are `bar.sync 0`,
shared-memory atomics, and one warp shuffle reduction (plus the deterministic
collect's block scan, which is itself shuffle + shared based).

`s_hist` and `s_suffix` are deliberately aliased to two different roles inside
one row iteration: during the rounds they are the 256-bin histogram and its
suffix sum; after the rounds `s_suffix[0]`/`s_suffix[1]` become the row-wide
gt/eq accumulators, and during collect `s_hist[0..4]` become the collect
counters. The source does the same by macro (:908-910, :1029-1033) and by direct
reuse of `suffix_sum[0..1]` as the gt/eq accumulators (:782-786, :825-827).

## Primitive vocabulary

Structural operations declare placement without moving data:

```python
specialize(...)        # compile-time variant selection
launch(...)            # compile-time launch topology and attributes
raw_shared(...)        # the one dynamic shared allocation
static_shared(...)     # a separate static __shared__ object (deterministic collect only)
view(...)              # typed view at a fixed byte offset in that allocation
reg_tile(...)          # per-thread register tile
```

Copies state their direction and width:

```python
copy_g2r_v(src_addr, dst_bits)      # one VEC_SIZE*sizeof(DType)-byte global -> register load
copy_g2r_scalar(src_addr, dst)      # one scalar global -> register load
copy_r2s(src, dst_slot)             # one register -> shared store
copy_s2r(src_slot, dst)             # one shared -> register load
copy_r2g_scalar(src, dst_addr)      # one scalar register -> global store
```

The compute vocabulary is deliberately primitive:

```python
to_ordered(dst, bits)     # monotone float bits -> unsigned key (RadixTopKTraits::ToOrdered)
from_ordered(dst, key)    # inverse (RadixTopKTraits::FromOrdered)
and_(dst, a, b); or_(dst, a, b); xor_(dst, a, b); not_(dst, a)
shr(dst, a, n); shl(dst, a, n)
add(dst, a, b); sub(dst, a, b); min_(dst, a, b); max_(dst, a, b)
cmp_eq/cmp_gt/cmp_ge/cmp_lt(pred, a, b)
select(dst, pred, a, b)
move(dst, src)
```

Schedule and cross-thread operations:

```python
thread_id(...); cta_id(...)
barrier()                        # CTA barrier
atomic_add_shared(dst, slot, v)  # shared-memory read-modify-write returning the old value
shfl_down(dst, src, delta)       # warp shuffle
block_exclusive_sum(dst, src)            # CTA-wide exclusive scan (cub BlockScan, RAKING_MEMOIZE)
block_exclusive_sum_pair(dst2, src2)     # the same scan over a (gt, eq) pair
```

Address expressions, loop bounds, and guards are shown directly; they do not
hide copies, computation, role changes, or synchronization.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

variant = specialize(DTYPE=("f32", "f16", "bf16"),
                     MODE=("Basic", "PageTableTransform", "RaggedTransform"),
                     DETERMINISTIC=(False, True),
                     VEC_SIZE=(8, 4, 2, 1),
                     target="sm_100a")
# instruction_selection: none; extent: compile-time instantiations

BLOCK      = 1024                      # BLOCK_THREADS (:2177)
RADIX      = 256
OBITS      = 32 if DTYPE == "f32" else 16          # sizeof(OrderedType)*8
NUM_ROUNDS = OBITS // 8                            # 4 (f32) | 2 (f16, bf16)
OBYTES     = OBITS // 8                            # 4 | 2

# Launcher arithmetic, replicated at specialization time
# (Basic :2177-2213; PageTable :1946-1980; Ragged :2063-2097; smem budget :43-59):
#   VEC_SIZE      = 1 if (MODE != Basic and row_starts) else gcd(16 // sizeof(DType), LENGTH)
#   avail         = max_smem_optin - 2080 - (8512 if DETERMINISTIC else 0)
#   max_chunk     = max(round_down(avail // OBYTES, VEC_SIZE), VEC_SIZE * BLOCK)
#   ctas_per_grp  = ceil_div(LENGTH, max_chunk)          == 1 in this specialization
#   CHUNK         = min(round_up(LENGTH, VEC_SIZE), max_chunk)  == LENGTH
#   NUM_GROUPS    = max(1, min(num_sms, NUM_ROWS))
# `aligned_size` (:608) is not a compile-time value in general: it derives from
# `actual_chunk_size`, which is `stride` (static) only in Basic mode and the
# per-row runtime `lengths[row_idx]` (:1235) in the other two modes.  It is
# therefore computed inside the row loop, not here.

FIXED_BYTES = 2080                     # round_up(4*(256+256+5), 16) (:1194-1200)
SMEM_BYTES  = FIXED_BYTES + CHUNK * OBYTES

launch_config = launch(
    grid=(NUM_GROUPS, 1, 1),
    block=(BLOCK, 1, 1),
    dynamic_smem_bytes=SMEM_BYTES,     # cudaFuncSetAttribute + cudaLaunchKernel (:2255-2259)
    launch_bounds=BLOCK,               # __launch_bounds__(1024)
)
# instruction_selection: none; extent: static launch metadata

# ===========================================================================
# Shared storage: one dynamic allocation carved by hand (:1192-1201)
# ===========================================================================

smem      = raw_shared(dtype="u8", bytes=SMEM_BYTES, alignment=16)
s_hist    = view(smem[0:1024],    dtype="u32", shape=(256,))   # local_histogram
s_suffix  = view(smem[1024:2048], dtype="u32", shape=(256,))   # suffix_sum
s_scalars = view(smem[2048:2068], dtype="u32", shape=(5,))     # prefix, rem_k, bucket, rem_k', out_ctr
s_ordered = view(smem[2080:SMEM_BYTES], dtype="u32" if OBYTES == 4 else "u16",
                 shape=(CHUNK,))                               # shared_ordered, 16B-aligned base
# instruction_selection: none; extent: static shared layout (PTX confirms the +2080 base)

# A second, *static* shared object exists only when DETERMINISTIC (:1095-1102):
#   union DeterministicCollectScanTempStorage {
#       cub::BlockScan<uint32_t,  1024, RAKING_MEMOIZE>::TempStorage scalar;
#       cub::BlockScan<CountPair, 1024, RAKING_MEMOIZE>::TempStorage pair;
#   } __shared__ scan_temp_storage;
if DETERMINISTIC:
    s_scan_temp = static_shared(bytes=8480, alignment=16)
    # lifetime: the whole RadixCollectIndicesDeterministic body -- the scan and both
    #   emit loops.  Disjoint from the dynamic arena, so a DETERMINISTIC config's
    #   total shared footprint is SMEM_BYTES + 8480.  The launcher pre-pays for it by
    #   subtracting 2 * sizeof(ScalarBlockScan::TempStorage) == 8512 from the chunk
    #   budget (:43-52), which is why the deterministic single-CTA domain is smaller.
# instruction_selection: none; extent: `.shared .align 16 .b8 ...scan_temp_storage[8480]`,
#   present in every DETERMINISTIC=true entry and absent from the others

def radix_topk_single_cta(
        input,                  # DType [NUM_ROWS, LENGTH]
        output_indices,         # int32 [NUM_ROWS, K]
        output_values,          # DType [NUM_ROWS, K]; Basic only
        aux_data,               # int32; PageTable: src_page_table, Ragged: offsets
        lengths,                # int32 [NUM_ROWS]; non-Basic only
        row_starts,             # int32 [NUM_ROWS] | null; non-Basic only
        page_table_row_starts,  # int32 [NUM_ROWS] | null; PageTable only
        row_to_batch,           # int32 [NUM_ROWS] | null; PageTable only
        aux_stride,             # int64; PageTable only
        ):
    group_id = cta_id(axis="x", extent=NUM_GROUPS)
    # instruction_selection: mov.u32 %ctaid.x; extent: scalar per thread
    tx = thread_id(extent=BLOCK, dtype="uint32")
    # instruction_selection: mov.u32 %tid.x; extent: scalar per thread

    total_iterations = ceil_div(NUM_ROWS, NUM_GROUPS)          # (:1213-1214)

    # =======================================================================
    # Persistent row loop (:1218-1220)
    # =======================================================================
    for it in range(total_iterations):                         # static trip count
        row_idx = group_id + it * NUM_GROUPS
        cmp_ge(p_done, row_idx, NUM_ROWS)
        # instruction_selection: setp.ge.u32; extent: scalar
        if p_done: break
        # instruction_selection: bra; extent: loop exit predicate

        # ---- per-row header (:1221-1247) ----------------------------------
        if MODE == "Basic":
            row_start = 0
            page_start = 0
            length = LENGTH                                    # stride
            k = K                                              # top_k_arr is null
        else:
            if row_starts is not null:
                copy_g2r_scalar(row_starts + row_idx, row_start)
                # instruction_selection: ld.global.b32; extent: one scalar load
            else:
                move(row_start, 0)
            if MODE == "PageTableTransform" and page_table_row_starts is not null:
                copy_g2r_scalar(page_table_row_starts + row_idx, page_start)
                # instruction_selection: ld.global.b32; extent: one scalar load
            else:
                move(page_start, row_start)
            copy_g2r_scalar(lengths + row_idx, length)
            # instruction_selection: ld.global.b32; extent: one scalar load
            k = K
        row_input   = input + row_idx * LENGTH + row_start      # (:1227)
        row_output  = output_indices + row_idx * K              # (:1240)
        # instruction_selection: address family (mul.wide.u32 / mul.wide.s32 / mul.lo.s32,
        #   cvt.u64.u32, cvt.u32.u64, shl.b64, add.s64, add.s32; cvta.to.global.u64 is
        #   also emitted here inside the row loop, not only in the entry block);
        #   extent: per row iteration

        # ---- mode trivial early-out (:1243-1307) --------------------------
        if MODE == "Basic":
            cmp_ge(p_all, k, length)
            # instruction_selection: setp.ge.u32; extent: scalar
            if p_all:                                          # k >= length: identity output
                i = tx
                while i < length:
                    cmp_lt(p_in, i, k)
                    # instruction_selection: setp.lt.u32; extent: scalar per iteration
                    if p_in:
                        copy_r2g_scalar(i, row_output + i)
                        # instruction_selection: st.global.b32; extent: one scalar store
                        copy_g2r_scalar(input + row_idx * LENGTH + i, v_raw)
                        # instruction_selection: ld.global.b32 (f32) | ld.global.b16 (f16, bf16); extent: one scalar load
                        copy_r2g_scalar(v_raw, output_values + row_idx * K + i)
                        # instruction_selection: st.global.b32 (f32) | st.global.b16 (f16, bf16); extent: one scalar store
                    i = i + BLOCK
                continue                                       # next row
        elif MODE == "PageTableTransform":
            if row_to_batch is not null:
                copy_g2r_scalar(row_to_batch + row_idx, batch_idx)
                # instruction_selection: ld.global.b32; extent: one scalar load
            else:
                move(batch_idx, row_idx)
            src_page_entry = aux_data + batch_idx * aux_stride
            # instruction_selection: cvt.u64.u32 + mul.lo.s64 + add.s64 (:1271 -- aux_stride is a
            #   runtime int64_t); extent: address arithmetic
            cmp_le(p_short, length, K)
            # instruction_selection: setp.le.u32; extent: scalar
            if p_short:
                i = tx
                while i < K:
                    cmp_lt(p_in, i, length)
                    # instruction_selection: setp.lt.u32; extent: scalar per iteration
                    if p_in:
                        copy_g2r_scalar(src_page_entry + page_start + i, page_id)
                        # instruction_selection: ld.global.b32; extent: one scalar load
                    else:
                        move(page_id, -1)
                        # instruction_selection: selp.b32; extent: scalar
                    copy_r2g_scalar(page_id, row_output + i)
                    # instruction_selection: st.global.b32; extent: one scalar store
                    i = i + BLOCK
                continue
        else:  # RaggedTransform
            copy_g2r_scalar(aux_data + row_idx, offset)
            # instruction_selection: ld.global.b32; extent: one scalar load
            cmp_le(p_short, length, K)
            # instruction_selection: setp.le.u32; extent: scalar
            if p_short:
                i = tx
                while i < K:
                    cmp_lt(p_in, i, length)
                    # instruction_selection: setp.lt.u32; extent: scalar per iteration
                    add(val, i, offset)
                    # instruction_selection: add.s32; extent: scalar
                    select(out_id, p_in, val, -1)
                    # instruction_selection: selp.b32; extent: scalar
                    copy_r2g_scalar(out_id, row_output + i)
                    # instruction_selection: st.global.b32; extent: one scalar store
                    i = i + BLOCK
                continue

        # =====================================================================
        # Stage 1: stage the row into shared memory as monotone keys
        # LoadToSharedOrdered (:605-623); chunk_start == 0, actual_chunk_size == length
        # =====================================================================
        aligned = length // VEC_SIZE * VEC_SIZE                # aligned_size (:608)
        # instruction_selection: and.b32 with ~(VEC_SIZE-1) (VEC_SIZE is a power of two);
        #   constant-folded in Basic mode where `length == stride` is static, a real
        #   instruction in PageTable/Ragged where `length` came from lengths[row_idx]

        i = tx * VEC_SIZE
        while i < aligned:                                     # #pragma unroll 2
            # instruction_selection: none; extent: loop control (setp.lt.u32/bra), unroll hint 2
            bits = reg_tile("b32" if DTYPE == "f32" else "b16", [VEC_SIZE])
            copy_g2r_v(row_input + i, bits)
            # instruction_selection: ld.global.v4.b32 (VEC_SIZE*sizeof(DType) == 16) |
            #   ld.global.v2.b32 | ld.global.b32 | ld.global.b16 for the narrower vec widths;
            #   extent: one vector load per iteration
            for j in static_range(VEC_SIZE):
                to_ordered(key[j], bits[j])
                # instruction_selection: setp.gt.s32 + selp.b32(0x80000000, -1) + xor.b32 (f32) |
                #   setp.gt.s16 + selp.b16(0x8000, -1) + xor.b16 (f16, bf16);
                #   extent: three scalar instructions per element, VEC_SIZE per iteration
                copy_r2s(key[j], s_ordered[i + j])
                # instruction_selection: st.shared.b32 (f32) | st.shared.b16 (f16, bf16), which the
                #   compiler coalesces to st.shared.v4.b32 / st.shared.v2.b32 when the VEC_SIZE
                #   group is contiguous; extent: one shared store per element
            i = i + BLOCK * VEC_SIZE

        i = aligned + tx
        while i < length:                                      # scalar tail (:619-621)
            # instruction_selection: none; extent: loop control.  Statically dead only in
            #   Basic mode when LENGTH % VEC_SIZE == 0; live in PageTable/Ragged, where the
            #   per-row runtime `length` can leave a remainder (PTX: `.loc 2 608 and.b32 ...,-4`
            #   and a live `.loc 2 619/620` ld.global.b32 + st.shared.b32 pair in the PageTable
            #   body; x7 in the Basic body, where `stride` is also a runtime kernel argument)
            copy_g2r_scalar(row_input + i, bits_s)
            # instruction_selection: ld.global.b32 (f32) | ld.global.b16 (f16, bf16); extent: one scalar load
            to_ordered(key_s, bits_s)
            # instruction_selection: setp.gt.s32/s16 + selp.b32/b16 + xor.b32/b16; extent: three scalar
            copy_r2s(key_s, s_ordered[i])
            # instruction_selection: st.shared.b32 | st.shared.b16; extent: one shared store
            i = i + BLOCK

        barrier()
        # instruction_selection: bar.sync 0; extent: one CTA-wide publication of s_ordered,
        #   closing LoadToSharedOrdered (:622)

        # =====================================================================
        # Stage 2: NUM_ROUNDS of 8-bit radix select
        # RadixSelectFromSharedMemory (:669-777).  Note the scalar init below runs
        # *after* the load barrier, and is closed by its own barrier (:677).
        # =====================================================================
        if tx == 0:                                            # (:669-676)
            move(s_scalars[0], 0)                              # prefix_cache
            move(s_scalars[1], k)                              # remaining_k_cache
            # instruction_selection: st.shared.v2.b32 [smem+2048], {0, k} (the two adjacent
            #   u32 scalars are written as one 64-bit store); extent: one paired shared store
            move(s_scalars[4], 0)                              # shared_output_counter
            # instruction_selection: st.shared.b32; extent: one scalar store
        barrier()
        # instruction_selection: bar.sync 0; extent: CTA-wide, scalar caches published (:677)

        rnd = 0
        while rnd < NUM_ROUNDS:                                # rolled loop (:690)
            # instruction_selection: add.s32 + setp.ne.b32 back-edge; extent: loop control.
            #   NVCC keeps this loop rolled -- every round-body .loc appears exactly once
            #   and the whole kernel has 28 bar.sync, which only the rolled form allows.
            shift = OBITS - (rnd + 1) * 8                      # (:692), runtime
            # instruction_selection: shl.b32 + sub.s32; extent: two scalar per round
            mask  = 0 if rnd == 0 else (~0 << (OBITS - rnd * 8))  # (:714-717), runtime
            # instruction_selection: sub.s32 + shl.b32 + selp.b32, hoisted by NVCC into the
            #   round-body preheader (`.loc 2 0`); extent: three scalar per round.  With the
            #   loop rolled the mask is a runtime value in every round, including rnd == 0.
            copy_s2r((s_scalars[0], s_scalars[1]), (prefix, remaining_k))
            # instruction_selection: ld.shared.b64 [smem+2048] + mov.b64 unpack (the two adjacent
            #   u32 scalars are read as one 64-bit shared load); extent: one 64-bit shared load per round

            b = tx
            while b < RADIX:                                   # clear the histogram (:704-706)
                move(s_hist[b], 0)
                # instruction_selection: st.shared.b32 (no vector coalescing: the whole kernel
                #   emits one st.shared.v4.b32 and it belongs to the key staging at :615);
                #   extent: one shared store per iteration
                b = b + BLOCK
            barrier()
            # instruction_selection: bar.sync 0; extent: CTA-wide, histogram cleared

            i = tx
            while i < length:                                  # #pragma unroll 2 (:709-722)
                # instruction_selection: none; extent: loop control (setp.lt.u32/bra), unroll hint 2
                copy_s2r(s_ordered[i], key)
                # instruction_selection: ld.shared.b32 (f32) | ld.shared.b16 (f16, bf16); extent: one shared load
                and_(masked, key, mask)
                # instruction_selection: and.b32 | and.b16 (:718) applied on every round -- the
                #   rolled loop leaves `mask` a runtime `selp.b32` result, so there is no
                #   rnd == 0 fold; extent: scalar
                cmp_eq(p_match, masked, prefix)
                # instruction_selection: setp.eq.b32 | setp.eq.b16; extent: scalar
                if p_match:
                    shr(bucket, key, shift)
                    and_(bucket, bucket, 0xFF)
                    # instruction_selection: shr.u32 + and.b32; extent: two scalar
                    atomic_add_shared(_, s_hist[bucket], 1)
                    # instruction_selection: atom.shared.add.u32; extent: one per matching element
                i = i + BLOCK
            barrier()
            # instruction_selection: bar.sync 0; extent: CTA-wide, histogram complete

            b = tx
            while b < RADIX:                                   # single-CTA copy (:743-745)
                copy_s2r(s_hist[b], h)
                # instruction_selection: ld.shared.b32; extent: one shared load per iteration
                copy_r2s(h, s_suffix[b])
                # instruction_selection: st.shared.b32; extent: one shared store per iteration
                b = b + BLOCK
            barrier()
            # instruction_selection: bar.sync 0; extent: CTA-wide, s_suffix seeded

            # RadixSuffixSum (:389-406), called at :750: 8 doubling steps, two barriers each
            for stride in static_range([1, 2, 4, 8, 16, 32, 64, 128]):
                cmp_lt(p_lane, tx, RADIX)
                # instruction_selection: setp.lt.u32; extent: scalar
                move(val, 0)
                if p_lane:
                    copy_s2r(s_suffix[tx], val)
                    # instruction_selection: ld.shared.b32; extent: one shared load
                    cmp_lt(p_pair, tx + stride, RADIX)
                    # instruction_selection: setp.lt.u32; extent: scalar
                    if p_pair:
                        copy_s2r(s_suffix[tx + stride], rhs)
                        # instruction_selection: ld.shared.b32; extent: one shared load
                        add(val, val, rhs)
                        # instruction_selection: add.s32; extent: scalar
                barrier()
                # instruction_selection: bar.sync 0; extent: CTA-wide, read phase closed
                if p_lane:
                    copy_r2s(val, s_suffix[tx])
                    # instruction_selection: st.shared.b32; extent: one shared store
                barrier()
                # instruction_selection: bar.sync 0; extent: CTA-wide, write phase closed

            # Threshold bucket (:752-766)
            if tx == 0:
                move(s_scalars[2], 0)
                move(s_scalars[3], remaining_k)
                # instruction_selection: st.shared.v2.b32 [smem+2056] (:755); extent: one paired shared store
            barrier()
            # instruction_selection: bar.sync 0; extent: CTA-wide
            cmp_lt(p_lane, tx, RADIX)
            # instruction_selection: setp.lt.u32; extent: scalar
            if p_lane:
                copy_s2r(s_suffix[tx], count_ge)
                # instruction_selection: ld.shared.b32; extent: one shared load
                move(count_gt, 0)
                cmp_lt(p_next, tx + 1, RADIX)
                # instruction_selection: setp.lt.u32; extent: scalar
                if p_next:
                    copy_s2r(s_suffix[tx + 1], count_gt)
                    # instruction_selection: ld.shared.b32; extent: one shared load
                cmp_ge(p_a, count_ge, remaining_k)
                cmp_lt(p_b, count_gt, remaining_k)
                # instruction_selection: setp.ge.u32 + setp.lt.u32; extent: two scalar
                if p_a and p_b:
                    move(s_scalars[2], tx)
                    sub(rk, remaining_k, count_gt)
                    # instruction_selection: sub.s32; extent: scalar
                    move(s_scalars[3], rk)
                    # instruction_selection: st.shared.v2.b32 [smem+2056] (:764); extent: one paired
                    #   shared store by the single winner lane
            barrier()
            # instruction_selection: bar.sync 0; extent: CTA-wide, bucket published

            if tx == 0:                                        # (:770-772)
                copy_s2r((s_scalars[2], s_scalars[3]), (bucket, rk))
                # instruction_selection: ld.shared.v2.b32 [smem+2056] (:771); extent: one paired shared load
                shl(hi, bucket, shift)
                or_(new_prefix, prefix, hi)
                # instruction_selection: shl.b32 + or.b32; extent: two scalar
                move(s_scalars[0], new_prefix)
                move(s_scalars[1], rk)
                # instruction_selection: st.shared.v2.b32 [smem+2048] (:772); extent: one paired shared store
            barrier()
            # instruction_selection: bar.sync 0; extent: CTA-wide, round state published (:774)
            rnd = rnd + 1

        copy_s2r(s_scalars[0], pivot)                          # ordered_pivot (:777)
        # instruction_selection: ld.shared.b32 (f32) | ld.shared.b16 + cvt.u16.u32 (f16, bf16);
        #   extent: one shared load

        # =====================================================================
        # Stage 3: row-wide counts of key > pivot (and == pivot when needed)
        # (:776-830).  TRACK_EQ_COUNT == DETERMINISTIC.
        # =====================================================================
        if tx == 0:                                            # (:782-786)
            if not DETERMINISTIC:
                move(s_suffix[0], 0)
                # instruction_selection: st.shared.b32 (:783); extent: one scalar store
            else:
                move(s_suffix[0], 0)
                move(s_suffix[1], 0)
                # instruction_selection: st.shared.v2.b32 [smem+1024] (:785); extent: one paired shared store
        barrier()
        # instruction_selection: bar.sync 0; extent: CTA-wide, accumulators cleared

        move(my_gt, 0)
        move(my_eq, 0)
        i = tx
        while i < length:                                      # #pragma unroll 2 (:789-801)
            # instruction_selection: none; extent: loop control, unroll hint 2
            copy_s2r(s_ordered[i], key)
            # instruction_selection: ld.shared.b32 | ld.shared.b16; extent: one shared load
            cmp_gt(p_gt, key, pivot)
            # instruction_selection: setp.gt.u32 | setp.gt.u16; extent: scalar
            add(my_gt, my_gt, p_gt)
            # instruction_selection: selp.b32 + add.s32; extent: two scalar
            if DETERMINISTIC:
                cmp_eq(p_eq, key, pivot)
                # instruction_selection: setp.eq.b32 | setp.eq.b16; extent: scalar
                add(my_eq, my_eq, p_eq)
                # instruction_selection: selp.b32 + add.s32; extent: two scalar
            i = i + BLOCK

        for delta in static_range([16, 8, 4, 2, 1]):           # warp reduction (:804-810)
            shfl_down(peer, my_gt, delta)
            # instruction_selection: shfl.sync.down.b32 %r|%p, %r, delta, 31, -1; extent: one per step
            add(my_gt, my_gt, peer)
            # instruction_selection: add.s32; extent: scalar
            if DETERMINISTIC:
                shfl_down(peer_eq, my_eq, delta)
                # instruction_selection: shfl.sync.down.b32; extent: one per step
                add(my_eq, my_eq, peer_eq)
                # instruction_selection: add.s32; extent: scalar

        lane = tx % 32
        # instruction_selection: and.b32 (tx & 31); extent: scalar
        if lane == 0 and my_gt > 0:
            atomic_add_shared(_, s_suffix[0], my_gt)
            # instruction_selection: atom.shared.add.u32; extent: one per warp leader
        if DETERMINISTIC and lane == 0 and my_eq > 0:
            atomic_add_shared(_, s_suffix[1], my_eq)
            # instruction_selection: atom.shared.add.u32; extent: one per warp leader
        barrier()
        # instruction_selection: bar.sync 0; extent: CTA-wide, counts published
        copy_s2r(s_suffix[0], gt_count)
        # instruction_selection: ld.shared.b32; extent: one scalar load
        if DETERMINISTIC:
            copy_s2r(s_suffix[1], eq_count)
            # instruction_selection: ld.shared.b32; extent: one scalar load

        # =====================================================================
        # Stage 3b: epilogue-scope aux recompute, per row, BEFORE the collect.
        # The source opens the mode epilogue scope at :1343 / :1372 and computes
        # these once per row; only then does it call collect_indices(...).  They are
        # second, distinct reads from the ones the trivial branch made at :1270-1271
        # and :1290, and both pairs survive into the PTX.
        # =====================================================================
        if MODE == "PageTableTransform":                       # (:1344-1345)
            if row_to_batch is not null:
                copy_g2r_scalar(row_to_batch + row_idx, batch_idx)
                # instruction_selection: ld.global.b32 (:1344); extent: one scalar load per row
            else:
                move(batch_idx, row_idx)
            src_page_entry = aux_data + batch_idx * aux_stride
            # instruction_selection: cvt.u64.u32 + mul.lo.s64 + add.s64 (:1345 -- aux_stride is a
            #   runtime int64_t); extent: address arithmetic, once per row
        elif MODE == "RaggedTransform":                        # (:1373)
            copy_g2r_scalar(aux_data + row_idx, offset)
            # instruction_selection: ld.global.b32 (:1373); extent: one scalar load per row

        # =====================================================================
        # Stage 4a: non-deterministic collect (:901-968), DETERMINISTIC == False
        # s_hist[0] = local_offset_gt, s_hist[1] = global_base_gt
        # =====================================================================
        if not DETERMINISTIC:
            if tx == 0:
                move(s_hist[0], 0)
                # instruction_selection: st.shared.b32; extent: one scalar store
                if gt_count > 0:
                    atomic_add_shared(base, s_scalars[4], gt_count)
                    # instruction_selection: atom.shared.add.u32; extent: one per CTA
                    move(s_hist[1], base)
                    # instruction_selection: st.shared.b32; extent: one scalar store
            barrier()
            # instruction_selection: bar.sync 0; extent: CTA-wide, collect base published (:923)

            i = tx
            while i < length:                                  # pass 1: key > pivot (:927-937)
                # instruction_selection: none; extent: loop control, unroll hint 2
                copy_s2r(s_ordered[i], key)
                # instruction_selection: ld.shared.b32 | ld.shared.b16; extent: one shared load
                cmp_gt(p_gt, key, pivot)
                # instruction_selection: setp.gt.u32 | setp.gt.u16; extent: scalar
                if p_gt:
                    atomic_add_shared(local_pos, s_hist[0], 1)
                    # instruction_selection: atom.shared.add.u32; extent: one per emitted element
                    copy_s2r(s_hist[1], global_base_gt)
                    # instruction_selection: ld.shared.b32 [smem+4] (:932); extent: one shared load per
                    #   emitted element -- the source re-reads global_base_gt inside the loop rather
                    #   than hoisting it, and the PTX keeps one per unrolled body
                    add(pos, global_base_gt, local_pos)
                    # instruction_selection: add.s32; extent: scalar
                    EMIT(i, key, pos)                          # mode epilogue below
                i = i + BLOCK
            barrier()
            # instruction_selection: bar.sync 0; extent: CTA-wide, gt pass closed before the eq pass (:944)

            i = tx
            while i < length:                                  # pass 2: key == pivot (:950-965)
                # instruction_selection: none; extent: loop control, unroll hint 2
                copy_s2r(s_ordered[i], key)
                # instruction_selection: ld.shared.b32 | ld.shared.b16; extent: one shared load
                cmp_eq(p_eq, key, pivot)
                # instruction_selection: setp.eq.b32 | setp.eq.b16; extent: scalar
                if p_eq:
                    atomic_add_shared(pos, s_scalars[4], 1)
                    # instruction_selection: atom.shared.add.u32; extent: one per candidate
                    cmp_lt(p_room, pos, k)
                    # instruction_selection: setp.lt.s32; extent: scalar
                    if p_room:
                        EMIT(i, pivot, pos)
                i = i + BLOCK

        # =====================================================================
        # Stage 4b: deterministic collect (:1017-1136), DETERMINISTIC == True
        # Single-CTA degenerates to a block-local scheme: gt entries occupy
        # [0, gt_count) and eq entries fill [gt_count, k).
        # =====================================================================
        else:
            if tx == 0:
                move(s_hist[0], 0)                             # gt prefix  == 0
                move(s_hist[1], 0)                             # eq prefix  == 0
                move(s_hist[2], gt_count)                      # row total gt
                sub(need, k, gt_count)
                max_(need, need, 0)
                # instruction_selection: sub.s32 + max.u32 (the source writes the ternary
                #   `(k > gt) ? (k - gt) : 0`); extent: two scalar
                move(s_hist[3], need)                          # eq needed
                move(s_hist[4], 0)
                # instruction_selection: st.shared.v4.b32 [smem] over s_hist[0..3] (:1041) +
                #   st.shared.b32 for s_hist[4] (:1042); extent: one 128-bit and one scalar store
            barrier()
            # instruction_selection: bar.sync 0; extent: CTA-wide, collect plan published
            copy_s2r((s_hist[0], s_hist[1], s_hist[2], s_hist[3]),
                     (gt_output_base, _eq_prefix, eq_output_base, eq_emit_limit))
            # instruction_selection: ld.shared.v4.b32 {..},[smem] (:1046 + :1085); extent: one
            #   128-bit shared load covering local_histogram[0..3] -- it carries
            #   s_cta_local_gt_prefix as well as the eq plan
            sub(gt_emit_limit, k, gt_output_base)
            max_(gt_emit_limit, gt_emit_limit, 0)
            # instruction_selection: max.u32 + sub.s32 (:1086-1087, the source's
            #   `(k > base) ? (k - base) : 0`); extent: two scalar.  base is 0 in this
            #   specialization but the source still computes it, and so does the PTX.

            if eq_emit_limit == 0:
                # DeterministicThreadStridedCollect (:256-286): gt-only fast path
                move(my_sel, 0)
                i = tx
                while i < length:
                    copy_s2r(s_ordered[i], key)
                    # instruction_selection: ld.shared.b32 | ld.shared.b16; extent: one shared load
                    cmp_gt(p_gt, key, pivot)
                    # instruction_selection: setp.gt.u32 | setp.gt.u16; extent: scalar
                    add(my_sel, my_sel, p_gt)
                    # instruction_selection: selp.b32 + add.s32; extent: two scalar
                    i = i + BLOCK
                block_exclusive_sum(my_prefix, my_sel)
                # instruction_selection: cub BlockScan RAKING_MEMOIZE ExclusiveSum: 5
                #   shfl.sync.up.b32 (warp_scan_shfl.cuh:179) plus st.shared.v2.b32 /
                #   ld.shared.v2.b32 raking traffic through s_scan_temp and bar.sync 0
                #   (block_scan_raking.cuh:165/325/341);
                #   extent: one CTA-wide exclusive scan
                cmp_gt(p_any, my_sel, 0)
                cmp_lt(p_room, my_prefix, gt_emit_limit)
                # instruction_selection: setp.gt.u32 + setp.lt.u32; extent: two scalar
                if p_any and p_room:
                    move(emit_pos, my_prefix)
                    add(emit_end, my_prefix, my_sel)
                    min_(emit_end, emit_end, gt_emit_limit)
                    # instruction_selection: add.s32 + min.u32; extent: two scalar
                    i = tx
                    while i < length:
                        copy_s2r(s_ordered[i], key)
                        # instruction_selection: ld.shared.b32 | ld.shared.b16; extent: one shared load
                        cmp_gt(p_gt, key, pivot)
                        # instruction_selection: setp.gt.u32 | setp.gt.u16; extent: scalar
                        if p_gt:
                            EMIT(i, key, gt_output_base + emit_pos)
                            add(emit_pos, emit_pos, 1)
                            # instruction_selection: add.s32; extent: scalar
                            cmp_eq(p_full, emit_pos, emit_end)
                            # instruction_selection: setp.eq.b32; extent: scalar
                            if p_full: break
                            # instruction_selection: bra; extent: early exit
                        i = i + BLOCK
                barrier()
                # instruction_selection: bar.sync 0; extent: CTA-wide, collect closed
            else:
                # Paired gt/eq scan (:1113-1135)
                move(my_gt_sel, 0)
                move(my_eq_sel, 0)
                i = tx
                while i < length:
                    copy_s2r(s_ordered[i], key)
                    # instruction_selection: ld.shared.b32 | ld.shared.b16; extent: one shared load
                    cmp_gt(p_gt, key, pivot)
                    cmp_eq(p_eq, key, pivot)
                    # instruction_selection: setp.gt.u32/u16 + setp.eq.b32/b16; extent: two scalar
                    add(my_gt_sel, my_gt_sel, p_gt)
                    add(my_eq_sel, my_eq_sel, p_eq)
                    # instruction_selection: selp.b32 x2 + add.s32 x2; extent: four scalar
                    i = i + BLOCK
                block_exclusive_sum_pair((gt_prefix, eq_prefix), (my_gt_sel, my_eq_sel))
                # instruction_selection: cub BlockScan RAKING_MEMOIZE ExclusiveScan over the
                #   (gt, eq) pair: 12 shfl.sync.up.b32 (util_ptx.cuh:98) plus st.shared.v2.b32 /
                #   ld.shared.v2.b32 raking traffic through s_scan_temp and bar.sync 0;
                #   extent: one CTA-wide exclusive pair scan.  Both scan variants are compiled
                #   into every DETERMINISTIC entry, so the two sites together account for the
                #   17 shfl.sync.up.b32 in the count table.
                move(pos_gt, gt_prefix)
                move(pos_eq, eq_prefix)
                i = tx
                while i < length:
                    copy_s2r(s_ordered[i], key)
                    # instruction_selection: ld.shared.b32 | ld.shared.b16; extent: one shared load
                    cmp_gt(p_gt, key, pivot)
                    cmp_lt(p_gt_room, pos_gt, gt_emit_limit)
                    # instruction_selection: setp.gt.u32/u16 + setp.lt.u32; extent: two scalar
                    if p_gt and p_gt_room:
                        EMIT(i, key, gt_output_base + pos_gt)
                        add(pos_gt, pos_gt, 1)
                        # instruction_selection: add.s32; extent: scalar
                    else:
                        cmp_eq(p_eq, key, pivot)
                        cmp_lt(p_eq_room, pos_eq, eq_emit_limit)
                        # instruction_selection: setp.eq.b32/b16 + setp.lt.u32; extent: two scalar
                        if p_eq and p_eq_room:
                            EMIT(i, key, eq_output_base + pos_eq)
                            add(pos_eq, pos_eq, 1)
                            # instruction_selection: add.s32; extent: scalar
                    i = i + BLOCK
                barrier()
                # instruction_selection: bar.sync 0; extent: CTA-wide, collect closed

        # =====================================================================
        # Stage 5: PageTable second pass (:1352-1358); Basic and Ragged emit
        # their final value inside EMIT.  `src_page_entry` was computed in Stage 3b.
        # =====================================================================
        if MODE == "PageTableTransform":
            barrier()
            # instruction_selection: bar.sync 0 (:1353); extent: CTA-wide, raw indices published
            i = tx
            while i < k:
                copy_g2r_scalar(row_output + i, idx)
                # instruction_selection: ld.global.b32; extent: one scalar load
                copy_g2r_scalar(src_page_entry + page_start + idx, page_id)
                # instruction_selection: ld.global.b32; extent: one scalar gather load
                copy_r2g_scalar(page_id, row_output + i)
                # instruction_selection: st.global.b32; extent: one scalar store
                i = i + BLOCK

# ===========================================================================
# EMIT: the mode epilogue the collect passes call per selected element
# (:1336-1372).  `pos` is the output slot, `i` the row-local index.
# ===========================================================================

def EMIT_Basic(i, key, pos):
    copy_r2g_scalar(i, row_output + pos)
    # instruction_selection: st.global.b32; extent: one scalar store
    from_ordered(bits, key)
    # instruction_selection: setp.gt.s32 + selp.b32(-1, 0x80000000) + xor.b32 (f32) |
    #   setp.gt.s16 + selp.b16(-1, 0x8000) + xor.b16 (f16, bf16); extent: three scalar
    copy_r2g_scalar(bits, output_values + row_idx * K + pos)
    # instruction_selection: st.global.b32 (f32) | st.global.b16 (f16, bf16); extent: one scalar store

def EMIT_PageTableTransform(i, key, pos):
    copy_r2g_scalar(i, row_output + pos)                # raw index; gathered in stage 5
    # instruction_selection: st.global.b32; extent: one scalar store

def EMIT_RaggedTransform(i, key, pos):   # `offset` was loaded once per row in Stage 3b
    add(val, i, offset)
    # instruction_selection: add.s32; extent: scalar
    copy_r2g_scalar(val, row_output + pos)
    # instruction_selection: st.global.b32; extent: one scalar store
```

## Static specialization boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| `DTYPE` | static per config | selects `OrderedType` (u32/u16), `NUM_ROUNDS` (4/2), the 32-bit vs 16-bit `to_ordered`/compare/shared-access opcode family, and `16 / sizeof(DType)` in the vec dispatch |
| `MODE` | static per config | selects the per-row header, the trivial branch, `EMIT`, and whether stage 5 exists |
| `DETERMINISTIC` | static per config | selects the collect path, whether eq counts are tracked, whether the 8480-byte static `scan_temp_storage` exists, and the launcher's matching 8512-byte smem headroom (hence `max_chunk` and the single-CTA domain) |
| `VEC_SIZE` | static per config | selects the load width; the scalar tail loop is statically dead only in Basic mode when `LENGTH % VEC_SIZE == 0`, and stays live in PageTable/Ragged |
| `SINGLE_CTA = true` | static (launcher decision) | removes `RadixRowState`, `det_scratch`, every group barrier, the triple-buffer global histogram and the trailing state reset |
| `NUM_ROWS`, `LENGTH`, `K` | static per config | `CHUNK == LENGTH`, `SMEM_BYTES`, `NUM_GROUPS`, `total_iterations` constant-fold. `aligned_size` folds only in Basic mode; in PageTable/Ragged it is derived per row from the runtime `lengths[row_idx]` |
| `row_starts` / `page_table_row_starts` / `row_to_batch` non-null | static per config (runtime pointer in the source) | selects the header loads; a non-null `row_starts` also forces `VEC_SIZE == 1` in the non-Basic launchers |
| `top_k_arr` | always null | Basic mode's per-row k branch is dead; `k == K` for every row |
| dynamic smem size | static per config | requested through the `tirx.use_dyn_shared_memory` launch tag, mirroring `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, SMEM_BYTES)` |
| unroll hints | static | the load loop, the histogram loop, the count loop and both non-deterministic collect passes carry the source's `#pragma unroll 2`, which NVCC turns into three emitted bodies each; the 8 suffix-sum doubling steps, the VEC_SIZE inner loop and the 5 shuffle steps are fully unrolled; the `NUM_ROUNDS` round loop stays **rolled** |

## TIRx module and benchmark contract

- `KERNEL_META = {"name": "radix_topk_single_cta", "category": "flashinfer",
  "runtime_cuda_archs": ["sm_100a"]}`.
- The kernel is expressed entirely in plain TIRx: `T.SMEMPool` arena with the
  source's byte offsets, explicit `while` block-strided loops, register
  buffers, and native `T.ptx.*` forms for the key operations
  (`bar.sync`, `atom.shared.add.u32`, `shfl.sync.down.b32`, the vector global
  loads, the shared loads/stores). No `Tx` tile primitives and no
  `T.cuda.func_call` anywhere in the pre-dispatch IR.
- `get_kernel(dtype, mode, num_rows, length, k, deterministic, row_starts,
  page_table_row_starts, row_to_batch)` returns the specialized primfunc;
  `prepare_data`, `run_test`, `prepare_bench`, `run_gpu`, `run_bench` follow the
  repository contract.
- The timed implementation is named `tirx`; the reference is the FlashInfer
  source kernel itself, launched through the raw `topk` FFI module with
  preallocated outputs and `FLASHINFER_TOPK_ALGO=multi_cta` pinning
  `TopKDispatch` to this kernel. Allocation, module build and validation stay
  outside timing.
- Correctness compares against that same source launch: exact positional
  equality for `DETERMINISTIC` configs, and set equality of the selected
  indices/values otherwise, because both sides leave the non-deterministic
  collect order unspecified.

## Instruction selection is a lowering consequence

The sketch never requests a hardware instruction beyond the shared atomics, the
warp shuffle, and the CTA barrier. The families below follow from storage
direction, shape, dtype and schedule. PTX names are taken from a fresh
line-info export of the exact source instantiations
(`.porting/radix_topk_single_cta/ptx/instantiate.cu`, explicit instantiation of
`RadixTopKKernel_Unified<1024, VEC, true, DET, MODE, DType, int32_t>`) built
with the production JIT flags
`nvcc -ptx -lineinfo -std=c++17 -use_fast_math -O3 -DNDEBUG -arch=compute_100a`
plus the FlashInfer `-D` set; they are audit evidence, not operands.

| Primitive/schedule pattern | PTX family (fresh SM100a export) |
| --- | --- |
| vectorized row load (`vec_t::cast_load`, VEC_SIZE*sizeof(DType) == 16) | `ld.global.v4.b32` |
| scalar tail / header / gather loads | `ld.global.b32`, `ld.global.b16` (16-bit dtypes) |
| `to_ordered` f32 | `setp.gt.s32` + `selp.b32 -2147483648, -1` + `xor.b32` |
| `to_ordered` f16/bf16 | `setp.gt.s16` + `selp.b16 -32768, -1` + `xor.b16` |
| `from_ordered` f32 / f16 / bf16 | the same triple with the `selp` operands swapped (`-1, -2147483648`) |
| stage into `s_ordered` | `st.shared.b32` / `st.shared.b16`, coalesced to `st.shared.v4.b32` / `st.shared.v2.b32` for contiguous VEC groups |
| histogram clear / seed / scalar publication | `st.shared.b32`, coalesced to `st.shared.v4.b32` / `st.shared.v2.b32` |
| histogram bump, gt/eq accumulate, collect counters | `atom.shared.add.u32` |
| shared reads of keys, histogram, suffix sum, scalars | `ld.shared.b32`, `ld.shared.b16`, `ld.shared.b64` (the paired `prefix`/`remaining_k` read), `ld.shared.v2.b32`, `ld.shared.v4.b32` (the deterministic collect plan) |
| every publication point | `bar.sync 0` |
| gt/eq warp reduction | `shfl.sync.down.b32 %r|%p, %r, {16,8,4,2,1}, 31, -1` |
| deterministic block scan (cub `BlockScan`, `BLOCK_SCAN_RAKING_MEMOIZE`) | `shfl.sync.up.b32` + `st.shared.v2.b32` / `ld.shared.v2.b32` raking + `bar.sync 0` |
| predicate arithmetic on counts | `setp.{gt,ge,lt,le,eq,ne}.{u32,s32,u16,s16,b32,b16}` + `selp.b32` + `add.s32` |
| bucket extraction | `shr.u32` + `and.b32` (no `bfe` in the export) |
| index/value stores | `st.global.b32`, `st.global.b16` (16-bit values) |
| address arithmetic | `mul.wide.u32`, `mul.wide.s32`, `mul.lo.s32`, `cvt.u64.u32`, `cvt.u32.u64`, `shl.b64`, `add.s64`, `add.s32`, `shl.b32`, `cvta.to.global.u64` (emitted inside the row loop too, not only in the entry block). `mul.lo.s64` appears only in PageTableTransform, twice per body (`:1271`, `:1345`), for the runtime `int64_t aux_stride` multiply. |

Static PTX opcode counts per exported instantiation (whole kernel body; the
round loop stays rolled, so each round body appears once, and every `#pragma unroll 2`
loop emits three bodies):

| Family | f32 vec4 Basic | f32 vec4 Basic det | f16 vec8 Basic | f32 vec1 Basic | f32 vec4 PageTable | f32 vec4 PageTable det | f32 vec4 Ragged | f32 vec4 Ragged det |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ld.global.v4.b32` | 3 | 3 | 3 | 0 | 3 | 3 | 3 | 3 |
| `ld.global.b32` | 9 | 9 | 1 | 12 | 27 | 27 | 5 | 5 |
| `ld.global.b16` | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 |
| `st.global.b32` | 14 | 32 | 7 | 14 | 20 | 17 | 21 | 18 |
| `st.global.b16` | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 |
| `ld.shared.b32` | 36 | 98 | 23 | 36 | 36 | 92 | 36 | 92 |
| `ld.shared.b16` | 0 | 0 | 13 | 0 | 0 | 0 | 0 | 0 |
| `ld.shared.b64` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| `ld.shared.v2.b32` | 1 | 34 | 1 | 1 | 1 | 34 | 1 | 34 |
| `ld.shared.v4.b32` | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 1 |
| `st.shared.b32` | 29 | 60 | 14 | 24 | 23 | 54 | 23 | 54 |
| `st.shared.b16` | 0 | 0 | 23 | 0 | 0 | 0 | 0 | 0 |
| `st.shared.v2.b32` | 4 | 38 | 4 | 4 | 4 | 38 | 4 | 38 |
| `st.shared.v4.b32` | 1 | 2 | 1 | 0 | 1 | 2 | 1 | 2 |
| `atom.shared.add.u32` | 11 | 5 | 11 | 11 | 11 | 5 | 11 | 5 |
| `bar.sync` | 28 | 33 | 28 | 28 | 29 | 34 | 28 | 33 |
| `shfl.sync.down.b32` | 5 | 10 | 5 | 5 | 5 | 10 | 5 | 10 |
| `shfl.sync.up.b32` | 0 | 17 | 0 | 0 | 0 | 17 | 0 | 17 |
| `xor.b32` | 23 | 31 | 0 | 14 | 13 | 13 | 13 | 13 |
| `xor.b16` | 0 | 0 | 35 | 0 | 0 | 0 | 0 | 0 |
| `mul.wide.u32` | 12 | 14 | 12 | 12 | 10 | 13 | 9 | 11 |
| `mul.wide.s32` | 6 | 15 | 12 | 6 | 6 | 3 | 6 | 3 |
| `mul.lo.s32` | 3 | 8 | 3 | 3 | 3 | 7 | 3 | 5 |
| `cvta.to.global.u64` | 5 | 8 | 6 | 5 | 7 | 11 | 5 | 7 |
| `max.u32` | 4 | 3 | 4 | 4 | 0 | 3 | 0 | 3 |
| `min.u32` | 1 | 2 | 1 | 1 | 1 | 2 | 1 | 2 |
| `mov.b64` | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 3 |

The `atom.shared.add.u32` count drops from 11 to 5 with `DETERMINISTIC` because
the deterministic collect replaces both atomic-driven passes with the block
scans, which contribute the 17 `shfl.sync.up.b32` (12 from the pair scan via
`util_ptx.cuh:98`, 5 from the scalar gt-only scan via `warp_scan_shfl.cuh:179`
-- both variants are compiled into the same kernel) and the paired
`st.shared.v2.b32`/`ld.shared.v2.b32` raking traffic instead. The extra
`bar.sync` in PageTable mode is the barrier before the in-place page gather.
The single `ld.shared.b64` in every entry is the paired `prefix_cache` /
`remaining_k_cache` read at the head of each round; the single
`ld.shared.v4.b32` in every deterministic entry is the collect-plan read of
`local_histogram[0..3]`. `xor.b32` is 0 for the 16-bit dtypes: all 35 monotone
key flips there are `xor.b16`. bf16 entries are identical to the f16 entries
apart from the source-level dtype; every opcode matches exactly.
