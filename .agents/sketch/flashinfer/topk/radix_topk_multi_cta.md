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
SINGLE_CTA=false specialization the host launchers select when a row
does not fit one CTA's shared memory.
-->

# radix_topk_multi_cta SM100: coarse WASP pipeline sketch

This non-executable design sketch describes the storage layout, thread roles,
control flow, and PTX-level operations of
[`tirx_kernels/flashinfer/topk/radix_topk_multi_cta.py`](../../../../tirx_kernels/flashinfer/topk/radix_topk_multi_cta.py).
That TIRx module is the authoritative implementation.

The instantiations are `DTYPE in {f32, f16, bf16}` x `MODE in {Basic,
PageTableTransform, RaggedTransform}` x `DETERMINISTIC in {false, true}` x
`VEC_SIZE in {4,2,1}` (f32) / `{8,4,2,1}` (f16, bf16), each mirroring one
`RadixTopKKernel_Unified<1024, VEC_SIZE, false, DETERMINISTIC, MODE, DType,
int32_t>` template instantiation. `NUM_ROWS`, `LENGTH`, `K` and the derived
`CHUNK`, `CTAS_PER_GROUP`, `NUM_GROUPS`, grid and dynamic-smem size are static per
config, exactly like the per-launch values `RadixTopKMultiCTA` (:2172),
`RadixTopKPageTableTransformMultiCTA` (:1939) and
`RadixTopKRaggedTransformMultiCTA` (:2055) compute at every call.

The accepted target is SM100/B200. This sketch covers only what differs from the
single-CTA sibling, whose approved sketch
[`radix_topk_single_cta.md`](radix_topk_single_cta.md) already fixes the
per-CTA machinery: the staged monotone-key load, the 256-bin histogram and suffix
sum, the CTA-local gt/eq count, the deterministic raking scan and both emission
loops, and the mode epilogues. Everything below is either a region that only
exists when `SINGLE_CTA=false`, or a region whose operand set changes because the
row is now split across `CTAS_PER_GROUP` CTAs.

Out of scope, because the source itself routes them elsewhere: the single-CTA
specialization; per-row `top_k_arr` (`csrc/topk.cu` always passes `nullptr`);
and deterministic groups above `RADIX_TOPK_MAX_DETERMINISTIC_CTAS_PER_GROUP == 256`,
which the launcher rejects with `cudaErrorInvalidConfiguration` (:1970, :2087,
:2204). Excluded instead by this port's chosen config domain, not by source
dispatch: groups wider than the SM count. The launchers do not reject that case
-- `num_groups = min(num_sms / ctas_per_group, num_rows)` merely clamps `0` up to
`1` (:1980-1981, :2097-2098, :2218-2219) and launches `total_ctas > num_sms`,
which the software barrier cannot survive on a non-cooperative launch. Reaching
it needs a row of roughly 8.5 M elements, outside the domain. Tile (`Tx`)
primitives are out of scope everywhere.

## Pipeline at a glance

| Warps | Role-local program | Publication/reuse edges |
| --- | --- | --- |
| all 32 warps of every CTA in a group (uniform) | Persistent row loop; per-row header; mode trivial early-out (which also clears the next round-0 global histogram buffer on CTA 0); stage this CTA's own chunk into `s_ordered`; then `NUM_ROUNDS` rounds where the CTA's local histogram is folded into the group's global histogram and read back; CTA-local gt/eq count; collect; mode epilogue; exit-mark. | Intra-CTA edges are the single-CTA ones (`bar.sync 0` around `s_ordered`, `s_hist`, `s_suffix`, `s_scalars`). **Inter-CTA edges all run through the group's `RadixRowState` in global memory**: the triple-buffered histogram publishes each round, `output_counter` publishes collect slots, the deterministic scratch publishes per-CTA gt/eq counts, and `arrival_counter` orders every one of them. |
| thread 0 of each CTA | the only thread that arrives at, spins on, and resets the group barrier | `red.relaxed.gpu.global.add.s32` publishes; `ld.global.acquire.gpu.b32` consumes; `bar.sync 0` fans the release out to the CTA |

There is still no warp specialization and no producer/consumer split *within* a
CTA. The new role split is **between CTAs of a group**: `cta_in_group == 0` owns
clearing the next histogram buffer and releasing `output_counter`, and the
last CTA to exit owns the state reset. Both are data-dependent, not
lane-dependent.

## Primitive vocabulary

Beyond the single-CTA vocabulary (`copy_g2r_v`, `copy_r2s`, `copy_s2r`,
`to_ordered`, `atomic_add_shared`, `shfl_down`, `block_exclusive_sum*`,
`barrier()` = `bar.sync 0`), this sketch adds the inter-CTA layer:

```python
group_arrive()          # one fenced arrival on the group counter
group_wait(target)      # thread-0 acquire spin until the counter reaches target
group_barrier()         # arrive + spin + bar.sync + phase++ + bar.sync
atomic_add_global(dst, slot, v)   # plain relaxed global read-modify-write, returns old
atomic_add_release_global(dst, slot, v)  # fenced; returns old (exit mark only)
store_release_global(slot, v)     # fenced release store
```

`group_barrier()` is shown expanded at its first use and folded thereafter,
because it is a fixed four-operation sequence whose count is the correctness
invariant of this kernel. Note it contains **two** CTA barriers, not one:
`wait_ge` ends with its own `__syncthreads()` (:133) and
`AdvanceRadixGroupBarrier` issues a second one after `barrier_phase++` (:182).

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

variant = specialize(DTYPE=("f32","f16","bf16"),
                     MODE=("Basic","PageTableTransform","RaggedTransform"),
                     DETERMINISTIC=(False, True),
                     VEC_SIZE=(8,4,2,1),
                     target="sm_100a")
# instruction_selection: none; extent: compile-time instantiations

BLOCK       = 1024
RADIX       = 256
OBITS       = 32 if DTYPE == "f32" else 16
NUM_ROUNDS  = OBITS // 8                       # 4 (f32) | 2 (f16, bf16)
OBYTES      = OBITS // 8

# Launcher arithmetic, replicated at specialization time (:2178-2213):
#   VEC_SIZE      = 1 if (MODE != Basic and row_starts) else gcd(16//sizeof, LENGTH)
#   avail         = max_smem_optin - 2080 - (8512 if DETERMINISTIC else 0)
#   max_chunk     = max(round_down(avail // OBYTES, VEC_SIZE), VEC_SIZE * BLOCK)
#   CTAS_PER_GROUP= ceil_div(LENGTH, max_chunk)        >= 2 in this specialization
#   CHUNK         = min(round_up(ceil_div(LENGTH, CTAS_PER_GROUP), VEC_SIZE), max_chunk)
#   NUM_GROUPS    = max(1, min(num_sms // CTAS_PER_GROUP, NUM_ROWS))
# The launcher rejects DETERMINISTIC with CTAS_PER_GROUP > 256
# (cudaErrorInvalidConfiguration, :2204).

FIXED_ALLOC = 2080            # what the launcher allocates (:2190-2191, 5 scalars)
ORDERED_OFF = 2064            # what THIS specialization addresses (:1194-1200, 4 scalars)
SMEM_BYTES  = FIXED_ALLOC + CHUNK * OBYTES

launch_config = launch(
    grid=(NUM_GROUPS * CTAS_PER_GROUP, 1, 1),
    block=(BLOCK, 1, 1),
    dynamic_smem_bytes=SMEM_BYTES,
    launch_bounds=BLOCK,
)
# instruction_selection: none; extent: static launch metadata

# ===========================================================================
# Storage
# ===========================================================================

smem      = raw_shared(dtype="u8", bytes=SMEM_BYTES, alignment=16)
s_hist    = view(smem[0:1024],    dtype="u32", shape=(256,))
s_suffix  = view(smem[1024:2048], dtype="u32", shape=(256,))
s_scalars = view(smem[2048:2064], dtype="u32", shape=(4,))   # prefix, rem_k, bucket, rem_k'
s_ordered = view(smem[2064:], dtype="u32" if OBYTES == 4 else "u16", shape=(CHUNK,))
# instruction_selection: none; extent: static shared layout.
#   NOTE the 16-byte gap: the launcher sizes the header for 5 scalars but this
#   specialization declares 4 (:1194-1200), so `shared_scalars[4]` -- the
#   single-CTA output counter -- aliases `s_ordered[0]`. It is passed to the
#   collect helper and never written on any multi-CTA branch; it MUST stay
#   write-dead here.

if DETERMINISTIC:
    s_scan_temp = static_shared(bytes=8480, alignment=16)     # (:1095-1102)
# instruction_selection: none; extent: `.shared .align 16 .b8 ...scan_temp_storage[8480]`

s_is_last = static_shared(bytes=4, alignment=4)               # (:208)
# instruction_selection: none; extent: `.shared .align 4 .u32 ...s_is_last_cta`.
#   A module-scope static shared scalar disjoint from the `smem` arena, declared
#   inside the reset helper and therefore live only across the exit block
#   (:212-217). It exists on every instantiation, deterministic or not.

# Per-group state in global memory (:139-152), one RadixRowState per group and,
# when DETERMINISTIC, one collect scratch per group immediately after them.
# Word offsets inside a state: hist[0..767], remaining_k 768, prefix 769,
# arrival 770, output 771, sum_topk 772; stride 773 words.
state    = global_view(row_states, base=group_id * 773)
det_scr  = global_view(row_states, base=NUM_GROUPS * 773 + group_id * 512)   # DETERMINISTIC only
# instruction_selection: none; extent: address arithmetic on the workspace pointer

def radix_topk_multi_cta(input, output_indices, output_values, aux_data, lengths,
                         row_starts, page_table_row_starts, row_to_batch,
                         row_states, aux_stride):
    cta = cta_id(axis="x", extent=NUM_GROUPS * CTAS_PER_GROUP)
    # instruction_selection: mov.u32 %ctaid.x; extent: scalar per thread
    tx  = thread_id(extent=BLOCK, dtype="uint32")
    # instruction_selection: mov.u32 %tid.x; extent: scalar per thread
    group_id     = cta // CTAS_PER_GROUP                     # (:1188)
    cta_in_group = cta % CTAS_PER_GROUP                      # (:1189)
    # instruction_selection: the source divisor is a runtime argument and lowers to
    #   div.u32 (.loc 2 1188) + rem.u32 (.loc 2 1189). CTAS_PER_GROUP is static in
    #   this port, so the divide becomes a constant-divisor sequence instead:
    #   shr.u32 alone when it is a power of two, otherwise mul.hi.u32 + shr.u32 +
    #   mul.lo.s32 + sub.s32. Deliberate divergence from the export.
    #   extent: two scalar per thread
    barrier_phase = 0        # per-CTA register; every CTA of a group must reach the
                             # same value at every barrier (:180)

    total_iterations = ceil_div(NUM_ROWS, NUM_GROUPS)        # (:1213-1214)

    # =======================================================================
    # Persistent row loop (:1218-1220). Rows stride by NUM_GROUPS, so all
    # CTAS_PER_GROUP CTAs of a group always work on the same row. `iter` is
    # load-bearing scheduler state: it drives the triple-buffer index through
    # `global_round` (:691) and the trivial branch's next-buffer clear (:1260).
    # =======================================================================
    iter = 0
    while iter < total_iterations:
        row = group_id + iter * NUM_GROUPS                    # (:1219)
        if row >= NUM_ROWS:                                   # (:1220)
            break
        # Live guard, not dead: total_iterations = ceil_div(NUM_ROWS, NUM_GROUPS),
        # so on the last iteration a high group_id overshoots (NUM_ROWS=5,
        # NUM_GROUPS=4 => total_iterations=2, group_id=3 => row=7). It stays
        # group-uniform -- every CTA of a group shares group_id -- so leaving the
        # loop here cannot desynchronize the barrier accounting.
        # ---- per-row header (:1221-1240) --------------------------------
        # identical to the single-CTA sketch: row_start / page_start / length
        # loads, row_input and row_output bases.
        # instruction_selection: ld.global.b32 per optional pointer; extent: scalar per row

        # ---- mode trivial early-out (:1243-1307) -------------------------
        # The emit loops differ from the single-CTA ones, and differ between modes.
        if MODE == "Basic" and k >= length:                    # (:1244)
            # Basic SPLITS the identity emit across the group: its own chunk bounds,
            # recomputed here because the shared ones below are not yet in scope.
            t_start  = cta_in_group * CHUNK                    # (:1246)
            t_actual = clamp(length - t_start, 0, CHUNK)       # (:1247-1248)
            i = tx
            while i < t_actual:                               # (:1250)
                if t_start + i < k:                           # (:1251) inner guard
                    store_global(row_output[t_start + i], t_start + i)
                    # instruction_selection: st.global.b32 (.loc 2 1252); extent: one scalar store
                    v = load_global(row_input[t_start + i])
                    # instruction_selection: ld.global.b32 | ld.global.b16 (.loc 2 1253);
                    #   extent: one scalar load
                    store_global(output_values[row * k + t_start + i], v)
                    # instruction_selection: st.global.b32 | st.global.b16 (.loc 2 1253);
                    #   extent: one scalar store
                i = i + BLOCK
        elif MODE == "PageTableTransform" and length <= k:     # (:1272)
            # PageTable and Ragged do NOT split: EVERY CTA of the group redundantly
            # writes the whole [0, k) range, and the -1 tail past `length`.
            i = tx
            while i < k:                                      # (:1273)
                if i < length:                                # (:1275) GUARDED load
                    page_id = load_global(src_page_entry[page_start + i])
                    # instruction_selection: setp.ge.u32 + @p bra around
                    #   ld.global.b32; extent: one predicated scalar load
                else:
                    page_id = -1
                    # instruction_selection: mov.b32 -1; extent: one scalar
                store_global(row_output[i], page_id)          # (:1274)
                # instruction_selection: st.global.b32; extent: one scalar store
                i = i + BLOCK
            # The load MUST stay inside the guard. Past `length` the slot is
            # outside this row's page-table range, so hoisting it out of the
            # ternary would be an out-of-bounds read, not just a different
            # opcode mix. The export confirms a branch, with no selp.b32 in
            # this block.
        elif MODE == "RaggedTransform" and length <= k:        # (:1291)
            i = tx
            while i < k:                                      # (:1292)
                store_global(row_output[i], select(i < length, i + offset, -1))
                # instruction_selection: setp.lt.u32 + add.s32 + selp.b32 +
                #   st.global.b32 (.loc 2 1293); extent: one scalar store. Pure
                #   arithmetic, so this one really is a select.
                i = i + BLOCK
        if <trivial branch taken>:
            # Common tail of all three trivial branches: a skipped iteration would
            # leave the next round-0 histogram buffer dirty, so CTA 0 clears it.
            next_first = ((iter + 1) * NUM_ROUNDS) % 3        # (:1260, :1280, :1298)
            if cta_in_group == 0:
                b = tx
                while b < RADIX:
                    store_global(state.hist[next_first * 256 + b], 0)
                    # instruction_selection: st.global.b32; extent: one scalar store per iteration
                    b = b + BLOCK
            # no barrier here: the next iteration opens with the initial group
            # barrier before any CTA touches a histogram (:681). The branch is
            # row-uniform, so every CTA of the group skips the same barriers.
            iter = iter + 1
            continue

        chunk_start = cta_in_group * CHUNK                    # (:1309)
        actual      = clamp(length - chunk_start, 0, CHUNK)    # (:1310-1311), may be 0
        # instruction_selection: mul.lo.s32 + add.s32 + min.u32 + sub.s32, with the
        #   `chunk_start < length` zero case folded into the setp.lt.u32 + and.pred
        #   guards of the loops that consume it rather than a materialized selp.b32;
        #   extent: two scalar
        #   A trailing CTA with actual == 0 still executes every barrier below and
        #   still publishes zero counts.

        # =====================================================================
        # Stage 1: stage THIS CTA's chunk (:602-623). Same body as single-CTA,
        # but the global base carries chunk_start and the bound is `actual`.
        # =====================================================================
        <single-CTA staging loop over `actual`, reading row_input + chunk_start>
        barrier()
        # instruction_selection: bar.sync 0; extent: CTA-wide, closes the load (:622)

        if tx == 0:
            store_shared_pair(s_scalars[0], 0, k)              # (:672)
            # instruction_selection: st.shared.v2.b32 [smem+2048]; extent: one paired store
        # NOTE: no shared output-counter init here; multi-CTA uses the global one.
        barrier()
        # instruction_selection: bar.sync 0; extent: CTA-wide (:677)

        # =====================================================================
        # Stage 1b: the group's first rendezvous (:679-687)
        # =====================================================================
        # -- group_barrier() expanded once, folded everywhere below --
        if tx == 0:                                            # (:176)
            fence_acq_rel_gpu()
            # instruction_selection: fence.acq_rel.gpu; extent: one per arrival
            atomic_add_global_noret(state.arrival, 1)
            # instruction_selection: red.relaxed.gpu.global.add.s32; extent: one per CTA
        target = (barrier_phase + 1) * CTAS_PER_GROUP          # (:179)
        if tx == 0:                                            # (:128) only thread 0 spins
            v = load_acquire_global(state.arrival)
            # instruction_selection: ld.global.acquire.gpu.b32; extent: one per spin trip
            while v < target:                                  # (:130-132) busy-poll, no backoff
                v = load_acquire_global(state.arrival)
                # instruction_selection: ld.global.acquire.gpu.b32; extent: one per spin trip
        barrier()
        # instruction_selection: bar.sync 0 (.loc 2 133, the tail of wait_ge);
        #   extent: CTA-wide, fans the acquire out to every thread
        barrier_phase = barrier_phase + 1                      # (:181)
        barrier()
        # instruction_selection: bar.sync 0 (.loc 2 182); extent: CTA-wide. The source
        #   issues a SECOND CTA barrier after the phase bump; both are part of the
        #   protocol and the port must emit both.
        # -- end group_barrier(): 2 x bar.sync 0 per phase --

        if cta_in_group == 0 and tx == 0:                      # (:683-686)
            fence_acq_rel_gpu()
            # instruction_selection: fence.acq_rel.gpu; extent: one per release store
            store_release_global(state.output, 0)
            # instruction_selection: st.release.gpu.global.b32; extent: one scalar store
        # ordering: the NUM_ROUNDS group barriers below all precede the first
        # reader of output_counter in the collect.

        # =====================================================================
        # Stage 2: NUM_ROUNDS rounds, reduced across the group (:690-775)
        # =====================================================================
        rnd = 0
        while rnd < NUM_ROUNDS:                                # rolled (:690)
            # instruction_selection: add.s32 + setp.ne.b32 back-edge; extent: loop control
            global_round = iter * NUM_ROUNDS + rnd             # (:691)
            cur_buf  = global_round % 3                        # (:700)
            next_buf = (global_round + 1) % 3                  # (:701)
            # instruction_selection: no rem.u32 -- the constant-3 modulus lowers to
            #   mul.hi.u32 (magic 0xAAAAAAAB = -1431655765) + shr.u32 + mul.lo.s32 + sub.s32 per
            #   value; extent: two scalars, 8 instructions per round
            shift, mask, prefix, remaining_k = <single-CTA round header>

            # local histogram, exactly as single-CTA
            <clear s_hist; barrier; prefix-masked build with atom.shared.add.u32; barrier>

            # NEW: fold this CTA's local histogram into the group's buffer, and let
            # CTA 0 clear the buffer the NEXT round will use, before the barrier.
            b = tx
            while b < RADIX:                                   # (:727-731)
                h = copy_s2r(s_hist[b])
                # instruction_selection: ld.shared.b32; extent: one shared load per iteration
                if h > 0:
                    atomic_add_global_noret(state.hist[cur_buf * 256 + b], h)
                    # instruction_selection: atom.global.add.u32 (plain relaxed, result
                    #   unused -- the source calls atomicAdd on global memory here);
                    #   extent: one per nonzero bin
                b = b + BLOCK
            if cta_in_group == 0:                              # (:732-736)
                b = tx
                while b < RADIX:
                    store_global(state.hist[next_buf * 256 + b], 0)
                    # instruction_selection: st.global.b32; extent: one scalar store per iteration
                    b = b + BLOCK
            group_barrier()                                    # (:737)
            # instruction_selection: fence.acq_rel.gpu + red.relaxed.gpu.global.add.s32 +
            #   ld.global.acquire.gpu.b32 spin + bar.sync 0 x2; extent: one barrier phase

            b = tx
            while b < RADIX:                                   # (:739-741)
                copy_r2s(s_suffix[b], load_global(state.hist[cur_buf * 256 + b]))
                # instruction_selection: ld.global.b32 + st.shared.b32; extent: one pair per iteration
                b = b + BLOCK
            barrier()
            # instruction_selection: bar.sync 0; extent: CTA-wide, group histogram staged

            <single-CTA RadixSuffixSum, threshold-bucket search, prefix publish>
            rnd = rnd + 1

        pivot = copy_s2r(s_scalars[0])                         # (:777)
        # instruction_selection: ld.shared.b32 (f32) | ld.shared.b16 + cvt.u16.u32; extent: one load

        # =====================================================================
        # Stage 3: gt (and eq) counts -- CTA-LOCAL (:774-830). The reduction and
        # the shared-atomic accumulation are the single-CTA ones; the group
        # aggregation happens in the collect, not here.
        #
        # OPERAND RULE for every delegated loop in this sketch (Stage 3, the
        # non-deterministic gt pass, and both deterministic emission paths):
        # under SINGLE_CTA=false the single-CTA bound `length` becomes `actual`
        # (:793, :928, :950, :1105, :1116, :1128) and every emitted original index
        # `i` becomes `chunk_start + i` (:933, :1109, :1131, :1134). Only the
        # bound and the emitted index change; the instruction selection is not.
        # =====================================================================
        <single-CTA count loop, warp shuffle reduction, warp-leader shared atomics>
        gt_count = copy_s2r(s_suffix[0])
        if DETERMINISTIC:
            eq_count = copy_s2r(s_suffix[1])

        # =====================================================================
        # Stage 3b: epilogue-scope aux recompute, per row (:1344-1345, :1373)
        # =====================================================================
        <single-CTA Stage 3b>

        if not DETERMINISTIC:
            # =================================================================
            # Stage 4a: non-deterministic collect (:906-963)
            # =================================================================
            if tx == 0:
                store_shared(s_hist[0], 0)                      # local_offset_gt
                # instruction_selection: st.shared.b32; extent: one scalar store
                if gt_count > 0:
                    base = atomic_add_global(state.output, gt_count)   # (:919)
                    # instruction_selection: atom.global.add.u32 (plain relaxed, result
                    #   used); extent: one per CTA
                    store_shared(s_hist[1], base)
                    # instruction_selection: st.shared.b32; extent: one scalar store
            barrier()
            # instruction_selection: bar.sync 0; extent: CTA-wide, collect base published

            <single-CTA gt pass: shared atomic slot, s_hist[1] re-read, EMIT>

            group_barrier()                                     # (:941)
            # instruction_selection: fence.acq_rel.gpu + red.relaxed.gpu.global.add.s32 +
            #   ld.global.acquire.gpu.b32 spin + bar.sync 0 x2; extent: one barrier phase.
            #   REQUIRED: without it a CTA could claim `== pivot` slots while a peer is
            #   still writing `> pivot` ones.

            i = tx
            while i < actual:                                   # (:949-963)
                key = copy_s2r(s_ordered[i])
                # instruction_selection: ld.shared.b32 | ld.shared.b16; extent: one shared load
                if key == pivot:
                    pos = atomic_add_global(state.output, 1)     # (:957)
                    # instruction_selection: atom.global.add.u32; extent: one per candidate
                    if pos < k:
                        EMIT(chunk_start + i, pivot, pos)
                i = i + BLOCK
        else:
            # =================================================================
            # Stage 4b: deterministic collect (:1048-1084)
            # =================================================================
            if tx == 0:
                store_shared(s_hist[1], 0)                      # eq prefix (:1051)
                # instruction_selection: st.shared.b32 [smem+4]; extent: one scalar store
                store_shared(s_hist[4], 0)                      # eq take  (:1052)
                # instruction_selection: st.shared.b32 [smem+16]; extent: one scalar store
                store_global(det_scr.gt[cta_in_group], gt_count)
                # instruction_selection: st.global.b32; extent: one scalar store
                store_global(det_scr.eq[cta_in_group], eq_count)
                # instruction_selection: st.global.b32; extent: one scalar store
            group_barrier()                                     # (:1056)
            # instruction_selection: fence.acq_rel.gpu + red.relaxed.gpu.global.add.s32 +
            #   ld.global.acquire.gpu.b32 spin + bar.sync 0 x2; extent: one barrier phase

            if tx == 0:                                         # (:1058-1079)
                gt_prefix = 0; row_total_gt = 0; eq_prefix = 0
                c = 0
                while c < CTAS_PER_GROUP:      # serial, bounded by the 256 cap
                    # instruction_selection: add.s32 + setp.lt.u32 back-edge; extent: loop control
                    c_gt = load_global(det_scr.gt[c])
                    # instruction_selection: ld.global.b32; extent: one scalar load per peer CTA
                    c_eq = load_global(det_scr.eq[c])
                    # instruction_selection: ld.global.b32; extent: one scalar load per peer CTA
                    if c < cta_in_group:
                        gt_prefix = gt_prefix + c_gt
                        eq_prefix = eq_prefix + c_eq
                        # instruction_selection: selp.b32 + add.s32 x2; extent: four scalar
                    row_total_gt = row_total_gt + c_gt
                    # instruction_selection: add.s32; extent: scalar
                    c = c + 1
                eq_needed = max(k - row_total_gt, 0)
                # instruction_selection: sub.s32 + max.u32; extent: two scalar
                store_shared_quad(s_hist[0], gt_prefix, eq_prefix, row_total_gt, eq_needed)
                # instruction_selection: st.shared.v4.b32 [smem] (.loc 2 1074);
                #   extent: one 128-bit store covering :1071-1074
                store_shared(s_hist[4], 0)                      # (:1075)
                # instruction_selection: st.shared.b32 [smem+16]; extent: one scalar store
                if eq_needed > eq_prefix:                       # (:1076)
                    eq_take = min(eq_count, eq_needed - eq_prefix)
                    # instruction_selection: sub.s32 + min.u32; extent: two scalar
                    store_shared(s_hist[4], eq_take)            # (:1077)
                    # instruction_selection: st.shared.b32 [smem+16]; extent: one scalar
                    #   store. The source writes this slot TWICE -- unconditional zero
                    #   then conditional overwrite -- not once.
            barrier()
            # instruction_selection: bar.sync 0; extent: CTA-wide, collect plan published

            plan = copy_s2r_quad(s_hist[0])
            # instruction_selection: ld.shared.v4.b32 {..},[smem]; extent: one 128-bit load
            gt_base  = plan[0]                                   # (:1085)
            eq_base  = plan[2] + plan[1]                         # row_total_gt + eq_prefix (:1083)
            # instruction_selection: add.s32; extent: scalar
            eq_limit = copy_s2r(s_hist[4])                       # (:1082)
            # instruction_selection: ld.shared.b32; extent: one scalar load
            gt_limit = max(k - gt_base, 0)                       # (:1086-1087)
            # instruction_selection: max.u32 + sub.s32; extent: two scalar

            <single-CTA deterministic emission: gt-only strided collect when
             eq_limit == 0, otherwise the paired raking scan and emit loop>

        # =====================================================================
        # Stage 5: PageTable in-place gather, split across the group (:1359-1371;
        # :1352-1358 is the SINGLE_CTA branch)
        # =====================================================================
        if MODE == "PageTableTransform":
            group_barrier()                                     # (:1361)
            # instruction_selection: fence.acq_rel.gpu + red.relaxed.gpu.global.add.s32 +
            #   ld.global.acquire.gpu.b32 spin + bar.sync 0 x2; extent: one barrier phase.
            #   REQUIRED: the gather reads back indices peers wrote.
            elems = ceil_div(k, CTAS_PER_GROUP)                  # (:1364)
            my_start = cta_in_group * elems
            my_end   = min(my_start + elems, k)
            # instruction_selection: mul.lo.s32 + add.s32 + min.u32; extent: three scalar
            i = my_start + tx
            while i < my_end:                                   # (:1367-1370)
                idx = load_global(row_output[i])                # (:1368)
                # instruction_selection: ld.global.b32; extent: one scalar load
                page_id = load_global(src_page_entry[page_start + idx])   # (:1369)
                # instruction_selection: ld.global.b32; extent: one scalar load
                store_global(row_output[i], page_id)            # (:1369)
                # instruction_selection: st.global.b32; extent: one scalar store
                i = i + BLOCK
            # no trailing barrier: CTAs wrote disjoint ranges and the next
            # iteration opens with the initial group barrier.

        iter = iter + 1

    # =======================================================================
    # Exit: the LAST CTA of the group restores the state for the next launch
    # (:203-238, called at :1384; issue #3610). Resetting from the leading CTA instead can zero
    # the counter while a peer is still spinning in its final wait, wedging the
    # stream permanently.
    # =======================================================================
    barrier()
    # instruction_selection: bar.sync 0 (.loc 2 212); extent: CTA-wide. Converges the
    #   CTA before tx0's release-ordered exit mark so the fence's cumulativity covers
    #   every thread's prior writes.
    if tx == 0:
        exit_target = (barrier_phase + 1) * CTAS_PER_GROUP     # (:214)
        fence_acq_rel_gpu()
        # instruction_selection: fence.acq_rel.gpu; extent: one
        old = atomic_add_global(state.arrival, 1)              # (:215)
        # instruction_selection: atom.relaxed.gpu.global.add.s32 (result used);
        #   extent: one per CTA
        store_shared(s_is_last, old + 1 == exit_target)        # (:215)
        # instruction_selection: add.s32 + setp.eq.b32 + selp.b32 + st.shared.b32;
        #   extent: three scalar, into the static `s_is_last_cta`
    barrier()
    # instruction_selection: bar.sync 0 (.loc 2 217); extent: CTA-wide, verdict published
    if copy_s2r(s_is_last) != 0:
        buf = 0
        while buf < 3:                                          # (:219-223)
            b = tx
            while b < RADIX:
                store_global(state.hist[buf * 256 + b], 0)
                # instruction_selection: st.global.b32; extent: one scalar store per iteration
                b = b + BLOCK
            buf = buf + 1
        if DETERMINISTIC:
            w = tx
            while w < 512:                                      # (:228-230, guarded :224)
                store_global(det_scr.word[w], 0)
                # instruction_selection: st.global.b32; extent: one scalar store per iteration
                w = w + BLOCK
        barrier()
        # instruction_selection: bar.sync 0 (.loc 2 233); extent: CTA-wide, clears complete
        if tx == 0:
            fence_acq_rel_gpu()
            # instruction_selection: fence.acq_rel.gpu; extent: one
            store_release_global(state.arrival, 0)
            # instruction_selection: st.release.gpu.global.b32; extent: one scalar store
```

## Barrier accounting

The group barrier count is the correctness invariant: every CTA of a group must
execute the identical sequence, because the targets are absolute multiples of
`CTAS_PER_GROUP` on a counter that is never reset mid-launch.

Each group barrier is two `bar.sync 0` (`wait_ge` :133 and the phase tail :182),
so the CTA-barrier column is twice the group-barrier column.

| region | group barriers | `bar.sync 0` | source |
| --- | ---: | ---: | --- |
| trivial row iteration | 0 | 0 | :1257-1306 |
| initial rendezvous | 1 | 2 | :681 |
| radix rounds | NUM_ROUNDS (4 f32 / 2 16-bit) | 2 x NUM_ROUNDS | :737 |
| collect (both paths) | 1 | 2 | :941 / :1056 |
| PageTable gather | +1 | +2 | :1361 |
| exit mark | 1 arrival, no wait | 3 (:212, :217, :233) | :215 |

Static PTX for the group protocol, from a fresh line-info export of the exact
instantiations (`.porting/radix_topk_multi_cta/ptx/instantiate.cu`, production JIT
flags), confirms the site counts: three `ld.global.acquire.gpu.b32` /
`red.relaxed.gpu.global.add.s32` pairs for Basic and Ragged (initial, rounds,
collect) and four for PageTable, one `atom.relaxed.gpu.global.add.s32` (the exit
mark), two `st.release.gpu.global.b32` (the output-counter clear and the reset),
and one `fence.acq_rel.gpu` per release site.

The `bar.sync` totals reconcile against the single-CTA kernel exactly once both
barriers per phase are counted: Basic non-det `28 + 3x2 + 3 - 1 = 36`, Basic det
`33 + 3x2 + 3 = 42`, PageTable non-det `29 + 4x2 + 3 - 2 = 38`. A port that
emitted one barrier per phase would land 3-4 short and, worse, would let threads
race past the acquire before thread 0 published the phase bump.

## Static specialization boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| `SINGLE_CTA = false` | static (launcher decision) | brings in the whole group protocol: `RadixRowState`, the triple-buffered global histogram, the arrival counter, the deterministic scratch and the exit reset |
| `CTAS_PER_GROUP` | static per config | barrier targets, chunk split, deterministic prefix loop trip count, PageTable gather split |
| `NUM_GROUPS` | static per config | grid, workspace extent, row-loop stride |
| `DETERMINISTIC` | static per config | collect path, the 8480-byte scan scratch, whether the group scratch exists, and the launcher's smem headroom (hence `max_chunk` and the domain) |
| `MODE` | static per config | per-row header, trivial branch, `EMIT`, and whether Stage 5 exists |
| `VEC_SIZE` | static per config | staging load width; forced to 1 by a non-null `row_starts` in the non-Basic launchers |
| `top_k_arr` | always null | Basic's per-row k branch is dead |
| ordered-key base 2064 vs 2080 allocation | static | `shared_scalars[4]` aliases `s_ordered[0]` and stays write-dead |

## TIRx module and benchmark contract

- `KERNEL_META = {"name": "radix_topk_multi_cta", "category": "flashinfer",
  "runtime_cuda_archs": ["sm_100a"]}`.
- Plain TIRx only: a `T.SMEMPool` arena at the source's byte offsets, explicit
  `while`/stepped `T.serial` loops, and native `T.ptx.*` forms for every key
  operation. Global and shared memory are reached exclusively through
  `T.ptx.ld/st.*` on `buffer.ptr_to([...])`, never a native `BufferLoad`/
  `BufferStore`, which the repository's low-level IR contract forbids. The
  acquire spin lives in a typed private `PrimFunc` reached through a `GlobalVar`
  in the returned `IRModule`, because `T.cuda.func_call` is likewise forbidden and
  flattening an acquire poll into its caller is a documented hazard.
- No `Tx` tile primitives anywhere.
- The workspace is one extra `uint32` global argument holding
  `RadixRowState[NUM_GROUPS]` and, when deterministic, the collect scratch after
  them; it is zero-initialized before the first launch and self-resets at exit.
- Correctness compares against the same FlashInfer kernel launched through the raw
  `topk` FFI module. `FLASHINFER_TOPK_ALGO=multi_cta` is read by
  `GetTopKAlgoOverride` (:3299-3305) and forces `ShouldUseFilteredTopK` to return
  false (:3342), which pins dispatch onto the **radix family** rather than
  FilteredTopK. It does not by itself select this specialization: within
  `RadixTopK*MultiCTA` the single-CTA/multi-CTA split is made by
  `ctas_per_group == 1`, so reaching this kernel additionally requires a config
  whose row does not fit one CTA's shared memory -- which every entry of `CONFIGS`
  satisfies by construction. Comparison is exact positional equality for
  deterministic configs, and equality of the selected score-value multiset
  otherwise, since the non-deterministic collect's cross-CTA atomics make both the
  order and, under ties, the selected set vary run to run.

## Instruction selection is a lowering consequence

| Primitive/schedule pattern | PTX family (fresh SM100a export) |
| --- | --- |
| group arrival | `fence.acq_rel.gpu` + `red.relaxed.gpu.global.add.s32` |
| group wait | `ld.global.acquire.gpu.b32` in a busy-poll loop, then `bar.sync 0` |
| exit mark | `fence.acq_rel.gpu` + `atom.relaxed.gpu.global.add.s32` (result used) |
| output-counter clear, arrival reset | `fence.acq_rel.gpu` + `st.release.gpu.global.b32` |
| global histogram accumulate | `atom.global.add.u32` (plain relaxed, result unused) |
| non-deterministic collect slots | `atom.global.add.u32` (result used) |
| global histogram clear / read-back / det scratch | `st.global.b32` / `ld.global.b32` |
| everything inside a CTA | as in the single-CTA sketch: `ld.global.v4.b32` staging, `setp/selp/xor` key flip, `st.shared.*`, `atom.shared.add.u32`, `bar.sync 0`, `shfl.sync.down.b32`, and for the deterministic collect `shfl.sync.up.b32` with `v2`/`v4` raking traffic |

Static opcode counts per exported instantiation:

| Family | f32 vec4 Basic | f32 vec4 Basic det | f16 vec8 Basic | f32 vec1 Basic | f32 vec4 PageTable | f32 vec4 PageTable det | f32 vec4 Ragged | f32 vec4 Ragged det |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ld.global.acquire.gpu.b32` | 3 | 3 | 3 | 3 | 4 | 4 | 3 | 3 |
| `red.relaxed.gpu.global.add.s32` | 3 | 3 | 3 | 3 | 4 | 4 | 3 | 3 |
| `atom.relaxed.gpu.global.add.s32` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| `st.release.gpu.global.b32` | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| `fence.acq_rel.gpu` | 6 | 6 | 6 | 6 | 7 | 7 | 6 | 6 |
| `atom.global.add.u32` | 5 | 1 | 5 | 5 | 5 | 1 | 5 | 1 |
| `atom.shared.add.u32` | 7 | 5 | 7 | 7 | 7 | 5 | 7 | 5 |
| `bar.sync` | 36 | 42 | 36 | 36 | 38 | 44 | 36 | 42 |
| `ld.global.v4.b32` | 3 | 3 | 3 | 0 | 3 | 3 | 3 | 3 |
| `ld.shared.b32` | 37 | 101 | 24 | 37 | 37 | 93 | 37 | 95 |
| `st.shared.b32` | 29 | 63 | 14 | 24 | 23 | 57 | 23 | 57 |
| `st.global.b32` | 19 | 40 | 12 | 19 | 25 | 25 | 26 | 26 |
| `shfl.sync.down.b32` | 5 | 10 | 5 | 5 | 5 | 10 | 5 | 10 |
| `shfl.sync.up.b32` | 0 | 17 | 0 | 0 | 0 | 17 | 0 | 17 |
| `ld.shared.v2.b32` | 1 | 34 | 1 | 1 | 1 | 35 | 1 | 34 |
| `st.shared.v2.b32` | 4 | 38 | 4 | 4 | 4 | 38 | 4 | 38 |
| `ld.shared.v4.b32` | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 1 |
| `st.shared.v4.b32` | 1 | 2 | 1 | 0 | 1 | 2 | 1 | 2 |

The single `atom.global.add.u32` in the deterministic columns is the global
histogram accumulate at `.loc 2 729`; the non-deterministic columns add four more
for the collect's `output_counter`, whose eq pass carries the source's
`#pragma unroll 2`. The `bar.sync` totals exceed the single-CTA kernel's 28/33 by
exactly the CTA-wide fan-out each group barrier and the exit reset contribute.
