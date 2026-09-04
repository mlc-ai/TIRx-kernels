<!--
This file is a design sketch for a TIRx port of FlashInfer
(https://github.com/flashinfer-ai/flashinfer @ 9f5051736e9fd5cab41c06118a7d4b5c1de23a6d),
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# FlashInfer uniform FP8 QKV paged MSA decode SM100: coarse WASP pipeline sketch

This is a non-executable operation sketch for
[`tirx_kernels/flashinfer/msa_ops/blackwell_msa_decode_uniform_fp8_qkv_paged_sm100.py`](../../../tirx_kernels/flashinfer/msa_ops/blackwell_msa_decode_uniform_fp8_qkv_paged_sm100.py),
which becomes the executable source of truth after this sketch passes review.
It freezes the source's twelve-warp persistent program, one Q TMA, interleaved
four-stage K/V ring, two score/probability instances, two output accumulators,
FP8 probability conversion, correction protocol, and BF16/LSE epilogue.

The source is
`csrc/blackwell_msa/sm100a/blackwell_msa_decode_uniform_fp8_qkv_paged.cu`
at FlashInfer commit `9f5051736e9fd5cab41c06118a7d4b5c1de23a6d`, entry
`kernel_blackwell_batch_attention_msa_decode_uniform_fp8_natural_sm100_v1`.
The target is only the production `uniform_fp8_qkv_decode.paged` route on
SM100a: causal paged decode, Q/K/V E4M3, O BF16, LSE FP32, D=page=128,
top-k=16, uniform `1 <= seqlen_q <= 32`, integral GQA with group size at most
16, and the default Q-offset path. SM103, non-paged input, other dtypes,
non-causal mode, explicit Q offsets, other page/head dimensions, and other
MSA route families are out of scope. No tile primitive, first-class layout,
`K.cuda.func_call`, or inline-CUDA function-call exemption is allowed.

After the independent sketch reviewer returns PASS, this file is immutable.

## Writer line-info evidence

The normal FlashInfer tvm-ffi JIT module and an optimized line-info PTX are
preserved under
`.porting/blackwell_msa_decode_uniform_fp8_qkv_paged_sm100/source_export/`.
The optimized PTX SHA-256 is
`37b8611fa62e7faadb73d8d2a14b9a8c57e1f39d90bc929ef5d7bcf70b482e49`.
It was built by CUDA 13.2.51 with `-O3 -DNDEBUG -use_fast_math -lineinfo`, has
1,022 `.loc` records, and declares `.version 9.2`, `.target sm_100a`,
`.maxntid 384`, and `.extern .shared .align 1024`. Every instruction
selection below is read from that optimized entry. The normal module was also
launched with real FP8 tensors on GB200; O and LSE were finite.

## Pipeline at a glance

All role predicates are independent source-order `if` blocks over a
warp-uniform `warp = shfl(threadIdx.x / 32, lane 0)`. Each live role walks
`work = blockIdx.x; work < total_q * Hkv; work += gridDim.x`.

| warps | role-local program | publication / reuse edges |
| --- | --- | --- |
| 0 and 4 | stage 0/1 score consumer: load one 32-row x 128-column S tile from TMEM, clamp its selected page, update online max/sum, publish correction scale, cast 128 probabilities to E4M3, overwrite the upper half of S with P, and finally publish row statistics | consume `s_full[stage]` and first-pair `p_store_turn[stage]`; produce `corr_sig[stage]`, `p_store_turn[other]`, and `p_full[stage]` |
| 1, 5, 6, 7, 9, 10, 11 | idle | none |
| 2 | elected producer lane: wait/reload Q, resolve logical pages and causal valid-column counts, then emit Q/K/V rank-3 TMAs through the four-stage aliased ring in `K0,K1,V0,K2,V1,...,K15,V14,V15` order | consume `q_empty`, `kv_empty[slot]`; produce `q_full`, `kv_full[slot]`, shared page metadata |
| 3 | sole tcgen05 issuer: alternate QK into S0/S1 and PV from P0/P1 into O0/O1; release the ring with tcgen05 commits and return Q only after the last QK | consume `q_full`, `kv_full`, `p_full`, `decode_done`; produce `s_full`, `kv_empty`, `q_empty`, `o_full` |
| 8 | correction and epilogue: rescale each previous O accumulator as online maxima advance, release P production, merge the two final O/statistic halves, normalize, and store BF16 O plus natural-log LSE | consume `corr_sig[0/1]`, `o_full`; produce `p_full[0/1]`, `decode_done` |

## Primitive vocabulary

There is no first-class layout. Structural declarations are linear storage and
scalar address functions only:

```python
linear_gmem(name, dtype, extent)
linear_smem(name, byte_offset, byte_extent)
linear_tmem(name, column_offset, column_extent)
reg_array(name, dtype, extent)
tensor_map(name, dtype, rank, dims, byte_strides, box, swizzle)
byte_address(base, scalar_offset)
tmem_address(base_col, row_bits, column)
```

Copies state their direction:

```python
copy_g2s_tma(map, coordinates, linear_smem_destination, mbarrier)
copy_t2r(linear_tmem_source, registers)
copy_r2t(registers, linear_tmem_destination)
copy_r2g(registers, linear_gmem_destination)
load_g2r(linear_gmem_source, register)
load_s2r(linear_smem_source, register)
store_r2s(register, linear_smem_destination)
```

Computation uses only `fill`, `max`, `select`, `add`, `sub`, `mul`,
`fma`, `exp2`, `log2`, `rcp`, `cast`, and `gemm`. Schedule operations
are explicit `elect`, barrier `init`/`wait`/`arrive`/`expect_tx`,
`commit`, proxy and TMEM `fence`, CTA/warp `barrier`, register-budget
changes, and TMEM `allocate`/`relinquish`/`free`.

## Fixed resources and scalar maps

```python
specialize(
    D=128, PAGE=128, TOPK=16,
    Q_DTYPE="e4m3", KV_DTYPE="e4m3", P_DTYPE="e4m3",
    O_DTYPE="bf16", ACC_DTYPE="f32",
    WARPS=12, THREADS=384, KV_STAGES=4,
    TMEM_COLUMNS=512, SMEM_BYTES=156672,
    target="sm_100a", ptx_isa="9.2",
)

launch(
    block=(384, 1, 1),
    grid=(physical_grid(total_q * Hkv, seqlen_q), 1, 1),
    dynamic_smem_bytes=156672,
    cluster=None,
)
# instruction_selection: `.maxntid 384`, no `.reqnctapercluster`, and
# `.extern .shared .align 1024`; extent: one shape-generic entry.

def physical_grid(work, q):
    default = min(work, 148)
    return 128 if (q >= 4 and work % 128 == 0 and
                   ceildiv(work, 128) == ceildiv(work, default)) else default
```

Runtime ABI, in source order:

```text
tensor_map Qmap, tensor_map Kmap, tensor_map Vmap,
bf16* O, f32* LSE,
i32* page_table, i32* kv_indptr, i32* selected_blocks,
i32* q_offsets, i32* kv_lens,
i32 total_q, i32 seqlen_q, i32 Hq, i32 Hkv,
f32 softmax_scale_log2, f32 output_scale, i32 max_pages
```

Each Q/K/V TensorMap is a 128-byte by-value kernel parameter emitted as
`.param .align 64 .b8[128]`; its parameter alignment is 64 bytes.

`kv_indptr` and `q_offsets` are ABI-live but device-dead. Host TensorMaps:

| map | fastest-first global dimensions | byte strides | box | other fields |
| --- | --- | --- | --- | --- |
| Q | `(128, total_q*Hq, 1)` | `(Q.stride(-2), 128)` | `(128,16,1)` | UINT8, SWIZZLE_128B, no L2 promotion, no OOB fill |
| K | `(128,128,pages*Hkv)` | `(128,16384)` | `(128,128,1)` | same |
| V | `(128,128,pages*Hkv)` | `(128,16384)` | `(128,128,1)` | same |

All device storage is one-dimensional:

| interval / columns | contents and explicit map | lifetime |
| --- | --- | --- |
| SMEM `[0,192)` | 24 eight-byte mbarriers | whole CTA |
| SMEM `[192,196)` | TMEM allocation mailbox | setup/teardown |
| SMEM `[1024,17408)` | Q; TMA writes 2,048 live bytes for 16 rows x 128 bytes, within a 16 KiB reservation | one work generation |
| SMEM `[17408,82944)` | four 16 KiB K/V stages; `ring(stage)=17408+16384*stage`; K and V are exact physical aliases | streaming work generation |
| SMEM `[82944,148480)` | allocator padding from the second logical K/V reservation | never addressed |
| SMEM `[148480,152576)` | page metadata; valid count at `4*tile`, physical `(page*Hkv+head)` at `64+4*tile` | one work generation |
| SMEM `[152576,153600)` | `acc_scale[stage*128+row]` | pair to correction |
| SMEM `[153600,154624)` | final `row_sum[stage*128+row]` | work epilogue |
| SMEM `[154624,155648)` | final `row_max[stage*128+row]` | work epilogue |
| SMEM `[155648,156672)` | source reservation tail | never addressed |
| TMEM columns `[0,128)` / `[128,256)` | S0/S1, with E4M3 P written at column `base+64` | one pair |
| TMEM columns `[256,384)` / `[384,512)` | O0/O1 FP32 accumulators | one work item |

K and V share the 64 KiB live ring. The 128 KiB allocator envelope is retained
only so every following source offset remains exact.

Barrier offsets and initial counts are exact:

| name | byte offsets | count |
| --- | --- | ---: |
| `q_full`, `q_empty` | 0, 8 | 1 |
| `kv_full[0..7]` | 16..72 | 1 |
| `kv_empty[0..3]` | 80..104 | 1 |
| `s_full[0..1]` | 112,120 | 1 |
| `p_full[0..1]` | 128,136 | 64 |
| `corr_sig[0..1]` | 144,152 | 32 |
| `p_store_turn[0..1]` | 160,168 | 32 |
| `o_full` | 176 | 1 |
| `decode_done` | 184 | 32 |

## Complete operation sketch

```python
def advance_ring(stage, phase):
    stage += 1
    if stage == 4:
        stage = 0
        phase ^= 1
    return stage, phase

tid = thread_id()
warp = shuffle_index(tid // 32, lane=0, clamp=0x1f, mask=0xffffffff)
# instruction_selection: `shfl.sync.idx.b32`; extent: one warp-uniform scalar.
lane = tid % 32

if warp == 0 and elect():
    # instruction_selection: `elect.sync`; extent: one elected lane in warp 0
    # for all 24 barrier initializations.
    for each (barrier_offset, count) in the table above:
        init(byte_address(smem, barrier_offset), count)
        # instruction_selection: `mbarrier.init.shared::cta.b64`; extent:
        # 24 statically unrolled barrier initializations by one elected lane.
    fence("mbarrier_init_release_cluster")
    # instruction_selection: `fence.mbarrier_init.release.cluster`; extent: one.

barrier("warp")
# instruction_selection: `bar.warp.sync -1`; extent: every warp.
if warp == 0:
    allocate(512, mailbox=byte_address(smem, 192))
    relinquish_allocation_permit()
    # instruction_selection: all lanes of warp 0 collectively execute
    # `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [mailbox],512`
    # then `tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned`;
    # neither instruction is elected.
barrier("cta")
# instruction_selection: `bar.sync 0`; extent: 384 threads.
fence("tmem_after_thread_sync")
# instruction_selection: `tcgen05.fence::after_thread_sync`; extent: all threads.
tmem_base = load_s2r(byte_address(smem, 192))
# instruction_selection: `ld.volatile.shared.b32`; extent: one scalar per thread.

if 8 <= warp <= 11:
    set_register_budget("decrease", 96)
    # instruction_selection: `setmaxnreg.dec.sync.aligned.u32 96`; extent: warps 8..11.
barrier("cta")
# instruction_selection: `bar.sync 0`; extent: CTA.
if 0 <= warp <= 7:
    set_register_budget("increase", 232)
    # instruction_selection: `setmaxnreg.inc.sync.aligned.u32 232`; extent: warps 0..7.
barrier("cta")
# instruction_selection: `bar.sync 0`; extent: CTA.

# Warps 0 and 4: two 32-row score/probability stages.
if warp == 0 or warp == 4:
    stage = warp // 4
    p_store_phase = 1 if stage == 0 else 0
    s_phase[0] = s_phase[1] = 0
    for work in range(block_id_x(), total_q * Hkv, grid_dim_x()):
        row = lane
        state_row = stage * 128 + row
        row_max = -inf
        row_sum = 0.0
        for pair in serial(0, 8):
            wait(s_full[stage], s_phase[stage]); s_phase[stage] ^= 1
            # instruction_selection: retrying
            # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
            # extent: one stage generation.
            fence("tmem_after_thread_sync")
            # instruction_selection: `tcgen05.fence::after_thread_sync`; extent: one.
            valid = load_s2r(valid_count[pair * 2 + stage])
            # instruction_selection: `ld.shared.b32`; extent: one scalar.
            copy_t2r(S(stage, row, 0:128), scores[0:128])
            # instruction_selection: four
            # `tcgen05.ld.sync.aligned.32x32b.x32.b32`; extent: 128 FP32 scores.

            if valid < 128:
                for quarter in unroll(0, 4):
                    limit = clamp(valid - 32 * quarter, 0, 32)
                    bits = low_mask(limit)
                    for element in unroll(0, 32):
                        scores[32 * quarter + element] = select(
                            bits & (1 << element), scores[32 * quarter + element], -inf)
                        # instruction_selection: integer shift/add mask construction
                        # and predicated `mov.b32`; extent: four 32-value quarters.

            block_max = fill(-inf, lanes=2)
            for chunk in unroll(0, 4):
                for pair_value in unroll(0, 16):
                    block_max[pair_value & 1] = max(
                        block_max[pair_value & 1],
                        max(scores[32*chunk+2*pair_value],
                            scores[32*chunk+2*pair_value+1]))
            block_max = max(block_max[0], block_max[1])
            # instruction_selection: `max.f32`; extent: fixed 128-value tree.
            new_max = max(row_max, block_max)
            safe_max = select(new_max == -inf, 0.0, new_max)
            max_scaled = mul(safe_max, softmax_scale_log2)
            delta = fma(row_max, softmax_scale_log2, -max_scaled)
            acc_scale = select(row_max > -inf, exp2(delta), 1.0)
            # instruction_selection: `max.f32`, `selp.f32`, `mul.ftz.f32`,
            # `fma.rn.ftz.f32`, `ex2.approx.ftz.f32`, `selp.f32`;
            # extent: one online-statistic scalar.
            row_max = new_max
            store_r2s(acc_scale, acc_scale_smem[state_row])
            # instruction_selection: `st.shared.b32`; extent: one row scale.
            fence("async_shared")
            # instruction_selection: `fence.proxy.async.shared::cta`; extent: one.
            arrive(corr_sig[stage])
            # instruction_selection: `mbarrier.arrive.release.cta.shared::cta.b64`;
            # extent: 32 row arrivals.

            for packed_pair in unroll(0, 64):
                scores2[packed_pair] = fma(
                    scores2[packed_pair], spread(softmax_scale_log2),
                    spread(-max_scaled), lanes=2)
                # instruction_selection: `fma.rn.ftz.f32x2`; extent: 64 pairs.
            for element in unroll(0, 128):
                scores[element] = exp2(scores[element])
                # instruction_selection: `ex2.approx.ftz.f32`; extent: 128 values.
            block_sum2 = fill(0.0, lanes=2)
            for packed_pair in unroll(0, 64):
                block_sum2 = add(block_sum2, scores2[packed_pair], lanes=2)
                # instruction_selection: `add.f32x2`; extent: 64 packed pairs.
            block_sum = add(block_sum2.x, block_sum2.y)
            # instruction_selection: `add.ftz.f32`; extent: one scalar.

            if pair == 0:
                wait(p_store_turn[stage], p_store_phase)
                # instruction_selection: retrying
                # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
                # extent: first pair only.
            for group4 in unroll(0, 32):
                probabilities[group4] = cast(scores[4*group4:4*group4+4], "e4m3x4")
                # instruction_selection: two
                # `cvt.rn.satfinite.e4m3x2.f32` plus `mov.b32`;
                # extent: four values, 32 groups.
            copy_r2t(probabilities[0:64], P(stage, row, 0:64))
            copy_r2t(probabilities[64:128], P(stage, row, 64:128))
            # instruction_selection: two
            # `tcgen05.st.sync.aligned.32x32b.x16.b32`; extent: 128 E4M3.
            row_sum = fma(row_sum, acc_scale, block_sum)
            # instruction_selection: `fma.rn.ftz.f32`; extent: one scalar.
            fence("tmem_store_wait")
            # instruction_selection: `tcgen05.wait::st.sync.aligned`; extent: one.
            fence("tmem_before_thread_sync")
            # instruction_selection: `tcgen05.fence::before_thread_sync`; extent: one.
            if pair == 0:
                arrive(p_store_turn[1-stage])
                # instruction_selection:
                # `mbarrier.arrive.release.cta.shared::cta.b64`;
                # extent: 32 lane arrivals.
                p_store_phase ^= 1
            arrive(p_full[stage])
            # instruction_selection:
            # `mbarrier.arrive.release.cta.shared::cta.b64`;
            # extent: 32 of 64 arrivals.

        store_r2s(row_sum, row_sum_smem[state_row])
        store_r2s(row_max, row_max_smem[state_row])
        # instruction_selection: two `st.shared.b32`; extent: final row state.
        fence("async_shared")
        # instruction_selection: `fence.proxy.async.shared::cta`; extent: one.
        arrive(corr_sig[stage])
        # instruction_selection:
        # `mbarrier.arrive.release.cta.shared::cta.b64`; extent: 32 arrivals.

# Warp 2: elected Q/K/V producer.
if warp == 2:
    q_empty_phase = 1
    for work in range(block_id_x(), total_q * Hkv, grid_dim_x()):
        query = work // Hkv
        kv_head = work % Hkv
        group = Hq // Hkv
        wait(q_empty, q_empty_phase); q_empty_phase ^= 1
        # instruction_selection: retrying
        # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
        # extent: full producer warp, one Q-empty generation.
        if elect():
            # instruction_selection: `elect.sync`; extent: one elected lane in
            # the producer warp for Q production only.
            q_row = query * Hq + kv_head * group
            expect_tx(q_full, 2048)
            # instruction_selection:
            # `mbarrier.arrive.expect_tx.release.cta.shared::cta.b64`;
            # extent: one 2,048-byte transaction expectation.
            copy_g2s_tma(Qmap, (0, q_row, 0), Q_smem, q_full)
            # instruction_selection:
            # `cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes`;
            # extent: one 16x128-byte Q tile.
        if elect():
            # instruction_selection: independent second `elect.sync`; extent:
            # one elected lane in the producer warp for all K/V production.
            ring_stage = 0
            ring_phase = 1

            for tile in unroll(0, 2):
                selected_position = 15 - tile
                batch = query // seqlen_q
                query_in_batch = query - batch * seqlen_q
                logical_page = load_g2r(selected_blocks[kv_head, query, selected_position])
                kv_len = load_g2r(kv_lens[batch])
                # instruction_selection: two `ld.global.nc.b32`; extent: scalars.
                valid = 0
                if logical_page >= 0:
                    block_start = logical_page * 128
                    valid = clamp(kv_len - block_start, 0, 128)
                    query_position = kv_len - seqlen_q + query_in_batch
                    valid = clamp(min(valid, query_position - block_start + 1), 0, 128)
                    # instruction_selection: signed setp/branches, `max.s32`,
                    # `min.relu.s32`, and `min.s32`; extent: one count.
                physical_page = 0
                if logical_page >= 0:
                    physical_page = load_g2r(page_table[batch * max_pages + logical_page])
                    # instruction_selection: branch-guarded `ld.global.nc.b32`;
                    # extent: one page-table scalar.
                    if physical_page < 0:
                        valid = 0
                        physical_page = 0
                page_head = physical_page * Hkv + kv_head
                store_r2s(valid, valid_count[tile])
                store_r2s(page_head, page_head_smem[tile])
                # instruction_selection: two `st.shared.b32`; extent: metadata.
                fence("async_shared")
                # instruction_selection: `fence.proxy.async.shared::cta`;
                # extent: one metadata-publication fence.
                wait(kv_empty[ring_stage], ring_phase)
                # instruction_selection: retrying
                # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
                # extent: one K ring-slot generation.
                expect_tx(kv_full[ring_stage], 16384)
                # instruction_selection:
                # `mbarrier.arrive.expect_tx.release.cta.shared::cta.b64`;
                # extent: one 16,384-byte K transaction expectation.
                copy_g2s_tma(Kmap, (0, 0, page_head), ring(ring_stage), kv_full[ring_stage])
                # instruction_selection:
                # `cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes`;
                # extent: one rank-3 K tile.
                ring_stage, ring_phase = advance_ring(ring_stage, ring_phase)

            for next_tile in serial(2, 16):
                v_tile = next_tile - 2
                v_page_head = load_s2r(page_head_smem[v_tile])
                # instruction_selection: `ld.shared.b32`; extent: one scalar.
                wait(kv_empty[ring_stage], ring_phase)
                # instruction_selection: retrying
                # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
                # extent: one V ring-slot generation.
                expect_tx(kv_full[ring_stage], 16384)
                # instruction_selection:
                # `mbarrier.arrive.expect_tx.release.cta.shared::cta.b64`;
                # extent: one 16,384-byte V transaction expectation.
                copy_g2s_tma(Vmap, (0, 0, v_page_head), ring(ring_stage), kv_full[ring_stage])
                # instruction_selection:
                # `cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes`;
                # extent: one rolled rank-3 V tile.
                ring_stage, ring_phase = advance_ring(ring_stage, ring_phase)

                batch = query // seqlen_q
                query_in_batch = query - batch * seqlen_q
                logical_page = load_g2r(selected_blocks[kv_head, query, 15-next_tile])
                kv_len = load_g2r(kv_lens[batch])
                # instruction_selection: two `ld.global.nc.b32`; extent: scalars.
                valid = 0
                if logical_page >= 0:
                    block_start = logical_page * 128
                    valid = clamp(kv_len - block_start, 0, 128)
                    query_position = kv_len - seqlen_q + query_in_batch
                    valid = clamp(min(valid, query_position - block_start + 1), 0, 128)
                    # instruction_selection: signed setp/branches, `max.s32`,
                    # `min.relu.s32`, and `min.s32`; extent: one count.
                physical_page = 0
                if logical_page >= 0:
                    physical_page = load_g2r(page_table[batch * max_pages + logical_page])
                    # instruction_selection: branch-guarded `ld.global.nc.b32`;
                    # extent: one page-table scalar.
                    if physical_page < 0:
                        valid = 0
                        physical_page = 0
                page_head = physical_page * Hkv + kv_head
                store_r2s(valid, valid_count[next_tile])
                store_r2s(page_head, page_head_smem[next_tile])
                # instruction_selection: two `st.shared.b32`; extent: metadata.
                fence("async_shared")
                # instruction_selection: `fence.proxy.async.shared::cta`;
                # extent: one metadata-publication fence.
                wait(kv_empty[ring_stage], ring_phase)
                # instruction_selection: retrying
                # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
                # extent: one K ring-slot generation.
                expect_tx(kv_full[ring_stage], 16384)
                # instruction_selection:
                # `mbarrier.arrive.expect_tx.release.cta.shared::cta.b64`;
                # extent: one 16,384-byte K transaction expectation.
                copy_g2s_tma(Kmap, (0, 0, page_head), ring(ring_stage), kv_full[ring_stage])
                # instruction_selection:
                # `cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes`;
                # extent: one rolled rank-3 K tile.
                ring_stage, ring_phase = advance_ring(ring_stage, ring_phase)

            for v_tile in unroll(14, 16):
                page_head = load_s2r(page_head_smem[v_tile])
                wait(kv_empty[ring_stage], ring_phase)
                # instruction_selection: retrying
                # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
                # extent: one V ring-slot generation at each unrolled site.
                expect_tx(kv_full[ring_stage], 16384)
                # instruction_selection:
                # `mbarrier.arrive.expect_tx.release.cta.shared::cta.b64`;
                # extent: one 16,384-byte V transaction expectation per site.
                copy_g2s_tma(Vmap, (0, 0, page_head), ring(ring_stage), kv_full[ring_stage])
                # instruction_selection:
                # `cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes`;
                # extent: two statically unrolled rank-3 V tiles.
                ring_stage, ring_phase = advance_ring(ring_stage, ring_phase)

# Warp 3: sole tcgen05 QK/PV issuer.
if warp == 3:
    q_full_phase = p0_phase = p1_phase = decode_phase = 0
    for work in range(block_id_x(), total_q * Hkv, grid_dim_x()):
        ring_stage = ring_phase = 0
        first_pv0 = first_pv1 = True
        wait(q_full, q_full_phase); q_full_phase ^= 1
        # instruction_selection: retrying
        # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
        # extent: one Q generation.

        wait(kv_full[ring_stage], ring_phase)
        # instruction_selection: retrying
        # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
        # extent: one K ring-slot generation.
        gemm(S0, Q_smem, ring(ring_stage), accumulate=False,
             descriptor_hi=0x40004040, instruction_descriptor=136314896,
             low_word_steps=(0,2,4,6))
        # instruction_selection: four elected
        # `tcgen05.mma.cta_group::1.kind::f8f6f4`; extent: QK 32x128x128.
        commit(s_full[0]); commit(kv_empty[ring_stage]); ring_stage, ring_phase = advance_ring(ring_stage, ring_phase)
        # instruction_selection: two elected
        # `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
        # extent: S0 publication and K-ring release.

        wait(kv_full[ring_stage], ring_phase)
        # instruction_selection: retrying
        # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
        # extent: one K ring-slot generation.
        gemm(S1, Q_smem, ring(ring_stage), accumulate=False,
             descriptor_hi=0x40004040, instruction_descriptor=136314896,
             low_word_steps=(0,2,4,6))
        # instruction_selection: four elected
        # `tcgen05.mma.cta_group::1.kind::f8f6f4`; extent: QK 32x128x128.
        commit(s_full[1]); commit(kv_empty[ring_stage]); ring_stage, ring_phase = advance_ring(ring_stage, ring_phase)
        # instruction_selection: two elected
        # `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
        # extent: S1 publication and K-ring release.

        for pair in serial(0, 7):
            wait(kv_full[ring_stage], ring_phase)
            # instruction_selection: retrying
            # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
            # extent: one V ring-slot generation.
            wait(p_full[0], p0_phase); p0_phase ^= 1
            # instruction_selection: retrying
            # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
            # extent: one P0 generation.
            fence("tmem_after_thread_sync")
            # instruction_selection: `tcgen05.fence::after_thread_sync`;
            # extent: one PV0 ordering fence.
            gemm(O0, P0, ring(ring_stage), accumulate=not first_pv0,
                 descriptor_hi=0x40004040, smem_b_low_or=0x04000000,
                 instruction_descriptor=136380432,
                 tmem_a_steps=(0,8,16,24), smem_b_steps=(0,256,512,768))
            # instruction_selection: four elected
            # `tcgen05.mma.cta_group::1.kind::f8f6f4`, with A in TMEM;
            # extent: PV 32x128x128.
            commit(kv_empty[ring_stage]); ring_stage, ring_phase = advance_ring(ring_stage, ring_phase)
            # instruction_selection: one elected
            # `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
            # extent: V-ring release.

            wait(kv_full[ring_stage], ring_phase)
            # instruction_selection: retrying
            # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
            # extent: one K ring-slot generation.
            gemm(S0, Q_smem, ring(ring_stage), accumulate=False,
                 descriptor_hi=0x40004040, instruction_descriptor=136314896,
                 low_word_steps=(0,2,4,6))
            # instruction_selection: four elected
            # `tcgen05.mma.cta_group::1.kind::f8f6f4`; extent: QK 32x128x128.
            commit(s_full[0]); commit(kv_empty[ring_stage]); ring_stage, ring_phase = advance_ring(ring_stage, ring_phase)
            # instruction_selection: two elected
            # `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
            # extent: S0 publication and K-ring release.

            wait(kv_full[ring_stage], ring_phase)
            # instruction_selection: retrying
            # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
            # extent: one V ring-slot generation.
            wait(p_full[1], p1_phase); p1_phase ^= 1
            # instruction_selection: retrying
            # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
            # extent: one P1 generation.
            fence("tmem_after_thread_sync")
            # instruction_selection: `tcgen05.fence::after_thread_sync`;
            # extent: one PV1 ordering fence.
            gemm(O1, P1, ring(ring_stage), accumulate=not first_pv1,
                 descriptor_hi=0x40004040, smem_b_low_or=0x04000000,
                 instruction_descriptor=136380432,
                 tmem_a_steps=(0,8,16,24), smem_b_steps=(0,256,512,768))
            # instruction_selection: four elected
            # `tcgen05.mma.cta_group::1.kind::f8f6f4`; extent: PV 32x128x128.
            commit(kv_empty[ring_stage]); ring_stage, ring_phase = advance_ring(ring_stage, ring_phase)
            # instruction_selection: one elected
            # `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
            # extent: V-ring release.

            wait(kv_full[ring_stage], ring_phase)
            # instruction_selection: retrying
            # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
            # extent: one K ring-slot generation.
            gemm(S1, Q_smem, ring(ring_stage), accumulate=False,
                 descriptor_hi=0x40004040, instruction_descriptor=136314896,
                 low_word_steps=(0,2,4,6))
            # instruction_selection: four elected
            # `tcgen05.mma.cta_group::1.kind::f8f6f4`; extent: QK 32x128x128.
            if pair == 6:
                commit(s_full[1]); commit(q_empty)
                # instruction_selection: two elected
                # `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
                # extent: S1 publication and Q-ring release.
            else:
                commit(s_full[1])
                # instruction_selection: one elected
                # `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
                # extent: S1 publication.
            commit(kv_empty[ring_stage]); ring_stage, ring_phase = advance_ring(ring_stage, ring_phase)
            # instruction_selection: one elected
            # `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
            # extent: K-ring release.
            first_pv0 = first_pv1 = False

        wait(kv_full[ring_stage], ring_phase)
        # instruction_selection: retrying
        # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
        # extent: one V ring-slot generation.
        wait(p_full[0], p0_phase); p0_phase ^= 1
        # instruction_selection: retrying
        # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
        # extent: one P0 generation.
        fence("tmem_after_thread_sync")
        # instruction_selection: `tcgen05.fence::after_thread_sync`;
        # extent: one final-PV0 ordering fence.
        gemm(O0, P0, ring(ring_stage), accumulate=not first_pv0,
             descriptor_hi=0x40004040, smem_b_low_or=0x04000000,
             instruction_descriptor=136380432,
             tmem_a_steps=(0,8,16,24), smem_b_steps=(0,256,512,768))
        # instruction_selection: four elected
        # `tcgen05.mma.cta_group::1.kind::f8f6f4`; extent: final PV0 32x128x128.
        commit(kv_empty[ring_stage]); ring_stage, ring_phase = advance_ring(ring_stage, ring_phase)
        # instruction_selection: one elected
        # `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
        # extent: V-ring release.

        wait(kv_full[ring_stage], ring_phase)
        # instruction_selection: retrying
        # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
        # extent: one V ring-slot generation.
        wait(p_full[1], p1_phase); p1_phase ^= 1
        # instruction_selection: retrying
        # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
        # extent: one P1 generation.
        fence("tmem_after_thread_sync")
        # instruction_selection: `tcgen05.fence::after_thread_sync`;
        # extent: one final-PV1 ordering fence.
        gemm(O1, P1, ring(ring_stage), accumulate=not first_pv1,
             descriptor_hi=0x40004040, smem_b_low_or=0x04000000,
             instruction_descriptor=136380432,
             tmem_a_steps=(0,8,16,24), smem_b_steps=(0,256,512,768))
        # instruction_selection: four elected
        # `tcgen05.mma.cta_group::1.kind::f8f6f4`; extent: final PV1 32x128x128.
        commit(kv_empty[ring_stage])
        # instruction_selection: one elected
        # `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
        # extent: final V-ring release.
        ring_stage, ring_phase = advance_ring(ring_stage, ring_phase)
        commit(o_full)
        # instruction_selection: one elected
        # `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
        # extent: output publication after the final ring cursor advance.
        wait(decode_done, decode_phase); decode_phase ^= 1
        # instruction_selection: retrying
        # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
        # extent: one generation with 32 correction arrivals.

# Warp 8: correction and epilogue.
if warp == 8:
    corr0_phase = corr1_phase = o_phase = 0
    for work in range(block_id_x(), total_q * Hkv, grid_dim_x()):
        query = work // Hkv
        kv_head = work % Hkv
        group = Hq // Hkv
        warp_in_role = warp - 8
        tmem_row_base = warp_in_role * 32
        row = tmem_row_base + lane
        row_bits = tmem_row_base << 16
        for pair in serial(0, 8):
            wait(corr_sig[0], corr0_phase); corr0_phase ^= 1
            # instruction_selection: retrying
            # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
            # extent: one correction-scale generation for stage 0.
            fence("tmem_after_thread_sync")
            # instruction_selection: `tcgen05.fence::after_thread_sync`;
            # extent: one O0 correction fence.
            scale0 = load_s2r(acc_scale_smem[row])
            # instruction_selection: `ld.shared.b32`; extent: one scale.
            if pair > 0:
                for col in unroll(0, 128, step=16):
                    copy_t2r(tmem_address(O0, row_bits, col:col+16), values0)
                    # instruction_selection:
                    # `tcgen05.ld.sync.aligned.32x32b.x16.b32`; extent: 16 FP32.
                    for packed_pair in unroll(0, 8):
                        values0_2[packed_pair] = mul(
                            values0_2[packed_pair], spread(scale0), lanes=2)
                        # instruction_selection: `mul.rn.ftz.f32x2`; extent: 8 pairs.
                    copy_r2t(values0, tmem_address(O0, row_bits, col:col+16))
                    # instruction_selection:
                    # `tcgen05.st.sync.aligned.32x32b.x16.b32`; extent: 16 FP32.
                fence("tmem_store_wait")
                # instruction_selection: `tcgen05.wait::st.sync.aligned`.
                # extent: all O0 stores from this correction step.
            arrive(p_full[0])
            # instruction_selection:
            # `mbarrier.arrive.release.cta.shared::cta.b64`; extent: 32 arrivals.

            wait(corr_sig[1], corr1_phase); corr1_phase ^= 1
            # instruction_selection: retrying
            # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
            # extent: one correction-scale generation for stage 1.
            fence("tmem_after_thread_sync")
            # instruction_selection: `tcgen05.fence::after_thread_sync`;
            # extent: one O1 correction fence.
            scale1 = load_s2r(acc_scale_smem[128 + row])
            # instruction_selection: `ld.shared.b32`; extent: one scale.
            if pair > 0:
                for col in unroll(0, 128, step=16):
                    copy_t2r(tmem_address(O1, row_bits, col:col+16), values1)
                    # instruction_selection:
                    # `tcgen05.ld.sync.aligned.32x32b.x16.b32`; extent: 16 FP32.
                    for packed_pair in unroll(0, 8):
                        values1_2[packed_pair] = mul(
                            values1_2[packed_pair], spread(scale1), lanes=2)
                        # instruction_selection: `mul.rn.ftz.f32x2`; extent: 8 pairs.
                    copy_r2t(values1, tmem_address(O1, row_bits, col:col+16))
                    # instruction_selection:
                    # `tcgen05.st.sync.aligned.32x32b.x16.b32`; extent: 16 FP32.
                fence("tmem_store_wait")
                # instruction_selection: `tcgen05.wait::st.sync.aligned`;
                # extent: all O1 stores from this correction step.
            arrive(p_full[1])
            # instruction_selection:
            # `mbarrier.arrive.release.cta.shared::cta.b64`; extent: 32 arrivals.

        wait(o_full, o_phase); o_phase ^= 1
        # instruction_selection: retrying
        # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
        # extent: one output generation.
        fence("tmem_after_thread_sync")
        # instruction_selection: `tcgen05.fence::after_thread_sync`;
        # extent: one output-visibility fence.
        wait(corr_sig[0], corr0_phase); corr0_phase ^= 1
        # instruction_selection: retrying
        # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
        # extent: final stage-0 statistic generation.
        wait(corr_sig[1], corr1_phase); corr1_phase ^= 1
        # instruction_selection: retrying
        # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`;
        # extent: final stage-1 statistic generation.
        fence("tmem_after_thread_sync")
        # instruction_selection: `tcgen05.fence::after_thread_sync`;
        # extent: one final-statistic visibility fence.
        sum0 = load_s2r(row_sum_smem[row])
        sum1 = load_s2r(row_sum_smem[128+row])
        max0 = load_s2r(row_max_smem[row])
        max1 = load_s2r(row_max_smem[128+row])
        # instruction_selection: four `ld.shared.b32`; extent: final state.
        final_max = max(max0, max1)
        d0 = select(max0 == -inf, 0.0, softmax_scale_log2 * (max0-final_max))
        d1 = select(max1 == -inf, 0.0, softmax_scale_log2 * (max1-final_max))
        merge0 = exp2(d0); merge1 = exp2(d1)
        final_sum = add(mul(sum0, merge0), mul(sum1, merge1))
        inv_sum = select(final_sum > 0.0, rcp(final_sum), 0.0)
        # instruction_selection: `max.f32`, scalar sub/mul, two
        # `ex2.approx.ftz.f32`, scalar mul/FMA, `rcp.approx.ftz.f32`,
        # and `selp.f32`; extent: one row.

        for col in unroll(0, 128, step=16):
            copy_t2r(tmem_address(O0, row_bits, col:col+16), values0)
            copy_t2r(tmem_address(O1, row_bits, col:col+16), values1)
            # instruction_selection: two
            # `tcgen05.ld.sync.aligned.32x32b.x16.b32`;
            # extent: 16 FP32 from each of O0 and O1.
            for element in unroll(0, 16):
                merged[element] = fma(
                    values0[element], merge0, mul(values1[element], merge1))
                # instruction_selection: `mul.ftz.f32` then `fma.rn.ftz.f32`;
                # extent: 16 scalars in source order.
            if row < group:
                for packed_pair in unroll(0, 8):
                    merged2[packed_pair] = mul(
                        merged2[packed_pair], spread(inv_sum * output_scale), lanes=2)
                    # instruction_selection: `mul.rn.ftz.f32x2`; extent: 8 pairs.
                packed_bf16 = cast(merged, "bf16x16")
                # instruction_selection: eight `cvt.rn.bf16x2.f32`.
                output_row = query * Hq + kv_head * group + row
                copy_r2g(packed_bf16[0:8], O[output_row, col:col+8])
                copy_r2g(packed_bf16[8:16], O[output_row, col+8:col+16])
                # instruction_selection: two `st.global.v4.b32`; extent: 16 BF16.

        if row < group:
            log_sum = log2(final_sum)
            lse_value = select(
                final_sum > 0.0,
                fma(final_max * softmax_scale_log2, LN2, log_sum * LN2),
                -inf)
            # instruction_selection: `lg2.approx.ftz.f32`, scalar mul/FMA,
            # and `selp.f32`; extent: one natural-log LSE.
            copy_r2g(lse_value, LSE[query * Hq + kv_head * group + row])
            # instruction_selection: `st.global.b32`; extent: one FP32.
        arrive(decode_done)
        # instruction_selection:
        # `mbarrier.arrive.release.cta.shared::cta.b64`; extent: 32 arrivals.

barrier("cta")
# instruction_selection: `bar.sync 0`; extent: CTA before TMEM teardown.
if warp == 0:
    free(tmem_base, 512)
    # instruction_selection: `tcgen05.dealloc.cta_group::1.sync.aligned.b32`;
    # extent: warp 0.
```

## Logical GEMM ownership

| GEMM | logical shape | A | B | accumulator | owner / consumer |
| --- | --- | --- | --- | --- | --- |
| QK stage 0/1 | 32 x 128 x 128 | FP8 Q SMEM, K-major | FP8 K ring SMEM, K-major | FP32 S0/S1 TMEM | warp 3 / warp 0 or 4 |
| PV stage 0/1 | 32 x 128 x 128 | E4M3 P in upper S0/S1 TMEM columns | FP8 V ring SMEM, MN-major via LBO descriptor bit | FP32 O0/O1 TMEM | warp 3 / warp 8 |

Both are `tcgen05.mma.cta_group::1.kind::f8f6f4`. QK uses instruction
descriptor 136314896 and low-word steps `+0,+2,+4,+6`; PV uses 136380432,
TMEM A steps `+0,+8,+16,+24`, and SMEM B steps `+0,+256,+512,+768` with the
16 KiB leading-byte-offset bit. The optimized entry has 32 static MMA sites;
rolled loops make the dynamic count larger.

## Bidirectional source / sketch / PTX map

| source region | sketch region | optimized PTX evidence |
| --- | --- | --- |
| 542-658 | resources and prologue | `.loc 542-658`; 24 barrier inits, TMEM alloc, two CTA barriers around `setmaxnreg` |
| 660-1257 | warps 0/4 | four x32 score loads, 131 static `max.f32`, 128 probability `ex2`, 64 FP8x2 conversions, two x16 P stores |
| 1262-1414 | warp 2 | `.loc 1272-1409`; `ld.global.nc.b32`, metadata stores, seven static rank-3 TMA sites |
| 1418-1752 | warp 3 | `.loc 1469/1509/1545/1587/1621/1662/1703/1738`; 32 static f8f6f4 MMA and 15 commit sites |
| 1755-1874 | warp 8 | `.loc 1771-1873`; 32 x16 TMEM loads, 16 x16 TMEM stores, 16 vector output stores, one rcp and one log2 |
| 1879-1884 | teardown | final `bar.sync 0`, `.loc 1882` TMEM deallocation |

## TIRx, correctness, and benchmark contract

- The executable module imports the device language only as
  `import tirx_kernels.kern as K` and uses rank-one SMEM plus explicit scalar
  offset functions. It may use existing `K.ptx[...]` instruction families; it
  must not modify `tirx_kernels/kern/`.
- `get_kernel`, `prepare_data`, `run_test`, `prepare_bench`, `run_gpu`,
  and `run_bench` are public. Source dependencies load lazily.
- Correctness first compares source and TIRx on identical deterministic FP8
  inputs. It checks complete overwrite/canaries, finite-state rules, repeated
  determinism, public-route coverage, partial pages, causal truncation,
  negative selected slots/physical pages, page permutation, GQA 1/4/16, scale
  folding, Q=1/3/4/7/16/32, and both 148-CTA and even-wave 128-CTA dispatch.
  Bitwise equality is attempted first; any fallback tolerance must be derived
  from observed BF16 output ULPs and FP32 LSE error and be tighter than the
  source repository's generic attention tolerance.
- The final performance matrix is B=128, KV=4096, Hq=64, Hkv=4, Q=1..16 plus
  B=16,Q=4 for the even-wave boundary. `prepare_bench` compiles, allocates,
  and validates outside timing. A timed closure performs exactly one kernel
  launch. Only `python -m tirx_kernels.bench_suite` is authoritative, and every
  row must satisfy `source_time / tirx_time > 0.99`.
- On GB200, source and TIRx must both declare PTX 9.2 and target `sm_100a` under
  the same CUDA toolchain.

## Instruction-selection summary

| decision | emitted consequence in the writer PTX |
| --- | --- |
| by-value Q/K/V maps and fixed CTA | three 128-byte params, each `.align 64`, `.maxntid 384`, no cluster attribute |
| rank-one 156672-byte arena | `.extern .shared .align 1024`; all live addresses are explicit byte offsets |
| one elected producer | seven static rank-3 TMA sites paired with seven expect-tx arrivals |
| four-slot K/V alias | one SMEM descriptor base plus 16384-byte stage stride; K-major QK and LBO-marked MN-major PV descriptors |
| two score roles | four x32 TMEM loads in one rolled body, scalar masking/max tree, 128 MUFU exp2, 64 packed adds, 64 E4M3x2 conversions, two x16 TMEM stores |
| sole MMA warp | 32 static f8f6f4 MMA and 15 static tcgen05 commit sites; all ring releases are ordered behind tcgen05 |
| correction loop | x16 TMEM load/mul/store with explicit TMEM store wait; final merge is scalar mul then FMA, normalization uses packed multiply |
| BF16 epilogue | eight BF16x2 conversions and two 128-bit stores per 16-column chunk; one approximate reciprocal and one approximate log2 per row |
