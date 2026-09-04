<!--
This file is a design sketch for a TIRx port of code from FlashInfer
(https://github.com/flashinfer-ai/flashinfer @ cc6e8794c49bf66172627bdb9742fcb17d18b839),
Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# FlashInfer SM103a M64 BF16 GQA16 flat MSA: coarse WASP pipeline sketch

This is a non-executable operation-level sketch for
[`tirx_kernels/flashinfer/msa_ops/blackwell_msa_prefill_m64_bf16_gqa16_flat_sm103.py`](../../tirx_kernels/flashinfer/msa_ops/blackwell_msa_prefill_m64_bf16_gqa16_flat_sm103.py),
which becomes the executable source of truth. It freezes the exact generated
CUDA specialization
`sm103a/blackwell_msa_prefill_m64_bf16_gqa16_flat.cu::kernel_minimax_sparse_prefill_exact_union_m64_sm100`
at FlashInfer commit `cc6e8794c49bf66172627bdb9742fcb17d18b839`.
The source file SHA-256 is
`c635c319626d23e8e3936c177ca610e4f2eb840296f3368dd42d7b72dc501025`.

The writer export is
`.porting/blackwell_msa_prefill_m64_bf16_gqa16_flat_sm103/source_export/writer/kernel.ptx`,
SHA-256 `12c49a1faaf81e89c46718f5b8d60473e6da96eecfdb83551a23fe4adf607472`.
CUDA 13.3.73 emitted PTX ISA 9.3 for `sm_103a`; the entry has 2,320 `.loc`
records, `.maxntid 512`, `.minnctapersm 1`, and a dynamic shared declaration
aligned to 1,024 bytes. All instruction selections below are derived from that
export.

After the independent sketch reviewer returns PASS, this file is immutable.

## Scope and invariants

The fixed specialization is flat BF16 Q/K/V, head dimension 128, KV block size
128, GQA ratio 16, TopK 16, at most 64 selectable KV blocks (KV length at most
8,192), eight queries per CTA, two four-query instances, one CTA per work item,
and `sm_103a`. Both causal and non-causal operation, ragged batches, explicit or
derived query offsets, optional softmax LSE, and optional temperature-scaled LSE
are in scope. Other dtypes, head dimensions, GQA ratios, TopK values, paged KV,
and other MSA schedules are out of scope.

The executable module imports the device language only as
`import tirx_kernels.kern as K`. It may use scalar K control flow, opaque
TensorMaps, one-dimensional shared/register allocations, and raw `K.ptx[...]`
instructions. It uses no tile primitive, first-class layout or mapping object,
`layout=` argument, direct TVM script namespace, `K.cuda.func_call`, inline-CUDA
function-call exemption, or modification under `tirx_kernels/kern/`.

## Pipeline at a glance

| warp role | tile program | publication / reuse edges |
| --- | --- | --- |
| warp 0 | elected lane initializes 19 mbarriers; full warp allocates and relinquishes 512 TMEM columns; also belongs to softmax instance 0 | init fence and warp/CTA rendezvous; thereafter follows the softmax-instance-0 edges below |
| warps 0-3 | softmax instance 0; each warp owns 16 logical rows and two lanes jointly own one 128-column score row | `union_ready -> s_full[0] -> corr_sig[0] -> p_full[0] -> corr_done[0]`; publishes final row sum/max/temperature sum to shared memory |
| warps 4-7 | same program for softmax instance 1 | corresponding `[1]` barriers and TMEM region offset 256 |
| warps 8-11 | correction plus epilogue; each warp owns 16 logical rows across both instances | seeds `p_full/corr_done`; conditionally rescales accumulated O in TMEM after `corr_sig`; publishes `corr_done`; after `o_full`, normalizes and stores output/LSE |
| warp 12 | sole MMA warp after a 48-register decrease; interleaves instance-0/1 QK and PV over the shared three-slot K/V ring | consumes `q_full`, `kv_full`, and `p_full`; commits `s_full`, `o_full`, and `kv_empty`; waits final deallocation then deallocates TMEM |
| warps 13-14 | idle after the 48-register decrease | no tile work |
| warp 15 | elected TMA issuer plus mask/union builder; loads Q, then for each union ordinal publishes live K0/K1 followed by live V0/V1 through the aliased ring | publishes `q_full`, `union_ready`, `kv_full`; consumes `kv_empty` |

## Primitive vocabulary

All physical storage is rank one. Logical dimensions and swizzles are scalar
index functions, not layout values:

```python
linear_smem(name, bytes, alignment)
reg_array(name, dtype, elements)
smem_byte(region, stage, scalar_index)
tmem_column(base, instance, kind, column, row_band)
tensor_map(name, rank, extents, byte_strides, box, dtype, swizzle_code)
smem_descriptor(high_word, shared_address_16B_units)
```

Movement and computation stay primitive:

```python
copy_g2s(tensor_map, coordinates, smem_byte, completion_barrier)
copy_t2r(tmem_column, registers)
copy_r2t(registers, tmem_column)
copy_r2g(registers, global_pointer, predicate)
load_smem(smem_byte, scalar)
store_smem(scalar, smem_byte)
gemm_ss(tmem_dst, smem_a_desc, smem_b_desc, instruction_desc, accumulate)
gemm_ts(tmem_dst, tmem_a, smem_b_desc, instruction_desc, accumulate)
fill(registers, value, predicate)
cast(register_pairs, dtype, rounding)
exp(registers); add(dst, lhs, rhs); mul(dst, lhs, rhs); fma(dst, a, b, c)
```

Barrier initialization, waits, arrivals, commits, fences, register
redistribution, and phase/stage updates are explicit schedule operations.

## Complete sketch

```python
# 1. Fixed launch and runtime ABI.
@kernel(
    target="sm_103a",
    grid=(ceil_div(total_q, 8) + batch_size - 1, num_kv_heads, 1),
    block=(512, 1, 1),
    min_blocks_per_sm=1,
    dynamic_smem_bytes=134784,
)
# instruction_selection: `.version 9.3`, `.target sm_103a`, `.maxntid 512`,
# `.minnctapersm 1`, `.extern .shared .align 1024`; extent: one entry.
def blackwell_msa_prefill_m64_bf16_gqa16_flat_sm103(
    q_map, k_map, v_map,                 # by-value 128-byte TensorMaps
    out_bf16, lse_f32, temperature_lse_f32,
    q2k_indices_i32, cu_q_i32, cu_k_i32, q_offsets_i32, kv_lens_i32,
    total_q, num_q_heads, num_kv_heads, topk, batch_size, uniform_q_len,
    causal, derive_q_offset, softmax_scale_log2, lse_temperature_scale,
    return_softmax_lse, return_temperature_lse,
):
    assert topk == 16 and num_q_heads == 16*num_kv_heads

    # Fastest-first map dimensions, exactly as encoded by the binding.
    q_map = tensor_map(
        q, rank=4, extents=(64, num_q_heads, total_q, 2),
        byte_strides=(256, num_q_heads*256, 128),
        box=(64, 16, 4, 2), element_strides=(1,1,1,1), dtype=BF16,
        interleave=NONE, swizzle_code=SW128B, l2_promotion=NONE, oob_fill=NONE,
    )
    k_map = tensor_map(
        k, rank=4, extents=(64, total_kv, 2, num_kv_heads),
        byte_strides=(num_kv_heads*256, 128, 256),
        box=(64, 64, 1, 1), element_strides=(1,1,1,1), dtype=BF16,
        interleave=NONE, swizzle_code=SW128B, l2_promotion=NONE, oob_fill=NONE,
    )
    v_map = tensor_map(
        v, rank=4, extents=(64, total_kv, 2, num_kv_heads),
        byte_strides=(num_kv_heads*256, 128, 256),
        box=(64, 64, 1, 1), element_strides=(1,1,1,1), dtype=BF16,
        interleave=NONE, swizzle_code=SW128B, l2_promotion=NONE, oob_fill=NONE,
    )

    tid, warp, lane = thread_id(), warp_uniform(thread_id() // 32), thread_id() % 32
    linear_tile, kv_head = block_id_x(), block_id_y()

    # 2. Exact one-dimensional shared arena and TMEM partition.
    smem = linear_smem("smem", 134784, alignment=1024)
    Q_FULL = 0; UNION_READY = 8
    KV_FULL = (16, 24, 32); KV_EMPTY = (40, 48, 56)
    S_FULL = (64, 72); P_FULL = (80, 88)
    CORR_SIG = (96, 104); CORR_DONE = (112, 120)
    O_FULL = (128, 136); TMEM_DEALLOC = 144; TMEM_MAILBOX = 152
    Q0 = 1024; Q1 = 17408
    KV = 33792                       # three slots, 32768 bytes each
    V = KV                           # exact source alias, reused per released stage
    SCALE = 132096                   # 512 f32 values
    MASK_LOW = 134144; MASK_HIGH = 134176
    UNION_COUNT = 134208             # two i32 values
    UNION_BLOCKS = 134216            # two arrays of 64 i32 values

    def kv_slot(stage): return KV + stage*32768
    def scale_index(kind, instance, row):
        # kind 0/1/2/3 = acc_scale/row_sum/row_max/temperature_sum.
        return SCALE + 4*(kind*128 + instance*64 + row)

    # 512 TMEM columns: scores0 [0,128), output0 [128,256),
    # scores1 [256,384), output1 [384,512).
    SCORES = (0, 256); OUTPUT = (128, 384)

    if warp == 0 and elect_one():
        init(Q_FULL, 1); init(UNION_READY, 32)
        for s in static_range(3): init(KV_FULL[s], 1)
        for s in static_range(3): init(KV_EMPTY[s], 1)
        for i in static_range(2): init(S_FULL[i], 1)
        for i in static_range(2): init(P_FULL[i], 256)
        for i in static_range(2): init(CORR_SIG[i], 128)
        for i in static_range(2): init(CORR_DONE[i], 128)
        for i in static_range(2): init(O_FULL[i], 1)
        init(TMEM_DEALLOC, 128)
        fence_mbarrier_init_release_cluster()
    # instruction_selection: 19 `mbarrier.init.shared::cta.b64` and one
    # `fence.mbarrier_init.release.cluster`; extent: elected warp-0 lane.

    warp_sync()
    # instruction_selection: one `bar.warp.sync -1`; extent: all sixteen warps.
    if warp == 0:
        allocate_tmem(TMEM_MAILBOX, columns=512)
        relinquish_tmem()
    # instruction_selection: `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32`
    # and `tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned`; extent: full warp 0.
    cta_sync(); fence_tmem_after_thread_sync()
    # instruction_selection: `bar.sync 0` then `tcgen05.fence::after_thread_sync`.
    taddr = load_smem_volatile_u32(TMEM_MAILBOX)
    # instruction_selection: `ld.volatile.shared.b32`; extent: one scalar/thread.

    if 12 <= warp <= 15: setmaxnreg_decrease(48)
    # instruction_selection: `setmaxnreg.dec.sync.aligned.u32 48`; extent: warps 12..15.

    # Scalar metadata/control helper; its body is expanded here and invoked
    # independently inside the softmax, correction, and loader branches.
    def decode_tile_metadata():
        batch, q_tile, tile_active = decode_linear_q_tile(
            linear_tile, uniform_q_len, cu_q_i32, batch_size, tile_extent=8)
        q_begin = load_global(cu_q_i32[batch])
        q_len = load_global(cu_q_i32[batch+1]) - q_begin
        q_local_base = 8*q_tile
        q_valid = clamp(q_len-q_local_base, 0, 8)*tile_active
        query_base = q_begin + q_local_base
        k_start = load_global(cu_k_i32[batch])
        kv_len = load_global(cu_k_i32[batch+1]) - k_start
        query_offset = select(derive_q_offset, kv_len-q_len,
                              load_global(q_offsets_i32[batch]))
        visible_blocks = ceil_div(query_offset+q_local_base+q_valid, 128)
        num_n_blocks = clamp(min(ceil_div(kv_len,128), visible_blocks) if causal
                             else ceil_div(kv_len,128), 0, 64)
        return (batch, q_tile, q_valid, query_base, q_local_base, k_start,
                kv_len, query_offset, num_n_blocks)
    # instruction_selection: scalar `ld.global.nc.b32` sequence metadata and
    # rolled scalar integer control; extent: one independent decode in each caller.

    # 3. Source-order role: warps 0..7, online softmax and P publication.
    with role(warp <= 7):
        setmaxnreg_increase(192)
        # instruction_selection: `setmaxnreg.inc.sync.aligned.u32 192`; extent: warps 0..7.
        (batch, q_tile, q_valid, query_base, q_local_base, k_start,
         kv_len, query_offset, num_n_blocks) = decode_tile_metadata()
        wait(UNION_READY, 0)
        instance = warp // 4; warp_in_instance = warp % 4
        logical_row = 16*warp_in_instance + lane % 16
        query_in_instance = logical_row // 16; col_half = lane // 16
        row_valid = query_in_instance < clamp(q_valid-4*instance, 0, 4)
        row_mask = (load_smem(MASK_LOW+4*(4*instance+query_in_instance)),
                    load_smem(MASK_HIGH+4*(4*instance+query_in_instance)))
        row_max, row_sum, temperature_sum = -inf, 0.0, 0.0
        s_phase = 0; corr_phase = 0
        for union_index in rolled_range(64):
            if union_index < UNION_COUNT[instance]:
                block = UNION_BLOCKS[64*instance+union_index]
                wait(S_FULL[instance], s_phase); s_phase ^= 1
                scores = reg_array("scores", f32, 64)
                copy_t2r(tmem_column(taddr, instance, "scores", 0, 32*warp_in_instance),
                          scores[0:32])
                copy_t2r(tmem_column(taddr, instance, "scores", 32, 32*warp_in_instance),
                          scores[32:64])
                # instruction_selection: two
                # `tcgen05.ld.sync.aligned.16x32bx2.x32.b32`; extent: one
                # lane-owned 64-column half-row.
                valid_cols = selected_and_causal_extent(row_valid, row_mask, block,
                                                        kv_len, query_offset,
                                                        q_local_base, instance)
                half_valid = clamp(valid_cols-col_half*64, 0, 64)
                fill(scores[half_valid:], -inf,
                     predicate=(valid_cols > 0 and half_valid < 64))
                # instruction_selection: scalar mask construction followed by
                # per-register `and.b32`, `setp.eq|ne.b32`, and `selp.b32`;
                # extent: 64 dynamic register selections, semantically replacing
                # only the invalid suffix.
                reduced_max = max_pairwise(scores)
                tile_max = select(half_valid <= 0, -inf, reduced_max)
                # instruction_selection: integer `setp.lt.s32` plus `selp.f32`
                # replaces the reduced maximum by `-inf` for a zero-valid half.
                tile_max = shuffle_xor_max(tile_max, 16)
                new_max = max(row_max, tile_max)
                acc_scale_log2 = fma(row_max, softmax_scale_log2,
                                     -safe(new_max)*softmax_scale_log2)
                if acc_scale_log2 >= -8:
                    selected_max, acc_scale, temperature_acc_scale = row_max, 1.0, 1.0
                else:
                    selected_max = new_max
                    acc_scale_candidate = exp(acc_scale_log2)
                    acc_scale = select(row_max != -inf, acc_scale_candidate, 1.0)
                    temperature_acc_scale_candidate = exp(
                        acc_scale_log2*lse_temperature_scale)
                    temperature_acc_scale = select(
                        row_max != -inf, temperature_acc_scale_candidate, 1.0)
                # instruction_selection: `max.f32`, `shfl.sync.bfly.b32`,
                # `fma.rn.f32`, two unconditional `ex2.approx.ftz.f32` candidates
                # inside the `< -8` branch, and two predicate-controlled
                # `selp.f32` candidate-versus-1.0 selections; extent: one update.
                row_max = selected_max
                if col_half == 0: store_smem(acc_scale, scale_index(0,instance,logical_row))
                fence_async_shared(); arrive(CORR_SIG[instance])
                # instruction_selection: `st.shared.b32`, async shared fence,
                # and `mbarrier.arrive.release.cta.shared::cta.b64`.

                bias = -safe(row_max)*softmax_scale_log2 if valid_cols > 0 else -inf
                block_temperature_sum = 0.0
                if return_temperature_lse:
                    fma(scores, softmax_scale_log2*lse_temperature_scale,
                        bias*lse_temperature_scale)
                    exp(scores); block_temperature_sum = reduce_sum_pair(scores, xor_lane=16)
                    copy_t2r(the_same_score_tmem_half_row, scores)
                    # instruction_selection: 32 packed `fma.rn.ftz.f32x2`,
                    fill(scores[half_valid:], -inf,
                         predicate=(valid_cols > 0 and half_valid < 64))
                    # 64 `ex2.approx.ftz.f32`, one shuffle-xor, then two more
                    # x32 TMEM loads to recover the unmodified scores, followed
                    # by the same mask-building plus 64 per-register
                    # `and.b32`/`setp.eq|ne.b32`/`selp.b32` dynamic selections,
                    # semantically replacing only the invalid suffix.
                fma(scores, softmax_scale_log2, bias); exp(scores)
                block_sum = reduce_sum_pair(scores, xor_lane=16)
                cast(probability_words, scores, BF16x2, rounding="rn")
                copy_r2t(probability_words,
                         tmem_column(taddr, instance, "scores", 64, 32*warp_in_instance))
                # instruction_selection: packed FMA, scalar `ex2.approx.ftz.f32`,
                # `cvt.rn.bf16x2.f32`, and two
                # `tcgen05.st.sync.aligned.16x32bx2.x16.b32`; extent: 128 BF16 P values.
                wait_tmem_stores(); arrive(P_FULL[instance])
                wait(CORR_DONE[instance], corr_phase); corr_phase ^= 1
                # instruction_selection: `tcgen05.wait::st.sync.aligned`, then full-warp mbarrier arrival and acquire parity wait.
                row_sum = fma(row_sum, acc_scale, block_sum)
                temperature_candidate = fma(temperature_sum,
                                            temperature_acc_scale,
                                            block_temperature_sum)
                temperature_sum = select(return_temperature_lse,
                                         temperature_candidate, temperature_sum)
                # instruction_selection: two scalar `fma.rn.f32`; the second
                # result is followed by a return-flag-controlled `selp.f32`.

        if col_half == 0:
            store_smem(row_sum, scale_index(1,instance,logical_row))
            store_smem(row_max, scale_index(2,instance,logical_row))
            store_smem(temperature_sum, scale_index(3,instance,logical_row))
        fence_async_shared(); arrive(CORR_SIG[instance])
        # instruction_selection: three scalar shared stores, fence, and final arrival.


    # 4. Source-order role: warps 8..11, correction and final output.
    with role(8 <= warp <= 11):
        setmaxnreg_decrease(80)
        # instruction_selection: `setmaxnreg.dec.sync.aligned.u32 80`; extent: warps 8..11.
        (batch, q_tile, q_valid, query_base, q_local_base, k_start,
         kv_len, query_offset, num_n_blocks) = decode_tile_metadata()
        wait(UNION_READY, 0)
        logical_row = 16*(warp-8) + lane % 16; col_half = lane // 16
        arrive(P_FULL[0]); arrive(P_FULL[1])
        wait(CORR_SIG[0], 0); arrive(CORR_DONE[0])
        wait(CORR_SIG[1], 0); arrive(CORR_DONE[1])
        # instruction_selection: source seeds both 256-arrival P barriers and
        # both correction barriers with full-warp arrivals before union index 1.
        corr_phase = [1, 1]
        for union_index in rolled_range(1, 64):
            for instance in static_range(2):
                if union_index < UNION_COUNT[instance]:
                    wait(CORR_SIG[instance], corr_phase[instance]); corr_phase[instance] ^= 1
                    acc_scale = load_smem(scale_index(0,instance,logical_row))
                    if warp_any(acc_scale < 1.0):
                        accum = reg_array("accum", f32, 64)
                        copy_t2r(output_half_row(instance, logical_row, col_half), accum)
                        mul(accum, accum, acc_scale)
                        copy_r2t(accum, output_half_row(instance, logical_row, col_half))
                        wait_tmem_stores()
                    # instruction_selection: `vote.sync.any.pred`, two x32
                    # TMEM loads, 32 `mul.rn.ftz.f32x2`, one x64 TMEM store,
                    # and `tcgen05.wait::st.sync.aligned` when correction is live.
                    arrive(P_FULL[instance]); arrive(CORR_DONE[instance])
                    # instruction_selection: two full-warp mbarrier arrivals.

        wait(O_FULL[0],0); wait(O_FULL[1],0)
        wait(CORR_SIG[0],corr_phase[0]); wait(CORR_SIG[1],corr_phase[1])
        fence_tmem_after_thread_sync()
        # instruction_selection: four acquire waits then
        # `tcgen05.fence::after_thread_sync`.
        for instance in static_range(2):
            final_sum = load_smem(scale_index(1,instance,logical_row))
            final_max = load_smem(scale_index(2,instance,logical_row))
            final_temperature_sum = load_smem(scale_index(3,instance,logical_row))
            reciprocal = rcp(final_sum)
            inv_sum = select((final_sum > 0.0) and not_nan(final_sum), reciprocal, 0.0)
            output_values = reg_array("output_values", f32, 64)
            copy_t2r(output_half_row(instance, logical_row, col_half), output_values)
            # instruction_selection: unconditional `rcp.approx.ftz.f32`, then
            # predicate/select to zero, and two x32 TMEM loads.
            if query_row_is_valid(instance, logical_row):
                for vector in static_range(8):
                    mul(output_values[8*vector:8*vector+8], inv_sum)
                    cast(out_words, output_values[8*vector:8*vector+8], BF16x2, rounding="rn")
                    copy_r2g(out_words, out_bf16[query, head, 64*col_half+8*vector], True)
                # instruction_selection: packed `mul.rn.ftz.f32x2`,
                # `cvt.rn.bf16x2.f32`, and eight `st.global.v4.b32` per live half-row.
                if col_half == 0 and return_softmax_lse:
                    log_sum = lg2(final_sum)
                    stat = select(final_sum > 0.0,
                                  fma(final_max*softmax_scale_log2, ln2, log_sum*ln2), -inf)
                    copy_r2g(stat, lse_f32[query, head], predicate=True)
                    # instruction_selection: unconditional
                    # `lg2.approx.ftz.f32`, `setp.gt`, two `mul.f32`, one ordered
                    # `fma.rn.f32`, `selp.f32` to computed value or `-inf`, and
                    # one scalar `st.global.b32` per valid enabled statistics row.
                if col_half == 0 and return_temperature_lse:
                    log_temperature_sum = lg2(final_temperature_sum)
                    stat = -inf
                    if not (final_temperature_sum <= 0.0 or unordered):
                        max_scaled = final_max*softmax_scale_log2
                        max_scaled_ln2 = max_scaled*ln2
                        log_term = log_temperature_sum*ln2
                        stat = fma(lse_temperature_scale, max_scaled_ln2, log_term)
                    copy_r2g(stat, temperature_lse_f32[query, head], predicate=True)
                # instruction_selection: unconditional `lg2.approx.ftz.f32`,
                # then, for temperature LSE, `setp.leu`, `mov.b32` of `-inf`, a
                # predicate branch around the three multiplies and ordered
                # `fma.rn.f32(lse_temperature_scale, max_scaled_ln2, log_term)`,
                # followed by scalar `st.global.b32`.
        wait_tmem_loads(); fence_tmem_before_thread_sync(); arrive(TMEM_DEALLOC)
        # instruction_selection: `tcgen05.wait::ld.sync.aligned`,
        # `tcgen05.fence::before_thread_sync`, and a full-warp mbarrier arrival.

    # 5. Source-order role: warp 12, QK then probability-times-V.
    with role(warp == 12):
        wait(UNION_READY, phase=0); max_union = max(UNION_COUNT[0], UNION_COUNT[1])
        stage, phase = 0, 0
        wait(Q_FULL, phase=0)
        first_pv = [True, True]; p_phase = [0, 0]
        for union_index in rolled_range(64):
            for instance in static_range(2):
                if union_index < UNION_COUNT[instance]:
                    take(stage_now, phase_now); advance_ring(stage, phase, 3)
                    wait(KV_FULL[stage_now], phase_now)
                    for qk_atom in static_range(8):
                        gemm_ss(
                            taddr+SCORES[instance],
                            smem_descriptor(0x40004040, Q0 if instance == 0 else Q1,
                                            atom=qk_q_atom(qk_atom)),
                            smem_descriptor(0x40004040, kv_slot(stage_now),
                                            atom=qk_k_atom(qk_atom)),
                            instruction_desc=69207184,
                            accumulate=(qk_atom != 0),
                        )
                    commit(S_FULL[instance]); commit(KV_EMPTY[stage_now])
                    # instruction_selection: eight elected
                    # `tcgen05.mma.cta_group::1.kind::f16` SS issues, followed
                    # by two elected `tcgen05.commit...shared::cluster.b64`.
            for instance in static_range(2):
                if union_index < UNION_COUNT[instance]:
                    take(stage_now, phase_now); advance_ring(stage, phase, 3)
                    wait(KV_FULL[stage_now], phase_now); wait(P_FULL[instance], p_phase[instance])
                    p_phase[instance] ^= 1
                    for pv_atom in static_range(8):
                        gemm_ts(
                            taddr+OUTPUT[instance],
                            taddr+SCORES[instance]+64+8*pv_atom,
                            smem_descriptor(0x40004040, kv_slot(stage_now),
                                            lbo_transpose=True, atom=128*pv_atom),
                            instruction_desc=69272720,
                            accumulate=(not first_pv[instance] or pv_atom != 0),
                        )
                    first_pv[instance] = False
                    if union_index+1 == UNION_COUNT[instance]: commit(O_FULL[instance])
                    commit(KV_EMPTY[stage_now])
                    # instruction_selection: eight elected
                    # `tcgen05.mma.cta_group::1.kind::f16` TS issues, an
                    # end-of-instance O commit, and one KV release commit.
        wait(TMEM_DEALLOC, 0)
        deallocate_tmem(load_smem_volatile_u32(TMEM_MAILBOX), 512)
        # instruction_selection: acquire parity wait, volatile shared load,
        # `tcgen05.dealloc.cta_group::1.sync.aligned.b32`; extent: warp 12.


    # 6. Source-order explicit idle role.
    with role(13 <= warp <= 14):
        pass

    # 7. Source-order role: warp 15, Q load, unions, and aliased K/V feed.
    with role(warp == 15):
        (batch, q_tile, q_valid, query_base, q_local_base, k_start,
         kv_len, query_offset, num_n_blocks) = decode_tile_metadata()
        if elect_one():
            expect_bytes(Q_FULL, 32768)
            copy_g2s(q_map, (0, 16*kv_head, query_base, 0), Q0, Q_FULL)
            copy_g2s(q_map, (0, 16*kv_head, query_base+4, 0), Q1, Q_FULL)
        # instruction_selection: one `mbarrier.arrive.expect_tx...` and two
        # `cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes`;
        # extent: two 4-query x 16-head x D128 Q boxes.

        token_mask_low, token_mask_high = 0, 0
        if lane < q_valid:
            for slot in rolled_range(16):
                selected_block = load_global(q2k_indices_i32[
                    (kv_head*total_q + query_base+lane)*16 + slot])
                if 0 <= selected_block < num_n_blocks:
                    set_corresponding_bit(token_mask_low, token_mask_high, selected_block)
        if lane < 8:
            store_smem(token_mask_low, MASK_LOW + 4*lane)
            store_smem(token_mask_high, MASK_HIGH + 4*lane)
        # instruction_selection: rolled `ld.global.nc.b32` selection loads and
        # eight-lane `st.shared.b32` mask stores; extent: 16 slots per valid query.
        named_barrier(8, 32)
        # instruction_selection: `barrier.sync 8, 32`; extent: loader warp.

        if lane < 2:
            instance = lane
            union_low = OR(load_smem(MASK_LOW + 4*(4*instance+r)) for r in static_range(4))
            union_high = OR(load_smem(MASK_HIGH + 4*(4*instance+r)) for r in static_range(4))
            count = 0
            for ordinal in rolled_range(popcount(union_high)):
                bit = 31-clz(union_high)
                store_smem(bit+32, UNION_BLOCKS + 4*(instance*64+count))
                union_high ^= 1 << bit; count += 1
            for ordinal in rolled_range(popcount(union_low)):
                bit = 31-clz(union_low)
                store_smem(bit, UNION_BLOCKS + 4*(instance*64+count))
                union_low ^= 1 << bit; count += 1
            if count == 0: store_smem(0, UNION_BLOCKS + 4*instance*64); count = 1
            store_smem(count, UNION_COUNT + 4*instance)
        # instruction_selection: `popc.b32`, `clz.b32`, and rolled
        # `st.shared.b32`; extent: one descending high-half then descending
        # low-half union list for each four-query instance.
        named_barrier(8, 32); fence_async_shared(); arrive(UNION_READY)
        # instruction_selection: `barrier.sync 8,32`,
        # `fence.proxy.async.shared::cta`, and one full-warp
        # `mbarrier.arrive.release.cta.shared::cta.b64`.

        load_stage, empty_phase = 0, 1
        max_union = max(load_smem(UNION_COUNT), load_smem(UNION_COUNT+4))
        for union_index in rolled_range(64):
            for instance in static_range(2):
                if union_index < load_smem(UNION_COUNT+4*instance):
                    block = load_smem(UNION_BLOCKS+4*(instance*64+union_index))
                    wait(KV_EMPTY[load_stage], empty_phase)
                    if elect_one():
                        expect_bytes(KV_FULL[load_stage], 32768)
                        for kv_part in static_range(2):
                            for half in static_range(2):
                                copy_g2s(k_map,
                                    (0, k_start+128*block+64*half, kv_part, kv_head),
                                    kv_slot(load_stage)+8192*(half+2*kv_part),
                                    KV_FULL[load_stage])
                    advance_ring(load_stage, empty_phase, 3)
            for instance in static_range(2):
                if union_index < load_smem(UNION_COUNT+4*instance):
                    block = load_smem(UNION_BLOCKS+4*(instance*64+union_index))
                    wait(KV_EMPTY[load_stage], empty_phase)
                    if elect_one():
                        expect_bytes(KV_FULL[load_stage], 32768)
                        for kv_part in static_range(2):
                            for half in static_range(2):
                                copy_g2s(v_map,
                                    (0, k_start+128*block+64*half, kv_part, kv_head),
                                    kv_slot(load_stage)+8192*(half+2*kv_part),
                                    KV_FULL[load_stage])
                    advance_ring(load_stage, empty_phase, 3)
        # instruction_selection per live block: acquire parity retry wait,
        # `mbarrier.arrive.expect_tx...32768`, and four rank-4 TMA loads;
        # extent: live K0/K1 then V0/V1 within each union ordinal; a released
        # stage may change payload kind while other stages remain live.

```

## Physical descriptor and alias facts

| item | exact source choice |
| --- | --- |
| QK SS descriptor high word | `0x40004040`; Q atoms advance by 2 then jump 506, K atoms advance by 2 then jump 1018 |
| QK instruction descriptor | `69207184` (`0x04200010`) |
| PV TS descriptor high word | `0x40004040`; B low address sets the source transpose/LBO bit and advances 128 per atom |
| PV instruction descriptor | `69272720` (`0x04210010`) |
| KV stage | 32,768 bytes = four 8,192-byte TMA boxes; stage address advances 2,048 units of 16 bytes in matrix descriptors |
| alias lifetime | within each union ordinal, live K0/K1 stages precede live V0/V1 stages; each released slot may change payload kind while other slots remain live |

## Pipeline inventory

| edge | slots / initial phase | producer | consumer | completion / release |
| --- | --- | --- | --- | --- |
| Q | one / ready 0 | warp 15 elected lane | warp 12 | TMA expected bytes 32,768; no reuse |
| union metadata | one / ready 0 | all warp-15 lanes arrive after two named barriers | warps 0-12 | arrival count 32; immutable thereafter |
| aliased K/V | three / producer empty phase 1, consumer full phase 0 | warp 15 | warp 12 | 32,768 expected bytes per live tile; tcgen05 commit releases each slot |
| scores | one singleton per instance, alternating parity | warp 12 QK | four softmax warps | tcgen05 commit `s_full[i]`; one logical tile at a time |
| probabilities | one singleton per instance, alternating parity | four softmax warps | warp 12 PV | 256 full-warp arrivals, including the correction role's seed/release arrivals |
| online correction | singleton `corr_sig/corr_done` per instance | softmax warps / correction warps | correction warps / softmax warps | 128 arrivals each phase; correction skipped collectively only when all scales are one |
| output | one singleton per instance | warp 12 final PV | correction/epilogue warps | tcgen05 commit after last selected block |
| deallocation | one count-128 barrier | warps 8-11 | warp 12 | full-warp arrivals after final TMEM reads |

## Verification and benchmark contract

- Correctness compares the TIRx output directly with the pinned FlashInfer
  source on byte-identical tensors. BF16 output is checked bitwise first; any
  fallback must be justified by an observed source/TIRx instruction-order
  difference and may not exceed one BF16 ULP. LSE outputs use exact special-value
  agreement and tight finite tolerances derived from the source approximation
  path, never broad attention tolerances.
- Guard regions, finite/poison overwrite, repeated determinism, causal and
  non-causal paths, explicit and derived offsets, q-tail boundaries, fully
  masked rows, both LSE flags, duplicate/disjoint selections, union capacity,
  ragged batches, and KV length 8,192 are required correctness cases.
- `prepare_bench` compiles before timing. `run_gpu` allocates and validates once
  and gives the canonical Proton timer exactly one source or TIRx launch per
  closure. Only `bench_suite` reference rows are performance evidence.
- The frozen benchmark matrix includes small, production, maximum-KV, ragged,
  non-causal, and both-LSE workloads. Every row must satisfy the strict ratio
  `flashinfer_time / tirx_time > 0.99`.
- Before accepting any benchmark row, both exported source PTX and generated
  TIRx PTX must declare `.version 9.3` and target `sm_103a` under the same CUDA
  13.3 toolchain.

## Instruction-selection summary

| source decision | emitted consequence in writer PTX |
| --- | --- |
| fixed 512-thread, one-CTA schedule | `.maxntid 512`, `.minnctapersm 1`, 512-column group-1 TMEM allocation |
| rank-one generated shared struct | dynamic `.extern .shared .align 1024`; exact offsets and K/V alias above |
| exact-union loader | rolled non-caching global selection loads, shared mask stores, two warp-local named barriers, `popc`/`clz`, then rank-4 TMA |
| two QK/PV instances | 16 SS f16 MMA sites and 16 TS f16 MMA sites in the rolled union loop, with instance-specific barriers and TMEM quadrants |
| online softmax | x32 TMEM loads, lane-16 shuffle reductions, packed f32x2 FMA/add/mul, approximate exp2, BF16 packing, x16 TMEM stores |
| correction threshold | packed online-rescale handshake plus warp-any guarded x64 TMEM load/scale/store path |
| fused epilogue | approximate reciprocal, packed normalization/conversion, vector global stores, optional approximate-log2 scalar statistics |

Static opcode counts use the corpus convention: instruction lines minus
predicated lines. The writer entry contains 19 mbarrier initializations, 18
rank-4 TMA loads, 12 x32 TMEM loads, four x16 TMEM stores, two x64 TMEM stores,
16 vector global stores, four approximate log2 sites, two approximate reciprocal
sites, and one each of TMEM allocate/relinquish/deallocate.
