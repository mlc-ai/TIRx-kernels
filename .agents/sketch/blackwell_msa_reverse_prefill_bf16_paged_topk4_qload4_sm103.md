<!--
This file is a design sketch for a TIRx port of code from FlashInfer
(https://github.com/flashinfer-ai/flashinfer @ cc6e8794c49bf66172627bdb9742fcb17d18b839),
Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# FlashInfer SM103a reverse-prefill BF16 paged TopK4 Q-load4: coarse WASP pipeline sketch

This is a non-executable operation-level sketch for
[`tirx_kernels/flashinfer/msa_ops/blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_sm103.py`](../../tirx_kernels/flashinfer/msa_ops/blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_sm103.py),
which becomes the executable source of truth. It freezes this exact two-kernel
FlashInfer route at commit `cc6e8794c49bf66172627bdb9742fcb17d18b839`:

- producer:
  `sm103a/blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4.cu::kernel_minimax_sparse_reverse_prefill_paged_bf16_gqa4_qload4_fp8partial_temp1reuse_sm100`;
- reducer:
  `sm103a/blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_const4_reduce.cu::kernel_minimax_sparse_reverse_prefill_combine_topk4_eighthwarp16_metaparallel_const4_qload4fp8partial_temp1reuse_sm100`.

The producer and reducer source SHA-256 values are respectively
`7beb023b5549012282284bc542fede7719db35f91942f27e7fadd7a9e9f57bcd`
and `42f28e0ba061b664109cd29de614c8769d4d038e4e583af12b9457a24636b138`.
The writer line-info exports are
`.porting/blackwell_msa_reverse_prefill_bf16_paged_topk4_qload4_sm103/source_export/writer/producer/kernel.ptx`
(SHA-256 `c25742648e7720e6e452894adb517843de2454424a0442b4d78addc0decaa03d`,
2610 `.loc` records) and the adjacent `reducer/kernel.ptx` (SHA-256
`09857f0dfabb8ab26e1398c833b03657383591b4a9deb7cf2fa60aca3c59c728`,
410 `.loc` records). CUDA 13.3.73 emitted PTX ISA 9.3 targeting `sm_103a`.
All instruction selections below come from those exports.

The fixed public specialization is batch 3 with Q lengths 4096 and paged KV
lengths 8192, Q/KV head counts 8/2, head dimension and page size 128, causal
masking with derived query offsets, GQA ratio 4, TopK and split count 4, 152-SM
planning, 389 producer work CTAs, and 3072 reducer CTAs. Planner seeds and page
permutations may change scheduler payloads without changing the device
specialization. Other dimensions, dtypes, TopK values, GQA ratios, page sizes,
architectures, and non-paged routes are out of scope.

After the independent sketch reviewer returns PASS, this file is immutable.

## Pipeline at a glance

| kernel / warps | role | tile program | publication / reuse edges |
| --- | --- | --- | --- |
| producer warp 0 | setup plus even-softmax worker | elected lane initializes 21 barriers; full warp allocates and relinquishes 512 TMEM columns; then it joins even groups | init fence, warp/CTA rendezvous, then even-stream edges |
| producer warps 0-3 | even groups | each warp owns eight query edges and all four GQA heads per edge; scores, BF16 P, FP8 partial O, scale, partial LSE | `s_full[0] -> p_full[0] -> o_full[0]`; releases `s_empty[0]`, `o_empty[0]` |
| producer warps 4-7 | odd groups | same program for odd-numbered 32-edge groups | corresponding stage-1 barriers |
| producer warps 8-11 | Q gather producers | each warp issues two gather4 TMA loads for eight edges into one alternating Q stage | `q_empty[stage] -> q_full[stage]`, four producer arrivals per stage |
| producer warp 12 | MMA consumer/producer | one-time K wait, alternating two-stage QK, two-groups-delayed PV, final two-group drain, TMEM deallocation | consumes Q/K/V/P and publishes S/Q-empty/O/P-empty |
| producer warps 13-14 | compiled-empty transform roles | common `setmaxnreg.dec(48)` only; source metadata body is eliminated | no role-body edge |
| producer warp 15 | paged K/V TMA producer | clamp physical page, load the single 128x128 K and V tiles | publishes singleton `k_full` and `v_full` |
| reducer all 8 warps | 32 independent eight-lane row cohorts per CTA | combine four LSE/scales; each lane accumulates one 16-column segment from four FP8 rows | no shared-memory or CTA synchronization |

## Primitive vocabulary and storage rule

All storage is rank one. Logical dimensions, 128-byte swizzles, MMA atoms,
stages, and lane ownership are expressed by scalar byte/index functions, never
by a first-class layout:

```python
linear_smem(name, bytes, alignment)
reg_array(name, dtype, elements)
tensor_map(name, rank, extents, byte_strides, box, dtype, swizzle_code)
smem_byte(base, stage, scalar_index)
tmem_column(base, stage, column, row_band)

copy_g2s(tensor_map, coordinates, smem_byte, completion_barrier)
copy_g2r(global_pointer, registers, predicate)
copy_t2r(tmem_column, registers)
copy_r2t(registers, tmem_column)
copy_r2g(registers, global_pointer, predicate)
gemm_ss(tmem_dst, smem_a_desc, smem_b_desc, instruction_desc, accumulate)
gemm_ts(tmem_dst, tmem_a, smem_b_desc, instruction_desc, accumulate)
fill(registers, value, predicate)
cast(registers, source_dtype, destination_dtype, rounding)
exp(registers); add(dst, lhs, rhs); mul(dst, lhs, rhs); fma(dst, a, b, c)
init(barrier, count); wait(barrier, phase); expect_bytes(barrier, bytes)
arrive(barrier); commit(barrier); fence(kind); barrier(kind)
```

No primitive hides another phase or output path. Descriptor construction and
routine pointer arithmetic are scalar bookkeeping; every operation that moves
or computes data remains explicit below.

## Complete producer sketch

```python
@kernel(
    target="sm_103a",
    grid=(num_work_items, 1, 1),
    block=(512, 1, 1),
    min_blocks_per_sm=1,
    dynamic_smem_bytes=148480,
)
# instruction_selection: `.version 9.3`, `.target sm_103a`, `.maxntid 512`,
# `.minnctapersm 1`, `.extern .shared .align 1024`; extent: one producer entry.
def reverse_prefill_producer(
    q_map, k_map, v_map, scheduler_metadata, k2q_row_ptr,
    k2q_qsplit_indices, partial_o, partial_scale, partial_lse,
    partial_temperature_lse, cu_q, cu_k, q_offsets, kv_lens, page_table,
    segment_end_21, ..., segment_end_2,
    total_q, num_q_heads, num_kv_heads, total_rows, nnz_per_head,
    work_capacity, num_work_items, topk, max_pages, causal,
    derive_q_offset, softmax_scale_log2, lse_temperature_scale,
    return_temperature_lse,
):
    tid = thread_id()
    warp = warp_uniform(tid // 32)
    lane = tid % 32
    work = block_id_x()

    # Exact rank-one dynamic arena. Q-store aliases Q; V-convert aliases V.
    # FP8 is physically reserved but has no emitted access in this specialization.
    smem = linear_smem("smem", 148480, alignment=1024)
    Q_FULL = [0, 8]                    # init count 4
    Q_EMPTY = [16, 24]                 # init count 1
    K_FULL = 32                        # init count 1
    V_FULL = 40                        # init count 1
    FP8_K_FULL = 48                    # init count 1, dead after init
    FP8_V_FULL = 56                    # init count 1, dead after init
    FP8_EMPTY = 64                     # init count 1, dead after init
    S_FULL = [72, 80]                  # init count 1
    S_EMPTY = [88, 96]                 # init count 128
    P_FULL = [104, 112]                # init count 128
    P_EMPTY = [120, 128]               # init count 1
    O_FULL = [136, 144]                # init count 1
    O_EMPTY = [152, 160]               # init count 128
    TMEM_MAILBOX = 168
    Q = Q_STORE = 1024                 # two 32768-byte stages
    K = 66560                          # one 32768-byte tile
    V = V_CONVERT = 99328              # one 32768-byte tile
    FP8_UNUSED = 132096                # 16384 reserved bytes

    def q_stage_byte(stage): return Q + stage*32768
    def q_edge_byte(stage, edge, half):
        return q_stage_byte(stage) + edge*512 + half*16384
    def k_or_v_part_byte(base, dim_half, token_half):
        return base + dim_half*16384 + token_half*8192
    def tmem_row_band(warp_in_group): return (warp_in_group*32) << 16
    def score_base(stage, warp_in_group):
        return taddr + stage*128 + tmem_row_band(warp_in_group)
    def probability_base(stage, warp_in_group):
        return score_base(stage, warp_in_group) + 64
    def output_base(stage, warp_in_group):
        return taddr + 256 + stage*128 + tmem_row_band(warp_in_group)

    if warp == 0 and elect_one():
        for stage in static_range(2): init(Q_FULL[stage], 4)
        for stage in static_range(2): init(Q_EMPTY[stage], 1)
        init(K_FULL, 1); init(V_FULL, 1)
        init(FP8_K_FULL, 1); init(FP8_V_FULL, 1); init(FP8_EMPTY, 1)
        for stage in static_range(2): init(S_FULL[stage], 1)
        for stage in static_range(2): init(S_EMPTY[stage], 128)
        for stage in static_range(2): init(P_FULL[stage], 128)
        for stage in static_range(2): init(P_EMPTY[stage], 1)
        for stage in static_range(2): init(O_FULL[stage], 1)
        for stage in static_range(2): init(O_EMPTY[stage], 128)
        fence("mbarrier_init.release.cluster")
    # instruction_selection: 21 `mbarrier.init.shared::cta.b64` and one
    # `fence.mbarrier_init.release.cluster`; extent: one elected warp-0 lane.

    barrier("warp_sync")
    # instruction_selection: `bar.warp.sync -1`; extent: all 16 warps.
    if warp == 0:
        allocate_tmem(TMEM_MAILBOX, columns=512)
        relinquish_tmem()
    # instruction_selection: `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32`
    # and `tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned`;
    # extent: the full setup warp.
    barrier("cta_sync")
    fence("tcgen05.after_thread_sync")
    # instruction_selection: `bar.sync 0` then `tcgen05.fence::after_thread_sync`.
    taddr = load_smem_volatile_u32(TMEM_MAILBOX)
    # instruction_selection: `ld.volatile.shared.b32`; extent: one scalar/thread.

    if 12 <= warp <= 15: setmaxnreg_decrease(48)
    # instruction_selection: `setmaxnreg.dec.sync.aligned.u32 48`; extent: warps 12-15.

    # All active roles independently load this work record. Parameters unused by
    # a role are eliminated. Group count is selected by the ordered 21..2 segment
    # boundary ladder and lies in [1, 21].
    def decode_work_record():
        group_count = descending_segment_lookup(work, segment_end_21, ..., segment_end_2)
        head_kv, row_linear, q_begin, q_count, batch, kv_block = \
            copy_g2r(scheduler_metadata[work*6:work*6+6])
        row_start = copy_g2r(k2q_row_ptr[head_kv*(total_rows+1)+row_linear]) + q_begin
        q_batch = copy_g2r(cu_q[batch])
        k_batch = copy_g2r(cu_k[batch])
        kv_len = copy_g2r(kv_lens[batch])
        if max_pages == 0: kv_len = copy_g2r(cu_k[batch+1]) - k_batch
        query_offset = copy_g2r(q_offsets[batch])
        if derive_q_offset:
            query_offset = kv_len - (copy_g2r(cu_q[batch+1]) - q_batch)
        return group_count, head_kv, row_start, q_count, batch, kv_block, q_batch, kv_len, query_offset
    # instruction_selection: `ld.global.nc.b32` scalar loads, with only each
    # caller's live subset retained; extent: one independent decode per active role.

    # The two source-ordered role branches below call this shared semantic body;
    # stage, warp band, parity, and trip count remain explicit at each call site.
    def softmax_parity_body(
        stage, warp_in_group, first_group, trip_count,
        group_count, head_kv, row_start, q_count, batch, kv_block,
        q_batch, kv_len, query_offset,
    ):
        my_row = warp_in_group*32 + lane

        for iteration in rolled_range(trip_count):
            group = 2*iteration + first_group
            phase = iteration & 1
            wait(S_FULL[stage], phase)
            # instruction_selection: rolled
            # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64` retry loop;
            # extent: one wait per live parity-group.

            edge = group*32 + my_row//4
            owner_lane = (lane//4)*4
            packed_q = -1
            if lane == owner_lane and edge < q_count:
                packed_q = copy_g2r(k2q_qsplit_indices[
                    head_kv*nnz_per_head + row_start + edge])
                # instruction_selection: `ld.global.nc.b32`; extent: eight elected
                # four-lane cohort leaders per softmax warp.
            packed_q = shuffle_idx(packed_q, owner_lane)
            # instruction_selection: `shfl.sync.idx.b32`; extent: one broadcast/thread.
            q_index = packed_q & 0x00ffffff
            valid_cols = 0
            if edge < q_count:
                valid_cols = clamp(kv_len-kv_block*128, 0, 128)
                if causal:
                    valid_cols = min(valid_cols, query_offset+q_index-kv_block*128+1)
                valid_cols = max(valid_cols, 0)

            s0 = reg_array("score_lo", f32, 64)
            s1 = reg_array("score_hi", f32, 64)
            copy_t2r(score_base(stage, warp_in_group), s0)
            copy_t2r(score_base(stage, warp_in_group)+64, s1)
            # instruction_selection: two
            # `tcgen05.ld.sync.aligned.32x32b.x64.b32`; extent: 128 FP32 scores/thread.
            fill(s0[clamp(valid_cols,0,64):], -inf, predicate=0 < valid_cols < 64)
            fill(s1[clamp(valid_cols-64,0,64):], -inf,
                 predicate=valid_cols > 0 and valid_cols-64 < 64)
            # instruction_selection: generated mask arithmetic and predicated
            # `selp.f32` register replacement; extent: invalid suffixes only.
            row_max = max(reduce_max(s0), reduce_max(s1))
            # instruction_selection: unrolled scalar `max.f32` tree; extent:
            # each lane's 128 values, with `-inf` selected for an empty half.
            safe_max = 0.0 if row_max == -inf else row_max
            score_bias = -safe_max*softmax_scale_log2 if valid_cols > 0 else -inf
            # instruction_selection: branch on `valid_cols`; the live path emits
            # `setp.eq.ftz.f32`, `neg.ftz.f32`, one `selp.f32`, and `mul.ftz.f32`;
            # extent: one safe maximum and bias per row.

            wait(P_EMPTY[stage], phase ^ 1)
            # instruction_selection: rolled acquire parity retry wait; extent: one/group.
            p0 = reg_array("probability_lo", f32, 64)
            p1 = reg_array("probability_hi", f32, 64)
            copy_t2r(score_base(stage, warp_in_group), p0)
            copy_t2r(score_base(stage, warp_in_group)+64, p1)
            # instruction_selection: two more
            # `tcgen05.ld.sync.aligned.32x32b.x64.b32`; extent: 128 scores/thread.
            fill(p0[clamp(valid_cols,0,64):], -inf, predicate=0 < valid_cols < 64)
            fill(p1[clamp(valid_cols-64,0,64):], -inf,
                 predicate=valid_cols > 0 and valid_cols-64 < 64)
            # instruction_selection: same generated suffix-mask instruction family.
            for pair in static_range(32):
                fma(p0[2*pair:2*pair+2], p0[2*pair:2*pair+2],
                    softmax_scale_log2, score_bias)
                fma(p1[2*pair:2*pair+2], p1[2*pair:2*pair+2],
                    softmax_scale_log2, score_bias)
            # instruction_selection: 64 packed `fma.rn.ftz.f32x2`; extent:
            # 32 pairs in each score half.
            exp(p0); exp(p1)
            # instruction_selection: 128 scalar `ex2.approx.ftz.f32`; extent: one/group/thread.
            row_sum = add(reduce_sum_f32x2(p0), reduce_sum_f32x2(p1))
            # instruction_selection: 64 `add.f32x2` plus scalar adds; extent: 128 values.
            pb0 = cast(p0, f32, bf16x2, rounding="rn")
            pb1 = cast(p1, f32, bf16x2, rounding="rn")
            # instruction_selection: `cvt.rn.bf16x2.f32`; extent: 64 packed words.
            copy_r2t(pb0, probability_base(stage, warp_in_group))
            copy_r2t(pb1, probability_base(stage, warp_in_group)+32)
            # instruction_selection: two
            # `tcgen05.st.sync.aligned.32x32b.x32.b32`; extent: 128 BF16 P values/thread.
            wait_tmem_stores()
            arrive(P_FULL[stage])
            # instruction_selection: `tcgen05.wait::st.sync.aligned` then
            # `mbarrier.arrive.release.cta.shared::cta.b64`; extent: one full-warp arrival/thread.
            wait_tmem_loads()
            arrive(S_EMPTY[stage])
            # instruction_selection: `tcgen05.wait::ld.sync.aligned` then one
            # `mbarrier.arrive.release.cta.shared::cta.b64`; extent: one/thread.

            wait(O_FULL[stage], phase)
            # instruction_selection: rolled acquire parity retry wait; extent: one/group.
            q_head_local = my_row - (my_row//4)*4
            output_valid = edge < q_count and 0 <= ((packed_q >> 24) & 255) < topk
            partial_row = 0
            inv_sum = 0.0
            if output_valid:
                split = (packed_q >> 24) & 255
                q_abs = q_batch + q_index
                q_head = head_kv*4 + q_head_local
                partial_row = split*(total_q*num_q_heads) + q_abs*num_q_heads + q_head
                inv_sum_candidate = rcp(row_sum)
                inv_sum = inv_sum_candidate if row_sum > 0.0 and not_nan(row_sum) else 0.0
                # instruction_selection: conditional `rcp.approx.ftz.f32` plus
                # predicate-controlled selection; extent: one live output row.

            row_abs_max = 0.0
            for segment in rolled_range(8):
                out16 = reg_array("out16_for_amax", f32, 16)
                copy_t2r(output_base(stage, warp_in_group)+segment*16, out16)
                # instruction_selection: rolled
                # `tcgen05.ld.sync.aligned.32x32b.x16.b32`; extent: eight loads/thread.
                row_abs_max = max(row_abs_max, max(reduce_max(out16), -reduce_min(out16)))
                # instruction_selection: scalar `max.f32` and `min.ftz.f32`
                # trees plus `neg.ftz.f32`; extent: 16 values per segment.
            wait_tmem_loads()
            # instruction_selection: `tcgen05.wait::ld.sync.aligned`; extent: one.
            dequant_scale = 0.0
            quant_scale = 0.0
            if row_abs_max > 0.0 and not_nan(row_abs_max):
                dequant_scale = row_abs_max*inv_sum*(1.0/448.0)
                quant_scale = 448.0/row_abs_max
                # instruction_selection: two `mul.ftz.f32` for dequantization
                # scale and one `div.approx.ftz.f32` for quantization scale;
                # extent: one finite nonzero row.
            if output_valid:
                copy_r2g(dequant_scale, partial_scale[partial_row], True)
                # instruction_selection: scalar `st.global.b32`; extent: one live row.

            for segment in rolled_range(8):
                out16 = reg_array("out16_for_quant", f32, 16)
                copy_t2r(output_base(stage, warp_in_group)+segment*16, out16)
                # instruction_selection: rolled
                # `tcgen05.ld.sync.aligned.32x32b.x16.b32`; extent: eight loads/thread.
                if output_valid:
                    mul(out16, out16, quant_scale)
                    # instruction_selection: eight `mul.rn.ftz.f32x2`; extent: 16 values.
                    fp8 = cast(out16, f32, e4m3x2, rounding="rn.satfinite")
                    # instruction_selection: eight
                    # `cvt.rn.satfinite.e4m3x2.f32`; extent: 16 FP8 values.
                    copy_r2g(fp8, partial_o[partial_row*128+segment*16], True)
                    # instruction_selection: `st.global.v4.b32`; extent: 16 bytes.
            if output_valid:
                log_sum = lg2(row_sum)
                partial_stat = fma(row_max*softmax_scale_log2,
                                   0.6931471805599453,
                                   log_sum*0.6931471805599453) if row_sum > 0.0 else -inf
                copy_r2g(partial_stat, partial_lse[partial_row], True)
                # instruction_selection: `lg2.approx.ftz.f32`, ordered
                # `mul.ftz.f32`/`fma.rn.ftz.f32`, then `st.global.b32`; extent: one live row.
            wait_tmem_loads()
            arrive(O_EMPTY[stage])
            # instruction_selection: `tcgen05.wait::ld.sync.aligned` then one
            # `mbarrier.arrive.release.cta.shared::cta.b64`; extent: one/thread.

    # Source-order role 1: even groups on warps 0-3.
    if warp <= 3:
        setmaxnreg_increase(176)
        # instruction_selection: first static
        # `setmaxnreg.inc.sync.aligned.u32 176`; extent: warps 0-3.
        group_count, head_kv, row_start, q_count, batch, kv_block, q_batch, kv_len, query_offset = decode_work_record()
        softmax_parity_body(
            0, warp, 0, ceil_div(group_count, 2),
            group_count, head_kv, row_start, q_count, batch, kv_block,
            q_batch, kv_len, query_offset)

    # Source-order role 2: odd groups on warps 4-7.
    if 4 <= warp <= 7:
        setmaxnreg_increase(176)
        # instruction_selection: second static
        # `setmaxnreg.inc.sync.aligned.u32 176`; extent: warps 4-7.
        group_count, head_kv, row_start, q_count, batch, kv_block, q_batch, kv_len, query_offset = decode_work_record()
        softmax_parity_body(
            1, warp-4, 1, group_count//2,
            group_count, head_kv, row_start, q_count, batch, kv_block,
            q_batch, kv_len, query_offset)

    # Warps 8-11 each own eight edge slots of every group and issue gather4.
    if 8 <= warp <= 11:
        setmaxnreg_decrease(112)
        # instruction_selection: `setmaxnreg.dec.sync.aligned.u32 112`;
        # extent: four Q-load warps.
        group_count, head_kv, row_start, q_count, batch, _, q_batch, _, _ = decode_work_record()
        qload_warp = warp-8
        for group in rolled_range(group_count):
            stage = group & 1
            phase = (group//2) & 1
            wait(Q_EMPTY[stage], phase ^ 1)
            # instruction_selection: rolled acquire parity retry wait; extent: one/group.
            if elect_one(): expect_bytes(Q_FULL[stage], 8192)
            # instruction_selection: elected
            # `mbarrier.arrive.expect_tx.release.cta.shared::cta.b64`; extent:
            # one 8192-byte contribution per Q-load warp.
            if elect_one():
                for local_edge in static_range(8):
                    edge = group*32 + qload_warp*8 + local_edge
                    safe_edge = edge if edge < q_count else 0
                    packed_q = copy_g2r(k2q_qsplit_indices[
                        head_kv*nnz_per_head + row_start + safe_edge])
                    # instruction_selection: `ld.global.nc.b32`; extent: eight elected loads/warp.
                    q_abs = q_batch + (packed_q & 0x00ffffff) if edge < q_count else 0
                    row_base = q_abs*num_q_heads + head_kv*4
                    copy_g2s(q_map, (0, row_base, row_base+1, row_base+2, row_base+3),
                             q_edge_byte(stage, qload_warp*8+local_edge, 0), Q_FULL[stage])
                    copy_g2s(q_map, (64, row_base, row_base+1, row_base+2, row_base+3),
                             q_edge_byte(stage, qload_warp*8+local_edge, 1), Q_FULL[stage])
                    # instruction_selection: two
                    # `cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes`;
                    # extent: two D64 halves for four GQA heads, eight edges/warp.

    # Warp 12 runs a fixed two-stage QK/PV software pipeline.
    if warp == 12:
        group_count, *_ = decode_work_record()
        wait(K_FULL, 0)
        # instruction_selection: acquire parity retry wait; extent: singleton K tile.

        def issue_qk(group):
            stage = group & 1
            phase = (group//2) & 1
            wait(Q_FULL[stage], phase)
            wait(S_EMPTY[stage], phase ^ 1)
            # instruction_selection: two acquire parity retry waits; extent: one QK group.
            qlo = smem_descriptor_low(q_stage_byte(stage))
            klo = smem_descriptor_low(K)
            for atom, delta in enumerate((0,2,4,6,1024,1026,1028,1030)):
                gemm_ss(taddr+stage*128,
                        descriptor(0x40004040, qlo+delta),
                        descriptor(0x40004040, klo+delta),
                        instruction_desc=0x08200490,
                        accumulate=(atom != 0))
            # instruction_selection: eight elected
            # `tcgen05.mma.cta_group::1.kind::f16`; extent: Q[128,128] x K[128,128].
            commit(S_FULL[stage]); commit(Q_EMPTY[stage])
            # instruction_selection: two elected
            # `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`.

        def issue_pv(group):
            stage = group & 1
            phase = (group//2) & 1
            wait(P_FULL[stage], phase)
            wait(O_EMPTY[stage], phase ^ 1)
            # instruction_selection: two acquire parity retry waits; extent: one PV group.
            vlo = smem_descriptor_low(V) | 0x04000000
            for atom, (ta_delta, vb_delta) in enumerate((
                (32,512), (40,640), (48,768), (56,896),
                (0,0), (8,128), (16,256), (24,384),
            )):
                gemm_ts(taddr+256+stage*128,
                        taddr+stage*128+64+ta_delta,
                        descriptor(0x40004040, vlo+vb_delta),
                        instruction_desc=0x08210490,
                        accumulate=(atom != 0))
            # instruction_selection: eight elected
            # `tcgen05.mma.cta_group::1.kind::f16`; extent: P[128,128] x V[128,128],
            # second K-half atoms first, then first K-half atoms.
            commit(O_FULL[stage]); commit(P_EMPTY[stage])
            # instruction_selection: two elected tcgen05 mbarrier commits.

        issue_qk(0)
        if group_count > 1: issue_qk(1)
        wait(V_FULL, 0)
        # instruction_selection: acquire parity retry wait; extent: singleton V tile.
        for group in rolled_range(2, group_count):
            issue_pv(group-2)
            issue_qk(group)
        drain_start = 0 if group_count == 1 else group_count-2
        for group in rolled_range(drain_start, group_count): issue_pv(group)
        for group in rolled_range(drain_start, group_count):
            wait(O_EMPTY[group & 1], (group//2) & 1)
            # instruction_selection: acquire parity retry wait; extent: last one or two groups.
        deallocate_tmem(load_smem_volatile_u32(TMEM_MAILBOX), 512)
        # instruction_selection: `ld.volatile.shared.b32` then
        # `tcgen05.dealloc.cta_group::1.sync.aligned.b32`; extent: warp 12.

    # Source keeps a role branch but its metadata-only body compiles away.
    if 13 <= warp <= 14:
        pass

    if warp == 15:
        _, head_kv, _, _, batch, kv_block, _, _, _ = decode_work_record()
        physical_page = copy_g2r(page_table[batch*max_pages+kv_block])
        # instruction_selection: `ld.global.nc.b32`; extent: one scalar/thread,
        # with the elected TMA lane's value live.
        physical_page = max(physical_page, 0)
        page_head = physical_page*num_kv_heads + head_kv
        if elect_one():
            expect_bytes(K_FULL, 32768)
            for dim_half in static_range(2):
                for token_half in static_range(2):
                    copy_g2s(k_map, (0, token_half*64, dim_half, page_head),
                             k_or_v_part_byte(K, dim_half, token_half), K_FULL)
            # instruction_selection: elected `mbarrier.arrive.expect_tx...32768`
            # plus four `cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes`.
            expect_bytes(V_FULL, 32768)
            for dim_half in static_range(2):
                for token_half in static_range(2):
                    copy_g2s(v_map, (0, token_half*64, dim_half, page_head),
                             k_or_v_part_byte(V, dim_half, token_half), V_FULL)
            # instruction_selection: the same expected-byte arrival and four
            # rank-4 TMA loads for V; extent: one 128x128 BF16 tile.
```

`partial_temperature_lse`, `lse_temperature_scale`, `return_temperature_lse`,
`work_capacity`, and `num_work_items` remain in the source ABI but emit no
producer-body access. The FP8 scratch region and three FP8-related barriers listed above
are reserved/initialized exactly as source setup requires, but only the three
barriers are physical objects; no post-init FP8 pipeline operation is invented.

## Complete constant-four reducer sketch

```python
@kernel(
    target="sm_103a",
    grid=(ceil_div(total_q*num_q_heads, 32), 1, 1),
    block=(256, 1, 1),
    dynamic_smem_bytes=0,
)
# instruction_selection: `.version 9.3`, `.target sm_103a`, `.maxntid 256`;
# extent: one reducer entry with no shared declaration.
def reverse_prefill_reduce_const4(
    partial_o, partial_scale, partial_lse, partial_temperature_lse,
    split_counts, out, lse, temperature_lse,
    total_q, num_q_heads, num_kv_heads, qhead_per_kv, topk,
    return_softmax_lse, return_temperature_lse,
):
    tid = thread_id()
    warp = warp_uniform(tid//32)
    lane = tid & 31
    row_group = tid//8
    lane_in_row = tid & 7
    leader_lane = row_group*8
    row = block_id_x()*32 + row_group
    total_rows_out = total_q*num_q_heads
    row_valid = row < total_rows_out

    lane_lse = -inf
    lane_scale = 0.0
    if row_valid and lane_in_row < 4:
        split_row = lane_in_row*total_rows_out + row
        lane_lse = copy_g2r(partial_lse[split_row])
        lane_scale = copy_g2r(partial_scale[split_row])
        # instruction_selection: two `ld.global.nc.b32`; extent: four active
        # lanes in each eight-lane row cohort.

    lse_max = lane_lse
    for delta in (1, 2, 4):
        lse_max = max(lse_max, shuffle_xor(lse_max, delta))
    # instruction_selection: three `shfl.sync.bfly.b32` and three `max.f32`;
    # extent: one eight-lane reduction replicated across the cohort.
    safe_lse_max = 0.0 if lse_max == -inf else lse_max
    lane_weight = 0.0
    if lane_in_row < 4:
        lane_weight = exp((lane_lse-safe_lse_max)*1.4426950408889634)
        if lane_lse == -inf: lane_weight = 0.0
    # instruction_selection: `sub.ftz.f32`, `mul.ftz.f32`, and
    # `ex2.approx.ftz.f32`; extent: four split lanes.
    lse_sum = lane_weight
    for delta in (1, 2, 4): lse_sum = add(lse_sum, shuffle_xor(lse_sum, delta))
    # instruction_selection: three `shfl.sync.bfly.b32` plus three
    # `add.ftz.f32`; extent: one eight-lane reduction.
    inv_candidate = rcp(lse_sum)
    inv_lse_sum = inv_candidate if lse_sum > 0.0 and not_nan(lse_sum) else 0.0
    lane_weight = mul(mul(lane_weight, inv_lse_sum), lane_scale)
    # instruction_selection: `rcp.approx.ftz.f32` and two `mul.ftz.f32`;
    # extent: one lane weight.
    weight = reg_array("weight", f32, 4)
    for split in static_range(4):
        weight[split] = shuffle_idx(lane_weight, leader_lane+split)
    # instruction_selection: four `shfl.sync.idx.b32`; extent: broadcast four weights.

    if lane_in_row == 0 and row_valid:
        final_lse = -inf
        if return_softmax_lse or return_temperature_lse:
            final_lse = (safe_lse_max + lg2(lse_sum)*0.6931471805599453
                         if lse_sum > 0.0 else -inf)
            # instruction_selection: `lg2.approx.ftz.f32` followed by one
            # `fma.rn.ftz.f32`; extent: one row statistic.
        if return_softmax_lse:
            copy_r2g(final_lse, lse[row], True)
            # instruction_selection: `st.global.b32`; extent: one enabled row.
        if return_temperature_lse:
            copy_r2g(final_lse, temperature_lse[row], True)
            # instruction_selection: `st.global.b32`; extent: one enabled row.

    if row_valid:
        col = lane_in_row*16
        accum = reg_array("accum", f32, 16)
        zero = 0.0
        # instruction_selection: one `mov.b32` zero scalar reused as the C
        # operand of all first-split FMAs; no standalone 16-register fill is emitted.
        for split in static_range(4):
            packed0 = copy_g2r(partial_o[(split*total_rows_out+row)*128+col:][:8])
            packed1 = copy_g2r(partial_o[(split*total_rows_out+row)*128+col+8:][:8])
            # instruction_selection: two `ld.global.nc.b64`; extent: 16 FP8 bytes/split.
            fp16_pairs = cast((packed0, packed1), e4m3x2, f16x2, rounding="rn")
            # instruction_selection: eight `cvt.rn.f16x2.e4m3x2`; extent: 16 values/split.
            values = cast(fp16_pairs, f16, f32, rounding="exact")
            # instruction_selection: sixteen `cvt.f32.f16`; extent: 16 values/split.
            for elem in static_range(16):
                c = zero if split == 0 else accum[elem]
                accum[elem] = fma(values[elem], weight[split], c)
            # instruction_selection: sixteen `fma.rn.ftz.f32`; extent: one split.
        packed_bf16 = cast(accum, f32, bf16x2, rounding="rn")
        # instruction_selection: eight `cvt.rn.bf16x2.f32`; extent: 16 outputs.
        copy_r2g(packed_bf16[:4], out[row*128+col], True)
        copy_r2g(packed_bf16[4:], out[row*128+col+8], True)
        # instruction_selection: two `st.global.v4.b32`; extent: 16 BF16 outputs.
```

`split_counts`, `partial_temperature_lse`, `num_kv_heads`, `qhead_per_kv`, and
`topk` are retained by the host ABI but are compile-dead in this fixed reducer.
The reducer deliberately uses four constant splits and writes the same combined
LSE to either requested final statistics output.

## TensorMap and global-memory contract

| map | fastest-first global extents | byte strides | box | dtype / modes |
| --- | --- | --- | --- | --- |
| Q | `(128, total_q*num_q_heads)` | `(256,)` | `(64,1)` plus four gathered row indices | BF16, interleave none, swizzle 128B, no L2 promotion, no OOB fill |
| K | `(64,128,2,num_pages*num_kv_heads)` | `(256,128,32768)` | `(64,64,1,1)` | BF16, same modes |
| V | `(64,128,2,num_pages*num_kv_heads)` | `(256,128,32768)` | `(64,64,1,1)` | BF16, same modes |

The planner produces six-int records `(head_kv, row_linear, q_begin, q_count,
batch, kv_block)`. Each packed reverse index stores the split slot in bits
24..31 and the batch-local query index in bits 0..23. The selected physical page
is `max(page_table[batch,max_pages,kv_block], 0)`; duplicate and negative page
entries are therefore intentional source behavior.

The live producer/reducer handshake is:

| object | allocation | live addressing |
| --- | --- | --- |
| partial O | uint8 `[4,total_q,num_q_heads,128]` | split-major full allocation |
| partial scale | float32 `[4,total_q,num_q_heads,4]` | only scalar prefix `[4,total_q,num_q_heads]` is accessed by both kernels |
| partial LSE | float32 `[4,total_q,num_q_heads]` | split-major full allocation |
| partial temperature LSE | same as partial LSE | ABI-only, untouched and unread |
| final O | BF16 `[total_q,num_q_heads,128]` | reducer row-major stores |
| final LSE / temperature LSE | float32 `[total_q,num_q_heads]` | optional; identical combined value |

## Producer pipeline inventory

| edge | slots / initial wait parity | producer | consumer | completion / release |
| --- | --- | --- | --- | --- |
| Q | two / empty waits `phase^1` | four Q-load warps, each 8192 expected bytes | warp 12 QK | q-full count 4; one elected tcgen05 commit releases q-empty |
| K | singleton / full phase 0 | warp 15 elected lane | warp 12 before first QK | 32768 expected bytes, no reuse edge |
| V | singleton / full phase 0 | warp 15 elected lane | warp 12 before first PV | 32768 expected bytes, no reuse edge |
| scores | two TMEM stage regions / empty `phase^1` | warp 12 QK | matching four softmax warps | elected tcgen05 s-full commit; 128 arrivals release s-empty |
| probabilities | two TMEM stage regions / empty `phase^1` | matching four softmax warps | warp 12 PV | 128 arrivals publish p-full; elected tcgen05 commit releases p-empty |
| output | two TMEM stage regions / empty `phase^1` | warp 12 PV | matching four softmax/quantize warps | elected tcgen05 o-full commit; 128 arrivals release o-empty |
| FP8 placeholders | three singleton barriers | none after initialization | none | no phase transition; exact source dead state |

## Source / sketch / PTX correspondence

| source lines | sketch region | PTX evidence |
| --- | --- | --- |
| producer 509-622 | ABI, arena, barrier/TMEM setup, register redistribution | file 1 `.loc` 509-620; 21 mbarrier init, allocation/relinquish, CTA fence, setmaxnreg |
| producer 623-1118 | even-group softmax, P publication, FP8 partial epilogue | `.loc` 716, 757, 812, 871, 876, 922, 925, 938, 944, 990, 993, 1006, 1012-1115 |
| producer 1119-1614 | odd-group twin | `.loc` 1212, 1253, 1308, 1367, 1372, 1418, 1421, 1434, 1440, 1486, 1489, 1502, 1508-1611 |
| producer 1615-1805 | four-warp alternating Q gather | `.loc` 1705-1800; 16 static gather4 sites |
| producer 1806-2232 | QK/PV software pipeline and TMEM deallocation | `.loc` 1892-2230; 40 static tcgen05 MMA sites and ten commit sites |
| producer 2233-2257 | compiled-empty transform branches | no emitted role-body opcode; only common source `.loc` before role dispatch remains |
| producer 2258-2315 | physical-page clamp and singleton K/V loads | `.loc` 2284, 2292-2313; eight rank-4 TMA sites |
| reducer 49-74 | row/cohort decode and split LSE/scale loads | file 1 `.loc` 53, 72-73 |
| reducer 75-132 | max/sum reductions, normalized/scaled weights, optional LSE stores | `.loc` 76-130; six butterfly shuffles, four indexed shuffles, three max sites |
| reducer 133-423 | four unrolled FP8 decode-and-FMA split blocks | `.loc` 144-420; eight u64 loads, 32 E4M3x2 conversions, 64 F16-to-F32 conversions, 64 FMAs |
| reducer 424-437 | BF16 conversion and final O stores | `.loc` 426-435; eight BF16x2 conversions and two vector stores |

## TIRx and verification contract

- The executable module imports the device language only as
  `import tirx_kernels.kern as K`; it uses no tile primitive, first-class
  layout, direct TVM script namespace, `K.cuda.func_call`, inline-CUDA escape,
  or modification beneath `tirx_kernels/kern/`.
- `get_kernel()` returns producer and reducer PrimFuncs in source launch order.
  Planning, TensorMap encoding, allocation, and JIT compilation occur before
  timing. The measured closure launches exactly this producer followed by this
  reducer on one stream.
- Correctness compares every live producer intermediate and every final output
  to the pinned source by integer view with `rtol=0`, `atol=0`; disabled/dead
  outputs retain poison sentinels. It also checks input immutability, guards,
  repeated determinism, negative/permuted/duplicate pages, causal boundaries,
  both LSE flags, and both planner seeds.
- All source and TIRx entries must declare PTX ISA 9.3 for GB300. Only
  `python -m tirx_kernels.bench_suite ... --with-references` supplies final
  performance evidence, and every frozen workload must satisfy the strict
  `source_time / tirx_time > 0.99` gate.

## Instruction-selection summary

The source export fixes rank-one dynamic SMEM, 512 TMEM columns, two Q/S/P/O
stages, singleton K/V tiles, gather4 Q TMA, four-part rank-4 K/V TMA, SS QK and
TS PV `tcgen05.mma`, rolled group loops, packed FTZ softmax arithmetic, E4M3
partial quantization, and the constant-four shuffle/FMA reducer. The exact
physical offsets, descriptor high words `0x40004040`, instruction descriptors
`0x08200490` and `0x08210490`, barrier counts, parities, and issue ordering—not
a layout object—select those instructions. Static counts in this sketch are raw
source PTX opcode sites with predicated instructions included; dynamic extents
are stated per operation occurrence.
