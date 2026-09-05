<!--
This file is a design sketch for a TIRx port of code from FlashInfer
(https://github.com/flashinfer-ai/flashinfer @ cc6e8794c49bf66172627bdb9742fcb17d18b839),
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# FlashInfer SM103a MSA Q1 BF16-query/FP8-KV paged decode sketch

This is a non-executable, operation-level transcription of
`csrc/blackwell_msa/sm103a/blackwell_msa_decode_q1_bf16_query_fp8_kv_xform2_paged.cu`
at FlashInfer commit `cc6e8794c49bf66172627bdb9742fcb17d18b839`. It freezes the
single-CTA, sixteen-warp schedule; the four-stage FP8-to-BF16 K/V ring; two
interleaved score/probability streams; online-softmax and correction protocol;
and the exact QK/PV `tcgen05.mma` issue order used by
`kernel_blackwell_batch_attention_msa_decode_q1_fp8_paged_xform2_v1`.

The executable source of truth will be
[`tirx_kernels/flashinfer/msa_ops/blackwell_msa_decode_q1_bf16_query_fp8_kv_xform2_paged_sm103.py`](../../tirx_kernels/flashinfer/msa_ops/blackwell_msa_decode_q1_bf16_query_fp8_kv_xform2_paged_sm103.py).
Its public domain is the exact promoted `q1_paged_xform2` route: 128 requests,
one query token per request, 64 BF16 query heads, four FP8-E4M3 KV heads,
head-dimension 128, 4096 physical 128-token pages, at most 32 logical pages per
request, sixteen selected logical blocks per request/KV-head, causal masking,
and implicit query offsets. Negative selected-block and physical-page entries,
partial causal pages, and arbitrary valid KV lengths in the fixed capacity are
included.

The writer export is
`.porting/blackwell_msa_decode_q1_bf16_query_fp8_kv_xform2_paged_sm103/source_export/writer/blackwell_msa_decode_q1_bf16_query_fp8_kv_xform2_paged.ptx`,
SHA256 `8272ec5138598a05540d68f0e642762432f6f861dcb6077099f0fd4d3c4c694f`.
CUDA nvcc 13.3.73 produced it with
`-std=c++17 -O3 -DNDEBUG -lineinfo -arch=compute_103a -ptx`; it has
`.version 9.3`, `.target sm_103a`, `.maxntid 512`, and 3349 `.loc`
records whose file 1 is the pinned CUDA translation unit. The companion cubin
has SHA256 `0cd47e48d11949396c03ea562e3c5d75aefb52a775aef99df9b1161bcfa27297`
and reports 128 registers/thread, a 128-byte stack frame, and no local
allocation. All instruction selections below are grounded in that export and
the line-mapped source. After an independent sketch-review PASS, this file is
immutable.

## Pipeline at a glance

| warps | role | work | publication / reuse edges |
| --- | --- | --- | --- |
| 0-3 | softmax stream 0 | rows 0-127 of S0, online max/sum, BF16 P0 | `s_full[0] -> corr_sig[0] -> p_full[0]`; final statistics through `corr_sig[0]` |
| 4-7 | softmax stream 1 | rows 0-127 of S1, online max/sum, BF16 P1 | `s_full[1] -> corr_sig[1] -> p_full[1]`; final statistics through `corr_sig[1]` |
| 8-11 | correction/output | rescale O0/O1 after every PV, merge both streams, normalize and store BF16 O plus FP32 natural-log LSE | `corr_sig[i] -> p_full[i]`; final `o_full[0]`; 128-thread `decode_done` |
| 12 | MMA | exact QK/PV chains over sixteen reverse-ordered selected pages | `q_full`, `kv_full[0:4]`, `p_full[0:2]` to `s_full`, `kv_empty[0:4]`, `o_full[0]`, and `q_empty` |
| 13 lane elected | Q and FP8 K/V TMA producer | Q once/work item; four K prefill pages; then sixteen V pages interleaved with K four pages ahead | `q_empty -> q_full`; `kv_empty[stage] -> kv_src_full[stage]` |
| 14-15 | FP8-to-BF16 producer auxiliary | 64-thread native E4M3x2-to-BF16 conversion into swizzled four-stage shared ring | `kv_src_full[stage] -> named barrier 10 -> kv_full[stage]` |

## Primitive vocabulary and storage rule

The implementation uses only `import tirx_kernels.kern as K`, scalar K
control flow, register arrays, `K.TensorMap`, one rank-one dynamic shared byte
arena, explicit byte offsets, and `K.ptx`. It uses no tile primitive, no
first-class layout object, no inline CUDA device call, no IR-check exemption,
and no modification beneath `tirx_kernels/kern/`.

```python
linear_smem(name, bytes, alignment)
reg_array(name, dtype, count)
smem_byte(base, offset)
tmem_column(base, column, row_band)
copy_g2s(tmap, x, y, z, smem_byte, completion)
copy_t2r(tmem_address, registers)
copy_r2t(registers, tmem_address)
copy_r2s(value, smem_byte)
copy_r2g(value, pointer, predicate)
mma_f16(tmem, a_descriptor, b_descriptor, instruction_descriptor, accumulate)
init / wait / expect_bytes / arrive / commit / fence / barrier
```

Descriptor and swizzle functions below are ordinary scalar integer expressions
that yield raw PTX operands or byte offsets; they never construct a layout
value. One movement or MMA operation denotes exactly one instruction or the
explicitly stated unrolled instruction family.

## Complete sketch

```python
@kernel(
    target="sm_103a",
    grid=(num_requests, num_kv_heads, 1),
    block=(512, 1, 1),
    min_blocks_per_sm=1,
    dynamic_smem_bytes=216704,
)
# instruction_selection: source ABI has three aligned 128-byte by-value
# TensorMaps, seven u64 pointers, four u32 scalars and one f32 scalar;
# .maxntid 512; .extern .shared .align 1024.
def blackwell_msa_decode_q1_bf16_query_fp8_kv_xform2_paged_sm103(
    Q_tmap, K_tmap, V_tmap,
    O, msa_lse,
    kv_indices, kv_indptr, task_kind, task_request, task_kv_head,
    num_requests, num_q_heads, num_kv_heads,
    softmax_scale_log2, msa_max_pages,
):
    tid = thread_id()
    warp = warp_uniform(tid // 32)
    lane = tid % 32
    work_first = block_x*num_kv_heads + block_y
    work_stride = grid_x*grid_y

    smem = linear_smem("smem", 216704, alignment=1024)
    smbase = shared_address(smem)

    # Barrier offsets and counts, in exact declaration order.
    Q_FULL, Q_EMPTY = 0, 8                         # 1 each
    KV_FULL = [16 + 8*s for s in range(8)]         # 1 each; active s=0..3
    KV_SRC_FULL = [80 + 8*s for s in range(4)]     # 1 each
    KV_EMPTY = [112 + 8*s for s in range(4)]       # 1 each
    S_FULL = [144, 152]                            # 1 each
    P_FULL = [160, 168]                            # 256 each
    CORR_SIG = [176, 184]                          # 128 each
    CORR_DONE = [192, 200]                         # 128 each, initialized but unused
    O_FULL = [208, 216]                            # 1 each; only index 0 active
    DECODE_DONE = 224                              # 128
    TMEM_ADDR = 232

    # Explicit shared byte regions.
    CORR = [1024, 1088]                            # 64 B each
    EXCH = [1152, 1408]                            # 256 B each
    QT = 1664                                      # 4096 B
    KV = 6144                                      # 4 * 32768 B, K/V alias
    P = [137216, 141312]                           # 4096 B each
    PAGE_INDICES = 145408                          # 4096 B
    ROW_MAX = 150144                               # 512 B
    ROW_SUM = 150656                               # 512 B
    KV_FP8 = 151168                                # 4 * 16384 B

    def kv_bf16_stage(s): return KV + s*32768
    def kv_fp8_stage(s): return KV_FP8 + s*16384
    def barrier(group, s=0): return smbase + group + s*8

    # P stores preserve the generated 128B-swizzled physical mapping.
    def probability_byte(stream, wg_tid, h):
        row = (wg_tid//64)*16 + h
        byte = row*128 + (wg_tid%64)*2
        return P[stream] + (byte ^ (((byte >> 7) & 7) << 4))

    # FP8 conversion destination preserves the generated MMA shared mapping.
    def converted_byte(off):
        elt = off*8
        row = ((elt%128)//64)*128 + elt//128
        byte = row*128 + ((elt%64)*16)//8
        return byte ^ ((row%8)*16)

    def qk_desc(stage):
        a0 = (((smbase + KV) >> 4) & 0x3fff) + stage*2048
        b0 = ((smbase + QT) >> 4) & 0x3fff
        return a0, b0

    def pv_desc(stage, stream):
        a0 = ((((smbase + KV) >> 4) & 0x3fff) | 0x04000000) + stage*2048
        b0 = (((smbase + P[stream]) >> 4) & 0x3fff) | 0x00800000
        return a0, b0

    def issue_qk(stage, dst, overwrite):
        alo, blo = qk_desc(stage)
        leader = elected_lane()
        # Eight elected cta_group::1 f16 MMAs. Descriptor high halves are
        # 0x40004040, i-desc is 0x08040490, and input-D is false only at
        # the first site when overwrite is true. The source computes one
        # election and reuses that predicate for all eight MMA sites.
        for site, (da, db) in enumerate((
            (0,0), (2,2), (4,4), (6,6),
            (1024,128), (1026,130), (1028,132), (1030,134),
        )):
            mma_f16(dst, join(0x40004040, alo+da),
                    join(0x40004040, blo+db), 0x08040490,
                    accumulate=(not overwrite) or site != 0, pred=leader)
        # instruction_selection: eight predicated
        # tcgen05.mma.cta_group::1.kind::f16, each guarded by elect.sync;
        # exact source low-descriptor increments 2/2,2/2,2/2,1018/122,
        # then 2/2,2/2,2/2.

    def issue_pv(stage, stream, dst, overwrite):
        alo, blo = pv_desc(stage, stream)
        leader = elected_lane()
        for site, (da, db) in enumerate((
            (0,0), (128,2), (256,4), (384,6),
            (512,128), (640,130), (768,132), (896,134),
        )):
            mma_f16(dst, join(0x40004040, alo+da),
                    join(0x40004040, blo+db), 0x08048490,
                    accumulate=(not overwrite) or site != 0, pred=leader)
        # instruction_selection: eight predicated elected
        # tcgen05.mma.cta_group::1.kind::f16; exact low-descriptor
        # increments 128/2,128/2,128/2,128/122, then 128/2 three times.

    # One elected lane of warp 0 initializes all 29 source barriers.
    if warp == 0 and elected_lane():
        init(Q_FULL, 1); init(Q_EMPTY, 1)
        for s in static_range(8): init(KV_FULL[s], 1)
        for s in static_range(4): init(KV_SRC_FULL[s], 1)
        for s in static_range(4): init(KV_EMPTY[s], 1)
        for s in static_range(2): init(S_FULL[s], 1)
        for s in static_range(2): init(P_FULL[s], 256)
        for s in static_range(2): init(CORR_SIG[s], 128)
        for s in static_range(2): init(CORR_DONE[s], 128)
        for s in static_range(2): init(O_FULL[s], 1)
        init(DECODE_DONE, 128)
        fence_mbarrier_init_release_cluster()
    # instruction_selection: 29 mbarrier.init.shared::cta.b64 sites and one
    # fence.mbarrier_init.release.cluster from one elected lane.
    warp_sync()

    # Warp 0 collectively allocates 128 TMEM columns and immediately
    # relinquishes the allocation permit. All 512 threads then rendezvous.
    if warp == 0:
        allocate_tmem_cta_group_1(smbase + TMEM_ADDR, 128)
        relinquish_tmem_cta_group_1()
    cta_barrier()
    fence_tcgen05_after_thread_sync()
    taddr = load_shared_u32(TMEM_ADDR)
    S_TMEM = [taddr + 0, taddr + 16]
    O_TMEM = [taddr + 32, taddr + 48]
    STATS_TMEM = [taddr + 64, taddr + 80]

    # Source register redistribution order precedes all role dispatch.
    if 12 <= warp <= 15:
        setmaxnreg_dec(48)
    # instruction_selection: setmaxnreg.dec.sync.aligned.u32 48

    # Warps 0-7: two independent 128-thread online-softmax streams.
    if warp <= 7:
        setmaxnreg_inc(192)
        stream = 1 if warp >= 4 else 0
        warp_in_wg = warp % 4
        wg_tid = warp_in_wg*32 + lane
        row_band = warp_in_wg*32
        my_s = S_TMEM[stream] + (row_band << 16)
        my_stats = STATS_TMEM[stream] + (row_band << 16)
        s_phase = 0
        for work in range(work_first, num_requests*num_kv_heads, work_stride):
            row_state = warp*16
            for h in static_range(16):
                store_shared_f32(ROW_MAX + 4*(row_state+h), -inf)
                store_shared_f32(ROW_SUM + 4*(row_state+h), 0.0)

            for pair in rolled_range(8):
                wait_acquire_cta(S_FULL[stream], s_phase, ticks=0x989680)
                s_phase ^= 1
                score = reg_array("score", f32, 16)
                copy_t2r_x16(my_s, score)
                # instruction_selection:
                # tcgen05.ld.sync.aligned.32x32b.x16.b32 once/pair/thread.
                valid_cols = load_shared_i32(PAGE_INDICES + 4*(pair*2+stream))
                token = wg_tid
                if token >= valid_cols:
                    for h in static_range(16): score[h] = -inf

                # Warp max over each of 16 query rows, then 128-token max
                # through the stream-local exchange array.
                for h in static_range(16):
                    for delta in (16, 8, 4, 2, 1):
                        score[h] = max_noftz(score[h], shfl_xor(score[h], delta))
                if lane < 16:
                    store_shared_f32(EXCH[stream] + 4*(warp_in_wg*16+lane), score[lane])
                named_barrier(8+stream, 128)
                if lane < 16:
                    tile_max_lane = max_noftz(
                        max_noftz(load_exch(0,lane), load_exch(1,lane)),
                        max_noftz(load_exch(2,lane), load_exch(3,lane)))
                tile_max = [shfl_idx(tile_max_lane, h) for h in static_range(16)]

                acc_scale = reg_array("acc_scale", f32, 16)
                for h in static_range(16):
                    old_max = load_shared_f32(ROW_MAX + 4*(row_state+h))
                    new_max = max_noftz(old_max, tile_max[h])
                    store_shared_f32(ROW_MAX + 4*(row_state+h), new_max)
                    delta = softmax_scale_log2*(old_max-new_max)
                    acc_scale[h] = select(old_max > -inf, exp2_approx_ftz(delta), 1.0)

                copy_r2t_x16(acc_scale, my_stats)
                wait_tmem_stores()
                fence_async_shared_cta()
                arrive_release_cta(CORR_SIG[stream])
                # instruction_selection: tcgen05.st x16, tcgen05.wait::st,
                # fence.proxy.async.shared::cta, then one mbarrier arrival/thread.

                copy_t2r_x16(my_s, score)
                if token >= valid_cols:
                    for h in static_range(16): score[h] = -inf
                exp_value = reg_array("exp_value", f32, 16)
                for h in static_range(16):
                    new_max = load_shared_f32(ROW_MAX + 4*(row_state+h))
                    safe_max = select(new_max == -inf, 0.0, new_max)
                    exp_value[h] = exp2_approx_ftz(
                        score[h]*softmax_scale_log2 - safe_max*softmax_scale_log2)
                    bf16_bits = cvt_rn_bf16(exp_value[h])
                    copy_r2s(bf16_bits, probability_byte(stream, wg_tid, h))
                for h in static_range(16):
                    for delta in (16, 8, 4, 2, 1):
                        exp_value[h] += shfl_xor(exp_value[h], delta)
                    old_sum = load_shared_f32(ROW_SUM + 4*(row_state+h))
                    store_shared_f32(
                        ROW_SUM + 4*(row_state+h),
                        fma_rn(old_sum, acc_scale[h], exp_value[h]))
                fence_async()
                arrive_release_cta(P_FULL[stream])
                # instruction_selection: ex2.approx.ftz.f32, cvt.rn.bf16.f32,
                # scalar st.shared.b16, shfl.sync.bfly reductions,
                # fence.proxy.async, then one p_full arrival/thread.

            # Reduce each stream's four warp-local sums and publish sum/max.
            named_barrier(8+stream, 128)
            if lane < 16:
                store_exch(warp_in_wg, lane, load_shared_f32(ROW_SUM+4*(row_state+lane)))
            named_barrier(8+stream, 128)
            if lane < 16:
                sum_w0 = load_exch(0, lane)
                sum_w1 = load_exch(1, lane)
                sum_w2 = load_exch(2, lane)
                sum_w3 = load_exch(3, lane)
                total_sum_lane = ((sum_w0 + sum_w1) + sum_w2) + sum_w3
            named_barrier(8+stream, 128)
            if warp_in_wg == 0 and lane < 16:
                store_shared_f32(CORR[stream] + 4*lane, total_sum_lane)
                store_shared_f32(EXCH[stream] + 4*lane,
                                 load_shared_f32(ROW_MAX+4*(row_state+lane)))
            named_barrier(8+stream, 128)
            arrive_release_cta(CORR_SIG[stream])

    # Warps 8-11: rescale partial outputs, merge streams, normalize and store.
    if 8 <= warp <= 11:
        setmaxnreg_dec(80)
        row_band = (warp%4)*32
        corr_row = row_band << 16
        d_idx = row_band + lane
        corr_phase = [0, 0]
        o_phase = 0
        for work in range(work_first, num_requests*num_kv_heads, work_stride):
            request = work // num_kv_heads
            kv_head = work % num_kv_heads
            group_size = num_q_heads // num_kv_heads

            for pair in rolled_range(8):
                wait_acquire_cta(CORR_SIG[0], corr_phase[0], ticks=0x989680)
                corr_phase[0] ^= 1
                fence_tcgen05_after_thread_sync()
                scale0 = reg_array("scale0", f32, 16)
                partial0 = reg_array("partial0", f32, 16)
                copy_t2r_x16(STATS_TMEM[0] + corr_row, scale0)
                copy_t2r_x16(O_TMEM[0] + corr_row, partial0)
                for h in static_range(16): partial0[h] *= scale0[h]
                copy_r2t_x16(partial0, O_TMEM[0] + corr_row)
                wait_tmem_stores()
                arrive_release_cta(P_FULL[0])

                wait_acquire_cta(CORR_SIG[1], corr_phase[1], ticks=0x989680)
                corr_phase[1] ^= 1
                fence_tcgen05_after_thread_sync()
                scale1 = reg_array("scale1", f32, 16)
                partial1 = reg_array("partial1", f32, 16)
                copy_t2r_x16(STATS_TMEM[1] + corr_row, scale1)
                copy_t2r_x16(O_TMEM[1] + corr_row, partial1)
                for h in static_range(16): partial1[h] *= scale1[h]
                copy_r2t_x16(partial1, O_TMEM[1] + corr_row)
                wait_tmem_stores()
                arrive_release_cta(P_FULL[1])

            # Consume the terminal statistics publication from both streams.
            wait_acquire_cta(CORR_SIG[0], corr_phase[0], ticks=0x989680)
            corr_phase[0] ^= 1
            wait_acquire_cta(CORR_SIG[1], corr_phase[1], ticks=0x989680)
            corr_phase[1] ^= 1
            for h in static_range(16):
                max0 = shfl_idx(load_shared_f32(EXCH[0]+4*h), h)
                max1 = shfl_idx(load_shared_f32(EXCH[1]+4*h), h)
                sum0 = shfl_idx(load_shared_f32(CORR[0]+4*h), h)
                sum1 = shfl_idx(load_shared_f32(CORR[1]+4*h), h)
                final_max[h] = max_noftz(max0, max1)
                d0 = select(max0 == -inf, 0.0,
                            softmax_scale_log2*(max0-final_max[h]))
                d1 = select(max1 == -inf, 0.0,
                            softmax_scale_log2*(max1-final_max[h]))
                merge_scale0[h] = exp2_approx_ftz(d0)
                merge_scale1[h] = exp2_approx_ftz(d1)
                sum1_scaled = mul_rn(sum1, merge_scale1[h])
                final_sum[h] = fma_rn(sum0, merge_scale0[h], sum1_scaled)

            wait_acquire_cta(O_FULL[0], o_phase, ticks=0x989680)
            o_phase ^= 1
            fence_tcgen05_after_thread_sync()
            inv_sum = reg_array("inv_sum", f32, 16)
            # Preserve the source instruction schedule: materialize all 16
            # reciprocal candidates before either final O fragment load.
            for h in static_range(16):
                inv_sum[h] = select(final_sum[h] > 0.0,
                                    rcp_approx_ftz(final_sum[h]), 0.0)
            out0 = reg_array("out0", f32, 16)
            out1 = reg_array("out1", f32, 16)
            copy_t2r_x16(O_TMEM[0] + corr_row, out0)
            copy_t2r_x16(O_TMEM[1] + corr_row, out1)
            for h in static_range(16):
                if h < group_size:
                    out1_scaled = mul_rn(out1[h], merge_scale1[h])
                    merged = fma_rn(out0[h], merge_scale0[h], out1_scaled)
                    q_row = request*num_q_heads + kv_head*group_size + h
                    if d_idx == 0:
                        natural_lse = -inf
                        if final_sum[h] > 0.0:
                            max_scaled = mul_rn(final_max[h], softmax_scale_log2)
                            max_ln = mul_rn(max_scaled, f32_bits(0x3F317218))
                            natural_lse = fma_rn(
                                log2_approx_ftz(final_sum[h]),
                                f32_bits(0x3F317218), max_ln)
                        store_global_f32(msa_lse[q_row], natural_lse)
                    store_global_bf16(O[q_row*128+d_idx],
                                      cvt_rn_bf16(merged*inv_sum[h]))
            arrive_release_cta(DECODE_DONE)
            # instruction_selection: approximate FTZ exp2/rcp/log2, scalar
            # st.global.f32 for lane d_idx==0, scalar st.global.b16 for O,
            # and one decode_done arrival/thread.

    # Warp 12: source-order MMA state machine.
    if warp == 12:
        q_phase = 0
        p_phase = [0, 0]
        done_phase = 0
        for work in range(work_first, num_requests*num_kv_heads, work_stride):
            inst0_stage = 0
            first_pv = [True, True]
            wait_acquire_cta(Q_FULL, q_phase, ticks=0x989680)
            q_phase ^= 1

            wait_acquire_cta(KV_FULL[0], 0, ticks=0x989680)
            issue_qk(0, S_TMEM[0], overwrite=True)
            commit_elected(S_FULL[0])
            commit_elected(KV_EMPTY[0])

            for pair in rolled_range(7):
                s0 = inst0_stage
                s1 = (inst0_stage+1)%4
                s0_next = (inst0_stage+2)%4

                wait_acquire_cta(KV_FULL[s1], 0, ticks=0x989680)
                issue_qk(s1, S_TMEM[1], overwrite=True)
                commit_elected(S_FULL[1])
                commit_elected(KV_EMPTY[s1])

                wait_acquire_cta(KV_FULL[s0], 1, ticks=0x989680)
                wait_acquire_cta(P_FULL[0], p_phase[0], ticks=0x989680)
                p_phase[0] ^= 1
                fence_tcgen05_after_thread_sync()
                issue_pv(s0, 0, O_TMEM[0], overwrite=first_pv[0])
                first_pv[0] = False
                commit_elected(KV_EMPTY[s0])

                wait_acquire_cta(KV_FULL[s0_next], 0, ticks=0x989680)
                issue_qk(s0_next, S_TMEM[0], overwrite=True)
                commit_elected(S_FULL[0])
                commit_elected(KV_EMPTY[s0_next])

                wait_acquire_cta(KV_FULL[s1], 1, ticks=0x989680)
                wait_acquire_cta(P_FULL[1], p_phase[1], ticks=0x989680)
                p_phase[1] ^= 1
                fence_tcgen05_after_thread_sync()
                issue_pv(s1, 1, O_TMEM[1], overwrite=first_pv[1])
                first_pv[1] = False
                commit_elected(KV_EMPTY[s1])
                inst0_stage = s0_next

            s0_last = inst0_stage
            s1_last = (inst0_stage+1)%4
            wait_acquire_cta(KV_FULL[s1_last], 0, ticks=0x989680)
            issue_qk(s1_last, S_TMEM[1], overwrite=True)
            final_qk_commit_leader = elected_lane()
            commit(S_FULL[1], pred=final_qk_commit_leader)
            commit(Q_EMPTY, pred=final_qk_commit_leader)
            commit_elected(KV_EMPTY[s1_last])

            wait_acquire_cta(KV_FULL[s0_last], 1, ticks=0x989680)
            wait_acquire_cta(P_FULL[0], p_phase[0], ticks=0x989680)
            p_phase[0] ^= 1
            fence_tcgen05_after_thread_sync()
            issue_pv(s0_last, 0, O_TMEM[0], overwrite=first_pv[0])
            commit_elected(KV_EMPTY[s0_last])

            wait_acquire_cta(KV_FULL[s1_last], 1, ticks=0x989680)
            wait_acquire_cta(P_FULL[1], p_phase[1], ticks=0x989680)
            p_phase[1] ^= 1
            fence_tcgen05_after_thread_sync()
            issue_pv(s1_last, 1, O_TMEM[1], overwrite=first_pv[1])
            commit_elected(KV_EMPTY[s1_last])
            commit_elected(O_FULL[0])
            wait_acquire_cta(DECODE_DONE, done_phase, ticks=0x989680)
            done_phase ^= 1
        # instruction_selection: source-order elected tcgen05 MMA and commit
        # instructions; no synthetic barrier, wait, or reordering.

    def page_info(request, kv_head, selected_slot):
        selected_block = load_global_i32(
            task_kind[(kv_head*num_requests+request)*16+selected_slot])
        kv_len = load_global_i32(task_kv_head[request])
        valid = 0
        physical_page = 0
        if selected_block >= 0:
            block_start = selected_block*128
            valid = clamp(kv_len-block_start, 0, 128)
            query_position = kv_len-1
            valid = clamp(min(valid, query_position-block_start+1), 0, 128)
            physical_page = load_global_i32(
                kv_indices[request*msa_max_pages+selected_block])
            if physical_page < 0:
                valid = 0
                physical_page = 0
        page_head = physical_page*num_kv_heads + kv_head
        return valid, page_head

    def issue_fp8_page(tmap, physical_stage, page_head, completion):
        expect_bytes(completion, 16384)
        copy_g2s(tmap, 0, 0,  page_head,
                 kv_fp8_stage(physical_stage), completion)
        copy_g2s(tmap, 0, 64, page_head,
                 kv_fp8_stage(physical_stage)+8192, completion)
        # instruction_selection: two
        # cp.async.bulk.tensor.3d.shared::cta.global
        # .mbarrier::complete_tx::bytes instructions, each a 128x64x1 box.

    # Warp 13: all lanes execute the Q-empty wait. A first election guards
    # only the Q TMA, and an independent second election guards the complete
    # reverse-page K/V producer schedule.
    if warp == 13:
        q_empty_phase = 1
        for work in range(work_first, num_requests*num_kv_heads, work_stride):
            request = work // num_kv_heads
            kv_head = work % num_kv_heads
            group_size = num_q_heads // num_kv_heads
            wait_acquire_cta(Q_EMPTY, q_empty_phase, ticks=0x989680)
            q_empty_phase ^= 1
            q_tma_leader = elected_lane()
            if q_tma_leader:
                q_row = request*num_q_heads + kv_head*group_size
                expect_bytes(Q_FULL, 4096)
                copy_g2s(Q_tmap, 0, q_row, 0, QT, Q_FULL)
                # instruction_selection: one 3D TMA 64x16x2 BF16 box.

            kv_producer_leader = elected_lane()
            if kv_producer_leader:
                # Prefill K selected slots 15,14,13,12 into stages 0,1,2,3.
                for ni in static_range(4):
                    valid, page_head = page_info(request, kv_head, 15-ni)
                    store_shared_i32(PAGE_INDICES+4*ni, valid)
                    fence_async_shared_cta()
                    wait_acquire_cta(KV_EMPTY[ni], 1, ticks=0x989680)
                    issue_fp8_page(K_tmap, ni, page_head, KV_SRC_FULL[ni])

                # Each stage next carries V for slot 15-ni and then K for slot
                # 11-ni. Parity 0 of KV_EMPTY protects the V write; parity 1
                # protects the following K write.
                for ni in rolled_range(16):
                    stage = ni%4
                    _valid, page_head = page_info(request, kv_head, 15-ni)
                    wait_acquire_cta(KV_EMPTY[stage], 0, ticks=0x989680)
                    issue_fp8_page(V_tmap, stage, page_head, KV_SRC_FULL[stage])
                    next_ni = ni+4
                    if next_ni < 16:
                        next_valid, next_page_head = page_info(
                            request, kv_head, 15-next_ni)
                        store_shared_i32(PAGE_INDICES+4*next_ni, next_valid)
                        fence_async_shared_cta()
                        wait_acquire_cta(KV_EMPTY[stage], 1, ticks=0x989680)
                        issue_fp8_page(K_tmap, stage, next_page_head,
                                       KV_SRC_FULL[stage])

    def convert_fp8_stage(stage):
        aux_tid = tid - 14*32
        for off in unrolled_range(aux_tid, 2048, 64, factor=4):
            src64 = load_shared_u64(kv_fp8_stage(stage)+off*8)
            packed = reg_array("packed", u32, 4)
            for cv in static_range(4):
                packed[cv] = cvt_rn_bf16x2_e4m3x2((src64>>(16*cv)) & 0xffff)
            store_shared_v4_u32(kv_bf16_stage(stage)+converted_byte(off),
                                packed[0], packed[1], packed[2], packed[3])
        fence_async_shared_cta()
        # instruction_selection: native cvt.rn.bf16x2.e4m3x2 four times per
        # 64-bit source word and one st.shared.v4.u32 per word; no f16 fallback.

    # Warps 14-15: 64-thread FP8 conversion, exact K/V phase sequence.
    if 14 <= warp <= 15:
        for stage in static_range(4):
            wait_acquire_cta(KV_SRC_FULL[stage], 0, ticks=0x989680)
            fence_async_shared_cta()
            convert_fp8_stage(stage)
            named_barrier(10, 64)
            if warp == 14 and elected_lane():
                arrive_release_cta(KV_FULL[stage])

        for ni in rolled_range(16):
            stage = ni%4
            wait_acquire_cta(KV_SRC_FULL[stage], 1, ticks=0x989680)
            fence_async_shared_cta()
            convert_fp8_stage(stage)                  # V
            named_barrier(10, 64)
            if warp == 14 and elected_lane():
                arrive_release_cta(KV_FULL[stage])    # KV_FULL parity 1
            if ni+4 < 16:
                wait_acquire_cta(KV_SRC_FULL[stage], 0, ticks=0x989680)
                fence_async_shared_cta()
                convert_fp8_stage(stage)              # next K
                named_barrier(10, 64)
                if warp == 14 and elected_lane():
                    arrive_release_cta(KV_FULL[stage])# KV_FULL parity 0

    # Source cleanup: all roles rendezvous only after their persistent loops.
    cta_barrier()
    if warp == 0:
        deallocate_tmem_cta_group_1(load_shared_u32(TMEM_ADDR), 128)
    # instruction_selection: bar.sync 0 then
    # tcgen05.dealloc.cta_group::1.sync.aligned.b32 by all lanes of warp 0.
```

## TensorMap fields and source ABI

| map | fastest-first global dimensions | byte strides for dimensions 1+ | box | format / swizzle / OOB |
| --- | --- | --- | --- | --- |
| Q | `(64, total_Q_heads, 2)` where `total_Q_heads=Q.numel()/128` | `(Q.stride(-2)*2, 128)` | `(64,16,2)` | BF16 / 128 B / none |
| K | `(128, page_tokens, physical_pages*num_kv_heads)` | `(128, page_tokens*128)` | `(128,64,1)` | UINT8 / none / none |
| V | same as K | same as K | `(128,64,1)` | UINT8 / none / none |

The host encodes these three TensorMaps before launch and passes every remaining
argument in the source order. `kv_indptr` and `task_request` are retained in
the ABI although this specialization does not read them. The fixed data shape
used for correctness and benchmarking is Q `[128,64,128]`, K/V
`[4096,4,128,128]`, O `[128,64,128]`, LSE `[128,64]`, page table
`[128,32]`, selected blocks `[4,128,16]`, and KV lengths `[128]`.

## Barrier and phase inventory

| edge | slots / count | first wait phase | subsequent phase behavior |
| --- | --- | --- | --- |
| Q reuse/full | one each / 1 | producer waits Q_EMPTY 1; MMA waits Q_FULL 0 | both toggle once per work item |
| FP8 source | 4 / 1 | auxiliary waits KV_SRC_FULL 0 for prefilled K | parity 1 carries V, parity 0 carries the next K |
| BF16 K/V full | 8 allocated / 1; indices 0-3 active | MMA waits 0 for K | parity 1 carries V, then returns to 0 for later K |
| BF16 K/V empty | 4 / 1 | producer prefill waits 1 | MMA commits after every QK/PV; producer alternates 0 for V and 1 for next K |
| score full | 2 / 1 | softmax waits 0 | one publication per page/stream, toggling across eight pairs |
| probability full | 2 / 256 | MMA waits 0 | 128 softmax arrivals plus 128 correction arrivals complete each phase |
| correction signal | 2 / 128 | correction waits 0 | eight per-page scale publications plus one terminal statistics publication |
| output full | 2 / 1; index 0 active | correction waits 0 | one completed O0/O1 pair per work item |
| decode done | one / 128 | MMA waits 0 | one correction arrival/thread per work item |

`CORR_DONE[0:2]`, `O_FULL[1]`, and `KV_FULL[4:8]` are initialized because
they are fields of the generated source protocol, but this exact specialization
does not reference them after initialization.

## Instruction-selection summary

| source decision | required PTX consequence |
| --- | --- |
| fixed 16-warp CTA, 216704-byte arena | `.maxntid 512`, aligned dynamic shared declaration, no cluster directives |
| role register redistribution | warp 0-7 `setmaxnreg.inc ... 192`; warp 8-11 `dec ... 80`; warp 12-15 `dec ... 48` |
| Q/K/V movement | rank-3 `cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes` |
| source wait helpers | acquire-CTA parity waits with `0x989680` time hint in retry loops |
| source QK/PV microkernels | elected `tcgen05.mma.cta_group::1.kind::f16`, eight sites per tile with exact descriptor steps and i-descs `0x08040490` / `0x08048490` |
| score/output/stat fragments | `tcgen05.ld/st.sync.aligned.32x32b.x16.b32`, explicit wait/fence placement |
| FP8 conversion | direct `cvt.rn.bf16x2.e4m3x2`, v4 shared stores, named barrier 10 |
| online softmax | non-FTZ `max.f32`; `ex2.approx.ftz.f32`; shuffle butterfly reductions with explicit left-associated four-warp sum; running sum uses `fma.rn.f32(old_sum, acc_scale, exp)`; RN BF16 probability stores |
| final normalization | sum merge is `mul.rn.f32(sum1, scale1)` then `fma.rn.f32(sum0, scale0, product)`; all 16 `rcp.approx.ftz.f32` precede final O loads; O merge uses the same mul-then-fma ordering; LSE uses two `mul.rn.f32` plus `fma.rn.f32(log2_sum, 0f3F317218, max_ln)`; RN BF16 output conversion |
| lifecycle | cta-group-1 TMEM allocate/relinquish before roles, CTA sync/fence, final CTA sync and TMEM deallocate |

## Correctness and benchmark contract

- Deterministic fixtures exercise minimum valid length, all 16 selected blocks,
  ragged partial pages, scattered physical pages, negative selected blocks,
  negative page-table entries, duplicate selections, all-invalid rows, and
  BF16/FP8 values that stress ties, cancellation, extrema, and rounding.
- Source and TIRx receive byte-identical Q/K/V and metadata. O and LSE are
  poison-filled before launch; allocation guards surround every writable and
  input buffer; inputs are hashed/compared after launch.
- The acceptance comparison is bitwise equality for the complete BF16 O and
  FP32 LSE tensors, including identical signed infinities for all-invalid rows.
  NaNs, remaining poison, guard modification, nondeterministic repeated output,
  and any differing bit fail. Absolute/relative errors are diagnostics only and
  do not relax the comparison.
- The source reference is loaded lazily from the exact GB300 checkout and must
  verify commit/source hashes. Neither correctness nor timing may substitute a
  PyTorch reconstruction for the CUDA source.
- `prepare_bench` compiles before timing and returns a prepared benchmark.
  `run_gpu` allocates once, performs one validation launch per implementation,
  and gives the canonical Proton timer one source launch and one TIRx launch per
  closure.
- Final performance authority is only `bench_suite`, with external reference
  enabled, Proton, five rounds, and one-second cooldown. Every benchmark config
  must have `tirx/source > 0.99`; focused iterations target the current worst
  configs, and the full suite runs only at the initial gate, meaningful
  milestones, and final acceptance.
- The source and TIRx artifacts must both report PTX ISA 9.3 and target
  `sm_103a` under CUDA 13.3. No other timing API is evidence for acceptance.
