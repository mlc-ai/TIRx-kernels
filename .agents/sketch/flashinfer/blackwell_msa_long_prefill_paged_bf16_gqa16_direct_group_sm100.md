<!--
This file is a design sketch for a TIRx port of FlashInfer
(https://github.com/flashinfer-ai/flashinfer @ 9f5051736e9fd5cab41c06118a7d4b5c1de23a6d),
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright TIRx authors
-->

# FlashInfer long paged BF16 GQA16 direct-group MSA prefill SM100 sketch

This is the non-executable operation sketch for
[`tirx_kernels/flashinfer/msa_ops/blackwell_msa_long_prefill_paged_bf16_gqa16_direct_group_sm100.py`](../../../tirx_kernels/flashinfer/msa_ops/blackwell_msa_long_prefill_paged_bf16_gqa16_direct_group_sm100.py).
It freezes
`sm100a/blackwell_msa_long_prefill_paged_bf16_gqa16_direct_group_sm100.cu::kernel_minimax_sparse_reverse_prefill_paged_bf16_gqa4_qload4_nobar_sm100`
at FlashInfer `9f5051736e9fd5cab41c06118a7d4b5c1de23a6d`.
Source SHA-256 is
`95751076c63d9a875c140013fba44851180eecb7454434e8ad8e232522c5983b`;
binding SHA-256 is
`6ec4f0ebe1912a5469318155fed11d1d95e0dc1e6a5959297455bcd4c52b8835`.

The optimized line-info writer export is
`.porting/blackwell_msa_long_prefill_paged_bf16_gqa16_direct_group_sm100/source_export/source.ptx`,
SHA-256 `3a291392a35e8ac3bb167d21459d400a36c80e2a17e949d2d464fc280ef74485`.
CUDA 13.2.51 emitted PTX 9.2 for `sm_100a`, 1,726 `.loc` records,
`.maxntid 512`, `.minnctapersm 1`, and `.extern .shared .align 1024`.
Static entry counts are 23 mbarrier initializations, 40 tcgen05 MMAs, ten
TMEM loads, six TMEM stores, twelve rank-4 TMAs, four register-budget
instructions, and one dependent-launch instruction. The normal tvm-ffi module
was also built and launched on GB200 through the production route.

After the independent sketch reviewer returns PASS, this file is immutable.

## Scope and representation contract

The specialization is paged BF16 Q/K/V, D=128, page=128, GQA16, TopK16,
batch one, causal, derived Q offset, query length at least 8,192, no requested
`m64` schedule, and page-table width exactly 8,192 on SM100a. It writes forward
partials: E4M3 partial-O bits, four BF16 dequant scales per partial row, FP32
partial LSE, and optional identical temperature partial LSE. Reduction is a
separate source kernel and is not timed or ported here.

The executable imports the device language only as
`import tirx_kernels.kern as K`. It uses scalar K control flow, opaque
TensorMaps, one-dimensional shared/register allocations, and supported
`K.ptx[...]` instructions. It uses no tile primitive, first-class layout or
mapping value, `layout=` argument, direct TVM script namespace,
`K.cuda.func_call`, inline-CUDA function-call exemption, or change under
`tirx_kernels/kern/`.

The seven group-boundary scalars, `work_capacity`, and `num_work_items` are
retained in the ABI but device-dead. Grid x is supplied as `num_work_items`.
The source hard-codes local `single_split=0`; therefore `out` is device-dead.
`lse_temperature_scale` is device-dead and enabled temperature LSE is bitwise
identical to partial LSE.

## Roles and topology

All branches are independent and appear in source order after setup.

| warps | role | work |
| --- | --- | --- |
| 0-3 | softmax-even | One thread per logical row of even query groups; S-to-register, mask/max, P production, O-to-register quantization and partial stores. |
| 4-7 | softmax-odd | The complete same program for odd query groups, with stage-1 barriers/TMEM. |
| 8-11 | qload | Each elected warp lane loads two scheduler edges; four warps jointly produce each 32-KiB Q stage. |
| 12 | MMA | Sole QK/PV tcgen05 issuer, stage interleave, drain, and TMEM deallocation. |
| 13-14 | transform | Source metadata decode only; optimized specialization emits no transform data operation. |
| 15 | load | Elected dependent-launch signal and one physical K/V page load. |

Register budgets are exact: warps 12-15 first decrease to 48; warps 0-3 and
4-7 independently increase to 200; warps 8-11 decrease to 64.

## ABI, TensorMaps, and linear storage

```text
Qmap, Kmap, Vmap,
i32* scheduler_metadata, i32* k2q_row_ptr, i32* k2q_qsplit_indices,
u8* partial_o, bf16* partial_scale, f32* partial_lse,
f32* partial_temperature_lse, bf16* out,
i32* cu_seqlens_q, i32* cu_seqlens_k, i32* q_offsets,
i32* kv_lens, i32* page_table,
i32 segment_end_128, segment_end_64, segment_end_32, segment_end_16,
i32 segment_end_8, segment_end_4, segment_end_2,
i32 total_q, i32 Hq, i32 Hkv, i32 total_rows, i32 nnz_per_head,
i32 work_capacity, i32 num_work_items, i32 topk, i32 max_pages,
i32 causal, i32 derive_q_offset, f32 softmax_scale_log2,
f32 lse_temperature_scale, i32 return_temperature_lse
```

Launch is `(grid=num_work_items, block=512)`, minimum one CTA/SM, 148,480
dynamic shared bytes, no cluster. Each map is a by-value, 64-byte-aligned,
128-byte parameter. All maps are BF16 rank four, element strides all one,
no interleave, 128-byte swizzle, no L2 promotion, and no OOB fill.

| map | fastest-first dimensions | byte strides | box |
| --- | --- | --- | --- |
| Q | `(64,Hq,2,total_q)` | `(256,128,Hq*256)` | `(64,16,1,1)` |
| K/V | `(64,128,2,physical_pages*Hkv)` | `(256,128,32768)` | `(64,64,1,1)` |

One linear shared arena contains:

| bytes | object |
| --- | --- |
| `[0,184)` | 23 eight-byte mbarriers |
| `[184,188)` | volatile TMEM mailbox |
| `[188,1024)` | unused alignment padding |
| `[1024,66560)` | Q stage 0/1, 32,768 bytes each |
| `[66560,99328)` | K tile, 32,768 bytes |
| `[99328,132096)` | V tile, aliased by source `v_convert_smem` |
| `[132096,148480)` | source-reserved, unused FP8 scratch |

TMEM score stages are columns 0 and 128; BF16 P overwrites columns 64 and
192. Output stages are columns 256 and 384. A logical row owned by warp-local
row origin `32*w` is addressed by `(32*w)<<16` in the tcgen05 address.

| barrier | offsets | initial count |
| --- | --- | ---: |
| `q_full[2]` | 0,8 | 4 |
| `q_empty[2]` | 16,24 | 1 |
| `k_full`, `v_full` | 32,40 | 1 |
| unused `fp8_k_full`, `fp8_v_full`, `fp8_empty` | 48,56,64 | 1 |
| `s_full[2]` | 72,80 | 1 |
| `s_empty[2]` | 88,96 | 128 |
| `p_full[2]` | 104,112 | 128 |
| `p_full_2[2]` | 120,128 | 128 |
| `p_empty[2]` | 136,144 | 1 |
| `o_full[2]` | 152,160 | 1 |
| `o_empty[2]` | 168,176 | 128 |

## Shared scalar decode

Each role independently reads scheduler row
`6*block_id_x + [0:6] = (head_kv,row_linear,q_begin,q_count,batch,kv_block)`.
It computes `row_start=k2q_row_ptr[head_kv*(total_rows+1)+row_linear]+q_begin`,
`q_batch=cu_q[batch]`, `k_batch=cu_k[batch]`, and `kv_len=kv_lens[batch]`
(flat fallback from `cu_k` remains). Query offset is explicit unless
`derive_q_offset`, when it is `kv_len-(cu_q[batch+1]-q_batch)`.

Softmax edge decode is distinct from Q-loader edge decode. Softmax initializes
`owned_packed=-1`; only owner lane 0 or 16 loads an index when
`edge<q_count`, then `shfl.sync.idx.b32` broadcasts within the sixteen-lane
subgroup. Q-loader instead loads `safe_edge=edge if valid else 0`, decodes its
query, and substitutes absolute query zero for invalid edges.

## Complete source-order operation sketch

```python
tid = thread_id()
warp = shfl_idx_b32(tid//32, source_lane=0, clamp=0x1f, mask=0xffffffff)
# instruction_selection: one `shfl.sync.idx.b32`; extent: every warp.
lane = tid % 32

if warp == 0 and elect_one():
    # instruction_selection: one `elect.sync _|predicate, 0xffffffff` by
    # warp 0; its elected lane performs the complete initialization sequence.
    for offset,count in the exact barrier table:
        mbarrier_init(offset,count)
        # instruction_selection: `mbarrier.init.shared::cta.b64`; 23 issues.
    fence_mbarrier_init()
    # instruction_selection: `fence.mbarrier_init.release.cluster`; one elected lane.
warp_sync()
# instruction_selection: `bar.warp.sync -1`; all sixteen warps.
if warp == 0:
    tmem_alloc(smem+184,512); tmem_relinquish()
    # instruction_selection: collective warp-0
    # `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32` then
    # `tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned`.
cta_sync()
# instruction_selection: `bar.sync 0`; 512 threads.
fence_tmem_after_thread_sync()
# instruction_selection: `tcgen05.fence::after_thread_sync`; 512 threads.
taddr = load_shared_volatile_u32(smem+184)
# instruction_selection: `ld.volatile.shared.b32`; one scalar/thread.
if 12 <= warp <= 15:
    setmaxnreg_decrease(48)
    # instruction_selection: `setmaxnreg.dec.sync.aligned.u32 48`; four warps.
```

### Softmax-even, warps 0-3

```python
if warp <= 3:
    setmaxnreg_increase(200)
    # instruction_selection: `setmaxnreg.inc.sync.aligned.u32 200`; four warps.
    decode_work()
    my_row = 32*warp+lane; token=my_row//16; qhead_local=my_row-16*token
    row_bits=(32*warp)<<16
    for iteration in serial(ceil_div(group_count,2)):
        group=2*iteration; phase=iteration&1
        wait(s_full[0],phase)
        # instruction_selection: retrying
        # `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64`; one generation.
        edge=8*group+token
        owned_packed=-1
        if lane==(lane//16)*16 and edge<q_count:
            owned_packed=k2q_qsplit_indices[head_kv*nnz_per_head+row_start+edge]
        packed=shfl_idx_b32(owned_packed,(lane//16)*16)
        # instruction_selection: predicated `ld.global.nc.b32`, then
        # `shfl.sync.idx.b32`; one owner per sixteen-lane subgroup.
        q_idx=packed&0xffffff
        valid_cols=clamp(kv_len-128*kv_block,0,128) if edge<q_count else 0
        if causal: valid_cols=min(valid_cols,query_offset+q_idx-128*kv_block+1)
        valid_cols=max(valid_cols,0)
        for quarter in static_range(4):
            tmem_ld_x32(scores[32*quarter:32*quarter+32],
                        taddr+row_bits+32*quarter)
        # instruction_selection: four
        # `tcgen05.ld.sync.aligned.32x32b.x32.b32`; 128 FP32 values/thread.
        for quarter in static_range(4):
            mask=low_bits(clamp(valid_cols-32*quarter,0,32))
            for element in static_range(32):
                scores[32*quarter+element]=select(mask&(1<<element),score,-inf)
        # instruction_selection: scalar `shl.b32`/`add.u32` mask construction
        # and 128 predicated selections for a partial row.
        row_max=max_reduce_four_x32(scores)
        # instruction_selection: four 32-value pairwise/alternating loops of
        # `max.f32`, followed by one `max.f32`; no `.ftz` modifier.
        score_bias=(-safe(row_max))*softmax_scale_log2 if valid_cols>0 else -inf
        # instruction_selection: `setp.eq.ftz.f32`/`selp.f32` selects safe
        # max, `neg.ftz.f32` negates it, `mul.ftz.f32` applies scale, and the
        # valid-column predicate selects this result or -inf.
        wait(p_empty[0],phase^1)
        # instruction_selection: same CTA-acquire mbarrier wait; one generation.
        for pair in static_range(64): fma_f32x2(scores[2*pair:2*pair+2],scale,bias)
        # instruction_selection: 64 `fma.rn.ftz.f32x2`.
        for segment in static_range(2):
            for pair in static_range(16): native_exp2(scores[32*segment+2*pair:])
            # instruction_selection: 32 `ex2.approx.ftz.f32` per segment.
            for pair in static_range(16): cvt_bf16x2(pwords[pair],scores[...])
            # instruction_selection: 16 `cvt.rn.bf16x2.f32` per segment.
            tmem_st_x16(taddr+row_bits+64+16*segment,pwords)
            # instruction_selection: one `tcgen05.st.sync.aligned.32x32b.x16.b32`.
        wait_tmem_store(); arrive(p_full[0])
        # instruction_selection: one `tcgen05.wait::st.sync.aligned` after
        # both x16 stores, then one
        # `mbarrier.arrive.release.cta.shared::cta.b64` per thread/generation.
        for upper_segment in static_range(2):
            base=64+32*upper_segment
            for pair in static_range(12): native_exp2(scores[base+2*pair:])
            # instruction_selection: 24 `ex2.approx.ftz.f32` per upper segment.
            for pair in static_range(12,16):
                packed_ex2_emulation(scores[base+2*pair:base+2*pair+2])
            # instruction_selection per emulated pair: two `max.f32`, one
            # `add.rm.ftz.f32x2`, two `sub.rn.ftz.f32x2`, three
            # `fma.rn.ftz.f32x2`, two `shl.b32`, two `add.u32`, and bit moves;
            # four pair iterations per upper segment.
            for pair in static_range(16): cvt_bf16x2(pwords32[16*upper_segment+pair],scores[...])
            # instruction_selection: 16 `cvt.rn.bf16x2.f32` per upper segment.
        tmem_st_x32(taddr+row_bits+96,pwords32)
        # instruction_selection: one `tcgen05.st.sync.aligned.32x32b.x32.b32`.
        row_sum=packed_add_reduce_128(scores)
        # instruction_selection: 64 `add.f32x2`, then one `add.f32`.
        wait_tmem_store(); arrive(p_full_2[0])
        wait_tmem_load(); arrive(s_empty[0])
        # instruction_selection: `tcgen05.wait::st.sync.aligned`, P2 arrival,
        # `tcgen05.wait::ld.sync.aligned`, S-empty arrival; once/thread/generation.

        wait(o_full[0],phase)
        # instruction_selection: CTA-acquire mbarrier wait; one generation.
        output_valid=(edge<q_count and 0<=((packed>>24)&15)<topk)
        partial_row=0; inv_sum=0
        if output_valid:
            partial_row=slot*total_q*Hq+(q_batch+q_idx)*Hq+16*head_kv+qhead_local
            reciprocal=rcp_approx(row_sum)
            inv_sum=reciprocal if row_sum>0 and row_sum==row_sum else 0
        # instruction_selection: inside `output_valid`, one
        # `rcp.approx.ftz.f32` is issued before `setp.gt.ftz.f32`/`selp.f32`
        # selects the reciprocal or zero.
        pending_scale=0
        for segment in serial(4):
            tmem_ld_x32(values,taddr+256+row_bits+32*segment)
            # instruction_selection: one
            # `tcgen05.ld.sync.aligned.32x32b.x32.b32` per segment, all lanes,
            # outside `output_valid`.
            segment_min=min_reduce_x32(values)
            # instruction_selection: 31 `min.ftz.f32` per segment.
            segment_max=max_reduce_x32(values)
            # instruction_selection: 33 `max.f32` per segment: sixteen pair
            # maxima, sixteen alternating accumulator updates, and one combine.
            residual_abs_max=max_f32(segment_max,-segment_min)
            # instruction_selection: one `max.f32`; no `.ftz` modifier.
            positive_not_nan=(residual_abs_max>0 and residual_abs_max==residual_abs_max)
            dequant_scale=residual_abs_max*inv_sum/448 if positive_not_nan else 0
            quant_scale=448/residual_abs_max if positive_not_nan else 0
            # instruction_selection: `setp.leu.ftz.f32` implements the exact
            # non-positive-or-NaN rejection; accepted values use two
            # `mul.ftz.f32` for dequant scale and one `div.approx.ftz.f32`
            # for quant scale, followed by zero/result selections.
            if output_valid and segment&1:
                cvt_bf16x2(scale_word,dequant_scale_previous,dequant_scale)
                # instruction_selection: one `cvt.rn.bf16x2.f32`.
                store_global_b32(partial_scale[4*partial_row+segment-1],scale_word)
                # instruction_selection: one `st.global.b32`.
            if not (segment&1): pending_scale=dequant_scale
            if output_valid:
                for pair in static_range(16): mul_f32x2(values[2*pair:],quant_scale)
                # instruction_selection: 16 `mul.rn.ftz.f32x2` per segment.
                for half in static_range(2):
                    for word in static_range(4):
                        cvt_e4m3x2(lo,values[16*half+4*word+0:]);
                        cvt_e4m3x2(hi,values[16*half+4*word+2:]); combine_u32(word,lo,hi)
                    # instruction_selection: eight
                    # `cvt.rn.satfinite.e4m3x2.f32` plus bit packing per half.
                    store_global_v4_b32(partial_o[128*partial_row+32*segment+16*half],words)
                    # instruction_selection: one `st.global.v4.b32` per half.
        if output_valid:
            log2_sum=lg2_approx(row_sum)
            candidate=(row_max*softmax_scale_log2)*ln2+log2_sum*ln2
            lse=candidate if row_sum>0 else -inf
            # instruction_selection: unconditional-in-output-valid
            # `lg2.approx.ftz.f32`, two `mul.ftz.f32`, one
            # `fma.rn.ftz.f32`, then `setp.gt.ftz.f32`/`selp.f32`.
            store_global_b32(partial_lse[partial_row],lse)
            if return_temperature_lse: store_global_b32(partial_temperature_lse[partial_row],lse)
            # instruction_selection: one mandatory and one predicated `st.global.b32`.
        wait_tmem_load(); arrive(o_empty[0])
        # instruction_selection: `tcgen05.wait::ld.sync.aligned`, then
        # `mbarrier.arrive.release.cta.shared::cta.b64`; once/thread/generation.
```

### Softmax-odd, warps 4-7

This is written out because its complete stage-1 protocol is an independent
source branch, not an opaque call.

```python
if 4 <= warp <= 7:
    setmaxnreg_increase(200)
    # instruction_selection: `setmaxnreg.inc.sync.aligned.u32 200`; four warps.
    decode_work()
    local_warp=warp-4; my_row=32*local_warp+lane
    token=my_row//16; qhead_local=my_row-16*token; row_bits=(32*local_warp)<<16
    for iteration in serial(group_count//2):
        group=2*iteration+1; phase=iteration&1
        wait(s_full[1],phase)
        # instruction_selection: CTA-acquire mbarrier wait; one generation.
        edge=8*group+token; owned_packed=-1
        if lane==(lane//16)*16 and edge<q_count:
            owned_packed=k2q_qsplit_indices[head_kv*nnz_per_head+row_start+edge]
        packed=shfl_idx_b32(owned_packed,(lane//16)*16)
        # instruction_selection: predicated `ld.global.nc.b32` and subgroup
        # `shfl.sync.idx.b32`, preserving -1 for invalid edges.
        q_idx=packed&0xffffff
        valid_cols=clamp(kv_len-128*kv_block,0,128) if edge<q_count else 0
        if causal: valid_cols=min(valid_cols,query_offset+q_idx-128*kv_block+1)
        valid_cols=max(valid_cols,0)
        for quarter in static_range(4):
            tmem_ld_x32(scores[32*quarter:],taddr+128+row_bits+32*quarter)
        # instruction_selection: four `tcgen05.ld.sync.aligned.32x32b.x32.b32`.
        for quarter in static_range(4):
            mask=low_bits(clamp(valid_cols-32*quarter,0,32))
            for element in static_range(32):
                scores[32*quarter+element]=select(mask&(1<<element),score,-inf)
        # instruction_selection: scalar `shl.b32`/`add.u32` mask construction
        # and 128 predicated selections, explicitly matching the even role.
        row_max=max_reduce_four_x32(scores)
        # instruction_selection: pairwise/alternating `max.f32`, four x32 plus final.
        score_bias=(-safe(row_max))*softmax_scale_log2 if valid_cols>0 else -inf
        # instruction_selection: `setp.eq.ftz.f32`/`selp.f32`,
        # `neg.ftz.f32`, `mul.ftz.f32`, and valid-column selection.
        wait(p_empty[1],phase^1)
        # instruction_selection: CTA-acquire mbarrier wait.
        for pair in static_range(64): fma_f32x2(scores[2*pair:],scale,bias)
        # instruction_selection: 64 `fma.rn.ftz.f32x2`.
        for segment in static_range(2):
            for pair in static_range(16): native_exp2(scores[32*segment+2*pair:])
            # instruction_selection: 32 `ex2.approx.ftz.f32` per segment.
            for pair in static_range(16): cvt_bf16x2(pwords[pair],scores[...])
            # instruction_selection: 16 `cvt.rn.bf16x2.f32`.
            tmem_st_x16(taddr+128+64+row_bits+16*segment,pwords)
            # instruction_selection: one `tcgen05.st.sync.aligned.32x32b.x16.b32`.
        wait_tmem_store(); arrive(p_full[1])
        # instruction_selection: one TMEM-store wait after both x16 stores,
        # then exactly one CTA-release mbarrier arrival per thread/generation.
        for upper_segment in static_range(2):
            base=64+32*upper_segment
            for pair in static_range(12): native_exp2(scores[base+2*pair:])
            # instruction_selection: 24 `ex2.approx.ftz.f32`.
            for pair in static_range(12,16): packed_ex2_emulation(scores[base+2*pair:])
            # instruction_selection: four packed emulations, each with
            # `max.f32`, add/sub f32x2, three `fma.rn.ftz.f32x2`, shifts/adds.
            for pair in static_range(16): cvt_bf16x2(pwords32[...],scores[...])
            # instruction_selection: 16 `cvt.rn.bf16x2.f32`.
        tmem_st_x32(taddr+128+64+row_bits+32,pwords32)
        # instruction_selection: one `tcgen05.st.sync.aligned.32x32b.x32.b32`.
        row_sum=packed_add_reduce_128(scores)
        # instruction_selection: 64 `add.f32x2`, then `add.f32`.
        wait_tmem_store(); arrive(p_full_2[1])
        wait_tmem_load(); arrive(s_empty[1])
        # instruction_selection: exact store-wait/P2-arrive/load-wait/S-empty-arrive order.

        wait(o_full[1],phase)
        # instruction_selection: CTA-acquire mbarrier wait.
        output_valid=(edge<q_count and 0<=((packed>>24)&15)<topk)
        partial_row=0; inv_sum=0
        if output_valid:
            partial_row=slot*total_q*Hq+(q_batch+q_idx)*Hq+16*head_kv+qhead_local
            reciprocal=rcp_approx(row_sum)
            inv_sum=reciprocal if row_sum>0 and row_sum==row_sum else 0
        # instruction_selection: inside `output_valid`, one
        # `rcp.approx.ftz.f32` is issued before `setp.gt.ftz.f32`/`selp.f32`
        # selects the reciprocal or zero.
        pending_scale=0
        for segment in serial(4):
            tmem_ld_x32(values,taddr+384+row_bits+32*segment)
            # instruction_selection: one `tcgen05.ld.sync.aligned.32x32b.x32.b32`
            # per segment outside `output_valid`.
            segment_min=min_reduce_x32(values)
            # instruction_selection: 31 `min.ftz.f32`.
            segment_max=max_reduce_x32(values)
            # instruction_selection: 33 `max.f32`: sixteen pair maxima,
            # sixteen accumulator updates, and one accumulator combine.
            residual_abs_max=max_f32(segment_max,-segment_min)
            # instruction_selection: one `max.f32`.
            positive_not_nan=(residual_abs_max>0 and residual_abs_max==residual_abs_max)
            dequant_scale=residual_abs_max*inv_sum/448 if positive_not_nan else 0
            quant_scale=448/residual_abs_max if positive_not_nan else 0
            # instruction_selection: `setp.leu.ftz.f32` implements the exact
            # non-positive-or-NaN rejection; accepted values use two
            # `mul.ftz.f32` for dequant scale and one `div.approx.ftz.f32`
            # for quant scale, followed by zero/result selections.
            if output_valid and segment&1:
                cvt_bf16x2(scale_word,pending_scale,dequant_scale)
                # instruction_selection: one `cvt.rn.bf16x2.f32`.
                store_global_b32(partial_scale[4*partial_row+segment-1],scale_word)
                # instruction_selection: one `st.global.b32`.
            if not (segment&1): pending_scale=dequant_scale
            if output_valid:
                for pair in static_range(16): mul_f32x2(values[2*pair:],quant_scale)
                # instruction_selection: 16 `mul.rn.ftz.f32x2`.
                for half in static_range(2):
                    for word in static_range(4):
                        cvt_e4m3x2(lo,values[16*half+4*word:]);
                        cvt_e4m3x2(hi,values[16*half+4*word+2:]); combine_u32(word,lo,hi)
                    # instruction_selection: eight
                    # `cvt.rn.satfinite.e4m3x2.f32` plus bit packing.
                    store_global_v4_b32(partial_o[...],words)
                    # instruction_selection: one `st.global.v4.b32` per half.
        if output_valid:
            log2_sum=lg2_approx(row_sum)
            candidate=(row_max*softmax_scale_log2)*ln2+log2_sum*ln2
            lse=candidate if row_sum>0 else -inf
            # instruction_selection: unconditional-in-output-valid
            # `lg2.approx.ftz.f32`, two `mul.ftz.f32`, one
            # `fma.rn.ftz.f32`, then `setp.gt.ftz.f32`/`selp.f32`.
            store_global_b32(partial_lse[partial_row],lse)
            if return_temperature_lse: store_global_b32(partial_temperature_lse[partial_row],lse)
            # instruction_selection: mandatory plus predicated `st.global.b32`.
        wait_tmem_load(); arrive(o_empty[1])
        # instruction_selection: TMEM-load wait then CTA-release mbarrier arrival.
```

### Q-loader, warps 8-11

```python
if 8 <= warp <= 11:
    setmaxnreg_decrease(64)
    # instruction_selection: `setmaxnreg.dec.sync.aligned.u32 64`; four warps.
    decode_work()
    for group in serial(group_count):
        stage=group&1; phase=(group//2)&1
        wait(q_empty[stage],phase^1)
        # instruction_selection: CTA-acquire mbarrier wait; one/warp/generation.
        if elect_one():
            # instruction_selection: one `elect.sync _|predicate, 0xffffffff`
            # in each producer warp elects the Q TMA issuer.
            expect_tx(q_full[stage],8192)
            # instruction_selection:
            # `mbarrier.arrive.expect_tx.release.cta.shared::cta.b64`, 8,192
            # bytes from each of four elected producer lanes.
            for local_token in static_range(2):
                edge=8*group+2*(warp-8)+local_token
                safe_edge=edge if edge<q_count else 0
                packed=k2q_qsplit_indices[head_kv*nnz_per_head+row_start+safe_edge]
                q_abs=q_batch+(packed&0xffffff) if edge<q_count else 0
                dst=Q[stage]+2048*(2*(warp-8)+local_token)
                tma4d(Qmap,(0,16*head_kv,0,q_abs),dst,q_full[stage])
                tma4d(Qmap,(0,16*head_kv,1,q_abs),dst+16384,q_full[stage])
            # instruction_selection: four elected-lane
            # `cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes`
            # per warp/generation, no L2 cache-hint modifier, total 8,192 bytes.
```

### MMA issuer, warp 12

An `elected_commit(bar)` below means an internal `elect.sync` followed by
predicated
`tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`;
exactly one lane issues each commit.

```python
if warp == 12:
    decode_work(); wait(k_full,0)
    # instruction_selection: CTA-acquire mbarrier wait.
    # Prime group 0 unconditionally and group 1 when group_count>1.
    for each explicit prime group stage in (0, conditional 1):
        wait(q_full[stage],0); wait(s_empty[stage],1)
        # instruction_selection: two CTA-acquire mbarrier waits.
        q_lo=((Q[stage]>>4)&0x3fff); k_lo=((K>>4)&0x3fff)
        mma_leader=elect_sync()
        # instruction_selection: one `elect.sync _|predicate, 0xffffffff`
        # elects the issuer for the entire eight-instruction QK tile.
        for k16 in static_range(8):
            tcgen05_mma_ss(taddr+128*stage,pack(q_lo,0x40004040),
                           pack(k_lo,0x40004040),136316048,
                           accumulate=(k16!=0),predicate=mma_leader)
            if k16<7:
                q_lo += (2,2,2,1018,2,2,2)[k16]
                k_lo += (2,2,2,1018,2,2,2)[k16]
        # instruction_selection: eight elected-lane
        # `tcgen05.mma.cta_group::1.kind::f16` SS issues, first predicate false,
        # remaining seven true; BF16xBF16->FP32, M/N=128, K=128.
        elected_commit(s_full[stage]); elected_commit(q_empty[stage])
        # instruction_selection: two elected tcgen05 commits in this order.

    wait(v_full,0)
    # instruction_selection: CTA-acquire mbarrier wait.
    for group in serial(2,group_count):
        pv=group-2; stage=pv&1; phase=(pv//2)&1
        wait(p_full[stage],phase); wait(o_empty[stage],phase^1)
        # instruction_selection: two CTA-acquire mbarrier waits.
        v_lo=((V>>4)&0x3fff)|0x04000000
        pv_half_leader=elect_sync()
        # instruction_selection: one `elect.sync _|predicate, 0xffffffff`
        # elects the issuer for this four-instruction PV half.
        for k16 in static_range(4):
            tcgen05_mma_ts(taddr+256+128*stage,
                           taddr+64+128*stage+8*k16,
                           pack(v_lo+128*k16,0x40004040),136381584,
                           accumulate=(k16!=0),predicate=pv_half_leader)
        # instruction_selection: four elected-lane TS `tcgen05.mma...kind::f16`,
        # first predicate false then true, consuming P[0:64].
        wait(p_full_2[stage],phase)
        # instruction_selection: CTA-acquire mbarrier wait.
        pv_second_leader=elect_sync()
        # instruction_selection: one `elect.sync _|predicate, 0xffffffff`
        # elects the issuer for the second four-instruction PV half.
        for k16 in static_range(4):
            tcgen05_mma_ts(taddr+256+128*stage,
                           taddr+96+128*stage+8*k16,
                           pack(v_lo+512+128*k16,0x40004040),136381584,
                           accumulate=True,predicate=pv_second_leader)
        # instruction_selection: four elected-lane TS `tcgen05.mma...kind::f16`,
        # all predicates true, consuming P[64:128].
        elected_commit(o_full[stage]); elected_commit(p_empty[stage])
        # instruction_selection: two elected tcgen05 commits.

        qstage=group&1; qphase=(group//2)&1
        wait(q_full[qstage],qphase); wait(s_empty[qstage],qphase^1)
        # instruction_selection: two CTA-acquire mbarrier waits.
        steady_qk_leader=elect_sync()
        # instruction_selection: one `elect.sync _|predicate, 0xffffffff`
        # elects this steady-state QK tile issuer.
        issue_the_same_eight_SS_QK_instructions_for(qstage)
        # instruction_selection: eight elected SS `tcgen05.mma...kind::f16`,
        # exact descriptors/walk and false-then-true predicates as the primes.
        elected_commit(s_full[qstage]); elected_commit(q_empty[qstage])
        # instruction_selection: two elected tcgen05 commits.

    drain_start=0 if group_count==1 else group_count-2
    for pv in serial(drain_start,group_count):
        stage=pv&1; phase=(pv//2)&1
        wait(p_full[stage],phase); wait(o_empty[stage],phase^1)
        # instruction_selection: P-full then O-empty CTA-acquire waits.
        v_lo=((V>>4)&0x3fff)|0x04000000
        pv_half_leader=elect_sync()
        # instruction_selection: one `elect.sync _|predicate, 0xffffffff`
        # elects the issuer for this four-instruction PV half.
        for k16 in static_range(4):
            tcgen05_mma_ts(taddr+256+128*stage,
                           taddr+64+128*stage+8*k16,
                           pack(v_lo+128*k16,0x40004040),136381584,
                           accumulate=(k16!=0),predicate=pv_half_leader)
        # instruction_selection: four elected TS `tcgen05.mma...kind::f16`,
        # overwrite then accumulate, first P half.
        wait(p_full_2[stage],phase)
        # instruction_selection: CTA-acquire mbarrier wait.
        pv_second_leader=elect_sync()
        # instruction_selection: one `elect.sync _|predicate, 0xffffffff`
        # elects the issuer for the second four-instruction PV half.
        for k16 in static_range(4):
            tcgen05_mma_ts(taddr+256+128*stage,
                           taddr+96+128*stage+8*k16,
                           pack(v_lo+512+128*k16,0x40004040),136381584,
                           accumulate=True,predicate=pv_second_leader)
        # instruction_selection: four elected TS `tcgen05.mma...kind::f16`,
        # all accumulate, second P half.
        elected_commit(o_full[stage]); elected_commit(p_empty[stage])
        # instruction_selection: O-full then P-empty elected tcgen05 commits.
    for completed in serial(drain_start,group_count):
        wait(o_empty[completed&1],(completed//2)&1)
        # instruction_selection: CTA-acquire mbarrier wait, one drained stage.
    tmem_dealloc(load_shared_volatile_u32(smem+184),512)
    # instruction_selection: mailbox `ld.volatile.shared.b32`, then collective
    # warp `tcgen05.dealloc.cta_group::1.sync.aligned.b32`, 512 columns.
```

### Transform and physical-page loader

```python
if 13 <= warp <= 14:
    decode_work()                    # source metadata-only branch

if warp == 15:
    if elect_one():
        # instruction_selection: `elect.sync _|predicate, 0xffffffff`, then
        # one elected `griddepcontrol.launch_dependents`.
        griddepcontrol_launch_dependents()
    decode_work()
    physical_page=page_table[batch*max_pages+kv_block]
    physical_page=max(physical_page,0)
    page_head=physical_page*Hkv+head_kv
    if elect_one():
        # instruction_selection: one `elect.sync _|predicate, 0xffffffff`
        # elects the sole K/V TMA issuer.
        expect_tx(k_full,32768)
        # instruction_selection:
        # `mbarrier.arrive.expect_tx.release.cta.shared::cta.b64`, 32,768 bytes.
        for dim_half in static_range(2):
            for token_half in static_range(2):
                tma4d_l2(Kmap,(0,64*token_half,dim_half,page_head),
                         K+8192*token_half+16384*dim_half,k_full,
                         hint=0x12F0000000000000)
        # instruction_selection: four elected
        # `cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint`
        # issues, 8,192 bytes each, with the stated 64-bit hint.
        expect_tx(v_full,32768)
        # instruction_selection: same expect-tx family and byte count.
        for dim_half in static_range(2):
            for token_half in static_range(2):
                tma4d_l2(Vmap,(0,64*token_half,dim_half,page_head),
                         V+8192*token_half+16384*dim_half,v_full,
                         hint=0x12F0000000000000)
        # instruction_selection: four more elected cache-hinted rank-4 TMA
        # issues, 8,192 bytes each. K and V are published independently.
```

There is no trailing CTA cleanup: warp 12 drains both O stages and performs
the source’s collective TMEM deallocation inside its role.

## Numerical acceptance contract

The port retains packed `fma.rn.ftz.f32x2` and `mul.rn.ftz.f32x2`, the exact
mixed native/polynomial exp2 path, `max.f32` (not `max.ftz.f32`),
`min.ftz.f32`, BF16 round-to-nearest packing, `rcp.approx.ftz.f32`,
`lg2.approx.ftz.f32`, and `cvt.rn.satfinite.e4m3x2.f32`. The positive-scale
predicate is exactly `x>0 && x==x`, so positive infinity is not excluded.

Correctness compares all inner partial buffers bitwise, including unwritten
sentinels, and checks prefix/suffix guards, input immutability, and repeat-run
determinism. There is no broad `atol`/`rtol` fallback in the acceptance gate.
