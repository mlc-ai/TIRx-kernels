<!--
Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0

This design sketch documents a TIRx port of FlashInfer's SM100
cp_delta_rule_dsl_sm100 pipeline at commit f2e04400e330fb2debe0bf8730d9424a1d37927f.
-->

# TIRx SM100 CP delta-rule prefill: coarse WASP pipeline sketch

This file is a non-executable design sketch. It is neither a mathematical
reference nor a substitute implementation. It records the source-shaped
execution skeleton of FlashInfer's four-launch context-parallel prefill chain:

1. a 64-token T precompute;
2. a tcgen05 M/N precompute;
3. exactly one production fixup specialization (SIMT-row4, UTCMMA-row64, or
   UTCMMA-row128);
4. the tcgen05 CP prefill.

The implementation represented here belongs in
[`tirx_kernels/flashinfer/gdn_prefill/gdn_cp_prefill_sm100.py`](../../../../tirx_kernels/flashinfer/gdn_prefill/gdn_cp_prefill_sm100.py).
That module becomes the executable source of truth only after this sketch first
passes review. The sketch itself is then frozen.

The port covers SM100a, `D=128`, matching FP16/BF16 Q/K/V/O/T, FP32 alpha and
beta, INT32/INT64 `cu_seqlens`, FP16/BF16/FP32 optional state, optional INT32
state indirection, GQA and GVA head sharing, source automatic/explicit chunk
selection, and the three production fixup branches. Forced SIMT-row8, HMMA
fixup, checkpoints, persistent scheduling, and 2-CTA instructions are out of
scope because `cp_delta_rule_dsl_sm100` does not select them.

## Pipeline at a glance

| Launch / warps | Source-owned tile program | Publication and reuse edges |
| --- | --- | --- |
| T precompute, warps 0..3 | warp 1 TMA-loads K; warp 2 software-loads beta; all 128 threads form K-K, beta-fold the strict lower triangle, run the four-level 8→16→32→64 inverse, and store T | `K/Beta full -> K-K -> inverse barriers -> T store` |
| MN, warps 0..3 | initialize/scale/store M, materialize X, and convert M/Z operands | `m_init`, `x_acc/x_ready`, `m_in`, `z_acc/z_ready`, `m_acc`, `done` |
| MN, warps 4..7 | initialize/scale/store N and form its Y operand | `n_init`, `n_in`, `y_acc/y_ready`, `n_acc`, `done` |
| MN, warp 8 | issue Z and M-update tcgen05 MMAs | K/X/M/Z pipeline edges |
| MN, warp 9 | TMA-load staged K/V/T | `load_k/load_v/load_t` |
| MN, warp 10 | software-load and prefix-process alpha | `alpha[4]` to both compute groups |
| MN, warp 11 | issue X, Y, and N-update tcgen05 MMAs | K/T/X/N/Y pipeline edges |
| SIMT-row4 fixup, 128 threads | one thread per column recurrently advances four rows, with 16-float M prefetch and inter-chunk register handoff | CTA barriers publish `sState`; GMEM fixed-state publication per chunk |
| UTCMMA fixup, warps 0..3 | convert/load/store the FP32 TMEM state accumulator | `n -> mma_ready -> mma_done -> fixed/final state` |
| UTCMMA fixup, warp 4 | issue TF32 recurrence MMA | `m + TMEM operand -> TMEM accumulator` |
| UTCMMA fixup, warp 5 | TMA-load M/N rings | `load_m/load_n` |
| Prefill, warps 0..3 | transform precomputed T and gate-scale QK | `load_t/load_gate/cg0_acc -> ainv_ready/qk_ready` |
| Prefill, warps 4..7 | load/recur/store state, form V-KS and decay-V, scale QS, and stage O | `kv_acc/state_inp/vks/nv/decay_v/o_store` |
| Prefill, warp 8 | issue paired QK tcgen05 MMAs | `load_q/load_k -> cg0_acc` |
| Prefill, warp 9 | update Q/K/V TensorMaps and TMA-load Q/K/V/T | `load_q/load_k/load_v/load_t` |
| Prefill, warp 10 | issue KS, QS, NV, QKV, and KV in source order | all CG1 operand/result edges |
| Prefill, warp 11 | update O TensorMap, maintain four-tile gate lookahead, and TMA-store O | `load_gate/o_store` |

The source role selection is not flattened. In MN the invalid-work branch is
followed by CG0, CG1, transfer-MMA, state-MMA, TMA, alpha, and other branches in
that order. In prefill CG0 is an independent `if`; CG1/warp8/warp10/warp9 are
one `if/elif` chain; warp11 is a final independent `if`.

## Primitive vocabulary

Structural operations do not move or compute values:

```python
tile(...)          # storage, dtype, logical shape, placement, alignment
view(...)          # indexing/layout-only view
alias(...)         # exact physical alias with an explicit lifetime
slice(...)         # logical subregion
transpose(...)     # view only
reg_tile(...)      # role-local register fragment
identity(...)      # logical coordinate tile
```


Directional copies are explicit:

```python
copy_g2s(src, dst, predicate=None, completion=None, descriptor=None)
copy_s2g(src, dst, predicate=None, descriptor=None)
copy_p2r(src, dst)  # launch-parameter descriptor template to registers
copy_g2r(src, dst, predicate=None, cache=None)
copy_r2g(src, dst, predicate=None, cache=None)
copy_s2r(src, dst)
copy_r2s(src, dst)
copy_t2r(src, dst)
copy_r2t(src, dst)
copy_r2r(src, dst)
```

The only computation primitives below are `fill`, `cast`, `add`, `sub`, `mul`,
`fma`, `neg`, `log2`, `exp2`, `add_packed_f32x2`, `shuffle_up`,
`shuffle_index`, `select`, and `gemm`. Schedule primitives are `setmaxnreg`,
`pipe_init`, `acquire`, `expect_tx`, `current`, `advance`, `wait`, `wait_ready`,
`commit`, `manual_arrive`, `release`, `tail`, `fence`, `barrier`, TensorMap
template copies/replacements and fences, async-copy group commit/wait, and TMEM
allocation/lifetime operations.
There is deliberately no compound operation named `inverse`,
`prefix_scan`, `update_state`, `run_pipeline`, or `gdn_prefill`.

## Complete sketch

```python
# ===========================================================================
# Static family, host dispatch, workspaces, and four-launch boundary
# ===========================================================================

@specialize(
    TARGET="sm_100a",
    D=128,
    IO_DTYPE=("f16", "bf16"),
    CU_DTYPE=("i32", "i64"),
    STATE_DTYPE=("f16", "bf16", "f32"),
    Q_HEADS=(1, 2, 4, 16),
    K_HEADS=(1, 2, 16),
    V_HEADS=(1, 4, 8, 16, 64),
    NEEDS_INITIAL_STATE=(False, True),
    STORE_FINAL_STATE=(False, True),
    USE_STATE_INDICES=(False, True),
    PREFILL_STORE_FINAL_STATE=False,
    PREFILL_STATE_DTYPE="f32",
    FIXUP_KIND=("simt_row4", "utcmma64", "utcmma128"),
    CHECKPOINTS=False,
    PERSISTENT=False,
    CTA_GROUP=1,
)
def cp_delta_rule_dsl_sm100_host(...):
    H_STATE = max(Q_HEADS, V_HEADS)
    assert Q_HEADS % K_HEADS == 0 and V_HEADS % K_HEADS == 0

    if cp_chunk_len is None:
        # Preserve choose_cp_chunk_len_host exactly: SM100 uses ratio 1/1;
        # HBM and GDDR select their source 1/2 and 1/3 thresholds.
        assert cp_chunk_len_granularity % 64 == 0
        ratio_num, ratio_den = (1,1)
        threshold_num, threshold_den = ((1,3) if IS_GDDR else (1,2))
        approx_ctas = ceil_div(total_tokens, cp_chunk_len_granularity) * H_STATE
        if approx_ctas * threshold_den < num_sms * threshold_num:
            square = ceil_div(max_seq_len * 64 * ratio_num, ratio_den)
            cp_chunk_len = max(64, round_up(ceil_sqrt(square), 64))
        else:
            target_chunks = max(1, num_sms // H_STATE)
            remaining_tokens = max(0, total_tokens - max_seq_len)
            remaining_sequences = max(0, num_sequences - 1)
            lo = 1
            hi = max(1, ceil_div(max_seq_len, cp_chunk_len_granularity))
            while lo < hi:
                mid = (lo + hi) // 2
                candidate = mid * cp_chunk_len_granularity
                m = min(remaining_sequences, remaining_tokens)
                remaining_bound = m + (remaining_tokens - m) // candidate
                candidate_chunks = ceil_div(max_seq_len, candidate) + remaining_bound
                if candidate_chunks <= target_chunks:
                    hi = mid
                else:
                    lo = mid + 1
            cp_chunk_len = lo * cp_chunk_len_granularity
    assert cp_chunk_len % 64 == 0

    total_t_blocks = chunk_bound(num_sequences, total_tokens, 64)
    total_cp_chunks = chunk_bound(num_sequences, total_tokens, cp_chunk_len)
    max_t_blocks_per_seq = ceil_div(max_seq_len, 64)
    max_cp_chunks_per_seq = ceil_div(max_seq_len, cp_chunk_len)

    T_ws = tile("gmem", IO_DTYPE, [total_t_blocks,H_STATE,64,64], alignment=128)
    M_ws = tile("gmem", "f32", [total_cp_chunks,H_STATE,128,128], alignment=128)
    N_ws = tile("gmem", "f32", [total_cp_chunks,H_STATE,128,128], alignment=128)
    Fixed_ws = tile("gmem", "f32", [total_cp_chunks,H_STATE,128,128], alignment=128)
    Init_ws = tile("gmem", "f32", [num_sequences,H_STATE,128,128], optional=True)
    PrefillMaps = tile(
        "gmem", "u8",
        [num_sequences,H_STATE,max_cp_chunks_per_seq,5,128],
        alignment=128,
    )

    launch(_t_precompute_sm100, K, beta, T_ws, cu_seqlens, ...)
    # instruction_selection: cudaLaunchKernelEx/CUDA driver launch; extent: one independent device launch
    launch(_mn_precompute_sm100, K, V, T_ws, alpha, M_ws, N_ws, cu_seqlens, ...)
    # instruction_selection: cudaLaunchKernelEx/CUDA driver launch; extent: one independent device launch

    parallel_states = num_sequences * H_STATE
    if parallel_states <= (num_sms * 2 // 32):
        fixup = _fixup_simt_row4_sm100
    elif parallel_states <= (num_sms // 2):
        fixup = _fixup_utcmma64_sm100
    else:
        fixup = _fixup_utcmma128_sm100
    launch(fixup, M_ws, N_ws, initial_state, Init_ws, Fixed_ws, final_state,
           state_indices, cu_seqlens, ...)
    # instruction_selection: cudaLaunchKernelEx/CUDA driver launch; extent: one selected independent device launch
    launch(_prefill_sm100, Q, K, V, alpha, T_ws, O, cu_seqlens,
           Fixed_ws, Init_ws, PrefillMaps,
           PREFILL_STORE_FINAL_STATE=False, PREFILL_STATE_DTYPE="f32", ...)
    # instruction_selection: cudaLaunchKernelEx/CUDA driver launch; extent: one independent device launch


# ===========================================================================
# Launch 1: T precompute, inherited CPDeltaRuleTPrecomputeSm120 body
# ===========================================================================

@kernel(
    grid=(H_STATE * max_t_blocks_per_seq, num_sequences, 1),
    block=(128,1,1), num_warps=4, min_blocks_per_sm=8, target="sm_100a",
)
def _t_precompute_sm100(K, Beta, T_out, Cu, K_HEADS, H_STATE,
                        total_t_blocks, num_sequences):
    BLK = 64
    D = 128
    tid = thread_id_x()
    warp = warp_uniform(tid // 32)
    lane = tid % 32
    bx = block_id_x()
    seq = block_id_y()
    sab_head = bx % H_STATE
    k_head = sab_head * K_HEADS // H_STATE
    block_in_seq = bx // H_STATE
    seq_start = copy_g2r(Cu[seq], reg("seq_start"))
    # instruction_selection: ld.global.b32; extent: scalar; INT64 specialization changes address stride/offset only
    seq_end = copy_g2r(Cu[seq+1], reg("seq_end"))
    # instruction_selection: ld.global.b32; extent: scalar; INT64 specialization changes address stride/offset only
    valid = block_in_seq < ceil_div(seq_end-seq_start, 64)
    if valid:
        tok0 = seq_start + block_in_seq * 64
        valid_len = min(64, seq_end - tok0)
        t_block = varlen_chunk_idx(seq, seq_start, block_in_seq, 64)

        # Struct order is semantically retained: K and beta pipeline barriers,
        # then 128B-aligned K, 16B-aligned inverse, and 16B-aligned beta.
        k_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
        beta_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
        sK = tile("smem", IO_DTYPE, [64,128], alignment=128,
                  layout="K_SW128; DS and transposed SD aliases")
        sInv = tile("smem", "f16", [64,64], alignment=16,
                    layout="8x8 atom, stride=(8,1)")
        sBeta = tile("smem", "f32", [64], alignment=16)

        pipe_init(k_bar, producer_threads=1, consumer_threads=4,
                  stages=1, expected_bytes=64*128*bytes(IO_DTYPE),
                  full_empty_barriers=True, defer_sync=False)
        # instruction_selection: mbarrier.init.shared.b64; extent: two issues for the one-stage K full/empty barrier pair
        fence("mbarrier_init", owner="k_pipeline_create")
        # instruction_selection: fence.mbarrier_init.release.cluster; extent: one built-in K-pipeline initialization fence
        barrier("cta", owner="k_pipeline_create")
        # instruction_selection: bar.sync 0; extent: one built-in K-pipeline CTA sync
        pipe_init(beta_bar, producer_threads=32, consumer_threads=128, stages=1,
                  full_empty_barriers=True, defer_sync=False)
        # instruction_selection: mbarrier.init.shared.b64; extent: two issues for the one-stage beta full/empty barrier pair
        fence("mbarrier_init", owner="beta_pipeline_create")
        # instruction_selection: fence.mbarrier_init.release.cluster; extent: one built-in beta-pipeline initialization fence
        fence("mbarrier_init", owner="source_explicit")
        # instruction_selection: fence.mbarrier_init.release.cluster; extent: one explicit source initialization fence after both pipelines
        barrier("cta", owner="source_explicit")
        # instruction_selection: bar.sync 0; extent: one explicit source CTA sync after both pipelines

        if warp == 1:
            descriptor_prefetch(K.tensor_map)
            # instruction_selection: prefetch.tensormap; extent: one K descriptor
            k_prod = acquire(k_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one stage acquire
            expect_tx(k_prod, 64*128*bytes(IO_DTYPE))
            # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one K-stage byte expectation before TMA issue
            copy_g2s(K[k_head,tok0:tok0+64,0:128], sK,
                     completion=k_prod.barrier)
            # instruction_selection: cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint; extent: two issues covering one 64x128 tile
            commit(k_prod)
            # instruction_selection: no standalone PTX instruction in this specialization; extent: one source-level K TMA producer cursor commit after the two-issue tile

        elif warp == 2:
            beta_prod = acquire(beta_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one stage acquire
            for row in (lane, lane+32):
                b = 0.0
                if row < valid_len:
                    copy_g2r(Beta[tok0+row,sab_head], b)
                    # instruction_selection: ld.global.b32; extent: scalar, two iterations per lane
                copy_r2s(b, sBeta[row])
                # instruction_selection: st.shared.b32; extent: scalar, two iterations per lane
            fence("async_shared_view")
            # instruction_selection: fence.proxy.async.shared::cta; extent: CTA shared view
            commit(beta_prod)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: one software stage

        k_cons = wait(k_bar, consumer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one stage wait
        beta_cons = wait(beta_bar, consumer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one stage wait
        fence("async_shared_view")
        # instruction_selection: fence.proxy.async.shared::cta; extent: CTA shared view
        barrier("cta")
        # instruction_selection: bar.sync 0; extent: full CTA

        rK_A = reg_tile(IO_DTYPE, "64x128 HMMA-A fragment")
        rK_B = reg_tile(IO_DTYPE, "128x64 HMMA-B fragment")
        copy_s2r(sK, rK_A)
        # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.shared.b16; extent: 64x128 tiled A load
        copy_s2r(transpose(sK), rK_B)
        # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.shared.b16; extent: 128x64 tiled B load
        rKK = reg_tile("f32", [64,64])
        fill(rKK, 0.0)
        # instruction_selection: mov.b32/mov.f32 zero initialization; extent: 64x64 accumulator fragment
        gemm(rKK, rK_A, rK_B, shape=(64,64,128), accumulate=False)
        # instruction_selection: mma.sync.aligned.m16n8k16.row.col.f32.{f16,bf16}.{f16,bf16}.f32; extent: 64x64x128 tiled loop
        release(k_cons)
        # instruction_selection: mbarrier.arrive.shared.b64; extent: one consumer release

        for element in rKK:
            row, col = coordinate(element)
            if row > col:
                rKK[element] = mul(rKK[element], sBeta[row])
                # instruction_selection: mul.rn.f32; extent: scalar per owned lower-triangle element
            else:
                fill(rKK[element], 0.0)
                # instruction_selection: mov.b32/mov.f32; extent: scalar per owned diagonal/upper element
        rLower = cast("f16", rKK)
        # instruction_selection: cvt.rn.f16x2.f32; extent: packed pairs across one 64x64 register fragment
        copy_r2s(rLower, sInv)
        # instruction_selection: stmatrix.sync.aligned.m8n8.x4.shared.b16; extent: 64x64 tile
        barrier(id=6, threads=128)
        # instruction_selection: bar.sync 6,128; extent: named CTA barrier

        # Level 1: eight independent 8x8 diagonal blocks. Threads 0..63 own
        # one row each and perform the source Gauss-elimination order.
        if tid < 64:
            block8 = tid // 8
            row8 = tid % 8
            rRow16 = reg_tile("f16", [8])
            copy_s2r(sInv[block8*8+row8, block8*8:block8*8+8], rRow16)
            # instruction_selection: ld.shared.v4.b32; extent: one 8xf16 row
            rRow = cast("f32", rRow16)
            # instruction_selection: cvt.f32.f16; extent: eight scalars
            for col8 in range(8):
                rRow[col8] = select(row8 == col8, 1.0,
                                    select(row8 < col8, 0.0, rRow[col8]))
                # instruction_selection: setp + selp.b32; extent: scalar, eight columns
            for src_row in range(7):
                row_scale = neg(rRow[src_row])
                # instruction_selection: neg.f32/xor sign bit; extent: scalar
                for col8 in range(src_row):
                    src_val = shuffle_index(rRow[col8], src_row,
                                            mask=(8-1)|((32-8)<<8))
                    # instruction_selection: shfl.sync.idx.b32; extent: scalar in 8-lane subgroup
                    if row8 > src_row:
                        rRow[col8] = fma(row_scale, src_val, rRow[col8])
                        # instruction_selection: fma.rn.f32; extent: scalar elimination update
                if row8 > src_row:
                    copy_r2r(row_scale, rRow[src_row])
                    # instruction_selection: mov.b32; extent: scalar
            rRow16 = cast("f16", rRow)
            # instruction_selection: cvt.rn.f16x2.f32; extent: four packed pairs per 8-value row
            copy_r2s(rRow16, sInv[block8*8+row8, block8*8:block8*8+8])
            # instruction_selection: st.shared.v4.b32; extent: one 8xf16 row
        barrier(id=6, threads=128)
        # instruction_selection: bar.sync 6,128; extent: named CTA barrier

        # Level 2: four warps extend four 8x8 diagonal pairs to 16x16.
        block16 = tid // 32
        rD8 = reg_tile("f16", "8x8 HMMA-A fragment")
        rC8 = reg_tile("f16", "8x8 HMMA-B fragment")
        copy_s2r(sInv.diag16(block16).D8, rD8)
        # instruction_selection: ldmatrix.sync.aligned.m8n8.x1.shared.b16; extent: one broadcast 8x8 tile
        copy_s2r(sInv.diag16(block16).C8, rC8)
        # instruction_selection: ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16; extent: one 8x8 tile
        rDC = reg_tile("f32", [16,8])
        fill(rDC, 0.0)
        # instruction_selection: mov.b32/mov.f32 zero initialization; extent: 16x8 accumulator
        gemm(rDC, rD8, rC8, shape=(16,8,8), accumulate=False)
        # instruction_selection: mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32; extent: one tiled MMA
        neg(rDC, rDC)
        # instruction_selection: xor.b32/neg.f32; extent: 16x8 accumulator fragment
        rDC16 = cast("f16", rDC)
        # instruction_selection: cvt.rn.f16x2.f32; extent: packed pairs across the 16x8 fragment
        rA8 = reg_tile("f16", "8x8 HMMA-B fragment")
        copy_s2r(sInv.diag16(block16).A8, rA8)
        # instruction_selection: ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16; extent: one 8x8 tile
        rO16 = reg_tile("f32", [16,8])
        fill(rO16, 0.0)
        # instruction_selection: mov.b32/mov.f32 zero initialization; extent: 16x8 accumulator
        gemm(rO16, rDC16, rA8, shape=(16,8,8), accumulate=False)
        # instruction_selection: mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32; extent: one tiled MMA
        rO16f = cast("f16", rO16)
        # instruction_selection: cvt.rn.f16x2.f32; extent: packed pairs across the 16x8 fragment
        copy_r2s(rO16f.first_m_slice, sInv.diag16(block16).lower_left8)
        # instruction_selection: stmatrix.sync.aligned.m8n8.x1.shared.b16; extent: one 8x8 tile
        barrier(id=6, threads=128)
        # instruction_selection: bar.sync 6,128; extent: named CTA barrier

        # Level 3: warps 0..1 extend two diagonal 16x16 pairs to 32x32.
        if tid < 64:
            block32 = tid // 32
            rD16 = reg_tile("f16", "16x16 HMMA-A fragment")
            rC16 = reg_tile("f16", "16x16 HMMA-B fragment")
            copy_s2r(sInv.diag32(block32).D16, rD16)
            # instruction_selection: ldmatrix.sync.aligned.m8n8.x2.shared.b16; extent: 16x16 tiled load
            copy_s2r(sInv.diag32(block32).C16, rC16)
            # instruction_selection: ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16; extent: 16x16 tiled load
            rDC32 = reg_tile("f32", [16,16])
            fill(rDC32, 0.0)
            # instruction_selection: mov.b32/mov.f32 zero initialization; extent: 16x16 accumulator
            gemm(rDC32, rD16, rC16, shape=(16,16,16), accumulate=False)
            # instruction_selection: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32; extent: 16x16 tiled loop
            neg(rDC32, rDC32)
            # instruction_selection: xor.b32/neg.f32; extent: 16x16 accumulator
            rDC32f = cast("f16", rDC32)
            # instruction_selection: cvt.rn.f16x2.f32; extent: packed pairs across the 16x16 fragment
            rA16 = reg_tile("f16", "16x16 HMMA-B fragment")
            copy_s2r(sInv.diag32(block32).A16, rA16)
            # instruction_selection: ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16; extent: 16x16 tiled load
            rO32 = reg_tile("f32", [16,16])
            fill(rO32, 0.0)
            # instruction_selection: mov.b32/mov.f32 zero initialization; extent: 16x16 accumulator
            gemm(rO32, rDC32f, rA16, shape=(16,16,16), accumulate=False)
            # instruction_selection: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32; extent: 16x16 tiled loop
            rO32f = cast("f16", rO32)
            # instruction_selection: cvt.rn.f16x2.f32; extent: packed pairs across the 16x16 fragment
            copy_r2s(rO32f, sInv.diag32(block32).lower_left16)
            # instruction_selection: stmatrix.sync.aligned.m8n8.x2.shared.b16; extent: 16x16 tile
        barrier(id=6, threads=128)
        # instruction_selection: bar.sync 6,128; extent: named CTA barrier

        # Level 4: all four warps extend the two 32x32 halves to 64x64.
        x = (tid // 32) // 2
        y = (tid // 32) % 2
        rD32 = reg_tile("f16", "per-warp 16x32 HMMA-A fragment")
        rC32 = reg_tile("f16", "per-warp 32x16 HMMA-B fragment")
        copy_s2r(sInv.D32.part(y), rD32)
        # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.shared.b16; extent: per-warp 16x32 tile
        copy_s2r(sInv.C32.part(x), rC32)
        # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16; extent: per-warp 32x16 tile
        rDC64 = reg_tile("f32", [16,16])
        fill(rDC64, 0.0)
        # instruction_selection: mov.b32/mov.f32 zero initialization; extent: 16x16 accumulator
        gemm(rDC64, rD32, rC32, shape=(16,16,32), accumulate=False)
        # instruction_selection: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32; extent: two K phases
        neg(rDC64, rDC64)
        # instruction_selection: xor.b32/neg.f32; extent: 16x16 accumulator
        rDC64f = cast("f16", rDC64)
        # instruction_selection: cvt.rn.f16x2.f32; extent: packed pairs across the 16x16 fragment
        rA32 = reg_tile("f16", "per-warp 16x32 HMMA-B fragment")
        copy_s2r(sInv.A32.part(x), rA32)
        # instruction_selection: ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16; extent: per-warp 16x32 operand loop
        rO64 = reg_tile("f32", [16,32])
        fill(rO64, 0.0)
        # instruction_selection: mov.b32/mov.f32 zero initialization; extent: 16x32 accumulator
        gemm(rO64, rDC64f, rA32, shape=(16,32,16), accumulate=False)
        # instruction_selection: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32; extent: 16x32 tiled loop
        rO64f = cast("f16", rO64)
        # instruction_selection: cvt.rn.f16.f32; extent: 16x32 fragment
        barrier(id=6, threads=128)
        # instruction_selection: bar.sync 6,128; extent: named CTA barrier
        if x == 0:
            copy_r2s(rO64f, sInv.lower_left32.part(y))
            # instruction_selection: stmatrix.sync.aligned.m8n8.x4.shared.b16; extent: per-warp 16x32 partial
        barrier(id=6, threads=128)
        # instruction_selection: bar.sync 6,128; extent: named CTA barrier
        if x == 1:
            rPartial = reg_tile("f16", [16,32])
            copy_s2r(sInv.lower_left32.part(y), rPartial)
            # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.shared.b16; extent: per-warp 16x32 partial
            add(rO64f, rO64f, rPartial)
            # instruction_selection: add.f16; extent: scalar loop across the 16x32 fragment
            copy_r2s(rO64f, sInv.lower_left32.part(y))
            # instruction_selection: stmatrix.sync.aligned.m8n8.x4.shared.b16; extent: per-warp 16x32 final

        barrier(id=6, threads=128)
        # instruction_selection: bar.sync 6,128; extent: named CTA barrier
        rInv = reg_tile("f16", [64,64])
        copy_s2r(sInv, rInv)
        # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.shared.b16; extent: 64x64 tiled load
        rT = reg_tile(IO_DTYPE, [64,64])
        for element in rT:
            col, row = coordinate(element)
            value = 0.0
            if row < valid_len and col < valid_len:
                value = mul(neg(sBeta[row]), cast("f32", rInv[element]))
                # instruction_selection: cvt.f32.f16 + mul.rn.f32/neg; extent: scalar per valid element
            rT[element] = cast(IO_DTYPE, value)
            # instruction_selection: cvt.rn.{f16,bf16}.f32; extent: scalar per owned element
        copy_r2g(rT, T_out[t_block,sab_head,:,:])
        # instruction_selection: st.global.b16; extent: scalar per owned output element across one 64x64 tile
        release(beta_cons)
        # instruction_selection: mbarrier.arrive.shared.b64; extent: one consumer release
```

# Launch 2: tcgen05 M/N precompute

The source uses one CTA per `(sequence, SAB head, CP-chunk slot)`. Invalid
slots still participate in TMEM allocation/free, but perform no load or math.

```python
@kernel(
    grid=(H_STATE * max_cp_chunks_per_seq, num_sequences, 1),
    block=(384,1,1), num_warps=12, cluster=(1,1,1), cta_group=1,
    tmem_columns=512, min_blocks_per_sm=1, target="sm_100a",
)
def _mn_precompute_sm100(K, V, T_in, Alpha, M_out, N_out, Cu,
                         cp_chunk_len, K_HEADS, V_HEADS, H_STATE,
                         total_cp_chunks, num_sequences):
    BLK = 64
    D = 128
    tid = thread_id_x()
    warp = warp_uniform(tid // 32)
    lane = tid % 32
    bx = block_id_x()
    seq = block_id_y()
    sab_head = bx % H_STATE
    chunk_in_seq = bx // H_STATE
    k_head = sab_head * K_HEADS // H_STATE
    v_head = sab_head * V_HEADS // H_STATE
    seq_start = copy_g2r(Cu[seq], reg("seq_start"))
    # instruction_selection: ld.global.b32; extent: scalar; INT64 specialization changes address stride/offset only
    seq_end = copy_g2r(Cu[seq+1], reg("seq_end"))
    # instruction_selection: ld.global.b32; extent: scalar; INT64 specialization changes address stride/offset only
    num_cp_chunks = ceil_div(seq_end-seq_start, cp_chunk_len)
    valid_work = chunk_in_seq < num_cp_chunks
    tok0 = seq_start + chunk_in_seq * cp_chunk_len
    chunk_len = min(cp_chunk_len, seq_end-tok0)
    num_blocks = ceil_div(chunk_len, 64)
    cp_chunk = varlen_chunk_idx(seq, seq_start, chunk_in_seq, cp_chunk_len)
    t_block0 = varlen_chunk_idx(
        seq, seq_start, chunk_in_seq * ceil_div(cp_chunk_len,64), 64)

    # Shared struct order and alignments are source order. Stage is the last
    # physical layout mode; the semantic aliases below group M/N before stage.
    load_k_bar = tile("smem", "mbarrier", [3], storage_words=6, initial_phase=0)
    load_v_bar = tile("smem", "mbarrier", [3], storage_words=6, initial_phase=0)
    load_t_bar = tile("smem", "mbarrier", [3], storage_words=6, initial_phase=0)
    alpha_bar = tile("smem", "mbarrier", [4], storage_words=8, initial_phase=0)
    m_init_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    n_init_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    x_acc_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    x_ready_bar = tile("smem", "mbarrier", [2], storage_words=4, initial_phase=0)
    m_in_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    n_in_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    z_acc_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    z_ready_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    m_acc_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    y_acc_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    y_ready_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    n_acc_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    done_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    tmem_holding = tile("smem", "i32", [1])
    sK = tile("smem", IO_DTYPE, [3,128,64], alignment=1024,
              layout="tcgen05 X-A and Z-B aliases")
    sK_trans = alias(sK, IO_DTYPE, [3,128,64],
                     layout="tcgen05 K-transpose B view",
                     lifetime="until both K consumers release")
    sV = tile("smem", IO_DTYPE, [3,128,64], alignment=1024,
              layout="tcgen05 X-A semantic layout")
    sT = tile("smem", IO_DTYPE, [3,64,64], alignment=1024,
              layout="tcgen05 X-B semantic layout")
    sX = tile("smem", IO_DTYPE, [2,128,64], alignment=1024,
              layout="tcgen05 update-B semantic layout")
    sAlpha = tile("smem", "f32", [4,64,3], alignment=16,
                  channels=("cumsum_log","cumprod","neg_end_rcp"))

    # TMEM offsets are columns and are not aliases while their lifetimes overlap.
    tM = tile("tmem", "f32", [128,128], column_offset=0, columns=128)
    tN = tile("tmem", "f32", [128,128], column_offset=128, columns=128)
    tScratch = tile("tmem", "f32", [128,64], column_offset=256, columns=64,
                    lifetime="Z or Y accumulator")
    tMInp = tile("tmem", IO_DTYPE, [128,128], column_offset=320, columns=64)
    tNInp = tile("tmem", IO_DTYPE, [128,128], column_offset=384, columns=64)
    tXY = tile("tmem", "f32", [2,128,64], column_offset=448, columns=64,
               lifetime="X/Y accumulator ring")

    pipe_init(load_k_bar, stages=3, kind="TmaUmma", producers=1, consumers=2,
              expected_bytes=128*64*bytes(IO_DTYPE))
    # instruction_selection: mbarrier.init.shared.b64 with TMA/UTMMA phase initialization; extent: three-stage K pipe
    pipe_init(load_v_bar, stages=3, kind="TmaAsync", producers=1, consumers=4,
              expected_bytes=128*64*bytes(IO_DTYPE))
    # instruction_selection: mbarrier.init.shared.b64 with async phase initialization; extent: three-stage V pipe
    pipe_init(load_t_bar, stages=3, kind="TmaUmma", producers=1, consumers=1,
              expected_bytes=64*64*bytes(IO_DTYPE))
    # instruction_selection: mbarrier.init.shared.b64 with TMA/UTMMA phase initialization; extent: three-stage T pipe
    pipe_init(alpha_bar, stages=4, kind="Async", producers=32, consumers=256)
    # instruction_selection: mbarrier.init.shared.b64; extent: four-stage alpha pipe
    pipe_init(m_init_bar, n_init_bar, x_acc_bar, x_ready_bar, m_in_bar, n_in_bar,
              z_acc_bar, z_ready_bar, m_acc_bar, y_acc_bar, y_ready_bar,
              n_acc_bar, done_bar, kinds="source AsyncUmma/UmmaAsync/Async")
    # instruction_selection: mbarrier.init.shared.b64; extent: full/empty barrier pairs for the exact one/two-stage math pipes
    pipeline_init_arrive(cluster=(1,1), relaxed=True)
    # instruction_selection: fence.mbarrier_init.release.cluster; extent: one deferred MN pipeline-initialization fence for the fixed (1,1) cluster
    pipeline_init_wait(cluster=(1,1))
    # instruction_selection: bar.sync 0; extent: one CTA sync completing MN pipeline initialization

    tmem_allocate(tmem_holding, columns=512,
                  allocator_warp=4, waiters=(warps(0,1,2,3,4,5,6,7,8,11)))
    # instruction_selection: tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32; extent: 512-column allocation

    if not valid_work:
        setmaxnreg("decrease", 24)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: current warp
        if warp in (0,1,2,3,4,5,6,7,8,11):
            tmem_wait_for_alloc()
            # instruction_selection: bar.sync 1,320; extent: the ten participating MN allocator-wait warps
        if warp == 4:
            tmem_relinquish()
            # instruction_selection: tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned; extent: CTA allocation permit
            tmem_free()
            # instruction_selection: tcgen05.dealloc.cta_group::1.sync.aligned.b32; extent: 512 columns

    elif 0 <= warp <= 3:
        # CG0 owns M and all X/Z format conversions.
        setmaxnreg("increase", 216)
        # instruction_selection: setmaxnreg.inc.sync.aligned.u32; extent: current warp
        tmem_wait_for_alloc()
        # instruction_selection: bar.sync 1,320; extent: CG0 participation in the ten-warp MN allocator barrier

        rM0 = reg_tile("f32", "32x32 TMEM-store partition of 128x128")
        for element in rM0:
            row, col = coordinate(element)
            rM0[element] = select(row == col, 1.0, 0.0)
            # instruction_selection: setp.eq + selp.b32; extent: scalar per owned M element
        copy_r2t(rM0, tM)
        # instruction_selection: tcgen05.st.sync.aligned.32x32b.x32.b32; extent: 128x128 identity tile
        fence("tmem_store")
        # instruction_selection: tcgen05.wait::st.sync.aligned; extent: TMEM stores
        m_init = acquire(m_init_bar, producer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one-stage acquire
        commit(m_init)
        # instruction_selection: mbarrier.arrive.shared.b64; extent: M initialized

        for blk in range(num_blocks):
            alpha_h = wait(alpha_bar, consumer="M")
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: alpha stage wait
            x_acc_h = wait(x_acc_bar, consumer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: X accumulator wait
            x_ready_h = acquire(x_ready_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: X stage acquire
            rX = reg_tile("f32", [128,64])
            copy_t2r(tXY[x_acc_h.index], rX)
            # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x8.b32; extent: one 128x64 FP32 tile
            rXio = cast(IO_DTYPE, rX)
            # instruction_selection: cvt.rn.{f16,bf16}x2.f32; extent: 32 packed-pair issues per CG0 thread across one 128x64 fragment
            copy_r2s(rXio, sX[x_ready_h.index])
            # instruction_selection: stmatrix.sync.aligned.m8n8.x4.trans.shared.b16; extent: one 128x64 tile
            fence("async_shared_view")
            # instruction_selection: fence.proxy.async.shared::cta; extent: sX publication
            release(x_acc_h)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: X accumulator reuse
            commit(x_ready_h)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: sX stage publication

            if blk > 0:
                rM = reg_tile("f32", [128,128])
                copy_t2r(tM, rM)
                # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x32.b32; extent: one 128x128 tile
                rMio = cast(IO_DTYPE, rM)
                # instruction_selection: cvt.rn.{f16,bf16}x2.f32; extent: 64 packed-pair issues per CG0 thread across one 128x128 fragment
                copy_r2t(rMio, tMInp)
                # instruction_selection: tcgen05.st.sync.aligned.32x32b.x16.b32; extent: one 128x128 operand tile
                fence("tmem_store")
                # instruction_selection: tcgen05.wait::st.sync.aligned; extent: TMEM stores
                m_in = acquire(m_in_bar, producer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: M-input acquire
                commit(m_in)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: M-input publication
                z_acc = wait(z_acc_bar, consumer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: Z accumulator wait
                rZ = reg_tile("f32", [128,64])
                copy_t2r(tScratch, rZ)
                # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x8.b32; extent: one 128x64 Z tile
                rZio = cast(IO_DTYPE, rZ)
                # instruction_selection: cvt.rn.{f16,bf16}x2.f32; extent: 32 packed-pair issues per CG0 thread across one 128x64 fragment
                copy_r2t(rZio, tMInp)
                # instruction_selection: tcgen05.st.sync.aligned.16x128b.x8.b32; extent: one 128x64 operand tile
                fence("tmem_store")
                # instruction_selection: tcgen05.wait::st.sync.aligned; extent: TMEM stores
                release(z_acc)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: Z accumulator reuse
                z_ready = acquire(z_ready_bar, producer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: Z-ready acquire
                commit(z_ready)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: Z-input publication

            block_coeff = copy_s2r(sAlpha[alpha_h.index,63,"cumprod"], reg("gamma_end"))
            # instruction_selection: ld.shared.b32; extent: scalar
            m_acc = wait(m_acc_bar, consumer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: M accumulator wait
            rMscale = reg_tile("f32", [128,128])
            copy_t2r(tM, rMscale)
            # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x32.b32; extent: one 128x128 tile
            mul(rMscale, rMscale, block_coeff)
            # instruction_selection: mul.rn.f32; extent: one 128x128 register-fragment loop
            copy_r2t(rMscale, tM)
            # instruction_selection: tcgen05.st.sync.aligned.32x32b.x32.b32; extent: one 128x128 tile
            fence("tmem_store")
            # instruction_selection: tcgen05.wait::st.sync.aligned; extent: TMEM stores
            release(m_acc)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: M accumulator reuse
            release(alpha_h)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: alpha stage reuse by M consumer

        rMout = reg_tile("f32", [128,128])
        copy_t2r(tM, rMout)
        # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x32.b32; extent: one 128x128 tile
        copy_r2g(rMout, M_out[cp_chunk,sab_head,:,:])
        # instruction_selection: st.global.v4.b32 vector family; extent: one 128x128 FP32 tile
        done = acquire(done_bar, producer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: done acquire
        commit(done)
        # instruction_selection: mbarrier.arrive.shared.b64; extent: M store complete

    elif 4 <= warp <= 7:
        # CG1 owns N, Y postprocessing, and the final TMEM free.
        setmaxnreg("increase", 216)
        # instruction_selection: setmaxnreg.inc.sync.aligned.u32; extent: current warp
        tmem_wait_for_alloc()
        # instruction_selection: bar.sync 1,320; extent: CG1 participation in the ten-warp MN allocator barrier
        rN0 = reg_tile("f32", [128,128])
        fill(rN0, 0.0)
        # instruction_selection: mov.b32/mov.f32 zero initialization; extent: 128x128 fragment
        copy_r2t(rN0, tN)
        # instruction_selection: tcgen05.st.sync.aligned.32x32b.x32.b32; extent: one 128x128 tile
        fence("tmem_store")
        # instruction_selection: tcgen05.wait::st.sync.aligned; extent: TMEM stores
        n_init = acquire(n_init_bar, producer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: N-init acquire
        commit(n_init)
        # instruction_selection: mbarrier.arrive.shared.b64; extent: N initialized

        for blk in range(num_blocks):
            v_h = wait(load_v_bar, consumer="CG1")
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: V stage wait
            alpha_h = wait(alpha_bar, consumer="N")
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: alpha stage wait
            rN = reg_tile("f32", [128,128])
            copy_t2r(tN, rN)
            # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x32.b32; extent: one 128x128 tile
            rNio = cast(IO_DTYPE, rN)
            # instruction_selection: cvt.rn.{f16,bf16}x2.f32; extent: 64 packed-pair issues per CG1 thread across one 128x128 fragment
            copy_r2t(rNio, tNInp)
            # instruction_selection: tcgen05.st.sync.aligned.32x32b.x16.b32; extent: one 128x128 operand tile
            fence("tmem_store")
            # instruction_selection: tcgen05.wait::st.sync.aligned; extent: TMEM stores
            n_in = acquire(n_in_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: N-input acquire
            commit(n_in)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: N-input publication
            y_acc = wait(y_acc_bar, consumer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: Y accumulator wait
            block_coeff = copy_s2r(sAlpha[alpha_h.index,63,"cumprod"], reg("gamma_end"))
            # instruction_selection: ld.shared.b32; extent: one scalar after the Y wait
            rY = reg_tile("f32", [128,64])
            copy_t2r(tScratch, rY)
            # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x8.b32; extent: one 128x64 Y tile
            rV = reg_tile(IO_DTYPE, [128,64])
            copy_s2r(sV[v_h.index], rV)
            # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16; extent: one 128x64 V tile
            for element in rY:
                _, token = coordinate(element)
                v_term = mul(cast("f32", rV[element]),
                             sAlpha[alpha_h.index,token,"neg_end_rcp"])
                # instruction_selection: cvt.f32.{f16,bf16} + mul.rn.f32; extent: scalar per Y element
                rY[element] = fma(block_coeff, rY[element], v_term)
                # instruction_selection: fma.rn.f32; extent: scalar per Y element
            rYio = cast(IO_DTYPE, rY)
            # instruction_selection: cvt.rn.{f16,bf16}x2.f32; extent: 32 packed-pair issues per CG1 thread across one 128x64 fragment
            copy_r2t(rYio, tNInp)
            # instruction_selection: tcgen05.st.sync.aligned.16x128b.x8.b32; extent: one 128x64 operand tile
            fence("tmem_store")
            # instruction_selection: tcgen05.wait::st.sync.aligned; extent: converted Y TMEM stores before Y-accumulator release
            release(y_acc)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: Y accumulator reuse
            mul(rN, rN, block_coeff)
            # instruction_selection: mul.rn.f32; extent: one 128x128 register-fragment loop
            copy_r2t(rN, tN)
            # instruction_selection: tcgen05.st.sync.aligned.32x32b.x32.b32; extent: one 128x128 tile
            fence("tmem_store")
            # instruction_selection: tcgen05.wait::st.sync.aligned; extent: N-scale helper TMEM stores
            fence("tmem_store")
            # instruction_selection: tcgen05.wait::st.sync.aligned; extent: caller synchronization immediately after N-scale helper
            y_ready = acquire(y_ready_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: Y-ready acquire
            commit(y_ready)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: Y-input publication
            n_acc = wait(n_acc_bar, consumer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: N accumulator wait
            release(n_acc)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: N accumulator reuse
            release(v_h)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: V stage reuse
            release(alpha_h)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: alpha stage reuse by N consumer

        rNout = reg_tile("f32", [128,128])
        copy_t2r(tN, rNout)
        # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x32.b32; extent: one 128x128 tile
        copy_r2g(rNout, N_out[cp_chunk,sab_head,:,:])
        # instruction_selection: st.global.v4.b32 vector family; extent: one 128x128 FP32 tile
        done_h = wait(done_bar, consumer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: wait for M store
        release(done_h)
        # instruction_selection: mbarrier.arrive.shared.b64; extent: done-barrier reuse
        tmem_relinquish()
        # instruction_selection: tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned; extent: CTA allocation permit
        tmem_free()
        # instruction_selection: tcgen05.dealloc.cta_group::1.sync.aligned.b32; extent: 512 columns

    elif warp == 8:
        # Transfer MMA issuer: Z=M*K^T and M += Z*X; first block uses K*X.
        setmaxnreg("decrease", 72)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: current warp
        descriptor_prefetch(K.tensor_map, V.tensor_map, T_in.tensor_map)
        # instruction_selection: prefetch.tensormap; extent: three descriptors
        tmem_wait_for_alloc()
        # instruction_selection: bar.sync 1,320; extent: warp 8 participation in the ten-warp MN allocator barrier
        m_init = wait(m_init_bar, consumer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: M-init wait
        release(m_init)
        # instruction_selection: mbarrier.arrive.shared.b64; extent: M-init reuse
        for blk in range(num_blocks):
            k_h = wait(load_k_bar, consumer="transfer")
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: K stage wait
            if blk > 0:
                m_in = wait(m_in_bar, consumer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: M-input wait
                z_acc = acquire(z_acc_bar, producer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: Z-acc acquire
                gemm(tScratch, tMInp, sK_trans[k_h.index], shape=(128,64,128),
                     dtype=IO_DTYPE, accumulate=False)
                # instruction_selection: tcgen05.mma.cta_group::1.kind::f16; extent: 128x64x128, eight K phases
                commit(z_acc)
                # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64; extent: Z tile completion
                z_ready = wait(z_ready_bar, consumer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: converted Z wait
                release(z_ready)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: Z-input reuse
                release(m_in)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: M-input reuse
            x_h = wait(x_ready_bar, consumer="M")
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: X stage wait
            m_acc = acquire(m_acc_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: M-acc acquire
            if blk == 0:
                gemm(tM, sK[k_h.index], sX[x_h.index], shape=(128,128,64),
                     dtype=IO_DTYPE, accumulate=True)
                # instruction_selection: tcgen05.mma.cta_group::1.kind::f16; extent: 128x128x64, four K phases, accumulate
            else:
                gemm(tM, tMInp, sX[x_h.index], shape=(128,128,64),
                     dtype=IO_DTYPE, accumulate=True)
                # instruction_selection: tcgen05.mma.cta_group::1.kind::f16; extent: 128x128x64, four K phases, accumulate
            commit(m_acc)
            # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64; extent: M-update completion
            release(x_h)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: X stage reuse by M issuer
            release(k_h)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: K stage reuse by transfer issuer

    elif warp == 11:
        # State MMA issuer: X=K*T, Y=N*K^T, N += Y*X.
        setmaxnreg("decrease", 72)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: current warp
        tmem_wait_for_alloc()
        # instruction_selection: bar.sync 1,320; extent: warp 11 participation in the ten-warp MN allocator barrier
        n_init = wait(n_init_bar, consumer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: N-init wait
        release(n_init)
        # instruction_selection: mbarrier.arrive.shared.b64; extent: N-init reuse
        for blk in range(num_blocks):
            k_h = wait(load_k_bar, consumer="state")
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: K stage wait
            t_h = wait(load_t_bar, consumer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: T stage wait
            x_acc = acquire(x_acc_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: X-acc acquire
            gemm(tXY[x_acc.index], sK[k_h.index], sT[t_h.index],
                 shape=(128,64,64), dtype=IO_DTYPE, accumulate=False)
            # instruction_selection: tcgen05.mma.cta_group::1.kind::f16; extent: 128x64x64, four K phases
            commit(x_acc)
            # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64; extent: X completion
            release(t_h)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: T stage reuse
            x_h = wait(x_ready_bar, consumer="N")
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: materialized X wait
            n_in = wait(n_in_bar, consumer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: N-input wait
            y_acc = acquire(y_acc_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: Y-acc acquire
            gemm(tScratch, tNInp, sK_trans[k_h.index], shape=(128,64,128),
                 dtype=IO_DTYPE, accumulate=False)
            # instruction_selection: tcgen05.mma.cta_group::1.kind::f16; extent: 128x64x128, eight K phases
            commit(y_acc)
            # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64; extent: Y completion
            release(n_in)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: N-input reuse
            release(k_h)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: K stage reuse by state issuer
            y_ready = wait(y_ready_bar, consumer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: converted Y wait
            n_acc = acquire(n_acc_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: N-acc acquire
            gemm(tN, tNInp, sX[x_h.index], shape=(128,128,64),
                 dtype=IO_DTYPE, accumulate=True)
            # instruction_selection: tcgen05.mma.cta_group::1.kind::f16; extent: 128x128x64, four K phases, accumulate
            commit(n_acc)
            # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64; extent: N-update completion
            release(y_ready)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: Y-input reuse
            release(x_h)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: X stage reuse by N issuer

    elif warp == 9:
        setmaxnreg("decrease", 24)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: current warp
        for blk in range(num_blocks):
            k_h = acquire(load_k_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: K stage acquire
            expect_tx(k_h, 128*64*bytes(IO_DTYPE))
            # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one K-stage byte expectation before TMA issue
            copy_g2s(K[k_head,tok0+blk*64:tok0+(blk+1)*64,:], sK[k_h.index],
                     completion=k_h.barrier)
            # instruction_selection: cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint; extent: two issues covering one 128x64 K tile
            v_h = acquire(load_v_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: V stage acquire
            expect_tx(v_h, 128*64*bytes(IO_DTYPE))
            # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one V-stage byte expectation before TMA issue
            copy_g2s(V[v_head,tok0+blk*64:tok0+(blk+1)*64,:], sV[v_h.index],
                     completion=v_h.barrier)
            # instruction_selection: cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint; extent: two issues covering one 128x64 V tile
            t_h = acquire(load_t_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: T stage acquire
            expect_tx(t_h, 64*64*bytes(IO_DTYPE))
            # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one T-stage byte expectation before TMA issue
            copy_g2s(T_in[t_block0+blk,sab_head,:,:], sT[t_h.index],
                     completion=t_h.barrier)
            # instruction_selection: cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint; extent: one issue covering one 64x64 T tile

    elif warp == 10:
        setmaxnreg("decrease", 24)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: current warp
        for blk in range(num_blocks):
            alpha_h = acquire(alpha_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: alpha stage acquire
            fence("async_shared_view")
            # instruction_selection: fence.proxy.async.shared::cta; extent: alpha stage before shared writes
            for row in (lane, lane+32):
                a = 1.0
                if blk*64+row < chunk_len:
                    copy_g2r(Alpha[tok0+blk*64+row,sab_head], a)
                    # instruction_selection: ld.global.b32; extent: scalar, two iterations per lane
                copy_r2s(a, sAlpha[alpha_h.index,row,"cumsum_log"])
                # instruction_selection: st.shared.b32; extent: scalar, two iterations per lane
            x0 = copy_s2r(sAlpha[alpha_h.index,lane,"cumsum_log"], reg("alpha0"))
            # instruction_selection: ld.shared.b32; extent: first scalar per lane
            x1 = copy_s2r(sAlpha[alpha_h.index,lane+32,"cumsum_log"], reg("alpha1"))
            # instruction_selection: ld.shared.b32; extent: second scalar per lane
            x0 = log2(add(x0, 1.0e-10))
            # instruction_selection: add.rn.f32 + lg2.approx.ftz.f32; extent: first scalar per lane
            x1 = log2(add(x1, 1.0e-10))
            # instruction_selection: add.rn.f32 + lg2.approx.ftz.f32; extent: second scalar per lane
            for delta in (1,2,4,8,16):
                peer0 = shuffle_up(x0, delta, mask=0xffffffff, clamp=0)
                # instruction_selection: shfl.sync.up.b32; extent: first scalar in five-step inclusive scan
                peer1 = shuffle_up(x1, delta, mask=0xffffffff, clamp=0)
                # instruction_selection: shfl.sync.up.b32; extent: second scalar in five-step inclusive scan
                if lane >= delta:
                    x0 = add(x0, peer0)
                    # instruction_selection: add.rn.f32; extent: first scalar in five-step inclusive scan
                    x1 = add(x1, peer1)
                    # instruction_selection: add.rn.f32; extent: second scalar in five-step inclusive scan
            carry = shuffle_index(x0, 31, mask=0xffffffff, clamp=31)
            # instruction_selection: shfl.sync.idx.b32; extent: first-half terminal carry
            x1 = add(x1, carry)
            # instruction_selection: add.rn.f32; extent: second-half carry
            end_log = shuffle_index(x1, 31, mask=0xffffffff, clamp=31)
            # instruction_selection: shfl.sync.idx.b32; extent: block-end broadcast
            for slot,row,x in ((0,lane,x0),(1,lane+32,x1)):
                cp = exp2(x)
                # instruction_selection: ex2.approx.ftz.f32; extent: one scalar per lane-owned row
                neg_end_rcp = neg(exp2(sub(end_log, x)))
                # instruction_selection: sub.rn.f32 + ex2.approx.ftz.f32 + neg.f32; extent: one scalar per lane-owned row
                copy_r2s(x, sAlpha[alpha_h.index,row,"cumsum_log"])
                # instruction_selection: st.shared.b32; extent: one scalar per lane-owned row
                copy_r2s(cp, sAlpha[alpha_h.index,row,"cumprod"])
                # instruction_selection: st.shared.b32; extent: one scalar per lane-owned row
                copy_r2s(neg_end_rcp, sAlpha[alpha_h.index,row,"neg_end_rcp"])
                # instruction_selection: st.shared.b32; extent: one scalar per lane-owned row
                if blk*64+row >= chunk_len:
                    fill(sAlpha[alpha_h.index,row,"neg_end_rcp"], 0.0)
                    # instruction_selection: st.shared.b32; extent: one scalar OOB channel mask
            fence("async_shared_view")
            # instruction_selection: fence.proxy.async.shared::cta; extent: alpha stage
            commit(alpha_h)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: one alpha stage

    else:
        setmaxnreg("decrease", 24)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: current warp
```

# Launch 3a: production SIMT-row4 fixup

This branch is selected only for at most nine `(sequence,state-head)` states on
a 148-SM B200. It intentionally retains one thread per column, the source
16-float transfer prefetch, and the last-K-tile inter-chunk handoff.

```python
@kernel(
    grid=(num_sequences * H_STATE * 32,1,1),
    block=(128,1,1), num_warps=4, min_blocks_per_sm=2, target="sm_100a",
)
def _fixup_simt_row4_sm100(M, N, Initial, InitWorkspace, Fixed, Final,
                            StateIndices, Cu, cp_chunk_len,
                            total_cp_chunks, num_sequences, H_STATE):
    ROWS = 4
    ROW_CTAS = 32
    D = 128
    tid = thread_id_x()
    bx = block_id_x()
    row_cta = bx % ROW_CTAS
    head_seq = bx // ROW_CTAS
    head = head_seq % H_STATE
    seq = head_seq // H_STATE
    seq_start = copy_g2r(Cu[seq], reg("seq_start"))
    # instruction_selection: ld.global.b32; extent: scalar; INT64 specialization changes address stride/offset only
    seq_end = copy_g2r(Cu[seq+1], reg("seq_end"))
    # instruction_selection: ld.global.b32; extent: scalar; INT64 specialization changes address stride/offset only
    num_chunks = ceil_div(seq_end-seq_start, cp_chunk_len)
    chunk_start = varlen_chunk_idx(seq, seq_start, 0, cp_chunk_len)
    gap_start = chunk_start + num_chunks
    if seq+1 < num_sequences:
        gap_end = varlen_chunk_idx(seq+1, seq_end, 0, cp_chunk_len)
    else:
        gap_end = total_cp_chunks
    state_idx = seq
    if USE_STATE_INDICES:
        state_idx = copy_g2r(StateIndices[seq], reg("state_idx"))
        # instruction_selection: ld.global.s32; extent: scalar

    sState = tile("smem", "f32", [4,128], alignment=128,
                  lifetime="CTA recurrence state")
    col = tid
    global_row0 = row_cta * 4
    setmaxnreg("increase", 256)
    # instruction_selection: setmaxnreg.inc.sync.aligned.u32; extent: current warp
    barrier("cta")
    # instruction_selection: bar.sync 0; extent: full CTA

    if num_chunks > 0:
        start = 0
        if NEEDS_INITIAL_STATE:
            for row in range(4):
                value_state = copy_g2r(
                    Initial[state_idx,head,global_row0+row,col], reg("state", row))
                # instruction_selection: ld.global.{b16,b32}; extent: scalar, four-row loop
                value = cast("f32", value_state)
                # instruction_selection: cvt.f32.{f16,bf16} or mov.b32; extent: scalar, four-row loop
                copy_r2s(value, sState[row,col])
                # instruction_selection: st.shared.b32; extent: scalar, four-row loop
                copy_r2g(value, InitWorkspace[seq,head,global_row0+row,col])
                # instruction_selection: st.global.b32; extent: scalar, four-row loop
        else:
            start = 1
            for row in range(4):
                value = copy_g2r(N[chunk_start,head,global_row0+row,col], reg("state", row))
                # instruction_selection: ld.global.b32; extent: scalar, four-row loop
                copy_r2s(value, sState[row,col])
                # instruction_selection: st.shared.b32; extent: scalar, four-row loop
                copy_r2g(value, Fixed[chunk_start,head,global_row0+row,col])
                # instruction_selection: st.global.b32; extent: scalar, four-row loop
        barrier("cta")
        # instruction_selection: bar.sync 0; extent: full CTA

        rAcc = reg_tile("f32", [4])
        rAccNext = reg_tile("f32", [4])
        rM = reg_tile("f32", [16])
        rMNext = reg_tile("f32", [16])
        if start < num_chunks:
            for row in range(4):
                copy_g2r(N[chunk_start+start,head,global_row0+row,col], rAcc[row])
                # instruction_selection: ld.global.b32; extent: scalar, four-row loop
            for j in range(16):
                copy_g2r(M[chunk_start+start,head,j,col], rM[j])
                # instruction_selection: ld.global.b32; extent: scalar, 16-value transfer tile

        for chunk in range(start, num_chunks):
            next_chunk = chunk + 1
            for ktile in range(7):
                for j in range(16):
                    copy_g2r(M[chunk_start+chunk,head,(ktile+1)*16+j,col], rMNext[j])
                    # instruction_selection: ld.global.b32; extent: scalar, 16-value prefetched K tile
                for row in range(4):
                    for j in range(16):
                        rAcc[row] = fma(sState[row,ktile*16+j], rM[j], rAcc[row])
                        # instruction_selection: fma.rn.f32; extent: 4x16 scalar accumulation loop
                copy_r2r(rMNext, rM)
                # instruction_selection: mov.b32; extent: 16-register handoff

            # The source prefetches both next N and its first M tile before the
            # final K accumulation, then publishes the current state.
            if next_chunk < num_chunks:
                for row in range(4):
                    copy_g2r(N[chunk_start+next_chunk,head,global_row0+row,col], rAccNext[row])
                    # instruction_selection: ld.global.b32; extent: scalar, four-row next-N prefetch
                for j in range(16):
                    copy_g2r(M[chunk_start+next_chunk,head,j,col], rMNext[j])
                    # instruction_selection: ld.global.b32; extent: scalar, 16-value next-M prefetch
            for row in range(4):
                for j in range(16):
                    rAcc[row] = fma(sState[row,112+j], rM[j], rAcc[row])
                    # instruction_selection: fma.rn.f32; extent: 4x16 scalar final-K accumulation loop
            barrier("cta")
            # instruction_selection: bar.sync 0; extent: full-CTA state publication fence
            for row in range(4):
                copy_r2s(rAcc[row], sState[row,col])
                # instruction_selection: st.shared.b32; extent: scalar, four-row loop
                copy_r2g(rAcc[row], Fixed[chunk_start+chunk,head,global_row0+row,col])
                # instruction_selection: st.global.b32; extent: scalar, four-row loop
            barrier("cta")
            # instruction_selection: bar.sync 0; extent: full-CTA next-chunk consumption fence
            if next_chunk < num_chunks:
                copy_r2r(rAccNext, rAcc)
                # instruction_selection: mov.b32; extent: four-register handoff
                copy_r2r(rMNext, rM)
                # instruction_selection: mov.b32; extent: 16-register handoff

        if STORE_FINAL_STATE:
            for row in range(4):
                value = copy_s2r(sState[row,col], reg("final", row))
                # instruction_selection: ld.shared.b32; extent: scalar, four-row loop
                value_out = cast(STATE_DTYPE, value)
                # instruction_selection: cvt.rn.{f16,bf16}.f32 or mov.b32; extent: scalar, four-row loop
                copy_r2g(value_out, Final[state_idx,head,global_row0+row,col])
                # instruction_selection: st.global.{b16,b32}; extent: scalar, four-row loop

    # Workspace padding between adjacent varlen sequences is explicitly zeroed.
    for slot in range(gap_start, gap_end):
        for row in range(4):
            fill(Fixed[slot,head,global_row0+row,col], 0.0)
            # instruction_selection: st.global.b32; extent: scalar, gap x four-row loop
```

# Launch 3b/3c: production UTCMMA-row64 and UTCMMA-row128 fixup

Both entry points use this exact source family. The only compile-time
differences are `ROWS`, `ROW_CTAS`, M-ring depth, register budget, TMEM copy
shape, and launch grid.

```python
@specialize(
    (_fixup_utcmma64_sm100,  ROWS=64,  ROW_CTAS=2, M_STAGES=2, COMPUTE_REGS=120),
    (_fixup_utcmma128_sm100, ROWS=128, ROW_CTAS=1, M_STAGES=1, COMPUTE_REGS=256),
)
@kernel(
    grid=(num_sequences * H_STATE * ROW_CTAS,1,1),
    block=(256,1,1), num_warps=8, cluster=(1,1,1),
    tmem_columns=256, min_blocks_per_sm=1, target="sm_100a",
)
def _fixup_utcmma_sm100(M, N, Initial, InitWorkspace, Fixed, Final,
                         StateIndices, Cu, cp_chunk_len,
                         total_cp_chunks, num_sequences, H_STATE):
    D = 128
    tid = thread_id_x()
    warp = warp_uniform(tid // 32)
    bx = block_id_x()
    row_cta = bx % ROW_CTAS
    head_seq = bx // ROW_CTAS
    head = head_seq % H_STATE
    seq = head_seq // H_STATE
    seq_start = copy_g2r(Cu[seq], reg("seq_start"))
    # instruction_selection: ld.global.b32; extent: scalar; INT64 specialization changes address stride/offset only
    seq_end = copy_g2r(Cu[seq+1], reg("seq_end"))
    # instruction_selection: ld.global.b32; extent: scalar; INT64 specialization changes address stride/offset only
    num_chunks = ceil_div(seq_end-seq_start, cp_chunk_len)
    chunk_start = varlen_chunk_idx(seq, seq_start, 0, cp_chunk_len)
    start = 0 if NEEDS_INITIAL_STATE else 1
    num_iters = num_chunks - start
    state_idx = seq
    if USE_STATE_INDICES:
        state_idx = copy_g2r(StateIndices[seq], reg("state_idx"))
        # instruction_selection: ld.global.s32; extent: scalar

    m_bar = tile("smem", "mbarrier", [M_STAGES], storage_words=2*M_STAGES,
                 initial_phase=0)
    n_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    ready_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    done_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    tmem_holding = tile("smem", "i32", [1])
    sM = tile("smem", "f32", [M_STAGES,128,128], alignment=1024,
              layout="tcgen05 TF32 B layout")
    sN = tile("smem", "f32", [1,ROWS,128], alignment=1024,
              layout="tcgen05 TF32 A semantic layout")
    tAcc = tile("tmem", "f32", [ROWS,128], column_offset=0, columns=128)
    tOpd = tile("tmem", "tf32", [ROWS,128], column_offset=128, columns=128)

    pipe_init(m_bar, stages=M_STAGES, kind="TmaUmma", producers=1, consumers=1,
              expected_bytes=128*128*4)
    # instruction_selection: mbarrier.init.shared.b64 with TMA/UTMMA phase initialization; extent: one/two-stage M pipe
    pipe_init(n_bar, stages=1, kind="TmaAsync", producers=1, consumers=4,
              expected_bytes=ROWS*128*4)
    # instruction_selection: mbarrier.init.shared.b64; extent: one-stage N pipe
    pipe_init(ready_bar, stages=1, kind="AsyncUmma", producers=128, consumers=1)
    # instruction_selection: mbarrier.init.shared.b64; extent: compute-to-MMA ready pipe
    pipe_init(done_bar, stages=1, kind="UmmaAsync", producers=1, consumers=128)
    # instruction_selection: mbarrier.init.shared.b64; extent: MMA-to-compute done pipe
    pipeline_init_arrive(cluster=(1,1), relaxed=True)
    # instruction_selection: fence.mbarrier_init.release.cluster; extent: one deferred UTC pipeline-initialization fence for the fixed (1,1) cluster
    pipeline_init_wait(cluster=(1,1))
    # instruction_selection: bar.sync 0; extent: one CTA sync completing UTC pipeline initialization
    tmem_allocate(tmem_holding, columns=256, allocator_warp=0,
                  waiters=warps(0,1,2,3,4))
    # instruction_selection: tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32; extent: 256-column allocation
    if warp <= 4:
        tmem_wait_for_alloc()
        # instruction_selection: bar.sync 1,160; extent: warps 0..4 immediately after TMEM allocation

    if num_chunks == 0:
        setmaxnreg("decrease", 32)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: current warp
        if warp == 0:
            tmem_relinquish()
            # instruction_selection: tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned; extent: CTA allocation permit
            tmem_free()
            # instruction_selection: tcgen05.dealloc.cta_group::1.sync.aligned.b32; extent: 256 columns

    elif 0 <= warp <= 3:
        setmaxnreg("increase", COMPUTE_REGS)
        # instruction_selection: setmaxnreg.inc.sync.aligned.u32; extent: current warp
        state_stage = 0
        if NEEDS_INITIAL_STATE:
            rStateIn = reg_tile(STATE_DTYPE, [ROWS,128])
            copy_g2r(Initial[state_idx,head,row_cta*ROWS:(row_cta+1)*ROWS,:], rStateIn)
            # instruction_selection: ld.global.b16 loop for FP16/BF16 or ld.global.b32 loop for FP32; extent: one ROWSx128 state tile
            rState = cast("f32", rStateIn)
            # instruction_selection: cvt.f32.{f16,bf16} or mov.b32; extent: one ROWSx128 tile
            copy_r2t(rState, tAcc)
            # instruction_selection: tcgen05.st.sync.aligned.{16x32bx2,32x32b}; extent: one ROWSx128 FP32 tile
            fence("tmem_store")
            # instruction_selection: tcgen05.wait::st.sync.aligned; extent: TMEM stores
            copy_t2r(tAcc, rState)
            # instruction_selection: tcgen05.ld.sync.aligned.{16x32bx2,32x32b}; extent: one ROWSx128 tile
            copy_r2g(rState, InitWorkspace[seq,head,row_cta*ROWS:(row_cta+1)*ROWS,:])
            # instruction_selection: st.global.v4.b32 vector family; extent: one ROWSx128 FP32 tile
        else:
            n0 = wait(n_bar, consumer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: first N wait
            rN0 = reg_tile("f32", [ROWS,128])
            copy_s2r(sN[n0.index], rN0)
            # instruction_selection: ld.shared.v4.b32 vector family; extent: one ROWSx128 tile
            copy_r2t(rN0, tAcc)
            # instruction_selection: tcgen05.st.sync.aligned.{16x32bx2,32x32b}; extent: one ROWSx128 FP32 tile
            fence("tmem_store")
            # instruction_selection: tcgen05.wait::st.sync.aligned; extent: TMEM stores
            rFirst = reg_tile("f32", [ROWS,128])
            copy_t2r(tAcc, rFirst)
            # instruction_selection: tcgen05.ld.sync.aligned.{16x32bx2,32x32b}; extent: one ROWSx128 tile
            copy_r2g(rFirst, Fixed[chunk_start,head,row_cta*ROWS:(row_cta+1)*ROWS,:])
            # instruction_selection: st.global.v4.b32 vector family; extent: one ROWSx128 FP32 tile
            release(n0)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: first N reuse

        for chunk in range(start, num_chunks):
            rAcc = reg_tile("f32", [ROWS,128])
            copy_t2r(tAcc, rAcc)
            # instruction_selection: tcgen05.ld.sync.aligned.{16x32bx2,32x32b}; extent: one ROWSx128 tile
            rOpd = cast("tf32", rAcc)
            # instruction_selection: cvt.rna.tf32.f32; extent: scalar loop across one ROWSx128 tile
            copy_r2t(rOpd, tOpd)
            # instruction_selection: tcgen05.st.sync.aligned.{16x32bx2,32x32b}; extent: one ROWSx128 TF32 operand tile
            fence("tmem_store")
            # instruction_selection: tcgen05.wait::st.sync.aligned; extent: TMEM stores
            n_h = wait(n_bar, consumer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: N stage wait
            ready = acquire(ready_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: ready acquire
            rN = reg_tile("f32", [ROWS,128])
            copy_s2r(sN[n_h.index], rN)
            # instruction_selection: ld.shared.v4.b32 vector family; extent: one ROWSx128 tile
            copy_r2t(rN, tAcc)
            # instruction_selection: tcgen05.st.sync.aligned.{16x32bx2,32x32b}; extent: one ROWSx128 FP32 tile
            fence("tmem_store")
            # instruction_selection: tcgen05.wait::st.sync.aligned; extent: TMEM stores
            commit(ready)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: TMEM operands/accumulator ready
            release(n_h)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: N stage reuse
            done = wait(done_bar, consumer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: recurrence MMA completion
            rFixed = reg_tile("f32", [ROWS,128])
            copy_t2r(tAcc, rFixed)
            # instruction_selection: tcgen05.ld.sync.aligned.{16x32bx2,32x32b}; extent: one ROWSx128 tile
            copy_r2g(rFixed, Fixed[chunk_start+chunk,head,row_cta*ROWS:(row_cta+1)*ROWS,:])
            # instruction_selection: st.global.v4.b32 vector family; extent: one ROWSx128 FP32 tile
            release(done)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: done-stage reuse

        if STORE_FINAL_STATE:
            rFinal = reg_tile("f32", [ROWS,128])
            copy_t2r(tAcc, rFinal)
            # instruction_selection: tcgen05.ld.sync.aligned.{16x32bx2,32x32b}; extent: one ROWSx128 tile
            copy_r2g(rFinal, Fixed[chunk_start+num_chunks-1,head,row_cta*ROWS:(row_cta+1)*ROWS,:])
            # instruction_selection: st.global.v4.b32 vector family; extent: repeated final Fixed ROWSx128 FP32 tile
            rFinalOut = cast(STATE_DTYPE, rFinal)
            # instruction_selection: cvt.rn.{f16,bf16}x2.f32 for FP16/BF16, with no standalone conversion for the FP32 alias; extent: 32 packed-pair issues per compute thread for ROWS=64 or 64 for ROWS=128
            copy_r2g(rFinalOut, Final[state_idx,head,row_cta*ROWS:(row_cta+1)*ROWS,:])
            # instruction_selection: st.global.v4.b32 for FP16/BF16 or st.global.v2.b64 for FP32; extent: one ROWSx128 final tile
        tmem_relinquish()
        # instruction_selection: tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned; extent: CTA allocation permit
        tmem_free()
        # instruction_selection: tcgen05.dealloc.cta_group::1.sync.aligned.b32; extent: 256 columns

    elif warp == 4:
        setmaxnreg("decrease", 32)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: current warp
        for _ in range(num_iters):
            m_h = wait(m_bar, consumer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: M stage wait
            ready = wait(ready_bar, consumer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: operand/accumulator wait
            done = acquire(done_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: done acquire
            for kphase in range(16):
                gemm(tAcc, tOpd, sM[m_h.index], shape=(ROWS,64,8), dtype="tf32",
                     accumulate=True, kphase=kphase)
                # instruction_selection: tcgen05.mma.cta_group::1.kind::tf32; extent: 16 K phases for ROWSx128x128 recurrence
            commit(done)
            # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64; extent: recurrence completion
            release(ready)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: operand reuse
            release(m_h)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: M stage reuse

    elif warp == 5:
        setmaxnreg("decrease", 32)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: current warp
        descriptor_prefetch(M.tensor_map, N.tensor_map)
        # instruction_selection: prefetch.tensormap; extent: two descriptors
        for chunk in range(num_chunks):
            n_h = acquire(n_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: N stage acquire
            expect_tx(n_h, ROWS*128*4)
            # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one N-stage byte expectation before TMA issue
            copy_g2s(N[chunk_start+chunk,head,row_cta*ROWS:(row_cta+1)*ROWS,:],
                     sN[n_h.index], completion=n_h.barrier)
            # instruction_selection: cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint; extent: four issues covering one ROWSx128 N tile
            if chunk >= start:
                m_h = acquire(m_bar, producer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: M stage acquire
                expect_tx(m_h, 128*128*4)
                # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one M-stage byte expectation before TMA issue
                copy_g2s(M[chunk_start+chunk,head,:,:], sM[m_h.index],
                         completion=m_h.barrier)
                # instruction_selection: cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint; extent: four issues covering one 128x128 M tile

    else:
        setmaxnreg("decrease", 32)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: current warp
```


# Launch 4: tcgen05 context-parallel prefill

This entry point keeps the source's two independent consumer groups and two
MMA issuers. Every physical 64-token slot in the padded pair loop is executed;
tail safety comes from bounded TensorMaps, a clamped T index, and neutral gate
values rather than from dropping the padded slot.

```python
@kernel(
    grid=(H_RATIO * H_BASE * max_cp_chunks_per_seq, num_sequences, 1),
    block=(384,1,1), num_warps=12, cluster=(1,1,1),
    tmem_columns=512, min_blocks_per_sm=1, target="sm_100a",
)
def _prefill_sm100(Q, K, V, Alpha, T_in, O, Cu, Fixed, InitWorkspace,
                   TensorMaps, cp_chunk_len, scale,
                   H_RATIO, H_BASE, H_STATE):
    BT = 64
    D = 128
    tid = thread_id_x()
    warp = warp_uniform(tid // 32)
    lane = tid % 32
    flat_work = block_id_x()
    seq = block_id_y()
    sab_head = flat_work % H_STATE
    chunk_in_seq = flat_work // H_STATE
    seq_start = copy_g2r(Cu[seq], reg("seq_start"))
    # instruction_selection: ld.global.b32; extent: scalar; INT64 specialization changes address stride/offset only
    seq_end = copy_g2r(Cu[seq+1], reg("seq_end"))
    # instruction_selection: ld.global.b32; extent: scalar; INT64 specialization changes address stride/offset only
    seq_len = seq_end - seq_start
    num_cp_chunks = ceil_div(seq_len, cp_chunk_len)
    chunk_len = 0
    if chunk_in_seq < num_cp_chunks:
        chunk_len = min(cp_chunk_len, seq_len - chunk_in_seq * cp_chunk_len)
    chunk_start = seq_start + chunk_in_seq * cp_chunk_len
    chunk_end = chunk_start + chunk_len
    cp_chunk = varlen_chunk_idx(seq, seq_start, chunk_in_seq, cp_chunk_len)
    t_blocks_per_cp_chunk = ceil_div(cp_chunk_len, BT)
    t_block_start = varlen_chunk_idx(
        seq, seq_start, chunk_in_seq * t_blocks_per_cp_chunk, BT)
    q_head = sab_head * Q_HEADS // H_STATE
    k_head = sab_head * K_HEADS // H_STATE
    v_head = sab_head * V_HEADS // H_STATE
    num_pairs = 2
    num_pairs_b = ceil_div(chunk_len, BT * num_pairs)
    num_chunks_b = num_pairs_b * num_pairs
    num_valid_chunks_b = ceil_div(chunk_len, BT)

    # The 128-byte GMEM descriptor slots are ABI-significant. Slot 3 remains
    # reserved for T, although T uses the fixed launch descriptor in this body.
    map_q = TensorMaps[linear_cta(),0,0:128]
    map_k = TensorMaps[linear_cta(),1,0:128]
    map_v = TensorMaps[linear_cta(),2,0:128]
    map_t_reserved = TensorMaps[linear_cta(),3,0:128]
    map_o = TensorMaps[linear_cta(),4,0:128]

    # SharedStorage order is retained: all mbarrier arrays, allocator word,
    # Q/K/V/T/Ainv/QK/O, then cumsumlog/cumprod; tile buffers are 1024B aligned.
    load_k_bar = tile("smem", "mbarrier", [3], storage_words=6, initial_phase=0)
    load_q_bar = tile("smem", "mbarrier", [2], storage_words=4, initial_phase=0)
    load_v_bar = tile("smem", "mbarrier", [3], storage_words=6, initial_phase=0)
    load_gate_bar = tile("smem", "mbarrier", [5], storage_words=10, initial_phase=0)
    load_t_bar = tile("smem", "mbarrier", [2], storage_words=4, initial_phase=0)
    q_state_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    kv_acc_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    cg0_acc_bar = tile("smem", "mbarrier", [2], storage_words=4, initial_phase=0)
    cg1_acc_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    ainv_bar = tile("smem", "mbarrier", [3], storage_words=6, initial_phase=0)
    qk_bar = tile("smem", "mbarrier", [2], storage_words=4, initial_phase=0)
    state_in_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    vks_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    nv_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    decay_v_bar = tile("smem", "mbarrier", [1], storage_words=2, initial_phase=0)
    o_bar = tile("smem", "mbarrier", [2], storage_words=4, initial_phase=0)
    tmem_holding = tile("smem", "i32", [1])

    sQ = tile("smem", IO_DTYPE, [2,64,128], alignment=1024,
              layout="tcgen05 Q A operand")
    sK = tile("smem", IO_DTYPE, [3,64,128], alignment=1024,
              layout="QK-B and KV-B-transposed aliases")
    sV = tile("smem", IO_DTYPE, [3,128,64], alignment=1024,
              layout="NV A operand")
    sT = tile("smem", IO_DTYPE, [2,64,64], alignment=1024,
              layout="precomputed T B operand")
    sAinv = tile("smem", IO_DTYPE, [3,64,64], alignment=1024,
                 layout="signed gate-scaled T B operand")
    sQk = tile("smem", IO_DTYPE, [2,64,64], alignment=1024,
               layout="gate-scaled QK B operand")
    sO = tile("smem", IO_DTYPE, [2,128,64], alignment=1024,
              layout="TMA epilogue tile")
    sCumsumLog = tile("smem", "f32", [5,64], alignment=16)
    sCumprod = tile("smem", "f32", [5,64], alignment=16)

    # TMEM offsets are columns, not byte offsets. The final two logical
    # shared-input slots alias columns 448..511 in source-controlled lifetimes.
    tState = tile("tmem", "f32", [128,128], column_offset=0, columns=128)
    tQState = tile("tmem", "f32", [128,64], column_offset=128, columns=64)
    tStateIn = tile("tmem", IO_DTYPE, [128,64], column_offset=192, columns=64)
    tCg0Acc = tile("tmem", "f32", [2,64,64], column_offset=256, columns=128)
    tCg1Acc = tile("tmem", "f32", [1,128,64], column_offset=384, columns=64)
    tSharedIn = tile("tmem", IO_DTYPE, [2,128,64], column_offset=448, columns=64,
                     alias_mode="two source logical stages")

    if warp == 8:
        descriptor_prefetch(Q.tensor_map, K.tensor_map, V.tensor_map,
                            T_in.tensor_map, O.tensor_map)
        # instruction_selection: prefetch.tensormap; extent: five launch descriptors before pipeline initialization and TMEM allocation

    pipe_init(load_k_bar, stages=3, kind="TmaUmma", producers=1,
              consumers=2, expected_bytes=64*128*bytes(IO_DTYPE))
    # instruction_selection: mbarrier.init.shared.b64 with TMA/UTMMA phase initialization; extent: three-stage K ring, issuers 8 and 10
    pipe_init(load_q_bar, stages=2, kind="TmaUmma", producers=1,
              consumers=2, expected_bytes=64*128*bytes(IO_DTYPE))
    # instruction_selection: mbarrier.init.shared.b64 with TMA/UTMMA phase initialization; extent: two-stage Q ring, issuers 8 and 10
    pipe_init(load_v_bar, stages=3, kind="TmaAsync", producers=1,
              consumers=4, expected_bytes=128*64*bytes(IO_DTYPE))
    # instruction_selection: mbarrier.init.shared.b64; extent: three-stage V ring, one elected consumer per CG1 warp
    pipe_init(load_gate_bar, stages=5, kind="Async", producers=32, consumers=256)
    # instruction_selection: mbarrier.init.shared.b64; extent: five-stage gate ring, CG0+CG1 consumers
    pipe_init(load_t_bar, stages=2, kind="TmaAsync", producers=1,
              consumers=4, expected_bytes=64*64*bytes(IO_DTYPE))
    # instruction_selection: mbarrier.init.shared.b64; extent: two-stage T ring, one elected consumer per CG0 warp
    pipe_init(kv_acc_bar, stages=1, kind="UmmaAsync", producers=1, consumers=128)
    # instruction_selection: mbarrier.init.shared.b64 phase pair; extent: one-stage recurrent-state publication, warp 10 to CG1
    pipe_init(q_state_bar, stages=1, kind="UmmaAsync", producers=1, consumers=128)
    # instruction_selection: mbarrier.init.shared.b64 phase pair; extent: one-stage Q-state publication, warp 10 to CG1
    pipe_init(cg0_acc_bar, stages=2, kind="UmmaAsync", producers=1, consumers=128)
    # instruction_selection: mbarrier.init.shared.b64 phase pairs; extent: two-stage QK accumulator ring, warp 8 to CG0
    pipe_init(cg1_acc_bar, stages=1, kind="UmmaAsync", producers=1, consumers=128)
    # instruction_selection: mbarrier.init.shared.b64 phase pair; extent: one-stage KS/NV accumulator, warp 10 to CG1
    pipe_init(ainv_bar, stages=3, kind="AsyncUmma", producers=128, consumers=1)
    # instruction_selection: mbarrier.init.shared.b64 phase pairs; extent: three-stage Ainv publication, CG0 to warp 10
    pipe_init(qk_bar, stages=2, kind="AsyncUmma", producers=128, consumers=1)
    # instruction_selection: mbarrier.init.shared.b64 phase pairs; extent: two-stage QK publication, CG0 to warp 10
    pipe_init(state_in_bar, stages=1, kind="AsyncUmma", producers=128, consumers=1)
    # instruction_selection: mbarrier.init.shared.b64 phase pair; extent: state-input publication, CG1 to warp 10
    pipe_init(vks_bar, stages=1, kind="AsyncUmma", producers=128, consumers=1)
    # instruction_selection: mbarrier.init.shared.b64 phase pair; extent: V-minus-KS publication, CG1 to warp 10
    pipe_init(nv_bar, stages=1, kind="AsyncUmma", producers=128, consumers=1)
    # instruction_selection: mbarrier.init.shared.b64 phase pair; extent: NV publication, CG1 to warp 10
    pipe_init(decay_v_bar, stages=1, kind="AsyncUmma", producers=128, consumers=1)
    # instruction_selection: mbarrier.init.shared.b64 phase pair; extent: decay-V publication, CG1 to warp 10
    pipe_init(o_bar, stages=2, kind="Async", producers=128, consumers=32)
    # instruction_selection: mbarrier.init.shared.b64; extent: two-stage output ring, CG1 to warp 11
    pipeline_init_arrive(cluster=(1,1), relaxed=True)
    # instruction_selection: fence.mbarrier_init.release.cluster; extent: one deferred prefill pipeline-initialization fence for the fixed (1,1) cluster
    pipeline_init_wait(cluster=(1,1))
    # instruction_selection: bar.sync 0; extent: one CTA sync completing prefill pipeline initialization

    if 4 <= warp <= 7:
        tmem_allocate(tmem_holding, columns=512, allocator_warp=4,
                      waiters=warps(0,1,2,3,4,5,6,7,8,10))
        # instruction_selection: tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32; extent: one 512-column allocation by CG1 allocator warp

    # CG0 is an independent source branch. CG1 begins a separate if/elif chain.
    if 0 <= warp <= 3:
        setmaxnreg("increase", 224)
        # instruction_selection: setmaxnreg.inc.sync.aligned.u32; extent: current CG0 warp
        tmem_wait_for_alloc()
        # instruction_selection: bar.sync 1,320; extent: CG0 participation in the ten-warp prefill allocator barrier
        for pair in range(num_pairs_b):
            for local in range(2):
                chunk = pair * 2 + local
                valid_tokens = chunk_len - chunk * BT
                is_final_block = chunk >= num_valid_chunks_b - 1
                gate_h = wait(load_gate_bar, consumer="CG0")
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one gate stage
                t_h = wait(load_t_bar, consumer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one T stage
                ainv_h = acquire(ainv_bar, producer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one Ainv stage

                rT = reg_tile(IO_DTYPE, [64,64])
                copy_s2r(sT[t_h.index], rT)
                # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.shared.b16; extent: one 64x64 T tile
                rAinv = reg_tile(IO_DTYPE, [64,64])
                for element in rAinv:
                    t, s = coordinate(element)
                    log_s = copy_s2r(sCumsumLog[gate_h.index,s], reg("log_s"))
                    # instruction_selection: ld.shared.b32; extent: one scalar per owned Ainv element
                    log_t = copy_s2r(sCumsumLog[gate_h.index,t], reg("log_t"))
                    # instruction_selection: ld.shared.b32; extent: one scalar per owned Ainv element
                    gamma = 0.0
                    value = 0.0
                    pred = s >= t
                    if is_final_block:
                        pred = pred and s < valid_tokens and t < valid_tokens
                    if pred:
                        gamma = exp2(sub(log_s, log_t))
                        # instruction_selection: sub.rn.f32 + ex2.approx.ftz.f32; extent: one causal valid Ainv element
                        value = mul(neg(gamma), cast("f32", rT[element]))
                        # instruction_selection: cvt.f32.{f16,bf16} + neg.f32 + mul.rn.f32; extent: one causal valid element
                    rAinv[element] = cast(IO_DTYPE, value)
                    # instruction_selection: cvt.rn.{f16,bf16}.f32; extent: one owned Ainv element
                copy_r2s(rAinv, transpose(sAinv[ainv_h.index]))
                # instruction_selection: stmatrix.sync.aligned.m8n8.x4.shared.b16; extent: one transposed 64x64 Ainv tile
                fence("async_shared_view")
                # instruction_selection: fence.proxy.async.shared::cta; extent: Ainv stage
                barrier(id=2, threads=128)
                # instruction_selection: bar.sync 2,128; extent: CG0 T-store named barrier
                release(t_h, elected_threads=4)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: one elected thread per CG0 warp
                commit(ainv_h)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: Ainv stage publication

                qk_out = acquire(qk_bar, producer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one QK output stage
                qk_acc = wait(cg0_acc_bar, consumer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one QK TMEM accumulator stage
                rQK = reg_tile("f32", [64,64])
                copy_t2r(tCg0Acc[qk_acc.index], rQK)
                # instruction_selection: tcgen05.ld.sync.aligned.16x128b.x4.b32; extent: one 64x64 FP32 accumulator tile
                rQKio = reg_tile(IO_DTYPE, [64,64])
                for element in rQKio:
                    s, t = coordinate(element)
                    log_s = copy_s2r(sCumsumLog[gate_h.index,s], reg("q_log_s"))
                    # instruction_selection: ld.shared.b32; extent: one scalar per owned QK element
                    log_t = copy_s2r(sCumsumLog[gate_h.index,t], reg("q_log_t"))
                    # instruction_selection: ld.shared.b32; extent: one scalar per owned QK element
                    gamma = 0.0
                    pred = s >= t
                    if is_final_block:
                        pred = pred and s < valid_tokens and t < valid_tokens
                    if pred:
                        gamma = exp2(sub(log_s, log_t))
                        # instruction_selection: sub.rn.f32 + ex2.approx.ftz.f32; extent: one causal valid QK element
                    rQKio[element] = cast(IO_DTYPE, mul(mul(rQK[element], gamma), scale))
                    # instruction_selection: mul.rn.f32 + cvt.rn.{f16,bf16}.f32; extent: one owned QK element
                copy_r2s(rQKio, sQk[qk_out.index])
                # instruction_selection: stmatrix.sync.aligned.m8n8.x4.shared.b16; extent: one 64x64 QK tile
                fence("async_shared_view")
                # instruction_selection: fence.proxy.async.shared::cta; extent: QK output stage
                release(qk_acc)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: QK accumulator reuse
                commit(qk_out)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: QK output publication
                release(gate_h)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: gate stage reuse by CG0
        tail(ainv_bar, producer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 polling loop plus phase/index advance; extent: three stage drains for the Ainv ring
        tail(qk_bar, producer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 polling loop plus phase/index advance; extent: two stage drains for the QK ring
```

```python
    if 4 <= warp <= 7:
        setmaxnreg("increase", 256)
        # instruction_selection: setmaxnreg.inc.sync.aligned.u32; extent: current CG1 warp
        tmem_wait_for_alloc()
        # instruction_selection: bar.sync 1,320; extent: CG1 participation in the ten-warp prefill allocator barrier
        if chunk_len > 0:
            initial_state_h = acquire(kv_acc_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: initial recurrent-state publication
            rState0 = reg_tile("f32", [128,128])
            if chunk_in_seq > 0:
                copy_g2r(Fixed[cp_chunk-1,sab_head,:,:], rState0)
                # instruction_selection: ld.global.L1::no_allocate.b32 loop; extent: previous fixed 128x128 FP32 state
            elif NEEDS_INITIAL_STATE:
                copy_g2r(InitWorkspace[seq,sab_head,:,:], rState0)
                # instruction_selection: ld.global.L1::no_allocate.v4.b32 loop; extent: first-CP FP32 initial-state workspace
            else:
                fill(rState0, 0.0)
                # instruction_selection: mov.b32/mov.f32 zero initialization; extent: one 128x128 recurrent-state tile
            copy_r2t(rState0, tState)
            # instruction_selection: tcgen05.st.sync.aligned.32x32b.x32.b32; extent: one 128x128 FP32 state tile
            fence("tmem_store")
            # instruction_selection: tcgen05.wait::st.sync.aligned; extent: initial state TMEM stores
            barrier(id=4, threads=128)
            # instruction_selection: bar.sync 4,128; extent: CG1 initial-state store barrier
            if tid % 128 == 0:
                manual_arrive(initial_state_h)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: one elected CG1 thread publishes initial state

            for chunk in range(num_chunks_b):
                is_pair_first = (chunk & 1) == 0
                if is_pair_first:
                    advance(kv_acc_bar, producer=True)
                    # instruction_selection: no standalone PTX instruction after optimization; extent: first source producer-cursor advance at each pair-first CG1 chunk
                    advance(kv_acc_bar, producer=True)
                    # instruction_selection: no standalone PTX instruction after optimization; extent: second consecutive source producer-cursor advance, restoring the one-stage physical phase
                gate_h = wait(load_gate_bar, consumer="CG1")
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one gate stage
                cumprod_total = copy_s2r(
                    sCumprod[gate_h.index,63], reg("cumprod_total"))
                # instruction_selection: ld.shared.b32; extent: final physical gate scalar

                # use_initial_state is forced true by the source class, so this
                # path is present for every physical chunk, including padding.
                kv_prev = wait(kv_acc_bar, consumer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: previous state
                state_in_h = current(state_in_bar, producer=True)
                # instruction_selection: mbarrier producer phase/address selection; extent: fixed one-stage state-input slot
                advance(state_in_bar, producer=True)
                # instruction_selection: xor/add pipeline phase/cursor advance; extent: one state-input stage without an empty wait
                rState = reg_tile("f32", [128,128])
                copy_t2r(tState[kv_prev.index], rState)
                # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x32.b32; extent: one 128x128 FP32 state tile
                rStateIo = cast(IO_DTYPE, rState)
                # instruction_selection: cvt.rn.{f16,bf16}x2.f32; extent: 64 packed-pair issues per CG1 thread across one 128x128 state-input tile
                copy_r2t(rStateIo, tStateIn[state_in_h.index])
                # instruction_selection: tcgen05.st.sync.aligned.32x32b.x16.b32; extent: one 128x64-per-phase IO state operand across K phases
                fence("tmem_store")
                # instruction_selection: tcgen05.wait::st.sync.aligned; extent: state-input TMEM stores
                commit(state_in_h)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: state-input publication
                mul(rState, rState, cumprod_total)
                # instruction_selection: mul.rn.f32; extent: one 128x128 recurrent-state tile
                copy_r2t(rState, tState[kv_prev.index])
                # instruction_selection: tcgen05.st.sync.aligned.32x32b.x32.b32; extent: one 128x128 decayed state tile
                fence("tmem_store")
                # instruction_selection: tcgen05.wait::st.sync.aligned; extent: decayed-state TMEM stores
                release(kv_prev)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: previous-state stage reuse

                rCumprod = reg_tile("f32", [128,64], mapping="CG1-owned result coordinates")
                for element_pair in pairs(rCumprod):
                    _, col0 = coordinate(element_pair.first)
                    _, col1 = coordinate(element_pair.second)
                    copy_s2r(sCumprod[gate_h.index,(col0,col1)],
                             rCumprod[element_pair])
                    # instruction_selection: ld.shared.b64; extent: one 2xf32 vector per coordinate pair, 32 issues per CG1 thread across the 128x64 fragment
                last_log = copy_s2r(sCumsumLog[gate_h.index,63], reg("last_log"))
                # instruction_selection: ld.shared.b32; extent: final physical gate-log scalar after all cumprod-vector loads
                rDecay = reg_tile("f32", [128,64], mapping="CG1-owned result coordinates")
                for element_pair in pairs(rDecay):
                    _, col0 = coordinate(element_pair.first)
                    _, col1 = coordinate(element_pair.second)
                    rLogs = reg_tile("f32", [2])
                    copy_s2r(sCumsumLog[gate_h.index,(col0,col1)], rLogs)
                    # instruction_selection: ld.shared.v2.b32; extent: one 2xf32 vector per coordinate pair, 32 issues per CG1 thread across the 128x64 fragment
                    d0, d1 = add_packed_f32x2(
                        (last_log,last_log), (neg(rLogs[0]),neg(rLogs[1])), ftz=False, rn=True)
                    # instruction_selection: add.rn.f32x2 with FTZ disabled; extent: one packed coordinate pair
                    rDecay[element_pair.first] = exp2(d0)
                    # instruction_selection: ex2.approx.ftz.f32; extent: first scalar in one packed coordinate pair
                    rDecay[element_pair.second] = exp2(d1)
                    # instruction_selection: ex2.approx.ftz.f32; extent: second scalar in one packed coordinate pair
                release(gate_h)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: gate stage reuse by CG1

                vks_h = current(vks_bar, producer=True)
                # instruction_selection: mbarrier producer phase/address selection; extent: fixed VKS notification slot
                advance(vks_bar, producer=True)
                # instruction_selection: xor/add pipeline phase/cursor advance; extent: fixed one-stage VKS pipe
                v_h = wait(load_v_bar, consumer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one V stage
                rV = reg_tile(IO_DTYPE, [128,64])
                copy_s2r(sV[v_h.index], rV)
                # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16; extent: one 128x64 V tile
                ks_h = wait(cg1_acc_bar, consumer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: KS accumulator
                rKS = reg_tile("f32", [128,64])
                copy_t2r(tCg1Acc[ks_h.index], rKS)
                # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x8.b32; extent: one 128x64 KS tile
                mul(rKS, rKS, rCumprod)
                # instruction_selection: mul.rn.f32; extent: one 128x64 coordinate-matched KS tile
                release(ks_h)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: KS accumulator reuse
                rKSIo = cast(IO_DTYPE, rKS)
                # instruction_selection: cvt.rn.{f16,bf16}x2.f32; extent: 32 packed-pair issues per CG1 thread across one 128x64 KS tile
                rVks = alias(rV, dtype=IO_DTYPE, shape=[128,64],
                             lifetime="loaded V is updated in place to V-minus-KS")
                for element_pair in pairs(rVks):
                    rVks[element_pair] = sub(rV[element_pair], rKSIo[element_pair])
                    # instruction_selection: sub.{f16,bf16}x2; extent: 32 packed-pair issues per CG1 thread across one 128x64 V-minus-KS tile
                copy_r2t(rVks, tSharedIn[0])
                # instruction_selection: tcgen05.st.sync.aligned.16x128b.x8.b32; extent: fixed shared-input slot 0
                fence("tmem_store")
                # instruction_selection: tcgen05.wait::st.sync.aligned; extent: VKS TMEM stores
                commit(vks_h)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: VKS publication

                qs_h = wait(q_state_bar, consumer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: QS accumulator
                rQS = reg_tile("f32", [128,64])
                copy_t2r(tQState[qs_h.index], rQS)
                # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x8.b32; extent: one 128x64 QS tile
                for element in rQS:
                    rQS[element] = mul(mul(rQS[element], rCumprod[element]), scale)
                    # instruction_selection: mul.rn.f32; extent: one coordinate-matched QS element, two multiplies
                copy_r2t(rQS, tQState[qs_h.index])
                # instruction_selection: tcgen05.st.sync.aligned.16x256b.x8.b32; extent: one scaled 128x64 Q-state tile
                fence("tmem_store")
                # instruction_selection: tcgen05.wait::st.sync.aligned; extent: scaled Q-state TMEM stores
                release(qs_h)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: QS stage handed back to issuer

                nv_acc = wait(cg1_acc_bar, consumer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: NV accumulator
                release(v_h, elected_threads=4)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: one elected thread per CG1 warp
                rNv = reg_tile("f32", [128,64])
                copy_t2r(tCg1Acc[nv_acc.index], rNv)
                # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x8.b32; extent: two 32-value-per-thread sub-tile issues covering one 128x64 NV tile
                rNvIo = cast(IO_DTYPE, rNv)
                # instruction_selection: cvt.rn.{f16,bf16}x2.f32; extent: 32 packed-pair issues per CG1 thread across one 128x64 NV tile
                release(nv_acc)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: NV accumulator reuse
                rDecayV = alias(rNv, dtype="f32", shape=[128,64],
                                lifetime="NV IO copy complete; reused in place as decay-V")
                for element in rDecayV:
                    rDecayV[element] = mul(rDecayV[element], rDecay[element])
                    # instruction_selection: mul.rn.f32; extent: one coordinate-matched decay-V element
                nv_h = current(nv_bar, producer=True)
                # instruction_selection: mbarrier producer phase/address selection; extent: fixed NV notification slot
                advance(nv_bar, producer=True)
                # instruction_selection: xor/add pipeline phase/cursor advance; extent: fixed one-stage NV pipe
                decay_h = current(decay_v_bar, producer=True)
                # instruction_selection: mbarrier producer phase/address selection; extent: fixed decay-V notification slot
                advance(decay_v_bar, producer=True)
                # instruction_selection: xor/add pipeline phase/cursor advance; extent: fixed one-stage decay-V pipe
                rDecayVIo = alias(
                    rNvIo, dtype=IO_DTYPE,
                    lifetime="each NV IO sub-tile is overwritten only after its slot-0 store")
                for sub_tile in range(2):
                    copy_r2t(rNvIo.subtile(sub_tile), tSharedIn[0].subtile(sub_tile))
                    # instruction_selection: tcgen05.st.sync.aligned.16x128b.x8.b32; extent: one of two 32-value-per-thread NV sub-tile issues into fixed slot 0
                    rDecayVIo.subtile(sub_tile) = cast(
                        IO_DTYPE, rDecayV.subtile(sub_tile))
                    # instruction_selection: cvt.rn.{f16,bf16}x2.f32; extent: 16 packed-pair issues per CG1 thread in each of two decay-V sub-tiles
                    copy_r2t(rDecayVIo.subtile(sub_tile),
                             tSharedIn[1].subtile(sub_tile))
                    # instruction_selection: tcgen05.st.sync.aligned.16x128b.x8.b32; extent: one of two 32-value-per-thread decay-V sub-tile issues into logical slot 1
                fence("tmem_store")
                # instruction_selection: tcgen05.wait::st.sync.aligned; extent: NV and decay-V TMEM stores
                commit(nv_h)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: NV publication before decay-V
                commit(decay_h)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: decay-V publication after NV

                o_h = acquire(o_bar, producer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one output stage
                o_qs = wait(q_state_bar, consumer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: completed QKV output state
                rO = reg_tile("f32", [128,64])
                copy_t2r(tQState[o_qs.index], rO)
                # instruction_selection: tcgen05.ld.sync.aligned.16x256b.x8.b32; extent: one 128x64 output tile
                rOio = cast(IO_DTYPE, rO)
                # instruction_selection: cvt.rn.{f16,bf16}x2.f32; extent: 32 packed-pair issues per CG1 thread across one 128x64 output tile
                copy_r2s(rOio, sO[o_h.index])
                # instruction_selection: stmatrix.sync.aligned.m8n8.x4.trans.shared.b16; extent: one 128x64 output tile
                fence("async_shared_view")
                # instruction_selection: fence.proxy.async.shared::cta; extent: output stage
                release(o_qs)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: QKV output-stage reuse
                commit(o_h)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: output publication

            if chunk_in_seq == num_cp_chunks - 1:
                final_h = wait(kv_acc_bar, consumer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: final recurrent state
                rFinal = reg_tile("f32", [128,128])
                copy_t2r(tState[final_h.index], rFinal)
                # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x32.b32; extent: one 128x128 final state tile
                rFinalDrain = alias(rFinal, dtype=PREFILL_STATE_DTYPE,
                                    lifetime="state_dtype == acc_dtype == f32 drain")
                # Structural register alias only: this specialization emits no conversion or move and performs no global store.
                release(final_h)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: final-state stage reuse
            else:
                final_h = wait(kv_acc_bar, consumer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: final local state drain
                release(final_h)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: final local state reuse

        tmem_relinquish()
        # instruction_selection: tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned; extent: CTA allocation permit
        tmem_free()
        # instruction_selection: tcgen05.dealloc.cta_group::1.sync.aligned.b32; extent: 512 columns
        tail(o_bar, producer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 polling loop plus phase/index advance; extent: two stage drains for the output ring
        tail(state_in_bar, producer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 polling loop plus phase/index advance; extent: one stage drain for the state-input ring
```

```python
    elif warp == 8:
        # First issuer: two QK tiles per source pair.
        setmaxnreg("decrease", 24)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: warp 8
        tmem_wait_for_alloc()
        # instruction_selection: bar.sync 1,320; extent: warp 8 participation in the ten-warp prefill allocator barrier
        for pair in range(num_pairs_b):
            for local in range(2):
                k_h = wait(load_k_bar, consumer="warp8")
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one K stage
                q_h = wait(load_q_bar, consumer="warp8")
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one Q stage
                qk_h = acquire(cg0_acc_bar, producer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one QK accumulator stage
                gemm(tCg0Acc[qk_h.index], sQ[q_h.index], sK[k_h.index],
                     shape=(64,64,128), dtype=IO_DTYPE, accumulate=False)
                # instruction_selection: tcgen05.mma.cta_group::1.kind::f16; extent: 64x64x128, eight K phases
                commit(qk_h)
                # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64; extent: QK tile completion
                release(q_h)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: Q stage reuse by warp 8
                release(k_h)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: K stage reuse by warp 8
        tail(cg0_acc_bar, producer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 polling loop plus phase/index advance; extent: two stage drains for the CG0 accumulator ring

    elif warp == 10:
        # Second issuer: KS, QS, NV, QKV, then recurrent KV in that order.
        setmaxnreg("decrease", 24)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: warp 10
        tmem_wait_for_alloc()
        # instruction_selection: bar.sync 1,320; extent: warp 10 participation in the ten-warp prefill allocator barrier
        for chunk in range(num_chunks_b):
            k_h = wait(load_k_bar, consumer="warp10")
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one K stage
            q_h = wait(load_q_bar, consumer="warp10")
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one Q stage

            state_h = wait(state_in_bar, consumer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one state-input stage
            ks_h = acquire(cg1_acc_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: KS accumulator stage
            gemm(tCg1Acc[ks_h.index], tStateIn[state_h.index], sK[k_h.index],
                 shape=(128,64,128), dtype=IO_DTYPE, accumulate=False)
            # instruction_selection: tcgen05.mma.cta_group::1.kind::f16; extent: KS 128x64x128, eight K phases
            commit(ks_h)
            # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64; extent: KS completion
            qs_h = acquire(q_state_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: QS accumulator stage
            gemm(tQState[qs_h.index], tStateIn[state_h.index], sQ[q_h.index],
                 shape=(128,64,128), dtype=IO_DTYPE, accumulate=False)
            # instruction_selection: tcgen05.mma.cta_group::1.kind::f16; extent: QS 128x64x128, eight K phases
            commit(qs_h)
            # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64; extent: QS completion
            release(state_h)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: state-input reuse
            release(q_h)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: Q stage reuse by warp 10

            nv_acc = acquire(cg1_acc_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: NV accumulator stage
            wait_ready(vks_bar, consumer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: VKS ready-only notification
            ainv_h = wait(ainv_bar, consumer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one Ainv stage
            gemm(tCg1Acc[nv_acc.index], tSharedIn[0], sAinv[ainv_h.index],
                 shape=(128,64,64), dtype=IO_DTYPE, accumulate=False)
            # instruction_selection: tcgen05.mma.cta_group::1.kind::f16; extent: NV 128x64x64, four K phases
            commit(nv_acc)
            # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64; extent: NV completion
            release(ainv_h)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: Ainv stage reuse

            qkv_h = acquire(q_state_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: QKV accumulator stage
            qk_h = wait(qk_bar, consumer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one QK stage
            wait_ready(nv_bar, consumer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: NV ready-only notification
            gemm(tQState[qkv_h.index], tSharedIn[0], sQk[qk_h.index],
                 shape=(128,64,64), dtype=IO_DTYPE, accumulate=True)
            # instruction_selection: tcgen05.mma.cta_group::1.kind::f16; extent: QKV 128x64x64, four K phases, forced-state accumulate
            release(qk_h)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: QK stage reuse
            commit(qkv_h)
            # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64; extent: QKV completion

            if chunk == 0:
                advance(kv_acc_bar, producer=True)
                # instruction_selection: xor/add pipeline phase/cursor advance; extent: source forced-initial-state first-chunk cursor correction
            kv_h = acquire(kv_acc_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: recurrent-state accumulator stage
            wait_ready(decay_v_bar, consumer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: decay-V ready-only notification
            gemm(tState[kv_h.index], tSharedIn[1], transpose(sK[k_h.index]),
                 shape=(128,128,64), dtype=IO_DTYPE, accumulate=True)
            # instruction_selection: tcgen05.mma.cta_group::1.kind::f16; extent: KV 128x128x64, four K phases, forced-state accumulate
            commit(kv_h)
            # instruction_selection: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64; extent: recurrent-state completion
            release(k_h)
            # instruction_selection: mbarrier.arrive.shared.b64; extent: K stage reuse by warp 10
        tail(cg1_acc_bar, producer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 polling loop plus phase/index advance; extent: one stage drain for the CG1 accumulator ring
        tail(q_state_bar, producer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 polling loop plus phase/index advance; extent: one stage drain for the Q-state ring
        tail(kv_acc_bar, producer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 polling loop plus phase/index advance; extent: one stage drain for the recurrent-state ring

    elif warp == 9:
        # Q/K/V descriptors are per-CTA GMEM TensorMaps; T retains the fixed
        # descriptor generated at launch and therefore never consumes slot 3.
        setmaxnreg("decrease", 24)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: warp 9
        rQMap = reg_tile("u64", [8])
        copy_p2r(Q.tensor_map, rQMap)
        # instruction_selection: ld.param.v2.b64; extent: four vector loads covering the 64-byte Q descriptor template
        copy_r2g(rQMap, map_q[0:64])
        # instruction_selection: st.global.v4.b64; extent: two vector stores into the 128-byte Q slot
        barrier("warp")
        # instruction_selection: bar.warp.sync -1; extent: warp 9 after Q descriptor copy
        rKMap = reg_tile("u64", [8])
        copy_p2r(K.tensor_map, rKMap)
        # instruction_selection: ld.param.v2.b64; extent: four vector loads covering the 64-byte K descriptor template
        copy_r2g(rKMap, map_k[0:64])
        # instruction_selection: st.global.v4.b64; extent: two vector stores into the 128-byte K slot
        barrier("warp")
        # instruction_selection: bar.warp.sync -1; extent: warp 9 after K descriptor copy
        rVMap = reg_tile("u64", [8])
        copy_p2r(V.tensor_map, rVMap)
        # instruction_selection: ld.param.v2.b64; extent: four vector loads covering the 64-byte V descriptor template
        copy_r2g(rVMap, map_v[0:64])
        # instruction_selection: st.global.v4.b64; extent: two vector stores into the 128-byte V slot
        barrier("warp")
        # instruction_selection: bar.warp.sync -1; extent: warp 9 after V descriptor copy
        fence("tensormap_init")
        # instruction_selection: fence.acq_rel.cta; extent: Q/K/V descriptor initialization

        async_copy_commit_group()
        # instruction_selection: cp.async.bulk.commit_group; extent: elected warp-9 descriptor-update preamble
        async_copy_wait_group_read(0)
        # instruction_selection: cp.async.bulk.wait_group.read 0; extent: descriptor-update preamble
        barrier("warp")
        # instruction_selection: bar.warp.sync -1; extent: warp 9 before descriptor replacement
        tensormap_replace(map_q, address=Q, dims=(chunk_end,Q_HEADS,D), strides=Q.strides)
        # instruction_selection: tensormap.replace.tile.global_{address,dim,stride}.global.b1024.{b64,b32}; extent: Q address plus all encoded dimensions/strides
        tensormap_replace(map_k, address=K, dims=(chunk_end,K_HEADS,D), strides=K.strides)
        # instruction_selection: tensormap.replace.tile.global_{address,dim,stride}.global.b1024.{b64,b32}; extent: K address plus all encoded dimensions/strides
        tensormap_replace(map_v, address=V, dims=(D,chunk_end,V_HEADS), strides=V.strides)
        # instruction_selection: tensormap.replace.tile.global_{address,dim,stride}.global.b1024.{b64,b32}; extent: V address plus all encoded dimensions/strides
        barrier("warp")
        # instruction_selection: bar.warp.sync -1; extent: warp 9 after Q/K/V descriptor replacement
        fence("tensormap_release")
        # instruction_selection: fence.proxy.tensormap::generic.release.gpu; extent: Q/K/V descriptor update publication

        for chunk in range(num_chunks_b):
            token0 = chunk_start + chunk * BT
            k_h = acquire(load_k_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one K stage
            expect_tx(k_h, 64*128*bytes(IO_DTYPE))
            # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one K-stage byte expectation before TMA issue
            if chunk == 0:
                tensormap_fence_update(map_k)
                # instruction_selection: fence.proxy.tensormap::generic.acquire.gpu; extent: K descriptor first use
            copy_g2s(K[token0:token0+64,k_head,:], sK[k_h.index],
                     descriptor=map_k, completion=k_h.barrier)
            # instruction_selection: cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint; extent: two issues covering one bounded 64x128 K tile
            q_h = acquire(load_q_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one Q stage
            expect_tx(q_h, 64*128*bytes(IO_DTYPE))
            # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one Q-stage byte expectation before TMA issue
            if chunk == 0:
                tensormap_fence_update(map_q)
                # instruction_selection: fence.proxy.tensormap::generic.acquire.gpu; extent: Q descriptor first use
            copy_g2s(Q[token0:token0+64,q_head,:], sQ[q_h.index],
                     descriptor=map_q, completion=q_h.barrier)
            # instruction_selection: cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint for GQA or cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint for GVA; extent: two issues covering one bounded 64x128 Q tile
            v_h = acquire(load_v_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one V stage
            expect_tx(v_h, 128*64*bytes(IO_DTYPE))
            # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one V-stage byte expectation before TMA issue
            if chunk == 0:
                tensormap_fence_update(map_v)
                # instruction_selection: fence.proxy.tensormap::generic.acquire.gpu; extent: V descriptor first use
            copy_g2s(V[token0:token0+64,v_head,:], sV[v_h.index],
                     descriptor=map_v, completion=v_h.barrier)
            # instruction_selection: cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint for GQA or cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint for GVA; extent: two issues covering one bounded 128x64 V tile

            t_chunk = chunk
            if chunk >= num_valid_chunks_b:
                t_chunk = num_valid_chunks_b - 1
            t_h = acquire(load_t_bar, producer=True)
            # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one T stage
            expect_tx(t_h, 64*64*bytes(IO_DTYPE))
            # instruction_selection: mbarrier.arrive.expect_tx.shared.b64; extent: one T-stage byte expectation before TMA issue
            copy_g2s(T_in[t_block_start+t_chunk,sab_head,:,:], sT[t_h.index],
                     descriptor=T_in.tensor_map, completion=t_h.barrier)
            # instruction_selection: cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint; extent: one issue covering one fixed-descriptor 64x64 T tile with padded-index clamp
        tail(load_q_bar, producer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 polling loop plus phase/index advance; extent: two stage drains for the Q ring
        tail(load_k_bar, producer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 polling loop plus phase/index advance; extent: three stage drains for the K ring
        tail(load_v_bar, producer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 polling loop plus phase/index advance; extent: three stage drains for the V ring
        tail(load_t_bar, producer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 polling loop plus phase/index advance; extent: two stage drains for the T ring
```

```python
    # Warp 11 is another independent source branch, not an elif child of CG0.
    if warp == 11:
        setmaxnreg("decrease", 24)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32; extent: warp 11
        rOMap = reg_tile("u64", [8])
        copy_p2r(O.tensor_map, rOMap)
        # instruction_selection: ld.param.v2.b64; extent: four vector loads covering the 64-byte O descriptor template
        copy_r2g(rOMap, map_o[0:64])
        # instruction_selection: st.global.v4.b64; extent: two vector stores into the 128-byte O slot
        barrier("warp")
        # instruction_selection: bar.warp.sync -1; extent: warp 11 after O descriptor copy
        fence("tensormap_init")
        # instruction_selection: fence.acq_rel.cta; extent: O descriptor initialization
        async_copy_commit_group()
        # instruction_selection: cp.async.bulk.commit_group; extent: elected warp-11 descriptor-update preamble
        async_copy_wait_group_read(0)
        # instruction_selection: cp.async.bulk.wait_group.read 0; extent: O descriptor-update preamble
        barrier("warp")
        # instruction_selection: bar.warp.sync -1; extent: warp 11 before O descriptor replacement
        tensormap_replace(map_o, address=O, dims=(D,chunk_end,H_STATE), strides=O.strides)
        # instruction_selection: tensormap.replace.tile.global_{address,dim,stride}.global.b1024.{b64,b32}; extent: O address plus all encoded dimensions/strides
        barrier("warp")
        # instruction_selection: bar.warp.sync -1; extent: warp 11 after O descriptor replacement
        fence("tensormap_release")
        # instruction_selection: fence.proxy.tensormap::generic.release.gpu; extent: O descriptor update publication
        tensormap_fence_update(map_o)
        # instruction_selection: fence.proxy.tensormap::generic.acquire.gpu; extent: O descriptor first use

        if chunk_len > 0:
            # Phase one always publishes the first two gate tiles.
            for prefetch in range(0,2):
                gate_out = acquire(load_gate_bar, producer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one gate stage
                gate_token0 = chunk_start + prefetch * BT
                is_last = prefetch >= num_valid_chunks_b - 1
                rLog = reg_tile("f32", [2], mapping=(lane,lane+32))
                for slot,row in enumerate((lane,lane+32)):
                    gate = 1.0
                    if not is_last or gate_token0 + row < chunk_end:
                        gate = copy_g2r(Alpha[gate_token0+row,sab_head], reg("gate",slot))
                        # instruction_selection: ld.global.b32; extent: one predicated scalar per lane-owned row
                    rLog[slot] = log2(add(gate, 1.0e-10))
                    # instruction_selection: add.rn.f32 + lg2.approx.ftz.f32; extent: one scalar per lane-owned row
                for slot in range(2):
                    for delta in (1,2,4,8,16):
                        peer = shuffle_up(rLog[slot], delta, mask=0xffffffff, clamp=0)
                        # instruction_selection: shfl.sync.up.b32; extent: one scalar in five-step inclusive scan
                        if lane >= delta:
                            rLog[slot] = add(rLog[slot], peer)
                            # instruction_selection: add.rn.f32; extent: one scalar in five-step inclusive scan
                    if slot == 1:
                        carry = shuffle_index(rLog[0], 31, mask=0xffffffff, clamp=31)
                        # instruction_selection: shfl.sync.idx.b32; extent: first-half terminal carry
                        rLog[1] = add(rLog[1], carry)
                        # instruction_selection: add.rn.f32; extent: second-half carry
                    copy_r2s(rLog[slot], sCumsumLog[gate_out.index,slot*32+lane])
                    # instruction_selection: st.shared.b32; extent: one cumsum-log scalar
                    cp = exp2(rLog[slot])
                    # instruction_selection: ex2.approx.ftz.f32; extent: one cumprod scalar
                    copy_r2s(cp, sCumprod[gate_out.index,slot*32+lane])
                    # instruction_selection: st.shared.b32; extent: one cumprod scalar
                commit(gate_out)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: one gate stage publication

            # Phase two exists only when the padded stream has more than one pair.
            if num_chunks_b > 2:
                for prefetch in range(2,4):
                    gate_out = acquire(load_gate_bar, producer=True)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one gate stage
                    gate_token0 = chunk_start + prefetch * BT
                    is_last = prefetch >= num_valid_chunks_b - 1
                    rLog = reg_tile("f32", [2], mapping=(lane,lane+32))
                    for slot,row in enumerate((lane,lane+32)):
                        gate = 1.0
                        if not is_last or gate_token0 + row < chunk_end:
                            gate = copy_g2r(Alpha[gate_token0+row,sab_head], reg("gate",slot))
                            # instruction_selection: ld.global.b32; extent: one predicated scalar per lane-owned row
                        rLog[slot] = log2(add(gate, 1.0e-10))
                        # instruction_selection: add.rn.f32 + lg2.approx.ftz.f32; extent: one scalar per lane-owned row
                    for slot in range(2):
                        for delta in (1,2,4,8,16):
                            peer = shuffle_up(rLog[slot], delta, mask=0xffffffff, clamp=0)
                            # instruction_selection: shfl.sync.up.b32; extent: one scalar in five-step inclusive scan
                            if lane >= delta:
                                rLog[slot] = add(rLog[slot], peer)
                                # instruction_selection: add.rn.f32; extent: one scalar in five-step inclusive scan
                        if slot == 1:
                            carry = shuffle_index(rLog[0], 31, mask=0xffffffff, clamp=31)
                            # instruction_selection: shfl.sync.idx.b32; extent: first-half terminal carry
                            rLog[1] = add(rLog[1], carry)
                            # instruction_selection: add.rn.f32; extent: second-half carry
                        copy_r2s(rLog[slot], sCumsumLog[gate_out.index,slot*32+lane])
                        # instruction_selection: st.shared.b32; extent: one cumsum-log scalar
                        cp = exp2(rLog[slot])
                        # instruction_selection: ex2.approx.ftz.f32; extent: one cumprod scalar
                        copy_r2s(cp, sCumprod[gate_out.index,slot*32+lane])
                        # instruction_selection: st.shared.b32; extent: one cumprod scalar
                    commit(gate_out)
                    # instruction_selection: mbarrier.arrive.shared.b64; extent: one gate stage publication

            for chunk in range(num_chunks_b):
                prefetch = chunk + 4
                if prefetch < num_chunks_b:
                    gate_out = acquire(load_gate_bar, producer=True)
                    # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one four-tile-lookahead gate stage
                    gate_token0 = chunk_start + prefetch * BT
                    is_last = prefetch >= num_valid_chunks_b - 1
                    rLog = reg_tile("f32", [2], mapping=(lane,lane+32))
                    for slot,row in enumerate((lane,lane+32)):
                        gate = 1.0
                        if not is_last or gate_token0 + row < chunk_end:
                            gate = copy_g2r(Alpha[gate_token0+row,sab_head], reg("gate",slot))
                            # instruction_selection: ld.global.b32; extent: one predicated scalar per lane-owned row
                        rLog[slot] = log2(add(gate, 1.0e-10))
                        # instruction_selection: add.rn.f32 + lg2.approx.ftz.f32; extent: one scalar per lane-owned row
                    for slot in range(2):
                        for delta in (1,2,4,8,16):
                            peer = shuffle_up(rLog[slot], delta, mask=0xffffffff, clamp=0)
                            # instruction_selection: shfl.sync.up.b32; extent: one scalar in five-step inclusive scan
                            if lane >= delta:
                                rLog[slot] = add(rLog[slot], peer)
                                # instruction_selection: add.rn.f32; extent: one scalar in five-step inclusive scan
                        if slot == 1:
                            carry = shuffle_index(rLog[0], 31, mask=0xffffffff, clamp=31)
                            # instruction_selection: shfl.sync.idx.b32; extent: first-half terminal carry
                            rLog[1] = add(rLog[1], carry)
                            # instruction_selection: add.rn.f32; extent: second-half carry
                        copy_r2s(rLog[slot], sCumsumLog[gate_out.index,slot*32+lane])
                        # instruction_selection: st.shared.b32; extent: one cumsum-log scalar
                        cp = exp2(rLog[slot])
                        # instruction_selection: ex2.approx.ftz.f32; extent: one cumprod scalar
                        copy_r2s(cp, sCumprod[gate_out.index,slot*32+lane])
                        # instruction_selection: st.shared.b32; extent: one cumprod scalar
                    commit(gate_out)
                    # instruction_selection: mbarrier.arrive.shared.b64; extent: one gate stage publication

                o_h = wait(o_bar, consumer=True)
                # instruction_selection: mbarrier.try_wait.parity.shared.b64 loop; extent: one output stage
                copy_s2g(sO[o_h.index],
                         O[chunk_start+chunk*BT:chunk_start+(chunk+1)*BT,sab_head,:],
                         descriptor=map_o)
                # instruction_selection: cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint; extent: two issues covering one bounded 64x128 output tile
                async_copy_commit_group()
                # instruction_selection: cp.async.bulk.commit_group; extent: one output TMA group
                async_copy_wait_group(0)
                # instruction_selection: cp.async.bulk.wait_group 0; extent: all prior output stores
                release(o_h)
                # instruction_selection: mbarrier.arrive.shared.b64; extent: output stage reuse
        tail(load_gate_bar, producer=True)
        # instruction_selection: mbarrier.try_wait.parity.shared.b64 polling loop plus phase/index advance; extent: five stage drains for the gate ring
```

## Logical GEMM ownership and instruction shape

| Launch | Logical result | Owner | A source | B source | Shape `(M,N,K)` | Accumulate |
| --- | --- | --- | --- | --- | --- | --- |
| T | `K @ K.T` | all four HMMA warps | `sK` | transposed `sK` | `64,64,128` | no |
| T inverse L2 | `-D8 @ C8`, then result `@ A8` | one warp per 16-row block | registers/shared | registers/shared | `16,8,8` twice | no |
| T inverse L3 | `-D16 @ C16`, then result `@ A16` | warps 0..1 | registers/shared | registers/shared | `16,16,16` twice | no |
| T inverse L4 | `-D32 @ C32`, then result `@ A32` | four warps | registers/shared | registers/shared | `16,16,32`; `16,32,16` | no |
| MN | `Z=M @ K.T` | warp 8 | TMEM M-input | shared K transpose | `128,64,128` | no |
| MN | first `M += K @ X`; later `M += M-input @ X` | warp 8 | shared K or TMEM M-input | shared X | `128,128,64` | yes |
| MN | `X=K @ T` | warp 11 | shared K | shared T | `128,64,64` | no |
| MN | `Y=N @ K.T` | warp 11 | TMEM N-input | shared K transpose | `128,64,128` | no |
| MN | `N += N-input @ X` | warp 11 | TMEM N-input | shared X | `128,128,64` | yes |
| UTC fixup | `state = N + state @ M` | warp 4 | TMEM TF32 state operand | shared TF32 M | `ROWS,128,128` | yes |
| Prefill | `QK=Q @ K.T` | warp 8 | shared Q | shared K | `64,64,128` | no |
| Prefill | `KS=state @ K` | warp 10 | TMEM IO state-input | shared K | `128,64,128` | no |
| Prefill | `QS=state @ Q` | warp 10 | TMEM IO state-input | shared Q | `128,64,128` | no |
| Prefill | `NV=(V-KS) @ Ainv` | warp 10 | TMEM IO VKS | shared Ainv | `128,64,64` | no |
| Prefill | `O=QS + NV @ QK` | warp 10 | TMEM IO NV | shared QK | `128,64,64` | yes, forced state |
| Prefill | `state += decayV @ K.T` | warp 10 | TMEM IO decayV | shared K transpose | `128,128,64` | yes, forced state |

All FP16 and BF16 MN/prefill rows above emit CTA-group-one `kind::f16`; BF16
semantics are encoded by the MMA descriptors, not by a distinct opcode kind. The
UTC fixup alone uses the source TF32 tcgen05 instruction family. T precompute
alone uses warp HMMA for its K-K and inverse subproblems.

## Pipeline inventory

| Launch | Edge | Stages | Producer | Consumer(s) | Publication/reuse rule |
| --- | --- | ---: | --- | --- | --- |
| T | K TMA | 1 | warp 1 elected thread | CTA | expect/copy, source producer-cursor commit, TMA byte completion, then one CTA consumer release |
| T | beta software | 1 | warp 2 | CTA | async-view fence then 128-thread arrival |
| MN | K / V / T | 3 / 3 / 3 | warp 9 | two issuers / CG1 / state issuer | retain separate source cursors and elected-thread releases |
| MN | alpha | 4 | warp 10 | CG0 and CG1 | two consumer releases per stage |
| MN | X | 2 | CG0 materializer / warp 11 issuer | warp 8 and warp 11 | distinct accumulator-ready and shared-ready edges |
| MN | M/N inputs and accumulators | 1 each | compute groups / issuers | issuers / compute groups | no ownership flattening across CG0 and CG1 |
| UTC fixup | M / N | `2 or 1` / `1` | warp 5 | warp 4 / CG0 | branch-static M depth; N also supplies the no-initial first state |
| UTC fixup | ready / done | 1 / 1 | CG0 / warp 4 | warp 4 / CG0 | TMEM operand store fenced before ready commit |
| Prefill | Q / K / V / T | 2 / 3 / 3 / 2 | warp 9 | issuers / issuers / CG1 / CG0 | Q/K are two-issuer TmaUmma pipes; V/T use four elected releases |
| Prefill | gate | 5 | warp 11 | CG0 and CG1 | four-tile lookahead; 32 producers and 256 consumers |
| Prefill | CG0 acc / Ainv / QK | 2 / 3 / 2 | warp 8 / CG0 / CG0 | CG0 / warp 10 / warp 10 | exact pair-wise cursor order |
| Prefill | CG1 acc / Q-state / state | 1 / 1 / 1 | warp 10 | CG1 | KS then NV reuse the shared accumulator; QS then QKV reuse Q-state |
| Prefill | state-input / VKS / NV / decayV | 1 each | CG1 | warp 10 | VKS/NV/decayV are ready-only fixed-slot notifications |
| Prefill | O | 2 | CG1 | warp 11 | store group is committed and waited before slot release |

## TensorMap ABI

| Slot | Tensor | Initialization/update | Bounds used by device body |
| ---: | --- | --- | --- |
| 0 | Q | warp 9 copies the launch template, initializes, then updates GMEM descriptor | token dimension ends at `chunk_end` |
| 1 | K | warp 9 copies the launch template, initializes, then updates GMEM descriptor | token dimension ends at `chunk_end` |
| 2 | V | warp 9 copies the launch template, initializes, then updates GMEM descriptor | transposed token dimension ends at `chunk_end` |
| 3 | T | reserved 128 bytes; deliberately not initialized or read | fixed launch descriptor; padded chunk index clamps to the last valid T block |
| 4 | O | warp 11 independently initializes and updates GMEM descriptor | transposed token dimension ends at `chunk_end` |

The workspace is exactly
`[num_sequences,H_STATE,max_cp_chunks_per_seq,5,128]` bytes with every
descriptor slot 128-byte aligned. Workspace allocation/reuse, descriptor
construction, and JIT compilation remain outside the timed closure.

## Static specializations and module contract

The executable module must expose the six source-shaped PrimFuncs through
`get_kernel`: T, MN, SIMT-row4, UTCMMA-row64, UTCMMA-row128, and prefill. The
host closure allocates the five workspaces once, chooses the same chunk length
and production fixup branch as the source, and launches exactly
`T -> MN -> selected fixup -> prefill` on the caller stream. It must not fuse
launches or use a reference call inside the TIRx timed path.

Only the selected fixup writes the user-requested final state. The top-level
host always invokes prefill with `state=None`, so prefill is statically
`PREFILL_STORE_FINAL_STATE=False` and `PREFILL_STATE_DTYPE=f32`; its last CP CTA
still drains TMEM and performs the source conversion path, but emits no global
state store.

The seven `BENCH_CONFIGS` are all performance-required:

| Label | Static path forced by the row |
| --- | --- |
| `fp16_q1_k1_v1_s2048_none_i32` | FP16, dense no-state, automatic chunk, SIMT-row4 |
| `bf16_q4_k1_v1_s8193_final_f32_i64` | BF16, GQA, FP32 final, INT64 offsets, SIMT-row4 |
| `bf16_q1_k1_v4_s9999+65530_initfinal_bf16_i32` | BF16, GVA, initial/final state, automatic 2048 chunk, SIMT-row4 |
| `fp16_q16_k16_v16_s4096+4096_init_f16_i64` | FP16, 16 heads, automatic 1024 chunk, UTCMMA-row64 |
| `bf16_q2_k2_v8_s2048x4_final_f32_i32` | BF16 GVA, automatic 512 chunk, UTCMMA-row64 |
| `bf16_q16_k16_v16_s128+192_indexed_bf16_i32` | indexed BF16 state, automatic 128 chunk, UTCMMA-row64 |
| `fp16_q16_k16_v64_s192+64_initfinal_f16_i64` | FP16 GVA, automatic 128 chunk, UTCMMA-row128 |

The three additional `CONFIGS` cover a 96-token tail with explicit 128 chunk,
an explicit 64-token initial/final-state case, and a zero-length sequence next
to a 256-token sequence. Correctness must compare O and every requested final
state, including indexed destinations, without silently dropping any row.

## Bidirectional source coverage anchors

| Source artifact | Covered sketch region |
| --- | --- |
| `gdn_cp_prefill.py::cp_delta_rule_dsl_sm100` and its four wrappers | static host dispatch, workspace shapes, four launch boundary |
| `delta_rule_cp_sm120.py::CPDeltaRuleTPrecomputeSm120` | Launch 1, including explicit 8→16→32→64 inverse |
| `gated_delta_net_cp.py::CPDeltaRuleMNPrecomputeUtcmma1Sm100` | Launch 2 roles, 17 barrier rings, SMEM/TMEM allocation, alpha processing, and all five logical MN MMAs |
| `delta_rule_cp_sm120.py::CPDeltaRuleFixupSimtSm120` | Launch 3a row4 recurrence, prefetch handoff, final state, and gap zeroing |
| `gated_delta_net_cp.py::CPDeltaRuleFixupUtcmmaSm100` | Launch 3b/3c TF32 recurrence and branch-static row specializations |
| `gated_delta_net_cp_prefill.py::CPDeltaRulePrefillTcgen05Sm100` | Launch 4 descriptor, role, pipeline, state, gate, MMA, tail, and O-store paths |
| `alpha.py::AlphaProcessor` | MN source `log2/shuffle/exp2` alpha channels |
| `collective_inverse_hmma.py` | T source four-level inverse with no compound inverse primitive |
| `varlen_helper.py` | scalar automatic chunk selection and sequence/chunk indices |

## Instruction-selection summary

- The 1024-byte-aligned source SMEM layouts, single-CTA TensorMaps, tile rank,
  and mbarrier completion pointers select `cp.async.bulk.tensor` G2S loads;
  the O descriptor and shared epilogue layout select the matching S2G form.
- T precompute's register/shared fragments and `16x8x{8,16}` warp tiles select
  `mma.sync` plus `ldmatrix/stmatrix`. All MN and prefill logical GEMMs retain
  their source shapes and TMEM/SMEM operand placement so they select
  `tcgen05.mma.cta_group::1.kind::f16` for both IO dtypes; BF16 remains
  descriptor-encoded. UTC recurrence instead selects the TF32 family.
- TMEM column offsets and copy repetition shapes select the corresponding
  `tcgen05.ld/st` 16-row or 32-row family. An explicit TMEM-store wait remains
  before every producer publication that makes a TMEM operand reusable.
- MN/prefill FP32-to-IO fragment conversion stays in packed pairs via
  `cvt.rn.{f16,bf16}x2.f32`; the VKS subtraction likewise remains packed IO
  arithmetic. T/Ainv/QK element conversions remain scalar where their PTX is
  scalar.
- Every ring is represented by its source stage count and producer/consumer
  cardinality. Pipeline setup, waits, byte expectations, and software
  publication/reuse select `mbarrier.init.shared.b64`,
  `mbarrier.try_wait.parity.shared.b64`,
  `mbarrier.arrive.expect_tx.shared.b64`, and
  `mbarrier.arrive.shared.b64`, respectively. Tcgen completion remains the
  distinct `tcgen05.commit...shared::cluster.b64` instruction family.
- Each dynamic Q/K/V/O TensorMap is copied from its launch parameter with four
  `ld.param.v2.b64` plus two `st.global.v4.b64`, warp-synchronized, and sealed
  by `fence.acq_rel.cta`. Its update is a separate bulk commit/read-wait,
  warp-sync, `tensormap.replace...global.b1024` sequence, warp-sync, and release
  fence; the first TMA use retains the descriptor acquire fence.
- Gate/alpha preprocessing stays scalar/packed: global `ld`,
  `lg2.approx.ftz.f32`, `shfl.sync`, FP32 add, `ex2.approx.ftz.f32`, and shared
  stores. The prefill decay
  difference specifically retains `add.rn.f32x2` with FTZ disabled.
- The SIMT fixup retains global vector loads, scalar FP32 FMA, CTA barriers,
  and the 16-register next-chunk handoff. It cannot silently select an MMA
  implementation. Conversely the two UTC fixups retain the TF32 MMA path and
  cannot degrade to the SIMT recurrence.

Reviewer PTX must be freshly compiled with line information from the frozen
FlashInfer commit and must cover each of the six device variants, both FP16 and
BF16 descriptor encodings (which both emit `kind::f16`), both
INT32/INT64 sequence loads, and each state-storage conversion used by the
configuration table. A source statement or PTX instruction without a sketch
home, or a sketch key operation without a source/PTX home, is a review failure.
