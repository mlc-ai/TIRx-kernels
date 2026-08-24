<!--
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Modifications Copyright (c) 2026 The TIRx Authors.
SPDX-License-Identifier: Apache-2.0

This design sketch documents a modified TIRx port of cuDNN Frontend's
python/cudnn/linear_attention/frost/kernel/kda_bprop_f16.py at commit
aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5.
-->

# cuDNN SM100 KDA BF16 backward: coarse WASP pipeline sketch

This is a non-executable execution sketch, not a Python module, builder API,
mathematical reference, or alternate implementation.  The implementation it
describes is maintained in
[`tirx_kernels/cudnn/linear_attention/kda_bprop_f16.py`](../../../tirx_kernels/cudnn/linear_attention/kda_bprop_f16.py),
which is the source of truth after the correctness gate.

The source specialization is fixed to `BT=16`, `DK=DV=128`, BF16 I/O,
FP32 accumulation, one CTA per cluster, 512 main-kernel threads, 1024 prologue
threads, 196608 bytes of main dynamic SMEM, and 512 TMEM columns.  Capability
is the following discrete, source-verified set rather than a Cartesian product:
`basic`, `tail`, `grouped(HQ,HK,HV)=(4,1,2)`, `l2norm`, `safe_gate`,
`beta_sigmoid`, `state`, `dynamic`, `order_scratch`, and
`order_generate`.  Each label means the exact profile frozen in
`.porting/kda_bprop_f16/capability_manifest.yaml`; combinations not present
there are not claimed.  Source probes reject FP16 (NaN outputs) and the tested
all-features profile (outside the frozen error envelope); neither is represented
here.

Writer line-info evidence is under
`.porting/kda_bprop_f16/source_export/basic_debug_001/`.  Its two PTX files have
both `.file` and `.loc`; branch deltas are under the sibling profile exports.
Unless stated otherwise, PTX line numbers below refer to the debug anchor's
`cutlass_host_KdaBwdCfg...sm_100a.ptx`.  Static counts use instruction lines
minus predicated lines.

## Pipeline at a glance

| Warps | Register target | Role-local program | Main publication/reuse edges |
| --- | ---: | --- | --- |
| 0..3 | 144 | forward gate prefix, beta, Q/K normalization, decay/restore operands, state/diagonal restaging | raw Q/K/Gate, beta, state, decay, Q/K raw TMEM |
| 4..7 | 168 | value-side TMEM drains/restages, dState recurrence, dBeta and dInitialState | raw V/dO, U/dY, dState, dV, dBeta |
| 8..11 | 144 | drain dQ/dK TMEM, assemble dGate, reverse prefix reduction | dQ/dK/dGate output stages, dState reuse |
| 12 | 56 | register `mma.sync` program for KK, beta-scaled strict-L power-series inverse, dM, positive/negative strict dM, and dBeta-M | beta, intermediate tiles, decay operands, dY/U |
| 13 | 56 | allocate 512 TMEM columns and issue the 15 logical `tcgen05.mma` chains | every TMEM accumulator/input edge and teardown |
| 14 | 56 | persistent scheduler plus Q/K/V/Gate/dO/checkpoint TensorMap loads | five raw rings, state ring, scheduler publication |
| 15 | 56 | register `A=tril(Q_decay@K_inv^T)` and `dA=tril(dO@U^T)`, then one-behind dQ/dK/dV/dGate TensorMap stores | A/dA intermediate tiles, dO reuse, output-stage reuse |

Role dispatch is the source-order `if/elif` chain: 14, 12, 13, 15, 0..3,
8..11, then 4..7.  Every role constructs its own persistent scheduler cursor;
the eight scheduler stages are never reset between work items.

## Primitive vocabulary

All storage is linear.  No declaration or operation carries a layout object,
layout value, tile primitive, or first-class view.  Logical matrices below are
only index ranges selected by scalar address functions.

```python
linear_buffer(space, dtype, elements, byte_offset, alignment, lifetime)
reg_array(dtype, elements)
smem_byte(base, stage, row, col, stage_bytes, row_bytes, elem_bytes, xor_bits)
tmem_cell(base, row, col)       # base + col + (row << 16)
gmem_element(base, scalar_index)
descriptor_slot(workspace, array, batch)  # workspace + (array*B+batch)*128
```

Directional movement primitives:

```python
copy_p2g(src, dst)                         # parameter descriptor -> GMEM
copy_g2s(src, dst, predicate, completion) # GMEM -> SMEM
copy_s2g(src, dst, predicate)             # SMEM -> GMEM
copy_g2r(src, dst, predicate=True)
copy_r2g(src, dst, predicate=True)
copy_s2r(src, dst)
copy_r2s(src, dst)
copy_t2r(src, dst)
copy_r2t(src, dst)
```

Computational primitives are only:

```python
fill(dst, value)
cast(dst, src)
add(dst, lhs, rhs)
sub(dst, lhs, rhs)
mul(dst, lhs, rhs)
fma(dst, lhs, rhs, acc)
div(dst, lhs, rhs)
exp2(dst, src)
tanh(dst, src)
rsqrt(dst, src)
min(dst, lhs, rhs)
max(dst, lhs, rhs)
select(dst, predicate, true_value, false_value)
shuffle_xor(dst, src, delta)
shuffle_up(dst, src, delta)
gemm(dst, lhs, rhs, accumulate=False)
```

`init`, `wait`, `arrive`, `expect_bytes`, `commit`, `release`, `fence`,
`barrier`, stage/phase advancement, register-budget changes, descriptor field
replacement, directional-copy groups, and TMEM allocation are schedule
operations.  Each key movement, computation, or synchronization occurrence
below is immediately followed by the selected instruction or instruction
family observed in the writer export.

## Complete sketch

```python
# ==========================================================================
# Static parameters and the two-launch runtime ABI
# ==========================================================================

@specialize(
    IO_DTYPE="bf16", BT=16, DK=128, DV=128,
    RAW_STAGES=2, STATE_STAGES=1, DECAY_STAGES=2,
    INTERMEDIATE_STAGES=2, DQ_STAGES=1, DK_STAGES=1,
    DGATE_STAGES=1, DV_STAGES=2, BETA_STAGES=4,
    QK_TMEM_STAGES=4, SCHED_STAGES=8,
    VERIFIED_PROFILE={
        "basic", "tail", "grouped", "l2norm",
        "safe_gate", "beta_sigmoid", "state", "dynamic",
        "order_scratch", "order_generate",
    },
)
def kda_bprop_f16_factory(...):
    return descriptor_order_prologue, persistent_backward

@kernel(grid=(1,1,1), block=(1024,1,1), warps=32, target="sm_100a")
def descriptor_order_prologue(
    base_maps[10], descriptor_workspace, cu_seqlens,
    q, k, v, gate, do, dq, dk, dv, dgate, state_checkpoints,
    staging_items, work_count, work_items, sched_all,
    batch_count, row_strides[10], checkpoint_every_n,
):
    tid = thread_id()
    warp = tid // 32

    if RUN_ORDER:
        # Exactly two rank-1 i32 arrays plus one two-cell i32 range.  The
        # 4096-entry capacity is fixed; no logical layout is attached.
        order_key = linear_buffer("smem", "i32", 4096, 0, 16,
                                  "whole order pass")
        order_idx = linear_buffer("smem", "i32", 4096, 16384, 16,
                                  "whole order pass")
        order_spread = linear_buffer("smem", "i32", 2, 32768, 8,
                                     "whole order pass")

        if tid == 0 and HAS_SCHED:
            for i in range(len(sched_all)):
                copy_r2g(i32(0), sched_all[i])
                # instruction_selection: st.global.u32; extent: scalar loop

        n = batch_count * HO if ORDER_GEN else copy_g2r(work_count[0], i32_reg())
        # instruction_selection: ld.global.u32; extent: one elected scalar
        if ORDER_GEN and tid == 0:
            copy_r2g(n, work_count[0])
            # instruction_selection: st.global.u32; extent: scalar

        if n > 4096:
            for item in range(tid, n, 1024):
                if ORDER_GEN:
                    synthesize_eight_i32_fields(item, row_regs)
                    # instruction_selection: integer scalar arithmetic;
                    # extent: one generated work row
                    copy_r2g(row_regs[0:8], work_items[item,0:8])
                    # instruction_selection: st.global.v4.b32 pairs;
                    # extent: one generated row
                else:
                    for field in range(8):
                        copy_g2r(staging_items[item,field], row_regs[field])
                        # instruction_selection: ld.global.u32;
                        # extent: scalar dynamic-stride loop
                        copy_r2g(row_regs[field], work_items[item,field])
                        # instruction_selection: st.global.u32;
                        # extent: scalar dynamic-stride loop
        else:
            if tid == 0:
                fill(order_spread[0:2], [INT_MAX, INT_MIN])
                # instruction_selection: st.shared.v2.u32; extent: two scalars
            barrier(0, 1024)
            # instruction_selection: barrier.sync 0, 1024; extent: CTA,
            # before any key write
            for e in range(4):
                item = tid + e * 1024
                if item < next_power_of_two(n):
                    key = item_chunk_span_or_negative_infinity(item)
                    copy_r2s(key, order_key[item])
                    # instruction_selection: st.shared.u32; extent: scalar
                    copy_r2s(item, order_idx[item])
                    # instruction_selection: st.shared.u32; extent: scalar
            atomic_min(order_spread[0], local_key_min)
            # instruction_selection: atom.shared::cta.min.s32; extent: scalar
            atomic_max(order_spread[1], local_key_max)
            # instruction_selection: atom.shared::cta.max.s32; extent: scalar
            barrier(0, 1024)
            # instruction_selection: barrier.sync 0, 1024; extent: CTA

            if order_spread[0] == order_spread[1]:
                if ORDER_GEN:
                    for row in range(tid, n, 1024):
                        synthesize_eight_i32_fields(row, row_regs)
                        # instruction_selection: integer scalar arithmetic;
                        # extent: one row
                        copy_r2g(row_regs[0:8], work_items[row,0:8])
                        # instruction_selection: st.global.v4.b32 pairs;
                        # extent: generated eight-field row
                else:
                    for row in range(tid, n, 1024):
                        for field in range(8):
                            copy_g2r(staging_items[row,field], row_regs[field])
                            # instruction_selection: ld.global.u32;
                            # extent: scalar dynamic-stride loop
                            copy_r2g(row_regs[field], work_items[row,field])
                            # instruction_selection: st.global.u32;
                            # extent: scalar dynamic-stride loop
            else:
                k = 2
                while k <= next_power_of_two(n):
                    j = k // 2
                    while j > 0:
                        for e in range(4):
                            i = tid + e * 1024
                            partner = i ^ j
                            if partner > i and bitonic_swap_predicate(i, partner, k):
                                copy_s2r(order_key[i], key_i)
                                # instruction_selection: ld.shared.u32; extent: scalar
                                copy_s2r(order_key[partner], key_j)
                                # instruction_selection: ld.shared.u32; extent: scalar
                                copy_r2s(key_j, order_key[i])
                                # instruction_selection: st.shared.u32; extent: scalar
                                copy_r2s(key_i, order_key[partner])
                                # instruction_selection: st.shared.u32; extent: scalar
                                swap_two_shared_indices(i, partner)
                                # instruction_selection: ld.shared/st.shared.u32;
                                # extent: one scalar pair
                        barrier(0, 1024)
                        # instruction_selection: barrier.sync 0, 1024; extent: CTA
                        j //= 2
                    k *= 2
                for row in range(tid, n, 1024):
                    copy_s2r(order_idx[row], source_row)
                    # instruction_selection: ld.shared.u32; extent: scalar
                    if ORDER_GEN:
                        synthesize_eight_i32_fields(source_row, row_regs)
                        # instruction_selection: integer scalar arithmetic;
                        # extent: one row
                        copy_r2g(row_regs[0:8], work_items[row,0:8])
                        # instruction_selection: st.global.v4.b32 pairs;
                        # extent: generated eight-field row
                    else:
                        for field in range(8):
                            copy_g2r(staging_items[source_row,field], row_regs[field])
                            # instruction_selection: ld.global.u32;
                            # extent: scalar dynamic-stride loop
                            copy_r2g(row_regs[field], work_items[row,field])
                            # instruction_selection: st.global.u32;
                            # extent: scalar dynamic-stride loop

    # Warps 0..8 build packed-token descriptors.  Their address is the base
    # tensor pointer plus cu_seqlens[b]*row_stride, and dim 2 is the exact
    # per-sequence token count.
    if warp < 9 and elected_lane():
        for batch in range(batch_count):
            dst = descriptor_slot(descriptor_workspace, warp, batch)
            copy_p2g(base_maps[warp][0:128], dst[0:128])
            # instruction_selection: 16 x st.global.b64 after descriptor
            # parameter materialization; extent: one 128-byte descriptor
            cu_b = copy_g2r(cu_seqlens[batch])
            cu_next = copy_g2r(cu_seqlens[batch + 1])
            sequence_address = tensor_base(warp)
            sequence_address += i64(cu_b) * i64(row_strides[warp])
            replace_descriptor_address(dst, sequence_address)
            # instruction_selection:
            # tensormap.replace.tile.global_address.global.b1024.b64;
            # extent: one descriptor field
            replace_descriptor_dim2(dst, cu_next - cu_b)
            # instruction_selection:
            # tensormap.replace.tile.global_dim.global.b1024.b32;
            # extent: one descriptor field
        fence("tensormap_generic_release_gpu")
        # instruction_selection:
        # one fence.proxy.tensormap::generic.release.gpu after the whole
        # elected-warp batch loop

    # Warp 9 uses a different scalar address/dimension recurrence.  Empty
    # sequences contribute zero checkpoints; no token base is reused here.
    if warp == 9 and elected_lane():
        checkpoint_prefix = 0
        for batch in range(batch_count):
            dst = descriptor_slot(descriptor_workspace, 9, batch)
            copy_p2g(base_maps[9][0:128], dst[0:128])
            cu_b = copy_g2r(cu_seqlens[batch])
            cu_next = copy_g2r(cu_seqlens[batch + 1])
            seqlen_b = cu_next - cu_b
            count_b = select(seqlen_b > 0,
                             (seqlen_b - 1) // checkpoint_every_n + 1,
                             0)
            checkpoint_address = tensor_base(9)
            checkpoint_address += i64(checkpoint_prefix) * i64(row_strides[9])
            replace_descriptor_address(dst, checkpoint_address)
            replace_descriptor_dim2(dst, count_b)
            checkpoint_prefix += count_b
        fence("tensormap_generic_release_gpu")
        # instruction_selection: one generic-to-TensorMap release fence after
        # all checkpoint descriptors

# Descriptor arrays 0..8 are rank-3, BF16 boxes [64,1,16] except Gate and
# dGate FP32 boxes [32,1,16].  Array 9 is rank-4 checkpoint BF16 with box
# [64,128,1,1].  All use 128-byte swizzle and 128-byte GMEM slots.

@kernel(
    grid=(num_sms,1,1), block=(512,1,1), warps=16,
    cluster=(1,1,1), cta_group=1, dynamic_smem_bytes=196608,
    tmem_columns=512, min_blocks_per_sm=1, target="sm_100a",
)
def persistent_backward(
    descriptor_workspace, n_desc, a_log, dt_bias, beta, cu_seqlens,
    dgate, dbeta, d_initial_state, d_final_state,
    work_items, work_count, sched_counter, scale,
):
    tid = thread_id()
    warp = warp_uniform(tid // 32)
    lane = tid % 32
    cta = block_id_x()
    grid_x = grid_dim_x()
    total_tiles = copy_g2r(work_count[0], i32_reg())
    # instruction_selection: ld.global.u32; extent: scalar, not hoisted across
    # the persistent scheduler's atomic path

    # ======================================================================
    # One rank-1 u8 SMEM arena and explicit integer physical mappings
    # ======================================================================

    smem = linear_buffer("smem", "u8", 196608, 0, 1024, "whole CTA")

    # No buffer below exists as a first-class multidimensional value.  These
    # names are byte intervals consumed only through smem_byte(...).
    BARRIERS_BASE = 0;       BARRIERS_END = 920       # 115 x 8-byte words
    TMEM_MAILBOX_BASE = 920; TMEM_MAILBOX_END = 924   # one i32
    SCHED_BASE = 928;        SCHED_END = 960          # eight i32 tickets
    BETA_BASE = 960;         BETA_END = 1216          # 4*16 f32
    NORM_BASE = 1216;        NORM_END = 1728          # 4*2*16 f32
    RED0_BASE = 1728;        RED0_END = 1984          # 4*16 f32
    RED1_BASE = 1984;        RED1_END = 2240          # 4*16 f32
    BETAM_BASE = 2240;       BETAM_END = 2304         # 16 f32
    PADDING_BASE = 2304;     PADDING_END = 3072
    STATE_BASE = 3072;       STATE_END = 35840        # 128*128 bf16
    DSTATE_BASE = 35840;     DSTATE_END = 68608       # 128*128 bf16
    KDECAY_BASE = 68608;     KDECAY_END = 76800       # 2*16*128 bf16
    KINV_BASE = 76800;       KINV_END = 84992
    KRESTORE_BASE = 84992;   KRESTORE_END = 93184
    QDECAY_BASE = 93184;     QDECAY_END = 101376
    DIAG_BASE = 101376;      DIAG_END = 109568        # 2*8*256 bf16
    INTER_BASE = 109568;     INTER_END = 114688       # 2*5*16*16 bf16
    DO_BASE = 114688;        DO_END = 122880          # 2*16*128 bf16
    DV_BASE = 122880;        DV_END = 131072          # 2*16*128 bf16
    Q_BASE = 131072;         Q_END = 139264           # 2*16*128 bf16
    K_BASE = 139264;         K_END = 147456
    V_BASE = 147456;         V_END = 155648
    GATE_BASE = 155648;      GATE_END = 172032        # 2*16*128 f32
    DY_BASE = 172032;        DY_END = 176128          # 16*128 bf16
    U_BASE = 176128;         U_END = 180224            # 16*128 bf16
    DQ_BASE = 180224;        DQ_END = 184320           # 16*128 bf16
    DK_BASE = 184320;        DK_END = 188416           # 16*128 bf16
    DGATE_BASE = 188416;     DGATE_END = 196608        # 16*128 f32
    assert DGATE_END == 196608

    # All source mappings are scalar byte functions.  They deliberately stay
    # separate because merging them changes selected bytes/descriptors.
    def xor128_element(row, col_in_segment, elem_bytes):
        return col_in_segment ^ ((row & 7) * (16 // elem_bytes))

    def raw_bf16_byte(base, stage, row16, channel128):
        segment = channel128 // 64
        in_segment = channel128 - segment * 64
        element = segment * (16 * 64) + row16 * 64
        element += xor128_element(row16, in_segment, 2)
        return base + stage * 4096 + element * 2

    def raw_f32_byte(base, stage, row16, channel128):
        segment = channel128 // 32
        in_segment = channel128 - segment * 32
        element = segment * (16 * 32) + row16 * 32
        element += xor128_element(row16, in_segment, 4)
        return base + stage * 8192 + element * 4

    def state_physical_byte(base, key_row128, value_col128):
        # TMA checkpoint storage: two 64-value segments, each with 128 key
        # rows.  Both state descriptors below point at these exact bytes.
        segment = value_col128 // 64
        in_segment = value_col128 - segment * 64
        element = segment * (128 * 64) + key_row128 * 64
        element += xor128_element(key_row128, in_segment, 2)
        return base + element * 2

    def dstate_physical_byte(base, value_row128, key_col128):
        # Recurrence storage: two 64-key segments, each with 128 value rows.
        segment = key_col128 // 64
        in_segment = key_col128 - segment * 64
        element = segment * (128 * 64) + value_row128 * 64
        element += xor128_element(value_row128, in_segment, 2)
        return base + element * 2

    def operand_lead16_byte(base, stage, row16, channel128):
        return raw_bf16_byte(base, stage, row16, channel128)

    def operand_amajor_byte(base, stage, channel128, row16):
        return raw_bf16_byte(base, stage, row16, channel128)

    def operand_transpose_byte(base, stage, channel128, row16):
        return raw_bf16_byte(base, stage, row16, channel128)

    def xor_linear_s(element_index, bbits, mbase, sshift):
        y = (element_index >> (mbase + sshift)) & ((1 << bbits) - 1)
        return element_index ^ (y << mbase)

    def diagonal_bf16_byte(base, stage, block8, row16, col16):
        logical = block8 * 256 + row16 * 16 + col16
        return base + stage * 4096 + xor_linear_s(logical, 1, 3, 3) * 2

    def intermediate_bf16_byte(base, stage, slot5, row16, col16):
        logical = slot5 * 256 + row16 * 16 + col16
        return base + stage * 2560 + xor_linear_s(logical, 1, 3, 3) * 2

    # Named raw/output scalar mappings.  Every function returns one integer
    # byte address; none returns a buffer, matrix, fragment, or view.
    def q_raw_bf16_byte(stage, token, channel):
        return raw_bf16_byte(Q_BASE, stage, token, channel)

    def k_raw_bf16_byte(stage, token, channel):
        return raw_bf16_byte(K_BASE, stage, token, channel)

    def v_raw_bf16_byte(stage, token, channel):
        return raw_bf16_byte(V_BASE, stage, token, channel)

    def do_raw_bf16_byte(stage, token, channel):
        return raw_bf16_byte(DO_BASE, stage, token, channel)

    def gate_raw_f32_byte(stage, token, channel):
        return raw_f32_byte(GATE_BASE, stage, token, channel)

    def dv_output_bf16_byte(stage, token, channel):
        return raw_bf16_byte(DV_BASE, stage, token, channel)

    def dq_output_bf16_byte(token, channel):
        return raw_bf16_byte(DQ_BASE, 0, token, channel)

    def dk_output_bf16_byte(token, channel):
        return raw_bf16_byte(DK_BASE, 0, token, channel)

    def dgate_output_f32_byte(token, channel):
        return raw_f32_byte(DGATE_BASE, 0, token, channel)

    def beta_f32_byte(stage, token):
        return BETA_BASE + (stage * 16 + token) * 4

    def norm_f32_byte(qk_stage_id, is_k, token):
        return NORM_BASE + (qk_stage_id * 32 + is_k * 16 + token) * 4

    def reduction0_f32_byte(warp_in_group, token):
        return RED0_BASE + (warp_in_group * 16 + token) * 4

    def reduction1_f32_byte(warp_in_group, token):
        return RED1_BASE + (warp_in_group * 16 + token) * 4

    def beta_matrix_f32_byte(token):
        return BETAM_BASE + token * 4

    def state_bf16_byte(key_row, value_col):
        return state_physical_byte(STATE_BASE, key_row, value_col)

    def dstate_direct_bf16_byte(value_row, key_col):
        return dstate_physical_byte(DSTATE_BASE, value_row, key_col)

    def dstate_alt_transpose_bf16_byte(value_row, key_col):
        # Alternate is a descriptor interpretation of the same physical byte.
        return dstate_physical_byte(DSTATE_BASE, value_row, key_col)

    def k_decay_operand_bf16_byte(stage, token, channel):
        return operand_lead16_byte(KDECAY_BASE, stage, token, channel)

    def k_inverse_operand_bf16_byte(stage, token, channel):
        return operand_lead16_byte(KINV_BASE, stage, token, channel)

    def k_restore_operand_bf16_byte(stage, token, channel):
        return operand_lead16_byte(KRESTORE_BASE, stage, token, channel)

    def q_decay_operand_bf16_byte(stage, token, channel):
        return operand_lead16_byte(QDECAY_BASE, stage, token, channel)

    def state_diag_bf16_byte(stage, block8, row, col):
        return diagonal_bf16_byte(DIAG_BASE, stage, block8, row, col)

    def intermediate_slot_bf16_byte(stage, slot, row, col):
        return intermediate_bf16_byte(INTER_BASE, stage, slot, row, col)

    # Raw descriptor integers.  The three address fields use hardware 16-byte
    # granules.  `layout_code` is already the exact tcgen05 encoding (2 for
    # the 128-byte raw/state/operand arrangement and 6 for the 32-byte
    # diagonal/intermediate arrangement), so no TMA-swizzle remapping occurs.
    # The result is one u64 integer, never a descriptor object.
    def raw_descriptor(base_byte, leading_bytes, stride_bytes, layout_code):
        value = ((base_byte >> 4) & 0x3FFF)
        value |= ((leading_bytes >> 4) & 0x3FFF) << 16
        value |= ((stride_bytes >> 4) & 0x3FFF) << 32
        value |= 1 << 46
        value |= (layout_code & 7) << 61
        return value & 0xFFFFFFFFFFFFFFFF

    def operand_lead16_desc(base_byte):
        return raw_descriptor(base_byte, 16, 1024, 2)

    def operand_amajor_desc(base_byte):
        return raw_descriptor(base_byte, 2048, 1024, 2)

    def operand_transpose_desc(base_byte):
        return raw_descriptor(base_byte, 2048, 1024, 2)

    def state_direct_desc(base_byte):
        return raw_descriptor(base_byte, 16, 1024, 2)

    def state_alt_desc(base_byte):
        return raw_descriptor(base_byte, 16384, 1024, 2)

    def diagonal_desc(base_byte):
        return raw_descriptor(base_byte, 16, 256, 6)

    def intermediate_desc(base_byte):
        return raw_descriptor(base_byte, 16, 256, 6)

    # TMEM operations take integer cells: low 16 bits are columns, high bits
    # are rows.  Every function receives the allocated integer base/row; no
    # closure refers to a later declaration and no function returns a view.
    def tmem_cell(allocated_base, allocated_row, row_delta, col):
        return allocated_base + col + ((allocated_row + row_delta) << 16)

    def cg0_tmem_row_delta(warp_id):
        return (warp_id % 4) * 32

    def cg1_tmem_row_delta(warp_id):
        return (warp_id % 4) * 32

    def cg1_tmem_row_lo_delta():
        return 0

    def cg1_tmem_row_hi_delta():
        return 16

    def cg2_tmem_row_delta(warp_id):
        # tcgen05.ld.32x32b distributes the 32 rows to lanes internally.
        return (warp_id % 4) * 32

    def cg0_qraw_tmem_cell(base, row, warp_id, qk_stage_id):
        return tmem_cell(base, row, cg0_tmem_row_delta(warp_id),
                         448 + qk_stage_id * 8)

    def cg0_kraw_tmem_cell(base, row, warp_id, qk_stage_id):
        return tmem_cell(base, row, cg0_tmem_row_delta(warp_id),
                         480 + qk_stage_id * 8)

    def cg0_state_tmem_cell(base, row, warp_id,
                            state_input_stage_id, value_half, value_group8):
        return tmem_cell(base, row, cg0_tmem_row_delta(warp_id),
                         192 + state_input_stage_id * 64
                         + value_half * 32 + value_group8 * 4)

    def cg1_tmem_cell_lo(base, row, col):
        return tmem_cell(base, row, cg1_tmem_row_lo_delta(), col)

    def cg1_tmem_cell_hi(base, row, col):
        return tmem_cell(base, row, cg1_tmem_row_hi_delta(), col)

    def cg1_dstate_tmem_cell(base, row, warp_id, col):
        return tmem_cell(base, row, cg1_tmem_row_delta(warp_id), col)

    def cg1_projection_tmem_cell(base, row, warp_id, value_half, col):
        return tmem_cell(base, row,
                         cg1_tmem_row_delta(warp_id) + value_half * 16,
                         col)

    def cg2_tmem_channel_cell(base, row, warp_id, col):
        return tmem_cell(base, row, cg2_tmem_row_delta(warp_id), col)

    # Register-MMA and stmatrix scalar lane coordinates shared by warps 12/15.
    def rhs_ldmatrix_row(lane_id):
        return lane_id % 8 + select(lane_id // 16 != 0, 8, 0)

    def rhs_ldmatrix_col(lane_id):
        return select(((lane_id // 8) & 1) != 0, 8, 0)

    def lhs_ldmatrix_row(lane_id):
        return lane_id % 8 + select(((lane_id // 8) & 1) != 0, 8, 0)

    def lhs_ldmatrix_col(lane_id):
        return select(lane_id // 8 >= 2, 8, 0)

    def stmatrix_row(lane_id):
        return lane_id % 8 + select(((lane_id // 8) & 1) != 0, 8, 0)

    def stmatrix_col(lane_id):
        return select(lane_id // 8 >= 2, 8, 0)

    def stmatrix_intermediate_byte(stage, slot, lane_id):
        return intermediate_slot_bf16_byte(
            stage, slot, stmatrix_row(lane_id), stmatrix_col(lane_id))

    def super_row_lo(lane_id):
        return lane_id // 4

    def super_row_hi(lane_id):
        return lane_id // 4 + 8

    # Stage bases used by TMA are still scalar byte addresses.  Register and
    # tcgen users select individual bytes/cells with the functions that follow.
    def q_raw_stage(stage):
        return q_raw_bf16_byte(stage, 0, 0)

    def k_raw_stage(stage):
        return k_raw_bf16_byte(stage, 0, 0)

    def v_raw_stage(stage):
        return v_raw_bf16_byte(stage, 0, 0)

    def do_raw_stage(stage):
        return do_raw_bf16_byte(stage, 0, 0)

    def gate_raw_stage(stage):
        return gate_raw_f32_byte(stage, 0, 0)

    def dv_stage_at(stage):
        return dv_output_bf16_byte(stage, 0, 0)

    def state_region():
        return state_bf16_byte(0, 0)

    # Exact source descriptor start-address increments for the 15 tcgen rows.
    # Each result below is a single u64 or TMEM integer address.
    def state_smem_subtile(key_phase16):
        return state_direct_desc(state_bf16_byte(key_phase16 * 16, 0))

    def k_decay_smem_subtile(stage, key_phase16):
        return operand_lead16_desc(
            k_decay_operand_bf16_byte(stage, 0, key_phase16 * 16))

    def do_smem_lead16_subtile(stage, value_phase16):
        return operand_lead16_desc(
            do_raw_bf16_byte(stage, 0, value_phase16 * 16))

    def do_smem_amajor_desc(stage):
        return operand_amajor_desc(do_raw_bf16_byte(stage, 0, 0))

    def q_decay_smem_transpose_desc(stage):
        return operand_transpose_desc(
            q_decay_operand_bf16_byte(stage, 0, 0))

    def k_decay_smem_transpose_desc(stage):
        return operand_transpose_desc(
            k_decay_operand_bf16_byte(stage, 0, 0))

    def k_inverse_smem_amajor_desc(stage):
        return operand_amajor_desc(
            k_inverse_operand_bf16_byte(stage, 0, 0))

    def k_restore_smem_subtile(stage, key_phase16):
        return operand_lead16_desc(
            k_restore_operand_bf16_byte(stage, 0, key_phase16 * 16))

    def u_smem_lead16_subtile(value_phase16):
        return operand_lead16_desc(raw_bf16_byte(
            U_BASE, 0, 0, value_phase16 * 16))

    def dv_smem_lead16_subtile(stage, value_phase16):
        return operand_lead16_desc(
            dv_output_bf16_byte(stage, 0, value_phase16 * 16))

    def dstate_smem_alt_desc(value_phase16):
        return state_alt_desc(
            dstate_alt_transpose_bf16_byte(value_phase16 * 16, 0))

    def state_diag_smem_subtile(stage, key_block16):
        return diagonal_desc(state_diag_bf16_byte(stage, key_block16, 0, 0))

    def intermediate_smem_desc(stage, slot):
        return intermediate_desc(intermediate_slot_bf16_byte(
            stage, slot, 0, 0))

    def tmem_state_input_subtile(base, row, stage, key_phase16):
        return tmem_cell(base, row, 0, 192 + stage * 64 + key_phase16 * 8)

    def tmem_dstate_input_subtile(base, row, key_phase16):
        return tmem_cell(base, row, 0, 128 + key_phase16 * 8)

    def tmem_y_input(base, row):
        return tmem_cell(base, row, 0, 432)

    def tmem_du_input(base, row):
        return tmem_cell(base, row, 0, 440)

    def tmem_negative_beta_dy_input(base, row):
        return tmem_cell(base, row, 0, 432)

    # Raw BF16 tcgen descriptors: lead=16 B -> encoded 1, stride=1024 B ->
    # encoded 64, layout code 2.  A-major/transpose variants encode
    # lead=BT*128=2048 B -> 128 with the same encoded stride/layout.  State
    # alternate encodes lead=16384 B -> 1024.  Diagonal/intermediate encode
    # lead=1, stride=16, layout code 6.  These are raw integer descriptor
    # fields, not first-class layouts.

    # ======================================================================
    # The 115-word protocol header
    # ======================================================================

    # Each row identifies every physical word.  `ready` and `done` offsets are
    # independent arrays, each containing `stages` adjacent eight-byte words.
    # `owner` is the elected initialization warp.  `None` means no paired word.
    protocol = (
      # name                    ready done stages ready_count done_count owner
      ("q",                       0,  16, 2,   1, 128, 14),
      ("k",                      32,  48, 2,   1, 128, 14),
      ("gate",                   64,  80, 2,   1, 256, 14),
      ("do",                     96, 112, 2,   1,  32, 14),
      ("v",                     128, 144, 2,   1, 128, 14),
      ("state",                 160, 168, 1,   1,   1, 14),
      ("state_cg0_done",        176, None, 1, 128, None, 14),
      ("beta",                  184, 216, 4,  32, 160, 14),
      ("state_k_acc",           248, None, 1,   1, None, 13),
      ("du_acc",                256, None, 1,   1, None, 13),
      ("u_acc",                 264, None, 1,   1, None, 13),
      ("dy_acc",                272, None, 1,   1, None, 13),
      ("dq_acc",                280, None, 1,   1, None, 13),
      ("dk_decay_acc",          288, None, 1,   1, None, 13),
      ("dk_inv_acc",            296, None, 1,   1, None, 13),
      ("dk_restore_acc",        304, None, 1,   1, None, 13),
      ("dqk_done",              312, None, 1, 128, None, 13),
      ("qk_raw",                320, 352, 4, 128, 128, 12),
      ("state_input",           384, 400, 2, 128,   1, 14),
      ("state_input_cg2_done",  416, None, 2, 128, None, 14),
      ("y_input",               432, None, 1, 128, None, 13),
      ("du_input",              440, None, 1, 128, None, 13),
      ("neg_beta_dy_input",     448, None, 1, 128, None, 13),
      ("k_decay_inv",           456, None, 2, 128, None, 12),
      ("q_decay_k_restore",     472, None, 2, 128, None, 12),
      ("decay_done",            488, None, 2,   1, None, 12),
      ("t_inv",                 504, 520, 2,  32,   1, 12),
      ("a",                     536, 552, 2,  32,   1, 12),
      ("da",                    568, 584, 2,  32,   1, 12),
      ("dm",                    600, 616, 2,  32,   1, 12),
      ("u_smem",                632, None, 1, 128, None, 13),
      ("dy_smem",               640, None, 1, 128, None, 13),
      ("dbeta_matrix",          648, None, 1,  32, None, 13),
      ("dstate_acc",            656, None, 1,   1, None, 13),
      ("dstate_input",          664, None, 1, 128, None, 13),
      ("dstate_smem",           672, 680, 1, 128,   1, 13),
      ("dstate_smem_cg2_done",  688, None, 1, 128, None, 13),
      ("dq_store",              696, 704, 1, 128,  32, 15),
      ("dk_store",              712, 720, 1, 128,  32, 15),
      ("dv_store",              728, 744, 2, 128,  32, 15),
      ("dgate_store",           760, 768, 1, 128,  32, 15),
      ("dstate0_stored",        776, None, 1, 128, None, 13),
      ("tmem_done",             784, None, 1, 256, None, 13),
      ("scheduler",             792, 856, 8,   1,  15, 15),
    )
    assert physical_words(protocol) == 115

    # Participant cursors are `(stage,phase)` and persist across work items.
    # The TMA raw/state/scheduler producers start `(0,1)`.  The CG0 QK-raw,
    # decay/intermediate, beta, state-input, and output-stage producers wait on
    # their paired done words with phase 1.  CG1's dState-SMEM and dV producers
    # likewise start their done cursors at phase 1.  Tcgen's `dqk_done`
    # consumer starts phase 1.  All ready consumers and remaining one-way
    # consumers start phase 0.  Wrap toggles phase and no role resets a cursor.
    initial_cursors = (
      ("tma.raw",0,1), ("tma.state",0,1), ("tma.scheduler",0,1),
      ("cg0.beta_done",0,1), ("cg0.qk_raw_done",0,1),
      ("cg0.decay_done",0,1), ("cg0.state_input_done",0,1),
      ("epilogue.a_done",0,1), ("epilogue.da_done",0,1),
      ("super.tinv_done",0,1), ("super.dm_done",0,1),
      ("cg1.dstate_smem_done",0,1), ("cg1.dv_done",0,1),
      ("cg2.dq_done",0,1), ("cg2.dk_done",0,1),
      ("cg2.dgate_done",0,1), ("tcgen.dqk_done",0,1),
      ("all_ready_consumers",0,0),
    )

    # Barrier construction is distributed across warps 14, 13, 12, and 15.
    # Every word is initialized exactly once by an elected lane.
    for assigned_edge_stage in role_assigned_protocol_words(warp):
        init(assigned_edge_stage, arrivals=assigned_arrival_count)
        # instruction_selection: mbarrier.init.shared.b64; extent: one of 115 words
    for diagonal_element in range(4096):
        copy_r2s(bf16(0), DIAG_BASE + diagonal_element * 2)
    # instruction_selection: st.shared.u16; extent: strided 8192-byte region
    fence("mbarrier_init_release_cluster")
    # instruction_selection: fence.mbarrier_init.release.cluster; extent: CTA protocol
    barrier(0, 512)
    # instruction_selection: barrier.sync 0, 512; extent: CTA

    # Named barriers: ID 3 covers all TMEM users, not only allocation.
    CG0_SYNC = (1,128)
    CG2_SYNC = (2,128)
    TMEM_LIFETIME = (3,416)
    CG1_SYNC = (4,128)

    # TMEM columns; BF16 inputs pack two elements per 32-bit cell.
    tmem_allocated = integer_base_from_mailbox(TMEM_MAILBOX_BASE)
    tmem_base = tmem_allocated & 0xFFFF
    tmem_row = tmem_allocated >> 16
    dstate_acc_col = 0
    dstate_input_col = 128
    state_input_col = 192
    state_k_or_dy_col = 320
    u_acc_col = 336
    du_acc_col = 352
    dq_acc_col = 368
    dk_decay_col = 384
    dk_inv_col = 400
    dk_restore_col = 416
    y_or_neg_beta_dy_col = 432
    du_input_col = 440
    qraw_col = 448                  # four stages, eight columns each
    kraw_col = 480                  # four stages, eight columns each
    assert kraw_col + 32 == 512

    # Persistent descriptor arrays are ten contiguous n_desc-entry arrays.
    desc_q = descriptor_array(0); desc_k = descriptor_array(1)
    desc_v = descriptor_array(2); desc_gate = descriptor_array(3)
    desc_do = descriptor_array(4); desc_dq = descriptor_array(5)
    desc_dk = descriptor_array(6); desc_dv = descriptor_array(7)
    desc_dgate = descriptor_array(8); desc_checkpoint = descriptor_array(9)

    def wait_edge(edge, stage, phase):
        wait(edge, stage, phase)
        # instruction_selection:
        # mbarrier.try_wait.parity.acquire.cta.shared::cta.b64;
        # extent: polling loop for one stage

    def arrive_edge(edge, stage):
        arrive(edge, stage)
        # instruction_selection: mbarrier.arrive.shared.b64; extent: one arrival

    def scheduler_publish_at_tile_entry(cursor_phase1, current):
        if DYN_SCHED:
            wait_edge("scheduler_done", cursor_phase1.stage, cursor_phase1.phase)
            if elected_lane():
                old = atomic_add(sched_counter[0], 1)
                # instruction_selection: atom.global.add.u32; extent: elected lane
                copy_r2s(grid_x + old,
                         SCHED_BASE + cursor_phase1.stage * 4)
                # instruction_selection: st.shared.u32; extent: scalar
            warp_sync(FULL_MASK)
            # instruction_selection: bar.warp.sync 0xffffffff; extent: warp 14
            copy_s2r(SCHED_BASE + cursor_phase1.stage * 4, next_tile)
            # instruction_selection: ld.shared.u32; extent: all warp lanes
            if elected_lane():
                arrive_edge("scheduler_ready", cursor_phase1.stage)
            return next_tile, advance(cursor_phase1)
        return current + grid_x, cursor_phase1

    def scheduler_consume_at_tile_exit(cursor_phase0, current):
        if DYN_SCHED:
            wait_edge("scheduler_ready", cursor_phase0.stage, cursor_phase0.phase)
            copy_s2r(SCHED_BASE + cursor_phase0.stage * 4, next_tile)
            # instruction_selection: ld.shared.u32; extent: scalar/all role lanes
            if elected_lane():
                arrive_edge("scheduler_done", cursor_phase0.stage)
            return next_tile, advance(cursor_phase0)
        return current + grid_x, cursor_phase0

    def decode_work(tile):
        row = copy_g2r(work_items[tile,0:8], eight_i32_regs)
        # instruction_selection: ld.global.v4.u32 pairs; extent: one work row
        return row  # batch, head, work-start/end, chunk-start/end, BOS/EOS

    # Descriptor array entries 0..9 already fold `bos` into their GMEM base.
    # TensorMap coordinates therefore remain sequence-relative.  Direct
    # scalar beta/dBeta addressing is the only path that adds `bos` again.
    def sequence_length(item):
        return item.eos - item.bos

    def chunk_start(chunk):
        return chunk * 16

    def token_valid(item, chunk, token):
        return chunk_start(chunk) + token < sequence_length(item)

    def global_token(item, chunk, token):
        return item.bos + chunk_start(chunk) + token

    def tensor_map_coordinate(head, chunk):
        return (0, head, chunk_start(chunk))

    def checkpoint_tensor_map_coordinate(head, chunk):
        return (0, 0, chunk, head)

    # ======================================================================
    # Source-order role selection
    # ======================================================================

    if warp == 14:
        # --------------------------------------------------------------
        # TMA + persistent scheduler role, source lines 1521..1696
        # anchor PTX 293, 380..629
        # --------------------------------------------------------------
        set_register_budget("decrease", 56)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32 56; extent: warp
        tile = cta
        sched = producer_cursor(stages=8, phase=1)
        raw = producer_cursor(stages=2, phase=1)
        state_cursor = producer_cursor(stages=1, phase=1)
        while tile < total_tiles:
            item = decode_work(tile)
            next_tile, sched = scheduler_publish_at_tile_entry(sched, tile)
            batch, head, wstart, wend, cstart, cend, bos, eos = item
            qh, kh, vh = grouped_input_heads(head, HQ, HK, HV, HO)
            if elected_lane():
                for desc in (
                    desc_q[item.batch], desc_k[item.batch], desc_v[item.batch],
                    desc_gate[item.batch], desc_do[item.batch],
                    desc_checkpoint[item.batch],
                ):
                    fence("tensormap_generic_acquire_gpu", desc)
                    # instruction_selection:
                    # fence.proxy.tensormap::generic.acquire.gpu;
                    # extent: six elected-lane descriptor fences
            for chunk in reverse_range(cend - 1, wstart - 1):
                input_coordinate_q = tensor_map_coordinate(qh, chunk)
                input_coordinate_k = tensor_map_coordinate(kh, chunk)
                input_coordinate_o = tensor_map_coordinate(head, chunk)
                input_coordinate_v = tensor_map_coordinate(vh, chunk)
                wait_edge("q_done", raw.stage, raw.phase)
                expect_bytes("q_ready", raw.stage, 4096)
                # instruction_selection: mbarrier.arrive.expect_tx.shared.b64;
                # extent: one raw stage transaction
                copy_g2s(desc_q[batch, input_coordinate_q],
                         q_raw_stage(raw.stage), True, "q_ready")
                # instruction_selection:
                # 2 x cp.async.bulk.tensor.3d.shared::cta.global.tile.
                # mbarrier::complete_tx::bytes; extent: two 64-channel subtiles

                wait_edge("k_done", raw.stage, raw.phase)
                expect_bytes("k_ready", raw.stage, 4096)
                # instruction_selection: mbarrier.arrive.expect_tx.shared.b64;
                # extent: one raw stage transaction
                copy_g2s(desc_k[batch, input_coordinate_k],
                         k_raw_stage(raw.stage), True, "k_ready")
                # instruction_selection: 2 x rank-3 TMA g2s; extent: two subtiles

                wait_edge("gate_done", raw.stage, raw.phase)
                expect_bytes("gate_ready", raw.stage, 8192)
                # instruction_selection: mbarrier.arrive.expect_tx.shared.b64;
                # extent: one raw stage transaction
                copy_g2s(desc_gate[batch, input_coordinate_o],
                         gate_raw_stage(raw.stage), True, "gate_ready")
                # instruction_selection: 4 x rank-3 TMA g2s; extent: four 32-channel subtiles

                wait_edge("do_done", raw.stage, raw.phase)
                expect_bytes("do_ready", raw.stage, 4096)
                # instruction_selection: mbarrier.arrive.expect_tx.shared.b64;
                # extent: one raw stage transaction
                copy_g2s(desc_do[batch, input_coordinate_o],
                         do_raw_stage(raw.stage), True, "do_ready")
                # instruction_selection: 2 x rank-3 TMA g2s; extent: two subtiles

                wait_edge("v_done", raw.stage, raw.phase)
                expect_bytes("v_ready", raw.stage, 4096)
                # instruction_selection: mbarrier.arrive.expect_tx.shared.b64;
                # extent: one raw stage transaction
                copy_g2s(desc_v[batch, input_coordinate_v],
                         v_raw_stage(raw.stage), True, "v_ready")
                # instruction_selection: 2 x rank-3 TMA g2s; extent: two subtiles

                if chunk >= FIRST_STATE_CHUNK:
                    wait_edge("state_cg0_done", state_cursor.stage, state_cursor.phase)
                    wait_edge("state_done", state_cursor.stage, state_cursor.phase)
                    expect_bytes("state_ready", state_cursor.stage, 32768)
                    # instruction_selection: mbarrier.arrive.expect_tx.shared.b64;
                    # extent: one state transaction
                    copy_g2s(desc_checkpoint[
                                 batch,
                                 checkpoint_tensor_map_coordinate(head, chunk)],
                             state_region(), True, "state_ready")
                    # instruction_selection: 2 x rank-4 TMA g2s;
                    # extent: two 64-row state subtiles
                    state_cursor = advance(state_cursor)
                raw = advance(raw)
            tile = next_tile

    elif warp == 12:
        # --------------------------------------------------------------
        # Super-MMA role, source lines 622..871; anchor PTX 650..1855
        # --------------------------------------------------------------
        set_register_budget("decrease", 56)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32 56; extent: warp
        tile = cta
        sched = consumer_cursor(stages=8, phase=0)
        dy_smem_cursor = consumer_cursor(stages=1, phase=0)
        chunk_serial_base = 0
        while tile < total_tiles:
            item = decode_work(tile)
            for chunk in reverse_range(item.cend - 1, item.wstart - 1):
                serial = chunk_serial_base + item.cend - 1 - chunk
                decay_stage = serial % 2
                inter_stage = serial % 2
                beta_stage_id = serial % 4
                wait_edge("t_inv_done", inter_stage, (serial//2 + 1) & 1)
                wait_edge("k_decay_inv", decay_stage, (serial//2) & 1)

                # KK = K_decay @ K_inverse^T.
                for k_block in range(8):
                    a_col = k_block * 16 + lhs_ldmatrix_col(lane)
                    b_col = k_block * 16 + rhs_ldmatrix_col(lane)
                    copy_s2r(k_decay_operand_bf16_byte(
                        decay_stage, lhs_ldmatrix_row(lane), a_col),
                        kk_lhs_regs[k_block])
                    copy_s2r(k_inverse_operand_bf16_byte(
                        decay_stage, rhs_ldmatrix_row(lane), b_col),
                        kk_rhs_regs[k_block])
                # instruction_selection: 16 x
                # ldmatrix.sync.aligned.m8n8.x4.shared.b16; extent: exact
                # lane-coordinate base for both operands over eight K blocks
                gemm(kk_regs, kk_lhs_regs, kk_rhs_regs)
                # instruction_selection:
                # mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32;
                # extent: eight K blocks, two m16n8 instructions each

                wait_edge("beta_ready", beta_stage_id, (serial//4) & 1)
                copy_s2r(beta_f32_byte(
                    beta_stage_id, super_row_lo(lane)), beta_rows.lo)
                copy_s2r(beta_f32_byte(
                    beta_stage_id, super_row_hi(lane)), beta_rows.hi)
                # instruction_selection: 2 x ld.shared.f32; extent: exact
                # row_lo=lane//4 and row_hi=row_lo+8 scalars
                arrive_edge("beta_done", beta_stage_id)
                # instruction_selection: mbarrier.arrive.shared.b64;
                # extent: all 32 super-warp threads
                select(strict_kk, strict_lower_mask, kk_regs, 0.0)
                # instruction_selection: selp.b32; extent: eight accumulator lanes
                mul(l_regs, strict_kk, beta_rows)
                # instruction_selection: mul.rn.f32; extent: eight lanes
                cast(l_bf16, l_regs)
                # instruction_selection: cvt.rn.bf16x2.f32; extent: four pairs
                cast(l_rounded_f32, l_bf16)
                # instruction_selection: cvt.f32.bf16; extent: eight lanes
                sub(tinv_regs, identity_fragment, l_rounded_f32)
                # instruction_selection: sub.rn.f32; extent: eight lanes

                # Three source power-series rounds:
                # Lpow <- round_bf16(Lpow @ Lpow)
                # Tinv <- round_bf16(Tinv) + round_bf16(Tinv) @ Lpow.
                lpow_bf16 = l_bf16
                for power_round in range(3):
                    gemm(lpow_square, lpow_bf16, move_matrix(lpow_bf16))
                    # instruction_selection: mma.sync m16n8k16 bf16;
                    # extent: two m16n8 instructions for one 16x16 product
                    cast(lpow_bf16, lpow_square)
                    # instruction_selection: cvt.rn.bf16x2.f32; extent: four pairs
                    cast(tinv_bf16, tinv_regs)
                    # instruction_selection: cvt.rn.bf16x2.f32; extent: four pairs
                    gemm(tinv_update, tinv_bf16, move_matrix(lpow_bf16))
                    # instruction_selection: mma.sync m16n8k16 bf16;
                    # extent: two m16n8 instructions for one 16x16 product
                    cast(tinv_rounded, tinv_bf16)
                    # instruction_selection: cvt.f32.bf16; extent: eight lanes
                    add(tinv_regs, tinv_rounded, tinv_update)
                    # instruction_selection: add.rn.f32x2; extent: four packed pairs
                cast(inv_bf16_regs, tinv_regs)
                # instruction_selection: cvt.rn.bf16x2.f32; extent: packed fragment
                copy_r2s(inv_bf16_regs,
                         stmatrix_intermediate_byte(inter_stage, 1, lane))
                # instruction_selection: stmatrix.sync.aligned.m8n8.x4.shared.b16;
                # extent: one 16x16 tile
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta; extent: warp
                arrive_edge("t_inv_ready", inter_stage)

                # dM = dY @ U^T; warp 15, not this warp, owns A/dA.
                wait_edge("dm_done", inter_stage, (serial//2 + 1) & 1)
                wait_edge("dy_smem", 0, dy_smem_cursor.phase)
                dy_smem_cursor = advance(dy_smem_cursor)
                for k_block in range(8):
                    a_col = k_block * 16 + lhs_ldmatrix_col(lane)
                    b_col = k_block * 16 + rhs_ldmatrix_col(lane)
                    copy_s2r(raw_bf16_byte(
                        DY_BASE, 0, lhs_ldmatrix_row(lane), a_col),
                        dy_regs[k_block])
                    copy_s2r(raw_bf16_byte(
                        U_BASE, 0, rhs_ldmatrix_row(lane), b_col),
                        u_regs[k_block])
                # instruction_selection: 16 x ldmatrix.sync.aligned.
                # m8n8.x4.shared.b16; extent: dY/U over eight value blocks
                gemm(dm_regs, dy_regs, u_regs)
                # instruction_selection: mma.sync m16n8k16 bf16;
                # extent: eight K blocks, two m16n8 instructions each
                select(dm_strict, strict_lower_mask, dm_regs, 0.0)
                # instruction_selection: selp.b32; extent: eight lanes
                mul(dm_strict, dm_strict, beta_rows)
                # instruction_selection: mul.rn.f32; extent: eight lanes
                cast(dm_bf16, dm_strict)
                # instruction_selection: cvt.rn.bf16x2.f32; extent: four pairs
                copy_r2s(dm_bf16,
                         stmatrix_intermediate_byte(inter_stage, 3, lane))
                # instruction_selection: stmatrix.sync...x4.shared.b16; extent: tile
                sub(negative_dm, 0.0, dm_strict)
                # instruction_selection: sub.rn.f32x2; extent: four packed pairs
                cast(negative_dm_bf16, negative_dm)
                # instruction_selection: cvt.rn.bf16x2.f32; extent: four pairs
                copy_r2s(negative_dm_bf16,
                         stmatrix_intermediate_byte(inter_stage, 4, lane))
                # instruction_selection: stmatrix.sync...x4.shared.b16; extent: tile
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta; extent: warp
                arrive_edge("dm_ready", inter_stage)

                # M term = -row_sum(strict(dM * KK)), one row per lane group.
                mul(m_product, select(strict_lower_mask, dm_regs, 0.0), kk_regs)
                # instruction_selection: mul.rn.f32; extent: eight lanes
                add(m_partial, m_product_even_lanes, m_product_odd_lanes)
                # instruction_selection: add.rn.f32x2; extent: packed pairs
                for delta in (1,2):
                    shuffle_xor(m_other, m_partial, delta)
                    # instruction_selection: shfl.sync.bfly.b32;
                    # extent: two fixed warp steps
                    add(m_partial, m_partial, m_other)
                    # instruction_selection: add.rn.f32; extent: two fixed steps
                if lane % 4 == 0:
                    sub(m_negative, 0.0, m_partial)
                    # instruction_selection: neg.f32/sub.rn.f32; extent: two scalars
                    copy_r2s(m_negative,
                             beta_matrix_f32_byte(super_row_lo(lane)))
                    copy_r2s(m_negative,
                             beta_matrix_f32_byte(super_row_hi(lane)))
                    # instruction_selection: st.shared.f32; extent: two scalars
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta; extent: warp
                arrive_edge("dbeta_matrix", 0)
            chunk_serial_base += item.cend - item.wstart
            tile, sched = scheduler_consume_at_tile_exit(sched, tile)

    elif warp == 13:
        # --------------------------------------------------------------
        # tcgen05 issuer, source lines 873..1519; anchor PTX 1886..3302
        # --------------------------------------------------------------
        set_register_budget("decrease", 56)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32 56; extent: warp
        allocate_tmem_warp_level(TMEM_MAILBOX_BASE, columns=512)
        # instruction_selection:
        # tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32;
        # extent: converged warp-level operation, not lane-predicated
        barrier(3, 416)
        # instruction_selection: barrier.sync 3, 416; extent: all TMEM users

        tile = cta
        sched = consumer_cursor(stages=8, phase=0)
        state_cursor = consumer_cursor(stages=1, phase=0)
        y_cursor = consumer_cursor(stages=1, phase=0)
        dstate_input_cursor = consumer_cursor(stages=1, phase=0)
        du_cursor = consumer_cursor(stages=1, phase=0)
        neg_dy_cursor = consumer_cursor(stages=1, phase=0)
        u_smem_cursor = consumer_cursor(stages=1, phase=0)
        dstate_smem_cursor = consumer_cursor(stages=1, phase=0)
        cg2_parts_done_cursor = consumer_cursor(stages=1, phase=1)
        dstate0_cursor = consumer_cursor(stages=1, phase=0)
        chunk_serial_base = 0
        while tile < total_tiles:
            item = decode_work(tile)
            for chunk in reverse_range(item.cend - 1, item.wstart - 1):
                serial = chunk_serial_base + item.cend - 1 - chunk
                has_dstate = (item.cend - 1 - chunk) > 0 or USE_DSTATE_IN
                raw_stage = serial % 2
                decay_stage = serial % 2
                intermediate_stage = serial % 2
                state_input_stage = serial % 2
                dv_stage = serial % 2
                raw_phase = (serial // 2) & 1
                decay_phase = (serial // 2) & 1
                inter_phase = (serial // 2) & 1
                state_input_phase = (serial // 2) & 1
                dv_ready_phase = (serial // 2) & 1
                # Each logical GEMM expands only to its repeated tcgen05.mma
                # instruction family.  Publication is a separate operation.
                wait_edge("k_decay_inv", decay_stage, decay_phase)
                if chunk >= FIRST_STATE_CHUNK:
                    wait_edge("state_ready", state_cursor.stage, state_cursor.phase)
                    for k_phase in range(8):
                        gemm(tmem_cell(tmem_base, tmem_row, 0, 320),
                                   state_smem_subtile(k_phase),
                                   k_decay_smem_subtile(decay_stage, k_phase),
                                   accumulate=k_phase > 0)
                    # instruction_selection: 8 x
                    # tcgen05.mma.cta_group::1.kind::f16; extent: GEMM 1
                    commit("state_k_acc")
                    commit("state_done")
                    # instruction_selection: one delayed tcgen05.commit to two
                    # mbarriers after all eight issues
                    state_cursor = advance(state_cursor)

                wait_edge("dqk_acc_done", cg2_parts_done_cursor.stage,
                          cg2_parts_done_cursor.phase)
                cg2_parts_done_cursor = advance(cg2_parts_done_cursor)
                wait_edge("state_input_ready", state_input_stage,
                          state_input_phase)
                wait_edge("do_ready", raw_stage, raw_phase)
                if chunk >= FIRST_STATE_CHUNK:
                    for k_phase in range(8):
                        gemm(tmem_cell(tmem_base, tmem_row, 0, 368),
                                   tmem_state_input_subtile(
                                       tmem_base, tmem_row,
                                       state_input_stage, k_phase),
                                   do_smem_lead16_subtile(raw_stage, k_phase),
                                   accumulate=k_phase > 0)
                    # instruction_selection: 8 x tcgen05.mma...kind::f16;
                    # extent: GEMM 2, no commit at this point

                wait_edge("q_decay_k_restore", decay_stage, decay_phase)
                if has_dstate:
                    wait_edge("dstate_input", dstate_input_cursor.stage,
                              dstate_input_cursor.phase)
                    for k_phase in range(8):
                        gemm(tmem_cell(tmem_base, tmem_row, 0, 352),
                                   tmem_dstate_input_subtile(
                                       tmem_base, tmem_row, k_phase),
                                   k_restore_smem_subtile(
                                       decay_stage, k_phase),
                                   accumulate=k_phase > 0)
                    # instruction_selection: 8 x tcgen05.mma...kind::f16;
                    # extent: GEMM 3, no commit at this point
                    for n_stripe in range(8):
                        gemm(tmem_cell(tmem_base, tmem_row, 0,
                                       n_stripe * 16),
                             tmem_dstate_input_subtile(
                                 tmem_base, tmem_row, n_stripe),
                             state_diag_smem_subtile(
                                 decay_stage, n_stripe),
                             accumulate=False)
                        # instruction_selection: one independent
                        # tcgen05.mma...kind::f16 per N stripe; extent: GEMM 4
                    dstate_input_cursor = advance(dstate_input_cursor)

                wait_edge("a_ready", intermediate_stage, inter_phase)
                gemm(tmem_cell(tmem_base, tmem_row, 0, 352),
                     do_smem_amajor_desc(raw_stage),
                     intermediate_smem_desc(intermediate_stage, 0),
                     accumulate=has_dstate)
                # instruction_selection: tcgen05.mma...kind::f16;
                # extent: one 16-K instruction for GEMM 5
                commit("du_acc")
                commit("a_done")
                # instruction_selection: one delayed tcgen05.commit carrying
                # exactly two arrivals after GEMMs 2..5

                gemm(tmem_cell(tmem_base, tmem_row, 0, 0),
                     do_smem_amajor_desc(raw_stage),
                     q_decay_smem_transpose_desc(decay_stage),
                     accumulate=has_dstate)
                # instruction_selection: tcgen05.mma...kind::f16;
                # extent: one 16-K instruction for GEMM 6

                wait_edge("t_inv_ready", intermediate_stage, inter_phase)
                wait_edge("y_input", y_cursor.stage, y_cursor.phase)
                gemm(tmem_cell(tmem_base, tmem_row, 0, 336),
                     tmem_y_input(tmem_base, tmem_row),
                     intermediate_smem_desc(intermediate_stage, 1))
                # instruction_selection: tcgen05.mma...kind::f16;
                # extent: one 16-K instruction for GEMM 7
                commit("u_acc")
                y_cursor = advance(y_cursor)

                wait_edge("du_input", du_cursor.stage, du_cursor.phase)
                gemm(tmem_cell(tmem_base, tmem_row, 0, 320),
                     tmem_du_input(tmem_base, tmem_row),
                     intermediate_smem_desc(intermediate_stage, 1))
                # instruction_selection: tcgen05.mma...kind::f16;
                # extent: one 16-K instruction for GEMM 8
                commit("dy_acc")
                commit("t_inv_done")
                du_cursor = advance(du_cursor)

                wait_edge("u_smem", u_smem_cursor.stage, u_smem_cursor.phase)
                if has_dstate:
                    wait_edge("dstate_smem", dstate_smem_cursor.stage,
                              dstate_smem_cursor.phase)
                    for k_phase in range(8):
                        gemm(tmem_cell(tmem_base, tmem_row, 0, 416),
                                   dstate_smem_alt_desc(k_phase),
                                   u_smem_lead16_subtile(k_phase),
                                   accumulate=k_phase > 0)
                    # instruction_selection: 8 x tcgen05.mma...kind::f16;
                    # extent: GEMM 9; both operands come from SMEM
                    commit("dk_restore_acc")
                    commit("dstate_smem_done")
                    dstate_smem_cursor = advance(dstate_smem_cursor)
                u_smem_cursor = advance(u_smem_cursor)

                wait_edge("neg_beta_dy_input", neg_dy_cursor.stage,
                          neg_dy_cursor.phase)
                gemm(tmem_cell(tmem_base, tmem_row, 0, 0),
                     tmem_negative_beta_dy_input(tmem_base, tmem_row),
                     k_decay_smem_transpose_desc(decay_stage),
                     accumulate=True)
                # instruction_selection: tcgen05.mma...kind::f16;
                # extent: one 16-K instruction for GEMM 10
                commit("dstate_acc")
                neg_dy_cursor = advance(neg_dy_cursor)

                wait_edge("da_ready", intermediate_stage, inter_phase)
                gemm(tmem_cell(tmem_base, tmem_row, 0, 400),
                     q_decay_smem_transpose_desc(decay_stage),
                     intermediate_smem_desc(intermediate_stage, 2))
                # instruction_selection: tcgen05.mma...kind::f16;
                # extent: one runtime issue for GEMM 11; no commit

                if chunk >= FIRST_STATE_CHUNK:
                    gemm(tmem_cell(tmem_base, tmem_row, 0, 368),
                         k_inverse_smem_amajor_desc(decay_stage),
                         intermediate_smem_desc(intermediate_stage, 2),
                         accumulate=True)
                else:
                    gemm(tmem_cell(tmem_base, tmem_row, 0, 368),
                         k_inverse_smem_amajor_desc(decay_stage),
                         intermediate_smem_desc(intermediate_stage, 2),
                         accumulate=False)
                # instruction_selection: tcgen05.mma...kind::f16;
                # extent: GEMM 12 has two mutually-exclusive static predicate
                # sites but exactly one runtime issue
                commit("dq_acc")
                commit("da_done")

                wait_edge("dv_store_ready", dv_stage, dv_ready_phase)
                if chunk >= FIRST_STATE_CHUNK:
                    for k_phase in range(8):
                        gemm(tmem_cell(tmem_base, tmem_row, 0, 384),
                                   tmem_state_input_subtile(
                                       tmem_base, tmem_row,
                                       state_input_stage, k_phase),
                                   dv_smem_lead16_subtile(
                                       dv_stage, k_phase),
                                   accumulate=k_phase > 0)
                    # instruction_selection: 8 x tcgen05.mma...kind::f16;
                    # extent: GEMM 13, entering state in TMEM times sDv
                commit("state_input_done")
                # The commit is unconditional even when GEMM 13 is predicated
                # away, so CG0 can reuse the state-input TMEM columns.

                wait_edge("dm_ready", intermediate_stage, inter_phase)
                gemm(tmem_cell(tmem_base, tmem_row, 0, 400),
                     k_decay_smem_transpose_desc(decay_stage),
                     intermediate_smem_desc(intermediate_stage, 4),
                     accumulate=True)
                # instruction_selection: tcgen05.mma...kind::f16;
                # extent: one 16-K instruction for GEMM 14
                commit("dk_inv_acc")

                if chunk >= FIRST_STATE_CHUNK:
                    gemm(tmem_cell(tmem_base, tmem_row, 0, 384),
                         k_inverse_smem_amajor_desc(decay_stage),
                         intermediate_smem_desc(intermediate_stage, 3),
                         accumulate=True)
                else:
                    gemm(tmem_cell(tmem_base, tmem_row, 0, 384),
                         k_inverse_smem_amajor_desc(decay_stage),
                         intermediate_smem_desc(intermediate_stage, 3),
                         accumulate=False)
                # instruction_selection: tcgen05.mma...kind::f16;
                # extent: GEMM 15 has two mutually-exclusive static predicate
                # sites but exactly one runtime issue
                commit("dk_decay_acc")
                commit("dm_done")
                commit("decay_done")

            wait_edge("dstate0_stored", dstate0_cursor.stage,
                      dstate0_cursor.phase)
            dstate0_cursor = advance(dstate0_cursor)
            chunk_serial_base += item.cend - item.wstart
            tile, sched = scheduler_consume_at_tile_exit(sched, tile)
        wait_edge("tmem_done", 0, phase=0)
        relinquish_tmem_permit_warp_level()
        # instruction_selection:
        # tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
        # extent: converged warp-level operation
        deallocate_tmem_warp_level(tmem_allocated, columns=512)
        # instruction_selection:
        # tcgen05.dealloc.cta_group::1.sync.aligned.b32;
        # extent: converged warp-level operation

    elif warp == 15:
        # --------------------------------------------------------------
        # Epilogue/store role, source lines 302..620; anchor PTX 3309..4254
        # --------------------------------------------------------------
        set_register_budget("decrease", 56)
        # instruction_selection: setmaxnreg.dec.sync.aligned.u32 56; extent: warp
        tile = cta
        sched = consumer_cursor(stages=8, phase=0)
        u_smem_cursor = consumer_cursor(stages=1, phase=0)
        dq_store_cursor = consumer_cursor(stages=1, phase=0)
        dk_store_cursor = consumer_cursor(stages=1, phase=0)
        dgate_store_cursor = consumer_cursor(stages=1, phase=0)
        dv_store_cursor = consumer_cursor(stages=2, phase=0)
        chunk_serial_base = 0
        while tile < total_tiles:
            item = decode_work(tile)
            if elected_lane():
                fence("tensormap_generic_acquire_gpu", desc_dq[item.batch])
                fence("tensormap_generic_acquire_gpu", desc_dk[item.batch])
                fence("tensormap_generic_acquire_gpu", desc_dv[item.batch])
                fence("tensormap_generic_acquire_gpu", desc_dgate[item.batch])
                # Exactly four output descriptor acquires, once per work item.
            has_pending_output = False
            pending_output_batch = 0
            pending_output_coordinate = (0, 0, 0)
            pending_output_writes = False
            for chunk in reverse_range(item.cend - 1, item.wstart - 1):
                serial = chunk_serial_base + item.cend - 1 - chunk
                raw_stage = serial % 2
                decay_stage = serial % 2
                intermediate_stage = serial % 2
                raw_phase = (serial // 2) & 1
                decay_phase = (serial // 2) & 1

                wait_edge("a_done", intermediate_stage,
                          ((serial // 2) + 1) & 1)
                wait_edge("q_decay_k_restore", decay_stage, decay_phase)
                for k_block in range(8):
                    a_col = k_block * 16 + lhs_ldmatrix_col(lane)
                    b_col = k_block * 16 + rhs_ldmatrix_col(lane)
                    copy_s2r(q_decay_operand_bf16_byte(
                        decay_stage, lhs_ldmatrix_row(lane), a_col),
                        q_regs[k_block])
                    copy_s2r(k_inverse_operand_bf16_byte(
                        decay_stage, rhs_ldmatrix_row(lane), b_col),
                        k_regs[k_block])
                # instruction_selection: 16 x ldmatrix.sync.aligned.
                # m8n8.x4.shared.b16; extent: exact A operand lane bases
                fill(a_regs, 0.0)
                for k_block in range(8):
                    gemm(a_regs, q_regs[k_block], transpose(k_regs[k_block]),
                         accumulate=k_block > 0)
                # instruction_selection: 16 x mma.sync m16n8k16 bf16;
                # extent: A = Q-decay @ K-inverse^T over eight K blocks
                for accum_lane in range(8):
                    select(a_regs[accum_lane],
                           tril_inclusive_mask_bit(accum_lane),
                           a_regs[accum_lane], 0.0)
                cast(a_bf16x2, a_regs)
                # instruction_selection: cvt.rn.bf16x2.f32; extent: four pairs
                copy_r2s(a_bf16x2,
                         stmatrix_intermediate_byte(
                             intermediate_stage, 0, lane))
                # instruction_selection: stmatrix.sync.aligned.m8n8.x4.shared.b16;
                # extent: one 16x16 A tile
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta; extent: warp
                arrive_edge("a_ready", intermediate_stage)

                wait_edge("u_smem", u_smem_cursor.stage,
                          u_smem_cursor.phase)
                for k_block in range(8):
                    a_col = k_block * 16 + lhs_ldmatrix_col(lane)
                    b_col = k_block * 16 + rhs_ldmatrix_col(lane)
                    copy_s2r(do_raw_bf16_byte(
                        raw_stage, lhs_ldmatrix_row(lane), a_col),
                        do_regs[k_block])
                    copy_s2r(raw_bf16_byte(
                        U_BASE, 0, rhs_ldmatrix_row(lane), b_col),
                        u_regs[k_block])
                # instruction_selection: 16 x ldmatrix.sync.aligned.
                # m8n8.x4.shared.b16; extent: exact dO/U lane bases
                fill(da_regs, 0.0)
                for k_block in range(8):
                    gemm(da_regs, do_regs[k_block], transpose(u_regs[k_block]),
                         accumulate=k_block > 0)
                # instruction_selection: 16 x mma.sync m16n8k16 bf16;
                # extent: dA = dO @ U^T over eight K blocks
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta; extent: warp
                arrive_edge("do_done", raw_stage)
                u_smem_cursor = advance(u_smem_cursor)
                for accum_lane in range(8):
                    select(da_regs[accum_lane],
                           tril_inclusive_mask_bit(accum_lane),
                           da_regs[accum_lane], 0.0)
                wait_edge("da_done", intermediate_stage,
                          ((serial // 2) + 1) & 1)
                cast(da_bf16x2, da_regs)
                # instruction_selection: cvt.rn.bf16x2.f32; extent: four pairs
                copy_r2s(da_bf16x2,
                         stmatrix_intermediate_byte(
                             intermediate_stage, 2, lane))
                # instruction_selection: stmatrix.sync.aligned.m8n8.x4.shared.b16;
                # extent: one 16x16 dA tile
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta; extent: warp
                arrive_edge("da_ready", intermediate_stage)

                if has_pending_output:
                    wait_edge("dq_store_ready", dq_store_cursor.stage,
                              dq_store_cursor.phase)
                    if pending_output_writes:
                        copy_s2g(dq_output_bf16_byte(0, 0),
                                 desc_dq[pending_output_batch,
                                         pending_output_coordinate], True)
                    # instruction_selection: 2 x
                    # cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group;
                    # extent: two 64-channel subtiles
                        commit_directional_copy_group("dq")
                    wait_edge("dk_store_ready", dk_store_cursor.stage,
                              dk_store_cursor.phase)
                    if pending_output_writes:
                        copy_s2g(dk_output_bf16_byte(0, 0),
                                 desc_dk[pending_output_batch,
                                         pending_output_coordinate], True)
                    # instruction_selection: 2 x rank-3 TMA s2g; extent: two subtiles
                        commit_directional_copy_group("dk")
                    wait_edge("dgate_store_ready", dgate_store_cursor.stage,
                              dgate_store_cursor.phase)
                    if pending_output_writes:
                        copy_s2g(dgate_output_f32_byte(0, 0),
                                 desc_dgate[pending_output_batch,
                                            pending_output_coordinate], True)
                    # instruction_selection: 4 x rank-3 TMA s2g; extent: four subtiles
                        commit_directional_copy_group("dgate")
                    wait_edge("dv_store_ready", dv_store_cursor.stage,
                              dv_store_cursor.phase)
                    if pending_output_writes:
                        copy_s2g(dv_output_bf16_byte(
                                     dv_store_cursor.stage, 0, 0),
                                 desc_dv[pending_output_batch,
                                         pending_output_coordinate], True)
                    # instruction_selection: 2 x rank-3 TMA s2g; extent: two subtiles
                        commit_directional_copy_group("dv")
                    # instruction_selection: one cp.async.bulk.commit_group
                    # immediately after each of dq/dk/dgate/dv
                    wait_directional_copy_group(3)
                    arrive_edge("dq_store_done", dq_store_cursor.stage)
                    wait_directional_copy_group(2)
                    arrive_edge("dk_store_done", dk_store_cursor.stage)
                    wait_directional_copy_group(1)
                    arrive_edge("dgate_store_done", dgate_store_cursor.stage)
                    wait_directional_copy_group(0)
                    arrive_edge("dv_store_done", dv_store_cursor.stage)
                    # instruction_selection: cp.async.bulk.wait_group.read
                    # ladder 3,2,1,0; release exactly the completed stage
                    dq_store_cursor = advance(dq_store_cursor)
                    dk_store_cursor = advance(dk_store_cursor)
                    dgate_store_cursor = advance(dgate_store_cursor)
                    dv_store_cursor = advance(dv_store_cursor)

                has_pending_output = True
                pending_output_batch = item.batch
                pending_output_coordinate = tensor_map_coordinate(
                    item.head, chunk)
                pending_output_writes = chunk < item.wend
            if has_pending_output:
                wait_edge("dq_store_ready", dq_store_cursor.stage,
                          dq_store_cursor.phase)
                if pending_output_writes:
                    copy_s2g(dq_output_bf16_byte(0, 0),
                             desc_dq[pending_output_batch,
                                     pending_output_coordinate], True)
                    commit_directional_copy_group("dq")
                wait_edge("dk_store_ready", dk_store_cursor.stage,
                          dk_store_cursor.phase)
                if pending_output_writes:
                    copy_s2g(dk_output_bf16_byte(0, 0),
                             desc_dk[pending_output_batch,
                                     pending_output_coordinate], True)
                    commit_directional_copy_group("dk")
                wait_edge("dgate_store_ready", dgate_store_cursor.stage,
                          dgate_store_cursor.phase)
                if pending_output_writes:
                    copy_s2g(dgate_output_f32_byte(0, 0),
                             desc_dgate[pending_output_batch,
                                        pending_output_coordinate], True)
                    commit_directional_copy_group("dgate")
                wait_edge("dv_store_ready", dv_store_cursor.stage,
                          dv_store_cursor.phase)
                if pending_output_writes:
                    copy_s2g(dv_output_bf16_byte(
                                 dv_store_cursor.stage, 0, 0),
                             desc_dv[pending_output_batch,
                                     pending_output_coordinate], True)
                    commit_directional_copy_group("dv")
                wait_directional_copy_group(3)
                arrive_edge("dq_store_done", dq_store_cursor.stage)
                wait_directional_copy_group(2)
                arrive_edge("dk_store_done", dk_store_cursor.stage)
                wait_directional_copy_group(1)
                arrive_edge("dgate_store_done", dgate_store_cursor.stage)
                wait_directional_copy_group(0)
                arrive_edge("dv_store_done", dv_store_cursor.stage)
                # Same explicit four-commit/3,2,1,0 ladder flushes the tail.
                dq_store_cursor = advance(dq_store_cursor)
                dk_store_cursor = advance(dk_store_cursor)
                dgate_store_cursor = advance(dgate_store_cursor)
                dv_store_cursor = advance(dv_store_cursor)
            chunk_serial_base += item.cend - item.wstart
            tile, sched = scheduler_consume_at_tile_exit(sched, tile)

    elif 0 <= warp <= 3:
        # --------------------------------------------------------------
        # CG0, source lines 1698..2107; anchor PTX 4299..6299
        # --------------------------------------------------------------
        set_register_budget("increase", 144)
        # instruction_selection: setmaxnreg.inc.sync.aligned.u32 144; extent: warpgroup
        barrier(3, 416)
        # instruction_selection: barrier.sync 3, 416; extent: TMEM users
        cg0_warp = warp
        prefix_dim = cg0_warp * 32 + lane
        row_group_start = cg0_warp * 4
        lane_row_group = lane // 8
        lane_in_row_group = lane - lane_row_group * 8
        decay_row = row_group_start + lane_row_group
        value_dim = cg0_warp * 32 + lane
        tile = cta
        sched = consumer_cursor(stages=8, phase=0)
        beta_done_cursor = consumer_cursor(stages=4, phase=1)
        qk_done_cursor = consumer_cursor(stages=4, phase=1)
        state_input_done_cursor = consumer_cursor(stages=2, phase=1)
        state_input_cg2_done_cursor = consumer_cursor(stages=2, phase=1)
        state_cursor = consumer_cursor(stages=1, phase=0)
        chunk_serial_base = 0
        while tile < total_tiles:
            item = decode_work(tile)
            num_compute_chunks = item.cend - item.wstart
            safe_a_exp = 1.0
            safe_dt_bias = 0.0
            if SAFE_GATE and num_compute_chunks > 0:
                copy_g2r(a_log[item.head], a_reg)
                exp2(safe_a_exp, a_reg * LOG2_E, ftz=True)
                # instruction_selection: one ex2.approx.ftz.f32 per nonempty
                # work item, outside the chunk loop
                copy_g2r(dt_bias[item.head, prefix_dim], safe_dt_bias)
                # instruction_selection: one ld.global.f32 per channel and
                # nonempty work item
            for chunk in reverse_range(item.cend - 1, item.wstart - 1):
                serial = chunk_serial_base + item.cend - 1 - chunk
                raw_stage = serial % 2
                decay_stage = serial % 2
                qk_stage = serial % 4
                state_input_stage = serial % 2
                beta_stage_id = serial % 4
                raw_phase = (serial // 2) & 1
                decay_phase = (serial // 2) & 1
                state_input_phase = (serial // 2) & 1
                for token in range(16):
                    valid_rows[token] = token_valid(item, chunk, token)

                # Only CG0 warp 0 produces beta.  It cannot overwrite the
                # stage until all 32 super-MMA threads and all 128 CG1 threads
                # have arrived at beta_done (expected count 160).
                if warp == 0:
                    wait_edge("beta_done", beta_stage_id,
                              beta_done_cursor.phase)
                    if lane < 16:
                        fill(beta_regs, 0.0)
                        if valid_rows[lane]:
                            copy_g2r(beta[global_token(item, chunk, lane),
                                          item.head], beta_regs)
                            # instruction_selection: ld.global.f32 or ld.global.b16
                            if BETA_SIGMOID:
                                tanh(beta_regs, beta_regs * 0.5)
                                add(beta_regs, beta_regs * 0.5, 0.5)
                                cast(beta_bf16, beta_regs)
                                cast(beta_regs, beta_bf16)
                                # instruction_selection: cvt.rn.bf16.f32 then
                                # cvt.f32.bf16 before the FP32 shared store
                        # Invalid tail beta stays exactly zero; sigmoid is not
                        # evaluated for the invalid lane.
                        copy_r2s(beta_regs,
                                 beta_f32_byte(beta_stage_id, lane))
                        # instruction_selection: st.shared.f32; extent: lanes 0..15
                    arrive_edge("beta_ready", beta_stage_id)
                    beta_done_cursor = advance(beta_done_cursor)

                wait_edge("gate_ready", raw_stage, raw_phase)
                wait_edge("q_ready", raw_stage, raw_phase)
                wait_edge("k_ready", raw_stage, raw_phase)
                for token in range(16):
                    copy_s2r(gate_raw_f32_byte(
                        raw_stage, token, prefix_dim), gate_regs[token])
                # instruction_selection: ld.shared.f32; vectorized as
                # ld.shared.v4.b32; extent: 16 rows for one key channel
                if SAFE_GATE:
                    tanh(sigmoid_regs,
                         safe_a_exp * (gate_regs + safe_dt_bias) * 0.5)
                    # instruction_selection: tanh.approx.f32; extent: fragment
                    add(sigmoid_regs, sigmoid_regs * 0.5, 0.5)
                    # instruction_selection: fma.rn.f32; extent: fragment
                    mul(log_decay, sigmoid_regs, GATE_SCALE_LOG2)
                    # instruction_selection: mul.rn.f32; extent: fragment
                    select(log_decay, valid_rows, log_decay, 0.0)
                else:
                    select(gate_ln, valid_rows, gate_regs, 0.0)
                    mul(log_decay, gate_ln, LOG2_E)
                    # instruction_selection: selp.f32 + mul.rn.f32;
                    # extent: raw natural-log gate, with only tail rows zeroed

                prefix_acc = 0.0
                for row_pair in range(8):
                    row0 = 2 * row_pair
                    row1 = row0 + 1
                    add_packed_f32x2(pair_vec,
                                     (prefix_acc, log_decay[row0]),
                                     (log_decay[row0], log_decay[row1]))
                    prefix0 = pair_vec[0]
                    row_pair_sum = pair_vec[1]
                    add(prefix1, prefix_acc, row_pair_sum)
                    assign(log_decay[row0], prefix0)
                    assign(log_decay[row1], prefix1)
                    assign(prefix_acc, prefix1)
                # instruction_selection: strict-order add.rn.f32x2 plus scalar
                # add.rn.f32; extent: eight packed row pairs, no shuffle
                exp2(gate_exp, log_decay, ftz=True)
                # instruction_selection: ex2.approx.ftz.f32; extent: fragment
                wait_edge("decay_done", decay_stage,
                          ((serial // 2) + 1) & 1)
                for token in range(16):
                    copy_r2s(gate_exp[token], gate_raw_f32_byte(
                        raw_stage, token, prefix_dim))
                # instruction_selection: st.shared.f32; vectorized as
                # st.shared.v4.b32; extent: 16 rows for one key channel
                diag_block = prefix_dim // 16
                diag_coord = prefix_dim - diag_block * 16
                cast(diag_bf16, gate_exp[15])
                copy_r2s(diag_bf16, state_diag_bf16_byte(
                    decay_stage, diag_block, diag_coord, diag_coord))
                # instruction_selection: cvt.rn.bf16.f32 + st.shared.u16;
                # extent: one diagonal element per CG0 thread

                for token in range(16):
                    copy_s2r(q_raw_bf16_byte(raw_stage, token, prefix_dim),
                             q_raw_column[token])
                    copy_s2r(k_raw_bf16_byte(raw_stage, token, prefix_dim),
                             k_raw_column[token])
                # instruction_selection: ld.shared.u16; extent: exactly 16 Q
                # and 16 K scalars selected by the BF16 xor mapping

                # Publish the raw, unnormalized BF16 Q/K packs before any L2
                # arithmetic.  CG2 consumes these exact values.
                wait_edge("qk_raw_done", qk_stage, qk_done_cursor.phase)
                cast(q_raw_words[0:8], q_raw_column[0:16])
                cast(k_raw_words[0:8], k_raw_column[0:16])
                # instruction_selection: cvt.rn.bf16x2.f32/mov.b32;
                # extent: eight packed token pairs for each operand
                copy_r2t(q_raw_words,
                         cg0_qraw_tmem_cell(
                             tmem_base, tmem_row, warp, qk_stage))
                copy_r2t(k_raw_words,
                         cg0_kraw_tmem_cell(
                             tmem_base, tmem_row, warp, qk_stage))
                # instruction_selection: 2 x
                # tcgen05.st.sync.aligned.32x32b.x16.b32
                wait_tmem_store()
                # instruction_selection: tcgen05.wait::st.sync.aligned
                arrive_edge("qk_raw_ready", qk_stage)
                qk_done_cursor = advance(qk_done_cursor)
                barrier(1, 128)
                # instruction_selection: barrier.sync 1,128; all raw Q/K
                # TMEM stores are issued before operand-fragment loads proceed

                for dim_half in range(2):
                    dim_base = dim_half * 64 + lane_in_row_group * 8
                    for dim_offset in range(8):
                        channel_i = dim_base + dim_offset
                        copy_s2r(q_raw_bf16_byte(
                            raw_stage, decay_row, channel_i),
                            q_regs[dim_half, dim_offset])
                        copy_s2r(k_raw_bf16_byte(
                            raw_stage, decay_row, channel_i),
                            k_regs[dim_half, dim_offset])
                # instruction_selection: ld.shared.v4.b32; extent: two
                # contiguous eight-BF16 fragments from each Q/K raw stage
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta; extent: warpgroup
                arrive_edge("q_done", raw_stage)
                arrive_edge("k_done", raw_stage)

                q_inv_norm = 1.0
                k_inv_norm = 1.0
                if L2NORM:
                    qk0_lo = opaque_f32_zero()
                    qk0_hi = opaque_f32_zero()
                    qk1_lo = opaque_f32_zero()
                    qk1_hi = opaque_f32_zero()
                    for dim_half in range(2):
                        for dim_offset in range(8):
                            q_value = cast_f32(q_regs[dim_half, dim_offset])
                            k_value = cast_f32(k_regs[dim_half, dim_offset])
                            if dim_offset % 2 == 0:
                                ffma2(qk0_lo, qk0_hi,
                                      q_value, k_value,
                                      q_value, k_value,
                                      qk0_lo, qk0_hi)
                            else:
                                ffma2(qk1_lo, qk1_hi,
                                      q_value, k_value,
                                      q_value, k_value,
                                      qk1_lo, qk1_hi)
                    # instruction_selection: paired mad.rn.f32/FFMA2 ordering;
                    # extent: both 64-channel halves and even/odd accumulators
                    add(q_sum, qk0_lo, qk1_lo)
                    add(k_sum, qk0_hi, qk1_hi)
                    for step in (4,2,1):
                        shuffle_xor(q_other, q_sum, step, width=8)
                        shuffle_xor(k_other, k_sum, step, width=8)
                        add(q_sum, q_sum, q_other)
                        add(k_sum, k_sum, k_other)
                    # instruction_selection: three fixed 8-lane
                    # shfl.sync.bfly.b32 reductions, steps 4,2,1
                    rsqrt(q_inv_norm,
                          max(q_sum, L2_NORM_EPS * L2_NORM_EPS), ftz=True)
                    # instruction_selection: rsqrt.approx.ftz.f32; extent: scalar
                    rsqrt(k_inv_norm,
                          max(k_sum, L2_NORM_EPS * L2_NORM_EPS), ftz=True)
                    # instruction_selection: rsqrt.approx.ftz.f32; extent: scalar
                    if lane_in_row_group == 0:
                        copy_r2s(q_inv_norm, norm_f32_byte(
                            qk_stage, 0, decay_row))
                        copy_r2s(k_inv_norm, norm_f32_byte(
                            qk_stage, 1, decay_row))
                        # instruction_selection: two st.shared.f32 by the
                        # elected lane in each eight-lane row group
                    # Only inverse norms, not normalized vectors, enter SMEM.

                for dim_half in range(2):
                    dim_base = dim_half * 64 + lane_in_row_group * 8
                    for dim_offset in range(8):
                        channel_i = dim_base + dim_offset
                        copy_s2r(gate_raw_f32_byte(
                            raw_stage, decay_row, channel_i),
                            gate_exp_regs[dim_half, dim_offset])
                        copy_s2r(gate_raw_f32_byte(
                            raw_stage, 15, channel_i),
                            gate_last_regs[dim_half, dim_offset])
                # instruction_selection: ld.shared.v4.b32; extent: four
                # aligned FP32 groups per dimension half for current/last row
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta; extent: warpgroup
                arrive_edge("gate_done", raw_stage)

                # K-decay/K-inverse producer phase.  Every factor is rounded
                # to a BF16 pair before packed multiplication.
                for pair in range(8):
                    mul(k_norm_f32x2, raw_k_f32x2[pair], k_inv_norm)
                    cast(k_base_bf16x2, k_norm_f32x2)
                    cast(gate_bf16x2, gate_exp_regs[pair])
                    rcp(gate_rcp_f32x2, gate_exp_regs[pair], approx=True,
                        ftz=True)
                    cast(gate_rcp_bf16x2, gate_rcp_f32x2)
                    mul_bf16x2(k_decay_bf16x2, k_base_bf16x2,
                               gate_bf16x2)
                    mul_bf16x2(k_inverse_bf16x2, k_base_bf16x2,
                               gate_rcp_bf16x2)
                    assign(saved_k_inverse_bf16x2[pair],
                           k_inverse_bf16x2)
                    # instruction_selection: cvt.rn.bf16x2.f32,
                    # rcp.approx.ftz.f32, mul.bf16x2; extent: eight pairs
                    token_pair = pair % 4
                    dim_half = pair // 4
                    channel0 = dim_half * 64 + lane_in_row_group * 8
                    channel0 += token_pair * 2
                    copy_r2s(k_decay_bf16x2,
                             k_decay_operand_bf16_byte(
                                 decay_stage, decay_row, channel0))
                    copy_r2s(k_inverse_bf16x2,
                             k_inverse_operand_bf16_byte(
                                 decay_stage, decay_row, channel0))
                    # instruction_selection: st.shared.v4.b32 across each
                    # eight-element vector group
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta; extent: warpgroup
                arrive_edge("k_decay_inv", decay_stage)

                # Q-decay/K-restore is a second, ordered producer phase.
                for pair in range(8):
                    mul(q_scaled_f32x2, raw_q_f32x2[pair],
                        q_inv_norm * scale)
                    cast(q_base_bf16x2, q_scaled_f32x2)
                    cast(gate_bf16x2, gate_exp_regs[pair])
                    cast(gate_last_bf16x2, gate_last_regs[pair])
                    mul_bf16x2(q_decay_bf16x2, q_base_bf16x2,
                               gate_bf16x2)
                    mul_bf16x2(k_restore_bf16x2,
                               saved_k_inverse_bf16x2[pair],
                               gate_last_bf16x2)
                    # instruction_selection: cvt.rn.bf16x2.f32 and
                    # mul.bf16x2; K-restore consumes the rounded K-inverse
                    token_pair = pair % 4
                    dim_half = pair // 4
                    channel0 = dim_half * 64 + lane_in_row_group * 8
                    channel0 += token_pair * 2
                    copy_r2s(q_decay_bf16x2,
                             q_decay_operand_bf16_byte(
                                 decay_stage, decay_row, channel0))
                    copy_r2s(k_restore_bf16x2,
                             k_restore_operand_bf16_byte(
                                 decay_stage, decay_row, channel0))
                    # instruction_selection: st.shared.v4.b32 vector groups
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta; extent: warpgroup
                arrive_edge("q_decay_k_restore", decay_stage)

                wait_edge("state_input_done", state_input_stage,
                          state_input_done_cursor.phase)
                wait_edge("state_input_cg2_done", state_input_stage,
                          state_input_cg2_done_cursor.phase)
                if chunk >= FIRST_STATE_CHUNK:
                    wait_edge("state_ready", state_cursor.stage,
                              state_cursor.phase)
                    for value_half in range(2):
                        for value_group8 in range(8):
                            for element in range(8):
                                key_row = value_half * 64
                                key_row += value_group8 * 8 + element
                                copy_s2r(state_bf16_byte(
                                    key_row, value_dim), state_fragment[element])
                            # instruction_selection: ld.shared.v4.b32; extent:
                            # one eight-BF16 vector selected by state mapping
                            copy_r2t(state_fragment,
                                     cg0_state_tmem_cell(
                                         tmem_base, tmem_row, warp,
                                         state_input_stage, value_half,
                                         value_group8))
                            # instruction_selection:
                            # tcgen05.st.sync.aligned.32x32b.x4.b32;
                            # extent: 16 vector stores per thread
                    wait_tmem_store()
                    # instruction_selection: tcgen05.wait::st.sync.aligned
                    arrive_edge("state_cg0_done", 0)
                    state_cursor = advance(state_cursor)
                arrive_edge("state_input_ready", state_input_stage)
                # This ready edge is unconditional when state is absent.
                state_input_done_cursor = advance(state_input_done_cursor)
                state_input_cg2_done_cursor = advance(state_input_cg2_done_cursor)
            chunk_serial_base += item.cend - item.wstart
            tile, sched = scheduler_consume_at_tile_exit(sched, tile)

    elif 8 <= warp <= 11:
        # --------------------------------------------------------------
        # CG2, source lines 2546..2823; anchor PTX 6300..8329
        # --------------------------------------------------------------
        set_register_budget("increase", 144)
        # instruction_selection: setmaxnreg.inc.sync.aligned.u32 144; extent: warpgroup
        barrier(3,416)
        # instruction_selection: barrier.sync 3, 416; extent: TMEM users
        cg2_warp = warp - 8
        channel = cg2_warp * 32 + lane
        cg2_linear_thread = channel
        tile = cta
        sched = consumer_cursor(stages=8, phase=0)
        dq_cursor = consumer_cursor(stages=1, phase=0)
        dk_decay_cursor = consumer_cursor(stages=1, phase=0)
        dk_inv_cursor = consumer_cursor(stages=1, phase=0)
        dk_restore_cursor = consumer_cursor(stages=1, phase=0)
        dstate_smem_cursor = consumer_cursor(stages=1, phase=0)
        dq_output_cursor = consumer_cursor(stages=1, phase=1)
        dk_output_cursor = consumer_cursor(stages=1, phase=1)
        dgate_output_cursor = consumer_cursor(stages=1, phase=1)
        chunk_serial_base = 0
        while tile < total_tiles:
            item = decode_work(tile)
            for chunk in reverse_range(item.cend - 1, item.wstart - 1):
                serial = chunk_serial_base + item.cend - 1 - chunk
                local_rev = item.cend - 1 - chunk
                raw_stage = serial % 2
                decay_stage = serial % 2
                qk_stage = serial % 4
                state_input_stage = serial % 2
                decay_phase = (serial // 2) & 1
                qk_phase = (serial // 4) & 1
                state_input_phase = (serial // 2) & 1
                has_dstate = local_rev > 0 or USE_DSTATE_IN

                # CG0's decay publication is the visibility guard for the
                # exponential gate values that overwrite the raw gate stage.
                wait_edge("k_decay_inv", decay_stage, decay_phase)
                for token in range(16):
                    copy_s2r(gate_exp[token],
                             gate_raw_f32_byte(raw_stage, token, channel))
                    # instruction_selection: ld.shared.f32; extent: 16 rows
                gate_last = gate_exp[15]
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta; extent: warpgroup
                arrive_edge("gate_done", raw_stage)

                wait_edge("qk_raw_ready", qk_stage, qk_phase)
                wait_edge("state_input_ready", state_input_stage,
                          state_input_phase)
                dgate_last_value = 0.0
                if has_dstate:
                    wait_edge("dstate_smem", dstate_smem_cursor.stage,
                              dstate_smem_cursor.phase)
                    for plane in range(2):
                        for row_half in range(2):
                            copy_t2r(cg2_tmem_channel_cell(
                                         tmem_base, tmem_row, warp,
                                         192 + state_input_stage * 64
                                         + plane * 32 + row_half * 16),
                                     state_words)
                            # instruction_selection:
                            # tcgen05.ld.sync.aligned.32x32b.x16.b32
                            fill(hacc[0:8], opaque_f32_zero())
                            for pair in range(16):
                                copy_s2r(dstate_pair,
                                         dstate_alt_transpose_bf16_byte(
                                             plane * 64 + row_half * 32
                                             + 2 * pair, channel))
                                copy_s2r(dstate_pair_hi,
                                         dstate_alt_transpose_bf16_byte(
                                             plane * 64 + row_half * 32
                                             + 2 * pair + 1, channel))
                                # instruction_selection: 2 x ld.shared.u16;
                                # extent: one dState pair per state word
                                state_pair = unpack_bf16x2(
                                    state_words[pair])
                                fma(hacc[(2 * pair) % 8],
                                    dstate_pair, state_pair.lo,
                                    hacc[(2 * pair) % 8])
                                fma(hacc[(2 * pair + 1) % 8],
                                    dstate_pair_hi, state_pair.hi,
                                    hacc[(2 * pair + 1) % 8])
                                # instruction_selection: mad.rn.f32;
                                # extent: 32 paired dState/state products
                            fadd2(pa0, pb0, hacc[0], hacc[2],
                                  hacc[4], hacc[6])
                            fadd2(pa1, pb1, hacc[1], hacc[3],
                                  hacc[5], hacc[7])
                            fadd2(part_a, part_b, pa0, pb0, pa1, pb1)
                            add(dgate_last_value, dgate_last_value,
                                part_a + part_b)
                            # instruction_selection: add.rn.f32x2 followed
                            # by scalar add.rn.f32; extent: source tree
                    arrive_edge("dstate_smem_cg2_done", 0)
                    dstate_smem_cursor = advance(dstate_smem_cursor)
                arrive_edge("state_input_cg2_done", state_input_stage)

                fill(dk_regs, 0.0)
                fill(dgate_last_kdot_parts, 0.0)
                if has_dstate:
                    wait_edge("dk_restore_acc", dk_restore_cursor.stage,
                              dk_restore_cursor.phase)
                    copy_t2r(cg2_tmem_channel_cell(
                        tmem_base, tmem_row, warp, 416),
                        dk_restore_part)
                    copy_t2r(cg2_tmem_channel_cell(
                        tmem_base, tmem_row, warp,
                        480 + qk_stage * 8), raw_k_words)
                    # instruction_selection: 2 x
                    # tcgen05.ld.sync.aligned.32x32b.x16.b32
                    for token in range(16):
                        rcp(gate_rcp, gate_exp[token], approx=True, ftz=True)
                        mul(dk_hat, dk_restore_part[token],
                            gate_last * gate_rcp)
                        # instruction_selection: rcp.approx.ftz.f32 +
                        # mul.rn.f32; extent: token
                        add(dk_regs[token], dk_regs[token], dk_hat)
                        raw_k = unpack_bf16(raw_k_words, token)
                        if L2NORM:
                            mul(raw_k, raw_k,
                                copy_s2r(norm_f32_byte(
                                    qk_stage, 1, token)))
                        fma(dgate_last_kdot_parts[token % 4], raw_k,
                            dk_hat, dgate_last_kdot_parts[token % 4])
                    # instruction_selection: cvt.f32.bf16, mul.rn.f32,
                    # mad.rn.f32; extent: 16 token channels
                    dk_restore_cursor = advance(dk_restore_cursor)

                wait_edge("dq_acc", dq_cursor.stage, dq_cursor.phase)
                copy_t2r(cg2_tmem_channel_cell(
                    tmem_base, tmem_row, warp, 368), dq_part)
                # instruction_selection:
                # tcgen05.ld.sync.aligned.32x32b.x16.b32
                for token in range(16):
                    mul(dq_regs[token], dq_part[token],
                        gate_exp[token] * scale)
                # instruction_selection: mul.rn.f32; extent: 16 tokens
                dq_cursor = advance(dq_cursor)

                wait_edge("dk_inv_acc", dk_inv_cursor.stage,
                          dk_inv_cursor.phase)
                copy_t2r(cg2_tmem_channel_cell(
                    tmem_base, tmem_row, warp, 400), dk_inv_part)
                # instruction_selection:
                # tcgen05.ld.sync.aligned.32x32b.x16.b32
                for token in range(16):
                    rcp(gate_rcp, gate_exp[token], approx=True, ftz=True)
                    add(dk_regs[token], dk_regs[token],
                        dk_inv_part[token] * gate_rcp)
                    # instruction_selection: rcp.approx.ftz.f32 + fma/add
                dk_inv_cursor = advance(dk_inv_cursor)

                wait_edge("dk_decay_acc", dk_decay_cursor.stage,
                          dk_decay_cursor.phase)
                copy_t2r(cg2_tmem_channel_cell(
                    tmem_base, tmem_row, warp, 384), dk_decay_part)
                # instruction_selection:
                # tcgen05.ld.sync.aligned.32x32b.x16.b32
                for token in range(16):
                    mul(dgate_regs[token], -gate_exp[token],
                        dk_decay_part[token])
                    add(dk_regs[token], dk_regs[token], dgate_regs[token])
                # instruction_selection: neg.f32, mul.rn.f32, add.rn.f32;
                # extent: 16 tokens
                dk_decay_cursor = advance(dk_decay_cursor)
                wait_tmem_load()
                # instruction_selection: tcgen05.wait::ld.sync.aligned
                arrive_edge("dqk_acc_done", 0)

                copy_t2r(cg2_tmem_channel_cell(
                    tmem_base, tmem_row, warp,
                    448 + qk_stage * 8), raw_q_words)
                copy_t2r(cg2_tmem_channel_cell(
                    tmem_base, tmem_row, warp,
                    480 + qk_stage * 8), raw_k_words)
                # instruction_selection: 2 x
                # tcgen05.ld.sync.aligned.32x32b.x8.b32
                for token in range(16):
                    raw_q = unpack_bf16(raw_q_words, token)
                    raw_k = unpack_bf16(raw_k_words, token)
                    if L2NORM:
                        mul(raw_q, raw_q,
                            copy_s2r(norm_f32_byte(qk_stage, 0, token)))
                        mul(raw_k, raw_k,
                            copy_s2r(norm_f32_byte(qk_stage, 1, token)))
                    fma(dgate_regs[token], raw_q, dq_regs[token],
                        raw_k * (2.0 * dgate_regs[token] - dk_regs[token]))
                    # instruction_selection: cvt.f32.bf16, fma.rn.f32,
                    # sub.rn.f32; extent: each token
                add(dgate_regs[15], dgate_regs[15],
                    sum(dgate_last_kdot_parts))

                if L2NORM:
                    # Q projection, kept distinct from K to preserve both
                    # TMEM column and norm-offset immediates.
                    for half in range(2):
                        copy_t2r(cg2_tmem_channel_cell(
                            tmem_base, tmem_row, warp,
                            448 + qk_stage * 8 + half * 4), q_half_words)
                        for pair in range(4):
                            token = half * 8 + pair * 2
                            q_pair = unpack_bf16x2(q_half_words[pair])
                            mul(q_dot[token], dq_regs[token], q_pair.lo)
                            mul(q_dot[token + 1], dq_regs[token + 1], q_pair.hi)
                            mul(q_dot[token], q_dot[token],
                                copy_s2r(norm_f32_byte(qk_stage, 0, token)))
                            mul(q_dot[token + 1], q_dot[token + 1],
                                copy_s2r(norm_f32_byte(
                                    qk_stage, 0, token + 1)))
                    # instruction_selection: 2 x tcgen05.ld.sync.aligned.
                    # 32x32b.x4.b32, cvt.f32.bf16, mul.rn.f32x2;
                    # extent: 16 Q tokens
                    for step in (1,2,4,8,16):
                        for token_pair in range(8):
                            shuffle_xor(q_other.lo,
                                        q_dot[2 * token_pair], step)
                            shuffle_xor(q_other.hi,
                                        q_dot[2 * token_pair + 1], step)
                            add_packed_f32x2(q_dot[2 * token_pair:
                                                   2 * token_pair + 2],
                                             q_other)
                    # instruction_selection: shfl.sync.bfly.b32 steps
                    # 1,2,4,8,16 plus add.rn.f32x2 for all token pairs
                    if lane == 0:
                        for token in range(16):
                            copy_r2s(q_dot[token], reduction1_f32_byte(
                                cg2_warp, token))
                    barrier(2, 128)
                    for half in range(2):
                        copy_t2r(cg2_tmem_channel_cell(
                            tmem_base, tmem_row, warp,
                            448 + qk_stage * 8 + half * 4), q_half_words)
                        for pair in range(4):
                            token = half * 8 + pair * 2
                            sum(q_dot_all.lo, [copy_s2r(
                                reduction1_f32_byte(w, token))
                                for w in range(4)])
                            sum(q_dot_all.hi, [copy_s2r(
                                reduction1_f32_byte(w, token + 1))
                                for w in range(4)])
                            q_pair = unpack_bf16x2(q_half_words[pair])
                            q_norm_lo = copy_s2r(norm_f32_byte(
                                qk_stage, 0, token))
                            q_norm_hi = copy_s2r(norm_f32_byte(
                                qk_stage, 0, token + 1))
                            mul(q_unit.lo, q_pair.lo, q_norm_lo)
                            mul(q_unit.hi, q_pair.hi, q_norm_hi)
                            mul(dq_regs[token],
                                dq_regs[token] - q_unit.lo * q_dot_all.lo,
                                q_norm_lo)
                            mul(dq_regs[token + 1],
                                dq_regs[token + 1]
                                - q_unit.hi * q_dot_all.hi, q_norm_hi)
                    # instruction_selection: 2 x tcgen05.ld .x4,
                    # ld.shared.f32, add/mul/sub.rn.f32x2; extent: Q projection
                    barrier(2, 128)

                    # K projection is the same fixed instruction sequence at
                    # the distinct K-raw columns and K-norm offset.
                    for half in range(2):
                        copy_t2r(cg2_tmem_channel_cell(
                            tmem_base, tmem_row, warp,
                            480 + qk_stage * 8 + half * 4), k_half_words)
                        for pair in range(4):
                            token = half * 8 + pair * 2
                            k_pair = unpack_bf16x2(k_half_words[pair])
                            mul(k_dot[token], dk_regs[token], k_pair.lo)
                            mul(k_dot[token + 1], dk_regs[token + 1], k_pair.hi)
                            mul(k_dot[token], k_dot[token],
                                copy_s2r(norm_f32_byte(qk_stage, 1, token)))
                            mul(k_dot[token + 1], k_dot[token + 1],
                                copy_s2r(norm_f32_byte(
                                    qk_stage, 1, token + 1)))
                    # instruction_selection: 2 x tcgen05.ld.sync.aligned.
                    # 32x32b.x4.b32, cvt.f32.bf16, mul.rn.f32x2;
                    # extent: 16 K tokens
                    for step in (1,2,4,8,16):
                        for token_pair in range(8):
                            shuffle_xor(k_other.lo,
                                        k_dot[2 * token_pair], step)
                            shuffle_xor(k_other.hi,
                                        k_dot[2 * token_pair + 1], step)
                            add_packed_f32x2(k_dot[2 * token_pair:
                                                   2 * token_pair + 2],
                                             k_other)
                    if lane == 0:
                        for token in range(16):
                            copy_r2s(k_dot[token], reduction1_f32_byte(
                                cg2_warp, token))
                    barrier(2, 128)
                    for half in range(2):
                        copy_t2r(cg2_tmem_channel_cell(
                            tmem_base, tmem_row, warp,
                            480 + qk_stage * 8 + half * 4), k_half_words)
                        for pair in range(4):
                            token = half * 8 + pair * 2
                            sum(k_dot_all.lo, [copy_s2r(
                                reduction1_f32_byte(w, token))
                                for w in range(4)])
                            sum(k_dot_all.hi, [copy_s2r(
                                reduction1_f32_byte(w, token + 1))
                                for w in range(4)])
                            k_pair = unpack_bf16x2(k_half_words[pair])
                            k_norm_lo = copy_s2r(norm_f32_byte(
                                qk_stage, 1, token))
                            k_norm_hi = copy_s2r(norm_f32_byte(
                                qk_stage, 1, token + 1))
                            mul(k_unit.lo, k_pair.lo, k_norm_lo)
                            mul(k_unit.hi, k_pair.hi, k_norm_hi)
                            mul(dk_regs[token],
                                dk_regs[token] - k_unit.lo * k_dot_all.lo,
                                k_norm_lo)
                            mul(dk_regs[token + 1],
                                dk_regs[token + 1]
                                - k_unit.hi * k_dot_all.hi, k_norm_hi)
                    # instruction_selection: 2 x tcgen05.ld .x4,
                    # ld.shared.f32, add/mul/sub.rn.f32x2; extent: K projection
                    barrier(2, 128)

                wait_tmem_load()
                # instruction_selection: tcgen05.wait::ld.sync.aligned
                arrive_edge("qk_raw_done", qk_stage)

                wait_edge("dq_store_done", 0, dq_output_cursor.phase)
                wait_edge("dk_store_done", 0, dk_output_cursor.phase)
                for token in range(16):
                    cast(dq_bf16, dq_regs[token])
                    cast(dk_bf16, dk_regs[token])
                    # instruction_selection: cvt.rn.bf16.f32 for dQ and dK;
                    # extent: one scalar per token and key-channel thread
                    copy_r2s(dq_bf16, dq_output_bf16_byte(token, channel))
                    copy_r2s(dk_bf16, dk_output_bf16_byte(token, channel))
                    # instruction_selection: st.shared.u16; vectorized as
                    # st.shared.v4.b32 across each eight-channel group
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta; extent: warpgroup
                arrive_edge("dq_store_ready", 0)
                arrive_edge("dk_store_ready", 0)
                dq_output_cursor = advance(dq_output_cursor)
                dk_output_cursor = advance(dk_output_cursor)

                if has_dstate and chunk >= FIRST_STATE_CHUNK:
                    add(dgate_regs[15], dgate_regs[15],
                        gate_last * dgate_last_value)
                suffix = 0.0
                for reverse_token in range(16):
                    token = 15 - reverse_token
                    add(suffix, suffix, dgate_regs[token])
                    assign(dgate_regs[token], suffix)
                # Strict sequential register suffix; no shuffle and no
                # SAFE_GATE derivative is applied in this role.  Instruction
                # selection is 16 ordered add.rn.f32 operations.
                wait_edge("dgate_store_done", 0,
                          dgate_output_cursor.phase)
                for token in range(16):
                    copy_r2s(dgate_regs[token],
                             dgate_output_f32_byte(token, channel))
                    # instruction_selection: st.shared.f32; vectorized as
                    # st.shared.v4.b32 for each four-channel group
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta; extent: warpgroup
                arrive_edge("dgate_store_ready", 0)
                dgate_output_cursor = advance(dgate_output_cursor)
            chunk_serial_base += item.cend - item.wstart
            tile, sched = scheduler_consume_at_tile_exit(sched, tile)
        arrive_edge("tmem_done", 0)

    elif 4 <= warp <= 7:
        # --------------------------------------------------------------
        # CG1, source lines 2109..2544; anchor PTX 8330..end
        # --------------------------------------------------------------
        set_register_budget("increase", 168)
        # instruction_selection: setmaxnreg.inc.sync.aligned.u32 168; extent: warpgroup
        barrier(3,416)
        # instruction_selection: barrier.sync 3, 416; extent: TMEM users
        cg1_warp = warp - 4
        token_row_coord = (lane // 16) * 8 + (lane & 7)
        value_col_offset = ((lane // 8) & 1) * 8
        value_dim = cg1_warp * 32 + lane
        value_dim_base = cg1_warp * 32
        cg1_linear_thread = cg1_warp * 32 + lane
        tile = cta
        sched = consumer_cursor(stages=8, phase=0)
        raw_cursor = consumer_cursor(stages=2, phase=0)
        state_k_cursor = consumer_cursor(stages=1, phase=0)
        u_acc_cursor = consumer_cursor(stages=1, phase=0)
        du_acc_cursor = consumer_cursor(stages=1, phase=0)
        dy_acc_cursor = consumer_cursor(stages=1, phase=0)
        dstate_acc_cursor = consumer_cursor(stages=1, phase=0)
        dstate_reuse_cursor = consumer_cursor(stages=1, phase=1)
        dbeta_m_cursor = consumer_cursor(stages=1, phase=0)
        dv_reuse_cursor = consumer_cursor(stages=2, phase=1)
        chunk_serial_base = 0
        while tile < total_tiles:
            item = decode_work(tile)
            num_compute_chunks = item.cend - item.wstart
            if USE_DSTATE_IN and num_compute_chunks > 0:
                wait_edge("dstate_smem_done", 0, dstate_reuse_cursor.phase)
                wait_edge("dstate_smem_cg2_done", 0,
                          dstate_reuse_cursor.phase)
                seed_true = item.cend == item.num_chunks
                for key in range(128):
                    copy_g2r(d_final_state[
                        item.batch, item.head, value_dim, key],
                        seed_regs[key])
                # instruction_selection: ld.global.L1::no_allocate.v4.b32;
                # extent: 32x128 stripe
                select(dstate_regs, seed_true, seed_regs, 0.0)
                for key_sub16 in range(8):
                    copy_r2t(dstate_regs[key_sub16],
                             cg1_dstate_tmem_cell(
                                 tmem_base, tmem_row, warp,
                                 key_sub16 * 16), dtype="f32")
                    cast(dstate_bf16_words,
                         dstate_regs[key_sub16])
                    copy_r2t(dstate_bf16_words,
                             cg1_dstate_tmem_cell(
                                 tmem_base, tmem_row, warp,
                                 128 + key_sub16 * 8), dtype="bf16")
                # instruction_selection: 8 x tcgen05.st.sync.aligned.
                # 32x32b.x16.b32 FP32 plus 8 x .x8.b32 packed BF16
                wait_tmem_store()
                # instruction_selection: tcgen05.wait::st.sync.aligned
                arrive_edge("dstate_input", 0)
                for key_sub16 in range(8):
                    copy_t2r(cg1_dstate_tmem_cell(
                        tmem_base, tmem_row, warp,
                        128 + key_sub16 * 8), dstate_bf16_words)
                    for half8 in range(2):
                        copy_r2s(dstate_bf16_words[half8],
                                 dstate_direct_bf16_byte(
                                     value_dim,
                                     key_sub16 * 16 + half8 * 8))
                # instruction_selection: tcgen05.ld.sync.aligned.32x32b.x8.b32
                # plus st.shared.v4.b32; extent: all 128 keys for one value row
                # Re-read the published TMEM input; do not repack registers.
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta; extent: warpgroup
                arrive_edge("dstate_smem", 0)
                dstate_reuse_cursor = advance(dstate_reuse_cursor)

            for chunk in reverse_range(item.cend - 1, item.wstart - 1):
                serial = chunk_serial_base + item.cend - 1 - chunk
                local_rev = item.cend - 1 - chunk
                raw_stage = serial % 2
                beta_stage_id = serial % 4
                dv_stage_id = serial % 2
                raw_phase = (serial // 2) & 1
                beta_phase = (serial // 4) & 1
                has_state = chunk >= FIRST_STATE_CHUNK
                has_dstate = local_rev > 0 or USE_DSTATE_IN
                wait_edge("v_ready", raw_stage, raw_phase)
                copy_s2r(v_raw_bf16_byte(
                    raw_stage, token_row_coord,
                    value_dim_base + value_col_offset), v_regs.lo)
                copy_s2r(v_raw_bf16_byte(
                    raw_stage, token_row_coord,
                    value_dim_base + 16 + value_col_offset), v_regs.hi)
                # instruction_selection: 2 x
                # ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16; extent:
                # exact COL-layout lane bases for the two value halves
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta; extent: warpgroup
                arrive_edge("v_done", raw_stage)
                wait_edge("beta_ready", beta_stage_id, beta_phase)
                for half in range(2):
                    token0 = ((half * 4 + (lane & 3)) ^ 4) * 2
                    copy_s2r(beta_f32_byte(beta_stage_id, token0),
                             beta_regs[half].lo)
                    copy_s2r(beta_f32_byte(beta_stage_id, token0 + 1),
                             beta_regs[half].hi)
                # instruction_selection: four ld.shared.f32; extent: exact
                # lane-permuted beta pairs
                cast(beta_bf16x2, beta_regs)
                # instruction_selection: cvt.rn.bf16x2.f32; extent: token pairs
                if has_state:
                    wait_edge("state_k_acc", state_k_cursor.stage,
                              state_k_cursor.phase)
                    copy_t2r(cg1_projection_tmem_cell(
                        tmem_base, tmem_row, warp, 0, 320),
                        state_k_regs.lo)
                    copy_t2r(cg1_projection_tmem_cell(
                        tmem_base, tmem_row, warp, 1, 320),
                        state_k_regs.hi)
                    # instruction_selection: 2 x
                    # tcgen05.ld.sync.aligned.16x256b.x2.b32
                    cast(state_k_bf16x2, state_k_regs)
                    # instruction_selection: cvt.rn.bf16x2.f32
                    sub_bf16x2(diff_bf16x2, v_bf16x2, state_k_bf16x2)
                    # instruction_selection: sub.bf16x2
                    state_k_cursor = advance(state_k_cursor)
                else:
                    assign(diff_bf16x2, v_bf16x2)
                mul_bf16x2(y_bf16x2, beta_bf16x2, diff_bf16x2)
                # instruction_selection: mul.bf16x2; extent: packed token pairs
                copy_r2t(y_bf16x2.lo,
                         cg1_tmem_cell_lo(tmem_base, tmem_row, 432))
                copy_r2t(y_bf16x2.hi,
                         cg1_tmem_cell_hi(tmem_base, tmem_row, 432))
                # instruction_selection: 2 x
                # tcgen05.st.sync.aligned.16x128b.x4.b32
                wait_tmem_store()
                # instruction_selection: tcgen05.wait::st.sync.aligned
                arrive_edge("y_input", 0)

                wait_edge("du_acc", du_acc_cursor.stage, du_acc_cursor.phase)
                copy_t2r(cg1_projection_tmem_cell(
                    tmem_base, tmem_row, warp, 0, 352), du_regs.lo)
                copy_t2r(cg1_projection_tmem_cell(
                    tmem_base, tmem_row, warp, 1, 352), du_regs.hi)
                # instruction_selection: 2 x
                # tcgen05.ld.sync.aligned.16x256b.x2.b32
                cast(du_bf16, du_regs)
                # instruction_selection: cvt.rn.bf16x2.f32; extent: packed fragment
                copy_r2t(du_bf16.lo,
                         cg1_tmem_cell_lo(tmem_base, tmem_row, 440))
                copy_r2t(du_bf16.hi,
                         cg1_tmem_cell_hi(tmem_base, tmem_row, 440))
                # instruction_selection: 2 x
                # tcgen05.st.sync.aligned.16x128b.x4.b32
                wait_tmem_store()
                # instruction_selection: tcgen05.wait::st.sync.aligned
                arrive_edge("du_input", 0)
                du_acc_cursor = advance(du_acc_cursor)

                wait_edge("u_acc", u_acc_cursor.stage, u_acc_cursor.phase)
                copy_t2r(cg1_projection_tmem_cell(
                    tmem_base, tmem_row, warp, 0, 336), u_regs.lo)
                copy_t2r(cg1_projection_tmem_cell(
                    tmem_base, tmem_row, warp, 1, 336), u_regs.hi)
                # instruction_selection: 2 x
                # tcgen05.ld.sync.aligned.16x256b.x2.b32
                cast(u_bf16, u_regs)
                # instruction_selection: cvt.rn.bf16x2.f32; extent: packed fragment
                copy_r2s(u_bf16.lo, raw_bf16_byte(
                    U_BASE, 0, token_row_coord,
                    value_dim_base + value_col_offset))
                copy_r2s(u_bf16.hi, raw_bf16_byte(
                    U_BASE, 0, token_row_coord,
                    value_dim_base + 16 + value_col_offset))
                # instruction_selection: 2 x stmatrix.sync.aligned.
                # m8n8.x4.trans.shared.b16; extent: exact lane bases
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta; extent: warpgroup
                arrive_edge("u_smem", 0)
                u_acc_cursor = advance(u_acc_cursor)

                wait_edge("dy_acc", dy_acc_cursor.stage, dy_acc_cursor.phase)
                copy_t2r(cg1_projection_tmem_cell(
                    tmem_base, tmem_row, warp, 0, 320), dy_regs.lo)
                copy_t2r(cg1_projection_tmem_cell(
                    tmem_base, tmem_row, warp, 1, 320), dy_regs.hi)
                # instruction_selection: 2 x
                # tcgen05.ld.sync.aligned.16x256b.x2.b32
                cast(dy_bf16, dy_regs)
                # instruction_selection: cvt.rn.bf16x2.f32; extent: packed fragment
                copy_r2s(dy_bf16.lo, raw_bf16_byte(
                    DY_BASE, 0, token_row_coord,
                    value_dim_base + value_col_offset))
                copy_r2s(dy_bf16.hi, raw_bf16_byte(
                    DY_BASE, 0, token_row_coord,
                    value_dim_base + 16 + value_col_offset))
                # instruction_selection: 2 x stmatrix.sync.aligned.
                # m8n8.x4.trans.shared.b16; extent: exact lane bases
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta; extent: warpgroup
                arrive_edge("dy_smem", 0)
                dy_acc_cursor = advance(dy_acc_cursor)
                beta_c0 = copy_s2r(beta_f32_byte(
                    beta_stage_id, (lane % 4) * 2))
                beta_c1 = copy_s2r(beta_f32_byte(
                    beta_stage_id, (lane % 4) * 2 + 1))
                beta_c8 = copy_s2r(beta_f32_byte(
                    beta_stage_id, (lane % 4) * 2 + 8))
                beta_c9 = copy_s2r(beta_f32_byte(
                    beta_stage_id, (lane % 4) * 2 + 9))
                # instruction_selection: four ld.shared.f32; extent: exact
                # beta scalars latched before beta_done
                beta_self = 0.0
                if BETA_SIGMOID and cg1_linear_thread < 16:
                    beta_self = copy_s2r(
                        beta_f32_byte(beta_stage_id, cg1_linear_thread))
                arrive_edge("beta_done", beta_stage_id)
                # 128 CG1 arrivals join the super warp's 32 arrivals.

                for pair in range(4):
                    beta_lo = select(pair * 2 >= 4, beta_c8, beta_c0)
                    beta_hi = select(pair * 2 >= 4, beta_c9, beta_c1)
                    fmul2(beta_dy.lo[pair],
                          dy_regs.lo[2 * pair], dy_regs.lo[2 * pair + 1],
                          beta_lo, beta_hi)
                    fmul2(beta_dy.hi[pair],
                          dy_regs.hi[2 * pair], dy_regs.hi[2 * pair + 1],
                          beta_lo, beta_hi)
                    sub(neg_beta_dy.lo[pair], 0.0, beta_dy.lo[pair])
                    sub(neg_beta_dy.hi[pair], 0.0, beta_dy.hi[pair])
                # sDv/GEMM13 consume the same values.  Instruction selection:
                # mul.rn.f32x2 and neg.f32; extent: both value halves
                cast(neg_beta_dy_bf16, neg_beta_dy)
                # instruction_selection: cvt.rn.bf16x2.f32; extent: fragment
                copy_r2t(neg_beta_dy_bf16.lo,
                         cg1_tmem_cell_lo(tmem_base, tmem_row, 432))
                copy_r2t(neg_beta_dy_bf16.hi,
                         cg1_tmem_cell_hi(tmem_base, tmem_row, 432))
                # instruction_selection: 2 x
                # tcgen05.st.sync.aligned.16x128b.x4.b32
                wait_tmem_store()
                # instruction_selection: tcgen05.wait::st.sync.aligned
                arrive_edge("neg_beta_dy_input", 0)

                # dBeta value term is accumulated before waiting for the dV
                # output stage, exactly preserving source issue order.
                fill(dbeta_value_parts[0:4], 0.0)
                for pair in range(4):
                    unpack_bf16x2(diff0_lo, diff0_hi,
                                  diff_bf16x2.lo[pair])
                    unpack_bf16x2(diff1_lo, diff1_hi,
                                  diff_bf16x2.hi[pair])
                    token_lo = 2 * (pair // 2)
                    token_hi = token_lo + 1
                    ffma2(tmp_lo, tmp_hi,
                          dy_regs.lo[2 * pair],
                          dy_regs.lo[2 * pair + 1],
                          diff0_lo, diff0_hi,
                          dbeta_value_parts[token_lo],
                          dbeta_value_parts[token_hi])
                    ffma2(dbeta_value_parts[token_lo],
                          dbeta_value_parts[token_hi],
                          dy_regs.hi[2 * pair],
                          dy_regs.hi[2 * pair + 1],
                          diff1_lo, diff1_hi, tmp_lo, tmp_hi)
                # instruction_selection: eight ordered paired mad.rn.f32
                # (FFMA2), first low then high value half for each pair

                # dV is exactly beta*dY; it is both the global output and the
                # SMEM B operand for logical GEMM 13.
                wait_edge("dv_store_done", dv_stage_id,
                          dv_reuse_cursor.phase)
                cast(dv_bf16, beta_dy)
                # instruction_selection: cvt.rn.bf16x2.f32; extent: packed fragment
                copy_r2s(dv_bf16.lo, dv_output_bf16_byte(
                    dv_stage_id, token_row_coord,
                    value_dim_base + value_col_offset))
                copy_r2s(dv_bf16.hi, dv_output_bf16_byte(
                    dv_stage_id, token_row_coord,
                    value_dim_base + 16 + value_col_offset))
                # instruction_selection: 2 x stmatrix.sync.aligned.
                # m8n8.x4.trans.shared.b16; extent: exact dV lane bases
                fence("async_shared_cta")
                # instruction_selection: fence.proxy.async.shared::cta; extent: warpgroup
                arrive_edge("dv_store_ready", dv_stage_id)
                dv_reuse_cursor = advance(dv_reuse_cursor)

                # dBeta is reduced across the four warps.
                barrier(4, 128)  # dbeta sync 1: all value terms complete
                wait_edge("dbeta_matrix", dbeta_m_cursor.stage,
                          dbeta_m_cursor.phase)
                for step in (4,8,16):
                    shuffle_xor(dbeta_other, dbeta_value_parts, step)
                    add(dbeta_value_parts, dbeta_value_parts, dbeta_other)
                # instruction_selection: shfl.sync.bfly.b32 + add.rn.f32;
                # extent: steps 4,8,16 for all four token accumulators
                if lane < 4:
                    for part in range(4):
                        token = (lane % 4) * 2 + (part % 2)
                        token += 8 * (part // 2)
                        copy_r2s(dbeta_value_parts[part],
                                 reduction0_f32_byte(cg1_warp, token))
                    # instruction_selection: four st.shared.f32 from each
                    # lane 0..3 in every CG1 warp
                barrier(4, 128)  # dbeta sync 2: shared partials visible
                if cg1_linear_thread < 16:
                    sum(dbeta_value, [copy_s2r(reduction0_f32_byte(
                        producer_warp, cg1_linear_thread))
                        for producer_warp in range(4)])
                    add(dbeta_out, dbeta_value,
                        copy_s2r(beta_matrix_f32_byte(
                            cg1_linear_thread)))
                    # instruction_selection: four ld.shared.f32 reductions,
                    # one dBeta-M ld.shared.f32, add.rn.f32 tree
                    if BETA_SIGMOID:
                        mul(dbeta_out, dbeta_out,
                            beta_self - beta_self * beta_self)
                        # instruction_selection: fma.rn.f32/mul.rn.f32;
                        # extent: sigmoid derivative on one output token
                    if (token_valid(item, chunk, cg1_linear_thread)
                            and chunk < item.wend):
                        if BETA_SIGMOID:
                            cast(dbeta_bf16, dbeta_out)
                            # instruction_selection: cvt.rn.bf16.f32
                            copy_r2g(dbeta_bf16,
                                     dbeta[global_token(
                                               item, chunk,
                                               cg1_linear_thread),
                                           item.head])
                            # instruction_selection: st.global.b16
                        else:
                            copy_r2g(dbeta_out,
                                     dbeta[global_token(
                                               item, chunk,
                                               cg1_linear_thread),
                                           item.head])
                            # instruction_selection: st.global.b32
                barrier(4, 128)  # dbeta sync 3: reduction SMEM reusable
                dbeta_m_cursor = advance(dbeta_m_cursor)

                wait_edge("dstate_acc", dstate_acc_cursor.stage,
                          dstate_acc_cursor.phase)
                if local_rev + 1 < num_compute_chunks:
                    wait_edge("dstate_smem_done", 0,
                              dstate_reuse_cursor.phase)
                    wait_edge("dstate_smem_cg2_done", 0,
                              dstate_reuse_cursor.phase)
                    for key_sub32 in range(4):
                        copy_t2r(cg1_dstate_tmem_cell(
                            tmem_base, tmem_row, warp,
                            key_sub32 * 32), dstate_regs, dtype="f32")
                        cast(dstate_bf16_words, dstate_regs)
                        copy_r2t(dstate_bf16_words,
                                 cg1_dstate_tmem_cell(
                                     tmem_base, tmem_row, warp,
                                     128 + key_sub32 * 16),
                                 dtype="bf16")
                    # instruction_selection: four tcgen05.ld .x32 FP32,
                    # BF16 pack, then four tcgen05.st .x16
                    wait_tmem_store()
                    # instruction_selection: tcgen05.wait::st.sync.aligned
                    arrive_edge("dstate_input", 0)
                    for key_sub32 in range(4):
                        copy_t2r(cg1_dstate_tmem_cell(
                            tmem_base, tmem_row, warp,
                            128 + key_sub32 * 16), dstate_bf16_words)
                        for half8 in range(4):
                            copy_r2s(dstate_bf16_words[half8],
                                     dstate_direct_bf16_byte(
                                         value_dim,
                                         key_sub32 * 32 + half8 * 8))
                    # instruction_selection: four tcgen05.ld .x16 and
                    # st.shared.v4.b32; extent: 128 key elements
                    fence("async_shared_cta")
                    # instruction_selection: fence.proxy.async.shared::cta; extent: warpgroup
                    arrive_edge("dstate_smem", 0)
                    dstate_reuse_cursor = advance(dstate_reuse_cursor)
                dstate_acc_cursor = advance(dstate_acc_cursor)
                raw_cursor = advance(raw_cursor)

            if WRITE_DSTATE0:
                if num_compute_chunks > 0 and item.wstart == 0:
                    for key_sub32 in range(4):
                        copy_t2r(cg1_dstate_tmem_cell(
                            tmem_base, tmem_row, warp, key_sub32 * 32),
                            dstate0_regs[key_sub32], dtype="f32")
                    for key in range(128):
                        copy_r2g(dstate0_regs[key], d_initial_state[
                            item.batch, item.head, value_dim, key])
                    # instruction_selection: tcgen05.ld.sync.aligned.
                    # 32x32b.x32.b32 then st.global.v4.b32; extent: 128 keys
                elif num_compute_chunks == 0:
                    for key in range(128):
                        if USE_DSTATE_IN:
                            copy_g2g(d_final_state[
                                item.batch, item.head, value_dim, key],
                                d_initial_state[
                                    item.batch, item.head, value_dim, key])
                        else:
                            copy_r2g(0.0, d_initial_state[
                                item.batch, item.head, value_dim, key])
                    # instruction_selection: ld.global/st.global.v4.b32 or
                    # st.global.v4.b32 zero; extent: 128 keys
                # When num_compute_chunks>0 and wstart!=0, this split item does
                # not own dInitial and intentionally performs no store.
            arrive_edge("dstate0_stored", 0)
            chunk_serial_base += item.cend - item.wstart
            tile, sched = scheduler_consume_at_tile_exit(sched, tile)
        arrive_edge("tmem_done", 0)
```

## Logical tcgen05 GEMMs

Every row is one primitive `gemm` occurrence above.  A row may emit a fixed
loop of the same `tcgen05.mma.cta_group::1.kind::f16` family; publication is a
separate `tcgen05.commit` operation.

| # | FP32 TMEM destination | A operand | B operand | MxNxK | runtime issues | accumulate |
| ---: | --- | --- | --- | --- | ---: | --- |
| 1 | state-K col 320 | state SMEM direct | K-decay SMEM transpose | 128x16x128 | 8 | no |
| 2 | dQ col 368 | state-input TMEM transpose | dO SMEM transpose | 128x16x128 | 8 | no |
| 3 | dU col 352 | dState-input TMEM transpose | K-restore SMEM lead16 | 128x16x128 | 8 | no |
| 4 | dState col 0 | dState-input TMEM transpose | state-scale diagonal SMEM | 128x128x128 | 8 | no |
| 5 | dU col 352 | dO SMEM transpose | A SMEM lead16 | 128x16x16 | 1 | yes |
| 6 | dState col 0 | dO SMEM transpose | Q-decay SMEM lead16 | 128x128x16 | 1 | yes |
| 7 | U col 336 | Y-input TMEM transpose | T-inverse SMEM lead16 | 128x16x16 | 1 | no |
| 8 | dY/state-K col 320 | dU-input TMEM transpose | T-inverse SMEM transpose | 128x16x16 | 1 | no |
| 9 | dK-restore col 416 | dState SMEM alternate transpose | U SMEM lead16 transpose | 128x16x128 | 8 | no |
| 10 | dState col 0 | negative-beta-dY TMEM transpose | K-decay SMEM lead16 | 128x128x16 | 1 | yes |
| 11 | dK-inverse col 400 | Q-decay SMEM transpose | dA SMEM lead16 | 128x16x16 | 1 | no |
| 12 | dQ col 368 | K-inverse SMEM A-major | dA SMEM | 128x16x16 | 1 | yes |
| 13 | dK-decay col 384 | state-input TMEM transpose | beta-dY (`sDv`) SMEM lead16 | 128x16x128 | 8 | no |
| 14 | dK-inverse col 400 | K-decay SMEM transpose | negative-dM SMEM lead16 | 128x16x16 | 1 | yes |
| 15 | dK-decay col 384 | K-inverse SMEM A-major | dM SMEM | 128x16x16 | 1 | yes |

The runtime issue vector is exactly `8,8,8,8,1,1,1,1,8,1,1,1,8,1,1`.
The debug anchor contains 59 static `tcgen05.mma` instruction lines: the 57
runtime issues plus the mutually-exclusive duplicate static predicate sites
for rows 12 and 15.  The delayed publication list has exactly 17 mbarrier
arrivals (2+2+1+2+2+1+2+1+3); no other logical row commits.

## Source/PTX/sketch correspondence

`basic-main` and `basic-prologue` below name the two PTX files in
`source_export/basic_debug_001`; all other labels name the sole main or
prologue PTX in the stated independent export directory.  Thus every PTX
range has one unambiguous file even though generated filenames contain the
full specialization.

| Source action | Source line(s) | Exact PTX evidence | Sketch line(s) |
| --- | ---: | ---: | ---: |
| order generation/scratch sorting | 2920..3054; `split_k.py` 574..712 | order tables below | 143..263 |
| sequence descriptor arrays 0..8 | 2865..2900; `thd.py` 50..99 | basic-prologue 140..end, nine repeated bodies | 264..289 |
| checkpoint prefix descriptor array 9 | 2901..2918; `thd.py` 50..99 | basic-prologue final repeated body | 290..311 |
| main ABI and exact rank-1 arena offsets | 3198..3271 | basic-main 35..101 | 321..372 |
| raw/state/output byte mappings | 3272..3298 and role-local address expressions | basic-main address arithmetic in each role range | 373..501 |
| seven raw shared-descriptor encodings | 3299..3395 | descriptor immediates at tcgen sites 2019..3257 | 502..712 |
| TMEM row/column and register lane coordinates | 638..661, 913..1217, 1700..1740, 2119..2150, 2568..2580 | basic-main 650..8329 role-local TMEM/ldmatrix/stmatrix operands | 537..712 |
| initialize all 115 words | 187..268, 3396..3486, `barrier.py` 183 | basic-main 102..266 | 715..796 |
| mbarrier-init fence / CTA sync | 3487..3488 | basic-main 285,287 | 797..800 |
| scheduler publish/consume | 267..297, `split_k.py` 524..572 | dynamic-main rows below | 845..876; publish 916; consume 1141,1378,1600,1939,2307,2716 |
| TMA descriptor acquire + Q/K/Gate/dO/V | 1521..1682 | basic-main 291..613 | 903..974 |
| checkpoint TMA | 1683..1696 | basic-main 614..629 | 976..990 |
| super KK register GEMM | 689..719 | basic-main 752..930 | 1012..1028 |
| beta-strict L and three inverse rounds | 720..799 | basic-main 931..1502 | 1030..1078 |
| dM, strict +/- tiles, dBeta-M | 800..871 | basic-main 1503..1855 | 1080..1139 |
| TMEM allocation/lifetime barrier | 903..905 | basic-main 1886..1891 | 1149..1154 |
| tcgen GEMM 1 | 1251..1266 | basic-main 2019..2159 | 1185..1199 |
| tcgen GEMM 2 | 1267..1286 | basic-main 2218..2315 | 1201..1216 |
| tcgen GEMM 3 | 1287..1304 | basic-main 2359..2450 | 1218..1230 |
| tcgen GEMM 4 | 1305..1323 | basic-main 2472..2577 | 1231..1241 |
| tcgen GEMMs 5/6 | 1324..1345 | basic-main 2610,2637 | 1243..1260 |
| tcgen GEMMs 7/8 | 1346..1385 | basic-main 2671,2698 | 1262..1280 |
| tcgen GEMM 9 | 1386..1402 | basic-main 2747..2929 | 1282..1296 |
| tcgen GEMMs 10/11 | 1403..1431 | basic-main 2967,3004 | 1298..1314 |
| tcgen GEMM 12, two static sites | 1432..1452 | basic-main 3028,3044 | 1316..1330 |
| tcgen GEMM 13 + state-input release | 1453..1471 | basic-main 3084..3181 | 1332..1346 |
| tcgen GEMMs 14/15 | 1472..1505 | basic-main 3221,3243,3257 | 1348..1372 |
| dState0 gate and TMEM teardown | 1506..1519 | basic-main 3298..3302 | 1374..1387 |
| epilogue A register GEMM/mask/store | 333..484 | basic-main 3495..3650 | 1423..1456 |
| epilogue dA register GEMM/mask/store | 485..536 | basic-main 3747..3899 | 1458..1496 |
| one-behind/tail output TMA ladder | 537..620 | basic-main 4139..4245 | 1498..1598 |
| CG0 beta, gate, raw Q/K | 1782..1922 | basic-main 4498..5163 | 1651..1779 |
| CG0 L2 and four rounded operands | 1923..2072 | basic-main 5164..5871 plus l2 rows below | 1781..1904 |
| CG0 unchanged state publication | 2073..2107 | basic-main 5902..6257 | 1906..1937 |
| CG2 state-dot and four drains | 2609..2736 | basic-main 6425..8002 | 1977..2133 |
| CG2 L2 Q/K projections | 2737..2776 | l2norm-main rows below | 2135..2260 |
| CG2 dQ/dK/dGate staging | 2777..2823 | basic-main 8003..8295 | 2266..2305 |
| CG1 dFinal seed/Y/dU/U/dY | 2165..2374 | state-main/basic-main rows below | 2339..2510 |
| CG1 beta-dY/dV/dBeta | 2375..2474 | basic-main 9077..9437; beta rows below | 2511..2647 |
| CG1 next-dState/dInitial | 2475..2544 | state-main rows below | 2649..2717 |

### Independent accepted-branch rows

| Frozen profile | Source branch | Exact independent-export PTX rows | Sketch line(s) |
| --- | ---: | ---: | ---: |
| `tail` | input TMA zero-fill 1643..1696; beta/gate scalar predicates 1782..1847; dBeta scalar predicate 2463..2473; output ownership 537..620 | basic-main 380..629 (unconditional TMA), 4139..4245 (output), 4450..4967 (beta/gate), 9380..9450 (dBeta) | TMA 929..974; output 1498..1598; beta/gate 1648..1699; dBeta 2628..2645 |
| `grouped` | head ratios at 1625..1628 | grouped-main 322..349 (signed ratio divisions before descriptor addresses 351..366) | 917..918 |
| `safe_gate` | 1762..1873 | safe_gate-main 4421..4967, `.loc 1 1762..1873` | 1630..1637,1686..1694 |
| `beta_sigmoid` beta | 1782..1807 | beta_sigmoid-main 4497..4507, `.loc 1 1782..1807` | 1654..1676 |
| `beta_sigmoid` dBeta | 2462..2473 | beta_sigmoid-main 9419..9463, `.loc 1 2462..2473` | 2623..2645 |
| `l2norm` forward | 1923..1972 | l2norm-main 5190..5377, `.loc 1 1923..1972` | 1781..1826 |
| `l2norm` backward | 2737..2776 | l2norm-main 8385..9991, `.loc 1 2737..2776` | 2135..2260 |
| `state` seed | 2165..2226 | state-main 8375..11044, `.loc 1 2165..2226` | 2339..2382 |
| `state` recurrence/output | 2475..2543 | state-main 10193..11562, `.loc 1 2475..2543` | 2649..2717 |
| `dynamic` | 267..297 | dynamic-main 358..376, 1944..1948, 3399..3403, 4439..4443, 6476..6480, 8548..8552, 10460..10464 | 845..876; 916,1141,1378,1600,1939,2307,2716 |
| `order_scratch` | `split_k.py` 574..712 | order_scratch-prologue 115..1055, `.loc 3 574..712` | 143..263 |
| `order_generate` | `split_k.py` 574..712 | order_generate-prologue 116..888, `.loc 3 574..712` | 143..263 |

The forward and reverse lookup is exhaustive at action granularity: every
storage construction, accepted compile-time branch, main role, synchronization
edge family, and logical tcgen row has exactly one owner above.  In the debug
anchor, `.file` 1 is the primary KDA source, file 4 barrier helpers, file 5
split-K scheduling, file 6 TMA helpers, file 7 MMA helpers, and file 8
pointwise helpers.  Independent prologue exports number their inlined split-K
file as 3, which is why the two order rows cite `.loc 3` explicitly.

## Static opcode evidence

| Static instruction family | Count | Sketch owner |
| --- | ---: | --- |
| `mbarrier.init.shared.b64` | 115 | distributed protocol initialization |
| `mbarrier.try_wait.parity.acquire...` | 74 | `wait_edge` sites |
| `mbarrier.arrive.shared.b64` | 45 | thread publication/reuse sites |
| `mbarrier.arrive.expect_tx.shared.b64` | 6 | TMA transaction starts |
| `tcgen05.mma...kind::f16` | 59 | warp 13 logical GEMMs |
| `tcgen05.commit...mbarrier` | 17 | warp 13 publication |
| `mma.sync...m16n8k16...bf16` | 76 | warps 12 and 15 |
| rank-3 TMA GMEM->SMEM | 12 | warp 14 raw loads |
| rank-4 TMA GMEM->SMEM | 2 | warp 14 checkpoint load |
| rank-3 TMA SMEM->GMEM | 20 | warp 15 store ladder |
| `tcgen05.alloc/relinquish/dealloc` | 1 each | warp 13 lifecycle |

## TIRx module and validation contract

The implementation must import device language only as
`import tirx_kernels.kern as K`.  Device code may not use `T`, `Tx`, `I`, any
`tirx.tile.*` API, a tile primitive, a first-class layout, or rank>1 SMEM.
The one `u8[196608]` arena and the integer offsets above are part of the frozen
translation.  `K.smem_pool(base=arena)` may allocate only the 920-byte protocol
header; all data mappings are raw integer address arithmetic.

`get_kernel` returns the prologue and main launch in order.  `prepare_data` and
the reference adapter build identical logical inputs but separate output and
workspace storage.  Correctness covers every accepted capability profile,
ragged BT tails, head ratios, state gradients, scheduler/order modes, direct
expanded outputs, and source plus FP64 recurrence references.  Benchmark timing
contains only the source `run_bwd` prologue+main and the matching two TIRx
launches; checkpoint/table construction remains outside the timed closure.

The benchmark suite is the only performance authority.  Low-level PTX/SASS or
profiler evidence may explain a candidate but cannot pass or select it.

## Instruction-selection summary

- Explicit one-dimensional placement plus byte offsets selects dynamic SMEM;
  integer swizzle bits and leading/stride constants select the same operand
  descriptor forms as the source without materializing a layout.
- Rank-3 and rank-4 TensorMap dimensionality, BF16/FP32 box widths, completion
  barriers, and 128-byte swizzle select the TMA families.
- Integer TMEM columns, CTA-group 1, BF16 operands, FP32 destinations, and the
  fixed logical M/N/K loops select the 59 `tcgen05.mma` sites.
- The exact warp-role chain, register targets, named-barrier counts, stage
  phases, persistent scheduler, one-behind stores, and teardown determine issue
  order; none is inferred from opcode counts alone.
