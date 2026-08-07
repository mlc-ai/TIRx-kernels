# FlashKDA BF16 M128: coarse WASP pipeline sketch

This file is a non-executable design sketch.  It is not a new IR, builder API,
or replacement for the audited Nymph transcription.  Its purpose is to show
the CUDA kernel as:

- explicit GMEM, SMEM, TMEM, and register tiles;
- explicit warp roles and source-order control flow;
- primitive tile dataflow inside each role;
- instruction selection derived from placement, shape, layout, and schedule,
  rather than descriptors or instruction hints.

The implementation represented by this sketch is maintained in
[`tirx_kernels/flashkda/bf16_fused_m128.py`](../../tirx_kernels/flashkda/bf16_fused_m128.py).
That module is the source of truth.

## Pipeline at a glance

| Warps | Role-local tile program | Main publication edge |
| --- | --- | --- |
| 0..3 | initialize/update the 128x128 TMEM state; form `beta * (V - U)` in registers; repack the next TMEM operand | `state_inp_ready`, `u_inp_ready`, `u2_inp_ready` |
| 4..7 | copy the 128x32 TMEM output through register tiles; use SMEM only for full-chunk output staging | `out_empty`, then GMEM output |
| 9 | issue four tile-GEMM chains over TMEM and staged SMEM operands: state-by-Q, state-by-K, residual-by-inverse, and residual-by-final-operand | `old_out_ready`, `u2_acc_ready`, `final_ready` |
| 10 | move one 32x128 V tile from GMEM to the current SMEM stage, with a masked segmented tail path | `v_full` |
| 12..31 | five interleaved four-warp instances prepare G/beta/Q/K, build Qd/Kd/Ki, run the small GEMMs, and construct the 32x32 lower-triangular operand from scalar/tile primitives | `qk_full` |
| 8, 11 | release registers and take no role branch | none |

The source-order `if / elif` role split and each role's loop/branch structure
remain visible in the complete sketch below.

## Primitive vocabulary

Structural operations do not compute values:

```python
tile(...)           # declare storage and logical shape
view(...)           # change logical indexing without moving data
alias(...)          # declare exact storage aliasing
slice(...)          # select a tile interval
transpose(...)       # transpose view
reg_tile(...)        # declare a role-local register tile
arange(...)          # construct logical row/column indices in registers
```

Copies always state their storage direction:

```python
copy_g2s(src, dst, mask=None, completion=None)  # global -> shared
copy_s2g(src, dst, mask=None)                   # shared -> global
copy_g2r(src, dst, mask=None)                   # global -> register
copy_r2g(src, dst, mask=None)                   # register -> global
copy_s2r(src, dst)                              # shared -> register
copy_r2s(src, dst)                              # register -> shared
copy_r2r(src, dst)                              # register -> register
copy_t2r(src, dst)                              # tensor memory -> register
copy_r2t(src, dst)                              # register -> tensor memory
```

The complete computational vocabulary used below is:

```python
fill(dst, value)
cast(dst, src)
add(dst, lhs, rhs)
sub(dst, lhs, rhs)
mul(dst, lhs, rhs)
div(dst, lhs, rhs)
fma(dst, lhs, rhs, acc)
exp(dst, src)
exp2(dst, src)
tanh(dst, src)
rsqrt(dst, src)
reduce_add(dst, src)
select(dst, predicate, true_value, false_value)
shuffle_xor(dst, src, lane_delta)
gemm(dst, lhs, rhs, accumulate=False)
```

For readability, a scalar expression such as `x = mul(a, b)` means the same
primitive with an implicit destination temporary.  A `copy_g2s` completion is
a producer/consumer edge, not an instruction-selection hint.

`pipe`, `init_pipe`, `expect_bytes`, `wait`, `arrive`, `barrier`, `fence`,
the directional-copy commit/wait groups, CTA synchronization, TMEM lifetime,
and stage/phase updates are schedule operations.  There are deliberately no
computational operations named `normalize`, `restore`, `inverse`,
`triangular_qk`, `tma`, `ldmatrix`, `stmatrix`, `tcgen05`, or `descriptor`.

## Complete sketch

```python
# ---------------------------------------------------------------------------
# Logical ABI and launch
# ---------------------------------------------------------------------------

@kernel(
    grid=num_seqs * num_heads,
    num_warps=32,
    threads_per_cta=1024,
    dynamic_smem_bytes=227328,
    tmem_columns=256,
    pipeline_stages=5,
    launch_bounds=1024,
)
def flashkda_bf16_fused_m128(
    q,                  # bf16 [tokens, heads, 128]
    k,                  # bf16 [tokens, heads, 128]
    v,                  # bf16 [tokens, heads, 128]
    g,                  # bf16 [tokens, heads, 128]
    beta,               # bf16 [tokens, heads], direct tail backing
    beta_full,          # bf16 [padded_tokens, padded_heads], full-tile backing
    A_log,              # f32  [heads]
    dt_bias,            # f32  [heads, 128]
    cu_seqlens,         # i64  [num_seqs + 1]
    seq_order,          # i32  [num_seqs]
    initial_state,      # bf16 [num_seqs, heads, 128, 128]
    out,                # bf16 [tokens, heads, 128]
    final_state,        # bf16 [num_seqs, heads, 128, 128]
    num_heads,
    use_initial_state,
    store_final_state,
    scale,
    lower_bound,
):
    STAGES = 5
    CHUNK = 32
    DIM = 128
    STAGE_BYTES = 41984

    # Tensor-map proxy acquisition is a lowering pattern implied by the
    # logical g2s/s2g copies.  Only its source-visible CTA ordering remains.
    cta_sync()

    tid = thread_id()
    warp = warp_uniform(tid // 32)
    lane = tid % 32

    # -----------------------------------------------------------------------
    # Shared-memory pool and logical aliases
    # -----------------------------------------------------------------------

    smem = tile("smem", "u8", [227328], byte_offset=0, alignment=1024)

    # Base 1024: gate input is overwritten by prepared Qd.
    qd = view(smem, "bf16", [5, 32, 128], byte_offset=1024,
              stage_stride=STAGE_BYTES, layout="b128")
    g_raw = alias(qd, "bf16", [5, 32, 128], layout="linear")

    # Base 9216: raw K is overwritten by prepared Kd.
    kd = view(smem, "bf16", [5, 32, 128], byte_offset=9216,
              stage_stride=STAGE_BYTES, layout="b128")
    k_raw = alias(kd, "bf16", [5, 32, 128], layout="b128")

    # Base 17408: Q prefetch / Ki / Kr / final 32x160 operand share storage.
    final_operand = view(smem, "bf16", [5, 32, 160], byte_offset=17408,
                         stage_stride=STAGE_BYTES, layout="b128")
    q_raw = alias(final_operand[:, :, 0:128], layout="b128")
    ki = alias(final_operand[:, :, 0:128], layout="b128")
    kr = alias(final_operand[:, :, 0:128], layout="b128")
    mqk_trans = alias(final_operand[:, :, 128:160], layout="b128")

    # Gate-prefix storage aliases the final-operand region at a different
    # lifetime and dtype.
    gate_prefix = view(smem, "f32", [5, 32, 128], byte_offset=25600,
                       stage_stride=STAGE_BYTES, layout="linear")

    inv = view(smem, "bf16", [5, 32, 32], byte_offset=29696,
               stage_stride=STAGE_BYTES, layout="b32")

    # V and inverse workspace have disjoint lifetimes.
    v_smem = view(smem, "bf16", [5, 32, 128], byte_offset=32384,
                  stage_stride=STAGE_BYTES, layout="linear")
    inv_work = alias(v_smem, "bf16", [5, 32, 32], layout="b128")

    # Full-path beta input and restore factors alias.
    beta_raw = view(smem, "bf16", [5, 32, 8], byte_offset=41984,
                    stage_stride=STAGE_BYTES, layout="linear")
    restore_factor = alias(beta_raw, "f32", [5, 129], layout="linear")

    gt_prefix = view(smem, "f32", [5, 128], byte_offset=41472,
                     stage_stride=STAGE_BYTES, layout="linear")
    gt = view(smem, "f32", [5, 128], byte_offset=31744,
              stage_stride=STAGE_BYTES, layout="linear")
    beta_value = view(smem, "f32", [5, 32], byte_offset=42500,
                      stage_stride=STAGE_BYTES, layout="linear")
    gate_rate = view(smem, "f32", [5], byte_offset=42628,
                     stage_stride=STAGE_BYTES, layout="linear")

    out_smem = view(smem, "bf16", [2, 32, 128], byte_offset=210944,
                    stage_stride=8192, layout="b128")

    # -----------------------------------------------------------------------
    # Seventeen synchronization edges: 77 physical mbarriers
    # -----------------------------------------------------------------------

    qk_full         = pipe(offset=0,   stages=5, arrivals=1)
    gate_raw_full   = pipe(offset=40,  stages=5, arrivals=1)
    qk_raw_full     = pipe(offset=80,  stages=5, arrivals=1)
    v_full          = pipe(offset=120, stages=5, arrivals=1)
    v_free          = pipe(offset=160, stages=5, arrivals=4)
    smem_free       = pipe(offset=200, stages=5, arrivals=1)
    raw_inputs_free = pipe(offset=240, stages=5, arrivals=1)
    state_inp_ready = pipe(offset=280, stages=5, arrivals=4)
    old_out_ready   = pipe(offset=320, stages=5, arrivals=1)
    u_inp_ready     = pipe(offset=360, stages=5, arrivals=4)
    u2_acc_ready    = pipe(offset=400, stages=5, arrivals=1)
    u2_inp_ready    = pipe(offset=440, stages=5, arrivals=4)
    final_ready     = pipe(offset=480, stages=5, arrivals=1)
    out_empty       = pipe(offset=520, stages=1, arrivals=1)
    dealloc_ready   = pipe(offset=528, stages=1, arrivals=2)
    diag_ready      = pipe(offset=536, stages=5, arrivals=2)
    inv16_ready     = pipe(offset=576, stages=5, arrivals=2)

    if warp == 0:
        leader = elected()
        for edge in (
            qk_full, gate_raw_full, qk_raw_full, v_full, v_free,
            smem_free, raw_inputs_free, state_inp_ready, old_out_ready,
            u_inp_ready, u2_acc_ready, u2_inp_ready, final_ready,
            out_empty, dealloc_ready, diag_ready, inv16_ready,
        ):
            for stage_index in range(edge.stages):
                init_pipe(
                    edge[stage_index],
                    arrivals=edge.arrivals,
                    predicate=leader,
                )
        fence("mbarrier_init_release_cluster")
    cta_sync()

    # -----------------------------------------------------------------------
    # TMEM allocation and aliases
    # -----------------------------------------------------------------------

    tmem_address = view(smem, "i32", [1], byte_offset=616)
    if warp == 0:
        tmem_alloc(tmem_address, columns=256)
    cta_sync()
    tmem_thread_fence()

    state_input = tile("tmem", "bf16", [128, 128], base_col=0)
    u2_acc = alias(state_input, "f32", [128, 32], base_col=0)

    state = tile("tmem", "f32", [128, 128], base_col=64)
    state_output = alias(state)

    output_acc = tile("tmem", "f32", [128, 32], base_col=192)

    u_acc = tile("tmem", "f32", [128, 32], base_col=224)
    u_input = alias(u_acc, "bf16", [128, 32], base_col=224)

    # The final 128x160 accumulate target is the contiguous state/output range.
    state_and_output = view("tmem", "f32", [128, 160], base_col=64)

    # Warps 9 and 10 need this as well; warps 8 and 11 execute it and then idle.
    if 8 <= warp <= 11:
        set_register_budget(direction="decrease", count=48)

    # This is the source's repeated five-way named-barrier branch, expanded at
    def prep_group_sync(instance):
        if instance == 0:
            barrier(11, threads=128)
        elif instance == 1:
            barrier(12, threads=128)
        else:
            if instance == 2:
                barrier(13, threads=128)
            elif instance == 3:
                barrier(14, threads=128)
            else:
                barrier(15, threads=128)

    # -----------------------------------------------------------------------
    # Role selection is the original else-if chain.
    # -----------------------------------------------------------------------

    if warp <= 3:
        # ===================================================================
        # COMPUTE ROLE: warps 0..3
        # ===================================================================

        set_register_budget(direction="increase", count=168)

        task_idx = block_id()
        seq_idx = copy_g2r(seq_order[task_idx // num_heads], reg_tile([], "i32"))
        head_idx = task_idx % num_heads
        bos = copy_g2r(cu_seqlens[seq_idx], reg_tile([], "i64"))
        eos = copy_g2r(cu_seqlens[seq_idx + 1], reg_tile([], "i64"))
        seq_len = eos - bos
        num_chunks = (seq_len + 31) // 32

        local_warp = warp
        state_rows = slice(local_warp * 32, (local_warp + 1) * 32)

        # Initial state: four 32-column blocks per warpgroup.
        for col_block in range(4):
            state_reg = reg_tile([32, 32], "f32")
            fill(state_reg, 0.0)

            if use_initial_state != 0:
                packed = reg_tile([32, 32], "bf16")
                copy_g2r(
                    initial_state[seq_idx, head_idx, state_rows,
                                  col_block * 32 : (col_block + 1) * 32],
                    packed,
                )
                cast(state_reg, packed)

            copy_r2t(
                state_reg,
                state[state_rows, col_block * 32 : (col_block + 1) * 32],
            )

        stage = 0
        phase_qk = 0
        phase_v = 0
        phase_old_out = 0
        phase_u2_acc = 0
        phase_final = 0

        for chunk in range(num_chunks):
            wait(qk_full[stage], phase_qk)

            for col_block in range(4):
                old_state = reg_tile([32, 32], "f32")
                copy_t2r(
                    state[state_rows, col_block * 32 : (col_block + 1) * 32],
                    old_state,
                )

                old_state_bf16 = reg_tile([32, 32], "bf16")
                cast(old_state_bf16, old_state)
                copy_r2t(
                    old_state_bf16,
                    state_input[state_rows,
                                col_block * 32 : (col_block + 1) * 32],
                )

                state_scale = reg_tile([32], "f32")
                copy_s2r(
                    gt[stage, col_block * 32 : (col_block + 1) * 32],
                    state_scale,
                )

                scaled_state = reg_tile([32, 32], "f32")
                mul(scaled_state, old_state, state_scale[None, :])
                copy_r2t(
                    scaled_state,
                    state[state_rows, col_block * 32 : (col_block + 1) * 32],
                )

            arrive(state_inp_ready[stage])

            wait(v_full[stage], phase_v)
            wait(old_out_ready[stage], phase_old_out)

            old_u = reg_tile([32, 32], "f32")
            copy_t2r(u_acc[state_rows, :], old_u)

            v_reg = reg_tile([32, 32], "f32")
            beta_reg = reg_tile([32], "f32")
            copy_s2r(transpose(v_smem[stage, :, state_rows]), v_reg)
            copy_s2r(beta_value[stage, :], beta_reg)

            residual = reg_tile([32, 32], "f32")
            sub(residual, v_reg, old_u)
            mul(residual, residual, beta_reg[None, :])

            residual_bf16 = reg_tile([32, 32], "bf16")
            cast(residual_bf16, residual)
            copy_r2t(residual_bf16, u_input[state_rows, :])

            arrive(v_free[stage])
            arrive(u_inp_ready[stage])

            wait(u2_acc_ready[stage], phase_u2_acc)

            u2_reg = reg_tile([32, 32], "f32")
            copy_t2r(u2_acc[state_rows, :], u2_reg)
            u2_bf16 = reg_tile([32, 32], "bf16")
            cast(u2_bf16, u2_reg)
            copy_r2t(u2_bf16, u_input[state_rows, :])

            arrive(u2_inp_ready[stage])
            wait(final_ready[stage], phase_final)

            stage = stage + 1
            if stage == 5:
                stage = 0
                phase_qk = phase_qk ^ 1
                phase_v = phase_v ^ 1
                phase_old_out = phase_old_out ^ 1
                phase_u2_acc = phase_u2_acc ^ 1
                phase_final = phase_final ^ 1

        if store_final_state != 0:
            for col_block in range(4):
                state_reg = reg_tile([32, 32], "f32")
                copy_t2r(
                    state[state_rows, col_block * 32 : (col_block + 1) * 32],
                    state_reg,
                )
                state_bf16 = reg_tile([32, 32], "bf16")
                cast(state_bf16, state_reg)
                copy_r2g(
                    state_bf16,
                    final_state[seq_idx, head_idx, state_rows,
                                col_block * 32 : (col_block + 1) * 32],
                )

        barrier(10, threads=128)
        if local_warp == 0 and elected():
            arrive(dealloc_ready)

    elif 4 <= warp <= 7:
        # ===================================================================
        # EPILOGUE ROLE: warps 4..7
        # ===================================================================

        set_register_budget(direction="decrease", count=48)

        task_idx = block_id()
        seq_idx = copy_g2r(seq_order[task_idx // num_heads], reg_tile([], "i32"))
        head_idx = task_idx % num_heads
        bos = copy_g2r(cu_seqlens[seq_idx], reg_tile([], "i64"))
        eos = copy_g2r(cu_seqlens[seq_idx + 1], reg_tile([], "i64"))
        seq_len = eos - bos
        num_chunks = (seq_len + 31) // 32

        local_warp = warp - 4
        state_rows = slice(local_warp * 32, (local_warp + 1) * 32)
        stage = 0
        output_stage = 0
        phase_final = 0

        for chunk in range(num_chunks):
            wait(final_ready[stage], phase_final)
            full = seq_len >= (chunk + 1) * 32

            if full:
                out_reg = reg_tile([32, 32], "f32")
                for dim_half in range(2):
                    copy_t2r(
                        output_acc[state_rows, dim_half * 16 : (dim_half + 1) * 16],
                        out_reg[:, dim_half * 16 : (dim_half + 1) * 16],
                    )

                barrier(9, threads=128)
                if local_warp == 0 and elected():
                    arrive(out_empty)

                if local_warp == 0 and chunk >= 2:
                    wait_s2g_read_group(1)
                barrier(9, threads=128)

                packed_out = reg_tile([32, 32], "bf16",
                                      layout="output_fragment")
                cast(packed_out, out_reg)
                copy_r2s(
                    transpose(packed_out),
                    out_smem[output_stage, :, state_rows],
                )

                fence("async_shared")
                barrier(9, threads=128)

                if local_warp == 0 and elected():
                    copy_s2g(
                        out_smem[output_stage, :, :],
                        out[bos + chunk * 32 : bos + (chunk + 1) * 32,
                            head_idx, :],
                    )
                    commit_s2g_group()

                output_stage = output_stage ^ 1
            else:
                out_reg = reg_tile([32, 32], "f32")
                copy_t2r(output_acc[state_rows, :], out_reg)

                barrier(9, threads=128)
                if local_warp == 0 and elected():
                    arrive(out_empty)

                out_bf16 = reg_tile([32, 32], "bf16")
                cast(out_bf16, out_reg)
                for token_col in range(32):
                    out_token = bos + chunk * 32 + token_col
                    if out_token < eos:
                        copy_r2g(
                            out_bf16[:, token_col],
                            out[out_token, head_idx, state_rows],
                        )

            stage = stage + 1
            if stage == 5:
                stage = 0
                phase_final = phase_final ^ 1

        if local_warp == 0:
            wait_s2g_group(0)
        barrier(9, threads=128)
        if local_warp == 0 and elected():
            arrive(dealloc_ready)

    elif warp == 9:
        # ===================================================================
        # MATRIX ROLE: warp 9
        # ===================================================================

        task_idx = block_id()
        seq_idx = copy_g2r(seq_order[task_idx // num_heads], reg_tile([], "i32"))
        bos = copy_g2r(cu_seqlens[seq_idx], reg_tile([], "i64"))
        eos = copy_g2r(cu_seqlens[seq_idx + 1], reg_tile([], "i64"))
        seq_len = eos - bos
        num_chunks = (seq_len + 31) // 32

        stage = 0
        phase_qk = 0
        phase_state_input = 0
        phase_out_empty = 1
        phase_u_input = 0
        phase_u2_input = 0

        for chunk in range(num_chunks):
            wait(qk_full[stage], phase_qk)
            wait(state_inp_ready[stage], phase_state_input)
            wait(out_empty, phase_out_empty)
            phase_out_empty = phase_out_empty ^ 1

            # [128,128] x [128,32] -> [128,32].  Eight visible K=16
            # GEMMs produce the source's eight independent issue statements.
            for k_block in range(8):
                gemm(
                    output_acc,
                    state_input[:, k_block * 16 : (k_block + 1) * 16],
                    transpose(qd[stage, :, k_block * 16 : (k_block + 1) * 16]),
                    accumulate=(k_block != 0),
                )

            for k_block in range(8):
                gemm(
                    u_acc,
                    state_input[:, k_block * 16 : (k_block + 1) * 16],
                    transpose(kd[stage, :, k_block * 16 : (k_block + 1) * 16]),
                    accumulate=(k_block != 0),
                )

            arrive(old_out_ready[stage])
            arrive(raw_inputs_free[stage])

            wait(u_inp_ready[stage], phase_u_input)
            for k_block in range(2):
                gemm(
                    u2_acc,
                    u_input[:, k_block * 16 : (k_block + 1) * 16],
                    inv[stage, k_block * 16 : (k_block + 1) * 16, :],
                    accumulate=(k_block != 0),
                )
            arrive(u2_acc_ready[stage])

            wait(u2_inp_ready[stage], phase_u2_input)
            for k_block in range(2):
                gemm(
                    state_and_output,
                    u_input[:, k_block * 16 : (k_block + 1) * 16],
                    final_operand[stage,
                                  k_block * 16 : (k_block + 1) * 16, :],
                    accumulate=True,
                )

            arrive(final_ready[stage])
            arrive(smem_free[stage])

            stage = stage + 1
            if stage == 5:
                stage = 0
                phase_qk = phase_qk ^ 1
                phase_state_input = phase_state_input ^ 1
                phase_u_input = phase_u_input ^ 1
                phase_u2_input = phase_u2_input ^ 1

        phase_dealloc = 0
        wait(dealloc_ready, phase_dealloc)
        tmem_dealloc(tmem_address, columns=256)
        tmem_relinquish()

    elif warp == 10:
        # ===================================================================
        # LOAD ROLE: warp 10
        # ===================================================================

        task_idx = block_id()
        seq_idx = copy_g2r(seq_order[task_idx // num_heads], reg_tile([], "i32"))
        head_idx = task_idx % num_heads
        bos = copy_g2r(cu_seqlens[seq_idx], reg_tile([], "i64"))
        eos = copy_g2r(cu_seqlens[seq_idx + 1], reg_tile([], "i64"))
        seq_len = eos - bos
        num_chunks = (seq_len + 31) // 32

        stage = 0
        phase_v_free = 1
        phase_qk = 0

        for chunk in range(num_chunks):
            wait(v_free[stage], phase_v_free)
            wait(qk_full[stage], phase_qk)

            full = seq_len >= (chunk + 1) * 32
            if full:
                if elected():
                    expect_bytes(v_full[stage], 8192)
                    copy_g2s(
                        v[bos + chunk * 32 : bos + (chunk + 1) * 32,
                          head_idx, :],
                        v_smem[stage, :, :],
                        completion=v_full[stage],
                    )
            else:
                for load_iter in range(16):
                    item = load_iter * 32 + lane
                    row = item // 16
                    segment = item % 16
                    token = bos + chunk * 32 + row
                    copy_g2s(
                        v[token, head_idx, segment * 8 : segment * 8 + 8],
                        v_smem[stage, row, segment * 8 : segment * 8 + 8],
                        mask=(token < eos),
                    )
                commit_g2s_group()
                wait_g2s_group(0)

            barrier(8, threads=32)
            if not full and elected():
                fence("async_shared")
                arrive(v_full[stage])

            stage = stage + 1
            if stage == 5:
                stage = 0
                phase_v_free = phase_v_free ^ 1
                phase_qk = phase_qk ^ 1

    elif 12 <= warp <= 31:
        # ===================================================================
        # PREP ROLE: five 4-warp instances on warps 12..31
        # ===================================================================

        set_register_budget(direction="decrease", count=48)

        task_idx = block_id()
        seq_idx = copy_g2r(seq_order[task_idx // num_heads], reg_tile([], "i32"))
        head_idx = task_idx % num_heads
        bos = copy_g2r(cu_seqlens[seq_idx], reg_tile([], "i64"))
        eos = copy_g2r(cu_seqlens[seq_idx + 1], reg_tile([], "i64"))
        seq_len = eos - bos
        num_chunks = (seq_len + 31) // 32

        instance = (warp - 12) // 4
        local_warp = (warp - 12) - instance * 4
        prep_tid = local_warp * 32 + lane
        num_prep_iters = (num_chunks + 4 - instance) // 5
        stage = instance

        if prep_tid == 0:
            a_log_reg = reg_tile([], "f32")
            rate_reg = reg_tile([], "f32")
            copy_g2r(A_log[head_idx], a_log_reg)
            exp(rate_reg, a_log_reg)
            copy_r2s(rate_reg, gate_rate[stage])
        prep_group_sync(instance)

        phase_raw_free = 1
        phase_gate_raw = 0
        phase_smem_free = 1
        phase_qk_raw = 0
        phase_diag = 0
        phase_inv16 = 0

        for prep_iter in range(num_prep_iters):
            chunk = prep_iter * 5 + instance
            full = seq_len >= (chunk + 1) * 32
            early_beta = fill(reg_tile([], "f32"), 0.0)
            early_gate0 = fill(reg_tile([], "f32"), 0.0)

            if full:
                wait(raw_inputs_free[stage], phase_raw_free)
                if local_warp == 0 and elected():
                    expect_bytes(gate_raw_full[stage], 8704)
                    copy_g2s(
                        g[bos + chunk * 32 : bos + (chunk + 1) * 32,
                          head_idx, :],
                        g_raw[stage, :, :],
                        completion=gate_raw_full[stage],
                    )
                    copy_g2s(
                        beta_full[bos + chunk * 32 : bos + (chunk + 1) * 32,
                                  (head_idx // 8) * 8 : (head_idx // 8 + 1) * 8],
                        beta_raw[stage, :, :],
                        completion=gate_raw_full[stage],
                    )
                    expect_bytes(qk_raw_full[stage], 16384)
                    copy_g2s(
                        k[bos + chunk * 32 : bos + (chunk + 1) * 32,
                          head_idx, :],
                        k_raw[stage, :, :],
                        completion=qk_raw_full[stage],
                    )

                wait(gate_raw_full[stage], phase_gate_raw)

                if local_warp == 2:
                    beta_pair = reg_tile([2], "f32")
                    beta_pair_bf16 = reg_tile([2], "bf16")
                    copy_s2r(
                        beta_raw[stage, lane, (head_idx % 8 // 2) * 2 :
                                              (head_idx % 8 // 2) * 2 + 2],
                        beta_pair_bf16,
                    )
                    cast(beta_pair, beta_pair_bf16)
                    beta_logit = select(
                        head_idx % 2 != 0,
                        beta_pair[1],
                        beta_pair[0],
                    )
                    beta_half = mul(beta_logit, 0.5)
                    beta_tanh = tanh(beta_half)
                    early_beta = add(mul(beta_tanh, 0.5), 0.5)

                if prep_tid < 128:
                    rate = copy_s2r(gate_rate[stage], reg_tile([], "f32"))
                    bias = copy_g2r(dt_bias[head_idx, prep_tid], reg_tile([], "f32"))
                    gate0_bf16 = copy_s2r(g_raw[stage, 0, prep_tid], reg_tile([], "bf16"))
                    gate0 = cast(reg_tile([], "f32"), gate0_bf16)
                    gate_arg = mul(rate, add(gate0, bias))
                    gate_tanh = tanh(mul(gate_arg, 0.5))
                    gate_sigmoid = add(mul(gate_tanh, 0.5), 0.5)
                    early_gate0 = mul(
                        mul(lower_bound, 1.4426950408889634),
                        gate_sigmoid,
                    )

            wait(smem_free[stage], phase_smem_free)
            if full and local_warp == 0 and elected():
                copy_g2s(
                    q[bos + chunk * 32 : bos + (chunk + 1) * 32,
                      head_idx, :],
                    q_raw[stage, :, :],
                    completion=qk_raw_full[stage],
                )

            if not full:
                for load_pass in range(4):
                    item = load_pass * 128 + prep_tid
                    row = item // 16
                    segment = item % 16
                    token = bos + chunk * 32 + row
                    copy_g2s(
                        g[token, head_idx, segment * 8 : segment * 8 + 8],
                        g_raw[stage, row, segment * 8 : segment * 8 + 8],
                        mask=(token < eos),
                    )
                commit_g2s_group()
                wait_g2s_group(0)
                prep_group_sync(instance)

            if local_warp == 2:
                beta_reg = early_beta
                if not full:
                    token = bos + chunk * 32 + lane
                    if token < eos:
                        beta_bf16 = copy_g2r(beta[token, head_idx], reg_tile([], "bf16"))
                        beta_logit = cast(reg_tile([], "f32"), beta_bf16)
                        beta_tanh = tanh(mul(beta_logit, 0.5))
                        beta_reg = add(mul(beta_tanh, 0.5), 0.5)
                copy_r2s(beta_reg, beta_value[stage, lane])

            # One thread computes a length-32 serial prefix for one dimension.
            if prep_tid < 128:
                rate = copy_s2r(gate_rate[stage], reg_tile([], "f32"))
                bias = copy_g2r(dt_bias[head_idx, prep_tid], reg_tile([], "f32"))
                prefix = fill(reg_tile([], "f32"), 0.0)

                for gate_row in range(32):
                    token = bos + chunk * 32 + gate_row
                    gate_log2 = fill(reg_tile([], "f32"), 0.0)
                    needs_compute = True

                    if gate_row == 0:
                        if full:
                            gate_log2 = early_gate0
                            needs_compute = False

                    if needs_compute:
                        if token < eos:
                            gate_bf16 = copy_s2r(
                                g_raw[stage, gate_row, prep_tid],
                                reg_tile([], "bf16"),
                            )
                            gate_f32 = cast(reg_tile([], "f32"), gate_bf16)
                            gate_arg = mul(rate, add(gate_f32, bias))
                            gate_tanh = tanh(mul(gate_arg, 0.5))
                            gate_sigmoid = add(mul(gate_tanh, 0.5), 0.5)
                            gate_log2 = mul(
                                mul(lower_bound, 1.4426950408889634),
                                gate_sigmoid,
                            )

                    prefix = add(prefix, gate_log2)
                    copy_r2s(prefix, gate_prefix[stage, gate_row, prep_tid])

            prep_group_sync(instance)

            if full:
                wait(qk_raw_full[stage], phase_qk_raw)

            common = mul(mul(lower_bound, 1.4426950408889634), 16.0)
            if prep_tid < 128:
                total = copy_s2r(
                    gate_prefix[stage, 31, prep_tid],
                    reg_tile([], "f32"),
                )
                factor = exp2(sub(total, common))
                copy_r2s(factor, restore_factor[stage, prep_tid])
            if prep_tid == 0:
                factor_scale = exp2(common)
                copy_r2s(factor_scale, restore_factor[stage, 128])

            # Each pass handles one [row, 8-element dimension segment].
            for work_pass in range(4):
                item = work_pass * 128 + prep_tid
                row = item // 16
                segment = item % 16
                token = bos + chunk * 32 + row
                valid = token < eos

                q_reg = fill(reg_tile([8], "f32"), 0.0)
                k_reg = fill(reg_tile([8], "f32"), 0.0)

                if full:
                    q_bf16 = reg_tile([8], "bf16")
                    k_bf16 = reg_tile([8], "bf16")
                    copy_s2r(
                        q_raw[stage, row, segment * 8 : segment * 8 + 8],
                        q_bf16,
                    )
                    copy_s2r(
                        k_raw[stage, row, segment * 8 : segment * 8 + 8],
                        k_bf16,
                    )
                    cast(q_reg, q_bf16)
                    cast(k_reg, k_bf16)
                elif valid:
                    q_bf16 = reg_tile([8], "bf16")
                    k_bf16 = reg_tile([8], "bf16")
                    copy_g2r(
                        q[token, head_idx, segment * 8 : segment * 8 + 8],
                        q_bf16,
                    )
                    copy_g2r(
                        k[token, head_idx, segment * 8 : segment * 8 + 8],
                        k_bf16,
                    )
                    cast(q_reg, q_bf16)
                    cast(k_reg, k_bf16)

                q_sum = reduce_add(mul(q_reg, q_reg))
                k_sum = reduce_add(mul(k_reg, k_reg))
                for delta in (8, 4, 2, 1):
                    q_sum = add(q_sum, shuffle_xor(q_sum, delta))
                    k_sum = add(k_sum, shuffle_xor(k_sum, delta))

                q_inv_norm = rsqrt(add(q_sum, 1e-6))
                k_inv_norm = rsqrt(add(k_sum, 1e-6))
                q_reg = mul(q_reg, q_inv_norm)
                k_reg = mul(k_reg, k_inv_norm)

                prefix8 = copy_s2r(
                    gate_prefix[stage, row, segment * 8 : segment * 8 + 8],
                    reg_tile([8], "f32"),
                )
                decay = exp2(sub(prefix8, common))

                qd_reg = mul(mul(q_reg, decay), scale)
                kd_reg = mul(k_reg, decay)
                ki_reg = div(k_reg, decay)

                qd_bf16 = cast(reg_tile([8], "bf16"), qd_reg)
                kd_bf16 = cast(reg_tile([8], "bf16"), kd_reg)
                ki_bf16 = cast(reg_tile([8], "bf16"), ki_reg)
                copy_r2s(qd_bf16, qd[stage, row, segment * 8 : segment * 8 + 8])
                copy_r2s(kd_bf16, kd[stage, row, segment * 8 : segment * 8 + 8])
                copy_r2s(ki_bf16, ki[stage, row, segment * 8 : segment * 8 + 8])

            prep_group_sync(instance)

            # Each local warp owns one 16x16 quadrant.
            pair_row = (local_warp // 2) * 16
            pair_col = (local_warp % 2) * 16
            acc = reg_tile([16, 16], "f32")

            if pair_row >= pair_col:
                for k_block in range(8):
                    kd_frag = reg_tile([16, 16], "bf16", layout="gemm_lhs")
                    ki_frag = reg_tile([16, 16], "bf16", layout="gemm_rhs")
                    copy_s2r(
                        kd[stage, pair_row : pair_row + 16,
                           k_block * 16 : (k_block + 1) * 16],
                        kd_frag,
                    )
                    copy_s2r(
                        ki[stage, pair_col : pair_col + 16,
                           k_block * 16 : (k_block + 1) * 16],
                        ki_frag,
                    )
                    gemm(
                        acc,
                        kd_frag,
                        transpose(ki_frag),
                        accumulate=(k_block != 0),
                    )

                beta_rows = copy_s2r(
                    beta_value[stage, pair_row : pair_row + 16],
                    reg_tile([16], "f32"),
                )
                rr = arange(pair_row, pair_row + 16)[:, None]
                cc = arange(pair_col, pair_col + 16)[None, :]
                seed = select(rr > cc, mul(acc, beta_rows[:, None]), 0.0)
                seed_bf16 = cast(reg_tile([16, 16], "bf16",
                                          layout="matrix_store"), seed)
                copy_r2s(
                    seed_bf16,
                    inv_work[stage, pair_row : pair_row + 16,
                              pair_col : pair_col + 16],
                )

                for k_block in range(8):
                    qd_frag = reg_tile([16, 16], "bf16", layout="gemm_lhs")
                    ki_frag = reg_tile([16, 16], "bf16", layout="gemm_rhs")
                    copy_s2r(
                        qd[stage, pair_row : pair_row + 16,
                           k_block * 16 : (k_block + 1) * 16],
                        qd_frag,
                    )
                    copy_s2r(
                        ki[stage, pair_col : pair_col + 16,
                           k_block * 16 : (k_block + 1) * 16],
                        ki_frag,
                    )
                    gemm(acc, qd_frag, transpose(ki_frag), accumulate=True)
            else:
                fill(acc, 0.0)

            rr = arange(pair_row, pair_row + 16)[:, None]
            cc = arange(pair_col, pair_col + 16)[None, :]
            mqk = select(rr >= cc, acc, 0.0)
            mqk_bf16 = cast(
                reg_tile([16, 16], "bf16", layout="matrix_store_transpose"),
                mqk,
            )
            copy_r2s(
                mqk_bf16,
                mqk_trans[stage, pair_col : pair_col + 16,
                           pair_row : pair_row + 16],
            )

            prep_group_sync(instance)

            if prep_tid < 128:
                total = copy_s2r(
                    gate_prefix[stage, 31, prep_tid],
                    reg_tile([], "f32"),
                )
                gt_value = exp2(total)
                copy_r2s(gt_value, gt[stage, prep_tid])

            # Local warps 2/3 restore rows 8..31 before block inversion.
            if local_warp >= 2:
                scale_reg = copy_s2r(
                    restore_factor[stage, 128],
                    reg_tile([], "f32"),
                )
                factor_reg = copy_s2r(
                    restore_factor[stage, :128],
                    reg_tile([128], "f32"),
                )

                for restore_pass in range(6):
                    row0 = 8 + (local_warp - 2) * 12 + restore_pass * 2
                    for row in (row0, row0 + 1):
                        qd_reg = cast(
                            reg_tile([128], "f32"),
                            copy_s2r(qd[stage, row, :], reg_tile([128], "bf16")),
                        )
                        kd_reg = cast(
                            reg_tile([128], "f32"),
                            copy_s2r(kd[stage, row, :], reg_tile([128], "bf16")),
                        )
                        ki_reg = cast(
                            reg_tile([128], "f32"),
                            copy_s2r(ki[stage, row, :], reg_tile([128], "bf16")),
                        )

                        qd_reg = mul(qd_reg, scale_reg)
                        kd_reg = mul(kd_reg, scale_reg)
                        kr_reg = mul(ki_reg, factor_reg)

                        copy_r2s(cast(reg_tile([128], "bf16"), qd_reg), qd[stage, row, :])
                        copy_r2s(cast(reg_tile([128], "bf16"), kd_reg), kd[stage, row, :])
                        copy_r2s(
                            cast(reg_tile([128], "bf16"), kr_reg),
                            final_operand[stage, row, 0:128],
                        )

            # Four independent 8x8 diagonal blocks are inverted by the exact
            # forward-substitution recurrence, expressed only with tile
            # slicing, select, mul, add, cast, and copies.
            if local_warp == 0:
                for block in range(4):
                    X_bf16 = reg_tile([8, 8], "bf16")
                    X = reg_tile([8, 8], "f32")
                    copy_s2r(
                        inv_work[stage,
                                 block * 8 : (block + 1) * 8,
                                 block * 8 : (block + 1) * 8],
                        X_bf16,
                    )
                    cast(X, X_bf16)

                    r = arange(8)[:, None]
                    c = arange(8)[None, :]
                    select(X, r == c, 1.0, X)

                    for pivot in range(7):
                        pivot_scale = mul(X[pivot + 1 : 8, pivot], -1.0)
                        update = mul(
                            pivot_scale[:, None],
                            X[pivot : pivot + 1, 0:pivot],
                        )
                        add(
                            X[pivot + 1 : 8, 0:pivot],
                            X[pivot + 1 : 8, 0:pivot],
                            update,
                        )
                        copy_r2r(pivot_scale, X[pivot + 1 : 8, pivot])

                    cast(X_bf16, X)
                    copy_r2s(
                        X_bf16,
                        inv_work[stage,
                                 block * 8 : (block + 1) * 8,
                                 block * 8 : (block + 1) * 8],
                    )

            if local_warp < 2:
                if elected():
                    arrive(diag_ready[stage])
                wait(diag_ready[stage], phase_diag)

            # Each of local warps 0/1 combines two 8x8 inverse blocks into one
            # 16x16 inverse block.  For [A 0; C D], the lower-left result is
            # -(D * C) * A; A and D already contain inverse blocks.
            if local_warp < 2:
                base = local_warp * 16
                A = reg_tile([8, 8], "bf16", layout="gemm_rhs")
                C = reg_tile([8, 8], "bf16", layout="gemm_rhs")
                D = reg_tile([8, 8], "bf16", layout="gemm_lhs")
                copy_s2r(inv_work[stage, base : base + 8, base : base + 8], A)
                copy_s2r(inv_work[stage, base + 8 : base + 16, base : base + 8], C)
                copy_s2r(inv_work[stage, base + 8 : base + 16,
                                  base + 8 : base + 16], D)

                DC = fill(reg_tile([8, 8], "f32"), 0.0)
                gemm(DC, D, C)
                mul(DC, DC, -1.0)
                DC_bf16 = cast(reg_tile([8, 8], "bf16"), DC)

                O = fill(reg_tile([8, 8], "f32"), 0.0)
                gemm(O, DC_bf16, A)
                O_bf16 = cast(reg_tile([8, 8], "bf16",
                                       layout="matrix_store"), O)
                copy_r2s(
                    O_bf16,
                    inv_work[stage, base + 8 : base + 16, base : base + 8],
                )

                if elected():
                    arrive(inv16_ready[stage])
                wait(inv16_ready[stage], phase_inv16)

            # Local warp 0 combines the two 16x16 blocks into the final 32x32
            # inverse with the same primitive block formula.
            if local_warp == 0:
                A = reg_tile([16, 16], "bf16", layout="gemm_rhs")
                C = reg_tile([16, 16], "bf16", layout="gemm_rhs")
                D = reg_tile([16, 16], "bf16", layout="gemm_lhs")
                copy_s2r(inv_work[stage, 0:16, 0:16], A)
                copy_s2r(inv_work[stage, 16:32, 0:16], C)
                copy_s2r(inv_work[stage, 16:32, 16:32], D)

                copy_r2s(D, inv[stage, 16:32, 16:32])

                DC = fill(reg_tile([16, 16], "f32"), 0.0)
                gemm(DC, D, C)
                mul(DC, DC, -1.0)
                DC_bf16 = cast(reg_tile([16, 16], "bf16"), DC)

                copy_r2s(A, inv[stage, 0:16, 0:16])

                O = fill(reg_tile([16, 16], "f32"), 0.0)
                gemm(O, DC_bf16, A)
                O_bf16 = cast(reg_tile([16, 16], "bf16",
                                       layout="matrix_store"), O)
                copy_r2s(O_bf16, inv[stage, 16:32, 0:16])
                zero = fill(
                    reg_tile([16, 16], "bf16", layout="matrix_store"),
                    0,
                )
                copy_r2s(zero, inv[stage, 0:16, 16:32])

            # Local warp 1 restores rows 0..7 after the inverse publication.
            elif local_warp == 1:
                scale_reg = copy_s2r(
                    restore_factor[stage, 128],
                    reg_tile([], "f32"),
                )
                factor_reg = copy_s2r(
                    restore_factor[stage, :128],
                    reg_tile([128], "f32"),
                )

                for restore_pass in range(4):
                    row0 = restore_pass * 2
                    for row in (row0, row0 + 1):
                        qd_reg = cast(
                            reg_tile([128], "f32"),
                            copy_s2r(qd[stage, row, :], reg_tile([128], "bf16")),
                        )
                        kd_reg = cast(
                            reg_tile([128], "f32"),
                            copy_s2r(kd[stage, row, :], reg_tile([128], "bf16")),
                        )
                        ki_reg = cast(
                            reg_tile([128], "f32"),
                            copy_s2r(ki[stage, row, :], reg_tile([128], "bf16")),
                        )

                        qd_reg = mul(qd_reg, scale_reg)
                        kd_reg = mul(kd_reg, scale_reg)
                        kr_reg = mul(ki_reg, factor_reg)

                        copy_r2s(cast(reg_tile([128], "bf16"), qd_reg), qd[stage, row, :])
                        copy_r2s(cast(reg_tile([128], "bf16"), kd_reg), kd[stage, row, :])
                        copy_r2s(
                            cast(reg_tile([128], "bf16"), kr_reg),
                            final_operand[stage, row, 0:128],
                        )

            fence("async_shared")
            prep_group_sync(instance)
            if local_warp == 0 and elected():
                arrive(qk_full[stage])

            # Each prep instance owns one ring slot and advances by a complete
            # five-stage rotation after every assigned chunk.
            for advance in range(5):
                stage = stage + 1
                if stage == 5:
                    stage = 0
                    phase_raw_free = phase_raw_free ^ 1
                    phase_gate_raw = phase_gate_raw ^ 1
                    phase_smem_free = phase_smem_free ^ 1
                    phase_qk_raw = phase_qk_raw ^ 1
                    phase_diag = phase_diag ^ 1
                    phase_inv16 = phase_inv16 ^ 1

    else:
        # Warps 8 and 11 take no role branch after releasing registers.
        pass
```

## Instruction selection is a lowering consequence

The sketch never requests a hardware instruction.  The following are derived
from the primitive operation together with its source/destination placement,
tile shape, layout, and synchronization use:

| Primitive pattern | Expected lowering family |
| --- | --- |
| full, aligned `copy_g2s(GMEM tile, staged SMEM tile, completion=edge)` | TMA load |
| masked 16-byte `copy_g2s` segments | `cp.async.cg` |
| full `copy_s2g(staged SMEM tile, GMEM tile)` followed by its copy group | TMA store |
| `copy_s2r(SMEM mma-layout, REG fragment-layout)` | `ldmatrix` |
| `copy_r2s(REG fragment-layout, SMEM mma-layout)` | `stmatrix` |
| ordinary `copy_s2r` / `copy_r2s` | shared load/store |
| `gemm(REG, REG, REG)` | warp `mma.sync` |
| `gemm(TMEM dst, TMEM lhs, SMEM rhs)` | TCGEN05 MMA |
| `copy_t2r` / `copy_r2t` | TCGEN05 load/store |

The visible K-block loops in the matrix role yield `8 + 8 + 2 + 2 = 20`
TCGEN05 MMA operations without an issue-count hint.  Likewise, the prep GEMM
tile shapes and directional copies determine the warp-MMA, ldmatrix, and
stmatrix expansion.  Their exact instruction counts belong to the lowering
snapshot and audit, not to this program vocabulary.
