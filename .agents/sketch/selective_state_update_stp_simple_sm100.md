<!--
Copyright (c) 2025 by FlashInfer team.
Copyright (c) 2026 The TIRX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Selective-state-update STP simple SM100: coarse WASP execution sketch

This is a non-executable sketch of FlashInfer's CUDA
`selective_state_update_kernel_simple`. It records the exact four-warp role
split, shared-memory exchange, packed state traversal, reduction order,
optional state-store paths, and low-occupancy dim tiling that the TIRx port must
preserve. The implementation represented by this sketch is maintained in
[`tirx_kernels/flashinfer/selective_state_update_stp_simple.py`](../../tirx_kernels/flashinfer/selective_state_update_stp_simple.py),
which becomes the source of truth after this sketch passes review.

The target is SM100/B200. Input, B, C, z and output are BF16. State is BF16,
FP16, FP32, or int16 with FP32 row scales; weight/dt/D/dt_bias is FP32 or BF16;
A is FP32; indices are int32 or int64. `DIM in {64,128,256}`,
`DSTATE in {64,96,128,256}`, `PHILOX_ROUNDS in {0,10}`, and both source launch
modes are in scope. The selected upstream int16 matrix uses DSTATE 64/128/256;
int16+DSTATE96 is not declared by this module. The selected stochastic-FP16
matrix uses DSTATE 64/128. MTP, producer/consumer STP kernels, and automatic
algorithm selection are out of scope.

## Pipeline at a glance

This kernel has no asynchronous pipeline. A CTA performs a cooperative preload,
meets at a CTA barrier, computes lane-striped DSTATE reductions, meets at a
second CTA barrier, and stores the output/optional scale. In the int16 path,
the normal SM100a source SASS also executes one `WARPSYNC.ALL` per row immediately
before the lane-0 max broadcast. That synchronization orders every lane's old
shared-scale read before lane 0 overwrites the row scale.

| Warp | Preload role | Compute ownership | Publication/reuse edges |
| --- | --- | --- | --- |
| 0 | x rows and optional FP32 state scales | `rows [0,rowsPerWarp)` | first CTA barrier; shared output/scale before second barrier |
| 1 | complete B vector for this batch/group | next `rowsPerWarp` rows | first CTA barrier; shared output/scale before second barrier |
| 2 | z rows, or explicit shared zeros when z is absent | next `rowsPerWarp` rows | first CTA barrier; shared output/scale before second barrier |
| 3 | complete C vector for this batch/group | final `rowsPerWarp` rows | first CTA barrier; shared output/scale before second barrier |

In the tiled launch `ROWS_PER_BLOCK=4`, every warp owns exactly one row. In the
full launch, `ROWS_PER_BLOCK=DIM` and `rowsPerWarp=DIM/4`. Every row is reduced
by one entire warp over DSTATE.

## Primitive vocabulary

Structural operations do not move or compute data:

```python
specialize(...)       # static dtype, DIM, DSTATE, ROWS_PER_BLOCK, Philox variant
launch(...)           # grid/block metadata
tile(...)             # GMEM/SMEM/register storage declaration
view(...)             # typed view without a copy
reg_tile(...)         # lane-private scalar/vector registers
```

Copies always name their storage direction:

```python
copy_g2r(src, dst, predicate=None)
copy_g2s(src, dst, predicate=None)
copy_s2r(src, dst)
copy_r2s(src, dst)
copy_r2g(src, dst, predicate=None)
```

The computation vocabulary is deliberately primitive:

```python
fill(dst, value)
cast(dst, src, rounding=None)
add(dst, lhs, rhs)
sub(dst, lhs, rhs)
mul(dst, lhs, rhs)
fma(dst, lhs, rhs, acc)
exp2(dst, src)
log2(dst, src)
div(dst, lhs, rhs)
abs(dst, src)
min(dst, lhs, rhs)
max(dst, lhs, rhs)
select(dst, predicate, true_value, false_value)
bit_and(dst, lhs, rhs)
bit_xor(dst, lhs, rhs)
mul_hi_u32(dst, lhs, rhs)
mul_lo_s32(dst, lhs, rhs)
add_s32(dst, lhs, rhs)
prmt_b32(dst, lhs, rhs, selector)
shuffle_down(src, lane_delta, clamp, member_mask) -> (dst, pred)
shuffle_index(src, source_lane, clamp, member_mask) -> (dst, pred)
warp_sync(member_mask)
```

`cta_sync`, `warp_sync`, predicates, loop indices and static address expressions
are schedule or control operations. There is no compound `update_state`, `softplus`,
`quantize`, `gate`, or `reduce` operation: those paths are expanded below.

## Complete sketch

```python
# ===========================================================================
# Static specializations, runtime ABI, and launch
# ===========================================================================

variant = specialize(
    INPUT_DTYPE="bf16",
    STATE_DTYPE=("bf16", "f16", "f32", "i16"),
    WEIGHT_DTYPE=("f32", "bf16"),
    MATRIX_A_DTYPE="f32",
    INDEX_DTYPE=("i32", "i64"),
    SCALE_STATE=(False, True),
    DIM=(64, 128, 256),
    DSTATE=(64, 96, 128, 256),
    ROWS_PER_BLOCK=(4, DIM),
    PHILOX_ROUNDS=(0, 10),
    target="sm_100a",
)

host_assert(SCALE_STATE == (STATE_DTYPE == "i16"))
host_assert(STATE_DTYPE != "i16" or DSTATE in (64, 128, 256))
host_assert(PHILOX_ROUNDS == 0 or
            (PHILOX_ROUNDS == 10 and STATE_DTYPE == "f16" and
             DSTATE in (64, 128) and not SCALE_STATE))

INPUT_VECTOR_BYTES = 16
STATE_VECTOR = min(16 // sizeof(STATE_DTYPE), DSTATE // 32)
STATE_VECTOR_BYTES = STATE_VECTOR * sizeof(STATE_DTYPE)
STATE_VECTOR_ALIGNMENT = largest_power_of_two_divisor(STATE_VECTOR_BYTES)
host_assert(address(x) % INPUT_VECTOR_BYTES == 0)
host_assert(x_stride_batch * sizeof(INPUT_DTYPE) % INPUT_VECTOR_BYTES == 0)
if z is present:
    host_assert(address(z) % INPUT_VECTOR_BYTES == 0)
    host_assert(z_stride_batch * sizeof(INPUT_DTYPE) % INPUT_VECTOR_BYTES == 0)
host_assert(address(B) % INPUT_VECTOR_BYTES == 0)
host_assert(address(C) % INPUT_VECTOR_BYTES == 0)
host_assert(B_stride_batch * sizeof(INPUT_DTYPE) % INPUT_VECTOR_BYTES == 0)
host_assert(C_stride_batch * sizeof(INPUT_DTYPE) % INPUT_VECTOR_BYTES == 0)
host_assert(address(state) % STATE_VECTOR_ALIGNMENT == 0)
host_assert(DIM * DSTATE * sizeof(STATE_DTYPE) % STATE_VECTOR_BYTES == 0)

launch_config = launch(
    grid=(batch, nheads, ceildiv(DIM, ROWS_PER_BLOCK)),
    block=(32, 4, 1),
    threads=128,
    dynamic_smem_bytes=0,
)

def selective_state_update_stp_simple(
    state, state_scale,
    x, dt, A, B, C, D, z, dt_bias,
    state_batch_indices, dst_state_batch_indices, rand_seed,
    output,
    state_stride_batch, state_scale_stride_batch,
    x_stride_batch, dt_stride_batch, B_stride_batch, C_stride_batch,
    z_stride_batch, out_stride_batch,
    state_batch_indices_stride_batch,
    dst_state_batch_indices_stride_batch,
    nheads, ngroups, dt_softplus, update_state, pad_slot_id,
):
    if PHILOX_ROUNDS > 0 and not SCALE_STATE:
        random_seed = 0
        # instruction_selection: mov.b64 zero; extent: one nullable-seed fallback
        if rand_seed is present:
            random_seed = copy_g2r(rand_seed[0])
            # instruction_selection: guarded ld.global.b64 after setp.eq.b64 and a pointer-null branch; extent: one scalar per thread, issued at kernel entry before thread/slot/scalar setup

    thread_x = thread_id(axis="x", extent=32)
    # instruction_selection: mov.u32 from %tid.x; extent: one physical x coordinate
    lane = bit_and(thread_x, 31)
    # instruction_selection: and.b32 with immediate 31; extent: one source-exact lane coordinate
    warp = thread_id(axis="y", extent=4)
    # instruction_selection: mov.u32 from %tid.y; extent: one warp coordinate
    batch_i, head, dim_tile = cta_id(
        extents=(batch, nheads, ceildiv(DIM, ROWS_PER_BLOCK)))
    # instruction_selection: mov.u32 from %ctaid.x/y/z; extent: three CTA coordinates
    # The physical %ctaid.z read is retained even when the full launch has grid.z=1.

    dim_offset = dim_tile * ROWS_PER_BLOCK
    group = head // (nheads // ngroups)
    rowsPerWarp = ceildiv(ROWS_PER_BLOCK, 4)

    if state_batch_indices is present:
        state_batch = copy_g2r(
            state_batch_indices[batch_i * state_batch_indices_stride_batch],
            result_dtype="i64", sign_extend=(INDEX_DTYPE == "i32"))
        # instruction_selection: ld.global.s32 directly into an i64 register for i32, or ld.global.b64 for i64; extent: one scalar per thread
    else:
        state_batch = cast("i64", batch_i)
        # instruction_selection: cvt.u64.u32; extent: one scalar

    if dst_state_batch_indices is present:
        dst_state_batch = copy_g2r(
            dst_state_batch_indices[batch_i * dst_state_batch_indices_stride_batch],
            result_dtype="i64", sign_extend=(INDEX_DTYPE == "i32"))
        # instruction_selection: ld.global.s32 directly into an i64 register for i32, or ld.global.b64 for i64; extent: one scalar per thread
    else:
        dst_state_batch = state_batch

    state_head = view(
        state,
        start=state_batch * state_stride_batch + head * DIM * DSTATE,
        shape=(DIM, DSTATE),
    )
    dst_state_head = view(
        state,
        start=dst_state_batch * state_stride_batch + head * DIM * DSTATE,
        shape=(DIM, DSTATE),
    )
    if SCALE_STATE:
        scale_rows = view(
            state_scale,
            start=state_batch * state_scale_stride_batch + head * DIM,
            shape=(DIM,),
        )
        dst_scale_rows = view(
            state_scale,
            start=dst_state_batch * state_scale_stride_batch + head * DIM,
            shape=(DIM,),
        )

    # =======================================================================
    # Exact static shared-memory field order and lifetimes
    # =======================================================================

    OFF_X = 0
    OFF_Z = align_up(OFF_X + 2 * ROWS_PER_BLOCK, 16)
    OFF_B = align_up(OFF_Z + 2 * ROWS_PER_BLOCK, 16)
    OFF_C = align_up(OFF_B + 2 * DSTATE, 16)
    OFF_OUT = OFF_C + 2 * DSTATE
    OFF_SCALE_TAIL = align_up(OFF_OUT + 4 * ROWS_PER_BLOCK, 16)
    SCALE_TAIL_BYTES = (4 * ROWS_PER_BLOCK) if SCALE_STATE else ROWS_PER_BLOCK
    SHARED_BYTES = align_up(OFF_SCALE_TAIL + SCALE_TAIL_BYTES, 16)

    shared_raw = tile("smem", "u8", [SHARED_BYTES], alignment=16)
    sX = view(shared_raw, "bf16", [ROWS_PER_BLOCK], byte_offset=OFF_X,
              lifetime="cooperative preload through row compute")
    sZ = view(shared_raw, "bf16", [ROWS_PER_BLOCK], byte_offset=OFF_Z,
              lifetime="cooperative preload through output epilogue")
    sB = view(shared_raw, "bf16", [DSTATE], byte_offset=OFF_B,
              lifetime="cooperative preload through all row reductions")
    sC = view(shared_raw, "bf16", [DSTATE], byte_offset=OFF_C,
              lifetime="cooperative preload through all row reductions")
    sOut = view(shared_raw, "f32", [ROWS_PER_BLOCK], byte_offset=OFF_OUT,
                lifetime="row reduction publication through output epilogue")
    if SCALE_STATE:
        sScale = view(shared_raw, "f32", [ROWS_PER_BLOCK], byte_offset=OFF_SCALE_TAIL,
                      lifetime="scale preload through destination-scale store")
    else:
        dummyScaleTail = view(shared_raw, "i8", [ROWS_PER_BLOCK],
                              byte_offset=OFF_SCALE_TAIL,
                              lifetime="unused conditional struct member")

    # =======================================================================
    # Per-CTA scalar setup, in source order
    # =======================================================================

    A_value = copy_g2r(A[head])
    # instruction_selection: ld.global.b32; extent: one scalar per thread
    dt_value = copy_g2r(dt[batch_i * dt_stride_batch + head])
    # instruction_selection: ld.global.b32 for FP32 or ld.global.b16 for BF16; extent: one scalar
    dt_value = cast("f32", dt_value)
    # instruction_selection: cvt.f32.bf16 for BF16, identity for FP32; extent: one scalar

    if dt_bias is present:
        bias_value = copy_g2r(dt_bias[head])
        # instruction_selection: ld.global.b32 for FP32 or ld.global.b16 for BF16; extent: one scalar
        bias_value = cast("f32", bias_value)
        # instruction_selection: cvt.f32.bf16 for BF16, identity for FP32; extent: one scalar
        dt_value = add(dt_value, bias_value)
        # instruction_selection: add.ftz.f32; extent: one scalar

    if dt_softplus:
        within_threshold = dt_value <= 20.0
        # instruction_selection: complementary setp.gtu.ftz.f32 combined with the dt_softplus predicate and a branch that also skips unordered/NaN; extent: one scalar predicate
        if within_threshold:
            exp_arg = mul(dt_value, LOG2_E)
            # instruction_selection: mul.ftz.f32; extent: one scalar
            exp_dt = exp2(exp_arg)
            # instruction_selection: ex2.approx.ftz.f32; extent: one scalar
            one_plus_exp = add(1.0, exp_dt)
            # instruction_selection: add.ftz.f32; extent: one scalar
            log2_value = log2(one_plus_exp)
            # instruction_selection: lg2.approx.ftz.f32; extent: one scalar
            dt_value = mul(log2_value, LN_2)
            # instruction_selection: mul.ftz.f32; extent: one scalar

    A_dt = mul(A_value, dt_value)
    # instruction_selection: mul.ftz.f32; extent: one scalar
    dA_exp_arg = mul(A_dt, LOG2_E)
    # instruction_selection: mul.ftz.f32; extent: one scalar
    dA = exp2(dA_exp_arg)
    # instruction_selection: ex2.approx.ftz.f32; extent: one scalar

    if D is present:
        d_value = copy_g2r(D[head])
        # instruction_selection: ld.global.b32 for FP32 or ld.global.b16 for BF16; extent: one scalar
        d_value = cast("f32", d_value)
        # instruction_selection: cvt.f32.bf16 for BF16, identity for FP32; extent: one scalar
    else:
        d_value = 0.0
        # instruction_selection: mov.b32 zero; extent: one scalar

    # =======================================================================
    # Four mutually exclusive preload roles
    # =======================================================================

    if warp == 0:
        for local_row in lane_strided_range(0, ROWS_PER_BLOCK, 32):
            d = dim_offset + local_row
            if d < DIM:
                x_value = copy_g2r(x[batch_i * x_stride_batch + head * DIM + d])
                # instruction_selection: ld.global.b16; extent: one scalar per active lane
                copy_r2s(x_value, sX[local_row])
                # instruction_selection: st.shared.b16; extent: one scalar per active lane
        if SCALE_STATE:
            for local_row in lane_strided_range(0, ROWS_PER_BLOCK, 32):
                d = dim_offset + local_row
                if d < DIM:
                    scale_value = copy_g2r(scale_rows[d])
                    # instruction_selection: ld.global.b32; extent: one scalar per active lane
                    copy_r2s(scale_value, sScale[local_row])
                    # instruction_selection: st.shared.b32; extent: one scalar per active lane

    elif warp == 1:
        for i in lane_vector_range(0, DSTATE, vector=8):
            b_vec = copy_g2r(B[batch_i * B_stride_batch + group * DSTATE + i : i + 8])
            # instruction_selection: ld.global.v4.b32; extent: one 16-byte PackedAligned<bf16,8> load per active lane/iteration
            copy_r2s(b_vec, sB[i : i + 8])
            # instruction_selection: st.shared.v4.b32; extent: one 16-byte PackedAligned<bf16,8> store per active lane/iteration

    elif warp == 2:
        if z is present:
            for local_row in lane_strided_range(0, ROWS_PER_BLOCK, 32):
                d = dim_offset + local_row
                if d < DIM:
                    z_value = copy_g2r(z[batch_i * z_stride_batch + head * DIM + d])
                    # instruction_selection: ld.global.b16; extent: one scalar per active lane
                    copy_r2s(z_value, sZ[local_row])
                    # instruction_selection: st.shared.b16; extent: one scalar per active lane
        else:
            for local_row in lane_strided_range(0, ROWS_PER_BLOCK, 32):
                d = dim_offset + local_row
                if d < DIM:
                    zero_i32 = fill("i32", 0)
                    # instruction_selection: mov.b32 zero; extent: one scalar per active lane
                    z_value = cast("bf16", zero_i32, rounding="rn")
                    # instruction_selection: cvt.rn.bf16.s32; extent: one scalar per active lane
                    copy_r2s(z_value, sZ[local_row])
                    # instruction_selection: st.shared.b16; extent: one scalar per active lane

    elif warp == 3:
        for i in lane_vector_range(0, DSTATE, vector=8):
            c_vec = copy_g2r(C[batch_i * C_stride_batch + group * DSTATE + i : i + 8])
            # instruction_selection: ld.global.v4.b32; extent: one 16-byte PackedAligned<bf16,8> load per active lane/iteration
            copy_r2s(c_vec, sC[i : i + 8])
            # instruction_selection: st.shared.v4.b32; extent: one 16-byte PackedAligned<bf16,8> store per active lane/iteration

    cta_sync()
    # instruction_selection: bar.sync 0; extent: all 128 threads after cooperative preload

    # =======================================================================
    # One warp owns each row; lanes stripe the DSTATE dimension
    # =======================================================================

    # For two-byte state and DSTATE 64/96/128/256 this is 2/3/4/8.
    # For FP32 it is 2/3/4/4.
    # Packed state load/store instruction family:
    #   two-byte state: 2 -> v2.b16, 3 -> three scalar b16,
    #                   4 -> v4.b16, 8 -> v4.b32 plus packing moves;
    #   FP32 state:     2 -> v2.b32, 3 -> three scalar b32, 4 -> v4.b32.
    # Shared BF16 B/C use the corresponding 2/3/4/8-element family:
    # v2.b16, three scalar b16, v4.b16, or v4.b32 plus packing moves.

    for local_row in static_range(warp * rowsPerWarp, (warp + 1) * rowsPerWarp):
        d = dim_offset + local_row
        if d >= DIM:
            break

        x_value = copy_s2r(sX[local_row])
        # instruction_selection: ld.shared.b16; extent: one scalar per lane
        x_value = cast("f32", x_value)
        # instruction_selection: cvt.f32.bf16; extent: one scalar

        if SCALE_STATE:
            decode_scale = copy_s2r(sScale[local_row])
            # instruction_selection: ld.shared.b32; extent: one scalar per lane
            new_state_max = -FLT_MAX
            # instruction_selection: no standalone instruction; the immediate is folded into the first max.ftz.f32 operand
            rNewState = reg_tile("f32", [DSTATE // 32])
        else:
            decode_scale = 1.0
            # instruction_selection: compile-time eliminated together with the decode multiplication; extent: no emitted PTX

        d_times_x = mul(d_value, x_value)
        # instruction_selection: mul.ftz.f32; extent: one scalar per lane
        lane_is_zero = select(lane == 0, 1.0, 0.0)
        # instruction_selection: setp.eq.b32 plus selp.f32; extent: one scalar lane predicate materialization
        out_value = mul(d_times_x, lane_is_zero)
        # instruction_selection: folded into the first later fma.rn.ftz.f32 accumulator input; source extent remains one scalar per lane

        for iteration, i in lane_vector_range_with_iteration(
                lane * STATE_VECTOR, DSTATE, 32 * STATE_VECTOR):
            rState = reg_tile(STATE_DTYPE, [STATE_VECTOR])
            fill(rState, 0)
            # instruction_selection: zero moves for the packed aggregate; extent: STATE_VECTOR elements
            if state_batch != pad_slot_id:
                copy_g2r(state_head[d, i : i + STATE_VECTOR], rState)
                # instruction_selection: ld.global.v2.b16 / three ld.global.b16 / ld.global.v4.b16 / ld.global.v4.b32+packing for two-byte state, or ld.global.v2.b32 / three ld.global.b32 / ld.global.v4.b32 for FP32; extent: one exact PackedAligned tile

            if STATE_VECTOR != 3:
                rB = reg_tile("bf16", [STATE_VECTOR])
                rC = reg_tile("bf16", [STATE_VECTOR])

            if PHILOX_ROUNDS > 0 and not SCALE_STATE:
                random_words = reg_tile("u32", [4])
                raw_sr_f16x2 = reg_tile("u32", [STATE_VECTOR])

            for e in static_range(STATE_VECTOR):
                if PHILOX_ROUNDS > 0 and not SCALE_STATE and e % 4 == 0:
                    offset = (
                        state_batch * state_stride_batch
                        + head * DIM * DSTATE + d * DSTATE + i + e)
                    c0 = low_u32(offset)
                    c1 = high_u32(offset)
                    c2 = 0
                    c3 = 0
                    k0 = low_u32(random_seed)
                    k1 = high_u32(random_seed)
                    for philox_i in static_range(10):
                        old_c0 = c0
                        old_c2 = c2

                        c0_hi = mul_hi_u32(0xCD9E8D57, old_c2)
                        # instruction_selection: mul.hi.u32; extent: one issue per round
                        c0_xor_c1 = bit_xor(c0_hi, c1)
                        # instruction_selection: xor.b32; extent: one issue per round
                        next_c0 = bit_xor(c0_xor_c1, k0)
                        # instruction_selection: xor.b32; extent: one issue per round

                        c2_hi = mul_hi_u32(0xD2511F53, old_c0)
                        # instruction_selection: mul.hi.u32; extent: one issue per round
                        c2_xor_c3 = bit_xor(c2_hi, c3)
                        # instruction_selection: xor.b32; extent: one issue per round
                        next_c2 = bit_xor(c2_xor_c3, k1)
                        # instruction_selection: xor.b32; extent: one issue per round

                        next_c1 = mul_lo_s32(old_c2, 0xCD9E8D57)
                        # instruction_selection: mul.lo.s32; extent: one issue per round
                        next_c3 = mul_lo_s32(old_c0, 0xD2511F53)
                        # instruction_selection: mul.lo.s32; extent: one issue per round
                        next_k0 = add_s32(k0, 0x9E3779B9)
                        # instruction_selection: add.s32; extent: one issue per round
                        next_k1 = add_s32(k1, 0xBB67AE85)
                        # instruction_selection: add.s32; extent: one issue per round

                        c0, c1, c2, c3 = next_c0, next_c1, next_c2, next_c3
                        k0, k1 = next_k0, next_k1
                        # The ten rounds are statically unrolled. In the first
                        # SM100a round only, c3==0 removes c2_xor_c3 and old_c2==0
                        # removes next_c1; constant propagation may also fold key
                        # updates. These are PTX-specialization folds, not changes
                        # to the source data-dependency graph above.
                    random_words = (c0, c1, c2, c3)

                state_value = cast("f32", rState[e])
                # instruction_selection: cvt.f32.bf16, cvt.f32.f16, cvt.rn.f32.s16, or identity for f32; extent: one scalar
                if SCALE_STATE:
                    state_value = mul(state_value, decode_scale)
                    # instruction_selection: mul.ftz.f32; extent: one scalar
                # With SCALE_STATE false, the source's multiply by the compile-time
                # 1.0 decode scale is eliminated and emits no PTX instruction.

                if STATE_VECTOR == 3:
                    b_element = copy_s2r(sB[i + e])
                    # instruction_selection: ld.shared.b16; extent: one scalar for this e, issued after this e's state conversion
                    B_value = cast("f32", b_element)
                    # instruction_selection: cvt.f32.bf16; extent: one scalar for this e, completed before the C load
                    c_element = copy_s2r(sC[i + e])
                    # instruction_selection: ld.shared.b16; extent: one scalar for this e, issued after the B conversion
                    C_value = cast("f32", c_element)
                    # instruction_selection: cvt.f32.bf16; extent: one scalar for this e, completed before compute
                else:
                    if e == 0:
                        copy_s2r(sB[i : i + STATE_VECTOR], rB)
                        # instruction_selection: ld.shared.v2.b16 / ld.shared.v4.b16 / ld.shared.v4.b32 plus unpack moves for STATE_VECTOR 2/4/8; extent: one exact BF16 tile after the first state conversion (and after Philox in SR variants)
                    B_value = cast("f32", rB[e])
                    # instruction_selection: cvt.f32.bf16; extent: one scalar for this e; the e=0 conversion completes before the C tile load
                    if e == 0:
                        copy_s2r(sC[i : i + STATE_VECTOR], rC)
                        # instruction_selection: ld.shared.v2.b16 / ld.shared.v4.b16 / ld.shared.v4.b32 plus unpack moves for STATE_VECTOR 2/4/8; extent: one exact BF16 tile after the first B conversion
                    C_value = cast("f32", rC[e])
                    # instruction_selection: cvt.f32.bf16; extent: one scalar for this e, completed before compute

                dB = mul(B_value, dt_value)
                # instruction_selection: mul.ftz.f32; extent: one scalar
                dB_x = mul(dB, x_value)
                # instruction_selection: mul.ftz.f32; extent: one scalar
                new_state = fma(state_value, dA, dB_x)
                # instruction_selection: fma.rn.ftz.f32; extent: one scalar

                if SCALE_STATE:
                    magnitude = abs(new_state)
                    # instruction_selection: abs.ftz.f32; extent: one scalar
                    new_state_max = max(new_state_max, magnitude)
                    # instruction_selection: max.ftz.f32; extent: one scalar
                    rNewState[iteration * STATE_VECTOR + e] = new_state
                elif PHILOX_ROUNDS > 0 and not SCALE_STATE:
                    random13 = bit_and(random_words[e % 4], 0x1FFF)
                    # instruction_selection: and.b32; extent: one 13-bit random operand
                    raw_sr_f16x2[e] = cast(
                        "raw_f16x2_u32", new_state,
                        rounding=("stochastic", random13, "zero_second_f32"))
                    # instruction_selection: cvt.rs.f16x2.f32 producing one full b32 result; extent: one FP32 value plus one zero dummy, with the desired FP16 value in the low 16 bits
                else:
                    rState[e] = cast(STATE_DTYPE, new_state, rounding="rn")
                    # instruction_selection: cvt.rn.bf16.f32, cvt.rn.f16.f32, or identity f32; extent: one scalar

                out_value = fma(new_state, C_value, out_value)
                # instruction_selection: mul.ftz.f32 for the first product when fused with the source-ordered lane seed, then fma.rn.ftz.f32 for accumulation; extent: one scalar accumulator

            if not SCALE_STATE and update_state and state_batch != pad_slot_id:
                if PHILOX_ROUNDS > 0:
                    packed_sr = reg_tile("u32", [STATE_VECTOR // 2])
                    for pair in static_range(STATE_VECTOR // 2):
                        packed_sr[pair] = prmt_b32(
                            raw_sr_f16x2[2 * pair],
                            raw_sr_f16x2[2 * pair + 1], selector=0x5410)
                        # instruction_selection: prmt.b32 with selector 0x5410; extent: one issue per FP16 pair (1 issue for STATE_VECTOR=2, 2 issues for STATE_VECTOR=4)
                    copy_r2g(
                        packed_sr,
                        word_view(dst_state_head[d, i : i + STATE_VECTOR], "u32"))
                    # instruction_selection: st.global.b32 for STATE_VECTOR=2 or st.global.v2.b32 for STATE_VECTOR=4; extent: one exact stochastic-FP16 PackedAligned tile
                else:
                    copy_r2g(rState, dst_state_head[d, i : i + STATE_VECTOR])
                    # instruction_selection: st.global.v2.b16 / three st.global.b16 / st.global.v4.b16 / st.global.v4.b32+packing for ordinary two-byte state, or st.global.v2.b32 / three st.global.b32 / st.global.v4.b32 for FP32; extent: one exact PackedAligned tile

        for delta in (16, 8, 4, 2, 1):
            peer_out, unused_pred = shuffle_down(
                out_value, delta, clamp=31, member_mask=0xFFFFFFFF)
            # instruction_selection: shfl.sync.down.b32 dst|pred, src, delta, 31, -1; extent: one scalar per lane, returned predicate unused
            out_value = add(out_value, peer_out)
            # instruction_selection: add.ftz.f32; extent: one scalar per lane
        if lane == 0:
            copy_r2s(out_value, sOut[local_row])
            # instruction_selection: st.shared.b32; extent: one row result

        if SCALE_STATE and update_state and state_batch != pad_slot_id:
            for delta in (16, 8, 4, 2, 1):
                peer_max, unused_pred = shuffle_down(
                    new_state_max, delta, clamp=31, member_mask=0xFFFFFFFF)
                # instruction_selection: shfl.sync.down.b32 dst|pred, src, delta, 31, -1; extent: one scalar per lane, returned predicate unused
                new_state_max = max(new_state_max, peer_max)
                # instruction_selection: max.ftz.f32; extent: one scalar per lane
            warp_sync(member_mask=0xFFFFFFFF)
            # instruction_selection: WARPSYNC.ALL in normal source SM100a SASS immediately before SHFL.IDX; extent: all 32 lanes. The source PTX represents the synchronization through the following shfl.sync.idx, while the concrete TIRx lowering must materialize the same SASS ordering edge when its compiler would otherwise emit a bare SHFL.IDX.
            new_state_max, unused_pred = shuffle_index(
                new_state_max, source_lane=0, clamp=31,
                member_mask=0xFFFFFFFF)
            # instruction_selection: shfl.sync.idx.b32 dst|pred, src, 0, 31, -1; extent: one scalar broadcast per lane, returned predicate unused
            encode_scale = 1.0
            # instruction_selection: mov.b32 1.0; extent: one default scalar
            if new_state_max != 0.0:
                # instruction_selection: setp.eq.ftz.f32 plus branch over the division when zero; extent: one scalar control edge
                encode_scale = div(32767.0, new_state_max)
                # instruction_selection: div.approx.ftz.f32; extent: one scalar, nonzero branch only
            new_decode_scale = div(1.0, encode_scale)
            # instruction_selection: rcp.approx.ftz.f32; extent: one scalar

            for iteration, i in lane_vector_range_with_iteration(
                    lane * STATE_VECTOR, DSTATE, 32 * STATE_VECTOR):
                quantized_s32 = reg_tile("i32", [STATE_VECTOR])
                for e in static_range(STATE_VECTOR):
                    scaled = mul(rNewState[iteration * STATE_VECTOR + e], encode_scale)
                    # instruction_selection: mul.ftz.f32; extent: one scalar
                    clipped_low = max(scaled, -32767.0)
                    # instruction_selection: max.ftz.f32; extent: one scalar
                    clipped = min(clipped_low, 32767.0)
                    # instruction_selection: min.ftz.f32; extent: one scalar
                    quantized_s32[e] = cast("i32", clipped, rounding="rni")
                    # instruction_selection: cvt.rni.ftz.s32.f32; extent: one scalar whose low 16 bits are the int16 result
                packed_i16 = reg_tile("u32", [STATE_VECTOR // 2])
                for pair in static_range(STATE_VECTOR // 2):
                    packed_i16[pair] = prmt_b32(
                        quantized_s32[2 * pair], quantized_s32[2 * pair + 1],
                        selector=0x5410)
                    # instruction_selection: prmt.b32 with selector 0x5410; extent: one issue per int16 pair (1/2/4 issues for STATE_VECTOR 2/4/8)
                copy_r2g(
                    packed_i16,
                    word_view(dst_state_head[d, i : i + STATE_VECTOR], "u32"))
                # instruction_selection: st.global.b32 / st.global.v2.b32 / st.global.v4.b32 for STATE_VECTOR 2/4/8; extent: one exact PackedAligned int16 tile

            if lane == 0:
                copy_r2s(new_decode_scale, sScale[local_row])
                # instruction_selection: st.shared.b32; extent: one row scale

    cta_sync()
    # instruction_selection: bar.sync 0; extent: all 128 threads before output/scale publication

    # =======================================================================
    # Source-order output and optional destination-scale epilogue
    # =======================================================================

    for local_lane_row in lane_strided_range(0, rowsPerWarp, 32):
        local_row = warp * rowsPerWarp + local_lane_row
        d = dim_offset + local_row
        if d < DIM:
            out_value = copy_s2r(sOut[local_row])
            # instruction_selection: ld.shared.b32; extent: one row result
            if z is present:
                z_value = copy_s2r(sZ[local_row])
                # instruction_selection: ld.shared.b16; extent: one scalar
                z_value = cast("f32", z_value)
                # instruction_selection: cvt.f32.bf16; extent: one scalar
                neg_z = sub(0.0, z_value)
                # instruction_selection: sub.ftz.f32; extent: one scalar
                exp_arg = mul(neg_z, LOG2_E)
                # instruction_selection: mul.ftz.f32; extent: one scalar
                exp_neg_z = exp2(exp_arg)
                # instruction_selection: ex2.approx.ftz.f32; extent: one scalar
                denominator = add(1.0, exp_neg_z)
                # instruction_selection: add.ftz.f32; extent: one scalar
                sigmoid_z = div(1.0, denominator)
                # instruction_selection: div.approx.ftz.f32; extent: one scalar
                silu_z = mul(z_value, sigmoid_z)
                # instruction_selection: mul.ftz.f32; extent: one scalar
                out_value = mul(out_value, silu_z)
                # instruction_selection: mul.ftz.f32; extent: one scalar
            output_value = cast("bf16", out_value, rounding="rn")
            # instruction_selection: cvt.rn.bf16.f32; extent: one scalar
            copy_r2g(
                output_value,
                output[batch_i * out_stride_batch + head * DIM + d],
            )
            # instruction_selection: st.global.b16; extent: one guarded row result

    if SCALE_STATE and update_state and state_batch != pad_slot_id:
        for local_lane_row in lane_strided_range(0, rowsPerWarp, 32):
            local_row = warp * rowsPerWarp + local_lane_row
            d = dim_offset + local_row
            if d < DIM:
                new_scale = copy_s2r(sScale[local_row])
                # instruction_selection: ld.shared.b32; extent: one row scale
                copy_r2g(new_scale, dst_scale_rows[d])
                # instruction_selection: st.global.b32; extent: one row scale
```

## Static specialization and launch boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| DIM, DSTATE, state/weight/index dtype, scale-state, Philox rounds | static | loop unrolling, register arrays, conversion and vector instruction families specialize |
| `ROWS_PER_BLOCK` | static 4 or DIM | shared footprint, row ownership and grid Z specialize |
| batch and nheads grid extents | static for TIRx launch | exactly match the selected benchmark shape |
| nheads/ngroups scalar values | runtime kernel values | preserves source integer group division |
| tensor strides, optional-pointer flags, softplus/update/pad | runtime or host-specialized pointer presence | preserves source branches and state alias behavior |

The host chooses `ROWS_PER_BLOCK=4` iff `batch*nheads < 2*num_sms`, matching
FlashInfer. Every benchmark and correctness call uses the explicit simple source
algorithm and validates that this predicate selected the intended specialization.

## TIRx module and benchmark contract

- `KERNEL_META` names `selective_state_update_stp_simple`, category
  `flashinfer`, compute capability 10.
- All optional dependencies are lazy. The frozen CUDA header SHA256 is
  `c0e13b64bf42f4f8155058dc9f5877f7aca90832f50a1e7602863894908e89fd`.
- `CONFIGS` covers the upstream shape/dtype/feature rows; `BENCH_CONFIGS` is the
  branch-representative matrix. Every timed source launch uses
  `algorithm="simple"`.
- TIRx and source receive independent mutable state/output/scale buffers.
  Compilation, allocations, source JIT, warmups, and preflight validation occur
  outside timed closures.
- The implementation and every pre-dispatch specialization contain no tile
  primitives.

## Instruction-selection summary

- Each CTA executes four warp-specialized preload programs, two `bar.sync 0`
  sites, and one warp-per-row DSTATE computation. Each updated int16 row also
  executes the source-SASS `WARPSYNC.ALL` immediately before its max broadcast.
- Each active lane traverses DSTATE using the exact `PackedAligned` width:
  2/3/4/8 elements for two-byte state at DSTATE 64/96/128/256 and 2/3/4/4 for
  FP32 state.
- Every row performs five `shfl.sync.down` sum steps. Int16 state additionally
  performs five max shuffles, one lane-0 broadcast, a register-resident second
  pass, clamping, round-to-nearest integer conversion, and one FP32 scale store.
- Philox state performs ten unrolled multiply-high/multiply/xor/add rounds per
  four state elements and selects SM100a `cvt.rs.f16x2.f32` for stochastic FP16
  conversion.
- The output epilogue is one BF16 store per row, preceded by the explicit
  `exp`/reciprocal/multiply SILU path only when z is present.
