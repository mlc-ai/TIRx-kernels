<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

This design sketch documents a modified TIRx port of FlashInfer's
csrc/tinygemm2_sm100.cu, whose generated Loom schedules are exact ports of
NVIDIA TensorRT-LLM's TinyGEMM2 kernel. See NOTICE and
licenses/ for upstream attribution.
-->

# TinyGEMM2 BF16 SM100: coarse WASP pipeline sketch

This non-executable design sketch describes the storage layout, warp roles,
pipeline, synchronization, and PTX-level operations of
[`tirx_kernels/flashinfer/gemm/tinygemm2_sm100.py`](../../tirx_kernels/flashinfer/gemm/tinygemm2_sm100.py).
That TIRx module is the authoritative implementation.

The four static specializations are `STAGES in {4,8}` crossed with
`USE_PDL in {False,True}`. The accepted target is SM100/B200, and the only
supported path is contiguous BF16 activation, weight, bias, and output with a
present bias. `M=O`, `N=B`, `K`, their loop bounds, and all output guards stay
runtime values. Bias-free execution, SM103 validation, and tile (`Tx`)
primitives are out of scope.

## Pipeline at a glance

| Warps | Role-local program | Publication/reuse edges |
| --- | --- | --- |
| 0..3 | Each warp accumulates one 256-wide quarter of every runtime 1024-wide K-loop using 16 BF16 MMAs; all four warp fragments are then reduced through shared memory. | wait `wt_ready[stage]` and `act_ready[stage]`; arrive `data_consumed[stage]` once per K-loop |
| 4..7 | One elected lane per warp loads one 256-wide weight quarter as four 64x16 TMA boxes. | wait prior `data_consumed[stage]`; expect 8192 bytes on `wt_ready[stage]` |
| 8..11 | One elected lane per warp loads one 256-wide activation quarter as four 64x8 TMA boxes; PDL instructions occur here before the loop. | wait prior `data_consumed[stage]`; expect 4096 bytes on `act_ready[stage]` |
| all 12 | Meet at named CTA barrier 2 after role work; compute warp 0 then performs reduction, bias addition, BF16 conversion, and guarded stores. | `bar.sync 2, 384` |

Stage 4 assigns one ring slot permanently to each compute/loader warp and
toggles parity every K-loop. Stage 8 assigns two alternating slots to each
warp and toggles parity every two K-loops.

## Primitive vocabulary

Structural operations declare placement without moving data:

```python
specialize(...)       # compile-time variant selection
launch(...)           # compile-time launch topology and attributes
tile(...)             # storage declaration with dtype, shape, and placement
view(...)             # typed/indexed view of storage; no copy
reg_tile(...)         # role-local register fragment
```

Copies state their direction:

```python
copy_g2s(src, dst, completion)  # TMA global -> shared, mbarrier completion
copy_g2r(src, dst)              # scalar global -> register
copy_s2r(src, dst)              # shared -> register, including ldmatrix
copy_r2s(src, dst)              # register -> shared
copy_r2g(src, dst)              # register -> global
```

The compute vocabulary is deliberately primitive:

```python
fill(dst, value)
add(dst, lhs, rhs, ftz=None)
add_i32(dst, lhs, rhs)
cast(dst, src, rounding=None)
gemm(dst, lhs, rhs, accumulate=True)
move(dst, src)
move_if(dst, src, predicate)
shift_right(dst, src, bits, signed)
set_predicate_ne(lhs, rhs)
elect_predicate(active_mask)
bitcast_move(dst_b32, src_f32)
host_assert(predicate)
```

`thread_id`, `shuffle_index`, `lane_id`, `cta_id`,
`prefetch`, `init`, `wait`, `expect_bytes`, `arrive`, `fence`, `cta_sync`,
`barrier`, `pdl_wait`, and `pdl_launch_dependents` are schedule or host-wrapper
operations.
Address expressions, stage/phase expressions, and guards are shown directly;
they do not hide copies, computation, role changes, or synchronization.

## Complete sketch

```python
# ===========================================================================
# Static specialization, runtime ABI, and launch
# ===========================================================================

variant = specialize(STAGES=(4, 8), USE_PDL=(False, True), target="sm_100a")
# instruction_selection: none; extent: four compile-time specializations

GRID_X = (specialized_O + 15) // 16
GRID_Y = (specialized_B + 7) // 8
launch_config = launch(
    grid=(GRID_X, GRID_Y, 1),
    block=(384, 1, 1),
    min_blocks_per_sm=1,
    dynamic_smem_bytes=(52352 if STAGES == 4 else 101504),
    programmatic_dependent_launch=USE_PDL,
)
# instruction_selection: none; extent: static launch metadata, including specialized grid extents

def tinygemm2_sm100(
    tmap_wt,       # by-value 128-byte TensorMap, first grid-constant argument
    tmap_act,      # by-value 128-byte TensorMap, second grid-constant argument
    output,        # bf16 [N,M], direct global pointer
    bias,          # bf16 [M], direct global pointer; always present
    M_i32,         # runtime O
    N_i32,         # runtime B
    K_i32,         # runtime reduction extent
):
    tid_u32 = thread_id(extent=384, dtype="uint32")
    # instruction_selection: mov.u32 from %tid.x; extent: scalar per thread
    tid = cast("i32", tid_u32)
    # instruction_selection: none; extent: signed view sharing the thread-id register
    warp_raw = shift_right("u32", tid, bits=5, signed=False)
    # instruction_selection: shr.u32; extent: scalar per thread
    warp = shuffle_index(warp_raw, src_lane=0, clamp=0x1F,
                         member_mask=0xFFFFFFFF)
    # instruction_selection: shfl.sync.idx.b32; extent: scalar per warp lane
    # generated_code_note: the normal generated-code prelude also emits one unused
    # warp_id_in_cta shfl.sync.idx.b32 PTX site; ptxas removes it, so final
    # SASS contains only the semantic shuffle above.
    lane = lane_id(tid % 32)
    # instruction_selection: integer and.b32; extent: scalar per thread
    lane_u32 = lane_id(tid_u32 % u32(32))
    # instruction_selection: none; extent: unsigned view coalesced with lane extraction

    # =======================================================================
    # Exact dynamic shared-memory layout and lifetimes
    # =======================================================================

    smem_raw = tile("shared", "u8", [52352 if STAGES == 4 else 101504],
                    byte_offset=0, requested_alignment=1024)
    # instruction_selection: none; extent: one compile-time-sized dynamic-SMEM allocation
    # emitted_alignment: the current normal module declares the dynamic shared
    # symbol with 64-byte alignment; 1024 is the TIRx pool allocation request,
    # not an emitted-symbol alignment guarantee.
    wt_ready = view(smem_raw, "mbarrier.b64", [STAGES], byte_offset=0)
    # instruction_selection: none; extent: STAGES barrier slots
    act_ready = view(smem_raw, "mbarrier.b64", [STAGES],
                     byte_offset=(32 if STAGES == 4 else 64))
    # instruction_selection: none; extent: STAGES barrier slots
    data_consumed = view(smem_raw, "mbarrier.b64", [STAGES],
                         byte_offset=(64 if STAGES == 4 else 128))
    # instruction_selection: none; extent: STAGES barrier slots
    smem_wt = view(smem_raw, "bf16", [STAGES, 4, 16, 64], byte_offset=1024,
                   layout="row_stride_128B_xor_col_by_row_mod_8")
    # instruction_selection: none; extent: 32768 or 65536 shared bytes
    smem_act = view(smem_raw, "bf16", [STAGES, 4, 8, 64],
                    byte_offset=(33792 if STAGES == 4 else 66560),
                    layout="row_stride_128B_xor_col_by_row_mod_8")
    # instruction_selection: none; extent: 16384 or 32768 shared bytes
    smem_red = view(smem_raw, "f32", [128, 4],
                    byte_offset=(50176 if STAGES == 4 else 99328))
    # instruction_selection: none; extent: 2048 shared bytes, compute lifetime after final MMA
    smem_bias = view(smem_raw, "bf16", [16],
                     byte_offset=(52224 if STAGES == 4 else 101376))
    # instruction_selection: none; extent: 32 active bytes; total allocation retains required padding

    # =======================================================================
    # TensorMap setup and mbarrier initialization
    # =======================================================================

    if tid == 0:
        prefetch(tmap_wt)
        # instruction_selection: prefetch.tensormap; extent: one by-value weight descriptor
        prefetch(tmap_act)
        # instruction_selection: prefetch.tensormap; extent: one by-value activation descriptor

    if warp == 0:
        elect_mask = move("b32", -1)
        # instruction_selection: mov.b32; extent: one active-mask register per warp-0 lane
        leader = move("b32", 0)
        # instruction_selection: mov.b32; extent: one zero result register per warp-0 lane
        elected_pred = elect_predicate(active_mask=elect_mask)
        # instruction_selection: elect.sync; extent: one predicate in warp 0
        move_if(leader, 1, predicate=elected_pred)
        # instruction_selection: predicated mov.s32; extent: elected warp-0 lane only
        for stage_init in static_range(STAGES):
            leader_nonzero = set_predicate_ne(leader, 0)
            # instruction_selection: setp.ne.b32; extent: one predicate for this wt-ready slot
            init(wt_ready[stage_init], arrival_count=1, predicate=leader_nonzero)
            # instruction_selection: mbarrier.init.shared::cta.b64, predicated; extent: one slot
        for stage_init in static_range(STAGES):
            leader_nonzero = set_predicate_ne(leader, 0)
            # instruction_selection: setp.ne.b32; extent: one predicate for this act-ready slot
            init(act_ready[stage_init], arrival_count=1, predicate=leader_nonzero)
            # instruction_selection: mbarrier.init.shared::cta.b64, predicated; extent: one slot
        for stage_init in static_range(STAGES):
            leader_nonzero = set_predicate_ne(leader, 0)
            # instruction_selection: setp.ne.b32; extent: one predicate for this consumed slot
            init(data_consumed[stage_init], arrival_count=32, predicate=leader_nonzero)
            # instruction_selection: mbarrier.init.shared::cta.b64, predicated; extent: one slot
        fence(scope="mbarrier_init", order="release", visibility="cluster")
        # instruction_selection: fence.mbarrier_init.release.cluster; extent: all 32 lanes of warp 0

    cta_sync()
    # instruction_selection: bar.sync 0; extent: all 384 threads, first CTA synchronization
    cta_sync()
    # instruction_selection: bar.sync 0; extent: all 384 threads, second CTA synchronization

    # =======================================================================
    # Role 1: compute warps 0..3
    # =======================================================================

    if warp <= 3:
        k_plus_1023 = add_i32(K_i32, 1023)
        # instruction_selection: add.s32; extent: runtime scalar in the compute role
        k_sign = shift_right("i32", k_plus_1023, bits=31, signed=True)
        # instruction_selection: shr.s32; extent: runtime scalar in the compute role
        k_correction = shift_right("u32", k_sign, bits=22, signed=False)
        # instruction_selection: shr.u32; extent: runtime scalar in the compute role
        k_corrected = add_i32(k_plus_1023, k_correction)
        # instruction_selection: add.s32; extent: runtime scalar in the compute role
        k_loops = shift_right("i32", k_corrected, bits=10, signed=True)
        # instruction_selection: shr.s32; extent: runtime scalar in the compute role
        cta_m = cta_id(axis="x", extent=GRID_X)
        # instruction_selection: mov.u32 from %ctaid.x; extent: scalar per compute thread
        mib = cta_m * 16
        cta_n = cta_id(axis="y", extent=GRID_Y)
        # instruction_selection: mov.u32 from %ctaid.y; extent: scalar per compute thread
        ni = cta_n * 8

        # Only tid 0..15 satisfy this inside the compute role, so warp 0
        # publishes the complete 16-element bias tile before the later named
        # CTA barrier. M is constrained to a multiple of 16.
        if tid < 16:
            bias_value = copy_g2r(bias[mib + tid])
            # instruction_selection: ld.global.nc.b16; extent: one bf16 per participating lane
            copy_r2s(bias_value, smem_bias[tid])
            # instruction_selection: st.shared.b16; extent: one bf16 per participating lane

        accum = reg_tile("f32", [4])
        # instruction_selection: none; extent: four f32 accumulator registers per compute lane
        fill(accum, 0.0)
        # instruction_selection: mov.b32 zero; extent: four scalar registers per compute lane

        lane_div8 = lane_u32 // 8
        lane_mod8 = lane_u32 % 8
        row_wt = lane_mod8 + (lane_div8 % 2) * 8
        col_off_wt = lane_div8 // 2
        row_act = lane_mod8

        for ki in runtime_range(k_loops, dtype="uint32", unroll_hint=2):
            if STAGES == 4:
                stage_c = warp
                phase_c = ki & 1
            else:
                stage_c = warp + 4 * (ki % 2)
                phase_c = (ki // 2) & 1

            wait(wt_ready[stage_c], parity=phase_c, scope="cta", order="acquire")
            # instruction_selection: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 loop; extent: one selected stage
            wait(act_ready[stage_c], parity=phase_c, scope="cta", order="acquire")
            # instruction_selection: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 loop; extent: one selected stage

            for subtile in static_range(4):
                base_wt = smem_wt[stage_c, subtile]
                base_act = smem_act[stage_c, subtile]
                for kii in static_range(4):
                    a_frag = reg_tile("b32", [4])
                    # instruction_selection: none; extent: four packed-bf16 registers per compute lane
                    b_frag = reg_tile("b32", [2])
                    # instruction_selection: none; extent: two packed-bf16 registers per compute lane

                    col_w = 2 * kii + col_off_wt
                    col_sw_w = (row_wt % 8) ^ col_w
                    copy_s2r(
                        base_wt.byte(row_wt * 128 + col_sw_w * 16), a_frag,
                        layout="m8n8_x4_b16",
                    )
                    # instruction_selection: ldmatrix.sync.aligned.m8n8.x4.shared.b16; extent: one x4 matrix fragment

                    col_a = 2 * kii + lane_div8
                    col_sw_a = (row_act % 8) ^ col_a
                    copy_s2r(
                        base_act.byte(row_act * 128 + col_sw_a * 16), b_frag,
                        layout="m8n8_x2_b16",
                    )
                    # instruction_selection: ldmatrix.sync.aligned.m8n8.x2.shared.b16; extent: one x2 matrix fragment

                    gemm(
                        accum, a_frag, b_frag, accumulate=True,
                        shape="m16n8k16", layouts="row.col",
                        dtypes="f32.bf16.bf16.f32",
                    )
                    # instruction_selection: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32; extent: one MMA

            fence(scope="proxy_async_shared", visibility="cta")
            # instruction_selection: fence.proxy.async.shared::cta; extent: one per compute warp per K-loop
            arrive(data_consumed[stage_c], scope="cta", order="release")
            # instruction_selection: mbarrier.arrive.release.cta.shared::cta.b64; extent: one arrival per lane, 32 total

        accum_bits = reg_tile("b32", [4])
        # instruction_selection: none; extent: four bit-preserving store operand registers
        for z in static_range(4):
            bitcast_move(accum_bits[z], accum[z])
            # instruction_selection: none; extent: one same-register f32-to-b32 logical view
        copy_r2s(accum_bits, smem_red[tid_u32, 0:4], vector="v4.b32")
        # instruction_selection: st.shared.v4.b32; extent: 16 bytes per compute lane, 128 lanes
        barrier(barrier_id=2, arrival_count=384)
        # instruction_selection: bar.sync 2, 384; extent: all 12 role warps

        if warp == 0:
            part = reg_tile("f32", [3, 4])
            # instruction_selection: none; extent: twelve f32 registers per warp-0 lane
            for other_warp in static_range(1, 4):
                copy_s2r(
                    smem_red[other_warp * 32 + tid_u32, 0:4],
                    part[other_warp - 1, 0:4], vector="v4.b32",
                )
                # instruction_selection: ld.shared.v4.b32; extent: 16 bytes per warp-0 lane

            # Keep the reduction left-associative:
            # own + warp1 + warp2 + warp3 for each fragment position.
            for z in static_range(4):
                add(accum[z], accum[z], part[0, z], ftz=True)
                # instruction_selection: add.ftz.f32; extent: one scalar accumulator
                add(accum[z], accum[z], part[1, z], ftz=True)
                # instruction_selection: add.ftz.f32; extent: one scalar accumulator
                add(accum[z], accum[z], part[2, z], ftz=True)
                # instruction_selection: add.ftz.f32; extent: one scalar accumulator

            tm = mib + lane // 4
            tn = ni + 2 * (lane % 4)
            bias_lo_bf16 = copy_s2r(smem_bias[lane // 4])
            # instruction_selection: ld.shared.b16; extent: one scalar
            bias_lo = cast("f32", bias_lo_bf16)
            # instruction_selection: cvt.f32.bf16; extent: one scalar
            bias_hi_bf16 = copy_s2r(smem_bias[lane // 4 + 8])
            # instruction_selection: ld.shared.b16; extent: one scalar
            bias_hi = cast("f32", bias_hi_bf16)
            # instruction_selection: cvt.f32.bf16; extent: one scalar
            o00 = reg_tile("f32", [1])
            # instruction_selection: none; extent: one scalar register
            o01 = reg_tile("f32", [1])
            # instruction_selection: none; extent: one scalar register
            o10 = reg_tile("f32", [1])
            # instruction_selection: none; extent: one scalar register
            o11 = reg_tile("f32", [1])
            # instruction_selection: none; extent: one scalar register
            add(o00, accum[0], bias_lo, ftz=True)
            # instruction_selection: add.ftz.f32; extent: one scalar
            add(o01, accum[1], bias_lo, ftz=True)
            # instruction_selection: add.ftz.f32; extent: one scalar
            add(o10, accum[2], bias_hi, ftz=True)
            # instruction_selection: add.ftz.f32; extent: one scalar
            add(o11, accum[3], bias_hi, ftz=True)
            # instruction_selection: add.ftz.f32; extent: one scalar

            if tn < N_i32:
                if tm < M_i32:
                    out00 = cast("bf16", o00, rounding="rn")
                    # instruction_selection: cvt.rn.bf16.f32; extent: one scalar
                    copy_r2g(out00, output[tn, tm])
                    # instruction_selection: st.global.b16; extent: one guarded scalar
            if tn + 1 < N_i32:
                if tm < M_i32:
                    out01 = cast("bf16", o01, rounding="rn")
                    # instruction_selection: cvt.rn.bf16.f32; extent: one scalar
                    copy_r2g(out01, output[tn + 1, tm])
                    # instruction_selection: st.global.b16; extent: one guarded scalar
            if tn < N_i32:
                if tm + 8 < M_i32:
                    out10 = cast("bf16", o10, rounding="rn")
                    # instruction_selection: cvt.rn.bf16.f32; extent: one scalar
                    copy_r2g(out10, output[tn, tm + 8])
                    # instruction_selection: st.global.b16; extent: one guarded scalar
            if tn + 1 < N_i32:
                if tm + 8 < M_i32:
                    out11 = cast("bf16", o11, rounding="rn")
                    # instruction_selection: cvt.rn.bf16.f32; extent: one scalar
                    copy_r2g(out11, output[tn + 1, tm + 8])
                    # instruction_selection: st.global.b16; extent: one guarded scalar

    # =======================================================================
    # Role 2: weight-loader warps 4..7
    # =======================================================================

    elif 4 <= warp <= 7:
        k_plus_1023 = add_i32(K_i32, 1023)
        # instruction_selection: add.s32; extent: runtime scalar in the weight-loader role
        k_sign = shift_right("i32", k_plus_1023, bits=31, signed=True)
        # instruction_selection: shr.s32; extent: runtime scalar in the weight-loader role
        k_correction = shift_right("u32", k_sign, bits=22, signed=False)
        # instruction_selection: shr.u32; extent: runtime scalar in the weight-loader role
        k_corrected = add_i32(k_plus_1023, k_correction)
        # instruction_selection: add.s32; extent: runtime scalar in the weight-loader role
        k_loops = shift_right("i32", k_corrected, bits=10, signed=True)
        # instruction_selection: shr.s32; extent: runtime scalar in the weight-loader role
        cta_m = cta_id(axis="x", extent=GRID_X)
        # instruction_selection: mov.u32 from %ctaid.x; extent: scalar per weight-loader thread
        mib = cta_m * 16
        wslot = warp % 4
        elect_mask = move("b32", -1)
        # instruction_selection: mov.b32; extent: one active-mask register per weight-loader lane
        leader = move("b32", 0)
        # instruction_selection: mov.b32; extent: one zero result register per weight-loader lane
        elected_pred = elect_predicate(active_mask=elect_mask)
        # instruction_selection: elect.sync; extent: one predicate in each weight-loader warp
        move_if(leader, 1, predicate=elected_pred)
        # instruction_selection: predicated mov.s32; extent: elected weight-loader lane only
        if leader:
            for ki in runtime_range(k_loops, dtype="uint32", unroll_hint=1):
                if STAGES == 4:
                    stage_w = wslot
                    phase_w = ki & 1
                else:
                    stage_w = wslot + 4 * (ki % 2)
                    phase_w = (ki // 2) & 1
                k_base_w = cast("i32", (ki * u32(4) + wslot) * u32(256))
                # instruction_selection: none; extent: one same-width TMA-coordinate view

                wait(data_consumed[stage_w], parity=phase_w ^ 1,
                     scope="cta", order="acquire")
                # instruction_selection: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 loop; extent: one selected stage
                expect_bytes(wt_ready[stage_w], bytes=8192,
                             scope="cta", order="release")
                # instruction_selection: mbarrier.arrive.expect_tx.release.cta.shared::cta.b64; extent: one producer transaction group
                for box in static_range(4):
                    copy_g2s(
                        tmap_wt.coord(k_base_w + box * 64, mib),
                        smem_wt[stage_w, box, 0:16, 0:64],
                        completion=wt_ready[stage_w],
                    )
                    # instruction_selection: cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes; extent: one 64x16 bf16 TMA box (2048 bytes)

            if STAGES == 4:
                # The stage-4 drain range has no reachable wait.
                drain = specialize(reachable=False, iter_range="di=0 only")
                # instruction_selection: none; extent: compile-time-eliminated stage4 drain
            else:
                # The stage-8 drain waits only at di=0.
                dki_w = k_loops
                dstage_w = wslot + 4 * (dki_w % 2)
                dphase_w = (dki_w // 2) & 1
                wait(data_consumed[dstage_w], parity=dphase_w ^ 1,
                     scope="cta", order="acquire")
                # instruction_selection: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 loop; extent: one stage8 drain wait

        barrier(barrier_id=2, arrival_count=384)
        # instruction_selection: bar.sync 2, 384; extent: all 12 role warps

    # =======================================================================
    # Role 3: activation-loader warps 8..11
    # =======================================================================

    elif 8 <= warp <= 11:
        k_plus_1023 = add_i32(K_i32, 1023)
        # instruction_selection: add.s32; extent: runtime scalar in the activation-loader role
        k_sign = shift_right("i32", k_plus_1023, bits=31, signed=True)
        # instruction_selection: shr.s32; extent: runtime scalar in the activation-loader role
        k_correction = shift_right("u32", k_sign, bits=22, signed=False)
        # instruction_selection: shr.u32; extent: runtime scalar in the activation-loader role
        k_corrected = add_i32(k_plus_1023, k_correction)
        # instruction_selection: add.s32; extent: runtime scalar in the activation-loader role
        k_loops = shift_right("i32", k_corrected, bits=10, signed=True)
        # instruction_selection: shr.s32; extent: runtime scalar in the activation-loader role
        cta_n = cta_id(axis="y", extent=GRID_Y)
        # instruction_selection: mov.u32 from %ctaid.y; extent: scalar per activation-loader thread
        ni = cta_n * 8
        aslot = warp % 4
        elect_mask = move("b32", -1)
        # instruction_selection: mov.b32; extent: one active-mask register per activation-loader lane
        leader = move("b32", 0)
        # instruction_selection: mov.b32; extent: one zero result register per activation-loader lane
        elected_pred = elect_predicate(active_mask=elect_mask)
        # instruction_selection: elect.sync; extent: one predicate in each activation-loader warp
        move_if(leader, 1, predicate=elected_pred)
        # instruction_selection: predicated mov.s32; extent: elected activation-loader lane only
        if leader:
            if USE_PDL:
                pdl_wait()
                # instruction_selection: griddepcontrol.wait; extent: one elected lane in each of four activation-loader warps
                pdl_launch_dependents()
                # instruction_selection: griddepcontrol.launch_dependents; extent: one elected lane in each of four activation-loader warps

            for ki in runtime_range(k_loops, dtype="uint32", unroll_hint=1):
                if STAGES == 4:
                    stage_a = aslot
                    phase_a = ki & 1
                else:
                    stage_a = aslot + 4 * (ki % 2)
                    phase_a = (ki // 2) & 1
                k_base_a = cast("i32", (ki * u32(4) + aslot) * u32(256))
                # instruction_selection: none; extent: one same-width TMA-coordinate view

                wait(data_consumed[stage_a], parity=phase_a ^ 1,
                     scope="cta", order="acquire")
                # instruction_selection: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 loop; extent: one selected stage
                expect_bytes(act_ready[stage_a], bytes=4096,
                             scope="cta", order="release")
                # instruction_selection: mbarrier.arrive.expect_tx.release.cta.shared::cta.b64; extent: one producer transaction group
                for box in static_range(4):
                    copy_g2s(
                        tmap_act.coord(k_base_a + box * 64, ni),
                        smem_act[stage_a, box, 0:8, 0:64],
                        completion=act_ready[stage_a],
                    )
                    # instruction_selection: cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes; extent: one 64x8 bf16 TMA box (1024 bytes), with descriptor-defined OOB zero fill

            if STAGES == 4:
                # The stage-4 drain range has no reachable wait.
                drain = specialize(reachable=False, iter_range="di=0 only")
                # instruction_selection: none; extent: compile-time-eliminated stage4 drain
            else:
                # The stage-8 drain waits only at di=0.
                dki_a = k_loops
                dstage_a = aslot + 4 * (dki_a % 2)
                dphase_a = (dki_a // 2) & 1
                wait(data_consumed[dstage_a], parity=dphase_a ^ 1,
                     scope="cta", order="acquire")
                # instruction_selection: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 loop; extent: one stage8 drain wait

        barrier(barrier_id=2, arrival_count=384)
        # instruction_selection: bar.sync 2, 384; extent: all 12 role warps

    # The mutually exclusive roles require no trailing cleanup operation.
```

## Host wrapper and validation

The current Python module and generated host wrapper perform the following
host-only work; none of it emits device PTX:

```python
def validate_problem(B, O, K):
    host_assert(B > 0)
    host_assert(K >= 64)
    host_assert(O >= 16 and O % 16 == 0)
    host_assert(B <= INT32_MAX and O <= INT32_MAX and K <= INT32_MAX)
    # instruction_selection: none; extent: four scalar problem checks

def require_sm100():
    host_assert(torch.cuda.is_available())
    host_assert(torch.cuda.get_device_capability() == (10, 0))
    # instruction_selection: none; extent: availability and target-scope checks

def prepare_data(B, O, K, device="cuda"):
    validate_problem(B, O, K)
    host_assert(torch.device(device).type == "cuda" and torch.cuda.is_available())
    input = contiguous_bf16_tensor(shape=(B, K))
    weight = contiguous_bf16_tensor(shape=(O, K))
    bias = contiguous_bf16_tensor(shape=(O,))
    output = contiguous_bf16_zeros(shape=(B, O))
    tmap_wt = encode_tensor_map(weight, dims=(K, O), stride_bytes=(K * 2,),
                                box=(64, 16))
    tmap_act = encode_tensor_map(input, dims=(K, B), stride_bytes=(K * 2,),
                                 box=(64, 8))
    # instruction_selection: none; extent: known tensor construction and two descriptor encodes

launch_args = (
    opaque_tensor_map_handle(tmap_wt),
    opaque_tensor_map_handle(tmap_act),
    output.view(-1),
    bias,
    O,
    B,
    K,
)
# instruction_selection: none; extent: flat seven-argument launch ABI

# The generated wrapper checks both opaque TensorMap handles. For output and
# bias it checks tensor-handle type, CUDA device type, scalar BF16 dtype,
# compactness, non-null data, zero byte offset, and a shared device id.
host_assert(output.ndim == 1 and output.shape[0] == B * O)
host_assert(bias.ndim == 1 and bias.shape[0] == O)
# instruction_selection: none; extent: flat output and bias wrapper validation
```

## TensorMap fields

| ABI parameter | Backing tensor | rank / global dims | global strides in bytes | box / element strides | interleave | swizzle | L2 promotion | float OOB fill |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `tmap_wt` | contiguous `weight[O,K]` | 2 / `(K,O)` | `(K*2,)` | `(64,16)` / `(1,1)` | none | 128 B | none | none |
| `tmap_act` | contiguous `input[B,K]` | 2 / `(K,B)` | `(K*2,)` | `(64,8)` / `(1,1)` | none | 128 B | none | none |

Coordinates are `(k_start, output_tile_start)` for weight and
`(k_start, batch_tile_start)` for activation. Hardware tensor-boundary handling
zero-fills integer-coordinate OOB elements for both the non-64 K tail and
activation rows beyond runtime `N`; the floating-point OOB-fill enum remains
`NONE`, matching the TensorMap encoder configuration.

## Static specialization boundary

| Fact | Static or runtime | Consequence |
| --- | --- | --- |
| `STAGES` | static 4 or 8 | ring offsets, shared footprint, stage/phase arithmetic, and drain shape specialize |
| `USE_PDL` | static bool | emits or removes the two grid dependency instructions and launch attribute |
| grid X/Y extents | static per requested shape | launch is exactly `(ceil(O/16),ceil(B/8))` |
| `M`, `N`, `K` ABI values | runtime | K-loop and all four epilogue guards remain in pre-dispatch IR |
| BF16/FP32 types, 384 threads, SM100a | static | preserves MMA, launch bounds, and resource contract |

Automatic selection is outside the kernel: choose stage 4 iff
`K <= 1024 or ceil(O/16) * ceil(B/8) > 2 * num_sms`; otherwise choose stage 8.
Direct `stage=4|8` overrides exist only for variant tests. Benchmarks always
use automatic stage and `USE_PDL=False`.

## TIRx module and validation contract

- `KERNEL_META = {"name": "tinygemm2_sm100", "category": "flashinfer",
  "compute_capability": 10}`.
- The executable kernel is expressed entirely in TIRx: warp uniformization uses
  a u32 `T.thread_id` plus one shared i32 view, followed by
  `T.cuda._shfl_sync`; election uses the pred-only lowering of
  `T.cuda.elect_sync`, spin waits use the existing
  `T.cuda.mbarrier_wait` acquire-CTA helper, and the compute loop uses
  `T.serial(..., unroll=2, dtype="uint32")`. The two loader loops also use u32
  induction variables; all three runtime trip-count expressions remain i32.
  Compilation uses only TIRx lowering; no text postprocessor or callback is
  installed.
- `CONFIGS` and `BENCH_CONFIGS` contain the eight configured parity shapes.
- `get_kernel`, `prepare_data`, `run_test`, and `run_bench` are public.
- The TIRx kernel and all four pre-dispatch specializations must contain no `Tx`
  tile primitives.
- The timed implementation is named `tirx`. TensorMap encoding, compilation,
  allocation, and correctness checks stay outside timing.

## Instruction-selection summary

- Per compute warp and runtime K-loop: two acquire mbarrier waits, sixteen
  `ldmatrix.x4`, sixteen `ldmatrix.x2`, sixteen
  `mma.sync.m16n8k16.bf16`, one async-proxy fence, and 32 consumed arrivals.
- Per loader warp and runtime K-loop: one reuse wait, one expect-tx arrival,
  and four 2-D TMA copies. Weight copies publish 8192 bytes; activation copies
  publish 4096 bytes.
- Each compute lane writes one shared `v4.b32`. Warp 0 loads three shared
  `v4.b32` vectors, performs the left-associative FP32 reduction and four bias
  adds, converts four scalars with round-to-nearest BF16, and executes four
  independent runtime-guarded global stores.
- PDL adds exactly one static `griddepcontrol.wait` site and one static
  `griddepcontrol.launch_dependents` site in the activation-loader leader
  branch; dynamically, each site is executed by one elected lane in each of
  four activation-loader warps.
