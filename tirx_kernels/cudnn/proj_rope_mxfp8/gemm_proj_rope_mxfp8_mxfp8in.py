# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""MXFP8 projection GEMM with YARN RoPE and dual-direction MXFP8 outputs.

Upstream source:
``python/cudnn/gemm/cutedsl/dense/proj_rope_mxfp8/gemm_proj_rope_mxfp8_mxfp8in.py``
(``gemm_proj_rope_mxfp8_kernel`` and ``gemm_proj_rope_mxfp8_host``), reached
through ``GemmProjRopeMxfp8Mxfp8InSm100`` in ``api.py``.
"""

import tirx_kernels.kern as K

_TILE_M = 128
_HEAD_DIM = 192
_QK_ROPE = 64
_BLOCK = 32
_K_DIM = 1536
_NUM_HEADS = 128
_MAX_ACTIVE_CLUSTERS = 148
_TRY_WAIT_TICKS = 10_000_000

_K_TILE = 128
_AB_STAGES = 3
_ACC_STAGES = 2
_SHARED_BYTES = 177_792
_A_OFFSET = 128
_A_STAGE_BYTES = 16_384
_B_OFFSET = 49_280
_B_STAGE_BYTES = 24_576
_SFA_OFFSET = 123_008
_SFA_STAGE_BYTES = 512
_SFB_OFFSET = 124_544
_SFB_STAGE_BYTES = 1_024
_SACC_OFFSET = 127_616
_SACC_STRIDE = 196
_TMEM_DEALLOC_OFFSET = 80
_TMEM_PTR_OFFSET = 88
_TMEM_COLUMNS = 512
_SFA_TMEM_COLUMN = 384
_SFB_TMEM_COLUMN = 400
_AB_DESC_BASE = 0x4000404000010000
_SF_DESC_BASE = 0x400800010000
_MMA_INSTR_BASE = 0x08B00000
_TMA_G2S_2D = (
    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
)
_MMA_MXFP8 = "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32"


def _descriptor_with_address(base, shared_address):
    address_field = K.cast(
        K.bitwise_and(K.shift_right(shared_address, K.uint32(4)), K.uint32(0x3FFF)), "uint64"
    )
    return K.bitwise_or(K.uint64(base), address_field)


def _advance(state):
    state.advance()


def _wait_plain(barrier, phase):
    ready = K.local_scalar("uint32", init=K.uint32(0))
    with K.While(ready == K.uint32(0)):
        K.ptx.mbarrier.try_wait.parity.shared.b64(
            ready, barrier, K.cast(phase, "uint32"), K.uint32(_TRY_WAIT_TICKS)
        )


def _elected():
    elected_lane = K.local_scalar("uint32")
    elected_pred = K.local_scalar("uint32")
    K.ptx.elect_sync(elected_lane, elected_pred, K.uint32(0xFFFFFFFF))
    return elected_pred == K.uint32(1)


def _fadd2(a0, a1, b0, b1):
    packed = K.local_scalar("uint64")
    out0 = K.local_scalar("float32")
    out1 = K.local_scalar("float32")
    K.ptx.add.rn.f32x2(packed, K.cuda.make_float2(a0, a1), K.cuda.make_float2(b0, b1))
    K.ptx.mov.b64(out0, out1, packed)
    return out0, out1


def _fmul2(a0, a1, b0, b1):
    packed = K.local_scalar("uint64")
    out0 = K.local_scalar("float32")
    out1 = K.local_scalar("float32")
    K.ptx.mul.rn.f32x2(packed, K.cuda.make_float2(a0, a1), K.cuda.make_float2(b0, b1))
    K.ptx.mov.b64(out0, out1, packed)
    return out0, out1


def _ffma2(a0, a1, b0, b1, c0, c1):
    packed = K.local_scalar("uint64")
    out0 = K.local_scalar("float32")
    out1 = K.local_scalar("float32")
    K.ptx.fma.rn.f32x2(
        packed, K.cuda.make_float2(a0, a1), K.cuda.make_float2(b0, b1), K.cuda.make_float2(c0, c1)
    )
    K.ptx.mov.b64(out0, out1, packed)
    return out0, out1


def _e8m0_inverse(scale_byte):
    bits = K.local_scalar("int32")
    result = K.local_scalar("float32")
    K.ptx.sub.s32(bits, K.int32(254), K.cast(scale_byte, "int32"))
    K.ptx.shl.b32(bits, bits, K.uint32(23))
    K.ptx.mov.b32(result, bits)
    return result


def _absmax(lhs, rhs):
    absolute = K.local_scalar("float32")
    result = K.local_scalar("float32")
    K.ptx.abs.f32(absolute, rhs)
    K.ptx.max.f32(result, lhs, absolute)
    return result


def _shuffle_xor_f32(value, delta):
    shuffled = K.local_scalar("uint32")
    K.ptx.shfl_sync.bfly.b32(
        shuffled,
        K.reinterpret("uint32", value),
        K.uint32(delta),
        K.uint32(31),
        K.uint32(0xFFFFFFFF),
    )
    return K.reinterpret("float32", shuffled)


def _config(label, tokens):
    return {"label": label, "tokens": tokens, "k_dim": _K_DIM, "num_heads": _NUM_HEADS}


KERNEL_META = {
    "name": "cudnn_sm100_gemm_proj_rope_mxfp8_mxfp8in",
    "category": "cudnn",
    "compute_capability": 10,
}

CONFIGS = [
    _config("t128_k1536_h128", 128),
    _config("t256_k1536_h128", 256),
    _config("t2048_k1536_h128", 2048),
    _config("t4096_k1536_h128", 4096),
]

BENCH_CONFIGS = [dict(config) for config in CONFIGS if config["tokens"] in (2048, 4096)]


def _validate_config(tokens, k_dim, num_heads):
    if tokens <= 0 or tokens % _TILE_M:
        raise ValueError(f"tokens must be a positive multiple of {_TILE_M}, got {tokens}")
    if k_dim != _K_DIM:
        raise ValueError(f"the verified MXFP8 projection specialization requires k_dim={_K_DIM}")
    if num_heads != _NUM_HEADS:
        raise ValueError(
            f"the verified MXFP8 projection specialization requires num_heads={_NUM_HEADS}"
        )


def _make_kernel(tokens, k_dim, num_heads):
    _validate_config(tokens, k_dim, num_heads)
    total_work = (tokens // _TILE_M) * num_heads
    num_clusters = min(total_work, _MAX_ACTIVE_CLUSTERS)
    m_tiles = tokens // _TILE_M
    swizzle = {128: 4, 256: 4, 2048: 16, 4096: 32}[tokens]
    t2r_x8 = tokens >= 2048

    def host_prelude(params):
        x_code = params["x_code"]
        w_code = params["w_code"]
        a_map = K.stack_alloca("tensormap", 1)
        b_map = K.stack_alloca("tensormap", 1)
        K.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            a_map,
            "float8_e4m3fn",
            2,
            x_code.data,
            k_dim,
            tokens,
            k_dim,
            _K_TILE,
            _TILE_M,
            1,
            1,
            0,
            3,
            2,
            0,
        )
        K.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            b_map,
            "float8_e4m3fn",
            2,
            w_code.data,
            k_dim,
            num_heads * _HEAD_DIM,
            k_dim,
            _K_TILE,
            _HEAD_DIM,
            1,
            1,
            0,
            3,
            2,
            0,
        )
        return a_map, b_map

    def kernel(
        x_code,
        x_scale,
        w_code,
        w_scale,
        cos,
        sin,
        out_fp8_row,
        out_scales_row,
        out_fp8_col,
        out_scales_col,
        *,
        host,
    ):
        # TIRX_PORT_START: gemm_proj_rope_mxfp8_mxfp8in_kernel
        del x_code, w_code
        a_map, b_map = host
        _block_x, _block_y, cluster_work_id = K.cta_id()
        del _block_x, _block_y
        warp = K.warp_id()
        lane = K.lane_id()

        roles = K.specialize(chain_dispatch=True)
        epilogue_role = roles.role("epilogue", warps=list(range(12)))
        mma_role = roles.role("mma", warps=[12])
        tma_role = roles.role("tma", warps=[13])

        smem = K.alloc_buffer((_SHARED_BYTES,), K.u8, scope="shared.dyn", align=1024)
        protocol_pool = K.smem_pool(base=smem)
        ab_pipe = K.Pipeline(
            protocol_pool,
            _AB_STAGES,
            full="tma",
            empty="tcgen05",
            init_empty=1,
            leader=K.bool(False),
        )
        acc_pipe = K.Pipeline(
            protocol_pool,
            _ACC_STAGES,
            full="tcgen05",
            empty="mbar",
            init_empty=4,
            leader=K.bool(False),
        )
        if protocol_pool.bytes != _TMEM_DEALLOC_OFFSET:
            raise AssertionError("pipeline protocol layout changed")
        protocol_pool.alloc((1,), K.u64, align=8)
        tmem_slot = protocol_pool.alloc((1,), K.u32, align=4)
        if protocol_pool.bytes != _TMEM_PTR_OFFSET + 4:
            raise AssertionError("TMEM protocol layout changed")

        with tma_role:
            K.ptx.prefetch.tensormap(K.address_of(a_map))
            K.ptx.prefetch.tensormap(K.address_of(b_map))

        # Initialize AB then ACC exactly at the two source publication edges.
        with K.If(warp == 0):
            with K.Then():
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, _AB_STAGES) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                ab_pipe.full.ptr_to([stage]), K.uint32(1)
                            )
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, _AB_STAGES) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                ab_pipe.empty.ptr_to([stage]), K.uint32(1)
                            )
        K.ptx.fence.mbarrier_init.release.cluster()
        K.ptx.bar.sync(K.uint32(0), K.uint32(448))
        with K.If(warp == 0):
            with K.Then():
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, _ACC_STAGES) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                acc_pipe.full.ptr_to([stage]), K.uint32(1)
                            )
                with K.If(_elected()):
                    with K.Then():
                        with K.unroll(0, _ACC_STAGES) as stage:
                            K.ptx.mbarrier.init.shared.b64(
                                acc_pipe.empty.ptr_to([stage]), K.uint32(4)
                            )
        K.ptx.fence.mbarrier_init.release.cluster()
        K.ptx.bar.sync(K.uint32(0), K.uint32(448))

        smem_base = K.local_scalar("uint32")
        K.assign(smem_base, K.cuda.cvta_generic_to_shared(smem.ptr_to([0])))
        a_descriptor = K.local_scalar(
            "uint64", init=_descriptor_with_address(_AB_DESC_BASE, smem_base + _A_OFFSET)
        )
        b_descriptor = K.local_scalar(
            "uint64", init=_descriptor_with_address(_AB_DESC_BASE, smem_base + _B_OFFSET)
        )
        sfa_descriptor = K.local_scalar(
            "uint64", init=_descriptor_with_address(_SF_DESC_BASE, smem_base + _SFA_OFFSET)
        )
        sfb_descriptor = K.local_scalar(
            "uint64", init=_descriptor_with_address(_SF_DESC_BASE, smem_base + _SFB_OFFSET)
        )

        def scheduler_coords(work):
            head_minor = work % swizzle
            quotient = work // swizzle
            m_idx = quotient % m_tiles
            head_major = quotient // m_tiles
            return m_idx, head_major * swizzle + head_minor

        def advance_work(work):
            K.assign(work, work + num_clusters)

        # CuTe expresses the same 13-warp TMEM allocation rendezvous at two
        # branch-local PCs. One physical PC preserves its participants/count and
        # avoids divergent named-barrier diagnostics under synccheck.
        with K.If(warp < K.uint32(13)):
            with K.Then():
                with K.If(warp == 0):
                    with K.Then():
                        K.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                            tmem_slot.ptr_to([0]), K.uint32(_TMEM_COLUMNS)
                        )
                K.ptx.bar.sync(K.uint32(2), K.uint32(416))

        with tma_role:
            tma_state = K.PipelineState(_AB_STAGES, phase=1)
            work = K.local_scalar("int32", init=cluster_work_id)
            sfa_words = K.alloc_local((4,), "uint32")
            sfb_words = K.alloc_local((8,), "uint32")
            with K.While(work < total_work):
                m_idx, head = scheduler_coords(work)
                with K.serial(k_dim // _K_TILE) as k_tile:
                    handle_stage = K.local_scalar("int32", init=tma_state.stage)
                    handle_phase = K.local_scalar("int32", init=tma_state.phase)
                    _wait_plain(ab_pipe.empty.ptr_to([handle_stage]), handle_phase)
                    K.ptx["fence.proxy.async.shared::cta"]()
                    _advance(tma_state)
                    with K.If(_elected()):
                        with K.Then():
                            K.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                ab_pipe.full.ptr_to([handle_stage]), K.uint32(40_960)
                            )

                    scale_word = k_tile * 4
                    sfa_row = m_idx * _TILE_M + lane
                    for row_block in range(4):
                        scale_offset = (sfa_row + row_block * 32) * (k_dim // _BLOCK)
                        K.ptx.ld.global_.b32(
                            sfa_words[row_block], x_scale.ptr_to([scale_offset + scale_word])
                        )
                    K.ptx.st.shared.v4.b32(
                        smem.ptr_to([_SFA_OFFSET + handle_stage * _SFA_STAGE_BYTES + lane * 16]),
                        sfa_words[0],
                        sfa_words[1],
                        sfa_words[2],
                        sfa_words[3],
                    )

                    sfb_row = head * _HEAD_DIM - (head % 2) * _QK_ROPE + lane
                    for row_block in range(8):
                        scale_offset = (sfb_row + row_block * 32) * (k_dim // _BLOCK)
                        K.ptx.ld.global_.b32(
                            sfb_words[row_block], w_scale.ptr_to([scale_offset + scale_word])
                        )
                    K.ptx.st.shared.v4.b32(
                        smem.ptr_to([_SFB_OFFSET + handle_stage * _SFB_STAGE_BYTES + lane * 16]),
                        sfb_words[0],
                        sfb_words[1],
                        sfb_words[2],
                        sfb_words[3],
                    )
                    K.ptx.st.shared.v4.b32(
                        smem.ptr_to(
                            [_SFB_OFFSET + handle_stage * _SFB_STAGE_BYTES + 512 + lane * 16]
                        ),
                        sfb_words[4],
                        sfb_words[5],
                        sfb_words[6],
                        sfb_words[7],
                    )
                    K.ptx.bar.warp.sync(K.uint32(0xFFFFFFFF))
                    K.ptx["fence.proxy.async.shared::cta"]()
                    with K.If(_elected()):
                        with K.Then():
                            K.ptx[_TMA_G2S_2D](
                                smem_base + _A_OFFSET + handle_stage * _A_STAGE_BYTES,
                                K.address_of(a_map),
                                K.cast(k_tile * _K_TILE, "int32"),
                                K.cast(m_idx * _TILE_M, "int32"),
                                ab_pipe.full.ptr_to([handle_stage]),
                                K.uint64(0),
                            )
                    with K.If(_elected()):
                        with K.Then():
                            K.ptx[_TMA_G2S_2D](
                                smem_base + _B_OFFSET + handle_stage * _B_STAGE_BYTES,
                                K.address_of(b_map),
                                K.cast(k_tile * _K_TILE, "int32"),
                                K.cast(head * _HEAD_DIM, "int32"),
                                ab_pipe.full.ptr_to([handle_stage]),
                                K.uint64(0),
                            )
                advance_work(work)
            for _ in range(_AB_STAGES):
                _wait_plain(ab_pipe.empty.ptr_to([tma_state.stage]), tma_state.phase)
                _advance(tma_state)

        with mma_role:
            tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(tmem_base, tmem_slot.ptr_to([0]))
            mma_state = K.PipelineState(_AB_STAGES, phase=0)
            acc_state = K.PipelineState(_ACC_STAGES, phase=1)
            work = K.local_scalar("int32", init=cluster_work_id)
            accumulate = K.local_scalar("uint32")
            with K.While(work < total_work):
                acc_stage = K.local_scalar("int32", init=acc_state.stage)
                acc_phase = K.local_scalar("int32", init=acc_state.phase)
                _wait_plain(acc_pipe.empty.ptr_to([acc_stage]), acc_phase)
                _advance(acc_state)
                _m_idx, head = scheduler_coords(work)
                del _m_idx
                sfb_mma_base = K.local_scalar(
                    "uint32", init=tmem_base + _SFB_TMEM_COLUMN + (head % 2) * 2
                )
                K.assign(accumulate, K.uint32(0))
                with K.serial(k_dim // _K_TILE) as _k_tile:
                    handle_stage = K.local_scalar("int32", init=mma_state.stage)
                    handle_phase = K.local_scalar("int32", init=mma_state.phase)
                    _wait_plain(ab_pipe.full.ptr_to([handle_stage]), handle_phase)
                    _advance(mma_state)
                    with K.If(_elected()):
                        with K.Then():
                            K.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](
                                K.cast(tmem_base + _SFA_TMEM_COLUMN, "uint32"),
                                sfa_descriptor
                                + K.cast(handle_stage * (_SFA_STAGE_BYTES // 16), "uint64"),
                            )
                    for half in range(2):
                        with K.If(_elected()):
                            with K.Then():
                                K.ptx["tcgen05.cp.cta_group::1.32x128b.warpx4"](
                                    K.cast(tmem_base + _SFB_TMEM_COLUMN + half * 4, "uint32"),
                                    sfb_descriptor
                                    + K.cast(
                                        handle_stage * (_SFB_STAGE_BYTES // 16) + half * 32,
                                        "uint64",
                                    ),
                                )
                    for kphase in range(4):
                        selector = K.uint32(kphase * 0x40000000)
                        sfa_tmem_addr = K.cast(tmem_base + _SFA_TMEM_COLUMN + selector, "uint32")
                        sfb_tmem_addr = K.cast(sfb_mma_base + selector, "uint32")
                        runtime_instr_desc = K.bitwise_and(
                            K.uint32(_MMA_INSTR_BASE), K.uint32(0x9FFFFFCF)
                        )
                        runtime_instr_desc = K.bitwise_or(
                            runtime_instr_desc,
                            K.bitwise_and(
                                K.shift_right(sfa_tmem_addr, K.uint32(1)), K.uint32(0x60000000)
                            ),
                        )
                        runtime_instr_desc = K.bitwise_or(
                            runtime_instr_desc,
                            K.bitwise_and(
                                K.shift_right(sfb_tmem_addr, K.uint32(26)), K.uint32(0x30)
                            ),
                        )
                        with K.If(_elected()):
                            with K.Then():
                                K.ptx[_MMA_MXFP8](
                                    K.cast(tmem_base + acc_stage * _HEAD_DIM, "uint32"),
                                    a_descriptor
                                    + K.cast(
                                        handle_stage * (_A_STAGE_BYTES // 16) + kphase * 2, "uint64"
                                    ),
                                    b_descriptor
                                    + K.cast(
                                        handle_stage * (_B_STAGE_BYTES // 16) + kphase * 2, "uint64"
                                    ),
                                    runtime_instr_desc,
                                    sfa_tmem_addr,
                                    sfb_tmem_addr,
                                    K.ptx.pred(K.cast(accumulate, "bool")),
                                )
                        K.assign(accumulate, K.uint32(1))
                    with K.If(_elected()):
                        with K.Then():
                            K.ptx[
                                "tcgen05.commit.cta_group::1.mbarrier::arrive::one."
                                "shared::cluster.b64"
                            ](ab_pipe.empty.ptr_to([handle_stage]))
                with K.If(_elected()):
                    with K.Then():
                        K.ptx[
                            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
                        ](acc_pipe.full.ptr_to([acc_stage]))
                advance_work(work)
            _advance(acc_state)
            _wait_plain(acc_pipe.empty.ptr_to([acc_state.stage]), acc_state.phase)

        with epilogue_role:
            tmem_base = K.local_scalar("uint32")
            K.ptx.ld.shared.b32(tmem_base, tmem_slot.ptr_to([0]))
            acc_state = K.PipelineState(_ACC_STAGES, phase=0)
            work = K.local_scalar("int32", init=cluster_work_id)
            values0 = K.alloc_local((_BLOCK,), "float32")
            values1 = K.alloc_local((_BLOCK,), "float32")
            drain_count = 8 if t2r_x8 else 32
            drain = K.alloc_local((drain_count,), "float32")
            staged_words = K.alloc_local((drain_count // 2,), "uint32")

            with K.While(work < total_work):
                m_idx, head = scheduler_coords(work)
                token_base = m_idx * _TILE_M
                with K.If(warp < 4):
                    with K.Then():
                        acc_stage = K.local_scalar("int32", init=acc_state.stage)
                        acc_phase = K.local_scalar("int32", init=acc_state.phase)
                        _wait_plain(acc_pipe.full.ptr_to([acc_stage]), acc_phase)
                        _advance(acc_state)
                        row = warp * 32 + lane
                        if t2r_x8:
                            for group in range(6):
                                for chunk in range(4):
                                    tmem_address = K.cast(
                                        tmem_base
                                        + (warp << 21)
                                        + acc_stage * _HEAD_DIM
                                        + group * 32
                                        + chunk * 8,
                                        "uint32",
                                    )
                                    K.ptx["tcgen05.ld.sync.aligned.32x32b.x8.b32"](
                                        *[drain[index] for index in range(8)], tmem_address
                                    )
                                    for pair in range(4):
                                        K.ptx.cvt.rn.bf16x2.f32(
                                            staged_words[pair], drain[pair * 2 + 1], drain[pair * 2]
                                        )
                                    row_byte = _SACC_OFFSET + 2 * (
                                        row * _SACC_STRIDE + group * 32 + chunk * 8
                                    )
                                    for vector in range(2):
                                        K.ptx.st.shared.v2.b32(
                                            smem.ptr_to([row_byte + vector * 8]),
                                            staged_words[vector * 2],
                                            staged_words[vector * 2 + 1],
                                        )
                        else:
                            for group in range(6):
                                tmem_address = K.cast(
                                    tmem_base + (warp << 21) + acc_stage * _HEAD_DIM + group * 32,
                                    "uint32",
                                )
                                K.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                                    *[drain[index] for index in range(32)], tmem_address
                                )
                                for pair in range(16):
                                    K.ptx.cvt.rn.bf16x2.f32(
                                        staged_words[pair], drain[pair * 2 + 1], drain[pair * 2]
                                    )
                                row_byte = _SACC_OFFSET + 2 * (row * _SACC_STRIDE + group * 32)
                                for vector in range(8):
                                    K.ptx.st.shared.v2.b32(
                                        smem.ptr_to([row_byte + vector * 8]),
                                        staged_words[vector * 2],
                                        staged_words[vector * 2 + 1],
                                    )
                        with K.If(_elected()):
                            with K.Then():
                                K.ptx.mbarrier.arrive.shared.b64(
                                    acc_pipe.empty.ptr_to([acc_stage]), K.uint32(1)
                                )

                K.ptx.bar.sync(K.uint32(1), K.uint32(384))
                cb = warp // 3
                fc = warp % 3
                tok0 = cb * _BLOCK
                f0 = fc * 64 + lane * 2
                col_amax0 = K.local_scalar("float32", init=K.float32(0.0))
                col_amax1 = K.local_scalar("float32", init=K.float32(0.0))

                with K.If(fc == 2):
                    with K.Then():
                        lib = lane % 16
                        pcol0 = 128 + 4 * lib
                        roff = 32 * (lane // 16)
                        rf = K.cast(lane // 16, "float32")
                        lf = K.float32(1.0) - rf
                        cidx0 = 2 * lib + roff
                        for row_index in range(_BLOCK):
                            shared_byte = _SACC_OFFSET + 2 * (
                                (tok0 + row_index) * _SACC_STRIDE + pcol0
                            )
                            p0_b16 = K.local_scalar("uint16")
                            q0_b16 = K.local_scalar("uint16")
                            p1_b16 = K.local_scalar("uint16")
                            q1_b16 = K.local_scalar("uint16")
                            K.ptx.ld.shared.v4.b16(
                                p0_b16, q0_b16, p1_b16, q1_b16, smem.ptr_to([shared_byte])
                            )
                            p0 = K.local_scalar("float32")
                            q0 = K.local_scalar("float32")
                            p1 = K.local_scalar("float32")
                            q1 = K.local_scalar("float32")
                            K.ptx.cvt.f32.bf16(p0, p0_b16)
                            K.ptx.cvt.f32.bf16(q0, q0_b16)
                            K.ptx.cvt.f32.bf16(p1, p1_b16)
                            K.ptx.cvt.f32.bf16(q1, q1_b16)
                            cos0_b16 = K.local_scalar("uint16")
                            cos1_b16 = K.local_scalar("uint16")
                            sin0_b16 = K.local_scalar("uint16")
                            sin1_b16 = K.local_scalar("uint16")
                            trig_offset = (token_base + tok0 + row_index) * _QK_ROPE + cidx0
                            K.ptx.ld.global_.b16(cos0_b16, cos.ptr_to([trig_offset]))
                            K.ptx.ld.global_.b16(cos1_b16, cos.ptr_to([trig_offset + 1]))
                            K.ptx.ld.global_.b16(sin0_b16, sin.ptr_to([trig_offset]))
                            K.ptx.ld.global_.b16(sin1_b16, sin.ptr_to([trig_offset + 1]))
                            c0 = K.local_scalar("float32")
                            c1 = K.local_scalar("float32")
                            s0 = K.local_scalar("float32")
                            s1 = K.local_scalar("float32")
                            K.ptx.cvt.f32.bf16(c0, cos0_b16)
                            K.ptx.cvt.f32.bf16(c1, cos1_b16)
                            K.ptx.cvt.f32.bf16(s0, sin0_b16)
                            K.ptx.cvt.f32.bf16(s1, sin1_b16)
                            pc0, pc1 = _fmul2(p0, p1, c0, c1)
                            qs0, qs1 = _fmul2(q0, q1, s0, s1)
                            lft0, lft1 = _ffma2(
                                qs0, qs1, K.float32(-1.0), K.float32(-1.0), pc0, pc1
                            )
                            ps0, ps1 = _fmul2(p0, p1, s0, s1)
                            qc0, qc1 = _fmul2(q0, q1, c0, c1)
                            rgt0, rgt1 = _fadd2(ps0, ps1, qc0, qc1)
                            ll0, ll1 = _fmul2(lft0, lft1, lf, lf)
                            v0, v1 = _ffma2(rgt0, rgt1, rf, rf, ll0, ll1)
                            K.assign(values0[row_index], v0)
                            K.assign(values1[row_index], v1)
                            K.assign(col_amax0, _absmax(col_amax0, v0))
                            K.assign(col_amax1, _absmax(col_amax1, v1))
                    with K.Else():
                        for row_index in range(_BLOCK):
                            shared_byte = _SACC_OFFSET + 2 * (
                                (tok0 + row_index) * _SACC_STRIDE + f0
                            )
                            v0_b16 = K.local_scalar("uint16")
                            v1_b16 = K.local_scalar("uint16")
                            K.ptx.ld.shared.v2.b16(v0_b16, v1_b16, smem.ptr_to([shared_byte]))
                            v0 = K.local_scalar("float32")
                            v1 = K.local_scalar("float32")
                            K.ptx.cvt.f32.bf16(v0, v0_b16)
                            K.ptx.cvt.f32.bf16(v1, v1_b16)
                            K.assign(values0[row_index], v0)
                            K.assign(values1[row_index], v1)
                            K.assign(col_amax0, _absmax(col_amax0, v0))
                            K.assign(col_amax1, _absmax(col_amax1, v1))

                col_scale_pair = K.local_scalar("uint16")
                K.ptx.cvt.rp.satfinite.ue8m0x2.f32(
                    col_scale_pair,
                    col_amax0 * K.float32(1.0 / 448.0),
                    col_amax1 * K.float32(1.0 / 448.0),
                )
                col_scale0 = K.cast(
                    K.shift_right(K.cast(col_scale_pair, "uint32"), K.uint32(8)), "uint8"
                )
                col_scale1 = K.cast(
                    K.bitwise_and(K.cast(col_scale_pair, "uint32"), K.uint32(0xFF)), "uint8"
                )
                inv_col0 = _e8m0_inverse(col_scale0)
                inv_col1 = _e8m0_inverse(col_scale1)
                col_scale_base = ((m_idx * 4 + cb) * num_heads + head) * _HEAD_DIM + f0
                K.ptx.st.global_.b8(out_scales_col.ptr_to([col_scale_base]), col_scale0)
                K.ptx.st.global_.b8(out_scales_col.ptr_to([col_scale_base + 1]), col_scale1)

                row_block = fc * 2 + lane // 16
                for row_index in range(_BLOCK):
                    v0 = values0[row_index]
                    v1 = values1[row_index]
                    absolute0 = K.local_scalar("float32")
                    absolute1 = K.local_scalar("float32")
                    row_amax = K.local_scalar("float32")
                    K.ptx.abs.f32(absolute0, v0)
                    K.ptx.abs.f32(absolute1, v1)
                    K.ptx.max.f32(row_amax, absolute0, absolute1)
                    for delta in (8, 4, 2, 1):
                        other = _shuffle_xor_f32(row_amax, delta)
                        K.ptx.max.f32(row_amax, row_amax, other)
                    row_scale_pair = K.local_scalar("uint16")
                    K.ptx.cvt.rp.satfinite.ue8m0x2.f32(
                        row_scale_pair, K.float32(0.0), row_amax * K.float32(1.0 / 448.0)
                    )
                    row_scale = K.cast(
                        K.bitwise_and(K.cast(row_scale_pair, "uint32"), K.uint32(0xFF)), "uint32"
                    )
                    token = token_base + tok0 + row_index
                    with K.If((lane % 16) == 0):
                        with K.Then():
                            row_scale_offset = (token * num_heads + head) * (
                                _HEAD_DIM // _BLOCK
                            ) + row_block
                            K.ptx.st.global_.b8(
                                out_scales_row.ptr_to([row_scale_offset]),
                                K.cast(row_scale, "uint8"),
                            )
                    inv_row = _e8m0_inverse(row_scale)
                    vr0, vr1 = _fmul2(v0, v1, inv_row, inv_row)
                    row_pair = K.local_scalar("uint16")
                    K.ptx.cvt.rn.satfinite.e4m3x2.f32(row_pair, vr1, vr0)
                    output_offset = (token * num_heads + head) * _HEAD_DIM + f0
                    K.ptx.st.global_.b16(out_fp8_row.ptr_to([output_offset]), row_pair)

                for row_index in range(_BLOCK):
                    token = token_base + tok0 + row_index
                    v0 = values0[row_index]
                    v1 = values1[row_index]
                    vc0, vc1 = _fmul2(v0, v1, inv_col0, inv_col1)
                    col_pair = K.local_scalar("uint16")
                    K.ptx.cvt.rn.satfinite.e4m3x2.f32(col_pair, vc1, vc0)
                    output_offset = (token * num_heads + head) * _HEAD_DIM + f0
                    K.ptx.st.global_.b16(out_fp8_col.ptr_to([output_offset]), col_pair)

                K.ptx.bar.sync(K.uint32(1), K.uint32(384))
                advance_work(work)

            with K.If(warp == 0):
                with K.Then():
                    K.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
                    K.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](
                        tmem_base, K.uint32(_TMEM_COLUMNS)
                    )

    kernel.__annotations__ = {
        "x_code": K.gptr[K.u8, (tokens * k_dim,)],
        "x_scale": K.gptr[K.u8, (tokens * (k_dim // _BLOCK),)],
        "w_code": K.gptr[K.u8, (num_heads * _HEAD_DIM * k_dim,)],
        "w_scale": K.gptr[K.u8, (num_heads * _HEAD_DIM * (k_dim // _BLOCK),)],
        "cos": K.gptr[K.bf16, (tokens * _QK_ROPE,)],
        "sin": K.gptr[K.bf16, (tokens * _QK_ROPE,)],
        "out_fp8_row": K.gptr[K.u8, (tokens * num_heads * _HEAD_DIM,)],
        "out_scales_row": K.gptr[K.u8, (tokens * num_heads * (_HEAD_DIM // _BLOCK),)],
        "out_fp8_col": K.gptr[K.u8, (tokens * num_heads * _HEAD_DIM,)],
        "out_scales_col": K.gptr[K.u8, ((tokens // _BLOCK) * num_heads * _HEAD_DIM,)],
    }
    return K.kernel(warps=14, arch="sm_100a", grid=[1, 1, num_clusters], host_prelude=host_prelude)(
        kernel
    )


def get_kernel(tokens, k_dim, num_heads):
    return _make_kernel(tokens, k_dim, num_heads).func


def _quantize_mxfp8(torch, values):
    rows, k_dim = values.shape
    blocks = values.float().reshape(rows, k_dim // _BLOCK, _BLOCK)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-127.0, 127.0)
    code = (
        (blocks * torch.pow(2.0, -exponent))
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(rows, k_dim)
    )
    scale = (exponent.squeeze(-1) + 127.0).to(torch.uint8)
    return code, scale


def _dequantize_input(torch, code, scale):
    rows, k_dim = code.shape
    decoded = torch.pow(2.0, scale.float() - 127.0).unsqueeze(-1)
    return (code.float().reshape(rows, k_dim // _BLOCK, _BLOCK) * decoded).reshape(rows, k_dim)


def prepare_data(tokens, k_dim, num_heads):
    """Allocate deterministic MXFP8 inputs and independent mutable outputs."""
    _validate_config(tokens, k_dim, num_heads)
    import torch

    torch.manual_seed(0)
    x_bf16 = torch.randn(tokens, k_dim, dtype=torch.bfloat16, device="cuda") * 0.5
    w_bf16 = torch.randn(num_heads * _HEAD_DIM, k_dim, dtype=torch.bfloat16, device="cuda") * 0.02
    x_code, x_scale = _quantize_mxfp8(torch, x_bf16)
    w_code, w_scale = _quantize_mxfp8(torch, w_bf16)
    cos = torch.randn(tokens, _QK_ROPE, dtype=torch.bfloat16, device="cuda")
    sin = torch.randn(tokens, _QK_ROPE, dtype=torch.bfloat16, device="cuda")

    def outputs():
        return {
            "out_fp8_row": torch.empty(
                tokens, num_heads, _HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda"
            ),
            "out_scales_row": torch.empty(
                tokens, num_heads, _HEAD_DIM // _BLOCK, dtype=torch.uint8, device="cuda"
            ),
            "out_fp8_col": torch.empty(
                tokens, num_heads, _HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda"
            ),
            "out_scales_col": torch.empty(
                tokens // _BLOCK, num_heads, _HEAD_DIM, dtype=torch.uint8, device="cuda"
            ),
        }

    return {
        "x_code": x_code,
        "x_scale": x_scale,
        "w_code": w_code,
        "w_scale": w_scale,
        "x_dequant_bf16": _dequantize_input(torch, x_code, x_scale).to(torch.bfloat16),
        "w_dequant_bf16": _dequantize_input(torch, w_code, w_scale).to(torch.bfloat16),
        "cos": cos,
        "sin": sin,
        "tirx": outputs(),
        "source": outputs(),
    }


def _without_label(config):
    return {key: value for key, value in config.items() if key != "label"}


def _tirx_launch(executable, data):
    import torch

    output = data["tirx"]
    arguments = (
        data["x_code"].view(torch.uint8).reshape(-1),
        data["x_scale"].reshape(-1),
        data["w_code"].view(torch.uint8).reshape(-1),
        data["w_scale"].reshape(-1),
        data["cos"].reshape(-1),
        data["sin"].reshape(-1),
        output["out_fp8_row"].view(torch.uint8).reshape(-1),
        output["out_scales_row"].reshape(-1),
        output["out_fp8_col"].view(torch.uint8).reshape(-1),
        output["out_scales_col"].reshape(-1),
    )

    def launch():
        executable(*arguments)

    launch._keep_alive = arguments
    return launch


def _compile_reference(data, config):
    from tirx_kernels.cudnn._reference import load_reference_module

    module = load_reference_module("cudnn.gemm.cutedsl.dense.proj_rope_mxfp8.api")
    output = data["source"]
    op = module.GemmProjRopeMxfp8Mxfp8InSm100(
        data["x_code"],
        data["x_scale"],
        data["w_code"],
        data["w_scale"],
        data["cos"],
        data["sin"],
        output["out_fp8_row"],
        output["out_scales_row"],
        output["out_fp8_col"],
        output["out_scales_col"],
    )
    if not op.check_support():
        raise RuntimeError(f"pinned source rejected {config}")
    op.compile()

    def launch():
        op.execute(
            data["x_code"],
            data["x_scale"],
            data["w_code"],
            data["w_scale"],
            data["cos"],
            data["sin"],
            output["out_fp8_row"],
            output["out_scales_row"],
            output["out_fp8_col"],
            output["out_scales_col"],
        )

    launch._keep_alive = (op, output)
    return launch


def _dequantize_row(torch, data, scale):
    tokens, num_heads, _ = data.shape
    decoded = torch.pow(2.0, scale.float() - 127.0).unsqueeze(-1)
    return (data.float().reshape(tokens, num_heads, _HEAD_DIM // _BLOCK, _BLOCK) * decoded).reshape(
        tokens, num_heads, _HEAD_DIM
    )


def _dequantize_col(torch, data, scale):
    tokens, num_heads, _ = data.shape
    decoded = torch.pow(2.0, scale.float() - 127.0).reshape(
        tokens // _BLOCK, 1, num_heads, _HEAD_DIM
    )
    return (data.float().reshape(tokens // _BLOCK, _BLOCK, num_heads, _HEAD_DIM) * decoded).reshape(
        tokens, num_heads, _HEAD_DIM
    )


def _match_fraction(torch, actual, expected):
    difference = (actual.float() - expected.float()).abs()
    return float((difference <= 0.1 + 0.1 * expected.float().abs()).float().mean().item())


def _validate_outputs(data):
    import torch

    tirx = data["tirx"]
    source = data["source"]
    row_match = _match_fraction(
        torch,
        _dequantize_row(torch, tirx["out_fp8_row"], tirx["out_scales_row"]),
        _dequantize_row(torch, source["out_fp8_row"], source["out_scales_row"]),
    )
    col_match = _match_fraction(
        torch,
        _dequantize_col(torch, tirx["out_fp8_col"], tirx["out_scales_col"]),
        _dequantize_col(torch, source["out_fp8_col"], source["out_scales_col"]),
    )
    if row_match < 0.95 or col_match < 0.95:
        raise AssertionError(
            f"TIRx versus pinned source mismatch: row_match={row_match}, "
            f"col_match={col_match}; required >=0.95"
        )
    return {"row_match": row_match, "col_match": col_match}


def run_test(**config):
    """Compile, run, and compare one specialization with the pinned source."""
    import torch

    from tirx_kernels.runner import compile_kernel

    kernel_config = _without_label(config)
    data = prepare_data(**kernel_config)
    tirx_launch = _tirx_launch(compile_kernel(get_kernel(**kernel_config)), data)
    source_launch = _compile_reference(data, kernel_config)
    tirx_launch()
    source_launch()
    torch.cuda.synchronize()
    return _validate_outputs(data)


def prepare_bench(**config):
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    kernel_config = _without_label(config)
    state = {"config": kernel_config, "executable": compile_kernel(get_kernel(**kernel_config))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **kwargs):
    """Validate once, then time closures containing exactly one kernel launch."""
    from tirx_kernels.runner import bench, defer_gpu_interrupts, external_references_enabled

    with defer_gpu_interrupts():
        import torch

    config = _without_label({**prepared["config"], **kwargs})
    with_source = external_references_enabled()
    gpu_state = prepared.get("gpu_state")
    if gpu_state is None:
        data = prepare_data(**config)
        gpu_state = {
            "data": data,
            "tirx_launch": _tirx_launch(prepared["executable"], data),
            "source_launch": None,
            "validated": False,
            "with_source": with_source,
        }
        prepared["gpu_state"] = gpu_state
    elif gpu_state["with_source"] != with_source:
        raise RuntimeError("reference timing mode changed within one prepared benchmark")

    data = gpu_state["data"]
    tirx_launch = gpu_state["tirx_launch"]
    if not gpu_state["validated"]:
        tirx_launch()
        torch.cuda.synchronize()
        if with_source:
            with defer_gpu_interrupts():
                source_launch = _compile_reference(data, config)
                gpu_state["source_launch"] = source_launch
                source_launch()
                torch.cuda.synchronize()
            _validate_outputs(data)
        gpu_state["validated"] = True

    source_launch = gpu_state["source_launch"]
    references = {"cudnn_frontend": lambda: source_launch} if source_launch is not None else None
    return bench(
        {"tirx": tirx_launch},
        references=references,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **config):
    return prepare_bench(**config).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_gpu",
    "run_test",
]
