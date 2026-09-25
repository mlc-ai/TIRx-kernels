# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""BF16 projection GEMM with YARN RoPE and dual-direction MXFP8 outputs.

Upstream source:
``python/cudnn/gemm/cutedsl/dense/proj_rope_mxfp8/gemm_proj_rope_mxfp8_bf16in.py``
(``gemm_proj_rope_mxfp8_kernel`` and ``gemm_proj_rope_mxfp8_host``).
"""

from functools import cache

import tirx_kernels.tirx_lite as txl

_TILE_M = 128
_HEAD_DIM = 192
_QK_ROPE = 64
_BLOCK = 32
_K_DIM = 1536
_NUM_HEADS = 128
_MAX_ACTIVE_CLUSTERS = 148
_TRY_WAIT_TICKS = 10_000_000

_AB_STAGES = 4
_ACC_STAGES = 2
_SHARED_BYTES = 214_144
_A_OFFSET = 128
_A_STAGE_BYTES = 16_384
_B_OFFSET = 65_664
_B_STAGE_BYTES = 24_576
_SACC_OFFSET = 163_968
_SACC_STRIDE = 196
_TMEM_DEALLOC_OFFSET = 96
_TMEM_PTR_OFFSET = 104
_TMEM_COLUMNS = 512
_A_DESC_BASE = 0x4000404000010000
_TMA_G2S_2D = (
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
    "mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
)
_MMA_F16 = "tcgen05.mma.cta_group::1.kind::f16"


def _descriptor_with_address(base, shared_address):
    base_value = txl.bitwise_or(
        txl.shift_left(txl.uint64(base >> 32), txl.uint64(32)), txl.uint64(base & 0xFFFFFFFF)
    )
    address_field = txl.cast(
        txl.bitwise_and(txl.shift_right(shared_address, txl.uint32(4)), txl.uint32(0x3FFF)), "uint64"
    )
    return txl.bitwise_or(base_value, address_field)


def _advance(state):
    state.advance()


def _wait_plain(barrier, phase):
    ready = txl.local_scalar("uint32", init=txl.uint32(0))
    with txl.While(ready == txl.uint32(0)):
        txl.ptx.mbarrier.try_wait.parity.shared.b64(
            ready, barrier, txl.cast(phase, "uint32"), txl.uint32(_TRY_WAIT_TICKS)
        )


def _elected():
    elected_lane = txl.local_scalar("uint32")
    elected_pred = txl.local_scalar("uint32")
    txl.ptx.elect_sync(elected_lane, elected_pred, txl.uint32(0xFFFFFFFF))
    return elected_pred == txl.uint32(1)


def _fadd2(a0, a1, b0, b1):
    packed = txl.local_scalar("uint64")
    out0 = txl.local_scalar("float32")
    out1 = txl.local_scalar("float32")
    txl.ptx.add.rn.f32x2(packed, txl.cuda.make_float2(a0, a1), txl.cuda.make_float2(b0, b1))
    txl.ptx.mov.b64(out0, out1, packed)
    return out0, out1


def _fmul2(a0, a1, b0, b1):
    packed = txl.local_scalar("uint64")
    out0 = txl.local_scalar("float32")
    out1 = txl.local_scalar("float32")
    txl.ptx.mul.rn.f32x2(packed, txl.cuda.make_float2(a0, a1), txl.cuda.make_float2(b0, b1))
    txl.ptx.mov.b64(out0, out1, packed)
    return out0, out1


def _ffma2(a0, a1, b0, b1, c0, c1):
    packed = txl.local_scalar("uint64")
    out0 = txl.local_scalar("float32")
    out1 = txl.local_scalar("float32")
    txl.ptx.fma.rn.f32x2(
        packed, txl.cuda.make_float2(a0, a1), txl.cuda.make_float2(b0, b1), txl.cuda.make_float2(c0, c1)
    )
    txl.ptx.mov.b64(out0, out1, packed)
    return out0, out1


def _e8m0_inverse(scale_byte):
    bits = txl.local_scalar("int32")
    result = txl.local_scalar("float32")
    txl.ptx.sub.s32(bits, txl.int32(254), txl.cast(scale_byte, "int32"))
    txl.ptx.shl.b32(bits, bits, txl.uint32(23))
    txl.ptx.mov.b32(result, bits)
    return result


def _absmax(lhs, rhs):
    absolute = txl.local_scalar("float32")
    result = txl.local_scalar("float32")
    txl.ptx.abs.f32(absolute, rhs)
    txl.ptx.max.f32(result, lhs, absolute)
    return result


def _shuffle_xor_f32(value, delta):
    shuffled = txl.local_scalar("uint32")
    txl.ptx.shfl_sync.bfly.b32(
        shuffled,
        txl.reinterpret("uint32", value),
        txl.uint32(delta),
        txl.uint32(31),
        txl.uint32(0xFFFFFFFF),
    )
    return txl.reinterpret("float32", shuffled)


def _config(label, tokens, w_out_in):
    return {
        "label": label,
        "tokens": tokens,
        "k_dim": _K_DIM,
        "num_heads": _NUM_HEADS,
        "w_out_in": w_out_in,
    }


KERNEL_META = {
    "name": "cudnn_sm100_gemm_proj_rope_mxfp8_bf16in",
    "category": "cudnn",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "reference_requirements": (
        {
            "package": "nvidia-cudnn-frontend",
            "git": {
                "url": "https://github.com/NVIDIA/cudnn-frontend.git",
                "commit": "aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5",
            },
            "import": "cudnn",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.8.0.dev0", "import": "cutlass"},
    ),
}

CONFIGS = [
    _config("t128_k1536_h128_w_out_in_false", 128, False),
    _config("t128_k1536_h128_w_out_in_true", 128, True),
    _config("t256_k1536_h128_w_out_in_false", 256, False),
    _config("t256_k1536_h128_w_out_in_true", 256, True),
    _config("t2048_k1536_h128_w_out_in_false", 2048, False),
    _config("t2048_k1536_h128_w_out_in_true", 2048, True),
    _config("t4096_k1536_h128_w_out_in_false", 4096, False),
    _config("t4096_k1536_h128_w_out_in_true", 4096, True),
]

BENCH_CONFIGS = [dict(config) for config in CONFIGS if config["tokens"] in (2048, 4096)]


def _validate_config(tokens, k_dim, num_heads, w_out_in):
    if tokens <= 0 or tokens % _TILE_M:
        raise ValueError(f"tokens must be a positive multiple of {_TILE_M}, got {tokens}")
    if k_dim != _K_DIM:
        raise ValueError(f"the verified BF16 projection specialization requires k_dim={_K_DIM}")
    if num_heads != _NUM_HEADS:
        raise ValueError(
            f"the verified BF16 projection specialization requires num_heads={_NUM_HEADS}"
        )
    if type(w_out_in) is not bool:
        raise TypeError("w_out_in must be bool")


@cache
def _make_kernel(tokens, k_dim, num_heads, w_out_in):
    _validate_config(tokens, k_dim, num_heads, w_out_in)
    num_clusters = min((tokens // _TILE_M) * num_heads, _MAX_ACTIVE_CLUSTERS)
    total_work = (tokens // _TILE_M) * num_heads
    m_tiles = tokens // _TILE_M
    b_desc_base = 0x4000404000010000 if w_out_in else 0x4000404002000000

    def host_prelude(params):
        x = params["x"]
        w = params["w"]
        a_map = txl.stack_alloca("tensormap", 1)
        b_map = txl.stack_alloca("tensormap", 1)

        txl.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            a_map,
            "bfloat16",
            2,
            x.data,
            k_dim,
            tokens,
            k_dim * 2,
            64,
            128,
            1,
            1,
            0,
            3,
            2,
            0,
        )
        if w_out_in:
            txl.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                b_map,
                "bfloat16",
                2,
                w.data,
                k_dim,
                num_heads * _HEAD_DIM,
                k_dim * 2,
                64,
                _HEAD_DIM,
                1,
                1,
                0,
                3,
                2,
                0,
            )
        else:
            txl.call_packed(
                "runtime.cuTensorMapEncodeTiled",
                b_map,
                "bfloat16",
                2,
                w.data,
                num_heads * _HEAD_DIM,
                k_dim,
                num_heads * _HEAD_DIM * 2,
                64,
                64,
                1,
                1,
                0,
                3,
                2,
                0,
            )
        return a_map, b_map

    def kernel(x, w, cos, sin, out_fp8_row, out_scales_row, out_fp8_col, out_scales_col, *, host):
        # TIRX_PORT_START: gemm_proj_rope_mxfp8_kernel
        del x, w
        a_map, b_map = host
        _block_x, _block_y, cluster_work_id = txl.cta_id()
        del _block_x, _block_y
        warp = txl.warp_id()
        lane = txl.lane_id()

        roles = txl.specialize(chain_dispatch=True)
        epilogue_role = roles.role("epilogue", warps=list(range(12)))
        mma_role = roles.role("mma", warps=[12])
        tma_role = roles.role("tma", warps=[13])

        smem = txl.alloc_buffer((_SHARED_BYTES,), txl.u8, scope="shared.dyn", align=1024)
        protocol_pool = txl.smem_pool(base=smem)
        ab_pipe = txl.Pipeline(
            protocol_pool,
            _AB_STAGES,
            full="tma",
            empty="tcgen05",
            init_empty=1,
            leader=txl.bool(False),
        )
        acc_pipe = txl.Pipeline(
            protocol_pool,
            _ACC_STAGES,
            full="tcgen05",
            empty="mbar",
            init_empty=4,
            leader=txl.bool(False),
        )
        if protocol_pool.bytes != _TMEM_DEALLOC_OFFSET:
            raise AssertionError("pipeline protocol layout changed")
        protocol_pool.alloc((1,), txl.u64, align=8)
        tmem_slot = protocol_pool.alloc((1,), txl.u32, align=4)
        if protocol_pool.bytes != _TMEM_PTR_OFFSET + 4:
            raise AssertionError("TMEM protocol layout changed")

        with tma_role:
            txl.ptx.prefetch.tensormap(txl.address_of(a_map))
            txl.ptx.prefetch.tensormap(txl.address_of(b_map))

        # AB and ACC publication retain the source's two fence/barrier edges.
        with txl.If(warp == 0):
            with txl.Then():
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, _AB_STAGES) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                ab_pipe.full.ptr_to([stage]), txl.uint32(1)
                            )
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, _AB_STAGES) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                ab_pipe.empty.ptr_to([stage]), txl.uint32(1)
                            )
        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(448))
        with txl.If(warp == 0):
            with txl.Then():
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, _ACC_STAGES) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                acc_pipe.full.ptr_to([stage]), txl.uint32(1)
                            )
                with txl.If(_elected()):
                    with txl.Then():
                        with txl.unroll(0, _ACC_STAGES) as stage:
                            txl.ptx.mbarrier.init.shared.b64(
                                acc_pipe.empty.ptr_to([stage]), txl.uint32(4)
                            )
        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.ptx.bar.sync(txl.uint32(0), txl.uint32(448))

        smem_base = txl.local_scalar("uint32")
        txl.assign(smem_base, txl.cuda.cvta_generic_to_shared(smem.ptr_to([0])))
        cluster_smem_u64 = txl.local_scalar("uint64")
        txl.ptx.cvta.to.shared__cluster.u64(cluster_smem_u64, smem.ptr_to([0]))
        cluster_smem = txl.local_scalar("uint32", init=txl.cast(cluster_smem_u64, "uint32"))
        a_descriptor = txl.local_scalar(
            "uint64", init=_descriptor_with_address(_A_DESC_BASE, smem_base + _A_OFFSET)
        )
        b_descriptor = txl.local_scalar(
            "uint64", init=_descriptor_with_address(b_desc_base, smem_base + _B_OFFSET)
        )

        def scheduler_coords(work):
            head_minor = work & 7
            quotient = work // 8
            m_idx = quotient % m_tiles
            head_major = quotient // m_tiles
            return m_idx, head_major * 8 + head_minor

        def advance_work(work):
            txl.assign(work, work + num_clusters)

        # CuTe emits this logical named-barrier rendezvous at two branch-local
        # PCs.  Synccheck rejects that source spelling as divergent (the pinned
        # source reports 21472 errors), so the same 13-warp arrival set is
        # represented at one physical PC while retaining the exact barrier id,
        # count, allocation owner, and publication edge.
        with txl.If(warp < txl.uint32(13)):
            with txl.Then():
                with txl.If(warp == 0):
                    with txl.Then():
                        txl.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                            tmem_slot.ptr_to([0]), txl.uint32(_TMEM_COLUMNS)
                        )
                txl.ptx.bar.sync(txl.uint32(2), txl.uint32(416))

        with tma_role:
            tma_state = txl.PipelineState(_AB_STAGES, phase=1)
            work = txl.local_scalar("int32", init=cluster_work_id)
            with txl.While(work < total_work):
                m_idx, head = scheduler_coords(work)
                with txl.serial(_K_DIM // 64) as k_tile:
                    _wait_plain(ab_pipe.empty.ptr_to([tma_state.stage]), tma_state.phase)
                    with txl.If(_elected()):
                        with txl.Then():
                            txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                                ab_pipe.full.ptr_to([tma_state.stage]), txl.uint32(40_960)
                            )
                    with txl.If(_elected()):
                        with txl.Then():
                            txl.ptx[_TMA_G2S_2D](
                                cluster_smem + _A_OFFSET + tma_state.stage * _A_STAGE_BYTES,
                                txl.address_of(a_map),
                                txl.int32(k_tile * 64),
                                txl.cast(m_idx * _TILE_M, "int32"),
                                ab_pipe.full.ptr_to([tma_state.stage]),
                                txl.uint16(1),
                                txl.uint64(0),
                            )
                    if w_out_in:
                        with txl.If(_elected()):
                            with txl.Then():
                                txl.ptx[_TMA_G2S_2D](
                                    cluster_smem + _B_OFFSET + tma_state.stage * _B_STAGE_BYTES,
                                    txl.address_of(b_map),
                                    txl.int32(k_tile * 64),
                                    txl.cast(head * _HEAD_DIM, "int32"),
                                    ab_pipe.full.ptr_to([tma_state.stage]),
                                    txl.uint16(1),
                                    txl.uint64(0),
                                )
                    else:
                        for copy in range(3):
                            with txl.If(_elected()):
                                with txl.Then():
                                    txl.ptx[_TMA_G2S_2D](
                                        cluster_smem
                                        + _B_OFFSET
                                        + tma_state.stage * _B_STAGE_BYTES
                                        + copy * 8192,
                                        txl.address_of(b_map),
                                        txl.cast(head * _HEAD_DIM + copy * 64, "int32"),
                                        txl.int32(k_tile * 64),
                                        ab_pipe.full.ptr_to([tma_state.stage]),
                                        txl.uint16(1),
                                        txl.uint64(0),
                                    )
                    _advance(tma_state)
                advance_work(work)
            for _ in range(_AB_STAGES):
                _wait_plain(ab_pipe.empty.ptr_to([tma_state.stage]), tma_state.phase)
                _advance(tma_state)

        with mma_role:
            tmem_base = txl.local_scalar("uint32")
            txl.ptx.ld.shared.b32(tmem_base, tmem_slot.ptr_to([0]))
            instruction_descriptor = txl.alloc_local((1,), "uint32")
            txl.cuda.tcgen05.encode_instr_descriptor(
                txl.address_of(instruction_descriptor[0]),
                d_dtype="float32",
                a_dtype="bfloat16",
                b_dtype="bfloat16",
                M=128,
                N=192,
                K=16,
                trans_a=False,
                trans_b=not w_out_in,
                n_cta_groups=1,
            )
            mma_state = txl.PipelineState(_AB_STAGES, phase=0)
            acc_state = txl.PipelineState(_ACC_STAGES, phase=1)
            work = txl.local_scalar("int32", init=cluster_work_id)
            accumulate = txl.local_scalar("uint32")
            with txl.While(work < total_work):
                _wait_plain(acc_pipe.empty.ptr_to([acc_state.stage]), acc_state.phase)
                txl.assign(accumulate, txl.uint32(0))
                with txl.serial(_K_DIM // 64) as _k_tile:
                    _wait_plain(ab_pipe.full.ptr_to([mma_state.stage]), mma_state.phase)
                    for kphase in range(4):
                        with txl.If(_elected()):
                            with txl.Then():
                                txl.ptx[_MMA_F16](
                                    txl.cast(tmem_base + acc_state.stage * _HEAD_DIM, "uint32"),
                                    a_descriptor
                                    + txl.cast(
                                        mma_state.stage * (_A_STAGE_BYTES // 16) + kphase * 2,
                                        "uint64",
                                    ),
                                    b_descriptor
                                    + txl.cast(
                                        mma_state.stage * (_B_STAGE_BYTES // 16)
                                        + kphase * (2 if w_out_in else 128),
                                        "uint64",
                                    ),
                                    instruction_descriptor[0],
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.ptx.pred(txl.cast(accumulate, "bool")),
                                )
                        txl.assign(accumulate, txl.uint32(1))
                    with txl.If(_elected()):
                        with txl.Then():
                            txl.ptx[
                                "tcgen05.commit.cta_group::1.mbarrier::arrive::one."
                                "shared::cluster.b64"
                            ](ab_pipe.empty.ptr_to([mma_state.stage]))
                    _advance(mma_state)
                with txl.If(_elected()):
                    with txl.Then():
                        txl.ptx[
                            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
                        ](acc_pipe.full.ptr_to([acc_state.stage]))
                _advance(acc_state)
                advance_work(work)
            _advance(acc_state)
            _wait_plain(acc_pipe.empty.ptr_to([acc_state.stage]), acc_state.phase)

        with epilogue_role:
            tmem_base = txl.local_scalar("uint32")
            txl.ptx.ld.shared.b32(tmem_base, tmem_slot.ptr_to([0]))
            acc_state = txl.PipelineState(_ACC_STAGES, phase=0)
            work = txl.local_scalar("int32", init=cluster_work_id)
            values0 = txl.alloc_local((_BLOCK,), "float32")
            values1 = txl.alloc_local((_BLOCK,), "float32")
            drain = txl.alloc_local((32,), "float32")
            staged_words = txl.alloc_local((16,), "uint32")

            with txl.While(work < total_work):
                m_idx, head = scheduler_coords(work)
                token_base = m_idx * _TILE_M

                with txl.If(warp < 4):
                    with txl.Then():
                        _wait_plain(acc_pipe.full.ptr_to([acc_state.stage]), acc_state.phase)
                        for group in range(6):
                            tmem_address = txl.cast(
                                tmem_base + (warp << 21) + acc_state.stage * _HEAD_DIM + group * 32,
                                "uint32",
                            )
                            txl.ptx["tcgen05.ld.sync.aligned.32x32b.x32.b32"](
                                *[drain[index] for index in range(32)], tmem_address
                            )
                            for pair in range(16):
                                txl.ptx.cvt.rn.bf16x2.f32(
                                    staged_words[pair], drain[pair * 2 + 1], drain[pair * 2]
                                )
                            row = warp * 32 + lane
                            row_byte = _SACC_OFFSET + 2 * (row * _SACC_STRIDE + group * 32)
                            for vector in range(8):
                                txl.ptx.st.shared.v2.b32(
                                    smem.ptr_to([row_byte + vector * 8]),
                                    staged_words[vector * 2],
                                    staged_words[vector * 2 + 1],
                                )
                        with txl.If(_elected()):
                            with txl.Then():
                                txl.ptx.mbarrier.arrive.shared.b64(
                                    acc_pipe.empty.ptr_to([acc_state.stage]), txl.uint32(1)
                                )

                txl.ptx.bar.sync(txl.uint32(1), txl.uint32(384))
                cb = warp // 3
                fc = warp % 3
                tok0 = cb * _BLOCK
                f0 = fc * 64 + lane * 2
                f1 = f0 + 1
                col_amax0 = txl.local_scalar("float32", init=txl.float32(0.0))
                col_amax1 = txl.local_scalar("float32", init=txl.float32(0.0))

                with txl.If(fc == 2):
                    with txl.Then():
                        lib = lane % 16
                        pcol0 = 128 + 4 * lib
                        pcol1 = pcol0 + 2
                        roff = 32 * (lane // 16)
                        rf = txl.cast(lane // 16, "float32")
                        lf = txl.float32(1.0) - rf
                        cidx0 = 2 * lib + roff
                        for row in range(_BLOCK):
                            shared_byte = _SACC_OFFSET + 2 * ((tok0 + row) * _SACC_STRIDE + pcol0)
                            p0_b16 = txl.local_scalar("uint16")
                            q0_b16 = txl.local_scalar("uint16")
                            p1_b16 = txl.local_scalar("uint16")
                            q1_b16 = txl.local_scalar("uint16")
                            txl.ptx.ld.shared.v4.b16(
                                p0_b16, q0_b16, p1_b16, q1_b16, smem.ptr_to([shared_byte])
                            )
                            p0 = txl.local_scalar("float32")
                            q0 = txl.local_scalar("float32")
                            p1 = txl.local_scalar("float32")
                            q1 = txl.local_scalar("float32")
                            txl.ptx.cvt.f32.bf16(p0, p0_b16)
                            txl.ptx.cvt.f32.bf16(q0, q0_b16)
                            txl.ptx.cvt.f32.bf16(p1, p1_b16)
                            txl.ptx.cvt.f32.bf16(q1, q1_b16)
                            cos0_b16 = txl.local_scalar("uint16")
                            cos1_b16 = txl.local_scalar("uint16")
                            sin0_b16 = txl.local_scalar("uint16")
                            sin1_b16 = txl.local_scalar("uint16")
                            trig_offset = (token_base + tok0 + row) * _QK_ROPE + cidx0
                            txl.ptx.ld.global_.v2.b16(cos0_b16, cos1_b16, cos.ptr_to([trig_offset]))
                            txl.ptx.ld.global_.v2.b16(sin0_b16, sin1_b16, sin.ptr_to([trig_offset]))
                            c0 = txl.local_scalar("float32")
                            c1 = txl.local_scalar("float32")
                            s0 = txl.local_scalar("float32")
                            s1 = txl.local_scalar("float32")
                            txl.ptx.cvt.f32.bf16(c0, cos0_b16)
                            txl.ptx.cvt.f32.bf16(c1, cos1_b16)
                            txl.ptx.cvt.f32.bf16(s0, sin0_b16)
                            txl.ptx.cvt.f32.bf16(s1, sin1_b16)
                            pc0, pc1 = _fmul2(p0, p1, c0, c1)
                            qs0, qs1 = _fmul2(q0, q1, s0, s1)
                            lft0, lft1 = _ffma2(
                                qs0, qs1, txl.float32(-1.0), txl.float32(-1.0), pc0, pc1
                            )
                            ps0, ps1 = _fmul2(p0, p1, s0, s1)
                            qc0, qc1 = _fmul2(q0, q1, c0, c1)
                            rgt0, rgt1 = _fadd2(ps0, ps1, qc0, qc1)
                            ll0, ll1 = _fmul2(lft0, lft1, lf, lf)
                            v0, v1 = _ffma2(rgt0, rgt1, rf, rf, ll0, ll1)
                            txl.assign(values0[row], v0)
                            txl.assign(values1[row], v1)
                            txl.assign(col_amax0, _absmax(col_amax0, v0))
                            txl.assign(col_amax1, _absmax(col_amax1, v1))
                    with txl.Else():
                        for row in range(_BLOCK):
                            shared_byte = _SACC_OFFSET + 2 * ((tok0 + row) * _SACC_STRIDE + f0)
                            v0_b16 = txl.local_scalar("uint16")
                            v1_b16 = txl.local_scalar("uint16")
                            txl.ptx.ld.shared.v2.b16(v0_b16, v1_b16, smem.ptr_to([shared_byte]))
                            v0 = txl.local_scalar("float32")
                            v1 = txl.local_scalar("float32")
                            txl.ptx.cvt.f32.bf16(v0, v0_b16)
                            txl.ptx.cvt.f32.bf16(v1, v1_b16)
                            txl.assign(values0[row], v0)
                            txl.assign(values1[row], v1)
                            txl.assign(col_amax0, _absmax(col_amax0, v0))
                            txl.assign(col_amax1, _absmax(col_amax1, v1))

                col_scale_pair = txl.local_scalar("uint16")
                txl.ptx.cvt.rp.satfinite.ue8m0x2.f32(
                    col_scale_pair,
                    col_amax0 * txl.float32(1.0 / 448.0),
                    col_amax1 * txl.float32(1.0 / 448.0),
                )
                col_scale0 = txl.cast(
                    txl.shift_right(txl.cast(col_scale_pair, "uint32"), txl.uint32(8)), "uint8"
                )
                col_scale1 = txl.cast(
                    txl.bitwise_and(txl.cast(col_scale_pair, "uint32"), txl.uint32(0xFF)), "uint8"
                )
                inv_col0 = _e8m0_inverse(col_scale0)
                inv_col1 = _e8m0_inverse(col_scale1)
                col_scale_base = ((m_idx * 4 + cb) * num_heads + head) * _HEAD_DIM + f0
                txl.ptx.st.global_.b8(out_scales_col.ptr_to([col_scale_base]), col_scale0)
                txl.ptx.st.global_.b8(out_scales_col.ptr_to([col_scale_base + 1]), col_scale1)

                row_block = fc * 2 + lane // 16
                for row in range(_BLOCK):
                    v0 = values0[row]
                    v1 = values1[row]
                    absolute0 = txl.local_scalar("float32")
                    absolute1 = txl.local_scalar("float32")
                    row_amax = txl.local_scalar("float32")
                    txl.ptx.abs.f32(absolute0, v0)
                    txl.ptx.abs.f32(absolute1, v1)
                    txl.ptx.max.f32(row_amax, absolute0, absolute1)
                    for delta in (8, 4, 2, 1):
                        other = _shuffle_xor_f32(row_amax, delta)
                        txl.ptx.max.f32(row_amax, row_amax, other)
                    row_scale_pair = txl.local_scalar("uint16")
                    txl.ptx.cvt.rp.satfinite.ue8m0x2.f32(
                        row_scale_pair, txl.float32(0.0), row_amax * txl.float32(1.0 / 448.0)
                    )
                    row_scale = txl.cast(
                        txl.bitwise_and(txl.cast(row_scale_pair, "uint32"), txl.uint32(0xFF)), "uint8"
                    )
                    inv_row = _e8m0_inverse(row_scale)
                    token = token_base + tok0 + row
                    with txl.If((lane % 16) == 0):
                        with txl.Then():
                            row_scale_offset = (token * num_heads + head) * (
                                _HEAD_DIM // _BLOCK
                            ) + row_block
                            txl.ptx.st.global_.b8(
                                out_scales_row.ptr_to([row_scale_offset]), row_scale
                            )
                    vr0, vr1 = _fmul2(v0, v1, inv_row, inv_row)
                    vc0, vc1 = _fmul2(v0, v1, inv_col0, inv_col1)
                    row_pair = txl.local_scalar("uint16")
                    col_pair = txl.local_scalar("uint16")
                    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(row_pair, vr1, vr0)
                    txl.ptx.cvt.rn.satfinite.e4m3x2.f32(col_pair, vc1, vc0)
                    output_offset = (token * num_heads + head) * _HEAD_DIM + f0
                    txl.ptx.st.global_.b16(out_fp8_row.ptr_to([output_offset]), row_pair)
                    txl.ptx.st.global_.b16(out_fp8_col.ptr_to([output_offset]), col_pair)

                txl.ptx.bar.sync(txl.uint32(1), txl.uint32(384))
                with txl.If(warp < 4):
                    with txl.Then():
                        _advance(acc_state)
                advance_work(work)

            with txl.If(warp == 0):
                with txl.Then():
                    txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
                    txl.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](
                        tmem_base, txl.uint32(_TMEM_COLUMNS)
                    )

    kernel.__annotations__ = {
        "x": txl.gptr[txl.bf16, (tokens * k_dim,)],
        "w": txl.gptr[txl.bf16, (num_heads * _HEAD_DIM * k_dim,)],
        "cos": txl.gptr[txl.bf16, (tokens * _QK_ROPE,)],
        "sin": txl.gptr[txl.bf16, (tokens * _QK_ROPE,)],
        "out_fp8_row": txl.gptr[txl.u8, (tokens * num_heads * _HEAD_DIM,)],
        "out_scales_row": txl.gptr[txl.u8, (tokens * num_heads * (_HEAD_DIM // _BLOCK),)],
        "out_fp8_col": txl.gptr[txl.u8, (tokens * num_heads * _HEAD_DIM,)],
        "out_scales_col": txl.gptr[txl.u8, ((tokens // _BLOCK) * num_heads * _HEAD_DIM,)],
    }
    return txl.kernel(warps=14, arch="sm_100a", grid=[1, 1, num_clusters], host_prelude=host_prelude)(
        kernel
    )


def get_kernel(tokens, k_dim, num_heads, w_out_in):
    return _make_kernel(tokens, k_dim, num_heads, w_out_in).func


def prepare_data(tokens, k_dim, num_heads, w_out_in):
    """Allocate deterministic source-equivalent inputs and independent outputs."""
    _validate_config(tokens, k_dim, num_heads, w_out_in)
    import torch

    torch.manual_seed(0)
    x = torch.randn(tokens, k_dim, dtype=torch.bfloat16, device="cuda") * 0.5
    w_shape = (num_heads * _HEAD_DIM, k_dim) if w_out_in else (k_dim, num_heads * _HEAD_DIM)
    w = torch.randn(w_shape, dtype=torch.bfloat16, device="cuda") * 0.02
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

    return {"x": x, "w": w, "cos": cos, "sin": sin, "tirx": outputs(), "source": outputs()}


def _without_label(config):
    return {key: value for key, value in config.items() if key != "label"}


def _tirx_launch(executable, data):
    import torch

    output = data["tirx"]
    x = data["x"].reshape(-1)
    w = data["w"].reshape(-1)
    cos = data["cos"].reshape(-1)
    sin = data["sin"].reshape(-1)
    qrow = output["out_fp8_row"].view(torch.uint8).reshape(-1)
    srow = output["out_scales_row"].reshape(-1)
    qcol = output["out_fp8_col"].view(torch.uint8).reshape(-1)
    scol = output["out_scales_col"].reshape(-1)

    def launch():
        executable(x, w, cos, sin, qrow, srow, qcol, scol)

    launch._keep_alive = (x, w, cos, sin, qrow, srow, qcol, scol)
    return launch


def _compile_reference(data, config):
    from tirx_kernels.cudnn._shared._reference import load_reference_module

    module = load_reference_module("cudnn.gemm.cutedsl.dense.proj_rope_mxfp8.api")
    output = data["source"]
    op = module.GemmProjRopeMxfp8Bf16InSm100(
        data["x"],
        data["w"],
        data["cos"],
        data["sin"],
        output["out_fp8_row"],
        output["out_scales_row"],
        output["out_fp8_col"],
        output["out_scales_col"],
        w_out_in=config["w_out_in"],
    )
    if not op.check_support():
        raise RuntimeError(f"pinned source rejected {config}")
    op.compile()

    def launch():
        op.execute(
            data["x"],
            data["w"],
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
    decoded_scale = torch.pow(2.0, scale.float() - 127.0).unsqueeze(-1)
    return (
        data.float().reshape(tokens, num_heads, _HEAD_DIM // _BLOCK, _BLOCK) * decoded_scale
    ).reshape(tokens, num_heads, _HEAD_DIM)


def _dequantize_col(torch, data, scale):
    tokens, num_heads, _ = data.shape
    decoded_scale = torch.pow(2.0, scale.float() - 127.0).reshape(
        tokens // _BLOCK, 1, num_heads, _HEAD_DIM
    )
    return (
        data.float().reshape(tokens // _BLOCK, _BLOCK, num_heads, _HEAD_DIM) * decoded_scale
    ).reshape(tokens, num_heads, _HEAD_DIM)


def _match_fraction(torch, actual, expected):
    difference = (actual.float() - expected.float()).abs()
    return float((difference <= 0.1 + 0.1 * expected.float().abs()).float().mean().item())


def _validate_outputs(data):
    """Apply the exact match-fraction contract used by the pinned source test."""
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
    """Compare the pure-K port with the pinned cuDNN Frontend implementation."""
    import torch

    from tirx_kernels.bench.runner import compile_kernel

    kernel_config = _without_label(config)
    data = prepare_data(**kernel_config)
    tirx_launch = _tirx_launch(compile_kernel(get_kernel(**kernel_config)), data)
    source_launch = _compile_reference(data, kernel_config)
    tirx_launch()
    source_launch()
    torch.cuda.synchronize()
    return _validate_outputs(data)


def prepare_bench(**config):
    """Compile TIRx before entering the benchmark's GPU child."""
    from tirx_kernels.bench.runner import compile_kernel, prepared_gpu_benchmark

    kernel_config = _without_label(config)
    state = {"config": kernel_config, "executable": compile_kernel(get_kernel(**kernel_config))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **kwargs):
    """Validate once, then time closures containing exactly one kernel launch."""
    from tirx_kernels.bench.runner import bench, defer_gpu_interrupts, external_references_enabled

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
    source_launch = gpu_state["source_launch"]
    if not gpu_state["validated"]:
        tirx_launch()
        torch.cuda.synchronize()
        if with_source:
            with defer_gpu_interrupts():
                if source_launch is None:
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
