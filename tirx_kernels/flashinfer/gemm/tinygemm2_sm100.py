# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400),
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer TinyGEMM2 BF16 kernel port for SM100.

Upstream source: csrc/tinygemm2_sm100.cu.
"""

import ctypes
from functools import cache, lru_cache
from pathlib import Path
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.kern as K

KERNEL_META = {
    "name": "tinygemm2_sm100",
    "category": "flashinfer",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "reference_requirements": (
        {
            "package": "flashinfer-python",
            "git": {
                "url": "https://github.com/flashinfer-ai/flashinfer.git",
                "commit": "f2e04400e330fb2debe0bf8730d9424a1d37927f",
            },
            "import": "flashinfer",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.8.0.dev0", "import": "cutlass"},
    ),
}
CONFIGS = [
    {"label": "b1_o128_k720", "B": 1, "O": 128, "K": 720},
    {"label": "b2_o16_k256", "B": 2, "O": 16, "K": 256},
    {"label": "b4_o2880_k2880", "B": 4, "O": 2880, "K": 2880},
    {"label": "b7_o128_k4096", "B": 7, "O": 128, "K": 4096},
    {"label": "b8_o1024_k1024", "B": 8, "O": 1024, "K": 1024},
    {"label": "b13_o1024_k2048", "B": 13, "O": 1024, "K": 2048},
    {"label": "b16_o2880_k2880", "B": 16, "O": 2880, "K": 2880},
    {"label": "b64_o4096_k3072", "B": 64, "O": 4096, "K": 3072},
]

BENCH_CONFIGS = CONFIGS

THREADS = 384
WT_OFF = 1024
WT_STAGE_BYTES = 4 * 2048
ACT_STAGE_BYTES = 4 * 1024
RED_BYTES = 2048
BIAS_BYTES = 32
_TMA_G2S_2D = "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"


def _select_stage(B: int, O: int, K: int, num_sms: int) -> int:
    """Mirror FlashInfer's B200 stage-ring dispatch predicate."""
    total_ctas = (O + 15) // 16 * ((B + 7) // 8)
    return 4 if K <= 1024 or total_ctas > 2 * num_sms else 8


def _validate_problem(B: int, O: int, K: int) -> None:
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")
    if K < 64:
        raise ValueError(f"K must be at least 64, got {K}")
    if O < 16 or O % 16:
        raise ValueError(f"O must be a positive multiple of 16, got {O}")
    i32_max = (1 << 31) - 1
    if B > i32_max or O > i32_max or K > i32_max:
        raise ValueError("B, O, and K must fit signed int32")


def _require_supported_arch() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("TinyGEMM2 SM100 requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability not in {(10, 0), (10, 3), (10, 7)}:
        raise SkipTest(
            "TinyGEMM2 requires SM100/B200, SM103/GB300, or SM107, "
            f"got sm_{capability[0]}{capability[1]}"
        )


def _tma_2d_g2s(dst, tensor_map, x, y, barrier):
    K.ptx[_TMA_G2S_2D](dst, K.address_of(tensor_map), x, y, barrier)


def _mma_bf16(accum, a_frag, b_frag):
    K.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
        accum[0],
        accum[1],
        accum[2],
        accum[3],
        a_frag[0],
        a_frag[1],
        a_frag[2],
        a_frag[3],
        b_frag[0],
        b_frag[1],
        accum[0],
        accum[1],
        accum[2],
        accum[3],
    )


def _materialize(value):
    local = K.local_scalar(str(value.ty.dtype), init=value)
    return local


def _make_tinygemm2_kernel(stages: int, use_pdl: bool, grid_x: int, grid_y: int):
    @K.kernel(warps=12, arch="sm_100a", min_blocks_per_sm=1, grid=(grid_x, grid_y))
    def tinygemm2_sm100(
        a_tmap_wt: K.TensorMap,
        b_tmap_act: K.TensorMap,
        c_output: K.gptr[K.bf16],
        d_bias: K.gptr[K.bf16],
        a_M: K.i32,
        b_N: K.i32,
        c_K: K.i32,
    ):
        block_m_scope, block_n_scope = K.cta_id()
        # TIRX_TRANSCRIBE_START tinygemm2_sm100

        tid_u32 = _materialize(K.cast(K.thread_id(), "uint32"))
        tid = _materialize(K.cast(tid_u32, "int32"))
        warp = K.warp_id()
        lane = _materialize(tid % 32)
        lane_u32 = _materialize(tid_u32 % K.uint32(32))
        block_m = _materialize(block_m_scope)
        block_n = _materialize(block_n_scope)

        smem_total = 52352 if stages == 4 else 101504
        smem = K.smem_pool()
        init_leader = K.local_scalar(K.u32, init=K.uint32(0))
        weight_ready = K.TMABar(smem, stages, leader=init_leader != K.uint32(0))
        activation_ready = K.TMABar(smem, stages, leader=init_leader != K.uint32(0))
        consumed = K.MBarrier(smem, stages, phase_offset=1, leader=init_leader != K.uint32(0))
        if smem.bytes != 3 * stages * 8:
            raise AssertionError(f"unexpected TinyGEMM2 barrier header: {smem.bytes}")
        weight_smem = smem.alloc((stages, 64, 64), K.bf16, swizzle=K.SW128B)
        activation_smem = smem.alloc((stages, 32, 64), K.bf16, swizzle=K.SW128B)
        reduction_smem = smem.alloc((128, 4), K.f32, align=16)
        bias_smem = smem.alloc((BIAS_BYTES // 2,), K.bf16, align=2)
        expected_used = (
            WT_OFF + stages * (WT_STAGE_BYTES + ACT_STAGE_BYTES) + RED_BYTES + BIAS_BYTES
        )
        if smem.bytes != expected_used:
            raise AssertionError(
                f"unexpected TinyGEMM2 typed storage footprint: {smem.bytes} != {expected_used}"
            )
        smem.commit(smem_total)

        with K.If(tid == 0), K.Then():
            K.ptx.prefetch.tensormap(K.address_of(a_tmap_wt))
            K.ptx.prefetch.tensormap(K.address_of(b_tmap_act))

        with K.If(warp == 0), K.Then():
            K.assign(init_leader, K.cuda.elect_sync())

        weight_ready.init(1)
        activation_ready.init(1)
        consumed.init(32)

        with K.If(warp == 0), K.Then():
            K.ptx.fence.mbarrier_init.release.cluster()

        K.ptx.bar.sync(K.uint32(0))
        K.ptx.bar.sync(K.uint32(0))

        roles = K.specialize()
        compute = roles.role("compute", warps=range(4))
        weight = roles.role("weight", warps=range(4, 8))
        activation = roles.role("activation", warps=range(8, 12))

        with compute:
            k_loops_c = _materialize(K.truncdiv(c_K + 1023, 1024))
            mib_c = _materialize(block_m * 16)
            ni_c = _materialize(block_n * 8)
            with K.If(tid < 16), K.Then():
                bias_bits = K.alloc_local([1], "uint16")
                K.ptx.ld.global_.b16(bias_bits[0], d_bias.ptr_to([mib_c + tid]))
                K.ptx.st.shared.b16(bias_smem.ptr_to([tid]), bias_bits[0])

            accum = K.alloc_local((4,), "float32", align=4)
            for z in range(4):
                K.ptx.mov.b32(accum[z], K.float32(0))

            lane_div8 = _materialize(lane_u32 // K.uint32(8))
            lane_mod8 = _materialize(lane_u32 % K.uint32(8))
            row_wt = _materialize(lane_mod8 + lane_div8 % K.uint32(2) * K.uint32(8))
            col_off_wt = _materialize(lane_div8 // K.uint32(2))
            row_act = _materialize(lane_mod8)
            compute_state = K.PipelineState(stages // 4, phase=0)

            def compute_iter():
                stage_c = _materialize(
                    K.cast(warp, "uint32") + K.uint32(4) * K.cast(compute_state.stage, "uint32")
                )
                phase_c = _materialize(K.cast(compute_state.phase, "uint32"))
                weight_ready.wait(stage_c, phase_c)
                activation_ready.wait(stage_c, phase_c)

                with K.unroll(4) as su:
                    with K.unroll(4) as kii:
                        a_frag = K.alloc_local((4,), "uint32", align=4)
                        b_frag = K.alloc_local((2,), "uint32", align=4)
                        col_w: K.uint32 = K.uint32(2 * kii) + col_off_wt
                        K.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                            a_frag[0],
                            a_frag[1],
                            a_frag[2],
                            a_frag[3],
                            weight_smem[stage_c].m8n8(su * 16 + row_wt, col_w * K.uint32(8)),
                        )
                        col_a: K.uint32 = K.uint32(2 * kii) + lane_div8
                        K.ptx.ldmatrix.sync.aligned.m8n8.x2.shared.b16(
                            b_frag[0],
                            b_frag[1],
                            activation_smem[stage_c].m8n8(su * 8 + row_act, col_a * K.uint32(8)),
                        )
                        _mma_bf16(accum, a_frag, b_frag)

                K.ptx.fence.proxy.async_.shared__cta()
                K.ptx.mbarrier.arrive.release.cta.shared__cta.b64(consumed.ptr_to([stage_c]))
                compute_state.advance()

            with K.serial(0, k_loops_c, unroll=2, dtype="uint32"):
                compute_iter()

            accum_bits = K.alloc_local((4,), "uint32", align=4)
            for z in range(4):
                K.ptx.mov.b32(accum_bits[z], K.reinterpret("uint32", accum[z]))
            K.ptx.st.shared.v4.b32(
                reduction_smem.ptr_to([tid, 0]),
                accum_bits[0],
                accum_bits[1],
                accum_bits[2],
                accum_bits[3],
            )
            K.ptx.barrier.sync(K.uint32(2), K.uint32(THREADS))

            with K.If(warp == 0), K.Then():
                part_bits = K.alloc_local((12,), "uint32", align=4)
                part = part_bits.view("float32")
                for other_warp in range(3):
                    K.ptx.ld.shared.v4.b32(
                        part_bits[other_warp * 4],
                        part_bits[other_warp * 4 + 1],
                        part_bits[other_warp * 4 + 2],
                        part_bits[other_warp * 4 + 3],
                        reduction_smem.ptr_to([32 + other_warp * 32 + tid, 0]),
                    )

                for z in range(4):
                    K.ptx["add.ftz.f32"](accum[z], accum[z], part[z])
                    K.ptx["add.ftz.f32"](accum[z], accum[z], part[4 + z])
                    K.ptx["add.ftz.f32"](accum[z], accum[z], part[8 + z])

                tm = _materialize(mib_c + lane // 4)
                tn = _materialize(ni_c + 2 * (lane % 4))
                bias_bits = K.alloc_local([2], "uint16")
                K.ptx.ld.shared.b16(bias_bits[0], bias_smem.ptr_to([lane // 4]))
                K.ptx.ld.shared.b16(bias_bits[1], bias_smem.ptr_to([lane // 4 + 8]))
                bias_lo = _materialize(K.cast(K.reinterpret("bfloat16", bias_bits[0]), "float32"))
                bias_hi = _materialize(K.cast(K.reinterpret("bfloat16", bias_bits[1]), "float32"))
                out_frag = K.alloc_local((4,), "float32", align=4)
                K.ptx["add.ftz.f32"](out_frag[0], accum[0], bias_lo)
                K.ptx["add.ftz.f32"](out_frag[1], accum[1], bias_lo)
                K.ptx["add.ftz.f32"](out_frag[2], accum[2], bias_hi)
                K.ptx["add.ftz.f32"](out_frag[3], accum[3], bias_hi)
                out_base = _materialize(tn * a_M + tm)
                out_next = _materialize(out_base + a_M)

                with K.If(tn < b_N), K.Then():
                    with K.If(tm < a_M), K.Then():
                        K.ptx.st.global_.b16(
                            c_output.ptr_to([out_base]),
                            K.reinterpret("uint16", K.cast(out_frag[0], "bfloat16")),
                        )
                with K.If(tn + 1 < b_N), K.Then():
                    with K.If(tm < a_M), K.Then():
                        K.ptx.st.global_.b16(
                            c_output.ptr_to([out_next]),
                            K.reinterpret("uint16", K.cast(out_frag[1], "bfloat16")),
                        )
                with K.If(tn < b_N), K.Then():
                    with K.If(tm + 8 < a_M), K.Then():
                        K.ptx.st.global_.b16(
                            c_output.ptr_to([out_base + 8]),
                            K.reinterpret("uint16", K.cast(out_frag[2], "bfloat16")),
                        )
                with K.If(tn + 1 < b_N), K.Then():
                    with K.If(tm + 8 < a_M), K.Then():
                        K.ptx.st.global_.b16(
                            c_output.ptr_to([out_next + 8]),
                            K.reinterpret("uint16", K.cast(out_frag[3], "bfloat16")),
                        )

        with weight:
            k_loops_w = _materialize(K.truncdiv(c_K + 1023, 1024))
            mib_w = _materialize(block_m * 16)
            wslot = _materialize(K.cast(warp, "uint32") % K.uint32(4))
            weight_state = K.PipelineState(stages // 4, phase=0)
            with K.If(K.cuda.elect_sync()), K.Then():
                with K.serial(0, k_loops_w, unroll=False, dtype="uint32") as ki:
                    stage_w = _materialize(
                        wslot + K.uint32(4) * K.cast(weight_state.stage, "uint32")
                    )
                    k_base_w = _materialize(
                        K.cast((ki * K.uint32(4) + wslot) * K.uint32(256), "int32")
                    )
                    consumed.wait(stage_w, weight_state.phase)
                    K.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(
                        weight_ready.ptr_to([stage_w]), K.uint32(WT_STAGE_BYTES)
                    )
                    for box in range(4):
                        _tma_2d_g2s(
                            weight_smem[stage_w].ptr_to(box * 16, 0),
                            a_tmap_wt,
                            k_base_w + box * 64,
                            mib_w,
                            weight_ready.ptr_to([stage_w]),
                        )
                    weight_state.advance()
                if stages == 8:
                    drain_stage_w = _materialize(
                        wslot + K.uint32(4) * K.cast(weight_state.stage, "uint32")
                    )
                    consumed.wait(drain_stage_w, weight_state.phase)
            K.ptx.barrier.sync(K.uint32(2), K.uint32(THREADS))

        with activation:
            k_loops_a = _materialize(K.truncdiv(c_K + 1023, 1024))
            ni_a = _materialize(block_n * 8)
            aslot = _materialize(K.cast(warp, "uint32") % K.uint32(4))
            activation_state = K.PipelineState(stages // 4, phase=0)
            with K.If(K.cuda.elect_sync()), K.Then():
                if use_pdl:
                    K.ptx.griddepcontrol.wait()
                    K.ptx.griddepcontrol.launch_dependents()
                with K.serial(0, k_loops_a, unroll=False, dtype="uint32") as ki:
                    stage_a = _materialize(
                        aslot + K.uint32(4) * K.cast(activation_state.stage, "uint32")
                    )
                    k_base_a = _materialize(
                        K.cast((ki * K.uint32(4) + aslot) * K.uint32(256), "int32")
                    )
                    consumed.wait(stage_a, activation_state.phase)
                    K.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(
                        activation_ready.ptr_to([stage_a]), K.uint32(ACT_STAGE_BYTES)
                    )
                    for box in range(4):
                        _tma_2d_g2s(
                            activation_smem[stage_a].ptr_to(box * 8, 0),
                            b_tmap_act,
                            k_base_a + box * 64,
                            ni_a,
                            activation_ready.ptr_to([stage_a]),
                        )
                    activation_state.advance()
                if stages == 8:
                    drain_stage_a = _materialize(
                        aslot + K.uint32(4) * K.cast(activation_state.stage, "uint32")
                    )
                    consumed.wait(drain_stage_a, activation_state.phase)
            K.ptx.barrier.sync(K.uint32(2), K.uint32(THREADS))

    return tinygemm2_sm100.func


def get_kernel(
    B: int,
    O: int,
    K: int,
    *,
    stage: int | None = None,
    use_pdl: bool = False,
    num_sms: int | None = None,
):
    _validate_problem(B, O, K)
    if stage is None:
        if num_sms is None:
            if not torch.cuda.is_available():
                raise ValueError("num_sms is required for automatic dispatch without CUDA")
            num_sms = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        stage = _select_stage(B, O, K, num_sms)
    if stage not in (4, 8):
        raise ValueError(f"stage must be 4 or 8, got {stage}")

    launch_params = ["blockIdx.x", "blockIdx.y", "threadIdx.x"]
    if use_pdl:
        launch_params.append("tirx.use_programtic_dependent_launch")
    launch_params.append("tirx.use_dyn_shared_memory")
    return _make_tinygemm2_kernel(stage, use_pdl, (O + 15) // 16, (B + 7) // 8).with_attr(
        "tirx.kernel_launch_params", launch_params
    )


class _AlignedTensorMap:
    """Host storage for one 128-byte-aligned TensorMap payload."""

    def __init__(self) -> None:
        self._storage = ctypes.create_string_buffer(128 + 128)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 127) & ~127)


def _encode_tensor_map(
    tensor: torch.Tensor,
    *,
    global_dims: tuple[int, int],
    global_stride_bytes: int,
    box_dims: tuple[int, int],
) -> _AlignedTensorMap:
    import tvm

    descriptor = _AlignedTensorMap()
    encode = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    encode(
        descriptor.ptr,
        "bfloat16",
        2,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *global_dims,
        global_stride_bytes,
        *box_dims,
        1,
        1,
        0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
        3,  # CU_TENSOR_MAP_SWIZZLE_128B
        0,  # CU_TENSOR_MAP_L2_PROMOTION_NONE
        0,  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    )
    return descriptor


def _build_tensor_maps(case: dict[str, Any]) -> dict[str, _AlignedTensorMap]:
    B, O, K = case["B"], case["O"], case["K"]
    return {
        "weight": _encode_tensor_map(
            case["weight"], global_dims=(K, O), global_stride_bytes=K * 2, box_dims=(64, 16)
        ),
        "input": _encode_tensor_map(
            case["input"], global_dims=(K, B), global_stride_bytes=K * 2, box_dims=(64, 8)
        ),
    }


def prepare_data(
    B: int, O: int, K: int, *, seed: int = 0, device: str | torch.device = "cuda"
) -> dict[str, Any]:
    _validate_problem(B, O, K)
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SkipTest("TinyGEMM2 SM100 data preparation requires CUDA")
    generator = torch.Generator(device=device).manual_seed(seed)
    input_tensor = (
        torch.randn((B, K), generator=generator, device=device, dtype=torch.float32) / 8
    ).bfloat16()
    weight = (
        torch.randn((O, K), generator=generator, device=device, dtype=torch.float32) / 8
    ).bfloat16()
    bias = torch.randn((O,), generator=generator, device=device, dtype=torch.float32).bfloat16()
    out = torch.zeros((B, O), dtype=torch.bfloat16, device=device)
    case = {
        "B": B,
        "O": O,
        "K": K,
        "input": input_tensor,
        "weight": weight,
        "bias": bias,
        "out": out,
    }
    case["tensor_maps"] = _build_tensor_maps(case)
    return case


def _tirx_args(case: dict[str, Any], output: torch.Tensor | None = None) -> tuple[Any, ...]:
    maps = case["tensor_maps"]
    return (
        maps["weight"].ptr,
        maps["input"].ptr,
        (case["out"] if output is None else output).view(-1),
        case["bias"],
        case["O"],
        case["B"],
        case["K"],
    )


@lru_cache(maxsize=1)
def _flashinfer_tinygemm2_spec():
    import flashinfer
    from flashinfer.jit import env as jit_env
    from flashinfer.jit import gen_jit_spec, sm100a_nvcc_flags

    filename = "tinygemm2_sm100.cu"
    candidates = (
        Path(flashinfer.__file__).resolve().parents[1] / "csrc" / filename,
        jit_env.FLASHINFER_CSRC_DIR / filename,
    )
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise RuntimeError(
            "FlashInfer TinyGEMM2 source is unavailable; checked " + ", ".join(map(str, candidates))
        )
    return gen_jit_spec(
        "tinygemm2_sm100",
        [source],
        extra_cuda_cflags=[*sm100a_nvcc_flags, "-gencode=arch=compute_103a,code=sm_103a"],
        extra_include_paths=[source.parent, source.parent.parent / "include"],
    )


@lru_cache(maxsize=1)
def _load_flashinfer_module():
    return _flashinfer_tinygemm2_spec().build_and_load()


def _flashinfer_variant(stage: int, use_pdl: bool):
    suffix = "_pdl" if use_pdl else ""
    return getattr(_load_flashinfer_module(), f"stage{stage}{suffix}_op")


@cache
def _compile_executable(B: int, O: int, stage: int, use_pdl: bool):
    from tirx_kernels.runner import compile_kernel

    return compile_kernel(get_kernel(B, O, 1024, stage=stage, use_pdl=use_pdl))


def _run_tirx(case: dict[str, Any], stage: int, use_pdl: bool, output: torch.Tensor) -> None:
    executable = _compile_executable(case["B"], case["O"], stage, use_pdl)
    executable(*_tirx_args(case, output))


def _run_flashinfer(case: dict[str, Any], stage: int, use_pdl: bool, output: torch.Tensor) -> None:
    _flashinfer_variant(stage, use_pdl)(case["input"], case["weight"], case["bias"], output)


def run_test(B: int, O: int, K: int) -> None:
    _require_supported_arch()
    case = prepare_data(B, O, K)
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    stage = _select_stage(B, O, K, num_sms)

    for use_pdl in (False, True):
        tirx_out = torch.zeros_like(case["out"])
        flashinfer_out = torch.zeros_like(case["out"])
        _run_tirx(case, stage, use_pdl, tirx_out)
        _run_flashinfer(case, stage, use_pdl, flashinfer_out)
        torch.cuda.synchronize()
        if not torch.equal(tirx_out, flashinfer_out):
            differing = int((tirx_out != flashinfer_out).sum().item())
            max_diff = float((tirx_out.float() - flashinfer_out.float()).abs().max().item())
            raise AssertionError(
                f"TinyGEMM2 bitwise mismatch for B={B}, O={O}, K={K}, "
                f"stage={stage}, use_pdl={use_pdl}: {differing} elements, "
                f"max_abs_diff={max_diff}"
            )


def prepare_bench(B: int, O: int, K: int):
    """Compile the hardware-profile dispatch before CUDA initialization."""
    from tirx_kernels.runner import hardware_num_sms, prepared_gpu_benchmark

    stage = _select_stage(B, O, K, hardware_num_sms())
    state = {
        "B": B,
        "O": O,
        "K": K,
        "stage": stage,
        "executable": _compile_executable(B, O, stage, False),
    }
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(
    prepared,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int = 5,
    cooldown_s: float = 1.0,
) -> dict[str, Any]:
    _require_supported_arch()
    from tirx_kernels.runner import bench, external_references_enabled

    B, O, K = prepared["B"], prepared["O"], prepared["K"]
    stage = prepared["stage"]
    executable = prepared["executable"]
    case = prepare_data(B, O, K)
    args = _tirx_args(case)

    if external_references_enabled():
        reference_out = torch.zeros_like(case["out"])
        _run_flashinfer(case, stage, False, reference_out)
        executable(*args)
        torch.cuda.synchronize()
        if not torch.equal(case["out"], reference_out):
            raise AssertionError("TinyGEMM2 benchmark preflight failed bitwise validation")

    def _flashinfer_builder():
        output = torch.empty_like(case["out"])
        op = _flashinfer_variant(stage, False)
        op(case["input"], case["weight"], case["bias"], output)
        torch.cuda.synchronize()

        def launch():
            op(case["input"], case["weight"], case["bias"], output)

        return launch

    return bench(
        {"tirx": lambda: executable(*args)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashinfer_sm100": _flashinfer_builder},
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    B: int,
    O: int,
    K: int,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int = 5,
    cooldown_s: float = 1.0,
) -> dict[str, Any]:
    return prepare_bench(B, O, K).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
