"""FlashInfer TinyGEMM2 BF16 kernel port for SM100."""

from __future__ import annotations

import ctypes
import hashlib
from functools import cache, lru_cache
from typing import Any
from unittest import SkipTest

import torch
import torch.nn.functional as F

from tvm.script import tirx as T

KERNEL_META = {"name": "tinygemm2_sm100", "category": "gemm", "compute_capability": 10}

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
SOURCE_SHA256 = "ea4d87f058b269e2f15d04f945849d7b26604c5b76eed55a06eb8e08d7bc891d"
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


def _require_sm100() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("TinyGEMM2 SM100 requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 0):
        raise SkipTest(
            f"TinyGEMM2 SM100 requires SM100/B200, got sm_{capability[0]}{capability[1]}"
        )


def _make_warp_uniform(value):
    return T.cuda._shfl_sync(T.uint32(0xFFFFFFFF), value, 0, 32)


@T.inline
def _mbarrier_wait(smem_raw, byte_offset, phase):
    T.evaluate(T.cuda.mbarrier_wait(smem_raw.ptr_to([byte_offset]), phase))


@T.inline
def _mbarrier_wait_address(address, phase):
    T.evaluate(T.cuda.mbarrier_wait(address, phase))


@T.inline
def _mbarrier_arrive(smem_raw, byte_offset):
    T.ptx.mbarrier.arrive.release.cta.shared__cta.b64(smem_raw.ptr_to([byte_offset]))


@T.inline
def _mbarrier_arrive_address(address):
    T.ptx.mbarrier.arrive.release.cta.shared__cta.b64(address)


@T.inline
def _mbarrier_expect_tx(smem_raw, byte_offset, num_bytes):
    T.ptx.mbarrier.arrive.expect_tx.release.cta.shared__cta.b64(
        smem_raw.ptr_to([byte_offset]), T.uint32(num_bytes)
    )


@T.inline
def _tma_2d_g2s(smem_raw, dst_offset, tensor_map, x, y, barrier_offset):
    T.ptx[_TMA_G2S_2D](
        smem_raw.ptr_to([dst_offset]),
        T.address_of(tensor_map),
        x,
        y,
        smem_raw.ptr_to([barrier_offset]),
    )


@T.inline
def _mma_bf16(accum, a_frag, b_frag):
    T.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
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


@T.jit
def _tinygemm2_sm100(
    a_tmap_wt: T.TensorMap(),
    b_tmap_act: T.TensorMap(),
    c_output_h: T.handle,
    d_bias_h: T.handle,
    a_M: T.int32,
    b_N: T.int32,
    c_K: T.int32,
    *,
    STAGES: T.constexpr,
    USE_PDL: T.constexpr,
    GRID_X: T.constexpr,
    GRID_Y: T.constexpr,
):
    c_output = T.match_buffer(c_output_h, (b_N * a_M,), "bfloat16")
    d_bias = T.match_buffer(d_bias_h, (a_M,), "bfloat16")
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    # TIRX_TRANSCRIBE_START tinygemm2_sm100

    tid_u32 = T.thread_id([THREADS], dtype="uint32")
    tid: T.int32 = T.cast(tid_u32, "int32")
    warp = _make_warp_uniform(tid // 32)
    lane = tid % 32
    lane_u32: T.uint32 = tid_u32 % T.uint32(32)
    block_m, block_n = T.cta_id([GRID_X, GRID_Y])

    act_off = T.meta_var(33792 if STAGES == 4 else 66560)
    red_off = T.meta_var(50176 if STAGES == 4 else 99328)
    bias_off = T.meta_var(52224 if STAGES == 4 else 101376)
    smem_total = T.meta_var(52352 if STAGES == 4 else 101504)
    act_ready_off = T.meta_var(32 if STAGES == 4 else 64)
    consumed_off = T.meta_var(64 if STAGES == 4 else 128)

    pool = T.SMEMPool()
    smem_raw = pool.alloc((smem_total,), "uint8", align=1024)
    smem_bias = T.decl_buffer(
        (BIAS_BYTES // 2,),
        "bfloat16",
        data=smem_raw.data,
        scope="shared.dyn",
        byte_offset=bias_off,
        align=2,
    )
    pool.commit()

    if tid == 0:
        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(a_tmap_wt)))
        T.evaluate(T.ptx.prefetch.tensormap(T.address_of(b_tmap_act)))

    if warp == 0:
        leader = T.cuda.elect_sync()
        for stage_init in T.unroll(STAGES):
            T.ptx.mbarrier.init.shared__cta.b64(
                smem_raw.ptr_to([stage_init * 8]), T.uint32(1), pred=leader
            )
        for stage_init in T.unroll(STAGES):
            T.ptx.mbarrier.init.shared__cta.b64(
                smem_raw.ptr_to([act_ready_off + stage_init * 8]), T.uint32(1), pred=leader
            )
        for stage_init in T.unroll(STAGES):
            T.ptx.mbarrier.init.shared__cta.b64(
                smem_raw.ptr_to([consumed_off + stage_init * 8]), T.uint32(32), pred=leader
            )
        T.ptx.fence.mbarrier_init.release.cluster()

    T.ptx.bar.sync(T.uint32(0))
    T.ptx.bar.sync(T.uint32(0))

    if warp <= 3:
        k_loops_c: T.int32 = T.truncdiv(c_K + 1023, 1024)
        mib_c: T.int32 = block_m * 16
        ni_c: T.int32 = block_n * 8
        smem_addr_c: T.uint32 = T.cuda.cvta_generic_to_shared(smem_raw.ptr_to([0]))
        smem_wt_addr_c: T.uint32 = smem_addr_c + T.uint32(WT_OFF)
        smem_act_addr_c: T.uint32 = smem_addr_c + T.uint32(act_off)
        if tid < 16:
            smem_bias[tid] = d_bias[mib_c + tid]

        accum = T.alloc_local((4,), "float32", align=4)
        for z in T.unroll(4):
            accum[z] = T.float32(0)

        lane_div8: T.uint32 = lane_u32 // T.uint32(8)
        lane_mod8: T.uint32 = lane_u32 % T.uint32(8)
        row_wt: T.uint32 = lane_mod8 + lane_div8 % T.uint32(2) * T.uint32(8)
        col_off_wt: T.uint32 = lane_div8 // T.uint32(2)
        row_act: T.uint32 = lane_mod8

        @T.inline
        def compute_iter(ki):
            if STAGES == 4:
                stage_c: T.uint32 = T.cast(warp, "uint32")
                phase_c: T.uint32 = ki & T.uint32(1)
            else:
                stage_c: T.uint32 = T.cast(warp, "uint32") + T.uint32(4) * (ki % T.uint32(2))
                phase_c: T.uint32 = ki // T.uint32(2) & T.uint32(1)

            _mbarrier_wait_address(smem_addr_c + stage_c * T.uint32(8), phase_c)
            _mbarrier_wait_address(
                smem_addr_c + T.uint32(act_ready_off) + stage_c * T.uint32(8), phase_c
            )

            for su in T.unroll(4):
                base_wt: T.uint32 = smem_wt_addr_c + (
                    stage_c * T.uint32(4) + T.uint32(su)
                ) * T.uint32(2048)
                base_act: T.uint32 = smem_act_addr_c + (
                    stage_c * T.uint32(4) + T.uint32(su)
                ) * T.uint32(1024)
                for kii in T.unroll(4):
                    a_frag = T.alloc_local((4,), "uint32", align=4)
                    b_frag = T.alloc_local((2,), "uint32", align=4)
                    col_w: T.uint32 = T.uint32(2 * kii) + col_off_wt
                    col_sw_w: T.uint32 = row_wt % T.uint32(8) ^ col_w
                    T.ptx.ldmatrix.sync.aligned.m8n8.x4.shared.b16(
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        base_wt + row_wt * T.uint32(128) + col_sw_w * T.uint32(16),
                    )
                    col_a: T.uint32 = T.uint32(2 * kii) + lane_div8
                    col_sw_a: T.uint32 = row_act % T.uint32(8) ^ col_a
                    T.ptx.ldmatrix.sync.aligned.m8n8.x2.shared.b16(
                        b_frag[0],
                        b_frag[1],
                        base_act + row_act * T.uint32(128) + col_sw_a * T.uint32(16),
                    )
                    _mma_bf16(accum, a_frag, b_frag)

            T.ptx.fence.proxy.async_.shared__cta()
            _mbarrier_arrive_address(smem_addr_c + T.uint32(consumed_off) + stage_c * T.uint32(8))

        for ki in T.serial(0, k_loops_c, unroll=2, dtype="uint32"):
            compute_iter(ki)

        accum_bits = T.alloc_local((4,), "uint32", align=4)
        for z in T.unroll(4):
            accum_bits[z] = T.reinterpret("uint32", accum[z])
        T.ptx.st.shared.v4.b32(
            smem_addr_c + T.uint32(red_off) + tid_u32 * T.uint32(16),
            accum_bits[0],
            accum_bits[1],
            accum_bits[2],
            accum_bits[3],
        )
        T.ptx.bar.sync(T.uint32(2), T.uint32(THREADS))

        if warp == 0:
            part_bits = T.alloc_local((12,), "uint32", align=4)
            part = part_bits.view("float32")
            for other_warp in T.unroll(3):
                T.ptx.ld.shared.v4.b32(
                    part_bits[other_warp * 4],
                    part_bits[other_warp * 4 + 1],
                    part_bits[other_warp * 4 + 2],
                    part_bits[other_warp * 4 + 3],
                    smem_addr_c
                    + T.uint32(red_off)
                    + (T.uint32(32 + other_warp * 32) + tid_u32) * T.uint32(16),
                )

            for z in T.unroll(4):
                T.ptx["add.ftz.f32"](accum[z], accum[z], part[z])
                T.ptx["add.ftz.f32"](accum[z], accum[z], part[4 + z])
                T.ptx["add.ftz.f32"](accum[z], accum[z], part[8 + z])

            tm: T.int32 = mib_c + lane // 4
            tn: T.int32 = ni_c + 2 * (lane % 4)
            bias_lo: T.float32 = T.cast(smem_bias[lane // 4], "float32")
            bias_hi: T.float32 = T.cast(smem_bias[lane // 4 + 8], "float32")
            out_frag = T.alloc_local((4,), "float32", align=4)
            T.ptx["add.ftz.f32"](out_frag[0], accum[0], bias_lo)
            T.ptx["add.ftz.f32"](out_frag[1], accum[1], bias_lo)
            T.ptx["add.ftz.f32"](out_frag[2], accum[2], bias_hi)
            T.ptx["add.ftz.f32"](out_frag[3], accum[3], bias_hi)
            out_base: T.int32 = tn * a_M + tm
            out_next: T.int32 = out_base + a_M

            if tn < b_N:
                if tm < a_M:
                    c_output[out_base] = T.cast(out_frag[0], "bfloat16")
            if tn + 1 < b_N:
                if tm < a_M:
                    c_output[out_next] = T.cast(out_frag[1], "bfloat16")
            if tn < b_N:
                if tm + 8 < a_M:
                    c_output[out_base + 8] = T.cast(out_frag[2], "bfloat16")
            if tn + 1 < b_N:
                if tm + 8 < a_M:
                    c_output[out_next + 8] = T.cast(out_frag[3], "bfloat16")

    elif 4 <= warp and warp <= 7:
        k_loops_w: T.int32 = T.truncdiv(c_K + 1023, 1024)
        mib_w: T.int32 = block_m * 16
        wslot: T.uint32 = T.cast(warp, "uint32") % T.uint32(4)
        if T.cuda.elect_sync():
            for ki in T.serial(0, k_loops_w, unroll=False, dtype="uint32"):
                if STAGES == 4:
                    stage_w: T.uint32 = wslot
                    phase_w: T.uint32 = ki & T.uint32(1)
                else:
                    stage_w: T.uint32 = wslot + T.uint32(4) * (ki % T.uint32(2))
                    phase_w: T.uint32 = ki // T.uint32(2) & T.uint32(1)
                k_base_w: T.int32 = T.cast((ki * T.uint32(4) + wslot) * T.uint32(256), "int32")
                _mbarrier_wait(
                    smem_raw, T.uint32(consumed_off) + stage_w * T.uint32(8), phase_w ^ T.uint32(1)
                )
                _mbarrier_expect_tx(smem_raw, stage_w * T.uint32(8), 8192)
                for box in T.unroll(4):
                    _tma_2d_g2s(
                        smem_raw,
                        T.uint32(WT_OFF) + (stage_w * T.uint32(4) + T.uint32(box)) * T.uint32(2048),
                        a_tmap_wt,
                        k_base_w + box * 64,
                        mib_w,
                        stage_w * T.uint32(8),
                    )
            if STAGES == 8:
                dki_w: T.uint32 = T.cast(k_loops_w, "uint32")
                dstage_w: T.uint32 = wslot + T.uint32(4) * (dki_w % T.uint32(2))
                dphase_w: T.uint32 = dki_w // T.uint32(2) & T.uint32(1)
                _mbarrier_wait(
                    smem_raw,
                    T.uint32(consumed_off) + dstage_w * T.uint32(8),
                    dphase_w ^ T.uint32(1),
                )
        T.ptx.bar.sync(T.uint32(2), T.uint32(THREADS))

    elif 8 <= warp and warp <= 11:
        k_loops_a: T.int32 = T.truncdiv(c_K + 1023, 1024)
        ni_a: T.int32 = block_n * 8
        aslot: T.uint32 = T.cast(warp, "uint32") % T.uint32(4)
        if T.cuda.elect_sync():
            if USE_PDL:
                T.evaluate(T.ptx.griddepcontrol.wait())
                T.ptx.griddepcontrol.launch_dependents()
            for ki in T.serial(0, k_loops_a, unroll=False, dtype="uint32"):
                if STAGES == 4:
                    stage_a: T.uint32 = aslot
                    phase_a: T.uint32 = ki & T.uint32(1)
                else:
                    stage_a: T.uint32 = aslot + T.uint32(4) * (ki % T.uint32(2))
                    phase_a: T.uint32 = ki // T.uint32(2) & T.uint32(1)
                k_base_a: T.int32 = T.cast((ki * T.uint32(4) + aslot) * T.uint32(256), "int32")
                _mbarrier_wait(
                    smem_raw, T.uint32(consumed_off) + stage_a * T.uint32(8), phase_a ^ T.uint32(1)
                )
                _mbarrier_expect_tx(smem_raw, T.uint32(act_ready_off) + stage_a * T.uint32(8), 4096)
                for box in T.unroll(4):
                    _tma_2d_g2s(
                        smem_raw,
                        T.uint32(act_off)
                        + (stage_a * T.uint32(4) + T.uint32(box)) * T.uint32(1024),
                        b_tmap_act,
                        k_base_a + box * 64,
                        ni_a,
                        T.uint32(act_ready_off) + stage_a * T.uint32(8),
                    )
            if STAGES == 8:
                dki_a: T.uint32 = T.cast(k_loops_a, "uint32")
                dstage_a: T.uint32 = aslot + T.uint32(4) * (dki_a % T.uint32(2))
                dphase_a: T.uint32 = dki_a // T.uint32(2) & T.uint32(1)
                _mbarrier_wait(
                    smem_raw,
                    T.uint32(consumed_off) + dstage_a * T.uint32(8),
                    dphase_a ^ T.uint32(1),
                )
        T.ptx.bar.sync(T.uint32(2), T.uint32(THREADS))


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
    return _tinygemm2_sm100.specialize(
        STAGES=stage, USE_PDL=use_pdl, GRID_X=(O + 15) // 16, GRID_Y=(B + 7) // 8
    ).with_attr("tirx.kernel_launch_params", launch_params)


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
def _load_flashinfer_module():
    from flashinfer.jit import gen_tinygemm2_sm100_module

    spec = gen_tinygemm2_sm100_module()
    source_hash = hashlib.sha256(spec.sources[0].read_bytes()).hexdigest()
    if source_hash != SOURCE_SHA256:
        raise RuntimeError(
            "FlashInfer TinyGEMM2 source does not match the frozen oracle: "
            f"{spec.sources[0]} sha256={source_hash}"
        )
    return spec.build_and_load()


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
    _require_sm100()
    case = prepare_data(B, O, K)
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    stage = _select_stage(B, O, K, num_sms)
    linear_ref = F.linear(
        case["input"].float(), case["weight"].float(), case["bias"].float()
    ).bfloat16()

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
        torch.testing.assert_close(tirx_out.float(), linear_ref.float(), atol=1e-2, rtol=1e-2)


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
    _require_sm100()
    from tvm.tirx.bench import bench

    case = prepare_data(B, O, K)
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    stage = _select_stage(B, O, K, num_sms)
    executable = _compile_executable(B, O, stage, False)
    args = _tirx_args(case)

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


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
