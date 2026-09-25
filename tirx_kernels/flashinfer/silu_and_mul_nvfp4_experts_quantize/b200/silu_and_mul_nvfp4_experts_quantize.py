# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400),
# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer ``cvt_fp16_to_fp4_expert`` port.

Ports ``tensorrt_llm::kernels::cvt_fp16_to_fp4_expert<T, UE8M0_SF=false,
DISABLE_FP4_QUANT_FAST_MATH=false, NVFP4_4OVER6_CONFIG=std::false_type>``
(``csrc/nv_internal/tensorrt_llm/kernels/quantization.cuh``), the SM100 kernel
behind ``flashinfer.activation.silu_and_mul_scaled_nvfp4_experts_quantize``.
The kernel fuses SiLU*mul gating with per-16-element NVFP4 quantization and a
swizzled 6D scale-factor layout, with an expert-partitioned grid-stride loop
and per-expert row masks.  Only the default-environment specialization is in
scope (fast-math reciprocal, E4M3 scale factors, no 4over6 refinement).
"""

from typing import Any

import tirx_kernels.tirx_lite as txl
from tirx_kernels.bench.runner import bench

KERNEL_META = {
    "name": "silu_and_mul_nvfp4_experts_quantize",
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

_DTYPES = ("float16", "bfloat16")
_MASK_MODES = ("rand", "full")
# Source constants (quantization.cuh): SF block = 16 elements; device kernel
# converts 16 elements (32 bytes) per thread under CUDA >= 12.9 + sm_100a.
SF_VEC_SIZE = 16
ELTS_PER_THREAD = 16
# Host launch sizing always sees ELTS_PER_THREAD == 8 (quantization.cu:729
# compiles with __CUDA_ARCH__ undefined).
HOST_ELTS_PER_THREAD = 8

_SM_COUNT_CACHE = None


def _sm_count() -> int:
    global _SM_COUNT_CACHE
    if _SM_COUNT_CACHE is None:
        from tirx_kernels.bench.runner import hardware_num_sms

        _SM_COUNT_CACHE = hardware_num_sms()
    return _SM_COUNT_CACHE


def _torch_dtype(dtype: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


def _padded_m(m: int) -> int:
    return (m + 127) // 128 * 128


def _padded_k_sf(k: int) -> int:
    """SF columns after swizzle padding (round_up(k / 16, 4))."""
    return (k // SF_VEC_SIZE + 3) // 4 * 4


def _launch_shape(n_experts: int, m: int, k: int) -> tuple[int, int]:
    """Mirror the source host grid/block computation (quantization.cu:729-745)."""
    m_topk = n_experts * m
    work_size_per_row = max(1, k // HOST_ELTS_PER_THREAD)
    total_work_size = m_topk * work_size_per_row
    block = min(work_size_per_row, 512)
    num_blocks_per_sm = 2048 // block
    grid = min((total_work_size + block - 1) // block, _sm_count() * num_blocks_per_sm)
    while grid <= _sm_count() and block > 64:
        grid *= 2
        block = (block + 1) // 2
    grid = (grid + n_experts - 1) // n_experts * n_experts
    return grid, block


def _validate(dtype: str, n_experts: int, m: int, k: int) -> None:
    if dtype not in _DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if n_experts < 1:
        raise ValueError(f"n_experts={n_experts} must be >= 1")
    if m < 1:
        raise ValueError(f"m={m} must be >= 1")
    if k <= 0 or k % SF_VEC_SIZE != 0:
        raise ValueError(f"k={k} outside the source dispatch domain (k % 16 != 0)")


# ---------------------------------------------------------------------------
# Native PTX helpers (all ops expressed with txl.ptx.* forms)
# ---------------------------------------------------------------------------


def _fp32_vec_to_e2m1_16(vals):
    """fp32_vec_to_e2m1 (16 elts -> uint64), native form of the source asm block.

    The dialect deliberately does not register the 4 x b8 `mov.b32` pack, so
    the byte gather is expressed as b16-pair shifts plus registered mov packs:
    `mov.b32 {w0, w1}` (2 x b16) and `mov.b64 {v0, v1}` (2 x b32).
    """
    bytes_ = txl.alloc_local([8], "uint8")
    for i in range(8):
        # cvt.rn.satfinite.e2m1x2.f32 d, hi, lo (second source operand is the low lane)
        txl.ptx.cvt.rn.satfinite.e2m1x2.f32(bytes_[i], vals[2 * i + 1], vals[2 * i])
    w = [
        txl.cast(bytes_[i], "uint16") | (txl.cast(bytes_[i + 1], "uint16") << txl.uint16(8))
        for i in (0, 2, 4, 6)
    ]
    v = txl.alloc_local([2], "uint32")
    txl.ptx.mov.b32(v[0], w[0], w[1])
    txl.ptx.mov.b32(v[1], w[2], w[3])
    out = txl.local_scalar("uint64")
    txl.ptx.mov.b64(out, v[0], v[1])
    return out


def _habs2(dtype):
    chain = txl.ptx.abs.f16x2 if dtype == "float16" else txl.ptx.abs.bf16x2

    def impl(a):
        out = txl.local_scalar("uint32")
        chain(out, a)
        return out

    return impl


def _hmax2(dtype):
    chain = txl.ptx.max.f16x2 if dtype == "float16" else txl.ptx.max.bf16x2

    def impl(a, b):
        out = txl.local_scalar("uint32")
        chain(out, a, b)
        return out

    return impl


def _hmax(dtype):
    # Scalar __hmax lowers to setp.gt.f16/bf16 + selp.b16 in the source.
    cmp_chain = txl.ptx.setp.gt.f16 if dtype == "float16" else txl.ptx.setp.gt.bf16

    def impl(a, b):
        pred = txl.local_scalar("uint32")
        out = txl.local_scalar("uint16")
        cmp_chain(pred, a, b)
        txl.ptx.selp.b16(out, a, b, txl.ptx.pred(pred))
        return out

    return impl


def _unpack_lo_f32(word, dtype):
    return txl.cast(
        txl.reinterpret(dtype, txl.cast(txl.bitwise_and(word, txl.uint32(0xFFFF)), "uint16")), "float32"
    )


def _unpack_hi_f32(word, dtype):
    return txl.cast(
        txl.reinterpret(dtype, txl.cast(txl.shift_right(word, txl.uint32(16)), "uint16")), "float32"
    )


def get_kernel(dtype: str, n_experts: int, m: int, k: int, mask_mode: str = "rand", **kwargs):
    """Return the TIRx specialization for one (dtype, n_experts, m, k) config."""
    _validate(dtype, n_experts, m, k)
    if mask_mode not in _MASK_MODES:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")
    grid_x, block_x = _launch_shape(n_experts, m, k)
    habs2 = _habs2(dtype)
    hmax2 = _hmax2(dtype)
    hmax = _hmax(dtype)

    @txl.kernel(warps=(block_x + 31) // 32, arch="sm_100a", min_blocks_per_sm=4, grid=grid_x)
    def silu_and_mul_nvfp4_experts_quantize(
        input_global: txl.gptr[dtype],
        sf_scale: txl.gptr[txl.f32],
        out_global: txl.gptr[txl.u64],
        sf_out: txl.gptr[txl.u8],
        mask: txl.gptr[txl.i32],
        num_rows: txl.i32,
        num_cols: txl.i32,
        num_experts: txl.i32,
        use_silu_and_mul: txl.i32,  # source ABI is bool; i32 keeps the same branch shape
    ):
        bx = txl.cta_id()
        tx = txl.thread_id()

        # Expert partition (quantization.cuh:642-663).
        tid32 = txl.local_scalar("int32", init=bx * block_x + tx)
        stride = txl.local_scalar("int32", init=txl.truncdiv(grid_x * block_x, num_experts))
        part_rem = txl.truncmod(grid_x * block_x, num_experts)
        expert_idx = txl.local_scalar("int32")
        tid_in_expert = txl.local_scalar("int32")
        actual_stride = txl.local_scalar("int32")
        txl.assign(expert_idx, txl.int32(0))
        txl.assign(tid_in_expert, txl.int32(0))
        txl.assign(actual_stride, stride)
        with txl.If(part_rem > 0):
            with txl.Then():
                bound = txl.local_scalar("int32", init=part_rem * (stride + 1))
                with txl.If(tid32 < bound):
                    with txl.Then():
                        txl.assign(expert_idx, txl.truncdiv(tid32, stride + 1))
                        txl.assign(tid_in_expert, txl.truncmod(tid32, stride + 1))
                        txl.assign(actual_stride, stride + 1)
                    with txl.Else():
                        txl.assign(expert_idx, part_rem + txl.truncdiv(tid32 - bound, stride))
                        txl.assign(tid_in_expert, txl.truncmod(tid32 - bound, stride))
                        txl.assign(actual_stride, stride)
            with txl.Else():
                txl.assign(expert_idx, txl.truncdiv(tid32, stride))
                txl.assign(tid_in_expert, txl.truncmod(tid32, stride))
                txl.assign(actual_stride, stride)

        m_rows = txl.truncdiv(num_rows, num_experts)
        padded_m = (m_rows + 127) // 128 * 128
        cols_per_row = txl.local_scalar("int32", init=txl.truncdiv(num_cols, txl.int32(ELTS_PER_THREAD)))
        use_mask = txl.reinterpret("uint64", txl.address_of(mask[0])) != txl.uint64(0)
        actual_cols = txl.local_scalar("int32", init=cols_per_row)
        with txl.If(use_silu_and_mul != 0), txl.Then():
            txl.assign(actual_cols, cols_per_row * 2)

        xw = txl.alloc_local([8], "uint32")
        yw = txl.alloc_local([8], "uint32")
        packed = txl.local_scalar("uint32")
        out_pair = txl.alloc_local([2], "float32")
        e_tmp = txl.local_scalar("float32")
        r_tmp = txl.local_scalar("float32")
        lm = txl.local_scalar("uint32")
        e4m3_u16 = txl.local_scalar("uint16")
        f16p = txl.local_scalar("uint32")
        fp = txl.alloc_local([16], "float32")
        e2m1_v = txl.local_scalar("uint64")
        sf_b8 = txl.local_scalar("uint8")

        # Grid-stride loop over this expert's chunks (quantization.cuh:675-720).
        def body():
            global_idx = txl.local_scalar("int32")
            loop_bound = txl.local_scalar("int32")
            txl.assign(global_idx, tid_in_expert + expert_idx * m_rows * cols_per_row)
            txl.assign(loop_bound, (expert_idx + 1) * m_rows * cols_per_row)
            with txl.While(global_idx < loop_bound):
                row_idx = txl.local_scalar("int32")
                col_idx = txl.local_scalar("int32")
                row_idx_in_expert = txl.local_scalar("int32")
                txl.assign(row_idx, txl.truncdiv(global_idx, cols_per_row))
                txl.assign(col_idx, txl.truncmod(global_idx, cols_per_row))
                txl.assign(row_idx_in_expert, row_idx - expert_idx * m_rows)

                with txl.If(use_mask), txl.Then():
                    mask_rows = txl.local_scalar("int32")
                    txl.ptx.ld.global_.s32(mask_rows, mask.ptr_to([expert_idx]))
                    with txl.If(row_idx_in_expert >= mask_rows), txl.Then():
                        txl.Break()

                in_offset = txl.local_scalar(
                    "int64", init=txl.cast(row_idx, "int64") * actual_cols + col_idx
                )
                txl.ptx.ld.global_.v4.b32(
                    xw[0],
                    xw[1],
                    xw[2],
                    xw[3],
                    txl.address_of(input_global[in_offset * ELTS_PER_THREAD]),
                )
                txl.ptx.ld.global_.v4.b32(
                    xw[4],
                    xw[5],
                    xw[6],
                    xw[7],
                    txl.address_of(input_global[in_offset * ELTS_PER_THREAD + 8]),
                )
                with txl.If(use_silu_and_mul != 0), txl.Then():
                    txl.ptx.ld.global_.v4.b32(
                        yw[0],
                        yw[1],
                        yw[2],
                        yw[3],
                        txl.address_of(input_global[(in_offset + cols_per_row) * ELTS_PER_THREAD]),
                    )
                    txl.ptx.ld.global_.v4.b32(
                        yw[4],
                        yw[5],
                        yw[6],
                        yw[7],
                        txl.address_of(
                            input_global[(in_offset + cols_per_row) * ELTS_PER_THREAD + 8]
                        ),
                    )
                    # silu_and_mul (utils:1142-1166): fp32 silu*mul per element,
                    # rounded back to DTYPE pairs in place.
                    with txl.unroll(8) as i:
                        x_lo = _unpack_lo_f32(xw[i], dtype)
                        x_hi = _unpack_hi_f32(xw[i], dtype)
                        y_lo = _unpack_lo_f32(yw[i], dtype)
                        y_hi = _unpack_hi_f32(yw[i], dtype)
                        txl.ptx.ex2.approx.ftz.f32(e_tmp, x_lo * txl.float32(-1.4426950408889634))
                        txl.ptx.mov.b32(out_pair[0], (x_lo / (txl.float32(1.0) + e_tmp)) * y_lo)
                        txl.ptx.ex2.approx.ftz.f32(e_tmp, x_hi * txl.float32(-1.4426950408889634))
                        txl.ptx.mov.b32(out_pair[1], (x_hi / (txl.float32(1.0) + e_tmp)) * y_hi)
                        if dtype == "float16":
                            txl.ptx.cvt.rn.f16x2.f32(packed, out_pair[1], out_pair[0])
                        else:
                            txl.ptx.cvt.rn.bf16x2.f32(packed, out_pair[1], out_pair[0])
                        txl.ptx.mov.b32(xw[i], packed)

                out_offset = txl.local_scalar(
                    "int64", init=txl.cast(row_idx, "int64") * cols_per_row + col_idx
                )

                # SFScale select (branch-lowered in the source).
                sfscale_val = txl.local_scalar("float32", init=txl.float32(1.0))
                with (
                    txl.If(txl.reinterpret("uint64", txl.address_of(sf_scale[0])) != txl.uint64(0)),
                    txl.Then(),
                ):
                    txl.ptx.ld.global_.f32(sfscale_val, sf_scale.ptr_to([expert_idx]))

                # SF swizzled output address (utils:1096-1140 + quantization.cuh:706-714).
                num_cols_padded = (
                    (num_cols + SF_VEC_SIZE * 4 - 1) // (SF_VEC_SIZE * 4) * (SF_VEC_SIZE * 4)
                )
                num_cols_sfout = num_cols_padded // SF_VEC_SIZE // 4
                sf_expert_base = txl.local_scalar(
                    "int32", init=expert_idx * padded_m * num_cols_sfout
                )
                num_k_tiles = (num_cols + SF_VEC_SIZE * 4 - 1) // (SF_VEC_SIZE * 4)
                sf_off = txl.local_scalar(
                    "int32",
                    init=txl.truncdiv(row_idx_in_expert, txl.int32(128)) * (num_k_tiles * 512)
                    + txl.truncdiv(col_idx, txl.int32(4)) * 512
                    + (row_idx_in_expert % 32) * 16
                    + txl.truncdiv(row_idx_in_expert % 128, txl.int32(32)) * 4
                    + (col_idx % 4),
                )
                sf_byte = txl.cast(sf_expert_base, "int64") * 4 + txl.cast(sf_off, "int64")

                # Local abs-max over the 8 packed pairs (silu-rounded values).
                txl.assign(lm, habs2(xw[0]))
                with txl.unroll(7) as i:
                    txl.assign(lm, hmax2(lm, habs2(xw[i + 1])))
                lm_lo = txl.cast(txl.bitwise_and(lm, txl.uint32(0xFFFF)), "uint16")
                lm_hi = txl.cast(txl.shift_right(lm, txl.uint32(16)), "uint16")
                vec_max = txl.cast(txl.reinterpret(dtype, hmax(lm_lo, lm_hi)), "float32")

                # SF computation (default env: fast-math rcp, E4M3).
                txl.ptx.rcp.approx.ftz.f32(r_tmp, txl.float32(6.0))
                sf_value = sfscale_val * (vec_max * r_tmp)
                txl.ptx.cvt.rn.satfinite.e4m3x2.f32(e4m3_u16, txl.float32(0.0), sf_value)
                txl.assign(sf_b8, txl.cast(e4m3_u16, "uint8"))
                txl.ptx.cvt.rn.f16x2.e4m3x2(f16p, e4m3_u16)
                sf_value_r = _unpack_lo_f32(f16p, "float16")
                output_scale = txl.local_scalar("float32", init=txl.float32(0.0))
                with txl.If(vec_max != 0.0), txl.Then():
                    txl.ptx.rcp.approx.ftz.f32(r_tmp, sfscale_val)
                    txl.ptx.rcp.approx.ftz.f32(e_tmp, sf_value_r * r_tmp)
                    txl.assign(output_scale, e_tmp)

                # SF byte store (STG.8, per thread).
                with (
                    txl.If(txl.reinterpret("uint64", txl.address_of(sf_out[0])) != txl.uint64(0)),
                    txl.Then(),
                ):
                    txl.ptx.st.global_.b8(txl.address_of(sf_out[sf_byte]), sf_b8)

                # Scale to e2m1 and pack (fp32_vec_to_e2m1 source asm block).
                with txl.unroll(8) as i:
                    txl.ptx.mov.b32(fp[2 * i], _unpack_lo_f32(xw[i], dtype) * output_scale)
                    txl.ptx.mov.b32(fp[2 * i + 1], _unpack_hi_f32(xw[i], dtype) * output_scale)
                txl.assign(e2m1_v, _fp32_vec_to_e2m1_16([fp[i] for i in range(16)]))
                txl.ptx.st.global_.b64(txl.address_of(out_global[out_offset]), e2m1_v)

                txl.assign(global_idx, global_idx + actual_stride)

        if block_x % 32:
            with txl.If(tx < block_x), txl.Then():
                body()
        else:
            body()

    return silu_and_mul_nvfp4_experts_quantize.func


def prepare_data(dtype: str, n_experts: int, m: int, k: int, mask_mode: str = "rand", **kwargs):
    """Create logical inputs: a [B, M, 2K], mask [B] int32, global_scale [B] fp32."""
    import torch

    _validate(dtype, n_experts, m, k)
    if mask_mode not in _MASK_MODES:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")
    torch.manual_seed(42)
    a = torch.randn(n_experts, m, 2 * k, dtype=_torch_dtype(dtype), device="cuda")
    if mask_mode == "full":
        mask = torch.full((n_experts,), m, dtype=torch.int32, device="cuda")
    else:
        mask = torch.randint(1, m + 1, (n_experts,), dtype=torch.int32, device="cuda")
    global_scale = torch.rand(n_experts, dtype=torch.float32, device="cuda") * 1.0 + 0.5
    return (a, mask, global_scale)


def _alloc_outputs(dtype: str, n_experts: int, m: int, k: int):
    import torch

    pm = _padded_m(m)
    pk_sf = _padded_k_sf(k)
    # Physical kernel-output layouts (thop fp4Quantize.cpp:242-248).
    out = torch.empty(n_experts, m, k // 2, dtype=torch.uint8, device="cuda")
    sf = torch.empty(n_experts, pm, pk_sf // 4, dtype=torch.int32, device="cuda")
    return out, sf


def _sf_valid_byte_mask(n_experts: int, m: int, k: int, mask) -> "object":
    """Boolean [B, pm*pk_sf] byte mask of SF slots the kernel writes (valid rows).

    Reproduces cvt_quant_to_fp4_get_sf_out_offset: bytes for row < mask[e] and
    kIdx < k/16 inside expert e's [pm, pk_sf] region.
    """
    import torch

    pm = _padded_m(m)
    pk_sf = _padded_k_sf(k)
    cols_per_row = k // SF_VEC_SIZE
    num_k_tiles = pk_sf // 4
    valid = torch.zeros(n_experts, pm, pk_sf, dtype=torch.bool, device=mask.device)
    for e in range(n_experts):
        rows = torch.arange(int(mask[e].item()), device=mask.device)
        kidx = torch.arange(cols_per_row, device=mask.device)
        rr, kk = torch.meshgrid(rows, kidx, indexing="ij")
        m_tile = rr // 128
        outer_m = rr % 32
        inner_m = (rr % 128) // 32
        k_tile = kk // 4
        inner_k = kk % 4
        off = (
            m_tile * (num_k_tiles * 128 * 4)
            + k_tile * (128 * 4)
            + outer_m * 16
            + inner_m * 4
            + inner_k
        )
        valid[e].view(-1)[off.view(-1)] = True
    return valid.view(n_experts, -1)


def _run_launch(ex, a, global_scale, out, sf, mask, n_experts, m, k):
    """Launch the TIRx kernel with the source ABI (5 tensors + 4 scalars)."""
    import torch

    ex(
        a.view(-1),
        global_scale,
        out.view(-1).view(torch.uint64),
        sf.view(-1).view(torch.uint8),
        mask,
        n_experts * m,
        k,
        n_experts,
        1,
    )


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.bench.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(dtype: str, n_experts: int, m: int, k: int, mask_mode: str = "rand", **kwargs):
    """Compile, launch, and validate one config against the flashinfer source."""
    import torch

    from tirx_kernels.bench.runner import compile_kernel

    a, mask, global_scale = prepare_data(
        dtype=dtype, n_experts=n_experts, m=m, k=k, mask_mode=mask_mode
    )
    kernel = get_kernel(dtype=dtype, n_experts=n_experts, m=m, k=k, mask_mode=mask_mode)
    ex = compile_kernel(kernel)
    out_tirx, sf_tirx = _alloc_outputs(dtype, n_experts, m, k)
    _run_launch(ex, a, global_scale, out_tirx, sf_tirx, mask, n_experts, m, k)
    torch.cuda.synchronize()

    import flashinfer

    # Source API allocates its own outputs and returns permuted logical views.
    ref_q, ref_sf = flashinfer.activation.silu_and_mul_scaled_nvfp4_experts_quantize(
        a, mask, global_scale
    )
    # ref_q logical [M, K/2, B] -> physical [B, M, K/2] uint8.
    ref_q = ref_q.permute(2, 0, 1)
    # ref_sf logical [32, 4, pm/128, 4, pk/64, B] -> physical (B, pm/128, pk/4, 32, 4, 4).
    ref_sf_u8 = ref_sf.permute(5, 2, 4, 0, 1, 3).contiguous().view(torch.uint8).view(n_experts, -1)

    for e in range(n_experts):
        rows = int(mask[e].item())
        torch.testing.assert_close(out_tirx[e, :rows], ref_q[e, :rows], rtol=0, atol=0)
    valid = _sf_valid_byte_mask(n_experts, m, k, mask)
    sf_tirx_u8 = sf_tirx.view(n_experts, -1).view(torch.uint8)
    torch.testing.assert_close(sf_tirx_u8[valid], ref_sf_u8[valid], rtol=0, atol=0)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs):
    """Benchmark the TIRx port against the source thop (kernel-only)."""
    config = dict(prepared["config"])
    dtype = config.pop("dtype")
    n_experts = config.pop("n_experts")
    m = config.pop("m")
    k = config.pop("k")
    mask_mode = config.pop("mask_mode")
    config.update(kwargs)
    kwargs = config
    executable = prepared["executable"]
    import torch

    a, mask, global_scale = prepare_data(
        dtype=dtype, n_experts=n_experts, m=m, k=k, mask_mode=mask_mode
    )
    ex = executable
    out_tirx, sf_tirx = _alloc_outputs(dtype, n_experts, m, k)

    funcs = {
        "tirx": lambda: _run_launch(ex, a, global_scale, out_tirx, sf_tirx, mask, n_experts, m, k)
    }

    def build_reference():
        from flashinfer.jit.fp4_quantization import gen_fp4_quantization_sm100_module

        mod = gen_fp4_quantization_sm100_module().build_and_load()
        out_ref = torch.empty(n_experts * m, k // 2, dtype=torch.uint8, device="cuda")
        pm = _padded_m(m)
        pk_sf = _padded_k_sf(k)
        sf_ref = torch.empty(n_experts * pm, pk_sf // 4, dtype=torch.int32, device="cuda")
        in_2d = a.view(n_experts * m, 2 * k)
        thop = mod.silu_and_mul_scaled_nvfp4_experts_quantize
        return lambda: thop(out_ref, sf_ref, in_2d, global_scale, mask, True)

    return bench(
        funcs,
        references={"flashinfer": build_reference},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    dtype: str,
    n_experts: int,
    m: int,
    k: int,
    mask_mode: str = "rand",
    *,
    warmup=None,
    repeat=None,
    timer=None,
    rounds=1,
    cooldown_s=1.0,
    **kwargs,
):
    config = dict(kwargs)
    prepared = prepare_bench(
        dtype=dtype, n_experts=n_experts, m=m, k=k, mask_mode=mask_mode, **config
    )
    return prepared.run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


def _cfg(dtype, n_experts, m, k, mask_mode="rand"):
    dt = {"float16": "fp16", "bfloat16": "bf16"}[dtype]
    suffix = "" if mask_mode == "rand" else f"_{mask_mode}"
    return {
        "label": f"{dt}_b{n_experts}_m{m}_k{k}{suffix}",
        "dtype": dtype,
        "n_experts": n_experts,
        "m": m,
        "k": k,
        "mask_mode": mask_mode,
    }


# Correctness matrix.  Covers: both dtypes on the source test shapes
# (tests/utils/test_fp4_quantize.py: (1,256,128), (2,128,64), (3,256,128),
# (1,120,64), (128,2048,2048)); the m % 128 != 0 SF row-padding path; the
# padded_k SF column-padding path (k/16 not a multiple of 4); mask edge modes
# (rand partial rows, full rows); multi-mTile m.
CONFIGS = [
    _cfg("float16", 1, 256, 128),
    _cfg("bfloat16", 1, 256, 128),
    _cfg("float16", 2, 128, 64),
    _cfg("bfloat16", 2, 128, 64),
    _cfg("float16", 3, 256, 128),
    _cfg("bfloat16", 3, 256, 128),
    _cfg("float16", 1, 120, 64),
    _cfg("bfloat16", 1, 120, 64),
    _cfg("float16", 2, 128, 64, "full"),
    _cfg("float16", 2, 64, 16),  # padded_k: k/16 = 1 -> 4
    _cfg("bfloat16", 2, 64, 48),  # padded_k: k/16 = 3 -> 4
    _cfg("float16", 4, 384, 1024),  # multi-mTile rows
    _cfg("float16", 128, 2048, 2048),  # largest source test shape
]

# Benchmark sweep: source's largest test shape plus realistic MoE sizes.
BENCH_CONFIGS = [
    _cfg("float16", 128, 2048, 2048),
    _cfg("bfloat16", 128, 2048, 2048),
    _cfg("float16", 8, 512, 2048),
    _cfg("bfloat16", 8, 512, 2048),
    _cfg("float16", 4, 128, 4096),
    _cfg("float16", 8, 16, 2048),  # decode-scale rows
]
