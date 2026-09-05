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

import tirx_kernels.kern as K
from tirx_kernels.runner import bench

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
        from tirx_kernels.runner import hardware_num_sms

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
# Native PTX helpers (all ops expressed with K.ptx.* forms)
# ---------------------------------------------------------------------------


def _fp32_vec_to_e2m1_16(vals):
    """fp32_vec_to_e2m1 (16 elts -> uint64), native form of the source asm block.

    The dialect deliberately does not register the 4 x b8 `mov.b32` pack, so
    the byte gather is expressed as b16-pair shifts plus registered mov packs:
    `mov.b32 {w0, w1}` (2 x b16) and `mov.b64 {v0, v1}` (2 x b32).
    """
    bytes_ = K.alloc_local([8], "uint8")
    for i in range(8):
        # cvt.rn.satfinite.e2m1x2.f32 d, hi, lo (second source operand is the low lane)
        K.ptx.cvt.rn.satfinite.e2m1x2.f32(bytes_[i], vals[2 * i + 1], vals[2 * i])
    w = [
        K.cast(bytes_[i], "uint16") | (K.cast(bytes_[i + 1], "uint16") << K.uint16(8))
        for i in (0, 2, 4, 6)
    ]
    v = K.alloc_local([2], "uint32")
    K.ptx.mov.b32(v[0], w[0], w[1])
    K.ptx.mov.b32(v[1], w[2], w[3])
    out = K.local_scalar("uint64")
    K.ptx.mov.b64(out, v[0], v[1])
    return out


def _habs2(dtype):
    chain = K.ptx.abs.f16x2 if dtype == "float16" else K.ptx.abs.bf16x2

    def impl(a):
        out = K.local_scalar("uint32")
        chain(out, a)
        return out

    return impl


def _hmax2(dtype):
    chain = K.ptx.max.f16x2 if dtype == "float16" else K.ptx.max.bf16x2

    def impl(a, b):
        out = K.local_scalar("uint32")
        chain(out, a, b)
        return out

    return impl


def _hmax(dtype):
    # Scalar __hmax lowers to setp.gt.f16/bf16 + selp.b16 in the source.
    cmp_chain = K.ptx.setp.gt.f16 if dtype == "float16" else K.ptx.setp.gt.bf16

    def impl(a, b):
        pred = K.local_scalar("uint32")
        out = K.local_scalar("uint16")
        cmp_chain(pred, a, b)
        K.ptx.selp.b16(out, a, b, K.ptx.pred(pred))
        return out

    return impl


def _unpack_lo_f32(word, dtype):
    return K.cast(
        K.reinterpret(dtype, K.cast(K.bitwise_and(word, K.uint32(0xFFFF)), "uint16")), "float32"
    )


def _unpack_hi_f32(word, dtype):
    return K.cast(
        K.reinterpret(dtype, K.cast(K.shift_right(word, K.uint32(16)), "uint16")), "float32"
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

    @K.kernel(warps=(block_x + 31) // 32, arch="sm_100a", min_blocks_per_sm=4, grid=grid_x)
    def silu_and_mul_nvfp4_experts_quantize(
        input_global: K.gptr[dtype],
        sf_scale: K.gptr[K.f32],
        out_global: K.gptr[K.u64],
        sf_out: K.gptr[K.u8],
        mask: K.gptr[K.i32],
        num_rows: K.i32,
        num_cols: K.i32,
        num_experts: K.i32,
        use_silu_and_mul: K.i32,  # source ABI is bool; i32 keeps the same branch shape
    ):
        bx = K.cta_id()
        tx = K.thread_id()

        # Expert partition (quantization.cuh:642-663).
        tid32 = K.local_scalar("int32", init=bx * block_x + tx)
        stride = K.local_scalar("int32", init=K.truncdiv(grid_x * block_x, num_experts))
        part_rem = K.truncmod(grid_x * block_x, num_experts)
        expert_idx = K.local_scalar("int32")
        tid_in_expert = K.local_scalar("int32")
        actual_stride = K.local_scalar("int32")
        K.assign(expert_idx, K.int32(0))
        K.assign(tid_in_expert, K.int32(0))
        K.assign(actual_stride, stride)
        with K.If(part_rem > 0):
            with K.Then():
                bound = K.local_scalar("int32", init=part_rem * (stride + 1))
                with K.If(tid32 < bound):
                    with K.Then():
                        K.assign(expert_idx, K.truncdiv(tid32, stride + 1))
                        K.assign(tid_in_expert, K.truncmod(tid32, stride + 1))
                        K.assign(actual_stride, stride + 1)
                    with K.Else():
                        K.assign(expert_idx, part_rem + K.truncdiv(tid32 - bound, stride))
                        K.assign(tid_in_expert, K.truncmod(tid32 - bound, stride))
                        K.assign(actual_stride, stride)
            with K.Else():
                K.assign(expert_idx, K.truncdiv(tid32, stride))
                K.assign(tid_in_expert, K.truncmod(tid32, stride))
                K.assign(actual_stride, stride)

        m_rows = K.truncdiv(num_rows, num_experts)
        padded_m = (m_rows + 127) // 128 * 128
        cols_per_row = K.local_scalar("int32", init=K.truncdiv(num_cols, K.int32(ELTS_PER_THREAD)))
        use_mask = K.reinterpret("uint64", K.address_of(mask[0])) != K.uint64(0)
        actual_cols = K.local_scalar("int32", init=cols_per_row)
        with K.If(use_silu_and_mul != 0), K.Then():
            K.assign(actual_cols, cols_per_row * 2)

        xw = K.alloc_local([8], "uint32")
        yw = K.alloc_local([8], "uint32")
        packed = K.local_scalar("uint32")
        out_pair = K.alloc_local([2], "float32")
        e_tmp = K.local_scalar("float32")
        r_tmp = K.local_scalar("float32")
        lm = K.local_scalar("uint32")
        e4m3_u16 = K.local_scalar("uint16")
        f16p = K.local_scalar("uint32")
        fp = K.alloc_local([16], "float32")
        e2m1_v = K.local_scalar("uint64")
        sf_b8 = K.local_scalar("uint8")

        # Grid-stride loop over this expert's chunks (quantization.cuh:675-720).
        def body():
            global_idx = K.local_scalar("int32")
            loop_bound = K.local_scalar("int32")
            K.assign(global_idx, tid_in_expert + expert_idx * m_rows * cols_per_row)
            K.assign(loop_bound, (expert_idx + 1) * m_rows * cols_per_row)
            with K.While(global_idx < loop_bound):
                row_idx = K.local_scalar("int32")
                col_idx = K.local_scalar("int32")
                row_idx_in_expert = K.local_scalar("int32")
                K.assign(row_idx, K.truncdiv(global_idx, cols_per_row))
                K.assign(col_idx, K.truncmod(global_idx, cols_per_row))
                K.assign(row_idx_in_expert, row_idx - expert_idx * m_rows)

                with K.If(use_mask), K.Then():
                    mask_rows = K.local_scalar("int32")
                    K.ptx.ld.global_.s32(mask_rows, mask.ptr_to([expert_idx]))
                    with K.If(row_idx_in_expert >= mask_rows), K.Then():
                        K.Break()

                in_offset = K.local_scalar(
                    "int64", init=K.cast(row_idx, "int64") * actual_cols + col_idx
                )
                K.ptx.ld.global_.v4.b32(
                    xw[0],
                    xw[1],
                    xw[2],
                    xw[3],
                    K.address_of(input_global[in_offset * ELTS_PER_THREAD]),
                )
                K.ptx.ld.global_.v4.b32(
                    xw[4],
                    xw[5],
                    xw[6],
                    xw[7],
                    K.address_of(input_global[in_offset * ELTS_PER_THREAD + 8]),
                )
                with K.If(use_silu_and_mul != 0), K.Then():
                    K.ptx.ld.global_.v4.b32(
                        yw[0],
                        yw[1],
                        yw[2],
                        yw[3],
                        K.address_of(input_global[(in_offset + cols_per_row) * ELTS_PER_THREAD]),
                    )
                    K.ptx.ld.global_.v4.b32(
                        yw[4],
                        yw[5],
                        yw[6],
                        yw[7],
                        K.address_of(
                            input_global[(in_offset + cols_per_row) * ELTS_PER_THREAD + 8]
                        ),
                    )
                    # silu_and_mul (utils:1142-1166): fp32 silu*mul per element,
                    # rounded back to DTYPE pairs in place.
                    with K.unroll(8) as i:
                        x_lo = _unpack_lo_f32(xw[i], dtype)
                        x_hi = _unpack_hi_f32(xw[i], dtype)
                        y_lo = _unpack_lo_f32(yw[i], dtype)
                        y_hi = _unpack_hi_f32(yw[i], dtype)
                        K.ptx.ex2.approx.ftz.f32(e_tmp, x_lo * K.float32(-1.4426950408889634))
                        K.ptx.mov.b32(out_pair[0], (x_lo / (K.float32(1.0) + e_tmp)) * y_lo)
                        K.ptx.ex2.approx.ftz.f32(e_tmp, x_hi * K.float32(-1.4426950408889634))
                        K.ptx.mov.b32(out_pair[1], (x_hi / (K.float32(1.0) + e_tmp)) * y_hi)
                        if dtype == "float16":
                            K.ptx.cvt.rn.f16x2.f32(packed, out_pair[1], out_pair[0])
                        else:
                            K.ptx.cvt.rn.bf16x2.f32(packed, out_pair[1], out_pair[0])
                        K.ptx.mov.b32(xw[i], packed)

                out_offset = K.local_scalar(
                    "int64", init=K.cast(row_idx, "int64") * cols_per_row + col_idx
                )

                # SFScale select (branch-lowered in the source).
                sfscale_val = K.local_scalar("float32", init=K.float32(1.0))
                with (
                    K.If(K.reinterpret("uint64", K.address_of(sf_scale[0])) != K.uint64(0)),
                    K.Then(),
                ):
                    K.ptx.ld.global_.f32(sfscale_val, sf_scale.ptr_to([expert_idx]))

                # SF swizzled output address (utils:1096-1140 + quantization.cuh:706-714).
                num_cols_padded = (
                    (num_cols + SF_VEC_SIZE * 4 - 1) // (SF_VEC_SIZE * 4) * (SF_VEC_SIZE * 4)
                )
                num_cols_sfout = num_cols_padded // SF_VEC_SIZE // 4
                sf_expert_base = K.local_scalar(
                    "int32", init=expert_idx * padded_m * num_cols_sfout
                )
                num_k_tiles = (num_cols + SF_VEC_SIZE * 4 - 1) // (SF_VEC_SIZE * 4)
                sf_off = K.local_scalar(
                    "int32",
                    init=K.truncdiv(row_idx_in_expert, K.int32(128)) * (num_k_tiles * 512)
                    + K.truncdiv(col_idx, K.int32(4)) * 512
                    + (row_idx_in_expert % 32) * 16
                    + K.truncdiv(row_idx_in_expert % 128, K.int32(32)) * 4
                    + (col_idx % 4),
                )
                sf_byte = K.cast(sf_expert_base, "int64") * 4 + K.cast(sf_off, "int64")

                # Local abs-max over the 8 packed pairs (silu-rounded values).
                K.assign(lm, habs2(xw[0]))
                with K.unroll(7) as i:
                    K.assign(lm, hmax2(lm, habs2(xw[i + 1])))
                lm_lo = K.cast(K.bitwise_and(lm, K.uint32(0xFFFF)), "uint16")
                lm_hi = K.cast(K.shift_right(lm, K.uint32(16)), "uint16")
                vec_max = K.cast(K.reinterpret(dtype, hmax(lm_lo, lm_hi)), "float32")

                # SF computation (default env: fast-math rcp, E4M3).
                K.ptx.rcp.approx.ftz.f32(r_tmp, K.float32(6.0))
                sf_value = sfscale_val * (vec_max * r_tmp)
                K.ptx.cvt.rn.satfinite.e4m3x2.f32(e4m3_u16, K.float32(0.0), sf_value)
                K.assign(sf_b8, K.cast(e4m3_u16, "uint8"))
                K.ptx.cvt.rn.f16x2.e4m3x2(f16p, e4m3_u16)
                sf_value_r = _unpack_lo_f32(f16p, "float16")
                output_scale = K.local_scalar("float32", init=K.float32(0.0))
                with K.If(vec_max != 0.0), K.Then():
                    K.ptx.rcp.approx.ftz.f32(r_tmp, sfscale_val)
                    K.ptx.rcp.approx.ftz.f32(e_tmp, sf_value_r * r_tmp)
                    K.assign(output_scale, e_tmp)

                # SF byte store (STG.8, per thread).
                with (
                    K.If(K.reinterpret("uint64", K.address_of(sf_out[0])) != K.uint64(0)),
                    K.Then(),
                ):
                    K.ptx.st.global_.b8(K.address_of(sf_out[sf_byte]), sf_b8)

                # Scale to e2m1 and pack (fp32_vec_to_e2m1 source asm block).
                with K.unroll(8) as i:
                    K.ptx.mov.b32(fp[2 * i], _unpack_lo_f32(xw[i], dtype) * output_scale)
                    K.ptx.mov.b32(fp[2 * i + 1], _unpack_hi_f32(xw[i], dtype) * output_scale)
                K.assign(e2m1_v, _fp32_vec_to_e2m1_16([fp[i] for i in range(16)]))
                K.ptx.st.global_.b64(K.address_of(out_global[out_offset]), e2m1_v)

                K.assign(global_idx, global_idx + actual_stride)

        if block_x % 32:
            with K.If(tx < block_x), K.Then():
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
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(dtype: str, n_experts: int, m: int, k: int, mask_mode: str = "rand", **kwargs):
    """Compile, launch, and validate one config against the flashinfer source."""
    import torch

    from tirx_kernels.runner import compile_kernel

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
