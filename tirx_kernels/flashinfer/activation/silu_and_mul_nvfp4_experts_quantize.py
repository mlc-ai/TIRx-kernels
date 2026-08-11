# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications Copyright (c) 2026 The TIRx Authors.
# Modifications are licensed under the Apache License, Version 2.0.
#
# This file is a TIRx port of cvt_fp16_to_fp4_expert in NVIDIA TensorRT-LLM's
# quantization.cuh, as vendored by FlashInfer at
# csrc/nv_internal/tensorrt_llm/kernels/quantization.cuh
# (flashinfer-ai/flashinfer @ f2e04400, v0.6.18).
# See LICENSE, NOTICE, and licenses/ for the applicable terms.

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

from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T
from tvm.tirx.bench import bench

KERNEL_META = {
    "name": "silu_and_mul_nvfp4_experts_quantize",
    "category": "flashinfer",
    "compute_capability": 10,
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
        import torch

        _SM_COUNT_CACHE = torch.cuda.get_device_properties(0).multi_processor_count
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
# Native PTX helpers (all ops expressed with T.ptx.* forms)
# ---------------------------------------------------------------------------


def _fp32_vec_to_e2m1_16(vals):
    """fp32_vec_to_e2m1 (16 elts -> uint64), native form of the source asm block.

    The dialect deliberately does not register the 4 x b8 `mov.b32` pack, so
    the byte gather is expressed as b16-pair shifts plus registered mov packs:
    `mov.b32 {w0, w1}` (2 x b16) and `mov.b64 {v0, v1}` (2 x b32).
    """
    bytes_ = T.alloc_local([8], "uint8")
    for i in range(8):
        # cvt.rn.satfinite.e2m1x2.f32 d, hi, lo (second source operand is the low lane)
        T.evaluate(T.ptx.cvt.rn.satfinite.e2m1x2.f32(bytes_[i], vals[2 * i + 1], vals[2 * i]))
    w = [
        T.cast(bytes_[i], "uint16") | (T.cast(bytes_[i + 1], "uint16") << T.uint16(8))
        for i in (0, 2, 4, 6)
    ]
    v = T.alloc_local([2], "uint32")
    T.evaluate(T.ptx.mov.b32(v[0], w[0], w[1]))
    T.evaluate(T.ptx.mov.b32(v[1], w[2], w[3]))
    out = T.alloc_local([1], "uint64")
    T.evaluate(T.ptx.mov.b64(out[0], v[0], v[1]))
    return out[0]


def _habs2(dtype):
    chain = T.ptx.abs.f16x2 if dtype == "float16" else T.ptx.abs.bf16x2

    def impl(a):
        out = T.alloc_local([1], "uint32")
        T.evaluate(chain(out[0], a))
        return out[0]

    return impl


def _hmax2(dtype):
    chain = T.ptx.max.f16x2 if dtype == "float16" else T.ptx.max.bf16x2

    def impl(a, b):
        out = T.alloc_local([1], "uint32")
        T.evaluate(chain(out[0], a, b))
        return out[0]

    return impl


def _hmax(dtype):
    # Scalar __hmax lowers to setp.gt.f16/bf16 + selp.b16 in the source.
    cmp_chain = T.ptx.setp.gt.f16 if dtype == "float16" else T.ptx.setp.gt.bf16

    def impl(a, b):
        pred = T.local_scalar("uint32")
        out = T.alloc_local([1], "uint16")
        T.evaluate(cmp_chain(pred, a, b))
        T.evaluate(T.ptx.selp.b16(out[0], a, b, T.ptx.pred(pred)))
        return out[0]

    return impl


def _unpack_lo_f32(word, dtype):
    return T.cast(
        T.reinterpret(dtype, T.cast(T.bitwise_and(word, T.uint32(0xFFFF)), "uint16")), "float32"
    )


def _unpack_hi_f32(word, dtype):
    return T.cast(
        T.reinterpret(dtype, T.cast(T.shift_right(word, T.uint32(16)), "uint16")), "float32"
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

    @T.prim_func
    def silu_and_mul_nvfp4_experts_quantize(
        in_ptr: T.handle,
        sf_scale_ptr: T.handle,
        out_ptr: T.handle,
        sf_out_ptr: T.handle,
        mask_ptr: T.handle,
        num_rows: T.int32,
        num_cols: T.int32,
        num_experts: T.int32,
        use_silu_and_mul: T.int32,  # source ABI is bool; i32 keeps the same branch shape
    ):
        input_global = T.match_buffer(
            in_ptr, shape=[num_rows * 2 * num_cols], dtype=dtype, scope="global"
        )
        sf_scale = T.match_buffer(
            sf_scale_ptr, shape=[num_experts], dtype="float32", scope="global"
        )
        out_global = T.match_buffer(
            out_ptr, shape=[num_rows * (num_cols // 16)], dtype="uint64", scope="global"
        )
        sf_out = T.match_buffer(
            sf_out_ptr,
            shape=[
                num_experts
                * ((num_rows // num_experts + 127) // 128 * 128)
                * ((num_cols // 16 + 3) // 4 * 4)
            ],
            dtype="uint8",
            scope="global",
        )
        mask = T.match_buffer(mask_ptr, shape=[num_experts], dtype="int32", scope="global")
        T.device_entry()
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 4})
        bx = T.cta_id([grid_x])
        tx = T.thread_id([block_x])

        # Expert partition (quantization.cuh:642-663).
        tid32 = bx * block_x + tx
        stride = T.truncdiv(grid_x * block_x, num_experts)
        part_rem = T.truncmod(grid_x * block_x, num_experts)
        expert_idx: T.int32 = T.int32(0)
        tid_in_expert: T.int32 = T.int32(0)
        actual_stride: T.int32 = stride
        if part_rem > 0:
            bound = part_rem * (stride + 1)
            if tid32 < bound:
                expert_idx = T.truncdiv(tid32, stride + 1)
                tid_in_expert = T.truncmod(tid32, stride + 1)
                actual_stride = stride + 1
            else:
                expert_idx = part_rem + T.truncdiv(tid32 - bound, stride)
                tid_in_expert = T.truncmod(tid32 - bound, stride)
                actual_stride = stride
        else:
            expert_idx = T.truncdiv(tid32, stride)
            tid_in_expert = T.truncmod(tid32, stride)
            actual_stride = stride

        m_rows = T.truncdiv(num_rows, num_experts)
        padded_m = (m_rows + 127) // 128 * 128
        cols_per_row = T.truncdiv(num_cols, T.int32(ELTS_PER_THREAD))
        use_mask = T.reinterpret("uint64", T.address_of(mask[0])) != T.uint64(0)
        actual_cols: T.int32 = cols_per_row
        if use_silu_and_mul != 0:
            actual_cols = cols_per_row * 2

        x64 = T.alloc_local([4], "uint64")
        y64 = T.alloc_local([4], "uint64")
        xw = T.decl_buffer(
            [8],
            "uint32",
            data=T.reinterpret(PointerType(PrimType("uint32")), T.address_of(x64[0])),
            scope="local",
        )
        yw = T.decl_buffer(
            [8],
            "uint32",
            data=T.reinterpret(PointerType(PrimType("uint32")), T.address_of(y64[0])),
            scope="local",
        )
        packed = T.alloc_local([1], "uint32")
        e_tmp = T.alloc_local([1], "float32")
        r_tmp = T.alloc_local([1], "float32")
        lm = T.alloc_local([1], "uint32")
        e4m3_u16 = T.alloc_local([1], "uint16")
        f16p = T.alloc_local([1], "uint32")
        fp = T.alloc_local([16], "float32")
        e2m1_v = T.alloc_local([1], "uint64")
        sf_b8 = T.alloc_local([1], "uint8")

        # Grid-stride loop over this expert's chunks (quantization.cuh:675-720).
        global_idx: T.int32 = tid_in_expert + expert_idx * m_rows * cols_per_row
        loop_bound: T.int32 = (expert_idx + 1) * m_rows * cols_per_row
        while global_idx < loop_bound:
            row_idx = T.truncdiv(global_idx, cols_per_row)
            col_idx = T.truncmod(global_idx, cols_per_row)
            row_idx_in_expert = row_idx - expert_idx * m_rows

            if use_mask:
                if row_idx_in_expert >= mask[expert_idx]:
                    break

            in_offset = T.cast(row_idx, "int64") * actual_cols + col_idx
            T.ptx.ld.global_.v4.b64(
                x64[0],
                x64[1],
                x64[2],
                x64[3],
                T.address_of(input_global[in_offset * ELTS_PER_THREAD]),
            )
            if use_silu_and_mul != 0:
                T.ptx.ld.global_.v4.b64(
                    y64[0],
                    y64[1],
                    y64[2],
                    y64[3],
                    T.address_of(input_global[(in_offset + cols_per_row) * ELTS_PER_THREAD]),
                )
                # silu_and_mul (utils:1142-1166): fp32 silu*mul per element,
                # rounded back to DTYPE pairs in place.
                for i in T.unroll(8):
                    x_lo = _unpack_lo_f32(xw[i], dtype)
                    x_hi = _unpack_hi_f32(xw[i], dtype)
                    y_lo = _unpack_lo_f32(yw[i], dtype)
                    y_hi = _unpack_hi_f32(yw[i], dtype)
                    T.evaluate(
                        T.ptx.ex2.approx.ftz.f32(e_tmp[0], x_lo * T.float32(-1.4426950408889634))
                    )
                    out_lo = (x_lo / (T.float32(1.0) + e_tmp[0])) * y_lo
                    T.evaluate(
                        T.ptx.ex2.approx.ftz.f32(e_tmp[0], x_hi * T.float32(-1.4426950408889634))
                    )
                    out_hi = (x_hi / (T.float32(1.0) + e_tmp[0])) * y_hi
                    if dtype == "float16":
                        T.evaluate(T.ptx.cvt.rn.f16x2.f32(packed[0], out_hi, out_lo))
                    else:
                        T.evaluate(T.ptx.cvt.rn.bf16x2.f32(packed[0], out_hi, out_lo))
                    xw[i] = packed[0]

            out_offset = T.cast(row_idx, "int64") * cols_per_row + col_idx

            # SFScale select (branch-lowered in the source).
            sfscale_val: T.f32 = T.float32(1.0)
            if T.reinterpret("uint64", T.address_of(sf_scale[0])) != T.uint64(0):
                sfscale_val = sf_scale[expert_idx]

            # SF swizzled output address (utils:1096-1140 + quantization.cuh:706-714).
            num_cols_padded = (
                (num_cols + SF_VEC_SIZE * 4 - 1) // (SF_VEC_SIZE * 4) * (SF_VEC_SIZE * 4)
            )
            num_cols_sfout = num_cols_padded // SF_VEC_SIZE // 4
            sf_expert_base = expert_idx * padded_m * num_cols_sfout
            num_k_tiles = (num_cols + SF_VEC_SIZE * 4 - 1) // (SF_VEC_SIZE * 4)
            sf_off = (
                T.truncdiv(row_idx_in_expert, T.int32(128)) * (num_k_tiles * 512)
                + T.truncdiv(col_idx, T.int32(4)) * 512
                + (row_idx_in_expert % 32) * 16
                + T.truncdiv(row_idx_in_expert % 128, T.int32(32)) * 4
                + (col_idx % 4)
            )
            sf_byte = T.cast(sf_expert_base, "int64") * 4 + sf_off

            # Local abs-max over the 8 packed pairs (silu-rounded values).
            lm[0] = habs2(xw[0])
            for i in T.unroll(7):
                lm[0] = hmax2(lm[0], habs2(xw[i + 1]))
            lm_lo = T.cast(T.bitwise_and(lm[0], T.uint32(0xFFFF)), "uint16")
            lm_hi = T.cast(T.shift_right(lm[0], T.uint32(16)), "uint16")
            vec_max = T.cast(T.reinterpret(dtype, hmax(lm_lo, lm_hi)), "float32")

            # SF computation (default env: fast-math rcp, E4M3).
            T.evaluate(T.ptx.rcp.approx.ftz.f32(r_tmp[0], T.float32(6.0)))
            sf_value = sfscale_val * (vec_max * r_tmp[0])
            T.evaluate(T.ptx.cvt.rn.satfinite.e4m3x2.f32(e4m3_u16[0], T.float32(0.0), sf_value))
            sf_b8[0] = T.cast(e4m3_u16[0], "uint8")
            T.evaluate(T.ptx.cvt.rn.f16x2.e4m3x2(f16p[0], e4m3_u16[0]))
            sf_value_r = _unpack_lo_f32(f16p[0], "float16")
            output_scale: T.f32 = T.float32(0.0)
            if vec_max != 0.0:
                T.evaluate(T.ptx.rcp.approx.ftz.f32(r_tmp[0], sfscale_val))
                T.evaluate(T.ptx.rcp.approx.ftz.f32(e_tmp[0], sf_value_r * r_tmp[0]))
                output_scale = e_tmp[0]

            # SF byte store (STG.8, per thread).
            if T.reinterpret("uint64", T.address_of(sf_out[0])) != T.uint64(0):
                T.ptx.st.global_.b8(T.address_of(sf_out[sf_byte]), sf_b8[0])

            # Scale to e2m1 and pack (fp32_vec_to_e2m1 source asm block).
            for i in T.unroll(8):
                fp[2 * i] = _unpack_lo_f32(xw[i], dtype) * output_scale
                fp[2 * i + 1] = _unpack_hi_f32(xw[i], dtype) * output_scale
            e2m1_v[0] = _fp32_vec_to_e2m1_16([fp[i] for i in range(16)])
            T.ptx.st.global_.b64(T.address_of(out_global[out_offset]), e2m1_v[0])

            global_idx = global_idx + actual_stride

    return silu_and_mul_nvfp4_experts_quantize


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
    """Benchmark the TIRx port against the source thop (kernel-only)."""
    import torch

    from tirx_kernels.runner import compile_kernel

    a, mask, global_scale = prepare_data(
        dtype=dtype, n_experts=n_experts, m=m, k=k, mask_mode=mask_mode
    )
    kernel = get_kernel(dtype=dtype, n_experts=n_experts, m=m, k=k, mask_mode=mask_mode)
    ex = compile_kernel(kernel)
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
