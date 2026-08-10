# Copyright (c) 2026 The TIRX Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""FlashInfer CuTe-DSL ``nvfp4_quantize_per_token`` port.

Ports ``NVFP4QuantizePerTokenKernel``
(``flashinfer/quantization/kernels/nvfp4_quantize.py``), the SM100 CuTe-DSL
kernel behind ``flashinfer.quantization.nvfp4_quantize(...,
per_token_activation=True, backend="cute-dsl")``.  One CTA (128 threads, 4
warps) per token row: a column-stride amax pass, a warp+block max reduction
through a 16-byte shared buffer, a per-row encode scale, then a second
column-stride pass that reuses the regular NVFP4 block quantizer.  Row
addresses are built from explicit 64-bit arithmetic, exactly like the source.

In-scope specialization: fp16/bf16 inputs, linear + swizzled 128x4/8x4 SF
layouts, ``enable_pdl=False`` (the griddepcontrol pair is ported behind the
same compile-time knob; TVM launches do not carry the PDL launch attribute, so
PDL stays off for test/bench parity on both sides).  Deferred variants:
``FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH`` exact-math path and the 4over6
dual-scale path (both env-gated).

The implementation structure follows the reviewer-approved sketch
``.agents/sketch/nvfp4_quantize_per_token.md``; shared instruction-level
helpers live in ``tirx_kernels/flashinfer/utils/fp_quant.py``.
"""

from tirx_kernels.flashinfer.utils.fp_quant import (
    absmax_8,
    cvt_e2m1x8,
    cvt_f32_to_e4m3,
    float2_scaled,
    fmax_f32,
    ld_global_f32,
    ld_global_v4_u32,
    mul_f32,
    nvfp4_compute_output_scale,
    pack_u32x2_to_u64,
    pair_max_to_f32,
    rcp_approx_ftz,
    sf_offset_8x4,
    sf_offset_128x4,
    st_global_u8,
    st_global_u64,
    warp_reduce_max,
)
from tvm.script import tirx as T
from tvm.tirx.bench import bench

FLOAT32_MAX = 3.4028234663852886e38


def _process_block_pt(in_global, elem_off, encode_scale, *, dtype):
    """process_nvfp4_block_half/bfloat (utils:1870/:1895) for the per-token
    kernel: loads at the 64-bit row base + column offset, quantizes with the
    row's global_encode_scale.  No stores; returns (scale_fp8_u8, packed64).
    """
    v0 = ld_global_v4_u32(T.address_of(in_global[elem_off]))
    v1 = ld_global_v4_u32(T.address_of(in_global[elem_off + 8]))
    words = [v0[i] for i in range(4)] + [v1[i] for i in range(4)]
    block_max = pair_max_to_f32(absmax_8(words, dtype), dtype)
    scale_float = mul_f32(encode_scale, mul_f32(block_max, rcp_approx_ftz(T.float32(6.0))))
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, encode_scale)
    s = []
    for i in range(8):
        lo, hi = float2_scaled(words[i], output_scale, dtype)
        s.append(lo)
        s.append(hi)
    packed64 = pack_u32x2_to_u64(cvt_e2m1x8(s[0:8]), cvt_e2m1x8(s[8:16]))
    return T.cast(scale_fp8_u32, "uint8"), packed64


KERNEL_META = {
    "name": "nvfp4_quantize_per_token",
    "category": "flashinfer",
    "compute_capability": 10,
}

_DTYPES = ("float16", "bfloat16")
_SF_LAYOUTS = ("linear", "128x4", "8x4")

# Source constants (nvfp4_quantize.py:654-655, quantization_cute_dsl_utils.py).
NVFP4_SF_VEC_SIZE = 16
_PER_TOKEN_THREADS = 128
_PER_TOKEN_WARPS = 4
_BLOCKS_PER_SM = 4
_ROW_TILE_128x4 = 128
_ROW_TILE_8x4 = 8


def _torch_dtype(dtype: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


def _validate(dtype: str, m: int, k: int, sf_layout: str) -> None:
    if dtype not in _DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if sf_layout not in _SF_LAYOUTS:
        raise ValueError(f"Unsupported sf_layout: {sf_layout}")
    if m < 1:
        raise ValueError(f"m={m} must be >= 1")
    if k <= 0 or k % NVFP4_SF_VEC_SIZE != 0:
        raise ValueError(f"k={k} outside the source dispatch domain (k % 16 != 0)")


def _padded_m(m: int, sf_layout: str) -> int:
    if sf_layout == "linear":
        return m
    tile = _ROW_TILE_8x4 if sf_layout == "8x4" else _ROW_TILE_128x4
    return (m + tile - 1) // tile * tile


def _padded_sf_cols(k: int, sf_layout: str) -> int:
    if sf_layout == "linear":
        return k // NVFP4_SF_VEC_SIZE
    return (k // NVFP4_SF_VEC_SIZE + 3) // 4 * 4


def _sf_numel(m: int, k: int, sf_layout: str) -> int:
    return _padded_m(m, sf_layout) * _padded_sf_cols(k, sf_layout)


def get_kernel(
    dtype: str, m: int, k: int, sf_layout: str = "128x4", enable_pdl: bool = False, **kwargs
):
    """Return the TIRx specialization for one (dtype, m, k, sf_layout) config."""
    _validate(dtype, m, k, sf_layout)
    nsb = k // NVFP4_SF_VEC_SIZE
    pad_cols = _padded_sf_cols(k, sf_layout)

    def sf_offset(row, col):
        if sf_layout == "8x4":
            return sf_offset_8x4(row, col, pad_cols)
        if sf_layout == "linear":
            return row * pad_cols + col  # compute_sf_index_linear_gpu
        return sf_offset_128x4(row, col, pad_cols)

    @T.prim_func
    def nvfp4_quantize_per_token(
        in_ptr: T.handle,
        out_ptr: T.handle,
        sf_ptr: T.handle,
        pts_ptr: T.handle,
        m_rows: T.int32,
        gsi_ptr: T.handle,
    ):
        in_global = T.match_buffer(in_ptr, shape=[m * k], dtype=dtype, scope="global")
        out_global = T.match_buffer(out_ptr, shape=[m * (k // 2)], dtype="uint8", scope="global")
        sf_out = T.match_buffer(
            sf_ptr, shape=[_sf_numel(m, k, sf_layout)], dtype="uint8", scope="global"
        )
        pts_out = T.match_buffer(pts_ptr, shape=[m], dtype="float32", scope="global")
        gsi = T.match_buffer(gsi_ptr, shape=[1], dtype="float32", scope="global")
        T.device_entry()
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 2})
        bx = T.cta_id([m])
        tx = T.thread_id([_PER_TOKEN_THREADS])

        if enable_pdl:
            T.evaluate(T.ptx.griddepcontrol.wait())

        red_buf = T.alloc_shared([_PER_TOKEN_WARPS], "float32")

        # One CTA per row; 64-bit row bases (kernel:786-817).
        row_idx = bx
        in_row = T.cast(row_idx, "int64") * k
        out_row = T.cast(row_idx, "int64") * (k // 2)

        # Pass 1: row amax (kernel:819-834).
        local_amax: T.float32 = T.float32(0.0)
        sf_col: T.int32 = tx
        while sf_col < nsb:
            elem_off = in_row + sf_col * NVFP4_SF_VEC_SIZE
            v0 = ld_global_v4_u32(T.address_of(in_global[elem_off]))
            v1 = ld_global_v4_u32(T.address_of(in_global[elem_off + 8]))
            words = [v0[i] for i in range(4)] + [v1[i] for i in range(4)]
            block_max = pair_max_to_f32(absmax_8(words, dtype), dtype)
            local_amax = fmax_f32(local_amax, block_max)
            sf_col = sf_col + _PER_TOKEN_THREADS

        # Warp + block max reduction (kernel:836-837; fp4_common:1356-1391).
        warp_amax = warp_reduce_max(local_amax)
        lane = T.truncmod(tx, T.int32(32))
        warp = T.truncdiv(tx, T.int32(32))
        if lane == 0:
            red_buf[warp] = warp_amax
        T.ptx.bar.sync(T.uint32(0), T.uint32(_PER_TOKEN_THREADS))
        block_val: T.float32 = T.float32(0.0)
        if lane < _PER_TOKEN_WARPS:
            block_val = red_buf[lane]
        row_amax = warp_reduce_max(block_val)

        gs_inv = ld_global_f32(gsi, 0)
        # _row_scales, fast-math path (kernel:729-738).
        encode_scale: T.float32 = T.float32(0.0)
        token_scale: T.float32 = T.float32(0.0)
        if row_amax == T.float32(0.0):
            encode_scale = T.float32(FLOAT32_MAX)
            token_scale = T.float32(0.0)
        else:
            token_scale = mul_f32(row_amax, gs_inv)
            encode_scale = rcp_approx_ftz(token_scale)

        if tx == 0:
            pts_out[row_idx] = token_scale
        T.ptx.bar.sync(T.uint32(0), T.uint32(_PER_TOKEN_THREADS))

        # Pass 2: quantize with the row encode scale (kernel:846-875).
        sf_col2: T.int32 = tx
        while sf_col2 < nsb:
            scale_fp8, packed64 = _process_block_pt(
                in_global, in_row + sf_col2 * NVFP4_SF_VEC_SIZE, encode_scale, dtype=dtype
            )
            # Source order: SF byte store, then output store (:869/:873).
            st_global_u8(T.address_of(sf_out[sf_offset(row_idx, sf_col2)]), scale_fp8)
            st_global_u64(T.address_of(out_global[out_row + sf_col2 * 8]), packed64)
            sf_col2 = sf_col2 + _PER_TOKEN_THREADS

        # Padding SF columns for swizzled layouts (kernel:877-882).
        if sf_layout != "linear":
            sf_pad: T.int32 = nsb + tx
            while sf_pad < pad_cols:
                st_global_u8(T.address_of(sf_out[sf_offset(row_idx, sf_pad)]), T.uint8(0))
                sf_pad = sf_pad + _PER_TOKEN_THREADS

        if enable_pdl:
            T.evaluate(T.ptx.griddepcontrol.launch_dependents())

    return nvfp4_quantize_per_token


def prepare_data(
    dtype: str,
    m: int,
    k: int,
    sf_layout: str = "128x4",
    enable_pdl: bool = False,
    zero_row: bool = False,
    **kwargs,
):
    """Create logical inputs: a [m, k] tensor and the [1] fp32 inverse global
    scale.  ``zero_row`` blanks row 0 to cover the row_amax == 0 path."""
    import torch

    _validate(dtype, m, k, sf_layout)
    torch.manual_seed(42)
    a = torch.randn(m, k, dtype=_torch_dtype(dtype), device="cuda")
    if zero_row:
        a[0].zero_()
    # Typical inverse base scale multiplier: 1 / (448 * 6).
    gs_inv = torch.tensor([1.0 / (448.0 * 6.0)], dtype=torch.float32, device="cuda")
    return a, gs_inv


def _alloc_outputs(m: int, k: int, sf_layout: str):
    import torch

    out = torch.empty(m, k // 2, dtype=torch.uint8, device="cuda")
    # The source host wrapper zero-fills the SF buffer (torch.zeros): padding
    # rows for swizzled layouts are pre-zeroed by the host, padding columns by
    # the kernel.
    sf = torch.zeros(_sf_numel(m, k, sf_layout), dtype=torch.uint8, device="cuda")
    pts = torch.empty(m, dtype=torch.float32, device="cuda")
    return out, sf, pts


def _sf_layout_enum(sf_layout: str):
    from flashinfer.tllm_enums import SfLayout

    return {
        "linear": SfLayout.layout_linear,
        "128x4": SfLayout.layout_128x4,
        "8x4": SfLayout.layout_8x4,
    }[sf_layout]


def _run_reference(a, gs_inv, sf_layout: str, enable_pdl: bool):
    """Run the FlashInfer CuTe-DSL source wrapper (allocates its own outputs)."""
    from flashinfer.quantization import nvfp4_quantize

    return nvfp4_quantize(
        a,
        gs_inv,
        sfLayout=_sf_layout_enum(sf_layout),
        backend="cute-dsl",
        per_token_activation=True,
        enable_pdl=enable_pdl,
    )


def run_test(
    dtype: str,
    m: int,
    k: int,
    sf_layout: str = "128x4",
    enable_pdl: bool = False,
    zero_row: bool = False,
    **kwargs,
):
    """Compile, launch, and validate one config against the flashinfer source."""
    import torch

    from tirx_kernels.runner import compile_kernel

    a, gs_inv = prepare_data(dtype=dtype, m=m, k=k, sf_layout=sf_layout, zero_row=zero_row)
    kernel = get_kernel(dtype=dtype, m=m, k=k, sf_layout=sf_layout, enable_pdl=enable_pdl)
    ex = compile_kernel(kernel)
    out_tirx, sf_tirx, pts_tirx = _alloc_outputs(m, k, sf_layout)
    ex(a.view(-1), out_tirx.view(-1), sf_tirx, pts_tirx, m, gs_inv)
    torch.cuda.synchronize()

    ref_fp4, ref_sf, ref_pts = _run_reference(a, gs_inv, sf_layout, enable_pdl)
    torch.testing.assert_close(out_tirx, ref_fp4, rtol=0, atol=0)
    torch.testing.assert_close(sf_tirx, ref_sf.reshape(-1), rtol=0, atol=0)
    torch.testing.assert_close(pts_tirx, ref_pts, rtol=0, atol=0)


def run_bench(
    dtype: str,
    m: int,
    k: int,
    sf_layout: str = "128x4",
    enable_pdl: bool = False,
    zero_row: bool = False,
    *,
    warmup=None,
    repeat=None,
    timer=None,
    rounds=1,
    cooldown_s=1.0,
    **kwargs,
):
    """Benchmark the TIRx port against the CuTe-DSL source (kernel-only)."""

    from tirx_kernels.runner import compile_kernel

    a, gs_inv = prepare_data(dtype=dtype, m=m, k=k, sf_layout=sf_layout, zero_row=zero_row)
    kernel = get_kernel(dtype=dtype, m=m, k=k, sf_layout=sf_layout, enable_pdl=enable_pdl)
    ex = compile_kernel(kernel)
    out_tirx, sf_tirx, pts_tirx = _alloc_outputs(m, k, sf_layout)

    def tirx_launch():
        ex(a.view(-1), out_tirx.view(-1), sf_tirx, pts_tirx, m, gs_inv)

    def build_reference():
        # Bypass the allocating public wrapper: call the cached compiled source
        # kernel directly with preallocated outputs (kernel-only timing).
        from flashinfer.quantization.kernels.nvfp4_quantize import (
            SF_LAYOUT_LINEAR,
            SF_LAYOUT_8x4,
            SF_LAYOUT_128x4,
            _get_compiled_kernel_nvfp4_per_token,
        )

        layout_code = {"linear": SF_LAYOUT_LINEAR, "128x4": SF_LAYOUT_128x4, "8x4": SF_LAYOUT_8x4}[
            sf_layout
        ]
        kernel_fn = _get_compiled_kernel_nvfp4_per_token(
            dtype,
            k,
            layout_code,
            enable_pdl,
            False,  # disable_fp4_quant_fast_math
            None,  # nvfp4_4over6_config
        )
        out_ref, sf_ref, pts_ref = _alloc_outputs(m, k, sf_layout)
        return lambda: kernel_fn(a, out_ref, sf_ref, pts_ref, m, gs_inv)

    return bench(
        {"tirx": tirx_launch},
        references={"flashinfer": build_reference},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def _cfg(dtype, m, k, sf_layout="128x4", enable_pdl=False, zero_row=False):
    dt = {"float16": "fp16", "bfloat16": "bf16"}[dtype]
    pdl = "_pdl" if enable_pdl else ""
    zr = "_zerorow" if zero_row else ""
    return {
        "label": f"{dt}_{sf_layout}_m{m}_k{k}{zr}{pdl}",
        "dtype": dtype,
        "m": m,
        "k": k,
        "sf_layout": sf_layout,
        "enable_pdl": enable_pdl,
        "zero_row": zero_row,
    }


# Correctness matrix.  Covers: both dtypes; linear/128x4/8x4 SF layouts; the
# row_amax == 0 path (zero_row); swizzled row padding (host-zeroed) and column
# padding (kernel-zeroed); multi-iteration column loops (K/16 > 128); minimal
# shapes; the PDL instruction variant.
CONFIGS = [
    _cfg("float16", 1, 16, "linear"),  # minimal (nsb=1)
    _cfg("float16", 128, 1024, "linear"),
    _cfg("bfloat16", 128, 1024, "linear"),
    _cfg("float16", 512, 4096, "linear"),
    _cfg("bfloat16", 512, 4096, "linear"),
    _cfg("float16", 13, 1040, "linear"),  # odd m, nsb=65
    _cfg("float16", 128, 1024, "linear", False, True),  # row_amax == 0 path
    _cfg("float16", 128, 1024, "128x4"),
    _cfg("bfloat16", 128, 1024, "128x4"),
    _cfg("float16", 120, 1024, "128x4"),  # row padding 120 -> 128 (host zeros)
    _cfg("float16", 128, 1040, "128x4"),  # col padding 65 -> 68 (kernel zeros)
    _cfg("float16", 512, 4096, "128x4"),
    _cfg("bfloat16", 512, 4096, "128x4"),
    _cfg("float16", 64, 8256, "128x4"),  # 4-iteration column loop (516 SF/row)
    _cfg("float16", 13, 1024, "8x4"),  # 8x4 row padding 13 -> 16
    _cfg("bfloat16", 128, 1024, "8x4"),
    _cfg("float16", 512, 4096, "128x4", True),  # PDL variant
]

# Benchmark sweep: linear and 128x4, realistic activation shapes.
BENCH_CONFIGS = [
    _cfg("float16", 4096, 4096, "linear"),
    _cfg("bfloat16", 4096, 4096, "linear"),
    _cfg("float16", 4096, 4096, "128x4"),
    _cfg("bfloat16", 4096, 4096, "128x4"),
    _cfg("float16", 16384, 7168, "linear"),
    _cfg("float16", 16384, 7168, "128x4"),
    _cfg("float16", 1024, 2048, "linear"),
    _cfg("float16", 1024, 2048, "128x4"),
    _cfg("float16", 128, 1024, "linear"),
    _cfg("float16", 128, 1024, "128x4"),
]
