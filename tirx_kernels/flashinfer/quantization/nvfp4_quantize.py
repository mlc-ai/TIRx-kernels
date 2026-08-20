# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer CuTe-DSL ``nvfp4_quantize`` port.

Ports ``NVFP4QuantizeLinearKernel`` / ``NVFP4QuantizeSwizzledKernel``
(``flashinfer/quantization/kernels/nvfp4_quantize.py``), the SM100 CuTe-DSL
kernels behind ``flashinfer.quantization.nvfp4_quantize(backend="cute-dsl")``
and ``flashinfer.quantization.silu_and_mul_nvfp4_quantize`` (the
``silu_and_mul=True`` compile variant of the same kernel classes).  Each
thread owns one 16-element SF block: two 128-bit loads (four with SwiGLU
fusion), a half2/bf16x2 absmax tree, no shuffle reduction, E4M3 block scale
computed from the global scale via ``rcp.approx.ftz(6.0)``, and 16 e2m1 values
packed into one ``st.global.u64``.

In-scope specialization: fp16/bf16 inputs, linear + swizzled 128x4/8x4 SF
layouts, device-tensor global scale, ``silu_and_mul`` on/off,
``enable_pdl=False`` (the griddepcontrol pair is ported behind the same
compile-time knob; TVM launches do not carry the PDL launch attribute, so PDL
stays off for test/bench parity on both sides).  Deferred variants:
``NVFP4QuantizeTMAKernel`` (env-gated off by default), the 4over6 dual-scale
path (env-gated), the ``FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH`` exact-math
path, and the fp8e4m3 input path.

The implementation structure follows the reviewer-approved sketch
``.agents/sketch/flashinfer/quantization/nvfp4_quantize.md``; shared instruction-level helpers live in
``tirx_kernels/flashinfer/utils/fp_quant.py``.
"""

from typing import Any

import tirx_kernels.kern as K
from tirx_kernels.flashinfer.utils.fp_quant import (
    absmax_8,
    cvt_e2m1x8,
    cvt_f32_to_e4m3,
    float2_scaled,
    ld_global_f32,
    ld_global_v4_u32,
    mul_f32,
    nvfp4_compute_output_scale,
    opaque_i32,
    pack_u32x2_to_u64,
    pair_max_to_f32,
    rcp_approx_ftz,
    sf_offset_8x4,
    sf_offset_128x4,
    silu_and_mul_pair,
    st_global_u8,
    st_global_u64,
)
from tirx_kernels.runner import bench

KERNEL_META = {"name": "nvfp4_quantize", "category": "flashinfer", "compute_capability": 10}

_DTYPES = ("float16", "bfloat16")
_SF_LAYOUTS = ("linear", "128x4", "8x4")

# Source constants (nvfp4_quantize.py:84-100, quantization_cute_dsl_utils.py).
NVFP4_SF_VEC_SIZE = 16
WARP_SIZE = 32
_BLOCKS_PER_SM = 4
_LINEAR_WARPS = 16  # _LINEAR_WARPS_PER_BLOCK
_LINEAR_SF_BLOCKS_PER_TB = 512  # _LINEAR_SF_BLOCKS_PER_TB
_MIN_THREADS = 128
_MAX_THREADS = 512
_ROW_TILE_128x4 = 128
_ROW_TILE_8x4 = 8

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


def _validate(dtype: str, m: int, k: int, sf_layout: str, fuse_silu: bool) -> None:
    if dtype not in _DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if sf_layout not in _SF_LAYOUTS:
        raise ValueError(f"Unsupported sf_layout: {sf_layout}")
    if m < 1:
        raise ValueError(f"m={m} must be >= 1")
    if k <= 0 or k % NVFP4_SF_VEC_SIZE != 0:
        raise ValueError(f"k={k} outside the source dispatch domain (k % 16 != 0)")


def _compute_optimal_threads(k: int) -> int:
    """Mirror ``_compute_optimal_threads`` (nvfp4_quantize.py:103-142)."""
    threads_per_row = k // NVFP4_SF_VEC_SIZE
    if threads_per_row > _MAX_THREADS:
        return _MAX_THREADS
    largest = (_MAX_THREADS // threads_per_row) * threads_per_row
    if largest >= _MIN_THREADS:
        return largest
    candidate = threads_per_row
    while candidate < _MIN_THREADS:
        candidate += threads_per_row
    if candidate <= _MAX_THREADS:
        return candidate
    return _MAX_THREADS


def _padded_m(m: int, sf_layout: str) -> int:
    tile = _ROW_TILE_8x4 if sf_layout == "8x4" else _ROW_TILE_128x4
    return (m + tile - 1) // tile * tile


def _padded_sf_cols(k: int) -> int:
    return (k // NVFP4_SF_VEC_SIZE + 3) // 4 * 4


def _sf_numel(m: int, k: int, sf_layout: str) -> int:
    if sf_layout == "linear":
        return m * (k // NVFP4_SF_VEC_SIZE)
    return _padded_m(m, sf_layout) * _padded_sf_cols(k)


def _linear_launch(m: int, k: int) -> tuple[int, int, int]:
    """(grid_x, block_x, total_sf_blocks) mirroring nvfp4_quantize.py:1941-1951."""
    total_sf_blocks = m * (k // NVFP4_SF_VEC_SIZE)
    grid = min(
        (total_sf_blocks + _LINEAR_SF_BLOCKS_PER_TB - 1) // _LINEAR_SF_BLOCKS_PER_TB,
        _sm_count() * _BLOCKS_PER_SM,
    )
    return grid, _LINEAR_WARPS * WARP_SIZE, total_sf_blocks


def _swizzled_launch(m: int, k: int, sf_layout: str) -> tuple[int, int, int]:
    """(grid_x, block_x, padded_m) mirroring nvfp4_quantize.py:1967-1980."""
    threads = _compute_optimal_threads(k)
    nsb = k // NVFP4_SF_VEC_SIZE
    rows_per_block = threads // nsb if nsb <= threads else 1
    padded_m = _padded_m(m, sf_layout)
    grid = min((padded_m + rows_per_block - 1) // rows_per_block, _sm_count() * _BLOCKS_PER_SM)
    return grid, threads, padded_m


def _process_block(in_global, row_idx, col_idx, gs, *, dtype, k, fuse_silu):
    """process_nvfp4_block_half/bfloat (utils:1870/:1895) or the
    silu_and_mul variant (utils:1948/:2004).  No stores; returns
    (scale_fp8_u8, packed64); the caller stores SF first, then the output.
    """
    in_cols = 2 * k if fuse_silu else k
    elem_base = col_idx * NVFP4_SF_VEC_SIZE
    base = row_idx * in_cols + elem_base
    v0 = ld_global_v4_u32(K.address_of(in_global[base]))
    v1 = ld_global_v4_u32(K.address_of(in_global[base + 8]))
    words = [v0[i] for i in range(4)] + [v1[i] for i in range(4)]
    if fuse_silu:
        u0 = ld_global_v4_u32(K.address_of(in_global[base + k]))
        u1 = ld_global_v4_u32(K.address_of(in_global[base + k + 8]))
        uwords = [u0[i] for i in range(4)] + [u1[i] for i in range(4)]
        words = [silu_and_mul_pair(words[i], uwords[i], dtype) for i in range(8)]

    block_max = pair_max_to_f32(absmax_8(words, dtype), dtype)
    # _nvfp4_standard_quant_from_amax, fast-math path (utils:1588).
    scale_float = mul_f32(gs, mul_f32(block_max, rcp_approx_ftz(K.float32(6.0))))
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, gs)

    s = []
    for i in range(8):
        lo, hi = float2_scaled(words[i], output_scale, dtype)
        s.append(lo)
        s.append(hi)
    packed_lo = cvt_e2m1x8(s[0:8])
    packed_hi = cvt_e2m1x8(s[8:16])
    packed64 = pack_u32x2_to_u64(packed_lo, packed_hi)
    return K.cast(scale_fp8_u32, "uint8"), packed64


def _materialize(value):
    local = K.alloc_local((1,), str(value.ty.dtype))
    K.assign(local[0], value)
    return local[0]


def get_kernel(
    dtype: str,
    m: int,
    k: int,
    sf_layout: str = "128x4",
    fuse_silu: bool = False,
    enable_pdl: bool = False,
    **kwargs,
):
    """Return the TIRx specialization for one (dtype, m, k, sf_layout, fuse_silu)."""
    _validate(dtype, m, k, sf_layout, fuse_silu)
    nsb = k // NVFP4_SF_VEC_SIZE
    pad_cols = _padded_sf_cols(k)

    def sf_offset(row, col):
        if sf_layout == "8x4":
            return sf_offset_8x4(row, col, pad_cols)
        return sf_offset_128x4(row, col, pad_cols)

    if sf_layout == "linear":
        grid_x, block_x, total_sf_blocks = _linear_launch(m, k)

        @K.kernel(warps=block_x // 32, arch="sm_100a", min_blocks_per_sm=2, grid=grid_x)
        def nvfp4_quantize_linear(
            in_global: K.gptr[dtype],
            out_global: K.gptr[K.u8],
            sf_out: K.gptr[K.u8],
            m_rows: K.i32,
            total_sf: K.i32,
            gs: K.gptr[K.f32],
        ):
            bx = K.cta_id()
            tx = K.thread_id()

            if enable_pdl:
                K.ptx.griddepcontrol.wait()

            # Device global scale: one broadcast load per thread (:322-325).
            gs_val = ld_global_f32(gs, 0)

            # Flat SF-block grid-stride loop (nvfp4_quantize.py:330-359).
            # The opaque stride keeps nvcc from strength-reducing the loop
            # into up-front pointer-induction chains; the source binary
            # recomputes addresses per iteration.
            stride = opaque_i32(grid_x * _LINEAR_SF_BLOCKS_PER_TB)
            sf_idx = K.alloc_local([1], "int32")
            K.assign(sf_idx[0], bx * _LINEAR_SF_BLOCKS_PER_TB + tx)
            with K.While(sf_idx[0] < total_sf):
                row_idx = K.truncdiv(sf_idx[0], K.int32(nsb))
                col_idx = K.truncmod(sf_idx[0], K.int32(nsb))
                scale_fp8, packed64 = _process_block(
                    in_global, row_idx, col_idx, gs_val, dtype=dtype, k=k, fuse_silu=fuse_silu
                )
                # Source order: SF byte store, then the output store (:351/:357).
                st_global_u8(K.address_of(sf_out[sf_idx[0]]), scale_fp8)
                out_off = row_idx * (k // 2) + col_idx * 8
                st_global_u64(K.address_of(out_global[out_off]), packed64)
                K.assign(sf_idx[0], sf_idx[0] + stride)

            if enable_pdl:
                K.ptx.griddepcontrol.launch_dependents()

        return nvfp4_quantize_linear.func

    grid_x, block_x, padded_m = _swizzled_launch(m, k, sf_layout)
    needs_col_loop = nsb > block_x
    rows_per_block = 1 if needs_col_loop else block_x // nsb

    @K.kernel(warps=(block_x + 31) // 32, arch="sm_100a", min_blocks_per_sm=2, grid=grid_x)
    def nvfp4_quantize_swizzled(
        in_global: K.gptr[dtype],
        out_global: K.gptr[K.u8],
        sf_out: K.gptr[K.u8],
        m_rows: K.i32,
        padded_rows: K.i32,
        gs: K.gptr[K.f32],
    ):
        bx = K.cta_id()
        tx = K.thread_id()

        if enable_pdl:
            K.ptx.griddepcontrol.wait()

        gs_val = ld_global_f32(gs, 0)

        def col_body():
            # Large K (K/16 > 512): one row per block iteration with a column
            # loop (nvfp4_quantize.py:523-576).
            row_idx = K.alloc_local([1], "int32")
            K.assign(row_idx[0], bx)
            with K.While(row_idx[0] < padded_rows):
                with K.If(row_idx[0] >= m_rows):
                    with K.Then():
                        # Padding row: zero-fill by ALL threads, stride block_x.
                        sc_pad = K.alloc_local([1], "int32")
                        K.assign(sc_pad[0], tx)
                        with K.While(sc_pad[0] < pad_cols):
                            st_global_u8(
                                K.address_of(sf_out[sf_offset(row_idx[0], sc_pad[0])]), K.uint8(0)
                            )
                            K.assign(sc_pad[0], sc_pad[0] + block_x)
                    with K.Else():
                        sc = K.alloc_local([1], "int32")
                        K.assign(sc[0], tx)
                        with K.While(sc[0] < nsb):
                            scale_fp8, packed64 = _process_block(
                                in_global,
                                row_idx[0],
                                sc[0],
                                gs_val,
                                dtype=dtype,
                                k=k,
                                fuse_silu=fuse_silu,
                            )
                            # Source order: SF byte store, then output (:557/:563).
                            st_global_u8(
                                K.address_of(sf_out[sf_offset(row_idx[0], sc[0])]), scale_fp8
                            )
                            out_off = row_idx[0] * (k // 2) + sc[0] * 8
                            st_global_u64(K.address_of(out_global[out_off]), packed64)
                            K.assign(sc[0], sc[0] + block_x)
                        # Padding SF columns of a data row (:568-574).
                        sc_tail = K.alloc_local([1], "int32")
                        K.assign(sc_tail[0], nsb + tx)
                        with K.While(sc_tail[0] < pad_cols):
                            st_global_u8(
                                K.address_of(sf_out[sf_offset(row_idx[0], sc_tail[0])]), K.uint8(0)
                            )
                            K.assign(sc_tail[0], sc_tail[0] + block_x)
                K.assign(row_idx[0], row_idx[0] + grid_x)

        def small_body():
            # Small K: multi-row processing (nvfp4_quantize.py:577-642).
            row_in_block = _materialize(K.truncdiv(tx, K.int32(nsb)))
            sf_idx_in_row = _materialize(K.truncmod(tx, K.int32(nsb)))

            row_batch_idx = K.alloc_local([1], "int32")
            row_idx2 = K.alloc_local([1], "int32")
            K.assign(row_batch_idx[0], bx)
            K.assign(row_idx2[0], row_batch_idx[0] * rows_per_block + row_in_block)
            with K.While(row_batch_idx[0] * rows_per_block < padded_rows):
                with K.If(row_idx2[0] < padded_rows), K.Then():
                    with K.If(row_idx2[0] >= m_rows):
                        with K.Then():
                            # Padding row: zero ALL padded SF columns; stride is
                            # threads_per_row == nsb (:597-603).
                            local_sf = K.alloc_local([1], "int32")
                            K.assign(local_sf[0], sf_idx_in_row)
                            with K.While(local_sf[0] < pad_cols):
                                st_global_u8(
                                    K.address_of(sf_out[sf_offset(row_idx2[0], local_sf[0])]),
                                    K.uint8(0),
                                )
                                K.assign(local_sf[0], local_sf[0] + nsb)
                        with K.Else():
                            with K.If(sf_idx_in_row < nsb), K.Then():
                                scale_fp8, packed64 = _process_block(
                                    in_global,
                                    row_idx2[0],
                                    sf_idx_in_row,
                                    gs_val,
                                    dtype=dtype,
                                    k=k,
                                    fuse_silu=fuse_silu,
                                )
                                # Source order: SF byte store, then output (:619/:625).
                                st_global_u8(
                                    K.address_of(sf_out[sf_offset(row_idx2[0], sf_idx_in_row)]),
                                    scale_fp8,
                                )
                                out_off = row_idx2[0] * (k // 2) + sf_idx_in_row * 8
                                st_global_u64(K.address_of(out_global[out_off]), packed64)
                            # Padding SF columns of a data row (:627-638).
                            if pad_cols != nsb:
                                pad_col = K.alloc_local([1], "int32")
                                K.assign(pad_col[0], nsb + sf_idx_in_row)
                                with K.While(pad_col[0] < pad_cols):
                                    st_global_u8(
                                        K.address_of(sf_out[sf_offset(row_idx2[0], pad_col[0])]),
                                        K.uint8(0),
                                    )
                                    K.assign(pad_col[0], pad_col[0] + nsb)
                K.assign(row_batch_idx[0], row_batch_idx[0] + grid_x)
                K.assign(row_idx2[0], row_batch_idx[0] * rows_per_block + row_in_block)

        if block_x % 32:
            with K.If(tx < block_x), K.Then():
                col_body() if needs_col_loop else small_body()
        elif needs_col_loop:
            col_body()
        else:
            small_body()

        if enable_pdl:
            K.ptx.griddepcontrol.launch_dependents()

    return nvfp4_quantize_swizzled.func


def prepare_data(
    dtype: str,
    m: int,
    k: int,
    sf_layout: str = "128x4",
    fuse_silu: bool = False,
    enable_pdl: bool = False,
    **kwargs,
):
    """Create logical inputs: a [m, k] (or [m, 2k] with fuse_silu) tensor and
    the [1] fp32 global scale."""
    import torch

    _validate(dtype, m, k, sf_layout, fuse_silu)
    torch.manual_seed(42)
    a = torch.randn(m, k * (2 if fuse_silu else 1), dtype=_torch_dtype(dtype), device="cuda")
    global_scale = torch.rand(1, dtype=torch.float32, device="cuda") + 0.5
    return a, global_scale


def _alloc_outputs(m: int, k: int, sf_layout: str):
    import torch

    out = torch.empty(m, k // 2, dtype=torch.uint8, device="cuda")
    sf = torch.empty(_sf_numel(m, k, sf_layout), dtype=torch.uint8, device="cuda")
    return out, sf


def _sf_layout_enum(sf_layout: str):
    from flashinfer.tllm_enums import SfLayout

    return {
        "linear": SfLayout.layout_linear,
        "128x4": SfLayout.layout_128x4,
        "8x4": SfLayout.layout_8x4,
    }[sf_layout]


def _run_reference(a, global_scale, sf_layout: str, fuse_silu: bool, enable_pdl: bool):
    """Run the FlashInfer CuTe-DSL source wrapper (allocates its own outputs)."""
    if fuse_silu:
        from flashinfer.quantization import silu_and_mul_nvfp4_quantize

        return silu_and_mul_nvfp4_quantize(
            a,
            global_scale,
            is_sf_swizzled_layout=sf_layout != "linear",
            is_sf_8x4_layout=sf_layout == "8x4",
            enable_pdl=enable_pdl,
        )
    from flashinfer.quantization import nvfp4_quantize

    return nvfp4_quantize(
        a,
        global_scale,
        sfLayout=_sf_layout_enum(sf_layout),
        backend="cute-dsl",
        enable_pdl=enable_pdl,
    )


def _run_launch(ex, a, gs, out, sf, m, k, sf_layout):
    if sf_layout == "linear":
        ex(a.view(-1), out.view(-1), sf, m, m * (k // NVFP4_SF_VEC_SIZE), gs)
    else:
        ex(a.view(-1), out.view(-1), sf, m, _padded_m(m, sf_layout), gs)


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(
    dtype: str,
    m: int,
    k: int,
    sf_layout: str = "128x4",
    fuse_silu: bool = False,
    enable_pdl: bool = False,
    **kwargs,
):
    """Compile, launch, and validate one config against the flashinfer source."""
    import torch

    from tirx_kernels.runner import compile_kernel

    a, gs = prepare_data(dtype=dtype, m=m, k=k, sf_layout=sf_layout, fuse_silu=fuse_silu)
    kernel = get_kernel(
        dtype=dtype, m=m, k=k, sf_layout=sf_layout, fuse_silu=fuse_silu, enable_pdl=enable_pdl
    )
    ex = compile_kernel(kernel)
    out_tirx, sf_tirx = _alloc_outputs(m, k, sf_layout)
    _run_launch(ex, a, gs, out_tirx, sf_tirx, m, k, sf_layout)
    torch.cuda.synchronize()

    ref_fp4, ref_sf = _run_reference(a, gs, sf_layout, fuse_silu, enable_pdl)
    torch.testing.assert_close(out_tirx, ref_fp4, rtol=0, atol=0)
    torch.testing.assert_close(sf_tirx, ref_sf.reshape(-1), rtol=0, atol=0)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs):
    """Benchmark the TIRx port against the CuTe-DSL source (kernel-only)."""
    config = dict(prepared["config"])
    dtype = config.pop("dtype")
    m = config.pop("m")
    k = config.pop("k")
    sf_layout = config.pop("sf_layout")
    fuse_silu = config.pop("fuse_silu")
    enable_pdl = config.pop("enable_pdl")
    config.update(kwargs)
    kwargs = config
    executable = prepared["executable"]

    a, gs = prepare_data(dtype=dtype, m=m, k=k, sf_layout=sf_layout, fuse_silu=fuse_silu)
    ex = executable
    out_tirx, sf_tirx = _alloc_outputs(m, k, sf_layout)

    def tirx_launch():
        _run_launch(ex, a, gs, out_tirx, sf_tirx, m, k, sf_layout)

    def build_reference():
        # Bypass the allocating public wrapper: call the cached compiled source
        # kernel directly with preallocated outputs (kernel-only timing).
        from flashinfer.quantization.kernels.nvfp4_quantize import (
            SF_LAYOUT_LINEAR,
            SF_LAYOUT_8x4,
            SF_LAYOUT_128x4,
            _get_compiled_kernel_nvfp4,
        )

        layout_code = {"linear": SF_LAYOUT_LINEAR, "128x4": SF_LAYOUT_128x4, "8x4": SF_LAYOUT_8x4}[
            sf_layout
        ]
        kernel_fn, _ = _get_compiled_kernel_nvfp4(
            dtype,
            k,
            layout_code,
            enable_pdl,
            False,  # disable_fp4_quant_fast_math
            None,  # nvfp4_4over6_config
            fuse_silu,
            True,  # global_scale_is_tensor
        )
        out_ref, sf_ref = _alloc_outputs(m, k, sf_layout)
        if sf_layout == "linear":
            grid, _, _ = _linear_launch(m, k)
            total_sf = m * (k // NVFP4_SF_VEC_SIZE)
            return lambda: kernel_fn(a, out_ref, sf_ref, m, total_sf, grid, gs)
        padded_m = _padded_m(m, sf_layout)
        grid, _, _ = _swizzled_launch(m, k, sf_layout)
        return lambda: kernel_fn(a, out_ref, sf_ref, m, padded_m, grid, gs)

    return bench(
        {"tirx": tirx_launch},
        references={"flashinfer": build_reference},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    dtype: str,
    m: int,
    k: int,
    sf_layout: str = "128x4",
    fuse_silu: bool = False,
    enable_pdl: bool = False,
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
        dtype=dtype,
        m=m,
        k=k,
        sf_layout=sf_layout,
        fuse_silu=fuse_silu,
        enable_pdl=enable_pdl,
        **config,
    )
    return prepared.run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


def _cfg(dtype, m, k, sf_layout="128x4", fuse_silu=False, enable_pdl=False):
    dt = {"float16": "fp16", "bfloat16": "bf16"}[dtype]
    silu = "_silu" if fuse_silu else ""
    pdl = "_pdl" if enable_pdl else ""
    return {
        "label": f"{dt}_{sf_layout}_m{m}_k{k}{silu}{pdl}",
        "dtype": dtype,
        "m": m,
        "k": k,
        "sf_layout": sf_layout,
        "fuse_silu": fuse_silu,
        "enable_pdl": enable_pdl,
    }


# Correctness matrix.  Covers: both dtypes; linear/128x4/8x4 SF layouts; the
# silu_and_mul fusion on both kernel classes; the swizzled multi-row vs
# needs_col_loop compile-time split (K/16 > 512); padding-row and
# padding-column zero-fill paths (m % 128, m % 8, k/16 % 4 != 0); minimal
# shapes; the PDL instruction variant.
CONFIGS = [
    _cfg("float16", 1, 16, "linear"),  # minimal (nsb=1)
    _cfg("float16", 128, 1024, "linear"),
    _cfg("bfloat16", 128, 1024, "linear"),
    _cfg("float16", 512, 4096, "linear"),
    _cfg("bfloat16", 512, 4096, "linear"),
    _cfg("float16", 13, 1040, "linear"),  # odd m, nsb=65
    _cfg("float16", 128, 1024, "128x4"),  # multi-row (threads 512, rpb 8)
    _cfg("bfloat16", 128, 1024, "128x4"),
    _cfg("float16", 120, 1024, "128x4"),  # row padding 120 -> 128
    _cfg("float16", 128, 1040, "128x4"),  # col padding 65 -> 68, threads 455
    _cfg("float16", 512, 4096, "128x4"),  # multi-row (threads 512, rpb 2)
    _cfg("bfloat16", 512, 4096, "128x4"),
    _cfg("float16", 64, 8256, "128x4"),  # needs_col_loop (516 SF/row)
    _cfg("float16", 64, 8208, "128x4"),  # col loop + col padding (513 -> 516)
    _cfg("float16", 13, 1024, "8x4"),  # 8x4 row padding 13 -> 16
    _cfg("bfloat16", 128, 1024, "8x4"),
    _cfg("float16", 128, 1024, "linear", True),  # silu linear
    _cfg("float16", 128, 1024, "128x4", True),  # silu multi-row
    _cfg("bfloat16", 512, 4096, "128x4", True),  # silu bf16
    _cfg("float16", 64, 8256, "128x4", True),  # silu col loop
    _cfg("float16", 512, 4096, "linear", False, True),  # PDL variant
]

# Benchmark sweep: linear and 128x4, realistic LLM shapes, plus SwiGLU.
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
    _cfg("float16", 4096, 3584, "128x4", True),  # SwiGLU (input [4096, 7168])
    _cfg("bfloat16", 4096, 3584, "128x4", True),
]
