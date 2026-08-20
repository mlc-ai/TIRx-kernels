# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer CuTe-DSL ``mxfp8_quantize`` port.

Ports ``MXFP8QuantizeLinearKernel`` / ``MXFP8QuantizeSwizzledKernel``
(``flashinfer/quantization/kernels/mxfp8_quantize.py``), the SM100 CuTe-DSL
kernels behind ``flashinfer.quantization.mxfp8_quantize(backend="cute-dsl")``.
Each thread block quantizes 32-element SF blocks to FP8-E4M3 with UE8M0 block
scales, using 128-bit global loads, half2/bf16x2 absmax trees, a 2- or 4-thread
butterfly-shuffle max reduction, and single ``st.global.u64`` output stores.

In-scope specialization: fp16/bf16 inputs, linear + swizzled 128x4/8x4 SF
layouts, 2T/SF and 4T/SF thread configurations (host dispatch
``m * (K / 32) >= 65536``), ``enable_pdl=False`` (the griddepcontrol pair is
ported behind the same compile-time knob; TVM launches do not carry the PDL
launch attribute, so PDL stays off for test/bench parity on both sides).

The implementation structure follows the reviewer-approved sketch
``.agents/sketch/flashinfer/quantization/mxfp8_quantize.md``; shared instruction-level helpers live in
``tirx_kernels/flashinfer/utils/fp_quant.py``.
"""

from typing import Any

import tirx_kernels.kern as K
from tirx_kernels.flashinfer.utils.fp_quant import (
    absmax_4,
    absmax_8,
    float_to_ue8m0,
    fp8x8_scaled,
    ld_global_v4_u32,
    mul_f32,
    pair_max_to_f32,
    reduce_max_2threads,
    reduce_max_4threads,
    sf_offset_8x4,
    sf_offset_128x4,
    st_global_u8,
    st_global_u64,
    ue8m0_to_inv_scale,
)
from tirx_kernels.runner import bench

KERNEL_META = {"name": "mxfp8_quantize", "category": "flashinfer", "compute_capability": 10}

_DTYPES = ("float16", "bfloat16")
_SF_LAYOUTS = ("linear", "128x4", "8x4")

# Source constants (quantization_cute_dsl_utils.py:37-59).
SF_VEC_SIZE = 32
WARP_SIZE = 32
INV_FLOAT8_E4M3_MAX = 1.0 / 448.0
# 2T/SF (large problems): 16 elements per thread, 2 threads per SF block.
ELTS_PER_THREAD = 16
THREADS_PER_SF = 2
SF_BLOCKS_PER_WARP = 16
# 4T/SF (small problems): 8 elements per thread, 4 threads per SF block.
ELTS_PER_THREAD_SMALL = 8
THREADS_PER_SF_SMALL = 4
SF_BLOCKS_PER_WARP_SMALL = 8
MXFP8_2T_SF_THRESHOLD = 65536

_BLOCKS_PER_SM = 4
_LINEAR_WARPS = 16  # MXFP8QuantizeLinearKernel.WARPS_PER_BLOCK
_MIN_WARPS = 4
_MAX_WARPS = 32
_DEFAULT_WARPS = 16
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


def _validate(dtype: str, m: int, k: int, sf_layout: str) -> None:
    if dtype not in _DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if sf_layout not in _SF_LAYOUTS:
        raise ValueError(f"Unsupported sf_layout: {sf_layout}")
    if m < 1:
        raise ValueError(f"m={m} must be >= 1")
    if k <= 0 or k % SF_VEC_SIZE != 0:
        raise ValueError(f"k={k} outside the source dispatch domain (k % 32 != 0)")


def _use_2t(m: int, k: int) -> bool:
    """Source host dispatch (mxfp8_quantize.py:933-934)."""
    return m * (k // SF_VEC_SIZE) >= MXFP8_2T_SF_THRESHOLD


def _thread_config(use_2t: bool) -> tuple[int, int, int]:
    """(elts_per_thread, threads_per_sf, sf_blocks_per_warp)."""
    if use_2t:
        return ELTS_PER_THREAD, THREADS_PER_SF, SF_BLOCKS_PER_WARP
    return ELTS_PER_THREAD_SMALL, THREADS_PER_SF_SMALL, SF_BLOCKS_PER_WARP_SMALL


def _compute_optimal_warps(k: int, sf_blocks_per_warp: int) -> int:
    """Mirror ``_compute_optimal_warps`` (mxfp8_quantize.py:95-143)."""
    import math

    num_sf_blocks = k // SF_VEC_SIZE
    gcd_val = math.gcd(num_sf_blocks, sf_blocks_per_warp)
    warp_multiple = num_sf_blocks // gcd_val
    if warp_multiple <= _MAX_WARPS:
        warps = (_MAX_WARPS // warp_multiple) * warp_multiple
        if warps >= _MIN_WARPS:
            return warps
        warps = warp_multiple
        while warps < _MIN_WARPS:
            warps += warp_multiple
        if warps <= _MAX_WARPS:
            return warps
    return _DEFAULT_WARPS


def _padded_m(m: int, sf_layout: str) -> int:
    tile = _ROW_TILE_8x4 if sf_layout == "8x4" else _ROW_TILE_128x4
    return (m + tile - 1) // tile * tile


def _padded_sf_cols(k: int) -> int:
    return (k // SF_VEC_SIZE + 3) // 4 * 4


def _sf_numel(m: int, k: int, sf_layout: str) -> int:
    if sf_layout == "linear":
        return m * (k // SF_VEC_SIZE)
    return _padded_m(m, sf_layout) * _padded_sf_cols(k)


def _linear_launch(m: int, k: int, use_2t: bool) -> tuple[int, int, int]:
    """(grid_x, block_x, total_sf_blocks) mirroring mxfp8_quantize.py:957-967."""
    _, _, sfbpw = _thread_config(use_2t)
    sf_blocks_per_tb = _LINEAR_WARPS * sfbpw
    total_sf_blocks = m * (k // SF_VEC_SIZE)
    grid = min(
        (total_sf_blocks + sf_blocks_per_tb - 1) // sf_blocks_per_tb, _sm_count() * _BLOCKS_PER_SM
    )
    return grid, _LINEAR_WARPS * WARP_SIZE, total_sf_blocks


def _swizzled_launch(m: int, k: int, sf_layout: str, use_2t: bool) -> tuple[int, int, int]:
    """(grid_x, block_x, padded_m) mirroring mxfp8_quantize.py:936-948."""
    _, tps, sfbpw = _thread_config(use_2t)
    warps = _compute_optimal_warps(k, sfbpw)
    threads = warps * WARP_SIZE
    col_units = threads // tps
    num_sf_blocks = k // SF_VEC_SIZE
    rows_per_block = col_units // num_sf_blocks if num_sf_blocks <= col_units else 1
    padded_m = _padded_m(m, sf_layout)
    grid = min((padded_m + rows_per_block - 1) // rows_per_block, _sm_count() * _BLOCKS_PER_SM)
    return grid, threads, padded_m


def _quantize_block(in_global, out_global, row_idx, sf_col, thread_in_unit, *, dtype, use_2t, k):
    """The per-SF-block program shared by both kernels (sketch QUANTIZE_BLOCK).

    Loads ELTS elements at (row_idx, sf_col*32 + thread_in_unit*ELTS), quantizes
    against the 2/4-lane reduced UE8M0 scale, stores the 8-byte group(s), and
    returns scale_ue8m0_u32 for the caller's conditional SF store.
    """
    elts, _, _ = _thread_config(use_2t)
    elem_idx = sf_col * SF_VEC_SIZE + thread_in_unit * elts
    in_off = K.cast(row_idx, "int64") * k + elem_idx
    v = ld_global_v4_u32(K.address_of(in_global[in_off]))
    if use_2t:
        vh = ld_global_v4_u32(K.address_of(in_global[in_off + 8]))
        words = [v[i] for i in range(4)] + [vh[i] for i in range(4)]
        global_max = reduce_max_2threads(pair_max_to_f32(absmax_8(words, dtype), dtype))
    else:
        words = [v[i] for i in range(4)]
        global_max = reduce_max_4threads(pair_max_to_f32(absmax_4(words, dtype), dtype))
    scale_ue8m0_u32 = float_to_ue8m0(mul_f32(global_max, K.float32(INV_FLOAT8_E4M3_MAX)))
    inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)
    out_off = K.cast(row_idx, "int64") * k + elem_idx
    st_global_u64(K.address_of(out_global[out_off]), fp8x8_scaled(words[:4], inv_scale, dtype))
    if use_2t:
        st_global_u64(
            K.address_of(out_global[out_off + 8]), fp8x8_scaled(words[4:], inv_scale, dtype)
        )
    return scale_ue8m0_u32


def _materialize(value):
    local = K.alloc_local((1,), str(value.ty.dtype))
    K.assign(local[0], value)
    return local[0]


def get_kernel(
    dtype: str, m: int, k: int, sf_layout: str = "linear", enable_pdl: bool = False, **kwargs
):
    """Return the TIRx specialization for one (dtype, m, k, sf_layout) config."""
    _validate(dtype, m, k, sf_layout)
    use_2t = _use_2t(m, k)
    elts, tps, sfbpw = _thread_config(use_2t)
    nsb = k // SF_VEC_SIZE
    pad_cols = _padded_sf_cols(k)

    def sf_offset(row, col):
        if sf_layout == "8x4":
            return sf_offset_8x4(row, col, pad_cols)
        return sf_offset_128x4(row, col, pad_cols)

    if sf_layout == "linear":
        grid_x, block_x, _ = _linear_launch(m, k, use_2t)
        sfbpt = _LINEAR_WARPS * sfbpw

        @K.kernel(
            warps=block_x // 32, arch="sm_100a", min_blocks_per_sm=_BLOCKS_PER_SM, grid=grid_x
        )
        def mxfp8_quantize_linear(
            in_global: K.gptr[dtype],
            out_global: K.gptr[K.u8],
            sf_out: K.gptr[K.u8],
            total_sf: K.i32,
        ):
            bx = K.cta_id()
            tx = K.thread_id()

            if enable_pdl:
                K.ptx.griddepcontrol.wait()

            warp_idx = K.truncdiv(tx, K.int32(WARP_SIZE))
            lane_idx = K.truncmod(tx, K.int32(WARP_SIZE))
            sf_idx_in_warp = K.truncdiv(lane_idx, K.int32(tps))
            thread_in_sf = K.truncmod(lane_idx, K.int32(tps))

            # Grid-stride loop over flat SF blocks (mxfp8_quantize.py:248-326).
            sf_idx = K.alloc_local([1], "int32")
            K.assign(sf_idx[0], bx * sfbpt + warp_idx * sfbpw + sf_idx_in_warp)
            with K.While(sf_idx[0] < total_sf):
                row_idx = K.truncdiv(sf_idx[0], K.int32(nsb))
                col_idx = K.truncmod(sf_idx[0], K.int32(nsb))
                scale_ue8m0_u32 = _quantize_block(
                    in_global,
                    out_global,
                    row_idx,
                    col_idx,
                    thread_in_sf,
                    dtype=dtype,
                    use_2t=use_2t,
                    k=k,
                )
                with K.If(thread_in_sf == 0), K.Then():
                    st_global_u8(K.address_of(sf_out[sf_idx[0]]), K.cast(scale_ue8m0_u32, "uint8"))
                K.assign(sf_idx[0], sf_idx[0] + grid_x * sfbpt)

            if enable_pdl:
                K.ptx.griddepcontrol.launch_dependents()

        return mxfp8_quantize_linear.func

    grid_x, block_x, padded_m = _swizzled_launch(m, k, sf_layout, use_2t)
    threads_per_row = nsb * tps
    col_units_per_block = block_x // tps
    needs_col_loop = nsb > col_units_per_block
    rows_per_block = 1 if needs_col_loop else col_units_per_block // nsb

    @K.kernel(warps=block_x // 32, arch="sm_100a", min_blocks_per_sm=_BLOCKS_PER_SM, grid=grid_x)
    def mxfp8_quantize_swizzled(
        in_global: K.gptr[dtype],
        out_global: K.gptr[K.u8],
        sf_out: K.gptr[K.u8],
        m_rows: K.i32,
        padded_rows: K.i32,
    ):
        bx = K.cta_id()
        tx = K.thread_id()

        if enable_pdl:
            K.ptx.griddepcontrol.wait()

        if needs_col_loop:
            # Large K: one row per block iteration with a column loop
            # (mxfp8_quantize.py:469-589).
            col_unit_idx = K.truncdiv(tx, K.int32(tps))
            thread_in_unit = K.truncmod(tx, K.int32(tps))

            row_idx = K.alloc_local([1], "int32")
            K.assign(row_idx[0], bx)
            with K.While(row_idx[0] < padded_rows):
                with K.If(row_idx[0] >= m_rows):
                    with K.Then():
                        # Padding row: zero out scale factors only (:481-489).
                        sc_pad = K.alloc_local([1], "int32")
                        K.assign(sc_pad[0], col_unit_idx)
                        with K.While(sc_pad[0] < pad_cols):
                            with K.If(thread_in_unit == 0), K.Then():
                                st_global_u8(
                                    K.address_of(sf_out[sf_offset(row_idx[0], sc_pad[0])]),
                                    K.uint8(0),
                                )
                            K.assign(sc_pad[0], sc_pad[0] + col_units_per_block)
                    with K.Else():
                        # Data row: quantize each SF block in the column loop.
                        sc = K.alloc_local([1], "int32")
                        K.assign(sc[0], col_unit_idx)
                        with K.While(sc[0] < nsb):
                            scale_ue8m0_u32 = _quantize_block(
                                in_global,
                                out_global,
                                row_idx[0],
                                sc[0],
                                thread_in_unit,
                                dtype=dtype,
                                use_2t=use_2t,
                                k=k,
                            )
                            with K.If(thread_in_unit == 0), K.Then():
                                st_global_u8(
                                    K.address_of(sf_out[sf_offset(row_idx[0], sc[0])]),
                                    K.cast(scale_ue8m0_u32, "uint8"),
                                )
                            K.assign(sc[0], sc[0] + col_units_per_block)
                        # Padding SF columns of a data row (:580-587).
                        sc_tail = K.alloc_local([1], "int32")
                        K.assign(sc_tail[0], nsb + col_unit_idx)
                        with K.While(sc_tail[0] < pad_cols):
                            with K.If(thread_in_unit == 0), K.Then():
                                st_global_u8(
                                    K.address_of(sf_out[sf_offset(row_idx[0], sc_tail[0])]),
                                    K.uint8(0),
                                )
                            K.assign(sc_tail[0], sc_tail[0] + col_units_per_block)
                K.assign(row_idx[0], row_idx[0] + grid_x)
        else:
            # Small K: multi-row processing (mxfp8_quantize.py:590-727).
            row_in_block = _materialize(K.truncdiv(tx, K.int32(threads_per_row)))
            local_tidx = _materialize(K.truncmod(tx, K.int32(threads_per_row)))
            sf_col_idx = _materialize(K.truncdiv(local_tidx, K.int32(tps)))
            thread_in_unit = _materialize(K.truncmod(local_tidx, K.int32(tps)))

            row_batch_idx = K.alloc_local([1], "int32")
            row_idx2 = K.alloc_local([1], "int32")
            K.assign(row_batch_idx[0], bx)
            K.assign(row_idx2[0], row_batch_idx[0] * rows_per_block + row_in_block)
            with K.While(row_batch_idx[0] * rows_per_block < padded_rows):
                with K.If(row_idx2[0] < padded_rows), K.Then():
                    with K.If(row_idx2[0] >= m_rows):
                        with K.Then():
                            # Padding row: zero ALL padded SF columns; the stride is
                            # num_sf_blocks_per_row (:609-620).
                            with K.If(thread_in_unit == 0), K.Then():
                                pad_col = K.alloc_local([1], "int32")
                                K.assign(pad_col[0], sf_col_idx)
                                with K.While(pad_col[0] < pad_cols):
                                    st_global_u8(
                                        K.address_of(sf_out[sf_offset(row_idx2[0], pad_col[0])]),
                                        K.uint8(0),
                                    )
                                    K.assign(pad_col[0], pad_col[0] + nsb)
                        with K.Else():
                            with K.If(sf_col_idx < nsb), K.Then():
                                scale_ue8m0_u32 = _quantize_block(
                                    in_global,
                                    out_global,
                                    row_idx2[0],
                                    sf_col_idx,
                                    thread_in_unit,
                                    dtype=dtype,
                                    use_2t=use_2t,
                                    k=k,
                                )
                                with K.If(thread_in_unit == 0), K.Then():
                                    st_global_u8(
                                        K.address_of(sf_out[sf_offset(row_idx2[0], sf_col_idx)]),
                                        K.cast(scale_ue8m0_u32, "uint8"),
                                    )
                            # Padding SF columns of a data row (:711-723).
                            if pad_cols != nsb:
                                with K.If(thread_in_unit == 0), K.Then():
                                    pad_col2 = K.alloc_local([1], "int32")
                                    K.assign(pad_col2[0], nsb + sf_col_idx)
                                    with K.While(pad_col2[0] < pad_cols):
                                        st_global_u8(
                                            K.address_of(
                                                sf_out[sf_offset(row_idx2[0], pad_col2[0])]
                                            ),
                                            K.uint8(0),
                                        )
                                        K.assign(pad_col2[0], pad_col2[0] + nsb)
                K.assign(row_batch_idx[0], row_batch_idx[0] + grid_x)
                K.assign(row_idx2[0], row_batch_idx[0] * rows_per_block + row_in_block)

        if enable_pdl:
            K.ptx.griddepcontrol.launch_dependents()

    return mxfp8_quantize_swizzled.func


def prepare_data(
    dtype: str, m: int, k: int, sf_layout: str = "linear", enable_pdl: bool = False, **kwargs
):
    """Create the logical input: a [m, k] fp16/bf16 tensor."""
    import torch

    _validate(dtype, m, k, sf_layout)
    torch.manual_seed(42)
    a = torch.randn(m, k, dtype=_torch_dtype(dtype), device="cuda")
    return (a,)


def _alloc_outputs(m: int, k: int, sf_layout: str):
    import torch

    out = torch.empty(m, k, dtype=torch.uint8, device="cuda")
    sf = torch.empty(_sf_numel(m, k, sf_layout), dtype=torch.uint8, device="cuda")
    return out, sf


def _sf_layout_enum(sf_layout: str):
    from flashinfer.tllm_enums import SfLayout

    return {
        "linear": SfLayout.layout_linear,
        "128x4": SfLayout.layout_128x4,
        "8x4": SfLayout.layout_8x4,
    }[sf_layout]


def _run_reference(a, sf_layout: str, enable_pdl: bool):
    """Run the FlashInfer CuTe-DSL source wrapper (allocates its own outputs)."""
    from flashinfer.quantization import mxfp8_quantize

    return mxfp8_quantize(
        a, backend="cute-dsl", sf_swizzle_layout=_sf_layout_enum(sf_layout), enable_pdl=enable_pdl
    )


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(
    dtype: str, m: int, k: int, sf_layout: str = "linear", enable_pdl: bool = False, **kwargs
):
    """Compile, launch, and validate one config against the flashinfer source."""
    import torch

    from tirx_kernels.runner import compile_kernel

    (a,) = prepare_data(dtype=dtype, m=m, k=k, sf_layout=sf_layout, enable_pdl=enable_pdl)
    kernel = get_kernel(dtype=dtype, m=m, k=k, sf_layout=sf_layout, enable_pdl=enable_pdl)
    ex = compile_kernel(kernel)
    out_tirx, sf_tirx = _alloc_outputs(m, k, sf_layout)
    if sf_layout == "linear":
        ex(a.view(-1), out_tirx.view(-1), sf_tirx, m * (k // SF_VEC_SIZE))
    else:
        ex(a.view(-1), out_tirx.view(-1), sf_tirx, m, _padded_m(m, sf_layout))
    torch.cuda.synchronize()

    ref_fp8, ref_sf = _run_reference(a, sf_layout, enable_pdl)
    torch.testing.assert_close(out_tirx, ref_fp8.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(sf_tirx, ref_sf.view(-1), rtol=0, atol=0)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs):
    """Benchmark the TIRx port against the CuTe-DSL source (kernel-only)."""
    config = dict(prepared["config"])
    dtype = config.pop("dtype")
    m = config.pop("m")
    k = config.pop("k")
    sf_layout = config.pop("sf_layout")
    enable_pdl = config.pop("enable_pdl")
    config.update(kwargs)
    kwargs = config
    executable = prepared["executable"]

    (a,) = prepare_data(dtype=dtype, m=m, k=k, sf_layout=sf_layout, enable_pdl=enable_pdl)
    ex = executable
    out_tirx, sf_tirx = _alloc_outputs(m, k, sf_layout)

    if sf_layout == "linear":
        total_sf = m * (k // SF_VEC_SIZE)

        def tirx_launch():
            ex(a.view(-1), out_tirx.view(-1), sf_tirx, total_sf)
    else:
        padded_m = _padded_m(m, sf_layout)

        def tirx_launch():
            ex(a.view(-1), out_tirx.view(-1), sf_tirx, m, padded_m)

    def build_reference():
        # Bypass the allocating public wrapper: call the cached compiled source
        # kernel directly with preallocated outputs (kernel-only timing).
        from flashinfer.quantization.kernels.mxfp8_quantize import (
            SF_LAYOUT_8x4,
            SF_LAYOUT_128x4,
            _get_compiled_kernel_mxfp8_linear,
            _get_compiled_kernel_mxfp8_swizzled,
        )

        is_bf16 = dtype == "bfloat16"
        use_2t = _use_2t(m, k)
        out_ref, sf_ref = _alloc_outputs(m, k, sf_layout)
        if sf_layout == "linear":
            kernel_fn, _ = _get_compiled_kernel_mxfp8_linear(is_bf16, k, enable_pdl, use_2t)
            total_sf = m * (k // SF_VEC_SIZE)
            grid, _, _ = _linear_launch(m, k, use_2t)
            return lambda: kernel_fn(a, out_ref, sf_ref, total_sf, grid)
        layout_code = SF_LAYOUT_8x4 if sf_layout == "8x4" else SF_LAYOUT_128x4
        kernel_fn, _ = _get_compiled_kernel_mxfp8_swizzled(
            is_bf16, k, enable_pdl, use_2t, layout_code
        )
        padded_m = _padded_m(m, sf_layout)
        grid, _, _ = _swizzled_launch(m, k, sf_layout, use_2t)
        return lambda: kernel_fn(a, out_ref, sf_ref, m, padded_m, grid)

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
    sf_layout: str = "linear",
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
        dtype=dtype, m=m, k=k, sf_layout=sf_layout, enable_pdl=enable_pdl, **config
    )
    return prepared.run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


def _cfg(dtype, m, k, sf_layout="linear", enable_pdl=False):
    dt = {"float16": "fp16", "bfloat16": "bf16"}[dtype]
    pdl = "_pdl" if enable_pdl else ""
    return {
        "label": f"{dt}_{sf_layout}_m{m}_k{k}{pdl}",
        "dtype": dtype,
        "m": m,
        "k": k,
        "sf_layout": sf_layout,
        "enable_pdl": enable_pdl,
    }


# Correctness matrix.  Covers: both dtypes; linear/128x4/8x4 SF layouts; the
# 2T/4T host dispatch boundary (m * k/32 >= 65536); the swizzled multi-row vs
# needs_col_loop compile-time split (fallback warps when K/32 and SF-blocks-per-
# warp are coprime); padding-row and padding-column zero-fill paths (m % 128,
# m % 8, k/32 % 4 != 0); minimal shapes; the PDL instruction variant.
CONFIGS = [
    _cfg("float16", 1, 32),  # minimal
    _cfg("float16", 128, 1024),  # 4T linear
    _cfg("bfloat16", 128, 1024),  # 4T linear
    _cfg("float16", 512, 4096),  # 2T linear (threshold boundary)
    _cfg("bfloat16", 512, 4096),  # 2T linear
    _cfg("float16", 13, 1056),  # odd m, k/32 = 33 (no linear padding)
    _cfg("float16", 128, 1024, "128x4"),  # 4T multi-row
    _cfg("bfloat16", 128, 1024, "128x4"),
    _cfg("float16", 120, 1024, "128x4"),  # row padding 120 -> 128
    _cfg("float16", 128, 1056, "128x4"),  # col padding 33 -> 36
    _cfg("float16", 512, 4096, "128x4"),  # 2T multi-row
    _cfg("bfloat16", 512, 4096, "128x4"),
    _cfg("float16", 256, 8256, "128x4"),  # 2T needs_col_loop (258 SF/row)
    _cfg("float16", 64, 8448, "128x4"),  # 4T needs_col_loop (264 SF/row)
    _cfg("float16", 13, 1024, "8x4"),  # 8x4 row padding 13 -> 16
    _cfg("bfloat16", 128, 1024, "8x4"),
    _cfg("float16", 512, 4096, "linear", True),  # PDL instruction variant
]

# Benchmark sweep: 2T and 4T regimes, linear and 128x4, realistic LLM shapes.
BENCH_CONFIGS = [
    _cfg("float16", 4096, 4096),
    _cfg("bfloat16", 4096, 4096),
    _cfg("float16", 4096, 4096, "128x4"),
    _cfg("bfloat16", 4096, 4096, "128x4"),
    _cfg("float16", 16384, 7168),
    _cfg("float16", 16384, 7168, "128x4"),
    _cfg("float16", 1024, 2048),
    _cfg("float16", 1024, 2048, "128x4"),
    _cfg("float16", 128, 1024),  # 4T regime
    _cfg("float16", 128, 1024, "128x4"),
]
