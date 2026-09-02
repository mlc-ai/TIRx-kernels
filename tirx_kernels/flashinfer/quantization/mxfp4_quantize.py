# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer CuTe-DSL ``mxfp4_quantize`` port.

Ports ``MXFP4QuantizeLinearKernel`` / ``MXFP4QuantizeSwizzledKernel``
(``flashinfer/quantization/kernels/mxfp4_quantize.py``), the SM100 CuTe-DSL
kernels behind ``flashinfer.quantization.mxfp4_quantize(backend="cute-dsl")``.
The 1T/SF specialization: each thread owns one 32-element SF block — four
128-bit loads, a 16-word half2/bf16x2 absmax tree, no shuffle reduction, UE8M0
block scale via ``rcp.approx.ftz(6.0)``, and 32 e2m1 values packed into two
``st.global.u64`` stores.

In-scope specialization: fp16/bf16 inputs, linear + swizzled 128x4/8x4 SF
layouts, 1T/SF thread configuration.  On low-SM devices such as Thor the
FlashInfer reference dispatches its 4T/SF implementation; both paths implement
the same quantization contract, so it remains the correctness oracle while
TIRx runs the existing 1T/SF kernel.  ``enable_pdl=False`` (the griddepcontrol pair is
ported behind the same compile-time knob; TVM launches do not carry the PDL
launch attribute, so PDL stays off for test/bench parity on both sides).

The implementation structure follows the reviewer-approved sketch
``.agents/sketch/flashinfer/quantization/mxfp4_quantize.md``; shared instruction-level helpers live in
``tirx_kernels/flashinfer/utils/fp_quant.py``.
"""

from typing import Any

import tirx_kernels.kern as K
from tirx_kernels.flashinfer.utils.fp_quant_kern import (
    absmax_8,
    cvt_e2m1x8,
    float2_scaled,
    float_to_ue8m0,
    hmax2,
    ld_global_v4_u32,
    mul_f32,
    pack_u32x2_to_u64,
    pair_max_to_f32,
    rcp_approx_ftz,
    sf_offset_8x4,
    sf_offset_128x4,
    st_global_u8,
    st_global_u64,
    ue8m0_to_inv_scale,
)
from tirx_kernels.runner import bench

KERNEL_META = {
    "name": "mxfp4_quantize",
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
_SF_LAYOUTS = ("linear", "128x4", "8x4")

# Source constants (mxfp4_quantize.py:76-96, quantization_cute_dsl_utils.py).
MXFP4_SF_VEC_SIZE = 32
WARP_SIZE = 32
_BLOCKS_PER_SM = 4
_LINEAR_WARPS = 16  # _LINEAR_WARPS_PER_BLOCK
_LINEAR_SF_BLOCKS_PER_TB = 512  # _LINEAR_SF_BLOCKS_PER_TB (1T/SF)
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


def _validate(dtype: str, m: int, k: int, sf_layout: str) -> None:
    if dtype not in _DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if sf_layout not in _SF_LAYOUTS:
        raise ValueError(f"Unsupported sf_layout: {sf_layout}")
    if m < 1:
        raise ValueError(f"m={m} must be >= 1")
    if k <= 0 or k % MXFP4_SF_VEC_SIZE != 0:
        raise ValueError(f"k={k} outside the source dispatch domain (k % 32 != 0)")


def _compute_optimal_threads(k: int) -> int:
    """Mirror ``_compute_optimal_threads`` 1T/SF (mxfp4_quantize.py:115-155)."""
    threads_per_row = k // MXFP4_SF_VEC_SIZE
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
    return (k // MXFP4_SF_VEC_SIZE + 3) // 4 * 4


def _sf_numel(m: int, k: int, sf_layout: str) -> int:
    if sf_layout == "linear":
        return m * (k // MXFP4_SF_VEC_SIZE)
    return _padded_m(m, sf_layout) * _padded_sf_cols(k)


def _linear_launch(m: int, k: int) -> tuple[int, int, int]:
    """(grid_x, block_x, total_sf_blocks) mirroring mxfp4_quantize.py:963-971."""
    total_sf_blocks = m * (k // MXFP4_SF_VEC_SIZE)
    grid = min(
        (total_sf_blocks + _LINEAR_SF_BLOCKS_PER_TB - 1) // _LINEAR_SF_BLOCKS_PER_TB,
        _sm_count() * _BLOCKS_PER_SM,
    )
    return grid, _LINEAR_WARPS * WARP_SIZE, total_sf_blocks


def _swizzled_launch(m: int, k: int, sf_layout: str) -> tuple[int, int, int]:
    """(grid_x, block_x, padded_m) mirroring mxfp4_quantize.py:979-986."""
    threads = _compute_optimal_threads(k)
    nsb = k // MXFP4_SF_VEC_SIZE
    rows_per_block = threads // nsb if nsb <= threads else 1
    padded_m = _padded_m(m, sf_layout)
    grid = min((padded_m + rows_per_block - 1) // rows_per_block, _sm_count() * _BLOCKS_PER_SM)
    return grid, threads, padded_m


def _process_block(in_global, row_idx, col_idx, *, dtype, k):
    """process_mxfp4_block_half/bfloat (utils:765/:839), 1T/SF, no stores.

    Returns (scale_ue8m0_u32, packed64_0, packed64_1); the caller stores the
    SF byte first, then the two 8-byte output groups (source order).
    """
    elem_base = col_idx * MXFP4_SF_VEC_SIZE
    base = K.cast(row_idx, "int64") * k + elem_base
    v0 = ld_global_v4_u32(K.address_of(in_global[base]))
    v1 = ld_global_v4_u32(K.address_of(in_global[base + 8]))
    v2 = ld_global_v4_u32(K.address_of(in_global[base + 16]))
    v3 = ld_global_v4_u32(K.address_of(in_global[base + 24]))
    words = [v0[i] for i in range(4)] + [v1[i] for i in range(4)]
    words += [v2[i] for i in range(4)] + [v3[i] for i in range(4)]

    max_first = absmax_8(words[0:8], dtype)
    max_second = absmax_8(words[8:16], dtype)
    block_max = pair_max_to_f32(hmax2(max_first, max_second, dtype), dtype)

    scale_ue8m0_u32 = float_to_ue8m0(mul_f32(block_max, rcp_approx_ftz(K.float32(6.0))))
    inv_scale = ue8m0_to_inv_scale(scale_ue8m0_u32)

    s = []
    for i in range(16):
        lo, hi = float2_scaled(words[i], inv_scale, dtype)
        s.append(lo)
        s.append(hi)
    packed = [cvt_e2m1x8(s[8 * j : 8 * j + 8]) for j in range(4)]
    packed64_0 = pack_u32x2_to_u64(packed[0], packed[1])
    packed64_1 = pack_u32x2_to_u64(packed[2], packed[3])
    return scale_ue8m0_u32, packed64_0, packed64_1


def _materialize(value):
    local = K.local_scalar(str(value.ty.dtype), init=value)
    return local


def get_kernel(
    dtype: str, m: int, k: int, sf_layout: str = "128x4", enable_pdl: bool = False, **kwargs
):
    """Return the TIRx specialization for one (dtype, m, k, sf_layout) config."""
    _validate(dtype, m, k, sf_layout)
    nsb = k // MXFP4_SF_VEC_SIZE
    pad_cols = _padded_sf_cols(k)

    def sf_offset(row, col):
        if sf_layout == "8x4":
            return sf_offset_8x4(row, col, pad_cols)
        return sf_offset_128x4(row, col, pad_cols)

    if sf_layout == "linear":
        grid_x, block_x, total_sf_blocks = _linear_launch(m, k)

        @K.kernel(warps=(block_x + 31) // 32, arch="sm_100a", min_blocks_per_sm=2, grid=grid_x)
        def mxfp4_quantize_linear(
            in_global: K.gptr[dtype],
            out_global: K.gptr[K.u8],
            sf_out: K.gptr[K.u8],
            m_rows: K.i32,
            total_sf: K.i32,
        ):
            bx = K.cta_id()
            tx = K.thread_id()

            if enable_pdl:
                K.ptx.griddepcontrol.wait()

            # Flat SF-block grid-stride loop (mxfp4_quantize.py:333-375, 1T/SF).
            sf_idx = K.local_scalar("int32", init=bx * _LINEAR_SF_BLOCKS_PER_TB + tx)
            with K.While(sf_idx < total_sf):
                row_idx = K.truncdiv(sf_idx, K.int32(nsb))
                col_idx = K.truncmod(sf_idx, K.int32(nsb))
                scale_ue8m0_u32, packed64_0, packed64_1 = _process_block(
                    in_global, row_idx, col_idx, dtype=dtype, k=k
                )
                # Source order: SF byte store, then the two output stores.
                st_global_u8(K.address_of(sf_out[sf_idx]), K.cast(scale_ue8m0_u32, "uint8"))
                out_off = K.cast(row_idx, "int64") * (k // 2) + col_idx * 16
                st_global_u64(K.address_of(out_global[out_off]), packed64_0)
                st_global_u64(K.address_of(out_global[out_off + 8]), packed64_1)
                K.assign(sf_idx, sf_idx + grid_x * _LINEAR_SF_BLOCKS_PER_TB)

            if enable_pdl:
                K.ptx.griddepcontrol.launch_dependents()

        return mxfp4_quantize_linear.func

    grid_x, block_x, padded_m = _swizzled_launch(m, k, sf_layout)
    needs_col_loop = nsb > block_x
    rows_per_block = 1 if needs_col_loop else block_x // nsb

    @K.kernel(warps=(block_x + 31) // 32, arch="sm_100a", min_blocks_per_sm=2, grid=grid_x)
    def mxfp4_quantize_swizzled(
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

        def body():
            # Large K (K/32 > 512): one row per block iteration with a column
            # loop; 1T/SF -> col_unit == tidx (mxfp4_quantize.py:506-642).
            row_idx = K.local_scalar("int32", init=bx)
            with K.While(row_idx < padded_rows):
                with K.If(row_idx >= m_rows):
                    with K.Then():
                        # Padding row: zero-fill by ALL threads, stride block_x.
                        sc_pad = K.local_scalar("int32", init=tx)
                        with K.While(sc_pad < pad_cols):
                            st_global_u8(
                                K.address_of(sf_out[sf_offset(row_idx, sc_pad)]), K.uint8(0)
                            )
                            K.assign(sc_pad, sc_pad + block_x)
                    with K.Else():
                        sc = K.local_scalar("int32", init=tx)
                        with K.While(sc < nsb):
                            scale_ue8m0_u32, packed64_0, packed64_1 = _process_block(
                                in_global, row_idx, sc, dtype=dtype, k=k
                            )
                            # Source order: SF byte store, then the output stores.
                            st_global_u8(
                                K.address_of(sf_out[sf_offset(row_idx, sc)]),
                                K.cast(scale_ue8m0_u32, "uint8"),
                            )
                            out_off = K.cast(row_idx, "int64") * (k // 2) + sc * 16
                            st_global_u64(K.address_of(out_global[out_off]), packed64_0)
                            st_global_u64(K.address_of(out_global[out_off + 8]), packed64_1)
                            K.assign(sc, sc + block_x)
                        # Padding SF columns of a data row (:634-640).
                        sc_tail = K.local_scalar("int32", init=nsb + tx)
                        with K.While(sc_tail < pad_cols):
                            st_global_u8(
                                K.address_of(sf_out[sf_offset(row_idx, sc_tail)]), K.uint8(0)
                            )
                            K.assign(sc_tail, sc_tail + block_x)
                K.assign(row_idx, row_idx + grid_x)

        def small_body():
            # Small K: multi-row processing; 1T/SF -> threads_per_row == nsb,
            # thread_in_sf == 0 always (mxfp4_quantize.py:643-794).
            row_in_block = _materialize(K.truncdiv(tx, K.int32(nsb)))
            sf_idx_in_row = _materialize(K.truncmod(tx, K.int32(nsb)))

            row_batch_idx = K.local_scalar("int32")
            row_idx2 = K.local_scalar("int32")
            K.assign(row_batch_idx, bx)
            K.assign(row_idx2, row_batch_idx * rows_per_block + row_in_block)
            with K.While(row_batch_idx * rows_per_block < padded_rows):
                with K.If(row_idx2 < padded_rows), K.Then():
                    with K.If(row_idx2 >= m_rows):
                        with K.Then():
                            # Padding row: zero ALL padded SF columns; stride is
                            # threads_per_row == nsb (:676-682).
                            local_sf = K.local_scalar("int32", init=sf_idx_in_row)
                            with K.While(local_sf < pad_cols):
                                st_global_u8(
                                    K.address_of(sf_out[sf_offset(row_idx2, local_sf)]), K.uint8(0)
                                )
                                K.assign(local_sf, local_sf + nsb)
                        with K.Else():
                            with K.If(sf_idx_in_row < nsb), K.Then():
                                (scale_ue8m0_u32, packed64_0, packed64_1) = _process_block(
                                    in_global, row_idx2, sf_idx_in_row, dtype=dtype, k=k
                                )
                                # Source order: SF byte store, then output stores.
                                st_global_u8(
                                    K.address_of(sf_out[sf_offset(row_idx2, sf_idx_in_row)]),
                                    K.cast(scale_ue8m0_u32, "uint8"),
                                )
                                out_off = K.cast(row_idx2, "int64") * (k // 2) + (
                                    sf_idx_in_row * 16
                                )
                                st_global_u64(K.address_of(out_global[out_off]), packed64_0)
                                st_global_u64(K.address_of(out_global[out_off + 8]), packed64_1)
                            # Padding SF columns of a data row (:782-791).
                            if pad_cols != nsb:
                                pad_col = K.local_scalar("int32", init=nsb + sf_idx_in_row)
                                with K.While(pad_col < pad_cols):
                                    st_global_u8(
                                        K.address_of(sf_out[sf_offset(row_idx2, pad_col)]),
                                        K.uint8(0),
                                    )
                                    K.assign(pad_col, pad_col + nsb)
                K.assign(row_batch_idx, row_batch_idx + grid_x)
                K.assign(row_idx2, row_batch_idx * rows_per_block + row_in_block)

        if block_x % 32:
            with K.If(tx < block_x), K.Then():
                body() if needs_col_loop else small_body()
        elif needs_col_loop:
            body()
        else:
            small_body()

        if enable_pdl:
            K.ptx.griddepcontrol.launch_dependents()

    return mxfp4_quantize_swizzled.func


def prepare_data(
    dtype: str, m: int, k: int, sf_layout: str = "128x4", enable_pdl: bool = False, **kwargs
):
    """Create the logical input: a [m, k] fp16/bf16 tensor."""
    import torch

    _validate(dtype, m, k, sf_layout)
    torch.manual_seed(42)
    a = torch.randn(m, k, dtype=_torch_dtype(dtype), device="cuda")
    return (a,)


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


def _run_reference(a, sf_layout: str, enable_pdl: bool):
    """Run the FlashInfer CuTe-DSL source wrapper (allocates its own outputs)."""
    from flashinfer.quantization import mxfp4_quantize

    return mxfp4_quantize(
        a, backend="cute-dsl", sfLayout=_sf_layout_enum(sf_layout), enable_pdl=enable_pdl
    )


def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    state = {"config": dict(kwargs), "executable": compile_kernel(get_kernel(**kwargs))}
    return prepared_gpu_benchmark(run_gpu, state)


def run_test(
    dtype: str, m: int, k: int, sf_layout: str = "128x4", enable_pdl: bool = False, **kwargs
):
    """Compile, launch, and validate one config against the flashinfer source."""
    import torch

    from tirx_kernels.runner import compile_kernel

    (a,) = prepare_data(dtype=dtype, m=m, k=k, sf_layout=sf_layout, enable_pdl=enable_pdl)
    kernel = get_kernel(dtype=dtype, m=m, k=k, sf_layout=sf_layout, enable_pdl=enable_pdl)
    ex = compile_kernel(kernel)
    out_tirx, sf_tirx = _alloc_outputs(m, k, sf_layout)
    if sf_layout == "linear":
        ex(a.view(-1), out_tirx.view(-1), sf_tirx, m, m * (k // MXFP4_SF_VEC_SIZE))
    else:
        ex(a.view(-1), out_tirx.view(-1), sf_tirx, m, _padded_m(m, sf_layout))
    torch.cuda.synchronize()

    ref_fp4, ref_sf = _run_reference(a, sf_layout, enable_pdl)
    torch.testing.assert_close(out_tirx, ref_fp4, rtol=0, atol=0)
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
        total_sf = m * (k // MXFP4_SF_VEC_SIZE)

        def tirx_launch():
            ex(a.view(-1), out_tirx.view(-1), sf_tirx, m, total_sf)
    else:
        padded_m = _padded_m(m, sf_layout)

        def tirx_launch():
            ex(a.view(-1), out_tirx.view(-1), sf_tirx, m, padded_m)

    def build_reference():
        # Bypass the allocating public wrapper: call the cached compiled source
        # kernel directly with preallocated outputs (kernel-only timing).
        from flashinfer.quantization.kernels.mxfp4_quantize import (
            SF_LAYOUT_LINEAR,
            SF_LAYOUT_8x4,
            SF_LAYOUT_128x4,
            _get_compiled_kernel_mxfp4,
        )

        is_bf16 = dtype == "bfloat16"
        layout_code = {"linear": SF_LAYOUT_LINEAR, "128x4": SF_LAYOUT_128x4, "8x4": SF_LAYOUT_8x4}[
            sf_layout
        ]
        kernel_fn, _ = _get_compiled_kernel_mxfp4(is_bf16, k, layout_code, enable_pdl, False)
        out_ref, sf_ref = _alloc_outputs(m, k, sf_layout)
        if sf_layout == "linear":
            total_sf = m * (k // MXFP4_SF_VEC_SIZE)
            grid, _, _ = _linear_launch(m, k)
            return lambda: kernel_fn(a, out_ref, sf_ref, m, total_sf, grid)
        padded_m = _padded_m(m, sf_layout)
        grid, _, _ = _swizzled_launch(m, k, sf_layout)
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
    sf_layout: str = "128x4",
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


def _cfg(dtype, m, k, sf_layout="128x4", enable_pdl=False):
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
# swizzled multi-row vs needs_col_loop compile-time split (K/32 > 512);
# padding-row and padding-column zero-fill paths (m % 128, m % 8,
# k/32 % 4 != 0); minimal shapes; the PDL instruction variant.  All 1T/SF
# The TIRx implementation is 1T/SF; a low-SM reference may use 4T/SF.
CONFIGS = [
    _cfg("float16", 1, 32, "linear"),  # minimal
    _cfg("float16", 128, 1024, "linear"),
    _cfg("bfloat16", 128, 1024, "linear"),
    _cfg("float16", 512, 4096, "linear"),
    _cfg("bfloat16", 512, 4096, "linear"),
    _cfg("float16", 13, 1056, "linear"),  # odd m, k/32 = 33
    _cfg("float16", 128, 1024, "128x4"),  # multi-row (threads 512, rpb 16)
    _cfg("bfloat16", 128, 1024, "128x4"),
    _cfg("float16", 120, 1024, "128x4"),  # row padding 120 -> 128
    _cfg("float16", 128, 1056, "128x4"),  # col padding 33 -> 36, threads 495
    _cfg("float16", 512, 4096, "128x4"),  # multi-row (threads 512, rpb 4)
    _cfg("bfloat16", 512, 4096, "128x4"),
    _cfg("float16", 64, 16544, "128x4"),  # needs_col_loop (517 SF/row)
    _cfg("float16", 64, 16416, "128x4"),  # col loop + col padding (513 -> 516)
    _cfg("float16", 13, 1024, "8x4"),  # 8x4 row padding 13 -> 16
    _cfg("bfloat16", 128, 1024, "8x4"),
    _cfg("float16", 512, 4096, "linear", True),  # PDL instruction variant
    _cfg("float16", 512, 4096, "128x4", True),
]

# Benchmark sweep: linear and 128x4, realistic LLM shapes.
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
