# This file is a TIRx port of code from FlashInfer
# (https://github.com/flashinfer-ai/flashinfer @ f2e04400), Copyright (c) 2024 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""FlashInfer stable top-k value-sort port.

Ports ``StableSortTopKByValueKernel`` (``include/flashinfer/topk.cuh:3090-3133``)
and its launcher ``StableSortTopKByValue`` (``:3135-3168``) -- the ``sorted_output``
epilogue of the top-k pipeline.  ``TopKDispatch`` appends it after **both** the
filtered and the radix branch (``:3470-3472``), so it is the one kernel in the
selection path that neither radix port nor the filtered port covered; all three
passed ``sorted_output=False``.

The kernel re-sorts each row's already-selected ``(values, indices)`` pairs by
value, descending, **in place**: one CTA per row, a blocked load into registers,
one ``cub::BlockRadixSort`` over the row, and a blocked writeback.  There is no
workspace, no cross-CTA state, and no dynamic shared memory.

Two details shape the port:

* **Descending order is not a descending sort.**  The collective is
  ``cub::BlockRadixSort<uint32_t, BT, IPT, uint32_t>`` and the method called is the
  plain *ascending* ``Sort(keys, indices, 0, end_bit)`` (``:3122``); the ordering
  is inverted by complementing the key in the kernel (``:3113``).  cub's
  ``DescendingBlockRadixRank`` is never instantiated, and because the key type is
  ``uint32`` its float twiddling is the identity -- the monotone map is
  FlashInfer's own ``RadixTopKTraits::ToOrdered``.
* **"Stable" preserves tie order, it does not create it.**  Equal values keep the
  relative order they arrived in; the ``(value desc, index asc)`` property the
  comment describes holds only because the producer already emitted indices
  ascending (the finalize kernel's index sort, or the deterministic radix
  collect).  Correctness cases therefore pin the input order explicitly.

``end_bit = 8 * sizeof(OrderedType)`` -- 32 bits for fp32 and 16 for fp16/bf16,
i.e. 8 or 4 digit passes at cub's ``RADIX_BITS = 4``.
"""

from typing import Any

import tirx_kernels.kern as K
from tirx_kernels.flashinfer.topk.filtered_topk import finalize_block_config
from tirx_kernels.flashinfer.topk.radix_topk_single_cta import DTYPES, dtype_bytes
from tirx_kernels.flashinfer.utils.block_radix_sort import (
    RADIX_BITS,
    alloc_sort_smem_static,
    emit_block_radix_sort_rolled,
)
from tirx_kernels.flashinfer.utils.topk_harness import (
    SOURCE_ALGO_FILTERED,
    pin_source_algo,
    source_module,
    torch_dtype,
)
from tirx_kernels.flashinfer.utils.topk_radix import (
    ld_global_bits,
    ld_global_u32,
    st_global_u16,
    st_global_u32,
)
from tirx_kernels.runner import bench

KERNEL_META = {
    "name": "stable_sort_topk_by_value",
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

# The kernel declares no dynamic shared memory and the port declares none either:
# the cub TempStorage is a static `__shared__` allocation, matching the source.
LAUNCH_TAGS = ("blockIdx.x", "threadIdx.x")

# --- source constants ------------------------------------------------------
# Block-local sort variants cover at most 256 * 8 = 2048 elements (:3138-3141).
STABLE_SORT_MAX_K = 2048
# `top_k_val <= 1` returns cudaSuccess without launching at all (:3142-3144), so
# those shapes are a launcher early-out rather than kernel behavior.
STABLE_SORT_MIN_K = 2

# The digit-pass loop is rolled, as nvcc leaves the source's `while` loop
# (block_radix_sort.cuh:377-430): one copy of the rank/exchange body instead of P
# copies, same dynamic work and the same 9P - 1 barriers.
#
# Measured on the complete matrix at 7 rounds: rolling beats unrolling at every
# rung, most sharply at (32, 4) -- bf16 k=128 0.896 -> 1.099, f16 0.896 -> 1.084,
# f32 0.972 -> 1.016.  An earlier single-round A/B appeared to show the opposite
# at that rung and motivated a per-rung threshold; that reading was inside the
# measurement noise (the same binary read 0.897 and 1.096 on consecutive
# single-round runs), and the threshold is gone.

# Input distributions.  Stability is a property of the incoming order, so the
# pattern axis is part of the tested domain rather than a testing nicety.
PATTERNS = ("unique", "tie_heavy", "all_equal", "neg", "padded", "pipeline")


def sort_end_bit(dtype: str) -> int:
    """``end_bit = sizeof(OrderedType) * 8`` (topk.cuh:3121).

    32 bits for fp32 and 16 for the 16-bit dtypes, i.e. 8 or 4 digit passes at
    cub's ``RADIX_BITS = 4``.  Widening the 16-bit case to 32 would still sort the
    ``~0u`` padding last, but would double the pass count for nothing.
    """
    return 32 if dtype == "float32" else 16


def sort_key_u32(bits):
    """``~RadixTopKTraits<float>::ToOrdered(v)`` as one XOR (topk.cuh:3112-3113).

    The source composes a monotone flip with a complement; nvcc folds the pair
    into a single mask, and the export carries **zero** ``not.b32``::

        setp.lt.s32 %p, bits, 0;                 # sign bit set?
        selp.b32    %m, 0, 2147483647, %p;       # mask = sign ? 0 : 0x7FFFFFFF
        xor.b32     key, %m, bits;

    The map is its own inverse, so the same helper serves the load and the
    writeback: with the sign bit set the mask is 0 and the map is the identity;
    with it clear the map flips only bits ``[0, 31)`` and leaves the sign clear,
    so applying it twice cancels.

    This is **not** ``topk_radix.to_ordered_u32`` (mask ``{0x80000000,
    0xFFFFFFFF}``), which is the plain monotone flip the radix ports use.

    A sign-extending load (``ld.global.s16`` + ``cvt.s32.s16``) lets the sign test
    become a bare ``setp.lt.s32`` and removes the ``and.b32 0x8000`` mask, cutting
    16 instructions out of this kernel's fixed region -- but it measured no better.
    Two complete matrices put it at 38/46 and 40/46 shapes above the gate against
    42/46 for the form kept here, with the f32 shapes (whose code the change does
    not touch) moving in the opposite direction as an internal control.  Static
    instruction count is not the binding resource here; the form below stays.
    """
    signed = K.reinterpret("int32", bits)
    # K.Select, not K.if_then_else: the source's ternary is a predicated
    # `selp.b32`, and if_then_else would lower to a real branch.
    mask = K.Select(signed < K.int32(0), K.uint32(0), K.uint32(0x7FFFFFFF))
    return K.bitwise_xor(mask, bits)


def sort_key_u16(bits):
    """The 16-bit form: ``setp.lt.s16`` + ``selp.b16 {0, 0x7FFF}`` + ``xor.b16``.

    Same involution, in the dtype's own width (topk_common.cuh:61-64 forward,
    :67 for half and :93 for nv_bfloat16 on the way back).
    """
    signed = K.reinterpret("int16", bits)
    mask = K.Select(signed < K.int16(0), K.uint16(0), K.uint16(0x7FFF))
    return K.bitwise_xor(mask, bits)


def _validate(dtype: str, num_rows: int, k: int) -> dict[str, Any]:
    """Reject anything the source launcher would not dispatch to this kernel."""
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if num_rows < 1:
        raise ValueError(f"num_rows={num_rows} must be positive")
    if k > STABLE_SORT_MAX_K:
        raise ValueError(
            f"k={k} exceeds the block-local sort capacity {STABLE_SORT_MAX_K}: the "
            "launcher returns cudaErrorInvalidValue (:3139-3141)"
        )
    if k < STABLE_SORT_MIN_K:
        raise ValueError(
            f"k={k} is a launcher early-out: `top_k_val <= 1` returns cudaSuccess "
            "without launching the kernel (:3142-3144)"
        )
    block_threads, items_per_thread = finalize_block_config(k)
    return {
        "block_threads": block_threads,
        "items_per_thread": items_per_thread,
        "end_bit": sort_end_bit(dtype),
        "grid": num_rows,
    }


# ---------------------------------------------------------------------------
# Target entry.
# ---------------------------------------------------------------------------
def get_kernel(
    dtype: str = "float32", num_rows: int = 16, k: int = 256, pattern: str = "unique", **kwargs
):
    """Return the TIRx specialization of `StableSortTopKByValueKernel` for one cell."""
    plan = _validate(dtype, num_rows, k)
    block_threads = plan["block_threads"]
    items_per_thread = plan["items_per_thread"]
    end_bit = plan["end_bit"]
    is32 = dtype == "float32"

    @K.kernel(warps=block_threads // 32, arch="sm_100a", grid=num_rows)
    def stable_sort_topk_by_value(
        out_idx: K.gptr[K.i32, (num_rows * k,)], out_val: K.gptr[dtype, (num_rows * k,)]
    ):
        row = K.cta_id()
        tx = K.thread_id()

        # --- shared layout: the cub TempStorage union only (:3097) ----------
        # Static `__shared__`, as the source declares it -- not the dynamic pool.
        # The pool hands out offsets from a runtime `extern __shared__` base,
        # which puts address arithmetic on every shared access; static offsets
        # are compile-time constants.
        c32, c16, xchg_keys, xchg_vals, scan = alloc_sort_smem_static(
            block_threads, items_per_thread
        )

        # Row bases (:3102-3103).  The kernel is entirely in place.
        row_base = K.cast(row, "int64") * K.int64(k)
        # One base per thread, so each item's address is that base plus a
        # compile-time offset, as the source does -- it keeps two address
        # registers for the whole prologue and reaches the other items by
        # displacement (`[%rd3+2]`, `[%rd3+4]`, ..., `[%rd4+12]`).  The
        # displacement itself is not reachable from here, since these intrinsics
        # take the address as a register operand, so this only removes the
        # repeated 64-bit arithmetic (17 ops to 12) and measured neutral.
        #
        # It has to be a snapshot rather than a lazy expression.  Left lazy, the
        # simplifier proves `row * k + tx * IPT + i` fits in 32 bits and narrows
        # every one of the 32 item addresses to an int index, which puts a
        # sign-extend on each access instead of reusing one 64-bit base.  That
        # costs 1.4% at k=300 -- the one rung where the `pos < k` branch does not
        # fold, so nothing else absorbs it (f32/f16/bf16 all 1.014x).
        thread_base = K.local_scalar(
            "int64", init=row_base + K.cast(tx * items_per_thread, "int64")
        )

        keys = K.alloc_local([items_per_thread], "uint32")
        values = K.alloc_local([items_per_thread], "uint32")
        ranks = K.alloc_local([items_per_thread], "int32")

        # --- blocked load, descending key, ~0u padding (:3108-3119) --------
        # Item order follows the source: load the value, complement it, then load
        # the index (:3108-3115, and the export shows exactly that sequence under
        # a branch around each pair).  Issuing all the loads ahead of all the
        # conversions instead was measured on the shapes that reproduce to
        # +/-0.001 and is the same speed, so the source's order stands.
        with K.unroll(items_per_thread) as i:
            # `pos < k` stays a runtime predicate: pos depends on %tid.x, so a
            # static k only turns the operand into an immediate.  It folds away
            # only at the six rungs where k == BLOCK_THREADS * ITEMS_PER_THREAD.
            pos = tx * items_per_thread + i
            with K.If(pos < k):
                with K.Then():
                    slot = thread_base + K.cast(i, "int64")
                    # Source order (:3108-3115): load the value, complement it,
                    # then load the index.
                    if is32:
                        K.assign(keys[i], sort_key_u32(ld_global_bits(out_val, slot, is32)))
                    else:
                        K.assign(
                            keys[i],
                            K.cast(
                                sort_key_u16(K.cast(ld_global_bits(out_val, slot, is32), "uint16")),
                                "uint32",
                            ),
                        )
                    K.assign(values[i], ld_global_u32(out_idx, slot))
                with K.Else():
                    # ~0u is maximal within [0, end_bit) and every real key fits
                    # in end_bit bits, so padding sorts to the tail and is never
                    # written.
                    K.assign(keys[i], K.uint32(0xFFFFFFFF))
                    K.assign(values[i], K.uint32(0xFFFFFFFF))

        # --- ascending, stable, blocked -> blocked (:3121-3122) -------------
        # Descending-by-value already lives in the complemented key, so this is
        # cub's plain `Sort`, not `SortDescending`.
        emit_block_radix_sort_rolled(
            c32,
            c16,
            scan,
            xchg_keys,
            xchg_vals,
            keys,
            values,
            ranks,
            tx,
            block_threads,
            items_per_thread,
            end_bit // RADIX_BITS,
            True,  # the indices ride as satellite values
            True,  # both keys and values are 32-bit
        )

        # --- blocked writeback (:3124-3132) --------------------------------
        with K.unroll(items_per_thread) as i3:
            pos3 = tx * items_per_thread + i3
            with K.If(pos3 < k), K.Then():
                slot3 = thread_base + K.cast(i3, "int64")
                st_global_u32(out_idx, slot3, values[i3])
                # The same helper as the load: the map is an involution.
                if is32:
                    st_global_u32(out_val, slot3, sort_key_u32(keys[i3]))
                else:
                    st_global_u16(out_val, slot3, sort_key_u16(K.cast(keys[i3], "uint16")))

    return stable_sort_topk_by_value.func.with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))


_ = (bench, dtype_bytes, PATTERNS)  # wired up with the config matrix and harness


# ---------------------------------------------------------------------------
# Reference: the source's own launcher, compiled directly.
#
# FlashInfer exposes no FFI or Python entry for this kernel alone -- the pipeline
# FFI launches main(+finalize)+sort as one call -- so a kernel-vs-kernel gate has
# to build `StableSortTopKByValue<DType, int32_t>` itself.  Precedent for the
# mechanism: `tirx_kernels/basic/nvfp4_gemm.py`.
# ---------------------------------------------------------------------------
_FLASHINFER_DATA_FALLBACK = "/home/bohanhou/flashinfer/flashinfer/data"
_REFERENCE_EXT = None


def flashinfer_data_dir() -> str:
    """``flashinfer/data`` of the installed source checkout.

    The headers this reference compiles against ship inside the flashinfer
    package rather than the wheel's include path, so resolve them from the
    imported package instead of a machine-specific absolute path.
    """
    import os

    try:
        import flashinfer
    except ImportError:
        return _FLASHINFER_DATA_FALLBACK
    candidate = os.path.join(os.path.dirname(flashinfer.__file__), "data")
    return candidate if os.path.isdir(candidate) else _FLASHINFER_DATA_FALLBACK


_REF_DECL = r"""
#include <torch/extension.h>
void stable_sort_ref(at::Tensor indices, at::Tensor values, int64_t num_rows, int64_t k,
                     int64_t max_len);
"""

# `sampling.cuh` must precede `topk.cuh`, exactly as csrc/topk.cu:16-17 does;
# without it `math::ptx_rcp` is unresolved.
_REF_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <flashinfer/sampling.cuh>
#include <flashinfer/topk.cuh>

void stable_sort_ref(at::Tensor indices, at::Tensor values, int64_t num_rows, int64_t k,
                     int64_t max_len) {
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t st = cudaSuccess;
  auto* idx = static_cast<int32_t*>(indices.data_ptr());
  if (values.scalar_type() == at::kFloat) {
    st = flashinfer::sampling::StableSortTopKByValue<float, int32_t>(
        idx, static_cast<float*>(values.data_ptr()), num_rows, k, max_len, stream);
  } else if (values.scalar_type() == at::kHalf) {
    st = flashinfer::sampling::StableSortTopKByValue<half, int32_t>(
        idx, static_cast<half*>(values.data_ptr()), num_rows, k, max_len, stream);
  } else if (values.scalar_type() == at::kBFloat16) {
    st = flashinfer::sampling::StableSortTopKByValue<nv_bfloat16, int32_t>(
        idx, static_cast<nv_bfloat16*>(values.data_ptr()), num_rows, k, max_len, stream);
  } else {
    TORCH_CHECK(false, "unsupported dtype for StableSortTopKByValue");
  }
  TORCH_CHECK(st == cudaSuccess, "StableSortTopKByValue: ", cudaGetErrorString(st));
}
"""


def load_reference_ext():
    """Build and load the shape-independent reference extension.

    JIT-compiling the extension initializes CUDA, so this must only run in the
    GPU stage (the lazy `references` builder) or in `run_test`, never in the
    CPU prepare stage. The arch still comes from the prepare-stage env rather
    than the device so the build matches the suite's compile profile.
    """
    global _REFERENCE_EXT
    if _REFERENCE_EXT is not None:
        return _REFERENCE_EXT

    import fcntl
    import os
    from pathlib import Path

    from torch.utils import cpp_extension

    from tirx_kernels.runner import PREPARE_CUDA_ARCH_ENV

    arch = os.environ.get(PREPARE_CUDA_ARCH_ENV, "sm_100a").removeprefix("sm_")
    cuda_flags = [
        # torch's cpp_extension injects these unconditionally; FlashInfer's own
        # JIT never defines them, and they break vec_dtypes.cuh's half/bf16 ->
        # float conversions.  Undefining restores parity with the production
        # build rather than diverging from it.
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        # The rest mirrors gen_topk_module() over gen_jit_spec()'s common set.
        "-std=c++17",
        "-use_fast_math",
        "-Xfatbin=-compress-all",
        "-DFLASHINFER_ENABLE_F16",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
        "-lineinfo",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
    ]
    data = flashinfer_data_dir()
    name = "tirx_stable_sort_topk_reference"
    build_directory = Path(cpp_extension._get_build_directory(name, verbose=False))
    build_directory.mkdir(parents=True, exist_ok=True)
    # PyTorch's FileBaton is not process-death-safe; hold an flock instead and
    # clear any baton a dead builder left behind.
    lock_fd = os.open(str(build_directory / "lock.flock"), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        (build_directory / "lock").unlink(missing_ok=True)
        _REFERENCE_EXT = cpp_extension.load_inline(
            name=name,
            cpp_sources=[_REF_DECL],
            cuda_sources=[_REF_SOURCE],
            functions=["stable_sort_ref"],
            with_cuda=True,
            extra_include_paths=[
                f"{data}/include",
                f"{data}/cccl/cub",
                f"{data}/cccl/libcudacxx/include",
                f"{data}/cccl/thrust",
            ],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=cuda_flags,
            build_directory=str(build_directory),
            verbose=False,
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    return _REFERENCE_EXT


# ---------------------------------------------------------------------------
# Harness.
# ---------------------------------------------------------------------------
def prepare_data(
    dtype: str = "float32", num_rows: int = 16, k: int = 256, pattern: str = "unique", **kwargs
):
    """One row-set of already-selected `(value, index)` pairs, in a pinned order.

    This kernel's output is a function of its **input order** -- radix sort is
    stable, so equal values keep the order they arrived in.  Every pattern
    therefore materializes both buffers explicitly and both sides consume
    identical clones; nothing is left to allocation order or to chance.

    `pipeline` is the shape the kernel actually sees in production: the output of
    a real deterministic top-k call, whose indices are already ascending.
    """
    import torch

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(1234)
    rows, kk = num_rows, k

    if pattern == "unique":
        values = torch.randn(rows, kk, dtype=torch.float32, device=device, generator=generator) * 4
    elif pattern == "tie_heavy":
        # A handful of distinct values: most comparisons are ties, so an unstable
        # rank shows up immediately.
        values = torch.randint(
            0, 4, (rows, kk), dtype=torch.int32, device=device, generator=generator
        ).to(torch.float32)
    elif pattern == "all_equal":
        # Every value identical: the output must equal the input verbatim.
        values = torch.full((rows, kk), 1.5, dtype=torch.float32, device=device)
    elif pattern == "neg":
        # All negative: the other half of the monotone-flip sign branch.
        values = -(
            torch.rand(rows, kk, dtype=torch.float32, device=device, generator=generator) * 8 + 0.5
        )
    elif pattern == "padded":
        # Producer shape for a short row: -1 indices carrying 0.0 values.
        values = torch.randn(rows, kk, dtype=torch.float32, device=device, generator=generator) * 4
        values[:, kk // 2 :] = 0.0
    elif pattern == "pipeline":
        values = None
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    if pattern == "pipeline":
        indices, values_t = _pipeline_inputs(dtype, rows, kk, generator)
    else:
        values_t = values.to(torch_dtype(dtype)).contiguous()
        indices = torch.arange(rows * kk, dtype=torch.int32, device=device).reshape(rows, kk)
        indices = (indices % 1_000_003).contiguous()
        if pattern == "padded":
            indices[:, kk // 2 :] = -1

    return {"values": values_t, "indices": indices.contiguous(), "pattern": pattern}


def _pipeline_inputs(dtype: str, rows: int, k: int, generator):
    """Run a real deterministic top-k and hand its output to the sort.

    That is exactly what `TopKDispatch` does: the sort only ever sees a producer's
    output, whose indices are already ascending, which is what makes the
    documented `(value desc, index asc)` result hold.
    """
    import torch

    pin_source_algo(SOURCE_ALGO_FILTERED)
    module = source_module()
    length = max(4096, 4 * k)
    scores = torch.randn(rows, length, dtype=torch.float32, device="cuda", generator=generator)
    scores = scores.to(torch_dtype(dtype)).contiguous()
    out_values = torch.empty(rows, k, dtype=torch_dtype(dtype), device="cuda")
    out_indices = torch.empty(rows, k, dtype=torch.int32, device="cuda")
    # The module-level FFI signature (csrc/topk.cu:41), not the `flashinfer.top_k`
    # wrapper's: indices are an out-parameter, and `sorted_output` stays False so
    # the producer hands us an UNSORTED row -- sorting it is this kernel's job.
    module.radix_topk(
        scores,
        out_indices,
        out_values,
        None,  # row_states: the filtered path never touches it
        k,
        False,  # sorted_output
        True,  # deterministic -> indices come out ascending, as in production
        0,  # tie_break = None
        False,  # dsa_graph_safe
    )
    torch.cuda.synchronize()
    return out_indices.contiguous(), out_values.contiguous()


def build_tirx_args(cfg: dict[str, Any], data: dict[str, Any], buffers: dict[str, Any]):
    """Bind the flat launch ABI once, outside any timed region."""
    return (buffers["indices"].reshape(-1), buffers["values"].reshape(-1))


def clone_inputs(data: dict[str, Any]):
    """A fresh in-place working set: the kernel overwrites what it reads."""
    return {"indices": data["indices"].clone(), "values": data["values"].clone()}


def run_reference(cfg: dict[str, Any], buffers: dict[str, Any]) -> None:
    """One launch of the source kernel over the given buffers, in place."""
    import torch

    ext = load_reference_ext()
    ext.stable_sort_ref(
        buffers["indices"].reshape(-1),
        buffers["values"].reshape(-1),
        cfg["num_rows"],
        cfg["k"],
        1 << 20,  # max_len: dead in the kernel body, present in the arg pack
    )
    torch.cuda.synchronize()


def assert_reference_is_stable_sort(cfg, data, ref) -> None:
    """Independent oracle on the reference launch itself.

    `torch.sort(..., descending=True, stable=True)` is the definition this kernel
    implements, so the reference must reproduce it exactly -- values and the
    indices they carry.
    """
    import torch

    order = torch.sort(
        data["values"].to(torch.float32), dim=-1, descending=True, stable=True
    ).indices
    want_v = torch.gather(data["values"].to(torch.float32), 1, order)
    want_i = torch.gather(data["indices"], 1, order)
    torch.testing.assert_close(ref["values"].to(torch.float32), want_v, rtol=0, atol=0)
    torch.testing.assert_close(ref["indices"], want_i, rtol=0, atol=0)


def run_test(**config):
    """Compile, launch, and validate one config against the FlashInfer source."""
    import unittest

    import torch

    from tirx_kernels.runner import compile_kernel

    try:
        import flashinfer  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"flashinfer unavailable: {exc}") from exc
    if not torch.cuda.is_available():  # pragma: no cover - environment dependent
        raise unittest.SkipTest("CUDA device unavailable")

    cfg = _normalize_config(config)
    _validate(cfg["dtype"], cfg["num_rows"], cfg["k"])

    data = prepare_data(**cfg)

    ref = clone_inputs(data)
    run_reference(cfg, ref)
    assert_reference_is_stable_sort(cfg, data, ref)

    ex = compile_kernel(get_kernel(**cfg))
    got = clone_inputs(data)
    ex(*build_tirx_args(cfg, data, got))
    torch.cuda.synchronize()

    # The kernel is fully deterministic and stability makes the output a function
    # of the input order alone, so the comparison is bit-exact and positional.
    torch.testing.assert_close(got["indices"], ref["indices"], rtol=0, atol=0)
    torch.testing.assert_close(got["values"], ref["values"], rtol=0, atol=0)


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {"dtype": "float32", "num_rows": 16, "k": 256, "pattern": "unique"}
    cfg.update({key: value for key, value in config.items() if key != "label"})
    return cfg


# ---------------------------------------------------------------------------
# Benchmark entry points.
# ---------------------------------------------------------------------------
def prepare_bench(**kwargs: Any):
    """Specialize and compile before the workload receives a GPU.

    The reference is NOT built here. `load_reference_ext()` JITs a CUDA
    extension, which initializes CUDA, and the suite fails any workload whose
    CPU prepare does that ("CPU prepare changed CUDA initialization state from
    False to True"). It is built by the lazy `references` builder in `run_gpu`
    instead, which is what the bench API expects and what the sibling ports do.
    """
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    cfg = _normalize_config(kwargs)
    return prepared_gpu_benchmark(
        run_gpu, {"config": cfg, "executable": compile_kernel(get_kernel(**cfg))}
    )


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **kwargs):
    """Kernel-only comparison against the source launch.

    Both implementations alternate over the same two working sets rather than
    holding one each; see the comment below for why, and what it costs.

    Note what the repeat loop measures.  This kernel is in place, so a working
    set is unsorted only until something sorts it, and every later call re-sorts
    an already-sorted buffer.  That is the steady state rather than the
    production workload.  Re-randomizing between iterations is not an option:
    the copy would land inside the timed region and a per-kernel timer would
    charge it to the kernel.

    Under alternation only whichever side runs first meets the unsorted state,
    where previously each side met it once.  That asymmetry is one call against a
    25 ms warmup and a 100 ms measurement on a kernel of a few microseconds, and
    the sort's work does not depend on the data: the pass count is fixed by
    ``end_bit`` and the measured shared bank conflicts are 250 either way.
    """
    cfg = dict(prepared["config"])
    ex = prepared["executable"]

    data = prepare_data(**cfg)
    # Two working sets, and both implementations alternate over both of them.
    #
    # The kernel is in place, so a working set cannot be shared within a launch.
    # Giving each side its own single clone, though, decides the measurement by
    # where that clone landed: two separate allocations do not share an L2
    # partition on this part, and with four single-warp CTAs one side ends up
    # local and the other remote.  Remote-partition latency is fixed in
    # nanoseconds, so it costs more at boost clock and lands entirely on whoever
    # drew it -- worth 0.957 to 1.099 on one shape with both kernels untouched,
    # and enough to make the two byte-identical 16-bit kernels read 0.937 and
    # 1.082.
    #
    # Alternating in opposite phase gives each implementation each buffer for
    # half its calls, so the placement term is common to both sides and cancels
    # in the ratio.  It is symmetric by construction: exchanging the two clones
    # exchanges the phases and leaves every measurement unchanged.
    clones = (clone_inputs(data), clone_inputs(data))
    tirx_args = tuple(build_tirx_args(cfg, data, buffers) for buffers in clones)
    tirx_phase = [0]

    def tirx_launch():
        step = tirx_phase[0]
        tirx_phase[0] = step + 1
        ex(*tirx_args[step & 1])

    def build_reference():
        ext = load_reference_ext()
        flat = tuple(
            (buffers["indices"].reshape(-1), buffers["values"].reshape(-1)) for buffers in clones
        )
        rows, kk = cfg["num_rows"], cfg["k"]
        ref_phase = [0]

        def reference_launch():
            step = ref_phase[0]
            ref_phase[0] = step + 1
            flat_i, flat_v = flat[(step + 1) & 1]
            ext.stable_sort_ref(flat_i, flat_v, rows, kk, 1 << 20)

        return reference_launch

    return bench(
        {"tirx": tirx_launch},
        references={"flashinfer": build_reference},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=1.0, **config):
    prepared = prepare_bench(**config)
    return prepared.run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


# ---------------------------------------------------------------------------
# Config matrix.
# ---------------------------------------------------------------------------
_DT_TAG = {"float32": "f32", "float16": "f16", "bfloat16": "bf16"}

# The six `(BLOCK_THREADS, ITEMS_PER_THREAD)` rungs of the launcher ladder
# (:3154-3166), plus the smallest shape that still launches (`k > 1`, :3142) and
# one k that lands mid-rung rather than on a boundary.
_K_LADDER = (128, 256, 512, 576, 1024, 2048)
_K_EDGES = (2, 300)
# grid is one CTA per row, against 148 SMs on B200.
_ROW_SWEEP = (1, 16, 64, 256)


def _cfg(dtype, num_rows, k, pattern="unique"):
    label = f"{_DT_TAG[dtype]}_r{num_rows}_k{k}"
    if pattern != "unique":
        label += f"_{pattern}"
    return {"label": label, "dtype": dtype, "num_rows": num_rows, "k": k, "pattern": pattern}


def _build_configs():
    """Representative cover of the reachable domain.

    Retained in full: every ``(BLOCK_THREADS, ITEMS_PER_THREAD)`` rung of the
    launcher ladder; both digit-pass counts (8 for fp32, 4 for the 16-bit dtypes);
    all three dtypes; every input pattern that can change the answer; and the
    ``num_rows`` occupancy axis.

    The k ladder is swept per dtype at a fixed small row count, and the pattern
    axis at one rung, so the matrix stays a cover rather than a cross.

    The rows axis is **not** independent of the dtype, though the first version
    of this matrix assumed it was.  At ``k = 512`` the fp32 ratio is flat across
    the sweep (1.031, 1.035, 1.031 at 4, 32 and 256 rows) while float16 moves
    from 0.979 to 1.021 to 0.990: with four single-warp CTAs on 148 SMs nothing
    overlaps a CTA's latency chain, and the four-pass kernels have half as much
    work to hide it behind.  So both regimes are carried for both pass counts.
    """
    configs = []

    # 1. Full k ladder on every dtype -- 6 rungs x 3 dtypes covers all 18 source
    #    instantiations, and both pass counts on every rung.
    for dtype in DTYPES:
        for k in _K_LADDER:
            configs.append(_cfg(dtype, 4, k))

    # 2. Ladder edges: the smallest launchable k, and a k that is not a boundary.
    for dtype in DTYPES:
        for k in _K_EDGES:
            configs.append(_cfg(dtype, 4, k))

    # 3. Occupancy: grid is one CTA per row against 148 SMs.
    for rows in _ROW_SWEEP:
        configs.append(_cfg("float32", rows, 256))
    # The widest rung with a full SM array: the k ladder above is swept at 4 rows
    # (latency-bound) and the occupancy axis at k=256 (narrow rung), so without
    # this the matrix never runs (256, 8) against a saturated grid.
    configs.append(_cfg("float32", 64, 2048))
    # The same argument for the four-pass family, which the fp32-only sweep above
    # leaves entirely unmeasured against a saturated grid.  The 16-bit dtypes are
    # exactly where the row count matters -- the ratio moves about four points
    # across the sweep where fp32 does not -- so measuring them only at 4 rows
    # reports one end of an axis as though it were the whole of it.  float16
    # carries this: bfloat16 compiles to a byte-identical kernel.
    configs.append(_cfg("float16", 64, 256))
    configs.append(_cfg("float16", 64, 512))

    # 4. Input patterns.  Stability is a property of the incoming order, so these
    #    are domain coverage, not test flavor: `tie_heavy` and `all_equal` are the
    #    only configs that can catch an unstable rank, and `neg` is the only one
    #    that exercises the other half of the ToOrdered sign branch.
    for dtype in ("float32", "float16"):
        for pattern in ("tie_heavy", "all_equal", "neg", "padded", "pipeline"):
            configs.append(_cfg(dtype, 4, 256, pattern=pattern))
    # The (64, 8) rung with a tie-dense pattern, on every dtype.  Keeping the
    # pattern axis symmetric across dtypes at one rung is what makes an f16/bf16
    # gap attributable at all: the two compile to byte-identical kernels, so any
    # spread between them is something other than the kernel.  It is worth
    # keeping for exactly that reason -- a large spread here is what exposed the
    # measurement's dependence on where each working set was allocated, which
    # `run_gpu` now cancels.
    for dtype in DTYPES:
        configs.append(_cfg(dtype, 4, 512, pattern="tie_heavy"))
        configs.append(_cfg(dtype, 4, 512, pattern="neg"))
    # A tie-heavy row at the widest rung: 2048 equal-ish values in one block sort.
    configs.append(_cfg("float32", 2, 2048, pattern="tie_heavy"))

    # Deduplicate on launch parameters (labels alone would keep near-twins).
    seen: dict[tuple, dict[str, Any]] = {}
    for cfg in configs:
        params = tuple(sorted((key, value) for key, value in cfg.items() if key != "label"))
        seen.setdefault(params, cfg)
    deduped = list(seen.values())
    labels = [cfg["label"] for cfg in deduped]
    assert len(set(labels)) == len(labels), "duplicate config label"
    for cfg in deduped:
        _validate(cfg["dtype"], cfg["num_rows"], cfg["k"])
    return deduped


def _build_bench_configs():
    """Timed matrix: every correctness config.

    Each one launches exactly one kernel on each side, and the pattern axis is a
    real performance regime (a fully-tied row exercises a different digit
    distribution than distinct values).
    """
    return list(CONFIGS)


CONFIGS = _build_configs()
BENCH_CONFIGS = _build_bench_configs()
