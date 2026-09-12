# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ edc7c2833327d699d70b616c6f264b6ae92599b2), Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Data preparation and validation for SM100 Flex Attention backward.

The timed operation is only the direct backward main kernel from
``python/cudnn/flex_attention/kernels/sm100/bwd/backward.py``. Mask planning,
forward/LSE generation, preprocessing, and output conversion are setup work.
"""

from __future__ import annotations

import ctypes
import hashlib
import inspect
import math
import os
import random
import re
import subprocess
import tempfile
from array import array
from pathlib import Path

import torch

from tirx_kernels.cudnn._reference import load_reference_module

_REDZONE_ELEMENTS = 256
_REDZONE_F32 = 12345.25
_REDZONE_NARROW = -77.0
_TMA_L2_256B = 2
_SOURCE_PTXAS = Path("/usr/local/cuda-13.2/bin/ptxas")
_SOURCE_PTX_VERSION = "9.2"
_BENCH_SEMAPHORE_SLOTS = 16_384
_SOURCE_PTX_DUMP = None

_PRODUCTION_MASKS = {
    "causal",
    "document_causal",
    "local",
    "sink_local",
    "tree_dfs",
    "tree_bfs",
    "longformer",
    "hstu",
}
_DOCUMENT_LENGTHS_128K = (
    11858,
    8270,
    12765,
    11038,
    4578,
    14018,
    11721,
    11988,
    4942,
    8393,
    7541,
    13495,
    10465,
)


class _AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(192)
        address = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((address + 63) & ~63)


def _roundup(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _without_label(config):
    return {key: value for key, value in config.items() if key != "label"}


def assert_nvrtc_ptx92():
    """Require the default GB200 compiler contract: unset mode, NVRTC 13.2/PTX 9.2."""
    if "TVM_CUDA_COMPILE_MODE" in os.environ:
        raise RuntimeError(
            "TVM_CUDA_COMPILE_MODE must be absent; this kernel always uses default NVRTC"
        )
    from cuda.bindings import nvrtc

    error, major, minor = nvrtc.nvrtcVersion()
    if int(error) != 0 or (int(major), int(minor)) != (13, 2):
        raise RuntimeError(
            f"GB200 requires NVRTC 13.2 (PTX {_SOURCE_PTX_VERSION}), got {int(major)}.{int(minor)}"
        )


def _prepare_source_ptx92():
    """Force the pinned CUTLASS DSL source through CUDA 13.2 PTX ISA 9.2."""
    global _SOURCE_PTX_DUMP

    assert_nvrtc_ptx92()
    if not _SOURCE_PTXAS.is_file():
        raise RuntimeError(f"required source assembler is missing: {_SOURCE_PTXAS}")
    version = subprocess.run(
        (str(_SOURCE_PTXAS), "--version"), check=True, capture_output=True, text=True
    ).stdout
    if "release 13.2" not in version:
        raise RuntimeError(f"source ptxas must be CUDA 13.2, got: {version.strip()}")

    if _SOURCE_PTX_DUMP is None:
        _SOURCE_PTX_DUMP = tempfile.TemporaryDirectory(prefix="tirx-flex-bwd-source-ptx-")
    os.environ["CUTE_DSL_PTXAS_PATH"] = str(_SOURCE_PTXAS)
    os.environ["CUTE_DSL_KEEP_PTX"] = "1"
    os.environ["CUTE_DSL_DUMP_DIR"] = _SOURCE_PTX_DUMP.name

    # Importing dispatch installs the source repository's system-ptxas hook.
    flex = load_reference_module("cudnn.flex_attention")
    dispatch = load_reference_module("cudnn.flex_attention.dispatch")
    ptxas = load_reference_module("cudnn.flex_attention.runtime.ptxas")
    ptxas.CUTE_DSL_PTXAS_PATH = str(_SOURCE_PTXAS)

    # CuTe-DSL reads CUTE_DSL_KEEP_PTX / CUTE_DSL_DUMP_DIR once, when its DSL
    # singleton is constructed; a process that imported cutlass before this
    # function ran would otherwise never dump the PTX the audit below needs.
    from cutlass.base_dsl.dsl import BaseDSL

    for dsl in list(type(BaseDSL)._instances.values()):
        envar = dsl.envar
        envar.keep_tokens = frozenset(set(envar.keep_tokens) | {"ptx"})
        envar.keep_ptx = True
        envar.dump_dir = _SOURCE_PTX_DUMP.name

    if not getattr(ptxas, "_tirx_ptx92_patched", False):
        original_compile = ptxas._compile_ptx

        def compile_ptx92(ptx_path, ptx_content):
            versions = re.findall(r"(?m)^\.version\s+([0-9]+\.[0-9]+)\s*$", ptx_content)
            # Raw directives seen from the pinned CuTe-DSL releases (4.6 emitted
            # 9.4, 4.7 emits 9.3); every one is canonicalized to ISA 9.2 below.
            if len(versions) != 1 or versions[0] not in {_SOURCE_PTX_VERSION, "9.3", "9.4"}:
                raise RuntimeError(f"unexpected source PTX version directives: {versions!r}")
            canonical, replacements = re.subn(
                r"(?m)^\.version\s+[0-9]+\.[0-9]+\s*$",
                f".version {_SOURCE_PTX_VERSION}",
                ptx_content,
                count=1,
            )
            if replacements != 1:
                raise RuntimeError("failed to canonicalize source PTX to ISA 9.2")
            ptxas._tirx_ptx92_audit.append(
                {
                    "raw_version": versions[0],
                    "canonical_version": _SOURCE_PTX_VERSION,
                    "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                }
            )
            cubin = original_compile(ptx_path, canonical)
            capture_dir = os.environ.get("TIRX_FLEX_BWD_SOURCE_CAPTURE_DIR")
            if capture_dir and "FlexAttentionBackwardSm100" in Path(ptx_path).name:
                destination = Path(capture_dir)
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "source_main.ptx").write_text(canonical)
                (destination / "source_main.cubin").write_bytes(cubin)
            return cubin

        def forbid_embedded_ptxas(_compiled_func):
            raise RuntimeError("source PTX 9.2 system-ptxas path failed; fallback is forbidden")

        ptxas._tirx_ptx92_audit = []
        ptxas._compile_ptx = compile_ptx92
        from cutlass._mlir import ir

        def canonicalize_embedded_binary(dsl, module):
            # TVM-FFI source callables bypass CudaDialectJitCompiledFunction.to()
            # and export the lowered module into their own execution engine, so
            # replace the already-lowered embedded binary immediately before the
            # wrapper is constructed from that module.
            ptx_path = dsl.compile_options.full_ptx_path
            if not (ptx_path and Path(ptx_path).is_file()):
                return
            canonical_cubin = ptxas._compile_ptx(
                Path(ptx_path), Path(ptx_path).read_text().rstrip("\x00")
            )
            binary_globals = []
            for operation in module.body.operations:
                if str(operation.operation.name) != "llvm.mlir.global":
                    continue
                attributes = operation.operation.attributes
                symbol = str(attributes["sym_name"].value)
                if symbol.endswith("_binary"):
                    binary_globals.append(operation)
            if len(binary_globals) != 1:
                raise RuntimeError(
                    f"expected one embedded source CUDA binary, got {len(binary_globals)}"
                )
            with module.context:
                binary_global = binary_globals[0].operation
                binary_global.attributes["value"] = ir.StringAttr.get(canonical_cubin)
                binary_global.attributes["global_type"] = ir.TypeAttr.get(
                    ir.Type.parse(f"!llvm.array<{len(canonical_cubin)} x i8>")
                )
                module.operation.verify()

        # CuTe-DSL 4.7 dropped the BaseDSL._build_jit_function helper: the
        # wrapper is now constructed inline at the end of compile_and_cache as
        # ``func_type(module, engine, ...)`` on the lowered module. Intercept
        # ``func_type`` so the binary replacement keeps the same insertion point.
        original_compile_and_cache = BaseDSL.compile_and_cache
        compile_and_cache_params = list(inspect.signature(original_compile_and_cache).parameters)
        func_type_index = compile_and_cache_params.index("func_type") - 1  # without self
        default_func_type = (
            inspect.signature(original_compile_and_cache).parameters["func_type"].default
        )

        def compile_and_cache_ptx92(self, *args, **kwargs):
            if len(args) > func_type_index:
                func_type = args[func_type_index]
            else:
                func_type = kwargs.pop("func_type", default_func_type)

            def build_jit_function_ptx92(module, *build_args, **build_kwargs):
                canonicalize_embedded_binary(self, module)
                return func_type(module, *build_args, **build_kwargs)

            if len(args) > func_type_index:
                args = (
                    *args[:func_type_index],
                    build_jit_function_ptx92,
                    *args[func_type_index + 1 :],
                )
            else:
                kwargs["func_type"] = build_jit_function_ptx92
            return original_compile_and_cache(self, *args, **kwargs)

        BaseDSL.compile_and_cache = compile_and_cache_ptx92
        compiled_cls = ptxas.cutlass.cutlass_dsl.cuda_jit_executor.CudaDialectJitCompiledFunction
        if compiled_cls._load_cuda_library is not ptxas._patched_load_cuda_library:
            ptxas.patch()
        ptxas._original_load_cuda_library = forbid_embedded_ptxas
        # KEEP_PTX is required to expose the source artifact to the hook. The
        # audit above retains its version/hash, so remove the temporary file.
        ptxas._user_wanted_ptx = False
        ptxas._tirx_ptx92_patched = True
    return flex, dispatch, ptxas


def validate_config(config):
    batch = int(config["batch"])
    q_heads = int(config["num_q_heads"])
    kv_heads = int(config["num_kv_heads"])
    d = int(config["head_dim"])
    dv = int(config["head_dim_v"])
    if batch < 1 or q_heads < 1 or kv_heads < 1:
        raise ValueError("batch and head counts must be positive")
    if q_heads % kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if config["dtype"] not in {"float16", "bfloat16"}:
        raise ValueError("dtype must be float16 or bfloat16")
    if d not in range(8, 129, 8) and d != 192:
        raise ValueError("head_dim must be 8..128 in steps of 8, or 192")
    if dv not in range(8, 129, 8):
        raise ValueError("head_dim_v must be 8..128 in steps of 8")
    if d == 192 and dv != 128:
        raise ValueError("head_dim 192 dispatches here only with head_dim_v 128")
    varlen = bool(config.get("varlen", False))
    q_lengths = config["seqlen_q"]
    k_lengths = config["seqlen_kv"]
    if varlen:
        if len(q_lengths) != batch or len(k_lengths) != batch:
            raise ValueError("varlen lengths must contain one entry per sample")
        if any(int(value) <= 0 for value in (*q_lengths, *k_lengths)):
            raise ValueError("sequence lengths must be positive")
    elif int(q_lengths) <= 0 or int(k_lengths) <= 0:
        raise ValueError("sequence lengths must be positive")


def _jittered_lengths(total, count, *, seed, jitter):
    generator = random.Random(seed)
    weights = [generator.uniform(1.0 - jitter, 1.0 + jitter) for _ in range(count)]
    scaled = [weight * total / sum(weights) for weight in weights]
    lengths = [max(1, math.floor(value)) for value in scaled]
    difference = total - sum(lengths)
    fractional_order = sorted(
        range(count), key=lambda index: scaled[index] - math.floor(scaled[index]), reverse=True
    )
    for index in range(difference):
        lengths[fractional_order[index % count]] += 1
    return lengths


def _bounded_partition(total, minimum, maximum, *, seed, prefer_multiple=False):
    if total < minimum:
        return [total]
    min_parts = math.ceil(total / maximum)
    max_parts = total // minimum
    count = round(total / ((minimum + maximum) / 2))
    count = min(max(count, min_parts), max_parts)
    if prefer_multiple and count == 1 and max_parts >= 2:
        count = 2
    generator = random.Random(seed)
    lengths = [generator.randint(minimum, maximum) for _ in range(count)]
    difference = total - sum(lengths)
    while difference:
        order = list(range(count))
        generator.shuffle(order)
        progressed = False
        for index in order:
            if difference > 0:
                delta = min(difference, maximum - lengths[index])
            else:
                delta = -min(-difference, lengths[index] - minimum)
            if delta:
                lengths[index] += delta
                difference -= delta
                progressed = True
            if difference == 0:
                break
        if not progressed:
            raise AssertionError("bounded partition cannot satisfy the requested total")
    return lengths


def _merge_intervals(intervals):
    merged = []
    for begin, end in sorted(intervals):
        if begin == end:
            continue
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    return [(begin, end) for begin, end in merged]


def _encode_intervals(intervals, *, max_intervals, starts_at_zero):
    merged = _merge_intervals(intervals)
    if starts_at_zero:
        encoded = [merged[0][1]]
        encoded.extend(value for pair in merged[1:] for value in pair)
        expected = 2 * max_intervals - 1
    else:
        encoded = [0]
        encoded.extend(value for pair in merged for value in pair)
        expected = 2 * max_intervals + 1
    encoded.extend([encoded[-1]] * (expected - len(encoded)))
    return encoded


def _tree_order(depth, traversal):
    node_count = 2**depth - 1
    if traversal == "bfs":
        return list(range(node_count))
    order = []

    def visit(node):
        if node >= node_count:
            return
        order.append(node)
        visit(2 * node + 1)
        visit(2 * node + 2)

    visit(0)
    return order


def _tree_ancestors(node):
    result = []
    while True:
        result.append(node)
        if node == 0:
            break
        node = (node - 1) // 2
    return list(reversed(result))


def _production_endpoints(mask_type, seqlen):
    """Exact endpoint builders from the pointed upstream 128K benchmark."""
    q = torch.arange(seqlen, dtype=torch.int32)
    if mask_type == "causal":
        return (q + 1).view(1, -1)
    if mask_type == "document_causal":
        lengths = (
            list(_DOCUMENT_LENGTHS_128K)
            if seqlen == 128 * 1024
            else _bounded_partition(seqlen, 4096, 16384, seed=42, prefer_multiple=True)
        )
        endpoints = torch.empty((3, seqlen), dtype=torch.int32)
        offset = 0
        for length in lengths:
            end = offset + length
            endpoints[0, offset:end] = 0
            endpoints[1, offset:end] = offset
            endpoints[2, offset:end] = torch.arange(offset + 1, end + 1, dtype=torch.int32)
            offset = end
        return endpoints
    if mask_type == "local":
        begin = torch.clamp(q - 512, min=0)
        return torch.stack((torch.zeros_like(begin), begin, q + 1))
    if mask_type == "sink_local":
        end = q + 1
        sink = torch.minimum(end, torch.scalar_tensor(4, dtype=torch.int32))
        begin = torch.clamp(q - 512, min=4)
        begin = torch.minimum(torch.maximum(begin, sink), end)
        return torch.stack((sink, begin, end))
    if mask_type in {"tree_dfs", "tree_bfs"}:
        depth = 7
        lengths = _jittered_lengths(seqlen, 2**depth - 1, seed=42, jitter=0.2)
        order = _tree_order(depth, mask_type.removeprefix("tree_"))
        starts = {}
        offset = 0
        for node in order:
            starts[node] = offset
            offset += lengths[node]
        endpoints = torch.empty((2 * depth - 1, seqlen), dtype=torch.int32)
        for node in order:
            start = starts[node]
            end = start + lengths[node]
            row_end = torch.arange(start + 1, end + 1, dtype=torch.int32)
            ancestors = [
                (starts[parent], starts[parent] + lengths[parent])
                for parent in _tree_ancestors(node)[:-1]
            ]
            template = _merge_intervals((*ancestors, (start, end)))
            encoded = _encode_intervals(template, max_intervals=depth, starts_at_zero=True)
            current_end_slot = 2 * len(template) - 2
            for slot, value in enumerate(encoded):
                endpoints[slot, start:end] = row_end if slot >= current_end_slot else value
        return endpoints
    if mask_type == "longformer":
        radius = 256
        global_tokens = sorted(
            {min(int((index + 0.5) * seqlen / 8), seqlen - 1) for index in range(8)}
        )
        global_set = set(global_tokens)
        flat = array("i")
        for q_index in range(seqlen):
            if q_index in global_set:
                intervals = [(0, seqlen)]
            else:
                intervals = [(max(0, q_index - radius), min(seqlen, q_index + radius + 1))]
                intervals.extend((token, token + 1) for token in global_tokens)
            flat.extend(_encode_intervals(intervals, max_intervals=9, starts_at_zero=False))
        return torch.tensor(flat, dtype=torch.int32).view(seqlen, 19).t().contiguous()
    if mask_type == "hstu":
        contexts = _bounded_partition(seqlen // 2, 4096, 8192, seed=42)
        targets = contexts.copy()
        targets[-1] += seqlen % 2
        endpoints = torch.empty((5, seqlen), dtype=torch.int32)
        offset = 0
        for context, target in zip(contexts, targets):
            context_end = offset + context
            document_end = context_end + target
            context_rows = torch.arange(offset + 1, context_end + 1, dtype=torch.int32)
            endpoints[0, offset:context_end] = 0
            endpoints[1, offset:context_end] = offset
            endpoints[2, offset:context_end] = context_rows
            endpoints[3, offset:context_end] = context_rows
            endpoints[4, offset:context_end] = context_rows
            target_rows = torch.arange(context_end, document_end, dtype=torch.int32)
            endpoints[0, context_end:document_end] = 0
            endpoints[1, context_end:document_end] = offset
            endpoints[2, context_end:document_end] = context_end
            endpoints[3, context_end:document_end] = target_rows
            endpoints[4, context_end:document_end] = target_rows + 1
            offset = document_end
        return endpoints
    raise ValueError(f"unsupported production mask: {mask_type}")


def _sample_endpoints(mask_type, q_length, k_length, nfunc, variant):
    q = torch.arange(q_length, dtype=torch.int32)
    end = torch.minimum(q + 1, torch.tensor(k_length, dtype=torch.int32))
    zeros = torch.zeros_like(q)
    if mask_type == "full":
        encoded = torch.full((1, q_length), k_length, dtype=torch.int32)
    elif mask_type == "causal":
        encoded = end.unsqueeze(0)
    elif mask_type in {"local", "document_causal"}:
        width = 31 + 17 * (variant % 5) if mask_type == "local" else max(1, k_length // 3)
        begin = torch.clamp(end - width, min=0)
        encoded = torch.stack((zeros, begin, end))
    elif mask_type == "sink_local":
        sink = torch.minimum(end, torch.tensor(4, dtype=torch.int32))
        begin = torch.clamp(end - 64, min=4)
        begin = torch.minimum(torch.maximum(begin, sink), end)
        encoded = torch.stack((sink, begin, end))
    else:
        # Two disjoint, query-dependent intervals exercise arbitrary masks,
        # while repeated terminal endpoints fill the source's odd nfunc ABI.
        local_begin = torch.clamp(end - (19 + variant % 13), min=0)
        far_begin = torch.minimum(
            torch.full_like(q, (variant * 11 + 7) % max(1, k_length)), local_begin
        )
        far_end = torch.minimum(far_begin + 1 + variant % 5, local_begin)
        encoded = torch.stack((far_end, local_begin, end))
    requested = max(int(nfunc), encoded.shape[0])
    if requested % 2 == 0:
        requested += 1
    if requested >= 33:
        requested = 31
    if encoded.shape[0] < requested:
        encoded = torch.cat((encoded, encoded[-1:].expand(requested - encoded.shape[0], -1)), dim=0)
    return encoded


def _make_endpoints(config, q_lengths, k_lengths):
    q_heads = int(config["num_q_heads"])
    per_head = "per_head" in config.get("mask_head_mode", "broadcast")
    mask_heads = q_heads if per_head else 1
    samples = []
    for sample, (q_length, k_length) in enumerate(zip(q_lengths, k_lengths)):
        if q_length == k_length and config["mask_type"] in _PRODUCTION_MASKS:
            exact = _production_endpoints(config["mask_type"], int(q_length))
            heads = [exact for _ in range(mask_heads)]
        else:
            heads = []
            for mask_head in range(mask_heads):
                heads.append(
                    _sample_endpoints(
                        config["mask_type"],
                        int(q_length),
                        int(k_length),
                        int(config.get("mask_nfunc", 1)),
                        sample * q_heads + mask_head,
                    )
                )
        samples.append(torch.stack(heads))
    nfunc = max(value.shape[1] for value in samples)
    padded = []
    for value in samples:
        if value.shape[1] < nfunc:
            value = torch.cat((value, value[:, -1:].expand(-1, nfunc - value.shape[1], -1)), dim=1)
        padded.append(value)
    return torch.cat(padded, dim=2).contiguous().cuda()


def _random_tensor(generator, shape, dtype, *, broadcast_heads=False):
    if broadcast_heads:
        base_shape = (*shape[:-2], 1, shape[-1])
        value = torch.randn(base_shape, generator=generator, dtype=dtype, device="cuda")
        value.mul_(0.125)
        return value.expand(shape)
    value = torch.randn(shape, generator=generator, dtype=dtype, device="cuda")
    value.mul_(0.125)
    return value


def _guarded_tensor(shape, dtype, fill):
    count = math.prod(shape)
    storage = torch.full((count + 2 * _REDZONE_ELEMENTS,), fill, dtype=dtype, device="cuda")
    tensor = storage[_REDZONE_ELEMENTS : _REDZONE_ELEMENTS + count].reshape(shape)
    return tensor, storage


def _new_outputs(q, k, v):
    result = {}
    for name, template in (("dq", q), ("dk", k), ("dv", v)):
        tensor, storage = _guarded_tensor(template.shape, template.dtype, _REDZONE_NARROW)
        tensor.fill_(float("nan"))
        result[name] = tensor
        result[name + "_storage"] = storage
    return result


def _workspace_state(bundle, q, k, v):
    state = _new_outputs(q, k, v)
    guards = {}
    for name, shape, dtype in bundle.workspace_specs:
        if shape is None:
            state[name] = None
            continue
        tensor, storage = _guarded_tensor(
            shape, dtype, _REDZONE_F32 if dtype == torch.float32 else 12345
        )
        tensor.zero_()
        state[name] = tensor
        guards[name] = storage
    state["workspace_guards"] = guards
    return state


def _as_preallocated(state):
    dispatch = load_reference_module("cudnn.flex_attention.dispatch")
    return dispatch._BwdPreallocated(
        state["dq"],
        state["dk"],
        state["dv"],
        *[
            state[name]
            for name in (
                "dq_accum",
                "dpsum",
                "lse_log2",
                "dk_accum",
                "dv_accum",
                "dq_semaphore",
                "dk_semaphore",
                "dv_semaphore",
            )
        ],
    )


def _encode_tensormap(tensor, dtype_name, dims, strides, box, swizzle):
    import tvm

    rank = len(dims)
    if len(strides) != rank - 1 or len(box) != rank:
        raise ValueError("TensorMap dimensions, strides, and box disagree")
    result = _AlignedTensorMap()
    tvm.get_global_func("runtime.cuTensorMapEncodeTiled")(
        result.ptr,
        dtype_name,
        rank,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *dims,
        *strides,
        *box,
        *((1,) * rank),
        0,
        swizzle,
        _TMA_L2_256B,
        0,
    )
    return result


def _tensor_geometry(tensor):
    """Return source-order sequence/head/batch geometry in bytes."""
    element_size = tensor.element_size()
    if tensor.ndim == 3:
        total_sequence = tensor.shape[0]
        heads = tensor.shape[1]
        sequence_stride = tensor.stride(0) * element_size
        head_stride = tensor.stride(1) * element_size
        batch_stride = tensor.numel() * element_size
        batches = 1
    else:
        total_sequence = tensor.shape[1]
        heads = tensor.shape[2]
        sequence_stride = tensor.stride(1) * element_size
        head_stride = tensor.stride(2) * element_size
        batch_stride = tensor.stride(0) * element_size
        batches = tensor.shape[0]
    return total_sequence, heads, batches, sequence_stride, head_stride, batch_stride


def _cooperative_col_tensormap(tensor, feature_dim, columns, rows, dtype_name):
    """Source CTA-group-two MN-major feature/sequence TensorMap."""
    total_sequence, heads, batches, sequence_stride, head_stride, batch_stride = _tensor_geometry(
        tensor
    )
    if tensor.ndim == 3:
        dims = (feature_dim, total_sequence, heads)
        strides = (sequence_stride, head_stride)
        box = (columns, rows, 1)
    else:
        dims = (feature_dim, total_sequence, heads, batches)
        strides = (sequence_stride, head_stride, batch_stride)
        box = (columns, rows, 1, 1)
    return _encode_tensormap(
        tensor,
        dtype_name,
        dims,
        strides,
        box,
        # CuTe's transpose layouts use S<3,4,3> for 64 half elements
        # (128 bytes) and S<2,4,3> for 32 half elements (64 bytes).  The
        # latter is the D192 Q.T/K.T atom; encoding it as SW128 leaves every
        # other 64-byte shared segment outside the TMA tile.
        {32: 2, 64: 3}[columns],
    )


def _input_tensormap(tensor, feature_dim, total_sequence, heads, dtype_name):
    element_size = tensor.element_size()
    if tensor.ndim == 3:
        sequence_stride = tensor.stride(0) * element_size
        head_stride = tensor.stride(1) * element_size
        batch_stride = tensor.numel() * element_size
        batches = 1
    else:
        sequence_stride = tensor.stride(1) * element_size
        head_stride = tensor.stride(2) * element_size
        batch_stride = tensor.stride(0) * element_size
        batches = tensor.shape[0]
    tile_dim = _roundup(int(feature_dim), 16)
    atom_features = math.gcd(64, tile_dim)
    swizzle = {16: 1, 32: 2, 64: 3}[atom_features]
    if tensor.ndim == 3:
        dims = (feature_dim, total_sequence, heads)
        strides = (sequence_stride, head_stride)
        box = (atom_features, 128, 1)
    else:
        dims = (feature_dim, total_sequence, heads, batches)
        strides = (sequence_stride, head_stride, batch_stride)
        box = (atom_features, 128, 1, 1)
    return _encode_tensormap(tensor, dtype_name, dims, strides, box, swizzle)


def _output_tensormap(tensor, feature_dim, sequence, heads, dtype_name):
    ncol = math.gcd(64, feature_dim // 2)
    swizzle = {16: 0, 32: 1, 64: 2, 128: 3}[ncol * tensor.element_size()]
    return _encode_tensormap(
        tensor,
        dtype_name,
        (feature_dim, sequence, heads, tensor.shape[0]),
        (
            tensor.stride(1) * tensor.element_size(),
            tensor.stride(2) * tensor.element_size(),
            tensor.stride(0) * tensor.element_size(),
        ),
        (ncol, 128, 1, 1),
        swizzle,
    )


def _build_maps(data):
    config = data["config"]
    inputs = data["inputs"]
    target = data["tirx"]
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    do = inputs["do"]
    dtype_name = config["dtype"]
    q_total = q.shape[0] if q.ndim == 3 else q.shape[1]
    k_total = k.shape[0] if k.ndim == 3 else k.shape[1]
    head_dim = int(config["head_dim"])
    head_dim_v = int(config["head_dim_v"])
    cooperative = (head_dim, head_dim_v) in ((128, 128), (192, 128))
    if cooperative and head_dim == 128:
        maps = {
            # Fixed BSHD uses rank four; packed varlen THD drops the batch
            # coordinate and uses the source's rank-three TensorMaps.
            "q": _cooperative_col_tensormap(q, head_dim, 64, 64, dtype_name),
            "qt": _cooperative_col_tensormap(q, head_dim, 64, 128, dtype_name),
            "k": _cooperative_col_tensormap(k, head_dim, 64, 128, dtype_name),
            "kt": _cooperative_col_tensormap(k, head_dim, 64, 256, dtype_name),
            "v": _cooperative_col_tensormap(v, head_dim_v, 64, 128, dtype_name),
            "do": _cooperative_col_tensormap(do, head_dim_v, 64, 128, dtype_name),
            "dot": _cooperative_col_tensormap(do, head_dim_v, 64, 64, dtype_name),
        }
    elif cooperative:
        maps = {
            # D192 changes only the transposed feature chunk: Q.T/K.T use
            # three 32-feature transfers while the row-major families retain
            # their 64-feature chunks.  These boxes are the source MLIR's
            # exact seven non-exec TMA load atoms.
            "q": _cooperative_col_tensormap(q, head_dim, 64, 64, dtype_name),
            "qt": _cooperative_col_tensormap(q, head_dim, 32, 128, dtype_name),
            "k": _cooperative_col_tensormap(k, head_dim, 64, 128, dtype_name),
            "kt": _cooperative_col_tensormap(k, head_dim, 32, 256, dtype_name),
            "v": _cooperative_col_tensormap(v, head_dim_v, 64, 128, dtype_name),
            "do": _cooperative_col_tensormap(do, head_dim_v, 64, 128, dtype_name),
            "dot": _cooperative_col_tensormap(do, head_dim_v, 64, 64, dtype_name),
        }
    else:
        maps = {
            "q": _input_tensormap(q, head_dim, q_total, int(config["num_q_heads"]), dtype_name),
            "k": _input_tensormap(k, head_dim, k_total, int(config["num_kv_heads"]), dtype_name),
            "v": _input_tensormap(v, head_dim_v, k_total, int(config["num_kv_heads"]), dtype_name),
            "do": _input_tensormap(do, head_dim_v, q_total, int(config["num_q_heads"]), dtype_name),
        }
    exact_mha = (
        int(config["num_q_heads"]) == int(config["num_kv_heads"])
        and int(config["head_dim"]) % 16 == 0
        and int(config["head_dim_v"]) % 16 == 0
    )
    if exact_mha and not bool(config.get("varlen", False)):
        maps["dk"] = _output_tensormap(
            target["dk"],
            head_dim,
            int(config["seqlen_kv"]),
            int(config["num_kv_heads"]),
            dtype_name,
        )
        maps["dv"] = _output_tensormap(
            target["dv"],
            head_dim_v,
            int(config["seqlen_kv"]),
            int(config["num_kv_heads"]),
            dtype_name,
        )
    else:
        maps["dk"] = maps["q"]
        maps["dv"] = maps["v"]
    return maps


def _tensor_or_dummy(value, *, dtype=torch.int32, dummy=None):
    if value is not None and value.numel():
        return value.reshape(-1)
    if dummy is not None:
        return dummy
    return torch.zeros((1,), dtype=dtype, device="cuda")


def _reset_source_state(state):
    for name in (
        "dq_accum",
        "dk_accum",
        "dv_accum",
        "dq_semaphore",
        "dk_semaphore",
        "dv_semaphore",
    ):
        if state.get(name) is not None:
            state[name].zero_()
    state["dq"].fill_(float("nan"))
    state["dk"].fill_(float("nan"))
    state["dv"].fill_(float("nan"))


def prepare_benchmark_semaphores(data, *, slots=_BENCH_SEMAPHORE_SLOTS):
    """Allocate caller-owned clean semaphore slots before timing starts."""
    if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
        raise ValueError("benchmark semaphore slots must be a positive integer")
    names = ("dq_semaphore", "dk_semaphore", "dv_semaphore")
    for owner in ("tirx", "source"):
        state = data[owner]
        state["benchmark_semaphore_slots"] = slots
        state["benchmark_semaphore_index"] = 0
        state["benchmark_semaphore_pool"] = {
            name: torch.zeros((slots, *state[name].shape), dtype=state[name].dtype, device="cuda")
            for name in names
            if state.get(name) is not None
        }


def _next_benchmark_semaphores(state):
    names = ("dq_semaphore", "dk_semaphore", "dv_semaphore")
    pool = state.get("benchmark_semaphore_pool")
    if pool is None:
        return tuple(state.get(name) for name in names)
    if not pool:
        return (None, None, None)
    index = state["benchmark_semaphore_index"]
    capacity = state["benchmark_semaphore_slots"]
    if index >= capacity:
        raise RuntimeError(
            f"benchmark exhausted {capacity} pre-zeroed semaphore slots; "
            "reduce warmup/repeat/rounds rather than resetting inside timing"
        )
    state["benchmark_semaphore_index"] = index + 1
    return tuple(pool[name][index] if name in pool else None for name in names)


def _prepare_target_workspace(data):
    config = data["config"]
    state = data["tirx"]
    bundle = data["source_bundle"]
    inputs = data["inputs"]
    cu_q = inputs["cu_q"]
    bundle.preprocess(
        inputs["o"],
        inputs["do"],
        state["dpsum"],
        inputs["lse"],
        state["lse_log2"],
        state["dq_accum"],
        cu_q,
        None,
        int(data["derived"]["max_q"]),
    )
    q_rows = state["dpsum"].numel() // int(config["num_q_heads"])
    d = _roundup(int(config["head_dim"]), 16)
    dv = _roundup(int(config["head_dim_v"]), 16)
    batch = int(config["batch"])
    cooperative = (int(config["head_dim"]), int(config["head_dim_v"])) in ((128, 128), (192, 128))
    k_tile = 256 if cooperative else 128
    if bool(config.get("varlen", False)):
        k_rows = (sum(data["derived"]["k_lengths"]) + (batch + 1) * k_tile - 1) // k_tile * k_tile
    else:
        k_rows = batch * _roundup(int(config["seqlen_kv"]), k_tile)
    sum_plane = int(config["num_q_heads"]) * q_rows
    dq_base = 2 * sum_plane
    dk_base = dq_base + int(config["num_q_heads"]) * q_rows * d
    dv_base = dk_base + int(config["num_kv_heads"]) * k_rows * d
    total = dv_base + int(config["num_kv_heads"]) * k_rows * dv
    combined, storage = _guarded_tensor((total,), torch.float32, _REDZONE_F32)
    combined.zero_()
    combined[:sum_plane].copy_(state["dpsum"].reshape(-1))
    combined[sum_plane : 2 * sum_plane].copy_(state["lse_log2"].reshape(-1))
    combined[dq_base : dq_base + state["dq_accum"].numel()].copy_(state["dq_accum"].reshape(-1))
    state.update(
        combined=combined,
        combined_storage=storage,
        q_rows=q_rows,
        k_rows=k_rows,
        dq_base=dq_base,
        dk_base=dk_base,
        dv_base=dv_base,
    )


def prepare_data(**config):
    validate_config(config)
    if not torch.cuda.is_available():
        import unittest

        raise unittest.SkipTest("CUDA is required")
    varlen = bool(config.get("varlen", False))
    batch = int(config["batch"])
    q_lengths = (
        tuple(int(value) for value in config["seqlen_q"])
        if varlen
        else (int(config["seqlen_q"]),) * batch
    )
    k_lengths = (
        tuple(int(value) for value in config["seqlen_kv"])
        if varlen
        else (int(config["seqlen_kv"]),) * batch
    )
    q_heads = int(config["num_q_heads"])
    kv_heads = int(config["num_kv_heads"])
    d = int(config["head_dim"])
    dv = int(config["head_dim_v"])
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[config["dtype"]]
    generator = torch.Generator(device="cuda").manual_seed(int(config["seed"]))
    q_shape = (sum(q_lengths), q_heads, d) if varlen else (batch, q_lengths[0], q_heads, d)
    k_shape = (sum(k_lengths), kv_heads, d) if varlen else (batch, k_lengths[0], kv_heads, d)
    v_shape = (sum(k_lengths), kv_heads, dv) if varlen else (batch, k_lengths[0], kv_heads, dv)
    o_shape = (sum(q_lengths), q_heads, dv) if varlen else (batch, q_lengths[0], q_heads, dv)
    q = _random_tensor(
        generator, q_shape, dtype, broadcast_heads=bool(config.get("zero_stride_q_do_heads", False))
    )
    k = _random_tensor(
        generator, k_shape, dtype, broadcast_heads=bool(config.get("zero_stride_kv_heads", False))
    )
    v = _random_tensor(
        generator, v_shape, dtype, broadcast_heads=bool(config.get("zero_stride_kv_heads", False))
    )
    endpoints = _make_endpoints(config, q_lengths, k_lengths)
    cu_q = cu_k = None
    plan_args = {}
    if varlen:
        cu_q = torch.tensor(
            (0, *[sum(q_lengths[: index + 1]) for index in range(batch)]),
            dtype=torch.int32,
            device="cuda",
        )
        cu_k = torch.tensor(
            (0, *[sum(k_lengths[: index + 1]) for index in range(batch)]),
            dtype=torch.int32,
            device="cuda",
        )
        plan_args = {
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": max(q_lengths),
            "max_seqlen_k": max(k_lengths),
        }
    flex, dispatch, ptxas = _prepare_source_ptx92()
    plan = flex.create_mask_plan(
        endpoints, q, k, v, pack_gqa=False, build_backward=True, **plan_args
    )
    outer, canonical_q, canonical_k = plan._runtime_args
    if varlen:
        cu_q, cu_k = canonical_q, canonical_k
        plan_args.update(cu_seqlens_q=cu_q, cu_seqlens_k=cu_k)
    scale = (
        1.0 / math.sqrt(d)
        if config.get("softmax_scale") is None
        else float(config["softmax_scale"])
    )
    o, lse = dispatch._flex_attn_fwd(
        q,
        k,
        v,
        softmax_scale=scale,
        pack_gqa=False,
        block_sparse_tensors=outer,
        return_lse=True,
        **plan_args,
    )
    do = _random_tensor(
        generator, o_shape, dtype, broadcast_heads=bool(config.get("zero_stride_q_do_heads", False))
    )
    templates = (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v))
    _, _, _, bundle = dispatch._flex_attn_bwd(
        q,
        k,
        v,
        o,
        do,
        lse,
        softmax_scale=scale,
        block_sparse_tensors=outer,
        deterministic=bool(config["deterministic"]),
        _compile_only=True,
        _compile_outputs=templates,
        **plan_args,
    )
    source = _workspace_state(bundle, q, k, v)
    tirx = _workspace_state(bundle, q, k, v)
    inputs = {
        "q": q,
        "k": k,
        "v": v,
        "o": o,
        "do": do,
        "lse": lse,
        "endpoints": endpoints,
        "outer": outer,
        "cu_q": cu_q,
        "cu_k": cu_k,
        "scale": scale,
        "plan_args": plan_args,
    }
    immutable = {
        name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)
    }
    data = {
        "config": dict(config),
        "inputs": inputs,
        "immutable": immutable,
        "source_bundle": bundle,
        "source": source,
        "tirx": tirx,
        "derived": {
            "q_lengths": q_lengths,
            "k_lengths": k_lengths,
            "max_q": max(q_lengths),
            "max_k": max(k_lengths),
        },
        "oracle": None,
    }
    _prepare_target_workspace(data)
    data["tensor_maps"] = _build_maps(data)
    if not ptxas._tirx_ptx92_audit:
        raise RuntimeError("the source kernel did not pass through the audited PTX 9.2 assembler")
    return data


def source_launch(data):
    dispatch = load_reference_module("cudnn.flex_attention.dispatch")
    inputs = data["inputs"]
    state = data["source"]
    _reset_source_state(state)
    return dispatch._flex_attn_bwd(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["o"],
        inputs["do"],
        inputs["lse"],
        softmax_scale=inputs["scale"],
        block_sparse_tensors=inputs["outer"],
        deterministic=bool(data["config"]["deterministic"]),
        _compiled=data["source_bundle"],
        _preallocated=_as_preallocated(state),
        **inputs["plan_args"],
    )


def source_main_launch(data, *, benchmark=False):
    dispatch = load_reference_module("cudnn.flex_attention.dispatch")
    inputs = data["inputs"]
    state = data["source"]
    bundle = data["source_bundle"]
    nested = inputs["outer"].bwd_tensors
    runtime_plan = dispatch._block_sparse_runtime_tuple(nested)

    def launch():
        semaphores = (
            _next_benchmark_semaphores(state)
            if benchmark
            else (state["dq_semaphore"], state["dk_semaphore"], state["dv_semaphore"])
        )
        bundle.main(
            inputs["q"].detach(),
            inputs["k"].detach(),
            inputs["v"].detach(),
            inputs["do"],
            state["lse_log2"],
            state["dpsum"],
            state["dq_accum"],
            state["dk_accum"] if state["dk_accum"] is not None else state["dk"],
            state["dv_accum"] if state["dv_accum"] is not None else state["dv"],
            inputs["scale"],
            inputs["cu_q"],
            inputs["cu_k"],
            *semaphores,
            runtime_plan,
        )

    return launch


def target_launch(executable, data, *, benchmark=False):
    config = data["config"]
    inputs = data["inputs"]
    state = data["tirx"]
    nested = inputs["outer"].bwd_tensors
    maps = data["tensor_maps"]
    dummy_i32 = torch.zeros((1,), dtype=torch.int32, device="cuda")
    dummy_u32 = torch.zeros((1,), dtype=torch.uint32, device="cuda")

    def flat(value, *, dtype=torch.int32):
        dummy = dummy_u32 if dtype == torch.uint32 else dummy_i32
        return _tensor_or_dummy(value, dtype=dtype, dummy=dummy)

    static_plan_tail = (
        flat(inputs["cu_q"]),
        flat(inputs["cu_k"]),
        flat(nested.mask_block_cnt),
        flat(nested.mask_block_idx),
        flat(nested.full_block_cnt),
        flat(nested.full_block_idx),
        flat(nested.cu_total_m_blocks),
        flat(nested.dq_write_order),
        flat(nested.dq_write_order_full),
        flat(nested.mask_block_offset),
        flat(nested.full_block_offset),
        flat(nested.mask_block_masks, dtype=torch.uint32),
        flat(nested.sequence_desc),
        flat(nested.fwd_work_desc),
    )
    cooperative = (int(config["head_dim"]), int(config["head_dim_v"])) in ((128, 128), (192, 128))

    def launch():
        semaphores = (
            _next_benchmark_semaphores(state)
            if benchmark
            else (state["dq_semaphore"], state["dk_semaphore"], state["dv_semaphore"])
        )
        plan_tail = (
            *static_plan_tail,
            *(flat(semaphore) for semaphore in semaphores),
            state["combined"],
            inputs["scale"],
        )
        if cooperative:
            executable(
                maps["q"].ptr,
                maps["qt"].ptr,
                maps["k"].ptr,
                maps["kt"].ptr,
                maps["v"].ptr,
                maps["do"].ptr,
                maps["dot"].ptr,
                maps["dv"].ptr,
                maps["dk"].ptr,
                state["dk"].reshape(-1),
                state["dv"].reshape(-1),
                *plan_tail,
            )
        else:
            executable(
                maps["q"].ptr,
                maps["k"].ptr,
                maps["v"].ptr,
                maps["do"].ptr,
                maps["dv"].ptr,
                maps["dk"].ptr,
                state["dk"].reshape(-1),
                state["dv"].reshape(-1),
                *plan_tail,
            )

    launch.tensor_maps = maps
    return launch


def finish_target(data):
    config = data["config"]
    inputs = data["inputs"]
    state = data["tirx"]
    bundle = data["source_bundle"]
    d = _roundup(int(config["head_dim"]), 16)
    dv = _roundup(int(config["head_dim_v"]), 16)
    dq_count = state["dq_accum"].numel()
    dq_accum = state["combined"][state["dq_base"] : state["dq_base"] + dq_count].reshape_as(
        state["dq_accum"]
    )
    bundle.post_dq(
        dq_accum, state["dq"], inputs["scale"], inputs["cu_q"], int(data["derived"]["max_q"])
    )
    if state["dk_accum"] is not None:
        dk_count = state["dk_accum"].numel()
        dv_count = state["dv_accum"].numel()
        dk_accum = state["combined"][state["dk_base"] : state["dk_base"] + dk_count].reshape_as(
            state["dk_accum"]
        )
        dv_accum = state["combined"][state["dv_base"] : state["dv_base"] + dv_count].reshape_as(
            state["dv_accum"]
        )
        bundle.post_dk(
            dk_accum, state["dk"], inputs["scale"], inputs["cu_k"], int(data["derived"]["max_k"])
        )
        bundle.post_dv(dv_accum, state["dv"], 1.0, inputs["cu_k"], int(data["derived"]["max_k"]))


def _logical_sample(tensor, sample, lengths):
    if tensor.ndim == 3:
        begin = sum(lengths[:sample])
        return tensor[begin : begin + lengths[sample]]
    return tensor[sample, : lengths[sample]]


def analytic_oracle(data):
    if data["oracle"] is not None:
        return data["oracle"]
    config = data["config"]
    inputs = data["inputs"]
    q_lengths = data["derived"]["q_lengths"]
    k_lengths = data["derived"]["k_lengths"]
    q_heads = int(config["num_q_heads"])
    kv_heads = int(config["num_kv_heads"])
    ratio = q_heads // kv_heads
    scale = float(inputs["scale"])
    endpoint_offset = 0
    dq_parts = []
    dk_parts = []
    dv_parts = []
    for sample, (q_length, k_length) in enumerate(zip(q_lengths, k_lengths)):
        q = _logical_sample(inputs["q"], sample, q_lengths).double()
        k = _logical_sample(inputs["k"], sample, k_lengths).double()
        v = _logical_sample(inputs["v"], sample, k_lengths).double()
        do = _logical_sample(inputs["do"], sample, q_lengths).double()
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        for head in range(q_heads):
            kv_head = head // ratio
            mask_head = head if inputs["endpoints"].shape[0] > 1 else 0
            ep = inputs["endpoints"][mask_head, :, endpoint_offset : endpoint_offset + q_length]
            key = torch.arange(k_length, device="cuda")[None, :]
            visible = key < ep[0, :, None]
            for index in range(1, ep.shape[0], 2):
                visible |= (key >= ep[index, :, None]) & (key < ep[index + 1, :, None])
            scores = torch.matmul(q[:, head], k[:, kv_head].transpose(0, 1)) * scale
            scores = scores.masked_fill(~visible, float("-inf"))
            probabilities = torch.softmax(scores, dim=-1)
            dprob = torch.matmul(do[:, head], v[:, kv_head].transpose(0, 1))
            delta = (probabilities * dprob).sum(dim=-1, keepdim=True)
            ds = probabilities * (dprob - delta)
            dq[:, head] = torch.matmul(ds, k[:, kv_head]) * scale
            dk[:, kv_head].add_(torch.matmul(ds.transpose(0, 1), q[:, head]) * scale)
            dv[:, kv_head].add_(torch.matmul(probabilities.transpose(0, 1), do[:, head]))
        dq_parts.append(dq)
        dk_parts.append(dk)
        dv_parts.append(dv)
        endpoint_offset += q_length
    if inputs["q"].ndim == 4:
        dq_value = torch.stack(dq_parts)
        dk_value = torch.stack(dk_parts)
        dv_value = torch.stack(dv_parts)
    else:
        dq_value = torch.cat(dq_parts)
        dk_value = torch.cat(dk_parts)
        dv_value = torch.cat(dv_parts)
    data["oracle"] = {"dq": dq_value, "dk": dk_value, "dv": dv_value}
    return data["oracle"]


def _assert_close(actual, expected, name, dtype, *, source_pair):
    if not torch.equal(torch.isnan(actual), torch.isnan(expected)):
        raise AssertionError(f"{name}: NaN classification mismatch")
    if not torch.equal(torch.isposinf(actual), torch.isposinf(expected)):
        raise AssertionError(f"{name}: +Inf classification mismatch")
    if not torch.equal(torch.isneginf(actual), torch.isneginf(expected)):
        raise AssertionError(f"{name}: -Inf classification mismatch")
    field = name.rsplit(" ", 1)[-1]
    if dtype == "float16":
        atol, rtol = (1e-3, 2e-3) if field != "dv" else (3e-2, 4e-3)
    else:
        atol, rtol = (4e-3, 9e-3) if field != "dv" else (6e-2, 1.2e-2)
    if source_pair:
        atol *= 1.05
        rtol *= 1.05
    try:
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=atol, rtol=rtol, equal_nan=False
        )
    except AssertionError as error:
        difference = (actual.float() - expected.float()).abs()
        raise AssertionError(
            f"{name}: max_abs={difference.max().item():.9g}, "
            f"mean_abs={difference.mean().item():.9g}; {error}"
        ) from error


def _validate_redzones(state, owner):
    for name in ("dq", "dk", "dv"):
        storage = state[name + "_storage"]
        expected = torch.tensor(_REDZONE_NARROW, dtype=storage.dtype, device=storage.device)
        if not bool(torch.all(storage[:_REDZONE_ELEMENTS] == expected)):
            raise AssertionError(f"{owner}.{name}: leading redzone modified")
        if not bool(torch.all(storage[-_REDZONE_ELEMENTS:] == expected)):
            raise AssertionError(f"{owner}.{name}: trailing redzone modified")
    for name, storage in state["workspace_guards"].items():
        expected = torch.tensor(
            _REDZONE_F32 if storage.dtype == torch.float32 else 12345,
            dtype=storage.dtype,
            device=storage.device,
        )
        if not bool(torch.all(storage[:_REDZONE_ELEMENTS] == expected)):
            raise AssertionError(f"{owner}.{name}: leading redzone modified")
        if not bool(torch.all(storage[-_REDZONE_ELEMENTS:] == expected)):
            raise AssertionError(f"{owner}.{name}: trailing redzone modified")
    if "combined_storage" in state:
        storage = state["combined_storage"]
        expected = torch.tensor(_REDZONE_F32, dtype=storage.dtype, device=storage.device)
        if not bool(torch.all(storage[:_REDZONE_ELEMENTS] == expected)):
            raise AssertionError(f"{owner}.combined: leading redzone modified")
        if not bool(torch.all(storage[-_REDZONE_ELEMENTS:] == expected)):
            raise AssertionError(f"{owner}.combined: trailing redzone modified")


def validate_outputs(data, *, with_oracle=True):
    dtype = data["config"]["dtype"]
    source = data["source"]
    target = data["tirx"]
    for name in ("dq", "dk", "dv"):
        _assert_close(target[name], source[name], f"tirx/source {name}", dtype, source_pair=True)
    if with_oracle:
        oracle = analytic_oracle(data)
        for owner, state in (("tirx", target), ("source", source)):
            for name in ("dq", "dk", "dv"):
                _assert_close(
                    state[name].double(),
                    oracle[name],
                    f"{owner}/oracle {name}",
                    dtype,
                    source_pair=False,
                )
    for name, expected in data["immutable"].items():
        if not torch.equal(data["inputs"][name], expected):
            raise AssertionError(f"immutable input {name} was modified")
    _validate_redzones(target, "tirx")
    _validate_redzones(source, "source")
