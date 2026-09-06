# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

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

"""SM100 arbitrary-mask Flex Attention backward main kernel.

The port targets the direct CUTLASS DSL main-kernel ABI; mask planning,
forward/LSE preparation, and gradient postprocessing remain outside the timed
region.
"""

import tirx_kernels.kern as K

from ._flex_attention_backward_sm100 import data as _data
from ._flex_attention_backward_sm100 import kernel as _kernel

KERNEL_LANGUAGE = K

KERNEL_META = {
    "name": "cudnn_sm100_flex_attention_backward",
    "category": "cudnn",
    "runtime_cuda_archs": ["sm_100a"],
    "reference_requirements": (
        {
            "package": "nvidia-cudnn-frontend",
            "git": {
                "url": "https://github.com/NVIDIA/cudnn-frontend.git",
                "commit": "edc7c2833327d699d70b616c6f264b6ae92599b2",
            },
            "import": "cudnn",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.8.0.dev0", "import": "cutlass"},
    ),
}

_HEAD_DIMS = tuple(range(8, 129, 8))
_PRODUCTION_MASKS = (
    "causal",
    "document_causal",
    "local",
    "sink_local",
    "tree_dfs",
    "tree_bfs",
    "longformer",
    "hstu",
)
_INTERACTION_DIMS = (
    (8, 8),
    (8, 128),
    (64, 64),
    (72, 40),
    (120, 104),
    (128, 8),
    (128, 128),
    (192, 128),
)
_INTERACTION_GQA_RATIOS = (2, 4, 8, 16)


def _config(label, **overrides):
    config = {
        "label": label,
        "batch": 1,
        "seqlen_q": 129,
        "seqlen_kv": 257,
        "num_q_heads": 2,
        "num_kv_heads": 2,
        "head_dim": 128,
        "head_dim_v": 128,
        "dtype": "bfloat16",
        "deterministic": False,
        "varlen": False,
        "mask_type": "causal",
        "mask_head_mode": "broadcast",
        "mask_nfunc": 1,
        "softmax_scale": None,
        "zero_stride_q_do_heads": False,
        "zero_stride_kv_heads": False,
        "dpsum_mode": "forward",
        "seed": 310000,
    }
    config.update(overrides)
    return config


def _correctness_configs():
    configs = []
    dimension_pairs = [(d_qk, d_v) for d_qk in _HEAD_DIMS for d_v in _HEAD_DIMS]
    dimension_pairs.append((192, 128))
    for index, (d_qk, d_v) in enumerate(dimension_pairs):
        dtype = "float16" if index % 2 == 0 else "bfloat16"
        configs.append(
            _config(
                f"c{index:03d}_d{d_qk}_dv{d_v}_{dtype}",
                head_dim=d_qk,
                head_dim_v=d_v,
                dtype=dtype,
                seed=310000 + index,
            )
        )

    for dim_index, (d_qk, d_v) in enumerate(_INTERACTION_DIMS):
        ratio = _INTERACTION_GQA_RATIOS[dim_index % len(_INTERACTION_GQA_RATIOS)]
        base = 257 + 4 * dim_index
        configs.extend(
            (
                _config(
                    f"c{base:03d}_d{d_qk}_dv{d_v}_fixed_mha_fp16",
                    batch=1,
                    seqlen_q=127,
                    seqlen_kv=255,
                    num_q_heads=4,
                    num_kv_heads=4,
                    head_dim=d_qk,
                    head_dim_v=d_v,
                    dtype="float16",
                    mask_type="full",
                    softmax_scale=0.125,
                    seed=310000 + base,
                ),
                _config(
                    f"c{base + 1:03d}_d{d_qk}_dv{d_v}_fixed_det_gqa{ratio}_mixed",
                    batch=2,
                    seqlen_q=257,
                    seqlen_kv=385,
                    num_q_heads=2 * ratio,
                    num_kv_heads=2,
                    head_dim=d_qk,
                    head_dim_v=d_v,
                    dtype="bfloat16",
                    deterministic=True,
                    mask_type="mixed",
                    mask_head_mode="per_head",
                    mask_nfunc=19,
                    seed=310000 + base + 1,
                ),
                _config(
                    f"c{base + 2:03d}_d{d_qk}_dv{d_v}_varlen_det_mha_local",
                    batch=3,
                    seqlen_q=(1, 127, 130),
                    seqlen_kv=(129, 255, 258),
                    num_q_heads=2,
                    num_kv_heads=2,
                    head_dim=d_qk,
                    head_dim_v=d_v,
                    dtype="float16",
                    deterministic=True,
                    varlen=True,
                    mask_type="local",
                    mask_head_mode="segment_local",
                    mask_nfunc=3,
                    zero_stride_kv_heads=True,
                    seed=310000 + base + 2,
                ),
                _config(
                    f"c{base + 3:03d}_d{d_qk}_dv{d_v}_varlen_gqa{ratio}_mixed",
                    batch=3,
                    seqlen_q=(96, 65, 1),
                    seqlen_kv=(80, 49, 257),
                    num_q_heads=2 * ratio,
                    num_kv_heads=2,
                    head_dim=d_qk,
                    head_dim_v=d_v,
                    dtype="bfloat16",
                    varlen=True,
                    mask_type="mixed",
                    mask_head_mode="per_head",
                    mask_nfunc=31,
                    zero_stride_q_do_heads=True,
                    dpsum_mode="nonzero",
                    seed=310000 + base + 3,
                ),
            )
        )
    assert len(configs) == 289
    return configs


def _benchmark_configs():
    configs = []
    for d_qk, d_v in ((128, 128), (192, 128)):
        for mask in _PRODUCTION_MASKS:
            index = len(configs)
            configs.append(
                _config(
                    f"p{index:02d}_d{d_qk}_dv{d_v}_{mask}_128k",
                    seqlen_q=131072,
                    seqlen_kv=131072,
                    num_q_heads=4,
                    num_kv_heads=4,
                    head_dim=d_qk,
                    head_dim_v=d_v,
                    dtype="bfloat16",
                    mask_type=mask,
                    seed=311000 + index,
                )
            )

    one_cta_dims = (
        (8, 8, "float16"),
        (8, 128, "bfloat16"),
        (40, 72, "float16"),
        (64, 64, "bfloat16"),
        (72, 40, "bfloat16"),
        (104, 120, "float16"),
        (120, 104, "bfloat16"),
        (128, 8, "float16"),
    )
    for pair_index, (d_qk, d_v, dtype) in enumerate(one_cta_dims):
        mask = _PRODUCTION_MASKS[pair_index]
        fixed_index = len(configs)
        configs.append(
            _config(
                f"p{fixed_index:02d}_d{d_qk}_dv{d_v}_{dtype}_fixed_{mask}",
                seqlen_q=4096,
                seqlen_kv=8192,
                num_q_heads=4,
                num_kv_heads=4,
                head_dim=d_qk,
                head_dim_v=d_v,
                dtype=dtype,
                mask_type=mask,
                seed=311000 + fixed_index,
            )
        )
        varlen_index = len(configs)
        configs.append(
            _config(
                f"p{varlen_index:02d}_d{d_qk}_dv{d_v}_{dtype}_varlen_det_gqa4_{mask}",
                batch=3,
                seqlen_q=(1025, 3073, 4097),
                seqlen_kv=(2049, 4097, 8193),
                num_q_heads=8,
                num_kv_heads=2,
                head_dim=d_qk,
                head_dim_v=d_v,
                dtype=dtype,
                deterministic=True,
                varlen=True,
                mask_type=mask,
                mask_head_mode="segment_local_per_head",
                seed=311000 + varlen_index,
            )
        )

    for d_qk, d_v in ((64, 64), (72, 40), (128, 128), (192, 128)):
        cases = (
            dict(
                dtype="float16",
                batch=2,
                seqlen_q=4097,
                seqlen_kv=8193,
                num_q_heads=4,
                num_kv_heads=4,
                deterministic=True,
                mask_type="sink_local",
            ),
            dict(
                dtype="bfloat16",
                batch=1,
                seqlen_q=8193,
                seqlen_kv=16385,
                num_q_heads=8,
                num_kv_heads=1,
                mask_type="mixed",
                mask_head_mode="per_head",
                mask_nfunc=19,
                softmax_scale=0.125,
            ),
            dict(
                dtype="bfloat16",
                batch=3,
                seqlen_q=(1, 4095, 4097),
                seqlen_kv=(129, 8191, 4097),
                num_q_heads=4,
                num_kv_heads=4,
                varlen=True,
                mask_type="document_causal",
                mask_head_mode="segment_local",
            ),
            dict(
                dtype="float16",
                batch=3,
                seqlen_q=(1025, 2049, 4097),
                seqlen_kv=(2049, 4097, 8193),
                num_q_heads=8,
                num_kv_heads=2,
                deterministic=True,
                varlen=True,
                mask_type="mixed",
                mask_head_mode="segment_local_per_head",
                mask_nfunc=31,
                dpsum_mode="nonzero",
            ),
        )
        for case_index, case in enumerate(cases):
            index = len(configs)
            configs.append(
                _config(
                    f"p{index:02d}_d{d_qk}_dv{d_v}_coverage{case_index}",
                    head_dim=d_qk,
                    head_dim_v=d_v,
                    seed=311000 + index,
                    **case,
                )
            )
    assert len(configs) == 48
    return configs


CONFIGS = _correctness_configs()
BENCH_CONFIGS = _benchmark_configs()
_BENCH_KERNEL_CONFIGS = tuple(
    {key: value for key, value in config.items() if key != "label"} for config in BENCH_CONFIGS
)


def get_kernel(**config):
    """Return the specialized direct main-kernel PrimFunc."""
    return _kernel.get_kernel(**config)


def prepare_data(**config):
    return _data.prepare_data(**config)


_PTXAS_REG_LEVEL = "5"


def _compile_kernel(kernel_config):
    """Compile with the measured source-native ptxas schedule via NVRTC."""
    import os

    from tirx_kernels.runner import compile_kernel

    _data.assert_nvrtc_ptx92()
    previous = os.environ.get("TVM_CUDA_PTXAS_REG_LEVEL")
    os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = _PTXAS_REG_LEVEL
    try:
        return compile_kernel(get_kernel(**kernel_config))
    finally:
        if previous is None:
            os.environ.pop("TVM_CUDA_PTXAS_REG_LEVEL", None)
        else:
            os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = previous


def run_test(**config):
    import torch

    kernel_config = {key: value for key, value in config.items() if key != "label"}
    data = prepare_data(**kernel_config)
    executable = _compile_kernel(kernel_config)
    # Mask planning and preprocessing run through PyTorch, while the compiled
    # TIRx executable launches through TVM.  Complete all setup work before
    # crossing that stream boundary.
    torch.cuda.synchronize()
    target = _data.target_launch(executable, data)
    target()
    # The direct main kernel is launched by TVM; wait before the PyTorch-side
    # gradient postprocess consumes its workspace and direct dK/dV outputs.
    torch.cuda.synchronize()
    _data.finish_target(data)
    _data.source_launch(data)
    torch.cuda.synchronize()
    # The c-prefixed correctness matrix is deliberately small enough for the
    # independent FP64 oracle.  Production benchmark shapes can be 128K x
    # 128K, so validate those directly against the pinned source instead of
    # materializing an O(sequence^2) oracle tensor.
    _data.validate_outputs(data, with_oracle=kernel_config not in _BENCH_KERNEL_CONFIGS)


def prepare_bench(**config):
    """Compile TIRx with default NVRTC before the benchmark GPU stage."""
    import os
    from pathlib import Path

    from tirx_kernels.runner import prepared_gpu_benchmark

    kernel_config = {key: value for key, value in config.items() if key != "label"}
    _data.assert_nvrtc_ptx92()
    capture_dir = os.environ.get("TIRX_FLEX_BWD_TARGET_CAPTURE_DIR")
    if capture_dir is None:
        executable = _compile_kernel(kernel_config)
    else:
        # bench_suite compiles inside its CPU-prepare scope, which calls
        # tvm.support.nvcc.compile_cuda directly.  Capture that returned NVRTC
        # cubin so profiler runs can replay the byte-identical benchmark image.
        import tvm.support.nvcc as nvcc

        original_compile_cuda = nvcc.compile_cuda

        def capture_compile_cuda(source, *args, **kwargs):
            if kwargs.get("compiler") != "nvrtc" or kwargs.get("target_format") != "cubin":
                raise RuntimeError(f"expected offline NVRTC cubin compile, got {kwargs!r}")
            cubin = original_compile_cuda(source, *args, **kwargs)
            destination = Path(capture_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "tvm_kernels.cu").write_text(source)
            (destination / "tvm_kernels.cubin").write_bytes(cubin)
            return cubin

        nvcc.compile_cuda = capture_compile_cuda
        try:
            executable = _compile_kernel(kernel_config)
        finally:
            nvcc.compile_cuda = original_compile_cuda
    state = {"config": kernel_config, "executable": executable}
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **kwargs):
    """Validate once, then time only matching source/TIRx direct main kernels."""
    import torch

    from tirx_kernels.runner import bench, external_references_enabled

    config = {**prepared["config"], **kwargs}
    kernel_config = {key: value for key, value in config.items() if key != "label"}
    data = prepare_data(**kernel_config)

    validation_launch = _data.target_launch(prepared["executable"], data)
    # Mask planning, preprocessing, and the source forward run through
    # PyTorch, while the compiled TIRx executable launches through TVM.
    # Complete setup before crossing that stream boundary.
    torch.cuda.synchronize()
    validation_launch()
    torch.cuda.synchronize()
    _data.finish_target(data)
    _data.source_launch(data)
    torch.cuda.synchronize()
    # The 289-case correctness matrix carries the tight FP64 analytic oracle.
    # Benchmark shapes compare the two PTX-9.2 implementations directly.
    _data.validate_outputs(data, with_oracle=False)

    # Deterministic/GQA kernels mutate caller-owned semaphores. Pre-zero every
    # repetition's slot before timing so neither reset nor postprocessing enters
    # the Proton scope. The bounded ring fails rather than silently wrapping.
    _data.prepare_benchmark_semaphores(data)
    tirx_launch = _data.target_launch(prepared["executable"], data, benchmark=True)
    references = None
    if external_references_enabled():
        references = {"cudnn_frontend": lambda: _data.source_main_launch(data, benchmark=True)}

    # These small timing budgets are intentional: the pre-zeroed semaphore ring
    # is finite, while five bench_suite rounds still provide independent samples.
    if warmup is None:
        warmup = 0
    if repeat is None:
        repeat = 1
    result = bench(
        {"tirx": tirx_launch},
        references=references,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )
    result["toolchain"] = {
        "target_compiler": "nvrtc-13.2",
        "source_assembler": "ptxas-13.2",
        "ptx_isa": "9.2",
    }
    return result


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **config):
    return prepare_bench(**config).run_gpu(
        warmup=warmup, repeat=repeat, timer=timer, rounds=rounds, cooldown_s=cooldown_s
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_gpu",
    "run_test",
]
