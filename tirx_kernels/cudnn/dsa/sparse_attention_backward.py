# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""DeepSeek Sparse Attention backward pass for SM100.

Upstream source:
``python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100.py``.

Each query token owns one CTA, and its attention heads form the M dimension of the
tensor-core GEMMs, so the top-k KV rows a token selects are gathered once and
amortized across every head. K and V are one shared MQA buffer, which fuses their
gradients: ``dKV`` accumulates ``dV`` over the leading 512 dimensions and ``dK``
over the full head dimension. The pass runs as four kernels -- a delta and
sink-folded-LSE preprocess, the 20-warp main kernel, an FP32-to-element-dtype dKV
conversion, and the attention-sink gradient reduction.
"""

from tirx_kernels.target import prepare_cuda_arch

from ._sparse_attention_backward import data as _data
from ._sparse_attention_backward import kernel as _kernel
from ._sparse_attention_backward import spec as _spec

KERNEL_META = {
    "name": "cudnn_sm100_dsa_sparse_attention_backward",
    "category": "cudnn",
    "runtime_cuda_archs": ["sm_100a", "sm_103a", "sm_107a", "sm_110a"],
    "reference_requirements": (
        {
            "package": "nvidia-cudnn-frontend",
            "git": {
                "url": "https://github.com/NVIDIA/cudnn-frontend.git",
                "commit": "aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5",
            },
            "import": "cudnn",
        },
        {"package": "nvidia-cutlass-dsl", "specifier": "==4.8.0.dev0", "import": "cutlass"},
    ),
}

CONFIGS = _spec.correctness_configs()
BENCH_CONFIGS = _spec.benchmark_configs()


def _without_label(config):
    return {key: value for key, value in config.items() if key != "label"}


def get_kernel(**config):
    """Return the four device functions in the upstream launch order."""
    return _kernel.get_kernel(**config)


def prepare_data(**config):
    """Allocate one input set shared by TIRx, the upstream kernel, and the oracle."""
    return _data.prepare_data(**config)


def run_test(**config):
    """Compare TIRx and the upstream kernel against the FP32 oracle."""
    import torch

    from tirx_kernels.runner import compile_kernel

    kernel_config = _without_label(config)
    data = prepare_data(**kernel_config)
    executables = [compile_kernel(func) for func in get_kernel(**kernel_config)]
    tirx_launch = _data.tirx_launch(executables, data, synchronize_stages=True)
    tirx_launch()
    if prepare_cuda_arch() == "sm_110a":
        torch.cuda.synchronize()
        _data.validate_outputs(data, sources=("tirx",))
    else:
        source_launch = _data.compile_reference(data)
        source_launch()
        torch.cuda.synchronize()
        _data.validate_outputs(data, sources=("tirx", "source"))
    return {
        "seqlen_q": data["derived"]["seqlen_q"],
        "seqlen_kv": data["derived"]["seqlen_kv"],
        "max_topk": data["derived"]["max_topk"],
    }


def prepare_bench(**config):
    """Compile only TIRx; reference imports and CUDA work stay in ``run_gpu``."""
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    kernel_config = _without_label(config)
    state = {
        "config": kernel_config,
        "executables": [compile_kernel(func) for func in get_kernel(**kernel_config)],
    }
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **kwargs):
    """Validate once against the upstream kernel, then time both launch closures."""
    import torch

    from tirx_kernels.runner import bench, external_references_enabled

    config = {**prepared["config"], **kwargs}
    kernel_config = _without_label(config)
    data = prepare_data(**kernel_config)
    tirx_launch = _data.tirx_launch(prepared["executables"], data)
    tirx_launch()
    torch.cuda.synchronize()

    references = None
    if external_references_enabled() and prepare_cuda_arch() != "sm_110a":
        source_launch = _data.compile_reference(data)
        source_launch()
        torch.cuda.synchronize()
        references = {"cudnn_frontend": lambda: source_launch}

    # The oracle is far more expensive than the kernels at the benchmark shapes;
    # ``run_test`` carries it over the correctness matrix instead.
    _data.validate_outputs(data, sources=("tirx",), with_oracle=False)
    return bench(
        {"tirx": tirx_launch},
        references=references,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


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
