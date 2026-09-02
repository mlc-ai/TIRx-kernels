# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""SM100 blk128 block-sparse attention backward three-kernel program."""

from ._block_sparse_attention_backward_sm100_blk128 import data as _data
from ._block_sparse_attention_backward_sm100_blk128 import kernel as _kernel
from ._block_sparse_attention_backward_sm100_blk128 import reference as _reference
from ._block_sparse_attention_backward_sm100_blk128 import spec as _spec

KERNEL_META = {
    "name": "cudnn_sm100_bsa_backward_blk128",
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
    """Return sum_OdO, main backward, and convert in source launch order."""
    return _kernel.get_kernel(**config)


def prepare_data(**config):
    return _data.prepare_data(**config)


def run_test(**config):
    """Validate TIRx against the analytic oracle and usable upstream specializations."""
    import torch

    from tirx_kernels.runner import compile_kernel

    kernel_config = _without_label(config)
    data = prepare_data(**kernel_config)
    executables = [compile_kernel(func) for func in get_kernel(**kernel_config)]
    tirx_launch = _data.tirx_launch(executables, data)
    tirx_launch()
    source_launch = _reference.compile_reference(data)
    source_launch()
    torch.cuda.synchronize()
    _data.validate_outputs(data, sources=("tirx", "source"))


def prepare_bench(**config):
    from tirx_kernels.runner import compile_kernel, prepared_gpu_benchmark

    kernel_config = _without_label(config)
    state = {
        "config": kernel_config,
        "executables": [compile_kernel(func) for func in get_kernel(**kernel_config)],
    }
    return prepared_gpu_benchmark(run_gpu, state)


def run_gpu(prepared, *, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **kwargs):
    import torch

    from tirx_kernels.runner import bench, external_references_enabled

    config = {**prepared["config"], **kwargs}
    data = prepare_data(**_without_label(config))
    tirx_launch = _data.tirx_launch(prepared["executables"], data)
    tirx_launch()
    torch.cuda.synchronize()
    references = None
    with_source = external_references_enabled()
    if with_source:
        source_launch = _reference.compile_reference(data)
        source_launch()
        torch.cuda.synchronize()
        references = {"cudnn_frontend": lambda: source_launch}
    _data.validate_outputs(
        data, sources=("tirx", "source") if with_source else ("tirx",), with_oracle=False
    )
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
