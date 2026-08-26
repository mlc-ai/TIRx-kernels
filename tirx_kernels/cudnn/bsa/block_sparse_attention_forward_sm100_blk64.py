# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ 7b5327b32907b9dd21d85a393d62f9573d7f0116), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Block-sparse attention forward pass using the cuDNN Frontend SM100 blk64 path.

Upstream sources:
``python/cudnn/block_sparse_attention/csrc/fwd/sm100_blk64/bsa_fwd_sm100.py``,
``python/cudnn/block_sparse_attention/csrc/fwd/sm100_blk64/bsa_fwd_helpers.py``,
``python/cudnn/block_sparse_attention/csrc/fwd/sm100_blk64/bsa_fwd_combine.py``, and
``python/cudnn/block_sparse_attention/_interface.py``.
"""

from ._block_sparse_attention_forward_sm100_blk64 import data as _data
from ._block_sparse_attention_forward_sm100_blk64 import kernel as _kernel
from ._block_sparse_attention_forward_sm100_blk64 import reference as _reference
from ._block_sparse_attention_forward_sm100_blk64 import spec as _spec

KERNEL_META = {
    "name": "cudnn_sm100_bsa_forward_blk64",
    "category": "cudnn",
    "compute_capability": 10,
}

CONFIGS = _spec.correctness_configs()
BENCH_CONFIGS = _spec.benchmark_configs()


def _without_label(config):
    return {key: value for key, value in config.items() if key != "label"}


def get_kernel(**config):
    return _kernel.get_kernel(**config)


def prepare_data(**config):
    return _data.prepare_data(**config)


def run_test(**config):
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
    if external_references_enabled():
        source_launch = _reference.compile_reference(data)
        source_launch()
        torch.cuda.synchronize()
        _data.validate_outputs(data, sources=("tirx", "source"), with_oracle=False)
        references = {"cudnn_frontend": lambda: source_launch}
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
