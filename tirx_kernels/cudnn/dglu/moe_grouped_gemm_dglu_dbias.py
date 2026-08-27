# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""MoE BF16 grouped GEMM with a fused dGLU backward epilogue.

Upstream source:
``python/cudnn/gemm/cutedsl/grouped/dglu/moe_grouped_gemm_dglu_dbias.py``.

Per expert, over its 256-aligned row range, the kernel forms
``alpha^2 * A @ B^T`` in BF16, differentiates the GLU whose forward
pre-activation is the interleaved ``C`` tensor, and writes the two gradients
back into ``C``'s 32-column interleaving together with the per-row ``dprob`` and
the optional per-expert ``dbias``.
"""

from ._moe_grouped_gemm_dglu_dbias import data as _data
from ._moe_grouped_gemm_dglu_dbias import kernel as _kernel
from ._moe_grouped_gemm_dglu_dbias import spec as _spec

KERNEL_META = {
    "name": "cudnn_sm100_moe_grouped_gemm_dglu_dbias",
    "category": "cudnn",
    "compute_capability": 10,
}

CONFIGS = _spec.correctness_configs()
BENCH_CONFIGS = _spec.benchmark_configs()


def _without_label(config):
    return {key: value for key, value in config.items() if key != "label"}


def get_kernel(**config):
    """Return the launch sequence for one static specialization."""
    return _kernel.get_kernel(**config)


def prepare_data(**config):
    """Allocate one input set shared by TIRx, the upstream kernel, and the oracle."""
    return _data.prepare_data(**config)


def run_test(**config):
    """Compare TIRx with the pinned upstream kernel and the FP32 oracle."""
    import torch

    from tirx_kernels.runner import compile_kernel

    kernel_config = _without_label(config)
    data = prepare_data(**kernel_config)
    executables = [compile_kernel(func) for func in get_kernel(**kernel_config)]
    tirx_launch = _data.tirx_launch(executables, data)
    tirx_launch()
    if _spec.upstream_disagrees_with_reference(kernel_config):
        torch.cuda.synchronize()
        _data.validate_outputs(data, sources=("tirx",))
    else:
        source_launch = _data.compile_reference(data)
        source_launch()
        torch.cuda.synchronize()
        # The upstream kernel is the arbiter; the FP32 oracle runs only on the
        # rows upstream cannot arbitrate (the sources=("tirx",) branch above).
        _data.validate_outputs(data, sources=("tirx", "source"), with_oracle=False)
    return {"tokens": data["derived"]["tokens_total"], "N": data["derived"]["N"]}


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
    """Validate once against the upstream kernel, then time pure launches."""
    import torch

    from tirx_kernels.runner import bench, external_references_enabled

    config = {**prepared["config"], **kwargs}
    kernel_config = _without_label(config)
    data = prepare_data(**kernel_config)
    tirx_launch = _data.tirx_launch(prepared["executables"], data)
    tirx_launch()
    torch.cuda.synchronize()
    with_source = external_references_enabled()
    references = None
    if with_source:
        source_launch = _data.compile_reference(data)
        source_launch()
        torch.cuda.synchronize()
        references = {"cudnn_frontend": lambda: source_launch}
    # The benchmark shapes make the FP32 oracle far more expensive than the
    # kernels being timed; agreement with the upstream kernel is the check here,
    # and ``run_test`` carries the oracle over the correctness matrix.
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
