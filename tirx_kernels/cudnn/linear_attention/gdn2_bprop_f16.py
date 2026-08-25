# This file is a TIRx port of code from cuDNN Frontend
# (https://github.com/NVIDIA/cudnn-frontend @ aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Blackwell BF16 Gated Delta Net v2 backward kernel.

Upstream source:
``python/cudnn/linear_attention/frost/kernel/gdn2_bprop_f16.py``
(``prologue_kernel``, ``kernel``, and the two-launch ``run_bwd`` entry).

GDN-2 carries a per-key-channel erase gate ``beta`` and a per-value-channel
write gate ``w``, so ``K_decay`` folds beta into the operand, ``Y`` uses
``w * v``, and the kernel emits ``dw`` alongside a per-channel ``dbeta``.
"""

import tirx_kernels.kern as K

KERNEL_META = {"name": "cudnn_sm100_gdn2_bprop_f16", "category": "cudnn", "compute_capability": 10}

CONFIGS = [
    {"label": "basic", "seq_lens": (64,), "heads": 1},
    {"label": "tail", "seq_lens": (17, 0, 33), "heads": 2},
    {"label": "grouped", "seq_lens": (64,), "heads": 4, "q_heads": 4, "k_heads": 1, "v_heads": 1},
    {"label": "l2norm", "seq_lens": (128,), "heads": 2, "l2norm": True},
    {"label": "safe_gate", "seq_lens": (64,), "heads": 2, "safe_gate": True},
    {"label": "beta_sigmoid", "seq_lens": (128,), "heads": 2, "beta_sigmoid": True},
    {
        "label": "state",
        "seq_lens": (32, 48),
        "heads": 2,
        "use_initial_state": True,
        "use_dstate_in": True,
        "use_dstate0": True,
    },
    {"label": "dynamic", "seq_lens": (32, 48), "heads": 2, "dynamic_scheduler": True},
    {"label": "order_scratch", "seq_lens": (32, 48), "heads": 2, "run_order": True},
    {
        "label": "order_generate",
        "seq_lens": (32, 48),
        "heads": 2,
        "run_order": True,
        "order_generate": True,
    },
]

# The performance matrix spans every accepted specialization at production scale
# plus the upstream benchmark's batch and sequence-length sweep, which runs at 64
# heads with the fused q/k L2 norm enabled. Checkpoint construction stays outside
# timing on both sides.
BENCH_CONFIGS = [
    {"label": "perf_basic_b1_s2048_h16", "seq_lens": (2048,), "heads": 16},
    {"label": "perf_tail_b2_ragged_h16", "seq_lens": (2047, 4093), "heads": 16},
    {
        "label": "perf_grouped_b1_s8192_hq64_hk16_hv16",
        "seq_lens": (8192,),
        "heads": 64,
        "q_heads": 64,
        "k_heads": 16,
        "v_heads": 16,
    },
    {"label": "perf_l2_b1_s8192_h16", "seq_lens": (8192,), "heads": 16, "l2norm": True},
    {"label": "perf_l2_b4_s8192_h64", "seq_lens": (8192,) * 4, "heads": 64, "l2norm": True},
    {"label": "perf_safe_b1_s8192_h16", "seq_lens": (8192,), "heads": 16, "safe_gate": True},
    {"label": "perf_beta_b1_s8192_h16", "seq_lens": (8192,), "heads": 16, "beta_sigmoid": True},
    {
        "label": "perf_state_b1_s8192_h16",
        "seq_lens": (8192,),
        "heads": 16,
        "use_initial_state": True,
        "use_dstate_in": True,
        "use_dstate0": True,
    },
    {
        "label": "perf_dynamic_b4_s2048_h64",
        "seq_lens": (2048,) * 4,
        "heads": 64,
        "dynamic_scheduler": True,
    },
    {
        "label": "perf_order_scratch_b4_s2048_h64",
        "seq_lens": (2048,) * 4,
        "heads": 64,
        "run_order": True,
    },
    {
        "label": "perf_order_generate_b4_s2048_h64",
        "seq_lens": (2048,) * 4,
        "heads": 64,
        "run_order": True,
        "order_generate": True,
    },
    # Upstream benchmark sweep: sequence length at batch 4, then batch at 8192.
    {"label": "perf_l2_b4_s2048_h64", "seq_lens": (2048,) * 4, "heads": 64, "l2norm": True},
    {"label": "perf_l2_b4_s4096_h64", "seq_lens": (4096,) * 4, "heads": 64, "l2norm": True},
    {"label": "perf_l2_b4_s16384_h64", "seq_lens": (16384,) * 4, "heads": 64, "l2norm": True},
    {"label": "perf_l2_b4_s32768_h64", "seq_lens": (32768,) * 4, "heads": 64, "l2norm": True},
    {"label": "perf_l2_b1_s8192_h64", "seq_lens": (8192,), "heads": 64, "l2norm": True},
    {"label": "perf_l2_b2_s8192_h64", "seq_lens": (8192,) * 2, "heads": 64, "l2norm": True},
    {"label": "perf_l2_b8_s8192_h64", "seq_lens": (8192,) * 8, "heads": 64, "l2norm": True},
    {"label": "perf_l2_b16_s8192_h64", "seq_lens": (8192,) * 16, "heads": 64, "l2norm": True},
]


# Frozen specialization, from ``gdn2_bprop_config.py``.
_BT = 16
_DK = 128
_DV = 128


def _normalized_config(config):
    """Fill the accepted-branch defaults for one config entry."""
    resolved = {key: value for key, value in config.items() if key != "label"}
    resolved["seq_lens"] = tuple(resolved.get("seq_lens", (64,)))
    resolved.setdefault("heads", 1)
    heads = resolved["heads"]
    resolved.setdefault("q_heads", heads)
    resolved.setdefault("k_heads", heads)
    resolved.setdefault("v_heads", heads)
    resolved.setdefault("dtype", "bfloat16")
    resolved.setdefault("num_sms", 148)
    resolved.setdefault("scale", 1.0 / (_DK**0.5))
    for key in (
        "l2norm",
        "safe_gate",
        "beta_sigmoid",
        "use_initial_state",
        "use_dstate_in",
        "use_dstate0",
        "dynamic_scheduler",
        "run_order",
        "order_generate",
    ):
        resolved.setdefault(key, False)
    return resolved


def _make_prologue(config):
    """Descriptor-building launch: 1 CTA of 1024 threads.

    Builds the fourteen runtime-patched TMA descriptor arrays and, when the work
    order is generated here rather than by the recompute pass, the longest-
    processing-time work-item table.
    """

    @K.kernel(warps=32, arch="sm_100a", grid=1)
    def prologue():
        # --- kernel sketch starts here ---
        pass

    return prologue


def _make_main(config):
    """Persistent backward launch: one CTA per SM, 512 threads in seven roles."""

    @K.kernel(warps=16, arch="sm_100a", min_blocks_per_sm=1, grid=1)
    def main():
        # --- kernel sketch starts here ---
        pass

    return main


def get_kernel(**config):
    resolved = _normalized_config(config)
    return [_make_prologue(resolved).func, _make_main(resolved).func]


def prepare_data(**config):
    raise NotImplementedError("gdn2 backward data construction is not implemented yet")


def run_test(**config):
    raise NotImplementedError("gdn2 backward correctness is not implemented yet")


def prepare_bench(**config):
    raise NotImplementedError("gdn2 backward benchmark setup is not implemented yet")


def run_bench(*, warmup=None, repeat=None, timer=None, rounds=1, cooldown_s=0.0, **config):
    raise NotImplementedError("gdn2 backward benchmarking is not implemented yet")


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_test",
]
