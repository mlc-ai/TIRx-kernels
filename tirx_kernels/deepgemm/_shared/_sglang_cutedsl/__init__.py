# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Vendored SGLang CuTeDSL FP8 paged MQA logits reference (Blackwell SM100).

Verbatim copies of ``python/sglang/kernels/ops/attention/cutedsl_fp8_paged_mqa_logits.py``
and ``.../dsa/cutedsl_paged_mqa_logits.py`` from sgl-project/sglang @ c7c03ec53b
(themselves ported from NVIDIA TensorRT-LLM), kept here so the benchmark reference
needs only CUTLASS DSL and torch rather than the SGLang package: the two ``sglang``
imports became a relative import and a local ``is_sm100_supported``, and the
``torch.ops.sglang`` custom-op registration was dropped.
"""
