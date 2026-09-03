#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Combine the two same-device Thor classic-kernel campaigns."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

_CATEGORY_ORDER = (
    "Dense GEMM",
    "Quantized GEMM",
    "Attention",
    "Normalization",
    "Activation / quantization",
    "TopK",
    "Recurrent / SSM",
)


def _is_ours(name: str) -> bool:
    return name in {"tir", "tirx"} or name.startswith(("tir-", "tirx-"))


def _category(kernel: str) -> str:
    if kernel == "fp16_bf16_gemm":
        return "Dense GEMM"
    if kernel == "nvfp4_gemm":
        return "Quantized GEMM"
    if kernel == "flash_attention4":
        return "Attention"
    if "norm" in kernel or "layernorm" in kernel:
        return "Normalization"
    if kernel == "act_and_mul" or "quantize" in kernel:
        return "Activation / quantization"
    if "topk" in kernel:
        return "TopK"
    if kernel.startswith(("gdn", "recurrent_kda", "selective_state_update")):
        return "Recurrent / SSM"
    raise ValueError(f"no category for {kernel}")


def _cv(row: dict, implementation: str) -> float:
    samples = [float(value) for value in row["round_samples"][implementation]]
    mean = statistics.fmean(samples)
    return 0.0 if len(samples) < 2 or mean == 0 else 100.0 * statistics.pstdev(samples) / mean


def _geomean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _primary_reference(row: dict) -> tuple[str, float]:
    implementations = row["impls"]
    if row["kernel"] == "flash_attention4":
        name = "flashattn_fa4_cutedsl"
        if name not in implementations:
            raise ValueError(
                f"FA4 report requires the like-for-like {name} baseline for "
                f"{row['kernel']}/{row['config']}, got {list(implementations)}"
            )
        return name, float(implementations[name])
    if row["kernel"] == "nvfp4_gemm":
        return "flashinfer", float(implementations["flashinfer"])
    references = [
        (name, float(value)) for name, value in implementations.items() if not _is_ours(name)
    ]
    if len(references) != 1:
        raise ValueError(
            f"expected one primary reference for {row['kernel']}/{row['config']}, "
            f"got {list(implementations)}"
        )
    return references[0]


def _render(representative: dict, additions: dict, paths: tuple[Path, Path]) -> str:
    runs = (representative, additions)
    for run in runs:
        if not run.get("references_enabled"):
            raise ValueError("every input run must enable references")
        failed = [row for row in run["results"] if row.get("status") != "ok"]
        if failed:
            raise ValueError(f"input run contains {len(failed)} failed rows")

    rows = []
    for run in runs:
        for row in run["results"]:
            ours = [(name, float(value)) for name, value in row["impls"].items() if _is_ours(name)]
            if len(ours) != 1:
                raise ValueError(f"expected one TIRx implementation for {row['kernel']}")
            reference = _primary_reference(row)
            speedup = reference[1] / ours[0][1]
            rows.append(
                {
                    "raw": row,
                    "category": _category(row["kernel"]),
                    "ours": ours[0],
                    "reference": reference,
                    "speedup": speedup,
                    "ours_cv": _cv(row, ours[0][0]),
                    "reference_cv": _cv(row, reference[0]),
                }
            )

    speedups = [row["speedup"] for row in rows]
    faster = sum(value > 1.05 for value in speedups)
    slower = sum(value < 0.95 for value in speedups)
    within = len(rows) - faster - slower
    noisy = sum(max(row["ours_cv"], row["reference_cv"]) > 10.0 for row in rows)
    date = representative["results"][0]["started_at"][:10]
    additions_git = additions.get("git") or {}
    additions_protocol = additions["results"][0]["benchmark_protocol"]
    flashinfer = (additions.get("baselines") or {}).get("flashinfer") or {}
    flashattn = (representative.get("baselines") or {}).get("flash_attn") or {}

    lines = [
        "# NVIDIA Thor classic-kernel same-device baselines",
        "",
        f"Measured on {date} on one NVIDIA Jetson AGX Thor Developer Kit. The table combines "
        "the original representative campaign with the classic-family additions. Every speedup "
        "is `reference latency / TIRx latency`; values above 1.0 favor TIRx.",
        "",
        "## Summary",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Measured exact-shape rows | **{len(rows)}/{len(rows)} passed** |",
        "| Main classic families with at least one numeric row | **12/17** |",
        f"| TIRx faster by more than 5% | **{faster}/{len(rows)}** |",
        f"| Within 5% | **{within}/{len(rows)}** |",
        f"| Reference faster by more than 5% | **{slower}/{len(rows)}** |",
        f"| Geometric-mean TIRx speedup | **{_geomean(speedups):.3f}x** |",
        f"| Rows with either CV above 10% | **{noisy}/{len(rows)}** |",
        "",
        "The mixed geomean is descriptive only: it gives one vote to each selected workload, "
        "not to each model invocation. The per-row numbers are the result to use.",
        "The source-to-benchmark mapping for every row is audited in "
        "[THOR_SOURCE_BENCHMARK_AUDIT.md](THOR_SOURCE_BENCHMARK_AUDIT.md).",
        "",
        "## Complete numeric table",
        "",
        "| Family | Kernel / config | TIRx µs | CV | Reference | Reference µs | CV | Speedup |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    category_index = {name: index for index, name in enumerate(_CATEGORY_ORDER)}
    for item in sorted(
        rows,
        key=lambda value: (
            category_index[value["category"]],
            value["raw"]["kernel"],
            value["raw"]["config"],
        ),
    ):
        raw = item["raw"]
        lines.append(
            f"| {item['category']} | `{raw['kernel']}/{raw['config']}` | "
            f"{item['ours'][1]:.3f} | {item['ours_cv']:.1f}% | "
            f"`{item['reference'][0]}` | {item['reference'][1]:.3f} | "
            f"{item['reference_cv']:.1f}% | **{item['speedup']:.3f}x** |"
        )
    nvfp4 = next(item for item in rows if item["raw"]["kernel"] == "nvfp4_gemm")
    cublaslt_us = float(nvfp4["raw"]["impls"]["cublaslt_nvfp4"])
    cublaslt_cv = _cv(nvfp4["raw"], "cublaslt_nvfp4")
    attention = next(item for item in rows if item["raw"]["kernel"] == "flash_attention4")

    lines.extend(
        [
            "",
            "The NVFP4 row also measured cuBLASLt at "
            f"**{cublaslt_us:.3f} µs** (CV {cublaslt_cv:.1f}%), or "
            f"**{cublaslt_us / nvfp4['ours'][1]:.3f}x** relative to TIRx. FlashInfer is retained "
            "as that row's primary baseline to follow the requested priority.",
            "",
            "The attention row uses upstream FA4 CuTeDSL as its primary baseline. On the same "
            "row, FlashInfer CuTeDSL measured "
            f"**{float(attention['raw']['impls']['flashinfer_cutedsl']):.3f} µs** and the legacy "
            "FlashInfer FA2 control measured "
            f"**{float(attention['raw']['impls']['flashinfer_fa2']):.3f} µs**; neither secondary "
            "control enters the geomean.",
            "",
            "## Main-family coverage without a publishable Thor number",
            "",
            "| Family | Status | Reason |",
            "|---|---|---|",
            "| Dense/batched FP8 GEMM | N/A | The exact pinned DeepGEMM entry rejects "
            "compute capability 11; FlashInfer `mm_fp8` has a different low-latency/scale "
            "contract. |",
            "| Grouped GEMM | N/A | The exact pinned DeepGEMM host dispatch rejects Thor; "
            "no contract-matched independent launch has passed yet. |",
            "| Fused MoE | N/A | `sm100_fp8_fp4_mega_moe` remains blocked by the "
            "compute-10-only DeepGEMM scale-layout host path; some cases also require "
            "multiple GPUs. |",
            "| Block-sparse / sparse-MLA attention | N/A | TIRx is numerically validated, "
            "but the inspected exact external source dispatchers have no compute-11 Thor path. |",
            "| MQA logits / indexer | N/A | TIRx is numerically validated; DeepGEMM and "
            "SGLang CuTeDSL timing peers are disabled on Thor by their compute-10 host dispatch. |",
            "",
            "These are unavailable comparisons, not zero performance and not failed TIRx "
            "correctness. They must not be included in the geomean until an exact independent "
            "Thor baseline launches.",
            "",
            "## Measurement provenance",
            "",
            "| Field | Value |",
            "|---|---|",
            "| GPU | NVIDIA Jetson AGX Thor Developer Kit, 20 SMs |",
            "| CUDA architecture | `sm_110a` |",
            "| Power mode | `MAXN`; dynamic clocks, because `jetson_clocks` requires root |",
            "| Timer | Proton, cold-L2 per timed iteration |",
            f"| Rounds / aggregation | {additions_protocol['rounds']} / arithmetic mean |",
            f"| Warmup / repeat | {additions_protocol['warmup']} ms / "
            f"{additions_protocol['repeat']} ms per implementation per round |",
            f"| TVM/TIR revision | `{additions_git.get('tir', '-')}` |",
            f"| TIRx-kernels revision | `{additions_git.get('tirx-kernels', '-')}` |",
            f"| FlashInfer version / revision | `{flashinfer.get('version', '-')}` / "
            f"`{flashinfer.get('git_sha', '-')}` |",
            f"| FlashAttention-4 revision | `{flashattn.get('git_sha', '-')}` |",
            "",
            "Rows above 10% CV are retained and visibly flagged by their CV columns. In "
            "particular, the BF16 4096-cube GEMM switched between fast and slow clock regimes "
            "within its five rounds. Its absolute mean and near-threshold ratio should be rerun "
            "with "
            "`sudo jetson_clocks` before publication.",
            "",
            "## Raw evidence",
            "",
            f"- Original representative run: `{paths[0].resolve()}`",
            f"- Classic additions run: `{paths[1].resolve()}`",
            "- Both runs used five round samples per implementation and had no interference "
            "retries in the selected final artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representative-run", type=Path, required=True)
    parser.add_argument("--additions-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("THOR_CLASSIC_BASELINE_RESULTS.md"))
    args = parser.parse_args()
    representative = json.loads(args.representative_run.read_text())
    additions = json.loads(args.additions_run.read_text())
    args.output.write_text(
        _render(representative, additions, (args.representative_run, args.additions_run))
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
