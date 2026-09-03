#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Render a same-device Thor TIRx versus native-library comparison."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

_CATEGORY_ORDER = (
    "Attention",
    "Normalization",
    "Activation / quantization",
    "TopK",
    "Recurrent / SSM",
)


def _config(row: dict) -> str:
    return row.get("config") or row.get("label") or "?"


def _is_ours(name: str) -> bool:
    return name in {"tir", "tirx"} or name.startswith(("tir-", "tirx-"))


def _implementations(row: dict) -> tuple[str, float, str, float]:
    implementations = row.get("impls") or {}
    ours = [(name, float(value)) for name, value in implementations.items() if _is_ours(name)]
    references = [
        (name, float(value)) for name, value in implementations.items() if not _is_ours(name)
    ]
    if len(ours) != 1:
        raise ValueError(
            f"expected exactly one TIRx implementation for "
            f"{row.get('kernel')}/{_config(row)}, got {list(implementations)}"
        )
    if row.get("kernel") == "flash_attention4":
        primary = "flashattn_fa4_cutedsl"
        if primary not in implementations:
            raise ValueError(
                f"FA4 report requires the like-for-like {primary} baseline for "
                f"{row.get('kernel')}/{_config(row)}, got {list(implementations)}"
            )
        return *ours[0], primary, float(implementations[primary])
    if len(references) != 1:
        raise ValueError(
            f"expected one reference implementation for {row.get('kernel')}/{_config(row)}, "
            f"got {list(implementations)}"
        )
    return *ours[0], *references[0]


def _category(kernel: str) -> str:
    if kernel == "flash_attention4":
        return "Attention"
    if "rmsnorm" in kernel:
        return "Normalization"
    if kernel == "act_and_mul" or "quantize" in kernel:
        return "Activation / quantization"
    if "topk" in kernel:
        return "TopK"
    if kernel.startswith(("gdn", "recurrent_kda", "selective_state_update")):
        return "Recurrent / SSM"
    raise ValueError(f"representative kernel has no report category: {kernel}")


def _geomean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _cv(row: dict, implementation: str) -> float:
    samples = [float(value) for value in (row.get("round_samples") or {}).get(implementation, [])]
    if len(samples) < 2 or statistics.fmean(samples) == 0:
        return 0.0
    return 100.0 * statistics.pstdev(samples) / statistics.fmean(samples)


def _verdict(speedup: float) -> str:
    if speedup > 1.05:
        return "TIRx faster"
    if speedup < 0.95:
        return "Reference faster"
    return "within 5%"


def _markdown(run: dict, *, run_path: Path) -> str:
    if not run.get("references_enabled"):
        raise ValueError("the run did not enable reference implementations")
    results = run.get("results") or []
    if not results:
        raise ValueError("the run contains no results")
    failed = [row for row in results if row.get("status") != "ok"]
    if failed:
        names = ", ".join(f"{row['kernel']}/{_config(row)}" for row in failed)
        raise ValueError(f"the run contains non-passing rows: {names}")

    rows = []
    by_category: dict[str, list[float]] = defaultdict(list)
    verdicts: dict[str, int] = defaultdict(int)
    for row in results:
        ours, ours_us, reference, reference_us = _implementations(row)
        speedup = reference_us / ours_us
        category = _category(row["kernel"])
        verdict = _verdict(speedup)
        verdicts[verdict] += 1
        by_category[category].append(speedup)
        rows.append(
            {
                "row": row,
                "category": category,
                "ours": ours,
                "ours_us": ours_us,
                "reference": reference,
                "reference_us": reference_us,
                "speedup": speedup,
                "verdict": verdict,
                "ours_cv": _cv(row, ours),
                "reference_cv": _cv(row, reference),
            }
        )

    speedups = [item["speedup"] for item in rows]
    overall = _geomean(speedups)
    attention_speedup = by_category["Attention"][0]
    measured_date = (results[0].get("started_at") or "")[:10] or "unknown date"
    pipeline = run.get("pipeline") or {}
    protocol = results[0].get("benchmark_protocol") or {}
    protocol_fields = ("warmup", "repeat", "rounds", "round_aggregate")
    for row in results[1:]:
        row_protocol = row.get("benchmark_protocol") or {}
        if any(row_protocol.get(field) != protocol.get(field) for field in protocol_fields):
            raise ValueError("representative rows do not share one timing protocol")
    selection = run.get("selection") or {}
    git = run.get("git") or {}
    baselines = run.get("baselines") or {}
    flashinfer = baselines.get("flashinfer") or {}
    flashattn = baselines.get("flash_attn") or {}
    noisy = sum(max(item["ours_cv"], item["reference_cv"]) > 10.0 for item in rows)

    lines = [
        "# NVIDIA Thor native-baseline performance",
        "",
        f"Measured on {measured_date} on one NVIDIA Jetson AGX Thor Developer Kit. "
        "Every row compares TIRx with a contract-matched implementation on the same GPU and exact "
        "input shape. FA4 uses upstream FlashAttention-4 CuTeDSL as its primary baseline.",
        "",
        "## At a glance",
        "",
        "| Question | Result |",
        "|---|---|",
        f"| Correctness and execution | **{len(rows)}/{len(rows)} passed**; "
        f"{pipeline.get('failure_count', 0)} failures and "
        f"{pipeline.get('interference_retry_count', 0)} interference retries |",
        f"| TIRx faster by more than 5% | **{verdicts['TIRx faster']}/{len(rows)}** |",
        f"| Within 5% | **{verdicts['within 5%']}/{len(rows)}** |",
        f"| Reference faster by more than 5% | **{verdicts['Reference faster']}/{len(rows)}** |",
        f"| Geometric-mean TIRx speedup | **{overall:.3f}x** |",
        "",
        "The mixed aggregate hides a wide spread: at the selected GQA-prefill shape, the "
        f"upstream FA4 CuTeDSL latency divided by TIRx latency is {attention_speedup:.3f}x. "
        "The plain RMSNorm and GELU activation paths favor FlashInfer, and most other rows are "
        "close. Use the per-family and "
        "per-kernel rows for tuning decisions rather than the single mixed-workload mean.",
        "",
        "## Results by family",
        "",
        "Speedup is `primary-reference latency / TIRx latency`; values above 1.0 favor TIRx.",
        "",
        "| Family | Rows | Geomean TIRx speedup |",
        "|---|---:|---:|",
    ]
    for category in _CATEGORY_ORDER:
        values = by_category.get(category)
        if values:
            lines.append(f"| {category} | {len(values)} | {_geomean(values):.3f}x |")

    lines.extend(
        [
            "",
            "## Complete representative table",
            "",
            "The 5% band is descriptive, not a statistical significance test.",
            "",
            "| Family | Kernel / config | TIRx µs | TIRx CV | Primary reference | "
            "Reference µs | Ref CV | TIRx speedup | Result |",
            "|---|---|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    category_index = {category: index for index, category in enumerate(_CATEGORY_ORDER)}
    for item in sorted(
        rows, key=lambda value: (category_index[value["category"]], value["row"]["kernel"])
    ):
        row = item["row"]
        lines.append(
            f"| {item['category']} | `{row['kernel']}/{_config(row)}` | "
            f"{item['ours_us']:.3f} | {item['ours_cv']:.1f}% | "
            f"`{item['reference']}` | {item['reference_us']:.3f} | "
            f"{item['reference_cv']:.1f}% | **{item['speedup']:.3f}x** | "
            f"{item['verdict']} |"
        )

    attention = next(item for item in rows if item["row"]["kernel"] == "flash_attention4")
    attention_impls = attention["row"]["impls"]
    secondary_attention = [
        (name, float(attention_impls[name]), _cv(attention["row"], name))
        for name in ("flashinfer_cutedsl", "flashinfer_fa2")
        if name in attention_impls
    ]
    if secondary_attention:
        lines.extend(
            [
                "",
                "Attention secondary controls (same tensors and shape; not used in the aggregate):",
                "",
                "| Implementation | Latency µs | CV | latency / TIRx |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, latency, cv in secondary_attention:
            lines.append(
                f"| `{name}` | {latency:.3f} | {cv:.1f}% | {latency / attention['ours_us']:.3f}x |"
            )

    lines.extend(
        [
            "",
            "## Selection and interpretation",
            "",
            "The roster deliberately samples serving-relevant operator families rather than every "
            "shape: attention, RMSNorm, fused activation, FP4 quantization, four TopK variants, "
            "and three recurrent/SSM paths. Each reference adapter lives beside its kernel and "
            "executes "
            "the same fused operation on the same generated inputs. The workload roster is "
            "[`scripts/thor_flashinfer_representative.yaml`](scripts/thor_flashinfer_representative.yaml).",
            "The per-kernel source and upstream-benchmark decisions are documented in "
            "[THOR_SOURCE_BENCHMARK_AUDIT.md](THOR_SOURCE_BENCHMARK_AUDIT.md).",
            "",
            "Because attention is the most prominent result, a separate complete 32-config "
            "sequence-length, GQA-ratio, and causal-mask sweep is reported in "
            "[THOR_FLASHINFER_ATTENTION.md](THOR_FLASHINFER_ATTENTION.md). That older sweep is "
            "explicitly a secondary FA2 generation comparison, not the FA4 headline baseline.",
            "",
            "The GDN and grouped-KDA choices follow production dispatch shapes present in SGLang's "
            "kernel configuration manifests. FlashInfer remains the timed implementation baseline "
            "for those ported kernels; SGLang supplies shape provenance rather than a second "
            "timing column. Attention is different: upstream FA4 CuTeDSL is primary, "
            "FlashInfer's own "
            "CuTeDSL path is a serving-library peer, and FA2 is retained only as a legacy control.",
            "",
            "This is a kernel-launch microbenchmark, not end-to-end request throughput. It does "
            "not measure scheduler, KV-cache management, batching policy, CPU work, or network "
            "overhead. "
            "The 13 rows demonstrate representative performance, while the exhaustive numerical "
            "coverage remains documented in [THOR_VALIDATION.md](THOR_VALIDATION.md).",
            "",
            "## Measurement provenance",
            "",
            "| Field | Value |",
            "|---|---|",
            "| GPU | NVIDIA Jetson AGX Thor Developer Kit, 20 SMs |",
            f"| CUDA architecture | `{selection.get('cuda_arch', 'sm_110a')}` |",
            "| Power/clock state | `MAXN`; dynamic clocks (`jetson_clocks` was not locked) |",
            f"| Timer | `{selection.get('timer_override', 'proton')}` |",
            f"| Rounds / aggregation | {protocol.get('rounds', 5)} / arithmetic mean |",
            f"| Warmup / repeat budget | {protocol.get('warmup', 25)} ms / "
            f"{protocol.get('repeat', 100)} ms per implementation per round |",
            f"| TVM/TIR revision | `{git.get('tir', '-')}` |",
            f"| TIRx-kernels revision | `{git.get('tirx-kernels', '-')}` |",
            f"| FlashInfer version / revision | `{flashinfer.get('version', '-')}` / "
            f"`{flashinfer.get('git_sha', '-')}` |",
            f"| FlashAttention-4 revision | `{flashattn.get('git_sha', '-')}` |",
            "| CUDA / PyTorch | CUDA 13.1 / PyTorch 2.9.1+cu130 |",
            "",
            f"The population coefficient of variation exceeded 10% for either implementation in "
            f"**{noisy}/{len(rows)}** rows. Dynamic clocks can move absolute latency between "
            "rounds; "
            "lock the production power mode and clocks before treating small differences as tuning "
            "wins. Both sides in every row nevertheless share the same process, GPU, timer, and "
            "five-round protocol.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python -m tirx_kernels.bench_suite \\",
            "  --workloads scripts/thor_flashinfer_representative.yaml \\",
            f"  --out-dir {run_path.parent.parent} \\",
            "  --with-references --timer proton --rounds 5 --cooldown 0 \\",
            "  --max-prepare-processes 1 --ready-backlog 1 --no-probe --no-report",
            "python scripts/report_thor_native.py \\",
            f"  --run {run_path} \\",
            "  --output THOR_NATIVE_BASELINE.md",
            "```",
            "",
            "The Thor CUDA, TVM, CUPTI, and `sm_110a` environment variables described in "
            "[THOR_VALIDATION.md](THOR_VALIDATION.md) must be set first.",
            "",
            "## Raw evidence",
            "",
            f"- Run JSON: `{run_path}`",
            f"- Run status: {len(rows)} `ok`; every reported implementation has five round samples",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("THOR_NATIVE_BASELINE.md"))
    args = parser.parse_args()

    run = json.loads(args.run.read_text())
    args.output.write_text(_markdown(run, run_path=args.run.resolve()))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
