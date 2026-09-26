<!--
Copyright (c) 2026 The TIRx Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Curated native TIRx kernels

This directory contains kernels implemented directly in TIRx and admitted
through portfolio-wide correctness and performance validation, rather than
ported from an external kernel library. The table below has one aggregate row
per kernel and covers every registered benchmark configuration. A pull request
that adds or changes a curated kernel must update its row from complete,
same-run candidate/reference measurements.

## Measured speedups

`Speedup` is `reference GPU time / TIRx GPU time`; values greater than one mean
TIRx is faster. Geomean is the unweighted geometric mean of the per-config
speedups. Min and max are the extrema across those same per-config ratios. All
ratios are calculated from unrounded arithmetic-mean times before display
rounding.

These results cover the 184 configurations that remain after the
DeepSeek-V3 FP8 MoE kernel was removed. They come from the 2026-09-25–26 full
curated sweep on NVIDIA GB200 (`sm_100a`). Each candidate/reference pair used
five independent timer rounds with a one-second cooldown before every
implementation in every round. The 149 FlashInfer-backed pairs used FlashInfer
`0.6.18.dev20260909` at commit
`13099a68b2f663cea0400a2c161d5264010dfc9c` and NVIDIA CUTLASS DSL `4.7.0`.
The remaining 35 pairs used their declared FLA, FlashKDA, or MiniMax MSA
reference. These are GPU-kernel-time comparisons, not end-to-end operator
latencies.

| Kernel | Configs | GPU | Timer | Reference | Geomean | Min | Max |
|---|---:|---|---|---|---:|---:|---:|
| [`curated_alphamoe_fp8_blockscale_qwen3next`](alphamoe_fp8_blockscale_qwen3next.py) | 6 | GB200 | CUDA event | FlashInfer `trtllm_fp8_block_scale_routed_moe` | 2.1881x | 1.7156x | 3.5527x |
| [`curated_kda_backward_packed`](kda_backward_packed.py) | 17 | GB200 | Proton | FLA `chunk_kda_bwd` | 6.8392x | 5.2177x | 8.8062x |
| [`curated_kda_decode_multishape`](kda_decode_multishape.py) | 30 | GB200 | Proton | FlashInfer `recurrent_kda` | 1.3261x | 1.0862x | 1.5894x |
| [`curated_kda_forward_portfolio_multishape`](kda_forward_portfolio_multishape.py) | 6 | GB200 | CUDA event | FlashKDA | 2.7873x | 2.2443x | 3.4193x |
| [`curated_mla_dsv4_multishape`](mla_dsv4_multishape.py) | 94 | GB200 | Proton | FlashInfer TRT-LLM DSv4 | 1.7124x | 0.8659x | 2.3419x |
| [`curated_msa_decode_multishape`](msa_decode_multishape.py) | 10 | GB200 | Proton | MiniMax MSA | 3.9878x | 1.0616x | 10.1356x |
| [`curated_msa_prefill_multishape`](msa_prefill_multishape.py) | 3 | GB200 | Proton | MiniMax MSA / FlashInfer TRT-LLM bridge | 2.5853x | 1.1983x | 4.4804x |
| [`curated_vsa_multishape`](vsa_multishape.py) | 18 | GB200 | Proton | FlashInfer BSA | 1.6824x | 1.1827x | 2.4299x |

## Updating the table

Run every benchmark config, including entries marked `default: false`, through
the bench suite with external references enabled. A plain `--filter curated`
run is insufficient because it keeps only the pinned `default: true` subset;
provide an explicit all-config workload file instead:

```bash
python -m tirx_kernels.bench_suite \
  --workloads .bench-suite/curated-all-configs.yaml \
  --with-references \
  --rounds 5 \
  --cooldown 1
```

Require every workload to use its declared timer, report an empty `errors`
mapping, and contain five samples for both TIRx and its reference. Compute each
config's ratio from the unrounded arithmetic means, then report the geometric
mean, minimum, and maximum of those ratios. Keep one aggregate row per kernel;
do not add individual-config measurements to this README.
