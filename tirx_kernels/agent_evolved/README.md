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

# Agent-evolved kernels

This directory contains curated kernels selected from measured agent-evolution
runs. The table below summarizes their registered benchmarks; multi-shape
sweeps may use one geometric-mean row. A pull request that adds or changes an
agent-evolved kernel must update its rows from a same-run candidate/reference
measurement.

## Measured speedups

`Speedup` is `reference GPU time / TIRx GPU time`. Values greater than one mean
TIRx is faster. The KDA rows below use CUDA-event timings around the launched
GPU work; they exclude compilation and host-side setup, so they are GPU
kernel-time comparisons, not end-to-end operator latency comparisons. The
upstream TIRx kernel and this PR used the same inputs, with five warmups and 30
timed iterations per configuration. The reported values are the median timings
from the run against upstream `main` at `156a5c0` and PR commit `5fb7b75`.

| Kernel | Config | GPU | Timer | TIRx (us) | Reference | Reference (us) | Speedup | Evidence |
|---|---|---|---|---:|---|---:|---:|---|
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h96_fixed` | B200 | CUDA events | 329.400 | Upstream TIRx | 394.400 | 1.197x | upstream `156a5c0` vs PR `5fb7b75` |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h96_mixed` | B200 | CUDA events | 256.000 | Upstream TIRx | 328.400 | 1.283x | upstream `156a5c0` vs PR `5fb7b75` |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h96_uniform` | B200 | CUDA events | 285.900 | Upstream TIRx | 317.500 | 1.111x | upstream `156a5c0` vs PR `5fb7b75` |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h64_fixed` | B200 | CUDA events | 327.700 | Upstream TIRx | 390.500 | 1.192x | upstream `156a5c0` vs PR `5fb7b75` |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h64_mixed` | B200 | CUDA events | 175.700 | Upstream TIRx | 232.800 | 1.325x | upstream `156a5c0` vs PR `5fb7b75` |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h64_uniform` | B200 | CUDA events | 200.000 | Upstream TIRx | 216.400 | 1.082x | upstream `156a5c0` vs PR `5fb7b75` |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `all-six-geomean` | B200 | CUDA events | 255.413 | Upstream TIRx | 305.251 | 1.195x | geometric mean of the six rows above |
| [`agent_evolved_moe_fp8_blockscale_dsv3`](moe_fp8_blockscale_dsv3.py) | `t14107` | B200 | CUPTI | 717.491 | FlashInfer TRT-LLM FP8 MoE | 2239.532 | 3.121x | `moe-20260908-004417` default-max row in full-sweep promotion run |

## Updating the table

Run all registered agent-evolved workloads and their references through the
bench suite:

```bash
python -m tirx_kernels.bench_suite \
  --filter agent_evolved \
  --with-references \
  --rounds 5 \
  --cooldown 1
```

For each accepted result, require the workload's declared timer, an empty
`errors` mapping, and a successful interference check. Copy the arithmetic
means from `impls`, compute the ratio from those unrounded values, and round
displayed times and speedups to three decimals. Never combine candidate and
reference times from different runs, GPUs, configs, timers, or dependency
revisions.
