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
runs. The table below is the performance summary for their registered benchmark
configs. A pull request that adds or changes an agent-evolved kernel must update
its rows from a same-run candidate/reference measurement.

## Measured speedups

`Speedup` is `reference GPU time / TIRx GPU time`. Values greater than one mean
TIRx is faster. For Proton rows, the timer reports the sum of GPU leaf-kernel
durations in each scope; it excludes CPU launch time and gaps between kernels,
so those rows are GPU kernel-time comparisons, not end-to-end operator latency
comparisons.

| Kernel | Config | GPU | Timer | TIRx (us) | Reference | Reference (us) | Speedup | Evidence |
|---|---|---|---|---:|---|---:|---:|---|
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h96_fixed` | B200 | Proton | 367.937 | FlashKDA | 1048.609 | 2.850x | `kda_forward-20260901-235533` v68 promotion rerun |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h96_mixed` | B200 | Proton | 315.741 | FlashKDA | 889.884 | 2.818x | `kda_forward-20260901-235533` v68 promotion rerun |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h96_uniform` | B200 | Proton | 313.221 | FlashKDA | 714.718 | 2.282x | `kda_forward-20260901-235533` v68 promotion rerun |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h64_fixed` | B200 | Proton | 372.790 | FlashKDA | 956.330 | 2.565x | `kda_forward-20260901-235533` v68 promotion rerun |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h64_mixed` | B200 | Proton | 225.736 | FlashKDA | 670.399 | 2.970x | `kda_forward-20260901-235533` v68 promotion rerun |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h64_uniform` | B200 | Proton | 212.998 | FlashKDA | 484.290 | 2.274x | `kda_forward-20260901-235533` v68 promotion rerun |

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
