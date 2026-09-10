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
kernel-time comparisons, not end-to-end operator latency comparisons. The PR
kernel and FlashKDA used the same inputs, with five warmups and 30 timed
iterations per configuration. The reported values are the median timings from
the B200 run using FlashKDA commit `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`.

| Kernel | Config | GPU | Timer | TIRx (us) | Reference | Reference (us) | Speedup | Evidence |
|---|---|---|---|---:|---|---:|---:|---|
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h96_fixed` | B200 | CUDA events | 354.700 | FlashKDA | 1081.000 | 3.048x | B200 run; 5 warmups, 30-event median |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h96_mixed` | B200 | CUDA events | 296.500 | FlashKDA | 921.800 | 3.109x | B200 run; 5 warmups, 30-event median |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h96_uniform` | B200 | CUDA events | 329.300 | FlashKDA | 747.100 | 2.268x | B200 run; 5 warmups, 30-event median |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h64_fixed` | B200 | CUDA events | 353.500 | FlashKDA | 986.000 | 2.790x | B200 run; 5 warmups, 30-event median |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h64_mixed` | B200 | CUDA events | 215.400 | FlashKDA | 700.300 | 3.251x | B200 run; 5 warmups, 30-event median |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h64_uniform` | B200 | CUDA events | 240.500 | FlashKDA | 514.800 | 2.141x | B200 run; 5 warmups, 30-event median |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `all-six-geomean` | B200 | CUDA events | 293.115 | FlashKDA | 801.260 | 2.734x | geometric mean of the six rows above |
| [`agent_evolved_moe_fp8_blockscale_dsv3`](moe_fp8_blockscale_dsv3.py) | `t14107` | B200 | CUPTI | 717.491 | FlashInfer TRT-LLM FP8 MoE | 2239.532 | 3.121x | `moe-20260908-004417` default-max row in full-sweep promotion run |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `packed_1024x8_h96` | B200 | kcoral | 829.482 | FLA `chunk_kda_bwd` | 7695.851 | 9.278x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `p01_hq4_hv8_t32768` | B200 | kcoral | 383.336 | FLA `chunk_kda_bwd` | 3147.457 | 8.211x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `p02_hq2_hv8_t18432` | B200 | kcoral | 231.976 | FLA `chunk_kda_bwd` | 2047.791 | 8.828x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `p03_hq4_hv4_t32768` | B200 | kcoral | 221.096 | FLA `chunk_kda_bwd` | 1751.478 | 7.922x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `p04_hq2_hv4_t18432` | B200 | kcoral | 138.889 | FLA `chunk_kda_bwd` | 1564.793 | 11.267x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `p05_hq4_hv8_t18432` | B200 | kcoral | 234.697 | FLA `chunk_kda_bwd` | 2057.659 | 8.767x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `p06_hq2_hv8_t32768` | B200 | kcoral | 376.115 | FLA `chunk_kda_bwd` | 2863.702 | 7.614x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `p07_hq2_hv4_t18432` | B200 | kcoral | 145.529 | FLA `chunk_kda_bwd` | 1458.870 | 10.025x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `p08_hq4_hv4_t18432` | B200 | kcoral | 139.745 | FLA `chunk_kda_bwd` | 1326.935 | 9.495x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `p09_hq2_hv8_t32768` | B200 | kcoral | 380.841 | FLA `chunk_kda_bwd` | 2851.699 | 7.488x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `p10_hq4_hv8_t18432` | B200 | kcoral | 236.081 | FLA `chunk_kda_bwd` | 1926.610 | 8.161x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `p11_hq4_hv4_t32768` | B200 | kcoral | 226.874 | FLA `chunk_kda_bwd` | 1719.478 | 7.579x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `p12_hq2_hv4_t32768` | B200 | kcoral | 216.529 | FLA `chunk_kda_bwd` | 1877.112 | 8.669x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `p13_hq4_hv4_t18432` | B200 | kcoral | 146.729 | FLA `chunk_kda_bwd` | 1407.337 | 9.591x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `p14_hq2_hv4_t32768` | B200 | kcoral | 222.105 | FLA `chunk_kda_bwd` | 1811.933 | 8.158x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `p15_hq4_hv8_t32768` | B200 | kcoral | 383.417 | FLA `chunk_kda_bwd` | 2915.541 | 7.604x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `p16_hq2_hv8_t18432` | B200 | kcoral | 242.296 | FLA `chunk_kda_bwd` | 1903.332 | 7.855x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |

The KDA-backward rows are not yet bench-suite Proton measurements. Their
candidate and reference times come from one official all-shape scoring run of
the evolution harness on the kcoral benchmark server, which reports GPU-only
latency and measured both implementations in that same run. Replace them with
`Proton` rows from the command below the next time a B200 is available.

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
