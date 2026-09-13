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
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h96_fixed` | GB200 | Proton | 293.067 | FlashKDA | 992.488 | 3.387x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h96_mixed` | GB200 | Proton | 241.896 | FlashKDA | 829.136 | 3.428x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h96_uniform` | GB200 | Proton | 243.664 | FlashKDA | 671.549 | 2.756x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h64_fixed` | GB200 | Proton | 238.815 | FlashKDA | 905.974 | 3.794x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h64_mixed` | GB200 | Proton | 165.080 | FlashKDA | 631.178 | 3.823x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `h64_uniform` | GB200 | Proton | 167.364 | FlashKDA | 455.826 | 2.724x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_kda_forward_b1_t8192`](kda_forward_b1_t8192.py) | `all-six-geomean` | GB200 | Proton | 220.192 | FlashKDA | 724.018 | 3.288x | geometric mean of the six rows above |
| [`agent_evolved_moe_fp8_blockscale_dsv3`](moe_fp8_blockscale_dsv3.py) | `t14107` | B200 | CUPTI | 717.491 | FlashInfer TRT-LLM FP8 MoE | 2239.532 | 3.121x | `moe-20260908-004417` default-max row in full-sweep promotion run |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `packed_1024x8_h96` | B200 | kcoral | 829.482 | FLA `chunk_kda_bwd` | 7695.851 | 9.278x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `all-seventeen-geomean` | B200 | kcoral | 249.000 | FLA `chunk_kda_bwd` | 2132.204 | 8.563x | geometric mean of the seventeen official workloads in that sweep |
| [`agent_evolved_msa_prefill_b1_q4096`](msa_prefill_b1_q4096.py) | `b1_q4096_kv4096_h64` | GB200 | Proton | 181.921 | MiniMax MSA | 571.860 | 3.143x | `msa_prefill_b1_q4096_kv4096_hq64_hkv4_d128_topk16_bf16_flat-20260910-221452` `frontier/qmajor-persistent` port rerun |
| [`agent_evolved_msa_prefill_multishape`](msa_prefill_multishape.py) | `flat_bf16_b1_q4096_kv4096_h64` | GB200 | Proton | 202.934 | MiniMax MSA | 568.516 | 2.802x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_msa_prefill_multishape`](msa_prefill_multishape.py) | `paged_bf16_b3_q4096_kv8192_h8` | GB200 | Proton | 64.195 | FlashInfer trtllm-gen | 429.737 | 6.694x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_msa_prefill_multishape`](msa_prefill_multishape.py) | `both-rows-geomean` | GB200 | Proton | 114.137 | MiniMax MSA / trtllm-gen | 494.280 | 4.331x | geometric mean of the two registered rows in that run |
| [`agent_evolved_vsa_s80000_h8_topk156`](vsa_s80000_h8_topk156.py) | `s80000_h8_blk128_topk156` | GB200 | Proton | 4681.178 | FlashInfer `bsa_attn_fwd` | 7010.635 | 1.498x | `vsa_s80000_h8_d128_blk128_topk156_bf16-20260910-221104` `frontier/group16-cluster-multicast` port rerun |

For `kcoral` rows the timer is the evolution harness's benchmark server, which
reports GPU-only latency for candidate and reference in one run. Geomean rows
divide the two geometric means; they are not means of per-workload ratios.

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
