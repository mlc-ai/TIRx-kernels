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
| [`agent_evolved_kda_decode_b128_t1`](kda_decode_b128_t1.py) | `b128_t1_h16_hv32_d128` | GB200 | Proton | 41.445 | FlashInfer `recurrent_kda` | 54.178 | 1.307x | `kda-decode-b128-t1` `frontier/clc-steal` port rerun |
| [`agent_evolved_msa_prefill_b1_q4096`](msa_prefill_b1_q4096.py) | `b1_q4096_kv4096_h64` | GB200 | Proton | 181.921 | MiniMax MSA | 571.860 | 3.143x | `msa_prefill_b1_q4096_kv4096_hq64_hkv4_d128_topk16_bf16_flat-20260910-221452` `frontier/qmajor-persistent` port rerun |
| [`agent_evolved_msa_decode_b128_q16`](msa_decode_b128_q16.py) | `b128_q16_kv4096_h64` | GB200 | Proton | 206.197 | MiniMax MSA | 1206.456 | 5.851x | `msa-decode-b128` `frontier/qmajor-union` port rerun |
| [`agent_evolved_mla_dsv4_prefill_b2`](mla_dsv4_prefill_b2.py) | `b2_qsum386_h128_swa16384_c16384_k1152` | GB200 | Proton | 77.402 | FlashInfer trtllm-gen DSv4 sparse MLA | 117.251 | 1.515x | `mla-dsv4-prefill-b2` `frontier/dual-issuer` port rerun |
| [`agent_evolved_vsa_multishape`](vsa_multishape.py) | `pooled_blk128_s80000_topk156` | GB200 | Proton | 4523.323 | FlashInfer `bsa_attn_fwd` | 7045.071 | 1.558x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_vsa_multishape`](vsa_multishape.py) | `bsr_blk64_s4096_topk32` | GB200 | Proton | 34.869 | FlashInfer `bsa_attn_blk64_fwd` | 58.166 | 1.668x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_vsa_multishape`](vsa_multishape.py) | `fastwan_blk64_s26624_h12_topk84_partial` | GB200 | Proton | 764.725 | FlashInfer `bsa_attn_blk64_fwd` | 938.857 | 1.228x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_vsa_multishape`](vsa_multishape.py) | `all-eighteen-geomean` | GB200 | Proton | 92.210 | FlashInfer BSA | 154.753 | 1.678x | geometric mean of the eighteen official rows in that run |

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
