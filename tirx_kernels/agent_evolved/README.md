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
| [`agent_evolved_alphamoe_fp8_blockscale_qwen3next`](alphamoe_fp8_blockscale_qwen3next.py) | `m1_official` | GB200 | CUDA event | 10.511 | FlashInfer `trtllm_fp8_block_scale_routed_moe` | 37.323 | 3.5507x | `run_bench(timer="event")` same-run pair; the locked evaluation harness (CUPTI, cold L2) scores 6.232 us vs 22.016 us (3.5326x) on this row. |
| [`agent_evolved_alphamoe_fp8_blockscale_qwen3next`](alphamoe_fp8_blockscale_qwen3next.py) | `m8_official` | GB200 | CUDA event | 20.526 | FlashInfer `trtllm_fp8_block_scale_routed_moe` | 50.255 | 2.4483x | `run_bench(timer="event")` same-run pair; the locked evaluation harness (CUPTI, cold L2) scores 16.088 us vs 44.552 us (2.7693x) on this row. |
| [`agent_evolved_alphamoe_fp8_blockscale_qwen3next`](alphamoe_fp8_blockscale_qwen3next.py) | `m16_official` | GB200 | CUDA event | 28.063 | FlashInfer `trtllm_fp8_block_scale_routed_moe` | 62.531 | 2.2282x | `run_bench(timer="event")` same-run pair; the locked evaluation harness (CUPTI, cold L2) scores 23.488 us vs 54.96 us (2.3399x) on this row. |
| [`agent_evolved_alphamoe_fp8_blockscale_qwen3next`](alphamoe_fp8_blockscale_qwen3next.py) | `m32_official` | GB200 | CUDA event | 40.355 | FlashInfer `trtllm_fp8_block_scale_routed_moe` | 83.535 | 2.0700x | `run_bench(timer="event")` same-run pair; the locked evaluation harness (CUPTI, cold L2) scores 35.576 us vs 74.36 us (2.0902x) on this row. |
| [`agent_evolved_alphamoe_fp8_blockscale_qwen3next`](alphamoe_fp8_blockscale_qwen3next.py) | `m64_official` | GB200 | CUDA event | 54.135 | FlashInfer `trtllm_fp8_block_scale_routed_moe` | 102.287 | 1.8895x | `run_bench(timer="event")` same-run pair; the locked evaluation harness (CUPTI, cold L2) scores 49.665 us vs 93.416 us (1.8809x) on this row. |
| [`agent_evolved_alphamoe_fp8_blockscale_qwen3next`](alphamoe_fp8_blockscale_qwen3next.py) | `m128_official` | GB200 | CUDA event | 65.895 | FlashInfer `trtllm_fp8_block_scale_routed_moe` | 119.799 | 1.8180x | `run_bench(timer="event")` same-run pair; the locked evaluation harness (CUPTI, cold L2) scores 60.96 us vs 112.256 us (1.8415x) on this row. |
| [`agent_evolved_alphamoe_fp8_blockscale_qwen3next`](alphamoe_fp8_blockscale_qwen3next.py) | `all-six-geomean` | GB200 | CUDA event | 30.907 | FlashInfer `trtllm_fp8_block_scale_routed_moe` | 70.237 | 2.2725x | geometric mean of the six official rows M in {1,8,16,32,64,128}, K=2048, I=128, E=512, top-10. All six pass `run_test` (element-wise bound `max(0.1+0.1|ref|, 2*abs_sum*2**-7)` against an independent oracle); repeated launches are not bitwise identical because the down projection accumulates with `red.global.add.bf16x2`, which the bound's accumulation-order term covers. The baseline is CUDA-graph captured exactly as the evaluation harness captures it, and the default Proton timer is deliberately not used because its per-kernel instrumentation overstates this many-kernel baseline. The locked harness scores a 2.3437x geomean. **This kernel supersedes the previous M=128-only entry and regresses M=128 from 59.984 us to 60.960 us (~1.6%) under the harness, accepted in exchange for covering all six rows and 3.53x at M=1.** Evolution run `alphamoe-20260916-193749`, member `cluster-split`; Synccheck and Racecheck clean. |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `packed_1024x8_h96` | B200 | kcoral | 829.482 | FLA `chunk_kda_bwd` | 7695.851 | 9.278x | `kda-bwd-portfolio-carry-3` `fused-zfold` all-shape sweep |
| [`agent_evolved_kda_backward_packed`](kda_backward_packed.py) | `all-seventeen-geomean` | B200 | kcoral | 249.000 | FLA `chunk_kda_bwd` | 2132.204 | 8.563x | geometric mean of the seventeen official workloads in that sweep |
| [`agent_evolved_msa_prefill_multishape`](msa_prefill_multishape.py) | `flat_bf16_b1_q4096_kv4096_h64` | GB200 | Proton | 177.421 | MiniMax MSA | 566.256 | 3.192x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_msa_prefill_multishape`](msa_prefill_multishape.py) | `flat_fp8_b3_q1024_kv8192_h32` | GB200 | Proton | 120.207 | MiniMax MSA | 174.347 | 1.450x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_msa_prefill_multishape`](msa_prefill_multishape.py) | `paged_bf16_b3_q4096_kv8192_h8` | GB200 | Proton | 82.292 | FlashInfer trtllm-gen | 418.327 | 5.084x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_msa_prefill_multishape`](msa_prefill_multishape.py) | `all-three-geomean` | GB200 | Proton | 120.623 | MiniMax MSA / trtllm-gen | 345.659 | 2.866x | geometric mean of the three official rows in that run |
| [`agent_evolved_msa_decode_multishape`](msa_decode_multishape.py) | `mtp_bf16_b128_q16_kv4096_h64` | GB200 | Proton | 206.988 | MiniMax MSA | 1207.386 | 5.833x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_msa_decode_multishape`](msa_decode_multishape.py) | `decode_fp8_b128_q1_kv4096_h64` | GB200 | Proton | 49.824 | MiniMax MSA | 510.976 | 10.256x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_msa_decode_multishape`](msa_decode_multishape.py) | `decode_fp16_b128_q1_kv4096_h64` | GB200 | Proton | 91.127 | FlashInfer trtllm-gen block-sparse | 96.427 | 1.058x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_msa_decode_multishape`](msa_decode_multishape.py) | `all-ten-geomean` | GB200 | Proton | 72.634 | MiniMax MSA | 289.600 | 3.987x | geometric mean of the ten official rows in that run |
| [`agent_evolved_kda_decode_multishape`](kda_decode_multishape.py) | `t1_b128_hv32_standard` | GB200 | Proton | 41.424 | FlashInfer `recurrent_kda` | 54.189 | 1.308x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_kda_decode_multishape`](kda_decode_multishape.py) | `t6_b128_hv32_spec` | GB200 | Proton | 160.250 | FlashInfer `recurrent_kda` | 222.751 | 1.390x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_kda_decode_multishape`](kda_decode_multishape.py) | `t3_b16_hv16_lower_bound` | GB200 | Proton | 10.075 | FlashInfer `recurrent_kda` | 15.976 | 1.586x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_kda_decode_multishape`](kda_decode_multishape.py) | `all-thirty-geomean` | GB200 | Proton | 22.153 | FlashInfer `recurrent_kda` | 29.263 | 1.321x | geometric mean of the thirty official rows in that run |
| [`agent_evolved_mla_dsv4_multishape`](mla_dsv4_multishape.py) | `prefill_h128_swa16384_topk4x_c16384_k1024_bf16_hnd` | GB200 | Proton | 79.082 | FlashInfer trtllm-gen DSv4 | 117.353 | 1.484x | PR #207 numerical-fix sweep; same-run pair, rounds=5, cooldown=1s |
| [`agent_evolved_mla_dsv4_multishape`](mla_dsv4_multishape.py) | `decode_h16_swa512_topk128x_c128_k132_fp8_hnd` | GB200 | Proton | 7.000 | FlashInfer trtllm-gen DSv4 | 15.388 | 2.198x | PR #207 numerical-fix sweep; same-run pair, rounds=5, cooldown=1s |
| [`agent_evolved_mla_dsv4_multishape`](mla_dsv4_multishape.py) | `prefill_h128_swa16384_topk4x_c16384_k1024_fp8_hnd` | GB200 | Proton | 88.025 | FlashInfer trtllm-gen DSv4 | 78.358 | 0.890x | PR #207 numerical-fix sweep; same-run pair, rounds=5, cooldown=1s |
| [`agent_evolved_mla_dsv4_multishape`](mla_dsv4_multishape.py) | `all-ninety-four-geomean` | GB200 | Proton | 9.621 | FlashInfer trtllm-gen DSv4 | 16.503 | 1.715x | geometric mean of all 94 same-run pairs in the PR #207 numerical-fix sweep |
| [`agent_evolved_vsa_multishape`](vsa_multishape.py) | `pooled_blk128_s80000_topk156` | GB200 | Proton | 4523.323 | FlashInfer `bsa_attn_fwd` | 7045.071 | 1.558x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_vsa_multishape`](vsa_multishape.py) | `bsr_blk64_s4096_topk32` | GB200 | Proton | 34.869 | FlashInfer `bsa_attn_blk64_fwd` | 58.166 | 1.668x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_vsa_multishape`](vsa_multishape.py) | `fastwan_blk64_s26624_h12_topk84_partial` | GB200 | Proton | 764.725 | FlashInfer `bsa_attn_blk64_fwd` | 938.857 | 1.228x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`agent_evolved_vsa_multishape`](vsa_multishape.py) | `all-eighteen-geomean` | GB200 | Proton | 92.210 | FlashInfer BSA | 154.753 | 1.678x | geometric mean of the eighteen official rows in that run |

The MLA rows measure the kernel after the FP8 numerical repairs (BF16 split
partials and P scaled by 448), against FlashInfer 0.6.18.post1 on GB200 with Proton. Each of
the 94 configurations uses five same-run candidate/reference rounds and
arithmetic-mean times; the aggregate row divides their geometric means.
GPU execution was serialized by the benchmark server's per-GPU lease.
Source and measurement details are recorded in [PR #207](https://github.com/mlc-ai/TIRx-kernels/pull/207).

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
