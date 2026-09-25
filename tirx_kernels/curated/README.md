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
ported from an external kernel library. The table below summarizes their
registered benchmarks; multi-shape sweeps may use one geometric-mean row. A
pull request that adds or changes a curated kernel must update its rows from a
same-run candidate/reference measurement.

## Measured speedups

`Speedup` is `reference GPU time / TIRx GPU time`. Values greater than one mean
TIRx is faster. The KDA forward rows below use CUDA-event timings around the
launched GPU work; they exclude compilation and host-side setup, so they are GPU
kernel-time comparisons, not end-to-end operator latency comparisons. The PR
kernel and FlashKDA used the same inputs, with five warmups and 30 timed
iterations per configuration. The reported values are the median timings from
the B200 run using FlashKDA commit `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`.

| Kernel | Config | GPU | Timer | TIRx (us) | Reference | Reference (us) | Speedup | Evidence |
|---|---|---|---|---:|---|---:|---:|---|
| [`curated_kda_forward_portfolio_multishape`](kda_forward_portfolio_multishape.py) | `h96_fixed` | GB200 | CUDA events | 327.711 | FlashKDA | 974.155 | 2.973x | `run_bench`; event timer because the fixed route launches two concurrent kernels |
| [`curated_kda_forward_portfolio_multishape`](kda_forward_portfolio_multishape.py) | `h96_mixed` | GB200 | CUDA events | 293.909 | FlashKDA | 842.822 | 2.868x | `run_bench`; event timer because the fixed route launches two concurrent kernels |
| [`curated_kda_forward_portfolio_multishape`](kda_forward_portfolio_multishape.py) | `h96_uniform` | GB200 | CUDA events | 308.971 | FlashKDA | 692.548 | 2.241x | `run_bench`; event timer because the fixed route launches two concurrent kernels |
| [`curated_kda_forward_portfolio_multishape`](kda_forward_portfolio_multishape.py) | `h64_fixed` | GB200 | CUDA events | 253.103 | FlashKDA | 885.340 | 3.498x | `run_bench`; event timer because the fixed route launches two concurrent kernels |
| [`curated_kda_forward_portfolio_multishape`](kda_forward_portfolio_multishape.py) | `h64_mixed` | GB200 | CUDA events | 206.077 | FlashKDA | 640.273 | 3.107x | `run_bench`; event timer because the fixed route launches two concurrent kernels |
| [`curated_kda_forward_portfolio_multishape`](kda_forward_portfolio_multishape.py) | `h64_uniform` | GB200 | CUDA events | 208.772 | FlashKDA | 470.625 | 2.254x | `run_bench`; event timer because the fixed route launches two concurrent kernels |
| [`curated_kda_forward_portfolio_multishape`](kda_forward_portfolio_multishape.py) | `all-six-geomean` | GB200 | CUDA events | 262.082 | FlashKDA | 730.288 | 2.786x | geometric mean of the six rows above |
| [`curated_moe_fp8_blockscale_dsv3`](moe_fp8_blockscale_dsv3.py) | `t14107` | B200 | CUPTI | 717.491 | FlashInfer TRT-LLM FP8 MoE | 2239.532 | 3.121x | `moe-20260908-004417` default-max row in full-sweep promotion run |
| [`curated_alphamoe_fp8_blockscale_qwen3next`](alphamoe_fp8_blockscale_qwen3next.py) | `m1_official` | GB200 | CUDA event | 10.375 | FlashInfer `trtllm_fp8_block_scale_routed_moe` | 36.239 | 3.4929x | Same-worker 7-round mean, cold L2. Pure GPU (Proton): 6.316 us original BF16 atomic / 6.519 us revised FP32 sum (+3.22%). CUDA-event pair below uses graph-captured FlashInfer. |
| [`curated_alphamoe_fp8_blockscale_qwen3next`](alphamoe_fp8_blockscale_qwen3next.py) | `m8_official` | GB200 | CUDA event | 20.775 | FlashInfer `trtllm_fp8_block_scale_routed_moe` | 50.431 | 2.4275x | Same-worker 7-round mean, cold L2. Pure GPU (Proton): 16.566 us original BF16 atomic / 17.014 us revised FP32 sum (+2.70%). CUDA-event pair below uses graph-captured FlashInfer. |
| [`curated_alphamoe_fp8_blockscale_qwen3next`](alphamoe_fp8_blockscale_qwen3next.py) | `m16_official` | GB200 | CUDA event | 28.795 | FlashInfer `trtllm_fp8_block_scale_routed_moe` | 62.625 | 2.1749x | Same-worker 7-round mean, cold L2. Pure GPU (Proton): 23.197 us original BF16 atomic / 24.082 us revised FP32 sum (+3.81%). CUDA-event pair below uses graph-captured FlashInfer. |
| [`curated_alphamoe_fp8_blockscale_qwen3next`](alphamoe_fp8_blockscale_qwen3next.py) | `m32_official` | GB200 | CUDA event | 41.852 | FlashInfer `trtllm_fp8_block_scale_routed_moe` | 81.551 | 1.9486x | Same-worker 7-round mean, cold L2. Pure GPU (Proton): 35.662 us original BF16 atomic / 37.172 us revised FP32 sum (+4.23%). CUDA-event pair below uses graph-captured FlashInfer. |
| [`curated_alphamoe_fp8_blockscale_qwen3next`](alphamoe_fp8_blockscale_qwen3next.py) | `m64_official` | GB200 | CUDA event | 57.358 | FlashInfer `trtllm_fp8_block_scale_routed_moe` | 101.126 | 1.7631x | Same-worker 7-round mean, cold L2. Pure GPU (Proton): 50.256 us original BF16 atomic / 52.710 us revised FP32 sum (+4.88%). CUDA-event pair below uses graph-captured FlashInfer. |
| [`curated_alphamoe_fp8_blockscale_qwen3next`](alphamoe_fp8_blockscale_qwen3next.py) | `m128_official` | GB200 | CUDA event | 68.520 | FlashInfer `trtllm_fp8_block_scale_routed_moe` | 118.180 | 1.7248x | Same-worker 7-round mean, cold L2. Pure GPU (Proton): 61.250 us original BF16 atomic / 64.277 us revised FP32 sum (+4.94%). CUDA-event pair below uses graph-captured FlashInfer. |
| [`curated_alphamoe_fp8_blockscale_qwen3next`](alphamoe_fp8_blockscale_qwen3next.py) | `all-six-geomean` | GB200 | CUDA event | 31.732 | FlashInfer `trtllm_fp8_block_scale_routed_moe` | 69.381 | 2.1865x | Numerical revision of optimization run `alphamoe-20260916-193749`, member `cluster-split`: unweighted BF16 expert results, fixed-order FP32 weighted FMAs, and one final BF16 rounding. All eight `run_test` configurations pass, including exact cancellation and subnormal arithmetic; repeated outputs are bitwise identical. Every official shape is within the 5% pure-GPU regression limit in this same-worker seven-round measurement (maximum 4.94%); the M=64/128 margin is small. All six shapes pass Memcheck and Synccheck; epoch-wrap and mutable-input stress checks also pass. Racecheck is inconclusive: the M=1 worker exited during a node outage, and the bounded M=64 check timed out after 240 s. These CUDA-event columns measure the FlashInfer speedup, not that acceptance gate. The broad quantized oracle bound allows upstream GEMM/FP8 rounding differences; stagewise arithmetic precision supplies the numerical argument. |
| [`curated_kda_backward_packed`](kda_backward_packed.py) | `packed_1024x8_h96` | GB200 | Proton | 874.732 | FLA `chunk_kda_bwd` | 7671.740 | 8.770x | PR #206 numerical and synchronization repair sweep; same-run pair, three counter-rotated orders |
| [`curated_kda_backward_packed`](kda_backward_packed.py) | `all-seventeen-geomean` | GB200 | Proton | 266.822 | FLA `chunk_kda_bwd` | 1830.163 | 6.859x | geometric mean of all seventeen same-run pairs; `p06` uses a clean candidate/reference run that does not launch the hang-prone old baseline |
| [`curated_msa_prefill_multishape`](msa_prefill_multishape.py) | `flat_bf16_b1_q4096_kv4096_h64` | GB200 | Proton | 189.102 | MiniMax MSA | 567.112 | 2.999x | PR #210 numerical-fix sweep; same-run pair, rounds=5, cooldown=1s |
| [`curated_msa_prefill_multishape`](msa_prefill_multishape.py) | `flat_fp8_b3_q1024_kv8192_h32` | GB200 | Proton | 146.047 | MiniMax MSA | 173.521 | 1.188x | PR #210 numerical-fix sweep; same-run pair, rounds=5, cooldown=1s |
| [`curated_msa_prefill_multishape`](msa_prefill_multishape.py) | `paged_bf16_b3_q4096_kv8192_h8` | GB200 | Proton | 93.250 | FlashInfer trtllm-gen | 418.663 | 4.490x | PR #210 numerical-fix sweep; same-run pair, rounds=5, cooldown=1s |
| [`curated_msa_prefill_multishape`](msa_prefill_multishape.py) | `all-three-geomean` | GB200 | Proton | 137.071 | MiniMax MSA / trtllm-gen | 345.378 | 2.520x | geometric mean of the three same-run pairs in the PR #210 numerical-fix sweep |
| [`curated_msa_decode_multishape`](msa_decode_multishape.py) | `mtp_bf16_b128_q16_kv4096_h64` | GB200 | Proton | 207.070 | MiniMax MSA | 1207.182 | 5.830x | GB200 same-run pair; Proton, rounds=5, cooldown=1s |
| [`curated_msa_decode_multishape`](msa_decode_multishape.py) | `decode_fp8_b128_q1_kv4096_h64` | GB200 | Proton | 49.094 | MiniMax MSA | 510.456 | 10.397x | GB200 same-run pair; Proton, rounds=5, cooldown=1s |
| [`curated_msa_decode_multishape`](msa_decode_multishape.py) | `decode_fp16_b128_q1_kv4096_h64` | GB200 | Proton | 91.542 | FlashInfer trtllm-gen block-sparse | 96.231 | 1.051x | GB200 same-run pair; Proton, rounds=5, cooldown=1s |
| [`curated_msa_decode_multishape`](msa_decode_multishape.py) | `all-ten-geomean` | GB200 | Proton | 72.142 | MiniMax MSA / trtllm-gen | 289.006 | 4.006x | geometric mean of the ten official rows in that same run |
| [`curated_kda_decode_multishape`](kda_decode_multishape.py) | `t1_b128_hv32_standard` | GB200 | Proton | 41.424 | FlashInfer `recurrent_kda` | 54.189 | 1.308x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`curated_kda_decode_multishape`](kda_decode_multishape.py) | `t6_b128_hv32_spec` | GB200 | Proton | 160.250 | FlashInfer `recurrent_kda` | 222.751 | 1.390x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`curated_kda_decode_multishape`](kda_decode_multishape.py) | `t3_b16_hv16_lower_bound` | GB200 | Proton | 10.075 | FlashInfer `recurrent_kda` | 15.976 | 1.586x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`curated_kda_decode_multishape`](kda_decode_multishape.py) | `all-thirty-geomean` | GB200 | Proton | 22.153 | FlashInfer `recurrent_kda` | 29.263 | 1.321x | geometric mean of the thirty official rows in that run |
| [`curated_mla_dsv4_multishape`](mla_dsv4_multishape.py) | `prefill_h128_swa16384_topk4x_c16384_k1024_bf16_hnd` | GB200 | Proton | 79.082 | FlashInfer trtllm-gen DSv4 | 117.353 | 1.484x | PR #207 numerical-fix sweep; same-run pair, rounds=5, cooldown=1s |
| [`curated_mla_dsv4_multishape`](mla_dsv4_multishape.py) | `decode_h16_swa512_topk128x_c128_k132_fp8_hnd` | GB200 | Proton | 7.000 | FlashInfer trtllm-gen DSv4 | 15.388 | 2.198x | PR #207 numerical-fix sweep; same-run pair, rounds=5, cooldown=1s |
| [`curated_mla_dsv4_multishape`](mla_dsv4_multishape.py) | `prefill_h128_swa16384_topk4x_c16384_k1024_fp8_hnd` | GB200 | Proton | 88.025 | FlashInfer trtllm-gen DSv4 | 78.358 | 0.890x | PR #207 numerical-fix sweep; same-run pair, rounds=5, cooldown=1s |
| [`curated_mla_dsv4_multishape`](mla_dsv4_multishape.py) | `all-ninety-four-geomean` | GB200 | Proton | 9.621 | FlashInfer trtllm-gen DSv4 | 16.503 | 1.715x | geometric mean of all 94 same-run pairs in the PR #207 numerical-fix sweep |
| [`curated_vsa_multishape`](vsa_multishape.py) | `pooled_blk128_s80000_topk156` | GB200 | Proton | 4523.323 | FlashInfer `bsa_attn_fwd` | 7045.071 | 1.558x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`curated_vsa_multishape`](vsa_multishape.py) | `bsr_blk64_s4096_topk32` | GB200 | Proton | 34.869 | FlashInfer `bsa_attn_blk64_fwd` | 58.166 | 1.668x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`curated_vsa_multishape`](vsa_multishape.py) | `fastwan_blk64_s26624_h12_topk84_partial` | GB200 | Proton | 764.725 | FlashInfer `bsa_attn_blk64_fwd` | 938.857 | 1.228x | GB200 run; Proton, rounds=5, cooldown=1s |
| [`curated_vsa_multishape`](vsa_multishape.py) | `all-eighteen-geomean` | GB200 | Proton | 92.210 | FlashInfer BSA | 154.753 | 1.678x | geometric mean of the eighteen official rows in that run |

The MLA rows measure the kernel after the FP8 numerical repairs (BF16 split
partials and P scaled by 448), against FlashInfer 0.6.18.post1 on GB200 with Proton. Each of
the 94 configurations uses five same-run candidate/reference rounds and
arithmetic-mean times; the aggregate row divides their geometric means.
GPU execution was serialized by the benchmark server's per-GPU lease.
Source and measurement details are recorded in [PR #207](https://github.com/mlc-ai/TIRx-kernels/pull/207).

For `kcoral` rows the timer is the optimization harness's benchmark server, which
reports GPU-only latency for candidate and reference in one run. Geomean rows
divide the two geometric means; they are not means of per-workload ratios.

The KDA backward rows use Proton GPU-kernel time with 30 ms warmup and 100 ms
repeat budgets in each of three counter-rotated candidate/reference orders.
All seventeen rows passed correctness against FLA commit
`9c8e42e762fce087c27b673af4922795d9edb85e`. The old `p06` merge-base kernel
can hang intermittently, so its published pair comes from the clean PR/FLA run;
it measured 398.105 us against 3260.885 us (8.191x).

## Updating the table

Run all registered curated native TIRx workloads and their references through the
bench suite:

```bash
python -m tirx_kernels.bench_suite \
  --filter curated \
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
