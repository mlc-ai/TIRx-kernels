# NVIDIA Thor TIRx FA4 versus FlashInfer FA2 (legacy control)

Measured on 2026-09-03 on one NVIDIA Jetson AGX Thor Developer Kit. This is the complete 32-config matrix exposed by the repository's `flash_attention4` module.

**Scope warning:** FlashInfer FA2 is a previous-generation control, not the primary baseline for a FlashAttention-4 kernel. Use upstream FA4 CuTeDSL for the like-for-like headline comparison. These 32 rows remain useful only for showing the generation gap.

## At a glance

| Metric | Result |
|---|---:|
| Correctness and execution | **32/32 passed** |
| TIRx faster than FlashInfer FA2 | **32/32** |
| Geometric-mean TIRx speedup | **2.187x** |
| Speedup range | **1.075x to 3.051x** |
| Rows with either CV above 10% | **3/32** |
| Failures / interference retries | **0 / 0** |

TIRx wins every tested sequence-length, GQA-ratio, and masking combination. The advantage is smallest for short causal MHA (`S=1024`, 32 KV heads) and reaches its maximum for non-causal GQA at `S=2048`, 8 KV heads.

## Sensitivity breakdown

Speedup is `FlashInfer latency / TIRx latency`; values above 1.0 favor TIRx.

### Sequence length

| Sequence length | Rows | Geomean | Minimum | Maximum |
|---:|---:|---:|---:|---:|
| 1024 | 8 | 1.514x | 1.075x | 1.872x |
| 2048 | 8 | 2.446x | 1.460x | 3.051x |
| 4096 | 8 | 2.529x | 1.880x | 2.912x |
| 8192 | 8 | 2.443x | 2.257x | 2.703x |

### KV heads (`Q heads = 32`)

| KV heads | GQA ratio | Rows | Geomean | Minimum | Maximum |
|---:|---:|---:|---:|---:|---:|
| 4 | 8:1 | 8 | 2.400x | 1.559x | 2.912x |
| 8 | 4:1 | 8 | 2.332x | 1.452x | 3.051x |
| 16 | 2:1 | 8 | 2.145x | 1.296x | 3.002x |
| 32 | 1:1 | 8 | 1.907x | 1.075x | 2.687x |

### Mask

| Mask | Rows | Geomean | Minimum | Maximum |
|---|---:|---:|---:|---:|
| non-causal | 16 | 2.374x | 1.506x | 3.051x |
| causal | 16 | 2.015x | 1.075x | 2.912x |

## Complete 32-config table

| Sequence | KV heads | Mask | TIRx µs | TIRx CV | FlashInfer FA2 µs | FI CV | TIRx speedup |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1024 | 4 | non-causal | 213.455 | 5.4% | 399.610 | 0.4% | **1.872x** |
| 1024 | 4 | causal | 163.530 | 6.1% | 254.951 | 0.9% | **1.559x** |
| 1024 | 8 | non-causal | 227.084 | 0.3% | 398.229 | 0.2% | **1.754x** |
| 1024 | 8 | causal | 177.550 | 0.4% | 257.873 | 0.8% | **1.452x** |
| 1024 | 16 | non-causal | 225.759 | 8.1% | 400.517 | 0.7% | **1.774x** |
| 1024 | 16 | causal | 208.861 | 0.7% | 270.609 | 0.6% | **1.296x** |
| 1024 | 32 | non-causal | 277.691 | 8.4% | 418.105 | 0.7% | **1.506x** |
| 1024 | 32 | causal | 289.832 | 4.2% | 311.618 | 3.9% | **1.075x** |
| 2048 | 4 | non-causal | 512.891 | 0.4% | 1435.080 | 0.0% | **2.798x** |
| 2048 | 4 | causal | 313.986 | 3.8% | 808.120 | 0.4% | **2.574x** |
| 2048 | 8 | non-causal | 470.224 | 4.1% | 1434.839 | 0.2% | **3.051x** |
| 2048 | 8 | causal | 344.751 | 3.7% | 817.442 | 0.4% | **2.371x** |
| 2048 | 16 | non-causal | 479.536 | 3.5% | 1439.384 | 0.2% | **3.002x** |
| 2048 | 16 | causal | 379.696 | 8.6% | 816.549 | 0.2% | **2.151x** |
| 2048 | 32 | non-causal | 551.635 | 5.2% | 1441.495 | 0.4% | **2.613x** |
| 2048 | 32 | causal | 577.615 | 11.0% | 843.109 | 0.4% | **1.460x** |
| 4096 | 4 | non-causal | 1987.072 | 0.9% | 5556.003 | 0.0% | **2.796x** |
| 4096 | 4 | causal | 1011.679 | 1.6% | 2946.408 | 0.0% | **2.912x** |
| 4096 | 8 | non-causal | 2053.507 | 3.7% | 5559.313 | 0.0% | **2.707x** |
| 4096 | 8 | causal | 1050.414 | 1.5% | 2950.855 | 0.1% | **2.809x** |
| 4096 | 16 | non-causal | 2035.250 | 1.8% | 5559.780 | 0.1% | **2.732x** |
| 4096 | 16 | causal | 1514.338 | 16.4% | 2968.476 | 0.1% | **1.960x** |
| 4096 | 32 | non-causal | 2068.856 | 3.1% | 5559.409 | 0.0% | **2.687x** |
| 4096 | 32 | causal | 1607.836 | 16.9% | 3023.292 | 0.2% | **1.880x** |
| 8192 | 4 | non-causal | 9160.271 | 3.7% | 21780.844 | 0.0% | **2.378x** |
| 8192 | 4 | causal | 4179.665 | 0.6% | 11299.507 | 0.0% | **2.703x** |
| 8192 | 8 | non-causal | 8738.660 | 0.6% | 21776.037 | 0.0% | **2.492x** |
| 8192 | 8 | causal | 4513.208 | 6.1% | 11298.578 | 0.0% | **2.503x** |
| 8192 | 16 | non-causal | 9318.753 | 3.9% | 21778.469 | 0.0% | **2.337x** |
| 8192 | 16 | causal | 4696.612 | 5.1% | 11320.345 | 0.0% | **2.410x** |
| 8192 | 32 | non-causal | 9650.756 | 1.9% | 21779.002 | 0.0% | **2.257x** |
| 8192 | 32 | causal | 4604.562 | 4.9% | 11461.928 | 0.1% | **2.489x** |

## Fairness contract

Both implementations receive the same FP16 Q, K, and V storage. Every row has batch size 1, 32 query heads, head dimension 128, equal Q/KV sequence lengths, matching causal mode, NHD layout, and softmax scale `1/sqrt(128)`. The FlashInfer adapter checks its output against TIRx with `rtol=0.01, atol=0.01` before returning the timed launch closure.

JIT compilation, module lookup, temporary-buffer allocation, and output allocation are outside both timed regions. Proton measures GPU kernel time for pure launches. Each implementation receives a 1000 ms warmup and 100 ms repeat budget in each of 5 rounds; the table reports their arithmetic mean.

FlashInfer provides more than FA2: its API also exposes FA3, CUTLASS, and CuTeDSL attention backends. The pinned FA3 binary is not loadable for `sm_110a`, while its CuTeDSL path does run on Thor. FA2 is retained here solely because this historical sweep measured it; it is a kernel microbenchmark, not end-to-end serving throughput.

## Provenance

| Field | Value |
|---|---|
| GPU | NVIDIA Jetson AGX Thor Developer Kit, 20 SMs |
| CUDA architecture | `sm_110a` |
| Power/clock state | `MAXN`; dynamic clocks (`jetson_clocks` was not locked) |
| Timer | `proton` |
| TVM/TIR revision | `15b607d6` |
| TIRx-kernels revision | `503d8d50` |
| FlashInfer version / revision | `0.6.18` / `f2e04400` |
| CUDA / PyTorch | CUDA 13.1 / PyTorch 2.9.1+cu130 |

The 1,000 ms warmup substantially reduces Thor's cold-DVFS bias, but locked clocks are still preferable for publication. Rows with CV above 10% remain visible rather than being silently discarded.

## Reproduce

```bash
python -m tirx_kernels.bench_suite \
  --workloads scripts/thor_flashinfer_attention_sweep.yaml \
  --out-dir /home/tlopexh/thor-validation/flashinfer-attention-sweep \
  --with-references --timer proton --rounds 5 --cooldown 0 \
  --max-prepare-processes 1 --ready-backlog 1 --no-probe --no-report
python scripts/report_thor_attention.py \
  --run /home/tlopexh/thor-validation/flashinfer-attention-sweep/runs/1.json --output THOR_FLASHINFER_ATTENTION.md
```

## Raw evidence

- Run JSON: `/home/tlopexh/thor-validation/flashinfer-attention-sweep/runs/1.json`
- Run status: 32 `ok`, 0 failures, 0 interference retries
