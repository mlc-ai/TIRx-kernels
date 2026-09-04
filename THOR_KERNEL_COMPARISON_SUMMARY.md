# NVIDIA Thor 经典 Kernel 对比总结

这是 2026-09-04 在单张 NVIDIA Jetson AGX Thor（`sm_110a`，20 SM）上完成的最终同卡对比。这里的“完整”指已选定的 20 个经典、contract-matched 代表 workload 全部跑完并通过数值检查；不是声称仓库全部 254 个 benchmark 都存在可在 Thor 启动的外部 baseline。

Speedup 定义为 `baseline latency / TIRx latency`，大于 1.0 表示 TIRx 更快。

## 结论

| 指标 | 结果 |
|---|---:|
| 数值与执行完成 | **20/20** |
| 严格 `baseline/TIRx > 0.99` | **13/20** |
| TIRx 快超过 5% | **2/20** |
| 差距在 ±5% 内 | **15/20** |
| Baseline 快超过 5% | **3/20** |
| 几何平均 TIRx speedup | **1.001x** |
| 任一侧 CV 超过 10% | **5/20** |

整体几何平均基本持平。真正仍慢超过 5% 的三项是 FA4（0.948x）、Fused Add RMSNorm（0.940x）和 Recurrent KDA Grouped（0.912x）。所有短轮次看似有效、但 15 轮无法复现的改动均已回退，没有为了表格数字保留不稳定优化。

## 完整结果表

| Kernel | Contract-matched baseline | TIRx (µs) | CV | Baseline (µs) | CV | Speedup | 结果 |
|---|---|---:|---:|---:|---:|---:|---|
| BF16 GEMM | cuBLAS | 1008.219 | 33.5% | 1015.492 | 14.3% | **1.007x** | ±5% |
| FP16 GEMM | cuBLAS | 1480.844 | 29.1% | 1490.247 | 15.6% | **1.006x** | ±5% |
| NVFP4 GEMM | FlashInfer CUTLASS FP4 | 287.421 | 6.2% | 290.775 | 0.8% | **1.012x** | ±5% |
| FlashAttention-4 | Upstream FA4 CuTeDSL | 985.467 | 3.9% | 934.073 | 2.7% | **0.948x** | Baseline 快 5.2% |
| Fused Add RMSNorm | FlashInfer CuTeDSL | 11.358 | 5.5% | 10.671 | 1.9% | **0.940x** | Baseline 快 6.0% |
| Fused DiT LayerNorm | FlashInfer CUDA | 265.420 | 8.9% | 265.092 | 8.9% | **0.999x** | ±5% |
| LayerNorm | FlashInfer CuTeDSL | 85.094 | 8.6% | 83.778 | 8.5% | **0.985x** | ±5% |
| QK RMSNorm | FlashInfer CuTeDSL | 5.496 | 2.6% | 5.417 | 2.9% | **0.986x** | ±5% |
| RMSNorm | FlashInfer CuTeDSL | 9.743 | 3.5% | 9.662 | 3.2% | **0.992x** | ±5% |
| RMSNorm Quant | FlashInfer CuTeDSL | 18.370 | 9.7% | 18.014 | 9.0% | **0.981x** | ±5% |
| GELU-and-Mul | FlashInfer CUDA | 2346.760 | 7.6% | 2289.409 | 1.0% | **0.976x** | ±5% |
| MXFP4 Quantize | FlashInfer CuTeDSL | 259.440 | 22.0% | 271.856 | 18.8% | **1.048x** | ±5% |
| NVFP4 Quantize | FlashInfer CuTeDSL | 261.147 | 12.2% | 262.187 | 12.6% | **1.004x** | ±5% |
| Fast TopK Clusters | FlashInfer | 194.392 | 0.5% | 194.424 | 0.7% | **1.000x** | ±5% |
| Filtered TopK | FlashInfer | 43.005 | 3.5% | 48.077 | 1.8% | **1.118x** | TIRx 快 11.8% |
| Radix TopK Multi-CTA | FlashInfer | 59.658 | 6.1% | 59.924 | 3.0% | **1.004x** | ±5% |
| Radix TopK Single-CTA | FlashInfer | 165.448 | 4.7% | 173.161 | 3.9% | **1.047x** | ±5% |
| GDN Decode BF16 ILP4 | FlashInfer CuTeDSL | 76.087 | 6.9% | 76.039 | 8.2% | **0.999x** | ±5% |
| Recurrent KDA Grouped | FlashInfer CuTeDSL | 479.960 | 10.7% | 437.735 | 10.7% | **0.912x** | Baseline 快 8.8% |
| Mamba SSU Horizontal | FlashInfer CUDA | 3528.442 | 0.0% | 3830.180 | 0.0% | **1.086x** | TIRx 快 8.6% |

NVFP4 同一行还测得 cuBLASLt 为 276.512 µs，即 `cuBLASLt/TIRx=0.962x`；主表按要求优先使用 FlashInfer。FA4 同一行的 FlashInfer CuTeDSL 为 977.107 µs，旧一代 FA2 为 2940.869 µs；主表使用与 FA4 同代、同语义的 upstream FA4 CuTeDSL。

## Thor 调优结果

| Kernel | 原始比值 | 最终比值 | 处理 |
|---|---:|---:|---|
| RMSNorm H=4096 | 0.872x | **0.992x** | 保留 `sm_110a` 专用 ptxas register level 5；`sm_100a` 仍为 level 10 |
| NVFP4 GEMM 4096³ | 0.929x | **1.012x** | 保留 Thor persistent grid=实际 SM 数（20）；B200 的 148-CTA 配置不变 |
| Fused Add RMSNorm | 0.863x | 0.940x | cap 126、level 0 + cap 132 的 15 轮复测失败，全部回退 |
| RMSNorm Quant | 0.880x | 0.981x | register-level 候选的 15 轮复测为 0.980x，回退 |
| Recurrent KDA Grouped | 0.889x | 0.912x | level 4 的 5 轮曾为 1.021x，15 轮降至 0.901x，回退 |
| FlashAttention-4 | 0.979x | 0.948x | level 2 的 5 轮曾为 0.998x，15 轮为 0.981x，回退 |

NVFP4 的 NCU 因果证据最明确：148→20 CTA 后，寄存器、共享内存和 L2 流量不变，线程指令从 73.14M 降至 70.12M，long-scoreboard 样本从 6631 降至 5843，NCU 时间从 319.616 降至 301.088 µs；两边均为零 local load/store、零 spill。删除的是 Thor 上多余 persistent CTA 的调度工作，不是修改数学算法。

## 测量方法与限制

- `python -m tirx_kernels.bench_suite`，Proton GPU timer。
- 每个实现 1000 ms warmup、100 ms repeat、15 个 round，算术平均。
- 同一 workload 的 TIRx 与 baseline 在同一进程、同一张 GPU 上配对；编译、JIT、autotune 和 allocation 不计时。
- 两个最终 run 都是 0 failure、0 interference retry。
- 机器处于 MAXN，但普通用户不能锁 `jetson_clocks`。因此绝对 µs 和高 CV 行不能当作锁频数据；同进程 ratio 是本报告采用的比较量。
- 多卡 kernel 按用户要求不纳入这张经典单卡表。缺少可在 Thor 启动的同合同外部 baseline 的家族记为 N/A，不记为性能失败。

## 原始证据

- 13 项代表组：`/home/tlopexh/thor-validation/final-tuned-representative-15r/runs/1.json`
- 7 项补充组：`/home/tlopexh/thor-validation/final-tuned-additions-15r/runs/1.json`
- NVFP4 独立 15 轮：`/home/tlopexh/TIRx-kernels/.porting/nvfp4_gemm/perf_gate/final-clean-15r/runs/1.json`
- RMSNorm 独立 15 轮：`/home/tlopexh/TIRx-kernels/.porting/flashinfer_rmsnorm/perf_gate/final-representative-15r/runs/1.json`
- 更详细的 family、baseline 和 provenance 表见 [THOR_CLASSIC_BASELINE_RESULTS.md](THOR_CLASSIC_BASELINE_RESULTS.md)。
