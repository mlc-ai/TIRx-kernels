# NVIDIA Thor 经典 Kernel 对比总结

这是 2026-09-04 在单张 NVIDIA Jetson AGX Thor（`sm_110a`，20 SM）上完成的最终同卡对比。这里的“完整”指已选定的 20 个经典、contract-matched 代表 workload 全部跑完并通过数值检查；不是声称仓库全部 254 个 benchmark 都存在可在 Thor 启动的外部 baseline。

Speedup 定义为 `baseline latency / TIRx latency`，大于 1.0 表示 TIRx 更快。

## 结论

| 指标 | 结果 |
|---|---:|
| 数值与执行完成 | **20/20** |
| 严格 `baseline/TIRx > 0.99` | **15/20** |
| TIRx 快超过 5% | **4/20** |
| 差距在 ±5% 内 | **16/20** |
| Baseline 快超过 5% | **0/20** |
| 几何平均 TIRx speedup | **1.072x** |
| 任一侧 CV 超过 10% | **8/20** |

最终可复现结果里没有 baseline 快超过 5% 的经典行。Fast TopK 在 Thor 上改为由已有 row 并行度决定 cluster 宽度后从 0.996x 提升到 2.833x；FA4 收敛到 0.978x，Fused Add RMSNorm 的独立 30 轮确认是 0.974x，Recurrent KDA Grouped 从 0.912x 修到 1.023x。所有短轮次看似有效、但长轮次无法复现的改动均已回退，没有为了表格数字保留不稳定优化。

如果把目标收紧为严格 `baseline/TIRx > 1.0`，后续对原先低于或接近 1.0 的 9 项做了同一轮 15-round paired recheck。Fast TopK、LayerNorm 和 Radix TopK Multi-CTA 已严格领先；其余 6 项仍是 0.967x–0.993x，不能声称已经全部快于 baseline。

| 严格复核项 | Speedup | 当前状态 |
|---|---:|---|
| Fast TopK Clusters | **2.833x** | 已优化并长轮次确认 |
| LayerNorm | **1.015x** | 已快于 baseline，无需改动 |
| Radix TopK Multi-CTA | **1.015x** | 已快于 baseline，无需改动 |
| RMSNorm Quant | **0.993x** | 尚差 0.7% |
| FlashAttention-4 | **0.982x** | 尚差 1.8% |
| QK RMSNorm | **0.980x** | 尚差 2.0% |
| Fused DiT LayerNorm | **0.980x** | 尚差 2.0% |
| RMSNorm | **0.979x** | 尚差 2.1% |
| Fused Add RMSNorm | **0.967x** | 尚差 3.3% |

这张严格复核表中的非 Fast-TopK 行来自
`/home/tlopexh/TIRx-kernels/.porting/thor_classic_strict/after-fasttopk-15r/runs/1.json`；Fast TopK 使用随后完成的完整 5-shape、15-round 受影响矩阵。

主表除 NVFP4 GEMM 和 Fused Add RMSNorm 外取自同一轮完整 20-row campaign。那两项在完整 campaign 与此前结果冲突，因此使用紧接其后的独立 30 轮成对复测；这样避免把 Thor 未锁频时的顺序/热状态漂移误报成 kernel 回归。

## 完整结果表

| Kernel | Contract-matched baseline | TIRx (µs) | CV | Baseline (µs) | CV | Speedup | 结果 |
|---|---|---:|---:|---:|---:|---:|---|
| BF16 GEMM | cuBLAS | 855.206 | 14.3% | 956.604 | 9.6% | **1.119x** | TIRx 快 11.9% |
| FP16 GEMM | cuBLAS | 1125.359 | 35.5% | 1179.606 | 22.8% | **1.048x** | ±5% |
| NVFP4 GEMM | FlashInfer CUTLASS FP4 | 355.540 | 6.9% | 366.789 | 8.7% | **1.032x** | ±5% |
| FlashAttention-4 | Upstream FA4 CuTeDSL | 957.935 | 2.4% | 937.332 | 2.1% | **0.978x** | ±5% |
| Fused Add RMSNorm | FlashInfer CuTeDSL | 14.249 | 7.0% | 13.873 | 5.4% | **0.974x** | ±5% |
| Fused DiT LayerNorm | FlashInfer CUDA | 267.990 | 0.5% | 266.755 | 0.6% | **0.995x** | ±5% |
| LayerNorm | FlashInfer CuTeDSL | 73.997 | 3.6% | 73.668 | 2.8% | **0.996x** | ±5% |
| QK RMSNorm | FlashInfer CuTeDSL | 5.505 | 3.4% | 5.270 | 2.1% | **0.957x** | ±5% |
| RMSNorm | FlashInfer CuTeDSL | 9.450 | 10.2% | 9.366 | 10.4% | **0.991x** | ±5% |
| RMSNorm Quant | FlashInfer CuTeDSL | 18.403 | 10.3% | 18.138 | 8.5% | **0.986x** | ±5% |
| GELU-and-Mul | FlashInfer CUDA | 2233.054 | 3.6% | 2233.962 | 3.4% | **1.000x** | ±5% |
| MXFP4 Quantize | FlashInfer CuTeDSL | 232.590 | 12.7% | 242.089 | 6.0% | **1.041x** | ±5% |
| NVFP4 Quantize | FlashInfer CuTeDSL | 210.232 | 7.1% | 212.055 | 13.3% | **1.009x** | ±5% |
| Fast TopK Clusters | FlashInfer | 68.033 | 8.2% | 192.713 | 1.2% | **2.833x** | TIRx 快 183.3% |
| Filtered TopK | FlashInfer | 43.536 | 5.9% | 48.248 | 2.1% | **1.108x** | TIRx 快 10.8% |
| Radix TopK Multi-CTA | FlashInfer | 65.835 | 5.4% | 64.276 | 3.2% | **0.976x** | ±5% |
| Radix TopK Single-CTA | FlashInfer | 177.806 | 2.4% | 184.081 | 2.4% | **1.035x** | ±5% |
| GDN Decode BF16 ILP4 | FlashInfer CuTeDSL | 76.791 | 9.8% | 77.799 | 10.1% | **1.013x** | ±5% |
| Recurrent KDA Grouped | FlashInfer CuTeDSL | 404.786 | 12.2% | 414.203 | 9.6% | **1.023x** | ±5% |
| Mamba SSU Horizontal | FlashInfer CUDA | 3529.556 | 0.0% | 3830.212 | 0.0% | **1.085x** | TIRx 快 8.5% |

NVFP4 的独立 30 轮同一行还测得 cuBLASLt 为 382.956 µs，即 `cuBLASLt/TIRx=1.077x`；主表按要求优先使用 FlashInfer。FA4 同一完整 campaign 行的 FlashInfer CuTeDSL 为 980.931 µs，旧一代 FA2 为 2945.198 µs；主表使用与 FA4 同代、同语义的 upstream FA4 CuTeDSL。

## Thor 调优结果

| Kernel | 原始比值 | 最终比值 | 处理 |
|---|---:|---:|---|
| RMSNorm H=4096 | 0.872x | **0.991x** | 保留 `sm_110a` 专用 ptxas register level 5；`sm_100a` 仍为 level 10 |
| NVFP4 GEMM 4096³ | 0.929x | **1.032x** | 保留 Thor persistent grid=实际 SM 数（20）；B200 的 148-CTA 配置不变 |
| Fused Add RMSNorm | 0.863x | **0.974x** | cap、register level、完整行 guard 和 async-copy 谓词化均未通过长轮次/guard gate，全部回退 |
| RMSNorm Quant | 0.880x | **0.986x** | register-level 候选的长轮次无增益，回退 |
| Recurrent KDA Grouped | 0.889x | **1.023x** | 保留 Thor 专用 phase-A 两 token staging；SM100 仍一次 stage 全部 token |
| FlashAttention-4 | 0.948x | **0.978x** | Thor causal D128 采用当前上游 192/72/56 role split；SM100 保持 200/64/48 |
| Fast TopK Clusters | 0.996x | **2.833x** | Thor 在独立 row 已提供至少 3 个 device wave、长度不超过 16K 时使用 1 CTA/row；SM100 和长行保持原启发式 |

NVFP4 的 NCU 因果证据最明确：148→20 CTA 后，寄存器、共享内存和 L2 流量不变，线程指令从 73.14M 降至 70.12M，long-scoreboard 样本从 6631 降至 5843，NCU 时间从 319.616 降至 301.088 µs；两边均为零 local load/store、零 spill。删除的是 Thor 上多余 persistent CTA 的调度工作，不是修改数学算法。

KDA 的 NCU 前后保持相同 grid、block、全局/共享访问数、228 registers 和零 spill。两-token staging 把 miscellaneous stall 从 1017 降到 0、LG-throttle 从 228 降到 95；两次独立 15 轮结果分别为 1.024x 和 1.023x，23/23 数值配置通过。

FA4 的两个 role split 都恰好使用完整 CTA register budget。Thor 改为当前上游的 192/72/56 后，两次 15 轮代表结果为 0.985x 和 0.978x；受影响的 16/16 causal D128 数值配置全部通过。`sm_100a` 路径没有改变。

Fast TopK 的 15 轮完整受影响矩阵覆盖 5 个 dtype/k 组合，全部数值通过且 speedup 为 2.198x–2.833x。代表行 NCU 中，4→1 CTA/row 后 grid 从 256 降到 64，global load 指令基本不变（41,597→41,472），warp 指令从 4.91M 降到 1.64M，共享读从 123,819 降到 28,956，barrier stall sample 从 1,442 降到 258；local load/store 与 spill 仍为零。这个优化删除的是 Thor 上已有足够 row 并行度时的重复 distributed reduction，而不是减少输入数据或放宽数值合同。

## 测量方法与限制

- `python -m tirx_kernels.bench_suite`，Proton GPU timer。
- 每个实现 1000 ms warmup、100 ms repeat、15 个 round，算术平均。
- 同一 workload 的 TIRx 与 baseline 在同一进程、同一张 GPU 上配对；编译、JIT、autotune 和 allocation 不计时。
- 完整 campaign 和两个独立确认 run 都是 0 failure、0 interference retry。
- 机器处于 MAXN，但普通用户不能锁 `jetson_clocks`。因此绝对 µs 和高 CV 行不能当作锁频数据；同进程 ratio 是本报告采用的比较量。
- 多卡 kernel 按用户要求不纳入这张经典单卡表。缺少可在 Thor 启动的同合同外部 baseline 的家族记为 N/A，不记为性能失败。

## 原始证据

- 完整 20 项复测：`/home/tlopexh/thor-validation/final-classic-after-tuning-15r/runs/1.json`
- NVFP4 独立 30 轮确认：`/home/tlopexh/TIRx-kernels/.porting/nvfp4_gemm/perf_gate/recheck-after-full-30r/runs/1.json`
- Fused Add RMSNorm 独立 30 轮确认：`/home/tlopexh/TIRx-kernels/.porting/flashinfer_fused_add_rmsnorm/perf_gate/recheck-current-after-full-30r/runs/1.json`
- FA4 独立 15 轮：`/home/tlopexh/TIRx-kernels/.porting/flash_attention4/perf_gate/thor-upstream-regs-15r/runs/1.json`
- Fast TopK 受影响 5 项 15 轮矩阵：`/home/tlopexh/TIRx-kernels/.porting/fast_topk_clusters/perf_gate/thor-row-parallel-15r/runs/1.json`
- Fast TopK 代表项独立 30 轮：`/home/tlopexh/TIRx-kernels/.porting/fast_topk_clusters/perf_gate/thor-cluster1-30r/runs/1.json`
- 更详细的 family、baseline 和 provenance 表见 [THOR_CLASSIC_BASELINE_RESULTS.md](THOR_CLASSIC_BASELINE_RESULTS.md)。
