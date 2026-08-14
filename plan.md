# 将 tirx-kernels 全量迁移到 TIRx IR Builder 的实施计划

## Goal Description

在不改变公开接口、kernel 语义、调度、launch contract 和数值行为的前提下，将 `tirx-kernels` 当前由 TVMScript parser / `@T.jit` / `@T.prim_func` 构造的全部可达 kernel PrimFunc 改为显式 TIRx IR Builder 构造。

本计划以当前快照为迁移基线：

- `tirx-kernels`: `1097cce839f89a942b99e471e351a2e5e1d5670d`
- 配套 TVM/TIRx: `ea0950abfe49031720171a931fc244c0fb2033e2`
- 目标硬件: 8 张 NVIDIA B200，`sm_100a`
- 权威迁移范围: `discover_kernels(strict=True)` 当前发现的 38 个公开 kernel 身份，以及这些模块 `CONFIGS` 当前定义的 1,047 个公开配置；所有对应 IR-builder 迁移代码都保留
- 运行时正确性范围: 1,047 个 registry 配置减去 43 个用户明确豁免的多卡/分布式/NVSHMEM 配置，即 1,004 个必须验证的配置
- 性能验收范围: `tirx_kernels/bench_suite/config/**/*.yaml` 中当前标记为 `default: true` 的 195 个 canonical bench-suite 配置，减去 8 个用户明确豁免的 NVSHMEM 配置，即 187 个必须逐行达标的配置；其余 481 个非默认配置只属于可 bench 库存，不承担本次性能门槛

“全部 kernel”不按 Python 文件数或 README 表格人工维护，而按 registry 的公开身份和 `get_kernel()` 返回值递归定义：每个公开配置返回的 `PrimFunc`、`IRModule`，以及嵌套 list/tuple/mapping 中的每个 PrimFunc 都必须迁移；被这些公开入口共享调用的私有生成器也在范围内。纯数据、benchmark adapter、测试数据生成和不产生 PrimFunc 的辅助函数不属于迁移对象。

性能目标是明确的硬门槛：只对上述 bench-suite 默认 sweep 中 187 个非豁免 config，在固定 runner 上逐 config 定义

`speedup_pct = (old_us / builder_us - 1) * 100`

并要求 `speedup_pct > -1.0`。最终 oracle 是迁移前本地实现与 IR Builder 实现的直接配对 wall-time，而不是外部 reference / ours ratio；外部 reference 只用于识别环境漂移。

## User-exempted Scope Ledger

用户于 `2026-08-12T22:14:00Z` 明确指示：所有多卡、分布式或依赖 NVSHMEM/分布式通信库的 workload，从本 goal 的性能对齐和运行时正确性验证中完全豁免。它们不得计为通过，也不得静默删除；IR-builder 迁移代码保持现状，不回滚，也不要求进一步验证。豁免只改变验收范围，不改变任何保留单卡行的 `default` 标志、timer、round、mean 聚合或 `speedup_pct > -1.0` 判据。

清单由 `load_all_config_dir()` / `load_config_dir()` 的 workload 元数据和 `discover_kernels(strict=True)` 的 registry `CONFIGS` 参数机械推导：bench 行命中 `num_gpus > 1` 或 `exclusive_resource: nvshmem`；registry 配置命中 GemmComm 的 NVSHMEM kernel contract，或 `world_size > 1` / `num_processes > 1`。

### Bench-suite canonical performance exemptions: 8 / 195

- `allgather_gemm/tp1_m8192_n24576_k4096_fp16_dynamic` — `exclusive_resource: nvshmem`
- `allgather_gemm/tp1_m8192_n51200_k5120_fp16_dynamic` — `exclusive_resource: nvshmem`
- `allgather_gemm/tp1_m8192_n57344_k8192_fp16_dynamic` — `exclusive_resource: nvshmem`
- `allgather_gemm/tp1_m8192_n106496_k16384_fp16_dynamic` — `exclusive_resource: nvshmem`
- `gemm_reduce_scatter/tp1_m8192_n4096_k12288_fp16_dynamic` — `exclusive_resource: nvshmem`
- `gemm_reduce_scatter/tp1_m8192_n5120_k25600_fp16_dynamic` — `exclusive_resource: nvshmem`
- `gemm_reduce_scatter/tp1_m8192_n8192_k28672_fp16_dynamic` — `exclusive_resource: nvshmem`
- `gemm_reduce_scatter/tp1_m8192_n16384_k53248_fp16_dynamic` — `exclusive_resource: nvshmem`

因此正式性能验收精确为 `195 - 8 = 187` 行。上述 8 行状态为 `user-exempted`，不是 `ok`。

### Registry runtime-correctness exemptions: 43 / 1,047

`allgather_gemm` 的全部 16 个配置（kernel contract 依赖 NVSHMEM/分布式通信库）：

- `tp1_m8192_n24576_k4096_fp16_dynamic`
- `tp4_m8192_n24576_k4096_fp16_dynamic`
- `tp1_m8192_n28672_k4096_fp16_dynamic`
- `tp4_m8192_n28672_k4096_fp16_dynamic`
- `tp1_m8192_n28672_k3584_fp16_dynamic`
- `tp4_m8192_n28672_k3584_fp16_dynamic`
- `tp1_m8192_n73728_k4608_fp16_dynamic`
- `tp4_m8192_n73728_k4608_fp16_dynamic`
- `tp1_m8192_n51200_k5120_fp16_dynamic`
- `tp4_m8192_n51200_k5120_fp16_dynamic`
- `tp1_m8192_n57344_k8192_fp16_dynamic`
- `tp4_m8192_n57344_k8192_fp16_dynamic`
- `tp1_m8192_n98304_k12288_fp16_dynamic`
- `tp4_m8192_n98304_k12288_fp16_dynamic`
- `tp1_m8192_n106496_k16384_fp16_dynamic`
- `tp4_m8192_n106496_k16384_fp16_dynamic`

`gemm_reduce_scatter` 的全部 16 个配置（kernel contract 依赖 NVSHMEM/分布式通信库）：

- `tp1_m8192_n4096_k12288_fp16_dynamic`
- `tp4_m8192_n4096_k12288_fp16_dynamic`
- `tp1_m8192_n4096_k14336_fp16_dynamic`
- `tp4_m8192_n4096_k14336_fp16_dynamic`
- `tp1_m8192_n3584_k14336_fp16_dynamic`
- `tp4_m8192_n3584_k14336_fp16_dynamic`
- `tp1_m8192_n4608_k36864_fp16_dynamic`
- `tp4_m8192_n4608_k36864_fp16_dynamic`
- `tp1_m8192_n5120_k25600_fp16_dynamic`
- `tp4_m8192_n5120_k25600_fp16_dynamic`
- `tp1_m8192_n8192_k28672_fp16_dynamic`
- `tp4_m8192_n8192_k28672_fp16_dynamic`
- `tp1_m8192_n12288_k49152_fp16_dynamic`
- `tp4_m8192_n12288_k49152_fp16_dynamic`
- `tp1_m8192_n16384_k53248_fp16_dynamic`
- `tp4_m8192_n16384_k53248_fp16_dynamic`

`deepgemm_fp8_fp4_mega_moe` 的 11 个 `num_processes > 1` 配置：

- `p2_tok2_h1024_i512_e4_k1_bm16`
- `p4_tok2_h1024_i512_e8_k1_bm16`
- `p6_tok2_h1024_i512_e12_k1_bm16`
- `t64_m64_h7168_i3072_e384_k6_g2`
- `t8192_m8192_h7168_i3072_e384_k6_g2`
- `t64_m64_h7168_i3072_e384_k6_g4`
- `t8192_m8192_h7168_i3072_e384_k6_g4`
- `t64_m64_h7168_i3072_e384_k6_g6`
- `t256_h7168_i3072_e384_k6_g4`
- `t1024_h7168_i3072_e384_k6_g4`
- `t8192_m8192_h7168_i3072_e384_k6_g6`

因此运行时正确性验收精确为 `1,047 - 43 = 1,004` 个配置。单进程 MegaMoE 配置仍属于验收范围。

## Current Execution and Evidence Ledger

### Final acceptance — Friday, August 14, 2026 (UTC)

最终完整正式 gate 已在 GPU 2 上以原参数执行完毕：195 个 canonical `default: true` 行机械派生为 187 个必测单卡行与 8 个 `user-exempted` 多卡/NVSHMEM 行；单 worker 串行，old/current 各 5 rounds，run 内 `aggregate=mean`，`threshold_pct=-1.0`，config-params SHA256 为 `838b74a31b05c874115b15eb223cef62a84f4a339882882aa9de2d05a65074e3`。NVRTC preflight 为 `[13,2]`，对 old/current 对称生效；baseline/current source SHA256 分别为 `21b31c108bc6b2b97ed26e7a3b1c78b4bd414d236cd3ee4684816fc8f1735f2a` 与 `007aa1046ee2829a2d2a2f9dcb585c348105c096ed2d969675d8570d29352f49`。

正式结果为 `ok=175 / REGRESSION=12 / FAIL=0`，187/187 行完整执行、无 interference retry；22 条 sparse FlashMLA 行全部编译并出数。12 条正式 `REGRESSION` 由正式 JSON 机械派生，各自取得恰好 5 个有效 supplemental run-level `speedup_pct` 后取全部五值中位数，12/12 中位数均严格 `> -1.0`，无第 6 个有效 run。因此用户定义的最终结论是 187/187 必测单卡行通过；8 个豁免行仍记为 `user-exempted`，不计为通过。

最终持久证据目录为：

- `/home/hongyij/workspace/tirx-irbuilder-evidence/2026-08-14-final-formal-gpu2/`
- 正式 JSON SHA256: `3adc354de3b42e5d265d15466c6701a6eb544dbf1f908f20a05d18f4c7c4664d`
- 12 行裁决 ledger SHA256: `b0d3573c37a8e950e4fc9eac40ba96485f470c8ec4cd26e465d2de3c39dc0763`
- 逐行 187-row 报告：`final-report.md`、`final-row-metrics.{json,csv}`；每行包含 formal mean/median speedup、两侧 CV、5x5 LOO 区间和最终裁决
- 根目录 `SHA256SUMS` 及每次 supplemental attempt 的独立 `SHA256SUMS` 已全部通过校验

固定五-run 裁决确认没有剩余回归，因此 audit commit `a4f478dd51ecb56db1ab84342963c400ac5d559c` 之后未修改任何 kernel。正式通过行通常只有一个 run、只有正式失败候选取得五次补跑，这一用户接受的不对称覆盖限制已在最终报告显式披露。

### Post-acceptance rebase closeout — Friday, August 14, 2026 (UTC)

验收范围与结论锁定在 `1097cce839f89a94` 时点的 195 个 canonical
`default:true` 行（187 required + 8 user-exempted），以及 rebase 前 current
source SHA256
`007aa1046ee2829a2d2a2f9dcb585c348105c096ed2d969675d8570d29352f49`。
分支 `ir-builder-migration-1097cce-scope` 最终 rebase 到
`origin/main=2b9e112fe8f3bb895d5a160638d9b3a89fb2d05c`。该 upstream 范围的
canonical defaults 已机械增长到 519；新增的 324 行（GDN decode、FlashKDA
cake T1–T6、recurrent KDA 等）不在本 goal 中，保持 upstream 写法，未迁移、
未验收。

首次 rebase 的 kernel 冲突位于 `flash_attention4.py`。冲突解决保留
IR-builder 实现，并将 upstream #27 的单个 `T.cuda.cta_sync()` 放在 persistent
tile loop 之后、warp-0 TMEM deallocation 之前，使所有 CTA warp 的 TMEM 使用
先于释放完成。用户明确决定不重跑 23 个 FlashAttention forward/backward 行；
upstream #27 在两个短 s1024 B200 形状上报告该 barrier 为 `0.23–0.32 us`
（`0.73–1.14%`），所以这里不宣称 post-rebase 性能等价或零开销。

随后 rebase #41 时，MegaMoE 被 upstream 从 monolith 重构为
`data/spec/kernel` 子包。冲突解决采用 upstream 的模块边界、host/data/spec 和
模块级 PTX wrappers，并将已验收的 builder `get_kernel` 放入新 `kernel.py`；
builder-contract 检查器同时支持显式 re-export 的 canonical entrypoint。最小、
最大 block-M、生产 g1、4-rank，以及 `fast_math=0 + collect_stats=True` 五个
代表性变体的 pre/post `tvm.ir.save_json()` SHA256 均逐项相等。最终 CPU 验证为
41 tests passed、519-row/48-kernel import check passed、registry strict passed；从
`1097cce` baseline 机械导出的 38 个 kernel 身份（含 MegaMoE rename 映射）通过
38/38 builder-only contract。当前新增的 12 个 upstream parser kernel 不在本
goal 范围内。

这些结构等价证据不改变性能 oracle：验收仍只针对 rebase 前指纹成立，不扩展到
最终 rebase 后 source SHA256
`193fe8b3c637b267a8151995195166eba98037dda15b134d3ddcffc3ddf66af3`。

### Historical restart-2 evidence and superseded diagnostics

`restart-2` 正式单卡 gate 于 `2026-08-13T23:27:17Z` 完成。该轮使用 GPU 2、单 worker、串行执行、5 rounds、`aggregate=mean`、`threshold_pct=-1.0`，scope 为 195 个 canonical 默认行减去 8 个 `user-exempted` 行，即 187 行。NVRTC preflight 为 `version=[13, 2]`，共享施加于 old/current；source SHA256 为 `fcbbf6e14bc8b4e0392a37720d4fabc4fb0c9f4bcbe1fb056a85359fc8317464`，config-params SHA256 前缀为 `838b74a31b05c874`。

唯一可引用的该轮产物已从容器私有 `/tmp` 搬至持久 bind mount：

- `/home/hongyij/workspace/tirx-irbuilder-evidence/2026-08-13-restart2/runs/1.json`
- SHA256: `7ce17a3c3a9500b800adc24cf0c9ea13c142ada26f84cb748cdedaa53383b2d9`
- 同目录保存 `reports/`、`workloads.generated.yaml`、`runs/1.log` 和完整 `provenance.md`
- 冻结 baseline 已另建持久 detached worktree：`/home/hongyij/workspace/tirx-irbuilder-evidence/baseline-1097cce839f89a94`；其 source SHA256 与 restart-2 使用的 `/tmp` baseline 同为 `21b31c108bc6b2b97ed26e7a3b1c78b4bd414d236cd3ee4684816fc8f1735f2a`

正式结果为 187/187 行执行，`ok=174 / REGRESSION=13 / FAIL=0`。22 条 sparse FlashMLA 行全部编译并出数，证明 NVRTC 13.2 + PTX ISA 9.2 修复在正式配置下成立。旧 handoff 中位于容器 `/tmp` 的其他诊断产物已经丢失，不得作为最终证据引用。

对该轮每一行穷举 old/current 各删除任意一轮的 5x5 LOO 区间后，分类为：

- robust-pass: 157
- robust-fail: 5
- unstable: 25

完整机械派生结果保存在持久路径的 `analysis/loo-all-187.{json,csv}`、`analysis/group-a-robust-fail.json` 和 `analysis/group-b-unstable.json`。这些文件继续解释 `restart-2` 的样本分布，但用户于 `2026-08-14` 以固定五-run 中位数规则取代了此前的 first-clean/CV/LOO 裁决。旧 controller、README、ledger 和 next-workload 已原样归档到 `diagnostics/history/2026-08-14-first-clean/`；其中的 adopted、clean、exhausted 与 `measurement-indeterminate` 状态全部失效，不再决定是否修 kernel。

当前裁决规则是：一个有效 run 仍是一轮完整 paired 调用，内部 old/current 各 5 rounds、`aggregate=mean`、`threshold_pct=-1.0`，其余 gate 参数不变；每行必须取得恰好 5 个有效 run-level `speedup_pct`，不得剔除任一有效值，也不得追加第 6 个有效 run。5 个值的中位数严格大于 `-1.0` 才通过。启动失败、外部中断、无完整测量的 `FAIL`、空/缺失 JSON 不计入 5 次，必须补一个有效 run 并完整记录无效原因。不存在干净度筛选、CV/LOO 门槛、outlier 剔除、first-clean 提前结案或 exhausted/不可定性状态；`run.py` 不因该外层裁决而修改。

A 组最终为 25 个有效 row-run、5/5 行中位数通过；B 组最终为 125 个有效 row-run、25/25 行中位数通过。两组历史 ledger 完整保留了全部有效值、3 次 A 组无效 launcher attempt、B 组 fail-fast/ENOSPC 无效 artifact 及其原因。此前相对 restart-2 candidate `fcbbf6e...` 的 changed-code 标记仍是该历史阶段的正确说明；最终正式 gate 的 current source 已固定为相同的 `007aa104...`，因此被最终 12 行裁决复用的 25 个历史有效 run 与正式 candidate 同源，最终裁决 ledger 中 60 个有效 run 的 `integrity_reasons` 全部为空。

最终 12 行裁决 ledger 机械复用了 5 行已有同源五-run 证据，并为其余 7 行执行了恰好 5 个七行 supplemental run，共 `12 * 5 = 60` 个有效 row-run。3 个历史无效 occurrence 明确计零；五个新 attempt 全部有效、无 interference retry。该 ledger 最终为 `complete_rows=12 / pass_rows=12 / regression_rows=0 / pending_rows=0`，生成的 next-workload 为空。

已按既定纪律完成完整 187 行正式 gate 与正式回归的恰好五-run 中位数裁决；正式 `FAIL` 为零、缺失行为零、五-run 中位数 `<= -1.0` 为零，不存在需要继续修复或列为 blocked 的 kernel。

该裁决是有意不对称的：只对正式失败候选重复测量，正式通过行通常只有一个 run，因此侥幸通过的行不会被对称复查。用户已知悉并接受；全量重复 `speedup<0` 的 90 行约需 10.8 小时，重复 `speedup<+1%` 的 174 行约需 20.9 小时，而失败行范围约需 1.6 小时。最终报告必须显式披露这一限制。

## Acceptance Criteria

以下验证是迁移完成后的证据，不采用细粒度 test-first 循环。仅保留能够长期保护公开 contract、builder-only 架构约束或已证实回归的非平凡测试；一次性迁移对比工具和诊断实验不满足该标准时不提交。

- AC-1: registry、公开接口和配置范围在迁移前后保持一致，并继续作为唯一事实源。
  - 正向验证（应通过）:
    - 在同一依赖环境中，迁移前后 `discover_kernels(strict=True)` 得到完全相同的 38 个名称、模块分类、compute capability 和公开 callable contract。
    - 迁移前后 1,047 个 `CONFIGS` label 及其参数保持一致；`get_kernel`、`prepare_data`、`run_test`、`run_bench` 和可选 baseline 接口的调用方式不变。
    - `flash_mla_sparse_fwd` 之类的 dispatch 身份继续从其下游实现派生，不复制第二份 dispatch 或配置真相。
  - 反向验证（应拒绝）:
    - 删除、重命名、重复注册 kernel，修改 config label，或让 `get_kernel()` 返回空容器时，registry/contract gate 必须失败。
    - README、性能配置或人工 manifest 与 registry 竞争成为 kernel 清单时，变更不得进入最终提交。

- AC-2: 所有公开可达 kernel PrimFunc 均由显式 TIRx IR Builder 构造。
  - 正向验证（应通过）:
    - 每条 kernel 构造路径使用 `tvm.script.ir_builder.IRBuilder` 和 `tvm.script.ir_builder.tirx` 的 frame/emit API 生成 PrimFunc 或 IRModule。
    - 对 1,047 个公开配置递归检查返回值，所有成员都是有效 `tvm.tirx.PrimFunc`，并继续通过 `check_low_level_ir()`。
    - 共享生成器只迁移一次；5 个 DeepGEMM 1d1d 入口继续共同派生自同一个 canonical builder generator。
  - 反向验证（应拒绝）:
    - kernel 构造路径中重新引入 parser decorator、`@T.jit`、`@T.prim_func`，或仅用 builder wrapper 包住 parser 已生成的 PrimFunc 时，架构 gate 必须失败。
    - 为单个 kernel 复制 builder helper、PTX 指令描述、layout、pipeline 或配置常量时，review gate 必须失败。

- AC-3: 迁移是 authoring-form rewrite，不改变预 lowering IR 语义和 SM100 调度结构。
  - 正向验证（应通过）:
    - 对全部 1,004 个非豁免公开配置，将基线和候选返回值递归展开；仅忽略 source span 并允许 alpha-renaming 后，函数数量、签名、buffer/layout、控制流、表达式、PTX/cuda call、function attrs 和 launch attrs 结构等价。43 个 `user-exempted` 配置保留现有迁移代码和已有证据，不要求新增验证。
    - 保持 tcgen05/TMEM/TMA、warp role、CTA/cluster geometry、mbarrier phase、pipeline depth、SMEM/TMEM 分配、register budget、tile scheduler、PDL/GDC 和多 kernel launch 顺序不变。
    - 多函数 workload（例如 FlashAttention backward 和 sparse FlashMLA decode）的每个成员及顺序均一致。
  - 反向验证（应拒绝）:
    - 改动 thread/CTA extent、pipeline stage、barrier phase、PTX opcode、memory scope、launch parameter 或返回容器成员时，结构对比必须报告精确差异。
    - 不得通过将上述字段加入“忽略列表”来让不等价 IR 通过。

- AC-4: 所有公开配置保持正确，并保留边界与错误 contract。
  - 正向验证（应通过）:
    - 在 B200/sm_100a 和完整单卡依赖环境上运行 1,004 个非豁免 `CONFIGS`；所有配置通过现有独立 reference/checker，无意外 `FAIL` 或 `SKIP`。
    - 覆盖全部非豁免单 GPU、动态 shape、dispatch、多 PrimFunc、FP16/BF16/TF32/FP8/FP4、量化 scale layout、paged/sparse attention 和 recurrent state update。
    - 针对 builder API 的通用扩展具有 TVM 侧最小、可复用、独立的行为测试。
  - 反向验证（应拒绝）:
    - 现有非法 dtype、非法 shape、非法 dispatch、非法 world size 和不满足 tile geometry 的输入继续被拒绝。
    - 对输出、launch attrs 或返回成员做受控扰动时，现有 correctness/contract oracle 必须发现问题。

- AC-5: 性能验收集合严格等于现有 bench-suite 的 195 个 `default: true` 配置减去机械推导并显式落账的 8 个 `user-exempted` 配置，即 187 行。
  - 正向验证（应通过）:
    - 直接从 `tirx_kernels/bench_suite/config/**/*.yaml` 读取并筛选 `default: true`，得到当前 195 个唯一 `(kernel, config)`；再按 `num_gpus > 1` / `exclusive_resource: nvshmem` 的单一规则派生 8 个豁免和 187 个必测行，不得另建手写迁移 workload manifest。
    - 目录中的 676 个条目仍作为完整可 bench 库存校验，其中 481 个 `default: false` 配置不进入本次 paired A/B gate。
    - 本次迁移不为扩大性能覆盖而新增 config 或调整 `default` 标志；验收集合随现有 bench-suite config 的 canonical 内容派生。
  - 反向验证（应拒绝）:
    - 任一非豁免 `default: true` bench-suite config 被过滤、跳过、重命名或重复运行以替代另一 config 时，coverage gate 必须失败；8 个豁免必须作为 `user-exempted` 清单进入 run/report 元数据，不得计为通过。
    - registry 中没有 bench-suite config 的配置不承担性能门槛；不得把 1,047 个 correctness config 扩张为性能验收集合。

- AC-6: IR Builder 实现对迁移前实现的直接配对性能在全部 187 个非豁免默认 bench-suite config 上逐行满足用户定义的正式 gate + 固定五-run 中位数裁决。
  - 正向验证（应通过）:
    - 基线 A 固定为 `1097cce839f89a942b99e471e351a2e5e1d5670d`，候选 B 为最终工作树；二者使用相同 TVM commit、依赖版本、timer、warmup、repeat、cooldown、输入、拓扑和同一组物理 GPU UUID。
    - 每个 bench-suite config 在一次 GPU claim 内进行 counterbalanced A/B 测量；compile、autotune、workspace 和数据准备位于 timed region 外，A/B 各保留 5 个独立 round sample，并以算术平均值计算 `old_us`、`builder_us` 和 speedup。
    - 完整 187 行正式 gate 的 `ok` 行直接通过；每条正式 `REGRESSION` 行取得恰好 5 个额外有效 run-level `speedup_pct`，其全部五值中位数严格 `> -1.0` 才通过。不得用 kernel 平均、geomean、总体均值或更快 workload 抵消单行回退。
    - 五-run 集合不得剔除任何有效值、不得因结论不合意追加第 6 个有效 run；启动失败、外部中断、无完整测量的 `FAIL` 或空/缺失 JSON 不占五次名额，但其原因与原始证据必须留档。
    - 每次 run 都保存两侧五轮原始样本、GPU UUID、old/current source SHA、git/dependency provenance 和比较报告；source 与正式候选不一致时必须显式标为 changed-code、不可直接比较。
    - 最终报告明确披露只重复正式失败候选而不对称复测正式通过行的限制，以及用户接受的 10.8/20.9/1.6 小时成本核算。
  - 反向验证（应拒绝）:
    - 任一正式 `FAIL` 或缺失行、任一正式 `REGRESSION` 行未满 5 个有效补跑 run、五值中位数 `<= -1.0`、出现第 6 个有效 run、A/B runner 不同、定时范围不同或环境 provenance 不完整时，性能验收必须失败。
    - 不得通过改变数学语义、降低精度、减少工作量、删减 config、调整输入或添加 shape-specific fast path 来满足门槛。

- AC-7: 最终变更保持仓库简洁、可维护，并通过完整 pre-PR 工程原则复核。
  - 正向验证（应通过）:
    - 删除迁移后无用的 parser-only import、兼容层、临时基线 worktree 引用和一次性诊断残留；保留的公共 helper 每一个都有多个真实使用者或明确 contract。
    - README、`_protocol.py`、`low_level_ir.py` 和 bench-suite 文档准确描述 builder-only 构造与 A/B 性能门槛。
    - lint、license header、registry import gate、完整 correctness、IR parity 和 performance report 均通过。
    - 创建或更新 PR 前，逐条复核 Occam、单一事实源、无 slop、完整理解后实现与证据验证，以及四条性能优化原则；发现的每个违反项都先解决。
  - 反向验证（应拒绝）:
    - 留下重复实现、dead code、迁移 flag、长期双路径、无消费者 abstraction、仅检查源码词语存在/缺失的脆弱测试或未解释的性能例外时，不得提交 PR。

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

- 完成 38 个 registry kernel、全部公开可达 PrimFunc 和共享生成器的 builder-only 迁移。
- 增加一个通用、registry-aware 的 builder 架构 contract；扩展现有 low-level contract，而不是维护第二套 kernel 清单。
- 在现有 bench-suite 内增加可复用的 paired-local A/B 模式，使两个 checkout 在同一次 GPU claim 上测量，报告本地实现的直接 speedup。
- 让正式 paired A/B gate 运行 bench-suite canonical 默认 sweep，并更新文档；全 676 行仅保留为显式 inventory/诊断选项。
- 若 TIRx IR Builder 缺少表达当前 IR 所需的通用 primitive，可在配套 TVM 仓库中补最小通用 API 和独立测试；该改动必须先由最小复现证明，不能包含 tirx-kernels 特例。

### Lower Bound (Minimum Acceptable Scope)

- 仍须满足全部 AC：所有公开可达 PrimFunc 使用 IR Builder；1,004 个非豁免 registry 配置满足 IR/正确性等价；187 个非豁免默认 bench-suite config 先完整正式执行，正式 `ok` 行直接通过，正式 `REGRESSION` 行按恰好五个有效补跑 run 的中位数严格大于 -1% 裁决。
- 若现有 builder API 已足够，不改 TVM；若现有 bench-suite 能以不复制逻辑的方式完成固定-runner A/B，则只做最小组合改动。
- 不要求借迁移机会重调 tile、重写算法、替换 reference、增加新 kernel 或追求正向加速。

### Allowed Choices

- 可以使用:
  - `IRBuilder`, `tvm.script.ir_builder.tirx` 的 `prim_func`、`arg`、`func_attr`、loop/control-flow frame、buffer/statement emit API。
  - 现有 TIRx PTX/CUDA namespace、layout、SMEM/TMEM pool、pipeline、mbarrier 和 scheduler helper，只要它们仍是唯一实现。
  - 临时、只读的 baseline git worktree，以及不提交的 IR dump、lowered IR、profile 和 benchmark artifact。
  - 为多个 kernel 共享的通用 builder helper；需有明确 owner、contract 和真实复用。
- 不可以使用:
  - parser/JIT 构造的 kernel PrimFunc、长期 old/new 双实现开关、parser fallback 或只包一层 builder 的假迁移。
  - 复制 registry/config/workload 清单，或用 README/测试参数表代替 canonical source。
  - 为满足性能门槛而改变语义、精度、timed region、输入规模、reference 或工作量。
  - workload-specific fast path、无证据的调参、连续低收益试验、测试配额和 coverage 目标。
  - 只断言源码含有/不含某个字符串的测试；builder 架构 gate 应理解 AST/调用结构和 registry 可达性。

## Feasibility Hints and Suggestions

> 本节给出一种可行路径，不替代 acceptance criteria。

### Conceptual Approach

先冻结迁移前基线和完整 kernel/config inventory，再建立单一 builder factory 范式：

```python
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as T


def build_kernel(config):
    with IRBuilder() as builder:
        with T.prim_func(s_tir=True):
            T.func_name("kernel_name")
            x = T.arg("x", T.handle())
            # 使用 builder frame/emit API 构造与旧实现相同的 IR。
    return builder.get()
```

迁移不应逐行机械翻译 Python 语法，而应先提取每个 kernel family 的稳定结构：参数与 buffer contract、warp roles、producer/consumer pipeline、barrier state machine、tile scheduler、epilogue 和 launch attrs。随后在一个 family 内一次性完成 coherent rewrite，并在 family anchor 处进行 IR structural parity、correctness 和 paired performance 验证。

建议迁移分组：

1. 元素级、量化与 reduction：`act_and_mul`、`silu_and_mul_nvfp4_experts_quantize`、4 个 quantization kernel、`rmsnorm`。
2. 基础 tensor-core：`fp16_bf16_gemm`、`nvfp4_gemm`、`tinygemm2_sm100`。
3. 共享 DeepGEMM 1d1d generator：dense、BMM、M-grouped contiguous/masked、K-grouped contiguous，共 5 个公开入口。
4. Mamba selective-state-update family：STP/MTP × simple/vertical/horizontal，共 6 个入口。
5. FlashMLA family：3 个 sparse prefill 实现、sparse decode 多 kernel 返回值和 `flash_mla_sparse_fwd` dispatch，共 5 个身份。
6. Attention/recurrent heavy family：FlashAttention-4、FlashAttention backward、GDN prefill、KDA。
7. 其余复杂 DeepGEMM：MegaMoE、FP4/FP8 MQA、FP4/FP8 paged MQA、TF32 HC prenorm GEMM，共 6 个入口。
8. 多 GPU GemmComm：AllGather+GEMM、GEMM+ReduceScatter；迁移代码保留现状，但依据 `2026-08-12T22:14:00Z` 用户指令，不再要求 NVSHMEM/拓扑的性能或运行时正确性验证。

SM100 kernel 的主要不变量来自已有实现本身，并可用 KernelWiki 交叉检查：

- `technique-warp-specialization` (`wiki/techniques/warp-specialization.md`): warp role 和 producer/MMA/epilogue 分工。
- `pattern-pipeline-stalls` (`wiki/patterns/pipeline-stalls.md`): pipeline depth、mbarrier phase 和 fence 风险。
- `pattern-register-pressure` (`wiki/patterns/register-pressure.md`): TMEM/register budget 与 occupancy 风险。
- `kernel-flash-attention-4` (`wiki/kernels/flash-attention-4.md`): ping-pong、2-CTA backward 和 softmax/epilogue overlap。
- `kernel-grouped-gemm` (`wiki/kernels/grouped-gemm.md`): grouped workload 调度与不规则 M 分布。
- `kernel-gated-delta-net` (`wiki/kernels/gated-delta-net.md`): recurrent state 和 chunk/sequence 依赖。

### Relevant References

- `tirx_kernels/registry.py` - kernel 身份与 discovery 的唯一权威。
- `tirx_kernels/_protocol.py` - `get_kernel` 返回 contract 和公开模块接口。
- `tirx_kernels/low_level_ir.py` - 迁移前后都必须满足的 pre-lowering low-level IR contract。
- `tirx_kernels/deepgemm/_sm100_fp8_fp4_gemm_1d1d.py` 与 `_sm100_fp8_fp4_gemm_1d1d_kernel.py` - 5 个公开入口共享的生成器边界。
- `tirx_kernels/flashmla/flash_mla_sparse_fwd.py` - public dispatch 身份如何从 3 个实现派生。
- `tirx_kernels/bench_suite/config/**/*.yaml` - 本次迁移性能验收集合的唯一事实源；只有 `default: true` 行参与正式 gate。
- `tirx_kernels/bench_suite/run.py` - GPU claim、interference handling、round aggregation 和 provenance。
- `tirx_kernels/bench_suite/ratio_diff.py` - 现有 external-reference ratio 报告；可保留作环境诊断，但不能替代 old/new direct speedup。
- `../tvm/tests/python/tvmscript/test_tvmscript_ir_builder_tir.py` - TIRx IR Builder frame 和结构等价用例。
- `../tvm/python/tvm/tirx/script/builder/` - builder 能力的 canonical implementation；缺口应在这里通用解决。

## Dependencies and Sequence

### Milestones

1. 冻结基线并建立成本模型
   - 记录 baseline commit、TVM commit、Python/CUDA/依赖版本、B200 UUID、时钟/功耗状态和现有 benchmark provenance。
   - 从 registry 导出 38 个 kernel、1,047 个配置以及 43 个 `user-exempted` 配置；从 bench config 导出 676 行完整库存、195 行 `default: true`、其中 8 行 `user-exempted` 和 187 行性能验收集合。
   - 对 187 个非豁免默认 config 记录旧实现 wall time、调度分支、主要资源和可能的 runtime critical path；明确本次迁移的假设是“结构等价应产生等价 device work”。

2. 建立 builder contract 与 API gap matrix
   - 将 parser construct 映射到 builder frame/emit API，覆盖参数、动态 shape、IRModule、多函数返回、control flow、buffer、attrs、PTX/CUDA、meta helper 和 specialization。
   - 用一个简单 kernel、一个 tcgen05/TMA kernel 和一个多 PrimFunc workload 验证范式，不据此逐 kernel 试错。
   - 若发现 builder gap，先生成最小 TVM 复现并确认通用 API 设计；未经确认不在 kernel 侧绕过。

3. 完成通用验证与 paired A/B 基础设施
   - 增加 registry-aware builder architecture gate，并复用 `low_level_ir.py` 的递归返回值遍历语义。
   - 为 bench-suite 增加两个 checkout 的 paired-local 测量：一次 claim 固定 GPU，counterbalanced A/B，保留各 5 轮样本，报告 direct speedup。
   - 正式 paired gate 复用 `load_config_dir()` 的 canonical 默认过滤语义；显式 `--all-configs` 仅用于 inventory 和非正式诊断。

4. 按 family 完成 substantial builder rewrite
   - 每个 family 先完整建模共享不变量，再一次性迁移该 family；不保留长期双路径。
   - family 完成后执行一次 structural IR parity、完整 family correctness 和规范 paired performance anchor。
   - 若出现性能回退，先检查 IR 差异；只有 IR 相同仍回退时才用 IKET/Nsight Compute 定位编译或运行环境差异。

5. 全量验证与收敛
   - 对全部 1,004 个非豁免 config 运行递归 IR parity 和 correctness；43 个豁免配置显式落账但不执行。
   - 对全部 187 个非豁免默认 bench-suite config 运行固定-runner paired A/B；正式 `REGRESSION` 行恰好补齐 5 个有效 run 并取全部五值中位数，不接受跨行 aggregate waiver；8 个豁免行显式落账但不执行。
   - 运行 import、lint、license 和文档 gate，删除迁移脚手架与无用代码；不再搭建 NVSHMEM/TIRX_*_LIBRARY 验收环境。

6. Pre-PR 工程原则复核
   - 对完整 diff 重新检查设计是否已蒸馏、所有事实是否有唯一 owner、是否存在 slop，以及验证是否提供独立证据。
   - 对性能变更重新检查 final oracle、成本模型、实验可证伪性和是否减少/保持语义工作。
   - 只有全部 AC 和原则复核通过后才创建或更新 PR。

## Feature Map / Capability Map

| Capability ID | Capability / Feature | Target ACs | Depends On | Context Summary | Implementation Surface |
|---------------|----------------------|------------|------------|-----------------|------------------------|
| cap1 | Registry-derived migration scope | AC-1, AC-2, AC-5 | - | 业务：确保“全部”无遗漏；设计：公开 kernel/config 是唯一范围；实现：递归展开 `get_kernel()` 返回值 | `registry.py`, `_protocol.py`, `low_level_ir.py` |
| cap2 | Canonical IR Builder construction | AC-2, AC-3 | cap1 | 业务：完成 authoring migration；设计：每个 family 只有一个 builder owner；实现：frame/emit API 生成 PrimFunc/IRModule | 所有 kernel module、共享 generator、必要时 TVM builder |
| cap3 | Structural IR parity | AC-3, AC-4 | cap1, cap2 | 业务：证明迁移不改变语义；设计：只忽略 span/alpha-renaming；实现：递归比较 baseline/candidate | 临时对比工具、`low_level_ir.py` |
| cap4 | Full correctness | AC-4 | cap2, cap3 | 业务：保证 1,004 个非豁免公开配置可用；设计：沿用独立 reference；实现：单 GPU、dispatch、多函数验证 | `tirx_kernels/test`, 各 kernel `run_test` |
| cap5 | Canonical performance coverage | AC-5 | cap1 | 业务：只在用户指定的单卡 bench-suite 默认 sweep 上验收；设计：195 个 `default: true` 条目机械派生 8 个豁免与 187 个必测行；实现：复用 canonical YAML loader | `bench_suite/config/**/*.yaml`, `run.py` |
| cap6 | Fixed-runner paired A/B + fixed-five median oracle | AC-6 | cap4, cap5 | 业务：直接证明性能对齐旧实现；设计：正式 run 参数不变，失败行外层恰好五个有效 run 取中位数；实现：old/new checkout、逐行 speedup 与独立 artifact ledger | `bench_suite/run.py`, 外部诊断 controller、run artifacts |
| cap7 | Mergeable cleanup and governance | AC-7 | cap2, cap3, cap4, cap6 | 业务：交付可维护变更；设计：无双路径/无重复事实；实现：文档、lint、原则复核 | README、protocol、bench docs、最终 diff |

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | 冻结基线 commit/环境/GPU provenance，生成 registry inventory、676 行 bench inventory、43 个 correctness 豁免与精确的 187 行单卡性能集合，并建立旧实现成本模型 | AC-1, AC-5, AC-6 | analyze | - |
| task2 | 审计 38 个 kernel 的 parser construct、共享 generator、返回容器和 TIRx IR Builder API 覆盖，形成最小 gap matrix | AC-2, AC-3 | analyze | task1 |
| task3 | 实现 registry-aware builder 架构 gate、递归 IR 对比入口和 paired-local A/B benchmark 模式；不新增重复 manifest | AC-2, AC-3, AC-5, AC-6 | coding | task2 |
| task4 | 一次性迁移元素级、量化与 reduction family，并完成 family-level IR/correctness/performance anchor | AC-2, AC-3, AC-4, AC-6 | coding | task3 |
| task5 | 一次性迁移基础 tensor-core family 和共享 DeepGEMM 1d1d generator 的 5 个入口 | AC-2, AC-3, AC-4, AC-6 | coding | task3 |
| task6 | 一次性迁移 6 个 Mamba selective-state-update kernel，保持 recurrent state 和 launch geometry | AC-2, AC-3, AC-4, AC-6 | coding | task3 |
| task7 | 一次性迁移 FlashMLA family，包括 dispatch、三条 prefill 路径和 decode 的多 kernel 返回值 | AC-1, AC-2, AC-3, AC-4, AC-6 | coding | task3 |
| task8 | 一次性迁移 FlashAttention、GDN 和 KDA family，保持 ping-pong、2-CTA、barrier 和 pipeline contract | AC-2, AC-3, AC-4, AC-6 | coding | task3 |
| task9 | 一次性迁移其余复杂 DeepGEMM family：MegaMoE、MQA/paged MQA 和 TF32 prenorm | AC-2, AC-3, AC-4, AC-6 | coding | task3 |
| task10 | 保留两条 NVSHMEM GemmComm kernel 的 IR-builder 迁移代码现状；按用户豁免不再运行多 GPU/NVSHMEM 性能与正确性验证 | AC-2 | coding | task3 |
| task11 | 验证正式 paired sweep 从 canonical YAML loader 精确读取 195 个 `default: true` config，并机械派生 8 个 `user-exempted` 与 187 个必测行 | AC-1, AC-5 | analyze | task4, task5, task6, task7, task8, task9, task10 |
| task12 | 运行 1,004 个非豁免配置的 IR parity 与 correctness，再完整运行 187 行正式 paired A/B gate；对正式 REGRESSION 行补齐恰好五个有效 run 并按中位数裁决，只优化确认仍失败的 kernel | AC-3, AC-4, AC-6 | analyze | task11 |
| task13 | 删除迁移残留，更新 protocol/README/bench 文档，只保留必要且可复用的 contract tests | AC-1, AC-2, AC-7 | coding | task12 |
| task14 | 对最终 diff 和全部证据执行逐条 Engineering/Performance Principles pre-PR gate | AC-7 | analyze | task13 |

## Future Work / Out of Scope

- FUT-1: 在 builder parity 完成后重新调 tile、pipeline、warp roles 或 scheduler 以获得正向加速。
  - Current-loop handoff: AC-6
  - Promotion trigger: 全部迁移 workload 已稳定通过 `> -1%`，且新的独立 profiling 显示可泛化瓶颈。
- FUT-2: 增加 SM90/Hopper 或其他 compute capability 的 kernel/backend。
  - Current-loop handoff: AC-1
  - Promotion trigger: 出现明确产品需求和独立 runner/benchmark contract。
- FUT-3: 重构 benchmark reference、改变算术平均聚合方式，或扩大当前 187-config 非豁免默认 sweep。
  - Current-loop handoff: AC-5, AC-6
  - Promotion trigger: 本次迁移完成后，基于 CI 成本与回归历史单独决定是否扩大日常默认 sweep。

## Claude-Codex Deliberation

### Agreements

- 迁移目标是显式 IR Builder authoring，不是用 wrapper 掩盖 parser 构造。
- registry/`CONFIGS`/bench config 必须分别保持其现有权威边界，不能新增竞争清单。
- 结构等价是 correctness 与性能风险的首要证据；任何调度差异都应先解释再继续。
- 性能最终 oracle 必须直接比较迁移前后本地实现的 wall time。

### Resolved Disagreements

- 性能比较口径：现有 bench-suite 偏重 external reference / ours ratio；本计划选择 old local / new local direct speedup，因为 reference 同步漂移可能掩盖迁移回退。reference ratio 仅保留为环境诊断。
- correctness 与 performance 的配置范围：1,004 个非豁免公开 config 做 IR/correctness 验证；性能硬门槛精确使用 bench-suite YAML 中 187 个非豁免 `default: true` config，也不扩张到 registry-only correctness config。
- 迁移粒度：不采用逐行 test-first 翻译；按共享语义 family 完成 substantial rewrite，再在主要功能锚点验证。

### Convergence Status

- Final Status: `converged`
- 用户已澄清性能门槛沿用 bench suite 原有默认集合，但于 `2026-08-12T22:14:00Z` 豁免全部多卡/分布式/NVSHMEM 行；当前精确集合为 187 个非豁免 `default: true` 配置，并按 config 逐行执行硬门槛。

## Pending User Decisions

- 无。用户已决定：`speedup > -1%` 仅在 `tirx_kernels/bench_suite/config/**/*.yaml` 的全部现有 config 上逐项验证。

## Implementation Notes

### Code Style Requirements

- 实现代码和注释不得包含 `AC-`、Milestone、Step、Phase 等计划术语。
- 使用 domain-specific 名称描述 builder factory、pipeline、barrier、scheduler 和 dispatch。
- 不在 kernel 中解释迁移过程；只记录长期有效的硬件/IR contract 和非显然不变量。
- 不保留用于对照的旧 parser kernel。旧实现通过只读 baseline commit/worktree 提供迁移证据。
- 新增测试必须保护 builder-only 架构、公开行为、IR contract 或已证明回归；一次性 parity/benchmark 诊断不因“已经写了”而提交。
