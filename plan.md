# Bench Suite CPU/GPU 流水线实施计划

## Goal Description

将 bench suite 从“workload 先占 GPU，再在 GPU 持有期内完成 Python 启动、模块解析、TIRx specialization/IR 生成和 TVM/NVRTC 编译”的单阶段模型，改造成显式的两阶段流水线：

1. **CPU prepare**：多个独立 workload 并行完成目标模块加载、shape specialization、IR 生成和编译；该阶段不得初始化 CUDA、分配显存、运行 setup kernel 或持有 GPU token。
2. **GPU stage**：prepared child 报告 READY 后，调度器根据实时可用 GPU 数量动态分配卡；每张卡同一时刻只执行一个 GPU stage，不同空闲卡并行执行。GPU stage 保留当前完整的 CUDA allocation、reference setup/autotune、正确性 warmup、timer、rounds、cooldown、原始 round samples、aggregation 和干扰检测语义。

prepared executable 必须从 CPU prepare 到 GPU stage 始终留在同一个 child process 内，避免序列化不可移植的 runtime object。child 在 READY 后等待父进程通过独立控制通道发送 GPU assignment，确认 CUDA 尚未初始化后设置 `CUDA_VISIBLE_DEVICES`，再进入 GPU stage。

每个 workload attempt 使用一个一次性 child process：该进程只执行一次 `CPU prepare → READY → ASSIGN → GPU stage → RESULT → exit`，跑过 GPU stage 后不得回收或复用去准备另一个 workload。这里的 prepare 并发上限只约束同时存在的一次性 prepare 子进程数量，不是常驻 worker pool；实现不得引入常驻 prepare worker、worker 复用池、fork 模板，或在 orchestrator 同一解释器中用编译线程池并行执行 workload specialization/编译。

稳态目标时间模型为：

```text
T_expected =
  T(first READY)
  + GPU stages 在实际 eligible GPU 时间线上的最短可行 list-scheduling makespan

T_unexplained = T_observed - T_expected - T_foreign_wait
```

CPU prepare 只应通过首个 READY 延迟进入关键路径；首个 GPU stage 开始后，只要 READY queue 中存在可满足的 workload，内部空闲且外部 eligible 的 GPU 就不应因 CPU prepare、轮询或进程生命周期管理而产生重复空洞。外部 GPU 占用、无法满足的多卡 claim 和干扰重试必须单独记录，不得伪装成调度开销。

现有 GEMM 原型提供了方向性证据：三个 workload 串行执行为 29.70 秒，并行 CPU prepare 加单卡串行 GPU stage 为 19.17 秒，端到端 1.55x；CPU prepare 每条约 3.9–4.2 秒，GPU stage 约 3.3–4.5 秒。该数据是成本模型证据，不是跨 workload、跨机器的固定秒数要求。

同时从配置源缩减日常 measured sweep：当前默认 sweep 为 234 条 workload，其中 33 个 kernel 的 `default: true` config 超过 3 条。每个受影响 kernel 只保留生产相关的小/中/大 3 个代表点后，当前树的默认 sweep 预计为 112 条，减少 122 条（52.1%）。其余 config 只改为 `default: false`，仍须全部支持显式运行并完成 pipeline 迁移。

## Acceptance Criteria

以下测试在实现完成后的主要功能锚点执行，用作独立行为证据；不采用细粒度 TDD 循环或覆盖率目标。

- AC-1: 提供统一的两阶段 benchmark 合同，明确区分 CPU-only prepare 与 GPU-only execution。
  - Positive Tests (expected to PASS):
    - fresh child 对 `fp16_bf16_gemm` 执行目标模块加载、shape specialization、IR 生成和 TVM/NVRTC 编译后成功返回 READY，且 prepare 前后 `torch.cuda.is_initialized()` 均为 `False`。
    - prepared object 在同一 child 中接收 GPU assignment 后成功执行原有 benchmark，并产生完整 `impls`、`round_samples`、`errors`、`timer` 和 `benchmark_protocol`。
    - 现有 `run_bench(...)` public API 继续通过 prepare + run_gpu 组合提供相同行为，standalone CLI 不因内部拆分而改变接口。
  - Negative Tests (expected to FAIL):
    - prepare 阶段初始化 CUDA、分配 CUDA tensor 或触发 GPU kernel 时，child 不得发送 READY，并返回明确的 prepare protocol failure。
    - prepared object 被尝试跨进程序列化或在另一个进程执行时必须被拒绝；实现不得依赖 pickle/runtime-module 序列化。

- AC-2: prepared child 支持 READY 后的晚绑定 GPU assignment，且控制协议不受普通日志输出影响。
  - Positive Tests (expected to PASS):
    - child 在 import torch/TVM 和完成 CPU prepare 后，通过专用 pipe/socket 发送结构化 READY；父进程发送一个物理 GPU index 后，child 在首次 CUDA 初始化前设置可见设备并只在该卡运行。
    - 两个 prepared child 被分别分配到两张空闲 GPU 时可并行运行，结果记录正确的物理 GPU identity。
    - stdout/stderr 中包含任意 kernel、compiler 或 reference 日志时，READY/ASSIGN/RESULT 协议仍正确解析。
  - Negative Tests (expected to FAIL):
    - child 在 ASSIGN 前已初始化 CUDA 时必须拒绝 assignment，不得静默运行到错误 GPU。
    - 非法、重复或与 workload `num_gpus` 不匹配的 assignment 必须被协议层拒绝。
    - 未收到 assignment 的 child 不得触碰 GPU，即使进程已 READY。

- AC-3: bench-suite 调度器实现并行 prepare、READY queue 和基于实时 eligible GPU 的事件驱动 GPU dispatch。
  - Positive Tests (expected to PASS):
    - 多个 prepare child 可并行运行，最大并发和最大 READY backlog 均有界且可配置。
    - 一张 eligible GPU 时，GPU stages 严格串行；N 张 eligible GPU 时，最多 N 个单卡 GPU stages 并行运行。
    - child RESULT 释放内部 GPU ownership 后立即唤醒 dispatcher；READY queue 非空且存在可满足 job 时，目标主机上的内部 dispatch latency p95 小于 100 ms。
    - 外部 GPU 从 busy 变为 eligible 后，可通过外部状态轮询进入 pool；内部 GPU release 不依赖 `POLL_INTERVAL`。
  - Negative Tests (expected to FAIL):
    - GPU token 在 child READY 之前被获取或保留。
    - 同一物理 GPU 同时分配给两个 GPU stages。
    - READY queue 有可满足 job 且 GPU internally free/externally eligible 时，调度器仍等待固定轮询周期才派发。

- AC-4: 新流水线完整保留 benchmark 的可观测测量语义。
  - Positive Tests (expected to PASS):
    - 相同 workload 在迁移前基线与 pipeline 路径下使用相同实现集合、实现顺序、timer、warmup/repeat、rounds、cooldown、reference builder、round aggregation 和结果 schema。
    - targeted A/B 中每个实现均保留规定数量的 raw round samples，且 `_finalize_bench_record` 的验证和算术平均逻辑不变。
    - 默认 `5 rounds + 1.0s cooldown` 保持不变；此前用于诊断的 `0.1s cooldown` 不进入默认协议。
  - Negative Tests (expected to FAIL):
    - 为获得 wall-time 提升而减少 rounds、timer budget、cooldown、reference coverage 或 correctness preparation。
    - pipeline 路径缺少 reference error、protocol metadata 或 raw samples 时仍被标为 `ok`。

- AC-5: fail-fast、取消、外部干扰检测和 retry 语义覆盖 PREPARING、READY 和 RUNNING_GPU 三种生命周期状态。
  - Positive Tests (expected to PASS):
    - 一个 workload 确定性失败时，停止启动新的 prepare，取消 PREPARING/READY children，并终止其他 RUNNING_GPU process groups，最终只记录真实 failure。
    - RUNNING_GPU 检测到 foreign active PID 时，释放 GPU claim、记录 `INTERFERED` 和 intruder PIDs，并以新 child/新 attempt 从 prepare 阶段重新排队。
    - Ctrl-C 或 suite cancellation 能回收所有 child/process-group、控制 FD、临时目录和 GPU ownership。
  - Negative Tests (expected to FAIL):
    - 被取消的 READY child 后续仍收到 GPU assignment。
    - 干扰后复用可能已污染的 prepared/CUDA state 继续计时。
    - child 退出或控制通道断开后 GPU ownership 未释放。

- AC-6: 流水线具备背压和资源边界，不以无限并发换取表面速度。
  - Positive Tests (expected to PASS):
    - 一次性 prepare 子进程并发上限与 READY backlog 上限由同一调度配置拥有，并在 run metadata 中记录有效值。
    - 当 READY backlog 满时停止启动新的 prepare；GPU 消费 READY jobs 后自动恢复补充。
    - targeted large-shape run 中 host RSS、文件描述符数量和 child 数量保持在配置边界内。
  - Negative Tests (expected to FAIL):
    - workload 数量增加时无界创建 Python/TVM compiler processes。
    - 仅因 READY backlog 满而丢弃 workload 或改变其执行顺序/attempt identity。

- AC-7: phase timeline 能独立解释端到端时间和每次 GPU idle 的原因。
  - Positive Tests (expected to PASS):
    - 每条 workload 至少记录 `process_started`、`prepare_started`、`ready`、`assigned`、`gpu_started`、`gpu_finished`、`result_received`、`process_reaped`。
    - suite 汇总 first-READY latency、prepare concurrency、READY starvation、GPU execution、external occupancy wait、interference retry、dispatch latency 和 finalization tail。
    - 基于 timeline 可计算 `T_expected`、`T_foreign_wait` 和 `T_unexplained`，并区分调度器空洞与外部负载。
  - Negative Tests (expected to FAIL):
    - 将 foreign GPU busy 时间计为内部 scheduler overhead。
    - timeline 时间戳逆序、缺少必要 transition 或同一 GPU 的 RUNNING interval 重叠时仍接受 run。

- AC-8: 每个 public kernel name 都可精确解析到唯一目标 module，且解析不依赖 import 整个 kernel tree或第二份手写真相。
  - Positive Tests (expected to PASS):
    - direct-path 和 alias-named kernel 均从源码中唯一权威的 `KERNEL_META` 派生 public-name → module 映射；pipeline child 只 import 命中的目标 module。
    - import 后 runtime `KERNEL_META` 与静态索引完全一致；所有 bench config 中的 kernel 均精确解析且无重复 public name。
    - 索引缓存以源码 identity/mtime 或等价 provenance 失效，不会把旧 module mapping 带入新 run。
    - `registry.load_kernel()` 可复用同一解析 primitive 或保持兼容包装，但不得继续在 exact-load 热路径 import 全 kernel tree。
  - Negative Tests (expected to FAIL):
    - 为 alias kernel 在 YAML 或独立 manifest 中维护重复的手写 module path 映射。
    - duplicate public name、非字面量/非法 metadata、静态索引与 runtime metadata 不一致时仍启动 prepare。
    - pipeline worker 为定位任一 kernel 而 import 整个 kernel tree。

- AC-9: 所有 benchable kernel 和 workload 全部使用两阶段 pipeline，不存在 one-stage fallback。
  - Positive Tests (expected to PASS):
    - capability gate 遍历 `bench_suite/config/**/*.yaml` 中全部 benchable config，确认其 kernel module 提供可执行的 CPU prepare/GPU stage adapter；缺一项即失败。
    - event/proton/cudagraph_proton、Kineto、MegaMoE、custom reference builder、alias-named、单 GPU 和多 GPU workload 全部进入统一 PREPARING → READY → ASSIGNED → RUNNING_GPU 生命周期。
    - 每条 run result 记录 `execution_mode: pipeline`；默认 sweep 和任意显式 workload 文件中不得出现 `legacy`、`fallback` 或 one-stage execution mode。
    - distributed rank workers 在获取完整原子 GPU claim 后才启动 CUDA/rank runtime，并保留 barrier、sample-wise max、Kineto span 和 process-group cleanup 语义。
  - Negative Tests (expected to FAIL):
    - 任一 kernel 缺少 adapter 时静默回退旧 `run_bench` 一阶段路径。
    - alias、distributed、Kineto 或 MegaMoE workload 因实现困难保留 GPU-held CPU prepare。
    - capability gate 只覆盖 default=true 行而遗漏可通过 `--workloads` 运行的 benchable config。

- AC-10: targeted 性能验证满足结构性关键路径目标，而非只改善代理指标。
  - Positive Tests (expected to PASS):
    - 在固定 B200 runner 上，只用少量代表性单卡 workload、默认 5 rounds/1.0s cooldown 做迁移前基线与 pipeline A/B；pipeline wall time 可由 timeline 复现。
    - 排除明确记录的 foreign wait 后，`T_unexplained <= max(0.5 seconds, 5% of T_observed)`。
    - 首个 GPU stage 开始后，在 prepare concurrency 足够且 READY backlog 未被配置限制的 targeted matrix 中，不出现由 CPU prepare starvation 导致的 recurring GPU idle gap。
    - 单卡运行和不占多卡的 scheduler/state-machine evidence 共同证明：同卡 GPU stages 不重叠、完整原子 claim 无法满足时不部分分配、外部 eligibility 变化仍通过统一 dispatch 合同处理。
    - pipeline A/B 的实现性能 ratio 没有可复现的系统性偏移；任何差异必须落在重复测量噪声内或有独立解释。
    - 多卡 runtime 验证状态必须写为 `exempted_by_human_unmeasured`，并单列迁移内容、结构性验证与未实测事实；不得写成 pass、`MISSING`、0 或空值。
  - Negative Tests (expected to FAIL):
    - 只报告 compiler 并发数、进程数或 GPU call 数下降，而没有端到端 wall time 和 correctness/ratio 证据。
    - 使用全量 suite 作为开发循环，或在有其他 session 跑 suite 时占用所有 GPU。
    - 通过修改 benchmark protocol 获得性能验收结果。

- AC-11: 默认 measured sweep 对每个 kernel 最多保留小/中/大 3 个代表 config，同时完整保留显式 benchmark 能力。
  - Positive Tests (expected to PASS):
    - `bench_suite/config/**/*.yaml` 中每个 kernel 的 `default: true` 数量不超过 3；当前配置树生成的默认 workload 数量从 234 变为 112。
    - 当前 33 个受影响 kernel 各保留 exactly 3 个 default config，经逐 kernel 审查分别代表低、中、高语义工作量，并尽可能覆盖不同生产 dispatch regime。
    - `load_config_dir()` 生成的默认 sweep 只包含选定代表点；`load_kernel_configs()` 和显式 `--workloads` 仍可解析并执行所有未选 config。
    - 所有 `default: false` config 仍通过 AC-9 的 all-config pipeline capability gate，不因退出日常 sweep 而失去 benchmark 支持。
    - 选择结果只由各 kernel YAML 中现有 `default` flags 拥有；报告或校验工具从这些 flags 派生，不维护第二份 selection manifest。
  - Negative Tests (expected to FAIL):
    - 任一 kernel 在默认 measured sweep 中保留 4 条或更多 config。
    - scheduler 在运行时按前 3 条、随机 3 条或 label 排序动态截断 workload。
    - 为缩小默认 sweep 删除 config、删除 `BENCH_CONFIGS` entry，或使未选 config 无法显式运行。
    - small/medium/large 三点实际落在同一规模/同一特殊分支，无法代表递增工作量或主要生产 regime。

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

完成一个通用、可观测、可取消的 prepared-child protocol；所有 benchable kernel、所有 config、所有 timer/reference 组合以及单卡/多卡/distributed workload 全部迁移；调度器根据实时 GPU eligibility 在多卡上 work-conserving dispatch；内部 release 为事件驱动；外部 occupancy 轮询；完整保留当前 measurement、failure、rank lifecycle 和 interference contracts；不存在 one-stage execution path。日常默认 measured sweep 在 YAML 源头对每个 kernel 最多保留 3 个经审查的小/中/大代表点，当前树从 234 条降至 112 条；其余 config 全部保留并可显式运行。

### Lower Bound (Minimum Acceptable Scope)

与 Upper Bound 收敛：通用两阶段 primitive、同进程 prepared child、精确目标 module 解析、单卡/原子多卡晚绑定、distributed rank prepared lifecycle、动态 READY scheduler 和 timeline 全部落地；静态 capability gate 证明所有 benchable workload 均为 pipeline-only；所有超过 3 个 default config 的 kernel 已完成小/中/大三点审查和 YAML flag 收敛；targeted 单卡和各可单卡运行的 timer 家族 A/B 满足 AC-1 至 AC-11。多卡代码仍须完整迁移，但 runtime 实测按人要求豁免，并以 `exempted_by_human_unmeasured` 单列；只提交 assignment mismatch、原子 claim、rank lifecycle 等不占多卡的结构性证据。任一 workload 仍保留 GPU-held CPU prepare、one-stage fallback、默认超过 3 个 config，或未选 config 失去显式 benchmark 能力，都不满足最低范围。

### Allowed Choices

- Can use:
  - `subprocess.Popen`、专用 pipe/socketpair、结构化 JSON/message framing 或等价的本地 IPC。
  - 同一 prepared child 内设置 late-bound `CUDA_VISIBLE_DEVICES`，前提是 CUDA 未初始化且有独立 integration evidence。
  - 有界的一次性 prepare 子进程并发、READY queue/backpressure、condition/event-driven internal wakeup；每个 workload attempt 一个 fresh process，RESULT 后直接退出。
  - 从源码中的 canonical `KERNEL_META` 派生完整 public-name → module 索引；runtime import 后必须复核。
  - 对 local、distributed、Kineto 和 MegaMoE 使用不同 adapter 实现，只要都满足相同两阶段生命周期合同。
  - 直接调整各 kernel YAML 的现有 `default` flags；使用 shape、数据量、序列长度、group/expert 数、dispatch path 和实测 GPU-stage cost 辅助判断小/中/大代表性。
- Cannot use:
  - 在 CPU prepare 前获取 GPU、在 READY 前初始化 CUDA、或把 compiled executable 跨进程序列化。
  - 通过减 rounds、cooldown、timer budget、reference coverage 或 correctness work 加速。
  - 无限 child 并发、无限 READY backlog、固定 GPU pinning、静态 workload→GPU 映射。
  - 为 public kernel name 再维护一份手写 module manifest/YAML module map。
  - 任何 one-stage、legacy 或 fallback execution mode。
  - 常驻 prepare worker、worker/process 复用池、fork 模板，或在 orchestrator 同一解释器中并行 specialize/编译多个 workload 的线程池。
  - 在 scheduler 中运行时截断 config，维护第二份默认 selection 文件，或删除未选 config。
  - 在实现迭代中运行全量 suite。

## Feasibility Hints and Suggestions

> 本节是可行性提示，不是额外强制设计；实现仍以 acceptance criteria 和证据为准。

### Conceptual Approach

建议把 child 生命周期建模为单向状态机：

```text
SPAWNED
  -> PREPARING_CPU
  -> READY
  -> ASSIGNED(gpu_indices)
  -> RUNNING_GPU
  -> RESULT
  -> REAPED

任意非终态 -> CANCELLED / FAILED
RUNNING_GPU + foreign activity -> INTERFERED -> REAPED -> new attempt
```

父进程负责：

1. 根据 prepare concurrency 和 READY backlog 补充 child。
2. 读取独立 control channel 上的 READY/RESULT/FAIL 消息。
3. 在 READY 与 GPU release 事件上运行 dispatcher；只有外部 GPU 状态变化使用定时 polling。
4. assignment 前执行现有 foreign-process precheck，原子记录 ownership，再发送物理 GPU indices。
5. RESULT 到达且 GPU work 已同步完成后立即释放 ownership并派发下一条；child 日志/退出回收异步进行。
6. RESULT 写出后采用最小化 child teardown，并测量 `gpu_finished -> process_reaped` tail；不得让普通 Python/CUDA destructor tail 阻塞下一条 GPU stage。

child 负责：

1. 在未绑定 GPU 的 fresh process 中只 import 目标 module。
2. 执行 `prepare_bench` 并验证 CUDA initialization state 未变化。
3. 发送 READY，阻塞等待 ASSIGN/CANCEL。
4. ASSIGN 后设置可见设备、再次验证尚未初始化 CUDA，再执行 `prepared.run_gpu(...)`。
5. Proton finalize、CUDA synchronize 和结果落盘完成后发送 RESULT；之后不再持有 GPU eligibility。

### Relevant References

- `tirx_kernels/basic/fp16_bf16_gemm.py` - 已验证的 `prepare_bench` / `PreparedBench.run_gpu` 原型。
- `tirx_kernels/runner.py` - 适合承载通用 prepared benchmark protocol 和 public `run_bench` composition helper。
- `tirx_kernels/bench/__main__.py` - 当前 workload child CLI；需要增加内部 prepare-worker mode 和独立控制 FD。
- `tirx_kernels/bench_suite/run.py` - 当前 GPU pool、动态 eligibility、process monitoring、fail-fast 和 result aggregation 的权威实现面。
- `tirx_kernels/registry.py` - 需要提炼 canonical exact module index，让 exact load 和 pipeline prepare 不再 import 全 kernel tree。
- `tirx_kernels/bench_suite/config/` - workload/module bucket 范围；注意部分 public kernel name 与 Python filename 不同，不能假设路径直接等于 module identity。
- `tirx_kernels/bench_suite/config/**/*.yaml` - 默认 measured coverage 的唯一权威；小/中/大选择只通过现有 `default` flags 表达。
- `tirx_kernels/bench_suite/README.md` - 最终记录 pipeline-only 两阶段模型、phase timing 和新调度参数。

## Dependencies and Sequence

### Milestones

1. 建立可归因 baseline：在不改变调度行为的前提下加入 phase timestamp 和 run-level cost model。
   - Phase A: 记录当前 process start、GPU acquire、child start/result/reap 和 external wait。
   - Phase B: 用少量 GEMM/elementwise/reference-heavy workloads确认旧路径的 GPU-held CPU time和 finalization tail。

2. 审查并缩减默认 measured config 集合，不改变完整 benchmark 能力。
   - Phase A: 枚举每个 kernel 的全部 config、当前 default 集合、规模轴、production dispatch regime 和已有 baseline GPU time。
   - Phase B: 对 33 个 default config 超过 3 条的 kernel，逐个确定小/中/大代表点并记录选择理由；不能用统一的首/中/尾机械规则替代语义判断。
   - Phase C: 只修改 YAML `default` flags，加入每 kernel `default <= 3`、当前 generated count = 112、未选 config 仍可解析的静态 gate。

3. 固化通用两阶段合同：从 GEMM 原型提炼 runner primitive，保持 standalone API 兼容。
   - Phase A: 定义 prepared object/protocol，而不是让 scheduler理解各 kernel 内部细节。
   - Phase B: 加入 CPU-only guard、late-binding prerequisite 和 result schema equivalence。

4. 实现全 kernel exact module resolution 和 prepared-child IPC。
   - Phase A: 从 canonical `KERNEL_META` 派生完整 public-name → module 索引，处理 alias、duplicate 和 cache invalidation，runtime import 后复核。
   - Phase B: 实现 READY/ASSIGN/CANCEL/RESULT framing、独立日志通道和 process-group lifecycle。
   - Phase C: 独立验证 late `CUDA_VISIBLE_DEVICES` binding；验证失败时停止该方向，不得静默降级。

5. 将 bench-suite 调度器改造成有界 producer/consumer pipeline。
   - Phase A: 并行 PREPARING children 与 bounded READY queue。
   - Phase B: event-driven internal GPU release/dispatch，保留 external occupancy polling。
   - Phase C: 多空闲卡并行、动态 busy/eligible 变化、GPU ownership invariants。

6. 恢复完整 failure semantics 和资源治理。
   - Phase A: PREPARING/READY/RUNNING_GPU cancellation 和 fail-fast。
   - Phase B: interference kill/requeue、attempt identity、temporary resource cleanup。
   - Phase C: backpressure、host RSS/FD/process bounds 和 minimal finalization tail。

7. 分波完成全 kernel 迁移，不在调度器中堆 workload-specific fast path。
   - Phase A: `fp16_bf16_gemm` 完成正式迁移并删除仅为实验存在的局部结构。
   - Phase B: 迁移一个 elementwise/quantization kernel和一个复杂 reference-builder kernel，修正通用合同缺口。
   - Phase C: 迁移全部 local single-GPU kernels，包括所有可由显式 workload 文件选择的非 default config。
   - Phase D: 迁移 Kineto、MegaMoE 和 multi-GPU/distributed rank lifecycles，删除 one-stage scheduler path。

8. 在主要功能锚点做独立验证并收口。
   - Phase A: protocol/state-machine/metadata行为测试。
   - Phase B: 少量单卡 workload 和动态外部负载 A/B；迁移前一侧从保留 one-stage scheduler 的 `a91a1b7` 独立 worktree 运行，迁移后一侧从当前 pipeline worktree 运行。两侧必须使用完全相同的显式 workload matrix、物理 GPU、默认 5 rounds/1.0s cooldown、timer/reference/correctness 协议和外层 wall timer；不跑全量、不占多张卡。
   - Phase C: 对多卡路径只做 assignment mismatch、原子 claim 和 rank lifecycle 等无多卡结构验证，并将 runtime 状态记为 `exempted_by_human_unmeasured`。
   - Phase D: 依据 timeline 检查 `T_unexplained`、ratio、correctness和资源边界；只保留可复现的端到端收益。迁移前/后同 matrix 的 pipeline overlap 加速与 YAML 默认覆盖从 234 降到 112 的独立工作量缩减必须分别归因，不得合并、相乘或汇总成一个加速比。

## Feature Map / Capability Map

| Capability ID | Capability / Feature | Target ACs | Depends On | Context Summary | Implementation Surface |
|---------------|----------------------|------------|------------|-----------------|------------------------|
| cap1 | CPU/GPU 两阶段 benchmark 合同 | AC-1, AC-4 | - | Business: 消除 GPU 持有期内的编译空转；Design: prepare/run_gpu 正交；Implementation: compiled executable 同 child 持有 | `runner.py`, migrated kernel modules |
| cap2 | 全 kernel exact module resolution | AC-8 | cap1 | Business: CPU prepare 只加载目标 kernel；Design: direct 和 alias 共用 canonical metadata；Implementation: complete source index + runtime validation | `registry.py`, `bench/__main__.py` |
| cap3 | Prepared-child IPC 与晚绑定 GPU | AC-2 | cap1, cap2 | Business: 让 CPU prepare 与 GPU availability 解耦；Design: READY/ASSIGN/RESULT；Implementation: dedicated control FD, no executable serialization | `bench/__main__.py`, runner protocol |
| cap4 | 动态 GPU pipeline scheduler | AC-3, AC-6 | cap3 | Business: N 张空闲卡维持 N 路 GPU stage；Design: bounded READY queue + event dispatch；Implementation: pool ownership, condition wakeups | `bench_suite/run.py` |
| cap5 | Failure/interference lifecycle | AC-5 | cap3, cap4 | Business: 加速不能削弱可信度；Design: state-aware cancel/retry；Implementation: process groups, monitor, new-attempt restart | `bench_suite/run.py` |
| cap6 | Phase observability 与成本模型 | AC-7, AC-10 | cap3, cap4 | Business: 用真实 wall-time oracle验收；Design: explain every idle interval；Implementation: timestamps, run summary, model residual | run JSON/report writers |
| cap7 | 全 workload pipeline coverage | AC-9 | cap1, cap2, cap3, cap4, cap5 | Business: 所有 bench 工作都获得流水线收益；Design: pipeline-only capability gate；Implementation: local/distributed adapters and one-stage deletion | all kernel modules, distributed launchers, README |
| cap8 | 小/中/大默认 measured coverage | AC-11 | - | Business: 将日常 sweep 数量减半且保留规模代表性；Design: 每 kernel 最多 3 个 YAML-owned default 点；Implementation: semantic curation, flags, static gates | `bench_suite/config/**/*.yaml`, config loaders, README |

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | 建立旧路径 phase timeline 和 targeted cost model；结合全部 config 的规模轴、dispatch regime 与 baseline GPU time，确定逐 kernel 小/中/大默认选择 | AC-7, AC-10, AC-11 | analyze | - |
| task2 | 落实 YAML `default` flags 和静态 coverage gates，确保当前默认 sweep 为 112 条且所有未选 config 仍可显式运行 | AC-11 | coding | task1 |
| task3 | 一次性实现通用 prepared benchmark contract、canonical exact module index、prepared-child IPC、late GPU binding 和 standalone API 兼容 | AC-1, AC-2, AC-4, AC-8 | coding | task1 |
| task4 | 将调度器完整改造成有界 CPU prepare/READY/GPU pipeline，并整合动态 GPU eligibility、背压、phase telemetry、fail-fast、取消、干扰 retry 和资源回收 | AC-3, AC-5, AC-6, AC-7 | coding | task3 |
| task5 | 按通用 contract 迁移全部 local single-GPU kernel/config，修正共性缺口并由 all-config capability gate 证明无遗漏 | AC-1, AC-4, AC-8, AC-9, AC-11 | coding | task2, task4 |
| task6 | 迁移 multi-GPU、distributed、Kineto 和 MegaMoE 生命周期，保留原子 claim/rank/timer 语义并删除 one-stage execution path | AC-2, AC-3, AC-4, AC-5, AC-9 | coding | task5 |
| task7 | 执行主要行为锚点与代表性单卡/动态负载 A/B；对多卡只做不占多卡的协议/原子 claim/rank lifecycle 结构验证，并在报告中以 `exempted_by_human_unmeasured` 单列未实测事实；验证关键路径、ratio、correctness、全 config coverage 和工程原则，并更新文档 | AC-1, AC-2, AC-3, AC-4, AC-5, AC-6, AC-7, AC-8, AC-9, AC-10, AC-11 | analyze | task6 |

## Future Work / Out of Scope

- FUT-1: 修改默认 rounds、cooldown 或 timer budget。
  - Current-loop handoff: AC-4 明确保持现有 protocol。
  - Promotion trigger: 独立统计研究证明新的 measurement protocol 同时保留回归灵敏度和 ratio稳定性。
- FUT-2: CUDA kernel 本身的 NCU 优化。
  - Current-loop handoff: AC-10 只优化 suite orchestration wall time。
  - Promotion trigger: pipeline 收口后，GPU stage 已成为主导且具体 kernel 仍低效。
- FUT-3: 全量默认 sweep 和 baseline promotion。
  - Current-loop handoff: 只做少量代表性 targeted run。
  - Promotion trigger: 没有其他 session 运行 suite，且全部主要锚点验证完成。

## Claude-Codex Deliberation

### Agreements

- 用户已确认最终结构：CPU prepare 高并发，GPU stage 依可用卡数动态并行，每卡串行，不让解析/生成/编译占用 GPU critical path。
- GEMM 原型证明拆分可行且端到端收益显著，下一步应优化真实 wall-time critical path而不是代理指标。
- measurement protocol、correctness、reference coverage、fail-fast和干扰隔离不能为速度让步。
- 所有 benchable workload 必须迁移，包括 exact/alias module resolution 和 distributed rank lifecycle；不接受 fallback。
- 日常默认 measured sweep 对每个 kernel 最多保留 3 个小/中/大代表 config；当前树从 234 条降至 112 条，其余 config 保留并继续支持显式 pipeline benchmark。

### Resolved Disagreements

- “没有额外开销”的定义：不使用不可验证的绝对零开销表述；采用 `T_unexplained <= max(0.5s, 5%)`、p95 dispatch <100ms、无 recurring CPU-starvation gap 的可观测合同。
- 配置路径是否可直接映射 module：仓库中存在多处 public name/filename alias，不能直接假设；由于用户要求全迁移，选择从 canonical `KERNEL_META` 派生完整索引并 runtime复核，不新增手写 manifest。
- “小/中/大”的选择方式：不采用数组首/中/尾或 label 排序的机械规则；逐 kernel 依据语义工作量、production dispatch regime 和已有 GPU time 选择，YAML `default` flags 是唯一权威。112 是当前库存下的派生结果，长期不变量是每 kernel `default <= 3`。
- 测试方法：skill模板提到 TDD，但仓库工程原则明确禁止以 TDD/测试配额代替系统理解；计划改为实现后在主要功能锚点提交必要的行为测试，并将性能实验作为独立一次性证据。

### Convergence Status

- Final Status: `converged`

## Pending User Decisions

- 无。量化指标已按用户意图解释为：关键路径结构和每 kernel 默认最多 3 个 config 是硬要求；当前总数 112 与跨机器绝对秒数是当前树/当前 runner 的派生证据；内部 dispatch和无法解释的残差是硬验收门槛。

## Implementation Notes

### Code Style Requirements

- 实现代码和注释不得包含 `AC-`、Milestone、Task、Phase 等计划流程术语。
- 使用领域命名表达状态和合同，例如 `PreparingWorker`、`ReadyJob`、`GpuAssignment`、`ExecutionTimeline`。
- GPU ownership、CUDA initialization state、child lifecycle和result acceptance必须各有一个权威状态拥有者。
- 不保留实验 harness、临时 profile 数据或 workload-specific scheduler fast path。
- 每次性能尝试先写明 hypothesis、expected gain、single variable和stopping condition；只保留可复现的端到端收益。
