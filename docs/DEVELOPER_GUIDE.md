# AgentHarness developer guide

## Architecture

The repository has two execution paths:

1. The benchmark path isolates every question in a subprocess:
   `run_subprocess` → `run_one_task` → strategy → judge.
2. The code-as-orchestration path runs trusted async Python:
   `ScriptRuntime` → `OrchestrationContext` → worker roles.

`single_agent` retains the original `react_base` workflow. The
`fixed_multi_agent` and `dynamic_multi_agent` strategies use the orchestration
runtime and predefined decomposer, solver, verifier, and synthesizer roles.

Important packages:

- `agent_harness/core`: agent loop, event bus, DAG and resource registries.
- `agent_harness/orchestration`: runtime, strategies, budgets, traces and patches.
- `benchmarks/families`: dataset registration.
- `benchmarks/judges`: benchmark-specific scoring.
- `benchmarks/runner`: per-question subprocess execution and aggregation.
- `workflows/react_base`: original single-agent ReAct baseline.

## Runtime sequence

For multi-agent strategies, a question is decomposed, solver branches run
concurrently, results are verified, and a synthesizer returns the final answer.
The dynamic strategy may patch only pending plan nodes at the four stage
boundaries. Patch validation is atomic and rejects stale versions, cycles,
unknown roles, started nodes, and budget violations.

Each multi-agent trial writes:

- `agent/final_answer.txt`: answer consumed by the existing judge.
- `trace.jsonl`: versioned event trace.
- `evolution.json`: offline diagnosis and candidate changes; never auto-deployed.
- `result.json`: score, budget, usage, patch counters and artifact paths.

The run directory also contains `results.json`, whose `evaluation` field
aggregates quality and resource use.

## Local validation

The project is Docker-first:

```bash
./scripts/docker.sh build
./scripts/docker.sh unit
./scripts/docker.sh lint
```

Offline orchestration tests use `FakeWorkerExecutor` and need no model keys.

## BrowseComp-200 smoke experiment

Place the dataset at:

```text
benchmarks/datasets/BrowseComp-200/standardized_data.jsonl
```

Copy `.env.example` to `.env` and configure a model endpoint before a real
run. Use the same seed, offset, limit, profile and budget for every strategy:

```bash
./scripts/docker.sh run python -m benchmarks.runner.run_subprocess \
  --benchmark browsecomp_200 --strategy dynamic_multi_agent \
  --profile default --limit 5 --seed 42 --concurrency 1 \
  --max-jobs 32 --max-parallel-jobs 4 --wall-time-seconds 1800 \
  --out ./results/bc200-dynamic-smoke
```

Repeat with `single_agent` and `fixed_multi_agent`. Keep the generated
`config.json`; it records the exact fixed sample and budget settings. Token
and call limits are included in total strategy cost rather than excluding
controller activity.

## Current PoC boundaries

- Orchestration code must be an imported trusted callable.
- Jobs are still executed in-process inside each isolated question worker.
- Job count, parallelism and wall time are enforced before/during execution.
  Token and call limits are checked from reported usage after calls complete.
- The offline evolution report creates candidates but does not deploy them.
- MAS-Bench and generated-code sandboxing are intentionally deferred.

# AgentHarness 开发者指南

## 架构

该存储库有两个执行路径：

1. 基准路径在子进程中隔离每个问题：
`run_subprocess` → `run_one_task` → 策略 → 判断.
2. 代码作为编排的路径运行可信的异步 Python:
`ScriptRuntime` → `OrchestrationContext` → 工作角色。

`single_agent` 保留了原始的 `react_base` 工作流程。
`fixed_multi_agent` 和 `dynamic_multi_agent` 策略使用编排
运行时和预定义的分解器、求解器、验证器和合成器角色。

重要软件包：

- `agent_harness/core`: agent 环路，事件总线，DAG 和资源注册表。
- `agent_harness/orchestration`: 运行时、策略、预算、跟踪和补丁。
- `benchmarks/families`: 数据集注册。
- `benchmarks/judges`: 基准测试特定评分。
- `benchmarks/runner`: 每个问题的子进程执行和聚合。
- `workflows/react_base`: 原始的单智能体 ReAct 基线。

## 运行时序列

对于多智能体策略，一个问题被分解，求解器分支运行
同时，结果得到验证，合成器返回最终答案。
动态策略可能仅修复四阶段中待处理的计划节点
边界。补丁验证是原子的，并拒绝过时的版本和循环。
未知的角色、已启动的节点以及预算违规。

每个多智能体试验都写入：

- `agent/final_answer.txt`: 现有裁判所消耗的答案。
- `trace.jsonl`: 版本化事件轨迹。
- `evolution.json`: 离线诊断和候选变更；从未自动部署。
- `result.json`: 分数、预算、使用情况、补丁计数器和文件路径。

运行目录还包含 `results.json`，其 `evaluation` 字段
聚合质量和资源使用。

## 本地验证

该项目以 Docker 为先：

```bash
```
./scripts/docker.sh build
./scripts/docker.sh 单元
./scripts/docker.sh lint
```

离线编排测试使用 `FakeWorkerExecutor`，不需要模型密钥。

## BrowseComp-200 烟雾实验

将数据集放置于：

```text
文本翻译器｜沉浸式翻译：免费在线 AI 双语对照
```
benchmarks/datasets/BrowseComp-200/standardized_data.jsonl
```

将 `.env.example` 复制到 `.env` 并在真实之前配置模型端点
运行。对每个策略使用相同的种子、偏移量、限制、配置文件和预算：

```bash
```
./scripts/docker.sh run python -m benchmarks.runner.run_subprocess
--benchmark browsecomp_200 --strategy dynamic_multi_agent \
--profile default --limit 5 --seed 42 --concurrency 1 \
--max-jobs 32 --max-parallel-jobs 4 --wall-time-seconds 1800 \
--out ./results/bc200-dynamic-smoke
```

使用 `single_agent` 和 `fixed_multi_agent` 重复。保持生成
`config.json`; 它记录了精确的固定样本和预算设置。Token
并且调用限制包含在总策略成本中，而不是排除
控制器活动。

## 当前 PoC 边界

- 编排代码必须是导入的受信任的可调用对象。
- 工作仍然在每个隔离的问题工作进程中执行。
- 任务数量、并行度和墙时间在执行前/执行期间被强制实施。
在通话完成后，会根据报告的使用情况检查令牌和调用限制。
- 离线演化报告会创建候选方案，但不会部署它们。
- MAS-Bench 和生成的代码沙盒化是故意推迟的。
