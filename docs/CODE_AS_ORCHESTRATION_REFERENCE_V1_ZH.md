# Code-as-Orchestration `reference_v1` 开发说明

> 2026-08-02 架构复审后，项目重新把 `reference_v1` 的三阶段原生 Python 控制流
> 作为当前主线：Decomposer 生成子问题 DAG；每个 frontier 子问题运行
> Solver–Verifier 多轮循环；Verifier critique 回到下一轮 Solver；最后统一交给
> Synthesizer。`reference_v2`–`reference_v5` 保留为历史检索流水线实验，不再作为
> 当前 smoke 默认脚本，也不再继续叠加题目语义补丁。

## 1. 当前架构边界

本次重整没有替换 AgentHarness Runtime，也没有继续向旧
`DynamicMultiAgentStrategy` 添加 BrowseComp 特例。当前主链路是：

```text
Benchmark Adapter
  → orchestrations.registry（只允许选择仓库内脚本）
  → orchestrations.reference_v1.solve(ctx)
  → ScriptRuntime / OrchestrationContext
  → AgentHarnessWorkerExecutor
  → Decomposer、Solver、Verifier、Synthesizer 的 Agent Loop 与工具
  → JSONL Trace + checkpoint.json + result.json
```

`agent_harness/orchestration/` 只负责通用执行能力：Job、并发、超时、取消、
幂等、事件、预算快照和 Checkpoint。BrowseComp 的拆解方式、对抗轮数以及综合
规则均位于 `orchestrations/reference_v1.py`，以后修改 workflow 不需要修改
Runtime。

旧 `fixed_multi_agent`、`dynamic_multi_agent` 保留用于历史对照；静态参考流程
使用 `--strategy code_orchestration --orchestration-script reference_v1`。
在线 Patch 和候选脚本自动部署目前冻结。

当前正式结构验证使用 `reference_v1`。V2–V5 仅用于历史归因对照。

结果评分同时记录 `strict_exact_match` 和 `unordered_team_match`。后者只用于诊断
“Romania vs Ireland”与“Ireland v Romania”这类无语义顺序差异，正式 `reward`
仍遵循命令选择的 benchmark scorer，不会暗中改变历史正确率口径。

v2 的 Phase 2 分两遍执行：第一遍按 DAG frontier 让每个子问题最多执行一轮，
在完整 DAG 都留下结构化状态前禁止 revision；第二遍才按 easy/medium/hard 上限
对未接受节点重试。Solver 每轮两个 Agent turn，输出至多五个候选及逐约束状态；
Evidence 固定包含 `claim/url/source/status`。

## 2. 文件职责

- `orchestrations/registry.py`：可信脚本白名单、Manifest 和源码 SHA-256。
- `orchestrations/contracts.py`：`SubtaskSpec`、`SubtaskPlanV1`、
  `SubproblemStateV1`、DAG 校验和确定性 fallback。
- `orchestrations/reference_v1.py`：三阶段可执行 Python workflow。
- `orchestrations/profiles/deepseek_v4_flash.yaml`：四个角色的模型路由。
- `orchestrations/generated/`：未来离线演化产生、尚未批准的脚本。
- `orchestrations/archive/`：未来保存历史版本、diff 和评估报告。
- `agent_harness/orchestration/context.py`：新增 `checkpoint()`、
  `load_checkpoint()`、`remaining_budget()` 和脚本事件 `emit()`。
- `agent_harness/orchestration/runtime.py`：检查点原子写入、恢复和 source hash
  防护。恢复的是显式 JSON 状态，不序列化 coroutine。
- `agent_harness/orchestration/roles.py`：Decomposer DAG schema、Solver 联网工具、
  Verifier 逐标准 verdict schema。
- `benchmarks/runner/run_subprocess.py`：总 benchmark CLI，并把脚本和 profile 参数
  传给每题子进程。
- `benchmarks/runner/run_one_task.py`：每题创建 Runtime、执行注册脚本、保存答案、
  Trace、Checkpoint 与评分。
- `tests/orchestration/test_reference_v1.py`：DAG、frontier、对抗循环、预算停止、
  模型路由、恢复和 source hash 测试。

## 3. 三阶段执行语义

Phase 1 调用 Decomposer 一次。输出至多三个互补子问题，每项包含唯一 ID、问题、
依赖、难度、成功标准、所需证据和允许工具。`validate_plan()` 检查重复 ID、未知
依赖、自依赖、环、数量和工具白名单。空输出或非法 JSON 会先进行一次通用结构
修复；仍失败时 Job 明确记录失败，脚本再使用覆盖完整原题的单子任务 fallback。
fallback 原因写入 `plan_validated`，执行器不再把空响应伪装成成功分解。

Phase 2 每次只选择所有依赖均已有结构化结果的 frontier。frontier 内用
`ctx.map()` 并行，不把有先后关系的过滤步骤错误并行。上游即使 incomplete，
其 solution、已验证证据、失败标准和不确定性也会传给下游。

每个子问题按难度运行原生 Python 循环：easy 一轮、medium 最多两轮、hard 最多
三轮。Solver 每个 Job 最多四个 Agent turn，随后 Verifier 最多两个 turn。
Verifier 拒绝后，下一轮 Solver 会收到上一轮完整 solution、critique、已确认 URL、
未满足标准和依赖结果。第一轮接受就提前退出。Worker 失败转换为 incomplete
best effort 并保存成失败轮次；若难度轮数和预算允许，下一轮 Solver 收到该错误
作为 critique 后继续尝试，不让整题静默丢失。

Phase 3 始终调用 Synthesizer。输入包含原题、完整 DAG、每个子问题的所有轮次、
accepted/incomplete 状态、局部已验证证据和失败标准。它可以使用 incomplete
结果中的已验证局部事实，但不得采用明确证伪的事实；证据不足时输出
`信息未查全`。

## 4. 预算与恢复

默认每题 72,000 Token。脚本按 Phase 1 10%、Phase 2 75%、Phase 3 至少保留
15% 的统一策略决定是否开始新一轮。开始 Solver–Verifier 前使用 12,000 Token
保守估算；预估后不能保留综合配额就停止新一轮并保留 best effort。Runtime 的
Job 数、并发和墙钟仍是硬限制，最终 Trace 继续报告 Token、LLM/工具调用超限。

每次阶段边界和每轮 Verifier 后写入 `checkpoint.json`。写入先生成临时文件再
原子替换。恢复时比较所选脚本的源码 hash；不一致会拒绝恢复。脚本从
`reference_v1_state` 和 `subtask:<id>` 读取显式状态，跳过已完成子任务，并从
下一轮继续。已经保存最终答案时恢复不会再创建 Agent Job。

## 5. Trace

实时 `orchestration_live.jsonl` 除 Runtime Job 事件外，还包含：

```text
script_started / script_completed
plan_created / plan_validated
frontier_started / frontier_completed
solver_round_started / solver_round_completed
verifier_round_completed
subtask_accepted / subtask_best_effort
checkpoint_saved / run_resumed
synthesis_started / synthesis_completed
```

脚本事件携带 `script_id`、版本和 `source_hash`；Job 事件继续携带 role、完整
instruction、状态、usage、耗时和 Agent 内部事件。每次 LLM 消息和工具调用的
详细内容仍由 `agent_traces/` 下的逐 Job JSONL 保存。最终投影写到 `trace.jsonl`，
汇总和路径进入 `result.json`。

## 6. 模型与工具

`deepseek_v4_flash.yaml` 当前把 decomposer、solver、verifier、synthesizer 都路由
到 `deepseek/deepseek-v4-flash`，但 ResourceManager 已真正按角色持有客户端。
以后只改 YAML 即可给 Verifier 换更强模型，不修改 workflow。Decomposer 和
Synthesizer 无工具；Solver 和 Verifier 可使用 `web_search`、`web_fetch`、
`run_python_code`。Decomposer 为每个 Subtask 声明的工具权限会被校验并真正下沉
到对应 Solver 和 Verifier Job，未知工具不会获得执行权限。

## 7. 离线验证与真实 smoke 命令

离线测试不消耗 API：

```bash
./scripts/docker.sh build
./scripts/docker.sh run python -m unittest discover -s tests/orchestration -v
./scripts/docker.sh lint
```

真实一题 smoke（按数据注册后的顺序选择一题）：

```bash
./scripts/docker.sh run python -m benchmarks.runner.run_subprocess \
  --benchmark browsecomp_zh \
  --strategy code_orchestration \
  --orchestration-script reference_v1 \
  --orchestration-profile deepseek_v4_flash \
  --max-tokens 72000 \
  --limit 1 --no-shuffle --concurrency 1 \
  --scoring exact_match \
  --out results/reference_v1_smoke_1
```

确认 `result.json`、`trace.jsonl`、`orchestration_live.jsonl`、`checkpoint.json`
和 `agent_traces/` 均完整后，再用固定 ID 1、6、13 或固定十题清单运行。不要在
未检查一题 Trace 前直接启动 50/200 题，也不要把旧 dynamic 策略的结果当成
`reference_v1` 结果。
# Reference v4：约束驱动检索闭环

`reference_v4` 保留 v1–v3 作为历史可复现实验，不再把检索规则继续堆叠进通用 DAG
循环。它把 BrowseComp 一类实体检索题的控制流显式写成四步 Python 编排：

```text
Decomposer 提取 C1/C2/... 约束
        ↓
两条互补 Discovery 路线并行检索（每条最多 4 turn）
        ↓
合并、按答案去重 CandidateTable
        ↓
按候选并行逐项验证（每个最多 3 turn）
        ↓
Python 重新计算 accepted → Synthesizer 输出最短答案
```

主要结构规则：

- Discovery 固定执行“联合约束宽搜—候选定向搜索—Fetch—结构化输出”，禁止凭空增加
  年份过滤条件；发现候选后立即停止宽泛 discovery。
- 两条主路线均无候选时，最多启动一次改变检索维度的 fallback；仍无候选则阻塞
  Verifier 和 Synthesizer，直接返回 `信息未查全`。
- Candidate 出现后，Verifier 必须使用 `C1/C2/...` 报告每项约束。Verifier 返回的
  `accepted` 不被采信；只有全部必需约束均为 supported，且存在非论文事实来源 URL，
  Python 才允许候选进入综合。
- 论文、Benchmark 页面和搜索摘要可用于 candidate discovery，但不能单独支持 accepted。
- 每个阶段实时写入事件和 checkpoint；恢复时复用已保存的约束、候选与验证结果。

运行一条题目的脚本参数示例：

```bash
./scripts/docker.sh run python -m benchmarks.runner.run_subprocess \
  --benchmark browsecomp_zh \
  --strategy code_orchestration \
  --orchestration-script reference_v4 \
  --orchestration-profile deepseek_v4_flash \
  --max-tokens 72000 \
  --limit 1 --no-shuffle --concurrency 1 \
  --scoring exact_match \
  --out results/reference_v4_smoke_1
```

具体数据集名、输出目录等参数以
`python -m benchmarks.runner.run_subprocess --help` 的当前输出为准。

# Reference v5：Python 控制的检索状态机

v5 冻结 v4，不再通过增加 Agent turn 或补充 Solver 提示词修复检索。核心变化是把
工具执行从 Agent Loop 提升到可信 Python 编排层：

```text
Decomposer → 原子约束 C1/C2/...
       ↓
两个 QueryPlanner → 结构化、互补、无工具的查询计划
       ↓
Python 去重/校验查询 → ctx.tool(web_search)
       ↓
ResultSelector → 中间实体、候选、信息增益
       ↓
无候选但有实体 → 最多一次 entity expansion
       ↓
Python Fetch 候选URL → 无工具 Verifier → 程序化准入
       ↓
Synthesizer
```

`ctx.tool(name, arguments, role=..., key=...)` 是新增的通用 Runtime 接口。它执行
角色权限检查、`max_tool_calls` 检查、运行内幂等复用，并实时写入
`tool_call_started/tool_call_succeeded/tool_call_failed`。所以 v5 不受 Agent Loop
“最后一轮禁止工具”的影响。

v5 的硬边界包括：每条初始路线最多2个查询，每个查询最多5条结果；查询不能重复，
也不能加入原题不存在的年份；最多进行一次中间实体跳转；最多验证3个候选；只有成功
Fetch 的候选URL同时被 Verifier 标记为 verified，且全部必需 C 编号均 supported，
候选才能进入 Synthesizer。

首次真实 smoke 暴露出路线类型如果由主脚本写死，会把针对单题的修复变成新的泛化
缺陷。当前不再固定 person/event 或任何领域词。新增无工具 `route_designer`，由模型
为每道题输出至多3条 `RouteSpec`：`pivot_entity_type`、
`expansion_entity_type`、`active_constraints`、`deferred_constraints` 和目标。
通用类型包括 person/event/work/organization/place/record/answer。Python 只校验类型
属于公共枚举、约束ID存在、路线ID唯一，并保证全部约束被分配；不替模型选择语义路线，
也不再提供足球、裁判或 Brazil 专用 fallback。每个实体仍必须携带由搜索摘要直接支持
的 `matched_constraints`；类型与 AI 已声明 RouteSpec 不一致、没有命中已知 C 编号或
没有URL的实体会被删除。模型选错路线属于可从 Trace 归因的模型能力问题，而不是代码
暗中替换路线。

当前仍有一个成本统计限制：`web_fetch` 内部使用摘要模型提取网页信息，但工具返回值
尚未携带该摘要调用的 Token usage。因此直接 Search/Fetch 的调用次数已纳入预算，
Agent LLM Token 也已纳入预算，但 Fetch 内部摘要 Token 暂未进入总 Token。这不影响
流程正确性测试，正式质量—成本比较前必须补齐。

真实一题 smoke：

```bash
./scripts/docker.sh run python -m benchmarks.runner.run_subprocess \
  --benchmark browsecomp_zh \
  --strategy code_orchestration \
  --orchestration-script reference_v5 \
  --orchestration-profile deepseek_v4_flash \
  --max-tokens 72000 --max-tool-calls 16 \
  --limit 1 --no-shuffle --concurrency 1 \
  --scoring exact_match \
  --out results/reference_v5_smoke_1
```
