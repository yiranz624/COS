# AgentHarness 代码、运行逻辑与实验协作指南

> 本文面向第一次接手 AgentHarness、多智能体编排和 Benchmark 实验的开发者。它同时说明 GitHub 原始仓库已有内容、当前工作区新增内容、单 Agent 与多 Agent 的真实调用链、工具与消息协议、Trace/Patch 机制、环境配置、实验命令和结果解读。

> **2026-08 架构更新：** 当前静态参考流程已经从
> `DynamicMultiAgentStrategy` 中拆出，成为顶层 `orchestrations/reference_v1.py`
> 可信 Python 脚本；在线 Patch 暂时冻结。新的 DAG、Solver–Verifier 循环、
> Checkpoint、角色模型路由与运行方法请以
> [CODE_AS_ORCHESTRATION_REFERENCE_V1_ZH.md](CODE_AS_ORCHESTRATION_REFERENCE_V1_ZH.md)
> 为准。本文后面关于旧 fixed/dynamic 策略的内容仅用于理解历史实现与对照实验。

> **2026-08 自主编排更新：** 新增 `generated_code_orchestration`。系统按题把原始问题、
> 环境能力清单交给独立 Planner，先生成自然语言总计划，再把该计划与优秀模板交给代码生成阶段，
> 由其生成受限的 `async def solve(ctx)`；代码经过
> AST 校验和离线预执行后才会交给 Runtime；计划、代码或执行失败时直接报错，不运行参考脚本。

## AI 自主生成 `solve` 的执行方式

该策略不会让模型修改整个工程。第一阶段 Planner 的输入固定为原始问题、角色与工具清单、预算与
允许的 `ctx` API，输出只能是自然语言总计划，不能包含 Python、伪代码或候选答案。第二阶段把该
计划、环境能力和 `ctx` API 文档与用法示例交给代码 Orchestrator，不提供完整的 `solve` 模板，输出只能是一份
`async def solve(ctx)` 源码。所有代码修复轮次继续引用同一份自然语言计划。

运行时 final verifier 的拒绝结论具有流程控制权：`accepted=false` 时，Orchestrator 必须把原始
researcher 结果、critique 与 failed criteria 交回 solver 重新生成候选；只有 `accepted=true` 的
候选才能进入 synthesizer，修正后仍没有通过验证则返回信息不足。Researcher 契约同时区分
`incomplete`（信息尚未查全）与 `incorrect`（候选或关键声明已被证伪），并通过
`incorrect_claims` 明确记录不得继续作为依据的错误信息。

生成链依次保存请求、生成并持久化自然语言总计划、依据计划生成代码、执行 AST 白名单校验、使用 Fake Context 预执行、计算源码
哈希、进行当前进程内注册，最后由 `ScriptRuntime` 执行。代码无效时最多修复三次；Planner 调用
失败、修复耗尽或生成代码运行异常时直接将该题标记为失败。Planner 和修复尝试的工件仍会保留，
不会用任何参考脚本的结果替代自主生成结果。

受限代码支持条件、异常处理、普通 `for` 循环和 `ctx.call/spawn/gather/parallel/map/cancel/
checkpoint/load_checkpoint/remaining_budget/emit`。禁止 import、文件与进程访问、反射、动态代码
执行和无限循环。模型及工具只能通过 Runtime 管理的角色 Job 使用。

```bash
./scripts/docker.sh run python -m benchmarks.runner.run_subprocess \
  --benchmark browsecomp \
  --strategy generated_code_orchestration \
  --planner-model deepseek/deepseek-v4-flash \
  --planner-max-repair-attempts 3 \
  --orchestration-profile deepseek_v4_flash \
  --limit 1 --no-shuffle --scoring exact_match
```

### 自主生成代码的当前校验边界

当前阶段只用语法、沙箱、角色、`ctx` API 和结果契约兼容性问题阻止脚本执行。
下列健壮性诊断会保存在 `validation_report.json` 的 `warnings` 字段中，但不会消耗
Planner 修复次数，也不会阻止编译、语义审查或 Runtime 执行：

- `loop_bound`：`for` 循环不是静态 `<= 8` 次；
- `missing_loop_budget_check`：多轮 Agent 循环没有调用 `ctx.remaining_budget()`；
- `unprotected_critical_call`：关键角色调用没有 `try/except`；
- `empty_failure_degradation`：异常分支没有保存或返回降级结果。

`while`、`async for`、comprehension、非法 import、动态代码、文件/进程访问、未知角色、
错误的同步/异步 `ctx` 调用以及不兼容的 API 参数仍属于阻塞错误。代码 reviewer 同样只审查
正常路径的可运行性、契约兼容性和对已批准计划的忠实程度，不因健壮性或边缘情况拒绝脚本。

`wall_time_seconds` 的框架默认值是 `None`（不限时）；只有命令行显式传入
`--wall-time-seconds` 才会设置整题超时。第 23 题辅助脚本不再硬编码 3600 秒。

每题的 `generated/plan/` 目录保存自然语言 Planner 的 system prompt、完整输入、原始输出、
`plan.md` 和 usage；`generated/` 目录还保存 `generation_request.json`、最终通过校验的 `solve.py`、
`validation_report.json` 和 `manifest.json`。此外，`attempts/attempt_NN/` 会永久保留每次生成及
修复的 Planner Prompt、原始响应、提取源码、usage、校验错误或调用异常；即使所有尝试都失败，
这些源码和诊断也不会被最终一次覆盖。`research_report.json` 将运行正确性、严格匹配、回退、预算、
失败类别及脚本结构特征关联起来，并把无回退、无违规且回答正确的脚本标记为 `promising`，仅供离线
演化候选使用，不自动部署。一次 benchmark 完成后，根目录的 `generated_scripts.json` 汇总全部成功
和失败脚本，方便跨题筛选与后续研究。`trace.jsonl` 包含生成、修复、注册、失败与回退事件；
`result.json` 的 `generation` 字段记录 Planner usage、文件路径和回退原因。生成脚本不会自动进入
正式白名单或推广给其他问题。

## 1. 项目目标与当前阶段

当前项目希望形成一个基于 **Code-as-Orchestration** 的多智能体动态编排与演化框架：

- 用可信的可执行 Python async 代码或有向图表示 Agent 分工、通信、串并行、循环、验证、异常处理和资源预算；
- 任务执行过程中，在明确的阶段边界根据实时 Trace 调整尚未执行的编排；
- 任务结束后，根据完整 Trace 诊断失败、生成下一版候选编排并检查其可复用性；
- 近期最小闭环是“静态编排 Runtime → 结构化 Trace → 受约束动态 Patch → 质量与成本联合评估”；
- PoC 使用 Decomposer、Solver、Verifier、Synthesizer，并比较单 Agent、固定多 Agent和动态多 Agent。

当前暂不接入 MAS-Bench，也不执行模型生成的任意 Python 编排代码；运行的编排函数是仓库内可信代码。

## 2. Git 基线与当前改动边界

### 2.1 原始 GitHub 基线

仓库地址：

```text
https://github.com/ycdeng-ntu/AgentHarness.git
```

当前基线：

```text
HEAD        e50cb18
origin/main e50cb18
```

主要历史：

```text
e50cb18 Merge pull request #1
c778e68 feat: add Docker-first BC200 evaluation
0797f51 feat: add Python orchestration runtime
0e1669e Initial open-source release of AgentHarness
```

当前修改尚未提交。查看原始文件可使用：

```bash
git show HEAD:workflows/react_base/nodes/main_agent.py
git diff HEAD -- workflows/react_base/nodes/main_agent.py
git status --short
```

### 2.2 原始仓库已有能力

原始仓库已经包含：

1. `react_base` 单 Agent ReAct workflow；
2. `web_search`、`web_fetch`、`run_python_code` 三个工具；
3. Agent Loop、Tool Call Parser、上下文压缩、预算 Observer 和 Trajectory；
4. BrowseComp-200 等 Benchmark runner 和 LLM Judge；
5. 每题独立子进程的运行方式；
6. 实验性 Python Code-as-Orchestration Runtime；
7. `call/spawn/gather/parallel/map`、取消、超时和并发/Job 数限制；
8. `decomposer/researcher/solver/verifier/synthesizer/general` 内置角色；
9. Fake Worker 和真实 AgentHarness Worker。

### 2.3 当前工作区新增能力

当前新增的核心文件包括：

```text
agent_harness/orchestration/contracts.py
agent_harness/orchestration/trace.py
agent_harness/orchestration/patching.py
agent_harness/orchestration/strategies.py
agent_harness/orchestration/evaluation.py
agent_harness/orchestration/evolution.py
```

主要增加：

- 统一三策略接口；
- `BudgetSpec`、usage 和联合评估；
- 版本化 Trace；
- 受约束 Patch 和验证；
- 动态策略的确定性阶段边界规则；
- 离线 Trace 诊断；
- Benchmark runner 的 `--strategy` 和 `--scoring`；
- `closed_book`、`smoke` 和 MiroMind/OpenAI-compatible 模型描述文件。

### 2.4 单 Agent 是否被修改

单 Agent 的核心 Runtime 没有被修改。以下目录仍与 `HEAD` 一致：

```text
agent_harness/core/
agent_harness/components/
agent_harness/infra/
agent_harness/scheduling/
agent_harness/state/
agent_harness/models/
```

但外围 workflow 有条件性改动：

```text
workflows/react_base/nodes/main_agent.py
workflows/react_base/prompts.py
```

新增逻辑允许 profile 设置：

```yaml
agent:
  allowed_tools: []
```

只有 `closed_book` 使用该配置并切换到闭卷提示词；原始 `default` 和 `keep5` 没有这个字段，联网工具行为不变。

需要特别注意：

- `closed_book`：禁用工具、只运行一轮、输出上限 2048，明显降低能力；
- `smoke`：最大三轮、输出上限 2048，只适合连通性检查，不适合正式准确率；
- `openrouter_deepseek_v4_flash.yaml`：通过 `openai` provider 调用 MiroMind，与原始 DeepSeek provider 的 thinking/tool-call 格式可能不同；
- `--scoring exact_match` 只改变评分，不改变模型生成；
- `SEMANTIC_JUDGE_MODEL`（兼容旧的 `BROWSECOMP_JUDGE_MODEL`）只改变
  Judge 模型，不改变解题 Agent。

## 3. 仓库目录结构

### 3.1 `agent_harness/core/`

通用 Agent 内核：

```text
core/runtime/loop/agent_loop.py       ReAct 主循环
core/runtime/loop/llm_client.py       调用 LLM、重试和流式响应
core/runtime/loop/tool_call_parser.py 工具调用格式解析
core/runtime/loop/tool_exec.py        工具并行执行、超时和错误包装
core/runtime/loop/compact.py          上下文压缩
core/runtime/loop/context_budget.py   上下文 Token 估算
core/runtime/resources/manager.py     按角色分配 LLM 和工具
core/runtime/registries/              Agent/Workflow/Service 注册表
core/runtime/dag/                     通用 DAG 结构
core/tool.py                          Tool 通用定义和 @tool 装饰器
core/messages.py                      system/user/assistant/tool 消息辅助函数
```

### 3.2 `agent_harness/components/`

Observer 和 Middleware：

```text
budget_observer.py          Token 预算
context_size_guard.py       输入上下文保护
leaked_tool_call_retry.py   工具调用文本泄漏检测
react_step_tracker.py       ReAct 步骤统计
trajectory.py               轨迹记录
summarization.py            外层上下文总结中间件
```

Observer 可以监听 LLM 响应、工具调用、工具结果和轮次结束，并返回 Intervention，例如重写参数、回滚本轮、注入消息或停止循环。

### 3.3 `agent_harness/infra/`

模型服务适配：

```text
openai_client.py
anthropic_client.py
llm_adapter.py
config.py
retriable.py
```

### 3.4 `agent_harness/scheduling/`

Workflow 调度：

```text
scheduler.py
pipeline_registry.py
workflow_loader.py
process_manager.py
```

### 3.5 `agent_harness/orchestration/`

多 Agent Code-as-Orchestration：

```text
runtime.py         ScriptRuntime、Job 生命周期、并发和取消
context.py         ctx.call/spawn/gather/parallel/map
agent_executor.py  将一个角色 Job 交给 Agent Loop
roles.py           内置角色定义
events.py          生命周期事件
store.py           运行期 Job/Event 内存存储
testing.py         FakeWorkerExecutor
strategies.py      三种策略
trace.py           TraceEventV1/RunTraceV1
patching.py        Plan、Patch 和 Validator
contracts.py       Budget、usage、策略和评估契约
evaluation.py      结果聚合
evolution.py       离线失败诊断
```

### 3.6 `workflows/react_base/`

原始单 Agent workflow：

```text
__init__.py            react_solver 角色和工具权限
spec.py                单节点 PipelineSpec
nodes/main_agent.py    单 Agent 入口和 force-final
profile.py             profile/model/provider 合并
providers.yaml         Provider 连接参数和 thinking 格式
profiles/              模型描述与行为 profile
prompts.py             联网和闭卷 system prompt、总结 prompt
observers/             react_base 专用回滚和轨迹逻辑
```

### 3.7 `benchmarks/`

```text
core/registry.py                数据集注册与读取
core/harbor_task_generator.py   生成 instruction.md/task.toml
core/kernel_adapter.py          启动 AgentHarness Runtime 和工具
harbor_agent/agent.py           Harbor 到 AgentHarness 的适配
runner/run_subprocess.py        总控，每题启动独立子进程
runner/run_one_task.py          执行和评分一道题
judges/                         Benchmark 专用 LLM Judge
datasets/                       数据集挂载目录
```

## 4. 核心名词

### 4.1 LLM

模型服务，例如 `deepseek/deepseek-v4-flash`。输入 system/user/history/tools，输出普通文本、thinking、结构化 tool calls 和 usage。

### 4.2 Tool

一个带名称、说明、JSON 参数 Schema 和异步 Python 函数的对象：

```python
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: ToolFn
```

模型看到的是 Tool JSON Schema，不会读取 Python 源码。

### 4.3 Agent

Agent 是：

```text
角色定义 + System Prompt + LLM + 工具权限 + Agent Loop + 预算/超时 + 输出格式
```

### 4.4 Job/节点

Agent 是可复用角色；Job 是某个角色针对一条 instruction 的一次具体执行。一次 `ctx.call()` 或 `ctx.spawn()` 会创建一个 Job，可视为编排运行图中的节点。

### 4.5 Pipeline

对节点和转换关系的静态定义。原始 `react_base` 只有一个 `main_agent` 节点。

### 4.6 Agent Loop

一个自动化 async Python 状态机，不是模型也不是提示词。它循环执行：

```text
调用 LLM → 解析 Tool Call → 执行工具 → 回填 Tool Result → 再调用 LLM
```

### 4.7 OrchestrationContext (`ctx`)

Python 编排函数与 ScriptRuntime 交互的操作句柄。它不是聊天上下文，也不是 Agent：

```python
ctx.task
ctx.metadata
ctx.run_id
ctx.call(...)
ctx.spawn(...)
ctx.gather(...)
ctx.parallel(...)
ctx.map(...)
ctx.inspect(...)
ctx.cancel(...)
```

## 5. Tool 机制：以 `web_search` 为例

### 5.1 Python 函数

位置：`plugins/tools/web_search.py`

```python
@tool(name="web_search")
async def web_search(
    q: str | list[str],
    gl: str = "us",
    hl: str = "en",
    location: str | None = None,
    num: int | None = None,
    tbs: str | None = None,
    page: int | None = None,
    autocorrect: bool | None = None,
) -> str:
    ...
```

参数：

| 参数 | 含义 |
|---|---|
| `q` | 必填，单个 query 或 query 列表 |
| `gl` | 地区代码，默认 `us` |
| `hl` | 语言代码，默认 `en` |
| `location` | 具体地理位置 |
| `num` | 返回结果数 |
| `tbs` | 时间过滤，如 `qdr:y` |
| `page` | 结果页码 |
| `autocorrect` | 是否拼写修正 |

### 5.2 注册与 Schema

`@tool` 根据函数注解和 docstring 生成 OpenAI-compatible Schema。模型逻辑上看到：

```json
{
  "type": "function",
  "function": {
    "name": "web_search",
    "description": "Perform web searches and retrieve rich results.",
    "parameters": {
      "type": "object",
      "properties": {
        "q": {"oneOf": [{"type": "string"}, {"type": "array"}]},
        "gl": {"type": "string", "default": "us"},
        "hl": {"type": "string", "default": "en"}
      },
      "required": ["q"]
    }
  }
}
```

### 5.3 模型调用

模型正常返回结构化 Function Call，而不是 shell 命令：

```json
{
  "tool_calls": [{
    "id": "call_001",
    "function": {
      "name": "web_search",
      "arguments": "{\"q\":\"Ireland Romania Brazilian referee\"}"
    }
  }]
}
```

ToolCallParser 将其转为：

```python
{"name": "web_search", "args": {"q": "..."}, "id": "call_001"}
```

Tool Executor 执行：

```python
await tool.ainvoke(args)
```

`web_search` 使用 `SERPER_API_KEY` 请求 Serper，最后返回纯文本：

```text
[1] Title: ...
    Date: ...
    Snippet: ...
    URL: ...
```

### 5.4 文本格式兼容

Parser 优先读取原生 OpenAI `tool_calls`，并兼容 JSON 文本、MCP XML、Qwen XML、Seed FunctionCall 和 MiroMind/DeepSeek 的 `<｜DSML｜tool_calls>`。DSML 会被解析为标准 `{name,args,id}` 调用，并同步转换为 OpenAI 风格的 assistant `tool_calls` 历史，使下一条 `tool` message 有合法的配对调用。模型使用 `open_url` 时会映射为仓库已有的 `web_fetch`；其他未知工具会由执行器返回安全错误，不再把原始 DSML 标记当作最终答案。

### 5.5 当前工具

| Tool | 文件 | 外部依赖 | 作用 |
|---|---|---|---|
| `web_search` | `plugins/tools/web_search.py` | `SERPER_API_KEY` | 搜索标题、摘要和 URL |
| `web_fetch` | `plugins/tools/web_fetch.py` | `JINA_API_KEY` + summary LLM | 抓网页并按问题提取信息 |
| `run_python_code` | `plugins/tools/run_python_code.py` | `E2B_API_KEY` | 在 E2B 沙箱运行 Python |

`web_fetch` 流程：Jina → 直接 HTTP fallback → summary LLM 提取。它本身可能产生额外 LLM 成本。

## 6. `instruction.md` 与任务文件

Benchmark 数据行：

```json
{
  "task_id": "1",
  "task_question": "Between 1990 and 1994...",
  "ground_truth": "Ireland v Romania"
}
```

生成的 `instruction.md`：

```markdown
# Task

Between 1990 and 1994 (Inclusive), what teams played in a soccer match...
```

它只包含任务标题和原始问题，不包含 system prompt、工具、历史或标准答案。

标准答案写在 `task.toml`：

```toml
[verifier.env]
GROUND_TRUTH = "Ireland v Romania"
ANSWER_TYPE = "exactMatch"

[metadata]
question_id = "1"
pipeline_id = "react_base"
profile = "default"
```

Agent 读取 `instruction.md` 和 `[metadata]`；`GROUND_TRUTH` 只供评分，不发送给解题模型。

## 7. 单 Agent 完整调用链

```text
run_subprocess
  → 每题启动 run_one_task 子进程
  → AgentHarnessAgent.setup
  → BenchmarkSession._bootstrap
  → 注册 LLM、Tools、Agents、Pipelines、Scheduler
  → AgentHarnessAgent.run(instruction)
  → BenchmarkSession.run
  → Scheduler.execute(react_base)
  → main_agent_node
  → run_agent_loop
  → _force_final_answer（必要时）
  → state.final_answer
  → agent/final_answer.txt
  → exact_match 或 LLM Judge
  → trial/result.json
  → run/results.json + summary.txt
```

### 7.1 初始 messages

```python
[
  {"role": "system", "content": "In this environment you have access..."},
  {"role": "user", "content": "Between 1990 and 1994..."}
]
```

Tools 通过独立的 API `tools=` 参数传入，不一定嵌入消息文本。

### 7.2 一轮搜索后的 messages

```python
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "原始问题"},
  {
    "role": "assistant",
    "content": "I will identify candidate matches.",
    "tool_calls": [{"id": "call_1", "name": "web_search", "args": {"q": "..."}}]
  },
  {
    "role": "tool",
    "tool_call_id": "call_1",
    "content": "[1] Title: ...\n    Snippet: ...\n    URL: ..."
  }
]
```

下一轮整个 history 会再次发送给模型。

### 7.3 如何终止

`react_base` 的自然终止不是必须跑满轮数，也没有配置 `finish` Tool：

```text
有可解析 Tool Call → 执行工具并继续
无 Tool Call、输出普通文本 → stopped_by="no_tool"，视为最终答案
```

模型根据 system prompt、问题和证据自行判断是否继续搜索。Python Runtime 不理解答案语义，只检查是否还有 Tool Call。

硬停止包括：

```text
max_turns
max_attempts
budget_exhausted
context_limit_reached
llm_error
paused
observer stop
```

若 `max_turns/context_limit_reached`，或 `no_tool` 但可见答案为空，`_force_final_answer()` 会在现有消息后追加工具禁用的总结 prompt，再调用同一模型提取最终答案。该调用不是 Judge，也必须计入成本。

### 7.4 Observer

`react_base` 还使用：

- Tool 参数规范化；
- 泄漏工具调用检测；
- 重复查询回滚；
- 拒答回滚；
- 空搜索结果回滚；
- Console 输出；
- Trajectory 文件；
- ReAct step 统计；
- Token 和上下文保护。

## 8. 多 Agent Runtime

### 8.1 两层控制

```text
编排层：Python 决定调用哪个角色、串并行、重试和 Patch
Agent 层：每个 Job 内部由 LLM 决定是否调用 Tool
```

当前不是 Supervisor LLM 自由路由子 Agent。更准确的说法是：

```text
AI 决定内容；Python 决定控制流。
```

### 8.2 `ctx.call/spawn/gather`

`ctx` 是 `OrchestrationContext`。

```python
solution = await ctx.call("solver", "Solve:\n...")
```

含义：创建 Solver Job、等待完成并返回 value。

```python
handle = ctx.spawn("solver", "Solve:\n...")
```

只创建 Job 并立即返回 `JobHandle`。

```python
results = await ctx.gather(*handles)
```

并发等待多个 Job，并按输入顺序返回。

`ctx.map(function, items)` 并发地将一个 async 编排函数应用于多个 item。

### 8.3 Agent 间通信

当前没有直接 Agent-to-Agent 聊天，也没有共享实时完整上下文。通信过程是：

```text
Agent A 输出
  → 返回 Python 编排器
  → Python 使用固定模板拼成 Agent B 的 instruction
  → Agent B 获得新的 system + user messages
```

例如：

```python
solution = await ctx.call("solver", f"Solve:\n{problem}")
verdict = await ctx.call(
    "verifier",
    f"Problem:\n{problem}\n\nCandidate:\n{solution}",
)
```

Verifier 不会自动获得 Solver 的完整 message history、thinking 或工具轨迹，除非 Python 显式传递。

### 8.4 内置角色

| 角色 | 工具 | 输出 |
|---|---|---|
| Decomposer | 无 | 结构化独立解题分支数组 |
| Researcher | search/fetch/python | `{status, answer, candidate_table, evidence, missing_constraints}` |
| Solver | 仅 Python | 候选解答文本 |
| Verifier | search/fetch/python | `{accepted, critique}` |
| Synthesizer | 无 | 最终文本 |
| General | search/fetch/python | 通用文本 |

Decomposer 的每个分支包含 `objective/requires_web/approach`。需要联网的
分支自动路由给 Researcher；不需要外部事实的分支才交给 Solver。Solver
仍然只有 Python 权限，避免所有分支无条件搜索。

## 9. 三种策略

### 9.1 `single_agent`

Benchmark 中保留原始 `react_base`：

```text
问题 → react_solver(LLM ↔ Tools) → 最终答案
```

`strategies.py` 内另有一个 orchestration-side `SingleAgentStrategy/general`，但 Benchmark 的 `--strategy single_agent` 走 `AgentHarnessAgent → react_base`，不要混淆。

### 9.2 `fixed_multi_agent`

```text
Decomposer
  ↓ 独立完整解题路线列表，并标记requires_web
多个分支并行：(Researcher或Solver) → Verifier
  ↓
Synthesizer
```

Python 固定执行：

```python
branches = await ctx.call("decomposer", ...)

async def solve_one(branch):
    role = "researcher" if branch["requires_web"] else "solver"
    solution = await ctx.call(role, full_task_and_branch_instruction)
    verdict = await ctx.call(
        "verifier",
        full_task_branch_and_candidate,
    )
    return {"branch": branch, "solution": solution, "verdict": verdict}

solved = await ctx.map(solve_one, branches)
answer = await ctx.call("synthesizer", f"Original task: ... Results: {solved}")
```

实际送入 Synthesizer 前还有一道确定性的 admission control：只有
`verdict.accepted=true` 的完整候选及证据进入事实答案池；被拒绝、失败、
未完成或未验证分支只传递状态、缺失约束和 critique，不传递其候选正文。
若没有被接受的候选，Synthesizer 必须输出 `信息未查全`，不能把“未找到”
误写成“确定不存在”。

这里的“分支”不是前后依赖的过滤步骤。每个分支都必须：

- 保留完整原题和所有联合约束；
- 能独立产生一个最终候选答案；
- 采用不同检索或推理路线；
- 明确标记是否需要当前外部事实；
- 需要联网时在 instruction 中明确要求使用 `web_search/web_fetch`。

旧版 `list[str]` 输出仍被 Runtime 兼容，但新版 Decomposer Schema 会优先
要求结构化对象。如果输出无效或为空，回退分支默认使用 Researcher 围绕
完整原题搜索，避免再次退化成无工具 Solver。

Decomposer 返回空可见输出或语义为空的 `[]` 时，Worker 直接生成一条本地
fallback 分支，不再调用 LLM 做无收益的格式修复；只有存在非空但不符合
Schema 的内容时才尝试一次 `_output_repair`。

联网分支会被 Python 规范化为当前工具确实能执行的三类路线：稀有线索或
精确短语检索、裁判/实体枚举后逐场核验、公开档案与比赛报告交叉验证。
不会把未配置的私有 API、不存在的 Kaggle 文件或本地数据库当成可执行路线。

固定策略记录 Trace，但不根据运行结果修改计划。

### 9.3 `dynamic_multi_agent`

使用相同初始阶段，但在边界处执行确定性规则：

1. Decomposition 完成：按目标和路线去重，最多保留 3 条独立路线；
2. 第一轮 Worker 完成：失败或无候选分支转成结构化 `incomplete`；
3. 仅把 `status=candidate` 的分支送给 Verifier；
4. Verifier 完成：预算允许时最多选择 1 条被拒绝候选做针对性 Revision；
5. Revision 只有重新验证为 `accepted=true` 后才进入最终答案池；
6. Synthesizer 前按 Verifier 结论机械划分 accepted 与 rejected/incomplete。

动态控制器当前是 Python 规则，不是额外 LLM，因此初期收益不会与额外控制器推理成本混淆。

## 10. Patch：受约束修改未执行计划

### 10.1 Plan

```python
PlanNode(
    node_id="solve:0",
    role="solver",
    instruction="Solve subproblem 0",
    dependencies=(),
    max_turns=4,
    state=NodeState.PENDING,
)
```

初始计划示例：

```text
solve:0 ─┐
solve:1 ─┼──→ synthesize
solve:2 ─┘
```

### 10.2 Patch

Patch 是针对某个计划版本的一组修改：

```python
OrchestrationPatchV1(
    base_version=1,
    trigger_stage="decomposition_completed",
    evidence=("duplicate_or_excess_subproblems",),
    operations=(
        PatchOperation(op="delete", node_id="solve:4"),
        PatchOperation(
            op="update",
            node_id="synthesize",
            changes={"dependencies": ("solve:0", "solve:1", "solve:2")},
        ),
    ),
    reason="deduplicate and cap decomposition fan-out",
)
```

支持：

```text
add/delete/replace/retry/update
set_parallelism/set_termination
```

### 10.3 校验

Patch 必须满足：

- `base_version` 等于当前计划版本；
- 只能修改 `PENDING` 节点；
- 不得引用未知角色；
- 依赖必须存在；
- 不得形成环；
- Job 数和并发不得突破硬预算；
- 终止条件不能为空；
- 所有操作原子应用，任一非法则全部拒绝。

接受后计划版本递增。

### 10.4 当前 PoC 边界

Patch Schema 和 Validator 已真实实现，但当前还不是“通用 DAG Executor 暂停后自动按新 DAG 继续”。动态策略仍由 Python：

1. 生成并验证 Patch；
2. 记录 before/after version 和证据；
3. 根据接受结果手动执行 retry/revision `ctx.call()`；
4. 手动更新 `problems/solutions`。

下一阶段应让 OrchestrationPlan 成为执行器的权威计划。

## 11. Trace、预算和离线演化

多 Agent 每题输出：

```text
agent/final_answer.txt
agent/trajectories/*.jsonl          单 Agent逐步轨迹
agent_traces/<job>_<role>.jsonl     多 Agent中每个Job的逐步轨迹
orchestration_live.jsonl            多 Agent顶层生命周期实时轨迹
stage_trace.jsonl                   阶段边界和Patch实时轨迹
trace.jsonl
evolution.json
result.json
```

Trace 记录 Run、Stage、Job、角色、分支、LLM/Tool 调用、Patch、失败、时间戳、父子关系、usage、预算快照和 orchestration version。

所有实时 JSONL 均采用“走一步、追加一行、立即 flush”的方式，即使进程中断，已完成事件仍然保留。每个 Agent 的轨迹包括：

- Agent 开始时的模型、system prompt、user instruction、最大轮数和可用 Tool Schema；
- 每一次物理模型 API 尝试的开始、成功或失败、尝试序号、HTTP 状态、错误和耗时；
- 每个逻辑 LLM 轮次实际接收的完整 messages 和 tools；
- 模型返回文本、thinking、Tool Call 和 Provider usage；
- 每个真实 Tool Call 的名称、参数、调用 ID、返回信息、错误状态和耗时；
- 最终轮次、停止原因、最终输出和累计工具调用数；
- force-final 的完整输入 messages、总结 prompt、输出、Token 和耗时。

如果某一轮没有 Tool Call，不写空的“工具调用开始/完成”事件。为了兼容程序消费，稳定协议字段（如 `t/turn/usage/tool_call_id`）继续保留；面向人阅读的事件名称、说明和详细字段使用中文。

预算契约包含：

```text
max_tokens
max_llm_calls
max_tool_calls
max_jobs
max_parallel_jobs
wall_time_seconds
```

当前 Job 数、并发和 wall time 在执行前/执行中约束；Token/调用上限主要依赖 Provider usage，在调用后聚合检查，尚不是所有路径上的严格预扣费硬限制。

当前 BrowseComp 多 Agent smoke 的题目级默认 Token 上限设为 `72000`：
旧编排首个样本使用 `58707` Token，以其约 `1.2` 倍得到 `70448`，
取整后保留少量综合余量。该值只是当前 DeepSeek/BrowseComp PoC 的初始
标定值，不是所有题目或模型的通用常数。样本增多后应改用成功样本 Token
分布的中位数或 P75，再乘 `1.1–1.2`，不能用失控失败样本的均值。

预算采用两层控制：

- 单次 completion：Researcher/Solver 最多 4096，Verifier/Synthesizer
  最多 2048，防止一次隐藏 reasoning 消耗 16384 Token 却返回空文本；
- 题目阶段配额：Discovery 目标上限为 60%，超过时写入阶段 Trace；总消耗
  达到 78% 前仍允许 candidate 进入 Verifier；达到 88% 后不再新增
  Revision/Re-verification；始终为 Synthesizer 预留最后 12%。未重新
  验证的 Revision 不会成为最终事实依据。动态 Researcher 最多 3 轮，
  最后一轮由 Runtime 强制禁用工具，并在调用模型前追加“现在必须根据已有
  证据返回最终结构化结果”的 user message。若模型仍请求工具，该请求不会
  执行，但工具名和参数会写入 Trace 的 `最终轮被拒绝的工具调用` 字段，
  同时记录拒绝原因；因此既不会突破预算，也不会丢失模型原始行为证据。

题目级 Token 是 Provider 调用完成后才能准确获得，因此并行运行中的多个
请求仍可能造成少量超限；当前属于“单次硬上限 + 阶段边界软上限”，并非
并行请求的逐 Token 实时中断。最终 `budget_violations` 仍记录实际超限。

Researcher 使用候选表：每行一个候选、每列对应原题的一项联合约束；连续
两次搜索没有新增候选或证据时停止，并输出“信息未查全”和缺失约束，而不
重复检索或猜测。其返回值现在是结构化状态：`candidate` 表示存在可验证
候选，`incomplete` 表示证据不足。空的可见输出也会确定性降级为
`incomplete`；候选表中的“No match found/未找到”等状态文本会被清除，
因为候选表只能包含真实候选实体。Revision Job 失败时保留原分支记录、记录
`revision_degraded`，Run 继续进入 Synthesizer。

强制收尾和结构化格式修复产生的额外 LLM 调用会复用同一个 usage collector，
并分别写入 `_forced_final` 或 `_output_repair` 实时 Trace，不再漏计成本。

`evolution.json` 对完整 Trace 做失败分类和瓶颈摘要，能够识别
`incomplete_branch`、`verification_skipped_budget` 和
`no_accepted_candidate`，保存候选建议，但不会自动部署新版本。

## 12. Benchmark、评分与输出

### 12.1 每题独立子进程

`run_subprocess.py` 为每道题启动 `run_one_task.py`，隔离：

- Python/asyncio 状态；
- Runtime registry；
- 内存 DB；
- Agent history；
- 单题超时和崩溃。

### 12.2 三种评分

```bash
--scoring llm_judge
```

使用 Benchmark 官方/项目配置的 LLM Judge，能处理别名、解释性答案和小数误差，但产生额外调用与不稳定性。

对于 BrowseComp / BrowseComp-200，此模式使用 BrowseComp-Plus 的语义等价
协议，允许标准答案的字符串变体，以及内容正确但更精确或更详细的回答。

```bash
--scoring semantic_judge
```

对任何 benchmark 强制使用 BrowseComp-Plus 语义等价 Judge。其结构化输出
（抽取答案、reasoning、correct、confidence、parse_error、模型和原始响应）
写入 `result.json` 的 `judge_details`。Judge 模型由
`SEMANTIC_JUDGE_MODEL` 指定，接口和密钥沿用 `JUDGE_BASE_URL/JUDGE_API_KEY`。

```bash
--scoring exact_match
```

本地保守归一化匹配，不调用 Judge。它忽略大小写、重复空格、简单 `FINAL ANSWER:` 包装和末尾标点，但不处理语义等价别名。

Judge 不参与解题，只在 Agent 结束后比较 `predicted_answer` 与 `ground_truth`，结果不会反馈给 Agent。

### 12.3 输出目录

容器中的：

```text
/app/results/<run-name>
```

映射到主机：

```text
results/<user>/<run-name>
```

典型结构：

```text
config.json
tasks/<qid>/instruction.md
tasks/<qid>/task.toml
trials/<qid>/agent.log
trials/<qid>/agent/final_answer.txt
trials/<qid>/agent/trajectories/*.jsonl
trials/<qid>/result.json
results.json
summary.txt
```

只有整批运行结束才会生成顶层 `results.json` 和 `summary.txt`。中断后使用相同 `--out` 重跑，会跳过已有有效 `result.json` 的题目。

## 13. 环境与密钥

项目采用 Docker-first，不需要在宿主机另外创建 Python venv。容器依赖位于 `/app/.venv`。

`.env` 示例（不要提交真实密钥）：

```env
# Main model via MiroMind/OpenAI-compatible API
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.miromind.site/v1
REACT_BASE_MODEL_CONFIG=openrouter_deepseek_v4_flash.yaml

# Search
SERPER_API_KEY=...

# Web fetch
JINA_API_KEY=...

# Code sandbox, optional for search-only experiments
E2B_API_KEY=...

# Dedicated judge, optional when using exact_match
JUDGE_API_KEY=...
JUDGE_BASE_URL=https://api.miromind.site/v1
SEMANTIC_JUDGE_MODEL=...
```

切勿把 `.env`、API Key、`docker compose config` 的完整输出发送到聊天、Git 或实验报告。若 Key 已暴露，应立即撤销和轮换。

验证变量是否进入容器，只输出布尔值：

```bash
./scripts/docker.sh run python -c \
'import os; print({k: bool(os.getenv(k)) for k in ("OPENAI_API_KEY", "SERPER_API_KEY", "JINA_API_KEY", "E2B_API_KEY")})'
```

## 14. Docker 与常用命令

```bash
cd /path/to/cos
./scripts/docker.sh build
./scripts/docker.sh info
./scripts/docker.sh shell
./scripts/docker.sh unit
./scripts/docker.sh lint
```

HPC 上曾出现 BuildKit 无法解析 PyPI DNS。`docker-compose.yml` 的 build network 已设置为 host，以便下载 hatchling 等隔离构建依赖。

## 15. 实验命令

### 15.1 一条闭卷单 Agent（低成本弱基线）

```bash
./scripts/docker.sh run python \
  -m benchmarks.runner.run_subprocess \
  --benchmark browsecomp_200 \
  --strategy single_agent \
  --profile closed_book \
  --scoring exact_match \
  --limit 1 \
  --offset 0 \
  --no-shuffle \
  --concurrency 1 \
  --max-tool-calls 0 \
  --out ./results/closed-book-exact-one
```

`closed_book` 不是原始单 Agent 能力配置，只用于验证无检索弱基线。

### 15.2 一条联网 smoke

```bash
./scripts/docker.sh run python \
  -m benchmarks.runner.run_subprocess \
  --benchmark browsecomp_200 \
  --strategy single_agent \
  --profile smoke \
  --scoring exact_match \
  --limit 1 \
  --offset 0 \
  --no-shuffle \
  --concurrency 1 \
  --max-tool-calls 10 \
  --out ./results/online-single-smoke-one
```

日志应出现：

```text
tools=['web_search', 'web_fetch', 'run_python_code']
web_search {...}
web_fetch {...}
```

不应出现 Key 缺失、401、Judge 503 或 raw DSML 被当作最终答案。

`smoke` 只有三轮，不能用作正式 BrowseComp 能力结论。

### 15.3 接近原始行为的联网单 Agent

```bash
./scripts/docker.sh run python \
  -m benchmarks.runner.run_subprocess \
  --benchmark browsecomp_200 \
  --strategy single_agent \
  --profile default \
  --scoring llm_judge \
  --limit 5 \
  --offset 0 \
  --no-shuffle \
  --concurrency 1 \
  --out ./results/bc200-single-default-5
```

正式比较前需确认 Judge 模型可用、Provider DSML/tool-call 兼容，并设置等预算上限。原始 `default` 最大 600 轮，可能非常昂贵，建议先设计正式受控 profile，而不是直接批量运行。

### 15.4 多策略固定题集

对相同 `limit/offset/no-shuffle` 或相同 seed，分别运行：

```text
--strategy single_agent
--strategy fixed_multi_agent
--strategy dynamic_multi_agent
```

并固定：

- 数据题目 ID；
- 主模型和 Provider；
- Judge 和评分方式；
- Token/LLM/Tool/Job/并发/wall-time 上限；
- 提示词和 profile；
- 答案提取方式。

### 15.5 多 Agent：1 条 smoke 到前 50 条

当前多 Agent Benchmark 使用通用 Runtime 模型配置，而不是
`react_base` 单 Agent 的行为 profile。关键环境变量为：

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=<MiroMind/OpenRouter兼容网关提供的密钥>
OPENAI_BASE_URL=https://api.miromind.site/v1
OPENAI_MODEL=deepseek/deepseek-v4-flash
SERPER_API_KEY=<Serper密钥，联网检索必需>
JINA_API_KEY=<Jina Reader密钥，建议配置>
E2B_API_KEY=<仅调用run_python_code时需要>
```

`JUDGE_API_KEY/JUDGE_BASE_URL` 在 `--scoring llm_judge` 或
`--scoring semantic_judge` 时需要。
若 MiroMind 没有 BrowseComp 默认 Judge 模型渠道，可先用
`--scoring exact_match` 做本地严格匹配；它不调用 Judge，但
Synthesizer 输出“答案 + 解释”时可能被判错，因此不能与官方 LLM Judge
分数混为一谈。

先固定数据顺序，对第 1 条执行 dynamic multi-agent smoke：

```bash
./scripts/docker.sh run python \
  -m benchmarks.runner.run_subprocess \
  --benchmark browsecomp_200 \
  --strategy dynamic_multi_agent \
  --scoring exact_match \
  --limit 1 \
  --offset 0 \
  --no-shuffle \
  --concurrency 1 \
  --max-jobs 32 \
  --max-parallel-jobs 4 \
  --wall-time-seconds 1800 \
  --out ./results/bc200-dynamic-smoke-1
```

smoke 完成后检查：

```bash
./scripts/docker.sh run python -m json.tool \
  ./results/bc200-dynamic-smoke-1/results.json

find ./results/bc200-dynamic-smoke-1/trials -maxdepth 3 -type f | sort
```

运行过程中，另开终端实时观察（把 `1` 换成实际题目 ID）：

```bash
tail -f results/<user>/bc200-dynamic-smoke-1/trials/1/orchestration_live.jsonl
tail -f results/<user>/bc200-dynamic-smoke-1/trials/1/stage_trace.jsonl
tail -f results/<user>/bc200-dynamic-smoke-1/trials/1/agent_traces/*.jsonl
```

这里有宿主机与容器路径映射：命令中的 `/app/results/...`（即
`./results/...`）会落到宿主机
`/path/to/cos/results/<user>/...`。因此宿主机查看 Trace
时需要多一层用户目录 `<user>`。

确认 smoke 的模型调用、搜索、最终答案、评分、Trace 和预算汇总均正常后，
再运行顺序前 50 条：

```bash
./scripts/docker.sh run python \
  -m benchmarks.runner.run_subprocess \
  --benchmark browsecomp_200 \
  --strategy dynamic_multi_agent \
  --scoring exact_match \
  --limit 50 \
  --offset 0 \
  --no-shuffle \
  --concurrency 1 \
  --max-jobs 32 \
  --max-parallel-jobs 4 \
  --wall-time-seconds 1800 \
  --out ./results/bc200-dynamic-first50
```

首次正式运行建议保持 `--concurrency 1`，先观察网关限流和单题成本；
确认稳定后才提高题目级并发。`--max-parallel-jobs 4` 是一道题内部的
Agent Job 并发，不等同于同时运行多少道题。固定
`--no-shuffle --offset 0 --limit 50` 可确保后续三策略使用同一批题。

## 16. 已观察到的问题与实验结论

### 16.1 闭卷试验

BrowseComp-200 数据集顺序前 11 个已完成样本：

```text
模型：deepseek/deepseek-v4-flash
策略：single_agent
profile：closed_book
工具：禁用
评分：normalized exact match
结果：0/11
```

标准答案没有出现在这些预测中，因此 0/11 不只是 exact-match 太严格。合理结论是：该模型在当前 BrowseComp 闭卷设置下未表现出有效解题能力，支持外部检索的必要性；不能泛化成“DeepSeek 对所有闭卷任务都没有能力”。

### 16.2 输出截断与额外调用

部分请求将 2048 completion tokens 全用于 thinking，可见答案为空，随后触发 `_force_final_answer()`。因此“一个 Agent、一轮”不等于“一次 API 调用”，成本统计必须包含 force-final 和 web_fetch summary LLM。

### 16.3 Judge 503

主模型调用曾返回 200，但 Judge 默认请求 `gpt-4.1-2025-04-14`，MiroMind 分组无对应渠道，返回 `503 model_not_found`。这是评分模型不可用，不是 Agent 解题 API 失败。

### 16.4 工具密钥缺失

没有 `SERPER_API_KEY` 时，`web_search` 会返回错误文本；没有 `JINA_API_KEY` 时，`web_fetch` 回退直接 HTTP，Wikipedia/Transfermarkt 可能返回 403/405。留空 Key 并不等于禁用工具；真正禁用应使用 profile 权限过滤。

## 17. 具体题目端到端示例

题目：

```text
Between 1990 and 1994 ... what teams played ...?
```

标准答案：

```text
Ireland v Romania
```

### 17.1 单 Agent

```text
instruction.md
  ↓
react_solver messages = [system, user question]
  ↓
LLM tool_call: web_search(q="...")
  ↓
Serper result 作为 role=tool 加入 history
  ↓
LLM tool_call: web_fetch(url="...", info_to_extract="...")
  ↓
Jina/summary result 加入 history
  ↓
LLM 普通文本：Ireland v Romania
  ↓ 无 tool_call，自然终止
final_answer.txt
  ↓
评分与 result.json
```

### 17.2 固定多 Agent

```text
Decomposer：输出结构化独立解题路线，并标记是否需要联网
  ↓
并行分支：
  Researcher 0（检索路线A，携带完整原题）→ Verifier 0
  Researcher 1（检索路线B，携带完整原题）→ Verifier 1
  Solver 2（无需联网的推理路线，携带完整原题）→ Verifier 2
  ↓
Python 把 problem/solution/verdict 列表拼进 Synthesizer instruction
  ↓
Synthesizer 输出最终答案
```

### 17.3 动态多 Agent

```text
Decomposer
  ↓ 去重/限长 Patch
并行 Researcher/Solver（按requires_web路由）
  ↓ 失败/无候选分支标记 incomplete
仅对 candidate 并行 Verifier
  ↓ 最多一条拒绝分支 revision Patch
  ↓ Revision 重新验证
accepted-only admission control → Synthesizer
```

每个角色 Job 内部仍然使用相同的 Agent Loop；Agent 间内容通过 Python instruction 模板传递，不是直接聊天。

## 18. 下一步建议

1. 配置并单独验证 Serper 与 Jina；
2. 用一题联网 smoke 检查 Tool Call、抓取、最终答案和本地评分；
3. 用真实 smoke 持续确认 MiroMind DeepSeek 的 DSML、thinking 和原生 Tool Call 适配；
4. 建立正式联网 profile，轮数和输出预算应高于 smoke、低于无约束 default；
5. 记录固定 5–10 道 smoke 题 ID；
6. 先完成单 Agent 联网基线；
7. 用联网 smoke 验证结构化分支能产生不同检索路线并命中完整联合约束；
8. 用 Fake Worker 完成三策略 Trace/Patch/预算端到端测试；
9. 在等预算下比较三种策略；
10. 最小闭环稳定后再接 MAS-Bench 和外部 Benchmark。

## 19. 协作同步摘要模板

可向协作者说明：

> 当前代码基于 AgentHarness `e50cb18`。原仓库已包含 `react_base` 单 Agent、三个工具、Benchmark 子进程链路和实验性 Python orchestration Runtime。新增内容主要是统一三策略、结构化 Trace、受约束 Patch、预算/评估和离线演化。为了完整记录每一次模型调用及重试，当前版本已扩展核心 Agent Loop、模型客户端和 Trajectory Observer 的观测接口，但没有改变 Tool Parser、Tool Executor 或模型作答/终止决策逻辑。闭卷与 smoke profile 是低成本实验配置，会明显限制模型能力，不能代表原始 `default` 基线。当前多 Agent 控制器是可信 Python 代码而非 Supervisor LLM；每个角色内部仍运行 LLM-Tool Agent Loop，角色之间通过编排器构造的新 instruction 传递信息。
