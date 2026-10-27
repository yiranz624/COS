# COS 中文说明

COS（Code as Orchestration）将多智能体协作表示为可执行程序：规划并生成编排代码，审查后执行，维护共享事实，再根据执行轨迹和事实依赖诊断失败并改进下一轮编排。

本仓库整理自 Rolechange V6 代码。为保持兼容，Python 包名仍为 `agent_harness`，主要策略参数仍为 `rolechange_self_evolving_v6`。

## 快速开始

建议使用 Linux 或 WSL2、Bash、Docker 和 Docker Compose。原有 Compose 配置使用 host 网络，需要确认本机 Docker 支持。

```bash
git clone https://github.com/yiranz624/COS.git
cd COS
cp .env.example .env
./scripts/docker-rolechange-v6.sh build
```

在 `.env` 填写模型、评测、搜索和网页读取服务的配置与密钥。若更换模型，请同步修改 `orchestrations/profiles/deepseek_v4_flash.yaml` 的角色模型，或创建新 profile。示例中的模型名称来自原始代码，请以实际服务商支持的模型为准。

从数据集原发布方取得合法可用的数据，并按加载器格式准备，例如：

```text
benchmarks/datasets/BrowseComp/standardized_data.jsonl
```

随后运行一个任务：

```bash
./scripts/docker-rolechange-v6.sh run env ROLECHANGE_V6_MODE=slice \
  python -m benchmarks.runner.run_subprocess \
  --benchmark browsecomp --strategy rolechange_self_evolving_v6 \
  --orchestration-profile deepseek_v4_flash \
  --limit 1 --no-shuffle --concurrency 1 --out results/COS-smoke
```

在线运行会调用外部服务并可能产生费用。容器内 `results/` 映射到宿主机 `results/<用户名>/`。数据集、密钥和历史运行记录未打包。

## 三种诊断模式

- `off`：不记录 Fact Flow，使用原反思检索方式。
- `flow_only`：记录 Fact Flow，保留原反思检索方式。
- `slice`：记录 Fact Flow，优先检索有界诊断切片，按需扩大范围。

V6 三种模式的 Planner 均保留精确 Fact ID 提示，因此 `off` 不等同于逐字相同的 V5 配置。

完整配置、数据字段、消融运行和离线回放见 [复现说明](docs/REPRODUCIBILITY.md)，项目结构与命令见 [英文 README](README.md)。本次发布不声明已复现论文的全部指标。

## 开发与来源

```bash
./scripts/docker-rolechange-v6.sh unit
./scripts/docker-rolechange-v6.sh lint
```

保留原项目 Apache 2.0 许可证，以及 AgentHarness / Apodex 来源说明。发布整理排除了内部分析、完整运行轨迹、环境快照和未确认可再分发的数据；历史事实修复测试的正文和链接替换为合成样例，保留引用修复结构。详见 [NOTICE](NOTICE)。

## 贡献者

[ruirui20060208-dot](https://github.com/ruirui20060208-dot) · [yiranz624](https://github.com/yiranz624) · [An-Moon](https://github.com/An-Moon) · [bzda-bit](https://github.com/bzda-bit) · [yinziqiu666](https://github.com/yinziqiu666)
