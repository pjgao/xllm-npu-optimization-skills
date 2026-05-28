# xllm-npu-optimization-skills

Agent-ready playbooks for large-model serving and AI Infra optimization on
Huawei Ascend NPU (910B3/A3), with xLLM as the first fully recorded landing
case and vLLM-Ascend / SGLang as reusable comparison baselines and shared
workflow targets.

主要框架目标：[xLLM](https://github.com/jd-opensource/xllm)、[vLLM-Ascend](https://github.com/vllm-project/vllm-ascend)、SGLang NPU 后端

## 仓库定位

本仓库沉淀一套 **NPU AI Infra Auto-Driven Workflow**：面向昇腾 NPU 上的
大模型推理框架，用统一证据标准驱动 benchmark、profiling、capacity
planning、incident triage、code review、kernel pilot、PR history 查询和
SOTA loop。

当前已有经验主要来自 xLLM + Qwen3.5-27B + MTP 在 910B3/A3 上的真实优化。
这些经验会继续保留为 case study；后续新增能力应尽量拆成“通用 NPU
流程”和“框架适配层”，使同一套流程能服务：

| 框架 | 角色 | 当前状态 |
|------|------|---------|
| xLLM | 首个完整落地框架 | 已有 benchmark、profiling、patch、MTP 优化和事故记录 |
| vLLM-Ascend | 公平对照与共同优化目标 | 已作为 benchmark 对照组，后续补 profiler/capacity/incident 适配 |
| SGLang NPU | 扩展目标 | 复用本仓库的 NPU 证据标准和框架适配层，逐步补齐后端经验 |

因此，本仓库提供：
- 多框架公平基准测试（xLLM / vLLM-Ascend / SGLang NPU）
- 昇腾 Profiling 五表分析报告
- NPU pipeline / layer / rank / MFU / capacity 证据栈（规划中逐步补齐）
- RLCR（Research-Learn-Code-Review）驱动的 SOTA 自治优化循环
- NPU 特化代码审查
- 生产事故诊断与 replay-first 排障
- 精度异常定位与 commit 二分
- PR 驱动模型知识库
- NPU 内核证据辅助（TileLang / AscendC / Triton-Ascend）

通用流程详见：[通用 NPU 大模型推理与 AI Infra 开发工作流](docs/npu-ai-infra-standard-workflow.md)。
后续实现计划详见：[待实现能力与路线规划](docs/implementation-roadmap.md)。

## 核心 Skills

| Skill | 说明 | 关键产物 |
|-------|------|---------|
| [`xllm-npu-benchmark`](skills/xllm-npu-benchmark/SKILL.md) | NPU 多框架公平基准测试；当前覆盖 xLLM vs vLLM-Ascend，规划扩展 SGLang NPU | `candidates.jsonl` / `summary.md` / `winning-commands.md` |
| [`xllm-npu-profiler`](skills/xllm-npu-profiler/SKILL.md) | 昇腾 Profiling 五表分析；当前以 xLLM trace 为主，规划统一多框架产物 schema | kernel / overlap / fuse / dispatch / memory 五表 |
| [`xllm-npu-sota-loop`](skills/xllm-npu-sota-loop/SKILL.md) | NPU SOTA 自治优化循环；以 xLLM 为首个 target，流程可迁移到 vLLM-Ascend / SGLang | run manifest / refined plan / RLCR ledger |
| [`xllm-npu-code-review`](skills/xllm-npu-code-review/SKILL.md) | NPU 特化代码审查；覆盖 C++ engine、图模式、KV Cache、HCCL、TileLang/AscendC | 分级 review finding |
| [`xllm-npu-accuracy-debug`](skills/xllm-npu-accuracy-debug/SKILL.md) | 精度异常定位、A/B 验证和 commit 二分 | accuracy report / bisect notes |
| [`xllm-npu-incident-triage`](skills/xllm-npu-incident-triage/SKILL.md) | NPU 生产事故诊断与 replay-first 排障 | incident bundle / replay report |

## 辅助层

| 组件 | 说明 |
|------|------|
| [`model-pr-optimization-history`](model-pr-optimization-history/SKILL.md) | PR 驱动模型历史档案；当前以 xLLM 为主，规划扩展 vLLM-Ascend / SGLang |
| [`kernel-pilot`](kernel-pilot/SKILL.md) | NPU 内核证据辅助（TileLang/AscendC/Triton-Ascend） |

## 环境要求

- 华为昇腾 910B3 (A3) NPU
- HDK Driver >= 25.2.0
- CANN >= 8.0.RC1
- 至少一个目标推理框架：xLLM、vLLM-Ascend、SGLang NPU 后端
- 对照框架按任务选择；做 SOTA 对比时必须记录各框架 commit、容器镜像和完整启动命令

## 目录结构

```
xllm-npu-optimization-skills/
├── AGENTS.md                                # Agent 使用指南
├── INSTRUCTIONS.md                          # 优化指令模板
├── README.md                                # 本文档
│
├── docs/                                    # 设计文档
│   ├── ENVIRONMENT.md                       # 环境配置指南
│   ├── environment-info.md                  # 环境信息采集
│   ├── implementation-roadmap.md            # 仓库实现路线
│   ├── npu-ai-infra-standard-workflow.md    # 通用 NPU AI Infra 标准流程
│   ├── pr-1400-qwen3-next-weight-transform-race.md # PR #1400 精度定位
│   ├── pr-1536-mtp-transpose-elimination.md # PR #1536 分析
│   ├── pr-1541-mtp-draft-overlap-minimal-validation.md # PR #1541 最小验证
│   ├── qwen35-27b-optimization-guide.md     # Qwen3.5-27B 优化指南
│   └── xllm-npu-optimization-design.md      # NPU 优化设计方案
│
├── references/                              # 全局引用文件
│   └── custom-code-style.md                 # xLLM NPU 代码风格指南
│
├── humanize/                                # 优化过程台账
│   ├── attempt-ledger.md                    # 尝试记录
│   ├── lineage.jsonl                        # 优化族谱
│   ├── optimization-ledger.md               # 优化台账
│   └── source-idea-ledger.md                # 优化思路来源
│
├── patches/                                 # 补丁文件
│   ├── qwen3_gated_delta_net_base.cpp       # MTP conv1d 修复补丁
│   └── qwen3_gated_delta_net_base.h
│
├── skills/                                  # 核心 Skills
│   ├── xllm-npu-benchmark/                  # 基准测试
│   │   ├── SKILL.md
│   │   ├── scripts/
│   │   │   ├── collect_evalscope_results.py # evalscope 结果收集脚本
│   │   │   ├── compare_npu_benchmark.py     # 框架对比脚本
│   │   │   └── validate_framework_cli.py     # CLI 验证脚本
│   │   └── references/
│   │       └── npu-fairness-rules.md         # NPU 公平性规则
│   │
│   ├── xllm-npu-profiler/                   # Profiling 分析
│   │   ├── SKILL.md
│   │   ├── scripts/
│   │   │   ├── analyze_xllm_npu_profile.py   # Profiling 解析脚本
│   │   │   └── render_triage_npu.py          # 五表 Markdown 渲染
│   │   └── references/
│   │       ├── npu-fuse-catalog.md          # NPU 融合算子目录
│   │       ├── npu-overlap-catalog.md       # NPU 重叠机会目录
│   │       ├── ascend-profiling-formats.md   # 昇腾 Profiling 格式说明
│   │       ├── qwen35-27b-kernel-profile.md  # Qwen3.5 kernel profiling 对比
│   │       └── source-map.md                 # xLLM profiler 源码地图
│   │
│   ├── xllm-npu-sota-loop/                  # SOTA 自治优化循环
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── optimization-paths.md        # 昇腾优化路径详细文档
│   │       └── stop-conditions.md           # 停止条件说明
│   │
│   ├── xllm-npu-code-review/                # NPU 代码审查
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── npu-code-patterns.md         # NPU 代码模式
│   │       └── common-pitfalls.md           # 常见陷阱
│   │
│   ├── xllm-npu-accuracy-debug/             # 精度异常定位
│   │   └── SKILL.md
│   │
│   └── xllm-npu-incident-triage/            # 生产事故诊断
│       ├── SKILL.md
│       └── references/
│           ├── npu-error-catalog.md          # NPU 错误目录
│           └── replay-workflow.md           # 复现工作流
│
├── model-pr-optimization-history/            # PR 驱动模型历史
│   ├── SKILL.md
│   ├── scripts/query.py
│   └── xllm/
│       ├── deepseek-v3.md
│       ├── qwen3-core.md
│       └── glm-5.md
│
└── kernel-pilot/                             # NPU 内核辅助
    ├── SKILL.md
    ├── knowledge/
    │   ├── tilelang-patterns.md
    │   └── ascendc-patterns.md
    ├── references/
    │   └── a3-specs.md                       # 910B3 (A3) 硬件规格
    └── tools/
        └── npu-op-benchmark.py               # 算子基准测试
```

## 安装方式

### opencode

将 `skills/` 下各 SKILL.md 目录 symlink 到 opencode skills 目录：

```bash
for skill_dir in skills/xllm-npu-*/; do
  ln -sf "$(pwd)/$skill_dir" ~/.config/opencode/skills/
done
```

### Claude Code / Codex / Codex CLI

```bash
# symlink 或 copy 到对应 runtime 的 skill 目录
for skill_dir in skills/xllm-npu-*/; do
  ln -sf "$(pwd)/$skill_dir" .claude/skills/
done
```

### 通用 Agent

直接复制目标 skill 目录到 agent 工作目录。

## 运行 RLCR（Research-Learn-Code-Review）优化循环的快速示例

```text
> 在 Qwen3-32B 上，对 xLLM、vLLM-Ascend、SGLang 在 A3 NPU 上做 SOTA 对比优化
```

预期执行路径：
1. `xllm-npu-benchmark`：建立公平基准
2. `xllm-npu-profiler`：Profiling 五表分析
3. `model-pr-optimization-history`：查询历史 PR
4. `xllm-npu-sota-loop`：RLCR 迭代优化
5. `kernel-pilot`：针对热点 kernel 的专项优化

通用多框架示例：

```text
> 在 Qwen3-32B 上，比较 xLLM、vLLM-Ascend、SGLang NPU 在 A3 上的
> OpenAI-compatible serving 性能，使用相同 workload 和 SLA，输出公平
> benchmark、profiling 五表、gap 判定和下一步优化 plan。
```

预期执行路径：
1. 记录三个框架的 commit、启动参数、NPU/CANN/容器环境。
2. 使用同一 workload 和 SLA 做 config-driven benchmark。
3. 只对落后且 gap > 1% 的框架进入 profiling。
4. 用统一 NPU 五表 + capacity / compute 证据定位瓶颈。
5. 对目标框架执行单 patch RLCR，结果回写 humanize ledger 和 PR history。
