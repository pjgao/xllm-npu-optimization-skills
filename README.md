# xllm-npu-optimization-skills

Agent-ready playbooks for xLLM inference optimization on Huawei Ascend NPU (910B3/A3).

对标参考仓库：[AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS)
竞品对照框架：[vLLM-Ascend](https://github.com/vllm-project/vllm-ascend)

## 仓库定位

为 xLLM 推理框架在华为昇腾 NPU 910B3 (A3) 上提供：
- 公平基准测试（xLLM vs vLLM-Ascend）
- 昇腾 Profiling 五表分析报告
- RLCR 驱动的 SOTA 自治优化循环
- NPU 特化代码审查
- 生产事故诊断
- PR 驱动模型知识库
- NPU 内核证据辅助

## 核心 Skills

| Skill | 说明 | 对标 |
|-------|------|------|
| [`xllm-npu-benchmark`](skills/xllm-npu-benchmark/SKILL.md) | xLLM vs vLLM-Ascend 公平基准测试 | `llm-serving-auto-benchmark` |
| [`xllm-npu-profiler`](skills/xllm-npu-profiler/SKILL.md) | 昇腾 Profiling 五表分析 | `llm-torch-profiler-analysis` |
| [`xllm-npu-sota-loop`](skills/xllm-npu-sota-loop/SKILL.md) | NPU SOTA 自治优化循环 | `sglang-sota-humanize-loop` |
| [`xllm-npu-code-review`](skills/xllm-npu-code-review/SKILL.md) | NPU 特化代码审查 | `sglang-humanize-review` |
| [`xllm-npu-incident-triage`](skills/xllm-npu-incident-triage/SKILL.md) | NPU 生产事故诊断 | `sglang-prod-incident-triage` |

## 辅助层

| 组件 | 说明 |
|------|------|
| [`model-pr-optimization-history`](model-pr-optimization-history/SKILL.md) | xLLM 模型家族 PR 驱动历史档案 |
| [`kernel-pilot`](kernel-pilot/SKILL.md) | NPU 内核证据辅助（TileLang/AscendC/Triton-Ascend） |

## 环境要求

- 华为昇腾 910B3 (A3) NPU
- HDK Driver >= 25.2.0
- CANN >= 8.0.RC1
- xLLM 框架（最新 commit）
- vLLM-Ascend（对照组，最新 commit）

## 目录结构

```
xllm-npu-optimization-skills/
├── AGENTS.md                                # Agent 使用指南
├── INSTRUCTIONS.md                          # 优化指令模板
├── README.md                                # 本文档
│
├── docs/                                    # 设计文档
│   ├── ENVIRONMENT.md                       # 环境配置指南
│   ├── ai-infra-analysis.md                # AI-Infra 框架分析
│   ├── environment-info.md                  # 环境信息采集
│   ├── pr-1536-mtp-transpose-elimination.md # PR #1536 分析
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

## 运行 RLCR 优化循环的快速示例

```text
> 在 Qwen3-32B 上，xLLM 与 vLLM-Ascend 在 A3 NPU 上做 SOTA 对比优化
```

预期执行路径：
1. `xllm-npu-benchmark`：建立公平基准
2. `xllm-npu-profiler`：Profiling 五表分析
3. `model-pr-optimization-history`：查询历史 PR
4. `xllm-npu-sota-loop`：RLCR 迭代优化
5. `kernel-pilot`：针对热点 kernel 的专项优化
