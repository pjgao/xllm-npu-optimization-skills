# Instructions

面向首次使用本仓库的工程师和 AI 助手的快速指引。

## 这是什么

一套**证据驱动的 NPU 大模型推理与 AI Infra 开发优化闭环**的 skill 集合。
当你拿到一个新模型（如 Qwen3.5-27B）需要在昇腾 910B3 上做到最优性能时，
按照这套 skill 一步步走，而不是盲目改代码。

当前仓库以 xLLM 为首个完整落地框架，但流程不是 xLLM 专用。后续应作为
xLLM、vLLM-Ascend、SGLang NPU 后端共用的一套标准流程：统一 benchmark
证据、统一 profiling 产物、统一 RLCR 账本，再通过框架适配层处理不同
启动命令、日志、metrics、profiling 和源码路径。

## 30 秒入门

1. **确定模型与硬件**  
   `export RUN_ROOT=/home/g00510989/runs/$(date +%Y%m%d)_<model>_npu_sota`

2. **加载主驱动 skill** `xllm-npu-sota-loop` — 它会按 6-Phase 引导你完成整个优化闭环。虽然名字里有 xLLM，但当前应理解为 NPU SOTA loop 的首个实现入口。

3. **每个 Phase 内**按需加载专项 skill：
   - 采基准 → `xllm-npu-benchmark` (推荐 evalscope `line_by_line` plugin；逐步扩展为 xLLM / vLLM-Ascend / SGLang NPU 共用)
   - 看瓶颈 → `xllm-npu-profiler`
   - 审代码 → `xllm-npu-code-review`
   - 排故障 → `xllm-npu-incident-triage`

4. **遇到硬骨头**（所有现成路径都试过了还差性能）→ 调用 `kernel-pilot` 自研算子

5. **全程参考** `model-pr-optimization-history` 避免重复劳动

## 必读文档

| 文档 | 位置 | 用途 |
|------|------|------|
| AGENTS.md | `./AGENTS.md` | AI 助手必读，含标准工作流与反模式 |
| 通用 NPU 标准流程 | `docs/npu-ai-infra-standard-workflow.md` | 多框架通用定位、Phase 定义、适配层原则 |
| 实现路线图 | `docs/implementation-roadmap.md` | 仓库待补能力与阶段规划 |
| 设计文档 | `docs/xllm-npu-optimization-design.md` | 仓库设计原理与 6-Phase 路线图 |
| Qwen3.5-27B 实战 | `docs/qwen35-27b-optimization-guide.md` | 端到端优化案例（含 5 表解读示例） |

## 仓库结构约定

```
skills/                          # 核心 skills（当前 xLLM 前缀为历史落地入口，流程逐步通用化）
├── xllm-npu-benchmark/          # NPU 多框架公平基准对比
├── xllm-npu-profiler/           # NPU 性能诊断五表
├── xllm-npu-sota-loop/          # NPU RLCR 闭环主驱动
├── xllm-npu-code-review/        # NPU 代码审查
└── xllm-npu-incident-triage/    # 事故诊断

kernel-pilot/                    # 算子自研 skill + 工具链
model-pr-optimization-history/   # PR 历史台账（自动追加）
references/                      # 共享知识（NPU 模式 / 融合目录 / 重叠目录）
docs/                            # 设计 + 分析 + 实战案例
```

后续新增通用能力时建议使用框架适配层：

```
frameworks/
├── xllm/
├── vllm-ascend/
└── sglang/
```

每个框架目录保存 launch template、help snapshot、metrics endpoint、
profiling 采集说明、source map 和 known pitfalls。现有 xLLM 经验仍保留在
docs、humanize、patches 和 model-pr-optimization-history 中，不做删除。

## 实测环境

首次实测于 2026-05-23：

| 项目 | 值 |
|------|----|
| Host | 192.168.13.154 |
| Container | xllm-gpj |
| NPU | Ascend 910B3 x2 (TP=2) |
| CANN | 8.5.0 |
| torch_npu | 2.7.1.post2 |
| xLLM commit | 455a99cb |
| 模型 | Qwen35-27B (混合注意力 + MTP) |
| Benchmark | evalscope `line_by_line` |

**已知问题**:
- MTP 模式因 `npu_add_rms_norm` kernel shape 不匹配 (error 561002) 崩溃，等待 xLLM 修复
- TTFT ~4s 较长 (长序列 prefill 慢)，可能需 chunked prefill 优化

详细事故报告见 `xllm-npu-incident-triage` skill 的 "实测事故案例" 章节。

这些记录是通用化后的首批 case study。扩展到 vLLM-Ascend 或 SGLang NPU 时，
应以相同格式保留环境、命令、失败原因、profiling 和最终判断。

## 常见疑问

**Q: 我可以直接跳到写 patch 吗？**  
A: 不可以。必须先经过 Phase 1 基准 + Phase 3 profiling。否则无法证明 patch 有效、无法证明打中了真正的瓶颈。

**Q: 为什么不让 xLLM 单独追赶 vLLM-Ascend？**  
A: vLLM-Ascend 默认配置常常不是最优的，而 xLLM 默认常常已调过。公平对比才能让结论可信。

**Q: 这个仓库是不是只服务 xLLM？**  
A: 不是。xLLM 是当前最完整的落地样本。标准流程应通用于 xLLM、vLLM-Ascend、SGLang NPU 等框架；区别只在框架适配层。

**Q: 我可以在 910B2 上用吗？**  
A: 可以，但部分 skill（kernel-pilot 的 A3 算子特性、block_size=128 默认值）需要调整。**请 fork 一份针对 B2 的变体**。

**Q: 我优化完一个模型后，下次能复用吗？**  
A: 能。`model-pr-optimization-history` 台账 + `optimization-ledger.md` 在每次 RLCR 时自动追加，下次查 `query.py` 即可。

**Q: benchmark 用什么工具？**  
A: 推荐 evalscope `line_by_line` plugin (容器内已安装)。每行 JSONL 是完整的 OpenAI 请求 body，直接发送。详见 `xllm-npu-benchmark` SKILL。

## 联系与反馈

- 设计问题 / 新 skill 需求：项目内部讨论
- opencode 使用问题：`/help` 或在 https://github.com/anomalyco/opencode/issues 反馈
