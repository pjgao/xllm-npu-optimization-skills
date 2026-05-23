# Instructions

面向首次使用本仓库的工程师和 AI 助手的快速指引。

## 这是什么

一套**驱动证据驱动的 NPU 推理优化闭环**的 skill 集合。当你拿到一个新模型（如 Qwen3.5-27B）需要在昇腾 910B3 上做到最优性能时，按照这套 skill 一步步走，而不是盲目改代码。

## 30 秒入门

1. **确定模型与硬件**  
   `export RUN_ROOT=/home/g00510989/runs/$(date +%Y%m%d)_<model>_npu_sota`

2. **加载主驱动 skill** `xllm-npu-sota-loop` — 它会按 6-Phase 引导你完成整个优化闭环

3. **每个 Phase 内**按需加载专项 skill：
   - 采基准 → `xllm-npu-benchmark` (推荐 evalscope `line_by_line` plugin)
   - 看瓶颈 → `xllm-npu-profiler`
   - 审代码 → `xllm-npu-code-review`
   - 排故障 → `xllm-npu-incident-triage`

4. **遇到硬骨头**（所有现成路径都试过了还差性能）→ 调用 `kernel-pilot` 自研算子

5. **全程参考** `model-pr-optimization-history` 避免重复劳动

## 必读文档

| 文档 | 位置 | 用途 |
|------|------|------|
| AGENTS.md | `./AGENTS.md` | AI 助手必读，含标准工作流与反模式 |
| 设计文档 | `docs/xllm-npu-optimization-design.md` | 仓库设计原理与 6-Phase 路线图 |
| 竞品分析 | `docs/ai-infra-analysis.md` | xLLM 与 vLLM-Ascend 能力对比 |
| Qwen3.5-27B 实战 | `docs/qwen35-27b-optimization-guide.md` | 端到端优化案例（含 5 表解读示例） |

## 仓库结构约定

```
skills/                          # 5 个核心 skills
├── xllm-npu-benchmark/          # 公平基准对比
├── xllm-npu-profiler/           # 性能诊断五表
├── xllm-npu-sota-loop/          # RLCR 闭环主驱动
├── xllm-npu-code-review/        # NPU 代码审查
└── xllm-npu-incident-triage/    # 事故诊断

kernel-pilot/                    # 算子自研 skill + 工具链
model-pr-optimization-history/   # PR 历史台账（自动追加）
references/                      # 共享知识（NPU 模式 / 融合目录 / 重叠目录）
docs/                            # 设计 + 分析 + 实战案例
```

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

## 常见疑问

**Q: 我可以直接跳到写 patch 吗？**  
A: 不可以。必须先经过 Phase 1 基准 + Phase 3 profiling。否则无法证明 patch 有效、无法证明打中了真正的瓶颈。

**Q: 为什么不让 xLLM 单独追赶 vLLM-Ascend？**  
A: vLLM-Ascend 默认配置常常不是最优的，而 xLLM 默认常常已调过。公平对比才能让结论可信。

**Q: 我可以在 910B2 上用吗？**  
A: 可以，但部分 skill（kernel-pilot 的 A3 算子特性、block_size=128 默认值）需要调整。**请 fork 一份针对 B2 的变体**。

**Q: 我优化完一个模型后，下次能复用吗？**  
A: 能。`model-pr-optimization-history` 台账 + `optimization-ledger.md` 在每次 RLCR 时自动追加，下次查 `query.py` 即可。

**Q: benchmark 用什么工具？**  
A: 推荐 evalscope `line_by_line` plugin (容器内已安装)。每行 JSONL 是完整的 OpenAI 请求 body，直接发送。详见 `xllm-npu-benchmark` SKILL。

## 联系与反馈

- 设计问题 / 新 skill 需求：项目内部讨论
- opencode 使用问题：`/help` 或在 https://github.com/anomalyco/opencode/issues 反馈
