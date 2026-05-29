---
name: xllm-npu-sota-loop
description: xLLM 昇腾 NPU 910B3 (A3) SOTA 自治优化循环。在 xLLM 与 vLLM-Ascend 之间执行 RLCR（Research-Learn-Code-Review）驱动的迭代优化，直到 xLLM 在固定工作负载上达到或超越 SOTA。当用户需要系统性提升 xLLM NPU 推理性能时使用。
---

# xLLM NPU SOTA 自治优化循环

## 概述

将 xLLM NPU 优化工作流作为一个 6-Phase 自治循环运行：

```
Phase 0   → 初始化
Phase 0.5 → 模型 PR 历史知识门控
Phase 1   → 固定公平基准测试（调用 xllm-npu-benchmark）
Phase 2   → 差异判定（阈值 1%）
Phase 3   → RLCR 前的必要 Profiling（调用 xllm-npu-profiler）
Phase 4   → 构建优化 Plan
Phase 5   → 启动 RLCR 迭代
```

核心原则：
- **不比较不公平数据**：永远不将 tuned xLLM 与 vLLM-Ascend defaults 比较
- **不跳过 profiling**：Phase 3 报告不存在前不得开始 patch
- **证据驱动**：所有优化决策有 benchmark 数据 + profiling trace + PR 历史支持

## Phase 0: 初始化

收集以下信息：

| 参数 | 示例 | 必填 |
|------|------|------|
| 模型 | Qwen3-32B | yes |
| 精度 | bf16 | yes |
| NPU 型号 | A3 | yes |
| CANN 版本 | 8.0.RC1 | yes |
| 框架集 | xllm, vllm-ascend | yes |
| artifact root | /runs/qwen3-32b_npu_sota/ | yes |

创建运行目录：

```bash
mkdir -p runs/$(date +%Y%m%d)_<model_slug>_npu_sota/{benchmark,profiles,analysis,history,kernel,patches,humanize}
```

确认环境就绪：
- `npu-smi info` 所有 NPU 健康
- 两个框架 commit hash 已记录
- 模型权重路径已验证

## Phase 0.5: 模型 PR 历史门控

调用 [`model-pr-optimization-history`](../../model-pr-optimization-history/SKILL.md) skill：

```text
> 查询 xllm 模型历史上与 <model_name> 相关的优化 PR
```

检查：
- 已知风险和待验证想法
- 之前优化尝试的结果（成功/失败/未尝试）
- 需要优先检查的文件和符号

写入 `history/xllm-model-history-notes.md`。

## Phase 1: 固定公平基准测试

调用 [`xllm-npu-benchmark`](../xllm-npu-benchmark/SKILL.md) skill：

```text
> 在 A3 NPU 上，使用 benchmark skill 对 xLLM 和 vLLM-Ascend 进行公平对比
> 模型：<model>，工作负载：真实流量 JSONL (line_by_line)
> 搜索层级：Tier 2
```

**推荐 benchmark 工具**: evalscope (`pip install evalscope`)

```bash
evalscope perf \
  --model <model_name> \
  --url http://127.0.0.1:<port>/v1/chat/completions \
  --api openai \
  --dataset line_by_line \
  --dataset-path /path/to/dataset.jsonl \
  --parallel 1 --number 5 \
  --connect-timeout 120 --read-timeout 300 \
  --outputs-dir $RUN_ROOT/benchmark/baseline/parallel_1_number_5/
```

规则：
- 不允许将 tuned xLLM 与 vLLM-Ascend defaults 比较
- 每框架各自独立搜索最优配置
- 记录完整启动命令

产出：
- `benchmark/*/benchmark_summary.json` — 汇总指标
- `benchmark/*/benchmark_percentile.json` — 延迟百分位
- `benchmark/*/perf_report.html` — HTML 报告

## Phase 2: 差异判定（阈值 1%）

读取 `benchmark/summary.md`，计算：

```
gap = (vllm_best_throughput - xllm_best_throughput) / vllm_best_throughput × 100%
```

| 结果 | 行为 |
|------|------|
| `gap <= 1%` | xLLM 达到 SOTA，写报告结束 |
| `gap > 1%` | 进入 Phase 3 |
| xLLM 已胜出 | 确认后写报告结束 |

## Phase 3: RLCR 前的必要 Profiling

调用 [`xllm-npu-profiler`](../xllm-npu-profiler/SKILL.md) skill：

```text
> 对 xLLM 和 vLLM-Ascend 使用 benchmark 的 winning-commands 采集 Profiling
> 使用慢场景的实际 input/output 长度
> Prefill 和 Decode 分离采集
```

规则：
- 必须使用 benchmark 慢场景的实际 input/output 长度（不用默认短输入）
- Prefill 和 Decode 必须分离采集
- 两阶段 trace（eager/graph-on）推荐

产出：
- `profiles/xllm/` — xLLM Profiling 数据
- `profiles/vllm/` — vLLM-Ascend Profiling 数据
- `analysis/root-cause.md` — 五表报告 + 根因分析

**此报告不存在前不得开始任何 patch。**

## Phase 4: 构建优化 Plan

基于五表报告，按以下昇腾优化路径分类确定优先级：

### 优化路径优先级

| 优先级 | 路径 | 说明 |
|--------|------|------|
| P0 | 融合算子替换（torch_npu） | 替换标准算子为 NPU 融合算子 |
| P0 | KV Cache 优化 | PA 模式 / NZ 格式 / MLA 压缩 |
| P0 | 图模式适配（GE/AclGraph） | Decode 启用图模式，Prefill 保持 eager |
| P1 | 权重预取（npu_prefetch） | 提前加载下一层权重到 SRAM |
| P1 | 多流重叠 | 计算-通信双流 |
| P1 | SuperKernel 融合 | 算子二进制融合 |
| P1 | 并行策略（TP/EP/DP） | 最优并行配置 |
| P2 | 通信优化（HCCL） | AllReduce/AllGather 重叠 |
| P2 | 投机解码 | Draft model 选择和调优 |
| P2 | MoE EPLB | 动态专家负载均衡 |

详细优化路径文档见 [references/optimization-paths.md](references/optimization-paths.md)。

Plan 写入 `humanize/refined-plan.md`，格式：

```markdown
## 优化 Plan

### 目标
将 xLLM 在 <model> 上的吞吐从 X req/s 提升到接近或超越 vLLM-Ascend 的 Y req/s

### 根因总结
（来自 Phase 3 的五表报告）

### 优化队列
1. [P0] 融合算子：...
   - 预计收益：...%
   - 涉及文件：...
   - 风险：...

2. [P0] KV Cache 优化：...
   - 预计收益：...%
   - 涉及文件：...
   - 风险：...
```

## Phase 5: RLCR 迭代

### 每轮 RLCR

```
Research: 分析五表报告，确定本轮 patch 方向
Learn:    查询 model-pr-history，学习已有方案
Code:     编写 patch
Review:   调用 xllm-npu-code-review 审查
Validate: 重新编译 + benchmark + profiling
Record:   记录到台账
```

### 台账文件

| 文件 | 说明 |
|------|------|
| `humanize/attempt-ledger.md` | 每次尝试记录（成功/失败/放弃） |
| `humanize/optimization-ledger.md` | 有效优化记录（有性能提升证据） |
| `humanize/source-idea-ledger.md` | 源码想法及出处（PR/论文/文档） |
| `humanize/lineage.jsonl` | 血缘追踪（patch → benchmark → profiler） |

### attempt-ledger 格式

```markdown
## Attempt #N
- 日期：YYYY-MM-DD
- 方向：融合算子 / KV Cache / 图模式 / ...
- 修改文件：file1.cpp, file2.h
- 结果：成功（+X%） / 失败（原因） / 放弃（原因）
- benchmark 文件：benchmark/round_N_results.jsonl
- profiler 文件：analysis/round_N_root_cause.md
```

### 验证要求

每轮 patch 后：
1. 拉取 PR 分支后，构建前先完成环境门禁：
   - 执行 `git submodule update --init --recursive`，确认依赖已同步到主仓记录的 commit。
   - 执行 `git status --short`，必须无主仓修改、无子模块 `m`/`M`/`+`/`-` 状态；若有变动，先定位来源，不直接编译。
   - 执行 `pgrep -af 'setup.py|cmake|make|ninja|tilelang|submodule--helper|git submodule'`，确认没有其他正在进行的编译、子模块更新或可能污染 build 目录的冲突进程。
2. 编译通过：`python setup.py build --device "npu"`
3. 单元测试通过：`python setup.py test --device "npu"`；需要合并验证时可使用 `python setup.py build test --device "npu"`。
4. 重新运行 benchmark winning-commands
5. 重新采集 profiling
6. 对比前后的五表报告

### 内核优化准入条件

全部满足才启用 kernel-pilot 路径：

1. xLLM 仍落后 vLLM-Ascend > 1%
2. 慢阶段某个 kernel 族占 AICore 时间 ≥ 1%
3. Profiler 五表对比显示该 kernel 是差异的合理解释
4. 有明确的正确性参考和代表性 shapes

满足后调用 [`kernel-pilot`](../../kernel-pilot/SKILL.md) skill。

## 停止条件

详见 [references/stop-conditions.md](references/stop-conditions.md)。

| 条件 | 说明 |
|------|------|
| xLLM 胜出 | 固定工作负载上击败 vLLM-Ascend SLA-passing 结果 |
| 平局 | 稳定 1% 阈值内 |
| 硬件限制 | NPU 不支持特定算子/特性 |
| 瓶颈已至 | 热点路径接近硬件/算法极限 |
| 无进展 | 连续 3 轮尝试无性能改善 |

## 输出

循环结束时交付：
- `benchmark/final_summary.md` — 最终对比结果
- `analysis/final_root_cause.md` — 最终五表报告
- `humanize/refined-plan.md` — 完整优化 plan
- `humanize/attempt-ledger.md` — 所有尝试记录
- `humanize/optimization-ledger.md` — 有效优化总结
- `patches/` — 所有 patch 文件

## 真实案例：Qwen3.5-27B MTP 优化（2026-05）

### Phase 0 → 初始化

| 参数 | 值 |
|------|---|
| 模型 | Qwen3.5-27B |
| 精度 | bf16 |
| NPU | 910B3 (A3), TP=2 |
| CANN | 8.5.0 |
| artifact root | `/home/g00510989/runs/20260523_qwen35_27b_npu_sota/` |

### Phase 1 → 基准测试

| 配置 | Throughput (tok/s) | TPOT (ms) | TTFT (ms) |
|------|-------------------|-----------|-----------|
| Baseline (并发 1) | 29.33 | 29.8 | 3564 |
| MTP (nst=1) | ~35 | ~25 | - |
| MTP-transpose (nst=1) | **39.54** | **21.9** | - |

### Phase 2 → 差异判定

MTP-transpose 比 baseline **+35% throughput, -27% TPOT** → 继续优化。

### Phase 3 → Profiling

- baseline / mtp / mtp-transpose 三组完整 NPU profiling
- 分析 `op_statistic_*.csv` / `op_summary_*.csv` 发现通信和 MatMul 是主要热点
- MTP crash 根因: `conv_weight` shape `[k, C/tp]` vs v2 kernel 期望 `[C/tp, k]`

### Phase 5 → RLCR Round 1

```
Research: profiling 显示 MTP conv1d kernel shape 传递错误
Learn:    查 PR 历史，发现 v2 kernel 对 weight shape 有严格约束
Code:     qwen3_gated_delta_net_base.cpp L581/588 改用 conv_weight_transposed_
Review:   code-review skill 确认修复正确性
Validate: 编译 + 重启 + benchmark
Record:   attempt-ledger.md #1 → 成功 (+9.5% throughput on MTP)
```

### 失败实验记录

| 实验 | 方向 | Throughput | 结果 |
|------|------|-----------|------|
| A1 chunked_prefill | 调度策略 | ~28 tok/s | 无改善 |
| B1 memory | 内存优化 | ~28 tok/s | 无改善 |
| C1 piecewise | CUDA Graph 策略 | ~29 tok/s | 无改善 |
| D1 no_padding | padding 消除 | ~28 tok/s | 无改善 |
| E1 batch | batch 策略 | ~29 tok/s | 无改善 |

**结论**: MTP-transpose 是当前最优配置，后续优化应基于此起步。
