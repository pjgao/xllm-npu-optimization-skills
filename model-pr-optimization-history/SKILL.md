---
name: model-pr-optimization-history
description: 查询 xLLM 历史 PR 中的优化信息，辅助当前模型的优化决策。
---

# Model PR Optimization History

通过查询 xLLM 代码库的历史 PR，获取模型相关的优化信息，避免重复工作。

## 使用场景

1. 开始优化某模型前，先查询历史 PR 中的已知优化
2. 遇到特定性能问题时，搜索历史 PR 中的解决方案
3. 了解 xLLM 对该模型的支持状态

## 工作流

### Step 1: 查询历史 PR

使用 `scripts/query.py` 查询 xLLM 仓库的 PR：

```bash
cd /home/gaopengju/projects/xllm

# 按模型查询
python ../xllm-npu-optimization-skills/model-pr-optimization-history/scripts/query.py \
    --model "Qwen3" \
    --type optimization

# 按关键词查询
python ../xllm-npu-optimization-skills/model-pr-optimization-history/scripts/query.py \
    --keyword "flash attention" \
    --type feature

# 查询特定时间段
python ../xllm-npu-optimization-skills/model-pr-optimization-history/scripts/query.py \
    --since "2024-01-01" \
    --until "2024-06-01"
```

### Step 2: 整理优化历史

将查询结果整理为模型档案，存入 `xllm/<model>.md`：

```markdown
## 已知优化 (来自 PR 历史)

| PR # | 优化内容 | 效果 | 日期 |
|------|---------|------|------|
| #123 | 实现 NPU fused attention | +15% throughput | 2024-03-15 |
| #456 | 添加 KV Cache NZ 格式支持 | -20% memory | 2024-04-20 |
```

### Step 3: 识别未覆盖优化

对比当前工作负载与历史优化，找出：
1. **已实现的优化**：无需重复实现
2. **部分实现的优化**：需要进一步完善
3. **未实现的优化**：新的优化机会

## 查询输出字段

| 字段 | 说明 |
|------|------|
| pr_number | PR 编号 |
| title | PR 标题 |
| author | 作者 |
| merged_at | 合并时间 |
| labels | PR 标签 |
| files_changed | 涉及的文件 |
| related_model | 相关模型 |
| optimization_type | 优化类型 |
| performance_impact | 性能影响描述 |

## 模型档案目录

- `model-pr-optimization-history/xllm/deepseek-v3.md` — DeepSeek-V3 (MoE)
- `model-pr-optimization-history/xllm/qwen3-core.md` — Qwen3 系列 (Dense)
- `model-pr-optimization-history/xllm/glm-5.md` — GLM-5 系列

## MTP 配置调优案例 (Qwen3.5-27B @ 910B3, 2026-05-23)

完整过程记录：

```
初始配置:
  --num_speculative_tokens 2 (nst=2)
  
预期: +50-80% 吞吐提升（每迭代多 2 token）

实测结果 (JD 真实数据集):
  Throughput: 14.47 tok/s vs baseline 29.88 tok/s → -52% 严重退化
  TTFT: 6899ms vs 3895ms → +77%
  TPOT: 44.6ms vs 29.6ms → +51%
  Accept Rate: 61.4% (看似良好)
  
根因分析:
  1. reserved_linear_bytes: 6.82GB (MTP) vs 2.29GB (baseline)，+4.5GB
     → gated_delta_net state 维护开销
  2. Prefill warmup 时间翻倍: 14052ms vs 7928ms
     → draft model prefill + 2-token verification = +6.1s
  3. 短输出场景 (avg 500-700 tok) TTFT 惩罚无法摊薄
  
解决方案:
  切换到 nst=1
  
调整后实测 (JD + random 数据集):
  Throughput: +20-23% (36.11 / 36.62 tok/s)
  TTFT: +0.6% (几乎无惩罚)
  TPOT: -22% (23.4ms vs 29.6ms)
  Accept Rate: 47-49%
  
经验教训:
  - nst=2 的 TTFT 惩罚远大于 decode 阶段的每 iteration 收益
  - Qwen3.5-27B 的 gated_delta_net 层在 MTP 模式下内存压力大
  - nst=1 是当前 910B3 上的最佳 balance point
  - 性能对比必须用同代码版本（控制变量）
```

### Case 2: 参数探索实验（chunked_prefill_A1/E1），全部失败 (2026-05-23)

```
背景:
  在 MTP nst=1 取得 +21% 之后，尝试 5 组参数调优继续压缩性能空间

实验设计:
  A1: chunked_prefill_size = 1024
  B1: max_memory_utilization = 0.85（默认 0.7）
  C1: graph_mode = PREFILL_PIECEWISE
  D1: decode_no_padding = True
  E1: max_seqs_per_batch = 32（默认 16）

结果: 全部无效或负向（-2% ~ -6%）

根因:
  - 单请求场景下，KV cache 容量、batch 调度开销均非瓶颈
  - 20K single-chunk prefill 已是最优，chunking 增加调度开销而非降低 TTFT
  - 所有实验的 TTFT 都比 baseline 高 20%（4639~4677ms vs 3895ms），
    说明这些参数在 prefill 阶段引入了额外 overhead

结论:
  参数层调优空间已用尽，需 profiling 进入 kernel-level 分析
```

## 维护

模型档案应随新 PR 合并而更新。建议：
- 每次 xLLM 新 PR 合并后，检查是否影响已归档的模型
- 季度审查一次档案的准确性
