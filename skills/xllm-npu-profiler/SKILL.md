---
name: xllm-npu-profiler
description: 昇腾 NPU 910B3 (A3) 上的 xLLM 推理 Profiling 分析。提供五表报告（kernel/重叠/融合/下发效率/内存效率），支持 xLLM 和 vLLM-Ascend 对比分析，Prefill/Decode 阶段分离。当用户需要定位 NPU 推理性能瓶颈时使用。
---

# xLLM NPU Profiling 分析

## 概述

在华为昇腾 NPU 910B3 (A3) 上进行推理 Profiling 分析，输出五表报告。

工作流入口：
- [scripts/analyze_xllm_npu_profile.py](scripts/analyze_xllm_npu_profile.py) — Profiling 数据解析
- [scripts/render_triage_npu.py](scripts/render_triage_npu.py) — 五表 Markdown 渲染

## 能力矩阵

| 能力 | xLLM | vLLM-Ascend |
|------|------|-------------|
| 已有 trace 分析 | yes | yes |
| 实时采集 | yes（XLLM_PROFILING=1） | yes（profiler 配置） |
| 阶段分离采集 | yes（Prefill/Decode 独立） | 有限 |
| 昇腾 msprof 采集 | yes | yes |
| 两阶段 trace 对比 | yes（eager/graph-on） | yes |

## 五表输出契约

`triage` 输出固定五张表，每张表仅渲染累计 GPU 时间占比 ≥ 1.0% 的行。

| 表名 | 内容 | 数据来源 |
|------|------|---------|
| **Kernel Table** | AICore 内核按 GPU 时间占比排序 | `op_statistic.csv` / `kernel_details.csv` |
| **Overlap-Opportunity Table** | 计算-通信重叠机会 | `step_trace_time.csv` 时间线分析 |
| **Fuse-Pattern Table** | 融合算子模式匹配 | 模型代码 + torch_npu 映射（确定性，非模糊匹配） |
| **下发效率表** | Hostbound 分析：空闲率/AICore 利用率 | `step_trace_time.csv` + `analysis.db` |
| **内存效率表** | xTensor/KV Cache 利用率与碎片 | xLLM metric + profiling 数据 |

## 使用场景

- 分析 xLLM 或 vLLM-Ascend 的昇腾 Profiling trace
- 实时采集运行中服务的 Profiling 数据
- 总结 Prefill/Decode 阶段哪些 kernel 族主导性能
- 判断代码路径是否仍有重叠/融合机会
- 检查已知优化路径是否被正确启用

## 主要工作流

### 1. 已有 trace 分析

```bash
python scripts/analyze_xllm_npu_profile.py \
  --input /path/to/profiling_results/ \
  --framework xllm
```

输入目录应包含昇腾 Profiling 标准输出：
- `step_trace_time.csv`
- `op_statistic.csv`
- `kernel_details.csv`
- `analysis.db`

### 2. xLLM 实时采集

```bash
XLLM_PROFILING=1 xllm serve /path/to/model \
  --tensor-parallel-size 4 \
  --graph-mode npugraph_ex

python scripts/analyze_xllm_npu_profile.py \
  --framework xllm \
  --url http://127.0.0.1:8080 \
  --output-dir /tmp/xllm-profile/ \
  --num-steps 5 \
  --profile-by-stage
```

`--profile-by-stage` 分别收集 Prefill 和 Decode 阶段的 trace（推荐）。
默认 warmup 10 步 + 活跃采集 5 步。

### 3. vLLM-Ascend 实时采集

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve /path/to/model \
  --tensor-parallel-size 4 \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/vllm-profile"}'

python scripts/analyze_xllm_npu_profile.py \
  --framework vllm-ascend \
  --url http://127.0.0.1:8000 \
  --output-dir /tmp/vllm-profile/ \
  --num-steps 5
```

### 4. 两阶段 trace 分析（eager/graph-on 对比）

```bash
python scripts/analyze_xllm_npu_profile.py triage \
  --mapping-input /path/to/eager_profile/ \
  --formal-input /path/to/graph_on_profile/ \
  --framework xllm
```

mapping trace 用于恢复 `kernel → cpu_op → python scope` 映射。
formal trace 用于实际性能分析。

### 5. 阶段分离采集

```bash
python scripts/analyze_xllm_npu_profile.py \
  --framework xllm \
  --url http://127.0.0.1:8080 \
  --output-dir /tmp/xllm-profile-staged/ \
  --profile-workload both \
  --profile-by-stage
```

自动分别发送：
- Prefill 请求：input=4090 → output=1
- Decode 请求：input=1 → output=2048

与 SOTA loop 联动时必须使用慢场景的实际 input/output 长度。

## 全局瓶颈判定

```
空闲率 >20%  → Hostbound（算子下发/GE编译/H2D传输瓶颈）
计算率 >85%  → Computing（热点 kernel 优化空间）
通信率 >10%  → Communication（AllReduce/AllGather 瓶颈）
内存碎片高   → Memory（xTensor/KV Cache 碎片）
```

判定后分发到子分析：
- **Hostbound**：图模式覆盖率、GE/AclGraph 编译开销、算子下发效率、H2D/D2H 拷贝
- **Computing**：Top-N 热点算子、MatMul/GEMM 形状分析、融合机会（torch_npu）
- **Communication**：AllReduce/AllGather 带宽、TP 通信开销、通信-计算重叠率
- **Memory**：KV Cache 碎片率、xTensor 池命中率、Block 分配策略

## 工作流

### 单 trace 分析

1. 如果用户仅需诊断，一个 trace 足够
2. 优先使用单 rank trace 而非合并 trace
3. 对于实时采集，确认框架的 Profiling 前置条件已满足
4. 优先使用 `--profile-by-stage`，Prefill/Decode 瓶颈差异显著
5. 采集前清空目标目录

### 两 trace 分析

1. 先采集 mapping trace（graph 关闭 / 低融合配置）
2. 再采集 formal trace（实际服务优化配置）
3. 运行 `triage` 生成五表报告
4. 按此顺序阅读结果：
   - Kernel Table → Overlap-Opportunity Table → Fuse-Pattern Table → 下发效率表 → 内存效率表
5. 在声称"新优化机会"前，对照：
   - [references/npu-fuse-catalog.md](references/npu-fuse-catalog.md) — 已知融合模式
   - [references/npu-overlap-catalog.md](references/npu-overlap-catalog.md) — 已知重叠机会
6. 优先报告：
   - 本应生效的已有融合/重叠路径
   - 本应生效但疑似禁用/不支持/回归的路径
   - 其他框架已有但本地缺失的上游模式
   - 仅当没有 catalog 条目匹配时，才报告真正的"新机会"

### 昇腾 Profiling 数据格式

详见 [references/ascend-profiling-formats.md](references/ascend-profiling-formats.md)。

关键文件：
- `step_trace_time.csv`：Step 级别时间追踪
- `op_statistic.csv`：算子调用统计
- `kernel_details.csv`：AICore kernel 详细时间
- `analysis.db`：SQLite 格式详细分析数据

## 渲染五表 Markdown

```bash
python scripts/render_triage_npu.py \
  --analysis-root /path/to/analysis_root \
  --output /path/to/analysis_bundle.md
```

输出将按模型分组，保留每个框架的五表。

## 参考资料

按需加载：
- [references/source-map.md](references/source-map.md) — xLLM profiler 入口和 trace 写入路径
- [references/npu-fuse-catalog.md](references/npu-fuse-catalog.md) — NPU 融合算子目录
- [references/npu-overlap-catalog.md](references/npu-overlap-catalog.md) — NPU 重叠机会目录
- [references/ascend-profiling-formats.md](references/ascend-profiling-formats.md) — 昇腾 Profiling 数据格式

## MTP 相关性能检查要点

当分析 MTP (Multi-Token Prediction) 相关 profiling trace 时，额外检查：

| 检查维度 | 健康值 | 异常信号 |
|---------|-------|---------|
| `--num_speculative_tokens` | `nst=1` | `nst>=2` 在 Qwen3.5-27B + 910B3 上实测为严重负优化 |
| Spec Accept Rate | 47-50% | <40% 表明 draft 质量或 nst 设置不当 |
| Decoded Tok/Iter | 1.88-1.98 (nst=1 理论最大 2) | <1.5 需检查 draft model 加载 |
| reserved_linear_bytes | <3 GB (nst=1) | >6 GB 表明 draft+verification 内存压力大 |
| KV Cache blocks | ≥80% baseline | <60% 表明 draft model 挤占了主 model KV 空间 |
| Prefill warmup 时间 | ≤baseline | 2x baseline → draft prefill penalty 主导延迟 |

**关键结论 (Qwen3.5-27B @ 910B3)**:
- `nst=1`：吞吐 +20-23%，TTFT 零惩罚，TPOT -22%，**推荐使用**
- `nst=2`：吞吐 -52%，TTFT +77%，**不推荐**（per-token decode 更快但 TTFT 惩罚主导总延迟）

## 输出契约

返回：
- trace 路径或生成的 profiling 路径
- 框架（xLLM / vLLM-Ascend）
- 模型/服务器参数
- 五表报告（kernel/overlap/fuse/下发/内存）
- 可选相似性标注（high/medium/low）
- Prefill/Decode 各阶段的主要瓶颈总结
- 重叠证据来源说明（单 trace / 两 trace）
