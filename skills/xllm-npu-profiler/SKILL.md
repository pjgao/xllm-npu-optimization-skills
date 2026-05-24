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

`triage` 输出固定五张表，每张表仅渲染累计 Device/AICore 时间占比 ≥ 1.0% 的行。

| 表名 | 内容 | 数据来源 |
|------|------|---------|
| **Kernel Table** | AICore/AI CPU 内核按 Device 时间占比排序 | `op_statistic*.csv` / `op_summary*.csv` |
| **Communication / Overlap-Opportunity Table** | 通信热点与计算-通信重叠机会 | `communication_statistic*.csv` + `task_time*.csv` |
| **Fuse-Pattern Table** | 融合算子模式匹配 | 模型代码 + torch_npu 映射（确定性，非模糊匹配） |
| **下发效率表** | Hostbound 分析：stream task 密度/间隔/等待 | `task_time*.csv` + `analysis.db` / `msprof_*.db` |
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
- `op_statistic*.csv`
- `op_summary*.csv`
- `task_time*.csv`
- `communication_statistic*.csv`
- `analysis.db` 或 `msprof_*.db`

脚本可以直接接收 `PROF_*` 根目录，也可以接收其中的
`mindstudio_profiler_output/` 子目录。MindStudio/msprof 自动生成的时间戳文件名会被
自动识别，例如 `op_statistic_20260523203844.csv`。

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
- `op_statistic*.csv`：算子调用统计，优先用于 Kernel Table
- `op_summary*.csv`：AICore/AI Vector 详细算子统计，作为 fallback
- `task_time*.csv`：task timeline，用于 stream 下发效率
- `communication_statistic*.csv`：HCCL 通信统计
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
- [references/qwen35-27b-kernel-profile.md](references/qwen35-27b-kernel-profile.md) — Qwen3.5-27B Baseline vs MTP kernel profiling 对比分析

## MTP 相关性能检查要点

当分析 MTP (Multi-Token Prediction) 相关 profiling trace 时，额外检查：

| 检查维度 | 健康值 | 异常信号 |
|---------|-------|---------|
| `--num_speculative_tokens` | 依 workload 扫描；20k/1k random TP=4 当前为 `nst=3` | 只看 accept rate 选择更大 nst，可能被 draft/verify 开销反噬 |
| Spec Accept Rate | nst=3 场景约 68-71% | <40% 表明 draft 质量或 nst 设置不当 |
| Decoded Tok/Iter | nst=3 场景约 3.2-3.4 | 明显低于 nst+1 上限且 TPOT 不降，需检查 draft/verify |
| reserved_linear_bytes | nst=3 约 4.54 GB | >6 GB 表明 draft+verification 内存压力大 |
| KV Cache blocks | nst=3 约 3414 blocks | 长上下文或高并发下 blocks 下降会限制容量 |
| Prefill warmup 时间 | ≤baseline | 2x baseline → draft prefill penalty 主导延迟 |

**关键结论 (Qwen3.5-27B @ 910B3)**:
- 旧 TP=2 / 96 input / 100 output 场景：`nst=1` 最优，`nst=2` 为严重负优化。
- 当前 TP=4 / random 20k input / 1k output / chunk prefill 场景：`nst=3` 最优；`nst=4/5` accept rate 更高但端到端吞吐下降。
- MTP 主要改善 decode/TPOT；长 prompt 的 TTFT 仍由 prefill 主导，不应用 TTFT 单独判断 MTP 是否有效。

### 2026-05-25 TP=4 / MTP=3 Profiling 记录

环境：xLLM `f514ad94`，Qwen35-27B + Qwen35-27B-mtp，Phy 8-11，`--enable_chunked_prefill=true`，`--num_speculative_tokens 3`。

Trace:
- Prefill-focused: `/home/g00510989/xllm/runs/20260524_tp4_random20k_1k/profiling/mtp3_chunk_prefill20k_out1_cards8_11_20260525_000017/PROF_000001_20260525000020794_DBKJLBLHERRILEQB`
- Decode-focused: `/home/g00510989/xllm/runs/20260524_tp4_random20k_1k/profiling/mtp3_chunk_decode32_out200_cards8_11_20260525_001223/PROF_000001_20260525001226480_GMAQBFHGBIOBQCIC`

Decode-focused 五表要点：
- Kernel: `MatMulV2` 36.3%, `allreduceAicpuKernel` 31.7%, `Transpose` 9.5%, `allgatherAicpuKernel` 2.4%, `fused_recurrent_gated_delta_rule_spec_fwd_kernel` 1.6%, `_causal_conv1d_update_kernel_npu_tiled_v2` 1.5%.
- Communication: HCCL 统计中 `hcom_allReduce_` 64.1%, `hcom_allGather_` 35.9%；TP=4 decode 仍有明显通信优化空间。
- Dispatch/Host: API 统计中 `aclrtSynchronizeStream` + `StreamSynchronize` 合计约 7.35s，`launch` 1.58s，`MemCopySync` 0.64s，小算子与同步开销非常重。
- Fuse pattern: MTP accept/verify 小算子累计明显，`Transpose` 505ms/36,701 calls、`Pack` 64.8ms/18,792 calls、`Range` 51.6ms/6,126 calls、`SoftmaxV2` 46.9ms/634 calls、`ArgMaxV2` 36.7ms/760 calls。
- Memory: rank0 日志 `reserved_linear_bytes=4.54 GB`、KV blocks=3414；MTP=3 可接受，但 nst=4/5 线性增加 linear reserve，应谨慎用于更高并发。

优先优化假设：
1. 先复查/恢复 PR #1536 的 MTP Transpose 消除逻辑。当前源码仍可见 `conv_weight.transpose(0, 1).contiguous()`、`mixed_qkv.transpose(1, 2)`、`run_spec_verify_conv()` 内部 round-trip transpose，profiling 也显示 Transpose 是 MTP 专属大头。
2. 其次做 MTP accept/verify 逻辑融合，目标是减少 `Range/Pack/Concat/Softmax/ArgMax/Cumsum` 等小 op 和 host sync。
3. 再评估 TP=4 AllReduce/AllGather overlap 或批量化通信，重点看 decode 阶段 `allreduceAicpuKernel` + HCCL allreduce/allgather 的高占比。

### 2026-05-25 MTP=3 Transpose 消除验证

代码：在 `qwen3_gated_delta_net_base.{cpp,h}` 恢复 MTP spec-verify Transpose 消除逻辑：
- `run_spec_verify_conv()` 改为接收/返回 `[B,T,C]`，避免内部 `transpose(1,2)` round-trip。
- spec-verify 分支缓存 `conv_weight_transposed_`，避免每步重复 `conv_weight.transpose(0,1).contiguous()`。
- `process_mixed_qkv()` 根据输入布局决定是否 transpose。

构建补充：当前 CANN op_api 的 `aclnnBeamSearchGroupGetWorkspaceSize()` 多了 `topK` 参数，需在 `beam_search_rec.cpp` 传 `top_tokens.size(-1)` 才能完整链接。

Benchmark 对比（TP=4, Phy 8-11, random 20k/1k, `parallel=1`, `number=5`, chunk prefill, MTP=3）：

| 版本 | Avg Latency(s) | TTFT(ms) | TPOT(ms) | Output TPS | Decoded/Iter | Accept |
|------|----------------|----------|----------|------------|--------------|--------|
| before | 14.564 | 2524.8 | 12.05 | 66.82 | 3.21 | 68.8% |
| transpose-opt | 14.03 | 2471.5 | 11.57 | 69.31 | 3.13 | 68.0% |

Decode-focused profiling 对比（32 input / 200 output, rank0）：

| 指标 | before | transpose-opt | 变化 |
|------|--------|----------------|------|
| Transpose time | 505.08 ms | 205.86 ms | -59.2% |
| Transpose calls | 36,701 | 12,604 | -65.7% |
| Device total | 5310.22 ms | 5005.63 ms | -5.7% |
| Decode TPOT | 10.62 ms | 9.90 ms | -6.8% |
| Output TPS | 66.25 | 69.70 | +5.2% |

精度冒烟：GSM8K `limit=10`，`mean_acc=1.0`，evalscope RC=0，无服务异常。

Artifact:
- Benchmark: `/home/g00510989/xllm/runs/20260525_mtp3_transpose_opt/benchmark/random_20k_1k_parallel_1_number_5/evalscope.log`
- Accuracy: `/home/g00510989/xllm/runs/20260525_mtp3_transpose_opt/accuracy/gsm8k_limit10/reports/Qwen35-27B/gsm8k.json`
- Profiling: `/home/g00510989/xllm/runs/20260525_mtp3_transpose_opt/profiling/mtp3_transpose_decode32_out200_cards8_11_20260525_005812/PROF_000001_20260525005815694_BQNAKJOMRDIQFFAA`

### 2026-05-25 MTP accept mask 小算子反例

目标：优化 `RejectionSampler::build_accepted_mask()`，尝试减少 MTP accept/verify 路径中的 `Range/ArgMax` 小算子。

验证环境：TP=4, Phy 8-11, random 20k/1k, `parallel=1`, `number=5`, chunk prefill, MTP=3。基线为上述 transpose-opt。

| 版本 | 实现 | Avg Latency(s) | TTFT(ms) | TPOT(ms) | Output TPS | Decoded/Iter | Accept |
|------|------|----------------|----------|----------|------------|--------------|--------|
| transpose-opt baseline | 原始 `argmax + arange + <=` | 14.03 | 2471.5 | 11.57 | 69.31 | 3.13 | 68.0% |
| cumprod mask | `accepted.to(int32).cumprod(dim=1).to(bool) + cat` | 14.12 | 2325.3 | 11.81 | 68.87 | 2.98 | 66.4% |
| bool-prefix mask | 短 `n_spec` 上 slice + `logical_and` 前缀 + cat | 14.16 | 2458.4 | 11.71 | 68.69 | 3.00 | 66.6% |

结论：
- 这两个 Torch-level mask 改写都未超过 transpose-opt baseline，不能作为有效优化提交。
- `cumprod` 虽减少 `Range/ArgMax`，但引入更重的前缀扫描；`bool-prefix` 避免 dtype 转换和扫描，但多个 slice/logical_and/cat 仍不足以抵消开销。
- 后续若继续做 accept/verify 小算子，应走真正的融合 kernel 或复用现有 `kernel::rejection_sample` 路径，并同时处理 selected-only draft probs；不要再在 eager Torch 层局部替换 `build_accepted_mask()`。

Artifact:
- Cumprod benchmark: `/home/g00510989/xllm/runs/20260525_mtp3_smallop/perf/random20k_1k_p1_n5/20260525_012954/Qwen35-27B/performance_summary.txt`
- Bool-prefix benchmark: `/home/g00510989/xllm/runs/20260525_mtp3_smallop_boolprefix/perf/random20k_1k_p1_n5/20260525_013755/Qwen35-27B/performance_summary.txt`
- 单测：两版均通过 `sampler_test`（11 passed, 2 MLU-only skipped）。

### 2026-05-25 MTP draft extend 提前准备 P0/P0b

目标：参考 vLLM async schedule，让 target 模型 device 计算期间尽量推进 draft 模型下发前置工作，减少 draft extend 空泡。

实现分层：
- P0 negative：在 target validate async 期间枚举所有可能 `last_idx` 的 draft extend 候选元数据。该方法等价但 CPU 开销过大，拖慢 decode 临界路径。
- P0b effective：只提前不依赖 target 输出的轻量工作，包括 `base_input.to(device_, dtype_)`、decode CPU view、原始 input token 缓存；target 返回后只为真实 accepted prefix 构造一条 draft extend 输入。

Benchmark 对比（TP=4, Phy 8-11, random 20k/1k, `parallel=1`, `number=5`, chunk prefill, MTP=3）：

| 版本 | Avg Latency(s) | TTFT(ms) | TPOT(ms) | Output TPS | Decoded/Iter | Accept |
|------|----------------|----------|----------|------------|--------------|--------|
| transpose-opt baseline | 14.03 | 2471.5 | 11.57 | 69.31 | 3.13 | 68.0% |
| P0 enumerate candidates | 14.81 | 2358.6 | 12.46 | 65.75 | 2.90 | 65.5% |
| P0b light prepare | 13.85 | 2466.0 | 11.39 | 70.18 | 3.09 | 67.6% |

精度冒烟：P0b GSM8K `limit=10`，`mean_acc=1.0`，evalscope RC=0。

经验规则：
- MTP 提前下发/提前准备必须先画出“依赖 target logits/embedding/accepted prefix”的边界；任何依赖这些结果的工作不能在当前等价实现中提前执行。
- 不要枚举 speculative future 候选来换重叠，尤其是 `num_speculative_tokens + 1` 个候选乘 batch 的 CPU 构图/向量拼接，会直接吃掉 decode 周期。
- 可以优先提前 `ForwardInput` 复制、CPU view、固定 layout 信息等确定性准备；收益通常小但风险低。
- 真正的 vLLM-style async draft dispatch 需要 shadow/pending draft KV 或双缓冲，并提供 validate 后的 commit/rollback 机制，否则会破坏 draft KV 与 target accepted prefix 的一致性。

Artifact:
- P0 negative benchmark: `/home/g00510989/xllm/runs/20260525_mtp3_async_draft_p0/perf/random20k_1k_p1_n5/20260525_071955/Qwen35-27B/performance_summary.txt`
- P0b benchmark: `/home/g00510989/xllm/runs/20260525_mtp3_async_draft_p0b/perf/random20k_1k_p1_n5/20260525_072650/Qwen35-27B/performance_summary.txt`
- P0b accuracy: `/home/g00510989/xllm/runs/20260525_mtp3_async_draft_p0b/accuracy/gsm8k_limit10/reports/Qwen35-27B/gsm8k.json`

## 输出契约

返回：
- trace 路径或生成的 profiling 路径
- 框架（xLLM / vLLM-Ascend）
- 模型/服务器参数
- 五表报告（kernel/overlap/fuse/下发/内存）
- 可选相似性标注（high/medium/low）
- Prefill/Decode 各阶段的主要瓶颈总结
- 重叠证据来源说明（单 trace / 两 trace）
