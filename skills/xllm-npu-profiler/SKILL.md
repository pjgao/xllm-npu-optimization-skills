---
name: xllm-npu-profiler
description: 昇腾 NPU 910B3 (A3) 上的 xLLM 推理 Profiling 分析。提供五表报告（kernel/重叠/融合/下发效率/内存效率），支持 xLLM 和 vLLM-Ascend 对比分析，Prefill/Decode 阶段分离。当用户需要定位 NPU 推理性能瓶颈时使用。
---

# xLLM NPU Profiling 分析

## 概述

在华为昇腾 NPU 910B3 (A3) 上进行推理 Profiling 分析，输出五表报告。

工作流入口：
- [scripts/run_profiling.sh](scripts/run_profiling.sh) — xLLM dynamic profiling 采集
- [scripts/multibatch_test.py](scripts/multibatch_test.py) — profiling 采集 workload 生成
- [scripts/analyze_xllm_npu_profile.py](scripts/analyze_xllm_npu_profile.py) — 已导出 PROF 数据解析
- [scripts/render_triage_npu.py](scripts/render_triage_npu.py) — 五表 Markdown 渲染

## 能力矩阵

| 能力 | xLLM | vLLM-Ascend |
|------|------|-------------|
| 已有 trace 分析 | yes | yes |
| 实时采集 | yes（`PROFILING_MODE=dynamic` + msprof attach） | yes（profiler 配置） |
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

### 2. xLLM 实时采集（当前推荐）

当前推荐使用 Ascend `msprof --dynamic=on` attach 到已启动的 xLLM 父进程，
并通过 FIFO 控制采集窗口。该方式能把 warmup 和正式采集分开，避免把模型加载、
ACL graph warmup、首轮编译等噪声混入 trace。

前置要求：
- xLLM 启动脚本中必须设置：
  ```bash
  export PROFILING_MODE=dynamic
  ```
- 采集脚本传入的 PID 必须是 xLLM 父进程 PID，可通过 `ps -ef | grep xllm`
  查找；不要使用 `npu-smi info` 中看到的 device worker PID。
- 采集端口、模型名、tokenizer 路径必须与 xLLM 服务启动配置一致。

脚本契约：
- 若使用本 skill 的 `scripts/run_profiling.sh` + `scripts/multibatch_test.py`
  组合采集，必须先在 xLLM 启动脚本中加入 `export PROFILING_MODE=dynamic`，
  再启动服务。
- `PROFILING_MODE=dynamic` 属于服务启动侧配置；`run_profiling.sh` 只负责
  `msprof --dynamic=on --pid` attach、warmup、写入 `start/stop` 控制采集窗口、
  以及 `msprof --export=on` 导出结果。
- 如果启动脚本未设置 `PROFILING_MODE=dynamic`，后续即使 `msprof` attach 成功，
  也可能无法得到预期的动态采集窗口或完整 profiling 数据。

典型采集流程：

```bash
# 1. 启动 xLLM 服务前开启 dynamic profiling mode
export PROFILING_MODE=dynamic
xllm ... --port 38050 ...

# 2. 找到 xLLM 父进程 PID
ps -ef | grep xllm

# 3. 使用 msprof dynamic attach 采集
scripts/run_profiling.sh <xllm_parent_pid> ./xllm_profile full
```

`run_profiling.sh` 的核心采集方式：

```bash
msprof \
  --dynamic=on \
  --output="$OUTPUT_DIR" \
  --model-execution=on \
  --runtime-api=on \
  --aicpu=on \
  --pid="$XLLM_PID" < "$PIPE_FILE" &

echo "start" >&3
python3 multibatch_test.py ...
echo "stop" >&3
msprof --export=on --output="$LATEST_PROF"
```

推荐参数语义：
- `mode=full`：先 warmup，再打开 profiling 采正式请求。
- `mode=warmup`：只预热，不采集。
- `mode=test`：只采正式请求，适合服务已预热后的补采。
- `BATCH_SIZE`：每批并发请求数，用于模拟并发用户。
- `NUM_BATCHES`：正式采集批次数。
- `WARMUP_BATCHES`：不采集 profiling 的预热批次数。
- `INPUT_TOKENS` / `OUTPUT_TOKENS`：用 tokenizer 生成接近目标 token 数的请求。

采集完成后分析导出的 `mindstudio_profiler_output/`：

```bash
python scripts/analyze_xllm_npu_profile.py \
  --input /path/to/xllm_profile_YYYYMMDD_HHMMSS/PROF_xxx \
  --framework xllm
```

### 3. vLLM-Ascend 实时采集

vLLM-Ascend 侧可用其 profiler 配置生成 trace；本 skill 的分析脚本仍只消费
已经导出的 profiling 目录。

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve /path/to/model \
  --tensor-parallel-size 4 \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/vllm-profile"}'

python scripts/analyze_xllm_npu_profile.py \
  --input /tmp/vllm-profile/PROF_xxx \
  --framework vllm-ascend \
  --output /tmp/vllm-profile-analysis.json
```

### 4. 两阶段 trace 分析（eager/graph-on 对比）

```bash
python scripts/analyze_xllm_npu_profile.py \
  --input /path/to/eager_profile/PROF_xxx \
  --framework xllm \
  --output /tmp/eager-profile-analysis.json

python scripts/analyze_xllm_npu_profile.py \
  --input /path/to/graph_on_profile/PROF_xxx \
  --framework xllm \
  --output /tmp/graph-profile-analysis.json
```

mapping trace 用于恢复 `kernel → cpu_op → python scope` 映射。
formal trace 用于实际性能分析。

### 5. 阶段分离采集

`analyze_xllm_npu_profile.py` 只分析已经导出的 `PROF_*` 目录，不负责发送请求
或启动采集。Prefill/Decode 阶段分离时，使用 `run_profiling.sh` 改不同 workload
参数后分别采集，再分别离线分析。

```bash
# Prefill-focused: 长输入、短输出
INPUT_TOKENS=4090 OUTPUT_TOKENS=1 \
  scripts/run_profiling.sh <xllm_parent_pid> /tmp/xllm-prefill full

python scripts/analyze_xllm_npu_profile.py \
  --input /tmp/xllm-prefill_YYYYMMDD_HHMMSS/PROF_xxx \
  --framework xllm \
  --output /tmp/xllm-prefill-analysis.json

# Decode-focused: 短输入、长输出
INPUT_TOKENS=1 OUTPUT_TOKENS=2048 \
  scripts/run_profiling.sh <xllm_parent_pid> /tmp/xllm-decode full

python scripts/analyze_xllm_npu_profile.py \
  --input /tmp/xllm-decode_YYYYMMDD_HHMMSS/PROF_xxx \
  --framework xllm \
  --output /tmp/xllm-decode-analysis.json
```

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
4. 优先手工分离 Prefill/Decode workload 采集，二者瓶颈差异显著
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
1. 先做非 MTP decode 与 MTP spec-verify 的路径级 diff，再决定是否改局部 transpose。重点检查两条路径是否复用同一个 causal conv wrapper、同一套 weight 预布局、同一套输入/输出 layout contract。
2. 其次做 MTP accept/verify 逻辑融合，目标是减少 `Range/Pack/Concat/Softmax/ArgMax/Cumsum` 等小 op 和 host sync。
3. 再评估 TP=4 AllReduce/AllGather overlap 或批量化通信，重点看 decode 阶段 `allreduceAicpuKernel` + HCCL allreduce/allgather 的高占比。

MTP transpose 专项强制检查：
- 如果 `Transpose` 主要出现在 MTP/speculative 路径，不能先直接修改 `transpose()` 调用点；必须先对比非 MTP decode 路径。
- 已知风险点：非 MTP decode 可能走 `causal_conv1d`，并在权重加载/算子封装侧提前完成 weight reshape/预布局；MTP spec-verify 可能走 `causal_conv1d_update`，手工构造 `CausalConv1dUpdateParams`，从而重新引入 input/state/weight layout 适配和 transpose。
- 对 main、preview 或任何新分支，如果看到 `conv_weight.transpose(0,1).contiguous()`、`mixed_qkv.transpose(1,2)`、`run_spec_verify_conv()` round-trip，只能先标记为“局部症状”；优先尝试让 MTP 复用非 MTP 的 `causal_conv1d` 路径，或新增等价 fused spec-verify causal conv。
- 报告结论必须区分“局部减损”和“结构性修复”。缓存 weight、删 round-trip transpose 属于局部减损；复用 `causal_conv1d` 或等价 fused 算子才是结构性修复。

### 2026-05-25 MTP=3 Transpose 消除验证

代码：在 `qwen3_gated_delta_net_base.{cpp,h}` 恢复 MTP spec-verify Transpose 消除逻辑：
- `run_spec_verify_conv()` 改为接收/返回 `[B,T,C]`，避免内部 `transpose(1,2)` round-trip。
- spec-verify 分支缓存 `conv_weight_transposed_`，避免每步重复 `conv_weight.transpose(0,1).contiguous()`。
- `process_mixed_qkv()` 根据输入布局决定是否 transpose。

复盘后的更准确根因：非 MTP decode 路径已经走 `causal_conv1d`，并复用了权重预 reshape/预布局路径，所以不会在每个 decode step 重复做 weight/layout transpose；MTP spec-verify 路径没有复用 `causal_conv1d`，而是手工构造 `CausalConv1dUpdateParams` 调用 `causal_conv1d_update`，因此重新引入了输入输出和 weight layout 适配。当前 transpose-opt 是有效减损，但不是最终结构性修复；下一版应优先尝试让 MTP 复用 `causal_conv1d` 路径或新增等价 fused spec-verify causal conv。

2026-05-27 复用路径 prototype 经验：
- 优先复用非 MTP decode 的 `causal_conv1d`，而不是继续局部修补 `causal_conv1d_update` 的 transpose。实现上让 MTP spec-verify 输入/输出保持 `[B,T,C]`，直接使用 `load_common_state_dict()` 中已经预布局的 conv weight。
- `aclnnCausalConv1d` 的 `query_start_loc/cache_indices/num_accepted_tokens` 是 host `IntArrayRef`，因此 MTP accepted-prefix 需要在 `ModelInputParams` 中保留 host vector；ACL graph capture params 也要同步做 padding。
- 重要风险：host `IntArrayRef` 在 ACL graph replay 中可能被固化为 capture 属性，而旧 `causal_conv1d_update` 的 accepted-prefix 是 device tensor，可以 replay 动态更新。graph on/off 必须分别做 10 条精度验证；如果 graph replay 下 accepted-prefix 不能动态变化，应改为 fused/tensor 参数算子或保守 fallback。
- 构建经验：`python setup.py build` 先做 submodule commit 校验；如果 `third_party/tilelang-ascend` 或 `third_party/torch_npu_ops` checkout 与主仓记录不一致，会在编译前失败。直接 Ninja 重链可能暴露 `ops_api.cpp` 与 `torch_npu_ops` 接口不匹配，不能用这种状态做性能结论。

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
