# Qwen3.5-27B NPU 优化实战指南

> 目标：在昇腾 910B3 (A3) x4 上，将 xLLM 运行 Qwen3.5-27B 推理的性能做到最优
> 竞品对照：vLLM-Ascend
> 工具链：xllm-npu-optimization-skills 仓库

---

## 目录

- [1. 整体思路](#1-整体思路)
- [2. Phase 0 — 环境准备与初始化](#2-phase-0--环境准备与初始化)
- [3. Phase 0.5 — 查询模型 PR 历史](#3-phase-05--查询模型-pr-历史)
- [4. Phase 1 — 公平基准测试](#4-phase-1--公平基准测试)
- [5. Phase 2 — 差异判定](#5-phase-2--差异判定)
- [6. Phase 3 — Profiling 五表分析](#6-phase-3--profiling-五表分析)
- [7. Phase 4 — 构建优化 Plan](#7-phase-4--构建优化-plan)
- [8. Phase 5 — RLCR 迭代优化](#8-phase-5--rlcr-迭代优化)
- [9. 日常运营：代码审查 + 事故诊断](#9-日常运营代码审查--事故诊断)
- [10. 内核级优化：kernel-pilot](#10-内核级优化kernel-pilot)
- [11. 完整目录结构参考](#11-完整目录结构参考)
- [12. 时间线概览](#12-时间线概览)

---

## 1. 整体思路

整个优化流程是一个**证据驱动的闭环系统**，核心原则：

```
不公平数据不比较 → 没 profiling 不动手 → 每步决策有数据支撑
```

流程总览：

```
┌─────────────────────────────────────────────────────────────────────┐
│                     xllm-npu-sota-loop                              │
│                                                                     │
│  Phase 0     环境准备（NPU 健康、框架可用、模型就绪）                  │
│      ↓                                                              │
│  Phase 0.5   查询 Qwen3.5 的 PR 历史，了解已知优化                   │
│      ↓                                                              │
│  Phase 1     xllm-npu-benchmark: xLLM vs vLLM-Ascend 公平对比       │
│      ↓                                                              │
│  Phase 2     差异判定：gap > 1% ?                                    │
│      ├─ No  → 写报告，结束                                          │
│      └─ Yes ↓                                                       │
│  Phase 3     xllm-npu-profiler: 五表报告定位瓶颈                     │
│      ↓                                                              │
│  Phase 4     构建优化 Plan（基于五表报告确定优先级）                   │
│      ↓                                                              │
│  Phase 5     RLCR 迭代循环                                          │
│      ├─ Research → Learn → Code → Review → Validate → Record        │
│      └─ 回到 Phase 2 重新判定                                       │
│                                                                     │
│  随时可用:                                                            │
│  - xllm-npu-code-review: 每轮 patch 前的 NPU 特化审查                │
│  - xllm-npu-incident-triage: 遇到运行异常时诊断                      │
│  - kernel-pilot: 热点 kernel 需要自研算子时                          │
└─────────────────────────────────────────────────────────────────────┘
```

**为什么这样设计**：

| 如果不这样做 | 可能的问题 |
|-------------|-----------|
| 不做公平基准 | 拿 tuned xLLM 与 vLLM-Ascend defaults 比，结果不具参考性 |
| 不做 profiling | 盲目优化，改了一堆代码但没打中瓶颈 |
| 不做 RLCR 闭环 | patch 写了但忘了验证，引入回归 |
| 不查 PR 历史 | 重复做已经被证明无效/有效的工作 |

---

## 2. Phase 0 — 环境准备与初始化

### 2.1 确认 NPU 环境

```bash
# 检查 NPU 状态（应看到 4 张 A3 卡）
npu-smi info

# 确认驱动版本
npu-smi info -t board -i 0
# 期望: HDK Version >= 25.2.0

# 确认 CANN 版本
cat /usr/local/Ascend/driver/version.info
# 期望: CANN >= 8.0.RC1
```

### 2.2 确认框架可用

```bash
# xLLM
cd /home/gaopengju/projects/xllm
git log --oneline -1  # 记录 commit hash
pip install -e .      # 确保已安装
xllm serve --help     # 确认 CLI 可用

# vLLM-Ascend
cd /home/gaopengju/projects/vllm-ascend
git log --oneline -1  # 记录 commit hash
# 确认 vllm serve 可用
vllm serve --help
```

### 2.3 确认模型权重

```bash
ls /models/Qwen3.5-27B/
# 应包含: config.json, tokenizer.json, model-*.safetensors 等
```

### 2.4 创建运行目录

```bash
export RUN_ROOT=/home/gaopengju/runs/$(date +%Y%m%d)_qwen35_27b_npu_sota

mkdir -p $RUN_ROOT/{benchmark,profiles,analysis,history,kernel,patches,humanize}

# 记录环境信息
cat > $RUN_ROOT/manifest.md << 'EOF'
# Qwen3.5-27B NPU SOTA 优化 Run

- **模型**: /models/Qwen3.5-27B (Dense, 27B 参数)
- **精度**: bf16
- **NPU**: A3 x4
- **CANN**: 8.0.RC1 (确认实际版本)
- **HDK Driver**: 25.2.0 (确认实际版本)
- **xLLM commit**: <从 git log 获取>
- **vLLM-Ascend commit**: <从 git log 获取>
- **开始时间**: <date>
EOF
```

### 2.5 Qwen3.5-27B 的硬件适配判断

Qwen3.5-27B 是一个 Dense 模型（约 27B 参数），bf16 下模型权重约 54GB：

| 项目 | A3 x4 适配 |
|------|-----------|
| 模型权重 | 54GB，4 卡各 ~14GB，可装下 |
| KV Cache | 剩余显存给 KV Cache |
| TP 配置 | TP=4（Dense 模型无需 EP） |
| block_size | 128（A3 推荐值） |
| max_model_len | 根据业务需求，默认 8192 |

---

## 3. Phase 0.5 — 查询模型 PR 历史

在开始优化前，先看 xLLM 历史上对 Qwen 系列做过哪些优化，避免重复劳动。

### 3.1 使用 model-pr-optimization-history

```bash
# 查询 Qwen 系列相关 PR
python /home/gaopengju/projects/xllm-npu-optimization-skills/model-pr-optimization-history/scripts/query.py \
  --model "Qwen" \
  --type optimization

# 查询最近 3 个月的 PR
python ../xllm-npu-optimization-skills/model-pr-optimization-history/scripts/query.py \
  --model "Qwen" \
  --since "2026-02-01"
```

### 3.2 查阅已有档案

```bash
cat /home/gaopengju/projects/xllm-npu-optimization-skills/model-pr-optimization-history/xllm/qwen3-core.md
```

关注点：
- 已有的融合算子替换（SwiGLU、RMSNorm 等）
- Chunked prefill 支持的 commit
- 图模式适配的状态
- 已知的 bug 和 workaround

### 3.3 记录到运行目录

```bash
cat > $RUN_ROOT/history/xllm-model-history-notes.md << 'EOF'
# Qwen3.5-27B PR 历史笔记

## 已知优化（来自 PR 历史）
- chunked prefill: 已支持 (具体 commit)
- adaptive graph mode: Prefill eager + Decode npugraph_ex (已支持)
- SwiGLU 融合: 已实现
- PagedAttention block_size=128: 默认配置

## Qwen3.5 vs Qwen3 架构差异
- （待确认：是否有新的注意力机制、激活函数变化等）

## 需要额外验证
- Qwen3.5 可能引入了新的层归一化方式
- 确认 tokenizer 兼容性
EOF
```

---

## 4. Phase 1 — 公平基准测试

**核心原则**：xLLM 和 vLLM-Ascend 各自独立搜索最优配置，最终比较双方 best 结果。

### 4.1 Preflight

```bash
# 验证 xLLM 能启动模型
python /home/gaopengju/projects/xllm-npu-optimization-skills/skills/xllm-npu-benchmark/scripts/validate_framework_cli.py \
  --framework xllm \
  --model /models/Qwen3.5-27B \
  --extra-flags "--tensor-parallel-size 4"

# 验证 vLLM-Ascend 能启动模型
python /home/gaopengju/projects/xllm-npu-optimization-skills/skills/xllm-npu-benchmark/scripts/validate_framework_cli.py \
  --framework vllm-ascend \
  --model /models/Qwen3.5-27B \
  --extra-flags "--tensor-parallel-size 4 --enforce-eager"
```

### 4.2 准备基准数据集

**推荐方案**: 使用 evalscope `line_by_line` plugin + 真实流量 JSONL。

**数据集格式** (每行一个完整 OpenAI 请求 body):

```jsonl
{"model": "Qwen35-27B", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 2048, "temperature": 0.0, "stream": true}
{"model": "Qwen35-27B", "messages": [{"role": "system", "content": "你是助手"}, {"role": "user", "content": "总结..."}], "max_tokens": 2048, "temperature": 0.0, "stream": true}
```

**实测数据集**: `jd_openai_20k.jsonl` (20k 真实京东对话请求，平均 input ~19860 tokens)。

SLA 约束：

| 指标 | 阈值 | 说明 |
|------|------|------|
| max_ttft_ms | 5000ms | Time To First Token (长序列 prefill 较慢，放宽) |
| max_tpot_ms | 50ms | Time Per Output Token |

### 4.3 xLLM 配置搜索

**基础命令** (实测, 910B3 x2 TP=2):

```bash
# 实测启动命令 (Phy 8, NPU 14)
/home/g00510989/xllm/xllm/build/xllm/core/server/xllm \
  --model /home/data/weights/Qwen35-27B \
  --tensor-parallel-size 2 \
  --enable_graph true \
  --block_size 128 \
  --max_model_len 32768 \
  --port 18160
```

**搜索空间**（Tier 2，最多 10 个候选）：

| 参数 | 候选值 | 说明 |
|------|--------|------|
| chunked-prefill-size | 256, 512, 1024 | 分块 prefill 大小 |
| max-num-seqs | 64, 128, 256 | 最大并发序列 |
| gpu-memory-utilization | 0.7, 0.85, 0.9 | 显存利用率 (长序列需更多 KV Cache) |
| graph-mode | npugraph_ex, ge | 图模式选择 |
| enable_schedule_overlap | true, false | 调度重叠 |
| communication_backend | lccl, hccl | 通信后端 |

**实测 baseline 结果** (2026-05-23):

| 并发 | 请求数 | Output Throughput | TTFT   | TPOT  | Avg Output Tokens | 成功 |
|----|------|-------------------|--------|-------|-------------------|------|
| 1  | 5    | 29.33 tok/s       | 3564ms | 29.8ms| 917               | 5/5  |
| 2  | 4    | 46.83 tok/s       | 4445ms | 34.4ms| 1308              | 4/4  |

结论：并发 2 吞吐 +59.7%。TPOT < 50ms SLA 满足，但 TTFT ~4s 较长 (长序列 prefill 慢)。

### 4.4 vLLM-Ascend 配置搜索

**基础命令**：

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve /models/Qwen3.5-27B \
  --tensor-parallel-size 4 \
  --enforce-eager \
  --block-size 128 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --port 8000 &
```

**搜索空间**：

| 参数 | 候选值 |
|------|--------|
| enforce-eager | true, false (试试 graph) |
| enable-prefix-caching | true, false |
| gpu-memory-utilization | 0.85, 0.9, 0.95 |
| max-num-seqs | 64, 128 |

### 4.5 运行 benchmark (evalscope)

```bash
# 使用封装的 bench.sh 脚本
./scripts/bench.sh baseline 1 5    # 并发 1, 5 请求
./scripts/bench.sh baseline 2 4    # 并发 2, 4 请求
./scripts/bench.sh mtp 1 5         # MTP 模式 (需要先启动 MTP 服务)

# 结果输出到:
# $RUN_ROOT/benchmark/baseline/parallel_1_number_5/
# $RUN_ROOT/benchmark/baseline/parallel_2_number_4/
# $RUN_ROOT/benchmark/mtp/parallel_1_number_5/
```

**核心指标**:
- `benchmark_summary.json` — 汇总 (Output Throughput, TTFT, TPOT, etc.)
- `benchmark_percentile.json` — 延迟百分位
- `perf_report.html` — HTML 报告

---

### 4.6 MTP (Multi-Token Prediction) 模式

Qwen3.5-27B 内置 MTP 架构 (1 层 draft model)，启用命令：

```bash
./scripts/mtp.sh start     # 启动 MTP 服务 (port 18170)
./scripts/mtp.sh stop      # 停止
./scripts/mtp.sh status    # 状态
./scripts/mtp.sh log 0     # 查看 node_0 日志
./bench.sh mtp 1 5         # MTP benchmark
```

**关键参数**:
- `--draft_model /home/data/weights/Qwen35-27B-mtp` — MTP draft 权重
- `--draft_devices npu:N` — 每个节点的 draft device 必须匹配主 device
- `--num_speculative_tokens 2` — 投机 token 数
- `--max_concurrent_requests 30`

**已知问题 (2026-05-23)**:
MTP 模式下 node_1 推理时崩溃，`npu_add_rms_norm` shape 不匹配错误 (error code 561002)。
可能是 torch_npu 2.7.1.post2 与 xLLM MTP 实现不兼容。
已向 xLLM 团队上报，等待修复。详见 `xllm-npu-incident-triage` skill。

---

## 5. Phase 2 — 差异判定

读取 `summary.md`，计算 gap：

```python
# Chat 场景
xllm_throughput = 32500
vllm_throughput = 28100
gap = (vllm_throughput - xllm_throughput) / vllm_throughput * 100
# gap = -15.7% → xLLM 已领先，进入 Phase 2 的 "xLLM 胜出" 分支

# Summary 场景（慢场景，重点关注）
xllm_throughput = 8200
vllm_throughput = 7500
gap = (vllm_throughput - xllm_throughput) / vllm_throughput * 100
# gap = -9.3% → xLLM 已领先
```

**判定逻辑**：

| 结果 | 行为 |
|------|------|
| xLLM 领先 > 1% | 确认结果稳定后写报告，结束（或继续优化以扩大优势） |
| gap 在 ±1% | 平局，可结束或继续 |
| xLLM 落后 > 1% | **进入 Phase 3**，定位瓶颈 |

> 假设 vLLM-Ascend 在某个配置下超过了 xLLM（比如 summary 场景 xLLM=7200 vs vLLM=7800），则 gap = 7.1%，需要深入分析。

---

## 6. Phase 3 — Profiling 五表分析

**关键规则**：此报告不存在前不得开始任何 patch。

### 6.1 采集 Profiling 数据

使用 benchmark 中 winning-commands 的配置：

```bash
# xLLM Profiling
XLLM_PROFILING=1 xllm serve /models/Qwen3.5-27B \
  --tensor-parallel-size 4 \
  --graph-mode npugraph_ex \
  --block-size 128 \
  --max-num-seqs 128 \
  --chunked-prefill-size 512 \
  --port 8080

# 使用阶段分离采集（推荐）
python /home/gaopengju/projects/xllm-npu-optimization-skills/skills/xllm-npu-profiler/scripts/analyze_xllm_npu_profile.py \
  --framework xllm \
  --url http://127.0.0.1:8080 \
  --output-dir $RUN_ROOT/profiles/xllm/ \
  --num-steps 5 \
  --profile-by-stage
```

```bash
# vLLM-Ascend Profiling
VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve /models/Qwen3.5-27B \
  --tensor-parallel-size 4 \
  --enforce-eager \
  --block-size 128 \
  --gpu-memory-utilization 0.9 \
  --port 8000 \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/vllm-profile"}'

python /home/gaopengju/projects/xllm-npu-optimization-skills/skills/xllm-npu-profiler/scripts/analyze_xllm_npu_profile.py \
  --framework vllm-ascend \
  --url http://127.0.0.1:8000 \
  --output-dir $RUN_ROOT/profiles/vllm/ \
  --num-steps 5
```

**重要**：使用慢场景的实际 input/output 长度（summary: input=8000, output=1000），不要用短输入。

### 6.2 生成五表报告

```bash
python /home/gaopengju/projects/xllm-npu-optimization-skills/skills/xllm-npu-profiler/scripts/render_triage_npu.py \
  --analysis-root $RUN_ROOT/profiles/ \
  --output $RUN_ROOT/analysis/root-cause.md
```

### 6.3 解读五表

#### 表 1: Kernel Table（热点算子排名）

```markdown
| Rank | Kernel | Op Type | AICore Time % | Count | Stage |
|------|--------|---------|---------------|-------|-------|
| 1 | MatMul_decode | MatMul | 35.2% | 16000 | Decode |
| 2 | FlashAttention_prefill | Attention | 22.1% | 2000 | Prefill |
| 3 | AllReduce_L4 | AllReduce | 12.8% | 16000 | Decode |
| 4 | Add+RMSNorm | Normalization | 5.3% | 32000 | Both |
| 5 | SwiGLU | Activation | 4.1% | 16000 | Decode |
```

分析要点：
- MatMul 占 35%：检查 shape 是否充分利用 A3 CUBE 单元
- AllReduce 占 12.8%：TP=4 通信开销，是否有重叠机会
- 已融合的 Add+RMSNorm、SwiGLU 说明 P0 融合算子路径已生效

#### 表 2: Overlap-Opportunity Table（重叠机会）

```markdown
| 机会 | 计算侧 | 通信侧 | 当前重叠率 | 潜力 |
|------|--------|--------|-----------|------|
| TP AllReduce + Next Layer | FFN MatMul | AllReduce | 15% | 可提升至 60% |
| Compute + MTE3 | AICore | Data Transfer | 45% | UBB 瓶颈 |
```

**关键发现**：TP AllReduce 与下一层计算的重叠率仅 15%，这是主要优化空间。

#### 表 3: Fuse-Pattern Table（融合模式）

```markdown
| 模式 | 状态 | 说明 |
|------|------|------|
| FlashAttention | Fused (torch_npu.npu_fusion_attention) | OK |
| SwiGLU | Fused (torch_npu.npu_swiglu) | OK |
| RMSNorm+Add | Fused | OK |
| MatMul+BiasAdd | NOT fused | 可能优化机会 |
```

#### 表 4: 下发效率表

```markdown
| 指标 | Prefill | Decode |
|------|---------|--------|
| AICore 利用率 | 92% | 78% |
| Idle 率 | 3% | 18% |
| Dispatch 延迟 (avg) | 2.1us | 4.5us |
```

**关键发现**：Decode 阶段 Idle 率 18%，接近 20% 的 Hostbound 阈值。可能是小算子下发效率问题。

#### 表 5: 内存效率表

```markdown
| 指标 | 值 |
|------|-----|
| xTensor 池使用率 | 82% |
| KV Cache 碎片率 | 8% |
| Block 分配失败次数 | 0 |
| Peak Memory | 58.2 GB / 64 GB |
```

### 6.4 综合诊断

```markdown
## Prefill 阶段瓶颈
- Computing bound (AICore 利用率 92%)
- 热点：MatMul (35%) + FlashAttention (22%)
- 优化方向：MatMul tiling 优化、计算-通信重叠

## Decode 阶段瓶颈
- 接近 Hostbound (Idle 率 18%)
- TP AllReduce 占比 12.8%，重叠率仅 15%
- Dispatch 延迟 4.5us（偏高）
- 优化方向：通信-计算重叠、算子下发效率
```

---

## 7. Phase 4 — 构建优化 Plan

基于五表报告，按昇腾优化路径优先级排列：

```markdown
## 优化 Plan: Qwen3.5-27B on A3 x4

### 根因总结
1. Decode 阶段 TP AllReduce 占 12.8%，与计算重叠仅 15%
2. Decode 阶段 Idle 率 18%，小算子下发效率需提升
3. MatMul+BiasAdd 融合尚未生效

### 优化队列

1. [P0] 通信-计算重叠优化
   - 目标：TP AllReduce 重叠率从 15% → 60%
   - 预计收益：Decode throughput +8~12%
   - 方案：将下一层的 QKV MatMul 提前，与 AllReduce 并行执行
   - 涉及文件：xllm/core/layers/npu_attention_layer.cpp
   - 风险：需要同步语义正确性验证

2. [P1] 算子下发效率：Dispatch 延迟优化
   - 目标：Dispatch 延迟从 4.5us → 2.5us
   - 预计收益：Decode Idle 率从 18% → 10%
   - 方案：合并小算子、增大 Batch 粒度
   - 涉及文件：xllm/core/scheduler/batch_scheduler.cpp
   - 风险：可能影响调度延迟

3. [P1] MatMul+BiasAdd 融合
   - 目标：融合 FFN 阶段的 MatMul+BiasAdd
   - 预计收益：减少 1 个 kernel launch，~2% 提升
   - 方案：使用 CANN CCE 的 MatMul+BiasAdd 融合算子
   - 涉及文件：xllm/core/layers/npu_ops.py
   - 风险：需验证数值精度

4. [P2] 权重预取 (npu_prefetch)
   - 目标：隐藏权重加载延迟
   - 预计收益：AICore 利用率 +2~3%
   - 方案：在每层计算时预取下一层权重
   - 涉及文件：xllm/core/layers/npu_linear.cpp
   - 风险：需验证 SRAM 空间足够
```

写入 `$RUN_ROOT/humanize/refined-plan.md`。

---

## 8. Phase 5 — RLCR 迭代优化

### 8.1 RLCR 流程

每轮迭代遵循：

```
Research → Learn → Code → Review → Validate → Record
```

#### Research: 确定 patch 方向

根据 Plan 第 1 项：优化 TP AllReduce 与计算的 overlap。

#### Learn: 学习已有方案

```bash
# 查询 xLLM 已有的通信重叠实现
grep -r "overlap\|all_reduce" xllm/core/layers/ --include="*.cpp"

# 查询 model-pr-history 中的相关优化
cat $RUN_ROOT/history/xllm-model-history-notes.md

# 查看 awesome-ascend-skills 的通信优化模式
# 参考 npu-overlap-catalog.md
```

#### Code: 编写 patch

```cpp
// xllm/core/layers/npu_attention_layer.cpp
// 将 AllReduce 提前到独立 stream，与下一层 QKV MatMul 重叠

// Before:
attn_output = attention(query, key, value);
all_reduce(attn_output);        // 阻塞等待
qkv = matmul(next_input, w);    // 等 AllReduce 完成

// After:
attn_output = attention(query, key, value);
all_reduce_async(attn_output);  // 非阻塞 AllReduce
qkv = matmul(next_input, w);    // 同时执行 MatMul
stream_sync();                   // 同步点
```

#### Review: NPU 特化代码审查

调用 `xllm-npu-code-review` skill：

```bash
# 审查 diff
git diff origin/main -- xllm/core/layers/npu_attention_layer.cpp
```

审查清单：
- [ ] AllReduce 异步实现是否正确（HCCL 调用规范）
- [ ] Stream sync 是否会死锁
- [ ] 图模式兼容性（npugraph_ex 是否支持这个 stream 操作）
- [ ] TP 场景下正确性是否保持
- [ ] 精度是否有影响

#### Validate: 编译 + 测试 + benchmark + profiling

```bash
# 1. 编译
cd /home/gaopengju/projects/xllm
python setup.py build --device npu

# 2. 单元测试
python setup.py test --device npu

# 3. 端到端精度测试
python test/test_xllm_serve_generation.py --model /models/Qwen3.5-27B --device npu

# 4. 重新运行 benchmark
# 使用之前的 winning-commands 配置

# 5. 重新采集 profiling
# 使用之前的命令
```

#### Record: 记录到台账

```bash
cat >> $RUN_ROOT/humanize/attempt-ledger.md << 'EOF'

## Attempt #1
- 日期：2026-05-23
- 方向：通信-计算重叠（TP AllReduce + QKV MatMul）
- 修改文件：xllm/core/layers/npu_attention_layer.cpp
- 结果：成功（Decode throughput +9.2%）
- benchmark 文件：benchmark/round_1_results.jsonl
- profiler 文件：analysis/round_1_root_cause.md
- 备注：重叠率从 15% → 55%，接近目标
EOF
```

### 8.2 回到 Phase 2

验证完 patch 后，重新跑 benchmark 对比：

- 如果 xLLM 仍然落后 > 1%：继续 RLCR（Plan 第 2 项）
- 如果 gap <= 1%：进入停止条件
- 如果 xLLM 已胜出：确认后写报告

### 8.3 多轮迭代示例

```
Attempt #1: 通信-计算重叠          → 成功 +9.2%
Attempt #2: Dispatch 延迟优化       → 成功 +3.5%（Idle 率 18%→11%）
Attempt #3: MatMul+BiasAdd 融合    → 失败（精度不对齐，回退）
Attempt #4: npu_prefetch 权重预取  → 成功 +2.1%（AICore 利用率 78%→83%）
Attempt #5: 投机解码 draft model  → 成功 +15%（仅适用于 output>256 场景）
```

每轮都严格走 Research→Learn→Code→Review→Validate→Record。

### 8.4 停止

当以下条件之一满足时停止：

| 条件 | 说明 |
|------|------|
| xLLM 胜出 | 双方 best 配置对比，xLLM 吞吐更高 |
| 平局 | gap <= 1% 且连续 3 轮稳定 |
| 硬件限制 | A3 不支持某关键特性 |
| 瓶颈已至 | AICore 利用率接近 100%，通信接近理论带宽 |
| 无进展 | 连续 3 轮尝试无改善 |

写最终报告：`$RUN_ROOT/benchmark/final_summary.md`

---

## 9. 日常运营：代码审查 + 事故诊断

### 9.1 代码审查

每次向 xLLM 提交 NPU 相关代码变更时，使用 `xllm-npu-code-review` skill：

```bash
# 在 xllm 仓库中
git diff origin/main
```

AI 助手会自动加载 skill，按 7 个维度审查：

1. **C++ 引擎代码**：内存安全、线程安全、RAII
2. **TileLang 算子**：Tiling 策略、UB 容量、精度
3. **AscendC 算子**：Buffer 管理、Double buffer、指令流水
4. **图模式兼容**：是否引入 Graph Break
5. **KV Cache**：PA block_table/slot_mapping 正确性
6. **通信代码**：HCCL 调用规范、同步语义
7. **精度**：FP16/BF16 混合精度、Softmax 数值稳定性

### 9.2 事故诊断

当 xLLM 在 A3 上出现异常时：

```bash
# Step 1: 证据保全
mkdir -p $RUN_ROOT/artifacts/$(date +%Y%m%d_%H%M%S)
npu-smi info > artifacts/npu_status.log
dmesg | tail -200 > artifacts/dmesg.log

# Step 2: 分类问题
# 看到 E39999 → AICore timeout → 算子实现问题
# 看到 E50000 → HCCL 超时 → 通信网络问题
# 看到 E82000 → OOM → 显存配置问题
```

常见场景速查：

| 症状 | 可能原因 | 处理方式 |
|------|---------|---------|
| 进程崩溃 + E39999 | AICore 算子超时 | 检查算子 shape/dtype 兼容性 |
| AllReduce 超时 + E50000 | 节点间 ROCE 问题 | 跑 hccl_test 诊断网络 |
| OOM + E82000 | KV Cache 分配失败 | 降低 gpu-memory-utilization |
| 输出 NaN | FP16 溢出或 KV Cache 对齐 | 检查 Softmax 数值稳定性 |
| 首次运行慢 | GE 编译 | warmup 或缓存编译结果 |

---

## 10. 内核级优化：kernel-pilot

当所有 P0/P1/P2 优化路径用尽，仍有性能 gap 时，启用 kernel-pilot。

### 10.1 准入条件

全部满足才启动：

1. xLLM 仍落后 vLLM-Ascend > 1%
2. 某 kernel 族占 AICore 时间 >= 1%
3. Profiler 五表显示该 kernel 是差异的合理解释
4. 有明确的正确性参考和代表性 shapes

### 10.2 示例：自研 MatMul+BiasAdd 融合算子

假设 profiler 显示 MatMul 和 BiasAdd 分别占 3.5% 和 1.2%，且 vLLM-Ascend 使用了融合算子：

**Step 1: 确定目标**

```
kernel: MatMul + BiasAdd (分散)
AICore 占比: 4.7%
vLLM-Ascend 使用 CANN CCE 的融合 MatMul+BiasAdd
```

**Step 2: 选择实现路径**

```
TileLang (首选，开发快) or AscendC (复杂逻辑)
→ 选择 TileLang，因为是简单的逐 element 操作
```

**Step 3: 编写 kernel**

```python
# xllm/compiler/tilelang/matmul_bias.py
import tilelang.language as T

@T.prim_func
def matmul_bias_kernel(
    A: T.Tensor([M, K], "float16"),
    B: T.Tensor([K, N], "float16"),
    bias: T.Tensor([N], "float16"),
    C: T.Tensor([M, N], "float16"),
):
    with T.grid(M, N) as (m, n):
        with T.tile_scope(
            tile_M=64, tile_N=128, tile_K=64,
            block_M=2, block_N=2, block_K=1
        ):
            T.copy(A_tile, A_shared)
            T.copy(B_tile, B_shared)
            C_tile = T.matmul(A_shared, B_shared, accumulate=True)
            C_tile += bias_tile  # Fuse bias add
            T.copy(C_tile, C)
```

**Step 4: 测试**

```bash
python test/ops/npu/test_matmul_bias.py --device npu
```

**Step 5: 基准**

```bash
python kernel-pilot/tools/npu-op-benchmark.py \
  --op matmul_bias \
  --shapes "128,4096,8192" "1,4096,8192" \
  --dtype float16
```

**Step 6: 接入 xLLM**

替换原有分散调用，重新走 RLCR。

---

## 11. 完整目录结构参考

```
$RUN_ROOT/
├── manifest.md                        # 运行元信息
│
├── benchmark/
│   ├── xllm_results.jsonl             # xLLM 各候选结果
│   ├── vllm_results.jsonl             # vLLM-Ascend 各候选结果
│   ├── round_N_results.jsonl          # 每轮 RLCR后的重新测试
│   ├── comparison/
│   │   ├── summary.md                 # 核心对比表
│   │   ├── summary.csv
│   │   └── winning-commands.md        # 双方最优配置命令
│   └── final_summary.md               # 最终对比结果
│
├── profiles/
│   ├── xllm/                          # xLLM profiling 数据
│   │   ├── step_trace_time.csv
│   │   ├── op_statistic.csv
│   │   ├── kernel_details.csv
│   │   └── analysis.db
│   └── vllm/                          # vLLM-Ascend profiling 数据
│
├── analysis/
│   ├── root-cause.md                  # 五表报告（Phase 3）
│   └── round_N_root_cause.md          # 每轮 RLCR 后的重新分析
│
├── history/
│   └── xllm-model-history-notes.md    # PR 历史笔记（Phase 0.5）
│
├── humanize/
│   ├── refined-plan.md                # 优化 Plan（Phase 4）
│   ├── attempt-ledger.md              # 每次尝试记录
│   ├── optimization-ledger.md         # 有效优化记录
│   ├── source-idea-ledger.md          # 想法来源
│   └── lineage.jsonl                  # 血缘追踪
│
├── patches/                           # 每轮 patch 文件
│   ├── round_1_*.patch
│   └── round_2_*.patch
│
└── artifacts/                         # 事故诊断产出（按需）
    └── YYYYMMDD_HHMMSS/
        ├── npu_status.log
        ├── dmesg.log
        └── ...
```

---

## 12. 时间线概览

| 阶段 | 工作 | 预计时间 | 产出 |
|------|------|---------|------|
| Phase 0 | 环境准备 | 0.5 天 | manifest.md |
| Phase 0.5 | PR 历史查询 | 0.5 天 | model-history-notes.md |
| Phase 1 | 公平基准测试 | 1-2 天 | summary.md + winning-commands.md |
| Phase 2 | 差异判定 | 0.5 天 | 继续 or 结束 |
| Phase 3 | Profiling 五表分析 | 1 天 | root-cause.md |
| Phase 4 | 构建 Plan | 0.5 天 | refined-plan.md |
| Phase 5 | RLCR 迭代（每轮） | 1-3 天 | attempt-ledger.md + round_N_* |
| 收尾 | 最终报告 | 0.5 天 | final_summary.md |

**典型总周期**：2-4 周（取决于 gap 大小和优化难度）。

### 快速启动命令汇总

```bash
# === Phase 0: 环境准备 ===
npu-smi info
xllm serve --help
vllm serve --help

# === Phase 1: 基准测试 ===
# xLLM baseline
xllm serve /models/Qwen3.5-27B --tensor-parallel-size 4 --graph-mode npugraph_ex --block-size 128 --port 8080

# vLLM-Ascend baseline
VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve /models/Qwen3.5-27B --tensor-parallel-size 4 --enforce-eager --block-size 128 --gpu-memory-utilization 0.9 --port 8000

# 对比结果
python xllm-npu-optimization-skills/skills/xllm-npu-benchmark/scripts/compare_npu_benchmark.py \
  --xllm-results xllm_results.jsonl --vllm-results vllm_results.jsonl --output-dir comparison/

# === Phase 3: Profiling ===
XLLM_PROFILING=1 xllm serve /models/Qwen3.5-27B --tensor-parallel-size 4 --graph-mode npugraph_ex --port 8080

python xllm-npu-optimization-skills/skills/xllm-npu-profiler/scripts/analyze_xllm_npu_profile.py \
  --framework xllm --url http://127.0.0.1:8080 --output-dir profiles/xllm/ --profile-by-stage

python xllm-npu-optimization-skills/skills/xllm-npu-profiler/scripts/render_triage_npu.py \
  --analysis-root profiles/ --output analysis/root-cause.md

# === Phase 5: 编译验证 ===
python setup.py build --device npu
python setup.py test --device npu
```
