# xllm NPU 优化方案设计文档

> 设计日期：2026-05-23
> 目标硬件：华为昇腾 910B3 (A3)，HDK Driver 25.2.0+
> 竞品对照：vLLM-Ascend
> 参考资料：昇腾 NPU 推理优化经验、xLLM 技术报告 (arXiv:2510.14686)

---

## 0. 通用化说明

本文最初以 xLLM 为主要落地对象，因此大量源码路径、案例和 patch 讨论都
围绕 xLLM 展开。这些内容是仓库当前最重要的实测经验，应继续保留。

仓库后续定位会从“xLLM NPU 优化 skill”扩展为“xLLM、vLLM-Ascend、
SGLang NPU 共用的 NPU 大模型推理和 AI Infra 开发工作流”。扩展时遵循
两层结构：

- **通用 NPU 证据层**：公平 benchmark、profiling 五表、capacity、
  compute simulation、MFU、incident artifact、review rubric、RLCR ledger。
- **框架适配层**：xLLM、vLLM-Ascend、SGLang 各自的启动命令、日志格式、
  metrics endpoint、profiling 采集方式、源码路径和 PR history。

因此，本文后续章节仍可作为 xLLM 适配层和首个 case study 阅读；通用流程
以 `docs/npu-ai-infra-standard-workflow.md` 为准。

---

## 1. xllm 框架现状分析

### 1.1 架构

**Service-Engine 解耦架构**：

- **Service 层**：
  - 智能调度模块：在线/离线任务共存，统一弹性调度
  - 工作负载自适应 PD（Prefill-Decode）分离
  - EPD（Encode-Prefill-Decode）分离（多模态输入）
  - 全局 KV Cache 管理（基于 Mooncake）和容错高可用

- **Engine 层**：
  - 全图流水线编排：请求调度 + 模型图 + 算子三层异步流水线
  - 自适应图模式：Prefill eager + Decode npugraph_ex/GE，动态 shape 参数化
  - xTensor 内存管理：物理离散→虚拟连续映射、按需分配、智能调度
  - 投机解码：Draft model 多核并行
  - 动态 EPLB：MoE 专家负载均衡

### 1.2 技术栈

- 语言：C++ 91.6%、CUDA 3.9%、Python 2.6%、Rust 0.2%
- 核心目录：`xllm/core/`
  - `framework/`：执行编排框架
  - `kernels/`：NPU 算子适配
  - `platform/`：多平台适配层（NPU/MLU/ILU/GPU）
  - `runtime/`：Worker 和 Executor
  - `scheduler/`：Batch 和 PD 调度器
  - `layers/`：模型层实现

### 1.3 模型支持（NPU）

| 类别 | 模型 |
|------|------|
| LLM | DeepSeek-V3/R1/V3.1/V3.2、Qwen2/2.5/3/QwQ、Qwen3 MoE、Kimi-k2、Llama2/3、GLM4.5/4.6/4.7/5 |
| VLM | Qwen2.5-VL、Qwen3-VL/MoE、GLM-4.6V、MiniCPM-V、MiMo-VL、VLM-R1 |
| 其他 | Qwen3-Rerank、Flux(DiT)、OneRec/Qwen(Rec) |

### 1.4 已有优化能力

- 自适应图模式（npugraph_ex / GE）
- xTensor 内存池
- PD/EPD 分离调度
- 全局 KV Cache（Mooncake）
- 投机解码
- 动态 EPLB
- 7 个 Agent Skills（code-review, tilelang-ascend-kernel 等）

### 1.5 已有性能基线

在同 TPOT 约束下：
- Qwen 系列：吞吐量 = MindIE × 1.7，vLLM-Ascend × 2.2
- DeepSeek 系列：吞吐量 = MindIE × 1.7

---

## 2. 优化方案总体架构

### 2.1 设计原则

1. **证据驱动**：所有优化决策必须有 benchmark 数据 + profiling trace + PR 历史
2. **阶段分离**：Prefill 和 Decode profiling 分开采集
3. **公平比较**：相同 NPU 硬件、模型、工作负载、SLA
4. **自治循环**：RLCR（Research-Learn-Code-Review）驱动持续优化
5. **NPU 特化**：五表报告、昇腾优化路径、AICore 证据门槛

### 2.2 架构总览

```
                        ┌──────────────────────────────────────┐
                        │    xllm-npu-sota-loop                  │
                        │    xLLM NPU SOTA 自治优化循环           │
                        │    (对标 vLLM-Ascend)                   │
                        └──────────────┬───────────────────────┘
                                       │
              ┌────────────────────────┼─────────────────────┐
              ▼                        ▼                      ▼
 ┌─────────────────────┐ ┌──────────────────────┐ ┌────────────────────────┐
 │ xllm-npu-benchmark  │ │ xllm-npu-profiler    │ │ xllm-model-pr-history  │
 │ NPU 推理基准测试     │ │ 昇腾 Profiling 分析  │ │ xllm 模型优化PR历史     │
 │ (对标 vLLM-Ascend)  │ │ (五表报告)           │ │                        │
 └─────────────────────┘ └──────────────────────┘ └────────────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
 ┌─────────────────────┐ ┌──────────────────────┐ ┌────────────────────────┐
 │ xllm-npu-code       │ │ xllm-npu-incident    │ │ xllm-npu-kernel-pilot  │
 │ review              │ │ triage               │ │ NPU 内核证据辅助        │
 │ NPU 特化代码审查    │ │ NPU 生产事故诊断     │ │ (TileLang/AscendC)     │
 └─────────────────────┘ └──────────────────────┘ └────────────────────────┘
```

---

## 3. 五个核心 Skills 设计

### 3.1 xllm-npu-benchmark（NPU 推理基准测试）

**对标**：`llm-serving-auto-benchmark`

**工作流**：

```
Step 1: Preflight（环境验证）
├── npu-smi info 确认 NPU A3 设备健康状态、可用显存
├── 验证框架 CLI 可用性（xllm serve / vllm serve）
├── 确认 CANN >= 8.0.RC1，HDK Driver >= 25.2.0
├── ASCEND_RT_VISIBLE_DEVICES 确认可见设备
└── 保存环境信息到 artifact/env.json

Step 2: 规范化工作负载
├── 统一 JSONL 格式（prompt + output_len）
├── 默认场景：
│   ├── chat: input=1000, output=1000
│   ├── summary: input=8000, output=1000
│   └── long-context: input=32000, output=1000
├── 所有框架使用相同 tokenizer、精度(bf16)、量化方案
└── 对齐 max_model_len 和 block_size

Step 3: 搜索层级选择
├── Tier 1 (smoke): 少量请求快速验证
├── Tier 2 (默认): 有界扫描，每框架 ≤10 候选
└── Tier 3 (exhaustive): 穷举搜索

Step 4: xllm 调优
├── 调优维度:
│   ├── 并行策略: tp/pp/ep
│   ├── 图模式: eager / npugraph_ex / GE
│   ├── KV Cache: block_size, PA模式
│   ├── 调度策略: chunked_prefill, PD分离策略
│   ├── 内存: xTensor 参数, mem_fraction_static
│   └── 算法: 投机解码配置, EPLB 参数
├── 记录每次运行的完整启动命令
└── 保留失败候选及原因

Step 5: vLLM-Ascend 调优（对照组）
├── 相同工作负载和 SLA
├── 调优: TP/PP/KV cache/graph mode/enable-prefix-caching
├── VLLM_WORKER_MULTIPROC_METHOD=spawn
└── 记录版本和 commit

Step 6: 结果归一化
├── 排序: SLA通过 > 请求吞吐 > 输出token吞吐 > p50 TTFT > p50 TPOT
├── 输出 Markdown/CSV 对比表
└── 生成 winning-commands.md + candidates.jsonl
```

**公平性规则**（NPU 适配版）：
- 相同 NPU 型号(A3)/卡数/模型权重/tokenizer/精度/量化/采样设置
- 记录 CANN 版本、HDK Driver 版本、框架 commit、Docker 镜像
- 候选之间重启或清除状态
- `ASCEND_RT_VISIBLE_DEVICES` 设置一致

---

### 3.2 xllm-npu-profiler（昇腾 NPU Profiling 分析）

**对标**：`llm-torch-profiler-analysis` + `profiling-analysis`

**五表输出契约**：

| 表名 | 内容 | 数据来源 |
|------|------|---------|
| **kernel table** | AICore 内核占比排序 | op_statistic.csv / kernel_details.csv |
| **overlap-opportunity table** | 计算-通信重叠机会 | step_trace_time.csv |
| **fuse-pattern table** | 融合算子模式匹配 | 模型代码 + torch_npu 映射 |
| **下发效率表** | Hostbound 分析：空闲率/AICore利用率 | step_trace_time.csv + analysis.db |
| **内存效率表** | xTensor/KV Cache 利用率 | xllm 内部 metric + 昇腾 profiling |

**工作流**：

```
Step 1: Profiling 数据采集
├── xllm 内置 Profiling：XLLM_PROFILING=1
├── 昇腾 Profiling：msprof / ascend profiling
├── 阶段分离：Prefill 和 Decode 分别采集（继承 AI-Infra 原则）
├── warmup 10 步 + 活跃采集 5 步
└── 保存 trace 到 artifact/profiles/

Step 2: 全局瓶颈判定
├── 空闲 >20% → Hostbound（下发问题）
├── 计算 >85% → Computing（热点 kernel）
├── 通信 >10% → Communication（集合通信）
└── 内存碎片率高 → Memory（xTensor/KV Cache）

Step 3: Sub-skill 分发
├── hostbound 子分析：图模式覆盖率、GE 编译开销、算子下发效率
├── computing 子分析：Top-N 热点算子、MatMul 形状、融合机会
├── communication 子分析：AllReduce/AllGather 带宽、TP通信开销
└── memory 子分析：KV Cache 碎片率、xTensor 池命中率

Step 4: 五表报告生成
├── Markdown 格式，标注 Prefill/Decode 阶段
├── 对比 baseline 和 competitor 差异
└── 输出 analysis/root-cause.md
```

**NPU 特化差异**：
- 解析昇腾专有格式（step_trace_time.csv, op_statistic.csv, kernel_details.csv, analysis.db）
- 支持 msprof / ascend profiling 命令
- 结合 xllm 自适应图模式分析 GE/AclGraph 编译开销
- 新增下发效率和内存效率两个维度

---

### 3.3 xllm-npu-sota-loop（NPU SOTA 自治优化循环）

**对标**：`sglang-sota-humanize-loop`

**6 Phase 工作流**：

```
Phase 0: 初始化
├── 收集：模型/精度/NPU型号(A3)/CANN版本/框架集/artifact root
├── 创建运行目录：
│   runs/YYYYMMDD_<model_slug>_npu_sota/
│     manifest.md / benchmark/ / profiles/ / analysis/
│     history/ / kernel/ / patches/ / humanize/
└── 确认环境就绪

Phase 0.5: 模型 PR 历史门控
├── 查询 model-pr-history 知识库
├── 检查已知风险和待验证想法
└── 写入 history/xllm-model-history-notes.md

Phase 1: 固定公平基准测试
├── 调用 xllm-npu-benchmark
├── 不允许将 tuned xllm 与 vLLM-Ascend defaults 比较
└── 生成 candidates.jsonl, summary.md, winning-commands.md

Phase 2: 差异判定
├── 阈值 1%，超出才启动 RLCR
└── xllm 胜出 → 确认后写报告结束

Phase 3: RLCR 前的必要 Profiling
├── 调用 xllm-npu-profiler
├── 必须使用 benchmark 慢场景的实际 input/output 长度
├── 生成五表报告 → analysis/root-cause.md
└── 此报告不存在前不得开始 patch

Phase 4: 构建优化 Plan
├── 基于五表报告确定 patch 优先级
├── 按昇腾优化路径分类（见 Section 4）
└── 写入 .humanize/xllm-npu-agent/refined-plan.md

Phase 5: 启动 RLCR 迭代
├── 每轮：分析 → 修改 → 验证 → 重跑 benchmark+profiler
├── 记录 attempt-ledger.md（每次尝试）
└── 记录 optimization-ledger.md（有效优化）
```

**停止条件**：

| 条件 | 说明 |
|------|------|
| xllm 胜出 | 固定工作负载上击败 vLLM-Ascend SLA-passing 结果 |
| 平局 | 稳定 1% 阈值内 |
| 硬件限制 | NPU 不支持特定算子/特性 |
| 瓶颈已至 | 热点路径接近硬件/算法极限 |

---

### 3.4 xllm-npu-code-review（NPU 特化代码审查）

**对标**：`sglang-humanize-review`

**核心关注点**：
- **C++ 引擎代码**：内存安全、AICore 资源利用、数据对齐
- **TileLang 算子**：Tiling 策略、UB 切分、多核并行
- **AscendC 算子**：Buffer 管理、指令流水、精度
- **图模式兼容**：GE/AclGraph 编译兼容、Graph Break 风险
- **KV Cache 实现**：PA block_table/slot_mapping 正确性、NZ 格式
- **通信代码**：HCCL 调用规范、AllReduce 重叠正确性
- **精度问题**：FP16/BF16 混合精度、KVCache Prefill/Decode 对齐

**审查知识库来源**：
- xllm 已有 `code-review` skill
- awesome-ascend-skills: `ascendc-code-review`
- awesome-ascend-skills: `model-infer-precision-debug`

---

### 3.5 xllm-npu-incident-triage（NPU 生产事故诊断）

**对标**：`sglang-prod-incident-triage`

**NPU 常见问题模式**：
- AICore timeout / 算子卡死
- HCCL 通信超时/失败
- NPU OOM（显存不足）
- KV Cache 碎片导致调度失败
- PD 分离后延迟异常
- GE/AclGraph 编译/运行时错误
- NPUGraph_ex replay 失败
- 精度异常（KVCache 对齐、FlashAttention 精度）

**诊断工作流**：

```
Step 1: 证据保全
├── npu-smi info 状态快照
├── xllm 日志收集
├── dmesg 驱动日志
└── 请求 replay 记录

Step 2: 问题分类
├── 运行时错误 → runtime-debug 子流程
├── 精度问题 → precision-debug 子流程
├── 性能退化 → profiler
└── 通信异常 → hccl-test 子流程

Step 3: 根因分析 + 修复验证
├── 参考昇腾社区经验（hiascend-forum）
├── 使用 npu-smi / ascend-dmi 诊断
└── 最小化复现 → 修复 → 回归验证
```

---

## 4. 昇腾 NPU 特化优化路径

基于 awesome-ascend-skills 推理优化 skills，SOTA 循环应覆盖以下路径：

| 优化路径 | 优先级 | 对应 awesome-ascend-skills | xllm 适配方式 |
|---------|--------|---------------------------|--------------|
| 融合算子替换(torch_npu) | P0 | model-infer-fusion | 替换标准算子为 torch_npu 融合算子 |
| KV Cache 优化 | P0 | model-infer-kvcache | PA模式/NZ格式/MLA压缩 |
| 图模式适配(GE/AclGraph) | P0 | model-infer-graph-mode | 仅 Decode 启用，Prefill 保持 eager |
| 权重预取(npu_prefetch) | P1 | model-infer-prefetch | 提前加载下一层权重到 SRAM |
| 多流重叠(stream overlap) | P1 | model-infer-multi-stream | 计算-通信双流 |
| SuperKernel融合 | P1 | model-infer-superkernel | 算子二进制融合 |
| 并行策略(TP/EP/DP) | P1 | model-infer-parallel-analysis | 最优并行配置推荐 |
| 通信优化(HCCL) | P2 | hccl-test | AllReduce/AllGather 重叠 |
| 投机解码 | P2 | xllm 自有 | Draft model 选择 |
| MoE EPLB | P2 | xllm 自有 | 动态专家负载均衡 |

### xllm 独有优化点

1. **自适应图模式切换**：Prefill eager + Decode npugraph_ex/GE
2. **xTensor 内存管理**：物理离散→虚拟连续映射、按需分配
3. **PD/EPD 分离调度**：动态 Prefill-Decode 或 Encode-Prefill-Decode
4. **全局 KV Cache（Mooncake）**：跨节点路由、分层缓存
5. **动态 EPLB**：MoE 专家负载均衡
6. **全图流水线编排**：三层异步流水线

---

## 5. 内核辅助层：xllm-npu-kernel-pilot

### 能力矩阵

| 能力 | 来源 |
|------|------|
| AscendC 算子开发 | awesome-ascend-skills: `ascendc` |
| TileLang 算子开发 | xllm 已有: `tilelang-ascend-kernel` |
| Triton-Ascend 迁移 | awesome-ascend-skills: `triton-ascend-migration` |
| NPU Profiling 采集 | awesome-ascend-skills: `ops-profiling` |
| 算子精度调试 | awesome-ascend-skills: `ascendc-precision-debug` |
| 算子性能调优 | awesome-ascend-skills: `pypto-op-perf-tune` |

### 准入条件

1. xllm 仍落后 vLLM-Ascend > 1%
2. 慢阶段某个 kernel 族占 AICore 时间 >= 1%
3. profiler 对比显示该 kernel 是差异的合理解释
4. 有明确的正确性参考和代表性 shapes

---

## 6. 知识库：model-pr-optimization-history

### 目录结构

```
model-pr-optimization-history/
├── SKILL.md
├── scripts/query.py
└── xllm/
    ├── deepseek-v3.md
    ├── qwen3-core.md
    ├── glm-5.md
    ├── kimi-k2.md
    └── ...
```

### 每个档案回答

- 哪些 PR 改了这个模型路径？
- PR 是 merged / closed / open？
- 哪些文件和符号变了？
- 有什么优化或正确性风险需要在修改前检查？
- 应对标哪个上游想法？

---

## 7. 实施路线图

### Phase 1：基础能力（2 周）

- [x] 仓库目录结构
- [x] README.md
- [ ] xllm-npu-benchmark: 建立 xllm vs vLLM-Ascend 公平比较基线
- [ ] xllm-npu-profiler: 建立五表输出契约
- [ ] 验证：在一个模型（Qwen3）上完成 benchmark+profiler 闭环

### Phase 2：核心循环（3 周）

- [ ] xllm-npu-sota-loop: 实现 RLCR 自治优化循环
- [ ] 整合 awesome-ascend-skills 的推理优化 skills
- [ ] 验证：在 Qwen3 上完成 SOTA loop 端到端运行

### Phase 3：运营能力（2 周）

- [ ] xllm-npu-code-review: 建立 NPU 代码审查知识库
- [ ] xllm-npu-incident-triage: 建立生产事故诊断流程
- [ ] model-pr-optimization-history: 开始积累 xllm PR 历史
- [ ] 验证：在 2+ 模型家族上积累 PR 历史

### Phase 4：内核深度（持续）

- [ ] kernel-pilot: TileLang/AscendC/Triton-Ascend 内核优化
- [ ] 针对 Top 热点 kernel 的 NPU 专项优化
- [ ] 验证：至少 3 个 kernel 族有性能提升证据
