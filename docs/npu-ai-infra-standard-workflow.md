# 通用 NPU 大模型推理与 AI Infra 开发工作流

> 本文定义本仓库的通用定位：不只服务 xLLM，而是沉淀一套可迁移到
> xLLM、vLLM-Ascend、SGLang 等框架的 NPU 大模型推理优化和 AI Infra
> 开发标准流程。现有 xLLM/Qwen3.5-27B/MTP 经验是第一批落地案例，不能
> 被通用化改造覆盖或丢失。

## 1. 仓库通用定位

本仓库是一套 **NPU AI Infra Auto-Driven SKILLS**：面向华为昇腾 NPU
上的大模型推理框架，帮助 Agent 和工程师完成 benchmark、profiling、
capacity planning、incident triage、code review、kernel pilot、PR
history 查询和 SOTA loop。

适用框架分三层：

| 层级 | 框架 | 角色 |
|---|---|---|
| 一等支持 | xLLM | 当前主要落地框架，已有真实优化案例、patch 和 profiling 记录 |
| 一等对照 | vLLM-Ascend | 公平 benchmark 对照组，也应逐步支持 profiler、capacity、incident 能力 |
| 扩展支持 | SGLang NPU | 按本仓库的 NPU 证据标准和框架适配层逐步补齐后端经验 |

因此，仓库当前 skill 名称保留 `xllm-npu-*`，但新增能力应尽量把
**NPU 通用流程** 和 **框架适配层** 分开：

- 通用层：NPU benchmark schema、profiling 五表、capacity 模型、MFU、
  incident artifact、review rubric、RLCR ledger。
- 适配层：xLLM、vLLM-Ascend、SGLang 的启动命令、日志格式、metrics
  endpoint、profiling 采集方式、源码路径、PR history。

## 2. 标准闭环

所有框架都使用同一条 evidence-driven 闭环：

```text
Phase 0     环境与目标初始化
Phase 0.5   框架/模型 PR history 查询
Phase 1     公平 benchmark
Phase 2     gap 判定
Phase 3     NPU profiling / capacity / compute 证据采集
Phase 4     优化计划
Phase 5     RLCR 迭代
Phase 6     记录、复盘、沉淀为知识库
```

### Phase 0：环境与目标初始化

必须记录：

- 模型、权重、tokenizer、precision、quantization。
- NPU 型号、卡数、CANN、HDK Driver、torch_npu、HCCL。
- 框架：xLLM / vLLM-Ascend / SGLang。
- 框架 commit、容器镜像、启动命令、可见 NPU。
- workload、SLA、artifact root。

### Phase 0.5：PR history 查询

优化前先查历史，而不是直接动代码：

- xLLM：查 `model-pr-optimization-history/xllm/`。
- vLLM-Ascend：后续应新增 `model-pr-optimization-history/vllm-ascend/`。
- SGLang NPU：后续应新增 `model-pr-optimization-history/sglang/`，沉淀
  SGLang NPU 后端相关 model dossier。

输出应写入：

```text
$RUN_ROOT/history/model-history-notes.md
```

### Phase 1：公平 benchmark

公平 benchmark 是全仓库最高优先级的证据入口。

通用规则：

- 同 NPU 型号、同卡数、同模型、同 tokenizer、同 precision、同 workload、
  同 sampling、同 SLA。
- 每个框架独立调优，不允许 tuned A 对比 default B。
- 保留失败候选和失败原因。
- 输出 normalized JSONL、summary、CSV、winning commands。

框架适配：

| 框架 | 启动/测试方式 |
|---|---|
| xLLM | `xllm serve` + evalscope/OpenAI-compatible benchmark |
| vLLM-Ascend | `vllm serve` + evalscope/vLLM benchmark |
| SGLang NPU | `python -m sglang.launch_server` + `sglang.bench_serving` 或 evalscope |

### Phase 2：gap 判定

通用 gap 公式：

```text
gap = (best_reference_throughput - target_throughput) / best_reference_throughput
```

默认阈值：

- `gap <= 1%`：视为平局或达标。
- `gap > 1%`：进入 profiling 与优化。
- target 胜出：写 final summary，但仍记录配置、workload 和证据。

### Phase 3：NPU 证据采集

NPU 证据层不应只看单个 kernel，而要组合：

1. Profiling 五表：kernel、communication/overlap、fuse、dispatch、memory。
2. Pipeline/layer 分析：代表性 layer、forward pass、rank skew。
3. Compute simulation：FLOPs、MFU、kernel-to-op mapping。
4. Capacity planning：HBM、KV blocks、xTensor/KV cache、MTP reserve。

不同框架可以有不同采集方式，但输出 schema 应保持一致，便于 Agent
在 SOTA loop 中统一判断。

### Phase 4：优化计划

优化 plan 必须从证据导出，至少包含：

- 根因判断。
- 涉及框架和源码路径。
- 优化路径分类：融合算子、KV cache、图模式、通信重叠、capacity、
  scheduler、MTP/spec decode、MoE/EP、kernel pilot。
- 预计收益和验证口径。
- 风险：精度、图模式、通信、内存、兼容性。

### Phase 5：RLCR 迭代

每轮都按同一结构：

```text
Research → Learn → Code → Review → Validate → Record
```

其中：

- Research：读 benchmark/profiler/capacity/compute 证据。
- Learn：查 PR history 和历史 attempt。
- Code：只做一个可验证 patch。
- Review：调用框架特化 NPU code review。
- Validate：编译、单测、精度、benchmark、profiling。
- Record：写 attempt ledger、optimization ledger、lineage。

#### PR 提交门禁

Validate 阶段通过后再提交 PR。对于 xLLM NPU 改动，默认门禁是：

- 在唯一权威 PR worktree 中执行，先确认 `git status --short` 无非预期修改。
- 编译前执行 `git submodule update --init --recursive`，并确认没有其他编译/测试进程写同一 build 目录。
- 使用 `python setup.py build test --device npu` 一次完成编译和 UT；不要把单独 build 成功当成提交通过。
- push 后核对 fork branch 与 PR head commit；PR 描述、检视意见回复和代码 push 必须分别确认。
- 对关键 review 文件从远端 PR ref 读取复查，避免本地多个 worktree 造成误判。

### Phase 6：沉淀

有效经验必须回写：

- `humanize/attempt-ledger.md`
- `humanize/optimization-ledger.md`
- `humanize/source-idea-ledger.md`
- `humanize/lineage.jsonl`
- 对应模型 PR history dossier
- 对应 profiler/capacity/incident case study

失败实验同样要保留，避免下一轮 Agent 重复踩坑。

## 3. 通用能力矩阵

| 能力 | 通用目标 | 当前状态 | 后续方向 |
|---|---|---|---|
| Benchmark | 多框架公平比较 | xLLM vs vLLM-Ascend 已有雏形 | 支持 xLLM/vLLM-Ascend/SGLang 统一 cookbook |
| Profiler | NPU 五表报告 | xLLM trace parser 初版 | 加 layer/rank/MFU/capacity 联动 |
| Capacity | HBM/KV/并发解释 | 文档中有经验 | 新增独立 skill 和 analyzer |
| Compute simulation | FLOPs/MFU/shape what-if | 缺失 | 新增 NPU compute simulator |
| Incident triage | replay-first 排障 | 流程文档初版 | 新增 artifact tool 和 replay helper |
| Code review | NPU 特化审查 | 规则清单初版 | 增加 review corpus 与查询 |
| PR history | 模型历史记忆 | xLLM 少量档案 | 扩展 xLLM/vLLM-Ascend/SGLang dossier |
| Kernel pilot | 热点 kernel 深挖 | TileLang/AscendC 知识初版 | 增加准入、benchmark、端到端 lineage |
| Humanize ledger | 长期优化记忆 | 文件存在 | 增加 schema 和 orchestrator |

## 4. 保留现有经验的原则

通用化时不得删除或稀释已有 xLLM 实战信息：

- Qwen3.5-27B MTP 配置扫描。
- nst=1 / nst=3 在不同 workload 下的经验。
- MTP transpose 消除。
- PR #1536 / #1541 分析。
- chunked prefill 反例。
- accept mask 小算子反例。
- MTP draft extend 提前准备 P0/P0b。
- CANN / torch_npu / xLLM commit / evalscope 真实环境信息。

这些内容应作为 `case studies`、`model-pr-history`、`humanize ledger`
继续保留，并作为未来 vLLM-Ascend/SGLang NPU 迁移时的对照样本。

## 5. 推荐命名策略

短期不强制重命名已有 skill，避免破坏已沉淀经验和引用路径。

推荐新增通用能力时使用以下策略：

- 保留：`xllm-npu-benchmark`、`xllm-npu-profiler` 等现有入口。
- 新增通用参考文件：`references/framework-adapter-contract.md`、
  `references/npu-result-schema.md`、`references/run-manifest-schema.md`。
- 新增框架适配目录：

```text
frameworks/
├── xllm/
├── vllm-ascend/
└── sglang/
```

每个适配目录保存：

- launch templates
- help snapshots
- metrics endpoint notes
- profiling collection notes
- source map
- known pitfalls

中长期再考虑把 skill 名称从 `xllm-npu-*` 演进为 `npu-serving-*`，
同时保留兼容入口。
