# NPU AI Infra Skills 待实现能力与路线规划

> 日期：2026-05-28  
> 依据：当前 `pjgao/xllm-npu-optimization-skills` 仓库结构、已有 xLLM NPU 实测经验、以及后续多框架 NPU 工作流目标。

## 1. 总体判断

当前仓库已经完成了“昇腾 NPU 大模型推理优化闭环”的骨架：有
`xllm-npu-sota-loop`、benchmark、profiler、code-review、incident-triage、
model-pr-history、kernel-pilot，也有 xLLM + Qwen3.5-27B + MTP 的真实案例。

后续定位不应局限于 xLLM，而应演进为 xLLM、vLLM-Ascend、SGLang NPU
共用的一套 AI Infra 开发和推理优化标准流程。现有 `xllm-npu-*` 命名可先
保留作为兼容入口，但新增设计应区分“通用 NPU 证据层”和“框架适配层”。

要成为可长期运行的 NPU AI Infra 工作流平台，它还缺少四类关键能力：

1. **可执行工具链不完整**：很多 skill 是流程描述，缺少足够脚本、schema、测试和 cookbook。
2. **NPU 证据层还偏浅**：已有五表框架，但缺少 layer/pipeline、compute simulation、MFU、跨 rank / stage 分析。
3. **长期闭环不够自动化**：Humanize 账本存在，但缺少 orchestrator、goal 模板、固定产物 schema 和自动校验。
4. **知识库规模不足**：xLLM PR history 只有少量模型档案，vLLM-Ascend /
   SGLang NPU 还缺少适配档案，缺少自动构建、质量测试和中英双语结构化 dossier。

通用化时必须保留已有经验：Qwen3.5-27B MTP、PR #1536、PR #1541、
chunked prefill 反例、小算子反例、MTP draft prepare P0/P0b、CANN /
torch_npu / evalscope 环境记录都应继续作为 case study 和 regression evidence。

## 2. 待实现能力清单

### 2.1 Benchmark：从 evalscope 收集器升级为多框架 config-driven search

当前已有：

- `collect_evalscope_results.py`
- `compare_npu_benchmark.py`
- `validate_framework_cli.py`
- 少量真实 Qwen3.5-27B benchmark 记录。

缺口：

- 缺少统一的多框架模型配置库。
- 缺少统一 `result-schema.md`，现在 JSONL 字段更多靠脚本约定。
- 缺少 launch command / benchmark command / SLA / failure reason 的强约束。
- 缺少候选生成器：当前更像“收集已跑结果”，不是“按 search_space 为 xLLM / vLLM-Ascend / SGLang NPU 渲染候选命令”。
- 缺少 tests，无法保护 benchmark schema 和排序逻辑。

应实现：

- `skills/xllm-npu-benchmark/references/result-schema.md`
- `skills/xllm-npu-benchmark/references/example-plan.yaml`
- `skills/xllm-npu-benchmark/configs/cookbook-npu-llm/*.yaml`
- `frameworks/{xllm,vllm-ascend,sglang}/launch-templates.md`
- `scripts/validate_cookbook_configs.py`
- 扩展 `compare_npu_benchmark.py` 支持 failure rows、scenario fields、SLA reason。

### 2.2 Profiler：从五表 summary 升级为多框架 NPU trace evidence stack

当前已有：

- `analyze_xllm_npu_profile.py`
- `render_triage_npu.py`
- 五表输出契约。

缺口：

- 缺少 layer / forward-pass / kernel boundary 级别分析。
- 缺少 NPU 版 Perfetto / timeline 时间映射。
- 缺少跨 rank 对齐和 rank skew 分析。
- 缺少 NPU op 到 xLLM / vLLM-Ascend / SGLang 源码路径的稳定映射。
- fuse / overlap catalog 仍偏静态，缺少 deterministic matcher 和测试。

应实现：

- 新 skill：`skills/xllm-npu-pipeline-analysis/`（短期沿用 xLLM 前缀，内部产物 schema 设计为多框架）
- 脚本：
  - `scripts/npu_layer_timeline_analyzer.py`
  - `scripts/npu_layer_kernel_breakdown.py`
  - `scripts/npu_timeline_mapper.py`
  - `scripts/npu_rank_skew_analyzer.py`
- 扩展 profiler 输出：从五表报告增加“代表性 layer deep dive”和“rank skew table”。

### 2.3 Compute Simulation：补通用 NPU FLOPs / MFU / shape what-if

当前 NPU 仓库还没有从模型 config 推导 operator flow、FLOPs、MFU、
kernel-to-op mapping 的独立能力。

缺口：

- 无法回答“这个 kernel 慢是不是接近硬件上限”。
- 无法把 profiler kernel time 转成 MFU 或 theoretical min time。
- 无法系统比较 TP/EP/DP、prefill/decode shape、MTP/spec decode、MoE/MLA 等路径的 compute cost。

应实现：

- 新 skill：`skills/xllm-npu-compute-simulation/`
- `references/npu-specs.json`：910B3/A3 的 bf16/fp16/int8/fp8 峰值、带宽、UB/L1/L2、AICore 数量。
- `references/model-config-index.json`：Qwen3/Qwen3.5/DeepSeek/GLM/Kimi 等 NPU serving 常用模型。
- `scripts/xllm_npu_compute_simulator.py`
- 支持 `--kernel-flow @profile.json`，输出 kernel-level MFU 表。

### 2.4 Capacity Planner：补多框架启动日志、KV cache、HBM 容量解释

当前仓库文档里讨论了 MTP reserved linear bytes、KV blocks，但没有独立 capacity skill。

缺口：

- 不能从 xLLM / vLLM-Ascend / SGLang 启动日志自动解释 KV cache 容量、xTensor/Block manager、MTP/spec decode reserve。
- benchmark 前无法预估 max_seqs、block_size、max_model_len 是否合理。
- incident triage 时无法快速区分 OOM、碎片、capacity 配置错误。

应实现：

- 新 skill：`skills/xllm-npu-capacity-planner/`
- `scripts/xllm_npu_capacity_analyzer.py`
- 输入：框架 startup log、模型 config、NPU 卡数、TP/PP/EP、block_size、max_model_len、MTP/spec decode 参数。
- 输出：HBM budget table、KV block capacity、reserved linear bytes、可服务请求估算、OOM 风险解释。

### 2.5 Incident Triage：从流程文档升级为 replay-first 工具

当前 `xllm-npu-incident-triage` 有流程和错误 catalog，但缺少统一的
`incident_artifact_tool.py` 与 replay helper。

缺口：

- 没有统一 collect bundle：health、server info、load、metrics、npu-smi、dmesg、xLLM logs。
- 没有 request dump / crash dump summarize 和 replay wrapper。
- 没有 decision tree 的可执行入口。

应实现：

- `skills/xllm-npu-incident-triage/scripts/incident_artifact_tool.py`
- `scripts/replay_trusted_request_dump.py`
- `references/decision-tree.md`
- `references/endpoints-and-signals.md`
- `references/replay-trace-profile.md`
- 将 NPU error catalog 与 CANN/HCCL/GE/AclGraph 错误码绑定到排障下一步。

### 2.6 Code Review：从规则清单升级为多框架 review corpus + query

当前 `xllm-npu-code-review` 已有审查维度，但缺少类似 `sglang-humanize-review` 的 human review corpus。

缺口：

- 没有收集 xLLM / vLLM-Ascend / SGLang NPU 相关历史 review comments、PR discussions、maintainer 风格。
- Agent 审查仍依赖规则，而不是历史真实 review 模式。
- 缺少 query 工具和测试，无法保障审查语料质量。

应实现：

- `skills/xllm-npu-code-review/scripts/collect_npu_review_corpus.py`
- `scripts/query_npu_review_corpus.py`
- `references/npu-review-corpus-*.jsonl.gz`
- `references/corpus-summary.md`
- tests：语料字段、路径、comment 分类、典型 review pattern。

### 2.7 Model PR History：从少量 xLLM markdown 扩展为多框架 PR-driven dossier 系统

当前已有 Qwen3、DeepSeek、GLM 档案和 `query.py`。

缺口：

- 档案数量太少，且当前基本集中在 xLLM。
- 缺少自动从 git / GitHub PR 重建 history 的工具。
- 缺少统一 card schema 和质量测试。
- 缺少中英双语 README 结构。

应实现：

- `tools/rebuild_npu_model_pr_history_from_git.py`
- `model-pr-optimization-history/{xllm,vllm-ascend,sglang}/<model>/README.zh.md`
- `README.en.md`
- `skills/model-optimization/model-pr-diff-dossier/references/card-schema.md`
- tests：
  - 每个 dossier 有 framework、PR、files、symbols、risk、validation notes。
  - query 工具能按 framework/model/keyword/path 返回稳定结果。

优先模型：

- Qwen3 / Qwen3-Next / Qwen3.5
- DeepSeek-V3/R1/V3.1/V3.2
- GLM-4.5/4.6/5
- Kimi-k2
- Qwen-VL / Qwen3-VL
- MoE / MTP / Gated Delta Net 相关路径。

### 2.8 Humanize / Goal：补真正的 run-level orchestrator

当前有 `humanize/attempt-ledger.md`、`optimization-ledger.md`、`lineage.jsonl`，但多为手写账本。

缺口：

- 没有固定 run manifest schema。
- 没有自动创建 run root、写入 phase 状态、记录 lineage 的工具。
- 没有 Codex goal prompt 模板和固定 SOTA loop prompt。
- 没有“固定公平 benchmark 在 loop 外，gap/profiler/patch/revalidation 在 loop 内”的机器校验。

应实现：

- `skills/xllm-npu-sota-loop/references/refined-plan-template.md`
- `prompts/xllm-npu-sota-a3-prompts.md`
- `prompts/xllm-npu-sota-a3-codex-goal-prompts.md`
- `scripts/xllm_npu_sota_run.py`
- `references/run-manifest-schema.md`
- `humanize/lineage.schema.jsonl.md`

### 2.9 Testing / CI：补仓库可信度底座

当前 NPU 仓库没有测试目录和本地质量检查配置。

缺口：

- 脚本改动没有回归保护。
- Markdown 结构和 skill 输出契约没有检查。
- cookbook config、schema、query 工具无法自动验证。

应实现：

- `tests/test_compare_npu_benchmark.py`
- `tests/test_collect_evalscope_results.py`
- `tests/test_npu_profile_parser.py`
- `tests/test_render_triage_npu.py`
- `tests/test_model_pr_dossier_quality.py`
- `tests/test_cookbook_configs.py`
- `.pre-commit-config.yaml`、`.codespellrc`、ruff/isort 配置。

## 3. 推荐实现路线

### Phase 0：仓库工程化底座（1 周）

目标：先让现有脚本和文档可测试、可验证。

交付：

- 增加 `tests/` 和基础 CI 本地命令。
- 为 benchmark result、run manifest、lineage 定义 schema。
- 给现有 benchmark/profiler 脚本补最小单测。
- 给 README 增加“当前能力 / 未实现能力 / roadmap”。

成功标准：

- `pytest tests` 可在无 NPU 环境下跑通 parser/schema/query 类测试。
- 现有脚本输出字段被测试固定下来。

### Phase 1：Benchmark cookbook 与公平搜索（1-2 周）

目标：把“收集结果”升级为“多框架配置驱动候选生成 + 结果比较”。

交付：

- `cookbook-npu-llm` 初版：Qwen3.5-27B、Qwen3-Next、DeepSeek、GLM。
- `frameworks/xllm`、`frameworks/vllm-ascend`、`frameworks/sglang` 初版适配说明。
- `validate_cookbook_configs.py`。
- result-schema 和 compare 脚本增强。
- 输出 `winning-commands.md`、`candidates.jsonl`、failure rows。

成功标准：

- 能在无 NPU 环境渲染 xLLM / vLLM-Ascend / SGLang NPU 候选命令。
- 在真实 A3 环境用 evalscope 跑完至少一个模型 Tier 1/Tier 2。

### Phase 2：NPU Profiler 深化（2 周）

目标：五表报告从 summary 走向可定位 layer/kernel/rank 的证据栈。

交付：

- `xllm-npu-pipeline-analysis` skill。
- layer timeline、kernel breakdown、rank skew 脚本。
- profiler renderer 增加 layer deep dive 和 rank skew table。

成功标准：

- 对 Qwen3.5-27B MTP trace 能定位 representative layer。
- 能输出 Prefill/Decode 分离的 kernel + layer + rank 三层证据。

### Phase 3：Compute / Capacity 双分析（2 周）

目标：补齐“理论上限”和“容量预算”。

交付：

- `xllm-npu-compute-simulation`。
- `xllm-npu-capacity-planner`。
- A3 specs、模型 config index。
- MFU / HBM / KV block / MTP reserve 报告模板。

成功标准：

- 对一个真实 trace 输出 kernel-level MFU。
- 对一个 xLLM 启动日志输出 KV/HBM capacity table。

### Phase 4：Incident + Review Corpus（2 周）

目标：把排障和审查从经验文档变成可查询工具。

交付：

- incident artifact tool、replay helper、decision tree。
- xLLM review corpus 采集和查询工具。
- NPU code-review 增加 corpus-grounded workflow。

成功标准：

- 能从一次 xLLM serving crash/hang 生成 artifact bundle 和 next-step report。
- code-review 能引用本地 corpus 中的历史模式，而不只是泛泛规则。

### Phase 5：PR History 自动化与模型档案扩展（持续 2-4 周）

目标：让 Learn 阶段真正有历史记忆。

交付：

- 自动重建 xLLM model PR history 工具。
- 至少 10 个模型家族 dossier。
- dossier quality tests。

成功标准：

- 新模型优化前可以 query 到相关 PR、文件、符号、风险面。
- 每个 dossier 都能回答“哪些 PR 改过、改了什么、风险是什么、下一步该查哪里”。

### Phase 6：SOTA Loop Orchestrator（2 周）

目标：把前面所有能力串成单次 run 的闭环。

交付：

- `xllm_npu_sota_run.py` 创建 run root、manifest、phase 状态。
- 自动写 `attempt-ledger.md`、`optimization-ledger.md`、`lineage.jsonl`。
- Codex goal prompt 模板。
- refined plan template。

成功标准：

- 对一个模型可以从 Phase 0 到 Phase 5 生成完整 artifacts。
- 每轮 patch 有 benchmark、profiler、review、record 的 lineage。

## 4. 优先级建议

最高优先级：

1. tests/schema/CI 底座。
2. benchmark cookbook + result schema。
3. profiler layer/rank deep dive。
4. run manifest + humanize lineage。

中优先级：

1. compute simulation。
2. capacity planner。
3. incident artifact tool。
4. model PR history 自动化。

低优先级：

1. architecture diagram skill。
2. 大规模双语档案美化。
3. 多框架泛化到除 xLLM/vLLM-Ascend 外的更多 NPU serving stack。

## 5. 结论

`pjgao/xllm-npu-optimization-skills` 当前已经形成了公平 benchmark、证据化
profiling、RLCR、review、history、kernel pilot 的主线。但它现在更像“以
xLLM 为首个落地样本的 NPU 工作流蓝图 + 少量工具”，下一阶段应该优先把它
补成“可测试、可运行、可积累历史证据、可适配 xLLM / vLLM-Ascend /
SGLang 的 NPU 优化平台”。

最现实的实现顺序是：先工程化和 schema，再 benchmark cookbook，再 profiler 深挖，然后补 compute/capacity/incident/review/history，最后用 orchestrator 把这些能力合成真正的 xLLM NPU SOTA loop。
