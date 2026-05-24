# Agent 使用指南

本仓库是专为京东 xLLM 推理框架在华为昇腾 NPU 910B3 (A3) 上做推理性能优化的 skill 集合。Agent 在协助 xLLM NPU 优化任务时，**必须遵循本仓库的 evidence-driven 闭环流程**。

## 仓库定位

- 对标仓库：AI-Infra-Auto-Driven-SKILLS（vLLM-Ascend 配套）
- 竞品参照：仅 vLLM-Ascend（不比对 MindIE / DeepSpeed）
- 目标模型：xLLM 上所有推理模型（Qwen3 / DeepSeek-V3 / GLM-5 / Llama 等）
- 目标硬件：昇腾 910B3 (A3)，HDK Driver 25.2.0+，CANN 8.0.RC1+

## 核心原则（必须遵守）

1. **unfair 数据不比较**：benchmark 阶段必须让 xLLM 和 vLLM-Ascend 各自独立搜索最优配置（严禁用一方的最优参数套另一方）
2. **没 profiling 不动手**：Phase 3 报告不存在前，不允许开始任何 patch
3. **每步决策有数据支撑**：所有判定必须基于五表中的可复现指标
4. **RLCR 闭环**：Research → Learn → Code → Review → Validate → Record，严禁"写完就合"

## Skills 总览

| Skill | 路径 | 何时加载 |
|-------|------|---------|
| xllm-npu-benchmark | `skills/xllm-npu-benchmark/SKILL.md` | 需要对比 xLLM vs vLLM-Ascend 性能时 |
| xllm-npu-profiler | `skills/xllm-npu-profiler/SKILL.md` | 需要定位 NPU 性能瓶颈、生成五表报告时 |
| xllm-npu-sota-loop | `skills/xllm-npu-sota-loop/SKILL.md` | 端到端驱动优化闭环、判断优化是否达标时 |
| xllm-npu-code-review | `skills/xllm-npu-code-review/SKILL.md` | 提交 NPU 特化代码前必须审查的 7 个维度 |
| xllm-npu-incident-triage | `skills/xllm-npu-incident-triage/SKILL.md` | xLLM 在 A3 上出现 crash / hang / OOM / 异常结果时 |
| kernel-pilot | `kernel-pilot/SKILL.md` | 所有现成优化路径用尽、需自研 NPU 算子时 |
| model-pr-optimization-history | `model-pr-optimization-history/SKILL.md` | 开始新模型优化前查询历史已做工作 |

## 标准工作流（6-Phase）

```
Phase 0     环境准备（NPU 健康、框架可用、模型就绪）
  ↓
Phase 0.5   查 PR 历史（model-pr-optimization-history）
  ↓
Phase 1     公平基准测试（xllm-npu-benchmark）
  ↓
Phase 2     差异判定
  ├─ xLLM 胜出 / 平局 → 写 final_summary.md → 结束
  └─ xLLM 落后 > 1% ↓
Phase 3     性能诊断（xllm-npu-profiler → 五表报告）
  ↓
Phase 4     构建 Plan（基于五表定位 + 优化路径优先级）
  ↓
Phase 5     RLCR 迭代（xllm-npu-sota-loop）
  ├─ Research → Learn → Code → Review(code-review) → Validate → Record
  └─ 回到 Phase 2
```

## 关键路径

- xLLM 本地仓库：`/home/gaopengju/projects/xllm`
- vLLM-Ascend 本地仓库：`/home/gaopengju/projects/vllm-ascend`
- 模型权重目录：`/models/`（如 `/models/Qwen3.5-27B`）
- 每次 SOTA run 输出目录：`/home/gaopengju/runs/<date>_<model>_npu_sota/`

## 启动新模型优化前必做

1. 查 `model-pr-optimization-history/xllm/` 看是否已有该模型档案
2. 没有则查 PR 历史并新建档案（按 `qwen3-core.md` 格式）
3. 新建 `$RUN_ROOT/manifest.md` 记录本次 run 的 commit hash / CANN 版本 / NPU 型号

## 常用脚本

```bash
# 基准对比
python skills/xllm-npu-benchmark/scripts/collect_evalscope_results.py \
  --root /path/to/evalscope/results \
  --framework xllm \
  --output-jsonl xllm.jsonl \
  --output-summary xllm-summary.md

python skills/xllm-npu-benchmark/scripts/compare_npu_benchmark.py \
  --xllm-results xllm.jsonl --vllm-results vllm.jsonl --output-dir comparison/

# Profiling 分析
python skills/xllm-npu-profiler/scripts/analyze_xllm_npu_profile.py \
  --framework xllm --url http://127.0.0.1:8080 --output-dir profiles/ --profile-by-stage

# 五表报告渲染
python skills/xllm-npu-profiler/scripts/render_triage_npu.py \
  --analysis-root profiles/ --output analysis/root-cause.md

# Model PR 历史查询
python model-pr-optimization-history/scripts/query.py --model "Qwen"

# Triage 报告渲染
python skills/xllm-npu-incident-triage/scripts/render_triage_npu.py \
  --artifacts ./artifacts --output ./triage-report-<id>.md

# 算子 benchmark
python kernel-pilot/tools/npu-op-benchmark.py --op swiglu --shapes "128,4096" --dtype float16
```

## xLLM 启动模板（A3）

```bash
# xLLM (baseline)
xllm serve /models/<MODEL> \
  --tensor-parallel-size 4 \
  --graph-mode npugraph_ex \
  --block-size 128 \
  --port 8080

# vLLM-Ascend (baseline)
VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve /models/<MODEL> \
  --tensor-parallel-size 4 \
  --enforce-eager \
  --block-size 128 \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

## 编译与测试

```bash
# xLLM NPU 构建
python setup.py build --device npu

# xLLM 单元测试
python setup.py test --device npu

# xLLM 端到端精度测试
python test/test_xllm_serve_generation.py --model /models/<MODEL> --device npu
```

## lint / typecheck

当前仓库为文档+脚本集合，主要 lint 为：

- Python 脚本：`ruff check skills/ model-pr-optimization-history/`
- Markdown：无强制 lint，但保持 80 字宽折行

## 何时停下来问用户

- 优化 gap 判定落在 ±1% 平局区：询问是否继续扩大优势或结束
- kernel-pilot 准入条件不全满足：是否强制启动（通常不建议）
- 优化路径冲突（如两个候选都要改同一文件）：让用户拍板优先级
- benchmark 显示 SLA 违反：询问是否放宽 SLA 或更换配置

## 反模式（禁止）

- ❌ 拿到 vLLM-Ascend default 直接当 baseline，让 xLLM 单方面追赶
- ❌ 没看五表就开始写 patch
- ❌ 多个 patch 一起上、不单独验证
- ❌ 修了 bug 但不更新 incident 台账
