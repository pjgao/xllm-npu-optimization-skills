# Agent 使用指南

本仓库是面向华为昇腾 NPU 910B3 (A3) 的大模型推理与 AI Infra 开发
skill 集合。当前最完整的落地对象是京东 xLLM，但标准流程应服务于
xLLM、vLLM-Ascend、SGLang NPU 后端等多框架。Agent 在协助任何 NPU
推理优化任务时，**必须遵循本仓库的 evidence-driven 闭环流程**。

## 仓库定位

- 仓库目标：沉淀 NPU 大模型推理和 AI Infra 开发的证据驱动标准流程
- 框架范围：xLLM、vLLM-Ascend、SGLang NPU 后端
- 对照原则：按任务选择对照框架；默认优先做 xLLM vs vLLM-Ascend，也允许扩展到 SGLang NPU
- 目标模型：NPU serving 上的主流推理模型（Qwen3 / DeepSeek-V3 / GLM-5 / Llama / Kimi 等）
- 目标硬件：昇腾 910B3 (A3)，HDK Driver 25.2.0+，CANN 8.0.RC1+
- 当前经验底座：xLLM + Qwen3.5-27B + MTP 的真实 benchmark、profiling、patch、事故记录

## 核心原则（必须遵守）

1. **unfair 数据不比较**：benchmark 阶段必须让每个参测框架各自独立搜索最优配置（严禁用一方的最优参数套另一方）
2. **没 profiling 不动手**：Phase 3 报告不存在前，不允许开始任何 patch
3. **每步决策有数据支撑**：所有判定必须基于五表中的可复现指标
4. **RLCR 闭环**：Research → Learn → Code → Review → Validate → Record，严禁"写完就合"
5. **经验不得丢失**：通用化时不得删除已有 xLLM/Qwen3.5/MTP 经验；失败实验、反例和环境信息也必须保留

## Skills 总览

| Skill | 路径 | 何时加载 |
|-------|------|---------|
| xllm-npu-benchmark | `skills/xllm-npu-benchmark/SKILL.md` | 需要对比 xLLM / vLLM-Ascend / SGLang NPU 性能时 |
| xllm-npu-profiler | `skills/xllm-npu-profiler/SKILL.md` | 需要定位 NPU 性能瓶颈、生成五表报告时 |
| xllm-npu-sota-loop | `skills/xllm-npu-sota-loop/SKILL.md` | 端到端驱动 NPU SOTA 优化闭环、判断优化是否达标时 |
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
  ├─ 目标框架胜出 / 平局 → 写 final_summary.md → 结束
  └─ 目标框架落后 > 1% ↓
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
- SGLang NPU 本地仓库：按实际机器记录到 `$RUN_ROOT/manifest.md`
- 模型权重目录：`/models/`（如 `/models/Qwen3.5-27B`）
- 每次 SOTA run 输出目录：`/home/gaopengju/runs/<date>_<model>_npu_sota/`

## 启动新模型优化前必做

1. 查对应框架的 `model-pr-optimization-history/<framework>/` 看是否已有该模型档案
2. 没有则查 PR 历史并新建档案；当前 xLLM 档案可按 `qwen3-core.md` 格式扩展
3. 新建 `$RUN_ROOT/manifest.md` 记录本次 run 的框架 commit hash / CANN 版本 / NPU 型号 / workload / SLA

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
export PROFILING_MODE=dynamic
ps -ef | grep xllm
MODEL=Qwen35-27B TOKENIZER=/home/data/weights/Qwen35-27B PORT=8080 \
  skills/xllm-npu-profiler/scripts/run_profiling.sh <xllm_parent_pid> profiles/xllm full

python skills/xllm-npu-profiler/scripts/analyze_xllm_npu_profile.py \
  --input profiles/xllm_YYYYMMDD_HHMMSS/PROF_xxx --framework xllm --output profiles/xllm-analysis.json

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

## 多框架 benchmark 模板（A3）

```bash
# xLLM
xllm serve /models/<MODEL> \
  --tensor-parallel-size 4 \
  --graph-mode npugraph_ex \
  --block-size 128 \
  --port 8080

# vLLM-Ascend
VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve /models/<MODEL> \
  --tensor-parallel-size 4 \
  --enforce-eager \
  --block-size 128 \
  --gpu-memory-utilization 0.9 \
  --port 8000

# SGLang NPU（示例，实际参数以后端支持为准）
python -m sglang.launch_server \
  --model-path /models/<MODEL> \
  --tp 4 \
  --host 0.0.0.0 \
  --port 30000
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

- ❌ 拿任意框架 default 直接当 baseline，让另一框架单方面追赶
- ❌ 没看五表就开始写 patch
- ❌ 多个 patch 一起上、不单独验证
- ❌ 修了 bug 但不更新 incident 台账
- ❌ 为了通用化重写文档时删掉已有 xLLM/MTP 实测经验
