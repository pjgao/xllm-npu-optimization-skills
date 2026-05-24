---
name: xllm-npu-benchmark
description: 在华为昇腾 NPU 910B3 (A3) 上进行 xLLM 与 vLLM-Ascend 的公平推理基准测试。当用户需要对比两个框架在相同模型/工作负载/NPU/SLA 条件下的性能时使用。支持分层搜索、JSONL 数据集、SLA 验证、CSV/Markdown 导出。
---

# xLLM NPU 基准测试

当用户需要在华为昇腾 NPU 910B3 (A3) 上公平对比 xLLM 与 vLLM-Ascend 时使用。

工作流入口：
- [scripts/collect_evalscope_results.py](scripts/collect_evalscope_results.py) — 收集 evalscope 原始产物并归一化为 JSONL
- [scripts/compare_npu_benchmark.py](scripts/compare_npu_benchmark.py) — 结果对比
- [scripts/validate_framework_cli.py](scripts/validate_framework_cli.py) — CLI 验证

## 前置条件

- xLLM 和 vLLM-Ascend 均可正常启动并服务目标模型
- 模型权重已下载到可见路径
- 目标明确：
  - 固定 QPS 基准测试，或
  - 搜索满足 `max_ttft_ms` / `max_tpot_ms` SLA 的最大 QPS

如果以上条件不满足，先解决后再启动大规模搜索。

环境一致性检查：
- 验证 `npu-smi info` 可见所有 NPU A3 设备且状态健康
- 验证 `ASCEND_RT_VISIBLE_DEVICES` 设置一致
- 确认 CANN >= 8.0.RC1，HDK Driver >= 25.2.0
- 验证两个框架 CLI `--help` 输出，确认关键参数可用

## 实测案例 (Qwen3.5-27B, 910B3 x2 TP=2, 2026-05-23)

**环境**: 192.168.13.154 / xllm-gpj 容器 / CANN 8.5.0 / torch_npu 2.7.1.post2  
**模型**: `/home/data/weights/Qwen35-27B` (64 层混合注意力) + MTP draft  
**Benchmark**: evalscope `line_by_line` plugin + `jd_openai_20k.jsonl` (20k 真实对话 + `max_tokens=2048, temperature=0.0, stream=true`)  
**结果**:

| 并发 | 总请求 | Output Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Avg Output Tokens | 成功 |
|-----|--------|--------------------------|-----------|-----------|------------------|------|
| 1   | 5      | 29.33                    | 3564      | 29.8      | 917              | 5/5  |
| 2   | 4      | 46.83                    | 4445      | 34.4      | 1308             | 4/4  |

结论：并发 2 相比并发 1，吞吐 +59.7% (46.83 vs 29.33 tok/s)。

详细结果：`benchmark/baseline/parallel_1_number_5/20260523_150528/` 和 `benchmark/baseline/parallel_2_number_4/20260523_151051/`。

## 工作流

### Step 1: Preflight

```bash
npu-smi info
# 确认 NPU A3 设备数量、显存、驱动版本

python scripts/validate_framework_cli.py \
  --framework xllm \
  --model /path/to/model \
  --extra-flags "--tensor-parallel-size 4"

python scripts/validate_framework_cli.py \
  --framework vllm-ascend \
  --model /path/to/model \
  --extra-flags "--tensor-parallel-size 4 --enforce-eager"
```

保存环境信息到 `artifact/env.json`：
- NPU 型号、数量、驱动版本
- CANN 版本
- 框架 commit hash、Docker 镜像
- `ASCEND_RT_VISIBLE_DEVICES`

### Step 2: 规范化工作负载

统一 JSONL 格式，每行一个请求：

```json
{"prompt": "请总结这篇文章的要点。", "output_len": 256}
{"prompt": [{"role": "user", "content": "解释量子纠缠。"}], "output_len": 512}
```

可选字段：

```json
{
  "prompt": [{"role": "user", "content": "使用天气工具"}],
  "output_len": 256,
  "extra_request_body": {"temperature": 0.0, "top_p": 0.95},
  "timestamp": 1710000000,
  "metadata": {"source": "custom"}
}
```

### Step 2.5: evalscope 实测用法 (推荐)

使用 `evalscope` 客户端 (容器内已安装) 进行端到端 benchmark。支持 OpenAI 兼容 API 的 xLLM/vLLM-Ascend 实例。

**数据集格式** (JSONL, 每行是一个完整的 OpenAI 请求 body):

```json
{"model": "Qwen35-27B", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 2048, "temperature": 0.0, "stream": true}
{"model": "Qwen35-27B", "messages": [{"role": "system", "content": "你是助手"}, {"role": "user", "content": "总结..."}], "max_tokens": 2048, "temperature": 0.0, "stream": true}
```

**命令**:

```bash
evalscope perf \
  --model Qwen35-27B \
  --url http://127.0.0.1:8080/v1/chat/completions \
  --api openai \
  --dataset line_by_line \
  --dataset-path /path/to/jd_openai_20k.jsonl \
  --parallel 1 \
  --number 5 \
  --connect-timeout 120 \
  --read-timeout 300 \
  --outputs-dir /path/to/results/
```

**输出**:
- `benchmark_summary.json` — 汇总指标 (throughput, TTFT, TPOT, etc.)
- `benchmark_percentile.json` — 延迟百分位(P50/P99)
- `benchmark_data.db` — SQLite 请求明细
- `benchmark.log` — 日志
- `perf_report.html` — HTML 报告

**归一化为 compare 输入**:

```bash
python scripts/collect_evalscope_results.py \
  --root /path/to/evalscope/results \
  --framework xllm \
  --output-jsonl /path/to/xllm_results.jsonl \
  --output-summary /path/to/xllm_summary.md
```

该脚本会递归扫描 `benchmark_summary.json`，读取同目录的
`benchmark_percentile.json`，并输出 `compare_npu_benchmark.py` 可直接消费的
JSONL。可选 `--sla-ttft-ms` / `--sla-tpot-ms` 用于把 p99 延迟 SLA 写入
`sla_pass`。

**参数说明**:
- `--dataset line_by_line`: 使用 line-by-line plugin，逐行发送 JSONL 中的完整请求 body
- `--parallel N`: 并发数
- `--number N`: 总请求数
- `--connect-timeout 120`: 连接超时(秒), xLLM cold start 需要更长时间
- `--read-timeout 300`: 读取超时(秒), 长序列生成需要更长时间

默认场景：

| 场景 | input_len | output_len | 说明 |
|------|-----------|------------|------|
| chat | 1000 | 1000 | 常规对话 |
| summary | 8000 | 1000 | 长文总结 |
| long-context | 32000 | 1000 | 超长上下文 |

所有框架使用相同的：
- tokenizer 路径
- 精度（bf16 或指定量化）
- 量化方案（若启用）
- 采样参数（temperature=0 等）

### Step 3: 搜索层级

| 层级 | 说明 | 候选数上限 |
|------|------|-----------|
| Tier 1 (smoke) | 快速验证，少量候选 | ≤3 |
| Tier 2 (default) | 有界扫描，高优先级参数优先 | ≤10 |
| Tier 3 (exhaustive) | 穷举搜索 | ≤30 |

### Step 4: xLLM 调优

调优维度：

**并行策略**：
- `--tensor-parallel-size`: TP 大小
- `--pipeline-parallel-size`: PP 大小
- `--expert-parallel-size`: EP 大小（MoE 模型）

**图模式**：
- `--graph-mode`: `eager` / `npugraph_ex` / `ge`
- Prefill 默认 eager，Decode 默认 npugraph_ex

**KV Cache**：
- `--block-size`: Block 大小（PA 模式）
- `--max-model-len`: 最大模型长度
- KV Cache NZ 格式

**调度**：
- `--max-num-seqs`: 最大并发序列数
- `--chunked-prefill-size`: 分块 prefill 大小
- PD 分离策略

**内存**：
- `--gpu-memory-utilization`: GPU 显存利用率
- xTensor 内存池参数

**算法**：
- `--speculative-model`: 投机解码 draft 模型
- `--eplb-strategy`: 动态 EPLB 策略

记录每次运行的完整启动命令。保留失败候选及原因（如 OOM：建议增加 NPU 卡数或使用更大显存）。

### Step 5: vLLM-Ascend 调优

相同工作负载和 SLA 下调优：

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve /path/to/model \
  --tensor-parallel-size 4 \
  --enforce-eager \
  --max-model-len 8192 \
  --block-size 128 \
  --gpu-memory-utilization 0.9 \
  --port 8000
```

调优维度：
- TP/PP/KV cache 配置
- graph mode 开关
- prefix caching
- chunked prefill

### Step 6: 结果归一化

调用 `compare_npu_benchmark.py`：

```bash
python scripts/collect_evalscope_results.py \
  --root /path/to/xllm/evalscope/results \
  --framework xllm \
  --output-jsonl /path/to/xllm_results.jsonl

python scripts/collect_evalscope_results.py \
  --root /path/to/vllm/evalscope/results \
  --framework vllm-ascend \
  --output-jsonl /path/to/vllm_results.jsonl

python scripts/compare_npu_benchmark.py \
  --xllm-results /path/to/xllm_results.jsonl \
  --vllm-results /path/to/vllm_results.jsonl \
  --output-dir /path/to/comparison/
```

排序规则：SLA通过 > 请求吞吐 > 输出token吞吐 > p50 TTFT > p50 TPOT

输出：
- `comparison/summary.md` — Markdown 对比表
- `comparison/summary.csv` — CSV 数据
- `comparison/winning-commands.md` — 最优配置命令
- `comparison/candidates.jsonl` — 所有候选记录

## 公平性规则

详见 [references/npu-fairness-rules.md](references/npu-fairness-rules.md)。

核心原则：
- 相同 NPU 型号 (A3) / 卡数 / 模型权重 / tokenizer / 精度 / 量化 / 采样设置
- 记录 CANN 版本、HDK Driver 版本、框架 commit、Docker 镜像
- 候选之间重启服务或清除状态
- `ASCEND_RT_VISIBLE_DEVICES` 设置一致
- 保留失败候选及失败原因
- 永远不要将 tuned xLLM 与 vLLM-Ascend defaults 比较，或反之

## 配置文件

示例 `qwen3-32b-a3.yaml`：

```yaml
model:
  path: /models/Qwen3-32B
  tokenizer: /models/Qwen3-32B
  precision: bf16

npu:
  device: A3
  count: 4
  visible_devices: "0,1,2,3"

dataset:
  kind: random
  scenarios:
    - name: chat
      input_len: 1000
      output_len: 1000
    - name: summary
      input_len: 8000
      output_len: 1000

benchmark:
  num_prompts: 80
  qps:
    search: true
    max_rounds: 5
  sla:
    max_ttft_ms: 500
    max_tpot_ms: 50

search:
  tier: 2
  max_candidates: 8

frameworks:
  xllm:
    base_flags:
      tensor-parallel-size: 4
      graph-mode: npugraph_ex
      block-size: 128
    search_space:
      chunked-prefill-size: [256, 512, 1024]
      max-num-seqs: [64, 128, 256]
      speculative-model: ["", "/models/draft-2b"]
  vllm-ascend:
    base_flags:
      tensor-parallel-size: 4
      enforce-eager: true
      block-size: 128
      gpu-memory-utilization: 0.9
    search_space:
      enable-prefix-caching: [true, false]
```

## 中断与恢复

使用 `search.resume: true` 恢复中断的搜索：

```yaml
search:
  tier: 2
  resume: true
```

- 每次试验结果追加到 `live_results.jsonl`
- SIGINT/SIGTERM 会保存已有结果
- Resume 假设候选顺序和数据集未变

## 返回值

运行完成后返回：
- 使用的层级
- 数据集类型（合成/真实流量）
- xLLM 最优配置及性能
- vLLM-Ascend 最优配置及性能
- 是否满足 SLA
- 最佳 QPS
- 文件路径：prepared dataset JSONL、results.jsonl、results.csv、关键日志

## 实测脚本: bench.sh

封装的 benchmark 脚本，支持 baseline / MTP 模式切换：

```bash
#!/bin/bash
MODE=${1:-baseline}
PARALLEL=${2:-1}
NUMBER=${3:-5}
RUN_ROOT=/home/g00510989/runs/20260523_qwen35_27b_npu_sota
DATASET=$RUN_ROOT/datasets/jd_openai_20k.jsonl

if [ "$MODE" = "mtp" ]; then
  URL=http://127.0.0.1:18170/v1/chat/completions
else
  URL=http://127.0.0.1:18160/v1/chat/completions
fi

OUT_DIR=$RUN_ROOT/benchmark/${MODE}/parallel_${PARALLEL}_number_${NUMBER}
mkdir -p $OUT_DIR

evalscope perf \
  --model Qwen35-27B \
  --url $URL \
  --api openai \
  --dataset line_by_line \
  --dataset-path $DATASET \
  --parallel $PARALLEL \
  --number $NUMBER \
  --connect-timeout 120 \
  --read-timeout 300 \
  --outputs-dir $OUT_DIR
```

**用法**:

```bash
./bench.sh baseline 1 5    # baseline 模式, 并发 1, 5 请求
./bench.sh mtp 2 4         # MTP 模式, 并发 2, 4 请求
```
