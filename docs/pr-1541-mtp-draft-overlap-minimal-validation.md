# PR #1541 MTP Draft Extend Preparation Overlap 最小化验证记录

日期：2026-05-25

## 背景

本轮验证目标是在 Preview 分支最新代码已回退 TileLang 相关修改的前提下，把 PR #1541 的修改最小化到 MTP draft extend preparation overlap 相关代码，并验证：

- 是否能正常构建；
- 10 条精度数据是否无异常；
- random 20k 输入 / 1k 输出场景性能是否劣化；
- 构建和验证过程中暴露的环境问题如何规避。

本地验证分支：

- 仓库：`/home/g00510989/xllm/xllm_pr1541_minimal`
- 分支：`pr1541-minimal`
- 提交：`eaff9517 perf: overlap mtp draft extend preparation`
- 主要改动：`xllm/core/runtime/mtp_worker_impl.{cpp,h}`

## 构建问题与经验

正常预期构建命令应是：

```bash
python setup.py build --device npu
```

本轮构建时间异常偏长，根因不在本次 PR 业务代码，而是环境和缓存问题叠加。

### 1. OPP 头文件与源码不匹配

现象：编译 `beam_search` 相关代码时，系统 OPP 头文件中的 `aclnnBeamSearchGroup` 签名仍是旧版本，包含额外 `topK` 参数；源码侧期望的是新签名。

检查方式：

```bash
grep -n "aclnnBeamSearchGroup" /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/xllm/op_api/include/aclnn_beam_search_group.h
```

处理经验：

- 不直接覆盖系统 `/usr/local/Ascend/.../opp/vendors/xllm`。
- 本轮使用 `/tmp/xllm_ops_pr1541/vendors/xllm` 作为临时 OPP vendor，并通过 `CMAKE_ARGS` 指向匹配的 header/lib。
- 验证报告中必须标注这种规避方式，避免把环境问题误读成代码问题。

### 2. build cache 命中错误架构 libtorch

现象：链接时报错：

```text
_deps/libtorch-src/lib/libtorch.so: file in wrong format
```

原因：`build/cmake.linux-aarch64-cpython-311/_deps/libtorch-src` 中缓存的是 x86-64 libtorch，而当前机器需要 aarch64 libtorch。

检查方式：

```bash
file build/cmake.linux-aarch64-cpython-311/_deps/libtorch-src/lib/libtorch.so
file /usr/local/lib64/python3.11/site-packages/torch/lib/libtorch.so
```

处理经验：

- 只修正 build artifact，不改源码。
- 本轮将错误架构缓存备份为 `.x86_64_bak`，并把 `_deps/libtorch-src` 指向系统 Python 环境中的 aarch64 torch。

### 3. Python.h 缺失

现象：编译扩展时找不到 `Python.h`。

处理经验：

```bash
export CPLUS_INCLUDE_PATH=/usr/include/python3.11
```

### 4. TileLang 目标重新生成导致构建时间拉长

虽然本次 PR 不依赖 TileLang 修改，但构建过程中可能触发已有 TileLang 目标或小算子重新生成，导致单次 build 明显慢于预期。

经验判断：

- 如果只改 MTP runtime 逻辑，长时间重编 TileLang 目标通常是构建缓存问题；
- 需要优先检查 build cache、OPP vendor 和 libtorch 架构，而不是先怀疑 PR 本身引入了大规模编译依赖。

## 验证方法

### 构建与单测

- build：通过
- `sampler_test`：13 个用例中 11 passed，2 skipped（MLU-only）

### 服务启动

物理 NPU：8, 9, 10, 11

关键启动参数：

```bash
ASCEND_RT_VISIBLE_DEVICES=8,9,10,11
--tensor-parallel-size 4
--communication_backend=lccl
--enable_chunked_prefill=true
--max_tokens_per_chunk_for_prefill=256
--enable_schedule_overlap=true
--enable_graph=true
--enable_prefix_cache=false
--num_speculative_tokens 3
--max_tokens_per_batch=32768
--max_seqs_per_batch=16
--block_size=128
```

模型：

- target：`/home/data/weights/Qwen35-27B`
- draft：`/home/data/weights/Qwen35-27B-mtp`

短请求 smoke：通过。

### 性能 workload

使用 evalscope random 20k/1k：

- `parallel=1`
- `number=5`
- `temperature=0.0`
- `stream=true`
- 输入 token 目标：20k
- 输出 token：1k

结果目录：

```text
/home/g00510989/xllm/runs/20260525_pr1541_minimal_eaff9517/perf/random20k_1k_p1_n5/20260525_152615/Qwen35-27B
```

### 精度 workload

使用 GSM8K `limit=10`：

```text
/home/g00510989/xllm/runs/20260525_pr1541_minimal_eaff9517/accuracy/gsm8k_limit10
```

报告：

```text
/home/g00510989/xllm/runs/20260525_pr1541_minimal_eaff9517/accuracy/gsm8k_limit10/reports/Qwen35-27B/gsm8k.json
```

## 验证结果

### 性能

| 指标 | 结果 |
|------|------|
| Success | 5/5 |
| Avg Latency | 15.41s |
| TTFT | 2350.46ms |
| TPOT | 13.07ms |
| Output Throughput | 63.26 tok/s |
| Total Throughput | 1255.16 tok/s |
| Decode TPS | 76.51 tok/s |
| Decoded Tok/Iter | 3.08 |
| Spec Accept Rate | 67.5% |
| Avg Input Tokens | 18840.00 |
| Avg Output Tokens | 1000 |
| P99 Latency | 16.75s |
| P99 TTFT | 3000.62ms |
| P99 TPOT | 14.06ms |

说明：evalscope random 虽设置 20k 输入，但经过 chat template/tokenizer 后，本轮实际平均输入 token 为 18840，p99 接近 20000。

### 精度

| Workload | Result |
|----------|--------|
| GSM8K limit=10 | 10/10 correct |
| mean_acc | 1.0 |

GSM8K 10 条数据未发现精度异常。

## 性能是否劣化

### 对比 2026-05-24 MTP=3 基线

基线来自 skill 中记录的 Qwen3.5-27B TP=4 random 20k/1k MTP 扫描结果。

| 指标 | PR #1541 最小化 | MTP=3 基线 | 变化 |
|------|----------------|------------|------|
| Avg Latency | 15.41s | 14.564s | +5.8% |
| TTFT | 2350.46ms | 2524.8ms | -6.9% |
| TPOT | 13.07ms | 12.05ms | +8.5% |
| Output TPS | 63.26 | 66.82 | -5.3% |
| Decode TPS | 76.51 | 82.99 | -7.8% |
| Decoded Tok/Iter | 3.08 | 3.21 | -4.0% |
| Accept Rate | 67.5% | 68.8% | -1.3pp |

结论：相比历史 MTP=3 基线，decode 侧存在轻微性能回落信号，主要体现在 TPOT 变差、Output TPS 和 Decode TPS 下降；TTFT 反而更好。当前结果不能描述为大幅劣化，但应标记为可疑轻微劣化。

### 对比 no MTP 基线

| 指标 | PR #1541 最小化 MTP=3 | no MTP 基线 | 变化 |
|------|----------------------|-------------|------|
| Avg Latency | 15.41s | 21.155s | -27.2% |
| TTFT | 2350.46ms | 2507.1ms | -6.3% |
| TPOT | 13.07ms | 18.67ms | -30.0% |
| Output TPS | 63.26 | 46.39 | +36.4% |

结论：MTP=3 仍显著优于 no MTP。

### 追加源码与短 decode 复核

继续审视后，当前提交不能称为 vLLM async schedule 同类优化。代码实际做的是：

1. 在 target validate `step_async()` 后创建 `DraftExtendPrepareContext`；
2. 在 `future.get()` 前提前执行部分 `base_input.to(device_, dtype_)` 和 `token_ids` / `positions` / `block_tables` 的 CPU view/copy；
3. target validate 返回后再根据 `validate_output.next_tokens` 和 `validate_output.embeddings` 完成 `finish_draft_extend_inputs()`；
4. 真正的 `draft_impl_->step_async(extend_input)` 仍然发生在 target 完成之后。

因此该提交没有实现“draft 模型提前下发计算”。它只把一部分 host/input 准备工作挪到 target future 等待之前，最多能覆盖很薄的一段准备时间；如果 `safe_to(..., CPU)` 或 `ForwardInput::to()` 在 target 计算期间触发 D2H/H2D 拷贝、默认 stream 依赖或 runtime 同步，反而会干扰 target 计算。

本轮补跑了一个 decode-focused smoke，用于和历史 profiling 形态对齐：

- Workload：random 32 input / 200 output，`parallel=1`，`number=1`
- 结果目录：`/home/g00510989/xllm/runs/20260525_pr1541_minimal_eaff9517/profiling/mtp3_minimal_decode32_out200_cards8_11_20260525_161500/evalscope/20260525_161923/Qwen35-27B`

| 版本 | Avg Latency | TTFT | TPOT | Output TPS | Decode TPS | Decoded/Iter | Accept |
|------|-------------|------|------|------------|------------|--------------|--------|
| before transpose profiling | 3.02s | 906.20ms | 10.62ms | 66.25 | 94.16 | 3.43 | 70.9% |
| transpose-opt profiling | 2.87s | 899.55ms | 9.90ms | 69.70 | 101.03 | 3.49 | 71.0% |
| PR #1541 minimal | 2.94s | 964.61ms | 9.94ms | 67.99 | 100.60 | 3.49 | 71.4% |

解读：

- 短 decode 场景没有出现灾难性退化，TPOT/Decode TPS 接近 transpose-opt profiling。
- 但它也没有证明调度提交带来收益：Output TPS 低于 transpose-opt profiling，TTFT 更差，且该场景 `number=1` 方差较大。
- 20k/1k 主验收 workload 仍显示 Output TPS、TPOT、Decode TPS 相比历史 MTP=3 基线回落；因此 PR 结论不变：不能作为有效性能优化合入。

msprof 动态采集尝试：

- 非沙箱启动服务成功，evalscope 请求成功。
- `msprof --dynamic=on --pid=<rank0>` 运行后未正常导出 `msprof_*.db`，只保留 evalscope 和 server logs。
- 后续若继续追，需要改用可控的 `msprof --application ...` 或确认 dynamic profiling 的 stop/export 流程，避免只得到交互式采集输出。

### 与早期 async draft overlap 全量实验的关系

早期全量实验中还包含非最小化修改，性能结果更好。当前最小化分支相对早期全量实验偏低，但该对比不是严格同环境：

- 本轮是最小化分支；
- 当前构建环境有 OPP header 和 libtorch cache 规避；
- 日志中仍可能出现 fused TileLang kernel 路径，需要确认 official Preview 分支的实际小算子状态；
- evalscope random 的实际输入 token 数有波动；
- `number=5` 单轮测试方差较大。

因此最终 PR 结论应采用 no-go 表述：已通过精度验证，但这只能说明结果没错，不能证明优化有效。该 PR 的核心卖点是 MTP draft preparation 提前下发/调度 overlap；当前数据没有证明该特性带来正收益，反而相比历史 MTP=3 基线有 decode 侧轻微回落信号。在缺少 profiling 解释和严格 A/B 正收益前，不应作为性能优化 PR 继续推进合入，应先 hold/draft。

## 后续建议

1. 先暂停合入/标记 draft，不再把该结果描述为已验证有效的性能优化。
2. 在 Preview 干净环境重新执行 `python setup.py build --device npu`，避免临时 OPP 和 libtorch cache 规避影响判断。
3. 用同一服务二进制分别跑 baseline commit 与 PR commit，固定 random seed、输入 token 统计和 NPU 映射。
4. 对 MTP=3 采集 profiling，重点看 draft extend preparation overlap 是否减少 host/device 空泡，同时确认是否引入新的同步点。
5. 若 profiling 不能证明空泡减少，或 A/B 仍无吞吐/TPOT 正收益，应撤回或重做该调度提交。
6. 若官方 Preview 已完全回退 TileLang 算子，验证日志中不应再依赖 fused TileLang kernel；否则需要把小算子状态作为结果解释的一部分。

## 2026-05-28 evalscope 复测记录

本轮按 skill 中的 evalscope 路径重新验证 TP=4、chunk prefill、MTP=3：

- 物理 NPU：12, 13, 14, 15
- target：`/home/data/weights/Qwen35-27B`
- draft：`/home/data/weights/Qwen35-27B-mtp`
- 关键参数：`--enable_chunked_prefill=true`、`--max_tokens_per_chunk_for_prefill=256`、`--enable_schedule_overlap=true`、`--enable_graph=true`、`--num_speculative_tokens 3`
- 性能目录：`/home/g00510989/xllm/runs/20260528_evalscope_validation/perf/mtp3_random20k_1k_p1_n5/20260528_075302/Qwen35-27B`
- 精度目录：`/home/g00510989/xllm/runs/20260528_evalscope_validation/accuracy/gsm8k_limit10_max2048_temp0`
- no-MTP 对照精度目录：`/home/g00510989/xllm/runs/20260528_evalscope_validation/accuracy/no_mtp_gsm8k_limit10_max2048_temp0`

### random 20k/1k 性能结果

| 指标 | 2026-05-28 MTP=3 | 2026-05-24 MTP=3 基线 | 变化 |
|------|------------------|-----------------------|------|
| Success | 5/5 | - | - |
| Avg Latency | 12.39s | 14.564s | -14.9% |
| TTFT | 2333.48ms | 2524.8ms | -7.6% |
| TPOT | 10.07ms | 12.05ms | -16.4% |
| Output TPS | 78.16 tok/s | 66.82 tok/s | +17.0% |
| Decode TPS | 99.30 tok/s | 82.99 tok/s | +19.7% |
| Decoded Tok/Iter | 3.36 | 3.21 | +4.7% |
| Spec Accept Rate | 70.2% | 68.8% | +1.4pp |

和历史 no-MTP 基线相比，MTP=3 仍有明显收益：Output TPS 78.16 vs 46.39（+68.5%），TPOT 10.07ms vs 18.67ms（-46.1%），Avg Latency 12.39s vs 21.155s（-41.4%）。

### GSM8K limit=10 精度复测

先按历史精度配置尝试 `max_tokens=30000`、`temperature=0.6`、`enable_thinking=true`，但首条样本耗时接近 593s，预计完整 10 条会超过 1 小时，因此中止并保留日志。

随后用 evalscope 同一 GSM8K `limit=10`，将生成配置收紧为 `max_tokens=2048`、`temperature=0.0` 做快速复核：

| 模式 | mean_acc | Avg Lat | TTFT | TPOT | Avg Out Tok |
|------|----------|---------|------|------|-------------|
| MTP=3 | 0.4 | 20.73s | 246.27ms | 11.12ms | 1834.0 |
| no MTP | 0.6 | 34.97s | 504.36ms | 18.27ms | 1886.3 |

这组精度结果不能作为“精度通过”证据，原因是两组平均输出 token 都接近 2048 上限，预测文件中能看到重复生成和截断迹象，答案抽取容易被截断影响。结论应表述为：evalscope 精度链路跑通，但当前 quick accuracy 配置无效；如果要给 PR 或优化结论背书，需要使用不截断的 GSM8K 配置、或选用输出更短且可稳定抽取的 10 条精度集重新验证。

### MTP 接受率与精度风险判断

本轮 random 20k/1k 的 MTP=3 接受率为 70.2%，历史 2026-05-24 MTP=3 基线为 68.8%，变化为 +1.4pp；Decoded Tok/Iter 为 3.36，历史基线为 3.21，变化为 +4.7%。这两个指标没有出现下降或异常抖动。

MTP 接受率是判断 draft/target 一致性的直接信号之一：如果主模型输出分布或 draft 校验路径出现明显精度问题，通常会先表现为 speculative accept rate、decoded token per iteration、或请求输出长度/停止原因异常。因此在本轮性能 workload 中接受率基本稳定，说明主模型精度大概率没有因为该调度/算子路径发生明显变化。

但接受率不能完全替代任务精度验证。最终表述建议为：MTP 接受率未见异常，支持“主模型精度大概率稳定”；GSM8K quick run 因输出截断和重复生成不能作为严格精度通过证据，后续仍需要一组不截断、可稳定抽取答案的 10 条精度验证作为 PR 背书。

### 复盘：为什么之前没有及时发现

1. 过度依赖 GSM8K `mean_acc`，没有先检查生成长度、截断率、预测文本和 stop reason。`max_tokens=2048` 的 quick run 看似完成了 10 条，但平均输出 token 已接近上限，预测文本存在重复生成，这种分数不能直接解释为模型精度变化。
2. 没有把 MTP 接受率作为精度 sanity check 的第一层信号。对 MTP 优化而言，accept rate 和 decoded token per iteration 比 GSM8K 小样本分数更贴近 draft/target 校验路径；接受率稳定时，应优先判断为“投机路径一致性未明显变坏”，再用任务精度补充确认。
3. 没有立即做 no-MTP 同配置对照。MTP=3 quick accuracy 为 0.4 时，只有和 no-MTP 同配置的 0.6 对照放在一起，才能看出主要问题可能来自评测生成配置/答案抽取，而不是直接归因于 MTP 改动。
4. 对历史精度配置的运行时成本预估不足。`max_tokens=30000 + enable_thinking` 在 GSM8K 上单样本可能接近 10 分钟，应提前标注这是严格验证配置，不适合临时 sanity；quick sanity 需要另选短输出数据集或限制样本。

后续避免方式：

- MTP 性能验证报告必须同时记录 `Spec Accept Rate`、`Decoded Tok/Iter`、输出 token 分布和请求成功率。
- 任何 `mean_acc` 异常都必须先检查预测文件、输出长度是否撞上 `max_tokens`、是否有重复生成或答案抽取失败，再判断是否是模型精度问题。
- 精度 sanity 至少包含同配置 no-MTP 对照；如果 MTP/no-MTP 都异常，优先排查 evalscope 配置、模板、采样参数和抽取逻辑。
- PR 结论里区分三类证据：接受率稳定、任务精度通过、性能收益成立，不能把其中一个替代另外两个。
