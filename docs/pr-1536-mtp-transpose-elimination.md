# PR #1536：Qwen3.5 MTP CausalConv1d 复用与 Transpose 消除记录

> PR: https://github.com/jd-opensource/xllm/pull/1536  
> 目标分支: `preview/qwen3.5-qwen3.6`  
> 结论更新时间: 2026-05-29  
> 核心经验: MTP 路径的 transpose 问题，根因不是简单的 layout 局部冗余，而是 MTP verify 未复用非 MTP decode 已优化过的 `causal_conv1d` 算子路径。

## 1. 问题背景

Qwen3.5 的 GatedDeltaNet 路径里有一段 causal 1D convolution。非 MTP decode 路径已经使用 `causal_conv1d` 相关算子，并在图外/host 侧提前处理接口参数和权重 layout，因此 device 侧 transpose 较少。

MTP spec verify 路径之前走的是 `causal_conv1d_update` 路径。这个路径没有复用非 MTP decode 的 host 参数准备和权重 reshape 机制，导致 verify 阶段频繁出现 transpose。最初直接从 transpose kernel 入手做 layout 改写，方向不够好，因为它绕过了现有高质量算子路径，没有先问清楚“为什么非 MTP decode 没有同样的 transpose”。

后续修正后的优化方向：

- 优先复用非 MTP decode 路径已验证的 `causal_conv1d` 算子。
- 参考 PR #1345 中 `prepare_input_params_for_linear_attention` 的思路，将 host tensor / 接口参数尽量在图外准备好。
- 避免在 MTP verify 的 device 计算路径中重复做权重 reshape / transpose。
- 保持 MTP 接受率基本稳定，用接受率和 evalscope 10 条请求做精度/行为 sanity check。

## 2. 根因

最初 profiling 看到 MTP 下 transpose kernel 调用多，容易误判为单纯的 `[B,T,C]` / `[B,C,T]` layout 局部问题。真正根因是两条路径使用的 causal conv 实现不同：

| 路径 | causal conv 实现 | 参数准备 | transpose 行为 |
|---|---|---|---|
| 非 MTP decode | `causal_conv1d` 已优化路径 | host / 图外提前准备 | 权重 layout 已规整，transpose 少 |
| MTP spec verify | `causal_conv1d_update` 路径 | verify 内部临时准备 | 重复权重/layout 转换，transpose 多 |

因此正确做法不是继续扩大局部 transpose 改写，而是让 MTP verify 尽量走与非 MTP decode 一致的 causal conv 接口和参数准备方式。

## 3. 修改原则

1. 最小化侵入：不引入新 TileLang 算子，不修改主模型数学逻辑。
2. 复用优先：优先复用非 MTP decode 已验证的 `causal_conv1d` 路径。
3. host 侧提前准备：把接口参数变化、host tensor 构造、权重 layout 处理提前到图外，避免每步 device 上做重复工作。
4. 不用 layout 猜测：不要用 `seq_len` 和 `channels` 的维度大小启发式判断 tensor layout，layout contract 必须由调用链明确保证。
5. 性能前先看环境：evalscope 前先记录全卡 `npu-smi` 和 xllm/evalscope 进程，确认测试卡没有其他进程。

## 4. 验证方法

本轮统一使用用户指定 evalscope 命令：

```bash
evalscope perf \
  --parallel 1 \
  --number 10 \
  --model Qwen35-27B \
  --url http://127.0.0.1:38050/v1/chat/completions \
  --api openai \
  --dataset random \
  --max-tokens 1024 \
  --min-tokens 1024 \
  --prefix-length 0 \
  --min-prompt-length 512 \
  --max-prompt-length 512 \
  --tokenizer-path /home/data/weights/Qwen35-27B \
  --extra-args '{"ignore_eos": true}'
```

服务参数要点：

- TP=4，4 个 xllm 进程。
- MTP=3，即 `--num_speculative_tokens 3`。
- 开启 chunk prefill：`--enable_chunked_prefill=true`，`--max_tokens_per_chunk_for_prefill=256`。
- 开启 schedule overlap：`--enable_schedule_overlap=true`。
- 测试前记录 `npu-smi info` 和 `pgrep -af '[x]llm/core/server/xllm|[e]valscope perf'`。

## 5. 2026-05-29 Preview 最新基线

基线代码来自官方最新 preview 分支：

- Worktree: `/home/g00510989/xllm/xllm_preview_qwen35_baseline`
- Branch: `origin/preview/qwen3.5-qwen3.6`
- Commit: `7cdaa866 bugfix: qwen3.5 resolve causal_conv1d tiling failure for concurrent decode requests. (#1585)`
- Submodule: 已执行 `git submodule update --init --recursive --force`
- Recursive submodule 检查：`git submodule status --recursive | awk '$1 ~ /^[-+]/ {print}'` 无输出
- Build: `python setup.py build --device npu` 成功
- 额外说明：fresh worktree + 完整 recursive submodule 会触发 TileLang/TVM、torch_npu_ops、xllm_ops、Rust tokenizer/safetensors 和测试目标全量构建，时间明显长于正常增量 build。

环境占用：

- 测试前 0-5 号板存在其他进程。
- NPU 6/7 对应物理卡 12,13,14,15 无运行进程，用作本轮测试。
- 第一次启动遇到 `port 48054 EADDRINUSE`，已停掉残留服务后改用 `MASTER_PORT=48150`、`HCCL_IF_BASE_PORT=58150` 重跑。

基线 evalscope 结果：

| 指标 | Preview 最新基线 MTP=3 |
|---|---:|
| Total / Success / Failed | 10 / 10 / 0 |
| Avg Latency | 13.27 s |
| TTFT | 461.37 ms |
| TPOT | 12.52 ms |
| ITL | 37.30 ms |
| Output Throughput | 74.64 tok/s |
| Decode Throughput | 79.87 tok/s |
| Decoded Tok/Iter | 2.99 |
| Spec Accept Rate | 66.5% |

本地记录：

- Run root: `/home/g00510989/xllm/runs/20260529_evalscope_512_1024_preview_qwen35_baseline_devices_12_15_retry`
- Evalscope log: `evalscope_512_1024_number10.log`
- Server logs: `server_mtp3/rank*.log`
- 环境记录: `npu_smi_before.txt`、`npu_smi_after.txt`、`process_before.txt`、`run_meta.txt`

## 6. 与复用 causal_conv1d 修改版对比

修改版本地 run：

- Run root: `/home/g00510989/xllm/runs/20260529_evalscope_512_1024_reuse_conv_devices_12_15_retry`
- 配置：同样 random 512 输入 / 1024 输出，parallel=1，number=10，MTP=3，chunk prefill 开启。

| 指标 | Preview 最新基线 | 复用 causal_conv1d 修改版 | 变化 |
|---|---:|---:|---:|
| Avg Latency | 13.27 s | 11.86 s | -10.6% |
| TTFT | 461.37 ms | 453.17 ms | -1.8% |
| TPOT | 12.52 ms | 11.15 ms | -10.9% |
| Output Throughput | 74.64 tok/s | 83.19 tok/s | +11.5% |
| Decode Throughput | 79.87 tok/s | 89.69 tok/s | +12.3% |
| Decoded Tok/Iter | 2.99 | 3.29 | +10.0% |
| Spec Accept Rate | 66.5% | 69.6% | +3.1 pct |

结论：在当前测试环境和 10 条 random 512/1024 evalscope 下，复用 causal_conv1d 的修改版没有性能劣化，TPOT 约下降 10.9%，Output TPS 约提升 11.5%。接受率没有下降，反而从 66.5% 到 69.6%，说明主模型行为没有出现明显异常；仍建议最终 PR 中保留 10 条输出 sanity check 和更长 number 的稳定性复测。

## 7. 这次为什么一开始没发现正确根因

复盘点：

1. 先看到了 profiling 里的 transpose 热点，就直接沿着 transpose kernel 做局部 layout 消除，没有先横向对比非 MTP decode 路径。
2. 没有把问题抽象为“同一个 GatedDeltaNet causal conv 在 MTP 与非 MTP 下为什么路径不同”，而是局限在 MTP 路径内部。
3. 没有第一时间检查非 MTP decode 已经如何通过 `causal_conv1d` 和 host 参数准备减少 transpose。
4. 对 PR #1345 `prepare_input_params_for_linear_attention` 这类“host tensor 图外准备”的模式复用不够主动。

以后遇到类似 transpose / contiguous / reshape 性能问题，必须先做这三个检查：

- 同一模型的非 MTP / MTP / prefill / decode 是否走了不同 kernel。
- 慢路径是否没有复用快路径已经完成的 host 参数准备或权重 layout 预处理。
- 能否通过复用现有算子路径解决，而不是先写新的 layout 特判。

## 8. PR 描述应包含的要点

PR #1536 的描述不应再强调“手工改 layout 直接消除 transpose”为主方案，而应改为：

- MTP verify 路径复用非 MTP decode 的 `causal_conv1d` 优化思路。
- 将 causal conv 接口参数和 host tensor 提前准备，避免每步重复构造/transpose。
- 修复 MTP verify 与非 MTP decode causal conv 路径不一致导致的 transpose 放大。
- 验证使用 evalscope random 512/1024、parallel=1、number=10、MTP=3、chunk prefill enabled。
- 明确列出 preview 最新基线与修改版 TTFT/TPOT/TPS/接受率对比。

PR 性能表建议直接使用 §6 的数据。

