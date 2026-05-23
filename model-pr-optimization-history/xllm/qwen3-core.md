# Qwen3 系列优化档案

## 模型变体

| 变体 | 参数量 | 架构 | 推荐卡数 | 推荐 TP |
|------|--------|------|---------|---------|
| Qwen3-0.6B | 0.6B | Dense | A3 x1 | 1 |
| Qwen3-4B | 4B | Dense | A3 x1 | 1 |
| Qwen3-8B | 8B | Dense | A3 x2 | 2 |
| Qwen3-14B | 14B | Dense | A3 x2 | 2 |
| Qwen3-32B | 32B | Dense | A3 x4 | 4 |
| Qwen3-235B-A22B | 235B (激活 22B) | MoE | A3 x8 | 8 |

## 架构特点

- Dense: 标准 Transformer + GQA + RoPE + SwiGLU
- MoE (235B): Expert 并行 + Dynamic expert load balancing
- 思考模式 (Thinking mode): 可开启/关闭

## 推荐启动命令

### Dense (32B)

```bash
xllm serve /models/Qwen3-32B \
    --tensor-parallel-size 4 \
    --graph-mode npugraph_ex \
    --block-size 128 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192
```

### MoE (235B)

```bash
xllm serve /models/Qwen3-235B-A22B \
    --tensor-parallel-size 8 \
    --expert-parallel-size 8 \
    --graph-mode npugraph_ex \
    --block-size 128 \
    --gpu-memory-utilization 0.9
```

## 性能基线 (来自 xLLM 论文 arXiv:2510.14686)

| 变体 | TP | Throughput (tokens/s) | TPOT (ms) | TTFT (ms) |
|------|----|----------------------|-----------|-----------|
| Qwen3-235B-A22B | 8P8D | ~300 (总) | ~50 | ~ |
| Qwen3-32B | 4 | ~ | ~ | ~ |
| Qwen3-8B | 1 | ~ | ~ | ~ |
| Qwen3.5-27B | 2 | 36.11 (MTP) / 29.88 (Base) | 24.2 / 28.3 | ~ |

声明：同 TPOT 下吞吐 = MindIE × 1.7 = vLLM-Ascend × 2.2

### Qwen3.5-27B（DeltaNet 混合架构）

| 配置 | 端口 | TP | Devices | 吞吐 (p=1,n=5) |
|------|------|----|---------|---------------|
| Baseline | 18150/18151 | 2 | Phy 14/15 | 29.88 tok/s |
| MTP (nst=1) | 18160/18161 | 2 | Phy 14/15 | 36.11 tok/s (+21%) |
| **MTP + Transpose消除** | 18170/18171 | 2 | Phy 14/15 | **39.54 tok/s (+32% vs Base, +9.5% vs MTP)** |

**MTP-Transpose 优化**（commit 验证于 2026-05-23）：
- 修改 `qwen3_gated_delta_net_base.cpp/h`
- 缓存 `conv_weight.transpose().contiguous()` 为成员变量
- `run_spec_verify_conv` 改为收发 `[B,T,C]` 格式，消除 round-trip transpose
- msprof 验证：Transpose kernel 从 14,400→960 calls（-93.3%），节省 190.8ms device time
- 补丁：`patches/qwen3_gated_delta_net_base.{cpp,h}`

## 已知优化

| 优化项 | 效果 | 状态 |
|--------|------|------|
| Chunked prefill | Prefill/Decode 混合 | merged |
| Adaptive graph mode | Prefill eager + Decode graph | merged |
| PagedAttention (block_size=128) | KV Cache 高效管理 | merged |
| SwiGLU 融合 | 加速 FFN | merged |
| xTensor async alloc | 减少内存分配延迟 | merged |

## 优化重点

### 1. Prefill 长序列

Qwen3 支持 128K context，长序列 Prefill 是瓶颈。

检查方式：
```bash
# 在 profiling 中查看长序列 (>4096) 的 prefill 时间
# 重点关注 QKV MatMul 和 Attention kernel
```

### 2. Chunked Prefill 与 Decode 重叠

xLLM 的核心优势：Prefill 和 Decode 共享同一 NPU 资源，交替执行。

检查方式：
```bash
# 查看 chunked prefill 的 chunk 大小配置
# 在 profiling 中观察 Prefill/Decode 交替模式
```

### 3. 图模式适配

Decode 阶段使用 npugraph_ex，确认没有 graph break。

检查方式：
```bash
# 在 profiling 中查看是否有 AICPU 算子（表示 graph break）
# 检查 GE 编译日志
```

## 待优化项

- [ ] Dense 模型 TP 通信重叠
- [ ] MoE 模型 EP 负载均衡
- [ ] Thinking mode 的 KV Cache 优化
- [ ] 长序列 Prefill 的 tiling 优化

## 参考

- xLLM arXiv:2510.14686 Qwen 相关章节
- vLLM-Ascend Qwen3 实现：`vllm_ascend/worker/multiproc_worker.py`
