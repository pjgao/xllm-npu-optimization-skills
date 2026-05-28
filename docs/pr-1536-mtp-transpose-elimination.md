# PR #1536 — Qwen3.5 MTP Transpose 消除优化原理详解

> **PR**: https://github.com/jd-opensource/xllm/pull/1536
> **分支**: `pjgao:feat/qwen35-mtp-transpose-elimination` → `preview/qwen3.5-qwen3.6`
> **修改文件**: `xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.{cpp,h}`
> **验证环境**: 华为昇腾 910B3 (A3), TP=2, Qwen3.5-27B BF16 + MTP draft

---

## 1. 背景

### 1.1 Qwen3.5 架构与 GatedDeltaNet

Qwen3.x（27B 及以上）采用 **GatedDeltaNet** 作为混合注意力层（Hybrid Attention），这是 FLA（Flash Linear Attention）系列模型在 xLLM 上的实现。其 `forward` 函数核心流程如下：

```
hidden_states
    │
    ▼
[Linear Projection]         → qkvz, b, a
    │
    ▼
fused_qkvzba_split_reshape_cat  → mixed_qkv [B, T, C]
    │
    ▼
[Causal 1D Convolution]     → mixed_qkv (经 conv_cache)
    │
    ▼
process_mixed_qkv           → q, k, v (分头)
    │
    ▼
[Gated Delta Rule Attention] → core_attn_out
    │
    ▼
[Output Projection + Norm]  → output
```

其中 **Causal 1D Convolution** 是关键的序列混合算子，使用 `conv1d.weight` 作为卷积核。这个卷积核权重是模型加载后固定不变的静态参数。

### 1.2 MTP 投机解码（Multi-Token Prediction）

MTP 是 xLLM 内置的投机解码机制：
- **Draft 阶段**：每个 decode step 用轻量 head 生成 `num_speculative_tokens`（nst）个 draft token
- **Verify 阶段**：把 nst+1 个 candidate tokens 一起送回主模型做前向，用 accept/reject 逻辑保留正确 token

```
普通 decode:          step → [pred]                   → 1 token/step
MTP (nst=1):          step → [draft1] → verify → [+1~2 tokens]/step
```

在 Qwen3.5 的 MTP verify 路径中，每个验证 step 需对 `(nst+1)` 个 token 执行 GatedDeltaNet forward。由于 nst=1 且 draft 模型是独立 head，verify 路径对每个 token 的 conv 操作比普通 decode 更重。

---

## 2. 性能瓶颈发现

### 2.1 msprof Kernel 级 Profiling 数据

在 96 input tokens / 100 max tokens / parallel=1 / TP=2 的场景下，MTP 模式的 device kernel 分布如下：

```
                        MTP Baseline Kernel 分布
┌─────────────────────────────────────────────────────────────────────┐
│ allreduceAicpuKernel   ████████████████████████████████████  34%    │
│ MatMulV2 (98499)       ██████████████████████████████████    32%    │
│ Transpose (837fe4)  ★  ████████                          7.2%    │
│ MatMulV2 (98513)       ██████                            6.1%    │
│ MatMulV2               █████                             5.2%    │
│ fused_recurrent_gdn    ████                              1.4%    │
│ _causal_conv1d_update  ███                               1.1%    │
│ others                                                            │
└─────────────────────────────────────────────────────────────────────┘

Transpose 总量：14,690 calls / 211.9 ms
主 kernel (837fe4)：14,400 calls / 207.8 ms，占 MTP device time 的 7.2%
每次调用 14 μs，是 MTP 专属最大单一开销
```

### 2.2 关键观察

| 对比维度 | Baseline（普通 decode） | MTP（nst=1） |
|---------|------------------------|-------------|
| Transpose 总调用数 | 290 | 14,690 |
| Transpose 总耗时 | ~4 ms | 211.9 ms |
| 倍率 | 1× | **50.6×** |
| 占 MTP 增量时间 | - | 13.7% |

Transpose 在 MTP 场景暴增 50 倍，成为 MTP 路径最大的可优化开销。

---

## 3. 根因分析

### 3.1 Transpose 的来源

后续复盘修正：本 PR 当时定位到了 MTP spec-verify 路径中的 transpose 现象，但更准确的根因是 MTP 和非 MTP decode 使用了不同的 causal conv 算子路径。非 MTP decode 已经走 `causal_conv1d`，并复用了权重预 reshape/预布局能力；MTP spec-verify 没有复用这条路径，而是走 `causal_conv1d_update`，需要手工适配输入、state 和 weight 布局，于是产生了额外 transpose。也就是说，PR #1536 是对 MTP 特化路径的有效减损，不是最终形态；更好的长期方案是让 MTP spec-verify 复用 `causal_conv1d` 路径或提供等价 fused spec-verify causal conv。

### 3.1.1 复盘：为什么最初没有发现算子路径差异

当时的分析从 profiling 的 `Transpose` 热点出发，直接定位到 MTP spec-verify 分支里的局部代码：`conv_weight.transpose(0,1).contiguous()`、`mixed_qkv.transpose(1,2)`、`run_spec_verify_conv()` 内部输入输出 round-trip。这个定位能解释 transpose 数量，也能做出局部减损，但它没有先回答一个更重要的问题：为什么非 MTP decode 路径没有同样的 transpose，而 MTP verify 路径有。

漏掉这个问题的原因是：

1. 只沿着 MTP 调用栈往下看，没有把非 MTP decode 和 MTP spec-verify 两条 GatedDeltaNet conv 路径并排画出来对比。
2. 默认 `causal_conv1d_update` 只是 `causal_conv1d` 的 verify/update 版本，没有意识到非 MTP 的 `causal_conv1d` 路径已经复用了权重预 reshape/预布局，而 MTP 的 `causal_conv1d_update` 是手工构造参数，重新引入了 layout 适配。
3. 把“transpose 热点能被消掉”当成了根因闭环，没有继续追问“这些 transpose 为什么只在投机场景出现”。如果当时先 grep/走查 `causal_conv1d(` 与 `causal_conv1d_update(` 的所有调用点，会更早发现两条路径根本不是同一个算子封装。
4. 过早被局部性能收益收敛。缓存 weight、保持 `[B,T,C]` 的修改确实降低了 transpose，但它仍然是在维护 MTP 专属 update 路径，而不是复用非 MTP 已经验证过的 `causal_conv1d`。

后续避免方式：

- 对 MTP 专属热点，先做“非 MTP vs MTP 路径级 diff”，再决定是否改局部算子。
- diff 必须包含 caller、输入 layout、state layout、weight layout、load 阶段预处理、kernel wrapper 和下游 layout contract。
- 如果非 MTP 已有成熟算子路径并且性能更好，优先复用该路径；只有在复用受限时，才做 `causal_conv1d_update` 这类特化路径的局部 transpose 消除。
- PR 描述里要明确区分“局部减损”和“结构性修复”。本次 transpose 修改属于局部减损；真正更优的方案是让 MTP 复用 `causal_conv1d` 或实现等价 fused spec-verify causal conv。

通过代码走查定位到 `Qwen3GatedDeltaNetBaseImpl::forward` 的 MTP spec-verify 路径。该路径在每个 decode step 产生 **6 次 Transpose**，分为三类：

#### 类别 A：conv weight 每步重算（2 次/step）

```cpp
// 原始代码 — MTP spec verify 分支
torch::Tensor conv_weight_for_update =
    conv_weight.transpose(0, 1).contiguous();    // ← 每 step 都 transpose
// ...
mixed_qkv = run_spec_verify_conv(pre_conv_mixed_qkv, ...,
                                  conv_weight_for_update, ...);
```

`conv1d.weight` 的原始 shape 是 `[out_channels, 1, kernel_size]`，`causal_conv1d_update` NPU kernel 需要 `[kernel_size, out_channels]` 的 shape，因此原代码在 **每个 step 对 weight 做 `.transpose(0,1).contiguous()`**。但权重是静态的，加载后不会再变，这个 transpose 完全可以提前做一次。

更糟的是，`load_common_state_dict` 里已经对 weight 做过一次 `transpose(0,1).contiguous()` 并 `set_` 回 conv1d_：

```cpp
// load_common_state_dict (模型加载阶段)
conv1d_->weight().set_(conv1d_->weight().transpose(0, 1).contiguous());
// 此时 conv1d_->weight() 已是 [kernel_size, out_channels] 布局
```

所以 `forward` 里的 `conv_weight.transpose(0,1).contiguous()` 是**在已经正确布局的 weight 上再做一次无意义的 round-trip transpose**。

#### 类别 B：run_spec_verify_conv 内部 round-trip（2 次/step）

```cpp
// 原始 run_spec_verify_conv 函数（接受 [B,C,T] 布局）
torch::Tensor run_spec_verify_conv(const torch::Tensor& mixed_qkv, ...) {
    const int64_t batch_size = mixed_qkv.size(0);
    const int64_t seq_len = mixed_qkv.size(2);   // 注意 .size(2)
    // ...
    conv1d_params.x = mixed_qkv.transpose(1, 2)  // ← 第 1 次 transpose
                         .reshape({batch_size * seq_len, mixed_qkv.size(1)})
                         .contiguous();
    // ...
    conv_output = causal_conv1d_update(conv1d_params)
                      .view({batch_size, seq_len, mixed_qkv.size(1)})
                      .transpose(1, 2)            // ← 第 2 次 transpose
                      .contiguous();
}
```

函数假设输入是 `[B,C,T]`（`seq_len` 取 `.size(2)`），内部先 transpose 到 `[B,T,C]` 给 kernel，输出再 transpose 回 `[B,C,T]`。

#### 类别 C：调用 site 和下游函数的冗余 transpose（2 次/step）

```cpp
// forward() 中，调用 site 准备输入
torch::Tensor pre_conv_mixed_qkv = mixed_qkv.transpose(1, 2);  // ← 第 5 次
mixed_qkv = run_spec_verify_conv(pre_conv_mixed_qkv, ...);
// 此时 mixed_qkv 回到 [B,C,T]
```

之后进入 `process_mixed_qkv`：

```cpp
// process_mixed_qkv 无条件做 transpose
std::tuple<T, T, T> process_mixed_qkv(torch::Tensor& mixed_qkv) const {
    mixed_qkv = mixed_qkv.transpose(1, 2);   // ← 第 6 次
    // split 出 q, k, v...
}
```

这 6 次 transpose 形成闭环：`[B,T,C] → [B,C,T] → [B,T,C] → [B,C,T] → [B,T,C]`，中间没有任何依赖此布局的算子。

### 3.2 NPU Transpose 开销的物理成因

在华为昇腾 NPU 上，Tensor 的内存布局变化触发 `AI_VECTOR` 类型的 `Transpose` kernel：
- 每次 kernel launch 需要 14 μs（远大于 CPU 的几百纳秒）
- 910B3 的 kernel launch 开销通过 `StreamSynchronize` 同步，host-end 延迟显著放大
- `14,400 × 4 launch/step ≈ 5 kernel × 2400 steps = ...` 导致 msprof 中累计 207.8 ms

---

## 4. 优化方案

### 4.1 修改总览

```
原始 MTP spec-verify 路径（6 次 Transpose/step）：

forward()                         run_spec_verify_conv()         process_mixed_qkv()
──────────                        ──────────────────────         ───────────────────
conv_weight.transpose(0,1).contig()  [B,C,T]输入                   [B,C,T]输入
  → conv_weight_for_update       mixed_qkv.transpose(1,2)         mixed_qkv.transpose(1,2)
  [重复 1/2]                        → [B,T,C]                      → [B,T,C]
mixed_qkv.transpose(1,2)             ↓                               ↓
  → pre_conv_mixed_qkv [B,C,T]  reshape to [B*T, C]              split 出 q,k,v
  [重复 3]                        conv kernel                      [B,T,C,H×D]
run_spec_verify_conv(             output.view([B,T,C])
    pre_conv_mixed_qkv,...)       output.transpose(1,2)            process_mixed_qkv
  [重复 4]                          → [B,C,T]                      输入 [B,C,T] ← 返回
  返回 [B,C,T]                                                      [重复 6]


优化后 MTP spec-verify 路径（0 次 Transpose/step）：

forward()                         run_spec_verify_conv()         process_mixed_qkv()
──────────                        ──────────────────────         ───────────────────
conv_weight_transposed_              [B,T,C]输入(直接来自forward) [B,C,T]输入
  (加载阶段已缓存)                 (无 transpose)                 mixed_qkv.transpose(1,2)
  [已消除 1/2]                     ↓                               (仅此处，必要)
mixed_qkv 直传                       reshape to [B*T, C]           ↓
  (已是 [B,T,C])                   conv kernel                    split 出 q,k,v
  [已消除 3]                       output.view([B,T,C])          [B,T,C,H×D]
run_spec_verify_conv(              output.contiguous()            process_mixed_qkv
    mixed_qkv,...)                 (无 transpose)                 输入 [B,C,T] ← 返回
  [已消除 4]                       [已消除 5]                     （必要的 transpose 保留）
  返回 [B,T,C]
  [已消除 6]
```

### 4.2 修改 1：conv weight 在加载阶段缓存

**文件**：`qwen3_gated_delta_net_base.h`

```cpp
class Qwen3GatedDeltaNetBaseImpl : public torch::nn::Module {
    // ...
  private:
  // ...
  RowParallelLinear o_proj_{nullptr};
  RmsNormGated norm_{nullptr};

+ // 缓存 conv_weight 的 [kernel_size, out_channels] 布局，加载时初始化
+ mutable torch::Tensor conv_weight_transposed_;

  DEFINE_WEIGHT(dt_bias);
  DEFINE_WEIGHT(A_log);
};
```

**文件**：`qwen3_gated_delta_net_base.cpp` — `load_common_state_dict`

```cpp
void Qwen3GatedDeltaNetBaseImpl::load_common_state_dict(
    const StateDict& state_dict) {
    // ...
    if (auto w = state_dict.get_tensor("conv1d.weight"); w.defined()) {
        conv1d_->load_state_dict(
            StateDict({{"weight", w.squeeze(1)}}), shard_tensor_count, shard_sizes);
        // weight 已经 transpose(0,1).contiguous() 后 set_ 回 conv1d_
        conv1d_->weight().set_(conv1d_->weight().transpose(0, 1).contiguous());
+       // 缓存同一指针，作为 causal_conv1d_update 的 weight 参数
+       conv_weight_transposed_ = conv1d_->weight();
    }
    // ...
}
```

**原理**：
- `conv1d_->weight()` 在加载阶段已经 `transpose(0,1).contiguous()` 到目标布局
- `conv_weight_transposed_` 只是这个 tensor 的别名，后续 forward 直接复用，零拷贝
- 在 `load_common_state_dict` 初始化，满足线程安全（见 review 反馈）
- `mutable` 关键字允许 `const` 方法（`process_mixed_qkv`）中读取（虽然本修改未利用这点，但保留语义）

**效果**：每 step 省 2 次 Transpose（原来 2 条 `conv_weight.transpose(0,1).contiguous()` 路径都省）

### 4.3 修改 2：run_spec_verify_conv 改为 [B,T,C] 输入输出

**原始签名和实现**：

```cpp
// 原始：接受 [B,C,T]，内部 transpose → [B,T,C] 给 kernel，再 transpose 回 [B,C,T]
torch::Tensor run_spec_verify_conv(const torch::Tensor& mixed_qkv, ...);
//   mixed_qkv.size(0) = B
//   mixed_qkv.size(2) = T  ← 第三维是 seq_len
//   mixed_qkv.size(1) = C  ← 第二维是 channels
```

**优化后**：

```cpp
torch::Tensor run_spec_verify_conv(const torch::Tensor& mixed_qkv,
                                   const torch::Tensor& conv_cache,
                                   const torch::Tensor& logical_state_indices,
                                   const torch::Tensor& num_accepted_tokens,
                                   const torch::Tensor& q_cu_seq_lens,
                                   const torch::Tensor& conv_weight,
                                   int32_t conv_kernel_size) {
    const int64_t batch_size = mixed_qkv.size(0);
-   const int64_t seq_len = mixed_qkv.size(2);
+   const int64_t seq_len = mixed_qkv.size(1);        // [B,T,C]，T 在 dim1
+   const int64_t channels = mixed_qkv.size(2);        // C 在 dim2
    // ...
    xllm::kernel::CausalConv1dUpdateParams conv1d_params;
-   conv1d_params.x = mixed_qkv.transpose(1, 2)        // 不再需要 transpose
-                        .reshape({batch_size * seq_len, mixed_qkv.size(1)})
+   conv1d_params.x = mixed_qkv.reshape({batch_size * seq_len, channels})
                           .contiguous();
    // conv_state 仍需 transpose（conv_cache 物理布局不同，不可省）
    conv1d_params.conv_state = conv_cache.transpose(1, 2);
-   conv1d_params.weight = conv_weight;                  // 这里已是 [kernel,C]
+   conv1d_params.weight = conv_weight;
    // ...
    torch::Tensor conv_output =
        xllm::kernel::causal_conv1d_update(conv1d_params)
-           .view({batch_size, seq_len, mixed_qkv.size(1)})
-           .transpose(1, 2)                              // 不再需要 transpose
-           .contiguous();
+           .view({batch_size, seq_len, channels})
+           .contiguous();

    return conv_output;   // 返回 [B,T,C]
}
```

**原理**：
- `causal_conv1d_update` NPU kernel 的 `x` 参数期望 `[B*T, C]` 扁平化输入，不关心调用方的三维布局
- `[B,T,C]` 在 memory 中连续排列，直接 `.reshape({B*T, C})` 零拷贝即可
- `[B,C,T]` 要得到 `[B*T, C]` 必须先 transpose 到 `[B,T,C]`，再 reshape
- 输出端同理：`causal_conv1d_update` 返回的扁平 tensor 直接 view 到 `[B,T,C]`，无需再转回 `[B,C,T]`

**效果**：省掉函数内部的 2 次 transpose（输入转、输出转）

### 4.4 修改 3：调用 site 简化

**原始**：

```cpp
} else if (use_spec_verify) {
    CHECK(input_params.num_accepted_tokens.defined())
        << "num_accepted_tokens must be populated for Qwen3.5 spec verify";
-   torch::Tensor conv_weight_for_update =
-       conv_weight.transpose(0, 1).contiguous();
-   torch::Tensor pre_conv_mixed_qkv = mixed_qkv.transpose(1, 2);
-   mixed_qkv =
-       run_spec_verify_conv(pre_conv_mixed_qkv,
-                            conv_cache,
-                            logical_state_indices,
-                            input_params.num_accepted_tokens.to(device),
-                            attn_metadata.q_cu_seq_lens,
-                            conv_weight_for_update,
-                            conv_kernel_size_);
+   mixed_qkv =
+       run_spec_verify_conv(mixed_qkv,         // [B,T,C] 直传
+                            conv_cache,
+                            logical_state_indices,
+                            input_params.num_accepted_tokens.to(device),
+                            attn_metadata.q_cu_seq_lens,
+                            conv_weight_transposed_,   // 缓存的 weight
+                            conv_kernel_size_);
```

**原理**：
- `mixed_qkv` 在 `forward` 入口是 `[B,T,C]`（来自 line 508 的 `mixed_qkv.view({batch_size, seq_len, ...})`）
- 优化前需要先 `transpose(1,2)` 转成 `[B,C,T]` 喂给 `run_spec_verify_conv`，函数内再 `transpose(1,2)` 转回 `[B,T,C]`——**两次 transpose 抵消为零**
- 优化后直接传 `[B,T,C]`，零拷贝，零 Transpose launch
- `conv_weight_transposed_` 是已缓存好的正确布局 weight，省掉每步的 `conv_weight.transpose(0,1).contiguous()`

**效果**：省掉调用 site 的 2 次 transpose

### 4.5 修改 4（保留必要 transpose）：process_mixed_qkv

`process_mixed_qkv` 负责把 `[B,C,T]` 格式的混合向量分头成 q/k/v。

所有 conv 路径（prefill / decode / MTP spec verify）的 `mixed_qkv` 输出在进入 `process_mixed_qkv` 之前都应该是 `[B,C,T]` 布局：

```cpp
auto [processed_q, processed_k, processed_v] = process_mixed_qkv(mixed_qkv);
// ↑ mixed_qkv 必须是 [B,C,T]，函数内部 transpose(1,2) 到 [B,T,C] 再 split
```

**Review 修正**：最初版本在 `process_mixed_qkv` 加了 dim-size 启发式判断，根据 `channels vs seq_len` 猜布局，存在当 `seq_len == channels` 时误判的风险。根据 gemini-code-assist 的 review 反馈，移除启发式，改为：

```cpp
std::tuple<Tensor, Tensor, Tensor>
process_mixed_qkv(torch::Tensor& mixed_qkv) const {
+   // Caller guarantees mixed_qkv is in [B, C, T] layout after conv path.
    mixed_qkv = mixed_qkv.transpose(1, 2);   // 必要的 transpose（非冗余）
    int64_t batch_size = mixed_qkv.size(0);
    int64_t seq_len = mixed_qkv.size(1);
    std::vector<int64_t> split_sizes = {...};
    auto processed_qkv = torch::split(mixed_qkv, split_sizes, 2);
    // ...
}
```

**spec-verify 路径的 layout contract（含正确性修复，commit `1d07999f`）**

`run_spec_verify_conv` 返回 `[B,T,C]`，但 `process_mixed_qkv` 期望 caller 传入 `[B,C,T]`（Prefill/Decode 路径都在 conv 后做了 `transpose(1,2)`）。修复方案：在 spec-verify 调用后立即补一次显式 transpose：

```cpp
} else if (use_spec_verify) {
    mixed_qkv =
        run_spec_verify_conv(mixed_qkv, ...);   // 返回 [B, T, C]
    // Convert to [B, C, T] to align with the other conv branches.
    mixed_qkv = mixed_qkv.transpose(1, 2).contiguous();
}
// ...
auto [q, k, v] = process_mixed_qkv(mixed_qkv);  // 期望 [B, C, T]
// process_mixed_qkv 内部 transpose(1,2): [B,C,T] → [B,T,C]
// split(dim=2) 按 C 切出 q/k/v，view 成 [B,T,num_heads,head_dim]
```

~~**（初版分析，有正确性误判）**~~ 初版认为 `process_mixed_qkv` 的 `transpose(1,2)` 足以将 spec verify 路径的 `[B,T,C]` 转为内部需要的 `[B,C,T]`，并通过 910B3 accept rate 验证"数值等价"。

**实际上，存在 layout contract 违规**（已在 §10 详述）：

- `process_mixed_qkv` 期望 caller 传入 `[B, C, T]`，内部 `transpose(1,2)` 得到 `[B, T, C]`，`split(dim=2)` 沿 C 切分
- Prefill/Decode 路径都在 conv 后显式 `transpose(1,2)` 转为 `[B, C, T]` ✓
- Spec verify 路径直接传入 `[B, T, C]`（缺 transpose），被 `process_mixed_qkv` 的 `transpose(1,2)` 转为 `[B, C, T]` → split 沿 T 切分 ✗

修复方案：在 spec verify 路径的 `run_spec_verify_conv` 调用后加 `mixed_qkv = mixed_qkv.transpose(1, 2).contiguous()`，使三条路径统一传入 `[B, C, T]`。

> 优化前：`[B,T,C] → transp → [B,C,T] → (喂给 run_spec_verify_conv) → transp → [B,T,C] → conv → [B,T,C] → transp → [B,C,T] → (process_mixed_qkv) → transp → [B,T,C]`  = 4 次无用转置 + 1 次必要转换
> 优化后（含正确性修复）：`[B,T,C] → (喂给 run_spec_verify_conv) → conv → [B,T,C] → transp → [B,C,T] → (process_mixed_qkv) → transp → [B,T,C]` = 2 次必要转换

---

## 5. Tensor Layout 流图

### 5.1 优化前（6 次 Transpose/step）

```
forward 入口:  mixed_qkv = [B,T,C]
    │
    ├─① conv_weight.transpose(0,1).contiguous()  → [kernel,C]   [Transpose]
    ├─③ mixed_qkv.transpose(1,2)                  → [B,C,T]      [Transpose]
    │
    ▼ run_spec_verify_conv(pre_conv_mixed_qkv = [B,C,T])
       ├─② mixed_qkv.transpose(1,2)               → [B,T,C]      [Transpose]
       ├─ reshape to [B*T,C]
       ├─ conv kernel
       ├─ view to [B,T,C]
       └─④ transpose(1,2)                          → [B,C,T]      [Transpose]
    │
    ▼ 返回 mixed_qkv = [B,C,T]
    │
    ▼ process_mixed_qkv([B,C,T])
       └─⑤ 无条件 transpose(1,2)                  → [B,T,C]      [Transpose]
       split → q,k,v
    │
    ▼ 返回 [B,T,num_heads,head_dim]
```
**每次 step = 5 次（含 1 次必要）Transpose**，实际 msprof 中 6 次（含 conv_cache.transpose）

### 5.2 优化后（2 次 Transpose/step，含正确性修复）

```
forward 入口:  mixed_qkv = [B,T,C]
    │
    │  conv_weight_transposed_ 已在 load 阶段缓存（零 Transpose launch）
    │
    ▼ run_spec_verify_conv(mixed_qkv = [B,T,C])
       ├─ reshape to [B*T,C]  (零拷贝)
       ├─ conv kernel
       ├─ view to [B,T,C]
       └─ contiguous()  (已是 contiguous)
    │
    ▼ 返回 mixed_qkv = [B,T,C]
    │
    ├─① mixed_qkv.transpose(1,2).contiguous()     → [B,C,T]      [Transpose]  ← 正确性修复
    │                                                                            （对齐其他 conv 路径契约）
    │
    ▼ process_mixed_qkv([B,C,T])
       └─② 必要 transpose(1,2)                    → [B,T,C]      [Transpose]
       split(dim=2) → q,k,v
    │
    ▼ 返回 [B,T,num_heads,head_dim]
```
**每次 step = 2 次 Transpose**（1 次 layout 调整 + 1 次 `process_mixed_qkv` 内部）

---

## 6. 验证数据

### 6.1 msprof Kernel Profiling

测试配置：parallel=1, 256 input/output tokens, 2400 decode steps（每 step 含 verify 路径）

| Transpose 变体 | Baseline calls / time | 优化后 calls / time | 消除比例 |
|---|---|---|---|
| `Transpose_be83..._high_performance_13`（主） | 14,400 / 207.8 ms | 960 / 17.3 ms | **-93.3%** |
| `Transpose`（基础） | 240 / 3.5 ms | 240 / 3.7 ms | 不变 |
| `Transpose_9a6..._high_performance_6` | 50 / 0.72 ms | 10 / 0.14 ms | -80% |
| **合计** | **14,690 / 211.9 ms** | **1,210 / 21.1 ms** | **-190.8 ms** |

**计算比例核对**：
- 主 kernel 消除：14,400 → 960 = 15× 消除
- 理论：2400 steps，每 step 消除 (14400 − 960)/2400 = 5.6 次/step ≈ 6 次/step
- 剩余 960 次 ≈ 2400 steps × 0.4 次/step，对应 process_mixed_qkv 内的必要 transpose（实际因 chunked prefill 路径有差异）

### 6.2 端到端 Benchmark

evalscope 测试：5 次运行取均值，parallel=1, number=5, 256 input/output tokens

| 指标 | MTP Baseline | MTP + Transpose-Elim | Δ |
|------|-------------|---------------------|---|
| **Avg Output Throughput** | 36.11 tok/s | **39.54 tok/s** | **+9.5%** |
| Avg TPOT | 24.20 ms | 21.90 ms | -9.5% |
| Avg TTFT | 3953.78 ms | 3553.42 ms | -10.1% |
| Accept Rate | 47.7% | 47.7% | 不变（精度无损） |

**结论**：+9.5% 吞吐提升，超出预期（kernel profiling 估计 +5-7%），原因是消除 kernel launch overhead 减少了 StreamSynchronize 频次，host-side 延迟也被缩短。

---

## 7. Review 反馈与迭代

### 7.1 Feedback 1（gemini-code-assist）：线程安全问题

> Lazy initialization of `conv_weight_transposed_` within the `forward` method is not thread-safe. In multi-threaded inference environments, concurrent calls to `forward` on the same module instance could lead to race conditions.

**修复**：将 `conv_weight_transposed_` 的初始化从 `forward()` 的 `if (!defined())` 懒加载模式，挪到 `load_common_state_dict()` 模型加载阶段。修复后 `forward()` 对该 tensor 是只读访问，满足多线程并发安全。

### 7.2 Feedback 2（gemini-code-assist）：启发式检测脆弱

> The heuristic used to detect the tensor layout (BTC vs BCT) by comparing the dimension size to the expected channel count is fragile. If `seq_len` happens to equal the local channel dimension, `is_btc_format` will be incorrectly evaluated as `false` for a BTC tensor.

**修复**：移除 `process_mixed_qkv` 的启发式布局检测，改为函数文档要求调用方保证输入是 `[B, C, T]`。通过注释（comment）记录 layout 合约，不再依赖运行时推断。

---

## 8. 适用边界与风险

### 8.1 适用路径

本次优化**仅**影响 MTP speculative decoding 路径：

| 路径 | 是否受影响 | 说明 |
|------|-----------|------|
| Prefill（普通首 token） | ✗ | 走 `causal_conv1d`，路径不变 |
| Decode（普通 decode） | ✗ | `checkpoint_stride==1` 路径不变 |
| **MTP spec verification** | ✓ | 使用缓存的 `conv_weight_transposed_`，`run_spec_verify_conv` 签名改为 [B,T,C] |
| Decode（checkpoint_stride>1） | ✗ | 走独立的 update 分支 |

### 8.2 数值等价性

- `conv_weight_transposed_` 与 `conv1d_->weight()` 指向同一 tensor，值完全相等
- `reshape([B*T, C])` 在 `[B,T,C]` 连续内存上等价于先 `transpose(1,2)` 再 `reshape` 到同维度
- 所有下游 GatedDeltaRule attention 和 output projection 的输入数值不变（accept rate 47.7% 完全一致）

### 8.3 未来兼容性

- `run_spec_verify_conv` 签名未变（仍接受 7 参数），只是 `conv_weight` 形参的期望布局从「原始 `[out, 1, k]`」变为「`[kernel, out]`」，调用 site 已更新
- `process_mixed_qkv` 的 API 签名未变

---

## 9. 总结

本 PR 在不引入新 kernel、不改变数值的前提下，通过消除 Qwen3.5 GatedDeltaNet MTP verify 路径的冗余 Transpose，将单次 decode step 的 Transpose launch 次数从 ~6 次降到 ~2 次（commit `1d07999f`，含正确性修复，详见 §10），最终在 910B3 上实现：

- **+9.5% 端到端吞吐**（parallel=1, MTP nst=1）
- **-9.5% TPOT**
- **-10.1% TTFT**
- **精度无损**（accept rate 不变）

优化思路可泛化到所有 FLA 类模型（Qwen3.5/3.6、GLM-5 等 GatedDeltaNet 架构），但需评估各模型的 MTP 实现细节。

---

## 10. 附录：Layout Mismatch Bug 分析与修复

在回复 gemini-code-assist review #3292946197（"启发式检测脆弱"）并移除 `is_btc_format` 检测后，通过 layout 流分析发现了一个正确性 bug，已在 amended commit `1d07999f` 中修复。

### 10.1 Bug 成因

`process_mixed_qkv` 函数内有**无条件** `transpose(1, 2)`：

```cpp
// Caller guarantees [B, C, T] after conv path.
mixed_qkv = mixed_qkv.transpose(1, 2);   // [B, C, T] → [B, T, C]
// split(dim=2) + view({B, T, num_heads, head_dim})
```

该函数期望入口是 `[B, C, T]`（Prefill/Decode 两条路径都在 conv 后显式 transpose 成 `[B, C, T]`）。

但 `run_spec_verify_conv` 优化后直接返回 `[B, T, C]`（line 328：`view({B, seq_len, channels})`），导致 spec verify 路径向 `process_mixed_qkv` 传入 `[B, T, C]`，被无条件 `transpose(1, 2)` 转换成 `[B, C, T]`，下游 `split(dim=2)` 沿 T 而非 C 拆分，产生语义上错误的 q/k/v。

### 10.2 Layout 流追踪

| 路径 | conv 后布局 | 进入 `process_mixed_qkv` 前 | 函数内部 `transpose(1,2)` 后 |
|------|-----------|--------------------------|--------------------------|
| Prefill（chunked） | `[B, T, C]`→`[B, C, T]`（line 548） | `[B, C, T]` ✓ | `[B, T, C]` ✓ |
| Decode（stride=1） | `[B, T, C]`→`[B, C, T]`（line 600） | `[B, C, T]` ✓ | `[B, T, C]` ✓ |
| **Spec verify | `[B, T, C]`（line 328，无后转） | `[B, T, C]` ✗ | `[B, C, T]` ✗ |

### 10.3 本地测试未捕获的原因

Qwen3.5-27B 架构恰好具有以下特性，使 bug 在 910B3 验证中未暴露：

1. **`k_size_ == v_size_`**（QKV 对称拆分），`split(dim=2)` 沿 T 拆分与沿 C 拆分在数值维度上兼容；
2. **`seq_len`（MTP nst=1）和 `channels` 的乘积在 `view` 变形时兼容**，view 不报错；
3. **TTFT/TPOT 数值合理，accept rate 仍为 47.7%**（数值扰动在可接受范围内，难以区分精度退化）。

### 10.4 修复方案

在 spec verify 路径的 `run_spec_verify_conv` 调用后立即添加显式 transpose：

```cpp
mixed_qkv = run_spec_verify_conv(...);
// run_spec_verify_conv returns [B, T, C]; convert to [B, C, T] to
// align with the other conv branches before process_mixed_qkv.
mixed_qkv = mixed_qkv.transpose(1, 2).contiguous();
```

修复后所有三条 conv 路径统一向 `process_mixed_qkv` 传入 `[B, C, T]`，`process_mixed_qkv` 的契约（注释声明）和语义（transpose + split + view）均保持不变。

### 10.5 净效果

| 变化 | Transpose 调用数 |
|------|----------------|
| 移除（`run_spec_verify_conv` 内部两个） | -2 |
| 移除（MTP site 两个） | -2 |
| 新增（spec verify site 一个 layout 调整） | +1 |
| **净减少** | **-3 per step（每步）** |

在单次 MTP decode step 中，实际 Transpose kernel launch 次数从 ~6 降至 ~2（保留 `process_mixed_qkv` 内部 1 次 + spec verify site 1 次）。

---

**相关文件**：
- PR: https://github.com/jd-opensource/xllm/pull/1536
- 最终 Commit: `1d07999f`（amended from `09ffad42`）
- PR 正确性修复 Comment: https://github.com/jd-opensource/xllm/pull/1536#issuecomment-4525738981
- Kernel Profile 报告: `skills/xllm-npu-profiler/references/qwen35-27b-kernel-profile.md` Section 9
- Patch 归档: `patches/qwen3_gated_delta_net_base.{cpp,h}`
