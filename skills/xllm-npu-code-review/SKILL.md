---
name: xllm-npu-code-review
description: xLLM 昇腾 NPU 特化代码审查。审查 C++ 引擎代码、TileLang 算子、AscendC 算子、图模式兼容性、KV Cache 实现、HCCL 通信代码。当用户需要对 xLLM NPU 相关代码进行审查时使用。
---

# xLLM NPU 代码审查

## 概述

针对 xLLM 在华为昇腾 NPU 上的代码变更进行特化审查。

审查范围涵盖：
- C++ 引擎代码（`xllm/core/`）
- TileLang 算子（`xllm/compiler/tilelang/`）
- AscendC 算子（`xllm/core/kernels/npu/`）
- 图模式相关代码（GE/AclGraph 集成）
- KV Cache 实现
- HCCL 通信代码

## 工作流

### Step 1: 确定 diff

```bash
git fetch origin main --quiet
CURRENT_BRANCH=$(git branch --show-current)
MERGE_BASE=$(git merge-base origin/main HEAD)

git diff --stat $MERGE_BASE..HEAD
git diff $MERGE_BASE..HEAD
```

如果当前分支是 `main`，询问用户要审查哪些 commit。

### Step 2: 阅读项目标准

阅读 [custom-code-style.md](../../references/custom-code-style.md)（如果存在）和 xLLM 仓库的 `AGENTS.md` 中的代码风格指南。

### Step 3: NPU 特化审查

#### 3.1 C++ 引擎代码

**正确性**：
- 逻辑是否处理边界条件
- 错误处理是否全面（无静默失败）
- 类型安全（`std::optional` 正确使用）
- 资源生命周期（RAII，无泄漏，正确的清理顺序）

**性能**：
- 热路径无性能退化
- 无大数据对象（tensor、vector）的不必要拷贝
- 线程安全：正确锁机制，无数据竞争

#### 3.2 TileLang 算子

**算子开发**（对照 `tilelang-ascend-kernel` skill）：
- Python 侧 kernel 定义是否正确（`build_<kernel>_kernel(...)` / `generate_source(...)`）
- Tiling 策略是否合理（UB 切分、Block 大小）
- 多核并行度是否充分
- 精度对齐是否正确

**Wrapper**（C++ 侧）：
- 是否避免了在 wrapper 中使用 `permute`/`contiguous`/`transpose`/`reshape`/`clone`
- 精度对齐是否在 Python 侧完成
- `DISPATCH_SCHEMA` 和 `SPECIALIZATIONS` 是否匹配
- 是否使用生成的 `make_<kernel>_specialization(...)` 而非手写

**MTP / Speculative 专属路径**：
- 如果热点或改动只出现在 MTP/spec-verify 路径，先与非 MTP decode 做路径级 diff，不要直接修局部 `transpose`/`contiguous`。
- Qwen3.5 GatedDeltaNet 已知模式：非 MTP decode 使用 `causal_conv1d` 并复用 weight 预 reshape/预布局；MTP spec-verify 若使用 `causal_conv1d_update` 并手工构造 params，容易重新引入 input/state/weight layout 适配和 transpose。
- review 时看到 `run_spec_verify_conv()`、`CausalConv1dUpdateParams`、`conv_weight.transpose(0,1).contiguous()`、`mixed_qkv.transpose(1,2)`，必须追问为什么不能复用非 MTP 的 `causal_conv1d` 路径或等价 fused spec-verify causal conv。
- PR 结论要标明这是“局部减损”还是“结构性修复”；只减少 transpose 调用不等于已经修复根因。

**语义保持**（Triton → TileLang 转换时）：
- 控制流、masked、参数语义是否保持
- 比较操作是否正确使用 `T.tile.compare`（产生 bit mask 而非 float tensor）
- `VSEL_*` 模式是否与源操作数匹配

#### 3.3 AscendC 算子

**Buffer 管理**：
- Global Memory 对齐是否正确
- Shared Memory 使用是否合理
- Unified Buffer 容量是否溢出

**指令流水线**：
- Vector 指令序列是否最优
- 是否利用了硬件 double buffer
- MTE（Memory Transfer Engine）拷贝是否合理

**精度**：
- FP16/BF16 混合精度是否正确
- 特殊值（NaN、Inf）处理

#### 3.4 图模式兼容性

**GE/AclGraph**：
- 新增算子是否支持图编译
- 是否引入 Graph Break（导致 fallback 到 eager 模式）
- 动态 shape 参数化是否正确
- 图模式下的内存布局是否与 eager 一致

#### 3.5 KV Cache 实现

**正确性**：
- PagedAttention 的 block_table 和 slot_mapping 是否正确
- NZ 格式转换是否有精度损失
- Prefill 和 Decode 的 KV Cache 写入是否对齐
- Block 分配/释放的线程安全

#### 3.6 HCCL 通信代码

**正确性**：
- AllReduce/AllGather 调用规范
- 通信-计算重叠的正确性
- 通信完成后的同步
- 错误处理和超时

#### 3.7 精度问题

- FP16/BF16 混合精度场景
- KV Cache Prefill/Decode 对齐
- 累加操作的精度（是否需要更高精度中间结果）
- Softmax 的数值稳定性

### Step 4: 审查输出

```markdown
## 审查结果

### 优点
[file:line 引用，具体的良好实践]

### 问题

#### Critical (必须修复)
[bug、安全漏洞、数据丢失风险、功能破坏]

#### Important (应该修复)
[NPU 特化问题：图模式不兼容、精度风险、通信问题]

#### Minor (可选)
[风格、优化建议、文档改进]

每个问题包含：
- **File:line** 引用
- **What** 问题是什么
- **Why** 为什么重要
- **How** 如何修复（如不明显）

### 建议
[更广泛的改进建议]

### 评估

**是否可以合并？** [Yes / No / With fixes]

**理由：** [1-2 句技术评估]
```

## 审查规则

**DO**：
- 审查 diff 范围内的代码
- 按真实严重程度分类
- 给出 file:line 精确引用
- 解释问题的重要性
- 承认优秀实践
- 给出明确结论

**DON'T**：
- 不通过审查就批准
- 将 nitpick 标为 Critical
- 对 diff 外的代码给意见
- 模糊建议（如"改进错误处理"不具体化）

## 真实案例：PR #1536 MTP Transpose 消除

### 背景

`qwen3_gated_delta_net_base.cpp` 中的 MTP (Multi-Token Prediction) 在 A3 上 crash：`causal_conv1d_update_v2 expects width in [1, 6], got 5120`。

### Diff 审查

```cpp
// qwen3_gated_delta_net_base.cpp L447-449 (weight load)
auto w = conv1d_->weight();
w.set_(torch::nn::functional::linear(...).set_data(
    tensor.set_(w.transpose(0,1).contiguous())));
conv_weight_transposed_ = tensor.clone();  // 新增：保存原始 [C/tp, k]
```

```cpp
// qwen3_gated_delta_net_base.cpp L581, L588 (kernel 调用)
// 修改前：使用 conv_weight (shape = [k, C/tp])
// 修改后：使用 conv_weight_transposed_ (shape = [C/tp, k])
auto conv_out = causal_conv1d_update_v2(x, conv_weight_transposed_, ...);
```

### 审查结果

#### Critical (已修复)
- **`qwen3_gated_delta_net_base.cpp:581`** — `conv_weight` 在 `.set_()` 后被改为 `[k, C/tp]`，传给 v2 kernel 时 width=5120 (channels) 而非 width=4 (kernel_size)，导致运行时 crash
- **修复**: 新增 `conv_weight_transposed_` 成员保存原始 `[C/tp, k]` shape，在 kernel 调用时使用

#### Important
- **`qwen3_gated_delta_net_base.h:93`** — `conv_weight_transposed_` 声明为 `mutable`，确保 `const` 方法中可修改，但需确认多线程访问安全

#### Minor
- 建议添加注释说明 `conv_weight_transposed_` 存在的原因（weight shape 在 load 后被 mutate）

### 评估

**是否可以合并？** Yes (after fix)

**理由：** 修复正确解决了 weight shape 传递反转问题，v2 kernel 对 kernel_width ∈ [1,6] 的约束得到满足。

## 参考资料

按需加载：
- [references/npu-code-patterns.md](references/npu-code-patterns.md) — NPU 代码模式规范
- [references/common-pitfalls.md](references/common-pitfalls.md) — 常见陷阱清单
