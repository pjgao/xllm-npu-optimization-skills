# PR #1400 Qwen3-Next 权重变换竞态精度定位记录

日期：2026-05-27

## 背景

PR #1400 被怀疑引入 Qwen3-Next 精度回归。最初 10 条简单 prompt 未发现异常，因此需要把验证强度从 L2 smoke 提升到 L3 数据集子集评测，并结合代码逻辑审查确认是否为 PR 引入。

## 验证路径

### L2：10 条简单 prompt

结果：未发现明显异常，输出基本是人话。

结论：不能证明 PR 没有精度问题，只能说明模型没有整体崩坏。对于权重加载、layout、采样或特定 task 才触发的问题，L2 覆盖不足。

### L3：CEval 子集前 10 题

使用 CEval `operating_system` 和 `computer_architecture` 两个子集，每个子集前 10 题进行 A/B 对照。

| 版本 | operating_system | computer_architecture | overall |
|------|------------------|-----------------------|---------|
| 未修复 | 9/10 | 10/10 | 19/20 |
| 修复后 | 10/10 | 10/10 | 20/20 |

稳定坏例：

- subset：`operating_system`
- index：8
- target：`B`
- 未修复 prediction：`答案：D`
- 修复后 prediction：`答案：B`

## 根因分析

代码审查重点放在 Qwen3-Next attention 权重加载路径：

- safetensors 多分片可能并发加载同一层不同权重。
- `load_state_dict()` 中存在 qkv reorder、q/k norm `add_(1.0)` 等就地权重变换。
- 普通 bool 只能表达本对象状态，不能保护跨线程的临界区。
- 当多个权重分片并发进入同一层加载逻辑时，存在重复变换或状态切换竞态风险。

这类问题通常不会让服务启动失败，也不一定让所有 prompt 都异常，但会在特定知识题或长链路推理中表现为答案漂移。

## 修复思路

1. 每层 attention 增加 mutex，串行化 `load_state_dict()` 中的权重状态变换。
2. 使用 `!was_loaded && is_weight_loaded()` transition guard，避免依赖独立 bool 判断。
3. 保证 qkv reorder、q/k norm offset 等就地变换只在完整权重加载后的状态转换点执行一次。

## 经验沉淀

- 10 条简单 prompt 只能作为 L2 smoke，不能作为精度 PR 的充分验收。
- 如果 L2 正常但用户怀疑精度回归，应尽快升级到 L3：选两个高风险 task 前 N 条建立坏例。
- 对权重加载类问题，要优先查并发 safetensors、in-place tensor transform、普通 bool 状态保护和 TP shard layout。
- 精度定位报告必须保留坏例的 target、prediction、subset、index 和 commit，便于后续二分或复测。
