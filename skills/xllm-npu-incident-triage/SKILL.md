---
name: xllm-npu-incident-triage
description: xLLM 昇腾 NPU 生产事故诊断。Replay-first 流程，收集证据、分类问题、定位根因、验证修复。覆盖 AICore timeout、HCCL 通信失败、NPU OOM、KV Cache 碎片、GE 编译错误等常见问题。当用户报告 xLLM NPU 运行异常时使用。
---

# xLLM NPU 生产事故诊断

## 概述

当 xLLM 在华为昇腾 NPU 上出现以下问题时，使用此 skill 进行诊断：

| 问题类型 | 典型症状 |
|---------|---------|
| AICore timeout | 算子执行超时、NPU hang |
| HCCL 通信失败 | AllReduce 超时、节点间断开 |
| NPU OOM | 显存不足、KV Cache 分配失败 |
| KV Cache 碎片 | 可用显存够但调度失败 |
| GE/AclGraph 错误 | 编译失败、replay 失败 |
| 精度异常 | 输出与预期不符、NaN/Inf |
| 性能退化 | 相比历史吞吐下降 |
| PD 分离异常 | Prefill/Decode 延迟不对称 |

## 核心原则

1. **Replay-first**：先复现，再诊断
2. **证据链**：保留从告警到根因的完整证据链
3. **最小化复现**：缩减到能复现的最小场景
4. **修复验证**：每个修复都有回归验证

## 工作流

详见 [references/replay-workflow.md](references/replay-workflow.md)。

### Step 1: 证据保全

在问题发生后立即采集：

```bash
# NPU 状态快照
npu-smi info > artifact/npu_status.log
npu-smi info -t board -i 0 > artifact/npu_board.log

# xLLM 日志
# 收集最近的日志、错误堆栈

# 系统日志
dmesg | tail -200 > artifact/dmesg.log

# 环境变量
env | grep -E "ASCEND|CANN|NPU" > artifact/env.log

# 进程状态
ps aux | grep xllm > artifact/process.log

# NPU 使用率
npu-smi info -t usages > artifact/npu_usages.log
```

保存环境信息到 `artifact/env.json`：
- NPU 型号、数量、驱动版本
- CANN 版本
- xLLM commit hash
- 模型路径和配置
- 触发时间线和操作序列

### Step 2: 问题分类

根据症状分发到对应的子分析流程：

#### 2.1 运行时错误 → Runtime 调试

症状：进程崩溃、段错误、算子执行失败

```bash
# 开启详细日志
export ASCEND_GLOBAL_LOG_LEVEL=1   # INFO 级别
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# 检查 NPU 驱动日志
cat /var/log/dcm/dcm.log | grep -i error
cat /var/log/messages | grep npu
```

常见原因：
- 算子不支持当前 shape/dtype
- AICore 资源耗尽
- 多进程 NPU 内存冲突
- 驱动版本与 CANN 版本不匹配

#### 2.2 精度异常 → 精度调试

症状：输出 NaN/Inf、与 GPU 结果不一致

检查项：
- FP16 溢出（检查 MatMul 前后值范围）
- KV Cache Prefill/Decode 写入对齐
- Softmax 数值稳定性（是否需要 float32 中间结果）
- LayerNorm/RMSNorm 精度
- 权重加载精度（量化误差）

工具：
```bash
# 开启 dump 功能
export ASCEND_OPP_PATH=$ASCEND_HOME/opp/built-in
export ASCEND_ENABLE_DUMP=1
# 配合 xLLM dump 配置
```

#### 2.3 性能退化 → Profiling 分析

症状：吞吐下降、延迟增加

调用 `xllm-npu-profiler` skill：
1. 采集当前 profiling
2. 与历史基线对比
3. 定位退化的具体算子或路径

常见原因：
- 新增算子未融合
- 图模式 fallback（某个 shape 不支持 compilation）
- KV Cache 碎片率增加
- 调度参数回归

#### 2.4 通信异常 → HCCL 测试

症状：AllReduce 超时、TP 训练/推理 hang

```bash
# HCCL 单测
cd /path/to/hccl-test
./run_test.sh allreduce

# 检查节点间网络
npu-smi info -t health
hccl_info --device

# 检查拓扑
npu-smi info -t topo
```

常见原因：
- ROCE 网络抖动
- 节点间时间偏移
- HCCL 版本问题（检查 `hccl_version.txt`）

#### 2.5 内存不足 → 内存调试

症状：OOM killed、KV Cache 分配失败

检查项：
- `npu-smi info` 实际显存使用
- xTensor 内存池碎片率
- KV Cache block 数量限制
- Batch size 与显存的关系

```bash
# 查看显存使用详情
npu-smi info -t memory

# xLLM 内部指标
# 检查 xTensor 池的分配日志
```

#### 2.6 PD 分离异常 → 调度调试

症状：Prefill 延迟突然增加、Decode 队列堆积

检查项：
- 请求调度日志（prefill vs decode 分配比例）
- 跨节点网络延迟
- Mooncake 全局 KV Cache 状态
- 节点健康状态

### Step 3: 复现工作流

详见 [references/replay-workflow.md](references/replay-workflow.md)。

目标：
1. 确定能稳定复现的最小输入
2. 确定问题发生的条件（特定 batch size、序列长度、并发数）
3. 记录复现命令，确保团队可独立验证

### Step 4: 根因分析 + 修复验证

```markdown
## 事故报告

### 时间线
（从告警到修复的完整时间线）

### 症状
（用户看到的异常现象）

### 根因
（技术细节：哪个代码路径、哪个配置、哪个外部因素）

### 修复
（代码变更/配置变更/环境修复）

### 验证
（回归测试结果）

### 预防措施
（如何避免同类问题再次发生）
```

## NPU 错误目录

详细错误码和处理建议见 [references/npu-error-catalog.md](references/npu-error-catalog.md)。

关键错误码摘要：

| 错误码 | 含义 | 常见原因 |
|--------|------|---------|
| E39999 | AICore 算子超时 | 算子实现 bug 或硬件问题 |
| E50000 | 通信超时 | ROCE 网络或拓扑问题 |
| E82000 | 显存不足 | 显存碎片或配置不当 |
| E40010 | 图编译失败 | 不支持的算子或 shape |
| E30001 | 输入参数错误 | dtype/shape 不匹配 |

## 参考资料

按需加载：
- [references/npu-error-catalog.md](references/npu-error-catalog.md) — 完整 NPU 错误码目录
- [references/replay-workflow.md](references/replay-workflow.md) — 复现工作流模板
- [references/ascend-profiling-formats.md](../xllm-npu-profiler/references/ascend-profiling-formats.md) — Profiling 数据格式

## 实测事故案例

### MTP npu_add_rms_norm Kernel Crash (2026-05-23)

**环境**: 192.168.13.154 / xllm-gpj 容器 / Qwen3.5-27B + MTP draft  
**症状**: xLLM MTP 模式下 node_1 (Phy 9) 推理时崩溃，`terminate called after throwing an instance of 'c10::Error'`  
**错误栈**:

```
npu_add_rms_norm:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:3783
NPU function error: call aclnnAddRmsNorm failed, error code is 561002

[ERROR] 2026-05-23-15:22:01 (PID:3481803, Device:0, RankID:-1) ERR00100 PTA call acl api failed.
EZ9999: Inner Error!
Input x2/x1 shape invaild, shape is not equal x1 shape.
  [FUNC:CheckInputOutputShape][FILE:add_rms_norm_tiling.cpp][LINE:172]
Input shape invalid.[FUNC:Tiling4AddRmsNorm][LINE:367]
AddRmsNorm do tiling failed, ret is -1.
Check NnopbaseExecutorDoTiling(executor) failed
Check NnopbaseExecutorTilingAndUpdateBinInfo(executor) failed
Check NnopbaseExecutorMatchCache(executor) failed
```

**根因分析**:
1. `npu_add_rms_norm` 算子要求 x1 和 x2 shape 相同 (residual connection)
2. MTP 模式下，draft model (1 层) 与 target model (64 层) 的 hidden_states shape 在 concat/add 时不匹配
3. 可能是 xLLM MTP 实现未正确处理 draft/target model 的 hidden_dim 差异，或 draft model 权重 format 与 target 不一致

**影响**: MTP benchmark 无法执行，阻塞 Phase 1 的 MTP vs baseline 对比

**临时规避**:
- 禁用 MTP，仅跑 baseline 模式
- 向 xLLM 团队上报 kernel crash，附上完整日志：`/home/g00510989/runs/20260523_qwen35_27b_npu_sota/logs/mtp/node_1.log`

**后续跟进**:
- 等待 xLLM 修复 `npu_add_rms_norm` shape 匹配逻辑
- 或升级 torch_npu 版本 (当前 2.7.1.post2 可能不兼容 MTP feature)
- 验证修复后重新运行 `./bench.sh mtp` 对比
