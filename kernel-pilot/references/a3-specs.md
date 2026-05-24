# 昇腾 910B3 (A3) 硬件规格

> 用于 kernel-pilot 算子开发时的硬件约束参考。

## 芯片规格

| 参数 | 规格 |
|------|------|
| 芯片型号 | Ascend 910B3 |
| 代号 | A3 |
| AICore 数量 | 32 |
| AICore 架构 | Da Vinci v2 (dav-2201) |
| FP16 算力 | ~320 TFLOPS (单卡) |
| BF16 算力 | ~320 TFLOPS (单卡) |
| INT8 算力 | ~640 TOPS (单卡) |
| HBM 容量 | 64GB HBM2e |
| HBM 带宽 | ~1.6 TB/s |

## 存储层次

### 片上存储

| 层级 | 容量 | 说明 |
|------|------|------|
| Unified Buffer (UB) | 2MB per AICore | 双 buffer 操作，实际可用约 1MB per tile |
| L1 Cache | 384KB per AICore | 矩阵运算高速缓存 |
| L0A / L0B / L0C | 架构内置 | Cube Unit 专用寄存器 |

### 片外存储

| 层级 | 容量 | 带宽 | 说明 |
|------|------|------|------|
| HBM | 64GB | ~1.6 TB/s | 全局内存 |
| L2 Cache | 共享 | - | 容量依具体配置 |

## Tiling 约束

- tile_size 确保 double buffer 不超 UB：`2 tiles of input + 1 tile of output < 2MB`
- 典型约束：`tile_M * tile_K * 2(fp16) + tile_K * tile_N * 2 < 2MB`
- `block_num` 建议 = AICore 数量 × 2 = 64（保证流水线填充）
- tile 尺寸建议对齐到 16B（fp16 = 8 elements 对齐）

## 互联

| 参数 | 规格 |
|------|------|
| HCCS 带宽 | ~240 GB/s (uni-directional) |
| HCCL 支持算子 | AllReduce, AllGather, ReduceScatter, Broadcast, Send/Recv |
| TP 典型配置 | 2/4/8 卡 |

## 通信原语性能参考

| 操作 | 规模 | 延迟参考 |
|------|------|---------|
| AllReduce (TP=2) | 4096 elements | <0.1ms |
| AllReduce (TP=2) | 1M elements | ~0.5ms |
| AllReduce (TP=8) | 1M elements | ~2ms |

## 算子执行模型

| 单元 | 功能 | 适用算子 |
|------|------|---------|
| Cube Unit | 矩阵乘法 | MatMul, Conv, BatchMatMul |
| Vector Unit | 向量运算 | Add, Mul, Exp, Softmax 等 |
| Scalar Unit | 标量运算 | 控制流、寻址 |
| MTE (Memory Transfer Engine) | 数据搬运 | DMA、scatter/gather |

## 开发工具链

| 工具 | 版本 | 路径 |
|------|------|------|
| CANN | 8.0.RC1+ | `/usr/local/Ascend/ascend-toolkit/latest` |
| HDC Driver | 25.2.0+ | - |
| TileLang | 内置 | `third_party/tilelang-ascend/` |
| Triton-Ascend | 内置 | `third_party/triton-ascend/` |
| msprof | CANN 内置 | `{npu_home}/include/experiment/msprof` |
| npu-smi | 系统 | `npu-smi info` 查看设备状态 |
