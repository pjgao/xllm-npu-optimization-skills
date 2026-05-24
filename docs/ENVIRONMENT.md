# xLLM NPU 优化环境关键信息

## 远程机器
- **机器**: 154（具体 IP 见 SSH config 或 `.ssh/config`）
- **容器名**: `xllm-gpj`
- **代码目录**: `/home/g00510989/xllm/xllm`
- **启动脚本**: `/home/g00510989/xllm/xllm.sh`
- **历史中间数据**: `/home/gaopengju/runs/20260523_qwen35_27b_npu_sota/`

## 编译与运行
- **编译命令**: `python setup.py build`（增量编译，生成二进制产物）
- **严禁删除 build 目录**！否则需要全量重新编译，耗时很长
- **运行**: 通过 `xllm.sh` 启动推理服务

## 网络代理（git 失败时使用）
```bash
export http_proxy=http://127.0.0.1:6789
export https_proxy=http://127.0.0.1:6789
```

## 当前 PR
- **PR**: https://github.com/jd-opensource/xllm/pull/1536
- **目标分支**: `jd-opensource/xllm:preview/qwen3.5-qwen3.6`
- **最新 commit**: `1d07999f`（含 layout contract 正确性修复）
- **本地仓库**: `/home/gaopengju/projects/xllm`（branch: `feat/qwen35-mtp-transpose-elimination`）

## 模型
- **当前优化目标**: Qwen3.5-27B（MTP, nst=1）
- **架构**: GatedDeltaNet (FLA)

## 上次验证数据（commit `09ffad42`，未含 layout fix）
- msprof: Transpose 960 calls / 17.3ms
- benchmark: 39.54 tok/s (+9.5%), TPOT 21.92ms, TTFT -10.1%
- accept rate: 47.7%

## 待验证（commit `1d07999f`，含 layout fix）
- 需重跑精度、msprof、benchmark
- commit `1d07999f` 多了 1 次 transpose（spec verify 后），实际 kernel 数可能微增
