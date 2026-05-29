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

### 2026-05-28 编译避坑：xllm_ops 与 TileLang
- **xllm_ops 全局 OPP 冲突**：多个 git worktree 共用 `/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/xllm` 时，编译 `xllm_ops` 会互相覆盖 `libcust_opapi.so`、op_api 头文件和算子元信息，导致不同 worktree 间出现难复现的 ABI/符号冲突。
- **推荐做法**：把 `xllm_ops.run` 安装到 build-local 目录，例如 `build/cmake.linux-aarch64-cpython-311/xllm_ops_opp/vendors/xllm`，CMake 的 include/link 路径也指向该目录；后续如果 `third_party/xllm_ops` git HEAD 未变化且本地 `op_api/lib/libcust_opapi.so` 存在，就跳过 precompile。
- **运行前环境**：若运行时需要自定义 OPP，可 source 本地 `bin/set_env.bash`，或把本地 vendor 目录追加到 `ASCEND_CUSTOM_OPP_PATH`、把本地 `op_api/lib` 追加到 `LD_LIBRARY_PATH`。
- **沙箱限制**：CANN `opc`/custom op 安装流程会调用 `psutil.net_if_addrs()`，在 Codex sandbox 内可能报 `PermissionError: [Errno 1] Operation not permitted`；已有本地 OPP 缓存后，普通增量 `python setup.py build --device npu` 可跳过该步骤。
- **TileLang fused_gdn_gating**：preview 分支 pin 的 `tilelang-ascend` 与当前 `fused_gdn_gating.py` 3 参数 `T.tile.sigmoid` 写法不兼容；该 TileLang kernel 还存在历史精度风险。标准构建应只保留可用的 TileLang rope / split_qkv_rmsnorm_mrope，把 `fused_gdn_gating` 走回 `torch_npu_ops` 实现，并同步移除对应 TileLang wrapper test 目标。

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
