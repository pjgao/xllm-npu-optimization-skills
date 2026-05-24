# xLLM NPU 优化关键环境信息（必须记住）

## 1. 远程机器
- **主机**: 154（SSH config: `154`，IP: 192.168.13.154）
- **容器**: `xllm-gpj`（Docker 容器名）
- **代码目录**: `/home/g00510989/xllm/xllm`
- **启动脚本**: `/home/g00510989/xllm/xllm.sh`

## 2. 编译方式
- `python setup.py build` → 增量编译，生成二进制产物
- **严禁删除 build 目录**！否则需要全量重新编译，非常耗时

## 3. 网络代理
git 操作失败时需设置：
```bash
export http_proxy=http://127.0.0.1:6789
export https_proxy=http://127.0.0.1:6789
```

## 4. 历史优化数据目录
- `/home/g00510989/runs/20260523_qwen35_27b_npu_sota/`

## 5. 当前 PR 状态
- PR: https://github.com/jd-opensource/xllm/pull/1536
- 最新 commit: `1d07999f`（含 layout contract 正确性修复）
- 需要重跑 profiling + benchmark + 精度验证

## 6. 本地仓库
- `/home/gaopengju/projects/xllm`（已 amend 并 force-push `1d07999f`）
