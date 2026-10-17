# WSL 复现历史 15.598 tok/s 与对应精度（2026-10-18）

目标是历史完整 InternVL3-1B 语言模型的 cap1 + headMSE 配方。正式代码和编译工作均在 Windows F 盘 / WSL，Mac 仅用于远程调度，板卡经 Windows/WSL 访问。

## 已完成的确定性复现

- 从本轮 Q8 GGUF 导出并重新校准的 28 个 INT16 基础图构建，未用历史 NBG 作为编译输入。
- 完整 24 层解码器，两层融合成 12 块；容量 1280；prefill 为 16 token，共 12 块；输出头为完整 151674 词表的 4 个分片。
- 12 decode + 12 prefill + 4 head 共 28 个新 NBG，SHA-256 全部等于历史对应产物。
- 24 个融合图的 decoder.json 与 decoder.data 字节一致；量化 YAML 文本序列化不同，但解析后的张量参数 24/24 完全相同。
- 通过 Windows 上传所有 28 个新 NBG；板端逐文件 SHA-256 核验通过，embedding 在核对哈希后复用硬链接。
- 同一历史 prompt、feedback，1024 已缓存历史 + 32 新输入，完整 128 次预测。新板端 logits（77,657,088 字节）与历史逐字节一致。
- logits SHA-256：3ac62892e34fa4a7448ad65c05b0dbe193bd73056c753535619178e034abb9b2。

## 精度（本轮独立重新计算）

| FP32 参考 | 全 logits 相对 L2 | 最低逐步 cosine | argmax 一致 | 平均 KL | PPL 比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 实际导出量化权重 | 0.0427090265 | 0.995529473 | 127/128 | 0.00170823024 | 1.008959615 |
| 原始 Q8 GGUF 解量化权重 | 0.0828495547 | 0.992111325 | 125/128 | 0.00939975120 | 1.031230652 |

上述数字均与历史对应值完全相同。两种参考衡量不同误差，不能混用：量化权重参考主要核对执行误差，原 Q8 参考包含权重再量化和低精度算术的总差异。L2 不是问答正确率。

## 差异根因与修复

1. 先前 6.98 tok/s 是 capacity=128、单层未融合、短提示自然输出的基线，与历史固定轨迹和性能模式不同。
2. 恢复 compact GQA、split KV、current matmul、INT16 softmax、两层融合、12 块分组、四分片 head 分组、fast greedy、prefill16。
3. gate/up 采用受限 MSE DFP，额外小数位最多 1；第 19、20 层乘积范围扩大 1 bit。
4. 历史第 2、3 层的 conversion.json 参数标签不足以复现实际量化表：当前编译器需 activation-headroom=0、MLP gate/up headroom=1，并保留 MLP 非线性与 down 的 FP16。先前误差版本的配置、日志和哈希证据保留；修复后两层融合的 decode、prefill 二进制均恢复历史哈希。
5. 输出头采用 DFP MSE、activation headroom=2、float-head-norm=false，4/4 哈希一致。

## 五轮性能结果（已完成）

- 五轮 decode：15.595251、15.582246、15.567494、15.599455、15.539996 tok/s。
- 中位数 **15.582246 tok/s**，历史为 15.598395，相差 -0.10353%；单轮最高 15.599455。
- TTFT 中位数 **0.629021 s**；总响应中位数 8.779322 s。
- 24 层 decoder 和完整词表 head 均在 NPU 执行；CPU 承担 embedding 查表、数据搬运、调用调度和 greedy 选择等辅助工作。
- 每轮 NPU 频率前后均为 1,008,000,000 Hz；正式响应起温 56.544–57.288°C；每轮驱动释放后占用 0 bytes。
- NPU 驱动分配 753,274,680 bytes（约 718.38 MiB）；进程峰值 RSS 中位数29.375 MiB，不能把进程 RSS 当作模型总内存。
- 五轮预测 token 全部与历史一致。完整 logits 的诊断轮用于精度核对，正式性能轮关闭转储。
- 条件是 1024 token 已缓存历史 + 32 新输入、128 次固定 feedback 预测；TTFT 不包含约 19.9 s 的历史缓存准备及模型加载。不能把这个 TTFT 当成冷启动处理全部 1056 token 的时间。

首轮温控未开始正式测量就超时：旧预热要求到约 70°C，再回到 55–65°C；当前板卡加热后约 65.4°C，未达到额外超调阈值。修复为在准入上沿内停止加热器、wait 回收后重新检查温度；正式响应必须仍在 55–65°C 起测。失败记录保留在 performance.json，重试使用独立 performance-attempt2.json。

## 验证范围

本报告复现的是历史 15.598 tok/s 对应的开发轨迹与精度，不代表产品独立质量全部通过。历史独立 v2 为 33/36，通过情况未在本轮重新跑完整套件；本轮没有用该留出集调参。

## F 盘清理

实验完成后清理了 29 个已校验的可重建或失败中间文件，包括中断恢复目录内与保留原始包校验相同的两个大归档，以及后续修复版本取代的编译产物。原始 GGUF、官方权重、校准源、成功模型、验证记录、slim20 与 TF 备份均保留。清理后成功模型 28/28 个 NBG 的 SHA-256 再次验证通过，保留符号链接全部有效。

WSL 文件系统空闲增加 4,864,225,280 bytes。随后压缩 OPiUbuntu 的虚拟盘，F 盘实际空闲增加 4,868,534,272 bytes（4.53 GiB），从约 59.21 GiB 增至 63.75 GiB。VHDX 从 64,467,501,056 bytes 降至 59,598,962,688 bytes。详细删除清单、原文件哈希和保留源路径见 evidence-wsl-peak155-20261018/cleanup-manifest.json。

## 重新执行

以下命令在 Windows 的 OPiUbuntu WSL 内执行。每次使用独立工作目录；第一次从 Q8 GGUF 重新导出和校准，第二步构建峰值配方，第三步上传、双 FP32 参考核对和五轮测速。保留历史目录仅用于哈希及精度比对，不把历史 NBG 当作编译输入。板卡认证沿用现有安全凭据注入或 SSH key；不要把密码写入命令、仓库或日志。validate_peak_155.py 会拒绝覆盖已有验收记录。

```bash
set -euo pipefail
repo=/mnt/f/Files/OPi-Lab/repos/OragnePi-3W_ForVLM
py=/opt/opi-lab/export-venv/bin/python
historical=/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/npu/results/full-intern1b-sharded-v2-c128
model=/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/models/internvl3-1b/InternVL3-1B-Instruct-Q8_0.gguf
run_id=$(date +%Y%m%d-%H%M%S)
work=/opt/opi-lab/peak155-$run_id
export OPI_BOARD_HOST=root@192.168.10.11
export OPI_KNOWN_HOSTS=/root/.ssh/opi_known_hosts
export OPI_BOARD_HOST_LOCK=/opt/opi-lab/restored-slim3-20261017/projects/orangepi-zero3w/npu/results/board-host.lock
cd "$repo/npu/kv-cache"
"$py" direct_source_pipeline.py --model "$model" --work "$work/source"
"$py" reproduce_peak_155.py --source "$work/source/export" --historical "$historical" --output "$work/build"
"$py" validate_peak_155.py --source "$work/source/export" --historical "$historical" --model "$model" --build-report "$work/build/build-state.json" --output "$work/validation" --tag "wsl-peak155-$run_id" --case "peak155-h1024-n32-$run_id"
```

上述命令是后续复用入口；本轮已完成的实际结果位于文末现有目录，未为补文档重复运行实验。22 项相关单元测试、Python 语法检查和 git diff --check 均通过。清理及虚拟盘压缩后，原始 Q8 GGUF、28 个成功 NBG 与完整 logits 的哈希再核对通过。

## 产物位置

- 正式仓库：F:/Files/OPi-Lab/repos/OragnePi-3W_ForVLM
- WSL 新源与新编译：/opt/opi-lab/source-validation-20261017/gguf-fresh/export
- 汇总与过程证据：/opt/opi-lab/source-validation-20261017/peak-155-20261018
- 构建入口：npu/kv-cache/reproduce_peak_155.py
- 部署/验收入口：npu/kv-cache/validate_peak_155.py
- 板端新产物：/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018
