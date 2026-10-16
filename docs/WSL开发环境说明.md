# Windows 10 / WSL2 开发环境说明

日期：2026-10-15。本文是迁移方案，尚未在目标 Windows 电脑验证；不代表已经安装或迁移完成。

## 目标与已知配置

目标电脑为 Windows 10 专业版 22H2，内部版本19045.6466，64位，32GB内存。该配置可用于现有 Orange Pi Zero 3W 项目的模型转换、NPU分层编译、SSH部署与测评。CPU具体型号、磁盘余量和虚拟化状态仍需检查。

建议路线：Windows → WSL2 Ubuntu → Docker内的厂商Linux编译工具 → SSH/SCP → 开发板。开发板继续运行原来的Linux系统，实际NPU推理仍在A733上进行。

## 资源建议

- WSL2先分配20GB内存、8GB swap，给Windows留出空间。swap是缓冲，不可当作高速内存。
- NPU编译先采用单任务。现有单编译容器内存上限为8GB，模型转换、Python和其他进程还需额外空间。
- 不同时加载多份完整FP32模型或启动多个转换/编译任务；根据峰值内存再调整并行度。
- SSD建议预留约200GB；若保留全部历史编译中间产物，需按实际目录大小增加容量。
- 项目与Docker数据放在WSL的Linux文件系统，例如`~/projects/orangepi-zero3w`。避免将编译工作目录放在`/mnt/c`，以减少跨文件系统小文件访问开销。

可选的`%UserProfile%\.wslconfig`配置如下，实际实施时才创建。修改后需重启WSL才能生效：

```ini
[wsl2]
memory=20GB
swap=8GB
```

## 环境安装与迁移顺序

1. 确认BIOS/UEFI已开启虚拟化，安装/更新WSL2，安装Ubuntu发行版；版本优先与当前项目Python依赖兼容。
2. 选择一种Docker方案：兼容该Windows版本的Docker Desktop及WSL集成，或者WSL Ubuntu内的Docker Engine。不要同时启动两套不明确的Docker守护进程。
3. 导入现有厂商镜像`ubuntu-npu:v2.0.10.2`。记录镜像ID/归档SHA，不能只凭相同tag判断内容一致。
4. 克隆项目并切换需要的分支，目前1.5B研究在`research/qwen15-rwkv`。
5. 创建新的Python虚拟环境，按当前环境导出的依赖清单安装；不要直接复制旧`.venv`。
6. 迁入模型、分词器、校准数据及必要编译产物，校验SHA。
7. 调整旧主机绝对路径、Docker挂载路径、开发板IP和SSH配置。运行目录优先统一放在同一Linux路径下。
8. 先测试SSH只读访问，再做小图编译和部署，最后验证完整模型。

Windows 10已结束常规支持，Docker Desktop最新版的兼容要求可能变化。实施时核对微软和Docker官方当前要求；不要默认所有最新版本都支持本系统。厂商编译镜像按x86-64 Linux使用，不将本方案直接套用于Windows ARM机器。

## 必须额外备份的内容

GitHub仅包含代码、文档和部分精简证据，不是完整环境备份。

| 内容 | 迁移方式与要求 |
| --- | --- |
| 厂商Docker镜像 | 在原机器用`docker save`导出，目标用`docker load`导入；验证镜像身份 |
| 原始模型、GGUF、分词器 | 单独复制并核对哈希，保留模型版本与来源 |
| ONNX、校准数据、NPU二进制 | 复制当前有效候选及必要失败样例，保留依赖关系与符号链接 |
| 原始评测和trace | 需要复现的结果另行归档，大文件通常不在GitHub |
| Python依赖 | 保存清单及Python版本，在WSL重建环境 |
| SSH密钥及访问配置 | 单独安全迁移或新建密钥授权；不提交Git、不放公共归档 |

当前研究工作目录：`/home/ubuntu/Documents/ChatGPT/端侧minibot/qwen15-rwkv-worktree`。

原项目目录：`/media/ubuntu/2e819926-8686-430b-9faf-c1a95a1c87b9/yejy/Files/orangepi-zero3w`。

两个目录均需按实际文件分布清点；研究目录中可能有指向原项目的符号链接，不能只复制链接而漏掉目标文件。迁移前生成文件清单和哈希，保留原机器可回退状态。

## 编译效果是否与Ubuntu一致

同一模型、校准数据、编译镜像和参数，在WSL2中应可复现原Ubuntu的功能效果；这需要实测确认。Windows不会天然降低板端精度或token/s。电脑端编译耗时会受CPU、SSD、WSL开销及32GB内存限制影响。

编译二进制可能因时间戳或工具链非确定性无法逐字节一致，因此分两级验证：先比较哈希；哈希不同时，继续比较实际输出和板端性能，而不是直接宣布不兼容。

验收建议：

1. 固定工具链、权重、校准数据及编译命令。
2. 分别编译已有参考的decoder图与输出头，在同一块板上比较输出误差和耗时。
3. 再比较完整模型，保持输入、KV历史、生成长度、采样设置、频率与近似温度一致，至少重复5次。
4. 记录板端精度、decode token/s、TTFT、内存与温度；编译时间单独报告。

当前1.5B NPU方案仍存在量化退化，迁移验收首先检查能否复现相同结果，不把“复现已知失败候选”写成“模型质量通过”。迁移本身不解决现有NPU瓶颈。

## 网络与部署

Windows/WSL需要能访问板端SSH地址。公司Wi-Fi的客户端隔离、VPN和防火墙可能阻断连接；同SSID不保证互通。先从Windows测试，再从WSL测试，以区分网络位置与WSL网络问题。

开发板配网和Windows开发环境是两件事。配网程序应独立运行在板上，不依赖WSL或某台办公电脑常开。相关方案见[配网程序说明](配网程序说明.md)。

## 参考

- [微软WSL安装说明](https://learn.microsoft.com/windows/wsl/install)
- [微软WSL配置说明](https://learn.microsoft.com/windows/wsl/wsl-config)
- [Docker Desktop Windows要求](https://docs.docker.com/desktop/setup/install/windows-install/)
- [Docker Engine Ubuntu安装说明](https://docs.docker.com/engine/install/ubuntu/)

上述链接供实施时核对当前要求；本文不声称已验证其当前所有版本组合。
