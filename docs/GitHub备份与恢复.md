# GitHub备份与恢复

用户指定仓库：<https://github.com/Behavior-Cloning/OragnePi-3W_ForVLM>。仓库名称沿用用户提供的拼写。

备份包括源码、脚本、systemd配置、固定模型来源、评测场景、文档、最终原始评测JSON及KV采样。模型权重、系统镜像、下载的第三方SDK/参考仓库、生成的NBG/ONNX、虚拟环境、凭据及机器配置按`.gitignore`排除。

这不是整张TF卡或系统镜像的备份。恢复时需要重新下载模型、准备兼容系统及SDK；模型版本与校验值见`configs/vlm-models.json`，系统镜像记录见根README。原有Wi-Fi与SSH凭据不随仓库发布。

```bash
git clone https://github.com/Behavior-Cloning/OragnePi-3W_ForVLM.git
cd OragnePi-3W_ForVLM
```

从[文档导航](README.md)、[项目总交接](HANDOFF.md)和[当前TODO](TODO.md)开始。2026-10-16当前1.5B研究在`research/qwen15-rwkv`分支，clone后需切到该分支才能看到最新研究代码和交接；不假定main已包含这些提交。

迁移Windows见[WSL开发环境说明](WSL开发环境说明.md)。先阅读部署脚本再在目标板上安装；不要直接复制历史IP、磁盘设备名或旧烧录命令，也不要只复制符号链接而遗漏模型/SDK目标。

本机origin保留用户提供的HTTPS地址。本次推送可用已有SSH密钥，通过命令级URL重写认证，不改全局Git配置。是否完成远端备份以远端分支SHA与本地HEAD一致为准；不能把创建本地commit当作已上传。
