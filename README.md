# Orange Pi Zero 3W 项目目录

本目录是机器人项目的统一 Git 仓库：硬件调研、设计文档、源码、脚本和测试一起维护。
后续开发以这里为准：

```text
/media/ubuntu/2e819926-8686-430b-9faf-c1a95a1c87b9/yejy/Files/orangepi-zero3w
```

## 开发入口

- [文档总导航](docs/README.md)：当前报告、部署手册、硬件选型与历史资料。
- [批量Prefill与CPU对标](npu/kv-cache/批量Prefill与CPU对标.md)：500M首字1.92→0.86秒；1B增量首字1.21→0.316秒，decode仍未超过CPU，含失败样例与内存代价。
- [MLP选择性量化加速](npu/kv-cache/MLP选择性量化加速.md)：上一阶段500M 13.92、1B 13.22 token/s；完整生成精度、固定历史五轮对照及配置审计。
- [Decoder融合与GQA优化](npu/kv-cache/Decoder融合与GQA优化.md)：双层计算图、减少K/V展开、完整输出核对及相同条件的NPU对照。
- [Q8 完整模型优化进展](npu/kv-cache/Q8完整模型优化进展.md)：此前500M/1B全层W8精度、输出头提速、五次对照及量化算法说明。
- [NPU INT8误差修复](npu/kv-cache/INT8误差定位与修复.md)：500M混合精度完整生成复测、1B输出头修复与未达标项。
- [完整模型验收计划](npu/kv-cache/完整模型验收计划.md)：500M/1B分别对比最优合格CPU，冻结精度与速度门槛。
- [NPU全深度生成与算子核对](npu/kv-cache/全深度生成与算子核对.md)：500M语言部分32层＋输出头＋真实token回送，FP16生成、CPU对照及INT8失败证据。
- [NPU KV cache适配进度](npu/kv-cache/README.md)：增量decoder、六模型层级验证、共享内存与实现边界。
- [GitHub备份与恢复](docs/GitHub备份与恢复.md)：备份范围、恢复步骤及大文件排除规则。
- [3B端侧选型与NPU可行性](docs/端侧3B选型与A733-NPU可行性.md)：5–10 token/s候选、AX650N生态、A733本次NPU试跑及LLM适配限制。
- [六模型最终评测总览](docs/所有模型评测总览.md)：已结项；陪伴质量、72项KV增量性能、系统开销与结论。
- [KV缓存增量性能报告](docs/benchmarks/2026-10-10-kv-fan/KV缓存增量性能报告.md)：三档窗口、四档history，全部实测表、曲线和原始采样。
- [六模型陪伴质量汇总](docs/六模型陪伴质量汇总.md)：全部候选统一16场景首轮结果、截断覆盖与原话证据。
- [两个Qwen陪伴模型首轮观察](docs/陪伴模型首轮观察.md)：真实多轮失败样例、速度、内存及部署验收。
- [陪伴机器人第一阶段评测](docs/陪伴机器人第一阶段评测.md)：16个多轮场景、统一人设、自然度/角色稳定评分与离线准备。
- [SmolVLM 图片对话网页](docs/SmolVLM网页部署.md)：源码、模型版本、下载、安装与验证状态。
- [SmolVLM 板端实测报告](docs/SmolVLM-256M实测报告.md)：延迟、吞吐、资源、冷启动、温度与效果样例。
- [SmolVLM 提示词优化](docs/SmolVLM提示词优化.md)：当前 system prompt、真实对比结果及中文对话限制。
- [对话保存与上下文](docs/对话保存与上下文.md)：删除按钮、浏览器历史、4096-token 窗口和超长处理。
- [256M / 500M / 1B 板端对比](docs/多模型板端对比.md)：同条件测试、真实失败样例、资源开销与模型切换。
- [硬件与端侧推理调研](docs/掌上多模态机器人硬件与端侧推理调研.md)：架构、选型、模型性能和便携续航。
- [原始调研摘录](docs/原始调研摘录.txt)：原始讨论存档，结论以整理文档为准。
- [源码目录](src/README.md)：后续采集、感知、大脑接口、动作和设备通信代码。
- [测试目录](tests/README.md)：网页集成测试与硬件验证。
- [NPU 适配与复测](npu/README.md)：已经运行成功的板端验证。

## Git 管理

代码与文档放在同一个仓库，主分支为 `main`。系统镜像、模型权重、下载的参考仓库、
运行日志和密码配置通过 `.gitignore` 排除；这些本地文件仍保留在磁盘上。
需要记录的实验结论、模型来源与校验值放进文档，不依赖未跟踪的大文件。

```bash
git status
git diff
# 按本次工作选择文件，检查暂存内容后再提交：
git add README.md docs/ src/ scripts/ tests/ npu/README.md npu/runtime-sha256.txt .gitignore .gitattributes
git diff --cached
git commit -m "Document hardware research and project setup"
```

远端仓库由用户指定：[Behavior-Cloning/OragnePi-3W_ForVLM](https://github.com/Behavior-Cloning/OragnePi-3W_ForVLM)。备份范围见上述说明，镜像、模型和凭据不上传。

## 板端历史状态（2026-10-08）

- 板卡：Orange Pi Zero 3W（A733，6GB LPDDR5）
- 推荐系统：Ubuntu 22.04 Jammy Server，Linux 6.6.98
- 官方文件：`images/Orangepizero3w_1.0.2_ubuntu_jammy_server_linux6.6.98.7z`
- 下载状态：已从 Orange Pi 官方百度网盘分享下载完成，并通过 `7z t` 完整性测试。
- 压缩包大小：`686359639` 字节
- 压缩包实际 MD5：`67a512262c63bab693ef68c2e44b89b8`
- 压缩包实际 SHA-256：`bb78cd785ce226ab65cef7295a3c7abe084310e8a6c1b68d1cfa0e01af47e6ec`
- 百度分享 API 的云端元数据 MD5 解码后是 `698ce3569bc137a14b45a6a54173d6ce`，它采用百度分片/合并后的云端口径，与下载字节的整文件 MD5 不同；不使用它判断本地文件损坏。
- 已解压镜像：`images/Orangepizero3w_1.0.2_ubuntu_jammy_server_linux6.6.98/Orangepizero3w_1.0.2_ubuntu_jammy_server_linux6.6.98.img`
- 镜像大小：`3636461568` 字节
- 镜像 SHA-256：`a569574cd0fdf4745f23f5e6c81cb652ce95f3f6d7bd506912ef5a1c58c90a79`，已用压缩包内附带的 `.img.sha` 校验成功。
- 烧录状态：已写入 64GB TF 卡（写入目标当时为 `/dev/sdb`），并完成镜像范围逐字节回读比对，结果 `FLASH_VERIFY_OK`。卡上已识别出 `ext4` 根分区，卷标为 `opi_root`；随后已安全断电弹出。
- 无头启动：已在 TF 卡 `/boot/orangepi_first_run.txt` 中启用首次启动 Wi-Fi，SSID 为 `Robot_Z1`、国家码为 `CN`、DHCP 自动获取地址；密码未写入本项目文档。配置设为首次启动完成后自动删除。
- 首次启动结果：成功。主机名 `orangepizero3w`，热点分配地址 `192.168.0.197`，SSH 已验证可登录，内核为 `6.6.98-sun60iw2`。
- 首次启动后根分区已自动扩展到约 `57.9G`；8 核 CPU、约 `5.7GiB` 内存均识别正常。
- Wi-Fi 信号约 69%，HTTPS、DNS、NTP 均正常；空闲 SoC 温度约 52°C。
- `/boot/orangepi_first_run.txt` 已按设置在首次启动完成后自动删除。
- 当前 `systemctl` 状态为 `degraded`：`run-rpc_pipefs.mount` 和 `dnsmasq.service` 失败；两项目前不影响 Wi-Fi、SSH 或外网访问，尚未修改服务配置。
- NPU 已完成首轮适配验证：内核 `vipcore` 1.13.0、`/dev/vipcore`、VIPLite 2.0.3.2 均正常；预装 `vpm_run` 与 YOLOv5 图片检测已真实运行成功。复测命令与结果见 `npu/README.md`。

## 目录

- `images/`：系统镜像和解压后的文件（Git 忽略）
- `docs/`：调研、架构、手册、引脚图和硬件资料
- `src/`：应用源代码
- `scripts/`：烧录、校验和部署脚本
- `tests/`：应用测试与回归测试说明
- `logs/`、`outputs/`、`data/`：本地运行数据（Git 忽略，按需创建）
- `models/`：模型权重（Git 忽略，按需创建）
- `configs/local/`：本机配置（Git 忽略，按需创建）
- `npu/`：适配记录与运行库清单受版本管理；参考仓库和生成结果默认忽略

## 没有 TF 卡能不能启动

官方产品页明确写的是：eMMC 可选 8/16/32/64/128GB，UFS 可选 32/64/128GB，默认空焊。产品图背面也能看到对应的存储焊盘；6GB 只代表 LPDDR5 运行内存，不代表有 6GB eMMC/UFS。除非购买链接明确写了 eMMC/UFS 容量，否则应按“没有板载存储”准备。

如果你收到实物，可以看背面：TF 卡槽在底边；eMMC/UFS 是靠近右侧的 BGA 存储芯片/焊盘。只有裸露焊盘而没有黑色芯片，就是未焊接。

建议购买一张质量可靠的 32GB 或 64GB A1/A2 microSD/TF 卡。首次启动、刷机和排障都用 TF 卡最省事。没有 TF 卡时可以研究 A733 的 USB 下载/MaskROM 方式，但它依赖主机工具和对应 BSP，通常只能临时加载或救砖，不能替代正常存储。

## 官方下载页

<http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-Zero-3W.html>
