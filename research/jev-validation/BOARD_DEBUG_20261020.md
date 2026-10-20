# Orange Pi 板卡连接与健康诊断

检查日期：2026-10-20；首轮约 23:18–23:28，重启复查约 23:30–23:31（Asia/Shanghai）。执行路径：Mac → Windows Tailscale → WSL OPiUbuntu → 板卡。首轮为只读诊断，随后按用户明确要求同步磁盘并正常重启一次；未更改网络、服务、内核、模型或磁盘配置。

## 结论

**板卡现在可通过 Windows 跳板连接，地址仍为 192.168.10.11，已连接 11045。** 连续 5 次新建 SSH 会话全部成功，现有 Web 和模型健康接口可用。此前一次 “No route to host” 未复现，根因尚未确定；不能把重新连接成功说成已经修复了某个确定故障。

板卡存在需要跟进的 MMC 驱动超时和两个服务启动失败。当前没有在可见的本次启动日志中发现 ext4 损坏、块设备 I/O 错误或只读故障重挂载；这不等于完成离线文件系统检查，更不能证明 TF 卡或供电绝无问题。

## 连接证据

| 项目 | 结果 |
| --- | --- |
| Windows 跳板 | 100.113.74.31；局域网地址 192.168.10.13 |
| 板卡 | root@192.168.10.11，orangepizero3w |
| Wi-Fi | 11045；2.4 GHz，信道 6，20 MHz；信号约 −59 dBm |
| 默认网关 | 192.168.10.1 |
| SSH 新会话测试 | 5/5 成功；每次执行 true，约 612–865 ms，中位数 724 ms |
| 网关 ping | 6/6 成功；1.53–59.36 ms，均值 11.87 ms，首包偏慢 |
| 板卡 DNS 解析 | systemd-resolved active；github.com 解析成功 |
| 模型服务 | 127.0.0.1:8081/health 返回 HTTP 200、status=ok |
| 页面服务 | 127.0.0.1:8080/ 返回 HTTP 200 |

SSH 时间是在 WSL 内计量的新连接、认证和执行命令总时间，不是模型延迟，也不是纯网络 RTT。板卡 ping Windows 无响应，但反向 SSH 正常，不能单凭 ICMP 推断 Windows 不在线。Wi-Fi power save 当前开启；仅凭首包较慢不能认定它是失联原因。

NetworkManager 本次启动日志显示连接成功，最近两小时没有新事件。boot-audit 中字段 disconnects 是宽泛关键词匹配计数，包含正常启动时的 disconnected 状态和 reason 字段，**不是 16 次掉线**。本次 TCP 22 探测会建立连接后立即关闭，相应 sshd kex 日志不代表一次真实登录故障。

## 当前系统资源（空闲快照）

| 项目 | 结果 |
| --- | --- |
| 系统 | Orange Pi 1.0.0 Jammy / Ubuntu 22.04.5；aarch64 |
| 内核 | 6.6.98-sun60iw2 #1.0.0 |
| 运行时间 | 约 3 天 4 小时；首轮检查间 boot ID 未变；随后按要求重启，见末节 |
| CPU | 6×A55 + 2×A76；最高频率约 1.794 / 2.002 GHz |
| CPU 当前负载 | load average 约 0.00；vmstat 短采样 idle 99–100%，iowait 0 |
| Linux 可见物理内存 | 5853 MiB；available 约 4418 MiB |
| zram swap | 2926 MiB；已用约 91 MiB；采样中无持续换入换出 |
| 根分区 | /dev/mmcblk1p1，ext4，rw；58 GiB 总量、46 GiB 已用、约 12 GiB 可用（80%） |
| inode | 已用约 3% |
| 温度 | 大核约 33.7°C、小核约 34.9°C、NPU 约 35.2°C；各传感器约 31–35°C |
| CPU 当前频率 | ondemand，空闲约 416 MHz |

这是空闲状态，并非模型推理、全链路或长时压力测试数据。/proc/pressure 不存在，表示当前内核未暴露该接口，不能用它估计资源压力。操作系统可见内存不是芯片规格表的“最高 16 GB”。

## MMC 超时：已经确认设备归属，原因仍待验证

通过 sysfs 对照：

| 控制器 | Linux 设备 | 用途 |
| --- | --- | --- |
| 4020000.sdmmc | mmc1 → mmcblk1；SD，SN64G | TF 卡，根文件系统所在设备 |
| 4021000.sdmmc | mmc2，SDIO；wlan0 的父设备 | Wi-Fi |
| 4022000.sdmmc | mmc0，本次未显示卡设备 | 本次不据此推断介质 |

本次可见 kernel journal 共约 21021 行，包含 **163 条** `wait dma hold bit clear timeout`，两类控制器均出现。部分发生在启动，后续周期性调谐也出现；日志常随后输出 tuning section / tuning result。仅看这些结果，不能判断是驱动记录方式、调谐缺陷、供电、信号完整性还是介质问题。

检查到的 ext4 日志是启动只读挂载后正常切换读写；未匹配到 EXT4-fs error、I/O error、buffer I/O、CRC error、corrupt、故障只读重挂载等特征。日志没有欠压记录，但平台未必具备可靠欠压监测，所以不能由此排除电源问题。dmesg 环形缓冲区只能看到其中 36 条超时，完整程度低于本次 journal；不能把 36 当成整个启动周期数量。

根分区当前带 `commit=600`，启动日志提示 writeback data mode。较长提交间隔会增加未持久化数据遭遇意外断电的风险，具体丢失量也取决于应用 fsync 等行为。本次没有修改挂载参数，也没有在挂载中的根分区运行 fsck。

后续故障定位顺序：

1. 先备份重要文件和当前镜像，记录供电器、线材、TF 型号、内核与驱动版本。
2. 在维护窗口做受控重启、正常关机与冷启动对照，检查 MMC 错误增量和 Wi-Fi 断连是否相关。
3. 如出现文件读写失败、只读重挂载或数据校验异常，停止写入实验，离线检查文件系统；不能在线修复已挂载根分区。
4. 对比同版官方镜像/内核与已知正常 TF、电源，逐项替换，不同时改变多个变量。
5. 如需验证 power-save 对连接抖动的影响，单独 A/B 测试；不要把它当作已确认修复。

目前适合继续轻量诊断；本次未执行长时间写盘或推理压力测试。

## 两个服务失败的原因与影响

### dnsmasq.service

启动失败：`failed to create listening socket for port 53: Address already in use`。当前 systemd-resolved 占用 127.0.0.53:53；dnsmasq 配置语法检查通过。现有证据符合 DNS 监听地址/端口冲突。

作为 Wi-Fi 客户端，板卡当前 DNS 解析可用，这个失败没有阻断 SSH。若后续热点配网依赖 dnsmasq 提供 DHCP/DNS，则需要修正其服务角色和接口绑定，再验证热点功能；不能因当前能上网就认定配网服务正常。本次没有停用 systemd-resolved 或重启 dnsmasq。

### run-rpc_pipefs.mount

启动失败：`unknown filesystem type 'rpc_pipefs'`。当前内核未提供可用的 rpc_pipefs 文件系统，可能涉及内核配置或对应模块不可用，尚未进一步区分。

这与 NFS/RPC 支持有关，不是根分区 ext4 挂载失败；当前本地模型、SSH 不依赖该挂载。后续确认项目是否使用 NFS，再决定补齐内核能力或移除不需要的启动依赖。本次没有为清除 failed 状态而修改服务。

## 当前模型服务状态

进程实际加载 `qwen2.5-1.5b-instruct-q4_k_m.gguf`，别名 `qwen25-15b-q4`，使用 llama-server、2 线程、ctx-size 4096、`--n-gpu-layers 0`。这是 **CPU 上的 Qwen2.5 1.5B 系列模型**，进程 RSS 约 1.10 GiB。服务元数据参数计数约 1.777B，与模型系列名的标注口径不同。

/dev/vipcore 存在、vipcore 模块已加载，只证明驱动设备节点可见，**不证明 NPU 推理已验证**。/v1/models 声明当前模型为文本输入/文本输出。网页标题仍包含视觉模型文字，不能据此判断 VLM 已加载。

本次仅访问 health、models 和静态首页，未触发生成、切换常驻模型或启动 Jev 测速。

## 原始证据

证据目录：[results/board-debug-20261020](results/board-debug-20261020/)。

- checks.json：第一轮只读检查。
- mmc-map.txt、storage.txt、boot-audit.txt：控制器映射、存储状态及本次启动日志计数。
- failed-services.txt：失败原因及 DNS 监听端口。
- http-health.txt：HTTP 健康检查和模型元数据。
- ssh-probes.json：5 次新 SSH 会话结果。
- identity.txt、resources.txt、network.txt、wifi-log.txt：系统和网络快照。

板卡日志的日历时间与 uptime 存在校时后的不一致；持续时间以单调时钟/uptime 为准。只有当前启动日志可见，无法追溯确认此前直接断电是否产生过已恢复或未记录的问题。

## 用户授权的正常重启与复查（23:30–23:31）

用户要求：“正常关机吧，直接重启一次看看”。执行 sync 后 systemctl reboot 一次，未直接断电。约 52.07 秒后首次确认 SSH 成功；该数值包含轮询间隔，不是精确的开机耗时。

- 旧 boot ID：c01cdf78-0d98-4c5a-99b4-c07a95f2aa10。
- 新 boot ID：68eed26c-c7e4-466f-a7f2-35e00f4e6d21，已确认真实重启。
- 自动重新连接 11045，地址仍为 192.168.10.11；SSH、NetworkManager active。
- 根分区仍可读写；可用内存约 4486 MiB，zram swap 已用 0。
- 模型 health、models 和 Web 首页均 HTTP 200；恢复的是已有 CPU Qwen2.5 1.5B Q4 服务。
- 重启后温度约 32–38°C，是启动后的快照，不是负载测试。
- 新启动约 1 秒处仍有 3 条 4020000.sdmmc（TF）DMA hold 超时；截至本次复查没有匹配到 ext4/块设备 I/O 错误。
- dnsmasq 与 rpc_pipefs 两项失败仍存在。

**正常重启成功，但 MMC 异常并未消失。** 不能据此认定 TF 磕碰导致，也不能据此排除物理接触或介质问题。若继续排查 TF 接触，应先确认备份，再正常关机、断开电源、检查/重新插稳卡片。此次只是系统软重启，未进行物理插拔、电源冷启动或离线 fsck。

证据：[reboot-check.json](results/board-debug-20261020/reboot-check.json)、[post-reboot.txt](results/board-debug-20261020/post-reboot.txt)。

## 重新插卡并上电后的复查（2026-10-20 23:40）

用户完成断电插拔并重新上电后，SSH 连接成功，新 boot ID 为 e5349fcf-9a5e-47aa-a625-a0c552ab6c8e。Wi-Fi 自动连接 11045，IP 仍为 192.168.10.11，根分区读写挂载，剩余约 12 GiB。

新启动日志有 4 条 DMA hold 超时：TF 控制器 4020000 三条、Wi-Fi SDIO 控制器 4021000 一条，随后均出现 tuning result。未匹配到文件系统/块设备 I/O 错误。重新插卡与冷启动未消除该现象，不能据此将原因确定为 TF 磕碰或排除物理损伤。dnsmasq 与 rpc_pipefs 两项服务失败仍在。

首次模型 health/models 请求返回 503；约 14 秒后的复查均返回 200、health=ok，符合启动加载阶段短暂不可用，不能将首次 503 判为持续故障。当前恢复的仍是 CPU Qwen2.5 1.5B Q4。启动后温度约 34–39°C，可用内存约 4485 MiB；未执行推理或压力测试。

原始证据：results/board-debug-20261020/post-reseat.txt 和 post-reseat-model.txt。
