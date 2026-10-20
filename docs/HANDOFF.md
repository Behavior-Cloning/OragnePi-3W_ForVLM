## Jev 逐模型优化补充（2026-10-21 01:03，进行中）

用户明确要求每个Jev优化决策速度/延迟并交给审计。审计`research/jev-validation/JEV_OPTIMIZATION_AUDIT.md`已记录62项。Nano中文tokenizer失效（100动作只有34种encoding、86.57%UNK）；Laya完整缩短基线已回传：10候选1.259秒，direct100 9.607秒6/20，two-stage2.482秒5/20，unknown/safety均0/4。完整连续样本峰值71.744℃、RSS1.922GB。不能把部署/提速称为质量达标。

WSL控制器4805执行基线共驻；4806排队启动优化。查看`results/run-board-pairs.log`、`results/run-board-optimization.log`，不直接抢板卡锁。JevStyle旧部署缺transitive LD路径，Windows已修复，等优化控制器重新同步。Laya ORT FP32和encoder-only INT8导出及四组动态形状sanity已通过，尚待板测。新优化记录源码SHA、包版本、PSS/CPU频率/swap流量。

存储服务器下载v2 PID90989/90990（curl断点续传，3线程/模型）；目标`/infinite/common/yejy_vla/jev_models`，无GPU仅下载。manifest仍running，不可报告下载完成。Tailscale保持启用，Mac→服务器直连；SSH MaxStartups间歇重置可用已有expect重新建master。

# 当前交接更新：2026-10-21 Jev 部署（进行中）

- 当前权威工作目录：F:\Files\OPi-Lab\repos\OragnePi-3W_ForVLM\research\jev-validation。Windows通过Tailscale 100.113.74.31；板卡192.168.10.11，通过Windows/WSL跳板操作并持有双锁。
- 用户已授权下载、板端测试、项目中间产物清理；原计划“只规划”限制已由后续用户指令替代。不要驱动真实电机。
- 远程116.172.93.223:19515为无GPU存储服务器，用户明确仅下载，目录 /infinite/common/yejy_vla/jev_models。SSH曾成功；后台下载9B/27B-VL已启动，但连接有MaxStartups/认证前reset，完成状态需查manifest。不要关闭Tailscale。
- Nano33、Laya322、JevStyle0.8Q4、OneJev0.8下载完成；前两者已推板并二次校验。Nano/Laya单模型筛查任务及随后JevStyle/共驻任务正在执行，详见research/jev-validation/results/run-board-*.log。不要并行启动第二个板卡controller。
- 500M/Intern1B为主要模型，保留1.5B对照。1.5B历史NPU约5.03t/s但质量失败；不得称未测过，也不得称可用INT4。W4实际INT8存储。
- 已清理28个int16 decoder.export.data共990086768字节；源权重、编译输入、最终NB、活跃FP32 oracle数据、TF备份均保留。此为WSL内部空间，不等同F盘物理缩小。
- 本轮报告：research/jev-validation/JEV_DEPLOYMENT_20261021.md。OneJev图片前向、视觉encoder共驻、音频原生Jev研究已列README TODO。尚未完成的矩阵不得写成完成。

---
以下保留此前交接历史，旧IP/关机状态以本段及最新报告为准。

# 智能体接管说明

更新日期：2026-10-14。本文是接管入口，优先于历史交接里的队列、在线状态和“继续运行”指令。研究结果基于提交 `1600c9e0959a4905df770e7eb2bc13b0c1464ab8`；本文件随后单独提交。

## 先看当前状态

- 用户要求关机安装风扇，已执行板端 `sync && systemctl poweroff`，返回成功；等待20秒后SSH连接超时。已确认关机请求被接受、板卡离线，未通过串口确认最终断电状态。
- **不要自动重启测试或把离线当故障。** 等用户完成风扇安装并重新上电，再做健康检查。工作站没有关机。
- 本轮所有测量队列均已结束，没有需要恢复的旧会话。`下次继续.md`较早章节里的PID、session及“正在运行”均为历史记录，不要照抄启动。
- 24/24对齐性能格已完成，但NPU仍慢于优化CPU，最新候选质量仍未全部通过。不能把“实验结束”写成“项目达标”。
- 本次handoff只整理文档，不改变代码、模型、生产配置或测量结果。

## 目标与不可变的比较口径

在A733 / 6GB Orange Pi Zero 3W上，优化SmolVLM-500M与InternVL3-1B的完整语言生成链路，重点是解码速度和增量首字延迟。暂不测试视觉。最终用途是陪伴机器人，关注对话自然度、角色稳定与真实多轮KV复用。

- 500M全部32层、1B全部24层，以及完整词表输出头；包括必要转换、同步、采样开销。两层探针只能定位瓶颈。
- 当前对应CPU量化基线为Q8_0；NPU主要W8/A16/KV16，敏感位置FP16，**不是纯INT8，也不是Q4**。不得拿生产网页的Qwen Q4替代这两个模型的对照。
- 每模型分别对比规定搜索范围内最快有效CPU配置，不能限制CPU线程或选慢FP32实现制造NPU优势。CPU关闭FA在当前长历史场景通常更快，具体配置以矩阵证据为准。
- 核心输入为已有有效KV历史512/1024 × 新增32/64/128 token（包含模板），同一输入和CPU反馈轨迹，固定128预测、127个decode间隔，预热后至少5轮。
- 温度以响应开始时分档：低温40–50°C、中温55–<65°C；不是全程恒温。用户要求主要低、中温，在中温附近优化，高温暂缓。风扇改变后须记录新散热条件，不把前后变化全归因代码。
- 首字为引擎增量TTFT，不含模型加载、历史KV准备、控温和HTTP缓冲；冷启动、真实网页延迟另报。
- 测量必须配套同轮CPU、内存、换页、I/O与温度。整机CPU指标8核合计为100%；RSS和driver共享内存不可直接相加当唯一物理占用。没有电功率实测。
- 原计划的最终目标为各核心档位比最优合格CPU快至少15%，同时质量、端到端、稳定性过关；这是研发目标，不是已实现或保证实现。

## 环境、路径与访问

本机项目真实路径（UUID结尾为87b9，不是旧消息里的87b91）：

```text
/media/ubuntu/2e819926-8686-430b-9faf-c1a95a1c87b9/yejy/Files/orangepi-zero3w
```

| 路径（相对项目根目录） | 用途 |
| --- | --- |
| `npu/kv-cache/` | NPU编译、运行、验证、报告源码及Markdown报告 |
| `npu/results/` | 本机完整实验、图、原始logits与运行记录；大部分不进Git |
| `npu/kv-cache/evidence-attack-20261013/` | 已归档紧凑证据 |
| `docs/` | 用户文档、调研与交接 |
| `src/`、`configs/`、`tests/` | 应用、模型配置与测试；先看各自README |

板卡最后地址为 `192.168.88.29`，SSH用户 `root`。重启后IP可能改变，不盲目操作其他设备。密码不写入Git，使用本机既有认证或由用户提供；脚本支持环境变量 `SSHPASS`，本机工具为 `/home/ubuntu/.local/bin/sshpass`。SDK入口见 [SDK获取记录](../npu/kv-cache/SDK获取记录.md) 及NPU README，先核实现有环境，避免重复下载大型工具包。

**所有板端操作（含只读）都要串行使用host与board双锁。** 先查锁和任务，`WRITE`是持锁，`WRITE*`是等待。已有runner若内置锁，不再外层重复获取同一把锁造成死锁。手动SSH的无密码示例：

```bash
# 在项目根目录运行；使用已经配置的SSH认证。
flock -w 20 npu/results/board-host.lock \
  ssh -T -o ForwardX11=no -o ConnectTimeout=8 root@192.168.88.29 \
  'flock -w 20 /run/opi-npu-kv-benchmark.lock sh -c "hostname; uptime; df -h /; free -m"'
```

关机前最后快照：TF可用约2.75GiB，RAM available 4618MiB，swap占1164MiB，driver分配0，温度38.936°C；主机空余约27GiB。这是历史空闲快照，重启后重新测量，swap占用不等于正在换页。新部署/完整logits可能写满TF，保留至少脚本要求的512MiB额外空间。只清理已在本机完整SHA核验的板端诊断副本，保留恢复清单，不删除模型、系统、唯一证据或无关文件。

## 已有结果与不能外推的边界

当前对齐矩阵24/24，每格五轮、同温档、同CPU反馈；**不同格可以是不同明确标记的候选版本，不是一个固定产品模型完成全矩阵**。

中温、新增64 token，数值为中位数：

| 模型 / 历史 | NPU / CPU token/s | NPU / CPU TTFT秒 | NPU / CPU整机CPU% |
| --- | ---: | ---: | ---: |
| 500M / 512 | 20.432 / 22.780 | 0.753 / 0.704 | 1.27 / 26.56 |
| 500M / 1024 | 19.089 / 20.373 | 1.223 / 0.902 | 1.14 / 26.35 |
| 1B / 512 | 15.935 / 18.900 | 0.793 / 0.655 | 1.24 / 26.55 |
| 1B / 1024 | 15.597 / 17.760 | 1.237 / 0.792 | 1.13 / 26.55 |

最新有效研发增量：

1. **500M prefill8**：全32层512＋64中温交替5轮，t32→t8首字0.921722→0.752878秒（−18.32%），总响应−1.95%，decode20.521247→20.432035基本持平；driver减少32.73MiB。只测这一性能格，其他格不是t8。执行L2约0.03685、最小cos约0.99667；原Q8 KL约0.00652、PPL比约0.98408，开发轨迹通过。
2. **1B cap1＋headMSE**：512＋64全24层decode15.028152→15.934702（＋6.03%），TTFT0.802761→0.793124；1024＋32约＋5.81%。不同实验收益不能相加。
3. **1B cap1＋原输出头**：512＋32全24层14.965381→15.490959（＋3.51%），原Q8 PPL比1.048677，接近1.05阈值。headMSE在该格PPL比1.053872失败，不能混同。
4. **down投影缩放拆分**：保留原INT8整数及有效权重，FP32等价、编译字节与实板局部输出通过，但两层API变化不足0.2%，不值得直接扩全模型。1B t8单位token局部成本比t16高约39%，也未扩展。500M t64此前更慢，不采用。

普通1B cap1两层profiling中FC/融合约91.51% busy cycles、attention matmul约4%；仅说明该层组。完整模型宿主开销即使全消除，剩余收益也只是几个百分点。主攻大矩阵执行、布局、访存与合理量化，不再把微小宿主改动当作突破。

## 质量、KV与产品状态

- 标准模型参考与自定义oracle已有核对；开发执行误差与相对原Q8质量是两道门槛，必须同时过，不可看到更快后放宽。
- 500M新t8：36数值＋36自然回答均完成，数值仅24/36通过；聚合KL/PPL通过不能掩盖单例失败。英语简单角色/猫名尚可，中文倾听三轮复读、角色记忆有额外退化；部分更正失败CPU也存在，需区分基模能力与量化退化。
- 1B最新组合独立v2为33/36、新head单独32/36，均未过整套质量。
- 某些结果JSON顶层 `passed` 表示运行成功，**不是产品质量通过**。看逐例门槛、整套判断和生成原文。
- 500M t8复用经模型/套件/逐文件SHA验证的标准Q8 FP32参考，是重复冻结套件回归，不是新独立留出。若用这些失败来指导调参，下一验收必须换新独立数据。
- 常驻模型与跨请求KV已全深度核对，reset、最后token计数、窗口裁剪有验证。不要退回每次重算全部history来比较。
- 旧候选有30分钟稳定、实际浏览器与真实ASR并发证据；新图不能自动继承。新t8仅有自己的HTTP/常驻验证、质量回归和退出driver=0，未做自己的30分钟验收。
- 实验NPU网页是**工作站HTTP/分词网关，经SSH调用板端常驻NPU/KV**，不是板端独立NPU网站。生产8080关机前仍是CPU服务（Qwen2.5-1.5B Q4 / 4096）；不要部署质量未通过的新图覆盖它。

## 源码与证据定位

从 [本轮技术攻关结果](../npu/kv-cache/本轮技术攻关结果.md)、[最新完整矩阵](../npu/kv-cache/最新CPU对齐与组合候选矩阵.md) 开始。总览末尾“继续完成待跑矩阵及输出头消融”为残留旧行动文案；本轮这两项已经结束，以本文及矩阵24/24为准。

关键入口在 `npu/kv-cache/`：`run_full_model.py`、`benchmark_variants.py`、`resident_http.py`、`run_resident_http_validation.py`、`compile_down_postscale.py`、`reuse_holdout_reference.py`。执行前看参数和锁实现，不凭历史session重建命令。

本机 `npu/results/` 下的定位名：

| 结果目录 | 内容 |
| --- | --- |
| `cpu-finalist-realignment-20261014` | 最快CPU轨迹重新对齐；状态含已保留拒绝项，不能只看根status推断缺格 |
| `smol500-prefill8-trial-20261014` | 新t8完整性能 |
| `holdout-npu-prefill8-smol500-20261014` | 新t8质量及实际自然回答 |
| `holdout-reference-prefill8-smol500-20261014` | SHA核对后的标准参考复用 |
| `cap1-originalhead-short-trial-20261014` | 1B512＋32原head消融 |
| `cap1-headmse-short-trial-20261014` | 1B512＋64组合 |
| `down-postscale-20261014` | 缩放拆分，包括失败首轮与修复后的attempt2 |

500M新t8源目录为 `full-smol500-stable-v3-c128`，decode变体 `w8-fused2-c768-dfpgateup`，prefill变体 `w8-fused2-c768-t8-dfpgateup`；部署tag `full-smol500-c768-dfpgateup-prefill8`，case `prefill8-dfpgateup-smol500-h512-n64`。不要仅凭源目录c128后缀误判实际容量，核对变体和manifest。复用图通过硬链接与SHA核验，不复制多套占满TF。

报告生成顺序（都在 `npu/kv-cache/`，执行前看各CLI）：`report_candidate_matrix.py` 默认及 `--finalists` → `report_candidate_speed.py` → `report_prefill_screen.py` → `report_down_postscale.py` → `report_prefill8_quality.py` → `plot_finalist_matrix.py` → `report_attack_summary.py`。现有报告已生成，不需为了接管重复跑板端实验。

## 重新上电后的工作顺序

1. 用户确认装好风扇并通电后，核实主机身份/IP、双锁、无旧任务、温度、TF/RAM、NPU驱动、内核异常和CPU网页健康；记录散热变化。先排查实际证据，不把掉线直接归因高温或掉卡。
2. 选择并冻结每模型候选、量化配置与独立质量协议，沿用现有严格门槛。用户优先提速，但不能拿不正常说话的模型宣布成功。
3. 优先形成有依据的大矩阵/decoder提速假设；局部真实输入筛选有明显收益且质量通过后，扩完整32/24层、完整head，再做五轮配对。保留失败attempt与来源SHA。
4. 500M t8是否扩512/1024×32/64/128取决于修复质量后的候选价值；不要重复已完成24/24旧版本矩阵。新风扇下用于归因的CPU/NPU应采用相近温度、同轮配置与轨迹。
5. 质量通过后再做新候选自然多轮、真正部署形态的TTFT、30分钟稳定及真实ASR共存；单独报告纯NPU主计算与CPU/NPU混合方案。
6. 每个阶段更新报告、实际输出、同轮资源和未达标项，提交并核对远端SHA。若硬件/编译器受限，给出实测瓶颈，不换慢CPU基线。

## Git备份与恢复边界

仓库：<https://github.com/Behavior-Cloning/OragnePi-3W_ForVLM>，分支 `main`。源码、报告、紧凑证据已备份；权重、系统镜像、SDK、完整logits、NBG/ONNX、虚拟环境、凭据不在Git。新机器clone后不能假设立即复现实板实验，按 [备份与恢复](GitHub备份与恢复.md) 准备本地资产。

本机已有推送认证可用命令级SSH重写，勿修改全局Git设置，不把密钥内容写进文档：

```bash
git -c core.sshCommand='ssh -o ForwardX11=no -o BatchMode=yes -o IdentitiesOnly=yes -i /home/ubuntu/.ssh/id_ed25519_github' \
  -c url.git@github.com:.insteadOf=https://github.com/ push origin main
```

提交前检查 `git diff --check`、文档链接、敏感内容与备份范围；推送后通过同样认证的 `git ls-remote origin refs/heads/main` 核对本地HEAD。保留不属于本任务的用户改动。不要将Git备份描述为完整TF系统备份。
