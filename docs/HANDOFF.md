## 2026-10-25 prefill / decode 全图候选测试进行中

用户要求先同步GitHub，再继续降低NPU首token及decode开销，最新强调prefill瓶颈。已验证GitHub同步到899c3b02db0939fe69ba3f9bdf478909e85b0ac2；本轮源文件checkpoint本地854fb86，后续结果待提交。权威任务research/intern-nextbench-20261025，候选源research/intern-nextopt-20261025与intern-context256-20261025。

controller_next.py candidate-matrix-r3于WSL PID7878启动，双锁串行6候选，仅读candidate-matrix-r3-controller.json及log；禁止重复启动、直接轮询占锁板卡。已完成部署SHA校验（现役29/ctx256 29/tail8 12文件），ARM greedy 221212例0不匹配。全模型证据未回收前不能宣称提速或无损。 首次r2在首组24次性能之后记录自然结果时缺system_body_tokens，退出1且driver0；失败证据保留。已按相同system prefix映射补日志字段，4个正反单测通过，r3重测不改变输入ID和量化。

输入固定64/128=正文59/123+模板5，动态正文32/64+模板8，总104/136/168/200，reused64/128，new40/72。每格3场景×2次、性能固定反馈32步，另12自然输入和4个cost输入自然生成，再4格×4步完整logits；语音合成回放常驻、RAG质量阶段查询，纯Intern NPU文本，无视觉/电机。ctx256最大当前prompt+32=232，仅短文本用；不可用于256图像追加或最大格64输出。比较mask-base/context256/tail8-c1280/greedy-c1280/context256-greedy/mask-repeat；不覆盖现役默认。

profile-smoke证实prefill占TTFT 97–98%；单块ctx256 prefill API约17.0→6.78ms、decode3.62→3.37ms，尾批8约17.0→9.1ms，单块5输出同字节。单块无并行音频且非全模型，报告需分开。所有runtime/hashes/边界详见任务README。

## 2026-10-25 同输入首字优化已完成：32正文小样本<1s，64正文未达标

运行ttft-aligned-matrix-v2完成returncode=0：180速度请求、60自然输出、24份完整logits诊断，6次driver关闭均0。控制器已结束，不要重复启动。报告：research/async-agent-20261025/results/ttft-aligned-matrix-v2/REPORT.md；统计analysis.json，独立数值numerical-audit.json。

输入契约：固定前缀64/128=system正文59/123+模板5；动态正文32/64另加模板8。总输入104/136/168/200，reuse64/128，实际new40/72，速度输出1。每请求只缓存system，同格完整IDs一致，未用exact或上一轮动态缓存。精确长度含标点填充，静态128重复相同核心；自然质量另测。

按静态64/128、动态32/64四格：plain TTFT P50=1.152739/1.759670/1.180238/1.798044s；tail8+group+mask=0.959470/1.591648/0.974067/1.607835s。32正文各9/9<1s，P95=0.962396/0.984416；64正文各0/9<1s。每格仅3场景×3次，P95为最大值，不是长期保证。64正文prefill本身1.56–1.58s，后续需图内/批图优化，宿主setup已约2–5ms。本候选尚未替换完整Agent默认运行时，未重跑优化版一小时。

group→mask四组4步全词表logits逐字节一致；plain→tail8分区已有2.69–3.06%相对L2变化，16/16argmax和12/12自然输出一致只代表开发小集合。任务完成静态64为4/6，128为5/6，弯腰退回静止，历史RAG兼容性错误未修。Top3存在摇头/移动等冲突，不能全执行。全程mock无电机。

语音常驻合成录音，RAG仅质量阶段调用，Intern NPU仅文本无视觉。约15.93分钟；整机CPU均7.87%，温度峰49.91°C，harness+resident PSS峰589.52MiB（不含其他常驻服务，driver另计），MemAvailable最低3279.84MiB，swap0。音频60586帧/150端点，最大帧滞后0.601s，不能说全链路无滞后或端到端<1s。

首次无v2因tokenizer SHA不符在加载前失败（0计时），已用板端原文件重新冻结完成v2。保留两份输入及失败证据，不能混用。候选源码提交ce1dbe3；结果报告另行提交。前阶段固定86的mask报告results/ttft-mask-matrix/REPORT.md已完成，勿拿它替代本对齐矩阵。

## 用户要求：所有性能汇报必须标出token口径

system正文、固定前缀含模板、完整动态正文min/P50/max、额外动态模板、总prompt、实际reused/new-prefill与输出token必须一起报。旧配对方案system正文81+system模板5=固定86；pair300-v2完整动态正文57/115/126，pair300-32为5/30/32；额外模板均8。勿把旧版摘要64上限误报为完整动态64，也勿把固定前缀86叫作纯正文86。

## 2026-10-25 首字运行时矩阵完成

135计时+30自然输出，5组关闭driver0；尾部批处理将正文32/48/64首字P50从1.450/1.765/2.103s降至1.064/1.375/1.672s，仍未稳定<1s。候选6/6自然输出与旧版逐token一致，RAG弯腰→抬头缺陷未修。report: research/async-agent-20261025/results/ttft-32-64-matrix/REPORT.md。正文32实际new46，模板8+system尾6。继续检查truncate_session的mask构造setup开销，尚未部署该新候选；勿重复跑已完成矩阵。

## 2026-10-25 32-token摘要配对已完成

两组各300秒/88推理：仅32正文+8模板，system86；模型TTFT P50 1.618→1.023s，P95 2.682→1.446s，尚未稳定<1s。32组26/88基本门控拒绝、输出长列表/超能力，质量未过关。见research/async-agent-20261025/PAIR32_REPORT.md。ttft-32-64-matrix已完成；runtime比较old/tail8/group/group-exact/old-repeat，结果见上文独立报告。

## 2026-10-25 1小时结果与32-token后续

stress60-v2板端225000帧/3600秒、1022推理完成，driver释放0；控制器SSH收尾3840s超时，已独立回收日志，不能称控制器exit0。事件→首token P50 2.348s、mock动作开始2.584s、decode13.50t/s，最高49.29°C，CPU平均6.237%，PSS峰585.66MiB且后期仍约0.20MiB/min增长。RAG动作兼容性仍未合格。详见research/async-agent-20261025/results/stress60-v2/REPORT.md。随后pair300-v2与pair300-32已由run_pair32.py顺序完成，独立目录/锁；不要重复启动。

## 2026-10-25 异步常驻Agent harness

见research/async-agent-20261025/REPORT.md。v1完成900秒/257推理，CPU平均6.47%、温度最高49.414°C、decode13.67t/s、事件→首token2.43s/动作开始3.56s(P50)。性能稳定但决策失败：仅30/257明确运动，拒绝/物品操作仍被RAG映射mock。v2提示+基本门控75秒回归完成，仍有弯腰→抬头兼容性错误，未通过动作验收；后续v2长期测试已完成，见本文件更晚日期记录。没有物理电机动作。两个版本证据分开保留，板卡无本次harness任务运行，driver释放0。后续优先修RAG兼容门控、规划评测、2h内存增长验证。

## 2026-10-22：无屏幕、无手指的128动作v2

用户确认头、双臂、腰及能平移/转向的低自由度腿部底盘；无眼、无屏、无手指。当前词典切换至 `research/home-companion-atoms128-v2-20261022/`，128项R编号、0技能；v1仅作历史。具体轴/范围待验证，全部未映射，没有动作下发或新版模型复测。详见新版清单、审计和HANDOFF。

## 2026-10-22：家庭陪伴128原子动作集合

按最新要求单独建立 `research/home-companion-atoms128-20261022/`，128原子、0组合技能；90机械设计+38屏幕效果，全部硬件未映射。本轮只建词典，未进行128候选模型复测，旧35+20测试结果保留。完整清单、检查和后续待办见该目录 ATOMIC_ACTIONS_128.md 与 HANDOFF.md。

## 2026-10-22 家庭／桌面动作RAG新基线

用户要求原子与高层动作都限定家庭陪伴，管理接口不得进入动作字典。已完成35原子、20家庭组合技能、60条输入、五编码器共1200次板上CPU检索。新报告及完整交接：`research/home-companion-rag-20261022/HOME_COMPANION_RAG_REPORT.md`、同目录`HANDOFF.md`。旧256项混合库只作历史证据，不能继续作为家庭动作字典。无硬冲突与全可执行、目标线索与任务完成分开统计；Top-K仍需编排。

## Jev 后续队列已回收（2026-10-21T03:43:42.499314）

优化及共驻自动回收见 research/jev-validation/JEV_DEPLOYMENT_20261021.md 最后一节与 results/continuation-status.json。仍需审计结果、补完整质量/长稳/视觉并同步 Mac 报告；远程下载独立运行，尚未由此脚本确认完成。

## 2026-10-21 Jev逐模型提速审计追加（实验继续）

用户要求每个Jev优化延迟/决策速度，已加入 JEV_OPTIMIZATION_AUDIT.md，由审计智能体逐项核验。Windows权威报告 research/jev-validation/JEV_DEPLOYMENT_20261021.md 已重写，旧版本存sources留痕。

Jev-Style17→2槽：10候选7.097→6.969s；100候选57.562→57.141s；PSS2.331→1.995GB（−14.4%）。全部28答案与评分逐值相同，质量仍4/8、安全0/2。只认内存改善；小幅时延差待交错复测，不能声称已可部署。

当前优化5972，自动续跑6704（替代6300/6535，增加20计时答案及18英文控制parity），先完核/线程/runtime矩阵，再独立进程与持续决策共驻。不得在控制器持锁期间另行板端采样。新版板端Jev身份清单在最终配对前计算；当前优化队列旧源码冻结sources/optimization-v1，不能拿新版hash追认旧run。

远程是CPU存储服务器，仅下载9B/27B-VL；01:34仍未完成。已连接过，SSH会间歇重置，Tailscale未关闭。温控PWM/PD5冲突预先存在；dmesg环形日志已轮换，不能以无后快照错误当作全区间健康证明。

最新seq2+ub256：10候选7.133s稍慢，100候选55.927s，PSS1.932GB；答案不变、10候选评分有数值差，仍安全0/2。远程下载新PID123340/123341，每仓库8并发断点续传。续跑会补从未启动配置，不把已有超时部分结果重写为完成。

未完成：优化全矩阵/优化后共驻、长稳与更完整质量、OneJev多模态、服务器完整SHA、完整mock执行链。保持原始证据与全部TODO，不标整项完成。

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
