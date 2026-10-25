# Intern 下一步 NPU 优化（2026-10-25）

## 2026-10-25 组合候选最终复测

combined-matrix-r2 已完成且独立审计通过：72性能、36自然、36cost、15诊断、3次正常closed driver0。static64/128（含模板5），dynamic正文32/64+模板8，new40/72，固定输出32。真8候选TTFT四格P50 0.493827/0.869842/0.494815/0.890454秒，最大0.915570，decode约15.29–15.50t/s。三组全词表同字节，旧C1280形状差异仍约3% L2。原r1 close根因未知，不宣称根治；未替换默认/未做新一小时测试。详见 ../intern-nextbench-20261025/analysis/combined-matrix-r2/REPORT.md 和独立审计。下方中间阶段的“正在跑”均已结束。


本目录是独立实验分支，未替换当前服务、原始量化图或冻结的 mask 候选。权威源在 Windows F/WSL；板端运行由主任务唯一 owner 串行安排。本目录的主机准备完成不代表性能已改善。

## 当前状态（2026-10-25，覆盖下方历史阶段的“待测试”描述）

完整六配置 `candidate-matrix-r3` 已完成，六次关闭的 NPU driver 均归零；完整数字、原始结果和限制见 [r3 报告](../intern-nextbench-20261025/analysis/candidate-matrix-r3/REPORT_NEXT.md)。这是端侧小样本开发测试，未替换默认服务，也不是长期稳定性承诺。

本轮固定输入为 **system 总计64/128 token**（正文59/123 + system模板5）；**动态正文32/64 token，另加动态模板8**，所以新计算40/72 token、总prompt104/136/168/200 token。只复用system前缀KV，性能固定反馈生成32 token，单独统计自然生成及全词表诊断。下表 TTFT 从已备好system缓存后的请求开始，不含预先计算system缓存的耗时，也不是用户真实说完到首字的完整语音延迟。

| C256 + 单遍greedy | 动态正文32（实际新40）TTFT P50/P95 | 动态正文64（实际新72）TTFT P50/P95 | decode P50（两种动态长度） |
|---|---:|---:|---:|
| system64（正文59+模板5） | 0.5786 / 0.5811秒 | 0.9544 / 1.0068秒 | 15.4141 / 15.2509 tok/s |
| system128（正文123+模板5） | 0.5815 / 0.5826秒 | 0.9782 / 1.0105秒 | 15.2028 / 15.1856 tok/s |

每格仅6次，重复fixture不等于6个独立真实用户场景；动态64的P95仍略高于1秒。C256-greedy固定反馈24/24、自然输出12/12与原基线token一致，但4步全词表logits相对原C1280仍有约2.93%–3.26%相对L2差异，因此不能称所有浮点输出逐位等价；已知自然回答及RAG问题也没有在此轮被修复。

**tail8-c1280已做完整板测，且出现首token差异，不能继续写“未测试”或称无损通过。** 相同四格TTFT P50为0.8108/1.4513/0.8283/1.4656秒，比原C1280各格约少0.15秒；固定反馈仅22/24逐token相同，`s64-d64-2`两次的首token由117057变为114209。原12条自然输出一致，但这些原自然样本并未触发真8行尾图；另加4条触发尾图的cost自然输出一致，也不能覆盖上述失败输入。单块有效行逐位一致不能代替全模型验证。

**combined-r1退出失败仍未找到根因。** C256 + greedy + 真8行组合在base/tail8各完成24条性能、12条自然、12条cost自然和5条诊断后，tail8进程关闭报 `double free or corruption (out)`，未返回正常closed，repeat阶段未执行；虽然5份诊断与该次combined-base逐字节一致，整个矩阵仍属于失败。原行为ASan最小/完整复测及原O2详细释放trace完整复测均正常关闭、driver归零，没有复现该故障，不能因此宣布问题消失。

`runtime-combined-v2`仅为保守的释放顺序候选：先全部group，再全部network，最后各自拥有的buffer；推理模块未改。主任务正在独立安排该候选的关闭smoke和完整配对矩阵，结果以主任务回收报告为准。**尚未证明根因修复，不替换默认服务。** 冻结源码清单与测试证据见 [v2说明](runtime-combined-v2/README.md) 和 [清单](runtime-combined-v2/FINAL_DELIVERY.json)。

## 历史阶段记录

以下保留早期准备与单块阶段的原始正文，记录当时的实现入口和限制。其中“全栈待板测”“常驻内存待测”“完整语言生成待回传”是历史状态；当前完成情况、数值差异和退出故障以上述状态及r3报告为准。旧释放顺序也只描述原候选，不代表已确认安全或当前v2实现。

## 已交付

1. `resident-profile` / `runtime_profile.h`：按 scalar prompt、decode decoder、first head、decode head 分解运行时；`OPI_PROFILE_STAGES=1` 启用，`OPI_PROFILE_NPU=1` 另加 SDK 逐网络查询，仅作诊断。构建与协议测试见 `build-profile.json`、`profile-protocol-test.json`。未启用 flags 时保留原数值与调度；剖析模式包含时钟读取开销，不能冒充原版纯性能。
2. 真 batch8 第0/1层融合块已通过板测，完整12块也已编译：容量1280，各块保留相同权重/偏置及全部量化参数，单块与全栈清单分别见 `prefill8-build-status.json`、`prefill8-stack-manifest.json`。板测契约见 [PREFILL8_PROBE.md](PREFILL8_PROBE.md)。4组中文fixture固定 system64/128（含5模板）、动态正文32/64（另8模板），真实非零FP32历史KV。单块尚不能证明全栈生成等价。
3. `runtime-tail8/resident-tail8`：独立的双预填充图运行时，已编译及主机状态测试；尚未凭此目录宣称整模型板测通过。真实板端结果应在主任务报告中引用原始输出。

## 双预填充图候选

普通图目录仍由命令第1参数指定，保留原 `layer*.nb`、head、`prefill-layer*.nb`（16行）。设置 `OPI_PREFILL8_DIR=<独立目录>`，其中须有12个 `prefill-layer{0,2,...,22}.nb` 真8行图，容量必须与主目录一致。只在batch参数为16、未设置实验scalar-block时启用，匹配失败立即退出。

- 不设置该变量：维持原16行/尾部padding/scalar逻辑，只增添两项 `done` 计数字段；不加载8行图。
- 设置后：完整16行批次不变；只有最终剩余**恰好8**个真实token时用真8图；其它尾长维持原逻辑。不更改prompt、token反馈顺序、量化或停止词。
- 每张8行图通过完整shape/dtype/quant契约核验后 alias 原decoder的K/V。两个预填充bank拥有各自mask和hidden buffers，reset/truncate全部同步恢复，防止前次suffix可见。
- IO alias **不是权重共享**。额外12张图增加权重和驱动内存，需按真实ready/closed/遥测核算；12张NBG总291,566,304字节；这是文件大小，实际新增driver内存仍须测量，不能与文件字节等同。
- `done.true_tail8_calls` 与 `done.true_tail8_seconds` 可验证实际使用次数；时间同时计入原 `prefill_seconds`，不得重复相加。
- 正常关闭先释放两个预填充bank，再释放其共享KV所属decoder；依赖现生产buffer ownership机制。板端必须核对driver最终归零。

`tail8-two-bank-host-test.json`：直接提取候选实际 reset/truncate 函数，使用真实SDK结构体及INT16 mask数据；ASan/UBSan下 plain/template/validate 各1,920张量检查通过。覆盖容量1280/256、有/无tail图、keep反复0→64→128→64→200→128→0→192→96→0，并主动污染两bank旧mask验证回退清理。此测试不包含NPU计算，也不验证整模型中文质量。

## 门槛与未完成

- 已完成同输入第0/1层单块 t16→t8→t16-repeat，5个输出的有效前8行逐位相同、driver全部释放；详见下方阶段更新。
- 全栈逐块保留原conversion里的量化例外，包括敏感2/3层的原来源以及19/20层product headroom；完整运行必须覆盖这些层，不能只凭第一块通过作结论。
- 全12块编译完成，常驻内存待板测；完整评估需同冻结system64/128×正文32/64、仅system前缀KV、固定32输出与自然输出分离，全词表logits比较、语音常驻资源监控。
- context256由另一个独立目录负责，不在此重复编译。不得把短容量图的速度与原容量不同输入混在一起归因。
- 尚未提交任何原生INT4或失败W4质量路线。保留原Intern W8基线，收益必须来自同语义、可接受精度的实测。

## 板端单块回传（阶段更新）

主任务 `research/intern-nextbench-20261025/results/graph-probes/prefill8/summary.json`：4例×t16/t8/t16-repeat，各50计时+3预热；有效8行的hidden和两层K/V全部逐位一致，重复基线无漂移，12次进程driver均归零。NPU API P50分别16.978→8.972、16.886→9.267、16.972→9.261、17.000→9.115ms。相对导出常量FP32参考最差L2为1.268%，两种batch共有，不能称原BF16无误差。

全12个t8图已按原各层variant/敏感headroom编译，见 `prefill8-stack-manifest.json`；原FP32 `.data`、实际导出量化常量payload及parsed量化均与对应t16完全一致。编译后将两个重复常量文件替换为已SHA验证的baseline符号链接，保留审计与复现同时减少scratch。完整resident语言生成结果仍待主任务回传。

单遍argmax候选与测试契约另见 [GREEDY_CANDIDATE.md](GREEDY_CANDIDATE.md)，此路径与t8独立，不在同一实验中混入两个变化。
