# 六候选完整矩阵：独立源代码审计

2026-10-25。只读取 Windows/WSL 源和已有回收结果；没有访问板卡、抢锁或启动实验。本文是源审计，不代表候选已跑完。

## 已确认的公平性

board_benchmark.py冻住aligned-inputs.json、tokenizer与每条token SHA；每配置同四格static64/128 × dynamic正文32/64，各3个相同fixture IDs。每条请求先reset、仅计算system prefix，再进行计时；严格reuse64/128、新算40/72（正文另8模板）。不同capacity不改变输入或反馈。

性能固定输出32步，反馈均为同32个token且diagnostic=False；feedback让EOS不提前终止。计时结果保留模型真实预测token，不把feedback本身当预测输出。decode公式是 `(generated_tokens - 1)/(total_seconds - ttft_seconds)`，本轮即31个decode间隔；分段2–8、9–16、17–24、25–32依次7、8、8、8个间隔，总计31。不是32/总时长。

12条自然质量没有fixed feedback，可以自然EOS；原有32输出上限保持。其精确token一致性只相对同一首配置，并非FP32准确率或高层动作正确率。禁止将baseline错误回答的复现当成合格行为。

全词表诊断每格第一例，共4例；每例4步，使用与性能相同feedback序列的前4个。diagnostic=True独立阶段，检查文件长度、有限值、哈希以及相对基线全词表误差；诊断I/O不进入性能样本。NPU SDK query也不允许进入performance。

## 容量约束

最大完整prompt200token；32个输出只需追加31次反馈，因此最大KV位置数231。capacity240/256都容得下该矩阵；严格公式为 `prompt_tokens + max_new - 1 <= capacity`。在prompt200下240最多41输出，256最多57输出，64输出不合法。未来图像额外256或其它长输入不能沿用此结论。

脚本在加载前检查全部fixture的最大计数，并对ready.layers/capacity断言；相同runtime下ctx256不代表换更小token工作负载。图与tail8overlay的容量必须匹配；expected_graph_sha256覆盖范围以配置显式清单为准。

## 资源统计限制

资源包括harness和对应resident进程PSS/RSS、整机CPU/内存、温度、频率、swap及Frontend累计帧/lag。analyzer单独提供各phase与performance阶段资源，NPU driver另外记录ready/推理/closed。

performance阶段资源样本仍包含每次system-only准备，而TTFT计时排除了准备；不能把该phase CPU等同纯generate计算。各候选共用同一Python进程，先前诊断数组、NumPy分配与allocator高水位可能保留，PSS峰值差不能直接解释为候选独立模型footprint。driver关闭归零且下一worker重新载入，是较直接的NPU驻留比较。温度/资源0.5s采样不能捕捉全部瞬态峰值。

同一组录音循环持续，候选进入阶段时语音片段相位不严格相同；驻留模型、文件集合和实时速率相同。mask-repeat能检查顺序漂移，但每格3场景×1轮烟测P95就是最大值；不能据微小百分比差异宣称确定加速，最终候选仍需要repeat和同任务质量检查。

## 分析器新增审计

- 配置variant顺序与实际事件序列一致；候选是否恰含六个预期名称单独记录。
- 全部性能请求32反馈值完全相同，prompt SHA与fixture一致、缓存计数正确、容量无溢出。
- ready.layers/capacity与settings匹配，expected_graph_sha256清单记录覆盖并匹配。
- diagnostic必须fixed4且flag启用；自然输出、固定反馈输出、全词表parity各自独立。
- true_tail8_calls/seconds字段聚合；其秒数已包含prefill_seconds，不能重复相加。
- closed.driver_bytes必须为0，runtime/monitor/audio/release错误都会标记审计失败。

## 最终配置已核实

config.candidates.json SHA256 `2d1c3cb9cb4e7f1624df572c5b0260ba252f51c815aeeee6f6fae524a46c997d`。

顺序恰为mask-base → context256 → tail8-c1280 → greedy-c1280 → context256-greedy → mask-repeat。所有配置同24层、词表151674、prefill16、CPU4/5、tail-min8/group/mask flags，只有候选意图对应的图、容量或greedy/tail8 flag变化。context256两组capacity256，其余1280；全部未开启stage profile/NPU query。

普通配置各29个SHA覆盖12decode+12prefill16+4head+embedding；tail8配置另加12overlay，共41。overlay为1280容量路线，未与ctx256混接；最终mask-repeat与首mask-base二进制/图/env一致，便于观察顺序漂移。

当前父控制器candidate-matrix-r2状态running、repeats=2，每配置24性能请求（每格6样本），并非完成结果。每格P95仍等于最大观测，且只有3独立成本场景各重复2次，不是6条独立用户场景。

## 真尾8的额外自然覆盖

父任务已在原12quality外增加quality_cost：四格各首个performance fixture自然解码，最多32token；保留标点成本padding，因此属于4条机制覆盖，不是4条新增独立质量测试。tail8配置须每条true_tail8_calls=1。分析器分别保存这4条的自然输出、token一致性、actualnew/reuse和tail8命中，并严格检查4条/variant；不得混入12quality计数。

分析器额外核对原quality与quality_cost自然请求的diagnostic=False、fixed_feedback=False、输出1–32、实际token计数和缓存数。全性能的case/repeat二维覆盖也逐项检查，防重复记录掩盖遗漏。
