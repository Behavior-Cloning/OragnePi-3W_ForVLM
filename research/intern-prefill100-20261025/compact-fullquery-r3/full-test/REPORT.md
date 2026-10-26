# 紧凑历史完整模型 A/B/A：未获得明确净收益

2026-10-26。候选 compact-fullquery-r3；源码冻结 d139d0d。测试完成rc0，但不推广、不覆盖已登记最快候选outlier8-c448-full-r1、原量化验证版本或baseline。当前用户目标为prefill约500ms并提高decode；本次4格均未达到500ms。

## 对齐条件与结果

Intern完整24层NPU纯文本，decode图、权重与量化、输入token IDs、system构建、CPU亲和、反馈序列相同。父版本自身是改变精度的outlier8候选，并非原量化baseline。A/B/A使用同一新runtime，开关只控制compact图；关闭时s128-d256诊断与旧runtime归档SHA完全一致。

system仅复用64/128 token；正文59/123+模板5。动态正文128/256、额外user/assistant模板8，实际new136/264。总输入200/328/264/392。每格3个场景；固定反馈输出32 token；decode=31/(total-TTFT)。TTFT包含setup、prefill、head，不含system冷构建、上游语音、音频终点或动作执行。

单位ms，P50；原方案指同场次outlier8父版本。

|system正文+模板=总量|动态正文+模板|实际new|原prefill|原TTFT|候选prefill|候选TTFT|回测prefill|回测TTFT|候选decode tok/s|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|59+5=64|128+8|136|593.9|614.1|608.9|631.6|580.9|600.8|15.45|
|59+5=64|256+8|264|1097.7|1118.3|1090.3|1113.9|1093.6|1114.3|15.30|
|123+5=128|128+8|136|579.5|599.6|571.7|594.5|581.1|601.1|15.35|
|123+5=128|256+8|264|1097.3|1118.1|1105.4|1129.3|1097.1|1117.8|15.57|

每格仅3次，P95按nearest-rank取该格最大值，完整统计见ANALYSIS.json。个别约几毫秒改善处于A/A回测波动范围；无一致净收益，decode仍约15，不能声称提速。system32/动态32/64/192未在本轮测，保留既有版本各自的覆盖范围，禁止拼成同版本矩阵。

## 改动和实际瓶颈

全局KV存储仍448；额外12张past128/query128图，只在新请求起始处且全部有效历史≤128时运行一次，所有有效KV按原量化原样拷贝，RoPE仍用绝对位置。其余新增token及尾8回到C448图。图payload与父图逐字节一致，量化表相同；query分组和attention形状不同仍会造成数值差异。

完整128行紧凑块耗时P50约498～538ms，尾8另约71～73ms，拷贝、发布和图执行均计入，不能拿此前两层API时间代替完整prefill。当前组合未突破瓶颈。

## 数值与文本

36次正式计时、36次自然生成、12份4步full-logits均回收。每份2426784B、finite、真实SHA核对通过。父版本A/A四份logits逐字节相同；候选相对父版本L2差异2.68%～3.14%。这不是准确率下降百分比。

候选11/12自然生成序列相同；差异一例s64-d128-2由“挥手致意。”变为“挥手打招呼。”，语义接近，不能据此判为动作错误。固定反馈序列11/12相同。自然生成使用同一组带标点补齐的长输入性能fixture，未使用独立中文质量集，不声称质量合格。

## 负载与释放

合成录音流式降噪/KWS/VAD/ASR常驻，RAG仅驻留未查询。无视觉、真实麦克风、电机。资源采样223.45s，不是长时间稳定性测试。

- 候选ready NPU device allocation约1316.5MiB，父版本约967.4MiB，增加约349MiB；这与进程PSS分开记录，不能相加当总内存。
- 候选阶段MemAvailable最低115.6MiB，中位149.0MiB；父版本阶段最低551.1MiB。候选阶段swap in8052/out9943页。PSS较低不能解释为省内存，存在换页。
- 候选阶段8核归一CPU P50 7.28%、P95 19.76%；该阶段峰温47.99°C，全程49.91°C，load1全程最高3.61。顺序运行温度未严格匹配，不归因温升于算法。
- 音频最终累计max lag约0.695s，候选采样阶段4726帧中231帧处理超16ms；累计最大lag不等于候选新增延迟，也不能据此宣称始终实时。
- 3次closed driver=0；最终NPU allocation/free813/813。无崩溃、OOM或driver残留不代表资源合格。

结论：保留为“完整测试但不推广”候选。下一方向优先更小query块与减少额外常驻图内存，同时继续同输入、同精度父版本配对；未实测前不写预期加速数字。

## 可追溯证据

- build-runtime.json：ARM binary与源码SHA；compile-status.json：12张图SHA/父payload/量化/端口顺序。
- host-test.json：6次ASan/UBSan通过，每次960张mask+258组KV拷贝；保留初始mask失败与修正记录。
- full-test/config.json/inputs.json/deployment-sha.json：冻结输入、部署SHA；results/events.jsonl/resources.jsonl：原始事件与负载。
- ANALYSIS.json重算f32 SHA，校验实际new/reuse、诊断标志、输出数量和A/A一致性。大binary/f32/原始资源保留Windows磁盘，报告和统计纳入Git。

板端追加图已在全部19份回收文件SHA一致后释放305411904B（约291.3MiB），保留父版本和默认模型。图/源码/诊断Windows证据仍在，SCRATCH_RELEASE.json记录清理；复现需新run ID重新部署。
