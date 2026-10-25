# 独立结果审计：suffixfixed-c1280-aba-r1

**证据完整性通过，候选不晋升，100ms完整目标未达到。** 同C1280下，首system片段保留原L2、后续使用I16，固定后半段10网组。候选每批3次提交，原图1次。首片段修复显著恢复数值一致性，但新冻结场景仍出现明确动作退化。

## 原始证据与口径

- 控制器rc0、meta complete；72性能、144自然、24诊断，三次close driver0。候选FIRST与SUFFIX标记各81，原图两版均0。
- 24份实际F32、共58,242,816字节：大小、SHA256、有限值、argmax与输出token全部核对。29图实际SHA、source/config/fixtures及额外原L2记录符合冻结合同。
- 原图前后24性能输出、48自然输出和8份四步完整logits均相同。候选23/24性能输出完全相同；8诊断32/32argmax同原，logits relativeL2仍2.750%–3.029%，并非数值逐位相同。
- 静态64/128=正文59/123+模板5；动态正文64/128/192/256+模板8=新prefill72/136/200/264。每请求reset后只准备systemKV；热TTFT不含该准备，但资源包含。
- 性能固定32反馈token，decode按31/(total−TTFT)。所有性能scalar_prompt_tokens=0；尾8实际pad为M16，无true8。自然输出独立，实际2–20token，三版中位均5。

## 性能

每格3个不同案例、每版一次；P95为nearest-rank，在n=3时就是最大值。

|静态|动态正文|原prefill→候选(s)|原TTFT→候选(s)|候选TTFT P95(s)|原decode→候选(tok/s)|
|---:|---:|---:|---:|---:|---:|
|64|64|1.568→1.018|1.590→1.039|1.041|14.62→14.62|
|64|128|2.823→1.881|2.845→1.903|1.952|14.60→14.66|
|64|192|4.097→2.735|4.120→2.757|2.808|14.65→14.63|
|64|256|5.367→3.600|5.389→3.623|3.624|14.60→14.66|
|128|64|1.585→1.058|1.610→1.079|1.082|14.60→14.61|
|128|128|2.795→1.918|2.819→1.940|1.941|14.64→14.63|
|128|192|4.125→2.781|4.148→2.804|2.808|14.65→14.97|
|128|256|5.395→3.657|5.420→3.684|3.685|14.65→14.65|

24条逐案例配对TTFT中位比例0.67350，约缩短32.65%；decode比例1.00298，基本不变。原图前后TTFT比例中位0.99834（范围0.96978–1.00766），decode1.00273。新方案速度收益不能掩盖质量回归。

## 自然语义与RAG

人工阅读完整输入、gold与实际输出；非盲审。旧输出只有在输入tokenSHA及回复字节完全相同时复用已冻结判定。缺方向/部件、部分完成、保守不响应、完全不能执行分开，不把字面相等当正确。

|样本集/版本|完整通过|兼容但不完整|安全不响应|对象/方向不明|非动作复述|明确冲突|
|---|---:|---:|---:|---:|---:|---:|
|旧24/original-grouped|9|9|3|3|0|0|
|新24/original-grouped|3|9|6|3|1|2|
|旧24/i16-first-suffix|9|9|2|4|0|0|
|新24/i16-first-suffix|5|7|6|2|1|3|

新集有2条完整性改善，同时出现1条明确退化：`s128-d192-arm-low-wave` 从“右臂挥动一次”变为“摆弄桌子”。后者引入未授权物体操作，不能执行；RAG又返回腰摆及底盘摆。旧集1条从“保持静止”变成“低幅度抬起”，仍没说明左臂或任何部件，也不能直接执行。原图已有两条要求摇头却回答点头的错误，候选同样存在。

RAG Top1明确冲突原图20/48、候选21/48；**每版48/48场景Top3都至少含一个明确冲突**。条件/起始姿态不明另计，不并入明确冲突。示例：右臂→左臂；右转→左转；直线后退→后退弧转；放下手臂→举臂；全身静止→移动底盘；腰左转后保持→左右扭腰并回转。正确的高层文本也可能被错误召回。所有动作mock_only且enabled_on_board=false，未驱动物理执行器。

新旧集都是每12场景×两种静态长度；新集未用来调该候选，但由人工长度组成，不是真人录音或已证明训练集外数据。详见含144回复与432召回项判定的INDEPENDENT_MANUAL_REVIEW.json。

## 系统资源

- 2861个资源样本覆盖1465.22s。最高50.468°C；全8核CPU加权平均7.348%（约0.588核）。PSS峰624.66MiB、RSS632.92MiB，仅所采Python语音/RAG进程及NPU worker。
- MemAvailable最低2666.96MiB。swap用量129.25→245.5MiB；swap-out增加30992页，swap-in0。阶段/加载/文件缓存不同，不能仅凭这组差值诊断泄漏。
- NPU driver ready原753,274,680B、候选786,436,280B，额外33,161,600B（31.63MiB），来自保留原L2及候选。三个worker关闭均0；不能把driver与进程RSS简单相加当物理占用。
- A76采样2,002,000kHz恒定；A55范围416,000–1,794,000kHz。没有NPU频率证据，不推断其是否降频。
- 性能阶段原/候选/重复全CPU为7.136%/7.277%/7.025%；PSS峰595.81/597.23/599.67MiB；温峰47.244/49.352/49.786°C。候选性能阶段swap-in/out均0，整体swap增长不能全归因于它。
- 合成流式音频92546帧、4553次overrun（4.920%），maxlag0.85117s、maxcompute0.16034s，232个端点，队列最大记录为1条整段WAV，无前端异常。累计maxlag在首个资源样本已出现，不能说是候选阶段新产生。前端ASR文本不驱动本轮冻结LLM输入。

## 限制与结论

这不是1小时稳定性、完整语音指令驱动agent或视觉测试；无真实麦克风、图像encoder或电机。只测八格且每格三个例子；boundary20及连续64–256长度未跑。固定反馈速度、完整logits一致性和自然任务正确性须分开。当前可保留为隔离研究版本，不将“32诊断argmax相同”或“43/48自然token相同”当生产资格。

## 逐例人工判定简表

A=原图，B=候选；原图重复48/48与A相同。完整原因及召回分项见JSON。

|case|A回复/类别|B回复/类别|
|---|---|---|
|s64-d64-return-home-nod|点头 / pass|点头 / pass|
|s64-d64-greeting-right-arm|抬起右臂。 / compatible_partial|抬起右臂。 / compatible_partial|
|s64-d64-edge-hold|保持全身静止。 / pass|保持全身静止。 / pass|
|s64-d128-waist-bow|鞠躬一次 / compatible_partial|鞠躬一次 / compatible_partial|
|s64-d128-head-left|头部向左转一点。 / pass|头部向左转一点。 / pass|
|s64-d128-base-back|后退两厘米 / compatible_partial|后退两厘米 / compatible_partial|
|s64-d192-lower-arms|放下手臂 / compatible_partial|放下手臂 / compatible_partial|
|s64-d192-base-turn-right|保持原地向右小角度转弯。 / compatible_partial|保持原地向右小角度转弯。 / compatible_partial|
|s64-d192-child-near-arm-hold|保持全身静止。 / pass|保持全身静止。 / pass|
|s64-d256-head-neutral|缓慢回正头部 / pass|缓慢回正头部 / pass|
|s64-d256-left-arm-correction|保持静止。 / safe_underresponse|低幅度抬起 / underspecified|
|s64-d256-base-left|缓慢移动底盘两厘米 / underspecified|缓慢移动底盘两厘米 / underspecified|
|s128-d64-return-home-nod|点头 / pass|点头 / pass|
|s128-d64-greeting-right-arm|抬起右臂 / compatible_partial|抬起右臂 / compatible_partial|
|s128-d64-edge-hold|保持全身静止。 / pass|保持全身静止。 / pass|
|s128-d128-waist-bow|鞠躬一次 / compatible_partial|鞠躬一次 / compatible_partial|
|s128-d128-head-left|左转 / underspecified|左转 / underspecified|
|s128-d128-base-back|后退两厘米 / compatible_partial|后退两厘米 / compatible_partial|
|s128-d192-lower-arms|放下手臂 / compatible_partial|放下手臂 / compatible_partial|
|s128-d192-base-turn-right|保持静止。 / safe_underresponse|保持静止。 / safe_underresponse|
|s128-d192-child-near-arm-hold|保持静止。 / pass|保持静止。 / pass|
|s128-d256-head-neutral|缓慢回正头部 / pass|缓慢回正头部 / pass|
|s128-d256-left-arm-correction|保持静止。 / safe_underresponse|保持静止。 / safe_underresponse|
|s128-d256-base-left|缓慢移动底盘两厘米。 / underspecified|缓慢移动底盘两厘米。 / underspecified|
|s64-d64-quiet-refusal|点头 / conflict|点头 / conflict|
|s128-d64-quiet-refusal|点头 / conflict|点头 / conflict|
|s64-d64-right-arm-rest|保持静止。 / safe_underresponse|保持静止。 / safe_underresponse|
|s128-d64-right-arm-rest|保持静止。 / safe_underresponse|保持静止。 / safe_underresponse|
|s64-d64-waist-center|保持腰部直立中位。 / compatible_partial|保持腰部直立中位。 / compatible_partial|
|s128-d64-waist-center|保持腰部直立中位。 / compatible_partial|保持腰部直立中位。 / compatible_partial|
|s64-d128-clear-forward|缓慢前进 / compatible_partial|缓慢前进 / compatible_partial|
|s128-d128-clear-forward|缓慢前进 / compatible_partial|缓慢前进 / compatible_partial|
|s64-d128-head-right-soft|轻转头部 / underspecified|轻转头部 / underspecified|
|s128-d128-head-right-soft|轻转头部 / underspecified|轻转头部 / underspecified|
|s64-d128-left-turn-space|左转 / underspecified|底盘原地向左小角度转弯后停住 / pass|
|s128-d128-left-turn-space|底盘原地向左小角度转弯 / compatible_partial|底盘原地向左小角度转弯 / compatible_partial|
|s64-d192-sleep-stop|停止底盘运动。 / compatible_partial|保持全身静止。 / pass|
|s128-d192-sleep-stop|保持全身静止。 / pass|保持全身静止。 / pass|
|s64-d192-arm-low-wave|用户站在机器人正前方，右臂活动范围已确认空旷，左侧有人靠近。 / unjudgeable|用户站在机器人正前方，右臂活动范围已确认空旷，左侧有人靠近。 / unjudgeable|
|s128-d192-arm-low-wave|右臂挥动一次 / compatible_partial|摆弄桌子 / conflict|
|s64-d192-head-down-once|保持静止。 / safe_underresponse|保持静止。 / safe_underresponse|
|s128-d192-head-down-once|保持静止。 / safe_underresponse|保持静止。 / safe_underresponse|
|s64-d256-left-arm-low|保持静止。 / safe_underresponse|保持静止。 / safe_underresponse|
|s128-d256-left-arm-low|保持静止。 / safe_underresponse|保持静止。 / safe_underresponse|
|s64-d256-toy-crossing|保持全身静止。 / pass|保持全身静止。 / pass|
|s128-d256-toy-crossing|保持静止。 / pass|保持静止。 / pass|
|s64-d256-waist-left-small|腰部向左小幅转动一次。 / compatible_partial|腰部向左小幅转动一次 / compatible_partial|
|s128-d256-waist-left-small|腰部向左小幅转动一次。 / compatible_partial|腰部向左小幅转动一次。 / compatible_partial|
