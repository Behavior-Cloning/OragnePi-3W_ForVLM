# C1280 FP16 Conv L2：长语义质量诊断

结论：该候选仍属于实验优化，未通过完整动作决策质量验收。48次自然请求正常完成，两个版本均只有9/24严格完成最终要求；没有观察到明确冲突的自然短语，但遗漏与主体/方向不明不能作为已验证可执行。旧C256最快结果不由这轮C1280实验替代。

## 实际输入和证据边界

固定system总token为64/128，其中正文59/123、模板5；动态正文64/128/192/256，另有user/assistant模板8，实际新增prefill72/136/200/264。每次仅恢复system KV，未缓存动态内容。自然输出上限32，48条均EOS结束，未截断。

12个新编家庭情景×2种system，共24条/版本；同一场景配对使用完全相同输入。各正文长度对应不同情景，且system128含原有重复指令，不能据此证明长度单一因素的质量影响。另有1条旧成本输入/版本，不计入24条语义集。

A=原C1280，B=仅prefill-layer2替换FP16 Conv1×1。其余28个依赖哈希、量化与runtime flags保持一致；system准备同样经过候选图，因此缓存策略相同不等于缓存数值相同。源码/输入/29依赖哈希、52请求token accounting、两个close driver0均核对通过。

## 人工语义复核

按原指令与冻结rubric评估完成、兼容性及明确冲突，不把“保持静止”一律判正确。以下是可见候选标签的复核，不是盲测。逐条原因及全部RAG排名见semantic-review.json。

|类别|原C1280|Conv L2|
|---|---:|---:|
|pass|9|9|
|compatible_partial|8|8|
|safe_underresponse|3|2|
|underspecified|4|5|
|conflicting_or_infeasible|0|0|

独立标签遮蔽审计将“保持原地向右小角度转弯”记compatible_partial，本复核按缺少底盘主体记underspecified；这是1条严格度分歧，双方完整通过数均9/24、明确冲突均0。未将分歧隐藏或改成通过。

22/24自然回复token完全相同。两条变化：
- s64-d128-waist-bow：A「鞠躬一次」→B「鞠躬一次。」。
- s64-d256-left-arm-correction：A「保持静止。」→B「低幅度抬起」。

鞠躬一例仅标点变化；左臂纠正一例从安全未执行变为欠明确动作。B的“低幅度抬起”仍缺左臂主体，不能据此证明已正确避开右侧障碍。

|案例|A回复|B回复|A类别|B类别|
|---|---|---|---|---|
|s64-d64-return-home-nod|点头|点头|pass|pass|
|s64-d64-greeting-right-arm|抬起右臂。|抬起右臂。|compatible_partial|compatible_partial|
|s64-d64-edge-hold|保持全身静止。|保持全身静止。|pass|pass|
|s64-d128-waist-bow|鞠躬一次|鞠躬一次。|compatible_partial|compatible_partial|
|s64-d128-head-left|头部向左转一点。|头部向左转一点。|pass|pass|
|s64-d128-base-back|后退两厘米|后退两厘米|compatible_partial|compatible_partial|
|s64-d192-lower-arms|放下手臂|放下手臂|compatible_partial|compatible_partial|
|s64-d192-base-turn-right|保持原地向右小角度转弯。|保持原地向右小角度转弯。|underspecified|underspecified|
|s64-d192-child-near-arm-hold|保持全身静止。|保持全身静止。|pass|pass|
|s64-d256-head-neutral|缓慢回正头部|缓慢回正头部|pass|pass|
|s64-d256-left-arm-correction|保持静止。|低幅度抬起|safe_underresponse|underspecified|
|s64-d256-base-left|缓慢移动底盘两厘米|缓慢移动底盘两厘米|underspecified|underspecified|
|s128-d64-return-home-nod|点头|点头|pass|pass|
|s128-d64-greeting-right-arm|抬起右臂|抬起右臂|compatible_partial|compatible_partial|
|s128-d64-edge-hold|保持全身静止。|保持全身静止。|pass|pass|
|s128-d128-waist-bow|鞠躬一次|鞠躬一次|compatible_partial|compatible_partial|
|s128-d128-head-left|左转|左转|underspecified|underspecified|
|s128-d128-base-back|后退两厘米|后退两厘米|compatible_partial|compatible_partial|
|s128-d192-lower-arms|放下手臂|放下手臂|compatible_partial|compatible_partial|
|s128-d192-base-turn-right|保持静止。|保持静止。|safe_underresponse|safe_underresponse|
|s128-d192-child-near-arm-hold|保持静止。|保持静止。|pass|pass|
|s128-d256-head-neutral|缓慢回正头部|缓慢回正头部|pass|pass|
|s128-d256-left-arm-correction|保持静止。|保持静止。|safe_underresponse|safe_underresponse|
|s128-d256-base-left|缓慢移动底盘两厘米。|缓慢移动底盘两厘米。|underspecified|underspecified|

## 首token翻转与全词表诊断

旧成本输入s64-d64-2：system64（59+5），正文64+模板8，prompt136，实际新增72。A自然输出「挥手致意。」，B「举起手臂，轻柔地挥动。」。两者均是兼容招呼意图的表达变化；仍需幅度/碰撞约束执行层，不能自动执行Top3。该例不计入长语义完成率。

同一固定反馈的4步×151674全词表F32已核验大小、SHA、finite与argmax。首步A=117057「挥手」，B=114209「举起」；A top1-top2 margin=0.04687500，B margin=0.02148438，原A赢家在B中的signed margin=-0.02148438。

4步argmax一致3/4，整体logits相对L2=0.02750844，maxabs=0.46093750。这只是固定反馈诊断，第二步以后不是两条自由生成轨迹的直接配对。首步两token具体logits：

```json
{
  "c1280-original": {
    "117057": 19.9375,
    "114209": 19.890625
  },
  "c1280-conv-l2": {
    "117057": 19.8125,
    "114209": 19.833984375
  }
}
```

## RAG与执行兼容性

RAG实际常驻并执行50次查询；它检索模型生成的动作短语，未直接使用完整原指令的禁止/纠正条件。下表人工判断仍以原始指令为准，cosine相似度不是动作兼容概率。没有物理电机执行。

|版本|Top3中至少一个明确冲突的列表|Top1类别|
|---|---:|---|
|c1280-original|23/24|{"compatible": 5, "conflicting": 9, "compatible_partial": 10}|
|c1280-conv-l2|23/24|{"compatible": 5, "conflicting": 10, "compatible_partial": 9}|

常见冲突包含：只点头却召回绕头；底盘不得移动却召回平移；已取消挥臂却召回转腰。部分动作虽可执行但没有完成目标，单列compatible_partial；参数/当前姿态不足列unknown，不直接算冲突。两个版本仍不能把Top3无条件全执行。

## 系统资源和时序

纯文本Intern NPU decoder＋录音WAV流式语音前端＋常驻RAG；没有图像encoder、真人麦克风或物理动作。语音输入独立于本文冻结文字请求，不能当作端到端“用户说完到回复”的测试。自然请求只有每格3次不同文本，报告的TTFT用于质量运行观测，不替代固定32输出性能AB；全logits诊断不纳入纯速度。system prefix准备单列并位于动态TTFT计时前。

|版本|整机平均CPU（8核）|最高温度|合计峰值PSS MiB|最低可用内存 MiB|ready driver字节|
|---|---:|---:|---:|---:|---:|
|c1280-original|7.43%|46.438°C|600.72|3305.08|753274680|
|c1280-conv-l2|7.64%|48.918°C|600.83|3300.33|753278472|

全程语音18739帧，overrun=939（5.01%），最大lag=0.641673s、单帧最大计算=0.639842s；队列最多保留一整段WAV。不能声称语音全程实时无积压。

全程最高温度48.918°C；swap-in增加0页、swap-out增加0页。详细分阶段CPU/频率/温度/RSS/PSS/音频累计计数见analysis-quality.json。PSS合计只含显式Python和resident PID，driver另列，二者不能直接相加当独立内存。两个resident正常退出driver归零。

## 验收结论

有限样本未发现自然文本明确冲突，但完成率37.5%，关键主体/方向遗漏仍存在；候选还产生了非逐位logits变化和一条语义轨迹变化，因此不批准为质量无损默认版本。保留原基线及候选，后续应先完善约束保持与RAG筛选，而不是用输出一致性代替质量。

分析器SHA256：`87521a7c79b9aa7ac014b403178e7c44196b8bf44371eb2fac5d3f675a4ece7a`。原始JSONL/两个F32全词表文件保留，全部分析为离线读取，没有新板端运行。
