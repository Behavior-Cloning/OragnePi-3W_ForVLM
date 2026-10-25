# M256 prefill 节点诊断：softmax 的非线性增长

日期：2026-10-26。实验 profile-m256-l0-r1 已完成，控制器 rc=0，两个探针均 released_bytes=0。此为原量化 Intern 的 L0/L1 两层单块诊断，无语音/RAG常驻，不是完整模型TTFT。

## 冻结条件与结果可信度

C512、物理查询M256、canonical history128，11段输入保持原样。NBG SHA e9b2d8af40f96341eda1e7ff07b867ccde9bbcf9a755746c86e6dafec2de11d5；带节点trace重新导出的NBG与原文件逐字节一致。IO与NPD两个模式的输出均1,441,792B，SHA2548582d5782a9e74f36c02ad70d65a082e16c652c4779d8786d3cfc62934414，与归档纯计时输出完全相同且finite。NPD实际SHA901764ad8309d0205af6c8168dae749964a6c1f561e3cbab328c0595dafbfd85，原始SDK字段解析和独立旧解析器逐条一致，无overflow。

IO诊断：5次run P50 210.242ms，flush+invalidate P50 0.046458ms，wall P50 210.289ms。显式IO同步仅占约0.0221%；这不包含所有潜在运行期内存传输。NPD诊断API268.991ms有profile开销，严禁作为正式速度样本。旧纯计时M256 P50 211.133ms、M32 10.920ms，均只是两层块。

## M32 与 M256 的原始NN计数

|算子|M32 busy cycles|M256 busy cycles|总计数倍率|按查询行数归一化倍率|
|---|---:|---:|---:|---:|
|softmax|1,069,254|111,509,399|104.29×|13.04×|
|fullconnect|4,466,100|41,695,927|9.34×|1.17×|
|add|1,436,207|16,626,139|11.58×|1.45×|
|concat|625,840|9,056,239|14.47×|1.81×|
|matmul|773,807|8,708,733|11.25×|1.41×|

M256 softmax占已观察NN busy计数53.86%，M32为10.66%。注意：M32 softmax形状[1,2,224,544]，M256为[1,2,1792,768]；查询行8倍之外，完整key宽度还增加768/544=1.412倍。按所有softmax元素归一化仍约9.23倍。不能将这当作纯行数效应，也不能把NN计数比例当wall-time或CPU占比。trace缺失engine target，缺少NN记录不证明CPU回退。

由此下一步优先测试softmax的查询行分块，而非继续扩大整个图batch。候选保持每一行完整768个key与原FP16输入/I16 fl15输出，切[1,2,1792,768]的axis2为8×224行，分别softmax后按axis2拼回；权重、原量化参数、mask/history均不变。数学操作可分，但NPU数值实现是否完全一致必须端侧验证。该候选目前仅编译，不计作已提速。

## 系统观测及目标边界

16个连续资源样本，最高39.866°C，最少可用内存4,024,786,944B；swap占用首尾均941824KiB。诊断短且无语音，不代表长时间全链路稳定。

完整目标仍为固定system64/128（正文59/123+模板5），动态正文64/128/192/256+模板8，实际新增prefill72/136/200/264；仅system KV复用，完整prefill与TTFT都<=100ms。目前未达到。本轮没有缩短输入、裁剪注意力范围或降低目标。

原始结果：results/profile-m256-l0-r1。复核脚本：runtime/profile-analysis/verify_m256.py、analyze_m256.py、map_trace_m256.py。计数对照：comparison-m32-m256.json。
