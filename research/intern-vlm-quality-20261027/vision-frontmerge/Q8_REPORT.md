# 视觉塔Q8量化：节省内存，当前CPU实现未提速

2026-10-27。**224输入/64视觉token：FP16约4.002秒，Q8约4.294秒，慢7.28%；视觉进程峰值RSS从约612.4MiB降到341.5MiB，少约270.9MiB（44.2%）。** 6道开发题核心事实均为5/6。Q8作为内存候选保留，不晋升速度最优或质量通过版本。

## 量化范围与一致性

源mmproj F16文件619,876,640 bytes，SHA `89e15ad85554c6a29910187c4f9689d992e8a7b586941cef68651bf91b6bc4b8`。仅144个ViT block线性矩阵改为GGUF Q8_0，298个其他张量（stem、projector、位置、norm、bias等）逐字节保持。metadata只更新file_type/quantization_version，其他逐字段一致。Q8文件336,761,120 bytes，SHA `65c3352c1066f38d1cc5d02d89d7c355f16cda959430355f606bb8d26449892a`，体积减少45.67%。

这是CPU GGML权重量化，不是已导出的NPU INT8图，没有做A733激活校准。每层权重量化误差有记录，但不能将权重L2当VQA准确率。

## 板端配对

同一worker binary、CPU6/7、2线程、图片和normalized pixels不变，常驻热推理。顺序FP16-224→Q8-224→Q8-448后置pool→FP16-224复测。每phase一遍预热＋两遍热样本。

|条件|ViT patch＋CLS|LLM视觉token|ViT＋投影热P50秒|进程峰值RSS MiB|6题核心事实|
|---|---:|---:|---:|---:|---:|
|FP16-224 前对照|256+1|64|4.015|612.4|5/6|
|Q8-224|256+1|64|4.294|341.5|5/6|
|Q8-448 后置池化|1024+1|64|26.712|423.9|6/6|
|FP16-224 后对照|256+1|64|3.989|612.8|5/6|

FP16前后控制合并4个热样本P50=4.002秒；Q8-224仅2热样本，不报告P95/长期稳定性。Q8-448没有同轮448前后对照，旧轮F16-448后置pool约24.31秒只能参考，不能当严格同期配对。

本轮最高温度65.1°C、MemAvailable最低1597.7MiB；顺序试验有热积累，不能把分组最高温度当独立热设计比较。加载时间受文件page cache影响，未清系统缓存。未测功耗/PSS/频率，不声称无降频。

## VQA口径

视觉embedding在板上产生；语言参考在Windows WSL用配套Q8模型运行。system正文30＋模板5＝35；动态图片64＋问题正文5–9＝69–73；额外模板/图像边界11；总输入115–119。每题清空KV，reuse0，实际new115–119；输出上限32。此处不是system常驻的板端NPU语言TTFT，更不是语音全链路。

Q8-224与FP16-224同样把木板误答树枝，其余5项核心事实正确；熊猫/大熊猫措辞变化不算错误。Q8-448后pool核心事实6/6，但仅2张图6道简单开发题，不足以证明真实生活/OCR/计数/空间关系可靠。

## 还有哪些优化空间

1. Q4可做下一档CPU实测，检查对应ARM内核/packing是否更合适；位宽更低不能自动承诺更快或质量不变。
2. 本轮没有算子profiling，尚不能把Q8变慢精确归因于反量化或某个kernel；应测量weight packing、activation conversion和GEMM时间，再决定CPU优化。
3. 转NPU需要真正导出ViT和projector、处理LayerNorm/GELU/attention/softmax、校准激活并测数据搬运。官方INT8算力不代表当前CPU Q8自动调用NPU。
4. 前置token合并和112保留64的投影适配仍需修复质量；量化本身不解决先前的信息/表征不匹配。112又尝试了保留全部位置的stride1相邻2×2投影适配，在同6题WSL参考中仍0/6，已停在host筛选，未浪费板端时间推广。

原文本NPU最优r6、所有基线与默认均未覆盖。Q8资产留Windows `F:/Files/OPi-Lab/models/internvl3-1b-vision/`；板上在`/opt/opi-vlm/experiments/vision-q8-r1/encoder-q8.gguf`。证据：vision-q8-manifest.json、Q8_AUDIT.json、q8-board-results/results/summary.json、q8-vqa-answers.jsonl。
