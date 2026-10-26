# ViT NPU 完整块栈与速度诊断 r1

日期：2026-10-27。候选未晋升默认，112×112 保留64 patch路线保持独立。

## 测试边界

224×224，256 patch＋1 CLS，完整24个Transformer块。输入stem在WSL预计算，projector仅离线分析；不是完整板端vision tower延迟，不包含JPEG、语言、语音或RAG。无system/dynamic文字输入；输出是257×1024 hidden，离线projector产生64×896视觉向量。2张开发图片、1次预热＋6次热运行，不能代表长期稳定或通用VQA。

## 速度与质量

| 项目 | 实测 |
|---|---:|
| 24块总时间中位数 | 1.833秒 |
| NPU调用时间之和中位数 | 1.307秒 |
| 层间主机转换／同步中位数 | 0.526秒（约29%） |
| NPU加载占用 | 692,745,312 bytes |
| 采集的最高温度 | 40.734°C |
| 关闭后driver bytes | 0 |

数值重复性通过，但最终hidden相对L2误差为83.27%／87.66%，cosine约0.67／0.68，**质量失败**。不能把此候选替换现有CPU视觉路径，也不能与CPU完整tower 2.247秒直接计算公平加速比。

逐层误差显示第13层（索引12）开始增大、第14层（索引13）突变。独立板端诊断对两张图分别使用FP32参考输入和上一层NPU实际输入，并以相同输入的PyTorch计算为参考：

| 单层 | 参考输入相对L2误差（两图） | 上游实际输入相对L2误差（两图） |
|---|---|---|
| 第13层 | 14.49%／17.34% | 14.68%／16.83% |
| 第14层 | 106.08%／99.90% | 98.26%／84.22% |

因此存在单层量化／执行偏差，不能仅归因于误差累积。具体算子根因尚未确认。12个独立进程均重复一致、关闭driver0。指标JSON中probe历史scope字符串称first block；以isolate-cases.json层索引为准，实际覆盖0、12、13层。

## W8A16局部速度候选

同一第1层、相同224输入，六个FC权重改W8 per-channel，其余保留校准INT16：NPU API约54.4→49.8ms，约8.5%降低。相对L2从3.814%／1.736%变为3.863%／1.802%，非无损。另有约25ms输入转换／同步／诊断读回，此项包含检查，不能直接当生产开销。仅两个开发输入，不是完整24层W8模型验证。

尝试第14层W8＋非线性FP16的诊断配置，在SDK VerifyGraph阶段报unsupported configuration、生成失败，无NB、无板端计时。保留失败证据，不覆盖INT16基线。

## 下一步

1. 第14层分段输出与选择性精度诊断，先定位LayerNorm、attention、GELU／MLP中的异常，恢复质量。
2. 减少24层间F32↔INT16转换和同步；需统一量化边界或合并图，不能直接拷贝不同量化尺度的整数。
3. 质量通过后再扩展W8、集成已编译stem/projector，并测真实JPEG→视觉64token→NPU语言；112保留64路线仍需修复projector质量。

原可复现常驻路径仍为CPU Q8重排视觉＋NPU语言：JPEG→首token P50 2.615秒，视觉2.247秒，语言首token0.309秒，decode16.57token/s；这是另外一次短测试。system正文30＋模板5＝35，reuse32；动态视觉64＋文字5–9，模板/边界11，实际new83–87。该路径也有回答错误扩写，尚未质量合格。

证据：stack-audit.json、isolate-audit.json、isolate-cases.json、compile-full-status.json、nonlinear-failed-conversion.json、results/与isolate-results/原始数据。原始权重、NB、f32保留磁盘但不提交Git。
