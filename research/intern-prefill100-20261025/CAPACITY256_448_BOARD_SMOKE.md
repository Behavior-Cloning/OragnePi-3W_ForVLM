# C256/C448 完整24层板端冒烟报告

日期：2026-10-26

本轮验证完整24层 Intern NPU 文本 worker 的容量缩减图。C256和C448使用plain Transformer图，保留原权重与量化数据；未使用row-scale、卷积替换或动态KV缓存捷径。

## 输入口径

| system正文 | system模板 | 固定总量 | dynamic正文 | dynamic模板 | 实际new prefill | prompt总量 |
|---:|---:|---:|---:|---:|---:|---:|
|59|5|64|64|8|72|136|
|59|5|64|256|8|264|328|
|123|5|128|256|8|264|392|

只复用system KV的测试中，reused_tokens为64或128；动态正文没有复用。首token包含worker setup、prefill、尾部标量处理和head，不包含语音链路或RAG。

## C256完整worker

C256的12组prefill图和board已有的C256 decode图可加载，worker ready为layers=24、capacity=256，关闭后driver_bytes=0。

- system64 + dynamic64：prompt136、new72；冷请求TTFT 0.830 s，prefill0.439 s，scalar prompt0.356 s。
- 复用system64、改变动态首token：prompt136、reused64、new72；TTFT 0.606 s。
- system64 + dynamic128公平变体：prompt200、reused64、new136；TTFT 0.822 s。
- system128 + dynamic64：prompt200、new200；TTFT 1.086 s。

C256不能覆盖system128 + dynamic256（总prompt392），因此不是完整目标方案。

## C448完整worker

C448 decode query1和prefill query32共24张图通过主机端16端口审计。板端加载为24层、capacity448，关闭后driver_bytes=0。

| system总量 | dynamic正文 | 实际new | prompt总量 | system复用 | TTFT | prefill | scalar尾部 |
|---:|---:|---:|---:|---:|---:|---:|---:|
|64|256|264|328|0|1.617 s|1.207 s|0.360 s|
|64|256|264|328|64|1.373 s|0.984 s|0.359 s|
|128|256|264|392|0|1.916 s|1.455 s|0.358 s|
|128|256|264|392|128|1.346 s|0.959 s|0.358 s|

最高观测温度约42.2°C，worker释放后NPU driver为0。测试使用合成token、纯文本Intern NPU和mock worker；无真实麦克风、视觉输入或电机执行。

## 结论

容量从512缩到448可以覆盖最大prompt，但没有把完整模型prefill压到100ms；最大边界在复用system后仍约1.35–1.37s。主要成本是多次query32 batch prefill和约8-token scalar尾部，继续缩容量不能解决全部目标。C256短输入复用system后约0.61s，但不覆盖最大边界，也不能作为全目标达标。

下一步应优先实测tail8图、减少scalar尾部，并评估更大query图或runtime调度开销；候选必须在system64/128 × dynamic64/128/192/256完整24层矩阵重新验证。
