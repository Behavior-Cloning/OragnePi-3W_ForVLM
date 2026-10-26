# Softmax形状改写：两层局部筛查

结论：不晋升。M64/past128相同输入下，二维softmax改写略慢，输出与A/A逐字节相同；没有完整模型TTFT或decode结论。M32/past448仅编译，未实测。

改写保留完整key轴，把[1,2,7Q,K]变为[14Q,K]、最后一维softmax后恢复。原量化与权重不变。无语音/RAG并发；只运行L0/L1纯VIP API。每组3次预热、30次计时。

|路径|样本|API P50 ms|P95 ms|输出有限/同字节|driver释放|
|---|---:|---:|---:|---|---|
|before|30|13.804|13.841|True/True|True|
|flat|30|13.904|13.946|True/True|True|
|after|30|13.811|13.840|True/True|True|

全部输出360448B，真实SHA已重算；峰值温度44.888°C，最低可用内存2122.6MiB。单个局部fixture不能证明全模型无损，也不能推断端到端提速。

输入是冻结的past128/query64局部张量，system/动态模板计数不适用于局部算子。没有生成token，TTFT和decode未测。完整token口径见独立M64全模型报告。

源代码提交4a4c57e3434e3bedbbb6be89ece8008cd9c1bbb7；图、配置、输入及输出SHA分别见config.json、deployment-sha.json、ANALYSIS.json。保持原始文件，不覆盖父图或默认部署。
