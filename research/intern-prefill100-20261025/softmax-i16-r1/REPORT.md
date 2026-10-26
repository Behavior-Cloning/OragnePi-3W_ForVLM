# I16 softmax输入局部筛查

不采用：未出现超出波动的速度收益，同时改变数值。仅L0/L1、query64、past128、有效history128，纯vip_run_network，不含完整prefill/TTFT/decode/语音/RAG。system正文与模板token口径不适用于局部张量。完整目标仍为system64/128（正文59/123+模板5），动态64/128/192/256+模板8，new72/136/200/264，system-only KV，prefill约500ms。

|版本|样本|P50 ms|P95 ms|
|---|---:|---:|---:|
|before|30|13.792860|13.853277|
|i16|30|13.791506|13.844777|
|after|30|13.796047|13.820527|

原权重/拓扑/其他量化未改；只把softmax前加mask结果从FP16改成已有score的I16尺度（两层fl3/5），-10000 mask会饱和，并非无损。原前后输出同字节，候选hidden相对L2差0.372097%，不是准确率下降百分比。全部输出有限、实际360448B与SHA重核、三次释放driver0。M32/past448也已编译但未运行，不外推。

90次正式计时均3次预热后执行；最高41.416°C、最低可用2073.23MiB。源码652d773；图/输入/输出/config SHA见build.json、deployment-sha.json、ANALYSIS.json。保留否定结果，不移动最快性能、质量或默认指针。
