# C512/M32 原始 system 保留，主批量与尾段逐行量化对照

同一新binary四组：原图、原始system＋仅主批量row-scale、原始system＋主批量及尾段row-scale、原图重复。静态正文59/123＋模板5＝64/128；动态正文64/128/192/256＋模板8＝新增72/136/200/264。仅复用system。性能输出固定反馈32token；自然输出上限32。system冷构建不计热TTFT。

纯Intern NPU文本、合成语音前端和RAG驻留；无视觉/真实麦克风/电机。保护路径每个块分3次执行（原组为1次），因此包括这部分调度开销。各格3案，不能称长期稳定；质量待人工复核，不能将token一致率当准确率。

|静态|动态正文|新增prefill|原图 TTFT ms|仅主批量量化 TTFT ms|主批量＋尾段量化 TTFT ms|原图重复 TTFT ms|主＋尾候选 prefill ms|主＋尾候选 decode tok/s|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|64|64|72|842.5|423.8|373.0|853.2|353.0|15.32|
|64|128|136|1556.5|702.3|652.7|1555.7|632.2|15.25|
|64|192|200|2308.7|984.2|934.7|2298.3|914.2|15.10|
|64|256|264|3057.5|1266.7|1217.7|3050.1|1195.1|15.28|
|128|64|72|851.8|422.3|370.3|852.3|350.1|15.05|
|128|128|136|1604.6|715.3|667.9|1603.0|646.1|15.05|
|128|192|200|2358.0|1025.0|978.5|2355.6|956.5|15.05|
|128|256|264|2995.0|1336.4|1289.5|3093.5|1267.0|15.46|

自然token完全一致数（各48例，非正确率）：{"original-m32": 48, "protected-main": 45, "protected-main-tail": 45, "original-m32-repeat": 48}。

System-only logits检查：[{"variant": "protected-main", "static_tokens": 64, "byte_equal": true}, {"variant": "protected-main", "static_tokens": 128, "byte_equal": true}, {"variant": "protected-main-tail", "static_tokens": 64, "byte_equal": true}, {"variant": "protected-main-tail", "static_tokens": 128, "byte_equal": true}, {"variant": "original-m32-repeat", "static_tokens": 64, "byte_equal": true}, {"variant": "original-m32-repeat", "static_tokens": 128, "byte_equal": true}]。不能据此证明所有KV逐字节一致。

最高观测温度51.460°C（资源采样峰51.460°C）；全8核CPU加权占用7.45%；进程PSS峰610.9MiB；MemAvailable最低1802.8MiB。含加载/诊断；不是纯热循环或全系统内存。四次driver释放0。

全部性能格≤100ms：{"original-m32": false, "protected-main": false, "protected-main-tail": false, "original-m32-repeat": false}。完整目标仍未验证。

音频结束统计：{"frames": 73760, "overruns": 3629, "max_lag_s": 0.648882095963927, "max_compute_s": 0.18390616998658516, "stage_seconds": {"denoise": 260.3438524513622, "vad": 27.35432424908504, "kws": 83.07505164356553, "stream_asr": 66.90816263202578}, "queue": 1, "error": null}。overrun与lag单列，不能假定全程实时。

Swap占用范围1521.2–1809.8MiB；监测区间swap-in4082页、swap-out78051页。观测时长19.41分钟，含加载、切换与诊断；不归因于单一候选。

|静态|动态正文|主＋尾候选TTFT P95 ms|主＋尾候选setup P50 ms|主＋尾候选head P50 ms|≤100ms次数/3|
|---:|---:|---:|---:|---:|---:|
|64|64|373.22|1.99|17.76|0/3|
|64|128|652.72|1.97|17.70|0/3|
|64|192|1015.46|1.99|17.73|0/3|
|64|256|1254.03|2.10|19.00|0/3|
|128|64|374.74|2.00|17.75|0/3|
|128|128|668.63|2.11|19.08|0/3|
|128|192|983.49|1.93|18.93|0/3|
|128|256|1292.32|1.99|19.13|0/3|

P95为nearest rank，n=3时等于最大值。分phase资源与逐case原图重复漂移见analysis.json。

尾段增量比较（B→C）：48自然样例token变化数=2；完整logits、尾段秒数及主批量前后变化按case独立存储analysis.json。仍需按指令判语义。
