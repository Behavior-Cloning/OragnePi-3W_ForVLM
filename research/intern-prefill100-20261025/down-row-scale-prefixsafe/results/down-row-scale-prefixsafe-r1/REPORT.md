# C512/M32 保留原始 system 路径的逐行量化对照

同一新binary四组：原图、全程row-scale、原始system＋动态row-scale、原图重复。静态正文59/123＋模板5＝64/128；动态正文64/128/192/256＋模板8＝新增72/136/200/264。仅复用system。性能输出固定反馈32token；自然输出上限32。system冷构建不计热TTFT。

纯Intern NPU文本、合成语音前端和RAG驻留；无视觉/真实麦克风/电机。保护路径每个块分3次执行（原组为1次），因此包括这部分调度开销。各格3案，不能称长期稳定；质量待人工复核，不能将token一致率当准确率。

|静态|动态正文|新增prefill|原图 TTFT ms|全程量化 TTFT ms|原system量化动态 TTFT ms|原图重复 TTFT ms|保护组 prefill ms|保护组 decode tok/s|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|64|64|72|841.7|421.6|434.2|841.0|412.9|15.38|
|64|128|136|1558.6|699.8|703.2|1555.4|682.8|15.16|
|64|192|200|2311.5|979.1|982.8|2309.6|962.1|15.12|
|64|256|264|3047.0|1259.7|1269.6|3057.5|1247.2|15.04|
|128|64|72|850.6|420.2|423.2|852.0|402.3|15.20|
|128|128|136|1606.3|701.8|720.5|1598.1|698.5|15.55|
|128|192|200|2357.0|980.4|1026.1|2338.9|1004.8|15.06|
|128|256|264|3095.4|1256.9|1339.6|3090.2|1318.0|15.13|

自然token完全一致数（各48例，非正确率）：{"original-m32": 48, "row-scale-m32": 44, "protected-system": 45, "original-m32-repeat": 48}。

System-only logits检查：[{"variant": "row-scale-m32", "static_tokens": 64, "byte_equal": false}, {"variant": "row-scale-m32", "static_tokens": 128, "byte_equal": false}, {"variant": "protected-system", "static_tokens": 64, "byte_equal": true}, {"variant": "protected-system", "static_tokens": 128, "byte_equal": true}, {"variant": "original-m32-repeat", "static_tokens": 64, "byte_equal": true}, {"variant": "original-m32-repeat", "static_tokens": 128, "byte_equal": true}]。不能据此证明所有KV逐字节一致。

最高观测温度52.080°C（资源采样峰50.592°C）；全8核CPU加权占用7.61%；进程PSS峰623.9MiB；MemAvailable最低1948.5MiB。含加载/诊断；不是纯热循环或全系统内存。四次driver释放0。

全部性能格≤100ms：{"original-m32": false, "row-scale-m32": false, "protected-system": false, "original-m32-repeat": false}。完整目标仍未验证。

音频结束统计：{"frames": 70433, "overruns": 3465, "max_lag_s": 0.8357626640063245, "max_compute_s": 0.16022441498353146, "stage_seconds": {"denoise": 253.1522483335866, "vad": 25.71156879371847, "kws": 79.21149202482775, "stream_asr": 63.84155654316419}, "queue": 1, "error": null}。overrun与lag单列，不能假定全程实时。

Swap占用范围1364.8–1521.5MiB；监测区间swap-in3465页、swap-out43463页。观测时长18.49分钟，含加载、切换与诊断；不归因于单一候选。

|静态|动态正文|保护组TTFT P95 ms|保护组setup P50 ms|保护组head P50 ms|≤100ms次数/3|
|---:|---:|---:|---:|---:|---:|
|64|64|435.99|1.93|18.96|0/3|
|64|128|704.68|2.12|17.83|0/3|
|64|192|984.68|1.98|17.88|0/3|
|64|256|1269.85|1.97|19.17|0/3|
|128|64|460.85|2.11|18.25|0/3|
|128|128|776.29|1.99|19.00|0/3|
|128|192|1029.58|2.01|18.20|0/3|
|128|256|1341.18|2.00|18.26|0/3|

P95为nearest rank，n=3时等于最大值。分phase资源与逐case原图重复漂移见analysis.json。
