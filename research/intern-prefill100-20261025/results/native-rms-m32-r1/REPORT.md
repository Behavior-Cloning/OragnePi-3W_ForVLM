# 原生 RMSNorm 两层算子筛查

仅L0/L1、物理M32、有效history128；不是完整system128+dynamic32请求，也不是整模TTFT。正常canonical和合成maxabs4096 hidden分别A/B/A。每进程3warm+30计时，仅vip_run_network，不计加载/转换/全模型head；无语音/RAG常驻。

|输入/路径|P50 ms|P95 ms|非finite元素|
|---|---:|---:|---:|
|normal-before|10.8673|10.9284|0|
|normal-native-rms|10.2023|10.2910|0|
|normal-after|10.8971|10.9345|0|
|stress-before|10.8669|10.8915|0|
|stress-native-rms|10.2010|10.3417|0|
|stress-after|10.8667|10.8907|0|

完整五输出按真实文件SHA、实际16端口对照；两组A/A字节一致。误差分段见ANALYSIS.json。原生gamma/epsilon表示与旧路径有差异，不称无损；synthetic stress不代表语言准确率。100ms全目标仍未实现。

资源：{"max_temperature_C": 41.726, "min_mem_available_B": 3503157248, "swapin_pages": 0, "swapout_pages": 0, "sampled_max_temperature_C": 41.726}。6次driver释放0。
