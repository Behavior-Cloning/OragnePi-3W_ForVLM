# Compact history + row-normalized down 两层筛查

有效history128全部保留；query64/256是局部批次，不是完整用户输入/系统长度。每形状同输入original/row/original，6进程、18预热、180计时。仅vip_run_network，无语音/RAG，不计加载与整模head。

|路径|局部API P50 ms|P95 ms（线性分位数）|非finite元素|
|---|---:|---:|---:|
|m64-before|435.1734|437.0709|0|
|m64-row|20.2526|20.2872|0|
|m64-after|435.4719|437.5265|0|
|m256-before|1760.6717|1765.3090|0|
|m256-row|102.4007|103.4713|0|
|m256-after|1760.0199|1764.8291|0|

误差按hidden/K/V五输出分别见ANALYSIS.json。A/A完整字节同；row改变量化，不代表无损或语义准确。较晚位置须扩大历史bank，禁止丢有效历史。完整100ms目标仍未验证。

资源：{"sampled_temperature_peak_C": 45.322, "boundary_temperature_peak_C": 44.268, "min_mem_available_B": 3206762496, "swapin_pages": 0, "swapout_pages": 0, "observed_temperature_peak_C": 45.322}。
