# Intern NPU 下一轮基准分析

只分析回收原始证据；没有启动板卡。原始文件保持不变。

状态：complete；结构/释放审计：True

输入定义：static包含system模板；动态正文32/64，另有8个模板token，实际new应为40/72。
速度仅32步固定反馈、diagnostic=False；自然质量和全词表诊断独立。

| 配置 | static/dynamic正文 | n | TTFT P50/P95秒 | decode P50 tok/s | batch prefill P50秒 | 首head P50秒 |
|---|---|---:|---:|---:|---:|---:|
| combined-base | s64-d32 | 6 | 0.5794 / 0.5822 | 15.4192 | 0.5611 | 0.0178 |
| combined-base | s64-d64 | 6 | 0.9550 / 0.9593 | 15.2672 | 0.9362 | 0.0178 |
| combined-base | s128-d32 | 6 | 0.5832 / 0.6124 | 15.2313 | 0.5635 | 0.0190 |
| combined-base | s128-d64 | 6 | 0.9788 / 1.0017 | 15.2405 | 0.9591 | 0.0191 |
| combined-tail8 | s64-d32 | 6 | 0.4938 / 0.4974 | 15.4985 | 0.4752 | 0.0177 |
| combined-tail8 | s64-d64 | 6 | 0.8698 / 0.8750 | 15.3844 | 0.8508 | 0.0177 |
| combined-tail8 | s128-d32 | 6 | 0.4948 / 0.4965 | 15.2851 | 0.4756 | 0.0185 |
| combined-tail8 | s128-d64 | 6 | 0.8905 / 0.9156 | 15.2978 | 0.8705 | 0.0189 |
| combined-repeat | s64-d32 | 6 | 0.5796 / 0.5831 | 15.4248 | 0.5613 | 0.0177 |
| combined-repeat | s64-d64 | 6 | 0.9556 / 1.0042 | 15.3113 | 0.9371 | 0.0178 |
| combined-repeat | s128-d32 | 6 | 0.5814 / 0.6104 | 15.2407 | 0.5615 | 0.0192 |
| combined-repeat | s128-d64 | 6 | 0.9764 / 0.9894 | 15.2619 | 0.9568 | 0.0191 |

P95采用nearest-rank；每格少于20次时P95通常等于样本最大值，重复同fixture不等于独立用户场景，不能当稳定尾延迟结论。

## 按同case/repeat配对比较

先逐对计算ratio，再汇总；TTFT candidate/base越小越好，decode candidate/base越大越好。mask-repeat用于顺序漂移检查，不能当新优化。

| 配置 | 格 | 配对数 | TTFT ratio P50 | decode ratio P50 |
|---|---|---:|---:|---:|
| combined-base | s64-d32 | 6 | 1.0000 | 1.0000 |
| combined-base | s64-d64 | 6 | 1.0000 | 1.0000 |
| combined-base | s128-d32 | 6 | 1.0000 | 1.0000 |
| combined-base | s128-d64 | 6 | 1.0000 | 1.0000 |
| combined-tail8 | s64-d32 | 6 | 0.8513 | 1.0046 |
| combined-tail8 | s64-d64 | 6 | 0.9106 | 1.0078 |
| combined-tail8 | s128-d32 | 6 | 0.8487 | 1.0034 |
| combined-tail8 | s128-d64 | 6 | 0.9094 | 1.0051 |
| combined-repeat | s64-d32 | 6 | 1.0008 | 0.9995 |
| combined-repeat | s64-d64 | 6 | 1.0007 | 1.0020 |
| combined-repeat | s128-d32 | 6 | 0.9974 | 1.0025 |
| combined-repeat | s128-d64 | 6 | 0.9943 | 1.0015 |

## 质量与诊断

一致性仅相对首个配置，不能据此认定模型决策正确。

真tail8计量（seconds已包含在prefill_seconds内）：

| 配置/格 | calls P50 | tail8秒 P50 |
|---|---:|---:|
| combined-base/s64-d32 | 0.0000 | 0.0000 |
| combined-base/s64-d64 | 0.0000 | 0.0000 |
| combined-base/s128-d32 | 0.0000 | 0.0000 |
| combined-base/s128-d64 | 0.0000 | 0.0000 |

### combined-base

固定反馈完全一致：24/24；自然输出完全一致：12/12。

| case | 输出 | Top1 | 与基线token一致 |
|---|---|---|---|
| s64-tired-greeting | 点头回应。 | 点头一次 | True |
| s64-reading-cancel | 保持静止。 | 底盘保持静止 | True |
| s64-wave | 保持静止。 | 底盘保持静止 | True |
| s64-nod | 点头回应。 | 点头一次 | True |
| s64-bend-not-look-up | 保持静止。 | 底盘保持静止 | True |
| s64-cancel-motion | 保持静止。 | 底盘保持静止 | True |
| s128-tired-greeting | 点头回应 | 点头一次 | True |
| s128-reading-cancel | 保持静止。 | 底盘保持静止 | True |
| s128-wave | 摆臂 | 左低位前后摆臂一次 | True |
| s128-nod | 点头回应。 | 点头一次 | True |
| s128-bend-not-look-up | 保持静止。 | 底盘保持静止 | True |
| s128-cancel-motion | 保持静止。 | 底盘保持静止 | True |

标点cost fixture自然解码（12条，非独立质量样本）：

| case | 输出 | token数含EOS | tail8调用 | 与基线token一致 |
|---|---|---:|---:|---|
| s64-d32-0 | 点头回应 | 3 | 0 | True |
| s64-d32-1 | 保持静止。 | 5 | 0 | True |
| s64-d32-2 | 挥手致意。 | 5 | 0 | True |
| s64-d64-0 | 点头回应 | 3 | 0 | True |
| s64-d64-1 | 保持静止。 | 5 | 0 | True |
| s64-d64-2 | 挥手致意。 | 5 | 0 | True |
| s128-d32-0 | 点头回应 | 3 | 0 | True |
| s128-d32-1 | 保持静止。 | 5 | 0 | True |
| s128-d32-2 | 挥手。 | 3 | 0 | True |
| s128-d64-0 | 点头回应 | 3 | 0 | True |
| s128-d64-1 | 保持静止。 | 5 | 0 | True |
| s128-d64-2 | 挥手致意 | 4 | 0 | True |
- s64-d32-0：存在=True，finite=True，SHA匹配=True，独立对照=基线
- s64-d64-0：存在=True，finite=True，SHA匹配=True，独立对照=基线
- s128-d32-0：存在=True，finite=True，SHA匹配=True，独立对照=基线
- s128-d64-0：存在=True，finite=True，SHA匹配=True，独立对照=基线
- s64-d64-2：存在=True，finite=True，SHA匹配=True，独立对照=基线

阶段profile：合批prefill未内部拆分；npu_run是SDK运行调用墙钟时间，不是净芯片算力时间。

| phase | calls | NPU调用wall秒 | 搬运/KV host秒 | embedding/sampling host秒 |
|---|---:|---:|---:|---:|

资源：CPU均值7.1355%，最高温度49.9100°C，PSS峰值584.9424MiB，最低MemAvailable3348.6641MiB；swap in/out=0/0页。
NPU driver常驻峰值：666.4093 MiB（单列，不能未经核对直接与宿主PSS相加）；closed=[{'event': 'closed', 'driver_bytes': 0}]。
性能阶段资源（排除载入/quality/diagnostic）：CPU均值6.9402%，最高温度48.9180°C，宿主PSS峰值584.0869MiB。
性能阶段load1 min/P50/max：0.9700/2.3300/2.8800。

| CPU | 频率min kHz | P50 kHz | max kHz |
|---|---:|---:|---:|
| cpu0 | 416000.0000 | 780000.0000 | 1794000.0000 |
| cpu1 | 416000.0000 | 780000.0000 | 1794000.0000 |
| cpu2 | 416000.0000 | 780000.0000 | 1794000.0000 |
| cpu3 | 416000.0000 | 780000.0000 | 1794000.0000 |
| cpu4 | 416000.0000 | 780000.0000 | 1794000.0000 |
| cpu5 | 416000.0000 | 780000.0000 | 1794000.0000 |
| cpu6 | 2002000.0000 | 2002000.0000 | 2002000.0000 |
| cpu7 | 2002000.0000 | 2002000.0000 | 2002000.0000 |

真tail8计量（seconds已包含在prefill_seconds内）：

| 配置/格 | calls P50 | tail8秒 P50 |
|---|---:|---:|
| combined-tail8/s64-d32 | 1.0000 | 0.1004 |
| combined-tail8/s64-d64 | 1.0000 | 0.1012 |
| combined-tail8/s128-d32 | 1.0000 | 0.1005 |
| combined-tail8/s128-d64 | 1.0000 | 0.1093 |

### combined-tail8

固定反馈完全一致：24/24；自然输出完全一致：12/12。

| case | 输出 | Top1 | 与基线token一致 |
|---|---|---|---|
| s64-tired-greeting | 点头回应。 | 点头一次 | True |
| s64-reading-cancel | 保持静止。 | 底盘保持静止 | True |
| s64-wave | 保持静止。 | 底盘保持静止 | True |
| s64-nod | 点头回应。 | 点头一次 | True |
| s64-bend-not-look-up | 保持静止。 | 底盘保持静止 | True |
| s64-cancel-motion | 保持静止。 | 底盘保持静止 | True |
| s128-tired-greeting | 点头回应 | 点头一次 | True |
| s128-reading-cancel | 保持静止。 | 底盘保持静止 | True |
| s128-wave | 摆臂 | 左低位前后摆臂一次 | True |
| s128-nod | 点头回应。 | 点头一次 | True |
| s128-bend-not-look-up | 保持静止。 | 底盘保持静止 | True |
| s128-cancel-motion | 保持静止。 | 底盘保持静止 | True |

标点cost fixture自然解码（12条，非独立质量样本）：

| case | 输出 | token数含EOS | tail8调用 | 与基线token一致 |
|---|---|---:|---:|---|
| s64-d32-0 | 点头回应 | 3 | 1 | True |
| s64-d32-1 | 保持静止。 | 5 | 1 | True |
| s64-d32-2 | 挥手致意。 | 5 | 1 | True |
| s64-d64-0 | 点头回应 | 3 | 1 | True |
| s64-d64-1 | 保持静止。 | 5 | 1 | True |
| s64-d64-2 | 挥手致意。 | 5 | 1 | True |
| s128-d32-0 | 点头回应 | 3 | 1 | True |
| s128-d32-1 | 保持静止。 | 5 | 1 | True |
| s128-d32-2 | 挥手。 | 3 | 1 | True |
| s128-d64-0 | 点头回应 | 3 | 1 | True |
| s128-d64-1 | 保持静止。 | 5 | 1 | True |
| s128-d64-2 | 挥手致意 | 4 | 1 | True |
- s64-d32-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}
- s64-d64-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}
- s128-d32-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}
- s128-d64-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}
- s64-d64-2：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}

阶段profile：合批prefill未内部拆分；npu_run是SDK运行调用墙钟时间，不是净芯片算力时间。

| phase | calls | NPU调用wall秒 | 搬运/KV host秒 | embedding/sampling host秒 |
|---|---:|---:|---:|---:|

资源：CPU均值7.3293%，最高温度49.9100°C，PSS峰值591.9268MiB，最低MemAvailable3065.1133MiB；swap in/out=0/1708页。
NPU driver常驻峰值：942.3537 MiB（单列，不能未经核对直接与宿主PSS相加）；closed=[{'event': 'closed', 'driver_bytes': 0}]。
性能阶段资源（排除载入/quality/diagnostic）：CPU均值7.1087%，最高温度49.9100°C，宿主PSS峰值588.2119MiB。
性能阶段load1 min/P50/max：3.3900/3.5000/3.6100。

| CPU | 频率min kHz | P50 kHz | max kHz |
|---|---:|---:|---:|
| cpu0 | 416000.0000 | 1287000.0000 | 1794000.0000 |
| cpu1 | 416000.0000 | 1287000.0000 | 1794000.0000 |
| cpu2 | 416000.0000 | 1287000.0000 | 1794000.0000 |
| cpu3 | 416000.0000 | 1287000.0000 | 1794000.0000 |
| cpu4 | 416000.0000 | 1287000.0000 | 1794000.0000 |
| cpu5 | 416000.0000 | 1287000.0000 | 1794000.0000 |
| cpu6 | 2002000.0000 | 2002000.0000 | 2002000.0000 |
| cpu7 | 2002000.0000 | 2002000.0000 | 2002000.0000 |

真tail8计量（seconds已包含在prefill_seconds内）：

| 配置/格 | calls P50 | tail8秒 P50 |
|---|---:|---:|
| combined-repeat/s64-d32 | 0.0000 | 0.0000 |
| combined-repeat/s64-d64 | 0.0000 | 0.0000 |
| combined-repeat/s128-d32 | 0.0000 | 0.0000 |
| combined-repeat/s128-d64 | 0.0000 | 0.0000 |

### combined-repeat

固定反馈完全一致：24/24；自然输出完全一致：12/12。

| case | 输出 | Top1 | 与基线token一致 |
|---|---|---|---|
| s64-tired-greeting | 点头回应。 | 点头一次 | True |
| s64-reading-cancel | 保持静止。 | 底盘保持静止 | True |
| s64-wave | 保持静止。 | 底盘保持静止 | True |
| s64-nod | 点头回应。 | 点头一次 | True |
| s64-bend-not-look-up | 保持静止。 | 底盘保持静止 | True |
| s64-cancel-motion | 保持静止。 | 底盘保持静止 | True |
| s128-tired-greeting | 点头回应 | 点头一次 | True |
| s128-reading-cancel | 保持静止。 | 底盘保持静止 | True |
| s128-wave | 摆臂 | 左低位前后摆臂一次 | True |
| s128-nod | 点头回应。 | 点头一次 | True |
| s128-bend-not-look-up | 保持静止。 | 底盘保持静止 | True |
| s128-cancel-motion | 保持静止。 | 底盘保持静止 | True |

标点cost fixture自然解码（12条，非独立质量样本）：

| case | 输出 | token数含EOS | tail8调用 | 与基线token一致 |
|---|---|---:|---:|---|
| s64-d32-0 | 点头回应 | 3 | 0 | True |
| s64-d32-1 | 保持静止。 | 5 | 0 | True |
| s64-d32-2 | 挥手致意。 | 5 | 0 | True |
| s64-d64-0 | 点头回应 | 3 | 0 | True |
| s64-d64-1 | 保持静止。 | 5 | 0 | True |
| s64-d64-2 | 挥手致意。 | 5 | 0 | True |
| s128-d32-0 | 点头回应 | 3 | 0 | True |
| s128-d32-1 | 保持静止。 | 5 | 0 | True |
| s128-d32-2 | 挥手。 | 3 | 0 | True |
| s128-d64-0 | 点头回应 | 3 | 0 | True |
| s128-d64-1 | 保持静止。 | 5 | 0 | True |
| s128-d64-2 | 挥手致意 | 4 | 0 | True |
- s64-d32-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}
- s64-d64-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}
- s128-d32-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}
- s128-d64-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}
- s64-d64-2：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}

阶段profile：合批prefill未内部拆分；npu_run是SDK运行调用墙钟时间，不是净芯片算力时间。

| phase | calls | NPU调用wall秒 | 搬运/KV host秒 | embedding/sampling host秒 |
|---|---:|---:|---:|---:|

资源：CPU均值7.0820%，最高温度50.2820°C，PSS峰值589.7588MiB，最低MemAvailable3349.3438MiB；swap in/out=0/0页。
NPU driver常驻峰值：666.4093 MiB（单列，不能未经核对直接与宿主PSS相加）；closed=[{'event': 'closed', 'driver_bytes': 0}]。
性能阶段资源（排除载入/quality/diagnostic）：CPU均值6.9750%，最高温度50.2820°C，宿主PSS峰值588.2588MiB。
性能阶段load1 min/P50/max：3.6800/3.9300/4.1100。

| CPU | 频率min kHz | P50 kHz | max kHz |
|---|---:|---:|---:|
| cpu0 | 416000.0000 | 1794000.0000 | 1794000.0000 |
| cpu1 | 416000.0000 | 1794000.0000 | 1794000.0000 |
| cpu2 | 416000.0000 | 1794000.0000 | 1794000.0000 |
| cpu3 | 416000.0000 | 1794000.0000 | 1794000.0000 |
| cpu4 | 416000.0000 | 1794000.0000 | 1794000.0000 |
| cpu5 | 416000.0000 | 1794000.0000 | 1794000.0000 |
| cpu6 | 2002000.0000 | 2002000.0000 | 2002000.0000 |
| cpu7 | 2002000.0000 | 2002000.0000 | 2002000.0000 |

## 限制与审计

- CPU按全8核容量归一化；PSS为harness+worker，不代表全板所有服务。
- 宿主Python跨variant复用，诊断数组与allocator高水位可能保留；每phase细分在JSON，PSS不能据差额推导独立模型footprint。
- 最大prompt200；32输出需要231个KV位置。capacity240最多41个输出，capacity256最多57个输出；不得把本矩阵直接改64输出。
- 音频由Frontend以真实时间播放预录WAV；RAG性能阶段仅常驻，质量阶段才查询。
- 无视觉、真实麦克风、电机；固定反馈不代表真实对话质量。
- SDK NPU查询仅诊断集合，查询时间不得纳入速度比较。
- diagnostic关闭时profile logit_dump的微秒值是空分支计时开销；实际目录检查确保性能阶段没有写词表。
- 结构、有限值与释放检查无异常。
