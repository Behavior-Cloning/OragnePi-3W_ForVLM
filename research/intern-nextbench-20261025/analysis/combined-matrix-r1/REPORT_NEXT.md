# Intern NPU 下一轮基准分析

只分析回收原始证据；没有启动板卡。原始文件保持不变。

状态：failed；结构/释放审计：False

输入定义：static包含system模板；动态正文32/64，另有8个模板token，实际new应为40/72。
速度仅32步固定反馈、diagnostic=False；自然质量和全词表诊断独立。

| 配置 | static/dynamic正文 | n | TTFT P50/P95秒 | decode P50 tok/s | batch prefill P50秒 | 首head P50秒 |
|---|---|---:|---:|---:|---:|---:|
| combined-base | s64-d32 | 6 | 0.5821 / 0.5829 | 15.3996 | 0.5637 | 0.0178 |
| combined-base | s64-d64 | 6 | 0.9594 / 0.9697 | 15.2737 | 0.9406 | 0.0180 |
| combined-base | s128-d32 | 6 | 0.5843 / 0.6151 | 15.2328 | 0.5646 | 0.0191 |
| combined-base | s128-d64 | 6 | 0.9810 / 1.0114 | 15.2137 | 0.9611 | 0.0191 |
| combined-tail8 | s64-d32 | 6 | 0.4943 / 0.4953 | 15.4147 | 0.4756 | 0.0177 |
| combined-tail8 | s64-d64 | 6 | 0.8698 / 0.8713 | 15.2976 | 0.8510 | 0.0178 |
| combined-tail8 | s128-d32 | 6 | 0.4963 / 0.5175 | 15.1919 | 0.4761 | 0.0190 |
| combined-tail8 | s128-d64 | 6 | 0.8907 / 0.9144 | 15.2365 | 0.8706 | 0.0191 |

P95采用nearest-rank；每格少于20次时P95通常等于样本最大值，重复同fixture不等于独立用户场景，不能当稳定尾延迟结论。

## 按同case/repeat配对比较

先逐对计算ratio，再汇总；TTFT candidate/base越小越好，decode candidate/base越大越好。mask-repeat用于顺序漂移检查，不能当新优化。

| 配置 | 格 | 配对数 | TTFT ratio P50 | decode ratio P50 |
|---|---|---:|---:|---:|
| combined-base | s64-d32 | 6 | 1.0000 | 1.0000 |
| combined-base | s64-d64 | 6 | 1.0000 | 1.0000 |
| combined-base | s128-d32 | 6 | 1.0000 | 1.0000 |
| combined-base | s128-d64 | 6 | 1.0000 | 1.0000 |
| combined-tail8 | s64-d32 | 6 | 0.8487 | 0.9966 |
| combined-tail8 | s64-d64 | 6 | 0.9055 | 1.0026 |
| combined-tail8 | s128-d32 | 6 | 0.8476 | 0.9963 |
| combined-tail8 | s128-d64 | 6 | 0.9078 | 1.0009 |

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

资源：CPU均值7.2308%，最高温度50.2820°C，PSS峰值580.9424MiB，最低MemAvailable3348.3867MiB；swap in/out=0/452页。
NPU driver常驻峰值：666.6593 MiB（单列，不能未经核对直接与宿主PSS相加）；closed=[{'event': 'closed', 'driver_bytes': 0}]。
性能阶段资源（排除载入/quality/diagnostic）：CPU均值7.0577%，最高温度49.1040°C，宿主PSS峰值580.0557MiB。
性能阶段load1 min/P50/max：1.1200/3.0000/3.2200。

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

真tail8计量（seconds已包含在prefill_seconds内）：

| 配置/格 | calls P50 | tail8秒 P50 |
|---|---:|---:|
| combined-tail8/s64-d32 | 1.0000 | 0.1010 |
| combined-tail8/s64-d64 | 1.0000 | 0.1017 |
| combined-tail8/s128-d32 | 1.0000 | 0.1017 |
| combined-tail8/s128-d64 | 1.0000 | 0.1102 |

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

资源：CPU均值7.3183%，最高温度51.5220°C，PSS峰值587.8682MiB，最低MemAvailable3047.6211MiB；swap in/out=1/2113页。
NPU driver常驻峰值：942.3537 MiB（单列，不能未经核对直接与宿主PSS相加）；closed=[]。
性能阶段资源（排除载入/quality/diagnostic）：CPU均值6.9932%，最高温度51.0880°C，宿主PSS峰值582.3643MiB。
性能阶段load1 min/P50/max：3.3600/3.4500/3.5400。

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
- runtime error
- runtime release_error
- configured/runtime variant sequence mismatch
- release unconfirmed combined-tail8
- workload meta not complete
