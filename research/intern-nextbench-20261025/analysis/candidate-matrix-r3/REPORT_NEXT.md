# Intern NPU 下一轮基准分析

只分析回收原始证据；没有启动板卡。原始文件保持不变。

状态：complete；结构/释放审计：True

输入定义：static包含system模板；动态正文32/64，另有8个模板token，实际new应为40/72。
速度仅32步固定反馈、diagnostic=False；自然质量和全词表诊断独立。

| 配置 | static/dynamic正文 | n | TTFT P50/P95秒 | decode P50 tok/s | batch prefill P50秒 | 首head P50秒 |
|---|---|---:|---:|---:|---:|---:|
| mask-base | s64-d32 | 6 | 0.9595 / 0.9732 | 13.6848 | 0.9314 | 0.0272 |
| mask-base | s64-d64 | 6 | 1.5953 / 1.6139 | 13.8860 | 1.5651 | 0.0270 |
| mask-base | s128-d32 | 6 | 0.9767 / 0.9820 | 13.6585 | 0.9435 | 0.0271 |
| mask-base | s128-d64 | 6 | 1.6104 / 1.6179 | 13.8366 | 1.5796 | 0.0262 |
| context256 | s64-d32 | 6 | 0.5881 / 0.6199 | 14.4257 | 0.5624 | 0.0251 |
| context256 | s64-d64 | 6 | 0.9644 / 1.0194 | 14.4133 | 0.9386 | 0.0250 |
| context256 | s128-d32 | 6 | 0.5935 / 0.6183 | 14.1962 | 0.5665 | 0.0274 |
| context256 | s128-d64 | 6 | 0.9894 / 0.9984 | 14.3224 | 0.9620 | 0.0272 |
| tail8-c1280 | s64-d32 | 6 | 0.8108 / 0.8127 | 13.6833 | 0.7804 | 0.0275 |
| tail8-c1280 | s64-d64 | 6 | 1.4513 / 1.4701 | 13.8832 | 1.4198 | 0.0273 |
| tail8-c1280 | s128-d32 | 6 | 0.8283 / 0.8305 | 13.6271 | 0.7938 | 0.0273 |
| tail8-c1280 | s128-d64 | 6 | 1.4656 / 1.4765 | 13.7938 | 1.4324 | 0.0267 |
| greedy-c1280 | s64-d32 | 6 | 0.9507 / 0.9728 | 14.6385 | 0.9296 | 0.0190 |
| greedy-c1280 | s64-d64 | 6 | 1.5833 / 1.6055 | 14.6030 | 1.5621 | 0.0190 |
| greedy-c1280 | s128-d32 | 6 | 0.9716 / 0.9733 | 14.6240 | 0.9473 | 0.0190 |
| greedy-c1280 | s128-d64 | 6 | 1.6014 / 1.6074 | 14.6381 | 1.5780 | 0.0190 |
| context256-greedy | s64-d32 | 6 | 0.5786 / 0.5811 | 15.4141 | 0.5603 | 0.0178 |
| context256-greedy | s64-d64 | 6 | 0.9544 / 1.0068 | 15.2509 | 0.9358 | 0.0179 |
| context256-greedy | s128-d32 | 6 | 0.5815 / 0.5826 | 15.2028 | 0.5615 | 0.0190 |
| context256-greedy | s128-d64 | 6 | 0.9782 / 1.0105 | 15.1856 | 0.9577 | 0.0190 |
| mask-repeat | s64-d32 | 6 | 0.9582 / 0.9821 | 13.8353 | 0.9286 | 0.0274 |
| mask-repeat | s64-d64 | 6 | 1.5931 / 1.6153 | 14.0397 | 1.5636 | 0.0280 |
| mask-repeat | s128-d32 | 6 | 0.9785 / 0.9821 | 13.7910 | 0.9471 | 0.0263 |
| mask-repeat | s128-d64 | 6 | 1.6139 / 1.6161 | 13.9283 | 1.5818 | 0.0261 |

P95采用nearest-rank；每格少于20次时P95通常等于样本最大值，重复同fixture不等于独立用户场景，不能当稳定尾延迟结论。

## 按同case/repeat配对比较

先逐对计算ratio，再汇总；TTFT candidate/base越小越好，decode candidate/base越大越好。mask-repeat用于顺序漂移检查，不能当新优化。

| 配置 | 格 | 配对数 | TTFT ratio P50 | decode ratio P50 |
|---|---|---:|---:|---:|
| mask-base | s64-d32 | 6 | 1.0000 | 1.0000 |
| mask-base | s64-d64 | 6 | 1.0000 | 1.0000 |
| mask-base | s128-d32 | 6 | 1.0000 | 1.0000 |
| mask-base | s128-d64 | 6 | 1.0000 | 1.0000 |
| context256 | s64-d32 | 6 | 0.6126 | 1.0467 |
| context256 | s64-d64 | 6 | 0.6043 | 1.0397 |
| context256 | s128-d32 | 6 | 0.6156 | 1.0391 |
| context256 | s128-d64 | 6 | 0.6150 | 1.0384 |
| tail8-c1280 | s64-d32 | 6 | 0.8444 | 0.9967 |
| tail8-c1280 | s64-d64 | 6 | 0.9085 | 0.9969 |
| tail8-c1280 | s128-d32 | 6 | 0.8485 | 0.9981 |
| tail8-c1280 | s128-d64 | 6 | 0.9145 | 0.9981 |
| greedy-c1280 | s64-d32 | 6 | 0.9883 | 1.0694 |
| greedy-c1280 | s64-d64 | 6 | 0.9935 | 1.0522 |
| greedy-c1280 | s128-d32 | 6 | 0.9953 | 1.0736 |
| greedy-c1280 | s128-d64 | 6 | 0.9941 | 1.0589 |
| context256-greedy | s64-d32 | 6 | 0.6030 | 1.1211 |
| context256-greedy | s64-d64 | 6 | 0.5996 | 1.0958 |
| context256-greedy | s128-d32 | 6 | 0.5941 | 1.1169 |
| context256-greedy | s128-d64 | 6 | 0.6058 | 1.1038 |
| mask-repeat | s64-d32 | 6 | 1.0008 | 1.0062 |
| mask-repeat | s64-d64 | 6 | 0.9998 | 1.0069 |
| mask-repeat | s128-d32 | 6 | 1.0042 | 1.0039 |
| mask-repeat | s128-d64 | 6 | 0.9998 | 1.0045 |

## 质量与诊断

一致性仅相对首个配置，不能据此认定模型决策正确。

真tail8计量（seconds已包含在prefill_seconds内）：

| 配置/格 | calls P50 | tail8秒 P50 |
|---|---:|---:|
| mask-base/s64-d32 | — | — |
| mask-base/s64-d64 | — | — |
| mask-base/s128-d32 | — | — |
| mask-base/s128-d64 | — | — |

### mask-base

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

标点cost fixture自然解码（额外4条，非独立质量样本）：

| case | 输出 | token数含EOS | tail8调用 | 与基线token一致 |
|---|---|---:|---:|---|
| s64-d32-0 | 点头回应 | 3 | None | True |
| s64-d64-0 | 点头回应 | 3 | None | True |
| s128-d32-0 | 点头回应 | 3 | None | True |
| s128-d64-0 | 点头回应 | 3 | None | True |
- s64-d32-0：存在=True，finite=True，SHA匹配=True，独立对照=基线
- s64-d64-0：存在=True，finite=True，SHA匹配=True，独立对照=基线
- s128-d32-0：存在=True，finite=True，SHA匹配=True，独立对照=基线
- s128-d64-0：存在=True，finite=True，SHA匹配=True，独立对照=基线

阶段profile：合批prefill未内部拆分；npu_run是SDK运行调用墙钟时间，不是净芯片算力时间。

| phase | calls | NPU调用wall秒 | 搬运/KV host秒 | embedding/sampling host秒 |
|---|---:|---:|---:|---:|

资源：CPU均值7.7816%，最高温度50.5300°C，PSS峰值582.5898MiB，最低MemAvailable3283.0234MiB；swap in/out=0/223页。
NPU driver常驻峰值：718.3787 MiB（单列，不能未经核对直接与宿主PSS相加）；closed=[{'event': 'closed', 'driver_bytes': 0}]。
性能阶段资源（排除载入/quality/diagnostic）：CPU均值7.5027%，最高温度49.6620°C，宿主PSS峰值581.8945MiB。
性能阶段load1 min/P50/max：1.2600/2.8200/3.1200。

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
| context256/s64-d32 | — | — |
| context256/s64-d64 | — | — |
| context256/s128-d32 | — | — |
| context256/s128-d64 | — | — |

### context256

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

标点cost fixture自然解码（额外4条，非独立质量样本）：

| case | 输出 | token数含EOS | tail8调用 | 与基线token一致 |
|---|---|---:|---:|---|
| s64-d32-0 | 点头回应 | 3 | None | True |
| s64-d64-0 | 点头回应 | 3 | None | True |
| s128-d32-0 | 点头回应 | 3 | None | True |
| s128-d64-0 | 点头回应 | 3 | None | True |
- s64-d32-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.030923186886969743, 'max_abs': 0.546875, 'top1_matches': 4, 'steps': 4}
- s64-d64-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.02925754370115933, 'max_abs': 0.54296875, 'top1_matches': 4, 'steps': 4}
- s128-d32-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.03261376990896858, 'max_abs': 0.55078125, 'top1_matches': 4, 'steps': 4}
- s128-d64-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0294437222264179, 'max_abs': 0.48828125, 'top1_matches': 4, 'steps': 4}

阶段profile：合批prefill未内部拆分；npu_run是SDK运行调用墙钟时间，不是净芯片算力时间。

| phase | calls | NPU调用wall秒 | 搬运/KV host秒 | embedding/sampling host秒 |
|---|---:|---:|---:|---:|

资源：CPU均值7.4931%，最高温度52.4520°C，PSS峰值573.8281MiB，最低MemAvailable3341.8594MiB；swap in/out=0/52页。
NPU driver常驻峰值：666.4093 MiB（单列，不能未经核对直接与宿主PSS相加）；closed=[{'event': 'closed', 'driver_bytes': 0}]。
性能阶段资源（排除载入/quality/diagnostic）：CPU均值7.2341%，最高温度51.3360°C，宿主PSS峰值569.9297MiB。
性能阶段load1 min/P50/max：3.1800/3.5200/3.6200。

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
| tail8-c1280/s64-d32 | 1.0000 | 0.1662 |
| tail8-c1280/s64-d64 | 1.0000 | 0.1686 |
| tail8-c1280/s128-d32 | 1.0000 | 0.1666 |
| tail8-c1280/s128-d64 | 1.0000 | 0.1688 |

### tail8-c1280

固定反馈完全一致：22/24；自然输出完全一致：12/12。

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

标点cost fixture自然解码（额外4条，非独立质量样本）：

| case | 输出 | token数含EOS | tail8调用 | 与基线token一致 |
|---|---|---:|---:|---|
| s64-d32-0 | 点头回应 | 3 | 1 | True |
| s64-d64-0 | 点头回应 | 3 | 1 | True |
| s128-d32-0 | 点头回应 | 3 | 1 | True |
| s128-d64-0 | 点头回应 | 3 | 1 | True |
- s64-d32-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.02639864102815006, 'max_abs': 0.453125, 'top1_matches': 4, 'steps': 4}
- s64-d64-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.02643529885109314, 'max_abs': 0.408203125, 'top1_matches': 4, 'steps': 4}
- s128-d32-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.030524630654413742, 'max_abs': 0.5859375, 'top1_matches': 4, 'steps': 4}
- s128-d64-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0270190912101052, 'max_abs': 0.3984375, 'top1_matches': 4, 'steps': 4}

阶段profile：合批prefill未内部拆分；npu_run是SDK运行调用墙钟时间，不是净芯片算力时间。

| phase | calls | NPU调用wall秒 | 搬运/KV host秒 | embedding/sampling host秒 |
|---|---:|---:|---:|---:|

资源：CPU均值7.7792%，最高温度53.3200°C，PSS峰值595.5146MiB，最低MemAvailable2951.5078MiB；swap in/out=0/736页。
NPU driver常驻峰值：1019.7486 MiB（单列，不能未经核对直接与宿主PSS相加）；closed=[{'event': 'closed', 'driver_bytes': 0}]。
性能阶段资源（排除载入/quality/diagnostic）：CPU均值7.4042%，最高温度53.3200°C，宿主PSS峰值595.2021MiB。
性能阶段load1 min/P50/max：3.4800/3.7400/4.0000。

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
| greedy-c1280/s64-d32 | — | — |
| greedy-c1280/s64-d64 | — | — |
| greedy-c1280/s128-d32 | — | — |
| greedy-c1280/s128-d64 | — | — |

### greedy-c1280

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

标点cost fixture自然解码（额外4条，非独立质量样本）：

| case | 输出 | token数含EOS | tail8调用 | 与基线token一致 |
|---|---|---:|---:|---|
| s64-d32-0 | 点头回应 | 3 | None | True |
| s64-d64-0 | 点头回应 | 3 | None | True |
| s128-d32-0 | 点头回应 | 3 | None | True |
| s128-d64-0 | 点头回应 | 3 | None | True |
- s64-d32-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}
- s64-d64-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}
- s128-d32-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}
- s128-d64-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}

阶段profile：合批prefill未内部拆分；npu_run是SDK运行调用墙钟时间，不是净芯片算力时间。

| phase | calls | NPU调用wall秒 | 搬运/KV host秒 | embedding/sampling host秒 |
|---|---:|---:|---:|---:|

资源：CPU均值7.4806%，最高温度52.8860°C，PSS峰值587.7803MiB，最低MemAvailable3265.9922MiB；swap in/out=0/0页。
NPU driver常驻峰值：718.3787 MiB（单列，不能未经核对直接与宿主PSS相加）；closed=[{'event': 'closed', 'driver_bytes': 0}]。
性能阶段资源（排除载入/quality/diagnostic）：CPU均值7.1961%，最高温度52.8860°C，宿主PSS峰值587.5225MiB。
性能阶段load1 min/P50/max：3.4000/3.5500/3.8900。

| CPU | 频率min kHz | P50 kHz | max kHz |
|---|---:|---:|---:|
| cpu0 | 416000.0000 | 416000.0000 | 1794000.0000 |
| cpu1 | 416000.0000 | 416000.0000 | 1794000.0000 |
| cpu2 | 416000.0000 | 416000.0000 | 1794000.0000 |
| cpu3 | 416000.0000 | 416000.0000 | 1794000.0000 |
| cpu4 | 416000.0000 | 416000.0000 | 1794000.0000 |
| cpu5 | 416000.0000 | 416000.0000 | 1794000.0000 |
| cpu6 | 2002000.0000 | 2002000.0000 | 2002000.0000 |
| cpu7 | 2002000.0000 | 2002000.0000 | 2002000.0000 |

真tail8计量（seconds已包含在prefill_seconds内）：

| 配置/格 | calls P50 | tail8秒 P50 |
|---|---:|---:|
| context256-greedy/s64-d32 | — | — |
| context256-greedy/s64-d64 | — | — |
| context256-greedy/s128-d32 | — | — |
| context256-greedy/s128-d64 | — | — |

### context256-greedy

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

标点cost fixture自然解码（额外4条，非独立质量样本）：

| case | 输出 | token数含EOS | tail8调用 | 与基线token一致 |
|---|---|---:|---:|---|
| s64-d32-0 | 点头回应 | 3 | None | True |
| s64-d64-0 | 点头回应 | 3 | None | True |
| s128-d32-0 | 点头回应 | 3 | None | True |
| s128-d64-0 | 点头回应 | 3 | None | True |
- s64-d32-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.030923186886969743, 'max_abs': 0.546875, 'top1_matches': 4, 'steps': 4}
- s64-d64-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.02925754370115933, 'max_abs': 0.54296875, 'top1_matches': 4, 'steps': 4}
- s128-d32-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.03261376990896858, 'max_abs': 0.55078125, 'top1_matches': 4, 'steps': 4}
- s128-d64-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0294437222264179, 'max_abs': 0.48828125, 'top1_matches': 4, 'steps': 4}

阶段profile：合批prefill未内部拆分；npu_run是SDK运行调用墙钟时间，不是净芯片算力时间。

| phase | calls | NPU调用wall秒 | 搬运/KV host秒 | embedding/sampling host秒 |
|---|---:|---:|---:|---:|

资源：CPU均值7.1926%，最高温度53.0720°C，PSS峰值575.2295MiB，最低MemAvailable3316.5117MiB；swap in/out=0/2页。
NPU driver常驻峰值：666.4093 MiB（单列，不能未经核对直接与宿主PSS相加）；closed=[{'event': 'closed', 'driver_bytes': 0}]。
性能阶段资源（排除载入/quality/diagnostic）：CPU均值6.9090%，最高温度52.7620°C，宿主PSS峰值574.9717MiB。
性能阶段load1 min/P50/max：3.4900/3.6000/3.7700。

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
| mask-repeat/s64-d32 | — | — |
| mask-repeat/s64-d64 | — | — |
| mask-repeat/s128-d32 | — | — |
| mask-repeat/s128-d64 | — | — |

### mask-repeat

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

标点cost fixture自然解码（额外4条，非独立质量样本）：

| case | 输出 | token数含EOS | tail8调用 | 与基线token一致 |
|---|---|---:|---:|---|
| s64-d32-0 | 点头回应 | 3 | None | True |
| s64-d64-0 | 点头回应 | 3 | None | True |
| s128-d32-0 | 点头回应 | 3 | None | True |
| s128-d64-0 | 点头回应 | 3 | None | True |
- s64-d32-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}
- s64-d64-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}
- s128-d32-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}
- s128-d64-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}

阶段profile：合批prefill未内部拆分；npu_run是SDK运行调用墙钟时间，不是净芯片算力时间。

| phase | calls | NPU调用wall秒 | 搬运/KV host秒 | embedding/sampling host秒 |
|---|---:|---:|---:|---:|

资源：CPU均值7.6013%，最高温度53.0720°C，PSS峰值587.5029MiB，最低MemAvailable3265.9727MiB；swap in/out=0/0页。
NPU driver常驻峰值：718.3787 MiB（单列，不能未经核对直接与宿主PSS相加）；closed=[{'event': 'closed', 'driver_bytes': 0}]。
性能阶段资源（排除载入/quality/diagnostic）：CPU均值7.4540%，最高温度53.0720°C，宿主PSS峰值587.1904MiB。
性能阶段load1 min/P50/max：3.3800/3.4900/3.6600。

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

## 限制与审计

- CPU按全8核容量归一化；PSS为harness+worker，不代表全板所有服务。
- 宿主Python跨variant复用，诊断数组与allocator高水位可能保留；每phase细分在JSON，PSS不能据差额推导独立模型footprint。
- 最大prompt200；32输出需要231个KV位置。capacity240最多41个输出，capacity256最多57个输出；不得把本矩阵直接改64输出。
- 音频由Frontend以真实时间播放预录WAV；RAG性能阶段仅常驻，质量阶段才查询。
- 无视觉、真实麦克风、电机；固定反馈不代表真实对话质量。
- SDK NPU查询仅诊断集合，查询时间不得纳入速度比较。
- diagnostic关闭时profile logit_dump的微秒值是空分支计时开销；实际目录检查确保性能阶段没有写词表。
- 结构、有限值与释放检查无异常。
