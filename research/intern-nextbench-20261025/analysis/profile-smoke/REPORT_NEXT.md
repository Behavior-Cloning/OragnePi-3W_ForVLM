# Intern NPU 下一轮基准分析

只分析回收原始证据；没有启动板卡。原始文件保持不变。

状态：complete；结构/释放审计：True

输入定义：static包含system模板；动态正文32/64，另有8个模板token，实际new应为40/72。
速度仅32步固定反馈、diagnostic=False；自然质量和全词表诊断独立。

| 配置 | static/dynamic正文 | n | TTFT P50/P95秒 | decode P50 tok/s | batch prefill P50秒 | 首head P50秒 |
|---|---|---:|---:|---:|---:|---:|
| mask-baseline | s64-d32 | 3 | 0.9543 / 0.9580 | 13.6934 | 0.9282 | 0.0273 |
| mask-baseline | s64-d64 | 3 | 1.5891 / 1.5920 | 13.8371 | 1.5581 | 0.0279 |
| mask-baseline | s128-d32 | 3 | 0.9741 / 0.9804 | 13.7130 | 0.9403 | 0.0272 |
| mask-baseline | s128-d64 | 3 | 1.6084 / 1.6141 | 13.9380 | 1.5758 | 0.0273 |
| profile-stage | s64-d32 | 3 | 0.9655 / 0.9750 | 13.8421 | 0.9348 | 0.0274 |
| profile-stage | s64-d64 | 3 | 1.5889 / 1.5942 | 14.1387 | 1.5619 | 0.0265 |
| profile-stage | s128-d32 | 3 | 0.9734 / 0.9803 | 13.7503 | 0.9410 | 0.0263 |
| profile-stage | s128-d64 | 3 | 1.6071 / 1.6114 | 13.8919 | 1.5749 | 0.0270 |

P95采用nearest-rank；每格仅3次的烟测P95等于样本最大值，不能当稳定尾延迟结论。

## 质量与诊断

一致性仅相对首个配置，不能据此认定模型决策正确。

### mask-baseline

固定反馈完全一致：12/12；自然输出完全一致：12/12。

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
- s64-d32-0：存在=True，finite=True，SHA匹配=True，独立对照=基线
- s64-d64-0：存在=True，finite=True，SHA匹配=True，独立对照=基线
- s128-d32-0：存在=True，finite=True，SHA匹配=True，独立对照=基线
- s128-d64-0：存在=True，finite=True，SHA匹配=True，独立对照=基线

阶段profile：合批prefill未内部拆分；npu_run是SDK运行调用墙钟时间，不是净芯片算力时间。

| phase | calls | NPU调用wall秒 | 搬运/KV host秒 | embedding/sampling host秒 |
|---|---:|---:|---:|---:|

资源：CPU均值7.6877%，最高温度47.6780°C，PSS峰值588.4697MiB，最低MemAvailable3280.5078MiB；swap in/out=0/0页。

### profile-stage

固定反馈完全一致：12/12；自然输出完全一致：12/12。

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
- s64-d32-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}
- s64-d64-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}
- s128-d32-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}
- s128-d64-0：存在=True，finite=True，SHA匹配=True，独立对照={'relative_l2': 0.0, 'max_abs': 0.0, 'top1_matches': 4, 'steps': 4}

阶段profile：合批prefill未内部拆分；npu_run是SDK运行调用墙钟时间，不是净芯片算力时间。

| phase | calls | NPU调用wall秒 | 搬运/KV host秒 | embedding/sampling host秒 |
|---|---:|---:|---:|---:|
| scalar_prompt | 0 | 0.00000 | 0.00000 | 0.00000 |
| decode_decoder | 372 | 17.62895 | 0.48228 | 0.02057 |
| first_head | 12 | 0.18576 | 0.00253 | 0.12776 |
| decode_head | 372 | 5.75088 | 0.07021 | 2.80012 |

资源：CPU均值7.7397%，最高温度49.1040°C，PSS峰值594.2783MiB，最低MemAvailable3272.7227MiB；swap in/out=0/0页。

## 限制与审计

- CPU按全8核容量归一化；PSS为harness+worker，不代表全板所有服务。
- 音频由Frontend以真实时间播放预录WAV；RAG性能阶段仅常驻，质量阶段才查询。
- 无视觉、真实麦克风、电机；固定反馈不代表真实对话质量。
- SDK NPU查询仅诊断集合，查询时间不得纳入速度比较。
- diagnostic关闭时profile logit_dump的微秒值是空分支计时开销；实际目录检查确保性能阶段没有写词表。
- 结构、有限值与释放检查无异常。
