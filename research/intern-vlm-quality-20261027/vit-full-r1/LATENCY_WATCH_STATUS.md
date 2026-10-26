# VQA尾延迟复测状态

2026-10-27：当前未得到有效18请求复测结果，不能用此次监测解释前次17.195秒TTFT。

- r1：内核无/proc/pressure/memory，首次采样失败；正常关闭模型。
- r2：内核无/proc/PID/io，首次采样失败；正常关闭模型。
- r3：上述可选字段已允许缺失；控制器240秒超时。远程是否仍运行未知，未重启任务。
- 回收日志SSH返回255；板卡ping三次0丢包，但SSH banner exchange超时。Windows/WSL正常。尚无证据确认OOM、swap或NPU异常，不能归因。
- 未降低保护阈值，未重启板卡、未覆盖默认模型。

待恢复后：首先确认原进程和锁状态，再回收r3日志；不要直接重跑。如果仍运行应先查明并收尾。复测记录包括每次请求前后进程stat（minor/major faults）、/proc/vmstat换页计数、MemAvailable、温度；缺失PSI和进程I/O记录为null。

已准备endpoint_probe.c和controller_endpoints.py，测试stem与projector的真实板端时间及数值；程序已交叉编译成功，但尚未部署测试。不能算完整视觉塔已完成。

System30+5=35，reuse32；动态视觉64+文字5–9，额外模板11，new83–87；每轮6题，计划3轮18请求。仍是预计算向量NPU语言测试，不是视觉/语言共驻端到端。
