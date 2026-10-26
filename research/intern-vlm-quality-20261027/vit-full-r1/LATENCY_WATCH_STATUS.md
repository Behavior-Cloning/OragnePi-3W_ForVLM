# VQA尾延迟复测结果

SSH长握手重试恢复，按锁保护独立回收原r3日志；18请求全部完成、换图缓存检查通过，关闭driver0。没有重复启动r3。此前240秒控制器超时不代表板端推理失败。

监测r1/r2因内核缺少PSI/进程io接口在首请求前退出；修正为null后r3完成。缺失指标不能当零。
system正文30+模板5=35，reuse32；动态视觉64+文字5–9，模板/边界11，new83–87，输出上限32；同6题重复3轮。视觉为预计算向量，不是全塔共驻。
TTFT中位0.30965秒、最大14.086388秒。首轮竹子14.086秒，后两轮同题约0.309秒；慢请求进程major faults为6、pswpin/out均0，无法直接归因swap。请求间系统MemAvailable从976964降至93620KiB；进程RSS小不代表系统内存足够。

恢复后发现独立系统服务opi-vlm-model.service自动运行CPU Qwen2.5-1.5B Q4，RSS1914848KiB，systemd MemoryCurrent2163376128bytes，NRestarts=5，Restart=on-failure；ExecMainStartTimestamp=2026-10-26 19:46:45 UTC。它不属于本轮启动进程，尚未停止。不能据此证明每次延迟根因，但本轮不是独占模型环境；应独占/混跑配对后归因。已向用户询问能否临时停服务并恢复，未得到答复前不停止。

| 轮次 | 问题 | TTFT秒 | 进程major faults | swap in/out页 |
|---|---|---:|---:|---|
| 0 | red-species | 0.328 | 12 | 0/0 |
| 0 | red-support | 0.322 | 8 | 0/0 |
| 0 | red-face | 0.314 | 5 | 2/0 |
| 0 | giant-species | 0.318 | 0 | 0/0 |
| 0 | giant-plant | 14.086 | 6 | 0/0 |
| 0 | giant-ear | 0.510 | 1 | 0/0 |
| 1 | red-species | 0.310 | 0 | 0/0 |
| 1 | red-support | 0.309 | 0 | 0/121 |
| 1 | red-face | 0.309 | 0 | 0/0 |
| 1 | giant-species | 0.309 | 0 | 0/0 |
| 1 | giant-plant | 0.309 | 0 | 0/0 |
| 1 | giant-ear | 0.309 | 0 | 0/0 |
| 2 | red-species | 0.308 | 1 | 0/0 |
| 2 | red-support | 0.664 | 6 | 0/0 |
| 2 | red-face | 0.366 | 1 | 0/0 |
| 2 | giant-species | 0.309 | 0 | 0/1 |
| 2 | giant-plant | 0.309 | 0 | 0/0 |
| 2 | giant-ear | 0.309 | 0 | 0/0 |
