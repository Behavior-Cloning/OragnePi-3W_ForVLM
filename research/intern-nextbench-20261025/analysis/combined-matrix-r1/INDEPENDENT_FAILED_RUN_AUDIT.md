# combined-matrix-r1 失败运行独立审计

**结论：数据采集有用，但本轮不能通过稳定性或部署准入。** combined-tail8完成全部推理后，在关闭时出现原生堆损坏错误；不能因推理速度达标或事后driver=0，就忽略生命周期失败。

仅读取Windows回收证据，重新核对10份完整f32文件；未访问板卡，也未修改主报告或HANDOFF。

## 完成范围

| variant | fixed32性能 | 原自然质量 | cost自然解码 | 4步全词表 | 正常关闭 |
|---|---:|---:|---:|---:|---|
| combined-base | 24 | 12 | 12 | 5 | driver0 |
| combined-tail8 | 24 | 12 | 12 | 5 | **失败，缺closed** |
| combined-repeat | 0 | 0 | 0 | 0 | 未启动 |

预设第三组基线复测没有执行，不能使用本轮判断前后漂移。meta.status=failed，analyzer.audit_pass=false，计数与原始事件一致。

## 数值与输出

- combined-tail8相对combined-base：24/24 fixed-feedback完整预测一致、12/12原自然输出一致、12/12 cost自然输出一致。
- 两配置共10份f32均2,426,784字节，finite且实际SHA匹配日志。5对文件逐字节完全相同，relativeL2和maxabs为0；20/20诊断top1一致。
- 追加的s64-d64-2也逐字节相同，说明本轮ctx256组合没有复现前轮ctx1280+tail8该case首token改变；不能外推成全部模型/输入都数值等价。
- cost12含标点padding，不是新增heldout质量集；自然一致性仍不是动作任务正确率。

## 失败前观测性能（不得作为已合格版本发布）

| static/dynamic正文 | base TTFT P50秒 | tail8 TTFT P50/P95秒 | tail8 decode P50 tok/s |
|---|---:|---:|---:|
| s64-d32 | 0.582107 | 0.494334 / 0.495264 | 15.415 |
| s64-d64 | 0.959372 | 0.869847 / 0.871279 | 15.298 |
| s128-d32 | 0.584317 | 0.496337 / 0.517531 | 15.192 |
| s128-d64 | 0.980982 | 0.890699 / 0.914436 | 15.237 |

各格6次观测，P95等于样本最大值。32输出计31个decode间隔；计时与词表dump独立。输入正文32/64另8模板，总prefill新算40/72。

## 故障证据与解释

- combined-tail8.log末尾：`double free or corruption (out)`。
- 首次错误发生worker.close等待closed事件时：`resident worker event pipe closed`，没有正常closed记录。
- finally再次close出现`I/O operation on closed epoll object`，是清理重试遇到已关闭selector；不能将其当作原始C堆损坏根因。
- post-failure-health显示NPU当前0、alloc/free802/802，仅说明故障后的资源已被回收，不证明正常shutdown路径。
- 原debug最小ASan案例未复现这一外部进展不能清除本轮失败；仍需完整复现场景、根因修复和原配置重跑。
- 回收健康记录根分区只剩1.3GiB、98%使用；新增编译产物/诊断需由owner管理空间。

## RAG Top3限制仍成立

- 本轮“保持静止”Top1是底盘保持静止，但Top2/3是向前平移/向后平移，执行Top3全部会违背静止要求。
- “点头回应”Top2是摇头，不能视为同时可执行的相关动作；头部回正也需判断动作顺序及意图。
- “摆臂”的左右臂候选可能可组合，但仍受资源锁、次序、幅度与硬件约束，不能凭embedding分数自动全部执行。
- 目前这里只记录RAG结果，没有真实电机，所有候选仍应通过兼容性判定。

详细真实SHA见INDEPENDENT_FAILED_RUN_AUDIT.json；完整统计见summary_next.json。
