# L2 慢点诊断：已排除外部缓存同步，定位到缺少NN计数的FP16 down路径

本文只分析C512融合双层图，不是整模型推理。全模型目标输入仍为system64/128，动态正文64/128/192/256加模板8；本次probe物理query为16或32行，前史由root冻结fixture给定。

- L2 M16：run P50 **112.0869 ms**，输入flush+输出invalidate **0.0373 ms**（0.0333% wall）。
- L2 M32：run P50 **222.0435 ms**，同步 **0.0923 ms**（0.0415% wall）。
- NPD仅诊断，各1次，无预热；没有把其API时长代替正式30次pure timing。
- 4份NPD均为真实VNPD v2/128B记录，按SDK原始8个segment reader解析，与既有A733reader逐字段一致。
- 4组IO和4组NPD的输出均与正式纯计时逐字节一致，原审计文件SHA保存在summary。

已用原参数补导出L0/L2 M16 trace，两份NBG均与板测原件SHA相同。L0 MLP down UID116有6个NN profile nodes，累计1,069,044 NN busy cycles。L2 FP16 down变为nodes232/294，两者没有NN profile记录；M16 L2全部已记录NN busy仅6.60M cycles，反而低于L0的7.85M。

**这定位了应优先检查的路径，但不能证明CPU回退，也不能给其占112ms的时间百分比。** SDK计数器为NN硬件计数；trace生产者没有target字段，新版reader支持target并不能填补缺失数据。NPD未观测到的节点可能使用其他路径，但当前证据无法区分TP/SHADER/SW等引擎。累计busy可以重叠，不能换算全图wall time。

L2已观测NN busy中，FC占34.81%，softmax占30.32%；这些仅是**已记录NN计数的构成**，不是全图耗时构成。逐节点数据与候选层名、UID歧义均保留在 `c512-m16-l2-mapped.json`，没有按层号猜测。

最小下一步是保持两处敏感down的FP16数值配置，测试独立FC→1×1 Conv等图改写，以相同输入比较纯API时间和完整输出；图侧正在准备该单块。若需要单算子原位时间，需要进一步可验证的engine profiling来源，不能用总差值冒充精确算子时间。详细开关、SDK入口、映射哈希和限制见 `../DIAGNOSTIC_PLAN.md`。

复现：

```sh
python3 research/intern-prefill100-20261025/runtime/profile-analysis/analyze.py
python3 research/intern-prefill100-20261025/runtime/profile-analysis/map_trace.py
```

`analyze.py` 为4份raw做SDK分段解析；`map_trace.py` 在NBG SHA相同后仅映射M16两图。所有board数据为root先前回收，本次分析没有访问板卡，也没有修改冻结运行时或探针。
