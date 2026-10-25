# 官方自动保存导出：独立审计

审计通过（仅 host 导出与来源完整性，不是板端数值/速度验收）。

- 两版使用原生成 main（SHA 447f9caf…），env、11 输入 argv 顺序与官方流程一致。
- 逐个回读两版 11 个 .tensor 文本，拼接 float32 与冻结 canonical 的 2,170,880B 完全相同，SHA b4bd76f7…。
- 每个 variant 内的 graph C / 权重 payload 与先前失败尝试完全同 SHA；修复的是 harness。
- 原图新 NBG 为 24,760,720B，SHA d7adc37f…，逐字节复现既有板测图。
- BITCAST 新 NBG 为 24,751,168B，SHA d194bada4a35f2759c5de89d7d0fcb10da982f213de5245b40c35085b7c4e50d。
- 独立解析 14 权重与 14 bias：整数 dtype / raw bytes / offset 相同，实际 dequant 数值也逐元素相同。6 权重＋6 bias 的 qtype 从 DFP 改 AFFINE；每通道权重轴 1→3 与新 [1,1,in,out] 布局一致。不能把同值表述为量化元数据字节相同。

原图/BITCAST 在旧手写 NULL-query harness 均 rc139；改用官方 Create→Verify→PreProcess→RunGraph 自动保存流程后均 rc0、生成新图。这验证了可用的导出修复流程，尚不能在环境、输入安装与调用顺序等同时变化之间定位唯一崩溃根因。旧失败证据保留。

实际新 NB 端口由 runtime agent 另做 binary metadata 审计；后续仍需板端 canonical f32 数值门禁及 A-B-A 计时，不能从成功导出推断性能或全模型质量。审计员未启动模型、未接触板卡。
