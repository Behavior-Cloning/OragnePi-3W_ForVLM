已完成板测，详见 ../../results/compact-gateup-a8-m64-r1/REPORT.md：不足1%速度收益、4.72%–9.39%hidden误差，不推广。以下保留试验设计。

# 当前DFP权重的M64 gate/up A8候选

目标是降低完整 Intern NPU prefill / TTFT，而本分支仅为两层速度与数值筛查，不能代替完整目标验收。

范围：layer0/1，64查询token，128有效历史token；同一canonical输入。固定system正文与聊天模板不在本两层探针中重新编码，不能称为system128+动态64的完整请求。无语音/RAG、无视觉、无电机。

两候选已编译成功，14个矩阵权重字节及量化保持不变。四个gate/up输入增加局部INT16→INT8转换：range0实际区间[-32,31.75]，range1[-16,15.875]，量化步长比原INT16分别变粗256/128倍。四个别名共用的零bias从INT64/fl19变为INT32/fl11或12，值仍为零；导出payload因此变化，不能称完整payload不变。没有使用质量测试输入选择量化范围。

历史M1/旧PC权重gate/up A8未提速且hidden误差9.42%，本分支并不推翻该结果；它检验不同的M64/current DFP工作负载。

原始试验设计：controller_compact_gateup_a8.py将按原始→range0→range1→原始顺序，每路3次预热+30次纯API计时；每路独立进程、相同输入，回收SDK端口日志、五输出及资源记录。analyze_compact_gateup_a8.py验证四路16实际端口一致、原始前后字节一致、输出finite及hidden/KV误差。性能与误差都需据实报告。

continue_prefixsafe_a8.py等待prefixsafe-i16-r2控制器complete/rc0、原PID退出、结果离线分析通过，再启动本分支；板卡访问仍由双锁保护。任何前置失败均停止续跑并保留证据，不自动重启。

静态审计：INDEPENDENT_STATIC_AUDIT.md/json。模型NB、原始权重、浮点输出和归档保留磁盘证据，不提交Git。
