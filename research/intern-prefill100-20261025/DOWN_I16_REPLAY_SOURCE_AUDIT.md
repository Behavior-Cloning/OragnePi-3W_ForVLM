# 原NPU输入下两种INT16候选重放源码审计

有限局部诊断源码审查通过，无板卡/SDK/锁操作。配置34输入与上一轮完全相同（31个不同输入、2个prompt、528有效行/16padding行）。三路original/i16-old/i16-range16各34次，两个pos0再做original-after，共104进程，208计时调用+104预热。所有分支同一probe、输入与原历史输出SHA；原5口预期90112字节。

old→range16图JSON及导出payload完全相同，YAML只改变b1 down输入fl12→11和相应max/min。生成C另有派生bias fl27→26；实际INT64 bias的7168字节全零，数值仍为0，不能写成生成C仅一处变化。三图外部nbg_meta全部相同。原图→任一I16候选会改变两个down的输入/权重/输出网格，并非相对原图数值等价。

控制器继承已审的冻结源/普通文件tar/逐SHA/双锁/超时及tmpfs保留阈值；runner未改。异常保留scratch待人工锁内回收，不自动重启。

结果需要全5口与历史原图校验，随后按pos0、其余system、dynamic三类有效行分别比较；K/V按[2,16,64]的query轴截valid。全部给定原图上游hidden/KV，因此只是teacher-forced两层局部误差，不含候选在完整网络中自传播误差。该诊断不能批准INT16模型部署，也不能代替全模质量、语音并发或100ms目标。
