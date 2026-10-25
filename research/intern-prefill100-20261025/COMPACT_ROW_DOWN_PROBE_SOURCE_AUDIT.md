# Compact row-down 两层探针源码审计

未见启动阻断。仅只读WSL文件及写自身审计，没有执行控制器/dryrun、SDK、板卡或锁。候选静态audit、builder、完整候选文件SHA及NB字节数绑定已核；M256原版绑定先前compact导出NB。上传源在锁前冻结，拿锁/逐上传/上传后重核，并由板端sha256sum核所有传输文件。

M64和M256各原/row/原；容量128、全部有效历史128、L2/L3两层，每形状同canonical输入，6进程18预热180纯API。保留旧probe和runner，逐进程检查实际capacity/query/fusion、5输出长度、30正finite API、无NPD/IO profile、无resident副本、重复输出一致、driver0。65°C冷却、78°C中断；hostlock和板端flock，180秒单进程、600秒runner、720秒观察，异常持双锁回收partial，不重启。

tmpfs需全部暂存文件另加1GiB+100MiB空闲；MemAvailable需再加256MiB运行余量。scratch保留待raw归档审核，OS和swap写入仍可能。不能因临时文件在tmpfs就称无swap。

离线分析器按5端口分段误差、A/A原始SHA、每形状16实际日志端口、线性P95、边界和采样温度以及swap检查，统计口径合理。非finite会计数且误差置null；实验complete不应变成数值合格。随后父任务已补归档成员唯一性与六个结果标签精确顺序、len(ev)==6断言，最新离线分析器SHA `90f9eb27f891e9cbf5fa6149e6f0e593b59fd87db53a6040fff8212aa1a8efcf`已复核。此增量未更改控制器、runner、图或输入。

仅验证已准备的局部对照：M64/M256是局部批次，不是完整动态正文；有效history超过128需更大bank。row实际采用两个I16 down权重，与FP16原版非无损。无语音/RAG、全模型、动作质量或100ms完整目标结论。
