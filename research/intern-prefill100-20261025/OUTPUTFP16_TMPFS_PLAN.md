# 单块候选使用tmpfs暂存

父Conv整模已完成并审计，现执行独立四路单块筛选前准备。原disk controller保持不变，新controller_down_outputfp16_tmpfs.py将约95.7MB模型、probe、配置、日志和输出放到/dev/shm唯一目录；Windows F保留所有源文件及回收证据。所有候选同一暂存策略，API计时排除上传、加载和转换，不与历史磁盘冷加载比较。

板端预检发现/dev/shm tmpfs rw,nosuid,nodev，free2495737856B，MemAvailable4178022400B；此为准备快照，执行前重新检查。继续保留scratch free > 上传量+1GiB+100MiB；同时要求MemAvailable > 上传量+输入+同样保留+256MiB运行余量，未降低原保留阈值。仅input在成功回收后SHA验证删除，其余临时数据保留供核验，随后再按SHA回收。没有删除已有模型、父48图或原始证据。harness不计划在板根分区写artifact，但系统自身仍可能写日志/swap，不能把它说成整机零磁盘写入。

原FP16 → 旧输入/权重/输出I16 → 新输入/权重I16、输出FP16 → 原FP16，同C512/M32 fused层2/3、history128、同canonical输入，3warm+30 API/路。探针、权重hash、输出5端口检查、前后基线与双锁保持不变，结果分析器不变。不测试pos0异常、全模型语义、语音并发或完整TTFT。新候选仍非原始FP16精度。

目标仍是system正文59/123+模板5=64/128，动态正文64/128/192/256+模板8，新72/136/200/264，prefill与首token均≤100ms。单块通过不等于完成目标。
