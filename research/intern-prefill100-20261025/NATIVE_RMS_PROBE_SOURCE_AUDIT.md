# Native RMS 有界探针源码审计

当前源码未见启动阻断；本审计没有执行控制器、调用SDK、访问板卡或锁。冻结控制器 SHA `26c7a7cde2ac9bc6265686e8bc3095a5aad95b84c49b8a58e5182da7027a1d72`。

已复核新增静态证据绑定：独立审计的NB SHA/bytes、builder SHA、候选图/量化/data/生成C/NB等文件逐SHA与当前值一致；audit本身及上传snapshot在拿锁前后和上传期间重核。原图/input/probe绑定历史基准和canonical manifest。

两种输入各原/native/原，6进程，每个3预热+30纯API，合计18预热、180计时。C512/M32、history128、L0/L1两层，5输出；原版和候选在各输入组使用相同字节。额外压力输入只放大前28672个hidden float，峰4096且全finite，其他K/V、mask、RoPE字节不变。这是合成数值压力，不是新语义样本，也没有FP32参考正确性。

runner清除OPI分析环境、固定CPU4/5、核geometry与计数、禁止NPD/IO profile、核同图repeat一致和released0；资源线程65°C冷却、78°C停止。WSL hostlock加板端flock，单进程180秒、runner600秒、观察720秒；异常保留失败并持双锁回收partial，不重启。tmpfs需容纳全部图/两个输入/脚本并留1GiB+100MiB，MemAvailable再留256MiB。所有scratch待结果验证后再由父任务清理。操作系统/交换写入仍可能发生。

输出finite/原图A-A/候选误差及实际6次释放仍须结果审计。此处没有完整模型、语音/RAG常驻或TTFT，不能据单块速度批准100ms目标。gamma实际有1个系数变化和epsilon执行路径差异已在静态审计披露，不能当无损融合。
