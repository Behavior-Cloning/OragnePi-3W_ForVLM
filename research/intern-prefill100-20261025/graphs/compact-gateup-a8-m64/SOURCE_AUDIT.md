# Compact gate/up A8 执行与分析源码审计

只读 WSL 审计通过，未发现阻止本次有界四路比较的缺陷。没有执行控制器、板卡、SDK或锁。源码 SHA 见 SOURCE_AUDIT.json。

续跑器必须看到 prefixsafe complete、returncode=0、PID退出，并成功执行离线分析后才启动 A8。A8 控制器另行获取 hostlock 和板端 flock，不会绕过正在运行的任务。它检查 PID 是否存在而非 cmdline；PID复用可能导致保守等待，但不会自动重启失联实验。等待期间任何源码修改必须重新核对本审计。

四路顺序原图/range0/range1/原图。同 compact C128/M64/L0、全部128有效历史、同 input SHA、同 probe。候选 NB 与独立静态审计一致。每路3预热+30纯API计时，共120 timed/12 warm；关闭NPD与IO profiling。输出五端口共360448字节，分析器借用的canonical输出段大小与compact元数据逐项一致。

所有上传文件有冻结 SHA，空间门禁包括上传量及1GiB+100MiB保留、内存另加256MiB，检查可执行tmpfs。温度78°C保护、65°C冷却；600秒板端上限、720秒观察上限。失败保留partial归档或 recovery_pending，无自动重启。

离线分析检查归档逐文件SHA、配置、实际几何、计数、释放0、finite和原图A/A字节一致；比较四路日志中的16个实际端口。changed_elements 使用数值比较而非位比较，会把±0当相同；独立结果审计需补原始uint32和字面端口合同。源码、固定输入、环境检查并不意味着候选质量合格。

仅一个canonical FP32前缀输入、两层块，无语音/RAG、整模或语义测试。API不含加载/转换/显式flush-invalidate；A8改变激活精度。任何单块速度结果均不能直接宣称完整TTFT或100ms目标达成。
