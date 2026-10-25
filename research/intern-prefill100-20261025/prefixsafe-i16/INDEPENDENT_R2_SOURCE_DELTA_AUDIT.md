# Prefixsafe r2 独立增量审计

只读 WSL 源与状态，无板卡/SDK/锁操作。r2 controller 冻结 SHA 为 9256fcbb33e4aaddbe0b4a8cd953379b7403dc53c2af6ef188bef0ed08907220。本增量审计与原 r1 源审计分别保留，不能混用。

r1 TF 可用 951,275,520 字节，未过 1 GiB 门禁；异常发生在 mkdir/上传/模型加载之前，无有效推理样本。attempt-r1-disk-preflight 内的源码、状态、日志、launch、dryrun 已逐 SHA 留证。

r2 只改 controller：独立 r2 名称，源码/二进制/process.log/exit 和结果都写唯一 tmpfs 路径；检查挂载为 tmpfs 且非 noexec；设置 PYTHONDONTWRITEBYTECODE=1。runner、runtime、worker、config、输入、三模式和输出策略字节未改。双锁、冻结 SHA 校验、2400秒上限及不自动重启保留。

记录的执行前 tmpfs 可用 2,495,737,856 字节、MemAvailable 4,170,485,760 字节，超过 700 MiB/2 GiB 入口阈值。这些是执行前阈值，不是全过程持续保留量。系统日志和 swap 仍可能写 TF；显式任务产物不写根盘不等于所有物理写入为零。findmnt 返回值未在状态中保留，但 running 状态表明其断言已通过。

没有发现本次存储路径调整阻止有界试验的新缺陷。暂存文件保留至验证回收和准确清理；r2 当前仅确认运行中，数值、语义和性能必须等待 results/prefixsafe-i16-r2 审计。原图仍来自只读依赖路径，未搬移或改写权重。
