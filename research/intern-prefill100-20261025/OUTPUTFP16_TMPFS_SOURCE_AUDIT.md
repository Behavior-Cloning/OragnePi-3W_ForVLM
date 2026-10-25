# Output-FP16 tmpfs 临时存储改动审计

结论：通过有界单块实验源码门。原磁盘控制器未改；本审计没有板卡或SDK访问，没有启动控制器。最终新控制器SHA `59d2936d6f22ffdddf3433d9fdf2ab3865eb4cfcad785d2b6e420578c314043d`；新dryconfig SHA `95430f105fcb839b3f768ddd562e34c9f9d7df1754e6d161de93913ddd3bfaf7`。

四路图SHA、顺序、canonical输入/参考、C512/M32/L2两层、相同probe、每路3暖机+30计时完全保留。所有图、binary、runner、config、日志和输出移动到唯一 `/dev/shm/opi-prefill100-probe-<run>`；输入仍是唯一独立tmpfs路径。只有显式临时产物路径不写根盘，tmpfs可能被swap换出，操作系统日志等仍可能写盘，不能声称零SD写入。

开始前在锁内检查tmpfs类型且不含noexec；源码不修改挂载权限。scratch可用空间必须大于图/代码+输入+1GiB+100MiB。初版遗漏同一tmpfs上的输入字节，本次已补；MemAvailable门槛进一步加256MiB运行余量。输入另有路径/.part不存在、2倍输入+16MiB检查。保留阈值1178599424字节没有降低。根节点给出的挂载/容量是其读取的证据，本审计未另行板卡验证。

上传前后与远端SHA校验、冻结源码检查、父Conv complete/rc0/meta门、hostlock/flock、唯一目录、900秒任务和180秒子进程超时、温度门、失败双锁回收均保留。成功回收且meta.complete才哈希删除输入；图/日志/输出tmpfs目录保留供核验后清理。失败保留部分产物并尝试回收，不自动重启。tmpfs重启或断电即失，必须及时保存证据。

原离线analyzer `abc2fd0535a1c039f591c6131d09efa51278a8df3f3b3dddb4f6bd33bb6d20dd` 按run定位WSL回收目录，使用配置中的SHA和本地来源检查，不硬编码板端/opt暂存目录；兼容新controller键和tmpfs路径，仍要求input_tmpfs_removed、4路计数、释放0、实际5输出SHA/finite。未混入NPD/IO诊断。输出32行、FP32派生历史128仍只是同一单块输入，不能外推完整TTFT、自然语义、语音并发或100ms全范围目标。存储策略变化可能影响加载/主机IO/资源，纯API计时排除了这些，不能声称端到端行为完全不变。

最终源与机器可读条目见 OUTPUTFP16_TMPFS_SOURCE_AUDIT.json。原controller仍为d76b32af…，runner仍28efcf56…，analyzer仍abc2fd05…。
