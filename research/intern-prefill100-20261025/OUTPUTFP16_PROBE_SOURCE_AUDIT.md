# Output-FP16 四路单块对照源码审计

最终结论：当前冻结配置与实际文件匹配，两个防漂移硬断言均已补齐并复核；允许在父Conv完成且空间/双锁条件成立后进行这个有界单块实验。审计未调用板卡、SDK或模型，只读取WSL文件。当前Conv父控制器仍running，本对照不得同时启动。

顺序为原图34f3b3a2… → 旧range16-I16 b5014320… → 新outputFP16 a10a3aa5… → 原图34f3b3a2…。同C512、M32、L2起两层，同probe eed89e71…，3暖机+30计时，共120个有效API样本（尚未实际执行）。四路输入均为同一tmpfs文件、SHA5063e49e…，2170880字节，11输入；参考5输出180224字节。history128来源为FP32派生参考，并非真实NPU捕获历史或pos0异常输入。

新旧I16的生成C/payload/网格关系沿用刚完成的独立静态审计23e893eb…，此次没有重复全常量解析。重新核对新NB/旧NB/源NB、manifest/audit绑定、16外部端口记录：三者外部JSON完全一致。旧I16输入fl12/fl11与权重I16不变；新版本只将两处down输出从I16改FP16并去掉两处后置DATACONVERT。相对原图仍有两处权重量化改变，绝非原始FP16精度；原pos0输入范围风险依然存在。

board_probe_suite.py未变（28efcf56…）：每路预校SHA，清除OPI_环境后仅设置warmup；NPD/IO profiling关闭，resident_copies0；30个API样本必须finite且>0，重复输出内部一致，release0，输出字节数匹配。每路子进程180秒、整任务900秒，65°C预冷/78°C中止，采样线程异常失败，不把转换/加载纳入API时间。实际五输出对参考误差/finite还需回收后的离线审计，不能把重复一致当准确。

控制器维持hostlock与flock，独占新目录，上传.part后rename并校验SHA；磁盘预留固定1GiB+100MiB（1178599424字节），tmpfs另查空间与MemAvailable，输入不计磁盘上传量。只有returncode0且meta.complete后才核SHA删除tmpfs输入；失败保存证据/partial并保留输入，不重启。执行前父Conv必须complete/rc0/meta.complete；dry不访问板卡。

此前发现的两项防漂移缺口（已修复）：

1. 把candidate manifest的NB SHA/尺寸显式等于独立静态审计中的candidate NB字段；同时对旧range16 manifest的geometry、sourceNB、输入/参考SHA和actual_ports_exact加断言。当前实际数据均正确，但原控制器只校manifest自报SHA；最终版本已绑定新NB/尺寸到独立审计，并钉住旧range16 b501…的geometry/source/input/ref/actualports及静态审计。
2. cfg写入源码SHA后、获取锁和上传之间，需再次将runner/controller及关键manifest比对cfg冻结SHA。原代码会在拿锁后新建frozen_sources，能够阻止上传中改动，但若runner在cfg生成后、snapshot前改变，可能接受了与cfg来源记录不同的runner。最终verify_frozen在拿锁前、拿锁后、上传前核对全部来源，上传仍逐文件前后及远端校验。

独立机器证据在OUTPUTFP16_PROBE_SOURCE_AUDIT.json。补丁后已重新冻结审计SHA。即使通过源码门，也只允许这个单块筛选实验；不代表全模型TTFT、语义质量、真实语音并发或长稳通过。空间不足应先核对已备份副本后清理，不能降低保留阈值。


## 新分析器复核

analyze_down_outputfp16.py四路顺序、各30计时/3暖机、metrics与events相等、scope=two_layer_probe、capacity512/query32/fusion2、NPD/IO排除、四次release0均有断言。每份真实180224字节f32与事件SHA匹配，按hidden28672、layer2K4096、layer3K4096、layer2V4096、layer3V4096切片；五端口分别检查finite、字节相等、relL2、maxabs。分母是被比较参考的L2范数，原图前后、候选/原图、旧I16/原图、候选/旧I16均独立保存。

离线纯NumPy辅助自检覆盖：边界与shape、相同输出零误差、单个KV元素扰动定位、NaN拒绝、长度不足拒绝；没有模型/SDK执行。P95使用numpy线性插值。共用的FP32参考是原图实际导出权重参考，不是重新为候选I16权重构建的oracle；误差包括新权重量化和NPU激活计算差异。

冻结controller d76b32af54462b8d6f7bc5687b4a9ff4ba79d9e16eeb97a30df00eb4500aaca9；analyzer abc2fd0535a1c039f591c6131d09efa51278a8df3f3b3dddb4f6bd33bb6d20dd；runner28efcf567d0aec9e412b96cf38409c32b8d67f8463c8a7dee0414a2d133980c9。所有实际结果仍待根节点后续显式启动/回收；本审计未启动。

最终分析器追加P95与参考定义的两个元数据字段和两句说明。独立逆向去掉这四处后准确复原旧SHA4a085f9c…，确认没有算法改变。
