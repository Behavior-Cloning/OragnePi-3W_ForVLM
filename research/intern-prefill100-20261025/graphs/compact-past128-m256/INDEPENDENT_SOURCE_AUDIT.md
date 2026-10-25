# Compact past128/M256 图、输入与控制器独立审计

**通过本次有界五组探针源码与有效输入检查，无源码阻断。** 审计仅Windows/WSL文件，无板卡、SDK、锁或模型执行。

独立重建全部11段compact输入，与实际新文件逐字节一致：四路KV仅保留前128个历史位置，原128:512各384槽全部为0；两路mask保留列0:128及512:768，删掉的128:512全部为−10000。hidden、全部256query的RoPE和token IDs不变，五输出参考同一份。新输入6,946,816字节，SHA `e05e1dbb2e4a88cee933b6cc8b968d946ce9b05a9e75dd1fa89bdcb39d8227a2`。不是对旧flat输入截字节，也未删有效历史或查询。

plain图恰10处几何改动、row224图恰26处；逐项逆替换能恢复整个原source graph。每图原quantize文件、源data和实际导出payload逐字节同；生成C全部32处常量声明同。nbg_meta仅四路past输入轴2改128和两路mask轴3改384；其余输入、全部输出的shape/dtype/quant完全同。端口证据限生成C与nbg_meta，不冒充独立SDK读取NB。row224只缩短已删mask空槽后的完整key轴，行块仍保持224和原序。

控制器固定C512 plain→C512 row224→C128 plain→C128 row224→C512 plain重复，五组统一使用新probe18b29578…，各3warmup+30纯API，预期150样本。新runner仅较旧runner增加逐case capacity/query/fusion/scope断言，仍关闭profile；显式flush/invalidate在vip_run_network计时外。双锁、独立tmpfs、冻结上传SHA、1GiB+100MiB预留和额外256MiB MemAvailable、有界超时及失败回收保留，无自动重启。

新probe支持M256>C128，这里的C128仅past输入维度，不是完整KV存储上限。回收后先验证C512原图输出复现旧binary基线，再分析五组全部finite、数值、释放、漂移。有限mask−10000及图形状变化都不能保证bitwise；本fixture符合删除空槽条件不代表任意past>128也适用。完整runtime必须保留足够KV容量、按实际history选择或回退，仍须全logits/自然质量与语音RAG并发验证。此单块实验不代表完整100ms目标达标。
