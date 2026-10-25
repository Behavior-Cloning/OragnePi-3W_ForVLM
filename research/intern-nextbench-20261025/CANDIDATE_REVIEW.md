# Context256 与真 tail8 候选独立审阅

2026-10-25；仅阅读 Windows 权威源与父控制器已回收证据，没有访问板卡或运行模型。

## 准入判断

两条路线可进入严格全栈对照。单块收益不能直接外推整模型；现有文档保留了源图量化、校准例外、全部输出比较与关闭检查，方向合理。

Context256单块：同源/校准、裁剪无效padding、五输出逐位相同及1280复测一致，prefill16 API均值17.015→6.778ms。单块history128还不是完整NPU KV生成，因此全24层必须重新核对固定反馈全部预测、4步全词表及12自然输出。各融合块必须使用自己的quantize和精度例外，尤其敏感2/3层，不能复制第0块量化配置。

真tail8：实际runtime只在最后剩余恰好8token时调用8行bank，其余批次不变。当前40/72新增token各应命中1次；prefix64/128准备阶段应不命中。尾8模块的时间已计入prefill_seconds，分析不得重复相加。主目录capacity与overlay必须完全相同：1280的tail8不能接ctx256主图。

## 最终矩阵的主要风险与控制

1. **短容量并非通用替代。** 本文最高prompt200，固定32输出执行31次反馈，最终KV231，符合256；改64输出则不符合。未来图片额外256token或更长上下文必须回到大容量图或另建配置，不能把这次text结果称完整VLM性能。
2. **量化近似可能累积。** 局部bitwise不保证其余层同样bitwise。完整自然EOS输出、fixed-feedback32预测与全词表必须分开核对，任何输出变化都不应只用余弦近似掩盖。
3. **缓存/两bank一致性。** reset、truncate、重复system缓存必须同时更新两套mask；已有主机ASan/UBSan测试是必要条件，不代替真实NPU全栈复用。特别检查切换static128→64和相同prefix但不同动态输入，避免旧suffix泄漏。
4. **容量/shape/所有权。** ready.layers/capacity、载入时shape/dtype/quant契约、关闭顺序与driver归零都需通过。新增overlay只alias KV，不共享权重；两套prefill图的驱动内存增长必须按实际ready/peak记录。
5. **实际启用而非配置声明。** 同样的binary配env无overlay时作为对照；检查true_tail8_calls=1及有效token计数，而非只看墙钟改善。配置SHA要覆盖主图、head、embedding与独立overlay，且记录symlink解析后路径。
6. **对照归因。** 首先1280基线、1280+tail8、256基线，最后才组合256+tail8；否则无法区分容量与query行数收益。每格同fixture、samefeedback、同语音负载，调度/温度接近并保留必要基线复测。
7. **诊断与速度隔离。** 词表dump和NPU SDK查询绝不进入性能样本。stage-only观测开销也需与未开启配置对照。预填充是当前TTFT主导段，decode头部采样优化不能解释几百毫秒prefill收益。
8. **板卡存储与资源。** 全套图可能扩大，原测板根文件系统约96%使用；推送前由唯一owner核对空间并复用明确同hash文件。代码不擅自清理旧基线、日志或原始权重。

## 交付工具更新

board_benchmark.py支持每variant可选expected_graph_sha256映射：key相对主graph目录，或绝对路径（用于embedding和tail8overlay），value为64位SHA256。流式4MiB读取，canonical path+size+mtime相同的文件仅hash一次，保存graph-provenance.json；只有显式列出的文件受该校验覆盖，缺省映射不能宣称整图验证。校验发生在载入前，不计入性能窗口。

quality事件区分static、system正文/模板、动态正文/模板、总prompt、actualnew/reuse、输出含EOS与去除末EOS长度。analyze_next.py聚合true_tail8_calls/seconds，并将NPU driver与宿主PSS分列。
