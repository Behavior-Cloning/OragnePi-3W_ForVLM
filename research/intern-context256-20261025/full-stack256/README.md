# c256 全层扩图

`compile_full_stack.py` 默认仅做preflight，`--execute`才启动编译。当前任务先在`single-block-admission.json`确认两个单块全部5输出与cap1280逐字节一致，再开始扩图。请以`status.json`是否complete判断是否结束。

24个候选：12个双层decode与12个双层prefill16，覆盖原模型24层。不是24个单层图。每个候选自己的source JSON/data/quantize均来自匹配过既有板端NB SHA的现役融合图；不会把layer0量化参数复制给其他层。

修改仅套用已审计的capacity相关JSON参数路径（1280→256，mask总宽随query变更）。权重文件以绝对symlink指向原文件，Docker以只读mount读取；每个数组的dtype、shape、元素数及SHA在`cases/*/l*/same-source-audit.json`，源文件编译前后SHA均核对。quantize文件直接复制并验证逐字节一致。不再转换原始GGUF、不重新校准、不改变敏感层处理。

每次仅一个Docker编译。已验证的两个layer0 NB在完整graph/quantize/data相同时以symlink复用，其他22个重新编译。输出图全部以导出的C代码验证11输入、5输出顺序。硬上限新增4GiB，每项前检查剩余至少8GiB；预计新目录约1.3GiB。这里的权重引用不增加一份模型占用。

完成时`deployment-overlay.json`列出24项替换NB及SHA。由父控制器在板端新目录建立overlay，原始head `layer24.nb`到`layer27.nb`及`embeddings.f32`直接复用，不应重复复制或改写。runtime必须明确capacity256，同时保留prefill16、tail_min8、group与mask flags不变。

后续全模型对照应使用原冻结四格：静态64/128含模板，动态正文32/64+模板8，真实完整prompt104/136/168/200；所有算法同ID、同cache。输出固定32（最大占用232）与自然回答分列。禁止沿用64输出的全四格，否则最长场景超出256。需要同时核对有效cache、实际新prefill、logits/生成token一致性、每阶段耗时及driver释放；在完整结果出来前不得报告本候选已完成模型精度验证。
