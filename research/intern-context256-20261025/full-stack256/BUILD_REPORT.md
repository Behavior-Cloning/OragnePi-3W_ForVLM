# Context256 全层编译完成

12个双层decode图和12个双层prefill16图已齐全，覆盖24层。复用2个已通过单块板测的layer0图，其余22图顺序编译；整个扩图流程约333.2秒。此阶段没有额外板卡访问，完整模型性能与输出仍须由父任务验证。

`full-audit.json`独立复核24份现役源图和候选：全部JSON变化只在预先审核的容量字段；每层quantize逐字节相同；权重精确引用原始只读文件，逐项保留1128个数组的dtype/shape/SHA。源l2文件本身有symlink，审计按解析后的真实路径比较。每份NB SHA及11输入/5输出顺序经过导出C契约复核。

NB合计572,194,784 bytes。新full目录常规文件1,242,402,403 bytes，低于4GiB预算；head/embedding没有复制。完成审计时F可用26,909,884,416 bytes。

交付映射见`deployment-overlay.json`：24个replacement NB和四个保持不变的head SHA；embedding复用既有板端文件并由部署控制器登记SHA。候选capacity256，prefill16/tail8/group/mask全部保持；所有模式（含quality与diagnostic）统一max_new32，不得把最大prompt200与max_new64组合。

`conversion.json`保留原源转换的历史量化/布局记录作为追溯；实际新容量由顶层capacity256、capacity_changes、decoder.json/tensor-inputmeta及编译产物确定。历史layout_changes描述原c1280源生成过程，不能将其中旧值误读为本次图的实际容量。

单块板测准入：decode与prefill16各5个输出均与cap1280字节一致；prefill单块API均值17.015→6.778ms，decode3.622→3.369ms。该比例不能直接当作整模型TTFT或decode增益。完整候选尚未证明全层数值一致、整模型回答正确、近容量边界稳定或长期负载稳定。
