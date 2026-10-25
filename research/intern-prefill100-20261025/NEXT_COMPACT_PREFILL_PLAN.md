# 下一实验：只去掉已mask的past空槽

本计划中的 C128 单块图、探针及两轮板端验证已完成。整模接入未实施，100ms完整目标未达到。详见 results/compact-past128-m256-abcd-a-r1/REPORT.md 和 results/compact-past128-screen-aba-r1/REPORT.md。

新增证据：L0/M64、M128、M256 compact纯两层API约13.91、37.41、83.43ms；L2/M256仍1760.63ms。M128前后基线漂移6.85%，不能只选慢基线。所有候选有数值变化，尚无完整模型质量验收。下一步先分析慢层down输入异常通道，验证精度与快路径的可能性；暂不批量编译整模大query图。graphs/down-outlier-screen为离线筛查，非板端性能或质量结果。

以下保留原实现要求与整模尚未完成项：

证据：runtime/profile-analysis/empty-past-slots-audit.json；canonical M256/history128 四路past K/V 128:512全0，两路mask这些列全−10000。当前softmax宽度512+256=768，可候选为128+256=384，保留全部有效历史与所有256查询。

实现要求：
1. 新建独立C128/M256图。基于原C512图共10处几何调整：两层past_k/past_v shape[2]512→128；mask shape[3]768→384；probs_past size[3]512→128；probs_new begin[3]512→128。原量化/权重不变。若沿用row224分块，16个新增slice最后维768→384，完整softmax key轴保留。
2. 输入重排必须逐段核查：过去K/V保留前128；mask保留列[0,128)与[512,768)；hidden/RoPE完全相同；不能把输入截成首N字节。保存新输入SHA和全部11段形状、五输出参考保持同一份。
3. **不能直接用当前探针binary**：runtime/probe-source/tensor_probe.c 包含 `if(context!=512||...)return 2;`。需独立新版本明确允许128/512并核输入/输出大小，保留旧二进制和SHA，先编译审计。不要在已跑实验的源文件原地修改。
4. 完整runtime也不是可直接切图：full_model.c / batch_prefill.h当前依赖统一cache capacity，且有prefill_tokens<=capacity检查。存储cache仍需512；专用prefill图的past输入容量应独立建模，分开选择与拷贝。确保只在所有有效past<=128时选此图；prefix准备和后续history>128的块走足够大的原路径，不能丢历史。避免因256查询>128 past容量而错误拒绝或写越界。
5. 单块先同一有效输入A/B/A、finite/输出/driver检查，再扩大history64/128和多层覆盖。若数值变化，继续全logits/自然中文输出/动作质量检查。不得由短块计时外推完整TTFT。

原row224的收益：两层211→116ms，尚且高于单块100ms；与8次M32的算术约87ms比较仍有不足。新图必须同时和当前最佳M32完整路径公平比较，而不仅与更慢M256 baseline比。完整system64/128（含5模板）、动态正文64/128/192/256+8模板、new72/136/200/264，仅system KV缓存，保持相同precision/output协议与语音RAG负载。

## 后续可复用证据

单一canonical C512/M32 down额外输出观测已过原5输出逐位一致门禁，见results/down-capture-c512-m32-aba-r1。已有actual-npu-range-diagnostic保留C1280/M16原整模上游输入34窗口，应先适配同图7输出并重放这些输入；只使用c1280-original轨迹，不能混I16候选污染历史。完成实测activation范围后才决定精度分区与整模接入。
