# M64紧凑历史候选 r4

目标保持prefill约500ms，并改善decode。基于outlier8-c448-full-r1，保留其既有量化（该父版本相对原模型改变量化）。仅调整prefill分组与空KV槽，完整有效历史不丢弃。

- 12张past128/query64图；全局storage448。仅当system已复用、当前pos≤128、剩余新增≥64时调度；system64可处理两块，system128可处理一块，其余回退现有M32/C448，最后8行走真实M8。不缓存dynamic。
- 新增权重图在板卡TF独立实验目录，避免再次占用tmpfs；NPU常驻图内存仍额外计量，不能称为权重共享。
- 13格成本：system32/64 × dynamic32/64/128/256，另system64/128 × dynamic64/128/192/256，交集只计一次。system正文27/59/123+模板5，dynamic模板8；实际new40/72/136/200/264。static32为截断成本fixture，不作语义判断。
- 3方案A/B/A，39请求/方案，每格3次，固定反馈输出32；共117正式计时。只改OPI_COMPACT64_DIR开关，同binary/同图量化/同输入/同system构建。
- 冻结12条新中文自然指令（正文96～101 tokens，无标点补齐），每条预先定义目标和禁止动作；两种system长度。有限新样例，不是独立第三方准确率基准。18份4步完整logits诊断与性能分开。
- 语音前端常驻合成流，RAG仅驻留，无真实麦克风/视觉/电机。记录NPU、进程与整机内存不同口径；温度78°C和MemAvailable96MiB下限退出。
- build_graphs.py保留父payload与原量化表，按实际生成C验证端口；build_runtime.py执行6组ASan/UBSan，覆盖掩码回退、KV拷贝与调度边界。

候选未完成端测/质量/资源审核前不更新最优指针。run ID不复用，Windows权威代码与原始证据保留。
