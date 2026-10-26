# 完整主图组M64候选 r11

保留r6基线；主prefill图组从M32/P448换为M64/P448，仍仅一套主图组常驻，加已有compact M64/P128和tail8。不叠加第三套主图组。复用r8已经编译/局部实测的l0/l2，编译其他10块；源权重、量化不改，形状变换仍须整模logits/自然质量校验。

目标矩阵沿用system32/64/128和动态32/64/128/192/256的13格并集；system正文27/59/123+模板5，总32/64/128；动态模板8，actualnew40/72/136/200/264。只复用system，固定反馈32输出，decode计后31token。语音合成流、RAG仅常驻，无视觉/电机。

新runtime只放宽已验证端口契约下的main64选项；不放宽KV容量/布局校验。沿用原调度：最终8行走真实tail8，其余不足64行的尾部以主64的valid_rows模式计算，额外padding行不记入有效新增token。必须单列执行query行数，并验证只提交有效KV。短格可能变慢。未测完成前不得替换r6最优或默认。

Main64 retains a logical 32-token reuse boundary (main32 behavior unchanged). Otherwise static32 would round to0 and violate the requested system-only cache comparison. Cold static32 preparation uses32 valid rows in a64-row graph. This partition can change numerical outputs and must pass full diagnostics; not claimed bit-identical in advance.
