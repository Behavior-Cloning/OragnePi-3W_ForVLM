# Joint attention prefill screen r10

既有split KV是已验证decode优化，不能重复包装为新收益。本实验只重访当前batched prefill：L2/L3、query32/past448与query64/past128；scores、value、both三候选分别与原split同输入配对，保留有效history128、绝对RoPE和完整mask。无system文本/模板/head输出，局部API不等于完整prefill/TTFT/decode。

源权重文件与所有未删除的量化配置保持；scores只改变输出拼接方式，value/both改变归约与中间舍入，不能预称无损。两个层都变更，接口顺序须与父图一致。当前完整模型性能候选r6和默认不变，只有明确收益后才考虑完整模型验证。
