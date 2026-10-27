# 等价1×1映射：输出相同但更慢，不晋升

真实板端NPU单层实验；每配置2张参考激活，2预热+10热API调用，CPU7，baseline-before → candidate → baseline-after。保留Transformer网络语义；不是整塔、端到端、VQA或decode测试，无system/dynamic文字token。单层诊断F32读回单列，不当作部署流水线开销。

|层/配置/图|NPU API中位ms|相对FP32 L2|与基线同字节|驱动MiB|
|---|---:|---:|---|---:|
|l00-base-before-i1|46.507|0.579%|True|17.68|
|l00-base-before-i2|46.497|0.478%|True|17.68|
|l00-conv-widen-i1|50.835|0.579%|True|25.01|
|l00-conv-widen-i2|51.038|0.478%|True|25.01|
|l00-base-after-i1|46.733|0.579%|True|17.68|
|l00-base-after-i2|46.721|0.478%|True|17.68|

先前A16DFP+W8PC卷积在SDK拒绝，I16DFP→I16affine转换也未注册。此版将原W8整数码用I16容器存储，原逐通道scale不变；6个FC的整数码范围[-127,127]、与原W8逐元素一致，reshape为1×1卷积。不是改变LLM/ViT架构，也没有新增模型知识。权重文件的实数值不变；FP32数学等价测试误差约1e-7。

SDK编译通过且所测单层两图输出均与原版逐字节相同，但NPU API约50.84/51.04ms，比原版46.5–46.7ms慢约9%；驱动约17.68→25.01MiB。无收益，不扩展24层、不晋升默认。局部输出一致不能代替全塔VQA认证。

全部输出finite、重复稳定、每进程关闭driver0。SHA重新从回收f32计算。FP32参考仅用于误差诊断，实际测速与执行都在NPU；不把FP32主机诊断当NPU推理速度。证据：compile-status.json、config.json、results/summary.json、AUDIT.json；权重/NBG/f32保留Windows磁盘，Git只保存来源、脚本、摘要和哈希。
