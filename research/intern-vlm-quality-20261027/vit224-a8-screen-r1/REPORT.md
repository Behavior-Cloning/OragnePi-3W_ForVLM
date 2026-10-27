# W8激活降位筛查：淘汰

真实板端NPU单层实验；每配置2张参考激活，2预热+10热API调用，CPU7，baseline-before → candidate → baseline-after。保留Transformer网络语义；不是整塔、端到端、VQA或decode测试，无system/dynamic文字token。单层诊断F32读回单列，不当作部署流水线开销。

|层/配置/图|NPU API中位ms|相对FP32 L2|与基线同字节|驱动MiB|
|---|---:|---:|---|---:|
|l00-base-before-i1|46.389|0.579%|True|17.68|
|l00-base-before-i2|46.479|0.478%|True|17.68|
|l00-input8-i1|64.952|8.162%|False|18.41|
|l00-input8-i2|65.023|9.415%|False|18.41|
|l00-input-output8-i1|60.689|10.309%|False|14.66|
|l00-input-output8-i2|60.494|10.913%|False|14.66|
|l00-base-after-i1|46.613|0.579%|True|17.68|
|l00-base-after-i2|46.615|0.478%|True|17.68|
|l12-base-before-i1|46.678|0.966%|True|19.33|
|l12-base-before-i2|46.758|0.933%|True|19.33|
|l12-input8-i1|65.304|27.463%|False|20.06|
|l12-input8-i2|65.284|27.213%|False|20.06|
|l12-input-output8-i1|60.925|33.160%|False|16.30|
|l12-input-output8-i2|60.995|31.934%|False|16.30|
|l12-base-after-i1|46.750|0.966%|True|19.33|
|l12-base-after-i2|46.821|0.933%|True|19.33|

在层0和12将FC输入（及另一版FC输出）由I16 DFP降低为I8，缩小小数位以保留相同物理范围。权重W8、敏感norm/GELU FP16保持；未使用额外校准/训练。候选约60–65ms，基线约46–47ms，且相对FP32误差最高约33%。结果说明此直接降位方案不可用，不能推断所有W8A8训练模型都不可用。未扩展24层，也未覆盖默认。

全部输出finite、重复稳定、每进程关闭driver0。SHA重新从回收f32计算。FP32参考仅用于误差诊断，实际测速与执行都在NPU；不把FP32主机诊断当NPU推理速度。证据：compile-status.json、config.json、results/summary.json、AUDIT.json；权重/NBG/f32保留Windows磁盘，Git只保存来源、脚本、摘要和哈希。
