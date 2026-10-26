# gate/up合并候选 r9

仅对同源输入、相同qtype/scale/rounding的FC拼接输出通道；首轮只融合L3（L2/L3块的b1），比较M32/P448与M64/P128。保留输入有效history128、不截KV；使用既有canonical L2输入。局部API吞吐与数值验证，不能记作完整模型TTFT/prefill/decode。

父路径为r6沿用的outlier8/M64r4图，原量化资格不因此恢复。源浮点权重值和量化编码参数保持，但编译布局可能改变数值；板端完整输出对比之前不宣称无损。默认和完整性能最优r6不变。
