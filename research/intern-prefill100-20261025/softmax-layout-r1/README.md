# Softmax二维布局候选

只将softmax输入[1,2,7Q,K]展平为[14Q,K]，沿完整key轴归一化，再恢复原shape。原权重payload/量化参数保持；编译器可能改变舍入与执行计划，因此需实际NPU验证，不先称无损。

已编译两层L0/L1 M64/past128及M32/past448。初筛仅测M64的同输入3预热+30正式API A/B/A；不含加载、输入转换、语音/RAG，不是完整TTFT。M32图仅编译，未测。

只有局部有明显收益且数值可用才扩完整模型，默认不变。证据分别写status.json、controller.json和results。
