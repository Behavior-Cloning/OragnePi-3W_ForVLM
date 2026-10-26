# M64/past448两层筛查 r8

r7分解显示长输入prefill约97%位于NPU API内；本实验评估保留完整448历史容量、query64的NPU图，查验相对M32是否有吞吐优势。只编译L0/L1与敏感L2/L3，使用M64r4的权重和量化，仅past128→448、mask和切片索引随之改变；不能称无损。无全模型部署/推广。

同一canonical128有效历史，query64对照past128→past448→past128，另测M32past448参考吞吐。probe只测重复同输入API，不含完整TTFT/head/语音/RAG；不能将两层时长乘12冒充整模。所有图与结果独立run ID，默认不改。若优势不足，就保留否定结果，不添加完整常驻bank。
