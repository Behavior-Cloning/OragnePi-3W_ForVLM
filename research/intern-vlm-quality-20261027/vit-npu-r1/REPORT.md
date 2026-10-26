# ViT NPU 首块部署探针

真实配套F16视觉权重，112输入经patch stem/位置插值形成64patch+CLS=65行，hidden1024。导出首个完整Transformer块，包括LayerNorm、Q/K/V、attention、GELU、MLP、layer scale与残差；不是只有矩阵乘法。224的257行ONNX也已导出并通过checker，尚未编译实板。

112首块INT16编译与板端运行完成，2次预热＋10次热API中位数11.029ms；驱动分配约21.70MB，释放0；温度40.672→39.680°C。加载、CPU输入转换、flush/invalidate、输出读取和视觉前处理不在API计时中。输入/输出66560元素，端口日志及SHA见board-results和board-audit.json。

相对Torch FP32同权重参考：relative L2=0.0235314，cosine=0.9997328，max abs=0.0806250。有限值检查通过，不能称无损或全塔精度合格。仅一图实板、两图开发校准；校准远不足最终部署验收。输出量化为INT16/DFP12，输入为F32（以端口日志为准）；当前并非宣称全图W8A8。GGML GELU按tanh近似建立此参考，尚需对官方exact-GELU与GGML端到端特征进一步核对。

保留112×112全部64个patch token，无额外token压缩。首块尚未涉及末端projector，既有112未训练adapter的VQA 0/6问题仍未解决。不能以11ms×24冒充完整塔延迟；不能与CPU整塔0.915秒直接算加速比。

后续：同输入FP16/INT16/INT8比较、224编译与实板、层级数值回归后扩展24层和projector，再联合NPU语言模型做VQA及常驻资源测试。CPU Q8慢于FP16的具体kernel原因仍未profiling确认。本实验只证明NPU首块可导出和执行。
