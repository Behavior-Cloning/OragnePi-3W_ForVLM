# Softmax I16候选

只更改softmax前的加mask结果：FP16改为已有attention score的I16量化尺度。所有权重、其他quantizer和拓扑不变。mask -10000会饱和，未证无损；先L0/L1局部A/B/A，速度与有限数值通过才考虑全模型。system-only KV、动态256、prefill约500ms整体目标不变。
