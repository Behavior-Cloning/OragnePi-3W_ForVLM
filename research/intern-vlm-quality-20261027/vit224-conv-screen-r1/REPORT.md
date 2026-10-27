# 编译失败记录

SDK拒绝A16 DFP输入与W8逐通道卷积组合（CONV2D）。没有有效NBG或板端计时，不作为加速结果。对应源代码、compile-status.json和日志保留。未覆盖当前W8全连接图。
