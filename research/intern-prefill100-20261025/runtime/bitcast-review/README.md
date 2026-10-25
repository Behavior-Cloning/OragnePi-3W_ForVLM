# BITCAST 后端只读核对与最小重建路线

未修改任何图或生成C、未编译、未访问板卡；只读安装SDK ELF/header和失败候选C。不能据此宣布BITCAST可运行或零开销。

安装libovxlib SHA a4e3e5ba4bd8f6c2f460177002920d66b00be2ddc87f3332fd26eaad4d15182e。`vsi_nn_op_BITCAST`确有注册：compute(0xa2340)进入kernel selector；VX backend `_bitcastsetup`(0x1b2470)直接调用公开`vxBitCastLayer(graph,input,output)`。op对象没有op_check指针，不受已观察到的DATACONVERT白名单检查；底层OpenVX verify/目标硬件仍可能拒绝。相同元素字节数时setup复制shape，但这不保证quantizer类型被后端接受。

当前失败affine候选export/vnn_decoder.c SHA 0d7a158d4a2da4ca114e5b91db1d04a4ca412105ae4507d7137c54d389389701。逐个追踪14个DATACONVERT的真实输入输出attr：全部signed INT16，输入DFP fl10/11/12/15，输出ASYM scale恰好2^-fl、zero_point0。所有65536个整数码的表示值等价，不是损失精度的再量化。证据generated-c-converters.json列node/uid/attrs；这仅证明转换边界格点相同，不证明后续Conv累加等价。

## 最小候选建议（尚未执行）

1. 在独立目录复制失败affine export的C/header/makefile和export.data，保留原SHA；不得覆盖失败证据。仅将经逐一等值断言的14行 `NEW_VXNODE(... VSI_NN_OP_DATACONVERT,1,1,uid)` 中op改为 `VSI_NN_OP_BITCAST`。不改tensor连接/shape/dtype/quantizer/数据/输出，拒绝全局无审计替换。
2. 用已有makefile.linux在同SDK重新编译该C（需删独立目录旧.o避免复用）；只是重新跑Pegasus会覆盖手工C，不能用它代替这个步骤。target环境仍VIP9000NANODI_PLUS_PID0X1000003B、shader_core1、trace开启。
3. 生成器可用安装header已确认的 `vsi_nn_GenerateNBG(graph,void*,size_t*)`。先创建图并VerifyGraph；然后buffer=NULL查询size，分配后第二次调用并保存。该API包装vxGenerateNBG，其header明确NULL查询容量。不需要先RunGraph校准，也不改权重。必须检查每次status、size、完整fwrite，保存log/NBG SHA。
4. 这是SDK-native NBG重建路线，不预先保证与旧pack-nbg-unify产物格式/端口完全兼容。先校验目标、11入5出、所有端口dtype/quant/shape及trace；若verify失败，记录具体下一层限制即止，不把失败候选传板。
5. 若生成成功，先单块固定输入做有限数值screen，再谈速度：原输入/KV/mask/位置完全固定，逐输出比较FP32参考与旧NPU；边界负数、0、最大值也须覆盖。后端可能保留额外copy/数据布局变化，必须测真实API时长，不能将BITCAST名字解释为免费alias。

详细原始证据：sdk-bitcast.json、sdk-nbg-generation-api.json、generated-c-converters.json。前端是否能显式表达BITCAST由graph agent另查；这里不假设Acuity层存在。
