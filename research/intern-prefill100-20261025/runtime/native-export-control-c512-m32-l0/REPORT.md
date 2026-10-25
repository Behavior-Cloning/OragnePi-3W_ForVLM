# 原始图同 native export harness 对照

原始 `graphs/cases/c512-m32-l0/export` 的 decoder C 和常量数据逐字复制，没有 BITCAST。main与最终make链接环境均与BITCAST候选逐位相同。原图导出目录的makefile是依赖common.target的模板，在首次make阶段失败，原件与日志保留；使用与BITCAST相同且已经make通过的Linux makefile后才执行真正的模型对照。

真实结果同样是：make0 → Create成功11输入5输出 → VerifyGraph0 → GenerateNBG(NULL,size)大小查询阶段SIGSEGV139。没有生成NB。两份运行的stdout日志SHA完全相同，端口metadata也相同。plain的vnn_decoder.c与decoder.export.data再次核对与原始SHA一致。

因此当前崩溃不能归因BITCAST；native NBG导出harness/SDK路径仍存在问题。NULL查询签名确有SDK头文件依据，但正式Pegasus pack流程并不调用此自定义路径，而使用生成的标准main、11个tensor输入、Verify/PreProcess/RunGraph以及VIV_VX_ENABLE_SAVE_NETWORK_BINARY等环境变量自动落盘。后续需要先重建并审核官方保存路径的plain对照，再决定重试BITCAST，不能跳过原始图控制。

范围修正：自定义main没有显式调用RunGraph，但SDK vxGenerateNBG内可能重新Verify并Process。因此此前注释“no simulator inference”不能作为已证明的内部行为；正确范围是没有完成的数值推理输出、没有板卡操作和速度数据。执行源码保留原样，本文纠正该注释的解释。

本次持统一编译锁，已结束释放。source-proof.json、RESULT.json、ports-verified.json及make/verify日志保留；不覆盖前一个BITCAST失败目录。
