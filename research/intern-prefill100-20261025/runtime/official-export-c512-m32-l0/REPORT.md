# 官方导出流程恢复与 BITCAST W8 Conv 候选

已完成主机侧图导出和精确属性审计，可供单块板端验证；没有完成板端数值或性能准入。本目录不访问板卡。

两个图的几何均为 first layer0、fusion width2、capacity512、query32。旧 NULL buffer 的 GenerateNBG 查询路径曾让 plain 与 BITCAST 同时崩溃，失败证据独立保留在相邻目录。改为SDK实际Pegasus使用的流程：原生成main，Create→Verify→11真实tensor预处理→RunGraph，通过环境变量自动保存NBG，不直接调用NULL sizequery。

## 已验证的调用与输入

- `VIV_VX_ENABLE_SAVE_NETWORK_BINARY=1`。
- `VIV_VX_SAVE_NETWORK_BINARY_PATH=/work/network_binary.nb`，这是文件名而非目录。
- `VIV_VX_NBG_NAME=decoder`，`VIV_VX_NBG_INPUT_RANK=NCHW`。
- target为VIP9000NANODI_PLUS_PID0X1000003B，1shader core、trace enabled、同SDK镜像，网络关闭，4CPU/8GiB限制，每阶段180s有界。
- 11输入严格按hidden0,k0,k1,v0,v1,mask0,mask1,cos0,cos1,sin0,sin1；来自已冻结 `fixtures-canonical/c512-m32-l0/probe-input.f32`，SHA b4bd76f70c9fa53260b78aff30e03a3f72cbc6bcda1dc9dcbd8214444028a9e5。它是实际中文FP32层状态和历史KV的对照，不是SDK随机输入；canonical历史是独立FP32生成，并非抓取NPU缓存。
- `prepare_inputs.py`分割每段并以9位有效数字输出.tensor，再回读float32，逐元素与原F32完全一致；两个版本用完全相同输入。源码及输入分段SHA在source-proof.json。

## 实际产物

|版本|字节|SHA256|
|---|---:|---|
|plain原图|24760720|d7adc37f977973a3ce3d40be0d2c1e8539ba187ce69f2fe12f6280d1b5b7d878|
|BITCAST W8 Conv|24751168|d194bada4a35f2759c5de89d7d0fcb10da982f213de5245b40c35085b7c4e50d|

plain文件与此前实际板测原图逐字相同；这验证了恢复的导出路径。两版本make及官方generate都rc0，约1.2–1.3s编译、6.6–6.8s生成。日志里的174/183ms是host simulator/export执行，**不是板卡NPU延迟**；Top5零输出也不作为模型数值准确性证据。

## 属性和量化审计

`inspect_nbg.c`通过SDK公开接口直接导入这两个实际NB：vxImportKernelFromURL(FILE)→vxGetKernelParameterByIndex→vxQueryParameter(META_FORMAT)→vxQueryMetaFormatAttribute。没有运行图。实际16端口为11输入5输出，顺序、shape、dtype和量化与原NB及预期一致；结果在plain-binary-ports.json与bitcast-binary-ports.json，归一化见binary-ports-normalized.json。

SDK的量化union查询会在非活跃字段返回别的成员位模式；因此DFP只使用fl（scale=2^-fl），不能把原始JSON中的无效scale别名解释成真实scale。FP16使用none。归一化脚本逐项匹配nbg_meta.expected.json。

独立审计INDEPENDENT_EXPORT_AUDIT.json解析28个常量C记录与真实data payload：14权重保持I8存储、整数码和值均exact；14bias存储和值exact。6个权重与6个bias的DFP元数据改为等值affine，per-channel权重轴1→3配合1×1Conv形状，scale值保持。因此**不能称quant文件逐字未变**。14 activation转换也已穷举65536个signedI16码确认等值网格，BITCAST opcode替换补丁保存在前一失败候选目录。

输出的计算顺序/舍入仍可能随FCL→Conv改变，元数据和值一致不等于实际输出逐位一致。下一步应由root在双锁下进行单块A-B-A，先验收hidden与4个KV输出、无漂移和driver释放，再讨论提速，不直接扩栈。

## 交付和复现

`deployment-manifest.json`包含root要求的geometry、NB路径/SHA/bytes、source_nb_sha、11/5实际端口、14weights/bias exact字段、canonical输入路径及全部证据。state=complete仅表示本次host交付完整，不表示部署验收通过。

执行代码：prepare_inputs.py、run_official.py、inspect_nbg.c、inspect_binary_ports.py；标准main/C/data保存在plain/bitcast可再生build目录（Git忽略），原始源码引用及SHA在source-proof.json，BITCAST补丁前一目录保留。main和C没有被重新生成覆盖。两个有界job已结束，统一编译锁释放。没有更改现役runtime或服务。
