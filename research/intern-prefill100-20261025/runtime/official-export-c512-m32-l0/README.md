# 可复用的 SDK NBG 导出与实际端口检查

当前产物已冻结：deployment-manifest.json SHA `3c00f3c5cfb039eecca77d1a603970e2873ddf2c167dffff7b44142c29dd7702`，BITCAST NB SHA `d194bada4a35f2759c5de89d7d0fcb10da982f213de5245b40c35085b7c4e50d`。Root正在板端A-B-A，勿覆盖两个build目录或改source。此处host审核通过不等于板端加速/精度通过。

## 正确导出流程

详情和失败历史见REPORT.md，相邻bitcast-c512-m32-l0/native-export-control-c512-m32-l0保留NULL size-query崩溃证据。

复用时创建独立目录，复制目标图原始生成的main.c、vnn_*.c/.h和常量data。使用本次成功的SDK Linux makefile，不带依赖`/common.target`的IDE模板。保持原target、1shader core、官方main预处理及图输入顺序。链接库环境需要`vsimulator/lib`和`cmdtools/common/lib`，否则libvdtproxy.so加载失败。

正式生成并不使用自定义vsi_nn_GenerateNBG(NULL)调用。运行原生成main，参数为data文件和依序的全部.tensor：Create→Verify→PreProcess→RunGraph。设置以下已由SDK casegenerator反汇编核对的环境：

```text
VIV_VX_ENABLE_SAVE_NETWORK_BINARY=1
VIV_VX_SAVE_NETWORK_BINARY_PATH=/work/network_binary.nb
VIV_VX_NBG_NAME=decoder
VIV_VX_NBG_INPUT_RANK=NCHW
```

SAVE_PATH是文件而非目录；NBG_NAME是生成网络名，复用于其他网络时必须从其生成源码确认，不硬套decoder。NCHW也必须按真实metadata确认。保留同VSIMULATOR_CONFIG和SHADER_CORE_COUNT，先在同图plain控制复现原NB，再更改候选。

`run_official.py`展示完整docker argv、环境、180秒阶段timeout、timeout停止容器、统一`/tmp/opi-prefill-compile.lock`；它有固定plain→bitcast两版本，后续实验应复制成独立controller，不能直接重跑覆盖本证据。host模拟导出时间不能作为板卡性能。

`prepare_source_tree.py`用于在新复制的空目录重建已记录的两源码树，拒绝覆盖现有目录；`prepare_inputs.py`把实际canonical输入分成11段，并%.9g文本保存后F32逐字回读校验。11端口为hidden0/k0/k1/v0/v1/mask0/mask1/cos0/cos1/sin0/sin1，不能按文件名自然排序。来自SDK自动生成的随机.tensor不应替代本canonical fixture。

## 独立读取实际 NB 端口

`inspect_nbg.c`是通用单文件工具，参数仅一个NB路径，不需要原图C/quant/metadata。使用SDK公开API：

```text
vxImportKernelFromURL(context, VX_VIVANTE_IMPORT_KERNEL_FROM_FILE, path)
vxQueryKernel(VX_KERNEL_PARAMETERS)
vxGetKernelParameterByIndex
vxQueryParameter(VX_PARAMETER_DIRECTION / TYPE / META_FORMAT)
vxQueryMetaFormatAttribute(VX_TENSOR_NUMBER_OF_DIMS / DIMS / DATA_TYPE / QUANT_FORMAT / ...)
```

只导入kernel并查询，不创建执行graph或RunGraph。编译/运行示例由`inspect_binary_ports.py`给出；复制工具到新目录时在同SDK下：

```text
gcc -std=c99 -O2 inspect_nbg.c -I$SDK/include -L$SDK/lib -L$SDK/../common/lib -lOpenVX -lGAL -lvdtproxy -o inspect_nbg
LD_LIBRARY_PATH=$SDK/lib:$SDK/../common/lib ./inspect_nbg /path/network_binary.nb
```

其中SDK=`/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator`，仍保留同target/shader环境。以上是容器内示意命令，完整安全参数列表在已验证Python封装；同编译锁下编译，不与图编译争资源。原始JSON包含端口index/direction/type/dtype/dims/quant及每个API状态。此次direction0/1为input/output，dtype15/4为FP16/I16，quant0/1为none/DFP，其他模型必须按对应SDK枚举判断，不能误套VIP runtime的另一组枚举。

DFP的有效成员为fl；union里scale查询可能返回fl的位模式，虽然API状态0也不是有效scale。归一化时按量化类型选择成员：DFP scale=2^-fl、zp0；FP16 none；其他量化类型需相应处理。本工具保留原始值以便审计，没有把无效union字段混入量化等价结论。

此图实际NB端口JSON与原图完全一致，normalized结果进一步匹配原nbg_meta中的维度反序与dtype/quant。源metadata给名称，实际NB查询给每个index的属性；外部排列不允许重新排序。`validate_delivery.py`可重新核对冻结SHA、两实际NB端口、canonical和28常量证据，只离线读取，已通过。

## 验收与复用边界

最终manifest的geometry采用嵌套对象，root controller读取`m['geometry']`。标记complete只表示host导出/审计完整。仍要板端测全部输出、正确定时、A-B-A漂移、driver释放。等值量化网格不保证算子实现舍入/融合相同；源W8码保持也不是原BF16 oracle。

不要删除所有原始参考、源码补丁或失败日志；大build树已Git忽略，Git提交由root统一处理。本目录所有WSL作业已结束，没有板卡访问。
