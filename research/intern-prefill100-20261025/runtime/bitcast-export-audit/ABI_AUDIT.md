# NBG 导出崩溃：独立 ABI / 官方流程复核

当前证据不支持把失败归因于 BITCAST。原始未改图和 BITCAST 图采用同一导出 main，均 Create 成功、11 输入 / 5 输出、Verify=0，然后在 size-query 阶段 rc139；二者日志 SHA 完全相同。没有 NBG 产物。“相同阶段”不代表已定位到同一崩溃指令，尚无 backtrace/core。

## NULL 查询符合接口契约

安装 SDK 的 include/ovxlib/vsi_nn_graph.h:256 声明 vsi_nn_GenerateNBG(vsi_nn_graph_t*, void*, size_t*)。底层 include/VX/vx_api.h:1241–1248 明确写 buffer 为 NULL 时查询所需 size；vx_size 为 size_t。本 main 的 size_t bytes=0; vsi_nn_GenerateNBG(g,NULL,&bytes) 与声明相符。

实际 libovxlib.so 的函数（0x4c980）读取 graph+8 内的 vx_graph，保留第二/第三参数，调用 vxGenerateNBG，再转成 VSI_SUCCESS / VSI_FAILURE。没有额外输出指针层级，也没有证据表明应先分配一个猜测大小的 buffer。

libOpenVX.so 的 vxGenerateNBG 检查 NULL buffer / 非 NULL size，size-query 分支会进入 vxVerifyGraph，之后进入 vxProcessGraph。因此当前 main 的注释 no simulator inference 不能保证底层未处理图；它不是单纯读取一个已有字节数。

## 当前 harness 与实际 Pegasus pack 路径不同

安装的 nbgenerator / ovxlib_case/casegenerator 反汇编证明官方路径设置：

|环境变量|实际值/构造|
|---|---|
|VIV_VX_ENABLE_SAVE_NETWORK_BINARY|字符串 1|
|VIV_VX_SAVE_NETWORK_BINARY_PATH|self.nb_file，默认 network_binary.nb；是文件路径而非目录|
|VIV_VX_NBG_NAME|若未预设，使用 self.network_name；本图 decoder|
|VIV_VX_NBG_INPUT_RANK|按输入布局 NCHW / NHWC；本图 nbg_meta 全部 nchw，使用 NCHW|

casegenerator 地址证据：0x474f6→0x47517 将 network_binary.nb 赋 self.nb_file；0x4d533→0x4d5e3 将它写入 SAVE_NETWORK_BINARY_PATH；0x4f243→0x4f2e4 用 self.network_name 初始化 NBG_NAME；0x4d76f→0x4d791 将 1 写入 ENABLE；0x4ff33→0x4ff55 设置 NCHW。

官方模板为 acuitylib/app/exporter/ovxlib_case/vxcode/template/main.c：Create → Verify → PreProcess 全部输入 → RunGraph，走自动保存路径，没有显式 GenerateNBG 查询。当前手写 main 则未安装 11 输入、没有上述自动保存配置。这是可证实的流程差异，不能把某一个差异单独宣布为已定位根因。

## 有依据的修复候选与验收

在新的独立目录恢复 SDK 原生成 main，保持 graph C / decoder.export.data / quantization / 11 输入 SHA 不变。设置 ENABLE=1，PATH=/work/network_binary.nb，NAME=decoder，INPUT_RANK=NCHW，使用原成功平台/链接库。先原始图验证能产出 NBG，再对 BITCAST 使用相同流程。禁止读取旧 NBG 冒充本次产物。

argv 为程序、decoder.export.data、随后 g->input 顺序的 11 tensor：hidden0、k0、k1、v0、v1、mask0、mask1、cos0、cos1、sin0、sin1。必须依据 exported C/nbg_meta，不能依文件名自然排序。使用冻结 canonical feed，逐数组 SHA 及拼接总 SHA 校验。

不要仅加环境后继续当前 GenerateNBG 调用来声称复制了 Pegasus；应恢复官方 main 完整顺序。验收为 rc0、非空新 NBG、外部 11/5 端口不变、实际源码/权重来源相符。成功后仍需板端 canonical 数值门禁，编译成功不证明 BITCAST 输出等价。

若原图官方流程仍失败，保留 host backtrace、实际加载库路径/SHA 和环境差异再定位。审计员未启动新模型、未接触板卡、未修改候选；尚未验证修复成功。

## 证据

ABI_AUDIT.json 保存接口、库与模板 SHA、两次运行状态。
vsi_nn_GenerateNBG.asm.txt、vxGenerateNBG.asm.txt 保存实际库反汇编。
pegasus-env-assignments.asm.txt 保存官方环境赋值。
official-main.c.txt 保存原模板。

## 后续受控验证结果

官方流程已由 runtime agent 执行并成功，两版 rc0。独立复核确认 plain 新 NB 逐字节复现旧图；BITCAST 生成独立新 NB，输入/源码/权重来源通过。详细见 ../official-export-c512-m32-l0/INDEPENDENT_EXPORT_AUDIT.md/json。先前“未验证修复成功”描述属于执行前时点；现在已有修复流程成功证据，但未定位旧 NULL-query 崩溃的唯一原因，亦尚无板端输出/速度资格。
