# W8 原整数码扩展为 I16 存储的 Conv 备用路线

状态：只读可行性与内存模拟完成。没有写出候选 C/payload、没有 SDK 编译、没有板测；不能视为可部署或提速结果。

后续状态：本方案随后获得授权，独立候选已在 `../candidates-w16-storage/` 完成静态审计和 SDK make/Verify。当前仍无 NB、无板测；最新状态见该目录 README。以下保留只读方案阶段的证据与计划。

目标是在 C512/M32/L0 的两个普通量化层中，保持原 DFP I16 激活、原量化权重的数值和原 I16 输出，避开 Conv 的 A16-DFP × W8 类型限制。14 个权重矩阵仅把已导出的有符号 I8 整数码扩展到 I16 存储，不从原 FP32 模型重新量化，也不填充新的低位精度。

## 已证明的部分

源是失败的原量化 Conv 图 `conv1x1-c512-m32-l0/export`，不是 affine 版本。其 14 个 FC 已改成 1×1 Conv，外部仍为 11 输入、5 输出；矩阵的原整数权重、DFP/PC scale、bias 与普通 FC 基线一致。原图因 Conv 输入类型表拒绝 A16-DFP × W8 而失败。

`plan_weight_i16_readonly.py` 读取实际生成的 `vnn_decoder.c` 和 `decoder.export.data`，在 RAM 中完成重建，并输出 `weight-i16-feasibility.json`。14/14 矩阵满足：

- `old_i8.astype(<i2)` 的每个整数值等于原码，包含 -128；反向转 I8 后 SHA 与原片段相同。
- 6 个 DFP 矩阵保留原 fl；8 个 PC-SYM 矩阵保留原 scale 向量、channel_dim=3 与零点语义。原码与扩展码的 FP32 反量化结果逐位相同。
- 除常量读取参数中的 storage dtype、offset 和 length 外，模拟 C 与原 C 完全相同。激活、bias、输出及端口的实际量化属性没有更改。
- 所有非权重常量字节不变。必须称“整数码及反量化数值相同”，不能称“权重原始字节相同”：I16 存储的字节数翻倍。

这只是常量和静态图契约的证明。Conv 的累加、舍入、溢出和输出再量化仍可能与 FC 不同，需要实际输出验证。

## SDK 类型依据和边界

`sdk-conv-types.json` 从当前安装的 `libovxlib.so` 注册表提取真实组合；库 SHA 为 `a4e3e5ba4bd8f6c2f460177002920d66b00be2ddc87f3332fd26eaad4d15182e`。

| 输入激活 | 权重 | 注册类型表 |
|---|---|---|
| I16 DFP (260) | I8 DFP (258) | 不支持 |
| I16 DFP (260) | I8 PC-SYM (770) | 不支持 |
| I16 DFP (260) | I16 DFP (260) | 支持 |
| I16 DFP (260) | I16 PC-SYM (772) | 支持 |

该表仅声明 2 个输入、0 个输出，不覆盖 bias/output 或实际后端实现。QuantCheck、VerifyGraph、NBG 导出、板端 NN kernel 和吞吐均未验证。因此不能由表推断一定能运行，更不能声称会加速。I16 权重增加带宽，也可能降低 MAC 吞吐。

## payload 的最小重建

实际数据有 32 个常量读取声明，对应 29 个唯一 `(offset,length)` 区间，恰好覆盖原文件，没有额外 header/gap。

| 项目 | 字节 |
|---|---:|
| 原 payload | 29,890,566 |
| 原 14 个 I8 权重 | 29,818,880 |
| 扩展后的 14 个 I16 权重 | 59,637,760 |
| 其它唯一常量 | 71,686 |
| 紧凑重建 payload | 59,709,446 |
| 紧凑重建新增 | 29,818,880（约 28.44 MiB） |
| 保留旧文件再追加全部 I16 权重 | 89,528,326（未加可选对齐） |

建议紧凑重建：按旧区间顺序处理，权重转 little-endian I16，非权重原字节复制，再统一重写全部 32 个常量引用的 offset/length；权重引用的 dtype 改为 `VSI_NN_TYPE_INT16`。`NEW_CONST_TENSOR` 宏会读取对应区间并将 `attr.dtype.vx_type` 设置为该 storage dtype。

必须保留别名：4 个 I64 zero-bias 声明共享 `(6,7168)`；另一个 grouped bias 声明供两个 block 的 gate/up 共用。禁止按声明重复拷贝而误算 offset。其它 shape、weight layout、节点连接和端口注册数组全部保持不变；1×1 卷积不跨 token。

模拟结果 SHA：

- 原 C：`10a772e70f4ecc8300129f59ae4717e67a4f7def316be4fdfc8007607e65f8e3`
- 原 payload：`839fa51843b33140e864d01423f9da81e14a996cbe3d666c57fa0e84040a55e2`
- 紧凑 payload（仅 RAM）：`08d1a7880e6924f938853df20a00f5a760527ca757345692ca789ccfb1c1e50f`
- 替换 C（仅 RAM）：`e995dd7c770ad4b1f3af50ca2aaa3bb01dbe1d4f0aaa379da98dced402ae238f`

逐 tensor 的旧/新 offset、长度、dtype、原码 SHA、扩展 SHA、scale SHA 均见 JSON；若实际落盘结果不同，先停止并查明差异。

## bias 与外接口不变式

保持原实际导出 C 的 I64 bias 字节和量化属性。由于输入 scale 与每个权重数值 scale 未变，bias 所需 scale = input scale × weight scale 也不变。不要机械套源 YAML：原工具会推导 bias；例如 v 投影原 YAML bias fl31，实际导出是 fl22，down 的实际 bias 为 PC-SYM I64。

I16 存储并不意味着权重数值范围扩大：实际矩阵只包含原 I8 范围，不能重标 fl、重新最大值校准，或用原 FP32 权重生成新的 I16 码。11 输入、5 输出的顺序、shape、dtype、scale、DFP fl 必须逐项与原 NB 接口核对。

## 获准后可执行的最小步骤（本轮未执行）

1. 新建独立目录，复制原失败 Conv 的 generated-C export；不覆盖源图、量化文件和失败日志。运行模拟脚本同等断言后，才落盘新 C/data 与完整差异清单。
2. 使用原 SDK 环境和 makefile 重编 C，整个过程持有 `/tmp/opi-prefill-compile.lock`。该路径不再调用 Pegasus，避免其重新导出覆盖手工常量或插入量化转换。
3. 驱动程序依次创建图、核对 11/5 端口、VerifyGraph、按实际 SDK 的 NBG 生成流程导出；每阶段日志独立保存。原 `export/main.c` 只是 verify/preprocess/process/postprocess 的模拟器 runner，并不会自动生成 NBG，不能把 `make` 成功当作 NB 成功。
4. NBG 导出还有独立风险：另一个 BITCAST 实验已验证 Create/Verify 成功，但 `vsi_nn_GenerateNBG(graph,NULL,&size)` 的 size query 崩溃。该故障不能归因于 I16 权重（本候选尚未编译），也不能默认同一 native-export 路线可用。实际实施须先识别受支持的序列化流程，不能反复盲试或将 Verify 成功包装为已部署。
5. 若成功生成 NB，再核对真实端口、节点/权重布局和常量证明，交 root 做同 canonical 输入的原 FC → Conv-I16-storage → 原 FC A-B-A：3 warmup、30 计时、5 输出分别比较、driver release=0。单块结果不替代全模型质量与端到端性能。

该路线的价值是保留 W8 数值，测试另一种受支持的 Conv 存储组合。收益仍未知，应作为有界备用实验；FC-axis3 已实测没有实质提速，不再扩栈重复。
