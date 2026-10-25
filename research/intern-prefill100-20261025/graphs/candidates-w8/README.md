# 普通 W8 FC 布局候选：C512 / M32 / layers 0–1

当前结论：两个独立单块编译均被 SDK 拒绝，没有生成可部署 NB，没有板卡操作。保留原模型全部量化与权重的 all-FC→Conv 失败后，只尝试了一次经授权的等值 affine 元数据候选；未继续降位宽、改默认或扩栈。

## 已完成的两次编译

1. `conv1x1-c512-m32-l0`：14 个私有输入 reshape 从 `[M,I]` 变 `[1,1,M,I]`，14 个 FC 变 1×1 Conv（stride 1 / padding VALID / group 1），权重只增加两个长度为 1 的维度。没有跨 token 卷积。原 quantize 文件字节完全相同。SDK 在 node 22（b0 v projection）拒绝 `DFP INT16, DFP INT8` 输入组合。
2. `conv1x1-affine-c512-m32-l0`：沿用完全相同图与数据，只把 14 个私有 reshape 的 I16 DFP 输出，以及 6 个 DFP I8 权重，用 `scale=2^-fl, zero_point=0, rtne` 等值 affine 表达。8 个 per-channel I8 权重、14 个原 FC 输出量化及源 bias 配置均不改。20 项变化逐个记录在 `affine-quantization-diff.json`；I16 全 65536 个整数码与 I8 全 256 个整数码均验证原/新 f32 数值网格一致。SDK 自动插入的 node 16 `DATACONVERT`（DFP I16→ASYM I16）在检查阶段失败。

`status.json` 和 `status-affine.json` 保留完整 FP32 等价、命令、SHA 和错误。原 FP32 全 5 输出最大 relative-L2 为 2.47e-6；使用实际已导出 FC 量化权重反量化后的 FP32 图最大 relative-L2 为 1.52e-6。这是 CPU 数学图检查，不是 NPU 精度验证。

两个失败导出都已生成 C 和常量，因此可以独立审计：14 个 weight、14 个 bias 的实际 dtype、raw bytes 与数值 scale 均与原导出一致，14 个输入/输出的 I16 数值网格一致。affine 候选的量化元数据不同，不能称 quantize 文件 bitwise 相同。审计文件分别为 `rejected-candidate-audit.json`、`rejected-affine-candidate-audit.json`。整个导出的常量 payload SHA 均为 `839fa51843b33140e864d01423f9da81e14a996cbe3d666c57fa0e84040a55e2`。

编译失败发生在生成 NB 前；外部图输入/输出与 tensor-inputmeta 没有修改，但不存在候选 NBG 的外部端口验收，也没有速度测量。预编译阶段遇到的 grouped-bias 注释解析问题已保留在 `attempts/precompile-grouped-bias`，该次尚未启动 SDK 编译。重复的数据与 payload 在 SHA 核对后改为链接，记录在 `dedup*.json`。

## 直接读取本机 SDK 的类型限制

不是引用其它版本文档：`inspect_sdk_conv_types.py` 和 `inspect_sdk_dataconvert_types.py` 从 Docker `ubuntu-npu:v2.0.10.2` 中的实际 `libovxlib.so` ELF 符号与只读表提取。库 SHA 为 `a4e3e5ba4bd8f6c2f460177002920d66b00be2ddc87f3332fd26eaad4d15182e`。枚举值来自同 SDK 头文件：I8=2、I16=4，qtype 左移 8 位，DFP=1、ASYM=2、PC-SYM=3。

| 输入组合 | CONV2D 注册表 |
|---|---|
| DFP I16 × DFP I8 | 不接受 |
| DFP I16 × PC I8 | 不接受 |
| ASYM I16 × ASYM I8 | 接受类型检查 |
| ASYM I16 × PC I8 | 接受类型检查 |
| DFP I8 × DFP I8 / PC I8 | 接受类型检查 |
| DFP I16 × DFP I16 / PC I16 | 接受类型检查 |

CONV2D 表是 **2 输入、0 输出、76 对**，不覆盖 bias、输出类型和底层 kernel 支持，不能据此宣称完整组合可用。QuantCheck 先于此表。

DATACONVERT 表为 **1 输入、1 输出、131 对**：DFP I16↔ASYM I16 不接受；FP32↔ASYM I16 也不接受。ASYM I16 仅与 FP16、自身、BOOL8 互转。FP16 桥接会丢失部分 I16 网格的有效位，不能作为无损修复。原 DFP I16↔DFP I8 转换在表中，但会改变精度。

## 只读后续方案；均未编译

**FC axis3（保留原全部 quantize 字节）**：14 个私有 reshape 改 `[1,1,M,I]`，FC 仍为 fullconnect，axis 从 1 改 3，原 `[I,O]` 权重不动，下游原 reshape 恢复外部形状。`inspect_fc_axis_semantics.py` 调用已安装 Acuity 的 `FullConnect.compute_shape_nhwc/get_variable_shape`，14 项均确认输出 `[1,1,M,O]`，权重仍 `[I,O]`；这是纯内存 shape 查询，不是导出/编译。证据 `sdk-fc-axis3-readonly.json`。原生成 C 是 legacy `VSI_NN_OP_FCL`，只给 weights；SDK exporter 另有 `vxFullconnectAxis`，但 axis3 是否真正改变 lowering/tiling 或被重新规范成原二维图仍未知。可能毫无收益或增加 reshape，必须单块测量。

**BITCAST 等码视图**：要表达的是相同 signed I16 整数码及相同数值网格，仅切换 qtype metadata。标准 Acuity 的 layer、OVX exporter、importer、optimize 共 394 个模块未找到 bitcast/bit_cast 原生层/mapper；不能把自定义生成 C 流程包装成前端原生支持。安装 SDK 后端存在 `VSI_NN_OP_BITCAST→vxBitCastLayer`（独立证据见 `../../runtime/bitcast-review/`）。可能需要对冻结生成 C 的等值 DATACONVERT 节点做受审计的 BITCAST 替换，再重建 NB；仍须通过 OpenVX verify、常量/metadata 检查和板上逐输出比对，不能默认零成本或数值等价。尚未编译此路线。

**内部 A8（改变精度，暂未授权执行）**：仅 14 个 FC 私有输入张量从 I16 降为 I8，其它注意力、归一化、残差、外 11 输入/5 输出及原 I16 FC 输出网格保持。共有 8 个不同上游激活，q/k/v 与 gate/up 可分别共享统计。原 W8 code/scale 不变；bias 的实际 scale 必须随 A scale 改为 A×W，不能机械照抄 YAML 的 bias fl（原工具导出本来就可能覆盖它）。因此 bias 的整数 bytes 不保证不变，该路线须另立精度基线。

A8 范围先用独立校准样本的 abs-max 或裁剪/MSE 搜索，明确记录饱和比例、scale、舍入、负端点；保持 DFP 时 `fl8=floor(log2(127/maxabs))` 可作保守初值，最终必须由真实激活和留出样本验证。下面只是旧量化 min/max 的初步估算，所有 14 处步长均会粗化 128 倍，不能直接当作新校准结果。

| FC 节点（b0/b1） | 作用 | A16 DFP fl | W8 格式 | 输出 DFP fl | 覆盖旧范围的 A8 初始 fl |
|---|---|---:|---|---:|---:|
| `b0_fullconnect_116` | down | 11 | per-channel symmetric | 11 | 4 |
| `b0_fullconnect_119` | o | 15 | per-channel symmetric | 15 | 8 |
| `b0_fullconnect_122` | v | 12 | DFP fl=10 | 15 | 5 |
| `b0_fullconnect_125` | up | 10 | DFP fl=9 | 12 | 3 |
| `b0_fullconnect_128` | k | 12 | per-channel symmetric | 6 | 5 |
| `b0_fullconnect_131` | gate | 10 | DFP fl=9 | 11 | 3 |
| `b0_fullconnect_134` | q | 12 | per-channel symmetric | 7 | 5 |
| `b1_fullconnect_116` | down | 11 | per-channel symmetric | 12 | 4 |
| `b1_fullconnect_119` | o | 12 | per-channel symmetric | 13 | 5 |
| `b1_fullconnect_122` | v | 10 | DFP fl=10 | 12 | 3 |
| `b1_fullconnect_125` | up | 10 | DFP fl=9 | 12 | 3 |
| `b1_fullconnect_128` | k | 10 | per-channel symmetric | 6 | 3 |
| `b1_fullconnect_131` | gate | 10 | DFP fl=9 | 11 | 3 |
| `b1_fullconnect_134` | q | 10 | per-channel symmetric | 10 | 3 |

初次单块校准建议至少 32 条不同自然家庭指令（含否定、停止、方向、数词、并列动作和含糊/拒绝），覆盖两档 system 和四档动态长度，形成 256 个输入条件；另留出 16 条不同指令 × 8 条件 =128 条验证，合计48条独立语义样例。长度扩展/填充不增加独立语义样本数。这个数量是进入单块候选评估的工程起点，不是准确率保证。现有 canonical 的一个 common prompt/M256 切片与24 core/20 boundary成本夹具不足以完成校准。

需要捕获真实基线 NPU 的对应内部激活，或先核对 FP32/Q8 回放与实际 NPU 的分布偏差；单纯 source FP32 fixture 不能冒充真实端侧校准。统计可流式累积直方图/极值，不必保存所有大张量。留出评估须至少逐端口 hidden/K/V 误差、有效历史区误差、饱和比例、固定反馈 logits/argmax，以及完整模型自然中文回答/动作质量；收益必须把新增转换成本算入完整 prefill，而非只计卷积内核。没有已测的 A8 加速数字。
