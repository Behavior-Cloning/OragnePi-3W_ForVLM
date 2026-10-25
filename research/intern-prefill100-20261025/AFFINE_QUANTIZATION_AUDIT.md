# DFP → affine 元数据候选独立审计

结论：**已证明本候选所改 20 项的整数数值网格相同，并独立核对实际导出的 14 份权重与 14 份 bias 字节、dtype、数值 scale 相同；没有证明 SDK 算子输出逐位相同。候选编译在 DATACONVERT 节点失败，没有可上板 NB。**

本审计只读取 WSL 文件和 SDK 容器内已安装的库／头文件；容器使用只读文件系统、禁网，未执行库、编译候选或访问板卡。机器证据与每节点参数在同目录 `AFFINE_QUANTIZATION_AUDIT.json`，所有引用源文件带 SHA。

## SDK 枚举与表的独立核对

直接解析 ELF64 小端 section／symbol table，而非复用图 agent 的解析器；结果与其原始表全部 76 行一致。库为安装包内 `vsimulator/lib/libovxlib.so`，SHA `a4e3e5ba4bd8f6c2f460177002920d66b00be2ddc87f3332fd26eaad4d15182e`。

实际 SDK 头文件确认 `VSI_NN_TYPE_INT8/INT16` 分别直接别名 `VX_TYPE_INT8=2`、`VX_TYPE_INT16=4`，`Q_SHIFT=8`；DFP=1、AFFINE_ASYMMETRIC=2、AFFINE_PERCHANNEL_SYMMETRIC=3。由此 I16-DFP=260，I8-DFP=258，I16-ASYM=516，I8-ASYM=514，I8-PC=770，**没有把 unsigned 或位宽枚举误读为 signed dtype**。

| CONV2D 注册输入组合 | 注册表中存在 |
|---|---|
| I16-DFP × I8-DFP `(260,258)` | 否 |
| I16-DFP × I8-PC `(260,770)` | 否 |
| I16-ASYM × I8-ASYM `(516,514)` | 是 |
| I16-ASYM × I8-PC `(516,770)` | 是 |

注册结构为 **2 个输入、0 个输出、每行 8 bytes、76 行**。它不能证明第 3 个 bias、output dtype、具体 backend 或整图合法；反汇编还显示 `vsi_nn_QuantCheck` 在类型表检查之前执行。类型表包含也不是性能承诺，更不能据此推断运行引擎或硬件吞吐。

## 本候选实物检查

对象是 `graphs/candidates-w8/conv1x1-affine-c512-m32-l0`，仅 C512、M32、fused2 的层 0/1，不是完整模型。

- 原 14 个 FC 私有输入 reshape 的 INT16-DFP 转为 signed INT16-affine；6 个 INT8-DFP 权重转为 signed INT8-affine；8 个 PC-INT8 权重不改。
- YAML 差异严格只有这 20 项，保留 `rounding: rtne`、校准 min/max、位宽与 signed 类型；`scale=2^-fl`、`zero_point=0`。其他条目、原输出量化和 bias YAML 没变。
- 独立枚举每个 I16 项全部 65,536 个 code、每个 I8 项全部 256 个 code，FP32 反量化值逐位相同。scale 在这些实际 fl 值下都是可精确表示的有限正 FP32 二进制幂。
- 使用同样 rtne 与饱和规则，对所有半格边界及其相邻 FP32 值做数学变换检查，同样通过。**这不是实际 SDK 转换指令的验证**。
- 实际导出 C 和 `decoder.export.data` 的 14 权重＋14 bias 按各自 offset／bytes 独立提取：dtype、字节与数值 scale 全同。PC scales／zero-points 向量按 FP32／INT32 字节 SHA 比较；检查 reshape 后 channel 轴仍对应同一输出通道和相同通道数。
- 14 组输入／输出共 28 个端口的实际导出数值网格相同；14 个算子的实际 C 属性都保持 `VX_ROUND_POLICY_TO_NEAREST_EVEN` 和 `VX_CONVERT_POLICY_SATURATE`。
- 全部 14 个实际 bias scale 均独立核对为 FP32 的输入 scale × 对应权重 scale。

必须检查**实际导出结果**：例如 `b0_fullconnect_122` 的源 YAML bias 是 DFP fl31，而导出 bias 实际为 INT64 DFP fl22，等于 activation fl12＋weight fl10；`b0_fullconnect_116` 的 YAML bias 写 DFP fl28，导出实际为 PC-SYM INT64、896 个通道 scale。只逐字比较 YAML 或只检查位宽会漏掉这类关键变化。本候选导出后的实际 bias 仍与原版一致。

## 实际阻断与等价边界

`export.log` 明确报 `Check node[16] DATACONVERT fail`，`status-affine.json` 为 failed，`export_nbg_unify/network_binary.nb` 不存在。节点是 `b0_reshape_121_dtype_converter_11010`，从前序节点 15 的 INT16 DFP fl12 转为 INT16-ASYM、scale=0.000244140625、zero_point=0。**抽象网格完全相同，SDK 仍拒绝该转换。**

不能用 FP16 中转后仍称“无损元数据变换”：本机数值反例为 code=2049、scale=2^-12，原值 0.500244140625 转 FP16 后为 0.5；这一 I16 网格中 53,248／65,536 个 code 的 FP16 往返值改变。此处只计算反例，没有编译或建议部署这种桥接。

即使未来通过转换节点，Conv 与 FC 也可能使用不同累加宽度、归约顺序、requant multiplier／shift 近似、中间饱和点或 kernel；输入反量化网格与权重相同不推出输出 bitwise。保持 RTNE 标签也不证明两条实际硬件路径每个中间操作一致。新的 DATACONVERT／PERMUTE／reshape 还可能增加延迟。

## 后续必须实测的门禁

1. 成功生成最终 NB，冻结图和运行时 SHA，验证全部外部端口、实际转换与逐节点参数。不能将失败导出的 C 视为可部署图。
2. 同一 canonical 输入单块 A-B-A：预热／重复计时、五端口实际 FP32 输出的 finite、SHA、逐端口误差与热漂移；保留原始文件。不要用拼接总误差掩盖某个 hidden 端口误差。
3. 再做完整 24 层配对的 full-vocabulary logits、top1 margin、短及真实长输入自然语义质量。token 相同不等于数值无误差，更不等于任务正确。
4. C512 与 C1280 的形状差异独立标明；不能把容量与算子变化合并后声称纯算子收益。
5. 100 ms 目标仍需 S64／128 × 正文 D64／128／192／256 全 8 格及边界；含完整动态 prefill 与首 head／argmax，不预计算动态 KV；语音／RAG 常驻、资源、driver 释放和长期稳定性分别检查。

当前只支持“所检查的源／导出数值网格和常量未改变”这一有限结论；**不支持输出 bitwise、成功上板、速度提升、无质量回退或达到 100 ms。**
