# FP16 down 布局候选与 L2 慢路径证据

## 当前已证实

C512/M16 L0 与 L2 的原 `decoder.json` 完全相同，排除输入形状/图拓扑不一致。L2 源来自现役 `peak155-repaired-v3-t16/l2`；其原 NB SHA 已与现役板端证据核对，不是误选全FP16模型。

差异集中在真实层2、3的 MLP down（b0/b1_fullconnect_116，4864→896）：L0 是 I16 activation / I8 per-channel weight / I16 output；L2 保留 FP16 activation / FP16 weight / FP16 output。L2 两个 down 权重各8,716,288字节，其它12个FC仍为I8。SwiGLU swish、乘积、reshape115/117也为FP16，共10个更多的FP16激活条目。单块API已观察M16为112.207ms，L0为8.423ms；M32/64/128/256的L2为222.12/456.72/922.93/1890.17ms，批量没有解决这一主要成本。

CPU/NPU输入输出flush+invalidate约0.04ms（root的独立IO诊断），相对run112ms很小。NPD仅报告NN计数器，FP16 down trace node232/294没有这些计数，不能据此声称CPU fallback或确定为TP/Shader；需要实际更换算子后的配对测量。

## 原图 trace 补导出

`build_trace_exports.py` 只增加环境变量 `VIV_VX_DUMP_NODE_TRACE_DB=1`，其余导出输入和参数不变。`trace-export/status.json` 中 L0/L2 两个M16 NBG与原实测件逐位相同，因此新 `export/node_trace_db.json` 可以映射原NPD。实际trace没有target字段，不能使用不存在的engine标签归因。

## 本次唯一 FP16 候选

`candidates-fp16/build_conv_candidate.py` 先用一个小ONNX Conv导入探明SDK实际schema（`schema.json/data/onnx`保留），再在独立 `conv1x1-m16-l2` 中修改：

- 两个down输入 reshape `[M,4864]` → NHWC `[1,1,M,4864]`。
- FullConnect → 1×1 Convolution，padding VALID，stride1，group1。
- 权重 `[4864,896]` → `[1,1,4864,896]`，只加维度，原数组字节不变。
- 后续原reshape117恢复 `[1,M,896]`，其它节点不改。

原 `decoder.quantize` 字节完全相同，没有改成A8/W8，也没有引入动态截断。全部14个FC/Conv实际导出的weight dtype和字节均与原件相同，包括两个FP16 down。所有11输入、5输出的shape和quant metadata也完全相同。SDK debug graph中两个down确实是CONV2D，不是仍被标为原FCL；SDK新增布局处理的运行成本尚待板测。

原F32图与候选F32图的5输出max相对L2=3.17e-7；将两个down weight按原FP16舍入再重展开，max相对L2=3.23e-7。只是CPU数学等价断言，不能代替FP16算子在NPU上的数值与速度验证。`graph_reference_conv.py`是新增局部解释器，原共享reference未改。

NBG：`conv1x1-m16-l2/export_nbg_unify/network_binary.nb`
SHA256：425bc11857e5691b30abf3d60700dcc735dd13f6b12bed3671f32a287dfe02b5
大小：33189152字节；编译耗时：19.471秒。

板测直接复用 `fixtures-canonical/c512-m16-l2/probe-input.f32` 和原 `probe-reference.f32`。推荐原FCL→Conv→原FCL，3warmup+30样本，5输出分端口finite/relative-L2/maxabs及逐位对照，普通测时与NPD分开，driver释放0。候选trace文件随图保留。如没有速度收益不扩全层；若有收益也必须完整模型固定反馈logits与自然中文质量配对，不能凭单块提升宣称≤100ms。

## 量化后备方向（尚未实现）

若FP16布局改写无效，可研究每输出通道的 INT8 base+INT8 residual 两条FC相加，提高权重重构精度。但FP16输入可能仍使SDK走浮点路径，必须显式、审慎评估FP16→I16转换，并通过实际节点/计数器证明走整数加速。需先从多个canonical真实场景、层2/3及全长度区间采集down输入激活；报告通道极值/分位数、饱和率和激活量化误差，再加权重base/residual误差评估乘积输出。它改变数值格式，不能与本FP16同精度候选混称。不能简单恢复普通W8而忽略原敏感层保护原因。

## C1280 现役完整模型的局部 overlay

`build_conv_c1280.py` 已准备并编译仅替换现役 `prefill-layer2.nb` 的C1280/M16候选；所有decode图、其它prefill块、head和embedding不动。原C1280量化文件字节相同，14矩阵实际导出weight dtype及字节相同，11入5出端口metadata完全相同。原C1280图→Conv与C512Conv→C1280容量扩展两种构造逐字典相同。CPU graph等价max relative L2=3.23e-7。

NBG SHA256：dc772225e8c2b9ec2f6376132ba12aa9938dbe966b678e2bd7672832299fcfa6；大小34050560字节；编译22.337秒。准确路径、原图SHA与范围见 `deployment-overlay-c1280.json`，完整审计见 `status-c1280.json`。该候选目前仅构建，未由本图侧任务部署或板测。

根任务的C512单块A-B-A已得到2.448× API提速，但候选不是bitwise等价；这只支持进入完整模型验收，不能跳过端到端固定反馈logits、自然中文与行动质量、温度/driver释放对照，更不能据此宣布100ms目标完成。

## 有界 C512/M32 单块后续候选

`build_conv_c512_m32.py` 仅编译一个M32/L2候选，目录 `conv1x1-c512-m32-l2`，未扩整栈，未修改此前冻结controller/runner/分析文件。原M32图→Conv与现有M16Conv→M32两种构造完全一致。原M32量化字节、14个实际导出矩阵dtype/bytes、11入5出metadata相同；CPU原F32图和FP16-down重展开图对照max relative L2=3.35e-7。

NBG SHA256：9b80e75c0ef95c367b5c1e423fd2ad76cb9ebb1bca4b115fda4b89dec70f9f7f；大小33973888字节；编译22.332秒。冻结输入及参考的路径/SHA、原M32远程基线和probe SHA在 `probe-manifest-c512-m32.json`。必须在同M32 canonical输入上做原→Conv→原，再考虑与M16每token成本比较；不同M单次API时间不是相同完整请求延迟。图侧未访问板卡，未宣称速度/端到端质量。
