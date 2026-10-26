# A733 上 InternVL3-1B 语言部分的理论包络

2026-10-27。结论：芯片峰值并不排斥短输入100ms，但不代表当前混合精度图能达到。INT8满利用率、理想W8权重只读一次、DRAM满规格三个乐观假设下，短输入首token下界约26–33ms；decode算力上界约3000 token/s，而理想W8带宽上界约39 token/s。实际r6约16.3 token/s不是与3000的直接同条件比较。

## 数据和条件

[全志A733规格](https://www.allwinnertech.com/index.php?a=index&c=product&id=139&solveid=34)列NPU3TOPS、32bit DRAM、LPDDR5 4800Mbps，并注明子型号差异。[Radxa A733资料](https://docs.radxa.com/en/cubie/a7s)标明3TOPS@INT8及多精度支持；这里只引用同SoC精度口径，不把Radxa板的性能当OrangePi实测。最大引脚数据率对应理论带宽4800e6×32/8=19.2GB/s。未测本板实际DRAM速率、NPU可用带宽/利用率，不把SoC支持上限当板卡实测值。

当前GGUF配套语言架构是Qwen2.5：24层，D896、FFN4864、14个Q头、2个KV头、head_dim64、vocab151674。VLM的“1B”不能全当每个文本decode必遍历的语言权重。这里推导的是纯语言阶段，vision、语音、排队另计。

## 运算量

单层七个投影的参数/MAC每token：Q和O共2D²，K和V共2D×128，gate/up/down共3D×4864。24层合计P=357,826,560；输出head H=896×151674=135,899,904。输入embedding是按token查一行，不把整个embedding表重复计入每个decode的流量。

约定一次MAC=2 operations。新prefill N个token、已缓存system S个token时，理想因果attention MAC为24×2D×[NS+N(N+1)/2]；线性部分为NP；只需最后位置logits的head为H。首token主要operations = 2×(NP+attention_MAC+H)。当前实现可能计算完整padding容量，实际大于这个理想值；若实现对所有prefill位置都算head，也要额外计入。

表格均输出1 token、只复用system，文字模板8；不包含视觉边界。输入视觉时必须按实际边界重新算N。

|system总量（正文+模板）|动态正文+模板|实际new|只受3TOPS算力限制的下界ms|再加入理想W8权重带宽的下界ms|
|---|---|---:|---:|---:|
|32（27+5）|32+8|40|9.69|25.71|
|32（27+5）|64+8|72|17.41|25.71|
|32（27+5）|128+8|136|32.93|32.93|
|64（59+5）|32+8|40|9.73|25.71|
|64（59+5）|64+8|72|17.47|25.71|
|64（59+5）|128+8|136|33.05|33.05|

使用 t ≥ max(operations/peak, bytes/bandwidth)。理想W8线性权重+head至少493,726,464bytes，单次遍历约25.715ms。此处忽略量化scale、额外outlier权重、其他算子、激活/KV中间流量和调度，故是乐观下界，不是完整性能预测。也未假设可以无限cache全部模型权重。

## Decode

每步主要operations≈2×(P+H+24×2D×context)。在无限带宽且全部计算能吃满3TOPS INT8时，context64/128/256的算力上界约3021/3005/2972 token/s。

若是理想W8，权重493.73MB加INT16 KV读写：KV每历史token为24层×K/V2×KV维128×2byte=12288byte。context128时每decode最低权重+KV约495.31MB，19.2GB/s下约38.76 token/s。若相同矩阵全为FP16，单权重流量近987.45MB，带宽上界约19.4 token/s。INT4若仍需按块解包、反量化或回退，不能直接承诺翻倍；官方INT4支持也不等于本图具有INT4吞吐。

r6当前mixed/outlier图不是这里假定的纯W8：真实图的权重精度、outlier分支、重复buffer和调度会增加成本。没有逐图有效DRAM流量和NPU计数器证据前，不能将16.3 token/s全部归因于带宽，也不能称剩余带宽空间都能兑现。实际可用上界受min(算力上界,带宽上界)约束，且还受串行/软件部分限制。

## 视觉塔为何也要提前减token

24层ViT，D1024、FFN4096。仅主要Transformer矩阵MAC：1024patch+CLS约361.18G，256patch+CLS约80.86G，64patch+CLS约19.83G。若有可完整利用3TOPS的INT8实现，三者理想计算下界约241ms、54ms、13ms；未加patch stem、投影和内存成本。当前板测是CPU FP16，不能与这些数字当同实现承诺。完整448 ViT本身的理想矩阵耗时已经超过100ms；想让包含视觉的整体首响应接近100ms，必须减少视觉计算、减少调用频次或有其他硬件路径，单纯在ViT后减少LLM token不够。

详细数值：ROOFLINE_CALC.json；视觉计算：vision-compression-architecture.json。此报告是公式估算，不宣称获得100ms端侧结果。
