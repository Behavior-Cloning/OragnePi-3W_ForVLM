# INT8误差定位与修复

本页保留早期定位结果。后续500M/1B的W8完整生成、输出头提速及重复测速见[Q8完整模型优化进展](Q8完整模型优化进展.md)，不要将下述阶段性状态当作最新状态。

2026-10-12。SmolVLM-500M语言部分，A733 VIP9000，ACUITY6.30.22、VIPLite2.0.3.2。
用户要求修复Q8误差。本报告将两层数值修复与完整生成验收分开记录。

当前结论：500M采用W8为主、两处敏感MLP与head保留FP16后，完整陪伴样例执行误差3.13%，65/67最高分token一致，五次重复通过；问候误差4.48%，6/6一致。约5.4 token/s，未达到超过最优CPU的目标。1B另外修复大词表输出头归零，FP16完整问候误差0.45%，17/17一致；1B INT8尚未完成。留出质量、长历史、30分钟和CPU最优配置验收均未完成。

## 已定位问题

在当前编译配置下，将V投影权重改成逐通道INT8后，完整decoder的首token new_v输出320个元素全部为0，K输出则正常。
V投影单独编译时可正常计算，说明不是INT8权重必然失真，也不能笼统说所有INT8矩阵乘不支持。
只量化V即可重现两层58%误差；V保留INT16、其余六矩阵INT8时恢复到约3.49%。
修复方案：其他六个矩阵保持逐通道INT8，V采用DFP INT8。七个大矩阵仍均为8位权重，激活/缓存以INT16为主，敏感算子依厂商图实现有FP16。
修复后首token V不再归零，两层相对L2约3.87%，5次一致；基线INT16约3.41%。这解决了该配置下的V归零异常，但并不表示所有层、所有输入量化误差都解决。
封闭编译器内部原因尚不可见，现有证据只能定位到逐通道V投影与完整图的组合路径；不能确认具体是融合、布局还是内部重标定的哪一条指令。

## 同输入消融矩阵

除指定权重量化路径外保持同一校准、图结构、128容量和128步追加KV输入。失败吞吐不计为可用模型速度。

| 变体 | 次数 | 两层ms/步 | 相对L2 | 最低余弦 | 层级门槛 |
| --- | ---: | ---: | ---: | ---: | --- |
| 原INT16对照 | 1 | 6.987 | 3.411% | 0.99743203 | PASS |
| 七矩阵逐通道INT8 | 1 | 4.570 | 58.453% | 0.75180481 | FAIL |
| 七矩阵DFP INT8 | 1 | 3.740 | 8.452% | 0.99408214 | FAIL |
| 只量化attention四矩阵 | 1 | 5.841 | 58.408% | 0.75189379 | FAIL |
| 只量化MLP三矩阵 | 1 | 5.309 | 3.481% | 0.99746312 | PASS |
| 只量化Q/K | 1 | 6.502 | 3.366% | 0.99720167 | PASS |
| 只量化V/O | 1 | 6.155 | 58.407% | 0.75191220 | FAIL |
| 只量化O | 1 | 6.378 | 3.416% | 0.99741793 | PASS |
| 其余六矩阵，O保留INT16 | 1 | 4.823 | 58.453% | 0.75180431 | FAIL |
| 只量化V | 1 | 6.655 | 58.408% | 0.75190387 | FAIL |
| 其余六矩阵，V保留INT16 | 5 | 5.048 | 3.494% | 0.99746590 | PASS |
| 全部INT8，V改DFP | 5 | 4.730 | 3.871% | 0.99728889 | PASS |

门槛沿用相对L2≤5%、每步余弦≥0.99；没有为修复放宽。两层通过不等于完整聊天无损。

## 单矩阵与V输出证据

- q8-linear-audit/int16：相对L2 0.0199%。独立矩阵含35个输入，其中32个来自未用于校准的真实token片段。
- q8-linear-audit/w8a16：相对L2 0.3895%。独立矩阵含35个输入，其中32个来自未用于校准的真实token片段。
- q8-v-audit/int16：相对L2 0.0588%。独立矩阵含35个输入，其中32个来自未用于校准的真实token片段。
- q8-v-audit/w8a16：相对L2 1.3507%。独立矩阵含35个输入，其中32个来自未用于校准的真实token片段。
- 异常首token：new_v非零 0/320，L2范数 0.000000。
- 修复首token：new_v非零 320/320，L2范数 0.489716。

所有测试都使用真实权重；单矩阵测试的RMSNorm输入由主机计算，仅用于定位，不是全模型运行方式。
GGUF Q8_0本身采用分块量化，重新导出成NPU逐通道/DFP INT8并非同一种Q8表示，不能保证逐token与原CPU Q8一致。

## 全深度验收

### w8-vdfp / 问候

>  My name is Alex.<end_of_utterance>

生成6个token，解码10.255 token/s，首token 1.450秒。
FP32参考（original权重）最高分token一致6/6，logits相对L2 20.116%。
加载 3.568 秒，进程 RSS 5.38 MiB，驱动分配 371.24 MiB，释放后 0 字节。RSS 不包含全部驱动内存，不能单报 RSS 充当模型内存。

### w8-vdfp / 陪伴

>  "Have a great day!".<end_of_utterance>

生成8个token，解码10.517 token/s，首token 2.366秒。
FP32参考（original权重）最高分token一致5/8，logits相对L2 28.219%。
加载 1.410 秒，进程 RSS 5.88 MiB，驱动分配 371.24 MiB，释放后 0 字节。RSS 不包含全部驱动内存，不能单报 RSS 充当模型内存。

### w8-residual / 陪伴

>  I understand how difficult it can be to feel lonely, and I want to help you feel more connected and supported.
>
> I would suggest that you reach out to your friends, family, or even a therapist for some support. They can provide you with a different perspective and help you find ways to make your life more fulfilling.
>
> I would also encourage you to take up a new hobby or activity that you have always wanted to try, such as learning a new language,

生成96个token，解码8.195 token/s，首token 2.355秒。
FP32参考（original权重）最高分token一致79/96，logits相对L2 29.111%。
加载 3.004 秒，进程 RSS 7.62 MiB，驱动分配 393.44 MiB，释放后 0 字节。RSS 不包含全部驱动内存，不能单报 RSS 充当模型内存。

相同实际导出权重（含所标注FP16例外）、FP32 激活参考：logits L2 28.266%，最低余弦 0.781473，最高分一致 78/96。

### w8-sensitive / 陪伴

>  I understand how challenging it can be to feel lonely, and I want to offer you a small comfort and suggestion.
>
>
>  I would be happy to help you feel more connected and supported.
>
>  I am here to support you and help you feel more at ease.
>
>
>  I am here to help you feel more supported and less alone.<end_of_utterance>

生成72个token，解码5.498 token/s，首token 3.970秒。
FP32参考（exported-int8权重）最高分token一致52/72，logits相对L2 33.619%。
加载 4.399 秒，进程 RSS 6.62 MiB，驱动分配 393.16 MiB，释放后 0 字节。RSS 不包含全部驱动内存，不能单报 RSS 充当模型内存。

相同实际导出权重（含所标注FP16例外）、FP32 激活参考：logits L2 33.619%，最低余弦 0.666281，最高分一致 52/72。

执行门槛 v1：FAIL。这是调试轨迹，不是独立留出质量验收。

### w8-wide / 陪伴

>  I understand how challenging your day was, and I want to offer you some comfort and support. Please know that I am here for you, and I want to help you feel better.
>
> One small thing we can do together is to reach out to someone you trust. You can talk to a close friend, family member, or even a therapist. Sometimes, just having someone to talk to can make a big difference in how you feel.
>
> If you're feeling down

生成96个token，解码5.838 token/s，首token 3.966秒。
FP32参考（exported-int8权重）最高分token一致86/96，logits相对L2 21.876%。
加载 0.743 秒，进程 RSS 7.88 MiB，驱动分配 393.16 MiB，释放后 0 字节。RSS 不包含全部驱动内存，不能单报 RSS 充当模型内存。

相同实际导出权重（含所标注FP16例外）、FP32 激活参考：logits L2 21.876%，最低余弦 0.890111，最高分一致 86/96。

执行门槛 v1：FAIL。这是调试轨迹，不是独立留出质量验收。

### w8-down-float / 陪伴

>  I understand how challenging your day was, and I want to help you feel better. One small thing we can do together is to take a walk together. It's a simple activity that can help you feel more connected to each other.<end_of_utterance>

生成48个token，解码5.200 token/s，首token 3.902秒。
FP32参考（exported-int8权重）最高分token一致39/48，logits相对L2 14.089%。
加载 0.822 秒，进程 RSS 6.62 MiB，驱动分配 393.16 MiB，释放后 0 字节。RSS 不包含全部驱动内存，不能单报 RSS 充当模型内存。

相同实际导出权重（含所标注FP16例外）、FP32 激活参考：logits L2 14.089%，最低余弦 0.975288，最高分一致 39/48。

执行门槛 v1：FAIL。这是调试轨迹，不是独立留出质量验收。

### w8-mixed-biasfix / 问候复测

>  My name is Alex.<end_of_utterance>

生成6个token，解码5.433 token/s，首token 2.608秒。
FP32参考（exported权重）最高分token一致6/6，logits相对L2 4.484%。
加载 0.289 秒，进程 RSS 5.38 MiB，驱动分配 408.34 MiB，释放后 0 字节。RSS 不包含全部驱动内存，不能单报 RSS 充当模型内存。

相同实际导出权重（含所标注FP16例外）、FP32 激活参考：logits L2 4.484%，最低余弦 0.997656，最高分一致 6/6。

执行门槛 v1：PASS。这是调试轨迹，不是独立留出质量验收。

### w8-mixed-biasfix / 陪伴

>  I'm so sorry you're feeling this way. It's okay to feel lonely sometimes, and there are many things we can do together to make each other feel better. One thing you could do is talk to your friend about how you're feeling. Sometimes, just sharing your feelings with someone who understands can help you feel better.<end_of_utterance>

生成67个token，解码5.393 token/s，首token 4.282秒。
FP32参考（original权重）最高分token一致61/67，logits相对L2 4.689%。
加载 0.295 秒，进程 RSS 7.88 MiB，驱动分配 408.34 MiB，释放后 0 字节。RSS 不包含全部驱动内存，不能单报 RSS 充当模型内存。

相同实际导出权重（含所标注FP16例外）、FP32 激活参考：logits L2 3.130%，最低余弦 0.998273，最高分一致 65/67。

执行门槛 v1：PASS。这是调试轨迹，不是独立留出质量验收。

### 权重与执行误差分离

独立主机 oracle 读取每层编译产物的实际 INT8 字节和尺度，使用 FP32 激活与 KV 可以生成连贯陪伴文本，原始 JSON 随证据归档。它不是 NPU 速度成绩。
w8-vdfp 使用 INT16 激活和 head；w8-residual 将残差与 head 改为 FP16；w8-sensitive 进一步将 RMSNorm 内部和 masked softmax 改为 FP16，同时保留 INT16 线性层边界。
层0原 masked score 的 INT16 fl=1：−10000 掩码占满动态范围，合法注意力分数步长变成 0.5。改用 FP16 可避免这项静态范围损失。未做单因素消融之前，不把组合修复的所有收益归给单一算子。

### 激活截断与大动态范围修复

独立逐层 trace 显示：w8-sensitive 的第一层个别位置误差超过50%；FP32 ONNX范围审计发现 MLP Norm 输出最大21.825，但INT16范围约±2，发生截断；attention输出最大1.314，范围约±0.5。旧校准仅抽取少量位置，未覆盖这些激活。
w8-wide 将非恒定INT16激活范围扩大16倍，第一层各位置误差降至约2%以内；但第4层和第32层的MLP包含上万的BOS离群值，扩大范围后down输出步长为8，普通token的小值又被舍弃。只将两处down输出改为FP16仍不足以解决MLP输入精度损失。
**当前有效配置 w8-mixed-biasfix**：第4/32层MLP保留FP16，其余decoder大矩阵W8，V用DFP，敏感Norm/softmax/残差用FP16，其余INT16激活保留范围余量，head FP16。224个decoder大矩阵中218个为INT8；按参数个数约95.31%为INT8，不包含embedding/head，不能称全模型纯Q8。
混合MLP第一次编译失败是沿用了整数校准的INT64 bias元数据；删除这三处浮点线性层的整数bias配置后，可由编译器按FP16路径推导并成功编译。该失败不能用来宣称SDK不支持所有混合精度。
当前只在128容量的开发样例通过；静态范围余量仍需更充分校准和独立留出集验证。保留FP16敏感MLP是精度修复，尚不是速度优化。

## CPU线程/绑核初筛（非最终目标线）

同一SmolVLM-500M Q8，固定生成128 token，一次预热加三次记录。自然EOS被关闭，不能以这些尾部文本评价对话质量。初筛未控制统一起始温度，最高接近89°C，不能据此完成公平性能验收；新版脚本已加入每次运行≤55°C的起始温度约束和频率记录。

| 线程 | CPU亲和性 | 中位decode token/s | 记录温度范围°C |
| --- | --- | ---: | ---: |
| 1 | 6 | 24.353 | 57.7–59.4 |
| 2 | 6,7 | 27.050 | 65.5–72.6 |
| 4 | 4-7 | 24.129 | 78.4–82.7 |
| 6 | 2-7 | 22.335 | 86.4–87.8 |
| 8 | 0-7 | 15.678 | 88.6–89.1 |
| 1 | 0-7 | 24.653 | 77.1–82.6 |
| 2 | 0-7 | 27.036 | 84.2–85.8 |
| 4 | 0-7 | 23.168 | 84.9–87.7 |
| 6 | 0-7 | 22.637 | 86.4–87.0 |

这里只覆盖现有Q8和线程/绑核，尚未覆盖其他量化、KV/attention参数及实现比较，不称“最好的CPU”。

## 修复配置重复检查

额外完成4次陪伴开发样例重复；与首次运行合计5次。所有已完成重复数值门槛：PASS。
这些运行写完整logits用于数值校验，不是最终干净性能矩阵，也不是30分钟稳定性验收。

## InternVL3-1B：完整输出头的独立问题

24层完整语言decoder与head已运行。未拆分的151674词输出头，仅前20602个logits非零，其余全部为0；20602 = 151674 mod 65536。证据指向大输出维度路径异常，封闭实现内部原因尚未确认。仅看前几个英文token会掩盖问题：连EOS也被置零，模型无法正常结束。
修复为4个NPU输出头图，每块最多49152词，CPU仅按词表顺序拼接并argmax；全部词表矩阵计算仍在NPU。规范化重复4次，未迁移到CPU。

### 未拆分（失败）

> Hello! I am Qwen, a powerful AI assistant created by Alphalab. I am here to help you with information, answer questions, and assist

logits相对L2 92.991%，预测最高分一致29/32；decode 2.025 token/s，首token 14.767秒，驱动分配 960.0 MiB。

### 分块修复FP16

> Hello! I am Qwen, a powerful AI assistant created by Alibaba Cloud.<|im_end|>

logits相对L2 0.454%，预测最高分一致17/17；decode 2.089 token/s，首token 14.672秒，驱动分配 960.6 MiB。

1B目前是FP16完整问候样例，不是INT8迁移完成，也未完成陪伴、长历史或CPU最优配置对比。不能用500M的混合精度成绩替代1B验收。

## 性能与验收边界

这些是容量128、短提示的调试运行；尚非五次固定128步、长历史或30分钟持续运行验收。旧 CPU Q8 约27.5 token/s仅为两大核旧参考，输出长度不同，且尚未完成 CPU 配置搜索。不得据此宣布 NPU 超过最优 CPU。
最终目标、CPU 搜索范围、冻结的执行与质量门槛见[完整模型验收计划](完整模型验收计划.md)。500M与InternVL3-1B必须分别验收。网页仍使用 CPU。

当前修复样例decode墙钟 12.237秒，其中decoder VIP调用 8.975秒、head VIP调用 2.390秒，合计占 92.9%。即使把这些调用以外的全部开销降到零，上限也只有约 5.81 token/s。
VIP调用墙钟包含驱动提交/等待，不等于NPU算术核心利用率。此结果说明仅优化宿主RoPE、mask、日志不足以追上现有CPU，后续必须针对图/内核/搬运优化；不能用此理论上限冒充已实现速度。

## 实现与复现

`compile_weight_int8.py --value-dfp`沿new_v图依赖查找V的实际fullconnect名称，避免把旧图节点编号硬编码到所有模型。
`--only-linear`提供消融，`--weight-mode dfp`提供独立量化形式对照。
`run_board_probe.py --trace`额外保留第0步两个层的hidden/K/V，正常速度测试默认不启用trace。
`compile_repaired_stack.py`将修正应用于32层；`--fp16-head`、`--float-residual`、`--float-sensitive`控制精度。不称作全模型所有张量均INT8。
`validate_full.py --reference-weights exported-int8`区分实际W8执行误差与原GGUF权重量化差异；`run_full_model.py --trace`可保存全部位置逐层hidden，诊断时间不得冒充干净性能测试。
扩全深度使用含BOS/对话特殊token的校准和稳定RMSNorm版本，不能把不同校准的全模型数字与本消融当成严格单变量对比。
