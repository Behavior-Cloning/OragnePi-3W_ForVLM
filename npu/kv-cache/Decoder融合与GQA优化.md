# Decoder 融合与 GQA 布局优化

2026-10-12。完整语言模型、W8为主的相同混合精度、容量128、增量KV；CPU基准仍为对应Q8。

这是继输出头和批量提交优化后的新实验。所有速度先区分诊断运行与受控重复测速，未完成的部分不算验收通过。

## 实现

- 将相邻两个已校准decoder图合为一个图，移除中间hidden输入/输出边界，保留原权重与每个张量的量化参数。500M为16个双层图，1B为12个双层图，仍执行全部语言层和输出头。
- 每层仍保留独立KV、mask、RoPE端口，不假定各层KV量化scale相同。ACUITY会按节点编号重排接口，脚本从导出的C代码核对真实顺序，运行器按K分组、V分组的接口绑定。
- GQA版本把Q整理为`[batch, KV头数, 每组Q头数, head_dim]`，直接与未展开的K/V相乘，移除重复K/V用的两组Tile/Reshape。500M去掉3倍展开，1B去掉7倍展开。权重位宽、KV逻辑、mask和RoPE不变；需要重新验证数值和实际性能。
- 原逐层图、批量提交路径继续保留。融合后不提供虚构的逐层耗时，记录块耗时；逐层hidden trace请求会明确拒绝。

另有等价贪心采样优化：每片输出头在INT16中找最大值，只转换每片最大值后按真实scale比较；并列时仍选最小token编号。对不支持的输出类型回退浮点扫描。诊断模式同时执行完整浮点argmax逐步核对；性能模式省去整词表转换。它仅适用于本轮greedy生成，不是top-p/temperature采样实现。板上NEON和主机标量测试覆盖负数、极值、向量尾部及并列最大值。

## 完整输出验证

| 模型 | 变体 | 输入 | 输出token数 | 与原版本token一致 | 完整logits逐字节一致 | 对编译常量FP32执行门槛 |
| --- | --- | --- | ---: | --- | --- | --- |
| 500M | w8-fused2 | greeting | 6 | True | True | 通过（L2 4.491%） |
| 500M | w8-fused2 | companion | 67 | True | True | 与原通过样例逐值一致 |
| 500M | w8-fused2-gqa | greeting | 6 | True | True | 与原通过样例逐值一致 |
| 500M | w8-fused2-gqa | companion | 67 | True | True | 通过（L2 3.132%） |
| 1B | w8-fused2 | greeting | 15 | True | True | 与原通过样例逐值一致 |
| 1B | w8-fused2 | companion | 64 | True | True | 通过（L2 3.435%） |
| 1B | w8-fused2-gqa | greeting | 15 | True | True | 与原通过样例逐值一致 |
| 1B | w8-fused2-gqa | companion | 64 | True | True | 通过（L2 3.435%） |

问候和陪伴均为开发样例；1B陪伴达到64-token上限，不是自然EOS完整回答。逐字节一致证明该样例没有新增差异，不代表独立留出或长历史已验收。

## 诊断记录（非受控测速）

| 模型 | 变体 | 输入 | token/s | 首字秒 | 驱动MiB | 起始温度°C |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 500M | w8-fused2 | greeting | 7.217 | 2.314 | 349.65 | 64.8 |
| 500M | w8-fused2 | companion | 7.089 | 3.846 | 349.65 | 64.2 |
| 500M | w8-fused2-gqa | greeting | 12.035 | 1.286 | 338.17 | 66.1 |
| 500M | w8-fused2-gqa | companion | 12.055 | 2.132 | 338.17 | 63.9 |
| 1B | w8-fused2 | greeting | 6.659 | 4.369 | 469.98 | 68.1 |
| 1B | w8-fused2 | companion | 6.439 | 5.468 | 469.98 | 64.8 |
| 1B | w8-fused2-gqa | greeting | 9.023 | 2.654 | 460.32 | 67.1 |
| 1B | w8-fused2-gqa | companion | 9.834 | 3.299 | 460.32 | 64.2 |

以上运行包含logits写盘，部分起始温度高于此前55°C控制线，不能与上轮中位成绩相除宣布提速。每次驱动释放状态见原始JSON。

## 重复性能对照

### smol500-fusion-performance.json

状态：failed。已记录2次（包含诊断/预热）。

起始温度上限：55.0°C。沿用原55°C协议。

### smol500-gqa-performance.json

状态：complete。已记录14次（包含诊断/预热）。

起始温度上限：68.0°C。补充对照：原版与新版均在当前散热条件下重新测量，不能与此前55°C成绩混算；未替代原条件验收。

| 变体 | 中位token/s（范围） | 首字秒 | 总响应秒 | CPU秒 | 驱动MiB | 起温范围°C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| grouped | 7.093（7.092–7.100） | 3.865 | 13.169 | 0.624 | 353.51 | 59.9–61.3 |
| fused-gqa | 12.011（11.984–12.017） | 2.132 | 7.627 | 0.550 | 338.17 | 59.9–60.9 |

相对本轮原版：decode中位提升69.3%；同轮5/5次更快；首字缩短44.8%。这不是相对CPU的加速比。
NPU开始/结束频率记录：[1008000000] Hz；CPU大核开始/结束频率、每次时延和温度见JSON。这是端点采样，不冒充全程连续频率曲线。

### intern1b-gqa-performance.json

状态：complete。已记录14次（包含诊断/预热）。

起始温度上限：68.0°C。补充对照：原版与新版均在当前散热条件下重新测量，不能与此前55°C成绩混算；未替代原条件验收。

| 变体 | 中位token/s（范围） | 首字秒 | 总响应秒 | CPU秒 | 驱动MiB | 起温范围°C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| grouped | 6.741（6.355–6.761） | 5.494 | 14.840 | 1.330 | 472.71 | 58.4–61.4 |
| fused-gqa | 9.916（9.355–10.006） | 3.291 | 9.645 | 1.112 | 460.32 | 58.1–61.0 |

相对本轮原版：decode中位提升47.1%；同轮5/5次更快；首字缩短40.1%。这不是相对CPU的加速比。
NPU开始/结束频率记录：[1008000000] Hz；CPU大核开始/结束频率、每次时延和温度见JSON。这是端点采样，不冒充全程连续频率曲线。

### smol500-greedy-performance.json

状态：complete。已记录14次（包含诊断/预热）。

起始温度上限：68.0°C。补充对照：原版与新版均在当前散热条件下重新测量，不能与此前55°C成绩混算；未替代原条件验收。

| 变体 | 中位token/s（范围） | 首字秒 | 总响应秒 | CPU秒 | 驱动MiB | 起温范围°C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float-sample | 12.049（11.046–12.061） | 2.127 | 7.605 | 0.538 | 338.17 | 62.6–63.5 |
| integer-sample | 12.956（12.795–12.981） | 2.122 | 7.216 | 0.139 | 338.17 | 62.2–63.1 |

相对本轮原版：decode中位提升7.5%；同轮5/5次更快；首字缩短0.2%。这不是相对CPU的加速比。
NPU开始/结束频率记录：[1008000000] Hz；CPU大核开始/结束频率、每次时延和温度见JSON。这是端点采样，不冒充全程连续频率曲线。
推理结束、释放NPU前的进程VmRSS范围：float-sample 7.00–7.25 MiB；integer-sample 6.88–7.12 MiB。设备驱动分配另列，两者不能简单相加当作整机内存峰值。

### intern1b-greedy-performance.json

状态：complete。已记录14次（包含诊断/预热）。

起始温度上限：68.0°C。补充对照：原版与新版均在当前散热条件下重新测量，不能与此前55°C成绩混算；未替代原条件验收。

| 变体 | 中位token/s（范围） | 首字秒 | 总响应秒 | CPU秒 | 驱动MiB | 起温范围°C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float-sample | 9.914（9.174–9.936） | 3.291 | 9.649 | 1.103 | 460.32 | 60.3–62.4 |
| integer-sample | 11.514（10.891–11.619） | 3.273 | 8.745 | 0.193 | 460.32 | 60.9–62.4 |

相对本轮原版：decode中位提升16.1%；同轮5/5次更快；首字缩短0.6%。这不是相对CPU的加速比。
NPU开始/结束频率记录：[1008000000] Hz；CPU大核开始/结束频率、每次时延和温度见JSON。这是端点采样，不冒充全程连续频率曲线。
推理结束、释放NPU前的进程VmRSS范围：float-sample 7.62–7.75 MiB；integer-sample 7.00–7.12 MiB。设备驱动分配另列，两者不能简单相加当作整机内存峰值。

受控脚本先验证相同提示词、token和全部logits，再各预热一次、交替至少五次测量。55°C原条件试跑等待300秒后仍未达标，运行器报`cooldown timeout`，该次失败保留。为继续研发，另建起温上限68°C的补充实验，默认55°C协议仍保留，未把补充实验当作原条件验收；正式CPU/NPU验收需重新统一温度条件。

早期重复脚本的`max_rss_kib`可能带入启动器校验大文件时的fork内存高水位，不能解释为模型自身RSS。本轮后续改用分块SHA256，并新增读取`/proc/self/status`的推理结束VmRSS；驱动分配始终独立记录。

## 实际生成样例

### 500M 紧凑GQA完整语言链路

 I'm so sorry you're feeling this way. It's okay to feel lonely sometimes, and there are many things we can do together to make each other feel better. One thing you could do is talk to your friend about how you're feeling. Sometimes, just sharing your feelings with someone who understands can help you feel better.<end_of_utterance>

与原版输出逐值一致。此回答自然EOS结束。

### 1B 紧凑GQA完整语言链路

I'm here to help you feel better, so I'll offer you two small comforting thoughts and a simple thing you can do to help lift your spirits tonight.

**Small Comforting Thought:**
"Remember, you are not alone. There are people who care about you and want to help you. Sometimes, just being

与原版输出逐值一致。此回答达到64-token上限，显示为截断文本。

## 尚未完成

独立留出和长历史验收、同轨迹最优CPU Q8比较、批量prefill、容量分档、30分钟稳定性仍未完成。当前结果不支持宣称NPU超过CPU。

复现入口：`compile_fused_stack.py`、`run_full_model.py --fusion-width 2`、`validate_full.py`、`benchmark_variants.py`；GQA编译增加`--compact-gqa`。模型/SDK/NBG保持在Git忽略目录，源码、编译参数来源哈希、验证结果与性能JSON同仓归档。
