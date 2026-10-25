# Prefill100 既有证据、计时契约与成本审阅

状态：仅完成证据审查与验收设计，100ms目标未达成。Windows/WSL为权威代码和证据来源；本审计没有访问板卡、修改模型或启动试验。下面把“已经实测”“基于参数的算术”“待测路线”分开，未引用未经验证的NPU TOPS数字。

## 实际模型与执行范围

`../minicpm-v46-npu-20261023/results/intern-architecture.json`来自现役 `InternVL3-1B-Instruct-Q8_0.gguf`：Qwen2语言结构24层、D=896、FFN=4864、14个query head、2个KV head、head_dim=64、rope_base=1e6；GGUF记录629,697,920个tensor元素。这个记录针对语言权重，不能把“InternVL3-1B”产品名理解为本次跑了完整视觉模型或恰好10亿个语言参数。

现役runtime构建/ready交叉确认24层、fused2即12个decoder块、完整词表151674、4个head分片、embedding为151674×896个float32（543,599,616字节）。它是板卡A733上的Intern NPU文本路径；没有视觉编码器。语音模型CPU常驻和RAG常驻不等于每次推理都发生连续RAG并发。

来源文件SHA256：

|文件|SHA256|
|---|---|
|intern-architecture.json|e2524d5503ed5c3e1689bc858138ec18ced48444c9425794cccd46053e6468d7|
|runtime-combined-v2/resident_session.c|efe0879dcc90d6a223f0f5763f754f7372200f365e630858be4b723100f5747a|
|runtime-combined-v2/batch_prefill.h|f80acdcb0c4574546bb9716cde040f9c069d2e4f7245263bb950c6d113aee094|

结构记录还由 `../../npu/kv-cache/evidence-large-20261012/intern1b-c128-manifest.json`交叉支持；该早期文件仅测2层，不能拿其scope冒充完整模型性能。当前完整模型证据来自nextbench完整运行的ready与结果。

## 原始 plain 到当前版本：独立复核

直接读取以下两组原始文件，未仅转述汇总：

- 原始：`../async-agent-20261025/results/ttft-aligned-matrix-v2/{events.jsonl,fixtures.json}`，measure.variant=plain。
- 当前：`../intern-nextbench-20261025/results/combined-matrix-r2/{events.jsonl,fixtures.json}`，measure.variant=combined-tail8。

两份fixture的`systems`和`performance`对象逐项完全相同。共有24个case+repeat的prompt/new/reuse计数一致，首个预测token24/24一致。原plain每格3场景×3次=9次，仅输出1token；当前每格3场景×2次=6次，固定反馈输出32token。固定反馈从第二步才影响输入，首token所见prompt相同，所以热态TTFT可比较；但属于不同时间的短样本跨轮测量。原plain没有第二个token，**不能从它得到decode速度比较**。

完整prefill必须对每次先计算 `prefill_seconds + scalar_prompt_seconds`，再取P50；下面已逐请求独立重算，不是将两个P50相加。

|system总token / 动态正文token|实际新增|原plain完整prefill P50|当前完整prefill P50|原plain TTFT P50|当前TTFT P50|
|---|---:|---:|---:|---:|---:|
|64 / 32|40|1.004744s|0.4752245s|1.152739s|0.493827s|
|64 / 64|72|1.635486s|0.8508000s|1.759670s|0.869842s|
|128 / 32|40|1.003112s|0.4756425s|1.180238s|0.4948145s|
|128 / 64|72|1.626598s|0.8705165s|1.798044s|0.8904535s|

原始events SHA256：`7d8f315473d400a814c1e54e45f97e24af0cd8416e64a0ac58014a07af4c0922`。当前events SHA256：`c9ddb8d10206d45e7bd5efa8a7cc180ef6bdc731f5242f47cef61482b1db9dc5`。

另一个基线 `candidate-matrix-r3/mask-base`已含grouped prefill、mask template及tail_min8宿主优化，不能称为最早plain。它与当前版本均固定反馈32，适合比较decode（约13.66～13.89→15.29～15.50 tok/s）。比较NPU驱动占用时，C1280 mask-base 718.38→当前C256+true8 942.35MiB，增加223.97MiB；而“增加275.94MiB”是相对同C256无true8的666.41MiB，不能混用参照。

本轮新目标正文64～256。上表正文32仅是历史背景，不属于目标主范围；正文64当前TTFT仍0.870～0.890秒，距100ms至少约8.7～8.9倍。正文128/192/256的完整同模型新矩阵尚未实测。不能把正文64的结果外推为整个范围已通过，也不能继续用C256跑最大输入。

## 计时与缓存：已核实的陷阱

旧 `resident_session.c` 在读完/校验token请求后开始计时（226行），TTFT到首个head/argmax结果产生（268～272行）。包含setup、session重置/截断、动态prefill、首head；不含调用者tokenization和输入管道传输。新的报告必须另列caller观测时间，不能把它说成用户说完到首token100ms。

旧 `prefill_seconds`只累积batch调用（245/253/259行），scalar prompt另计（264行）。只显示前者会让batch边界剩余token的成本消失；完整prefill应逐次加上scalar。头部约17.7～18.9ms（正文64当前两格P50），因此达到TTFT100ms不能把100ms全部预算都留给prefill。剩余预算还包含setup、同步和发布工作；分项P50不具有逐请求可加性。

旧reuse默认按主batch倍数下取整（233行）。直接将batch16改为128/256，会把system64或system128的复用下取整到0。runtime agent当前设计独立M16 system bank、协议v2显式prefix_tokens上限，不扩大system，也不启用未经验证EXACT_PREFIX捷径。必须在端侧断言actual_reused=64/128，不能仅从配置推断缓存已保留。

新最大输入为128+256+8=392 token，固定32输出缓存峰值423（input+output-1）。C512可容纳，C256不合法。静态system预热允许，动态摘要提前prefill、重用前一轮同摘要KV、减少正文或层数都改变了目标，不可当原目标优化。

## 可复算的计算成本，不是芯片吞吐承诺

按上述真实语言结构，一个token的每层dense投影MAC为：

`Q/O: 2*896^2；K/V: 2*896*(2*64)；MLP gate/up/down: 3*896*4864`

合计每层14,909,440 MAC，24层357,826,560 MAC/token。首个完整词表head另需896×151674=135,899,904 MAC。embedding查表不按dense投影计。

|摘要正文D|实际新增N=D+8|24层dense投影 GMAC|加一次完整首head GMAC|
|---:|---:|---:|---:|
|64|72|25.7635|25.8994|
|128|136|48.6644|48.8003|
|192|200|71.5653|71.7012|
|256|264|94.4662|94.6021|

这是标准dense执行的算术工作量账本，不是板卡实测TOPS，也不是预测延迟。没有包含QK/AV注意力、softmax、norm、rope、反量化、布局变换、padding、KV/CPU搬运及SDK调用。仅逻辑因果注意力QK+AV可按 `2*D_model*24*(N*S + N*(N+1)/2)` MAC继续估计；编译图实际按固定C/M、mask和padding计算，其成本可能更大，须从图与端侧profile确认。grouped调度不使24层依赖链变成并行层，也不会消除这些投影。

若仅将上述dense+首head工作塞进100ms，所需有效速率约259～946 GMAC/s，且尚未留出其余算子的预算。这只是由模型维度和目标时间相除得到的必要计算吞吐账本；**不能据此断言硬件一定能达到，或引用未经证实的峰值反推100ms可行**。当前缺少该板在这些具体M/C/量化图上的稳定有效吞吐证据。

## 优先路线与风险

1. **扩大动态batch，先测M16/32/64/128/256、C512的L0/L2。** 当前正文64含72新增token，仍有多次16行完整层栈遍历；大M有机会降低重复权重读取和提交次数、改善矩阵利用率。但它不减少模型理论工作量，且较大attention mask/激活和编译内存可能抵消收益。单块只用于筛选，需要12块完整模型实测才有TTFT结论。
2. **保留M16 system bank，并显式选择动态bulk/tail bank。** 先验证prefix复用、pos/mask/KV和last-row；根据真实新增token选批，避免264 token误写成“单M256即可”。尾余数需要另处理，所有成本纳入计时。
3. **用阶段profile定位搬运/准备/提交成本。** runtime提议按base/bulk/tail记录host_prepare/npu_run/host_publish/read_last，完整prefill墙钟交叉对齐。诊断profile与正式速度分离；测完整完成而非异步enqueue。CPU减少不足以证明NPU算子本体也提速。
4. **同权重shape变化仍做数值回归。** 当前C256对C1280完整logits相对L2约2.926%～3.261%，虽小fixture token相同，不能称数学无损。更大M/C也要配对自然输出和full-logits。
5. **量化、近似或不同模型仅作为明确分支。** W4/INT4、激活/KV精度及校准改变必须同时报告质量和性能；剪层、缩小模型、正文压短可以讨论，但不是“不变模型、64～256正文100ms”目标已达成。

## 同其他agent约定的交付证据

graphs agent：`graphs/`下source-provenance/plan/status/deployment-manifest；现役calibrated fused2源，C512×M16/32/64/128/256，L0/L2，11输入5输出。初始声明量化payload不变、仅C/M整数维度改变；这仍是交付契约，须在产物出现后独立核SHA。

runtime agent：独立M16 system bank，协议v2 prefix_tokens，动态bulk32/64/128/256，M16 padded8尾路径、M8可选；ready包含capacity512、loaded_bank_tokens与full_depth24，probe明确scope=two_layer_probe。预期prefillext含host_prepare_s/npu_run_s/host_publish_s/read_last_s，并增加含scalar的dynamic_prefill_seconds。以上候选尚未被本审计宣告端侧通过。

保留关闭风险：原combined-r1曾在close时double free/corruption；调查未复现，v2三最小循环及完整短矩阵正常释放，根因仍未知。大batch/多bank会改变内存生命周期与占用，必须继续逐worker正常closed/driver0与稳定性回归。当前板盘余量约1.27GB是上次post-health，不能作为此刻实时空间保证；root部署前需查。
