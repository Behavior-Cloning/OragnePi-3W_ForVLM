# 中文 Llama / Intern 量化比较独立审计与协议

2026-10-23。只读核对 Windows/WSL 源码、GGUF 元数据、编译配置与已存板测 provenance；未访问板卡，未加载模型推理或启动编译。本文件是比较协议，不表示完整 Llama NPU 验证完成。

## 结论

Q4、Q8源均可测，但当前 Llama 是 Q4_K_M/Q8_0源→attention FP16 + MLP W8/A16 + FP16完整词表头。**不是原生NPU INT4，也尚非与Intern近似相同的NPU量化范围。** Q8来源对齐仅解决源文件精度差异。

Intern的1B是VLM命名；实际text decoder约629.70M独立参数，Llama约1235.81M。Llama虽然16层少于Intern24层，每层更宽，不能由层数直接推断更快。

## 结构与参数

|项目|InternVL3-1B text decoder|FuseChat-Llama-3.2-1B-Instruct|
|---|---:|---:|
|语言架构|Qwen2|Llama|
|层数|24|16|
|hidden / FFN|896 / 4864|2048 / 8192|
|Q / KV heads / head dim|14 / 2 / 64|32 / 8 / 64|
|完整词表|151674|128256|
|attention矩阵参数|44,040,192|167,772,160|
|MLP矩阵参数|313,786,368|805,306,368|
|输出头矩阵参数|135,899,904|262,668,288|
|输入embedding参数|135,899,904|与输出头共享同一源矩阵|
|独立模型参数|629,697,920|1,235,814,400|
|当前KV capacity|1280|256（首版）|
|prefill实现|NPU批16+标量尾部|首版NPU逐token|

来源：官方Intern config的llm_config、gguf-fresh/export/trained-manifest.json；Llama GGUF tensor shape/type及本目录model.py/trained_probe.py。仅mmap读取元数据，没有解码大权重。Intern包含层内bias/norm共70,656及最终norm896参数；Llama norm共67,584。Llama GGUF还含32个RoPE调频常量，tensor元素和1,235,814,432，不能把常量当新增可训练参数。Intern视觉模块不在本次text范围。

Llama源只有token_embd.weight、无独立output.weight，HeadShard复用其源矩阵；运行时仍保留CPU FP32 embedding mmap和NPU head副本，独立参数共享不等于运行内存共享。

## NPU量化范围

统一分母：全部decoder FC矩阵加一套完整输出头矩阵；排除lookup、bias/norm、激活、KV和prefill重复图。下表是权重元素比例，不是内存或耗时比例。

|当前配置|W8主FC参数|FP16主FC参数|W8占比|FP16占比|
|---|---:|---:|---:|---:|
|Intern现有板上路线|485,010,176|8,716,288|98.2346%|1.7654%|
|Llama当前混合构建目标|805,306,368|430,440,448|65.1676%|34.8324%|

Intern decoder168个FC中166个i8；真实第2/3层（零起始）MLP down矩阵FP16，每个4864×896。对应融合图l2的b0_fullconnect_116和b1_fullconnect_116，**不是attention例外**。decode和批16 prefill清单一致。四个输出头全i8，行数49152+49152+49152+4218，完整覆盖词表。decoder hidden为FP16、KV与主要边界为i16/DFP，部分非线性/残差FP16，输出头logits配置i16；W8不代表全部激活8位。

Llama每层4个attention矩阵FP16、3个MLP矩阵W8，FC附近A16校准，非线性及hidden/KV接口FP16；最终RMSNorm与4×32064全词表头FP16。比例来自当前结构和编译规则，20图完成后仍须逐图检查实际quantize、生成代码、板上IO与SHA，不能将构建目标当执行证据。

复算：Llama W8=16×3×2048×8192；FP16=16×(2×2048²+2×2048×512)+128256×2048。Intern decoder FC=24×(2×896²+2×896×128+3×896×4864)，扣2×4864×896得到decoder W8，再加151674×896的W8头。

Intern来源GGUF Q8_0 SHA：e87f98e2945c91da1a324034143f2f3bad014dcf6b34d440fcf885c00704378f。Llama Q4_K_M SHA：8bcd46580c2eb069fde82b4c16f8f2981901d653fea1340ec560b6097f08f97a。Q4_K_M实含Q4_K、Q6_K和FP32小向量，embedding为Q6_K，不能说全部源权重4位。Q8完成后补实际验收SHA、独立校准和构建记录。

## 证据链

WSL基路径：/opt/opi-lab/source-validation-20261017/gguf-fresh/export。

- decode：w8-fused2-c1280-cap1-headmse/l{0,2,...,22}/decoder.{json,quantize}。
- prefill：w8-fused2-c1280-t16-cap1-productwide/l{0,2,...,22}/decoder.{json,quantize}。
- 实际输出头：w8-fused2-c1280-cap1-headmse/l{24,25,26,27}。**不要用w8-balanced/l24..27替代**；它们虽也W8，NBG SHA却与板上不同。
- 24个decoder/prefill当前quantize SHA与peak-155-20261018/build-state.json逐项一致；该状态的NBG SHA与research/intern1b-operator-opt-20261023/results/20261023T064955/events.jsonl板测provenance逐项一致。
- 四个实际头当前NBG SHA与同一provenance逐项一致，重新读取quantize确认全部i8。
- 头NBG SHA依次为5809759670990c046d874d65312f983895122b7996a4d9f2af2cacd07ec5d896；c3d8fd2b1b9ffad1acd5d383e221db5650a56e633a0be5c09a3a3403c25abc86；336bebcee5bba8592518347d93c2b6596125b94730e030276c69720a6eff87c4；a23c5ea9aad93532cf5900d9b67a3a55dd1d7c4c425059df3e5551c02e7f2aa2。

未逐个还原tiny constant/bias的独立参数存储比例，也未重新证明每个算子的内部accumulator精度；这些项目未知，不能从qtype补猜。

## 同中文输入与缓存

统一使用当前Llama system：**你是家庭陪伴机器人，请用简短中文回答。** 各模型保留原生chat模板。独立使用官方Intern tokenizer.json编码，该完整system恰为16 tokens，对齐Intern批16，无需旧37-token前缀的32/37缓存折中。Llama相同system为22 tokens。

|题目|Intern system/动态/总数|Llama system/动态/总数|
|---|---:|---:|
|你好，我今天有点累。|16 / 15 / 31|22 / 16 / 38|
|明早七点提醒我起床。|16 / 16 / 32|22 / 19 / 41|
|不要开灯，请陪我聊两句。|16 / 17 / 33|22 / 20 / 42|

这是同语义应用比较，tokenizer差异是真实成本。长度控制microbenchmark可独立分表，但不可改成不同内容后称同问题。旧Intern system37、旧Llama CPU缓存26、输入48–50的成绩不能直接作此协议配对结果。

模型常驻并预热一次；每问从只含system的快照恢复，禁止额外命中旧user/assistant/整问缓存。记录requested_prefix、actual_reused、dynamic_processed、total_input。无system缓存与完整system缓存分组；冷加载及建立system缓存成本单列。

当前KV capacity1280对256、融合2层+批16对单层+标量prefill是工程差异。首轮可作“当前可部署实现”比较，必须显示这些差异；要归因于模型或量化本身，再做相同capacity/相当prefill策略控制实验。不得用历史不同问题的0.8秒对比新问题的更低值，声称全部来自算子加速。

## 最小最终测试集合

1. 三臂：Intern现有Q8源W8、Llama Q4源混合、Llama Q8源相同混合。Llama仅来源变化，其余校准文本/位置、图形状、规则、runtime相同；逐层校准状态由每源自己的FP32反量化oracle传播。Q4/Q8的source、calibration、oracle、graph SHA和log独立保存。owner已告知后续修复source级EVIDENCE隔离，运行中旧Q4证据仍需归档。
2. 功能优先：上述3问自由greedy生成，遵循原生EOS，至少允许64新tokens（当前控制器max96合适），保留tokenIDs、全文及截断标记。再加3个独立家庭问题覆盖名字记忆、数字时间、否定，冻结预期要点。逐题看遵循、幻称工具已执行、否定、重复、中文与截断，不用跨模型逐token一致或“文字通顺”替代正确性。6题仅smoke，不能称广泛聊天准确率。
3. 数值诊断单列：每模型用自己的FP32源oracle固定反馈，3问至少首8步全词表logits，记录相对L2、KL、argmax、top1/top2 margin；不跨词表比较logits。该诊断不纳入计时。自由生成失败则逐层定位，不把单层5%误差粗筛当全模型精度验收。Q4/Q8源相对原BF16的损失仍未知。
4. 性能：每臂3问×5次×2缓存状态=30次，三臂90次。固定32步用于吞吐时明确标记忽略EOS/固定反馈，不能当自然对话。交错会话顺序，统一CPU亲和、并发服务、起始温度及输出长度，记录CPU/NPU频率。逐题中位数和范围；15样本的P95仅探索统计。
5. 指标：输入提交→首token TTFT（runtime和控制端时间分别给，不把SSH建立/加载混入常驻TTFT），prefill/head/宿主分解；第2..32个token吞吐=31/(t32-t1)；自然完整回答耗时、生成字符数另列。tokens/s跨tokenizer不等于等量中文字符/秒。连续记录进程CPU秒/墙钟（核数）、系统CPU、RSS/PSS、NPU driver bytes、MemAvailable、swap in/out、温度频率和关闭后driver归零。RSS不代表全部NPU模型占用。
6. 通过短问后，追加64/128动态tokens附近的家庭记忆+新问，每档3个同语义样例×3次/臂，逐模型报告真实token数。不得用重复前缀缓存或人为pad代表真实场景。256动态tokens超过当前Llama capacity256可容纳的system+输出空间，必须先另建容量才能测试。

## 同等量化的验收边界

当前W8比例65.17%对98.23%只能作为各自混合实现的可用性/性能对照，不能称近似相同量化。下一步验证Llama W8全词表头；即使只有头变W8，主FC W8也仅约86.42%，仍保留全部attention FP16。再尝试Q8源W8 attention和少量敏感FP16例外，以完整生成质量为准，不为凑百分比牺牲答案。

若始终不能得到质量可用且范围接近的Llama量化，应结论为“未取得可用的同等量化比较”，给带限制的真实路线对照。反向提高Intern FP16比例可作额外控制，但不应隐藏现有更优可用基线。全FP16是诊断参照；原生packed INT4仍是独立未证明项。当前协议不支持替换Intern服务的结论。
