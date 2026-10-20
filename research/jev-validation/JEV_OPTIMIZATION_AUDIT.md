# Jev 逐模型提速审计与验收 TODO

新增要求日期：**2026-10-21**。用户硬要求：**每个 Jev 都要实际尝试优化延迟和决策速度，不能只下载、部署或测基线。** 本文是该要求的验收清单，结果以 Windows F 盘原始证据为准。用户新授权替代旧计划“暂不下载或测速”的历史限制；不包含后训练或实体动作。

权威目录：`F:/Files/OPi-Lab/repos/OragnePi-3W_ForVLM/research/jev-validation`。关联：[验证计划](JEV_VALIDATION_PLAN.md)、[动作词典](ACTION_DICTIONARY.md)、[运行时源码来源](sources/runtime-source-provenance.json)。本审计智能体只维护本文，没有访问板卡、占锁或触发负载。

## 1. 状态与完成定义

| 模型 | 创建本文时的准备状态 | 优化验收状态 |
| --- | --- | --- |
| Nano33，sdmlai/nano-jev，约33M | Windows有下载清单；已取得部分screen证据，中文direct100为0/109；见下方修订 | **TODO：未完成优化** |
| Laya multilingual，约322M | 本轮短基线JSONL已完整回收：direct100为6/20，分层5/20；ORT桌面形状检查完成，ARM优化仍待测 | **TODO：未完成优化** |
| Jev-Style 0.8B Q4_K_M | ARM64 scorer编译成功；曾发现传递依赖libggml加载失败，修复后的真实评分/基线须补测 | **TODO：未完成优化** |
| OneJev1.6-0.8B多模态 | Windows有下载清单；不等于权重完整或板端视觉前向成功 | **TODO：多模态部署、基线、优化均未验收** |

排队、开始运行和编译成功均不等于通过。单模型须有板端离线基线、实际优化尝试、同输入优化前后配对、准确率核查、资源与共驻证据，才能关闭本轮优化TODO。实际尝试无加速或发生精度回退，也应报告，但不能写成已实现提速。某模型不推荐，也不能省略它的优化尝试与失败原因。

当前初筛脚本为 [board_optimize.py](board_optimize.py)、[run_board_optimization.py](run_board_optimization.py)：三款文本模型A76双线程/单线程、A55双线程；Nano/Laya动态INT8；Nano batch1对8；JevStyle seq slots17→2及ubatch1024→256。现调整为每格3次预热、20次计时、固定8例direct100质量初筛；原方案为30次计时/20例质量检查。**这是初筛，不是完整验收**；尚缺OneJev、A55单线程、完整状态长度/候选策略矩阵和正式样本量。

### 2026-10-21 初筛调整与已取得证据

已回读 [Nano部分screen原始记录](results/board/nano33-screen-t2-20261020T160914-6913.jsonl)：10候选200次，P50 **1.441117s**、P95约**1.452s**；direct100已跑109例，P50 **14.206490s**，正确率**0/109**，其中完整literal为**0/100**。原生英文相关性正负对照2例均正确；只能说明该基本评分链路可工作，不能证明中文动作迁移有效。采样观测RSS最高485707776字节（约486MB/463MiB），最高传感器温度72.334℃。v1计时限制仍适用。

主任务为避免在已明显失效的中文任务上反复运行超长矩阵，主动终止自己的Nano原始screen并回收部分证据。JSONL只有loaded/native_control/measure/accuracy事件，**没有complete/telemetry尾记录**；不能称全量screen或长时稳定性通过。后续仍须实际尝试优化，但缩短为3次预热、20次延迟、8例direct100质量初筛，不再用初筛样本量冒充正式验收。

Nano增加ORT FP32/INT8对照：主任务报告FP32导出最大误差8.94e-7，INT8误差0.106，均是导出/桌面一致性信息，**板端速度、精度和实际收益待测**。INT8数值误差不能直接换算动作正确率或速度。Laya现采用encoder-only Linear动态INT8，决策头保留FP32，已记录quantized_modules；仍待本板前向与正确率核查。

新脚本加入78℃停止保护。只读复核已修正为所有thermal_zone传感器最大值，并在计时和质量循环每次推理前检查；板端实际触发/清理效果待验收。该阈值是本轮实验保护配置，不代表板卡官方工作温度上限。

8例质量集已修正并只读核验为2literal+2paraphrase+2unknown+2safety；覆盖修复不等于精度验收通过，每类仅2例仍是初筛。

Nano中文失败已获得分词证据：[chinese-tokenizer-audit.json](results/chinese-tokenizer-audit.json)。100个动作描述仅34种不同token序列，UNK占86.57%，大量不同动作（含左右）编码完全相同；Laya/JevStyle均100/100 distinct、UNK为0。该信息丢失不能靠线程、量化或ORT修复。Nano仍完成速度优化对照，但**不推荐当前模型执行中文动作选择**，英文RAG用途须独立评估。不能把分词修复与运行时提速混为一项，也不能仅更换tokenizer而不验证对应权重。

### 2026-10-21 再次确认：每个模型必须有实际优化证据

用户再次强调逐个Jev优化延迟与决策速度。后续验收不能只列下载、启动、部署或汇总一个最快数值；原生基线、核线程、运行时/量化、候选缓存/批次、准确率、资源、共驻需分别可追溯。主任务报告控制器4805执行Laya/JevStyle短筛和共驻、4806排队优化；这些PID仅为当时调度记录，恢复任务应重新核对进程与锁，不能据此判断已完成。

| 模型 | 可核验基线/准备证据 | 已准备但尚未通过的优化 | 必须补齐的配对与共驻 |
| --- | --- | --- | --- |
| Nano33 | 部分screen JSONL与[summary](results/summary.json)；中文tokenizer已证实丢失信息 | A76/A55、1/2线程、PyTorch动态INT8、batch1/8、ORT FP32/INT8；[ONNX导出误差](results/nano-onnx-export.json)只属WSL | 本板每配置延迟/质量/资源；静态分词缓存实测；各优化优胜配置三组大脑共驻。中文不推荐不豁免速度实验 |
| Laya322 | [WSL INT8兼容性](results/desktop-int8-compatibility.json)只属桌面；[板端短基线](results/board/laya-screen-t2-20261020T164634-10370.jsonl)已完成，见下方独立核验 | encoder-only INT8、核线程；同版本[上游导出脚本](sources/laya_export_onnx_v0412.py)已保存，FP32 ORT及适用encoder INT8待执行 | 同输入原生/ORT/INT8精度和延迟；缓存/分组/批次；完整资源与优化配置共驻 |
| JevStyle0.8 | [编译来源](sources/runtime-source-provenance.json)与本地patch；[pairs日志](results/run-board-pairs.log)含libggml.so.0/libggml-base.so.0 not found历史证据 | 已在adapter加子进程库路径，修复后须真实加载评分；核线程、slots17→2、ubatch1024→256待运行 | 不能把ldd或编译成功当基线；补加载、输出、无截断、缓存/批次、精度、资源与三组共驻 |
| OneJev多模态 | 下载清单存在，尚无本板完整图像前向证据 | 原生ARM、视觉/决策分段、核线程、适用量化/ORT、同帧缓存与候选batch仍TODO | 图像质量、实际新帧延迟与缓存命中分开，视觉峰值内存、三个实际大脑后端共驻均待测 |

上述WSL动态INT8只完成小样例兼容性，不是板端精度或提速结论。Nano ONNX文件SHA和误差已在导出JSON记录；0.106是该smoke的最大logit绝对误差，不能当“10.6%准确率损失”。Laya导出应复用同版本脚本，先FP32；其上游全图MatMul量化有已知精度损失，不能盲目量化决策头。ORT线程须通过SessionOptions设置，torch线程参数不限制ORT。

本次只读复核发现当前`board_adapters.py`和`board_optimize.py`已不同于`sources/run-manifest-20261021.json`记录的SHA。这不说明历史证据损坏，但每轮实际部署版本须新建run manifest/快照并关联结果，不能覆盖旧清单来追认旧run。当时优化采样缺PSS、频率与swap流量；最新Windows脚本已实现这些字段及源文件SHA/package版本记录，**已实现待板端实测**，不能补写到旧run。首次修订后的8例质量仍不足以证明无回退，且现有共驻队列主要是基线，优化优胜配置还需重新共驻。

### 2026-10-21 资源与优胜共驻支持复核

最新Windows `board_optimize.py`已加入进程树PSS、各CPU频率、swap累计换入换出、线程数、实际源文件/fixture SHA及包版本，源码只读核验通过；状态为**已实现待实测**。必须在随后回传的run中看到对应provenance和telemetry，不能将新字段套用于旧screen。旧run-manifest继续保留其历史身份。

最新Windows `board_co_resident.py`已支持`PAIR_CONFIGS`环境变量/线程配置，以及`PAIR_MODELS`、`PAIR_LLMS`和`CO_LABEL`，用于优胜配置补共驻；当前板端正在执行的基线runner尚未加载这版，故优胜组合仍待部署实测。随后Windows源码已加入PAIR_CONFIGS.cores、主线程sched_setaffinity及pair_loaded.affinity，状态为已实现待实测。注意sched_setaffinity(0)仅改调用线程，既存PyTorch worker池未必跟随；跨A55/A76配置应使用独立子进程或核验/设置各TID亲和性，不能只记主线程。

前一次只读回收检查尚无Laya/pairs完整JSONL；随后Laya短基线已回传并完成下节独立核验。共驻仍须等待对应JSONL，不由控制器文字进展推断成功，也不直接上板读取。

已核对共驻脚本的后端/模型接口：

| 标签 | 脚本路径与参数 | 已核验边界 |
| --- | --- | --- |
| smol500 | `full-smol500-c1280-dfpgateup`；32层、capacity1280、vocab49280、prefill32、eos49279 | 构建宏DIM960/KV5/HD64/FUSED2/TRANSPOSED1/HEAD_SHARDS1与历史图一致；归档embedding189235200字节等于49280×960×4；根图名为layer0,2…30、head layer32及prefill-layer0,2…30 |
| intern1b | `wsl-fresh-peak155-20261018`；24层、capacity1280、vocab151674、prefill16、eos151645 | 复用resident-agent-20261019/resident；既有说明DIM896/KV2/HD64/FUSED2/HEAD_SHARDS4。当前实际二进制身份、图加载及done仍以本轮回传为准 |
| qwen15-cpu | `127.0.0.1:8081`，读取/v1/models、apply-template/tokenize/completion | 脚本明确CPU路径，没有标成NPU；应核对模型identity与服务进程参数，仍为CPU文本LLM，不是1.5B NPU VLM |

上述为源码、构建命令和历史库存的匹配，**不是本轮板端推理成功证据**。实际验收要有npu_ready、有效done、模型/图身份、driver释放，以及无shape/contract错误。`complete`事件当前在错误捕获后也可出现，须同时核查llm_error/pair_error、每组有效样本数及预期3×3配对覆盖；不能看到complete就判全部通过。Smol/Intern仍只验证decoder时不能写完整视觉链路。

Laya ONNX在Windows已通过上游batch1/2/3及实际10/100候选、目标64/256状态四组检查；encoder INT8四组sanity也已回读通过，详见下节。桌面通过与ARM速度/资源仍是不同验收项。最终优化配置的共驻、精度与长时测试继续保持未完成。

### 2026-10-21 Laya短基线完整回传与独立复算

证据：[laya-screen-t2-20261020T164634-10370.jsonl](results/board/laya-screen-t2-20261020T164634-10370.jsonl)。逐行解析153条记录，无坏行或error；有telemetry和complete。这里只能称**这一轮缩短后的基线完整结束**，不是140例全量验证、优化完成或产品验收。独立复算与summary分块统计一致：

| 条件 | 有效n | P50 / P95 | 原始模型正确数 | 样例边界 |
| --- | --- | --- | --- | --- |
| 10候选，状态下限64 | 30次 | 1.259440 / 1.271345秒 | 21/30 | 前10个literal重复3遍，仅10个独立case；实际状态64或66token |
| direct100，状态下限64 | 20例 | 9.606919 / 9.673428秒 | 6/20 | 12literal+4unknown+4safety，没有paraphrase |
| two-stage10×10，同20例 | 20例 | 2.481585 / 2.518602秒 | 5/20 | 同上；包含两次评分总时间，组路由正确9/20 |
| two-stage，状态下限128 | 10次 | 3.330855 / 3.408433秒 | 1/10 | 实际128–130token，额外背景填充，不能称等同真实长对话 |
| two-stage，状态下限256 | 10次 | 5.374929 / 5.391003秒 | 1/10 | 实际256–258token，仍只是小样本 |

direct100中literal6/12、unknown0/4、safety0/4；分层literal5/12、unknown0/4、safety0/4。分层降低本组延迟，但本轮正确率也较低，**不能据此推荐为可靠中文动作决策器**。此处safety是原始模型选择，没有验证确定性执行器覆盖；不能写“安全系统0%”或用未来安全门掩盖模型错误。当前subset未含paraphrase，必须单列未测，不能把20例总正确率称自然语言改写精度。

候选扫描4/8/16/32/64/100各10次，P50分别0.932821/1.148642/1.620948/2.867475/5.809191/9.556759秒。100候选扫描7/10与direct100的6/20来自不同样例集合，不可合并或择高值报道。扫描结果用于候选成本初筛，不足以证明随候选变化的准确率趋势。

载入约9.907622秒，CPU亲和性[6,7]；各块平均CPU约1.98核（约198%单核口径，8核归一约24.8%）。计入1162条连续telemetry和请求快照后，整run观测RSS最高**1922039808字节（约1.922GB/1.790GiB）**、最高传感器温度**71.744℃**、MemAvailable最低**2861195264字节（约2.665GiB）**、swap_used最高524288字节。监测错误列表为空。这些是观测峰值，不是PSS，也未证明没有降频。旧summary的分块资源只取请求结束快照，若只取其最大值会漏掉连续采样峰值（约1.836GB/70.446℃）；最终资源报告须纳入telemetry。

Laya桌面ORT证据：[FP32导出与形状检查](results/laya-onnx-export.json)、[encoder INT8导出](results/laya-onnx-int8-export.json)、[INT8桌面配对](results/laya-onnx-int8-parity.json)。FP32文件1288263205字节；INT8文件957698187字节，encoder常量权重MatMul实际量化88个，head FP32、per_channel=false。四组形状的FP32/INT8答案均与参考一致，但都基于同一条“请求帮助”指令、两种候选数及两种状态长度，**不是四条独立质量样例**，不能据此判精度无回退，更不能引用其中桌面秒数为板端性能。ARM延迟、8例/后续留出精度、资源和共驻仍待测。

## 2. 共用必测项

- [ ] **B01 身份与离线**：保存模型repo/revision、权重/tokenizer SHA256、实际运行时/编译参数/脚本哈希和backend；核验完整文件清单，关闭联网回退，实际完成评分。本地恢复源码不能写成未使用的上游提交。
- [ ] **B02 冷热分离与预热**：分别测进程启动、载入、首次推理和常驻热态。正式条件预热20次、至少200有效样本；短矩阵标初筛，优胜配置补足。尾延迟评估至少1000次，不把20次初筛样本的P99当稳定结论。
- [ ] **B03 核/线程**：核验拓扑、全部工作线程/子进程亲和性；A55/A76分别测1核1线程、2核1线程、2核2线程，记录PyTorch/BLAS/OMP/ORT/scorer实际线程，防止过度线程化。
- [ ] **B04 输入与候选**：状态目标64/128/256、候选4/8/16/32/64/100，保存各tokenizer实际state/head/total tokens。当前补背景至目标的方法只是长度下限，不能标严格等长；完整保留动作和用户输入，不静默截断。
- [ ] **B05 策略**：完整直接评分与10组→10动作分层分别测；分层计两次调用总成本和路由错误。若检索候选，单列recall@k/检索耗时，不得只选含答案的候选集。候选顺序随机复测，配对固定种子。
- [ ] **B06 计时**：统一板端monotonic；分列排队、构建、分词、forward、后处理、安全门、mock ack及端到端。不能混入SSH和资源采样成本；无法拆分时标完整模型调用。mock ack不等于实体动作完成。
- [ ] **B07 精度无回退核查**：同一冻结集比较literal、paraphrase、unknown、安全子集、路由准确率，保存逐例变化。关键急停新增错误阻止推广；140条合成例仅初步诊断，不能证明自然场景无回退。最终配置需独立留出集，不在评测集调阈值后再称独立验证。
- [ ] **B08 安全分离**：原始模型安全选择与确定性门覆盖率分别统计；门纠正输出不能掩盖模型退化。补齐能力白名单、状态版本、取消/过期与mock回执。
- [ ] **B09 资源**：连续采RSS/PSS、子进程CPU、MemAvailable、swap流量、温度/频率和错误；给单核100%及全板归一化口径。前后快照不能当峰值；采样峰值注明间隔与遗漏短峰可能。
- [ ] **B10 配对与热状态**：baseline/optimized区组交替，同输入/顺序/计时范围；记录起始温度频率、n、P50/P95/最大值、失败率和绝对差。避免页缓存、运行顺序、热降频造成假加速。
- [ ] **B11 失败与版本**：唯一run_id，不覆盖失败证据；每格记录success/unsupported/error/timeout。异常停监测、清理残留进程。v1脚本已冻结，修正后的v2不能追认v1指标；绑定实际部署脚本哈希。

## 3. Nano33：完整cross-encoder

- [ ] **N01 基线**：核验原生英文相关性对照、中文动作和标签对应；它对各候选做cross-encoder评分，不能当固定100类头或纯embedding检索器。
- [ ] **N02 常驻**：模型/tokenizer常驻，eval与inference mode；排除重复载入和不必要复制。
- [ ] **N03 核线程**：完成B03，检查内外层线程，防止候选batch与BLAS线程叠加。
- [ ] **N04 候选batch/分组**：batch1/4/8/16按内存测试，比较完整请求延迟和激活内存；全部候选均评分。直接100与分层比较包含路由成本与错误。
- [ ] **N05 原生/ONNX**：已新增ORT FP32/INT8候选，板端结果待验收；PyTorch为参考，核验ARM ONNX Runtime算子、动态shape、输出一致性后配对测速；只导出ONNX不算部署。
- [ ] **N06 动态INT8**：实际尝试适用Linear或ORT动态量化，记录转换节点数量、ARM kernel、排序与准确率；不能只凭设置int8=True就认定有效。记录不支持或变慢的结果。
- [ ] **N07 精度保护**：中文基线失效需明确迁移能力不足；仍保留调度/运行时优化尝试，不推荐更快的错误答案。NaN/Inf必须报错，不能悄悄选择首候选。
- [ ] **N09 候选缓存**：冻结question、action_set_version与tokenizer SHA后缓存静态分词/模板，状态每轮更新；保存cache hit/miss、分词和完整调用耗时，核验缓存前后逐token与输出相同。不能缓存包含旧state的cross-encoder表征，也不能只报预先分词forward为完整决策延迟。
- [ ] **N08 共驻与交付**：三组大脑逐一空闲共驻/同时推理，交付优化前后延迟、资源、精度变化及推荐或不推荐原因。

## 4. Laya multilingual 322M：动态候选编码器

- [ ] **L01 基线**：冻结原生版本、配置与输出语义；override接受4096不代表该checkpoint长上下文质量有效。
- [ ] **L02 常驻**：模型/tokenizer常驻；可缓存不变的分词/候选元数据，不可缓存依赖本轮状态的双向attention输出。
- [ ] **L03 核线程**：完成B03，测A76单/双线程，并核对工作线程、频率。
- [ ] **L04 候选分组**：直接10/20/100及10×10按支持情况测；检查单选项48-token cap、head budget、完整instruction及marker数。不能隐性裁剪动作或指令。
- [ ] **L05 原生路径**：评估支持的attention实现与合法batch；改变选项layout等输入语义时作为独立策略对照，不认作无条件免费提速。
- [ ] **L06 动态INT8/ONNX**：当前尝试encoder-only Linear动态INT8、head保持FP32并记录quantized_modules；验证量化及自定义attention/head能实际在ARM前向，再和原生浮点对照；统计量化节点。外部INT4/ONNX包需核验同源结构/精度，不能只按文件变小宣称加速。
- [ ] **L07 精度保护**：单列分层路由、最终选择、unknown及安全项；关键子集回退则不推荐。候选完整性与顺序复测不可省略。
- [ ] **L09 缓存与批次实测**：确认Laya已有question-token复用机制的实际命中/收益，不把库默认行为当本轮新增优化；比较关闭/开启或冷/热模板并保留相同state。候选组并批与多请求batch分开计吞吐、等待和单请求总延迟，不能用batch吞吐冒充交互速度。
- [ ] **L08 共驻与交付**：三组大脑配对，说明大核争用、内存变化；保留原生回退配置。

## 5. Jev-Style 0.8B：Q4自定义scorer

- [ ] **J01 基线**：实际使用候选评分协议及yes/no读出，普通chat生成不能代替；核验离线单样本、完整输入、候选对应和有限分数。
- [ ] **J02 常驻**：scorer子进程常驻，独立记录载入、首轮、热态；日志、退出状态和板端超时齐全，避免阻塞readline永久占锁。
- [ ] **J03 核线程**：完成B03；记录threads、batch/ubatch线程与子进程亲和性，CPU资源包含scorer。
- [ ] **J04 候选分组**：直接100、合法分块全量与10×10配对；分块覆盖全部候选，列合并规则和总耗时。
- [ ] **J05 原生scorer调参**：比较n_ctx、n_seq_max、n_ubatch、batch与支持的kernel。减少slots/ctx必须容纳内部序列与完整输入，先正确性再测速。当前排队项为slots17→2和ubatch1024→256，尚无结果不能写提速成功。
- [ ] **J06 量化适用性**：Q4_K_M本身已量化，优先GGUF/ARM原生优化；PyTorch dynamic INT8不直接适用于此scorer，应记不适用及理由。若比较同源Q8/F16/Q4，先核验资源与精度，不能由文件尺寸推断速度。
- [ ] **J07 状态一致性**：改变ubatch、跨请求共享、分块或many_mode均需核对recurrent state及动作一致性。当前独立decide的keep_prefix=false为基线；未经证明不能复用旧决策状态。分数差异和动作差异分开记录。
- [ ] **J08 上游隔离**：上游`models/jev08/jev_style_decision_gguf.py`及manifest不改；本地扩展在`jev_style_decision_board.py`，单独保存patch/文件哈希。2026-10-21只读核对见新增n_seq_max参数两处差异；不能把上游SHA当作本地运行代码SHA。
- [ ] **J10 候选缓存与批次**：合法缓存rendered候选/模板，状态前缀必须随新请求更新；分别测试tokenize缓存、完整decide、支持的many_mode/分块批次，核对评分一致性。静态分词缓存不等于recurrent/KV复用，不能共享失效的旧状态。
- [ ] **J09 共驻与交付**：记录内部序列内存、子进程和三组大脑争用，交付配对收益或无收益结论。

## 6. OneJev1.6-0.8B：真实多模态链路

**没有实际图像、视觉前向和候选决策，不能勾选本模型部署或优化完成。预生成标签、仅文本和下载均不能替代。**

- [ ] **O01 完整部署**：核验权重、tokenizer、processor、视觉encoder及自定义代码，记录视觉/语言/决策头实际总参数、权重与RAM；0.8B标称不替代完整资源核算。
- [ ] **O02 图像基线**：冻结图片SHA、尺寸与候选；分列解码、预处理、视觉encoder、融合/decision、后处理和完整调用，在ARM上真实离线推理。
- [ ] **O03 常驻预热**：模型、processor、图常驻；区分首次形状编译/分配与热态。同帧特征缓存需校验frame_id/内容哈希并单列命中场景，新图片不能冒称缓存速度。
- [ ] **O04 核线程**：完成B03，分别定位预处理、视觉和决策瓶颈。
- [ ] **O05 分组及图片成本**：直接/分层、分辨率/patch预算按支持测试；降低分辨率/ROI会改变信息，须原图质量对照，不能称无损加速。
- [ ] **O06 原生/量化**：先验证原生ARM，再评估ONNX/native runtime、动态INT8或模型支持的量化；分别核验视觉/融合/决策算子。不支持保留具体错误与替代尝试，不能用TOPS估算。
- [ ] **O07 精度核查**：固定图片/候选比较动作、视觉依据、unknown、安全项；不能仅比较文本输出或权重差异。
- [ ] **O08 共驻**：与Smol500M、Intern1B、Qwen1.5B实际后端分别空闲共驻/同时推理，测视觉内存峰值。大脑仅文本decoder时必须标“多模态Jev+文本大脑”，完整VLM+Jev仍TODO。
- [ ] **O10 候选与图片批次**：候选模板缓存按action_set版本失效；同帧视觉特征缓存与不同新帧batch分别测，记录等待、完整图像至决策延迟和内存，不能用同帧命中代替动态视觉性能。
- [ ] **O09 未实施不得关闭**：本轮若无视觉前向，报告/handoff明确保留多模态部署、基线、优化、共驻四项TODO，不能因其他三款完成而关闭总任务。

## 7. 共驻与KV边界

| 对照大脑 | 当前可用路径/边界 | 必须验证 |
| --- | --- | --- |
| SmolVLM500M系列 | NPU decoder图与新编译resident；视觉encoder是否加入需另证 | 独占、空闲共驻、同时推理；视觉缺失不能称完整VLM |
| InternVL3 1B系列 | NPU decoder resident；视觉encoder是否加入需另证 | 固定prefix行为、TTFT/decode、Jev延迟和资源配对 |
| Qwen2.5 1.5B | 当前8081是CPU文本LLM，**不是NPU VLM** | 纳入服务PID资源及大核争用，不冒称1.5B NPU VLM共驻 |

- [ ] **C01 真重叠**：prefix准备在正式并发前完成，屏障同步LLM/Jev；记录同一monotonic起止及重叠时长。Jev仅覆盖prefix准备不能算正式生成并发。
- [ ] **C02 KV对齐**：resident按batch tokens向下对齐复用。当前Smol完整prefix92、batch32→预期复用64；Intern prefix37、batch16→预期复用32。分别记录完整长度、expected_aligned_reused_tokens、实际复用及重算量；batch变化后重新计算。不能assert复用等于完整prefix。
- [ ] **C03 资源范围**：当前树、scorer及外部llama-server均采样；记录其他常驻模型。树内RSS可重复共享页，优先各进程PSS与系统MemAvailable，不把RSS相加当实际物理增量。
- [ ] **C04 配对**：Jev独占、大脑独占、两者驻留空闲、同时推理均测；记录增量内存、TTFT、decode、Jev延迟、温度/频率/错误。仅能加载不证明争用可接受。
- [ ] **C05 稳定性**：5次/条件只是初筛；优胜组合补正式样本、30分钟及后续2小时模拟，核查内存增长、过期/取消、异常。未跑长时明确未测，不由短时成功推断稳定。
- [ ] **C06 后端证据**：CPU/NPU由实际执行确认；1.5B历史NPU质量失败实验不能作为可用部署。未来Jev NPU路径须独立验证，不由现有LLM能力推断支持。

## 8. 优化前后证据表

每个实际尝试填一行，保留失败/变慢配置；加速倍数=基线延迟/优化延迟，须同计时范围，同时报告绝对差。

| 模型/优化 | 基线run_id/代码SHA | 优化run_id/代码SHA | 输入/候选同一性 | n | P50/P95前→后 | CPU/PSS/温度/频率 | 各子集精度前→后 | 并发TTFT/decode变化 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nano33 | TODO | TODO | TODO | TODO | 未测 | 未测 | 未测 | 未测 | 未完成 |
| Laya322 | laya-screen-t2-20261020T164634-10370；原始短基线已回收 | TODO：ARM优化run未验收 | 优化配对待核验 | 基线10候选30次；direct/分层各20例 | 基线1.259/1.271秒（10候选）；优化未测 | 基线整run RSS1.922GB、71.744℃；PSS/频率未测 | direct6/20、分层5/20；各unknown/safety0/4，无paraphrase | 未测 | 基线结束，优化未完成 |
| JevStyle0.8 | TODO | TODO | TODO | TODO | 未测 | 未测 | 未测 | 未测 | 未完成 |
| OneJev多模态 | TODO | TODO | TODO | TODO | 未测 | 未测 | 未测 | 未测 | 未完成 |

每个run保存：模型/运行时/脚本身份、backend/量化、实际输入长度/候选/路由、图片身份、亲和性/线程、预热与有效n、逐例答案/评分、阶段计时、连续资源、异常及完成标记。原始JSONL、摘要、报告应互相可追溯。初筛资源缺PSS/频率等时写缺失，不补估计数。

## 9. 下一步优先级

1. **P0 先保证基线和数据有效。** 收回Nano/Laya screen、JevStyle ARM评分结果，检查错误与输出；用修正计时/资源/重叠边界的新版本。保留旧证据与限制。
2. **P1 文本三模型逐一实测优化。** 执行已排队亲和性、线程、batch、动态INT8及scorer参数初筛；检查实际算子/量化数量而非配置标签，补A55单线程、候选分层和状态长度条件。中文错误明显时先解释能力问题，不能只追低毫秒。
3. **P1 OneJev真实多模态基线。** 完整校验后运行ARM图像前向与分段计时；兼容性失败保留具体证据和替代路径尝试。文本模型优化完成不关闭此项。
4. **P2 优胜配置正式配对。** 扩展同输入精度与延迟样本；适用时比较原生/ONNX/动态INT8，JevStyle优先GGUF原生，OneJev按实际算子能力决定。一次改变一类变量。
5. **P2 共驻与长时。** 三组大脑按实际后端逐一验证优化前后争用，重点A76、内存和温度，保留固定prefix对齐规则。共驻只测基线不能证明优化配置同样有效。
6. **P3 推荐与handoff。** 每个模型给出可用配置或不推荐理由、真实收益/精度变化、未完成事项和下一条命令；Windows证据落盘并提供Mac可读报告。

## 10. 最终防遗漏门槛

- [ ] Nano33有真实优化前后配对、精度及共驻证据。
- [ ] Laya322有真实优化前后配对、精度及共驻证据。
- [ ] JevStyle0.8有真实优化前后配对、精度及共驻证据。
- [ ] OneJev真实多模态部署/优化通过；否则继续保留TODO，不能标完成。
- [ ] 冷热、实际候选/输入、CPU/NPU、视觉是否执行均明确。
- [ ] 无加速、精度回退、unsupported、超时、样本不足如实保留；没有用部署基线替代优化完成。
- [ ] 权威源码/证据保存Windows，Mac可打开报告，handoff更新。

审计创建：2026-10-21。所有未勾选项必须以实际结果驱动更新；本文未填未经核验的速度或准确率。
