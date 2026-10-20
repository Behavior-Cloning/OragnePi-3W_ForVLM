# Jev 逐模型提速审计与验收 TODO

新增要求日期：**2026-10-21**。用户硬要求：**每个 Jev 都要实际尝试优化延迟和决策速度，不能只下载、部署或测基线。** 本文是该要求的验收清单，结果以 Windows F 盘原始证据为准。用户新授权替代旧计划“暂不下载或测速”的历史限制；不包含后训练或实体动作。

权威目录：`F:/Files/OPi-Lab/repos/OragnePi-3W_ForVLM/research/jev-validation`。关联：[验证计划](JEV_VALIDATION_PLAN.md)、[动作词典](ACTION_DICTIONARY.md)、[运行时源码来源](sources/runtime-source-provenance.json)。本审计智能体只维护本文，没有访问板卡、占锁或触发负载。

## 1. 状态与完成定义

| 模型 | 创建本文时的准备状态 | 优化验收状态 |
| --- | --- | --- |
| Nano33，sdmlai/nano-jev，约33M | Windows有下载清单；已取得部分screen证据，中文direct100为0/109；见下方修订 | **TODO：未完成优化** |
| Laya multilingual，约322M | 本轮短基线JSONL已完整回收：direct100为6/20，分层5/20；ORT桌面形状检查完成，ARM优化仍待测 | **TODO：未完成优化** |
| Jev-Style 0.8B Q4_K_M | 基线、slots17→2及ubatch1024→256已回收；最低PSS峰降低399.1MB，速度随候选数方向不同，direct100质量仍4/8 | **已有优化尝试；质量/共驻/完整验收TODO** |
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

前一次只读回收检查尚无Laya/pairs完整JSONL；随后Laya短基线已回传并完成下节独立核验。其后基线共驻JSONL亦已回传，独立核验见下节；优化配置共驻仍待实测。本审计始终只读Windows回传文件。

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

### 2026-10-21 基线共驻回传审计：成功覆盖与失败边界

证据：[共驻JSONL](results/board/co-resident-20261020T165652.jsonl)、[控制器日志](results/run-board-pairs.log)、[JevStyle动态库错误](results/board/jev08-engine.log)。JSONL共116条、无坏行；3个大脑×Nano/Laya×3个模式×5次，共**18条件、90条有效测量**，另有9次大脑初始基线。旧runner未写complete事件是当时版本行为；控制器日志明确`co-resident exit 0`，18条件覆盖完整。**不等于9个Jev×大脑组合全部通过**：JevStyle在3个大脑下均加载失败，错误由stderr确认缺libggml.so.0，不是模型精度结果；修复后的独立评分和三组共驻仍TODO。

NPU生命周期与KV已逐条核验：Smol32层/capacity1280，ready driver701586072字节；Intern24层/capacity1280，driver753274680字节。两者closed事件均driver_bytes=0，无close_error。包括初始和配对在内，Smol23次输出全部reused_tokens=64、Intern23次全部=32，符合完整prefix92/37向batch32/16对齐；每个Nano/Laya组合的5对NPU输出token序列与非并发参考全部一致。Smol均生成32token；Intern17–32token由EOS提前结束，比较按同case配对，不能假设所有输出等长。

CPU identity为`qwen25-15b-q4`，接口声明text→text、n_ctx4096、n_embd1536、Q4_K Medium；这是现有CPU文本LLM路径，不能写1.5B NPU VLM。23次返回timings.cache_n均37；输出9–28token。CPU TTFT是HTTP首个非空文本片段时间，NPU TTFT是runner首token事件，二者定义要注明。没有图片编码，不是完整VLM视觉并发。

每条件仅5次，下表是本轮中位数初筛；“单跑”指另一模型仍驻留但不调用，不能称完全独占系统。所有NPU阶段还存在后台CPU Qwen服务。

| 大脑+Jev | Jev单跑→并发 P50秒 | 大脑单跑→并发 TTFT P50秒 | decode单跑→并发 token/s |
| --- | --- | --- | --- |
| Smol NPU + Nano | 1.488→1.587 | 2.649→2.693 | 17.486→17.428 |
| Smol NPU + Laya | 1.322→1.411 | 2.631→2.696 | 17.480→17.386 |
| Intern NPU + Nano | 1.459→1.551 | 0.927→0.951 | 13.744→13.504 |
| Intern NPU + Laya | 1.316→1.406 | 0.924→0.965 | 13.759→13.502 |
| Qwen CPU + Nano | 1.456→2.865 | 1.343→2.832 | 9.763→9.600 |
| Qwen CPU + Laya | 1.325→2.501 | 1.360→2.666 | 9.666→9.656 |

30条concurrent都有正时间重叠，区间重算与overlap_seconds一致；但每次只发一次Jev请求，不能代表持续决策负载。**Smol两组10次Jev均在首token前结束，Qwen+Laya5次也均如此**；Intern实际decode重叠仅约0.37–0.80秒，Qwen+Nano仅4/5次有0–0.38秒decode重叠。因此decode下降小不能解读为持续decode并发无争用。优化优胜共驻应补连续Jev请求覆盖整段生成，分别报告prefill和decode竞争。CPU组合的首字和Jev延迟明显增加，需继续核绑定/调度优化。

资源整合414条连续采样与快照共612条，未见监测错误；每条PID列表均唯一，既存CPU服务PID967在**612/612**条中出现，没有漏计或单次重复累计服务。旧记录没有保存进程cmdline，服务身份依据接口及已有运行配置，后续provenance可加强。整run观测sumRSS最高3831353344字节、sumPSS最高3821982720字节（约3.822GB/3.560GiB），MemAvailable最低2092118016字节（约1.948GiB），最高温度65.224℃。swap_used最高51118080字节；累计换入增加393216字节、换出增加50237440字节，**不是零swap活动**。NPU driver分配另列，不能未经会计口径核对就与PSS简单相加。

同一Python主进程PID11305连续加载各模型；第一组Laya释放后，在下一组尚未加载Nano时PSS仍约1.05GB。后续Nano组占用含此前运行遗留的allocator/运行时分配，所以这些数是当时真实进程集负载，**不是每模型干净独占footprint或增量内存**。要比较模型纯增量，应每配置用独立进程或同起始状态配对。本轮证明六个基线组合可完成短测，不证明长时稳定、不证明三个大脑一起驻留，也不证明JevStyle/OneJev或优化配置通过。

### 2026-10-21 持续并发与新优化队列审阅

主任务将原优化控制器5770在大ONNX文件SCP传输阶段主动停止，改为新控制器5972：执行顺序JevStyle→Laya→Nano，Laya大图在对应配置运行前按需`rsync --partial --append-verify`，保留部分文件用于续传。此为传输/调度调整，不是模型性能失败，也不等于任一优化格通过。只读Windows日志已见ORT依赖满足及provenance/loaded，尚无回传opt结果JSONL可验收；PID仅为当时标识，后续恢复须重新确认。源码还保留Nano图的前置SCP；“按需rsync”准确适用于大Laya图，不应写所有文件都改成该路径。

新共驻源码加入`concurrent_sustained`：屏障同时启动LLM与连续Jev，LLM在finally设置finished；Jev轮询直到LLM完成且至少调用一次，通常最多在LLM结束后多跑一个已开始的decision。由此正常成功路径可覆盖prefill及decode，改善旧版单次Jev仅覆盖prefill的缺陷。状态为**已实现、待部署实测**，不能沿用旧90条测量声称持续并发验证。当前Windows源码审阅SHA为`51777d2a6d27fdfaf20eb43f66ec50f24d20d19038766148ffc38ba7e1a8ba9e`，优化控制器SHA为`5e28ee8cabebf0f34469ee71be18ab9eb23ec1b1cef949464e8ff081806e163d`；后续修改应由实际run provenance绑定版本。

亲和性修复已改为枚举主Python进程各线程及Jev scorer子进程各线程，并记录worker_affinities；没有把NPU resident worker或外部CPU Qwen服务改到Jev核集合。正常路径方向正确；仍须warmup后核查懒创建线程，确保亲和性与报告一致。当前结束时仅恢复主线程，既存工作线程会保留最后一次集合，下一pair虽重新设置，仍建议完整恢复受改TID。主线程/子进程适用范围不能扩成全进程树，否则会破坏NPU host固定CPU4,5。

持续模式待修/待验证的异常与计量边界：

- [ ] **S01 真正有界终止**：120秒检查仅在每次decision开始前，不能打断一次阻塞的scorer.readline；LLM自身可能等HTTP180秒或resident300秒。先等待a.result及ThreadPool退出也会等待另一任务。需有可核验的板端进程级timeout/kill兜底或模型请求deadline；不能称当前循环保证120秒结束。
- [ ] **S02 热保护与错误清理**：Jev先达到78℃或抛错时，当前仍可能等待LLM完成；应明确取消/中止策略并保留错误样本。确保monitor stop/join、模型关闭、env及所有受改亲和性在close异常时也能执行。没有结果不能宣称保护已验证。
- [ ] **S03 持续调用证据**：每次decision保存case_id、monotonic起止与状态token量，并计算实际覆盖首token后的时长。当前只记循环总区间与calls时长，不能精确区分每次模型调用和间隙。末次尾巴需单列，不能把LLM结束后的Jev成本算成同时推理窗口。
- [ ] **S04 公平配对**：持续循环可能访问0–9全部case，而单跑模式仅i0–4。总median比值不等于严格同输入加速/退化；按case比较，或给两模式相同冻结序列。统计每轮call_count与总耗时，per-decision median不能混作循环总延迟。
- [ ] **S05 CPU计量范围**：现snap包括主进程、NPU/scorer子进程、外部llama-server的累计CPU，整请求资源范围完整；但主进程含Jev、客户端/监测开销，sustained前后Δ含最后尾巴。整请求CPU和重叠窗口CPU分别报告，不能将全部归给纯模型forward。
- [ ] **S06 失败证据不丢失**：当前pair_error不带失败时已产生的calls/intervals/连续samples；需保留部分证据并标失败。加载/首轮失败不能造成后续配置被默认视为完成，仍核查每格数量与退出状态。

以上均为只读源码审阅，未运行持续负载。待最终优胜配置回传后，再判断持续decode竞争、热保护及资源结果。所有模型优化验收继续未完成。

后续修订记录：主任务已为future `run_final_pairs.py`改为每个LLM/Jev配置独立进程，使用GNU `timeout 600s`和15秒后强制终止兜底；持续模式改为重复同一case_id并记录每call起止，pair_error保存部分intervals/samples。此为**已实现待实测**，可以改善状态残留、公平配对与整进程失控，但**120秒仍只是轮边界检查，不代表单调用可中断**，细粒度取消/热保护立即停止尚未验收。独立配置仍需核对驱动释放、外部CPU服务状态、实际工作线程亲和性及失败回收。主任务报告JevStyle优化已成功加载并真实评分，第一个配置完整JSONL尚待回传复算，不能提前写优化通过。


### 2026-10-21 JevStyle 首次有效基线独立审计

证据：[opt-jev08-a76-t2-20261020T171123.jsonl](results/board/opt-jev08-a76-t2-20261020T171123.jsonl)，SHA256 `b8e3cd426438797992b8c0289118554f8bf45d4bbc028386c69883d535519664`。32条记录：provenance、loaded、20条opt_measure、8条opt_quality、complete、telemetry各按预期出现；无坏行、error或telemetry采样错误。源码含3次预热，未单独记录首轮耗时；模型构造/加载18.313355秒，与热态decide计时分开。

实际为CPU llama.cpp自定义评分器，Jev-Style-0.8B-Decision-v3 Q4_K_M，A76亲和集合[6,7]、2线程；n_ctx4096、slots17、ubatch1024由已绑定源码默认值确认。每次独立decide为share_prefix=true、keep_prefix=false、mode=fused，非普通聊天生成。运行记录没有各TID亲和列表，不能仅据主线程affinity证明所有懒创建worker的分配。

| 条件 | 有效n | P50/P95秒 | 结果 | 输入与候选 |
| --- | --- | --- | --- | --- |
| 10候选热态decide | 20次，10个独立literal各2次 | 7.097038 / 7.117650 | 20/20 | 状态64token；完整渲染310token、head242；全10候选 |
| direct100热态decide | 8个独立样本 | 57.562466 / 57.658276 | 4/8 | 状态64token；分2块；总input2362–2368、head2226；全100候选 |

两组逐条检查scores和probabilities键集合分别恰为A000–A009、A000–A099，所有值有限；未发现遗漏或静默截断。direct100的input/head计数是各块求和，不能把2362视为单条超过max_len2048的序列。延迟为完整decide（包括渲染/分词、评分调用、结果整理），不是纯单次矩阵前向。P95使用线性插值；8例质量时长P95只有初筛意义。

质量子集：literal2/2、paraphrase1/2、unknown1/2、safety0/2。失败包括para-000期望A000却返回A006，unknown-0期望A003/A004却返回A007，stop-0/stop-1期望A000却返回A086/A061。20/20的10候选literal结果不能写成总体100%正确率，也不能掩盖direct100安全样例失败；这里检验的是原始模型选择，尚未验证外层确定性安全策略。

连续1185个样本覆盖642.313秒；合并逐请求快照后，进程树RSS峰2339840000字节、PSS峰2331338752字节（约2.331GB/2.171GiB），系统available最小2930180096字节，温度峰66.670℃，线程峰6。20次计时CPU秒/墙钟秒合计比为1.854核，包含Python宿主、监控线程及scorer子进程，并非纯scorer算子CPU占用。A76采样频率有416000和2002000kHz；全过程含加载空闲，不能把416MHz直接解释为热降频。最大loadavg为2.210/1.925/1.823。系统swap使用峰4194304字节，累计swap-in增量0、swap-out增量843776字节；不能写零swap。资源为该优化进程树及全系统指标，不含所有其他常驻进程RSS，也不是本轮共驻结果。

5项run sources均与首次审计时Windows源文件一致，随后已冻结于`sources/optimization-v1`；后续源码修订不改变旧run身份：board_optimize.py `91268ab577d601435157efed1c833a5d97c21433f1b38f64cefed89811796b70`、board_adapters.py `f09bffe913cc08d7d568ab759b56b50536d43bc621c5b26666015150617d80eb`、jev_style_decision_board.py `95ba2fc161342aa6b3b34028aa0bc65388e3972dc77bc8c41df50d23df976b0b`，以及actions/cases。运行包版本torch2.7.0、transformers4.57.1、laya0.4.12、onnxruntime1.23.1。下载清单绑定revision `edf37c26a1098f83cf4264b8adbe0dca2d2ebb0c`与Q4权重SHA `0a19bc29bacc33e0d871146c8612b24dd14c2ed2e61cedeb7a928b0852628bac`；但此run未自行记录板端权重及二进制SHA，仍须补闭环，不能把Windows清单当作本run端侧重算哈希。运行时来自恢复的本地llama.cpp快照，不是作者声明的上游参考提交。

J01基线记录已完成；J02冷热首轮、J03线程核分配矩阵、J04分层配对、J05调参、J07精度一致性、J09共驻及全部优化最终验收继续保留TODO。本次首次可运行a76-t2本身没有前后加速对照。

### 分开判定：运行优化、质量、可部署

后续每款模型分别填写三项结论，不能用一项替代另一项：

1. **运行优化结果**：模型确实在ARM执行，完整候选与同状态输入下比较优化前后延迟、CPU/PSS；成功运行但更慢写“运行成功、无提速”，内存降低但延迟不变写“减内存、未证实提速”。部分run、超时或不支持须保留原始错误与有效样本数。
2. **质量结果**：逐例比较同8例literal/paraphrase/unknown/safety及相应任务原生控制，报告答案变化、分数变化和完整候选。8例只属初筛；优化保持基线答案也不代表模型能力合格，尤其不能把“未回退”写成“安全通过”。
3. **部署结论**：需要目标任务质量、真实后端、离线完整性、共同常驻/持续竞争、资源与异常恢复证据。速度快或短时共驻成功只能支持对应局部结论。Nano中文信息丢失、JevStyle安全初筛失败、OneJev尚无视觉前向都不能被运行优化成功覆盖。

### provenance补充需求（源码只读审查，实施状态见下方修订）

现有优化run记录5项脚本/fixture SHA及4个Python包版本，未将实际加载的模型、tokenizer、推理二进制/库完整绑定到该次运行。建议由主任务保存一次**板端部署清单**，在不计入推理的准备阶段计算并核对哈希，然后每run引用清单SHA；无需在每个热态请求反复散列大权重。审计智能体未改代码。

- 实际加载文件的规范路径、字节数、SHA256、repo/revision：Nano与Laya的权重/决策头/config/tokenizer；ORT模式记录实际FP32/INT8图及外部数据；JevStyle记录GGUF、readout/release配置、tokenizer；OneJev记录视觉/文本权重、processor及测试图片身份。Windows下载清单和桌面导出清单保留作为上游来源，不能替代板端文件核对。
- 运行程序及库：JevStyle的jev-score、实际解析到的libllama/libggml等库，恢复源码树SHA与编译命令/架构选项；记录实际库路径，避免LD_LIBRARY_PATH改变后只记程序SHA。PyTorch/ORT/Laya记录实际import位置与版本；自定义wrapper及导出代码另存SHA。
- 实际运行参数：完整非敏感命令、解析后的threads、n_ctx、slots、batch/ubatch、head/max_len、quant、provider、图优化级别、INT8模块或节点范围，以及预热后PID/TID亲和性。当前environment里的null只代表未设置该环境变量，实际默认值需结合绑定源码解析。
- 数据与计时：actions/cases顺序和SHA、实际状态/渲染输入token数或哈希、候选数/分块策略、图片预处理信息；loaded/首轮/热态分离，开始结束单调时间、完成/退出码、错误与遥测采样失败保留。模型文件检查耗时不得混入热态延迟。
- 旧run只可追加“事后补验”，不能伪装成当时记录；若无法证明旧run所用文件等同当前文件，明确身份闭环限制。最少要在最终推荐配置重测前完成部署清单。

### 2026-10-21 provenance实现修订：尚待实际最终配对记录

只读核验新增[board_identity.py](board_identity.py)（SHA `be7c234106ae2f352dd1fcfd703bf04ac4f138a5ba28096d2d7ea4c234239fcc`）与[run_final_pairs.py](run_final_pairs.py)（SHA `8c1ba21974b0eeb8d8e38798f5eadb599025aa07244bab209978506c0915601e`）：最终配对前在实验锁内、计时外遍历所选Jev模型目录、bin、根目录Python脚本与fixture，流式计算SHA并核对散列前后size/mtime；保存板端清单回Windows。新co runner每次引用清单SHA并复核size/mtime，发现变化则停止。此机制能补强**未来最终配对**身份，但不追认旧优化run；当前只是源码已实现，尚无实际清单/最终run可验收。

覆盖边界：bin目录哈希不等于运行时实际加载库路径证明；外部既有CPU Qwen服务权重/二进制和既有NPU图不在该Jev目录遍历范围，NPU图按脚本说明仍依赖原始实验manifest。若最终报告声称整组共同常驻模型身份完整，还须链接这些既有服务/图的实际路径与来源证据。包版本已记录，但wheel内容/import路径未记录。每次配对是size/mtime复核而非再次全量SHA，应按此描述。

当前队列已传旧版脚本，后续回收应与`sources/optimization-v1`冻结的5文件比较（已逐项核验与首基线provenance一致），不能以新Windows文件不匹配误判旧run来源不明。新版board_optimize SHA `ea14b8c1c0cdacae0ef558a777a6527212b45a178d3431179e4da1a51aa727d3`加入3条warmup计时和预热后进程树TID亲和事件；新版board_adapters SHA `36a46998135b8a47065e5adb369c42ab4bbfdd4219a634ea17ab04cfeb5c9f1e`明确写出JevStyle解析后slots、ubatch、ctx、渲染上限、CPU/Q4等参数。新字段只有实际使用新版的run才能声称存在。

### 2026-10-21 JevStyle slots17→2优化配对审计

证据：[opt-jev08-a76-seq2-20261020T172214.jsonl](results/board/opt-jev08-a76-seq2-20261020T172214.jsonl)，SHA `9e200129bb731709bd5295a5fd6a6b13a9214b5785d8163cd32f65740b55e5a6`；控制器明确`jev08 a76-seq2 exit 0`。32条预期记录齐全，complete与telemetry存在，无错误；5项source SHA全部匹配`sources/optimization-v1`及a76-t2。环境唯一有效变化为JEV_N_SEQ_MAX=2；线程、A76亲和集合、ubatch及输入fixture保持基线设置。

| 指标 | a76-t2（slots17） | a76-seq2 | 配对解释 |
| --- | --- | --- | --- |
| 10候选P50 / P95 | 7.097038 / 7.117650秒 | 6.968772 / 6.982379秒 | P50减少0.128266秒，观测1.0184× |
| direct100 P50 / P95 | 57.562466 / 57.658276秒 | 57.140560 / 57.196208秒 | P50减少0.421906秒，观测1.0074× |
| PSS峰值 | 2331338752字节 | 1995433984字节 | 减少335904768字节，约14.4% |
| RSS峰值 | 2339840000字节 | 2003603456字节 | 减少336236544字节 |
| 系统available最小 | 2930180096字节 | 3267133440字节 | 本run可用余量增加；仍非共同常驻结果 |
| 温度峰值 | 66.670℃ | 68.853℃ | 运行顺序和热历史不同，不能据此认定新配置更热 |
| 计时平均CPU核数 | 1.854 | 1.868 | 宿主/监控/scorer进程树口径 |

20次10候选与8例direct100逐条case_id、expected、state_tokens、候选集合及渲染token计数一致；所有scores/probabilities对基线最大绝对差均为0，全部答案相同。10候选仍20/20（10独立例重复），direct100仍4/8：literal2/2、paraphrase1/2、unknown1/2、safety0/2。保持坏基线的答案不等于质量通过。

连续1164样本覆盖627.116秒，无采样错误；线程峰6；cpu6/7采样全为2002000kHz。swap存量峰4194304字节、此run的swap-in/out增量均0，不得据此删除基线实际swap-out增量。加载9.946秒低于前run18.313秒，但没有清理文件缓存/交错重复，不能将全部差值归因slots减少。

**三项结论**：运行优化已实际尝试且完成，主要明确收益为PSS减少约336MB；0.7%–1.8%的延迟改善只是顺序初筛观察，缺AB/BA或多轮交错，尚不足宣称稳定大幅提速。质量仅证明本小样本与基线等价，原有安全/未知/改写错误未解决。可部署结论仍未通过，待优化配置共同常驻、持续竞争与目标质量补测；尚未完成其他核/线程、ubatch及OneJev项。

### 自动续跑选择口径审计

`finish_jev_validation.py`只从complete、计时n≥20、quality_n=8且8例答案字典与完整a76-t2基线一致的配置中，按**10候选P50最小**选择后续共同常驻配置，并明确其为性能比较而非安全/部署批准。它不综合direct100速度、PSS与P95；最终推荐须人工结合这些指标。完整候选/有限评分、输入与源码同一性仍由原始JSONL审计，不能只看摘要complete。

首次审阅发现Nano原生英文控制未进入资格过滤，主任务已修订：compare SHA `162e19d7266d3ea7cf0d9e5609e6a818d5f127e3358a45da1841019c35b22215`保存20次timing答案与完整native答案序列；continuation SHA `f2ca36d9f91e7d47b104eb564476e64f144a5562c9ee91e795580c6740671ef7`要求20次timing、8例质量答案与基线一致，Nano额外要求18次native答案序列一致。已只读核验。随后continuation SHA `68eae91f102c8f8adc7021c5c0f3dabaf170f35e95ca11a46dcfb0cc593db8ce`已补严格守门：Nano基线native_n不足18即记录missing_native_controls并跳过，之前回退绕过限制的问题已修复。即使全部原生控制不变，也只是6个英文样例各3次，独立任务质量评估仍必要。部分配置可能因900秒整体上限在质量子集结束前停止；超时保留有效计时与已完成质量例，写明未完成，不能算全格通过，也不能无证据归因为模型崩溃。

### OneJev待办证据边界复核

只读Windows模型文件确认：`models/onejev08/config.json`声明`Qwen3_5ForConditionalGeneration`/`qwen3_5`、记录transformers_version5.17.0，且含12层vision_config；当前板端run记录transformers4.57.1。版本/架构差异是兼容性调查依据，但当前results尚未发现OneJev加载失败、真实图片前向或优化尝试记录，不能写成已经实证“无法部署”。仍为多模态部署/优化TODO。

直接读取safetensors头（未加载模型）核验：473个权重tensor均BF16，总852985920个元素、1705971840字节数据；其中名称含visual/vision的153个tensor、100592896个元素。文件总1706030528字节，包含头。配置中的dtype=float32与存储BF16不同，实际加载/执行dtype及PSS必须实测，不可由文件大小直接推断运行内存，也不能漏掉视觉权重。

下一次实际尝试应使用隔离兼容运行时和冻结图片，先记录架构加载成功/失败、processor和真实视觉forward，再分段计时及核线程/量化适用性测试；若不支持须保存具体错误及尝试范围。当前三款文本模型自动优化队列不包含OneJev，队列结束不能关闭此项。

### 2026-10-21 JevStyle seq2+ubatch256配对审计

证据：[opt-jev08-a76-seq2-ub256-20261020T173250.jsonl](results/board/opt-jev08-a76-seq2-ub256-20261020T173250.jsonl)，SHA `52976ba0bedf1301c3190cd39e875da925ce38764a2083a0dcac34fe3b2e0788`，控制器exit0。32条记录齐全，含20次计时、8例质量、complete及telemetry，无错误；5项源码/fixture哈希匹配optimization-v1。实际环境为slots2、ubatch256，A76[6,7]和2线程未变。

| 条件 | 本配置P50 / P95秒 | 对原始slots17基线 | 对slots2/ubatch1024 |
| --- | --- | --- | --- |
| 10候选 | 7.132898 / 7.150681 | 慢约0.51%，速度比0.9950× | 慢约2.36%，速度比0.9770× |
| direct100 | 55.927122 / 56.035296 | P50减少1.635344秒，1.0292× | P50减少1.213438秒，1.0217× |

全部28次的case_id、expected、64状态token、渲染token/分块数和完整10/100候选集合均与前两配置一致；所有评分有限，答案均未变化。direct100分数和概率逐值相同；**10候选分数发生变化**，对两前配置最大logit绝对差0.07283783、概率绝对差0.01480284。不能写“所有评分完全一致”；将来若使用概率阈值触发拒答/回退，要另核验阈值行为，即使argmax目前未变。

质量仍为10候选20/20、direct100为4/8（literal2/2、paraphrase1/2、unknown1/2、safety0/2）。没有精度改善或安全通过证据。连续1161样本覆盖621.053秒，无采样错误；RSS峰1940680704字节、PSS峰1932236800字节（比原始基线少399101952字节、约17.1%；比seq2少63197184字节）；系统available最小3329880064字节，温度峰68.912℃，线程峰6，平均计时CPU1.883核，A76采样均2002000kHz。swap存量峰4MiB，本run换入/换出增量均0。载入9.868秒与其他顺序run受文件缓存状态影响，不能当作严格冷加载提升。

**三项结论**：运行优化尝试有效且进一步减少内存，但10候选变慢、100候选观测变快，不能概括为“全场景加速”；尚缺交错重复与更大样本。质量仍保留原有错误，置信度变化需要留痕。部署未通过，待后续常驻争用、真实目标任务质量及异常恢复。自动按10候选P50选配置可能选择seq2而非本配置，这是已声明的选择口径，并非算错。

### 2026-10-21 后续队列与库身份修订

只读核验新版board_identity SHA `653be072f46e144ba61efdd72dd0c676633ad499a3cc51fca77440a52ca80e30`已用配置相同的LD_LIBRARY_PATH调用ldd，记录解析文本并将实际解析文件纳入SHA清单；主任务另在共驻pair_loaded增加scorer进程maps路径。状态为已实现待最终run确认，不追认旧结果。外部CPU服务与NPU图身份覆盖边界仍成立。

新版run_board_optimization SHA `12ffef1ede0eda82cc689e29c3542855c3779139d655a3c7d0c5a15651cfaa4f`支持JEV_RESUME=1：Windows已存在同配置任意opt JSONL就保留原尝试（包括部分/超时），只补未启动配置；大Laya图rsync加压缩/断点、最多3次传输尝试。当前5972仍运行已冻结旧控制器`sources/run_board_optimization_inflight_20261021.py`，新功能仅后续续跑生效；续跑当时PID6704仅是状态标识。补跑前上传新优化脚本会产生新provenance，审计应逐run选择对应源码。不能以“补过队列”推断所有原始超时格完整通过。

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

- [x] **J01 基线**：opt-jev08-a76-t2已实际使用候选评分协议及yes/no读出；10/100候选完整且全部有限，8例direct100答对4例。仅基线采集完成，不代表质量或提速通过。
- [ ] **J02 常驻**：scorer子进程常驻，独立记录载入、首轮、热态；日志、退出状态和板端超时齐全，避免阻塞readline永久占锁。
- [ ] **J03 核线程**：完成B03；记录threads、batch/ubatch线程与子进程亲和性，CPU资源包含scorer。
- [ ] **J04 候选分组**：直接100、合法分块全量与10×10配对；分块覆盖全部候选，列合并规则和总耗时。
- [ ] **J05 原生scorer调参**：比较n_ctx、n_seq_max、n_ubatch、batch与支持的kernel。减少slots/ctx必须容纳内部序列与完整输入，先正确性再测速。slots17→2已完成配对：PSS峰降低14.4%，延迟仅观测改善0.7%–1.8%；ubatch1024→256已回收，10候选比seq2慢2.36%、direct100快1.0217×，再省63.2MB。小幅延迟差尚无交错重复确认。
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
- [ ] **C04 配对**：Nano/Laya与三大脑的基线各模式5次已核验；JevStyle加载失败、OneJev未测、优化优胜及持续decode负载待补。Jev独占、大脑独占、两者驻留空闲、同时推理均测；记录增量内存、TTFT、decode、Jev延迟、温度/频率/错误。仅能加载不证明争用可接受。
- [ ] **C05 稳定性**：5次/条件只是初筛；优胜组合补正式样本、30分钟及后续2小时模拟，核查内存增长、过期/取消、异常。未跑长时明确未测，不由短时成功推断稳定。
- [ ] **C06 后端证据**：CPU/NPU由实际执行确认；1.5B历史NPU质量失败实验不能作为可用部署。未来Jev NPU路径须独立验证，不由现有LLM能力推断支持。

## 8. 优化前后证据表

每个实际尝试填一行，保留失败/变慢配置；加速倍数=基线延迟/优化延迟，须同计时范围，同时报告绝对差。

| 模型/优化 | 基线run_id/代码SHA | 优化run_id/代码SHA | 输入/候选同一性 | n | P50/P95前→后 | CPU/PSS/温度/频率 | 各子集精度前→后 | 并发TTFT/decode变化 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nano33 | TODO | TODO | TODO | TODO | 未测 | 未测 | 未测 | 未测 | 未完成 |
| Laya322 | laya-screen-t2-20261020T164634-10370；原始短基线已回收 | TODO：ARM优化run未验收 | 优化配对待核验 | 基线10候选30次；direct/分层各20例 | 基线1.259/1.271秒（10候选）；优化未测 | 基线整run RSS1.922GB、71.744℃；独立screen无PSS | direct6/20、分层5/20；各unknown/safety0/4，无paraphrase | 三大脑基线共驻各5次已核验；优化配置待测 | 基线结束，优化未完成 |
| JevStyle0.8 slots17→2 | opt-jev08-a76-t2-20261020T171123；脚本91268ab… | opt-jev08-a76-seq2-20261020T172214；同脚本 | 全部case/64状态token/10或100候选一致，评分逐值相同 | 各20次计时、8例质量 | 10候选7.097/7.118→6.969/6.982秒；direct100为57.562/57.658→57.141/57.196秒 | PSS峰2.331→1.995GB；66.670→68.853℃；1.854→1.868核 | 4/8→4/8，全部答案与评分不变；safety仍0/2 | 修复后共驻TODO | 内存减少14.4%；小幅延迟改善待交错重复，质量未通过 |
| JevStyle0.8 ubatch1024→256 | opt-jev08-a76-seq2-20261020T172214；脚本91268ab… | opt-jev08-a76-seq2-ub256-20261020T173250；同脚本 | 输入/候选完整一致；10候选分数微变、argmax不变 | 各20次计时、8例质量 | 10候选6.969/6.982→7.133/7.151秒；direct100为57.141/57.196→55.927/56.035秒 | PSS峰1.995→1.932GB；68.853→68.912℃；1.868→1.883核 | 4/8→4/8；direct100分数一致；10候选概率最大差0.014803 | 修复后共驻TODO | 再减63.2MB，10候选变慢/100候选变快；质量未通过 |
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
