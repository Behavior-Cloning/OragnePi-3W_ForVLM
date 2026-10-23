# 完整 Llama NPU 独立审计

2026-10-23。本审计只读 Windows F / WSL 的源码、编译清单及回传结果，不访问板卡、不持板锁、不编译或运行模型、不提交Git。唯一编辑本文件。板卡owner为intern_operator_opt；Q8首层由另一个owner处理。

当前状态：Q4源20图已编译，实际板端结果尚待回传。本文件的代码预审不能代替完整运行或质量验收。Q8已存在主机FP32自然参考，但NPU只在准备独立首层probe，不能标为完整Q8已执行。

## 执行范围核对

`deploy_run.py` 的完整执行必须有20图manifest，编译runtime使用 DIM2048 / KV8 / HD64 / FUSED_LAYERS1 / HEAD_SHARDS4 / HEAD_CHUNK32064。`resident_session.c` 加载layer0..15并加入16-network group；每个prompt token及生成反馈token都调用该group。没有CPU decoder或CPU prefill回退，prefill是逐token NPU，未使用批16优化。

末层hidden由CPU读取并复制到四个head输入，四个head一起由NPU执行。`model.HeadShard`包含final RMSNorm和linear；范围依次[0,32064)、[32064,64128)、[64128,96192)、[96192,128256)，无漏词表或top-k裁剪。CPU逐头扫描完整输出选greedy最大值；diagnostic模式遍历并保存全部128256维FP32 logits，且用完整扫描再次核对选词结果。头输出FP16不代表词表只算部分token。

CPU边界是embedding lookup、scaled RoPE角度/cos/sin构造、KV复制、调度与greedy；rotary乘法、所有层的attention/MLP/norm以及最终norm/词表线性在NPU。CPU FP32 embedding mmap与NPU头副本同时存在，源权重共享不等于运行时内存共享。

层间hidden通过vip_set_input别名链接；各层KV属于自己层。group完整运行后回写各层新KV，下一token使用；后层只依赖前层hidden，不依赖前层新KV，因此该延后写回符合此逐token因果结构。运行前核对hidden IO合同为FP16，无CPU误flush链接hidden。退出正常路径销毁两个group、所有20个网络与VIP context，再记录driver_bytes。

20图manifest的索引完整0..19，总NBG字节1,593,110,816；各图ONNX与本源FP32算术检查最大相对L2约1.255e-6。四头均FP16；decoder配置为attention FP16、MLP三FC W8及邻接A16，其余敏感路径FP16。抽查第0/15层生成C各3个INT8声明与11个INT16声明；第16/19头均无INT8/INT16声明。精度范围应以全部20图的编译证据和最终板端IO/SHA共同判断，不能只看配置名称。

## Tokenizer、RoPE与缓存

输入源为Q4_K_M GGUF，SHA256 `8bcd46580c2eb069fde82b4c16f8f2981901d653fea1340ec560b6097f08f97a`。Q4_K_M实含Q4_K/Q6_K/F32；解量化后再编译为混合NPU格式，不是原生NPU INT4，也不是原厂BF16参考。

`paligemma/llama-zh/oracle-natural/tokenizer-parity.json` 保留官方tokenizer SHA `bbc1904d35169c542dffbe1f7589a5994ec7426d9e5b609d07bab876f32e97ab` 与9组中文、数字日期、英文换行、system和三个完整prompt的逐token对照，全部一致。Q4/Q8源tokenizer元数据已核对相同。当前frozen inputs与完整prompt对照分别为38/41/42 tokens；system22，动态16/19/20，EOS集合128001/128008/128009。

RoPE从GGUF rope_freqs构造逆频率，CPU cos/sin与NPU乘法分开；未复用旧仅base500000的错误Llama3公式。freeze_inputs已有官方Transformers Llama3频率对照。head保持完整词表及原模型三个终止token。

OPI_PREFIX_LIMIT通过命令里的env显式传22，虽ResidentWorker清理外部OPI环境，该命令仍会设置此值。匹配历史前缀后硬截到22，禁止额外重用上个user/assistant。每次truncate会关闭历史后缀mask；旧KV即使留在buffer中也不可见。每请求控制器断言无cache组reused=0、system组reused=22。system preparation独立reset后生成1步，从而填满22-token system KV；时间在后续TTFT之外。再次请求虽然共享同一常驻session，动态user部分重新执行。后续raw应核对这些断言的实际值与cached/no-cache自然输出一致性。

## 自然生成、数值诊断与计时边界

本轮明确为三题pilot，quality-cases.json的十二题尚未被runner消费；QUANT_COMPARISON_PROTOCOL建议的更多题/重复次数是后续质量可用时的方案，不是本轮完成门槛，不能为凑样本重复坏输出。

执行计划分开：3个无缓存fixed-feedback诊断，每题取同源FP32 oracle前8步并dump完整logits；3题自然无cache（max96、按原EOS结束）；独立system准备；3题自然system-cache；3题system-cache固定32步反馈。固定32不足时补EOS且强制继续，不可当作32有效自然内容tokens。diagnostic的dump开销不应混入吞吐结论；summary分组保留。

同源FP32自然参考另在oracle-natural/q4/results，max96，前三题81/96/47 tokens（含实际EOS），原32步诊断前缀已逐token对齐。不能因自然输出分叉后上下文不同，就直接把随后自然logits差异当同条件量化误差；首8步fixed-feedback才满足同反馈历史条件。

C引擎TTFT从完整请求payload读取后开始，包含reset/truncate/setup、所有新prompt标量NPU执行及首个完整头选词，排除模型加载、前次system准备与Python传输。token elapsed和done总时间是引擎边界；控制端另存start/end，不能互换。decode=(generated-1)/(total-first)，自然组可能含EOS；fixed32应注明31次首后预测。getrusage(RUSAGE_SELF)提供resident进程真实CPU时间，不依赖含空格comm解析。Python遥测另使用最后右括号后stat字段11/12，规避此前Pali parser问题。

CPU平均核数应注明是C进程CPU/请求墙钟。summary当前按请求核数做算术均值；若要报告整个阶段平均应按持续时间加权。系统busy以整段遥测首尾/proc/stat前8字段扣idle/iowait计算，包含加载和请求间隙，不能称精确热请求CPU。RSS/PSS为Python+直接resident子进程之和，driver单独记；RSS不覆盖全部NPU内存。采样频率当前包含CPU策略，未连续记录NPU频率，不能宣称已排除一切NPU降频。

## 已发现及修复中的资源问题

原board_run.py没有TERM/INT handler，外timeout可能跳过finally丢遥测；温度monitor只置flag，要等下一个token才退出，长prefill期间不能立即中止，且缺MemAvailable低阈值。已在启动前通知owner/root。

候选board_run_guarded.py静态复核：TERM/INT/USR1转RuntimeError；每250ms采样温度≥78°C或MemAvailable<256MiB即SIGUSR1主线程，打断ResidentWorker的等待/构造；其BaseException路径abort自己的resident。finally忽略重复信号，停止子进程、保存guard/telemetry与driver mem_profile。没有创建另一个独立resident session，因此不复现旧Pali隔离进程组遗留方式。WSL test_guards.py 实际注入TERM、热阈值与低内存三种路径，guard-test-results.json均记录清理和telemetry保存通过；这属于假worker故障回归，不是板端超阈值测试。仍需确认最终实际传输文件为修复版，不能把候选源码当运行证据。

监控是采样保护，不是严格硬限。只有正常closed且driver_bytes=0或异常后可核对driver profile归零才能确认释放；仅命令exit0不足以代替完整完成和质量通过。不得为审计而访问板卡。

## 结果验收待办

回传后独立核对20图传输SHA、runtime/runner快照、所有request数量与缓存计数、全词表logits原始文件shape/finite/SHA、相对L2/KL/argmax及自然文本；区分源模型已有缺陷与NPU新增漂移。复算自然/固定反馈耗时、CPU、温度和driver释放。Q4/Q8/Intern之间还需独立来源与量化范围说明；当前Llama主FC约65.17% W8，而Intern约98.23% W8，不能因都叫W8就称同等量化公平结论。

预审时源码SHA（活动源码后续可改变，以最终执行快照为准）：

- `resident_session.c`: `ddb293a3332ff11f6c5500a369c69de6bec74877f7ff7306d947a2aa7726ef6d`
- `resident_worker.py`: `2c110a6821ce9ee21aa2c07b062f27a49e18a9289e0fd8aea12927f80448321d`
- `board_run.py`: `60db50379a964a6d96e8615902de69212acee0300b4f323cc25465ecf8c6d2da`
- `board_run_guarded.py`: `c47003c958b39131e94431b8b4e12471e86ee06f488c5fdda4b3fc12179e71e1`
- `deploy_run.py`: `0534d753af5631c66c7ca85ca57d08fa177d5e0621f6e05cc86f8874e99fa346`
- `summarize_board.py`: `08a3c2886f33fdfcbddcd686324859701dd6cd07ec77cb115b34d125f3846e73`
- `model.py`: `f8ee1b6a3fe2d094046ef6c949ffa98d8833b10119b4a972beb491671ea97c7f`
