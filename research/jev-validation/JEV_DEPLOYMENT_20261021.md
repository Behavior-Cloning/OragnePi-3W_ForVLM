# Jev 板端部署、优化与 VLM 共驻报告（进行中）

更新：2026-10-21 01:45（北京时间附近）。代码和原始结果以 Windows F 盘仓库为准；Mac 保存可打开的报告副本。最新逐配置结果见 `results/optimization-comparison.json` 和审计文档。本轮尚未完成所有优化、共驻和服务器下载。

## 当前结论

三种文本 Jev 已在板卡 CPU 上实际推理。Nano 的中文输入存在严重分词信息丢失；Laya 与 Jev-Style 的危险状态和不明确请求处理未达到动作控制要求。目前均不能直接作为机器人动作输出。速度和准确率分别验收，不用安全门覆盖模型原始错误率。

第一组 Jev-Style 优化把推理槽位17降至2，10候选P50从7.097s到6.969s，PSS峰值2.331GB到1.995GB，内存降低14.4%。全部28次答案及候选评分保持一致。约1.8%的延迟差尚需交错重复确认；质量仍4/8、安全0/2。

500M/1B的文本decoder实际使用NPU；Jev使用CPU。现有Nano/Laya短时共驻结果支持继续使用NPU大模型以减少A76争用，但尚未包含完整图片前向、语音链路或长时持续决策，不能据此宣称完整多模态agent已验收。

## 模型、版本和下载

|模型|实际状态|用途/限制|
|---|---|---|
|Nano-Jev v1.0，33,360,385参数|完整下载、板端哈希和推理已验证|原生英文RAG交叉编码器；中文动作不适合|
|Laya multilingual 322M|ModelScope完整权重、板端推理；ORT FP32/encoder INT8已导出|CPU基线已完成，板端运行时优化排队中|
|Jev-Style 0.8B Decision-v3 Q4_K_M|完整下载、ARM64槽位评分器已构建并实测|普通聊天生成不能替代槽位评分协议；运行在CPU|
|OneJev1.6-0.8B|ModelScope完整权重已下载|配置记录transformers 5.17与Qwen3.5架构；现有4.57环境尚未试加载，兼容性待实证；图片前向、部署及优化留README TODO|
|JEV-9B / JEV-27B-VL|远程CPU存储服务器下载中|仅下载归档，无GPU是预期配置，不做服务器推理|

远程目录：`/infinite/common/yejy_vla/jev_models`。01:34快照：9B约18.44GB目标，已校验0.339GB、部分文件4.217GB；27B-VL约56.05GB目标，已校验0.472GB、部分文件1.682GB。部分文件尚未完成整文件SHA，不能算已校验权重。SSH可重新认证；Tailscale保持开启。

主VLM范围保留SmolVLM-500M与InternVL3-1B，Qwen2.5-1.5B保留对照和历史证据。Jev ARM运行时来自slim20恢复的llama.cpp源码快照，并非上游声明的441df11提交；完整源码树摘要及编译器见 `sources/runtime-source-provenance.json`。

## 基线性能与质量

板卡A733：6个A55+2个A76，约5.72GiB RAM。下表CPU为A76核6/7、2线程，模型热态常驻。加载时间不混入每句决策。不同模型分词器、候选机制及状态token数不同，不能把这张表当等token算子性能比较。

|模型/路径|计时样本|P50 / P95 秒|该样例池结果|
|---|---:|---:|---|
|Nano，10候选|200次，10条literal重复|1.441 / 1.452|20%|
|Nano，100候选|109条，主动中止后续|14.206 / 14.265|0/109；100 literal+9 paraphrase|
|Laya，10候选|30次，10条literal重复|1.259 / 1.271|21/30|
|Laya，100候选|20条|9.607 / 9.673|6/20|
|Laya，先分组再组内选择|同20条|2.482 / 2.519|5/20；分组正确9/20|
|Jev-Style，10候选|20次，10条literal重复|7.097 / 7.118|20/20|
|Jev-Style，100候选|均衡8条|57.562 / 57.658|4/8；literal2/2、paraphrase1/2、unknown1/2、safety0/2|

Laya的20条为12 literal+4 unknown+4 safety，无paraphrase；两条路径unknown/safety均各0/4。Jev-Style的100候选分两块评分，全部候选都参与；2362–2368 token为两块总和，没有把状态或候选静默截断。以上质量池不同，不能横向比较百分比来排名模型。

Nano的100个中文动作仅产生34种不同token序列，86.57%的描述token为UNK，“看向左侧/右侧”等出现相同编码。这种信息丢失无法用量化或多线程修复；后续仍测其英文原生relevance/sufficiency/groundedness控制，分别评价用途。

## 每个 Jev 的优化要求与执行

|模型|正在验证的优化|验收要求|
|---|---|---|
|Nano33|ORT FP32/INT8、PyTorch动态INT8、batch1/8、A55/A76与1/2线程|中文失效结论保留；英文原生答案也须配对|
|Laya322M|ORT FP32、encoder-only ORT INT8、encoder-only PyTorch INT8、核/线程|决策头保留FP32，检查候选评分及准确率漂移|
|Jev-Style0.8B|序列槽17→2、ubatch1024→256、核/线程|同时核对延迟、PSS、全候选评分与质量|
|OneJev多模态|待兼容运行时和视觉前向跑通后再优化|视觉编码、决策分段计时及共驻，不把下载当部署|
|远程9B/27B-VL|当前仅下载和SHA校验|服务器无GPU，运行优化留后续有算力环境的TODO|

初筛统一3次预热、20次10候选计时、8条100候选质量（四类各2条）。每配置独立进程，配置前等待低于70℃，调用前78℃保护，单配置900秒上限。超时需保留部分结果，不记全矩阵通过；请求内温度不能保证实时中止。

优化候选必须匹配基线20次计时答案与8条质量答案；Nano额外匹配18次英文原生控制。之后按10候选P50选性能配置，供共驻比较。这不是100候选综合最优、部署批准或安全保证。

Jev-Style已回收：

|配置|10候选P50 / P95|100候选P50 / P95|PSS峰值|质量|
|---|---|---|---|---|
|A76双线程、17槽|7.097 / 7.118s|57.562 / 57.658s|2.331GB|4/8，安全0/2|
|A76双线程、2槽|6.969 / 6.982s|57.141 / 57.196s|1.995GB|与基线相同，所有评分逐值相同|
|A76双线程、2槽、ubatch256|7.133 / 7.151s|55.927 / 56.035s|1.932GB|答案相同，10候选评分有数值漂移|

进一步把ubatch降到256，10候选中位数比原始基线慢约0.5%，100候选快约2.84%，PSS降低17.1%。100候选评分逐值相同；10候选最大logit差0.072838、概率差0.014803，但答案不变。不同候选规模的速度方向不同，不能称全面加速。

两槽配置最大温度68.853℃，基线66.670℃；顺序热历史不同，不能归因配置。两槽run swap存量约4MiB但增量0；基线swap-out增加843,776字节。约1.8%/0.7%的延迟改善只是单次顺序观测，内存收益更明显。

Laya导出FP32与encoder INT8已做动态形状检查；10/100候选×64/256状态的四组检查都来自同一请求，不代表四条独立精度样例。INT8量化88个encoder MatMul、头FP32。Nano ONNX FP32最大logit误差8.94e-7，INT8约0.106，仍需板端质量对照。

原始当前队列脚本冻结在 `sources/optimization-v1/`。新脚本增加预热分段计时、预热后各TID亲和与实际默认参数，仅对后续运行生效。最终共驻前将板端重算Jev模型/脚本/bin SHA，每run引用清单；这属于后验补验，不能追认旧run身份，也不包含外部1.5B服务和NPU图的全部身份链。

## 目前的常驻并发结果

源 `results/board/co-resident-20261020T165652.jsonl`。每格5次，Nano/Laya共18格90次；该轮Jev-Style因共享库路径错误失败，已修复并完成单模型评分，最终共驻待补。常驻指整个被测配对期间权重保留，不表示两个VLM同时加载。1.5B对照是CPU文本模型。

|大模型 / Jev|Jev独占→并发P50|LLM TTFT独占→并发P50|decode独占→并发|
|---|---|---|---|
|Smol500 NPU / Nano|1.488→1.587s|2.649→2.693s|17.49→17.43 tok/s|
|Smol500 NPU / Laya|1.322→1.411s|2.631→2.696s|17.48→17.39 tok/s|
|Intern1B NPU / Nano|1.459→1.551s|0.927→0.951s|13.74→13.50 tok/s|
|Intern1B NPU / Laya|1.316→1.406s|0.924→0.965s|13.76→13.50 tok/s|
|Qwen1.5B CPU / Nano|1.456→2.865s|1.343→2.832s|9.76→9.60 tok/s|
|Qwen1.5B CPU / Laya|1.325→2.501s|1.360→2.666s|9.67→9.66 tok/s|

全部30个并发调用有真实时间重叠，但Smol两组及Qwen+Laya中Jev均在首token前完成；Intern的decode重叠仅0.37–0.80秒。不能据小decode降幅推断持续决策也无竞争。优化后的配对将每组使用独立进程，并追加持续Jev循环覆盖生成窗口，记录每次决策和LLM的时间区间。

NPU worker CPU侧核4/5、Jev6/7。Smol系统前缀92token，按块实际复用64；Intern前缀37token、实际复用32。输入模板不同，TTFT不能直接推导模型大小效率；此处也不是此前长固定前缀15.5tok/s的同一场景。该轮对应生成token序列配对5/5相同。

## CPU、内存、温度与系统健康

|运行|CPU平均计算占用|内存观测|温度与系统可用内存|
|---|---|---|---|
|Nano基线10候选|约1.964核，整机8核口径24.6%|请求边界RSS最高486MB；因中止缺完整连续峰值|边界最高72.334℃，可用最低约4.09GiB|
|Laya完整基线|见原始逐调用CPU数据|连续1162样本RSS峰1.922GB|71.744℃，可用最低2.861GB|
|Jev-Style17槽|约1.854核，整机23.2%|连续1185样本PSS峰2.331GB|66.670℃，可用最低2.930GB|
|Jev-Style2槽|约1.868核，整机23.3%|连续1164样本PSS峰1.995GB|68.853℃，可用最低3.267GB|
|首轮共驻全部组|非持续压力负载，见逐进程采样|612快照含414连续样本；进程集PSS峰3.822GB|最高65.224℃，可用最低2.092GB|

GB为十进制字节，GiB为二进制。进程PSS、系统MemAvailable、NPU driver分别报告，不简单相加。单Jev阶段原有CPU 1.5B服务仍常驻，不能把系统净余量当Jev自身占用。

共驻swap最高51,118,080B，swap-in增加393,216B、swap-out增加50,237,440B。NPU driver加载分别约702MB/753MB，两个worker关闭后计数回到0。该轮顺序轮换时Python仍保留Laya分配器内存，不能据它推断各模型独立内存；最终配对改为新进程。

板卡此前有MMC DMA超时；前后dmesg环形日志已经完全轮换，后快照未出现MMC/ext4/OOM，不能据此排除整个实验区间发生过事件。另有预先存在的 `Cannot set pwm` 与PD5引脚占用冲突，涉及内核温控PWM配置。当前没有修改GPIO/散热配置，需结合实际硬件单独排查；不能仅凭观测温度有界断言温控设备正常。证据与限制见 `results/dmesg-pairs-delta.json`。

Windows 01:22快照：F盘可用60,218,482,688B（约56.1GiB）；Windows内存可用约24.0GiB，vmmem工作集约2.24GiB；WSL配置16GB，使用约472MiB、可用约15GB、swap0。后续模型传输和编译会改变这些数值。

## INT4 与历史1.5B实验

恢复工作树确有Qwen2.5-1.5B完整NPU实验：28层decoder、4个全词表head分片，ctx128，输入44/输出64，5次运行。W4-vdfp-sensitive group28的decode P50为5.032149tok/s、TTFT 8.042304s、driver约675MiB；输出重复，质量失败。

其W4值存于INT8的−7…7，并无证据证明原生packed INT4加速。重量化常量的FP32 oracle也重复，不能把这组性能称作可用INT4。500M/1B目前没有完整合格INT4结果。历史源与1.5B对照包均保留，证据位于恢复工作树 `qwen15-rwkv-worktree/docs/research/Qwen15-HANDOFF.md` 及其speed summary。

## F盘清理与保留

|删除内容|字节|回收口径/清单|
|---|---:|---|
|28层可再生int16编译中间export.data|990,086,768|WSL内部；`cleanup-20261021.json`|
|已安装的WSL MSI与Ubuntu rootfs下载缓存|708,706,003|直接F盘；`install-cache-cleanup-20261021.json`|
|未用于构建的中断llama源码压缩包|12,251,136|直接F盘；`cleanup-abandoned-source-20261021.json`|

直接F盘回收共720,957,139B；WSL内部另回收约0.92GiB，尚未压缩VHDX，不能把两者相加声称F盘物理释放。清理后还新增权重，因此可用空间快照不会简单增加这些数值。原始权重、最终NB、活跃FP32 oracle需要的文件、TF备份、slim20恢复包、语音模型、500M/1B复现包及1.5B历史证据全部保留。

## 继续执行与审计

- [ ] 完成Nano/Laya/Jev-Style逐配置优化；每个模型记录尝试、失败、速度、质量和资源，不能只登记部署。
- [ ] 回收优胜配置与500M NPU、1B NPU、1.5B CPU的独立配对和持续决策竞争。
- [ ] 在有可用质量的配置上扩展独立样例和稳定性；20次计时/8条质量不是1000次或长时验收。
- [ ] 多模态OneJev的运行时、真实图片前向、视觉编码和VLM共驻；已明确列README TODO。
- [ ] 远程9B/27B-VL整文件下载完成后核对全部SHA；当前仍在下载。
- [ ] 接入确定性安全门、mock回执、状态版本和过期结果取消；已有mock单元测试不等于完整agent集成。
- [ ] 温控PWM/PD5冲突结合实际硬件排查；保持原始健康证据。

优化控制器PID5972持实验锁；后续队列PID6704等待优化结束再执行配置筛选与共驻。后续队列会补从未启动的配置，并保留已有超时/部分结果不冒充通过。每配置完成即回收JSONL到Windows，审计智能体仅读取回收文件，不争抢板卡。运行状态以 `results/continuation-status.json` 及控制器日志为准，PID只是本次快照。

远程下载于01:39切换为每仓库最多8并发，保留此前所有已验文件和断点；新任务PID123340/123341。传输异常日志现在记录curl退出码，尚不能声称带宽改善或下载完成。

## 自动回收的优化初筛结果（非最终部署验收）

|模型|配置|完成|计时n|P50秒|P95秒|质量正确/8|相对基线速度倍数|
|---|---|---|---:|---:|---:|---:|---:|
|jev08|a55-t2|False|20|27.7969|27.8836|0/0|0.255|
|jev08|a76-seq2|True|20|6.9688|6.9824|4/8|1.018|
|jev08|a76-seq2-ub256|True|20|7.1329|7.1507|4/8|0.995|
|jev08|a76-t1|False|20|13.4982|13.5047|3/5|0.526|
|jev08|a76-t2|True|20|7.097|7.1176|4/8|1.0|
|laya|a55-t2|True|20|5.6847|5.7247|2/8|0.224|
|laya|a76-int8|True|20|1.9321|1.955|2/8|0.66|
|laya|a76-ort-fp32|True|20|1.059|1.0629|2/8|1.203|
|laya|a76-ort-int8|True|20|0.5289|0.5337|2/8|2.41|
|laya|a76-t1|True|20|2.26|2.2644|2/8|0.564|
|laya|a76-t2|True|20|1.2744|1.299|2/8|1.0|
|nano33|a55-t2|True|20|7.0389|7.066|0/8|0.204|
|nano33|a76-batch1|True|20|1.8566|1.8627|0/8|0.773|
|nano33|a76-int8|True|20|2.2301|2.2385|0/8|0.643|
|nano33|a76-ort-fp32|True|20|1.2219|1.2334|0/8|1.174|
|nano33|a76-ort-int8|True|20|0.5523|0.5536|0/8|2.597|
|nano33|a76-t1|True|20|2.5693|2.5795|0/8|0.558|
|nano33|a76-t2|True|20|1.4344|1.4426|0/8|1.0|

每格仅20次计时与8条质量初筛；原始JSONL、资源峰值、决策差异见optimization-comparison.json。Nano中文输入失效结论不因速度改善而改变。

### 优胜配置实际共驻回收

- co-resident-optimized-intern1b-jev08-20261020T193140.jsonl：文件结束标记=True；错误数=0；有效条件={'intern1b/jev08/jev_only': 5, 'intern1b/jev08/llm_only': 5, 'intern1b/jev08/concurrent': 5, 'intern1b/jev08/concurrent_sustained': 5}
- co-resident-optimized-intern1b-laya-20261020T192956.jsonl：文件结束标记=True；错误数=0；有效条件={'intern1b/laya/jev_only': 5, 'intern1b/laya/llm_only': 5, 'intern1b/laya/concurrent': 5, 'intern1b/laya/concurrent_sustained': 5}
- co-resident-optimized-intern1b-nano33-20261020T192826.jsonl：文件结束标记=True；错误数=0；有效条件={'intern1b/nano33/jev_only': 5, 'intern1b/nano33/llm_only': 5, 'intern1b/nano33/concurrent': 5, 'intern1b/nano33/concurrent_sustained': 5}
- co-resident-optimized-qwen15-cpu-jev08-20261020T193947.jsonl：文件结束标记=True；错误数=0；有效条件={'qwen15-cpu/jev08/jev_only': 5, 'qwen15-cpu/jev08/llm_only': 5, 'qwen15-cpu/jev08/concurrent': 5, 'qwen15-cpu/jev08/concurrent_sustained': 5}
- co-resident-optimized-qwen15-cpu-laya-20261020T193702.jsonl：文件结束标记=True；错误数=0；有效条件={'qwen15-cpu/laya/jev_only': 5, 'qwen15-cpu/laya/llm_only': 5, 'qwen15-cpu/laya/concurrent': 5, 'qwen15-cpu/laya/concurrent_sustained': 5}
- co-resident-optimized-qwen15-cpu-nano33-20261020T193438.jsonl：文件结束标记=True；错误数=0；有效条件={'qwen15-cpu/nano33/jev_only': 5, 'qwen15-cpu/nano33/llm_only': 5, 'qwen15-cpu/nano33/concurrent': 5, 'qwen15-cpu/nano33/concurrent_sustained': 5}
- co-resident-optimized-smol500-jev08-20261020T192447.jsonl：文件结束标记=True；错误数=0；有效条件={'smol500/jev08/jev_only': 5, 'smol500/jev08/llm_only': 5, 'smol500/jev08/concurrent': 5, 'smol500/jev08/concurrent_sustained': 5}
- co-resident-optimized-smol500-laya-20261020T192159.jsonl：文件结束标记=True；错误数=0；有效条件={'smol500/laya/jev_only': 5, 'smol500/laya/llm_only': 5, 'smol500/laya/concurrent': 5, 'smol500/laya/concurrent_sustained': 5}
- co-resident-optimized-smol500-nano33-20261020T191931.jsonl：文件结束标记=True；错误数=0；有效条件={'smol500/nano33/jev_only': 5, 'smol500/nano33/llm_only': 5, 'smol500/nano33/concurrent': 5, 'smol500/nano33/concurrent_sustained': 5}

持续决策窗口的具体时间、每次case和LLM指标保留在原始记录中；需人工审计后再决定推荐。未涵盖视觉encoder、全语音链路、多模态OneJev、长时稳定性与1000次尾延迟，不将这些TODO勾选。
