## 2026-10-27：r11完整M64替换评测已完成，未达到500ms

- 当前板端测试已结束，controller rc0；117计时、36自然、18完整logits，3次driver0，post-health alloc/free1077/1077。报告research/intern-prefill100-20261025/main-m64-r11/full-test/REPORT.md；tag exp/intern-main-m64-r11-20261026。此前在测章节为历史快照，勿重跑。
- 静态32/64/128=正文27/59/123+模板5；动态32/64/128/192/256+模板8=new40/72/136/200/264；只复用system。输出固定32，decode后31；上游语音/摘要/system构建不计入TTFT。覆盖完整13格并集，每格3例。
- 同轮system64/dynamic256 prefill904.2→863.5ms、TTFT923.5→884.1ms；system128/dynamic256较快原版prefill965.8→902.1ms。短dynamic32约187→216ms退步。主图替换不新增同时驻留主bank，仍增约28.6MiB NPUready；CPU7、前端CPU6。
- candidate6份logits均与原版有差异，relativeL2约2.72–3.02%（非准确率）。自然10/12同序列；7完成4漏1冲突，后退→向前仍错。r6仍资源/短输入优先；r11只登记长输入最低观测；qualitybest仍空、默认未改。
- 约7.04min采样，峰温52.452°C，候选CPU8核归一P50约6.31%，最低available419.9MiB，swap近满；累计音频lag1.551s、约5%帧超16ms，不能说全程实时或长稳合格。500ms全范围目标未达成。
- 18份板端rawf32与Windows逐SHA核对后已删除副本，释放43,682,112B，/opt剩573,566,976B。保留候选模型便于复测；Windows源/大图/rawf32保留，Mac将同步可打开报告；Github状态以F:/Files/OPi-Lab/transfers/github-sync-verified.json独立回执为准。下一轮继续NPU算子/分块同输入比较，不将短输入提速冒充dynamic256达标。

## 2026-10-26：恢复工作，r10筛查完成，r11主M64替换在测

- r10结果ec78d66，tag exp/intern-joint-attention-r10-20261026：6张图上板、10probe/300API样本，scores慢10–40%，value无一致收益且改变输出张量。不扩整模，不影响r6性能指针、质量最优空及默认。
- r11已编译12张M64/P448主图，复用r8的L0/L2，其余10张新编译；不新增同时常驻主图组。运行时main32/main64同binary，CPU7，frontendCPU6。logical32 cache边界保证system32确实复用32；dynamic32的main64计算40有效/64物理行，需单列padding。
- controller：research/intern-prefill100-20261025/main-m64-r11/full-test/controller.json；由controller.py持双锁，独立板目录/opt/opi-vlm/experiments/main-m64-r11。只查WSL状态/进程，不重复启动、不绕锁轮询板卡。上传后顺序main32-before/main64/main32-after。
- 冻结输入仍r6同SHA；system正文27/59/123+模板5=32/64/128；dynamic正文32/64/128/192/256+模板8=new40/72/136/200/264；只复用system，固定输出32。13格117计时、36自然、18完整logits。须核对原版六诊断与r6同字节，候选独立质量评价，不能先称无损。
- GitHub最近已核验c4cabe6，本次新增提交尚未同步。使用Mac本机http.proxy=http://127.0.0.1:7897和bare relay；勿恢复旧Windows SOCKS。500ms全范围目标仍未达成。

## 2026-10-26：r9同量化gate/up合并筛查完成，不推广

- 当前无板端测试运行。源码f3b109f、冻结runner5d7b539、结果4c6e9e0；tag exp/intern-gateup-fusion-r9-20261026。报告research/intern-prefill100-20261025/gateup-fusion-r9/REPORT.md。
- attempt1因节点ID非整数后缀在图加载失败，未产生板端样本；attempt2两图成功，按M32/P448、M64/P128分别A/B/A，180计时/6进程。只融合L3，源权重值及量化编码参数保持，继承outlier8父量化。
- 局部有效历史128行、query32/64行；system/动态文本/模板/生成token/整模prefill/TTFT/decode不适用。M32 API14.753→14.734ms仅0.13%，小于原版前后漂移；M64 19.471→19.902ms约慢2.21%。两形状完整hidden/KV同字节，不能外推语义正确性或全模型提速。
- 6次driver0；37份文件对照Windows SHA后释放181870066B。Windows保留图和rawf32，默认未改，完整性能候选仍affinity-m64-r6，质量合格最优仍空。新版本登记/verify_versions更新。
- 下一步检查注意力past/new分开矩阵计算能否减少重复调度。不能改token数、截历史或复用动态KV来宣布达到500ms。r6 system正文59/123+模板5=64/128，动态正文256+模板8=new264、只复用system，固定反馈输出32；prefill仍约0.90–0.97秒，目标未完成。
- 本节提交的GitHub同步状态单独以F:/Files/OPi-Lab/transfers/github-sync-verified.json为准；后面旧章节的远端SHA是各自当时快照。

## 2026-10-26：r7耗时定位与r8完整历史M64局部筛查完成

- 当前无板端测试运行。r7控制器rc0，源码8843be3，报告a8c100a，tag exp/intern-timing-r7-20261026；同r6图/CPU7只加入计时。6格54性能/36自然/15完整logits，数值同例同字节、三个driver0。
- system正文123+模板5=128、动态正文256+模板8=new264、固定输出32：r7 trace-on prefill P50 966.9ms，其中NPU API940.8ms（97.3%）、输入准备21.8ms、发布3.9ms。SDK分网络profiling返回0不能使用；API包含驱动等待，不等于纯硬件计算。CPU小修难以达到全范围500ms。
- r8源码6415a67、冻结runner731ae40、原结果bf4244e、报告措辞修订275d311，tag exp/intern-m64-p448-screen-r8-20261026不移动。仅L0/L1和L2/L3：M64P448 API17.630/23.139ms，M32P448较快参考9.698/14.901ms，单位行吞吐约1.10/1.29倍；不是整模结果。hidden相对L2差0.04147%/0.10712%，不等于准确率下降。
- r8全部10probe×30重复一致/finite/driver0，全部53板端文件SHA对照Windows后释放178789412B。Windows编译图与rawf32保留，可新run复现；默认与r6完整性能指针不变。
- r7资源采样3.93min，峰50.034°C，最低available465.6MiB，swap满，语音723/14737采样帧超预算、累计lag1.755s，不能称实时/长稳合格。自然12回归仍6完成5漏动作1方向冲突。
- 下一步优先NPU矩阵算子实现/合并。实际图每层7个FC、Q/K/V及gate/up单独；合并前必须检查是否同源输入及各自量化参数，不能将相同数学式当bitwise相同。R8局部吞吐不足证明full256达到500ms；不要用小输入或截历史替代完整目标。
- GitHub已通过Mac本机HTTP代理和bare relay完成推送；独立ls-remote核验分支codex/wsl-intern1b-reproduction=c96d22a，8个实验tag及其peeled commit全部匹配。代码仍以Windows为权威；本次不改默认部署。同步收据见research/intern-prefill100-20261025/REMOTE_SYNC_STATUS.json。使用完成后用ssh -S /tmp/vlm-win-control -O cancel -D 127.0.0.1:10847 pc@100.113.74.31撤销本轮forward。

---

以下为历史交接快照；任务是否仍运行须以最新控制器及实际进程为准，不按旧记录重启。

## 2026-10-26：CPU7亲和性r6已完成，更新性能候选（默认未改）

- 无测试在运行。实现ada85e7，报告a530d33，tag exp/intern-affinity-m64-r6-20261026。r6复用M64r4 binary/图，只把worker CPU4,5→CPU7→CPU4,5，语音CPU6；完整13格117性能、36自然、18诊断，rc0/三个driver0。
- 仅system KV，system正文59+模板5=64、动态正文128+模板8=new136：prefill421.1ms、TTFT440.3ms；system正文123+5=128同动态prefill484.8ms、TTFT503.4ms。输出固定反馈32token，decode约16.3，较同轮较快A/A逐案提升3.96%以上。动态256 new264的prefill约904.5/968.6ms、TTFT923.4/987.6ms，完整500ms目标仍未完成。
- 同binary/图/量化/输入，不是NPU算子提速；全部18份full logits同案例同字节，39/39固定反馈和12/12自然回归输出相同。但自然仍6完成5遗漏1方向冲突，质量合格最优为空，生产默认未改。
- 无新增NPU模型bank；ready约1272.3MiB，CPU7阶段最低available455.3MiB，CPU P50/P956.31/14.88%，峰温51.522°C。swap仍满/有换页、前端约4.9%超帧预算、全程累计lag1.759s；7.27min采样不能称长期实时合格。
- 当前资源优先性能候选affinity-m64-r6，逐格最低观测也指向本轮CPU7；M64r4作为依赖/历史保留，r5历史不覆盖。已修正manifest中r5“仍运行”的过期描述；r5已完成且extra bank已释放。
- Mac报告副本需同步本轮REPORT/ANALYSIS/QUALITY_REVIEW以及BEST/manifest/verifier/audit。Windows权威，所有原始结果SHA保留；模型/rawf32不入Git。
- 继续优化应优先定位长输入prefill的图执行/CPU数据准备分段，或验证CPU7与其他候选组合；新run ID、双锁，不能直接相加历史收益。语音/RAG驻留但RAG未查询、纯NPU文本，无视觉/真实麦克风/电机。
- GitHub本轮ls-remote在25s超时；远端同步仍未确认，不声明新提交已上传。

## 2026-10-26：双past256 r5已完成，未推广；下一步CPU调度

- 无测试在运行。r5实现762dbfc、完整13格报告17e914f、tag exp/intern-compact-m64-p256-r5-20261026。完整500ms目标仍未完成，decode没有确认提速。
- 相同输入A/B/A：system正文123+模板5=128、动态正文128+模板8=new136，候选prefill500.9ms、TTFT523.8ms，父版重复A525.0/546.0ms。system64动态256 TTFT979.3ms，system128动态2561031.5ms；性能输出固定反馈32token，仅system KV。
- 代价：额外NPU ready314.5MiB，最低available115.6MiB，swap已满（本轮页换入出0）；语音累计lag1.291s。采样7.73min、峰温53.134°C、NPU始终1008MHz performance。
- 12条已有自然回归输出一致，仍6完成/5遗漏/1明确方向冲突。18份四步full logits finite，三次关闭driver0；不能视作质量合格或长期稳定。
- 删除本次新增板端 /opt/opi-vlm/experiments/compact-m64-p256-r5/compact256：12张图SHA与Windows一致、driver0确认后释放297655008B；原始结果、Windows图、父模型保留。三个局部probe tmp目录另释放208346794B，全部逐SHA核验。
- 版本选择：M64r4保留为资源优先候选，新增逐格最低观测（r5长输入部分更快），质量合格最优为空，默认未改。VERSION_MANIFEST与verify_versions校验5矩阵138行token/60份full logits。
- 否定结果softmax-i16-r1：API13.79ms无收益、hidden改变，未推广。past256局部probe已完成，不外推为全模型成绩。
- 下一步建议同一M64r4 binary/图/inputs、CPU4,5→CPU7→CPU4,5的A/B/A，语音固定CPU6，RAG驻留但不查询。先不加载r5额外图，完整记录prefill/TTFT/decode/CPU/温度/内存/audio lag/NPU频率与每个token口径。不要修改既有run IDs。当前控制器已终止，必须继续双锁。
- GitHub近期推送与复核超时，未宣称新提交已同步。Windows为权威仓库，Mac仅报告副本。

## 2026-10-26：M64性能候选与版本追踪更新

GitHub同步待确认：代码/报告636534f已在Windows提交；HTTPS推送超时，随后ls-remote也超时。最后成功查询仍为5aee7ee，不能声称远端已更新。完整回执见research/intern-prefill100-20261025/github-sync.json。

- 性能候选compact-m64-r4；源码4a4c57e，结果328918c，tag exp/intern-compact-m64-r4-20261026。不是生产晋升，默认未改，质量合格最优为空。
- system正文59+模板5=64，动态正文128+模板8=new136，只复用system64：prefill464.5ms/TTFT485.6ms P50，每格3案、输出固定反馈32token。system128同动态TTFT545.9ms；动态256约1.01–1.06s；decode约15.3–15.6，未确认decode提速。
- 共13格，覆盖static32/64×dynamic32/64/128/256及64/128×64/128/192/256；正文与模板分列见BEST_VALIDATED。32静态截断、128静态重复补齐仅成本用途。
- 新12自然请求父版/候选各6完成、5遗漏、1方向冲突（取消前进要求后退却向前）。候选一例低头退化、一个弯腰改善；不得执行电机或把10/12输出parity称准确率。
- 7.58分钟采样，最高51.65°C，最低available456MiB；候选NPU增加305MiB，语音仍有1.57s累计滞后/swap。三次关闭driver0。
- M64独立图在板卡 /opt/opi-vlm/experiments/compact-m64-r4，当前无worker运行；保留父图，不覆盖默认。Softmax局部形状改写略慢，不推广；M32仅编译。
- 权威报告 research/intern-prefill100-20261025/compact-m64-r4/full-test/REPORT.md。VERSION_MANIFEST分开baseline、最快候选、原量化数值验证、质量合格最优、默认；verify_versions.py已新增M64和局部否定结果校验。
- 500ms完整范围与质量目标仍未完成，下一步优化past128之后的prefill并保持同输入A/B/A。此轮只有Intern NPU文本、合成语音、RAG驻留，无视觉/真实麦克风/电机。

## 2026-10-26 尾段优化完整模型结果已回收

- compact-row-down-r1 局部筛查完成：C128/history128，M64 为435.17→20.25ms，M256 为1760.67→102.40ms；A/A输出字节一致，row路径输出均finite，hidden相对L2约0.294%/0.300%，6次driver0，最高45.322°C。该结果只覆盖两层API，不含整模head/TTFT/语音/RAG，不能当整模成绩。
- C256专用probe binary已编译（支持context256）并重新运行成功；结果回收至`results/compact-c256-row-r1-v2`。M256 plain局部API P50 90.108ms/P95 90.243ms，A/A字节一致；row P50 109.017ms且相对L2误差0.4606，慢约21%，明确淘汰row组合。该90ms只覆盖两层API，不是整模TTFT，不能宣称目标完成。
- C256板端探针首次运行失败：`compact-c256-row-r1` 上传与哈希校验通过，但现有C128版 `tensor-probe-compact` 在 `m256-before` 返回码2；无速度/精度结果，不能重试同一组合。日志显示C256图端口已是256行，下一步需编译C256专用probe binary，再重新做A/B。
- 为避免C128对动态256的历史截断，已离线编译C256/M256 plain与row图（`graphs/compact-c256-m256/`），保留past256和query256，导出均rc0；这是图编译证据，不是板端速度或整模结果。下一步需建立C256 compact probe和worker接口审计，再测试板端。
- 不能直接把该图替换进整模：当前 `board_fulli16.py` 和 `resident_worker_v1.py` 仍强制capacity=512、原始图输入协议；compact图是capacity128的past布局，直接替换会破坏动态256和缓存位置口径。下一步必须建立C256/完整worker接口配对，保留system128与dynamic256的全部有效上下文后再上板。
- down-row-scale-tail-r1 完成，96性能/192自然/32诊断/8prefix/8cache，四次driver0；Windows结果目录及Mac报告已保存。
- system64/128含模板5，正文59/123；动态正文64/128/192/256+模板8，new72/136/200/264；仅system KV复用。主批量+尾段TTFT P50在两system下约370–373/653–668/935–979/1218–1290ms，decode约15。相较仅主批量，每格额外省约47–52ms。各格3例，100ms目标未达，不是长期稳定认证。
- B→C两条自然输出变化：一条仅去句号；另一条从复述场景变成“右臂轻轻挥一下”，但未明确低幅/停住等完整约束；其余先前头部对象缺失问题不因此解决。独立审计进行中，未推广默认。
- 最高观测51.460°C，资源含加载与诊断；存在swap和3629/73760音频overrun，不宣称全程实时。
- continuation PID53069已接续compact-row-down-r1，目前上传/运行阶段；只读WSL controller/status，不重复启动或轮询板卡。该实验为两层M64/M256 history128，非整模TTFT；有效历史不截断。

## 2026-10-26 当前 prefill 优化进度（尚未达到100ms）

- `down-row-scale-prefixsafe/results/down-row-scale-prefixsafe-r1` 完整 A/B/C/A 已结束，归档53文件、96性能/192自然/32诊断/8前缀诊断/4重复缓存探针；四次driver释放0。固定system64/128含5模板、动态正文64/128/192/256另加8模板，实际新增72/136/200/264，只复用system。
- 保留原system、仅动态row-scale：TTFT P50约423–434/703–721/983–1026/1270–1340ms，decode约15tok/s；各格3例，不是稳定性认证，100ms目标未达。完整logits有数值差异。48自然输出45完全相同，其中两条改写/标点兼容，一条头部动作丢失对象；RAG召回违禁底盘动作，未推广默认。
- 温度口径修正：定时资源峰50.592°C，请求边界纳入后最高52.080°C。音频3465/70433帧overrun，不宣称全程实时；swap存在活动。独立审计及人工审核已写回结果目录。
- `results/native-rms-m32-r1` 六进程两层筛查完成：普通输入API P50 10.867→10.202ms（约6.1%）；hidden相对L2差异0.778%，大幅值压力输入均finite。history128、query32，无音频/RAG，不等于完整system/dynamic请求或全模TTFT；native gamma/epsilon表示改变，未认定无损。
- 当前 `down-row-scale-tail` controller PID51776 / finish51777，独立目录运行完整M8 tail对照，禁止重复启动或绕过双锁轮询板卡；仅读WSL状态。两组均原system，B/C仅动态M8敏感块不同，其他输入/量化/亲和/语音驻留一致。待回收后检查质量/真实tail时间及温度完整口径。
- 离线 `graphs/build_compact_row_down_screen.py` PID51887 已完成编译M64 plain/row与M256 row，history容量128保留全部有效历史；只为两层组合筛查，未板测、不是完整长动态请求支持。更晚位置必须保留其全部历史，不得以截断换速度。
- 顺序接续已运行：`continue_compact_row.py` PID53069，读WSL `continue-compact-row-status.json`。等待tail控制器退出且分析完成、archive_verified及4driver0，再以既有双锁启动compact-row-down-r1；完成后自动运行局部analyzer，失败不自动重跑。当前仅waiting_tail。
- compact-row候选已编译完成；`controller_compact_row_down.py` / `runtime/profile-analysis/analyze_compact_row_down.py` 与M64 L2紧凑fixture已准备，独立静态审计及dryrun已通过（board_operations=0，上传预算142895826B），未启动板测。native两层结果独立raw审计已通过，但不代表语义质量。
- 代码权威仍Windows F/WSL；新测试未作为生产模型，100ms完整目标保持active。当前已提交本任务阶段成果 `f0fab4bce5ba6a008794692e878d032f886475be`；其后新runner/fixture尚未提交。GitHub ls-remote 再次curl28超时，不能确认远端同步；原有push也不据此认定成功或失败。

## 2026-10-26 Next tail and native RMSNorm candidates PREPARED, not started

- Current prefixsafe controller49958 / finish50000 still live at latest check; do not repeat or poll occupied board separately.
- `down-row-scale-tail` dryrun and independent runner audit passed. Four phases original/protected-main/protected-main-tail/original repeat. Same binary0ed64448…, M32f0bebd8b… and M8candidate2732f92a…. Both candidates retain original system M32/M8; B/C only changes dynamic M8 sensitive block. 96perf/192natural/32diag/8prefix/8cache,248 original-prefix branches each protected phase. No launch or automatic continuation installed. Review current full result before launching.
- Native SDK RMSNormalize discovered in pinned image and compiled `graphs/native-rms-c512-m32-l0`:4 original max-scaled RMS subgraphs→4 native RMSNORM nodes, epsilon≈1e-6/gamma preserved, other projections unchanged by builder assertions. Compilation is not numeric/engine/speed proof. Builder50982 completed, status.json holds NB hash; independent static audit pending. `controller_native_rms.py` prepared, requires audit then dryrun: canonical M32/history128 and synthetic large-hidden maxabs4096, original/native/original each3warm+30. No board test yet.
- Keep complete100ms goal; original M8 tail124–127ms means tail work necessary, yet even zero tail leaves~296–1133ms TTFT at S64. Full main bulk also needs optimization. No input shortening, context truncation, dynamic-cache shortcut or promotion.

## 2026-10-26 C512 original-system / row-dynamic comparison RUNNING

- Latest local commit017b37df999adf32b87422a95fff6d35025089bb; GitHub push outcome unconfirmed (HTTPS push and subsequent Mac/WSL ls-remote timeout). Last independently verified remote5aee7ee. Re-read ref before retry, no force. New PREFILL_BREAKDOWN.md/json saved after commit and copied SHA-verified to Mac: current original M8 tail124–127ms; even hypothetical zero tail leaves TTFT296–1133ms at S64, so main bulk must also improve.

- `research/intern-prefill100-20261025/down-row-scale-prefixsafe`: controller49958, finish50000, run `down-row-scale-prefixsafe-r1`. Last observed controller status running and exact /proc command live. Do not restart or separately poll board while locks owned. Read WSL controller status/finish-status and actual PID identity.
- Same hardened binary0ed64448… across original/all-row/protected-system/original-repeat. Protected reset requires max_new1; continuation caps reuse at S64/128. Fixed system正文59/123+模板5; dynamic正文64/128/192/256+模板8=new72/136/200/264. Output fixed32 for performance; natural up to32. System-only cache, no exact-prefix/dynamic reuse.
- 96 performance,192 natural,32x4 diagnostic,8 prefix-only logits,4 repeated-input cache probes expected. Fresh hash original prefix M32/M8 graphs before protected load. Entire protected dynamic pipeline has3 submissions/block vs grouped1, cost included. Prefix logits equality is not full KV byte proof.
- Synthetic speech and RAG resident, frozen text model input, no visual/microphone/motors. Results not yet available; complete100ms objective unmet. No default promotion.
- Initial finish49959 raced missing initial state; preserved `.initial-failed` evidence, only WSL observer repaired/restarted50000. Board controller49958 never restarted. finish waits and analyzes only, no implicit rerun.

## 2026-10-26 C512/M32 row-normalized down full-model A/B/A COMPLETE, not promoted

- Active objective remains complete prefill AND model TTFT <=100ms at system64/128 and dynamic body64–256; unmet.
- `research/intern-prefill100-20261025/down-row-scale-full/results/down-row-scale-full-r1`: controller rc0, analysis and independent result audit complete. PID47730/47822 exited; do not rerun.
- Same Intern NPU text C512/M32 binary, true original M8 tail and decode. Only M32 fused layer2/3 has row-normalized I16 down. Candidate also computes system KV; not precision-equivalent. Original A/B/A output stable. Speech synthetic stream and RAG resident; model input frozen text, no image/microphone/motors.
- Input: system body59/123+template5=64/128; dynamic body64/128/192/256+8=new72/136/200/264. System-only reuse. Per variant24 fixed-feedback32 performance,48 natural regression,8x4 full-logit diagnostics. All72/144/24 raw records and36 archive files verified;3driver0.
- S64 D64/128/192/256 candidate TTFT P50=420.3/700.1/978.5/1260.6ms (baseline846.0/1581.7/2326.5/3048.1). S128 candidate421.4/700.2/980.5/1259.6ms. Prefill400–1239ms. Decode15.1–15.4tok/s. Each cell3 cases only; not long stability.
- Natural44/48 exact vs original; A/A48/48. Candidate logits relativeL2~2.87–3.22%,argmax31/32; A/A byte exact. Right-turn request lost→hold; left-arm reply missing limb;2 preexisting explicit conflicts remain. ROOT_QUALITY_REVIEW and independent manual review saved. No default promotion or Top3 physical execution.
- Whole-run max50.282C, mean8-coreCPU7.584%,PSS588.0MiB (listed processes only),MemAvailable min2027.5MiB. Swap932→1374.5MiB;swapin7278/out120684pages;audio2701/54669overruns,maxlag.794s. Not no-swap or all-real-time; do not attribute all swap to candidate. Observed resources14.31min, not15min stability qualification.
- Tail M8 candidate static-audited NB2732f92a… only, not board-tested or included in above. Old tmpfs run directory was removed after SHA-verifying49 files and29 symlinks; Windows archive retained,95,627,447 logical tmpfs bytes removed. Persistent original M32/M8 banks retained. See down-row-scale-full/cleanup-tmpfs.json; do not reuse removed overlay.
- Offline power2 diagnostic completed (PID49377 exited):528 captured rows,34 windows. Floor-power2 scale reduces simulated dynamic down relativeL2 L2 .05666%→.04364%,L3 .07584%→.06433%; no graph/kernel/timing or full-model evidence. ANALYSIS SHA444715bd…. Next private runtime down-row-scale-prefixsafe preserves original system routing, hardened max_new1 and strict system-only cache cap; compiled0ed64448…, runner4phase prepared but not yet started. No production change.
- Offline logical dense work accounting `results/dense-work-budget`:357826560MAC/new token,25.764–94.466GMAC for new72–264. This is not runtime latency bound or hardware peak claim.
- Mac full report and5 analysis/audit files SHA-verified in reports/intern-prefill100-20261025/down-row-scale-full-r1.
- GitHub verified5aee7eee3990c13be97d851f8f886f7beee42bd3 via Mac bare relay; newer result/analysis fixes still need commit and sync. Windows remains authoritative.


## 2026-10-26 row-scale板测完成，准备C512整模接轨

controller46617和finish46639均complete/rc0，勿重复启动。results/down-row-scale-original-input-replay-r1：104进程全部driver0、34原历史/2AA逐字节相同，16实际端口相同。两层API原120.4467ms→row18.3396ms（旧range16为17.4842）；hidden首token99.4167%→0.12055%，首块后续31.0200%→0.41985%，动态0.32080%→0.29781%。这不是整模TTFT或质量批准。

下一步graphs/build_down_row_scale_c512.py已启动，WSL PID46905，目录graphs/down-row-scale-c512-m32-l2；先读status.json和该PID。完成后需静态审计并接入历史较快C512/M32的完整对照，不能继续把C1280诊断配置与0.85秒历史混比。C512原整模冻结配置capacity512-full/c512-m16-m32-aba-r1-config.json，binary SHA bf19922f…，m32 graph与m8 tail在旧板端run目录，切勿原地覆盖。板上此刻无本任务推理进程，新编译在WSL，访问仍需双锁。

当前row-scale replay已启动：WSL controller PID46617、finish PID46639；仅读WSL down-row-scale-original-input-replay-r1-controller.json、finish-down-row-scale-status.json及日志，不另行访问占锁板卡。finish仅在complete/rc0后回收分析，不自动批准候选或报告达成。

## 2026-10-26 prefixsafe与A8已完成；逐row归一化待板测

完整≤100ms目标未达，且没有默认替换生产图。prefixsafe-i16-r2 complete/rc0：216主请求、6prefix、3driver0；whole-system原路径恢复两个system的prefix logits逐字节相等，动态仍有2.61%–3.17% logits误差。自然输出43/48一致不是准确率；三路都有两条“禁止点头却点头”，候选仍有缺失动作对象。速度为C1280/M16诊断组，不能替代历史C512最佳成绩。报告prefixsafe-i16/results/prefixsafe-i16-r2/REPORT.md，Mac已有副本。

compact-gateup-a8-m64-r1 complete/rc0：两层API原13.75ms→13.67/13.68ms，收益不足1%，hidden误差9.39%/4.72%，不推广。实际16端口全同、A/A字节一致、4driver0，所有温度观察点峰43.834°C（资源流采样峰43.214）。独立审计和Mac报告可用。续跑器45443已complete，不得重启旧任务。

新的离线真实down误差归因：activation量化更主要；固定通道缩放对另一开发prompt出现截断，放弃。无校准逐row max归一化候选graphs/down-row-scale-c1280-m16-l2已编译，NB SHA958ba4d609a51aaedd312ba9af43f1d3daad7f7672181d0f04da4ef448f47223；仅离线估算，未板测/未批准。下一步controller_down_row_scale_replay.py，34原NPU输入、104进程、原/range16/row-scale+两处原after；脚本dry-run通过，等静态审计完成再启动。分析器runtime/profile-analysis/analyze_down_row_scale_replay.py。模型仍全Transformer，局部算子/量化重表达，不是改CNN。

## 2026-10-26 当前续跑：prefixsafe全模与A8局部筛查

完整100ms目标未完成。prefixsafe-i16-r2控制器PID43926仍运行，必须只查WSL状态，不能另行轮询占锁板卡。固定64/128含模板5，动态正文64/128/192/256另加模板8，只复用system，输出1测首token；另有48自然样例与固定logits质量诊断。本轮C1280/M16无语音RAG，不是最佳C512运行时的最终速度验证。

新续跑器PID45443，task/continue_prefixsafe_a8.py：等待前任务complete+rc0+PID退出和离线分析成功后，才启动compact-gateup-a8-m64-r1。查看continue-prefixsafe-a8-status.json与对应controller/log，勿重复启动。A8编译与静态审计通过，未有板测速度/质量结论；它是M64/history128两层纯API筛查，不能称完整TTFT。权重保留，激活网格有损、零bias存储格式派生变化已记录。

GitHub c323710同步尚未核实，最近ls-remote15秒超时；不要报告已推送。

## 2026-10-26 原始NPU输入INT16定位与整system保护候选

- 目标仍为static64/128、dynamic正文64–256（另模板8）的完整新prefill和TTFT≤100ms，未达到。
- `research/intern-prefill100-20261025/results/down-capture-c1280-replay-r1/REPORT.md`：直接捕获原版NPU down输入/输出，34窗口/528有效行；异常集中同一system首token，诊断原5输出逐字节不变。
- `results/down-i16-original-input-replay-r1/REPORT.md`：104进程208计时；原始上游teacher-forced两层API120.443ms→17.476ms；range16 hidden误差首token99.4167%、首块其余31.0200%、后续system0.33847%、动态0.32080%。不能以局部误差替代全模型质量。全部释放driver0。
- 两组tmpfs重复副本SHA核对后分别清理303983051/333477996B；raw证据在WSL保留。
- `prefixsafe-i16/`：新私有runtime显式header7设置整system原精度阈值；只在reset完整prefix请求允许，拒绝缓存续用时更改边界；候选未晋升。已编译及4项协议测试通过，r1因TF空余951275520B低于1GiB预留，在上传和加载前终止；证据在attempt-r1-disk-preflight。r2改程序和输出全tmpfs，WSL controller PID43926；启动/结果以其controller.json、实际进程和events为准。不要重复启动。
- GitHub最后确认d5a3204；后续push超时结果未确认，不得声称已同步。

## 2026-10-26 板端down capture诊断已完成

- `research/intern-prefill100-20261025/results/down-capture-c512-m32-aba-r1/REPORT.md`：同一canonical history128/query32，原/额外两输出/原，90pure+9warm，原5端口逐位相同，3driver0。
- 实际两层内部down激活相对导入图FP32 relL2 7.87%/7.03%；原量化未改，不是候选质量退化。仅FP32输入历史，不是真实整模NPU KV。
- 诊断图/探针为独立版本，未改生产。完整100ms目标未达。下一步多输入/真实NPU上游capture与量化校准，不能凭单样本或离线通道名单启用混合精度。

## 2026-10-26 异常通道筛查更新（Windows权威）

- 新增 `research/intern-prefill100-20261025/graphs/down-outlier-screen/REPORT.md`。48个导入图窗口按6/6场景校准/留出；固定top128仍漏掉L3范围8的留出异常（局部down投影最大relL2约1.58%）。未部署。
- 补充24完整prompt source-Q8 FP32全位置捕获：system部分L2/L3最大约2013.63/1180.65，动态最大9.33/10.68。与导入图路径不同；只能作为分布证据，非真实NPU中间值或完整质量。
- 下一步先做原量化图的额外down激活输出诊断并核正常输出是否被扰动，再决定分通道精度。禁止把小样本固定名单、纯块耗时或FP32离线误差当整模达标。
- 两个compact实测及清理均完成，板卡控制器已退出。目标仍system64/128、动态正文64–256+8模板、新72–264，完整prefill/TTFT≤100ms；未达到。

## 2026-10-26 紧凑past128两轮已完成（100ms目标仍未完成）

新实验 compact-past128-m256-abcd-a-r1（150纯API）与 compact-past128-screen-aba-r1（270纯API）均complete/rc0，共14次NPU释放0。两controller已退出，无待重启任务。原始WSL results保留；两轮tmpfs逐文件留存核SHA后共清理392,209,436B。新probe18b29578…只允许128/512 past并增加11/5端口元素检查，旧probe未改。

普通L0/L1同有效history128：M64原34.56→compact13.91ms；M128原90.39/84.20→37.41ms（AA漂移6.85%，报告收益55.57%–58.61%，不能说频率稳定）；M256原210.69→83.43ms。紧凑M256 row224反慢至93.82ms，五输出与紧凑plain相同。L2/L3原FP16 down慢块M2561887.40→1760.63ms，仅降6.7%；去空槽解决不了主瓶颈。全部紧凑输出有数值变化，hidden相对原版0.042%～0.076%，未整模/自然质量合格。最高43.648°C、swap无增长，未长期全链路。

报告 results/compact-past128-m256-abcd-a-r1/REPORT.md 与 results/compact-past128-screen-aba-r1/REPORT.md，独立审计同目录。compact图只删全零且mask−10000的384空past槽，完整128有效KV、query、绝对RoPE与tokenIDs保留；capacity128只是past输入，不是整模cache。真实整模binary源应看capacity512-runtime/runtime（不能把旧runtime/快照当最新），prefill/decode目前同shape共享KV，专用past128需要明确gather和成本/形状路由。

下一重点FP16 down通道离群分布与保真提速，考虑少数outlier通道FP16+其余高精度整数路径的可行性，先读现有数据避免重做已否定路线。此方案尚未实现，不能称已提速。禁止直接恢复旧整体I16或旧Conv默认；需全logits/中文质量。完整100ms目标保持system64/128含5模板、body64/128/192/256+8动态模板、实际new72/136/200/264，only-system KV。单两层块API不是TTFT，flush/gather/head必须算完整延迟。

## 2026-10-26 M256 softmax 定位与分行候选（目标未完成）

权威任务 research/intern-prefill100-20261025。profile-m256-l0-r1 与 softmax-row224-m256-aba-r1 均 complete/rc0，当前这两个controller已退出，无待重启任务。原始结果保留results对应目录。M256 softmax NN busy计数相对M32增104.287倍（8倍查询行、1.412倍key宽度，按元素9.234倍）；I/O显式同步仅0.046ms。仅为NN计数，不是wall占比。

新图不改Transformer架构，只将两个softmax的query/group行分8×224，完整768key轴保留，原权重/量化payload不变。90次纯API A/B/A P50 210.421→115.811→210.398ms（降低44.96%），3次driver释放0。候选hidden relL2差0.043864%，非无损；没有整模/自然质量/不同history覆盖，保留候选不替换默认。M256物理256查询不等于正文256+模板8。报告 runtime/profile-analysis/REPORT_M256.md、results/softmax-row224-m256-aba-r1/REPORT.md；独立审计同目录。

下一步优先核对空past槽mask，评估固定system<=128时仅移除无效槽、保留全部有效历史的专用prefill图；需配对验证绝对RoPE/键顺序和后续块routing。100ms完整目标仍是system64/128 + body64/128/192/256 +8动态模板，实际新增72/136/200/264。不得用两层块115.8ms宣布完整TTFT达标；不得截断有效上下文换速度。WSL keeper PID539原sleep21600需在后续长任务前确认余量。

## M32 trace定位已补齐

`research/intern-prefill100-20261025/runtime/profile-analysis/REPORT_M32.md`：M32 L0/L2 trace重导出的NB与实测件逐位相同，旧NPD SHA/源config/解析manifest核对后已映射。L0已记录NN计数FC44.52%、add14.32%、softmax10.66%；这不是wall-time占比。原M16 softmax30%不适用于M32。下一测量建议M256 L0单独NPD，查明其每行API比M32更慢的算子来源；尚未采集/未启动该实验。没有板端后台任务。

## 2026-10-26 output-FP16 tmpfs单块筛选已完成

`research/intern-prefill100-20261025/results/down-outputfp16-c512-m32-tmpfs-r1/REPORT.md`：rc0，4×30纯API、12warm，实际4份f32哈希/finite及5端口、4release0独立审计通过。原FP16 221.865→221.810ms；旧I16 14.125ms，新I16计算/FP16输出14.101ms，额外收益仅0.172%，hidden相对L2略升(.303886%→.309506%)，不扩整模、不晋升。板端scratch改tmpfs但保留1GiB+100MiB余量且额外MemAvailable守卫；所有证据已回F。精确SHA清理94,721,054B临时复制品，原模型保留，无后台板端任务。下一方向是普通整数块FC/softmax与输入量化误差，M32 trace需先与实际NBG一致。100ms全范围目标未达成。

## 2026-10-26 C512/M32 FP16 Conv整模结果已回收

`research/intern-prefill100-20261025/capacity512-conv-full/results/c512-m32-conv-aba-r1/REPORT.md`：控制器rc0，72性能/144自然/24诊断，3close driver0。相同static64/128（正文59/123+5模板）、dynamic正文64/128/192/256+8模板，新72/136/200/264，输出32固定反馈。TTFT P50动态64为.5678/.5683秒，128为.9940/.9978秒，256为1.8867/1.9198秒；配对比值中位数.64135，decode约15。仍远未100ms，44/48自然token同且有部件/约束丢失，拒绝默认晋升。原图A/A全同，候选logits relL2 2.696–3.229%。源图/候选/原始证据均保留；下一outputFP16 probe未启动，磁盘951275520B不足原预留与上传要求，不能降低阈值强跑。root质量判定已入QUALIFICATION_DECISION.md；独立审计已完成：Top1冲突19→21/48，两版Top3全部48例至少一冲突，禁止直接执行。动态128的P95/max为1.026/1.011秒，中位数低于1秒不代表稳定达标。

## 2026-10-26 独立结果审计与输出FP16四路单块准备

C512 M16/M32父轮独立审计完成：INDEPENDENT_RESULT_AUDIT.md/json、NUMERIC、MANUAL均在capacity512-full/results/c512-m16-m32-aba-r1。计时与文件完整性通过，质量/100ms均不合格。M32 TTFT配对降14.1%，decode基本不变；M32自然旧24完整9例、新24完整4例，另有兼容但遗漏、安全不响应、信息不足、复述和冲突，不能用token43/48当质量率。RAG Top1明确冲突19/48，Top3在48/48情景中都有冲突项，不能依次执行。当前Conv run36947/watch36977仍运行，只看WSL状态，不并跑新板测。

新 `controller_down_outputfp16.py` / `analyze_down_outputfp16.py` / `OUTPUTFP16_PROBE_PLAN.md` 已准备，host dryrun生成down-outputfp16-c512-m32-r1-config.json，**尚未启动**。四路为原FP16→旧range16 I16→I16输出FP16→原FP16；相同C512/M32两层、canonical输入、3warm+30API，五端口数值分别比较。execute先要求Conv父轮完整结束/回收，不会排队偷跑。保留原1GiB+100MiB磁盘预留；预计传95.7MB，当前空间预计不足，须父轮后按Windows SHA备份清理旧副本，不能降低门槛。

根节点已修两项source audit防漂移问题：candidate NB/bytes硬绑定独立审计；旧range16的geometry/sourceNB/actual_ports/input/ref/known NB hash及audit SHA固定；所有上传source在拿锁前冻结，并核对cfg源码SHA，拿锁后再次校验。独立审计仍在复查这些controller/analyzer，勿声称该单块已测。

最新GitHub同步：代码与原始证据提交567f34d已推送，独立ls-remote匹配；以下旧网络超时仅为历史。后续资源报告补充/独立审计仍可能是工作区更新。

## 2026-10-26 C512 M16/M32完整结果与新Conv对照

父轮 `capacity512-full/results/c512-m16-m32-aba-r1` 已complete/rc0，72性能、144自然、24四步实际logits，3close driver0；PID34040/34264已结束。M32配对TTFT中位较M16降低14.10%。静态64/128＝正文59/123＋模板5；动态正文64/128/192/256＋模板8，新72/136/200/264；仅system KV、固定反馈输出32。M32 TTFT P50：S64为842.9/1557.3/2309.0/3047.1ms；S128为853.8/1606.4/2354.0/3095.8ms；decode约15.1tok/s。每格仅3成本案，不是长期分位数资格。

自然43/48序列同基线，M16重复48/48；logits relative L22.93–3.29%，非无损。根节点已逐案记录ROOT_QUALITY_REVIEW.json/QUALIFICATION_DECISION.md；两版仍有摇头→点头、缺方向/幅度、场景复述。M16“摆弄桌子”在M32变成右臂挥动一次，但不能称整个质量合格。最高49.786°C，全8核加权CPU7.392%，PSS峰618.91MiB；swap319→829MiB/换出130560页。音频68916帧、3393超帧预算，最大lag1.047s，不是严格实时。独立结果审计进行中。

**新活跃run** `capacity512-conv-full/c512-m32-conv-aba-r1`，WSL控制器PID36947，watcher PID36977 (`finish_conv.py`)。三版均M32、原图→原精度FP16Conv-L2→原图；只主图两处down改写，真实M8尾段仍原图。已审source/candidate固定，勿覆盖。只读WSL controller/log/finish状态；占锁时不要另外访问板卡。watcher只等特定PID并在回收后分析，不自动晋升或启动其他测试。当前没有此轮速度结果。

启动前free993349632B未满足预留；按已冻结计划逐SHA删除24个已完整保存在Windows的旧fulli16-range16 logits副本58,242,816B，free1051643904B。清单retained-diagnostics/deleted-1792956618.json。未删父轮48图/生产模型/当前结果。新upload35,607,575B；磁盘阈值未降低。

离线输出FP16新量化候选已由独立静态审计通过，仅允许后续有界数值/速度实验；没有板测，保留pos0输入饱和风险。100ms目标尚未达到，不能缩小正文或偷偷复用动态KV。

## 2026-10-26 输出FP16的独立量化候选准备

当前C512原精度完整对照 `capacity512-full/c512-m16-m32-aba-r1` 仍由WSL PID34040运行，PID34264等待结束自动分析；本轮已确认进程活跃，未重启或直接轮询板卡。继续先回收/审阅此轮，再决定是否启动已审 `capacity512-conv-full` 原精度候选。

新增离线候选 `graphs/candidates-down-i16-outputfp16-explicit`：只把旧range16的两处down输出改回FP16，显式输入转换/I16权重网格保留。4个最小图与真实两层导出均rc0，实际NB端口查询全部通过；两层NB SHA a10a3aa5…，29,747,392字节。没有板测/速度/任务质量结论；仍有pos0输入饱和风险，禁止直接默认部署。当前不排自动板测，详见候选README与独立审计。编译与查询都持WSL SDK锁，未触板卡运行版本。

GitHub此次独立ls-remote20秒和push40秒再次观测超时；只确认远端0431164，f06c5753及后续新提交是否同步尚待核实。不要把超时当远端未收到，也不要宣称已同步。

## 2026-10-26 原精度 C512 批处理对照准备与 I16 资格结论

版本补充：新候选准备和证据分析watcher已提交本地f06c5753；GitHub此前0431164已独立确认，本次f06推送40秒及复核20秒均观测超时，待重新核实，不能宣称已同步f06。

跟进：WSL PID34040为唯一c512-m16-m32-aba-r1控制器，约1.16GB上传及哈希/overlay验证已完成，controller状态running，已启动板端完整测量；以本地controller.json与活PID为准。PID34264运行finish_capacity512.py，只观察该控制器/proc身份，结束后对回收结果自动执行analyze_fulli16.py，绝不访问板卡或重启。仅读WSL status/log；原始结果目录未生成前不可宣称性能。

预备下一隔离分支 capacity512-conv-full：M32原图→FP16Conv-L2→原图，尾M8仍原图，14权重与量化保留；复用已实测候选9b80e75c…，host preflight通过，源码独立审计通过（controller c04f5c…，纯空白修订已独立复核），未启动。execute须父run完整结束，需先读当前质量/数值结果；没有自动续跑。预计上传35.6MB；仍保留768MiB+150MiB磁盘阈值，空间不够时只可逐SHA清理已回收旧诊断，严禁删父run引用的48图。100ms目标未达到。


完整性审计通过，候选不晋升。72性能、144自然输出、24份四步完整logits，控制器rc0，3次driver释放为0。实际结果和人工逐例判定见 REPORT.md 与 INDEPENDENT_RESULT_AUDIT.md。

系统固定64/128 token＝正文59/123＋模板5；动态正文64/128/192/256＋模板8＝新增prefill72/136/200/264。每次仅复用systemKV。性能输出32固定反馈token，质量为自然生成上限32；预热/固定prefix准备不计热TTFT。C1280、M16，尾8仍pad16，不是真实M8。

候选TTFT P50 1.039–3.684s，逐例配对降低约32.65%，decode约14.6–15 tok/s。距离完整100ms目标仍远。32诊断argmax全部相同，但完整logits relative L2 2.75–3.03%；自然输出43/48完全相同不代表质量合格。新例 s128-d192-arm-low-wave 从“右臂挥动一次”变为“摆弄桌子”，引入未授权物体操作；因此不得将候选设为默认。

语音前端/RAG常驻、合成WAV流式输入；LLM冻结输入不由ASR触发。无视觉、真实麦克风或电机。最高50.468°C、全8核CPU加权7.35%、所采进程PSS峰624.66MiB、MemAvailable最低2667MiB。swap129.25→245.5MiB、swap-out30992页；不能据此单独诊断泄漏。92546音频帧，4.92%overrun、累计最大lag0.851s，不能称严格逐帧实时。全部48场景Top3含至少一个冲突动作，不得并行/顺序执行Top3。

下一隔离试验 capacity512-full 保留原权重/量化，C512 M16→M32→M16重复，统一真实M8尾段；不启用失败I16替换。C1280数据仅作历史对照，不能将容量和batch收益混为单变量证据。C512足够覆盖最高128+256+8+31=423位置，未缩减用户输入范围。最终目标仍是整个动态prefill与TTFT低于100ms，尚未达到。

板端清理完成：三个旧Laya/Nano派生ONNX在Windows保留并SHA一致后删除，共1,161,213,577B；空闲2,220,736,512B。清单 research/intern-prefill100-20261025/retained-diagnostics/derived-jev-cleanup.json。新C512源码审计通过，唯一run c512-m16-m32-aba-r1已由WSL控制器PID34040启动，launch.json固定源码SHA。只读WSL控制器状态/日志，不绕过双锁轮询板卡；无新性能结论。GitHub已推送并独立ls-remote核实0431164f067473ebd76f87b5ed5a06df720814eb。本句及sync元数据是后续工作树记录。

## 2026-10-26 首段量化修复与C512/M32准备

权威路径 research/intern-prefill100-20261025。GitHub分支codex/wsl-intern1b-reproduction已独立核对b2141f27d723329a8c91ed4506552053d4fa08ee；之后新增分析器与C512准备见本次提交。

当前板测：fulli16-suffixfix/suffixfixed-c1280-aba-r1，控制器PID28142，最近核对完整cmdline仍活跃、status running。仅读WSL status/log，禁止重复启动或绕锁板端访问。结束后运行本目录analyze_fulli16.py；独立审计智能体正在等待回收并按旧/新场景分别评审。完整100ms目标仍未达到。

C512原图补齐2张、M32 13张已编译并通过实际接口核对；独立C512/M32/M8 runtime主机缓存/ASan测试通过，均尚未部署。细节及后续门控见CAPACITY512_NEXT.md。不要启动明确禁用的fulli16-firstfix目录（旧SDK分组失败方案）。

## 2026-10-26：修复后完整矩阵已启动

当前唯一板端控制器：`research/intern-prefill100-20261025/fulli16-suffixfix/controller_fulli16.py`，WSL PID 28142，run `suffixfixed-c1280-aba-r1`。先前suffixgroup诊断rc0、8份hybrid完整logits字节一致、3driver0，独立完整矩阵源码审查无阻断后启动。只读WSL该run-controller.json/log并核实PID，不重复启动、不绕过锁直查板卡。

三版原group／首块保留原L2＋I16固定suffixgroup／原group重复；system64/128（正文59/123＋5），动态正文64/128/192/256＋模板8，新72/136/200/264；性能输出32。72perf、144natural（各旧24＋新冻结24）、24diag×4全词表；语音回放与RAG常驻，纯文本NPU，无视觉/真实麦克风/电机。结果未完成，100ms目标未达成。旧失败fulli16-firstfix保持禁止启动。

## 2026-10-26：固定suffix分组诊断通过，整模矩阵准备完成

`suffixgroup-firstfix` 已rc0，3driver0；修复suffixgroup与修复nongroup 8份完整logits同字节，4自然同原图，残余固定feedback logits relL2约2.9%。独立审计通过。只证明4条开发样例恢复，不作完整准确率或热延迟声明。

下一实验 `fulli16-suffixfix/controller_fulli16.py suffixfixed-c1280-aba-r1 --execute` 尚需确认独立源审后启动；72perf输出32、144natural(旧24+新24每组)、24diag×4；system64/128＋动态正文64/128/192/256＋动态模板8，同C1280 A-B-A，语音/RAG常驻。额外原L2 SHA明确检查，候选81次首块/固定suffix标记断言。旧 `fulli16-firstfix` 已代码raise禁止，勿混淆目录。

## 2026-10-26：首块修复已因果验证；固定分组验证进行中

- `first-chunk-fp16-diagnostic` rc0，4定位案例固定logits relL2从46–53%降到约2.9%，16/16argmax、4/4自然回复恢复原图；后退方向/距离恢复，但原基线仍省略“缓慢”，不是全面质量合格。见REPORT与独立结果审计。
- `first-chunk-grouped` detach/rebuild尝试rc1：SDK -9 表示网络销毁group后仍不能独立执行。失败已保留。`fulli16-firstfix/controller_fulli16.py`显式raise，禁止启动准备好的大矩阵。
- 当前板端唯一控制器 `suffixgroup-firstfix/controller.py` WSL PID20417，run suffixgroup-firstfix-r1。永久分组：block0、block1单独执行，suffix10固定group。只读WSL controller.json/log，确认PID/终态后再行动；未确认前不启动大矩阵、不绕过锁轮询板卡。
- C512 36图已完成实际端口审计（port-audit.json），权重/量化不变，尚未板端部署；2GiBquery137失败证据保留，8GiB全部通过。
- 目标静态64/128、动态正文64–256、完整prefill和TTFT<=100ms仍未达成。准备的大矩阵含旧24+新冻结24质量案例，需替换已拒绝runtime并重新预检/源码审计，不能直接运行。

## 2026-10-26：I16整模结果与首chunk异常值定位（进行中）

- `research/intern-prefill100-20261025/fulli16-range16/results/fulli16-range16-c1280-aba-r1/` 已rc0回收，72性能/72自然/24诊断，3次driver0；同C1280首token约降33%，decode仍约14.6tokens/s，尚未达100ms。
- 输入固定64/128（正文59/123+5）；动态正文64/128/192/256+模板8，新prefill72/136/200/264；性能输出32固定feedback。候选不能晋升：完整logits漂移大，2条取消前进后的后退场景被RAG召回为前进。
- 已纠正尾处理口径：当前72性能请求全部scalar_prompt_tokens=0，使用padded16处理尾8；true8运行时目前仅支持capacity256。不要按“省8次scalar decode”估计收益。
- `actual-npu-range-diagnostic` 已完成8worker/136IO捕获，4组control/capture logits字节同，8driver0。首system chunk在相同11输入下hidden差99.39%；真实输入上的FP32解释器估计down首行峰值远超I16范围，其余64窗口无这四项越界。见REPORT及独立审计，内部值不是NPU直接测量。
- `first-chunk-fp16-diagnostic/controller.py` WSL PID14259正在因果诊断：只在pos0保留原L2，后续I16；不作性能结论，不绕过锁轮询板卡。只读controller.json/log。检查真实PID及结果后继续，不重复启动。
- `capacity512-range16` 36个图已编译；独立SDK实际端口审计PID14551运行。2GiB容器query退出137保留证据，8GiB重试；无板端部署/质量合格声明。

## 2026-10-26 正在运行：完整模型I16 range16 A-B-A

`research/intern-prefill100-20261025/fulli16-range16/controller_fulli16.py` WSL PID8417，run `fulli16-range16-c1280-aba-r1`，双锁+3600s deadline。只读WSL controller JSON/log；勿板端并发轮询、勿重启。WSL keeper539。原图→修复I16→原图，72固定反馈32输出性能、72旧集自然输出、24×4全词表诊断。相同静态64/128、动态正文64/128/192/256另8模板。合成语音/RAG常驻，但ASR未触发冻结输入。源审计无阻断。等待rc0回收后用analyze_fulli16.py，并由审计智能体独立复核。未有整模I16速度结论。

独立新12家庭场景/24提示已冻结`independent-household-v2/semantic-inputs.json`（b35122a9a4db5b013ac23c1395cd96d1a5397f543f97cf3e6e663b5a314bfc50），明确完整背景句拼接以对齐token，不是真人语料、不声称训练集隔离。新场景尚未跑模型、禁止先拿来调参。

M8同量化尾块候选已编译并实际端口核验，`graphs/candidates-down-i16-c1280-m8-range16/integration-manifest.json`，NB fae2ea7300300b86bc377b8a81a83d94d30a25bedfd9f653820472f2aea1fa3e。完整payload与M16 range16相同，后续单独评测true8，勿将编译成功说成速度收益。

## 2026-10-26 当前检查点：I16 down数值尾部定位

权威报告 research/intern-prefill100-20261025/DOWN_I16_NUMERICAL_REPORT.md。A/B/A单块222.025/14.111/222.068ms已独立审计；48窗口原版/候选96运行已完成，hidden最大2.9111%而非仅canonical0.300%。FP32范围诊断4窗口b1输入超±8；新range16只将b1输入fl12→11，全部payload相同、16外部端口同。range16复测rc0完成，48窗口最大hidden relative L2降至0.3468%，旧新原图48份逐位相同；当前无活跃板端控制器。WSL keeper539保活。已用样例属于调参回归，C1280/M16修复范围集成候选现已编译并核验16实际端口与原图一致，manifest在graphs/candidates-down-i16-c1280-range16。下一步以独立overlay部署做完整模型对照、另冻结独立新场景；旧14.1ms不能冒充新范围版本正式速度。

C1280旧范围I16候选虽已编译，但未部署且应先修范围。目标system正文59/123+模板5=64/128，动态正文64/128/192/256+模板8，新72/136/200/264；单块query32不可冒充完整新增prefill，100ms目标未达。旧最快短输入C256参照继续保留。下方记录为历史检查点。

## I16 down最新进展：显式转换候选已导出

见research/intern-prefill100-20261025/DOWN_I16_EXPLICIT_REPORT.md及graphs/candidates-down-i16-explicit/deployment-manifest.json。完整两层NB已生成并通过实际端口/权重静态审计，但0板测；下一步L2独立精度候选A-B-A，不可使用要求14权重全等的旧controller。无活跃板任务，WSL keeper539。

## 新增：敏感down I16量化候选诊断

见 `research/intern-prefill100-20261025/DOWN_I16_EXPORT_REPORT.md`。三份独立候选均在SDK Verify -3失败，无NB、无板测；单轨迹校准仅初筛。下一步隔离最小算子/核对fraction约束，不能宣称已提速。WSL keeper539（SSH session1296）仍用于保活，未启动板卡任务。

## 2026-10-25 最新检查点：整数Conv的M32/M256均无收益，不扩整模型

- 100ms完整prefill/模型TTFT目标仍未达。固定口径：system正文59/123＋模板5＝64/128；动态正文64/128/192/256＋模板8，实际新prefill72/136/200/264，只复用system。TTFT不含语音、压缩、排队或system准备。总览 `research/intern-prefill100-20261025/OPTIMIZATION_PROGRESS_20261025.md`。
- 原最快短输入参照继续保留：C256＋true8，S64/D64与S128/D64 TTFT P50 0.869842/0.8904535s，输出32固定反馈；C256是图容量，不是system长度。C1280单项FP16 L2 Conv的1.230/1.271s不是更快合并版，不把单项21%收益叠加或外推到C256。
- `results/bitcast-c512-m32-r2` 已完成rc0并独立审计：C512/M32/L0/1单块，3warm＋30times，A/B/A API P50 10.8287535/13.5229945/10.8320665ms；候选耗时增加24.880%，原前后漂移+0.031%。五输出45,056个f32逐字节同，3release0，峰值采样40.176°C。该M32配置不扩栈。
- `results/w16conv-c512-m32-r2` 已完成rc0并独立审计：同契约A/B/A P50 10.8661680/13.1410940/10.8389805ms；候选耗时增加20.936%，原前后漂移−0.250%。五输出逐字节同，3release0，峰值采样40.548°C。W16是原I8码符号扩展到I16存储、保持数值/scale；driver 45,351,216 B，较原增加17,996,544 B，不是增精度量化。此M32配置也不扩栈。
- 上述两组仅一个canonical输入、历史128位置＋物理query32行的两层API，排除加载/转换/宿主IO，没有语音/RAG常驻；不能替代完整模型TTFT、中文任务质量或长稳验收。1×1 Conv只是Linear `y=Wx+b`的另一种SDK实现，不跨token、不改Transformer/attention/KV；数学等价不自动保证低精度bitwise或提速。
- 导出问题的证据更新：显式GenerateNBG(NULL)在原图和BITCAST都崩溃；SDK官方main＋真实11tensor输入＋SAVE_NETWORK_BINARY流程可复现原NB SHA并导出候选。旧失败保留，不能把NULL-query崩溃归为BITCAST本身失败，也未单独确定崩溃根因。
- `results/bitcast-c512-m256-r2` 已完成rc0，主分析已生成，独立审计已完成：C512/M256、真实层0/1两层单块，同canonical历史128位置＋256物理query行，3warm＋30times；A/B/A API P50 210.7638185/213.5066215/210.8479020ms，候选耗时增加1.301%，前后基线漂移+0.040%。五输出共360,448个f32逐字节同，3release0，111点最高41.974°C。M256不是动态正文token数，也不是整模型prefill/TTFT；没有语音/RAG并发。NB清单仍为 `runtime/official-export-c512-m256-l0/deployment-manifest.json`，SHA `907689cdee9a1f767f4b64d8854bcbae27b5c16871f26500519ed977fee4c250`。
- M256 r1在上传阶段因WSL保活失效中断，0有效样本，失败证据保留；r2使用独立SSH保活。13,238,272 B canonical输入放tmpfs，A/B/A相同输入均在计时前预载，SHA核验后收尾清除；不要误称r1跑出失败性能，也不要推断所有板端NB/日志已清理。
- 普通整数Conv当前M32和M256都无速度收益，停止扩整模型、不替换默认。下一步重点是回到历史C256＋true8最快配置评估合并，或单独建立量化校准新路线；必须同输入、同缓存、同质量门槛重新验证，不承诺100ms。后续板测仍仅root持hostlock＋板flock，编译统一 `/tmp/opi-prefill-compile.lock`。
- fullconv-long-quality-r1此前已完成：两版严格完整表达9/24，22/24自然token相同，成本case挥手→举起跨margin，RAG Top3仍有冲突；M32单块bitwise不能推翻这些质量限制。旧fullconv overlay上传副本已按SHA清理，若再跑须重新上传；Windows源和证据保留。
- Windows为权威仓库，Mac仅控制/报告副本。本地HEAD `57a45e075880d4182b0bf3c455bedae5d344f980` 因网络错误尚未成功推送；最近已核实GitHub远端 `d35fd319da7a636f41de281ff8dd6a88a5d652e8`。新实验文件与本次文档仍需后续提交/同步；本次文档更新没有提交、推送或板端操作。

## 2026-10-25 当前：长语义质量核验运行中，短窗口最快结果保留

- 用户指明旧表0.87～0.89秒。重新审计raw确认：C256 combined-tail8，system总64/128（正文59/123+模板5），动态正文64+模板8，new72，TTFT P50 .869842/.8904535s，6次各格，输出32固定反馈。S128/D64最大.915570，不能说每次<.9s。
- 最新C1280单项L2 Conv在相同两格1.229562/1.271436s，不是短窗口最高性能新版。输入没有增加，改变的是图容量/尾批路径；21%只相对同C1280原图。原最快配置继续保留，后续必须同时列容量、token与缓存，禁止将隔离候选称“最新最快”。ORIGINAL_COMPARISON.md已纠正解释。
- fullconv-quality/controller_quality.py fullconv-long-quality-r1 --execute --timeout1200，WSL PID11625已结束，controller complete/rc0，48 quality+2 quality_cost+2 diagnostic+2 closed+1 complete已回收，正在独立质量审计。只读其controller JSON/log；root唯一板端控制，双锁，禁止重复启动或修改冻结source。24长自然×A/B=48+2成本自然+2份4步logits；依赖上一轮fullconv board overlay及候选NB，不得删除。
- DFP-I16激活+DFP-I8权重直接Conv被拒；等网格affine候选也在DATACONVERT node16失败，无可部署NB。AFFINE_QUANTIZATION_AUDIT.md/JSON给出静态网格/权重证明，但不等于端侧数值/速度通过。第三路线仍在只读可行性研究。
- GitHub已verified d35fd319da7a636f41de281ff8dd6a88a5d652e8，包含290a512结果与原最快配置口径修正。当前100ms目标未达，quality已回收待审计，不提前宣称可替换默认。

## 2026-10-25 Fullconv A-B-A已结束，下一步长语义/首token翻转诊断

- `fullconv-c1280-aba-r1` PID7665已退出、rc0，72perf/36自然/24diagnostic完整回收并重算f32SHA，3次driver0。`fullconv/results/fullconv-c1280-aba-r1/ANALYSIS.md`与analysis-fullconv.json已生成。
- 同C1280、同input/cache/原binary，只L2 FP16 Conv：逐case TTFT中位1.266倍快（延迟约-21%）。静态64/128、动态正文64/128/192/256+8模板、新72/136/200/264、固定32输出。D64 TTFT1.230/1.271秒，D2564.258/4.269秒；不是C256短窗口速度，100ms目标未达。
- A前后8份logits逐字节相同；B的8份全logits relL2约2.67%～2.95%，32/32诊断argmax同。但成本case `s64-d64-2` 首预测117057→114209（其余23固定32序列同），它未在原8诊断里；必须追加此case诊断和自然输出，不能宣称无损。12短自然回复全同、任务仍9/12；RAG Top3冲突仍在，mock未执行。
- `semantic-long/inputs.json`已冻结独立12情景×2system=24自然长case，动态精确64/128/192/256，无padding/截断；gold rubric分任务完成/保守未完成/冲突。SHA07ae4c67e0816ba0782d595a48edd8238c4812b7798d13a4d2ccab80311a483f。尚未跑。runtimeagent准备独立长语义+翻转case诊断控制器，root后续审并启动，不重复全A-B-A。
- 普通W8层14FC→Conv保持原DFP激活/权重的编译被SDK拒绝，失败证据保留；graphagent准备一个等整数网格的DFP→affine(scale2^-fl,zp0)元数据候选（仍I16/I8），未证明输出等价，不改现役/不自动扩栈。auditagent独立核对该metadata假设。板卡当前无已知任务占锁，接下来root单独管理。
- 已推送GitHub verified fd47d11b150fbd0e30e08cda74433d3c0e0e2235；上述新增结果与分析后续需再提交。Windows权威，Mac仅报告副本。

## 2026-10-25 当前运行：C1280 Conv 单块替换完整A-B-A

- 用户目标确认：短固定system总64～128、动态正文64～256，完整动态prefill与TTFT均≤100ms。未达标，不能降低长度或混入动态KV命中。
- root已启动 `fullconv/controller_fullconv.py fullconv-c1280-aba-r1 --execute --timeout 3600`，WSL PID7665，状态running。只读WSL `fullconv/fullconv-c1280-aba-r1-controller.json` 与log，不另起任务或直接轮询占锁板。双锁由controller持有。
- 原C1280→仅prefill-layer2 Conv overlay→原C1280重复，共72固定32步性能、36自然、24×4全logits诊断。全部同fixture/flags/原binary ff758...；新图dc772225e8c2b9ec2f6376132ba12aa9938dbe966b678e2bd7672832299fcfa6。fixed prefix每次独立重建，cache不跨variant。静态64/128（正文59/123+模板5），动态64/128/192/256+8模板、新72/136/200/264，固定输出32。每variant每格3样本；P95不能作为生产保证。候选影响system准备数值，需完整logits和自然质量验收。
- `fp16conv-m32-r1` 已完成：单块原221.992→Conv86.625→原221.982ms，2.563x；hidden relL2 .0007863（非bitwise），三个release0。M16候选也完成。全部只是C512两层probe。
- 为保留运行空间，按48个实际SHA核对后仅删除本任务板端probe NB/input上传副本及已回收f32，共471,333,088B；Windows源及全部raw证据仍保留，板端日志/生产权重保留。manifest `cleanup-board-probe-copies-20261025.json`；free 789,561,344→1,260,937,216B。旧single-block/M16/M32控制器如再跑须重传其板端文件，不能假设路径仍存在。
- runtimeagent只新增fullconv离线分析，图agent仅准备独立普通W8块FC→Conv单图，auditagent仅准备长中文holdout；均不得触板/改运行文件。先等待fullconv结果，回收后核对returncode/counts/driver0/质量，再决定下一步。

## 2026-10-25 Prefill长输入基线与FP16 Conv单块实测

- 权威任务 `research/intern-prefill100-20261025`；目标仍为动态正文64～256、完整动态prefill和模型TTFT均≤100ms，未实现。不可把单块计时、少输入、上轮动态KV或C256小窗口冒充达标。
- GitHub远程已通过ls-remote核对 `0f09467c55f32c4a0352d8087d95ba1ffb18fc6f`，分支 `codex/wsl-intern1b-reproduction`；以下新结果/脚本需后续提交，不意味着已包含在该提交。
- `fullbaseline-c1280-r1` PID4991已退出、rc0，48性能/12自然/8诊断完整；系统64/128含5模板，动态64/128/192/256另8模板、新72/136/200/264、输出固定32。TTFT各长度约1.6/2.9/4.1/5.4s，decode14.65～14.69。driver0。报告 `fullbaseline/results/fullbaseline-c1280-r1/REPORT.md`，含swap-out46.13MiB、音频启动最大lag626ms和质量9/12边界。
- `fp16conv-m16-r1` PID6598已退出、rc0；C512/M16两层块A-B-A:112.345→45.894→112.362ms（2.448x），hidden relL2 .0007477，非bitwise。保持原FP16权重/量化，3次release0。报告 `results/fp16conv-m16-r1/REPORT.md`。单块没语音/RAG，不是全模型TTFT。
- 全12个C1280 prefill图只有L2内部真实层2/3的两处down FP16，其余22层down I16/I8/I16。图agent准备仅替换C1280 prefill-layer2.nb的Conv候选，runtimeagent准备同C1280完整A-B-A控制器；尚未部署完整候选，decoder不改。节点trace缺engine字段，FP16 down没有NN计数不等于已证实CPU fallback。
- 板卡只能通过Windows/WSL并持hostlock+板flock，root独占板；agent只Windows编译/分析。当前已完成两任务，新的完整候选须先审配置并唯一命名启动。最新比较文件 `ORIGINAL_COMPARISON.md` 已补长输入表。

## 2026-10-25：100ms prefill新目标与原始数据对比

任务目录 research/intern-prefill100-20261025。目标仍未完成：system固定64/128（正文59/123+模板5），动态正文64–256+模板8，只缓存system，要求完整prefill及TTFT均≤100ms。完整主格/边界与限制见PLAN.md/AUDIT.md。

用户要求原始对比：ORIGINAL_COMPARISON.md 已重新读取最早 plain 与最新 combined-r2 的 raw events，systems/performance 输入逐项一致。旧prefill需要把scalar_prompt加回再取P50。system64/128 ×正文32/64：原TTFT 1.153/1.760/1.180/1.798秒，最新 .494/.870/.495/.890秒；原每格9次输出1，最新6次固定反馈输出32。跨轮历史对照，不能全部归因单项优化；decode只能使用同32步基线。

C512 M16/32/64/128/256 × L0/L2 共10块已编译，原权重/量化/导出payload一致。各M独立FP32前向会引入小输入差，因此保留旧fixtures，新增fixtures-canonical从每layer唯一M256历史切片，110字段逐位核验，reference重新计算。runtime/已交付协议v2，独立M16 system bank、显式prefix长度、bulk和padding可选；host测试通过，不等于全模型NPU验证。

single-block-c512-r1 已结束、rc0，数据回收 results/single-block-c512-r1，12组各3warm+30timed，全部正常released0。分析API P50：L0 M16/32/64/128/256为8.42/10.92/34.62/84.21/211.13ms；L2为112.21/222.12/456.72/922.93/1890.17ms；M16前后漂移约0.17%/0.13%。这是两层单块重复输入API，无语音/RAG背景，不是完整prefill或TTFT。大batch并未解决敏感L2瓶颈，下一步独立NPD/IO诊断和L2量化/算子对照；尚未编译完整C512栈，勿宣称100ms可达或全范围已测。

控制器已释放hostlock；后续board动作仍仅root经WSL hostlock和board flock。WSL keepalive sleep21600 PID624（需验证存活而非沿用旧PID）。板端本轮新增约378MB文件；pre-health可用1.27GB，后续传大栈前重新核对空间。旧Intern候选仍保留，旧关闭崩溃根因未定位；RAG Top3冲突仍未修，不执行电机。

IO/NPD8组已完成并回收results/single-block-profile-r1，4组输出与纯计时bitwise一致。L2 M16 run112.09ms、flush/invalidate合计0.037ms；M32 run222.04ms、同步0.092ms。实际NPD已用SDK解析，但当前仅NN计数，缺可验证node trace前不能声称CPU fallback或下游engine时间占比。

完整新范围基线fullbaseline-c1280-r1由fullbaseline/controller_fullbaseline.py启动，WSL PID4991，配置48条固定32步+12自然+8格各4步诊断，语音/RAG常驻。只读该目录controller状态和log，勿绕锁轮询板卡、勿重复启动。C1280+mask/group/greedy，无真8，只system64/128复用。输入正文64/128/192/256另加8模板；输出32是固定反馈性能，不是自然回答。

图侧正在独立补node trace和FP16 FC→1x1Conv候选，未上板。Mac原始报告副本 reports/intern-prefill100-20261025/ORIGINAL_COMPARISON.md。本阶段代码/证据将单独提交，GitHub是否已同步以实际commit及远程SHA验证为准。

## 2026-10-25 同输入NPU prefill/decode优化已完成本轮测试

权威Windows仓库 F:/Files/OPi-Lab/repos/OragnePi-3W_ForVLM，分支codex/wsl-intern1b-reproduction。Mac→Windows Tailscale100.113.74.31→WSL→板卡192.168.10.11；板卡操作使用board_access.hostlock与板端flock。当前本轮控制器均结束，没有待续跑任务；不得重复启动旧run。

最新报告 research/intern-nextbench-20261025/analysis/combined-matrix-r2/REPORT.md，配套INDEPENDENT_RESULT_AUDIT.md/json、summary_next.json、post-health.json。Mac副本在reports/intern-nextopt-20261025/combined-matrix-r2。combined-matrix-r2 rc0/meta complete：72性能、36自然、36cost自然、15份4步完整logits、3次正常关闭driver0。

对齐契约：static64/128=system正文59/123+模板5；dynamic正文32/64+模板8；总输入104/136/168/200，reused64/128，new40/72；性能输出32固定反馈，不是自然32token回答，每格3fixture×2观察。真8组合四格TTFT P50 0.493827/0.869842/0.494815/0.890454秒，P95(max)0.497368/0.874951/0.496498/0.915570；24/24性能请求<1s，decode P50 15.499/15.384/15.285/15.298t/s。当前有限短测试达标，未承诺长期全部<1s；不含VAD/压缩/排队/RAG。

机制：C1280→C256短图减少无效容量计算，真8图替代8尾部的16padding，单遍argmax提速decode。输入没有减少。C256真8比本轮C256基线再省约0.086–0.088秒，但driver从666.4增至942.4MiB。整轮CPU平均7.20%（8核归一化），温度峰50.282℃，harness+worker PSS峰591.9MiB，MemAvailable最低3065.1MiB，load1峰4.11；swap-out1708页=6.672MiB（仅加载），swap-in0，性能阶段无新增swapout。收尾driver0、alloc/free550/550、可用约4478MiB。音频31514帧=504.224秒、77端点、1566 overruns、maxlag0.649秒，不能称无音频迟到；无audio error。

相对本轮C256基线，24固定预测/12自然/12cost完全相同，5×4步完整logits bitwise相同；与r3 C256greedy共有4diag也bitwise同。C1280→C256的logits relativeL2仍约2.93–3.26%，不能称旧图无损。自然目标完成9/12、RAG Top3有移动/摇头冲突，禁止全Top3执行。无视觉、真实麦克风或电机；语音常驻合成录音，RAG仅quality调用；并非优化版1小时完整Agent重测。

关闭故障边界：原combined-matrix-r1在close报double free/corruption，失败证据保留。原ASan及O2详细trace全顺序均未复现。runtime-combined-v2仅调整全groups→全networks→自有buffers顺序，推理模块不变；3次最小加载关闭+本轮三完整配置正常，根因仍未知，不能称根治/默认部署。见analysis/CLOSE_INVESTIGATION.md。v2 binary SHA ff758d8eb013b7ebd05eae1f95634ecef065b92f0bf1fca7b292256351a2900d。

C256限制：最大prompt200、32输出使用231KV，不支持该格64输出或追加256图像tokens。原容量图保留。下一步研究文档intern-context256-20261025/OPT_NEXT_PREFILL.md提出仅裁prefill内部无效历史H176/H192，未编译实测。后续重点：确定旧close根因、更多语义/heldout质量、RAG门控、优化版长期稳定与视觉，不要把本轮性能测试结束当整个项目完成。

GitHub已确认源码checkpoint c660150；本条最新交接与本轮最终报告/审计在同一个后续结果提交归档，最终SHA以git log为准。Windows HTTPS push曾超时，Mac临时bare仓库仅中继Windows bundle，源码依然Windows编辑。凭据仅进程内使用，没有写入仓库。没有活动heartbeat。

## 2026-10-25 r3阶段记录（后续组合已结束，见最新记录）

GitHub已同步f36733f（含实验源协议）；权威任务research/intern-nextbench-20261025。candidate-matrix-r3结束rc0/meta complete：144性能、72自然、24cost自然、24×4步完整logits、6次driver关闭均0。报告analysis/candidate-matrix-r3/REPORT.md和INDEPENDENT_RESULT_AUDIT.md/json；Mac reports/intern-nextopt-20261025/candidate-matrix-r3可打开。

输入固定64/128=正文59/123+模板5；动态正文32/64+模板8；总104/136/168/200；reused64/128，new40/72；性能输出32固定反馈。ctx256-greedy四格TTFT P50 .578593/.954445/.581457/.978168s，P95 .581129/1.006803/.58258/1.010522s；decode15.19–15.41t/s。仍不能说64全部<1s。相同quant/payload，图尺寸使logits相对L2约2.93–3.26%；该候选24/24固定序列、12/12自然、4/4cost自然token与mask-base相同。tail8-c1280固定仅22/24整序列一致，s64-d64-2首token“挥手”→“举起”；四diag只覆盖case0，不能掩盖。未替换默认运行时。

整轮最高53.32℃，CPU均7.59%(8核归一化)，PSS峰595.5MiB；ctx256-greedy driver666.4MiB，对照718.4MiB，tail8-c1280 1019.7MiB。swap-out1013页=3.96MiB，swap-in0；音频maxlag.704s首次出现在初始图hash阶段。ctx256最大当前prompt200+32输出需231KV，不适用于最大格64输出/图像256追加。仅Intern NPU文本、合成语音、RAG质量阶段查询、mock无电机；并非新一小时全agent稳定性测试。

后续combined-matrix-r1推理完成但close堆损坏失败；combined-matrix-r2已完成，详见最新记录。这里的旧PID/自动队列均已结束，不要重启。

r2首次因质量日志缺system_body_tokens停止且driver0，保留失败证据；已严格按同prefix元数据补齐并通过4正反测试，r3完整通过，不混用失败数据。

## 2026-10-25 同输入首字优化已完成：32正文小样本<1s，64正文未达标

运行ttft-aligned-matrix-v2完成returncode=0：180速度请求、60自然输出、24份完整logits诊断，6次driver关闭均0。控制器已结束，不要重复启动。报告：research/async-agent-20261025/results/ttft-aligned-matrix-v2/REPORT.md；统计analysis.json，独立数值numerical-audit.json。

输入契约：固定前缀64/128=system正文59/123+模板5；动态正文32/64另加模板8。总输入104/136/168/200，reuse64/128，实际new40/72，速度输出1。每请求只缓存system，同格完整IDs一致，未用exact或上一轮动态缓存。精确长度含标点填充，静态128重复相同核心；自然质量另测。

按静态64/128、动态32/64四格：plain TTFT P50=1.152739/1.759670/1.180238/1.798044s；tail8+group+mask=0.959470/1.591648/0.974067/1.607835s。32正文各9/9<1s，P95=0.962396/0.984416；64正文各0/9<1s。每格仅3场景×3次，P95为最大值，不是长期保证。64正文prefill本身1.56–1.58s，后续需图内/批图优化，宿主setup已约2–5ms。本候选尚未替换完整Agent默认运行时，未重跑优化版一小时。

group→mask四组4步全词表logits逐字节一致；plain→tail8分区已有2.69–3.06%相对L2变化，16/16argmax和12/12自然输出一致只代表开发小集合。任务完成静态64为4/6，128为5/6，弯腰退回静止，历史RAG兼容性错误未修。Top3存在摇头/移动等冲突，不能全执行。全程mock无电机。

语音常驻合成录音，RAG仅质量阶段调用，Intern NPU仅文本无视觉。约15.93分钟；整机CPU均7.87%，温度峰49.91°C，harness+resident PSS峰589.52MiB（不含其他常驻服务，driver另计），MemAvailable最低3279.84MiB，swap0。音频60586帧/150端点，最大帧滞后0.601s，不能说全链路无滞后或端到端<1s。

首次无v2因tokenizer SHA不符在加载前失败（0计时），已用板端原文件重新冻结完成v2。保留两份输入及失败证据，不能混用。候选源码提交ce1dbe3；结果报告另行提交。前阶段固定86的mask报告results/ttft-mask-matrix/REPORT.md已完成，勿拿它替代本对齐矩阵。

## 用户要求：所有性能汇报必须标出token口径

system正文、固定前缀含模板、完整动态正文min/P50/max、额外动态模板、总prompt、实际reused/new-prefill与输出token必须一起报。旧配对方案system正文81+system模板5=固定86；pair300-v2完整动态正文57/115/126，pair300-32为5/30/32；额外模板均8。勿把旧版摘要64上限误报为完整动态64，也勿把固定前缀86叫作纯正文86。

## 2026-10-25 首字运行时矩阵完成

135计时+30自然输出，5组关闭driver0；尾部批处理将正文32/48/64首字P50从1.450/1.765/2.103s降至1.064/1.375/1.672s，仍未稳定<1s。候选6/6自然输出与旧版逐token一致，RAG弯腰→抬头缺陷未修。report: research/async-agent-20261025/results/ttft-32-64-matrix/REPORT.md。正文32实际new46，模板8+system尾6。继续检查truncate_session的mask构造setup开销，尚未部署该新候选；勿重复跑已完成矩阵。

## 2026-10-25 32-token摘要配对已完成

两组各300秒/88推理：仅32正文+8模板，system86；模型TTFT P50 1.618→1.023s，P95 2.682→1.446s，尚未稳定<1s。32组26/88基本门控拒绝、输出长列表/超能力，质量未过关。见research/async-agent-20261025/PAIR32_REPORT.md。ttft-32-64-matrix已完成；runtime比较old/tail8/group/group-exact/old-repeat，结果见上文独立报告。

## 2026-10-25 1小时结果与32-token后续

stress60-v2板端225000帧/3600秒、1022推理完成，driver释放0；控制器SSH收尾3840s超时，已独立回收日志，不能称控制器exit0。事件→首token P50 2.348s、mock动作开始2.584s、decode13.50t/s，最高49.29°C，CPU平均6.237%，PSS峰585.66MiB且后期仍约0.20MiB/min增长。RAG动作兼容性仍未合格。详见research/async-agent-20261025/results/stress60-v2/REPORT.md。随后pair300-v2与pair300-32已由run_pair32.py顺序完成，独立目录/锁；不要重复启动。

## 2026-10-25 异步常驻Agent harness

见research/async-agent-20261025/REPORT.md。v1完成900秒/257推理，CPU平均6.47%、温度最高49.414°C、decode13.67t/s、事件→首token2.43s/动作开始3.56s(P50)。性能稳定但决策失败：仅30/257明确运动，拒绝/物品操作仍被RAG映射mock。v2提示+基本门控75秒回归完成，仍有弯腰→抬头兼容性错误，未通过动作验收；后续v2长期测试已完成，见本文件更晚日期记录。没有物理电机动作。两个版本证据分开保留，板卡无本次harness任务运行，driver释放0。后续优先修RAG兼容门控、规划评测、2h内存增长验证。

## 2026-10-22：无屏幕、无手指的128动作v2

用户确认头、双臂、腰及能平移/转向的低自由度腿部底盘；无眼、无屏、无手指。当前词典切换至 `research/home-companion-atoms128-v2-20261022/`，128项R编号、0技能；v1仅作历史。具体轴/范围待验证，全部未映射，没有动作下发或新版模型复测。详见新版清单、审计和HANDOFF。

## 2026-10-22：家庭陪伴128原子动作集合

按最新要求单独建立 `research/home-companion-atoms128-20261022/`，128原子、0组合技能；90机械设计+38屏幕效果，全部硬件未映射。本轮只建词典，未进行128候选模型复测，旧35+20测试结果保留。完整清单、检查和后续待办见该目录 ATOMIC_ACTIONS_128.md 与 HANDOFF.md。

## 2026-10-22 家庭／桌面动作RAG新基线

用户要求原子与高层动作都限定家庭陪伴，管理接口不得进入动作字典。已完成35原子、20家庭组合技能、60条输入、五编码器共1200次板上CPU检索。新报告及完整交接：`research/home-companion-rag-20261022/HOME_COMPANION_RAG_REPORT.md`、同目录`HANDOFF.md`。旧256项混合库只作历史证据，不能继续作为家庭动作字典。无硬冲突与全可执行、目标线索与任务完成分开统计；Top-K仍需编排。

## Jev 后续队列已回收（2026-10-21T03:43:42.499314）

优化及共驻自动回收见 research/jev-validation/JEV_DEPLOYMENT_20261021.md 最后一节与 results/continuation-status.json。仍需审计结果、补完整质量/长稳/视觉并同步 Mac 报告；远程下载独立运行，尚未由此脚本确认完成。

## 2026-10-21 Jev逐模型提速审计追加（实验继续）

用户要求每个Jev优化延迟/决策速度，已加入 JEV_OPTIMIZATION_AUDIT.md，由审计智能体逐项核验。Windows权威报告 research/jev-validation/JEV_DEPLOYMENT_20261021.md 已重写，旧版本存sources留痕。

Jev-Style17→2槽：10候选7.097→6.969s；100候选57.562→57.141s；PSS2.331→1.995GB（−14.4%）。全部28答案与评分逐值相同，质量仍4/8、安全0/2。只认内存改善；小幅时延差待交错复测，不能声称已可部署。

当前优化5972，自动续跑6704（替代6300/6535，增加20计时答案及18英文控制parity），先完核/线程/runtime矩阵，再独立进程与持续决策共驻。不得在控制器持锁期间另行板端采样。新版板端Jev身份清单在最终配对前计算；当前优化队列旧源码冻结sources/optimization-v1，不能拿新版hash追认旧run。

远程是CPU存储服务器，仅下载9B/27B-VL；01:34仍未完成。已连接过，SSH会间歇重置，Tailscale未关闭。温控PWM/PD5冲突预先存在；dmesg环形日志已轮换，不能以无后快照错误当作全区间健康证明。

最新seq2+ub256：10候选7.133s稍慢，100候选55.927s，PSS1.932GB；答案不变、10候选评分有数值差，仍安全0/2。远程下载新PID123340/123341，每仓库8并发断点续传。续跑会补从未启动配置，不把已有超时部分结果重写为完成。

未完成：优化全矩阵/优化后共驻、长稳与更完整质量、OneJev多模态、服务器完整SHA、完整mock执行链。保持原始证据与全部TODO，不标整项完成。

## Jev 逐模型优化补充（2026-10-21 01:03，进行中）

用户明确要求每个Jev优化决策速度/延迟并交给审计。审计`research/jev-validation/JEV_OPTIMIZATION_AUDIT.md`已记录62项。Nano中文tokenizer失效（100动作只有34种encoding、86.57%UNK）；Laya完整缩短基线已回传：10候选1.259秒，direct100 9.607秒6/20，two-stage2.482秒5/20，unknown/safety均0/4。完整连续样本峰值71.744℃、RSS1.922GB。不能把部署/提速称为质量达标。

WSL控制器4805执行基线共驻；4806排队启动优化。查看`results/run-board-pairs.log`、`results/run-board-optimization.log`，不直接抢板卡锁。JevStyle旧部署缺transitive LD路径，Windows已修复，等优化控制器重新同步。Laya ORT FP32和encoder-only INT8导出及四组动态形状sanity已通过，尚待板测。新优化记录源码SHA、包版本、PSS/CPU频率/swap流量。

存储服务器下载v2 PID90989/90990（curl断点续传，3线程/模型）；目标`/infinite/common/yejy_vla/jev_models`，无GPU仅下载。manifest仍running，不可报告下载完成。Tailscale保持启用，Mac→服务器直连；SSH MaxStartups间歇重置可用已有expect重新建master。

# 当前交接更新：2026-10-21 Jev 部署（进行中）

- 当前权威工作目录：F:\Files\OPi-Lab\repos\OragnePi-3W_ForVLM\research\jev-validation。Windows通过Tailscale 100.113.74.31；板卡192.168.10.11，通过Windows/WSL跳板操作并持有双锁。
- 用户已授权下载、板端测试、项目中间产物清理；原计划“只规划”限制已由后续用户指令替代。不要驱动真实电机。
- 远程116.172.93.223:19515为无GPU存储服务器，用户明确仅下载，目录 /infinite/common/yejy_vla/jev_models。SSH曾成功；后台下载9B/27B-VL已启动，但连接有MaxStartups/认证前reset，完成状态需查manifest。不要关闭Tailscale。
- Nano33、Laya322、JevStyle0.8Q4、OneJev0.8下载完成；前两者已推板并二次校验。Nano/Laya单模型筛查任务及随后JevStyle/共驻任务正在执行，详见research/jev-validation/results/run-board-*.log。不要并行启动第二个板卡controller。
- 500M/Intern1B为主要模型，保留1.5B对照。1.5B历史NPU约5.03t/s但质量失败；不得称未测过，也不得称可用INT4。W4实际INT8存储。
- 已清理28个int16 decoder.export.data共990086768字节；源权重、编译输入、最终NB、活跃FP32 oracle数据、TF备份均保留。此为WSL内部空间，不等同F盘物理缩小。
- 本轮报告：research/jev-validation/JEV_DEPLOYMENT_20261021.md。OneJev图片前向、视觉encoder共驻、音频原生Jev研究已列README TODO。尚未完成的矩阵不得写成完成。

---
以下保留此前交接历史，旧IP/关机状态以本段及最新报告为准。

# 智能体接管说明

更新日期：2026-10-14。本文是接管入口，优先于历史交接里的队列、在线状态和“继续运行”指令。研究结果基于提交 `1600c9e0959a4905df770e7eb2bc13b0c1464ab8`；本文件随后单独提交。

## 先看当前状态

- 用户要求关机安装风扇，已执行板端 `sync && systemctl poweroff`，返回成功；等待20秒后SSH连接超时。已确认关机请求被接受、板卡离线，未通过串口确认最终断电状态。
- **不要自动重启测试或把离线当故障。** 等用户完成风扇安装并重新上电，再做健康检查。工作站没有关机。
- 本轮所有测量队列均已结束，没有需要恢复的旧会话。`下次继续.md`较早章节里的PID、session及“正在运行”均为历史记录，不要照抄启动。
- 24/24对齐性能格已完成，但NPU仍慢于优化CPU，最新候选质量仍未全部通过。不能把“实验结束”写成“项目达标”。
- 本次handoff只整理文档，不改变代码、模型、生产配置或测量结果。

## 目标与不可变的比较口径

在A733 / 6GB Orange Pi Zero 3W上，优化SmolVLM-500M与InternVL3-1B的完整语言生成链路，重点是解码速度和增量首字延迟。暂不测试视觉。最终用途是陪伴机器人，关注对话自然度、角色稳定与真实多轮KV复用。

- 500M全部32层、1B全部24层，以及完整词表输出头；包括必要转换、同步、采样开销。两层探针只能定位瓶颈。
- 当前对应CPU量化基线为Q8_0；NPU主要W8/A16/KV16，敏感位置FP16，**不是纯INT8，也不是Q4**。不得拿生产网页的Qwen Q4替代这两个模型的对照。
- 每模型分别对比规定搜索范围内最快有效CPU配置，不能限制CPU线程或选慢FP32实现制造NPU优势。CPU关闭FA在当前长历史场景通常更快，具体配置以矩阵证据为准。
- 核心输入为已有有效KV历史512/1024 × 新增32/64/128 token（包含模板），同一输入和CPU反馈轨迹，固定128预测、127个decode间隔，预热后至少5轮。
- 温度以响应开始时分档：低温40–50°C、中温55–<65°C；不是全程恒温。用户要求主要低、中温，在中温附近优化，高温暂缓。风扇改变后须记录新散热条件，不把前后变化全归因代码。
- 首字为引擎增量TTFT，不含模型加载、历史KV准备、控温和HTTP缓冲；冷启动、真实网页延迟另报。
- 测量必须配套同轮CPU、内存、换页、I/O与温度。整机CPU指标8核合计为100%；RSS和driver共享内存不可直接相加当唯一物理占用。没有电功率实测。
- 原计划的最终目标为各核心档位比最优合格CPU快至少15%，同时质量、端到端、稳定性过关；这是研发目标，不是已实现或保证实现。

## 环境、路径与访问

本机项目真实路径（UUID结尾为87b9，不是旧消息里的87b91）：

```text
/media/ubuntu/2e819926-8686-430b-9faf-c1a95a1c87b9/yejy/Files/orangepi-zero3w
```

| 路径（相对项目根目录） | 用途 |
| --- | --- |
| `npu/kv-cache/` | NPU编译、运行、验证、报告源码及Markdown报告 |
| `npu/results/` | 本机完整实验、图、原始logits与运行记录；大部分不进Git |
| `npu/kv-cache/evidence-attack-20261013/` | 已归档紧凑证据 |
| `docs/` | 用户文档、调研与交接 |
| `src/`、`configs/`、`tests/` | 应用、模型配置与测试；先看各自README |

板卡最后地址为 `192.168.88.29`，SSH用户 `root`。重启后IP可能改变，不盲目操作其他设备。密码不写入Git，使用本机既有认证或由用户提供；脚本支持环境变量 `SSHPASS`，本机工具为 `/home/ubuntu/.local/bin/sshpass`。SDK入口见 [SDK获取记录](../npu/kv-cache/SDK获取记录.md) 及NPU README，先核实现有环境，避免重复下载大型工具包。

**所有板端操作（含只读）都要串行使用host与board双锁。** 先查锁和任务，`WRITE`是持锁，`WRITE*`是等待。已有runner若内置锁，不再外层重复获取同一把锁造成死锁。手动SSH的无密码示例：

```bash
# 在项目根目录运行；使用已经配置的SSH认证。
flock -w 20 npu/results/board-host.lock \
  ssh -T -o ForwardX11=no -o ConnectTimeout=8 root@192.168.88.29 \
  'flock -w 20 /run/opi-npu-kv-benchmark.lock sh -c "hostname; uptime; df -h /; free -m"'
```

关机前最后快照：TF可用约2.75GiB，RAM available 4618MiB，swap占1164MiB，driver分配0，温度38.936°C；主机空余约27GiB。这是历史空闲快照，重启后重新测量，swap占用不等于正在换页。新部署/完整logits可能写满TF，保留至少脚本要求的512MiB额外空间。只清理已在本机完整SHA核验的板端诊断副本，保留恢复清单，不删除模型、系统、唯一证据或无关文件。

## 已有结果与不能外推的边界

当前对齐矩阵24/24，每格五轮、同温档、同CPU反馈；**不同格可以是不同明确标记的候选版本，不是一个固定产品模型完成全矩阵**。

中温、新增64 token，数值为中位数：

| 模型 / 历史 | NPU / CPU token/s | NPU / CPU TTFT秒 | NPU / CPU整机CPU% |
| --- | ---: | ---: | ---: |
| 500M / 512 | 20.432 / 22.780 | 0.753 / 0.704 | 1.27 / 26.56 |
| 500M / 1024 | 19.089 / 20.373 | 1.223 / 0.902 | 1.14 / 26.35 |
| 1B / 512 | 15.935 / 18.900 | 0.793 / 0.655 | 1.24 / 26.55 |
| 1B / 1024 | 15.597 / 17.760 | 1.237 / 0.792 | 1.13 / 26.55 |

最新有效研发增量：

1. **500M prefill8**：全32层512＋64中温交替5轮，t32→t8首字0.921722→0.752878秒（−18.32%），总响应−1.95%，decode20.521247→20.432035基本持平；driver减少32.73MiB。只测这一性能格，其他格不是t8。执行L2约0.03685、最小cos约0.99667；原Q8 KL约0.00652、PPL比约0.98408，开发轨迹通过。
2. **1B cap1＋headMSE**：512＋64全24层decode15.028152→15.934702（＋6.03%），TTFT0.802761→0.793124；1024＋32约＋5.81%。不同实验收益不能相加。
3. **1B cap1＋原输出头**：512＋32全24层14.965381→15.490959（＋3.51%），原Q8 PPL比1.048677，接近1.05阈值。headMSE在该格PPL比1.053872失败，不能混同。
4. **down投影缩放拆分**：保留原INT8整数及有效权重，FP32等价、编译字节与实板局部输出通过，但两层API变化不足0.2%，不值得直接扩全模型。1B t8单位token局部成本比t16高约39%，也未扩展。500M t64此前更慢，不采用。

普通1B cap1两层profiling中FC/融合约91.51% busy cycles、attention matmul约4%；仅说明该层组。完整模型宿主开销即使全消除，剩余收益也只是几个百分点。主攻大矩阵执行、布局、访存与合理量化，不再把微小宿主改动当作突破。

## 质量、KV与产品状态

- 标准模型参考与自定义oracle已有核对；开发执行误差与相对原Q8质量是两道门槛，必须同时过，不可看到更快后放宽。
- 500M新t8：36数值＋36自然回答均完成，数值仅24/36通过；聚合KL/PPL通过不能掩盖单例失败。英语简单角色/猫名尚可，中文倾听三轮复读、角色记忆有额外退化；部分更正失败CPU也存在，需区分基模能力与量化退化。
- 1B最新组合独立v2为33/36、新head单独32/36，均未过整套质量。
- 某些结果JSON顶层 `passed` 表示运行成功，**不是产品质量通过**。看逐例门槛、整套判断和生成原文。
- 500M t8复用经模型/套件/逐文件SHA验证的标准Q8 FP32参考，是重复冻结套件回归，不是新独立留出。若用这些失败来指导调参，下一验收必须换新独立数据。
- 常驻模型与跨请求KV已全深度核对，reset、最后token计数、窗口裁剪有验证。不要退回每次重算全部history来比较。
- 旧候选有30分钟稳定、实际浏览器与真实ASR并发证据；新图不能自动继承。新t8仅有自己的HTTP/常驻验证、质量回归和退出driver=0，未做自己的30分钟验收。
- 实验NPU网页是**工作站HTTP/分词网关，经SSH调用板端常驻NPU/KV**，不是板端独立NPU网站。生产8080关机前仍是CPU服务（Qwen2.5-1.5B Q4 / 4096）；不要部署质量未通过的新图覆盖它。

## 源码与证据定位

从 [本轮技术攻关结果](../npu/kv-cache/本轮技术攻关结果.md)、[最新完整矩阵](../npu/kv-cache/最新CPU对齐与组合候选矩阵.md) 开始。总览末尾“继续完成待跑矩阵及输出头消融”为残留旧行动文案；本轮这两项已经结束，以本文及矩阵24/24为准。

关键入口在 `npu/kv-cache/`：`run_full_model.py`、`benchmark_variants.py`、`resident_http.py`、`run_resident_http_validation.py`、`compile_down_postscale.py`、`reuse_holdout_reference.py`。执行前看参数和锁实现，不凭历史session重建命令。

本机 `npu/results/` 下的定位名：

| 结果目录 | 内容 |
| --- | --- |
| `cpu-finalist-realignment-20261014` | 最快CPU轨迹重新对齐；状态含已保留拒绝项，不能只看根status推断缺格 |
| `smol500-prefill8-trial-20261014` | 新t8完整性能 |
| `holdout-npu-prefill8-smol500-20261014` | 新t8质量及实际自然回答 |
| `holdout-reference-prefill8-smol500-20261014` | SHA核对后的标准参考复用 |
| `cap1-originalhead-short-trial-20261014` | 1B512＋32原head消融 |
| `cap1-headmse-short-trial-20261014` | 1B512＋64组合 |
| `down-postscale-20261014` | 缩放拆分，包括失败首轮与修复后的attempt2 |

500M新t8源目录为 `full-smol500-stable-v3-c128`，decode变体 `w8-fused2-c768-dfpgateup`，prefill变体 `w8-fused2-c768-t8-dfpgateup`；部署tag `full-smol500-c768-dfpgateup-prefill8`，case `prefill8-dfpgateup-smol500-h512-n64`。不要仅凭源目录c128后缀误判实际容量，核对变体和manifest。复用图通过硬链接与SHA核验，不复制多套占满TF。

报告生成顺序（都在 `npu/kv-cache/`，执行前看各CLI）：`report_candidate_matrix.py` 默认及 `--finalists` → `report_candidate_speed.py` → `report_prefill_screen.py` → `report_down_postscale.py` → `report_prefill8_quality.py` → `plot_finalist_matrix.py` → `report_attack_summary.py`。现有报告已生成，不需为了接管重复跑板端实验。

## 重新上电后的工作顺序

1. 用户确认装好风扇并通电后，核实主机身份/IP、双锁、无旧任务、温度、TF/RAM、NPU驱动、内核异常和CPU网页健康；记录散热变化。先排查实际证据，不把掉线直接归因高温或掉卡。
2. 选择并冻结每模型候选、量化配置与独立质量协议，沿用现有严格门槛。用户优先提速，但不能拿不正常说话的模型宣布成功。
3. 优先形成有依据的大矩阵/decoder提速假设；局部真实输入筛选有明显收益且质量通过后，扩完整32/24层、完整head，再做五轮配对。保留失败attempt与来源SHA。
4. 500M t8是否扩512/1024×32/64/128取决于修复质量后的候选价值；不要重复已完成24/24旧版本矩阵。新风扇下用于归因的CPU/NPU应采用相近温度、同轮配置与轨迹。
5. 质量通过后再做新候选自然多轮、真正部署形态的TTFT、30分钟稳定及真实ASR共存；单独报告纯NPU主计算与CPU/NPU混合方案。
6. 每个阶段更新报告、实际输出、同轮资源和未达标项，提交并核对远端SHA。若硬件/编译器受限，给出实测瓶颈，不换慢CPU基线。

## Git备份与恢复边界

仓库：<https://github.com/Behavior-Cloning/OragnePi-3W_ForVLM>，分支 `main`。源码、报告、紧凑证据已备份；权重、系统镜像、SDK、完整logits、NBG/ONNX、虚拟环境、凭据不在Git。新机器clone后不能假设立即复现实板实验，按 [备份与恢复](GitHub备份与恢复.md) 准备本地资产。

本机已有推送认证可用命令级SSH重写，勿修改全局Git设置，不把密钥内容写进文档：

```bash
git -c core.sshCommand='ssh -o ForwardX11=no -o BatchMode=yes -o IdentitiesOnly=yes -i /home/ubuntu/.ssh/id_ed25519_github' \
  -c url.git@github.com:.insteadOf=https://github.com/ push origin main
```

提交前检查 `git diff --check`、文档链接、敏感内容与备份范围；推送后通过同样认证的 `git ls-remote origin refs/heads/main` 核对本地HEAD。保留不属于本任务的用户改动。不要将Git备份描述为完整TF系统备份。

## 2026-10-26 C256/C448 board smoke

C256/C448 full 24-layer board smoke evidence is in `research/intern-prefill100-20261025/CAPACITY256_448_BOARD_SMOKE.md`. C256: system64/dynamic64 cold TTFT 0.830s; system KV reuse 0.606s. C448: system64/dynamic256 with system reuse TTFT 1.373s; system128/dynamic256 with reuse 1.346s. Token accounting is system正文59/123 + system模板5, dynamic正文64/256 + dynamic模板8, actual new72/264. Both workers released with driver_bytes=0. This is synthetic text-only Intern NPU and mock worker; no microphone, vision or motors. The 100ms full-model target remains unproven and unmet.


## 2026-10-26 C512 matrix32

- 新增对齐矩阵：静态32/64（正文27/59 + 模板5）× 动态32/64/128/256（模板8，实际new 40/72/136/264）。
- 完整24层 Intern NPU、只复用 system KV；M16/M32/M16-repeat 共72个性能请求和24个诊断请求均完成，driver close=0。
- 当前 M32 TTFT P50：490.9–3046.5 ms；最佳静态64/动态32为490.9 ms，100 ms目标仍未达到。decode约15.15–15.42 tok/s。
- 结果报告：research/intern-prefill100-20261025/matrix32/REPORT.md；Mac副本：reports/intern-prefill100-20261025/matrix32/REPORT.md。
- 静态32是成本fixture，不作语义质量结论；语音为合成前端，无视觉、真实麦克风或电机。


## 2026-10-26 版本冻结与原量化C448尾段对照

- 权威索引：research/intern-prefill100-20261025/VERSION_MANIFEST.json、BEST_VALIDATED.md。历史性能候选、有限数值验证和生产默认分开；默认未改。
- 实验提交223654e：C448原量化A/B/A共72计时、36自然输出、6四步logits，3次关闭driver=0。模型/输入SHA在c448-matrix-r4/config.json，报告同目录REPORT.md。
- system正文59/123 + 模板5 =64/128；动态正文64/128/192/256 +模板8 =实际new72/136/200/264；只复用system，固定反馈输出32。tail8 TTFT P50为s64:810.6/1501.8/2213.0/2935.0ms，s128:824.5/1539.9/2223.0/2955.1ms，decode约15.1tok/s。100ms目标未完成。
- r4完整logits按输入三组逐字节相同，自然12案/固定反馈24案序列相同；有限一致性不等于正确率。最高50.716°C，进程PSS峰596.7MiB；语音1555/31334帧超16ms、累计最大滞后2.053s，swap发生，未获全链路稳定资格。RAG仅驻留无检索；Intern文本NPU、合成录音，无视觉/电机。
- 原373/653/935/1218ms是C512 row-scale候选，精度改变；保留最快性能记录，撤回C256 plain/原精度标签。C256/M32与C448/M32 smoke继承range16 L2/L3，撤回已验证最佳。详见BEST_VALIDATED.md；旧数据保留。
- r2 tail8加载退出的容量白名单缺陷已在r3独立runtime修复，保留端口校验；原量化图诊断与r4计时完成。不要重启r4。
- static32/64 × dynamic32/64/128/256历史baseline见matrix32/REPORT.md；r4未覆盖static32/dynamic32，不拼接冒充新矩阵。
- 后续保持独立候选优化FP16 down瓶颈；数值/语义/资源资格分别复核。质量最优指针暂空，不推广速度候选。运行verify_versions.py复核登记；Windows代码权威，Mac仅报告副本。

## 2026-10-26 outlier8完整模型测试

- 上一轮有实质进展：离群通道拆分完成离线筛查、局部480次NPU API和完整24层A/B/A。局部输入为12个历史捕获窗口、每层8/4864通道保留FP16；其余down走I16，其他12个FC权重字节不变。原双层块120.47→18.42ms，仅局部API。
- 权威目录research/intern-prefill100-20261025/down-outlier-split-r2。代码/实验提交48d61ae，补充资源报告提交32303f3；full-test/REPORT.md与ANALYSIS.json已核验。所有controller terminal complete，勿重跑。
- system正文59/123+模板5=64/128；动态正文64/128/192/256+模板8=actual new72/136/200/264；只复用各自system，候选改变量化也影响system构建。速度输出固定反馈32，计时不含冷system/ASR。
- 新候选TTFT P50：s64=343.3/601.0/858.9/1115.5ms，s128=345.2/599.9/860.2/1118.6ms；prefill约323–1098ms；decode15.15–15.51tok/s。更快观察已登记，100ms全范围目标仍未达成。
- 12自然序列与原版完全相同；冻结约束复核9条直接完成请求，3条保守不动（wave/bend）与原版相同。不是100%正确率。full logits相对差2.81%/3.05%，固定反馈23/24序列相同；原组前后logits字节相同。质量最优仍空，默认未改。
- 完整实验语音/RAG常驻，RAG仅加载无检索；纯Intern文本NPU、合成录音、无视觉/电机。最高50.902°C，CPU8核归一化P50 6.78%，PSS峰580.7MiB，候选NPU ready约967MiB（口径分开）。语音1345/26921帧超16ms、最大累计lag1.247s；swap发生，不能称全链路实时/长期稳定。3次关闭driver0。
- 候选NB主/尾SHA、runtime/82条模型路径哈希与fixture在full-test/config.json。原量化r4与历史row-scale记录保留，跨轮次排名不代表算法单变量因果。
- 局部板端scratch236727836B已逐SHA回收后删除；完整候选在/tmp/opi-outlier-full-c448-r1，当前无活跃worker，driver释放，Windows原始f32/资源证据仍在。下一步可接紧凑past128/full-query图，但保留完整有效KV/绝对位置，不能外推局部两层到整模100ms；须继续8格与数值/语义/资源对照。

## 2026-10-26 compact-fullquery-r3：保持版本选择，未推广

当前目标已更新为prefill约500ms并优化decode。原100ms仅为历史记录。最新源码d139d0d，报告提交27dc3f27f61415f572e6b7dd41101b44f98c96cf，tag exp/intern-compact-m128-r3-20261026。Windows为权威；Mac报告副本已同步至reports/intern-prefill100-20261025，12个文件SHA核对通过。

完整24层NPU A/B/A已完成rc0：system正文59/123+模板5=64/128；动态正文128/256+模板8=new136/264，只复用system；固定反馈输出32。候选TTFT约595～632ms/1114～1129ms，decode约15，无一致净提速。12自然样例11同序列、一例同义措辞变化；完整logits相对父版本2.68%～3.14%，非准确率。关闭新路径时长输入诊断与旧outlier8 runtime字节一致。

额外NPU resident约349MiB，可用内存最低116MiB、swap活跃；3closed driver0，最高49.91°C。语音常驻合成流；RAG仅驻留；无视觉/电机。非长时间认证。追加板端图305411904B已在19份回收文件SHA核对后释放，父版本和默认不动。

最优跟踪：performance_candidate仍outlier8-c448-full-r1；latest_original_quantization_checked仍c448-original-tail8-r4；quality_qualified_best为空；latest_evaluated_candidate为compact-fullquery-r3但不推广。详见research/intern-prefill100-20261025/VERSION_MANIFEST.json、BEST_VALIDATED.md和compact-fullquery-r3/full-test/REPORT.md。

下一步可评估更小query块与避免额外整套图常驻；不要把两层局部API或编译结果算作完整TTFT。GitHub远端本次校验仍超时，未确认同步。不要重启已结束控制器571或复用run ID。

## GitHub同步收据：c96d22a（2026-10-26）

已通过Mac本机系统代理（127.0.0.1:7897）推送Windows权威快照c96d22aaa91592ef1d356cb60c0048304599c1d2，分支codex/wsl-intern1b-reproduction。8个不可变实验标签均已上传，独立ls-remote核对分支、tag对象和peeled commit全部相符。Windows网络SOCKS旧尝试失败，相关转发已撤销；未关闭Tailscale，未复制凭据。

当前完整模型资源优先性能候选仍为affinity-m64-r6；r7计时诊断和r8两层局部试验均不改变最优指针。baseline、原量化验证、逐格最快观测、质量合格最优（当前未指定）及生产默认分别记录在VERSION_MANIFEST.json。verify_versions.py本次再次通过15份报告提交绑定、195条token记录、93份full-logits、7组矩阵核查。默认未改，500ms长动态输入目标仍未完成。

本收据证明c96d22a代码/报告快照；收据后续提交本身的最终远端校验保存在F:/Files/OPi-Lab/transfers/github-sync-verified.json，避免在同一文件内自引用提交号。Mac仅临时bare relay与报告副本，工程代码始终在Windows编辑。


## 2026-10-27 视觉token压缩、112保留64与CPU Q8首轮

权威新任务research/intern-vlm-quality-20261027，读VISION_TOKEN_COMPRESSION.md、vision-frontmerge/SCREEN_REPORT.md、vision-frontmerge/Q8_REPORT.md、VERSION_TRACKING.json、ROOFLINE.md。文本NPU r6和默认未改。

已测CPU6/7两线程、配套F16 mmproj：448原生24.3–24.6s/256token；448前merge4.066s/64但6题1对；224原生4.011s/64且5/6；112保留64经通道重复适配0.915s但0/6；448后pool24.308s且核心事实6/6（一条输出截断）。2图6题只是开发筛选，不能称可靠。112 stride1邻域投影后续host筛选仍0/6，未部署ARM。

视觉Q8仅改144个ViT线性矩阵，其余298张量保持；619.9MB→336.8MB。严格224前后对照FP16热P50 4.002s→Q8 4.294s（慢7.28%）；RSS约612.4→341.5MiB；同6题均5/6。只标内存候选；Q8-448后pool26.712s、6/6。不能声称量化已提速。当前Vision均CPU、语言VQA参考在WSL Q8，尚无端到端板端NPU VQA。system30+5=35，图像64+问题5–9，模板/边界11；总115–119，CPU参考每题清KV所以reuse0。

板端run vision-frontmerge-r4和vision-q8-r1已完成并回收；不重复启动。112/224实际输入尺寸，未放大回448。前序r2输出buffer为空导致embedding回收失败已剔除；r4修复。Q8权重在F:/Files/OPi-Lab/models/internvl3-1b-vision及板/opt/opi-vlm/experiments/vision-q8-r1/encoder-q8.gguf；source SHA89e15ad8...，Q8 SHA65c3352c...，全SHA见manifest。板卡TF可用剩余约200MB，进一步部署先核查空间，保留原基线/已验证源。

源码仍只在Windows编辑；Mac报告镜像目录：/Users/yejunyin/Documents/ChatGPT/小机器人开发/reports/intern-vlm-quality-20261027/（本轮传输并核对MAC_REPORT_FILES.json）。CPU视觉编译依赖在/opt/opi-lab/builds/intern-frontmerge-20261027-r1/source，父repo Git hash不是llama.cpp版本；dependency-source-sha256和patch、snapshots/board-r4冻结测量源。当前build-host/vision-worker已含neighbor候选，不能拿它冒充board-r4 binary；板r4的SHA64050efb...及source snapshot保持。未开启后台测试。

理论估算ROOFLINE.md基于官方3TOPS@INT8/LPDDR5最高19.2GB/s和真实语言维度；理想W8短输入TTFT下界26–33ms、decode带宽上界约39tok/s，而无限带宽算力上界约3000tok/s。这是条件性乐观估算，当前混合NPU真实流量/吞吐待测，不能将其当已实现指标。

下一步：Q4/CPU kernel profiling、NPU vision export和激活校准；修复112/前merge质量；接真实视觉embedding到NPU r6及system-only cache；扩充VQA并测试语音并发。

## 2026-10-27 NPU视觉输入接口候选 r1（未上板）

新增 research/intern-vlm-quality-20261027/npu-vision-input-r1；从r6所用compact-m64-r4 runtime独立复制，ARM编译成功，候选SHA8e5381670c1cd724ac1e08e713d0c2aae83cc4a71b2eeaf9766868e897bc814c。main/compact/tail8/补齐batch及scalar都支持896维真实视觉向量覆盖。协议flags bit1+reserved视觉行数，再附image_start/system_reuse_cap和F32向量；限制最多64行并检查长度/finite。前后请求均限制system cache，避免同占位符换图误用KV。

Host ASan/UBSan检查通过：文字回退、图像边界、padding、reset、越界/截断/NaN/Inf拒绝。仅证明接口单元检查和编译，尚无NPU数值parity、图像A→B缓存实测或端到端VQA；默认及baseline未变。下一步先上板验证旧文本完整logits一致、换图/图转文字与fresh session一致，再接已冻结224视觉输出及真实常驻视觉链路。没有启动后台任务。

## NPU视觉输入 r1 板端首轮完成

见 research/intern-vlm-quality-20261027/npu-vision-input-r1/REPORT.md。9份文字/等价注入完整logits相同，真实图像换图和图→文字cache测试通过；6道开发VQA 5/6（木板→树枝仍错）。真实CPU视觉embedding预计算，语言为板端完整Intern NPU，未测共驻端到端。system30+5=35，image64+text5–9，额外模板11，reuse32/new83–87，语言TTFT0.308–0.325s，不含约4s视觉。四次worker关闭driver0。ViT NPU初步导出工作另见vit-npu-r1，112保留64路线保留，尚未完成部署。

## ViT NPU首块实跑（112路线保留）

vit-npu-r1已导出112/224首块ONNX；112 INT16完成编译、板端2预热10热API测试，中位11.029ms；relativeL2 2.353%、cosine0.999733、释放0。只首块、单图测试，非完整视觉塔，不外推24层速度/精度。112保持64patch，末端adapter质量待修；下一步精度消融及全塔。当前没有后台任务运行。

## ViT NPU首块精度矩阵

vit-npu-r1/MATRIX_REPORT.md：112/224×FP16/INT16/INT8已上板，14次进程关闭driver0。112 API FP16约634ms、INT16约11ms、INT8约6ms；224约2598/54/42ms。INT8 relativeL2约40–64%不能用；INT16约1.38–3.81%，需检查全塔累积；FP16约0.13–0.15%但慢。不外推完整塔。CPU Q8速度另在vision-repack-r1进行同binary开关重排buffer实验，查controller.json，勿重复启动。

## CPU视觉Q8重排候选完成

vision-repack-r1/REPORT.md：同binary重排off/on/off，224→64从4.28s降2.26s、约47.3%减少。144 Q8矩阵启用CPU_REPACK，RSS约343MiB。向量relativeL2新旧4.76/7.74%，不是无损；6题核心事实命中，但竹子回答错误扩写并截断，不能称全答对。新向量NPU VQA和换图缓存检查完成，driver0。快候选不覆盖默认；当前无后台测试。112保留64与完整ViT NPU仍待完成。

## 常驻视觉＋NPU语言端到端 r2

见research/intern-vlm-quality-20261027/resident-vlm-r2/REPORT.md。r1加载可用内存42.5MiB触发保护，0请求；清理已逐SHA归档的225.9MB临时ViT矩阵副本后r2同阈值完成12热请求，两个模型各只加载一次。JPEG→首token P50 2.615s、完整回答3.071s；视觉2.247s、语言首token0.309s、decode16.57tok/s；最高59.83C，整机CPU P50 25.68%，可用内存最低197MiB，测量期swap增量0。system30+5=35/reuse32；动态图64+文5–9，模板11/new83–87；无图像或动态缓存。无音频/RAG/视觉NPU全塔/长测资格；竹子错误扩写与截断未解决。当前无后台测试。

## ViT完整24块NPU与单层诊断

见research/intern-vlm-quality-20261027/vit-full-r1/REPORT.md。24块1.833s（NPU1.307s，主机转换同步0.526s），最终hidden误差83/88%，质量失败。第14层正确参考输入也有约100%误差，非单纯累积；单层W8A16 API54.4→49.8ms，不能外推全塔。非线性FP16诊断编译unsupported失败。stem/projector已编译未板测；112保留64路线不删除。所有板测关闭driver0；无后台任务，未推广默认。

## ViT选择性精度修复

见vit-full-r1/SELECTIVE_REPORT.md：24块1.744s，误差3.77/3.82%。首次内存保护终止，逐SHA归档清理旧图512MB后r2完成driver0。NPU视觉块+离线stem/projector→NPU语言VQA5/6，木板仍错；一次TTFT17.195s/API0.909s待诊断，不报稳定0.32s。system30+5=35/reuse32，动态图64+文5–9/模板11/new83–87，输出7–11。尚非完整塔共驻，未晋升默认，无后台任务。

## VQA尾延迟监测待恢复

见vit-full-r1/LATENCY_WATCH_STATUS.md。r1/r2缺内核PSI和进程io接口，已兼容缺失；r3控制连接240秒超时，远端状态未知，勿重跑。板卡ping正常，SSH banner超时；Windows正常。先恢复连接、确认进程和锁、回收r3证据。endpoint probe已交叉编译未板测。

## 复测恢复及背景CPU模型

r3 SSH恢复后原任务日志已回收，18题完成driver0，TTFT P50 .30965s/max14.086s，首轮竹子慢但后两轮正常；慢请求swap in/out=0，进程majorfault6。背景opi-vlm-model.service CPU Qwen2.5-1.5B占约2GiB、NRestarts5，已问用户是否可临停，未批准前保持。原“远端未知”状态已解决，不重跑r3。endpoint四进程板测完成：stem API1.8ms/误差.034–.051%，projector7.3ms/误差3.64–5.89%，driver0；全26图串联待做。见LATENCY_WATCH_STATUS.md与ENDPOINT_REPORT.md。
