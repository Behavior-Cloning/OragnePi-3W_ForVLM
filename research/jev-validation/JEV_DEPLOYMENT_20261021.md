# Jev 板端部署与 VLM 共驻试验（进行中）

日期：2026-10-21。权威代码和实验资料位于 Windows F 盘，Mac 仅保存报告副本。当前报告尚未代表完整矩阵完成。

## 本轮范围与版本

主 VLM 保留 SmolVLM-500M、InternVL3-1B；Qwen2.5-1.5B 为用户指定对照保留。Jev 的动作均为虚拟标签，没有接入电机。模型原始选项准确率与未来确定性安全门必须分开评估。

| 模型 | 下载/部署 | 边界 |
|---|---|---|
| Nano-Jev v1.0 33.36M | HF mirror；ModelScope同名无记录；已板端校验 | 英文 RAG 交叉编码器，不能据尺寸推断中文动作能力 |
| Laya multilingual 322M | ModelScope，已板端校验 | CPU PyTorch，中文动作集正在测 |
| Jev-Style 0.8B Decision-v3 Q4_K_M | HF mirror，官方 manifest SHA 已逐文件核对 | WSL 已交叉编译 ARM64 jev-score；必须使用槽位评分协议，非普通聊天生成 |
| OneJev1.6-0.8B | ModelScope 完整权重已下载 | 多模态 encoder/图片前处理/板端部署尚列 TODO |
| JEV-9B / JEV-27B-VL | 远程存储服务器后台下载任务已启动 | 无 GPU 是预期配置，仅下载存档；尚不能标记完整校验成功 |

Jev-Style 的官方测试 runtime commit 为 `441df11f65ea0b6d0c72965aaf70c8241070ddcb`。网络下载缓慢，本轮交叉编译使用 slim20 恢复的 llama.cpp 源码快照；该快照支持 qwen35 与 n_outputs_max。其完整目录摘要、头文件摘要和编译器版本见 `sources/runtime-source-provenance.json`，不可把父仓库 Git SHA 冒充 llama.cpp 版本。最终效果必须以本轮实际运行结果为准。

## 已观察到的烟测

Laya 的4候选测试：加载约9.91s；预热后调用约0.485–0.496s；进程RSS约1.65GiB。输入“请看向左侧”却选了右侧（3次一致），因此该烟测仅证明成功推理，不能证明模型可用。后续以140条冻结样例的结果替代这个小样本结论。

## 测试定义与限制

- 小模型运行在 CPU A76核6/7，2线程；NPU用于500M/1B文本decoder。未包含视觉encoder前向，不称完整图片VLM测试。
- Nano最初10候选热态块：20次预热+200次测量；Laya缩短基线为5次预热+30次测量；优化初筛为3次预热+20次测量。候选4/8/16/32/64/100及较长状态先进行初筛，不能称全矩阵200次。
- 64/128/256是合成state长度下限，完整保留原请求，记录各分词器真实token数；不同分词器不能当作同token工作量横比。
- Nano对所有候选分别交叉编码，按8个一批执行；Jev-Style和Laya按自身原生候选机制。禁止静默截候选/状态。
- `sources/board_bench-screen-v1.py`保留最初板端脚本；该版调用计时包含一次轻量资源采样，后续脚本已经把采样移到计时外。最终报告应区分脚本版本。
- 全程保留原有1.5B CPU服务时，所谓Jev单模型推理仅表示Jev独占计算，不表示系统没有其它常驻权重。
- RSS不是NPU driver内存；共驻实验另记录NPU worker的driver计数。PSS更适合共享内存总量，不能把多个RSS简单相加当作物理内存。

## 历史 1.5B / INT4 纠正

恢复工作树内确有Qwen2.5-1.5B完整NPU实验：28 decoder层、4个完整词表输出头、ctx128、输入44/输出64、5次运行；decode中位数5.032149 tokens/s、TTFT 8.042304s、driver约675MiB。但输出重复，质量失败。

该候选名为w4-vdfp-sensitive、group28，W4值仍以INT8的-7..7存放。没有证据证明原生packed INT4存储和带宽加速。导出常量的FP32 oracle也重复，表明问题已在重数量化常量处发生，不应把这组速度当成可用INT4结果。500M/1B没有完整合格INT4结果。本轮保留这些失败证据，不删除对应源权重或1.5B对照包。

## 清理

实际删除 `gguf-fresh/export/int16/l0…l27/export/decoder.export.data` 共990,086,768字节。每层检查decoder.json/data/quantize与最终NB均在，ONNX嵌入权重保留。活跃w8-balanced/fused的FP32 oracle所需export.data全部保留。路径及SHA见 `results/cleanup-20261021.json`。

这是WSL内部回收约0.92GiB，不能直接声称F盘物理回收同等容量；尚未压缩VHDX。清理后又新增模型，F盘快照可用约58GiB；WSL内存快照约0.5GiB使用、15GiB可用，编译时会更高。

## 剩余审计项

- [ ] 汇总 Nano/Laya/Jev-Style 板端时延、准确率、温度、内存。
- [ ] 500M/1B NPU与1.5B CPU的分别常驻/同时推理矩阵；明确1.5B是文本模型。
- [ ] 服务器大权重完整性校验结束。
- [ ] 多模态OneJev的真实图片前向、视觉编码器与VLM共驻（README已单列）。
- [ ] 确定性安全门、mock回执、state_id/过期结果取消、最优候选1000次稳定性；当前不是实体执行链路。
- [ ] 完整A55/A76、1/2线程、状态长度和候选数矩阵；初筛不足时不能宣称已完成。

## 2026-10-21 追加优化要求（执行中）

每个 Jev 都必须尝试提速，并比较优化前后延迟、决策准确率、资源和与VLM的竞争；不能用部署成功替代优化验收。审计智能体已记录 `JEV_OPTIMIZATION_AUDIT.md`，62个待验收项。

已经准备的矩阵：Nano/Laya/JevStyle分别测A76单线程/双线程、A55双线程；Nano候选batch1/8、PyTorch动态INT8、ORT FP32/INT8；Laya仅编码器Linear动态INT8而保持决策头FP32；JevStyle序列槽17→2、ubatch1024→256。每个配置先3次预热、20次计时、8条质量对照（literal、paraphrase、unknown、safety各2条），属于初筛，胜出后才扩展验收。OneJev视觉未执行，继续列部署和优化TODO。

Nano的ONNX FP32导出烟测logit最大误差8.94e-7，INT8约0.106。INT8有数值误差，不据模型文件变小判定精度合格；需比较argmax和完整质量集。Laya直接量化整个决策头曾触发weight属性兼容错误，失败日志保留；当前编码器量化方案已通过桌面运行检查，板端结果待测。

另清理已安装的WSL 3.0.3 MSI和Ubuntu rootfs安装缓存共708,706,003字节，直接位于F盘；保留运行中的VHDX、TF备份及slim20原始恢复包。校验及清理清单见 `results/install-cache-cleanup-20261021.json`。

## Nano-Jev 板端基线（部分全量，主动结束慢速筛查）

A76核6/7，PyTorch FP32、2线程，模型常驻。10候选state实际66 token：20次预热后200次测量，中位数1.4411s，P95 1.4520s；平均使用1.964个CPU核，折算全机8核约24.6%。10条固定literal重复的准确率20%，不能当100动作准确率。

100候选：已完成109例，中位数14.2065s，P95 14.2649s；0/109正确，其中literal完整100/100条均未答对，paraphrase只完成9条。未运行到剩余样例/分层路径，不将文件标为complete。长期原始screen主动结束，完整保留JSONL与KeyboardInterrupt日志，再用固定小样本比较优化。

原生英文RAG两条控制：相关段落选directly answers（0.254s，首轮），无关段落选irrelevant（0.172s）。它是能运行但中文动作任务失效，不能用参数小推断适合本机器人。

观测到进程树RSS最高485,707,776字节（约463MiB），系统可用内存最低约4.09GiB，swap最高0.5MiB；测量快照最高72.334℃。此为请求边界观测最大值，原screen被中断未输出完整连续telemetry，不能称绝对峰值。新优化脚本设78℃保护。

后续优化初筛调整为3次预热+20次延迟+8条diverse direct100，减少反复执行已知低质量慢速路径；这是初筛，胜出后需补200次和更完整精度验证。

### Nano 中文失效的输入诊断

100个动作描述在Nano分词器中只剩34种不同token序列，86.57%的描述token为UNK；“看向左侧”和“看向右侧”等大量不同指令完全编码为同一串token。Laya/JevStyle均100种不同序列、UNK为0。证据见 `results/chinese-tokenizer-audit.json`。

这类输入信息丢失不能靠多线程、INT8或ONNX提速修复。Nano即使被加速也不建议直接处理中文动作；它的英文原生RAG功能应单独评估。若以后做级联，语言/候选可区分性检查应直接转交合适模型，不把Nano低质量结果当动作输出。

## Laya 基线完成（2026-10-21 01:00 附近）

原始文件 `results/board/laya-screen-t2-20261020T164634-10370.jsonl`，A76双核/2线程、PyTorch CPU常驻；153行含完成标记和连续telemetry。10候选30次（10个literal重复3遍）P50/P95为1.259/1.271秒，正确21/30。direct100的20例P50/P95为9.607/9.673秒、正确6/20；两级选择同20例为2.482/2.519秒、正确5/20，组路由正确9/20。

这20例实际是12 literal + 4 unknown + 4 safety，没有paraphrase；两条路径的unknown/safety都分别0/4，不具备直接用于动作输出的质量。安全门尚未计入这些模型原始准确率。后续优化对照采用四类均衡8例，不能把不同样例池百分比横比。

合并1162个连续样本，观测最高RSS 1,922,039,808字节（约1.79GiB），最高71.744℃，MemAvailable最低2,861,195,264字节（约2.66GiB）。请求边界快照会漏掉峰值，最终统计采用连续样本。

Laya ORT FP32通过上游batch1/2/3概率一致性检查，追加10/100候选×64/256状态四组决策相同。encoder-only ORT INT8实际量化88个常量MatMul，决策头保持FP32，四组决策亦相同。这四组均来自同一请求，只是导出/动态形状检查，不代表完整精度通过。板端FP32/INT8性能与质量仍待测。

## 当前运行与明确失败

- 基线共驻正在依次测Smol500 NPU、Intern1B NPU、Qwen1.5B CPU；500M和1B已加载并完成各自Nano/Laya调用，完整汇总尚待控制器回传。
- JevStyle首轮因`libggml.so.0`转移依赖未找到而加载失败；Windows adapter已增加局部LD_LIBRARY_PATH，下一轮部署修复后必须真实评分验证。本轮失败不算部署成功。
- 优化控制器等待基线锁释放。每配置独立进程；温度达到78℃停止该配置，下一配置需冷却至70℃以下。
- 远程下载使用curl断点续传替换易超时的urllib读大块实现；已完成文件保留SHA校验，9B/27B权重分片仍未完整。服务器无GPU不影响下载。
- 本轮另删除未完成且未用于构建的llama源码压缩包12,251,136字节，清单`results/cleanup-abandoned-source-20261021.json`。

## 首轮实际并发（每格5次；不是长时稳定性结论）

源`results/board/co-resident-20261020T165652.jsonl`。Nano/Laya与三种大模型各完成Jev-only、LLM-only、实际重叠三格，共18格90次；JevStyle因共享库错误在三种模型下都未完成。本版脚本无complete记录，但控制器正常退出；不可据退出码把失败三组当通过。

| 大模型 / Jev | Jev独占→并发P50 | LLM TTFT独占→并发P50 | decode独占→并发 |
|---|---|---|---|
| Smol500 NPU / Nano |1.488→1.587s|2.649→2.693s|17.49→17.43 tok/s|
| Smol500 NPU / Laya |1.322→1.411s|2.631→2.696s|17.48→17.39 tok/s|
| Intern1B NPU / Nano |1.459→1.551s|0.927→0.951s|13.74→13.50 tok/s|
| Intern1B NPU / Laya |1.316→1.406s|0.924→0.965s|13.76→13.50 tok/s|
| Qwen1.5B CPU / Nano |1.456→2.865s|1.343→2.832s|9.76→9.60 tok/s|
| Qwen1.5B CPU / Laya |1.325→2.501s|1.360→2.666s|9.67→9.66 tok/s|

全部为热态常驻，NPU worker CPU侧固定4/5，Jev6/7。Smol系统前缀92 token、块对齐复用64；Intern系统前缀37、复用32。输入模板/分词器不同，不能用Smol/Intern TTFT直接推导模型大小效率。当前NPU Intern decode也不能替代先前固定长prefix的15.5 tok/s实验。

该组支持“NPU解码比两个模型竞争A76更适合并发”的初步判断；未包含视觉前向、语音链路或两个VLM同时常驻，也未证明模型动作质量可用。JevStyle需要修复后补测，优化获胜配置也要重新共驻。

共驻审计补充：30个并发调用均有正重叠，但Smol两组的Jev都在首token前结束，Qwen1.5B+Laya也是如此；Intern的decode重叠仅约0.37–0.80秒。表中decode降幅很小不能推导持续决策时也几乎无竞争。优胜配置补测新增持续Jev循环，覆盖整个生成过程。

该共驻run的612个快照（含414个连续采样）最高进程集RSS 3,831,353,344B、PSS 3,821,982,720B，MemAvailable最低2,092,118,016B，最高65.224℃。swap最高51,118,080B，本轮累计swap-in增加393,216B、swap-out增加50,237,440B，存在交换。NPU driver加载分别701,586,072B/753,274,680B，两个worker关闭后均回到0。模型顺序轮换后PyTorch仍保留分配器内存，这些是实测进程集占用，不是各模型独立净增量；也不把driver与PSS直接相加。
