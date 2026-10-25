# Intern prefill / TTFT 100ms 验收审计

状态：目标尚未达成；本文件是验收与防遗漏清单。只在 Windows 权威仓库编辑；本审计没有启动、连接或轮询板卡。root 独占板端部署与双锁。证据比较见 EVIDENCE_REVIEW.md。

## 目标冻结与判定

- [x] 保持现役 Intern 文本模型24层、完整151674词表、同tokenizer与语义输入；静态system总长64/128保持不变。
- [x] 动态摘要正文范围为64～256 token；不能将范围目标缩为仅64或仅某个批大小通过。
- [ ] 主验收必须包含 system S={64,128} ×正文 D={64,128,192,256} 的8格；相同case ID、实际input IDs/SHA横向配对。摘要正文之外模板token单独计数。
- [ ] 按当前模板，actual_new=D+8，即72/136/200/264；不能将正文64写成“仅64新增prefill”。模板若改变，另建版本并实测token数，不覆盖旧口径。
- [ ] 每格同时检查完整动态prefill≤0.100s与模型TTFT≤0.100s。未指定百分位时，不只用P50宣告通过：冻结样本集的实测最大值也须≤0.100s，并列P50/P95/max/N；任何格失败都不能写“64～256全达标”。有限样本不代表未来所有请求的服务保证。
- [ ] 主格使用至少3种不同家庭陪伴语义、不同token组成；计时前冻结重复数及顺序。建议正式每格≥30次；少量3×1或3×2只记smoke/探索，不把其P95当生产尾延迟。
- [ ] 测试批边界：对实际新增N=D+模板数，覆盖每个启用M∈{16,32,64,128,256}的整数倍附近N=kM±1、kM、kM±7、kM±8、kM±9，取D∈[64,256]有效交集并去重；包含D=64/65、255/256。边界按实际新增计，不能只取正文整倍数。
- [ ] 最终若要声称连续区间都覆盖，另做D=64..256整数扫长（两种system，至少一条冻结内容族），并明确其样本局限；否则结论限定为8主格与列出的边界集。仍不得将未测192/256省略。

## 缓存、容量和时间边界

- [ ] 只允许模型常驻、system KV预热；动态请求在触发后首次到达。禁止先算动态KV、命中上一次同摘要KV、偷偷重用输出或只计最后一个chunk却宣称完整prefill。
- [ ] 每次硬断言 actual_reused==S、actual_new==len(all_input_ids)-S、dynamic_body_tokens==D、full_prompt==S+D+template；动态部分哪怕重复也不能计入缓存命中。记录prefix IDs/SHA、重置/截断过程及请求序号。
- [x] 已发现旧runtime对reuse向主batch倍数下取整；直接M128/256会令system64或128失去复用。runtime候选设计采用独立M16 system bank，协议显式prefix_tokens=64/128，并限制reuse不能超prefix；尚待真实实现/执行核验。
- [ ] 确认没有通过扩大system静态长度、添加离线动态前缀或OPI_EXACT_PREFIX未经验证路径掩盖问题。跨system64/128交替与同system连续请求均检查。
- [ ] C256不可用于新最大范围。C512最大prompt=128+256+8=392，固定32预测需392+31=423个KV槽；保守含32计424仍≤512。必须按实际输入+输出-1断言，不能截断正文来通过容量检查。
- [ ] 完整动态prefill=所有batch prompt工作+所有scalar prompt工作（按每次相加后统计，不能两组P50相加）；新增dynamic_prefill_seconds或等价墙钟覆盖两者。
- [ ] TTFT原模型边界：输入token请求已读入/验证后开始，含setup/truncate、全部动态prefill、final norm/full-vocab head与argmax，到首token产生。不得扣掉NPU同步、CPU搬运或首head。额外报告caller提交→首token观测时间，避免将排队/IPC/tokenization隐藏成端到端100ms。
- [ ] speed阶段diagnostic=False，不写全词表dump、不查询逐算子SDK性能计数；阶段profile开关的开销独立配对。模型加载/system预热单列，不混入热态TTFT，也不能不报告启动成本。
- [ ] 固定32反馈性能、自然EOS质量、4步完整词表诊断分开。decode=(Nout-1)/(total-TTFT)；output1试验没有可比较decode。

## 图与runtime交付接口

- [ ] graphs候选：C512×M16/32/64/128/256，先L0/L2两个fused2块探测；每块11输入5输出。scope必须标two_layer_probe，禁止将一块延迟当24层完整TTFT。
- [ ] 每个shape记录source/quantize/data/导出量化payload SHA，证实除C/M整数维度外未换权重或校准。NB SHA、编译器/SDK版本、目标容量、M、fused层数记录到manifest。
- [ ] L0/L2初筛只能决定下一步；正式必须24层、12个fused2块、4个完整词表head分片就绪，ready验证full_depth24、capacity512、loaded_bank_tokens。
- [ ] runtime预期独立M16 system bank +动态bulk M32/64/128/256；剩余支持M16 padded8，M8图可选。记录每请求各bank调用数、真实有效行数、padding行数、scalar回退数。不得只展示理想整批输入。
- [ ] prefillext按base/bulk/tail分别列host_prepare_s / npu_run_s / host_publish_s / read_last_s；明确包含关系，分项不得重复加到prefill总时间。异步提交必须测实际完成，不能只测enqueue。
- [ ] 图拼接/alias、position、mask、KV布局、last-row选择逐项验证；批大小变化需要完整数值检查，不能假定同权重即同输出。

## 数值质量与资源

- [ ] 使用同一冻结输入、初始KV和固定反馈，比较baseline和候选输出预测与完整4步logits；对实际f32重新算SHA/长度/finite、相对L2、maxabs、top1一致。first-only不能代替所有层/步骤。
- [ ] 自然生成覆盖家庭陪伴、否定、禁止底盘移动、位置方向、时间/数字、停止/安全、超出能力请求；逐例评审高层动作是否可执行/冲突，不只看token相等。
- [ ] shape-only也要测质量：既有C256对C1280 full-logits已有约3%relL2漂移。任何W4/INT4、激活精度、KV精度或校准改变必须单列实验并配对质量；不得混成“同精度原模型100ms”。新小模型/剪层/截词仅另列替代方案，不能交付为原目标达成。
- [ ] 语音降噪/VAD/KWS/流式ASR与RAG维持相同常驻/亲和配置；分别声明“模型常驻”与“实际执行并发”。连续RAG执行干扰需单列，不把常驻测试当全时并发。
- [ ] 每次0.5s采样整机CPU、load1、各核频率、温度、harness+worker RSS/PSS、系统MemAvailable/swapin/out、NPU driver；同时记录音频frames/overrun/maxlag/队列语义与错误。CPU百分比注明整机8核还是单核。
- [ ] 大M新增bank可能增加权重副本/激活/KV/mask与编译占用；确认WSL/板盘剩余空间，删除仅有清单且可再生项，保留fallback与失败证据。
- [ ] 每个worker正常closed driver0；异常后driver0与正常closed分开。原r1关闭崩溃根因未确定，v2短矩阵成功不能直接称小时稳定；候选压力回归时长由root冻结并如实报告。
- [ ] 采用前baseline→候选→后baseline；同case/repeat配对并报告漂移、CPU频率/温度变化。不要把跨时段P50比值包装为无噪声加速。
- [ ] RAG Top3仍含矛盾动作，compatibility未裁定，保持mock；性能达标不会自动使其可同时执行。

## 结案输出

- [ ] 每个主格/边界的PASS/FAIL/MISSING完整表；标出最慢格及未达标差距，不用全局平均掩盖256失败。
- [ ] 独立报告冻结参数、weights/graph/binary/tokenizer/fixture SHA、原始日志、阶段时间、质量差异、资源与正常释放证据。
- [ ] 100ms全范围若未达到，明确剩余瓶颈和证据化可行性；不给没有端侧完整测量支撑的承诺。保留当前可用baseline，目标状态保持未完成。

## 单块probe控制器独立审查

本节仅针对L0/L2候选初筛，不能结案100ms全模型目标。

- [x] board_probe_suite.py / controller_probe.py源码已完成只读静态检查；补充失败回收与样本守卫后允许root启动单块初筛。尚不代表板测通过。
- [ ] scope明确two_layer_probe，严格允许声明manifest的C512/M16,32,64,128,256/L0,2；禁止模糊匹配到其他模型或整模进程。
- [ ] 每个候选3次warmup+30次正式repeat，warmup不混入统计，输出所有样本而非只保留汇总。
- [ ] A-B-A明确A的具体shape/layer/hash，前后A使用同二进制、同input和设置；不能仅把候选列表首尾同名当完整热漂移控制。
- [ ] canonical输入实物核验：same token stream不等于same float tensor。每layer统一history、hidden/KV来源，所有M按规定切片；mask/rope根据shape/position构造；reference由这一套实际输入重新计算。
- [ ] 只在主机完成source/input/reference SHA与长度校验后部署，板端再次验证实际NB/binary/input哈希；不得信任仅文件名匹配。
- [ ] 正式样本保留NPU调用完成时间、host负担、起止温度、频率、内存/driver；单块NPU时间不标成完整TTFT或端到端100ms。
- [ ] 有限超时、温度上限、内存/磁盘失败时保留partial；异常路径会停止自有子进程并记录退出码，不吞掉信号/崩溃。
- [ ] hostlock与板端flock都覆盖部署/执行/证据回收需要保护的区段；不得另开SSH绕锁读板。stdout非JSON/远端命令失败不会误报complete。
- [ ] 无论成功失败均回收已有meta/events/resources/stdout/stderr；确认正常释放/driver0；原始失败证据不覆盖。

已发现并通知root的待解决证据：`prepare_fixtures.py`对各M独立执行FP32前向。相同layer的M16/32对M256输入切片存在小差异：L0 past-K/V最大3.0517578125e-5；L2 hidden最大4.7683715820e-6、past-K/V最大2.0980834961e-5；M128 L2 hidden最大3.8146972656e-6。该差异来自实际输入文件逐项比较，不是NPU输出。该旧fixtures版本只能称同token流，不能称严格相同canonical float输入。建议以M256统一history/hidden/KV裁切重建，再重算reference；保留原fixture作可追溯证据。


### 复审更新：canonical与控制器

新 `fixtures-canonical/` 保留旧files，已由本审计独立逐字节核验10cases×11fields共110个输入字段，都等于同layer唯一M256源的规定裁切；mask对7组分别裁history512与当前M列。10个reference文件finite、长度和实际SHA均匹配canonical status。canonical manifest SHA256：fc51a89b51525a36eb32a69a26479d506a9d8bdbade8ec34961c12f0c9480303。原输入微差问题在这套新fixture中已消除。

代码静态审查确认：每层M16→64→128→32→256→M16 repeat；每case独立子进程、3 warmup +30 timed；env清除所有OPI后仅设warmup，禁止NPD/IO profile/resident copies；time样本finite且>0；probe源码逐输出检查finite与重复一致；dump尺寸及SHA回收。binary被冻结为eed89e71c0030ea15fb90d490611b6d6efdcf88a530906fdc57a1c42498e4da3，manifest/input/reference均验证实际SHA。

最初发现的回收缺口已修正：SSH/deadline异常保留原error、标recovery_pending，在重新获取hostlock与短flock后尝试回收partial archive；若远端仍持锁则保留pending，不另起任务。正常非零退出也回收已有输出。板端监控0.25s采样，78°C触发停止，启动前≥65°C最多等180s，probe180s超时，外层timeout1200s/kill10s。资源线程join后仍alive会标失败，不误记complete。

双锁边界：Windows hostlock覆盖部署、运行等待和回收；部署与回收命令各持flock，远端整个suite持flock。运行时只读exitfile以发现退出，不用旁路SSH轮询NPU状态。正式执行、回收meta、输出与FP32 reference数值校验仍须看实际结果；`complete`先表示suite执行完整，不能单凭重复一致判定精度合格。

限制：这套probe只测一个fused2块、无audio/RAG常驻，资源是整机CPU/mem/driver等，并未采process RSS/PSS；不能据此交付全系统资源指标或整模TTFT。A是每layer的M16，仅包围同layer候选批组，需按相同A前后30样本比较漂移。建议显式assert结果scope/capacity/query_rows/fusion_layers，已有NB SHA与dump尺寸提供间接约束。

本次静态复审源码SHA（后续修改需重新对照）：

- `controller_probe.py`: `d550d669d3509d0cae4311b1d642b2eaa5ea9ff0484554c049a0066b09d40dd4`
- `board_probe_suite.py`: `28efcf567d0aec9e412b96cf38409c32b8d67f8463c8a7dee0414a2d133980c9`


### Runtime v2 静态复审（不等于板端验证）

已读取 `runtime/schedule_prefill.h`、`resident_session.c`、`batch_prefill.h`、`resident_worker.py`及主机测试证据。delivery manifest内全部source和binary逐个计算实际SHA/字节数，零不符；resident为 `340e45340b43173109e615a9e4015bc5b1858b3efe945be4af826efa53baad62`，probe仍为此前冻结的eed89e71...。ARM build rc0，测试日志全部rc0；本审计没有重复运行或上板。

静态成立的契约：

- 协议v2显式prefix64/128；reuse比较在prefix硬截止并按16对齐，动态请求若未准确复用整个prefix立即失败，Python再核验reused/new。不同system必须先reset prepare；不存在动态KV偷偷命中的路径。
- runtime限定24层、C512和独立M16 system bank。scheduler在pos<prefix时只走M16，bulk仅用于动态部分；拒绝EXACT_PREFIX/scalar-block旧实验开关。C512最大主格392 prompt、32预测占423 KV槽，容量检查在读输入前执行。
- bulk完整行优先，余数走M16、可选真8、或scalar；可选bulk padding默认关闭且检查物理shape不越容量。run_prefill_rows只向KV/history发布valid真实行、last_hidden读取valid-1；当前batch因果mask阻止真实行看见未来padding。正文256+模板8仍为264真实新增，不会被记成256。
- 完整dynamic_prefill_seconds逐request为batch+scalar；TTFT从请求读完/验证后开始，含setup、完整动态prefill、full head/argmax。固定反馈decode按(N-1)/(total-first)；diagnostic开关独立。额外分段时钟仅profile启用，SDK run墙钟不能称纯芯片算术时间。
- 各bank记录calls/logical_rows/physical_rows，padding是physical-logical。profile关闭时细分为0只表示未插桩，不能解释成零开销。多bank共享KV而非权重，需完整resident实测driver/PSS。

已有主机证据范围：scheduler 12840 requests /162904 choices满足逻辑token覆盖、system M16和容量；三bank mask/reset测试6组×3600 tensors；所有INT16 greedy221212 cases零不符；mock所有权128 sessions双close后资源计数0；worker协议3例通过。它们没有验证NPU数学、SDK真实生命周期或全模型性能，不将mock close通过写成板端故障根除。

结论：未发现阻止下一步受控测试的prefix/尾batch静态错误。正式完整模型仍须对每个实际输入做system-only reuse/实际new、bank逻辑行总和+scalar==new、padding、输出数、full logits/自然质量、真实normal closed driver0检查；C/M改变可能改变数值，既有C256约3%logit漂移已说明同权重不保证bitwise。当前只有L0/L2候选图，不能将runtime编译成功或单块probe替代24层性能与100ms验收。
