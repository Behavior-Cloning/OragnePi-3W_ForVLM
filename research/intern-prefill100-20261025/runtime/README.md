# Prefill100 运行时与探针

状态：仅Windows/WSL实现、ARM交叉编译和主机边界测试完成。板端由root串行安排；当前graphs目录只有L0/L2两个融合块的各batch候选，尚无可据此宣称可用的完整24层C512模型。没有达到或证明100ms目标。

## 输入和缓存契约

- full-depth resident固定24层、C512、M16 system bank。system为64/128 token总长（含5模板；正文59/123）。动态正文64/128/192/256另加8模板，所以动态实际72/136/200/264，总输入136/200/264/328或200/264/328/392。固定32输出占到总输入+31，最大423，完整容纳，不能省略模板或截短输入。
- 协议v2 magic `0x4f504932`，8个uint32依次为magic、id、prompt_count、max_new、feedback_count、reset、diagnostic、prefix_tokens。最后字段必须64或128，且不大于prompt_count。旧v1协议不接受。`resident_worker.py`已匹配v2；旧worker不修改。
- 必须先单独发送system请求：`generate(system_ids,max_new=1,reset=True,prefix_tokens=len(system_ids))`。随后完整请求：`generate(full_ids,max_new=32,feedback=frozen32,prefix_tokens=len(system_ids))`。自然输出不传feedback，diagnostic另跑。二者不会误把固定32反馈算自然质量。
- 完整动态请求在C端强制`actual_reused==prefix_tokens`，Python还检查`new==len(full_ids)-prefix_tokens`。重用比较硬性在prefix截止，绝不保存动态内容作为可重用前缀。不同system即使相同长度也须重新prepare。system64/128切换需先reset prepare。
- 不使用历史数值不等价的EXACT_PREFIX开关。system始终M16划分，大batch不能把64/128缓存向下取整到0。

## 图与参数

命令位置参数保持 `resident-prefill100 graph_dir 24 512 151674 embedding.f32 16 151645`；graph_dir需完整decoder/head `layer*.nb`和M16 `prefill-layer{0,2,...,22}.nb`。所有input/output实际shape、dtype、quant都检查。

命令需通过`env`显式注入：`OPI_GROUP_PREFILL=1 OPI_MASK_TEMPLATE=1 OPI_TTFT_TAIL_MIN=8 OPI_GREEDY_I16_SINGLE_PASS=1`。Worker会清除继承的所有OPI_变量，只有命令显式设置会生效。

- 额外动态bulk图：同时设`OPI_PREFILL_BULK_DIR=...`和`OPI_PREFILL_BULK_TOKENS=32|64|128|256`。未设时为同一新binary的M16 baseline。bulk目录仍每融合块一个`prefill-layerN.nb`。
- 可选真8尾图：`OPI_PREFILL8_DIR=...`，仅接受相同C512合同。目前图agent尚未编译M8，所以默认尾部8真实token用M16 padding；日志清楚报告8 padded rows。
- `OPI_BULK_PAD_FINAL=1`独立开关，默认关闭。开启时，如果动态剩余真实行大于16但小于bulk M、且物理M行不越C512，则用一次padding bulk图。所有padded行重复最后真实输入、因果mask阻止真实查询看见未来padding；只写回真实valid_rows的KV，history和valid计数不包含padding。剩8行仍走真8或M16，正文256+模板8不会被藏掉。
- 例如实际动态72在M128 padding模式是一张128行图的72个有效行；动态136在M256 padding模式同理。M256最大bulk仍无法一次覆盖264；默认是256+真实8尾，不宣称完成单批。
- IO别名只共享decoder KV，不是权重共享。M16、bulk、可选M8图的常驻权重分别分配，须实测driver及PSS，不能仅按单套权重文件估计。
- 所有boolean环境开关按是否存在判定，`=0`也会开启；关闭时应不设置。

## 计时与诊断

原`ttft_seconds`覆盖动态请求开始后的setup、全部实际动态prefill、首head和argmax；`dynamic_prefill_seconds=prefill_seconds+scalar_prompt_seconds`，包含所有真实动态prompt工作。system准备是独立request并标记`prefix_prepare_request`，不能混入或偷偷抹去未缓存system的开销。

`done`新增`prefix_tokens`、`padded_rows`、`prefill_banks`；每bank有rows、calls、logical_rows、physical_rows。固定反馈和诊断标志保留；运行时就绪后另发`runtime_contract`（worker已接收），明确protocol2/full_depth_text_decoder、C512、各bank规模及padding/profile开关。

仅`OPI_PREFILL_PROFILE=1`启用额外时钟，按各bank拆分：

- `prepare_s`：embedding/RoPE、mask更新、输入及共享KV flush。
- `npu_s`：SDK `vip_run_group`墙钟，包含SDK等待与NPU执行，不称纯芯片算术时间。
- `publish_s`：输出KV invalidate、真实行KV写回和decoder mask更新。
- `read_s`：最后真实hidden invalidate/read。
- `profile_total_s`：上述完整batch调用。计时加和应与完整prefill墙钟近似，剩余是外层调度/时钟开销。profile关闭时这些细分为0，不能误解为零成本。

正式速度需profile关闭；细分诊断单独跑，不能拿插桩结果假装原始性能。

## 独立单块探针

`tensor-probe-prefill100 nb input.f32 output.f32 repeats result.json`保持原argc6接口。`OPI_PROBE_WARMUP=3`；正式30次重复默认无新增分段计时。输入按实际11个端口顺序拼接FP32，输出为5个端口拼接的反量化FP32，与原probe相同。强制C512、融合2层、M8/16/32/64/128/256。

`OPI_PROBE_IO_PROFILE=1`独立诊断模式返回每次input_flush/run/output_invalidate/wall；默认旧execute路径。结果保留api_seconds、温度、driver allocated/released、重复输出差异等，并增加`scope=two_layer_probe`、capacity/query_rows、input/output_logical_bytes。这些逻辑字节是用户IO规模，不是实测DDR总流量；SDK公开profiling结构仅提供inference_time和total_cycle，不能据此编造带宽或TOPS下界。探针没有system/head/整24层，绝不称完整TTFT。

## 当前实际证据与100ms目标

旧combined-r2在C256、system64/128、动态正文64+8模板时，完整prefill中位约0.851/0.871秒、TTFT约0.870/0.890秒，首head约18ms；每请求真8尾全栈调用约100ms。它说明旧实现只压CPU setup无法达到100ms，但不证明新图、算法或硬件不可能更快。新的C512大batch单块实测必须先给出每batch run/flush和数值对照；相加单块只能是投影预算，不能当端侧全模型实测速度。

## 主机验证与释放范围

`host-tests.json`记录ASan/UBSan：scheduler完整覆盖正文0..320、prefix64/128、bulk全部档位、tail开关及padding开关；不漏/重复真实token、不越容量。三bank reset/truncate以实际函数测试C512及所有bulk形状，INT16/FP16各plain/template/validate；生命周期mock覆盖三bank及二次close；greedy复用221212项测试。`worker-test.json`验证协议、显式prefix及错误重用拒绝。它们不验证NPU计算或模型质量。

释放采用此前v2候选的全group→全network→自有buffer屏障，仍保留旧偶发close故障根因未定位的限制。候选不得替换默认服务；实际板测必须核对closed driver0和外部health。所有旧冻结源码/二进制未改。
