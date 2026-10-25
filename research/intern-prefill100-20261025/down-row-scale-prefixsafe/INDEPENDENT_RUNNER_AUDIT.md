# Protected-system 四阶段 runner 独立源码审计

结论：当前冻结源码未发现阻止这次有界端侧试验的问题。此结论来自 WSL 文件读取、内存中的合同负例和伪 Worker I/O 测试；**没有访问板卡、SDK、运行锁，也没有启动控制器或模型**。运行结果、精度和资源稳定性仍待实际证据。

## 对照与缓存协议

四阶段固定为 original-m32 → row-scale-m32 → protected-system → original-m32-repeat，同一硬化二进制0ed64448…，同C512/M32、词表、输入、反馈和原M8尾图。主图29文件只允许 prefill-layer2.nb 差异；全row与保护组使用同候选 f0bebd8b…，保护组额外装载原M32/M8敏感块。两附加原图在保护组load前以空cache重新计算SHA，并记录 extra_prefix_graph_verified；不再依赖第一阶段的旧核验。

只有保护组启用 protect_system、两个FIRST路径及非零header7。每次正式性能/自然/诊断请求之前reset并准备纯system，输出1；续请求必须reused==64/128、new==body+8，主batch次数2/4/6/8加真实M8一次，scalar_prompt_tokens=0。原始和全row两组也执行相同准备步骤，避免动态复用造成不公平。动态正文仍64/128/192/256，模板另占8，最大prompt+31=423≤512。

Worker新增header7的8字序列实测于 BytesIO 假通道：S64/S128精确编码、header7零兼容；非reset、准备输出2、长度不符、负值、bool全部在写入前拒绝。这不等于执行C runtime。12个配置负例（顺序、保护flag、原M32/M8缺失/错图、binary、容量、batch、EXACT_PREFIX、反馈长度、额外主图变化、尾图变化）均被合同拒绝。

## 额外诊断与分支数量

完整默认配置预期：96性能请求（固定feedback32）、192自然请求（max32/EOS）、32四步完整logits诊断、8单步system-only完整logits诊断、4缓存边界probe、4次关闭。40份完整f32共82,510,656B。准备与warmup不混入性能统计，所有额外诊断/probe标为timing_eligible=False。

保护组的system-only诊断之后，各S64/S128提交同一动态请求两次，均不reset；即使第二次历史完全匹配，硬化runtime仍须截到S，runner断言new=body+8。这样覆盖了旧实现可能复用动态KV的情况。

保护组共有83次前缀准备，预期原块分支248次：warm2、performance72、natural144、diagnostic24、system-only6。两个额外重复动态probe不应走原块。源码按顺序比较每条(pos,rows,valid,limit)，而非只计总次数；每条均应rows=valid=32、pos<S、limit=S，并核suffix marker数。其他三组原块分支均为0。prefix logits相等仍不等于全部KV字节相等。

## 常驻负载与控制器

语音Frontend持续流式回放合成WAV，RAG常驻、自然输出查询Top3；LLM输入为冻结fixture，当前ASR输出不驱动LLM，所以不是说话结束到大脑输出的全链路延迟。无视觉、真实麦克风、电机。资源0.5秒采样，78°C守护、后台线程错误及退出未结束均标失败；必须报告实际swap、overrun和lag。

控制器用hostlock及板端flock，唯一tmpfs目录；本地及远端检查新binary可执行。冻结上传SHA、本地复核、远端逐SHA检查、29个overlay精确symlink路径、候选字节数和SHA均受约束。预检要求tmpfs free>上传+200MiB且MemAvailable>上传+2GiB；仅为入场条件，不是长期可用内存承诺。执行timeout3600秒、观察3750秒，无自动重启。异常标recovery_pending，需后续人工回收，不能把失联当完成。scratch不会在结果核验前自动删除。

## 离线分析及验收边界

分析器覆盖96/192/32及8/4新增事件、四阶段顺序、实际f32SHA/finite/argmax、前缀token、缓存和248分支日志、三套图及附加原图SHA、归档SHA/成员与4次driver0。finish只观察本控制器PID和终态，成功后执行离线分析，不重新启动板测。

仅3个成本场景/格、一次通过；48自然样例已用于先前实验，是回归集。输出一致率不是语义准确率，仍需逐条看动作对象、方向、否定词和RAG冲突。保护路径热执行3次提交而非一次整组，这一开销不可从测量删除。boundary20未运行，≤100ms完整目标仍不能由源码或8格小样本代替。

离线分析器已补performance-only资源与逐case配对漂移汇总：分桶条件与已有helper一致；相同32输出下decode比值为31间隔耗时之反比，分母口径正确。swap页数、观测时长与n=3 nearest-rank P95=max的报告说明均已核对。冻结文件SHA见JSON，后续分析器如修改应另记新SHA；运行代码不得随意变更。

## 启动后仅WSL工具增量

父节点在源码审阅后启动唯一controller49958。本审计仅读取WSL状态，当前为running，无回收结果。上传快照与审计全部运行源码SHA相同。finish初版先于控制器首状态写入而退出，之后只调整观察器：状态不存在时检查精确PID/cmdline并等待，进程缺失则失败，不重启模型。新finish SHA38331a4a…，离线analyzer SHA9ca4c2c5…。未修改runner/runtime/contract/config或重复启动板测；不是端侧运行通过结论。
