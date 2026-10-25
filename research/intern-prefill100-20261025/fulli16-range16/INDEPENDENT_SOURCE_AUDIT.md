# Full I16 Range16 源码与契约审计

无启动阻断。只读审查运行源和冻结配置，未访问板卡、未编辑源码。源码及config SHA见同名JSON。

- 原图→I16→原图，C1280、同binary/flags，29个graph/embedding映射仅prefill-layer2.nb改变。两处down新量化被明确标注，不声称字节/输出全等。
- 性能24输入与旧fullbaseline完全相同；自然24与原semantic24完全相同。8格每格3案例；实际动态新增72/136/200/264，最大prompt+31=423。
- 每次测量前reset并仅准备system缓存，实际reuse/new有硬断言。性能固定反馈32步且diagnostic=False、禁止dump；decode是首token后的31个间隔。完整动态prefill按每请求batch+scalar再统计。
- 72性能、72自然、24诊断请求×4词表步骤与3close是预期完整数量。自然EOS/32上限、diagnostic4steps与性能32fixed分开。

统计限制：每格仅3性能样本。旧自然集合已参与调试，不能称独立验证。系统负载采样包含每案重建systemKV、加载和诊断，需按phase筛选；这不等于生产场景一次固定KV后的长期负载。合成语音前端持续运行，但ASR文本不触发冻结LLM请求；RAG常驻，实际检索在自然输出后执行。无视觉、麦克风、电机。
