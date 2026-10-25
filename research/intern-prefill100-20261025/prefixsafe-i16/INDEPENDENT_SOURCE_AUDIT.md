# Prefixsafe I16 独立源码审计

本次冻结源未发现阻止有界隔离试验的缺陷；已开始运行，尚未审计结果。只读取 Windows/WSL 文件，没有板卡、SDK、锁或模型调用。

C 增量仅 batch_prefill.h 和 resident_session.c。header7 只允许 reset 且请求恰为完整前缀、M16 整除、无真M8；普通 continuation 不能变更保护长度，必须复用整段 protected prefix。reset_session 保留策略字段。harness 每次 system-only reset/prep 后才发完整 prompt，并强断言实际 reused=64/128、new=正文+8，未预计算动态 KV。

原图/候选敏感块绑定相同有效输入输出 buffer，所有合同检查保留。block0/1 独立、后十块永久分组，无 detach/rejoin。静态范围内选择原敏感块，动态选择 I16；scalar decode 没有修改。29 项主图只有 prefill-layer2.nb 不同，额外原图 SHA 显式核对；运行统一新二进制。

48 自然 fixture 与父轮字节内容相同；8 格 speed/diagnostic 仅去掉失效的旧 timing 描述，全部输入/token hash 不变。三模式共 48 次输出1性能请求、144 次自然请求（max32/EOS）、24 次固定反馈4步诊断，另有216次system准备和6次prefix-only全词表检查。预期原块选择日志数为0/74/444。输出1不能评估 decode。

已复核补充门禁：ready=24层/C1280/无真M8、monitor join 后仍活跃判失败、上传 SHA 同时等于冻结快照。四个 Python 协议测试记录通过，不能冒充 C 板端实测。双锁、2400秒执行上限、2550秒观察上限与不自动重启保留。异常分支不会自动回收 partial，后续须确认终止并回收，不能直接重启。

这是 C1280/M16、无语音/RAG、无视觉/电机的有界回归。单格一例两次、20个边界未跑，48自然集已用过。候选必须以实际 prefix logits、固定反馈数值与逐例自然语义审查，不因速度或 exact-match 数量直接晋升，也未达到全范围100ms目标。
