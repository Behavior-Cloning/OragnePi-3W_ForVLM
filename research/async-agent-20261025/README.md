# 异步家庭陪伴 Agent 端侧 harness — 2026-10-25

## 本轮约定

GTCRN、Silero VAD、Streaming Zipformer Small 常驻，16 ms PCM 输入由独立墙钟驱动。音频来源为已有12段中文合成语音循环，不是真实麦克风；不会等待大脑完成才开始下一段语音。ASR final 之后清空应用语段与重置 VAD，无一分钟音频归档；内部库缓存未直接量化。

最近60秒队列保存最终转写，按 VAD 端点减静音阈值的代理时间过期。进行中的partial单列且不当作最终转写。未接入声音事件或情绪识别。

独立墙钟以固定随机种子生成2–5秒事件。单NPU worker + 最多一个待处理事件，新事件覆盖未开始的旧事件并记录。只有worker实际开始接受事件时才快照并压缩，规则保留最近完整子句、否定/取消优先，按Intern tokenizer最多64 token。超预算完整子句不硬截断，遗漏通过字段披露。

Intern1B NPU文本decoder常驻，固定system64 token；不加载视觉encoder。worker自动复用最长完全一致前缀，不每轮重建模型或重跑前缀预热。脑输出32 token上限。BGE-small中文模型常驻，使用新版R001–R128词典索引。每次检索Top3，只有Top1用于模拟执行。

模拟动作持续8秒。新大脑结果经RAG完成后，在执行器线程中取消旧动作并替换；10Hz mock tick用generation ID验证旧动作没有继续执行。没有真实电机、TTS、物理急停、关节安全验证。RAG相似度不是可执行性裁决，v1未设完整合法性门控，因此性能成功不代表动作选择质量合格。

## 测试与运行

Windows/WSL为权威代码目录。`controller.py RUN SECONDS`通过board_access.hostlock及板端flock持锁部署和执行，结束复制JSONL、meta、NPU日志。不要与其他板端benchmark并行。WSL需独立keepalive，避免SSH会话关闭后WSL退出。

`smoke01`45秒是旧提示+每次前缀恢复版本，不能与`stress15-v1`900秒结果混算。结果以source_sha256识别。`summarize.py results/RUN`生成汇总；原始数据保留用于独立复核。

监测主Python进程及真实NPU subprocess，CPU按整机8核容量归一化；NPU driver内存独立列出，不与RSS直接相加当作物理总量。温度78°C终止。每500ms采样资源，每帧记录耗时与完成滞后；后者含计算时间。

## 验收界限和后续

- [ ] 查看完整900秒结果，核对音频时长、事件账目、退出与driver清零。
- [ ] 核对说话/NPU重叠、各5分钟延迟和资源趋势。
- [ ] 人工审查高层动作输出与Top1兼容性，添加拒绝无效输出的门控。
- [ ] 基于含否定、改口、取消、长句的独立人工标签集评估压缩质量。
- [ ] 实际麦克风、回声/噪声、视觉encoder、真实物理执行尚未覆盖。
