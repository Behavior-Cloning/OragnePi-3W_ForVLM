# C1280 FP16 Conv：长输入自然质量和首 token 翻转诊断

准备完成，仅 Windows/WSL host preflight，尚未上板。由 root 审核后独占板卡执行。旧 fullconv A–B–A 的源码、二进制、结果均不修改。

采用 `semantic-long/inputs.json` 的原始24条质量输入：12个新写家庭场景×system64/128。system正文59/123＋模板5；动态正文64/128/192/256＋模板8，每个长度格3条真实语义长文本。字节完整复制，固定SHA，保留 source 和 GOLD_RUBRIC。不是之前的12条短指令，也不是填充成本输入。

两阶段原C1280→已实测单L2 Conv overlay。每阶段24条长自然解码、1条旧成本 `s64-d64-2` 自然解码、1条该成本输入的4步固定反馈全词表诊断。总计48长自然＋2成本自然＋2诊断/8步。自然max_new32，允许EOS提前结束，不以强制反馈代替自然生成。没有重复性能矩阵。

成本输入单独标记 `quality_cost/cost_padding`，只定位首预测117057→114209的翻转，不计长自然质量分母。其完整prompt136、prefix64、动态正文64＋模板8，prompt SHA a3fb1ef3e7e58b7ac78e62aef8f079c5c4de196f3318a79a0a07ec2781d5669e。诊断实际保存两份各2,426,784字节F32，核对finite、argmax、SHA；回收后比较原赢家/候选赢家margin、误差和自然路径是否改变。

每请求reset并单独准备system，只复用system KV。候选system也走新L2，因此同前缀文字与策略不代表KV数值相同。记录真实token数、prompt SHA、prefix准备、TTFT和完整动态prefill。自然解码耗时仅描述质量请求，不能混入之前固定32性能统计。

语音前端以预录WAV流式回放，降噪/VAD/KWS/ASR与RAG驻留，每条自然（含成本）实际查Top3但不执行动作。0.5秒监测包含已知resident PID、温度、频率、内存、CPU、swap、音频计数。不是用户说完→首token的完整事件链。

## 运行

默认只做host配置与本地SHA预检，不导入board_access、不访问板卡：

```sh
/opt/opi-lab/export-venv/bin/python research/intern-prefill100-20261025/fullconv-quality/host_preflight.py
/opt/opi-lab/export-venv/bin/python research/intern-prefill100-20261025/fullconv-quality/controller_quality.py quality-dry
```

Root批准执行后的唯一新run（示例名字需保持唯一）：

```sh
/opt/opi-lab/export-venv/bin/python -u research/intern-prefill100-20261025/fullconv-quality/controller_quality.py fullconv-long-quality-r1 --execute --timeout 1200
```

所有板卡访问需WSL hostlock＋board flock；原图目录不变。复用依赖：
`/opt/opi-vlm/intern-prefill100-20261025/fullconv/runs/fullconv-c1280-aba-r1/overlay`
及同目录 `prefill-layer2-conv.nb`。该34,050,560字节候选图 SHA dc772225e8c2b9ec2f6376132ba12aa9938dbe966b678e2bd7672832299fcfa6 必须保留，不能作为旧run临时文件清理。

每次加载前核对29链接/完整图SHA（共享目标利用相同size/mtime缓存SHA），两版仅prefill-layer2不同。新上传约1.4MB源码、fixture和小binary，不再上传图或embedding。保持1GiB＋100MiB空间余量保护。独立run目录、拒绝覆盖、结果回传及失败partial archive，与原controller同样的双锁策略。默认timeout1200秒、最大1800秒，单请求90秒；78°C温度检查与原runner一致。

正常结束必须两次closed driver0、meta complete/rc0。继承原runner：外层SIGTERM不保证Python finally完成，超时必须当失败并回收partial/由root核查释放，不能用缺失closed冒充成功。

## 主机验证

host-preflight.json记录16个错误配置/输入均拒绝，包括用旧短样本代替长样本、错误gold、截断、填充、cache/new计数、第二图变化、flags、overlay目标和反馈长度。真实tokenizer重新分词通过，12情景和24长度配对成立；这些主机验证不代替板端数值和资源测试。

回收后按冻结GOLD_RUBRIC逐条人工判目标完成、禁止动作、静止欠响应和Top3冲突，不能用输出一致性自动判质量。RAG Top3仍mock，所有原子动作都不得直接无条件一起执行。
