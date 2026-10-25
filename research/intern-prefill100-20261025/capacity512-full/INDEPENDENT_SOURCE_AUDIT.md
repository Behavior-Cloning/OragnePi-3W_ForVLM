# C512 原精度 M16/M32/M16-repeat 独立源码审计

**结论：当前冻结源可进入有界板端实验；未发现启动阻断。此结论不代表端侧输出、质量、速度或100ms目标合格。**

审计只读取 Windows/WSL 文件并写本目录审计；未连接板卡、未启动控制器。

## 实际复核

- 重算48个NB的SHA与大小，总计1,159,978,624字节；每个产物的实际导出payload和量化文件均与其原精度源逐SHA相同，图JSON只有已声明容量/批量几何变更。
- 独立重算48份SDK实际端口JSON：11输入+5输出，仅C1280→C512及M16→M32几何改变，dtype与量化接口保留。未重新调用SDK。
- L2的M16/M8选用capacity512-original-l2；M32选用original32/l2。排除了range16目录中的I16 L2产物；其余层虽位于该命名目录，实际源及payload/quant已核实为原版。
- 同一ARM二进制 bf19922fb62bdde70a15ccde742abf9b4ffa6e5b116e73929fa7310442dc1e11；构建记录11个源文件SHA全部符合。配置无FIRST原图/I16候选开关，主/尾均为正常完整12网络group。
- inputs.json与上一冻结fixture完全逐字节相等；使用真实Intern tokenizer逐条验证24性能+48自然输入。静态64/128=正文59/123+5模板，动态正文64/128/192/256+8模板，实际新增72/136/200/264；最大KV使用423≤512。
- 每请求reset并只计算system，下一请求仅复用静态prefix；runtime对齐为M16或M32，64/128均整除。runner硬断言实际reuse/new。
- runtime优先完整主bank，最后恰8行使用独立M8图；每条完整请求断言true_tail8_calls=1、tail_rows=8、scalar_prompt_tokens=0、batch_calls=body/M+1。
- 性能32步固定反馈、诊断关闭，decode计31个间隔；自然最多32步且EOS可提前结束，不能把固定反馈输出作为任务正确性。8诊断/版各4步全词表，单独转储且不计速度。
- 总计72性能、144自然、24诊断请求；各版24/48/8。M16→M32→M16-repeat，独立worker/driver释放校验。
- Controller默认dry，execute前冻结配置；hostlock+board flock、唯一目录、传输逐SHA、70链接目标及48实际文件校验、超时和失败partial回收齐备。传输额外预留768MiB+150MiB，无重复板端tar。

## 边界与待实测

- Source review permits a bounded experiment only; no runtime, numeric, quality, speed or 100 ms qualification is implied.
- M32 also prepares system with M32; system input tokens are identical but numerical KV may differ. This is the M32-bank scheme versus M16-bank scheme, not dynamic-only batch isolation.
- System reset/recompute and cold load are excluded from hot TTFT. TTFT includes request setup, actual complete new prefill and first head/argmax; audio utterance completion-to-action is not measured here.
- 72 performance = 3 variants x 8 cells x 3 cases x 1 repeat. Per-cell n=3 cannot establish stable P95 or long-term stability. 20 boundary fixtures are not run; full64-256 range100ms objective remains unverified.
- 48 quality = 24 development plus24 previously frozen household cases; both sets have now been seen in prior runs, not a fresh unseen holdout. Exact parity is not task correctness.
- Original precision means same existing mixed quantizers and exported constants, not all-FP32. Shape/capacity changes can alter numerical outputs; inspect full logits and natural semantics before adoption.
- Synthetic prerecorded speech and RAG stay resident; fixture prompts are not produced by live ASR. No image encoder, real microphone or physical motor. RAG Top3 conflicts remain and no action promotion is authorized.
- Shared head/embedding and frontend dependencies exist on board and are SHA checked before load; this audit did not access or verify board availability. Actual disk guard is reevaluated under both locks.
- On SSH/timeout observation failure no automatic rerun; partial archive recovery requires reacquiring both locks and can remain pending. Board timeout/process release still needs observed driver0 proof.

## 当前源SHA

|文件|SHA256|
|---|---|
|board_fulli16.py|beec047dceb67710c166c0e9d3d22f71bdd9c3f02d5a1875b53bc9cf206f931f|
|controller_fulli16.py|467a272097de603eea460770972233c2548598cecdba8a21670547953816a1aa|
|contract.py|0ee0160287bf99a65ba3bfc77038beab926aa2a9a29e9ed73cb2f6e74e78aadd|
|config.fulli16.json|2294bfa53da6fa3947ed81cb581c0dc26ea6419fa51a2d64583991d13ccd6c10|
|artifacts.json|2ffeb37c27866361b3802df0c7873e66789cce0930d9d494421a5e315e33518f|
|inputs.json|d314a51d07b4a39ca9533cb805540ac90ce132819355c7ffe67aa4591d8cd49c|
|resident_worker_v1.py|2c110a6821ce9ee21aa2c07b062f27a49e18a9289e0fd8aea12927f80448321d|
|context_rag.py|81afe39a529dd5de5ca7b3a0ca21c870fe42777684cdebb1daec3dd7e6f4998e|
|analyze_fulli16.py|edec8a4a7f7628449a3baa9f4c5b3bfc8630dc7492b5ff8e7347bfeffa6035b5|
|README.md|8af6344b105362e5d75b6df1e6dad023ae4fd4d5fa8ec0d9f665dad667c67d25|

结构化逐产物证据：INDEPENDENT_ARTIFACT_CHECK.json；真实分词验证：INDEPENDENT_TOKEN_CHECK.json。
