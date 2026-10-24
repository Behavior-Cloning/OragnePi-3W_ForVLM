# 32–64 token TTFT 候选：运行时核对与比较约束

2026-10-25。只读核对 Windows 权威源码；本记录未连接板卡，未改变当前配对测试、服务或控制器。目标为真实中文摘要输入的首 token 响应，不把固定填充的吞吐测试当任务质量证明。

## 已核对的路径与开关

- 当前 harness.py / harness32.py：`/opt/opi-vlm/resident-agent-20261019/resident`，CPU affinity 4,5，没有显式尾部合批环境变量。
- 已有优化程序：`/opt/opi-vlm/intern1b-operator-opt-20261023/resident`。
- 共用原图：`/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018`；参数 `24 1280 151674 <图目录>/embeddings.f32 16 151645`。24 层 Intern 文本 decoder，16 行 prefill；不更换量化权重。
- 新程序 `OPI_TTFT_TAIL_MIN=8`：尾部剩余 8–15 个 prompt tokens 时以 16 行图执行，填充未来行但只写回有效行。少于 8 行仍走标量。
- `OPI_GROUP_PREFILL=1`：将 12 个双层 prefill 图一次提交，隐藏状态设备侧连接，最后逐层回写 KV。不能与 OPI_PREFILL_HOST_HIDDEN 或非负 OPI_PREFILL_SCALAR_BLOCK 一起开启。
- `OPI_EXACT_PREFIX=1`：LCP 缓存复用不再按 16 向下取整。它改变批处理分区，不能先验认为量化结果无损。
- **group/exact 按 getenv 是否存在判断。设为字符串 0 也开启；关闭时必须移除环境变量。** 当前ResidentWorker已经清除所有继承的OPI_*，矩阵再按组设置值；新启动器也须保留此行为。
- `OPI_SESSION_FD` 由 ResidentWorker 建立私有 JSON pipe；不能直接执行 binary 后期待 stdout 是协议输出。
- 编译参数见 `intern1b-operator-opt-20261023/build-command.txt`，含 DIM896/KV2/HD64/FUSED_LAYERS2/KEY_CACHE_TRANSPOSED0/HEAD_SHARDS4/HEAD_CHUNK49152；不要套用 Smol 或 Llama 定义。

## 为什么 64 token 不能先承诺 <1秒

历史报告的每次 16 行 prefill 约 0.30–0.31 秒，短标量 prompt token 约 47 毫秒，首输出头约 16 毫秒，缓存截断约 16 毫秒。此处只是旧测的估算，不是新测量：

|真正新计算 tokens|整批数|理想无尾部时的 TTFT 量级|
|---:|---:|---:|
|32|2|约 0.63 秒|
|48|3|约 0.93 秒，余量很小|
|64|4|约 1.23 秒|

正文 32/48/64 不等于真正新计算 tokens。当前固定前缀 86 tokens，默认缓存向下取整到 80，额外 6 个固定 tokens 也要重算，user/assistant 模板另计。上一轮动态文本前缀偶然命中又可能减少计算，必须记录真实 reused/new_tokens，不能把这种命中包装成任意 64-token 输入性能。

32 正文借助 tail8/固定前缀布局可能达到 <1秒，需要实测；64 正文通常还要 4–5 批。group 旧测仅节省 0.7%–1.6%，远不足以消除这部分差距。若新数据依旧如此，64 真正新输入需要新的 prefill 图内编译/算子优化或不同批大小图，不能靠改开关许诺目标。降低频率外的硬件超频不在此计划内。

## 已有精度证据与风险

2026-10-23 REPORT.md：group 对同 tail8 基线 11/11 自然回答逐 token 相同，两组各16步全词表 logits 逐位相同。exact/group-exact 为 8/11 自然回答一致；固定反馈 logits 相对 L2 约3.095%/4.994%，argmax 各16/16。后者只是小样本开发证据，不能保证本次动作规划正确。

该历史 baseline 已含 tail8，不是当前 harness 的旧 resident。fused decoder+head 额外约107.66MiB驱动内存只提升约0.9% decode，首 token 几乎不变，不优先用于此目标。

## 推荐对照顺序

1. 在既有 pair300-32 完成、证据复制且双锁释放后才开始，复用原图；当前板端脚本不原地覆盖。
2. old → tail8 → tail8+group → tail8+group+exact → old-repeat。同CPU亲和、同驻留语音/RAG负载、同温控；优先同一条件配对，不把历史 CPU6,7 和现 CPU4,5 的结果混算。
3. 固定中文正文 32/48/64 token，每档至少3条不同场景与多次重复；以 Intern tokenizer 计数。为凑精确长度追加标点的样本只能用于计算成本，另留无填充的真实家庭语义输入检查质量。
4. 每次计时前 reset 并准备同一固定前缀，排除偶然动态 LCP 复用；prefix准备时间单独报，不纳入常驻推理 TTFT。该受控成本测量之后还要真实事件连续链路复测，报告从事件触发到首 token 的排队和音频竞争。
5. 记录正文长度、system长度、模板额外长度、总prompt、reused/new_tokens、batch_calls/tail_rows/scalar_prompt_tokens、setup/prefill/scalar/head分段、wall TTFT与事件TTFT。不能只报告最快一次，至少逐档P50/P95及重复数；少量重复的P95不宣称长期保证。
6. 性能计时可 max_new=1；自然动作质量另测 max_new=32，比较同输入逐 token 结果、否定/取消/不移动等约束及RAG动作兼容性。固定反馈数值诊断与自然输出分开，不将token一致等同任务正确。
7. 每组记录程序与图SHA、亲和、频率、温度、内存、swap；收尾driver_bytes=0。下游仍mock，不接真实电机。已知“弯腰→抬头”RAG误配不能因为TTFT变快而忽略。

## 本次核对的 Windows 源码 SHA256

- `resident_session.c`: `11c6316278f66b02fbb967ca2f793b9cecf8995f3a4ee4366e07d5e98605cba3`
- `batch_prefill.h`: `f80acdcb0c4574546bb9716cde040f9c069d2e4f7245263bb950c6d113aee094`
- `board_run.py`: `72ccc1dd09b1f47946cc8ae26f96dad9961b735c1bb0aee5b55f3f0f1ca5240f`
- `REPORT.md`: `835a281d927e58972d85dde43cf163de1d4ed9831e9383b11f0b79524b60a89a`
- `build-command.txt`: `8fd8e11afc203b1fec82c684e71d6ffd007ee4b812134458862f14357dbdeec0`


## 环境变量疑点复核

2026-10-25只读审计撤回继承变量泄漏疑点：仓库中resident_worker.py在Popen之前过滤所有OPI_*，随后矩阵通过env命令逐组显式注入。因此当前Worker启动路径无此污染证据；新启动器应保留过滤行为。group/exact仍按变量存在判断，不能用值0关闭。
