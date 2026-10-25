# 最终候选运行时兼容性审计（2026-10-25，矩阵运行前）

只读审计：未访问板卡，未修改冻结运行时、二进制、部署脚本或正在运行的结果。完整矩阵结果尚未回传，因此以下是实现与实验契约审计，不是整模型通过声明。

## 结论

两个候选的请求协议与 `ResidentWorker` / `board_benchmark.py` 兼容；精确argmax、双mask bank回退和capacity边界没有发现当前配置下的源码阻断问题。原12条自然quality均不会调用真8图；已通知主任务并在正式矩阵启动前补齐4条 `quality_cost`，详见末尾复核更新。

## 必须区分的验证覆盖

性能输入固定system64/128（正文59/123+5模板），动态正文32/64另8模板，所以新增40/72tokens。只有system缓存，bulk16后均恰好剩8，`tail8-c1280` 每个性能请求、每个四步logits诊断必须 `true_tail8_calls==1`、`tail_rows==8`、无scalar prompt。system独立准备64/128恰好16整除，真8调用应为0。

原12条自然quality新增token为23、22、29、12、15、16，各在两个system下重复，余数均不是8。因此该阶段的 `true_tail8_calls` 应全为0；它能验证候选关闭分支兼容性，却不能证明真8图上的自然生成质量。建议在原quality之外，对冻结performance的4个cell首例（或全部12例）增加无feedback自然生成，并断言真8确实调用一次。保留原输入及成本补齐标记，不能把带padding输入称作新增独立家庭语料。

## 环境与部署

`ResidentWorker` 清除继承的全部 `OPI_` 后，单独注入session FD与dump路径。benchmark把每个variant的环境构造成命令前缀 `env KEY=VALUE ...`，因此显式 `OPI_PREFILL8_DIR`、`OPI_GREEDY_I16_SINGLE_PASS`、mask/group等不会被清除；variant间也不会泄漏。session FD/dump目录禁止由variant覆盖，避免协议路径串扰。flags按是否存在启用，字符串`0`依旧启用；当前部署配置均为明确`1`或省略。

当前六variant定义：mask-base/c1280、context256、tail8/c1280、greedy/c1280、context256+greedy、mask-repeat；三种图来源独立，tail8并未与256图混合。所有计时variant的 `speed_profileflag=none`，没有设置 `OPI_PROFILE_STAGES` 或 `OPI_PROFILE_NPU`。controller复制需要的candidate，配置重写到本run私有绝对路径，运行前按显式清单逐文件SHA验证；tail8 variant清单包含额外12图。

SCP源candidate在WSL F挂载上mode0777，正常scp创建会带执行位（受远端umask约束）；但当前controller只明确chmod mask/profile，tail8/greedy依赖复制行为。建议主任务统一chmod实际使用的全部binary。此项已通知，属于部署稳健性补齐，尚未观察到真实执行失败。

未来配置校验可改为拒绝key存在的 `OPI_PROFILE_NPU`，而不只检查value真值，以匹配C的存在即启用规则；当前配置没有该key，故本次不受影响。

## 真8运行时状态与容量

candidate只在batch参数16、显式tail目录有效、未启scalar-block实验时加载额外8图。load_prefill对每个共享KV端口核对count、dims、shape、dtype、quant和DFP scale，形状不匹配会失败，不可能悄悄把c1280的tail图接到c256 decoder。

每张tail图拥有独立mask/hidden，只有已核对的decoder KV被alias。reset与truncate同时恢复16和8两bank mask与previous_position。即使bulk16已经更新KV，尚未运行过的tail8仍从同一keep到当前pos打开合法历史列；新位置的KV来自同一decoder共享缓冲。尾图只输出并发布8个真实token，不存在额外padding KV。关闭先销毁两个prefill组及其独立buffer，再销毁共享KV所属decoder；需实际closed.driver_bytes==0确认SDK释放。

所有性能/quality请求的最大实际KV长度为 `200+32-1=231`，256容量合法；fixture/controller/C三层均检查，未截断。context256+greedy不依赖tail图；若未来组合，必须另编c256真8并通过上述IO契约，不得复用c1280文件。

已有主机测试直接提取candidate实际reset/truncate函数，容量1280/256，有/无tail，反复128→64与200→128，以及故意污染两bank后回退，plain/template/validate各1,920张量检查、ASan/UBSan均通过。该测试仍不代替板端自然生成和完整缓存验证。

## 单遍最早最大值正确性

原NEON第一遍查最大值后，第二遍逐元素查最早位置。新候选对8个lane分别保留最大值与首次索引：只有严格大于时更新，所以每lane并列保留旧位置；再从8个lane选择最大值及最小索引。余下不足8项只在严格大于时更新，因其全局位置晚于已处理向量，不会破坏最早并列规则。全INT16_MIN、负数和跨块相等均覆盖。

16-bit索引仅用于count<=65535；现4head为49152/49152/49152/4218，合法。超65535回原函数，避免位置回绕。没有改模型logits、EOS、token反馈或head scale。各head最大值仍由原get按各自DFP scale还原后比较，head按词表顺序且严格`>`更新，跨head并列仍取最早token。

主机同一GCC vector源221,212例ASan/UBSan验证通过，包含全部INT16值、4值字母表长度1–8穷举、真实shard长度、边界65535/超界、2,000随机及并列情况。ARM独立自检也随部署文件提供，实际结果须由root的 `board_verify_candidates.py` 回收，不以主机测试冒充ARM执行。

## 计时公平性与报告边界

- 每个测量请求先reset并准备固定system，再发送完整prompt，断言实际reuse64/128及new40/72；没有动态跨请求cache。system准备单列在prefix_prepare_metrics，**不计入warm-system TTFT**。
- 固定32反馈为吞吐组，即使预测EOS也继续；自然输出以EOS或32上限结束；全词表4步dump另组且timing_eligible=False。不可混合汇总。
- 每variant都有独立加载与预热；模型冷加载不在测量TTFT内。固定顺序存在时间/温度混杂，mask-repeat有助于检查漂移，但不能替代随机交错/长期稳定性试验。
- 音频前端常驻并循环预录WAV，RAG常驻；RAG查询只发生在quality完成后。这里TTFT是收到固定token输入后的语言模型首token，不是“真实用户说完→首字”完整事件链指标。资源应按variant分组，记录音频lag、频率与温度。
- tail8增加驻留权重/driver内存，是速度与资源折中；不能只比延迟而省略内存差额。
- greedy候选还带关闭的profile分支，mask-base是另一冻结binary。可报告“该候选运行时相对现役基线”的实测变化；若要把收益精确归因于单遍argmax本身，最好同 `resident-greedy` binary不开/开flag进行A/B。profile-smoke已提供profile骨架兼容性证据，但不等于所有编译布局零开销。
- 一个16行/8行首块通过，不足以宣称24层中文生成精度无损；整栈必须重新比固定输出、全词表logits、实际调用真8的自然输出与driver释放。

## 本次审计读取的源码快照

```json
{
  "intern-nextopt-20261025/runtime-tail8/resident_session.c": "59d420aa8433b2c6d67bb74350ed8443724e3cc4d91173fe6f045fbfbb41c610",
  "intern-nextopt-20261025/runtime-tail8/batch_prefill.h": "f80acdcb0c4574546bb9716cde040f9c069d2e4f7245263bb950c6d113aee094",
  "intern-nextopt-20261025/runtime-greedy/resident_session.c": "1eb4daae30f8fa04064deb8e8e45ab1e71e4f9dd0951b8a2997b16b6b6de2b5b",
  "intern-nextopt-20261025/runtime-greedy/greedy_single_pass.h": "6909b4b67ac0b86b2eb49c107cd03364c0718b6413d214412ecb23726c242e16",
  "intern-nextbench-20261025/board_benchmark.py": "3ac4634b24554d07d1c1478c0a9cacec3ebc2c98fb6e67efffb601dc23c868fa",
  "intern-nextbench-20261025/controller_next.py": "1571c1e2f9aa24809dae71c38606e81e8ee80aa6f78e1e43a85e1b62276ea85a",
  "intern-nextbench-20261025/resident_worker.py": "2c110a6821ce9ee21aa2c07b062f27a49e18a9289e0fd8aea12927f80448321d",
  "intern-nextbench-20261025/deploy_candidates.py": "30f56a1a3ce766e6256163183f1137c34ffbc469bb312d2b2246e895c59a13ae"
}
```

若主任务随后修正benchmark/controller或增加natural-aligned阶段，应以实际回传provenance SHA更新最终执行覆盖；此报告保留审计时点，不改写已发生的证据。

## 启动前复核更新

主任务已修正部署执行位：controller对全部needed binary显式chmod。board_benchmark在原12条quality后增加4个已冻结成本输入的无feedback自然生成（quality_cost、cost_padding=True）；tail8变体断言true_tail8_calls==1，保留这些输入并非新增独立质量样本的标记。部署回传ARM自检221,212例、0不一致。上述两项覆盖缺口已补齐，运行时源/二进制仍保持冻结。

修订后的脚本SHA：

```json
{
  "board_benchmark.py": "3ac4634b24554d07d1c1478c0a9cacec3ebc2c98fb6e67efffb601dc23c868fa",
  "controller_next.py": "1571c1e2f9aa24809dae71c38606e81e8ee80aa6f78e1e43a85e1b62276ea85a"
}
```
