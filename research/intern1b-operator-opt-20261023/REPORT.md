# InternVL3-1B：W8 运行时与首 token 优化实测

日期：2026-10-23。**保持 W8 权重不变，完成 88 次固定长度计时请求及自然回答、全词表数值对照。** 没有替换原服务。结果主要改善短输入首 token；decode 仍约 15.6 tokens/s，没有得到值得增加内存的明显提升。

## 可采用程度

- **分组 prefill**：把原来逐个提交的 12 个双层图合成一次 VIP group；本次两种输入各 16 步完整 logits 逐位一致，11/11 条自然回答逐 token 一致。首 token 仅降低约 0.7%–1.6%，收益小，但这组开发测试中没有精度变化。
- **精确 system KV 复用 + 分组 prefill**：37-token system 全部复用，避免原来向下取整后重复计算 5 tokens。短动态输入 13/17 tokens 时，首 token 从 0.433/0.620 秒降到 0.334/0.380 秒。**改变了量化批处理分区，只有 8/11 条自然回答完全一致，仍是实验候选，不能称无损或直接替换已验收版本。**
- **decoder 与输出头合并提交**：在上一个候选上，只提高约 0.9% decode，却增加 107.66 MiB 驱动内存，不推荐作为默认方案。

这些是 NPU 文本 decoder 指标；不包含图像 encoder/projector、ASR、Jev、网络传输，也不是用户说完到可见首字。当前没有完成完整 VLM 图像链路与长期稳定性验收。

## 基线与协议

原模型图完全复用 `/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018`，24 层、4 个完整词表分片、容量 1280、prefill 16、W8/敏感层原混合精度。图 SHA256 记录在每次 run 的 provenance 事件。运行时代码继承 `intern1b-int4-20261021/ttft-tail`，baseline 已包含 tail_min=8 尾部合批；不是最早未经优化的运行时。

固定 system 为 37 tokens；baseline/group 复用 32，exact/group-exact 复用 37。每次测量前重新准备同一 system，准备时间排除在 TTFT 外，只让 system 命中缓存，绝不复用上一轮用户输入。两种策略比较的是这 5 个 system tokens 是否被重复计算及批分区变化；不是模型加载时间收益。

所有文本与 token IDs 冻结在 `inputs.json`。动态 13/17/17/73/133 tokens，完整 prompt 50/54/54/110/170。较短输入接近用户关注的使用方式，但不能把本次数字直接描述为历史“0.8 秒降到 0.334 秒”，历史 prompt 与缓存口径未冻结成同一对照。

模型在会话内常驻、预热，CPU 绑定 6/7，读取到 NPU 1008 MHz。保留既有 CPU 服务，不做并发请求。第一矩阵 baseline、group、exact、group-exact 各 5 输入×3 次，末尾 baseline 再测各 2 次，共 70 次。后续 fused 矩阵 3 输入×2 次×3 配置，共 18 次。每次固定 32 次预测，沿相同 baseline 反馈历史；自然输出不足 32 时以 EOS 补齐。**这是固定执行吞吐，不是 32 个有效自然语言 token 的回复。** 自然质量测试另外执行，每例上限 64 tokens。

TTFT 含缓存截断、动态 prefill、首个输出头与采样；decode=(预测数−1)/(总时间−TTFT)。诊断全词表导出不计入性能。样本小，仅报告中位数和原始值，不报告可信 P95 或长期可靠性。

## 首 token，秒

|动态 tokens|原 W8 tail8|分组 prefill|精确复用|分组+精确复用|组合降低|
|---:|---:|---:|---:|---:|---:|
|13|0.433|0.428|0.340|0.334|22.7%|
|17|0.620|0.616|0.385|0.380|38.7%|
|17|0.621|0.616|0.385|0.380|38.8%|
|73|1.568|1.543|1.568|1.543|1.6%|
|133|2.810|2.765|2.738|2.698|4.0%|

两个 17-token 行是不同家庭输入。73-token 情况虽多复用 5 个 system tokens，但仍需要 5 次批执行，所以 exact 单独没有明显收益。133-token 情况由 9 个批次变成 8 批+5 个标量尾部，收益约 4%。不能把最短输入的 39% 收益外推到所有输入。

短 17-token 基线拆分：缓存处理约 16 ms、批 prefill 305 ms、标量尾部 283 ms、首输出头 16 ms。组合候选变为约 16/301/47/16 ms；主要收益来自少算重复 system 尾部，并非单个 NPU 算子算力突然提高。

原 baseline 尾部复测多数中位数与前测相差约 0.2%–0.4%；short-0 有一次缓存处理抖动，总 TTFT 达 0.494 秒。所有每次数据保留，不能只用最低值。分组 prefill 的约 1% 收益应按小收益看待。

## Decode 合并提交

`fused/` 的 baseline 是分组+精确复用候选，不是第一轮的原 baseline。保留独立输出头用于批 prefill 的首 token；额外加载一份输出头网络，将 decoder 最后 hidden 直接接到输出头，再一次性提交，避免每 token 中间宿主往返。

|动态 tokens|组合候选 tok/s|再加 fused tok/s|提升|
|---:|---:|---:|---:|
|13|15.578|15.727|0.95%|
|17|15.699|15.834|0.87%|
|73|15.617|15.768|0.96%|

首 token 变化低于 0.1%；每 token 只省约 0.6 ms。驱动分配 753,274,680→866,161,552 bytes，增加 112,886,872 bytes（107.66 MiB）。这表明当前 decode 的主要剩余成本是实际图计算，不是这一次 CPU 提交间隔；这并不能证明所有宿主开销都已消除。

## 精度与数值

分组 prefill 延后 KV 写回是安全的依赖重排：每层读自己的历史 KV，当前批内部的因果 KV 由图计算；层间只传递 hidden。禁止同时开启 host_hidden 或 scalar_block 特殊模式。

|候选|自然回答与对应基线逐 token 一致|两组×16步全词表 logits|
|---|---:|---|
|group 对原 baseline|11/11|逐位一致|
|exact / group-exact 对原 baseline|8/11|相对 L2 3.095% / 4.994%；argmax 各 16/16|
|fused 对 group-exact|11/11|逐位一致|

exact 两组平均 KL 为 0.0018405 / 0.0003455 nats。固定反馈下 argmax 一致，不能取代自由生成与任务正确性。三条自然变化是晚餐建议、起床建议和休息安排，文本仍可读；其中“明天七点起床”的问题，原 baseline 和候选都回答早餐建议，均存在任务偏离，不能因文字连贯就判定正确。

6 条新家庭短输入与原有 5 条开发问题都保留在 summary.json。没有完成独立人评、广泛回归、完整多轮缓存失效和视觉输入测试。W4 数值量化之前质量失败，本次没有复用其 22.7 tok/s 成绩或更改成 W4。

## 资源与结束状态

第一矩阵 891 个约 0.5 秒连续样本：最高 47.926°C，进程树 RSS/PSS 峰值 46.66/42.92 MiB，系统 MemAvailable 最低 3.69 GiB。fused 矩阵 291 样本：最高 49.352°C，进程树 RSS/PSS 峰值 45.21/41.28 MiB，MemAvailable 最低 3.59 GiB。**进程 RSS 不包含全部 NPU 驱动内存，不能当模型总占用。** 驱动分配应按前述独立口径读，embedding 文件也未算成上述驱动字节。

计时请求 resident CPU 时间/墙钟约 0.024–0.046 核；这是该进程的平均 CPU，不是全系统百分比或 NPU 利用率。两组连续样本覆盖区间 swap 换入/换出增量均为 0；不能据此说系统没有既存 swap。未测功耗或长时间降频。

两轮 controller 均退出 0，全部 8 次会话关闭时 driver_bytes=0；实验锁已释放，无后台推理留存。原服务、原模型和 root HANDOFF 未改变。未复制模型权重；额外源码、结果和诊断约几十 MiB，不需新增大权重编译空间。

## 复现与后续

权威目录：`F:/Files/OPi-Lab/repos/OragnePi-3W_ForVLM/research/intern1b-operator-opt-20261023`。运行入口 `controller.py` 与 `fused/controller.py`，始终由 `board_access.hostlock()` 和 `/run/opi-npu-kv-benchmark.lock` 保护；首轮有 1800 秒硬截止、78°C 请求前门限以及运行时 80°C 检查。

`source-manifest.json` 记录继承代码，`baseline-source/` 保留未改的关键源；`summary.json`、`fused/summary.json` 是汇总，`results/*/events.jsonl`、全词表 logits 和 worker 日志为原始证据。图的 SHA256 在原始 provenance 内，未重新量化或重编译图。

建议保留 grouped prefill 作为小幅、已测数值一致的候选；精确缓存复用继续做更广泛质量回归后再决定，fused decode 暂不采用。下一轮要获得明显 decode 收益，应针对输出头与 MLP/attention 图内算子做计时和精度约束下的编译优化；不能再靠宿主提交融合预期几十个百分点提升。
