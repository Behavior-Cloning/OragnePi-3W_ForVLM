# Intern NPU 同输入 prefill / decode 候选验证（2026-10-25）

权威工作目录在 Windows F / WSL。所有板端操作必须经过 board_access.hostlock 与 /run/opi-npu-kv-benchmark.lock；现有 controller 运行期间只读 WSL 状态，不另起板端请求。

## 最新状态

- 第一轮完整对照 `candidate-matrix-r3` 已完成，见 [报告](analysis/candidate-matrix-r3/REPORT.md) 和独立审计。C256+greedy四格TTFT P50约0.58/0.95/0.58/0.98秒，decode约15.2–15.4 token/s；正文64尚有超过1秒样本。
- `combined-matrix-r1` 的C256+greedy+真8尾批速度更快且小集合输出一致，但关闭时堆损坏，整轮失败。详见 [失败审计](analysis/combined-matrix-r1/INDEPENDENT_FAILED_RUN_AUDIT.md)。禁止把失败前计时当作已合格版本。
- 原ASan全顺序和原O2详细关闭日志全顺序都未复现，根因仍未定位。`runtime-combined-v2` 仅保守调整资源释放顺序，3次加载/关闭已过；完整 `combined-matrix-r2` 已rc0/meta complete，72性能/36自然/36cost/15诊断、3关闭driver0，见 [最终报告](analysis/combined-matrix-r2/REPORT.md)。正文32/64的TTFT P50约0.49/0.87–0.89秒，24次性能最大0.916秒；仍仅有限复测，旧close根因未知。旧代码、失败日志与新候选分别保留。
- 所有图与候选均未替换现役Agent默认配置。RAG Top3语义冲突仍未解决。

## 输入与计时契约

| 固定前缀 | system 正文 | system 模板 | 动态正文 | 动态模板 | 总输入 | reused | new prefill |
|---:|---:|---:|---:|---:|---:|---:|---:|
|64|59|5|32|8|104|64|40|
|64|59|5|64|8|136|64|72|
|128|123|5|32|8|168|128|40|
|128|123|5|64|8|200|128|72|

同格完整 token IDs 一致；每请求重置并只准备 system KV，不借用前次动态缓存。长度精确对齐的标点填充仅用于性能 fixture。语音前处理常驻、录音实时回放；RAG 模型常驻，仅自然质量阶段查询。Intern 1B 仅 NPU 文本路径，无视觉、真实麦克风和电机。

performance 每格3场景×repeats，输出32步，固定反馈 token 100662，计时不导出 logits；不能当作32-token自然回复。quality 使用12个自然请求，最多32输出，按真实EOS停止。diagnostic 四种长度各4步固定反馈，输出完整 float32 logits 供哈希与逐值比较。TTFT是已准备system KV后的模型调用到首token，不含语音判停、压缩、排队或RAG执行。

## 本轮候选

- mask-base / mask-repeat：现役容量1280，tail8+group+mask，前后夹测。
- context256：相同量化/常量，decoder与prefill图容量256，复用head与embedding；当前最大prompt200，32输出占用231个KV位置（末token尚未反馈）。不能用于原图像256 token追加场景，也不能承诺最大格64输出。
- tail8-c1280：额外12个8-token图只处理恰好8-token的末批；观察额外driver内存，不能把1280尾图与256主图混合。
- greedy-c1280 / context256-greedy：INT16输出头单遍argmax，保留最早同分下标语义；完整模型parity通过前仅为候选。

## 已完成证据与运行入口

profile-smoke已完成；analysis/profile-smoke/REPORT_NEXT.md记载97–98% TTFT来自prefill。单块探针在results/graph-probes，详细分析见邻目录intern-context256-20261025和intern-nextopt-20261025。单块探针未并行语音负载，不等于全模型收益。

deploy_candidates.py校验图SHA并执行ARM greedy自测，输出verified-graphs.json与config.candidates.json。部署ready后单次启动：

```sh
python3 -u controller_next.py NEW_UNIQUE_RUN --config config.candidates.json --repeats 2 --timeout 2100
```

控制器使用不可复用run名，记录运行源/配置并回收证据。完成后：

```sh
python3 analyze_next.py results/NEW_UNIQUE_RUN --out analysis/NEW_UNIQUE_RUN
```

最终验收需检查returncode/meta、每格token与输出数、完整logits/自然输出parity、实际尾批调用、前后基线、温度/CPU/PSS/swap/音频滞后及driver close=0。PSS为被采样进程树，不等于整机独立模型占用；新图需实测后决定是否启用。原RAG语义兼容性问题未解决，本任务不放开Top3动作执行。

原始模型、ELF、object及大f32保留Windows/板端磁盘，Git忽略；脚本、配置、哈希、汇总和报告版本化。
