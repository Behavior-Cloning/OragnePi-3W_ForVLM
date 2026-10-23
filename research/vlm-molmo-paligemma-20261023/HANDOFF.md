# 2026-10-23 模型部署与存储清理交接（进行中）

## 最新用户方向与下一执行目标

用户已明确先做纯文本、中文，并要求中文Llama1B NPU推理实测。当前目标是完整16层+全词表输出头的中文生成，不以首层代替整模。llama32-full-npu-20261023由runtime agent准备；先scalar prefill+decode均在NPU，CPU仅embedding/采样/调度。每层校准独立，不能复用layer0 scales。完整编译必须等本轮小测试全部结束且WSL压缩完成后启动。

新追加Llama全FP16首层6输入：hidden相对L2 0.049%–1.10%，中位48.13ms；源仍Q4 GGUF。当前正在实测attention FP16 + MLP W8候选及全FP16对照（6旧+9中文留出），候选NBG66,878,544B。根据实测选精度，未有16层NPU速度结论。

Pali当前已完成四图文CPU请求及三中文text-only CPU请求。中文纯text热TTFT6.088/6.474s、decode约1.5tok/s，常驻、无图，仍无跨轮system KV；中文质量有复述/英语混杂。Pali首层真实NPU FP16/W818输入完成：61.52/11.86ms，最大hidden相对误差0.576%/0.852%。18次仅13唯一输入，属于3中文句子同校准集合。下一步完整Pali NPU暂后置于中文Llama整模目标。

Pali旧图文进程CPU采集因线程名带空格读取/proc/pid/stat出错，作废；整机CPU仍有效。已修复解析器并在后来纯中文text测试记录了原stat/有效过程CPU（热1.988核、全8核24.85%）。

## 权威位置与操作边界

Windows F:/Files/OPi-Lab/repos/OragnePi-3W_ForVLM；分支 codex/wsl-intern1b-reproduction。Mac仅控制和报告副本。板卡所有访问走Windows/WSL，使用board_access.hostlock()与/run/opi-npu-kv-benchmark.lock，任何时刻只允许一个实验所有者。

## 当前已完成

- Intern W8运行时优化：88次固定执行计时、11题自然生成、全词表数值对照；group prefill约1%首token改善且本组数值一致。exact system KV可改善最短输入22.7–38.8%，但只8/11回答逐token一致，仍为候选。decode融合约0.9%收益却多107.66MiB驱动，不默认采用。已提交6c55831。
- 中文FuseChat-Llama3.2-1B：CPU A76双核Q4_K_M常驻；6个新问题、每次仅26-token固定prefix缓存，TTFT中位0.638s/decode12.47tok/s、RSS约1.60GiB、最高58.09°C。小样本中文表现有明显缺陷，未获质量验收。整题缓存0.08s结果被排除。
- Llama真实GGUF解量化→NPU层0试验完成，W8与MLP FP16的hidden误差都大，FP16例外更慢；不是完整NPU部署。已提交6c55831。
- MolmoE官方元数据/源码、随机路由/静态expert/dynamic NonZero编译探针完成；专家W8权重 alone 6.00GiB超过5853MiB总RAM。原28.89GB文件未下载；没有整模性能结果。
- PaliGemma2原BF16下载SHA、ARM/x86构建、转换修复、三问BF16/SFP逐token回归完成，单进程每问重跑vision；板端测试仍在进行。
- 172个可再生文件已删除：直接F 1,421,901,036B；WSL内部6,958,298,487B。保留源权重/NBG/重建证据。
- TF原图LZX压缩失败但全SHA复核一致，原图保留；slim20扫描证实不含官方OS安装镜像，仅代码不能替代恢复材料。

## 尚待本轮完成

- [已完成] Pali板端图文四请求与另三纯中文text请求，报告分开计时；一词回答不算稳态decode。
- [已完成] Pali真BF16层0 NPU FP16/W8，18冻结输入板端误差；仅单层结果不外推整模速度。
- 全部WSL/板端实验退出并释放锁后，运行maintenance/compact_wsl.ps1中的原生命令压缩OPiUbuntu VHDX；记录前后F空闲、VHDX长度、重启验证。不能在传输/编译/推理运行时压缩。
- 独立审计、最终REPORT/README/Mac副本、仅本轮目录Git提交。

## 证据入口

- REPORT.md、AUDIT.md、paligemma/REPORT.md、paligemma/REBUILD.md。
- ../intern1b-operator-opt-20261023/REPORT.md、../llama32-compat-20261023/REPORT.md。
- ../paligemma2-npu-compat-20261023/：原始BF16参考与FP16/W8编译探针。
- molmo/REPORT.md、memory-lower-bound.json。
- maintenance/cleanup-manifest.jsonl、maintenance/BACKUP_DECISION.md、maintenance/tf-compression.json。

## 后续未验收边界

所有新模型的完整语音agent并发、长时热稳定性、家庭图像/中文质量、64/128动态输入公平对照未在本轮补齐。Pali完整NPU、Molmo静态expert混合调度、Llama整模NPU仍不能称已完成。已有Intern主服务未被替换。
