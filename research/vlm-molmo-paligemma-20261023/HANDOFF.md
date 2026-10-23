# 2026-10-23 模型部署与清理交接（进行中）

## 最新方向
用户先要求中文纯文本，完整Llama 1B在NPU实测，并与Intern 1B在近似同等量化条件比较；最新明确Q4也允许比较。保留Q4源功能路线，补Q8源配对。源GGUF量化与实际NPU W8/FP16混合必须区分，不能把Q4源解量化再W8编译称作NPU INT4。

## 当前执行
- intern_operator_opt独占板卡；root及其他agent不可直接访问板卡。所有板卡操作经Windows/WSL、board_access.hostlock与板端flock。
- research/llama32-full-npu-20261023：Q4源16层attention FP16+MLP W8和4片FP16完整输出头逐层编译中，WSL PID748，日志build-q4.log。每层自己的真实校准，不复用layer0尺度。scalar prefill/decode均NPU；CPU embedding lookup、位置参数、KV复制、调度与greedy选词单列。
- Q8源已经下载校验：paligemma/llama-zh/q8-source/FuseChat-Llama-3.2-1B-Instruct-Q8_0.gguf，1,321,079,360B，SHA c7db2b30d2b176998db90304d1ac6dd259379bd75f9ea8b6a094be88ea47b1b4；同revision和tokenizer。models agent准备独立FP32自然生成参考；quant_comparison_audit只读核对精度清单/公平测试协议。
- 全16层NPU尚未实际跑通，TTFT/decode不填写猜测值。参考与自由生成、完整词表logits同反馈对照、缓存仅system、所有EOS、真实资源峰值/释放都需完成。
- WSL16GiB；新keeper是sleep21600。F物理空闲已compact到35GB，Q8下载后约31.46GB；所有任务保留至少8GiB，不在编译/传输时压缩VHDX。

## 已完成与证据
- Intern W8：88计时、11自然回答、完整logits。group prefill约1%无损收益；exact prefix短输入0.433→0.334s、0.620→0.380s，8/11同token，仅候选。decode合并约0.9%收益、多107.66MiB驱动，不默认启用。../intern1b-operator-opt-20261023/REPORT.md。
- 中文Llama CPU Q4：6新问题固定prefix26，TTFT P50 0.638s/decode12.47tok/s；旧整题缓存0.08s组排除。新NPU模板和token数不同，不能直接拿旧CPU数声称公平比较。
- Llama NPU层0：旧W8误差6–35%；全FP16 15状态中位48.299ms，9中文误差0.0986–0.2527%；attention FP16+MLP W8中位16.248ms，中文误差2.141–3.698%。仅首层，不代表整模中文质量。../llama32-compat-20261023/REPORT.md。
- Pali CPU图文四请求完成，后三次TTFT52.140/48.467/48.236s；中文无图三请求7.154/6.088/6.474s、decode约1.5tok/s。旧图文进程CPU解析错误作废，新文本81条原stat校验通过，热1.988核。中文输出SFP跨平台3/3同token、相对BF16为2/3，质量不合格。paligemma/REPORT.md。
- Pali真BF16层0 NPU FP16/W8，18状态/13唯一输入；61.515/11.861ms、max L2 0.576%/0.852%；非整模、非INT4。双向prefix不能直接套Intern跨轮system KV语义。完整Pali NPU暂后置。../paligemma2-npu-compat-20261023/。
- MolmoE实际7.2B总/1.5B激活，专家W8 alone 6GiB超板RAM；动态NonZero不支持，静态小探针可编译。未下载28.89GB原权重、无整模速度。molmo/REPORT.md。
- 清理172可重生成文件：直接F 1,421,901,036B；WSL逻辑6,958,298,487B。源权重/NBG/重建证据保留。maintenance/cleanup-manifest.jsonl。
- WSL原生compact与重启已成功：VHDX 88,689,606,656→68,562,190,336B；F物理实际增加20,127,416,320B（含历史空闲块，不全归因本轮删除），Docker/库/NBG SHA验证通过。maintenance/wsl-compact.json和wsl-restart-verification.json。
- TF镜像63,864,569,856B保留，LZX失败但全SHA复核完整；slim20不含官方OS镜像、boot/kernel/vipcore恢复材料未齐，代码不能替代整卡恢复。maintenance/BACKUP_DECISION.md。
- AUDIT.md已独立复算66份NPU输出、原始SHA、CPU文本raw stat和compact证据。旧阶段审计已结束，不把完整16层/Q8标已完成。

## 版本与管理
权威仓库F:/Files/OPi-Lab/repos/OragnePi-3W_ForVLM，branch codex/wsl-intern1b-reproduction；Mac只控制/报告副本。提交6c55831、b637966，尚未push。后续只stage本任务目录；其他audio/Jev/minicpm及全局docs/HANDOFF修改不是本轮所有。当前root负责此文件和主REPORT/README；owner负责full目录，量化agent只写QUANT_COMPARISON_PROTOCOL.md。

## 下步验收
1. 20图编译完→SHA传输→完整NPU中文自然输出与同源FP32对照，质量问题先定位。
2. Q4/Q8源数据与校准/证据隔离；明确NPU每类权重精度和参数比例。若当前FP16头过大，功能后再试W8头，用全词表logits与自然质量验证。
3. 同中文语义/系统缓存策略/输出长度与Intern比较，实际token数和上下文容量分别披露；无图/无语音，不冒充全链路。
4. 最终独立审计、报告/README/HANDOFF、Mac副本、secret scan和仅本轮文件提交。
