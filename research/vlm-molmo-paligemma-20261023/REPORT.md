# 模型部署与优化报告（2026-10-23，进行中）

当前主力仍保留 InternVL3-1B W8 NPU。已获得新的中文 Llama CPU结果、Intern优化对照、MolmoE架构/编译兼容性结论；PaliGemma2正在完成图文部署验证。

## Intern 1B W8：继续优化的结果

系统前缀37tokens，短动态输入13/17tokens。原基线只复用32，精确候选复用37。全部模型常驻；本组仅文本decoder，不含视觉或语音。

| 方案 | 实测 | 输出检查 | 结论 |
|---|---|---|---|
| group prefill | TTFT降低约0.7–1.6% | 11/11自然回答相同；32步完整logits逐位一致 | 本轮样例数值一致的小幅收益 |
| group + exact system KV | 两短输入0.433→0.334s、0.620→0.380s | 8/11自然回答同token | 保留实验分支，暂不替换基线 |
| decoder + vocab head合并提交 | decode约提高0.9% | 输出一致性详见原始报告 | 多107.66MiB驱动，不推荐默认启用 |

88次计时请求，最高49.35°C，8次会话结束driver_bytes归零。decode整体仍约15.6tok/s，没有测出值得采用的大幅加速。输入更短使本次TTFT低于此前约0.8s，不能只比较两个不同prompt的数字来归因优化。

细节：../intern1b-operator-opt-20261023/REPORT.md；独立审计：AUDIT.md。

## Llama 3.2 1B 中文增强

未能核实用户举例的shenzhi-wang具体1B仓库。选用FuseAI/FuseChat-Llama-3.2-1B-Instruct作为可追溯中文增强候选，使用bartowski量化的Q4_K_M，通过ModelScope镜像获取并SHA校验。原作者配置确认16层、D2048、32Q/8KV/head64。它是文本模型，不具备视觉输入。

板卡CPU A76核6/7、2线程，独立进程，未替换当前NPU服务。

| 测试 | 首token | decode | 样本与缓存 |
|---|---:|---:|---|
| 热模型、不复用前缀 | P50 1.411s | P50 12.44tok/s | 3问题×2，48–50输入tokens |
| 固定prefix复用、新问题 | P50 0.638s，范围0.614–0.748s | P50 12.47tok/s | 6独立问题，全部cache_n=26，动态20–24tokens |

固定prefix组最高58.094°C，RSS峰约1.60GiB。测试中也发现整句重复会命中47–49tokens缓存、出现约0.08s；该组已保留证据但排除在固定system前缀成绩外。

家庭陪伴中文质量尚不能判定合格：有无依据的“晚安”、欢迎回家却补“明天再见”、英文essentials混入，以及未遵守简短回答的现象。32-token输出上限使多条回复截断；这组是功能/速度试验，不是综合准确率评测。

NPU尝试：已修正Llama3专用RoPE缩放，真实Q4/Q6源权重单层导出通过ONNX参考对照并编译执行。W8单层hidden误差约6%–35%，仅MLP保留FP16未修复。追加全FP16控制后，误差降至0.049%–1.10%，单层48.13ms、比W8慢5.43倍。这说明当前大误差与量化/混精路径有关，不等于架构无法在NPU执行。上述百分比是向量相对L2，不是中文回答错误率。

用户最新要求已转为完整中文NPU生成测试：正在验证attention FP16+MLP W8候选，同时准备16层、全词表输出头、模型常驻且每次新问题的测试程序。尚未取得整模型NPU的TTFT/decode，不能把单层耗时外推整模。

## MolmoE-1B

尝试未达到整模部署，未产生板端TTFT/decode。它实际约7.2B总参数/1.5B激活，16层×64专家。仅专家INT8权重就有6.00GiB，超过板卡5853MiB总RAM；现有W8全模型常驻路线无法成立。

实际兼容性试验：TopK路由、静态专家图可编译；按官方torch.where动态分发构造的图在Acuity导入时报不支持动态NonZero。现有llama.cpp转换器对官方元数据也未直接通过。详见molmo/REPORT.md、memory-lower-bound.json和成功/失败日志。这些算子图使用随机权重，不作为语言质量或整模性能证据。

## PaliGemma2-3B-mix-224

ModelScope镜像BF16两片6,064,583,376B均已校验。WSL原始BF16真实图文参考已输出church与white，单问墙钟39.58s/24.95s，峰RSS约6.35GiB；这是x86参考运行，不能算板端成绩。

ARM NEON构建和SFP压缩转换已完成，修复了转换器卷积元数据和prefix-LM位置推进问题，并保留可重放patch。最终SFP为4,045,406,208B。同一教堂图片的building、color、caption三问，与原始BF16全部输出token一致；caption包含11个token（含EOS）。单进程常驻、每问重新执行视觉编码的WSL回归也已通过。完整SFP已经上板，四次图文请求全部完成且token与BF16参考一致。首次图文TTFT74.244s；后三次52.140/48.467/48.236s。caption有10个内容token+EOS，排首token后的内容decode约1.468tok/s；一词短答不报告稳态decode。这些都包含视觉编码与256图像tokens，不能代表纯文本。最高67.319°C，RSS峰约3.91GiB，整机CPU忙碌率热请求约28%；旧进程CPU采集因线程名带空格而无效，已明确作废并修复解析器。

中文纯文本三请求现已完成：输入31/28/30tokens，无图；首请求TTFT7.154s，后续热请求6.088/6.474s，逐例decode约1.51/1.53/1.52tok/s。热进程平均1.988核（全8核24.85%），整机busy约27.69–28.16%，RSS/PSS约3.18GiB，最高56.916°C，进程swap0，无guard。修复后的CPU采集有效。板上输出与WSL SFP逐token一致，原BF16有两题相同、一题分叉。BF16/SFP都存在复述和英语混杂，不推荐据此作为纯中文陪伴chat模型。

官方无图Pali路径仍为双向prefix；原gemma.cpp无图分支未按此处理，已以独立patch修复后完成上述对照。Pali文字权重可常驻，但当前不能按原语义直接复用独立system KV。全NPU扩展暂后置，优先落实用户要求的中文Llama1B整模NPU测试。

真实BF16的Gemma2第0层已通过官方Transformers和ONNX数值对照：6个完整短文本/掩码案例、18个冻结单步输入，ONNX相对L2误差最大4.70e-7。NPU的FP16/W8两种图均已板端执行：单层API耗时中位61.52/11.86ms，最大hidden相对L2误差0.576%/0.852%。18次对应3句中文、两种mask、三个位置，共13个唯一冻结输入，属于同校准集合，不能代替26层完整生成的质量验收。这是W8而非INT4。详见paligemma/REPORT.md和../paligemma2-npu-compat-20261023/。

## F盘清理与TF备份

已删除172项可重生成产物：直接F盘1,421,901,036B；WSL内部6,958,298,487B。全部有删除前SHA、路径、原因和删除后状态，原始权重、可用NBG、源码/校准/日志均保留。WSL内部回收不等同F物理空闲立即增加，后续VHDX回收需等运行任务全部结束。

TF原图63,864,569,856B（59.48GiB）没有删除。LZX无损压缩尝试失败，随后全文件SHA重算与原记录相同。整盘备份不必永久保留，但当前官方系统包、启动/内核/NPU模块恢复链尚不齐全，仅Git代码不能完整重装。建议补齐小型恢复包并验证恢复路径后再替代整卡旧镜像。详见maintenance/BACKUP_DECISION.md。

## 来源

- Molmo：[官方模型](https://huggingface.co/allenai/MolmoE-1B-0924)、[论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Deitke_Molmo_and_PixMo_Open_Weights_and_Open_Data_for_State-of-the-Art_CVPR_2025_paper.pdf)。
- PaliGemma2：[模型](https://huggingface.co/google/paligemma2-3b-mix-224)、[gemma.cpp](https://github.com/google/gemma.cpp)。
- 中文Llama：[FuseChat作者模型](https://huggingface.co/FuseAI/FuseChat-Llama-3.2-1B-Instruct)、[作者代码/训练说明](https://github.com/SLIT-AI/FuseChat-3.0)。社区衍生模型不代替上游Llama许可要求。
