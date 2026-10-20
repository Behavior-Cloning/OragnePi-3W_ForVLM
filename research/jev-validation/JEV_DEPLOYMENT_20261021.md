# Jev 板端部署与 VLM 共驻试验（进行中）

日期：2026-10-21。权威代码和实验资料位于 Windows F 盘，Mac 仅保存报告副本。当前报告尚未代表完整矩阵完成。

## 本轮范围与版本

主 VLM 保留 SmolVLM-500M、InternVL3-1B；Qwen2.5-1.5B 为用户指定对照保留。Jev 的动作均为虚拟标签，没有接入电机。模型原始选项准确率与未来确定性安全门必须分开评估。

| 模型 | 下载/部署 | 边界 |
|---|---|---|
| Nano-Jev v1.0 33.36M | HF mirror；ModelScope同名无记录；已板端校验 | 英文 RAG 交叉编码器，不能据尺寸推断中文动作能力 |
| Laya multilingual 322M | ModelScope，已板端校验 | CPU PyTorch，中文动作集正在测 |
| Jev-Style 0.8B Decision-v3 Q4_K_M | HF mirror，官方 manifest SHA 已逐文件核对 | WSL 已交叉编译 ARM64 jev-score；必须使用槽位评分协议，非普通聊天生成 |
| OneJev1.6-0.8B | ModelScope 完整权重已下载 | 多模态 encoder/图片前处理/板端部署尚列 TODO |
| JEV-9B / JEV-27B-VL | 远程存储服务器后台下载任务已启动 | 无 GPU 是预期配置，仅下载存档；尚不能标记完整校验成功 |

Jev-Style 的官方测试 runtime commit 为 `441df11f65ea0b6d0c72965aaf70c8241070ddcb`。网络下载缓慢，本轮交叉编译使用 slim20 恢复的 llama.cpp 源码快照；该快照支持 qwen35 与 n_outputs_max。其完整目录摘要、头文件摘要和编译器版本见 `sources/runtime-source-provenance.json`，不可把父仓库 Git SHA 冒充 llama.cpp 版本。最终效果必须以本轮实际运行结果为准。

## 已观察到的烟测

Laya 的4候选测试：加载约9.91s；预热后调用约0.485–0.496s；进程RSS约1.65GiB。输入“请看向左侧”却选了右侧（3次一致），因此该烟测仅证明成功推理，不能证明模型可用。后续以140条冻结样例的结果替代这个小样本结论。

## 测试定义与限制

- 小模型运行在 CPU A76核6/7，2线程；NPU用于500M/1B文本decoder。未包含视觉encoder前向，不称完整图片VLM测试。
- 10候选热态块：20次预热+200次测量。候选4/8/16/32/64/100及较长状态先进行初筛，不能称全矩阵200次。
- 64/128/256是合成state长度下限，完整保留原请求，记录各分词器真实token数；不同分词器不能当作同token工作量横比。
- Nano对所有候选分别交叉编码，按8个一批执行；Jev-Style和Laya按自身原生候选机制。禁止静默截候选/状态。
- `sources/board_bench-screen-v1.py`保留最初板端脚本；该版调用计时包含一次轻量资源采样，后续脚本已经把采样移到计时外。最终报告应区分脚本版本。
- 全程保留原有1.5B CPU服务时，所谓Jev单模型推理仅表示Jev独占计算，不表示系统没有其它常驻权重。
- RSS不是NPU driver内存；共驻实验另记录NPU worker的driver计数。PSS更适合共享内存总量，不能把多个RSS简单相加当作物理内存。

## 历史 1.5B / INT4 纠正

恢复工作树内确有Qwen2.5-1.5B完整NPU实验：28 decoder层、4个完整词表输出头、ctx128、输入44/输出64、5次运行；decode中位数5.032149 tokens/s、TTFT 8.042304s、driver约675MiB。但输出重复，质量失败。

该候选名为w4-vdfp-sensitive、group28，W4值仍以INT8的-7..7存放。没有证据证明原生packed INT4存储和带宽加速。导出常量的FP32 oracle也重复，表明问题已在重数量化常量处发生，不应把这组速度当成可用INT4结果。500M/1B没有完整合格INT4结果。本轮保留这些失败证据，不删除对应源权重或1.5B对照包。

## 清理

实际删除 `gguf-fresh/export/int16/l0…l27/export/decoder.export.data` 共990,086,768字节。每层检查decoder.json/data/quantize与最终NB均在，ONNX嵌入权重保留。活跃w8-balanced/fused的FP32 oracle所需export.data全部保留。路径及SHA见 `results/cleanup-20261021.json`。

这是WSL内部回收约0.92GiB，不能直接声称F盘物理回收同等容量；尚未压缩VHDX。清理后又新增模型，F盘快照可用约58GiB；WSL内存快照约0.5GiB使用、15GiB可用，编译时会更高。

## 剩余审计项

- [ ] 汇总 Nano/Laya/Jev-Style 板端时延、准确率、温度、内存。
- [ ] 500M/1B NPU与1.5B CPU的分别常驻/同时推理矩阵；明确1.5B是文本模型。
- [ ] 服务器大权重完整性校验结束。
- [ ] 多模态OneJev的真实图片前向、视觉编码器与VLM共驻（README已单列）。
- [ ] 确定性安全门、mock回执、state_id/过期结果取消、最优候选1000次稳定性；当前不是实体执行链路。
- [ ] 完整A55/A76、1/2线程、状态长度和候选数矩阵；初筛不足时不能宣称已完成。

## 2026-10-21 追加优化要求（执行中）

每个 Jev 都必须尝试提速，并比较优化前后延迟、决策准确率、资源和与VLM的竞争；不能用部署成功替代优化验收。审计智能体已记录 `JEV_OPTIMIZATION_AUDIT.md`，58个检查项。

已经准备的矩阵：Nano/Laya/JevStyle分别测A76单线程/双线程、A55双线程；Nano候选batch1/8、PyTorch动态INT8、ORT FP32/INT8；Laya仅编码器Linear动态INT8而保持决策头FP32；JevStyle序列槽17→2、ubatch1024→256。每个配置先3次预热、30次计时、20条跨组/未知/危险质量对照，属于初筛，胜出后才扩展验收。OneJev视觉未执行，继续列部署和优化TODO。

Nano的ONNX FP32导出烟测logit最大误差8.94e-7，INT8约0.106。INT8有数值误差，不据模型文件变小判定精度合格；需比较argmax和完整质量集。Laya直接量化整个决策头曾触发weight属性兼容错误，失败日志保留；当前编码器量化方案已通过桌面运行检查，板端结果待测。

另清理已安装的WSL 3.0.3 MSI和Ubuntu rootfs安装缓存共708,706,003字节，直接位于F盘；保留运行中的VHDX、TF备份及slim20原始恢复包。校验及清理清单见 `results/install-cache-cleanup-20261021.json`。
