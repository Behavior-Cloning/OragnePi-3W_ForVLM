# FuseChat-Llama-3.2-1B：WSL 单层 NPU 编译兼容性

2026-10-23。**单个真实语言 layer0 已在 A733 NPU 执行。W8 与仅 MLP FP16 两版精度失败；追加全 FP16 控制后，六组 hidden 相对 L2 降到 0.0487%–1.1043%，证明该层的浮点 NPU 路径基本可用，但其单层耗时增至 48.13 ms（W8 为 8.87 ms），尚不能推广为完整模型可用。** 本实验仅运行单层，不能称完整模型 NPU 部署成功。板卡测试由父任务指定的唯一操作者执行，WSL 导出工作没有争抢板卡。

## 已完成

- 源模型：FuseAI/FuseChat-Llama-3.2-1B-Instruct，量化文件由 bartowski 提供；直接使用父任务已下载的 Q4_K_M GGUF，未下载或展开全模型 BF16。
- 架构真实核对：16 层、hidden 2048、32 个 Q heads / 8 个 KV heads、head dim 64、FFN 8192、词表 128256，输入和输出 embedding 共享。不能套用 Intern 的 24 层 / 896 hidden。
- layer0 共 60,821,504 参数，从真实 Q4_K/Q6_K/F32 张量解量化；校准使用15个真实 BPE token 的 embedding 和增量KV状态，其中6个状态进入校准。它只验证兼容性，英文短样例不构成中文或任务精度校准。
- ONNX 增量 KV 与 FP32 全序列因果注意力对照，15 token 最大绝对误差 3.81e-06，最大相对 L2 7.3e-07。这是相对同一 GGUF 解量化权重的算术检查，不是相对原 BF16 全模型的质量检查。
- ACUITY import、输入元数据、INT16 校准均成功；随后保留残差/归一化/softmax敏感路径 FP16、7 个 FC 权重改成 per-channel INT8，NBG 编译成功，耗时 20.27 秒。
- 生成 C 的 7 个权重常量均为 `VSI_NN_TYPE_INT8`，证据在 `storage-audit.json`；不是原生 INT4 加速。
- NBG 55,284,520 bytes（52.72 MiB），位于 `/opt/opi-lab/llama32-compat-20261023/compiled/export_nbg_unify/network_binary.nb`。

## 真实板卡单层配对

原 W8 与 MLP FP16 例外使用相同六组冻结输入、相同正确 RoPE cos/sin、相同 FP32 GGUF参考。每个位置各执行一次，不是稳定吞吐矩阵。位置对应15-token英文短序列中的六个校准状态，**不是独立留出数据**。记录的耗时仅该层 VIP run API，不含全模型、加载、prefill首字或输出词表。

|位置|7FC W8耗时 ms|hidden相对L2|4FC W8+3FC FP16耗时 ms|hidden相对L2|
|---:|---:|---:|---:|---:|
|0|8.887|34.78%|40.904|34.70%|
|1|9.033|35.08%|40.864|35.18%|
|2|9.035|8.28%|40.771|7.59%|
|4|8.857|6.05%|40.820|5.48%|
|8|8.838|15.05%|40.808|14.39%|
|14|8.834|19.42%|40.867|18.91%|

六次状态调用中位数 8.872→40.842 ms。原层驱动分配55,811,336 bytes，MLP FP16例外110,754,840 bytes；两次均释放为0。状态边界温度原层44.02–45.32°C，例外39.18–40.18°C；没有连续全系统遥测，不报告整机内存/CPU峰值或降频结论。

误差已从回传 `outputs/*.f32` 在WSL独立重算，和板端JSON一致。全部6个状态都超过旧探针5%相对L2筛查线；这不是任务准确率，但足以拒绝当前适配为可用模型。`layer_probe` 返回0只表示执行完成/释放成功，并非质量PASS。

MLP FP16 配对包括三个FC和MLP内部全部浮点，注意力四FC仍W8。它只能说明“仅把MLP改浮点”没有消除主要误差；不能证明MLP完全无误，也不能未经逐算子采样断言问题一定在注意力、KV或RMSNorm。随后按用户优先 NPU 的要求，追加下节的一组全 FP16 对照；其他逐算子定位未展开。

## 追加全 FP16 控制：量化路径是主要嫌疑

此次只增加一个 layer0 变体、沿用同一导入图、同一 Q4_K_M 解量化权重和同六组冻结输入。所有权重/激活量化元数据改为 FP16，删除旧 INT64 bias 元数据；生成 C 可核对 7 个 FC 权重为 FP16，全部普通张量为 FP16，编译器自行生成的三个偏置常量为 FP32，无 INT8/INT16 存储声明。该声明不推断 NPU 内部累加精度。

|位置|全 FP16 单层 API ms|hidden 相对 L2|旧 7FC W8 相对 L2|
|---:|---:|---:|---:|
|0|48.187|0.0487%|34.78%|
|1|48.138|0.1144%|35.08%|
|2|48.078|0.1010%|8.28%|
|4|48.745|0.0922%|6.05%|
|8|48.090|1.0417%|15.05%|
|14|48.130|1.1043%|19.42%|

六组状态全部低于历史 5% 单层粗筛线；输出已由 `analyze_all_fp16.py` 对回传 f32 使用 FP64 范数独立重算，和板端结果一致。**5% 只是兼容性筛查，不是生成质量标准；最大残余 1.10% 仍需解释，也没有连续 NPU KV 反馈和多层误差传播证据。** 这些状态属于校准集，仍不是中文留出测试。

中位 API 时间 **48.134 ms**，比 W8 的 8.872 ms 慢 **5.43 倍**；驱动占用 122,347,136 bytes，释放为 0。六次调用边界温度 41.54–42.47°C；没有持续整机遥测。单层时间不可直接外推完整模型 tok/s，更不能和当前 Intern 1B 的整模型约 15 tok/s 直接比较。

结合“仅 MLP 浮点没有修复、全浮点大幅改善”，当前主要误差更可能来自量化/混精执行路径；这包含注意力权重与激活、归一化边界、KV/score 量化等共同变化，**仍不足以锁定某一个算子或断言某一个量化常量必错**。因此值得继续做范围明确的 NPU 定位，优先沿注意力与 KV 路径逐张量对比、检查校准范围/饱和与量化边界；当前不应盲目编译完整 16 层，也不推荐以全 FP16 替代现有 Intern W8 主线。下一步应先取得更接近 FP32 的单层留出与连续 KV 结果，再决定全模型工作。

`compile_all_fp16.py` 使用硬链接复用 decoder.data/json/inputmeta，实际新增 243,944,662 bytes（约 232.6 MiB），低于 300 MiB 限额。NBG 为 121,813,728 bytes，SHA256 `031202e00e8609a07b4434700ba15f1323a4831421239787b0fee7500bdce36f`，编译 22.60 秒。构建目录 `/opt/opi-lab/llama32-compat-20261023/all-fp16`。源码、conversion、storage audit、编译日志及 `board-results-all-fp16/`、`all-fp16-board-comparison.json` 已保留；板上该变体副本由唯一操作者清理。本追加结束，不留后台任务。

## 必须修改的兼容性差异

原 TrainedDecoder 虽支持 `llama` 类型和 GGUF Q/K 交错 RoPE 行到 HF split-half 的逆置换，但原 `rope()` 只使用 freq_base。本模型 GGUF 带 `rope_freqs.weight`，必须使用 `inverse_frequency = base_inverse_frequency / rope_freqs`。已与 Transformers 4.57.6 `_compute_llama3_parameters` 对照，最大绝对差 1.16e-10。

因此不能只把 Intern 的 C 宏改成2048就声称支持 Llama3.2：还需接入频率缩放、Llama聊天模板、BOS以及三个结束 token（128001/128008/128009），核对权重共享、词表输出切片和KV布局。当前冻结输入含正确 cos/sin，准备的 layer_probe 直接读取，不走错误的旧 C RoPE 公式。

现有 Intern 的 GGUF 解量化、增量 KV、GQA、RMSNorm、SwiGLU、ACUITY 导出框架大部分可以复用。**已证明一个真实层能编译，不等于其余15层、完整输出头、常驻缓存和全模型精度已经通过。**

## 精度与后续门槛

Q4_K_M → 解量化 FP32 → W8 会保留原 Q4 损失并增加再次量化误差，绝不能作为原 BF16 的精度基线。若继续全模型部署，应获取原 BF16/FP16权重或同时保留同源 GGUF CPU参考，采用相同反馈历史做逐层/全词表对照，并增加中文家庭语料校准和独立留出集。

当前只有上述单层NPU耗时与数值误差，没有全模型 TTFT/decode 或生成质量成绩；不能由单层耗时推算整体 tok/s。尚未完成其余15层、输出头、连续NPU产生KV的误差传播、中文真实任务、system缓存与完整生成。六组 frozen state 使用参考KV，不能替代连续NPU KV回归。

## 空间与复现

前两个版本新增构建产物按硬链接去重共约585.7 MB（逻辑文件大小829.1 MB）；243,296,898-byte ONNX在成功导入校准后验证SHA再删除（`onnx-cleanup.json`），保留可复现源码、原始GGUF、decoder.json/data/quantize、校准输入、NBG和编译证据。没有删除源权重，未修改其他模型。

1. `/opt/opi-lab/export-venv/bin/python export_layer0.py` 导出并检查。
2. `python3 controller.py` 执行导入/校准、删除可再生ONNX、W8导出；首次本地Python3.10不支持hashlib.file_digest，已修为流式SHA函数，失败日志保留。随后仅恢复W8阶段，未重复导出/校准。MLP候选通过相同编译器加 `--float-mlp` 生成，保留独立conversion和NBG；decoder.data用硬链接共享，只读，不覆盖原始文件。
3. 板上后续编译命令：`cc -O2 -DMODEL_DIM=2048 -DMODEL_KV=8 -DMODEL_HD=64 layer_probe.c -lNBGlinker -lVIPhal -lm -o layer_probe`；运行 `layer_probe network_binary.nb fixtures outputs`，调用方仍须双锁并建立输出目录。

权威源码 `F:/Files/OPi-Lab/repos/OragnePi-3W_ForVLM/research/llama32-compat-20261023`；构建 `/opt/opi-lab/llama32-compat-20261023`。本目录没有修改 root HANDOFF；前两版实验已随提交 `6c55831` 保存，本次全 FP16 追加由父任务统一提交。原始输入/reference在WSL fixtures，板端输出和日志分别保留在 `board-results/` 与 `board-results-mlp-fp16/`。板端临时NBG由唯一操作者负责清理，WSL两份NBG、原始GGUF、校准和失败证据保留。

## 追加 attention FP16 + MLP W8 与中文留出

按完整中文 NPU 部署方向，新增一个受控混精候选：注意力四 FC、KV、外部输入输出、softmax 全 FP16；MLP 三 FC 权重 W8，只有 FC 邻接的 11 个必要张量保留原校准 INT16，SiLU 保持 FP16。校准没有更换，另生成三个从未参与该校准的中文句子，每句取早期/中段/末尾共九个冻结状态；输入为 raw 中文+BOS，没有套聊天模板。Tokenizer 从 GGUF tokens/merges 构造并验证中文逐字还原；正式生成仍需和 llama.cpp tokenization 对照。参考使用同一 Q4 GGUF 解量化 FP32，增量与完整因果层输出相对 L2 ≤1.21e-6。

|同15状态比较|attention FP16 + MLP W8|全 FP16|
|---|---:|---:|
|单层 API 中位时间|16.248 ms|48.299 ms|
|六个英文校准状态L2|0.6323%–4.0911%|0.0487%–1.1043%|
|九个中文留出状态L2|2.1412%–3.6978%|0.0986%–0.2527%|
|驱动已加载 bytes|67,411,824|122,347,136|
|驱动已释放后 bytes|0|0|

原始 f32 均独立重算，与板端记录一致。**混精在这15个状态全部通过旧 5% 单层粗筛，比全 FP16 快约 2.97 倍；它是值得推进完整模型试验的候选，不是已通过中文生成精度认证。** 相比原全 W8 的8.87 ms仍更慢，但原版数值不合格，不能只拿速度择优。九个中文状态只来自三个句子，且都使用参考 KV；多层累计、生成时连续 NPU KV、自反馈token、完整词表、system-cache和真实中文回答尚待全模型实测。当前剩余约4%的单层误差需要在完整同token logits/生成对照中继续关注；不可直接用单层耗时外推decode。

候选 NBG66,878,544 bytes，SHA256 `227225b1ea4d8a44b508990b2602948d4930537dfdf5a0f8083b80a523826a8b`。本组总新增约144.4MB（含输入tar，复用decoder.data硬链接），没有下载模型。`compile_attention_fp16_mlp_w8.py` 已支持每层独立 INT16 校准结果直接构造 MLP per-channel W8，无需先生成另一个全 W8 NBG；本次执行脚本 snapshot、source manifest、quant节点清单、storage audit和原始15状态配对证据均保存。完整16层+4片全词表头的 exporter/runtime 由主任务另行准备，本子任务不启动整模大编译；板上本次临时副本已清，无遗留后台进程。
