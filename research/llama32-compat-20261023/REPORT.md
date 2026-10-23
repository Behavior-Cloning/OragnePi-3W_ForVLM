# FuseChat-Llama-3.2-1B：WSL 单层 NPU 编译兼容性

2026-10-23。**单个真实语言 layer0 已成功在 A733 NPU 执行，但精度失败：六组 hidden 相对 L2 误差为 6.05%–35.08%。把 MLP 改为 FP16 后仍有 5.48%–35.18% 误差，且慢约 4.6 倍，拒绝采用。** 本实验仅运行单层，不能称完整模型 NPU 部署成功。板卡测试由父任务指定的唯一操作者执行，WSL 导出工作没有争抢板卡。

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

MLP FP16 配对包括三个FC和MLP内部全部浮点，注意力四FC仍W8。它只能说明“仅把MLP改浮点”没有消除主要误差；不能证明MLP完全无误，也不能未经逐算子采样断言问题一定在注意力、KV或RMSNorm。下轮可用全FP16对照、注意力/KV范围与实际算子输出定位，本轮按任务边界结束，不继续扩大矩阵。

## 必须修改的兼容性差异

原 TrainedDecoder 虽支持 `llama` 类型和 GGUF Q/K 交错 RoPE 行到 HF split-half 的逆置换，但原 `rope()` 只使用 freq_base。本模型 GGUF 带 `rope_freqs.weight`，必须使用 `inverse_frequency = base_inverse_frequency / rope_freqs`。已与 Transformers 4.57.6 `_compute_llama3_parameters` 对照，最大绝对差 1.16e-10。

因此不能只把 Intern 的 C 宏改成2048就声称支持 Llama3.2：还需接入频率缩放、Llama聊天模板、BOS以及三个结束 token（128001/128008/128009），核对权重共享、词表输出切片和KV布局。当前冻结输入含正确 cos/sin，准备的 layer_probe 直接读取，不走错误的旧 C RoPE 公式。

现有 Intern 的 GGUF 解量化、增量 KV、GQA、RMSNorm、SwiGLU、ACUITY 导出框架大部分可以复用。**已证明一个真实层能编译，不等于其余15层、完整输出头、常驻缓存和全模型精度已经通过。**

## 精度与后续门槛

Q4_K_M → 解量化 FP32 → W8 会保留原 Q4 损失并增加再次量化误差，绝不能作为原 BF16 的精度基线。若继续全模型部署，应获取原 BF16/FP16权重或同时保留同源 GGUF CPU参考，采用相同反馈历史做逐层/全词表对照，并增加中文家庭语料校准和独立留出集。

当前只有上述单层NPU耗时与数值误差，没有全模型 TTFT/decode 或生成质量成绩；不能由单层耗时推算整体 tok/s。尚未完成其余15层、输出头、连续NPU产生KV的误差传播、中文真实任务、system缓存与完整生成。六组 frozen state 使用参考KV，不能替代连续NPU KV回归。

## 空间与复现

两个版本新增构建产物按硬链接去重共约585.7 MB（逻辑文件大小829.1 MB）；243,296,898-byte ONNX在成功导入校准后验证SHA再删除（`onnx-cleanup.json`），保留可复现源码、原始GGUF、decoder.json/data/quantize、校准输入、NBG和编译证据。没有删除源权重，未修改其他模型。

1. `/opt/opi-lab/export-venv/bin/python export_layer0.py` 导出并检查。
2. `python3 controller.py` 执行导入/校准、删除可再生ONNX、W8导出；首次本地Python3.10不支持hashlib.file_digest，已修为流式SHA函数，失败日志保留。随后仅恢复W8阶段，未重复导出/校准。MLP候选通过相同编译器加 `--float-mlp` 生成，保留独立conversion和NBG；decoder.data用硬链接共享，只读，不覆盖原始文件。
3. 板上后续编译命令：`cc -O2 -DMODEL_DIM=2048 -DMODEL_KV=8 -DMODEL_HD=64 layer_probe.c -lNBGlinker -lVIPhal -lm -o layer_probe`；运行 `layer_probe network_binary.nb fixtures outputs`，调用方仍须双锁并建立输出目录。

权威源码 `F:/Files/OPi-Lab/repos/OragnePi-3W_ForVLM/research/llama32-compat-20261023`；构建 `/opt/opi-lab/llama32-compat-20261023`。本目录没有修改root HANDOFF或提交Git。原始输入/reference在WSL fixtures，板端输出和日志分别保留在 `board-results/` 与 `board-results-mlp-fp16/`。板端临时NBG由唯一操作者负责清理，WSL两份NBG、原始GGUF、校准和失败证据保留。
