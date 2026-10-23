# PaliGemma2-3B decoder layer0 NPU 兼容性验证（2026-10-23）

状态：真实官方 BF16 权重的首个解码层已通过官方 Transformers FP32 / ONNX 参考校验，并成功编译两种 A733 NPU NBG。板卡已完成两种方案各18个冻结输入的真实执行；输出重算与驱动释放核验完成。当前不能报告 NPU 整模型速度或精度。

## 架构与输入边界

原始权重来自本项目已下载的 PaliGemma2-3B safetensors，按 index mmap 只读取语言层 0 及测试文本对应的 embedding 行，没有从 GGUF 再量化。BF16 权重转 FP32 执行参考；它是该 BF16 权重的 FP32 算术参考，并非全模型 BF16 实际推理基准。

层维度 2304、8 个 query heads、4 个 KV heads、每头 256；attention 拼接为 2048，再由 o_proj 投影回 2304。不能使用 2304/8 推断 head_dim。保留四处 `RMSNorm(x) * (1 + weight)`、GELU-tanh 门控 MLP、query 缩放 1/16、attention tanh softcap 50、RoPE theta 10000，以及层 0 的 sliding-attention 类型。最终 logits softcap 30 位于层外，本探针不包含词表头。

RMSNorm 使用代数等价的按最大绝对值缩放写法，防止 FP16 中固定小 epsilon 直接下溢。已逐例对照官方 Gemma2DecoderLayer，而非只对照自写公式。

三个中文文本分别为“你好，请介绍一下自己。”、“我今天有点难过，陪我聊两句。”、“桌子上有什么东西？”，含 BOS 后长度为 8 / 12 / 6 tokens。每句分别校验 causal 与全前缀 bidirectional 的完整层输出，共 6 个案例；取首位置、第二位置、末位置形成 18 个冻结单步输入。输入 cache 容量 32，当前 token 单独追加，共 33 个 attention 槽位。历史 K/V 来自参考实现，不是 NPU 输出循环回填。

18个输入按完整输入字节哈希实际只有13个唯一snapshot（不同句子的BOS、同句两种mask的末token存在重复）；对应3句中文、6个完整文本/掩码案例，不能称18个独立样本。这些输入同时用于校准与探针，不是独立留出集。bidirectional 文本校验不是实际图片特征；4096 sliding window 只检查边界掩码逻辑，未运行 4096-token 数值测试。

## 已完成结果

| 检查 | 结果 |
|---|---|
| 自定义 FP32 层对官方 FP32 层，6 例 | 最大绝对误差 3.052e-5；最大相对 L2 1.485e-7 |
| ONNX 对官方参考，18 例 | 最大绝对误差 7.629e-5；最大相对 L2 4.700e-7 |
| FP16 NPU 编译 | 成功，NBG 156,015,160 bytes |
| 7 个 FC 权重 INT8、FC 输入/输出 INT16、其余 FP16 编译 | 成功，NBG 70,649,288 bytes |
| NPU 端实机数值/时延 | FP16与W8各18例完成，见下表 |
| 全模型生成、图像理解、连续 KV、TTFT / tokens/s | 尚未验证 |

混合方案对七个实际矩阵按输出通道对称 INT8 量化；FC 前增加显式 dtype_converter，FC 激活的 INT16 DFP 范围采用这批参考输入峰值的两倍。生成 C 中确认为 7 个 INT8 权重常量，FP16 方案为 0 个；敏感归一化、softmax、softcap、GELU 等维持 FP16。这里不声称是全 W8A8，更不能将当前小校准集视为全场景保证。

编译器内部模拟时间只用于确认导出流程，不计为 A733 实测。两个 NBG 只包含一个 decoder layer，不含视觉编码器、projector、其余 25 层及 257216 词表头，不能直接乘除单层耗时推断完整模型速度。

## 复现文件

权重：`research/vlm-molmo-paligemma-20261023/paligemma/model`。源码与结论在本目录；大产物在 `/opt/opi-lab/paligemma2-npu-compat-20261023`。

- `export_layer0.py`：真实权重、官方 FP32 对照、18 输入及 ONNX。
- `import_calibrate.py`、`compile_nbg.py`：离线 ACUITY v6.30.22 / Ubuntu NPU v2.0.10.2，目标 VIP9000NANODI_PLUS_PID0X1000003B；每个阶段硬超时 300 秒。
- `prepare_w8.py`、`w8-fc-audit.json`：混合量化设置、每个矩阵来源 SHA 与范围。
- `export-summary.json`：所有参考误差、BF16 原张量 SHA、架构、校准范围。
- `compile-fp16.json`、`compile-w8.json`、`provenance.json`：编译命令、NBG SHA、生成 C 审计与源文件 SHA。
- `prepare_fixtures.py`、`layer_probe.c`、`board_probe.c`：18 个冻结输入的真实板卡探针；返回 0 只代表执行/释放完成，不代表模型质量通过。当前只数值检查 hidden 输出，KV 仅检查形状。

FP16 NBG SHA256：`8c40180b2c4705cf30130d7ce6718cd25db73bbeda06dd45e307db2a8a35fcb9`。混合 NBG SHA256：`b10e8ad86553da2daacd48d07395648d80497984062bdf978bf018ff263d3217`。

导出及编译进程均已退出，无后台计算任务。本目录去重后新产物约 1.102 GB（1.026 GiB），低于 1.2 GiB 约束；Windows F 剩余约 14.90 GB。不会改动当前 Intern/Smol 的 NPU 图或服务。

## 下一步与不能跳过的验证

这两个NBG的同输入实机hidden核验与驱动归零已完成，值得继续最小batch32双向prefix与权重共享门槛。即便单层通过，也仍需更多独立文本与图像特征校准、KV 数值与循环、滑窗边界、26 层累计误差、词表头与 softcap、完整生成质量、图像链路、常驻资源与延迟。现阶段不替代已验证的 InternVL3-1B。

## 实机结果补充：18次单层调用，不是完整模型速度

|指标|FP16|混合W8|
|---|---:|---:|
|实际完成调用|18/18|18/18|
|单次layer0 NPU run中位|61.515ms|11.861ms|
|run最小–最大|61.340–62.044ms|11.728–12.191ms|
|最坏hidden相对L2|0.575774%|0.852273%|
|最小hidden cosine|0.99999597|0.99996473|
|加载后driver|156,670,232B|71,303,464B|
|关闭后driver|0B|0B|
|边界采样最高温度|46.500°C|46.376°C|

原始float32输出已独立重算，全部有限，和探针打印值一致，见board-summary.json及board-results-fp16/board-results-w8。18次调用包含首个执行，没有另设warmup或重复稳定性矩阵；温度是调用边界采样，不是连续峰值。样本13唯一、来自校准，无词表头，没有递归回填NPU KV或26层误差累计。

W8在这个单层中约5.19倍快于FP16，但不能用11.86ms×26推导整模型TTFT或tokens/s。当前支持继续最小下一步；中文纯文本、prefix-LM与共享权重方案见[FULL_NPU_FEASIBILITY.md](FULL_NPU_FEASIBILITY.md)，尚未实现。纯文本也经官方无pixel mask函数确认使用双向prefix。视觉在最新范围中后置。

本轮所有导出/编译进程已退出；板卡owner已清理本次临时NBG/fixture副本，F盘原始输出及编译产物仍保留。
