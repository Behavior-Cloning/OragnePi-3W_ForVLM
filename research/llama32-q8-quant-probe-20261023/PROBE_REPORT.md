# Llama Q8 / 同校准 Q4 量化探针（待板测）

目前仅完成编译，尚无本组板端数值或耗时；不能据此判断模型可用。

| 候选 | 实际 FC weight quant 条目 | FC 输入/输出边界 | NBG 大小 |
|---|---|---|---:|
| Q8 源 all-FC W8 首层 | 7/7 `i8`，per-channel symmetric | 7/7 A16 | 55.24 MB |
| Q8 源 attention FP16 + MLP W8 | 3/7 `i8`，4/7 `float16` | MLP A16，attention FP16 | 66.84 MB |
| Q4 源同中文校准 all-FC W8 | 7/7 `i8`，per-channel symmetric | 7/7 A16 | 55.29 MB |
| Q8 源 W8 完整输出头 | 每片 1 个 `[2048,32064]` `i8` FC，共 4 片 | FC A16，RMSNorm 内部 FP16 | 四片共 234.16 MB |

以上读取实际保存的 `decoder.json` 与 `decoder.quantize`，不是编译命令参数的推断。明细、逐 FC scale 范围、FP16 tensor 例外与文件 SHA 见 `precision-audit.json`。第一轮有界构建清理 scratch 时未保留生成 C；这些材料不是 NBG 内部反汇编证据。板端运行器会对真实 NBG 调用 vip_query，输出 external IO 类型、量化格式和 fixed-point position，完成后补记录。

All-FC W8 首层的 hidden 输入/输出与 residual、两处 norm 内部、softmax 采用 FP16；KV 与 RoPE 等保留校准的 A16。Mixed 版本显式量化表把非 FC-adjacent tensor 改为 floating，未显式列出的内部节点由编译器推断；不要简单把整图写成全 INT8。

完整头的四片输出均为 INT16，但 fl 分别是 **11、10、11、10**，反量化比例分别为 **1/2048、1/1024、1/2048、1/1024**。运行器使用每片真实 IO 参数还原 F32 后再合并 128256 个 logits，禁止比较四片的 raw INT16。外部 hidden 输入的 quant 条目为空，待板上真实查询确认，暂不假定 A16。

校准/留出为四条/四条家庭陪伴中文请求，首层各选五个位置；共 40 条记录，但固定 system 引起共享：两组各 11 个不同输入，真正未见校准的留出输入为 **8 个**。head 有 16 个真实 16 层 FP32 输出状态，8 校准、8 留出。Head 尚未包含 NPU decoder 累计误差。

Q4 与 Q8 控制组使用相同 token IDs、位置、容量 256、precision flags、数据生成代码，并已 assert。旧 Q4 英文校准与本组中文校准不是同一实验，不能把改善单归因 Q8 源精度。

板测结果：等待负责人返回原始输出。需分别报告 hidden/newK/newV 相对误差、真正未见输入子集、四片合并全词表 argmax、hot 单图时间与 driver 回收。单图耗时不能外推完整模型 TTFT/decode。
