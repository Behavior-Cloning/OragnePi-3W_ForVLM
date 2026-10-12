# MLP 选择性量化加速

2026-10-12。继紧凑GQA和精确整数greedy之后，继续优化500M与1B的全部语言生成链路。对应CPU仍为GGUF Q8_0。本报告不是NPU超过CPU的验收。

## 改动与适用范围

原候选为了避免异常激活破坏生成，在每个模型的两个敏感层保留了整个MLP的FP16。本轮把这些层的gate/up矩阵恢复为逐输出通道对称INT8，输入输出使用DFP INT16；SiLU、门控乘积和down矩阵仍为FP16。最终实测候选还缩小了这两个层中attention/KV的INT16范围。其余层及输出头保持原产物。

| 模型 | 修改的层索引（从0开始） | 敏感层中的FP16矩阵 | MLP激活范围余量 | 这两个层的其他INT16余量 |
| --- | --- | --- | --- | --- |
| 500M | 3、31 | 每层3个→1个 | 4 bit（校准范围×16） | 1 bit（原为4） |
| 1B | 2、3 | 每层3个→1个 | 1 bit（校准范围×2） | 0 bit（原为1） |

这是已有GGUF Q8权重上的PTQ，并非AWQ/GPTQ，也不是全模型所有算子都用INT8。更窄范围可降低舍入误差，但对未见输入的截断风险仍需留出和长历史测试。没有用留出集调参。

编译复用先逐项核对原始图、权重、量化参数的SHA256以及融合/GQA选项，仅重编译变化的块，旧产物保持不变。

配置审计更正：初版`--mlp-headroom-bits`误作用于非MLP的INT16范围循环，实际MLP仍用原余量。因此目录名h1/h0及旧conversion中的CLI参数不能直接当作有效MLP范围。v2已修正参数路由；用500M `--activation-headroom-bits 1 --mlp-headroom-bits 4`、1B `--activation-headroom-bits 0 --mlp-headroom-bits 1`重新编译四层，量化参数语义完全一致，NBG逐字节一致。完整差异与重编译哈希在`mlp-precision-audit.json`。本报告按真实产物说明组合调整，不把质量或速度变化单独归因于MLP。

## 开发样例的精度门槛

仍按完整logits相对L2≤5%、最低逐步余弦≥0.99、argmax一致率≥95%检查；参考使用实际编译常量重建的全深度FP32。它是执行误差门槛，不能替代相对原模型的量化质量验收。

| 模型 | 候选 | 场景 | L2 | 最低余弦 | argmax一致 | 门槛 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 500M | 初版范围 | companion | 3.770% | 0.998671 | 42/45 | 拒绝 |
| 500M | 选择性MLP | greeting | 4.357% | 0.997489 | 6/6 | 通过 |
| 500M | 选择性MLP | companion | 3.629% | 0.997628 | 88/90 | 通过 |
| 500M | 选择性MLP | 固定输入轨迹 | 3.563% | 0.998505 | 66/67 | 通过 |
| 500M | 原GQA基线 | 固定输入轨迹 | 3.132% | 0.998321 | 65/67 | 通过 |
| 1B | 初版范围 | greeting | 3.321% | 0.999083 | 15/15 | 通过 |
| 1B | 初版范围 | companion | 3.447% | 0.989685 | 63/64 | 拒绝 |
| 1B | 选择性MLP | greeting | 3.374% | 0.999105 | 15/15 | 通过 |
| 1B | 选择性MLP | companion | 3.439% | 0.990319 | 64/64 | 通过 |
| 1B | 选择性MLP | 固定输入轨迹 | 3.419% | 0.990319 | 64/64 | 通过 |
| 1B | 原GQA基线 | 固定输入轨迹 | 3.435% | 0.990168 | 64/64 | 通过 |

初版500M因argmax一致率低于95%被拒绝，1B因最低余弦低于0.99被拒绝；最终实测版缩小两个敏感层中非MLP的INT16范围后通过上述开发样例。未放宽阈值，未把失败方案纳入合格速度成绩。

## 五轮完整链路速度对照

每个模型使用原GQA候选的陪伴输出作为固定输入轨迹（teacher forcing），两版都计算全部decoder、KV更新、全词表head与greedy预测，再把相同的冻结token喂回下一步。预测仍被保存，但固定轨迹的预测拼接不作为自由生成回答。500M为30个prompt token加67次预测；1B为49加64。容量128，不冒充128步生成或长历史测试。

运行器新增固定反馈功能；原模型重放自己的输出时，全部logits与原自由生成逐字节相同。默认仍是自然生成，EOS正常结束；固定模式明确记录feedback token、要求文件长度/词表范围合法，并执行约定的全部步数。板端另验证短/长反馈文件、负token、词表越界和KV超容全部拒绝；人为指定首个预测为EOS，自然模式停止而固定模式继续，完整logits仍匹配原轨迹。

各变体先做完整诊断、再预热一次，交替五轮。每个变体在重复运行中要求自身预测和完整诊断logits与已通过数值门槛的冻结结果一致。不同量化变体允许预测不同，但输入轨迹、工作量完全相同。全部启用等价整数greedy，性能运行不写全词表logits。

本轮仍使用起温≤68°C的补充协议，原/新版都重新测量，不混用此前55°C成绩。每秒采样网页健康状态和系统指标，繁忙或异常时拒绝该实验；短于采样间隔的干扰仍可能漏掉。

### 500M

实验状态：complete；已记录14轮（含诊断和预热）。

| 变体 | decode token/s 中位（范围） | 首字秒 | 总响应秒 | CPU秒 | 驱动MiB | 起温°C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 12.968（12.346–12.982） | 2.122 | 7.211 | 0.146 | 338.17 | 58.5–60.3 |
| selective | 13.923（13.691–14.083） | 1.935 | 6.676 | 0.206 | 328.13 | 58.2–60.5 |

相对同轮基线decode中位提升7.4%，5/5轮更快；首字缩短8.8%；驱动分配减少10.04MiB。加载单独计时，首字/响应表不含加载。

| 变体 | 推理后RSS MiB | 每秒采样最少可用RAM MiB | swap使用MiB范围 | 采样最高温°C |
| --- | ---: | ---: | ---: | ---: |
| baseline | 7.00–7.12 | 4086.4 | 991.00–991.50 | 61.1 |
| selective | 6.75–7.12 | 4087.1 | 991.00–991.50 | 62.1 |

NPU起止频率：[1008000000]Hz；所有测量释放后驱动分配为0。RSS与设备驱动分配分别记录，不能直接相加当作整机峰值；原始每token耗时、整机CPU负载、加载和系统采样均在JSON。

### 1B

实验状态：complete；已记录14轮（含诊断和预热）。

| 变体 | decode token/s 中位（范围） | 首字秒 | 总响应秒 | CPU秒 | 驱动MiB | 起温°C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 11.488（10.425–11.598） | 3.273 | 8.756 | 0.257 | 460.32 | 60.3–62.7 |
| selective | 13.219（11.901–13.256） | 2.731 | 7.492 | 0.194 | 442.51 | 60.5–62.9 |

相对同轮基线decode中位提升15.1%，5/5轮更快；首字缩短16.6%；驱动分配减少17.81MiB。加载单独计时，首字/响应表不含加载。

| 变体 | 推理后RSS MiB | 每秒采样最少可用RAM MiB | swap使用MiB范围 | 采样最高温°C |
| --- | ---: | ---: | ---: | ---: |
| baseline | 7.00–7.12 | 3970.8 | 992.25–995.50 | 63.2 |
| selective | 7.00–7.12 | 3971.0 | 992.50–995.50 | 63.8 |

NPU起止频率：[1008000000]Hz；所有测量释放后驱动分配为0。RSS与设备驱动分配分别记录，不能直接相加当作整机峰值；原始每token耗时、整机CPU负载、加载和系统采样均在JSON。

## 自由生成与原始Q8参考

### 500M

 I'm so sorry you're feeling this way. It's okay to feel lonely sometimes, and there are many things we can do together to make it better. One thing you could do is reach out to a friend or family member who might be able to help you feel more connected. You could also try talking to a trusted adult, like a teacher or counselor, who can offer you support and help you find ways to make yourself feel better.<end_of_utterance>

500M在第90个token自然EOS结束。
该自由轨迹相对原始GGUF Q8反量化FP32参考：logits L2 5.331%，平均KL 0.009529，argmax 84/90。该值包括再次量化与执行误差，不等于“相对原模型无损”。不同自由轨迹不直接比较总响应时间。

### 1B

I'm here to help you feel better, so I'll offer you two small comforting thoughts and a simple thing you can do to help lift your spirits tonight.

**Small Comforting Thought:**
"Remember, you are not alone. There are people who care about you and want to support you. Sometimes, just being

1B在64-token上限处截断，不是自然EOS完整回答。
该自由轨迹相对原始GGUF Q8反量化FP32参考：logits L2 5.053%，平均KL 0.005392，argmax 60/64。该值包括再次量化与执行误差，不等于“相对原模型无损”。不同自由轨迹不直接比较总响应时间。

## 未达成的目标与复现

当前仍未证明超过优化CPU Q8；500M历史CPU约27.5、1B约20.3 token/s仅为参考，尚未完成同轨迹正式对照。长历史128/512/2048、独立留出陪伴质量、批量prefill及30分钟稳定性仍待完成。继续提速应优先处理剩余FP16 down矩阵、decoder算子开销与prefill；不能把短历史开发样例当成最终验收。

编译：`compile_weight_int8.py --float-mlp --int8-mlp-projections gate up --mlp-headroom-bits N`；融合复用：`compile_fused_stack.py --reuse-blocks-from w8-fused2-gqa`。运行固定轨迹：`run_full_model.py --feedback-from <自然生成tokens.json>`。重复测速：`benchmark_variants.py --fixed-feedback --variant-reference <标签> <已验收目录>`。

运行`python3 npu/kv-cache/report_mlp_optimization.py`从本地原始结果重建报告和小型证据。模型、SDK、NBG继续保留本地忽略目录；源码、编译哈希、成功/失败验证和原始性能数据归档Git。
