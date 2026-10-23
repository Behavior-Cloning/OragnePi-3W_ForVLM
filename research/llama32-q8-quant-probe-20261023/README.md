# Q8-source Llama 首层和完整输出头量化探针

目的：对齐 Intern 的 Q8 GGUF 源、W8 FC 路线，判断 Q4 来源与注意力量化误差是否可分离。这里不是完整模型推理或最终速度结论。

- 源权重：FuseChat-Llama-3.2-1B-Instruct-Q8_0.gguf，SHA256 `c7db2b30d2b176998db90304d1ac6dd259379bd75f9ea8b6a094be88ea47b1b4`。真实 Q8 解量化 FP32 参考，不是原始 BF16。
- 独立目录 `/opt/opi-lab/llama32-q8-quant-probe-20261023`，不覆盖 full-model Q4/Q8 资产。
- 4 条中文校准、4 条独立中文留出；每条首层 5 个状态（BOS、首部、固定 system 尾、用户文本、整段末尾），共 40，20/20 分开统计。固定 system 相同导致共享状态：校准/留出各 11 个唯一输入，留出仅 8 个输入未出现在校准中；另报这 8 个真正未见输入的误差，不能将全部 20 个留出记录说成独立测试。固定容量 256。
- 两个首层版本：7 个 FC 全 W8，RMSNorm/softmax/residual FP16；attention FP16 + 3 个 MLP FC W8。后者与旧 Q4 mixed 同结构。采用自身新中文校准，不使用旧英文激活范围。
- 完整词表 128256 拆为 4×32064；4 个 W8 head 都测，不能以第一分片代表完整输出头。Head 前置 RMSNorm 内部 FP16，FC 输入输出边界使用校准后的 A16；外部 hidden 输入在 quant 文件未显式给出，须以板端 vip_query 为准，不能预先标成 A16。
- Head 的 16 个输入来自每条中文语句真实 16 层 FP32 传播的两处 hidden；8 校准 + 8 留出。保存全词表 argmax、top10。该测试不包含累计 NPU decoder 误差。
- 两个首层的量化输入和头输入分别保存原始 F32。运行时由同一 `put()` 转为 graph 实际输入精度；逐输出读取并保存 hidden/newK/newV。
- 每状态重复 3 次，第一遍不计 hot 中位数，后 2 遍计入；总 80 次 hot 首层测量。每 fixture 只有一个独立输入，不把重复测量算作独立质量样本。

执行顺序：`prepare.py` → `build_probe.py`。每个 Docker 容器最多 4 CPU/8 GB，逐个运行，F 盘至少 8 GiB 预留，额外 1 GiB 编译余量。代码无板卡访问。

板卡统一由主负责人在两个锁下运行。`quant_probe.c` 使用继承的 `board_probe.c` 类型转换，命令 `quant_probe network.nb fixture_dir output_dir -1` 执行 decoder，最后参数 0/1/2/3 执行对应头分片。各 variant 独立输出目录；`analyze.py --results <retrieved-dir>` 分析 raw 输出。必须确认返回码、40/16 fixture 完整、driver release 0、全词表合并后 argmax 再判断。

状态：Q8 六图（两首层 + 四 head）与 Q4 同中文控制首层，全部编译成功。Q8 六图合计 356,244,248 B；Q4 控制首层 55,286,632 B。已交板卡负责人串行测试，尚无 NPU 板测结果，不可声称精度通过或加速。

追加 Q4 控制：`build_q4_control.py` 只构建一个 all-FC W8 首层，产物在 `BUILD/q4-control`。相同 prompt/token ID/position/容量/校准算法/精度 flags，源变成 Q4_K_M；经过代码 assert 确认 token 和位置相同。对照图 SHA `d43ba5cd9ba14630981e41c7bab6f51f6ab927c0f8e560f36eb898ddf7d7edfc`。必须使用自己的 `q4-control/fixtures`，不可喂 Q8 的 activation/reference。此对照用于分离源权重精度与新中文校准的影响。

Q4 分析命令：`python analyze.py --results <q4-results-parent> --manifest q4-control/fixtures-manifest.json --build /opt/opi-lab/llama32-q8-quant-probe-20261023/q4-control`，其结果父目录包含 `layer0-allw8/`。

