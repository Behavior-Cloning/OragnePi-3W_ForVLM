# A733 NPU增量Transformer板端实验

2026-10-12。结论：已经在这块Zero 3W上跑通**带持久KV的Transformer decoder**，并扩到已下载SmolVLM-256M的前两个真实语言层。SDK、编译器和VIPLite链路可用。完整聊天模型尚未适配完成，网页仍使用原CPU后端。

## 实验对象与证据

- 随机小网络：两层，hidden=64、4个Q头、2个KV头、head_dim=16、FFN=192。32/128/512容量分别连续走到末尾，INT16输出通过初步误差门槛。
- 真实权重：`SmolVLM-256M-Instruct-Q8_0.gguf`的语言层0、1，hidden=576、9个Q头、3个KV头、head_dim=64、FFN=1536。这个语言模型共有30层，视觉部分未测。
- GGUF的Q8_0权重先解码为浮点，再交给ACUITY生成INT16/FP16；**并非NPU原生执行GGUF Q8**。SHA-256、RoPE参数、token片段和主机误差保存在`evidence-20261012/trained-manifest-c*.json`。
- 两段不同的有效BPE token序列分别用于校准和留出验证，重复至容量；不是大规模自然语言分布评测。验证输入来自实际embedding，第二层消费第一层的实际输出，KV由NPU实际输出持续回填。
- 主机参考使用PyTorch完整因果attention，只作正确性核对。板端CPU/NPU性能测试全都逐token追加KV，未做整段重算性能测试。

原始数据、每档5次的中位数、CPU线程数、API时间和history分段见[测量汇总](evidence-20261012/测量汇总.md)。该处“步/秒”仅为**两层hidden更新**，不是完整模型token/s，不能拿它直接和此前llama.cpp六模型报告比较。

## 与板上CPU的直接对照

全部使用同样的两个语言层、留出输入、固定容量图和逐步KV，测量时CPU/NPU任务顺序执行。下表CPU为单线程不绑核，误差参考为反量化权重的FP32计算。

| KV容量 | CPU FP32每步 | NPU INT16每步 | NPU FP16每步 | 观察 |
| ---: | ---: | ---: | ---: | --- |
| 128 | 4.086ms | 2.980ms | 10.234ms | INT16约1.37×；FP16比CPU慢 |
| 512 | 5.711ms | 6.743ms | 25.182ms | 两种NPU路径都比CPU慢 |

CPU 2/4线程本次没有超过单线程；另测绑定CPU6的单线程结果约4.117/5.703ms，与不绑核接近。图较小、每token串行，线程增加可能被调度/同步成本抵消，不能推出所有CPU模型都应该单线程。

当前NPU优势是短容量INT16以及释放CPU：INT16宿主进程约占单核9%，CPU单线程约占100%；这不是功率结论。128容量API本身的对照为CPU约3.857ms、NPU INT16约2.690ms，去掉大部分外层编排后方向一致。512容量API对照为5.470/6.178ms，同样未见加速。

因此没有证据支持“把当前完整陪伴模型搬到NPU一定更快”。而且CPU FP32与NPU INT16不是同精度；上述1.37×是当前两种软件/精度组合的层级实测比值，不能视为硬件等精度加速比，更不能替代CPU GGUF量化后端对完整模型的对照。

## 当前速度与精度的取舍

NPU INT16：128容量约2.98ms/步，512容量约6.74ms/步；相对L2误差分别约3.99%、2.98%。FP16：约10.23/25.18ms，误差约0.273%/0.336%。数字为本次5次运行中位数，属于相应软件/精度组合。

通过门槛为逐步最小余弦相似度≥0.99、整体相对L2≤5%，只是层级冒烟测试门槛。两层通过不代表30层误差不会放大，更不代表最终token排序、中文表达和陪伴人设无损。下一阶段必须与标准模型的最终logits/生成做独立对照；现有主机oracle也采用本项目的GGUF反量化与RoPE行重排，尚未与完整HF/llama.cpp的中间层做独立比对。

## KV具体怎样工作

每层输入：当前hidden、历史K/V、mask、当前位置RoPE cos/sin；输出：当前hidden_out及一个位置的新K/V。每个网络只加载、prepare一次，缓冲持续映射。每步取回NPU新K/V，按查询到的量化参数转换后写入对应位置，再供下一步使用。旧token的Q/K/V投影和MLP不重新计算。

已实现GQA、RMSNorm、RoPE、SwiGLU、持久KV、常驻网络及增量KV输出。INT16输出scale与下一次cache输入scale可能不同，运行器执行反量化/再量化，不假定可以直接字节复制。mask端口可被编译器保留为FP16，具体IO元数据随证据归档。

板端额外检查：把所有未用cache槽预填非零数值，128容量INT16/FP16输出均与零初始化逐元素完全一致；请求129步但容量128时均在运行前拒绝，退出码2。证据见`evidence-20261012/board-guards.json`。各次进程重新启动后误差结果一致，但尚未实现同一运行器进程内的对话重置API。

当前每步attention仍扫描整个静态容量，包括被mask的空槽，图内还可能有Concat/GQA展开复制。宿主只逻辑更新一个KV位置，但flush仍覆盖完整输入缓冲。这解释了**同一容量内history增加影响较小，容量从128改成512明显变慢**。CPU基线采用相同固定容量ONNX以对齐计算；这不能代表llama.cpp只访问有效history的最佳实现。

未实现：FlashAttention、PagedAttention、动态长度内核、会话并发、容量迁移、自动裁剪、批量prefill。容量用完明确报错；不自动覆盖旧人设或继续错误RoPE。

## 系统开销与测量限制

测试系统为Ubuntu Jammy、Linux 6.6.98-sun60iw2，VIPLite 2.0.3.2，NPU芯片ID `0x1000003b`，现有1008MHz配置。CPU保留原ondemand governor，没有修改网页或系统服务。CPU FP32基线为ONNX Runtime 1.23.2，单线程/2线程/4线程，另有绑定CPU6（A76）的单线程结果，均关闭线程池空转。

NPU进程CPU占用约一个核的4–10%，不是整机百分比。前两层的NPU驱动分配约11.7–18.0MiB，另有进程RSS等开销；不能只看几MiB RSS。测试完成驱动分配回到0，未见本轮缓冲遗留。CPU/NPU都与原网页服务并存，无法保证所有后台活动完全一致。

加载耗时只是新进程的网络加载/prepare，文件可以命中页缓存，不是断电冷启动。FP32 CPU使用Python编排，NPU使用C编排，所以总时间同时包含各自调度差异；另外列出两层推理API用时供判断。CPU RSS包含Python/ORT及图优化，不与NPU驱动分配单列直接等价。

没有功率计，因此没有实测W、J/token和能效提升结论。短时温度不能证明长期满载散热。也未据此宣称整模型优于当前CPU聊天后端。

## 复现

主机依赖：torch 2.7.0、numpy、onnx、onnxruntime、gguf 0.19.0、PyYAML。SDK见[获取记录](SDK获取记录.md)。转换器容器不联网，限制4核/8GiB内存，目标固定为`VIP9000NANODI_PLUS_PID0X1000003B`。

```bash
npu/.venv/bin/python npu/kv-cache/trained_probe.py \
  --model models/smolvlm-256m/SmolVLM-256M-Instruct-Q8_0.gguf \
  --output npu/results/trained-smol-c128 --capacity 128

python3 npu/kv-cache/compile_probe.py --source npu/results/trained-smol-c128 \
  --output npu/results/trained-smol-c128-l0-int16 --layer 0 --dtype int16
python3 npu/kv-cache/compile_probe.py --source npu/results/trained-smol-c128 \
  --output npu/results/trained-smol-c128-l1-int16 --layer 1 --dtype int16

python3 npu/kv-cache/run_board_probe.py --capacity 128 \
  --fixture npu/results/trained-smol-c128 \
  --layer0 npu/results/trained-smol-c128-l0-int16/export_nbg_unify/network_binary.nb \
  --layer1 npu/results/trained-smol-c128-l1-int16/export_nbg_unify/network_binary.nb \
  --tag trained-smol-c128-int16 --dim 576 --kv 3 --head-dim 64 --rope-base 100000

python3 npu/kv-cache/run_cpu_comparison.py --capacity 128
python3 npu/kv-cache/collect_results.py
```

SSH默认使用密钥；可从外部环境提供`SSHPASS`，代码/报告不保存密码。CPU脚本要求板上`/opt/opi-vlm/npu-kv-probe/venv`安装numpy 1.26.4和onnxruntime 1.23.2。本次系统缺少ensurepip，因此用`venv --without-pip`，将已下载的对应aarch64 wheels解压至该隔离环境site-packages；未改系统Python。把容量128改成512、dtype改成fp16可复现其他档位。

NBG、ONNX、SDK和权重按Git规则留本地；代码、版本信息、校验值和小型原始结果纳入Git。生成产物不能仅凭文件存在视为通过，须同时检查转换阶段退出码和板端误差门槛。

## 下一阶段

先确认30层及最终logits的精度，再考虑分层混合精度、容量分档和减少缓存复制。只有完整模型能稳定输出且同条件实测有收益，才接入网页或扩大到Qwen 1B以上。现有结果已经证明“这个NPU能执行带KV的真实Transformer层”；尚未证明“完整陪伴LLM已经得到加速”。
