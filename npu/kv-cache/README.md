# A733增量KV cache适配

最新decoder开发见[Decoder融合与GQA优化](Decoder融合与GQA优化.md)：500M/1B全部语言层已实现双层图与紧凑GQA，保留原始单层图和量化配置作为对照。速度、输出一致性与温度条件分开记录，不将短历史开发实验当作最终CPU/NPU验收。

更新（2026-10-12）：SDK与板端链路已跑通，SmolVLM-500M完整语言部分的32层、输出头及预测token回送已验证，可连续生成英语文本。详见[全深度生成与算子核对](全深度生成与算子核对.md)；六模型前两层结果见[层级报告](六模型NPU层级实测.md)。初始256M两层报告保留在[早期板端实验](NPU增量Transformer实测.md)。

最新状态：[Q8 完整模型优化进展](Q8完整模型优化进展.md)。500M/1B 均已有 W8 为主、少量敏感 MLP 保留 FP16、输出头 W8 的完整生成候选，问候和陪伴开发输入通过数值门槛；速度约 6–7 token/s，精确重复成绩见报告。按用户要求，接下来增加提速投入，对比对应 CPU Q8；未超过 CPU，独立留出、长历史和持续运行未验收。此前定位过程保留在 [INT8修复报告](INT8误差定位与修复.md)。极小输入 RMSNorm 仍有 FP16 下溢边界，不能宣称全输入通过。网页仍为原 CPU 后端。下面保留 10 月 10 日主机与共享内存验证，勿将其当成完整模型速度。

## 已实现

`probe.py`为新写的两层小型decoder验证工具，使用确定性随机权重，不是训练好的聊天模型。固定形状不等于不能实现KV缓存：把历史K/V作为显式输入，输出当前token的K/V增量，即可保留计算状态；关键还要看厂商编译器是否支持这些算子和数据精度。

- 每步只为一个新token计算Q/K/V和MLP，历史token不重新做投影。
- 历史K/V缓冲区固定容量，宿主每层每步只更新一个位置。
- 包含GQA（4个Q头、2个KV头）、绝对位置RoPE、RMSNorm、SwiGLU。
- 图中将新K/V临时拼到固定历史缓冲末尾，attention mask屏蔽所有未写入位置。真实绝对位置由RoPE指定，与物理槽位置分离。
- 图只输出hidden和单token的K/V，不输出整个历史缓冲；主机两层总计每token只写512字节FP32缓存数据。
- 超出容量会明确报错；不会悄悄丢弃人设或用错误位置继续。重置会清空缓存和位置。

这还不是PagedAttention、FlashAttention或高效批量prefill。固定容量attention每步仍访问完整容量（无效位置被mask）；图内Concat可能产生临时复制。VIPLite运行器目前flush完整输入缓存，只有逻辑KV更新是单位置。后续容量分档可减少短历史访问量，但不能在没有KV迁移验证时宣称已支持。

## 正确性证据

主机环境：PyTorch 2.7.0、ONNX 1.21.0，ONNX ReferenceEvaluator（CPU）。如安装onnxruntime，脚本优先使用其CPU后端。

| 容量 | 逐token核对 | 与全序列因果计算最大绝对误差 | 未用槽污染检查 | 溢出拒绝 | 重置 |
| --- | ---: | ---: | --- | --- | --- |
| 32 | 32 | 5.0664e-7 | 通过 | 通过 | 通过 |
| 128 | 128 | 4.7684e-7 | 通过 | 通过 | 通过 |

随后安装ONNX Runtime，在其CPU执行器独立复核32/128/512三个容量，共672步全部通过；各档最大绝对误差均4.7684e-7。记录见`host-validation-ort.json`。这些数据仅为主机验证；10月12日新增的NPU误差和速度独立记录在板端报告。

参考计算一次处理完整序列并使用因果mask；增量计算由独立ONNX执行器运行，逐步累积两层KV。每步hidden与对应完整计算位置比较。特意将未写入缓存填入非零污染值，确保mask有效。误差门槛atol/rtol均2e-5。

运行：

```bash
python3 npu/kv-cache/probe.py --output npu/results/kv-cache
```

需要numpy、torch、onnx；onnxruntime可选。生成静态ONNX（opset13）、四档位置校准样本、输入输出契约和验证JSON。生成物在Git忽略目录，可由源码重建；小型结果摘要另保存在本目录`host-validation.json`。

## NPU共享内存实测

Zero 3W的6GB是CPU/NPU共享的系统LPDDR5，不存在独立6GB显存。Linux实测MemTotal=5993660KiB，约5.716GiB；检查时可用4608720KiB，约4.395GiB，这是动态值，不能全当作NPU可用模型预算。

`memory_probe.c`直接调用板上VIPLite，不修改驱动。依次申请16/64/256MiB缓冲，每次映射后逐页写入/校验、flush、释放；同时仅存在一块测试缓冲。

| 请求 | 实际分配字节 | 校验错误 | flush状态 |
| --- | ---: | ---: | ---: |
| 16MiB | 16777216 | 0 | 0 |
| 64MiB | 67108864 | 0 | 0 |
| 256MiB | 268435456 | 0 | 0 |

结束驱动mem_profile显示当前使用0，分配/释放次数均9。证明至少该规模的VIP缓冲可用；**256MiB只是本次测试上限，不是硬件显存容量或最大分配量。** 参数contiguousSize/reservePhysicalSize为0不能解读为NPU没有内存。片上SRAM/网络memory pool不等于独立显存，本次未核定片上SRAM物理总量。

板上构建和运行：

```bash
cc -O2 -Wall -Wextra memory_probe.c -lNBGlinker -lVIPhal -o memory_probe
timeout 30 ./memory_probe
```

## 性能边界

厂商标称NPU约3TOPS、板上最高频1008MHz。本次未核实TOPS精度口径，不能直接换算token/s。此前预装224×224小网络约2.77ms仅是该小网络指标。

社区报告固定窗口SmolLM2-135M约20.7 token/s、360M约8.4 token/s，均W32、INT16、无KV，非本板实测，不能作为本原型的加速结果。

推理权重、KV、临时图缓冲及系统服务共同占内存；INT16权重约2字节/参数，1B仅权重即约2GB，3B约6GB，当前6GB整板难以容纳完整3B INT16及运行开销。INT4理论上更小，但此A733工具链没有经本项目验证可用的完整LLM INT4路径。

KV容量公式（每会话）：`2 × 层数 × KV头数 × head_dim × token容量 × 每元素字节数`。GQA减少KV头数可省内存；不能把“模型1B”直接换算成固定KV大小。

## 编译环境与下一关

Radxa官方A733说明要求ACUITY镜像`ubuntu-npu:v2.0.10.2`。10月10日共享下载受阻；10月12日经用户授权从同事电脑获取工具包并校验、加载成功，见[SDK获取记录](SDK获取记录.md)。

已使用ACUITY 6.30.22和Vivante IDE 5.11.0，编译目标为`VIP9000NANODI_PLUS_PID0X1000003B`。无需升级板上驱动。工具文档：<https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/cubie-acuity-env>。

当前验证范围及后续顺序：

1. 已完成小decoder和真实前两层编译，运行时查询布局及量化scale；以反量化/再量化写回KV，未直接假定scale相同。
2. 已完成网络常驻、缓冲常驻、单位置更新、flush/invalidate与误差对照。尚未实现动态容量迁移和运行中缓存淘汰。
3. 六模型均已尝试前两层，含精度失败档位；500M和1B已完成全部语言层＋LM head及实际token反馈。最新样例logits与FP32参考核对保存在Q8优化报告，不能凭少量文本认定聊天无损。
4. 当前优先提速：运行器支持`--decoder-group`实验模式，直接共享相邻层FP16 hidden并批量提交；`benchmark_submission.py`验证完整logits逐字节一致，再进行重复测速。尚不是计算图融合。后续方向为decoder内部布局/算子、批量prefill与decode分图、容量分档、图融合和多轮缓存管理。

KV cache不会自动修复已有量化精度或编译器崩溃问题。现阶段不承诺3B上NPU、也没有NPU上的FlashAttention/PagedAttention结果。
