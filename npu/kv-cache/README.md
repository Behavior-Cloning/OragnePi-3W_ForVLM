# A733增量KV cache适配：第一阶段

2026-10-10。状态：**主机计算图正确性通过；板上共享内存分配通过；NPU decoder未编译、未运行，完整LLM适配未完成。** 不替换现有CPU网页服务。

## 已实现

`probe.py`为新写的两层小型decoder验证工具，使用确定性随机权重，不是训练好的聊天模型。固定形状不等于不能实现KV缓存：把历史K/V作为显式输入，输出当前token的K/V增量，即可保留计算状态；关键还要看厂商编译器是否支持这些算子和数据精度。

- 每步只为一个新token计算Q/K/V和MLP，历史token不重新做投影。
- 历史K/V缓冲区固定容量，宿主每层每步只更新一个位置。
- 包含GQA（4个Q头、2个KV头）、绝对位置RoPE、RMSNorm、SwiGLU。
- 图中将新K/V临时拼到固定历史缓冲末尾，attention mask屏蔽所有未写入位置。真实绝对位置由RoPE指定，与物理槽位置分离。
- 图只输出hidden和单token的K/V，不输出整个历史缓冲；主机两层总计每token只写512字节FP32缓存数据。
- 超出容量会明确报错；不会悄悄丢弃人设或用错误位置继续。重置会清空缓存和位置。

这还不是PagedAttention、FlashAttention或高效批量prefill。固定容量attention每步仍访问完整容量（无效位置被mask）；图内Concat可能产生临时复制。真实NPU延迟、缓存刷写成本和长容量吞吐需编译后实测。后续容量分档可减少短历史访问量，但不能在没有KV迁移验证时宣称已支持。

## 正确性证据

主机环境：PyTorch 2.7.0、ONNX 1.21.0，ONNX ReferenceEvaluator（CPU）。如安装onnxruntime，脚本优先使用其CPU后端。

| 容量 | 逐token核对 | 与全序列因果计算最大绝对误差 | 未用槽污染检查 | 溢出拒绝 | 重置 |
| --- | ---: | ---: | --- | --- | --- |
| 32 | 32 | 5.0664e-7 | 通过 | 通过 | 通过 |
| 128 | 128 | 4.7684e-7 | 通过 | 通过 | 通过 |

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

## 当前阻塞及下一关

Radxa官方A733说明要求ACUITY镜像`ubuntu-npu:v2.0.10.2`。本机尚未安装；官方Allwinner共享链接可访问说明页，但本次直链GET得到HTML、匿名Sharing API返回119而非可下载ZIP。没有绕过登录，也没有误把网页当模型工具包加载。

因此没有可运行的NBG生成器，无法确认新图的NPU编译、量化准确性和加速效果。需要获得匹配的官方工具包，或另一个已授权下载地址。入口：<https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/cubie-acuity-env>；共享：<https://netstorage.allwinnertech.com:5001/sharing/Mh23BhPHq>。

环境补齐后的顺序：

1. 编译这两个小decoder图，检查输出K/V和输入缓存的量化scale、布局；不同量化参数不能直接memcpy回缓存。
2. 常驻VIP网络，持久映射缓存，只更新新槽，正确flush/invalidate；逐token与CPU对照，检验误差累积。
3. 测量NPU算子、数据复制和端到端时间，确认确有收益再扩到真实小模型。
4. 真实模型增加批量prefill与单token decode分图、容量分档、采样和多轮缓存管理；之后才考虑1B及更多模型。

KV cache不会自动修复已有量化精度或编译器崩溃问题。现阶段不承诺3B上NPU、也没有NPU上的FlashAttention/PagedAttention结果。
