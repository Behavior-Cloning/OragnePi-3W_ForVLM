# 端侧3B选型、AX650N生态与A733 NPU可行性

核查日期：2026-10-10。门槛为能运行约1B模型，并在约3B模型decode阶段达到至少5 token/s，最好10 token/s。当前仅文字陪伴场景。

## 选型结论

公开数据支持把RK3588、RK3576、AX650N、Jetson Orin Nano Super列为候选。它们不是同模型、同量化、同history的横向排名；达标指公开测试中的模型和配置，不保证4K/8K长历史下仍达到5–10 token/s。建议实际验收另设约128/2048/4096输入token三档、batch=1、KV真实追加、128输出token、持续运行与整板功耗。

| 平台/具体板型示例 | 约1B支持 | 约3B公开decode结果 | 证据及限制 |
| --- | --- | --- | --- |
| RK3588：Orange Pi 5系列、Radxa ROCK 5系列（建议8GB起） | TinyLlama1.1B W8A8：24.43 token/s；Qwen2.5-1.5B：16.69 | TeleChat2-3B W8A8：10.18；Qwen2.5-VL-3B W8A8：8.66 token/s | RKLLM官方芯片平台基准；纯LLM输入128、输出64、CPU/NPU最高频；VLM一行采用单独短输入条件。具体板卡还受频率、内存、散热影响 |
| RK3576：Radxa ROCK 4D、Orange Pi 4A等（购买时核对SoC，建议8GB） | TinyLlama1.1B W4A16：19.71 token/s | TeleChat2-3B W4A16：8.97；Qwen2.5-VL-3B W4A16：7.87 token/s | 同官方基准；不能据此认为RK3576硬件全面超过RK3588，不同量化路径影响很大 |
| AX650N：爱芯派Pro、AX650N DEMO Board（核对内存，优先8GB及以上） | 官方提供约1B InternVL等适配项目 | Qwen2.5-VL-3B：6.3；SmolLM3-3B：5.83 token/s | AXERA官方demo发布值，非本机复测；缺乏统一长history、整机功率和量化条件，不能把前者语言部分直接称为INT4实测 |
| Jetson Orin Nano Super 8GB | 可运行受支持的小模型；本次未找到同表约1B精确值 | Llama3.2-3B INT4，MLC：43.07 token/s | NVIDIA官方发布值；不是本项目Qwen3B同条件测试，也不是4K长历史保证，性能余量较大 |
| 当前A733 Zero 3W 6GB（CPU） | InternVL3-1B短历史约20.32，约3733输入token时9.93 | Qwen2.5-VL-3B Q4短历史5.42，约3733输入token时2.64 | 本项目实测；短历史勉强触线，长对话未达5 token/s |

不把未核实的树莓派CPU、各种视觉NPU卡直接列为3B达标产品。Orin主要用CUDA GPU进行LLM加速，不是与AX650N相同的NPU部署路径。未实测上述候选整板功耗，不做J/token排名。

## AX650N生态具体怎样

爱芯元智是芯片与工具链厂商；爱芯派Pro是AX650N开发板产品。爱芯派2则是AX630C，官方规格为双A53、4GiB LPDDR4X，不应把Pro/AX650N结果套给它。商品名相似不能代替SoC、内存和BSP版本核对。

工具链由几部分组成：

1. Pulsar2：把受支持的模型导出、量化、编译成`.axmodel`，不是拿现有GGUF文件直接运行。
2. AX Engine / pyaxengine：板上C/C++或Python调用运行时；编译器、BSP和模型包版本需要匹配。
3. AX-LLM：厂商维护的LLM应用项目，提供命令行、KV缓存、多轮聊天及OpenAI兼容服务。当前读取的说明列出AX650A/N SDK≥3.6.2，支持Qwen2.5/Qwen3、MiniCPM、SmolLM2、Llama3等，另有多个VLM适配。
4. 预编译ModelZoo及模型专用仓库：可避免自己从零转模；模型的可用上下文长度、prefill档位受具体编译包配置约束，不能任意拉大网页滑块。
5. AXCL：PCIe加速卡/模块的主机访问生态，不等同于独立开发板。若选择加速卡，还需要兼容的主机、接口与驱动，不能假设接到当前OPI即可工作。

官方组织还维护视觉、ASR/TTS相关项目，例如zipformer、CosyVoice、Kokoro等；这说明生态有对应开发入口，不等于这些任务能在某一板卡上同时低延迟运行。陪伴机器人还需测试ASR/LLM/TTS争用NPU与内存的情况。

性能证据：

- Qwen2.5-VL-3B官方README报告Language Model Decode=6.3 token/s；其Vision Encoder表的780ms不当作纯文字首字延迟使用。
- SmolLM3-3B报告每步171.375ms，即5.83 token/s；示例prefill TTFT约8.29s，但输入长度未统一，不能横向比较首字延迟。示例文本本身也有重复现象，跑通不代表中文陪伴质量通过。
- InternVL3.5-1B GPTQ INT4项目根据层与后处理耗时估算28.09 token/s；它不是端到端网页实测，且README支持多个芯片，测试条件未完整列明，不作为精确AX650N板卡购买保证。

判断：比目前找到的A733语言模型NPU实验更接近可用应用，有厂商维护的缓存与服务端；但仍依赖特定转模路径和二进制SDK，不像通用CUDA/GGUF那样可以任意换模型。买前要求卖家演示目标模型在2K/4K历史下追加聊天，并提供版本、真实TTFT和功率曲线。

## 当前Zero 3W的NPU检查与试跑

本次真实连接192.168.88.29，调用已部署的`opi-npu-test status`和`opi-npu-test smoke 3`，没有替换驱动或更改网页模型。

| 项目 | 本次结果 |
| --- | --- |
| 内核 | 6.6.98-sun60iw2 |
| 设备 | /dev/vipcore存在，chip ID=0x1000003b |
| 驱动与运行时 | vipcore 1.13.0 / VIPLite 2.0.3.2 |
| NPU频率 | performance，1008MHz |
| 检查时温度 | 46.1°C |
| 示例模型 | 板上预装224×224×3输入、2元素输出的小网络，并非LLM |
| 三次profile inference | 2761 / 2784 / 2757微秒，算术均值约2.767ms |
| 结果 | 三次均成功，vpm run ret=0 |

这证明硬件、驱动与已编译NBG调用正常，不能证明当前LLM有加速收益。没有同模型CPU对照，不能报告加速倍数或省电百分比。

已检查现有A733社区工程及在线同名说明：

- 使用ACUITY把固定形状ONNX编译成NBG，运行于VIPLite；没有把GGUF直接推送到NPU的通用开关。
- 已公开的SmolLM2-135M/360M方案为INT16、固定窗口、每token重算窗口，**未实现KV cache**。报告W32/W64可用，W≥128输出质量有问题。此限制属于该实现及转模路径，不是证明A733硬件原理上无法实现KV缓存。
- 社区报告135M/W32约20.7 token/s、360M/W32约8.4；这些不是我们板上LLM实测，极短上下文也不满足当前陪伴场景。
- 该工程报告1.7B的NBG生成阶段崩溃；不能把“小型网络能运行”外推为1B/3B完整LLM已适配。
- 本机Docker镜像列表没有所需ACUITY转换镜像；现有参考目录没有可用的LLM NBG/ONNX产物。当前缺少目标模型的KV增量图、兼容编译环境及运行器，因而本次未完成1B/3B NPU推理或CPU/NPU对照。

用户此前明确只测试KV增量模式，因此本次没有另跑无KV、32-token窗口的完整重算LLM演示冒充加速结果。没有启动长时间转换或后台LLM试验。

## 是否值得继续把LLM放NPU

在拥有成熟LLM运行时的平台（例如RKLLM、AX-LLM），有可能降低CPU占用、提高吞吐或能效，但仍要同模型、相近量化质量、相同history对照。不能用不同芯片上的不同模型速度推导当前A733的收益。

对现有A733，完整1B/3B NPU语言解码属于新的适配研发工作：需把各层导出为prefill/decode图，让KV作为可更新状态、校验RoPE/attention/RMSNorm/量化误差，控制CPU↔NPU数据复制，再对照本次CPU基线。这不是一个部署参数。阶段验收应先做小层块的KV正确性，然后真实模型多轮正确性，最后性能和功率；目前缺少第一阶段所需的兼容转模环境。

若目标是尽快得到陪伴机器人，优先继续CPU语言模型；未来视觉/语音可按已适配算子与模型分别考虑NPU。仅加速视觉编码器不会提升纯文字decode token/s。

## 本次核对的来源

- [Rockchip模型及工具链](https://github.com/airockchip/rknn-llm)
- [Rockchip官方性能表](https://github.com/airockchip/rknn-llm/blob/main/benchmark.md)
- [NVIDIA Orin Nano Super发布与性能表](https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/)
- [AX-LLM](https://github.com/AXERA-TECH/ax-llm)
- [Qwen2.5-VL-3B AXERA demo](https://github.com/AXERA-TECH/Qwen2.5-VL-3B-Instruct.axera)
- [SmolLM3-3B AXERA demo](https://github.com/AXERA-TECH/SmolLM3-3B.axera)
- [InternVL3.5-1B GPTQ INT4 AXERA demo](https://github.com/AXERA-TECH/InternVL3_5-1B_GPTQ_INT4.axera)
- [爱芯派2官方规格](https://github.com/AXERA-TECH/axera-pi2-docs/blob/main/source/doc_introduction.md)
- [A733社区LLM实现与限制](https://github.com/petayyyy/a733_npu_driver/blob/main/docs/03-run-llm-npu.md)
- [本项目此前NPU适配](../npu/README.md)
- [本项目CPU KV基准](benchmarks/2026-10-10-kv-fan/KV缓存增量性能报告.md)

网页/仓库会更新；外部数据仅作为此次购买筛选证据，没有在本地对应硬件上复测。
