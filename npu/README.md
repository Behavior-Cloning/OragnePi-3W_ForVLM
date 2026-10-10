# Orange Pi Zero 3W NPU 适配记录

最新研发状态（2026-10-10）：[KV cache原型与共享内存实测](kv-cache/README.md)。主机逐token正确性和16/64/256MiB板端VIP缓冲分配通过；NPU decoder编译与完整LLM加速尚未完成。下面保留先前的驱动和预装模型验证记录。

## 已验证状态（2026-10-08）

- 系统：Orange Pi Ubuntu Jammy Server 1.0.2，内核 `6.6.98-sun60iw2`
- SoC：Allwinner A733，NPU 为单核 Vivante VIP9000，厂商标称约 3 TOPS；
  官方产品简表没有明确该 TOPS 数字的精度口径，因此不直接拿它与别家的
  INT8/INT4 TOPS 等号比较
- 内核模块：`vipcore` 1.13.0，模块 vermagic 与当前内核完全一致
- 设备：`/dev/vipcore`（字符设备 199:0）
- 用户态：VIPLite 2.0.3.2、`libVIPhal.so`、`libNBGlinker.so`
- 模型格式：预编译 NBG（本镜像示例使用 `.nb` 扩展名）
- NPU 频率：可用 492/852/1008 MHz，当前 performance governor 为 1008 MHz

板上自带的 `/opt/vpm_run` 样例已实测通过，返回 `vpm run ret=0`；其
`224x224x3` 小网络的 profile inference time 约为 `2.7 ms`。

板上自带的 `/opt/yolov5` 也已实测通过。输入官方 dog 示例图时检出：

- dog：82%
- truck：69%
- bicycle：47%

整个进程（启动、加载、读图、OpenCV 预处理、NPU 推理、后处理、写图）
墙钟约 `413 ms`，不能把它当成纯 NPU 延迟。输出保存在
`results/yolov5-dog-result.png`。

## 板上命令

项目中的 `scripts/opi-npu-test` 已部署到板上的
`/usr/local/bin/opi-npu-test`：

```bash
opi-npu-test status
opi-npu-test smoke 1

mkdir -p ~/npu-output
cd ~/npu-output
opi-npu-test yolo
# 输出：~/npu-output/result.png
```

也可以换自己的图片：

```bash
cd ~/npu-output
opi-npu-test yolo /绝对路径/图片.jpg
```

预装 YOLO 程序按上游 OpenCV 的 `libopencv_*.so.4.5` 名称链接，而 Jammy
提供的 ARM64 包使用 `libopencv_*.so.4.5d`。为避免改系统全局库，本次只在
`/opt/yolov5/compat-lib/` 建了 3 个局部兼容软链接，并由启动脚本设置
`LD_LIBRARY_PATH`。

## 自己的模型如何上 NPU

这不是 ONNX 直接在板上运行的通用后端。典型流程是：

1. 在 x86 Ubuntu/Docker 主机准备固定输入形状的 ONNX。
2. 优先使用当前官方文档中的 `ubuntu-npu:v2.0.10.2` ACUITY 容器，A733
   选择 NPU version `v3`、NPU SW `v2.0`。
3. 目标必须是 `VIP9000NANODI_PLUS_PID0X1000003B`（本机芯片 ID 实测为
   `0x1000003b`）。
4. 生成 NBG/`.nb`，连同量化参数和预/后处理代码部署到板上。
5. 先与 ONNX/CPU 结果做余弦相似度、Top-k 或任务指标校验，再测速度。

不要随意替换 `vipcore.ko`、`libVIPhal.so` 或 `libNBGlinker.so`。当前驱动、
用户态库和 NBG 编译器存在严格版本耦合；升级内核也可能让模块 ABI 失配。

社区参考工程（实验性质，不是 Orange Pi 官方 SDK）：

- <https://github.com/petayyyy/a733_npu_driver>
- <https://github.com/wuclark/OrangePiZero3W-GPU-VPU-NPU>

第一个工程已浅克隆到 `reference/a733_npu_driver/`，固定到检查时提交
`aae9287f6d0de4af2f3e253343c58a78e2f3d04f`，用于下一阶段 ONNX→NBG
转模和轻量 VLM/LLM 实验；它不会自动安装或替换板上的驱动。该仓库默认转换
容器较旧且部分路径不是本机路径，不应整仓照搬部署；需要的 runner 应单独取出，
并针对本机使用 `/usr/include` 和 `/usr/lib`。

当前转换资料：

- <https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/cubie-acuity-env>
- <https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/cubie-acuity-usage>
- <https://docs.radxa.com/en/cubie/a7z/app-dev/npu-dev/model-zoo>
- <https://docs.radxa.com/cubie/a7z/app-dev/npu-dev/voice-assistant>

## 建议的应用架构

- 目标检测、分类、轻量分割、姿态估计：优先放 NPU。
- 图片问答：先用 NPU 做视觉编码/检测，再由 CPU 小语言模型生成回答。
- 语音识别：先走 CPU `whisper.cpp` tiny/base；官方同 SoC 的离线语音参考可按
  `16kHz WAV → KWS → Zipformer ASR → Matcha/HiFi-GAN` 分阶段移植。Zero 3W
  当前没有接入麦克风，先用 WAV 回归，再接 USB 麦/USB 声卡。
- NPU 只有一个核心；摄像头、KWS、ASR、TTS 若共用 NPU，应由一个常驻服务排队，
  不能把它们当成可同时并行的多个加速器。
- 电机闭环、急停、编码器等硬实时任务：交给 STM32/ESP32，OPI 负责高级感知和决策。

当前板空闲温度约 63–66°C；进行持续摄像、NPU 和 CPU 混合负载前应安装散热片，
并确认风扇实际转动。
