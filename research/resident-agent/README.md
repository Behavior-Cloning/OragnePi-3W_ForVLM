# 板端常驻语音 → 记忆 → NPU 文本模型实验

本目录是独立实验入口，保留 `--mode para`、`--mode stream`、`--mode both` 两条 ASR 路径。源码在 Windows F 盘管理；Mac 只接收报告。**没有实现视觉 encoder/projector NPU 部署，不是完整 VLM 验收。**

## 运行形态

GTCRN、Silero VAD、KWS、Paraformer、Streaming Zipformer、tokenizer、SQLite 记忆与 InternVL3-1B 的 24 层 NPU 文本模型均留在板端。GTCRN/KWS/VAD 在模型生成期间继续处理模拟静音。双 ASR 模型同时加载、轮流识别同一批 WAV。每个回合先播放固定唤醒正例，再按真实时间播放命令；不模拟真人麦克风、打断、多人抢话或扬声器回声。

CPU6：语音前端；CPU7：离线 ASR、分词、记忆与控制；CPU4/5：NPU 宿主单线程的亲和集合（并非两个工作线程）。NPU decoder/head 之外仍有 CPU 嵌入查表、采样和协议开销。原 CPU llama-server 未被替换，不参与推理但保留背景内存占用。

输入采用 exact GGUF tokenizer 和 ChatML：64-token system + 动态记忆/历史/用户输入。动态预算上限轮换 64/128/256，不把上限冒充每轮实际 token 数。输出最多 48 token，必须同时看 `finish_reason`。`--history-turns 0` 仅保留 system＋召回记忆＋本轮输入。

## 板端依赖与文件

实验根目录：`/opt/opi-vlm/resident-agent-20261019`。

- `agent_replay.py`、`resident`、`resident_worker.py`、`memory_lab.py`、`prompt_budget.py`。
- `resident_worker.py` 来自仓库 `npu/kv-cache`，两个记忆文件来自 `research/memory-prototypes`。
- `tokenizer.json`、`tokenizer-meta.json` 由本目录 `export_tokenizer.py` 从原 1B Q8 GGUF 导出；源 SHA256 为 `e87f98e2945c91da1a324034143f2f3bad014dcf6b34d440fcf885c00704378f`。
- `deps`：ARM64 tokenizers 0.22.2 wheel 解包；语音 NumPy/sherpa 依赖复用 `/opt/opi-vlm/audio-lab-20261018/deps`。
- 语音模型与 worker 复用 `/opt/opi-vlm/audio-lab-20261018`、`/opt/opi-vlm/asr-small-20261018`、`/opt/opi-vlm/audio-streaming-20261018/kws`；原始 Python worker 对应仓库 `research/audio-models`、`research/asr-small`。
- NPU 图和 embedding：`/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018`。24 层、容量 1280、dim896、KV2、head_dim64、fused2、head_shards4、prefill16。
- `speech-texts.json`、`speech/01.wav` 至 `12.wav` 是 Windows Huihui 合成生活场景；唤醒词样例使用已有 `pos-alone-normal.wav`。权重、音频、依赖和 sqlite 不加入 Git。

必须先获取仓库 `docs/HANDOFF.md` 规定的 host lock，再用板端 `/run/opi-npu-kv-benchmark.lock` 串行执行。长任务由 WSL 控制器持有 host lock；保留 WSL 前台 keeper，避免会话结束导致 WSL 退出。不得绕过锁去并发跑第二个 NPU 测试。

在上述依赖已经部署的板端、持锁后执行示例：

```sh
cd /opt/opi-vlm/resident-agent-20261019
python3 agent_replay.py --mode both --seconds 1800 --turns 3 --silence 1.6 --out unique-run-name
```

每个输出目录必须新建，防止混入旧结果。结束后复制 `meta.json`、`turns.jsonl`、`resources.jsonl` 到 Windows，分别命名为 `NAME-meta.json` 等，然后运行：

```sh
python3 research/resident-agent/summarize.py /mnt/f/Files/OPi-Lab/results/resident-agent-20261019 NAME
```

`endpoint_sweep.py OUTPUT` 是离线样本索引扫描，不能将其耗时称为端到端延迟。通过后仍须 paced replay 验证。当前输出的最短阈值只是这 12 条合成样本上的开发选择，未经过独立真人/家庭噪声测试。

## 度量边界

- 首字为板端 tokenizer 解码得到首个非空可见文本；完整回复为生成退出，不含 TTS 播放。
- `vad_to_first_s` 从真实 VAD endpoint 事件算；`segment_proxy_to_first_s` 从 VAD 语段末样本点算。后者是代理值，没有人工冻结音素终点。
- WAV 末尾含静音，`file_end_to_first_s` 只能作文件回放指标，不能冒充真人说完的延迟。
- 流式 ASR 的说话期间计算不属于端点后的残余等待；端点之后的零输入 drain 计算单独记录。
- `stage_timing_version=2` 对流式 drain 去重并记录每轮前端模型累计计算。较早长测的 VAD timer 包含 drain，不能与 stream timer 相加。
- CPU 总量按 8 核合计 100%，同时给单核百分比；RSS/PSS 与 driver 映射不可直接相加当物理占用。
- replay 无丢帧机制，积压会记录 lag；这不等于真实音频设备零丢帧。
- 0.25 秒/0.8 秒阈值的截句样本保留为失败证据，不纳入成功延迟统计。

## KV 修复验证

`validate_prefix.py` 依次覆盖变化发生在 64/128 token 后、短 prompt、重复 prompt、无共同前缀。复用路径和 reset 路径 dump 全 logits，比较 SHA 与逐字节一致性。它是数值回归，不是自然回答质量或相对原 GGUF 的完整量化质量验收。

## 对照缓存控制修正

30分钟长测中两条ASR交替处理相同语音并共用NPU session，后一条可能命中前一条prompt的额外KV，因此路径延迟比较有偏，保留为混合缓存稳定性/资源证据。后续优化短测在每轮唤醒前调用system-only prepare，恢复64-token固定前缀；新请求逐轮断言/检查reused_tokens=64。prepare在命令播放前，不属于说完后延迟，单独保存prefix_prepare_seconds。

## 流式与非流式横向时序图

`plot_timeline.py ROWS OUTPUT --font FONT_PATH` 从 `memoryonly-both-turns.jsonl` 取 source_id=2 的两条配对实测，核对prompt token和回答完全相同后绘制PNG/SVG/PDF。命令输入起点为0，system/KWS准备在图外；说话末尾用VAD段末代理，非人工标注。前端以斜线运行窗口表示，不能当作逐帧API追踪或连续CPU忙碌。端点、首字、结束为实测锚点；ASR和记忆边界根据耗时回推，保留调度误差说明。该图是单个真实配对样本，不是拼接P50。
