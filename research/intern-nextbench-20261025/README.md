# Intern NPU 下一轮统一基准（2026-10-25）

仅实现并完成无板卡静态检查；本目录自身不启动 SSH、flock 或板卡任务。控制器必须持有 Windows/WSL hostlock 和板卡 `/run/opi-npu-kv-benchmark.lock`。执行器仅检索，无真实电机。

## 控制器接口

板端准备本目录五个源码/fixture文件，以及控制器指定的 binary / graph。配置中的相对 fixture_path 相对于 board_benchmark.py；其他运行时路径为板端绝对路径。

```sh
python board_benchmark.py --config config.json --out /absolute/new/output --repeats 1 --phases performance,quality,diagnostic
```

完整重复将 `--repeats` 改3；省略时读取 JSON repeats（示例为1）。JSON quality、diagnostic控制默认阶段；CLI --phases显式覆盖。每个variant单独worker，退出后验证driver_bytes=0。输出目录禁止覆盖。

配置示例在 config.example.json。defaults提供graph、capacity、layers、vocab、prefill_batch、eos、affinity；variant可覆盖这些值，并提供name、binary、env、speed_profileflag。env用env命令传给worker，隔离继承OPI变量。配置中的capacity必须匹配实际图和ready容量；256容量足够本矩阵最高200输入+31次反馈，不能把1280图仅改参数冒充256图。

## 固定矩阵与计时

- 沿用此前aligned-inputs.json原始文件、tokenizer SHA、prefix/prompt token SHA和fixture IDs。
- 四格static64/128 × dynamic正文32/64，每格相同3场景；repeat1为12次、repeat3为36次。
- static包含system模板（正文59/123，system模板5）。动态聊天模板另8token；实际新计算40/72，不是32/64。
- 每次reset并仅准备对应system前缀；system准备在单独prefix_prepare_metrics中，排除TTFT。强制实际reuse=64/128、new=40/72。
- 性能阶段反馈固定32个token（默认ID100662重复），diagnostic=False。保留32个实际模型预测、runtime token时间、宿主观察时间、各阶段done字段、2–8/9–16/17–24/25–32输出位置的decode时间和速率。
- 固定反馈隔离各候选生成文本差异，是控制计算路径的性能样本；不能冒充自然对话或长文本质量。第一个输出来自prompt，随后31次decode使用固定反馈。
- 12条自然质量样本每variant独立运行，最多32输出、自然EOS，记录reply、top3、RAG耗时和与首variant的token一致率；这些是旧开发集，不是新holdout。
- 每格第一场景，单独4步fixed-feedback diagnostic，输出全词表FP32 logits；每文件4×151674×4=2426784字节。诊断耗时不纳入性能成绩。保存SHA、相对L2、max_abs和逐步top1一致率。比较第一variant，不是FP32 oracle。

## 驻留负载和观测

复用Frontend流式GTCRN/VAD/ASR等及Retriever；模型保持加载。Frontend以真实时间播放已有合成WAV，每次最多排队一条完整录音，因此这与逐16ms读取的小时harness缓冲语义不同。RAG在自然质量阶段调用，性能阶段仅常驻，不应声称测了持续并发RAG查询。

resources.jsonl每0.5s记录harness+resident两个PID、整机CPU/内存/温度/频率、音频frames/overruns/maxlag/stage_seconds、当前variant和phase。78°C停测；音频线程、监控异常会传播到主任务。无需图片、真实麦克风、电机。

## Profile与安全隔离

speed_profileflag可为none / stage_only / npu_query，作为报告标签。OPI_PROFILE_STAGES=1允许进入速度对照，其计时开销应与baseline比较。OPI_PROFILE_NPU只允许另一次 --phases diagnostic；任何性能阶段包含该env会直接拒绝。SDK查询开销不能当优化速度。

已检查复制的ResidentWorker及resident_session.c协议：feedback_count与diagnostic flag独立；仅flag开启时打开和写入logits文件；有feedback时EOS不提前终止。性能/自然阶段每次检查diagnostic目录为空。若接入新二进制改变协议，必须重新审核与烟测。

结果：events.jsonl、resources.jsonl、meta.json、config.json、fixtures.json、provenance.json、每variant原生日志及diagnostic子目录。源码/fixture通过哈希记录，图目录记录nb文件名大小；完整图哈希由控制器部署manifest负责。

异常finally尽量close并记录driver释放；若worker已崩溃只能abort，不能伪造driver0。完整成功须meta.complete、每variant closed.driver_bytes=0且音频无错误。控制器仍需独立post-health。

## 后续候选加载校验

可在variant设置expected_graph_sha256，格式为{相对主graph路径或绝对路径: SHA256字符串}。embedding/overlay可使用绝对路径。加载前以4MiB块流式校验，canonical path/size/mtime一致时复用一次哈希结果；写graph-provenance.json。ready必须匹配layers与capacity。未列出的文件不在校验覆盖范围。更多风险见CANDIDATE_REVIEW.md。analyze_next.py支持true_tail8计量，seconds包含在prefill_seconds，禁止重复相加；driver与PSS分列。
