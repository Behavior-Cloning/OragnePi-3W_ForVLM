# Llama 3.2 中文完整 NPU 实验

当前目标是 FuseChat-Llama-3.2-1B-Instruct 的完整 16 层中文文本推理。首版使用 Q4_K_M GGUF 解量化后的权重，注意力 FP16、MLP 三个矩阵 W8/A16、全词表 head FP16。此处 Q4 是源文件精度，不代表 NPU INT4，也不能与 Intern 主体 W8 直接声称等精度。Q8 源是独立候选，须先过单层数值门槛再决定完整编译。

## 实现范围

- 16 个 decoder 图和 4 个 head 图，覆盖全部 128256 词表；head 含 final RMSNorm。
- 首版 prefill 和 decode 都调用同一套 NPU 图，prefill 为逐 token 基线，无 CPU prefill。
- CPU 做 embedding 查表、RoPE cos/sin 值构造、KV 拷贝、调度和 greedy 选择；rotary 乘法、注意力、MLP、norm 和词表投影在 NPU。
- 容量 256，system 22 token；只允许复用 system KV，禁止完整题目缓存。三个冻结中文问题共 38/41/42 token。
- 自然生成上限 96；另测固定反馈 32 步，不能把固定反馈文本视为自然生成质量。

## 证据与命令

`cpu-sanity.json` 是真实 layer0 相对官方 HF 算术对照；`reference-q4.json` 是完整 16 层 FP32 源权重参考，并有全词表 raw logits。分层校准使用各层真实前置输出，每层独立确定量化参数，校准文本与三条质量问题分离。

`build_sequential.py --precision attn-fp16-mlp-w8 --execute` 顺序构建 20 个图并保留日志、量化元数据和 SHA。`deploy_run.py --stage-only --execute` 仅作带 SHA 的提前传输；完整 20 图 manifest 通过后，`deploy_run.py --execute` 才允许运行。两者均持 WSL hostlock 和板端 flock。rsync 支持中断续传和压缩，不建立额外模型 tar。

`board_run.py` 连续记录温度、RSS/PSS、系统剩余内存、swap、频率、整机与进程 CPU；进程 CPU 从 /proc/pid/stat 最后一个右括号之后解析。78°C 触发退出，正常结束必须验证 NPU driver bytes 归零。`summarize_board.py` 对原始事件和 full-vocab logits 汇总。

## 尚待实际结果

当前编译与传输不等于已完成端侧功能验证。最终是否可用取决于全模型 logits、中文自然回答和实测延迟。12 问家庭中文筛查见 `chinese-quality-cases.json`，通过三个功能问题后再扩大。语音、视觉、长期压力测试不在这一阶段。
