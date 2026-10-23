# FuseChat 中文 Llama 1B：完整 NPU 功能与性能验证

2026-10-23。当前已完成全部 16 层 decoder 与 4 个完整词表 head 的 NPU 编译、逐文件 SHA 校验、板卡部署；第一轮实际推理正在执行。尚未回传的性能与精度不填估计值。

## 模型和精度

首组源是 FuseChat-Llama-3.2-1B-Instruct-Q4_K_M.gguf，经解量化作为 FP32 算术参考；NPU 注意力 FP16、MLP 三 FC W8/A16、head FP16。这不是原始 BF16 基准，不是 NPU INT4，也不是与 Intern 相同的 NPU 精度比例。16 decoder +4 head NBG 共1,593,110,816 bytes，F32 embedding文件1,050,673,152 bytes；文件体积不等同于常驻 RAM，实际以driver和PSS单列。

架构16层、D2048、32Q/8KV、HD64、FFN8192、词表128256、Llama3 scaled RoPE。缩放频率与官方 HF 对照通过，layer0 FP32 实现相对官方 LlamaDecoderLayer 的误差为2.7110e-7；完整 head 首分片相对独立公式1.5004e-7。官方同源 tokenizer.json 与 GGUF 重建的9组文本/prompt IDs完全一致。

每层使用它前面真实 FP32 各层算出的输入校准；三条校准中文与质量问题分离，未复制layer0激活尺度到全部16层。每个导出图都对真实样本通过 PyTorch→ONNX Runtime 数值检查后编译。

## 实际执行范围

- 首版 prefill 和 decode 都经过16层 NPU，完整词表4分片投影和final RMSNorm也在NPU；prefill暂为逐token功能基线。
- CPU负责embedding查表、RoPE cos/sin数值构造、KV拷贝、调度、greedy选择；rotary乘法、norm、QKV、注意力和MLP均在NPU。
- system「你是家庭陪伴机器人，请用简短中文回答。」共22 token；三完整问题38/41/42 token。只复用system22，禁止把上一题用户内容算成缓存收益。
- 数值对照3题×8步，沿同一FP32源参考反馈，保存每步完整128256 logits；诊断写盘不纳入速度成绩。
- 自然输出上限96，分别测无缓存与常驻system缓存；另测固定反馈32步吞吐，固定反馈不能当自然回答质量。

题目为「你好，我今天有点累。」「明早七点提醒我起床。」「不要开灯，请陪我聊两句。」这是3题pilot，不是12题完整质量筛查。源Q4 FP32参考自然长度81/96/47，第二题仍被96上限截断；源模型本身存在冗长和将直接请求转为泛建议的问题。

## 资源保护与复现

权威代码在Windows F盘，编译在WSL；板卡由唯一owner持hostlock和flock运行。只在无推理期间rsync压缩、续传和SHA验证，不重复打包大模型tar。实验私有目录不替换原服务。

250ms采样温度、剩余内存、RSS/PSS、swap、频率、整机与进程CPU；进程CPU按 /proc/pid/stat 最后右括号后解析。78°C或MemAvailable低于256MiB会通过主线程信号立即停止自己的worker。TERM、热、低内存三个WSL假worker故障注入均验证日志保存和清理成功；此故障测试不属于板卡性能数据。正常关闭必须验证driver bytes归零。

完整构建证据在 build-evidence/q4-source/attn-fp16-mlp-w8；旧目录是指向该目录的兼容symlink。运行入口deploy_run.py，原始输出在board-results-q4-source-20261023T171713，汇总入口summarize_board.py。测试正在执行时不要把仅编译完成当成质量通过。

## 匹配对照与下一步门槛

Intern匹配组已准备相同三题/相同system语义：system16、总31/32/33，原W8图和batch16+tail8，严格只复用system。其文本decoder为24×D896，Llama为16×D2048，不能只看“1B”命名推断算量相等。prefill实现和FP16比例也不同，报告必须分别列明。

Q8-source的全W8首层、mixed首层、4个W8 head及相同中文校准的Q4首层控制由独立探针准备。先检查真实端侧误差，特别是完整词表argmax及真正未见输入，再决定是否扩展全W8完整模型。当前尚未开始该组板测，不报告推测速度。


## 续跑与内存统计修正
Q4首轮连续RSS/PSS采样只覆盖Python控制器，18,636/17,003 KiB不能作为完整宿主模型内存；NPU driver 1,660,910,976字节另列。已修正summary字段并在Intern新runner显式采样resident PID。Q8探针原启动失败由驱动内存0x格式解析引起，已修复并恢复传输；探针完成后continue_matched.py串行执行Intern同题对照。尚未取得最终近似同量化结论。
