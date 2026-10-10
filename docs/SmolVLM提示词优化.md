# SmolVLM-256M 提示词优化记录

2026-10-09，在现有 Orange Pi 上比较并部署；未更换模型或引入云端服务。

## 原配置与问题

原网页没有 system prompt，仅把 user/assistant 历史发送给 llama-server，
temperature=0.2、repeat_penalty=1.1、max_tokens=256。多轮历史确实有传递，
并非页面每轮丢失上下文。推理引擎使用 GGUF 自带聊天模板。
通过 `/apply-template` 核对 system/user 正确渲染为 `System:`、`User:` 与
`<end_of_utterance>` 分隔，没有把角色模板另行替换成其他模型的格式。

官方模型卡标注语言为 **English**：
<https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct>。
用户截图里的中文闲聊并非该模型可靠支持的用途。提示词能调整行为，不能补足语言和推理能力。

## 部署内容

提示词单独保存在 [system_prompt.txt](../src/vlm_chat/system_prompt.txt)：

```text
You are Orange, an AI assistant. Answer the latest question briefly and accurately. Use the conversation history. Do not invent facts. If an image is provided, describe only what you can see.
```

含义：你是 AI 助手 Orange（网页名“橙语”）；简短、准确回答最新问题，利用历史，不编造事实，
有图片时只描述能看见的内容。这些是指令，并非模型一定能遵守的保证。

网关在所有请求最前面注入且只注入一次 system 消息，保留真实用户文字、图片和历史。
客户端不能覆盖该消息；没有按关键词硬编码回复，也没有给用户对话塞入虚构的示例历史。
其余采样参数保持原配置；贪心采样的试验没有显示足够改善，未采用。

板端文件为 `/opt/opi-vlm/app/system_prompt.txt`，修改后需重启 `opi-vlm-web`。
已备份旧网关为 `/opt/opi-vlm/app/server.py.before-system-prompt`。
建议点“新建对话”体验：旧会话里模型已经说出的错误身份会继续作为历史影响生成，
改 system prompt 不会自动修正过去回答，也不会删除本地历史。

## 对比方法与结果

测试脚本：[evaluate_chat_prompt.py](../scripts/evaluate_chat_prompt.py)。
所有候选都在真实板端运行；固定 seed=42、最大输出 96 tokens，
每轮将实际生成结果放回历史，测试连续对话。不是每轮重置，也没有人工修正回答。
测试输出长度上限与网页 256 不同；这是行为对比，不是速度基准。

共比较 11 个“提示词/语言/采样”组合，每组 4 轮，另外做 8 轮选定提示词验证。
中文问题包括用户给出的三句话，及“我刚才说吃了什么？”；英文使用其等义问题。
原始数据保留所有失败，不仅展示最好的样例：

- [第一轮：无提示词、中英文提示、贪心采样](benchmarks/2026-10-09-prompts/prompt-comparison.json)
- [第二轮：英文简短提示、中文短提示与示例](benchmarks/2026-10-09-prompts/prompt-followup.json)
- [选定提示词的额外问题验证及中文复测](benchmarks/2026-10-09-prompts/prompt-validation.json)

| 项目 | 原版或其他候选 | 选定版本 |
| --- | --- | --- |
| 英文身份 | 原版：“My name is [Your Name]. I am a human.” | “I am Orange!”；另测人类还是 AI 时回答 “I am an AI.” |
| 英文心情低落 | 原版：“[Your Name] feels the same way.” | “It's okay to feel sad sometimes.” |
| 英文记住吃过什么 | 原版会提到 braised chicken，但混用占位身份 | 简短回答 “Braised chicken.” |
| 新话题记忆 | 未用于选提示词 | 给出兔子 Luna 后追问名字，正确保留 Luna |
| 未提供的信息 | — | 问兔子颜色却答 “I am an orange color.”，**失败** |
| 中文闲聊 | 多种提示均有严重跑偏、重复、编造 | 仍出现 “Candy City” 身份故事及无意义中文，**失败** |

英文中对“今天吃了鸡”的回应是 “You're a little hungry, right?”，也存在无根据推断；
因此不能把这次测试描述为“已经具备可靠对话能力”。带中文对话示例的候选仍乱答，
较长中文身份提示甚至加重重复，未部署。

## 验证与建议

网关/浏览器集成测试通过，覆盖 system 注入、禁止客户端 system 覆盖、原图和历史保留、
流式回答、历史持久化及异常处理。真实板端图片多轮回归结果另见部署验收记录。
原性能报告对应**没有 system prompt 的旧配置**；本次添加 prompt 会增加少量输入 token，
不要将旧性能数值当作新配置逐项重测结果。

当前合理用途仍是简单的英文图片问答。若目标是自然中文陪聊和看图连续对话，
下一步应对比中文能力更好的模型，而不是继续堆叠 system prompt。
本次未自动下载或切换其他模型。
