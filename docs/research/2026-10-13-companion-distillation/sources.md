# 调研来源索引

核对日期：2026-10-13。主要核对作者 README、数据卡、论文摘要；不是完整数据下载审计。下列规模应以正文注明的版本和单位理解。链接指向可变分支/数据卡，实际训练需记录 revision 与文件哈希。

| 编号 | 原始来源 | 本次核对重点 |
| --- | --- | --- |
| D01 | [LCCC / CDial-GPT](https://github.com/thu-coai/CDial-GPT) | base/large 单轮、多轮数量；来源；研究用途声明 |
| D02 | [CPED](https://github.com/scutcyr/CPED) | 按剧集划分、对话/话语统计、情绪及人格标签 |
| D03 | [RoleLLM](https://github.com/InteractiveNLP-Team/RoleLLM-public)、[RoleBench](https://huggingface.co/datasets/ZenMoore/RoleBench) | 100 角色、168,093 样本、中英与合成方法 |
| D04 | [WildChat-1M 数据卡](https://huggingface.co/datasets/allenai/WildChat-1M) | 当前过滤版 837,989 行与百万级名称区别；真实用户/模型回答；ODC-BY |
| D05 | [PersonaHub](https://github.com/tencent-ailab/persona-hub) | 人物描述而非对话；早期与后续规模；数据/代码不同许可 |
| D06 | [EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues) | 25k 情绪场景对话；作者下载入口 |
| D07 | [ESConv / FailedESConv](https://github.com/thu-coai/Emotional-Support-Conversation) | 1,300 对话、8 类策略、196 失败样本与研究用途 |
| D08 | [SoulChat](https://github.com/scutcyr/SoulChat)、[SoulChatCorpus](https://www.modelscope.cn/datasets/YIRONGCHEN/SoulChatCorpus) | 2024-06 发布计数与早期论文规模不同；模型声明不能直接套为数据许可 |
| D09 | [CPsyCoun](https://github.com/CAS-SIAT-XinHai/CPsyCoun)、[公开对话集](https://huggingface.co/datasets/CAS-SIAT-XinHai/CPsyCoun) | 3,134 重建对话；原始报告申请与公开对话区别 |
| D10 | [SoulChat2.0](https://github.com/scutcyr/SoulChat2.0)、[PsyDTCorpus](https://modelscope.cn/datasets/YIRONGCHEN/PsyDTCorpus) | 5,000 段、90,365 轮；风格数字分身合成 |
| D11 | [PsyQA](https://github.com/thu-coai/PsyQA) | 仓库样例与完整数据申请条件 |
| D12 | [DailyDialog 原论文](https://aclanthology.org/I17-1099/) | 日常人工对话、情绪/意图；本次不使用未复核精确数量 |
| D13 | [PersonaChat 原论文](https://aclanthology.org/P18-1205/)、[ParlAI 说明](https://github.com/facebookresearch/ParlAI/blob/main/parlai/tasks/personachat/README.md) | 给定 persona 的对话任务；加工版本区别 |
| D14 | [BlendedSkillTalk](https://github.com/facebookresearch/ParlAI/blob/main/parlai/tasks/blended_skill_talk/README.md) | 约 7k；人类与模型建议混合采集；来源交叉；CC-BY4.0 |
| D15 | [MSC](https://github.com/facebookresearch/ParlAI/blob/main/projects/msc/README.md)、[论文](https://arxiv.org/abs/2107.07567) | 多 session 与摘要数据；examples 不是独立 conversation |
| D16 | [RealPersonaChat](https://github.com/nu-dialogue/real-persona-chat) | 日语；公开版 13,583/408,619/233；主观评价；CC-BY-SA4.0 |
| D17 | [UltraChat](https://github.com/thunlp/UltraChat) | 1.57M 项目规模与精选子集区别 |
| D18 | [UltraFeedback](https://github.com/OpenBMB/UltraFeedback) | 63,967 prompts、四候选、修正评分版本 |
| D19 | [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | 训练 20,324、验证 1,038；人工属性与直接 preference；CC-BY4.0 |
| E01 | [CharacterEval](https://github.com/morecry/CharacterEval) | 中文角色评测规模、13 指标、4 维度 |
| E02 | [LoCoMo](https://github.com/snap-research/locomo) | README.MD 与 locomo10 公开版本；图片发布方式 |
| E03 | [LongMemEval](https://github.com/xiaowu0162/LongMemEval)、[cleaned 数据](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | 500 问题、长历史、oracle 与清洗版 |
| E04 | [LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2) | 2026 版本、451 题、超长期任务定位 |

算法链接见 [报告第 12 节](README.md#12-参考文献索引)。GKD、MiniLLM 的论文摘要及 TRL 实现说明用于核对方法区别；训练配置和预算是本报告建议，非这些来源对当前板卡/工作站的性能承诺。LoRA 原论文入口本次网络读取曾超时，保留其标准文献链接，不声称完成该论文全文复核。

检索期间遇到若干旧仓库地址 404，已用作者实际入口替换或不纳入正文；未把搜索未成功的数据当作已验证推荐。数据获取阶段仍需检查文件实际存在、行数、字段、重复情况及版本信息。
