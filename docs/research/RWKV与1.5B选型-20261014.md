# 1.5B适配与RWKV视觉模型调研

核查日期2026-10-14；外部数字是作者公开结果，不是本板实测。实现进度另见实验记录，不能把论文分数当板端量化质量。

## 选择

Transformer只选Qwen2.5-1.5B-Instruct：官方1.54B参数、28层、hidden1536、Q12/KV2、RoPE与SwiGLU，中文与角色设定支持较成熟，已有本项目CPU Q4_K_M与陪伴对话记录。它是纯文本模型，不带视觉。本轮视觉要求由下面独立的RWKV选型处理，不把Qwen称为VLM。

CPU延用现有Q4_K_M；NPU尝试逐通道对称W4/A16，来源是现有Q4_K_M反量化权重，属于二次量化，**不是相同算法、不是原始BF16量化、更不是已证原生INT4内核**。性能与质量必须对照CPU实际输出；编译成功不能算部署成功。已有500M/1B Q8研究不改基线。

RWKV视觉仅选一个后续部署候选：**VisualRWKV-v7.00-0.4B**，权重文件`VisualRWKV-v0700-0B4-v1.0-20250109.pth`。理由是有公开视觉任务分数、权重公开，比1.5B版本更适合6GB板；不推荐0.1B作为中文陪伴主力，也不额外部署多个RWKV比较候选。

但0.4B指语言骨干标签，不能说整套仅0.4B。视觉栈是SigLIP SO400M + DINOv2 Large + SAM ViT-B，分别处理448/448/1024分辨率。公开PTH文件2573556208字节（约2.40GiB），1.5B视觉文件4733391152字节（约4.41GiB）。PTH大小不是运行峰值；激活、FP32状态、框架和视觉预处理另占内存。板上TF仅约2.8GiB余量，直接放原始0.4B权重会破坏512MiB预留，需先做工作站转换/测内存，不能强塞。

## “脑子”是否提升：有，但不是全面追平

RWKV-7论文Table 3在同一评测框架下给出以下**基础语言模型**结果。HellaSwag用归一化准确率；MMLU为知识/推理类选择题准确率。均不是Instruct陪伴聊天评分，不能与本项目Qwen-Instruct量化实测混排。

| 论文中的模型 | HellaSwag ↑ | MMLU ↑ | 九项平均 ↑ |
| --- | ---: | ---: | ---: |
| RWKV6-World2.1-1.6B | 61.1 | 25.1 | 58.3 |
| RWKV7-World3-1.5B | 70.8 | 43.3 | 67.6 |
| Qwen2.5-1.5B | 67.7 | 61.0 | 67.4 |
| SmolLM2-1.7B | 71.5 | 50.3 | 66.6 |

因此可以说：相较RWKV6，RWKV7在公开常识与知识测试上明显进步，HellaSwag已接近/超过这个Transformer对照；但MMLU仍明显低于Qwen2.5，平均值相近不等于所有能力相近。不能承诺中文角色扮演、事实更正和长对话就自然可靠。训练数据量/配方也不同，不能把所有提升归因递归结构本身。

当前官方仓库已列出更晚的`rwkv7-g1k-1.5b-20260930-ctx25600.pth`，另有0.4B G1d。模型卡明确这些是含聊天/推理数据预训练的BASE模型，仍适合后训练。未找到该G1k权重与上述模型在一致协议下、可直接引用的完整新分数，**不把2025论文数字贴到2026 G1k上**。也不能将新版纯文本RWKV语言头直接换进2025 VisualRWKV后沿用视觉分数，投影/视觉指令对齐需要重新训练验证。

## 小视觉RWKV与Transformer的公开指标

主比较指标选择**GQA准确率**（视觉关系与问答），补充TextVQA和ScienceQA-IMG以避免单指标选优。VisualRWKV源自官方README；MobileVLM源自其官方模型表。不同团队的预处理/训练/评测设置未统一重跑，属于选型参考，不是严格架构控制实验。这里GQA是视觉问答数据集，别与Grouped Query Attention混淆。

| 模型 | GQA ↑ | TextVQA ↑ | ScienceQA-IMG ↑ |
| --- | ---: | ---: | ---: |
| VisualRWKV v6，语言1.6B | 58.23 | 48.70 | 57.02 |
| VisualRWKV v7，语言0.4B（候选） | 62.30 | 41.05 | 54.98 |
| VisualRWKV v7，语言1.5B | 63.20 | 49.49 | 59.74 |
| MobileVLM-1.7B（Transformer） | 56.1 | 41.5 | 57.3 |
| MobileVLM V2-1.7B（Transformer） | 59.3 | 52.1 | 66.7 |

0.4B视觉RWKV的GQA并不弱，TextVQA与早期MobileVLM相近；但相比V2在文字识别问答和科学题上落后，不能说全面比Transformer聪明。三个大视觉编码器提供了重要能力，不能把这组成绩归功于0.4B语言模型或据此推断低延迟。它是合适的**实验候选**，不是现阶段可替换陪伴主模型的保证。

## 为什么RWKV适合端侧、也为什么不一定更快

RWKV-7用矩阵递归状态，每token更新状态，不保存随history增长的Transformer KV序列。固定模型的解码状态内存/运算量不随历史长度线性增长，适合长时间单用户对话。但它是压缩记忆，不等于无损“无限上下文”；昵称更正、精确检索和多轮角色要实测。

标准RWKV7（head64）状态元素约L×D×(64+2)：0.4B L24/D1024时FP32约6.19MiB，1.5B L24/D2048时约12.38MiB，未计运行临时张量/模型权重/视觉。这与Transformer逐位置KV不同，不能沿用现有K/V输入契约或窗口裁剪方式。撤回/编辑对话需恢复状态快照或从可信检查点重放，不能从递归状态中直接删除中间一条消息。

实际速度仍受权重访存、矩阵大小、ARM内核、低位量化和视觉编码限制。消费GPU/手机NPU的高token/s不能移植成A733实测。A733的ACUITY尚无本项目验证的RWKV7递归、GroupNorm和状态精度路径；初始应CPU验证，必要时仅大矩阵上NPU、状态FP32留CPU并标注混合方案。低精度状态递推会累计误差，不能照搬Transformer KV16就宣布安全。

## 生态与部署核查

- RWKV-LM提供v7纯NumPy、PyTorch递归参考；官方rwkv-mobile列出llama.cpp CPU、ncnn v6/v7未量化，以及Qualcomm QNN、MediaTek、CoreML等后端。**未列A733/VIPLite**，这些NPU支持不等于我们的板子支持。
- VisualRWKV v7主`src/model.py`在导入时编译CUDA扩展，并使用BF16块运算，无法直接在无CUDA板上照跑。
- 同一revision的`generate()`每步调用`self.rwkv(x)`，随后拼接新embedding并裁剪到ctx_len；这个公开生成示例没有复用递归状态。RWKV架构有固定状态的优势，不代表此演示实现已经兑现。板端必须另实现并验证状态复用，否则仍会重复处理整个输入序列。
- `VisualRWKV-v7/v7.00/app/app_cpu.py`虽然名字是CPU、目录是v7，内容title仍是v5，加载`howard-hou/visualrwkv-5`权重与CLIP。它不是已验证的VisualRWKV7 CPU实现，不能照抄后宣称v7部署完成。
- 需要分开视觉编码器、投影、RWKV7递归模块；以官方GPU序列实现对照CPU递归完整logits，确认图像token插入、分块/扫描、LayerNorm/GroupNorm、v_first与状态更新一致。
- 下游上线要测真实图片→视觉embedding→完整语言输出，不能用工作站图片描述作为板端“视觉理解”。

## 后续验收

### 用户用途更新（2026-10-14）

主要用途是桌面陪伴机器人、情感陪伴、日常聊天。公开指标优先HellaSwag与TextVQA，MMLU作为次要知识参考，不再以GQA单项成绩作为推荐理由。HellaSwag衡量生活情境后续的常识选择，不能测共情；TextVQA衡量图中文字结合画面问答，不能代替一般桌面感知，也不等同于中文OCR专项测试。

VisualRWKV-v7-0.4B继续作为唯一RWKV实验候选，但TextVQA41.05弱于MobileVLM V2-1.7B的52.1，尚不能推荐为最终产品模型。实际验收优先：自然接话、情绪回应贴切、不强行建议、角色/称呼稳定、接受记忆更正、避免复读、不虚构视觉感知，以及首字/持续解码延迟。视觉留出题优先桌面便签、包装文字、屏幕提示，另加少量物品/位置题；分别记录中文和英文，避免英文公开指标掩盖中文能力。

Qwen：先全28层完整输出与CPU同prompt核对，再扩512/1024历史及新增32/64/128，记录TTFT/token/s、RAM/driver、温度与换页；Q4二次量化失败必须保留，不能把W8结果当Q4成功。

VisualRWKV：固定0.4B候选，先核对权重SHA和总参数、CPU递归/官方参考一致性，再单张图片端到端；通过后板端部署，测试首图编码、后续仅文本续聊、状态复用/复位、256/512/1024历史与内存。公开GQA是选择依据，本地少量图片测试只能算smoke，不冒充重跑GQA。自然中文陪伴使用独立新题，包含昵称、更正、拒绝虚构感知与停止扮演。

## 可追溯来源

1. [Qwen2.5-1.5B-Instruct模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，参数/架构/上下文。
2. [RWKV-7论文v1](https://arxiv.org/html/2503.14456v1)，Table 3及视觉表，公开能力对照。
3. [VisualRWKV README](https://github.com/howard-hou/VisualRWKV)、[模型列表](https://github.com/howard-hou/VisualRWKV/blob/main/MODEL_ZOO.md)，代码revision `8e1becc44d0ef22270490282a1bb0153a79e521a`。
4. [VisualRWKV权重](https://huggingface.co/howard-hou/visualrwkv-6)，revision `63b213510eed8bb30c3cb44bd3e9a5254c8a32e1`，0.4B LFS SHA256 `88f58bb0c223efe8cb106c6caf72d76086ed221d65a3b3103289838b1db0f796`。
5. [MobileVLM官方表](https://github.com/Meituan-AutoML/MobileVLM)，GQA/SQA/TextVQA；只是公开对照，不额外选作部署模型。
6. [RWKV7-G1模型卡及文件](https://huggingface.co/BlinkDL/rwkv7-g1)、[RWKV-LM](https://github.com/BlinkDL/RWKV-LM)，当前版本与参考实现。
7. [rwkv-mobile](https://github.com/MollySophia/rwkv-mobile)，后端支持边界。
8. [VisualRWKV v7视觉代码](https://github.com/howard-hou/VisualRWKV/blob/main/VisualRWKV-v7/v7.00/src/vision.py)、[主模型](https://github.com/howard-hou/VisualRWKV/blob/main/VisualRWKV-v7/v7.00/src/model.py)、[旧CPU演示](https://github.com/howard-hou/VisualRWKV/blob/main/VisualRWKV-v7/v7.00/app/app_cpu.py)，实际依赖核查。
