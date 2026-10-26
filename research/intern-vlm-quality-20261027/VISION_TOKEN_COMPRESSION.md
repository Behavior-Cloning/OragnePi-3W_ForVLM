# InternVL 视觉 token 压缩：位置、兼容性与验证状态

日期：2026-10-27。当前板端语言模型是 InternVL3-1B-Instruct 的 Qwen2.5 分支（24层、hidden896）；不是 InternVL3.5-Flash。本文记录研究与候选，不代表部署验收完成。

## 当前用户约束

静态 system 总量32/64，含模板5（正文27/59）；动态内容中图像+所有文字合计约128。图像目标64、文字剩余最多64；聊天模板和图像边界单列。用户随后明确优先 token 压缩，尤其 ViT Transformer 层前减少 token；多尺寸输入研究暂停，主线固定单图块448×448。

## 压缩位置决定节省哪段计算

|路径|ViT patch token（另有1个CLS）|送入LLM视觉token|当前模型支持情况|主要收益/代价|
|---|---:|---:|---|---|
|原版448、2×2 pixel shuffle+MLP|1024|256|原生已训练|质量参照，超出128动态部署预算|
|InternVL3.5-Flash高压缩分支|1024|64或256|3.5 Flash专门训练；当前3.0模型没有|省LLM prefill/KV；不省完整ViT层计算|
|448、末端256→64均值池化|1024|64|额外未训练实验|省LLM；不能继承Flash质量结论|
|448、patch embedding后局部2×2合并|256|64|新增未训练实验，本次优先实现|ViT和LLM都减算，细节损失必须测|
|前置可训练局部merge/压缩stem|256或其他固定预算|64目标|需适配训练/蒸馏，无现成当前模型权重|可能改善前置压缩质量，不能先承诺|
|ViT前若干层后再筛选/合并|随层递减|须重建规则网格或改projector|外部改造，例如DynamicViT/ToMe类思路|先获得语义再裁减，保留前几层开销；乱序稀疏token不能直接接原pixel shuffle|
|真正224输入+位置编码插值|256|64|结构可适配，但本轮按用户指示暂停|可省ViT算量；不是Flash方案|
|图像文件JPEG压缩/普通去噪|1024（若仍448）|256|预处理|只压文件/改善输入，不自动减少模型矩阵计算|

以上列的是当前checkpoint结构支持及可明确实现的实验族，不把论文中的任意方法当现成开关。对当前原生模型，只有2×2 pixel shuffle和对应已训练投影；把ratio从0.5改0.25会改变projector输入维度，不能只改常量。

## Flash 引用核对

官方已经发布 [InternVL3_5-1B-Flash](https://huggingface.co/OpenGVLab/InternVL3_5-1B-Flash)，源代码冻结 revision `4db3421dfc95423459cd20e4bf4f61d42decf581`。`extract_feature_flash` 先执行完整vision_model，再在1024 patch特征上计算router，并计算64/256两分支。64分支是4×4 space-to-depth和单独mlp2；并非对已经投影的256×D向量再次简单shuffle即可等价。空间重排在形状层面可分步，但跨越不同学习到的MLP后，数值/语义不等价。

[论文](https://arxiv.org/html/2508.18265v2) 的ViCO包括一致性训练、再冻结主模型训练router两个阶段。“平均约50% token减少、接近原性能”是其评测结论，不能外推为强制每图64、所有OCR无损，也不是ViT FLOPs降低50%。保留256分支正是给困难图块留预算；我们硬限制128动态内容时，这个回退会超预算。

另一个源码细节：当前Flash `generate_flash` 用 `gate > quantile(gate,0.1)` 与绝对阈值联合选64；单个图块时分位值等于自身，严格大于条件为假。`flash-route-predicate-audit.json` 用同一表达式验证单图块默认选256。这是代码路径审计，尚不是加载权重后的VQA实测；不能认为单张单块开Flash就自动得到64。

1B-Flash 的语言部分是Qwen3（28层、hidden1024、8 KV heads），当前是Qwen2.5（24层、hidden896、2 KV heads）。更换Flash需要重新导出/量化NPU图，不能复用r6现有图，也不能把Flash投影直接接896维旧语言模型。

## 本次优先候选：fixed448-frontmerge-r1

保持原始解码后的448×448输入与原patch14卷积；1024×1024维patch embedding在ViT第一层之前按相邻2×2均值合并，得到16×16=256个patch。位置向量用同样的空间合并，CLS独立保留且保持GGUF的末尾顺序；24层ViT输入257个位置。其后保留原2×2 pixel shuffle和原projector，输出8×8=64个896维向量。

这是无训练的低成本筛选基线，不称“好的/可靠的压缩器”。它覆盖全图，避免随机丢块，却可能丢失小物体、数字和边缘。若VQA不合格，优先评估带蒸馏的merge，或较温和的分阶段合并；不会只因加速成功就覆盖默认。

同binary提供native/front/late三个模式；448输入不变。native输出256，late在完整编码+投影后平均到64，front在第一层前合并且末端输出64。同图、同线程、同CPU亲和测冷加载和常驻热推理，质量另用原图/空白/换图控制。native256仅作超预算质量参照。

## 计算量估算，不是板端速度

视觉塔24层、D=1024、FFN=4096。每层主要MAC约 N(4D²+2D×FFN)+2N²D，N包含CLS。1025→257后，24层主要MAC由361.18G降到80.86G，约剩22.39%，理想主要MAC比4.47倍。不是所有算子都减少16倍：线性层约4倍、注意力矩阵约16倍；patch stem、合并、投影、内存搬运、调度另计。实际延迟必须板测，FP16/混合精度不能直接拿3TOPS INT8相除作为实测。详细算数见vision-compression-architecture.json。

## 验收和版本约束

- 首先检查shape、finite、全图空间顺序；完整448原路径保持不变。
- VQA按物体/颜色/数量/空间关系/动作与小字分项，冻结输入SHA和答案；比较正确率、遗漏与错误断言。延迟和质量分开报告。
- 测视觉塔+projector总耗时、预处理耗时、NPU语言prefill/首token/decode、常驻RSS/PSS和温度。未分段测量的encoder/projector不虚报独立数字。
- 系统32/64、图像64、文字≤64、模板/图像边界和实际new分别计数；只复用system KV；换图必须使图像部分缓存失效。
- 没有VQA证据前，质量最优、正式默认指针均不变。候选、编译成功、部署成功、质量通过分别登记。

## 原生动态图像功能（已核对，当前不开展多分辨率试验）

[原模型配置](https://huggingface.co/OpenGVLab/InternVL3-1B-Instruct/blob/f91c6391476dbc8fd07674d2589e8168d3ca667f/config.json)：448、patch14、shuffle0.5、最多12图块、可加缩略图。官方动态高分辨率是按宽高比排布多个448方块，单块256；多块时缩略图再加256。该功能增加细节和token，而非将单图自动压成64。视觉位置编码支持bicubic插值，但chat提取代码按方形网格重排；任意长方形输入不能直接声称兼容。源文件和SHA已冻结在official-sources和flash-official-sources。

## 后续追加：112×112保留64token

按用户后续指示重新加入112输入；与前面暂停多尺寸的状态相比，这是新的明确授权。112/14=8，ViT接收64patch+CLS。**112主分支跳过pixel shuffle，保留64个空间token**；不走原生112→16路线。为匹配原projector的4096维入口，筛选版对每个1024维特征进行四份通道重复后使用原投影权重（空间token数不变），属于未训练适配，不是官方原生路径。这个做法仅用于判定速度/质量是否值得继续，不预称可靠；后续若质量不足需要训练1024→896投影或蒸馏适配。

并列测试：448-native256、448前置merge→64、448后置pool→64、224-native64、112-keep64。后两者位置编码按官方bicubic/align_corners=false逻辑处理patch网格，GGUF的CLS末位置独立保留。没有将112先放大到448；也没有把一个空间token伪装成多个token。
