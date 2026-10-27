# 家居陪伴 VQA 配对评估（2026-10-27）

独立候选实验，未替换默认模型。112×112→16 与224×224→64视觉token，均用24个ViT block的144组FC权重per-channel INT8，INT16图边界、敏感norm/GELU FP16、stem/projector原精度。不是W8A8。语言侧同一Intern NPU worker与权重。

## 冻结测试集

`vqa-cases.json`：50张公开COCO照片，60题；可见表情、发色、桌面物品、桌面相对位置、室内场景、室外场景，各10题。图像整体bicubic缩放，没有人物裁剪或问题相关裁剪。10张桌面图各两题，其余每图一题。语言没有收到参考答案。

`frozen-cases.sha256`记录推理前冻结SHA，`freeze_home_cases.py`记录构建过程。人工改写/新增的问题单独标明，不把原问题10人答案当作新题标注。原始英文问题和答案仅作来源溯源。最初通用60题被用户场景要求替换，归档`scope1-superseded-*`，未用旧题启动此次板卡推理。图片URL和SHA可追溯；公开图片可能进入预训练，不能声称无训练集污染。

评价按可见核心事实人工审核：同义答案通过，矛盾或罗列不相干候选失败；无法判断单独标记，准确率分母不剔除。不是官方VQAv2软评分，也不是情绪/心理诊断。发色样本偏金发，表情偏微笑/非微笑；不能外推所有人群和情绪类别。固定60题，不按模型回答调整题目或参考。

## 执行与口径

`controller.py`持WSL与板端双锁；每个分辨率完整视觉塔常驻、3次预热、60次热推理（重复10张桌面图依旧实际编码）。视觉结束并释放NPU后，语言模型常驻，逐题重置后只prime固定system。每次上限32输出token。未测两塔同时常驻、语音并发、真实麦克风或长时稳定。

视觉分段记录JPEG预处理、tower模型、IPC wall、JPEG→向量；语言记录实际new/reused、prefill、TTFT、decode、总时间。两个阶段时间相加仅为分段估算，不称实测JPEG端到端。冷加载与system prime另列，不计热TTFT。RSS不包括全部NPU驱动内存，需配合driver_bytes和系统MemAvailable阅读。

图文件按独立文件上传，SHA校验；完成后仅移除本实验板端图副本，Windows C盘原图保留。正常关闭必须driver_bytes=0。`config-112/224.json`及`upload-sha256.json`冻结二进制、模型、脚本。原始vectors、图片、大图不入Git，留在Windows磁盘；报告和逐题判分入Git。

112/224来自现有InternVL3-1B-Instruct视觉权重与位置插值适配，保留官方2×2 pixel shuffle和投影结构；并非官方112分辨率验证结果。该配对比较部署方案，未隔离纯分辨率损失与INT8量化损失；不能把112错误全部归因于量化。

重建图文页：`python fetch_frozen_images.py`从冻结HTTPS地址下载并校验SHA，再在具备Pillow的Python环境执行`analyze_home.py`。分析仅使用已提交JSON证据和图片，不调用板卡；`write_home_report.py`生成Markdown。不要重复执行有独立状态文件的controller。
