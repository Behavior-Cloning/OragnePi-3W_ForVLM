# Jev 验证工作目录

项目位置：`F:\Files\OPi-Lab\repos\OragnePi-3W_ForVLM\research\jev-validation`  
WSL 路径：`/mnt/f/Files/OPi-Lab/repos/OragnePi-3W_ForVLM/research/jev-validation`

当前范围：根据 2026-10-20 用户追加授权，开始模型下载、板端部署与共驻测试；不进行后训练或实体动作。进展以 results 与本轮报告为准。

- [板卡诊断报告](BOARD_DEBUG_20261020.md)：已通过 Windows 跳板 SSH 登录板卡，发现 MMC 超时与两个服务失败，列明证据与影响。
- [Jev 分阶段验证计划](JEV_VALIDATION_PLAN.md)：模型选择、100 动作方案、延迟/精度/资源指标、异步并发与后训练边界。
- [100 动作可读词典](ACTION_DICTIONARY.md)：全部虚拟动作，4–8 中文字符。
- [actions.json](actions.json)、[groups.json](groups.json)、[cases.json](cases.json)：100 动作、10 组、140 初始样例。
- [fixture-manifest.json](fixture-manifest.json)：样例哈希。
- [诊断原始记录](results/board-debug-20261020/)。

模型状态（2026-10-21，实验进行中）：Laya multilingual 322M、Jev-Style 0.8B Q4_K_M、OneJev1.6-0.8B、Nano-Jev 33.36M 已下载；主权重已核对来源 SHA256。Laya 和 Nano 已推送并再次核对板端哈希，正在跑模型准确率/热态延迟；OneJev 多模态前向尚未实施。详见 [本轮报告](JEV_DEPLOYMENT_20261021.md)。models/、.venv/ 与编译缓存被 .gitignore 排除。

`collect_board_debug.py` 是第一轮诊断脚本；后续额外只读检查的结果单独保存在同一结果目录。板卡检查未修改服务与磁盘配置；随后按用户要求完成一次正常系统重启，复查结果已追加到诊断报告。诊断资料记录内网地址，分享前按实际需要处理；未包含登录密码。

## 本轮追加 TODO（2026-10-20）

- [ ] 非常小 Jev：核实小于322M的完整公开权重、中文/英语用途、ARM部署和延迟，不能按Tiny名称推断规模。
- [ ] 多模态 Jev：OneJev1.6-0.8B权重下载与结构核对；离线图片预处理、视觉encoder、决策分段计时及峰值内存。
- [ ] 多模态 Jev 与 Smol500M / Intern1B / Qwen1.5B分别共驻、混跑；未做视觉前向时不得标记完成。
- [ ] 音频原生JeV（如Toddler-Jev）调研：基座依赖、适配器、语音延迟和板端可行性；其3B基座不是超小模型。
- [ ] 远程无 GPU 存储服务器：下载 JEV-9B、JEV-27B-VL 完整权重与配套代码，记录 revision/SHA；不要求 GPU、不做服务器推理。下载任务已启动，完成校验尚待确认。

## 逐模型提速审计（2026-10-21 新要求）

见 [JEV_OPTIMIZATION_AUDIT.md](JEV_OPTIMIZATION_AUDIT.md)。每个模型都必须做优化前后配对、准确率变化和常驻并发验证；仅部署成功不能勾选完成。`board_optimize.py` 执行第一轮核/线程/量化/候选批大小/runtime配置筛查，随后扩展最优配置。
