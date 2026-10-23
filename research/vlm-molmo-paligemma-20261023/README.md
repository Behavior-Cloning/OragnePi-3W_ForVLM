# 2026-10-23：VLM 尝试部署与 Intern W8 优化

权威代码与结果位于 Windows F / WSL；Mac 仅控制与报告副本。板卡经 Windows 跳板、双锁串行操作。以下为进行中状态，不把下载或单算子编译当作整模部署。

## 工作项
- [x] Intern W8 同输入配对：group prefill、exact-prefix、decode+head 合并，88 次计时；详见 ../intern1b-operator-opt-20261023/REPORT.md。
- [x] 独立审计 Intern 原始记录、源码 SHA、清理依赖：AUDIT.md。
- [x] MolmoE-1B 官方配置/源码获取校验；64 专家 Top8 路由形状的随机算子图实际通过 Acuity 转换。
- [x] MolmoE 元数据经过现有 llama.cpp converter dry-run，失败于 tokenizer/vocab 不兼容；原始 28.89 GB 权重未下载。
- [ ] MolmoE 完整模型板端部署、视觉与生成质量、延迟。当前未实现，不报告虚构速度。
- [x] PaliGemma2-3B-mix-224 BF16 两片权重（6,064,583,376 字节）ModelScope SHA 校验。
- [x] PaliGemma2 SFP压缩转换、ARM构建，三问WSL视觉输出与原始BF16逐token一致。
- [x] PaliGemma2板卡常驻CPU图文四请求、纯中文文本三请求及资源测量；旧图文进程CPU因解析错误作废，文本采集已修复。
- [x] PaliGemma2真实BF16单层NPU FP16/W8实测，18状态（13唯一输入），driver正常释放；不是完整部署。
- [ ] PaliGemma2完整NPU图文部署；单层试验不视为此项完成。
- [x] FuseChat-Llama-3.2-1B-Instruct 中文增强 GGUF Q4_K_M 下载校验；原作者 config 16 层。
- [x] FuseChat板卡CPU性能/中文输出：6个新问题cache_n均26，TTFT P50 0.638s、decode 12.47tok/s。
- [x] FuseChat单层NPU量化定位：全FP16与attention FP16 + MLP W8配对15状态；9中文留出混合误差2.14–3.70%，需整模验收。
- [x] 同模型同revision Q8_0权重下载与SHA校验，词表与张量结构和Q4完全一致。
- [ ] Q4/Q8源完整16层+全词表输出头NPU中文生成，最终与Intern近似量化条件比较；当前Q4正在逐层编译。
- [x] 清理 172 个可重生成产物，逐项 SHA 和删除记录 maintenance/cleanup-manifest.jsonl。
- [x] TF 镜像 LZX 压缩尝试失败，原图保留；随后完整SHA重算与原记录一致。见maintenance/BACKUP_DECISION.md。
- [x] WSL全部实验停止后原生compact，实际回收20.13GB，F空闲约35GB；重启/环境/关键模型SHA验证通过。
- [x] 已完成阶段汇总与版本提交6c55831、b637966；独立审计见AUDIT.md。
- [ ] 完整Llama NPU/Q8配对结果、最终汇总、更新Mac副本与后续提交。

## 目前确认的边界
Intern 的无损 group prefill 收益约 1%；精确 system KV 复用在两个短场景 TTFT 0.433→0.334 s、0.620→0.380 s，但只 8/11 自然回答同 token，不能标为无损替换。Decode 合并提交只约 1% 改善且多 107.66 MiB 驱动内存，不推荐默认采用。

本次 Intern 测的是文本 decoder；没有把图片编码、语音链路算在上述时间中。MolmoE 是 7.2B 总参数/1.5B 激活，不能按名称 1B 估驻留内存。Pali 和 Llama 已有CPU整模与NPU首层实测；完整Llama NPU进行中。

清理直接 F 文件 1,421,901,036 字节；WSL 内部 6,958,298,487 字节。WSL 内部回收不等同 F 盘物理空闲立即增加。原始模型、NBG、重建来源、TF 镜像均保留。
