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
- [ ] PaliGemma2板卡常驻CPU图文四请求、资源测量；正在传输。
- [ ] PaliGemma2真实BF16单层NPU兼容性；已通过ONNX数值对照，FP16/W8两图已编译，板端待验证。
- [ ] PaliGemma2完整NPU图文部署；单层试验不视为此项完成。
- [x] FuseChat-Llama-3.2-1B-Instruct 中文增强 GGUF Q4_K_M 下载校验；原作者 config 16 层。
- [x] FuseChat板卡CPU性能/中文输出：6个新问题cache_n均26，TTFT P50 0.638s、decode 12.47tok/s。
- [x] FuseChat单层NPU兼容性尝试完成；已执行W8/MLP FP16对照，但数值误差不合格，不继续冒称整模部署。
- [x] 清理 172 个可重生成产物，逐项 SHA 和删除记录 maintenance/cleanup-manifest.jsonl。
- [x] TF 镜像 LZX 压缩尝试失败，原图保留；随后完整SHA重算与原记录一致。见maintenance/BACKUP_DECISION.md。
- [ ] WSL全部实验停止后压缩VHDX，测量真实F空间回收并验证重启。
- [ ] 完成汇总、Mac报告副本与版本提交（Intern/Llama NPU已提交6c55831）。

## 目前确认的边界
Intern 的无损 group prefill 收益约 1%；精确 system KV 复用在两个短场景 TTFT 0.433→0.334 s、0.620→0.380 s，但只 8/11 自然回答同 token，不能标为无损替换。Decode 合并提交只约 1% 改善且多 107.66 MiB 驱动内存，不推荐默认采用。

本次 Intern 测的是文本 decoder；没有把图片编码、语音链路算在上述时间中。MolmoE 是 7.2B 总参数/1.5B 激活，不能按名称 1B 估驻留内存。Pali 和 Llama 先取得真实 CPU 基线，不冒称 NPU 实测。

清理直接 F 文件 1,421,901,036 字节；WSL 内部 6,958,298,487 字节。WSL 内部回收不等同 F 盘物理空闲立即增加。原始模型、NBG、重建来源、TF 镜像均保留。
