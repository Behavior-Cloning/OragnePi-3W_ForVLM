# 当前运行版本（2026-10-27）

主大脑为 InternVL3-1B NPU resident worker。静态 system 总量32–64 tokens；动态正文（图像＋问题＋历史）最多128，模板另计。只复用 system KV。

CPU Qwen2.5-1.5B Q4 与旧 llama.cpp 网页服务不再维护，不作为回退后端。旧服务停用且不自动恢复；网页/控制 HTTP 入口明确报告退役，不能将模型名改成 Intern 来冒充 NPU 接入。

## 当前代码入口

- 视觉与语言共驻 harness：research/intern-vlm-quality-20261027/resident-npu-vlm-r1/README.md、run_prepared.py。
- NPU vision worker：research/intern-vlm-quality-20261027/vit-full-r1/vision_worker.py。
- 版本与精度证据：research/intern-vlm-quality-20261027/VERSION_TRACKING.json。
- 共驻运行仍须通过双锁与内存预检。当前实验方案不是已验收的生产 HTTP 服务；网页适配 NPU worker 尚未完成，不自动启动实验模型。

旧安装/批量预下载/模型切换脚本退出并指向本文；Qwen下载被拒绝。历史 benchmark、权重校验和、精度记录保留。不得将这些历史记录解释为当前运行配置。
