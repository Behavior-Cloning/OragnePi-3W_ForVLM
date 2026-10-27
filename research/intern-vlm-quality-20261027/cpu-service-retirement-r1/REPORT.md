# 旧 CPU Qwen 服务退役（2026-10-27）

用户确认主大脑使用 Intern-1B，并授权关闭旧 Qwen 服务。

## 来源与用途

`opi-vlm-model.service` 原始单元描述为 SmolVLM 256M CPU inference。2026-10-10 的 model.conf 覆盖实际启动为 Qwen2.5-1.5B-Instruct Q4_K_M，llama-server，CPU 两线程，ctx4096，127.0.0.1:8081，无 GPU/NPU offload。
仓库 docs/benchmarks/2026-10-10-companion 与 history 中有该模型的测试记录。旧 opi-vlm-web.service（8080）Wants 此后端，因此判断为早期网页/陪伴对话实验遗留服务，不是本次 Intern NPU worker。
当前 NPU VLM harness config.json 指向 resident-visual、Intern 图与独立 vision-resident，不使用 8081。

## 已执行并验证

- systemctl disable --now opi-vlm-model.service：inactive/dead，MainPID=0，disabled；8081 无监听。
- 原始服务和模型权重保留。
- 旧网页仍有 Wants 依赖，单纯 disable 不足以防止被拉起；增加 99-retired.conf，要求明确的 /opt/opi-vlm/enable-legacy-cpu-model 文件存在才允许启动。该文件不存在；显式启动验证 ConditionResult=no、服务保持 inactive。
- 恢复旧后端须有新的用户要求，再移除这一退役 drop-in 并 daemon-reload；不自动恢复。

## 资源与后续边界

关闭后 MemAvailable 约 2.44 GB，NPU driver 当前0。双模型预检要求2.854 GB，仍差约416 MB，原因是剩余内存/临时模型资产占用，已不再是 CPU 服务未停。未绕过内存门控，也未启动新的共驻推理；需继续核对可回收的实验 tmpfs 资产。此处不宣称全链路测试完成。
