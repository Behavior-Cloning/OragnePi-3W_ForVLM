# 测试

应用的协议、数据处理和回归测试放在本目录。现有网页集成测试见下文。

`companion/phase1.json` 和 `companion/persona.txt` 是陪伴对话第一阶段的固定样例与角色。
执行 `python3 scripts/prepare_companion_phase1.py` 可离线校验，详见
[场景与评分协议](../docs/陪伴机器人第一阶段评测.md)。材料已准备不代表模型已实测。

`scripts/evaluate_companion.py` 为板端文字多轮执行器，`scripts/report_companion.py` 生成原始对话报告；
`vlm_text_board_smoke.cjs` 验证当前板端纯文字模型的网页流式对话，不发送图片。

已有硬件验证参见 `../npu/README.md`；板上可运行：

```bash
opi-npu-test status
opi-npu-test smoke 1
```

硬件测试与可在开发电脑离线运行的测试分开组织。大型音视频数据、模型权重和
生成日志保留在仓库忽略目录中；测试所需公开数据记录来源与校验值。
# SmolVLM 网页集成测试

`vlm_chat_browser.cjs` 运行真实 Python 网关、模拟流式模型服务和 Chromium，验证图片上传、
多轮上下文、历史保存、停止/异常处理、删除、移动布局和请求校验。使用方式见
[部署文档](../docs/SmolVLM网页部署.md)。模型本身的质量与板端速度需要另行实测。
