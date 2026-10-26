# C448 tail8 r2 恢复记录（2026-10-26）

860 MB上传已完成，39个文件与WSL原包逐文件SHA256一致，见upload-r2-verified-sha.json。
独立运行目录 /tmp/candidate-c448-tail8-smoke-r2。未替换默认。
输入system正文59+模板5=64；动态正文256+模板8=264，prompt328。每轮reset后独立构建system，实际reuse64/new264；输出固定反馈8token。无语音/RAG并发，此smoke不作为生产结果。

原图3次TTFT为1.183754/1.031291/1.018675秒，prefill为1.166214/1.013514/1.001360秒，decode为16.513896/16.465859/16.606142 token/s。相同输入的首预测出现20002/3837/20002差异，重复一致性未通过；不得晋升最佳。

原图正常close driver0。tail8阶段只有加载日志，无ready、推理或close事件；控制器PID及板端worker均已消失，未回收退出码，因此整体不完整，不能判定tail8模型失败或收益。独立查询driver当前0，alloc/free550/550。控制器消失的原因尚待检查，之后须采用板端脱离SSH存活并保存exitcode的控制器，新run ID重测。

本轮未达到100ms目标。后续优先查重复首token差异、图端口和缓存初始化，性能结果与质量资格分开。
