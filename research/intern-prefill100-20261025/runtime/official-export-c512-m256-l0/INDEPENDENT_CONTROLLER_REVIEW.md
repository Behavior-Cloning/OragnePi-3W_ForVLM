# M256 单块 A-B-A 启动审查

无阻断项。控制器 SHA `dac95faea4c8b3f8ae7150d3611f3e024a2e46fdb69771d79ba72039be31248d`，分析器 SHA `3b93bc2d4fdb26b0d75711a6c3ab68afeb9e081d4969ae8d88d01d88eabd1e69`，绑定 manifest SHA `b50c3194c9e33dd445af6f1a90475197f521278106f57d8e2c5ebb342c7ed215`。只读审查与Python语法编译检查，没有执行控制器、模型或板端命令。

检查覆盖 C512/M256/L0/fused2、同 canonical 输入的 A-B-A、每版3预热+30API计时、双锁、唯一run与超时回收、全SHA、五f32输出共1,441,792B。输入移到唯一tmpfs路径只改变预载存储，三版仍同输入且计时外预载；磁盘reserve不变。磁盘与tmpfs量分开记录。结果回收后仅rc0且meta complete时复核SHA再删tmpfs；失败保留。

本审查不构成性能或质量验收。没有语音/RAG，测试仅两层融合块，不能报告整模TTFT。
