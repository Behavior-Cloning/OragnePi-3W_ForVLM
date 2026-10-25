# C512与更大prefill批处理的隔离runtime候选

未部署板卡；不替换当前运行版本。源于suffixgroup-firstfix，仅新增：
- true M8尾部允许C512，以及主bank M32；所有实际KV/tensor类型、shape与量化接口仍在加载时核对。
- 首段原精度策略在M8 bank需显式OPI_FIRST_PREFILL8_L2，缺少该图则拒绝加载，避免用M16图接入M8接口。
- 保持永久suffix group，不在推理中destroy/rebuild group。

ARM编译通过，binary见build.json。主机测试扩展至C512 M16/M32、392/423历史位置，6种普通/模板/校验模式×4320张mask校验，FP16与INT16两种ASan+UBSan校验均通过。此测试只验证缓存/掩码不变量，不等于NPU图、SDK组生命周期或完整模型数值验证。首轮编译变量名错误保留build-first-failed.log，已修复。

板测前置条件：当前C1280修复方案实际质量审计；C512图16个实际接口检查；锁内存储预算；先同输入诊断与release检查，再无logits转储计时。系统正文59/123+模板5=64/128；动态正文64/128/192/256+模板8=72/136/200/264；不截断输入，只复用system。
