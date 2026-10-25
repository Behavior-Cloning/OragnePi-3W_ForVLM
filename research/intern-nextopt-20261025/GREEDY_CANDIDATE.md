# 单遍 INT16 argmax 候选

独立于真batch8和context256，复用原图/量化/前缀缓存。冻结 `resident-profile` 未修改。

现有 `greedy_i16` 第一遍已用NEON查最大值，但第二遍最早并列索引扫描仍逐元素读映射输出。实际汇编见 `profile-greedy-disassembly.txt`；4head的dtype5/quant1、DFP fl8/8/9/8均满足原快速分支，并非意外fallback。sampling阶段只包含argmax、每head一次get和跨片比较，没有温度/频率读取。

`profile-sampling-audit.json` 独立复核12个fixed32测量、372decode步，sampling平均7.5272ms/步。4个诊断的first/decode共8个聚合点，第二遍读取元素数与时间r=0.9775；这是小诊断样本的相关性，不能证明因果或代替新候选板测。stage为墙钟时间，包含可能的调度等待。

## 候选契约

- `runtime-greedy/resident-greedy` SHA `3c80005f937c85f0f92370ace1fec35a91b75cb9200adbca90be20193257c5fd`，75,008 B。
- `OPI_GREEDY_I16_SINGLE_PASS=1` 开启（按存在判断，`0`也会开启）；不设置则原算法。
- `OPI_PROFILE_STAGES=1` 可继续分解same-binary A/B的sampling；不设置时用于纯速度对比。不启用 `OPI_PROFILE_NPU`，避免附加查询。
- 仅在现有INT16/DFP契约合格时替换argmax。每个shard仍独立按原DFP反量化后比较；不会直接跨不同scale比较raw int16。
- 单次读取8lane，保存每lane目前最大值及最早索引，再在8lane局部结果间选择全局最早最大值。严格 `>` 更新保持并列最早；INT16_MIN全负/全相等均正确。
- 16-bit索引只用于count<=65535，当前最大head49152。更大count走原算法，避免溢出。
- 向量实现使用GCC vector extension，同一源在x86主机与ARM64编译；未使用近似数值运算，不改变logits。

主机ASan/UBSan验证221,212组：长度1至8的4值字母表穷举、覆盖全部65536个INT16值、跨lane/跨块并列、真实4218/49152长度、65535边界与超界fallback、2,000随机输入，无不一致。完整结果见 `greedy-host-validation.json`。

ARM自检 `runtime-greedy/test-greedy-arm` SHA `096435fe19729ebcbcc0cf63fc3174d141a3fc2b2544a3ef6deac4d53b8ed28f`，先在owner持锁期间执行一次确认ARM SIMD同样通过。主机验证不证明真实映射输出buffer上的速度；必须实测same binary baseline→single-pass→baseline，固定输入和feedback、全logits/自然输出一致后才推荐。
