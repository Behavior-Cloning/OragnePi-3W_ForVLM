# FP16 down Conv 候选只读审核与渐进部署建议

当前输入目标：固定 system 总64/128 tokens；动态正文64/128/192/256，另加模板8（动态真实新增72/136/200/264）；输出主基准32 tokens。本文没有新增板测或编译，没有改候选图与冻结运行时。

## 静态审核结论

候选 `graphs/candidates-fp16/conv1x1-m16-l2` 可进入单块实测，未发现静态阻断。独立审核证据在 `conv-review/metadata-audit.json`。

- 仅4个源graph节点变化：b0/b1 reshape115及down fullconnect116。
- 输入 `[1,1,16,4864]`，权重 `[1,1,4864,896]`，输出 `[1,1,16,896]`，NHWC，1×1、stride1、VALID、group1、无padding；每个空间位置对应一个token，没有跨token卷积。
- 两个down由矩阵乘法改为1×1卷积，后续reshape恢复原隐藏维度。46个源constant数组的数据类型和值字节全同，仅两个weight形状从二维改四维；bias实际为全零。
- 独立从已导出的C与payload读取并比较14个FC权重，dtype与字节全部一致：2个down仍FP16，其余12个仍INT8。原quantizer逐字节相同，11入5出端口metadata完全相同。
- 候选NBG为33,189,152 B、SHA `425bc11857e5691b30abf3d60700dcc735dd13f6b12bed3671f32a287dfe02b5`；原C512 M16 L2为33,184,080 B。文件差约5KB不代表运行内存差。

主机等价记录使用一个真实canonical fixture，对全融合双层进行FP32解释：hidden相对L2约3.2e-7，另外四个KV输出均有限且误差很小。另做down权重FP16再展开控制；但它仍是FP32算术，没有模拟NPU中间激活量化、累加精度或FP16饱和。这只能验证数学改写、通道和token排列，没有证明端侧逐位一致或中文质量。

## 为什么保留FP16也可能改善

FCL与1×1 Conv数学等价，但编译器可能选择不同lowering、矩阵分块、内存布局或执行路径。当前原L2 down没有NN计数，而原L0 INT8 down有；这是尝试改写的依据，不是Conv必定进入NN或必定提速的证据。必须使用实机API时间与NPD证明候选行为。保持weight字节与接口不能强制保持累加顺序，FP16舍入、FP32累加、融合bias和后续转换都可能改变输出。

实际M16候选硬编码16行，不能给它32行输入，也不能直接放入C1280 runtime；C512与C1280的历史KV/mask形状不同。M32需独立导出对应shape，核对所有reshape、mask、新KV长度和端口，不只是修改文件名。

## 推荐准入判据

1. 单块同图前后重复：保持相同input、温度/频率与flags，baseline→candidate→baseline，各3 warmup/30正式；profiling分开。检查全部5输出有限、无重复漂移、driver归零。
2. 比较两种误差：candidate对原NPU、candidate对同源FP32 oracle，逐hidden/newK/newV分别报告。b0 K/V处于第一次down之前，应该逐字节相同；若不同，优先排查layout或编译引入的其他差异。b1 K/V和hidden可能受新累加顺序影响。
3. 建议把candidate对baseline相对L2≤1e-3、source-oracle误差不增加超过10%作为**进入小规模整模试验的保守筛选线**，并同时观察maxabs及异常token；这不是通用精度保证。超出即暂停扩栈分析，不把原误差≈0.6%当可任意增加的预算。
4. 单一fixture通过仍不足：至少补3条不同中文真实隐藏状态、前史64/128与真实尾部valid8，对全部输出比较，padding不进入KV/history。自然输出另测，不用fixed-feedback替代质量。
5. 优先要求收益远超baseline重复漂移。若M16有清晰收益（建议≥20%）且数值筛选通过，可扩一个M32 L2 Conv probe；若只快1%或更慢，不立即扩完整C512 12块。
6. M32应与同输入覆盖的2次M16比较总运行/调度，不能仅因一次调用减少就认定更好；同时比较前16行的数值及合法32行的KV顺序。建议先M32，M64/128/256延后，因为原始大batch已实测每token成本恶化。

## 优先的最小全模型路线：C1280只替换一个prefill块

逐个检查现役 `/opt/opi-lab/source-validation-20261017/gguf-fresh/export/w8-fused2-c1280-t16-cap1-productwide/l{0,2,...,22}/decoder.quantize`，结果保存 `conv-review/prefill-fp16-inventory.json`：**只有融合块L2内真实层2和3的down是FP16；其余22层down全部I16输入/I8权重/I16输出，其他FC权重也没有额外FP16例外。**

因此若当前C512 M16单块Conv有效，建议下一步只导出 **C1280 M16 L2 Conv**，保留原C1280上下文和输出协议。独立overlay目录中只有 `prefill-layer2.nb` 为新文件；其他11个prefill、12个decoder、4个head以及embedding均symlink原文件。保持现有C1280 resident binary和flags完全相同，进行原图→overlay→原图配对，不让C512 runtime混入此对照。

这个操作不改decode图，但会同时影响system准备及动态prefill经过L2的计算。每个variant必须重新准备system-only KV，不跨variant重用KV；system准备时间单列。对固定system64/128与动态正文64/128/192/256+模板8，输出32，同feedback、自然中文、全词表logits分组。实际reuse必须等于64/128。若原流程开启true8，最后尾8会继续使用旧true8图，报告需注明，不能声称全部prefill都换了Conv；否则M16处理尾8时按原逻辑padding，记录真实valid8。

资源优势：只增加约34MB级NBG（现原C1280 L2为34,045,488 B；新图大小须编译后读取），编译时约200～250MB临时材料，包括119MB reshape后的源data、39MB导出payload及34MB NBG。板端overlay复用其余文件，不复制543,599,616 B embedding。既有进程退出后加载overlay，driver是替换一图而不是两套模型常驻；driver真实差额必须测，不能由NB大小推断。

## 后续C512完整运行时的最小图集

`resident-prefill100`需要全部C512图，当前10个probe覆盖L0/L2两块、5种M，不能当作完整模型：

- 必需：12个C512 scalar decode `layer0/2/.../22.nb`；12个C512 M16 `prefill-layer0/2/.../22.nb`；4个全词表head `layer24..27.nb`可复用；embedding可复用。
- 现有M16 L0/L2可以复用为原精度base，尚缺12个decode+10个M16，即最少22个新图。若L2替换Conv，原L2与Conv需分别保留并有明确variant manifest。
- bulk仅在一个M已获实测准入后增加12个图；不可一次常驻所有M。M32目前已有原L0/L2两块，剩10块；Conv L2需另外生成。
- true8不是功能必需。可先原M16 valid8尾批；真M8需要额外12图，且C1280/C256 M8不能直接复用。先完整M16基线再独立引入M32或M8，避免一次改变多个维度。

历史C256实测为12decode+12M16+4head driver **698,780,824 B**；另驻留12个true8后 **988,129,464 B**（额外289,348,640 B）。这直接证明新增bank并未共享权重。C512不能照抄此RAM数字，应先stage新图后测ready driver和整体MemAvailable/PSS。

磁盘和RAM规划依据保存于 `conv-review/resource-basis.json`：C256 decode文件284.70MB、M16文件287.49MB、head文件112.60MB。用当前C512 L0/L2文件作粗略 `11×L0+L2` 预算，一个bank分别约M16 296.92MB、M32 306.29MB、M64 318.35MB、M128 357.21MB、M256 440.91MB；其他层压缩率不同，**不是全栈实际大小**。

据此C512最小decode+M16新增文件约0.58～0.65GB，额外一个M32 bank再约0.30～0.34GB，true8再约0.29～0.32GB。head/embedding用overlay复用。WSL按逐图编译、保留可重建证据后清理可再生临时材料，可避免同时复制12份119MB权重；F盘此审计时约25.49GB空闲，仍须每步保留≥8GiB。板端最近root报告仅约1.27GB空闲，完整新增0.6GB会跌破原≥1GiB余量，因此**优先一个34MB图的C1280 overlay**；大扩栈前须root清理授权范围内临时副本或重新确认空间，本次没有板端查询。

RAM仅给规划余量：最小C512 driver预计与C256约0.70GB同一量级并增加KV/mask工作区，暂为0.72～0.85GB保守规划，非实测；加一个bulk应另预留0.35～0.50GB（大M更多）。embedding文件映射最多另有543.6MB页面，PSS与driver映射可能重叠，不能简单相加后当系统总占用。真正准入要用板端MemAvailable、PSS采样覆盖及speech驻留负载验证，资源不足就不加载额外bank。

以上是渐进方案，不是已编译或已部署的完整C512模型；没有由双层速度外推100ms达标。
