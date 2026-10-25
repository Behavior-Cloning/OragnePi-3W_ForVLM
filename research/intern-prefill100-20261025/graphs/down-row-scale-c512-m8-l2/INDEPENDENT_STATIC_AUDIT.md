# C512/M8 逐行归一化 down：独立静态审计

静态文件契约通过；没有板卡、SDK、编译或推理调用，不批准数值质量、速度或默认部署。候选 NB SHA：2732f92a517d8b74ce221750f1020a8e2bb563ede5bb65d38c1739d143eaa884（28,736,560 B）。

源图为 C512/M8/L2 的已有 range16 I16 down。独立回退新增节点/重接输入后，graph 与源图完全相同。只新增每层 abs → reduce-max → clip → divide → restore-multiply，共 10 节点；两个既有 I16 converter 输入转接到归一化值，fl 改为14，两个 FC 输出改 FP16。其他旧图/量化键保持原值。

生成 C 确认 reduce 对 [4864,8] 的 axis0 求 max，得到 [1,8]，按 token 独立归一化；divide 广播为 [4864,8]，restore 广播为 [896,8]。scale clip 为 [2^-14,65504]，中间新 tensor 为 FP16；归一化结果经 I16 DFP fl14 后送原 FCL，FC 输出与还原乘法为 FP16。每条生成 C 连接已独立逐项核对。改变 FC 输出精度也删除了两处原生成输出 DATACONVERT，属于预期改变。

46 个常量别名的 dtype、字节长度及原始字节全部相同，14 个矩阵权重的网格也全部相同。完整 decoder.export.data SHA 均为 50bc2f49cd09df57aabdc0a9154915bb3891a9fdce445500228091f5f765e55f。仅两个全零 INT64 bias 的派生 FL 改变：L2 28→30、L3 26→29；数值仍全零。不能声称所有量化元数据相同。

16 个外部 normal tensor 生成 C 属性、IO 顺序和 nbg_meta 全等。这里是生成文件核验，实际 SDK 端口属性尚待 probe 在运行时记录。

在实数代数、有限输入且 bias=0 下，s·((x/s)W)=xW；本候选实际两个 bias 均为0，必要条件成立。实际 FP16 divide/FC输出/乘法、I16 fl14 舍入、此前 I16 权重误差以及 NPU 累加仍会产生误差，不是数值无损。fl14 可表示 [-2,1.99993896484375]，归一化峰值不大于1；实际特殊值、下溢清零、累加溢出均未通过编译成功证明。

离线误差结果是两条复用 prompt 的真实原NPU激活，配 FP32 矩阵计算及指定 FP16 边界，不是 NPU 内核模拟器。十个新增算子可能增加开销，离线误差变小不能推出提速。后续必须先同输入 NPU replay（含pos0/AA/finite/端口），再谈整模传播、自然语义、system缓存、真实延迟和资源。完整100ms目标仍未达成。

## 本次 M8 增量

来源为 capacity512-range16/cases/prefill8/l2；相对该I16来源，除每层5个归一化/还原节点及4个既有激活量化键变化外，图、权重、其他量化完全相同。10新节点与相关14个量化键也与已审C1280/M16、C512/M32的变换一致。常量payload同50bc2f49…，两个zero bias仅派生FL28→30、26→29；两个旧输出DATACONVERT因FC输出FP16而去除。

11输入/5输出几何字面核验：hidden[1,8,896]，history KV[1,2,512,64]，mask[1,1,56,520]，rope[1,1,8,64]，新KV[1,2,8,64]。生成C的16外口和nbg_meta与源M8完全相同；未执行实际SDK端口查询。

此M8仍未板测，也未纳入正在运行的M32-only A/B/A；不得改动该基线的尾块后继续沿用同run名。后续独立实验若换成此M8，必须明确“主块+尾块两处更换”，分别核对true_tail8_calls与实际端口、whole-model logits/自然语义、TTFT/资源。相对原精度M8的两处I16 down权重并非无损；既有M16或M32局部结果不能代替本版实测。
