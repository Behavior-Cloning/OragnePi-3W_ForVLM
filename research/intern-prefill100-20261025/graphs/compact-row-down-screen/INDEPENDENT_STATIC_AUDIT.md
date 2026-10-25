# Compact history + row-down 独立静态审计

三图静态文件审计通过，未进行SDK、板卡、锁、编译或推理调用，不批准数值或速度。独立审计由图/常量复核与fixture逐元素复核两部分合并。

m64-plain仅10处几何尺寸512→128变化；逆向恢复后graph与原版完全相同，量化配置与完整生成payload保持。m64-row、m256-row在此基础增加既有12节点（两组逐row abs/max/clip/divide/恢复及converter）、4条重连；源权重数组不变，但实际两个down由原FP16改为既有I16（fl16/fl15）。不能把源weight文件相同写成实际精度无损。

三图46常量别名逐dtype/量化属性/shape/原始bytes检查；每图14个矩阵、14个bias，row与已审row参考完全一致，其余12个FC与各自原图完全一致。两down bias实际payload均为零。额外核对所有perchannel scales/zero_points数组的数值文字，不能仅比较C指针名。row reduce实际沿[4864,M]的通道axis0，对每一row归一化；floor2^-14、max65504、I16fl14输入、FCoutFP16与参考相同。

生成C/meta只缩短四个past和两个mask输入的形状，其他外部端口、dtype、量化、IO顺序相同。此处不是SDK实际端口查询；需要后续probe日志进一步确认。

M64与M256 compact fixture均逐11段核验：删除的384历史KV位置全零、mask列全-10000；保留全部128有效历史、所有query key列；hidden/RoPE逐字节不动。M64输入SHA4899d29e…（1245184字节），M256为069b0c2e…（6946816字节）。参考文件保持原canonicalFP32参考，不能当新量化模型oracle。超过128有效history的后续位置需要更大bank，不可用丢历史冒充加速。

|候选|NB字节|NB SHA256|
|---|---:|---|
|m64-plain|33488288|d97d1ee81b439538967782f22bc4b079c3a66bc36ab5e8aa7496fa7fff6ce260|
|m64-row|29293936|556fd183b0e566a7b4b29426d21b660e3800c9e8adac400012a95010f31f9109|
|m256-row|33837216|6291f4e4c4737730acc6797152def9f4001d4db5cb200ebea382505b5541bd68|
