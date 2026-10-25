# I16 down 保留FP16输出：独立静态审计

**结论：四个最小图与新两层候选的已检文件没有发现额外未声明改变。仅通过导出与静态一致性审查，不代表板端数值、速度、引擎路径或整模质量合格。**

审计只读取Windows/WSL文件，解析生成C、计算SHA和NumPy常量对照；未调用SDK、未访问板卡、未改运行源。根节点已完成的SDK端口查询结果仅作只读复核。

## 四个最小图

|项目|全I16控制|FP16输出版|
|---|---|
|输入|I16 DFP fl12|不变|
|权重|I16 DFP fl16|dtype/码值/字节不变|
|偏置|I64 DFP fl28，全0|不变|
|输出|I16 DFP fl14|FP16，无量化|

- FC/Conv各两版均rc0，实际NB SHA与各自记录相符；每组图JSON和源data逐字节相同，仅project/output两条输出quant改动。
- 四图实际导出payload均为SHA58dc987de2364196ad5ce532896f8d9a9c453e8215ecc1412334e0b427a6f058。生成C的weight/bias类型、偏移、字节和量化属性一致。
- 独立将源[4864,896]/HWIO权重转换为实际OI连续布局，执行fl16 nearest-even+饱和后，全部I16码值逐元素匹配；零偏置也匹配。直接按源flatten比较会错误忽略布局，已显式处理。
- 生成C：FC仍直接FCL；Conv仍前后PERMUTE+CONV2D，只有算子结果/外部输出类型改变，没有新增偷偷转换回I16的生成节点。此观察不能推断NB内部实际引擎。
- 根节点保存的四个实际NB端口JSON独立复核通过：输入I16/fl12完全相同；输出同shape、I16/fl14→FP16/NONE。

## 两层C512/M32候选

- NB SHA a10a3aa5c978509852d4607b40117fab7d1947cc624cb10b510e4897e6ca3c75，29,747,392字节，rc0导出。
- 相对range16版本，decoder.json和decoder.data逐字节完全相同。量化文件仅@b0_fullconnect_116:out0及@b1_fullconnect_116:out0改为FP16；其余配置不变。
- 整个decoder.export.data为SHA50bc2f49cd09df57aabdc0a9154915bb3891a9fdce445500228091f5f765e55f，与range16逐字节相同。
- 独立检查全部实际34个NEW_CONST_TENSOR声明/46个语义别名：dtype、shape、量化赋值、offset、长度、payload SHA及scale/zp数组不变。NET_CONST_TENSOR_NUM=64是分配上限，不能误报为64个实际常量。
- 14矩阵和14偏置均被覆盖。b0 down权重仍I16/fl16、bias I64/fl28；b1权重I16/fl15、bias I64/fl26。两个down结果现在是FLOAT16/NONE。
- 派生变化已明确：生成节点194→192，删除b0_reshape_117_dtype_converter_11008与b1_reshape_117_dtype_converter_11009两处输出DATACONVERT；保留输入转换。其余共同节点算子/参数一致，未新增节点，其他生成张量属性不变。
- nbg_meta一致；根节点实际导入NB得到的16个外部端口，与原C512/M32原精度图完全相同。已复核查询脚本、工具SHA、结果文件与候选NB SHA，不重复调用SDK。

## 限制

- Static export review and recorded SDK port-query review only; auditor made no SDK, board or model-inference calls.
- rc0/NB generated and matching ports establish exportability/interface only; no board numeric, performance, engine route or full-model qualification.
- New candidate keeps prior I16 weight quantization, not original FP16 weights. b0 input fl12 range[-8,7.999756], b1 input fl11 range[-16,15.999512]; first-system-chunk input outliers remain unsafe.
- Down output FP16 removes fixed-point output-grid saturation but does not prove correct accumulator precision, no clipping inside SDK, or input-range correction.
- Actual external16-port equality cannot reveal internal engine placement; generatedFCL/Conv or removedDATACONVERT is not NN/TP/CPU mapping evidence.
- DFP port records have overlapping quantization union fields; use quant_type and fl. Tiny displayed scale values for DFP are not meaningful affine scales.
- Four minimal cases compare source integer grids and constants; generalizing their result to full Intern or 64-256 input quality requires independent numerical tests.
- Prior semantic48 windows and first-chunk issues were development evidence. No new holdout quality, thermal, resource, long stability or100ms result exists here.

逐常量、生成类型与源码SHA见 INDEPENDENT_STATIC_EVIDENCE.json、INDEPENDENT_STATIC_AUDIT.json。
