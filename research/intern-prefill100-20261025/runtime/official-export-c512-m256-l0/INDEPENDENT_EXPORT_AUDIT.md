# M256 BITCAST 独立导出审计

审计通过。仅只读 Windows 文件及重算输入/常量，没有重新编译、运行模型或访问板卡。

- 14 weights + 14 bias：实际 dtype、offset、整数 raw bytes 和按真实 PC 通道轴展开的反量化 f32 均与原图相同。整个 payload SHA 为 `839fa51843b33140e864d01423f9da81e14a996cbe3d666c57fa0e84040a55e2`。
- 12 个常量 qtype 元数据发生变化；14 个内部 I16 转换逐个从实际 C 输入边与张量属性核验，全部 65,536 整数码的 DFP/affine 网格逐字节相等。相对 affine C 只有 14 个 DATACONVERT→BITCAST opcode 修改。
- plain 与 BITCAST 的 11 个真实文本输入均重新读取为 f32，完整重建 canonical 输入 13238272 B，SHA `e3e1224831b8b33be92e3d3c2ee26c7d30e9ea3b9eb5ff8fe1583de18718a932`；argv 顺序、官方 main 以及 autosave 环境通过。
- plain NB 逐字节重现原 M256 NB；BITCAST NB `907689cdee9a1f767f4b64d8854bcbae27b5c16871f26500519ed977fee4c250`，35975168 B。
- 实际 NB 查询的 11 输入/5 输出原始元数据两版逐字节相同，并逐字段匹配 shape、dtype、方向与有效 DFP fl；未把 inactive union 的 scale 垃圾值解释为 affine scale。

限制：等值网格和端口不代替板端输出/延迟验证；不代表整模 TTFT、动态 256 tokens 的实际全链路收益或语义正确率。
