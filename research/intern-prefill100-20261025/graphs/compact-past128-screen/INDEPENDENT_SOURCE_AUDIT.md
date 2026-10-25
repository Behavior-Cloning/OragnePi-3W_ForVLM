# Compact past128 三图增量审计

**通过源码/图/fixture独立检查，尚非板测资格结论。** 仅Windows/WSL文件读取，无SDK、编译、模型、锁或板卡操作。

M64/L0、M128/L0、M256/L2各图均只有声明的10处几何变化；逆替换可恢复整个原图，原quantize文件、source.data和实际导出payload逐字节相同。生成C常量声明逐字节同，nbg_meta仅四路past和两路mask的输入形状改变；其他输入与全部输出shape/dtype/quant保持。

三份compact fixture的全部11段均由审计独立按轴重排核对：保留前128有效KV，删去的384槽为零；mask保留有效历史和全部M个query列，删列均−10000。hidden、绝对RoPE、token IDs和输出参考逐字节保持。L2版本严格对应原L2权重与L2 FP32参考history，没有混入L0或此前I16 down候选。

| 图 | NB SHA前缀 | compact输入F32字节 | 输出F32字节 |
|---|---|---:|---:|
| m64-l0 | 96fa305b43f264fb | 1,245,184 | 360,448 |
| m128-l0 | 1fc1d0618d5029e5 | 2,686,976 | 720,896 |
| m256-l2 | 9e028961f7326501 | 6,946,816 | 1,441,792 |

这些仍是三条canonical FP32派生history轨迹，不是完整24层/真实NPU KV分布或自然质量验证。保留有效输入不保证硬件bitwise；后续纯API需要相同新probe、对应C512对照与重复、全端口finite/数值及释放检查。C128表示past物理输入，不应当限制或截断真实history。完整动态64–256和100ms目标仍待验收。

## 本轮控制器补充

已核 `controller_compact_past128_screen.py`（d64af076…）和冻结config。三图各按原C512→compactC128→原C512重复顺序，九组全部对应正确layer/M/原图SHA/候选SHA/fixture及参考；同probe18b295…、未改runner a67ace…，每组3warmup+30纯API，共27warmup/270纯API样本。三份fixture metadata独立冻结，上传文件与source SHA在拿锁前后及上传前后/远端校验；所有NB和两种输入计入tmpfs与MemAvailable预留，双锁、唯一目录、有界超时与失败partial回收/不重启保持。当前仅核源码与契约，不把控制器已启动视为实测成功；未访问板卡或锁。
