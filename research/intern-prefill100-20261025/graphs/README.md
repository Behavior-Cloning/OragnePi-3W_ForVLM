# Intern prefill ≤100 ms：图侧候选

目标是同一 Intern NPU 文本权重，固定静态前缀 64/128，动态正文 64–256（另加实际聊天模板），prefill 与 TTFT 均争取 ≤100 ms。当前仅编译单块候选；没有声称达成目标。

## 当前编译矩阵

- C512、M16/32/64/128/256；普通 L0（层0+1）和敏感 L2（层2+3）。L2沿用现役 repaired-v3 原数据。
- 最长输入 128+256+8=392，固定32输出的保守容量需求424，C512可容纳。容量增加不是减少动态输入。
- `compile_candidates.py --execute` 顺序编译，不调用板卡；`status.json` 与 `deployment-manifest.json` 每个图完成即更新。
- 路径：`cases/c512-m{M}-l{0,2}/export_nbg_unify/network_binary.nb`。这是单块图，不能直接当完整24层模型部署。
- `source-provenance.json` 固定 HEAD、Docker镜像、现役板端图SHA及源数据。原 `decoder.quantize` 字节、原始 `decoder.data`、实际 `export/decoder.export.data` 均逐SHA相等；数据只读同文件引用。仅图的C/M整数维度变更：M16改10处，其它M改49处。无重校准，无重新量化。
- 编译器：ACUITY 6.30.22，Vivante IDE5.11.0，目标 VIP9000NANODI_PLUS_PID0X1000003B。旧脚本没有声明M上界；实际SDK是否接受每个M，以各图编译结果为证。编译器的x86模拟运行时长不能作为板端速度。
- 每图限时600秒、4 CPU、8GiB内存、无网络；顺序执行。图侧硬空间预算2GiB，F盘保留至少8GiB。原始和导出常量用已核验symlink节省磁盘，NB自包含不依赖这些symlink。

## 板端对照契约

11输入：hidden、两层各past K、past V、mask、cos、sin；5输出：最终hidden、两层各new K和new V。以各图 `conversion.json` 的实际导出顺序为准。

所有M应使用同一高层输入ID、同一静态缓存、同一有效历史KV/RoPE位置。固定静态64/128须仍以16粒度构建，不能随bulk M改变复用量；不能提前算动态KV。大M剩余输入由16/8尾图处理，不能丢token。L2需前两层生成的真实hidden与历史，不能套用L0输入。

单次M16与M64/M128/M256的前16行可以做同义数值对照；完整吞吐比较须处理同样总token数量并逐批推进KV。API耗时除以M仅是单块指标，不能外推全模型TTFT或≥100ms目标。最终需要每个runtime同IDs/cache、全文logits固定反馈、自然中文输出、音频并发、连续稳定性及driver释放验证。

## 有效历史裁剪（后续候选）

保留外部C512缓存契约，内部仅裁掉必然无效的历史。对最长system128、新增动态264且尾8的调度：M64的最大bulk开始位置320，M128为256，M256为128，可分别用H320/H256/H128，令past+current总宽均384。调用位置必须满足pos≤H，且被裁列全部是无效mask。短请求/静态构建/其它调度必须单独检查或回退。需要复制新增中间张量的原量化签名，并核查导出端口与payload；编译通过不证明数值等价。新增slice/concat可能抵消收益，须A-B-A实测。

初次无keeper编译被WSL终止，证据在 `attempts/wsl-unkept-20261025`，未计为有效。现采用显式WSL `sleep21600`保活；启动脚本也检查keeper存在。编译日志全部保留。

## 首组实际构建结果

10/10 编译成功，独立审计通过（`independent-audit.json`）。NB合计 324900928 字节；图侧新增常规文件 391975021 字节（含此前WSL中断证据）；F盘余 25927299072 字节。编译累计 167.863 秒。共460个原始常量数组独立重哈希；全部端口精度与源图相同。

|M|L0 NB bytes|L2 NB bytes|
|---|---:|---:|
|16|23975984|33184080|
|32|24760720|33921520|
|64|25762288|34961104|
|128|29002000|38190960|
|256|35976976|45165296|

内部裁剪候选暂缓，待首组同义fixture板测后决定。当前无单块NPU数值或延迟结果，不代表100ms目标已达成。

## 严格共同输入 fixture（当前板测应选这个）

`derive_canonical_fixtures.py` 从每个首层编号唯一的 M256 原FP32历史中，逐位切出 M16/32/64/128/256；新输出位于上级 `fixtures-canonical/`。此前各M独立重算FP32前缀会出现微小差异，经FP16/INT16舍入后仍有部分输入不同，因此旧 `fixtures/` 只作历史证据，不用于严格batch对照。

`canonical-fixture-independent-audit.json` 核验10组、110个输入字段完全来自对应canonical切片；11端口shape、7组GQA重复mask、512个历史槽位和当前M列的因果语义均检查。每图独立重算原FP32 graph与实际导出权重的FP32参考，跨M有效行最大相对L2分别8.17e-7、1.15e-6，属于CPU运算顺序差异；输入本身严格相同。不能把该FP32参考当成已量化激活/已部署NPU缓存的精确oracle。

完整历史来源仍是Q8源权重的FP32因果前向；L2输入经过原层0/1，layer2/3各自历史正确。尚未包含全模型NPU累积激活误差。板端需对照所有5个输出，记录finite、相对L2、最大abs、反复运行和资源释放；当前源/shape/fixture审计通过不等于NPU数值已通过。
