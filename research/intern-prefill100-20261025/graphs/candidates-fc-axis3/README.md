# C512 M32 L0–1 FC axis3 单块候选

SDK 编译和静态审计完成；未做板端测试，未替换默认模型。NB 位于 `fc-axis3-c512-m32-l0/export_nbg_unify/network_binary.nb`，大小 24,760,720 bytes，SHA `33e3629e5a5be9400ff187c30c7f25762be51e6708035f04560802e955d9e457`。

只有 28 处图参数改变：14 个 FC 的私有输入 reshape 从 `[M,I]` 改为 `[1,1,M,I]`，对应 fullconnect 的 axis 从 1 改 3。fullconnect 算子和 `[I,O]` 权重形状保持，后续原 reshape 不改。原 `decoder.data`、`decoder.quantize`、`tensor-inputmeta.yml` 和编译导出的完整 constants payload 均字节相同。实际 14 个 I8 权重、14 个 I64 bias 及28个 FC 输入/输出 I16 的量化格式、整数网格、scale 和 channel axis 均逐项通过；见 `deployment-manifest.json`。

原 source-f32 图与候选 FP32 全5输出等价检查通过，实际已导出 FC 权重反量化版本也通过。此项为数学图核对，不能代替 NPU 输出验收。

实际生成 C 仍是14个 `VSI_NN_OP_FCL`，未切到 Conv 或带 axis 的另一个算子。14个 FCL 实际输入/输出二维形状与原图完全相同；SDK在前后加入28个 reshape，RESHAPE2由32增至60。记录在 `lowering-shapes.json`。这些reshape是否被后端消除及性能影响尚未测量，不能声称获得吞吐提升。

外部11输入/5输出的顺序、名称、shape、dtype及量化元数据与原图完全相同；canonical输入与参考可原样使用，路径和SHA在manifest中。原图 NB SHA为 `d7adc37f977973a3ce3d40be0d2c1e8539ba187ce69f2fe12f6280d1b5b7d878`。

第一次审计脚本错误地将 `verify_contract(folder,width)` 的第二参数传成 first_layer=0，导致断言失败；实际应为fusion width=2。已保留 `initial-audit-failure.json`、初始脚本和日志。修正参数后只运行 `audit_fc_axis3.py` 读取已编译产物，未重编NB、未重排端口。先前短暂的“SDK端口重排”判断已纠正。

编译使用唯一 `/tmp/opi-prefill-compile.lock` 串行锁。初始脚本未在审计异常前保存纯SDK耗时，因此manifest中的该值为null，不能用包含FP32检查的脚本总时间充当编译时间。原始大权重通过已验证symlink复用，重复export payload也经SHA核对后链接到原图；`dedup-export-payload.json`记录清理。

板测应先用同canonical输入做原图→候选→原图重复，分别核对5个输出与释放、峰温、P50/P95。只有单块验证通过后才考虑进一步实验；本次不扩栈。
