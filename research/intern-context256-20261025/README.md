# Intern context256：单块预筛选

当前只完成 Windows/WSL 导出、编译及静态审计，**没有运行板卡**。候选仅包含 layer0+1 的双层融合 decode 与 prefill16，不能用于宣称整模型加速或精度通过。

## 源与量化

来自现役 c1280 相同 l0/l1 校准源，使用 compact GQA、split KV、INT16 softmax、gate/up DFP-MSE max-extra-bits=1。旧 NB 与 `active-source-match.json` 中既有板端证据 SHA 一致。新旧 `decoder.quantize` **逐字节相同**；decode 48 数组/29,847,824 元素、prefill16 46 数组/29,847,822 元素逐项相同。JSON 图全部变化都局限于容量相关整数减少1024，没有重新校准。详见 `graph-quantization-audit.json`、`source-provenance.json`。

这是 ACUITY 已校准图直接导出 NB；没有另造 ONNX，也没有以 ONNX 转换替换源模型。

## 可交付路径

- `artifacts/decode/export_nbg_unify/network_binary.nb`：SHA 868aa8f01e2b58b1f351da93e9a2ef7ad6d58e139d1a78b5ef9e3d3cca190abf
- `artifacts/prefill16/export_nbg_unify/network_binary.nb`：SHA bba28d9f9920b19aa0426580a6350f67f530e3df2e8e8e6ef6055592d7f8f0c4
- 两个目录各有 `decoder.json`、`decoder.data`、`decoder.quantize`、`tensor-inputmeta.yml`、`conversion.json` 和原始编译日志。
- `tensor-probe-arm64`：以现有 `tensor_probe.c`/生产 IO 转换逻辑交叉编译，命令见 `probe-build.json`，副本在 `probe-source/`。
- `fixtures/cap1280-decode/probe-input.f32`、`fixtures/cap256-decode/probe-input.f32`
- `fixtures/cap1280-prefill16/probe-input.f32`、`fixtures/cap256-prefill16/probe-input.f32`

## 输入对齐与边界

冻结同一个 `s128-d64-0` 中文家庭场景（完整输入200 tokens），先取真实有效 history=128，再取1或16个query。history来自原始Q8 GGUF的独立FP32模型，不是抓取的板端量化KV；两种capacity使用**同一个**已生成输入，缩短版本只裁掉masked padding中的 `[256,1280)` 槽位。有效KV、位置、hidden、RoPE、当前query全部原样保留。crop脚本断言被移除mask均无效，绝不重算有效history。FP32图层面两种capacity的5个输出逐字节相同，见 `fixture-graph-parity.json`；这不能替代板端数值验证。

DIM896/KV2/HD64/FUSED_LAYERS2、K未转置。每图11输入：hidden，然后两层K、两层V、两层mask、两层cos、两层sin。5输出：最终hidden、layer0新K、layer1新K、layer0新V、layer1新V。仅输出当前query，不输出history，因此同query比较全部输出即可；不存在要排除的历史padding输出。

本轮整模型若扩展，最长prompt200+输出32共232，满足256；**不要套用64输出全矩阵**，最长请求超出256。当前单块history128+query16=144。还未覆盖接近容量边界或NPU实际KV输入。

## 父控制器板测契约（本目录没有执行）

每次受同一双锁控制、同一CPU亲和/温度门限；先cap1280，再cap256，再cap1280复测，防漂移。每图3次warmup、30次计时，NPD关闭；`tensor-probe` 统计的 API 时间只包 `vip_run_network`，不含flush/invalidate、load或fixture读写，不能称整链路TTFT。

```sh
OPI_PROBE_WARMUP=3 taskset -c 4,5 ./tensor-probe-arm64 GRAPH.nb INPUT.f32 OUTPUT.f32 30 RESULT.json
```

cap1280基线直接用现役 `layer0.nb` 与 `prefill-layer0.nb`（须校验上述现役SHA），分别配cap1280输入；候选配对应新NB和cap256输入。每次检查 `released_bytes=0`、`differing_repeats=0`。用 `compare_probe_outputs.py --old BASE.f32 --new CAND.f32 --tokens 1或16 --output comparison.json` 输出逐端口max-abs/relative-L2/字节一致性；不要只看余弦。

若需抓取NPU真实有效KV，可用 `crop_fixture.py --old OLD_GRAPH_DIR --new NEW_GRAPH_DIR --input CAP1280_INPUT.f32 --output NEW_DIR --history H` 同样裁剪，务必使用对应query图、H+query<=256。

## 空间与状态

构建前F可用28,679,995,392 bytes；两图编译总约29.85秒；当前本目录常规文件逻辑字节约 479,330,624（含源快照、图、产物与fixture，symlink不重复计数），F当前可用 28,167,127,040 bytes。编译严格顺序，未触碰旧产物，没有任何板卡操作。尚待父任务单块板测；只有通过后才考虑全模型c256栈。


## 后续进展

单块已由父控制器板测通过，见`SINGLE_BLOCK_REPORT.md`。全24图也已完成编译和独立静态审计，见`full-stack256/BUILD_REPORT.md`、`full-stack256/full-audit.json`与`full-stack256/deployment-overlay.json`。前文“没有运行板卡”仅描述最初交付阶段，本子任务全程仍无自行板卡操作。完整模型端侧测试由父任务安排。
