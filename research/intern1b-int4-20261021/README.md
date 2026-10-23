# InternVL3-1B W4板端性能对照

用户授权：2026-10-21测试Intern 1B INT4版本，并检验完整模型能否获得至少50%吞吐提升。

本目录为独立实验，不修改主线程Jev队列、服务或既有模型。权威脚本和结果位于Windows F盘；编译文件位于F盘WSL对应的`/opt/opi-lab/intern1b-int4-20261021`。板端隔离目录`/opt/opi-vlm/intern1b-int4-side-20261021`。

- `USER_TARGET.md`：50%指标定义与测试边界。
- `compile_candidate.py`、`compile_all.py`：保留原模型的激活/KV配置及敏感FP16例外，将其它线性权重量化为四位数值。
- `storage-audit-l0.json`、`weight-reconstruction-audit.json`：生成C和真实常量显示INT8容器−7..7，未证明packed INT4硬件算术；scale和期望量化值已核查。
- `probe-summary.json`：两层固定合成输入的执行性能。该输入的hidden差异很大，不能由小图速度判断语言质量。
- `run_full_benchmark.py`、`board_benchmark.py`：双锁、独立模型目录、W8/W4交错会话，模型及system-prefix KV常驻，采样温度/CPU/内存/swap。
- `build-manifests/`：28个图的编译命令、源文件hash、量化范围、产物hash。
- `summary.json`、`verification.json`、`REPORT.md`：已完成实测与独立核验；速度/质量分开验收。

本次为NPU文本decoder，不含图片编码器、ASR、Jev或真实完整语音agent。既有CPU Qwen服务保持空闲常驻。候选仍是实验品，不部署替换用户服务。

追加用户要求：首token优化，以及以VLM的动态prefill/system KV复用为重点。`ttft-tail/`保留尾部合批候选与60个计时请求；`cache-study/`保留64/128-token system KV命中/重算的36个计时请求。当前接口尚未接入视觉encoder/projector，不能称完整VLM验收。
