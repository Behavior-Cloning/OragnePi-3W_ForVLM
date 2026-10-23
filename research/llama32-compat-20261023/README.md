# Llama3.2 compatibility

See REPORT.md. One actual GGUF layer0 executes on NPU but precision failed (6.05–35.08% hidden relative L2). MLP FP16 exception still fails (5.48–35.18%) and costs 4.6x layer latency. Full model unvalidated. Do not omit GGUF rope_freqs scaling. Original BF16 precision is not established by Q4-to-W8 conversion.

## 全 FP16 追加（2026-10-23）

`compile_all_fp16.py` 复用导入图并硬链接原权重，不覆盖前两版；`analyze_all_fp16.py` 独立比较三版同六状态输出。全 FP16 隐藏态 L2 0.0487%–1.1043%，单层 API 中位48.134 ms，驱动122.35MB、释放0。改善支持继续定位量化路径，尚无完整16层NPU生成结果。详见 REPORT.md 新增章节。

追加中文留出：attention FP16 + 3MLP W8候选15状态粗筛通过，单层16.248ms、最大L2 4.0912%；全FP16配对48.299ms、最大1.1043%。完整中文生成待全16层实测，源仍Q4 GGUF。详见REPORT最后一节。
