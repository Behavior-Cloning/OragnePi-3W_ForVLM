# Llama3.2 compatibility

See REPORT.md. One actual GGUF layer0 executes on NPU but precision failed (6.05–35.08% hidden relative L2). MLP FP16 exception still fails (5.48–35.18%) and costs 4.6x layer latency. Full model unvalidated. Do not omit GGUF rope_freqs scaling. Original BF16 precision is not established by Q4-to-W8 conversion.
