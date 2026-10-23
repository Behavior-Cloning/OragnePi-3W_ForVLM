# MiniCPM-V4.6 isolated experiment — 2026-10-23

Read REPORT.md and results/w8-final-summary.json. W8 works via explicit FC converters, calibrated INT16 boundaries and FP16 sensitive ops. Conservative grouped 3.766 t/s; attention-only integer MatMul candidate 4.136 t/s. Fastest 5.917 t/s fails repetition checks. Same-text Intern 15.523 t/s. Full vision and production precision remain unfinished.

All experiment compile/inference/validation processes exited; NPU released to zero. Main Git index/HANDOFF and existing services unchanged. No subagents used.

Board keeps w8scaled-all, w8fixed-attn-all, w8fixed-linear-all, w8fixed-all and shared embeddings. Old FP16 NBG copies removed only after WSL SHA verification; board baseline-fp16 alone cannot run. WSL compiled-epssafe NBGs retained. ONNX/decoder.data/object cleanup requires re-export before recompilation. Source, calibration, graph metadata, raw outputs and NBGs retained. Never rsync --inplace into hardlinked models.

Use board_access hostlock and /run/opi-npu-kv-benchmark.lock. Code remains in this isolated task on Windows F/WSL; Mac report is a copy. Remaining: precision validation, sensitive recurrent-state handling, batch prefill/prefix state cache, controlled long context, full image encoder, long-run telemetry. Do not claim MiniCPM matches Intern.
