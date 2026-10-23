# Intern W8 operator/runtime experiment

See REPORT.md and the independent audit at ../vlm-molmo-paligemma-20261023/AUDIT.md. Completed paired board experiments; no service replacement. Models unchanged. Source and summaries are versioned; raw events/logits retained under results/.

Do not label exact-prefix quality lossless: natural 8/11 exact matches; fixed-feedback logits differ. Fused decode adds 107.66 MiB for about 0.9% throughput; not recommended. All 8 NPU sessions released driver allocation to zero.
