# 共驻r1内存保护退出

视觉Q8重排worker和Intern NPU语言worker加载期间，MemAvailable降至43568KiB（约42.5MiB），低于96MiB保护阈值；当时温度约40.610°C。未开始任何VQA请求，不计性能样本。两进程正常close，NPU driver=0。不是模型推理准确率失败，也不是热保护。

后续已对/tmp/opi-vit-matrix-r1全部文件与Windows原始文件逐SHA/长度核对并再次检查后删除临时副本，释放225867824bytes。恢复清单vit-npu-r1/matrix-board-cleanup.json，模型基线及当前VLM使用图未删。独立r2使用同代码/同阈值重新测量，r1失败证据保留。
