# FuseChat Llama 3.2 1B 中文增强模型：Q8_0 来源

2026-10-23 完成下载及 SHA-256 校验。本目录只包含与现有 Q4_K_M 相同模型、相同镜像 revision 的 Q8_0 权重，不是另外的中文模型。

- 上游微调模型：`FuseAI/FuseChat-Llama-3.2-1B-Instruct`。
- 量化仓库：`bartowski/FuseChat-Llama-3.2-1B-Instruct-GGUF`；实际下载来源为 ModelScope 镜像。
- Revision：`726613878e6c586af188734e31c3b2c527c4502c`。
- 文件：`FuseChat-Llama-3.2-1B-Instruct-Q8_0.gguf`，1,321,079,360 bytes。
- SHA-256：`c7db2b30d2b176998db90304d1ac6dd259379bd75f9ea8b6a094be88ea47b1b4`。
- 精确请求 URL：https://modelscope.cn/api/v1/models/bartowski/FuseChat-Llama-3.2-1B-Instruct-GGUF/repo?Revision=726613878e6c586af188734e31c3b2c527c4502c&FilePath=FuseChat-Llama-3.2-1B-Instruct-Q8_0.gguf

GGUF 直接读取核验：16 层，hidden 2048，FFN 8192，32 个 Q heads / 8 个 KV heads，head dimension 64，词表 128256。147 个张量中，113 个类型为 Q8_0（type code 8），34 个为 F32（code 0），存储张量元素合计 1,235,814,432。全部 tokenizer 元数据逐字段哈希、全部张量名称/形状、全部 llama.* 配置均与现有 Q4_K_M 相同。

这次仅准备和校验 Q8 原始 GGUF，没有启动板卡推理或将它当作已完成的 NPU 模型。后续仍须分别注明 GGUF 源量化和 NPU 导出混合精度；Q8 源有助于接近 Intern 旧 Q8 实验，但不意味着两条路线所有算子与校准条件相同。

证据：`source-file-record.json`、`download-manifest.json`、`download.log`、`gguf-inspection.json`、`q4-q8-structural-parity.json`。下载器从原 `download_models.py` 复用单文件下载函数，保持断点续传、空间底线与完整哈希校验。token 仅从受保护配置读取，不写入源码/URL/日志。

模型卡与许可链沿用父目录的 `original-README.md`、`original-config.json`、`base-LICENSE.txt`、`base-USE_POLICY.md`；中文训练数据和作者基准均为模型卡陈述，不能代替本板精度验证。
