# 新增陪伴模型：电脑预下载

2026-10-10更新：两个模型已分别于03:07和03:08下载完成并通过SHA-256校验，09日的网络失败已恢复。
已在电脑上再次核对两份哈希。用户重新接通板卡后，两份权重均已传输、在板卡再次校验并部署；首轮陪伴实测已完成，见[结果](陪伴模型首轮观察.md)。

以下为2026-10-09下载过程记录：
Hugging Face / hf-mirror 和 ModelScope 下载均遇到低速、SSL超时或连接重置。
已保留分块，并改用每模型4并发的有限重试后台任务（每模型最多12轮）。
电脑须保持开机联网；这不是自动评测或自动开机任务。

| 模型 | 目标文件（项目根目录下） | 字节数 |
| --- | --- | ---: |
| Qwen3-1.7B Q4_K_M | models/qwen3-17b-q4/Qwen3-1.7B-Q4_K_M.gguf | 1107409472 |
| Qwen2.5-1.5B-Instruct Q4_K_M | models/qwen25-15b-q4/qwen2.5-1.5b-instruct-q4_k_m.gguf | 1117320736 |

总计约2.23GB（2.07GiB）；分块和拼接缓存会额外占用空间。
Qwen3采用Unsloth量化版；“非思考”是下次部署时要验收的运行配置，并非另一份权重。

源版本由 `configs/vlm-models.json` 固定。ModelScope镜像的单文件记录与源SHA一致：

- Qwen3镜像仓库 `unsloth/Qwen3-1.7B-GGUF`，版本 `1e3f49488e445e2148d2e065ea0612c320da65c3`；SHA-256 `b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897`。
- Qwen2.5镜像仓库 `Qwen/Qwen2.5-1.5B-Instruct-GGUF`，版本 `e8b19c78f775ccbcf6df15ceace6bb5276f09765`；SHA-256 `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`。

执行入口：`bash scripts/predownload_companion_models.sh`。重复启动由flock阻止。
日志：`outputs/model-downloads/qwen3-17b-q4.log` 和 `qwen25-15b-q4.log`。
只有日志出现 `VERIFIED` / `Already verified` 且随后 `COMPLETE` 才代表该模型下载校验成功。
下载器只在整文件SHA-256匹配后发布最终 `.gguf`；`.chunks` 和 `.assembling` 不可部署。
失败后可重跑同一入口续传；不要把缓存目录体积当作有效下载进度。
