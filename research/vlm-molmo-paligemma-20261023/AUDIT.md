# 独立审计：Intern 优化、模型部署边界与 F 盘清理

审计日期：2026-10-23。审计只读取 Windows F 盘/WSL 文件、原始事件及源码；没有登录板卡、运行推理、持板卡锁或删除文件。本文件为本审计者唯一写入。实测由独立控制器执行。

## 审计结论

终审补充：Pali 完整 CPU 图文四请求及无图中文三请求已验收原始记录；Pali 单层 NPU 两版与 Llama 新两版15状态配对已独立复算。WSL compact 已成功，详见末尾终审。完整16层中文 Llama NPU 和后续Q8源公平比较仍是新进行中任务，本审计不标其完成。

- Intern 的 grouped prefill 在本轮证据中保持数值一致，TTFT 改善约 1%–1.6%；decode 没有提升。
- exact-prefix 从 32 个缓存 system token 改为 37 个，短 prompt TTFT 可降至约 0.33–0.38 秒；这是改变 KV 复用及 batch 分区的候选，11 条自然输入只有 8 条输出逐 token 一致，不能标成无损优化。
- decoder/head 分组在新增基线之上约提高 0.9%–1.0% decode，驱动内存多 112,886,872 bytes（107.66 MiB）。本轮两组 logits 及 11 条文本与该新增基线一致，但收益不足以支持默认启用。
- 所有 Intern 数字均是文本 decoder，不包含视觉、ASR、TTS、网络或模型冷加载。约 15.5–15.8 tokens/s 的固定反馈结果不能称为 32 个有效自然文本 token 的生成速度。
- 清理清单的 172 个删除项已逐项复核，必需的源权重、NBG、量化参数和重建来源仍在。WSL 内部释放与 F 盘立即增加空闲是两个指标。

## Intern 原始证据与源码

目录：`../intern1b-operator-opt-20261023/`。

主矩阵：`results/20261023T064955/events.jsonl`，145 条记录，包含 70 次计时、44 条自然回答、8 次诊断、5 次释放、complete；没有 error。融合矩阵：`fused/results/20261023T065909/events.jsonl`，59 条记录，18 次计时、22 条自然回答、4 次诊断、3 次释放、complete；没有 error。

两次 provenance 记录中的源码 SHA256 与审计时 F 盘对应文件全部一致。两次使用相同 28 个 NBG；没有把换权重收益算作调度收益。所有 closed 记录 driver_bytes=0。

静态检查：`batch_prefill.h` 共享的 KV 是各自层的历史缓存；当前 batch 的因果注意力在图内计算，后层依赖前层 hidden，而不依赖前层新 KV。先完整执行 group、再统一写回各层 KV 的依赖关系合理；源码也在 grouped 模式拒绝 host_hidden 和 scalar_block 组合。逐层 hidden/历史 KV 绑定前检查 tensor shape、dtype 和量化契约。实测的全词表逐位一致为这一实现提供了直接支持。

`resident_session.c` 的 exact-prefix 跳过 16-token 整批向下取整；truncate_session 仍关闭旧后缀 mask，至少重算最后 prompt token。它改变 scalar/batch 分区，不能期待与原 reset/分区路径逐位等价；源码原注释仍描述“保留完整 batch”，报告应以开关的实际行为为准。当前测试覆盖 system-only 准备后的请求，没有完成任意历史变化、图片变化和长期多轮缓存失效验收。

## 计时口径与数据复算

所有主矩阵配置使用 tail_min=8、容量 1280、prefill16、CPU6/7、常驻模型。每个请求前 reset 并准备同一 37-token system，准备耗时在 TTFT 外。原基线仅复用 32，exact 配置复用 37；因此原基线仍重算 5 个 system token。控制器明确断言复用量，未把上一轮重复用户输入算入收益。

主矩阵每配置每输入 3 次；末尾原基线每输入复测 2 次。表内为独立从事件重算的中位数：

| 配置 | prompt / reused | short-0 TTFT s | short-1 TTFT s | dynamic-64 TTFT s | dynamic-128 TTFT s |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | 50或54/32；长输入110或170/32 | 0.432657 | 0.620297 | 1.567668 | 2.810113 |
| group | 同 baseline | 0.428392 | 0.615912 | 1.543346 | 2.764679 |
| exact | 同 prompt，reused37 | 0.339687 | 0.384670 | 1.567774 | 2.737579 |
| group-exact | 同 prompt，reused37 | 0.334322 | 0.380343 | 1.543197 | 2.698172 |
| baseline-repeat | 同 baseline | 0.463044 | 0.619119 | 1.565290 | 2.802980 |

short-4 为另一条 54-token 输入，其结果与 short-1 接近。dynamic-64/128 的真实动态文本分别为 73/133 tokens，不能把名称当实际长度。short-0 末尾基线出现约 7% TTFT 波动，因此极小收益应谨慎判断。主矩阵 decode 中位数整体仍约 15.54–15.72 tokens/s，无显著提升。不能将本次 50/54-token、system 已准备的 TTFT 与旧“约 0.8 秒”的不同 prompt/缓存场景直接相减。

每个计时请求强制 32 次输出预测，沿原基线的同一 token 反馈轨迹运行；自然回复不足 32 时补 EOS。decode=(32-1)/(总时长-TTFT)。自然回答另跑到 EOS 或 64-token 上限。测试顺序按配置分段，并非逐请求 ABAB 随机配对；末尾基线复测用于观察漂移，样本不足以支持可靠 P95 或长期稳定性。

## 质量门槛

group：11/11 自然回答与原基线逐 token 一致；short-0 和 dynamic-128 各 16 步全词表 logits 逐位一致。

exact / group-exact：8/11 自然回答逐 token 一致，晚餐、起床建议、休息安排三个输出变化；两配置输出彼此一致。两个诊断场景 argmax 均 16/16，但 logits 相对 L2 分别 3.095%/4.994%，平均 KL 分别 0.0018405/0.0003455 nats。变化后的文本可读，不应单凭文字不同判定质量失败；也不能据此声称无损或产品质量通过。原基线及候选均存在“七点起床建议答成早餐”的问题，基线不是正确性 oracle。这些开发样例没有代表性任务准确率验收效力。

融合矩阵的名为 baseline 的配置实际已启用 group+exact，与主矩阵原 baseline 不同。融合候选相对这个新增基线：11/11 自然回答相同，两组各 16 步 logits 逐位一致；decode 分别从 15.578/15.699/15.617 到 15.727/15.834/15.768 tokens/s，TTFT 基本不变。驱动分配从 753,274,680 到 866,161,552 bytes，应同时披露这个成本。

主矩阵最高采样温度 47.926°C，最小 MemAvailable 3,966,771,200 bytes；融合矩阵最高 49.352°C，最小 3,852,333,056 bytes。两段采样 swap-in/out 增量均 0，NPU 采样频率都是 1,008,000,000 Hz。已有 swap 占用仍存在；不能说系统完全无 swap。RSS/PSS 不覆盖全部驱动内存。

旧 W4 候选的 22.7 tokens/s 不可作为质量通过的替代版本：10月21日报告显示自然回答严重退化、相同反馈 argmax 0/64；生成 C 常量是 INT8 容器且数值为 -7..7，尚无原生 packed INT4 证据。

## 清理保留依赖复核

`maintenance/cleanup-manifest.jsonl` 有 344 行：172 个唯一文件各一条 verified_before_delete 与 deleted。配对的路径、bytes、SHA256 全部一致；审计时所有删除路径均不存在。记录中的 SHA 是删除前计算值，审计者没有在文件已删除后重算原文件 SHA。

| 位置 | 文件数 | 删除字节 | 实际影响 |
| --- | ---: | ---: | --- |
| F 盘 Jev 两个 FP32 ONNX | 2 | 1,421,901,036 | 直接 F 空间，约1.324GiB |
| WSL Intern W4 中间数据 | 56 | 4,280,522,011 | WSL内部约3.987GiB |
| WSL MiniCPM export 中间数据 | 114 | 2,677,776,476 | WSL内部约2.494GiB |

WSL 合计 6,958,298,487 bytes（约6.480GiB），不是 F 盘即时释放数。删除前审计曾发现 121 个 MiniCPM export 数据；实际脚本仅删同级保留 NBG 的 114 项，不能用 121 个估算值作最终成果。

已核对仍存在：Laya/Nano safetensors、对应 exporter、Laya 原导出源脚本、量化脚本和两套 INT8 ONNX；Int4 全部 28 个 NBG，以及每个 manifest 指向的 W8 decoder.data/decoder.json/decoder.quantize/tensor-inputmeta.yml；MiniCPM 160 个 NBG 路径（路径数并非去重后的模型数量）、原始 safetensors 和共享 embedding；Intern 原始 Q8 GGUF、官方原权重和现役 W8 构建目录。重新编译已清理的候选需先重建中间数据，直接复测保留 NBG 不受影响。

没有删除 slim20 内容寻址 objects、离线 Docker 编译环境、板端部署快照、唯一完整 TF 备份或运行中 VHDX。TF 备份无损 NTFS LZX 压缩已结束但失败，未压缩成功任何文件。后续控制器全文件复核 SHA256 为 93c6a03aa6c7b527a148c2ed1e68d77bfa2de3820df302e6c2c936a66ebb4087，与原记录一致，逻辑大小仍63,864,569,856 bytes；审计读取 maintenance/tf-compression.json，没有重复全盘hash。不能将这次压缩计为释放空间。原系统包、内核/启动/NPU恢复链的缺口见 maintenance/BACKUP_DECISION.md。

## 中文 Llama CPU 缓存与速度审计

模型为 FuseAI/FuseChat-Llama-3.2-1B-Instruct 的 bartowski Q4_K_M 文件，实际GGUF SHA256记录为 `8bcd46580c2eb069fde82b4c16f8f2981901d653fea1340ec560b6097f08f97a`。这是文本模型，不是视觉模型。文件位于 `paligemma/llama-zh/`，CPU测试日志位于同级 `paligemma/llama-*-results.jsonl`。

独立从原始JSONL重新计算，排除warmup：

| 组别 | 请求数 | 缓存 tokens | TTFT P50 s | decode P50 tok/s | 解释 |
| --- | ---: | --- | ---: | ---: | --- |
| llama-board | 6 | 全部0 | 1.410956 | 12.441155 | 热模型，零前缀复用，3问题×2 |
| llama-fresh | 6 | 全部26 | 0.637808 | 12.465499 | 六条不同新问题，共享system及模板前缀 |
| llama-cached | 6 | 27,26,49,47,47,49 | 0.089553 | 12.445942 | 后四例仅剩1个prompt token，不能用于system-only收益 |

fresh 的动态 prompt_n 为24/23/20/20/20/21；TTFT范围0.614454–0.748193秒。`llama_fresh_runner.py` 的请求体已冻结system、不同user问题、ChatML之外的Llama专用模板、temperature0、seed42、n_predict32，server为CPU6/7、2线程、context1024、-ngl0。启动与首个warmup不进入六例P50。脚本记录首次非空流式content的到达时间，包含本机HTTP/可见内容边界；Intern记录引擎首token，两者计时边界不同，当前资料不是严格同prompt的CPU/NPU优劣对照。

fresh 六例有4条生成达到32-token上限，有两条自然结束为17/9 tokens。decode来自llama-server的predicted_per_second，并非把32强制反馈执行与自然有效文本混算。家庭陪伴质量仍未验收：回家问候出现“明天再见”，下雨提示夹英文essentials，补水建议出现重复句子，部分回答截断。主REPORT保留这些局限，未将“能输出中文”写成质量合格。

## Llama NPU 单层：执行成功，精度拒绝

已读取 `../llama32-compat-20261023/REPORT.md`、export_layer0.py、layer_probe.c、analyze_board.py、两个board-summary以及板端raw输出。审计独立从12个 `*-output.f32` 和保留的reference重算L2，并重新核对每个raw输出SHA256，全部与summary一致。

| 路径 | 六状态API耗时中位数 | hidden相对L2范围 | 驱动分配/释放 |
| --- | ---: | ---: | --- |
| 7FC W8 | 8.871995 ms | 6.048%–35.075% | 55,811,336 bytes → 0 |
| 4FC W8+MLP三FC及内部FP16 | 40.841800 ms | 5.478%–35.180% | 110,754,840 bytes → 0 |

MLP FP16候选慢约4.60倍，最坏误差未改善；拒绝作为解决方案。六个状态都超过旧5%数值筛查线，但这里的误差不是任务准确率；probe返回0仅代表执行/释放完成，不能当quality PASS。

源码边界核对：仅layer0、6个位置0/1/2/4/8/14，capacity32，每位置一遍run；每次加载独立冻结input和参考KV，未连续传播NPU输出KV。它们来自15-token英文序列及同一校准集合，不是独立留出集。真实layer0权重从Q4_K/Q6_K/F32 GGUF解量化，再转W8，不是原厂BF16基线。RoPE显式使用base逆频率除以GGUF rope_freqs，并与Transformers Llama3参数对照；不能把旧Intern公式直接复用。

这些证据支持“真实单层已导出、编译、板端运行且已发现精度问题”。不支持16层全模型、完整词表、全模型TTFT/tok/s、中文质量或完整部署完成，也不支持把8.87ms乘层数当实测速度。

## Molmo 内存下限与动态路由失败

官方卡说明约1.5B active、7.2B total。参见 https://huggingface.co/allenai/MolmoE-1B-0924 。不能按名称1B估算驻留内存。

已直接复核 `molmo/source/config.json`：n_layers16、moe_num_experts64、d_model2048、mlp_ratio1、mlp_hidden_size=null。`modeling_molmoe.py` 的 MolmoeMLP 将FFN设为 (mlp_ratio*d_model)/2=1024，gate/up/down三矩阵，所有专家分别实例化。故仅专家参数：16×64×3×2048×1024=6,442,450,944。INT8每参数1byte时恰为6GiB，比记录的5853MiB总RAM（6,137,315,328bytes）还多305,135,616bytes（291MiB）。这排除了注意力、视觉、embedding、KV、量化元数据、工作区、runtime和OS，已是保守下限。

结论限定为“全部专家以W8完整常驻于现有板RAM”的路线不能成立，不是所有MoE实现都不可能。真正packed INT4、分层/专家换入是不同实现，尚未验证；先前W4生成C常量是INT8容器，尚未证明NBG内部原生packed INT4布局，不能拿未证实的减半假设绕过这个W8下限，也不能反向断言所有INT4编译器都不可能压缩。

真实转换证据：

- TopK-router、softmax-control的import/inputmeta/quantize/export均exit0，产生343,984/276,072-byte NBG。
- 静态一路专家D2048/FF1024图所有阶段exit0，产生12,608,056-byte NBG，状态COMPILED_NOT_BOARD_VALIDATED。
- 官方torch.where/索引方式构造的动态一路专家在import阶段exit1。原始 `compiled/dynamic-one-expert/import.log` 明确报告不支持指定动态shape中的NonZero。

这些图使用固定seed随机权重，是转换器兼容性实验。预训练大权重28,888,362,419bytes没有下载，不能声称完整Molmo权重加载、板端路由运行、视觉推理或整模性能。失败仅证明目前直译动态分发路径受阻，不能据此否认CPU路由+NPU静态专家等替代实现。当前无Molmo板端TTFT/decode可报。

## 主报告与进行中的工作（早期审计记录）

本次复核的主 `REPORT.md` 已正确区分：Intern文本decoder优化、Llama CPU整模、Llama NPU单层精度拒绝、Molmo内存/转换限制，以及PaliGemma仍在进行。主报告的0.638s来自fresh六例，而非0.0896s整句缓存组；Llama NPU与Molmo未被写成完整部署成功。group prefill“一致”的证据范围仍仅上述开发样例，不能扩展为一般产品质量保证。

本段是板测前的审计状态；Pali板端结果及WSL compact后续已完成，最终验收见末尾。先前阶段没有把x86参考当ARM速度、没有将计划中的compact提前算成释放成果。

## slim20 官方镜像补查

针对“slim20 可能含官方系统镜像”的补充，已只读扫描 `/mnt/f/Files/slim20` 的 15,845 个文件，并检查根目录和内层 compact-manifest。两份清单均描述 29,075 个逻辑文件、15,788 个唯一对象，omitted 中明确含 `system image`。恢复说明也把系统刷机镜像列为排除内容，不能仅因它是恢复归档就推定含 OS。

已查官方 686,359,639-byte 7z SHA256 `bb78cd785ce226ab65cef7295a3c7abe084310e8a6c1b68d1cfa0e01af47e6ec`，以及 3,636,461,568-byte 原 img SHA256 `a569574cd0fdf4745f23f5e6c81cb652ce95f3f6d7bd506912ef5a1c58c90a79`：对象库无对应 SHA 路径，manifest 的 files/aliases 无对应映射，扫描范围内无这两个长度的文件，也无 `.img` / `.7z` / `.zip` 文件。归档 extraction-verification 记录原归档已做 CRC 校验并删除。保留的约 7.34GiB ubuntu-npu.tar 是 Docker 编译环境，不是板卡启动 OS。

因此本次补查没有发现可替代 TF 整盘快照的官方系统镜像，并没有推翻此前恢复链存在缺口的判断。代码、用户态 VIPLite 库和应用服务快照不能重建 bootloader、DTB、特定内核及匹配 vipcore 驱动。整盘快照不是永久必须保留；但在可验证的小型系统镜像、匹配启动/驱动、机器配置与恢复步骤准备完成前，不能把“已有代码”写成等价恢复能力。未挂载或修改 TF 文件系统，未重复全盘 hash。

## PaliGemma 完整 CPU 测量预审（执行前历史记录）

已检查 `paligemma/run_pali_board.py`、`pali_board_runner.py`、`summarize_pali_board.py`、`summarize_when_done.py`、instrumented `gemma.cpp-main/gemma/run.cc` 与 patch。控制器在传输结束、板端文件 SHA 与转换产物匹配、ARM/x86 编译完成、instrumented x86 token parity 通过后，才持 hostlock 与板端 flock 执行。此处预审不表示这些条件已在板卡实际完成。

计时边界：OPI_REQUEST_BEGIN 在读图、resize 和 GenerateImageTokens 之前；OPI_FIRST_TOKEN 是第一个非 EOS 输出 token 的生成回调，早于 stdout 实际显示和空白清理。故 e2e_ttft 包含图像预处理、视觉编码、文本包装/分词、prefill 与首 token，不能称用户端 HTTP/首字符延迟。image_s 单独为 IMAGE_BEGIN→IMAGE_END；after_vision_ttft 仍含包装/分词等，不是纯 decoder prefill。模型加载在请求前；`--map=1` 的首请求可能承担尚未触及的模型页 fault，未清 OS page cache，不能称冷启动基准。

每次请求重新生成同一图片的特征；PALIGEMMA 路径在请求后将 abs_pos 重置为0，未实现固定 system KV 或跨请求图片特征复用。四请求只有同一图片的 building-first、building-hot、color-hot、caption-hot；hot 意味模型/OS缓存变热，不代表复用前次完整 prompt。这个小样本仅支持图文 smoke test，不能提供稳健 P50/P95 或通用视觉准确率。

已报告并修复的阻断问题：原 runner 没有 TERM/INT handler，而推理子进程位于独立 session，外层 timeout 可能使 Python 直接退出而跳过 finally。当前 handler 抛出异常进入 finally，忽略清理期间的重复 TERM/INT，向子进程组发 TERM、等10秒，再 KILL+wait；Popen 与线程启动也位于 try 内。外层超时已提高为1150秒、kill宽限20秒、SSH1180秒，覆盖内部120秒ready+4×240秒请求+20秒退出及清理预算。执行者重启了等待中的控制器及summary以加载新代码，原传输继续。`cleanup-signal-test/result.json` 的真实 WSL 假子进程回归记录0.114秒清理、remaining=[]、passed=true；这验证信号路径，不是板端资源超限测试。

监控每0.5秒采样进程RSS/PSS/Swap、全机MemAvailable、温度、进程utime+stime、CPU频率及所有当时线程亲和。温度≥78°C或MemAvailable<256MiB时向推理进程组TERM，主循环随后退出并清理。它是采样保护；不能声称严格硬内存限制、未采样峰值或每次读取失败时仍能保证阈值。最终需检查guard、采样覆盖和线程是否始终约束于CPU6/7。RSS/PSS不等于全系统占用，before/after其他进程记录提供共存背景。

summary 已使用完整 token 事件重新解码，不依赖 stdout/stderr 两个异步线程的相对时序；EOS ID=1保留在输出计数中，EOS-only缺首token时返回空TTFT。仅1个内容token加EOS不报告decode速度；caption若足够长，分别报告排首token后含EOS与不含EOS的单例平均，不能称稳态吞吐分布。CPU按连续采样ticks在请求起止时插值，输出平均使用核数和占全8核百分比；它不是全机总CPU占用。模型加载若没有被连续采样完整包围就标未测，不外推。

x86 instrumented parity 的 building/color 与参考 token 相同，caption 也与BF16参考的11个token（含EOS）相同。这只是实现/转换 smoke parity；没有用x86延迟冒充ARM板卡速度，也不能把单张图的回答一致称为模型任务质量验收。

## PaliGemma decoder layer0 NPU 预审（执行前历史记录）

目录：`../paligemma2-npu-compat-20261023/`。独立读取官方 safetensors 实际 layer0 的11个BF16张量，对BF16原始16-bit字节重算SHA256，全部与 export-summary 的 weight_bf16_sha256 一致；不是从GGUF再次解量化的基线。

源码保留2304 hidden、8Q/4KV heads、head_dim256（attention拼接2048再投影2304）、四处1+weight的Gemma2 RMSNorm、GELU-tanh、1/16 query缩放、attention softcap50、RoPE10000及layer0滑窗类型。归一化使用稳定的等价缩放表达式，并用官方 Gemma2DecoderLayer FP32 对照；最终词表头softcap30不在单层图内。6例完整文本层对官方FP32最大相对L2为1.4853e-7，18冻结单步ORT对照最大相对L2为4.7001e-7，支持导出参考实现正确。

18个输入来自3条中文文本、causal/全prefix bidirectional两种mask、每例首/次/末位置。manifest 的18个 inputs SHA只有13个不同值：3个causal BOS相同、每句causal末位置与prefix末位置相同。它们全部同时用于校准和探针，不是18个独立留出样本；prefix文本不是图像特征。历史KV取官方参考，不连续回填NPU生成KV；4096滑窗只检查逻辑边界，没有实际长序列运行。

两种真实NBG编译成功：FP16为156,015,160bytes，SHA256 `8c40180b2c4705cf30130d7ce6718cd25db73bbeda06dd45e307db2a8a35fcb9`；混合方案70,649,288bytes，SHA256 `b10e8ad86553da2daacd48d07395648d80497984062bdf978bf018ff263d3217`。混合方案为7个FC的INT8权重、INT16 FC边界与其余FP16，不能简写成全W8A8。编译器模拟耗时不是板端API耗时。

layer_probe只验证18个冻结状态的hidden输出，KV仅有shape契约检查；返回0明确只代表执行/释放完成。当前没有以编译成功或FP32/ORT通过冒充NPU数值通过，没有单层耗时倍乘估计整模TTFT/decode。完整视觉链路、26层、词表头、连续KV及多图质量仍属于不同验收层次，待板端原始结果后补充实际精度、计时和driver释放。

## 最终原始结果审计

本节取代前文“待板端结果”的状态。只读取已回传到F盘的结果、WSL reference和输入，没有访问板卡，没有重新运行推理/编译。当前主REPORT已正确区分CPU整模与NPU单层，并保留新16层实验尚无实测结果的边界。

### Pali CPU 图文四请求

独立从 `pali-board-results.jsonl` 的 BEGIN/IMAGE/FIRST/TOKEN/END 事件重算计时与token；与summary一致，全部四请求complete exit0，telemetry guard为空。模型为SFP CPU压缩格式，不是NPU INT4。图文三个独特问题（其中building重复一次）全部token与保留的真实BF16参考一致；单张图三题仍只构成功能与转换回归。

| 请求 | 图像阶段 s | 完整图文TTFT s | 内容token数 | 排首token、排EOS decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| building_first | 10.850049 | 74.244343 | 1 | 不报告 |
| building_hot | 13.739674 | 52.139607 | 1 | 不报告 |
| color_hot | 10.737947 | 48.466938 | 1 | 不报告 |
| caption_hot | 10.690903 | 48.236397 | 10 | 1.467777 |

caption若保留EOS，首后10步为1.469670 tok/s；不含EOS则9个内容间隔。前三题1内容+EOS不能用于稳态decode。模型常驻mmap、每问重新执行视觉和prefix prefill、没有固定system KV；首请求不代表清空page cache的严格冷启动。TTFT是引擎首个非EOS token回调，包含视觉，不含外部网络/ASR/TTS。

387次遥测中，采样RSS最高4,103,556KiB（3.91346GiB），PSS最高4,101,149KiB（3.91116GiB），温度最高67.319°C，进程Swap最高46,316KiB。全部被采到的线程亲和均是CPU6或CPU7。名义间隔0.5秒，实际最大相邻间隔约1.184秒，因此这些是采样峰值与采样保护，不能称连续硬阈值保证。MemAvailable包含可回收文件页，不能加上RSS推导物理余量；旧服务仍存在但未并发压测。

**图文进程CPU无效。** 主线程comm为`P0 X00 C000`，旧代码直接对整个 `/proc/PID/stat` split，固定下标因此错读fault字段。原始图文telemetry没有raw stat，无法反推真实utime/stime；当前summary中的所有process CPU均为null、process_cpu_valid=false，validity文件保留原因。先前预审只检查了公式，没有发现字段解析问题，该部分结论在这里撤回。不得采用旧0.13核或1.67%的数字，也不能用后续文本CPU替补图文CPU。

独立重算未受影响的 `/proc/stat` 系统ticks，热三问全机busy分别27.3004%、28.5299%、28.4068%；定义为前8字段总增量扣除idle与iowait，guest不重复加入。首请求未完整被采样包围所以为空。这是全系统占八核总量，不能直接归因于Pali进程。

### Pali CPU 无图中文三请求及新CPU采样

检查patch06：无图PALIGEMMA分支仍使用BOS+文本+换行、全prefix双向attention；没有默认为普通因果chat。三份实际执行source manifest（CLI二进制、runner、cases）的SHA均与当前对应文件一致。它是独立新CLI，不能把旧图文版记录覆盖为新版本。

81条遥测均保存raw stat。审计逐条以最后右括号后的字段11/12重解析utime/stime，全部等于保存的cpu_ticks；再用连续ticks在请求边界插值重新计算CPU核数及系统busy，全部与summary一致。WSL真实带空格进程名回归的正确CPU增量20ticks、旧parser增量0，也支持修复有效。两个热请求按持续时间加权1.9883835核，即全8核24.8548%。加载初始化未被采样完整括住，CPU仍正确标未测。

| 请求 | 输入tokens | TTFT s | 首后内容decode tok/s | 平均进程CPU核数 | 全机busy |
| --- | ---: | ---: | ---: | ---: | ---: |
| home首请求 | 31 | 7.154482 | 1.507616 | 1.815270 | 25.5479% |
| sleep热请求 | 28 | 6.088309 | 1.532539 | 1.996189 | 28.1598% |
| rain热请求 | 30 | 6.473649 | 1.520301 | 1.979345 | 27.6936% |

home/sleep各16内容tokens，达到上限且无EOS；rain为11内容tokens+EOS。均无视觉、没有固定system KV，三题不是分布或P95。最高采样RSS/PSS为3.17670/3.17467GiB，最高温度56.916°C，进程Swap采样均0，guard为空，complete exit0；所有采到线程亲和均在CPU6/7。实际最大采样间隔约0.753秒。

独立对比板端事件tokens与WSL SFP：3/3完全一致；与真实BF16参考：home/rain一致，sleep从共同前4token后分叉，为2/3一致。不能把“SFP跨平台一致”写成“相对BF16无损”。输出仍存在复述用户指令、截断和raincoat英语混入；原BF16也存在行为问题，未通过中文陪伴质量验收。约6秒热TTFT与1.52tok/s只代表这三例CPU文本实现，不能代替NPU整模结果。

### Pali NPU：36个原始hidden输出独立复算

对FP16/W8各18个 `board-results-*/outputs/p*-output.f32` 按little-endian FP32读取，用独立FP64平方和计算相对L2；所有shape为2304、值有限、结果与board-summary相符。对应18份inputs与18份reference的SHA全部与fixture manifest一致。板测model-source记录的两份NBG SHA与之前编译清单一致。

| 实际配置 | 18次API中位 ms | hidden相对L2范围 | driver加载bytes | 释放后bytes |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 61.515195 | 0.167135%–0.575774% | 156,670,232 | 0 |
| 7FC权重INT8、边界INT16、其余FP16 | 11.861215 | 0.513982%–0.852273% | 71,303,464 | 0 |

18调用为3句×2种mask×3位置，只有13个唯一input SHA；与校准同源，没有独立留出数据。全部输入使用reference KV，并非连续NPU KV反馈；这里只数值检查hidden。最大误差小于1%是积极的单层兼容性证据，不是中文生成准确率或26层完整模型验收。没有将API单层时间倍乘换算整模TTFT/tok/s。

### Llama NPU：新增两版15状态配对

独立读取 `board-results-all-fp16-zh` 与 `board-results-attention-fp16-mlp-w8-zh` 共30份原始输出，并重算2048维hidden相对L2；与comparison及板端结果一致。30个输出SHA全部与comparison一致，15份inputs及15份reference SHA全部与heldout manifest一致；15份输入均为不同SHA。

| 配置 | 15次API中位 ms | 六英文校准状态L2 | 九中文留出状态L2 | driver加载/释放bytes |
| --- | ---: | ---: | ---: | ---: |
| 全FP16 | 48.298940 | 0.048685%–1.104277% | 0.098556%–0.252742% | 122,347,136 → 0 |
| attention FP16 + MLP W8 | 16.248440 | 0.632267%–4.091120% | 2.141176%–3.697810% | 67,411,824 → 0 |

这是新加入的15状态对照，不能和前文旧六状态allFP16中位48.134ms混写成同一组。中文九状态来自三个未参与英文校准的句子，每句早/中/末三个位置，不是九个独立问题；使用raw中文+BOS，未套聊天模板，所有历史KV由参考提供。5%是既有单层兼容性粗筛，不能用“全部低于5%”宣称中文生成质量合格。混合候选比全FP16约快2.97倍，仍留下最高约4.09%单层误差，值得后续整模同token/logits验证；不是完整部署已经完成。

所有这些Llama参考与NBG仍源自Q4_K_M GGUF中的Q4_K/Q6_K/F32张量解量化。`Q4源→FP32→W8`不是NPU原生INT4，不能还原原始BF16质量；全FP16控制也只是对Q4解量化权重采用FP16算术。与Intern公平比较必须分别披露源量化、NPU权重/激活格式、FP16例外比例、相同提示词/反馈与缓存条件。补Q8源及完整16层+词表头是正在进行的新任务，此处没有为它们给出整模速度或质量结论。

### WSL compact 与重启证据

已读取实际 `compact_wsl.ps1`：先 `wsl.exe --terminate OPiUbuntu`，成功后执行内置 `wsl.exe --manage OPiUbuntu --compact`，两步exit0。`maintenance/wsl-compact.json` 记录23.646秒完成。独立复核前后差值：VHDX 88,689,606,656→68,562,190,336bytes，减少20,127,416,320bytes；F空闲14,871,744,512→34,999,160,832bytes，增加同一字节数。

回收为20.13GB十进制（18.7451GiB），当时F空闲约35.00GB（32.5955GiB）。这是该次压缩前后快照，不承诺后续编译后仍有相同空闲；包含早前累积空闲块，不能把全部20.13GB归因于本次6.96GB WSL文件删除。

`wsl-restart-verification.json` 记录重启验证成功：checkpoint b637966e3c0fa361dd932ff4825eb16aad375b6e、Docker镜像inspect exit0、PyTorch2.7.0+cpu/Transformers4.57.6导入exit0，以及attention-FP16/MLP-W8 NBG SHA `227225b1ea4d8a44b508990b2602948d4930537dfdf5a0f8083b80a523826a8b` 未变。它验证这些关键依赖，没有声称对全部磁盘文件逐项hash或完成板卡恢复测试。压缩已结束并经父任务明确允许后，本审计才重新使用WSL读取文件。

`paligemma/final-board-staging-cleanup.json` 记录本轮SFP板端副本先核SHA再删、F原件保留、临时NBG/fixtures删除以及无本任务gemma残留。记录时板卡空闲6,207,696,896bytes；这是控制器已保存证据，本审计没有为此重新登录板卡。新的整模实验仍需独立遵循自己的资源、质量与释放验收。
