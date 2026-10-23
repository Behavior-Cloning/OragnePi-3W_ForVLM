# PaliGemma2-3B 与中文 Llama3.2-1B 部署实测（2026-10-23）

状态：Llama CPU、Pali 完整图文 CPU 四请求，以及 Pali FP16/W8 首层 NPU 18×2冻结输入均已实测完成；最新中文纯text三请求也已完成；Pali仍仅有首层NPU结果，完整中文生成实测为CPU。所有输出仅用于测试，没有执行机器人动作。

## 最新中文纯text结果（无图、CPU、3请求）

按最新要求停止扩展视觉实验。Pali无图推理在HF仍需要双向文本prefix；gemma.cpp新增patch06，同时使用BOS+文本+换行模板。原始BF16/SFP的实际输入token逐个校齐，未执行vision；新版CLI/source hash与图文版本分开保存。31/28/30总输入token，没有复用固定system KV，每问重新prefill。模型进程常驻，首次请求与热请求分列。

| 请求 | 输入token | TTFT | 排除首token后的decode | 进程CPU核数 | 整机CPU busy |
|---|---:|---:|---:|---:|---:|
| home（首请求） | 31 | 7.154s | 1.508 tok/s | 1.815 | 25.55% |
| sleep（热） | 28 | 6.088s | 1.533 tok/s | 1.996 | 28.16% |
| rain（热） | 30 | 6.474s | 1.520 tok/s | 1.979 | 27.69% |

这些是逐例值，不是P50/P95。home/sleep达到16-token上限（15个首后生成步、无EOS）；rain为11内容token+EOS，首后11步含EOS约1.520tok/s，去掉EOS则10步约1.520tok/s。热两请求按耗时加权进程CPU为1.988核，即全8核24.85%；整机busy27.69%–28.16%，热iowait0。RSS峰值3,331,008KiB、PSS3,328,882KiB（约3.18GiB），最高56.916°C，进程swap0、无采样保护触发、所有线程在CPU6/7。模型加载CPU因0.5秒采样未完整括住mmap初始化，仍标未测；已正确修复stat含空格进程名的解析，不能把这些有效text CPU数据与旧图文无效process CPU字段混用。

板上三条输出逐token全部匹配WSL SFP。与原始BF16对照，home/rain完整相同，sleep在共同前4token之后分叉，这是保留的量化漂移案例。两种精度本身都存在复述指令：SFP sleep为“家庭陪伴机器人：我爱你，我准备睡觉了，请用一句话和…”（被16-token上限截断）；rain为“家庭陪伴机器人：没有 raincoat，要带伞。”。仅三例不能给中文准确率，但足以说明这不是已调好的中文chat。CPU热TTFT约6.1–6.5秒、decode约1.52tok/s，明显不适合作本项目当前中文交互主力；后续如研究Pali应以NPU整模和中文行为验证为前提。当前优先中文Llama1B NPU路线。

证据：pali-text-board-results.jsonl、pali-text-board-summary.json、pali-text-board-telemetry.json、pali-text-board-source-manifest.json、text-input-tokenizer-parity.json、pali-text-wsl-parity.json。原始无BOS参考错误尝试保留为.initial-no-bos，不作为有效参考。

## 中文 Llama3.2-1B：已经在板卡 CPU 跑通

使用 FuseAI/FuseChat-Llama-3.2-1B-Instruct 的 bartowski Q4_K_M GGUF，实际经ModelScope镜像API下载（不是从HF网站直接下载），精确revision/请求URL/SHA见 llama-zh/download-manifest.json。16 层、hidden 2048、32 Q / 8 KV heads、head_dim 64。权重 807,690,816 bytes，SHA256 `8bcd46580c2eb069fde82b4c16f8f2981901d653fea1340ec560b6097f08f97a`，下载与板上副本均核对。原始 config、作者模型卡、量化发布者卡、基础 Llama3.2 LICENSE/USE_POLICY 保留在 llama-zh/。

用户给出的 shenzhi-wang/Llama3.2-1B-Chinese-Chat 尚未获得可验证精确仓库（ModelScope 404、HF 连接/查询受限），不能推断它一定不存在，也没有用同作者 8B 冒充。本次选用可核实的中文增强 FuseChat 1B。其[作者模型卡](https://huggingface.co/FuseAI/FuseChat-Llama-3.2-1B-Instruct)说明训练包含 9,952 条中文样例，来自 alpaca_gpt4_zh / Magpie Qwen2 Pro Chinese，使用 Qwen2.5-72B 回答；作者报告 AlignBench 2.9→4.3。这些是作者指标，不是本次板上中文准确率。量化来源为 [bartowski](https://huggingface.co/bartowski/FuseChat-Llama-3.2-1B-Instruct-GGUF)，许可沿用 [Meta Llama3.2](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) 约束，不能当作 MIT/Apache。

运行环境：板卡 A733，CPU6/7 两个 A76、2 threads；llama-server，n_gpu_layers=0，ctx=1024，模型常驻。已有 8081 CPU 服务没有被停止。以下均为 CPU 数据，不是 NPU。

| 模式 | 有效请求 | 输入/缓存 | TTFT | decode | 资源 |
|---|---:|---|---:|---:|---|
| 无 prompt cache | 6（3题各2次，另1次预热） | 48–50 总 token，cache_n=0 | P50 1.411 s | P50 12.44 tok/s | RSS 峰值约1.59 GiB，PSS约1.58 GiB，最高57.54°C |
| 短固定前缀缓存、每次新问题 | 6 个不同家庭陪伴问题，另1次预热 | 每次 cache_n=26，动态20–24 token | P50 0.638 s；范围0.614–0.748 s | P50 12.47 tok/s | RSS 1,675,836 KiB（约1.60 GiB），PSS 1,670,887 KiB；最高58.09°C |

初次进程 ready 约5.04 s，和常驻请求延迟分开。另一组反复完全相同问题得到约0.08 s的首 token，实为整题 KV 复用（动态只剩1 token）；该组不能代表一般新问题，所以不采用它作最终延迟结论。有效新问题组见 llama-fresh-results.jsonl / llama-fresh-summary.json；无缓存组见 llama-board-results.jsonl。

这只说明中文增强1B在 CPU 上有可用吞吐。6题肉眼检查仍有重复、答非所问、中英混杂和32-token截断，例如喝水提醒重复“回家后”，安静要求变成解释家长照顾婴儿；不能直接推荐成品陪伴质量。尚未做同题 BF16/量化质量基线或正式中文基准。它与 Intern NPU 的输入长度不同，不能以0.638对0.8直接得出更快的模型结论。

Llama 单层 NPU 的实际冻结输入 probe 已按另一实验组要求执行并回传，普通 W8 与 MLP FP16 例外均成功运行和释放。执行成功不等于数值合格，更不等于整模部署；该组数值误差及速度详见 [Llama单层NPU报告](../../llama32-compat-20261023/REPORT.md)。板上临时 GGUF / probe 大文件已按清单移除，F 原始权重和全部结果仍保留。

## PaliGemma2-3B：真实图文链路与转换回归

精确模型：[google/paligemma2-3b-mix-224](https://huggingface.co/google/paligemma2-3b-mix-224)，适合直接多任务推理的 mix 版本。文件从 ModelScope 镜像获取，镜像账号名本身不能证明官方上传者；访问 HF gated 文件得到401、Kaggle压缩权重403，没有绕过其访问限制。保存了 ModelScope 完整文件 manifest 和 SHA；原始 BF16 两片共6,064,583,376 bytes。实际 PyTorch 计数3,032,242,416参数。

模型包含 SigLIP 27层视觉编码器（hidden1152、patch14、224×224输入→256视觉token）和 Gemma2 26层文本解码器（hidden2304、8Q/4KV、head256）。它的图像和文本前缀用双向 prefix-LM attention；还有 Gemma2 softcap、局部/全局交替 attention、GELU、四个norm。不能直接替换 Intern 的 Qwen2 导出配置然后宣称 NPU 已适配。

在 WSL 完成以下实际路径：原始 BF16 safetensors → 修复后的 gemma.cpp SFP 压缩 → x86/ARM CPU runtime。SFP 是 gemma.cpp CPU 压缩格式，不是已证明的 NPU INT4。文本 linear 用 SFP，embedding/vision 保持 BF16，bias F32。最终文件4,045,406,208 bytes，SHA256 `c3bea2caf002da77624d807b79f26519455fc8326eca99442b21b3e270c44b73`。

模型转换确实遇到并修复两项错误，另修复 Python binding：

1. 已失效的 PALIGEMMA_448 enum binding 导致 Python 扩展编译失败，按当前 enum 删除过期项。
2. converter 重建 TensorInfo 丢失 shape flatten 标志，使 img_emb_kernel 写成225792×3而运行时需1152×588；改为从完整 TensorInfoRegistry 取元数据。实际图像 patch 的 PyTorch conv2d 对 OHWI patch matmul 校验最大差5.36e-6、相对L2 4.53e-7，排除了只改shape但算术错误的问题。
3. prefix-LM prefill 已将末token位置回退，随后 stream bookkeeping 又多推进一位，导致首输出位置断言失败；修正为发送末prompt token不推进，保留位置断言。最终回归通过。

固定上游 commit `3ed403e28707c0e3eb5f48b0e487b63a446d8e2e`；原tar、原失败日志、独立patch、完整构建说明见 runtime-source.json、patches/、REBUILD.md。patch-replay-verification.json 确认五份patch能从原tar重放到实际构建源文件。ARM使用静态NEON，避免 GCC11 的 SVE向量编译错误。

原始 BF16 与 SFP 用同一张上游教堂图片（SHA256 `01a854bbbccf7d293f3e6bc5dac34d9964a5e9cb5bde48f5be1a91565c6bfcd7`）做实际图文回归：

| 问题 | BF16 输出token | 压缩SFP输出token | 结果 |
|---|---|---|---|
| What type of building is it? | 44201,1：church | 44201,1：church | 完全一致 |
| What color is the church? | 7384,1：white | 7384,1：white | 完全一致 |

另补 caption 问题“Describe this image in one sentence.”，原始 BF16 与 SFP 的11个token（含EOS）也逐个一致，文本为“A large church with two towers on either side.”，见 caption-sfp-wsl-parity.json。这是同一图片三问 smoke test，不能称广泛视觉精度合格。原始 BF16 在 WSL CPU 的进程峰值约6.35GiB，超过板上总内存，因此没有盲推 BF16。旧示例文字写gray，并非本次人工标签；本次只对照实跑BF16的white。

WSL 单进程常驻 SFP 两问、每问重新编码图片，仍得到相同全部token，详见 instrumented-wsl-parity.json。此时桌面CPU热第二问图像阶段1.519 s、端到端图文TTFT7.946 s；这些不是板卡速度，不纳入与Intern板卡NPU的横向排名。

## 板上 Pali 实验状态与测量口径

四个图文请求已完成，全部exit0、无采样保护触发、每个生成token均与WSL/BF16参考一致。所有线程亲和均在CPU6/7。模型文件使用 rsync partial/append-verify，完成后单独SHA核对；板上最低保留1GiB存储余量。runtime使用 CPU6/7、2threads、seq_len512、max_generated32、mmap、to_bf16=0。模型在同一进程常驻；每个请求重新执行图片读取、resize与视觉编码，再做prefix-LM prefill和decode。没有将旧图像特征缓存隐藏在TTFT之外，也没有声称已实现Pali固定system KV。

事件采用 steady_clock，记录 model_ready/load、request_begin、image_begin/end、first_token、request_end，并保留每个生成token ID。首token按完整图文request开始计算，包括vision和prefill。首请求包含首次触页影响，热请求另列。只测几问，将逐例报告，不编造P50/P95分布。降噪、ASR、TTS和其他完整agent链路未并发接入本次新模型测试。

资源采样间隔0.5秒，含RSS/PSS、swap、MemAvailable、各温区、各核频率、CPU ticks和系统loadavg；每次采样若温度≥78°C或MemAvailable<256MiB则触发终止，这是0.5秒采样保护，不是严格无超调的硬限。命令有总timeout。原有服务保持原状，前后记录其RSS/PSS与监听端口。如触发内存保护，只能说明当前保留服务条件不足，不能推断3B独占板卡也必然不能运行。mmap意味着进程保持加载，但页面可以被系统回收，不等于全权重锁定物理内存。没有并发请求旧8081服务，因此这也不是两模型同时活跃的共存压测。温度是传感器读数，未测芯片结温/长期热稳定性。

## Pali 板卡 CPU 实测结果（图文请求，已结束）

进程mmap ready约0.316 s，只是文件映射/初始化，不代表4GB权重全部已触页；首次请求另含首次触页影响。每个热请求重新运行vision，没有偷用旧图像特征。下面是逐例数据，不是P50/P95。

| 请求 | 图像阶段 | 完整图文TTFT | vision之后到首token | 全部完成 | 整机CPU busy |
|---|---:|---:|---:|---:|---:|
| building_first | 10.850s | 74.244s | 63.394s | 83.250s | 未完整覆盖 |
| building_hot | 13.740s | 52.140s | 38.400s | 52.814s | 27.30% |
| color_hot | 10.738s | 48.467s | 37.729s | 49.168s | 28.53% |
| caption_hot | 10.691s | 48.236s | 37.545s | 55.076s | 28.41% |

caption输出为“A large church with two towers on either side.”，10个内容token+EOS。排除首token后，含EOS的10个生成步平均1.470 tok/s；只算余下9个内容token则1.468 tok/s。前面三个短答案只有1个内容token+EOS，不能把唯一EOS间隔称作稳态decode性能。

RSS峰值4,103,556KiB（约3.91GiB）、PSS峰值4,101,149KiB（约3.91GiB）；MemAvailable最低4,409,204KiB。这里MemAvailable包含可回收mmap文件页，所以不能把它加上RSS来推算物理内存，也不能由该值声称还有4.2GiB无代价余量。进程swap峰值46,316KiB（约45.2MiB）。最高温度67.319°C。现有llama-server进程PSS由449,287KiB降至221,452KiB，表明其他服务页面驻留会变动，未验证两模型同时活跃。

整机CPU busy使用/proc/stat前8个字段，扣除idle和iowait，guest不重复计；热请求约27.30%–28.53%，这是全系统占8核总量的比例，不是Pali进程CPU。对应iowait分别0.716%、约0%、0.002%。原始模型进程CPU采集有缺陷：gemma把主线程命名为“P0 X00 C000”，旧stat.split固定下标因空格偏移，误读fault字段。旧cpu_ticks不可恢复，已将所有Pali进程CPU结果置null，并保留原始telemetry及validity说明；绝不采用错误的1.67%数值。未来parser已改为在最后右括号后解析并保存raw stat，真实Linux带空格进程名回归通过（正确CPU增加20ticks、旧方法错误为0）。不为此重跑长图文矩阵。

Llama fresh末段固定9.734秒连续推理采样窗口平均1.990个CPU核，即全8核24.87%；完整可观测25.519秒（含加载/间隙）平均1.719核。它的线程名llama-server没有上述空格偏移。由于旧Llama请求未保存绝对起止，这不是精确六请求加权均值；冷加载独立CPU未测。原始推理时长/内存/温度结果不受影响。

## Pali 真权重首层 NPU 结果（与CPU分栏理解）

同一真实BF16 layer0、18个冻结状态，FP32/ORT参考先行，板上两种图均成功执行并释放driver到0：

| 量化配置 | 样本 | 单层run中位耗时 | 最差hidden相对L2 | NPU driver加载量 |
|---|---:|---:|---:|---:|
| 全FP16 | 18 | 61.515ms | 0.5758% | 156,670,232B |
| W8 | 18 | 11.861ms | 0.8523% | 71,303,464B |

这是积极的NPU算子/数值兼容证据：W8单层比FP16更快，冻结hidden误差小于1%。但只验证第0层hidden；KV数值和递归、26层累计误差、全词表head、完整中文生成均未验证。不能把单层ms转换成整模tokens/s，也不能声称Pali整模已经在NPU生成文本。原始输出与分析在 research/paligemma2-npu-compat-20261023/board-results-fp16 和 board-results-w8。

## 存储清理与边界

已清本次板上临时 Llama GGUF 807,690,816 bytes及单层probe复制品/fixtures，F权威原件与结果保留。另根据父任务授权，确认无Jev进程/映射后，删除可再生Laya FP32 ONNX 1,288,263,205 bytes，SHA256 `cd811914b9c1f72b4b6c08138b3d9052305cc4d8b8faaabdd7d7fe6fd0c17129`；F仍有643,835,514-byte Laya原safetensors和导出脚本。Nano FP32、INT8部署、语音模型、现役VLM NB均未动。板上剩余6,215,266,304 bytes后开始Pali传输。完成CPU图文/中文纯text后，按授权清理本次板上4,045,406,208-byte SFP副本，F校验原件保留；最终SHA/释放空间见 final-board-staging-cleanup.json（最后probe控制器生成）。清单在 board-*-cleanup.json。

## 尚未完成

- Pali真正NPU整模转换与精度验证；目前没有NPU decode指标。
- Llama中文系统性质量验证、64/128动态token与Intern相同输入配对、全agent资源竞争。
- Pali更多家庭图片、中文问题、精度量化误差与长时间压力测试。
