# 板端 VLM 图片对话网页

## 当前状态（2026-10-09）

网络已修复，开发板连接 professional-EXT5G，地址为 **192.168.88.29**。
网页已安装到板端，入口为 <http://192.168.88.29:8080>；模型权重已下载并校验，
推理引擎已原生编译并正常运行。当前默认已切到 **InternVL3-1B-Instruct**，网页显示 InternVL3-1B；
三个模型均保留在 TF 卡上。两项服务均 active/enabled。
1B 切换后已通过中文连续对话、真实图片/追问、刷新恢复历史和超长输入拒绝验收；
首次看图约 221 秒、同图动物颜色追问约 2.3 秒，原始记录见多模型报告。
原始 256M 部署完成了网页集成测试、真实图片连续问答、性能评测和系统重启自启动验证，
历史数据见 [SmolVLM-256M 实测报告](SmolVLM-256M实测报告.md)；切换 1B 后未再次重启整板。
随后已增加经过对比的 system prompt，见 [提示词优化记录](SmolVLM提示词优化.md)；
SmolVLM 英文部分行为有改善，中文闲聊仍不可靠。
新增 500M、InternVL3-1B-Instruct 的来源与实测见 [多模型板端对比](多模型板端对比.md)，
模型可使用报告中的切换脚本更换，网页顶部显示实际服务模型。
电脑地址为 192.168.88.21，DHCP 地址未来可能变化。

## 功能与边界

详细保存/删除语义和窗口上限见 [对话保存与上下文](对话保存与上下文.md)。
新增输入框旁模型选择、1024/2048/4096/8192 窗口滑块、“应用并加载”和实时资源面板。
配置是全板共享；仅模型服务重启，网页保持在线显示等待。已移出上下文的历史仍保留在浏览器。

- 图片上传或粘贴，浏览器先压缩为最长边 768 像素的 JPEG。
- 流式回答、停止生成、多轮追问；每轮重新发送会话上下文，包括前面上传的图片。
- 多会话列表、删除会话，IndexedDB 保存本浏览器的消息和图片。
- 顶部“删除对话”与列表逐条删除，确认框显示标题；生成期间禁止删除。
- 后端返回长度截断标记时，回答下方显示提示，并随历史保存。
- 电脑和手机布局，无第三方 CDN、无云端模型 API。
- 上下文可调，单次最多生成 256 token。取消原约15轮限制，超窗时按完整问答轮次移出最早历史，界面提示。
  图片也消耗上下文；最新问题/图片本身超窗时仍报错，不静默截断。
- 256M 指参数规模。模型偏向英文和基础视觉描述，不能期待大模型级中文、OCR 或推理能力。
- 当前各候选模型均走 CPU，尚未转换到 A733 NPU。
- 服务无登录认证，仅面向可信局域网；不要直接映射到公网。使用者可调用推理服务，
  历史在各自浏览器保存，不做跨设备同步；清理浏览器数据会删除历史。

## 结构

```text
浏览器 :8080 → Python 网页/API 网关 → 127.0.0.1:8081 llama-server
                                            ↓
                              Q8 语言权重 + F16 视觉编码器
```

- `src/vlm_chat/server.py`：Python 标准库 HTTP 网关、请求校验、流式转发。
- `src/vlm_chat/system_prompt.txt`：服务端提示词，修改后重启网页服务；不会删除已有对话。
- `src/vlm_chat/web/`：原生 HTML/CSS/JavaScript 界面。
- `configs/systemd/`：板端服务配置，专用非登录用户运行，开机启动。
- `scripts/download_smolvlm.py`：固定版本、分段续传、完整 SHA-256 校验。
- `scripts/install_vlm_on_board.sh`：板端原生编译与安装，需要 root。
- `scripts/select_vlm_model.py`：校验权重、切换推理模型与网页标识，失败回滚。
- `src/vlm_chat/control.py`：仅回环访问的固定操作控制器；负责权重校验、配置白名单、重启、回滚和资源采样。
  网页进程仍为低权限用户，通过本地端口8083调用；控制器没有通用shell命令接口。
- `configs/vlm-models.json`：各候选的固定版本、文件大小和 SHA-256；模型只有完整下载后才可选择。
- `tests/vlm_chat_browser.cjs`：真实网关 + 模拟模型后端 + Chromium 集成测试。

## 版本与来源

模型仓库：<https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF>

固定 revision：`b9e4379657e1450d04d02eec8e345667265b0a00`。
下载镜像默认使用 hf-mirror.com；下载完成后按上游 LFS 元数据中的 SHA-256 验证。
原始模型：<https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct>，Apache-2.0。

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| SmolVLM-256M-Instruct-Q8_0.gguf | 175054528 | `2a31195d3769c0b0fd0a4906201666108834848db768af11de1d2cef7cd35e65` |
| mmproj-SmolVLM-256M-Instruct-f16.gguf | 190031616 | `0802360aca1748f112ea510b8ff277c65b1361c8ef30ed89b83c9c7a60d08e96` |

推理引擎：<https://github.com/ggml-org/llama.cpp>，固定 commit
`d81235049384534c167caea52b85a694f6103d14`（查询到的 v0.6.0）。
多模态说明：<https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/mtmd/README.md>。
固定源码归档实测 SHA-256：`6b58785f0a82898f4c3442417ff962e2d4b231b1bee5d033902ad90b27901e14`。
本配置已在板端完成编译和真实推理验证。

## 下载与安装

在项目根目录下载：

```bash
python3 scripts/download_smolvlm.py
```

只将完整校验后的权重部署到板端。传输项目的 `src/vlm_chat`、`scripts`、
`configs/systemd`、`models/smolvlm-256m`；下载分块不用传输。
如果已下载 llama.cpp 归档，将其放在
`data/llama-d81235049384534c167caea52b85a694f6103d14.tar.gz`，避免板端重复下载。
在板上执行：

```bash
sudo bash scripts/install_vlm_on_board.sh
curl http://127.0.0.1:8080/api/health
systemctl status opi-vlm-model opi-vlm-web
journalctl -u opi-vlm-model -n 80 --no-pager
```

安装到 `/opt/opi-vlm`。模型服务只绑定回环地址，网页绑定端口 8080。
推理线程初始设为 2，并绑定 A76 核心 6、7；编译并行度为 2。
额外 prompt 缓存上限 128 MiB，关闭引擎 context shift；网关收到窗口错误后按消息边界裁剪重试。
模型窗口通过控制器重启实际改变，不只是修改前端数字；3B模型服务内存上限5GiB，其他模型4GiB。
确认健康接口返回 `ready: true` 后，用同一局域网设备打开 `http://板卡IP:8080`。

## 本地验证

网页单独预览（未运行模型时显示“模型未就绪”）：

```bash
python3 src/vlm_chat/server.py --host 127.0.0.1 --port 18080
```

浏览器测试需要 Node.js、Playwright 和 Chrome；测试端口默认 19080/19081。
依赖可单独装到工具环境，无需纳入仓库。

```bash
node tests/vlm_chat_browser.cjs
```

2026-10-09 已通过：图片上传、分块 UTF-8 流、追问保留原图、刷新恢复历史、停止生成、
后端错误展示、删除持久化、非当前会话删除/取消、长度截断提示持久化、390px 手机布局、跨站请求与远程图片 URL 拒绝。
这是应用集成验证；模型输出由测试后端提供，不代表模型真实回答或性能。

原始 256M 板端验收已完成：真实照片描述、同图追问、刷新恢复历史，及系统重启后自动恢复网页与模型服务。
板端程序 `/opt/opi-vlm/bin/llama-server` SHA-256 为
`153e7bf6327152b85e4e89983edff7dd40a19970f239abe2c37c77bc4e776266`。

## 板端评测工具

`scripts/benchmark_vlm.py` 使用 Python 标准库，无额外模型测试依赖。
需在板端以 root 运行，用于读取模型进程 I/O、服务控制和模型文件缓存验证。
评测期间不要同时发送其他对话，脚本会重启模型服务。

```bash
python3 /opt/opi-vlm/benchmark_vlm.py \
  --image /opt/opi-vlm/opi-vlm-dog-ui.jpg \
  --output /opt/opi-vlm/benchmark-results.json
```

默认纯文本、单图、带图追问各 3 次；保留 token 数、服务端 timings、TTFT、总延迟、
每 100ms 资源采样、CPU 时间、RSS、cgroup 内存、温度、页错误与进程 I/O。
服务热缓存重启 3 次，定向驱逐两份权重缓存后的启动 3 次，并用 `fincore` 验证缓存状态。
这不等于整板断电冷启动；未接电流计，不报告虚构瓦数、焦耳或续航。
图片效果检查另存英文目标识别、颜色、空间关系和中文回答，不将少量样例当作标准准确率。
