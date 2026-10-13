# NPU HTTP多轮接口验证

模型和KV在开发板上常驻，完整32/24层及词表头由NPU执行；分词、HTTP网关和测试客户端在工作电脑，通过SSH传递token和事件。该布置用于验证接口，尚非板端独立部署，也没有替换8080的CPU网页。

实验页为`resident_chat.html`，接口为`resident_http.py`。支持流式文字、停止、新对话和整轮历史裁剪；只有仍保留在当前页面的聊天记录，刷新不持久保存。独立留出、人工质量评阅、30分钟稳定性与真实音频并发另行验收。

| 模型 | 实板HTTP状态 | 完整层数 | 上下文容量 | 已完成请求 |
| --- | --- | ---: | ---: | ---: |
| 500M | passed | 32 | 1280 | 5 |
| 1B | passed | 24 | 1280 | 5 |

三个开发对话轮次使用模型自己的实际输出续聊，检查下一轮KV复用；再测试超窗删除完整旧轮、前缀变化重置，释放驱动至0，并重新加载后核对同一问题的完整输出token一致。这里没有用独立留出指导调参。

500M第一次HTTP试验发现模板在`Assistant:`后补一个空格，而实际生成正文已以空格开头，重套模板产生双空格并使跨轮KV前缀失配。修复只在序列化旧assistant时消费模板的一个分隔空格，显示原文保留；原失败记录不删。真实生成token前缀已离线核对，新attempt另行实板验收。

本地单测验证UTF-8跨token切分、半途关闭流后丢弃worker、跨域拒绝及忙碌请求返回409。浏览器通过独立Chrome实际点击；仅当结果含browser.status=passed时，才计入停止、新对话及页面多轮验收。

## SmolVLM-500M-Instruct-Q8_0

| 请求 | 复用/新增token | 生成token | 引擎首token s | HTTP首个正文 s | 移除旧消息 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0/40 | 16 | 1.000 | 3.508 | 0 |
| 2 | 55/15 | 19 | 0.705 | 0.710 | 0 |
| 3 | 88/17 | 11 | 0.806 | 0.813 | 0 |
| 4 | 0/32 | 3 | 0.997 | 1.006 | 2 |
| 5 | 0/32 | 3 | 0.638 | 3.268 | 0 |

HTTP首个正文时间包括必要的模型加载、SSH和UTF-8凑齐，不与五轮引擎首token直接相除。此组短开发对话没有控温配对，不作正式速度结论。

实际自由生成内容：

第1轮用户：I am making a cup of tea. Chat with me for a moment.

>  Hi! I'm ready to help. What would you like to make?

第2轮用户：The rain outside sounds relaxing.

>  It's a lovely day outside, isn't it? The rain is soothing and calming.

第3轮用户：What were we talking about just now?

>  I was talking about making a cup of tea.

浏览器验证：`passed`。two browser turns、stop aborts worker、new conversation after stop。

网页第1轮：首token 0.859 s · 18.94 token/s · 复用 0 / 新算 37 token · 本轮含模型加载

>  Flowers.

网页第2轮：首token 0.792 s · 19.35 token/s · 复用 39 / 新算 17 token

>  Flowers.

实际回答仅作为开发样例展示，不能据此称角色稳定或质量无损；尚未与同模型CPU逐问题配对评阅。


## InternVL3-1B-Instruct-Q8_0

| 请求 | 复用/新增token | 生成token | 引擎首token s | HTTP首个正文 s | 移除旧消息 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0/40 | 45 | 1.070 | 5.121 | 0 |
| 2 | 84/16 | 48 | 0.341 | 0.347 | 0 |
| 3 | 147/19 | 19 | 0.497 | 0.504 | 0 |
| 4 | 0/32 | 10 | 0.858 | 0.866 | 2 |
| 5 | 0/32 | 10 | 0.637 | 2.338 | 0 |

HTTP首个正文时间包括必要的模型加载、SSH和UTF-8凑齐，不与五轮引擎首token直接相除。此组短开发对话没有控温配对，不作正式速度结论。

实际自由生成内容：

第1轮用户：I am making a cup of tea. Chat with me for a moment.

> Sure, I'd be happy to help you with your cup of tea! What do you have in mind? Do you prefer a specific type of tea, or is there something else you'd like to know or discuss?

第2轮用户：The rain outside sounds relaxing.

> Rainy days can be quite relaxing, and they can also be a great opportunity to enjoy a cup of tea. Tea can be a soothing and calming beverage, especially when you're feeling stressed or tired. It's a great way to unwind

第3轮用户：What were we talking about just now?

> Just now, we were discussing the relaxing nature of rainy days and the benefits of tea.

浏览器验证：`passed`。two browser turns、stop aborts worker、new conversation after stop。

网页第1轮：首token 0.900 s · 14.66 token/s · 复用 0 / 新算 37 token · 本轮含模型加载

> Great! What kind of flowers are you going to arrange them on?

网页第2轮：首token 0.433 s · 14.47 token/s · 复用 51 / 新算 18 token

> I'm not sure which flowers you have on your desk. Could you please describe them?

实际回答仅作为开发样例展示，不能据此称角色稳定或质量无损；尚未与同模型CPU逐问题配对评阅。


复现：`run_resident_http_validation.py`，报告生成：`report_resident_http.py`。
[紧凑证据](evidence-attack-20261013/resident-http.json)。
