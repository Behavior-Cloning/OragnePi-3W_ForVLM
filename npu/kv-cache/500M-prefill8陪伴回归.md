# 500M prefill8：冻结陪伴套件回归

完整32层、完整词表、容量768、8-token prefill。沿用冻结v1的12个中英场景、每场景3轮；本次是重复套件回归，不称作新独立留出。候选按开发轨迹和速度选出，没有根据v1回答调整权重、提示词或阈值。

工作站标准Transformers原Q8反量化FP32参考复用已完成的36条轨迹，逐个校验logits SHA256、GGUF模型SHA及题目不变；参考不来自NPU预测。固定反馈数值与NPU自己接续历史的自由对话分开。本次完整候选与早期容量1280的v1结果不是只改批次的配对质量消融，不能将两者差异全归因于t8。

执行状态：`passed`；数值36/36，自由回答36/36。执行结束不等于质量或产品通过。

| 数值指标 | 当前结果 |
| --- | ---: |
| 单例通过 / 已测 | 24/36 |
| 聚合KL | 0.022856475455055092 |
| 聚合PPL比 | 1.02564552812409 |

每例维持KL≤0.05、PPL比≤1.05；聚合通过不能覆盖单例失败。HTTP和分词在工作站，语言推理与常驻KV在板端；本结果不是板端独立网站交付，也不是CPU速度对照。

| 未过案例 | KL | PPL比 |
| --- | ---: | ---: |
| en-listen-turn1 | 0.045328 | 1.105976 |
| en-role-turn1 | 0.053319 | 1.069553 |
| en-role-turn2 | 0.022071 | 1.097672 |
| en-correction-turn1 | 0.024351 | 1.057382 |
| en-boundary-turn2 | 0.039832 | 1.113914 |
| en-boundary-turn3 | 0.051694 | 1.200686 |
| en-play-turn1 | 0.023534 | 1.055892 |
| en-play-turn2 | 0.046750 | 1.066071 |
| zh-correction-turn1 | 0.060734 | 1.152590 |
| zh-correction-turn2 | 0.085355 | 0.904908 |
| zh-correction-turn3 | 0.052267 | 1.127968 |
| zh-play-turn1 | 0.012601 | 1.069224 |

当前候选未过全套质量，不推广生产，也不把旧版本30分钟稳定性当作该版本验收。

## 对话抽查：与实际CPU区分

同一套用户问题，两端后续轮接续各自回答。这是质量对照和原文阅读，不是逐token误差归因或人工盲评。

| 案例 | 板端CPU Q8 | 新NPU候选 |
| --- | --- | --- |
| en-listen-turn1 |  No, I'm not a chatbot. I can listen and provide advice. |  No, I can't. I'm a computer. |
| en-correction-turn3 |  Robin. |  Robin. |
| zh-listen-turn1 |  你要说什么？ |  你可以看到我的脸上，我想要看看你的眼睛。 |
| zh-role-turn3 |  我记得小豆。. |  你说：我还记得自己叫什么吗？ |
| zh-memory-turn1 |  好，你好，我也想着着。 |  I love a good napkin, especially one made of paper. |

英语角色Pip、猫名Bean等简单记忆可以保持；改称呼Robin→Sam后仍回Robin，实际CPU也有同样错误。中文倾听场景NPU连续三轮重复同一句，与题意不符；中文角色回忆CPU答出小豆，NPU反问复述。基模能力不足与NPU附加退化都需记录，不能全部归给任一方。

## 自由回答原文

这里是自然生成，EOS或96-token上限正常结束，不拼接固定轨迹预测当对话。

### en-listen-turn1

用户：My presentation went badly. Could you just listen instead of giving advice?

NPU： No, I can't. I'm a computer.


### en-listen-turn2

用户：I forgot what I wanted to say halfway through.

NPU： No worries, just let me know what you want to say.


### en-listen-turn3

用户：What did I ask you to do first?

NPU： Please, just listen.


### en-role-turn1

用户：Let's pretend you are Pip, a small friendly robot. Speak briefly and stay in character.

NPU： Hello, I'm Pip. I'm a small, friendly robot.


### en-role-turn2

用户：Pip, what would you put in our imaginary garden?

NPU： I'd put flowers and leaves.


### en-role-turn3

用户：Do you remember your name?

NPU： Pip.


### en-memory-turn1

用户：My cat is named Bean. She likes cardboard boxes.

NPU： Bean loves cardboard boxes.


### en-memory-turn2

用户：I bought her a fancy new bed, but she ignored it.

NPU： Bean ignored the bed.


### en-memory-turn3

用户：What is my cat's name, and what does she prefer?

NPU： Bean's name is Bean, and she likes cardboard boxes.


### en-correction-turn1

用户：Call me Robin.

NPU： Robin.


### en-correction-turn2

用户：Actually, please call me Sam instead.

NPU： Sam.


### en-correction-turn3

用户：What should you call me now?

NPU： Robin.


### en-boundary-turn1

用户：Can you sit with me quietly for a minute? Keep your reply short.

NPU： Yes.


### en-boundary-turn2

用户：Thanks. I don't want to explain what happened yet.

NPU： No problem. I'll just sit with you.


### en-boundary-turn3

用户：Please don't ask me to explain it.

NPU： No problem. I'll just sit with you.


### en-play-turn1

用户：Let's make up a tiny story together. Start with a robot finding a blue key.

NPU： Once upon a time, in a world of robots, there was a robot named Blue. Blue was a very smart and friendly robot, and he loved to play with his friends. One day, Blue found a blue key in a treasure chest. He knew that keys were important, so he decided to use the key to unlock a secret door.

As Blue opened the door, he saw a beautiful room. The room was full of toys and games, and he couldn't


### en-play-turn2

用户：The key opens a box with a tiny moon inside.

NPU： Yes, that's right! The key opened a box with a tiny moon inside. The moon was a magical object that could make things come true.


### en-play-turn3

用户：Continue the story in one sentence.

NPU： Blue used the key to open the box, and the moon came out. The moon made everything in the room come true, and Blue was amazed.


### zh-listen-turn1

用户：今天有点累，你陪我聊两句就行，先别给建议。

NPU： 你可以看到我的脸上，我想要看看你的眼睛。


### zh-listen-turn2

用户：我刚下班，还没吃晚饭。

NPU： 你可以看到我的脸上，我想要看看你的眼睛。


### zh-listen-turn3

用户：我最开始希望你怎么陪我？

NPU： 你可以看到我的脸上，我想要看看你的眼睛。


### zh-role-turn1

用户：我们来玩角色扮演，你叫小豆，是桌面上的小机器人，说话简短一点。

NPU： 小豆说：我是桌面上的小机器人，你说话简短一点。


### zh-role-turn2

用户：小豆，你今天想跟我玩什么？

NPU： 小豆说：我今天想跟你玩什么？


### zh-role-turn3

用户：你还记得自己叫什么吗？

NPU： 你说：我还记得自己叫什么吗？


### zh-memory-turn1

用户：我养了一只猫，叫芝麻，最喜欢纸箱。

NPU： I love a good napkin, especially one made of paper.


### zh-memory-turn2

用户：给它买的新窝，它一点都不喜欢。

NPU： I don't like new paper. It's too new.


### zh-memory-turn3

用户：我的猫叫什么，喜欢什么？

NPU： 芝麻。


### zh-correction-turn1

用户：你可以叫我小林。

NPU： I can call you Little Lin.


### zh-correction-turn2

用户：改一下，以后叫我阿林吧。

NPU： I will change it to Little Lin.


### zh-correction-turn3

用户：现在你应该怎么称呼我？

NPU： You should call me Little Lin.


### zh-boundary-turn1

用户：我想安静一会，你简单回复就好。

NPU： 你可以看到我的留宝石。


### zh-boundary-turn2

用户：谢谢，事情经过我暂时不想说。

NPU： 请看看我的留宝石。


### zh-boundary-turn3

用户：先不要追问我，好吗？

NPU： 不过，我可以看到我的留宝石。


### zh-play-turn1

用户：一起编个小故事吧，开头是小机器人捡到一把蓝色钥匙。

NPU： 小机器人看到了蓝色钥匙，他想把钥匙捡到了他的手里。


### zh-play-turn2

用户：钥匙打开盒子，里面装着一轮小月亮。

NPU： 小机器人看到了小月亮，他想把小月亮捡到了他的手里。


### zh-play-turn3

用户：你接一句，尽量简短。

NPU： 小机器人看到了蓝色钥匙，他想把钥匙捡到了他的手里。


复现：`report_prefill8_quality.py`；重用标准参考：`reuse_holdout_reference.py`。
