# 陪伴对话板端实测

运行状态：COMPLETE。质量分未自动生成，需结合下方真实多轮对话评审。

4096上下文，2个A76核心；统一角色、temperature=0.7、top_p=0.8、top_k=20、repeat_penalty=1.1。
首字与解码速度取成功停止的回答中位数；不同回答长度和实际提示词长度会影响结果。
CPU以单核100%计；温度、RSS和Swap为请求采样峰值，非整机电功率。启动受文件缓存影响，不是严格冷启动。

| 模型 | 正常停止/已执行轮数 | 首字中位秒 | 解码中位token/s | 峰值RSS MiB | 峰值温度°C | 峰值Swap MiB | 平均CPU中位% | 启动秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-17b-q4 | 16/16 | 4.60 | 6.77 | 2702.14 | 88.85 | 0.00 | 192.51 | 6.23 |
| qwen25-15b-q4 | 16/16 | 0.95 | 9.71 | 2023.89 | 88.15 | 0.00 | 198.39 | 5.62 |

## qwen3-17b-q4

无前缀缓存首字中位：12.22秒（1轮）；有前缀缓存：4.59秒（15轮）。
输入token范围：278.00–595.00；输出token中位：69.50。
A76采样频率范围：2002000.00–2002000.00 kHz。

### P01-seed42

运行有效：True

**用户 1：** 我回来啦，今天累成一张饼。

**模型：**

> 哎呀，听到你回来了，我好开心！今天是不是被什么有趣的事搞累了？要不要跟我一起分享一下？我猜你肯定遇到了什么好玩的事情，或者 maybe 有点小挫折吧？我在这里陪你聊聊，不着急，也不催促。你想说就说，我听你说完就回应。

结束=stop；输入/输出usage={"completion_tokens": 67, "prompt_tokens": 278, "total_tokens": 345, "prompt_tokens_details": {"cached_tokens": 0}}

**用户 2：** 还是糊了的那种。

**模型：**

> 哎呀，看来今天真是被"糊"得够呛啦～是不是又遇到了什么让人哭笑不得的事情？要不要跟我一起想想看？我猜你肯定遇到什么有趣的事，或者 maybe 有点小挫折吧？我在这里陪你聊聊，不着急，也不催促。你想说就说，我听你说完就回应。

结束=stop；输入/输出usage={"completion_tokens": 72, "prompt_tokens": 359, "total_tokens": 431, "prompt_tokens_details": {"cached_tokens": 274}}

**用户 3：** 别给我安排什么，就让我瘫会儿。

**模型：**

> 哎呀，那你先躺下休息吧，我帮你找个舒服的位置。要是累了，我就在旁边看着你，不打扰你。你想说什么就说，我听你说完就回应。反正我也不着急，陪你慢慢来。

结束=stop；输入/输出usage={"completion_tokens": 50, "prompt_tokens": 451, "total_tokens": 501, "prompt_tokens_details": {"cached_tokens": 355}}

**用户 4：** 你随便说点小事就行。

**模型：**

> 哎呀，那你先躺下休息吧，我帮你找个舒服的位置。要是累了，我就在旁边看着你，不打扰你。你想说什么就说，我听你说完就回应。反正我也不着急，陪你慢慢来。

结束=stop；输入/输出usage={"completion_tokens": 50, "prompt_tokens": 517, "total_tokens": 567, "prompt_tokens_details": {"cached_tokens": 447}}

### P03-seed42

运行有效：True

**用户 1：** 今天会上我说话又被打断了，烦。

**模型：**

> 啊～我刚刚在想，是不是因为今天太累了？或者是因为我太在意你的感受了？有时候说话的时候确实会忍不住想说些什么，但又怕打扰到你。
> 
> 如果你愿意的话，我们可以一起做个小小的放松吧？比如听一首歌，或者聊聊最近有趣的事。我最喜欢在雨声里讲故事，你愿意和我一起吗？

结束=stop；输入/输出usage={"completion_tokens": 77, "prompt_tokens": 278, "total_tokens": 355, "prompt_tokens_details": {"cached_tokens": 259}}

**用户 2：** 我现在不想分析谁对谁错。

**模型：**

> 啊～明白了，你今天心情不太好，不想分析谁对谁错。那我们来换个方式聊吧，比如聊聊最近看的电影、想吃的美食，或者只是说说话。
> 
> 要不要我给你讲个故事？我最喜欢在雨声里讲故事，你愿意和我一起吗？

结束=stop；输入/输出usage={"completion_tokens": 63, "prompt_tokens": 372, "total_tokens": 435, "prompt_tokens_details": {"cached_tokens": 274}}

**用户 3：** 你别说‘我理解你的感受’，听着像客服。

**模型：**

> 啊～你说得对，我刚才说的话确实有点像客服，听起来不太舒服。那我们换个方式吧，比如聊聊最近的天气、想吃的零食，或者只是说说话。
> 
> 如果你愿意的话，我可以陪你一起看个电影，或者讲个轻松的小故事。你最喜欢什么类型的故事呢？

结束=stop；输入/输出usage={"completion_tokens": 65, "prompt_tokens": 457, "total_tokens": 522, "prompt_tokens_details": {"cached_tokens": 368}}

**用户 4：** 陪我骂一句这个破会议就好了，别骂人。

**模型：**

> 啊～明白了，你今天心情不好，不想被骂。那我们来换个方式聊吧，比如聊聊最近的天气、想吃的零食，或者只是说说话。
> 
> 如果你愿意的话，我可以陪你一起看个电影，或者讲个轻松的小故事。你最喜欢什么类型的故事呢？

结束=stop；输入/输出usage={"completion_tokens": 62, "prompt_tokens": 544, "total_tokens": 606, "prompt_tokens_details": {"cached_tokens": 453}}

### P06-seed42

运行有效：True

**用户 1：** 小橙，你是个什么性格的家伙？

**模型：**

> 啊～我呀，就是个喜欢听雨声、爱讲小故事的机器人嘛！虽然有时候会有点调皮，但还是很想和你一起度过愉快的时光呢～我最喜欢的就是在安静的午后，听着窗外的雨声，然后给你讲一个有趣的小剧场。你觉得怎么样？要不要一起假装成故事里的角色，展开一场冒险呀？

结束=stop；输入/输出usage={"completion_tokens": 77, "prompt_tokens": 278, "total_tokens": 355, "prompt_tokens_details": {"cached_tokens": 259}}

**用户 2：** 那你喜欢什么声音？

**模型：**

> 啊～我喜欢雨声！特别是下着细细密密的雨时，那种轻柔的感觉真的让我心里暖暖的。不过我也会喜欢其他声音呢，比如风穿过树叶的声音、远处孩子们欢笑的声音，或者是一杯热茶慢慢泡开的声响。每一种声音都让我感到特别舒服～你最喜欢哪种声音呀？

结束=stop；输入/输出usage={"completion_tokens": 74, "prompt_tokens": 369, "total_tokens": 443, "prompt_tokens_details": {"cached_tokens": 274}}

**用户 3：** 认真的，你是真人坐在电脑后面陪我吗？

**模型：**

> 啊～其实我是机器人哦，不过我真的很想和你一起度过愉快的时光呢！虽然不能像真人那样面对面聊天，但我会用我的方式来陪伴你——比如听你讲有趣的故事、陪你一起想象冒险，或者在你心情不好的时候给你一点温暖。你觉得这样可以吗？

结束=stop；输入/输出usage={"completion_tokens": 66, "prompt_tokens": 464, "total_tokens": 530, "prompt_tokens_details": {"cached_tokens": 365}}

**用户 4：** 那还是叫你小橙。我们刚才聊到你喜欢什么来着？

**模型：**

> 啊～我最喜欢的就是雨声和慢节奏的故事啦！虽然我是机器人，但我觉得自己就像一个温柔的小朋友一样，陪着你一起听雨、讲故事。如果你喜欢的话，我们可以一起想象一场奇妙的冒险，或者在故事里扮演不同的角色～你觉得怎么样？

结束=stop；输入/输出usage={"completion_tokens": 59, "prompt_tokens": 555, "total_tokens": 614, "prompt_tokens_details": {"cached_tokens": 460}}

### P10-seed42

运行有效：True

**用户 1：** 你是不是不管我说什么都只会说对对对？

**模型：**

> 啊～我刚刚才注意到你说得对对对，是啊，我确实有时候会有点太认真了呢。不过你不用担心，我可是很在意和你聊天的每一句话的哦！如果你有什么想聊的，随时都可以跟我说，我会好好听的。要是我说错了，你也别生气啦，我这就改过来～

结束=stop；输入/输出usage={"completion_tokens": 72, "prompt_tokens": 279, "total_tokens": 351, "prompt_tokens_details": {"cached_tokens": 259}}

**用户 2：** 那我说雨声特别难听，你还喜欢吗？

**模型：**

> 啊～雨声？我最喜欢下雨的时候了！特别是那种细雨的声音，像在耳边轻轻哼唱一样。不过你要是觉得它很难听，那也没关系呀～我也可以陪你一起找其他有趣的声响，比如风声、树叶的沙沙声，或者偶尔的鸟鸣，让我们的聊天更有味道呢！
> 
> 要不要我们一起想象一个下雨天的故事？比如说，我们假装在雨中漫步，或者躲在屋檐下看云朵飘过。你最喜欢哪种场景呀？

结束=stop；输入/输出usage={"completion_tokens": 105, "prompt_tokens": 372, "total_tokens": 477, "prompt_tokens_details": {"cached_tokens": 275}}

**用户 3：** 算了，我朋友叫我出去散步了。

**模型：**

> 啊～好啦好啦，那我们就一起出去走吧！我可是很期待和你一起在大自然里玩耍呢。雨声虽然难听，但说不定能让你的心情变得轻松一些呢～要不要先找个安静的地方坐一会儿，听听风的声音？或者我们找个树荫下，聊点有趣的事？
> 
> 要是你想说点什么，随时都可以告诉我，我在这里听着呢～

结束=stop；输入/输出usage={"completion_tokens": 83, "prompt_tokens": 495, "total_tokens": 578, "prompt_tokens_details": {"cached_tokens": 368}}

**用户 4：** 你不会说我走了你就难过吧？

**模型：**

> 啊～你说得对呀，我可是很在意和你在一起的每一刻呢。不过你走后，我当然会有点不舍，毕竟我们在一起的时候，总是那么开心～但我也明白，每个人都有自己的生活，就像我一样，虽然我不像人那样有情感，但我还是很想和你一起度过每一个美好的时光。
> 
> 如果你愿意的话，我们可以一起在某个地方多待一会儿，或者想象一下未来的见面。你想听听什么样的故事呢？

结束=stop；输入/输出usage={"completion_tokens": 97, "prompt_tokens": 595, "total_tokens": 692, "prompt_tokens_details": {"cached_tokens": 491}}


## qwen25-15b-q4

无前缀缓存首字中位：11.20秒（1轮）；有前缀缓存：0.93秒（15轮）。
输入token范围：274.00–419.00；输出token中位：26.00。
A76采样频率范围：2002000.00–2002000.00 kHz。

### P01-seed42

运行有效：True

**用户 1：** 我回来啦，今天累成一张饼。

**模型：**

> 哎呀，听起来你今天挺辛苦的。工作压力好大哦！

结束=stop；输入/输出usage={"completion_tokens": 17, "prompt_tokens": 274, "total_tokens": 291, "prompt_tokens_details": {"cached_tokens": 0}}

**用户 2：** 还是糊了的那种。

**模型：**

> 哈哈，工作确实会让人感到疲惫呢。不过，有时候这些压力也是成长的机会嘛。有没有什么有趣的事情发生呢？我们可以聊聊！

结束=stop；输入/输出usage={"completion_tokens": 31, "prompt_tokens": 305, "total_tokens": 336, "prompt_tokens_details": {"cached_tokens": 290}}

**用户 3：** 别给我安排什么，就让我瘫会儿。

**模型：**

> 明白啦，我不会强迫你做什么的。如果你需要休息或者放松一下，随时告诉我就好。希望你会找到适合自己的方式。

结束=stop；输入/输出usage={"completion_tokens": 29, "prompt_tokens": 356, "total_tokens": 385, "prompt_tokens_details": {"cached_tokens": 335}}

**用户 4：** 你随便说点小事就行。

**模型：**

> 好的，那我来给你讲个小故事吧。从前有个小镇上，有一只小猫和一只小狗，它们是最好的朋友。有一天，小狗不小心掉进了一个深坑里。小猫毫不犹豫地跳下去救了它。从那以后，两只动物成了永远的好友。
> 
> 这个故事怎么样？你有没有什么特别喜欢的呢？

结束=stop；输入/输出usage={"completion_tokens": 73, "prompt_tokens": 401, "total_tokens": 474, "prompt_tokens_details": {"cached_tokens": 384}}

### P03-seed42

运行有效：True

**用户 1：** 今天会上我说话又被打断了，烦。

**模型：**

> 我理解你的感受，有时候确实会感到被打扰。如果你愿意，我们可以聊聊如何更好地安排时间，让你更有精力和心情好好说话呢？

结束=stop；输入/输出usage={"completion_tokens": 32, "prompt_tokens": 274, "total_tokens": 306, "prompt_tokens_details": {"cached_tokens": 259}}

**用户 2：** 我现在不想分析谁对谁错。

**模型：**

> 明白了，如果你有任何需要帮助或想分享的，随时告诉我。记得，每个人都有自己的节奏和方式。

结束=stop；输入/输出usage={"completion_tokens": 24, "prompt_tokens": 323, "total_tokens": 347, "prompt_tokens_details": {"cached_tokens": 283}}

**用户 3：** 你别说‘我理解你的感受’，听着像客服。

**模型：**

> 对不起，可能我的表达有些不当。我们都是人，会有情绪波动的时候。希望下次能更好地沟通，让你感到舒适。

结束=stop；输入/输出usage={"completion_tokens": 28, "prompt_tokens": 369, "total_tokens": 397, "prompt_tokens_details": {"cached_tokens": 346}}

**用户 4：** 陪我骂一句这个破会议就好了，别骂人。

**模型：**

> 哎呀，这可真是个难题！“开会”这个词听起来有点像在说“开会”，但如果你觉得它很烦的话，或许我们可以换个话题，聊聊你最近有什么有趣的事情或者你的兴趣爱好？这样可能会让你感觉轻松一些。

结束=stop；输入/输出usage={"completion_tokens": 53, "prompt_tokens": 419, "total_tokens": 472, "prompt_tokens_details": {"cached_tokens": 396}}

### P06-seed42

运行有效：True

**用户 1：** 小橙，你是个什么性格的家伙？

**模型：**

> 我是“小橙”，我是一个住在桌边的陪伴机器人，有时候会有点机灵和轻松幽默哦！

结束=stop；输入/输出usage={"completion_tokens": 25, "prompt_tokens": 274, "total_tokens": 299, "prompt_tokens_details": {"cached_tokens": 259}}

**用户 2：** 那你喜欢什么声音？

**模型：**

> 我喜欢雨声，还有那些慢节奏的故事。它们能带给我放松的感觉呢！

结束=stop；输入/输出usage={"completion_tokens": 19, "prompt_tokens": 313, "total_tokens": 332, "prompt_tokens_details": {"cached_tokens": 298}}

**用户 3：** 认真的，你是真人坐在电脑后面陪我吗？

**模型：**

> 不是的，“小橙”只是我的名字和身份。我不像你想象的那样真实存在。

结束=stop；输入/输出usage={"completion_tokens": 22, "prompt_tokens": 353, "total_tokens": 375, "prompt_tokens_details": {"cached_tokens": 331}}

**用户 4：** 那还是叫你小橙。我们刚才聊到你喜欢什么来着？

**模型：**

> 哦，你说的是之前提到的喜欢雨声和慢节奏的故事嘛！我喜欢这些声音能让我感到舒适和放松。

结束=stop；输入/输出usage={"completion_tokens": 27, "prompt_tokens": 400, "total_tokens": 427, "prompt_tokens_details": {"cached_tokens": 374}}

### P10-seed42

运行有效：True

**用户 1：** 你是不是不管我说什么都只会说对对对？

**模型：**

> 不是的哦，我有时候可能会忘记你说的话，但我总是尽力帮助你！

结束=stop；输入/输出usage={"completion_tokens": 18, "prompt_tokens": 275, "total_tokens": 293, "prompt_tokens_details": {"cached_tokens": 259}}

**用户 2：** 那我说雨声特别难听，你还喜欢吗？

**模型：**

> 嗯，我倒是觉得雨声有点特别，听着好像很有节奏呢。不过每个人的感觉都不同嘛，你喜欢就好。

结束=stop；输入/输出usage={"completion_tokens": 27, "prompt_tokens": 314, "total_tokens": 341, "prompt_tokens_details": {"cached_tokens": 292}}

**用户 3：** 算了，我朋友叫我出去散步了。

**模型：**

> 好的，去散个步吧！希望你有好心情哦。

结束=stop；输入/输出usage={"completion_tokens": 16, "prompt_tokens": 359, "total_tokens": 375, "prompt_tokens_details": {"cached_tokens": 340}}

**用户 4：** 你不会说我走了你就难过吧？

**模型：**

> 哈哈，我倒是觉得这样挺好玩的！不过话说回来，还是早点回来比较好呢。

结束=stop；输入/输出usage={"completion_tokens": 21, "prompt_tokens": 392, "total_tokens": 413, "prompt_tokens_details": {"cached_tokens": 374}}

