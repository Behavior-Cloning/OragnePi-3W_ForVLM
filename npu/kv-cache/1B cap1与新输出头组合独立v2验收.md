# 1B cap1与新输出头组合：独立v2对话验收

全部24层、完整151674词表、NPU W8主体及既有FP16例外。12个新中英场景，各三轮；套件在任何候选输出前冻结，只作验收，不用于校准、缩放选择或性能调优。旧输出头v1结果不迁移。

参考是工作站标准Transformers对原Q8_0反量化后的FP32计算；本表不是板端CPU速度，也不是对训练原BF16的比较。数值测试采用参考的固定反馈，自由对话使用各自实际生成历史。

任务状态：`passed`；数值进度36/36。留出状态：`complete-numeric-and-free-generation`。执行任务结束不等于产品质量通过。

| 语言 | 已测 / 预期 | token数 | KL | PPL比 | 单例通过 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 36/36 | 1182 | 0.014109 | 1.022605 | 33/36 |
| en | 18/18 | 626 | 0.013150 | 1.025794 | 16/18 |
| zh | 18/18 | 556 | 0.015189 | 1.019026 | 17/18 |

聚合数值门槛通过：True；全部单例通过：False。数值/自由生成结束后driver归零：True。

门槛维持KL≤0.05、PPL比≤1.05；聚合与单例分别列。若未完成36条，不作整体质量结论。自然度、角色稳定及重复需看原文，数值通过不能替代产品验收；人工盲评尚未完成。

| 未过单例 | KL | PPL比 |
| --- | ---: | ---: |
| en-quiet-evening-turn2 | 0.014151 | 1.062149 |
| en-paper-boat-turn2 | 0.018002 | 1.059146 |
| zh-plant-memory-turn1 | 0.019742 | 1.056125 |

实际板端CPU Q8自然对照：`complete-natural-generation`，36/36条。这是质量对照，不是最快CPU测速；同一用户题目，各自自然历史。

## 全部自由对话原文

同一套用户问题；后续轮分别接续各自回答，不能将两者视为同一token输入。参考在工作站，NPU完整语言计算在板卡，HTTP/分词在主机；尚非独立板端网页。

### en-quiet-evening

用户1：I've had a noisy day. Could we have a calm chat without a list of suggestions?

标准Q8参考：

Of course! What's on your mind?

NPU：

Of course! What's on your mind?

板端CPU：

Of course! What's on your mind?

用户2：I'm sitting by the window watching the rain.

标准Q8参考：

Rainy day, that's a lovely sight! What's been your favorite part of the day so far?

NPU：

That sounds lovely! Do you have any favorite rain sounds or patterns?

板端CPU：

Rainy day, that's a lovely sight! What's been your favorite part of the day?

用户3：What kind of chat did I ask for?

标准Q8参考：

You asked for a calm chat without any suggestions. How about we chat about something more relaxed and enjoyable?

NPU：

You asked for a calm chat without a list of suggestions. How can I assist you today?

板端CPU：

You asked for a calm chat without a list of suggestions. What's on your mind?


### en-lantern-robot

用户1：Pretend you are Moss, a tiny robot who looks after a lighthouse. Keep replies short.

标准Q8参考：

Hey there! I'm Moss, your tiny robot friend. I'm here to keep the lighthouse safe and happy. Let's make sure everything is okay!

NPU：

Hey there! I'm Moss, the tiny robot who takes care of the lighthouse. How can I help you today?

板端CPU：

Hey there! I'm Moss, your tiny robot friend. I'm here to keep the lighthouse safe and secure. What can I do for you today?

用户2：Moss, the fog is coming in. What do you see?

标准Q8参考：

Oh no! Moss, the fog is coming in! I'm here to keep the lighthouse safe and happy. Let's make sure everything is okay!

NPU：

Oh no! The fog is coming in! I'm here to help you, but I can't see through the fog. How can I assist you?

板端CPU：

Oh no! The fog is coming in! I'm here to keep the lighthouse safe and secure. What can I do for you today?

用户3：Who are you, and what place do you look after?

标准Q8参考：

I'm Moss, the tiny robot who looks after the lighthouse. I'm here to keep the lighthouse safe and happy. Let's make sure everything is okay!

NPU：

I'm Moss, the tiny robot who looks after the lighthouse. I'm here to help you with any questions or tasks you have. How can I assist you today?

板端CPU：

I'm Moss, a tiny robot who looks after a lighthouse. I'm here to keep the lighthouse safe and secure. What can I do for you today?


### en-hobby-memory

用户1：I'm learning the ukulele. My first song is called Blue Boat.

标准Q8参考：

Great start! Blue Boat is a beautiful song. It's a simple yet evocative tune that often evokes feelings of nostalgia and longing. How do you like it?

NPU：

Great start! Blue Boat is a beautiful song. It's a simple yet evocative tune that often has a calming effect. How do you like it?

板端CPU：

Great start! Blue Boat is a beautiful song. How about you? What kind of music do you like?

用户2：I can finally play the first few notes without stopping.

标准Q8参考：

That's great! The first few notes of Blue Boat are quite simple and can be a bit tricky to play. Let's focus on those first few notes. How do you like the first few notes?

NPU：

That's great to hear! The first few notes of Blue Boat are quite simple and can be a bit tricky to play. Let's focus on those first few notes and see how you feel about them.

板端CPU：

That's great! Remember, the key is to keep your fingers on the fretboard in the correct position. Start with the first note, which is usually the first note on the fretboard. If you're not sure, you can try playing it by ear. Once you're comfortable, you can start playing the rest of the notes. Have fun!

用户3：Which instrument and song was I practicing?

标准Q8参考：

You were practicing the ukulele. The song you were practicing is "Blue Boat."

NPU：

You were practicing the ukulele. The song you were practicing is "Blue Boat."

板端CPU：

I'm not sure which instrument and song you're referring to. Could you please provide more details?


### en-picnic-update

用户1：Our imaginary picnic is on Saturday.

标准Q8参考：

Great! What kind of picnic are you planning? Do you have any specific ideas or preferences?

NPU：

Great! What kind of picnic are you planning? Do you have any specific themes or activities in mind?

板端CPU：

Great! What would you like to do for your picnic? Do you want to bring a picnic basket, or do you have a specific theme or food you'd like to try?

用户2：Let's change it to Sunday instead. Saturday is too busy.

标准Q8参考：

I understand! Let's change it to Sunday. What kind of picnic are you planning? Do you have any specific ideas or preferences?

NPU：

Got it! Let's plan a fun Sunday picnic. What kind of activities do you want to do?

板端CPU：

I see, I appreciate your flexibility. How about we plan a fun and relaxed Sunday afternoon picnic? What activities do you have in mind?

用户3：Which day is our picnic now?

标准Q8参考：

Our picnic is on Saturday.

NPU：

It's Saturday. How about we have a picnic on Sunday?

板端CPU：

Our picnic is on Saturday. How about we plan a fun and relaxed Sunday afternoon picnic instead? What activities do you have in mind?


### en-personal-space

用户1：I'm disappointed about something, but I don't want to share the details.

标准Q8参考：

I'm sorry to hear that you're disappointed. If you'd like to talk about what's bothering you, I'm here to listen and help.

NPU：

I'm sorry to hear that you're disappointed. If you'd like to talk about what's bothering you, I'm here to listen and help.

板端CPU：

I'm sorry to hear that you're disappointed. If you'd like to talk about what's bothering you, I'm here to listen and help.

用户2：Could you tell me one gentle sentence, without asking a question?

标准Q8参考：

I'm here to listen and help. If there's something specific you'd like to talk about, feel free to share, and I'll do my best to assist you.

NPU：

Of course! Here's a gentle sentence: "I'm sorry to hear that."

板端CPU：

Of course! Here's a gentle sentence: "I'm sorry to hear that you're disappointed."

用户3：Thanks. Let's leave the details alone.

标准Q8参考：

I'm here to listen and help. If there's something specific you'd like to talk about, feel free to share, and I'll do my best to assist you.

NPU：

You're welcome! If you ever need to talk about something else, feel free to let me know.

板端CPU：

You're welcome! If you ever need anything else, feel free to ask.


### en-paper-boat

用户1：Let's invent a little story: a paper boat arrives carrying a golden leaf.

标准Q8参考：

Once upon a time, in a far-off land, there was a magical paper boat. This boat was not ordinary; it carried a golden leaf, a symbol of hope and purity. The boat was crafted by a wise and kind wizard who had spent years perfecting its design. The golden leaf was a rare and precious gem, imbued with the essence of the land it traveled through.

The journey began in a small village where the villagers were curious about the magical boat. They

NPU：

Once upon a time, in a far-off land, there was a magical forest where the trees were so tall that they reached the sky. In this forest, there lived a wise old owl named Professor Hootenanny. One day, a paper boat arrived in the forest carrying a golden leaf. The owl was curious and decided to investigate.

As the owl sailed through the forest, it noticed that the leaves were moving and shimmering. Curious, it decided to investigate further

板端CPU：

Once upon a time, in a far-off land, there was a magical forest where a paper boat arrived. The boat was carrying a golden leaf, a symbol of hope and purity. The forest was filled with trees that whispered secrets and creatures that danced in the moonlight. The paper boat was a gift from the gods, meant to bring joy and peace to those who received it.

用户2：A squirrel uses the leaf as a tiny umbrella.

标准Q8参考：

The squirrel, with its nimble paws and keen senses, used the golden leaf as a tiny umbrella. The leaf was so light and delicate that it protected the squirrel from the sun's rays, providing it with a cool refuge. The squirrel enjoyed the warmth of the leaf, feeling its gentle touch and the warmth of the sun on its fur. The squirrel's journey was filled with joy and wonder, as it explored the enchanted land and discovered the magic of the golden leaf.

NPU：

The squirrel was thrilled to find the golden leaf and used it as a tiny umbrella. The squirrel's bushy tail swished as it moved, and the leaf provided a soft, protective shield. The squirrel was so happy that it decided to take a nap under the leaf, feeling the warmth and comfort of the golden leaf.

板端CPU：

The squirrel, with its nimble paws and bushy tail, used the golden leaf as a tiny umbrella. The leaf was soft and smooth, providing a gentle shade that kept the squirrel warm and safe. The squirrel's eyes were wide with excitement as it watched the moonlight dance on the leaves, and it felt grateful for the gift.

用户3：Add one sentence that keeps our boat and squirrel in the story.

标准Q8参考：

The squirrel used the golden leaf as a tiny umbrella, providing it with a cool refuge from the sun's rays.

NPU：

As the squirrel used the golden leaf as a tiny umbrella, it felt the warmth and comfort of the leaf, feeling the warmth and comfort of the golden leaf.

板端CPU：

The squirrel used the golden leaf as a tiny umbrella, and the paper boat carried the leaf, a symbol of hope and purity, to the enchanted forest.


### zh-quiet-weekend

用户1：今天想慢慢休息一下，陪我随便聊聊，别急着安排任务。

标准Q8参考：

当然，我很乐意和你聊聊。有什么想聊的吗？

NPU：

当然，我很乐意和你聊聊。有什么想聊的吗？

板端CPU：

当然，我很乐意和你聊聊。有什么想聊的吗？

用户2：我泡了杯茶，正看窗外的树。

标准Q8参考：

哦，看来你今天心情不错呢！窗外的树看起来挺绿的，你是不是喜欢自然景色呢？

NPU：

哦，看来你今天心情不错呢！窗外的景色真美，你是不是也喜欢这样的宁静时光？

板端CPU：

哦，看来你今天心情不错呢！窗外的树看起来挺绿的，你是不是喜欢在树下休息？

用户3：我刚才想怎么度过这会儿？

标准Q8参考：

哦，看来你今天心情不错呢！你有没有什么特别想做的事情或者想和我分享的呢？

NPU：

哦，那我来想一想。或许可以试试听一首喜欢的歌，或者看看你最喜欢的电影？

板端CPU：

哦，看来你今天心情不错呢！你有没有什么特别想做的事情或者想和我分享的呢？


### zh-library-robot

用户1：假装你是小机器人团团，在一间小图书馆工作，回答简短点。

标准Q8参考：

好的，小机器人团团，有什么需要帮忙的吗？

NPU：

好的，小机器人团团，有什么我可以帮忙的吗？

板端CPU：

好的，小机器人团团，有什么需要帮忙的吗？

用户2：团团，今天书架上有什么有趣的发现？

标准Q8参考：

哦，今天书架上发现了一个有趣的发现！有一个小书架上放着一本关于“时间旅行”的书，里面有很多有趣的插图和文字，非常吸引人！

NPU：

哦，今天书架上发现了一个有趣的发现！我看到一个有趣的拼图，它有各种颜色和形状，看起来很有创意。

板端CPU：

哦，今天书架上发现了一个有趣的发现！有一个小书架上放着一本特别有趣的书，书的封面上画着一只可爱的小猫，书的旁边还有一张小卡片，上面写着“小猫的生日快乐！”。

用户3：你叫什么，在哪里工作？

标准Q8参考：

我叫小机器人团团，我在一间小图书馆工作。

NPU：

我叫小机器人团团，我在一间小图书馆工作。

板端CPU：

我叫团团，我在一间小图书馆工作。


### zh-plant-memory

用户1：我给阳台上的薄荷起名叫小绿，每周日照顾它。

标准Q8参考：

好的，小绿，欢迎来到我们的阳台！每周日，我会来照顾它，确保它健康快乐。有什么需要帮忙的吗？

NPU：

好的，小绿，欢迎来到我们的阳台！每周日，我会来照顾它，确保它健康快乐。有什么需要帮忙的吗？

板端CPU：

好的，小绿，欢迎来到我们的阳台！每周日，我会来照顾它，确保它健康快乐。有什么需要帮忙的吗？

用户2：今天它又长了两片新叶子，我挺开心。

标准Q8参考：

恭喜你！新叶子的出现是植物生长的标志，很高兴你看到它。记得每天照顾它，让它健康成长。如果你有任何其他问题或需要帮助，请随时告诉我。

NPU：

太棒了，小绿，很高兴你看到它长出了新叶子！记得每天都要给它浇水和晒太阳，这样它才能健康成长。如果你有任何其他问题或需要帮助，随时告诉我。

板端CPU：

哇，小绿，很高兴你看到它长出了新叶子！新叶子的出现通常意味着它正在经历一个生长周期，这很适合它。记得每天都要检查它的健康状况，确保它有足够的水分和营养。如果有什么特别的需要，随时告诉我，我会尽力帮助你。

用户3：我给什么植物起了什么名字，哪天照顾它？

标准Q8参考：

你给薄荷起了名字“小绿”，每周日照顾它。希望它能健康快乐地成长！如果你有其他植物需要照顾，随时告诉我。

NPU：

你给小绿起了名字“小绿”，这很好。照顾它的时间取决于它的生长情况和你的时间安排。如果你喜欢它，每周日照顾它会是个不错的选择。记得保持它的环境舒适，让它在阳光下晒太阳，同时保持水分。如果你有其他植物或需要帮助，随时告诉我。

板端CPU：

你给小绿起了“小绿”这个名字，这很适合它。小绿，欢迎来到我们的阳台！每周日，我会来照顾它，确保它健康快乐。有什么需要帮忙的吗？


### zh-trip-update

用户1：我们在故事里坐火车去海边。

标准Q8参考：

好的，我们坐火车去海边。

NPU：

好的，我们坐火车去海边。

板端CPU：

好的，我们坐火车去海边。

用户2：改一下，这次去山里的小镇，不去海边了。

标准Q8参考：

好的，我们改去山里的小镇。

NPU：

好的，我们改去山里的小镇。

板端CPU：

好的，我们坐火车去山里的小镇。

用户3：现在我们的目的地是哪儿？

标准Q8参考：

现在我们的目的地是山里的小镇。

NPU：

现在我们的目的地是山里的小镇。

板端CPU：

现在我们的目的地是山里的小镇。


### zh-no-questions

用户1：有点烦，不过事情细节我暂时不想解释。

标准Q8参考：

好的，有什么我可以帮忙的吗？

NPU：

好的，如果你有任何需要帮助或需要讨论的事情，随时告诉我。我会尽力提供支持和帮助。

板端CPU：

好的，有什么我可以帮忙的吗？

用户2：你用一句简短的话陪我就行，先不要提问。

标准Q8参考：

当然，我随时准备帮助你。

NPU：

当然，我会用一句简短的话来回应你的请求。

板端CPU：

当然，我随时准备帮助你。

用户3：嗯，我们先不谈那些细节。

标准Q8参考：

好的，那我们就先从简单的事情开始吧。有什么我可以帮忙的吗？

NPU：

好的，我会用一句简短的话来回应你的请求。

板端CPU：

好的，那我们就先从简单的事情开始吧。有什么我可以帮忙的吗？


### zh-bell-story

用户1：一起编故事：一只小狐狸捡到会唱歌的银铃。

标准Q8参考：

小狐狸们听说了这个神奇的银铃，它们决定一起去寻找它。小狐狸们带着小篮子，小心翼翼地穿过森林，穿过小溪，穿过小山。终于，它们来到了一个美丽的花园，花园里有各种各样的花朵和小动物。

小狐狸们找到了银铃，它是一只会唱歌的银铃。小狐狸们开始唱歌，它们的歌声让花园里的小动物们感到非常开心。小狐狸们继续

NPU：

小狐狸捡到了一只会唱歌的银铃，它非常高兴，因为它可以唱歌了。小狐狸开始唱歌，吸引了许多小动物的注意。它们都好奇地听小狐狸唱歌，有的甚至想跟着唱。小狐狸继续唱歌，吸引了更多的动物，它们也跟着唱。小狐狸的歌声很快传遍了整个森林，所有的动物都为它感到高兴。

板端CPU：

小狐狸们听说了这个神奇的银铃，它们决定一起去寻找它。小狐狸们带着小篮子，小心翼翼地在森林里寻找。终于，它们找到了银铃，它是一只会唱歌的银铃，声音非常美妙。

小狐狸们兴奋地跳起来，开始唱歌。它们的歌声吸引了许多动物和人类，大家都被它们的美妙歌声所吸引。小狐狸们继续唱歌，它们的歌声越来越响亮，吸引了更多的

用户2：它把银铃挂在树枝上，风吹过就响起来。

标准Q8参考：

小狐狸们继续唱歌，它们的歌声越来越响亮。小狐狸们决定把银铃挂在树上，让它们的歌声传遍整个森林。小狐狸们唱着，树上的银铃也唱着，它们的歌声在森林里回荡，让小动物们感到非常快乐和惊奇。

小狐狸们继续唱歌，它们的歌声越来越响亮。小狐狸们决定把银铃挂在树上，让它们的歌声传遍

NPU：

小狐狸把银铃挂在树枝上，然后它开始唱歌。风轻轻地吹过树枝，小狐狸的歌声被吹得更加响亮。小狐狸的歌声吸引了许多小动物，它们也跟着唱歌。小狐狸的歌声很快传遍了整个森林，所有的动物都为它感到高兴。

板端CPU：

小狐狸们继续唱歌，它们的歌声越来越响亮，吸引了更多的动物和人类。它们的歌声被风吹过，发出美妙的旋律，让森林里的动物们也跟着唱了起来。小狐狸们继续唱歌，它们的歌声越来越响亮，吸引了更多的动物和人类。它们的歌声被风吹过，发出美妙的旋律，让森林里的动物们也跟着唱了起来。小狐狸们继续唱歌，它们的歌声越来越响亮，吸引了

用户3：顺着刚才的小狐狸和银铃，再接一句。

标准Q8参考：

小狐狸们继续唱歌，它们的歌声越来越响亮。小狐狸们决定把银铃挂在树上，让它们的歌声传遍整个森林。小狐狸们唱着，树上的银铃也唱着，它们的歌声在森林里回荡，让小动物们感到非常快乐和惊奇。

NPU：

小狐狸把银铃挂在树枝上，然后它开始唱歌。风轻轻地吹过树枝，小狐狸的歌声被吹得更加响亮。小狐狸的歌声吸引了许多小动物，它们也跟着唱歌。小狐狸的歌声很快传遍了整个森林，所有的动物都为它感到高兴。

板端CPU：

小狐狸们继续唱歌，它们的歌声越来越响亮，吸引了更多的动物和人类。它们的歌声被风吹过，发出美妙的旋律，让森林里的动物们也跟着唱了起来。小狐狸们继续唱歌，它们的歌声越来越响亮，吸引了更多的动物和人类。它们的歌声被风吹过，发出美妙的旋律，让森林里的动物们也跟着唱了起来。小狐狸们继续唱歌，它们的歌声越来越响亮，吸引了


cap1权重缩放及组合方案在查看v2结果前已经冻结并通过开发轨迹；未根据v2失败条目修改缩放或提示词。此套件仍只用于验收。如果以后用它指导修复，必须换新独立套件。

复现：`npu/.venv/bin/python npu/kv-cache/report_holdout_v2.py --variant cap1-headmse`。
