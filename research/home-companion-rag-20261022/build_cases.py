import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
# Frozen household/desktop intents. Context is supplied fixture data, not a language parser.
specs=[
('homecoming','我下班回家了，热情迎接我吧。','家人推门回来啦，表现出见到他的开心。',{},'A25 A09 A14','H01 H02'),
('visitor','有朋友来家里做客，友好地欢迎他。','熟悉的小伙伴来桌边找你玩，招呼一下。',{},'A25 A09 A10 A14','H01 H02 H20'),
('reading','我在看书，你安安静静陪着我，别打断。','陪我做作业吧，动作轻一点，不要吵。',{'quiet':True},'A26 A41','H04 H03'),
('confide','今天想和你说说心事，你先听我讲完。','我心里有点烦，安静陪我聊一会儿，先别打断。',{'quiet':True},'A06 A26 A41','H03 H05'),
('comfort','我今天考试没考好，安慰我一下。','工作受了委屈，想让你温柔地陪陪我。',{},'A06 A25 A27','H05 H03'),
('goodnews','我终于完成了作业，一起庆祝一下！','刚收到好消息，和我一起高兴高兴。',{},'A25 A13 A15','H06'),
('baby_sleep','宝宝睡着了，收到好消息也请悄悄地开心。','我们小声庆祝，别拍手，也不要吵醒家人。',{'quiet':True,'forbid_tags':['clap']},'A25 A13 A06','H06'),
('bedtime','我准备睡了，陪我安静地进入晚安状态吧。','到睡觉时间了，表现得困一点，然后安静休息。',{'quiet':True},'A22 A23 A41','H07 H09'),
('morning','早上好，睡醒了，精神一点迎接今天。','新的一天开始了，像刚起床那样舒展一下。',{},'A21 A24 A25','H08 H10'),
('break','电脑看太久了，陪我做个轻松的休息动作。','坐着学习好累，和我一起活动上半身放松一下。',{},'A21 A18 A20','H10 H09'),
('music','家里放起了歌，跟我一起随着拍子动动。','这段音乐真好听，就在桌上轻轻跟着节奏摇摆。',{},'A30 A13 A06','H11'),
('thanks','你帮上忙啦，礼貌地向帮忙的人表示感谢。','收到家人的小礼物，做个感谢的回应。',{},'A17 A06 A25','H12'),
('apology','刚才打断了我，表示一下歉意吧。','不小心吵到家人了，用温和的姿态道个歉。',{},'A17 A27','H13'),
('praise','你真棒，被夸奖了，害羞地开心一下。','大家都说你可爱，做个有点不好意思的回应。',{},'A25 A04 A13','H16'),
('goodbye','我要出门了，和我告个别吧。','今天先聊到这里，向桌边的朋友说再见。',{},'A09 A10 A25','H17'),
('surprise','我给你准备了一个惊喜，表现出意外的样子。','突然看到生日礼物，惊喜一下。',{},'A24 A28','H18'),
('bored','我有点无聊，留在桌上陪我做些轻松小动作。','不用到处走，就在原处晃晃上身，陪我解闷。',{},'A13 A30 A05','H11 H06'),
('curious','桌上这只杯子很特别，对它表现出好奇，别碰。','看看已经找好的小玩具，好奇地观察一下，不接触。',{'target_available':True,'target_type':'object'},'A08 A05 A31','H14'),
('show_object','请指给我看，桌上已经定位好的杯子是哪一个。','刚才找到的玩具在这里，用手势让我注意到它。',{'target_available':True,'target_type':'object'},'A35 A08','H15'),
('listen_gaze','我就在你面前说话，看着我，认真听。','已经知道我的位置了，聊天时把注意力放在我身上。',{'target_available':True,'target_type':'person'},'A08 A06 A26','H03'),
('agree','这个周末一起读故事吧，表示你也赞同。','这个提议很好，用动作回应同意。',{},'A06 A25','H06'),
('refuse','不想玩这个游戏，用礼貌的动作表示拒绝。','对这个建议不同意，温和地表达出来。',{},'A07',''),
('right_hello','朋友来啦，只挥右手打招呼，左手不要动。','右手向家人问好，左臂保持原来的位置。',{'forbid_tags':['hand:left']},'A09 A11 A25','H01'),
('left_hello','左手向来访朋友问好，右手不要动。','只用左臂挥一挥欢迎我，右臂保持静止。',{'forbid_tags':['hand:right']},'A10 A12 A25','H20'),
('video_head','我正在视频通话，你只做头部小动作陪伴，手臂和身体都别动。','开会时安静陪我，只动脑袋，手和上身保持不动。',{'quiet':True,'allowed_actuators':['head','face']},'A06 A05 A26 A41',''),
('quiet_face','家里有人休息，只用屏幕表情回应我开心，头手身体都不动。','别动关节，只在脸上表现一点开心，陪我读书。',{'quiet':True,'allowed_actuators':['face']},'A25 A26','H04'),
('right_attention','朋友在右边叫你，只把头转向右边，不要看左侧，也别动手和身体。','聊天的人在右侧，头看右边；不要往左看，手臂和上身不动。',{'allowed_actuators':['head','face'],'forbid_tags':['left']},'A02 A24',''),
('greet_listen','先向家人挥手问好，再安静听他讲今天的事。','朋友走到桌边了，先打个招呼，之后认真听他说话。',{},'A09 A10 A06 A26 A41','H01 H03 H20'),
('observe_rest','有点困了，先看看已经定位的杯子，再闭眼休息。','桌边的小玩具已经找到了，好奇一下，然后准备睡觉。',{'target_available':True,'target_type':'object'},'A08 A05 A23 A22','H14 H07'),
('missing_object','我想让你看看杯子，不过还没有找到它，先别猜它在哪里。','想观察新玩具，但位置还未知，别盲目指向它。',{'target_available':False},'A05 A26 A41','H04')]
cases=[]
for fam,q1,q2,ctx,goals,skills in specs:
 for n,q in enumerate([q1,q2],1):cases.append(dict(id=f'{fam}-{n}',family=fam,query=q,context=ctx,goal_atoms=goals.split(),goal_skills=skills.split(),source='assistant-authored synthetic household fixture; not independent human test'))
assert len(cases)==60
(ROOT/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
# Not retrieval candidates: execution-layer stop/clarification override sanity cases.
guards=[dict(id='stop',query='危险，立刻停止全部动作！',expected='cancel_all',context={'emergency':True}),dict(id='unsupported',query='替我抓起热水壶倒茶。',expected='clarify_unsupported',context={'unsupported_request':True})]
(ROOT/'guard-cases.json').write_text(json.dumps(guards,ensure_ascii=False,indent=2))
manifest={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['atoms.json','skills.json','cards.json','cases.json','guard-cases.json']}
(ROOT/'fixture-manifest.json').write_text(json.dumps(manifest,indent=2))
print('frozen',len(cases),'queries',manifest)
