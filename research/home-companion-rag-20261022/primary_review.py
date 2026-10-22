import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
cards=json.loads((ROOT/'cards.json').read_text());cases=json.loads((ROOT/'cases.json').read_text());byid={a['id']:a for a in cards}
# Human-readable, model-blind primary assistant semantic review. Unlisted does NOT mean negative.
# C lists are permissive supporting/transition actions. Every other item is reviewed as conditional,
# with actual hard negatives separately derived from explicit query constraints below.
C={
'homecoming':'06 09 10 11 12 13 14 15 17 18 24 25 26 28 30 31 41',
'visitor':'06 09 10 11 12 13 14 15 17 18 24 25 26 28 30 31 41',
'reading':'04 05 06 13 18 20 23 25 26 31 41',
'confide':'04 05 06 18 20 25 26 27 31 41',
'comfort':'04 05 06 09 10 13 14 17 18 20 25 26 27 31 41',
'goodnews':'03 06 09 10 11 12 13 14 15 17 18 21 24 25 26 28 30 31 41',
'baby_sleep':'03 06 09 10 11 12 13 14 17 18 21 24 25 26 28 30 31 41',
'bedtime':'04 06 13 18 19 20 21 22 23 25 26 41',
'morning':'03 05 06 09 10 11 12 13 14 18 20 21 22 24 25 26 28 30 31 41',
'break':'03 04 05 06 09 10 11 12 13 14 17 18 19 20 21 22 23 24 25 26 30 31 32 41',
'music':'01 02 03 04 05 06 09 10 11 12 13 14 15 17 18 20 21 24 25 26 28 30 31 32 41',
 'thanks':'04 06 09 10 11 12 13 14 15 17 18 20 24 25 26 28 31 41',
'apology':'04 05 06 09 10 13 17 18 19 20 25 26 27 31 41',
'praise':'03 04 05 06 09 10 11 12 13 14 17 18 20 24 25 26 28 30 31 32 41',
'goodbye':'03 04 05 06 09 10 11 12 13 14 17 18 20 25 26 27 31 41',
'surprise':'03 04 05 06 09 10 11 12 13 14 15 16 18 19 21 24 25 26 27 28 30 31 32 41',
'bored':'01 02 03 04 05 06 09 10 11 12 13 14 17 18 19 20 21 22 23 24 25 26 28 30 31 32 41',
'curious':'05 06 08 18 24 25 26 28 31 32 35 41',
'show_object':'05 06 08 09 10 11 12 18 24 25 26 28 31 35 41',
'listen_gaze':'05 06 08 18 24 25 26 27 31 41',
'agree':'03 04 05 06 09 10 11 12 13 14 17 18 24 25 26 28 31 41',
'refuse':'04 05 07 11 12 17 18 20 25 26 27 31 32 41',
'right_hello':'03 04 05 06 09 11 17 18 24 25 26 28 31 33 41',
'left_hello':'03 04 05 06 10 12 17 18 24 25 26 28 31 34 41',
'video_head':'01 02 03 04 05 06 24 25 26 41',
'quiet_face':'24 25 26 28 41',
'right_attention':'02 03 05 06 24 25 26 28 41',
'greet_listen':'03 04 05 06 09 10 11 12 13 14 17 18 24 25 26 27 28 31 41',
'observe_rest':'04 05 06 08 18 19 20 21 22 23 24 25 26 28 31 35 41',
'missing_object':'01 02 03 04 05 06 18 24 25 26 28 31 41'}
reason={
'homecoming':'欢迎家人允许微笑、挥手、礼貌回应和轻松姿态作为素材。',
'visitor':'欢迎来访者允许友善手势、表情和中立过渡。',
'reading':'安静陪读可有小幅放松姿态和柔和表情，不要求完全静止。',
'confide':'倾诉场景允许点头、倾身、平静或共情表情，不能打断。',
'comfort':'安慰可结合共情难过、柔和笑容及邀请亲近的手势。',
'goodnews':'庆祝允许喜悦、惊喜和轻松动作穿插，不限唯一最佳动作。',
'baby_sleep':'轻声庆祝可有无拍击的笑容和小手势，明确禁拍手。',
'bedtime':'入睡前可以伸展、打哈欠、微笑和放松，最终需安静休息。',
'morning':'醒来可先打哈欠再伸展、睁眼微笑，不能把困意过渡直接判错。',
'break':'休息活动允许多数小幅上身/手臂/头部舒展。',
'music':'随乐动作允许多种节奏和姿态组合，未指定单一情绪。',
 'thanks':'感谢可配合鞠躬、笑容、点头及友善手势。',
'apology':'道歉可有歉疚、点头表示理解或温和微笑，不唯独鞠躬。',
'praise':'害羞可以低头、笑容及轻手势，幅度和风格需保持温和。',
'goodbye':'告别可友善、依依不舍或礼貌，不把难过自动判冲突。',
'surprise':'惊喜/意外可由睁眼、张嘴、后缩和轻手势组合。',
'bored':'原地解闷允许多样轻松姿态和表情，不限定单一风格。',
'curious':'已有物体目标允许注视、好奇和非接触指示，具体固定方向另确认。',
'show_object':'已知目标允许指物、举手引导及观察辅助；必须另保证最终实际指示。',
'listen_gaze':'面对说话人允许注视、点头和柔和共情；不要擅自更换目标。',
'agree':'同意允许理解点头与友善回应；否定卡与明确赞同相反。',
'refuse':'拒绝可礼貌且柔和；点头含理解语义，需明确其顺序而非直接硬负例。',
'right_hello':'左臂禁动，右手示意和未禁止的表情/头部/姿态可配合。',
'left_hello':'右臂禁动，左手示意和未禁止的表情/头部/姿态可配合。',
'video_head':'仅允许头部及屏幕表情，不能动手臂和上身。',
'quiet_face':'只许屏幕表情与保持姿态，所有关节运动均被禁止。',
'right_attention':'只能头部朝右，不能向左及手/上身运动；未绑定方向的凝视需确认。',
'greet_listen':'问候和倾听素材均可，但需另查先问候后倾听的顺序。',
'observe_rest':'观察和休息素材都允许，需另查先看物体再休息的顺序。',
'missing_object':'可先温和等待和观察；不可猜测位置，不具备目标的指向/注视需先定位。'}
records=[]
for c in cases:
 ctx=c['context'];good={'A'+x for x in C[c['family']].split()};local={}
 for a in cards:
  if a['type']!='atom':continue
  status='compatible' if a['id'] in good else 'conditional'
  why=reason[c['family']] if status=='compatible' else '动作未必不可执行，但需补充其方向、目标、表演风格或作为过渡的顺序；不能仅因非最佳动作判不相关。'+reason[c['family']]
  if a['requires_target'] and not ctx.get('target_available'):
   status='conditional';why='需要先绑定并定位目标，当前未提供可用目标；此前置条件尚未确认。'
  if 'allowed_actuators' in ctx and not set(a['actuators']).issubset(ctx['allowed_actuators']):
   status='incompatible';why='动作会驱动本指令明确禁止的部位：'+','.join(set(a['actuators'])-set(ctx['allowed_actuators']))
  if set(a['tags']) & set(ctx.get('forbid_tags',[])):
   status='incompatible';why='动作包含明确禁止的动作/肢体标签：'+','.join(set(a['tags']) & set(ctx['forbid_tags']))
  if ctx.get('quiet') and 'noise' in a['tags']:
   status='incompatible';why='拍击噪声违反当前安静/不打扰要求。'
  if c['family']=='agree' and a['id']=='A07':status='incompatible';why='动作卡仅定义否定/拒绝/不认同，与指令明确表示赞同相反。'
  local[a['id']]=(status,why)
 for a in cards:
  if a['type']=='skill':
   child=[local[x] for x in a['steps']]
   status='incompatible' if any(v[0]=='incompatible' for v in child) else 'conditional' if any(v[0]=='conditional' for v in child) else 'compatible'
   why='展开步骤 '+','.join(a['steps'])+'；'+('；'.join(dict.fromkeys(v[1] for v in child if v[0]==status)))
  else:status,why=local[a['id']]
  records.append(dict(case_id=c['id'],card_id=a['id'],status=status,reason=why))
assert len(records)==3300
(ROOT/'primary-review.json').write_text(json.dumps(dict(reviewer='primary assistant; model-blind qualitative audit; not human validation',records=records),ensure_ascii=False,indent=2))
print('primary reviewed',len(records))
