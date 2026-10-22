import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=Path('/mnt/f/Files/OPi-Lab/experiments/board-rag-atoms-20261021/larger256-20261022-v1')
old=json.loads((OLD/'cards-256.json').read_text())
allowed={f'A{i:02d}' for i in list(range(1,29))+list(range(30,36))+[41]}
atoms=[a.copy() for a in old if a['id'] in allowed]
# Explicit actuator footprints repair missing bilateral tags; no query labels used.
bilateral={'A13','A14','A15','A16','A21'}
for a in atoms:
 a['scope']='home_desktop_companion';a['implementation']='mock semantic primitive; hardware mapping unvalidated'
 tags=set(a['tags'])
 if a['id'] in bilateral:tags.update(['hand:left','hand:right'])
 a['tags']=sorted(tags)
 a['actuators']=sorted(tags & {'head','arm','torso','face'})
 if a['id']=='A21':a['actuators']=sorted(set(a['actuators'])|{'torso'})
 if a['id']=='A17':a['actuators']=sorted(set(a['actuators'])|{'head'})
 a['desktop_envelope']='bounded in place; no translation, jumping, grasping or object contact'
byid={a['id']:a for a in atoms}
skill_specs=[
('H01','迎接家人回家','看到家人回到身边，用温和的笑容、右手招呼和点头回应。','A25 A09 A06'),
('H02','欢迎朋友来访','向熟悉的朋友展开双臂表示欢迎，配合友善笑容。','A25 A14 A06'),
('H03','安静倾听倾诉','对方倾诉时不打断，用柔和表情和点头回应，停留等待。','A26 A06 A41'),
('H04','陪伴读书工作','在桌边保持柔和平静的表情与安静姿态，减少干扰。','A26 A41'),
('H05','安慰失落的伙伴','先以难过表情表示共情，再轻点头，用温和微笑安慰。','A27 A06 A25'),
('H06','小声庆祝好消息','用微笑、双手轻摆和点头表达高兴，避免拍击和跳跃。','A25 A13 A06'),
('H07','睡前道晚安','用哈欠和慢慢合眼表现准备休息，最后安静保持姿态。','A22 A23 A41'),
('H08','早晨醒来舒展','睁开眼睛并伸展双臂身体，回到中立姿态，露出笑容。','A24 A21 A18 A25'),
('H09','疲倦时放松','松下肩膀、打个哈欠，随后保持安静，表现想休息。','A20 A22 A41'),
('H10','陪伴做休息操','舒展双臂和身体，恢复中立姿势，用笑容鼓励休息。','A21 A18 A25'),
('H11','随着音乐轻摆','上身和双手随节拍轻摆，配合小幅点头，不移动底座。','A30 A13 A06'),
('H12','礼貌表达感谢','小幅弯腰低头再恢复姿态，用笑容表达谢意。','A17 A18 A25'),
('H13','为打扰表示歉意','轻轻弯腰低头，配合歉疚表情，然后恢复姿态。','A17 A27 A18'),
('H14','好奇观察桌面物品','注视已经定位的物品，歪头并小幅前倾，不接触物品。','A08 A05 A31'),
('H15','指示已知的物品','看向已定位的物品并用手指示，引导对方注意，不抓取。','A08 A35'),
('H16','害羞地接受夸奖','微微低头和轻摆双手，带着含蓄笑容回应表扬。','A04 A25 A13'),
('H17','向家人挥手告别','用右手挥动和微笑表示再见，轻点头结束。','A09 A25 A06'),
('H18','听到意外时惊讶','睁大眼睛、略张嘴，上身轻轻后缩，表现惊讶。','A24 A28 A32'),
('H19','表达委屈和失落','显出难过表情，垂下肩膀并低头。','A27 A20 A04'),
('H20','左手问候朋友','用左手挥动表示问候，配合微笑点头，右臂不动。','A10 A25 A06')]
skills=[]
for sid,name,desc,seq in skill_specs:
 steps=seq.split();children=[byid[x] for x in steps]
 skills.append(dict(id=sid,type='skill',name=name,description=desc,steps=steps,tags=sorted(set(t for a in children for t in a['tags'])),energy='high' if any(a['energy']=='high' for a in children) else 'low',requires_target=any(a['requires_target'] for a in children),actuators=sorted(set(t for a in children for t in a['actuators'])),scope='home_desktop_companion',implementation='mock composition of atomic cards'))
assert len(atoms)==35 and len(skills)==20
for name,data in [('atoms.json',atoms),('skills.json',skills),('cards.json',atoms+skills),('removed-from-old.json',[dict(id=a['id'],name=a['name'],type=a['type']) for a in old if a['id'] not in allowed])]:
 (ROOT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2))
print('frozen cards',len(atoms),len(skills))
