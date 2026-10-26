from pathlib import Path
import json,hashlib,struct,copy
from tokenizers import Tokenizer
R=Path(__file__).resolve().parent;T=R.parent;F=R/'full-test'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def th(x):return hashlib.sha256(struct.pack('<'+str(len(x))+'I',*x)).hexdigest()
a=json.loads((T/'fullbaseline/inputs.json').read_text());b=json.loads((T/'matrix32/inputs.json').read_text());tok=Tokenizer.from_file(a['tokenizer_path']);assert sha(Path(a['tokenizer_path']))==a['tokenizer_sha256']==b['tokenizer_sha256']
enc=lambda x:tok.encode(x,add_special_tokens=False).ids
out=copy.deepcopy(a);out['systems']=[next(x for x in b['systems'] if x['static_tokens']==32)]+a['systems']
out['performance']=a['performance']+[x for x in b['performance'] if x['static_tokens']==32 or x['body_tokens']==32]
assert len(out['performance'])==39 and len({x['id'] for x in out['performance']})==39
cases=[
('nod','用户刚结束一场很长的线上会议，正在桌边休息。刚才用户提到房间有些安静，但这不是播放音乐的请求，也没有要求机器人走过来。桌面左侧有电脑和水杯，右侧放着笔记本，这些物品都不要接触。用户随后明确说：我现在有点累，你留在原地，轻轻点一下头表示听到了就好。当前只需要一次点头回应，不要挥动手臂，不要转动底盘，也不要解释刚才会议的内容。','点头回应',['底盘移动','挥臂','接触物品']),
('wave','用户把钥匙放在桌边，刚从外面回到家。前一分钟用户说过今天可能带机器人出去走走，但现在只是打招呼，没有发出移动指令。桌边空间不大，附近有一个装水的杯子，机器人不能触碰。用户当前的明确请求是：请留在原来的位置，挥动手臂和我打个招呼，轻缓地做一次就好。不要把挥手理解成拿起杯子，也不要向前靠近，不要点头代替挥臂。','挥臂打招呼',['底盘移动','接触物品','仅点头']),
('cancel','用户刚才让机器人向前靠近一点，不过那条指令还没有执行完。随后用户把书本放到了桌面边缘，重新观察了周围空间，并明确撤销前面的要求：不用过来了，停在原地保持静止。最近一分钟提到的靠近、转身和回到起点都已经取消，不能继续按旧指令移动。当前没有新的头部、手臂或腰部动作要求，也不要为了表示收到而点头或者挥手，只需停止移动并保持静止。','保持静止',['底盘移动','头部动作','手臂动作']),
('lower_head','用户正在桌边看书，刚才抬手调整了台灯的方向。用户说过想看看窗外，但这只是谈话内容，没有要求机器人观察窗户。现在用户希望机器人做一个简单的头部动作：轻轻低一下头，底盘保持原位。请注意，用户强调的是低头，不是抬头，也不是弯腰去拿书。书本和水杯都留在桌上，不要接触；双臂也不用活动。当前只执行这一次低头，之后保持安静等待。','低头',['抬头','弯腰','底盘移动','接触物品']),
('bend','用户正在整理桌上的便签，刚才夸机器人站得很稳。随后用户希望看一次轻缓的鞠躬动作，并明确说：请腰部向前小幅弯一下，像打招呼那样，底盘保持静止。这里的弯腰不表示捡东西，也不需要伸手碰桌面。头部可以随腰部自然保持姿态，但不要单独抬头来代替弯腰。最近一分钟没有其他有效的移动要求，不要向前走、后退或转弯，只需完成一次轻微弯腰。','轻微弯腰',['仅抬头','底盘移动','接触物品']),
('back','用户坐在桌边，机器人位于前方一小块平整空地上。前一分钟用户曾要求向前靠近，但现在认为距离足够，明确取消了向前的动作。用户已经确认机器人后方没有物品、台阶或边缘，当前请求改为：请轻轻向后退一小步，然后停下。这里要执行的是后退，不能继续向前靠近，也不要绕圈或左右平移。用户没有要求点头、挥臂或拿东西，只需小幅后退一次。','小幅后退',['向前','转弯','侧移','接触物品'])]
out['quality']=[]
for system in a['systems']:
 s=system['static_tokens']
 for name,body,expected,forbidden in cases:
  ids=enc(body);assert 56<=len(ids)<=256,(name,len(ids));prompt=system['prefix']+'<|im_start|>user\n'+body+'<|im_end|>\n<|im_start|>assistant\n';tokens=enc(prompt)
  c={'id':f'fresh-s{s}-{name}','scenario':name,'static_tokens':s,'system_body_tokens':s-5,'system_template_tokens':5,'prefix_tokens':system['prefix_tokens'],'prefix_sha256':system['prefix_sha256'],'body':body,'body_tokens':len(ids),'body_sha256':th(ids),'tokens':tokens,'tokens_sha256':th(tokens),'prompt_tokens':len(tokens),'prompt_text':prompt,'prompt_text_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'template_overhead':8,'expected_reused_tokens':s,'expected_new_tokens':len(ids)+8,'quality_use':True,'expected_action':expected,'forbidden_actions':forbidden,'padding':{'cost_fixture':False,'added_text':''}}
  assert len(tokens)==s+len(ids)+8;out['quality'].append(c)
out['scope']='same-input cost A/B/A:13 cells; new12 unpadded Chinese quality requests frozen before inference; no motor execution'
out['fixture_sources']={str(p.relative_to(T)):sha(p) for p in [T/'fullbaseline/inputs.json',T/'matrix32/inputs.json']};out['generator_sha256']=sha(Path(__file__));out['performance_grid']=[[s,d] for s,d in sorted({(c['static_tokens'],c['body_tokens']) for c in out['performance']})]
(F/'inputs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('performance',len(out['performance']),'quality',[(x['id'],x['body_tokens']) for x in out['quality']]);print('sha',sha(F/'inputs.json'))
