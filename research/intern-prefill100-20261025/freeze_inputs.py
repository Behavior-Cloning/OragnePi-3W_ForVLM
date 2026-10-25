#!/usr/bin/env python3
"""Freeze longer same-model cost fixtures; Windows only, no board writes."""
from pathlib import Path
import hashlib,json,struct
from tokenizers import Tokenizer
R=Path(__file__).resolve().parent
OLD=R.parent/'intern-nextbench-20261025/aligned-inputs.json'
base=json.loads(OLD.read_text());path=Path(base['tokenizer_path'])
assert hashlib.sha256(path.read_bytes()).hexdigest()==base['tokenizer_sha256']
tok=Tokenizer.from_file(str(path));enc=lambda x:tok.encode(x,add_special_tokens=False).ids
sha=lambda ids:hashlib.sha256(struct.pack('<'+str(len(ids))+'I',*ids)).hexdigest()
extra=[
 '最近一分钟用户都在房间里，谈到了今天的天气和回家后的安排。',
 '桌面上有书本和水杯，暂时不需要接触任何物体，也不需要移动位置。',
 '此前的移动要求已经取消，后续没有恢复这条要求，机器人只需回应当前指令。',
 '用户讲话声音平缓，没有要求播放声音或操作家用设备。',
 '机器人站在当前位置，头部和双臂可以活动，底盘继续保持静止。',
 '这是日常陪伴交流，应当优先遵守明确的取消要求与桌面空间限制。',
 '用户没有发出其他有效指令，当前需要的动作仍然与最开始的请求一致。',
 '动作应轻缓，只完成一次，不连续重复，不拿取书本、水杯或其他物品。']
def fill(seed,budget):
 original=seed;added=[]
 for sentence in extra*4:
  if len(enc(seed+sentence))<=budget:seed+=sentence;added.append(sentence)
  if len(enc(seed))>=budget-8:break
 n=0
 while len(enc(seed))<budget:
  seed+='。';n+=1;assert n<1000
 assert len(enc(seed))==budget
 return seed,{'original_text':original,'added_context':added,'punctuation_characters':n,'cost_fixture':True,'not_independent_accuracy_sample':True}
core=[64,128,192,256];bounds=[119,120,121,183,184,185,247,248,249,255]
performance=[];boundary=[]
for system in base['systems']:
 for d in core+bounds:
  for i in range(3 if d in core else 1):
   prev=next(x for x in base['performance'] if x['static_tokens']==system['static_tokens'] and x['body_tokens']==64 and x['id'].endswith('-'+str(i)))
   if d==64:case=dict(prev);case['timing']='32 fixed-feedback output steps; TTFT ends at first token'
   else:
    seed=prev['padding']['original_text'];body,padding=fill(seed,d)
    full=system['prefix']+'<|im_start|>user\n'+body+'<|im_end|>\n<|im_start|>assistant\n';ids=enc(full)
    assert len(ids)==system['static_tokens']+d+8 and ids[:system['static_tokens']]==system['prefix_tokens']
    case={**prev,'id':f's{system["static_tokens"]}-d{d}-{i}','body':body,'body_tokens':d,'body_sha256':sha(enc(body)),'padding':padding,'tokens':ids,'prompt_text':full,'tokens_sha256':sha(ids),'prompt_text_sha256':hashlib.sha256(full.encode()).hexdigest(),'prompt_tokens':len(ids),'expected_new_tokens':d+8,'timing':'32 fixed-feedback output steps; TTFT ends at first token'}
   (performance if d in core else boundary).append(case)
for c in performance+boundary:assert len(c['tokens'])+31<=512
result={**base,'schema_version':2,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'parent_fixture_sha256':hashlib.sha256(OLD.read_bytes()).hexdigest(),'definitions':{**base['definitions'],'body_tokens':'Core64/128/192/256; explicit boundaries separate','new_tokens':'dynamic body plus8 role template','cache_policy':'Only frozen system64/128; dynamic KV reuse forbidden'},'performance':performance,'boundary':boundary}
(R/'inputs.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
probe=next(c for c in performance if c['static_tokens']==128 and c['body_tokens']==256 and c['id'].endswith('-0'))
p=R/'frozen-probe-prompt';p.mkdir(exist_ok=True)
(p/'prompt.json').write_text(json.dumps({'token_ids':probe['tokens'],'source_case':probe['id'],'tokens_sha256':probe['tokens_sha256'],'scope':'single-block common token stream; take history128 plus requested queryM; not full-model latency'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'core_cases':len(performance),'boundary_cases':len(boundary),'max_prompt':max(len(c['tokens']) for c in performance),'tokenizer_sha256':base['tokenizer_sha256'],'sha256':hashlib.sha256((R/'inputs.json').read_bytes()).hexdigest()}))
