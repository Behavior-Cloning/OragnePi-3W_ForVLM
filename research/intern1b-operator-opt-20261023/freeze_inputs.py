from pathlib import Path
import json
from tokenizers import Tokenizer
r=Path(__file__).resolve().parent
old=json.loads((r.parent/'intern1b-int4-20261021/inputs.json').read_text())
t=Tokenizer.from_file('/opt/opi-lab/source-validation-20261017/official-InternVL3-1B/tokenizer.json')
e=lambda s:t.encode(s,add_special_tokens=False).ids
sys=old['system'];prefix=old['prefix'];assert e('<|im_start|>system\n'+sys+'<|im_end|>\n')==prefix
texts=['你好，请介绍一下自己。','我有点难过，陪我聊两句。','二加三等于几？','晚上好，今天辛苦了。','请原地挥手，别向前移动。','晚安，我要睡觉了。']
cases=[]
for i,q in enumerate(texts):
 tokens=prefix+e('<|im_start|>user\n'+q+'<|im_end|>\n<|im_start|>assistant\n')
 cases.append(dict(id='short-'+str(i),query=q,tokens=tokens,dynamic_actual=len(tokens)-len(prefix)))
perf=[cases[0],cases[1],cases[4],*old['performance'][:2]]
(r/'inputs.json').write_text(json.dumps(dict(system=sys,prefix=prefix,performance=perf,quality=cases+old['quality']),ensure_ascii=False,indent=2))
print('prefix',len(prefix),'performance',[(x['id'],len(x['tokens']),x['dynamic_actual']) for x in perf])
