from pathlib import Path
import torch,json,time
from board_adapters import Adapter
r=Path(__file__).resolve().parent;out=[]
for name in ['laya','nano33']:
 m=Adapter(name,r,2)
 for lang in ['zh','en']:
  opts={'left':'看向左侧','right':'看向右侧','stop':'立即停止','help':'请求帮助'} if lang=='zh' else {'left':'Look left','right':'Look right','stop':'Stop immediately','help':'Ask for help'}
  question='选择符合用户要求的安全动作。' if lang=='zh' else 'Which action best matches the user request?'
  for state in (['请看向左侧。','请看向右侧。','立即停止。'] if lang=='zh' else ['Please look to your left.','Please look to your right.','Stop immediately.']):
   t=time.perf_counter();v=m.decide(state,opts,question);z=dict(model=name,language=lang,state=state,result=v,seconds=time.perf_counter()-t,execution_host='Windows WSL desktop; not board performance');out.append(z);print(json.dumps(z,ensure_ascii=False),flush=True)
 m.close();del m
(r/'results/desktop-language-controls.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
