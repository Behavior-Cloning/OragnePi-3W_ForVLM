import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'deps'))
from tokenizers import Tokenizer
from resident_worker import ResidentWorker
r=Path(__file__).parent;o=r/'natural-smoke';o.mkdir(exist_ok=False)
t=Tokenizer.from_file(str(r/'tokenizer.json'))
g=Path('/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018')
w=ResidentWorker(['taskset','-c','4,5',str(r/'resident'),str(g),'24','1280','151674',str(g/'embeddings.f32'),'16','151645'],r,o/'worker.log')
rows=[]
try:
 for question in ['你好，请用一句话介绍你自己。','我有鸡蛋和番茄，晚饭可以做什么？','今天有点累，你能鼓励我一句吗？']:
  p='<|im_start|>system\n你是家庭陪伴机器人小白。请用简洁自然的中文回答用户。<|im_end|>\n<|im_start|>user\n'+question+'<|im_end|>\n<|im_start|>assistant\n'
  z=list(w.generate(t.encode(p,add_special_tokens=False).ids,48))[-1]
  rows.append({'question':question,'answer':t.decode(z['tokens']),'metrics':z});print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
finally:
 closed=w.close();(o/'result.json').write_text(json.dumps({'runs':rows,'closed':closed},ensure_ascii=False,indent=2))
