import json,time,os,sys
from pathlib import Path
from resident_worker import ResidentWorker
r=Path(__file__).resolve().parent;g='/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018';tokens=json.loads((r/'intern-comparison-input.json').read_text())['tokens'];assert len(tokens)==31
f=(r/'intern-comparison.jsonl').open('x',buffering=1)
def emit(x):f.write(json.dumps(x)+'\n')
w=ResidentWorker(['taskset','-c','6,7','/opt/opi-vlm/resident-agent-20261019/resident',g,'24','1280','151674',g+'/embeddings.f32','16','151645'],str(r),str(r/'intern-comparison-worker.log'))
emit({'event':'ready','details':w.ready})
try:
 for i in range(3):
  z=list(w.generate(tokens,16,reset=True))[-1];assert z['reused_tokens']==0;emit({'event':'round','round':i,'details':z,'temperatures_mC':{p.parent.name:int(p.read_text()) for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')}})
finally:emit({'event':'closed','details':w.close()})
