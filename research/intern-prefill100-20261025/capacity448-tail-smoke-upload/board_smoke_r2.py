import json,time,os
from pathlib import Path
from resident_worker_v1 import ResidentWorker
root=Path(__file__).resolve().parent
base=Path('/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018')
graph=root/'graph';graph.mkdir(exist_ok=False)
for folder in ['decode','prefill']:
 for p in (root/folder).glob('*.nb'): (graph/p.name).symlink_to(p)
for name in ['embeddings.f32']+['layer%d.nb'%i for i in range(24,28)]: (graph/name).symlink_to(base/name)
tokens=json.loads((root/'request.json').read_text())[0]['tokens']
assert len(tokens)==328 and tokens[64]==151644
f=(root/'events-r2.jsonl').open('x')
def emit(x): f.write(json.dumps(x)+'\n');f.flush()
for phase,tail in [('base',False),('tail8',True),('base-repeat',False)]:
 env=['OPI_GROUP_PREFILL=1','OPI_MASK_TEMPLATE=1','OPI_TTFT_TAIL_MIN=8','OPI_GREEDY_I16_SINGLE_PASS=1']
 if tail:env+=['OPI_PREFILL8_DIR='+str(root/'tail8')]
 cmd=['env',*env,'/opt/opi-vlm/intern-prefill100-20261025/capacity512-full/runs/c512-m16-m32-aba-r1/resident-cap512',str(graph),'24','448','151674',str(graph/'embeddings.f32'),'32','151645']
 w=ResidentWorker(cmd,str(root),str(root/(phase+'-r2.log')),timeout=180)
 emit({'phase':phase,'ready':w.ready})
 try:
  for repeat in range(3):
   prep=list(w.generate(tokens[:64],max_new=1,reset=True))[-1]
   ev=list(w.generate(tokens,max_new=8,feedback=[100662]*8))
   done=ev[-1]
   assert done['reused_tokens']==64,done
   assert done['new_tokens']==264,done
   emit({'phase':phase,'repeat':repeat,'system_body':59,'system_template':5,'system_total':64,'dynamic_body':256,'dynamic_template':8,'new_prefill':264,'output':8,'events':ev})
 finally:emit({'phase':phase,'closed':w.close()})
f.close()
