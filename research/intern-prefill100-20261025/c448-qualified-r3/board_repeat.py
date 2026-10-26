from pathlib import Path
import json,time,hashlib,traceback
from resident_worker_v1 import ResidentWorker
R=Path(__file__).resolve().parent;src=Path('/tmp/candidate-c448-tail8-smoke-r2')
base=Path('/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018')
fixture=json.loads((R/'inputs.json').read_text());c=next(x for x in fixture['performance'] if x['id']=='s64-d256-0')
f=(R/'events.jsonl').open('x');state={'status':'running','started':time.time()}
def emit(x):f.write(json.dumps(x)+'\n');f.flush()
def write(): (R/'status.json').write_text(json.dumps(state,indent=2))
write()
try:
 for phase,original,tail in [('original',True,False),('original-tail8',True,True),('original-repeat',True,False),('range16',False,False)]:
  graph=R/phase;graph.mkdir();taildir=graph/'tail8';taildir.mkdir()
  for p in (src/'decode').glob('*.nb'):(graph/p.name).symlink_to(p)
  for p in (src/'prefill').glob('*.nb'):(graph/p.name).symlink_to(R/'original32.nb' if original and p.name=='prefill-layer2.nb' else p)
  for p in (src/'tail8').glob('*.nb'):(taildir/p.name).symlink_to(R/'original8.nb' if original and p.name=='prefill-layer2.nb' else p)
  for name in ['embeddings.f32']+['layer%d.nb'%i for i in range(24,28)]:(graph/name).symlink_to(base/name)
  dumps=R/(phase+'-dumps');dumps.mkdir()
  env=['OPI_GROUP_PREFILL=1','OPI_MASK_TEMPLATE=1','OPI_TTFT_TAIL_MIN=8','OPI_GREEDY_I16_SINGLE_PASS=1']
  if tail:env+=['OPI_PREFILL8_DIR='+str(taildir)]
  cmd=['env',*env,'taskset','-c','4,5',str(R/'resident-c448'),str(graph),'24','448','151674',str(graph/'embeddings.f32'),'32','151645']
  state.update(phase=phase);write()
  w=ResidentWorker(cmd,str(R),str(R/(phase+'.log')),dump_dir=str(dumps),timeout=180)
  emit({'phase':phase,'ready':w.ready});assert w.ready['true_tail8_enabled']==tail
  try:
   for repeat in range(3):
    pre=list(w.generate(c['prefix_tokens'],max_new=1,reset=True))[-1]
    ev=list(w.generate(c['tokens'],max_new=4,feedback=[100662]*4,diagnostic=True))
    m=ev[-1];assert m['new_tokens']==264 and m['reused_tokens']==64
    p=dumps/(str(m['id'])+'-logits.f32');assert p.stat().st_size==2426784
    emit({'phase':phase,'case':c['id'],'repeat':repeat,'diagnostic':True,'system_body':59,'system_template':5,'system_total':64,'dynamic_body':256,'dynamic_template':8,'output':4,'prefix':pre,'events':ev,'logits':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
   # Natural command retention, separate from diagnostics and timing.
   for q in fixture['quality'][:6]:
    list(w.generate(q['prefix_tokens'],max_new=1,reset=True))
    ev=list(w.generate(q['tokens'],max_new=32))
    emit({'phase':phase,'quality':True,'case':q['id'],'system_total':q['static_tokens'],'dynamic_body':q['body_tokens'],'dynamic_template':8,'events':ev})
  finally:emit({'phase':phase,'closed':w.close()})
 state.update(status='complete',ended=time.time());write()
except BaseException:
 state.update(status='failed',error=traceback.format_exc(),ended=time.time());write();raise
finally:f.close()
