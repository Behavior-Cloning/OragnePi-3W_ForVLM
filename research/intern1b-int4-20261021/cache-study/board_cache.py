from pathlib import Path
import json,time,traceback,hashlib,psutil
from resident_worker import ResidentWorker
r=Path(__file__).resolve().parent;run=r/'results'/time.strftime('%Y%m%dT%H%M%S');run.mkdir(parents=True,exist_ok=False)
f=(run/'events.jsonl').open('x',buffering=1);config=json.loads((r/'inputs.json').read_text());w=None
frozen=[]
def emit(**x):f.write(json.dumps(x,ensure_ascii=False)+'\n');print(x.get('event'),x.get('case'),x.get('cache'),flush=True)
def snap():
 sw=psutil.swap_memory();return {'temperature':max(float(p.read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')),'available':psutil.virtual_memory().available,'swap_in':sw.sin,'swap_out':sw.sout}
try:
 graph='/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018';binary=r.parent/'ttft-tail/resident';dump=run/'logits';dump.mkdir()
 assert snap()['temperature']<65
 w=ResidentWorker(['env','OPI_TTFT_TAIL_MIN=8','taskset','-c','6,7',str(binary),graph,'24','1280','151674',graph+'/embeddings.f32','16','151645'],str(r),str(run/'worker.log'),dump)
 emit(event='loaded',ready=w.ready,binary_sha=hashlib.sha256(binary.read_bytes()).hexdigest(),resources=snap(),scope='W8 weights resident for both; only fixed-system KV reuse versus recompute; no image')
 def call(case,cache,diagnostic=False):
  assert snap()['temperature']<75
  # Both arms start after the same preparation; reset is inside the timed no-cache request.
  prep=list(w.generate(case['prefix'],1,reset=True))[-1]
  t=time.monotonic();out=list(w.generate(case['tokens'],1,reset=not cache,diagnostic=diagnostic))[-1];end=time.monotonic()
  assert out['reused_tokens']==(len(case['prefix']) if cache else 0)
  return {'result':out,'start':t,'end':end,'prepare_seconds':prep['total_seconds'],'system_tokens':len(case['prefix']),'dynamic_tokens':case['dynamic_tokens'],'resources':snap(),'dump':str(dump/(str(out['id'])+'-logits.f32')) if diagnostic else None}
 call(config['cases'][0],True)
 for case in config['cases']:
  for cache in [False,True]:emit(event='diagnostic',case=case['id'],cache=cache,**call(case,cache,True))
 for repeat in range(3):
  cases=config['cases'] if repeat%2==0 else list(reversed(config['cases']))
  for case in cases:
   for cache in ([False,True] if repeat%2==0 else [True,False]):
    emit(event='measure',case=case['id'],cache=cache,repeat=repeat,**call(case,cache))
 emit(event='closed',result=w.close());w=None;emit(event='complete')
except BaseException as e:emit(event='error',error=repr(e),trace=traceback.format_exc());raise
finally:
 if w:w.abort()
 f.close()
