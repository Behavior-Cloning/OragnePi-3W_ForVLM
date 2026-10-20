import argparse,json,time,os,psutil,traceback,threading
from pathlib import Path
from board_adapters import Adapter
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--label',required=True);a=ap.parse_args();r=Path(__file__).resolve().parent
f=(r/'results'/('opt-'+a.model+'-'+a.label+'-'+time.strftime('%Y%m%dT%H%M%S')+'.jsonl')).open('x',buffering=1)
def emit(x):f.write(json.dumps(x,ensure_ascii=False)+'\n');print(x.get('event'),x.get('i'),flush=True)
def cpu():
 p=psutil.Process();return sum(sum(x.cpu_times()[:2]) for x in [p,*p.children(recursive=True)])
def snap():
 p=psutil.Process();return dict(at=time.monotonic(),rss=sum(x.memory_info().rss for x in [p,*p.children(recursive=True)]),available=psutil.virtual_memory().available,swap=psutil.swap_memory().used,temp=float(Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000)
m=None;stop=threading.Event();samples=[]
def monitor():
 while not stop.wait(.5):
  try:samples.append(snap())
  except Exception as e:samples.append({'error':repr(e)})
mt=threading.Thread(target=monitor,daemon=True);mt.start()
try:
 t=time.perf_counter();m=Adapter(a.model,r,a.threads);emit(dict(event='loaded',seconds=time.perf_counter()-t,label=a.label,affinity=sorted(os.sched_getaffinity(0)),optimization=m.optimization,environment={k:os.environ.get(k) for k in ['JEV_DYNAMIC_INT8','NANO_BATCH','JEV_N_SEQ_MAX','JEV_UBATCH']},resources=snap()))
 cases=json.loads((r/'cases.json').read_text());actions=json.loads((r/'actions.json').read_text());opts={x['id']:x['description'] for x in actions[:10]}
 for i in range(3):m.decide(m.state(cases[i],64)[0],opts)
 for i in range(30):
  c=cases[i%10];state,n=m.state(c,64);cp=cpu();t=time.perf_counter();z=m.decide(state,opts);elapsed=time.perf_counter()-t
  emit(dict(event='opt_measure',i=i,case_id=c['id'],seconds=elapsed,cpu_seconds=cpu()-cp,state_tokens=n,answer=z['answer'],expected=c['expected'],correct=z['answer'] in c['expected'],result=z,resources=snap()))
 # Evaluate same diverse fixed 20-case subset with complete direct100 candidates.
 opts={x['id']:x['description'] for x in actions}
 subset=cases[::11][:12]+cases[-8:]
 for i,c in enumerate(subset):
  state,n=m.state(c,64);t=time.perf_counter();z=m.decide(state,opts);emit(dict(event='opt_quality',i=i,case_id=c['id'],suite=c['suite'],seconds=time.perf_counter()-t,state_tokens=n,answer=z['answer'],expected=c['expected'],correct=z['answer'] in c['expected'],result=z))
 emit(dict(event='complete'))
except Exception as e:emit(dict(event='error',error=repr(e),trace=traceback.format_exc()));raise
finally:
 stop.set();mt.join(timeout=3);emit(dict(event='telemetry',samples=samples))
 if m:m.close()
 f.close()
