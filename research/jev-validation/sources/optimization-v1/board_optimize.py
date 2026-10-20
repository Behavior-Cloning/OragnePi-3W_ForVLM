import argparse,json,time,os,psutil,traceback,threading,hashlib,platform,importlib.metadata
from pathlib import Path
from board_adapters import Adapter
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--label',required=True);a=ap.parse_args();r=Path(__file__).resolve().parent
f=(r/'results'/('opt-'+a.model+'-'+a.label+'-'+time.strftime('%Y%m%dT%H%M%S')+'.jsonl')).open('x',buffering=1)
def emit(x):f.write(json.dumps(x,ensure_ascii=False)+'\n');print(x.get('event'),x.get('i'),flush=True)
def cpu():
 p=psutil.Process();return sum(sum(x.cpu_times()[:2]) for x in [p,*p.children(recursive=True)])
def snap():
 p=psutil.Process();procs=[p,*p.children(recursive=True)];mem=[x.memory_full_info() for x in procs];sw=psutil.swap_memory()
 return dict(at=time.monotonic(),rss=sum(x.rss for x in mem),pss=sum(getattr(x,"pss",0) for x in mem),available=psutil.virtual_memory().available,swap=sw.used,swap_in=sw.sin,swap_out=sw.sout,temp=max(float(x.read_text())/1000 for x in Path('/sys/class/thermal').glob('thermal_zone*/temp')),frequency_khz={x.parents[1].name:int(x.read_text()) for x in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_cur_freq')},loadavg=os.getloadavg(),threads=sum(x.num_threads() for x in procs))
m=None;stop=threading.Event();samples=[]
def monitor():
 while not stop.wait(.5):
  try:samples.append(snap())
  except Exception as e:samples.append({'error':repr(e)})
mt=threading.Thread(target=monitor,daemon=True);mt.start()
try:
 emit(dict(event="provenance",machine=platform.platform(),sources={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ["board_optimize.py","board_adapters.py","jev_style_decision_board.py","actions.json","cases.json"]},packages={n:importlib.metadata.version(n) for n in ["torch","transformers","laya","onnxruntime"]}))
 t=time.perf_counter();m=Adapter(a.model,r,a.threads);emit(dict(event='loaded',seconds=time.perf_counter()-t,label=a.label,affinity=sorted(os.sched_getaffinity(0)),optimization=m.optimization,environment={k:os.environ.get(k) for k in ['JEV_DYNAMIC_INT8','NANO_BATCH','JEV_N_SEQ_MAX','JEV_UBATCH','NANO_ORT','LAYA_ORT']},resources=snap()))
 cases=json.loads((r/'cases.json').read_text());actions=json.loads((r/'actions.json').read_text());opts={x['id']:x['description'] for x in actions[:10]}
 for i in range(3):m.decide(m.state(cases[i],64)[0],opts)
 for i in range(20):
  c=cases[i%10]
  if snap()['temp']>=78:raise RuntimeError('thermal guard reached 78C')
  state,n=m.state(c,64);cp=cpu();t=time.perf_counter();z=m.decide(state,opts);elapsed=time.perf_counter()-t
  emit(dict(event='opt_measure',i=i,case_id=c['id'],seconds=elapsed,cpu_seconds=cpu()-cp,state_tokens=n,answer=z['answer'],expected=c['expected'],correct=z['answer'] in c['expected'],result=z,resources=snap()))
 # Evaluate the same balanced 8-case subset with complete direct100 candidates.
 opts={x['id']:x['description'] for x in actions}
 subset=[c for suite in ['literal','paraphrase','unknown','safety'] for c in [x for x in cases if x['suite']==suite][:2]]
 for i,c in enumerate(subset):
  if snap()['temp']>=78:raise RuntimeError('thermal guard reached 78C')
  state,n=m.state(c,64);t=time.perf_counter();z=m.decide(state,opts);emit(dict(event='opt_quality',i=i,case_id=c['id'],suite=c['suite'],seconds=time.perf_counter()-t,state_tokens=n,answer=z['answer'],expected=c['expected'],correct=z['answer'] in c['expected'],result=z))
 if a.model=='nano33':
  native=[("relevance-positive","University College London was founded in 1826.","How relevant is this passage to the query: When was UCL founded?",{"irrelevant":"irrelevant","partial":"partially relevant","answer":"directly answers"},"answer"),
  ("relevance-negative","The cat is sleeping on the sofa.","How relevant is this passage to the query: When was UCL founded?",{"irrelevant":"irrelevant","partial":"partially relevant","answer":"directly answers"},"irrelevant"),
  ("sufficient-positive","[1] UCL: University College London was founded in 1826.","Does the context contain enough information to answer: When was UCL founded?",{"yes":"yes","no":"no"},"yes"),
  ("sufficient-negative","[1] Cats: The cat is sleeping on the sofa.","Does the context contain enough information to answer: When was UCL founded?",{"yes":"yes","no":"no"},"no"),
  ("grounded-positive","University College London was founded in 1826.","Is this claim supported by the context? Claim: UCL was founded in 1826.",{"yes":"yes","no":"no"},"yes"),
  ("grounded-negative","University College London was founded in 1826.","Is this claim supported by the context? Claim: UCL was founded in 1950.",{"yes":"yes","no":"no"},"no")]
  for j in range(3):
   for name,state,question,opts,want in native:
    if snap()['temp']>=78:raise RuntimeError('thermal guard reached 78C')
    t=time.perf_counter();z=m.decide(state,opts,question);emit(dict(event='native_measure',case_id=name,i=j,seconds=time.perf_counter()-t,answer=z['answer'],expected=want,correct=z['answer']==want,result=z))
 emit(dict(event='complete'))
except Exception as e:emit(dict(event='error',error=repr(e),trace=traceback.format_exc()));raise
finally:
 stop.set();mt.join(timeout=3);emit(dict(event='telemetry',samples=samples))
 if m:m.close()
 f.close()
