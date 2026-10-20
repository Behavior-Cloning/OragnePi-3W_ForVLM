import argparse,json,time,os,threading,traceback,psutil,random
from pathlib import Path
from board_adapters import Adapter
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--threads',type=int,default=2);ap.add_argument('--phase',default='screen');args=ap.parse_args()
r=Path(__file__).resolve().parent;run_id=time.strftime('%Y%m%dT%H%M%S')+'-'+str(os.getpid());out=r/'results'/f'{args.model}-{args.phase}-t{args.threads}-{run_id}.jsonl';f=out.open('x',buffering=1)
def emit(x):f.write(json.dumps(x,ensure_ascii=False)+'\n');print(x.get('event'),x.get('block'),x.get('i'),flush=True)
def sample():
 proc=psutil.Process();children=proc.children(recursive=True);rss=proc.memory_info().rss+sum(p.memory_info().rss for p in children if p.is_running())
 temps={p.parent.name:float(p.read_text().strip())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')}
 return dict(monotonic=time.monotonic(),cpu_tree_seconds=sum(sum(p.cpu_times()[:2]) for p in [proc]+children),rss_tree_bytes=rss,system_available_bytes=psutil.virtual_memory().available,system_used_bytes=psutil.virtual_memory().used,swap_used_bytes=psutil.swap_memory().used,temperature_c=temps,loadavg=os.getloadavg(),cpu_system_percent=psutil.cpu_percent())
t0=time.perf_counter()
try:m=Adapter(args.model,r,args.threads)
except Exception as e:emit(dict(event='load_error',error=repr(e),trace=traceback.format_exc()));raise
emit(dict(event='loaded',seconds=time.perf_counter()-t0,affinity=sorted(os.sched_getaffinity(0)),resource=sample()))
actions=json.loads((r/'actions.json').read_text());groups=json.loads((r/'groups.json').read_text());cases=json.loads((r/'cases.json').read_text())
allopts={a['id']:a['description'] for a in actions};groupopts={g['id']:g['description'] for g in groups}
try:
 if args.model=='nano33':
  for passage in ['University College London was founded in 1826.','The cat is sleeping on the sofa.']:
   t=time.perf_counter();v=m.decide(passage,dict(irrelevant='irrelevant',partial='partially relevant',answer='directly answers'),'How relevant is this passage to the query: When was UCL founded?');emit(dict(event='native_control',passage=passage,seconds=time.perf_counter()-t,result=v))
 stop=threading.Event();samples=[];monitor_errors=[]
 def monitor():
  while not stop.wait(.5):
   try:samples.append(sample())
   except Exception as e:monitor_errors.append(repr(e))
 monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
 def one(case,route='two-stage',budget=64,count=100):
  state,n=m.state(case,budget);st=time.perf_counter();cp=sample()['cpu_tree_seconds']
  if route=='two-stage':
   group=m.decide(state,groupopts);opts={a['id']:a['description'] for a in actions if a['group_id']==group['answer']};z=m.decide(state,opts);details=[group,z]
  else:
   pairs=list(allopts.items())[:count];random.Random(42+sum(map(ord,case['id']))).shuffle(pairs);opts=dict(pairs);z=m.decide(state,opts);details=[z]
  return dict(seconds=time.perf_counter()-st,cpu_seconds=sample()['cpu_tree_seconds']-cp,state_tokens=n,answer=z['answer'],expected=case['expected'],correct=z['answer'] in case['expected'],details=details)
 raw_one=one
 def one(*a,**kw):
  try:return raw_one(*a,**kw)
  except Exception as e:return dict(status='error_or_unsupported',error=repr(e),trace=traceback.format_exc())
 # 20 warmups are excluded from 200 steady-state measurements.
 for i in range(20):one(cases[i%10],route='direct',count=10)
 for i in range(200):emit(dict(event='measure',block='10-candidates-64tokens',i=i,**one(cases[i%10],route='direct',count=10),resource=sample()))
 if args.phase!='timing':
  for route in ['direct','two-stage']:
   for i,c in enumerate(cases):emit(dict(event='accuracy',block=route,i=i,case_id=c['id'],suite=c['suite'],**one(c,route=route),resource=sample()))
  for count in [4,8,16,32,64,100]:
   for i in range(10):emit(dict(event='candidate_matrix',block=f'{count}-candidates',i=i,**one(cases[i%count],route='direct',count=count),resource=sample()))
  for budget in [128,256]:
   for i in range(10):emit(dict(event='measure',block=f'two-stage-{budget}tokens',i=i,**one(cases[(i*11)%len(cases)],budget=budget),resource=sample()))
 stop.set();monitor_thread.join(timeout=3);emit(dict(event='telemetry',samples=samples,errors=monitor_errors));emit(dict(event='complete',resource=sample()))
except Exception as e:emit(dict(event='error',error=repr(e),trace=traceback.format_exc()));raise
finally:m.close();f.close()
