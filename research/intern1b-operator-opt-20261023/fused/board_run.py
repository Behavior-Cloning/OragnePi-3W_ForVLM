from pathlib import Path
import json,time,threading,psutil,os,hashlib,traceback
from resident_worker import ResidentWorker
r=Path(__file__).resolve().parent;run=r/'results'/time.strftime('%Y%m%dT%H%M%S');run.mkdir(parents=True,exist_ok=False)
f=(run/'events.jsonl').open('x',buffering=1);inputs=json.loads((r/'inputs.json').read_text());feedback={};prefix=inputs['prefix'];inputs['performance']=[inputs['performance'][i] for i in [0,2,3]];stop=threading.Event();samples=[]
def emit(**x):f.write(json.dumps(x,ensure_ascii=False)+'\n');print(x.get('event'),x.get('variant'),x.get('case'),flush=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as s:
  for b in iter(lambda:s.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def snap():
 procs=[psutil.Process(),*psutil.Process().children(recursive=True)];rows=[]
 for p in procs:
  try:m=p.memory_full_info();rows.append({'pid':p.pid,'rss':m.rss,'pss':m.pss,'cpu_seconds':sum(p.cpu_times()[:2])})
  except psutil.Error:pass
 sw=psutil.swap_memory()
 return {'at':time.monotonic(),'processes':rows,'available':psutil.virtual_memory().available,'swap_used':sw.used,'swap_in':sw.sin,'swap_out':sw.sout,'temperature':{x.parent.name:float(x.read_text())/1000 for x in Path('/sys/class/thermal').glob('thermal_zone*/temp')},'cpu_percent':psutil.cpu_percent(),'loadavg':os.getloadavg(),'npu_frequency':Path('/sys/class/devfreq/3600000.npu/cur_freq').read_text().strip()}
def monitor():
 while not stop.wait(.5):
  try:samples.append(snap())
  except Exception as e:samples.append({'error':repr(e)})
mt=threading.Thread(target=monitor,daemon=True);mt.start();worker=None
try:
 directory=Path('/opt/opi-vlm/npu-kv-probe/wsl-fresh-peak155-20261018')
 emit(event='provenance',graphs=str(directory),graph_sha256={p.name:sha(p) for p in sorted(directory.glob('*.nb'))},source_sha256={p.name:sha(p) for p in r.iterdir() if p.suffix in ['.c','.h','.py','.json']})
 configs=[('baseline',['OPI_GROUP_PREFILL=1','OPI_EXACT_PREFIX=1']),('fused',['OPI_GROUP_PREFILL=1','OPI_EXACT_PREFIX=1','OPI_FUSED_DECODE=1']),('baseline-repeat',['OPI_GROUP_PREFILL=1','OPI_EXACT_PREFIX=1'])]
 for session,(variant,env) in enumerate(configs):
  for wait in range(60):
   if max(snap()['temperature'].values())<60:break
   time.sleep(5)
  else:raise RuntimeError('cooldown below60 timed out')
  dumps=run/(str(session)+'-'+variant+'-diagnostic');dumps.mkdir()
  cmd=['env','OPI_TTFT_TAIL_MIN=8',*env,'taskset','-c','6,7',str(r/'resident'),str(directory),'24','1280','151674',str(directory/'embeddings.f32'),'16','151645']
  worker=ResidentWorker(cmd,str(r),str(run/(str(session)+'-'+variant+'-worker.log')),dump_dir=dumps)
  emit(event='loaded',session=session,variant=variant,ready=worker.ready,resources=snap())
  def prepare():return list(worker.generate(prefix,1,reset=True))[-1]
  def call(case,max_new=32,fb=None,diagnostic=False):
   if max(snap()['temperature'].values())>=78:raise RuntimeError('78C thermal gate')
   prepare();started=time.monotonic();rows=list(worker.generate(case['tokens'],max_new,feedback=fb,diagnostic=diagnostic));result=rows[-1]
   expected=len(prefix) if 'OPI_EXACT_PREFIX=1' in env else len(prefix)//16*16
   if result['reused_tokens']!=expected:raise RuntimeError('unexpected system-prefix reuse')
   return dict(result=result,events=rows[:-1],start=started,end=time.monotonic())
  emit(event='warmup',session=session,variant=variant,**call(inputs['quality'][0],16))
  if session!=2:
   for case in inputs['quality']:emit(event='quality',session=session,variant=variant,case=case['id'],**call(case,64))
   if session==0:
    for case in inputs['performance']:
     z=call(case);tokens=z['result']['tokens'];feedback[case['id']]=tokens+[151645]*(32-len(tokens));emit(event='reference_feedback',case=case['id'],feedback=feedback[case['id']])
    (run/'feedback.json').write_text(json.dumps(feedback))
   for case in [inputs['performance'][0],inputs['performance'][-1]]:
    z=call(case,16,feedback[case['id']][:16],True)
    emit(event='diagnostic',variant=variant,case=case['id'],dump=str(dumps/(str(z['result']['id'])+'-logits.f32')),**z)
  for repeat in range(2):
   cases=inputs['performance'] if repeat%2==0 else list(reversed(inputs['performance']))
   for case in cases:emit(event='measure',session=session,variant=variant,repeat=repeat,case=case['id'],dynamic_tokens=case['dynamic_actual'],**call(case,32,feedback[case['id']]))
  emit(event='closed',variant=variant,result=worker.close());worker=None
 emit(event='complete')
except Exception as e:emit(event='error',error=repr(e),trace=traceback.format_exc());raise
finally:
 if worker:worker.abort()
 stop.set();mt.join(timeout=3);emit(event='telemetry',samples=samples);f.close()
